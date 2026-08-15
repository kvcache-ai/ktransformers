# PR: AMXINT4_SMART — Mixed-Precision MoE Routing from GGUF

## Summary

Adds **AMXINT4_SMART**, a new AMX backend method that routes each MoE layer's
storage precision by its original GGUF quantization, plus the full GGUF-native
substrate underneath it. Mixed-precision GGUFs (e.g. UD-Q4_K_XL: Q4_K gate/up,
Q5_K/Q6_K down) no longer pay a second 4-bit requant on their 5-8-bit tensors:
the layer is stored at the tightest class that preserves its tensors and is
served by the existing, verified kernels.

## Motivation

The AMX backends historically ran a single precision per layer:

- AMXINT4 (per-row) re-quantizes everything to one scale per row — on
  already-4-bit values that discards the GGUF's per-block scale structure
  (~0.15 relative RMS per-row requant error, ~0.217 whole-layer class).
- AMXINT8 is safe for 5-8-bit tensors but doubles RAM for the 4-bit ones.
- BF16 is lossless but full-size.

A whole-layer choice is always wrong for part of the layer. SMART routes per
attribute instead.

## Changes

### Routing (`python/utils/amx.py`, `python/experts.py`)

- 3-dtype storage rule, deterministic from the GGUF types, no calibration:
  - ≤ Q4 (`Q2_K`/`Q3_K`/`Q4_K`/`Q4_0`/`Q4_1`) → **AMXINT4** (per-row)
  - (Q4, Q8] (`Q5_0`/`Q5_K`/`Q6_K`/`Q8_0`/`IQ*`) → **AMXINT8**
  - `F32`/`F16`/`BF16` → **BF16**
- `upstream = max(node(gate), node(up))`, `downstream = node(down)`.
- Mixed stage pairs `(0→1)`, `(1→2)`, `(0→2)` are **computational wrappers**:
  the layer keeps its per-attribute precisions in RAM (gate/up at the upstream
  node, down at the downstream node — the smallest possible footprint) and
  the mixed GEMMs run through the fused two-stage kernels at compute time.
  Each fused wrapper serves **both orientations** of its pair — the wrapper
  decides at entry, one step before the GEMM loops, which stage is the wider
  one (from the per-attribute nodes) and picks the kernel composition
  accordingly (e.g. the same `F4x8` wrapper runs `(0,1)` with gate/up at
  INT4 and `(1,0)` with gate/up at INT8). Only the activations are widened
  in the fused decode, never the stored weights; a tensor is only ever
  expanded when its adjacent stage lives in a larger format.
- Per-layer source/storage logging and the (prev, cur) stage-edge log between
  consecutive layers.
- Registered in `INFERENCE_METHODS` / backend routing alongside the other AMX
  methods.

### GGUF substrate (`operators/gguf/`, `operators/amx/*`, `python/utils/gguf_cache.py`)

- Strip dequant from mmap'd GGUF blocks (`kt::gguf::dequant_rows_bf16`,
  AVX-512) with **no full BF16/FP32 tensor materialization**.
- Online-quant strip load paths (`from_mat_strip`) for INT8 / INT4 / KGroup /
  BF16.
- First-boot disk cache (`.kt` packed files under `KT_GGUF_CACHE_DIR`) with
  `KT_GGUF_CACHE=0|refresh` controls.
- Exotic GGML types fall back to ggml `to_float` — slow first boot, never a
  hard failure.

### BF16 on pre-SPR hosts (`operators/amx/la/amx_raw_kernels.hpp`)

- `KT_DOT_BF16`: dpbf16ps emulation over AVX-512 FP32 (classic bf16 pair
  trick), enabling the BF16 node on Cascade Lake / Gold 62xx hosts.

### KGroup kernel (`operators/amx/la/amx_kernels.hpp`)

- `make_kblock_abscale` generalized to any `k_group_size ≥ 16` (16-lane
  quarters), bit-identical to the legacy path at gs=32.

### Structural fixes

- **TP_MOE tps sizing**: `TP_MOE_Common<T, Concrete=Derived>` — CRTP-derived
  MOEs (K2 family) now allocate the tps as the full derived type (ASAN-
  confirmed heap-corruption class fix).
- **Virtual forward hooks**: `fill_down_a` / `down_output` on `AMX_MOE_BASE`
  so the derived wrappers can route the A-fill and the C writeback.

### Docs

- `docs/amxint4-smart.md` — full feature introduction (routing rule, accuracy
  table, enabling, status).
- README.md / README_zh.md pointers.

## Verification

- `test/per_commit/test_moe_gguf_amxint8_cache.py` — green end to end:
  cache equivalence, KGroup, BF16, SMART routing (all four dtype mixes:
  class asserted + output error bound), layout asserts.
- Accuracy classes (mini synthetic, E=8/H=256/I=512/topk=2):
  | Configuration | rel. error |
  |---|---|
  | Full BF16 | 0.0050 |
  | SMART, Q4-up/Q8_0-down → F4x8 fused | 0.154 |
  | SMART, Q5_K-down → F4x8 fused | 0.155 |
  | SMART, Q6_K-up/Q4_K-down → F4x8 (flipped) | 0.178 |
  | SMART, BF16-up/Q6_K-down → F8x16 (flipped) | 0.0136 |
  | SMART, BF16-up/Q4_K-down → F4x16 (flipped) | 0.171 |
  | AMXINT8 | 0.0188 |
  | AMXINT4_KGROUP | 0.1680 |
  | AMXINT4 per-row | 0.2167 |

## Test plan

```bash
# build (no AMX hardware needed; AVX-512 + VNNI)
CPUINFER_BUILD_TYPE=Release CPUINFER_ENABLE_AMX=OFF \
CPUINFER_ENABLE_AVX512=ON CPUINFER_ENABLE_AVX512_VNNI=ON \
python -m pip install .

# per-commit suite
python test/per_commit/test_moe_gguf_amxint8_cache.py

# serve a GGUF model (MiniMax-M2.7 UD-Q4_K_XL class)
python -m sglang.launch_server ... --kt-method AMXINT4_SMART \
  --kt-weight-path /path/to/model-GGUF --kt-cpuinfer 64 --kt-threadpool-count 2
```

## Notes / limitations

- The dedicated `FusedTwoStage` kernels and the fused-MOE host wrapper are the
  compute path for the mixed pairs; uniform pairs use the plain single-
  precision MOEs. The per-attribute compute mode is the default behavior for
  mixed layers.
- On an all-Q4-family GGUF the rule degrades to the plain per-row INT4
  routing; the INT8/BF16 escape hatches are per-layer automatic via the dtype
  rule.
- Decode-speed A/B of per-row INT4 vs INT8 at real dims on the target model
  is the next benchmark item.
