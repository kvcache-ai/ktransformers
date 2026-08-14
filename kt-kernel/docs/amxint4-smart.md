# AMXINT4_SMART — Mixed-Precision MoE Routing from GGUF

**AMXINT4_SMART** is the AMX backend's per-layer, dtype-aware routing mode.
Instead of forcing the whole model through one precision, it reads each MoE
layer's original GGUF quantization and stores the layer at the tightest
precision that preserves it — AMXINT4, AMXINT8, or BF16 — so a mixed-precision
model (Q4_K gate/up with Q5_K/Q6_K down, for example) never pays the per-row
4-bit requant penalty on the tensors that did not come quantized that way.

It is the new primary method for GGUF-native deployment: point
`--kt-weight-path` at a GGUF directory and let the loader decide.

---

## The problem

GGUF models mix quantization levels across tensors. A typical "Q4_K_XL" MoE
checkpoint ships gate/up experts as `Q4_K` but the down experts as `Q5_K` /
`Q6_K` (MiniMax-M2.7 UD-Q4_K_XL: 186/186 MoE tensors are Q4_K/Q5_K/Q6_K).
The AMX backends historically had a single precision per layer:

- **AMXINT4 (per-row)** re-quantizes every tensor to one scale per row. On
  already-4-bit values that doubles the quantization: the per-row step
  discards the GGUF's per-32-block scale structure and costs ~0.15 relative
  RMS (≈0.217 whole-layer error class) — information the model author
  deliberately kept.
- **AMXINT8** is safe for the 5-8-bit tensors but wastes the 4-bit ones
  (2x RAM vs the GGUF, slower decode than the 4-bit path).
- **BF16** is the lossless fallback for F32/F16/BF16 tensors, at full size.

Choosing one precision for a whole layer is therefore always wrong for part
of it. AMXINT4_SMART routes per attribute instead.

---

## The 3-dtype storage rule

Each GGUF tensor is classified by its GGML type:

| Original GGUF quant | Node | Stored layer class |
|---|---|---|
| `Q2_K`, `Q3_K`, `Q4_K`, `Q4_0`, `Q4_1` (≤ Q4) | 0 — INT4 | **AMXINT4** (per-row) |
| `Q5_0`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ*`, `I8` ((Q4, Q8]) | 1 — INT8 | **AMXINT8** |
| `F32`, `F16`, `BF16` | 2 — BF16 | **BF16** |

A layer's attributes are classified independently:

- `upstream node = max(node(gate), node(up))`
- `downstream node = node(down)`
- **the layer is stored at `max(upstream, downstream)`** and served by the
  existing kernel of that class.

This is the whole routing rule. It is deterministic, derived from the GGUF
itself, and requires no calibration, thresholds, or error measurements at
load time.

### Stage pairs are wrappers, not new kernels

The mixed pairs — `(0→1)` int4×int8, `(1→2)` int8×bf16, `(0→2)` int4×bf16 —
are implemented as **load-time conversion wrappers**: the layer's weights are
converted to the higher precision of the pair and served by the existing
AMXINT8 / BF16 kernel. The layer's original quantization is preserved in the
log; its storage is the wrapper's target format.

Consequences on the real model (MiniMax-M2.7 UD-Q4_K_XL): gate/up are Q4_K,
down are Q5_K/Q6_K → upstream node 0, downstream node 1 → **every layer is
stored AMXINT8** through the (0→1) wrapper. The 4-bit tensors pay an INT8
re-quant at load (≈1% error, dominated by the GGUF's own quantization), and
decode runs on the proven AMXINT8 kernel — no per-row INT4 error anywhere.

---

## The GGUF substrate

Everything underneath the routing was built for GGUF-native operation:

- **Strip dequant, no materialization**: expert tensors are dequantized from
  the mmap'd GGUF blocks in AVX-512 strips (`kt::gguf::dequant_rows_bf16`)
  and packed straight into the backend format — no full BF16/FP32 tensor ever
  exists in RAM or Python. RAM stays at the stored class's size.
- **Online quant**: the INT8 / INT4 (per-row) / KGroup / BF16 load paths all
  consume the strip stream (`from_mat_strip`), shared by every backend.
- **First-boot disk cache**: packed `.kt` files under
  `<gguf-dir>/.kt_cache` (per method + TP count); later boots load the cache
  and skip the GGUF entirely.
  - `KT_GGUF_CACHE_DIR=<dir>` — relocate the cache
  - `KT_GGUF_CACHE=0` — disable (quantize every boot)
  - `KT_GGUF_CACHE=refresh` — force a rebuild
- **BF16 on pre-Sapphire-Rapids hosts**: the BF16 backend is powered by
  `KT_DOT_BF16`, a dpbf16ps emulation over AVX-512 FP32 (classic bf16 pair
  trick), so the BF16 node runs on Cascade Lake / Gold 62xx without any
  bf16 hardware.
- **Exotic GGML fallback**: IQ\* and other rare types fall back to ggml's
  `to_float` — a slower first boot, never a hard failure.

---

## Enabling

```bash
python -m sglang.launch_server \
  [your normal SGLang parameters...] \
  --kt-method AMXINT4_SMART \
  --kt-weight-path /path/to/model-UD-Q4_K_XL-GGUF \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2
```

No conversion script. First boot dequantizes + packs + writes the cache;
every later boot loads the cache. The method name for the routing layer is
registered in `python/experts.py` alongside `AMXINT4` / `AMXINT8` /
`AMXINT4_KGROUP` / `BF16`.

At load, each layer logs its source and storage:

```
[AMXMoEWrapper] Layer 60: GGUF source /path/UD-Q4_K_XL (INT8 cache: .../.kt_cache/amxint4_smart-tp2-...)
```

and the stage edge between consecutive layers is reported as a boundary
event (e.g. `(60, 61): (1->0)`) before the next layer loads.

---

## Measured accuracy (mini synthetic: E=8, H=256, I=512, topk=2)

| Configuration | Relative error vs BF16 reference |
|---|---|
| Full BF16 (upper bound) | 0.0050 |
| **AMXINT4_SMART**, Q8_0-down → INT8 wrapper | 0.0184 |
| **AMXINT4_SMART**, Q5_K-down → INT8 wrapper | 0.0191 |
| **AMXINT4_SMART**, Q6_K-up → INT8 wrapper | 0.0194 |
| AMXINT8 (whole layer) | 0.0188 |
| AMXINT4_KGROUP | 0.1680 |
| AMXINT4 per-row (whole layer) | 0.2167 |

The SMART INT8-stored classes land at the AMXINT8 accuracy — the 4-bit
tensors' information is preserved through the wrapper conversion instead of
being destroyed by a second 4-bit requant.

## Verification

The per-commit suite exercises the full path end to end:
`test/per_commit/test_moe_gguf_amxint8_cache.py` — cache equivalence,
KGroup, BF16, SMART routing (all four dtype mixes, asserting both the routed
class and the output error bound), and the layout asserts. The suite is the
gate for this feature and is green.

## Status

- **Shipped**: the routing, the GGUF substrate, the BF16 arm, the caches,
  the logging.
- **Parked (bound but unrouted by design)**: the dedicated `FusedTwoStage`
  kernels (`operators/amx/la/amx_fused.hpp`, VNNI int16-staging decode,
  math-verified) and the fused-MOE host wrapper. The routing intentionally
  uses the conversion wrappers; the parked kernels are the fallback path if a
  per-attribute compute mode is ever wanted behind a flag.
- **Next**: a decode-speed A/B of per-row INT4 vs INT8 at real dims on the
  target model, and the KGroup decode pass.
