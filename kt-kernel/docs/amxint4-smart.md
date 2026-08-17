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
- the layer is kept **per attribute**: gate/up stay at the upstream node,
  down at the downstream node — each tensor in RAM at its smallest native
  precision, nothing expanded unless an adjacent stage lives in a larger
  format.

This is the whole routing rule. It is deterministic, derived from the GGUF
itself, and requires no calibration, thresholds, or error measurements at
load time.

### Stage pairs are computational wrappers, not stored conversions

The mixed pairs — `(0→1)` int4×int8, `(1→2)` int8×bf16, `(0→2)` int4×bf16 —
are **computational wrappers**: the layer keeps its per-attribute precisions
in RAM (gate/up at the upstream node, down at the downstream node — the
smallest possible footprint) and the mixed GEMMs run through the fused
two-stage kernels at compute time. Only the activations are widened in the
fused decode, never the stored weights; a tensor is only ever expanded when
its adjacent stage lives in a larger format, trading a little RAM traffic for
the hardware throughput of the wider node with minimal accuracy loss.

Each fused wrapper serves **both orientations** of its stage pair. The
wrapper decides at entry — one step before the GEMM loops — which stage is
the wider one, from the layer's per-attribute nodes, and picks the kernel
composition accordingly. The same `F4x8` wrapper therefore runs `(0,1)`
layers with gate/up at per-row INT4 and `(1,0)` layers with gate/up at INT8;
in both cases only the used buffer group is allocated, so RAM stays at the
per-attribute precisions. The stage dispatch is decided at the same entry
point (the int-family stages run `integer_mat_mul`, the BF16 stage
`float_mat_vec`), so the compositions stay symmetric at minimal runtime cost.

Consequences on the real model (MiniMax-M2.7 UD-Q4_K_XL): gate/up are Q4_K,
down are Q5_K/Q6_K → the layers land on the FusedInt4xInt8 wrapper; the
4-bit tensors keep their native footprint; decode widens only the
activations.

#### Display convention (user-facing form)

The pair id `(0, 1)` is an **internal routing key only** — it is not a
storage format. The user-facing display of a routed layer reads as:

```
IN-RAM:  (gate: Q4 -> AMXINT4, up: Q4 -> AMXINT4, down: Q5 -> AMXINT8)
Operators Engaged:
  In (bf16) --(In_Type, node 0)--> AMXINT4 [gate/up per-row INT4]
    -> h = silu(g)*u  (1, 1)  (bf16 intermediate, node-1 quantization for the down-A)
  --AMXINT8 [down per-row INT8]--> Out (bf16)
  via the FusedInt4xInt8 two-stage wrapper (one fused class per precision
  pair; the internal pair (0,1)/(1,0) selects the orientation at entry)
```

The `(1, 1)` is the intermediate state: the fused decode materializes
`h = silu(g)·u` in bf16 between the stages (the same scratch the plain
decode uses), then quantizes it per-row for the down stage — one
(1 token × 1 expert) computation at a time.

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
| **AMXINT4_SMART**, IN-RAM (gate Q4→INT4, up Q4→INT4, down Q8_0→INT8) → F4x8 | 0.154 |
| **AMXINT4_SMART**, IN-RAM (gate Q4→INT4, up Q4→INT4, down Q5_K→INT8) → F4x8 | 0.155 |
| **AMXINT4_SMART**, IN-RAM (gate Q6_K→INT8, up Q6_K→INT8, down Q4_K→INT4) → F4x8 (flipped) | 0.178 |
| **AMXINT4_SMART**, BF16-up/Q6_K-down → F8x16 (flipped) | 0.0136 |
| **AMXINT4_SMART**, BF16-up/Q4_K-down → F4x16 (flipped) | 0.171 |
| AMXINT8 (whole layer) | 0.0188 |
| AMXINT4_KGROUP | 0.1680 |
| AMXINT4 per-row (whole layer) | 0.2167 |

The SMART mixed pairs land at the accuracy class of the **narrower** stage
(0.154–0.178 when the down is per-row INT4, 0.0136 when the down is INT8 —
in both orientations the down is the accuracy-critical stage since it is the
largest tensor); RAM stays at the per-attribute precisions.

## Verification

The per-commit suite exercises the full path end to end:
`test/per_commit/test_moe_gguf_amxint8_cache.py` — cache equivalence,
KGroup, BF16, SMART routing (all four dtype mixes, asserting both the routed
class and the output error bound), and the layout asserts. The suite is the
gate for this feature and is green.

## Status

- **Shipped**: the routing, the per-attribute storage, the fused compute
  wrappers for the mixed pairs, the GGUF substrate, the BF16 arm, the caches,
  the logging.
- **Next**: a decode-speed A/B of per-row INT4 vs INT8 at real dims on the
  target model, and the KGroup decode pass.
