# AscendC Fused MXFP4 -> W8A8 FRACTAL_NZ Kernels (`ascendc_mxfp4`)

High-throughput AscendC kernels and Python runtime for streaming MoE prefill on Ascend 910B (`dav-c220`), converting raw GGUF `block_mxfp4` weights directly into `FRACTAL_NZ` (`ACL_FORMAT_FRACTAL_NZ = 29`) int8 W8A8 slots consumed by `torch_npu.npu_grouped_matmul` / `npu_fused_experts`.

## What's Inside

- `mxfp4_fused_kernel.cpp`:
  - `mxfp4_fused`: de-interleaved `codes` + `e8m0 scale` -> int8 `[lo|hi]` planes + `oscale`.
  - `mxfp4_fused_blk`: raw GGUF `block_mxfp4` (`[nb * 17]` bytes/row) -> int8 `[lo|hi]` planes + `oscale`.
  - **`mxfp4_fused_blk_nz`** *(new)*: fuses de-interleave, E8M0+FP4 LUT decode, per-row absmax quantization, halfblock K-order rearrange, batched hardware 16×16 transpose (`TransDataTo5HD<half>` with `repeatTimes=8` over 128 columns), and **4,096 B contiguous bursts (`dstStride=0`)** directly into `FRACTAL_NZ [E, OUT/32, IN/16, 16, 32]`. Fits in `181.1 KB < 192 KB` Unified Buffer on `dav-c220` via non-overlapping Phase 1 / Phase 2 reuse.
- `mxfp4_fused_op.py`:
  - Builds/caches `libmxfp4fused.so` with `bisheng` on first use.
  - Fixes a latent `out_nz is None` allocation bug where `torch.empty(...)` created an ND tensor and forced `out_nz[c:ce].copy_(nz)` to de-format `FRACTAL_NZ` back to `ND`.
  - Automatically dispatches `convert_proj_blk` / `mxfp4_layer_to_nz_slots_blk` to `mxfp4_fused_blk_nz` when `IN in {2048, 4096}`, `OUT % 32 == 0`, and the target buffer is `FRACTAL_NZ`, with transparent fallback otherwise.
  - Exposes **`mxfp4_layer_to_nz_slots_blk_async(host_b13, host_b2, H, I, out_w13, out_w2)`**: streams pinned host GGUF blocks through 2 small ping-pong device buffers (`chunk=8` experts, `~140.5 MiB` total instead of `~3.42 GB` full-layer copies) on a dedicated ACL H2D stream overlapped continuously across `w13 -> w2`, kicks off Chunk 0 H2D before syncing `compute_stream` to overlap the previous layer's Attention/GMM, and tears down ACL streams/events via `atexit`.
- `test_mxfp4_fused_op.py`:
  - Host CPU guardrail/LUT unit tests + Ascend 910B bit-exactness, `FRACTAL_NZ` format, and fallback verification suite.

## Caller Integration (`sglang` `kt_stream_prefill.py`)

When `KT_MXFP4_DEPOOL=1`, `KT_MXFP4_GGUF_DEDUP=1`, and `KT_MXFP4_BLK_KERNEL=1`:

1. Allocate streaming slots as real `FRACTAL_NZ` (`reserve_slot(E, H, I, dev)`) instead of ND (`reserve_slot_depool`).
2. Call `mxfp4_layer_to_nz_slots_blk_async(host_b13, host_b2, H, I, slot13, slot2)` directly on the pinned staging buffers instead of copying full-layer `blk13/blk2` to device HBM first.

## Environment Variables

| Variable | Default | Description |
|---|---:|---|
| `KT_MXFP4_DIRECT_NZ` | `1` | Enable direct `FRACTAL_NZ` kernel (`0` forces legacy `mxfp4_fused_blk`). |
| `KT_MXFP4_BLOCKDIM` | auto (device AIV count, fallback `40`) | AI Vector Core launch count (`40` on 910B4-1 avoids tail waves). |
| `KT_MXFP4_ASYNC_CHUNK` | `8` | Experts per ping-pong H2D chunk in `mxfp4_layer_to_nz_slots_blk_async`. |
| `KT_MXFP4_SO_DIR` | module dir or user cache | Directory to cache compiled `libmxfp4fused.so`. |

## Verified Hardware Results (Ascend 910B4-1, DeepSeek-V4-Flash, `E=256, H=4096, I=2048, L=43`)

- **Bit-exactness**: 100% byte-identical (`torch.equal == True`) across all `6,446,645,248 B` (`6.44 GB`) of Layer 0 (`w13`, `w2`, `s13`, `s2`, and `FRACTAL_NZ` resident `index_select` gather); real `npu_grouped_matmul` `max_abs = 0.0`.
- **Single-layer breakdown (Layer 0, `3.423 GB` GGUF input)**:
  - Legacy `mxfp4_layer_to_nz_slots_blk`: `402.32 ms`
  - Raw PCIe Gen4 H2D floor (`25.60 GB/s`): `133.68 ms`
  - `mxfp4_fused_blk_nz` kernel-only (`bd=40`): **`131.45 ms`** (`w13=85.28 ms, w2=46.23 ms`)
  - `mxfp4_layer_to_nz_slots_blk_async` (`bd=40, chunk=16`): `142.08 ms` (`2.83×` vs legacy, 94.1% of raw PCIe H2D roofline)
  - `mxfp4_layer_to_nz_slots_blk_async` (`bd=40, chunk=8`, early Chunk-0 H2D): **`139.53 ms` (2.91× vs legacy, 95.8% of raw PCIe H2D roofline, −140 MB ping-pong HBM)**
- **End-to-end ABBA (`base0 -> opt0 -> opt1 -> base1`, 52/52 requests bit-exact `output_ids`)**:
  - `1024 -> 64`: TTFT `19.74 -> 11.13 s` (**−43.6%**, `1.77×`)
  - `4096 -> 64`: TTFT `20.13 -> 11.12 s` (**−44.8%**, `1.81×`)
  - `7168 -> 64`: TTFT `20.52 -> 10.23 s` (**−50.2%**, `2.01×`, `705.1 tok/s`)
  - `4096 -> 1000`: TTFT `20.10 -> 9.98 s` (**−50.3%**, `2.02×`; `opt0` `9.06 s / 2.22×`), post-request HBM **`−850 MB`**

## Running Tests

```bash
# Host CPU unit tests (no NPU required)
python3 -m unittest kt-kernel/tools/ascendc_mxfp4/test_mxfp4_fused_op.py -v

# Full Ascend 910B bit-exactness + FRACTAL_NZ + fallback verification
python3 -m unittest kt-kernel/tools/ascendc_mxfp4/test_mxfp4_fused_op.py -v
```
