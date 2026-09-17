# Recommended Configurations for Consumer Hardware

This guide gives starting-point configurations for running MoE models with
kt-kernel on common consumer setups (a single desktop CPU plus one NVIDIA
GPU). It focuses on the knobs that matter most on consumer hardware: the CPU
kernel variant, the expert quantization format, and how many experts to keep
on the GPU.

These are sensible defaults, not tuned records. Always verify on your own
machine and adjust based on available VRAM and system RAM.

## 1. CPU kernel variant

kt-kernel detects your CPU at import time and loads the best available kernel
variant, so you normally do not need to set anything. The progressive
hierarchy (best to fallback) is:

| Variant | Needs | Typical CPUs |
|---|---|---|
| `amx` | AMX + full AVX512 | Intel Sapphire/Emerald Rapids (server) |
| `avx512_bf16` / `avx512_vbmi` / `avx512_vnni` / `avx512_base` | AVX512 subsets | Recent Intel / AMD Zen4+ |
| `avx2` | AVX2 only | Most consumer desktops (Intel Haswell+, AMD Zen+) |

You can override detection when debugging:

```bash
export KT_KERNEL_CPU_VARIANT=avx2   # or avx512_vnni, amx, ...
export KT_KERNEL_DEBUG=1            # print the detected/loaded variant
```

Most consumer desktops land on `avx2` or one of the `avx512` variants. See the
[AVX2 Tutorial](./AVX2-Tutorial.md) for the AVX2 path.

## 2. Choosing an expert quantization method

Pick the method that matches your weights and CPU variant:

| Method | Good for | Notes |
|---|---|---|
| `GPTQ_INT4` | Pre-quantized GPTQ INT4 checkpoints | Widely available community weights |
| `RAWINT4` | Native INT4 (per-channel / K-group) | AVX512/AMX; VNNI-accelerated variant available |
| `MXFP4` | MXFP4 checkpoints (e.g. DeepSeek-V4-Flash routed experts) | Microscaling 4-bit |
| `FP8` / `FP8_PERCHANNEL` | FP8 checkpoints | Needs enough RAM for FP8 experts |
| `AMXINT4` / `AMXINT8` | AMX servers | Not for consumer AVX2-only CPUs |

On an AVX2-only desktop, prefer `GPTQ_INT4` or `MXFP4`. On AVX512/AMX, the
native `RAWINT4`/`FP8` paths (see
[Native Precision Tutorial](./Native-Precision-Tutorial.md)) are usually
faster.

## 3. GPU expert offloading

The most impactful consumer knob is how many experts sit on the GPU. Hot
experts on the GPU cut PCIe traffic; the rest run on the CPU.

| Flag | Meaning |
|---|---|
| `--kt-num-gpu-experts N` | Experts per MoE layer kept on the GPU |
| `--kt-gpu-experts-ratio R` | Fraction (0-1) of all experts on GPU (overrides the count) |
| `--kt-expert-placement-strategy` | `uniform` (default), `frequency`, `front-loading`, `random` |
| `--init-expert-location FILE.pt` | Activation stats for the `frequency` strategy |

Start with `uniform`, raise `--kt-num-gpu-experts` until VRAM is nearly full,
then try `frequency` with recorded activation stats for a further bump. Full
details in the [Expert Scheduling Tutorial](./experts-sched-Tutorial.md).

## 4. Example starting points

These assume a single GPU and enough system RAM to hold the CPU-side experts.
Tune `--kt-num-gpu-experts` to fill VRAM.

| GPU (VRAM) | Suggested method | GPU experts | Placement |
|---|---|---|---|
| RTX 3090 / 4090 (24 GB) | `GPTQ_INT4` or `MXFP4` | small, e.g. 4-8 / layer | `uniform`, then `frequency` |
| RTX 5090 (32 GB) | `GPTQ_INT4` / `RAWINT4` | more, e.g. 8-16 / layer | `frequency` |
| 2x 24 GB | `RAWINT4` / `FP8` | raise ratio | `frequency` |

System RAM, not VRAM, usually caps which model fits: the CPU holds the bulk of
the experts. Ensure RAM comfortably exceeds the CPU-side expert footprint.

## 5. Troubleshooting

- Out of VRAM: lower `--kt-num-gpu-experts` or `--kt-gpu-experts-ratio`.
- Slow first tokens: expected while experts warm up; try `frequency` placement.
- Wrong CPU variant loaded: set `KT_KERNEL_DEBUG=1` and, if needed,
  `KT_KERNEL_CPU_VARIANT` to force one.
- Using the CLI instead of raw flags? See the [KT CLI guide](./kt-cli.md).

## Related docs

- [AVX2 Tutorial](./AVX2-Tutorial.md)
- [Native Precision Tutorial](./Native-Precision-Tutorial.md)
- [Expert Scheduling Tutorial](./experts-sched-Tutorial.md)
- [KT CLI](./kt-cli.md)
