# Qwen3-30B-A3B BF16 Full-FT Performance

## Summary

This work removes two CPU bottlenecks in raw-BF16 routed-expert full-parameter training:

1. Expert weight gradients now aggregate all routed rows into BF16 mat-mat operations. The common driver dispatches to AVX512-BF16 on non-AMX CPUs and to the existing AMX microkernel on AMX CPUs.
2. Updated full BF16 expert weights are packed directly into persistent TP-local forward buffers. The reload no longer allocates, copies, and frees six temporary TP partitions per layer and step.

On a dual-socket AMD EPYC 9355 with one RTX 5090, Qwen3-30B-A3B at batch 1 and sequence length 1024 improves from 22.653 to 200.701 token/s. The comparison uses steps 8-15 of two 15-step runs.

## Implementation

The main implementation is commit `34d2102` on `fullft-development`:

- `BF16DWeightKernel` packs route-major activation and gradient panels into the existing raw BF16 `BufferA` and `BufferB` layouts.
- Work is split by `expert x intermediate tile x {gate/up, down}` and scheduled through the NUMA-local work-stealing pool.
- Gate and up share the packed input panel. Down reuses its packed intermediate panel.
- FP32 accumulation and BF16 gradient storage are unchanged.
- `BufferB::from_mat_strided()` packs a TP slice directly from the full `[E,F,H]` or `[E,H,F]` parameter tensor.
- The old scalar fallback remains available for non-BF16 kernels.

The staged profiler distinguishes wall-clock stages from summed worker CPU time. Worker stages use the `backward.base_weight_grad.worker_cpu.*` namespace and intentionally have no wall-clock parent percentage.

## Correctness

| Check | Result |
|---|---|
| AVX512-BF16 dW routes `1,31,32,33,65,1792,1825` and tail shapes | Exact match after BF16 store |
| Contiguous, transposed, and strided raw-BF16 repack | Pass |
| Synthetic full-FT gradients, TP1 and TP2, qlen `8,31,32,33,65` | 30/30 pass |
| Real HF layer-0 expert-0 gradients, TP1 and TP2 | Relative L2 `0.00355-0.00378`, cosine approximately 1 |
| 15-step end-to-end training | Finite loss and grad norm; gate/up/down parameters all changed |

The final run changed 16887 gate, 16272 up, and 14711 down values in each 131072-element parameter sample.

## End-to-End Result

Configuration:

- Model: `Qwen3-30B-A3B-Instruct-2507`
- Full-FT BF16 expert path, TP=2 NUMA partitions
- Batch size 1, sequence length 1024, gradient accumulation 1
- One RTX 5090, GPU 6
- 15 steps; steps 8-15 used for the stable comparison
- C++ staged profiler enabled; Torch trace disabled

| Stage | Before | After | Speedup |
|---|---:|---:|---:|
| End-to-end step | 45.204 s | 5.102 s | 8.86x |
| Throughput | 22.653 token/s | 200.701 token/s | 8.86x |
| Forward | 0.704 s | 0.668 s | 1.05x |
| Backward | 38.595 s | 3.250 s | 11.88x |
| Optimizer | 0.789 s | 0.687 s | 1.15x |
| Base-weight reload | 4.892 s | 0.272 s | 17.98x |
| TP0 base dW | 35.753 s | 0.933 s | 38.32x |

The backward/forward ratio falls from 54.9 to 4.87. Stable direct packing itself takes 0.261 s/step; TP partition and cleanup stages are zero. Step 1 optimizer initialization and the first full reload in step 2 are cold-start outliers and are excluded from stable throughput.

The output is retained at:

```text
/mnt/sft_yyj_yyj/qwen3-30b-a3b-fullft/outputs/20260717_070714-bench-1024-15step
```

It occupies 16 MiB because the large Torch trace was disabled. GPU 6 peaked at 31886 MiB. Its sampled utilization averaged 1.84% over model load and training, confirming that the remaining critical path is still mostly CPU-side.

## Detailed Profile

Stable steps 8-15, TP0 critical path:

| Stage | Time per step |
|---|---:|
| TP backward total | 1.447 s |
| Base dW wall time | 0.933 s |
| TP wrapper buffer clear | 0.352 s |
| Initial CPU MoE forward | 0.344 s |
| Checkpoint recompute CPU MoE forward | 0.505 s |
| Backward weight repack | 0.493 s |

Worker CPU times are summed across the NUMA pool and therefore are not additive wall time:

| dW worker stage | Summed CPU time per step |
|---|---:|
| Pack A | 11.248 s |
| Pack B | 9.933 s |
| Gate/up kernel | 3.614 s |
| Down kernel | 1.881 s |
| BF16 store | 1.960 s |

Packing is about 74% of the measured dW worker CPU time. It is now a better optimization target than replacing the BF16 microkernel.

## AMX Regression

The same tests were built on an Intel Xeon Platinum 8488C with AMX enabled. The binary contains `tileloadd` and `tdpbf16ps`, and all dW and repack cases pass.

The optional benchmark in `test_bf16_dweight --benchmark` compares the previous direct AMX tile loop with the common driver at route K=1024:

| Path | Median time |
|---|---:|
| Legacy tile loop | 1285.9 ns |
| Common driver | 1295.1 ns |
| Ratio | 1.0072 |

The measured AMX overhead is 0.72%, below the 5% regression limit.

## Next Targets

1. Reduce route-panel packing cost through wider transpose/copy primitives and panel reuse across adjacent output tiles.
2. Avoid clearing inactive or already-overwritten gradient regions in the TP wrapper.
3. Separate unavoidable backward repack from overlap gaps and remove only the exposed wall-clock part.
4. Revisit checkpoint recompute after the CPU dW and buffer-clear costs are lower.

