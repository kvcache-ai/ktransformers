# DeepSeek V3.1 FP8 SFT: measured results and limits

These measurements use the frozen production implementation before PR packaging
and rebasing. They are not new measurements of a rebased binary. The baseline
already includes the delayed CPU-consumer wait from #2188; FP8 still repacked
synchronously because its asynchronous submission returned early.

## Eight-GPU training comparison

Host: two AMD EPYC 9355 CPUs (64 physical cores, 128 logical CPUs, two NUMA nodes)
and eight RTX 5090 GPUs. CPU execution uses AVX512-BF16/VBMI, not AMX FP8 FMA.
Toolchain: GCC 11.4, CUDA 12.8.61, PyTorch 2.9.1+cu128, Python 3.11.15 and a
frozen KT-enabled Transformers/Accelerate/LLaMA-Factory stack.

Both variants use the original DeepSeek V3.1 block-E4M3 checkpoint, 58 CPU MoE
layers, 256 experts/layer, top-k 8, hidden size 7168, intermediate size 2048,
64 CPU workers/TP2, shared backward buffers, CPU retain / GPU recompute, no GPU
experts, LoRA rank 8/alpha 16/dropout 0, FP32 LoRA masters, GPU BF16 computation,
fused AdamW, learning rate 1e-4 and cosine scheduling. Batch is one sequence per
GPU, length 1024, gradient accumulation one; all 8192 global tokens/update are
non-padding. The same 16 fully supervised fixture rows repeat across steps.

Each independent process performs 18 real optimizer updates: two warmup and
16 measured updates, totaling 131072 measured global tokens. Profiling/tracing
and final adapter serialization are disabled for these runs.

| Order | Variant | Global tokens/s | Seconds/update |
| --- | --- | ---: | ---: |
| A1 | baseline | 62.0345 | 132.0555 |
| B1 | optimized | 136.4766 | 60.0249 |
| B2 | same optimized binary | 136.3259 | 60.0913 |
| A2 | return to baseline | 62.2725 | 131.5509 |

Mean throughput is **62.1535 -> 136.4012 tokens/s (2.1946x)**. Sample coefficients
of variation are 0.2707% and 0.0782%, with only two independent runs per variant.
This is descriptive repeatability, not a confidence interval or significance test.
The maximum rank duration is used; rank/global tokens are not counted twice.
All four runs used identical non-KT framework paths/versions and rank-0 Python
tree hashes. Each rank recorded its actual native hash and completed 18 updates.

## Numerical evidence

All four loss curves begin at 12.671875 and end at 11.8515625. Recorded losses
and gradient norms are finite. Final gradient norms are 0.3530–0.3574.

| Comparison | Maximum per-step loss difference | Maximum grad-norm difference |
| --- | ---: | ---: |
| A2 / A1 | 0.0078125 | 0.0074846 |
| B2 / B1 | 0.015625 | 0.0118591 |
| B1 / A1 | 0.015625 | 0.0123862 |
| B2 / A1 | 0.015625 | 0.0172519 |

Small CPU parameter probes also differ between same-binary repeats. These are
64-value local samples, not full adapter comparisons. The short repeated fixture
and lack of held-out evaluation do **not** establish real-corpus convergence,
full-parameter equivalence, or save/resume correctness.

Development validation separately passed 29 small native pipeline cases and
3562 bitwise matrix comparisons across production and experimental variants.
The PR excludes unselected prototypes and provides a production-only regression:
274 comparisons over two seeds, dispatch boundaries, block scales, projection
directions and M up to 8192. See `test_raw_fp8_gemm.cpp` and the pipeline tests.

During PR integration, a strict repeat-equality assertion exposed one BF16
`down_lora_a` element differing by 2.3283064e-10 (one BF16 ULP). Both baseline
and optimized **synchronous** repeats reproduce it. The unchanged parallel
token-block reductions merge FP32 partials in mutex-acquisition order. The
pipeline test now allows one BF16 ULP only for those reduced LoRA gradients,
keeps zero tolerance elsewhere, and tests rejection beyond the bound. This is
not a change to the production kernel or evidence about the sole cause of
full-training drift.

## Repack, decoded storage and overlap

- Whole-layer TP2 FP8 repack: 105.04 ms -> 41.28 ms, versus a matched copy at
  40.77 ms. This is about 2.54x faster and 98.8% of the matched-copy rate.
- M=1 logical-weight throughput reaches 92.1% / 89.2% of the two matched-read
  references. These are logical traffic metrics, **not DRAM hardware counters**.
- High-M production GEMM reaches about 57.7–73.4% of the independent BF16-dot
  reference. Not every shape is memory-bound; compute/cache/scheduling work remains.
- Frozen base and shared backward BufferB remain FP8 plus scales. The high-M
  decoded tile is 32 KiB/worker; no full-layer BF16 weight copy persists.

A separate three-update trace captured eight GPU workers, 641892 CUDA kernel
events and 171 complete asynchronous CPU repack ranges. Repack ranges total
8.4854 seconds; intersection with the union of non-NCCL GPU-kernel intervals is
4.3602 seconds (51.3849%). Attention/GEMM events are present, but the non-NCCL
set also contains copy/pointwise kernels. This observed overlap includes warmup,
excludes CUDA memcpy activity and is not critical-path savings or a steady-state
hidden-time percentage. An earlier capture without CUDA events was inconclusive.

All 32 formal rank records report CUDA peak allocated of 18,014,291,968 bytes.
CPU RSS snapshots were taken at different stages/profiling settings and do not
constitute a controlled CPU-memory regression comparison.

BF16 model performance, INT8 optimization, AMX runtime performance and long-term
training quality are outside this result. See the [design](sft-repack.md) and
[harness contract](../bench/sft_perf/README.md) for implementation and reproduction.
