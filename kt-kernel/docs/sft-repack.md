# SFT backward repack: separate numeric layout from execution lifetime

Frozen FP8 expert weights remain E4M3 bytes with one scale per 128×128 weight
block. The scale grid has shape [N/128, K/128], not a fixed 128×128 shape.
Forward and backward-input GEMMs share the inference FP8 matrix policy.
BF16 decoding is bounded to a tile; there is no persistent layer-sized BF16
copy of the frozen weights.

For a PyTorch weight W[out, in], forward is Y = X Wᵀ and backward is dX = dY W.
The packed GEMM interface consumes A Bᵀ in both cases, so backward packs Wᵀ
as B. This is a layout requirement, not another mathematical transpose of dX.

## The three independent concerns

| Concern | Owner | Precision dependence |
| --- | --- | --- |
| Produce transposed packed weights | BufferB repack operation | FP8 bytes/scales or BF16 elements |
| Submit, wait, reuse and protect storage | sft_repack.hpp and TP_MOE_SFT | Shared execution protocol |
| Reuse or recompute expert activations | Existing activation policy/cache | Already shared |

FP8 previously returned early from asynchronous submission and repacked
synchronously inside backward. It now follows the same prefetch, wait,
synchronous fallback and destruction rules as BF16.

## Execution order

```text
CPU expert backward(L) completes
    +-- submit repack(L-1) --> parallel NUMA repack --------+
    +-- GPU non-expert backward(L)                         |
    +-- GPU checkpoint recompute(L-1), when enabled        |
    +-- actual CPU submission: wait ----------------------+
CPU expert backward(L-1) consumes ready packed weights
```

The first backward layer uses synchronous fallback. If another operator has
replaced the shared buffer after prefetch, the consumer prepares its own version
again. The shared pool holds one layer's backward weights, with one TP slice
per NUMA partition; it does not retain a separate transpose for every layer.

With CPU recomputation enabled, expert forward must also wait because repack
uses the same CPU worker pool. CPU retain / GPU recompute lets GPU work continue
before that CPU boundary. Activation reuse itself predates this change.

## Ownership and failure handling

SFT scratch storage is process-wide and WorkerPool does not support concurrent
job submission. A non-template process-wide RAII lease is acquired on the
submitting thread before NUMA dispatch and held through the CPU operation.
GPU work does not acquire it; NUMA workers must not acquire it themselves.

Every CPU entrypoint retires its own pending producer **before** acquiring the
lease. Reversing that order can deadlock: the producer also needs the lease.
The lease is outside NUMA dispatch, so it does not serialize the TP partitions.

- `RepackTask` uses one `std::future<void>` for completion and exceptions,
  instead of independent thread/in-flight/error state. Its serial host-side
  dispatcher submits/waits. The enclosing operator drains before freeing
  captured data; destruction joins without throwing.
- `RepackState` publishes readiness only after successful preparation.
  Partial failure invalidates the previous owner. A version identifies an
  operator's weight generation, not just a layer number; reload gets a new
  version and different models/dtypes cannot alias by layer number.
- Each NUMA partition keeps a stable shared state object even if the pool's
  metadata vector grows. CPU consumers retain the execution lease throughout
  their GEMMs, preventing another producer from overwriting shared weights.
- WorkerPool callbacks catch repack exceptions locally. The submitter joins
  all work before rethrowing. This boundary does not imply arbitrary exceptions
  in every nested worker callback are recoverable.

If multiple concurrent CPU execution streams are needed later, first give them
independent worker/scratch pools, then narrow the lease. Removing it alone
does not make the existing shared execution model concurrent.

## Precision-specific work

Gate, up and down share one work-stealing queue. Use whole-matrix tasks when
experts fill the pool; otherwise split disjoint output-row partitions for
BF16/FP8. Existing whole-matrix compensation handling is preserved.

FP8 transposes a packed 32×32-byte tile directly: transpose sixteen-by-sixteen
32-bit words, swap the middle bytes of each logical 2×2 block, and absorb the
packing permutation into vector load/store addresses. Scale bits are transposed
with their corresponding 128×128 blocks, without decode or requantization.

The common FP8 GEMM dispatch selects a small-M streaming path or, for M > 32,
a shared 128×128 decoded tile with a two-row dot leaf. Decoded storage is 32 KiB
per worker invocation. BF16 dot products accumulate in FP32; block scales retain
the reference accumulation order and release FMA rounding. Other group/shape
layouts keep the existing generic traversal.

## Validation

With `KTRANSFORMERS_CPU_DEBUG=ON`, CMake builds the independent repack/profile
protocol tests and native FP8 repack/GEMM tests. The GEMM regression compares
production output bitwise to the pre-optimization row policy over M tails,
K=128/256/1024/7168/7296, both projection directions and hot experts up to M=8192.

```sh
g++ -std=c++17 -O2 -pthread kt-kernel/cpu_backend/test/test_sft_repack.cpp -o /tmp/test_sft_repack
/tmp/test_sft_repack
cd kt-kernel
python -m pytest -q test/per_commit/test_sft_repack_pipeline.py
```

The native pipeline suite checks FP8/BF16 TP1/TP2, nonuniform scales,
forward/input/router/LoRA gradients, synchronous/asynchronous equivalence,
prefetch reuse, mixed-dtype shared-buffer replacement, reload and failure reuse.
This is correctness coverage, not a BF16 model performance benchmark.

Packed weights/scales and base GEMM outputs are checked bitwise. Repeated
forward, input/router gradients and non-reduced LoRA gradients use zero
tolerance. The existing token-block LoRA reductions merge FP32 partials under
mutexes in worker-completion order; baseline synchronous repeats can differ by
one BF16 rounding step. Only those reduced BF16 gradients allow one adjacent
representable value, checked by integer ULP distance. The bound has a rejection
test and does not relax the packed-weight or GEMM comparisons.

See [the training harness](../bench/sft_perf/README.md) for independent-process
FP8 A/B/B/A measurements and separate GPU-overlap diagnostics. A microbenchmark,
an explicit wait duration or a short loss curve cannot substitute for those
different measurements, nor establish long-term convergence.
