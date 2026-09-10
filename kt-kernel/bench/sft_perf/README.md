# DeepSeek V3.1 native FP8 SFT performance harness

This harness measures real eight-GPU LLaMA-Factory/FSDP2 LoRA training with a
frozen FP8 expert base. Each run launches fresh processes and writes a new
result directory; it does not modify installed framework packages.

## Configure before running

Copy `stack.example.json` and `deepseek_v31.yaml` outside the source checkout.
Replace every placeholder with an absolute path, an explicitly approved UTC
deadline, and the actual revision/hash of each separately built extension.
`training_config` selects your copied YAML. If omitted, the adjacent
`deepseek_v31.yaml` is used; its placeholder model/data paths are not runnable.

The current training recipe is deliberately specific: eight visible GPUs,
Linux with pidfd support, AVX512-BF16/VBMI, 64 CPU workers and two NUMA slices.
`train_entry.py` pins rank 0 to logical CPUs 0–127 and peers to 96–127, with
OpenMP thread counts 64 and 4 respectively. This matches the measured dual
EPYC 9355 host; inspect/adapt that policy before using a different topology.
The model contract requires all 58 DeepSeek V3.1 routed-expert backends on rank 0.
This is not a generic launcher for arbitrary models or machine sizes.

The frozen environment must supply the compatible KT-enabled Transformers,
Accelerate and LLaMA-Factory stack, PyTorch, PyYAML, numactl and nvidia-smi.
Start with a separate profiled smoke run:

```sh
python run.py --stack /absolute/path/stack.json --variant candidate \
  --label smoke-001 --warmup 1 --measured 2 --profile
```

After freezing warmup and measured lengths, omit profiling and run A/A noise
checks and independent-process A/B/B/A comparisons:

```sh
python sequence.py --stack /absolute/path/stack.json \
  --baseline baseline --candidate candidate --prefix comparison-001 \
  --warmup 2 --measured 16
```

## Measurement contract

- The CPU expert base remains native FP8; GPU compute is BF16 and LoRA optimizer
  updates are real. `pure_bf16: false` preserves FP32 LoRA masters required by the
  DeepSeek FP32 router in the measured FSDP2 stack.
- Rank 0 owns CPU experts and gathers tokens from eight ranks. CPU TP consists
  of two NUMA slices, not eight replicas of the expert pool.
- The window begins at warmup step-end and ends at the last measured step-end.
  Repack drain, CUDA completion and rank barriers occur only at those boundaries;
  the maximum rank duration is used.
- The global non-padding token counter already sums data-parallel tokens. Do
  not multiply it by eight again.
- Input preparation, epoch transitions, intermediate logging, optimizer and KT
  pointer updates remain inside the window. The final measured-step log is
  outside it, identically in both variants.
- Final adapter serialization is disabled in the benchmark process only.
  Forward, backward and optimizer implementations are not replaced.
- Actual optimizer callback counts and small local-shard parameter samples are
  recorded outside measurement. They do not prove full-parameter equivalence.
- Failed, incomplete, profiled and contract-mismatched runs are not accepted as
  formal throughput samples. Do not pool different warmup/measured horizons.
- A short repeated fixture measures throughput and numerical behavior on that
  fixture, not held-out quality or long-term SFT convergence.

## Ownership and evidence

`run.py` checks GPU occupancy/disk space, holds an experiment lease, enforces
deadline and timeout, and supervises only its own launcher and marked ranks.
`process_guard.py` matches the exact entrypoint and unique run directory,
rechecks start identity and binds signals to Linux pidfds. An unreaped direct
child remains owned even if its command line changes. Detached ranks are drained
after launcher exit; no broad process-name or system-wide Nsight cleanup is used.
A nominally successful launcher that still needs workers terminated is failed.

Results are append-only. No previous result, model weight or adapter is deleted.
Records include checkpoint config/index hashes and shard size/mtime metadata
(not full shard hashes), dataset hashes, training arguments, harness hashes,
per-rank imports/native hashes, losses and resource samples.
`report.py` validates identity and measurement contracts before comparison.
`sequence.py` aborts on failure rather than retrying or excluding samples.

`common.py` owns pure metric/contract helpers, `train_entry.py` observes
framework callbacks, and `run.py` owns external process lifetime.
`record_build.py` records source/archive/extension hashes, selected CMake
settings and generated compile/link commands; this is not a hermetic build.
`profile_report.py` normalizes native stage counters without adding nested
stages or inferring overlap from the explicit wait duration.

## Optional overlap timeline

Only a variant built with CUDA NVTX headers and declared `trace_supported` may
use `--profile --trace`. `KT_SFT_TRACE=1` enables thread-affine RAII ranges in
the `kt.sft` domain. Normal measurements explicitly disable tracing.

The default captures the first `repack.async` range, a warmup diagnostic.
`--trace-tail` starts at the first `cpu.backward` and collects through natural
process exit; it requires `--trace` and at most three total optimizer updates.
Nsight's `--kill=none` leaves process lifetime with the existing supervisor.

```sh
nsys export --type sqlite --output /new/capture.sqlite /path/capture.nsys-rep
python trace_report.py --sqlite /new/capture.sqlite --output /new/analysis.json
```

The reader opens SQLite read-only and supports the tested Nsight 2024.6 schema.
It rejects absent/empty CUDA captures, joins registered NVTX strings/domains,
and intersects complete repack intervals with GPU-kernel interval unions.
NCCL classification is name-based; inspect remaining names manually.
Non-NCCL includes copy/pointwise kernels, and this metric excludes CUDA memcpy
activity. It is neither pure arithmetic overlap nor critical-path savings.

The independent production GEMM regression is
`operators/amx/test/test_raw_fp8_gemm.cpp`. Tuning prototypes, machine-local
manifests and raw performance traces are intentionally not shipped here.
