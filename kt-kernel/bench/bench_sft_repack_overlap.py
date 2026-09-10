# SPDX-License-Identifier: Apache-2.0
"""Compare serialized and overlapped real SFT repack + GPU matrix work.

This is a pipeline microbenchmark, not an end-to-end model training benchmark.
The numerical tests' Layer fixture keeps weights and ownership identical across
both modes. All GPU operations use preallocated tensors.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test" / "per_commit"))
from test_sft_repack_pipeline import Layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["bf16", "fp8"], default="fp8")
    parser.add_argument("--tp", type=int, choices=[1, 2], default=2)
    parser.add_argument("--threads-per-tp", type=int, default=16)
    parser.add_argument("--numa-map", default="0,1")
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--gpu-repeats", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required for overlap measurement")
    if (
        min(
            args.threads_per_tp,
            args.experts,
            args.hidden,
            args.intermediate,
            args.gpu_repeats,
            args.rounds,
        )
        <= 0
    ):
        parser.error("sizes and repetition counts must be positive")
    if args.hidden % 128 or args.intermediate % (128 * args.tp):
        parser.error("hidden and TP-local intermediate sizes must be multiples of 128")

    os.environ["KT_SFT_PROFILE"] = "1"
    import kt_kernel

    extension = kt_kernel.kt_kernel_ext
    if not hasattr(extension.moe, "AMXFP8_SFT_MOE"):
        parser.error("an AVX512-BF16/VBMI SFT build is required")
    # Match the fixture's pool construction but honor physical NUMA placement.
    config = extension.WorkerPoolConfig()
    config.subpool_count = args.tp
    config.subpool_numa_map = [int(value) for value in args.numa_map.split(",")][: args.tp]
    if len(config.subpool_numa_map) != args.tp:
        parser.error("numa-map must supply at least tp NUMA IDs")
    config.subpool_thread_count = [args.threads_per_tp] * args.tp
    pool = extension.CPUInfer(config)
    layers = [
        Layer(
            extension,
            pool,
            args.dtype,
            seed,
            experts=args.experts,
            hidden=args.hidden,
            intermediate=args.intermediate,
            qlen=1,
            layer_idx=seed,
            keep_reference=False,
        )
        for seed in (101, 103)
    ]

    a = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.empty_like(a)
    weight = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16) * 0.01
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)

    def gpu_work():
        x, y = a, b
        for _ in range(args.gpu_repeats):
            torch.mm(x, weight, out=y)
            x, y = y, x

    def sample(mode):
        # Force a different resident owner before every measured preparation.
        layers[1].moe.submit_backward_repack()
        layers[1].moe.wait_backward_repack()
        layers[0].moe.reset_profile_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        layers[0].moe.submit_backward_repack()
        if mode == "serialized":
            layers[0].moe.wait_backward_repack()
        gpu_start.record()
        gpu_work()
        gpu_end.record()
        torch.cuda.synchronize()
        wait_start = time.perf_counter()
        layers[0].moe.wait_backward_repack()
        wait_ms = (time.perf_counter() - wait_start) * 1000
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats = layers[0].moe.get_profile_stats()
        repack_ns = stats["wrapper.weights.backward_repack.total_ns"]
        assert layers[0].repack_calls() == args.tp, "FP8 async repack was disabled or skipped"
        return {
            "mode": mode,
            "elapsed_ms": elapsed_ms,
            "repack_ms": repack_ns / 1e6,
            "gpu_ms": gpu_start.elapsed_time(gpu_end),
            "post_gpu_wait_ms": wait_ms,
        }

    gpu_work()
    sample("serialized")
    sample("overlapped")
    rows = []
    # Alternating ABBA order reduces monotonic warmup/thermal bias.
    for _ in range(args.rounds):
        rows.extend(sample(mode) for mode in ("serialized", "overlapped", "overlapped", "serialized"))
    medians = {
        mode: statistics.median(row["elapsed_ms"] for row in rows if row["mode"] == mode)
        for mode in ("serialized", "overlapped")
    }
    report = json.dumps(
        {
            "config": vars(args),
            "cpu_variant": kt_kernel.__cpu_variant__,
            "samples": rows,
            "median_ms": medians,
            "pipeline_speedup_pct": 100 * (medians["serialized"] / medians["overlapped"] - 1),
        },
        indent=2,
        default=str,
    )
    if args.json_output:
        args.json_output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
