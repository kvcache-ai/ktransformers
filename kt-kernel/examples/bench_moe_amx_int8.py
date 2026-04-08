#!/usr/bin/env python
# coding=utf-8
"""
AMX INT8 MoE Benchmark Script

Benchmarks performance of AMX-accelerated INT8 MOE operations with configurable parameters.
Supports uniform workload distribution across experts and optional CUDA stream mode.

Usage:
    python bench_moe_amx_int8.py [options]

Examples:
    # Default parameters
    python bench_moe_amx_int8.py

    # Custom parameters
    python bench_moe_amx_int8.py --layer_num 4 --expert_num 256 --workload 8 --use_cuda_stream

    # Full configuration
    python bench_moe_amx_int8.py --layer_num 2 --expert_num 128 --num_experts_per_tok 8 \
        --workload 4 --hidden_size 7168 --intermediate_size 2048 \
        --warmup_iter 100 --test_iter 1000 --use_cuda_stream
"""

import os
import sys
import time
import argparse

# Add build path for development
sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch

try:
    from kt_kernel import kt_kernel_ext

    HAS_KT_KERNEL = True
except ImportError as e:
    HAS_KT_KERNEL = False
    import_error = str(e)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AMX INT8 MoE Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model parameters
    parser.add_argument("--layer_num", type=int, default=2, help="Number of MoE layers")
    parser.add_argument("--expert_num", type=int, default=256, help="Number of experts per layer")
    parser.add_argument(
        "--num_experts_per_tok", type=int, default=8, help="Number of experts selected per token (top-k)"
    )
    parser.add_argument("--hidden_size", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("--intermediate_size", type=int, default=2048, help="Intermediate dimension size")

    # Workload parameters
    parser.add_argument("--workload", type=int, default=1, help="Workload (qlen, number of tokens)")
    parser.add_argument("--max_len", type=int, default=25600, help="Maximum sequence length for buffer allocation")

    # Benchmark parameters
    parser.add_argument("--warmup_iter", type=int, default=100, help="Number of warmup iterations")
    parser.add_argument("--test_iter", type=int, default=1000, help="Number of test iterations")

    # Execution mode
    parser.add_argument("--use_cuda_stream", action="store_true", help="Use CUDA stream mode (submit_with_cuda_stream)")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler and export trace.json")
    parser.add_argument("--profile_path", type=str, default="./trace.json", help="Path to save profile trace")

    # Worker configuration
    parser.add_argument("--cpuinfer_threads", type=int, default=60, help="Total CPU inference threads")
    parser.add_argument("--numa_count", type=int, default=2, help="Number of NUMA nodes")
    parser.add_argument(
        "--num_gpu_experts", type=int, default=0, help="Number of experts to place on GPU (first N experts)"
    )

    return parser.parse_args()


def generate_uniform_workload(expert_num, num_experts_per_tok, workload):
    """
    Generate expert_ids and weights with uniform workload distribution.

    workload = qlen (number of tokens)
    Each token selects num_experts_per_tok experts.
    Total expert calls = workload * num_experts_per_tok
    """
    qlen = workload

    # Randomly select num_experts_per_tok experts (uniform, no duplicates)
    # All tokens will use the same expert combination
    selected_experts = torch.randperm(expert_num)[:num_experts_per_tok].tolist()

    # Create expert_ids: all tokens use the same expert combination
    expert_ids = [selected_experts for _ in range(qlen)]

    # Create on GPU then copy to CPU (faster)
    expert_ids = torch.tensor(expert_ids, dtype=torch.long, device="cuda").to("cpu").contiguous()
    print(f"Selected experts (all tokens use same): {selected_experts}")
    print(f"Expert IDs shape: {expert_ids.shape}")

    # Uniform weights (normalized) - create on GPU then copy
    weights = torch.ones((qlen, num_experts_per_tok), dtype=torch.float32, device="cuda") / num_experts_per_tok
    weights = weights.to("cpu").contiguous()

    return expert_ids, weights, qlen


def run_benchmark(args):
    """Run the AMX INT8 MoE benchmark."""

    print("=" * 60)
    print("AMX INT8 MoE Benchmark")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Layers:              {args.layer_num}")
    print(f"  Experts per layer:   {args.expert_num}")
    print(f"  Experts per token:   {args.num_experts_per_tok}")
    print(f"  Hidden size:         {args.hidden_size}")
    print(f"  Intermediate size:   {args.intermediate_size}")
    print(f"  Workload (qlen):     {args.workload}")
    print(f"  Use CUDA stream:     {args.use_cuda_stream}")
    print(f"  Warmup iterations:   {args.warmup_iter}")
    print(f"  Test iterations:     {args.test_iter}")
    print(f"  CPU threads:         {args.cpuinfer_threads}")
    print(f"  NUMA nodes:          {args.numa_count}")

    # Generate uniform workload
    expert_ids, weights, qlen = generate_uniform_workload(args.expert_num, args.num_experts_per_tok, args.workload)
    print(f"\nActual qlen:           {qlen}")
    print(f"Total expert calls:    {qlen * args.num_experts_per_tok}")

    with torch.inference_mode():
        # Initialize CPUInfer
        if args.numa_count > 1:
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            worker_config.subpool_count = args.numa_count
            worker_config.subpool_numa_map = list(range(args.numa_count))
            threads_per_numa = args.cpuinfer_threads // args.numa_count
            worker_config.subpool_thread_count = [threads_per_numa] * args.numa_count
            cpu_infer = kt_kernel_ext.CPUInfer(worker_config)
        else:
            cpu_infer = kt_kernel_ext.CPUInfer(args.cpuinfer_threads)

        # Physical to logical mapping (identity)
        physical_to_logical_map = torch.arange(args.expert_num, dtype=torch.int64, device="cpu").contiguous()

        # GPU experts mask - set first num_gpu_experts to True if specified
        gpu_experts_mask = torch.zeros(args.expert_num, dtype=torch.bool, device="cpu")
        if args.num_gpu_experts > 0:
            num_gpu = min(args.num_gpu_experts, args.expert_num)
            gpu_experts_mask[:num_gpu] = True
            print(f"  GPU experts: {num_gpu} (experts 0-{num_gpu-1})")

        # Initialize MoE layers
        print("\nInitializing MoE layers...")
        moes = []
        for layer_idx in range(args.layer_num):
            # Create random weights on GPU then copy to CPU (faster)
            gate_proj = (
                torch.randn(
                    (args.expert_num, args.intermediate_size, args.hidden_size), dtype=torch.bfloat16, device="cuda"
                )
                .to("cpu")
                .contiguous()
            )
            up_proj = (
                torch.randn(
                    (args.expert_num, args.intermediate_size, args.hidden_size), dtype=torch.bfloat16, device="cuda"
                )
                .to("cpu")
                .contiguous()
            )
            down_proj = (
                torch.randn(
                    (args.expert_num, args.hidden_size, args.intermediate_size), dtype=torch.bfloat16, device="cuda"
                )
                .to("cpu")
                .contiguous()
            )

            # Configure MoE
            config = kt_kernel_ext.moe.MOEConfig(
                args.expert_num,
                args.num_experts_per_tok,
                args.hidden_size,
                args.intermediate_size,
                gpu_experts_mask.data_ptr(),
            )
            config.max_len = args.max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.pool = cpu_infer.backend_

            moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
            cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            cpu_infer.sync()

            moes.append(moe)
            print(f"  Layer {layer_idx} initialized")

        # Prepare input/output tensors
        input_tensor = torch.randn((qlen, args.hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        output_tensor = torch.zeros((qlen, args.hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        bsz_tensor = torch.tensor([qlen], dtype=torch.int32, device="cpu")

        # CUDA stream setup (if enabled)
        cuda_stream = None
        if args.use_cuda_stream:
            if not torch.cuda.is_available():
                print("\nWarning: CUDA not available, falling back to non-stream mode")
                args.use_cuda_stream = False
            else:
                cuda_stream = torch.cuda.current_stream().cuda_stream
                print(f"\nUsing CUDA stream: {cuda_stream}")

        # Warmup
        print(f"\nWarmup ({args.warmup_iter} iterations)...")
        for i in range(args.warmup_iter):
            moe = moes[i % args.layer_num]
            task = moe.forward_task(
                bsz_tensor.data_ptr(),
                args.num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                False,  # incremental
            )

            if args.use_cuda_stream:
                cpu_infer.submit_with_cuda_stream(cuda_stream, task)
                cpu_infer.sync_with_cuda_stream(cuda_stream)
            else:
                cpu_infer.submit(task)
                cpu_infer.sync()

        # Benchmark
        print(f"Benchmarking ({args.test_iter} iterations)...")

        if args.use_cuda_stream:
            torch.cuda.synchronize()

        # Setup profiler if enabled
        profiler = None
        if args.profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
            )
            profiler.__enter__()

        start_time = time.perf_counter()

        for i in range(args.test_iter):
            moe = moes[i % args.layer_num]

            if args.profile:
                torch.cuda.nvtx.range_push(f"iter_{i}")

            task = moe.forward_task(
                bsz_tensor.data_ptr(),
                args.num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                False,
            )

            if args.use_cuda_stream:
                if args.profile:
                    torch.cuda.nvtx.range_push("submit")
                cpu_infer.submit_with_cuda_stream(cuda_stream, task)
                if args.profile:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_push("sync")
                cpu_infer.sync_with_cuda_stream(cuda_stream)
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            else:
                cpu_infer.submit(task)
                cpu_infer.sync()

            if args.profile:
                torch.cuda.nvtx.range_pop()

        if args.use_cuda_stream:
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Export profiler trace
        if profiler:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(args.profile_path)
            print(f"\nProfile trace saved to: {args.profile_path}")

        # Calculate metrics
        # Note: each iteration processes ONE layer (round-robin: moe = moes[i % layer_num])
        time_per_iter_us = total_time / args.test_iter * 1e6

        # Bandwidth calculation
        # Weight size per expert: 3 * hidden_size * intermediate_size * bytes_per_elem
        bytes_per_elem = 1.0  # INT8
        weight_bytes_per_expert = 3 * args.hidden_size * args.intermediate_size * bytes_per_elem

        # Total weight bytes accessed per iteration (one layer per iteration)
        # Each token activates num_experts_per_tok experts
        total_experts_activated = qlen * args.num_experts_per_tok
        weight_bytes_per_iter = total_experts_activated * weight_bytes_per_expert

        bandwidth_gbs = weight_bytes_per_iter * args.test_iter / total_time / 1e9

        # FLOPS calculation
        # Per expert: 3 * hidden * intermediate * 2 (multiply-add)
        flops_per_expert = 3 * args.hidden_size * args.intermediate_size * 2
        total_flops = total_experts_activated * flops_per_expert * args.test_iter
        tflops = total_flops / total_time / 1e12

        # Results
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  Total time:           {total_time:.3f} s")
        print(f"  Time per iteration:   {time_per_iter_us:.2f} us  (= time per layer)")
        print(f"  Memory bandwidth:     {bandwidth_gbs:.2f} GB/s")
        print(f"  Compute throughput:   {tflops:.3f} TFLOPS")
        print("=" * 60)

        return {
            "total_time_s": total_time,
            "time_per_iter_us": time_per_iter_us,
            "bandwidth_gbs": bandwidth_gbs,
            "tflops": tflops,
        }


def main():
    args = parse_args()

    if not HAS_KT_KERNEL:
        print(f"Error: kt_kernel not available: {import_error}")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
