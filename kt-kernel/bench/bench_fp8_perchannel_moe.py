"""
Performance benchmark for FP8 Per-Channel MoE kernel (GLM-4.7-FP8 style).

This benchmark measures the performance of the FP8 Per-Channel MoE operator with:
- FP8 (E4M3) weights with per-channel scaling (one scale per output row)
- BF16 activations
- AVX-512 DPBF16 compute path
"""

import os
import sys
import time
import json
import subprocess
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import torch
from kt_kernel import kt_kernel_ext
from tqdm import tqdm

# Test parameters
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
num_experts_per_tok = 8
max_len = 25600

layer_num = 2
qlen = 1
warm_up_iter = 1000
test_iter = 3000
CPUINFER_PARAM = 80

CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

# Result file path
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
json_path = os.path.join(script_dir, "bench_results.jsonl")


def get_git_commit():
    """Get current git commit info"""
    result = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()
        result["commit"] = commit
        result["commit_message"] = commit_msg
        dirty_output = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        result["dirty"] = bool(dirty_output)
        if dirty_output:
            result["dirty_files"] = dirty_output.splitlines()
    except Exception as e:
        result["commit"] = None
        result["error"] = str(e)
    return result


def get_system_info():
    """Get system information"""
    info = {}
    uname = platform.uname()
    info["system_name"] = uname.system
    info["node_name"] = uname.node

    cpu_model = None
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    info["cpu_model"] = cpu_model
    info["cpu_core_count"] = os.cpu_count()
    return info


def record_results(result, filename=json_path):
    """Append result to JSON file"""
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def generate_fp8_perchannel_weights_direct(shape: tuple):
    """
    Directly generate random FP8 weights and per-channel scales.

    Args:
        shape: (expert_num, n, k) - weight tensor shape

    Returns:
        fp8_weights: uint8 tensor with random FP8 E4M3 values
        scales: fp32 tensor with per-channel scales, shape [expert_num, n]
    """
    e, n, k = shape

    # Directly generate random FP8 weights as uint8
    # FP8 E4M3 format: 1 sign + 4 exp + 3 mantissa
    fp8_weights = torch.randint(0, 256, (e, n, k), dtype=torch.uint8, device="cuda").to("cpu").contiguous()

    # Generate random per-channel scales (one per output row)
    # Use reasonable scale range (e.g., 2^-8 to 2^8)
    exponents = torch.randint(-8, 9, (e, n), dtype=torch.int32, device="cuda").to("cpu").contiguous()
    scales = (2.0 ** exponents.float()).to(torch.float32).contiguous()

    return fp8_weights, scales


def bench_fp8_perchannel_moe():
    """Benchmark FP8 Per-Channel MoE performance"""
    with torch.inference_mode():
        print("=" * 70)
        print("FP8 Per-Channel MoE Kernel Performance Benchmark")
        print("=" * 70)

        # Generate FP8 weights with per-channel scales
        print("\nGenerating FP8 weights with per-channel scales...")
        torch.manual_seed(42)
        gate_fp8, gate_scales = generate_fp8_perchannel_weights_direct((expert_num, intermediate_size, hidden_size))
        up_fp8, up_scales = generate_fp8_perchannel_weights_direct((expert_num, intermediate_size, hidden_size))
        down_fp8, down_scales = generate_fp8_perchannel_weights_direct((expert_num, hidden_size, intermediate_size))

        physical_to_logical_map = torch.tensor(range(expert_num), device="cpu", dtype=torch.int64).contiguous()

        # Build MoE layers
        print("Building FP8 Per-Channel MoE layers...")
        moes = []
        for _ in tqdm(range(layer_num), desc="Initializing MOEs"):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.quant_config.bits = 8
            config.quant_config.group_size = 0  # Not used for per-channel
            config.quant_config.zero_point = False
            config.quant_config.per_channel = True  # Enable per-channel mode

            config.gate_proj = gate_fp8.data_ptr()
            config.up_proj = up_fp8.data_ptr()
            config.down_proj = down_fp8.data_ptr()
            config.gate_scale = gate_scales.data_ptr()
            config.up_scale = up_scales.data_ptr()
            config.down_scale = down_scales.data_ptr()
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.AMXFP8PerChannel_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            moes.append(moe)

        # Generate input data
        print("Generating input data...")
        gen_iter = 1000
        expert_ids = (
            torch.rand(gen_iter * qlen, expert_num, device="cpu")
            .argsort(dim=-1)[:, :num_experts_per_tok]
            .reshape(gen_iter, qlen * num_experts_per_tok)
            .contiguous()
        )
        weights = torch.rand((gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu").contiguous()
        input_tensor = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        output_tensor = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
        qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

        # Warmup
        print(f"Warming up ({warm_up_iter} iterations)...")
        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    qlen_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

        # Benchmark
        print(f"Running benchmark ({test_iter} iterations)...")
        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    qlen_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start

        # Calculate metrics
        time_per_iter_us = total_time / test_iter * 1e6

        # FLOPS calculation:
        # Each expert performs: gate(intermediate x hidden) + up(intermediate x hidden) + down(hidden x intermediate)
        # GEMM/GEMV: 2 * m * n * k flops (multiply + accumulate = 2 ops per element)
        # For vector-matrix multiply (qlen=1): 2 * n * k per matrix
        flops_per_expert = (
            2 * intermediate_size * hidden_size  # gate
            + 2 * intermediate_size * hidden_size  # up
            + 2 * hidden_size * intermediate_size  # down
        )
        total_flops = qlen * num_experts_per_tok * flops_per_expert * test_iter
        tflops = total_flops / total_time / 1e12

        # Bandwidth calculation (FP8 = 1 byte per element)
        bytes_per_elem = 1.0
        # Weight memory: gate + up + down per expert
        bandwidth = (
            hidden_size
            * intermediate_size
            * 3
            * num_experts_per_tok
            * (1 / num_experts_per_tok * expert_num * (1 - (1 - num_experts_per_tok / expert_num) ** qlen))
            * bytes_per_elem
            * test_iter
            / total_time
            / 1e9
        )

        # Print results
        print("\n" + "=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print(f"Quant mode: FP8 (E4M3) with per-channel scaling")
        print(f"Total time: {total_time:.4f} s")
        print(f"Iterations: {test_iter}")
        print(f"Time per iteration: {time_per_iter_us:.2f} us")
        print(f"Bandwidth: {bandwidth:.2f} GB/s")
        print(f"TFLOPS: {tflops:.4f}")
        print("")

        # Record results
        result = {
            "test_name": os.path.basename(__file__),
            "quant_mode": "fp8_e4m3_perchannel",
            "total_time_seconds": total_time,
            "iterations": test_iter,
            "time_per_iteration_us": time_per_iter_us,
            "bandwidth_GBs": bandwidth,
            "flops_TFLOPS": tflops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "test_parameters": {
                "expert_num": expert_num,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_experts_per_tok": num_experts_per_tok,
                "quant_type": "per_channel",
                "layer_num": layer_num,
                "qlen": qlen,
                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "CPUInfer_parameter": CPUINFER_PARAM,
            },
        }
        result.update(get_git_commit())
        result.update(get_system_info())
        record_results(result)

        return tflops, bandwidth


if __name__ == "__main__":
    bench_fp8_perchannel_moe()
