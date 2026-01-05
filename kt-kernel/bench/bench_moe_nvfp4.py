#!/usr/bin/env python
# coding=utf-8
"""
Description  : Benchmark for NVFP4 MoE operator with E2M1 FP4 quantization
Author       : KVCache.AI
Date         : 2024-12-01
Version      : 1.0.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""
import os
import sys
import time
import json
import subprocess
import platform

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import torch
from kt_kernel import kt_kernel_ext

# Test parameters
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
num_experts_per_tok = 8
block_size = 16  # NVFP4 requires block_size=16
max_len = 25600

layer_num = 5
qlen = 1
warm_up_iter = 1000
test_iter = 10000
CPUINFER_PARAM = 64

CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()


def generate_nvfp4_weights_direct(shape: tuple, block_size: int = 16):
    """
    Directly generate random NVFP4 weights and FP8 E4M3 scales.

    Args:
        shape: (expert_num, n, k) - weight tensor shape
        block_size: block size for scaling (16 for NVFP4)

    Returns:
        packed_fp4: uint8 tensor with packed FP4 values [expert_num, n, k // 2]
        scales_fp8: uint8 tensor with FP8 E4M3 scales [expert_num, n, k // block_size]
    """
    e, n, k = shape
    num_blocks = k // block_size

    # Directly generate random packed FP4 weights as uint8
    # Each byte contains two 4-bit E2M1 values
    packed_fp4 = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cpu").contiguous()

    # Generate FP8 E4M3 format scales
    # FP8 E4M3: 1 sign + 4 exp + 3 mantissa, valid range for positive scales
    scales_fp8 = torch.randint(1, 127, (e, n, num_blocks), dtype=torch.uint8, device="cpu").contiguous()

    return packed_fp4, scales_fp8


def get_git_commit():
    """Get current git commit info and check for uncommitted changes."""
    result = {}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        commit_msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode("utf-8").strip()
        result["commit"] = commit
        result["commit_message"] = commit_msg

        dirty_output = subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
        if dirty_output:
            result["dirty"] = True
            result["dirty_files"] = dirty_output.splitlines()
        else:
            result["dirty"] = False
    except Exception as e:
        result["commit"] = None
        result["commit_message"] = None
        result["dirty"] = None
        result["error"] = str(e)
    return result


def get_system_info():
    """Get system information including CPU model, memory size, core count and socket count."""
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
        except Exception as e:
            cpu_model = f"Error: {e}"
    info["cpu_model"] = cpu_model

    mem_total_gb = None
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = float(line.split(":", 1)[1].split()[0])
                        mem_total_gb = round(mem_kb / (1024 * 1024), 2)
                        break
        except Exception as e:
            mem_total_gb = f"Error: {e}"
    info["memory_size_GB"] = mem_total_gb

    info["cpu_core_count"] = os.cpu_count()

    sockets = set()
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "physical id" in line:
                        sockets.add(line.split(":", 1)[1].strip())
        except Exception as e:
            sockets = set()
    info["cpu_socket_count"] = len(sockets) if len(sockets) > 0 else 1

    return info


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.splitext(os.path.basename(script_path))[0]
json_path = os.path.join(script_dir, script_name + ".jsonl")


def record_results(result, filename=json_path):
    """Append results to a JSONL file."""
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def bench_moe_nvfp4():
    """Benchmark NVFP4 MoE operator."""
    with torch.inference_mode():
        print("=" * 70)
        print("NVFP4 MoE Kernel Performance Benchmark")
        print("=" * 70)

        # Generate NVFP4 weights directly (no quantization from fp32)
        print("\nGenerating NVFP4 weights directly...")
        torch.manual_seed(42)
        gate_packed, gate_scales = generate_nvfp4_weights_direct(
            (expert_num, intermediate_size, hidden_size), block_size
        )
        up_packed, up_scales = generate_nvfp4_weights_direct((expert_num, intermediate_size, hidden_size), block_size)
        down_packed, down_scales = generate_nvfp4_weights_direct(
            (expert_num, hidden_size, intermediate_size), block_size
        )

        # Build MoE layers
        print("Building NVFP4 MoE layers...")
        moes = []
        for _ in tqdm(range(layer_num), desc="Initializing MOEs"):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.quant_config.bits = 4
            config.quant_config.group_size = block_size
            config.quant_config.zero_point = False

            config.gate_proj = gate_packed.data_ptr()
            config.up_proj = up_packed.data_ptr()
            config.down_proj = down_packed.data_ptr()
            config.gate_scale = gate_scales.data_ptr()
            config.up_scale = up_scales.data_ptr()
            config.down_scale = down_scales.data_ptr()
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.NVFP4_MOE(config)
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
        bytes_per_elem = 0.5  # 4-bit = 0.5 bytes per element

        # FLOPS calculation
        flops_per_expert = (
            2 * intermediate_size * hidden_size  # gate
            + 2 * intermediate_size * hidden_size  # up
            + 2 * hidden_size * intermediate_size  # down
        )
        total_flops = qlen * num_experts_per_tok * flops_per_expert * test_iter
        tflops = total_flops / total_time / 1e12

        # Bandwidth calculation
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
        )  # GB/s

        # Print results
        print("\n" + "=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print(f"Quant mode: NVFP4 (E2M1 + FP8 E4M3 scales, block_size={block_size})")
        print(f"Total time: {total_time:.4f} s")
        print(f"Iterations: {test_iter}")
        print(f"Time per iteration: {time_per_iter_us:.2f} us")
        print(f"Bandwidth: {bandwidth:.2f} GB/s")
        print(f"TFLOPS: {tflops:.4f}")
        print("")

        # Record results
        result = {
            "test_name": os.path.basename(__file__),
            "quant_mode": "nvfp4",
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
                "block_size": block_size,
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

        return result


if __name__ == "__main__":
    bench_moe_nvfp4()
