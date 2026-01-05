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
import numpy as np
from kt_kernel import kt_kernel_ext

# Test parameters
expert_num = 16
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
layer_num = 5
block_size = 16  # NVFP4 block size

qlen = 1
warm_up_iter = 1000
test_iter = 10000
physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

# CPUInfer configuration
worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = 2
worker_config.subpool_numa_map = [0, 1]
worker_config.subpool_thread_count = [32, 32]
CPUINFER_PARAM = 64
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)

# NVFP4 E2M1 lookup table (on GPU)
E2M1_VALUES_GPU = None


def get_e2m1_lut(device):
    """Get E2M1 lookup table on specified device."""
    global E2M1_VALUES_GPU
    if E2M1_VALUES_GPU is None or E2M1_VALUES_GPU.device != device:
        E2M1_VALUES_GPU = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=device)
    return E2M1_VALUES_GPU


def float_to_e2m1_gpu(vals: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated conversion of floats to E2M1 4-bit representation."""
    device = vals.device
    e2m1_lut = get_e2m1_lut(device)

    signs = (vals < 0).to(torch.uint8)
    abs_vals = torch.abs(vals)

    # Find closest E2M1 value using broadcasting
    diffs = torch.abs(abs_vals.unsqueeze(-1) - e2m1_lut)
    best_idx = torch.argmin(diffs, dim=-1).to(torch.uint8)

    return (signs << 3) | best_idx


def float_to_fp8_e4m3_gpu(vals: torch.Tensor) -> torch.Tensor:
    """GPU-accelerated conversion of floats to FP8 E4M3 representation."""
    device = vals.device

    # Handle zero
    nonzero_mask = vals != 0

    signs = (vals < 0).to(torch.uint8)
    abs_vals = torch.abs(vals)

    # Clamp values
    abs_vals = torch.clamp(abs_vals, 1.0 / 512, 448)

    # Calculate exponent and mantissa
    exp = torch.floor(torch.log2(abs_vals)).to(torch.int32)
    exp_biased = exp + 7
    exp_biased = torch.clamp(exp_biased, 0, 15)

    # Calculate mantissa
    mantissa = torch.round((abs_vals / (2.0**exp) - 1.0) * 8).to(torch.int32)
    mantissa = torch.clamp(mantissa, 0, 7)

    # Handle overflow case
    overflow_mask = exp_biased >= 15
    mantissa = torch.where(overflow_mask, torch.tensor(7, device=device, dtype=torch.int32), mantissa)

    result = (signs << 7) | (exp_biased.to(torch.uint8) << 3) | mantissa.to(torch.uint8)
    result = torch.where(nonzero_mask, result, torch.tensor(0, device=device, dtype=torch.uint8))

    return result


def quantize_to_nvfp4_gpu(weights: torch.Tensor, block_size: int = 16) -> tuple:
    """
    Quantize FP32/BF16 weights to NVFP4 format using GPU.

    Args:
        weights: [N, K] weight tensor (on GPU)
        block_size: Block size for quantization (must be 16 for NVFP4)

    Returns:
        packed_fp4: [N, K // 2] packed FP4 data (numpy, uint8)
        scales_fp8: [N, K // 16] FP8 E4M3 block scales (numpy, uint8)
    """
    assert block_size == 16, "NVFP4 requires block_size=16"

    n, k = weights.shape
    assert k % block_size == 0, f"K ({k}) must be divisible by block_size ({block_size})"

    device = weights.device
    weights_f32 = weights.float()
    num_blocks = k // block_size

    # Reshape to [N, num_blocks, block_size]
    weights_blocked = weights_f32.reshape(n, num_blocks, block_size)

    # Compute block scales: max(abs(block)) / 6.0
    block_max = torch.abs(weights_blocked).amax(dim=2)  # [N, num_blocks]
    block_scales = torch.where(block_max > 0, block_max / 6.0, torch.ones_like(block_max))

    # Convert scales to FP8 E4M3
    scales_fp8 = float_to_fp8_e4m3_gpu(block_scales.flatten()).reshape(n, num_blocks)

    # Scale weights by inverse block scale
    block_scales_inv = torch.where(block_scales > 1e-10, 1.0 / block_scales, torch.zeros_like(block_scales))
    scaled_weights = weights_blocked * block_scales_inv.unsqueeze(-1)  # [N, num_blocks, block_size]

    # Quantize to E2M1
    scaled_weights_flat = scaled_weights.reshape(n, -1)  # [N, K]
    quantized = float_to_e2m1_gpu(scaled_weights_flat)  # [N, K]

    # Pack two 4-bit values into one byte
    quantized_reshaped = quantized.reshape(n, k // 2, 2)
    packed_fp4 = quantized_reshaped[:, :, 0] | (quantized_reshaped[:, :, 1] << 4)

    return packed_fp4.cpu().numpy(), scales_fp8.cpu().numpy()


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
        bytes_per_elem = 0.5  # 4-bit = 0.5 bytes per element

        moes = []
        gate_packed_list = []
        gate_scales_list = []
        up_packed_list = []
        up_scales_list = []
        down_packed_list = []
        down_scales_list = []

        print("Initializing NVFP4 MoE layers...")
        for layer_index in tqdm(range(layer_num), desc="Layers"):
            # Generate random weights on GPU
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cuda")
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cuda")
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device="cuda")

            # Batch quantize all experts at once on GPU
            # Reshape [expert_num, N, K] -> [expert_num * N, K] for batch processing
            gate_flat = gate_proj.reshape(expert_num * intermediate_size, hidden_size)
            up_flat = up_proj.reshape(expert_num * intermediate_size, hidden_size)
            down_flat = down_proj.reshape(expert_num * hidden_size, intermediate_size)

            gate_packed_np, gate_scales_np = quantize_to_nvfp4_gpu(gate_flat)
            up_packed_np, up_scales_np = quantize_to_nvfp4_gpu(up_flat)
            down_packed_np, down_scales_np = quantize_to_nvfp4_gpu(down_flat)

            # Convert to torch tensors
            gate_packed = torch.from_numpy(gate_packed_np).contiguous()
            gate_scales = torch.from_numpy(gate_scales_np).contiguous()
            up_packed = torch.from_numpy(up_packed_np).contiguous()
            up_scales = torch.from_numpy(up_scales_np).contiguous()
            down_packed = torch.from_numpy(down_packed_np).contiguous()
            down_scales = torch.from_numpy(down_scales_np).contiguous()

            gate_packed_list.append(gate_packed)
            gate_scales_list.append(gate_scales)
            up_packed_list.append(up_packed)
            up_scales_list.append(up_scales)
            down_packed_list.append(down_packed)
            down_scales_list.append(down_scales)

            # Create MOE config
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = gate_packed.data_ptr()
            config.up_proj = up_packed.data_ptr()
            config.down_proj = down_packed.data_ptr()
            config.gate_scale = gate_scales.data_ptr()
            config.up_scale = up_scales.data_ptr()
            config.down_scale = down_scales.data_ptr()
            config.pool = CPUInfer.backend_

            # Quant config for NVFP4
            config.quant_config.bits = 4
            config.quant_config.group_size = block_size
            config.quant_config.zero_point = False

            # Create NVFP4 MOE
            moe = kt_kernel_ext.moe.NVFP4_MOE(config)

            # Load weights
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()

            moes.append(moe)

        print(f"Initialized {layer_num} NVFP4 MoE layers")

        # Generate test data
        gen_iter = 3000
        expert_ids = (
            torch.rand(gen_iter * qlen, expert_num, device="cpu")
            .argsort(dim=-1)[:, :num_experts_per_tok]
            .reshape(gen_iter, qlen * num_experts_per_tok)
            .to("cpu")
            .contiguous()
        )
        weights = (
            torch.rand((gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu").to("cpu").contiguous()
        )
        input_tensor = (
            torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        )
        output_tensor = (
            torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        )
        bsz_tensor = torch.tensor([qlen], device="cpu")

        # Warm-up iterations
        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor[i % layer_num].data_ptr(),
                    output_tensor[i % layer_num].data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

        # Test iterations
        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            CPUInfer.submit(
                moes[i % layer_num].forward_task(
                    bsz_tensor.data_ptr(),
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

        # Calculate performance metrics
        time_per_iter_us = total_time / test_iter * 1e6
        bandwidth = (
            hidden_size
            * intermediate_size
            * 3
            * num_experts_per_tok
            * (1 / 8 * 256 * (1 - (31 / 32) ** qlen))
            * bytes_per_elem
            * test_iter
            / total_time
            / 1e9
        )  # GB/s
        flops = (
            hidden_size * intermediate_size * qlen * 3 * num_experts_per_tok * 2 * test_iter / total_time / 1e12
        )  # TFLOPS

        print("\n" + "=" * 60)
        print("NVFP4 MoE Benchmark Results")
        print("=" * 60)
        print(f"Quant mode: nvfp4 (E2M1 + FP8 E4M3 scales)")
        print(f"Time(s): {total_time:.4f}")
        print(f"Iteration: {test_iter}")
        print(f"Time(us) per iteration: {time_per_iter_us:.2f}")
        print(f"Bandwidth: {bandwidth:.2f} GB/s")
        print(f"Flops: {flops:.4f} TFLOPS")
        print("")

        # Record results
        result = {
            "quant_mode": "nvfp4",
            "total_time_seconds": total_time,
            "iterations": test_iter,
            "time_per_iteration_us": time_per_iter_us,
            "bandwidth_GBs": bandwidth,
            "flops_TFLOPS": flops,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "test_parameters": {
                "expert_num": expert_num,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "max_len": max_len,
                "num_experts_per_tok": num_experts_per_tok,
                "layer_num": layer_num,
                "qlen": qlen,
                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "block_size": block_size,
                "CPUInfer_parameter": CPUINFER_PARAM,
            },
        }
        result.update(get_git_commit())
        result.update(get_system_info())
        record_results(result)

        return result


if __name__ == "__main__":
    bench_moe_nvfp4()
