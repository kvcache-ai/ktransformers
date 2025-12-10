#!/usr/bin/env python
# coding=utf-8
"""
Benchmark AMX_K2_MOE_TP int4 path with packed weights and BF16 scales.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import kt_kernel_ext
import torch

# Benchmark parameters (single MoE, no layer loop)
expert_num = 384
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
qlen = 1
warm_up_iter = 1000
test_iter = 5000
k_group_size = 32

physical_to_logical_map = (
    torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()
)

worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = 2
worker_config.subpool_numa_map = [0, 1]
worker_config.subpool_thread_count = [40, 40]
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)


def get_git_commit():
    result = {}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        commit_msg = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%B"])
            .decode("utf-8")
            .strip()
        )
        result["commit"] = commit
        result["commit_message"] = commit_msg

        dirty_output = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
        )
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
        except Exception:
            sockets = set()
    info["cpu_socket_count"] = len(sockets) if len(sockets) > 0 else 1

    return info


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.splitext(os.path.basename(script_path))[0]
json_path = os.path.join(script_dir, script_name + ".jsonl")


def record_results(result, filename=json_path):
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def pack_to_int32(
    value: torch.Tensor, num_bits: int, packed_dim: int = 1
) -> torch.Tensor:
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be torch.int8 before packing")
    if not (1 <= num_bits <= 8):
        raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")

    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def pack_tensor_per_row(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    e, rows, cols = q.shape
    flat = q.view(e * rows, cols)
    packed = pack_to_int32(flat, num_bits)
    return packed.view(e, rows, -1).contiguous()


def quantize_k2_tensor(weights: torch.Tensor, group_size: int):
    """
    K2 int4 quantization producing int32-packed weights (8 int4s each) and BF16 scales.
    """
    weights_f32 = weights.to(torch.float32)
    e, rows, cols = weights_f32.shape
    if cols % group_size != 0 or cols % 2 != 0:
        raise ValueError(
            f"cols ({cols}) must be divisible by group_size ({group_size}) and 2"
        )

    reshaped = weights_f32.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 7.0).squeeze(-1)
    q = torch.round(reshaped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(e, rows, cols)
    packed = pack_tensor_per_row(q, num_bits=4).view(e, rows, cols // 8).contiguous()
    scales = scales.to(torch.bfloat16).contiguous().view(
        e, rows, cols // group_size
    ).contiguous()
    return packed, scales


def build_quantized_layer_weights():
    gate_proj = torch.randn(
        (expert_num, intermediate_size, hidden_size),
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
    up_proj = torch.randn(
        (expert_num, intermediate_size, hidden_size),
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
    down_proj = torch.randn(
        (expert_num, hidden_size, intermediate_size),
        dtype=torch.float32,
        device="cpu",
    ).contiguous()

    gate_q, gate_scales = quantize_k2_tensor(gate_proj, k_group_size)
    up_q, up_scales = quantize_k2_tensor(up_proj, k_group_size)
    down_q, down_scales = quantize_k2_tensor(down_proj, k_group_size)

    return {
        "gate_qweight": gate_q,
        "up_qweight": up_q,
        "down_qweight": down_q,
        "gate_scales": gate_scales,
        "up_scales": up_scales,
        "down_scales": down_scales,
    }


def bench_k2_moe():
    with torch.inference_mode():
        bytes_per_elem = 0.5 + 2.0 / k_group_size

        quant_data = build_quantized_layer_weights()
        config = kt_kernel_ext.moe.MOEConfig(
            expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
        )
        config.max_len = max_len
        config.quant_config.bits = 4
        config.quant_config.group_size = k_group_size
        config.quant_config.zero_point = False

        config.gate_proj = quant_data["gate_qweight"].data_ptr()
        config.up_proj = quant_data["up_qweight"].data_ptr()
        config.down_proj = quant_data["down_qweight"].data_ptr()

        config.gate_scale = quant_data["gate_scales"].data_ptr()
        config.up_scale = quant_data["up_scales"].data_ptr()
        config.down_scale = quant_data["down_scales"].data_ptr()
        config.pool = CPUInfer.backend_

        moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        gen_iter = 3000
        expert_ids = (
            torch.rand(gen_iter * qlen, expert_num, device="cpu")
            .argsort(dim=-1)[:, :num_experts_per_tok]
            .reshape(gen_iter, qlen * num_experts_per_tok)
            .contiguous()
        )
        weights = torch.rand(
            (gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu"
        ).contiguous()
        input_tensor = torch.randn(
            (qlen, hidden_size), dtype=torch.bfloat16, device="cpu"
        ).contiguous()
        output_tensor = torch.empty_like(input_tensor)
        bsz_tensor = torch.tensor([qlen], device="cpu")

        for i in tqdm(range(warm_up_iter), desc="Warm-up"):
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor.data_ptr(),
                    output_tensor.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

        start = time.perf_counter()
        for i in tqdm(range(test_iter), desc="Testing"):
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids[i % gen_iter].data_ptr(),
                    weights[i % gen_iter].data_ptr(),
                    input_tensor.data_ptr(),
                    output_tensor.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start

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
        )
        flops = (
            hidden_size
            * intermediate_size
            * qlen
            * 3
            * num_experts_per_tok
            * 2
            * test_iter
            / total_time
            / 1e12
        )

        print("Quant mode: int4_k2")
        print("Time(s): ", total_time)
        print("Iteration: ", test_iter)
        print("Time(us) per iteration: ", time_per_iter_us)
        print("Bandwidth: ", bandwidth, "GB/s")
        print("Flops: ", flops, "TFLOPS")
        print("")

        result = {
            "quant_mode": "int4_k2",
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
                "qlen": qlen,
                "warm_up_iter": warm_up_iter,
                "test_iter": test_iter,
                "k_group_size": k_group_size,
                "bytes_per_elem": bytes_per_elem,
            },
        }
        result.update(get_git_commit())
        result.update(get_system_info())
        record_results(result)


if __name__ == "__main__":
    bench_k2_moe()
