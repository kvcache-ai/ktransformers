#!/usr/bin/env python
# coding=utf-8
"""
Benchmark write_weight_scale_to_buffer for AMX_K2_MOE_TP (int4 packed weights + bf16 scales).
"""
import json
import os
import platform
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

import kt_kernel_ext
import torch

# Benchmark parameters (single MoE, mirror examples/test_k2_write_buffer.py)
expert_num = 384
num_experts_per_tok = expert_num
gpu_tp_count = 4

warm_up_iter = 3
test_iter = 7

gpu_experts_num = expert_num

hidden_size = 7168
intermediate_size = 2048
group_size = 32
max_len = 1

physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
CPUInfer = kt_kernel_ext.CPUInfer(96)


def get_git_commit():
    result = {}
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        )
        commit_msg = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%B"])
            .decode("utf-8")
            .strip()
        )
        result["commit"] = commit
        result["commit_message"] = commit_msg

        dirty_output = (
            subprocess.check_output(["git", "status", "--porcelain"]).decode("utf-8").strip()
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


def allocate_weights():
    per_mat_weight_bytes = (hidden_size * intermediate_size) // 2
    per_mat_scale_elems = (hidden_size * intermediate_size) // group_size

    gate_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    up_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    down_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)

    gate_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)
    up_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)
    down_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)

    return (
        gate_q.contiguous(),
        up_q.contiguous(),
        down_q.contiguous(),
        gate_scale.contiguous(),
        up_scale.contiguous(),
        down_scale.contiguous(),
        per_mat_weight_bytes,
        per_mat_scale_elems,
    )


def build_moe():
    (
        gate_q,
        up_q,
        down_q,
        gate_scale,
        up_scale,
        down_scale,
        per_mat_weight_bytes,
        per_mat_scale_elems,
    ) = allocate_weights()

    config = kt_kernel_ext.moe.MOEConfig(
        expert_num, num_experts_per_tok, hidden_size, intermediate_size
    )
    config.max_len = max_len
    config.quant_config.bits = 4
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    config.pool = CPUInfer.backend_

    config.gate_proj = gate_q.data_ptr()
    config.up_proj = up_q.data_ptr()
    config.down_proj = down_q.data_ptr()
    config.gate_scale = gate_scale.data_ptr()
    config.up_scale = up_scale.data_ptr()
    config.down_scale = down_scale.data_ptr()

    moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(config)
    CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    CPUInfer.sync()

    # Buffer sizing per TP
    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    scale_elems_per_expert_per_tp = per_mat_scale_elems // gpu_tp_count
    total_weight_bytes_per_tp = gpu_experts_num * weight_bytes_per_expert_per_tp
    total_scale_elems_per_tp = gpu_experts_num * scale_elems_per_expert_per_tp

    w13_weight_bufs = [
        torch.empty(2 * total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)
    ]
    w13_scale_bufs = [
        torch.empty(2 * total_scale_elems_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)
    ]
    w2_weight_bufs = [
        torch.empty(total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)
    ]
    w2_scale_bufs = [
        torch.empty(total_scale_elems_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)
    ]

    buffer_ptrs = {
        "w13_weight_ptrs": [buf.data_ptr() for buf in w13_weight_bufs],
        "w13_scale_ptrs": [buf.data_ptr() for buf in w13_scale_bufs],
        "w2_weight_ptrs": [buf.data_ptr() for buf in w2_weight_bufs],
        "w2_scale_ptrs": [buf.data_ptr() for buf in w2_scale_bufs],
    }

    buffer_shapes = {
        "per_mat_weight_bytes": per_mat_weight_bytes,
        "per_mat_scale_elems": per_mat_scale_elems,
        "weight_bytes_per_expert_per_tp": weight_bytes_per_expert_per_tp,
        "scale_elems_per_expert_per_tp": scale_elems_per_expert_per_tp,
        "total_weight_bytes_per_tp": total_weight_bytes_per_tp,
        "total_scale_elems_per_tp": total_scale_elems_per_tp,
    }

    keep_tensors = {
        "gate_q": gate_q,
        "up_q": up_q,
        "down_q": down_q,
        "gate_scale": gate_scale,
        "up_scale": up_scale,
        "down_scale": down_scale,
        "w13_weight_bufs": w13_weight_bufs,
        "w13_scale_bufs": w13_scale_bufs,
        "w2_weight_bufs": w2_weight_bufs,
        "w2_scale_bufs": w2_scale_bufs,
    }

    return moe, buffer_ptrs, buffer_shapes, keep_tensors


def bench_write_buffer():
    moe, buffer_ptrs, buffer_shapes, keep_tensors = build_moe()

    total_weights = hidden_size * intermediate_size * expert_num * 3
    # Throughput accounting consistent with examples/test_k2_write_buffer.py
    bytes_per_call = total_weights // group_size + total_weights // 2

    # Warm-up
    for _ in tqdm(range(warm_up_iter), desc="Warm-up"):
        CPUInfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                gpu_experts_num=gpu_experts_num,
                **buffer_ptrs,
            )
        )
        CPUInfer.sync()
    
    total_time = 0
    for _ in tqdm(range(test_iter), desc="Testing"):
        start = time.perf_counter()
        CPUInfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                gpu_experts_num=gpu_experts_num,
                **buffer_ptrs,
            )
        )
        CPUInfer.sync()
        end = time.perf_counter()
        total_time += end - start
        time.sleep(0.6)
        print(end - start)


    
    time_per_iter_us = total_time / test_iter * 1e6
    bandwidth_gbs = bytes_per_call * test_iter / total_time / 1e9

    print("write_weight_scale_to_buffer benchmark")
    print("Time(s): ", total_time)
    print("Iteration: ", test_iter)
    print("Time(us) per iteration: ", time_per_iter_us)
    print("Bandwidth: ", bandwidth_gbs, "GB/s")
    print("")

    result = {
        "op": "write_weight_scale_to_buffer",
        "total_time_seconds": total_time,
        "iterations": test_iter,
        "time_per_iteration_us": time_per_iter_us,
        "bandwidth_GBs": bandwidth_gbs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "test_parameters": {
            "expert_num": expert_num,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "group_size": group_size,
            "max_len": max_len,
            "num_experts_per_tok": num_experts_per_tok,
            "gpu_tp_count": gpu_tp_count,
            "gpu_experts_num": gpu_experts_num,
            "warm_up_iter": warm_up_iter,
            "test_iter": test_iter,
            "bytes_per_call": bytes_per_call,
        },
        "buffer_shapes": buffer_shapes,
        "keep_tensors_alive": list(keep_tensors.keys()),
    }
    result.update(get_git_commit())
    result.update(get_system_info())
    record_results(result)


if __name__ == "__main__":
    bench_write_buffer()
