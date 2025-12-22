#!/usr/bin/env python
# coding=utf-8
"""
Benchmark write_weight_scale_to_buffer for AMX_FP8_MOE_TP (FP8 weights + float32 scales).

Uses two MOE instances that alternate writing to simulate realistic multi-layer scenarios.
"""
import json
import os
import platform
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from kt_kernel import kt_kernel_ext
from kt_kernel_ext.moe import AMXFP8_MOE
import torch

# Benchmark parameters
expert_num = 256
num_experts_per_tok = 8
gpu_tp_count = 2

warm_up_iter = 3
test_iter = 7

gpu_experts_num = expert_num

hidden_size = 7168
intermediate_size = 2048
group_size = 128  # FP8 uses 128x128 block-wise scales
max_len = 1

physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
CPUInfer = kt_kernel_ext.CPUInfer(80)


def get_git_commit():
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
        result["error"] = str(e)
    return result


def get_system_info():
    info = {}
    info["system_name"] = platform.uname().system
    info["node_name"] = platform.uname().node
    info["cpu_core_count"] = os.cpu_count()
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    mem_kb = float(line.split(":", 1)[1].split()[0])
                    info["memory_size_GB"] = round(mem_kb / (1024 * 1024), 2)
                    break
    return info


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
script_name = os.path.splitext(os.path.basename(script_path))[0]
json_path = os.path.join(script_dir, script_name + ".jsonl")


def record_results(result, filename=json_path):
    with open(filename, "a") as f:
        f.write(json.dumps(result) + "\n")


def allocate_weights():
    per_mat_weight_bytes = hidden_size * intermediate_size
    n_blocks_n_gate_up = (intermediate_size + group_size - 1) // group_size
    n_blocks_k = (hidden_size + group_size - 1) // group_size
    per_mat_scale_elems_gate_up = n_blocks_n_gate_up * n_blocks_k
    per_mat_scale_elems_down = n_blocks_k * n_blocks_n_gate_up

    gate_q = (
        torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8, device="cuda")
        .to("cpu")
        .contiguous()
    )
    up_q = (
        torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8, device="cuda")
        .to("cpu")
        .contiguous()
    )
    down_q = (
        torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8, device="cuda")
        .to("cpu")
        .contiguous()
    )
    gate_scale = (
        torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32, device="cuda").to("cpu").contiguous()
    )
    up_scale = (
        torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32, device="cuda").to("cpu").contiguous()
    )
    down_scale = (
        torch.randn(expert_num * per_mat_scale_elems_down, dtype=torch.float32, device="cuda").to("cpu").contiguous()
    )

    return (
        gate_q,
        up_q,
        down_q,
        gate_scale,
        up_scale,
        down_scale,
        per_mat_weight_bytes,
        per_mat_scale_elems_gate_up,
        per_mat_scale_elems_down,
    )


def build_moe(layer_idx=0):
    """Build a single MOE instance with the given layer_idx."""
    (
        gate_q,
        up_q,
        down_q,
        gate_scale,
        up_scale,
        down_scale,
        per_mat_weight_bytes,
        per_mat_scale_elems_gate_up,
        per_mat_scale_elems_down,
    ) = allocate_weights()

    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.max_len = max_len
    config.layer_idx = layer_idx
    config.quant_config.bits = 8
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    config.pool = CPUInfer.backend_
    config.gate_proj = gate_q.data_ptr()
    config.up_proj = up_q.data_ptr()
    config.down_proj = down_q.data_ptr()
    config.gate_scale = gate_scale.data_ptr()
    config.up_scale = up_scale.data_ptr()
    config.down_scale = down_scale.data_ptr()

    moe = AMXFP8_MOE(config)
    CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    CPUInfer.sync()

    keep_tensors = {
        "gate_q": gate_q,
        "up_q": up_q,
        "down_q": down_q,
        "gate_scale": gate_scale,
        "up_scale": up_scale,
        "down_scale": down_scale,
    }

    buffer_shapes = {
        "per_mat_weight_bytes": per_mat_weight_bytes,
        "per_mat_scale_elems_gate_up": per_mat_scale_elems_gate_up,
        "per_mat_scale_elems_down": per_mat_scale_elems_down,
    }

    return moe, buffer_shapes, keep_tensors


def allocate_buffers(buffer_shapes):
    """Allocate shared output buffers for single expert."""
    per_mat_weight_bytes = buffer_shapes["per_mat_weight_bytes"]
    per_mat_scale_elems_gate_up = buffer_shapes["per_mat_scale_elems_gate_up"]
    per_mat_scale_elems_down = buffer_shapes["per_mat_scale_elems_down"]

    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    scale_elems_per_expert_per_tp_gate_up = per_mat_scale_elems_gate_up // gpu_tp_count
    scale_elems_per_expert_per_tp_down = per_mat_scale_elems_down // gpu_tp_count

    # Each buffer stores data for a single expert
    w13_weight_bufs = [torch.empty(2 * weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w13_scale_bufs = [
        torch.empty(2 * scale_elems_per_expert_per_tp_gate_up, dtype=torch.float32) for _ in range(gpu_tp_count)
    ]
    w2_weight_bufs = [torch.empty(weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w2_scale_bufs = [torch.empty(scale_elems_per_expert_per_tp_down, dtype=torch.float32) for _ in range(gpu_tp_count)]

    buffer_ptrs = {
        "w13_weight_ptrs": [buf.data_ptr() for buf in w13_weight_bufs],
        "w13_scale_ptrs": [buf.data_ptr() for buf in w13_scale_bufs],
        "w2_weight_ptrs": [buf.data_ptr() for buf in w2_weight_bufs],
        "w2_scale_ptrs": [buf.data_ptr() for buf in w2_scale_bufs],
    }

    keep_tensors = {
        "w13_weight_bufs": w13_weight_bufs,
        "w13_scale_bufs": w13_scale_bufs,
        "w2_weight_bufs": w2_weight_bufs,
        "w2_scale_bufs": w2_scale_bufs,
    }

    return buffer_ptrs, keep_tensors


def bench_write_buffer():
    # Build two MOE instances with different layer_idx
    moe_0, buffer_shapes, keep_tensors_0 = build_moe(layer_idx=0)
    moe_1, _, keep_tensors_1 = build_moe(layer_idx=1)
    moes = [moe_0, moe_1]

    # Allocate shared buffers
    buffer_ptrs, buffer_keep_tensors = allocate_buffers(buffer_shapes)

    total_weights = hidden_size * intermediate_size * expert_num * 3
    total_scale_bytes = (
        (buffer_shapes["per_mat_scale_elems_gate_up"] * 2 + buffer_shapes["per_mat_scale_elems_down"]) * expert_num * 4
    )
    bytes_per_call = total_weights + total_scale_bytes

    # Warm-up: alternate between two MOEs
    for _ in tqdm(range(warm_up_iter), desc="Warm-up"):
        for moe_idx, moe in enumerate(moes):
            for expert_id in range(gpu_experts_num):
                CPUInfer.submit(
                    moe.write_weight_scale_to_buffer_task(gpu_tp_count=gpu_tp_count, expert_id=expert_id, **buffer_ptrs)
                )
                CPUInfer.sync()

    total_time = 0
    for iter_idx in tqdm(range(test_iter), desc="Testing"):
        start = time.perf_counter()
        # Alternate between two MOEs
        for moe_idx, moe in enumerate(moes):
            for expert_id in range(gpu_experts_num):
                CPUInfer.submit(
                    moe.write_weight_scale_to_buffer_task(gpu_tp_count=gpu_tp_count, expert_id=expert_id, **buffer_ptrs)
                )
                CPUInfer.sync()
        end = time.perf_counter()
        iter_time = end - start
        total_time += iter_time
        print(f"Iter {iter_idx}: {iter_time*1000:.2f} ms")
        time.sleep(0.3)

    # bytes_per_call is for one MOE, we have 2 MOEs
    bytes_per_iter = bytes_per_call * 2
    time_per_iter_ms = total_time / test_iter * 1000
    bandwidth_gbs = bytes_per_iter * test_iter / total_time / 1e9

    print(f"\n{'='*60}")
    print("FP8 write_weight_scale_to_buffer benchmark (2 MOEs alternating)")
    print(f"{'='*60}")
    print(f"Time per iteration: {time_per_iter_ms:.2f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    print(f"Experts per MOE: {gpu_experts_num}, MOEs: 2")
    print(f"Time per expert: {time_per_iter_ms/(gpu_experts_num*2)*1000:.2f} us")

    result = {
        "op": "write_weight_scale_to_buffer_fp8",
        "time_per_iteration_ms": time_per_iter_ms,
        "bandwidth_GBs": bandwidth_gbs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_parameters": {
            "expert_num": expert_num,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "group_size": group_size,
            "gpu_tp_count": gpu_tp_count,
            "bytes_per_iter": bytes_per_iter,
            "num_moes": 2,
        },
    }
    result.update(get_git_commit())
    result.update(get_system_info())
    record_results(result)


if __name__ == "__main__":
    bench_write_buffer()
