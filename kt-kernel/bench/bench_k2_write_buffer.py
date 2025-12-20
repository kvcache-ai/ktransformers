#!/usr/bin/env python
# coding=utf-8
"""
Benchmark write_weight_scale_to_buffer for AMX_K2_MOE_TP (int4 packed weights + bf16 scales).

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
import torch

# Benchmark parameters
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
CPUInfer = kt_kernel_ext.CPUInfer(80)


def get_git_commit():
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
        per_mat_scale_elems,
    ) = allocate_weights()

    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.max_len = max_len
    config.layer_idx = layer_idx
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
        "per_mat_scale_elems": per_mat_scale_elems,
    }

    return moe, buffer_shapes, keep_tensors


def allocate_buffers(buffer_shapes):
    """Allocate shared output buffers for single expert."""
    per_mat_weight_bytes = buffer_shapes["per_mat_weight_bytes"]
    per_mat_scale_elems = buffer_shapes["per_mat_scale_elems"]

    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    scale_elems_per_expert_per_tp = per_mat_scale_elems // gpu_tp_count

    # Each buffer stores data for a single expert
    w13_weight_bufs = [torch.empty(2 * weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w13_scale_bufs = [torch.empty(2 * scale_elems_per_expert_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)]
    w2_weight_bufs = [torch.empty(weight_bytes_per_expert_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w2_scale_bufs = [torch.empty(scale_elems_per_expert_per_tp, dtype=torch.bfloat16) for _ in range(gpu_tp_count)]

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
    # Throughput accounting: scale bytes (bf16) + weight bytes (int4 packed)
    bytes_per_call = total_weights // group_size * 2 + total_weights // 2

    # Warm-up: alternate between two MOEs
    for _ in tqdm(range(warm_up_iter), desc="Warm-up"):
        for moe_idx, moe in enumerate(moes):
            for expert_id in range(gpu_experts_num):
                CPUInfer.submit(
                    moe.write_weight_scale_to_buffer_task(
                        gpu_tp_count=gpu_tp_count,
                        expert_id=expert_id,
                        **buffer_ptrs,
                    )
                )
                CPUInfer.sync()

    total_time = 0
    for iter_idx in tqdm(range(test_iter), desc="Testing"):
        start = time.perf_counter()
        # Alternate between two MOEs
        for moe_idx, moe in enumerate(moes):
            for expert_id in range(gpu_experts_num):
                CPUInfer.submit(
                    moe.write_weight_scale_to_buffer_task(
                        gpu_tp_count=gpu_tp_count,
                        expert_id=expert_id,
                        **buffer_ptrs,
                    )
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
    print("K2 write_weight_scale_to_buffer benchmark (2 MOEs alternating)")
    print(f"{'='*60}")
    print(f"Time per iteration: {time_per_iter_ms:.2f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    print(f"Experts per MOE: {gpu_experts_num}, MOEs: 2")
    print(f"Time per expert: {time_per_iter_ms/(gpu_experts_num*2)*1000:.2f} us")

    result = {
        "op": "write_weight_scale_to_buffer_k2",
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
