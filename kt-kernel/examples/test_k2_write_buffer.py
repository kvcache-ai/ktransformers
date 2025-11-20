import os
import sys
import time

import torch
import numpy as np


# Ensure we can import the local extension
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

import kt_kernel_ext
from kt_kernel_ext import CPUInfer


def make_cpu_infer(thread_num=80):
    return CPUInfer(thread_num)


def build_config(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size, group_size):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    cfg.max_len = 1
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False
    cfg.pool = cpuinfer.backend_
    return cfg


def allocate_weights(expert_num, hidden_size, intermediate_size, group_size):
    # packed int4 weights: 2 values per byte
    per_mat_weight_bytes = (hidden_size * intermediate_size) // 2
    per_mat_scale_elems = (hidden_size * intermediate_size) // group_size

    gate_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    up_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    down_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)

    gate_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)
    up_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)
    down_scale = torch.randn(expert_num * per_mat_scale_elems, dtype=torch.bfloat16)

    return (
        gate_q,
        up_q,
        down_q,
        gate_scale,
        up_scale,
        down_scale,
        per_mat_weight_bytes,
        per_mat_scale_elems,
    )


def main():
    torch.manual_seed(123)

    expert_num = 384  # Total experts
    gpu_experts = 384  # Number of experts on GPU
    num_experts_per_tok = 8
    hidden_size = 7168
    intermediate_size = 2048
    group_size = 32

    cpuinfer = make_cpu_infer()
    cfg = build_config(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size, group_size)

    (
        gate_q,
        up_q,
        down_q,
        gate_scale,
        up_scale,
        down_scale,
        per_mat_weight_bytes,
        per_mat_scale_elems,
    ) = allocate_weights(expert_num, hidden_size, intermediate_size, group_size)

    cfg.gate_proj = gate_q.data_ptr()
    cfg.up_proj = up_q.data_ptr()
    cfg.down_proj = down_q.data_ptr()
    cfg.gate_scale = gate_scale.data_ptr()
    cfg.up_scale = up_scale.data_ptr()
    cfg.down_scale = down_scale.data_ptr()

    moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(cfg)

    physical_to_logical_map = (
        torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
    )
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    # Allocate buffers for GPU experts (first gpu_experts experts)
    total_weight_bytes = gpu_experts * per_mat_weight_bytes
    total_scale_elems = gpu_experts * per_mat_scale_elems

    gate_weight_buf = torch.empty(total_weight_bytes, dtype=torch.uint8)
    gate_scale_buf = torch.empty(total_scale_elems, dtype=torch.bfloat16)
    up_weight_buf = torch.empty(total_weight_bytes, dtype=torch.uint8)
    up_scale_buf = torch.empty(total_scale_elems, dtype=torch.bfloat16)
    down_weight_buf = torch.empty(total_weight_bytes, dtype=torch.uint8)
    down_scale_buf = torch.empty(total_scale_elems, dtype=torch.bfloat16)

    for i in range(3):
        cpuinfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_experts=gpu_experts,
                gate_weight_ptr=gate_weight_buf.data_ptr(),
                gate_scale_ptr=gate_scale_buf.data_ptr(),
                up_weight_ptr=up_weight_buf.data_ptr(),
                up_scale_ptr=up_scale_buf.data_ptr(),
                down_weight_ptr=down_weight_buf.data_ptr(),
                down_scale_ptr=down_scale_buf.data_ptr(),
            )
        )
        cpuinfer.sync()

    print(f"Total experts: {expert_num}, GPU experts: {gpu_experts}")
    print(f"Per matrix weight bytes: {per_mat_weight_bytes}")
    print(f"Per matrix scale elements: {per_mat_scale_elems}")
    print(f"Total weight bytes per matrix: {total_weight_bytes}")
    print(f"Total scale elements per matrix: {total_scale_elems}")

    begin_time = time.perf_counter_ns()
    cpuinfer.submit(
        moe.write_weight_scale_to_buffer_task(
            gpu_experts=gpu_experts,
            gate_weight_ptr=gate_weight_buf.data_ptr(),
            gate_scale_ptr=gate_scale_buf.data_ptr(),
            up_weight_ptr=up_weight_buf.data_ptr(),
            up_scale_ptr=up_scale_buf.data_ptr(),
            down_weight_ptr=down_weight_buf.data_ptr(),
            down_scale_ptr=down_scale_buf.data_ptr(),
        )
    )
    cpuinfer.sync()
    end_time = time.perf_counter_ns()
    elapsed_ns = (end_time - begin_time)
    print(f"write_weight_scale_to_buffer time: {elapsed_ns:.2f} ns")

    def split_tensor(tensor, chunk):
        return [tensor[i * chunk : (i + 1) * chunk] for i in range(expert_num)]

    gate_q_slices = split_tensor(gate_q, per_mat_weight_bytes)
    up_q_slices = split_tensor(up_q, per_mat_weight_bytes)
    down_q_slices = split_tensor(down_q, per_mat_weight_bytes)

    gate_s_slices = split_tensor(gate_scale, per_mat_scale_elems)
    up_s_slices = split_tensor(up_scale, per_mat_scale_elems)
    down_s_slices = split_tensor(down_scale, per_mat_scale_elems)

    # Get GPU experts (first gpu_experts experts)
    gpu_expert_indices = list(range(gpu_experts))

    expected_gate_weight = torch.cat([gate_q_slices[e] for e in gpu_expert_indices])
    expected_gate_scale = torch.cat([gate_s_slices[e] for e in gpu_expert_indices])
    expected_up_weight = torch.cat([up_q_slices[e] for e in gpu_expert_indices])
    expected_up_scale = torch.cat([up_s_slices[e] for e in gpu_expert_indices])
    expected_down_weight = torch.cat([down_q_slices[e] for e in gpu_expert_indices])
    expected_down_scale = torch.cat([down_s_slices[e] for e in gpu_expert_indices])

    print(f"Gate weight buf size: {gate_weight_buf.shape}, expected size: {expected_gate_weight.shape}")
    print(f"Gate scale buf size: {gate_scale_buf.shape}, expected size: {expected_gate_scale.shape}")

    # Check first few bytes for debugging
    print(f"First 10 bytes of gate_weight_buf: {gate_weight_buf[:10].tolist()}")
    print(f"First 10 bytes of expected_gate_weight: {expected_gate_weight[:10].tolist()}")

    # Try to see if data is there but shifted
    if not torch.equal(gate_weight_buf, expected_gate_weight):
        print("Gate weight mismatch detected. Checking if all zeros...")
        if torch.all(gate_weight_buf == 0):
            print("ERROR: gate_weight_buf is all zeros!")
        else:
            print(f"gate_weight_buf has data, but doesn't match expected")

    assert torch.equal(gate_weight_buf, expected_gate_weight), "Gate weight bytes mismatch"
    assert torch.allclose(gate_scale_buf, expected_gate_scale), "Gate scale values mismatch"
    assert torch.equal(up_weight_buf, expected_up_weight), "Up weight bytes mismatch"
    assert torch.allclose(up_scale_buf, expected_up_scale), "Up scale values mismatch"
    assert torch.equal(down_weight_buf, expected_down_weight), "Down weight bytes mismatch"
    assert torch.allclose(down_scale_buf, expected_down_scale), "Down scale values mismatch"

    print(f"write_weight_scale_to_buffer passed: extracted {gpu_experts} GPU experts from total {expert_num} experts")


if __name__ == "__main__":
    main()
