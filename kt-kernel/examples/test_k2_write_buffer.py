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

    expert_num = 256 # Total experts
    gpu_experts = expert_num  # Number of experts on GPU
    gpu_tp_count = 2  # Number of TP parts
    
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

    # TP configuration

    # Since weights are col-major, we can directly divide the total size by tp_count
    # Each matrix is divided into gpu_tp_count parts in memory order

    # Calculate sizes per TP part (direct division since col-major)
    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    scale_elems_per_expert_per_tp = per_mat_scale_elems // gpu_tp_count

    # Total sizes for all gpu_experts
    total_weight_bytes_per_tp = gpu_experts * weight_bytes_per_expert_per_tp
    total_scale_elems_per_tp = gpu_experts * scale_elems_per_expert_per_tp

    # Create buffer lists for w13 (gate+up) and w2 (down)
    w13_weight_bufs = []
    w13_scale_bufs = []
    w2_weight_bufs = []
    w2_scale_bufs = []

    for tp_idx in range(gpu_tp_count):
        # w13 combines gate and up, so needs 2x the size
        w13_weight_bufs.append(torch.empty(2 * total_weight_bytes_per_tp, dtype=torch.uint8))
        w13_scale_bufs.append(torch.empty(2 * total_scale_elems_per_tp, dtype=torch.bfloat16))
        w2_weight_bufs.append(torch.empty(total_weight_bytes_per_tp, dtype=torch.uint8))
        w2_scale_bufs.append(torch.empty(total_scale_elems_per_tp, dtype=torch.bfloat16))

    # Get data pointers for all buffers
    w13_weight_ptrs = [buf.data_ptr() for buf in w13_weight_bufs]
    w13_scale_ptrs = [buf.data_ptr() for buf in w13_scale_bufs]
    w2_weight_ptrs = [buf.data_ptr() for buf in w2_weight_bufs]
    w2_scale_ptrs = [buf.data_ptr() for buf in w2_scale_bufs]

    print(f"Total experts: {expert_num}, GPU experts: {gpu_experts}")
    print(f"GPU TP count: {gpu_tp_count}")
    print(f"Original per matrix weight bytes: {per_mat_weight_bytes}")
    print(f"Original per matrix scale elements: {per_mat_scale_elems}")
    print(f"Weight bytes per expert per TP: {weight_bytes_per_expert_per_tp}")
    print(f"Scale elements per expert per TP: {scale_elems_per_expert_per_tp}")
    print(f"Total weight bytes per TP (w13): {2 * total_weight_bytes_per_tp}")
    print(f"Total weight bytes per TP (w2): {total_weight_bytes_per_tp}")
    print(f"Total scale elements per TP (w13): {2 * total_scale_elems_per_tp}")
    print(f"Total scale elements per TP (w2): {total_scale_elems_per_tp}")

    for i in range(5):
        cpuinfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                gpu_experts_num=gpu_experts,
                w13_weight_ptrs=w13_weight_ptrs,
                w13_scale_ptrs=w13_scale_ptrs,
                w2_weight_ptrs=w2_weight_ptrs,
                w2_scale_ptrs=w2_scale_ptrs,
            )
        )
        cpuinfer.sync()

    begin_time = time.perf_counter_ns()
    cpuinfer.submit(
        moe.write_weight_scale_to_buffer_task(
            gpu_tp_count=gpu_tp_count,
            gpu_experts_num=gpu_experts,
            w13_weight_ptrs=w13_weight_ptrs,
            w13_scale_ptrs=w13_scale_ptrs,
            w2_weight_ptrs=w2_weight_ptrs,
            w2_scale_ptrs=w2_scale_ptrs,
        )
    )
    cpuinfer.sync()
    end_time = time.perf_counter_ns()
    elapsed_ms = (end_time - begin_time) / 1000000
    total_weights = hidden_size * intermediate_size * expert_num * 3
    total_bytes = total_weights // group_size + total_weights // 2
    print(f"write_weight_scale_to_buffer time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {total_bytes / (elapsed_ms * 1e6):.2f} GB/s")
    def split_expert_tensor(tensor, chunk):
        """Split tensor by experts"""
        return [tensor[i * chunk : (i + 1) * chunk] for i in range(expert_num)]

    # Split by experts first
    gate_q_experts = split_expert_tensor(gate_q, per_mat_weight_bytes)
    up_q_experts = split_expert_tensor(up_q, per_mat_weight_bytes)
    down_q_experts = split_expert_tensor(down_q, per_mat_weight_bytes)

    gate_scale_experts = split_expert_tensor(gate_scale, per_mat_scale_elems)
    up_scale_experts = split_expert_tensor(up_scale, per_mat_scale_elems)
    down_scale_experts = split_expert_tensor(down_scale, per_mat_scale_elems)

    # CPU TP count is always 2 in this test setup (one TP per NUMA node)
    cpu_tp_count = 2

    # Verify buffers for each TP part
    for tp_idx in range(gpu_tp_count):
        expected_w13_weights = []
        expected_w13_scales = []
        expected_w2_weights = []
        expected_w2_scales = []

        weight13_per_tp = per_mat_weight_bytes // gpu_tp_count
        scale13_per_tp = per_mat_scale_elems // gpu_tp_count
        # Process each GPU expert
        for expert_idx in range(gpu_experts):
            # For w13 (gate and up), the slicing is straightforward

            start_weight = tp_idx * weight13_per_tp
            end_weight = (tp_idx + 1) * weight13_per_tp
            start_scale = tp_idx * scale13_per_tp
            end_scale = (tp_idx + 1) * scale13_per_tp

            # Gate
            gate_weight_tp = gate_q_experts[expert_idx][start_weight:end_weight]
            gate_scale_tp = gate_scale_experts[expert_idx][start_scale:end_scale]

            # Up
            up_weight_tp = up_q_experts[expert_idx][start_weight:end_weight]
            up_scale_tp = up_scale_experts[expert_idx][start_scale:end_scale]

            # Down matrix needs special handling because it's sliced column-wise
            # We need to reconstruct it from column slices
            down_weight_tp_parts = []
            down_scale_tp_parts = []

            # Iterate through each column to extract the corresponding parts
            for col_idx in range(hidden_size):
                col_weight_start = col_idx * (intermediate_size // 2)
                col_scale_start = col_idx * (intermediate_size // group_size)

                # Direct mapping: each CPU TP corresponds to a GPU TP
                tp_slice_weight_size = (intermediate_size // gpu_tp_count) // 2
                tp_slice_scale_size = (intermediate_size // gpu_tp_count) // group_size

                tp_weight_offset = col_weight_start + tp_idx * tp_slice_weight_size
                tp_scale_offset = col_scale_start + tp_idx * tp_slice_scale_size

                down_weight_tp_parts.append(
                    down_q_experts[expert_idx][tp_weight_offset:tp_weight_offset + tp_slice_weight_size]
                )
                down_scale_tp_parts.append(
                    down_scale_experts[expert_idx][tp_scale_offset:tp_scale_offset + tp_slice_scale_size]
                )

            # Concatenate all column slices for this TP
            down_weight_tp = torch.cat(down_weight_tp_parts)
            down_scale_tp = torch.cat(down_scale_tp_parts)

            expected_w13_weights.append(gate_weight_tp)
            expected_w13_weights.append(up_weight_tp)
            expected_w13_scales.append(gate_scale_tp)
            expected_w13_scales.append(up_scale_tp)
            expected_w2_weights.append(down_weight_tp)
            expected_w2_scales.append(down_scale_tp)

        # Concatenate all experts for this TP part
        expected_w13_weight = torch.cat(expected_w13_weights)
        expected_w13_scale = torch.cat(expected_w13_scales)
        expected_w2_weight = torch.cat(expected_w2_weights)
        expected_w2_scale = torch.cat(expected_w2_scales)

        print(f"=== Checking TP part {tp_idx} ===")

        # Assert all checks pass
        assert torch.equal(w13_weight_bufs[tp_idx], expected_w13_weight), f"w13 weight bytes mismatch for TP {tp_idx}"
        assert torch.allclose(w13_scale_bufs[tp_idx], expected_w13_scale), f"w13 scale values mismatch for TP {tp_idx}"
        assert torch.equal(w2_weight_bufs[tp_idx], expected_w2_weight), f"w2 weight bytes mismatch for TP {tp_idx}"
        assert torch.allclose(w2_scale_bufs[tp_idx], expected_w2_scale), f"w2 scale values mismatch for TP {tp_idx}"

    print(f"\n✓ write_weight_scale_to_buffer passed: extracted {gpu_experts} GPU experts across {gpu_tp_count} TP parts from total {expert_num} experts")


if __name__ == "__main__":
    main()
