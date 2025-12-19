import os
import sys
import time

import torch
import numpy as np


# Ensure we can import the local extension
# REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# if REPO_ROOT not in sys.path:
#     sys.path.insert(0, REPO_ROOT)

from kt_kernel import kt_kernel_ext
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

    expert_num = 256  # Total experts
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

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
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
    # Each buffer stores data for a single expert
    w13_weight_bufs = []
    w13_scale_bufs = []
    w2_weight_bufs = []
    w2_scale_bufs = []

    for tp_idx in range(gpu_tp_count):
        # w13 combines gate and up, so needs 2x the size per expert
        w13_weight_bufs.append(torch.empty(2 * weight_bytes_per_expert_per_tp, dtype=torch.uint8))
        w13_scale_bufs.append(torch.empty(2 * scale_elems_per_expert_per_tp, dtype=torch.bfloat16))
        w2_weight_bufs.append(torch.empty(weight_bytes_per_expert_per_tp, dtype=torch.uint8))
        w2_scale_bufs.append(torch.empty(scale_elems_per_expert_per_tp, dtype=torch.bfloat16))

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
    print(f"Weight bytes per TP (w13, per expert): {2 * weight_bytes_per_expert_per_tp}")
    print(f"Weight bytes per TP (w2, per expert): {weight_bytes_per_expert_per_tp}")

    # Warm up
    for i in range(5):
        for expert_idx in range(gpu_experts):
            cpuinfer.submit(
                moe.write_weight_scale_to_buffer_task(
                    gpu_tp_count=gpu_tp_count,
                    expert_idx=expert_idx,
                    w13_weight_ptrs=w13_weight_ptrs,
                    w13_scale_ptrs=w13_scale_ptrs,
                    w2_weight_ptrs=w2_weight_ptrs,
                    w2_scale_ptrs=w2_scale_ptrs,
                )
            )
            cpuinfer.sync()

    # Timing
    begin_time = time.perf_counter_ns()
    for expert_idx in range(gpu_experts):
        cpuinfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                expert_idx=expert_idx,
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

    # Verify buffers for each TP part
    # Since we process one expert at a time, buffer contains only the last expert's data
    last_expert_idx = gpu_experts - 1

    for tp_idx in range(gpu_tp_count):
        weight13_per_tp = per_mat_weight_bytes // gpu_tp_count
        scale13_per_tp = per_mat_scale_elems // gpu_tp_count

        # For w13 (gate and up), the slicing is straightforward
        start_weight = tp_idx * weight13_per_tp
        end_weight = (tp_idx + 1) * weight13_per_tp
        start_scale = tp_idx * scale13_per_tp
        end_scale = (tp_idx + 1) * scale13_per_tp

        # Gate
        gate_weight_tp = gate_q_experts[last_expert_idx][start_weight:end_weight]
        gate_scale_tp = gate_scale_experts[last_expert_idx][start_scale:end_scale]

        # Up
        up_weight_tp = up_q_experts[last_expert_idx][start_weight:end_weight]
        up_scale_tp = up_scale_experts[last_expert_idx][start_scale:end_scale]

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
                down_q_experts[last_expert_idx][tp_weight_offset : tp_weight_offset + tp_slice_weight_size]
            )
            down_scale_tp_parts.append(
                down_scale_experts[last_expert_idx][tp_scale_offset : tp_scale_offset + tp_slice_scale_size]
            )

        # Concatenate all column slices for this TP
        down_weight_tp = torch.cat(down_weight_tp_parts)
        down_scale_tp = torch.cat(down_scale_tp_parts)

        # Expected buffer contains: gate_weight, up_weight for w13
        expected_w13_weight = torch.cat([gate_weight_tp, up_weight_tp])
        expected_w13_scale = torch.cat([gate_scale_tp, up_scale_tp])
        expected_w2_weight = down_weight_tp
        expected_w2_scale = down_scale_tp

        print(f"=== Checking TP part {tp_idx} (expert {last_expert_idx}) ===")
        print(f"  w13 weight shape: actual={w13_weight_bufs[tp_idx].shape}, expected={expected_w13_weight.shape}")
        print(f"  w13 scale shape: actual={w13_scale_bufs[tp_idx].shape}, expected={expected_w13_scale.shape}")
        print(f"  w2 weight shape: actual={w2_weight_bufs[tp_idx].shape}, expected={expected_w2_weight.shape}")
        print(f"  w2 scale shape: actual={w2_scale_bufs[tp_idx].shape}, expected={expected_w2_scale.shape}")

        # Assert all checks pass
        if not torch.equal(w13_weight_bufs[tp_idx], expected_w13_weight):
            diff_mask = w13_weight_bufs[tp_idx] != expected_w13_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            print(f"  w13 weight mismatch at index {first_diff_idx}")
            print(f"    actual: {w13_weight_bufs[tp_idx][first_diff_idx:first_diff_idx+10]}")
            print(f"    expected: {expected_w13_weight[first_diff_idx:first_diff_idx+10]}")
            raise AssertionError(f"w13 weight bytes mismatch for TP {tp_idx}")

        if not torch.allclose(w13_scale_bufs[tp_idx], expected_w13_scale):
            diff = torch.abs(w13_scale_bufs[tp_idx].float() - expected_w13_scale.float())
            max_diff_idx = diff.argmax().item()
            print(f"  w13 scale mismatch, max diff at index {max_diff_idx}")
            print(f"    actual: {w13_scale_bufs[tp_idx][max_diff_idx]}")
            print(f"    expected: {expected_w13_scale[max_diff_idx]}")
            raise AssertionError(f"w13 scale values mismatch for TP {tp_idx}")

        if not torch.equal(w2_weight_bufs[tp_idx], expected_w2_weight):
            diff_mask = w2_weight_bufs[tp_idx] != expected_w2_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            print(f"  w2 weight mismatch at index {first_diff_idx}")
            print(f"    actual: {w2_weight_bufs[tp_idx][first_diff_idx:first_diff_idx+10]}")
            print(f"    expected: {expected_w2_weight[first_diff_idx:first_diff_idx+10]}")
            raise AssertionError(f"w2 weight bytes mismatch for TP {tp_idx}")

        if not torch.allclose(w2_scale_bufs[tp_idx], expected_w2_scale):
            diff = torch.abs(w2_scale_bufs[tp_idx].float() - expected_w2_scale.float())
            max_diff_idx = diff.argmax().item()
            print(f"  w2 scale mismatch, max diff at index {max_diff_idx}")
            print(f"    actual: {w2_scale_bufs[tp_idx][max_diff_idx]}")
            print(f"    expected: {expected_w2_scale[max_diff_idx]}")
            raise AssertionError(f"w2 scale values mismatch for TP {tp_idx}")

    print(f"\n✓ write_weight_scale_to_buffer passed: verified expert {last_expert_idx} across {gpu_tp_count} TP parts")


if __name__ == "__main__":
    main()
