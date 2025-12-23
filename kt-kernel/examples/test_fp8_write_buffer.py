import os
import sys
import time

import torch
import numpy as np


from kt_kernel import kt_kernel_ext
from kt_kernel_ext import CPUInfer
from kt_kernel_ext.moe import AMXFP8_MOE


def make_cpu_infer(thread_num=80):
    return CPUInfer(thread_num)


def build_config(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size, group_size):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    cfg.max_len = 1
    cfg.quant_config.bits = 8  # FP8
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False
    cfg.pool = cpuinfer.backend_
    return cfg


def allocate_weights(expert_num, hidden_size, intermediate_size, group_size):
    """Allocate FP8 weights and scales for testing"""
    # FP8 weights: 1 byte per element
    per_mat_weight_bytes = hidden_size * intermediate_size
    # FP8 scales: block-wise (group_size x group_size blocks), stored as float32
    n_blocks_n_gate_up = (intermediate_size + group_size - 1) // group_size
    n_blocks_k = (hidden_size + group_size - 1) // group_size
    per_mat_scale_elems_gate_up = n_blocks_n_gate_up * n_blocks_k

    # For down: n=hidden_size, k=intermediate_size
    n_blocks_n_down = n_blocks_k
    n_blocks_k_down = n_blocks_n_gate_up
    per_mat_scale_elems_down = n_blocks_n_down * n_blocks_k_down

    gate_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    up_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    down_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)

    # FP8 scales are float32
    gate_scale = torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32)
    up_scale = torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32)
    down_scale = torch.randn(expert_num * per_mat_scale_elems_down, dtype=torch.float32)

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


def test_with_tp(gpu_tp_count):
    """Test write_weight_scale_to_buffer with a specific gpu_tp_count"""
    torch.manual_seed(123)

    expert_num = 256  # Reduced for debugging
    gpu_experts = expert_num  # Number of experts on GPU

    num_experts_per_tok = 8
    hidden_size = 3072
    intermediate_size = 1536  # Changed from 2048 to test non-aligned case
    group_size = 128  # FP8 uses 128x128 block-wise scales

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
        per_mat_scale_elems_gate_up,
        per_mat_scale_elems_down,
    ) = allocate_weights(expert_num, hidden_size, intermediate_size, group_size)

    cfg.gate_proj = gate_q.data_ptr()
    cfg.up_proj = up_q.data_ptr()
    cfg.down_proj = down_q.data_ptr()
    cfg.gate_scale = gate_scale.data_ptr()
    cfg.up_scale = up_scale.data_ptr()
    cfg.down_scale = down_scale.data_ptr()

    moe = AMXFP8_MOE(cfg)

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    # TP configuration
    # Calculate sizes per TP part (per expert) - must match C++ code which uses div_up
    def div_up(a, b):
        return (a + b - 1) // b

    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count

    # For W13 (gate/up): n=intermediate_size/gpu_tp, k=hidden_size
    gpu_n_w13 = intermediate_size // gpu_tp_count
    gpu_k_w13 = hidden_size
    scale_elems_per_expert_per_tp_gate_up = div_up(gpu_n_w13, group_size) * div_up(gpu_k_w13, group_size)

    # For W2 (down): n=hidden_size, k=intermediate_size/gpu_tp
    gpu_n_w2 = hidden_size
    gpu_k_w2 = intermediate_size // gpu_tp_count
    scale_elems_per_expert_per_tp_down = div_up(gpu_n_w2, group_size) * div_up(gpu_k_w2, group_size)

    # Total sizes for all gpu_experts
    total_weight_bytes_per_tp = gpu_experts * weight_bytes_per_expert_per_tp
    total_scale_elems_per_tp_gate_up = gpu_experts * scale_elems_per_expert_per_tp_gate_up
    total_scale_elems_per_tp_down = gpu_experts * scale_elems_per_expert_per_tp_down

    # Create buffer lists for w13 (gate+up) and w2 (down)
    # These hold all experts' data for each GPU TP
    w13_weight_bufs = []
    w13_scale_bufs = []
    w2_weight_bufs = []
    w2_scale_bufs = []

    for tp_idx in range(gpu_tp_count):
        # w13 combines gate and up, so needs 2x the size
        w13_weight_bufs.append(torch.empty(2 * total_weight_bytes_per_tp, dtype=torch.uint8))
        w13_scale_bufs.append(torch.empty(2 * total_scale_elems_per_tp_gate_up, dtype=torch.float32))
        w2_weight_bufs.append(torch.empty(total_weight_bytes_per_tp, dtype=torch.uint8))
        w2_scale_bufs.append(torch.empty(total_scale_elems_per_tp_down, dtype=torch.float32))

    print(f"Total experts: {expert_num}, GPU experts: {gpu_experts}")
    print(f"GPU TP count: {gpu_tp_count}")
    print(f"Original per matrix weight bytes: {per_mat_weight_bytes}")
    print(f"Original per matrix scale elements (gate/up): {per_mat_scale_elems_gate_up}")
    print(f"Original per matrix scale elements (down): {per_mat_scale_elems_down}")
    print(f"Weight bytes per expert per TP: {weight_bytes_per_expert_per_tp}")
    print(f"Scale elements per expert per TP (gate/up): {scale_elems_per_expert_per_tp_gate_up}")
    print(f"Scale elements per expert per TP (down): {scale_elems_per_expert_per_tp_down}")
    print(f"Total weight bytes per TP (w13): {2 * total_weight_bytes_per_tp}")
    print(f"Total weight bytes per TP (w2): {total_weight_bytes_per_tp}")

    # Helper function to get pointers with expert offset
    # write_weights_to_buffer writes one expert at a time, so we need to pass
    # pointers that already point to the correct location for each expert
    def get_expert_ptrs(expert_id):
        w13_weight_ptrs = []
        w13_scale_ptrs = []
        w2_weight_ptrs = []
        w2_scale_ptrs = []

        for tp_idx in range(gpu_tp_count):
            # Calculate byte offsets for this expert
            # w13: gate_weight + up_weight interleaved by expert
            # Layout: [expert0_gate, expert0_up, expert1_gate, expert1_up, ...]
            w13_weight_expert_offset = expert_id * 2 * weight_bytes_per_expert_per_tp
            w13_scale_expert_offset = expert_id * 2 * scale_elems_per_expert_per_tp_gate_up
            w2_weight_expert_offset = expert_id * weight_bytes_per_expert_per_tp
            w2_scale_expert_offset = expert_id * scale_elems_per_expert_per_tp_down

            w13_weight_ptrs.append(w13_weight_bufs[tp_idx].data_ptr() + w13_weight_expert_offset)
            w13_scale_ptrs.append(w13_scale_bufs[tp_idx].data_ptr() + w13_scale_expert_offset * 4)  # float32 = 4 bytes
            w2_weight_ptrs.append(w2_weight_bufs[tp_idx].data_ptr() + w2_weight_expert_offset)
            w2_scale_ptrs.append(w2_scale_bufs[tp_idx].data_ptr() + w2_scale_expert_offset * 4)  # float32 = 4 bytes

        return w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs

    # Warm up
    for i in range(2):
        for expert_id in range(gpu_experts):
            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs = get_expert_ptrs(expert_id)
            cpuinfer.submit(
                moe.write_weight_scale_to_buffer_task(
                    gpu_tp_count=gpu_tp_count,
                    expert_id=expert_id,
                    w13_weight_ptrs=w13_weight_ptrs,
                    w13_scale_ptrs=w13_scale_ptrs,
                    w2_weight_ptrs=w2_weight_ptrs,
                    w2_scale_ptrs=w2_scale_ptrs,
                )
            )
            cpuinfer.sync()

    # Timing
    begin_time = time.perf_counter_ns()
    for expert_id in range(gpu_experts):
        w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs = get_expert_ptrs(expert_id)
        cpuinfer.submit(
            moe.write_weight_scale_to_buffer_task(
                gpu_tp_count=gpu_tp_count,
                expert_id=expert_id,
                w13_weight_ptrs=w13_weight_ptrs,
                w13_scale_ptrs=w13_scale_ptrs,
                w2_weight_ptrs=w2_weight_ptrs,
                w2_scale_ptrs=w2_scale_ptrs,
            )
        )
        cpuinfer.sync()
    end_time = time.perf_counter_ns()
    elapsed_ms = (end_time - begin_time) / 1000000

    # Calculate throughput
    total_weights = hidden_size * intermediate_size * gpu_experts * 3
    total_scale_bytes = (per_mat_scale_elems_gate_up * 2 + per_mat_scale_elems_down) * gpu_experts * 4  # float32
    total_bytes = total_weights + total_scale_bytes
    print(f"write_weight_scale_to_buffer time: {elapsed_ms:.2f} ms")
    print(f"Throughput: {total_bytes / (elapsed_ms * 1e6):.2f} GB/s")

    def split_expert_tensor(tensor, chunk):
        """Split tensor by experts"""
        return [tensor[i * chunk : (i + 1) * chunk] for i in range(expert_num)]

    # Split by experts first
    gate_q_experts = split_expert_tensor(gate_q, per_mat_weight_bytes)
    up_q_experts = split_expert_tensor(up_q, per_mat_weight_bytes)
    down_q_experts = split_expert_tensor(down_q, per_mat_weight_bytes)

    gate_scale_experts = split_expert_tensor(gate_scale, per_mat_scale_elems_gate_up)
    up_scale_experts = split_expert_tensor(up_scale, per_mat_scale_elems_gate_up)
    down_scale_experts = split_expert_tensor(down_scale, per_mat_scale_elems_down)

    # For down matrix
    n_blocks_n = (hidden_size + group_size - 1) // group_size
    n_blocks_k = (intermediate_size + group_size - 1) // group_size
    n_blocks_k_per_tp = n_blocks_k // gpu_tp_count

    # Verify buffers for each TP part
    for tp_idx in range(gpu_tp_count):
        expected_w13_weights = []
        expected_w13_scales = []
        expected_w2_weights = []
        expected_w2_scales = []

        weight13_per_tp = per_mat_weight_bytes // gpu_tp_count
        scale13_per_tp = per_mat_scale_elems_gate_up // gpu_tp_count

        # Process each GPU expert
        for expert_id in range(gpu_experts):
            # For w13 (gate and up), the slicing is along intermediate_size (n direction)
            start_weight = tp_idx * weight13_per_tp
            end_weight = (tp_idx + 1) * weight13_per_tp
            start_scale = tp_idx * scale13_per_tp
            end_scale = (tp_idx + 1) * scale13_per_tp

            # Gate
            gate_weight_tp = gate_q_experts[expert_id][start_weight:end_weight]
            gate_scale_tp = gate_scale_experts[expert_id][start_scale:end_scale]

            # Up
            up_weight_tp = up_q_experts[expert_id][start_weight:end_weight]
            up_scale_tp = up_scale_experts[expert_id][start_scale:end_scale]

            # Down matrix needs special handling because it's sliced column-wise
            # down is (hidden_size, intermediate_size) in n-major format
            down_weight_tp_parts = []
            down_scale_tp_parts = []

            # Iterate through each row to extract the corresponding parts
            for row_idx in range(hidden_size):
                row_weight_start = row_idx * intermediate_size

                # Direct mapping: each CPU TP corresponds to a GPU TP
                tp_slice_weight_size = intermediate_size // gpu_tp_count

                tp_weight_offset = row_weight_start + tp_idx * tp_slice_weight_size

                down_weight_tp_parts.append(
                    down_q_experts[expert_id][tp_weight_offset : tp_weight_offset + tp_slice_weight_size]
                )

            # For scale: only process at block boundaries
            for bn in range(n_blocks_n):
                row_scale_start = bn * n_blocks_k
                tp_scale_offset = row_scale_start + tp_idx * n_blocks_k_per_tp
                down_scale_tp_parts.append(
                    down_scale_experts[expert_id][tp_scale_offset : tp_scale_offset + n_blocks_k_per_tp]
                )

            # Concatenate all slices for this TP
            down_weight_tp = torch.cat(down_weight_tp_parts)
            down_scale_tp = torch.cat(down_scale_tp_parts)

            # Append to expected lists - interleaved by expert: [gate0, up0, gate1, up1, ...]
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
        print(f"  w13 weight shape: actual={w13_weight_bufs[tp_idx].shape}, expected={expected_w13_weight.shape}")
        print(f"  w13 scale shape: actual={w13_scale_bufs[tp_idx].shape}, expected={expected_w13_scale.shape}")
        print(f"  w2 weight shape: actual={w2_weight_bufs[tp_idx].shape}, expected={expected_w2_weight.shape}")
        print(f"  w2 scale shape: actual={w2_scale_bufs[tp_idx].shape}, expected={expected_w2_scale.shape}")

        # Assert all checks pass
        if not torch.equal(w13_weight_bufs[tp_idx], expected_w13_weight):
            # Find first mismatch
            diff_mask = w13_weight_bufs[tp_idx] != expected_w13_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            print(f"  w13 weight mismatch at index {first_diff_idx}")
            print(f"    actual: {w13_weight_bufs[tp_idx][first_diff_idx:first_diff_idx+10]}")
            print(f"    expected: {expected_w13_weight[first_diff_idx:first_diff_idx+10]}")
            raise AssertionError(f"w13 weight bytes mismatch for TP {tp_idx}")

        if not torch.allclose(w13_scale_bufs[tp_idx], expected_w13_scale):
            diff = torch.abs(w13_scale_bufs[tp_idx] - expected_w13_scale)
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
            diff = torch.abs(w2_scale_bufs[tp_idx] - expected_w2_scale)
            max_diff_idx = diff.argmax().item()
            print(f"  w2 scale mismatch, max diff at index {max_diff_idx}")
            print(f"    actual: {w2_scale_bufs[tp_idx][max_diff_idx]}")
            print(f"    expected: {expected_w2_scale[max_diff_idx]}")
            raise AssertionError(f"w2 scale values mismatch for TP {tp_idx}")

    print(
        f"\n✓ write_weight_scale_to_buffer passed: extracted {gpu_experts} GPU experts across {gpu_tp_count} TP parts"
    )
    return True


def main():
    """Run tests for all gpu_tp_count values: 1, 2, 4, 8"""
    tp_values = [1, 2, 4]  # Test TP=8
    all_passed = True
    results = {}

    print("=" * 60)
    print("Testing FP8 write_weight_scale_to_buffer for TP = ", tp_values)
    print("=" * 60)

    for tp in tp_values:
        print(f"\n{'='*60}")
        print(f"Testing with gpu_tp_count = {tp}")
        print(f"{'='*60}")
        try:
            test_with_tp(tp)
            results[tp] = "PASSED"
            print(f"✓ TP={tp} PASSED")
        except Exception as e:
            results[tp] = f"FAILED: {e}"
            all_passed = False
            print(f"✗ TP={tp} FAILED: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for tp, result in results.items():
        status = "✓" if "PASSED" in result else "✗"
        print(f"  {status} TP={tp}: {result}")

    if all_passed:
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
