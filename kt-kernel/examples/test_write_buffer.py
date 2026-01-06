"""
Test write_weight_scale_to_buffer for AMX MOE operators.

Supports:
- FP8: FP8 weights (1 byte) + float32 scales
- BF16: Native BF16 weights (2 bytes), no scales

Usage:
    python test_write_buffer.py          # Run all modes
    python test_write_buffer.py fp8      # Run FP8 only
    python test_write_buffer.py bf16     # Run BF16 only
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from kt_kernel import kt_kernel_ext
from kt_kernel_ext import CPUInfer


def make_cpu_infer(thread_num=80):
    return CPUInfer(thread_num)


def div_up(a, b):
    return (a + b - 1) // b


def build_config_fp8(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size, group_size):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    cfg.max_len = 1
    cfg.quant_config.bits = 8  # FP8
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False
    cfg.pool = cpuinfer.backend_
    return cfg


def build_config_bf16(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    cfg.max_len = 1
    cfg.pool = cpuinfer.backend_
    return cfg


def allocate_weights_fp8(expert_num, hidden_size, intermediate_size, group_size):
    """Allocate FP8 weights and scales for testing"""
    # FP8 weights: 1 byte per element
    per_mat_weight_bytes = hidden_size * intermediate_size
    # FP8 scales: block-wise (group_size x group_size blocks), stored as float32
    n_blocks_n_gate_up = div_up(intermediate_size, group_size)
    n_blocks_k = div_up(hidden_size, group_size)
    per_mat_scale_elems_gate_up = n_blocks_n_gate_up * n_blocks_k

    # For down: n=hidden_size, k=intermediate_size
    n_blocks_n_down = n_blocks_k
    n_blocks_k_down = n_blocks_n_gate_up
    per_mat_scale_elems_down = n_blocks_n_down * n_blocks_k_down

    gate_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    up_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)
    down_q = torch.randint(0, 256, (expert_num * per_mat_weight_bytes,), dtype=torch.uint8)

    gate_scale = torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32)
    up_scale = torch.randn(expert_num * per_mat_scale_elems_gate_up, dtype=torch.float32)
    down_scale = torch.randn(expert_num * per_mat_scale_elems_down, dtype=torch.float32)

    return {
        "gate_q": gate_q,
        "up_q": up_q,
        "down_q": down_q,
        "gate_scale": gate_scale,
        "up_scale": up_scale,
        "down_scale": down_scale,
        "per_mat_weight_bytes": per_mat_weight_bytes,
        "per_mat_scale_elems_gate_up": per_mat_scale_elems_gate_up,
        "per_mat_scale_elems_down": per_mat_scale_elems_down,
    }


def allocate_weights_bf16(expert_num, hidden_size, intermediate_size):
    """Allocate BF16 weights for testing (no scales)"""
    # BF16 weights: 2 bytes per element
    per_mat_weight_elems = hidden_size * intermediate_size
    per_mat_weight_bytes = per_mat_weight_elems * 2  # BF16 = 2 bytes

    gate_proj = torch.randn(expert_num * per_mat_weight_elems, dtype=torch.bfloat16)
    up_proj = torch.randn(expert_num * per_mat_weight_elems, dtype=torch.bfloat16)
    down_proj = torch.randn(expert_num * per_mat_weight_elems, dtype=torch.bfloat16)

    return {
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
        "per_mat_weight_bytes": per_mat_weight_bytes,
        "per_mat_weight_elems": per_mat_weight_elems,
    }


def test_fp8_write_buffer(gpu_tp_count):
    """Test write_weight_scale_to_buffer with FP8 weights"""
    torch.manual_seed(123)

    expert_num = 256
    gpu_experts = expert_num
    num_experts_per_tok = 8
    hidden_size = 3072
    intermediate_size = 1536
    group_size = 128

    cpuinfer = make_cpu_infer()
    cfg = build_config_fp8(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size, group_size)
    weights = allocate_weights_fp8(expert_num, hidden_size, intermediate_size, group_size)

    cfg.gate_proj = weights["gate_q"].data_ptr()
    cfg.up_proj = weights["up_q"].data_ptr()
    cfg.down_proj = weights["down_q"].data_ptr()
    cfg.gate_scale = weights["gate_scale"].data_ptr()
    cfg.up_scale = weights["up_scale"].data_ptr()
    cfg.down_scale = weights["down_scale"].data_ptr()

    moe = kt_kernel_ext.moe.AMXFP8_MOE(cfg)

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    per_mat_weight_bytes = weights["per_mat_weight_bytes"]
    per_mat_scale_elems_gate_up = weights["per_mat_scale_elems_gate_up"]
    per_mat_scale_elems_down = weights["per_mat_scale_elems_down"]

    # Calculate sizes per TP part
    weight_bytes_per_expert_per_tp = per_mat_weight_bytes // gpu_tp_count
    gpu_n_w13 = intermediate_size // gpu_tp_count
    gpu_k_w13 = hidden_size
    scale_elems_per_expert_per_tp_gate_up = div_up(gpu_n_w13, group_size) * div_up(gpu_k_w13, group_size)
    gpu_n_w2 = hidden_size
    gpu_k_w2 = intermediate_size // gpu_tp_count
    scale_elems_per_expert_per_tp_down = div_up(gpu_n_w2, group_size) * div_up(gpu_k_w2, group_size)

    total_weight_bytes_per_tp = gpu_experts * weight_bytes_per_expert_per_tp
    total_scale_elems_per_tp_gate_up = gpu_experts * scale_elems_per_expert_per_tp_gate_up
    total_scale_elems_per_tp_down = gpu_experts * scale_elems_per_expert_per_tp_down

    # Create buffer lists
    w13_weight_bufs = [torch.empty(2 * total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w13_scale_bufs = [
        torch.empty(2 * total_scale_elems_per_tp_gate_up, dtype=torch.float32) for _ in range(gpu_tp_count)
    ]
    w2_weight_bufs = [torch.empty(total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w2_scale_bufs = [torch.empty(total_scale_elems_per_tp_down, dtype=torch.float32) for _ in range(gpu_tp_count)]

    print(f"[FP8] GPU TP count: {gpu_tp_count}, Experts: {expert_num}")
    print(f"[FP8] Weight bytes per expert per TP: {weight_bytes_per_expert_per_tp}")
    print(f"[FP8] Scale elements per expert per TP (gate/up): {scale_elems_per_expert_per_tp_gate_up}")

    def get_expert_ptrs(expert_id):
        w13_weight_ptrs = []
        w13_scale_ptrs = []
        w2_weight_ptrs = []
        w2_scale_ptrs = []
        for tp_idx in range(gpu_tp_count):
            w13_weight_expert_offset = expert_id * 2 * weight_bytes_per_expert_per_tp
            w13_scale_expert_offset = expert_id * 2 * scale_elems_per_expert_per_tp_gate_up
            w2_weight_expert_offset = expert_id * weight_bytes_per_expert_per_tp
            w2_scale_expert_offset = expert_id * scale_elems_per_expert_per_tp_down

            w13_weight_ptrs.append(w13_weight_bufs[tp_idx].data_ptr() + w13_weight_expert_offset)
            w13_scale_ptrs.append(w13_scale_bufs[tp_idx].data_ptr() + w13_scale_expert_offset * 4)
            w2_weight_ptrs.append(w2_weight_bufs[tp_idx].data_ptr() + w2_weight_expert_offset)
            w2_scale_ptrs.append(w2_scale_bufs[tp_idx].data_ptr() + w2_scale_expert_offset * 4)
        return w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs

    # Warm up
    for _ in range(2):
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
    elapsed_ms = (end_time - begin_time) / 1e6

    total_bytes = (
        hidden_size * intermediate_size * gpu_experts * 3
        + (per_mat_scale_elems_gate_up * 2 + per_mat_scale_elems_down) * gpu_experts * 4
    )
    print(f"[FP8] write_weight_scale_to_buffer time: {elapsed_ms:.2f} ms")
    print(f"[FP8] Throughput: {total_bytes / (elapsed_ms * 1e6):.2f} GB/s")

    # Verify correctness
    def split_expert_tensor(tensor, chunk):
        return [tensor[i * chunk : (i + 1) * chunk] for i in range(expert_num)]

    gate_q = weights["gate_q"]
    up_q = weights["up_q"]
    down_q = weights["down_q"]
    gate_scale = weights["gate_scale"]
    up_scale = weights["up_scale"]
    down_scale = weights["down_scale"]

    gate_q_experts = split_expert_tensor(gate_q, per_mat_weight_bytes)
    up_q_experts = split_expert_tensor(up_q, per_mat_weight_bytes)
    down_q_experts = split_expert_tensor(down_q, per_mat_weight_bytes)
    gate_scale_experts = split_expert_tensor(gate_scale, per_mat_scale_elems_gate_up)
    up_scale_experts = split_expert_tensor(up_scale, per_mat_scale_elems_gate_up)
    down_scale_experts = split_expert_tensor(down_scale, per_mat_scale_elems_down)

    n_blocks_n = div_up(hidden_size, group_size)
    n_blocks_k = div_up(intermediate_size, group_size)
    n_blocks_k_per_tp = n_blocks_k // gpu_tp_count

    for tp_idx in range(gpu_tp_count):
        expected_w13_weights = []
        expected_w13_scales = []
        expected_w2_weights = []
        expected_w2_scales = []

        weight13_per_tp = per_mat_weight_bytes // gpu_tp_count
        scale13_per_tp = per_mat_scale_elems_gate_up // gpu_tp_count

        for expert_id in range(gpu_experts):
            start_weight = tp_idx * weight13_per_tp
            end_weight = (tp_idx + 1) * weight13_per_tp
            start_scale = tp_idx * scale13_per_tp
            end_scale = (tp_idx + 1) * scale13_per_tp

            gate_weight_tp = gate_q_experts[expert_id][start_weight:end_weight]
            gate_scale_tp = gate_scale_experts[expert_id][start_scale:end_scale]
            up_weight_tp = up_q_experts[expert_id][start_weight:end_weight]
            up_scale_tp = up_scale_experts[expert_id][start_scale:end_scale]

            down_weight_tp_parts = []
            down_scale_tp_parts = []
            tp_slice_weight_size = intermediate_size // gpu_tp_count

            for row_idx in range(hidden_size):
                row_weight_start = row_idx * intermediate_size
                tp_weight_offset = row_weight_start + tp_idx * tp_slice_weight_size
                down_weight_tp_parts.append(
                    down_q_experts[expert_id][tp_weight_offset : tp_weight_offset + tp_slice_weight_size]
                )

            for bn in range(n_blocks_n):
                row_scale_start = bn * n_blocks_k
                tp_scale_offset = row_scale_start + tp_idx * n_blocks_k_per_tp
                down_scale_tp_parts.append(
                    down_scale_experts[expert_id][tp_scale_offset : tp_scale_offset + n_blocks_k_per_tp]
                )

            down_weight_tp = torch.cat(down_weight_tp_parts)
            down_scale_tp = torch.cat(down_scale_tp_parts)

            expected_w13_weights.append(gate_weight_tp)
            expected_w13_weights.append(up_weight_tp)
            expected_w13_scales.append(gate_scale_tp)
            expected_w13_scales.append(up_scale_tp)
            expected_w2_weights.append(down_weight_tp)
            expected_w2_scales.append(down_scale_tp)

        expected_w13_weight = torch.cat(expected_w13_weights)
        expected_w13_scale = torch.cat(expected_w13_scales)
        expected_w2_weight = torch.cat(expected_w2_weights)
        expected_w2_scale = torch.cat(expected_w2_scales)

        if not torch.equal(w13_weight_bufs[tp_idx], expected_w13_weight):
            diff_mask = w13_weight_bufs[tp_idx] != expected_w13_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            raise AssertionError(f"[FP8] w13 weight mismatch for TP {tp_idx} at index {first_diff_idx}")

        if not torch.allclose(w13_scale_bufs[tp_idx], expected_w13_scale):
            raise AssertionError(f"[FP8] w13 scale mismatch for TP {tp_idx}")

        if not torch.equal(w2_weight_bufs[tp_idx], expected_w2_weight):
            diff_mask = w2_weight_bufs[tp_idx] != expected_w2_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            raise AssertionError(f"[FP8] w2 weight mismatch for TP {tp_idx} at index {first_diff_idx}")

        if not torch.allclose(w2_scale_bufs[tp_idx], expected_w2_scale):
            raise AssertionError(f"[FP8] w2 scale mismatch for TP {tp_idx}")

    print(f"[FP8] TP={gpu_tp_count} PASSED (verified {gpu_experts} experts across {gpu_tp_count} TP parts)")
    return True


def test_bf16_write_buffer(gpu_tp_count):
    """Test write_weight_scale_to_buffer with BF16 weights (no scales)"""
    torch.manual_seed(123)

    expert_num = 256
    gpu_experts = expert_num
    num_experts_per_tok = 8
    hidden_size = 3072
    intermediate_size = 1536

    cpuinfer = make_cpu_infer()
    cfg = build_config_bf16(cpuinfer, expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    weights = allocate_weights_bf16(expert_num, hidden_size, intermediate_size)

    cfg.gate_proj = weights["gate_proj"].data_ptr()
    cfg.up_proj = weights["up_proj"].data_ptr()
    cfg.down_proj = weights["down_proj"].data_ptr()
    cfg.gate_scale = 0
    cfg.up_scale = 0
    cfg.down_scale = 0

    moe = kt_kernel_ext.moe.AMXBF16_MOE(cfg)

    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64, device="cpu").contiguous()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    per_mat_weight_elems = weights["per_mat_weight_elems"]

    # Calculate sizes per TP part (BF16 = 2 bytes per element)
    weight_elems_per_expert_per_tp = per_mat_weight_elems // gpu_tp_count
    weight_bytes_per_expert_per_tp = weight_elems_per_expert_per_tp * 2

    total_weight_bytes_per_tp = gpu_experts * weight_bytes_per_expert_per_tp

    # Create buffer lists (BF16: weights only, no scales)
    w13_weight_bufs = [torch.empty(2 * total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    w2_weight_bufs = [torch.empty(total_weight_bytes_per_tp, dtype=torch.uint8) for _ in range(gpu_tp_count)]
    # Empty scale buffers (not used for BF16 but needed for interface)
    w13_scale_bufs = [torch.empty(1, dtype=torch.float32) for _ in range(gpu_tp_count)]
    w2_scale_bufs = [torch.empty(1, dtype=torch.float32) for _ in range(gpu_tp_count)]

    print(f"[BF16] GPU TP count: {gpu_tp_count}, Experts: {expert_num}")
    print(f"[BF16] Weight bytes per expert per TP: {weight_bytes_per_expert_per_tp}")

    def get_expert_ptrs(expert_id):
        w13_weight_ptrs = []
        w13_scale_ptrs = []
        w2_weight_ptrs = []
        w2_scale_ptrs = []
        for tp_idx in range(gpu_tp_count):
            w13_weight_expert_offset = expert_id * 2 * weight_bytes_per_expert_per_tp
            w2_weight_expert_offset = expert_id * weight_bytes_per_expert_per_tp

            w13_weight_ptrs.append(w13_weight_bufs[tp_idx].data_ptr() + w13_weight_expert_offset)
            w13_scale_ptrs.append(w13_scale_bufs[tp_idx].data_ptr())  # Not used
            w2_weight_ptrs.append(w2_weight_bufs[tp_idx].data_ptr() + w2_weight_expert_offset)
            w2_scale_ptrs.append(w2_scale_bufs[tp_idx].data_ptr())  # Not used
        return w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs

    # Warm up
    for _ in range(2):
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
    elapsed_ms = (end_time - begin_time) / 1e6

    total_bytes = hidden_size * intermediate_size * gpu_experts * 3 * 2  # BF16 = 2 bytes
    print(f"[BF16] write_weight_scale_to_buffer time: {elapsed_ms:.2f} ms")
    print(f"[BF16] Throughput: {total_bytes / (elapsed_ms * 1e6):.2f} GB/s")

    # Verify correctness (BF16: weights only, no scales)
    def split_expert_tensor(tensor, chunk):
        return [tensor[i * chunk : (i + 1) * chunk] for i in range(expert_num)]

    gate_proj = weights["gate_proj"]
    up_proj = weights["up_proj"]
    down_proj = weights["down_proj"]

    # View BF16 as uint8 for byte-level comparison
    gate_bytes = gate_proj.view(torch.uint8)
    up_bytes = up_proj.view(torch.uint8)
    down_bytes = down_proj.view(torch.uint8)

    per_mat_bytes = per_mat_weight_elems * 2  # BF16 = 2 bytes
    gate_experts = split_expert_tensor(gate_bytes, per_mat_bytes)
    up_experts = split_expert_tensor(up_bytes, per_mat_bytes)
    down_experts = split_expert_tensor(down_bytes, per_mat_bytes)

    for tp_idx in range(gpu_tp_count):
        expected_w13_weights = []
        expected_w2_weights = []

        weight_bytes_per_tp = per_mat_bytes // gpu_tp_count

        for expert_id in range(gpu_experts):
            start_weight = tp_idx * weight_bytes_per_tp
            end_weight = (tp_idx + 1) * weight_bytes_per_tp

            gate_weight_tp = gate_experts[expert_id][start_weight:end_weight]
            up_weight_tp = up_experts[expert_id][start_weight:end_weight]

            # Down matrix: sliced column-wise (BF16 = 2 bytes per element)
            down_weight_tp_parts = []
            tp_slice_elems = intermediate_size // gpu_tp_count
            tp_slice_bytes = tp_slice_elems * 2

            for row_idx in range(hidden_size):
                row_byte_start = row_idx * intermediate_size * 2
                tp_byte_offset = row_byte_start + tp_idx * tp_slice_bytes
                down_weight_tp_parts.append(down_experts[expert_id][tp_byte_offset : tp_byte_offset + tp_slice_bytes])

            down_weight_tp = torch.cat(down_weight_tp_parts)

            expected_w13_weights.append(gate_weight_tp)
            expected_w13_weights.append(up_weight_tp)
            expected_w2_weights.append(down_weight_tp)

        expected_w13_weight = torch.cat(expected_w13_weights)
        expected_w2_weight = torch.cat(expected_w2_weights)

        if not torch.equal(w13_weight_bufs[tp_idx], expected_w13_weight):
            diff_mask = w13_weight_bufs[tp_idx] != expected_w13_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            raise AssertionError(f"[BF16] w13 weight mismatch for TP {tp_idx} at index {first_diff_idx}")

        if not torch.equal(w2_weight_bufs[tp_idx], expected_w2_weight):
            diff_mask = w2_weight_bufs[tp_idx] != expected_w2_weight
            first_diff_idx = diff_mask.nonzero()[0].item() if diff_mask.any() else -1
            raise AssertionError(f"[BF16] w2 weight mismatch for TP {tp_idx} at index {first_diff_idx}")

    print(f"[BF16] TP={gpu_tp_count} PASSED (verified {gpu_experts} experts across {gpu_tp_count} TP parts)")
    return True


def test_with_tp(quant_mode: str, gpu_tp_count: int):
    """Test write_weight_scale_to_buffer with specified mode and TP count"""
    if quant_mode == "fp8":
        return test_fp8_write_buffer(gpu_tp_count)
    elif quant_mode == "bf16":
        return test_bf16_write_buffer(gpu_tp_count)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")


def main(quant_modes=None):
    """Run tests for specified quant modes"""
    if quant_modes is None:
        quant_modes = ["fp8", "bf16"]

    tp_values = [1, 2, 4]
    all_passed = True
    results = {}

    for quant_mode in quant_modes:
        print("\n" + "=" * 60)
        print(f"Testing {quant_mode.upper()} write_weight_scale_to_buffer")
        print("=" * 60)

        for tp in tp_values:
            print(f"\n--- Testing {quant_mode.upper()} with gpu_tp_count = {tp} ---")
            try:
                test_with_tp(quant_mode, tp)
                results[(quant_mode, tp)] = "PASSED"
            except Exception as e:
                results[(quant_mode, tp)] = f"FAILED: {e}"
                all_passed = False
                print(f"[{quant_mode.upper()}] TP={tp} FAILED: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for (mode, tp), result in results.items():
        status = "PASS" if "PASSED" in result else "FAIL"
        print(f"  [{status}] {mode.upper()} TP={tp}: {result}")

    if all_passed:
        print("\nALL TESTS PASSED")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ["fp8", "bf16"]:
            main([mode])
        else:
            print(f"Unknown mode: {mode}. Use 'fp8' or 'bf16'")
            sys.exit(1)
    else:
        main()
