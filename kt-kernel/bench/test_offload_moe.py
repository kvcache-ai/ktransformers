#!/usr/bin/env python3
"""
Test script for AVX2 FP8 MoE with CPU Offloading Engine
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import kt_kernel_ext

# Initialize CPUInfer
CPUINFER_PARAM = 304
CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

def fp8_quantize(tensor: torch.Tensor, group_size: int = 128):
    """Quantize BF16 tensor to FP8 with block-wise scales."""
    # FP8 E4M3 range: -448 to 448
    fp8_max = 448.0

    shape = tensor.shape
    assert len(shape) == 2, "Expected 2D tensor [rows, cols]"
    rows, cols = shape

    # Block-wise quantization
    n_blocks_n = (rows + group_size - 1) // group_size
    n_blocks_k = (cols + group_size - 1) // group_size

    # Output FP8 weights
    fp8_weights = torch.zeros((rows, cols), dtype=torch.uint8)
    scales = torch.zeros((n_blocks_n, n_blocks_k), dtype=torch.float32)

    for bn in range(n_blocks_n):
        row_start = bn * group_size
        row_end = min(row_start + group_size, rows)

        for bk in range(n_blocks_k):
            col_start = bk * group_size
            col_end = min(col_start + group_size, cols)

            block = tensor[row_start:row_end, col_start:col_end].to(torch.float32)

            # Find max absolute value for scaling
            abs_max = block.abs().max().item()
            if abs_max > 0:
                scale = fp8_max / abs_max
            else:
                scale = 1.0

            scales[bn, bk] = scale

            # Quantize
            quantized = (block * scale).round().clamp(-448, 447).to(torch.int16)
            # Convert to E4M3 (simplified - just use as uint8)
            fp8_weights[row_start:row_end, col_start:col_end] = quantized.to(torch.uint8)

    return fp8_weights.contiguous(), scales.contiguous()

def test_offload_moe():
    print("=" * 60)
    print("Testing AVX2 FP8 MoE Offload")
    print("=" * 60)

    # Small model config for quick test
    expert_num = 8
    hidden_size = 1024
    intermediate_size = 2048
    num_experts_per_tok = 2
    qlen = 4
    group_size = 128

    print(f"Config: experts={expert_num}, hidden={hidden_size}, intermediate={intermediate_size}, topk={num_experts_per_tok}")

    # Create random BF16 weights
    torch.manual_seed(42)
    gate_proj_bf16 = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up_proj_bf16 = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down_proj_bf16 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16)

    # Quantize to FP8
    print("Quantizing weights to FP8...")
    gate_proj_fp8 = []
    gate_scales = []
    up_proj_fp8 = []
    up_scales = []
    down_proj_fp8 = []
    down_scales = []

    for e in range(expert_num):
        g_fp8, g_scale = fp8_quantize(gate_proj_bf16[e], group_size)
        u_fp8, u_scale = fp8_quantize(up_proj_bf16[e], group_size)
        d_fp8, d_scale = fp8_quantize(down_proj_bf16[e], group_size)

        gate_proj_fp8.append(g_fp8)
        gate_scales.append(g_scale)
        up_proj_fp8.append(u_fp8)
        up_scales.append(u_scale)
        down_proj_fp8.append(d_fp8)
        down_scales.append(d_scale)

    # Stack into per-expert pointer arrays (2D: [tp_partition][expert])
    # For CPU only, tp_partition = 1
    gate_projs = [[w.data_ptr() for w in gate_proj_fp8]]
    gate_scales_2d = [[s.data_ptr() for s in gate_scales]]
    up_projs = [[w.data_ptr() for w in up_proj_fp8]]
    up_scales_2d = [[s.data_ptr() for s in up_scales]]
    down_projs = [[w.data_ptr() for w in down_proj_fp8]]
    down_scales_2d = [[s.data_ptr() for s in down_scales]]

    # Create config for offloaded MoE
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.pool = CPUInfer.backend_
    config.max_len = 2048
    config.m_block = 1
    config.group_min_len = 10
    config.group_max_len = 1024
    config.layer_idx = 0

    # Set quant config for FP8 block-wise
    quant_config = kt_kernel_ext.QuantConfig()
    quant_config.quant_method = "fp8"
    quant_config.group_size = group_size
    quant_config.zero_point = False
    quant_config.per_channel = False
    config.quant_config = quant_config

    # Set per-expert pointers (required for offloaded MoE) - 2D arrays [tp][expert]
    config.gate_projs = gate_projs
    config.up_projs = up_projs
    config.down_projs = down_projs
    config.gate_scales = gate_scales_2d
    config.up_scales = up_scales_2d
    config.down_scales = down_scales_2d

    # Create offloaded MoE
    print("Creating AVX2FP8_MOE_OFFLOAD...")
    moe = kt_kernel_ext.moe.AVX2FP8_MOE_OFFLOAD(config)

    print("Loading weights...")
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()
    print("Weights loaded!")

    # Test forward pass
    print("Running forward pass...")
    expert_ids = torch.randint(0, expert_num, (qlen * num_experts_per_tok,), dtype=torch.int64).contiguous()
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
    input_tensor = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    output_tensor = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

    # Warm up
    for i in range(3):
        print(f"Warmup {i+1}/3...")
        CPUInfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        CPUInfer.sync()
        print(f"  done")

    # Benchmark
    iterations = 10
    start = time.perf_counter()
    for i in range(iterations):
        print(f"Benchmark {i+1}/{iterations}...")
        CPUInfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        CPUInfer.sync()
        print(f"  done")
    end = time.perf_counter()

    total_time = end - start
    time_per_iter_ms = total_time / iterations * 1000

    print(f"\nResults:")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {total_time:.4f}s")
    print(f"  Time per iter: {time_per_iter_ms:.4f}ms")
    print(f"  Throughput: {1000/time_per_iter_ms:.2f} iter/s")

    # Check output
    print(f"  Output stats: mean={output_tensor.mean().item():.4f}, std={output_tensor.std().item():.4f}")
    print("\nTest PASSED!")

    return True

if __name__ == "__main__":
    try:
        test_offload_moe()
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)