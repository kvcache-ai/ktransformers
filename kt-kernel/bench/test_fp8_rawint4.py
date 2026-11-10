#!/usr/bin/env python3
"""
Test script for FP8 and RAWINT4 AVX2 MoE kernels with weight pre-decoding.
Tests the performance improvement from weight pre-decoding optimization.
"""

import os
import sys
import time
import torch
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

try:
    import kt_kernel_ext
    print("Successfully imported kt_kernel_ext")
except ImportError as e:
    print(f"Failed to import kt_kernel_ext: {e}")
    sys.exit(1)


def test_fp8_moe():
    """Test FP8 MoE kernel with weight pre-decoding."""
    print("\n" + "="*60)
    print("Testing FP8 MoE Kernel (AVX2FP8_MOE)")
    print("="*60)

    # Test parameters
    expert_num = 8
    hidden_size = 2048
    intermediate_size = 5632  # Multiple of 128 for block-wise quantization
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1

    # Create CPUInfer first
    cpuinfer = kt_kernel_ext.CPUInfer(cpuinfer_threads)

    # Create config
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.pool = cpuinfer.backend_
    config.max_len = 2048  # Must be >= max sequence length (qlen) to avoid buffer overflow
    config.quant_config = kt_kernel_ext.QuantConfig()
    config.quant_config.group_size = 128  # Block-wise FP8
    config.quant_config.zero_point = False

    # Generate test data
    print("Generating FP8 test weights...")
    np.random.seed(42)
    gate_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    up_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    down_proj = np.random.randn(expert_num, hidden_size, intermediate_size).astype(np.float32)

    # Quantize to FP8 E4M3
    def quantize_fp8_blockwise(weight, group_size=128):
        """Quantize FP32 weight to FP8 E4M3 with block-wise scaling."""
        n, k = weight.shape
        n_blocks_n = (n + group_size - 1) // group_size
        n_blocks_k = (k + group_size - 1) // group_size

        quantized = np.zeros((n, k), dtype=np.uint8)
        scales = np.zeros((n_blocks_n, n_blocks_k), dtype=np.float32)

        for i in range(n_blocks_n):
            for j in range(n_blocks_k):
                row_start = i * group_size
                row_end = min(row_start + group_size, n)
                col_start = j * group_size
                col_end = min(col_start + group_size, k)

                block = weight[row_start:row_end, col_start:col_end]
                max_val = np.max(np.abs(block))
                scale = max_val / 448.0 if max_val > 0 else 1.0
                scales[i, j] = scale

                # FP8 E4M3 range: [-448, 448]
                q_block = np.round(block / scale).clip(-448, 448)
                quantized[row_start:row_end, col_start:col_end] = q_block.astype(np.uint8)

        return quantized, scales

    gate_quant, gate_scales = quantize_fp8_blockwise(gate_proj.reshape(-1, hidden_size))
    up_quant, up_scales = quantize_fp8_blockwise(up_proj.reshape(-1, hidden_size))
    down_quant, down_scales = quantize_fp8_blockwise(down_proj.reshape(-1, intermediate_size))

    gate_quant = gate_quant.reshape(expert_num, intermediate_size, hidden_size)
    up_quant = up_quant.reshape(expert_num, intermediate_size, hidden_size)
    down_quant = down_quant.reshape(expert_num, hidden_size, intermediate_size)

    gate_scales = gate_scales.reshape(expert_num, -1)
    up_scales = up_scales.reshape(expert_num, -1)
    down_scales = down_scales.reshape(expert_num, -1)

    # Convert to torch tensors
    gate_proj_t = torch.from_numpy(gate_quant).contiguous()
    up_proj_t = torch.from_numpy(up_quant).contiguous()
    down_proj_t = torch.from_numpy(down_quant).contiguous()
    gate_scales_t = torch.from_numpy(gate_scales).contiguous()
    up_scales_t = torch.from_numpy(up_scales).contiguous()
    down_scales_t = torch.from_numpy(down_scales).contiguous()

    # Set weight pointers in config
    config.gate_proj = gate_proj_t.data_ptr()
    config.up_proj = up_proj_t.data_ptr()
    config.down_proj = down_proj_t.data_ptr()
    config.gate_scale = gate_scales_t.data_ptr()
    config.up_scale = up_scales_t.data_ptr()
    config.down_scale = down_scales_t.data_ptr()
    config.gate_type = 22  # GGML_TYPE_FP8_E4M3
    config.up_type = 22
    config.down_type = 22
    config.hidden_type = 30  # GGML_TYPE_BF16

    # Create wrapper - use AVX2FP8_MOE class directly
    print("Creating FP8 MoE wrapper...")
    moe = kt_kernel_ext.moe.AVX2FP8_MOE(config)

    # Load weights
    print("Loading FP8 weights...")
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    config.physical_to_logical_map = physical_to_logical_map.data_ptr()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    # Test forward pass
    print("Running FP8 forward pass...")
    batch_size = 1
    seq_len = 1024  # Prefill length
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    output_tensor = torch.empty(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Expert routing (random for testing)
    expert_ids = torch.randint(0, expert_num, (batch_size, seq_len * num_experts_per_tok), dtype=torch.int32)
    weights = torch.rand(batch_size, seq_len, num_experts_per_tok, dtype=torch.float32)

    qlen_tensor = torch.tensor([seq_len], dtype=torch.int32)

    # Warmup
    for _ in range(5):
        cpuinfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        cpuinfer.sync()

    # Benchmark
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        cpuinfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        cpuinfer.sync()
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms
    print(f"FP8 MoE forward (prefill 1024 tokens): {avg_time:.2f} ms/iter ({iterations} iters)")
    print(f"Output shape: {output_tensor.shape}")

    return avg_time


def test_rawint4_moe():
    """Test RAWINT4 MoE kernel with weight pre-decoding."""
    print("\n" + "="*60)
    print("Testing RAWINT4 MoE Kernel (AVX2RawInt4_MOE)")
    print("="*60)

    # Test parameters
    expert_num = 8
    hidden_size = 2048
    intermediate_size = 5632  # Multiple of group_size
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1
    group_size = 32  # K-group size for RAWINT4

    # Create CPUInfer first
    cpuinfer = kt_kernel_ext.CPUInfer(cpuinfer_threads)

    # Create config
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.pool = cpuinfer.backend_
    config.max_len = 2048  # Must be >= max sequence length (qlen) to avoid buffer overflow
    config.quant_config = kt_kernel_ext.QuantConfig()
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False

    # Generate test data
    print("Generating RAWINT4 test weights...")
    np.random.seed(42)
    gate_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    up_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    down_proj = np.random.randn(expert_num, hidden_size, intermediate_size).astype(np.float32)

    # Quantize to RAWINT4 (signed int4, packed 2 per byte) - vectorized version
    def quantize_rawint4(weight, group_size=32):
        """Quantize FP32 weight to signed INT4 with K-group scales."""
        n, k = weight.shape
        assert k % group_size == 0, "k must be divisible by group_size"

        num_groups = k // group_size

        # Reshape to (n, num_groups, group_size)
        weight_reshaped = weight.reshape(n, num_groups, group_size)

        # Compute scales per group (vectorized)
        max_vals = np.max(np.abs(weight_reshaped), axis=2)
        scales = np.where(max_vals > 0, max_vals / 7.0, 1.0).astype(np.float32)

        # Quantize (vectorized)
        q_blocks = np.round(weight_reshaped / scales[:, :, np.newaxis]).clip(-8, 7).astype(np.int8)

        # Pack 2 int4 values per byte (vectorized)
        # q_blocks shape: (n, num_groups, group_size)
        # We want to pack pairs: [0,1] -> byte 0, [2,3] -> byte 1, etc.
        even_vals = (q_blocks[:, :, 0::2] + 8) & 0x0F  # shape: (n, num_groups, group_size//2)
        odd_vals = (q_blocks[:, :, 1::2] + 8) & 0x0F   # shape: (n, num_groups, group_size//2)
        packed = even_vals | (odd_vals << 4)  # shape: (n, num_groups, group_size//2)

        # Reshape to (n, k//2)
        quantized = packed.reshape(n, k // 2).astype(np.uint8)

        return quantized, scales

    gate_quant, gate_scales = quantize_rawint4(gate_proj.reshape(-1, hidden_size), group_size)
    up_quant, up_scales = quantize_rawint4(up_proj.reshape(-1, hidden_size), group_size)
    down_quant, down_scales = quantize_rawint4(down_proj.reshape(-1, intermediate_size), group_size)

    gate_quant = gate_quant.reshape(expert_num, intermediate_size, hidden_size // 2)
    up_quant = up_quant.reshape(expert_num, intermediate_size, hidden_size // 2)
    down_quant = down_quant.reshape(expert_num, hidden_size, intermediate_size // 2)

    gate_scales = gate_scales.reshape(expert_num, -1)
    up_scales = up_scales.reshape(expert_num, -1)
    down_scales = down_scales.reshape(expert_num, -1)

    # Convert scales to BF16 (RAWINT4 expects BF16 scales in flat-buffer mode)
    gate_scales_bf16 = torch.from_numpy(gate_scales).bfloat16().contiguous()
    up_scales_bf16 = torch.from_numpy(up_scales).bfloat16().contiguous()
    down_scales_bf16 = torch.from_numpy(down_scales).bfloat16().contiguous()

    # Set weight pointers in config BEFORE creating MOE (TP_MOE copies config in constructor)
    config.gate_proj = torch.from_numpy(gate_quant).contiguous().data_ptr()
    config.up_proj = torch.from_numpy(up_quant).contiguous().data_ptr()
    config.down_proj = torch.from_numpy(down_quant).contiguous().data_ptr()
    config.gate_scale = gate_scales_bf16.data_ptr()
    config.up_scale = up_scales_bf16.data_ptr()
    config.down_scale = down_scales_bf16.data_ptr()
    config.gate_type = 22  # GGML_TYPE_FP8_E4M3 (RAWINT4 uses this for packed int4)
    config.up_type = 22
    config.down_type = 22
    config.hidden_type = 30  # GGML_TYPE_BF16

    # Create wrapper - use AVX2RawInt4_MOE class directly
    print("Creating RAWINT4 MoE wrapper...")
    moe = kt_kernel_ext.moe.AVX2RawInt4_MOE(config)

    # Load weights
    print("Loading RAWINT4 weights...")
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    config.physical_to_logical_map = physical_to_logical_map.data_ptr()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpuinfer.sync()

    # Test forward pass
    print("Running RAWINT4 forward pass...")
    batch_size = 1
    seq_len = 1024  # Prefill length
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    output_tensor = torch.empty(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Expert routing (random for testing)
    expert_ids = torch.randint(0, expert_num, (batch_size, seq_len * num_experts_per_tok), dtype=torch.int32)
    weights = torch.rand(batch_size, seq_len, num_experts_per_tok, dtype=torch.float32)

    qlen_tensor = torch.tensor([seq_len], dtype=torch.int32)

    # Warmup
    for _ in range(5):
        cpuinfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        cpuinfer.sync()

    # Benchmark
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        cpuinfer.submit(moe.forward_task(
            qlen_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_tensor.data_ptr(),
            output_tensor.data_ptr(),
            False,
        ))
        cpuinfer.sync()
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms
    print(f"RAWINT4 MoE forward (prefill 1024 tokens): {avg_time:.2f} ms/iter ({iterations} iters)")
    print(f"Output shape: {output_tensor.shape}")

    return avg_time


def test_decode_performance():
    """Test decode performance (m=1) where pre-decoding should still help for first token."""
    print("\n" + "="*60)
    print("Testing Decode Performance (m=1)")
    print("="*60)

    expert_num = 8
    hidden_size = 2048
    intermediate_size = 5632
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1

    # Test both FP8 and RAWINT4
    for method in ["FP8", "RAWINT4"]:
        print(f"\n--- {method} Decode ---")

        # Create CPUInfer first
        cpuinfer = kt_kernel_ext.CPUInfer(cpuinfer_threads)

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
        config.pool = cpuinfer.backend_
        config.max_len = 2048  # Must be >= max sequence length (qlen) to avoid buffer overflow
        config.quant_config = kt_kernel_ext.QuantConfig()
        config.quant_config.group_size = 128 if method == "FP8" else 32
        config.quant_config.zero_point = False

        # Quick test with dummy weights
        physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)

        if method == "FP8":
            gate_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size), dtype=np.uint8)
            up_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size), dtype=np.uint8)
            down_proj = np.random.randint(0, 256, (expert_num, hidden_size, intermediate_size), dtype=np.uint8)
            gate_scales = np.random.randn(expert_num, (intermediate_size // 128) * (hidden_size // 128)).astype(np.float32)
            up_scales = np.random.randn(expert_num, (intermediate_size // 128) * (hidden_size // 128)).astype(np.float32)
            down_scales = np.random.randn(expert_num, (hidden_size // 128) * (intermediate_size // 128)).astype(np.float32)
            gate_scale_ptr = gate_scales.ctypes.data
            up_scale_ptr = up_scales.ctypes.data
            down_scale_ptr = down_scales.ctypes.data
            config.gate_type = 22  # GGML_TYPE_FP8_E4M3
            config.up_type = 22
            config.down_type = 22
            config.hidden_type = 30  # GGML_TYPE_BF16
        else:
            gate_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=np.uint8)
            up_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=np.uint8)
            down_proj = np.random.randint(0, 256, (expert_num, hidden_size, intermediate_size // 2), dtype=np.uint8)
            gate_scales = np.random.randn(expert_num, (intermediate_size // 32) * hidden_size).astype(np.float32)
            up_scales = np.random.randn(expert_num, (intermediate_size // 32) * hidden_size).astype(np.float32)
            down_scales = np.random.randn(expert_num, (hidden_size // 32) * intermediate_size).astype(np.float32)
            # RAWINT4 expects BF16 scales
            gate_scales_bf16 = torch.from_numpy(gate_scales).bfloat16().contiguous()
            up_scales_bf16 = torch.from_numpy(up_scales).bfloat16().contiguous()
            down_scales_bf16 = torch.from_numpy(down_scales).bfloat16().contiguous()
            gate_scale_ptr = gate_scales_bf16.data_ptr()
            up_scale_ptr = up_scales_bf16.data_ptr()
            down_scale_ptr = down_scales_bf16.data_ptr()
            config.gate_type = 22  # GGML_TYPE_FP8_E4M3 (RAWINT4 uses this for packed int4)
            config.up_type = 22
            config.down_type = 22
            config.hidden_type = 30  # GGML_TYPE_BF16

        config.gate_proj = gate_proj.ctypes.data
        config.up_proj = up_proj.ctypes.data
        config.down_proj = down_proj.ctypes.data
        config.gate_scale = gate_scale_ptr
        config.up_scale = up_scale_ptr
        config.down_scale = down_scale_ptr
        config.physical_to_logical_map = physical_to_logical_map.data_ptr()

        if method == "FP8":
            moe = kt_kernel_ext.moe.AVX2FP8_MOE(config)
        else:
            moe = kt_kernel_ext.moe.AVX2RawInt4_MOE(config)

        cpuinfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpuinfer.sync()

        # Decode test (seq_len=1)
        input_tensor = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)
        output_tensor = torch.empty(1, 1, hidden_size, dtype=torch.bfloat16)
        expert_ids = torch.randint(0, expert_num, (1, num_experts_per_tok), dtype=torch.int32)
        weights = torch.rand(1, 1, num_experts_per_tok, dtype=torch.float32)
        qlen_tensor = torch.tensor([1], dtype=torch.int32)

        # Warmup
        for _ in range(10):
            cpuinfer.submit(moe.forward_task(
                qlen_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                False,
            ))
            cpuinfer.sync()

        # Benchmark
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            cpuinfer.submit(moe.forward_task(
                qlen_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_tensor.data_ptr(),
                output_tensor.data_ptr(),
                False,
            ))
            cpuinfer.sync()
        end = time.perf_counter()

        avg_time = (end - start) / iterations * 1000  # ms
        print(f"{method} Decode (m=1): {avg_time:.4f} ms/iter ({iterations} iters)")


if __name__ == "__main__":
    print("Testing AVX2 FP8 and RAWINT4 MoE Kernels with Weight Pre-Decoding")
    print("="*70)

    try:
        # Test prefill performance
        fp8_time = test_fp8_moe()
        rawint4_time = test_rawint4_moe()

        # Test decode performance
        test_decode_performance()

        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"FP8 Prefill (1024 tokens): {fp8_time:.2f} ms/iter")
        print(f"RAWINT4 Prefill (1024 tokens): {rawint4_time:.2f} ms/iter")
        print("\nWeight pre-decoding should show improvement for both prefill and decode")
        print("by avoiding on-the-fly FP8/INT4 to FP32 conversion in every GEMM call.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)