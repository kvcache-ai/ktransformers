#!/usr/bin/env python3
"""
Test script for FP8 and RAWINT4 AVX2 MoE kernels with pre-permutation caching.
Tests the performance improvement from pre-permutation optimization.
"""

import os
import sys
import time
import torch
import numpy as np

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

try:
    from kt_kernel import kt_kernel_ext
    from kt_kernel.experts import KTMoEWrapper
    print("Successfully imported kt_kernel")
except ImportError as e:
    print(f"Failed to import kt_kernel: {e}")
    sys.exit(1)


def test_fp8_moe():
    """Test FP8 MoE kernel with pre-permutation caching."""
    print("\n" + "="*60)
    print("Testing FP8 MoE Kernel")
    print("="*60)

    # Test parameters
    expert_num = 8
    hidden_size = 4096
    intermediate_size = 14336  # Multiple of 128 for block-wise quantization
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1

    # Create config
    config = kt_kernel_ext.GeneralMOEConfig()
    config.expert_num = expert_num
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_experts_per_tok = num_experts_per_tok
    config.cpuinfer_threads = cpuinfer_threads
    config.threadpool_count = threadpool_count
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

    # Create wrapper
    print("Creating FP8 MoE wrapper...")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=cpuinfer_threads,
        threadpool_count=threadpool_count,
        weight_path="",
        chunked_prefill_size=2048,
        method="FP8",
    )

    # Load weights
    print("Loading FP8 weights...")
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    wrapper.load_weights_from_tensors(
        gate_proj=torch.from_numpy(gate_quant),
        up_proj=torch.from_numpy(up_quant),
        down_proj=torch.from_numpy(down_quant),
        physical_to_logical_map_cpu=physical_to_logical_map,
    )

    # Test forward pass
    print("Running FP8 forward pass...")
    batch_size = 1
    seq_len = 1024  # Prefill length
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = wrapper.forward(input_tensor)

    # Benchmark
    iterations = 100
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iterations):
        output = wrapper.forward(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms
    print(f"FP8 MoE forward: {avg_time:.2f} ms/iter ({iterations} iters)")
    print(f"Output shape: {output.shape}")

    return avg_time


def test_rawint4_moe():
    """Test RAWINT4 MoE kernel with pre-permutation caching."""
    print("\n" + "="*60)
    print("Testing RAWINT4 MoE Kernel")
    print("="*60)

    # Test parameters
    expert_num = 8
    hidden_size = 4096
    intermediate_size = 14336  # Multiple of group_size
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1
    group_size = 32  # K-group size for RAWINT4

    # Create config
    config = kt_kernel_ext.GeneralMOEConfig()
    config.expert_num = expert_num
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_experts_per_tok = num_experts_per_tok
    config.cpuinfer_threads = cpuinfer_threads
    config.threadpool_count = threadpool_count
    config.quant_config = kt_kernel_ext.QuantConfig()
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False

    # Generate test data
    print("Generating RAWINT4 test weights...")
    np.random.seed(42)
    gate_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    up_proj = np.random.randn(expert_num, intermediate_size, hidden_size).astype(np.float32)
    down_proj = np.random.randn(expert_num, hidden_size, intermediate_size).astype(np.float32)

    # Quantize to RAWINT4 (signed int4, packed 2 per byte)
    def quantize_rawint4(weight, group_size=32):
        """Quantize FP32 weight to signed INT4 with K-group scales."""
        n, k = weight.shape
        assert k % group_size == 0, "k must be divisible by group_size"

        # Pack 2 int4 values per byte
        quantized = np.zeros((n, k // 2), dtype=np.uint8)
        scales = np.zeros((n, k // group_size), dtype=np.float32)

        for i in range(n):
            for g in range(k // group_size):
                col_start = g * group_size
                col_end = col_start + group_size
                block = weight[i, col_start:col_end]

                max_val = np.max(np.abs(block))
                scale = max_val / 7.0 if max_val > 0 else 1.0
                scales[i, g] = scale

                # Signed INT4 range: [-8, 7]
                q_block = np.round(block / scale).clip(-8, 7).astype(np.int8)

                # Pack 2 values per byte
                for j in range(0, group_size, 2):
                    byte_val = ((q_block[j] + 8) & 0x0F) | (((q_block[j+1] + 8) & 0x0F) << 4)
                    quantized[i, col_start // 2 + j // 2] = byte_val

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

    # Create wrapper
    print("Creating RAWINT4 MoE wrapper...")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=cpuinfer_threads,
        threadpool_count=threadpool_count,
        weight_path="",
        chunked_prefill_size=2048,
        method="RAWINT4",
    )

    # Load weights
    print("Loading RAWINT4 weights...")
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)
    wrapper.load_weights_from_tensors(
        gate_proj=torch.from_numpy(gate_quant),
        up_proj=torch.from_numpy(up_quant),
        down_proj=torch.from_numpy(down_quant),
        physical_to_logical_map_cpu=physical_to_logical_map,
    )

    # Test forward pass
    print("Running RAWINT4 forward pass...")
    batch_size = 1
    seq_len = 1024  # Prefill length
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        _ = wrapper.forward(input_tensor)

    # Benchmark
    iterations = 100
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(iterations):
        output = wrapper.forward(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.perf_counter()

    avg_time = (end - start) / iterations * 1000  # ms
    print(f"RAWINT4 MoE forward: {avg_time:.2f} ms/iter ({iterations} iters)")
    print(f"Output shape: {output.shape}")

    return avg_time


def test_decode_performance():
    """Test decode performance (m=1) where pre-permutation should NOT be used."""
    print("\n" + "="*60)
    print("Testing Decode Performance (m=1)")
    print("="*60)

    expert_num = 8
    hidden_size = 4096
    intermediate_size = 14336
    num_experts_per_tok = 2
    cpuinfer_threads = 4
    threadpool_count = 1

    # Test both FP8 and RAWINT4
    for method in ["FP8", "RAWINT4"]:
        print(f"\n--- {method} Decode ---")

        config = kt_kernel_ext.GeneralMOEConfig()
        config.expert_num = expert_num
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.num_experts_per_tok = num_experts_per_tok
        config.cpuinfer_threads = cpuinfer_threads
        config.threadpool_count = threadpool_count
        config.quant_config = kt_kernel_ext.QuantConfig()
        config.quant_config.group_size = 128 if method == "FP8" else 32
        config.quant_config.zero_point = False

        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path="",
            chunked_prefill_size=2048,
            method=method,
        )

        # Quick test with dummy weights
        physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64)

        if method == "FP8":
            gate_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size), dtype=np.uint8)
            up_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size), dtype=np.uint8)
            down_proj = np.random.randint(0, 256, (expert_num, hidden_size, intermediate_size), dtype=np.uint8)
            gate_scales = np.random.randn(expert_num, (intermediate_size // 128) * (hidden_size // 128)).astype(np.float32)
            up_scales = np.random.randn(expert_num, (intermediate_size // 128) * (hidden_size // 128)).astype(np.float32)
            down_scales = np.random.randn(expert_num, (hidden_size // 128) * (intermediate_size // 128)).astype(np.float32)
        else:
            gate_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=np.uint8)
            up_proj = np.random.randint(0, 256, (expert_num, intermediate_size, hidden_size // 2), dtype=np.uint8)
            down_proj = np.random.randint(0, 256, (expert_num, hidden_size, intermediate_size // 2), dtype=np.uint8)
            gate_scales = np.random.randn(expert_num, (intermediate_size // 32) * hidden_size).astype(np.float32)
            up_scales = np.random.randn(expert_num, (intermediate_size // 32) * hidden_size).astype(np.float32)
            down_scales = np.random.randn(expert_num, (hidden_size // 32) * intermediate_size).astype(np.float32)

        wrapper.load_weights_from_tensors(
            gate_proj=torch.from_numpy(gate_proj),
            up_proj=torch.from_numpy(up_proj),
            down_proj=torch.from_numpy(down_proj),
            physical_to_logical_map_cpu=physical_to_logical_map,
        )

        # Decode test (seq_len=1)
        input_tensor = torch.randn(1, 1, hidden_size, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = wrapper.forward(input_tensor)

        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            output = wrapper.forward(input_tensor)
        end = time.perf_counter()

        avg_time = (end - start) / iterations * 1000  # ms
        print(f"{method} Decode (m=1): {avg_time:.4f} ms/iter ({iterations} iters)")


if __name__ == "__main__":
    print("Testing AVX2 FP8 and RAWINT4 MoE Kernels with Pre-Permutation Caching")
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
        print("\nPre-permutation caching should show improvement for prefill (m >= 16)")
        print("and no overhead for decode (m < 16).")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)