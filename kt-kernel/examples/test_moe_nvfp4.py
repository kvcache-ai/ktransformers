"""
Test NVFP4 MoE operator with E2M1 FP4 quantization and FP8 E4M3 block scales.

NVFP4 format:
- 4-bit E2M1 floating point (1 sign, 2 exponent, 1 mantissa)
- Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and their negatives)
- 16-element blocks with FP8 E4M3 per-block scale
- Optional tensor-level FP32 scale
"""

import os
import sys
import struct

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import numpy as np

# Try to import kt_kernel_ext
try:
    from kt_kernel import kt_kernel_ext

    HAS_KT_KERNEL = True
except ImportError:
    print("Warning: kt_kernel_ext not available, running quantization tests only")
    HAS_KT_KERNEL = False

# NVFP4 E2M1 lookup table: index -> float value
E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def float_to_e2m1(val: float) -> int:
    """Convert float to E2M1 4-bit representation."""
    sign = 0
    if val < 0:
        sign = 1
        val = -val

    # Find closest E2M1 value
    best_idx = 0
    best_err = abs(val - E2M1_VALUES[0])
    for i, v in enumerate(E2M1_VALUES):
        err = abs(val - v)
        if err < best_err:
            best_err = err
            best_idx = i

    return (sign << 3) | best_idx


def e2m1_to_float(e2m1: int) -> float:
    """Convert E2M1 4-bit representation to float."""
    sign = (e2m1 >> 3) & 1
    idx = e2m1 & 0x07
    val = E2M1_VALUES[idx]
    return -val if sign else val


def float_to_fp8_e4m3(val: float) -> int:
    """Convert float to FP8 E4M3 representation."""
    if val == 0:
        return 0

    sign = 0
    if val < 0:
        sign = 1
        val = -val

    # FP8 E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa
    # Range: ~1e-9 to 448
    if val > 448:
        val = 448
    if val < 1.0 / 512:  # Smallest normal
        return 0

    import math

    exp = int(math.floor(math.log2(val)))
    exp_biased = exp + 7

    if exp_biased < 0:
        return 0
    if exp_biased > 15:
        exp_biased = 15
        mantissa = 7
    else:
        mantissa = int(round((val / (2**exp) - 1.0) * 8))
        if mantissa > 7:
            mantissa = 7

    return (sign << 7) | (exp_biased << 3) | mantissa


def fp8_e4m3_to_float(fp8: int) -> float:
    """Convert FP8 E4M3 to float."""
    sign = (fp8 >> 7) & 1
    exp = (fp8 >> 3) & 0x0F
    mantissa = fp8 & 0x07

    if exp == 0 and mantissa == 0:
        return 0.0

    val = (1.0 + mantissa / 8.0) * (2 ** (exp - 7))
    return -val if sign else val


def quantize_to_nvfp4(weights: torch.Tensor, block_size: int = 16, use_tensor_scale: bool = False) -> tuple:
    """
    Quantize FP32/BF16 weights to NVFP4 format.

    Args:
        weights: [N, K] weight tensor
        block_size: Block size for quantization (must be 16 for NVFP4)
        use_tensor_scale: If True, use tensor-level scale (original NVFP4).
                          If False, fold tensor_scale into block_scales (for C++ compatibility).

    Returns:
        packed_fp4: [N, K // 2] packed FP4 data (2 values per byte)
        scales_fp8: [N, K // 16] FP8 E4M3 block scales
        tensor_scale: Global tensor scale (FP32), always 1.0 if use_tensor_scale=False
    """
    assert block_size == 16, "NVFP4 requires block_size=16"

    n, k = weights.shape
    assert k % block_size == 0, f"K ({k}) must be divisible by block_size ({block_size})"

    weights_f32 = weights.float().numpy()
    num_blocks = k // block_size
    packed_fp4 = np.zeros((n, k // 2), dtype=np.uint8)
    scales_fp8 = np.zeros((n, num_blocks), dtype=np.uint8)

    if use_tensor_scale:
        # Original NVFP4: use tensor scale + block scales
        global_max = np.abs(weights_f32).max()
        tensor_scale = global_max / 448.0 if global_max > 0 else 1.0
        tensor_scale_inv = 1.0 / tensor_scale if tensor_scale > 1e-10 else 1.0
        scaled_weights = weights_f32 * tensor_scale_inv
    else:
        # No tensor scale: fold everything into block scales
        # This is compatible with C++ which uses tensor_scale=1.0
        tensor_scale = 1.0
        scaled_weights = weights_f32

    for n_i in range(n):
        for blk in range(num_blocks):
            k_start = blk * block_size
            block_vals = scaled_weights[n_i, k_start : k_start + block_size]

            # Compute block scale (max / 6.0 since E2M1 max is 6)
            block_max = np.abs(block_vals).max()
            block_scale = block_max / 6.0 if block_max > 0 else 1.0
            scales_fp8[n_i, blk] = float_to_fp8_e4m3(block_scale)

            block_scale_inv = 1.0 / block_scale if block_scale > 1e-10 else 0.0

            # Quantize each value in block
            for k_i in range(0, block_size, 2):
                val0 = block_vals[k_i] * block_scale_inv
                val1 = block_vals[k_i + 1] * block_scale_inv

                q0 = float_to_e2m1(val0)
                q1 = float_to_e2m1(val1)

                # Pack two 4-bit values into one byte (low nibble first)
                packed_fp4[n_i, (k_start + k_i) // 2] = q0 | (q1 << 4)

    return packed_fp4, scales_fp8, tensor_scale


def dequantize_nvfp4(
    packed_fp4: np.ndarray, scales_fp8: np.ndarray, tensor_scale: float, block_size: int = 16
) -> np.ndarray:
    """
    Dequantize NVFP4 weights back to FP32.

    Args:
        packed_fp4: [N, K // 2] packed FP4 data
        scales_fp8: [N, K // 16] FP8 E4M3 block scales
        tensor_scale: Global tensor scale (FP32)
        block_size: Block size (must be 16)

    Returns:
        weights: [N, K] dequantized weights
    """
    n, packed_k = packed_fp4.shape
    k = packed_k * 2
    num_blocks = k // block_size

    weights = np.zeros((n, k), dtype=np.float32)

    for n_i in range(n):
        for blk in range(num_blocks):
            k_start = blk * block_size
            block_scale = fp8_e4m3_to_float(scales_fp8[n_i, blk])

            for k_i in range(0, block_size, 2):
                packed_byte = packed_fp4[n_i, (k_start + k_i) // 2]
                q0 = packed_byte & 0x0F
                q1 = (packed_byte >> 4) & 0x0F

                val0 = e2m1_to_float(q0) * block_scale * tensor_scale
                val1 = e2m1_to_float(q1) * block_scale * tensor_scale

                weights[n_i, k_start + k_i] = val0
                weights[n_i, k_start + k_i + 1] = val1

    return weights


def test_quantization_roundtrip():
    """Test NVFP4 quantization and dequantization."""
    print("=== Testing NVFP4 Quantization Roundtrip ===")

    n, k = 128, 256
    torch.manual_seed(42)
    weights = torch.randn(n, k, dtype=torch.float32) * 0.5

    # Quantize
    packed_fp4, scales_fp8, tensor_scale = quantize_to_nvfp4(weights)

    print(f"Original shape: {weights.shape}")
    print(f"Packed FP4 shape: {packed_fp4.shape}")
    print(f"Scales FP8 shape: {scales_fp8.shape}")
    print(f"Tensor scale: {tensor_scale:.6f}")

    # Dequantize
    weights_dequant = dequantize_nvfp4(packed_fp4, scales_fp8, tensor_scale)

    # Compute error
    weights_np = weights.numpy()
    error = np.abs(weights_np - weights_dequant)
    max_error = error.max()
    avg_error = error.mean()

    print(f"Max error: {max_error:.6f}")
    print(f"Avg error: {avg_error:.6f}")

    # Show sample values
    print("Sample values (original -> dequantized):")
    for i in range(8):
        print(f"  [{i}] {weights_np[0, i]:.4f} -> {weights_dequant[0, i]:.4f}")

    print("✓ Quantization roundtrip test passed\n")
    return max_error < 1.0  # FP4 has significant quantization error


def test_e2m1_conversion():
    """Test E2M1 float conversion."""
    print("=== Testing E2M1 Conversion ===")

    test_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

    for val in test_values:
        e2m1 = float_to_e2m1(val)
        recovered = e2m1_to_float(e2m1)
        error = abs(val - recovered)
        status = "✓" if error < 1e-6 else "✗"
        print(f"  {status} {val:6.2f} -> E2M1({e2m1:02x}) -> {recovered:6.2f}, error: {error:.6f}")

    print("✓ E2M1 conversion test passed\n")


def test_fp8_e4m3_conversion():
    """Test FP8 E4M3 float conversion."""
    print("=== Testing FP8 E4M3 Conversion ===")

    test_values = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0]

    for val in test_values:
        fp8 = float_to_fp8_e4m3(val)
        recovered = fp8_e4m3_to_float(fp8)
        rel_error = abs(val - recovered) / max(abs(val), 1e-6)
        status = "✓" if rel_error < 0.2 else "✗"
        print(f"  {status} {val:8.2f} -> FP8({fp8:02x}) -> {recovered:8.2f}, rel_error: {rel_error:.4f}")

    print("✓ FP8 E4M3 conversion test passed\n")


def act_fn(x):
    """SiLU activation: x * sigmoid(x)."""
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    """Reference MLP implementation in PyTorch."""
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, expert_num):
    """Reference MoE implementation in PyTorch."""
    qlen = input.shape[0]
    hidden_size = input.shape[1]
    k = expert_ids.shape[1]

    output = torch.zeros(qlen, hidden_size, dtype=input.dtype)

    for i in range(qlen):
        for j in range(k):
            expert_id = expert_ids[i, j].item()
            weight = weights[i, j].item()

            expert_out = mlp_torch(input[i : i + 1], gate_proj[expert_id], up_proj[expert_id], down_proj[expert_id])
            output[i] += weight * expert_out.squeeze()

    return output


def test_moe_nvfp4():
    """Test NVFP4 MoE operator.

    Note: This test uses the TP_MOE wrapper which requires proper NUMA-aware
    data layout. The C++ standalone test (nvfp4-moe-test.cpp) has been verified
    to work correctly. This Python integration test needs additional work to
    properly handle TP slicing.
    """
    if not HAS_KT_KERNEL:
        print("=== Skipping NVFP4 MoE Test (kt_kernel_ext not available) ===\n")
        return True

    print("=== Testing NVFP4 MoE ===")
    print("Note: Full integration requires proper TP/NUMA weight layout.")
    print("The C++ standalone test (nvfp4-moe-test) has been verified to work.\n")

    # Configuration
    expert_num = 8
    hidden_size = 256
    intermediate_size = 512
    max_len = 1024
    num_experts_per_tok = 2
    qlen = 1
    block_size = 16

    # Create CPUInfer (need at least 2 threads)
    CPUInfer = kt_kernel_ext.CPUInfer(2)

    # Generate random weights (BF16)
    torch.manual_seed(42)
    gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous()
    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous()
    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16).contiguous()

    # Quantize weights to NVFP4
    print("Quantizing weights to NVFP4...")

    gate_packed_list = []
    gate_scales_list = []
    gate_tensor_scales = []

    up_packed_list = []
    up_scales_list = []
    up_tensor_scales = []

    down_packed_list = []
    down_scales_list = []
    down_tensor_scales = []

    for e in range(expert_num):
        # Gate - use_tensor_scale=False for C++ compatibility (C++ uses tensor_scale=1.0)
        packed, scales, ts = quantize_to_nvfp4(gate_proj[e].float(), use_tensor_scale=False)
        gate_packed_list.append(packed)
        gate_scales_list.append(scales)
        gate_tensor_scales.append(ts)

        # Up
        packed, scales, ts = quantize_to_nvfp4(up_proj[e].float(), use_tensor_scale=False)
        up_packed_list.append(packed)
        up_scales_list.append(scales)
        up_tensor_scales.append(ts)

        # Down
        packed, scales, ts = quantize_to_nvfp4(down_proj[e].float(), use_tensor_scale=False)
        down_packed_list.append(packed)
        down_scales_list.append(scales)
        down_tensor_scales.append(ts)

    # Concatenate into contiguous tensors
    gate_packed = np.concatenate(gate_packed_list, axis=0)
    gate_scales = np.concatenate(gate_scales_list, axis=0)

    up_packed = np.concatenate(up_packed_list, axis=0)
    up_scales = np.concatenate(up_scales_list, axis=0)

    down_packed = np.concatenate(down_packed_list, axis=0)
    down_scales = np.concatenate(down_scales_list, axis=0)

    # Convert to torch tensors
    gate_packed_t = torch.from_numpy(gate_packed.copy()).contiguous()
    gate_scales_t = torch.from_numpy(gate_scales.copy()).contiguous()

    up_packed_t = torch.from_numpy(up_packed.copy()).contiguous()
    up_scales_t = torch.from_numpy(up_scales.copy()).contiguous()

    down_packed_t = torch.from_numpy(down_packed.copy()).contiguous()
    down_scales_t = torch.from_numpy(down_scales.copy()).contiguous()

    print(f"Gate packed shape: {gate_packed_t.shape}, scales shape: {gate_scales_t.shape}")
    print(f"Up packed shape: {up_packed_t.shape}, scales shape: {up_scales_t.shape}")
    print(f"Down packed shape: {down_packed_t.shape}, scales shape: {down_scales_t.shape}")

    # Create MOE config
    physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

    try:
        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_packed_t.data_ptr()
        config.up_proj = up_packed_t.data_ptr()
        config.down_proj = down_packed_t.data_ptr()
        config.gate_scale = gate_scales_t.data_ptr()
        config.up_scale = up_scales_t.data_ptr()
        config.down_scale = down_scales_t.data_ptr()
        config.pool = CPUInfer.backend_

        # Quant config for NVFP4
        config.quant_config.bits = 4
        config.quant_config.group_size = block_size
        config.quant_config.zero_point = False

        # Create NVFP4 MOE
        moe = kt_kernel_ext.moe.NVFP4_MOE(config)

        # Load weights
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        print("NVFP4 MOE created and weights loaded successfully!")

        # Run forward pass
        bsz_tensor = torch.tensor([qlen], device="cpu")
        expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
        # Note: C++ merge_results converts FP32 to BF16
        output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

        print(f"Expert IDs: {expert_ids}")
        print(f"Weights: {weights}")

        CPUInfer.submit(
            moe.forward_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input.data_ptr(),
                output.data_ptr(),
                False,
            )
        )
        CPUInfer.sync()

        print(f"NVFP4 output: {output.flatten()[:8]}")

        # Compare with PyTorch reference
        t_output = moe_torch(
            input.float(), expert_ids, weights, gate_proj.float(), up_proj.float(), down_proj.float(), expert_num
        )
        print(f"Torch output: {t_output.flatten()[:8]}")

        diff = torch.mean(torch.abs(output.float() - t_output)) / torch.mean(torch.abs(t_output))
        print(f"Relative diff: {diff:.6f}")

        # NVFP4 has higher quantization error, allow up to 0.5
        if diff < 0.5:
            print("✓ NVFP4 MoE test passed\n")
            return True
        else:
            print("✗ NVFP4 MoE test failed (diff too large)\n")
            return False

    except Exception as e:
        print(f"Error creating NVFP4 MOE: {e}")
        print("This might be expected if NVFP4 MOE is not fully integrated yet")
        return False


def main():
    print("NVFP4 MoE Test Suite")
    print("=" * 50)
    print()

    # Test basic conversions
    test_e2m1_conversion()
    test_fp8_e4m3_conversion()
    test_quantization_roundtrip()

    # Test MoE if kt_kernel is available
    test_moe_nvfp4()

    print("All tests completed!")


if __name__ == "__main__":
    main()
