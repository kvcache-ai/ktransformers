"""
Test script for GemmKernel224FP8 (FP8 MoE) kernel validation.

This script:
1. Generates random BF16 weights
2. Quantizes them to FP8 format with 128x128 block-wise scales
3. Runs the FP8 MoE kernel
4. Compares results with PyTorch reference using dequantized BF16 weights

FP8 format notes:
- Weight: FP8 (E4M3) stored as uint8, shape [expert_num, n, k]
- Scale: BF16, shape [expert_num, n // group_size, k // group_size], group_size=128
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
import kt_kernel_ext

torch.manual_seed(42)

# Model config
hidden_size = 7168
intermediate_size = 2048
max_len = 25600

expert_num = 16
num_experts_per_tok = 8

qlen = 1
layer_num = 1
CPUInfer = kt_kernel_ext.CPUInfer(40)
validation_iter = 1
fp8_group_size = 128  # FP8 uses 128x128 block quantization
debug_print_count = 16

physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()


def act_fn(x):
    """SiLU activation function"""
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    """Reference MLP computation in PyTorch"""
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    """Reference MoE computation in PyTorch"""
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return t_output


# FP8 E4M3 constants
FP8_E4M3_MAX = 448.0  # Maximum representable value in FP8 E4M3


def fp8_e4m3_to_float(fp8_val: int) -> float:
    """
    Convert FP8 E4M3 value to float.
    FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits
    """
    sign = (fp8_val >> 7) & 1
    exp = (fp8_val >> 3) & 0xF
    mant = fp8_val & 0x7

    if exp == 0:
        # Subnormal or zero
        if mant == 0:
            return -0.0 if sign else 0.0
        # Subnormal: value = (-1)^sign * 2^(-6) * (0.mant)
        return ((-1) ** sign) * (2 ** -6) * (mant / 8.0)
    elif exp == 15:
        # NaN (FP8 E4M3 doesn't have Inf, all exp=15 are NaN)
        return float('nan')
    else:
        # Normal: value = (-1)^sign * 2^(exp-7) * (1.mant)
        return ((-1) ** sign) * (2 ** (exp - 7)) * (1.0 + mant / 8.0)


def float_to_fp8_e4m3(val: float) -> int:
    """
    Convert float to FP8 E4M3 value.
    """
    if val != val:  # NaN
        return 0x7F  # NaN representation

    sign = 1 if val < 0 else 0
    val = abs(val)

    if val == 0:
        return sign << 7

    # Clamp to max representable value
    val = min(val, FP8_E4M3_MAX)

    # Find exponent
    import math
    if val < 2**-9:  # Subnormal threshold
        # Subnormal
        mant = int(round(val / (2**-9)))
        mant = min(mant, 7)
        return (sign << 7) | mant

    exp = int(math.floor(math.log2(val))) + 7
    exp = max(1, min(exp, 14))  # Clamp exponent to valid range

    # Calculate mantissa
    mant = int(round((val / (2 ** (exp - 7)) - 1.0) * 8))
    mant = max(0, min(mant, 7))

    # Handle overflow to next exponent
    if mant > 7:
        mant = 0
        exp += 1
        if exp > 14:
            exp = 14
            mant = 7

    return (sign << 7) | (exp << 3) | mant


def quantize_to_fp8_blockwise(weights: torch.Tensor, group_size: int = 128):
    """
    Quantize BF16/FP32 weights to FP8 with block-wise scaling.

    Args:
        weights: [expert_num, n, k] tensor in BF16/FP32
        group_size: Block size for quantization (default 128 for DeepSeek)

    Returns:
        fp8_weights: [expert_num, n, k] uint8 tensor
        scales: [expert_num, n // group_size, k // group_size] BF16 tensor (scale_inv)
    """
    weights_f32 = weights.to(torch.float32)
    e, n, k = weights_f32.shape

    assert n % group_size == 0, f"n ({n}) must be divisible by group_size ({group_size})"
    assert k % group_size == 0, f"k ({k}) must be divisible by group_size ({group_size})"

    n_blocks = n // group_size
    k_blocks = k // group_size

    # Reshape to [e, n_blocks, group_size, k_blocks, group_size]
    reshaped = weights_f32.view(e, n_blocks, group_size, k_blocks, group_size)
    # Move to [e, n_blocks, k_blocks, group_size, group_size] for block processing
    reshaped = reshaped.permute(0, 1, 3, 2, 4)

    # Calculate max abs per block
    max_abs = reshaped.abs().amax(dim=(-2, -1), keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-12)

    # Scale to FP8 range: scale = max_abs / FP8_MAX
    # We store scale_inv = scale (for dequantization: fp8 * scale)
    scales = (max_abs / FP8_E4M3_MAX).squeeze(-1).squeeze(-1)  # [e, n_blocks, k_blocks]

    # Quantize: q = round(val / scale)
    scaled = reshaped / (scales.unsqueeze(-1).unsqueeze(-1) + 1e-12)

    # Convert to FP8 E4M3 using vectorized approach
    # Clamp to FP8 representable range
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Simple quantization: round to nearest representable FP8 value
    # For simplicity, we use a lookup table approach
    fp8_q = torch.zeros_like(scaled, dtype=torch.uint8)

    # Vectorized FP8 quantization
    sign_mask = (scaled < 0).to(torch.uint8) << 7
    abs_scaled = scaled.abs()

    # Handle different ranges
    # Subnormal: 0 < |x| < 2^-6
    subnormal_mask = (abs_scaled > 0) & (abs_scaled < 2**-6)
    subnormal_mant = (abs_scaled / (2**-9)).round().clamp(0, 7).to(torch.uint8)

    # Normal values
    normal_mask = abs_scaled >= 2**-6
    log2_val = torch.log2(abs_scaled.clamp(min=2**-9))
    exp = (log2_val.floor() + 7).clamp(1, 14).to(torch.int32)
    mant = ((abs_scaled / (2.0 ** (exp.float() - 7)) - 1.0) * 8).round().clamp(0, 7).to(torch.uint8)

    # Combine
    fp8_q = torch.where(subnormal_mask, sign_mask | subnormal_mant, fp8_q)
    fp8_q = torch.where(normal_mask, sign_mask | (exp.to(torch.uint8) << 3) | mant, fp8_q)

    # Reshape back to [e, n, k]
    fp8_q = fp8_q.permute(0, 1, 3, 2, 4).reshape(e, n, k)

    # Scales shape: [e, n_blocks, k_blocks] -> store as [e, n_blocks, k_blocks]
    scales_bf16 = scales.to(torch.bfloat16).contiguous()

    return fp8_q.contiguous(), scales_bf16


def dequantize_fp8_blockwise(fp8_weights: torch.Tensor, scales: torch.Tensor, group_size: int = 128):
    """
    Dequantize FP8 weights back to BF16 for reference computation.

    Args:
        fp8_weights: [expert_num, n, k] uint8 tensor
        scales: [expert_num, n // group_size, k // group_size] BF16 tensor
        group_size: Block size

    Returns:
        dequantized: [expert_num, n, k] BF16 tensor
    """
    e, n, k = fp8_weights.shape
    n_blocks = n // group_size
    k_blocks = k // group_size

    # Convert FP8 to float
    # Build lookup table for FP8 E4M3 -> float
    fp8_lut = torch.tensor([fp8_e4m3_to_float(i) for i in range(256)], dtype=torch.float32)

    # Use lookup table
    fp8_float = fp8_lut[fp8_weights.to(torch.int64)]

    # Reshape for block-wise scaling
    fp8_reshaped = fp8_float.view(e, n_blocks, group_size, k_blocks, group_size)
    fp8_reshaped = fp8_reshaped.permute(0, 1, 3, 2, 4)  # [e, n_blocks, k_blocks, group_size, group_size]

    # Apply scales
    scales_f32 = scales.to(torch.float32).unsqueeze(-1).unsqueeze(-1)  # [e, n_blocks, k_blocks, 1, 1]
    dequantized = fp8_reshaped * scales_f32

    # Reshape back
    dequantized = dequantized.permute(0, 1, 3, 2, 4).reshape(e, n, k)

    return dequantized.to(torch.bfloat16).contiguous()


def build_random_fp8_weights():
    """
    Generate random BF16 weights and quantize to FP8.

    Returns:
        dict with fp8 weights, scales, and original bf16 for reference
    """
    torch.manual_seed(42)

    # Generate random BF16 weights with small values
    gate_proj = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16)
    up_proj = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16)
    down_proj = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 100.0).to(torch.bfloat16)

    # Quantize to FP8
    gate_fp8, gate_scales = quantize_to_fp8_blockwise(gate_proj, fp8_group_size)
    up_fp8, up_scales = quantize_to_fp8_blockwise(up_proj, fp8_group_size)
    down_fp8, down_scales = quantize_to_fp8_blockwise(down_proj, fp8_group_size)

    # Dequantize for reference computation
    gate_deq = dequantize_fp8_blockwise(gate_fp8, gate_scales, fp8_group_size)
    up_deq = dequantize_fp8_blockwise(up_fp8, up_scales, fp8_group_size)
    down_deq = dequantize_fp8_blockwise(down_fp8, down_scales, fp8_group_size)

    print(f"FP8 weights shape: gate={gate_fp8.shape}, up={up_fp8.shape}, down={down_fp8.shape}")
    print(f"Scales shape: gate={gate_scales.shape}, up={up_scales.shape}, down={down_scales.shape}")

    # Debug: Print FP8 weight and scale info for expert 0
    print("\n=== DEBUG: FP8 Weight and Scale Info (Expert 0) ===")
    print(f"gate_fp8[0] first 8x8 block:")
    for i in range(8):
        print(f"  row {i}: {gate_fp8[0, i, :8].numpy().tobytes().hex(' ')}")
    print(f"gate_fp8[0] stats: min={gate_fp8[0].min()}, max={gate_fp8[0].max()}")
    print(f"gate_scales[0] first 4x4 block:\n{gate_scales[0, :4, :4]}")
    print(f"gate_scales[0] stats: min={gate_scales[0].min()}, max={gate_scales[0].max()}")

    print(f"\nup_fp8[0] first 8x8 block:")
    for i in range(8):
        print(f"  row {i}: {up_fp8[0, i, :8].numpy().tobytes().hex(' ')}")
    print(f"up_fp8[0] stats: min={up_fp8[0].min()}, max={up_fp8[0].max()}")
    print(f"up_scales[0] first 4x4 block:\n{up_scales[0, :4, :4]}")
    print(f"up_scales[0] stats: min={up_scales[0].min()}, max={up_scales[0].max()}")

    print(f"\ndown_fp8[0] first 8x8 block:")
    for i in range(8):
        print(f"  row {i}: {down_fp8[0, i, :8].numpy().tobytes().hex(' ')}")
    print(f"down_fp8[0] stats: min={down_fp8[0].min()}, max={down_fp8[0].max()}")
    print(f"down_scales[0] first 4x4 block:\n{down_scales[0, :4, :4]}")
    print(f"down_scales[0] stats: min={down_scales[0].min()}, max={down_scales[0].max()}")

    return {
        "gate_fp8": gate_fp8.contiguous(),
        "up_fp8": up_fp8.contiguous(),
        "down_fp8": down_fp8.contiguous(),
        "gate_scales": gate_scales.contiguous(),
        "up_scales": up_scales.contiguous(),
        "down_scales": down_scales.contiguous(),
        "gate_deq": gate_deq.contiguous(),
        "up_deq": up_deq.contiguous(),
        "down_deq": down_deq.contiguous(),
    }


def build_moes_from_fp8_data(fp8_data: dict):
    """
    Build FP8 MoE modules from quantized data.
    """
    moes = []
    with torch.inference_mode(mode=True):
        for _ in range(layer_num):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.quant_config.bits = 8
            config.quant_config.group_size = fp8_group_size
            config.quant_config.zero_point = False

            # Set FP8 weight pointers
            config.gate_proj = fp8_data["gate_fp8"].data_ptr()
            config.up_proj = fp8_data["up_fp8"].data_ptr()
            config.down_proj = fp8_data["down_fp8"].data_ptr()

            # Set scale pointers
            config.gate_scale = fp8_data["gate_scales"].data_ptr()
            config.up_scale = fp8_data["up_scales"].data_ptr()
            config.down_scale = fp8_data["down_scales"].data_ptr()
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.AMXRAWFp8_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            moes.append(moe)
    return moes


def run_fp8_moe_test():
    """
    Run FP8 MoE validation test.
    """
    print("\n" + "=" * 70)
    print("FP8 MoE Kernel Validation Test")
    print("=" * 70)

    # Build FP8 weights
    print("\nGenerating and quantizing weights...")
    fp8_data = build_random_fp8_weights()

    # Build MoE modules
    print("\nBuilding FP8 MoE modules...")
    moes = build_moes_from_fp8_data(fp8_data)

    # Get dequantized weights for reference
    gate_deq = fp8_data["gate_deq"]
    up_deq = fp8_data["up_deq"]
    down_deq = fp8_data["down_deq"]

    diffs = []
    with torch.inference_mode(mode=True):
        for i in range(validation_iter):
            torch.manual_seed(100 + i)
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.ones((qlen, num_experts_per_tok), dtype=torch.float32).contiguous() / 100
            input_tensor = torch.full((qlen, hidden_size), 0.01, dtype=torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            moe = moes[i % layer_num]
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_tensor.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            # Reference computation using dequantized weights
            t_output = moe_torch(
                input_tensor, expert_ids, weights, gate_deq, up_deq, down_deq
            )

            t_output_flat = t_output.flatten()
            output_flat = output.flatten()

            diff = torch.mean(torch.abs(output_flat - t_output_flat)) / (torch.mean(torch.abs(t_output_flat)) + 1e-12)
            diffs.append(diff.item())
            print(f"Iteration {i}: relative L1 diff = {diff:.6f}")

            if i < 3:  # Print detailed output for first few iterations
                print(f"  kernel output: {output_flat[:debug_print_count]}")
                print(f"  torch output:  {t_output_flat[:debug_print_count]}")

    mean_diff = float(sum(diffs) / len(diffs))
    max_diff = float(max(diffs))
    min_diff = float(min(diffs))

    print("\n" + "=" * 70)
    print("FP8 MoE Test Results")
    print("=" * 70)
    print(f"Mean relative L1 diff: {mean_diff*100:.4f}%")
    print(f"Max relative L1 diff:  {max_diff*100:.4f}%")
    print(f"Min relative L1 diff:  {min_diff*100:.4f}%")

    # Pass/Fail criteria
    threshold = 15.0  # 15% relative error threshold for FP8
    if mean_diff * 100 < threshold:
        print(f"\nPASS: Mean error {mean_diff*100:.4f}% < {threshold}% threshold")
    else:
        print(f"\nFAIL: Mean error {mean_diff*100:.4f}% >= {threshold}% threshold")

    return {"mean": mean_diff, "max": max_diff, "min": min_diff}


if __name__ == "__main__":
    run_fp8_moe_test()
