"""
Test script for AMX_BF16_MOE_TP (native BF16 MoE) kernel validation.

This script:
1. Generates random BF16 weights
2. Runs the BF16 MoE kernel
3. Compares results with PyTorch reference

BF16 format notes:
- Weight: BF16 stored as ggml_bf16_t, shape [expert_num, n, k]
- No scales needed (native BF16 precision)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
from kt_kernel import kt_kernel_ext

torch.manual_seed(42)

# Model config
hidden_size = 2048
intermediate_size = 768
max_len = 25600

expert_num = 128
num_experts_per_tok = 8

qlen = 1
layer_num = 5
CPUInfer = kt_kernel_ext.CPUInfer(3)
validation_iter = 5
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


def build_bf16_weights():
    """
    Generate random BF16 weights.

    Returns:
        dict with BF16 weights for gate, up, down projections
    """
    torch.manual_seed(42)

    # Generate random BF16 weights with small values
    gate_proj = (
        (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )
    up_proj = (
        (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )
    down_proj = (
        (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 100.0)
        .to(torch.bfloat16)
        .contiguous()
    )

    print(f"BF16 weights shape: gate={gate_proj.shape}, up={up_proj.shape}, down={down_proj.shape}")

    # Debug: Print BF16 weight info for expert 0
    print("\n=== DEBUG: BF16 Weight Info (Expert 0) ===")
    print(f"gate_proj[0] first 8 values: {gate_proj[0, 0, :8]}")
    print(f"gate_proj[0] stats: min={gate_proj[0].min()}, max={gate_proj[0].max()}")
    print(f"up_proj[0] first 8 values: {up_proj[0, 0, :8]}")
    print(f"down_proj[0] first 8 values: {down_proj[0, 0, :8]}")

    return {
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
    }


def build_moes_from_bf16_data(bf16_data: dict):
    """
    Build BF16 MoE modules from BF16 weight data.
    """
    moes = []
    with torch.inference_mode(mode=True):
        for _ in range(layer_num):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len

            # Set BF16 weight pointers (no scales needed)
            config.gate_proj = bf16_data["gate_proj"].data_ptr()
            config.up_proj = bf16_data["up_proj"].data_ptr()
            config.down_proj = bf16_data["down_proj"].data_ptr()

            # No scales for BF16
            config.gate_scale = 0
            config.up_scale = 0
            config.down_scale = 0
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            moes.append(moe)
    return moes


def run_bf16_moe_test():
    """
    Run BF16 MoE validation test.
    """
    print("\n" + "=" * 70)
    print("BF16 MoE Kernel Validation Test")
    print("=" * 70)

    # Build BF16 weights
    print("\nGenerating BF16 weights...")
    bf16_data = build_bf16_weights()

    # Build MoE modules
    print("\nBuilding BF16 MoE modules...")
    moes = build_moes_from_bf16_data(bf16_data)

    # Get weights for reference computation
    gate_proj = bf16_data["gate_proj"]
    up_proj = bf16_data["up_proj"]
    down_proj = bf16_data["down_proj"]

    diffs = []
    with torch.inference_mode(mode=True):
        for i in range(validation_iter):
            torch.manual_seed(114514 + i)
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.randn((qlen, num_experts_per_tok), dtype=torch.float32).contiguous() / 10
            input_tensor = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() * 3
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

            assert not torch.isnan(output).any(), "NaN values detected in CPU expert output."
            assert not torch.isinf(output).any(), "Inf values detected in CPU expert output."

            # Reference computation using BF16 weights
            t_output = moe_torch(input_tensor, expert_ids, weights, gate_proj, up_proj, down_proj)

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
    print("BF16 MoE Test Results")
    print("=" * 70)
    print(f"Mean relative L1 diff: {mean_diff*100:.4f}%")
    print(f"Max relative L1 diff:  {max_diff*100:.4f}%")
    print(f"Min relative L1 diff:  {min_diff*100:.4f}%")

    # Pass/Fail criteria (BF16 should be very accurate, <5% error)
    threshold = 5.0
    if mean_diff * 100 < threshold:
        print(f"\nPASS: Mean error {mean_diff*100:.4f}% < {threshold}% threshold")
    else:
        print(f"\nFAIL: Mean error {mean_diff*100:.4f}% >= {threshold}% threshold")

    return {"mean": mean_diff, "max": max_diff, "min": min_diff}


if __name__ == "__main__":
    run_bf16_moe_test()
