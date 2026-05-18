#!/usr/bin/env python
# coding=utf-8
"""AVX2 BF16 MoE accuracy tests for KT-Kernel.

Tests accuracy of AVX2 BF16 MOE operations against torch reference.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from kt_kernel import kt_kernel_ext

# Small test parameters for fast validation
expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 128
validation_iter = 3
CPUINFER_PARAM = 60


def act_fn(x):
    """SiLU activation."""
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    """PyTorch reference MLP."""
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    """PyTorch reference MoE."""
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


def test_avx2_bf16_accuracy(qlen, label):
    """Test AVX2 BF16 MoE accuracy."""
    physical_to_logical_map = torch.tensor(range(expert_num), device="cpu", dtype=torch.int64).contiguous()
    CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        # Generate BF16 weights
        gate_proj = (
            (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0)
            .to(torch.bfloat16)
            .contiguous()
        )
        up_proj = (
            (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0)
            .to(torch.bfloat16)
            .contiguous()
        )
        down_proj = (
            (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 10.0)
            .to(torch.bfloat16)
            .contiguous()
        )

        # Create MOE config
        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        config.gate_scale = 0
        config.up_scale = 0
        config.down_scale = 0
        config.pool = CPUInfer.backend_

        # Create AVX2 BF16 MOE
        moe = kt_kernel_ext.moe.AVX2BF16_MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        print(f"\n--- {label} (qlen={qlen}) ---")
        for i in range(validation_iter):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)

            # Run AVX2 BF16 MOE
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            # Run torch reference (in float32 for accuracy)
            t_output = moe_torch(
                input_data.float(), expert_ids, weights, gate_proj.float(), up_proj.float(), down_proj.float()
            ).to(torch.bfloat16)

            # Calculate relative difference
            diff = torch.mean(torch.abs(output.float() - t_output.float())) / (
                torch.mean(torch.abs(t_output.float())) + 1e-8
            )
            print(f"  Iteration {i}: diff = {diff:.6f}")

            # BF16 should be very accurate (< 0.01)
            assert diff < 0.02, f"AVX2 BF16 accuracy test failed: diff={diff:.6f} >= 0.02"

    print(f"  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("AVX2 BF16 MoE Accuracy Test")
    print("=" * 60)

    try:
        # Test decode path (qlen=1)
        test_avx2_bf16_accuracy(qlen=1, label="Decode")

        # Test prefill path (qlen=16)
        test_avx2_bf16_accuracy(qlen=16, label="Prefill")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
