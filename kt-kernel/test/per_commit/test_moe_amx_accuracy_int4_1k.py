#!/usr/bin/env python
# coding=utf-8
"""AMX MOE INT4_1K accuracy tests for KT-Kernel.

Tests accuracy of AMX-accelerated INT4_1K group quantization MOE operations against torch reference.
"""

import os
import sys
import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Register this test for CPU CI with estimated runtime of 120 seconds
register_cpu_ci(est_time=120, suite="default")

# Check if dependencies are available
try:
    import torch
    import kt_kernel_ext
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)

# Test parameters (from original test_moe_amx.py)
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
qlen = 1
layer_num = 1
validation_iter = 2
k_group_size = 64
physical_to_logical_map = None


def act_fn(x):
    """Activation function for MoE."""
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    """PyTorch reference implementation of MLP."""
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    """PyTorch reference implementation of MoE."""
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
        expert_out = mlp_torch(
            tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i]
        )
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


@pytest.mark.cpu
def test_moe_amx_int4_1k_accuracy():
    """Test AMX INT4_1K MOE accuracy against PyTorch reference implementation."""
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")

    global physical_to_logical_map
    physical_to_logical_map = torch.tensor(
        data=range(expert_num), device="cpu", dtype=torch.int64
    ).contiguous()

    CPUInfer = kt_kernel_ext.CPUInfer(90)

    with torch.inference_mode(mode=True):
        # Initialize MoE layers
        gate_proj = (
            torch.randn(
                (expert_num, intermediate_size, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            .to("cpu")
            .contiguous()
        )
        up_proj = (
            torch.randn(
                (expert_num, intermediate_size, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            .to("cpu")
            .contiguous()
        )
        down_proj = (
            torch.randn(
                (expert_num, hidden_size, intermediate_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            .to("cpu")
            .contiguous()
        )

        # Create MOE config
        config = kt_kernel_ext.moe.MOEConfig(
            expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0
        )
        config.max_len = max_len
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        config.gate_scale = 0
        config.pool = CPUInfer.backend_

        # Configure INT4_1K quantization settings
        config.quant_config.bits = 4
        config.quant_config.group_size = k_group_size
        config.quant_config.zero_point = True

        # Initialize INT4_1K MOE
        moe = kt_kernel_ext.moe.AMXInt4_1KGroup_MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        # Run validation iterations
        for i in range(validation_iter):
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            input_data = input_data / 100

            # Run AMX MOE
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

            # Run torch reference
            t_output = moe_torch(
                input_data, expert_ids, weights, gate_proj, up_proj, down_proj
            )

            # Calculate relative difference
            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(
                torch.abs(t_output)
            )
            print(f"Iteration {i}, diff = {diff:.6f}")

            # INT4_1K should have diff < 0.35
            assert diff < 0.35, f"INT4_1K accuracy test failed: diff={diff:.6f} >= 0.35"


def run_all_tests():
    """Run all tests in this file (for standalone execution)."""
    if not HAS_DEPS:
        print(f"⚠ Dependencies not available: {import_error}")
        print("Skipping AMX MOE INT4_1K accuracy tests")
        return

    try:
        print("Running AMX MOE INT4_1K accuracy test...")
        test_moe_amx_int4_1k_accuracy()
        print("✓ AMX MOE INT4_1K accuracy test passed")
        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
