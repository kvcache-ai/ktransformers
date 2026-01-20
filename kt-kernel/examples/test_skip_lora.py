#!/usr/bin/env python
# coding=utf-8
"""
Unit test for SkipLoRA feature in AMX_SFT_MOE_TP.

This test verifies that when SkipLoRA=true (method="AMXBF16_SFT_SkipLoRA"):
1. Forward pass works identically (LoRA is still used in forward)
2. Backward pass only computes base weight contribution to grad_input
3. LoRA weight gradients are NOT computed (should remain zero)

Usage:
    python test_skip_lora.py [--tp-count 1] [--threshold 0.05]
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch

# Try to import kt_kernel
try:
    from kt_kernel.experts import KTMoEWrapper

    HAS_KT_KERNEL = True
except ImportError as e:
    HAS_KT_KERNEL = False
    print(f"WARNING: kt_kernel not available: {e}")


# ============================================================================
# Configuration
# ============================================================================
DEFAULT_TP_COUNT = 1
DEFAULT_THRESHOLD = 0.05

# Test dimensions (smaller for faster testing)
TEST_CONFIG = {
    "expert_num": 8,
    "hidden_size": 256,  # Smaller for faster testing
    "intermediate_size": 512,
    "qlen": 32,
    "k": 2,
    "num_threads": 8,
    "max_len": 256,
}

# LoRA configuration
LORA_RANK = 8
LORA_ALPHA = 16
LORA_SCALING = LORA_ALPHA / LORA_RANK

# Weight scaling for numerical stability
WEIGHT_SCALE = 0.01
INPUT_SCALE = 0.1


# ============================================================================
# Python Reference Implementation
# ============================================================================


def silu(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


def silu_backward(gate_out, up_out, grad_intermediate):
    """Backward pass for SiLU activation"""
    sigmoid_gate = torch.sigmoid(gate_out)
    silu_gate = gate_out * sigmoid_gate
    grad_up_out = grad_intermediate * silu_gate
    silu_grad = sigmoid_gate * (1 + gate_out - gate_out * sigmoid_gate)
    grad_gate_out = grad_intermediate * up_out * silu_grad
    return grad_gate_out, grad_up_out


class PythonMoEReference:
    """Python reference implementation for MoE with LoRA"""

    def __init__(
        self,
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        lora_scaling,
    ):
        self.gate_proj = gate_proj.float()
        self.up_proj = up_proj.float()
        self.down_proj = down_proj.float()
        self.gate_lora_a = gate_lora_a.float()
        self.gate_lora_b = gate_lora_b.float()
        self.up_lora_a = up_lora_a.float()
        self.up_lora_b = up_lora_b.float()
        self.down_lora_a = down_lora_a.float()
        self.down_lora_b = down_lora_b.float()
        self.lora_scaling = lora_scaling
        self.expert_num = gate_proj.shape[0]
        self.intermediate_size = gate_proj.shape[1]
        self.hidden_size = gate_proj.shape[2]
        self.lora_rank = gate_lora_a.shape[1]

    def forward_with_cache(self, input_tensor, expert_ids, routing_weights):
        """Forward pass returning cache for backward"""
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]
        input_f = input_tensor.float()

        output = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)
        forward_cache = {}

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                weight = routing_weights[i, j].item()
                x = input_f[i : i + 1]

                # Gate
                gate_base = torch.mm(x, self.gate_proj[eid].t())
                gate_lora = torch.mm(torch.mm(x, self.gate_lora_a[eid].t()), self.gate_lora_b[eid].t())
                gate_out = gate_base + gate_lora * self.lora_scaling

                # Up
                up_base = torch.mm(x, self.up_proj[eid].t())
                up_lora = torch.mm(torch.mm(x, self.up_lora_a[eid].t()), self.up_lora_b[eid].t())
                up_out = up_base + up_lora * self.lora_scaling

                # Activation
                act_out = silu(gate_out) * up_out

                # Down
                down_base = torch.mm(act_out, self.down_proj[eid].t())
                down_lora = torch.mm(torch.mm(act_out, self.down_lora_a[eid].t()), self.down_lora_b[eid].t())
                down_out = down_base + down_lora * self.lora_scaling

                output[i] += down_out.squeeze() * weight

                # Store cache
                cache_key = f"e{eid}_i{i}_j{j}"
                forward_cache[cache_key] = {"gate_out": gate_out, "up_out": up_out, "act_out": act_out}

        return output, forward_cache

    def backward_base_only(self, grad_output, input_tensor, expert_ids, routing_weights, forward_cache):
        """
        Backward pass computing ONLY base weight contribution to grad_input.
        This is what SkipLoRA should produce.
        """
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]
        grad_output_f = grad_output.float()

        grad_input = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                weight = routing_weights[i, j].item()
                grad_out = grad_output_f[i : i + 1] * weight

                # Get forward cache
                cache_key = f"e{eid}_i{i}_j{j}"
                gate_out = forward_cache[cache_key]["gate_out"]
                up_out = forward_cache[cache_key]["up_out"]

                # backward_down: grad_intermediate = grad_out @ down_proj (base only)
                grad_intermediate = torch.mm(grad_out, self.down_proj[eid])

                # backward_activation
                grad_gate_out, grad_up_out = silu_backward(gate_out, up_out, grad_intermediate)

                # backward_gate_up: grad_input = grad_gate_out @ gate_proj + grad_up_out @ up_proj (base only)
                grad_input_gate = torch.mm(grad_gate_out, self.gate_proj[eid])
                grad_input_up = torch.mm(grad_up_out, self.up_proj[eid])

                grad_input[i] += (grad_input_gate + grad_input_up).squeeze()

        return grad_input

    def backward_full(self, grad_output, input_tensor, expert_ids, routing_weights, forward_cache):
        """
        Full backward pass including LoRA contribution to grad_input.
        """
        qlen = input_tensor.shape[0]
        k = expert_ids.shape[1]
        grad_output_f = grad_output.float()

        grad_input = torch.zeros(qlen, self.hidden_size, dtype=torch.float32)

        for i in range(qlen):
            for j in range(k):
                eid = expert_ids[i, j].item()
                weight = routing_weights[i, j].item()
                grad_out = grad_output_f[i : i + 1] * weight

                # Get forward cache
                cache_key = f"e{eid}_i{i}_j{j}"
                gate_out = forward_cache[cache_key]["gate_out"]
                up_out = forward_cache[cache_key]["up_out"]

                # backward_down: grad_intermediate = grad_out @ down_proj (base only, LoRA doesn't contribute)
                grad_intermediate = torch.mm(grad_out, self.down_proj[eid])

                # backward_activation
                grad_gate_out, grad_up_out = silu_backward(gate_out, up_out, grad_intermediate)

                # backward_gate_up: include LoRA contribution
                # Base
                grad_input_gate_base = torch.mm(grad_gate_out, self.gate_proj[eid])
                grad_input_up_base = torch.mm(grad_up_out, self.up_proj[eid])

                # LoRA
                grad_input_gate_lora = (
                    torch.mm(torch.mm(grad_gate_out, self.gate_lora_b[eid]), self.gate_lora_a[eid]) * self.lora_scaling
                )
                grad_input_up_lora = (
                    torch.mm(torch.mm(grad_up_out, self.up_lora_b[eid]), self.up_lora_a[eid]) * self.lora_scaling
                )

                grad_input[i] += (
                    grad_input_gate_base + grad_input_up_base + grad_input_gate_lora + grad_input_up_lora
                ).squeeze()

        return grad_input


# ============================================================================
# Test Functions
# ============================================================================


def compare_tensors(name, tensor1, tensor2, threshold):
    """Compare two tensors and print results"""
    t1 = tensor1.float().numpy() if isinstance(tensor1, torch.Tensor) else tensor1
    t2 = tensor2.float().numpy() if isinstance(tensor2, torch.Tensor) else tensor2

    if t1.shape != t2.shape:
        print(f"\033[91m[FAIL]\033[0m {name} - Shape mismatch: {t1.shape} vs {t2.shape}")
        return False

    abs_diff = np.abs(t1 - t2)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    rel_error = mean_abs_diff / (np.mean(np.abs(t2)) + 1e-12)

    t1_nan = np.sum(np.isnan(t1))
    t1_inf = np.sum(np.isinf(t1))

    passed = rel_error < threshold and t1_nan == 0 and t1_inf == 0

    if passed:
        print(f"\033[92m[PASS]\033[0m {name} - rel_error: {rel_error:.2e}, max_abs_diff: {max_abs_diff:.2e}")
    else:
        print(f"\033[91m[FAIL]\033[0m {name} - rel_error: {rel_error:.2e}, max_abs_diff: {max_abs_diff:.2e}")
        print(f"    t1 mean: {np.mean(t1):.6e}, t2 mean: {np.mean(t2):.6e}")
        print(f"    t1 NaN: {t1_nan}, Inf: {t1_inf}")

    return passed


def check_zeros(name, tensor, threshold=1e-10):
    """Check if tensor is all zeros"""
    t = tensor.float().numpy() if isinstance(tensor, torch.Tensor) else tensor
    max_val = np.max(np.abs(t))

    if max_val < threshold:
        print(f"\033[92m[PASS]\033[0m {name} - All zeros (max: {max_val:.2e})")
        return True
    else:
        print(f"\033[91m[FAIL]\033[0m {name} - NOT all zeros (max: {max_val:.2e}, mean: {np.mean(np.abs(t)):.2e})")
        return False


def test_skip_lora(tp_count, threshold):
    """Main test function for SkipLoRA"""
    print("=" * 80)
    print("Testing SkipLoRA Feature")
    print("=" * 80)

    if not HAS_KT_KERNEL:
        print("\033[91mERROR: kt_kernel not available, cannot run test\033[0m")
        return False

    config = TEST_CONFIG
    torch.manual_seed(42)

    # Initialize weights
    print("\n[1] Initializing weights...")
    gate_proj = (
        torch.rand(config["expert_num"], config["intermediate_size"], config["hidden_size"], dtype=torch.bfloat16)
        * WEIGHT_SCALE
    ).contiguous()
    up_proj = (
        torch.rand(config["expert_num"], config["intermediate_size"], config["hidden_size"], dtype=torch.bfloat16)
        * WEIGHT_SCALE
    ).contiguous()
    down_proj = (
        torch.rand(config["expert_num"], config["hidden_size"], config["intermediate_size"], dtype=torch.bfloat16)
        * WEIGHT_SCALE
    ).contiguous()

    gate_lora_a = (
        torch.rand(config["expert_num"], LORA_RANK, config["hidden_size"], dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    gate_lora_b = (
        torch.rand(config["expert_num"], config["intermediate_size"], LORA_RANK, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_a = (
        torch.rand(config["expert_num"], LORA_RANK, config["hidden_size"], dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    up_lora_b = (
        torch.rand(config["expert_num"], config["intermediate_size"], LORA_RANK, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_a = (
        torch.rand(config["expert_num"], LORA_RANK, config["intermediate_size"], dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()
    down_lora_b = (
        torch.rand(config["expert_num"], config["hidden_size"], LORA_RANK, dtype=torch.bfloat16) * WEIGHT_SCALE
    ).contiguous()

    # Generate test data
    print("\n[2] Generating test data...")
    input_tensor = (
        torch.rand((config["qlen"], config["hidden_size"]), dtype=torch.bfloat16) * INPUT_SCALE
    ).contiguous()
    expert_ids = torch.stack(
        [torch.randperm(config["expert_num"])[: config["k"]] for _ in range(config["qlen"])]
    ).contiguous()
    routing_weights = torch.rand(config["qlen"], config["k"], dtype=torch.float).contiguous()
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    grad_output = (torch.rand((config["qlen"], config["hidden_size"]), dtype=torch.bfloat16) * INPUT_SCALE).contiguous()

    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Expert IDs shape: {expert_ids.shape}")
    print(f"  Grad output shape: {grad_output.shape}")

    # Create Python reference
    print("\n[3] Creating Python reference...")
    py_ref = PythonMoEReference(
        gate_proj,
        up_proj,
        down_proj,
        gate_lora_a,
        gate_lora_b,
        up_lora_a,
        up_lora_b,
        down_lora_a,
        down_lora_b,
        LORA_SCALING,
    )

    # Run Python reference forward and backward
    print("\n[4] Running Python reference...")
    py_output, py_cache = py_ref.forward_with_cache(input_tensor, expert_ids, routing_weights)
    py_grad_input_base_only = py_ref.backward_base_only(
        grad_output, input_tensor, expert_ids, routing_weights, py_cache
    )
    py_grad_input_full = py_ref.backward_full(grad_output, input_tensor, expert_ids, routing_weights, py_cache)

    print(f"  Python forward output mean: {py_output.mean():.6e}")
    print(f"  Python grad_input (base only) mean: {py_grad_input_base_only.mean():.6e}")
    print(f"  Python grad_input (full) mean: {py_grad_input_full.mean():.6e}")

    # Create KTMoEWrapper instances
    print("\n[5] Creating C++ MoE instances via KTMoEWrapper...")

    # Create normal MoE (AMXBF16_SFT)
    wrapper_normal = KTMoEWrapper(
        layer_idx=0,
        num_experts=config["expert_num"],
        num_experts_per_tok=config["k"],
        hidden_size=config["hidden_size"],
        moe_intermediate_size=config["intermediate_size"],
        num_gpu_experts=0,
        cpuinfer_threads=config["num_threads"],
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=config["max_len"],
        method="AMXBF16_SFT",
        mode="sft",
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        max_cache_depth=2,
    )

    # Create SkipLoRA MoE (AMXBF16_SFT_SkipLoRA)
    wrapper_skip = KTMoEWrapper(
        layer_idx=0,
        num_experts=config["expert_num"],
        num_experts_per_tok=config["k"],
        hidden_size=config["hidden_size"],
        moe_intermediate_size=config["intermediate_size"],
        num_gpu_experts=0,
        cpuinfer_threads=config["num_threads"],
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=config["max_len"],
        method="AMXBF16_SFT_SkipLoRA",
        mode="sft",
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        max_cache_depth=2,
    )

    # Load weights
    print("\n[6] Loading weights...")
    physical_to_logical_map = torch.arange(config["expert_num"], dtype=torch.int64)

    wrapper_normal.gate_proj = gate_proj
    wrapper_normal.up_proj = up_proj
    wrapper_normal.down_proj = down_proj
    wrapper_normal.load_weights(physical_to_logical_map)
    wrapper_normal.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    wrapper_skip.gate_proj = gate_proj
    wrapper_skip.up_proj = up_proj
    wrapper_skip.down_proj = down_proj
    wrapper_skip.load_weights(physical_to_logical_map)
    wrapper_skip.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    # Run forward on both
    print("\n[7] Running C++ forward...")
    output_normal = wrapper_normal.forward_sft(input_tensor, expert_ids, routing_weights, save_for_backward=True)
    output_skip = wrapper_skip.forward_sft(input_tensor, expert_ids, routing_weights, save_for_backward=True)

    print(f"  Normal forward output mean: {output_normal.float().mean():.6e}")
    print(f"  SkipLoRA forward output mean: {output_skip.float().mean():.6e}")

    # Run backward on both
    print("\n[8] Running C++ backward...")
    grad_input_normal, grad_loras_normal = wrapper_normal.backward(grad_output)
    grad_input_skip, grad_loras_skip = wrapper_skip.backward(grad_output)

    print(f"  Normal grad_input mean: {grad_input_normal.float().mean():.6e}")
    print(f"  SkipLoRA grad_input mean: {grad_input_skip.float().mean():.6e}")

    # ============================================================================
    # Comparisons
    # ============================================================================
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)

    all_passed = True

    # Test 1: Forward outputs should be identical
    print("\n[Test 1] Forward output comparison (normal vs SkipLoRA)")
    passed = compare_tensors("Forward output (normal vs skip)", output_normal, output_skip, threshold)
    all_passed = all_passed and passed

    # Test 2: Forward output vs Python reference
    print("\n[Test 2] Forward output vs Python reference")
    passed = compare_tensors("Forward output (C++ normal vs Python)", output_normal, py_output, threshold)
    all_passed = all_passed and passed

    # Test 3: Normal backward grad_input should match Python full backward
    print("\n[Test 3] Normal backward grad_input vs Python full backward")
    passed = compare_tensors("grad_input (C++ normal vs Python full)", grad_input_normal, py_grad_input_full, threshold)
    all_passed = all_passed and passed

    # Test 4: SkipLoRA backward grad_input should match Python base-only backward
    print("\n[Test 4] SkipLoRA backward grad_input vs Python base-only backward")
    passed = compare_tensors(
        "grad_input (C++ skip vs Python base-only)", grad_input_skip, py_grad_input_base_only, threshold
    )
    all_passed = all_passed and passed

    # Test 5: SkipLoRA should have zero LoRA gradients
    print("\n[Test 5] SkipLoRA LoRA gradients should be zero")
    passed = check_zeros("grad_gate_lora_a (skip)", grad_loras_skip["grad_gate_lora_a"])
    all_passed = all_passed and passed
    passed = check_zeros("grad_gate_lora_b (skip)", grad_loras_skip["grad_gate_lora_b"])
    all_passed = all_passed and passed
    passed = check_zeros("grad_up_lora_a (skip)", grad_loras_skip["grad_up_lora_a"])
    all_passed = all_passed and passed
    passed = check_zeros("grad_up_lora_b (skip)", grad_loras_skip["grad_up_lora_b"])
    all_passed = all_passed and passed
    passed = check_zeros("grad_down_lora_a (skip)", grad_loras_skip["grad_down_lora_a"])
    all_passed = all_passed and passed
    passed = check_zeros("grad_down_lora_b (skip)", grad_loras_skip["grad_down_lora_b"])
    all_passed = all_passed and passed

    # Test 6: Normal should have non-zero LoRA gradients
    print("\n[Test 6] Normal LoRA gradients should be non-zero")
    normal_lora_grad_sum = (
        grad_loras_normal["grad_gate_lora_a"].abs().sum()
        + grad_loras_normal["grad_gate_lora_b"].abs().sum()
        + grad_loras_normal["grad_up_lora_a"].abs().sum()
        + grad_loras_normal["grad_up_lora_b"].abs().sum()
        + grad_loras_normal["grad_down_lora_a"].abs().sum()
        + grad_loras_normal["grad_down_lora_b"].abs().sum()
    )
    if normal_lora_grad_sum > 1e-6:
        print(f"\033[92m[PASS]\033[0m Normal LoRA gradients are non-zero (sum: {normal_lora_grad_sum:.6e})")
    else:
        print(f"\033[91m[FAIL]\033[0m Normal LoRA gradients are unexpectedly zero (sum: {normal_lora_grad_sum:.6e})")
        all_passed = False

    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("\033[92mALL TESTS PASSED\033[0m")
    else:
        print("\033[91mSOME TESTS FAILED\033[0m")
    print("=" * 80)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test SkipLoRA feature")
    parser.add_argument("--tp-count", type=int, default=DEFAULT_TP_COUNT, help="TP partition count")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Relative error threshold")
    args = parser.parse_args()

    success = test_skip_lora(args.tp_count, args.threshold)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
