#!/usr/bin/env python
# coding=utf-8
"""
MOE SFT Wrapper Test File

This file tests the SFT MoE Wrapper interface (KTMoEWrapper with mode="sft").
It validates that the wrapper correctly wraps the underlying C++ implementation.

Key differences from test_moe_sft_amx.py:
- Uses KTMoEWrapper factory interface instead of direct C++ bindings
- Tests the Python wrapper layer (KExpertsSFTBuffer, AMXSFTMoEWrapper)
- Validates that wrapper behaves identically to direct C++ calls
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import torch.nn.functional as F

# Try to import kt_kernel
try:
    from kt_kernel.experts import KTMoEWrapper
    from kt_kernel.experts_sft import KExpertsSFTBuffer, BaseSFTMoEWrapper

    HAS_KT_KERNEL = True
except ImportError:
    try:
        # Alternative import path (for development)
        sys.path.insert(0, os.path.dirname(__file__) + "/../python")
        from experts import KTMoEWrapper
        from experts_sft import KExpertsSFTBuffer, BaseSFTMoEWrapper

        HAS_KT_KERNEL = True
    except ImportError as e:
        print(f"Warning: Could not import kt_kernel: {e}")
        HAS_KT_KERNEL = False
        KTMoEWrapper = None

# =============================================================================
# Test Configuration
# =============================================================================

# Model configuration (based on DeepSeek-V3 architecture)
expert_num = 256  # Total number of experts
hidden_size = 7168  # Hidden dimension
intermediate_size = 2048  # MLP intermediate dimension
max_len = 25600  # Maximum sequence length
num_experts_per_tok = 8  # Number of experts per token (top-k)
qlen = 4  # Sequence length for testing
layer_num = 1  # Number of layers to test

# LoRA configuration
lora_rank = 16  # LoRA rank (r)
lora_alpha = 32.0  # LoRA scaling factor (alpha)
lora_scaling = lora_alpha / lora_rank  # Effective scaling: alpha / r

# Test configuration
validation_iter = 2  # Number of validation iterations
debug_print_count = 8  # Number of values to print in debug output
num_threads = 60  # Number of CPU threads for inference

# TP configuration
TP_COUNT = 4  # TP mode: multiple NUMA subpools
NO_TP_COUNT = 1  # No-TP mode: single subpool

# Precision thresholds
BF16_FORWARD_THRESHOLD = 0.05
BF16_BACKWARD_THRESHOLD = 0.10


# =============================================================================
# Activation Functions
# =============================================================================


def act_fn(x: torch.Tensor) -> torch.Tensor:
    """Activation function for MoE MLP (SiLU/Swish)"""
    return x / (1.0 + torch.exp(-x))


# =============================================================================
# LoRA Linear Layer Reference Implementation
# =============================================================================


def lora_linear_forward(
    x: torch.Tensor, weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float
) -> torch.Tensor:
    """LoRA linear layer forward pass."""
    base_out = torch.mm(x, weight.t())
    lora_out = torch.mm(torch.mm(x, lora_a.t()), lora_b.t()) * scaling
    return base_out + lora_out


def lora_linear_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> tuple:
    """LoRA linear layer backward pass."""
    grad_input = torch.mm(grad_output, weight)
    grad_input += torch.mm(torch.mm(grad_output, lora_b), lora_a) * scaling
    lora_intermediate = torch.mm(x, lora_a.t())
    grad_lora_b = torch.mm(grad_output.t(), lora_intermediate) * scaling
    grad_lora_a = torch.mm(torch.mm(lora_b.t(), grad_output.t()), x) * scaling
    return grad_input, grad_lora_a, grad_lora_b


# =============================================================================
# MLP Reference Implementation (Single Expert with LoRA)
# =============================================================================


def mlp_lora_forward(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> tuple:
    """MLP forward pass with LoRA adapters on all projections."""
    gate_out = lora_linear_forward(x, gate_proj, gate_lora_a, gate_lora_b, scaling)
    up_out = lora_linear_forward(x, up_proj, up_lora_a, up_lora_b, scaling)
    gate_activated = act_fn(gate_out)
    intermediate = gate_activated * up_out
    output = lora_linear_forward(intermediate, down_proj, down_lora_a, down_lora_b, scaling)

    saved_tensors = {
        "x": x,
        "gate_out": gate_out,
        "up_out": up_out,
        "gate_activated": gate_activated,
        "intermediate": intermediate,
    }

    return output, saved_tensors


def mlp_lora_backward(
    grad_output: torch.Tensor,
    saved_tensors: dict,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> dict:
    """MLP backward pass with LoRA adapters."""
    x = saved_tensors["x"]
    gate_out = saved_tensors["gate_out"]
    up_out = saved_tensors["up_out"]
    gate_activated = saved_tensors["gate_activated"]
    intermediate = saved_tensors["intermediate"]

    grad_intermediate, grad_down_lora_a, grad_down_lora_b = lora_linear_backward(
        grad_output, intermediate, down_proj, down_lora_a, down_lora_b, scaling
    )

    grad_gate_activated = grad_intermediate * up_out
    grad_up_out = grad_intermediate * gate_activated

    sigmoid_gate = torch.sigmoid(gate_out)
    grad_gate_out = grad_gate_activated * sigmoid_gate * (1 + gate_out * (1 - sigmoid_gate))

    grad_x_up, grad_up_lora_a, grad_up_lora_b = lora_linear_backward(
        grad_up_out, x, up_proj, up_lora_a, up_lora_b, scaling
    )

    grad_x_gate, grad_gate_lora_a, grad_gate_lora_b = lora_linear_backward(
        grad_gate_out, x, gate_proj, gate_lora_a, gate_lora_b, scaling
    )

    grad_input = grad_x_up + grad_x_gate

    return {
        "grad_input": grad_input,
        "grad_gate_lora_a": grad_gate_lora_a,
        "grad_gate_lora_b": grad_gate_lora_b,
        "grad_up_lora_a": grad_up_lora_a,
        "grad_up_lora_b": grad_up_lora_b,
        "grad_down_lora_a": grad_down_lora_a,
        "grad_down_lora_b": grad_down_lora_b,
    }


# =============================================================================
# MOE SFT Reference Implementation (PyTorch)
# =============================================================================


def moe_sft_torch_forward(
    input: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> tuple:
    """MoE SFT forward pass with LoRA adapters (PyTorch reference)."""
    qlen = input.shape[0]
    k = expert_ids.shape[1]

    cnts = expert_ids.new_zeros((qlen, expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // k]

    outputs = []
    saved_tensors_list = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            saved_tensors_list.append(None)
            continue

        end_idx = start_idx + int(num_tokens)
        tokens_for_expert = sorted_tokens[start_idx:end_idx]

        expert_out, saved = mlp_lora_forward(
            tokens_for_expert,
            gate_proj[i],
            up_proj[i],
            down_proj[i],
            gate_lora_a[i],
            gate_lora_b[i],
            up_lora_a[i],
            up_lora_b[i],
            down_lora_a[i],
            down_lora_b[i],
            scaling,
        )

        outputs.append(expert_out)
        saved["expert_id"] = i
        saved["start_idx"] = start_idx
        saved["end_idx"] = end_idx
        saved_tensors_list.append(saved)
        start_idx = end_idx

    if outputs:
        outs = torch.cat(outputs, dim=0)
    else:
        outs = sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    output = new_x.view(qlen, k, -1).type(weights.dtype).mul_(weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)

    moe_saved = {
        "input": input,
        "expert_ids": expert_ids,
        "weights": weights,
        "idxs": idxs,
        "tokens_per_expert": tokens_per_expert,
        "expert_saved_tensors": saved_tensors_list,
    }

    return output, moe_saved


def moe_sft_torch_backward(
    grad_output: torch.Tensor,
    moe_saved: dict,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_lora_a: torch.Tensor,
    gate_lora_b: torch.Tensor,
    up_lora_a: torch.Tensor,
    up_lora_b: torch.Tensor,
    down_lora_a: torch.Tensor,
    down_lora_b: torch.Tensor,
    scaling: float,
) -> dict:
    """MoE SFT backward pass (PyTorch reference)."""
    input = moe_saved["input"]
    expert_ids = moe_saved["expert_ids"]
    weights = moe_saved["weights"]
    idxs = moe_saved["idxs"]
    tokens_per_expert = moe_saved["tokens_per_expert"]
    expert_saved_list = moe_saved["expert_saved_tensors"]

    qlen, k = expert_ids.shape

    grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
    grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1]).to(grad_output.dtype)
    sorted_grad_output = grad_output_expanded[idxs]

    grad_input_sorted = torch.zeros_like(sorted_grad_output)

    grad_gate_lora_a = torch.zeros_like(gate_lora_a)
    grad_gate_lora_b = torch.zeros_like(gate_lora_b)
    grad_up_lora_a = torch.zeros_like(up_lora_a)
    grad_up_lora_b = torch.zeros_like(up_lora_b)
    grad_down_lora_a = torch.zeros_like(down_lora_a)
    grad_down_lora_b = torch.zeros_like(down_lora_b)

    for i, saved in enumerate(expert_saved_list):
        if saved is None:
            continue

        start_idx = saved["start_idx"]
        end_idx = saved["end_idx"]
        grad_out_expert = sorted_grad_output[start_idx:end_idx]

        grads = mlp_lora_backward(
            grad_out_expert,
            saved,
            gate_proj[i],
            up_proj[i],
            down_proj[i],
            gate_lora_a[i],
            gate_lora_b[i],
            up_lora_a[i],
            up_lora_b[i],
            down_lora_a[i],
            down_lora_b[i],
            scaling,
        )

        grad_input_sorted[start_idx:end_idx] = grads["grad_input"]
        grad_gate_lora_a[i] = grads["grad_gate_lora_a"]
        grad_gate_lora_b[i] = grads["grad_gate_lora_b"]
        grad_up_lora_a[i] = grads["grad_up_lora_a"]
        grad_up_lora_b[i] = grads["grad_up_lora_b"]
        grad_down_lora_a[i] = grads["grad_down_lora_a"]
        grad_down_lora_b[i] = grads["grad_down_lora_b"]

    grad_input_flat = torch.zeros_like(grad_input_sorted)
    grad_input_flat[idxs] = grad_input_sorted
    grad_input = grad_input_flat.view(qlen, k, -1).sum(dim=1)

    return {
        "grad_input": grad_input,
        "grad_gate_lora_a": grad_gate_lora_a,
        "grad_gate_lora_b": grad_gate_lora_b,
        "grad_up_lora_a": grad_up_lora_a,
        "grad_up_lora_b": grad_up_lora_b,
        "grad_down_lora_a": grad_down_lora_a,
        "grad_down_lora_b": grad_down_lora_b,
    }


# =============================================================================
# Weight Initialization Utilities
# =============================================================================


def init_base_weights(expert_num: int, hidden_size: int, intermediate_size: int, dtype=torch.bfloat16):
    """Initialize base MoE weights (frozen during fine-tuning)."""
    gate_proj = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous()
    )
    up_proj = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous()
    )
    down_proj = (
        torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, device="cuda").to("cpu").contiguous()
    )

    return gate_proj, up_proj, down_proj


def init_lora_weights(expert_num: int, hidden_size: int, intermediate_size: int, rank: int, dtype=torch.bfloat16):
    """Initialize LoRA weights."""
    gate_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    gate_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    up_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    up_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    down_lora_a = (
        torch.randn((expert_num, rank, intermediate_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    )
    down_lora_b = torch.zeros((expert_num, hidden_size, rank), dtype=dtype, device="cpu").contiguous()

    return (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)


# =============================================================================
# Test Functions
# =============================================================================


def test_wrapper_forward(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Test KTMoEWrapper SFT forward pass accuracy.

    Compares the wrapper implementation against PyTorch reference.

    Args:
        quant_mode: Quantization method (e.g., "AMXBF16_SFT")
        tp_count: Number of NUMA subpools (1 = No-TP, >1 = TP mode)
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Testing KTMoEWrapper SFT Forward Pass - {quant_mode} [{tp_mode_str}, tp_count={tp_count}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available, cannot run test")
        sys.exit(1)

    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)
    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    # Create SFT wrapper using KTMoEWrapper factory
    print(f"\n[INFO] Creating KTMoEWrapper with mode='sft', tp_count={tp_count}...")
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=num_threads,
        threadpool_count=tp_count,
        weight_path="",  # Not used for tensor loading
        chunked_prefill_size=max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )

    # Verify wrapper type
    assert isinstance(wrapper, BaseSFTMoEWrapper), f"Expected BaseSFTMoEWrapper, got {type(wrapper)}"
    print(f"[INFO] Wrapper type: {type(wrapper).__name__}, tp_count={tp_count}")

    # Load base weights from tensors
    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj

    physical_map = torch.arange(expert_num, dtype=torch.int64)
    wrapper.load_weights(physical_map)

    # Initialize LoRA weights
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    print("[INFO] Wrapper initialized successfully")

    threshold = BF16_FORWARD_THRESHOLD

    # Run validation iterations
    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        expert_ids = (
            torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # PyTorch reference forward
        torch_output, _ = moe_sft_torch_forward(
            input_data,
            expert_ids,
            weights,
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
        )

        # Wrapper forward
        output = wrapper.forward_sft(input_data, expert_ids, weights, save_for_backward=False)

        # Compare results
        diff = torch.mean(torch.abs(output - torch_output)) / (torch.mean(torch.abs(torch_output)) + 1e-8)
        print(f"Relative difference: {diff:.6f}")

        if diff < threshold:
            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"FAILED: diff={diff:.6f} >= {threshold}")
            sys.exit(1)

    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n[OK] KTMoEWrapper SFT Forward Pass Test - {quant_mode} [{tp_mode_str}] PASSED")


def test_wrapper_backward(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Test KTMoEWrapper SFT backward pass accuracy.

    Compares the wrapper gradients against PyTorch reference.

    Args:
        quant_mode: Quantization method (e.g., "AMXBF16_SFT")
        tp_count: Number of NUMA subpools (1 = No-TP, >1 = TP mode)
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Testing KTMoEWrapper SFT Backward Pass - {quant_mode} [{tp_mode_str}, tp_count={tp_count}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available, cannot run test")
        sys.exit(1)

    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)
    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    # Create SFT wrapper
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=validation_iter,
    )

    # Load weights
    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    physical_map = torch.arange(expert_num, dtype=torch.int64)
    wrapper.load_weights(physical_map)
    wrapper.init_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)

    print(f"[INFO] Wrapper created with tp_count={tp_count}")

    threshold = BF16_BACKWARD_THRESHOLD

    # Run validation iterations
    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        expert_ids = (
            torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
        grad_output = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # PyTorch reference forward + backward
        _, moe_saved = moe_sft_torch_forward(
            input_data,
            expert_ids,
            weights,
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
        )

        torch_grads = moe_sft_torch_backward(
            grad_output,
            moe_saved,
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
        )

        # Wrapper forward (with save_for_backward=True)
        output = wrapper.forward_sft(input_data, expert_ids, weights, save_for_backward=True)

        # Wrapper backward
        grad_input, grad_loras = wrapper.backward(grad_output)

        # Compare gradients
        diff_input = torch.mean(torch.abs(grad_input - torch_grads["grad_input"])) / (
            torch.mean(torch.abs(torch_grads["grad_input"])) + 1e-8
        )
        print(f"grad_input diff: {diff_input:.6f}")
        assert diff_input < threshold, f"grad_input accuracy failed: {diff_input:.6f}"

        # Check LoRA gradients for activated experts
        activated = [i for i, n in enumerate(moe_saved["tokens_per_expert"]) if n > 0]

        for name, amx_grad, torch_grad in [
            ("gate_lora_a", grad_loras["grad_gate_lora_a"], torch_grads["grad_gate_lora_a"]),
            ("gate_lora_b", grad_loras["grad_gate_lora_b"], torch_grads["grad_gate_lora_b"]),
            ("up_lora_a", grad_loras["grad_up_lora_a"], torch_grads["grad_up_lora_a"]),
            ("up_lora_b", grad_loras["grad_up_lora_b"], torch_grads["grad_up_lora_b"]),
            ("down_lora_a", grad_loras["grad_down_lora_a"], torch_grads["grad_down_lora_a"]),
            ("down_lora_b", grad_loras["grad_down_lora_b"], torch_grads["grad_down_lora_b"]),
        ]:
            amx_subset = amx_grad[activated]
            torch_subset = torch_grad[activated]
            diff = torch.mean(torch.abs(amx_subset - torch_subset)) / (torch.mean(torch.abs(torch_subset)) + 1e-8)
            print(f"  {name} diff: {diff:.6f}")
            assert diff < threshold, f"{name} accuracy failed: {diff:.6f}"

        print(f"PASSED (threshold: {threshold})")

    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n[OK] KTMoEWrapper SFT Backward Pass Test - {quant_mode} [{tp_mode_str}] PASSED")


def test_wrapper_training_loop(quant_mode: str = "AMXBF16_SFT", tp_count: int = TP_COUNT):
    """
    Test complete training loop with KTMoEWrapper.

    Simulates a real training scenario with forward, backward, and optimizer step.

    Args:
        quant_mode: Quantization method (e.g., "AMXBF16_SFT")
        tp_count: Number of NUMA subpools (1 = No-TP, >1 = TP mode)
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Testing Complete Training Loop - {quant_mode} [{tp_mode_str}, tp_count={tp_count}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available, cannot run test")
        sys.exit(1)

    torch.manual_seed(42)

    # Initialize base weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    # Initialize LoRA weights as parameters
    gate_lora_a = (
        torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        / 100
    )
    gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
    up_lora_a = (
        torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16, device="cuda").to("cpu").contiguous()
        / 100
    )
    up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
    down_lora_a = (
        torch.randn(expert_num, lora_rank, intermediate_size, dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .contiguous()
        / 100
    )
    down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=torch.bfloat16).contiguous()

    # Make LoRA B non-zero
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    # Wrap as parameters
    gate_lora_a_param = torch.nn.Parameter(gate_lora_a)
    gate_lora_b_param = torch.nn.Parameter(gate_lora_b)
    up_lora_a_param = torch.nn.Parameter(up_lora_a)
    up_lora_b_param = torch.nn.Parameter(up_lora_b)
    down_lora_a_param = torch.nn.Parameter(down_lora_a)
    down_lora_b_param = torch.nn.Parameter(down_lora_b)

    lora_params = [
        gate_lora_a_param,
        gate_lora_b_param,
        up_lora_a_param,
        up_lora_b_param,
        down_lora_a_param,
        down_lora_b_param,
    ]

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    # Create wrapper
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=num_threads,
        threadpool_count=tp_count,
        weight_path="",
        chunked_prefill_size=max_len,
        method=quant_mode,
        mode="sft",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_cache_depth=1,
    )

    # Load weights
    wrapper.gate_proj = gate_proj
    wrapper.up_proj = up_proj
    wrapper.down_proj = down_proj
    physical_map = torch.arange(expert_num, dtype=torch.int64)
    wrapper.load_weights(physical_map)
    wrapper.init_lora_weights(
        gate_lora_a_param.data,
        gate_lora_b_param.data,
        up_lora_a_param.data,
        up_lora_b_param.data,
        down_lora_a_param.data,
        down_lora_b_param.data,
    )

    print(f"[INFO] Wrapper created with tp_count={tp_count}")

    num_training_steps = 3

    for step in range(num_training_steps):
        print(f"\n--- Training Step {step} ---")

        # Generate batch
        expert_ids = (
            torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
        target = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # Forward pass
        output = wrapper.forward_sft(input_data, expert_ids, weights, save_for_backward=True)

        # Compute loss
        loss = torch.mean((output.float() - target.float()) ** 2)
        print(f"  Loss: {loss.item():.6f}")

        # Compute gradient
        grad_output = 2 * (output.float() - target.float()) / output.numel()
        grad_output = grad_output.to(torch.bfloat16).contiguous()

        # Backward pass
        grad_input, grad_loras = wrapper.backward(grad_output)

        # Copy gradients to parameters
        gate_lora_a_param.grad = grad_loras["grad_gate_lora_a"]
        gate_lora_b_param.grad = grad_loras["grad_gate_lora_b"]
        up_lora_a_param.grad = grad_loras["grad_up_lora_a"]
        up_lora_b_param.grad = grad_loras["grad_up_lora_b"]
        down_lora_a_param.grad = grad_loras["grad_down_lora_a"]
        down_lora_b_param.grad = grad_loras["grad_down_lora_b"]

        # Print gradient norms
        print(f"  gate_lora_a grad norm: {gate_lora_a_param.grad.norm().item():.6e}")

        # Save weight snapshots
        gate_lora_a_before = gate_lora_a_param.data.clone()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Sync updated weights to wrapper
        wrapper.update_lora_weights()

        # Verify weights changed
        gate_a_diff = (gate_lora_a_param.data - gate_lora_a_before).abs().mean().item()
        print(f"  gate_lora_a weight change: {gate_a_diff:.10e}")
        assert gate_a_diff > 0, "Weights should change after optimizer step"

    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n[OK] Training Loop Test - {quant_mode} [{tp_mode_str}] PASSED")


def test_mode_validation(tp_count: int = TP_COUNT):
    """
    Test that mode and method validation works correctly.

    Args:
        tp_count: Number of NUMA subpools (used for creating test wrappers)
    """
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n{'='*60}")
    print(f"Testing Mode and Method Validation [{tp_mode_str}]")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel not available, cannot run test")
        sys.exit(1)

    # Test invalid mode
    try:
        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_gpu_experts=0,
            cpuinfer_threads=num_threads,
            threadpool_count=tp_count,
            weight_path="",
            chunked_prefill_size=max_len,
            method="AMXINT4",
            mode="invalid_mode",  # Invalid mode
        )
        print("FAILED: Should have raised ValueError for invalid mode")
        sys.exit(1)
    except ValueError as e:
        print(f"  [OK] Invalid mode raises ValueError: {e}")

    # Test mismatched method for inference mode
    try:
        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_gpu_experts=0,
            cpuinfer_threads=num_threads,
            threadpool_count=tp_count,
            weight_path="",
            chunked_prefill_size=max_len,
            method="AMXBF16_SFT",  # SFT method
            mode="inference",  # Inference mode
        )
        print("FAILED: Should have raised ValueError for mismatched method")
        sys.exit(1)
    except ValueError as e:
        print(f"  [OK] Mismatched method raises ValueError: {e}")

    # Test mismatched method for SFT mode
    try:
        wrapper = KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            num_gpu_experts=0,
            cpuinfer_threads=num_threads,
            threadpool_count=tp_count,
            weight_path="",
            chunked_prefill_size=max_len,
            method="AMXINT4",  # Inference method
            mode="sft",  # SFT mode
        )
        print("FAILED: Should have raised ValueError for mismatched method")
        sys.exit(1)
    except ValueError as e:
        print(f"  [OK] Mismatched method raises ValueError: {e}")

    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print(f"\n[OK] Mode and Method Validation Test [{tp_mode_str}] PASSED")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_tests_for_tp_mode(tp_count: int, quant_mode: str = "AMXBF16_SFT"):
    """Run all tests for a specific TP configuration."""
    tp_mode_str = "TP" if tp_count > 1 else "No-TP"
    print("\n" + "=" * 70)
    print(f" KTMoEWrapper SFT Test Suite - {tp_mode_str} Mode (tp_count={tp_count})")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  qlen: {qlen}")
    print(f"  num_threads: {num_threads}")
    print(f"  tp_count: {tp_count}")
    print("=" * 70)

    # Test mode validation
    test_mode_validation(tp_count=tp_count)

    # Test forward pass
    test_wrapper_forward(quant_mode, tp_count=tp_count)

    # Test backward pass
    test_wrapper_backward(quant_mode, tp_count=tp_count)

    # Test training loop
    test_wrapper_training_loop(quant_mode, tp_count=tp_count)

    print("\n" + "-" * 70)
    print(f" {tp_mode_str} Mode Tests PASSED!")
    print("-" * 70)


def run_all_tests(quant_mode: str = "AMXBF16_SFT", tp_mode: str = "all"):
    """
    Run all KTMoEWrapper SFT tests.

    Args:
        quant_mode: Quantization method to test
        tp_mode: "all" (both), "tp" (TP only), or "no-tp" (No-TP only)
    """
    print("\n" + "=" * 70)
    print(" KTMoEWrapper SFT Test Suite")
    print("=" * 70)

    try:
        if tp_mode in ("all", "no-tp"):
            # Run No-TP tests (single subpool)
            run_tests_for_tp_mode(tp_count=NO_TP_COUNT, quant_mode=quant_mode)

        if tp_mode in ("all", "tp"):
            # Run TP tests (multiple subpools)
            run_tests_for_tp_mode(tp_count=TP_COUNT, quant_mode=quant_mode)

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KTMoEWrapper SFT Test Suite")
    parser.add_argument(
        "--mode",
        choices=["all", "forward", "backward", "training", "validation"],
        default="all",
        help="Test mode: all runs complete suite, others run specific tests",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="AMXBF16_SFT",
        help="SFT method to test (e.g., AMXBF16_SFT, AMXINT8_SFT)",
    )
    parser.add_argument(
        "--tp",
        choices=["all", "tp", "no-tp"],
        default="all",
        help="TP mode: 'all' (test both), 'tp' (TP only), 'no-tp' (No-TP only)",
    )
    parser.add_argument(
        "--tp-count",
        type=int,
        default=None,
        help="Override tp_count for individual tests (ignored when --mode=all)",
    )
    args = parser.parse_args()

    # Determine tp_count for individual tests
    if args.tp_count is not None:
        tp_count = args.tp_count
    elif args.tp == "no-tp":
        tp_count = NO_TP_COUNT
    else:
        tp_count = TP_COUNT

    if args.mode == "all":
        run_all_tests(quant_mode=args.method, tp_mode=args.tp)
    elif args.mode == "forward":
        test_wrapper_forward(args.method, tp_count=tp_count)
    elif args.mode == "backward":
        test_wrapper_backward(args.method, tp_count=tp_count)
    elif args.mode == "training":
        test_wrapper_training_loop(args.method, tp_count=tp_count)
    elif args.mode == "validation":
        test_mode_validation(tp_count=tp_count)
