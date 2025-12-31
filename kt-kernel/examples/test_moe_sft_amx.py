#!/usr/bin/env python
# coding=utf-8
"""
MOE SFT AMX Test File

This file defines the test interfaces for the moe_sft_amx operator, which is used
for LoRA fine-tuning of MoE (Mixture of Experts) layers with Intel AMX acceleration.

The operator supports:
- Forward pass with LoRA adapters (on gate/up/down projections)
- Backward pass computing gradients for input and LoRA weights
- BF16 and INT8 quantization modes
- Asynchronous execution via CPUInfer

Data flow:
    C++ backward -> grad buffer -> Python param.grad -> optimizer.step()
                                -> next forward syncs back to C++

NOTE: The moe_sft_amx operator is not yet implemented. This test file defines
the expected interfaces that the operator should implement.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import torch.nn.functional as F

# Try to import kt_kernel_ext (will fail until operator is implemented)
try:
    from kt_kernel import kt_kernel_ext

    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    kt_kernel_ext = None

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

# Precision thresholds
BF16_FORWARD_THRESHOLD = 0.05  # Maximum relative error for BF16 forward
INT8_FORWARD_THRESHOLD = 0.15  # Maximum relative error for INT8 forward
BF16_BACKWARD_THRESHOLD = 0.10  # Maximum relative error for BF16 backward
INT8_BACKWARD_THRESHOLD = 0.25  # Maximum relative error for INT8 backward

# Note: physical_to_logical_map is no longer needed in the new SFT API
# The new API uses zero-copy pointers and load_weights_task() without mapping


# =============================================================================
# Activation Functions
# =============================================================================


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU (Swish) activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def act_fn(x: torch.Tensor) -> torch.Tensor:
    """Activation function for MoE MLP (SiLU/Swish)"""
    return x / (1.0 + torch.exp(-x))


# =============================================================================
# LoRA Linear Layer Reference Implementation
# =============================================================================


def lora_linear_forward(
    x: torch.Tensor, weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float
) -> torch.Tensor:
    """
    LoRA linear layer forward pass.

    Computes: y = x @ W^T + (x @ A^T @ B^T) * scaling

    Args:
        x: Input tensor [batch, in_features]
        weight: Base weight matrix [out_features, in_features] (frozen)
        lora_a: LoRA A matrix [rank, in_features] (trainable)
        lora_b: LoRA B matrix [out_features, rank] (trainable)
        scaling: LoRA scaling factor (alpha / rank)

    Returns:
        Output tensor [batch, out_features]
    """
    # Base output: x @ W^T
    base_out = torch.mm(x, weight.t())

    # LoRA output: (x @ A^T @ B^T) * scaling
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
    """
    LoRA linear layer backward pass.

    Computes gradients for input and LoRA weights (A and B matrices).
    Base weight W is frozen and does not receive gradients.

    Args:
        grad_output: Gradient from upstream [batch, out_features]
        x: Input tensor from forward pass [batch, in_features]
        weight: Base weight matrix [out_features, in_features] (frozen)
        lora_a: LoRA A matrix [rank, in_features]
        lora_b: LoRA B matrix [out_features, rank]
        scaling: LoRA scaling factor (alpha / rank)

    Returns:
        Tuple of (grad_input, grad_lora_a, grad_lora_b)
    """
    # Gradient for input: grad_output @ W + grad_output @ B @ A * scaling
    grad_input = torch.mm(grad_output, weight)
    grad_input += torch.mm(torch.mm(grad_output, lora_b), lora_a) * scaling

    # Gradient for lora_b: (grad_output^T @ (x @ A^T)) * scaling
    # Shape: [out_features, rank]
    lora_intermediate = torch.mm(x, lora_a.t())  # [batch, rank]
    grad_lora_b = torch.mm(grad_output.t(), lora_intermediate) * scaling

    # Gradient for lora_a: (B^T @ grad_output^T @ x) * scaling
    # Shape: [rank, in_features]
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
    debug_print: bool = False,
) -> tuple:
    """
    MLP forward pass with LoRA adapters on all projections.

    Computes: down(silu(gate(x)) * up(x))
    where each linear layer has LoRA: linear(x) = x @ W^T + (x @ A^T @ B^T) * scaling

    Args:
        x: Input tensor [batch, hidden_size]
        gate_proj: Gate projection weight [intermediate_size, hidden_size]
        up_proj: Up projection weight [intermediate_size, hidden_size]
        down_proj: Down projection weight [hidden_size, intermediate_size]
        gate_lora_a/b: LoRA weights for gate projection
        up_lora_a/b: LoRA weights for up projection
        down_lora_a/b: LoRA weights for down projection
        scaling: LoRA scaling factor
        debug_print: Whether to print debug information

    Returns:
        Tuple of (output, saved_tensors) where saved_tensors contains
        intermediate values needed for backward pass
    """
    # Gate projection with LoRA
    gate_out = lora_linear_forward(x, gate_proj, gate_lora_a, gate_lora_b, scaling)

    # Up projection with LoRA
    up_out = lora_linear_forward(x, up_proj, up_lora_a, up_lora_b, scaling)

    # Activation: silu(gate) * up
    gate_activated = act_fn(gate_out)
    intermediate = gate_activated * up_out

    # Down projection with LoRA
    output = lora_linear_forward(intermediate, down_proj, down_lora_a, down_lora_b, scaling)

    if debug_print:
        print(f"  gate_out[:8] = {gate_out.flatten()[:8]}")
        print(f"  up_out[:8] = {up_out.flatten()[:8]}")
        print(f"  intermediate[:8] = {intermediate.flatten()[:8]}")
        print(f"  output[:8] = {output.flatten()[:8]}")

    # Save tensors for backward pass
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
    """
    MLP backward pass with LoRA adapters.

    Computes gradients for input and all LoRA weights.

    Args:
        grad_output: Gradient from upstream [batch, hidden_size]
        saved_tensors: Dictionary of tensors saved during forward pass
        gate_proj, up_proj, down_proj: Base projection weights (frozen)
        gate_lora_a/b, up_lora_a/b, down_lora_a/b: LoRA weights
        scaling: LoRA scaling factor

    Returns:
        Dictionary containing:
        - grad_input: Gradient for input
        - grad_gate_lora_a/b: Gradients for gate LoRA weights
        - grad_up_lora_a/b: Gradients for up LoRA weights
        - grad_down_lora_a/b: Gradients for down LoRA weights
    """
    x = saved_tensors["x"]
    gate_out = saved_tensors["gate_out"]
    up_out = saved_tensors["up_out"]
    gate_activated = saved_tensors["gate_activated"]
    intermediate = saved_tensors["intermediate"]

    # Backward through down projection
    grad_intermediate, grad_down_lora_a, grad_down_lora_b = lora_linear_backward(
        grad_output, intermediate, down_proj, down_lora_a, down_lora_b, scaling
    )

    # Backward through activation: d(silu(gate) * up) / d(gate, up)
    # grad_gate_activated = grad_intermediate * up_out
    # grad_up_out = grad_intermediate * gate_activated
    grad_gate_activated = grad_intermediate * up_out
    grad_up_out = grad_intermediate * gate_activated

    # Backward through silu: d(silu(x)) / dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    #                                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    sigmoid_gate = torch.sigmoid(gate_out)
    grad_gate_out = grad_gate_activated * sigmoid_gate * (1 + gate_out * (1 - sigmoid_gate))

    # Backward through up projection
    grad_x_up, grad_up_lora_a, grad_up_lora_b = lora_linear_backward(
        grad_up_out, x, up_proj, up_lora_a, up_lora_b, scaling
    )

    # Backward through gate projection
    grad_x_gate, grad_gate_lora_a, grad_gate_lora_b = lora_linear_backward(
        grad_gate_out, x, gate_proj, gate_lora_a, gate_lora_b, scaling
    )

    # Total gradient for input
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
    debug_print: bool = False,
) -> tuple:
    """
    MoE SFT forward pass with LoRA adapters.

    Routes tokens to selected experts and applies MLP with LoRA.

    Args:
        input: Input tensor [qlen, hidden_size]
        expert_ids: Selected expert IDs per token [qlen, num_experts_per_tok]
        weights: Routing weights per expert [qlen, num_experts_per_tok]
        gate_proj: Gate projections [expert_num, intermediate_size, hidden_size]
        up_proj: Up projections [expert_num, intermediate_size, hidden_size]
        down_proj: Down projections [expert_num, hidden_size, intermediate_size]
        gate_lora_a/b: LoRA weights for gate [expert_num, rank, in_dim] / [expert_num, out_dim, rank]
        up_lora_a/b: LoRA weights for up projection
        down_lora_a/b: LoRA weights for down projection
        scaling: LoRA scaling factor
        debug_print: Whether to print debug information

    Returns:
        Tuple of (output, saved_tensors_per_expert)
    """
    qlen = input.shape[0]
    k = expert_ids.shape[1]  # num_experts_per_tok

    # Count tokens per expert
    cnts = expert_ids.new_zeros((qlen, expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)

    # Sort tokens by expert
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // k]

    if debug_print:
        activated_experts = [i for i, n in enumerate(tokens_per_expert) if n > 0]
        print(f"[MOE SFT DEBUG] Activated experts: {activated_experts}")

    outputs = []
    saved_tensors_list = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            saved_tensors_list.append(None)
            continue

        end_idx = start_idx + int(num_tokens)
        tokens_for_expert = sorted_tokens[start_idx:end_idx]

        # Forward through MLP with LoRA
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
            debug_print=(debug_print and i == expert_ids[0, 0].item()),
        )

        outputs.append(expert_out)
        saved["expert_id"] = i
        saved["start_idx"] = start_idx
        saved["end_idx"] = end_idx
        saved_tensors_list.append(saved)
        start_idx = end_idx

    # Concatenate outputs
    if outputs:
        outs = torch.cat(outputs, dim=0)
    else:
        outs = sorted_tokens.new_empty(0)

    # Reorder outputs back to original order
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs

    # Apply routing weights and sum
    output = new_x.view(qlen, k, -1).type(weights.dtype).mul_(weights.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)

    if debug_print:
        print(f"[MOE SFT DEBUG] Final output[:8] = {output.flatten()[:8]}")

    # Save additional tensors for backward
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
    """
    MoE SFT backward pass.

    Computes gradients for input and all LoRA weights across all experts.

    Args:
        grad_output: Gradient from upstream [qlen, hidden_size]
        moe_saved: Dictionary of tensors saved during forward
        gate_proj, up_proj, down_proj: Base projection weights (frozen)
        gate_lora_a/b, up_lora_a/b, down_lora_a/b: LoRA weights
        scaling: LoRA scaling factor

    Returns:
        Dictionary containing:
        - grad_input: Gradient for input [qlen, hidden_size]
        - grad_gate_lora_a/b: Gradients for gate LoRA [expert_num, ...]
        - grad_up_lora_a/b: Gradients for up LoRA [expert_num, ...]
        - grad_down_lora_a/b: Gradients for down LoRA [expert_num, ...]
    """
    input = moe_saved["input"]
    expert_ids = moe_saved["expert_ids"]
    weights = moe_saved["weights"]
    idxs = moe_saved["idxs"]
    tokens_per_expert = moe_saved["tokens_per_expert"]
    expert_saved_list = moe_saved["expert_saved_tensors"]

    qlen, k = expert_ids.shape

    # Expand grad_output for each expert
    # grad_output: [qlen, hidden_size] -> [qlen, k, hidden_size]
    grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
    grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1])  # [qlen*k, hidden_size]

    # Reorder to match sorted token order
    sorted_grad_output = grad_output_expanded[idxs]

    # Initialize gradient accumulators
    grad_input_sorted = torch.zeros_like(sorted_grad_output)

    # Initialize LoRA gradient accumulators
    grad_gate_lora_a = torch.zeros_like(gate_lora_a)
    grad_gate_lora_b = torch.zeros_like(gate_lora_b)
    grad_up_lora_a = torch.zeros_like(up_lora_a)
    grad_up_lora_b = torch.zeros_like(up_lora_b)
    grad_down_lora_a = torch.zeros_like(down_lora_a)
    grad_down_lora_b = torch.zeros_like(down_lora_b)

    # Backward through each expert
    for i, saved in enumerate(expert_saved_list):
        if saved is None:
            continue

        start_idx = saved["start_idx"]
        end_idx = saved["end_idx"]
        grad_out_expert = sorted_grad_output[start_idx:end_idx]

        # Backward through MLP
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

    # Reorder gradients back to original order
    grad_input_flat = torch.zeros_like(grad_input_sorted)
    grad_input_flat[idxs] = grad_input_sorted

    # Sum gradients for each token (from multiple experts)
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
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device="cpu").contiguous() / 100
    )  # Scale down for numerical stability

    up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=dtype, device="cpu").contiguous() / 100

    down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=dtype, device="cpu").contiguous() / 100

    return gate_proj, up_proj, down_proj


def init_lora_weights(expert_num: int, hidden_size: int, intermediate_size: int, rank: int, dtype=torch.bfloat16):
    """
    Initialize LoRA weights.

    LoRA A matrices are initialized with Kaiming uniform.
    LoRA B matrices are initialized to zero (so initial output equals base model).
    """
    # Gate projection LoRA
    gate_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cpu").contiguous() / 100
    gate_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    # Up projection LoRA
    up_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cpu").contiguous() / 100
    up_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    # Down projection LoRA
    down_lora_a = torch.randn((expert_num, rank, intermediate_size), dtype=dtype, device="cpu").contiguous() / 100
    down_lora_b = torch.zeros((expert_num, hidden_size, rank), dtype=dtype, device="cpu").contiguous()

    return (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)


# =============================================================================
# Test Functions
# =============================================================================


def test_moe_sft_forward(quant_mode: str):
    """
    Test MOE SFT forward pass accuracy.

    Compares the AMX implementation against PyTorch reference.

    Args:
        quant_mode: Quantization mode ("bf16" or "int8")
    """
    assert quant_mode in ["bf16", "int8"], f"Invalid quant_mode: {quant_mode}"

    print(f"\n{'='*60}")
    print(f"Testing MOE SFT Forward Pass - {quant_mode.upper()} mode")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)
    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel_ext not available, running PyTorch reference only")

    # Initialize CPUInfer (when kt_kernel is available)
    if HAS_KT_KERNEL:
        CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

        # Create MOE SFT config using the new API
        config = kt_kernel_ext.moe.MOESFTConfig()
        config.expert_num = expert_num
        config.num_experts_per_tok = num_experts_per_tok
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.max_cache_depth = 1
        config.max_len = max_len
        config.layer_idx = 0
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        # Set LoRA weight pointers directly in config (zero-copy)
        config.gate_lora_a = gate_lora_a.data_ptr()
        config.gate_lora_b = gate_lora_b.data_ptr()
        config.up_lora_a = up_lora_a.data_ptr()
        config.up_lora_b = up_lora_b.data_ptr()
        config.down_lora_a = down_lora_a.data_ptr()
        config.down_lora_b = down_lora_b.data_ptr()
        config.pool = CPUInfer.backend_

        # Create MOE SFT instance
        if quant_mode == "bf16":
            moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
        else:
            moe = kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)

        # Load base weights
        CPUInfer.submit(moe.load_weights_task())
        CPUInfer.sync()

        # Warm up
        CPUInfer.submit(moe.warm_up_task())
        CPUInfer.sync()

    # Run validation iterations
    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        bsz_tensor = torch.tensor([qlen], device="cpu")
        expert_ids = (
            torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize
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
            debug_print=(iter_idx == 0),
        )

        if HAS_KT_KERNEL:
            # AMX forward using forward_sft_task (no separate sync needed - uses config pointers)
            output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
            CPUInfer.submit(
                moe.forward_sft_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    True,  # save_for_backward
                )
            )
            CPUInfer.sync()

            # Convert output to bf16 for comparison
            output_bf16 = output.to(torch.bfloat16)

            # Compare results
            diff = torch.mean(torch.abs(output_bf16 - torch_output)) / (torch.mean(torch.abs(torch_output)) + 1e-8)
            print(f"Relative difference: {diff:.6f}")

            threshold = BF16_FORWARD_THRESHOLD if quant_mode == "bf16" else INT8_FORWARD_THRESHOLD
            assert diff < threshold, f"Forward pass accuracy test failed: diff={diff:.6f} >= {threshold}"
            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"PyTorch output shape: {torch_output.shape}")
            print(f"PyTorch output[:8]: {torch_output.flatten()[:8]}")

    print(f"\n[OK] MOE SFT Forward Pass Test - {quant_mode.upper()} mode PASSED")


def test_moe_sft_backward(quant_mode: str):
    """
    Test MOE SFT backward pass accuracy.

    Compares the AMX implementation gradients against PyTorch reference.

    Args:
        quant_mode: Quantization mode ("bf16" or "int8")
    """
    assert quant_mode in ["bf16", "int8"], f"Invalid quant_mode: {quant_mode}"

    print(f"\n{'='*60}")
    print(f"Testing MOE SFT Backward Pass - {quant_mode.upper()} mode")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)
    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel_ext not available, running PyTorch reference only")

    # Initialize CPUInfer (when kt_kernel is available)
    if HAS_KT_KERNEL:
        CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

        # Create MOE SFT config using the new API
        config = kt_kernel_ext.moe.MOESFTConfig()
        config.expert_num = expert_num
        config.num_experts_per_tok = num_experts_per_tok
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.max_cache_depth = 1
        config.max_len = max_len
        config.layer_idx = 0
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        # Set LoRA weight pointers directly in config (zero-copy)
        config.gate_lora_a = gate_lora_a.data_ptr()
        config.gate_lora_b = gate_lora_b.data_ptr()
        config.up_lora_a = up_lora_a.data_ptr()
        config.up_lora_b = up_lora_b.data_ptr()
        config.down_lora_a = down_lora_a.data_ptr()
        config.down_lora_b = down_lora_b.data_ptr()
        config.pool = CPUInfer.backend_

        # Create MOE SFT instance
        if quant_mode == "bf16":
            moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
        else:
            moe = kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)

        # Load base weights
        CPUInfer.submit(moe.load_weights_task())
        CPUInfer.sync()

        # Warm up
        CPUInfer.submit(moe.warm_up_task())
        CPUInfer.sync()

    # Run validation iterations
    for iter_idx in range(validation_iter):
        print(f"\n--- Iteration {iter_idx} ---")

        # Generate random inputs
        bsz_tensor = torch.tensor([qlen], device="cpu")
        expert_ids = (
            torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
            .to(torch.int64)
            .contiguous()
        )
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        weights = weights / weights.sum(dim=-1, keepdim=True)
        input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

        # Random gradient from upstream
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

        if HAS_KT_KERNEL:
            # AMX forward (with save_for_backward=True)
            # LoRA weights already set in config (zero-copy), no sync needed
            output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
            CPUInfer.submit(
                moe.forward_sft_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    True,  # save_for_backward
                )
            )
            CPUInfer.sync()

            # Allocate gradient buffers
            grad_input = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            grad_gate_lora_a = torch.zeros_like(gate_lora_a)
            grad_gate_lora_b = torch.zeros_like(gate_lora_b)
            grad_up_lora_a = torch.zeros_like(up_lora_a)
            grad_up_lora_b = torch.zeros_like(up_lora_b)
            grad_down_lora_a = torch.zeros_like(down_lora_a)
            grad_down_lora_b = torch.zeros_like(down_lora_b)

            # AMX backward
            CPUInfer.submit(
                moe.backward_task(
                    grad_output.data_ptr(),
                    grad_input.data_ptr(),
                    grad_gate_lora_a.data_ptr(),
                    grad_gate_lora_b.data_ptr(),
                    grad_up_lora_a.data_ptr(),
                    grad_up_lora_b.data_ptr(),
                    grad_down_lora_a.data_ptr(),
                    grad_down_lora_b.data_ptr(),
                )
            )
            CPUInfer.sync()

            # Compare gradients
            threshold = BF16_BACKWARD_THRESHOLD if quant_mode == "bf16" else INT8_BACKWARD_THRESHOLD

            # Input gradient
            diff_input = torch.mean(torch.abs(grad_input - torch_grads["grad_input"])) / (
                torch.mean(torch.abs(torch_grads["grad_input"])) + 1e-8
            )
            print(f"grad_input diff: {diff_input:.6f}")
            assert diff_input < threshold, f"grad_input accuracy failed: {diff_input:.6f}"

            # LoRA gradients (check activated experts only)
            activated = [i for i, n in enumerate(moe_saved["tokens_per_expert"]) if n > 0]

            for name, amx_grad, torch_grad in [
                ("gate_lora_a", grad_gate_lora_a, torch_grads["grad_gate_lora_a"]),
                ("gate_lora_b", grad_gate_lora_b, torch_grads["grad_gate_lora_b"]),
                ("up_lora_a", grad_up_lora_a, torch_grads["grad_up_lora_a"]),
                ("up_lora_b", grad_up_lora_b, torch_grads["grad_up_lora_b"]),
                ("down_lora_a", grad_down_lora_a, torch_grads["grad_down_lora_a"]),
                ("down_lora_b", grad_down_lora_b, torch_grads["grad_down_lora_b"]),
            ]:
                amx_subset = amx_grad[activated]
                torch_subset = torch_grad[activated]
                diff = torch.mean(torch.abs(amx_subset - torch_subset)) / (torch.mean(torch.abs(torch_subset)) + 1e-8)
                print(f"  {name} diff: {diff:.6f}")
                assert diff < threshold, f"{name} accuracy failed: {diff:.6f}"

            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"PyTorch grad_input shape: {torch_grads['grad_input'].shape}")
            print(f"PyTorch grad_gate_lora_a[:2,:4,:4]: {torch_grads['grad_gate_lora_a'][:2,:4,:4]}")

    print(f"\n[OK] MOE SFT Backward Pass Test - {quant_mode.upper()} mode PASSED")


def test_moe_sft_lora_weight_sync():
    """
    Test LoRA weight synchronization between Python and C++.

    Verifies that:
    1. Initial config correctly sets LoRA weight pointers (zero-copy)
    2. Modified weights are correctly reflected via update_lora_weights_task
    3. Forward pass uses the updated weights
    """
    print(f"\n{'='*60}")
    print("Testing LoRA Weight Synchronization")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("WARNING: kt_kernel_ext not available, skipping sync test")
        return

    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)
    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

    # Create MOE SFT config using the new API
    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = expert_num
    config.num_experts_per_tok = num_experts_per_tok
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.max_cache_depth = 1
    config.max_len = max_len
    config.layer_idx = 0
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    # Set initial LoRA weight pointers in config (zero-copy)
    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = CPUInfer.backend_

    moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)

    # Load base weights
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    # Warm up
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    # Test data
    bsz_tensor = torch.tensor([qlen], device="cpu")
    expert_ids = (
        torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)])
        .to(torch.int64)
        .contiguous()
    )
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    input_data = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

    # First forward with initial LoRA weights (already set in config, zero-copy)
    output1 = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output1.data_ptr(),
            False,  # save_for_backward
        )
    )
    CPUInfer.sync()

    # Modify LoRA weights (simulating optimizer.step())
    # Since we use zero-copy pointers, the C++ side sees updates automatically
    gate_lora_a.add_(0.1)
    gate_lora_b.add_(0.1)
    up_lora_a.add_(0.1)
    up_lora_b.add_(0.1)
    down_lora_a.add_(0.1)
    down_lora_b.add_(0.1)

    # If the tensor is reallocated (e.g., after torch operations that create new tensors),
    # we need to update the pointers using update_lora_weights_task
    # For in-place operations like add_(), no sync is needed due to zero-copy

    # Second forward with updated LoRA weights
    output2 = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output2.data_ptr(),
            False,  # save_for_backward
        )
    )
    CPUInfer.sync()

    # Outputs should be different after weight update
    diff = torch.mean(torch.abs(output1 - output2))
    print(f"Output difference after weight update: {diff:.6f}")
    assert diff > 1e-6, "Outputs should differ after LoRA weight update"

    # Test explicit update_lora_weights_task (for when tensors are reallocated)
    # Create new tensors (simulating tensor reallocation)
    new_gate_lora_a = gate_lora_a.clone()
    new_gate_lora_b = gate_lora_b.clone()
    new_up_lora_a = up_lora_a.clone()
    new_up_lora_b = up_lora_b.clone()
    new_down_lora_a = down_lora_a.clone()
    new_down_lora_b = down_lora_b.clone()

    # Update pointers using update_lora_weights_task
    CPUInfer.submit(
        moe.update_lora_weights_task(
            new_gate_lora_a.data_ptr(),
            new_gate_lora_b.data_ptr(),
            new_up_lora_a.data_ptr(),
            new_up_lora_b.data_ptr(),
            new_down_lora_a.data_ptr(),
            new_down_lora_b.data_ptr(),
        )
    )
    CPUInfer.sync()

    # Third forward with new tensor pointers
    output3 = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output3.data_ptr(),
            False,  # save_for_backward
        )
    )
    CPUInfer.sync()

    # Output3 should match output2 (same weights, different tensor locations)
    diff_same = torch.mean(torch.abs(output2 - output3))
    print(f"Output difference after pointer update (should be ~0): {diff_same:.6f}")
    assert diff_same < 1e-5, f"Outputs should match after pointer update: {diff_same:.6f}"

    print("[OK] LoRA Weight Synchronization Test PASSED")


def test_moe_sft_training_loop():
    """
    Test complete training loop: forward → backward → optimizer.step.

    This simulates a real training scenario where:
    1. Forward pass computes output and saves activations
    2. Backward pass computes gradients for LoRA weights
    3. Optimizer updates LoRA weights
    4. Next forward uses updated weights (zero-copy via pointers)
    """
    print(f"\n{'='*60}")
    print("Testing Complete Training Loop")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Initialize weights
    gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    # Initialize LoRA weights as contiguous tensors
    # We use regular tensors (not nn.Parameter) since we manually handle gradients
    gate_lora_a = torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16).contiguous() / 100
    gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
    up_lora_a = torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16).contiguous() / 100
    up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
    down_lora_a = torch.randn(expert_num, lora_rank, intermediate_size, dtype=torch.bfloat16).contiguous() / 100
    down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=torch.bfloat16).contiguous()

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    # Wrap tensors as nn.Parameters for optimizer
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

    # Create optimizer
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    # Initialize kt_kernel if available
    moe = None
    CPUInfer = None
    if HAS_KT_KERNEL:
        CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

        # Create MOE SFT config using the new API
        config = kt_kernel_ext.moe.MOESFTConfig()
        config.expert_num = expert_num
        config.num_experts_per_tok = num_experts_per_tok
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.max_cache_depth = 1
        config.max_len = max_len
        config.layer_idx = 0
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()
        # Set LoRA weight pointers directly in config (zero-copy)
        config.gate_lora_a = gate_lora_a_param.data.data_ptr()
        config.gate_lora_b = gate_lora_b_param.data.data_ptr()
        config.up_lora_a = up_lora_a_param.data.data_ptr()
        config.up_lora_b = up_lora_b_param.data.data_ptr()
        config.down_lora_a = down_lora_a_param.data.data_ptr()
        config.down_lora_b = down_lora_b_param.data.data_ptr()
        config.pool = CPUInfer.backend_

        moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)

        # Load base weights
        CPUInfer.submit(moe.load_weights_task())
        CPUInfer.sync()

        # Warm up
        CPUInfer.submit(moe.warm_up_task())
        CPUInfer.sync()
    else:
        print("WARNING: kt_kernel_ext not available, running PyTorch-only training loop")

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

        if HAS_KT_KERNEL and moe is not None:
            # Use kt_kernel for forward and backward
            bsz_tensor = torch.tensor([qlen], device="cpu")

            # Forward pass (with save_for_backward=True)
            output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
            CPUInfer.submit(
                moe.forward_sft_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    True,  # save_for_backward
                )
            )
            CPUInfer.sync()

            # Simple MSE loss
            loss = torch.mean((output - target.float()) ** 2)
            print(f"  Loss (AMX): {loss.item():.6f}")

            # Compute gradient of loss w.r.t. output
            grad_output = 2 * (output - target.float()) / output.numel()
            grad_output = grad_output.to(torch.bfloat16).contiguous()

            # Allocate gradient buffers
            grad_input = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            grad_gate_lora_a = torch.zeros_like(gate_lora_a_param.data)
            grad_gate_lora_b = torch.zeros_like(gate_lora_b_param.data)
            grad_up_lora_a = torch.zeros_like(up_lora_a_param.data)
            grad_up_lora_b = torch.zeros_like(up_lora_b_param.data)
            grad_down_lora_a = torch.zeros_like(down_lora_a_param.data)
            grad_down_lora_b = torch.zeros_like(down_lora_b_param.data)

            # Backward pass
            CPUInfer.submit(
                moe.backward_task(
                    grad_output.data_ptr(),
                    grad_input.data_ptr(),
                    grad_gate_lora_a.data_ptr(),
                    grad_gate_lora_b.data_ptr(),
                    grad_up_lora_a.data_ptr(),
                    grad_up_lora_b.data_ptr(),
                    grad_down_lora_a.data_ptr(),
                    grad_down_lora_b.data_ptr(),
                )
            )
            CPUInfer.sync()

            # Copy gradients to parameters
            gate_lora_a_param.grad = grad_gate_lora_a
            gate_lora_b_param.grad = grad_gate_lora_b
            up_lora_a_param.grad = grad_up_lora_a
            up_lora_b_param.grad = grad_up_lora_b
            down_lora_a_param.grad = grad_down_lora_a
            down_lora_b_param.grad = grad_down_lora_b

        else:
            # PyTorch reference forward + backward
            output, moe_saved = moe_sft_torch_forward(
                input_data.detach(),
                expert_ids,
                weights,
                gate_proj,
                up_proj,
                down_proj,
                gate_lora_a_param.data.contiguous(),
                gate_lora_b_param.data.contiguous(),
                up_lora_a_param.data.contiguous(),
                up_lora_b_param.data.contiguous(),
                down_lora_a_param.data.contiguous(),
                down_lora_b_param.data.contiguous(),
                lora_scaling,
            )

            # Simple MSE loss
            loss = torch.mean((output.float() - target.float()) ** 2)
            print(f"  Loss (PyTorch): {loss.item():.6f}")

            # Compute gradient of loss w.r.t. output
            grad_output = 2 * (output.float() - target.float()) / output.numel()
            grad_output = grad_output.to(torch.bfloat16).contiguous()

            # Backward pass
            grads = moe_sft_torch_backward(
                grad_output,
                moe_saved,
                gate_proj,
                up_proj,
                down_proj,
                gate_lora_a_param.data.contiguous(),
                gate_lora_b_param.data.contiguous(),
                up_lora_a_param.data.contiguous(),
                up_lora_b_param.data.contiguous(),
                down_lora_a_param.data.contiguous(),
                down_lora_b_param.data.contiguous(),
                lora_scaling,
            )

            # Copy gradients to parameters
            gate_lora_a_param.grad = grads["grad_gate_lora_a"]
            gate_lora_b_param.grad = grads["grad_gate_lora_b"]
            up_lora_a_param.grad = grads["grad_up_lora_a"]
            up_lora_b_param.grad = grads["grad_up_lora_b"]
            down_lora_a_param.grad = grads["grad_down_lora_a"]
            down_lora_b_param.grad = grads["grad_down_lora_b"]

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Print weight update magnitude
        print(f"  gate_lora_a norm: {gate_lora_a_param.data.norm().item():.6f}")
        print(f"  gate_lora_b norm: {gate_lora_b_param.data.norm().item():.6f}")

        # Note: Since we use zero-copy pointers, the C++ side sees updates automatically
        # No explicit sync needed after optimizer.step() for in-place updates

    print("\n[OK] Training Loop Test PASSED")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all_tests():
    """Run all MOE SFT tests."""
    print("\n" + "=" * 70)
    print(" MOE SFT AMX Test Suite")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  qlen: {qlen}")
    print("=" * 70)

    try:
        # Forward pass tests
        test_moe_sft_forward("bf16")
        test_moe_sft_forward("int8")

        # Backward pass tests
        test_moe_sft_backward("bf16")
        test_moe_sft_backward("int8")

        # Weight sync test
        test_moe_sft_lora_weight_sync()

        # Full training loop test
        test_moe_sft_training_loop()

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
