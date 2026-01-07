#!/usr/bin/env python
# coding=utf-8
"""
MOE SFT AMX Test File - Non-TP (Single NUMA Node) Version

This file tests the SFT MoE AMX operator with a single NUMA node configuration
to isolate whether numerical bugs are in the basic SFT logic or TP partitioning.

Key difference from test_moe_sft_amx.py:
- Uses WorkerPoolConfig to force single subpool (tp_count=1)
- Only tests BF16 forward pass for simplicity
"""

import os
import sys
import math
from typing import Literal, Dict

sys.path.insert(0, os.path.dirname(__file__) + "/../build")
print("sys.path:", sys.path)

import torch
import torch.nn.functional as F

# Try to import kt_kernel_ext
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

# Performance test configuration
perf_warmup_iter = 5  # Number of warmup iterations for performance test
perf_test_iter = 20  # Number of iterations for performance measurement
perf_qlen = 128  # Sequence length for performance testing

# Precision thresholds
BF16_FORWARD_THRESHOLD = 0.05  # Maximum relative error for BF16 forward
BF16_BACKWARD_THRESHOLD = 0.10  # Maximum relative error for BF16 backward
INT4_FORWARD_THRESHOLD = 0.35  # Maximum relative error for INT4 forward (same as inference)
INT4_BACKWARD_THRESHOLD = 0.40  # Maximum relative error for INT4 backward


# =============================================================================
# Quantization Mode Utilities
# =============================================================================


def get_moe_sft_class(quant_mode: str):
    """根据量化模式返回对应的 MOE SFT 类。

    Args:
        quant_mode: 量化模式，支持 "bf16", "int8", "int4", "int4_1", "int4_1kgroup", "int4_kgroup"

    Returns:
        对应的 MOE SFT 类
    """
    if not HAS_KT_KERNEL:
        raise RuntimeError("kt_kernel_ext not available")

    if quant_mode == "bf16":
        return kt_kernel_ext.moe.AMXBF16_SFT_MOE
    elif quant_mode == "int8":
        return kt_kernel_ext.moe.AMXInt8_SFT_MOE
    elif quant_mode == "int4":
        return kt_kernel_ext.moe.AMXInt4_SFT_MOE
    elif quant_mode == "int4_1":
        return kt_kernel_ext.moe.AMXInt4_1_SFT_MOE
    elif quant_mode == "int4_1kgroup":
        return kt_kernel_ext.moe.AMXInt4_1KGroup_SFT_MOE
    elif quant_mode == "int4_kgroup":
        return kt_kernel_ext.moe.AMXInt4_KGroup_SFT_MOE
    else:
        raise ValueError(
            f"Unsupported quant_mode: {quant_mode}. Supported: bf16, int8, int4, int4_1, int4_1kgroup, int4_kgroup"
        )


def get_threshold(quant_mode: str, is_backward: bool = False) -> float:
    """根据量化模式返回精度阈值（与推理测试保持一致）。

    Args:
        quant_mode: 量化模式
        is_backward: 是否为 backward 阈值

    Returns:
        精度阈值
    """
    # INT4 variants (int4, int4_1, int4_1kgroup, int4_kgroup) 使用更高的阈值
    if quant_mode in ("int4", "int4_1", "int4_1kgroup", "int4_kgroup"):
        if is_backward:
            return INT4_BACKWARD_THRESHOLD  # 0.40
        return INT4_FORWARD_THRESHOLD  # 0.35
    # BF16 和 INT8 使用相同阈值
    if is_backward:
        return BF16_BACKWARD_THRESHOLD  # 0.10
    return BF16_FORWARD_THRESHOLD  # 0.05


# =============================================================================
# K2 Quantization Utilities (for INT4_KGROUP mode)
# =============================================================================


def pack_to_int32(value: torch.Tensor, num_bits: int, packed_dim: Literal[0, 1] = 1) -> torch.Tensor:
    """Pack int4 values into int32 tensor.

    Args:
        value: int8 tensor to pack
        num_bits: number of bits per value (4 for int4)
        packed_dim: dimension to pack along

    Returns:
        int32 tensor with packed values
    """
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be torch.int8 before packing")
    if not (1 <= num_bits <= 8):
        raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")

    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def pack_tensor_per_row(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack tensor per row for K2 quantization.

    Args:
        q: [expert_num, rows, cols] int8 tensor
        num_bits: number of bits per value

    Returns:
        Packed int32 tensor
    """
    e, rows, cols = q.shape
    flat = q.view(e * rows, cols)
    packed = pack_to_int32(flat, num_bits)
    return packed.view(e, rows, -1).contiguous()


def quantize_k2_tensor(weights: torch.Tensor, group_size: int):
    """
    K2 symmetric max-abs/7 quantization per k-group.

    Args:
        weights: [expert_num, rows (N), cols (K)] bfloat16 tensor

    Returns:
        packed_q: int32 tensor storing 8 int4s per element with shape [expert_num, rows * (cols // 8)]
        scales: bfloat16 tensor with shape [expert_num, rows * (cols // group_size)]
    """
    weights_f32 = weights.to(torch.float32)
    e, rows, cols = weights_f32.shape
    if cols % group_size != 0 or cols % 2 != 0:
        raise ValueError(f"cols ({cols}) must be divisible by group_size ({group_size}) and 2")

    reshaped = weights_f32.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-8)
    scales = (max_abs / 7.0).squeeze(-1)
    q = torch.round(reshaped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(e, rows, cols)
    packed = pack_tensor_per_row(q, num_bits=4).view(e, rows, cols // 8).contiguous()
    scales = scales.to(torch.bfloat16).contiguous().view(e, rows, cols // group_size).contiguous()

    return packed, scales


def init_base_weights_for_k2(
    expert_num: int, hidden_size: int, intermediate_size: int, group_size: int = 128
) -> Dict[str, torch.Tensor]:
    """Initialize pre-quantized K2 weights for INT4_KGROUP mode.

    Args:
        expert_num: number of experts
        hidden_size: hidden dimension
        intermediate_size: intermediate dimension
        group_size: quantization group size

    Returns:
        Dictionary containing:
        - gate_qweight, up_qweight, down_qweight: packed int4 weights
        - gate_scales, up_scales, down_scales: bf16 scales
        - gate_proj_bf16, up_proj_bf16, down_proj_bf16: original bf16 weights for reference
    """
    # Create random BF16 weights
    gate_proj_bf16 = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up_proj_bf16 = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down_proj_bf16 = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16)

    # Quantize to int4
    gate_qweight, gate_scales = quantize_k2_tensor(gate_proj_bf16, group_size)
    up_qweight, up_scales = quantize_k2_tensor(up_proj_bf16, group_size)
    down_qweight, down_scales = quantize_k2_tensor(down_proj_bf16, group_size)

    return {
        "gate_qweight": gate_qweight.contiguous(),
        "up_qweight": up_qweight.contiguous(),
        "down_qweight": down_qweight.contiguous(),
        "gate_scales": gate_scales.contiguous(),
        "up_scales": up_scales.contiguous(),
        "down_scales": down_scales.contiguous(),
        # Keep original BF16 for gradient verification
        "gate_proj_bf16": gate_proj_bf16.contiguous(),
        "up_proj_bf16": up_proj_bf16.contiguous(),
        "down_proj_bf16": down_proj_bf16.contiguous(),
    }


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
    # Note: weights is float32, grad_output is bf16. Multiplication promotes to float32.
    # We must convert back to bf16 to match weight dtypes in subsequent matrix operations.
    grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
    grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1]).to(grad_output.dtype)

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
    """Initialize base MoE weights (frozen during fine-tuning).

    NOTE: Weights are NOT divided by 100 (matching inference test).
    This ensures output values are in a normal range for bf16 precision.
    Uses CUDA for fast random generation, then moves to CPU.
    """
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
    """
    Initialize LoRA weights.

    LoRA A matrices are initialized with small random values.
    LoRA B matrices are initialized to zero (so initial output equals base model).
    Uses CUDA for fast random generation, then moves to CPU.
    """
    # Gate projection LoRA
    gate_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    gate_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    # Up projection LoRA
    up_lora_a = torch.randn((expert_num, rank, hidden_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    up_lora_b = torch.zeros((expert_num, intermediate_size, rank), dtype=dtype, device="cpu").contiguous()

    # Down projection LoRA
    down_lora_a = (
        torch.randn((expert_num, rank, intermediate_size), dtype=dtype, device="cuda").to("cpu").contiguous() / 100
    )
    down_lora_b = torch.zeros((expert_num, hidden_size, rank), dtype=dtype, device="cpu").contiguous()

    return (gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b)


# =============================================================================
# Test Functions
# =============================================================================


def test_moe_sft_forward_no_tp(quant_mode: str = "bf16"):
    """
    Test MOE SFT forward pass accuracy with single NUMA node (no TP).

    Compares the AMX implementation against PyTorch reference.
    Uses WorkerPoolConfig to force single subpool.

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    print(f"\n{'='*60}")
    print(f"Testing MOE SFT Forward Pass - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize weights based on quant_mode
    k2_weights = None  # Will be set for K2 mode
    if quant_mode == "int4_kgroup":
        # K2 needs pre-quantized int4 weights
        k2_weights = init_base_weights_for_k2(expert_num, hidden_size, intermediate_size, group_size=128)
        # Use original BF16 for reference computation
        gate_proj = k2_weights["gate_proj_bf16"]
        up_proj = k2_weights["up_proj_bf16"]
        down_proj = k2_weights["down_proj_bf16"]
    else:
        # Other modes use BF16 weights
        gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel_ext not available, cannot run test")
        sys.exit(1)

    # Initialize CPUInfer with single NUMA node configuration
    # This forces tp_count=1, bypassing TP partitioning
    print("\n[INFO] Creating CPUInfer with single NUMA node (NO TP)...")
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)
    print("[INFO] CPUInfer created with single subpool (tp_count=1)")

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

    # Bug #26 fix: K2 uses pre-quantized weights with scales
    if quant_mode == "int4_kgroup" and k2_weights is not None:
        config.gate_proj = k2_weights["gate_qweight"].data_ptr()
        config.up_proj = k2_weights["up_qweight"].data_ptr()
        config.down_proj = k2_weights["down_qweight"].data_ptr()
        config.gate_scale = k2_weights["gate_scales"].data_ptr()
        config.up_scale = k2_weights["up_scales"].data_ptr()
        config.down_scale = k2_weights["down_scales"].data_ptr()
    else:
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

    # Bug #23 fix: Set quant_config for AWQ/K2 modes
    # Bug #25 fix: AWQ (int4_1kgroup) uses zero_point, K2 (int4_kgroup) does NOT
    if quant_mode == "int4_1kgroup":  # AWQ supports zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = True
    elif quant_mode == "int4_kgroup":  # K2 does NOT support zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False

    # Create MOE SFT instance based on quant_mode
    MOE_SFT_CLASS = get_moe_sft_class(quant_mode)
    moe = MOE_SFT_CLASS(config)
    print(f"[INFO] Using {quant_mode.upper()} MOE SFT class: {MOE_SFT_CLASS.__name__}")

    # Load base weights
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    # Warm up
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    # Get threshold for this quant_mode
    threshold = get_threshold(quant_mode)

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

        # AMX forward using forward_sft_task
        output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
        CPUInfer.submit(
            moe.forward_sft_task(
                bsz_tensor.data_ptr(),
                num_experts_per_tok,
                expert_ids.data_ptr(),
                weights.data_ptr(),
                input_data.data_ptr(),
                output.data_ptr(),
                False,  # save_for_backward=False to avoid cache overflow
            )
        )
        CPUInfer.sync()

        # Compare results
        diff = torch.mean(torch.abs(output - torch_output)) / (torch.mean(torch.abs(torch_output)) + 1e-8)
        print(f"Relative difference: {diff:.6f}")

        if diff < threshold:
            print(f"PASSED (threshold: {threshold})")
        else:
            print(f"FAILED: diff={diff:.6f} >= {threshold}")
            # Don't exit immediately, continue to show all iterations

    print(f"\n--- Final Result ---")
    if diff < threshold:
        print(f"[OK] MOE SFT Forward Pass Test - {quant_mode.upper()} mode (NO TP) PASSED")
    else:
        print(f"[FAILED] MOE SFT Forward Pass Test - {quant_mode.upper()} mode (NO TP) FAILED")
        print(f"  This means the bug is in the basic SFT forward logic, not TP partitioning.")
        sys.exit(1)


def test_moe_sft_backward_no_tp(quant_mode: str = "bf16"):
    """
    Test MOE SFT backward pass accuracy with single NUMA node (no TP).

    Compares the AMX implementation gradients against PyTorch reference.
    Uses WorkerPoolConfig to force single subpool.

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    print(f"\n{'='*60}")
    print(f"Testing MOE SFT Backward Pass - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize weights based on quant_mode
    k2_weights = None  # Will be set for K2 mode
    if quant_mode == "int4_kgroup":
        # K2 needs pre-quantized int4 weights
        k2_weights = init_base_weights_for_k2(expert_num, hidden_size, intermediate_size, group_size=128)
        # Use original BF16 for reference computation
        gate_proj = k2_weights["gate_proj_bf16"]
        up_proj = k2_weights["up_proj_bf16"]
        down_proj = k2_weights["down_proj_bf16"]
    else:
        # Other modes use BF16 weights
        gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Make LoRA B non-zero for testing
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel_ext not available, cannot run test")
        sys.exit(1)

    # Initialize CPUInfer with single NUMA node configuration
    print("\n[INFO] Creating CPUInfer with single NUMA node (NO TP)...")
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)
    print("[INFO] CPUInfer created with single subpool (tp_count=1)")

    # Create MOE SFT config - max_cache_depth must match validation_iter for backward
    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = expert_num
    config.num_experts_per_tok = num_experts_per_tok
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.max_cache_depth = validation_iter  # Need cache for backward
    config.max_len = max_len
    config.layer_idx = 0

    # Bug #26 fix: K2 uses pre-quantized weights with scales
    if quant_mode == "int4_kgroup" and k2_weights is not None:
        config.gate_proj = k2_weights["gate_qweight"].data_ptr()
        config.up_proj = k2_weights["up_qweight"].data_ptr()
        config.down_proj = k2_weights["down_qweight"].data_ptr()
        config.gate_scale = k2_weights["gate_scales"].data_ptr()
        config.up_scale = k2_weights["up_scales"].data_ptr()
        config.down_scale = k2_weights["down_scales"].data_ptr()
    else:
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()

    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = CPUInfer.backend_

    # Bug #23 fix: Set quant_config for AWQ/K2 modes
    # Bug #25 fix: AWQ (int4_1kgroup) uses zero_point, K2 (int4_kgroup) does NOT
    if quant_mode == "int4_1kgroup":  # AWQ supports zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = True
    elif quant_mode == "int4_kgroup":  # K2 does NOT support zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False

    # Create MOE SFT instance based on quant_mode
    MOE_SFT_CLASS = get_moe_sft_class(quant_mode)
    moe = MOE_SFT_CLASS(config)
    print(f"[INFO] Using {quant_mode.upper()} MOE SFT class: {MOE_SFT_CLASS.__name__}")

    # Load base weights
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    # Warm up
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    # Get threshold for this quant_mode
    threshold = get_threshold(quant_mode, is_backward=True)

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

        # AMX forward (with save_for_backward=True)
        output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
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

        # Compare gradients (threshold already set before loop)
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

    print(f"\n[OK] MOE SFT Backward Pass Test - {quant_mode.upper()} mode (NO TP) PASSED")


def test_moe_sft_lora_weight_sync_no_tp(quant_mode: str = "bf16"):
    """
    Test LoRA weight synchronization with single NUMA node (no TP).

    Verifies that:
    1. Initial config correctly sets LoRA weight pointers (zero-copy)
    2. Modified weights are correctly reflected via update_lora_weights_task
    3. Forward pass uses the updated weights

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    print(f"\n{'='*60}")
    print(f"Testing LoRA Weight Synchronization - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel_ext not available, cannot run test")
        sys.exit(1)

    torch.manual_seed(42)

    # Initialize weights based on quant_mode
    k2_weights = None  # Will be set for K2 mode
    if quant_mode == "int4_kgroup":
        # K2 needs pre-quantized int4 weights
        k2_weights = init_base_weights_for_k2(expert_num, hidden_size, intermediate_size, group_size=128)
        # Use original BF16 for reference computation
        gate_proj = k2_weights["gate_proj_bf16"]
        up_proj = k2_weights["up_proj_bf16"]
        down_proj = k2_weights["down_proj_bf16"]
    else:
        # Other modes use BF16 weights
        gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    lora_weights = init_lora_weights(expert_num, hidden_size, intermediate_size, lora_rank)
    gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b = lora_weights

    # Initialize CPUInfer with single NUMA node
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)

    # Create MOE SFT config
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

    # Bug #26 fix: K2 uses pre-quantized weights with scales
    if quant_mode == "int4_kgroup" and k2_weights is not None:
        config.gate_proj = k2_weights["gate_qweight"].data_ptr()
        config.up_proj = k2_weights["up_qweight"].data_ptr()
        config.down_proj = k2_weights["down_qweight"].data_ptr()
        config.gate_scale = k2_weights["gate_scales"].data_ptr()
        config.up_scale = k2_weights["up_scales"].data_ptr()
        config.down_scale = k2_weights["down_scales"].data_ptr()
    else:
        config.gate_proj = gate_proj.data_ptr()
        config.up_proj = up_proj.data_ptr()
        config.down_proj = down_proj.data_ptr()

    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = CPUInfer.backend_

    # Bug #23 fix: Set quant_config for AWQ/K2 modes
    # Bug #25 fix: AWQ (int4_1kgroup) uses zero_point, K2 (int4_kgroup) does NOT
    if quant_mode == "int4_1kgroup":  # AWQ supports zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = True
    elif quant_mode == "int4_kgroup":  # K2 does NOT support zero_point
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False

    # Create MOE SFT instance based on quant_mode
    MOE_SFT_CLASS = get_moe_sft_class(quant_mode)
    moe = MOE_SFT_CLASS(config)
    print(f"[INFO] Using {quant_mode.upper()} MOE SFT class: {MOE_SFT_CLASS.__name__}")

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

    # First forward with initial LoRA weights
    output1 = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output1.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()

    # Modify LoRA weights (simulating optimizer.step())
    gate_lora_a.add_(0.1)
    gate_lora_b.add_(0.1)
    up_lora_a.add_(0.1)
    up_lora_b.add_(0.1)
    down_lora_a.add_(0.1)
    down_lora_b.add_(0.1)

    # Bug #22 fix: After modifying LoRA weights, sync to kernel
    # (partitioned weights are copied, not zero-copy)
    CPUInfer.submit(
        moe.update_lora_weights_task(
            gate_lora_a.data_ptr(),
            gate_lora_b.data_ptr(),
            up_lora_a.data_ptr(),
            up_lora_b.data_ptr(),
            down_lora_a.data_ptr(),
            down_lora_b.data_ptr(),
        )
    )
    CPUInfer.sync()

    # Second forward with updated LoRA weights
    output2 = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output2.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()

    # Outputs should be different after weight update
    diff = torch.mean(torch.abs(output1 - output2))
    print(f"Output difference after weight update: {diff:.6f}")
    assert diff > 1e-6, "Outputs should differ after LoRA weight update"

    # Test explicit update_lora_weights_task (for when tensors are reallocated)
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
    output3 = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    CPUInfer.submit(
        moe.forward_sft_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output3.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()

    # Output3 should match output2 (same weights, different tensor locations)
    diff_same = torch.mean(torch.abs(output2 - output3))
    print(f"Output difference after pointer update (should be ~0): {diff_same:.6f}")
    assert diff_same < 1e-5, f"Outputs should match after pointer update: {diff_same:.6f}"

    print(f"[OK] LoRA Weight Synchronization Test - {quant_mode.upper()} mode (NO TP) PASSED")


def test_moe_sft_training_loop_no_tp(quant_mode: str = "bf16"):
    """
    Test complete training loop with single NUMA node (no TP).

    This simulates a real training scenario where:
    1. Forward pass computes output and saves activations
    2. Backward pass computes gradients for LoRA weights
    3. Optimizer updates LoRA weights
    4. Next forward uses updated weights (zero-copy via pointers)

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    print(f"\n{'='*60}")
    print(f"Testing Complete Training Loop - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Initialize base weights based on quant_mode
    k2_weights = None  # Will be set for K2 mode
    if quant_mode == "int4_kgroup":
        # K2 needs pre-quantized int4 weights
        k2_weights = init_base_weights_for_k2(expert_num, hidden_size, intermediate_size, group_size=128)
        # Use original BF16 for reference computation
        gate_proj = k2_weights["gate_proj_bf16"]
        up_proj = k2_weights["up_proj_bf16"]
        down_proj = k2_weights["down_proj_bf16"]
    else:
        # Other modes use BF16 weights
        gate_proj, up_proj, down_proj = init_base_weights(expert_num, hidden_size, intermediate_size)

    # Initialize LoRA weights as contiguous tensors
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

    # Initialize kt_kernel
    moe = None
    CPUInfer = None
    if HAS_KT_KERNEL:
        pool_config = kt_kernel_ext.WorkerPoolConfig()
        pool_config.subpool_count = 1
        pool_config.subpool_numa_map = [0]
        pool_config.subpool_thread_count = [num_threads]
        CPUInfer = kt_kernel_ext.CPUInfer(pool_config)

        # Create MOE SFT config
        config = kt_kernel_ext.moe.MOESFTConfig()
        config.expert_num = expert_num
        config.num_experts_per_tok = num_experts_per_tok
        config.hidden_size = hidden_size
        config.intermediate_size = intermediate_size
        config.lora_rank = lora_rank
        config.lora_alpha = lora_alpha
        config.max_cache_depth = 1  # One forward-backward pair at a time
        config.max_len = max_len
        config.layer_idx = 0

        # Bug #26 fix: K2 uses pre-quantized weights with scales
        if quant_mode == "int4_kgroup" and k2_weights is not None:
            config.gate_proj = k2_weights["gate_qweight"].data_ptr()
            config.up_proj = k2_weights["up_qweight"].data_ptr()
            config.down_proj = k2_weights["down_qweight"].data_ptr()
            config.gate_scale = k2_weights["gate_scales"].data_ptr()
            config.up_scale = k2_weights["up_scales"].data_ptr()
            config.down_scale = k2_weights["down_scales"].data_ptr()
        else:
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()

        config.gate_lora_a = gate_lora_a_param.data.data_ptr()
        config.gate_lora_b = gate_lora_b_param.data.data_ptr()
        config.up_lora_a = up_lora_a_param.data.data_ptr()
        config.up_lora_b = up_lora_b_param.data.data_ptr()
        config.down_lora_a = down_lora_a_param.data.data_ptr()
        config.down_lora_b = down_lora_b_param.data.data_ptr()
        config.pool = CPUInfer.backend_

        # Bug #23 fix: Set quant_config for AWQ/K2 modes
        # Bug #25 fix: AWQ (int4_1kgroup) uses zero_point, K2 (int4_kgroup) does NOT
        if quant_mode == "int4_1kgroup":  # AWQ supports zero_point
            config.quant_config.group_size = 128
            config.quant_config.zero_point = True
        elif quant_mode == "int4_kgroup":  # K2 does NOT support zero_point
            config.quant_config.group_size = 128
            config.quant_config.zero_point = False

        # Create MOE SFT instance based on quant_mode
        MOE_SFT_CLASS = get_moe_sft_class(quant_mode)
        moe = MOE_SFT_CLASS(config)
        print(f"[INFO] Using {quant_mode.upper()} MOE SFT class: {MOE_SFT_CLASS.__name__}")

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
            bsz_tensor = torch.tensor([qlen], device="cpu")

            # Forward pass (with save_for_backward=True)
            output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
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
            loss = torch.mean((output.float() - target.float()) ** 2)
            print(f"  Loss (AMX): {loss.item():.6f}")

            # Compute gradient of loss w.r.t. output
            grad_output = 2 * (output.float() - target.float()) / output.numel()
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

        # Print gradient norms to verify gradients are computed
        print(f"  gate_lora_a grad norm: {gate_lora_a_param.grad.norm().item():.6e}")
        print(f"  gate_lora_b grad norm: {gate_lora_b_param.grad.norm().item():.6e}")

        # Save weight snapshots before optimizer step
        gate_lora_a_before = gate_lora_a_param.data.clone()
        gate_lora_b_before = gate_lora_b_param.data.clone()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Calculate weight changes
        gate_a_diff = (gate_lora_a_param.data - gate_lora_a_before).abs().mean().item()
        gate_b_diff = (gate_lora_b_param.data - gate_lora_b_before).abs().mean().item()

        # Print weight norms with higher precision
        print(f"  gate_lora_a norm: {gate_lora_a_param.data.norm().item():.10f}")
        print(f"  gate_lora_b norm: {gate_lora_b_param.data.norm().item():.10f}")
        print(f"  gate_lora_a weight change (mean abs): {gate_a_diff:.10e}")
        print(f"  gate_lora_b weight change (mean abs): {gate_b_diff:.10e}")

        # Verify weights are actually being updated
        assert gate_a_diff > 0, "gate_lora_a weights should change after optimizer step"
        assert gate_b_diff > 0, "gate_lora_b weights should change after optimizer step"

    print(f"\n[OK] Training Loop Test - {quant_mode.upper()} mode (NO TP) PASSED")


# =============================================================================
# Performance Test Functions
# =============================================================================


def test_moe_sft_performance_no_tp(quant_mode: str = "bf16"):
    """
    Test MOE SFT performance (forward + backward latency and throughput) with NO TP.

    Measures:
    - Forward pass latency (ms)
    - Backward pass latency (ms)
    - Forward + Backward combined latency (ms)
    - Throughput (tokens/second)

    Args:
        quant_mode: Quantization mode, "bf16" or "int8"
    """
    import time

    print(f"\n{'='*60}")
    print(f"Performance Test - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  qlen (batch size): {perf_qlen}")
    print(f"  warmup iterations: {perf_warmup_iter}")
    print(f"  test iterations: {perf_test_iter}")
    print(f"  num_threads: {num_threads}")
    print(f"  TP mode: DISABLED (single NUMA node)")
    print(f"{'='*60}")

    if not HAS_KT_KERNEL:
        print("ERROR: kt_kernel_ext not available, cannot run performance test")
        sys.exit(1)

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

    # Initialize CPUInfer with single NUMA node configuration (NO TP)
    print("\n[INFO] Creating CPUInfer with single NUMA node (NO TP)...")
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 1
    pool_config.subpool_numa_map = [0]
    pool_config.subpool_thread_count = [num_threads]
    CPUInfer = kt_kernel_ext.CPUInfer(pool_config)
    print("[INFO] CPUInfer created with single subpool (tp_count=1)")

    # Create MOE SFT config
    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = expert_num
    config.num_experts_per_tok = num_experts_per_tok
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.lora_rank = lora_rank
    config.lora_alpha = lora_alpha
    config.max_cache_depth = 1  # Only need one for forward-backward pair
    config.max_len = max_len
    config.layer_idx = 0
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = CPUInfer.backend_

    # Create MOE SFT instance based on quant_mode
    MOE_SFT_CLASS = get_moe_sft_class(quant_mode)
    moe = MOE_SFT_CLASS(config)
    print(f"[INFO] Using {quant_mode.upper()} MOE SFT class: {MOE_SFT_CLASS.__name__}")

    # Load base weights
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    # Warm up
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()

    # Prepare test data
    bsz_tensor = torch.tensor([perf_qlen], device="cpu")
    expert_ids = (
        torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(perf_qlen)])
        .to(torch.int64)
        .contiguous()
    )
    weights = torch.rand((perf_qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    input_data = torch.randn((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
    output = torch.zeros((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    grad_output = torch.randn((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
    grad_input = torch.zeros((perf_qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    grad_gate_lora_a = torch.zeros_like(gate_lora_a)
    grad_gate_lora_b = torch.zeros_like(gate_lora_b)
    grad_up_lora_a = torch.zeros_like(up_lora_a)
    grad_up_lora_b = torch.zeros_like(up_lora_b)
    grad_down_lora_a = torch.zeros_like(down_lora_a)
    grad_down_lora_b = torch.zeros_like(down_lora_b)

    # =========================================================================
    # Warmup Phase
    # =========================================================================
    print(f"\n[INFO] Warmup phase ({perf_warmup_iter} iterations)...")
    for _ in range(perf_warmup_iter):
        # Forward pass
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

    # =========================================================================
    # Forward Performance Test
    # =========================================================================
    print(f"\n[INFO] Testing forward pass performance ({perf_test_iter} iterations)...")
    forward_times = []
    for _ in range(perf_test_iter):
        start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        forward_times.append((end_time - start_time) * 1000)  # Convert to ms

    # =========================================================================
    # Backward Performance Test
    # =========================================================================
    print(f"[INFO] Testing backward pass performance ({perf_test_iter} iterations)...")
    backward_times = []
    for _ in range(perf_test_iter):
        # Need a forward pass first to populate cache
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

        start_time = time.perf_counter()
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
        end_time = time.perf_counter()
        backward_times.append((end_time - start_time) * 1000)  # Convert to ms

    # =========================================================================
    # Combined Forward + Backward Performance Test
    # =========================================================================
    print(f"[INFO] Testing combined forward+backward performance ({perf_test_iter} iterations)...")
    combined_times = []
    for _ in range(perf_test_iter):
        start_time = time.perf_counter()

        # Forward pass
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

        end_time = time.perf_counter()
        combined_times.append((end_time - start_time) * 1000)  # Convert to ms

    # =========================================================================
    # Results Summary
    # =========================================================================
    import statistics

    avg_forward = statistics.mean(forward_times)
    std_forward = statistics.stdev(forward_times) if len(forward_times) > 1 else 0
    min_forward = min(forward_times)
    max_forward = max(forward_times)

    avg_backward = statistics.mean(backward_times)
    std_backward = statistics.stdev(backward_times) if len(backward_times) > 1 else 0
    min_backward = min(backward_times)
    max_backward = max(backward_times)

    avg_combined = statistics.mean(combined_times)
    std_combined = statistics.stdev(combined_times) if len(combined_times) > 1 else 0
    min_combined = min(combined_times)
    max_combined = max(combined_times)

    # Calculate throughput (tokens per second)
    forward_throughput = perf_qlen / (avg_forward / 1000)  # tokens/second
    backward_throughput = perf_qlen / (avg_backward / 1000)  # tokens/second
    combined_throughput = perf_qlen / (avg_combined / 1000)  # tokens/second

    print(f"\n{'='*60}")
    print(f"Performance Results - {quant_mode.upper()} mode (NO TP)")
    print(f"{'='*60}")
    print(f"\nForward Pass:")
    print(f"  Average latency: {avg_forward:.3f} ms (±{std_forward:.3f})")
    print(f"  Min latency:     {min_forward:.3f} ms")
    print(f"  Max latency:     {max_forward:.3f} ms")
    print(f"  Throughput:      {forward_throughput:.1f} tokens/s")

    print(f"\nBackward Pass:")
    print(f"  Average latency: {avg_backward:.3f} ms (±{std_backward:.3f})")
    print(f"  Min latency:     {min_backward:.3f} ms")
    print(f"  Max latency:     {max_backward:.3f} ms")
    print(f"  Throughput:      {backward_throughput:.1f} tokens/s")

    print(f"\nCombined (Forward + Backward):")
    print(f"  Average latency: {avg_combined:.3f} ms (±{std_combined:.3f})")
    print(f"  Min latency:     {min_combined:.3f} ms")
    print(f"  Max latency:     {max_combined:.3f} ms")
    print(f"  Throughput:      {combined_throughput:.1f} tokens/s")

    print(f"\n[OK] Performance Test - {quant_mode.upper()} mode (NO TP) completed")

    return {
        "quant_mode": quant_mode,
        "forward_avg_ms": avg_forward,
        "forward_std_ms": std_forward,
        "forward_throughput": forward_throughput,
        "backward_avg_ms": avg_backward,
        "backward_std_ms": std_backward,
        "backward_throughput": backward_throughput,
        "combined_avg_ms": avg_combined,
        "combined_std_ms": std_combined,
        "combined_throughput": combined_throughput,
    }


def run_performance_tests():
    """Run performance tests for AMXBF16 and AMXINT8 modes (NO TP)."""
    print("\n" + "=" * 70)
    print(" MOE SFT AMX Performance Test Suite - Non-TP Version")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  expert_num: {expert_num}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_experts_per_tok: {num_experts_per_tok}")
    print(f"  lora_rank: {lora_rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  perf_qlen: {perf_qlen}")
    print(f"  num_threads: {num_threads}")
    print(f"  TP mode: DISABLED (single NUMA node)")
    print("=" * 70)

    # Only test BF16 and INT8 as requested
    quant_modes = ["bf16", "int8"]

    results = []
    try:
        for quant_mode in quant_modes:
            result = test_moe_sft_performance_no_tp(quant_mode)
            results.append(result)

        # Print comparison table
        print("\n" + "=" * 70)
        print(" Performance Comparison Summary (NO TP)")
        print("=" * 70)
        print(f"\n{'Mode':<10} {'Forward(ms)':<15} {'Backward(ms)':<15} {'Combined(ms)':<15} {'Throughput(tok/s)':<20}")
        print("-" * 75)
        for r in results:
            print(
                f"{r['quant_mode'].upper():<10} "
                f"{r['forward_avg_ms']:<15.3f} "
                f"{r['backward_avg_ms']:<15.3f} "
                f"{r['combined_avg_ms']:<15.3f} "
                f"{r['combined_throughput']:<20.1f}"
            )
        print("-" * 75)

        # Calculate speedup if we have both results
        if len(results) == 2:
            bf16_combined = results[0]["combined_avg_ms"]
            int8_combined = results[1]["combined_avg_ms"]
            speedup = bf16_combined / int8_combined
            print(f"\nINT8 vs BF16 speedup: {speedup:.2f}x")

        print("\n" + "=" * 70)
        print(" PERFORMANCE TESTS COMPLETED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Performance test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all_tests():
    """Run all MOE SFT tests for all quantization modes in non-TP mode."""
    print("\n" + "=" * 70)
    print(" MOE SFT AMX Test Suite - Non-TP Version (Single NUMA Node)")
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
    print(f"  TP mode: DISABLED (single NUMA node)")
    print("=" * 70)

    # Quantization modes to test
    quant_modes = ["bf16", "int8"]
    # quant_modes = ["int4_1kgroup", "int4_kgroup"]
    # quant_modes = ["int4_kgroup"]

    try:
        for quant_mode in quant_modes:
            print(f"\n{'='*70}")
            print(f" Testing MOE SFT AMX - {quant_mode.upper()} Mode (NO TP)")
            print(f"{'='*70}")

            # Forward pass test
            test_moe_sft_forward_no_tp(quant_mode)

            # Backward pass test
            test_moe_sft_backward_no_tp(quant_mode)

            # Weight sync test
            test_moe_sft_lora_weight_sync_no_tp(quant_mode)

            # Full training loop test
            test_moe_sft_training_loop_no_tp(quant_mode)

        print("\n" + "=" * 70)
        print(" ALL TESTS PASSED!")
        print(f" Tested quantization modes: {', '.join(m.upper() for m in quant_modes)}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MOE SFT AMX Test Suite - Non-TP Version")
    parser.add_argument(
        "--mode",
        choices=["all", "accuracy", "perf"],
        default="all",
        help="Test mode: 'all' runs both, 'accuracy' runs correctness tests, 'perf' runs performance tests",
    )
    parser.add_argument(
        "--qlen",
        type=int,
        default=None,
        help=f"Override perf_qlen for performance tests (default: {perf_qlen})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help=f"Override warmup iterations for performance tests (default: {perf_warmup_iter})",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help=f"Override test iterations for performance tests (default: {perf_test_iter})",
    )
    args = parser.parse_args()

    # Override performance test parameters if specified
    if args.qlen is not None or args.warmup is not None or args.iter is not None:
        # Need to use global to modify module-level variables
        if args.qlen is not None:
            globals()["perf_qlen"] = args.qlen
        if args.warmup is not None:
            globals()["perf_warmup_iter"] = args.warmup
        if args.iter is not None:
            globals()["perf_test_iter"] = args.iter

    if args.mode == "all":
        run_all_tests()
        run_performance_tests()
    elif args.mode == "accuracy":
        run_all_tests()
    elif args.mode == "perf":
        run_performance_tests()
