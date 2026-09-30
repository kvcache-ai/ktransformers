# SPDX-License-Identifier: Apache-2.0
"""Small, auditable MXFP4 LoRA reference used by CPU SFT tests.

Production code must never materialize a persistent BF16 copy of the routed
expert weights.  Tests intentionally provide both a dense oracle and a
group-at-a-time dX implementation so the transpose-free kernel can be checked
without treating its own output as ground truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import torch
import torch.nn.functional as F


MXFP4_GROUP_SIZE = 32
E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

LORA_NAMES = (
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
    "down_lora_a",
    "down_lora_b",
)


def _validate_mxfp4_layout(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[int, int]:
    if group_size != MXFP4_GROUP_SIZE:
        raise ValueError(
            f"native MXFP4 requires group_size={MXFP4_GROUP_SIZE}, got {group_size}"
        )
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed MXFP4 weights must be a rank-2 uint8 tensor")
    if scales.dtype != torch.bfloat16 or scales.ndim != 2:
        raise ValueError("MXFP4 scales must be a rank-2 BF16 tensor")
    n, packed_k = packed.shape
    k = packed_k * 2
    if k % group_size:
        raise ValueError(f"MXFP4 K={k} must be divisible by group_size={group_size}")
    expected_scales = (n, k // group_size)
    if tuple(scales.shape) != expected_scales:
        raise ValueError(
            f"MXFP4 scales have shape {tuple(scales.shape)}, expected {expected_scales}"
        )
    return n, k


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Decode low-nibble-first E2M1 bytes without applying group scales."""

    if packed.dtype != torch.uint8:
        raise ValueError("packed E2M1 storage must use torch.uint8")
    low = (packed & 0x0F).to(torch.long)
    high = ((packed >> 4) & 0x0F).to(torch.long)
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    return E2M1_VALUES.to(packed.device)[codes]


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack integer E2M1 codes, preserving the checkpoint nibble order."""

    if codes.ndim < 1 or codes.shape[-1] % 2:
        raise ValueError("E2M1 code count must be even")
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) > 15):
        raise ValueError("E2M1 codes must be in [0, 15]")
    codes = codes.to(torch.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def ue8m0_to_bf16(encoded: torch.Tensor) -> torch.Tensor:
    """Losslessly place an unsigned E8M0 exponent in BF16 exponent bits."""

    if encoded.dtype != torch.uint8:
        raise ValueError("UE8M0 scales must use torch.uint8 storage")
    encoded_i32 = encoded.to(torch.int32)
    bits = encoded_i32 << 7
    bits = torch.where(encoded_i32 == 0, 0x0040, bits).to(torch.int16)
    return bits.view(torch.bfloat16).contiguous()


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Dense test oracle for an ``[N, K]`` native MXFP4 matrix."""

    _, k = _validate_mxfp4_layout(packed, scales, group_size)
    decoded = unpack_e2m1(packed).reshape(packed.shape[0], k)
    expanded_scales = scales.float().repeat_interleave(group_size, dim=1)
    return decoded * expanded_scales


def transpose_free_mxfp4_dx(
    grad_output: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = MXFP4_GROUP_SIZE,
) -> torch.Tensor:
    """Reference ``dX = dY @ W`` using only original MXFP4 rows/groups.

    At no point is a transposed weight or full dequantized matrix constructed.
    The loop order mirrors the intended C++ kernel: each K slab owns its output,
    then walks original output rows N and decodes exactly one group at a time.
    """

    n, k = _validate_mxfp4_layout(packed, scales, group_size)
    if grad_output.ndim != 2 or grad_output.shape[1] != n:
        raise ValueError(
            f"grad_output must have shape [M, {n}], got {tuple(grad_output.shape)}"
        )
    grad_output = grad_output.float()
    grad_input = torch.zeros(
        (grad_output.shape[0], k), dtype=torch.float32, device=grad_output.device
    )
    packed_per_group = group_size // 2
    lut = E2M1_VALUES.to(packed.device)
    for group in range(k // group_size):
        k_begin = group * group_size
        packed_begin = group * packed_per_group
        for row in range(n):
            group_codes = packed[row, packed_begin : packed_begin + packed_per_group]
            low = (group_codes & 0x0F).to(torch.long)
            high = ((group_codes >> 4) & 0x0F).to(torch.long)
            codes = torch.stack((low, high), dim=-1).reshape(group_size)
            values = lut[codes] * scales[row, group].float()
            grad_input[:, k_begin : k_begin + group_size].add_(
                grad_output[:, row : row + 1] * values
            )
    return grad_input


class FrozenMXFP4Linear(torch.autograd.Function):
    """Test-only frozen linear whose backward uses the transpose-free oracle."""

    @staticmethod
    def forward(ctx, inputs, packed, scales):
        ctx.save_for_backward(packed, scales)
        ctx.input_dtype = inputs.dtype
        weight = dequantize_mxfp4(packed, scales)
        return inputs.float().matmul(weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        packed, scales = ctx.saved_tensors
        grad_input = transpose_free_mxfp4_dx(grad_output, packed, scales)
        return grad_input.to(ctx.input_dtype), None, None


def v4_swiglu(
    gate_raw: torch.Tensor, up_raw: torch.Tensor, limit: float
) -> torch.Tensor:
    """DeepSeek-V4 asymmetric clamped SwiGLU."""

    if not limit > 0:
        raise ValueError(f"DeepSeek-V4 swiglu limit must be positive, got {limit}")
    gate = torch.clamp(gate_raw, max=limit)
    up = torch.clamp(up_raw, min=-limit, max=limit)
    return F.silu(gate) * up


@dataclass(frozen=True)
class MXFP4Projection:
    packed: torch.Tensor
    scales: torch.Tensor


@dataclass(frozen=True)
class MXFP4ExpertWeights:
    gate: MXFP4Projection
    up: MXFP4Projection
    down: MXFP4Projection


def tensor_storage_hash(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        cpu_bytes = (
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
        digest.update(len(cpu_bytes).to_bytes(8, byteorder="little"))
        digest.update(cpu_bytes)
    return digest.hexdigest()


def base_storage_hash(experts: Iterable[MXFP4ExpertWeights]) -> str:
    tensors = []
    for expert in experts:
        for projection in (expert.gate, expert.up, expert.down):
            tensors.extend((projection.packed, projection.scales))
    return tensor_storage_hash(tensors)


def _base_linear(
    inputs: torch.Tensor,
    projection: MXFP4Projection,
    *,
    transpose_free_backward: bool,
) -> torch.Tensor:
    if transpose_free_backward:
        return FrozenMXFP4Linear.apply(inputs, projection.packed, projection.scales)
    weight = dequantize_mxfp4(projection.packed, projection.scales)
    return F.linear(inputs.float(), weight)


def _lora_linear(
    inputs: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return scaling * F.linear(F.linear(inputs.float(), lora_a), lora_b)


def routed_mxfp4_lora(
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    experts: list[MXFP4ExpertWeights],
    lora: Mapping[str, torch.Tensor],
    *,
    lora_scaling: float,
    swiglu_limit: float,
    transpose_free_backward: bool,
) -> torch.Tensor:
    """Unoptimized routed-expert LoRA equation used as a gradient oracle."""

    if expert_ids.ndim != 2 or route_weights.shape != expert_ids.shape:
        raise ValueError(
            "expert_ids and route_weights must have the same [M, top_k] shape"
        )
    if inputs.ndim != 2 or inputs.shape[0] != expert_ids.shape[0]:
        raise ValueError("inputs and routing tensors disagree on token count")
    missing = set(LORA_NAMES).difference(lora)
    if missing:
        raise ValueError(f"missing LoRA tensors: {sorted(missing)}")

    token_outputs = []
    for token_idx in range(inputs.shape[0]):
        routed_output = None
        token_input = inputs[token_idx : token_idx + 1]
        for slot in range(expert_ids.shape[1]):
            expert_idx = int(expert_ids[token_idx, slot])
            expert = experts[expert_idx]
            gate_raw = _base_linear(
                token_input,
                expert.gate,
                transpose_free_backward=transpose_free_backward,
            ) + _lora_linear(
                token_input,
                lora["gate_lora_a"][expert_idx],
                lora["gate_lora_b"][expert_idx],
                lora_scaling,
            )
            up_raw = _base_linear(
                token_input, expert.up, transpose_free_backward=transpose_free_backward
            ) + _lora_linear(
                token_input,
                lora["up_lora_a"][expert_idx],
                lora["up_lora_b"][expert_idx],
                lora_scaling,
            )
            intermediate = v4_swiglu(gate_raw, up_raw, swiglu_limit)
            expert_output = _base_linear(
                intermediate,
                expert.down,
                transpose_free_backward=transpose_free_backward,
            ) + _lora_linear(
                intermediate,
                lora["down_lora_a"][expert_idx],
                lora["down_lora_b"][expert_idx],
                lora_scaling,
            )
            weighted = expert_output * route_weights[token_idx, slot]
            routed_output = (
                weighted if routed_output is None else routed_output + weighted
            )
        token_outputs.append(routed_output)
    return torch.cat(token_outputs, dim=0)


def run_routed_reference(
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    experts: list[MXFP4ExpertWeights],
    lora: Mapping[str, torch.Tensor],
    grad_output: torch.Tensor,
    *,
    lora_scaling: float,
    swiglu_limit: float,
    transpose_free_backward: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    differentiable_input = inputs.float().detach().requires_grad_(True)
    differentiable_routes = route_weights.float().detach().requires_grad_(True)
    differentiable_lora = {
        name: tensor.float().detach().requires_grad_(True)
        for name, tensor in lora.items()
    }
    output = routed_mxfp4_lora(
        differentiable_input,
        expert_ids,
        differentiable_routes,
        experts,
        differentiable_lora,
        lora_scaling=lora_scaling,
        swiglu_limit=swiglu_limit,
        transpose_free_backward=transpose_free_backward,
    )
    targets = (
        differentiable_input,
        differentiable_routes,
        *(differentiable_lora[name] for name in LORA_NAMES),
    )
    gradients = torch.autograd.grad(
        output,
        targets,
        grad_outputs=grad_output.float(),
        retain_graph=False,
        create_graph=False,
    )
    return (
        output.detach(),
        gradients[0].detach(),
        gradients[1].detach(),
        {name: value.detach() for name, value in zip(LORA_NAMES, gradients[2:])},
    )


def relative_l2_and_cosine(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    difference = torch.linalg.vector_norm(actual - expected)
    expected_norm = torch.linalg.vector_norm(expected)
    actual_norm = torch.linalg.vector_norm(actual)
    relative_l2 = float(difference / torch.clamp(expected_norm, min=1.0e-12))
    cosine = float(
        torch.dot(actual, expected)
        / torch.clamp(actual_norm * expected_norm, min=1.0e-20)
    )
    return relative_l2, cosine
