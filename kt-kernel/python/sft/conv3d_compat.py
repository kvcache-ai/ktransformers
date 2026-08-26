# SPDX-License-Identifier: Apache-2.0

"""Instance-scoped Qwen VLM compatibility for the KT SFT stack."""

from __future__ import annotations

from types import MethodType
from typing import Any

from packaging.version import Version


_SUPPORTED_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}
_COMPATIBLE_ATTR = "_kt_conv3d_compatible"
_ORIGINAL_FORWARD_ATTR = "_kt_original_conv3d_forward"


def _canonicalize_qwen3_vl_fused_expert_weights(
    gate_up: Any,
    down: Any,
    moe_config: Any,
) -> tuple[Any, Any, Any] | None:
    """Convert only the transposed Qwen3-VL fused expert layout to KT layout."""
    expert_num = moe_config.expert_num
    intermediate_size = moe_config.intermediate_size
    if gate_up.dim() != 3 or down.dim() != 3:
        return None

    hidden_size = gate_up.shape[1]
    if tuple(gate_up.shape) != (expert_num, hidden_size, 2 * intermediate_size):
        return None
    if tuple(down.shape) != (expert_num, intermediate_size, hidden_size):
        return None

    gate_proj, up_proj = (
        tensor.transpose(1, 2).contiguous()
        for tensor in gate_up.split(intermediate_size, dim=2)
    )
    return gate_proj, up_proj, down.transpose(1, 2).contiguous()


def _requires_conv3d_patch() -> bool:
    import torch

    version = Version(torch.__version__.split("+", 1)[0])
    return Version("2.9.0") <= version < Version("2.10.0")


def _is_supported_vlm(config: Any) -> bool:
    return (
        getattr(config, "model_type", None) in _SUPPORTED_MODEL_TYPES
        and getattr(config, "vision_config", None) is not None
    )


def _contract_errors(module: Any) -> list[str]:
    errors = []
    if tuple(module.stride) != tuple(module.kernel_size):
        errors.append(f"stride={module.stride} != kernel_size={module.kernel_size}")
    if any(value != 0 for value in module.padding):
        errors.append(f"padding={module.padding} != 0")
    if any(value != 1 for value in module.dilation):
        errors.append(f"dilation={module.dilation} != 1")
    if module.groups != 1:
        errors.append(f"groups={module.groups} != 1")
    return errors


def _kt_conv3d_forward(module: Any, inputs: Any) -> Any:
    """Evaluate the non-overlapping Conv3D contract using autograd-native ops."""
    import torch.nn.functional as functional

    batch_size = inputs.shape[0]
    kernel = tuple(module.kernel_size)
    inputs = (
        inputs.unfold(2, kernel[0], kernel[0])
        .unfold(3, kernel[1], kernel[1])
        .unfold(4, kernel[2], kernel[2])
    )
    depth, height, width = inputs.shape[2:5]
    inputs = inputs.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
        -1, module.in_channels * kernel[0] * kernel[1] * kernel[2]
    )
    outputs = functional.linear(
        inputs,
        module.weight.reshape(module.out_channels, -1),
        module.bias,
    )
    return outputs.view(batch_size, depth, height, width, module.out_channels).permute(
        0, 4, 1, 2, 3
    )


def patch_vlm_conv3d(model: Any) -> list[str]:
    """Patch supported Conv3D instances without changing ``torch.nn.Conv3d`` globally."""
    if not _requires_conv3d_patch() or not _is_supported_vlm(
        getattr(model, "config", None)
    ):
        return []

    import torch

    modules = [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Conv3d)
    ]
    unsupported = []
    for name, module in modules:
        errors = _contract_errors(module)
        if errors:
            unsupported.append(f"{name}: {', '.join(errors)}")

    if unsupported:
        raise RuntimeError(
            "the KT Conv3D fallback does not support: " + "; ".join(unsupported)
        )

    patched_names = []
    for name, module in modules:
        if not getattr(module, _COMPATIBLE_ATTR, False):
            setattr(module, _ORIGINAL_FORWARD_ATTR, module.forward)
            module.forward = MethodType(_kt_conv3d_forward, module)
            setattr(module, _COMPATIBLE_ATTR, True)
        patched_names.append(name)

    setattr(model, "_kt_vlm_conv3d_compatible", bool(modules))
    return patched_names


def is_vlm_conv3d_compatible(model: Any) -> bool:
    """Return whether every Conv3D instance on a torch 2.9 KT VLM was patched."""
    if not _requires_conv3d_patch():
        return True

    import torch

    modules = [
        module for module in model.modules() if isinstance(module, torch.nn.Conv3d)
    ]
    return bool(modules) and all(
        getattr(module, _COMPATIBLE_ATTR, False) for module in modules
    )
