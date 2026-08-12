# SPDX-License-Identifier: Apache-2.0

"""VLM Conv3D compatibility for the torch 2.9 KT training stack."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from packaging.version import Version


_MIN_SWIFT_VERSION = Version("4.4.2")
_MAX_SWIFT_VERSION = Version("4.5.0")
_SUPPORTED_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}


def _requires_conv3d_patch() -> bool:
    import torch

    version = Version(torch.__version__.split("+", 1)[0])
    return Version("2.9.0") <= version < Version("2.10.0")


def _is_supported_vlm(config: Any) -> bool:
    return (
        getattr(config, "model_type", None) in _SUPPORTED_MODEL_TYPES
        and getattr(config, "vision_config", None) is not None
    )


def _is_patch_active() -> bool:
    import torch

    original = getattr(torch.nn.Conv3d, "_original_forward", None)
    return original is not None and torch.nn.Conv3d.forward is not original


def _enable_swift_patch() -> None:
    try:
        swift_version_raw = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "torch 2.9.x VLM training with KTransformers requires ms-swift>=4.4.2,<4.5"
        ) from exc

    swift_version = Version(swift_version_raw)
    if not _MIN_SWIFT_VERSION <= swift_version < _MAX_SWIFT_VERSION:
        raise RuntimeError(
            f"unsupported ms-swift version {swift_version_raw}; expected "
            ">=4.4.2,<4.5 for torch 2.9.x VLM training"
        )

    if not _is_patch_active():
        try:
            importlib.import_module("swift.model.utils")
        except Exception as exc:
            raise RuntimeError(
                f"failed to activate the ms-swift Conv3D patch: {exc}"
            ) from exc

    if not _is_patch_active():
        raise RuntimeError("ms-swift did not patch torch.nn.Conv3d.forward")


def prepare_vlm_conv3d(config: Any) -> None:
    """Activate the verified Conv3D implementation before loading a KT VLM."""
    if _requires_conv3d_patch() and _is_supported_vlm(config):
        _enable_swift_patch()


def validate_vlm_conv3d(model: Any) -> list[str]:
    """Validate Conv3D modules against the replacement implementation contract."""
    if not _requires_conv3d_patch():
        return []

    _enable_swift_patch()

    import torch

    module_names: list[str] = []
    unsupported: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv3d):
            continue

        module_name = name or "<root>"
        module_names.append(module_name)
        reasons = []
        if tuple(module.stride) != tuple(module.kernel_size):
            reasons.append(
                f"stride={module.stride} != kernel_size={module.kernel_size}"
            )
        if any(value != 0 for value in module.padding):
            reasons.append(f"padding={module.padding} != 0")
        if any(value != 1 for value in module.dilation):
            reasons.append(f"dilation={module.dilation} != 1")
        if module.groups != 1:
            reasons.append(f"groups={module.groups} != 1")
        if reasons:
            unsupported.append(f"{module_name}: {', '.join(reasons)}")

    if unsupported:
        raise RuntimeError(
            "the ms-swift Conv3D replacement does not support: "
            + "; ".join(unsupported)
        )
    return module_names
