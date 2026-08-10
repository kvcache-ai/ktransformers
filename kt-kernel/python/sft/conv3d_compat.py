"""Conv3D compatibility support for VLM fine-tuning on torch 2.9.x."""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Any

from packaging.version import Version


MIN_SWIFT_VERSION = Version("4.4.2")
MAX_SWIFT_VERSION = Version("4.5.0")


@dataclass(frozen=True)
class Conv3DCompatibility:
    required: bool
    active: bool
    torch_version: str
    swift_version: str | None
    swift_module: str | None

    def to_dict(self) -> dict[str, bool | str | None]:
        return asdict(self)


def _is_torch_29(version: str) -> bool:
    parsed = Version(version.split("+", 1)[0])
    return Version("2.9.0") <= parsed < Version("2.10.0")


def is_swift_conv3d_patch_active() -> bool:
    """Return whether ms-swift replaced ``torch.nn.Conv3d.forward``."""
    import torch

    original = getattr(torch.nn.Conv3d, "_original_forward", None)
    return original is not None and torch.nn.Conv3d.forward is not original


def validate_swift_conv3d_modules(model: Any) -> list[str]:
    """Validate every Conv3D module against the ms-swift replacement contract."""
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
            "the verified ms-swift Conv3D replacement does not support: "
            + "; ".join(unsupported)
        )
    return module_names


def enable_swift_conv3d_patch() -> Conv3DCompatibility:
    """Load and verify the ms-swift Conv3D patch when torch 2.9.x needs it.

    ms-swift 4.x installs the patch while importing ``swift.model.utils``.
    This function must therefore run in every training rank before loading the
    VLM. It intentionally fails closed when a torch 2.9.x process cannot
    activate a supported ms-swift implementation.
    """
    import torch

    required = _is_torch_29(torch.__version__)
    if not required:
        return Conv3DCompatibility(False, False, torch.__version__, None, None)

    try:
        swift_version_raw = importlib.metadata.version("ms-swift")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "torch 2.9.x Qwen VLM training requires ms-swift>=4.4.2,<4.5; "
            "install it in the training environment before loading the model"
        ) from exc

    swift_version = Version(swift_version_raw)
    if not MIN_SWIFT_VERSION <= swift_version < MAX_SWIFT_VERSION:
        raise RuntimeError(
            f"unsupported ms-swift version {swift_version_raw}; expected >=4.4.2,<4.5 "
            "for the verified swift.model.utils Conv3D patch"
        )

    module_name = "swift.model.utils"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"failed to import the ms-swift Conv3D patch from {module_name}: {exc}"
        ) from exc

    if not is_swift_conv3d_patch_active():
        raise RuntimeError(
            "ms-swift was imported, but torch.nn.Conv3d.forward was not patched; "
            "the VLM must not be loaded in this process"
        )

    return Conv3DCompatibility(
        True, True, torch.__version__, swift_version_raw, module_name
    )


def self_test_swift_conv3d_patch() -> dict[str, bool | float | str | None]:
    """Check Qwen3.5-shaped Conv3D forward and backward against PyTorch."""
    import torch

    compatibility = enable_swift_conv3d_patch()
    if not compatibility.required:
        return {
            **compatibility.to_dict(),
            "self_test": "not_required",
            "max_abs_diff": 0.0,
        }

    original_forward = torch.nn.Conv3d._original_forward
    reference = torch.nn.Conv3d(
        3, 4, kernel_size=(2, 16, 16), stride=(2, 16, 16), bias=True
    ).double()
    patched = copy.deepcopy(reference)
    validate_swift_conv3d_modules(patched)
    reference_input = torch.randn(
        1, 3, 2, 16, 16, dtype=torch.float64, requires_grad=True
    )
    patched_input = reference_input.detach().clone().requires_grad_(True)

    expected = original_forward(reference, reference_input)
    actual = patched(patched_input)
    expected.square().sum().backward()
    actual.square().sum().backward()

    comparisons = {
        "output": (actual, expected),
        "input_grad": (patched_input.grad, reference_input.grad),
        "weight_grad": (patched.weight.grad, reference.weight.grad),
        "bias_grad": (patched.bias.grad, reference.bias.grad),
    }
    max_abs_diff = 0.0
    for name, (candidate, baseline) in comparisons.items():
        try:
            torch.testing.assert_close(candidate, baseline, rtol=1e-10, atol=1e-10)
        except AssertionError as exc:
            raise RuntimeError(
                f"ms-swift Conv3D {name} self-test failed: {exc}"
            ) from exc
        max_abs_diff = max(
            max_abs_diff, float((candidate - baseline).detach().abs().max())
        )

    return {
        **compatibility.to_dict(),
        "self_test": "passed",
        "max_abs_diff": max_abs_diff,
    }
