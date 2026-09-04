# SFT backend selection and runtime capability reporting
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import warnings

INT8_BACKEND = "INT8"
INT8_SFT_METHOD = "INT8_SFT"
INT8_WEIGHT_LAYOUT = "kt-int8-n32-k64-vnni-v1"
_INT8_SFT_METHOD_ALIASES = frozenset({INT8_SFT_METHOD, "AMXINT8_SFT"})
FP8_BACKEND = "FP8"
FP8_SFT_METHOD = "AMXFP8_SFT"
FP8_WEIGHT_LAYOUT = "block-e4m3-128x128"
FP8_KERNEL = "avx512-fp8-decode-bf16"
_FP8_SFT_METHOD_ALIASES = frozenset({FP8_SFT_METHOD, "FP8_SFT"})
RAWINT4_BACKEND = "RAWINT4"
RAWINT4_SFT_METHOD = "RAWINT4_SFT"
RAWINT4_WEIGHT_LAYOUT = "compressed-tensors-rawint4-g32-v1"
RAWINT4_KERNEL = "amx-int4-kgroup-g32"
RAWINT4_GROUP_SIZE = 32
_RAWINT4_SFT_METHOD_ALIASES = frozenset({RAWINT4_SFT_METHOD, "AMXINT4_KGroup_SFT"})


def is_int8_sft_method(method: str) -> bool:
    return str(method) in _INT8_SFT_METHOD_ALIASES


def is_fp8_sft_method(method: str) -> bool:
    return str(method) in _FP8_SFT_METHOD_ALIASES


def is_rawint4_sft_method(method: str) -> bool:
    return str(method) in _RAWINT4_SFT_METHOD_ALIASES


def normalize_sft_backend(
    backend: str,
    *,
    expert_weight_format: str | None,
) -> str:
    """Return the public, hardware-neutral backend name.

    ``AMXINT8`` remains an accepted input for existing configurations, but the
    selected backend is reported as ``INT8`` because the same packed weights
    run through either AMX-INT8 or AVX512-VNNI depending on the loaded wheel
    variant.
    """

    normalized = str(backend).strip().lower()
    if normalized == "auto":
        if expert_weight_format == "int8":
            return INT8_BACKEND
        if expert_weight_format == "fp8":
            return FP8_BACKEND
        if expert_weight_format == "rawint4":
            return RAWINT4_BACKEND
        return "AMXBF16"
    if normalized == "int8":
        return INT8_BACKEND
    if normalized == "amxint8":
        warnings.warn(
            "kt_backend='AMXINT8' is deprecated; use kt_backend='auto' " "or kt_backend='INT8'",
            FutureWarning,
            stacklevel=3,
        )
        return INT8_BACKEND
    if normalized == "fp8":
        return FP8_BACKEND
    if normalized == "amxfp8":
        warnings.warn(
            "kt_backend='AMXFP8' is deprecated; use kt_backend='auto' or "
            "kt_backend='FP8'",
            FutureWarning,
            stacklevel=3,
        )
        return FP8_BACKEND
    if normalized == "rawint4":
        return RAWINT4_BACKEND
    if normalized == "amxint4_kgroup":
        warnings.warn(
            "kt_backend='AMXINT4_KGroup' is deprecated; use " "kt_backend='auto' or kt_backend='RAWINT4'",
            FutureWarning,
            stacklevel=3,
        )
        return RAWINT4_BACKEND
    return str(backend)


@dataclass(frozen=True)
class INT8Runtime:
    cpu_variant: str
    kernel: str
    weight_layout: str


@dataclass(frozen=True)
class FP8Runtime:
    cpu_variant: str
    kernel: str
    weight_layout: str


@dataclass(frozen=True)
class RAWINT4Runtime:
    cpu_variant: str
    kernel: str
    weight_layout: str


@dataclass(frozen=True)
class RAWINT4CheckpointContract:
    bits: int = 4
    group_size: int = RAWINT4_GROUP_SIZE
    signed: bool = True
    symmetric: bool = True
    zero_point: bool = False
    format: str = "pack-quantized"


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def get_rawint4_checkpoint_contract(model_config: Any) -> RAWINT4CheckpointContract:
    """Validate the compressed-tensors layout consumed by KGroup SFT."""

    text_config = _config_value(model_config, "text_config")
    quant_config = _config_value(model_config, "quantization_config")
    if quant_config is None and text_config is not None:
        quant_config = _config_value(text_config, "quantization_config")
    if quant_config is None:
        raise ValueError("RAWINT4 SFT requires model quantization_config metadata")

    quant_method = str(_config_value(quant_config, "quant_method", "")).lower().replace("_", "-")
    weight_format = str(_config_value(quant_config, "format", "")).lower().replace("_", "-")
    status = str(_config_value(quant_config, "quantization_status", "")).lower()
    if quant_method != "compressed-tensors":
        raise ValueError("RAWINT4 SFT requires quant_method='compressed-tensors', " f"got {quant_method!r}")
    if weight_format != "pack-quantized":
        raise ValueError("RAWINT4 SFT requires format='pack-quantized', " f"got {weight_format!r}")
    if status != "compressed":
        raise ValueError("RAWINT4 SFT requires quantization_status='compressed', " f"got {status!r}")

    groups = _config_value(quant_config, "config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("RAWINT4 SFT requires non-empty quantization_config.config_groups")

    checked = 0
    for group_name, group in groups.items():
        if (
            _config_value(group, "input_activations") is not None
            or _config_value(group, "output_activations") is not None
        ):
            raise ValueError(f"RAWINT4 SFT group {group_name!r} must be weight-only quantization")
        weights = _config_value(group, "weights")
        if weights is None:
            continue
        checked += 1
        expected = {
            "type": "int",
            "num_bits": 4,
            "strategy": "group",
            "group_size": RAWINT4_GROUP_SIZE,
            "dynamic": False,
            "symmetric": True,
            "actorder": None,
            "block_structure": None,
        }
        observed = {name: _config_value(weights, name) for name in expected}
        if observed != expected:
            raise ValueError(
                f"RAWINT4 SFT group {group_name!r} has unsupported weight scheme: "
                f"expected {expected}, got {observed}"
            )
        explicit_zero_point = (
            "zero_point" in weights
            if isinstance(weights, Mapping)
            else _config_value(weights, "zero_point") is not None
        )
        if explicit_zero_point:
            raise ValueError(f"RAWINT4 SFT group {group_name!r} must not declare zero-point metadata")
    if checked == 0:
        raise ValueError("RAWINT4 SFT quantization_config has no weight scheme")
    return RAWINT4CheckpointContract()


def get_int8_runtime() -> INT8Runtime:
    """Validate and describe the INT8 SFT implementation actually loaded."""

    import kt_kernel

    extension = kt_kernel.kt_kernel_ext
    required_metadata = ("__cpu_variant__", "__int8_kernel__", "__int8_weight_layout__")
    missing_metadata = [name for name in required_metadata if not hasattr(extension, name)]
    if missing_metadata:
        raise RuntimeError(
            "The loaded kt-kernel extension predates production INT8 runtime metadata "
            f"({missing_metadata}); install a newly built multi-variant wheel."
        )
    moe_extension = getattr(extension, "moe", None)
    if moe_extension is None or not hasattr(moe_extension, "AMXInt8_SFT_MOE"):
        raise RuntimeError(
            "The loaded kt-kernel extension does not provide INT8 SFT. "
            "Install a wheel containing the AVX512-BF16 or AMX CPU variant."
        )

    cpu_variant = str(extension.__cpu_variant__).lower()
    kernel = str(extension.__int8_kernel__).lower()

    if kernel not in {"amx-int8", "avx512-vnni", "onednn-vnni"}:
        raise RuntimeError(
            "INT8 SFT requires an AMX-INT8 or AVX512-VNNI+BF16 extension "
            "(native or oneDNN); "
            f"loaded cpu_variant={cpu_variant!r}, effective_kernel={kernel!r}"
        )

    layout = str(extension.__int8_weight_layout__)
    if layout != INT8_WEIGHT_LAYOUT:
        raise RuntimeError(
            "INT8 SFT weight-layout mismatch: " f"expected {INT8_WEIGHT_LAYOUT!r}, extension reports {layout!r}"
        )
    return INT8Runtime(
        cpu_variant=cpu_variant,
        kernel=kernel,
        weight_layout=layout,
    )


def get_fp8_runtime() -> FP8Runtime:
    """Validate and describe the native block-FP8 SFT implementation."""

    import kt_kernel

    extension = kt_kernel.kt_kernel_ext
    required_metadata = ("__cpu_variant__", "__fp8_kernel__", "__fp8_weight_layout__")
    missing_metadata = [name for name in required_metadata if not hasattr(extension, name)]
    if missing_metadata:
        raise RuntimeError(
            "The loaded kt-kernel extension predates native FP8 runtime metadata "
            f"({missing_metadata}); install a newly built AVX512-BF16+VBMI wheel."
        )
    moe_extension = getattr(extension, "moe", None)
    if moe_extension is None or not hasattr(moe_extension, "AMXFP8_SFT_MOE"):
        raise RuntimeError(
            "The loaded kt-kernel extension does not provide native FP8 SFT. "
            "Install a wheel containing AMXFP8_SFT_MOE."
        )

    cpu_variant = str(extension.__cpu_variant__).lower()
    kernel = str(extension.__fp8_kernel__).lower()
    if kernel != FP8_KERNEL:
        raise RuntimeError(
            "Native FP8 SFT requires the AVX512-BF16+VBMI FP8 decode kernel; "
            f"loaded cpu_variant={cpu_variant!r}, effective_kernel={kernel!r}"
        )
    layout = str(extension.__fp8_weight_layout__)
    if layout != FP8_WEIGHT_LAYOUT:
        raise RuntimeError(
            "FP8 SFT weight-layout mismatch: "
            f"expected {FP8_WEIGHT_LAYOUT!r}, extension reports {layout!r}"
        )
    return FP8Runtime(
        cpu_variant=cpu_variant,
        kernel=kernel,
        weight_layout=layout,
    )


def get_rawint4_runtime() -> RAWINT4Runtime:
    """Validate and describe the signed group-32 RAWINT4 SFT runtime."""

    import kt_kernel

    extension = kt_kernel.kt_kernel_ext
    required_metadata = (
        "__cpu_variant__",
        "__rawint4_kernel__",
        "__rawint4_weight_layout__",
    )
    missing_metadata = [name for name in required_metadata if not hasattr(extension, name)]
    if missing_metadata:
        raise RuntimeError(
            "The loaded kt-kernel extension predates RAWINT4 SFT runtime metadata "
            f"({missing_metadata}); install a wheel containing the KGroup SFT backend."
        )
    moe_extension = getattr(extension, "moe", None)
    if moe_extension is None or not hasattr(moe_extension, "AMXInt4_KGroup_SFT_MOE"):
        raise RuntimeError(
            "The loaded kt-kernel extension does not provide RAWINT4 SFT. "
            "Install a wheel containing AMXInt4_KGroup_SFT_MOE."
        )

    cpu_variant = str(extension.__cpu_variant__).lower()
    kernel = str(extension.__rawint4_kernel__).lower()
    if kernel != RAWINT4_KERNEL:
        raise RuntimeError(
            "RAWINT4 SFT requires the signed group-32 AMX KGroup kernel; "
            f"loaded cpu_variant={cpu_variant!r}, effective_kernel={kernel!r}"
        )
    layout = str(extension.__rawint4_weight_layout__)
    if layout != RAWINT4_WEIGHT_LAYOUT:
        raise RuntimeError(
            "RAWINT4 SFT weight-layout mismatch: " f"expected {RAWINT4_WEIGHT_LAYOUT!r}, extension reports {layout!r}"
        )
    return RAWINT4Runtime(
        cpu_variant=cpu_variant,
        kernel=kernel,
        weight_layout=layout,
    )
