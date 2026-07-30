# INT8 SFT backend selection and runtime capability reporting
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import warnings

INT8_BACKEND = "INT8"
INT8_SFT_METHOD = "INT8_SFT"
INT8_WEIGHT_LAYOUT = "kt-int8-n32-k64-vnni-v1"
_INT8_SFT_METHOD_ALIASES = frozenset({INT8_SFT_METHOD, "AMXINT8_SFT"})


def is_int8_sft_method(method: str) -> bool:
    return str(method) in _INT8_SFT_METHOD_ALIASES


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
        return INT8_BACKEND if expert_weight_format == "int8" else "AMXBF16"
    if normalized == "int8":
        return INT8_BACKEND
    if normalized == "amxint8":
        warnings.warn(
            "kt_backend='AMXINT8' is deprecated; use kt_backend='auto' " "or kt_backend='INT8'",
            FutureWarning,
            stacklevel=3,
        )
        return INT8_BACKEND
    return str(backend)


@dataclass(frozen=True)
class INT8Runtime:
    cpu_variant: str
    kernel: str
    weight_layout: str


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
