"""Memory-efficient norm reduction for KT-owned CPU BF16 gradients."""

from __future__ import annotations

import torch

from . import kt_kernel_ext


if not hasattr(kt_kernel_ext, "_bf16_sumsq_ptr"):
    raise ImportError("The loaded kt-kernel extension does not support CPU BF16 gradient norm reduction")


def bf16_sumsq(tensor: torch.Tensor) -> float:
    """Return sum(float(value) ** 2) without materializing a float tensor."""
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
        raise TypeError("bf16_sumsq requires a strided CPU BF16 tensor")
    if not tensor.is_contiguous():
        raise ValueError("bf16_sumsq requires a contiguous tensor")
    return kt_kernel_ext._bf16_sumsq_ptr(tensor.data_ptr(), tensor.numel())
