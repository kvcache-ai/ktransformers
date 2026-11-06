# Accelerator-agnostic stream helpers for the CPU-MoE stream bridge
# SPDX-License-Identifier: Apache-2.0

"""
Thin wrappers over ``torch.cuda`` / ``torch.npu`` / ``torch.xpu``.

``CPUInfer.submit_with_cuda_stream`` / ``sync_with_cuda_stream`` take the stream as a raw
integer whose meaning depends on the accelerator the extension was built for:

* ``cuda`` -- ``torch.cuda.Stream.cuda_stream`` (``cudaStream_t``). ROCm (HIP) and MACA torch
  builds also report ``device.type == "cuda"`` and alias the CUDA names.
* ``npu``  -- ``torch.npu.Stream.npu_stream`` (``aclrtStream``).
* ``xpu``  -- ``torch.xpu.Stream.sycl_queue`` (``sycl::queue*``), consumed by
  ``cpu_backend/xpu_stream_bridge.cpp``.

These helpers let callers (e.g. the SGLang integration) obtain that handle, synchronize, or
query graph-capture state without branching on the device type themselves.

torch_musa registers its own ``musa`` device type instead of posing as CUDA, so MUSA is not
covered here (nor by ``experts_base._wait_device``); such callers pass the handle themselves.
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional, Union

import torch

DeviceLike = Union[torch.device, str]

# device.type -> attribute of ``torch.<type>.Stream`` holding the raw stream handle.
_STREAM_HANDLE_ATTR = {
    "cuda": "cuda_stream",
    "npu": "npu_stream",
    "xpu": "sycl_queue",
}


def _as_device(device: DeviceLike) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def device_module(device: DeviceLike) -> Optional[ModuleType]:
    """Return ``torch.cuda`` / ``torch.npu`` / ``torch.xpu`` for ``device``, else None.

    None for ``cpu``, for device types kt-kernel has no stream bridge for, and when the
    backend module is missing from this torch build (e.g. ``npu`` without torch_npu).
    HIP (ROCm) torch reports ``cuda`` and is served by ``torch.cuda``.
    """
    dev_type = _as_device(device).type
    if dev_type not in _STREAM_HANDLE_ATTR:
        return None
    return getattr(torch, dev_type, None)


def current_stream_handle(device: DeviceLike) -> int:
    """Raw handle of ``device``'s current stream, as taken by ``CPUInfer.*_with_cuda_stream``.

    Raises ValueError for ``cpu`` and for any device type without a stream bridge.
    """
    device = _as_device(device)
    mod = device_module(device)
    if mod is None:
        raise ValueError(
            f"kt_kernel: no accelerator stream handle for device {str(device)!r}; expected one "
            f"of {sorted(_STREAM_HANDLE_ATTR)} with its torch backend module available"
        )
    stream = mod.current_stream(device)
    return int(getattr(stream, _STREAM_HANDLE_ATTR[device.type]))


def synchronize(device: DeviceLike) -> None:
    """``torch.<backend>.synchronize(device)``; a no-op for ``cpu``."""
    device = _as_device(device)
    if device.type == "cpu":
        return
    mod = device_module(device)
    if mod is None:
        raise ValueError(f"kt_kernel: cannot synchronize {str(device)!r}: no torch backend module")
    mod.synchronize(device)


def is_capturing(device: DeviceLike) -> bool:
    """True iff ``device``'s current stream is being graph-captured; False on any failure.

    Never raises: callers use it to decide whether a ``synchronize()`` is legal right now.
    """
    try:
        mod = device_module(device)
        if mod is None:
            return False
        return bool(mod.is_current_stream_capturing())
    except Exception:
        return False


__all__ = [
    "DeviceLike",
    "device_module",
    "current_stream_handle",
    "synchronize",
    "is_capturing",
]
