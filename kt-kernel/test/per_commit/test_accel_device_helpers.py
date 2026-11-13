#!/usr/bin/env python
# coding=utf-8
"""Accelerator device-helper tests for KT-Kernel.

Checks ``kt_kernel.accel`` (per-device stream-handle / synchronize / capture-state helpers) and the
XPU fallback logic in ``kt_kernel.experts_base`` without any GPU: ``torch.xpu`` is replaced by a fake
module so the dispatch logic can be verified on CPU-only CI hosts.
"""

import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Register this test for CPU CI with estimated runtime of 30 seconds
register_cpu_ci(est_time=30, suite="default")

# Check if dependencies are available. ``except Exception`` and not ``except ImportError``: importing
# the package runs the CPU-variant probe, which raises RuntimeError when the installed extension needs
# a newer ISA than this host -- that has to degrade to a skip, not to a collection error.
try:
    import kt_kernel  # Import kt_kernel first to register kt_kernel_ext
    import torch
    from kt_kernel import accel, experts_base

    HAS_DEPS = True
except Exception as e:
    HAS_DEPS = False
    import_error = f"{type(e).__name__}: {e}"

# Value returned by the fake ``torch.xpu.current_stream().sycl_queue``.
FAKE_SYCL_QUEUE = 0x5A5A1234


def _require_deps():
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")


class _FakeStream:
    def __init__(self, sycl_queue):
        self.sycl_queue = sycl_queue


def _install_fake_xpu(monkeypatch, capturing=False, capture_raises=False):
    """Replace ``torch.xpu`` with a stand-in that records what the helpers call."""
    fake = types.SimpleNamespace(calls=[])
    fake.is_available = lambda: True
    fake.current_stream = lambda *args, **kwargs: _FakeStream(FAKE_SYCL_QUEUE)

    def synchronize(device=None):
        fake.calls.append(("synchronize", device))

    def is_current_stream_capturing():
        fake.calls.append(("is_current_stream_capturing",))
        if capture_raises:
            raise RuntimeError("fake torch.xpu: capture query failed")
        return capturing

    fake.synchronize = synchronize
    fake.is_current_stream_capturing = is_current_stream_capturing
    monkeypatch.setattr(torch, "xpu", fake, raising=False)
    return fake


def _synchronize_calls(fake):
    return [call for call in fake.calls if call[0] == "synchronize"]


class _FakeCPUInferNoStream:
    """CPUInfer of a CPU-only build: no stream-callback entry points."""


class _FakeCPUInferWithStream:
    """CPUInfer of a GPU build: has the stream-callback entry points."""

    def submit_with_cuda_stream(self, *args, **kwargs):
        raise AssertionError("submit_with_cuda_stream must not be called by these tests")

    def sync_with_cuda_stream(self, *args, **kwargs):
        raise AssertionError("sync_with_cuda_stream must not be called by these tests")


def _install_fake_ext(monkeypatch, module, gpu_backend, cpu_infer_cls):
    """Replace ``module.kt_kernel_ext`` with a stub exposing only the probed attributes."""
    ext = types.SimpleNamespace(CPUInfer=cpu_infer_cls)
    if gpu_backend is not None:
        ext.gpu_backend = gpu_backend
    monkeypatch.setattr(module, "kt_kernel_ext", ext)
    return ext


def _reset_xpu_bridge_memo(monkeypatch, module):
    """Forget the memoized ``_xpu_native_bridge_available()`` result.

    The memo may be a ``functools.lru_cache`` (``cache_clear``), a function attribute, or a module
    global following the ``_ascend_worker_degraded: Optional[bool] = None`` convention; handle all.
    """
    probe = getattr(module, "_xpu_native_bridge_available", None)
    assert probe is not None, "experts_base._xpu_native_bridge_available is missing"
    cache_clear = getattr(probe, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    for name, value in list(vars(probe).items()):
        if value is None or isinstance(value, bool):
            monkeypatch.setattr(probe, name, None)
    for name, value in list(vars(module).items()):
        if "xpu" in name.lower() and (value is None or isinstance(value, bool)):
            monkeypatch.setattr(module, name, None)


def _load_private_experts_base(name):
    """Execute a pristine copy of experts_base.py so one-time state (warnings, memos) is fresh."""
    spec = importlib.util.spec_from_file_location(name, Path(experts_base.__file__))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@pytest.mark.cpu
def test_package_exports_device_helpers():
    _require_deps()
    assert kt_kernel.current_stream_handle is accel.current_stream_handle
    assert kt_kernel.synchronize_device is accel.synchronize
    assert "current_stream_handle" in kt_kernel.__all__
    assert "synchronize_device" in kt_kernel.__all__


@pytest.mark.cpu
def test_cpu_device_helpers():
    _require_deps()
    with pytest.raises(ValueError):
        accel.current_stream_handle("cpu")
    with pytest.raises(ValueError):
        accel.current_stream_handle(torch.device("cpu"))
    with pytest.raises(ValueError):
        kt_kernel.current_stream_handle("cpu")
    assert accel.device_module(torch.device("cpu")) is None
    assert accel.synchronize(torch.device("cpu")) is None
    assert kt_kernel.synchronize_device("cpu") is None
    assert accel.is_capturing(torch.device("cpu")) is False


@pytest.mark.cpu
def test_fake_xpu_dispatch(monkeypatch):
    _require_deps()
    fake = _install_fake_xpu(monkeypatch)

    assert accel.device_module(torch.device("xpu")) is fake
    assert accel.current_stream_handle("xpu") == FAKE_SYCL_QUEUE
    assert accel.current_stream_handle(torch.device("xpu")) == FAKE_SYCL_QUEUE
    assert kt_kernel.current_stream_handle("xpu") == FAKE_SYCL_QUEUE
    assert isinstance(accel.current_stream_handle("xpu"), int)

    accel.synchronize(torch.device("xpu"))
    assert len(_synchronize_calls(fake)) == 1
    kt_kernel.synchronize_device("xpu")
    assert len(_synchronize_calls(fake)) == 2

    assert accel.is_capturing(torch.device("xpu")) is False
    _install_fake_xpu(monkeypatch, capturing=True)
    assert accel.is_capturing(torch.device("xpu")) is True


@pytest.mark.cpu
def test_is_capturing_false_when_query_raises(monkeypatch):
    _require_deps()
    _install_fake_xpu(monkeypatch, capture_raises=True)
    assert accel.is_capturing(torch.device("xpu")) is False


@pytest.mark.cpu
def test_bypass_stream_callback_without_native_xpu_bridge(monkeypatch):
    _require_deps()
    monkeypatch.delenv("KT_FORCE_SYNC_SUBMIT", raising=False)

    # CPU-only extension: no gpu_backend attribute, no stream-callback entry points.
    _install_fake_ext(monkeypatch, experts_base, gpu_backend=None, cpu_infer_cls=_FakeCPUInferNoStream)
    _reset_xpu_bridge_memo(monkeypatch, experts_base)
    assert experts_base._xpu_native_bridge_available() is False
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is True
    # cuda keeps the stream-callback path.
    assert experts_base._should_bypass_stream_callback(torch.device("cuda")) is False

    # The probe is memoized: swapping the extension underneath does not change the answer...
    _install_fake_ext(monkeypatch, experts_base, gpu_backend="xpu", cpu_infer_cls=_FakeCPUInferWithStream)
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is True
    # ...until the memo is reset.
    _reset_xpu_bridge_memo(monkeypatch, experts_base)
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is False

    # An extension built for another GPU backend does not provide the XPU bridge either.
    _install_fake_ext(monkeypatch, experts_base, gpu_backend="cuda", cpu_infer_cls=_FakeCPUInferWithStream)
    _reset_xpu_bridge_memo(monkeypatch, experts_base)
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is True


@pytest.mark.cpu
def test_no_bypass_with_native_xpu_bridge(monkeypatch):
    _require_deps()
    monkeypatch.delenv("KT_FORCE_SYNC_SUBMIT", raising=False)
    _install_fake_ext(monkeypatch, experts_base, gpu_backend="xpu", cpu_infer_cls=_FakeCPUInferWithStream)
    _reset_xpu_bridge_memo(monkeypatch, experts_base)

    assert experts_base._xpu_native_bridge_available() is True
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is False
    # An XPU build bridges the handle to a sycl::queue*, so any other device type must take the
    # synchronous path rather than have its cudaStream_t / aclrtStream reinterpreted.
    assert experts_base._should_bypass_stream_callback(torch.device("cuda")) is True

    # KT_FORCE_SYNC_SUBMIT=1 still forces the synchronous path on every device.
    monkeypatch.setenv("KT_FORCE_SYNC_SUBMIT", "1")
    assert experts_base._should_bypass_stream_callback(torch.device("xpu")) is True
    assert experts_base._should_bypass_stream_callback(torch.device("cuda")) is True


@pytest.mark.cpu
def test_bypass_warning_is_logged_once(monkeypatch, caplog):
    _require_deps()
    monkeypatch.delenv("KT_FORCE_SYNC_SUBMIT", raising=False)
    # A pristine module copy guarantees the one-time warning has not fired yet in this process.
    module = _load_private_experts_base("kt_experts_base_xpu_warning_under_test")
    _install_fake_ext(monkeypatch, module, gpu_backend=None, cpu_infer_cls=_FakeCPUInferNoStream)

    with caplog.at_level(logging.WARNING):
        assert module._should_bypass_stream_callback(torch.device("xpu")) is True
        assert module._should_bypass_stream_callback(torch.device("xpu")) is True
        assert module._should_bypass_stream_callback(torch.device("xpu")) is True

    warnings = [r.getMessage() for r in caplog.records if "CPUINFER_USE_XPU" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one rebuild hint, got {warnings!r}"
    assert "CPUINFER_USE_XPU=1" in warnings[0]


@pytest.mark.cpu
def test_wait_device_xpu_synchronizes_unless_capturing(monkeypatch):
    _require_deps()
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)

    fake = _install_fake_xpu(monkeypatch, capturing=False)
    experts_base._wait_device(torch.device("xpu"))
    assert len(_synchronize_calls(fake)) == 1

    # torch.xpu reports the current stream is capturing: no synchronize (it would raise mid-capture).
    fake = _install_fake_xpu(monkeypatch, capturing=True)
    experts_base._wait_device(torch.device("xpu"))
    assert _synchronize_calls(fake) == []

    # sglang's global capture flag also suppresses the synchronize.
    fake = _install_fake_xpu(monkeypatch, capturing=False)
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: True)
    experts_base._wait_device(torch.device("xpu"))
    assert _synchronize_calls(fake) == []

    # A failing capture query is treated as "not capturing".
    monkeypatch.setattr(experts_base, "_sglang_is_capture_mode", lambda: False)
    fake = _install_fake_xpu(monkeypatch, capture_raises=True)
    experts_base._wait_device(torch.device("xpu"))
    assert len(_synchronize_calls(fake)) == 1


@pytest.mark.cpu
def test_wait_device_cpu_is_noop(monkeypatch):
    _require_deps()
    fake = _install_fake_xpu(monkeypatch)
    experts_base._wait_device(torch.device("cpu"))
    assert fake.calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
