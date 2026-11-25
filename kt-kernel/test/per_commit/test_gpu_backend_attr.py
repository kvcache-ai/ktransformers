#!/usr/bin/env python
# coding=utf-8
"""``kt_kernel_ext.gpu_backend`` must agree with the stream bindings the build actually has.

``gpu_backend`` is the single thing the Python side uses to decide how to schedule CPU-MoE work:
``experts_base._xpu_native_bridge_available()`` only takes the async stream-callback path when it is
``"xpu"`` *and* ``CPUInfer.submit_with_cuda_stream`` exists, and falls back to a synchronous submit
otherwise. The attribute comes from an ``#if`` chain in ``ext_bindings.cpp`` whose order matters:
the SYCL-iGPU build defines ``KTRANSFORMERS_USE_CUDA_HOST_CALLBACKS`` rather than
``KTRANSFORMERS_USE_CUDA``, so ``USE_SYCL`` has to be tested first, and the XPU arm has to stay
distinct from it. A reordering would not break the build -- it would silently change which code
path xpu takes at runtime -- so the invariants are asserted here instead.

Runs on any host: it only inspects the loaded extension module, never a GPU.
"""

import os
import sys

import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

if os.environ.get("KT_KERNEL_EXT_DIR"):
    sys.path.insert(0, os.environ["KT_KERNEL_EXT_DIR"])

# Register this test for CPU CI with estimated runtime of 10 seconds
register_cpu_ci(est_time=10, suite="default")

# ``except Exception`` and not ``except ImportError``: importing the package runs the CPU-variant
# probe, which raises RuntimeError when the installed extension needs a newer ISA than this host.
try:
    import kt_kernel  # import the package first, it registers kt_kernel_ext

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    HAS_DEPS = True
except Exception as e:
    HAS_DEPS = False
    import_error = f"{type(e).__name__}: {e}"

# Every value the #if chain in ext_bindings.cpp can produce.
KNOWN_BACKENDS = frozenset({"sycl", "cuda", "rocm", "musa", "maca", "ascend", "xpu", "cpu"})

# Entry points bound only under ``#if defined(KTRANSFORMERS_USE_XPU)``.
XPU_BRIDGE_BINDINGS = ("xpu_stream_bridge_mode", "xpu_stream_bridge_events", "shutdown_xpu_stream_bridge")

# CPUInfer methods bound only under ``#ifndef KTRANSFORMERS_CPU_ONLY``.
STREAM_CALLBACK_METHODS = ("submit_with_cuda_stream", "sync_with_cuda_stream")


def _require_deps():
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")


@pytest.mark.cpu
def test_gpu_backend_is_a_known_value():
    _require_deps()
    backend = getattr(kt_kernel_ext, "gpu_backend", None)
    assert isinstance(backend, str), f"kt_kernel_ext.gpu_backend is missing or not a string: {backend!r}"
    assert backend in KNOWN_BACKENDS, f"unknown kt_kernel_ext.gpu_backend {backend!r}, expected one of {KNOWN_BACKENDS}"


@pytest.mark.cpu
def test_gpu_backend_matches_stream_callback_bindings():
    """``gpu_backend == "cpu"`` and the stream-callback entry points must not disagree.

    A build that reports no GPU backend but still binds ``submit_with_cuda_stream`` would hand
    callers a method whose body compiles away to nothing: the CPU MoE would never be enqueued and
    the H2D copy would read an untouched output buffer, silently. The two come from different
    macros (the ``gpu_backend`` ``#if`` chain versus ``KTRANSFORMERS_CPU_ONLY``), so a CMake branch
    that selects no GPU backend has to define ``KTRANSFORMERS_CPU_ONLY`` as well.
    """
    _require_deps()
    backend = kt_kernel_ext.gpu_backend
    bound = [name for name in STREAM_CALLBACK_METHODS if hasattr(kt_kernel_ext.CPUInfer, name)]
    if backend == "cpu":
        assert bound == [], (
            f"gpu_backend is 'cpu' but CPUInfer still binds {bound} with an empty body; the CMake "
            "branch this was built from must add_compile_definitions(KTRANSFORMERS_CPU_ONLY=1)"
        )
    else:
        assert sorted(bound) == sorted(
            STREAM_CALLBACK_METHODS
        ), f"gpu_backend is {backend!r} but CPUInfer only binds {bound}"


@pytest.mark.cpu
def test_xpu_bridge_bindings_imply_the_xpu_backend():
    """The XPU bridge bindings and ``gpu_backend == "xpu"`` come from the same build flag.

    Both directions are asserted: ``experts_base`` keys off the attribute while the SGLang
    integration and the tests key off the bindings, so either one being present without the other
    is a silent mis-dispatch.
    """
    _require_deps()
    backend = kt_kernel_ext.gpu_backend
    present = [name for name in XPU_BRIDGE_BINDINGS if hasattr(kt_kernel_ext, name)]
    if backend == "xpu":
        assert sorted(present) == sorted(
            XPU_BRIDGE_BINDINGS
        ), f"gpu_backend is 'xpu' but the bridge bindings are incomplete: {present}"
        for name in STREAM_CALLBACK_METHODS:
            assert hasattr(kt_kernel_ext.CPUInfer, name), f"gpu_backend is 'xpu' but CPUInfer has no {name}"
    else:
        assert present == [], f"gpu_backend is {backend!r} but the extension exports the XPU bridge: {present}"


@pytest.mark.cpu
def test_experts_base_agrees_with_the_attribute(monkeypatch):
    """The probe on the per-layer forward path reads exactly these two facts."""
    _require_deps()
    from kt_kernel import experts_base

    expected = kt_kernel_ext.gpu_backend == "xpu" and hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream")
    # The probe memoizes for the lifetime of the process, and another test in this session may have
    # answered it against a stubbed extension; re-probe the real one.
    monkeypatch.setattr(experts_base, "_xpu_native_bridge", None)
    assert experts_base._xpu_native_bridge_available() is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
