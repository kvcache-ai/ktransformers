"""Pure-Python tests for MESH CUDA graph compatibility checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "python" / "utils" / "mesh" / "runtime_helpers.py"
SPEC = importlib.util.spec_from_file_location("mesh_runtime_helpers", MODULE_PATH)
runtime_helpers = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(runtime_helpers)


class _CPUOnlyInfer:
    def submit(self, task):
        del task

    def sync(self, allow_pending=0):
        del allow_pending


class _CudaHookInfer:
    def __init__(self):
        self.submitted = []
        self.synced = []

    def submit_with_cuda_stream(self, cuda_stream, task):
        self.submitted.append((cuda_stream, task))

    def sync_with_cuda_stream(self, cuda_stream, allow_pending=0):
        self.synced.append((cuda_stream, allow_pending))


class _Wrapper:
    _submit_cpuinfer_task = runtime_helpers._submit_cpuinfer_task
    _sync_cpuinfer = runtime_helpers._sync_cpuinfer
    _cpuinfer_cuda_stream_hooks_available = runtime_helpers._cpuinfer_cuda_stream_hooks_available
    _ensure_cuda_graph_stream_compatible = runtime_helpers._ensure_cuda_graph_stream_compatible

    def __init__(self, cpu_infer, capture_active=True):
        self.cpu_infer = cpu_infer
        self.capture_active = capture_active

    def _cuda_graph_capture_active(self):
        return self.capture_active


def test_cpu_only_cpuinfer_rejects_cuda_graph_capture():
    wrapper = _Wrapper(_CPUOnlyInfer(), capture_active=True)

    with pytest.raises(RuntimeError, match="CPUINFER_USE_CUDA=1"):
        wrapper._submit_cpuinfer_task(("fn", "args"), cuda_stream=123)

    with pytest.raises(RuntimeError, match="--disable-cuda-graph"):
        wrapper._sync_cpuinfer(0, cuda_stream=123)


def test_cuda_stream_hooks_are_used_during_cuda_graph_capture():
    cpu_infer = _CudaHookInfer()
    wrapper = _Wrapper(cpu_infer, capture_active=True)

    wrapper._submit_cpuinfer_task("task", cuda_stream=123)
    wrapper._sync_cpuinfer(1, cuda_stream=123)

    assert cpu_infer.submitted == [(123, "task")]
    assert cpu_infer.synced == [(123, 1)]

