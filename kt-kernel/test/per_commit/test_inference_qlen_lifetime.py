# SPDX-License-Identifier: Apache-2.0
"""CPU-only regressions for batch-size pointers consumed by async native tasks."""

import ctypes
import gc
import importlib.util
from pathlib import Path
import weakref
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")


@pytest.fixture
def buffers(monkeypatch):
    path = Path(__file__).parents[2] / "python/experts_base.py"
    spec = importlib.util.spec_from_file_location("kt_kernel._test_experts_base", path)
    module = importlib.util.module_from_spec(spec)
    # The buffer owner is pure Python; do not require a built native extension.
    fake_package = ModuleType("kt_kernel")
    fake_package.kt_kernel_ext = SimpleNamespace()
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "kt_kernel", fake_package)
        spec.loader.exec_module(module)
    cls = module.KExpertsCPUBuffer
    # Test object ownership without requiring a CUDA pinned allocator/device.
    for name in ("zeros", "full"):
        original = getattr(torch, name)

        def cpu_factory(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_factory)
    monkeypatch.setattr(cls, "capture_bs", [])
    monkeypatch.setattr(cls, "capture_buffers", {})
    monkeypatch.setattr(cls, "temp_bs", 0)
    monkeypatch.setattr(cls, "temp_buffer", ())
    monkeypatch.setattr(cls, "_qlen_cache", {}, raising=False)
    return cls


def get_buffer(cls, rows):
    return cls.get_buffer(torch.empty(rows, 4, dtype=torch.bfloat16), 2)


def test_native_qlen_pointer_survives_temporary_shape_replacement(buffers):
    old = get_buffer(buffers, 2)
    scalar = weakref.ref(old[5][0])
    raw_pointer = old[5][0].data_ptr()
    del old
    get_buffer(buffers, 5)
    gc.collect()
    # A native host callback retains only raw_pointer, not a Python tensor owner.
    assert scalar() is not None, "async native qlen pointer lost its owner on shape change"
    assert ctypes.c_int32.from_address(raw_pointer).value == 2


def test_qlen_address_is_stable_when_shape_returns(buffers):
    old = get_buffer(buffers, 17)
    addresses = [tensor.data_ptr() for tensor in old[5]]
    get_buffer(buffers, 64)
    again = get_buffer(buffers, 17)
    assert [tensor.data_ptr() for tensor in again[5]] == addresses
    assert all(tensor.item() == 17 for tensor in again[5])


def test_cached_lengths_are_distinct_and_not_mutated_by_other_shapes(buffers):
    old = get_buffer(buffers, 512)
    get_buffer(buffers, 22)
    assert all(tensor.dtype == torch.int32 and tensor.item() == 512 for tensor in old[5])
    assert all(tensor.item() == 22 for tensor in buffers.temp_buffer[5])


def test_repeated_shape_flips_reuse_a_small_immutable_length_cache(buffers):
    for _ in range(5):
        for rows in (2, 17, 512, 22):
            get_buffer(buffers, rows)
    assert len(buffers._qlen_cache) == 4
    for (rows, depth), lengths in buffers._qlen_cache.items():
        assert len(lengths) == depth
        assert all(length.item() == rows for length in lengths)


def test_buffer_depth_is_part_of_length_cache_key(buffers, monkeypatch):
    first = get_buffer(buffers, 17)[5]
    monkeypatch.setattr(buffers, "buffer_depth", 3)
    # Force allocation of a new temporary shape, then return with the new depth.
    get_buffer(buffers, 22)
    changed = get_buffer(buffers, 17)[5]
    assert len(first) == 2 and len(changed) == 3
    assert all(length.item() == 17 for length in first + changed)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
