"""Pure-Python tests for SafeTensorLoader mmap views."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
import types

import numpy as np
import pytest
from safetensors.numpy import save_file


MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python", "utils", "loader.py")


class _FakeSafeOpen:
    def __init__(self, file_path: str, framework: str = "pt"):
        del framework
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        self._keys = [key for key in header if key != "__metadata__"]

    def keys(self):
        return list(self._keys)

    def close(self):
        return None


def _import_loader_module():
    spec = importlib.util.spec_from_file_location("loader_mmap_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    fake_torch = types.ModuleType("torch")
    fake_safetensors = types.ModuleType("safetensors")
    fake_safetensors.safe_open = _FakeSafeOpen
    fake_gguf = types.ModuleType("gguf")
    fake_gguf_reader = types.ModuleType("gguf.gguf_reader")
    fake_gguf_reader.GGUFReader = object

    restore = {name: sys.modules.get(name) for name in ("torch", "safetensors", "gguf", "gguf.gguf_reader")}
    sys.modules["torch"] = fake_torch
    sys.modules["safetensors"] = fake_safetensors
    sys.modules["gguf"] = fake_gguf
    sys.modules["gguf.gguf_reader"] = fake_gguf_reader
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in restore.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_get_mmap_tensor_returns_file_backed_array(tmp_path):
    """SafeTensorLoader should expose zero-copy numpy views over safetensors data."""
    path = tmp_path / "model.safetensors"
    expected = np.arange(12, dtype=np.uint8).reshape(3, 4)
    save_file({"blk.0.ffn_up_exps.0.numa.0.weight": expected}, str(path))

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))
    tensor = loader.get_mmap_tensor("blk.0.ffn_up_exps.0.numa.0.weight")

    assert isinstance(tensor.base, np.memmap)
    assert tensor.flags.writeable is False
    assert np.array_equal(tensor, expected)


def test_load_experts_mmap_builds_dense_numa_expert_layout(tmp_path):
    """load_experts_mmap should return [numa][expert] arrays without copying."""
    tensors = {}
    expected_values = {}
    for numa_id in range(2):
        for expert_id in range(2):
            weight = np.full((2, 4), fill_value=10 * numa_id + expert_id, dtype=np.uint8)
            scale = np.full((2,), fill_value=0.5 + numa_id + expert_id, dtype=np.float32)
            for proj in ("gate", "up", "down"):
                weight_key = f"blk.0.ffn_{proj}_exps.{expert_id}.numa.{numa_id}.weight"
                scale_key = f"blk.0.ffn_{proj}_exps.{expert_id}.numa.{numa_id}.scale"
                tensors[weight_key] = weight
                tensors[scale_key] = scale
                expected_values[(proj, numa_id, expert_id, "weight")] = weight
                expected_values[(proj, numa_id, expert_id, "scale")] = scale
    save_file(tensors, str(tmp_path / "model.safetensors"))

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))
    experts = loader.load_experts_mmap("blk.0")

    assert len(experts["up"]) == 2
    assert len(experts["up"][0]) == 2
    assert isinstance(experts["gate"][1][1].base, np.memmap)
    assert np.array_equal(experts["gate"][1][1], expected_values[("gate", 1, 1, "weight")])
    assert np.array_equal(experts["up_scale"][0][1], expected_values[("up", 0, 1, "scale")])
    assert np.array_equal(experts["down"][1][0], expected_values[("down", 1, 0, "weight")])


def _write_minimal_amx_expert_file(path):
    tensors = {}
    for proj in ("gate", "up", "down"):
        tensors[f"blk.0.ffn_{proj}_exps.0.numa.0.weight"] = np.arange(3, dtype=np.uint8)
        tensors[f"blk.0.ffn_{proj}_exps.0.numa.0.scale"] = np.arange(3, dtype=np.float32)
    save_file(tensors, str(path))


def test_load_experts_iouring_rejects_unaligned_direct_io(tmp_path):
    """O_DIRECT mode must fail clearly instead of silently falling back to buffered reads."""
    _write_minimal_amx_expert_file(tmp_path / "model.safetensors")

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))

    with pytest.raises(RuntimeError, match="512-byte aligned"):
        loader.load_experts_iouring("blk.0", use_direct_io=True)


def test_load_experts_iouring_allows_buffered_debug_path(tmp_path):
    """Buffered io_uring remains available only when direct I/O is explicitly disabled."""
    _write_minimal_amx_expert_file(tmp_path / "model.safetensors")

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))
    experts = loader.load_experts_iouring("blk.0", use_direct_io=False)

    assert experts["direct_io"] is False
    assert len(experts["gate"]) == 1
    assert len(experts["gate"][0]) == 1
    assert experts["gate"][0][0][0] >= 0
    loader.close_all_handles()


def test_bf16_loader_loads_packed_experts_as_mmap_views(tmp_path):
    """BF16SafeTensorLoader should split packed expert tensors without copying."""
    gate_up = np.arange(2 * 8 * 4, dtype=np.uint16).reshape(2, 8, 4)
    down = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    save_file(
        {
            "model.layers.0.mlp.experts.gate_up_proj": gate_up,
            "model.layers.0.mlp.experts.down_proj": down,
        },
        str(tmp_path / "model.safetensors"),
    )

    loader_module = _import_loader_module()
    loader = loader_module.BF16SafeTensorLoader(str(tmp_path))
    experts = loader.load_experts_mmap("model.layers.0")

    assert len(experts["gate"]) == 2
    assert isinstance(experts["gate"][0].base.base, np.memmap)
    assert np.array_equal(experts["gate"][1], gate_up[1, :4, :])
    assert np.array_equal(experts["up"][0], gate_up[0, 4:, :])
    assert np.array_equal(experts["down"][1], down[1])
