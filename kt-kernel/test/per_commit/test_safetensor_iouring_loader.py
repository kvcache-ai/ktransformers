"""Pure-Python tests for SafeTensorLoader io_uring file-slot views."""

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


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array

    def to(self, device: str):
        del device
        return self

    def contiguous(self):
        return self


class _FakeSafeOpen:
    def __init__(self, file_path: str, framework: str = "pt"):
        del framework
        self.file_path = file_path
        with open(file_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        self._keys = [key for key in header if key != "__metadata__"]

    def keys(self):
        return list(self._keys)

    def get_tensor(self, key):
        raise KeyError(key)

    def close(self):
        return None


def _import_loader_module():
    spec = importlib.util.spec_from_file_location("loader_iouring_test", MODULE_PATH)
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


def _write_minimal_amx_expert_file(path):
    tensors = {}
    for proj in ("gate", "up", "down"):
        tensors[f"blk.0.ffn_{proj}_exps.0.numa.0.weight"] = np.arange(3, dtype=np.uint8)
        tensors[f"blk.0.ffn_{proj}_exps.0.numa.0.scale"] = np.arange(3, dtype=np.float32)
    save_file(tensors, str(path))


def test_load_experts_iouring_rejects_unaligned_direct_io(tmp_path):
    _write_minimal_amx_expert_file(tmp_path / "model.safetensors")

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))

    with pytest.raises(RuntimeError, match="512-byte aligned"):
        loader.load_experts_iouring("blk.0", use_direct_io=True)


def test_load_experts_iouring_allows_buffered_debug_path(tmp_path):
    _write_minimal_amx_expert_file(tmp_path / "model.safetensors")

    loader_module = _import_loader_module()
    loader = loader_module.SafeTensorLoader(str(tmp_path))
    experts = loader.load_experts_iouring("blk.0", use_direct_io=False)

    assert experts["direct_io"] is False
    assert len(experts["gate"]) == 1
    assert len(experts["gate"][0]) == 1
    assert experts["gate"][0][0][0] >= 0
    loader.close_all_handles()


def test_bf16_loader_builds_iouring_slots_for_packed_experts(tmp_path):
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
    slots = loader.load_experts_iouring("model.layers.0", tp_count=2, use_direct_io=False)

    assert slots["packed_bf16"] is True
    assert len(slots["gate"]) == 2
    assert len(slots["gate"][0]) == 2
    assert slots["gate"][0][0][2] == slots["up"][0][0][2]
    assert slots["down"][0][0][2] == down[0].nbytes
    loader.close_all_handles()
