# SPDX-License-Identifier: Apache-2.0
"""Check that a backward transfer across a group boundary preserves packed weights."""

import json
import platform
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file


def decode(codes, scales):
    magnitudes = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    bits = np.stack((codes & 15, codes >> 4), axis=-1).reshape(codes.shape[0], -1)
    values = magnitudes[bits & 7] * np.where(bits & 8, -1, 1)
    return values * np.exp2(scales.astype(np.float32) - 127).repeat(32, axis=1)


@pytest.mark.skipif(
    platform.machine() != "x86_64" or platform.system() != "Linux" or not shutil.which("g++"),
    reason="The example uses the Linux x86-64 OpenMP host transpose",
)
def test_grouped_transpose(tmp_path, monkeypatch):
    from kt_kernel.sft.npu_mxfp4 import NpuExpertStore

    source = Path(__file__).resolve().parents[2] / "examples/npu_mxfp4/transpose.cpp"
    library = tmp_path / "transpose.so"
    subprocess.run(["g++", "-std=c++17", "-O2", "-fopenmp", "-shared", "-fPIC", source, "-o", library], check=True)
    rng = np.random.default_rng(817)
    active = [0, 15, 16, 32]
    tensors = {}
    for expert in active:
        for key, n, k in (("w1", 64, 128), ("w3", 64, 128), ("w2", 128, 64)):
            prefix = f"layers.0.ffn.experts.{expert}.{key}"
            tensors[prefix + ".weight"] = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
            tensors[prefix + ".scale"] = rng.integers(124, 130, (n, k // 32), dtype=np.uint8)
    name = "weights.safetensors"
    save_file(tensors, tmp_path / name)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(tensors, name)}))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    config = SimpleNamespace(n_routed_experts=128, hidden_size=128, moe_intermediate_size=64, swiglu_limit=10)
    store = NpuExpertStore(tmp_path, config, torch.device("cpu"), library)
    try:
        assert not list(store.backward_groups(0, []))
        groups = []
        for group, slab, offsets in store.backward_groups(0, [(e, None, None) for e in active]):
            groups.append([e for e, _, _ in group])
            for expert in groups[-1]:
                for i, (key, n, k) in enumerate(store.projections):
                    prefix = f"layers.0.ffn.experts.{expert}.{key}"
                    expected = decode(tensors[prefix + ".weight"], tensors[prefix + ".scale"]).T
                    actual = np.zeros((k, n), dtype=np.float32)
                    for w, s in store.transposed(slab, offsets, expert, i):
                        codes = w.T.numpy().view(np.uint8)
                        scales = s.transpose(0, 1).reshape(k, -1).numpy()
                        actual += decode(codes, scales)
                    np.testing.assert_array_equal(actual, expected)
        assert groups == [[0, 15], [16], [32]]
    finally:
        store.close()
