# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch")


SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
PACKAGE_NAME = "kt_sft_fp8_loader_under_test"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(SFT_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load_module(filename: str):
    name = f"{PACKAGE_NAME}.{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, SFT_PATH / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arch = _load_module("arch.py")
_load_module("dist_utils.py")
weights = _load_module("weights.py")


def _moe_config(expert_num=2, intermediate_size=128):
    return arch.MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=expert_num,
        intermediate_size=intermediate_size,
        num_experts_per_tok=1,
    )


def _write_checkpoint(tmp_path: Path, *, omit_key: str | None = None):
    tensors = {}
    prefix = "model.layers.0.mlp.experts"
    for expert_idx in range(2):
        for projection, shape in (
            ("gate_proj", (128, 128)),
            ("up_proj", (128, 128)),
            ("down_proj", (128, 128)),
        ):
            weight_key = f"{prefix}.{expert_idx}.{projection}.weight"
            scale_key = f"{prefix}.{expert_idx}.{projection}.weight_scale_inv"
            tensors[weight_key] = torch.zeros(shape, dtype=torch.float8_e4m3fn)
            scale_dtype = torch.bfloat16 if expert_idx == 0 else torch.float32
            tensors[scale_key] = torch.ones((1, 1), dtype=scale_dtype)
    if omit_key is not None:
        tensors.pop(omit_key)
    checkpoint = tmp_path / "model.safetensors"
    safetensors_torch.save_file(tensors, checkpoint)
    metadata = {"weight_map": {key: checkpoint.name for key in tensors}}
    return checkpoint, metadata


def test_raw_fp8_loader_preserves_per_expert_weights_and_normalizes_scales(tmp_path):
    checkpoint, metadata = _write_checkpoint(tmp_path)
    with patch.object(
        weights.torch,
        "stack",
        side_effect=AssertionError("native FP8 loader must not stack tensors"),
    ):
        loaded = weights.load_block_fp8_experts_from_checkpoint_files(
            [str(checkpoint)],
            metadata,
            "model.layers",
            _moe_config(),
            layer_idx=0,
            hidden_size=128,
        )

    assert isinstance(loaded, weights.BlockFP8ExpertWeights)
    assert loaded.block_size == (128, 128)
    for projection in (loaded.gate_proj, loaded.up_proj, loaded.down_proj):
        assert len(projection) == 2
        assert all(tensor.dtype == torch.float8_e4m3fn for tensor in projection)
        assert all(tensor.device.type == "cpu" for tensor in projection)
        assert all(tensor.is_contiguous() for tensor in projection)
    for scales in (loaded.gate_scale, loaded.up_scale, loaded.down_scale):
        assert len(scales) == 2
        assert all(scale.dtype == torch.float32 for scale in scales)
        assert all(tuple(scale.shape) == (1, 1) for scale in scales)


def test_raw_fp8_loader_can_index_an_unsharded_local_checkpoint(tmp_path):
    checkpoint, _ = _write_checkpoint(tmp_path)
    loaded = weights.load_block_fp8_experts_from_checkpoint_files(
        [str(checkpoint)],
        None,
        "model.layers",
        _moe_config(),
        layer_idx=0,
        hidden_size=128,
    )
    assert len(loaded.gate_proj) == 2


def test_raw_fp8_loader_rejects_fused_checkpoint_metadata(tmp_path):
    checkpoint, metadata = _write_checkpoint(tmp_path)
    metadata["weight_map"]["model.layers.0.mlp.experts.gate_up_proj"] = checkpoint.name
    with pytest.raises(ValueError, match="non-fused"):
        weights.load_block_fp8_experts_from_checkpoint_files(
            [str(checkpoint)],
            metadata,
            "model.layers",
            _moe_config(),
            layer_idx=0,
            hidden_size=128,
        )


def test_raw_fp8_loader_requires_every_scale(tmp_path):
    missing = "model.layers.0.mlp.experts.1.down_proj.weight_scale_inv"
    checkpoint, metadata = _write_checkpoint(tmp_path, omit_key=missing)
    with pytest.raises(FileNotFoundError, match="missing 1 native FP8 tensor"):
        weights.load_block_fp8_experts_from_checkpoint_files(
            [str(checkpoint)],
            metadata,
            "model.layers",
            _moe_config(),
            layer_idx=0,
            hidden_size=128,
        )


def test_raw_fp8_loader_rejects_wrong_block_size(tmp_path):
    checkpoint, metadata = _write_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="128, 128"):
        weights.load_block_fp8_experts_from_checkpoint_files(
            [str(checkpoint)],
            metadata,
            "model.layers",
            _moe_config(),
            layer_idx=0,
            hidden_size=128,
            block_size=(64, 128),
        )
