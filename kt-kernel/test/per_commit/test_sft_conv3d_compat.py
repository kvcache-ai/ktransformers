# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "python" / "sft" / "conv3d_compat.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kt_sft_conv3d_compat_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
conv3d_compat = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = conv3d_compat
SPEC.loader.exec_module(conv3d_compat)


def test_prepare_vlm_conv3d_only_patches_supported_vlm(monkeypatch):
    calls = []
    monkeypatch.setattr(conv3d_compat, "_requires_conv3d_patch", lambda: True)
    monkeypatch.setattr(
        conv3d_compat, "_enable_swift_patch", lambda: calls.append(True)
    )

    conv3d_compat.prepare_vlm_conv3d(
        SimpleNamespace(model_type="qwen3_vl_moe", vision_config=object())
    )
    conv3d_compat.prepare_vlm_conv3d(
        SimpleNamespace(model_type="qwen3_moe", vision_config=None)
    )

    assert calls == [True]


def test_validate_vlm_conv3d_accepts_qwen_patch_embed(monkeypatch):
    monkeypatch.setattr(conv3d_compat, "_requires_conv3d_patch", lambda: True)
    monkeypatch.setattr(conv3d_compat, "_enable_swift_patch", lambda: None)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 4, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    )

    assert conv3d_compat.validate_vlm_conv3d(model) == ["0"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"stride": (1, 16, 16)}, "stride="),
        ({"padding": (0, 1, 0)}, "padding="),
        ({"dilation": (1, 2, 1)}, "dilation="),
        ({"groups": 3, "out_channels": 6}, "groups="),
    ),
)
def test_validate_vlm_conv3d_rejects_unsupported_contract(monkeypatch, kwargs, reason):
    monkeypatch.setattr(conv3d_compat, "_requires_conv3d_patch", lambda: True)
    monkeypatch.setattr(conv3d_compat, "_enable_swift_patch", lambda: None)
    options = {
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": (2, 16, 16),
        "stride": (2, 16, 16),
    }
    options.update(kwargs)
    model = torch.nn.Sequential(torch.nn.Conv3d(**options))

    with pytest.raises(RuntimeError, match=reason):
        conv3d_compat.validate_vlm_conv3d(model)
