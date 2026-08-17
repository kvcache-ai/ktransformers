# SPDX-License-Identifier: Apache-2.0

import copy
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


def _vlm(conv3d):
    model = torch.nn.Sequential(conv3d)
    model.config = SimpleNamespace(model_type="qwen3_vl_moe", vision_config=object())
    return model


def test_patch_vlm_conv3d_matches_native_forward_and_backward(monkeypatch):
    monkeypatch.setattr(conv3d_compat, "_requires_conv3d_patch", lambda: True)
    native_forward = torch.nn.Conv3d.forward
    reference = _vlm(
        torch.nn.Conv3d(
            3, 4, kernel_size=(2, 4, 4), stride=(2, 4, 4), bias=True
        ).double()
    )
    patched = copy.deepcopy(reference)
    reference_input = torch.randn(
        2, 3, 4, 8, 8, dtype=torch.float64, requires_grad=True
    )
    patched_input = reference_input.detach().clone().requires_grad_(True)

    assert conv3d_compat.patch_vlm_conv3d(patched) == ["0"]
    assert conv3d_compat.patch_vlm_conv3d(patched) == ["0"]
    assert torch.nn.Conv3d.forward is native_forward
    assert conv3d_compat.is_vlm_conv3d_compatible(patched)

    expected = reference(reference_input)
    actual = patched(patched_input)
    expected.square().sum().backward()
    actual.square().sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(patched_input.grad, reference_input.grad)
    torch.testing.assert_close(patched[0].weight.grad, reference[0].weight.grad)
    torch.testing.assert_close(patched[0].bias.grad, reference[0].bias.grad)


def test_patch_vlm_conv3d_rejects_unsupported_contract_atomically(monkeypatch):
    monkeypatch.setattr(conv3d_compat, "_requires_conv3d_patch", lambda: True)
    model = _vlm(torch.nn.Conv3d(3, 4, kernel_size=(2, 4, 4), stride=(1, 4, 4)))

    with pytest.raises(RuntimeError, match="stride="):
        conv3d_compat.patch_vlm_conv3d(model)

    assert not hasattr(model[0], "_kt_conv3d_compatible")
