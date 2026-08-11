import importlib.util
import sys
from pathlib import Path

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


def test_validate_swift_conv3d_modules_accepts_qwen_patch_embed():
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 4, kernel_size=(2, 16, 16), stride=(2, 16, 16))
    )

    assert conv3d_compat.validate_swift_conv3d_modules(model) == ["0"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"stride": (1, 16, 16)}, "stride="),
        ({"padding": (0, 1, 0)}, "padding="),
        ({"dilation": (1, 2, 1)}, "dilation="),
        ({"groups": 3, "out_channels": 6}, "groups="),
    ),
)
def test_validate_swift_conv3d_modules_rejects_unsupported_contract(kwargs, reason):
    options = {
        "in_channels": 3,
        "out_channels": 4,
        "kernel_size": (2, 16, 16),
        "stride": (2, 16, 16),
    }
    options.update(kwargs)
    model = torch.nn.Sequential(torch.nn.Conv3d(**options))

    with pytest.raises(RuntimeError, match=reason):
        conv3d_compat.validate_swift_conv3d_modules(model)
