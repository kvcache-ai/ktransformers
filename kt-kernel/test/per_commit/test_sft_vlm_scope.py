# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from kt_kernel.sft.lora import kt_adapt_peft_lora
from kt_kernel.sft.wrapper import _build_kt_plugin_from_args, _get_frozen_expert_method


@pytest.mark.parametrize(
    ("method", "expected"),
    (
        ("AMXBF16_SFT", "AMXBF16_SFT_SkipLoRA"),
        ("AMXINT8_SFT", "AMXINT8_SFT_SkipLoRA"),
        ("AMXINT4_SFT", "AMXINT4_SFT_SkipLoRA"),
        ("AMXBF16_SFT_SkipLoRA", "AMXBF16_SFT_SkipLoRA"),
    ),
)
def test_frozen_expert_method_preserves_backend_precision(method, expected):
    assert _get_frozen_expert_method(method) == expected


def test_plugin_builder_freezes_kt_experts_for_vision_only_lora():
    model_args = SimpleNamespace(kt_use_lora_experts=True)
    finetuning_args = SimpleNamespace(
        finetuning_type="lora",
        lora_rank=4,
        lora_alpha=8,
        vlm_lora_scope="vision",
    )

    plugin = _build_kt_plugin_from_args(model_args, finetuning_args)
    config = plugin.kt_config

    assert config.kt_lora_rank == 4
    assert config.kt_lora_alpha == 8
    assert config.kt_use_lora_experts is False
    assert config.kt_freeze_experts is True


@pytest.mark.parametrize("scope", ("default", "text", "all"))
def test_plugin_builder_keeps_existing_modes_unchanged(scope):
    model_args = SimpleNamespace(kt_use_lora_experts=True)
    finetuning_args = SimpleNamespace(
        finetuning_type="lora",
        lora_rank=4,
        lora_alpha=8,
        vlm_lora_scope=scope,
    )

    config = _build_kt_plugin_from_args(model_args, finetuning_args).kt_config

    assert config.kt_lora_rank == 4
    assert config.kt_lora_alpha == 8
    assert config.kt_use_lora_experts is True
    assert config.kt_freeze_experts is False


def test_frozen_experts_do_not_require_language_peft_adapters():
    wrapper = SimpleNamespace(
        moe_config=SimpleNamespace(experts_attr="experts"),
        layer_idx=0,
        experts=torch.nn.ModuleList([torch.nn.Linear(2, 2)]),
        _kt_freeze_experts=True,
        _kt_managed_lora_enabled=True,
        _peft_lora_modules={0: {"stale": object()}},
        _fused_expert_lora_params=[torch.nn.Parameter(torch.ones(1))],
    )
    model = SimpleNamespace(_kt_wrappers=[wrapper])

    kt_adapt_peft_lora(model)

    assert wrapper._kt_managed_lora_enabled is False
    assert wrapper._peft_lora_modules is None
    assert wrapper._fused_expert_lora_params == []
