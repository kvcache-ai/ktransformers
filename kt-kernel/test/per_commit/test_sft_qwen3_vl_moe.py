# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kt_kernel.sft.layer import KTMoELayerWrapper
from kt_kernel.sft.weights import extract_moe_weights


MODULE_PATH = Path(__file__).resolve().parents[2] / "python" / "sft" / "arch.py"
SPEC = importlib.util.spec_from_file_location("kt_sft_arch_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
arch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arch
SPEC.loader.exec_module(arch)


def _qwen3_vl_moe_modules():
    configuration = pytest.importorskip(
        "transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe",
        reason="Qwen3-VL-MoE support requires a recent transformers build",
    )
    modeling = pytest.importorskip(
        "transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        reason="Qwen3-VL-MoE support requires a recent transformers build",
    )
    return configuration, modeling


def _text_config():
    configuration, _ = _qwen3_vl_moe_modules()
    return configuration.Qwen3VLMoeTextConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
    )


def test_qwen3_vl_moe_architecture_and_checkpoint_prefix():
    config = SimpleNamespace(
        architectures=["Qwen3VLMoeForConditionalGeneration"],
        text_config=_text_config(),
    )

    moe_config = arch.get_moe_arch_config(config)

    assert moe_config.expert_num == 2
    assert moe_config.intermediate_size == 4
    assert moe_config.num_experts_per_tok == 1
    assert moe_config.has_shared_experts is False
    assert arch._get_layers_prefix(config) == "model.language_model.layers"


def test_qwen3_vl_moe_fused_weights_and_router_match_transformers():
    _, modeling = _qwen3_vl_moe_modules()
    text_config = _text_config()
    original_moe = modeling.Qwen3VLMoeTextSparseMoeBlock(text_config)
    moe_config = arch.MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=4,
        num_experts_per_tok=1,
    )

    gate_proj, up_proj, down_proj = extract_moe_weights(original_moe, moe_config)
    torch.testing.assert_close(gate_proj, original_moe.experts.gate_up_proj[:, :4])
    torch.testing.assert_close(up_proj, original_moe.experts.gate_up_proj[:, 4:])
    torch.testing.assert_close(down_proj, original_moe.experts.down_proj)

    hidden_states = torch.randn(2, 3, text_config.hidden_size)
    with torch.no_grad():
        _, expected_weights, expected_ids = original_moe.gate(
            hidden_states.reshape(-1, text_config.hidden_size)
        )

    wrapped_moe = KTMoELayerWrapper(
        original_moe=original_moe,
        wrapper=None,
        lora_params=None,
        moe_config=moe_config,
        hidden_size=text_config.hidden_size,
        layer_idx=0,
        full_weight_grad=False,
    )
    actual_ids, actual_weights = wrapped_moe._compute_routing(hidden_states)

    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_weights, expected_weights.to(torch.bfloat16), rtol=2e-2, atol=2e-2
    )
