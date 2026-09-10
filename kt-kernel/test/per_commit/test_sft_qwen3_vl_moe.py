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


@pytest.mark.parametrize(
    ("architecture", "outer_model_type"),
    [
        ("KimiK25ForConditionalGeneration", "kimi_k25"),
        ("KimiK25ForConditionalGeneration", "kimi_k2_5"),
        ("KimiK26ForConditionalGeneration", "kimi_k26"),
        ("KimiK26ForConditionalGeneration", "kimi_k2_6"),
    ],
)
def test_kimi_k2_family_uses_nested_text_config_and_checkpoint_prefix(architecture, outer_model_type):
    text_config = SimpleNamespace(
        model_type="kimi_k2",
        n_routed_experts=384,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        n_shared_experts=1,
        num_hidden_layers=61,
    )
    config = SimpleNamespace(
        architectures=[architecture],
        model_type=outer_model_type,
        text_config=text_config,
    )

    moe_config = arch.get_moe_arch_config(config)

    assert moe_config.expert_num == 384
    assert moe_config.intermediate_size == 2048
    assert moe_config.num_experts_per_tok == 8
    assert moe_config.has_shared_experts is True
    assert moe_config.router_type == "deepseek_gate"
    assert arch._get_layers_prefix(config) == "language_model.model.layers"


def test_kimi_nested_text_model_type_rejects_unknown_outer_alias():
    config = SimpleNamespace(
        architectures=["FutureMoonshotConditionalGeneration"],
        model_type="future_moonshot",
        text_config=SimpleNamespace(
            model_type="kimi_k2",
            n_routed_experts=384,
            moe_intermediate_size=2048,
            num_experts_per_tok=8,
            n_shared_experts=1,
        ),
    )

    assert arch.get_moe_arch_config(config).expert_num == 384
    with pytest.raises(arch.KTAMXModelNotSupportedError, match="Unknown composite Kimi architecture"):
        arch._get_layers_prefix(config)


def test_kimi_text_only_model_uses_standard_decoder_prefix():
    config = SimpleNamespace(
        architectures=["KimiK2ForCausalLM"],
        model_type="kimi_k2",
        n_routed_experts=384,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        n_shared_experts=1,
    )

    assert arch.get_moe_arch_config(config).expert_num == 384
    assert arch._get_layers_prefix(config) == "model.layers"


def test_non_expert_placement_uses_outer_output_embedding_api():
    calls = []
    output_embeddings = SimpleNamespace(to=lambda device: calls.append(device))
    model = SimpleNamespace(
        layers=[],
        get_output_embeddings=lambda: output_embeddings,
    )
    moe_config = arch.MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=4,
        num_experts_per_tok=1,
    )

    arch.move_non_experts_to_gpu(model, moe_config, device="cuda:3")

    assert calls == ["cuda:3"]


def test_non_expert_placement_falls_back_to_legacy_lm_head():
    calls = []
    model = SimpleNamespace(
        layers=[],
        get_output_embeddings=lambda: None,
        lm_head=SimpleNamespace(to=lambda device: calls.append(device)),
    )
    moe_config = arch.MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=4,
        num_experts_per_tok=1,
    )

    arch.move_non_experts_to_gpu(model, moe_config, device="cuda:2")

    assert calls == ["cuda:2"]


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
