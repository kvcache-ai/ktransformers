import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sglang" / "python"))

from sglang.srt.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)


def _make_qwen35_text_stub():
    model = object.__new__(Qwen3_5ForCausalLM)
    model.config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=2,
        linear_key_head_dim=3,
        linear_num_value_heads=1,
        linear_value_head_dim=5,
        moe_intermediate_size=7,
        vocab_size=128,
        attn_output_gate=True,
    )
    return model


def test_qwen35_split_gdn_lora_hidden_dims():
    model = _make_qwen35_text_stub()

    assert model.get_hidden_dim("qkv_proj", 0) == (16, 48)
    assert model.get_hidden_dim("o_proj", 0) == (16, 16)
    assert model.get_hidden_dim("in_proj_qkv", 0) == (16, 17)
    assert model.get_hidden_dim("in_proj_z", 0) == (16, 5)
    assert model.get_hidden_dim("in_proj_b", 0) == (16, 1)
    assert model.get_hidden_dim("in_proj_a", 0) == (16, 1)
    assert model.get_hidden_dim("out_proj", 0) == (5, 16)
    assert model.get_hidden_dim("gate_up_proj", 0) == (16, 14)
    assert model.get_hidden_dim("down_proj", 0) == (7, 16)


def test_qwen35_wrapper_filters_lora_modules_and_delegates_dims():
    text_model = _make_qwen35_text_stub()
    wrapper = object.__new__(Qwen3_5MoeForConditionalGeneration)
    wrapper.__dict__["model"] = text_model

    assert wrapper.get_hidden_dim("in_proj_qkv", 0) == (16, 17)
    assert wrapper.should_apply_lora("model.layers.0.linear_attn.in_proj_qkv")
    assert wrapper.should_apply_lora("model.language_model.layers.0.self_attn.qkv_proj")
    assert wrapper.should_apply_lora("model.layers.0.linear_attn.out_proj")

    assert not wrapper.should_apply_lora("model.layers.0.mlp.experts.0.gate_proj")
    assert not wrapper.should_apply_lora("visual.blocks.0.attn.qkv_proj")
    assert not wrapper.should_apply_lora("language_model.model.layers.0.self_attn.qkv_proj")
