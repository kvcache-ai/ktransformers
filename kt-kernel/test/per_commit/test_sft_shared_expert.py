# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from kt_kernel.sft.arch import MOEArchConfig, move_non_experts_to_gpu
from kt_kernel.sft.layer import KTMoELayerWrapper


class _FakeWrapper:
    def __init__(self):
        self._full_weight_grad = False
        self._uses_authoritative_optimizer_grads = False
        self.share_backward_bb = False
        self._next_backward_wrapper = None
        self.output = None
        self.weights_shape = None

    def submit_forward(self, hidden_states, _expert_ids, weights, save_for_backward=True):
        self.output = torch.zeros_like(hidden_states)
        self.weights_shape = weights.shape
        self.save_for_backward = save_for_backward

    def sync_forward(self, output_device=None):
        output = self.output.clone()
        return output if output_device is None else output.to(output_device)

    def backward(self, grad_output, output_device=None):
        grad_input = torch.zeros_like(grad_output)
        grad_weights = torch.zeros(self.weights_shape, dtype=torch.bfloat16)
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights

    def clear_checkpoint_output(self):
        pass


class _SharedMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _SingularSharedMoE(nn.Module):
    def __init__(self, hidden_size: int, expert_num: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, expert_num, bias=False)
        self.experts = nn.Identity()
        self.shared_expert = _SharedMLP(hidden_size, intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)


class _PluralSharedMoE(nn.Module):
    def __init__(self, hidden_size: int, expert_num: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, expert_num, bias=False)
        self.experts = nn.Identity()
        self.shared_experts = nn.Linear(hidden_size, hidden_size, bias=False)


def _moe_config(*, has_shared_experts: bool) -> MOEArchConfig:
    return MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=3,
        num_experts_per_tok=1,
        has_shared_experts=has_shared_experts,
    )


def _wrap(original_moe: nn.Module, *, has_shared_experts: bool) -> KTMoELayerWrapper:
    return KTMoELayerWrapper(
        original_moe=original_moe,
        wrapper=_FakeWrapper(),
        lora_params=None,
        moe_config=_moe_config(has_shared_experts=has_shared_experts),
        hidden_size=4,
        layer_idx=0,
        full_weight_grad=False,
    )


def test_qwen_singular_shared_expert_matches_gated_forward_and_updates_weights():
    torch.manual_seed(0)
    original_moe = _SingularSharedMoE(hidden_size=4, expert_num=2, intermediate_size=3)
    hidden_states = torch.randn(2, 3, 4)
    with torch.no_grad():
        expected = original_moe.shared_expert(hidden_states)
        expected = torch.sigmoid(original_moe.shared_expert_gate(hidden_states)) * expected

    layer = _wrap(original_moe, has_shared_experts=True)
    named_parameters = dict(layer.named_parameters())
    shared_parameter_names = {
        "shared_expert.gate_proj.weight",
        "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight",
        "shared_expert_gate.weight",
    }
    assert shared_parameter_names.issubset(named_parameters)
    assert not any(name.startswith("shared_experts.") for name in named_parameters)

    train_input = hidden_states.clone().requires_grad_(True)
    actual = layer(train_input)
    torch.testing.assert_close(actual, expected)

    before_step = {name: named_parameters[name].detach().clone() for name in shared_parameter_names}
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    actual.square().sum().backward()

    for name in shared_parameter_names:
        grad = named_parameters[name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0

    optimizer.step()
    for name in shared_parameter_names:
        assert not torch.equal(named_parameters[name], before_step[name])


def test_plural_shared_expert_keeps_legacy_ungated_behavior():
    torch.manual_seed(1)
    original_moe = _PluralSharedMoE(hidden_size=4, expert_num=2)
    hidden_states = torch.randn(1, 2, 4)
    with torch.no_grad():
        expected = original_moe.shared_experts(hidden_states)

    layer = _wrap(original_moe, has_shared_experts=True)
    with torch.no_grad():
        actual = layer(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert "shared_experts.weight" in layer.state_dict()


def test_huggingface_qwen35_shared_expert_contract():
    configuration = pytest.importorskip(
        "transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        reason="Qwen3.5 support requires a recent transformers build",
    )
    modeling = pytest.importorskip(
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        reason="Qwen3.5 support requires a recent transformers build",
    )
    config = configuration.Qwen3_5MoeTextConfig(
        hidden_size=4,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=3,
        num_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
    )
    original_moe = modeling.Qwen3_5MoeSparseMoeBlock(config)
    hidden_states = torch.randn(1, 2, 4)
    with torch.no_grad():
        expected = original_moe.shared_expert(hidden_states)
        expected = torch.sigmoid(original_moe.shared_expert_gate(hidden_states)) * expected

    layer = _wrap(original_moe, has_shared_experts=True)
    with torch.no_grad():
        actual = layer(hidden_states)

    torch.testing.assert_close(actual, expected)
    assert "shared_expert.gate_proj.weight" in layer.state_dict()
    assert "shared_expert_gate.weight" in layer.state_dict()


def test_declared_shared_expert_missing_from_module_fails_fast():
    original_moe = nn.Module()
    original_moe.gate = nn.Linear(4, 2, bias=False)
    original_moe.experts = nn.Identity()

    with pytest.raises(ValueError, match="neither 'shared_experts' nor 'shared_expert'"):
        _wrap(original_moe, has_shared_experts=True)


def test_qwen_singular_shared_expert_without_gate_fails_fast():
    original_moe = nn.Module()
    original_moe.gate = nn.Linear(4, 2, bias=False)
    original_moe.experts = nn.Identity()
    original_moe.shared_expert = _SharedMLP(hidden_size=4, intermediate_size=3)

    with pytest.raises(ValueError, match="requires 'shared_expert_gate'"):
        _wrap(original_moe, has_shared_experts=True)


class _MoveRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


class _PlacementModel(nn.Module):
    def __init__(self):
        super().__init__()
        moe = nn.Module()
        moe.gate = _MoveRecorder()
        moe.experts = nn.Identity()
        moe.shared_expert = _MoveRecorder()
        moe.shared_expert_gate = _MoveRecorder()

        layer = nn.Module()
        layer.self_attn = _MoveRecorder()
        layer.input_layernorm = _MoveRecorder()
        layer.post_attention_layernorm = _MoveRecorder()
        layer.mlp = moe

        self.model = nn.Module()
        self.model.embed_tokens = _MoveRecorder()
        self.model.norm = _MoveRecorder()
        self.model.layers = nn.ModuleList([layer])
        self.lm_head = _MoveRecorder()


def test_qwen_shared_expert_and_gate_follow_non_experts_to_accelerator():
    model = _PlacementModel()
    move_non_experts_to_gpu(model, _moe_config(has_shared_experts=True), device="test:0")

    moe = model.model.layers[0].mlp
    assert moe.shared_expert.devices == ["test:0"]
    assert moe.shared_expert_gate.devices == ["test:0"]
