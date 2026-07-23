# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from kt_kernel.sft.layer import KTMoELayerWrapper


class _LoRARouter(torch.nn.Module):
    def __init__(self, base=None, detach_output: bool = False):
        super().__init__()
        self.base = base if base is not None else torch.nn.Linear(4, 3, bias=False)
        self.lora_a = torch.nn.Linear(4, 2, bias=False)
        self.lora_b = torch.nn.Linear(2, 3, bias=False)
        self.base.requires_grad_(False)
        torch.nn.init.zeros_(self.lora_b.weight)
        self.detach_output = detach_output

    def forward(self, hidden_states):
        output = self.base(hidden_states) + self.lora_b(self.lora_a(hidden_states))
        return output.detach() if self.detach_output else output


class _TopKRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 4))
        self.top_k = 2
        self.norm_topk_prob = True


class _OriginalMoE(torch.nn.Module):
    def __init__(self, router):
        super().__init__()
        self.gate = router
        self.experts = torch.nn.ModuleList()


class _FakeWrapper:
    def __init__(self, reuse_checkpoint_forward=False):
        self._full_weight_grad = False
        self._uses_authoritative_optimizer_grads = False
        self.reuse_checkpoint_forward = reuse_checkpoint_forward
        self.share_backward_bb = False
        self._kt_has_cached_forward = False
        self._checkpoint_output_cpu = None
        self.submit_calls = 0
        self.sync_calls = 0
        self.cached_output_calls = 0
        self.backward_calls = 0

    def submit_forward(self, hidden_states, expert_ids, weights, save_for_backward=True):
        self.submit_calls += 1
        self.input = hidden_states.detach().clone()
        self.expert_ids = expert_ids.detach().clone()
        self.weights = weights.detach().clone()
        self.route_factors = self.expert_ids.to(self.weights.dtype) + 1
        self.weighted_factor = (self.weights * self.route_factors).sum(dim=-1, keepdim=True)
        self.output = self.input * self.weighted_factor
        assert save_for_backward

    def sync_forward(self, output_device=None):
        self.sync_calls += 1
        output = self.output.clone()
        return output if output_device is None else output.to(output_device)

    def cache_checkpoint_output(self, output, _qlen):
        self._checkpoint_output_cpu = output
        self._kt_has_cached_forward = True

    def get_checkpoint_output(self, _qlen, output_device=None):
        self.cached_output_calls += 1
        output = self._checkpoint_output_cpu
        return output if output_device is None else output.to(output_device)

    def clear_checkpoint_output(self):
        self._checkpoint_output_cpu = None
        self._kt_has_cached_forward = False

    def backward(self, grad_output, output_device=None):
        self.backward_calls += 1
        grad_input = grad_output * self.weighted_factor
        grad_factor = (grad_output * self.input).sum(dim=-1, keepdim=True)
        grad_weights = grad_factor * self.route_factors
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights


def _make_layer(router, *, wrapper=None):
    config = SimpleNamespace(
        router_attr="gate",
        experts_attr="experts",
        has_shared_experts=False,
        router_type="linear",
        num_experts_per_tok=2,
    )
    layer = KTMoELayerWrapper(
        original_moe=_OriginalMoE(router),
        wrapper=wrapper,
        lora_params=None,
        moe_config=config,
        hidden_size=4,
        layer_idx=7,
        full_weight_grad=False,
    )
    layer.train()
    return layer


def _make_topk_layer(*, wrapper=None):
    layer = _make_layer(_TopKRouter(), wrapper=wrapper)
    layer.gate = _LoRARouter(base=layer.gate)
    return layer


def test_trainable_router_preserves_routing_graph_in_lora_mode():
    router = _LoRARouter()
    layer = _make_layer(router)
    hidden_states = torch.randn(1, 3, 4)

    _, topk_weights = layer._compute_routing(hidden_states)
    topk_weights.float().square().sum().backward()

    assert topk_weights.requires_grad
    assert router.base.weight.grad is None
    assert router.lora_b.weight.grad is not None
    assert torch.count_nonzero(router.lora_b.weight.grad) > 0


def test_transformers_v5_topk_router_lora_preserves_routing_graph():
    layer = _make_topk_layer()

    _, topk_weights = layer._compute_routing(torch.randn(1, 3, 4))
    topk_weights.float().square().sum().backward()

    assert layer._original_router is not None
    assert layer.gate.base.weight.grad is None
    assert layer.gate.lora_b.weight.grad is not None
    assert torch.count_nonzero(layer.gate.lora_b.weight.grad) > 0


def test_frozen_router_keeps_routing_outside_autograd():
    router = torch.nn.Linear(4, 3, bias=False)
    router.requires_grad_(False)
    layer = _make_layer(router)

    _, topk_weights = layer._compute_routing(torch.randn(1, 3, 4, requires_grad=True))

    assert not topk_weights.requires_grad


def test_trainable_router_detach_fails_fast():
    layer = _make_layer(_LoRARouter(detach_output=True))

    with pytest.raises(RuntimeError, match="trainable router produced detached routing weights"):
        layer._compute_routing(torch.randn(1, 3, 4))


def test_router_lora_gradient_survives_non_reentrant_checkpoint_reuse():
    backend = _FakeWrapper(reuse_checkpoint_forward=True)
    layer = _make_topk_layer(wrapper=backend)
    hidden_states = torch.randn(1, 3, 4, requires_grad=True)

    checkpoint(layer, hidden_states, use_reentrant=False).sum().backward()

    assert layer.gate.lora_b.weight.grad is not None
    assert torch.count_nonzero(layer.gate.lora_b.weight.grad) > 0
    assert backend.submit_calls == 1
    assert backend.sync_calls == 1
    assert backend.cached_output_calls == 1
    assert backend.backward_calls == 1
    assert not backend._kt_has_cached_forward


def test_router_lora_a_and_b_update_after_two_optimizer_steps():
    torch.manual_seed(0)
    layer = _make_topk_layer(wrapper=_FakeWrapper())
    optimizer = torch.optim.AdamW(
        [layer.gate.lora_a.weight, layer.gate.lora_b.weight],
        lr=1e-2,
        weight_decay=0.0,
    )
    hidden_states = torch.randn(1, 4, 4)
    lora_a_before = layer.gate.lora_a.weight.detach().clone()
    lora_b_before = layer.gate.lora_b.weight.detach().clone()

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        layer(hidden_states).float().square().sum().backward()
        assert layer.gate.lora_b.weight.grad is not None
        assert torch.count_nonzero(layer.gate.lora_b.weight.grad) > 0
        if step == 1:
            assert layer.gate.lora_a.weight.grad is not None
            assert torch.count_nonzero(layer.gate.lora_a.weight.grad) > 0
        optimizer.step()

    assert not torch.equal(layer.gate.lora_a.weight, lora_a_before)
    assert not torch.equal(layer.gate.lora_b.weight, lora_b_before)
