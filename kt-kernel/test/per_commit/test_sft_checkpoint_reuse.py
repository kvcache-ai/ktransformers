# SPDX-License-Identifier: Apache-2.0

import torch
from torch.utils.checkpoint import checkpoint

from kt_kernel.sft.autograd import KTMoEFunction
from kt_kernel.sft.dist_utils import _checkpoint_hook_mode
from kt_kernel.sft.wrapper import _supports_checkpoint_forward_reuse


def test_checkpoint_forward_reuse_supports_pure_full_and_lora_modes():
    assert _supports_checkpoint_forward_reuse(full_weight_grad=True, lora_rank=0)
    assert _supports_checkpoint_forward_reuse(full_weight_grad=False, lora_rank=8)
    assert not _supports_checkpoint_forward_reuse(full_weight_grad=True, lora_rank=8)
    assert not _supports_checkpoint_forward_reuse(full_weight_grad=False, lora_rank=0)


class _FakeWrapper:
    def __init__(self):
        self._full_weight_grad = False
        self.share_backward_bb = False
        self._kt_has_cached_forward = False
        self._checkpoint_output_cpu = None
        self.submit_calls = 0
        self.sync_calls = 0
        self.cached_output_calls = 0
        self.backward_calls = 0

    def submit_forward(self, hidden_states, _expert_ids, weights, save_for_backward=True):
        self.submit_calls += 1
        self.input = hidden_states.detach().clone()
        self.weights = weights.detach().clone()
        self.output = self.input * self.weights
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
        grad_input = grad_output * self.weights
        grad_weights = (grad_output * self.input).sum(dim=-1, keepdim=True)
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights


class _CheckpointedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapper = _FakeWrapper()
        self.route_weights = torch.nn.Parameter(torch.tensor([[0.25], [0.5], [0.75]], dtype=torch.bfloat16))

    def forward(self, hidden_states):
        batch, seq_len, hidden_size = hidden_states.shape
        qlen = batch * seq_len
        mode = _checkpoint_hook_mode()
        cache_forward = mode == "first_forward"
        reuse_forward = mode == "recompute" and self.wrapper._kt_has_cached_forward
        expert_ids = torch.zeros((batch, seq_len, 1), dtype=torch.int64)

        if not reuse_forward:
            self.wrapper.submit_forward(
                hidden_states.view(qlen, hidden_size),
                expert_ids.view(qlen, 1),
                self.route_weights,
                save_for_backward=True,
            )

        return KTMoEFunction.apply(
            hidden_states,
            expert_ids,
            self.route_weights,
            self.wrapper,
            hidden_states.new_empty(()),
            hidden_size,
            1,
            0,
            True,
            False,
            None,
            cache_forward,
            reuse_forward,
            None,
            None,
            None,
        )


def test_non_reentrant_checkpoint_reuses_cpu_expert_forward_and_preserves_gradients():
    module = _CheckpointedExpert()
    hidden_states = torch.arange(12, dtype=torch.float32).view(1, 3, 4).requires_grad_(True)

    output = checkpoint(module, hidden_states, use_reentrant=False)
    output.sum().backward()

    expected_input_grad = module.route_weights.detach().float().view(1, 3, 1).expand_as(hidden_states)
    expected_weight_grad = hidden_states.detach().sum(dim=-1).view(3, 1).to(torch.bfloat16)
    torch.testing.assert_close(hidden_states.grad, expected_input_grad)
    torch.testing.assert_close(module.route_weights.grad, expected_weight_grad)
    assert module.wrapper.submit_calls == 1
    assert module.wrapper.sync_calls == 1
    assert module.wrapper.cached_output_calls == 1
    assert module.wrapper.backward_calls == 1
    assert not module.wrapper._kt_has_cached_forward
