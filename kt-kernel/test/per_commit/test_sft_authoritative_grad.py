# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import torch

from kt_kernel.sft.base import BaseSFTMoEWrapper, _supports_authoritative_optimizer_grads
from kt_kernel.sft.autograd import KTMoEFunction
from kt_kernel.sft.dist_utils import _distributed_rank_world_size
from kt_kernel.sft.lora import kt_adapt_peft_lora, update_kt_lora_pointers


class _TaskRunner:
    def __init__(self):
        self.pending = None
        self.fail_next_submit = False
        self.fail_next_sync = False

    def submit(self, task):
        if self.fail_next_submit:
            self.fail_next_submit = False
            raise RuntimeError("synthetic submit failure")
        if self.pending is not None:
            raise RuntimeError("task already pending")
        self.pending = task

    def sync(self):
        task = self.pending
        self.pending = None
        if self.fail_next_sync:
            self.fail_next_sync = False
            raise RuntimeError("synthetic C++ failure")
        if task is not None:
            task()


def test_capability_is_limited_to_cpu_only_amxbf16_sft():
    assert _supports_authoritative_optimizer_grads("AMXBF16_SFT", 0)
    assert not _supports_authoritative_optimizer_grads("AMXBF16_SFT", 1)
    assert not _supports_authoritative_optimizer_grads("AMXINT8_SFT", 0)
    assert not _supports_authoritative_optimizer_grads("AMXINT4_SFT", 0)
    assert not _supports_authoritative_optimizer_grads("AMXBF16_SFT_SkipLoRA", 0)


class _FakeAuthoritativeWrapper(BaseSFTMoEWrapper):
    """Minimal backend exercising BaseSFTMoEWrapper's real lifecycle."""

    def __init__(self, parameter_count=1):
        # Avoid constructing CPUInfer or importing a real AMX extension.
        self._uses_authoritative_optimizer_grads = True
        self._init_authoritative_optimizer_grads()
        self._cache_depth = 1
        self._base_weights_dirty = False
        self.cpu_infer = _TaskRunner()
        self.buffer = SimpleNamespace()
        self.write_value = 2.0
        self.task_modes = []
        self.fail_return_grads = False
        self.staging_copy_count = 0
        self.parameters = []
        self.grad_views = []
        self._full_weight_grad = True
        self.share_backward_bb = False
        for idx in range(parameter_count):
            parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
            grad_view = torch.full_like(parameter, -99.0)
            self.parameters.append(parameter)
            self.grad_views.append(grad_view)
            self.register_authoritative_optimizer_grad(f"fake.{idx}", parameter, grad_view)

    def _get_buffer(self, _qlen):
        return self.buffer

    def _copy_grad_output_to_cpu(self, _buffer, _grad_output, _qlen):
        self.staging_copy_count += 1
        return None

    def _return_grads(self, _buffer, qlen, _output_device):
        if self.fail_return_grads:
            raise RuntimeError("synthetic return failure")
        return torch.zeros(qlen, 1), torch.zeros(qlen, 1)

    def sync_forward(self, output_device=None):
        output = torch.zeros(1, 1)
        return output if output_device is None else output.to(output_device)

    def clear_checkpoint_output(self):
        return None

    def _make_forward_task(self, _buffer, _save_for_backward):
        raise NotImplementedError

    def _make_backward_task(
        self,
        _buffer,
        accumulate_optimizer_grads=False,
        optimizer_grad_scale=1.0,
    ):
        self.task_modes.append((bool(accumulate_optimizer_grads), float(optimizer_grad_scale)))

        def task():
            value = self.write_value * float(optimizer_grad_scale)
            for grad_view in self.grad_views:
                if accumulate_optimizer_grads:
                    grad_view.add_(value)
                else:
                    grad_view.fill_(value)

        return task

    def reset_cache(self):
        self._cache_depth = 1

    # Abstract backend hooks not needed by these tests.
    def load_weights(self, physical_to_logical_map_cpu):
        raise NotImplementedError

    def init_lora_weights(self, *args, **kwargs):
        raise NotImplementedError

    def update_lora_weights(self):
        raise NotImplementedError

    def update_base_weights(self):
        raise NotImplementedError


def test_sync_backward_overwrite_accumulate_publish_and_step_release():
    backend = _FakeAuthoritativeWrapper()
    parameter = backend.parameters[0]
    grad_view = backend.grad_views[0]

    assert parameter.grad is None
    backend.backward(torch.ones(1, 1), optimizer_grad_scale=0.5)
    assert backend.task_modes == [(False, 0.5)]
    assert parameter.grad is grad_view
    torch.testing.assert_close(grad_view, torch.ones_like(grad_view))

    backend.reset_cache()
    backend.write_value = 4.0
    backend.backward(torch.ones(1, 1), optimizer_grad_scale=0.5)
    assert backend.task_modes[-1] == (True, 0.5)
    assert parameter.grad is grad_view
    torch.testing.assert_close(grad_view, torch.full_like(grad_view, 3.0))

    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter_before_step = parameter.detach().clone()
    optimizer.step()
    torch.testing.assert_close(parameter, parameter_before_step - 0.1 * grad_view)

    layer = SimpleNamespace(
        layer_idx=0,
        wrapper=backend,
        _kt_managed_lora_enabled=True,
        _lora_pointers_dirty=False,
        _full_weight_grad=True,
    )
    update_kt_lora_pointers(SimpleNamespace(_kt_wrappers=[layer]))
    assert layer._lora_pointers_dirty
    assert backend._base_weights_dirty
    assert parameter.grad is None

    raw_before_zero_grad = grad_view.clone()
    optimizer.zero_grad(set_to_none=False)
    assert parameter.grad is None
    torch.testing.assert_close(grad_view, raw_before_zero_grad)

    backend.reset_cache()
    backend.backward(torch.ones(1, 1))
    assert backend.task_modes[-1] == (False, 1.0)


def test_mixed_foreign_and_changed_metadata_fail_fast():
    backend = _FakeAuthoritativeWrapper(parameter_count=2)
    backend.parameters[0].grad = backend.grad_views[0]
    with pytest.raises(RuntimeError, match="Mixed authoritative"):
        backend._prepare_authoritative_optimizer_grad_write(1.0)

    backend.parameters[1].grad = backend.grad_views[1]
    backend.parameters[0].grad = backend.grad_views[0].view_as(backend.grad_views[0])
    with pytest.raises(RuntimeError, match="externally replaced"):
        backend._prepare_authoritative_optimizer_grad_write(1.0)

    metadata_backend = _FakeAuthoritativeWrapper()
    metadata_backend.grad_views[0].data = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="metadata changed"):
        metadata_backend._prepare_authoritative_optimizer_grad_write(1.0)


def test_failed_sync_closes_window_and_retry_overwrites():
    backend = _FakeAuthoritativeWrapper()
    backend.cpu_infer.fail_next_sync = True

    with pytest.raises(RuntimeError, match=r"synthetic C\+\+ failure"):
        backend.backward(torch.ones(1, 1))
    assert backend.parameters[0].grad is None
    assert backend.validate_authoritative_optimizer_grad_state() == "closed"
    assert backend._cache_depth == 0

    backend.reset_cache()
    backend.backward(torch.ones(1, 1))
    assert backend.task_modes[-1] == (False, 1.0)
    assert backend.parameters[0].grad is backend.grad_views[0]


def test_post_cpp_return_failure_closes_sync_and_async_windows():
    backend = _FakeAuthoritativeWrapper()
    backend.fail_return_grads = True

    with pytest.raises(RuntimeError, match="synthetic return failure"):
        backend.backward(torch.ones(1, 1))
    assert backend.parameters[0].grad is None
    assert backend._cache_depth == 0

    backend.reset_cache()
    backend.submit_backward_async(torch.ones(1, 1))
    with pytest.raises(RuntimeError, match="synthetic return failure"):
        backend.sync_backward()
    assert backend.parameters[0].grad is None
    assert backend._cache_depth == 0
    assert backend._async_bwd_qlen is None

    backend.fail_return_grads = False
    backend.reset_cache()
    backend.backward(torch.ones(1, 1))
    assert backend.task_modes[-1] == (False, 1.0)


def test_async_submit_failure_invalidates_cache_and_pending_state():
    backend = _FakeAuthoritativeWrapper()
    backend.cpu_infer.fail_next_submit = True

    with pytest.raises(RuntimeError, match="synthetic submit failure"):
        backend.submit_backward_async(torch.ones(1, 1))
    assert backend.parameters[0].grad is None
    assert backend._cache_depth == 0
    assert backend._async_bwd_qlen is None


def test_pending_async_backward_rejects_reentrant_submit_before_staging_copy():
    backend = _FakeAuthoritativeWrapper()
    backend.submit_backward_async(torch.ones(1, 1))

    with pytest.raises(RuntimeError, match="already pending"):
        backend.submit_backward_async(torch.full((1, 1), 7.0))
    with pytest.raises(RuntimeError, match="already pending"):
        backend.backward(torch.full((1, 1), 9.0))
    assert backend.staging_copy_count == 1

    backend.sync_backward()
    assert backend.parameters[0].grad is backend.grad_views[0]
    torch.testing.assert_close(backend.grad_views[0], torch.full_like(backend.grad_views[0], 2.0))


def test_hybrid_requires_expert_lora_instead_of_silently_falling_back_to_full():
    class _Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 4, bias=False)
            self.up_proj = torch.nn.Linear(4, 4, bias=False)
            self.down_proj = torch.nn.Linear(4, 4, bias=False)

    layer = SimpleNamespace(
        layer_idx=0,
        moe_config=SimpleNamespace(weight_names=("gate_proj", "up_proj", "down_proj")),
        experts=torch.nn.ModuleList([_Expert()]),
        _experts_attr="experts",
        _fused_experts=False,
        _lora_rank=4,
        _full_weight_grad=True,
    )
    with pytest.raises(RuntimeError, match="No PEFT LoRA found"):
        kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[layer]))


def test_launcher_environment_preserves_rank0_ownership_before_process_group_init():
    previous_rank = os.environ.get("RANK")
    previous_world = os.environ.get("WORLD_SIZE")
    try:
        os.environ["RANK"] = "1"
        os.environ["WORLD_SIZE"] = "2"
        assert _distributed_rank_world_size() == (1, 2)
    finally:
        if previous_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = previous_rank
        if previous_world is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = previous_world


def test_async_backward_publishes_only_after_successful_sync():
    backend = _FakeAuthoritativeWrapper()
    parameter = backend.parameters[0]

    backend.submit_backward_async(torch.ones(1, 1), optimizer_grad_scale=0.25)
    assert backend.task_modes == [(False, 0.25)]
    assert parameter.grad is None
    backend.sync_backward()
    assert parameter.grad is backend.grad_views[0]

    backend.reset_cache()
    backend.write_value = 8.0
    backend.submit_backward_async(torch.ones(1, 1), optimizer_grad_scale=0.25)
    assert backend.task_modes[-1] == (True, 0.25)
    assert parameter.grad is backend.grad_views[0]
    backend.sync_backward()
    torch.testing.assert_close(backend.grad_views[0], torch.full_like(backend.grad_views[0], 2.5))


def test_failed_async_sync_closes_window_and_clears_pending_state():
    backend = _FakeAuthoritativeWrapper()
    backend.cpu_infer.fail_next_sync = True

    backend.submit_backward_async(torch.ones(1, 1))
    with pytest.raises(RuntimeError, match=r"synthetic C\+\+ failure"):
        backend.sync_backward()

    assert backend.parameters[0].grad is None
    assert backend.validate_authoritative_optimizer_grad_state() == "closed"
    assert backend._cache_depth == 0
    with pytest.raises(RuntimeError, match="No pending backward"):
        backend.sync_backward()


def test_autograd_returns_no_base_gradient_and_preserves_published_alias():
    backend = _FakeAuthoritativeWrapper()
    parameter = backend.parameters[0]
    grad_view = backend.grad_views[0]
    hidden_states = torch.ones(1, 1, 1, requires_grad=True)
    expert_ids = torch.zeros(1, 1, dtype=torch.int64)
    route_weights = torch.ones(1, 1, requires_grad=True)

    output = KTMoEFunction.apply(
        hidden_states,
        expert_ids,
        route_weights,
        backend,
        parameter,
        1,
        1,
        0,
        True,
        False,
        None,
        False,
        False,
        parameter,
        None,
        None,
    )
    output.sum().backward()

    # KTMoEFunction returned None for both references to the base Parameter;
    # the alias published by the backend must therefore remain the sole grad.
    assert parameter.grad is grad_view
    torch.testing.assert_close(grad_view, torch.full_like(grad_view, 2.0))
