# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from kt_kernel.sft.amx import AMXSFTMoEWrapper
from kt_kernel.sft.autograd import KTMoEFunction
from kt_kernel.sft.base import BaseSFTMoEWrapper, _supports_authoritative_optimizer_grads
from kt_kernel.sft.dist_utils import _distributed_rank_world_size
from kt_kernel.sft.lora import kt_adapt_peft_lora, sync_kt_lora_gradients, update_kt_lora_pointers


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


class _EventTaskRunner(_TaskRunner):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def submit(self, task):
        self.events.append("pool_submit")
        super().submit(task)


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


class _FakeLegacyWrapper(_FakeAuthoritativeWrapper):
    """Legacy lifecycle with observable task-construction arguments."""

    def __init__(self):
        super().__init__()
        self._uses_authoritative_optimizer_grads = False
        self.task_kwargs = []

    def _make_backward_task(self, _buffer, **kwargs):
        self.task_kwargs.append(dict(kwargs))
        accumulate_optimizer_grads = bool(kwargs.get("accumulate_optimizer_grads", False))
        optimizer_grad_scale = float(kwargs.get("optimizer_grad_scale", 1.0))
        self.task_modes.append((accumulate_optimizer_grads, optimizer_grad_scale))

        def task():
            value = self.write_value * optimizer_grad_scale
            for grad_view in self.grad_views:
                if accumulate_optimizer_grads:
                    grad_view.add_(value)
                else:
                    grad_view.fill_(value)

        return task


def test_legacy_backward_forwards_nonunit_optimizer_scale_sync_and_async():
    backend = _FakeLegacyWrapper()

    backend.backward(torch.ones(1, 1), optimizer_grad_scale=0.5)
    assert backend.task_kwargs == [{"accumulate_optimizer_grads": False, "optimizer_grad_scale": 0.5}]
    assert backend.task_modes == [(False, 0.5)]

    backend.reset_cache()
    backend.submit_backward_async(torch.ones(1, 1), optimizer_grad_scale=0.25)
    assert backend.task_kwargs[-1] == {
        "accumulate_optimizer_grads": False,
        "optimizer_grad_scale": 0.25,
    }
    backend.sync_backward()

    backend.reset_cache()
    backend.backward(torch.ones(1, 1))
    assert backend.task_kwargs[-1] == {}
    assert backend.task_modes[-1] == (False, 1.0)


def test_legacy_backward_rejects_invalid_optimizer_scale_before_staging_copy():
    backend = _FakeLegacyWrapper()

    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            backend.backward(torch.ones(1, 1), optimizer_grad_scale=scale)
    assert backend.staging_copy_count == 0


class _RecordingMoe:
    def __init__(self):
        self.calls = []

    def backward_task(self, *args):
        self.calls.append(args)
        return args


def _fake_amx_backend(method: str, *, skip_lora: bool = False):
    backend = object.__new__(AMXSFTMoEWrapper)
    backend.method = method
    backend._is_skip_lora = skip_lora
    backend._uses_authoritative_optimizer_grads = False
    backend._full_weight_grad = True
    backend.lora_rank = 0
    backend.grad_gate_proj_buf = torch.empty(1)
    backend.grad_up_proj_buf = torch.empty(1)
    backend.grad_down_proj_buf = torch.empty(1)
    backend.moe = _RecordingMoe()
    return backend


def _fake_amx_backward_buffer():
    return SimpleNamespace(
        grad_output_cpu=torch.empty(1),
        grad_input_cpu=torch.empty(1),
        grad_weights=torch.empty(1),
    )


@pytest.mark.parametrize("method", ["AMXBF16_SFT", "AMXINT8_SFT", "AMXINT4_SFT"])
def test_legacy_amx_task_uses_scaled_tail_only_when_required(method):
    backend = _fake_amx_backend(method)
    buffer = _fake_amx_backward_buffer()

    backend._make_backward_task(buffer)
    assert len(backend.moe.calls[-1]) == 12

    backend._make_backward_task(buffer, accumulate_optimizer_grads=False, optimizer_grad_scale=0.5)
    scaled_call = backend.moe.calls[-1]
    assert len(scaled_call) == 14
    assert scaled_call[-2:] == (False, 0.5)


def test_skip_lora_task_keeps_legacy_signature_when_scale_is_supplied():
    backend = _fake_amx_backend("AMXBF16_SFT_SkipLoRA", skip_lora=True)

    backend._make_backward_task(
        _fake_amx_backward_buffer(),
        accumulate_optimizer_grads=False,
        optimizer_grad_scale=0.5,
    )

    call = backend.moe.calls[-1]
    assert len(call) == 12
    assert call[2:8] == (0, 0, 0, 0, 0, 0)
    assert call[9:12] == (0, 0, 0)


def test_forward_waits_for_pending_backward_repack_before_pool_submit():
    backend = _FakeAuthoritativeWrapper()
    events = []
    backend.cpu_infer = _EventTaskRunner(events)
    backend._weights_loaded = True
    backend.moe = SimpleNamespace(
        submit_backward_repack=lambda: events.append("repack_submit"),
        wait_backward_repack=lambda: events.append("repack_wait"),
    )
    backend._validate_forward_inputs = lambda *_args: None
    backend._get_buffer = lambda _qlen: SimpleNamespace()
    backend._copy_inputs_to_buffer = lambda *_args: None
    backend._make_forward_task = lambda *_args: (lambda: None)

    backend.submit_backward_repack()
    backend.submit_forward(
        torch.ones(1, 1),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.ones(1, 1),
        save_for_backward=True,
    )

    assert events == ["repack_submit", "repack_wait", "pool_submit"]
    assert not backend._backward_repack_pending


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


class _DistributedLegacyBackend:
    """Rank-0 fake that models an overwrite-only C++ dWeight producer."""

    def __init__(self, total_qlen):
        self.total_qlen = total_qlen
        self._full_weight_grad = True
        self._uses_authoritative_optimizer_grads = False
        self.share_backward_bb = False
        self.grad_gate_proj_buf = torch.empty(1)
        self.grad_up_proj_buf = None
        self.grad_down_proj_buf = None
        self.optimizer_grad_scales = []

    def sync_forward(self, output_device=None):
        output = torch.zeros(self.total_qlen, 1)
        return output if output_device is None else output.to(output_device)

    def backward(self, grad_output, output_device=None, optimizer_grad_scale=1.0):
        scale = float(optimizer_grad_scale)
        self.optimizer_grad_scales.append(scale)
        self.grad_gate_proj_buf.fill_(float(grad_output.float().sum()) * scale)
        grad_input = grad_output.clone()
        grad_weights = torch.zeros(grad_output.shape[0], 1)
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights


def _distributed_legacy_grad_worker(rank, init_file, gas_steps, result_queue):
    import torch.distributed as dist

    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=10),
        )
        all_qlens = [1, 2]
        local_qlen = all_qlens[rank]
        parameter = torch.nn.Parameter(torch.tensor([10.0])) if rank == 0 else None
        backend = _DistributedLegacyBackend(sum(all_qlens)) if rank == 0 else None

        for microbatch_idx in range(gas_steps):
            hidden_states = torch.ones(1, local_qlen, 1, requires_grad=True)
            expert_ids = torch.zeros(1, local_qlen, 1, dtype=torch.int64)
            route_weights = torch.ones(1, local_qlen, 1, requires_grad=True)
            lora_ref = parameter if parameter is not None else torch.empty((), requires_grad=True)
            output = KTMoEFunction.apply(
                hidden_states,
                expert_ids,
                route_weights,
                backend,
                lora_ref,
                1,
                1,
                0,
                True,
                False,
                all_qlens,
                False,
                False,
                parameter,
                None,
                None,
            )
            output.mul(float(microbatch_idx + 1)).sum().backward()

        if rank == 0:
            grad_before_step = float(parameter.grad.item())
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            optimizer.step()
            payload = (
                grad_before_step,
                float(parameter.item()),
                tuple(backend.optimizer_grad_scales),
            )
        else:
            payload = None
        dist.barrier()
        result_queue.put((rank, "ok", payload))
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _distributed_sync_helper_worker(rank, init_file, result_queue):
    import torch.distributed as dist

    original_all_reduce = None
    original_cuda = None
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=10),
        )
        ordinary_module = torch.nn.Linear(1, 1, bias=False)
        ordinary_parameter = next(ordinary_module.parameters())
        ordinary_parameter.grad = torch.full_like(ordinary_parameter, float(rank + 1))
        backend = (
            SimpleNamespace(
                grad_gate_proj_buf=torch.ones(1),
                grad_up_proj_buf=torch.ones(1),
                grad_down_proj_buf=torch.ones(1),
            )
            if rank == 0
            else None
        )
        wrapper = SimpleNamespace(
            layer_idx=0,
            wrapper=backend,
            _uses_authoritative_optimizer_grads=False,
            _full_weight_grad=True,
            _kt_world_size_at_wrap=2,
            lora_experts=ordinary_module,
        )
        model = SimpleNamespace(_kt_wrappers=[wrapper])

        original_all_reduce = dist.all_reduce
        original_cuda = torch.Tensor.cuda

        def forbidden_all_reduce(*_args, **_kwargs):
            raise AssertionError("sync_kt_lora_gradients must not issue all_reduce")

        dist.all_reduce = forbidden_all_reduce
        torch.Tensor.cuda = lambda self, *_args, **_kwargs: self
        sync_kt_lora_gradients(model)
        dist.all_reduce = original_all_reduce
        torch.Tensor.cuda = original_cuda

        dist.barrier()
        result_queue.put((rank, "ok", float(ordinary_parameter.grad.item())))
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        if original_all_reduce is not None:
            dist.all_reduce = original_all_reduce
        if original_cuda is not None:
            torch.Tensor.cuda = original_cuda
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_forked_workers(target, init_file, *worker_args):
    context = mp.get_context("fork")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=target,
            args=(rank, str(init_file), *worker_args, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()

    try:
        results = [result_queue.get(timeout=20) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    for process in processes:
        assert process.exitcode == 0
    assert all(status == "ok" for _, status, _ in results), results
    return sorted(results)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("gas_steps", [1, 2])
def test_distributed_legacy_full_grad_is_averaged_before_optimizer_step(tmp_path, gas_steps):
    results = _run_forked_workers(
        _distributed_legacy_grad_worker,
        tmp_path / f"legacy-grad-gas-{gas_steps}",
        gas_steps,
    )
    _, _, rank0_payload = results[0]
    grad_before_step, parameter_after_step, scales = rank0_payload
    expected_grad = (sum((1, 2)) / 2) * sum(range(1, gas_steps + 1))

    assert grad_before_step == pytest.approx(expected_grad)
    assert parameter_after_step == pytest.approx(10.0 - 0.1 * expected_grad)
    assert scales == (0.5,) * gas_steps
    assert results[1][2] is None


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_sync_helper_is_collective_free_and_leaves_ordinary_module_grads_unchanged(tmp_path):
    results = _run_forked_workers(
        _distributed_sync_helper_worker,
        tmp_path / "sync-helper-no-collective",
    )

    assert results[0][2] == 1.0
    assert results[1][2] == 2.0
