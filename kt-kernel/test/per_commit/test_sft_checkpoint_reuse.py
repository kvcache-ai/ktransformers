# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
import multiprocessing as mp
import os
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from kt_kernel.sft.autograd import KTMoEFunction
from kt_kernel.sft.base import BaseSFTMoEWrapper, _CheckpointCacheState
from kt_kernel.sft.config import KTActivationPolicy
from kt_kernel.sft.dist_utils import (
    _checkpoint_hook_mode,
    get_activation_checkpoint_context_fn,
)
from kt_kernel.sft.layer import (
    KTMoELayerWrapper,
    _activation_checkpoint_action,
    _validate_activation_checkpoint_phase,
)
from kt_kernel.sft.wrapper import _supports_checkpoint_forward_reuse


def test_checkpoint_forward_reuse_supports_pure_full_and_lora_modes():
    assert _supports_checkpoint_forward_reuse(full_weight_grad=True, lora_rank=0)
    assert _supports_checkpoint_forward_reuse(full_weight_grad=False, lora_rank=8)
    assert not _supports_checkpoint_forward_reuse(full_weight_grad=True, lora_rank=8)
    assert not _supports_checkpoint_forward_reuse(full_weight_grad=False, lora_rank=0)


@pytest.mark.parametrize(
    ("policy", "phase", "expected"),
    [
        (KTActivationPolicy(cpu="retain", gpu="retain"), "none", "normal"),
        (KTActivationPolicy(cpu="retain", gpu="recompute"), "first_forward", "cache_first_forward"),
        (KTActivationPolicy(cpu="retain", gpu="recompute"), "recompute", "reuse_recompute"),
        (KTActivationPolicy(cpu="recompute", gpu="recompute"), "first_forward", "normal"),
        (KTActivationPolicy(cpu="recompute", gpu="recompute"), "recompute", "normal"),
    ],
)
def test_checkpoint_action_matches_activation_policy(policy, phase, expected):
    assert _activation_checkpoint_action(policy, phase) == expected


def test_gpu_retain_rejects_an_active_checkpoint_phase():
    with pytest.raises(RuntimeError, match="conflicts with active gradient checkpointing"):
        _validate_activation_checkpoint_phase(
            KTActivationPolicy(cpu="retain", gpu="retain"),
            "first_forward",
        )


def test_gpu_recompute_requires_the_kt_checkpoint_context_for_training():
    with pytest.raises(RuntimeError, match="requires non-reentrant gradient checkpointing"):
        _validate_activation_checkpoint_phase(
            KTActivationPolicy(cpu="recompute", gpu="recompute"),
            "none",
            requires_backward=True,
        )


class _CacheStateHarness:
    checkpoint_cache_state = BaseSFTMoEWrapper.checkpoint_cache_state
    poison_checkpoint_output = BaseSFTMoEWrapper.poison_checkpoint_output
    validate_checkpoint_output = BaseSFTMoEWrapper.validate_checkpoint_output
    validate_checkpoint_cache_empty = BaseSFTMoEWrapper.validate_checkpoint_cache_empty
    cache_checkpoint_output = BaseSFTMoEWrapper.cache_checkpoint_output
    get_checkpoint_output = BaseSFTMoEWrapper.get_checkpoint_output
    clear_checkpoint_output = BaseSFTMoEWrapper.clear_checkpoint_output

    def __init__(self):
        self._checkpoint_cache_state = _CheckpointCacheState.EMPTY
        self._checkpoint_cache_error = None
        self._checkpoint_output_cpu = None
        self._checkpoint_output_qlen = 0


def test_checkpoint_cache_state_transitions_ready_to_empty():
    cache = _CacheStateHarness()
    expected = torch.arange(6, dtype=torch.bfloat16).view(3, 2)

    cache.cache_checkpoint_output(expected, 3)

    assert cache.checkpoint_cache_state is _CheckpointCacheState.READY
    torch.testing.assert_close(cache.get_checkpoint_output(3), expected)
    cache.clear_checkpoint_output()
    assert cache.checkpoint_cache_state is _CheckpointCacheState.EMPTY


def test_checkpoint_cache_qlen_mismatch_is_permanently_poisoned():
    cache = _CacheStateHarness()
    cache.cache_checkpoint_output(torch.zeros(3, 2), 3)

    with pytest.raises(RuntimeError, match="qlen mismatch"):
        cache.get_checkpoint_output(2)

    assert cache.checkpoint_cache_state is _CheckpointCacheState.POISONED
    cache.clear_checkpoint_output()
    assert cache.checkpoint_cache_state is _CheckpointCacheState.POISONED
    with pytest.raises(RuntimeError, match="poisoned"):
        cache.cache_checkpoint_output(torch.zeros(3, 2), 3)


def test_checkpoint_cache_preflight_rejects_a_live_cache_before_submit():
    cache = _CacheStateHarness()
    cache.cache_checkpoint_output(torch.zeros(3, 2), 3)

    with pytest.raises(RuntimeError, match="still live"):
        cache.validate_checkpoint_cache_empty()

    assert cache.checkpoint_cache_state is _CheckpointCacheState.POISONED


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
        self.save_for_backward_calls = []
        self.poison_calls = 0

    def submit_forward(self, hidden_states, _expert_ids, weights, save_for_backward=True):
        self.submit_calls += 1
        self.input = hidden_states.detach().clone()
        self.weights = weights.detach().clone()
        self.output = self.input * self.weights
        self.save_for_backward_calls.append(bool(save_for_backward))

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

    def poison_checkpoint_output(self, _error):
        self.poison_calls += 1
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
    assert module.wrapper.save_for_backward_calls == [True]
    assert not module.wrapper._kt_has_cached_forward


class _FailBeforeExpertRecompute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.expert = _CheckpointedExpert()

    def forward(self, hidden_states):
        if _checkpoint_hook_mode() == "recompute":
            raise RuntimeError("injected checkpoint recompute failure")
        return self.expert(hidden_states)


def test_checkpoint_recompute_failure_poisons_retained_cache():
    module = _FailBeforeExpertRecompute()
    hidden_states = torch.arange(12, dtype=torch.float32).view(1, 3, 4).requires_grad_(True)
    output = checkpoint(
        module,
        hidden_states,
        use_reentrant=False,
        context_fn=get_activation_checkpoint_context_fn(),
    )

    with pytest.raises(RuntimeError, match="injected checkpoint recompute failure"):
        output.sum().backward()

    assert module.expert.wrapper.poison_calls == 1
    assert not module.expert.wrapper._kt_has_cached_forward


class _DistributedFakeWrapper(_FakeWrapper):
    def __init__(self, failure_stage=None):
        super().__init__()
        self._uses_authoritative_optimizer_grads = False
        self.reuse_checkpoint_forward = True
        self.failure_stage = failure_stage

    def submit_forward(self, hidden_states, expert_ids, weights, save_for_backward=True):
        if self.failure_stage == "submit":
            raise RuntimeError("injected submit failure")
        super().submit_forward(hidden_states, expert_ids, weights, save_for_backward=save_for_backward)
        self.row_scale = torch.arange(
            1,
            hidden_states.shape[0] + 1,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).view(-1, 1)
        self.output = self.output * self.row_scale

    def sync_forward(self, output_device=None):
        if self.failure_stage == "sync":
            raise RuntimeError("injected sync failure")
        return super().sync_forward(output_device=output_device)

    def validate_checkpoint_output(self, qlen):
        if not self._kt_has_cached_forward or self._checkpoint_output_cpu is None:
            raise RuntimeError("No cached checkpoint forward output is available")
        if self._checkpoint_output_cpu.shape[0] != qlen:
            raise RuntimeError("Cached checkpoint qlen mismatch")

    def backward(self, grad_output, output_device=None, optimizer_grad_scale=1.0):
        del optimizer_grad_scale
        if self.failure_stage == "backward":
            raise RuntimeError("injected backward failure")
        self.backward_calls += 1
        grad_input = grad_output * self.weights * self.row_scale
        grad_weights = (grad_output * self.input * self.row_scale).sum(dim=-1, keepdim=True)
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights

    def update_lora_weights(self):
        if self.failure_stage == "refresh":
            raise RuntimeError("injected refresh failure")


class _DistributedOriginalMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = torch.nn.Linear(4, 2, bias=False)
        self.experts = torch.nn.ModuleList()


def _make_distributed_layer(
    backend,
    policy=KTActivationPolicy(cpu="retain", gpu="recompute"),
    local_gpu_experts=None,
):
    moe_config = SimpleNamespace(
        router_attr="gate",
        experts_attr="experts",
        has_shared_experts=False,
        router_type="linear",
        num_experts_per_tok=1,
    )
    layer = KTMoELayerWrapper(
        original_moe=_DistributedOriginalMoE(),
        wrapper=backend,
        lora_params=None,
        moe_config=moe_config,
        hidden_size=4,
        layer_idx=3,
        full_weight_grad=False,
        uses_authoritative_optimizer_grads=False,
        activation_policy=policy,
        lora_experts=local_gpu_experts,
    )
    layer.train()
    return layer


def _run_two_rank_workers(target, init_file, *worker_args):
    context = mp.get_context("spawn")
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
        results = [result_queue.get(timeout=25) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=25)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    return results


def _distributed_policy_flow_worker(rank, init_file, policy, use_checkpoint, result_queue):
    import torch.distributed as dist

    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=15),
        )
        torch.manual_seed(123)
        backend = _DistributedFakeWrapper() if rank == 0 else None
        layer = _make_distributed_layer(backend, policy=policy)
        qlen = rank + 2
        hidden_states = torch.randn(1, qlen, 4, requires_grad=True)

        if use_checkpoint:
            output = checkpoint(
                layer,
                hidden_states,
                use_reentrant=False,
                context_fn=get_activation_checkpoint_context_fn(),
            )
        else:
            output = layer(hidden_states)
        output.float().sum().backward()

        row_offset = 0 if rank == 0 else 2
        expected_input_grad = torch.arange(
            row_offset + 1,
            row_offset + qlen + 1,
            dtype=hidden_states.dtype,
        ).view(1, qlen, 1).expand_as(hidden_states)
        local_ok = bool(
            hidden_states.grad is not None
            and torch.allclose(hidden_states.grad, expected_input_grad)
            and layer.gate.weight.grad is not None
            and torch.isfinite(layer.gate.weight.grad).all()
        )
        if rank == 0:
            payload = (
                backend.submit_calls,
                backend.sync_calls,
                backend.cached_output_calls,
                backend.backward_calls,
                backend._kt_has_cached_forward,
                tuple(backend.save_for_backward_calls),
            )
        else:
            payload = None
        dist.barrier()
        result_queue.put((rank, "ok", local_ok, payload))
    except Exception as exc:
        result_queue.put((rank, "error", False, str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    ("policy", "use_checkpoint", "expected_rank0"),
    [
        (
            KTActivationPolicy(cpu="retain", gpu="retain"),
            False,
            (1, 1, 0, 1, False, (True,)),
        ),
        (
            KTActivationPolicy(cpu="retain", gpu="recompute"),
            True,
            (1, 1, 1, 1, False, (True,)),
        ),
        (
            KTActivationPolicy(cpu="recompute", gpu="recompute"),
            True,
            (2, 2, 0, 1, False, (False, True)),
        ),
    ],
)
@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_supported_policy_flow(tmp_path, policy, use_checkpoint, expected_rank0):
    init_file = tmp_path / f"policy-flow-{policy.cpu}-{policy.gpu}-init"
    results = _run_two_rank_workers(
        _distributed_policy_flow_worker,
        init_file,
        policy,
        use_checkpoint,
    )
    assert all(status == "ok" for _, status, _, _ in results), results
    assert all(local_ok for _, _, local_ok, _ in results)
    rank0_payload = next(payload for rank, _, _, payload in results if rank == 0)
    assert rank0_payload == expected_rank0


def _distributed_policy_mismatch_worker(rank, init_file, result_queue):
    import torch.distributed as dist

    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=15),
        )
        policy = (
            KTActivationPolicy(cpu="retain", gpu="retain")
            if rank == 0
            else KTActivationPolicy(cpu="recompute", gpu="recompute")
        )
        layer = _make_distributed_layer(
            _DistributedFakeWrapper() if rank == 0 else None,
            policy=policy,
        )
        try:
            layer(torch.randn(1, 2, 4, requires_grad=True))
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("policy mismatch was not rejected")
        dist.barrier()
        result_queue.put((rank, "ok", "control flow differs across ranks" in message))
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_policy_mismatch_fails_coherently_without_hang(tmp_path):
    results = _run_two_rank_workers(
        _distributed_policy_mismatch_worker,
        tmp_path / "policy-mismatch-init",
    )
    assert all(status == "ok" and matched for _, status, matched in results), results


def _distributed_rank0_failure_worker(rank, init_file, failure_stage, result_queue):
    import torch.distributed as dist

    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=15),
        )
        layer = _make_distributed_layer(
            _DistributedFakeWrapper(failure_stage=failure_stage) if rank == 0 else None,
            policy=KTActivationPolicy(cpu="recompute", gpu="recompute"),
        )
        if failure_stage == "refresh" and rank == 0:
            layer.wrapper._weights_loaded = True
            layer.wrapper._lora_initialized = True
            layer._lora_pointers_dirty = True
        hidden_states = torch.randn(1, rank + 2, 4, requires_grad=True)
        try:
            output = checkpoint(
                layer,
                hidden_states,
                use_reentrant=False,
                context_fn=get_activation_checkpoint_context_fn(),
            )
            if failure_stage == "backward":
                output.float().sum().backward()
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError(f"injected {failure_stage} failure was not propagated")
        dist.barrier()
        result_queue.put((rank, "ok", f"injected {failure_stage} failure" in message))
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("failure_stage", ["refresh", "submit", "sync", "backward"])
@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_rank0_failure_is_propagated_without_hang(tmp_path, failure_stage):
    results = _run_two_rank_workers(
        _distributed_rank0_failure_worker,
        tmp_path / f"rank0-{failure_stage}-init",
        failure_stage,
    )
    assert all(status == "ok" and matched for _, status, matched in results), results


class _RankOneFailingLocalExpert(torch.nn.Module):
    def forward(self, hidden_states):
        import torch.distributed as dist

        if dist.get_rank() == 1:
            raise RuntimeError("injected rank1 local expert failure")
        return torch.zeros_like(hidden_states)


def _distributed_rank1_local_failure_worker(rank, init_file, result_queue):
    import torch.distributed as dist

    try:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=15),
        )
        layer = _make_distributed_layer(
            _DistributedFakeWrapper() if rank == 0 else None,
            policy=KTActivationPolicy(cpu="recompute", gpu="recompute"),
            local_gpu_experts=_RankOneFailingLocalExpert(),
        )
        hidden_states = torch.randn(1, rank + 2, 4, requires_grad=True)
        try:
            checkpoint(
                layer,
                hidden_states,
                use_reentrant=False,
                context_fn=get_activation_checkpoint_context_fn(),
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("rank1 local expert failure was not propagated")
        dist.barrier()
        result_queue.put((rank, "ok", "injected rank1 local expert failure" in message))
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_two_rank_nonzero_rank_local_failure_is_propagated_without_hang(tmp_path):
    results = _run_two_rank_workers(
        _distributed_rank1_local_failure_worker,
        tmp_path / "rank1-local-failure-init",
    )
    assert all(status == "ok" and matched for _, status, matched in results), results
