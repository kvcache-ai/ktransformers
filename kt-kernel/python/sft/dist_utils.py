# Distributed and checkpoint utilities for SFT
# SPDX-License-Identifier: Apache-2.0

"""
Shared distributed communication and gradient-checkpoint detection helpers.

This is a leaf module — no imports from other sft/ submodules.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import os
from typing import Iterator

import torch


_CHECKPOINT_PHASE: ContextVar[str | None] = ContextVar(
    "kt_activation_checkpoint_phase",
    default=None,
)
_CHECKPOINT_PHASE_IDS = {
    "none": 0,
    "first_forward": 1,
    "recompute": 2,
    "other": 3,
    "error": 4,
}
_CHECKPOINT_ACTION_IDS = {
    "normal": 0,
    "cache_first_forward": 1,
    "reuse_recompute": 2,
}
_ACTIVATION_POLICY_IDS = {
    None: 0,
    "recompute": 1,
    "retain": 2,
}


@contextmanager
def _activation_checkpoint_phase(phase: str) -> Iterator[None]:
    token = _CHECKPOINT_PHASE.set(phase)
    try:
        yield
    finally:
        _CHECKPOINT_PHASE.reset(token)


def get_activation_checkpoint_context_fn():
    """Return a ``torch.utils.checkpoint`` non-reentrant context factory."""

    def context_fn():
        return (
            _activation_checkpoint_phase("first_forward"),
            _activation_checkpoint_phase("recompute"),
        )

    return context_fn


def _distributed_rank_world_size() -> tuple[int, int]:
    """Return rank/world size during both model construction and runtime.

    Accelerate/torchrun may construct the model before the process group is
    initialized.  In that phase the standard launcher environment is the only
    reliable way to keep rank-0 KT ownership and rank-0 buffer capacity
    consistent across processes.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())

    rank_text = os.environ.get("RANK")
    world_text = os.environ.get("WORLD_SIZE")
    if rank_text is None or world_text is None:
        return 0, 1
    try:
        rank = int(rank_text)
        world_size = int(world_text)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid distributed launcher environment: RANK={rank_text!r}, WORLD_SIZE={world_text!r}"
        ) from exc
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise RuntimeError(
            f"Invalid distributed launcher environment: rank={rank}, world_size={world_size}"
        )
    return rank, world_size


def _all_gather_qlens(local_qlen: int, device: torch.device, world_size: int) -> list[int]:
    import torch.distributed as dist

    local_qlen_t = torch.tensor([int(local_qlen)], device=device, dtype=torch.int64)
    gathered = [torch.empty(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local_qlen_t)
    return [int(t.item()) for t in gathered]


def _all_gather_checkpoint_state(
    local_qlen: int,
    *,
    layer_idx: int,
    phase: str,
    action: str,
    cpu_policy: str | None = None,
    gpu_policy: str | None = None,
    owner_valid: bool = True,
    device: torch.device,
    world_size: int,
) -> list[int]:
    """Agree on checkpoint control flow before entering conditional collectives.

    Sequence lengths may differ by rank. Layer, phase, and action must not:
    disagreement would otherwise let ranks enter different gather/scatter paths.
    """
    import torch.distributed as dist

    phase_id = _CHECKPOINT_PHASE_IDS.get(phase, -1)
    action_id = _CHECKPOINT_ACTION_IDS.get(action, -1)
    cpu_policy_id = _ACTIVATION_POLICY_IDS.get(cpu_policy, -1)
    gpu_policy_id = _ACTIVATION_POLICY_IDS.get(gpu_policy, -1)
    local_state = torch.tensor(
        [
            int(local_qlen),
            int(layer_idx),
            phase_id,
            action_id,
            cpu_policy_id,
            gpu_policy_id,
            int(owner_valid),
        ],
        device=device,
        dtype=torch.int64,
    )
    if world_size == 1:
        gathered = [local_state]
    else:
        if not dist.is_initialized():
            raise RuntimeError(
                "Checkpoint state agreement requires an initialized process group"
            )
        actual_world_size = int(dist.get_world_size())
        if actual_world_size != int(world_size):
            raise RuntimeError(
                f"Checkpoint state world-size mismatch: got {actual_world_size}, "
                f"expected {world_size}"
            )
        gathered = [
            torch.empty(7, device=device, dtype=torch.int64)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, local_state)

    states = [tuple(int(value) for value in state.tolist()) for state in gathered]
    invalid = []
    for rank, (
        qlen,
        gathered_layer,
        gathered_phase,
        gathered_action,
        gathered_cpu_policy,
        gathered_gpu_policy,
        _gathered_owner_valid,
    ) in enumerate(states):
        if qlen < 0:
            invalid.append(f"rank {rank} qlen={qlen}")
        if gathered_layer < 0:
            invalid.append(f"rank {rank} layer={gathered_layer}")
        if gathered_phase < 0:
            invalid.append(f"rank {rank} phase=invalid")
        if gathered_action < 0:
            invalid.append(f"rank {rank} action=invalid")
        if gathered_cpu_policy < 0:
            invalid.append(f"rank {rank} cpu_policy=invalid")
        if gathered_gpu_policy < 0:
            invalid.append(f"rank {rank} gpu_policy=invalid")
    controls = {
        (layer, phase_id_, action_id_, cpu_policy_id_, gpu_policy_id_, owner_valid_)
        for (
            _,
            layer,
            phase_id_,
            action_id_,
            cpu_policy_id_,
            gpu_policy_id_,
            owner_valid_,
        ) in states
    }
    if invalid or len(controls) != 1:
        detail = ", ".join(
            f"rank {rank}: qlen={qlen}, layer={layer}, phase_id={phase_id_}, "
            f"action_id={action_id_}, cpu_policy_id={cpu_policy_id_}, "
            f"gpu_policy_id={gpu_policy_id_}, owner_valid={owner_valid_}"
            for rank, (
                qlen,
                layer,
                phase_id_,
                action_id_,
                cpu_policy_id_,
                gpu_policy_id_,
                owner_valid_,
            ) in enumerate(states)
        )
        prefix = f"Invalid checkpoint state ({'; '.join(invalid)}). " if invalid else ""
        raise RuntimeError(
            f"{prefix}KT checkpoint control flow differs across ranks: {detail}"
        )
    return [state[0] for state in states]


def _qlen_offsets(all_qlens: list[int]) -> list[int]:
    offsets = [0]
    for q in all_qlens:
        offsets.append(offsets[-1] + int(q))
    return offsets


def _dist_gather_varlen_to_rank0(
    local_tensor: torch.Tensor,
    *,
    all_qlens: list[int],
    rank: int,
    world_size: int,
) -> list[torch.Tensor] | None:
    import torch.distributed as dist

    local_tensor = local_tensor.contiguous()
    local_expected = int(all_qlens[rank])
    if local_tensor.shape[0] != local_expected:
        raise RuntimeError(
            f"Local leading dim mismatch on rank {rank}: got {local_tensor.shape[0]}, expected {local_expected}"
        )

    if rank == 0:
        gathered: list[torch.Tensor | None] = [None] * world_size
        gathered[0] = local_tensor
        ops: list[dist.P2POp] = []
        for src in range(1, world_size):
            qlen_src = int(all_qlens[src])
            recv_shape = (qlen_src, *local_tensor.shape[1:])
            recv = torch.empty(recv_shape, device=local_tensor.device, dtype=local_tensor.dtype)
            gathered[src] = recv
            if qlen_src > 0:
                ops.append(dist.P2POp(dist.irecv, recv, src))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        out: list[torch.Tensor] = []
        for idx, t in enumerate(gathered):
            if t is None:
                raise RuntimeError(f"Missing gathered tensor for rank {idx} on rank0.")
            out.append(t)
        return out

    if local_expected > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, local_tensor, 0)])
        for req in reqs:
            req.wait()
    return None


def _dist_scatter_varlen_from_rank0(
    *,
    rank0_chunks: list[torch.Tensor] | None,
    all_qlens: list[int],
    rank: int,
    world_size: int,
    feature_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    import torch.distributed as dist

    local_qlen = int(all_qlens[rank])
    local_out = torch.empty((local_qlen, *feature_shape), device=device, dtype=dtype)

    if rank == 0:
        if rank0_chunks is None or len(rank0_chunks) != world_size:
            raise RuntimeError("rank0_chunks must contain one chunk per rank on rank0.")
        if int(rank0_chunks[0].shape[0]) != local_qlen:
            raise RuntimeError(
                f"Rank0 local chunk mismatch: got {rank0_chunks[0].shape[0]}, expected {local_qlen}"
            )
        if local_qlen > 0:
            local_out.copy_(rank0_chunks[0])
        ops: list[dist.P2POp] = []
        for dst in range(1, world_size):
            qlen_dst = int(all_qlens[dst])
            if qlen_dst <= 0:
                continue
            chunk = rank0_chunks[dst].contiguous()
            if int(chunk.shape[0]) != qlen_dst:
                raise RuntimeError(
                    f"Rank{dst} chunk mismatch on rank0: got {chunk.shape[0]}, expected {qlen_dst}"
                )
            ops.append(dist.P2POp(dist.isend, chunk, dst))
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
        return local_out

    if local_qlen > 0:
        reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, local_out, 0)])
        for req in reqs:
            req.wait()
    return local_out



def _checkpoint_hook_mode() -> str:
    """Infer checkpoint phase from current saved_tensors_hooks top.

    Returns one of:
      - "first_forward": non-reentrant checkpoint's _checkpoint_hook
      - "recompute": non-reentrant checkpoint's _recomputation_hook
      - "none": no default saved_tensors_hooks on top
      - "other": unknown hook stack entry
      - "error": failed to query hook stack
    """
    explicit_phase = _CHECKPOINT_PHASE.get()
    if explicit_phase is not None:
        return explicit_phase

    try:
        top = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:
        return "error"
    if top is None:
        return "none"
    try:
        pack_fn, _ = top
        mod = getattr(pack_fn, "__module__", "")
        qual = getattr(pack_fn, "__qualname__", getattr(pack_fn, "__name__", ""))
        tag = f"{mod}.{qual}"
    except Exception:
        return "other"
    if "_recomputation_hook.__init__.<locals>.pack_hook" in tag:
        return "recompute"
    if "_checkpoint_hook.__init__.<locals>.pack_hook" in tag:
        return "first_forward"
    return "other"


def _maybe_zero3_gathered_parameters(params: list[torch.nn.Parameter]):
    if not params:
        return nullcontext()
    try:
        from transformers.integrations import is_deepspeed_zero3_enabled
    except Exception:
        return nullcontext()
    if not is_deepspeed_zero3_enabled():
        return nullcontext()
    try:
        import deepspeed  # type: ignore
    except Exception:
        return nullcontext()
    return deepspeed.zero.GatheredParameters(params, modifier_rank=0)
