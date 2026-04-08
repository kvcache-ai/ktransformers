# Distributed and checkpoint utilities for SFT
# SPDX-License-Identifier: Apache-2.0

"""
Shared distributed communication and gradient-checkpoint detection helpers.

This is a leaf module — no imports from other sft/ submodules.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any

import torch


def _all_gather_qlens(local_qlen: int, device: torch.device, world_size: int) -> list[int]:
    import torch.distributed as dist

    local_qlen_t = torch.tensor([int(local_qlen)], device=device, dtype=torch.int64)
    gathered = [torch.empty(1, device=device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(gathered, local_qlen_t)
    return [int(t.item()) for t in gathered]


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


def _is_in_checkpoint_first_forward() -> bool:
    """Best-effort detection for non-reentrant checkpoint first forward."""
    try:
        for frame_info in inspect.stack(context=0):
            fn = frame_info.function
            file = frame_info.filename or ""
            if fn == "custom_gradient_checkpointing_func" and file.endswith("checkpointing.py"):
                return True
    except Exception:
        return False
    return False


def _checkpoint_hook_mode() -> str:
    """Infer checkpoint phase from current saved_tensors_hooks top.

    Returns one of:
      - "first_forward": non-reentrant checkpoint's _checkpoint_hook
      - "recompute": non-reentrant checkpoint's _recomputation_hook
      - "none": no default saved_tensors_hooks on top
      - "other": unknown hook stack entry
      - "error": failed to query hook stack
    """
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
