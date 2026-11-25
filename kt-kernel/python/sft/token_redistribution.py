# SPDX-License-Identifier: Apache-2.0
"""Redistribute token rows among ranks with replicated CPU experts."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from .autograd import _sync_any_rank_exception


@dataclass(frozen=True)
class _Transfer:
    source: int
    worker: int
    source_start: int
    worker_start: int
    rows: int


@dataclass(frozen=True)
class TokenRedistributionPlan:
    """Preserve each rank's local prefix and move only its surplus rows.

    Counts are ordered by group-local rank. All ranks must use the same plan
    and call the transfers in the same order on an initialized Gloo group.
    Use detached buffers. The caller owns expert execution, gradient transport,
    and parameter gradient aggregation.
    """

    source_counts: tuple[int, ...]
    worker_counts: tuple[int, ...]
    _transfers: tuple[_Transfer, ...]

    @classmethod
    def from_counts(cls, counts: Sequence[int]) -> "TokenRedistributionPlan":
        source = tuple(counts)
        if not source or any(type(n) is not int or n < 0 for n in source):
            raise ValueError("Expected a nonempty sequence of nonnegative token counts")
        rows, remainder = divmod(sum(source), len(source))
        # Assign remainder rows to ranks already holding the most data.
        priority = sorted(range(len(source)), key=lambda rank: (-source[rank], rank))
        target = [rows] * len(source)
        for rank in priority[:remainder]:
            target[rank] += 1

        donors = [[i, target[i], n - target[i]] for i, n in enumerate(source) if n > target[i]]
        receivers = [[i, n, target[i] - n] for i, n in enumerate(source) if n < target[i]]
        transfers = []
        donor_index = receiver_index = 0
        while donor_index < len(donors):
            src, start, excess = donors[donor_index]
            dst, offset, missing = receivers[receiver_index]
            count = min(excess, missing)
            transfers.append(_Transfer(src, dst, start, offset, count))
            donors[donor_index] = [src, start + count, excess - count]
            receivers[receiver_index] = [dst, offset + count, missing - count]
            donor_index += count == excess
            receiver_index += count == missing
        return cls(source, tuple(target), tuple(transfers))

    @property
    def moved_rows(self) -> int:
        return sum(t.rows for t in self._transfers)

    def to_workers(self, rows: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return self._exchange(rows, group, reverse=False)

    def to_sources(self, rows: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        return self._exchange(rows, group, reverse=True)

    def _exchange(self, rows: torch.Tensor, group: dist.ProcessGroup | None, reverse: bool) -> torch.Tensor:
        if dist.get_backend(group) != "gloo" or dist.get_world_size(group) != len(self.source_counts):
            raise ValueError("The plan requires a Gloo group matching its rank count")
        rank = dist.get_rank(group)
        incoming, outgoing = (
            (self.worker_counts, self.source_counts) if reverse else (self.source_counts, self.worker_counts)
        )
        if rows.device.type != "cpu" or rows.ndim == 0 or rows.shape[0] != incoming[rank] or rows.requires_grad:
            raise ValueError("Detached CPU tensor leading dimension must match the plan's local count")
        involved = [t for t in self._transfers if rank in (t.source, t.worker)]
        if not involved:
            return rows
        if 0 in rows.shape[1:]:
            return rows.new_empty((outgoing[rank], *rows.shape[1:]))
        rows = rows.contiguous()
        result = rows.new_empty((outgoing[rank], *rows.shape[1:]))
        kept = min(incoming[rank], outgoing[rank])
        result[:kept].copy_(rows[:kept])
        ops = []
        for t in involved:
            send, recv, send_offset, recv_offset = (
                (t.worker, t.source, t.worker_start, t.source_start)
                if reverse
                else (t.source, t.worker, t.source_start, t.worker_start)
            )
            if rank == send:
                peer, buffer, operation = recv, rows.narrow(0, send_offset, t.rows), dist.isend
            else:
                peer, buffer, operation = send, result.narrow(0, recv_offset, t.rows), dist.irecv
            if group is not None:
                peer = dist.get_global_rank(group, peer)
            ops.append(dist.P2POp(operation, buffer, peer, group))
        for request in dist.batch_isend_irecv(ops):
            request.wait()
        return result


def _exchange_columns(exchange, group, *columns):
    columns = [column.detach().cpu().contiguous() for column in columns]
    buffers = [column.view(torch.uint8) for column in columns]
    packed = exchange(torch.cat(buffers, dim=1), group)
    return [
        chunk.contiguous().view(column.dtype)
        for chunk, column in zip(packed.split([buffer.shape[1] for buffer in buffers], dim=1), columns)
    ]


def _replica_lora_params(layer, group):
    entries = layer.wrapper.authoritative_optimizer_grads
    dirty = torch.tensor([not getattr(layer, "_kt_replica_synced", False) or layer._lora_pointers_dirty])
    dist.broadcast(dirty, src=0, group=group)
    if dirty.item():
        for entry in entries:
            dist.broadcast(entry.parameter.detach().view(torch.uint8), src=0, group=group)
        layer.update_lora_pointers()
        layer._lora_pointers_dirty = False
        layer._kt_replica_synced = True


def _flat_grads(entries):
    return torch.cat([entry.grad_view.flatten().float() for entry in entries])


class ReplicatedMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, ids, scores, lora_ref, layer, counts, checkpoint_action, save_for_backward):
        group, backend = layer._kt_replica_group, layer.wrapper
        rank = dist.get_rank(group)
        plan = TokenRedistributionPlan.from_counts(counts)
        worker_rows = plan.worker_counts[rank]
        error = None
        try:
            if checkpoint_action != "reuse_recompute":
                _replica_lora_params(layer, group)
        except Exception as exc:
            error = exc
        _sync_any_rank_exception(error, device=torch.device("cpu"), context="Refreshing KT replicas", group=group)

        if checkpoint_action != "reuse_recompute":
            inputs = _exchange_columns(
                plan.to_workers, group, hidden.reshape(counts[rank], hidden.shape[-1]), ids, scores
            )
        try:
            if not worker_rows:
                output = torch.empty((0, hidden.shape[-1]), dtype=torch.bfloat16)
            elif checkpoint_action == "reuse_recompute":
                output = backend.get_checkpoint_output(worker_rows)
            else:
                if checkpoint_action == "cache_first_forward":
                    backend.validate_checkpoint_cache_empty()
                output = backend.forward(*inputs, save_for_backward=save_for_backward)
                if checkpoint_action == "cache_first_forward":
                    backend.cache_checkpoint_output(output, worker_rows)
        except Exception as exc:
            error = exc
        _sync_any_rank_exception(error, device=torch.device("cpu"), context="KT replica forward", group=group)

        ctx.plan, ctx.layer = plan, layer
        ctx.score_shape, ctx.score_dtype, ctx.score_device = scores.shape, scores.dtype, scores.device
        ctx.save_for_backward(hidden.new_empty(()))
        return plan.to_sources(output, group).to(hidden).reshape_as(hidden)

    @staticmethod
    def backward(ctx, grad_output):
        layer, plan = ctx.layer, ctx.plan
        group, backend = layer._kt_replica_group, layer.wrapper
        rank = dist.get_rank(group)
        cpu = torch.device("cpu")
        error = None
        try:
            _ = ctx.saved_tensors
        except Exception as exc:
            error = exc
        _sync_any_rank_exception(error, device=cpu, context="KT replica checkpoint recompute", group=group)
        grad = plan.to_workers(grad_output.detach().flatten(0, 1).cpu(), group)

        entries = backend.authoritative_optimizer_grads
        previous = None
        try:
            if rank == 0 or not grad.shape[0]:
                # Native lazy clearing only tracks locally active experts.
                if rank == 0 and backend.validate_authoritative_optimizer_grad_state() == "open":
                    previous = _flat_grads(entries)
                    backend.release_authoritative_optimizer_grads()
                for entry in entries:
                    entry.grad_view.zero_()
            if grad.shape[0]:
                dx, ds = backend.backward(grad, optimizer_grad_scale=1.0 / len(plan.source_counts))
            else:
                dx = grad.bfloat16()
                ds = torch.empty((0, ctx.score_shape[-1]), dtype=torch.bfloat16)
                if rank == 0:
                    backend._prepare_authoritative_optimizer_grad_write(1.0)
                    backend._publish_authoritative_optimizer_grads()
            backend.clear_checkpoint_output()
        except Exception as exc:
            error = exc
        try:
            _sync_any_rank_exception(error, device=cpu, context="KT replica backward", group=group)
            gradients = _flat_grads(entries)
            dist.reduce(gradients, dst=0, group=group)
            if rank == 0:
                if previous is not None:
                    gradients.add_(previous)
                for entry, chunk in zip(entries, gradients.split([entry.grad_view.numel() for entry in entries])):
                    entry.grad_view.copy_(chunk.view_as(entry.grad_view))
            dx, ds = _exchange_columns(plan.to_sources, group, dx, ds.bfloat16())
        except Exception:
            backend.release_authoritative_optimizer_grads()
            raise
        if rank != 0:
            backend.release_authoritative_optimizer_grads()
        dx = dx.to(grad_output).reshape_as(grad_output)
        ds = ds.to(device=ctx.score_device, dtype=ctx.score_dtype).reshape(ctx.score_shape)
        return dx, None, ds, None, None, None, None, None
