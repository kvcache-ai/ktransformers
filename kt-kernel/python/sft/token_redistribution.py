# SPDX-License-Identifier: Apache-2.0
"""Redistribute token rows among ranks with replicated CPU experts."""

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist


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
        """Move CPU input rows to the workers that will execute their experts."""
        return self._exchange(rows, group, reverse=False)

    def to_sources(self, rows: torch.Tensor, group: dist.ProcessGroup | None = None) -> torch.Tensor:
        """Return worker outputs or gradients in their original row order."""
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
