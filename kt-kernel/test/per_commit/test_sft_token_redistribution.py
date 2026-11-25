# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kt_kernel.sft.token_redistribution import TokenRedistributionPlan


@pytest.mark.parametrize(
    ("counts", "expected", "moved"),
    [
        ([40, 8, 8, 8], (16, 16, 16, 16), 24),
        ([0, 7, 1, 5], (3, 4, 3, 3), 5),
        ([0, 0], (0, 0), 0),
        ([1, 0], (1, 0), 0),
        ([2, 2], (2, 2), 0),
    ],
)
def test_plan_moves_only_surplus(counts, expected, moved):
    plan = TokenRedistributionPlan.from_counts(counts)
    assert plan.worker_counts == expected
    assert plan.moved_rows == moved


@pytest.mark.parametrize("counts", [[], [1, -1], [1.5, 0]])
def test_rejects_invalid_counts(counts):
    with pytest.raises(ValueError, match="nonnegative token counts"):
        TokenRedistributionPlan.from_counts(counts)


def _roundtrip(rank, path):
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=4)
    try:
        for counts in ([40, 8, 8, 8], [0, 7, 1, 5], [0, 0, 0, 0], [2, 2, 2, 2]):
            plan = TokenRedistributionPlan.from_counts(counts)
            source = torch.arange(counts[rank] * 2, dtype=torch.int64).reshape(-1, 2) + 1000 * rank
            dist.barrier()
            worker = plan.to_workers(source)
            assert worker.shape == (plan.worker_counts[rank], 2)
            kept = min(counts[rank], worker.shape[0])
            assert torch.equal(worker[:kept], source[:kept])
            returned = plan.to_sources(3 * worker + 1)
            assert torch.equal(returned, 3 * source + 1)
            empty = torch.empty((counts[rank], 0))
            assert plan.to_workers(empty).shape == (plan.worker_counts[rank], 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_redistribution_roundtrip(tmp_path):
    mp.spawn(_roundtrip, args=(tmp_path / "gloo-init",), nprocs=4)
