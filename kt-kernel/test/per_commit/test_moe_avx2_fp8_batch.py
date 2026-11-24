"""Check that AVX2 FP8 expert results are independent of the routed batch size."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ci.ci_register import register_cpu_ci
from kt_kernel import kt_kernel_ext

register_cpu_ci(est_time=5, suite="default")


def make_moe(backend, group_size, projections, expert_num, top_k, hidden, intermediate, max_len):
    worker = kt_kernel_ext.WorkerPoolConfig()
    worker.subpool_count = 1
    worker.subpool_numa_map = [0]
    worker.subpool_thread_count = [4]
    cpu = kt_kernel_ext.CPUInfer(worker)
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, top_k, hidden, intermediate, 0)
    cfg.max_len = max_len
    cfg.pool = cpu.backend_
    cfg.quant_config.bits = 8
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False
    if group_size == 32:
        cfg.swiglu_alpha = 1.702
        cfg.swiglu_limit = 7.0
    for name, (weight, scale) in zip(("gate", "up", "down"), projections):
        setattr(cfg, f"{name}_proj", weight.data_ptr())
        setattr(cfg, f"{name}_scale", scale.data_ptr())
    moe = backend(cfg)
    expert_map = torch.arange(expert_num, dtype=torch.int64)
    cpu.submit(moe.load_weights_task(expert_map.data_ptr()))
    cpu.sync()
    return moe, cpu


def forward(moe, cpu, x, ids, weights):
    batch = torch.tensor([x.size(0)], dtype=torch.int32)
    y = torch.empty_like(x)
    cpu.submit(
        moe.forward_task(
            batch.data_ptr(), ids.size(1), ids.data_ptr(), weights.data_ptr(), x.data_ptr(), y.data_ptr(), False
        )
    )
    cpu.sync()
    return y


@pytest.mark.cpu
@pytest.mark.parametrize(
    "backend_name,group_size,hidden,intermediate",
    [
        ("AVX2FP8_MOE", 128, 144, 272),
        ("AVX2MXFP8_MOE", 32, 96, 160),
    ],
)
def test_batched_experts_match_decode(backend_name, group_size, hidden, intermediate):
    backend = getattr(kt_kernel_ext.moe, backend_name, None)
    if backend is None:
        pytest.skip(f"{backend_name} is unavailable")
    rng = torch.Generator().manual_seed(47)
    expert_num, top_k, m = 3, 2, 7

    def projection(n, k):
        weight = torch.randint(256, (expert_num, n, k), dtype=torch.uint8, generator=rng)
        if group_size == 32:
            scale = torch.randint(114, 118, (expert_num, n, k // 32), dtype=torch.uint8, generator=rng)
        else:
            scale = torch.rand((expert_num, (n + 127) // 128, (k + 127) // 128), generator=rng) * 0.01 + 0.001
        return weight, scale

    projections = (projection(intermediate, hidden), projection(intermediate, hidden), projection(hidden, intermediate))
    moe, cpu = make_moe(backend, group_size, projections, expert_num, top_k, hidden, intermediate, m)
    x = torch.randn((m, hidden), generator=rng).to(torch.bfloat16) * 0.1
    # Give the three experts 7, 4 and 3 rows, including partial row tiles.
    ids = torch.tensor([[0, 1]] * 4 + [[0, 2]] * 3)
    weights = torch.rand((m, top_k), generator=rng)
    expected = torch.cat([forward(moe, cpu, x[i : i + 1], ids[i : i + 1], weights[i : i + 1]) for i in range(m)])
    actual = forward(moe, cpu, x, ids, weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_fp8_reduces_before_scaling():
    backend = getattr(kt_kernel_ext.moe, "AVX2FP8_MOE", None)
    if backend is None:
        pytest.skip("AVX2FP8_MOE is unavailable")
    dim = 128
    gate = torch.zeros((1, dim, dim), dtype=torch.uint8)
    gate[:, :, 0] = 0x50
    gate[:, :, 1] = 0xD0
    up = torch.full_like(gate, 0x38)
    scale = torch.ones((1, 1, 1))
    projections = ((gate, scale * 1e38), (up, scale), (up, scale))
    moe, cpu = make_moe(backend, dim, projections, 1, 1, dim, dim, 3)
    x = torch.ones((3, dim), dtype=torch.bfloat16)
    ids = torch.zeros((3, 1), dtype=torch.int64)
    weights = torch.ones((3, 1))
    # The two gate products cancel before the block scale is applied.
    torch.testing.assert_close(forward(moe, cpu, x, ids, weights), torch.zeros_like(x), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
