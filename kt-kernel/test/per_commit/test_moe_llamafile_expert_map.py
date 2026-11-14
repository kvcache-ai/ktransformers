#!/usr/bin/env python
# coding=utf-8
"""Llamafile (GGUF) MoE honours physical_to_logical_map.

Two MoEs share the same quantized experts: one loaded with the identity map,
one with a permutation. Routing the second through the physical ids that map to
the first one's logical ids must give bit-identical outputs, both with one NUMA
part (the load aliases the source when the map is the identity) and with two
(per-part copies).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default")

expert_num = 16
hidden_size = 512
intermediate_size = 512
num_experts_per_tok = 4
max_len = 64

ggml_type = kt_kernel_ext.kvcache.ggml_type


def make_cpu_infer(subpool_count):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = subpool_count
    config.subpool_numa_map = [0] * subpool_count
    config.subpool_thread_count = [4] * subpool_count
    return kt_kernel_ext.CPUInfer(config)


def quantize(w):
    flat = w.contiguous().view(-1)
    return kt_kernel_ext.utils.from_float(flat.data_ptr(), flat.numel(), ggml_type.Q8_0)


def make_moe(cpu_infer, weights, physical_to_logical_map):
    gate_q, up_q, down_q = weights
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
    config.m_block = 32
    config.group_min_len = 10
    config.max_len = max_len
    config.group_max_len = max_len
    config.gate_proj = gate_q.data_ptr()
    config.up_proj = up_q.data_ptr()
    config.down_proj = down_q.data_ptr()
    config.gate_type = ggml_type.Q8_0
    config.up_type = ggml_type.Q8_0
    config.down_type = ggml_type.Q8_0
    config.hidden_type = ggml_type.BF16
    config.pool = cpu_infer.backend_
    moe = kt_kernel_ext.moe.MOE(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpu_infer.sync()
    return moe


def run(cpu_infer, moe, expert_ids, weights, x):
    qlen = x.shape[0]
    out = torch.empty((qlen, hidden_size), dtype=torch.bfloat16)
    bsz = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            bsz.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            x.data_ptr(),
            out.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return out


@pytest.mark.cpu
@pytest.mark.parametrize("subpool_count", [1, 2])
@pytest.mark.parametrize("qlen", [1, 16])
def test_llamafile_expert_map(subpool_count, qlen):
    torch.manual_seed(0)
    cpu_infer = make_cpu_infer(subpool_count)
    with torch.inference_mode():
        weights = (
            quantize(torch.randn(expert_num, intermediate_size, hidden_size) / 32),
            quantize(torch.randn(expert_num, intermediate_size, hidden_size) / 32),
            quantize(torch.randn(expert_num, hidden_size, intermediate_size) / 32),
        )
        identity = torch.arange(expert_num, dtype=torch.int64)
        p2l = torch.randperm(expert_num).to(torch.int64)  # slot i holds logical expert p2l[i]
        l2p = torch.empty_like(p2l)
        l2p[p2l] = torch.arange(expert_num, dtype=torch.int64)

        moe_ref = make_moe(cpu_infer, weights, identity)
        moe_map = make_moe(cpu_infer, weights, p2l)

        logical = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
        physical = l2p[logical].contiguous()
        w = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        x = torch.randn((qlen, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()

        out_ref = run(cpu_infer, moe_ref, logical, w, x)
        out_map = run(cpu_infer, moe_map, physical, w, x)
        assert torch.equal(out_ref, out_map), (
            f"subpools={subpool_count} qlen={qlen}: max diff {(out_ref.float() - out_map.float()).abs().max().item()}"
        )


if __name__ == "__main__":
    for subpools in (1, 2):
        for q in (1, 16):
            test_llamafile_expert_map(subpools, q)
            print(f"subpools={subpools} qlen={q}: identical")
    print("ALL TESTS PASSED")
