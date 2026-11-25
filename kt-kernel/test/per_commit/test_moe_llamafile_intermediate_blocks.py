#!/usr/bin/env python
# coding=utf-8
"""Llamafile (GGUF) MoE with an intermediate size that is not a multiple of 256.

Q8_0 experts with intermediate size 640 (a size some 512-expert GGUF
checkpoints use): the TP split falls on 32-wide blocks of the down projection (320 + 320
on two NUMA parts) and forward_many tiles the rows by 64/32 instead of 256.
The reference is a float32 MoE on the dequantized weights with the same Q8_0
activation rounding the kernel applies.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default")

expert_num = 8
hidden_size = 1024
intermediate_size = 640
num_experts_per_tok = 2
max_len = 64
REL_L2_LIMIT = 0.01

ggml_type = kt_kernel_ext.kvcache.ggml_type


def make_cpu_infer(subpool_count):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = subpool_count
    config.subpool_numa_map = [0] * subpool_count
    config.subpool_thread_count = [4] * subpool_count
    return kt_kernel_ext.CPUInfer(config)


def roundtrip(x, t):
    x = x.contiguous().view(-1)
    q = kt_kernel_ext.utils.from_float(x.data_ptr(), x.numel(), t)
    return q, kt_kernel_ext.utils.to_float(q.data_ptr(), x.numel(), t)


def quantize(w):
    q, deq = roundtrip(w, ggml_type.Q8_0)
    return q, deq.view_as(w)


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def moe_torch(x, expert_ids, weights, gate, up, down):
    out = torch.zeros(x.shape[0], hidden_size, dtype=torch.float32)
    for t in range(x.shape[0]):
        xq = roundtrip(x[t], ggml_type.Q8_0)[1]
        for j in range(expert_ids.shape[1]):
            e = int(expert_ids[t, j])
            h = act_fn(gate[e] @ xq) * (up[e] @ xq)
            out[t] += weights[t, j] * (down[e] @ roundtrip(h, ggml_type.Q8_0)[1])
    return out


@pytest.mark.cpu
@pytest.mark.parametrize("subpool_count", [1, 2])
@pytest.mark.parametrize("qlen", [1, 32])
def test_llamafile_intermediate_not_multiple_of_256(subpool_count, qlen):
    torch.manual_seed(0)
    cpu_infer = make_cpu_infer(subpool_count)
    with torch.inference_mode():
        gate_q, gate = quantize(torch.randn(expert_num, intermediate_size, hidden_size) / 32)
        up_q, up = quantize(torch.randn(expert_num, intermediate_size, hidden_size) / 32)
        down_q, down = quantize(torch.randn(expert_num, hidden_size, intermediate_size) / 32)

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
        identity = torch.arange(expert_num, dtype=torch.int64)
        cpu_infer.submit(moe.load_weights_task(identity.data_ptr()))
        cpu_infer.sync()

        expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
        x = torch.randn((qlen, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
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

        ref = moe_torch(x.float(), expert_ids, weights, gate, up, down)
        rel = (torch.linalg.norm(out.float() - ref) / torch.linalg.norm(ref)).item()
        print(f"subpools={subpool_count} qlen={qlen}: relative L2 {rel:.4f}")
        assert rel < REL_L2_LIMIT, f"subpools={subpool_count} qlen={qlen}: relative L2 {rel:.4f} >= {REL_L2_LIMIT}"


if __name__ == "__main__":
    for subpools in (1, 2):
        for q in (1, 32):
            test_llamafile_intermediate_not_multiple_of_256(subpools, q)
    print("ALL TESTS PASSED")
