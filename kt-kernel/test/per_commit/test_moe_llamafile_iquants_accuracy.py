#!/usr/bin/env python
# coding=utf-8
"""Llamafile (GGUF) MoE accuracy for IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_NL experts.

ggml's quantizers for some of these types need an importance matrix, so the
experts are random blocks (every grid index, sign and scale pattern is a valid
block) with a fixed fp16 scale. The reference is a float32 MoE on the weights
ggml dequantizes, with the activations rounded through the same vec_dot type
the kernel uses (Q8_K for the 256-wide i-quants, Q8_0 for IQ4_NL), so what is
left is the kernel's own error plus the bf16 output.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default")

expert_num = 8
hidden_size = 1024
intermediate_size = 512
num_experts_per_tok = 2
max_len = 64
REL_L2_LIMIT = 0.01

ggml_type = kt_kernel_ext.kvcache.ggml_type

# type name -> (block bytes, block elements, activation type of its vec_dot)
TYPES = {
    "IQ2_XS": (74, 256, "Q8_K"),
    "IQ2_S": (82, 256, "Q8_K"),
    "IQ3_XXS": (98, 256, "Q8_K"),
    "IQ3_S": (110, 256, "Q8_K"),
    "IQ4_NL": (18, 32, "Q8_0"),
}


def make_cpu_infer(subpool_count):
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = subpool_count
    config.subpool_numa_map = [0] * subpool_count
    config.subpool_thread_count = [4] * subpool_count
    return kt_kernel_ext.CPUInfer(config)


def random_blocks(type_name, shape, gen):
    """Random blocks of the type with an fp16 scale d = 2^-7; returns (bytes, dequantized float32 of shape)."""
    block_bytes, block_elems, _ = TYPES[type_name]
    n = shape[0] * shape[1] * shape[2]
    blocks = torch.randint(0, 256, (n // block_elems, block_bytes), dtype=torch.uint8, generator=gen)
    blocks[:, :2] = torch.tensor([2.0**-7], dtype=torch.float16).view(torch.uint8)
    q = blocks.contiguous().view(-1)
    deq = kt_kernel_ext.utils.to_float(q.data_ptr(), n, getattr(ggml_type, type_name))
    return q, deq.view(*shape)


def roundtrip(x, type_name):
    x = x.contiguous().view(-1)
    t = getattr(ggml_type, type_name)
    q = kt_kernel_ext.utils.from_float(x.data_ptr(), x.numel(), t)
    return kt_kernel_ext.utils.to_float(q.data_ptr(), x.numel(), t)


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def moe_torch(x, expert_ids, weights, gate, up, down, act_type):
    out = torch.zeros(x.shape[0], hidden_size, dtype=torch.float32)
    for t in range(x.shape[0]):
        xq = roundtrip(x[t], act_type)
        for j in range(expert_ids.shape[1]):
            e = int(expert_ids[t, j])
            h = act_fn(gate[e] @ xq) * (up[e] @ xq)
            out[t] += weights[t, j] * (down[e] @ roundtrip(h, act_type))
    return out


@pytest.mark.cpu
@pytest.mark.parametrize("type_name", list(TYPES))
@pytest.mark.parametrize("qlen", [1, 32])
def test_llamafile_iquant_accuracy(type_name, qlen):
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    cpu_infer = make_cpu_infer(2)
    act_type = TYPES[type_name][2]
    with torch.inference_mode():
        gate_q, gate = random_blocks(type_name, (expert_num, intermediate_size, hidden_size), gen)
        up_q, up = random_blocks(type_name, (expert_num, intermediate_size, hidden_size), gen)
        down_q, down = random_blocks(type_name, (expert_num, hidden_size, intermediate_size), gen)

        t = getattr(ggml_type, type_name)
        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.m_block = 32
        config.group_min_len = 10
        config.max_len = max_len
        config.group_max_len = max_len
        config.gate_proj = gate_q.data_ptr()
        config.up_proj = up_q.data_ptr()
        config.down_proj = down_q.data_ptr()
        config.gate_type = t
        config.up_type = t
        config.down_type = t
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

        ref = moe_torch(x.float(), expert_ids, weights, gate, up, down, act_type)
        rel = (torch.linalg.norm(out.float() - ref) / torch.linalg.norm(ref)).item()
        print(f"{type_name} qlen={qlen}: relative L2 {rel:.4f}")
        assert rel < REL_L2_LIMIT, f"{type_name} qlen={qlen}: relative L2 {rel:.4f} >= {REL_L2_LIMIT}"


if __name__ == "__main__":
    for name in TYPES:
        for q in (1, 32):
            test_llamafile_iquant_accuracy(name, q)
    print("ALL TESTS PASSED")
