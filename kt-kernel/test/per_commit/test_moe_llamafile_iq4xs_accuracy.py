#!/usr/bin/env python
# coding=utf-8
"""Llamafile (GGUF) MoE accuracy test for IQ4_XS experts.

The experts are quantized to IQ4_XS with ggml's own quantizer and the reference
is a float32 MoE on the dequantized weights whose activations go through the
same Q8_K rounding the kernel applies, so what is left is the kernel's own error
plus the bf16 output. The unsigned AVX2 IQ4_XS kernel saturated int16 in
maddubs and landed at 3-9% relative L2 here.
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
validation_iter = 3
CPUINFER_PARAM = 16
REL_L2_LIMIT = 0.01

ggml_type = kt_kernel_ext.kvcache.ggml_type


def quantize(w):
    """float32 [experts, rows, cols] -> (IQ4_XS bytes, dequantized float32 of the same shape)."""
    flat = w.contiguous().view(-1)
    q = kt_kernel_ext.utils.from_float(flat.data_ptr(), flat.numel(), ggml_type.IQ4_XS)
    deq = kt_kernel_ext.utils.to_float(q.data_ptr(), flat.numel(), ggml_type.IQ4_XS)
    return q, deq.view_as(w)


def q8k(x):
    """Round a float32 vector the way the kernel quantizes its activations (Q8_K)."""
    x = x.contiguous()
    q = kt_kernel_ext.utils.from_float(x.data_ptr(), x.numel(), ggml_type.Q8_K)
    return kt_kernel_ext.utils.to_float(q.data_ptr(), x.numel(), ggml_type.Q8_K)


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def moe_torch(x, expert_ids, weights, gate, up, down):
    out = torch.zeros(x.shape[0], hidden_size, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(expert_ids.shape[1]):
            e = int(expert_ids[t, j])
            xq = q8k(x[t])
            h = act_fn(gate[e] @ xq) * (up[e] @ xq)
            out[t] += weights[t, j] * (down[e] @ q8k(h))
    return out


@pytest.mark.cpu
@pytest.mark.parametrize("qlen,label", [(1, "Decode"), (32, "Prefill")])
def test_llamafile_iq4xs_accuracy(qlen, label):
    torch.manual_seed(0)
    CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
    physical_to_logical_map = torch.arange(expert_num, dtype=torch.int64).contiguous()

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
        config.gate_type = ggml_type.IQ4_XS
        config.up_type = ggml_type.IQ4_XS
        config.down_type = ggml_type.IQ4_XS
        config.hidden_type = ggml_type.BF16
        config.pool = CPUInfer.backend_

        moe = kt_kernel_ext.moe.MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        print(f"\n--- {label} (qlen={qlen}) ---")
        for i in range(validation_iter):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = torch.randn((qlen, hidden_size), dtype=torch.float32).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)

            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            ref = moe_torch(input_data.float(), expert_ids, weights, gate, up, down)
            rel = (torch.linalg.norm(output.float() - ref) / torch.linalg.norm(ref)).item()
            print(f"  Iteration {i}: relative L2 = {rel:.4f}")
            assert rel < REL_L2_LIMIT, f"IQ4_XS llamafile MoE: relative L2 {rel:.4f} >= {REL_L2_LIMIT}"

    print("  PASSED")


if __name__ == "__main__":
    test_llamafile_iq4xs_accuracy(1, "Decode")
    test_llamafile_iq4xs_accuracy(32, "Prefill")
    print("ALL TESTS PASSED")
