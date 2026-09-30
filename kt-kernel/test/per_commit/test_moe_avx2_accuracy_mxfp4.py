#!/usr/bin/env python
# coding=utf-8
"""AVX2 MXFP4 / NVFP4-layout MoE accuracy tests for KT-Kernel.

hidden_size is 4096 on purpose: gemm_mxfp4 walks the activation rows in tiles of
GemmKernelAVX2MXFP4::M_TILE_BYTES / (k * 4) = 16 rows at k = 4096, so the
concentrated-routing cases below (every token hits the same experts, per-expert
m == qlen) cover one full tile, a tile plus a one-row remainder, and several
tiles ending in a partial 4-token block.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch
from kt_kernel import kt_kernel_ext

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="default")

expert_num = 4
hidden_size = 4096
intermediate_size = 256
num_experts_per_tok = 2
max_len = 128
validation_iter = 2
CPUINFER_PARAM = 8

# OCP MXFP4 (E2M1) codepoints, same order as the kernel's LUT.
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def quantize_fp4(weights, group_size):
    """[E, N, K] bf16 -> packed nibbles as int32 [E, N, K/8], bf16 scales [E, N, K/gs], bf16 dequant."""
    packed_list, scale_list, dequant_list = [], [], []
    for e in range(weights.shape[0]):
        w = weights[e].float()
        rows, cols = w.shape
        grouped = w.view(rows, cols // group_size, group_size)
        scales = (grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 6.0).squeeze(-1)
        normalized = grouped / scales.unsqueeze(-1)
        idx = (normalized.unsqueeze(-1) - E2M1_VALUES.view(1, 1, 1, 16)).abs().argmin(dim=-1)
        dequant_list.append((E2M1_VALUES[idx] * scales.unsqueeze(-1)).view(rows, cols).to(torch.bfloat16))
        nibbles = idx.to(torch.uint8).view(rows, cols // 2, 2)
        packed = ((nibbles[..., 1] << 4) | nibbles[..., 0]).view(rows, cols // 8, 4)
        packed_list.append(
            packed[..., 0].to(torch.int32)
            | (packed[..., 1].to(torch.int32) << 8)
            | (packed[..., 2].to(torch.int32) << 16)
            | (packed[..., 3].to(torch.int32) << 24)
        )
        scale_list.append(scales.to(torch.bfloat16))
    return (
        torch.stack(packed_list).contiguous(),
        torch.stack(scale_list).contiguous(),
        torch.stack(dequant_list),
    )


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    output = torch.zeros((input.shape[0], down_proj.shape[1]), dtype=torch.float32)
    x = input.float()
    for e in range(expert_num):
        tokens, slots = (expert_ids == e).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        h = x[tokens]
        inter = act_fn(h @ gate_proj[e].float().t()) * (h @ up_proj[e].float().t())
        output.index_add_(0, tokens, (inter @ down_proj[e].float().t()) * weights[tokens, slots].unsqueeze(-1))
    return output


CASES = [
    # (qlen, routing, label)
    (1, "balanced", "decode"),
    (16, "concentrated", "prefill, one full row tile"),
    (17, "concentrated", "prefill, tile + single-row remainder"),
    (70, "concentrated", "prefill, 4 tiles + partial 4-token block"),
    (64, "balanced", "prefill, mixed per-expert m"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("group_size", [32, 16])
@pytest.mark.parametrize("qlen,routing,label", CASES)
def test_avx2_mxfp4_accuracy(group_size, qlen, routing, label):
    torch.manual_seed(42)
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    cpu_infer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        gate_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size)) / 10.0).to(torch.bfloat16)
        up_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size)) / 10.0).to(torch.bfloat16)
        down_bf16 = (torch.randn((expert_num, hidden_size, intermediate_size)) / 10.0).to(torch.bfloat16)
        gate_q, gate_s, gate_deq = quantize_fp4(gate_bf16, group_size)
        up_q, up_s, up_deq = quantize_fp4(up_bf16, group_size)
        down_q, down_s, down_deq = quantize_fp4(down_bf16, group_size)

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_q.data_ptr()
        config.up_proj = up_q.data_ptr()
        config.down_proj = down_q.data_ptr()
        config.gate_scale = gate_s.data_ptr()
        config.up_scale = up_s.data_ptr()
        config.down_scale = down_s.data_ptr()
        config.quant_config.bits = 4
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
        config.pool = cpu_infer.backend_

        moe = kt_kernel_ext.moe.AVX2MXFP4_MOE(config)
        cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpu_infer.sync()

        print(f"\n--- group_size={group_size} qlen={qlen} {routing}: {label} ---")
        for i in range(validation_iter):
            if routing == "concentrated":
                expert_ids = torch.randperm(expert_num)[:num_experts_per_tok].unsqueeze(0).expand(qlen, -1).contiguous()
            else:
                expert_ids = torch.stack(
                    [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
                ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size)) / 100.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
            cpu_infer.submit(
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
            cpu_infer.sync()

            t_output = moe_torch(input_data, expert_ids, weights, gate_deq, up_deq, down_deq)
            diff = torch.mean(torch.abs(output.float() - t_output)) / (torch.mean(torch.abs(t_output)) + 1e-8)
            print(f"  Iteration {i}: diff = {diff.item():.6f}")
            assert diff < 0.02, f"AVX2 MXFP4 accuracy test failed: diff={diff.item():.6f} >= 0.02"

    print("  PASSED")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
