#!/usr/bin/env python
# coding=utf-8
"""GPTQ INT4 MoE accuracy tests for KT-Kernel x86 backends."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch
import kt_kernel_ext

expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 128
group_size = 128
validation_iter = 3
CPUINFER_PARAM = 16


def gptq_sym_int4_quantize(weight_bf16):
    """Quantize [N, K] BF16 weight to GPTQ symmetric int4 layout."""
    n, k = weight_bf16.shape
    assert k % 8 == 0
    assert k % group_size == 0

    weight_fp32 = weight_bf16.float()
    qweight = torch.zeros((k // 8, n), dtype=torch.int32)
    scales = torch.zeros((k // group_size, n), dtype=torch.float32)

    for ni in range(n):
        for g in range(k // group_size):
            k_start = g * group_size
            k_end = k_start + group_size
            block = weight_fp32[ni, k_start:k_end]
            amax = block.abs().max().item()
            scale = amax / 7.0 if amax > 0 else 1.0
            scales[g, ni] = scale

            for kk in range(k_start, k_end, 8):
                packed = 0
                for nib in range(8):
                    q = int(round(weight_fp32[ni, kk + nib].item() / scale)) + 8
                    q = max(0, min(15, q))
                    packed |= q << (nib * 4)
                if packed >= 2**31:
                    packed -= 2**32
                qweight[kk // 8, ni] = packed

    return qweight, scales


def gptq_sym_int4_dequantize(qweight, scales, out_features, in_features):
    """Dequantize GPTQ qweight/scales back to fp32 [N, K]."""
    result = torch.zeros((out_features, in_features), dtype=torch.float32)
    for ni in range(out_features):
        for g in range(in_features // group_size):
            scale = scales[g, ni].item()
            k_start = g * group_size
            k_end = k_start + group_size
            for kk in range(k_start, k_end, 8):
                packed = int(qweight[kk // 8, ni].item())
                for nib in range(8):
                    result[ni, kk + nib] = (((packed >> (nib * 4)) & 0xF) - 8) * scale
    return result


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_data, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_data, gate_proj.t())
    up_buf = torch.mm(input_data, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input_data, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_data[idxs // expert_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens = sorted_tokens[start_idx:end_idx]
        out = mlp_torch(tokens, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*expert_ids.shape, -1).float().mul_(weights.unsqueeze(-1)).sum(1)).to(new_x.dtype)


def available_backends():
    backends = []
    if hasattr(kt_kernel_ext.moe, "AVX2GPTQInt4_MOE"):
        backends.append(("AVX2GPTQInt4_MOE", kt_kernel_ext.moe.AVX2GPTQInt4_MOE, 0.12))

    if hasattr(kt_kernel_ext.moe, "AVXVNNI256GPTQInt4_MOE"):
        has_avx_vnni = False
        try:
            with open("/proc/cpuinfo", "r") as f:
                has_avx_vnni = any(("avx_vnni" in line or "avxvnni" in line) for line in f if line.startswith("flags"))
        except OSError:
            has_avx_vnni = False
        if has_avx_vnni:
            backends.append(("AVXVNNI256GPTQInt4_MOE", kt_kernel_ext.moe.AVXVNNI256GPTQInt4_MOE, 0.20))
    return backends


def run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen):
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    cpu_infer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        gate_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        up_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        down_bf16 = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )

        gate_qw_list, gate_scale_list = [], []
        up_qw_list, up_scale_list = [], []
        down_qw_list, down_scale_list = [], []

        for e in range(expert_num):
            qw, sc = gptq_sym_int4_quantize(gate_bf16[e])
            gate_qw_list.append(qw)
            gate_scale_list.append(sc)

            qw, sc = gptq_sym_int4_quantize(up_bf16[e])
            up_qw_list.append(qw)
            up_scale_list.append(sc)

            qw, sc = gptq_sym_int4_quantize(down_bf16[e])
            down_qw_list.append(qw)
            down_scale_list.append(sc)

        gate_qw = torch.stack(gate_qw_list).contiguous()
        gate_scales = torch.stack(gate_scale_list).contiguous()
        up_qw = torch.stack(up_qw_list).contiguous()
        up_scales = torch.stack(up_scale_list).contiguous()
        down_qw = torch.stack(down_qw_list).contiguous()
        down_scales = torch.stack(down_scale_list).contiguous()

        gate_deq = torch.stack(
            [
                gptq_sym_int4_dequantize(gate_qw_list[e], gate_scale_list[e], intermediate_size, hidden_size)
                for e in range(expert_num)
            ]
        )
        up_deq = torch.stack(
            [
                gptq_sym_int4_dequantize(up_qw_list[e], up_scale_list[e], intermediate_size, hidden_size)
                for e in range(expert_num)
            ]
        )
        down_deq = torch.stack(
            [
                gptq_sym_int4_dequantize(down_qw_list[e], down_scale_list[e], hidden_size, intermediate_size)
                for e in range(expert_num)
            ]
        )

        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_qw.data_ptr()
        config.up_proj = up_qw.data_ptr()
        config.down_proj = down_qw.data_ptr()
        config.gate_scale = gate_scales.data_ptr()
        config.up_scale = up_scales.data_ptr()
        config.down_scale = down_scales.data_ptr()
        config.quant_config.bits = 4
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
        config.pool = cpu_infer.backend_

        moe = backend_cls(config)
        cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpu_infer.sync()

        print(f"\n--- {backend_name} (qlen={qlen}) ---")
        for i in range(validation_iter):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
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

            ref_output = moe_torch(input_data.float(), expert_ids, weights, gate_deq, up_deq, down_deq).to(
                torch.bfloat16
            )
            diff = torch.mean(torch.abs(output.float() - ref_output.float())) / (
                torch.mean(torch.abs(ref_output.float())) + 1e-8
            )
            print(f"  Iteration {i}: diff = {diff.item():.6f}")
            assert diff < threshold, f"{backend_name} accuracy test failed: diff={diff.item():.6f} >= {threshold}"


def test_gptq_int4_accuracy():
    backends = available_backends()
    if not backends:
        print("Skipping GPTQ INT4 accuracy tests: no x86 GPTQ backend available")
        return

    for backend_name, backend_cls, threshold in backends:
        run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=1)
        run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=16)


if __name__ == "__main__":
    print("=" * 60)
    print("GPTQ INT4 MoE Accuracy Test")
    print("=" * 60)
    test_gptq_int4_accuracy()
    print("PASSED")
