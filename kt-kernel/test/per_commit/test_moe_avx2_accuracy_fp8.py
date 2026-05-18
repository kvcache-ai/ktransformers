#!/usr/bin/env python
# coding=utf-8
"""AVX2 FP8 MoE accuracy tests for KT-Kernel."""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from kt_kernel import kt_kernel_ext

expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 128
group_size = 128
validation_iter = 3
CPUINFER_PARAM = 60


def fp8_e4m3_quantize(tensor_bf16):
    """Quantize BF16 tensor to FP8 E4M3 with block-wise scales (128x128)."""
    n, k = tensor_bf16.shape
    tensor_fp32 = tensor_bf16.float()

    n_blocks_n = (n + group_size - 1) // group_size
    n_blocks_k = (k + group_size - 1) // group_size

    fp8_data = torch.zeros(n, k, dtype=torch.uint8)
    scales = torch.zeros(n_blocks_n, n_blocks_k, dtype=torch.float32)

    # FP8 E4M3 max value: 2^8 * (1 + 7/8) = 448
    fp8_max = 448.0

    for bn in range(n_blocks_n):
        for bk in range(n_blocks_k):
            n_start = bn * group_size
            n_end = min(n_start + group_size, n)
            k_start = bk * group_size
            k_end = min(k_start + group_size, k)

            block = tensor_fp32[n_start:n_end, k_start:k_end]
            amax = block.abs().max().item()
            if amax == 0:
                scale = 1.0
            else:
                scale = amax / fp8_max
            scales[bn, bk] = scale

            # Quantize
            for i in range(n_end - n_start):
                for j in range(k_end - k_start):
                    val = block[i, j].item() / scale
                    fp8_data[n_start + i, k_start + j] = float_to_fp8_e4m3(val)

    return fp8_data, scales


def float_to_fp8_e4m3(val):
    """Convert float to FP8 E4M3."""
    if math.isnan(val):
        return 0x7F
    sign = 1 if val < 0 else 0
    val = abs(val)
    if val == 0:
        return sign << 7
    # Clamp to max
    if val >= 448.0:
        return (sign << 7) | 0x7E  # max finite
    # Find exponent
    exp = int(math.floor(math.log2(val))) + 7
    if exp <= 0:
        # Subnormal
        man = int(round(val * (2**6) * 8))
        man = min(man, 7)
        return (sign << 7) | man
    if exp >= 15:
        return (sign << 7) | 0x7E  # clamp to max
    # Normal
    man = int(round((val / (2 ** (exp - 7)) - 1.0) * 8))
    man = min(man, 7)
    return (sign << 7) | (exp << 3) | man


def fp8_e4m3_to_float(byte_val):
    """Convert FP8 E4M3 byte to float."""
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0xF
    man = byte_val & 0x7
    if exp == 0 and man == 0:
        return 0.0
    if exp == 0:
        val = (2**-6) * (man / 8.0)
    elif exp == 15:
        return float("nan")
    else:
        val = (2 ** (exp - 7)) * (1.0 + man / 8.0)
    return -val if sign else val


def fp8_dequantize(fp8_data, scales):
    """Dequantize FP8 + scales back to float32."""
    n, k = fp8_data.shape
    result = torch.zeros(n, k, dtype=torch.float32)
    n_blocks_n = scales.shape[0]
    n_blocks_k = scales.shape[1]

    for i in range(n):
        for j in range(k):
            bn = i // group_size
            bk = j // group_size
            scale = scales[bn, bk].item()
            fp8_val = fp8_e4m3_to_float(fp8_data[i, j].item())
            result[i, j] = fp8_val * scale
    return result


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]
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


def test_avx2_fp8_accuracy(qlen, label):
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

    with torch.inference_mode():
        # Generate BF16 weights, quantize to FP8
        gate_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        up_bf16 = (torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )
        down_bf16 = (torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32) / 10.0).to(
            torch.bfloat16
        )

        # Quantize each expert
        gate_fp8_list, gate_scale_list = [], []
        up_fp8_list, up_scale_list = [], []
        down_fp8_list, down_scale_list = [], []

        for e in range(expert_num):
            gf, gs = fp8_e4m3_quantize(gate_bf16[e])
            gate_fp8_list.append(gf)
            gate_scale_list.append(gs)
            uf, us = fp8_e4m3_quantize(up_bf16[e])
            up_fp8_list.append(uf)
            up_scale_list.append(us)
            df, ds = fp8_e4m3_quantize(down_bf16[e])
            down_fp8_list.append(df)
            down_scale_list.append(ds)

        # Stack into contiguous tensors
        gate_fp8 = torch.stack(gate_fp8_list).contiguous()
        gate_scales = torch.stack(gate_scale_list).contiguous()
        up_fp8 = torch.stack(up_fp8_list).contiguous()
        up_scales = torch.stack(up_scale_list).contiguous()
        down_fp8 = torch.stack(down_fp8_list).contiguous()
        down_scales = torch.stack(down_scale_list).contiguous()

        # Dequantize for reference computation
        gate_deq = torch.stack([fp8_dequantize(gate_fp8_list[e], gate_scale_list[e]) for e in range(expert_num)])
        up_deq = torch.stack([fp8_dequantize(up_fp8_list[e], up_scale_list[e]) for e in range(expert_num)])
        down_deq = torch.stack([fp8_dequantize(down_fp8_list[e], down_scale_list[e]) for e in range(expert_num)])

        # Create MOE config
        config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
        config.max_len = max_len
        config.gate_proj = gate_fp8.data_ptr()
        config.up_proj = up_fp8.data_ptr()
        config.down_proj = down_fp8.data_ptr()
        config.gate_scale = gate_scales.data_ptr()
        config.up_scale = up_scales.data_ptr()
        config.down_scale = down_scales.data_ptr()
        config.quant_config.bits = 8
        config.quant_config.group_size = group_size
        config.quant_config.zero_point = False
        config.pool = CPUInfer.backend_

        moe = kt_kernel_ext.moe.AVX2FP8_MOE(config)
        CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        CPUInfer.sync()

        print("\n--- %s (qlen=%d) ---" % (label, qlen))
        for i in range(validation_iter):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32) / 100.0).to(torch.bfloat16).contiguous()
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

            # Reference: use dequantized FP32 weights
            t_output = moe_torch(input_data.float(), expert_ids, weights, gate_deq, up_deq, down_deq).to(torch.bfloat16)

            diff = torch.mean(torch.abs(output.float() - t_output.float())) / (
                torch.mean(torch.abs(t_output.float())) + 1e-8
            )
            print("  Iteration %d: diff = %.6f" % (i, diff.item()))
            assert diff < 0.1, "FP8 accuracy test failed: diff=%.6f >= 0.1" % diff.item()

    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("AVX2 FP8 MoE Accuracy Test")
    print("=" * 60)
    try:
        test_avx2_fp8_accuracy(qlen=1, label="Decode")
        test_avx2_fp8_accuracy(qlen=16, label="Prefill")
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print("\nTEST FAILED: %s" % e)
        import traceback

        traceback.print_exc()
        sys.exit(1)
