#!/usr/bin/env python3
"""
Reference comparison test for AVX2RAWINT4_MOE_OFFLOAD.
Compares C++ offload MoE output against pure PyTorch INT4 dequant + matmul.
RAWINT4: 4-bit packed weights, per K-group scales, value = (nibble - 8) * scale.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import kt_kernel_ext

CPUINFER_PARAM = 304
CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)


def rawint4_quantize(tensor, k_group_size=32):
    """Quantize BF16/FP32 tensor to RAWINT4 with per K-group scales.
    Stores 2 nibbles per byte. Value = (nibble - 8) * scale.
    """
    rows, cols = tensor.shape
    k_groups = cols // k_group_size
    assert cols % k_group_size == 0, f"cols {cols} not divisible by k_group_size {k_group_size}"

    packed = torch.zeros((rows, cols // 2), dtype=torch.uint8)
    scales = torch.zeros((rows, k_groups), dtype=torch.float32)

    for r in range(rows):
        for g in range(k_groups):
            k_start = g * k_group_size
            k_end = k_start + k_group_size
            block = tensor[r, k_start:k_end].to(torch.float32)
            abs_max = block.abs().max().item()
            scale = abs_max / 7.0 if abs_max > 0 else 1.0  # max |nibble-8| = 8, but use 7 for headroom
            scales[r, g] = scale

            # Quantize: nibble = round(v / scale) + 8, clamped to [0, 15]
            quantized = ((block / scale).round() + 8).clamp(0, 15).to(torch.uint8)

            # Pack two nibbles per byte
            for c_in_group in range(0, k_group_size, 2):
                c = k_start + c_in_group
                nibble_lo = quantized[c_in_group].item()
                nibble_hi = quantized[c_in_group + 1].item() if c_in_group + 1 < k_group_size else 0
                packed[r, c // 2] = (nibble_hi << 4) | nibble_lo

    return packed.contiguous(), scales.contiguous()


def rawint4_dequant(packed, scales, k_group_size=32):
    """Dequantize RAWINT4 weights to FP32."""
    rows, packed_cols = packed.shape
    cols = packed_cols * 2
    k_groups = cols // k_group_size
    out = torch.zeros((rows, cols), dtype=torch.float32)

    for r in range(rows):
        for g in range(k_groups):
            scale = scales[r, g].item()
            k_start = g * k_group_size
            for c_in_group in range(k_group_size):
                c = k_start + c_in_group
                byte_idx = c // 2
                is_lo = (c % 2 == 0)
                byte_val = packed[r, byte_idx].item()
                nibble = byte_val & 0x0F if is_lo else (byte_val >> 4) & 0x0F
                out[r, c] = (nibble - 8) * scale
    return out


def test_rawint4_reference():
    torch.manual_seed(42)
    expert_num = 2
    hidden_size = 64
    intermediate_size = 64
    num_experts_per_tok = 1
    qlen = 1
    k_group_size = 32

    gate_w = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up_w = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down_w = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16)

    gate_fp8, gate_scales = [], []
    up_fp8, up_scales = [], []
    down_fp8, down_scales = [], []
    for e in range(expert_num):
        g, gs = rawint4_quantize(gate_w[e], k_group_size)
        u, us = rawint4_quantize(up_w[e], k_group_size)
        d, ds = rawint4_quantize(down_w[e], k_group_size)
        gate_fp8.append(g); gate_scales.append(gs)
        up_fp8.append(u); up_scales.append(us)
        down_fp8.append(d); down_scales.append(ds)

    gate_projs = [[w.data_ptr() for w in gate_fp8]]
    up_projs = [[w.data_ptr() for w in up_fp8]]
    down_projs = [[w.data_ptr() for w in down_fp8]]
    gate_scales_2d = [[s.data_ptr() for s in gate_scales]]
    up_scales_2d = [[s.data_ptr() for s in up_scales]]
    down_scales_2d = [[s.data_ptr() for s in down_scales]]

    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.pool = CPUInfer.backend_
    config.max_len = 2048
    config.m_block = 1
    config.group_min_len = 10
    config.group_max_len = 1024
    config.layer_idx = 0

    quant_config = kt_kernel_ext.QuantConfig()
    quant_config.quant_method = "rawint4"
    quant_config.group_size = k_group_size
    quant_config.zero_point = False
    quant_config.per_channel = False
    config.quant_config = quant_config

    config.gate_projs = gate_projs
    config.up_projs = up_projs
    config.down_projs = down_projs
    config.gate_scales = gate_scales_2d
    config.up_scales = up_scales_2d
    config.down_scales = down_scales_2d

    moe = kt_kernel_ext.moe.AVX2RawInt4_MOE_OFFLOAD(config)
    CPUInfer.submit(moe.load_weights_task())
    CPUInfer.sync()

    expert_ids = torch.tensor([0], dtype=torch.int64).contiguous()
    router_weights = torch.tensor([[1.0]], dtype=torch.float32).contiguous()
    input_tensor = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    output_cpp = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

    CPUInfer.submit(moe.forward_task(qlen_tensor.data_ptr(), num_experts_per_tok,
        expert_ids.data_ptr(), router_weights.data_ptr(),
        input_tensor.data_ptr(), output_cpp.data_ptr(), False))
    CPUInfer.sync()

    # PyTorch reference
    e = 0  # expert 0
    g_deq = rawint4_dequant(gate_fp8[e], gate_scales[e], k_group_size)
    u_deq = rawint4_dequant(up_fp8[e], up_scales[e], k_group_size)
    d_deq = rawint4_dequant(down_fp8[e], down_scales[e], k_group_size)
    x = input_tensor[0].to(torch.float32)
    gate_out = g_deq @ x
    up_out = u_deq @ x
    hidden_act = torch.nn.functional.silu(gate_out) * up_out
    output_ref = d_deq @ hidden_act

    output_cpp_f32 = output_cpp.to(torch.float32)

    max_diff = (output_cpp_f32 - output_ref).abs().max().item()
    mean_diff = (output_cpp_f32 - output_ref).abs().mean().item()
    ref_max = output_ref.abs().max().item()

    print(f"RAWINT4 Reference max abs: {ref_max:.6f}")
    print(f"RAWINT4 Max abs diff:      {max_diff:.6f}")
    print(f"RAWINT4 Mean abs diff:     {mean_diff:.6f}")
    print(f"RAWINT4 cpp[:5]  = {output_cpp_f32.flatten()[:5].tolist()}")
    print(f"RAWINT4 ref[:5]  = {output_ref.flatten()[:5].tolist()}")

    if max_diff < 50.0:
        print("\n>>> RAWINT4 MATCH: output matches PyTorch reference within tolerance")
        return True
    else:
        print("\n>>> RAWINT4 MISMATCH: output does NOT match reference")
        return False


if __name__ == "__main__":
    ok = test_rawint4_reference()
    sys.exit(0 if ok else 1)
