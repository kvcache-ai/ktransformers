#!/usr/bin/env python3
"""
Reference comparison test for AVX2FP8_MOE_OFFLOAD.
Compares C++ offload MoE output against pure PyTorch FP8 dequant + matmul.
FP8 E4M3 is properly encoded/decoded using lookup tables matching the C++ kernel.
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import kt_kernel_ext

CPUINFER_PARAM = 304
CPUInfer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)

# ---- FP8 E4M3 decode lookup table (matches C++ fp8x8_to_fp32x8) ----
_FP8_DECODE_LUT = torch.zeros(256, dtype=torch.float32)
for _i in range(256):
    _s = (_i >> 7) & 1
    _e = (_i >> 3) & 0xF
    _m = _i & 0x7
    if _e == 15:
        _v = 0.0
    elif _e == 0:
        _v = (_m / 8.0) * (2.0 ** -6)
    else:
        _v = (1.0 + _m / 8.0) * (2.0 ** (_e - 7))
    if _s:
        _v = -_v
    _FP8_DECODE_LUT[_i] = _v


def fp8_e4m3_to_float32(fp8_uint8):
    return _FP8_DECODE_LUT[fp8_uint8.to(torch.int64)]


def float_to_fp8_e4m3(values):
    """Encode float values to FP8 E4M3 bytes by nearest-neighbor lookup."""
    flat = values.flatten()
    diffs = torch.abs(flat.unsqueeze(1) - _FP8_DECODE_LUT.unsqueeze(0))
    nearest = torch.argmin(diffs, dim=1)
    return nearest.to(torch.uint8).reshape(values.shape)


def fp8_quantize(tensor, group_size=128):
    fp8_max = 448.0
    rows, cols = tensor.shape
    n_blocks_n = (rows + group_size - 1) // group_size
    n_blocks_k = (cols + group_size - 1) // group_size
    fp8_weights = torch.zeros((rows, cols), dtype=torch.uint8)
    scales = torch.zeros((n_blocks_n, n_blocks_k), dtype=torch.float32)
    for bn in range(n_blocks_n):
        row_start = bn * group_size
        row_end = min(row_start + group_size, rows)
        for bk in range(n_blocks_k):
            col_start = bk * group_size
            col_end = min(col_start + group_size, cols)
            block = tensor[row_start:row_end, col_start:col_end].to(torch.float32)
            abs_max = block.abs().max().item()
            scale = abs_max / fp8_max if abs_max > 0 else 1.0
            scales[bn, bk] = scale
            scaled = (block / scale).round().clamp(-448, 447)
            fp8_block = float_to_fp8_e4m3(scaled)
            fp8_weights[row_start:row_end, col_start:col_end] = fp8_block
    return fp8_weights.contiguous(), scales.contiguous()


def fp8_dequant(fp8_weights, scales, group_size=128):
    raw_fp32 = fp8_e4m3_to_float32(fp8_weights)
    rows, cols = fp8_weights.shape
    n_blocks_n = (rows + group_size - 1) // group_size
    n_blocks_k = (cols + group_size - 1) // group_size
    out = torch.zeros((rows, cols), dtype=torch.float32)
    for bn in range(n_blocks_n):
        row_start = bn * group_size
        row_end = min(row_start + group_size, rows)
        for bk in range(n_blocks_k):
            col_start = bk * group_size
            col_end = min(col_start + group_size, cols)
            scale = scales[bn, bk]
            out[row_start:row_end, col_start:col_end] = (
                raw_fp32[row_start:row_end, col_start:col_end] * scale
            )
    return out


def test_reference():
    torch.manual_seed(42)
    expert_num = 2
    hidden_size = 128
    intermediate_size = 128
    num_experts_per_tok = 1
    qlen = 1
    group_size = 128

    gate_w = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    up_w = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16)
    down_w = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16)

    gate_fp8, gate_scales = fp8_quantize(gate_w[0], group_size)
    up_fp8, up_scales = fp8_quantize(up_w[0], group_size)
    down_fp8, down_scales = fp8_quantize(down_w[0], group_size)

    gate_projs = [[gate_fp8.data_ptr()]]
    up_projs = [[up_fp8.data_ptr()]]
    down_projs = [[down_fp8.data_ptr()]]
    gate_scales_2d = [[gate_scales.data_ptr()]]
    up_scales_2d = [[up_scales.data_ptr()]]
    down_scales_2d = [[down_scales.data_ptr()]]

    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.pool = CPUInfer.backend_
    config.max_len = 2048
    config.m_block = 1
    config.group_min_len = 10
    config.group_max_len = 1024
    config.layer_idx = 0

    quant_config = kt_kernel_ext.QuantConfig()
    quant_config.quant_method = "fp8"
    quant_config.group_size = group_size
    quant_config.zero_point = False
    quant_config.per_channel = False
    config.quant_config = quant_config

    config.gate_projs = gate_projs
    config.up_projs = up_projs
    config.down_projs = down_projs
    config.gate_scales = gate_scales_2d
    config.up_scales = up_scales_2d
    config.down_scales = down_scales_2d

    moe = kt_kernel_ext.moe.AVX2FP8_MOE_OFFLOAD(config)
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
    g_deq = fp8_dequant(gate_fp8, gate_scales, group_size)
    u_deq = fp8_dequant(up_fp8, up_scales, group_size)
    d_deq = fp8_dequant(down_fp8, down_scales, group_size)
    x = input_tensor[0].to(torch.float32)
    gate_out = g_deq @ x
    up_out = u_deq @ x
    hidden_act = torch.nn.functional.silu(gate_out) * up_out
    output_ref = d_deq @ hidden_act

    output_cpp_f32 = output_cpp.to(torch.float32)

    max_diff = (output_cpp_f32 - output_ref).abs().max().item()
    mean_diff = (output_cpp_f32 - output_ref).abs().mean().item()
    ref_max = output_ref.abs().max().item()

    print(f"Reference max abs: {ref_max:.6f}")
    print(f"Max abs diff:      {max_diff:.6f}")
    print(f"Mean abs diff:     {mean_diff:.6f}")
    print(f"cpp[:5]  = {output_cpp_f32.flatten()[:5].tolist()}")
    print(f"ref[:5]  = {output_ref.flatten()[:5].tolist()}")

    if max_diff < 50.0:
        print("\n>>> MATCH: output matches PyTorch reference within tolerance")
        return True
    else:
        print("\n>>> MISMATCH: output does NOT match reference")
        return False


if __name__ == "__main__":
    ok = test_reference()
    sys.exit(0 if ok else 1)
