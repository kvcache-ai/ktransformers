import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
from kt_kernel import kt_kernel_ext

torch.manual_seed(42)

hidden_size = 7168
intermediate_size = 2048
max_len = 25600

expert_num = 16
num_experts_per_tok = 8

layer_num = 1
CPUInfer = kt_kernel_ext.CPUInfer(40)
validation_iter = 3
k_group_size = 32
debug_print_count = 16

QLEN_LIST = [1, 32]
DISPATCH_THRESHOLD = 4 * expert_num / num_experts_per_tok

physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

E2M1_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


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
        expert_out = mlp_torch(sorted_tokens[start_idx:end_idx], gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def quantize_mxfp4_tensor(weights: torch.Tensor, group_size: int):
    weights_f32 = weights.to(torch.float32)
    e, rows, cols = weights_f32.shape
    reshaped = weights_f32.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-8)
    scales = (max_abs / 6.0).squeeze(-1)
    normalized = reshaped / scales.unsqueeze(-1)
    e2m1_vals = E2M1_VALUES.view(1, 1, 1, 1, 16)
    normalized_expanded = normalized.unsqueeze(-1)
    distances = torch.abs(normalized_expanded - e2m1_vals)
    closest_indices = distances.argmin(dim=-1)
    dequant = E2M1_VALUES[closest_indices].to(torch.float32) * scales.unsqueeze(-1)
    dequant = dequant.view(e, rows, cols)
    nibbles = closest_indices.to(torch.uint8)
    nibbles = nibbles.view(e, rows, cols // 2, 2)
    lo = nibbles[..., 0]
    hi = nibbles[..., 1]
    packed_bytes = (hi << 4) | lo
    bytes_view = packed_bytes.view(e, rows, cols // 8, 4)
    packed_int32 = (
        bytes_view[..., 0].to(torch.int32) |
        (bytes_view[..., 1].to(torch.int32) << 8) |
        (bytes_view[..., 2].to(torch.int32) << 16) |
        (bytes_view[..., 3].to(torch.int32) << 24)
    )
    packed_int32 = packed_int32.view(e, rows, cols // 8).contiguous()
    scales = scales.to(torch.bfloat16).contiguous().view(e, rows, cols // group_size).contiguous()
    return packed_int32, scales, dequant


WEIGHT_PATTERNS = {
    "uniform_scale":     ("All k-groups share the same abs max / scale",   lambda g: torch.full((g,), 0.02, dtype=torch.float32)),
    "alternating_scale": ("Alternate small / large abs max per k-group",   lambda g: torch.where(torch.arange(g) % 2 == 0, torch.full((g,), 0.015), torch.full((g,), 0.03))),
    "ramp_scale":        ("Linearly increasing abs max per k-group",       lambda g: torch.linspace(0.005, 0.04, steps=g, dtype=torch.float32)),
    "random":            ("Random bf16 weights (baseline)",                None),
}


def build_structured_tensor(shape, pattern):
    if pattern == "random":
        torch.manual_seed(42)
        return (torch.randn(shape, dtype=torch.bfloat16) / 100.0).contiguous()
    e, rows, cols = shape
    groups = cols // k_group_size
    group_vals = WEIGHT_PATTERNS[pattern][1](groups).to(torch.float32)
    block = group_vals.view(1, 1, groups, 1).expand(e, rows, groups, k_group_size).clone()
    row_signs = torch.where(
        (torch.arange(rows) % 2 == 0),
        torch.ones(rows, dtype=torch.float32),
        -torch.ones(rows, dtype=torch.float32),
    ).view(1, rows, 1, 1)
    col_offsets = torch.linspace(-0.0005, 0.0005, steps=k_group_size, dtype=torch.float32).view(1, 1, 1, k_group_size)
    block = block * row_signs + col_offsets
    return block.reshape(shape).to(torch.bfloat16).contiguous()


def prepare_weights(pattern):
    gate_proj = build_structured_tensor((expert_num, intermediate_size, hidden_size), pattern)
    up_proj   = build_structured_tensor((expert_num, intermediate_size, hidden_size), pattern)
    down_proj = build_structured_tensor((expert_num, hidden_size, intermediate_size), pattern)
    gate_q, gate_s, gate_dq = quantize_mxfp4_tensor(gate_proj, k_group_size)
    up_q,   up_s,   up_dq   = quantize_mxfp4_tensor(up_proj,   k_group_size)
    down_q, down_s, down_dq = quantize_mxfp4_tensor(down_proj, k_group_size)
    return {
        "gate_qweight": gate_q.contiguous(), "up_qweight": up_q.contiguous(), "down_qweight": down_q.contiguous(),
        "gate_scales":  gate_s.contiguous(), "up_scales":  up_s.contiguous(), "down_scales":  down_s.contiguous(),
        "dequantized":  {"gate_proj": gate_dq.to(torch.bfloat16), "up_proj": up_dq.to(torch.bfloat16), "down_proj": down_dq.to(torch.bfloat16)},
    }


def build_moes(quant_data):
    AVX2MXFP4_MOE = getattr(kt_kernel_ext.moe, "AVX2MXFP4_MOE", None)
    if AVX2MXFP4_MOE is None:
        raise RuntimeError("AVX2MXFP4_MOE not found — rebuild kt-kernel with the AVX2 MXFP4 path")
    moes = []
    with torch.inference_mode(mode=True):
        for _ in range(layer_num):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.quant_config.bits = 4
            config.quant_config.group_size = k_group_size
            config.quant_config.zero_point = False
            config.gate_proj  = quant_data["gate_qweight"].data_ptr()
            config.up_proj    = quant_data["up_qweight"].data_ptr()
            config.down_proj  = quant_data["down_qweight"].data_ptr()
            config.gate_scale = quant_data["gate_scales"].data_ptr()
            config.up_scale   = quant_data["up_scales"].data_ptr()
            config.down_scale = quant_data["down_scales"].data_ptr()
            config.pool = CPUInfer.backend_
            moe = AVX2MXFP4_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            moes.append(moe)
    return moes


def run_case(pattern, qlen):
    print("\n" + "=" * 70)
    desc = WEIGHT_PATTERNS[pattern][0]
    path = "mat-vec" if qlen <= DISPATCH_THRESHOLD else "mat-mat"
    print(f"Running case: {pattern} -> {desc}  (qlen={qlen}, path={path})")
    print("=" * 70)

    quant_data = prepare_weights(pattern)
    moes = build_moes(quant_data)
    dq = quant_data["dequantized"]

    diffs = []
    with torch.inference_mode(mode=True):
        for i in range(validation_iter):
            torch.manual_seed(100 + i)
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.randn((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_tensor = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

            moe = moes[i % layer_num]
            CPUInfer.submit(moe.forward_task(
                bsz_tensor.data_ptr(), num_experts_per_tok,
                expert_ids.data_ptr(), weights.data_ptr(),
                input_tensor.data_ptr(), output.data_ptr(), False,
            ))
            CPUInfer.sync()

            t_output = moe_torch(input_tensor.to(torch.bfloat16), expert_ids, weights,
                                 dq["gate_proj"], dq["up_proj"], dq["down_proj"]).to(torch.bfloat16)

            diff = torch.mean(torch.abs(output.float() - t_output.float())) / (
                torch.mean(torch.abs(t_output.float())) + 1e-12)
            diffs.append(diff.item())
            print(f"[{pattern}] iter {i}: rel-L1 = {diff:.4f}")
            print(f"  output   {output.flatten()[:debug_print_count]}")
            print(f"  t_output {t_output.flatten()[:debug_print_count]}")

    return {"case": pattern, "description": desc,
            "mean": sum(diffs)/len(diffs), "max": max(diffs), "min": min(diffs)}


def run_fp4_moe_avx2_test():
    summary = []
    for qlen in QLEN_LIST:
        path = "mat-vec" if qlen <= DISPATCH_THRESHOLD else "mat-mat"
        print(f"\n##### qlen={qlen}  path={path} #####")
        for pattern in WEIGHT_PATTERNS:
            r = run_case(pattern, qlen)
            r.update({"qlen": qlen, "path": path})
            summary.append(r)

    print("\n=== AVX2 MXFP4 — Relative Error Summary ===")
    print(f"{'Case':<20} {'qlen':>5} {'path':<8} {'Mean':>10} {'Max':>10} {'Min':>10}")
    for r in summary:
        print(f"{r['case']:<20} {r['qlen']:>5} {r['path']:<8} "
              f"{r['mean']*100:9.2f}% {r['max']*100:9.2f}% {r['min']*100:9.2f}%")


if __name__ == "__main__":
    run_fp4_moe_avx2_test()
