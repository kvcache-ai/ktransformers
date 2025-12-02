import math
import os
import sys
from typing import Dict, Literal

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import torch
import kt_kernel_ext

torch.manual_seed(42)

hidden_size = 7168
intermediate_size = 2048
max_len = 25600

expert_num = 16
num_experts_per_tok = 8

qlen = 1
layer_num = 1
CPUInfer = kt_kernel_ext.CPUInfer(40)
validation_iter = 10
k_group_size = 32
debug_print_count = 16

physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()


def _pattern_uniform(groups: int) -> torch.Tensor:
    return torch.full((groups,), 0.02, dtype=torch.float32)


def _pattern_alternating(groups: int) -> torch.Tensor:
    vals = torch.full((groups,), 0.015, dtype=torch.float32)
    vals[1::2] = 0.03
    return vals


def _pattern_ramp(groups: int) -> torch.Tensor:
    return torch.linspace(0.005, 0.04, steps=groups, dtype=torch.float32)


WEIGHT_PATTERNS = {
    "uniform_scale": ("All k-groups share the same abs max / scale", _pattern_uniform),
    "alternating_scale": ("Alternate small / large abs max per k-group", _pattern_alternating),
    "ramp_scale": ("Linearly increasing abs max per k-group", _pattern_ramp),
    "random": ("Random bf16 weights (baseline)", None),
}


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    print(f"gate_buf: {gate_buf}")
    print(f"up_buf: {up_buf}")
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    print(f"intermediate: {intermediate}")
    print(f"mlp output: {ret}")
    return ret


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
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return t_output


def pack_to_int32(value: torch.Tensor, num_bits: int, packed_dim: Literal[0, 1] = 1) -> torch.Tensor:
    if value.dtype is not torch.int8:
        raise ValueError("Tensor must be torch.int8 before packing")
    if not (1 <= num_bits <= 8):
        raise ValueError(f"num_bits must be in [1, 8], got {num_bits}")

    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed

def pack_tensor_per_row(q: torch.Tensor, num_bits: int) -> torch.Tensor:
    e, rows, cols = q.shape
    flat = q.view(e * rows, cols)
    packed = pack_to_int32(flat, num_bits)
    return packed.view(e, rows, -1).contiguous()


def quantize_k2_tensor(weights: torch.Tensor, group_size: int):
    """
    Symmetric max-abs/7 quantization per k-group following compressed_tensors packing.
    Args:
        weights: [expert_num, rows (N), cols (K)]
    Returns:
        packed_q: int32 tensor storing 8 int4s per element with shape [expert_num, rows * (cols // 8)]
        scales: bfloat16 tensor with shape [expert_num, rows * (cols // group_size)]
    """
    weights_f32 = weights.to(torch.float32)
    e, rows, cols = weights_f32.shape
    if cols % group_size != 0 or cols % 2 != 0:
        raise ValueError(f"cols ({cols}) must be divisible by group_size ({group_size}) and 2")

    reshaped = weights_f32.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True)
    max_abs = torch.clamp(max_abs, min=1e-8)
    scales = (max_abs / 7.0).squeeze(-1)
    q = torch.round(reshaped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(e, rows, cols)
    packed = pack_tensor_per_row(q, num_bits=4).view(e, rows, cols // 8).contiguous()
    scales = scales.to(torch.bfloat16).contiguous().view(e, rows, cols // group_size).contiguous()

    print(f"Quantized weights: {packed.shape}, scales: {scales.shape}")
    print(f"Quantized tensors: \n{packed},\n {scales}")
    return packed, scales


def build_structured_tensor(shape: torch.Size, pattern: str) -> torch.Tensor:
    if pattern == "random":
        torch.manual_seed(42)
        return (torch.randn(shape, dtype=torch.bfloat16, device="cpu") / 100.0).contiguous()

    e, rows, cols = shape
    groups = cols // k_group_size
    group_builder = WEIGHT_PATTERNS[pattern][1]
    group_vals = group_builder(groups).to(torch.float32)
    block = group_vals.view(1, 1, groups, 1).expand(e, rows, groups, k_group_size).clone()
    row_signs = torch.where(
        (torch.arange(rows) % 2 == 0),
        torch.ones(rows, dtype=torch.float32),
        -torch.ones(rows, dtype=torch.float32),
    ).view(1, rows, 1, 1)
    col_offsets = torch.linspace(-0.0005, 0.0005, steps=k_group_size, dtype=torch.float32).view(1, 1, 1, k_group_size)
    block = block * row_signs + col_offsets
    return block.reshape(shape).to(torch.bfloat16).contiguous()


def prepare_k2_quantized_weights(pattern: str) -> Dict[str, torch.Tensor]:
    if pattern not in WEIGHT_PATTERNS:
        raise ValueError(f"Unknown weight pattern: {pattern}")

    gate_proj = build_structured_tensor((expert_num, intermediate_size, hidden_size), pattern)
    up_proj = build_structured_tensor((expert_num, intermediate_size, hidden_size), pattern)
    down_proj = build_structured_tensor((expert_num, hidden_size, intermediate_size), pattern)

    gate_q, gate_scales = quantize_k2_tensor(gate_proj, k_group_size)
    up_q, up_scales = quantize_k2_tensor(up_proj, k_group_size)
    down_q, down_scales = quantize_k2_tensor(down_proj, k_group_size)

    return {
        "gate_qweight": gate_q.contiguous(),
        "up_qweight": up_q.contiguous(),
        "down_qweight": down_q.contiguous(),
        "gate_scales": gate_scales.contiguous(),
        "up_scales": up_scales.contiguous(),
        "down_scales": down_scales.contiguous(),
        "original_fp16": {
            "gate_proj": gate_proj.to(torch.float16).contiguous(),
            "up_proj": up_proj.to(torch.float16).contiguous(),
            "down_proj": down_proj.to(torch.float16).contiguous(),
        },
    }


def build_moes_from_quantized_data(quant_data: Dict[str, torch.Tensor]):
    moes = []
    with torch.inference_mode(mode=True):
        for _ in range(layer_num):
            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.quant_config.bits = 4
            config.quant_config.group_size = k_group_size
            config.quant_config.zero_point = False

            config.gate_proj = quant_data["gate_qweight"].data_ptr()
            config.up_proj = quant_data["up_qweight"].data_ptr()
            config.down_proj = quant_data["down_qweight"].data_ptr()

            config.gate_scale = quant_data["gate_scales"].data_ptr()
            config.up_scale = quant_data["up_scales"].data_ptr()
            config.down_scale = quant_data["down_scales"].data_ptr()
            config.pool = CPUInfer.backend_

            moe = kt_kernel_ext.moe.AMXInt4_KGroup_MOE(config)
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()
            # CPUInfer.submit(moe.warm_up_task())
            # CPUInfer.sync()
            moes.append(moe)
    return moes


def run_case(pattern: str) -> Dict[str, float]:
    print("\n" + "=" * 70)
    desc = WEIGHT_PATTERNS[pattern][0]
    print(f"Running case: {pattern} -> {desc}")
    print("=" * 70)

    quant_data = prepare_k2_quantized_weights(pattern)
    moes = build_moes_from_quantized_data(quant_data)

    original_weights = quant_data["original_fp16"]
    gate_fp16 = original_weights["gate_proj"]
    up_fp16 = original_weights["up_proj"]
    down_fp16 = original_weights["down_proj"]

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
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_tensor.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            input_tensor_fp16 = input_tensor.to(torch.float16)
            t_output = moe_torch(
                input_tensor_fp16, expert_ids, weights, gate_fp16, up_fp16, down_fp16
            ).to(torch.bfloat16)

            t_output = t_output.flatten()
            output = output.flatten()

            diff = torch.mean(torch.abs(output - t_output)) / (torch.mean(torch.abs(t_output)) + 1e-12)
            diffs.append(diff.item())
            print(f"[{pattern}] Iteration {i}: relative L1 diff = {diff:.4f}")
            print(f"           output   {output}")
            print(f"           t_output {t_output}")

    mean_diff = float(sum(diffs) / len(diffs))
    max_diff = float(max(diffs))
    min_diff = float(min(diffs))
    return {"case": pattern, "description": desc, "mean": mean_diff, "max": max_diff, "min": min_diff}


def run_k2_moe_test():
    summary_rows = []
    for case_name in WEIGHT_PATTERNS.keys():
        results = run_case(case_name)
        summary_rows.append(results)
        # break

    print("\n=== Case vs. Relative Error Summary ===")
    print(f"{'Case':<20} {'Mean':>10} {'Max':>10} {'Min':>10}")
    for row in summary_rows:
        print(f"{row['case']:<20} {row['mean']*100:9.2f}% {row['max']*100:9.2f}% {row['min']*100:9.2f}%")


if __name__ == "__main__":
    run_k2_moe_test()
