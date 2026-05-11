#!/usr/bin/env python
# coding=utf-8
"""
Torch MoE benchmark with multiple execution paths:
1) expert: Python loop over experts
2) batched_bmm: batched matmul path (selected experts only)
3) batched_einsum: einsum path (selected experts only)
"""
import argparse
import os
import time

import torch
import torch.nn.quantized as nnq

scale, zero_point = 0.1, 0

# Keep defaults aligned with bench_moe_amx.py.
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
num_experts_per_tok = 8
layer_num = 5
qlen = 1
warm_up_iter = 1000
test_iter = 10000
gen_iter = 3000

num_threads = 64
interop_threads = 1
exclude_input_quant_time = True


def parse_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_torch_threads(threads: int, interop: int):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(interop)


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def build_common_inputs():
    expert_ids = (
        torch.rand(gen_iter * qlen, expert_num, device="cpu")
        .argsort(dim=-1)[:, :num_experts_per_tok]
        .reshape(gen_iter, qlen, num_experts_per_tok)
        .contiguous()
    )
    weights = torch.rand((gen_iter, qlen, num_experts_per_tok), dtype=torch.float32, device="cpu").contiguous()
    inputs = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device="cpu").contiguous()
    return expert_ids, weights, inputs


def build_float_projections(proj_dtype: torch.dtype):
    gate_projs, up_projs, down_projs = [], [], []
    for _ in range(layer_num):
        gate = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu").contiguous()
        up = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu").contiguous()
        down = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device="cpu").contiguous()
        gate_projs.append(gate.to(proj_dtype))
        up_projs.append(up.to(proj_dtype))
        down_projs.append(down.to(proj_dtype))
    return gate_projs, up_projs, down_projs


def build_int8_projections():
    gate_projs, up_projs, down_projs = [], [], []
    for _ in range(layer_num):
        gate = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu").contiguous()
        up = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device="cpu").contiguous()
        down = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device="cpu").contiguous()

        q_gate_layer, q_up_layer, q_down_layer = [], [], []
        for i in range(expert_num):
            gate_q = torch.quantize_per_tensor(gate[i], scale, zero_point, torch.qint8)
            up_q = torch.quantize_per_tensor(up[i], scale, zero_point, torch.qint8)
            down_q = torch.quantize_per_tensor(down[i], scale, zero_point, torch.qint8)

            q_gate = nnq.Linear(hidden_size, intermediate_size)
            q_up = nnq.Linear(hidden_size, intermediate_size)
            q_down = nnq.Linear(intermediate_size, hidden_size)
            q_gate.set_weight_bias(gate_q, None)
            q_up.set_weight_bias(up_q, None)
            q_down.set_weight_bias(down_q, None)

            q_gate_layer.append(q_gate)
            q_up_layer.append(q_up)
            q_down_layer.append(q_down)

        gate_projs.append(q_gate_layer)
        up_projs.append(q_up_layer)
        down_projs.append(q_down_layer)

    return gate_projs, up_projs, down_projs


def moe_expert_float(input_fp, expert_ids_one, weights_one, gate_proj, up_proj, down_proj):
    counts = expert_ids_one.new_zeros((expert_ids_one.shape[0], expert_num))
    counts.scatter_(1, expert_ids_one, 1)
    tokens_per_expert = counts.sum(dim=0)

    idxs = expert_ids_one.reshape(-1).argsort()
    sorted_tokens = input_fp[idxs // expert_ids_one.shape[1]]

    outputs = []
    start_idx = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        token_block = sorted_tokens[start_idx:end_idx]
        gate_buf = torch.mm(token_block.to(gate_proj.dtype), gate_proj[expert_idx].t())
        up_buf = torch.mm(token_block.to(up_proj.dtype), up_proj[expert_idx].t())
        inter = act_fn(gate_buf) * up_buf
        out = torch.mm(inter.to(down_proj.dtype), down_proj[expert_idx].t())
        outputs.append(out)
        start_idx = end_idx

    concat_out = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    reordered = torch.empty_like(concat_out)
    reordered[idxs] = concat_out
    return (
        reordered.view(*expert_ids_one.shape, -1)
        .type(weights_one.dtype)
        .mul_(weights_one.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(reordered.dtype)
    )


def moe_expert_int8(input_fp, expert_ids_one, weights_one, gate_proj, up_proj, down_proj, input_q=None):
    counts = expert_ids_one.new_zeros((expert_ids_one.shape[0], expert_num))
    counts.scatter_(1, expert_ids_one, 1)
    tokens_per_expert = counts.sum(dim=0)

    idxs = expert_ids_one.reshape(-1).argsort()
    if input_q is None:
        input_q = torch.quantize_per_tensor(input_fp.to(torch.float32), scale, zero_point, torch.quint8)
    sorted_tokens_q = input_q[idxs // expert_ids_one.shape[1]]

    outputs = []
    start_idx = 0
    for expert_idx, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        token_block_q = sorted_tokens_q[start_idx:end_idx]
        gate_buf = gate_proj[expert_idx](token_block_q).dequantize()
        up_buf = up_proj[expert_idx](token_block_q).dequantize()
        inter = act_fn(gate_buf) * up_buf
        inter_q = torch.quantize_per_tensor(inter, scale, zero_point, torch.quint8)
        out = down_proj[expert_idx](inter_q).dequantize()
        outputs.append(out)
        start_idx = end_idx

    concat_out = torch.cat(outputs, dim=0) if outputs else torch.empty((0, hidden_size), dtype=torch.float32)
    reordered = torch.empty_like(concat_out)
    reordered[idxs] = concat_out
    return (
        reordered.view(*expert_ids_one.shape, -1)
        .type(weights_one.dtype)
        .mul_(weights_one.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(reordered.dtype)
    )


def moe_batched_bmm(input_fp, expert_ids_one, weights_one, gate_proj, up_proj, down_proj):
    q, k = expert_ids_one.shape
    x = input_fp.to(gate_proj.dtype)
    flat_ids = expert_ids_one.reshape(-1)

    gate_sel = gate_proj.index_select(0, flat_ids).view(q, k, intermediate_size, hidden_size)
    up_sel = up_proj.index_select(0, flat_ids).view(q, k, intermediate_size, hidden_size)
    down_sel = down_proj.index_select(0, flat_ids).view(q, k, hidden_size, intermediate_size)

    x_rep = x.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_size).unsqueeze(-1)
    gate_buf = (
        torch.bmm(gate_sel.reshape(-1, intermediate_size, hidden_size), x_rep).squeeze(-1).view(q, k, intermediate_size)
    )
    up_buf = (
        torch.bmm(up_sel.reshape(-1, intermediate_size, hidden_size), x_rep).squeeze(-1).view(q, k, intermediate_size)
    )

    inter = act_fn(gate_buf) * up_buf
    out = (
        torch.bmm(
            down_sel.reshape(-1, hidden_size, intermediate_size),
            inter.reshape(-1, intermediate_size).unsqueeze(-1),
        )
        .squeeze(-1)
        .view(q, k, hidden_size)
    )

    return (out.type(weights_one.dtype) * weights_one.unsqueeze(-1)).sum(dim=1).type(out.dtype)


def moe_batched_einsum(input_fp, expert_ids_one, weights_one, gate_proj, up_proj, down_proj):
    q, k = expert_ids_one.shape
    x = input_fp.to(gate_proj.dtype)
    flat_ids = expert_ids_one.reshape(-1)

    gate_sel = gate_proj.index_select(0, flat_ids).view(q, k, intermediate_size, hidden_size)
    up_sel = up_proj.index_select(0, flat_ids).view(q, k, intermediate_size, hidden_size)
    down_sel = down_proj.index_select(0, flat_ids).view(q, k, hidden_size, intermediate_size)

    gate_buf = torch.einsum("qh,qkih->qki", x, gate_sel)
    up_buf = torch.einsum("qh,qkih->qki", x, up_sel)
    inter = act_fn(gate_buf) * up_buf
    out = torch.einsum("qki,qkhi->qkh", inter, down_sel)

    return (out.type(weights_one.dtype) * weights_one.unsqueeze(-1)).sum(dim=1).type(out.dtype)


def run_one_iter(
    exec_path, quant_mode, input_tensor, expert_ids_one, weights_one, gate_proj, up_proj, down_proj, input_q=None
):
    if quant_mode == "qint8":
        if exec_path != "expert":
            raise ValueError("qint8 only supports expert path in this benchmark")
        return moe_expert_int8(input_tensor, expert_ids_one, weights_one, gate_proj, up_proj, down_proj, input_q)

    if exec_path == "expert":
        return moe_expert_float(input_tensor, expert_ids_one, weights_one, gate_proj, up_proj, down_proj)
    if exec_path == "batched_bmm":
        return moe_batched_bmm(input_tensor, expert_ids_one, weights_one, gate_proj, up_proj, down_proj)
    if exec_path == "batched_einsum":
        return moe_batched_einsum(input_tensor, expert_ids_one, weights_one, gate_proj, up_proj, down_proj)

    raise ValueError(f"Unknown exec_path={exec_path}")


def bench_moe(quant_mode: str, exec_path: str = "expert"):
    with torch.inference_mode(mode=True):
        if quant_mode == "fp32":
            proj_type = torch.float32
            bytes_per_elem = 4.0
        elif quant_mode == "fp16":
            proj_type = torch.float16
            bytes_per_elem = 2.0
        elif quant_mode == "bf16":
            proj_type = torch.bfloat16
            bytes_per_elem = 2.0
        elif quant_mode == "qint8":
            proj_type = torch.qint8
            bytes_per_elem = 1.0
        else:
            raise ValueError(f"Unsupported quant_mode={quant_mode}")

        if quant_mode == "qint8":
            gate_projs, up_projs, down_projs = build_int8_projections()
        else:
            gate_projs, up_projs, down_projs = build_float_projections(proj_type)

        expert_ids, weights, inputs = build_common_inputs()
        pre_quant_inputs = None
        if quant_mode == "qint8" and exclude_input_quant_time:
            pre_quant_inputs = [
                torch.quantize_per_tensor(inputs[i].to(torch.float32), scale, zero_point, torch.quint8)
                for i in range(layer_num)
            ]

        for i in range(warm_up_iter):
            layer_idx = i % layer_num
            gen_idx = i % gen_iter
            input_q = pre_quant_inputs[layer_idx] if pre_quant_inputs is not None else None
            run_one_iter(
                exec_path,
                quant_mode,
                inputs[layer_idx],
                expert_ids[gen_idx],
                weights[gen_idx],
                gate_projs[layer_idx],
                up_projs[layer_idx],
                down_projs[layer_idx],
                input_q,
            )

        start = time.perf_counter()
        for i in range(test_iter):
            layer_idx = i % layer_num
            gen_idx = i % gen_iter
            input_q = pre_quant_inputs[layer_idx] if pre_quant_inputs is not None else None
            run_one_iter(
                exec_path,
                quant_mode,
                inputs[layer_idx],
                expert_ids[gen_idx],
                weights[gen_idx],
                gate_projs[layer_idx],
                up_projs[layer_idx],
                down_projs[layer_idx],
                input_q,
            )
        end = time.perf_counter()

        total_time = end - start
        time_us = total_time / test_iter * 1e6

        work_elems = hidden_size * intermediate_size * 3 * num_experts_per_tok * qlen
        bandwidth = work_elems * bytes_per_elem * test_iter / total_time / 1e9
        flops = work_elems * 2 * test_iter / total_time / 1e12

        print("Quant mode:", quant_mode)
        print("Exec path:", exec_path)
        print("Time(s):", total_time)
        print("Iteration:", test_iter)
        print("Time(us) per iteration:", time_us)
        print("Bandwidth:", bandwidth, "GB/s")
        print("Flops:", flops, "TFLOPS")
        if quant_mode == "qint8":
            print("Exclude input quantization time:", exclude_input_quant_time)
            print("Note: intermediate quant/dequant is still inside forward path.")
        print("")


def main():
    global expert_num
    global hidden_size
    global intermediate_size
    global num_experts_per_tok
    global layer_num
    global qlen
    global warm_up_iter
    global test_iter
    global gen_iter
    global num_threads
    global interop_threads
    global exclude_input_quant_time

    parser = argparse.ArgumentParser(description="Torch MoE benchmark")
    parser.add_argument("--expert-num", type=int, default=expert_num)
    parser.add_argument("--hidden-size", type=int, default=hidden_size)
    parser.add_argument("--intermediate-size", type=int, default=intermediate_size)
    parser.add_argument("--num-experts-per-tok", type=int, default=num_experts_per_tok)
    parser.add_argument("--layer-num", type=int, default=layer_num)
    parser.add_argument("--qlen", type=int, default=qlen)
    parser.add_argument("--warm-up-iter", type=int, default=warm_up_iter)
    parser.add_argument("--test-iter", type=int, default=test_iter)
    parser.add_argument("--gen-iter", type=int, default=gen_iter)
    parser.add_argument("--threads", type=int, default=num_threads)
    parser.add_argument("--interop-threads", type=int, default=interop_threads)
    parser.add_argument("--modes", type=str, default="bf16,qint8")
    parser.add_argument("--exec-paths", type=str, default="expert,batched_bmm,batched_einsum")
    parser.add_argument("--include-input-quant-time", action="store_true", default=False)
    args = parser.parse_args()

    expert_num = args.expert_num
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    num_experts_per_tok = args.num_experts_per_tok
    layer_num = args.layer_num
    qlen = args.qlen
    warm_up_iter = args.warm_up_iter
    test_iter = args.test_iter
    gen_iter = args.gen_iter
    num_threads = args.threads
    interop_threads = args.interop_threads
    exclude_input_quant_time = not args.include_input_quant_time

    configure_torch_threads(num_threads, interop_threads)

    modes = parse_csv(args.modes)
    exec_paths = parse_csv(args.exec_paths)

    print("[config] torch bench")
    print(
        f"[config] E={expert_num}, H={hidden_size}, I={intermediate_size}, topk={num_experts_per_tok}, "
        f"layers={layer_num}, qlen={qlen}"
    )
    print(f"[config] warmup={warm_up_iter}, test={test_iter}, gen_iter={gen_iter}")
    print(f"[config] threads={num_threads}, interop_threads={interop_threads}")
    print(f"[config] modes={modes}, exec_paths={exec_paths}")
    print(f"[config] exclude_input_quant_time={exclude_input_quant_time}")

    for mode in modes:
        for path in exec_paths:
            if mode == "qint8" and path != "expert":
                print(f"Skip mode={mode}, exec_path={path}: qint8 only supports expert path")
                print("")
                continue
            bench_moe(mode, path)


if __name__ == "__main__":
    main()
