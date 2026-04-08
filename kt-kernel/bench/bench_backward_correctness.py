#!/usr/bin/env python3
"""
Correctness test: AMX SFT backward vs PyTorch reference backward.

Compares grad_input and all 6 LoRA gradients between kt-kernel's AMX fused
backward and the PyTorch reference implementation.

Usage:
    cd /path/to/kt-kernel
    python3 bench/bench_backward_correctness.py
"""
import os
import torch
import torch.nn.functional as F

from _load_kt_kernel import load_local_kt_kernel

kt_kernel_ext = load_local_kt_kernel().kt_kernel_ext

# ─── Config ───
EXPERT_NUM = 8
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS = 8
MAX_LEN = 8192 + 64
NUM_THREADS = 64
LORA_RANK = 8
LORA_ALPHA = 32
LORA_SCALING = LORA_ALPHA / LORA_RANK

SEQLENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


# ─── PyTorch reference (from test_moe_sft_amx_no_tp.py) ───

def act_fn(x):
    return F.silu(x)

def lora_linear_forward(x, weight, lora_a, lora_b, scaling):
    base_out = torch.mm(x, weight.t())
    lora_out = torch.mm(torch.mm(x, lora_a.t()), lora_b.t()) * scaling
    return base_out + lora_out

def lora_linear_backward(grad_output, x, weight, lora_a, lora_b, scaling):
    if grad_output.dtype != x.dtype:
        x = x.to(grad_output.dtype)
    if grad_output.dtype != weight.dtype:
        weight = weight.to(grad_output.dtype)
    if grad_output.dtype != lora_a.dtype:
        lora_a = lora_a.to(grad_output.dtype)
    if grad_output.dtype != lora_b.dtype:
        lora_b = lora_b.to(grad_output.dtype)
    grad_input = torch.mm(grad_output, weight)
    grad_input += torch.mm(torch.mm(grad_output, lora_b), lora_a) * scaling
    grad_lora_b = torch.mm(grad_output.t(), torch.mm(x, lora_a.t())) * scaling
    grad_lora_a = torch.mm(torch.mm(lora_b.t(), grad_output.t()), x) * scaling
    return grad_input, grad_lora_a, grad_lora_b

def mlp_lora_forward(x, gate_proj, up_proj, down_proj,
                     gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                     down_lora_a, down_lora_b, scaling):
    gate_out = lora_linear_forward(x, gate_proj, gate_lora_a, gate_lora_b, scaling)
    up_out = lora_linear_forward(x, up_proj, up_lora_a, up_lora_b, scaling)
    gate_activated = act_fn(gate_out)
    intermediate = gate_activated * up_out
    output = lora_linear_forward(intermediate, down_proj, down_lora_a, down_lora_b, scaling)
    saved = {"x": x, "gate_out": gate_out, "up_out": up_out,
             "gate_activated": gate_activated, "intermediate": intermediate}
    return output, saved

def mlp_lora_backward(grad_output, saved, gate_proj, up_proj, down_proj,
                      gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                      down_lora_a, down_lora_b, scaling):
    x = saved["x"]
    gate_out = saved["gate_out"]
    up_out = saved["up_out"]
    gate_activated = saved["gate_activated"]
    intermediate = saved["intermediate"]
    grad_intermediate, grad_down_lora_a, grad_down_lora_b = lora_linear_backward(
        grad_output, intermediate, down_proj, down_lora_a, down_lora_b, scaling)
    grad_gate_activated = grad_intermediate * up_out
    grad_up_out = grad_intermediate * gate_activated
    sigmoid_gate = torch.sigmoid(gate_out)
    grad_gate_out = grad_gate_activated * sigmoid_gate * (1 + gate_out * (1 - sigmoid_gate))
    grad_x_up, grad_up_lora_a, grad_up_lora_b = lora_linear_backward(
        grad_up_out, x, up_proj, up_lora_a, up_lora_b, scaling)
    grad_x_gate, grad_gate_lora_a, grad_gate_lora_b = lora_linear_backward(
        grad_gate_out, x, gate_proj, gate_lora_a, gate_lora_b, scaling)
    return {
        "grad_input": grad_x_up + grad_x_gate,
        "grad_gate_lora_a": grad_gate_lora_a, "grad_gate_lora_b": grad_gate_lora_b,
        "grad_up_lora_a": grad_up_lora_a, "grad_up_lora_b": grad_up_lora_b,
        "grad_down_lora_a": grad_down_lora_a, "grad_down_lora_b": grad_down_lora_b,
    }

def moe_sft_torch_forward(input, expert_ids, weights,
                          gate_proj, up_proj, down_proj,
                          gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                          down_lora_a, down_lora_b, scaling):
    qlen = input.shape[0]
    k = expert_ids.shape[1]
    cnts = expert_ids.new_zeros((qlen, EXPERT_NUM))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // k]
    outputs, saved_list = [], []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        if num_tokens == 0:
            saved_list.append(None)
            continue
        end_idx = start_idx + int(num_tokens)
        tokens_for_expert = sorted_tokens[start_idx:end_idx]
        expert_out, saved = mlp_lora_forward(
            tokens_for_expert, gate_proj[i], up_proj[i], down_proj[i],
            gate_lora_a[i], gate_lora_b[i], up_lora_a[i], up_lora_b[i],
            down_lora_a[i], down_lora_b[i], scaling)
        outputs.append(expert_out)
        saved["expert_id"] = i
        saved["start_idx"] = start_idx
        saved["end_idx"] = end_idx
        saved_list.append(saved)
        start_idx = end_idx
    outs = torch.cat(outputs, 0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    output = new_x.view(qlen, k, -1).type(weights.dtype).mul_(weights.unsqueeze(-1)).sum(1).type(new_x.dtype)
    moe_saved = {"input": input, "expert_ids": expert_ids, "weights": weights,
                 "idxs": idxs, "tokens_per_expert": tokens_per_expert,
                 "expert_saved_tensors": saved_list}
    return output, moe_saved

def moe_sft_torch_backward(grad_output, moe_saved,
                           gate_proj, up_proj, down_proj,
                           gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
                           down_lora_a, down_lora_b, scaling):
    input = moe_saved["input"]
    expert_ids = moe_saved["expert_ids"]
    weights = moe_saved["weights"]
    idxs = moe_saved["idxs"]
    tokens_per_expert = moe_saved["tokens_per_expert"]
    expert_saved_list = moe_saved["expert_saved_tensors"]
    qlen, k = expert_ids.shape
    grad_output_expanded = grad_output.unsqueeze(1) * weights.unsqueeze(-1)
    grad_output_expanded = grad_output_expanded.view(-1, grad_output.shape[-1]).to(grad_output.dtype)
    sorted_grad_output = grad_output_expanded[idxs]
    grad_input_sorted = torch.zeros_like(sorted_grad_output)
    g_gate_a = torch.zeros_like(gate_lora_a)
    g_gate_b = torch.zeros_like(gate_lora_b)
    g_up_a = torch.zeros_like(up_lora_a)
    g_up_b = torch.zeros_like(up_lora_b)
    g_down_a = torch.zeros_like(down_lora_a)
    g_down_b = torch.zeros_like(down_lora_b)
    for i, saved in enumerate(expert_saved_list):
        if saved is None:
            continue
        start_idx, end_idx = saved["start_idx"], saved["end_idx"]
        grads = mlp_lora_backward(
            sorted_grad_output[start_idx:end_idx], saved,
            gate_proj[i], up_proj[i], down_proj[i],
            gate_lora_a[i], gate_lora_b[i], up_lora_a[i], up_lora_b[i],
            down_lora_a[i], down_lora_b[i], scaling)
        grad_input_sorted[start_idx:end_idx] = grads["grad_input"]
        g_gate_a[i] = grads["grad_gate_lora_a"]
        g_gate_b[i] = grads["grad_gate_lora_b"]
        g_up_a[i] = grads["grad_up_lora_a"]
        g_up_b[i] = grads["grad_up_lora_b"]
        g_down_a[i] = grads["grad_down_lora_a"]
        g_down_b[i] = grads["grad_down_lora_b"]
    grad_input_flat = torch.zeros_like(grad_input_sorted)
    grad_input_flat[idxs] = grad_input_sorted
    grad_input = grad_input_flat.view(qlen, k, -1).sum(dim=1)
    return {
        "grad_input": grad_input,
        "grad_gate_lora_a": g_gate_a, "grad_gate_lora_b": g_gate_b,
        "grad_up_lora_a": g_up_a, "grad_up_lora_b": g_up_b,
        "grad_down_lora_a": g_down_a, "grad_down_lora_b": g_down_b,
    }


# ─── Compare helper ───

def compare_tensors(name, amx_t, ref_t):
    """Compare two tensors, return (max_abs_diff, cos_sim, relative_error)."""
    amx_f = amx_t.float()
    ref_f = ref_t.float()
    diff = (amx_f - ref_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_norm = ref_f.norm().item()
    amx_norm = amx_f.norm().item()
    rel_err = max_diff / (ref_norm / ref_f.numel()**0.5 + 1e-12)
    # Cosine similarity
    cos = F.cosine_similarity(amx_f.flatten().unsqueeze(0),
                               ref_f.flatten().unsqueeze(0)).item()
    return {
        'name': name, 'max_diff': max_diff, 'mean_diff': mean_diff,
        'amx_norm': amx_norm, 'ref_norm': ref_norm,
        'cos_sim': cos, 'rel_err': rel_err,
    }


def run_correctness():
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(42)

    E, H, I, k, r = EXPERT_NUM, HIDDEN_SIZE, INTERMEDIATE_SIZE, N_ROUTED_EXPERTS, LORA_RANK

    print(f"Config: E={E}, H={H}, I={I}, k={k}, r={r}, scaling={LORA_SCALING}")
    print(f"Torch threads: {torch.get_num_threads()}\n")

    # ─── Weights ───
    gate_proj = (torch.randn(E, I, H, dtype=torch.bfloat16) / 100).contiguous()
    up_proj = (torch.randn(E, I, H, dtype=torch.bfloat16) / 100).contiguous()
    down_proj = (torch.randn(E, H, I, dtype=torch.bfloat16) / 100).contiguous()

    gate_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    gate_lora_b = (torch.randn(E, I, r, dtype=torch.bfloat16) / 100).contiguous()
    up_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    up_lora_b = (torch.randn(E, I, r, dtype=torch.bfloat16) / 100).contiguous()
    down_lora_a = (torch.randn(E, r, I, dtype=torch.bfloat16) / 100).contiguous()
    down_lora_b = (torch.randn(E, H, r, dtype=torch.bfloat16) / 100).contiguous()

    # ─── AMX setup ───
    pool_config = kt_kernel_ext.WorkerPoolConfig()
    pool_config.subpool_count = 2
    pool_config.subpool_numa_map = [0, 1]
    pool_config.subpool_thread_count = [NUM_THREADS // 2, NUM_THREADS // 2]
    cpu_infer = kt_kernel_ext.CPUInfer(pool_config)

    config = kt_kernel_ext.moe.MOESFTConfig()
    config.expert_num = E
    config.num_experts_per_tok = k
    config.hidden_size = H
    config.intermediate_size = I
    config.lora_rank = r
    config.lora_alpha = LORA_ALPHA
    config.max_cache_depth = 2
    config.max_len = MAX_LEN
    config.layer_idx = 0
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    config.gate_lora_a = gate_lora_a.data_ptr()
    config.gate_lora_b = gate_lora_b.data_ptr()
    config.up_lora_a = up_lora_a.data_ptr()
    config.up_lora_b = up_lora_b.data_ptr()
    config.down_lora_a = down_lora_a.data_ptr()
    config.down_lora_b = down_lora_b.data_ptr()
    config.pool = cpu_infer.backend_

    moe_amx = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
    cpu_infer.submit(moe_amx.load_weights_task())
    cpu_infer.sync()
    cpu_infer.submit(moe_amx.warm_up_task())
    cpu_infer.sync()
    print("AMX setup done.\n")

    # ─── Header ───
    print(f"{'qlen':>6} | {'grad_input':>20} | {'gate_A':>12} {'gate_B':>12} | "
          f"{'up_A':>12} {'up_B':>12} | {'down_A':>12} {'down_B':>12} | {'fwd_cos':>8}")
    print("-" * 140)

    all_pass = True

    for qlen in SEQLENS:
        torch.manual_seed(42 + qlen)

        expert_ids = torch.stack(
            [torch.randperm(E, dtype=torch.int64)[:k] for _ in range(qlen)]).contiguous()
        weights_moe = torch.rand(qlen, k, dtype=torch.float32).contiguous()
        weights_moe = weights_moe / weights_moe.sum(dim=-1, keepdim=True)
        input_data = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()
        grad_out = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()

        # ═══ AMX forward + backward ═══
        bsz_tensor = torch.tensor([qlen], device="cpu")
        output_amx = torch.zeros(qlen, H, dtype=torch.bfloat16).contiguous()
        grad_in_amx = torch.zeros(qlen, H, dtype=torch.bfloat16).contiguous()
        g_gate_a_amx = torch.zeros_like(gate_lora_a)
        g_gate_b_amx = torch.zeros_like(gate_lora_b)
        g_up_a_amx = torch.zeros_like(up_lora_a)
        g_up_b_amx = torch.zeros_like(up_lora_b)
        g_down_a_amx = torch.zeros_like(down_lora_a)
        g_down_b_amx = torch.zeros_like(down_lora_b)
        g_weights_amx = torch.zeros(qlen, k, dtype=torch.float32).contiguous()

        cpu_infer.submit(moe_amx.forward_sft_task(
            bsz_tensor.data_ptr(), k,
            expert_ids.data_ptr(), weights_moe.data_ptr(),
            input_data.data_ptr(), output_amx.data_ptr(), True))
        cpu_infer.sync()

        cpu_infer.submit(moe_amx.backward_task(
            grad_out.data_ptr(), grad_in_amx.data_ptr(),
            g_gate_a_amx.data_ptr(), g_gate_b_amx.data_ptr(),
            g_up_a_amx.data_ptr(), g_up_b_amx.data_ptr(),
            g_down_a_amx.data_ptr(), g_down_b_amx.data_ptr(),
            g_weights_amx.data_ptr()))
        cpu_infer.sync()

        # ═══ PyTorch reference forward + backward ═══
        output_ref, moe_saved = moe_sft_torch_forward(
            input_data, expert_ids, weights_moe,
            gate_proj, up_proj, down_proj,
            gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
            down_lora_a, down_lora_b, LORA_SCALING)

        ref_grads = moe_sft_torch_backward(
            grad_out, moe_saved,
            gate_proj, up_proj, down_proj,
            gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
            down_lora_a, down_lora_b, LORA_SCALING)

        # ═══ Compare ═══
        fwd_cos = F.cosine_similarity(
            output_amx.float().flatten().unsqueeze(0),
            output_ref.float().flatten().unsqueeze(0)).item()

        comparisons = [
            compare_tensors("grad_input", grad_in_amx, ref_grads["grad_input"]),
            compare_tensors("gate_A", g_gate_a_amx, ref_grads["grad_gate_lora_a"]),
            compare_tensors("gate_B", g_gate_b_amx, ref_grads["grad_gate_lora_b"]),
            compare_tensors("up_A", g_up_a_amx, ref_grads["grad_up_lora_a"]),
            compare_tensors("up_B", g_up_b_amx, ref_grads["grad_up_lora_b"]),
            compare_tensors("down_A", g_down_a_amx, ref_grads["grad_down_lora_a"]),
            compare_tensors("down_B", g_down_b_amx, ref_grads["grad_down_lora_b"]),
        ]

        # Print compact row
        gi = comparisons[0]
        ga, gb = comparisons[1], comparisons[2]
        ua, ub = comparisons[3], comparisons[4]
        da, db = comparisons[5], comparisons[6]

        def fmt_cos(c):
            v = c['cos_sim']
            if v > 0.999:
                return f"{v:.6f}"
            elif v > 0.99:
                return f"{v:.5f}"
            else:
                return f"{v:.4f}"

        print(f"{qlen:>6} | cos={fmt_cos(gi)} md={gi['max_diff']:.2e} | "
              f"{fmt_cos(ga)} {fmt_cos(gb)} | "
              f"{fmt_cos(ua)} {fmt_cos(ub)} | "
              f"{fmt_cos(da)} {fmt_cos(db)} | "
              f"{fwd_cos:.6f}")

        # Check pass/fail (cosine > 0.99 for bf16)
        min_cos = min(c['cos_sim'] for c in comparisons)
        if min_cos < 0.99:
            print(f"  *** WARN: min cosine similarity {min_cos:.6f} < 0.99")
            all_pass = False

    print()
    if all_pass:
        print("PASSED: All gradients match within bf16 tolerance (cos > 0.99)")
    else:
        print("FAILED: Some gradients have low similarity")

    # Detailed report for last qlen
    print(f"\nDetailed report (qlen={SEQLENS[-1]}):")
    for c in comparisons:
        print(f"  {c['name']:>12}: cos={c['cos_sim']:.8f}  max_diff={c['max_diff']:.4e}  "
              f"mean_diff={c['mean_diff']:.4e}  norms: amx={c['amx_norm']:.4f} ref={c['ref_norm']:.4f}")


if __name__ == '__main__':
    run_correctness()
