#!/usr/bin/env python3
"""
Benchmark: AMX pre-packed expert GEMM vs Torch TN-layout GEMM.

Demonstrates the layout mismatch between forward-optimized AMX block format
and backward transposed GEMM. Pre-packed BufferB benefits forward but the
same layout is suboptimal for backward's transposed access pattern.

Measures:
  - AMX forward  (pre-packed BufferB, base + LoRA fused)
  - AMX backward (mix of pre-packed + transposed buffers, base + LoRA)
  - Torch forward  (standard TN layout via F.linear, base + LoRA)
  - Torch backward (same TN layout, autograd, base + LoRA)

X-axis: sequence length (num_tokens per expert)

Usage:
    cd /path/to/kt-kernel
    python3 bench/bench_prepack_vs_torch_gemm.py
"""
import os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from _load_kt_kernel import load_local_kt_kernel

kt_kernel_ext = load_local_kt_kernel().kt_kernel_ext

# ─── Model dimensions (Qwen3-30B-A3B MoE layer) ───
EXPERT_NUM = 8
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS = 8
MAX_LEN = 8192 + 64
NUM_THREADS = 64

# ─── LoRA config ───
LORA_RANK = 8
LORA_ALPHA = 32
LORA_SCALING = LORA_ALPHA / LORA_RANK

SEQLENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


# ═══════════════════════════════════════════════════════════════════
#  Torch MoE (standard TN layout GEMMs via F.linear)
# ═══════════════════════════════════════════════════════════════════

class SwiGLUExpert(nn.Module):
    """Single expert with base weights + LoRA (TN layout for F.linear)."""
    def __init__(self, hidden, inter, gate_w, up_w, down_w,
                 gate_la, gate_lb, up_la, up_lb, down_la, down_lb, scaling):
        super().__init__()
        # F.linear(x, W) computes x @ W^T — W stored as [out, in] row-major
        self.gate_w = nn.Parameter(gate_w, requires_grad=False)
        self.up_w = nn.Parameter(up_w, requires_grad=False)
        self.down_w = nn.Parameter(down_w, requires_grad=False)
        # LoRA: A is [r, in], B is [out, r]
        self.gate_la = nn.Parameter(gate_la, requires_grad=True)
        self.gate_lb = nn.Parameter(gate_lb, requires_grad=True)
        self.up_la = nn.Parameter(up_la, requires_grad=True)
        self.up_lb = nn.Parameter(up_lb, requires_grad=True)
        self.down_la = nn.Parameter(down_la, requires_grad=True)
        self.down_lb = nn.Parameter(down_lb, requires_grad=True)
        self.scaling = scaling

    def forward(self, x):
        g = F.linear(x, self.gate_w) + self.scaling * F.linear(F.linear(x, self.gate_la), self.gate_lb)
        u = F.linear(x, self.up_w) + self.scaling * F.linear(F.linear(x, self.up_la), self.up_lb)
        a = F.silu(g) * u
        return F.linear(a, self.down_w) + self.scaling * F.linear(F.linear(a, self.down_la), self.down_lb)


def moe_torch(x, expert_ids, weights, experts):
    """Route tokens to experts, compute, and combine."""
    T, k = expert_ids.shape
    E = len(experts)
    tok_cnt = torch.zeros(E, dtype=torch.int64)
    for e in expert_ids.view(-1):
        tok_cnt[e] += 1
    order = expert_ids.view(-1).argsort()
    packed = x[order // k]
    outputs, start = [], 0
    for e in range(E):
        num = tok_cnt[e].item()
        if not num:
            continue
        outputs.append(experts[e](packed[start:start+num]))
        start += num
    out_all = torch.cat(outputs, 0) if outputs else packed.new_empty(0, x.size(-1))
    out_restore = torch.empty_like(out_all)
    out_restore[order] = out_all
    out_restore = out_restore.view(T, k, -1)
    return (out_restore * weights.unsqueeze(-1)).sum(1)


# ═══════════════════════════════════════════════════════════════════
#  Benchmark
# ═══════════════════════════════════════════════════════════════════

def run_benchmark():
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(42)

    H, I, E, k = HIDDEN_SIZE, INTERMEDIATE_SIZE, EXPERT_NUM, N_ROUTED_EXPERTS
    r = LORA_RANK

    print(f"Config: E={E}, H={H}, I={I}, k={k}, threads={NUM_THREADS}")
    print(f"LoRA: rank={r}, alpha={LORA_ALPHA}, scaling={LORA_SCALING}")
    print(f"Torch threads: {torch.get_num_threads()}\n")

    # ─── Base weights ───
    gate_proj = torch.randn(E, I, H, dtype=torch.bfloat16).contiguous()
    up_proj = torch.randn_like(gate_proj)
    down_proj = torch.randn(E, H, I, dtype=torch.bfloat16).contiguous()

    # ─── LoRA weights ───
    gate_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    gate_lora_b = (torch.randn(E, I, r, dtype=torch.bfloat16) / 100).contiguous()
    up_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    up_lora_b = (torch.randn(E, I, r, dtype=torch.bfloat16) / 100).contiguous()
    down_lora_a = (torch.randn(E, r, I, dtype=torch.bfloat16) / 100).contiguous()
    down_lora_b = (torch.randn(E, H, r, dtype=torch.bfloat16) / 100).contiguous()

    # ─── AMX SFT setup ───
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
    print("AMX weights pre-packed (BufferB format).")

    # ─── Torch experts (standard TN layout, base + LoRA) ───
    torch_experts = nn.ModuleList([
        SwiGLUExpert(H, I, gate_proj[e], up_proj[e], down_proj[e],
                     gate_lora_a[e], gate_lora_b[e],
                     up_lora_a[e], up_lora_b[e],
                     down_lora_a[e], down_lora_b[e],
                     LORA_SCALING)
        for e in range(E)])
    print("Torch experts created (row-major TN layout, base + LoRA).")
    print()

    # ─── Header ───
    hdr = (f"{'qlen':>6} | "
           f"{'AMX fwd':>9} {'AMX bwd':>9} {'AMX tot':>9} | "
           f"{'Torch fwd':>10} {'Torch bwd':>10} {'Torch tot':>10} | "
           f"{'Fwd spd':>7} {'Bwd spd':>7} {'Tot spd':>7}")
    print(hdr)
    print("-" * len(hdr))

    results = []

    for qlen in SEQLENS:
        test_iter = max(10, min(500, 1000 // max(qlen, 1)))
        warmup_iter = max(3, test_iter // 5)

        expert_ids = torch.stack(
            [torch.randperm(E, dtype=torch.int64)[:k] for _ in range(qlen)]).contiguous()
        weights_moe = torch.rand(qlen, k, dtype=torch.float32).contiguous()
        weights_moe = weights_moe / weights_moe.sum(dim=-1, keepdim=True)
        input_data = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()
        grad_out = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()

        # AMX buffers
        bsz_tensor = torch.tensor([qlen], device="cpu")
        output_amx = torch.zeros(qlen, H, dtype=torch.bfloat16).contiguous()
        grad_in_amx = torch.zeros_like(input_data)
        grad_gate_lora_a = torch.zeros_like(gate_lora_a)
        grad_gate_lora_b = torch.zeros_like(gate_lora_b)
        grad_up_lora_a = torch.zeros_like(up_lora_a)
        grad_up_lora_b = torch.zeros_like(up_lora_b)
        grad_down_lora_a = torch.zeros_like(down_lora_a)
        grad_down_lora_b = torch.zeros_like(down_lora_b)
        grad_weights = torch.zeros(qlen, k, dtype=torch.float32).contiguous()

        # ════════════════════════════════════════════════
        #  AMX: forward + backward (pre-packed BufferB)
        # ════════════════════════════════════════════════
        for _ in range(warmup_iter):
            cpu_infer.submit(moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(), k,
                expert_ids.data_ptr(), weights_moe.data_ptr(),
                input_data.data_ptr(), output_amx.data_ptr(), True))
            cpu_infer.sync()
            cpu_infer.submit(moe_amx.backward_task(
                grad_out.data_ptr(), grad_in_amx.data_ptr(),
                grad_gate_lora_a.data_ptr(), grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(), grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(), grad_down_lora_b.data_ptr(),
                grad_weights.data_ptr()))
            cpu_infer.sync()

        amx_fwd_times = []
        amx_bwd_times = []
        for _ in range(test_iter):
            t0 = time.perf_counter()
            cpu_infer.submit(moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(), k,
                expert_ids.data_ptr(), weights_moe.data_ptr(),
                input_data.data_ptr(), output_amx.data_ptr(), True))
            cpu_infer.sync()
            t1 = time.perf_counter()
            cpu_infer.submit(moe_amx.backward_task(
                grad_out.data_ptr(), grad_in_amx.data_ptr(),
                grad_gate_lora_a.data_ptr(), grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(), grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(), grad_down_lora_b.data_ptr(),
                grad_weights.data_ptr()))
            cpu_infer.sync()
            t2 = time.perf_counter()
            amx_fwd_times.append(t1 - t0)
            amx_bwd_times.append(t2 - t1)

        amx_fwd_times.sort()
        amx_bwd_times.sort()
        amx_fwd = amx_fwd_times[len(amx_fwd_times) // 2]
        amx_bwd = amx_bwd_times[len(amx_bwd_times) // 2]

        # ════════════════════════════════════════════════
        #  Torch: forward + backward (TN layout)
        # ════════════════════════════════════════════════
        torch.set_num_threads(16)
        grad_out_pt = grad_out.clone()

        for _ in range(warmup_iter):
            inp = input_data.clone().requires_grad_(True)
            out = moe_torch(inp, expert_ids, weights_moe, torch_experts)
            out.backward(grad_out_pt)
            for e in torch_experts:
                for p in e.parameters():
                    if p.grad is not None:
                        p.grad = None

        torch_fwd_times = []
        torch_bwd_times = []
        for _ in range(test_iter):
            inp = input_data.clone().requires_grad_(True)
            t0 = time.perf_counter()
            out = moe_torch(inp, expert_ids, weights_moe, torch_experts)
            t1 = time.perf_counter()
            out.backward(grad_out_pt)
            t2 = time.perf_counter()
            torch_fwd_times.append(t1 - t0)
            torch_bwd_times.append(t2 - t1)
            for e in torch_experts:
                for p in e.parameters():
                    if p.grad is not None:
                        p.grad = None

        torch.set_num_threads(NUM_THREADS)  # restore for AMX

        torch_fwd_times.sort()
        torch_bwd_times.sort()
        torch_fwd = torch_fwd_times[len(torch_fwd_times) // 2]
        torch_bwd = torch_bwd_times[len(torch_bwd_times) // 2]

        amx_tot = amx_fwd + amx_bwd
        torch_tot = torch_fwd + torch_bwd
        fwd_spd = torch_fwd / amx_fwd if amx_fwd > 0 else 0
        bwd_spd = torch_bwd / amx_bwd if amx_bwd > 0 else 0
        tot_spd = torch_tot / amx_tot if amx_tot > 0 else 0

        results.append({
            'qlen': qlen,
            'amx_fwd_ms': round(amx_fwd * 1000, 3),
            'amx_bwd_ms': round(amx_bwd * 1000, 3),
            'amx_tot_ms': round(amx_tot * 1000, 3),
            'torch_fwd_ms': round(torch_fwd * 1000, 3),
            'torch_bwd_ms': round(torch_bwd * 1000, 3),
            'torch_tot_ms': round(torch_tot * 1000, 3),
            'fwd_speedup': round(fwd_spd, 2),
            'bwd_speedup': round(bwd_spd, 2),
            'tot_speedup': round(tot_spd, 2),
        })

        print(f"{qlen:>6} | "
              f"{amx_fwd*1000:>7.2f}ms {amx_bwd*1000:>7.2f}ms {amx_tot*1000:>7.2f}ms | "
              f"{torch_fwd*1000:>8.2f}ms {torch_bwd*1000:>8.2f}ms {torch_tot*1000:>8.2f}ms | "
              f"{fwd_spd:>6.2f}x {bwd_spd:>6.2f}x {tot_spd:>6.2f}x")

    # ─── Summary table (forward vs backward ratio) ───
    print(f"\n{'─'*60}")
    print("AMX backward/forward ratio (>1 means backward is slower):")
    print(f"{'qlen':>6} | {'AMX bwd/fwd':>11} | {'Torch bwd/fwd':>13}")
    print(f"{'─'*6}-+-{'─'*11}-+-{'─'*13}")
    for r in results:
        amx_ratio = r['amx_bwd_ms'] / r['amx_fwd_ms'] if r['amx_fwd_ms'] > 0 else 0
        torch_ratio = r['torch_bwd_ms'] / r['torch_fwd_ms'] if r['torch_fwd_ms'] > 0 else 0
        print(f"{r['qlen']:>6} | {amx_ratio:>10.2f}x | {torch_ratio:>12.2f}x")

    output = {
        'config': {
            'expert_num': E, 'hidden_size': H, 'intermediate_size': I,
            'n_routed_experts': k, 'num_threads': NUM_THREADS,
            'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
        },
        'description': (
            'Both AMX and Torch paths compute base SwiGLU + LoRA (same workload). '
            'AMX uses pre-packed BufferB (VNNI block format) for forward GEMMs. '
            'Backward requires transposed weight access. Torch uses standard '
            'row-major TN layout for both directions via autograd.'
        ),
        'results': results,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'bench_prepack_vs_torch_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_benchmark()
