#!/usr/bin/env python3
"""
Microbenchmark: AMX SFT backward vs PyTorch backward for MoE expert layer.

Compares kt-kernel's AMX-optimized SFT backward (with fused LoRA gradients)
against PyTorch's CPU autograd backward (what PEFT uses).

AMX path: forward_sft (save_for_backward=True) + backward (fused base+LoRA grads)
PEFT path: PyTorch autograd forward+backward (per-expert SwiGLU + LoRA)

All base weights repacked in advance (covered by attention time in practice).

Usage:
    cd /path/to/kt-kernel
    python3 bench/bench_backward_amx_vs_torch.py
"""
import os, time, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from _load_kt_kernel import load_local_kt_kernel

kt_kernel_ext = load_local_kt_kernel().kt_kernel_ext

# ─── Model dimensions ───
EXPERT_NUM = 8
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS = 8       # = EXPERT_NUM → all experts routed
MAX_LEN = 8192 + 64       # must be aligned to M_STEP (32)
NUM_THREADS = 64

# ─── LoRA config ───
LORA_RANK = 8
LORA_ALPHA = 32
LORA_SCALING = LORA_ALPHA / LORA_RANK

SEQLENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


# ═══════════════════════════════════════════════════════════════════
#  Torch MoE + LoRA (PEFT-style): per-expert forward, full autograd
# ═══════════════════════════════════════════════════════════════════

class SwiGLUExpertLoRA(nn.Module):
    """Single expert: frozen base weights + LoRA adapters."""
    def __init__(self, hidden, inter, rank, scaling, gate_w, up_w, down_w):
        super().__init__()
        self.gate_w = nn.Parameter(gate_w, requires_grad=False)
        self.up_w = nn.Parameter(up_w, requires_grad=False)
        self.down_w = nn.Parameter(down_w, requires_grad=False)
        self.gate_A = nn.Linear(hidden, rank, bias=False, dtype=torch.bfloat16)
        self.gate_B = nn.Linear(rank, inter, bias=False, dtype=torch.bfloat16)
        self.up_A = nn.Linear(hidden, rank, bias=False, dtype=torch.bfloat16)
        self.up_B = nn.Linear(rank, inter, bias=False, dtype=torch.bfloat16)
        self.down_A = nn.Linear(inter, rank, bias=False, dtype=torch.bfloat16)
        self.down_B = nn.Linear(rank, hidden, bias=False, dtype=torch.bfloat16)
        nn.init.zeros_(self.gate_B.weight)
        nn.init.zeros_(self.up_B.weight)
        nn.init.zeros_(self.down_B.weight)
        self.s = scaling

    def forward(self, x):
        g = F.linear(x, self.gate_w) + self.gate_B(self.gate_A(x)) * self.s
        u = F.linear(x, self.up_w) + self.up_B(self.up_A(x)) * self.s
        a = F.silu(g) * u
        return F.linear(a, self.down_w) + self.down_B(self.down_A(a)) * self.s


def moe_torch_lora(x, expert_ids, weights, experts):
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

    # ─── LoRA weights (shared between AMX and torch for fair comparison) ───
    gate_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    gate_lora_b = torch.zeros(E, I, r, dtype=torch.bfloat16).contiguous()
    up_lora_a = (torch.randn(E, r, H, dtype=torch.bfloat16) / 100).contiguous()
    up_lora_b = torch.zeros(E, I, r, dtype=torch.bfloat16).contiguous()
    down_lora_a = (torch.randn(E, r, I, dtype=torch.bfloat16) / 100).contiguous()
    down_lora_b = torch.zeros(E, H, r, dtype=torch.bfloat16).contiguous()

    # Make LoRA B non-zero for realistic gradient flow
    gate_lora_b.normal_().div_(100)
    up_lora_b.normal_().div_(100)
    down_lora_b.normal_().div_(100)

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
    print("AMX weights repacked.")

    # ─── Torch+LoRA experts ───
    torch_experts = nn.ModuleList([
        SwiGLUExpertLoRA(H, I, r, LORA_SCALING, gate_proj[e], up_proj[e], down_proj[e])
        for e in range(E)])
    print("Setup complete.\n")

    hdr = (f"{'qlen':>6} | {'AMX+LoRA ms':>11} {'GFLOPS':>8} | "
           f"{'Torch+LoRA ms':>13} {'GFLOPS':>8} | {'Speedup':>7}")
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

        # Preallocate AMX buffers
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
        #  AMX SFT: forward_sft + backward (fused LoRA)
        # ════════════════════════════════════════════════

        # Warmup
        for _ in range(warmup_iter):
            cpu_infer.submit(moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(), k,
                expert_ids.data_ptr(), weights_moe.data_ptr(),
                input_data.data_ptr(), output_amx.data_ptr(),
                True))  # save_for_backward
            cpu_infer.sync()
            cpu_infer.submit(moe_amx.backward_task(
                grad_out.data_ptr(), grad_in_amx.data_ptr(),
                grad_gate_lora_a.data_ptr(), grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(), grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(), grad_down_lora_b.data_ptr(),
                grad_weights.data_ptr()))
            cpu_infer.sync()

        amx_times = []
        for _ in range(test_iter):
            # Forward (with cache for backward)
            cpu_infer.submit(moe_amx.forward_sft_task(
                bsz_tensor.data_ptr(), k,
                expert_ids.data_ptr(), weights_moe.data_ptr(),
                input_data.data_ptr(), output_amx.data_ptr(),
                True))
            cpu_infer.sync()

            t0 = time.perf_counter()
            cpu_infer.submit(moe_amx.backward_task(
                grad_out.data_ptr(), grad_in_amx.data_ptr(),
                grad_gate_lora_a.data_ptr(), grad_gate_lora_b.data_ptr(),
                grad_up_lora_a.data_ptr(), grad_up_lora_b.data_ptr(),
                grad_down_lora_a.data_ptr(), grad_down_lora_b.data_ptr(),
                grad_weights.data_ptr()))
            cpu_infer.sync()
            amx_times.append(time.perf_counter() - t0)

        amx_times.sort()
        amx_time = amx_times[len(amx_times) // 2]

        # ════════════════════════════════════════════════
        #  Torch + LoRA backward (retain_graph)
        # ════════════════════════════════════════════════
        grad_out_pt = grad_out.clone()

        # Warmup
        for _ in range(warmup_iter):
            inp = input_data.clone().requires_grad_(True)
            out = moe_torch_lora(inp, expert_ids, weights_moe, torch_experts)
            out.backward(grad_out_pt)
            for e in torch_experts:
                for p in e.parameters():
                    if p.grad is not None:
                        p.grad = None

        # Build graph once, backward many with retain_graph
        inp_pt = input_data.clone().requires_grad_(True)
        out_pt = moe_torch_lora(inp_pt, expert_ids, weights_moe, torch_experts)

        for _ in range(warmup_iter):
            if inp_pt.grad is not None: inp_pt.grad = None
            for e in torch_experts:
                for p in e.parameters():
                    if p.grad is not None: p.grad = None
            out_pt.backward(grad_out_pt, retain_graph=True)

        torch_times = []
        for _ in range(test_iter):
            if inp_pt.grad is not None: inp_pt.grad = None
            for e in torch_experts:
                for p in e.parameters():
                    if p.grad is not None: p.grad = None
            t0 = time.perf_counter()
            out_pt.backward(grad_out_pt, retain_graph=True)
            torch_times.append(time.perf_counter() - t0)

        del out_pt, inp_pt
        torch_times.sort()
        torch_time = torch_times[len(torch_times) // 2]

        # ─── GFLOPS ───
        # Base backward: 5 GEMMs per expert = 10 * q * k * H * I
        # LoRA backward: 3 projections × fwd+bwd matmuls
        base_flops = 10.0 * qlen * k * H * I
        lora_flops = 3.0 * 6 * 2 * qlen * k * r * (H + I)
        total_flops = base_flops + lora_flops

        amx_gflops = total_flops / amx_time / 1e9
        torch_gflops = total_flops / torch_time / 1e9
        speedup = torch_time / amx_time

        results.append({
            'qlen': qlen,
            'amx_lora_time_ms': round(amx_time * 1000, 3),
            'torch_lora_time_ms': round(torch_time * 1000, 3),
            'amx_gflops': round(amx_gflops, 2),
            'torch_gflops': round(torch_gflops, 2),
            'speedup': round(speedup, 2),
        })

        print(f"{qlen:>6} | {amx_time*1000:>9.3f}ms {amx_gflops:>7.1f} | "
              f"{torch_time*1000:>11.3f}ms {torch_gflops:>7.1f} | "
              f"{speedup:>6.2f}x")

    output = {
        'config': {
            'expert_num': E, 'hidden_size': H, 'intermediate_size': I,
            'n_routed_experts': k, 'num_threads': NUM_THREADS,
            'lora_rank': r, 'lora_alpha': LORA_ALPHA,
        },
        'results': results,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_backward_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    run_benchmark()
