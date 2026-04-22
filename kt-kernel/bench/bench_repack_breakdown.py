#!/usr/bin/env python3
"""
Benchmark: Measure repack (layout conversion) vs compute (GEMM) time breakdown
as a function of sequence length in AMX SFT MoE forward/backward.

Generates sft_trace.json via the built-in tracing infrastructure, plus a
metadata JSON for the plotter to correlate trace events with seqlens.

Usage:
    cd /path/to/kt-kernel
    SFT_TRACE_PATH=bench/repack_trace.json python3 bench/bench_repack_breakdown.py
"""
import os
import time
import json

# Set trace path BEFORE importing extension (read by std::call_once in init_trace)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_PATH = os.path.join(SCRIPT_DIR, 'repack_trace.json')
os.environ['SFT_TRACE_PATH'] = TRACE_PATH

import torch

from _load_kt_kernel import load_local_kt_kernel

kt_kernel_ext = load_local_kt_kernel().kt_kernel_ext

# ─── Config (Qwen3-30B-A3B) ───
EXPERT_NUM = 8
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
N_ROUTED_EXPERTS = 8
LORA_RANK = 8
LORA_ALPHA = 32
MAX_LEN = 8192 + 64
NUM_THREADS = 64

SEQLENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
WARMUP_ITERS = 3
WARMUP_QLEN = 64


def run():
    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(42)

    H, I, E, k, r = HIDDEN_SIZE, INTERMEDIATE_SIZE, EXPERT_NUM, N_ROUTED_EXPERTS, LORA_RANK

    print(f"Config: E={E}, H={H}, I={I}, k={k}, r={r}, threads={NUM_THREADS}")
    print(f"Trace will be written to: {TRACE_PATH}")

    # ─── Weights ───
    gate_proj = torch.randn(E, I, H, dtype=torch.bfloat16).contiguous()
    up_proj = torch.randn_like(gate_proj)
    down_proj = torch.randn(E, H, I, dtype=torch.bfloat16).contiguous()
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
    print("AMX weights pre-packed.\n")

    def make_inputs(qlen):
        expert_ids = torch.stack(
            [torch.randperm(E, dtype=torch.int64)[:k] for _ in range(qlen)]).contiguous()
        weights_moe = torch.rand(qlen, k, dtype=torch.float32).contiguous()
        weights_moe = weights_moe / weights_moe.sum(dim=-1, keepdim=True)
        input_data = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()
        grad_out = (torch.randn(qlen, H, dtype=torch.bfloat16) / 10).contiguous()
        return expert_ids, weights_moe, input_data, grad_out

    def run_fwd_bwd(qlen, expert_ids, weights_moe, input_data, grad_out):
        bsz_tensor = torch.tensor([qlen], device="cpu")
        output = torch.zeros(qlen, H, dtype=torch.bfloat16).contiguous()
        grad_in = torch.zeros_like(input_data)
        grad_gla = torch.zeros_like(gate_lora_a)
        grad_glb = torch.zeros_like(gate_lora_b)
        grad_ula = torch.zeros_like(up_lora_a)
        grad_ulb = torch.zeros_like(up_lora_b)
        grad_dla = torch.zeros_like(down_lora_a)
        grad_dlb = torch.zeros_like(down_lora_b)
        grad_w = torch.zeros(qlen, k, dtype=torch.float32).contiguous()

        t0 = time.perf_counter()
        cpu_infer.submit(moe_amx.forward_sft_task(
            bsz_tensor.data_ptr(), k,
            expert_ids.data_ptr(), weights_moe.data_ptr(),
            input_data.data_ptr(), output.data_ptr(), True))
        cpu_infer.sync()
        t1 = time.perf_counter()
        cpu_infer.submit(moe_amx.backward_task(
            grad_out.data_ptr(), grad_in.data_ptr(),
            grad_gla.data_ptr(), grad_glb.data_ptr(),
            grad_ula.data_ptr(), grad_ulb.data_ptr(),
            grad_dla.data_ptr(), grad_dlb.data_ptr(),
            grad_w.data_ptr()))
        cpu_infer.sync()
        t2 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000

    # ─── Warmup (generates trace events we'll skip) ───
    print(f"Warming up ({WARMUP_ITERS} iters at qlen={WARMUP_QLEN})...")
    wup = make_inputs(WARMUP_QLEN)
    for _ in range(WARMUP_ITERS):
        run_fwd_bwd(WARMUP_QLEN, *wup)
    print("Warmup done.\n")

    # ─── Measurement: 1 fwd+bwd per seqlen ───
    results = []
    print(f"{'qlen':>6} | {'fwd (ms)':>10} {'bwd (ms)':>10} {'bwd/fwd':>8}")
    print("-" * 44)

    for qlen in SEQLENS:
        inp = make_inputs(qlen)
        fwd_ms, bwd_ms = run_fwd_bwd(qlen, *inp)
        ratio = bwd_ms / fwd_ms if fwd_ms > 0 else 0
        results.append({
            'qlen': qlen,
            'fwd_ms': round(fwd_ms, 3),
            'bwd_ms': round(bwd_ms, 3),
        })
        print(f"{qlen:>6} | {fwd_ms:>9.2f}ms {bwd_ms:>9.2f}ms {ratio:>7.2f}x")

    # ─── Save metadata ───
    meta = {
        'config': {
            'expert_num': E, 'hidden_size': H, 'intermediate_size': I,
            'num_experts_per_tok': k, 'lora_rank': r,
            'num_threads': NUM_THREADS, 'torch_version': torch.__version__,
        },
        'warmup_iters': WARMUP_ITERS,
        'warmup_qlen': WARMUP_QLEN,
        'seqlens': SEQLENS,
        'results': results,
        'trace_path': TRACE_PATH,
    }
    meta_path = os.path.join(SCRIPT_DIR, 'repack_breakdown_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")
    print(f"Trace will be written to {TRACE_PATH} on exit.")


if __name__ == '__main__':
    run()
