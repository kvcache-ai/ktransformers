#!/usr/bin/env python3
"""
Benchmark: Optimizer step performance with fragmented vs fused MoE LoRA parameters.

Demonstrates the Base-LoRA layout conflict:
  - Forward/backward produces per-expert LoRA gradients as many small tensors
  - Optimizer prefers consolidated, contiguous, batch-processable state
  - With E experts × 3 projections × 2 matrices × L layers = 6EL individual params
  - KT fuses these into 6L contiguous [E, ...] buffer params

X-axis: number of experts
Y-axis: optimizer.step() time (ms)

Simulates real Qwen3-MoE dimensions:
  H=7168, I=2048, r=8, variable layers and experts.
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn


def create_fragmented_params(num_experts, num_layers, H, I, r, dtype=torch.bfloat16):
    """Create individual per-expert LoRA parameters (vanilla PEFT style).
    Total: 6 * num_experts * num_layers parameters.
    """
    params = []
    for _layer in range(num_layers):
        for _expert in range(num_experts):
            params.append(nn.Parameter(torch.randn(r, H, dtype=dtype, device="cpu")))
            params.append(nn.Parameter(torch.randn(I, r, dtype=dtype, device="cpu")))
            params.append(nn.Parameter(torch.randn(r, H, dtype=dtype, device="cpu")))
            params.append(nn.Parameter(torch.randn(I, r, dtype=dtype, device="cpu")))
            params.append(nn.Parameter(torch.randn(r, I, dtype=dtype, device="cpu")))
            params.append(nn.Parameter(torch.randn(H, r, dtype=dtype, device="cpu")))
    return params


def create_fused_params(num_experts, num_layers, H, I, r, dtype=torch.bfloat16):
    """Create KT-style fused buffer parameters.
    Total: 6 * num_layers parameters (independent of num_experts).
    """
    params = []
    for _layer in range(num_layers):
        params.append(nn.Parameter(torch.randn(num_experts, r, H, dtype=dtype, device="cpu")))
        params.append(nn.Parameter(torch.randn(num_experts, I, r, dtype=dtype, device="cpu")))
        params.append(nn.Parameter(torch.randn(num_experts, r, H, dtype=dtype, device="cpu")))
        params.append(nn.Parameter(torch.randn(num_experts, I, r, dtype=dtype, device="cpu")))
        params.append(nn.Parameter(torch.randn(num_experts, r, I, dtype=dtype, device="cpu")))
        params.append(nn.Parameter(torch.randn(num_experts, H, r, dtype=dtype, device="cpu")))
    return params


def total_elements(params):
    return sum(p.numel() for p in params)


def fill_grads(params):
    """Simulate backward pass: fill all grads with random data."""
    for p in params:
        if p.grad is None:
            p.grad = torch.randn_like(p)
        else:
            p.grad.normal_()


def bench_optimizer_step(params, optimizer, warmup=2, iters=5):
    """Benchmark optimizer.step() time."""
    for _ in range(warmup):
        fill_grads(params)
        optimizer.step()

    times = []
    for _ in range(iters):
        fill_grads(params)
        t0 = time.perf_counter()
        optimizer.step()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def bench_grad_clip(params, max_norm=1.0, warmup=2, iters=5):
    """Benchmark gradient clipping."""
    for _ in range(warmup):
        fill_grads(params)
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    times = []
    for _ in range(iters):
        fill_grads(params)
        t0 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def run_one_config(E, L, H, I, r, foreach, warmup, iters):
    """Run fragmented vs fused for one expert count. Returns dict."""
    row = {"experts": E, "layers": L, "foreach": foreach}

    # --- Fragmented ---
    params_frag = create_fragmented_params(E, L, H, I, r)
    n_elem_frag = total_elements(params_frag)
    opt_frag = torch.optim.AdamW(params_frag, lr=1e-4, foreach=foreach)
    n_frag = len(params_frag)

    step_frag = bench_optimizer_step(params_frag, opt_frag, warmup, iters)
    clip_frag = bench_grad_clip(params_frag, warmup=warmup, iters=iters)

    row["fragmented_n_params"] = n_frag
    row["fragmented_n_elements"] = n_elem_frag
    row["fragmented_step_ms"] = min(step_frag)
    row["fragmented_step_median_ms"] = sorted(step_frag)[len(step_frag) // 2]
    row["fragmented_clip_ms"] = min(clip_frag)
    del params_frag, opt_frag

    # --- Fused (KT-style) ---
    params_fused = create_fused_params(E, L, H, I, r)
    n_elem_fused = total_elements(params_fused)
    opt_fused = torch.optim.AdamW(params_fused, lr=1e-4, foreach=foreach)
    n_fused = len(params_fused)

    step_fused = bench_optimizer_step(params_fused, opt_fused, warmup, iters)
    clip_fused = bench_grad_clip(params_fused, warmup=warmup, iters=iters)

    row["fused_n_params"] = n_fused
    row["fused_n_elements"] = n_elem_fused
    row["fused_step_ms"] = min(step_fused)
    row["fused_step_median_ms"] = sorted(step_fused)[len(step_fused) // 2]
    row["fused_clip_ms"] = min(clip_fused)
    del params_fused, opt_fused

    assert n_elem_frag == n_elem_fused, \
        f"Element mismatch: frag={n_elem_frag} vs fused={n_elem_fused}"

    return row


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=str, default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--inter", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--threads", type=int, default=0, help="0 = use default")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    expert_counts = [int(x) for x in args.experts.split(",")]
    H, I, r, L = args.hidden, args.inter, args.rank, args.layers
    threads = torch.get_num_threads()

    print(f"Config: H={H}, I={I}, r={r}, layers={L}")
    print(f"Expert counts: {expert_counts}")
    print(f"Per-expert elements: {2*(r*H) + 2*(I*r) + (r*I) + (H*r):,}")
    print(f"Torch threads: {threads}")
    print(f"Torch version: {torch.__version__}")
    print()

    all_results = {}

    for foreach in [True, False]:
        label = "foreach" if foreach else "no-foreach"
        print(f"=== AdamW foreach={foreach} ===")
        print(f"{'experts':>7} | {'params':>7} {'fused':>6} | "
              f"{'elements':>12} | "
              f"{'frag step':>10} {'fused step':>10} {'speedup':>8} | "
              f"{'frag clip':>10} {'fused clip':>10}")
        print("-" * 105)

        results = []
        for E in expert_counts:
            row = run_one_config(E, L, H, I, r, foreach, args.warmup, args.iters)
            results.append(row)

            spd = row["fragmented_step_ms"] / max(row["fused_step_ms"], 0.01)
            print(f"{E:>7} | {row['fragmented_n_params']:>7} {row['fused_n_params']:>6} | "
                  f"{row['fragmented_n_elements']:>12,} | "
                  f"{row['fragmented_step_ms']:>9.1f}ms {row['fused_step_ms']:>9.1f}ms {spd:>7.1f}x | "
                  f"{row['fragmented_clip_ms']:>9.1f}ms {row['fused_clip_ms']:>9.1f}ms")

        all_results[label] = results
        print()

    # Save
    output = {
        "config": {
            "hidden_size": H, "intermediate_size": I, "lora_rank": r,
            "num_layers": L, "torch_threads": threads,
            "warmup": args.warmup, "iters": args.iters,
            "torch_version": torch.__version__,
        },
        "results_foreach": all_results.get("foreach", []),
        "results_no_foreach": all_results.get("no-foreach", []),
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "bench_optimizer_step_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
