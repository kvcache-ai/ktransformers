#!/usr/bin/env python3
"""
Minimal LLAMAFILE repro harness to catch intermittent RuntimeError/RE.

Requirements:
- kt_kernel_ext built with LLAMAFILE (and CUDA stream integration)
- Valid GGUF weights directory (WEIGHT_PATH)

Usage:
  WEIGHT_PATH=/path/to/gguf python examples/repro_llamafile_re.py

Optional env:
  DEVICE=cuda|cpu           # default: auto (cuda if available)
  N_ITERS=1000              # iterations
  BATCH=4                   # batch size
  H=2048                    # hidden size
  EXPERTS=128               # total experts
  TOPK=8                    # experts per token
  INTER=768                 # intermediate size (must be divisible by 256)
  GPU_EXPERTS=100           # num experts on GPU side
  TP=2                      # threadpool_count
  CPU_THREADS=32            # cpuinfer_threads
  MAX_DEFER=2               # max_deferred_experts_per_token
  MODE=split|forward        # split=submit+sync, forward=wrapper.forward
  SEED=1                    # random seed

Debug tips:
  - Set CUDA_LAUNCH_BLOCKING=1 to catch async errors deterministically.
  - Try varying N_ITERS, BATCH, TOPK, MAX_DEFER.
  - Capture stdout/stderr for failure iteration index.
"""

from __future__ import annotations

import os
import sys
import faulthandler
import torch

from kt_kernel import KTMoEWrapper


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def get_stream_for(device: torch.device | str):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.current_stream(device).cuda_stream
    return 0


def main() -> int:
    faulthandler.enable()

    weight_path = (os.environ.get("WEIGHT_PATH") or "").strip()
    if not weight_path:
        print("ERROR: WEIGHT_PATH env is required.")
        return 2
    if not os.path.exists(weight_path):
        print(f"ERROR: WEIGHT_PATH does not exist: {weight_path}")
        return 2

    device_str = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    n_iters = getenv_int("N_ITERS", 1000)
    batch = getenv_int("BATCH", 4)
    hidden = getenv_int("H", 2048)
    experts = getenv_int("EXPERTS", 128)
    topk = getenv_int("TOPK", 8)
    inter = getenv_int("INTER", 768)
    gpu_experts = getenv_int("GPU_EXPERTS", 100)
    tp = getenv_int("TP", 2)
    cpu_threads = getenv_int("CPU_THREADS", 32)
    max_defer = getenv_int("MAX_DEFER", 2)
    seed = getenv_int("SEED", 1)
    mode = (os.environ.get("MODE") or "split").lower()

    if inter % 256 != 0:
        print(f"ERROR: INTER must be divisible by 256 for LLAMAFILE (got {inter}).")
        return 2

    print(
        f"LLAMAFILE Repro: device={device}, iters={n_iters}, batch={batch}, H={hidden}, topk={topk}, E={experts}, inter={inter}, TP={tp}, CPU_THREADS={cpu_threads}, mode={mode}"
    )
    print(f"Weights: {weight_path}")

    torch.manual_seed(seed)

    # Create wrapper and load weights once
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        num_gpu_experts=gpu_experts,
        cpuinfer_threads=cpu_threads,
        threadpool_count=tp,
        weight_path=weight_path,
        chunked_prefill_size=512,
        method="LLAMAFILE",
        max_deferred_experts_per_token=max_defer,
    )
    wrapper.load_weights()

    # Optional capture of small batch sizes
    KTMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])

    stream = get_stream_for(device)

    # Allocate once and reuse to reduce allocator noise
    hidden_states = torch.empty(batch, hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.empty(batch, topk, dtype=torch.long, device=device)
    topk_weights = torch.empty(batch, topk, dtype=torch.float32, device=device)

    def fill_random():
        hidden_states.normal_(mean=0.0, std=1.0)
        topk_ids.random_(0, experts)
        topk_weights.uniform_()
        topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True) + 1e-6)

    # Warmup
    fill_random()
    _ = wrapper.forward(hidden_states, topk_ids, topk_weights, stream)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Main loop
    for i in range(n_iters):
        try:
            fill_random()
            if mode == "forward":
                _ = wrapper.forward(hidden_states, topk_ids, topk_weights, stream)
            else:
                wrapper.submit_forward(hidden_states, topk_ids, topk_weights, stream)
                # Optional small GPU op to put work on the same stream
                if device.type == "cuda":
                    hidden_states.add_(0)  # no-op but enqueued on current stream
                _ = wrapper.sync_forward(hidden_states, stream)

            if (i + 1) % 50 == 0:
                print(f"ok: iter {i + 1}/{n_iters}")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        except Exception as e:
            print(f"FAIL at iter {i}: {repr(e)}")
            # Flush GPU work for better diagnostics
            if device.type == "cuda":
                try:
                    torch.cuda.synchronize(device)
                except Exception as _:
                    pass
            return 1

    print("All iterations completed without error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

