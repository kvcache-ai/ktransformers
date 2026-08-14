#!/usr/bin/env python
"""
Verify the per-layer quantize->release mechanism for GGUF->AMXINT8 loading.

Root cause being validated: ~AMX_MOE_BASE() was `= default`, so the
aligned_alloc'd BufferB weight blocks (~3.6GB/layer at MiniMax scale:
256 experts x 3 matrices x (1536x3072 int8 + scales)) were leaked every
time a layer's MoE object was destroyed. The per-layer load loop therefore
accumulated ~3.6GB/layer and OOM'd around layer 60 of 61.

Each iteration reproduces one layer of the GGUF->AMXINT8 load:
  1. allocate BF16 source tensors (what the GGUF adapter produces)
  2. create AMXInt8_MOE (BufferB aligned_alloc in ctor)
  3. load_weights_task() -> C++ from_mat() writes every BufferB page
     (this is what makes the memory resident, exactly like real quantize)
  4. destroy the MoE object (wrapper dropped / del self.moe)
  RSS must return to baseline after each destroy (sawtooth), not grow.

Usage:
    python test_moe_dtor_release.py [--realsize] [--n 20]
"""
import argparse
import gc
import os
import sys

import torch

# kt_kernel/__init__.py aliases kt_kernel_ext into sys.modules — import it first.
import kt_kernel  # noqa: F401
import kt_kernel_ext
from kt_kernel_ext import CPUInfer, WorkerPoolConfig
from kt_kernel_ext.moe import AMXInt8_MOE, MOEConfig


def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        pages = int(f.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)


def peak_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return float(line.split()[1]) / (1024 ** 2)
    return -1.0


def make_pool(threads: int = 16):
    wc = WorkerPoolConfig()
    wc.subpool_count = 1
    wc.subpool_thread_count = [threads]
    wc.subpool_numa_map = [0]
    return CPUInfer(wc)


def quantize_one_layer(pool, layer_idx, num_experts, hidden, inter, k, max_len, out_dir):
    """Exactly mirrors AMXMoEWrapper GGUF branch: BF16 -> INT8 -> keep moe."""
    mask = torch.zeros(num_experts, dtype=torch.bool)

    # BF16 source tensors (the online-quant path the GGUF branch feeds)
    gate = torch.zeros(num_experts, inter, hidden, dtype=torch.bfloat16)
    up = torch.zeros(num_experts, inter, hidden, dtype=torch.bfloat16)
    down = torch.zeros(num_experts, hidden, inter, dtype=torch.bfloat16)

    cfg = MOEConfig(num_experts, k, hidden, inter, mask.data_ptr())
    cfg.layer_idx = layer_idx
    cfg.pool = pool.backend_
    cfg.max_len = max_len
    cfg.save = False
    cfg.load = False
    cfg.path = out_dir
    cfg.gate_proj = gate.data_ptr()
    cfg.up_proj = up.data_ptr()
    cfg.down_proj = down.data_ptr()

    moe = AMXInt8_MOE(cfg)
    phys_map = torch.arange(num_experts, dtype=torch.int64)
    # mirror AMXMoEWrapper.load_weights_from_tensors:
    #   self.cpu_infer.submit(self.moe.load_weights_task(ptr)); self.cpu_infer.sync()
    pool.submit(moe.load_weights_task(phys_map.data_ptr()))
    pool.sync()

    # Free the BF16 source exactly like load_weights() does after sync
    del gate, up, down
    gc.collect()
    return moe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realsize", action="store_true", help="MiniMax-scale dims (256 experts, 1536x3072)")
    ap.add_argument("--n", type=int, default=20, help="layers to quantize/release")
    args = ap.parse_args()

    if args.realsize:
        NE, HIDDEN, INTER, K, MAX_LEN = 256, 3072, 1536, 8, 512
    else:
        NE, HIDDEN, INTER, K, MAX_LEN = 64, 1024, 512, 8, 512

    per_layer_gb = NE * (2 * INTER * HIDDEN + HIDDEN * INTER + (2 * INTER + HIDDEN) * 4) / (1024 ** 3)
    print(f"[test] dims: experts={NE} hidden={HIDDEN} inter={INTER}")
    print(f"[test] expected resident BufferB per layer: ~{per_layer_gb:.2f} GiB")

    pool = make_pool()
    base = rss_gb()
    print(f"[test] baseline RSS: {base:.2f} GiB")

    # Phase 1: quantize layers, KEEPING moe objects (runtime behavior: wrappers persist)
    held = []
    peak = base
    for i in range(args.n):
        moe = quantize_one_layer(pool, i, NE, HIDDEN, INTER, K, MAX_LEN, "/tmp/kt_moe_dtor_test")
        held.append(moe)
        r = rss_gb()
        peak = max(peak, r)
        if i % 5 == 4 or i == args.n - 1:
            print(f"[test] layer {i + 1:>3} quantized, {len(held)} layers resident: RSS {r:8.2f} GiB")

    resident = rss_gb()
    print(f"[test] ALL {args.n} layers resident: RSS {resident:.2f} GiB (peak {peak:.2f} GiB)")
    print(f"[test]   expected resident: ~{base + args.n * per_layer_gb:.1f} GiB")

    # Phase 2: destroy all (process/convert teardown) — RSS must return to baseline
    del moe  # loop variable still holds the last created object
    held.clear()
    gc.collect()
    freed = rss_gb()
    print(f"[test] destroyed all {args.n} MoE objects: RSS {freed:.2f} GiB (drop {resident - freed:+.2f} GiB)")
    # The shared scratch arena (m_local/pools via shared_mem_buffer_numa) is
    # process-lifetime by design — it stays resident. What must return is the
    # per-layer BufferB: expect drop ~= n * per_layer_gb.
    # NOTE: only enforceable when each matrix block is large enough for glibc
    # to munmap on free (empirically ~1MB+ blocks drop; ~0.5MB blocks stay in
    # the brk arena, reusable but not returned to the OS). MiniMax's real
    # blocks are 1536x3072 int8 = 4.7MB -> enforced.
    matrix_block_mb = (INTER * HIDDEN) / (1024 ** 2)
    drop_enforceable = matrix_block_mb >= 1.0
    drop_ok = (resident - freed) >= 0.9 * args.n * per_layer_gb if drop_enforceable else True
    if drop_enforceable:
        print(f"[test] matrix block {matrix_block_mb:.2f} MB -> RSS-drop check enforced")
    else:
        print(f"[test] matrix block {matrix_block_mb:.2f} MB -> small-scale, RSS-drop check skipped "
              f"(glibc arena retention; reuse verified by phase 3)")

    # Phase 3: quantize+destroy per layer (convert-script behavior) — RSS must be a sawtooth, not a ramp
    base3 = rss_gb()
    peak3 = base3
    for i in range(args.n):
        moe = quantize_one_layer(pool, 100 + i, NE, HIDDEN, INTER, K, MAX_LEN, "/tmp/kt_moe_dtor_test")
        del moe  # rebinding next iteration releases the previous ref; final one freed below
        gc.collect()
        peak3 = max(peak3, rss_gb())
    if "moe" in locals():
        del moe
        gc.collect()
    r3 = rss_gb()
    print(f"[test] phase-3 quantize+destroy x{args.n}: RSS {base3:.2f} -> {r3:.2f} GiB "
          f"(peak {peak3:.2f}, growth {r3 - base3:+.2f} GiB)")
    print(f"[test] VmHWM (hard peak committed): {peak_gb():.2f} GiB")

    # Without the destructor fix, phase 3 would grow by ~n * per_layer_gb
    # (the leak). With it, growth must be ~0 (arena stays, buffers reused).
    ok = drop_ok and (r3 - base3) < 0.3 * args.n * per_layer_gb
    print(f"[test] RESULT: {'PASS - per-layer release works' if ok else 'FAIL - memory still accumulates'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()