#!/usr/bin/env python
# coding=utf-8
"""Benchmark MXFP4 MoE kernel — V4-Flash shape, mat-vec / mat-mat 双路径覆盖。

Synthesizes V4-Flash-shaped MXFP4 weights (random nibbles + bf16 group scales),
runs the chosen backend over a list of batch sizes M, prints a throughput table.

Routing modes (决定是否触发 mat-mat 路径):
    balanced     —— 每 token randperm(EXPERT_NUM)[:TOP_K]; 平均 per-expert m ≈
                    M*top_k/expert_num. V4 真实路由分布. 大 batch (M=1024) 才
                    平均触发 mat-mat (per-expert m ≥ 4).
    concentrated —— 所有 M token 共用同一组 top_k expert; per-expert m = M.
                    M=4 立即触发 mat-mat —— 用来直观放大 mat-mat 性能优势.

Usage:
    python bench/bench_fp4_moe.py --backend v1
    python bench/bench_fp4_moe.py --backend v1 --routing concentrated
    python bench/bench_fp4_moe.py --all --routing concentrated   # 所有可用 backend 对比

`--backend` 是预留扩展点; 当前编译只绑定 v1 (AMXFP4_KGroup_MOE)。要选 v2/v3
需要 ext_bindings 里加新绑定。`--all` 会自动检测哪些 backend 可用。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
from kt_kernel import kt_kernel_ext  # noqa: E402

# ----- V4-Flash MoE shape -----
HIDDEN = 4096
INTER = 2048
EXPERT_NUM = 256
TOP_K = 6
K_GROUP_SIZE = 32

# ----- bench knobs -----
# M=1024 时平均 per-expert m ≈ M*6/256 = 24, balanced 路由也能触发 mat-mat。
DEFAULT_M_LIST = [1, 4, 16, 64, 256, 1024]
WARMUP_ITER = 200
TEST_ITER = 2000

# ----- WorkerPool: 2 NUMA × 40 thread (matches bench_k2_moe_amx.py) -----
WORKER_NUMA = 2
WORKER_THREADS_PER_NUMA = 40

# ----- Backend registry: name → kt_kernel_ext.moe class (None = not bound) -----
BACKENDS = {
    "v1": getattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE", None),
    # 预留扩展点; 加新 backend 时在 ext_bindings 绑定后这里加一行即可。
    "v2": getattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE_V2", None),
}

# OCP MXFP4 (E2M1) codepoints — same order as the kernel's LUT.
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def quantize_mxfp4_tensor(weights: torch.Tensor, group_size: int):
    """[E, N, K] fp32/bf16 → packed nibbles uint8 [E, N, K/2] + bf16 scales [E, N, K/gs]."""
    w = weights.to(torch.float32)
    e, rows, cols = w.shape
    assert cols % group_size == 0 and cols % 2 == 0
    reshaped = w.view(e, rows, cols // group_size, group_size)
    max_abs = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 6.0).squeeze(-1)
    normalized = reshaped / scales.unsqueeze(-1)
    distances = torch.abs(normalized.unsqueeze(-1) - E2M1_VALUES.view(1, 1, 1, 1, 16))
    nibbles = distances.argmin(dim=-1).to(torch.uint8).view(e, rows, cols // 2, 2)
    lo = nibbles[..., 0]
    hi = nibbles[..., 1]
    packed = ((hi << 4) | lo).contiguous()  # uint8 [E, N, K/2]
    scales = scales.to(torch.bfloat16).contiguous()  # bf16 [E, N, K/gs]
    return packed, scales


def build_synth_weights():
    torch.manual_seed(0)
    gate = torch.randn((EXPERT_NUM, INTER, HIDDEN), dtype=torch.float32) / 100
    up = torch.randn((EXPERT_NUM, INTER, HIDDEN), dtype=torch.float32) / 100
    down = torch.randn((EXPERT_NUM, HIDDEN, INTER), dtype=torch.float32) / 100
    gw, gs = quantize_mxfp4_tensor(gate, K_GROUP_SIZE)
    uw, us = quantize_mxfp4_tensor(up, K_GROUP_SIZE)
    dw, ds = quantize_mxfp4_tensor(down, K_GROUP_SIZE)
    return {
        "gate_w": gw, "up_w": uw, "down_w": dw,
        "gate_s": gs, "up_s": us, "down_s": ds,
    }


def build_moe(backend: str, weights, cpu_infer):
    cls = BACKENDS.get(backend)
    if cls is None:
        raise RuntimeError(
            f"backend={backend} not bound in this build. Available: "
            f"{[k for k, v in BACKENDS.items() if v is not None]}"
        )
    cfg = kt_kernel_ext.moe.MOEConfig(EXPERT_NUM, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = max(DEFAULT_M_LIST)
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = K_GROUP_SIZE
    cfg.quant_config.zero_point = False
    cfg.gate_projs = [[t.data_ptr() for t in weights["gate_w"]]]
    cfg.up_projs = [[t.data_ptr() for t in weights["up_w"]]]
    cfg.down_projs = [[t.data_ptr() for t in weights["down_w"]]]
    cfg.gate_scales = [[t.data_ptr() for t in weights["gate_s"]]]
    cfg.up_scales = [[t.data_ptr() for t in weights["up_s"]]]
    cfg.down_scales = [[t.data_ptr() for t in weights["down_s"]]]
    moe = cls(cfg)
    p2l = torch.arange(EXPERT_NUM, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
    cpu_infer.sync()
    return moe


def make_expert_ids(M: int, routing: str) -> torch.Tensor:
    """[M, TOP_K] int64 (kernel forward_binding casts to const int64_t*)."""
    if routing == "concentrated":
        # 所有 M token 共用同组 top_k expert → per-expert m = M
        hot = torch.randperm(EXPERT_NUM)[:TOP_K]
        return hot.unsqueeze(0).expand(M, TOP_K).contiguous().to(torch.int64)
    # balanced: 每 token 独立 randperm
    return torch.stack(
        [torch.randperm(EXPERT_NUM)[:TOP_K] for _ in range(M)]
    ).to(torch.int64).contiguous()


def bench_one_m(moe, cpu_infer, M: int, routing: str):
    bsz = torch.tensor([M], dtype=torch.int32)
    expert_ids = make_expert_ids(M, routing)
    routing_w = torch.rand((M, TOP_K), dtype=torch.float32).contiguous()
    x = (torch.randn((M, HIDDEN), dtype=torch.bfloat16) / 100).contiguous()
    y = torch.empty((M, HIDDEN), dtype=torch.bfloat16).contiguous()

    for _ in range(WARMUP_ITER):
        cpu_infer.submit(moe.forward_task(
            bsz.data_ptr(), TOP_K, expert_ids.data_ptr(),
            routing_w.data_ptr(), x.data_ptr(), y.data_ptr(), False))
        cpu_infer.sync()

    start = time.perf_counter()
    for _ in range(TEST_ITER):
        cpu_infer.submit(moe.forward_task(
            bsz.data_ptr(), TOP_K, expert_ids.data_ptr(),
            routing_w.data_ptr(), x.data_ptr(), y.data_ptr(), False))
        cpu_infer.sync()
    total = time.perf_counter() - start

    per_iter_us = total / TEST_ITER * 1e6
    tok_per_s = M * TEST_ITER / total
    unique_e = int(torch.unique(expert_ids).numel())
    avg_m_per_expert = float(M * TOP_K) / max(unique_e, 1)
    return {
        "M": M, "iters": TEST_ITER, "total_s": total,
        "per_iter_us": per_iter_us, "tokens_per_s": tok_per_s,
        "unique_experts": unique_e, "avg_m_per_expert": avg_m_per_expert,
    }


def run_backend(backend: str, weights, cpu_infer, m_list, routing):
    print(f"\n[bench-fp4] backend={backend} routing={routing}")
    moe = build_moe(backend, weights, cpu_infer)
    rows = []
    for M in m_list:
        r = bench_one_m(moe, cpu_infer, M, routing)
        rows.append(r)
        print(f"  M={M:>5}  avg_m/expert={r['avg_m_per_expert']:>6.1f}  "
              f"per-iter={r['per_iter_us']:>9.1f} us  tok/s={r['tokens_per_s']:>9.1f}")
    return rows


def print_single_table(backend, rows, routing):
    print(f"\n=== Summary ({backend}, routing={routing}) ===")
    print(f"{'M':>5}  {'avg_m':>6}  {'per-iter us':>12}  {'tok/s':>10}")
    for r in rows:
        print(f"{r['M']:>5}  {r['avg_m_per_expert']:>6.1f}  "
              f"{r['per_iter_us']:>12.1f}  {r['tokens_per_s']:>10.1f}")


def print_compare_table(all_rows: dict, routing: str):
    backends = list(all_rows.keys())
    if len(backends) < 2:
        print_single_table(backends[0], all_rows[backends[0]], routing)
        return
    base = backends[0]
    print(f"\n=== {' vs '.join(backends)} (routing={routing}, base={base}) ===")
    header = f"{'M':>5}  {'avg_m':>6}"
    for be in backends:
        header += f"  {be + ' us':>10}"
    for be in backends[1:]:
        header += f"  {be + '/' + base:>8}"
    print(header)
    n_rows = len(all_rows[base])
    for i in range(n_rows):
        line = f"{all_rows[base][i]['M']:>5}  {all_rows[base][i]['avg_m_per_expert']:>6.1f}"
        for be in backends:
            line += f"  {all_rows[be][i]['per_iter_us']:>10.1f}"
        for be in backends[1:]:
            ratio = all_rows[be][i]['per_iter_us'] / all_rows[base][i]['per_iter_us']
            line += f"  {ratio:>8.3f}"
        print(line)


def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:
        return {"commit": None, "error": str(e)}


def get_system_info():
    info = {"node": platform.node(), "system": platform.system()}
    cpu_model = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info["cpu"] = cpu_model
    info["cpu_cores"] = os.cpu_count()
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=list(BACKENDS.keys()), default="v1",
                   help="单 backend 模式（被 --all 覆盖）。当前可用: "
                        + ",".join(k for k, v in BACKENDS.items() if v is not None))
    p.add_argument("--all", action="store_true",
                   help="跑所有已绑定的 backend 并打印对比表（自动跳过未绑定的）")
    p.add_argument("--routing", choices=["balanced", "concentrated"], default="balanced",
                   help="balanced=每 token randperm (V4 真实); "
                        "concentrated=所有 token 共用同组 top_k (per-expert m=M, 放大 mat-mat)")
    p.add_argument("--m-list", type=str, default=None,
                   help=f"Comma-separated M values, default: {','.join(map(str, DEFAULT_M_LIST))}")
    p.add_argument("--numa", type=int, default=WORKER_NUMA)
    p.add_argument("--threads-per-numa", type=int, default=WORKER_THREADS_PER_NUMA)
    args = p.parse_args()

    m_list = [int(x) for x in args.m_list.split(",")] if args.m_list else DEFAULT_M_LIST

    if args.all:
        backends = [k for k, v in BACKENDS.items() if v is not None]
        if not backends:
            raise RuntimeError("No MXFP4 backend bound in this build.")
        print(f"[bench-fp4] --all: detected backends = {backends}")
    else:
        if BACKENDS.get(args.backend) is None:
            raise RuntimeError(
                f"backend={args.backend} not bound. Available: "
                f"{[k for k, v in BACKENDS.items() if v is not None]}"
            )
        backends = [args.backend]

    print(f"[bench-fp4] shape=H{HIDDEN}/I{INTER}/E{EXPERT_NUM}/k{TOP_K}/gs{K_GROUP_SIZE}  routing={args.routing}")
    print(f"[bench-fp4] WorkerPool: numa={args.numa} threads_per_numa={args.threads_per_numa}")
    print(f"[bench-fp4] m_list: {m_list}")

    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = args.numa
    wp.subpool_numa_map = list(range(args.numa))
    wp.subpool_thread_count = [args.threads_per_numa] * args.numa
    cpu_infer = kt_kernel_ext.CPUInfer(wp)

    print("[bench-fp4] synthesizing MXFP4 weights …")
    weights = build_synth_weights()

    all_rows = {}
    for be in backends:
        all_rows[be] = run_backend(be, weights, cpu_infer, m_list, args.routing)

    if len(backends) > 1:
        print_compare_table(all_rows, args.routing)
    else:
        print_single_table(backends[0], all_rows[backends[0]], args.routing)

    # JSONL log
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_fp4_moe.jsonl")
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    git = get_git_commit()
    sys_info = get_system_info()
    with open(out_path, "a") as f:
        for be, rows in all_rows.items():
            record = {
                "backend": be,
                "routing": args.routing,
                "shape": {"hidden": HIDDEN, "inter": INTER, "expert_num": EXPERT_NUM,
                          "top_k": TOP_K, "k_group_size": K_GROUP_SIZE},
                "worker_pool": {"numa": args.numa, "threads_per_numa": args.threads_per_numa},
                "rows": rows,
                "git": git, "system": sys_info, "timestamp": ts,
            }
            f.write(json.dumps(record) + "\n")
    print(f"\n[bench-fp4] appended {len(backends)} record(s) → {out_path}")


if __name__ == "__main__":
    main()
