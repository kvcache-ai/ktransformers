#!/usr/bin/env python3
"""Same-input GPTQ INT4 MoE correctness/latency benchmark (requires a SYCL build).

Run under the kt-kernel Python environment after sourcing oneAPI. No HTTP/token
count estimates: times include CPUInfer submit, complete MoE execution and sync.
"""

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from kt_kernel import kt_kernel_ext as ext


def packed_weights(experts, n, k, gs, generator):
    # Generate stored GPTQ nibbles directly, without a resident FP32 weight copy.
    q = torch.randint(0, 16, (experts, k // 8, n, 8), generator=generator, dtype=torch.int64)
    shifts = torch.arange(8) * 4
    packed = (q << shifts).sum(-1).to(torch.int32).contiguous()
    scales = (torch.rand(experts, k // gs, n, generator=generator) * 0.003 + 0.002).contiguous()
    return packed, scales


def dequant(packed, scales, gs):
    k8, n = packed.shape
    shifts = torch.arange(8) * 4
    values = ((packed.long().unsqueeze(-1) >> shifts) & 15) - 8
    return values.permute(0, 2, 1).reshape(k8 * 8, n).float() * scales.repeat_interleave(gs, 0)


def reference(x, ids, routing, projections, gs, mask=None, physical_map=None, limit=0.0, alpha=0.0):
    result = torch.zeros_like(x, dtype=torch.float32)
    for e in range(projections[0][0].shape[0]):
        logical = int(physical_map[e]) if physical_map is not None else e
        if mask is not None and mask[logical]:
            continue
        where = (ids == e).nonzero()
        if not len(where):
            continue
        tokens, slots = where[:, 0], where[:, 1]
        source = x[tokens].float()
        gate, up, down = [dequant(p[0][logical], p[1][logical], gs) for p in projections]
        # Baseline converts gate/up to BF16 before activation, then down to BF16.
        g = (source @ gate).bfloat16().float()
        u = (source @ up).bfloat16().float()
        if limit > 0:
            g = g.clamp(-limit, limit) if alpha > 0 else g.clamp(max=limit)
            u = u.clamp(-limit, limit)
        mid = (
            (g * torch.sigmoid(g * alpha) * (u + 1) if alpha > 0 else torch.nn.functional.silu(g) * u)
            .bfloat16()
            .float()
        )
        y = (mid @ down).bfloat16().float() * routing[tokens, slots, None]
        result.index_add_(0, tokens, y)
    return result.bfloat16().float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backends", default="legacy,packed,xmx-fp16,xmx-int8")
    p.add_argument("--rows", default="1,16,64,512")
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=512)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--model-path", type=Path, help="Use the actual GPTQ model weights instead of random weights")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--route-sets", type=int, default=1, help="Rotate expert assignments; 1 measures fixed-route reuse")
    p.add_argument("--swiglu-limit", type=float, default=0.0)
    p.add_argument("--swiglu-alpha", type=float, default=0.0)
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--edge-cases", action="store_true", help="duplicate/invalid IDs, mask, permutation and incremental output"
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    backends = args.backends.split(",")
    if not set(backends) <= {
        "legacy",
        "packed",
        "avx2",
        "xmx-fp16",
        "xmx-int8",
        "vector",
        "dp4a",
        "xmx-dp4a",
        "dp4a-xmx",
        "vector-xmx",
        "xmx-vector",
        "xmx-split-xmx",
    }:
        p.error("Unknown backend")
    if args.repeat <= 0 or args.warmup < 0 or args.route_sets <= 0 or not 0 < args.topk <= args.experts:
        p.error("Invalid repetitions or expert count")
    if min(args.hidden, args.intermediate, args.group_size, args.threads) <= 0:
        p.error("Dimensions, group size and thread count must be positive")
    if args.hidden % args.group_size or args.intermediate % args.group_size:
        p.error("Projection dimensions must be divisible by group size")
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(20261014)
    rows_list = [int(n) for n in args.rows.split(",")]
    if not rows_list or min(rows_list) <= 0:
        p.error("Rows must be positive")
    if args.model_path:
        from safetensors import safe_open

        index = json.loads((args.model_path / "model.safetensors.index.json").read_text())["weight_map"]
        actual = []
        for name in ("gate", "up", "down"):
            tensors = [[], []]
            for expert in range(args.experts):
                stem = f"model.language_model.layers.{args.layer}.mlp.experts.{expert}.{name}_proj"
                for dest, suffix in zip(tensors, ("qweight", "scales")):
                    key = stem + "." + suffix
                    with safe_open(args.model_path / index[key], framework="pt", device="cpu") as f:
                        t = f.get_tensor(key)
                        dest.append(t.float().contiguous() if suffix == "scales" else t.clone())
            actual.append(tuple(torch.stack(ts) for ts in tensors))
        proj = actual
    else:
        proj = [
            packed_weights(args.experts, args.intermediate, args.hidden, args.group_size, generator),
            packed_weights(args.experts, args.intermediate, args.hidden, args.group_size, generator),
            packed_weights(args.experts, args.hidden, args.intermediate, args.group_size, generator),
        ]
    cases = []
    for rows in rows_list:
        x = (torch.randn(rows, args.hidden, generator=generator) * 0.5).bfloat16().contiguous()
        ids = torch.rand(rows, args.experts, generator=generator).argsort(1)[:, : args.topk].contiguous()
        routing = torch.softmax(torch.randn(rows, args.topk, generator=generator), 1).contiguous()
        if args.edge_cases:
            ids[:, 0] = -1
            if args.topk > 2:
                ids[:, 2] = ids[:, 1]
        cases.append((rows, x, ids, routing))
    mask = torch.zeros(args.experts, dtype=torch.bool)
    mapping = torch.arange(args.experts, dtype=torch.int64)
    if args.edge_cases:
        mask[::4] = True
        mapping = torch.randperm(args.experts, generator=generator)
    refs = (
        {
            r: reference(x, ids, w, proj, args.group_size, mask, mapping, args.swiglu_limit, args.swiglu_alpha)
            for r, x, ids, w in cases
        }
        if args.check
        else {}
    )
    cpu = ext.CPUInfer(args.threads)
    results = []
    for backend in backends:
        kernels = {
            "vector": ("vector", "vector"),
            "dp4a": ("dp4a", "dp4a"),
            "xmx-dp4a": ("xmx", "dp4a"),
            "dp4a-xmx": ("dp4a", "xmx"),
            "xmx-split-xmx": ("xmx-split", "xmx"),
            "vector-xmx": ("vector", "xmx"),
            "xmx-vector": ("xmx", "vector"),
        }.get(backend, ("xmx", "xmx"))
        os.environ["KT_SYCL_XMX_GATE_KERNEL"], os.environ["KT_SYCL_XMX_DOWN_KERNEL"] = kernels
        os.environ["KT_SYCL_XMX_MODE"] = (
            "fp16" if backend in ("vector", "xmx-fp16", "legacy", "packed", "avx2") else "int8"
        )
        cls = {"legacy": "SYCLGPTQInt4_MOE", "packed": "AVXVNNI256GPTQInt4Packed_MOE", "avx2": "AVX2GPTQInt4_MOE"}.get(
            backend, "SYCLXMXGPTQInt4_MOE"
        )
        config = ext.moe.MOEConfig(args.experts, args.topk, args.hidden, args.intermediate, mask.data_ptr())
        config.pool = cpu.backend_
        config.max_len = max(rows_list)
        config.quant_config.bits = 4
        config.quant_config.group_size = args.group_size
        config.quant_config.zero_point = False
        config.swiglu_limit = args.swiglu_limit
        config.swiglu_alpha = args.swiglu_alpha
        for name, (qw, sc) in zip(("gate", "up", "down"), proj):
            setattr(config, name + "_projs", [[t.data_ptr() for t in qw]])
            setattr(config, name + "_scales", [[t.data_ptr() for t in sc]])
        moe = getattr(ext.moe, cls)(config)
        cpu.submit(moe.load_weights_task(mapping.data_ptr()))
        cpu.sync()
        for rows, x, ids, weights in cases:
            out = torch.zeros_like(x)
            qlen = torch.tensor([rows], dtype=torch.int32)
            routes = [
                torch.where(ids >= 0, (ids + index * args.topk) % args.experts, ids).contiguous()
                for index in range(args.route_sets)
            ]

            def run(incremental=False, route=0):
                cpu.submit(
                    moe.forward_task(
                        qlen.data_ptr(),
                        args.topk,
                        routes[route].data_ptr(),
                        weights.data_ptr(),
                        x.data_ptr(),
                        out.data_ptr(),
                        incremental,
                    )
                )
                cpu.sync()

            for _ in range(args.warmup):
                run()
            times = []
            for index in range(args.repeat):
                start = time.perf_counter_ns()
                run(route=index % args.route_sets)
                times.append((time.perf_counter_ns() - start) / 1e6)
            record = {
                "backend": backend,
                "rows": rows,
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "samples_ms": times,
            }
            record["kernels"] = {
                key: os.getenv(key)
                for key in (
                    "KT_SYCL_XMX_MODE",
                    "KT_SYCL_XMX_GATE_KERNEL",
                    "KT_SYCL_XMX_DOWN_KERNEL",
                    "KT_SYCL_XMX_DECODE_GATE_KERNEL",
                    "KT_SYCL_XMX_DECODE_DOWN_KERNEL",
                    "KT_SYCL_XMX_GATE_SPLITS",
                    "KT_SYCL_XMX_DOWN_SPLITS",
                    "KT_SYCL_XMX_DECODE_MERGE",
                    "KT_SYCL_XMX_TRACE",
                    "KT_SYCL_XMX_FUSED_EXPERT",
                    "KT_SYCL_XMX_PREFILL_ROWS",
                    "KT_SYCL_XMX_PREFILL_DOWN_ROWS",
                    "KT_SYCL_XMX_PREFILL_N",
                    "KT_SYCL_XMX_PREFILL_LOAD",
                    "KT_SYCL_XMX_PREFILL_MERGE",
                )
            }
            if args.check:
                run()  # Compare the base route, separately from timed requests.
                ref = refs[rows]
                error = out.float() - ref
                record.update(
                    relative_rmse=(error.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)).item(),
                    max_abs_error=error.abs().max().item(),
                    finite=bool(torch.isfinite(out).all()),
                )
                quantized = backend not in ("legacy", "avx2", "vector", "xmx-fp16")
                if not record["finite"] or record["relative_rmse"] > (0.08 if quantized else 0.025):
                    raise AssertionError(f"correctness failed: {record}")
                if args.edge_cases and backend not in ("legacy", "packed", "avx2"):
                    previous = out.clone()
                    run(incremental=True)
                    expected = (previous.float() * 2).bfloat16()
                    torch.testing.assert_close(out, expected, rtol=0.02, atol=0.002)
            print(json.dumps(record), flush=True)
            results.append(record)
        del moe
        gc.collect()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        config = vars(args) | {
            "output": str(args.output),
            "model_path": str(args.model_path) if args.model_path else None,
        }
        environment = {
            key: os.getenv(key)
            for key in (
                "KT_SYCL_XMX_MODE",
                "KT_SYCL_XMX_ROWS",
                "KT_SYCL_XMX_GRAPH",
                "KT_SYCL_XMX_HOST_QUANT",
                "KT_SYCL_XMX_FUSED_QUANT",
                "KT_SYCL_XMX_NUMA_DISPATCH",
                "KT_SYCL_XMX_GATE_KERNEL",
                "KT_SYCL_XMX_DOWN_KERNEL",
                "KT_SYCL_XMX_DECODE_GATE_KERNEL",
                "KT_SYCL_XMX_DECODE_DOWN_KERNEL",
                "KT_SYCL_XMX_VECTOR_SPLITS",
                "KT_SYCL_XMX_GATE_SPLITS",
                "KT_SYCL_XMX_DOWN_SPLITS",
                "KT_SYCL_XMX_PREFILL_ROWS",
                "KT_SYCL_XMX_PREFILL_DOWN_ROWS",
                "KT_SYCL_XMX_PREFILL_N",
                "KT_SYCL_XMX_PREFILL_LOAD",
                "KT_SYCL_XMX_PREFILL_MERGE",
                "KT_SYCL_XMX_DECODE_MERGE",
                "KT_SYCL_XMX_FUSED_EXPERT",
                "KT_SYCL_XMX_TRACE",
                "KT_SYCL_XMX_TRACE_EVERY",
                "KT_SYCL_PROFILE",
                "ONEAPI_DEVICE_SELECTOR",
            )
        }
        args.output.write_text(json.dumps({"config": config, "environment": environment, "results": results}, indent=2))


if __name__ == "__main__":
    main()
