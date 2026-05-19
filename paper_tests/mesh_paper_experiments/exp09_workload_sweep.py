#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ensure_dir, format_template, parse_runtime_logs, run_command, summarize_cache_stats_jsonl, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run workload generality sweep for MESH and optional mmap baseline.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mesh-cmd-template", required=True)
    parser.add_argument("--mmap-cmd-template", default="")
    parser.add_argument("--workloads", default="LongBench,HumanEval,MT-Bench")
    parser.add_argument("--budget-gb", type=int, default=24)
    parser.add_argument("--policy", default="sieve")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    rows = []
    variants = [("mesh", args.mesh_cmd_template)]
    if args.mmap_cmd_template:
        variants.append(("mmap", args.mmap_cmd_template))

    for workload in [x.strip() for x in args.workloads.split(",") if x.strip()]:
        for variant, template in variants:
            for repeat in range(args.repeats):
                run_dir = ensure_dir(outdir / f"{workload}_{variant}_r{repeat}".replace("/", "_"))
                cmd = format_template(
                    template,
                    run_dir=run_dir,
                    policy=args.policy,
                    budget_gb=args.budget_gb,
                    workload=workload,
                    repeat=repeat,
                    variant=variant,
                )
                meta = run_command(cmd, run_dir, timeout=args.timeout, prefix="run")
                runtime = parse_runtime_logs([run_dir / "run.stdout.txt", run_dir / "run.stderr.txt"])
                stats_path = run_dir / "cache_stats.jsonl"
                stats = summarize_cache_stats_jsonl(stats_path) if stats_path.exists() else {}
                row = {
                    "workload": workload,
                    "variant": variant,
                    "repeat": repeat,
                    "budget_gb": args.budget_gb,
                    "policy": args.policy,
                    "run_dir": str(run_dir),
                    "returncode": meta["returncode"],
                    "elapsed_s": meta["elapsed_s"],
                    "decode_tps_avg": runtime.get("decode_tps_avg"),
                    "prefill_tps_hmean": runtime.get("prefill_tps_hmean"),
                    "hit_rate": stats.get("hit_rate"),
                    "iouring_read_mib": stats.get("iouring_read_mib"),
                }
                rows.append(row)
                write_csv(outdir / "workload_sweep_runs.csv", rows)

    summary = []
    for workload in sorted({r["workload"] for r in rows}):
        for variant in sorted({r["variant"] for r in rows if r["workload"] == workload}):
            vals = [r for r in rows if r["workload"] == workload and r["variant"] == variant]
            def avg(key):
                xs = [float(v[key]) for v in vals if v.get(key) not in (None, "")]
                return sum(xs) / len(xs) if xs else None
            summary.append(
                {
                    "workload": workload,
                    "variant": variant,
                    "runs": len(vals),
                    "decode_tps_avg": avg("decode_tps_avg"),
                    "prefill_tps_hmean": avg("prefill_tps_hmean"),
                    "hit_rate": avg("hit_rate"),
                    "iouring_read_mib": avg("iouring_read_mib"),
                }
            )
    write_csv(outdir / "workload_sweep_summary.csv", summary)
    write_json(outdir / "workload_sweep_summary.json", {"runs": rows, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
