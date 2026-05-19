#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ensure_dir, format_template, parse_runtime_logs, run_command, summarize_cache_stats_jsonl, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MESH replacement-policy sweep.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cmd-template", required=True)
    parser.add_argument("--policies", default="sieve,s3fifo,slru,wtinylfu,lru,roundrobin")
    parser.add_argument("--budget-gb", type=int, default=24)
    parser.add_argument("--workload", default="LongBench")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    rows = []
    for policy in [x.strip() for x in args.policies.split(",") if x.strip()]:
        for repeat in range(args.repeats):
            run_dir = ensure_dir(outdir / f"{policy}_r{repeat}")
            cmd = format_template(
                args.cmd_template,
                run_dir=run_dir,
                policy=policy,
                budget_gb=args.budget_gb,
                workload=args.workload,
                repeat=repeat,
                variant="mesh",
            )
            meta = run_command(cmd, run_dir, timeout=args.timeout, prefix="run")
            runtime = parse_runtime_logs([run_dir / "run.stdout.txt", run_dir / "run.stderr.txt"])
            stats_path = run_dir / "cache_stats.jsonl"
            stats = summarize_cache_stats_jsonl(stats_path) if stats_path.exists() else {}
            row = {
                "policy": policy,
                "repeat": repeat,
                "budget_gb": args.budget_gb,
                "workload": args.workload,
                "run_dir": str(run_dir),
                "returncode": meta["returncode"],
                "elapsed_s": meta["elapsed_s"],
                "decode_tps_avg": runtime.get("decode_tps_avg"),
                "prefill_tps_hmean": runtime.get("prefill_tps_hmean"),
                "hit_rate": stats.get("hit_rate"),
                "iouring_read_mib": stats.get("iouring_read_mib"),
            }
            rows.append(row)
            write_csv(outdir / "policy_sweep_runs.csv", rows)

    summary = []
    for policy in sorted({r["policy"] for r in rows}):
        vals = [r for r in rows if r["policy"] == policy]
        def avg(key):
            xs = [float(v[key]) for v in vals if v.get(key) not in (None, "")]
            return sum(xs) / len(xs) if xs else None
        summary.append(
            {
                "policy": policy,
                "runs": len(vals),
                "decode_tps_avg": avg("decode_tps_avg"),
                "prefill_tps_hmean": avg("prefill_tps_hmean"),
                "hit_rate": avg("hit_rate"),
                "iouring_read_mib": avg("iouring_read_mib"),
            }
        )
    write_csv(outdir / "policy_sweep_summary.csv", summary)
    write_json(outdir / "policy_sweep_summary.json", {"runs": rows, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
