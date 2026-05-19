#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    IostatCapture,
    ensure_dir,
    format_template,
    parse_iostat_jsonl,
    parse_runtime_logs,
    run_command,
    summarize_cache_stats_jsonl,
    summarize_iostat,
    write_csv,
    write_json,
)


def run_variant(args, variant: str, template: str, budget_gb: int, repeat: int) -> dict:
    run_dir = ensure_dir(Path(args.outdir) / f"{variant}_budget{budget_gb}g_r{repeat}")
    cmd = format_template(
        template,
        run_dir=run_dir,
        variant=variant,
        budget_gb=budget_gb,
        repeat=repeat,
        policy=args.policy,
        workload=args.workload,
    )
    iostat_path = run_dir / "iostat_raw.jsonl"
    if args.capture_iostat:
        with IostatCapture(iostat_path, interval_s=args.iostat_interval, devices=args.iostat_devices):
            meta = run_command(cmd, run_dir, timeout=args.timeout, prefix="run")
    else:
        meta = run_command(cmd, run_dir, timeout=args.timeout, prefix="run")

    runtime = parse_runtime_logs([run_dir / "run.stdout.txt", run_dir / "run.stderr.txt"])
    iostat_summary = summarize_iostat(parse_iostat_jsonl(iostat_path)) if iostat_path.exists() else {}
    cache_stats_path = run_dir / "cache_stats.jsonl"
    cache = summarize_cache_stats_jsonl(cache_stats_path) if cache_stats_path.exists() else {}
    row = {
        "variant": variant,
        "budget_gb": budget_gb,
        "repeat": repeat,
        "run_dir": str(run_dir),
        "returncode": meta["returncode"],
        "elapsed_s": meta["elapsed_s"],
        "prefill_tokens_logged": runtime.get("prefill_tokens_logged"),
        "prefill_tps_hmean": runtime.get("prefill_tps_hmean"),
        "decode_tps_avg": runtime.get("decode_tps_avg"),
        "decode_tps_last": runtime.get("decode_tps_last"),
        "cache_hit_rate": cache.get("hit_rate"),
        "iouring_read_mib": cache.get("iouring_read_mib"),
        "iouring_read_request_count": cache.get("iouring_read_request_count"),
        "iostat_read_mib_s_avg": iostat_summary.get("read_mib_s_avg"),
        "iostat_read_mib_s_peak": iostat_summary.get("read_mib_s_peak"),
        "iostat_util_pct_peak": iostat_summary.get("util_pct_peak"),
    }
    write_json(run_dir / "paper_run_summary.json", row)
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["variant"], int(row["budget_gb"])), []).append(row)
    out = []
    for (variant, budget), vals in sorted(groups.items()):
        def avg(key: str):
            xs = [float(v[key]) for v in vals if v.get(key) not in (None, "")]
            return sum(xs) / len(xs) if xs else None

        out.append(
            {
                "variant": variant,
                "budget_gb": budget,
                "runs": len(vals),
                "decode_tps_avg": avg("decode_tps_avg"),
                "prefill_tps_hmean": avg("prefill_tps_hmean"),
                "cache_hit_rate": avg("cache_hit_rate"),
                "iouring_read_mib": avg("iouring_read_mib"),
                "iostat_read_mib_s_avg": avg("iostat_read_mib_s_avg"),
                "iostat_read_mib_s_peak": avg("iostat_read_mib_s_peak"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MESH vs mmap memory-budget sweep for the residency turning point.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--budgets-gb", default="40,32,24,16,8")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--mesh-cmd-template", required=True)
    parser.add_argument("--mmap-cmd-template", required=True)
    parser.add_argument("--policy", default="sieve")
    parser.add_argument("--workload", default="LongBench")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--capture-iostat", action="store_true")
    parser.add_argument("--iostat-interval", type=int, default=1)
    parser.add_argument("--iostat-devices", nargs="*", default=None)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    budgets = [int(x) for x in args.budgets_gb.split(",") if x.strip()]
    rows = []
    for budget in budgets:
        for repeat in range(args.repeats):
            rows.append(run_variant(args, "mesh", args.mesh_cmd_template, budget, repeat))
            rows.append(run_variant(args, "mmap", args.mmap_cmd_template, budget, repeat))
            write_csv(Path(args.outdir) / "turning_point_runs.csv", rows)

    summary = aggregate(rows)
    write_csv(Path(args.outdir) / "turning_point_summary.csv", summary)
    write_json(Path(args.outdir) / "turning_point_summary.json", {"runs": rows, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
