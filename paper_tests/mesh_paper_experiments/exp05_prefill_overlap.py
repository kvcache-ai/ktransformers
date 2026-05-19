#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from common import parse_iostat_jsonl, summarize_iostat, write_csv, write_json


def f(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MESH prefill compute/IO overlap from trace CSV.")
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--iostat-jsonl", default="")
    parser.add_argument("--tp", default="0")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.trace_csv).open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if args.tp != "all" and str(row.get("tp")) != args.tp:
                continue
            if str(row.get("is_prefill_chunk", "1")) not in ("1", "true", "True"):
                continue
            rows.append(row)

    by_layer = defaultdict(list)
    for row in rows:
        by_layer[int(float(row.get("layer", 0)))].append(row)

    layer_rows = []
    totals = defaultdict(float)
    for layer, vals in sorted(by_layer.items()):
        compute_ms = sum(f(r, "compute_ms") for r in vals)
        static_compute_ms = sum(f(r, "static_compute_ms") for r in vals)
        scratch_compute_ms = sum(f(r, "scratch_compute_ms") for r in vals)
        scratch_wait_ms = sum(f(r, "scratch_wait_ms") for r in vals)
        scratch_submit_ms = sum(f(r, "scratch_submit_ms") for r in vals)
        scratch_reader_ms = sum(f(r, "scratch_reader_total_ms") for r in vals)
        read_bytes = sum(f(r, "expected_scratch_read_bytes") for r in vals)
        active = sum(f(r, "active") for r in vals)
        scratch_active = sum(f(r, "scratch_active") for r in vals)
        layer_row = {
            "layer": layer,
            "events": len(vals),
            "compute_ms": compute_ms,
            "static_compute_ms": static_compute_ms,
            "scratch_compute_ms": scratch_compute_ms,
            "scratch_wait_ms": scratch_wait_ms,
            "scratch_submit_ms": scratch_submit_ms,
            "scratch_reader_total_ms": scratch_reader_ms,
            "expected_read_mib": read_bytes / 1048576.0,
            "active_sum": active,
            "scratch_active_sum": scratch_active,
            "io_wait_pct_of_compute_plus_wait": scratch_wait_ms / (compute_ms + scratch_wait_ms) * 100.0
            if compute_ms + scratch_wait_ms > 0
            else 0.0,
        }
        layer_rows.append(layer_row)
        for key, value in layer_row.items():
            if isinstance(value, (int, float)) and key not in ("layer", "io_wait_pct_of_compute_plus_wait"):
                totals[key] += float(value)

    iostat_summary = {}
    if args.iostat_jsonl:
        iostat_summary = summarize_iostat(parse_iostat_jsonl(args.iostat_jsonl))

    summary = {
        "trace_csv": args.trace_csv,
        "tp": args.tp,
        "events": len(rows),
        "totals": dict(totals),
        "iostat": iostat_summary,
    }
    compute_total = totals.get("compute_ms", 0.0)
    wait_total = totals.get("scratch_wait_ms", 0.0)
    summary["totals"]["io_wait_pct_of_compute_plus_wait"] = (
        wait_total / (compute_total + wait_total) * 100.0 if compute_total + wait_total > 0 else 0.0
    )
    write_csv(outdir / "prefill_overlap_by_layer.csv", layer_rows)
    write_json(outdir / "prefill_overlap_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
