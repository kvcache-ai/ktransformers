#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import iter_jsonl, write_csv, write_json


def active_set(obj: dict, field: str) -> set[int]:
    if field in obj and isinstance(obj[field], list):
        return {int(x) for x in obj[field]}
    counts = obj.get("router_counts") or obj.get("cpu_counts") or []
    return {i for i, v in enumerate(counts) if int(v) > 0}


def count_items(obj: dict, field: str) -> list[tuple[int, int]]:
    counts = obj.get(field)
    if isinstance(counts, list):
        return [(i, int(v)) for i, v in enumerate(counts) if int(v) > 0]
    return [(i, 1) for i in active_set(obj, "active_experts")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze expert temporal locality and long-tail activation.")
    parser.add_argument("--trace-jsonl", required=True, help="KT_MESH_PREFILL_EXPERT_FREQ_PATH JSONL or compatible trace")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--field", default="active_experts")
    parser.add_argument("--count-field", default="cpu_counts")
    parser.add_argument("--max-distance", type=int, default=64)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[int, list[set[int]]] = defaultdict(list)
    longtail = Counter()
    for obj in iter_jsonl(args.trace_jsonl):
        layer = int(obj.get("layer", -1))
        if layer < 0:
            continue
        s = active_set(obj, args.field)
        if not s:
            continue
        by_layer[layer].append(s)
        for expert_id, count in count_items(obj, args.count_field):
            longtail[(layer, expert_id)] += count

    jac_rows = []
    for d in range(1, args.max_distance + 1):
        vals = []
        for layer, seq in by_layer.items():
            for i in range(0, max(0, len(seq) - d)):
                a = seq[i]
                b = seq[i + d]
                union = len(a | b)
                vals.append(len(a & b) / union if union else 0.0)
        jac_rows.append(
            {
                "distance": d,
                "pairs": len(vals),
                "jaccard_avg": sum(vals) / len(vals) if vals else None,
            }
        )

    counts = sorted(longtail.values(), reverse=True)
    total = sum(counts)
    cumulative = 0
    longtail_rows = []
    for rank, count in enumerate(counts, start=1):
        cumulative += count
        longtail_rows.append(
            {
                "rank": rank,
                "activation_count": count,
                "cumulative_count": cumulative,
                "cumulative_pct": cumulative / total * 100.0 if total else None,
                "rank_pct": rank / len(counts) * 100.0 if counts else None,
            }
        )

    summary = {
        "trace_jsonl": args.trace_jsonl,
        "layers": len(by_layer),
        "events": sum(len(v) for v in by_layer.values()),
        "distinct_layer_experts": len(longtail),
        "total_layer_expert_activations": total,
        "top_50pct_activation_share": next(
            (r["cumulative_pct"] for r in longtail_rows if r["rank_pct"] >= 50.0), None
        ),
    }
    write_csv(outdir / "temporal_jaccard.csv", jac_rows)
    write_csv(outdir / "expert_longtail.csv", longtail_rows)
    write_json(outdir / "temporal_locality_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
