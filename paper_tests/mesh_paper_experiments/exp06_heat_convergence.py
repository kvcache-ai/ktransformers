#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import iter_jsonl, summarize_cache_stats_jsonl, write_csv, write_json


def timeline_for(label: str, path: str) -> list[dict]:
    cumulative: dict[int, tuple[int, int]] = {}
    out = []
    for obj in iter_jsonl(path):
        layer = int(obj.get("layer", -1))
        if layer < 0:
            continue
        hit = int(obj.get("hit_count", 0) or 0)
        miss = int(obj.get("miss_count", 0) or 0)
        cumulative[layer] = (hit, miss)
        total_hit = sum(v[0] for v in cumulative.values())
        total_miss = sum(v[1] for v in cumulative.values())
        out.append(
            {
                "label": label,
                "source": path,
                "event_idx": len(out),
                "layer": layer,
                "dump_tick": int(obj.get("dump_tick", 0) or 0),
                "total_hit": total_hit,
                "total_miss": total_miss,
                "hit_rate": total_hit / (total_hit + total_miss) if total_hit + total_miss > 0 else None,
                "iouring_read_mib": sum(
                    int(o.get("iouring_read_bytes", 0) or 0)
                    for o in [obj]
                )
                / 1048576.0,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hit-rate convergence curves from MESH cache stats JSONL.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--stats",
        action="append",
        default=[],
        help="label:path, for example sieve:/tmp/sieve/cache_stats.jsonl",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    summaries = {}
    for item in args.stats:
        if ":" not in item:
            raise ValueError("--stats must be label:path")
        label, path = item.split(":", 1)
        rows.extend(timeline_for(label, path))
        summaries[label] = summarize_cache_stats_jsonl(path)

    write_csv(outdir / "heat_convergence.csv", rows)
    write_json(outdir / "heat_convergence_summary.json", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
