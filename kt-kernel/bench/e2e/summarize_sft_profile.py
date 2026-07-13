#!/usr/bin/env python3
"""Summarize one-step KT SFT profiling artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TOKEN_FIELDS = ("label_tokens_global", "attention_tokens_global", "input_tokens_global")
PROFILE_RE = re.compile(
    r"\[(?P<tag>KT_K2_SFT_FWD_PROFILE|KT_K2_SFT_PROFILE|KT_K2_SFT_TP_PROFILE)\]\s+(?P<body>.*)"
)


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1.0 - frac) + values[high] * frac


def token_count(row: dict[str, Any]) -> int | None:
    for field in TOKEN_FIELDS:
        if row.get(field) is not None:
            return int(row[field])
    return None


def summarize_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing = [row for row in rows if row.get("record_type") == "training_step_timing"]
    samples = []
    for row in timing:
        forward_ms = float(row.get("forward_ms", 0.0))
        backward_ms = float(row.get("backward_ms", 0.0))
        step_ms = forward_ms + backward_ms
        tokens = token_count(row)
        samples.append({
            "global_step_before": row.get("global_step_before"),
            "sync_gradients": row.get("sync_gradients"),
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "step_ms": step_ms,
            "forward_pct": 100.0 * forward_ms / step_ms if step_ms > 0 else None,
            "backward_pct": 100.0 * backward_ms / step_ms if step_ms > 0 else None,
            "tokens": tokens,
            "tps": tokens / (step_ms / 1000.0) if tokens is not None and step_ms > 0 else None,
        })
    return {"count": len(samples), "samples": samples}


def summarize_python(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") != "python_phase":
            continue
        key = (row.get("direction"), row.get("phase"), row.get("layer_idx"), row.get("rank"))
        grouped[key].append(float(row.get("elapsed_ms", 0.0)))
    summaries = []
    for (direction, phase, layer_idx, rank), values in grouped.items():
        summaries.append({
            "direction": direction,
            "phase": phase,
            "layer_idx": layer_idx,
            "rank": rank,
            "count": len(values),
            "total_ms": sum(values),
            "mean_ms": statistics.fmean(values),
            "p50_ms": percentile(values, 50),
            "max_ms": max(values),
        })
    summaries.sort(key=lambda row: row["total_ms"], reverse=True)
    return {"group_count": len(summaries), "top": summaries[:limit]}


def coerce(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_profile_line(line: str) -> dict[str, Any] | None:
    match = PROFILE_RE.search(line)
    if not match:
        return None
    row: dict[str, Any] = {
        "record_type": "cxx_profile",
        "direction": "forward" if match.group("tag") == "KT_K2_SFT_FWD_PROFILE" else "backward",
    }
    for part in match.group("body").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        row[key] = coerce(value)
    row["profile_kind"] = (
        "tp_wrapper"
        if match.group("tag") == "KT_K2_SFT_TP_PROFILE"
        else ("tp_shard" if "tp_part" in row else "single_tp")
    )
    return row


def read_cxx_profiles(log_path: Path | None) -> list[dict[str, Any]]:
    if log_path is None or not log_path.exists():
        return []
    rows = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = parse_profile_line(line)
            if row is not None:
                rows.append(row)
    return rows


def summarize_cxx(rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[str(row.get("direction"))].append(row)
    out: dict[str, Any] = {"row_count": len(rows)}
    for direction, direction_rows in by_direction.items():
        field_totals: dict[str, float] = defaultdict(float)
        total_values = []
        for row in direction_rows:
            for key, value in row.items():
                if key.endswith("_us") and isinstance(value, (int, float)):
                    field_totals[key] += float(value) / 1000.0
            if isinstance(row.get("total_us"), (int, float)):
                total_values.append(float(row["total_us"]) / 1000.0)
        fields = sorted(
            ({"field": key, "total_ms": value} for key, value in field_totals.items()),
            key=lambda item: item["total_ms"],
            reverse=True,
        )
        out[direction] = {
            "row_count": len(direction_rows),
            "total_us_rows_ms": sum(total_values),
            "total_us_p50_ms": percentile(total_values, 50),
            "top_fields": fields[:limit],
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-jsonl", type=Path, required=True)
    parser.add_argument("--python-jsonl", type=Path, default=None)
    parser.add_argument("--train-log", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timing_rows = read_jsonl(args.timing_jsonl)
    python_rows = read_jsonl(args.python_jsonl)
    cxx_rows = read_cxx_profiles(args.train_log)
    summary = {
        "schema": "kt_sft_one_step_profile_summary_v1",
        "timing_jsonl": str(args.timing_jsonl),
        "python_jsonl": str(args.python_jsonl) if args.python_jsonl else None,
        "train_log": str(args.train_log) if args.train_log else None,
        "timing": summarize_timing(timing_rows),
        "python_profile": summarize_python(python_rows, args.top),
        "cxx_profile": summarize_cxx(cxx_rows, args.top),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
