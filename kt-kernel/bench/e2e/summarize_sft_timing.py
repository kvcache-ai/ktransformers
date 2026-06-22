#!/usr/bin/env python3
"""Summarize KT_E2E_TIMING_JSONL into SFT throughput metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


TOKEN_FIELDS = ("label_tokens_global", "attention_tokens_global", "input_tokens_global")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty value list")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1.0 - frac) + values[high] * frac


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") == "training_step_timing":
                rows.append(row)
    return rows


def token_count(row: dict[str, Any], field: str | None, fallback: int | None) -> int | None:
    if field is not None:
        value = row.get(field)
        return int(value) if value is not None else fallback
    for candidate in TOKEN_FIELDS:
        value = row.get(candidate)
        if value is not None:
            return int(value)
    return fallback


def summarize(rows: list[dict[str, Any]], token_field: str | None, fallback_tokens: int | None) -> dict[str, Any]:
    samples = []
    for row in rows:
        tokens = token_count(row, token_field, fallback_tokens)
        if tokens is None:
            continue
        step_ms = float(row["forward_ms"]) + float(row["backward_ms"])
        if step_ms <= 0:
            continue
        samples.append(
            {
                "tokens": tokens,
                "forward_ms": float(row["forward_ms"]),
                "backward_ms": float(row["backward_ms"]),
                "step_ms": step_ms,
                "tps": tokens / (step_ms / 1000.0),
                "sync_gradients": bool(row.get("sync_gradients", False)),
                "global_step_before": int(row.get("global_step_before", -1)),
            }
        )

    if not samples:
        raise ValueError("no usable timing rows; pass --tokens-per-micro-step for old logs without token fields")

    tps_values = [sample["tps"] for sample in samples]
    forward = [sample["forward_ms"] for sample in samples]
    backward = [sample["backward_ms"] for sample in samples]
    step = [sample["step_ms"] for sample in samples]
    tokens = [sample["tokens"] for sample in samples]
    total_tokens = sum(tokens)
    total_time_s = sum(step) / 1000.0

    return {
        "sample_count": len(samples),
        "total_tokens": total_tokens,
        "total_time_s": total_time_s,
        "aggregate_tps": total_tokens / total_time_s if total_time_s > 0 else None,
        "per_micro_step_tps_mean": statistics.fmean(tps_values),
        "per_micro_step_tps_p50": percentile(tps_values, 50),
        "per_micro_step_tps_p90": percentile(tps_values, 90),
        "forward_ms_mean": statistics.fmean(forward),
        "forward_ms_p50": percentile(forward, 50),
        "backward_ms_mean": statistics.fmean(backward),
        "backward_ms_p50": percentile(backward, 50),
        "step_ms_mean": statistics.fmean(step),
        "step_ms_p50": percentile(step, 50),
        "tokens_per_micro_step_mean": statistics.fmean(tokens),
        "sync_gradient_sample_count": sum(1 for sample in samples if sample["sync_gradients"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="KT_E2E_TIMING_JSONL file")
    parser.add_argument("--drop-first", type=int, default=1, help="Warmup rows to skip before summarizing")
    parser.add_argument(
        "--token-field",
        choices=TOKEN_FIELDS,
        default=None,
        help="Token field to use; defaults to label, then attention, then input tokens.",
    )
    parser.add_argument(
        "--tokens-per-micro-step",
        type=int,
        default=None,
        help="Fallback token count for old logs that do not include token fields.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.jsonl)
    used_rows = rows[args.drop_first :]
    summary = {
        "schema": "kt_sft_timing_summary_v1",
        "jsonl": str(args.jsonl),
        "drop_first": args.drop_first,
        "token_field": args.token_field or "auto",
        "tokens_per_micro_step_fallback": args.tokens_per_micro_step,
        "raw_row_count": len(rows),
        **summarize(used_rows, args.token_field, args.tokens_per_micro_step),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
