#!/usr/bin/env python3
"""Run and parse K2 SFT packed backward profile smoke cases.

This harness intentionally reuses examples/test_k2_sft_moe_amx.py so every
profile sample is also guarded by the existing numerical checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


PROFILE_RE = re.compile(r"\[KT_K2_SFT_PROFILE\]\s+(?P<body>.*)")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_compare_metrics(value: str) -> list[str]:
    metrics = parse_csv_strings(value)
    for metric in metrics:
        parts = metric.split(":", 1)
        name = parts[-1].strip()
        scope = parts[0].strip() if len(parts) == 2 else None
        if not name or (scope is not None and not scope):
            raise ValueError(f"invalid compare metric: {metric!r}")
    return metrics


def _coerce_profile_value(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_profile_line(line: str) -> dict[str, int | float | str] | None:
    match = PROFILE_RE.search(line)
    if match is None:
        return None

    parsed: dict[str, int | float | str] = {}
    for item in match.group("body").split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key] = _coerce_profile_value(value)
    if not parsed:
        return None
    parsed["profile_kind"] = "tp_shard" if "tp_part" in parsed else "single_tp"
    return parsed


def parse_profiles(text: str) -> list[dict[str, int | float | str]]:
    return [profile for line in text.splitlines() if (profile := parse_profile_line(line)) is not None]


def load_jsonl_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def build_pythonpath(repo_root: Path, env: dict[str, str]) -> str:
    candidates = [
        repo_root / "build" / "lib.linux-x86_64-cpython-311",
        repo_root / "python",
    ]
    parts = [str(path) for path in candidates if path.exists()]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def run_case(repo_root: Path, args: argparse.Namespace, qlen: int, tp_count: int, repeat_idx: int) -> dict:
    env = os.environ.copy()
    env["KT_K2_SFT_PROFILE_PACKED_BWD"] = "1"
    env["PYTHONPATH"] = build_pythonpath(repo_root, env)

    cmd = [
        sys.executable,
        str(repo_root / "examples" / "test_k2_sft_moe_amx.py"),
        "--qlen",
        str(qlen),
        "--expert-num",
        str(args.expert_num),
        "--threads",
        str(args.threads),
        "--rank",
        str(args.rank),
        "--tp-count",
        str(tp_count),
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed_s = time.perf_counter() - started
    combined = "\n".join(part for part in (proc.stderr, proc.stdout) if part)
    profiles = parse_profiles(combined)

    case = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "command": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "repeat": repeat_idx,
        "qlen": qlen,
        "tp_count": tp_count,
        "rank": args.rank,
        "expert_num": args.expert_num,
        "threads": args.threads,
        "profiles": profiles,
    }
    if proc.returncode != 0:
        case["stdout_tail"] = proc.stdout[-4000:]
        case["stderr_tail"] = proc.stderr[-4000:]
    if not profiles:
        case["profile_error"] = "no profile lines parsed"
    if args.require_tp_profile and tp_count > 1 and not any("tp_part" in row for row in profiles):
        case["profile_error"] = "no TP shard profile lines parsed"
    return case


def summarize_profiles(cases: Iterable[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for case in cases:
        for row in case["profiles"]:
            key = (
                case["qlen"],
                case["tp_count"],
                case["rank"],
                row.get("profile_kind"),
                row.get("tp_part", "all"),
            )
            groups[key].append(row)

    summaries = []
    stage_keys = [
        "grad_weights_us",
        "down_us",
        "down_lora_grads_us",
        "down_route_us",
        "down_write_us",
        "down_base_us",
        "down_lora_bprop_us",
        "down_lora_a_us",
        "down_lora_b_us",
        "down_lora_matmat_du_dx_us",
        "down_lora_matmat_da_db_us",
        "activation_us",
        "gate_up_us",
        "gate_up_base_us",
        "gate_up_lora_u_us",
        "gate_up_lora_b_us",
        "gate_up_lora_b_write_us",
        "gate_up_lora_a_input_us",
        "gate_up_lora_matmat_du_dx_us",
        "gate_up_lora_matmat_da_db_us",
        "gate_up_write_us",
        "total_us",
    ]
    for (qlen, tp_count, rank, profile_kind, tp_part), rows in sorted(groups.items()):
        summary = {
            "qlen": qlen,
            "tp_count": tp_count,
            "rank": rank,
            "profile_kind": profile_kind,
            "tp_part": tp_part,
            "samples": len(rows),
        }
        for key in stage_keys:
            values = [float(row[key]) for row in rows if key in row]
            if values:
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_p50"] = median(values)
                summary[f"{key}_avg"] = mean(values)
                summary[f"{key}_max"] = max(values)
        summaries.append(summary)
    return summaries


def _summary_group_key(summary: dict) -> tuple:
    return (
        summary.get("qlen"),
        summary.get("tp_count"),
        summary.get("rank"),
        summary.get("profile_kind"),
        summary.get("tp_part", "all"),
    )


def _compare_metric_parts(metric: str) -> tuple[str | None, str]:
    if ":" not in metric:
        return None, metric
    scope, name = metric.split(":", 1)
    return scope, name


def _add_metric_scope(row: dict, scope: str | None) -> dict:
    if scope is not None:
        row["metric_scope"] = scope
    return row


def compare_summaries(
    current: Iterable[dict],
    baseline: Iterable[dict],
    metrics: Iterable[str],
    max_p50_regression: float,
    max_p50_regression_us: float = 0.0,
    require_matches: bool = True,
    min_samples: int = 0,
) -> tuple[list[dict], list[dict], list[dict], bool]:
    baseline_by_key = {_summary_group_key(summary): summary for summary in baseline}
    comparisons = []
    missing = []
    sample_warnings = []
    failed = False
    for current_summary in current:
        group_key = _summary_group_key(current_summary)
        baseline_summary = baseline_by_key.get(group_key)
        if baseline_summary is None:
            missing.append(
                {
                    "qlen": current_summary["qlen"],
                    "tp_count": current_summary["tp_count"],
                    "rank": current_summary["rank"],
                    "profile_kind": current_summary["profile_kind"],
                    "tp_part": current_summary["tp_part"],
                    "reason": "missing_baseline_group",
                }
            )
            continue
        for metric_spec in metrics:
            metric_scope, metric = _compare_metric_parts(metric_spec)
            if metric_scope is not None and current_summary["profile_kind"] != metric_scope:
                continue
            p50_key = f"{metric}_p50"
            if p50_key not in current_summary:
                missing.append(_add_metric_scope(
                    {
                        "qlen": current_summary["qlen"],
                        "tp_count": current_summary["tp_count"],
                        "rank": current_summary["rank"],
                        "profile_kind": current_summary["profile_kind"],
                        "tp_part": current_summary["tp_part"],
                        "metric": metric,
                        "reason": "missing_current_metric",
                    },
                    metric_scope,
                ))
                continue
            if p50_key not in baseline_summary:
                missing.append(_add_metric_scope(
                    {
                        "qlen": current_summary["qlen"],
                        "tp_count": current_summary["tp_count"],
                        "rank": current_summary["rank"],
                        "profile_kind": current_summary["profile_kind"],
                        "tp_part": current_summary["tp_part"],
                        "metric": metric,
                        "reason": "missing_baseline_metric",
                    },
                    metric_scope,
                ))
                continue
            current_value = float(current_summary[p50_key])
            baseline_value = float(baseline_summary[p50_key])
            if baseline_value <= 0.0:
                missing.append(_add_metric_scope(
                    {
                        "qlen": current_summary["qlen"],
                        "tp_count": current_summary["tp_count"],
                        "rank": current_summary["rank"],
                        "profile_kind": current_summary["profile_kind"],
                        "tp_part": current_summary["tp_part"],
                        "metric": metric,
                        "reason": "nonpositive_baseline_metric",
                    },
                    metric_scope,
                ))
                continue
            current_samples = int(current_summary.get("samples", 0))
            baseline_samples = int(baseline_summary.get("samples", 0))
            if min_samples > 0 and (current_samples < min_samples or baseline_samples < min_samples):
                sample_warnings.append(_add_metric_scope(
                    {
                        "qlen": current_summary["qlen"],
                        "tp_count": current_summary["tp_count"],
                        "rank": current_summary["rank"],
                        "profile_kind": current_summary["profile_kind"],
                        "tp_part": current_summary["tp_part"],
                        "metric": metric,
                        "current_samples": current_samples,
                        "baseline_samples": baseline_samples,
                        "min_samples": min_samples,
                        "reason": "insufficient_samples",
                    },
                    metric_scope,
                ))
            ratio = current_value / baseline_value
            delta = current_value - baseline_value
            regressed = ratio > max_p50_regression and delta > max_p50_regression_us
            failed = failed or regressed
            comparisons.append(_add_metric_scope(
                {
                    "qlen": current_summary["qlen"],
                    "tp_count": current_summary["tp_count"],
                    "rank": current_summary["rank"],
                    "profile_kind": current_summary["profile_kind"],
                    "tp_part": current_summary["tp_part"],
                    "metric": metric,
                    "current_samples": current_samples,
                    "baseline_samples": baseline_samples,
                    "current_p50": current_value,
                    "baseline_p50": baseline_value,
                    "delta_p50": delta,
                    "ratio": ratio,
                    "max_ratio": max_p50_regression,
                    "max_delta": max_p50_regression_us,
                    "regressed": regressed,
                },
                metric_scope,
            ))
    if require_matches and (missing or not comparisons):
        failed = True
    if sample_warnings:
        failed = True
    return comparisons, missing, sample_warnings, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile K2 SFT packed backward smoke cases and emit JSONL.")
    parser.add_argument("--qlens", default="4", help="Comma-separated qlen list.")
    parser.add_argument("--tp-counts", default="1,2", help="Comma-separated TP count list.")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--expert-num", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).with_name("k2_sft_packed_backward_profile.jsonl"),
    )
    parser.add_argument("--require-tp-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-jsonl", type=Path, help="Compare current p50 summaries against baseline cases.")
    parser.add_argument(
        "--compare-metrics",
        default="total_us,down_base_us,gate_up_base_us",
        help="Comma-separated metrics compared through *_p50. Use profile_kind:metric to scope a metric.",
    )
    parser.add_argument("--max-p50-regression", type=float, default=1.10)
    parser.add_argument(
        "--max-p50-regression-us",
        type=float,
        default=0.0,
        help="Allowed absolute p50 increase in microseconds before a ratio regression fails.",
    )
    parser.add_argument("--require-baseline-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-compare-samples", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    qlens = parse_csv_ints(args.qlens)
    tp_counts = parse_csv_ints(args.tp_counts)
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    cases = []
    failed = False
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.jsonl.open("a", encoding="utf-8") as output:
        for qlen in qlens:
            for tp_count in tp_counts:
                for repeat_idx in range(args.repeat):
                    case = run_case(repo_root, args, qlen, tp_count, repeat_idx)
                    output.write(json.dumps(case, sort_keys=True) + "\n")
                    output.flush()
                    cases.append(case)
                    failed = failed or case["returncode"] != 0 or "profile_error" in case

    summaries = summarize_profiles(cases)
    result = {"cases": len(cases), "jsonl": str(args.jsonl), "summaries": summaries}
    if args.baseline_jsonl is not None:
        baseline_summaries = summarize_profiles(load_jsonl_cases(args.baseline_jsonl))
        comparisons, missing_comparisons, sample_warnings, compare_failed = compare_summaries(
            summaries,
            baseline_summaries,
            parse_compare_metrics(args.compare_metrics),
            args.max_p50_regression,
            args.max_p50_regression_us,
            args.require_baseline_match,
            args.min_compare_samples,
        )
        result["baseline_jsonl"] = str(args.baseline_jsonl)
        result["comparisons"] = comparisons
        result["missing_comparisons"] = missing_comparisons
        result["sample_warnings"] = sample_warnings
        failed = failed or compare_failed
    print(json.dumps(result, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
