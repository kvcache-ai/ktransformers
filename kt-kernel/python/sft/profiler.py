# Staged profiling helpers for KT SFT MoE.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _find_kt_wrappers(model: Any):
    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ("base_model", "model"):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break
    return wrappers


def _profile_moe(layer_wrapper: Any):
    backend = getattr(layer_wrapper, "wrapper", layer_wrapper)
    return getattr(backend, "moe", None)


def collect_kt_sft_profile(model: Any, reset: bool = False) -> dict[str, Any]:
    """Collect JSON-compatible staged timing snapshots from all KT MoE layers."""
    layers: dict[int, dict[str, float]] = {}
    enabled = False
    for layer_wrapper in _find_kt_wrappers(model) or []:
        moe = _profile_moe(layer_wrapper)
        if moe is None or not hasattr(moe, "get_profile_stats"):
            continue
        raw = {str(key): float(value) for key, value in moe.get_profile_stats(reset).items()}
        layer_idx = int(raw.get("layer_idx", getattr(layer_wrapper, "layer_idx", len(layers))))
        layers[layer_idx] = raw
        enabled = enabled or bool(raw.get("wrapper.enabled", 0.0))
    return {"enabled": enabled, "layers": layers}


def reset_kt_sft_profile(model: Any) -> None:
    """Reset staged timing counters on all KT MoE layers."""
    for layer_wrapper in _find_kt_wrappers(model) or []:
        moe = _profile_moe(layer_wrapper)
        if moe is not None and hasattr(moe, "reset_profile_stats"):
            moe.reset_profile_stats()


def _split_timer_key(key: str) -> tuple[str, str] | None:
    suffix = ".total_ns"
    if not key.endswith(suffix):
        return None
    timer = key[: -len(suffix)]
    if timer.startswith("wrapper."):
        return "wrapper", timer[len("wrapper.") :]
    if timer.startswith("tp."):
        parts = timer.split(".", 2)
        if len(parts) == 3:
            return f"tp.{parts[1]}", parts[2]
    return None


def _parent_stage(stage: str) -> str | None:
    if stage.startswith("backward.base_weight_grad.worker_cpu."):
        return None
    if stage.startswith("backward.base_weight_grad."):
        return "backward.base_weight_grad"
    if stage.startswith("weights.base_reload."):
        return "weights.base_reload"
    if stage == "backward.down.total" or stage == "backward.gate_up.total":
        return "backward.total"
    if stage.startswith("backward.down."):
        return "backward.down.total"
    if stage.startswith("backward.gate_up."):
        return "backward.gate_up.total"
    if stage.endswith(".total"):
        return None
    if stage.startswith("forward."):
        return "forward.total"
    if stage.startswith("backward."):
        return "backward.total"
    if stage.startswith("tp.forward."):
        return "tp.forward.total"
    if stage.startswith("tp.backward."):
        return "tp.backward.total"
    return None


def _aggregate_rows(profile: dict[str, Any]) -> list[dict[str, float | str]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"total_ns": 0.0, "calls": 0.0, "tokens": 0.0, "bytes": 0.0}
    )
    for raw in profile.get("layers", {}).values():
        scope_tokens: dict[str, float] = {"wrapper": raw.get("wrapper.tokens", 0.0)}
        tp_count = int(raw.get("tp_count", 0.0))
        for tp_idx in range(tp_count):
            scope_tokens[f"tp.{tp_idx}"] = raw.get(f"tp.{tp_idx}.tokens", 0.0)

        for key, total_ns in raw.items():
            parsed = _split_timer_key(key)
            if parsed is None or total_ns <= 0.0:
                continue
            scope, stage = parsed
            calls = raw.get(key[: -len("total_ns")] + "calls", 0.0)
            row = totals[(scope, stage)]
            row["total_ns"] += total_ns
            row["calls"] += calls
            row["tokens"] += scope_tokens.get(scope, 0.0)
            row["bytes"] += raw.get(key[: -len("total_ns")] + "bytes", 0.0)

    rows: list[dict[str, float | str]] = []
    for (scope, stage), values in totals.items():
        calls = values["calls"]
        tokens = values["tokens"]
        parent = _parent_stage(stage)
        parent_ns = totals.get((scope, parent), {}).get("total_ns", 0.0) if parent else 0.0
        rows.append(
            {
                "scope": scope,
                "stage": stage,
                "calls": calls,
                "total_ms": values["total_ns"] / 1e6,
                "avg_ms": values["total_ns"] / calls / 1e6 if calls else 0.0,
                "us_per_token": values["total_ns"] / tokens / 1e3 if tokens else 0.0,
                "mib": values["bytes"] / (1024.0 * 1024.0),
                "parent_pct": values["total_ns"] / parent_ns * 100.0 if parent_ns else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (str(row["scope"]), -float(row["total_ms"]), str(row["stage"])))


def format_kt_sft_profile(profile: dict[str, Any]) -> str:
    """Format a compact cross-layer timing table while preserving TP scopes."""
    if not profile.get("enabled"):
        return "KT SFT profiler disabled or no profiled KT layers found. Set KT_SFT_PROFILE=1 before model creation."

    rows = _aggregate_rows(profile)
    header = (
        f"{'scope':<9} {'stage':<34} {'calls':>7} {'total_ms':>11} "
        f"{'avg_ms':>10} {'us/token':>10} {'MiB':>10} {'parent%':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['scope']:<9} {row['stage']:<34} {row['calls']:>7.0f} "
            f"{row['total_ms']:>11.3f} {row['avg_ms']:>10.3f} "
            f"{row['us_per_token']:>10.3f} {row['mib']:>10.2f} {row['parent_pct']:>8.1f}%"
        )
    return "\n".join(lines)
