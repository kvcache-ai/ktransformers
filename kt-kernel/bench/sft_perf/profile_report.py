# SPDX-License-Identifier: Apache-2.0
"""Summarize native stage counters without inventing a critical-path timeline."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from common import sha256, write_json

ROW_BINS = ("1_7", "8_32", "33_63", "64_127", "128_255", "256_511", "512_1023", "1024_plus")


def row_distribution(layers):
    present = ["tp.0.expert_rows.maximum" in values for values in layers.values()]
    if not any(present):
        return None  # Historical profiles predate the optional counters.
    if not all(present):
        raise ValueError("expert-row histogram is present on only some layers")
    bins = [{"range": name, "observations": 0, "rows": 0} for name in ROW_BINS]
    for layer, values in layers.items():
        for row in bins:
            for counter in ("observations", "rows"):
                row[counter] += values[f"tp.0.expert_rows.{row['range']}.{counter}"]
        observed = sum(values[f"tp.0.expert_rows.{name}.observations"] for name in ROW_BINS)
        if observed != values["tp.0.active_experts"]:
            raise ValueError(f"layer {layer}: histogram count differs from active expert observations")
    total_observations = sum(row["observations"] for row in bins)
    total_rows = sum(row["rows"] for row in bins)
    for row in bins:
        row["observation_fraction"] = row["observations"] / total_observations if total_observations else 0
        row["row_fraction"] = row["rows"] / total_rows if total_rows else 0
    return {
        "scope": "CPU expert forward-routing observations from TP0 only; do not double-count identical TP routing",
        "maximum_rows": max(values["tp.0.expert_rows.maximum"] for values in layers.values()),
        "observations": total_observations,
        "routed_rows": total_rows,
        "bins": bins,
        "interpretation": "Row share is proportional to base GEMM FLOPs only when expert matrix dimensions match.",
    }


def summarize(profile, measured):
    if not profile.get("enabled") or not profile.get("layers") or measured <= 0:
        raise ValueError("an enabled, nonempty profile and positive measured-step count are required")
    tp_stages = defaultdict(list)
    wrapper_stages = defaultdict(list)
    routed_rows, active_experts = 0, 0
    for layer, values in profile["layers"].items():
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values.values()
        ):
            raise ValueError(f"layer {layer}: negative, nonnumeric or non-finite profile counter")
        count = values["tp_count"]
        if count != int(count) or count <= 0:
            raise ValueError(f"layer {layer}: invalid TP count")
        if values["wrapper.workloads"] != measured:
            raise ValueError(f"layer {layer}: workload count differs from measured steps")
        for key, value in values.items():
            if key.startswith("tp.0.") and key.endswith(".total_ns"):
                stage = key[len("tp.0.") : -len(".total_ns")]
                parts = [values[f"tp.{part}.{stage}.total_ns"] for part in range(int(count))]
                tp_stages[stage].append(max(parts))
            elif key.startswith("wrapper.") and key.endswith(".total_ns"):
                stage = key[len("wrapper.") : -len(".total_ns")]
                wrapper_stages[stage].append(value)
        routed_rows += values["wrapper.routed_rows"]
        active_experts += values["wrapper.active_experts"]
    convert = lambda stages: {key: math.fsum(values) / measured / 1e9 for key, values in sorted(stages.items())}
    return {
        "measured_steps": measured,
        "layer_count": len(profile["layers"]),
        "sum_layer_max_tp_accumulated_seconds_per_step": convert(tp_stages),
        "sum_layer_wrapper_seconds_per_step": convert(wrapper_stages),
        "routed_rows_per_active_expert_observation": routed_rows / active_experts if active_experts else None,
        "expert_row_distribution": row_distribution(profile["layers"]),
        "interpretation": (
            "Diagnostic counters only. TP maxima are taken after accumulation, not per-call timeline maxima. "
            "Stages may nest or overlap: do not sum totals with children, or subtract repack wait from "
            "repack duration to claim critical-path savings. No formal throughput comparison is implied."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--measured", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; diagnostic summaries are append-only")
    result = summarize(json.loads(args.profile.read_text()), args.measured)
    result["profile_sha256"] = sha256(args.profile)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
