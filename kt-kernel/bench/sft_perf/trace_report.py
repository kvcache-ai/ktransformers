# SPDX-License-Identifier: Apache-2.0
"""Inspect captured CPU-repack/GPU-kernel overlap without summing parallel time."""

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from common import sha256, write_json


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if right is None:
            total += end - start
        elif end > right:
            total += end - max(start, right)
        right = max(right, end) if right is not None else end
    return total


def summarize_range(cpu, kernels):
    start, end = cpu["start"], cpu["end"]
    if end <= start:
        raise ValueError("CPU range is not a complete positive-duration interval")
    events = []
    per_device = defaultdict(lambda: defaultdict(list))
    other_intervals = []
    for kernel in kernels:
        left, right = max(start, kernel["start"]), min(end, kernel["end"])
        if right <= left:
            continue
        category = "nccl_named" if "nccl" in kernel["name"].lower() else "other"
        per_device[kernel["deviceId"]][category].append((left, right))
        if category == "other":
            other_intervals.append((left, right))
        events.append({**kernel, "category": category, "clipped_overlap_ns": right - left})
    return {
        "cpu": cpu,
        "duration_ns": end - start,
        "per_device_union_ns": {
            device: {category: union_ns(intervals) for category, intervals in categories.items()}
            for device, categories in per_device.items()
        },
        "any_gpu_other_kernel_union_ns": union_ns(other_intervals),
        "any_gpu_other_kernel_overlap_fraction": union_ns(other_intervals) / (end - start),
        "clipped_kernel_events": events,
    }


def read_trace(path: Path, range_name="repack.async"):
    # Read-only URI: never create an empty database for a mistyped result path.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        database.row_factory = sqlite3.Row
        tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS", "StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if not required <= tables:
            raise ValueError(f"capture lacks required NVTX/CUDA tables: {sorted(required - tables)}")
        # Nsight's own nvtx_sum report joins domains on domainId and process ID
        # bits in globalTid. Keep the host bits too when selecting this domain.
        ranges = [
            dict(row)
            for row in database.execute(
                """
            SELECT e.start, e.end, e.globalTid, e.domainId, e.int64Value AS layer,
                   e.category, COALESCE(e.text, s.value) AS name
            FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
            WHERE e.end IS NOT NULL AND COALESCE(e.text, s.value) = ?
              AND EXISTS (
                SELECT 1 FROM NVTX_EVENTS d LEFT JOIN StringIds ds ON d.textId = ds.id
                WHERE d.eventType = 75 AND COALESCE(d.text, ds.value) = 'kt.sft'
                  AND d.domainId = e.domainId AND (d.globalTid >> 24) = (e.globalTid >> 24)
              ) ORDER BY e.start
        """,
                (range_name,),
            )
        ]
        if not ranges:
            raise ValueError(f"no complete {range_name!r} range in the kt.sft domain")
        kernels = [dict(row) for row in database.execute("""
            SELECT k.start, k.end, k.deviceId, k.streamId, k.globalPid,
                   s.value AS name
            FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        """)]
    if not kernels:
        raise ValueError("capture contains no CUDA kernel events; overlap is unobservable")
    return {
        "sqlite_sha256": sha256(path),
        "kernel_events_in_capture": len(kernels),
        "devices_with_kernel_events": sorted({row["deviceId"] for row in kernels}),
        "processes_with_kernel_events": sorted({row["globalPid"] for row in kernels}),
        "ranges": [summarize_range(cpu, kernels) for cpu in ranges],
        "interpretation": (
            "Captured instrumented diagnostic intervals, possibly including warmup; "
            "not an uninstrumented steady-state hidden-time fraction. "
            "NCCL classification is name-based; inspect other kernel names for actual work. "
            "Parallel GPU intervals are unioned, never summed as CPU time. "
            "No claim of critical-path savings, all-layer coverage, or memory-copy overlap. "
            "Activities crossing capture boundaries may be incomplete."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--range", default="repack.async")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; reports are append-only")
    result = read_trace(args.sqlite, args.range)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "kernel_events_in_capture": result["kernel_events_in_capture"],
                "ranges": [
                    {key: value for key, value in row.items() if key != "clipped_kernel_events"}
                    for row in result["ranges"]
                ],
                "interpretation": result["interpretation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
