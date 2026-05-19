#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from common import ensure_dir, memory_stat_snapshot, parse_runtime_logs, write_csv, write_json


def sample_memory(cgroup_path: str, out_path: Path, stop: threading.Event, interval_s: float) -> None:
    rows = []
    while not stop.is_set():
        snap = memory_stat_snapshot(cgroup_path)
        snap["sample_wall"] = time.time()
        rows.append(snap)
        time.sleep(interval_s)
    write_csv(out_path, rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure mmap page-fault and cgroup page-cache failure mode.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--cgroup-path", required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--perf-stat", action="store_true")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    cmd = args.cmd
    if args.perf_stat:
        cmd = (
            "perf stat -e page-faults,minor-faults,major-faults,cycles,instructions "
            f"-o {outdir / 'perf_stat.txt'} -- {cmd}"
        )

    stop = threading.Event()
    sampler = threading.Thread(
        target=sample_memory,
        args=(args.cgroup_path, outdir / "memory_samples.csv", stop, args.sample_interval),
        daemon=True,
    )
    sampler.start()

    start = time.time()
    stdout_path = outdir / "run.stdout.txt"
    stderr_path = outdir / "run.stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, shell=True, stdout=out, stderr=err, text=True, preexec_fn=os.setsid)
        timed_out = False
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            rc = proc.wait(timeout=30)
    end = time.time()
    stop.set()
    sampler.join(timeout=5)

    runtime = parse_runtime_logs([stdout_path, stderr_path])
    samples = []
    mem_csv = outdir / "memory_samples.csv"
    if mem_csv.exists():
        import csv

        with mem_csv.open("r", encoding="utf-8", newline="") as f:
            samples = list(csv.DictReader(f))
    peak = {}
    for key in ("memory_current", "file", "anon", "kernel", "slab"):
        vals = [int(float(row[key])) for row in samples if row.get(key) not in (None, "")]
        if vals:
            peak[f"{key}_peak_bytes"] = max(vals)
            peak[f"{key}_peak_gib"] = max(vals) / 1073741824.0

    summary = {
        "cmd": args.cmd,
        "returncode": rc,
        "timed_out": timed_out,
        "elapsed_s": end - start,
        "runtime": runtime,
        "memory_peaks": peak,
        "cgroup_path": args.cgroup_path,
    }
    write_json(outdir / "mmap_failure_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
