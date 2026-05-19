#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from common import ensure_dir, run_command, write_csv, write_json


SIZE_RE = re.compile(r"^(?P<num>[0-9.]+)(?P<unit>[KMGTP]?i?B?|[kmgpt]?)$")
UNIT = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "ki": 1024,
    "kib": 1024,
    "mi": 1024**2,
    "mib": 1024**2,
    "gi": 1024**3,
    "gib": 1024**3,
}


def parse_size(raw: str) -> int:
    m = SIZE_RE.match(raw.strip())
    if not m:
        raise ValueError(f"invalid size: {raw}")
    return int(float(m.group("num")) * UNIT[m.group("unit").lower()])


def run_dd(file_path: Path, block_size: str, bytes_to_read: int, direct: bool) -> dict:
    bs = parse_size(block_size)
    count = max(1, math.ceil(bytes_to_read / bs))
    flags = ["iflag=direct"] if direct else []
    cmd = ["dd", f"if={file_path}", "of=/dev/null", f"bs={block_size}", f"count={count}", "status=none"] + flags
    start = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    end = time.time()
    elapsed = end - start
    submitted = min(file_path.stat().st_size, count * bs)
    return {
        "file": str(file_path),
        "file_bytes": file_path.stat().st_size,
        "block_size": block_size,
        "block_size_bytes": bs,
        "direct": int(direct),
        "requested_bytes": bytes_to_read,
        "submitted_bytes": submitted,
        "elapsed_s": elapsed,
        "throughput_mib_s": submitted / 1048576.0 / elapsed if elapsed > 0 else None,
        "throughput_gib_s": submitted / 1073741824.0 / elapsed if elapsed > 0 else None,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure storage read path throughput for paper data-path claims.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--files", nargs="+", required=True, help="safetensors or cache blob files to read")
    parser.add_argument("--read-size", default="8GiB")
    parser.add_argument("--block-sizes", default="4M,16M,64M,256M")
    parser.add_argument("--direct", choices=["0", "1", "both"], default="1")
    parser.add_argument("--expert-bench-cmd", default="", help="Optional command for expert-shaped io_uring bench")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    bytes_to_read = parse_size(args.read_size)
    block_sizes = [x.strip() for x in args.block_sizes.split(",") if x.strip()]
    direct_modes = [False, True] if args.direct == "both" else [args.direct == "1"]

    rows = []
    for file_raw in args.files:
        file_path = Path(file_raw)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        for direct in direct_modes:
            for bs in block_sizes:
                row = run_dd(file_path, bs, bytes_to_read, direct)
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    write_csv(outdir / "storage_read_summary.csv", rows)
    write_json(outdir / "storage_read_summary.json", {"rows": rows})

    if args.expert_bench_cmd:
        bench_dir = ensure_dir(outdir / "expert_shaped_bench")
        meta = run_command(args.expert_bench_cmd, bench_dir, timeout=args.timeout, prefix="expert_bench")
        write_json(outdir / "expert_shaped_bench_meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
