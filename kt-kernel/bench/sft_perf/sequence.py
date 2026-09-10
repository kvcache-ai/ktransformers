# SPDX-License-Identifier: Apache-2.0
"""Run one explicit A/B/B/A plan with the existing per-process supervisor."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

from common import write_json
from report import compare, read_run
from run import main as run_one


def abba_plan(stack, baseline, candidate, prefix):
    if Path(prefix).name != prefix or prefix in ("", ".", ".."):
        raise ValueError("prefix must be one non-empty path component")
    if baseline == candidate:
        raise ValueError("A/B/B/A requires distinct variant names")
    for name in (baseline, candidate):
        if name not in stack["variants"] or stack["variants"][name].get("diagnostic_only", False):
            raise ValueError("both variants must exist and permit formal throughput runs")
    return [
        {"label": f"{prefix}-{suffix}", "variant": variant, "group": group}
        for suffix, variant, group in (
            ("a1", baseline, "A"),
            ("b1", candidate, "B"),
            ("b2", candidate, "B"),
            ("a2", baseline, "A"),
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args()
    if min(args.warmup, args.measured, args.timeout) <= 0:
        parser.error("warmup, measured and timeout must be positive")
    stack = json.loads(args.stack.read_text())
    plan = abba_plan(stack, args.baseline, args.candidate, args.prefix)
    root = Path(stack["run_root"]).resolve(strict=True)
    if any((root / row["label"]).exists() for row in plan):
        parser.error("one or more planned result directories already exist")
    directory = root / f"{args.prefix}.sequence"
    directory.mkdir()  # A sequence, like an individual run, is never reused.
    frozen_stack = directory / "stack.json"
    write_json(frozen_stack, stack)
    status = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "completed": [],
        "warmup": args.warmup,
        "measured": args.measured,
    }
    write_json(directory / "status.json", status)

    def interrupted(signum, frame):
        # Raising through run_one preserves its owned-process finally cleanup.
        raise KeyboardInterrupt(f"sequence received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    groups = {"A": [], "B": []}
    try:
        for row in plan:
            status["active"] = row["label"]
            write_json(directory / "status.json", status)
            run_one(
                [
                    "--stack",
                    str(frozen_stack),
                    "--variant",
                    row["variant"],
                    "--label",
                    row["label"],
                    "--warmup",
                    str(args.warmup),
                    "--measured",
                    str(args.measured),
                    "--timeout",
                    str(args.timeout),
                ]
            )
            groups[row["group"]].append(read_run(root / row["label"]))
            if groups["A"] and groups["B"]:
                # Validate contract/binary consistency immediately, not after
                # spending hours on the remaining runs of an invalid comparison.
                result = compare(groups["A"], groups["B"])
                write_json(directory / f"comparison-{len(status['completed']) + 1}.json", result)
            status["completed"].append(row["label"])
            write_json(directory / "status.json", status)
        status["status"] = "PASS"
    except BaseException as error:
        status.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "status.json", status)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
