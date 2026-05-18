"""Smoke tests for the residency-policy replay benchmark."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_residency_policies.py"


def test_compare_residency_policies_script_replays_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "layer_idx": 0,
                        "capacity": 2,
                        "num_experts": 4,
                        "accesses": [0, 1, 0, 2],
                    }
                ),
                json.dumps(
                    {
                        "layer_idx": 0,
                        "capacity": 2,
                        "num_experts": 4,
                        "accesses": [2, 0, 3],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_json = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace",
            str(trace_path),
            "--policies",
            "lru,sieve",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "policy" in result.stdout
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {row["policy"] for row in rows} == {"lru", "sieve"}
