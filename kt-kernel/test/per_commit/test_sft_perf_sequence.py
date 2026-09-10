# SPDX-License-Identifier: Apache-2.0
"""Sequence contracts only; no training or external processes are started."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

directory = Path(__file__).resolve().parents[2] / "bench" / "sft_perf"
sys.path.insert(0, str(directory))
try:
    spec = importlib.util.spec_from_file_location("sft_perf_sequence", directory / "sequence.py")
    sequence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sequence)
finally:
    sys.path.pop(0)


def test_independent_abba_order():
    plan = sequence.abba_plan({"variants": {"old": {}, "new": {}}}, "old", "new", "long18")
    assert [row["variant"] for row in plan] == ["old", "new", "new", "old"]
    assert [row["group"] for row in plan] == ["A", "B", "B", "A"]
    assert [row["label"] for row in plan] == ["long18-a1", "long18-b1", "long18-b2", "long18-a2"]


@pytest.mark.parametrize("prefix", ["", ".", "..", "/outside", "nested/name"])
def test_sequence_labels_cannot_escape_run_root(prefix):
    with pytest.raises(ValueError, match="path component"):
        sequence.abba_plan({"variants": {"a": {}, "b": {}}}, "a", "b", prefix)


def test_diagnostic_or_missing_variant_is_not_formal():
    stack = {"variants": {"a": {}, "b": {"diagnostic_only": True}}}
    for candidate in ("b", "missing"):
        with pytest.raises(ValueError, match="formal throughput"):
            sequence.abba_plan(stack, "a", candidate, "test")
    with pytest.raises(ValueError, match="distinct variant"):
        sequence.abba_plan(stack, "a", "a", "test")


@pytest.mark.parametrize("fail_candidate", [False, True])
def test_sequence_reuses_supervisor_and_aborts_on_failure(tmp_path, monkeypatch, fail_candidate):
    stack = tmp_path / "input.json"
    stack.write_text(json.dumps({"run_root": str(tmp_path), "variants": {"old": {}, "new": {}}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sequence.py",
            "--stack",
            str(stack),
            "--baseline",
            "old",
            "--candidate",
            "new",
            "--prefix",
            "long18",
            "--warmup",
            "2",
            "--measured",
            "16",
        ],
    )
    commands = []

    def launch(argv):
        commands.append(argv)
        if fail_candidate and len(commands) == 2:
            raise RuntimeError("test candidate failure")

    def read(path):
        return {
            "run": path.name,
            "binary": "old" if path.name[-2] == "a" else "new",
            "contract": {"updates": 18},
            "metrics": {"tokens_per_second": 100},
        }

    monkeypatch.setattr(sequence, "run_one", launch)
    monkeypatch.setattr(sequence, "read_run", read)
    if fail_candidate:
        with pytest.raises(RuntimeError, match="test candidate failure"):
            sequence.main()
    else:
        sequence.main()
    assert len(commands) == (2 if fail_candidate else 4)
    for command in commands:
        assert command[command.index("--stack") + 1] == str(tmp_path / "long18.sequence" / "stack.json")
        assert command[command.index("--warmup") + 1] == "2"
        assert command[command.index("--measured") + 1] == "16"
        assert "--profile" not in command and "--trace" not in command
    status = json.loads((tmp_path / "long18.sequence" / "status.json").read_text())
    assert status["status"] == ("FAIL" if fail_candidate else "PASS")
    assert len(status["completed"]) == (1 if fail_candidate else 4)
    assert (tmp_path / "long18.sequence" / "comparison-4.json").exists() is not fail_candidate


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
