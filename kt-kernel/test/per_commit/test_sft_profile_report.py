# SPDX-License-Identifier: Apache-2.0
"""Profile summaries must not add parallel TP timings or silently accept drift."""

import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

_directory = Path(__file__).resolve().parents[2] / "bench" / "sft_perf"
sys.path.insert(0, str(_directory))
try:
    _spec = importlib.util.spec_from_file_location("sft_profile_report", _directory / "profile_report.py")
    report = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(report)
finally:
    sys.path.pop(0)


def profile():
    return {
        "enabled": True,
        "layers": {
            "3": {
                "tp_count": 2,
                "wrapper.workloads": 2,
                "wrapper.routed_rows": 1024,
                "wrapper.active_experts": 4,
                "tp.0.forward.total.total_ns": 4e9,
                "tp.1.forward.total.total_ns": 6e9,
                "wrapper.weights.backward_repack.total_ns": 2e9,
                "wrapper.weights.backward_repack_wait.total_ns": 1e9,
            }
        },
    }


def test_parallel_parts_are_max_not_sum():
    result = report.summarize(profile(), 2)
    assert result["sum_layer_max_tp_accumulated_seconds_per_step"]["forward.total"] == 3
    assert result["sum_layer_wrapper_seconds_per_step"]["weights.backward_repack"] == 1
    assert result["routed_rows_per_active_expert_observation"] == 256
    assert "critical-path" in result["interpretation"]


def test_step_count_drift_rejected():
    with pytest.raises(ValueError, match="workload count"):
        report.summarize(profile(), 3)


def test_nonfinite_counter_rejected():
    value = profile()
    value["layers"]["3"]["tp.1.forward.total.total_ns"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        report.summarize(value, 2)


def test_missing_parallel_stage_rejected():
    value = profile()
    del value["layers"]["3"]["tp.1.forward.total.total_ns"]
    with pytest.raises(KeyError):
        report.summarize(value, 2)


def test_histogram_uses_one_tp_partition_and_weights_by_rows():
    value = profile()
    layer = value["layers"]["3"]
    for part in (0, 1):
        layer[f"tp.{part}.expert_rows.maximum"] = 1024
        layer[f"tp.{part}.active_experts"] = 2
        for name in report.ROW_BINS:
            layer[f"tp.{part}.expert_rows.{name}.observations"] = 0
            layer[f"tp.{part}.expert_rows.{name}.rows"] = 0
        for name, rows in (("1_7", 1), ("1024_plus", 1024)):
            layer[f"tp.{part}.expert_rows.{name}.observations"] = 1
            layer[f"tp.{part}.expert_rows.{name}.rows"] = rows
    result = report.summarize(value, 2)["expert_row_distribution"]
    assert result["observations"] == 2
    assert result["routed_rows"] == 1025
    assert result["bins"][0]["observation_fraction"] == 0.5
    assert result["bins"][0]["row_fraction"] == 1 / 1025
    assert result["maximum_rows"] == 1024


def test_partial_histogram_rejected():
    layers = {"3": {"tp.0.expert_rows.maximum": 256}, "4": {}}
    with pytest.raises(ValueError, match="only some layers"):
        report.row_distribution(layers)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
