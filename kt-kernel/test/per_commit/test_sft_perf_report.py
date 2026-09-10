# SPDX-License-Identifier: Apache-2.0
"""Independent-run comparisons must not silently mix training contracts."""

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
    _spec = importlib.util.spec_from_file_location("sft_perf_report", _directory / "report.py")
    report = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(report)
finally:
    sys.path.pop(0)


def row(name, throughput, binary="baseline", **contract):
    return {
        "run": name,
        "binary": binary,
        "contract": {"tokens": 8192, **contract},
        "metrics": {"tokens_per_second": throughput},
    }


def test_descriptive_statistics_use_independent_runs():
    result = report.compare([row("a1", 100), row("a2", 110)], [row("b1", 210, "candidate")])
    assert result["median_throughput_ratio"] == 2
    assert result["baseline_summary"]["independent_runs"] == 2
    assert result["candidate_summary"]["sample_stdev"] is None
    assert "automatic significance" in result["interpretation"]


def test_one_process_cannot_be_counted_twice():
    with pytest.raises(ValueError, match="may not appear twice"):
        report.compare([row("same", 100)], [row("same", 100)])


def test_contract_drift_is_not_averaged_away():
    with pytest.raises(ValueError, match="experimental contract differs"):
        report.compare([row("a", 100)], [row("b", 110, tokens=16384)])


def test_a_variant_cannot_mix_binary_versions():
    with pytest.raises(ValueError, match="mixes different native binaries"):
        report.compare([row("a1", 100), row("a2", 110, "another")], [row("b", 200)])


def test_empty_side_is_rejected():
    with pytest.raises(ValueError, match="both sides"):
        report.compare([], [row("b", 100)])


def test_parameter_drift_is_relative_to_actual_update():
    initial = {"device": "cpu", "group": 0, "index": 0, "shape": [2], "values": [1.0, 2.0]}
    reference = {
        "contract": {"initial_parameters": [{"optimizer_probe": [initial]}]},
        "final_parameter_samples": [[{**initial, "values": [2.0, 2.0]}]],
    }
    candidate = {"final_parameter_samples": [[{**initial, "values": [2.0, 2.5]}]]}
    result = report.numerical_diagnostics(reference, candidate)["parameter_probes"][0]
    assert result["max_abs_diff"] == 0.5
    assert result["relative_to_reference_update_l2"] == 0.5
    candidate["final_parameter_samples"][0][0]["values"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        report.numerical_diagnostics(reference, candidate)


def test_numerical_step_mismatch_is_rejected():
    with pytest.raises(ValueError, match="different optimizer steps"):
        report.numerical_diagnostics({"losses": [{"step": 1, "loss": 1.0}]}, {})


def test_repeatability_uses_one_binary_and_independent_samples():
    result = report.repeatability([row("a1", 100), row("a2", 110)])
    assert result["repeatability_summary"]["independent_runs"] == 2
    assert result["repeatability_summary"]["sample_stdev"] is not None
    assert "median_throughput_ratio" not in result
    with pytest.raises(ValueError, match="one binary"):
        report.repeatability([row("a1", 100), row("a2", 110, "candidate")])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
