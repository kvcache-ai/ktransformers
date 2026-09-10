# SPDX-License-Identifier: Apache-2.0
"""Framework-independent tests for the FP8 end-to-end measurement contract."""

import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

_path = Path(__file__).resolve().parents[2] / "bench" / "sft_perf" / "common.py"
_spec = importlib.util.spec_from_file_location("sft_perf_common", _path)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


def config():
    return {
        "stage": "sft",
        "do_train": True,
        "do_eval": False,
        "finetuning_type": "lora",
        "use_kt": True,
        "include_num_input_tokens_seen": "non_padding",
        "save_strategy": "no",
        "logging_nan_inf_filter": False,
        "kt_config": {"kt_expert_weight_format": "fp8"},
        "max_steps": 5,
    }


def test_global_tokens_are_not_multiplied_by_rank_count():
    result = common.window_metrics(8192, 40960, 32.0, 4)
    assert result["tokens_per_second"] == 1024
    assert result["seconds_per_optimizer_step"] == 8


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_durations_rejected(seconds):
    with pytest.raises(ValueError):
        common.window_metrics(0, 1024, seconds, 1)


def test_non_advancing_token_counter_rejected():
    with pytest.raises(ValueError):
        common.window_metrics(1024, 1024, 1.0, 1)


def test_warmup_and_optimizer_steps_are_explicit():
    common.validate_training_config(config(), 1, 4)
    with pytest.raises(ValueError):
        common.validate_training_config(config(), 1, 3)
    with pytest.raises(ValueError):
        common.validate_training_config(config(), 0, 5)


@pytest.mark.parametrize(
    "key,value",
    [
        ("logging_nan_inf_filter", True),
        ("use_kt", False),
        ("stage", "pt"),
        ("resume_from_checkpoint", "previous-run"),
        ("include_num_input_tokens_seen", "all"),
    ],
)
def test_contract_drift_rejected(key, value):
    value_config = config()
    value_config[key] = value
    with pytest.raises(ValueError):
        common.validate_training_config(value_config, 1, 4)


def test_atomic_json_replaces_only_its_own_result(tmp_path):
    path = tmp_path / "result.json"
    common.write_json(path, {"status": "RUNNING"})
    common.write_json(path, {"status": "PASS"})
    assert '"PASS"' in path.read_text()
    assert not path.with_suffix(".json.tmp").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
