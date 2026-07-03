from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_profile_module():
    path = Path(__file__).resolve().parents[1] / "bench" / "bench_k2_sft_packed_backward_profile.py"
    spec = importlib.util.spec_from_file_location("bench_k2_sft_packed_backward_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_profile_line_tp_shard():
    module = _load_profile_module()
    row = module.parse_profile_line(
        "[KT_K2_SFT_PROFILE] layer=0 tp_part=1 qlen=4 active=2 tokens=8 "
        "grad_weights_us=15 down_us=18141 down_lora_grads_us=389 "
        "down_route_us=30 down_write_us=4 down_base_us=17000 "
        "down_lora_bprop_us=100 down_lora_a_us=120 down_lora_b_us=140 "
        "activation_us=16 gate_up_us=35410 gate_up_base_us=21000 "
        "gate_up_lora_u_us=1200 gate_up_lora_b_us=6000 "
        "gate_up_lora_b_write_us=80 gate_up_lora_a_input_us=5800 "
        "gate_up_write_us=90 total_us=53973"
    )

    assert row["profile_kind"] == "tp_shard"
    assert row["tp_part"] == 1
    assert row["down_lora_grads_us"] == 389
    assert row["down_base_us"] == 17000
    assert row["down_lora_b_us"] == 140
    assert row["gate_up_base_us"] == 21000
    assert row["gate_up_lora_b_write_us"] == 80
    assert row["gate_up_lora_a_input_us"] == 5800
    assert row["total_us"] == 53973


def test_parse_compare_metrics_accepts_optional_profile_kind_scope():
    module = _load_profile_module()

    assert module.parse_compare_metrics("total_us,tp_shard:down_lora_grads_us") == [
        "total_us",
        "tp_shard:down_lora_grads_us",
    ]
    with pytest.raises(ValueError):
        module.parse_compare_metrics("tp_shard:")
    with pytest.raises(ValueError):
        module.parse_compare_metrics(":down_lora_grads_us")


def test_summarize_profiles_groups_tp_and_single_tp():
    module = _load_profile_module()
    cases = [
        {
            "qlen": 4,
            "tp_count": 2,
            "rank": 2,
            "profiles": [
                {
                    "profile_kind": "tp_shard",
                    "tp_part": 0,
                    "down_us": 10,
                    "down_base_us": 6,
                    "gate_up_us": 30,
                    "gate_up_base_us": 12,
                    "total_us": 40,
                },
                {
                    "profile_kind": "tp_shard",
                    "tp_part": 0,
                    "down_us": 14,
                    "down_base_us": 8,
                    "gate_up_us": 34,
                    "gate_up_base_us": 16,
                    "total_us": 48,
                },
                {
                    "profile_kind": "single_tp",
                    "down_us": 20,
                    "gate_up_us": 80,
                    "total_us": 100,
                },
            ],
        }
    ]

    summaries = module.summarize_profiles(cases)
    tp_summary = next(item for item in summaries if item["profile_kind"] == "tp_shard")
    single_summary = next(item for item in summaries if item["profile_kind"] == "single_tp")

    assert tp_summary["samples"] == 2
    assert tp_summary["down_us_min"] == 10
    assert tp_summary["down_us_p50"] == 12
    assert tp_summary["down_us_avg"] == 12
    assert tp_summary["down_us_max"] == 14
    assert tp_summary["down_base_us_avg"] == 7
    assert tp_summary["gate_up_base_us_p50"] == 14
    assert tp_summary["gate_up_base_us_avg"] == 14
    assert tp_summary["total_us_max"] == 48
    assert single_summary["tp_part"] == "all"
    assert single_summary["total_us_min"] == 100
    assert single_summary["total_us_p50"] == 100
    assert single_summary["total_us_avg"] == 100


def test_compare_summaries_skips_profile_kind_scoped_metrics_for_other_kinds():
    module = _load_profile_module()
    baseline = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "single_tp",
            "tp_part": "all",
            "samples": 3,
            "total_us_p50": 100.0,
        },
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "samples": 3,
            "total_us_p50": 50.0,
            "down_lora_grads_us_p50": 10.0,
        },
    ]
    current = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "single_tp",
            "tp_part": "all",
            "samples": 3,
            "total_us_p50": 95.0,
        },
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "samples": 3,
            "total_us_p50": 49.0,
            "down_lora_grads_us_p50": 7.0,
        },
    ]

    comparisons, missing, sample_warnings, failed = module.compare_summaries(
        current,
        baseline,
        ["total_us", "tp_shard:down_lora_grads_us"],
        max_p50_regression=1.10,
        min_samples=3,
    )

    scoped_cmp = next(item for item in comparisons if item["metric"] == "down_lora_grads_us")
    assert len(comparisons) == 3
    assert missing == []
    assert sample_warnings == []
    assert failed is False
    assert scoped_cmp["profile_kind"] == "tp_shard"
    assert scoped_cmp["metric_scope"] == "tp_shard"
    assert scoped_cmp["ratio"] == 0.7


def test_compare_summaries_flags_p50_regression():
    module = _load_profile_module()
    baseline = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "down_base_us_p50": 100.0,
            "gate_up_base_us_p50": 200.0,
        }
    ]
    current = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "down_base_us_p50": 111.0,
            "gate_up_base_us_p50": 198.0,
        }
    ]

    comparisons, missing, sample_warnings, failed = module.compare_summaries(
        current, baseline, ["down_base_us", "gate_up_base_us"], max_p50_regression=1.10
    )

    down_cmp = next(item for item in comparisons if item["metric"] == "down_base_us")
    gate_cmp = next(item for item in comparisons if item["metric"] == "gate_up_base_us")
    assert missing == []
    assert sample_warnings == []
    assert failed is True
    assert down_cmp["current_samples"] == 0
    assert down_cmp["baseline_samples"] == 0
    assert down_cmp["ratio"] == 1.11
    assert down_cmp["regressed"] is True
    assert gate_cmp["ratio"] == 0.99
    assert gate_cmp["regressed"] is False


def test_compare_summaries_allows_small_absolute_p50_regression():
    module = _load_profile_module()
    baseline = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "down_lora_grads_us_p50": 596.0,
            "gate_up_base_us_p50": 12000.0,
        }
    ]
    current = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "down_lora_grads_us_p50": 660.0,
            "gate_up_base_us_p50": 13400.0,
        }
    ]

    comparisons, missing, sample_warnings, failed = module.compare_summaries(
        current,
        baseline,
        ["down_lora_grads_us", "gate_up_base_us"],
        max_p50_regression=1.10,
        max_p50_regression_us=100.0,
    )

    lora_cmp = next(item for item in comparisons if item["metric"] == "down_lora_grads_us")
    gate_cmp = next(item for item in comparisons if item["metric"] == "gate_up_base_us")
    assert missing == []
    assert sample_warnings == []
    assert failed is True
    assert lora_cmp["ratio"] > 1.10
    assert lora_cmp["delta_p50"] == 64.0
    assert lora_cmp["regressed"] is False
    assert gate_cmp["delta_p50"] == 1400.0
    assert gate_cmp["regressed"] is True


def test_compare_summaries_requires_baseline_metric_matches():
    module = _load_profile_module()
    baseline = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "samples": 3,
            "down_base_us_p50": 100.0,
        }
    ]
    current = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 1,
            "samples": 3,
            "down_base_us_p50": 95.0,
        }
    ]

    comparisons, missing, sample_warnings, failed = module.compare_summaries(
        current, baseline, ["down_base_us"], max_p50_regression=1.10
    )

    assert comparisons == []
    assert sample_warnings == []
    assert failed is True
    assert missing == [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 1,
            "reason": "missing_baseline_group",
        }
    ]


def test_compare_summaries_can_require_min_samples():
    module = _load_profile_module()
    baseline = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "samples": 5,
            "down_base_us_p50": 100.0,
        }
    ]
    current = [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "samples": 1,
            "down_base_us_p50": 95.0,
        }
    ]

    comparisons, missing, sample_warnings, failed = module.compare_summaries(
        current, baseline, ["down_base_us"], max_p50_regression=1.10, min_samples=3
    )

    assert len(comparisons) == 1
    assert missing == []
    assert failed is True
    assert sample_warnings == [
        {
            "qlen": 16,
            "tp_count": 2,
            "rank": 2,
            "profile_kind": "tp_shard",
            "tp_part": 0,
            "metric": "down_base_us",
            "current_samples": 1,
            "baseline_samples": 5,
            "min_samples": 3,
            "reason": "insufficient_samples",
        }
    ]
