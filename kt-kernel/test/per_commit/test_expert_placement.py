"""Tests for expert placement planning utilities."""

import os
import sys
import types

import pytest
import torch

# Add parent directory to path for CI registration.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Add kt-kernel/python directly so this test does not require kt_kernel_ext.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# experts_base imports kt_kernel.kt_kernel_ext at module import time. The tests
# below only exercise pure-Python placement helpers, so a minimal stub is enough.
if "kt_kernel" not in sys.modules:
    kt_kernel_stub = types.ModuleType("kt_kernel")
    kt_kernel_stub.kt_kernel_ext = types.SimpleNamespace()
    sys.modules["kt_kernel"] = kt_kernel_stub

from expert_placement import plan_gpu_expert_placement
from experts_base import generate_gpu_experts_masks


register_cpu_ci(est_time=5, suite="default")


@pytest.mark.cpu
def test_frequency_strategy_preserves_global_topk_behavior():
    activation_freq = torch.tensor(
        [
            [0.1, 0.5, 0.3, 0.8],
            [0.2, 0.4, 0.9, 0.1],
        ]
    )

    mask = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)

    assert mask.dtype == torch.bool
    assert str(mask.device) == "cpu"
    assert mask.sum().item() == 3
    assert mask[1, 2]
    assert mask[0, 3]
    assert mask[0, 1]


@pytest.mark.cpu
def test_score_aware_layer_balanced_uses_ema_scores():
    current = torch.tensor(
        [
            [10.0, 1.0, 1.0, 1.0],
            [1.0, 9.0, 1.0, 1.0],
        ]
    )
    previous = torch.tensor(
        [
            [0.0, 20.0, 0.0, 0.0],
            [0.0, 0.0, 18.0, 0.0],
        ]
    )

    mask, report = plan_gpu_expert_placement(
        current,
        num_gpu_experts=2,
        strategy="score_aware_layer_balanced",
        previous_scores=previous,
        alpha=0.5,
        max_experts_per_layer=1,
    )

    # EMA scores make layer0 expert1 and layer1 expert2 the hottest experts.
    assert mask.sum().item() == 2
    assert mask[0, 1]
    assert mask[1, 2]
    assert report["strategy"] == "score_aware_layer_balanced"
    assert report["previous_scores_provided"] is True
    assert report["layer_max"] == 1


@pytest.mark.cpu
def test_score_aware_layer_balanced_respects_min_and_max_layer_bounds():
    activation_freq = torch.tensor(
        [
            [100.0, 90.0, 80.0, 70.0],
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    mask, report = plan_gpu_expert_placement(
        activation_freq,
        num_gpu_experts=6,
        strategy="score_aware_layer_balanced",
        min_experts_per_layer=1,
        max_experts_per_layer=2,
    )

    per_layer = mask.sum(dim=1).tolist()

    assert mask.sum().item() == 6
    assert per_layer == [2, 2, 2]
    assert report["layer_min"] == 2
    assert report["layer_max"] == 2


@pytest.mark.cpu
def test_score_aware_layer_balanced_is_deterministic_for_ties():
    activation_freq = torch.ones(3, 4)

    mask1 = generate_gpu_experts_masks(
        activation_freq,
        num_gpu_experts=5,
        strategy="score_aware_layer_balanced",
        max_experts_per_layer=2,
    )
    mask2 = generate_gpu_experts_masks(
        activation_freq,
        num_gpu_experts=5,
        strategy="score_aware_layer_balanced",
        max_experts_per_layer=2,
    )

    assert torch.equal(mask1, mask2)


@pytest.mark.cpu
def test_return_report_from_public_wrapper():
    activation_freq = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    mask, report = generate_gpu_experts_masks(
        activation_freq,
        num_gpu_experts=2,
        return_report=True,
    )

    assert mask.sum().item() == 2
    assert report["actual_num_gpu_experts"] == 2
    assert 0.0 <= report["expected_hit_rate"] <= 1.0


@pytest.mark.cpu
def test_invalid_previous_scores_shape_raises():
    activation_freq = torch.ones(2, 4)
    previous_scores = torch.ones(2, 3)

    with pytest.raises(ValueError, match="previous_scores shape"):
        plan_gpu_expert_placement(
            activation_freq,
            num_gpu_experts=2,
            strategy="score_aware_layer_balanced",
            previous_scores=previous_scores,
        )


@pytest.mark.cpu
def test_impossible_layer_constraints_raise():
    activation_freq = torch.ones(4, 8)

    with pytest.raises(ValueError, match="too small"):
        plan_gpu_expert_placement(
            activation_freq,
            num_gpu_experts=2,
            strategy="score_aware_layer_balanced",
            min_experts_per_layer=1,
        )

    with pytest.raises(ValueError, match="exceeds max_experts_per_layer"):
        plan_gpu_expert_placement(
            activation_freq,
            num_gpu_experts=12,
            strategy="score_aware_layer_balanced",
            max_experts_per_layer=2,
        )


@pytest.mark.cpu
def test_zero_and_oversized_budget_are_clamped_for_frequency():
    activation_freq = torch.ones(2, 3)

    zero_mask = generate_gpu_experts_masks(activation_freq, num_gpu_experts=0)
    full_mask = generate_gpu_experts_masks(activation_freq, num_gpu_experts=100)

    assert zero_mask.sum().item() == 0
    assert full_mask.sum().item() == 6