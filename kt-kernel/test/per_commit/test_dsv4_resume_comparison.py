"""Resume comparisons distinguish PEFT serialization order from state changes."""

import importlib.util
from pathlib import Path

import torch


spec = importlib.util.spec_from_file_location(
    "dsv4_resume_comparison", Path(__file__).parents[1] / "compare_dsv4_resume.py"
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_adapter_target_order_is_not_semantic():
    check = comparison.Comparison(0.0, 0.0)
    left = {"target_modules": ["q_proj", "v_proj"], "r": 8}
    right = {"target_modules": ["v_proj", "q_proj"], "r": 8}
    check.adapter_config(left, right)
    assert not check.failures
    assert right["target_modules"] == ["v_proj", "q_proj"]


def test_different_targets_and_regex_remain_failures():
    for targets in (["q_proj", "k_proj"], "q_proj|v_proj"):
        check = comparison.Comparison(0.0, 0.0)
        check.adapter_config(
            {"target_modules": ["q_proj", "v_proj"]}, {"target_modules": targets}
        )
        assert check.failures


def test_optimizer_parameter_order_remains_semantic():
    check = comparison.Comparison(0.0, 0.0)
    check.tree("optimizer", {"params": [0, 1]}, {"params": [1, 0]})
    assert check.failures


def test_numerical_differences_remain_failures():
    check = comparison.Comparison(0.0, 0.0)
    check.tensor("adapter", torch.tensor([1.0]), torch.tensor([1.001]))
    assert check.nonidentical_tensors == 1
    assert check.failures


def test_identical_nonfinite_tensors_are_not_a_pass():
    for value in (float("inf"), float("nan")):
        check = comparison.Comparison(0.0, 0.0)
        check.tensor("adapter", torch.tensor([value]), torch.tensor([value]))
        assert check.failures == ["adapter: non-finite tensor"]
