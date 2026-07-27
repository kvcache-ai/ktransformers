# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint


SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SFT_PATH / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = _load_module("kt_sft_activation_config_under_test", "config.py")
dist_utils = _load_module("kt_sft_activation_dist_utils_under_test", "dist_utils.py")


def _make_config(policy=None):
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key != "KT_REUSE_CHECKPOINT_FORWARD"
    }
    kwargs = {} if policy is None else {"kt_activation_policy": policy}
    with (
        patch.dict(os.environ, clean_env, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        return config.KTConfig(**kwargs)


def test_activation_policy_defaults_to_recompute_recompute():
    policy = _make_config().kt_activation_policy
    assert policy == config.KTActivationPolicy(cpu="recompute", gpu="recompute")


@pytest.mark.parametrize(
    ("cpu", "gpu"),
    [
        ("retain", "retain"),
        ("retain", "recompute"),
        ("recompute", "recompute"),
    ],
)
def test_activation_policy_accepts_supported_combinations(cpu, gpu):
    policy = _make_config({"cpu": cpu, "gpu": gpu}).kt_activation_policy
    assert (policy.cpu, policy.gpu) == (cpu, gpu)


@pytest.mark.parametrize(
    "policy",
    [
        {"cpu": "retain"},
        {"gpu": "retain"},
        {"cpu": "retain", "gpu": "retain", "extra": "recompute"},
        {"cpu": "drop", "gpu": "recompute"},
        {"cpu": "retain", "gpu": "drop"},
    ],
)
def test_activation_policy_rejects_invalid_shape_or_value(policy):
    with pytest.raises((ValueError, TypeError)):
        _make_config(policy)


def test_activation_policy_rejects_cpu_recompute_gpu_retain():
    with pytest.raises(NotImplementedError, match="not implemented"):
        _make_config({"cpu": "recompute", "gpu": "retain"})


def test_activation_policy_is_immutable():
    policy = _make_config({"cpu": "retain", "gpu": "recompute"}).kt_activation_policy
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.cpu = "recompute"


@pytest.mark.parametrize(
    ("legacy_value", "expected_cpu"),
    [("1", "retain"), ("true", "retain"), ("0", "recompute"), ("false", "recompute")],
)
def test_legacy_reuse_env_maps_policy_only_when_policy_is_absent(
    legacy_value,
    expected_cpu,
):
    with (
        patch.dict(
            os.environ,
            {"KT_REUSE_CHECKPOINT_FORWARD": legacy_value},
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.warns(FutureWarning, match="deprecated"),
    ):
        policy = config.KTConfig().kt_activation_policy
    assert (policy.cpu, policy.gpu) == (expected_cpu, "recompute")


def test_explicit_policy_conflicts_with_legacy_reuse_env():
    with (
        patch.dict(
            os.environ,
            {"KT_REUSE_CHECKPOINT_FORWARD": "1"},
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="conflicts"),
    ):
        config.KTConfig(
            kt_activation_policy={"cpu": "retain", "gpu": "recompute"}
        )


def _make_int8_config(**overrides):
    kwargs = {
        "kt_expert_weight_format": "int8",
        "kt_weight_path": "/dev/shm/kt-int8-test-run",
        "kt_train_mode": "lora",
        "kt_lora_rank": 8,
        "kt_num_gpu_experts": 0,
        "kt_share_backward_bb": True,
    }
    kwargs.update(overrides)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        return config.KTConfig(**kwargs)


def test_int8_format_selects_backend_and_accepts_cpu_activation_retain():
    cfg = _make_int8_config(
        kt_activation_policy={"cpu": "retain", "gpu": "recompute"}
    )
    assert cfg.kt_backend == "AMXINT8"
    assert cfg.kt_expert_weight_format == "int8"
    assert cfg.kt_weight_lifecycle == "persistent"
    assert (cfg.kt_activation_policy.cpu, cfg.kt_activation_policy.gpu) == (
        "retain",
        "recompute",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kt_backend": "AMXBF16"}, "conflicts"),
        ({"kt_train_mode": "full"}, "frozen-base LoRA"),
        ({"kt_train_mode": "hybrid"}, "frozen-base LoRA"),
        ({"kt_num_gpu_experts": 1}, "kt_num_gpu_experts=0"),
        ({"kt_use_lora_experts": True}, "GPU LoRA experts"),
        ({"kt_share_backward_bb": False}, "kt_share_backward_bb=true"),
        ({"kt_expert_checkpoint_path": "/checkpoint"}, "pre-quantized"),
        ({"kt_weight_path": None}, "kt_weight_path"),
    ],
)
def test_int8_rejects_unsupported_training_or_weight_sources(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_int8_config(**overrides)


def test_legacy_int8_backend_env_infers_format_when_nonconflicting():
    with (
        patch.dict(
            os.environ,
            {
                "ACCELERATE_KT_BACKEND": "AMXINT8",
                "ACCELERATE_KT_WEIGHT_PATH": "/dev/shm/kt-int8-legacy-run",
            },
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        cfg = config.KTConfig()
    assert cfg.kt_expert_weight_format == "int8"
    assert cfg.kt_backend == "AMXINT8"


def test_ephemeral_lifecycle_is_int8_only():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="supported only for INT8"),
    ):
        config.KTConfig(
            kt_expert_weight_format="bf16",
            kt_weight_lifecycle="ephemeral",
        )


def test_unknown_backend_fails_instead_of_falling_back_to_bf16():
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="unknown kt_backend"),
    ):
        config.KTConfig(kt_backend="AMXINT8_typo")


def test_explicit_checkpoint_context_takes_precedence_over_hook_probe():
    context_fn = dist_utils.get_activation_checkpoint_context_fn()
    first_forward, recompute = context_fn()
    with patch.object(
        dist_utils.torch._C._autograd,
        "_top_saved_tensors_default_hooks",
        side_effect=AssertionError("fallback must not be queried"),
    ):
        with first_forward:
            assert dist_utils._checkpoint_hook_mode() == "first_forward"
        with recompute:
            assert dist_utils._checkpoint_hook_mode() == "recompute"


def test_explicit_checkpoint_context_marks_real_non_reentrant_checkpoint_phases():
    phases = []

    def fn(value):
        phases.append(dist_utils._checkpoint_hook_mode())
        return value.sin()

    value = torch.tensor([0.5], requires_grad=True)
    output = checkpoint(
        fn,
        value,
        use_reentrant=False,
        context_fn=dist_utils.get_activation_checkpoint_context_fn(),
    )
    output.sum().backward()
    assert phases == ["first_forward", "recompute"]


@pytest.mark.parametrize(
    ("phase", "action"),
    [
        ("none", "normal"),
        ("first_forward", "cache_first_forward"),
        ("recompute", "reuse_recompute"),
    ],
)
def test_checkpoint_state_agreement_supports_single_rank(phase, action):
    qlens = dist_utils._all_gather_checkpoint_state(
        17,
        layer_idx=3,
        phase=phase,
        action=action,
        device=dist_utils.torch.device("cpu"),
        world_size=1,
    )
    assert qlens == [17]


def test_checkpoint_state_agreement_rejects_invalid_action():
    with pytest.raises(RuntimeError, match="action=invalid"):
        dist_utils._all_gather_checkpoint_state(
            17,
            layer_idx=3,
            phase="none",
            action="skip_everything",
            device=dist_utils.torch.device("cpu"),
            world_size=1,
        )
