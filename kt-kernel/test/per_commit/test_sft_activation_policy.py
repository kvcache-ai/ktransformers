# SPDX-License-Identifier: Apache-2.0

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint


SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
PACKAGE_NAME = "kt_sft_activation_under_test"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(SFT_PATH)]
sys.modules[PACKAGE_NAME] = package


def _load_module(filename: str):
    name = f"{PACKAGE_NAME}.{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, SFT_PATH / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


config = _load_module("config.py")
dist_utils = _load_module("dist_utils.py")


def _make_config(policy=None):
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "ACCELERATE_KT_ACTIVATION_POLICY",
            "KT_REUSE_CHECKPOINT_FORWARD",
        }
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


def test_config_from_hf_wrapper_preserves_outer_runtime_metadata():
    public_config = {
        "kt_backend": "AMXBF16",
        "kt_num_threads": 7,
        "kt_skip_expert_loading": False,
    }
    checkpoint_files = ["/models/qwen/model-00001-of-00002.safetensors"]
    sharded_metadata = {"weight_map": {"expert.weight": checkpoint_files[0]}}

    class HfTrainerKTConfig:
        config = public_config
        _runtime_metadata = {
            "kt_checkpoint_files": checkpoint_files,
            "kt_sharded_metadata": sharded_metadata,
            "kt_skip_expert_loading": True,
        }

        def __getattr__(self, name):
            if name in self._runtime_metadata:
                return self._runtime_metadata[name]
            try:
                return self.config[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        cfg = config.KTConfig.from_object(HfTrainerKTConfig())

    assert public_config == {
        "kt_backend": "AMXBF16",
        "kt_num_threads": 7,
        "kt_skip_expert_loading": False,
    }
    assert cfg.kt_backend == "AMXBF16"
    assert cfg.kt_num_threads == 7
    assert cfg.kt_checkpoint_files is checkpoint_files
    assert cfg.kt_sharded_metadata is sharded_metadata
    assert cfg.kt_skip_expert_loading is True


@pytest.mark.parametrize("unknown", ["kt_backned", "totally_unknown"])
def test_config_from_hf_wrapper_rejects_unknown_public_fields(unknown):
    class HfTrainerKTConfig:
        config = {unknown: "FP8"}
        kt_checkpoint_files = ["/models/qwen/model.safetensors"]
        kt_skip_expert_loading = True

    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(TypeError, match=unknown),
    ):
        config.KTConfig.from_object(HfTrainerKTConfig())


def test_hf_public_mapping_cannot_impersonate_container_attributes():
    class HfTrainerKTConfig:
        config = {
            "kt_config": {"kt_backend": "FP8"},
            "kt_backned": "AMXBF16",
        }

        def __getattr__(self, name):
            try:
                return self.config[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    with pytest.raises(TypeError, match="kt_backned"):
        config.KTConfig.from_object(HfTrainerKTConfig())


@pytest.mark.parametrize("enabled", [True, False, None])
def test_config_mapping_accepts_framework_enabled_without_mutation(enabled):
    payload = {"enabled": enabled, "kt_backend": "AMXBF16"}
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        cfg = config.KTConfig.from_object(payload)
    assert payload == {"enabled": enabled, "kt_backend": "AMXBF16"}
    assert cfg.kt_backend == "AMXBF16"


def test_config_mapping_rejects_invalid_enabled_and_unknown_fields():
    with pytest.raises(TypeError, match="enabled"):
        config.KTConfig.from_object({"enabled": "yes"})
    with pytest.raises(TypeError, match="kt_backned"):
        config.KTConfig.from_object({"kt_backned": "FP8"})
    with pytest.raises(TypeError, match="field names must be strings"):
        config.KTConfig.from_object({1: "AMXBF16"})


def test_nested_typed_config_runtime_overlay_is_copy_only():
    base = _make_config()
    assert config.KTConfig.from_object(base) is base
    assert config.KTConfig.from_object(SimpleNamespace(config=base)) is base

    checkpoint_files = []
    sharded_metadata = {}
    outer = SimpleNamespace(
        config=base,
        kt_checkpoint_files=checkpoint_files,
        kt_sharded_metadata=sharded_metadata,
        kt_skip_expert_loading=False,
    )
    with patch.object(
        config,
        "configure_omp_threads",
        side_effect=AssertionError("typed KTConfig must not be reinitialized"),
    ):
        resolved = config.KTConfig.from_object(outer)

    assert resolved is not base
    assert base.kt_checkpoint_files is None
    assert base.kt_sharded_metadata is None
    assert base.kt_skip_expert_loading is None
    assert resolved.kt_checkpoint_files is checkpoint_files
    assert resolved.kt_sharded_metadata is sharded_metadata
    assert resolved.kt_skip_expert_loading is False


def test_nested_config_prefers_kt_config_and_rejects_cycles():
    outer = SimpleNamespace(
        kt_config={"kt_num_threads": 3},
        config={"kt_num_threads": 4},
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        assert config.KTConfig.from_object(outer).kt_num_threads == 3

    cyclic = SimpleNamespace()
    cyclic.kt_config = cyclic
    with pytest.raises(ValueError, match="Cyclic"):
        config.KTConfig.from_object(cyclic)

    empty_plugin = SimpleNamespace(enabled=True, kt_config=None)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        assert isinstance(config.KTConfig.from_object(empty_plugin), config.KTConfig)


@pytest.mark.parametrize(
    "value",
    [object(), SimpleNamespace(), SimpleNamespace(kt_backend="AMXBF16")],
)
def test_config_rejects_objects_without_an_explicit_public_container(value):
    with pytest.raises(TypeError, match="public container"):
        config.KTConfig.from_object(value)


def test_nested_mapping_only_accepts_outer_runtime_overrides():
    checkpoint_files = []
    outer = SimpleNamespace(
        kt_config={
            "kt_backend": "AMXBF16",
            "kt_num_threads": 0,
            "kt_skip_expert_loading": True,
        },
        kt_backend="FP8",
        kt_num_threads=12,
        kt_checkpoint_files=checkpoint_files,
        kt_skip_expert_loading=False,
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        resolved = config.KTConfig.from_object(outer)

    assert resolved.kt_backend == "AMXBF16"
    assert resolved.kt_num_threads == 0
    assert resolved.kt_checkpoint_files is checkpoint_files
    assert resolved.kt_skip_expert_loading is False

    outer.kt_config = {"kt_backned": "FP8"}
    with pytest.raises(TypeError, match="kt_backned"):
        config.KTConfig.from_object(outer)

    outer.kt_config = {"kt_backend": "AMXBF16"}
    outer.kt_backned = "FP8"
    with pytest.raises(TypeError, match="kt_backned"):
        config.KTConfig.from_object(outer)


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


def test_forwarded_activation_policy_env_is_applied():
    with (
        patch.dict(
            os.environ,
            {
                "ACCELERATE_KT_ACTIVATION_POLICY": (
                    '{"cpu":"retain","gpu":"recompute"}'
                )
            },
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
    ):
        policy = config.KTConfig().kt_activation_policy
    assert policy == config.KTActivationPolicy(cpu="retain", gpu="recompute")


def test_forwarded_activation_policy_env_rejects_invalid_json():
    with (
        patch.dict(
            os.environ,
            {"ACCELERATE_KT_ACTIVATION_POLICY": "retain/recompute"},
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="must be a JSON object"),
    ):
        config.KTConfig()


def test_explicit_policy_conflicts_with_forwarded_policy_env():
    with (
        patch.dict(
            os.environ,
            {
                "ACCELERATE_KT_ACTIVATION_POLICY": (
                    '{"cpu":"retain","gpu":"recompute"}'
                )
            },
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="conflicts"),
    ):
        config.KTConfig(
            kt_activation_policy={"cpu": "retain", "gpu": "recompute"}
        )


def test_forwarded_policy_env_conflicts_with_legacy_reuse_env():
    with (
        patch.dict(
            os.environ,
            {
                "ACCELERATE_KT_ACTIVATION_POLICY": (
                    '{"cpu":"retain","gpu":"recompute"}'
                ),
                "KT_REUSE_CHECKPOINT_FORWARD": "1",
            },
            clear=True,
        ),
        patch.object(config, "configure_omp_threads", return_value=1),
        pytest.raises(ValueError, match="conflicts with legacy"),
    ):
        config.KTConfig()


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
    assert cfg.kt_backend == "INT8"
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
    with pytest.warns(FutureWarning, match="deprecated"):
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
    assert cfg.kt_backend == "INT8"


@pytest.mark.parametrize("backend", ["auto", "AUTO", "INT8", "int8"])
def test_int8_accepts_hardware_neutral_backend_names(backend):
    cfg = _make_int8_config(kt_backend=backend)
    assert cfg.kt_backend == "INT8"


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


def _make_fp8_config(**overrides):
    kwargs = {
        "kt_expert_weight_format": "fp8",
        "kt_weight_path": "/models/native-fp8",
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


def test_fp8_format_selects_backend_and_accepts_checkpoint_provenance():
    cfg = _make_fp8_config(
        kt_activation_policy={"cpu": "retain", "gpu": "recompute"}
    )
    assert cfg.kt_backend == "FP8"
    assert cfg.kt_expert_weight_format == "fp8"
    assert cfg.kt_weight_path == "/models/native-fp8"
    assert cfg.kt_weight_lifecycle == "persistent"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kt_backend": "AMXBF16"}, "conflicts"),
        ({"kt_train_mode": "full"}, "frozen-base LoRA"),
        ({"kt_train_mode": "hybrid"}, "frozen-base LoRA"),
        ({"kt_lora_rank": 0}, "kt_lora_rank > 0"),
        ({"kt_num_gpu_experts": 1}, "kt_num_gpu_experts=0"),
        ({"kt_use_lora_experts": True}, "GPU LoRA experts"),
        ({"kt_share_backward_bb": False}, "kt_share_backward_bb=true"),
        ({"kt_weight_lifecycle": "ephemeral"}, "persistent"),
    ],
)
def test_fp8_rejects_unsupported_training_modes(overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_fp8_config(**overrides)


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
