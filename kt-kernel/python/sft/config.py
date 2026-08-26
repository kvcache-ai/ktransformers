# KT-Kernel SFT configuration
# SPDX-License-Identifier: Apache-2.0

"""
KTConfig: kt-kernel's own configuration dataclass.

This is the kt-kernel equivalent of DeepSpeed's JSON config —
it holds all kt-kernel-specific settings and is passed through
KTransformersPlugin.kt_config (similar to DeepSpeedPlugin.hf_ds_config).
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
import logging
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .backend import FP8_BACKEND, INT8_BACKEND, normalize_sft_backend


logger = logging.getLogger(__name__)

_CPU_TOPOLOGY_ROOT = Path("/sys/devices/system/cpu")
_ACTIVATION_POLICY_ENV = "ACCELERATE_KT_ACTIVATION_POLICY"
_LEGACY_REUSE_ENV = "KT_REUSE_CHECKPOINT_FORWARD"
_KNOWN_SFT_BACKENDS = {
    "auto",
    "int8",
    "fp8",
    "amxbf16",
    "amxint8",
    "amxfp8",
    "amxint4",
    "amxbf16_skiplora",
    "amxint8_skiplora",
    "amxint4_skiplora",
}

ActivationRetention = Literal["retain", "recompute"]
ExpertWeightFormat = Literal["bf16", "int8", "fp8"]
WeightLifecycle = Literal["persistent", "ephemeral"]


@dataclass(frozen=True)
class KTActivationPolicy:
    """Activation lifetime policy shared by every KT distributed rank."""

    cpu: ActivationRetention = "recompute"
    gpu: ActivationRetention = "recompute"

    def __post_init__(self) -> None:
        valid = {"retain", "recompute"}
        if self.cpu not in valid:
            raise ValueError(
                f"activation_policy.cpu must be one of {sorted(valid)}, got {self.cpu!r}"
            )
        if self.gpu not in valid:
            raise ValueError(
                f"activation_policy.gpu must be one of {sorted(valid)}, got {self.gpu!r}"
            )
        if self.cpu == "recompute" and self.gpu == "retain":
            raise NotImplementedError(
                "activation_policy cpu=recompute, gpu=retain is not implemented"
            )

    @classmethod
    def from_value(
        cls,
        value: "KTActivationPolicy | Mapping[str, str]",
    ) -> "KTActivationPolicy":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "kt_activation_policy must be a KTActivationPolicy or a mapping "
                "with exactly the keys 'cpu' and 'gpu'"
            )
        expected = {"cpu", "gpu"}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if unexpected:
                details.append(f"unexpected={unexpected}")
            raise ValueError(
                "kt_activation_policy must contain exactly 'cpu' and 'gpu'"
                + (f" ({', '.join(details)})" if details else "")
            )
        return cls(cpu=value["cpu"], gpu=value["gpu"])


def _legacy_activation_policy_from_env(value: str) -> KTActivationPolicy:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        cpu = "retain"
    elif normalized in {"0", "false", "no", "off"}:
        cpu = "recompute"
    else:
        raise ValueError(
            f"{_LEGACY_REUSE_ENV} must be a boolean value, got {value!r}"
        )
    warnings.warn(
        f"{_LEGACY_REUSE_ENV} is deprecated; configure kt_activation_policy "
        "through the training frontend instead",
        FutureWarning,
        stacklevel=3,
    )
    return KTActivationPolicy(cpu=cpu, gpu="recompute")


def _activation_policy_from_env(value: str) -> KTActivationPolicy:
    try:
        policy = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_ACTIVATION_POLICY_ENV} must be a JSON object with exactly "
            "the keys 'cpu' and 'gpu'"
        ) from exc
    return KTActivationPolicy.from_value(policy)


def _env_int(key: str, default: int | None) -> int | None:
    value = os.environ.get(key, None)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(key: str, default: float | None) -> float | None:
    value = os.environ.get(key, None)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(key: str, default: bool) -> bool:
    value = os.environ.get(key, None)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes")


def _available_cpu_ids() -> set[int]:
    """Return CPUs available to this process, respecting affinity/cpuset limits."""
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return set(range(os.cpu_count() or 1))


def _read_cpu_topology(cpu_id: int) -> tuple[int, int] | None:
    topology = _CPU_TOPOLOGY_ROOT / f"cpu{cpu_id}" / "topology"
    try:
        package_id = int((topology / "physical_package_id").read_text().strip())
        core_id = int((topology / "core_id").read_text().strip())
    except (OSError, ValueError):
        return None
    return package_id, core_id


def detect_physical_cpu_count() -> int:
    """Count physical cores available to the current process.

    Linux exposes a stable ``(physical_package_id, core_id)`` pair for every
    logical CPU. Counting those pairs avoids assigning one OpenMP worker to
    each SMT sibling. If topology is unavailable, fall back to the number of
    affinity-visible logical CPUs.
    """
    cpu_ids = _available_cpu_ids()
    physical_cores = {
        topology
        for cpu_id in cpu_ids
        if (topology := _read_cpu_topology(cpu_id)) is not None
    }
    return max(1, len(physical_cores) if physical_cores else len(cpu_ids))


def _set_torch_num_threads(num_threads: int) -> None:
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(num_threads)


def configure_omp_threads() -> int:
    """Configure OpenMP for KT SFT CPU tensor work.

    ``accelerate launch`` defaults GPU jobs to ``OMP_NUM_THREADS=1`` when the
    caller did not choose a value. That makes Full-FT CPU gradient accumulation,
    AdamW, and zeroing effectively serial. Treat that value as the launcher
    default and select the affinity-visible physical core count instead.

    ``ACCELERATE_KT_OMP_NUM_THREADS`` is the unambiguous KT-specific override,
    including when an intentional single-thread run is required. An existing
    generic ``OMP_NUM_THREADS`` value greater than one is also preserved.
    """
    kt_override = _env_int("ACCELERATE_KT_OMP_NUM_THREADS", None)
    current_omp = _env_int("OMP_NUM_THREADS", None)

    if kt_override is not None:
        num_threads = kt_override
        source = "ACCELERATE_KT_OMP_NUM_THREADS"
    elif current_omp is not None and current_omp > 1:
        num_threads = current_omp
        source = "OMP_NUM_THREADS"
    else:
        num_threads = detect_physical_cpu_count()
        source = "available physical cores"

    if num_threads < 1:
        raise ValueError(f"OpenMP thread count must be positive, got {num_threads}")

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    _set_torch_num_threads(num_threads)
    logger.info("KT SFT configured OMP_NUM_THREADS=%d from %s", num_threads, source)
    return num_threads


@dataclass
class KTConfig:
    """
    KT-Kernel configuration for SFT training.

    All field names use the ``kt_`` prefix so they match the dict keys used in
    HfTrainerKTConfig / YAML configs.  This means ``KTConfig(**dict)`` works
    directly — no name-mapping or prefix-stripping needed.

    Can be created from:
    - Direct construction: KTConfig(kt_backend="AMXBF16", kt_weight_path="/path/...")
    - Dict: KTConfig(**config_dict)
    - Environment variables: KTConfig() reads ACCELERATE_KT_* env vars as defaults
    """

    # Backend selection
    kt_backend: str | None = None
    kt_num_threads: int | None = None
    kt_tp_enabled: bool | None = None
    kt_threadpool_count: int | None = None

    # Weight loading
    kt_weight_path: str | None = None
    kt_non_expert_weight_path: str | None = None
    kt_expert_weight_format: ExpertWeightFormat | str | None = None
    kt_weight_lifecycle: WeightLifecycle | str | None = None
    kt_expert_checkpoint_path: str | None = None  # HF expert checkpoint or KT Full checkpoint directory
    kt_num_gpu_experts: int | None = None
    kt_skip_expert_loading: bool | None = None
    kt_share_backward_bb: bool | None = None  # default True — always saves memory
    kt_share_cache_pool: bool | None = None  # auto-set by trainer_config_process, not user-facing
    kt_force_fused_expert_lora: bool | None = None

    # Cache
    kt_max_cache_depth: int | None = None
    kt_model_max_length: int | None = None
    kt_activation_policy: KTActivationPolicy | Mapping[str, str] | None = None

    # LoRA
    kt_lora_rank: int | None = None
    kt_lora_alpha: float | None = None
    kt_lora_dropout: float | None = None

    # Training mode
    kt_train_mode: str | None = None  # "lora" | "full" | "hybrid"
    kt_full_weight_grad: bool | None = None  # auto-set True when train_mode in (full, hybrid)

    # LoRA Experts (GPU-side extra experts)
    kt_use_lora_experts: bool | None = None
    kt_lora_expert_num: int | None = None
    kt_lora_expert_intermediate_size: int | None = None

    # Runtime state (set during wrapping, not by user)
    kt_checkpoint_files: list[str] | None = None
    kt_sharded_metadata: dict | None = None

    # Custom wrapping
    kt_wrap_fn: Callable[..., Any] | None = None
    kt_wrap_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_object(cls, obj: Any) -> "KTConfig":
        """Create KTConfig from a mapping or compatible public container."""
        return cls._from_object(obj, seen=set())

    @classmethod
    def _from_object(cls, obj: Any, *, seen: set[int]) -> "KTConfig":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, Mapping):
            return cls(**cls._validated_mapping(obj))

        object_id = id(obj)
        if object_id in seen:
            raise ValueError("Cyclic KT configuration container")
        seen.add(object_id)

        field_names = {field.name for field in dataclasses.fields(cls)}
        attributes = vars(obj) if hasattr(obj, "__dict__") else {}
        unknown = sorted(
            name
            for name in attributes
            if name.startswith("kt_")
            and name not in field_names
            and name != "kt_config"
        )
        if unknown:
            raise TypeError(f"Unknown KTConfig fields: {unknown}")

        missing = object()
        has_public_container = False
        for container_name in ("kt_config", "config"):
            if inspect.getattr_static(obj, container_name, missing) is missing:
                continue
            has_public_container = True
            nested = getattr(obj, container_name, None)
            if nested is None:
                continue
            if nested is obj:
                raise ValueError(f"Cyclic KT configuration container: {container_name}")

            if isinstance(nested, Mapping):
                kwargs = cls._validated_mapping(nested)
                for name in (
                    "kt_checkpoint_files",
                    "kt_sharded_metadata",
                    "kt_skip_expert_loading",
                ):
                    value = getattr(obj, name, None)
                    if value is not None:
                        kwargs[name] = value
                return cls(**kwargs)

            resolved = cls._from_object(nested, seen=seen)
            overrides: dict[str, Any] = {}
            for name in (
                "kt_checkpoint_files",
                "kt_sharded_metadata",
                "kt_skip_expert_loading",
            ):
                value = getattr(obj, name, None)
                if value is not None and value is not getattr(resolved, name, None):
                    overrides[name] = value
            if not overrides:
                return resolved
            result = copy.copy(resolved)
            for name, value in overrides.items():
                setattr(result, name, value)
            return result

        if has_public_container:
            return cls()
        raise TypeError(
            "KT configuration must be a KTConfig, mapping, or public container "
            "with a static 'kt_config' or 'config' attribute"
        )

    @classmethod
    def _validated_mapping(cls, obj: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(obj)
        invalid_keys = sorted(repr(name) for name in payload if not isinstance(name, str))
        if invalid_keys:
            raise TypeError(f"KTConfig field names must be strings: {invalid_keys}")
        if "enabled" in payload:
            enabled = payload.pop("enabled")
            if enabled is not None and not isinstance(enabled, bool):
                raise TypeError("KT framework field 'enabled' must be a bool or None")
        field_names = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload).difference(field_names))
        if unknown:
            raise TypeError(f"Unknown KTConfig fields: {unknown}")
        return payload

    def __post_init__(self):
        configure_omp_threads()
        explicit_backend = self.kt_backend is not None
        env_backend = os.environ.get("ACCELERATE_KT_BACKEND")
        if self.kt_expert_weight_format is None:
            self.kt_expert_weight_format = os.environ.get(
                "ACCELERATE_KT_EXPERT_WEIGHT_FORMAT"
            )
        if self.kt_expert_weight_format is not None:
            self.kt_expert_weight_format = str(self.kt_expert_weight_format).strip().lower()
            if self.kt_expert_weight_format not in {"bf16", "int8", "fp8"}:
                raise ValueError(
                    "kt_expert_weight_format must be one of ['bf16', 'fp8', 'int8'], "
                    f"got {self.kt_expert_weight_format!r}"
                )
        if self.kt_weight_lifecycle is None:
            self.kt_weight_lifecycle = os.environ.get(
                "ACCELERATE_KT_WEIGHT_LIFECYCLE", "persistent"
            )
        self.kt_weight_lifecycle = str(self.kt_weight_lifecycle).strip().lower()
        if self.kt_weight_lifecycle not in {"persistent", "ephemeral"}:
            raise ValueError(
                "kt_weight_lifecycle must be one of ['ephemeral', 'persistent'], "
                f"got {self.kt_weight_lifecycle!r}"
            )

        forwarded_policy = os.environ.get(_ACTIVATION_POLICY_ENV)
        forwarded_policy_is_explicit = (
            forwarded_policy is not None and forwarded_policy.strip() != ""
        )
        legacy_reuse = os.environ.get(_LEGACY_REUSE_ENV)
        legacy_reuse_is_explicit = legacy_reuse is not None and legacy_reuse.strip() != ""
        if self.kt_activation_policy is not None and forwarded_policy_is_explicit:
            raise ValueError(
                f"kt_activation_policy conflicts with {_ACTIVATION_POLICY_ENV}; "
                f"unset {_ACTIVATION_POLICY_ENV}"
            )
        if self.kt_activation_policy is not None and legacy_reuse_is_explicit:
            raise ValueError(
                f"kt_activation_policy conflicts with legacy {_LEGACY_REUSE_ENV}; "
                f"unset {_LEGACY_REUSE_ENV}"
            )
        if forwarded_policy_is_explicit and legacy_reuse_is_explicit:
            raise ValueError(
                f"{_ACTIVATION_POLICY_ENV} conflicts with legacy "
                f"{_LEGACY_REUSE_ENV}; unset {_LEGACY_REUSE_ENV}"
            )
        if self.kt_activation_policy is None:
            if forwarded_policy_is_explicit:
                self.kt_activation_policy = _activation_policy_from_env(
                    forwarded_policy
                )
            elif legacy_reuse_is_explicit:
                self.kt_activation_policy = _legacy_activation_policy_from_env(legacy_reuse)
            else:
                self.kt_activation_policy = KTActivationPolicy()
        else:
            self.kt_activation_policy = KTActivationPolicy.from_value(
                self.kt_activation_policy
            )
        if self.kt_backend is None:
            if env_backend:
                self.kt_backend = env_backend
            elif self.kt_expert_weight_format in {"int8", "fp8"}:
                self.kt_backend = "auto"
            else:
                self.kt_backend = "AMXBF16"
        backend_lower = str(self.kt_backend).lower()
        if backend_lower not in _KNOWN_SFT_BACKENDS:
            raise ValueError(
                f"unknown kt_backend {self.kt_backend!r}; expected one of "
                f"{sorted(_KNOWN_SFT_BACKENDS)}"
            )
        self.kt_backend = normalize_sft_backend(
            self.kt_backend,
            expert_weight_format=self.kt_expert_weight_format,
        )
        backend_lower = str(self.kt_backend).lower()
        if self.kt_expert_weight_format is None:
            if backend_lower == INT8_BACKEND.lower():
                # Backward compatibility for the legacy environment-only entry.
                self.kt_expert_weight_format = "int8"
            elif backend_lower == FP8_BACKEND.lower():
                self.kt_expert_weight_format = "fp8"
            elif backend_lower == "amxbf16":
                self.kt_expert_weight_format = "bf16"
        expected_backend = {
            "bf16": "amxbf16",
            "int8": INT8_BACKEND.lower(),
            "fp8": FP8_BACKEND.lower(),
        }.get(self.kt_expert_weight_format)
        if expected_backend is not None and backend_lower != expected_backend:
            source = "kt_backend" if explicit_backend else "ACCELERATE_KT_BACKEND"
            raise ValueError(
                f"kt_expert_weight_format={self.kt_expert_weight_format!r} conflicts "
                f"with {source}={self.kt_backend!r}"
            )
        if self.kt_num_threads is None:
            self.kt_num_threads = _env_int("ACCELERATE_KT_NUM_THREADS", 1)
        if self.kt_tp_enabled is None:
            self.kt_tp_enabled = _env_bool("ACCELERATE_KT_TP_ENABLED", False)
        if self.kt_threadpool_count is None:
            self.kt_threadpool_count = _env_int("ACCELERATE_KT_THREADPOOL_COUNT", 1)
        if self.kt_weight_path is None:
            self.kt_weight_path = os.environ.get("ACCELERATE_KT_WEIGHT_PATH", None)
        if self.kt_non_expert_weight_path is None:
            self.kt_non_expert_weight_path = os.environ.get(
                "ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH", None
            )
        if self.kt_expert_checkpoint_path is None:
            self.kt_expert_checkpoint_path = os.environ.get("ACCELERATE_KT_EXPERT_CHECKPOINT_PATH", None)
        if self.kt_num_gpu_experts is None:
            self.kt_num_gpu_experts = _env_int("ACCELERATE_KT_NUM_GPU_EXPERTS", 0)
        if self.kt_max_cache_depth is None:
            self.kt_max_cache_depth = _env_int("ACCELERATE_KT_MAX_CACHE_DEPTH", 2)
        if self.kt_share_backward_bb is None:
            self.kt_share_backward_bb = _env_bool("ACCELERATE_KT_SHARE_BACKWARD_BB", True)
        if self.kt_share_cache_pool is None:
            self.kt_share_cache_pool = False
        if self.kt_force_fused_expert_lora is None:
            self.kt_force_fused_expert_lora = _env_bool(
                "ACCELERATE_KT_FORCE_FUSED_EXPERT_LORA", False
            )
        if self.kt_use_lora_experts is None:
            self.kt_use_lora_experts = _env_bool("ACCELERATE_KT_USE_LORA_EXPERTS", False)
        if self.kt_lora_expert_num is None:
            self.kt_lora_expert_num = _env_int("ACCELERATE_KT_LORA_EXPERT_NUM", None)
        if self.kt_lora_expert_intermediate_size is None:
            self.kt_lora_expert_intermediate_size = _env_int("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", None)
        if self.kt_lora_rank is None:
            self.kt_lora_rank = _env_int("ACCELERATE_KT_LORA_RANK", None)
        if self.kt_lora_alpha is None:
            self.kt_lora_alpha = _env_float("ACCELERATE_KT_LORA_ALPHA", None)
        if self.kt_lora_alpha is None and self.kt_lora_rank is not None:
            self.kt_lora_alpha = float(self.kt_lora_rank * 2)
        if self.kt_lora_dropout is None:
            self.kt_lora_dropout = _env_float("ACCELERATE_KT_LORA_DROPOUT", 0.0)
        if not 0.0 <= self.kt_lora_dropout < 1.0:
            raise ValueError(
                f"kt_lora_dropout must be in [0, 1), got {self.kt_lora_dropout}"
            )
        if self.kt_train_mode is None:
            self.kt_train_mode = os.environ.get("ACCELERATE_KT_TRAIN_MODE", "lora")
        if self.kt_full_weight_grad is None:
            self.kt_full_weight_grad = self.kt_train_mode in ("full", "hybrid")
        if self.kt_model_max_length is None:
            self.kt_model_max_length = _env_int("ACCELERATE_KT_MODEL_MAX_LENGTH", None)
        if self.kt_skip_expert_loading is None:
            if "ACCELERATE_KT_SKIP_EXPERT_LOADING" in os.environ:
                self.kt_skip_expert_loading = _env_bool("ACCELERATE_KT_SKIP_EXPERT_LOADING", True)

        if self.kt_non_expert_weight_path and self.kt_expert_weight_format != "int8":
            raise ValueError(
                "kt_non_expert_weight_path is supported only with "
                "kt_expert_weight_format='int8'"
            )
        if self.kt_expert_weight_format == "int8":
            if str(self.kt_backend).lower() != INT8_BACKEND.lower():
                raise ValueError(
                    "INT8 SFT requires kt_backend='auto' or kt_backend='INT8'"
                )
            if self.kt_train_mode != "lora" or bool(self.kt_full_weight_grad):
                raise ValueError(
                    "INT8 SFT supports frozen-base LoRA only; Full and Hybrid are not supported"
                )
            if int(self.kt_num_gpu_experts or 0) != 0:
                raise ValueError("INT8 SFT requires kt_num_gpu_experts=0")
            if bool(self.kt_use_lora_experts):
                raise ValueError(
                    "INT8 SFT does not support GPU LoRA experts; use pure expert LoRA"
                )
            if not bool(self.kt_share_backward_bb):
                raise ValueError("INT8 SFT requires kt_share_backward_bb=true")
            if self.kt_expert_checkpoint_path:
                raise ValueError(
                    "INT8 SFT requires pre-quantized .kt weights; "
                    "kt_expert_checkpoint_path is not supported"
                )
            if not self.kt_weight_path:
                raise ValueError(
                    "INT8 SFT requires kt_weight_path pointing to pre-quantized .kt weights"
                )
        if self.kt_expert_weight_format == "fp8":
            if str(self.kt_backend).lower() != FP8_BACKEND.lower():
                raise ValueError(
                    "FP8 SFT requires kt_backend='auto' or kt_backend='FP8'"
                )
            if self.kt_train_mode != "lora" or bool(self.kt_full_weight_grad):
                raise ValueError(
                    "FP8 SFT supports frozen-base LoRA only; Full and Hybrid are not supported"
                )
            if not self.kt_lora_rank or int(self.kt_lora_rank) <= 0:
                raise ValueError("FP8 SFT requires kt_lora_rank > 0")
            if int(self.kt_num_gpu_experts or 0) != 0:
                raise ValueError("FP8 SFT requires kt_num_gpu_experts=0")
            if bool(self.kt_use_lora_experts):
                raise ValueError(
                    "FP8 SFT does not support GPU LoRA experts; use pure expert LoRA"
                )
            if not bool(self.kt_share_backward_bb):
                raise ValueError("FP8 SFT requires kt_share_backward_bb=true")
            if self.kt_weight_lifecycle != "persistent":
                raise ValueError("FP8 SFT requires kt_weight_lifecycle='persistent'")
        if self.kt_weight_lifecycle == "ephemeral":
            if self.kt_expert_weight_format != "int8":
                raise ValueError(
                    "kt_weight_lifecycle='ephemeral' is supported only for INT8 .kt weights"
                )
