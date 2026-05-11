# KT-Kernel SFT configuration
# SPDX-License-Identifier: Apache-2.0

"""
KTConfig: kt-kernel's own configuration dataclass.

This is the kt-kernel equivalent of DeepSpeed's JSON config —
it holds all kt-kernel-specific settings and is passed through
KTransformersPlugin.kt_config (similar to DeepSpeedPlugin.hf_ds_config).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Callable


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
    kt_expert_checkpoint_path: str | None = None
    kt_num_gpu_experts: int | None = None
    kt_skip_expert_loading: bool | None = None
    kt_share_backward_bb: bool | None = None  # default True — always saves memory
    kt_share_cache_pool: bool | None = None  # auto-set by trainer_config_process, not user-facing

    # Cache
    kt_max_cache_depth: int | None = None
    kt_model_max_length: int | None = None

    # LoRA
    kt_lora_rank: int | None = None
    kt_lora_alpha: float | None = None

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
        """Create KTConfig from an attribute-based object (HfTrainerKTConfig, etc.)."""
        _field_names = {f.name for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}
        for name in _field_names:
            val = getattr(obj, name, None)
            if val is not None:
                kwargs[name] = val
        return cls(**kwargs)

    def __post_init__(self):
        if self.kt_backend is None:
            self.kt_backend = os.environ.get("ACCELERATE_KT_BACKEND", "AMXBF16")
        if self.kt_num_threads is None:
            self.kt_num_threads = _env_int("ACCELERATE_KT_NUM_THREADS", 1)
        if self.kt_tp_enabled is None:
            self.kt_tp_enabled = _env_bool("ACCELERATE_KT_TP_ENABLED", False)
        if self.kt_threadpool_count is None:
            self.kt_threadpool_count = _env_int("ACCELERATE_KT_THREADPOOL_COUNT", 1)
        if self.kt_weight_path is None:
            self.kt_weight_path = os.environ.get("ACCELERATE_KT_WEIGHT_PATH", None)
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
        if self.kt_model_max_length is None:
            self.kt_model_max_length = _env_int("ACCELERATE_KT_MODEL_MAX_LENGTH", None)
        if self.kt_skip_expert_loading is None:
            if "ACCELERATE_KT_SKIP_EXPERT_LOADING" in os.environ:
                self.kt_skip_expert_loading = _env_bool("ACCELERATE_KT_SKIP_EXPERT_LOADING", True)
