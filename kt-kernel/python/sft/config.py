# KT-Kernel SFT configuration
# SPDX-License-Identifier: Apache-2.0

"""
KTConfig: kt-kernel's own configuration dataclass.

This is the kt-kernel equivalent of DeepSpeed's JSON config —
it holds all kt-kernel-specific settings and is passed through
KTransformersPlugin.kt_config (similar to DeepSpeedPlugin.hf_ds_config).
"""

from __future__ import annotations

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

    All kt-kernel-specific settings live here. Accelerate's KTransformersPlugin
    holds a reference to this via its `kt_config` field (similar to
    DeepSpeedPlugin.hf_ds_config).

    Can be created from:
    - Direct construction: KTConfig(backend="AMXBF16", weight_path="/path/...")
    - Dict: KTConfig(**config_dict)
    - Environment variables: KTConfig() reads ACCELERATE_KT_* env vars as defaults
    """

    # Backend selection
    backend: str | None = None
    num_threads: int | None = None
    tp_enabled: bool | None = None
    threadpool_count: int | None = None

    # Weight loading
    weight_path: str | None = None
    expert_checkpoint_path: str | None = None
    num_gpu_experts: int | None = None
    skip_expert_loading: bool | None = None
    share_backward_bb: bool | None = None

    # Cache
    max_cache_depth: int | None = None
    model_max_length: int | None = None

    # LoRA
    lora_rank: int | None = None
    lora_alpha: float | None = None

    # LoRA Experts (GPU-side extra experts)
    use_lora_experts: bool | None = None
    lora_expert_num: int | None = None
    lora_expert_intermediate_size: int | None = None

    # Runtime state (set during wrapping, not by user)
    checkpoint_files: list[str] | None = None
    sharded_metadata: dict | None = None

    # Custom wrapping
    wrap_fn: Callable[..., Any] | None = None
    wrap_kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.backend is None:
            self.backend = os.environ.get("ACCELERATE_KT_BACKEND", "AMXBF16")
        if self.num_threads is None:
            self.num_threads = _env_int("ACCELERATE_KT_NUM_THREADS", 1)
        if self.tp_enabled is None:
            self.tp_enabled = _env_bool("ACCELERATE_KT_TP_ENABLED", False)
        if self.threadpool_count is None:
            self.threadpool_count = _env_int("ACCELERATE_KT_THREADPOOL_COUNT", 1)
        if self.weight_path is None:
            self.weight_path = os.environ.get("ACCELERATE_KT_WEIGHT_PATH", None)
        if self.expert_checkpoint_path is None:
            self.expert_checkpoint_path = os.environ.get("ACCELERATE_KT_EXPERT_CHECKPOINT_PATH", None)
        if self.num_gpu_experts is None:
            self.num_gpu_experts = _env_int("ACCELERATE_KT_NUM_GPU_EXPERTS", 0)
        if self.max_cache_depth is None:
            self.max_cache_depth = _env_int("ACCELERATE_KT_MAX_CACHE_DEPTH", 2)
        if self.share_backward_bb is None:
            self.share_backward_bb = _env_bool("ACCELERATE_KT_SHARE_BACKWARD_BB", False)
        if self.use_lora_experts is None:
            self.use_lora_experts = _env_bool("ACCELERATE_KT_USE_LORA_EXPERTS", False)
        if self.lora_expert_num is None:
            self.lora_expert_num = _env_int("ACCELERATE_KT_LORA_EXPERT_NUM", None)
        if self.lora_expert_intermediate_size is None:
            self.lora_expert_intermediate_size = _env_int("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", None)
        if self.lora_rank is None:
            self.lora_rank = _env_int("ACCELERATE_KT_LORA_RANK", None)
        if self.lora_alpha is None:
            self.lora_alpha = _env_float("ACCELERATE_KT_LORA_ALPHA", None)
        if self.lora_alpha is None and self.lora_rank is not None:
            self.lora_alpha = float(self.lora_rank * 2)
        if self.model_max_length is None:
            self.model_max_length = _env_int("ACCELERATE_KT_MODEL_MAX_LENGTH", None)
        if self.skip_expert_loading is None:
            if "ACCELERATE_KT_SKIP_EXPERT_LOADING" in os.environ:
                self.skip_expert_loading = _env_bool("ACCELERATE_KT_SKIP_EXPERT_LOADING", True)
