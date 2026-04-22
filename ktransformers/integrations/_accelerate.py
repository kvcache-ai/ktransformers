"""KT integration for HuggingFace accelerate.

Provides ``KTransformersPlugin`` (analogous to ``DeepSpeedPlugin``)
and the env-var bridge used by ``accelerate launch``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from accelerate.utils import DistributedType, parse_flag_from_env


# ── KTransformersPlugin ───────────────────────────────────────────────────

@dataclass
class KTransformersPlugin:
    """Plugin to enable KTransformers MoE wrapping inside Accelerate.

    Follows the DeepSpeed pattern: only accelerate-framework interaction
    fields are defined here. KT-kernel-specific configuration is passed
    through the opaque ``kt_config`` field.
    """

    enabled: bool | None = None
    kt_config: Any = None
    bypass_device_map_check: bool | None = None
    skip_device_placement: bool | None = None
    allowed_distributed_types: tuple[DistributedType, ...] = (
        DistributedType.NO,
        DistributedType.FSDP,
        DistributedType.MULTI_GPU,
    )
    require_single_process: bool = False

    def __post_init__(self) -> None:
        if self.enabled is None:
            self.enabled = parse_flag_from_env("ACCELERATE_USE_KT", default=False)

        if self.kt_config is None:
            try:
                from kt_kernel.sft import KTConfig
                self.kt_config = KTConfig()
            except ImportError:
                self.kt_config = None
        elif isinstance(self.kt_config, dict):
            try:
                from kt_kernel.sft import KTConfig
                self.kt_config = KTConfig(**self.kt_config)
            except ImportError:
                pass

        if self.kt_config is not None and self.enabled:
            if getattr(self.kt_config, "skip_expert_loading", None) is None:
                self.kt_config.skip_expert_loading = True

        if self.bypass_device_map_check is None:
            self.bypass_device_map_check = parse_flag_from_env(
                "ACCELERATE_KT_BYPASS_DEVICE_MAP", default=True
            )

        if self.skip_device_placement is None:
            self.skip_device_placement = parse_flag_from_env(
                "ACCELERATE_KT_SKIP_DEVICE_PLACEMENT", default=True
            )


# ── env-var bridge ────────────────────────────────────────────────────────

_KT_ENV_MAPPING: dict[str, str] = {
    "kt_backend": "ACCELERATE_KT_BACKEND",
    "kt_num_gpu_experts": "ACCELERATE_KT_NUM_GPU_EXPERTS",
    "kt_num_threads": "ACCELERATE_KT_NUM_THREADS",
    "kt_tp_enabled": "ACCELERATE_KT_TP_ENABLED",
    "kt_threadpool_count": "ACCELERATE_KT_THREADPOOL_COUNT",
    "kt_max_cache_depth": "ACCELERATE_KT_MAX_CACHE_DEPTH",
    "kt_weight_path": "ACCELERATE_KT_WEIGHT_PATH",
    "kt_use_lora_experts": "ACCELERATE_KT_USE_LORA_EXPERTS",
    "kt_lora_expert_num": "ACCELERATE_KT_LORA_EXPERT_NUM",
    "kt_lora_expert_intermediate_size": "ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE",
    "kt_lora_rank": "ACCELERATE_KT_LORA_RANK",
    "kt_lora_alpha": "ACCELERATE_KT_LORA_ALPHA",
    "kt_model_max_length": "ACCELERATE_KT_MODEL_MAX_LENGTH",
    "kt_skip_expert_loading": "ACCELERATE_KT_SKIP_EXPERT_LOADING",
    "kt_share_backward_bb": "ACCELERATE_KT_SHARE_BACKWARD_BB",
}


def apply_kt_config_to_env(kt_config: dict[str, Any] | None, current_env: dict[str, str] | None = None) -> dict[str, str]:
    """Mirror ``kt_config`` entries into env vars (FSDP/TP style).

    Called by ``accelerate launch`` so downstream code can detect KT
    before the user script starts.
    """
    if current_env is None:
        current_env = dict(os.environ)

    if not kt_config:
        return current_env

    enabled = kt_config.get("enabled", True)
    if "ACCELERATE_USE_KT" not in current_env and enabled is not None:
        current_env["ACCELERATE_USE_KT"] = str(bool(enabled)).lower()

    if not enabled:
        return current_env

    for key, env_key in _KT_ENV_MAPPING.items():
        if env_key in current_env or key not in kt_config:
            continue
        value = kt_config[key]
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        current_env[env_key] = str(value)

    return current_env
