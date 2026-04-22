"""KT integration for HuggingFace transformers.

Provides ``HfTrainerKTConfig`` (analogous to ``HfTrainerDeepSpeedConfig``)
and the expert-weight-skipping hooks needed during model loading.
"""

import os
import re
import weakref
from typing import Any

# ── global weakref (mirrors transformers.integrations.kt pattern) ──────────

_kt_config_weak_ref: weakref.ReferenceType | None = None


# ── HfTrainerKTConfig ─────────────────────────────────────────────────────

class HfTrainerKTConfig:
    """Lightweight KT config wrapper (like ``HfTrainerDeepSpeedConfig``).

    A weakref is stored in module globals so model-loading code
    (``from_pretrained``) can decide whether to skip MoE expert weights
    before a ``Trainer`` / ``Accelerator`` exists.

    In practice ``TrainingArguments`` stores a reference via ``self.hf_kt_config``.
    """

    _ENV_MAPPING: dict[str, tuple[str, type]] = {
        "kt_backend": ("ACCELERATE_KT_BACKEND", str),
        "kt_num_gpu_experts": ("ACCELERATE_KT_NUM_GPU_EXPERTS", int),
        "kt_num_threads": ("ACCELERATE_KT_NUM_THREADS", int),
        "kt_tp_enabled": ("ACCELERATE_KT_TP_ENABLED", bool),
        "kt_threadpool_count": ("ACCELERATE_KT_THREADPOOL_COUNT", int),
        "kt_max_cache_depth": ("ACCELERATE_KT_MAX_CACHE_DEPTH", int),
        "kt_weight_path": ("ACCELERATE_KT_WEIGHT_PATH", str),
        "kt_use_lora_experts": ("ACCELERATE_KT_USE_LORA_EXPERTS", bool),
        "kt_lora_expert_num": ("ACCELERATE_KT_LORA_EXPERT_NUM", int),
        "kt_lora_expert_intermediate_size": ("ACCELERATE_KT_LORA_EXPERT_INTERMEDIATE_SIZE", int),
        "kt_lora_rank": ("ACCELERATE_KT_LORA_RANK", int),
        "kt_lora_alpha": ("ACCELERATE_KT_LORA_ALPHA", float),
        "kt_model_max_length": ("ACCELERATE_KT_MODEL_MAX_LENGTH", int),
        "kt_skip_expert_loading": ("ACCELERATE_KT_SKIP_EXPERT_LOADING", bool),
        "kt_share_backward_bb": ("ACCELERATE_KT_SHARE_BACKWARD_BB", bool),
    }

    def __init__(self, kt_config_dict: Any | None):
        self._kt_config = kt_config_dict if kt_config_dict is not None else {}

        if isinstance(self._kt_config, dict):
            for key, (env_key, typ) in self._ENV_MAPPING.items():
                if key in self._kt_config:
                    continue
                env_val = os.environ.get(env_key)
                if env_val is None or env_val == "":
                    continue
                if typ is bool:
                    self._kt_config[key] = env_val.lower() in ("1", "true", "yes")
                elif typ is int:
                    self._kt_config[key] = int(env_val)
                elif typ is float:
                    self._kt_config[key] = float(env_val)
                else:
                    self._kt_config[key] = env_val

        set_kt_config(self)

    def _get(self, key: str, default: Any = None) -> Any:
        cfg = self._kt_config
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        cfg = self.__dict__.get("_kt_config", {})
        if isinstance(cfg, dict) and name in cfg:
            return cfg[name]
        if hasattr(cfg, name):
            return getattr(cfg, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def trainer_config_process(self, args: Any) -> None:
        if getattr(args, "gradient_checkpointing", False):
            self._kt_config.setdefault("kt_share_cache_pool", True)

    @property
    def enabled(self) -> bool:
        enabled = self._get("enabled", None)
        return True if enabled is None else bool(enabled)

    @property
    def kt_weight_path(self) -> str | None:
        return self._get("kt_weight_path", None)

    @property
    def kt_skip_expert_loading(self) -> bool | None:
        explicit = self._get("kt_skip_expert_loading", None)
        if explicit is not None:
            return bool(explicit)
        return True if self.enabled else False


# ── global accessors ──────────────────────────────────────────────────────

def set_kt_config(kt_config: Any) -> None:
    global _kt_config_weak_ref
    _kt_config_weak_ref = weakref.ref(kt_config)


def unset_kt_config() -> None:
    global _kt_config_weak_ref
    _kt_config_weak_ref = None


def _get_kt_config() -> Any | None:
    if _kt_config_weak_ref is None:
        return None
    return _kt_config_weak_ref()


def is_kt_expert_loading_enabled() -> bool:
    kt_config: Any | None = _get_kt_config()
    if kt_config is not None:
        enabled = getattr(kt_config, "enabled", None)
        if enabled is False:
            return False
        skip_loading = getattr(kt_config, "kt_skip_expert_loading", None)
        if skip_loading is not None:
            return bool(skip_loading)
        return True

    env_enabled = os.environ.get("ACCELERATE_USE_KT", "").lower() in ("1", "true", "yes")
    if not env_enabled:
        return False
    env_skip = os.environ.get("ACCELERATE_KT_SKIP_EXPERT_LOADING", None)
    if env_skip is not None:
        return env_skip.lower() in ("1", "true", "yes")
    return True


# ── expert key regex ──────────────────────────────────────────────────────

KT_EXPERT_REGEX = re.compile(r"\.experts\.\d+\.")


def _is_expert_key(name: str) -> bool:
    return bool(KT_EXPERT_REGEX.search(name))
