"""KTransformers integrations for HuggingFace transformers and accelerate.

Provides:
- ``HfTrainerKTConfig`` — KT training configuration (like DeepSpeed config)
- ``KTransformersPlugin`` — accelerate plugin for KT
- ``apply_all()`` / ``remove_all()`` — patch management
"""

from ._transformers import HfTrainerKTConfig, set_kt_config, unset_kt_config, is_kt_expert_loading_enabled
from ._accelerate import KTransformersPlugin, apply_kt_config_to_env
from ._patch import apply_all, remove_all, is_patched

__all__ = [
    "HfTrainerKTConfig",
    "KTransformersPlugin",
    "set_kt_config",
    "unset_kt_config",
    "is_kt_expert_loading_enabled",
    "apply_kt_config_to_env",
    "apply_all",
    "remove_all",
    "is_patched",
]
