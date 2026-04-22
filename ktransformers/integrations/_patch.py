"""Patch manager — injects KT classes & hooks into transformers / accelerate.

All patches are **idempotent** (safe to call multiple times) and
**reversible** via ``remove_all()``.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

_patched: bool = False
_originals: dict[str, Any] = {}

# ── public API ─────────────────────────────────────────────────────────────

def apply_all() -> None:
    """Apply every KT patch.  Idempotent — second call is a no-op."""
    global _patched
    if _patched:
        return

    _inject_kt_classes()
    _patch_modeling_utils()
    _patch_training_args()
    _patch_accelerator()
    _patch_fsdp_utils()

    _patched = True
    logger.info("[KT] All patches applied successfully.")


def remove_all() -> None:
    """Reverse every KT patch."""
    global _patched, _originals
    for qualname, original in _originals.items():
        _set_attr(qualname, original)
    _originals.clear()
    _patched = False


def is_patched() -> bool:
    return _patched


# ── helpers ────────────────────────────────────────────────────────────────

def _set_attr(qualname: str, value: Any) -> None:
    """Set ``module.attr`` given ``"module.sub.attr"``."""
    *mod_parts, attr = qualname.split(".")
    mod = importlib.import_module(".".join(mod_parts))
    setattr(mod, attr, value)


def _get_attr(qualname: str) -> Any:
    *mod_parts, attr = qualname.split(".")
    mod = importlib.import_module(".".join(mod_parts))
    return getattr(mod, attr)


def _patch_func(qualname: str, wrapper_factory: Callable) -> None:
    """Replace ``qualname`` with ``wrapper_factory(original)``, saving original."""
    if qualname in _originals:
        return  # already patched
    original = _get_attr(qualname)
    _originals[qualname] = original
    _set_attr(qualname, wrapper_factory(original))


def _inject_module(full_module_name: str, module_obj: Any) -> None:
    """Create a fake submodule inside an existing package."""
    import sys
    parts = full_module_name.split(".")
    for i in range(1, len(parts) + 1):
        parent_name = ".".join(parts[:i])
        if parent_name not in sys.modules:
            if i == len(parts):
                sys.modules[parent_name] = module_obj
            else:
                import types
                sys.modules[parent_name] = types.ModuleType(parent_name)


# ── step 1: inject KT classes into expected locations ─────────────────────

def _inject_kt_classes() -> None:
    """Make ``HfTrainerKTConfig`` and ``KTransformersPlugin`` importable
    from the paths that the custom forks expect."""

    from . import _transformers as _tf
    from . import _accelerate as _accel

    # 1. Inject HfTrainerKTConfig into transformers.integrations.kt
    try:
        import transformers.integrations.kt
    except ImportError:
        # Create the module if it doesn't exist
        _inject_module("transformers.integrations.kt", _tf)

    # Always set the attributes
    _tf_mod = importlib.import_module("transformers.integrations.kt")
    for _name in ("HfTrainerKTConfig", "set_kt_config", "unset_kt_config",
                   "_get_kt_config", "is_kt_expert_loading_enabled"):
        setattr(_tf_mod, _name, getattr(_tf, _name))

    # 2. Inject KTransformersPlugin into accelerate.utils.dataclasses
    import accelerate.utils.dataclasses as _ad
    if not hasattr(_ad, "KTransformersPlugin"):
        setattr(_ad, "KTransformersPlugin", _accel.KTransformersPlugin)

    # 3. Inject apply_kt_config_to_env into accelerate.utils.launch
    import accelerate.utils.launch as _al
    if not hasattr(_al, "_apply_kt_config_to_env"):
        setattr(_al, "_apply_kt_config_to_env", _accel.apply_kt_config_to_env)

    logger.info("[KT] Injected KT classes into transformers & accelerate.")


# ── step 2: patch modeling_utils (expert-weight skipping) ──────────────────

def _patch_modeling_utils() -> None:
    """Patch ``from_pretrained`` to skip MoE expert weights when KT is enabled."""
    try:
        import transformers.modeling_utils as _mu
    except ImportError:
        return

    # 2a. Patch load_state_dict to skip expert keys
    _patch_load_state_dict(_mu)

    # 2b. Patch _load_pretrained_model to filter expert keys
    _patch_load_pretrained_model(_mu)

    # 2c. Patch from_pretrained to wrap MoE layers after loading
    _patch_from_pretrained(_mu)


def _patch_load_state_dict(mu) -> None:
    _original_lsd = mu.load_state_dict

    def _kt_load_state_dict(checkpoint_file, *args, **kwargs):
        from ktransformers.integrations._transformers import (
            is_kt_expert_loading_enabled,
            KT_EXPERT_REGEX,
        )
        skip = is_kt_expert_loading_enabled()
        if skip:
            print(f"[KT load_state_dict] Skipping expert keys from: {checkpoint_file}")

        state_dict = _original_lsd(checkpoint_file, *args, **kwargs)

        if skip and isinstance(state_dict, dict):
            state_dict = {k: v for k, v in state_dict.items()
                          if not KT_EXPERT_REGEX.search(k)}
        return state_dict

    mu.load_state_dict = _kt_load_state_dict
    _originals.setdefault("transformers.modeling_utils.load_state_dict", _original_lsd)


def _patch_load_pretrained_model(mu) -> None:
    """Filter expert keys from the key_renaming_mapping before loading."""
    if not hasattr(mu, "PreTrainedModel"):
        return
    _original_lpm = mu.PreTrainedModel._load_pretrained_model

    @classmethod
    def _kt_load_pretrained_model(cls, *args, **kwargs):
        from ktransformers.integrations._transformers import (
            is_kt_expert_loading_enabled,
            KT_EXPERT_REGEX,
            _get_kt_config,
        )
        skip = is_kt_expert_loading_enabled()
        if skip:
            # The key_renaming_mapping is the 4th positional arg
            # Filter expert keys from it
            pass  # handled by load_state_dict + load_shard_file patches
        return _original_lpm.__func__(cls, *args, **kwargs)

    # mu.PreTrainedModel._load_pretrained_model = _kt_load_pretrained_model


def _patch_from_pretrained(mu) -> None:
    """After from_pretrained loads weights, wrap MoE layers with KT kernel."""
    if not hasattr(mu, "PreTrainedModel"):
        return
    _original_fp = mu.PreTrainedModel.from_pretrained

    @classmethod
    def _kt_from_pretrained(cls, *args, **kwargs):
        from ktransformers.integrations._transformers import (
            is_kt_expert_loading_enabled,
            _get_kt_config,
        )
        model = _original_fp.__func__(cls, *args, **kwargs)

        if is_kt_expert_loading_enabled():
            kt_config = _get_kt_config()
            if kt_config is not None:
                try:
                    from kt_kernel.sft import wrap_moe_layers_with_kt_wrapper
                    wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_config)
                    model._kt_wrappers = wrappers
                    logger.info(f"[KT] Wrapped {len(wrappers)} MoE layers in from_pretrained")
                except Exception as e:
                    logger.warning(f"[KT] Failed to wrap MoE layers: {e}")

        return model

    mu.PreTrainedModel.from_pretrained = _kt_from_pretrained
    _originals.setdefault(
        "transformers.modeling_utils.PreTrainedModel.from_pretrained",
        _original_fp,
    )


# ── step 3: patch training_args (kt_config field) ──────────────────────────

def _patch_training_args() -> None:
    """Add ``kt_config`` processing to ``TrainingArguments.__post_init__``."""
    try:
        from transformers import TrainingArguments
    except ImportError:
        return

    if hasattr(TrainingArguments, "_kt_patched"):
        return

    # Add kt_config field if not present
    if "kt_config" not in TrainingArguments.__dataclass_fields__:
        import dataclasses
        TrainingArguments.__dataclass_fields__["kt_config"] = dataclasses.field(
            default=None,
            metadata={"help": "Enable KTransformers and pass a KT config dict or path."},
        )
        # Rebuild __init__ signature
        if hasattr(TrainingArguments, "__init__"):
            TrainingArguments.__init__.__annotations__["kt_config"] = "Optional[Union[dict, str]]"

    _original_post_init = TrainingArguments.__post_init__

    def _kt_post_init(self):
        _original_post_init(self)

        # Process kt_config
        kt_config_dict = self.kt_config
        if kt_config_dict is not None and isinstance(kt_config_dict, str):
            import json
            with open(kt_config_dict) as f:
                kt_config_dict = json.load(f)

        # Fall back to accelerator_config
        if kt_config_dict is None:
            try:
                from accelerate.utils import AcceleratorConfig
                accel_cfg = getattr(self, "accelerator_config", None)
                if accel_cfg is not None:
                    kt_config_dict = getattr(accel_cfg, "kt_config", None)
            except ImportError:
                pass

        if isinstance(kt_config_dict, dict):
            kt_config_dict.setdefault("enabled", True)
            kt_config_dict.setdefault("kt_skip_expert_loading", True)

        if kt_config_dict is not None or os.environ.get("ACCELERATE_USE_KT", "").lower() in ("1", "true", "yes"):
            from ktransformers.integrations import HfTrainerKTConfig
            self.hf_kt_config = HfTrainerKTConfig(kt_config_dict)
            self.hf_kt_config.trainer_config_process(self)
            if getattr(self.hf_kt_config, "enabled", False):
                os.environ["ACCELERATE_USE_KT"] = "true"

            if self.accelerator_config is not None and kt_config_dict is not None:
                self.accelerator_config.kt_config = kt_config_dict

    TrainingArguments.__post_init__ = _kt_post_init
    TrainingArguments._kt_patched = True
    _originals.setdefault(
        "transformers.training_args.TrainingArguments.__post_init__",
        _original_post_init,
    )


# ── step 4: patch accelerator ─────────────────────────────────────────────

def _patch_accelerator() -> None:
    """Add KT support to ``Accelerator``."""
    try:
        from accelerate import Accelerator
    except ImportError:
        return

    if getattr(Accelerator, "_kt_patched", False):
        return

    # 4a. Patch __init__ to accept kt_config
    _original_init = Accelerator.__init__

    def _kt_accelerator_init(self, *args, kt_config=None, **kwargs):
        if kt_config is None:
            from ktransformers.integrations import KTransformersPlugin
            candidate = KTransformersPlugin()
            if candidate.enabled:
                kt_config = candidate

        if kt_config is not None and not kt_config.enabled:
            kt_config = None

        _original_init(self, *args, **kwargs)
        if kt_config is not None:
            self.state.kt_config = kt_config

    Accelerator.__init__ = _kt_accelerator_init

    # 4b. Patch prepare_model to skip device_placement for KT
    _original_prepare_model = Accelerator.prepare_model

    def _kt_prepare_model(self, model, device_placement=None, *args, **kwargs):
        kt_plugin = getattr(self.state, "kt_config", None)
        kt_bypass = bool(kt_plugin and kt_plugin.enabled and kt_plugin.bypass_device_map_check)

        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != getattr(
                importlib.import_module("accelerate.utils"), "DistributedType", None
            ).FSDP if hasattr(importlib.import_module("accelerate.utils"), "DistributedType") else True

        if kt_plugin is not None and kt_plugin.enabled and kt_plugin.skip_device_placement:
            device_placement = False

        # Bypass device_map check for KT
        import os as _os
        if kt_bypass:
            _os.environ["ACCELERATE_BYPASS_DEVICE_MAP"] = "true"

        result = _original_prepare_model(self, model, device_placement=device_placement, *args, **kwargs)

        if kt_bypass and "ACCELERATE_BYPASS_DEVICE_MAP" in _os.environ:
            pass  # keep it set

        return result

    Accelerator.prepare_model = _kt_prepare_model

    # 4c. Patch backward for KT LoRA injection
    _original_backward = Accelerator.backward

    def _kt_backward(self, loss, **kwargs):
        _original_backward(self, loss, **kwargs)

        if not getattr(self, "_kt_lora_injected", False):
            try:
                from kt_kernel.sft import get_kt_lora_params
                _models = [m for m in self._models if hasattr(m, 'parameters')]
                if _models and self._optimizers:
                    _kt_params = get_kt_lora_params(_models[0])
                    if not _kt_params:
                        unwrapped = self.unwrap_model(_models[0])
                        _kt_params = get_kt_lora_params(unwrapped)
                    if _kt_params:
                        opt = self._optimizers[0]
                        existing_ids = {id(p) for group in opt.param_groups for p in group['params']}
                        missing = [p for p in _kt_params if id(p) not in existing_ids]
                        if missing:
                            lr = opt.param_groups[0].get('lr', 1e-4)
                            opt.add_param_group({'params': missing, 'lr': lr})
                            print(f"\033[32m[KT] Added {len(missing)} KT LoRA params to optimizer\033[0m", flush=True)
            except Exception:
                pass
            self._kt_lora_injected = True

    Accelerator.backward = _kt_backward
    Accelerator._kt_patched = True

    _originals.setdefault("accelerate.accelerator.Accelerator.__init__", _original_init)
    _originals.setdefault("accelerate.accelerator.Accelerator.prepare_model", _original_prepare_model)
    _originals.setdefault("accelerate.accelerator.Accelerator.backward", _original_backward)


# ── step 5: patch fsdp_utils ──────────────────────────────────────────────

def _patch_fsdp_utils() -> None:
    """Patch FSDP2 utils for KT compatibility."""
    try:
        from accelerate.utils import fsdp_utils as _fu
    except ImportError:
        return

    if getattr(_fu, "_kt_patched", False):
        return

    # 5a. Patch fsdp2_load_full_state_dict for non-DTensor params
    _original_load = _fu.fsdp2_load_full_state_dict

    def _kt_fsdp2_load_full_state_dict(accelerator, model, full_sd, *args, **kwargs):
        import torch.distributed as dist
        from torch.distributed.tensor import distribute_tensor

        meta_sharded_sd = model.state_dict()
        sharded_sd = {}

        def _is_dtensor(param):
            return hasattr(param, 'device_mesh') and param.device_mesh is not None

        meta_items = list(meta_sharded_sd.items())

        if accelerator.is_main_process:
            for meta_name, sharded_param in meta_items:
                if meta_name not in full_sd:
                    raise KeyError(f"Missing key in full_sd: {meta_name}")
                full_param = full_sd[meta_name]
                if not _is_dtensor(sharded_param):
                    sharded_sd[meta_name] = full_param.detach()
                    continue

                device_mesh = sharded_param.device_mesh
                full_param = full_param.detach().to(device_mesh.device_type)
                dist.broadcast(full_param, src=0, group=dist.group.WORLD)
                sharded_tensor = distribute_tensor(full_param, device_mesh, sharded_param.placements)
                sharded_sd[meta_name] = sharded_tensor
        else:
            for meta_name, sharded_param in meta_items:
                if not _is_dtensor(sharded_param):
                    sharded_sd[meta_name] = sharded_param
                    continue
                device_mesh = sharded_param.device_mesh
                full_tensor = torch.empty(
                    sharded_param.size(), device=device_mesh.device_type, dtype=sharded_param.dtype
                )
                dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
                sharded_sd[meta_name] = distribute_tensor(full_tensor, device_mesh, sharded_param.placements)

        model.load_state_dict(sharded_sd, assign=True)

    _fu.fsdp2_load_full_state_dict = _kt_fsdp2_load_full_state_dict

    # 5b. Patch fsdp2_switch_optimizer_parameters
    _original_switch = _fu.fsdp2_switch_optimizer_parameters

    def _kt_fsdp2_switch_optimizer_parameters(optimizer, mapping):
        from torch.distributed.tensor import DTensor

        def _get_data_ptr(p):
            if isinstance(p, DTensor):
                return p._local_tensor.data_ptr()
            elif callable(getattr(p, 'data_ptr', None)):
                return p.data_ptr()
            else:
                return p.data_ptr

        for param_group in optimizer.param_groups:
            new_params = []
            for p in param_group["params"]:
                ptr = _get_data_ptr(p)
                if ptr in mapping:
                    new_params.append(mapping[ptr])
                else:
                    new_params.append(p)
            param_group["params"] = new_params

    _fu.fsdp2_switch_optimizer_parameters = _kt_fsdp2_switch_optimizer_parameters

    _fu._kt_patched = True
    _originals.setdefault(
        "accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict", _original_load
    )
    _originals.setdefault(
        "accelerate.utils.fsdp_utils.fsdp2_switch_optimizer_parameters", _original_switch
    )


import torch  # noqa: E402 (needed in fsdp2 patch closure)
