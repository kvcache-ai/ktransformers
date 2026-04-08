# Model wrapping entry points for SFT
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util as _u
import logging
import os
from typing import Any

import torch
import torch.nn as nn

from .arch import (
    KTAMXConfigError,
    KTAMXNotAvailableError,
    MOEArchConfig,
    _get_layers_prefix,
    _get_model_container_and_layers,
    get_moe_arch_config,
    get_moe_module,
)
from .layer import KTMoELayerWrapper
from .lora import LoRAExperts
from .weights import (
    _clear_original_expert_weights,
    extract_moe_weights,
    load_experts_from_checkpoint_files,
)

logger = logging.getLogger(__name__)

KT_KERNEL_AVAILABLE = _u.find_spec("kt_kernel") is not None

if KT_KERNEL_AVAILABLE:
    try:
        from kt_kernel.experts import KTMoEWrapper
    except Exception:
        KTMoEWrapper = None
        KT_KERNEL_AVAILABLE = False
else:
    KTMoEWrapper = None


# =============================================================================
# Device-map builders
# =============================================================================


def _get_kt_config(kt_plugin: Any):
    """Extract KTConfig from a KTransformersPlugin or compatible object.

    Handles three cases:
    1. KTransformersPlugin with .kt_config (new style) → return kt_config
    2. Object with old field names (kt_num_threads etc.) → convert to KTConfig
    3. KTConfig directly → return as-is
    """
    from .config import KTConfig

    # New-style KTransformersPlugin
    kt_config = getattr(kt_plugin, "kt_config", None)
    if kt_config is not None and isinstance(kt_config, KTConfig):
        return kt_config

    # Already a KTConfig
    if isinstance(kt_plugin, KTConfig):
        return kt_plugin

    # Old-style object (HfTrainerKTConfig, old KTransformersPlugin, dict-like) — convert
    # Map old field names (kt_xxx) to new field names (xxx)
    _OLD_TO_NEW = {
        "kt_backend": "backend", "kt_num_threads": "num_threads",
        "kt_tp_enabled": "tp_enabled", "kt_threadpool_count": "threadpool_count",
        "kt_weight_path": "weight_path", "kt_expert_checkpoint_path": "expert_checkpoint_path",
        "kt_num_gpu_experts": "num_gpu_experts", "kt_max_cache_depth": "max_cache_depth",
        "kt_use_lora_experts": "use_lora_experts", "kt_lora_expert_num": "lora_expert_num",
        "kt_lora_expert_intermediate_size": "lora_expert_intermediate_size",
        "kt_skip_expert_loading": "skip_expert_loading",
        "kt_share_backward_bb": "share_backward_bb",
        "kt_checkpoint_files": "checkpoint_files",
        "kt_sharded_metadata": "sharded_metadata",
    }
    kwargs = {}
    for old_name, new_name in _OLD_TO_NEW.items():
        val = getattr(kt_plugin, old_name, None)
        if val is not None:
            kwargs[new_name] = val
    # Fields that don't have kt_ prefix
    for name in ("lora_rank", "lora_alpha", "model_max_length", "wrap_fn", "wrap_kwargs"):
        val = getattr(kt_plugin, name, None)
        if val is not None:
            kwargs[name] = val
    return KTConfig(**kwargs)


def build_kt_device_map(config, kt_plugin, device: str = "cuda:0") -> dict[str, str | int]:
    """
    Build device_map for KT model loading with hybrid GPU/CPU expert placement.
    """
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_experts = moe_config.expert_num
    cfg = _get_kt_config(kt_plugin)
    num_gpu_experts = getattr(cfg, "num_gpu_experts", 0) or 0

    device_map: dict[str, str | int] = {}

    device_map["model.embed_tokens"] = device
    device_map["model.norm"] = device
    device_map["lm_head"] = device

    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"
        device_map[layer_prefix] = device
        moe_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}"

        for expert_idx in range(num_experts):
            expert_key = f"{moe_prefix}.{moe_config.experts_attr}.{expert_idx}"
            if expert_idx < num_gpu_experts:
                device_map[expert_key] = device
            else:
                device_map[expert_key] = "cpu"

    logger.info(
        f"Built KT device_map: {num_gpu_experts} GPU experts, {num_experts - num_gpu_experts} CPU experts"
    )

    return device_map


def build_kt_device_map_simplified(config, kt_plugin, device: str = "cuda:0") -> dict[str, str | int]:
    """
    Simplified device_map builder: map full layers to GPU, override routed experts to CPU.
    """
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    cfg = _get_kt_config(kt_plugin)
    num_gpu_experts = getattr(cfg, "num_gpu_experts", 0) or 0

    device_map: dict[str, str | int] = {}

    device_map["model.embed_tokens"] = device
    device_map["model.norm"] = device
    device_map["lm_head"] = device

    for layer_idx in range(num_layers):
        layer_prefix = f"{layers_prefix}.{layer_idx}"
        device_map[layer_prefix] = device

        experts_prefix = f"{layer_prefix}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}"

        if num_gpu_experts == 0:
            device_map[experts_prefix] = "cpu"
        else:
            return build_kt_device_map(config, kt_plugin, device=device)

    logger.info("Built simplified KT device_map: all layers on GPU, routed experts on CPU")
    return device_map


# =============================================================================
# MoE layer wrapping
# =============================================================================


def wrap_moe_layers_with_kt_wrapper(model: nn.Module, kt_plugin: Any) -> list[KTMoELayerWrapper]:
    """
    Replace model's MoE layers with KTMoEWrapper-based wrappers.

    Loads expert weights into the C++ KT kernel. No LoRA initialization ---
    LoRA is handled by PEFT and later adapted via kt_adapt_peft_lora().
    Only rank 0 initializes KT kernel and loads weights.
    """
    import torch.distributed as dist

    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError("kt_kernel not found. Please install kt_kernel to enable KT MoE support.")

    # Only rank 0 should initialize KT and load weights
    is_rank_0 = True
    if dist.is_initialized():
        is_rank_0 = dist.get_rank() == 0

    moe_config = get_moe_arch_config(model.config)
    hidden_size = model.config.hidden_size

    cfg = _get_kt_config(kt_plugin)

    # Read lora_rank/lora_alpha for C++ wrapper initialization (buffer allocation only)
    lora_rank = getattr(cfg, "lora_rank", 1) or 1
    lora_alpha = getattr(cfg, "lora_alpha", 1.0) or 1.0

    # Read LoRA Experts configuration
    _raw_le = getattr(cfg, "use_lora_experts", None)
    use_lora_experts = bool(_raw_le) if _raw_le is not None else False
    lora_expert_num = getattr(cfg, "lora_expert_num", 2) or 2
    lora_expert_intermediate_size = getattr(cfg, "lora_expert_intermediate_size", 1024) or 1024

    if is_rank_0:
        logger.info(
            f"LoRA Experts config: use_lora_experts={use_lora_experts}, "
            f"num={lora_expert_num}, intermediate_size={lora_expert_intermediate_size}"
        )

    wrappers: list[KTMoELayerWrapper] = []
    moe_layer_count = 0

    kt_backend_map = {
        "AMXBF16": "AMXBF16_SFT",
        "AMXINT8": "AMXINT8_SFT",
        "AMXINT4": "AMXINT4_SFT",
        "AMXBF16_SkipLoRA": "AMXBF16_SFT_SkipLoRA",
        "AMXINT8_SkipLoRA": "AMXINT8_SFT_SkipLoRA",
        "AMXINT4_SkipLoRA": "AMXINT4_SFT_SkipLoRA",
    }
    # Build case-insensitive lookup to handle common typos like "SkipLora" vs "SkipLoRA"
    _kt_backend_map_lower = {k.lower(): v for k, v in kt_backend_map.items()}
    kt_backend = getattr(cfg, "backend", "AMXBF16")
    kt_method = kt_backend_map.get(kt_backend) or _kt_backend_map_lower.get(kt_backend.lower(), "AMXBF16_SFT")
    if kt_method != kt_backend_map.get(kt_backend):
        logger.warning(
            f"kt_backend '{kt_backend}' matched via case-insensitive lookup -> '{kt_method}'. "
            f"Please use the exact name from: {list(kt_backend_map.keys())}"
        )

    if "SkipLoRA" in kt_method:
        logger.info(f"Using SkipLoRA backend: {kt_method} (MoE LoRA gradients will be skipped)")

    threadpool_count = getattr(cfg, "threadpool_count", 1) if getattr(cfg, "tp_enabled", False) else 1

    kt_weight_path = getattr(cfg, "weight_path", None)
    use_kt_weight_path = kt_weight_path is not None
    if use_kt_weight_path:
        logger.info(f"Loading INT8 weights from kt_weight_path: {kt_weight_path}")

    checkpoint_files = getattr(cfg, "checkpoint_files", None)
    sharded_metadata = getattr(cfg, "sharded_metadata", None)

    # When kt_expert_checkpoint_path is set, always resolve from it (overrides any existing
    # checkpoint_files which may come from AttnOnlyBf16 and lack expert weights).
    kt_expert_checkpoint_path = getattr(cfg, "expert_checkpoint_path", None)
    if kt_expert_checkpoint_path:
        logger.info(f"Resolving expert checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")
        resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=kt_expert_checkpoint_path)
        if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
            checkpoint_files = resolved_files
            sharded_metadata = resolved_meta
            cfg.checkpoint_files = checkpoint_files
            cfg.sharded_metadata = sharded_metadata
            logger.info(f"Resolved {len(checkpoint_files)} checkpoint files from kt_expert_checkpoint_path")
        else:
            logger.warning(f"Failed to resolve checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")

    use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path

    logger.debug(
        f"Weight source: kt_weight_path={kt_weight_path!r}, "
        f"kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}, "
        f"checkpoint_files count={len(checkpoint_files) if checkpoint_files else 0}, "
        f"use_kt_weight_path={use_kt_weight_path}, use_checkpoint_files={use_checkpoint_files}"
    )

    if use_checkpoint_files:
        logger.info("Loading expert weights from checkpoint files (online conversion).")
    elif use_kt_weight_path and bool(checkpoint_files):
        logger.info("BF16 checkpoint files available for backward gradient computation.")
    elif (not use_kt_weight_path) and bool(getattr(cfg, "skip_expert_loading", False)):
        # If HF expert weights were skipped during `from_pretrained`, we must source expert weights externally.
        model_name_or_path = getattr(getattr(model, "config", None), "name_or_path", None)
        if model_name_or_path:
            resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=model_name_or_path)
            if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
                checkpoint_files = resolved_files
                sharded_metadata = resolved_meta
                cfg.checkpoint_files = checkpoint_files
                cfg.sharded_metadata = sharded_metadata
                use_checkpoint_files = True
                logger.info("KT skip_expert_loading enabled; using checkpoint files for online expert loading.")

        if not use_checkpoint_files:
            raise KTAMXConfigError(
                "KT skip_expert_loading is enabled but no `kt_weight_path` was provided and no safetensors checkpoint "
                "files could be resolved for on-the-fly expert loading."
            )

    import torch.distributed as _dist
    _rank = _dist.get_rank() if _dist.is_initialized() else 0

    model_container, layers = _get_model_container_and_layers(model, purpose="wrapping")
    logger.info(f"Total layers={len(layers)}, is_rank_0={is_rank_0}")

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        logger.debug(f"Wrapping MoE layer {layer_idx} (method={kt_method})")

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        wrapper = None

        if is_rank_0:
            # Get block_size from quantization_config if available (for FP8 dequant)
            _quant_cfg = getattr(model.config, "quantization_config", None)
            _block_size = None
            if _quant_cfg is not None:
                _block_size = getattr(_quant_cfg, "weight_block_size", None)

            if use_kt_weight_path:
                logger.debug(f"Layer {layer_idx}: forward + backward from kt_weight_path (.kt files)")
            elif use_checkpoint_files:
                layers_prefix = _get_layers_prefix(model.config)
                gate_proj, up_proj, down_proj = load_experts_from_checkpoint_files(
                    checkpoint_files=checkpoint_files,
                    sharded_metadata=sharded_metadata,
                    layers_prefix=layers_prefix,
                    moe_config=moe_config,
                    layer_idx=layer_idx,
                    block_size=_block_size,
                )
            else:
                gate_proj, up_proj, down_proj = extract_moe_weights(moe_module, moe_config)
                gate_proj = gate_proj.cpu().to(torch.bfloat16).contiguous()
                up_proj = up_proj.cpu().to(torch.bfloat16).contiguous()
                down_proj = down_proj.cpu().to(torch.bfloat16).contiguous()

        chunked_prefill_size = getattr(cfg, "model_max_length", None)
        if chunked_prefill_size is None:
            chunked_prefill_size = getattr(model.config, "max_position_embeddings", 4096)

        # Only rank 0 creates KTMoEWrapper and loads weights
        if is_rank_0:
            wrapper = KTMoEWrapper(
                layer_idx=layer_idx,
                num_experts=moe_config.expert_num,
                num_experts_per_tok=moe_config.num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=moe_config.intermediate_size,
                num_gpu_experts=0,
                cpuinfer_threads=getattr(cfg, "num_threads", 1),
                threadpool_count=threadpool_count,
                weight_path=kt_weight_path or "",
                chunked_prefill_size=chunked_prefill_size,
                method=kt_method,
                mode="sft",
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_cache_depth=getattr(cfg, "max_cache_depth", 2),
            )

            # Set share_backward_bb BEFORE load_weights (config is built during load)
            share_backward_bb = getattr(cfg, "share_backward_bb", None)
            if share_backward_bb is None:
                share_backward_bb = os.environ.get("ACCELERATE_KT_SHARE_BACKWARD_BB", "").lower() in ("true", "1", "yes")
            wrapper.share_backward_bb = share_backward_bb

            physical_to_logical_map = torch.arange(moe_config.expert_num, dtype=torch.int64, device="cpu")

            if use_kt_weight_path:
                logger.debug(f"Layer {layer_idx}: calling wrapper.load_weights() (C++ direct .kt load)")
                wrapper.load_weights(physical_to_logical_map)
            else:
                logger.debug(
                    f"Layer {layer_idx}: calling wrapper.load_weights_from_tensors() "
                    f"(BF16 tensor path, gate_proj shape={gate_proj.shape if gate_proj is not None else None})"
                )
                wrapper.load_weights_from_tensors(
                    gate_proj=gate_proj,
                    up_proj=up_proj,
                    down_proj=down_proj,
                    physical_to_logical_map_cpu=physical_to_logical_map,
                )

            wrapper.gate_proj = None
            wrapper.up_proj = None
            wrapper.down_proj = None

        # Create LoRA Experts if enabled
        lora_experts = None
        if use_lora_experts:
            lora_experts = LoRAExperts(
                num_experts=lora_expert_num,
                hidden_size=hidden_size,
                intermediate_size=lora_expert_intermediate_size,
                device="cuda",
                dtype=torch.bfloat16,
            )

        layer_wrapper = KTMoELayerWrapper(
            original_moe=moe_module,
            wrapper=wrapper,
            lora_params=None,
            moe_config=moe_config,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            lora_experts=lora_experts,
        )
        layer_wrapper._skip_lora = "SkipLoRA" in kt_method

        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)
        # Base weights have been copied into the C++ kernel's internal BufferB format.
        # Do not hold a Python-side reference --- it wastes ~1 GB/layer.
        del gate_proj, up_proj, down_proj

        wrappers.append(layer_wrapper)
        moe_layer_count += 1

        # Replace original expert weights with meta placeholders.
        # Experts remain in the model tree (via wrapper.experts) so PEFT can discover them.
        # Rank 0 already copied weights to C++ kernel via load_weights_from_tensors.
        _clear_original_expert_weights(moe_module, moe_config)

    logger.info(f"Wrapped {moe_layer_count} MoE layers with KTMoEWrapper")

    # Link wrappers for async backward repack (higher layer triggers repack for lower)
    for i in range(1, len(wrappers)):
        if wrappers[i].wrapper is not None and wrappers[i - 1].wrapper is not None:
            wrappers[i].wrapper._next_backward_wrapper = wrappers[i - 1].wrapper
    if wrappers and wrappers[0].wrapper is not None:
        wrappers[0].wrapper._next_backward_wrapper = None

    gc.collect()
    return wrappers


# =============================================================================
# Plugin builder
# =============================================================================


def _build_kt_plugin_from_args(model_args: Any, finetuning_args: Any | None = None):
    """
    Build a KTransformersPlugin from model_args and optional finetuning_args.

    Imported here to avoid circular dependency --- callers that need the plugin
    class should import it from the appropriate dataclasses module.
    """
    from .config import KTConfig
    from accelerate.utils.dataclasses import KTransformersPlugin

    kt_config = KTConfig(
        backend=getattr(model_args, "kt_backend", None),
        num_threads=getattr(model_args, "kt_num_threads", None),
        tp_enabled=getattr(model_args, "kt_tp_enabled", None),
        threadpool_count=getattr(model_args, "kt_threadpool_count", None),
        max_cache_depth=getattr(model_args, "kt_max_cache_depth", None),
        num_gpu_experts=getattr(model_args, "kt_num_gpu_experts", None),
        weight_path=getattr(model_args, "kt_weight_path", None),
        expert_checkpoint_path=getattr(model_args, "kt_expert_checkpoint_path", None),
        use_lora_experts=getattr(model_args, "kt_use_lora_experts", None),
        lora_expert_num=getattr(model_args, "kt_lora_expert_num", None),
        lora_expert_intermediate_size=getattr(model_args, "kt_lora_expert_intermediate_size", None),
        lora_rank=getattr(finetuning_args, "lora_rank", None) if finetuning_args else None,
        lora_alpha=getattr(finetuning_args, "lora_alpha", None) if finetuning_args else None,
        model_max_length=getattr(model_args, "model_max_length", None),
    )
    return KTransformersPlugin(enabled=True, kt_config=kt_config)


def get_kt_loading_kwargs(
    config,
    kt_plugin,
    torch_dtype: torch.dtype | str | None = torch.bfloat16,
    trust_remote_code: bool | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Get kwargs for AutoModel.from_pretrained() for KT loading."""
    kwargs: dict[str, Any] = {
        "config": config,
        "torch_dtype": torch_dtype,
        "device_map": "cpu",
        "low_cpu_mem_usage": True,
    }
    if trust_remote_code is not None:
        kwargs["trust_remote_code"] = trust_remote_code
    if token is not None:
        kwargs["token"] = token
    return kwargs


def _resolve_checkpoint_files(
    model_name_or_path: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    trust_remote_code: bool | None = None,
) -> tuple[list[str] | None, dict | None]:
    """Resolve HF checkpoint files. Depends on transformers internals."""
    try:
        from transformers.modeling_utils import _get_resolved_checkpoint_files
    except Exception:
        return None, None
    try:
        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=model_name_or_path,
            subfolder="",
            variant=None,
            gguf_file=None,
            from_tf=False,
            from_flax=False,
            use_safetensors=None,
            cache_dir=cache_dir,
            force_download=False,
            proxies=None,
            local_files_only=False,
            token=token,
            user_agent={"file_type": "model", "framework": "pytorch"},
            revision=revision or "main",
            commit_hash=None,
            is_remote_code=bool(trust_remote_code),
            transformers_explicit_filename=None,
        )
    except Exception:
        return None, None
    return checkpoint_files, sharded_metadata


def load_kt_model(
    config,
    model_args: Any | None = None,
    finetuning_args: Any | None = None,
    kt_plugin=None,
    model_name_or_path: str | None = None,
    trust_remote_code: bool | None = None,
    token: str | None = None,
    torch_dtype: torch.dtype | str | None = torch.bfloat16,
    **kwargs,
) -> nn.Module:
    """Load model with KTMoEWrapper backend."""
    from .arch import get_moe_arch_config, move_non_experts_to_gpu, get_expert_device, KTAMXNotAvailableError, KTAMXConfigError

    if kt_plugin is None:
        if model_args is None:
            raise KTAMXConfigError("Either kt_plugin or model_args must be provided to load_kt_model().")
        kt_plugin = _build_kt_plugin_from_args(model_args, finetuning_args)

    if model_name_or_path is None and model_args is not None:
        model_name_or_path = getattr(model_args, "model_name_or_path", None)
    if model_name_or_path is None:
        raise KTAMXConfigError("model_name_or_path is required to load_kt_model().")

    if trust_remote_code is None and model_args is not None:
        trust_remote_code = getattr(model_args, "trust_remote_code", None)
    if token is None and model_args is not None:
        token = getattr(model_args, "hf_hub_token", None)
    cache_dir = getattr(model_args, "cache_dir", None) if model_args is not None else None
    revision = getattr(model_args, "revision", None) if model_args is not None else None

    _ = get_moe_arch_config(config)

    logger.info("Loading model with KTMoEWrapper backend")

    from transformers import AutoModelForCausalLM
    from transformers.integrations.kt import set_kt_config, unset_kt_config

    loading_kwargs = get_kt_loading_kwargs(
        config, kt_plugin, torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code, token=token,
    )
    if model_args is not None:
        for key in ("cache_dir", "revision"):
            value = getattr(model_args, key, None)
            if value is not None:
                loading_kwargs[key] = value
    loading_kwargs.update(kwargs)

    cfg = _get_kt_config(kt_plugin)

    if getattr(cfg, "skip_expert_loading", None) is None:
        checkpoint_files, sharded_metadata = _resolve_checkpoint_files(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir, revision=revision,
            token=token, trust_remote_code=trust_remote_code,
        )
        if checkpoint_files and all(f.endswith(".safetensors") for f in checkpoint_files):
            if getattr(cfg, "weight_path", None) is None:
                cfg.skip_expert_loading = True
            else:
                cfg.skip_expert_loading = False
            cfg.checkpoint_files = checkpoint_files
            cfg.sharded_metadata = sharded_metadata
        else:
            cfg.skip_expert_loading = False

    set_kt_config(kt_plugin)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **loading_kwargs)
    finally:
        unset_kt_config()

    moe_config = get_moe_arch_config(config)
    move_non_experts_to_gpu(model, moe_config, device="cuda:0")

    existing_wrappers = getattr(model, "_kt_wrappers", None)
    if existing_wrappers:
        logger.info(f"MoE layers already wrapped ({len(existing_wrappers)} layers), skipping re-wrap")
        wrappers = existing_wrappers
    else:
        wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_plugin)

    model._kt_wrappers = wrappers
    model._kt_tp_enabled = bool(getattr(cfg, "tp_enabled", False))
    model._kt_use_lora_experts = bool(getattr(cfg, "use_lora_experts", False))

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model
