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
    extract_kgroup_moe_weights,
    extract_moe_weights,
    has_kgroup_experts_in_kt_weight_path,
    load_kgroup_experts_from_kt_weight_path,
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


def _is_kgroup_sft_method(method: str) -> bool:
    return "KGroup" in method


def _log_kgroup_backward_capability(layer_idx: int | None, threadpool_count: int) -> None:
    label = "KGroup SFT" if layer_idx is None else f"Layer {layer_idx}: KGroup SFT"
    if threadpool_count == 1:
        logger.info(
            "%s TP=1 packed-dequant backward uses packed weights directly.",
            label,
        )
    else:
        logger.info(
            "%s TP=%s packed-dequant backward uses per-TP packed shards.",
            label,
            threadpool_count,
        )


def _sync_after_kt_wrap(cfg: Any) -> None:
    """Synchronize ranks after KT wrapping/loading so later FSDP collectives stay aligned."""
    if not getattr(cfg, "kt_sync_after_wrap", True):
        return
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            logger.info("Synchronizing ranks after KT MoE wrapping.")
            dist.barrier()
    except Exception as exc:
        logger.warning("KT post-wrap synchronization failed: %s", exc)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


# =============================================================================
# Device-map builders
# =============================================================================


def _get_kt_config(kt_plugin: Any):
    """Extract KTConfig from a KTransformersPlugin or compatible object.

    KTConfig field names use kt_ prefix, matching the dict keys in
    HfTrainerKTConfig exactly — no name-mapping needed.
    """
    from .config import KTConfig

    if isinstance(kt_plugin, KTConfig):
        return kt_plugin

    kt_config = getattr(kt_plugin, "kt_config", None)
    if kt_config is not None and isinstance(kt_config, KTConfig):
        return kt_config

    return KTConfig.from_object(kt_plugin)


def build_kt_device_map(config, kt_plugin, device: str = "cuda:0") -> dict[str, str | int]:
    """
    Build device_map for KT model loading with hybrid GPU/CPU expert placement.
    """
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    num_layers = config.num_hidden_layers
    num_experts = moe_config.expert_num
    cfg = _get_kt_config(kt_plugin)
    num_gpu_experts = getattr(cfg, "kt_num_gpu_experts", 0) or 0

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
    num_gpu_experts = getattr(cfg, "kt_num_gpu_experts", 0) or 0

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
    _text_cfg = getattr(model.config, "text_config", model.config)
    hidden_size = _text_cfg.hidden_size

    cfg = _get_kt_config(kt_plugin)
    if getattr(cfg, "kt_text_only_sft", False) and is_rank_0:
        logger.info("KT text-only SFT mode enabled; multimodal components are expected to stay unused.")

    # Read lora_rank/lora_alpha for C++ wrapper initialization (buffer allocation only)
    lora_rank = getattr(cfg, "kt_lora_rank", 1) or 1
    lora_alpha = getattr(cfg, "kt_lora_alpha", 1.0) or 1.0

    # Read LoRA Experts configuration
    _raw_le = getattr(cfg, "kt_use_lora_experts", None)
    use_lora_experts = bool(_raw_le) if _raw_le is not None else False
    lora_expert_num = getattr(cfg, "kt_lora_expert_num", 2) or 2
    lora_expert_intermediate_size = getattr(cfg, "kt_lora_expert_intermediate_size", 1024) or 1024

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
        "AMXINT4_KGroup": "AMXINT4_KGroup_SFT",
        "AMXBF16_SkipLoRA": "AMXBF16_SFT_SkipLoRA",
        "AMXINT8_SkipLoRA": "AMXINT8_SFT_SkipLoRA",
        "AMXINT4_SkipLoRA": "AMXINT4_SFT_SkipLoRA",
        "AMXINT4_KGroup_SkipLoRA": "AMXINT4_KGroup_SFT_SkipLoRA",
    }
    # Build case-insensitive lookup to handle common typos like "SkipLora" vs "SkipLoRA"
    _kt_backend_map_lower = {k.lower(): v for k, v in kt_backend_map.items()}
    kt_backend = getattr(cfg, "kt_backend", "AMXBF16")
    kt_method = kt_backend_map.get(kt_backend) or _kt_backend_map_lower.get(kt_backend.lower(), "AMXBF16_SFT")
    if kt_method != kt_backend_map.get(kt_backend):
        logger.warning(
            f"kt_backend '{kt_backend}' matched via case-insensitive lookup -> '{kt_method}'. "
            f"Please use the exact name from: {list(kt_backend_map.keys())}"
        )

    skip_expert_lora_adaptation = bool(getattr(cfg, "kt_skip_expert_lora_adaptation", False)) or _env_bool(
        "ACCELERATE_KT_SKIP_EXPERT_LORA_ADAPTATION"
    )
    force_fused_expert_lora = bool(getattr(cfg, "kt_force_fused_expert_lora", False))
    if force_fused_expert_lora and skip_expert_lora_adaptation:
        raise KTAMXConfigError(
            "kt_force_fused_expert_lora is incompatible with kt_skip_expert_lora_adaptation. "
            "Forced fused expert LoRA requires KT expert LoRA adaptation to stay enabled."
        )
    if force_fused_expert_lora and "SkipLoRA" in kt_method:
        raise KTAMXConfigError(
            "kt_force_fused_expert_lora is incompatible with SkipLoRA backends. "
            "Use a non-SkipLoRA kt_backend when training fused expert LoRA."
        )
    if skip_expert_lora_adaptation and "SkipLoRA" not in kt_method:
        skip_method = f"{kt_method}_SkipLoRA"
        if skip_method not in kt_backend_map.values():
            raise RuntimeError(f"KT skip expert LoRA adaptation is not supported for backend method {kt_method}.")
        logger.info(
            "KT expert LoRA adaptation is disabled; using SkipLoRA backend %s instead of %s.",
            skip_method,
            kt_method,
        )
        kt_method = skip_method

    if "SkipLoRA" in kt_method:
        logger.info(f"Using SkipLoRA backend: {kt_method} (MoE LoRA gradients will be skipped)")

    is_kgroup_method = _is_kgroup_sft_method(kt_method)
    group_size = getattr(cfg, "kt_group_size", None)
    if group_size is None:
        group_size = 32 if is_kgroup_method else 128
    zero_point = getattr(cfg, "kt_zero_point", None)
    if zero_point is None:
        zero_point = False if is_kgroup_method else True
    if is_kgroup_method and getattr(cfg, "kt_share_backward_bb", False):
        logger.warning("KGroup SFT uses packed weights directly for TP=1 backward; disabling kt_share_backward_bb.")
        cfg.kt_share_backward_bb = False

    threadpool_count = getattr(cfg, "kt_threadpool_count", 1) if getattr(cfg, "kt_tp_enabled", False) else 1
    if is_kgroup_method and threadpool_count > 1:
        if moe_config.intermediate_size % threadpool_count != 0:
            raise RuntimeError(
                "KGroup SFT TP requires intermediate_size divisible by kt_threadpool_count. "
                f"Got intermediate_size={moe_config.intermediate_size}, kt_threadpool_count={threadpool_count}."
            )
        local_intermediate_size = moe_config.intermediate_size // threadpool_count
        if local_intermediate_size % group_size != 0:
            raise RuntimeError(
                "KGroup SFT TP requires intermediate_size/kt_threadpool_count divisible by kt_group_size. "
                f"Got local_intermediate_size={local_intermediate_size}, kt_group_size={group_size}."
            )
    if is_kgroup_method:
        _log_kgroup_backward_capability(None, threadpool_count)

    kt_weight_path = getattr(cfg, "kt_weight_path", None)
    use_kt_weight_path = kt_weight_path is not None
    if use_kt_weight_path:
        weight_kind = "KGroup packed" if is_kgroup_method else "INT8"
        logger.info(f"Loading {weight_kind} weights from kt_weight_path: {kt_weight_path}")

    checkpoint_files = getattr(cfg, "kt_checkpoint_files", None)
    sharded_metadata = getattr(cfg, "kt_sharded_metadata", None)

    # Non-KGroup backends may resolve kt_expert_checkpoint_path for BF16 backward weights.
    # KGroup TP=1 training uses packed weights directly and intentionally skips BF16 shadow loading.
    kt_expert_checkpoint_path = getattr(cfg, "kt_expert_checkpoint_path", None)
    if kt_expert_checkpoint_path and not is_kgroup_method:
        logger.info(f"Resolving expert checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")
        resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=kt_expert_checkpoint_path)
        if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
            checkpoint_files = resolved_files
            sharded_metadata = resolved_meta
            cfg.kt_checkpoint_files = checkpoint_files
            cfg.kt_sharded_metadata = sharded_metadata
            logger.info(f"Resolved {len(checkpoint_files)} checkpoint files from kt_expert_checkpoint_path")
        else:
            logger.warning(f"Failed to resolve checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")
    elif kt_expert_checkpoint_path and is_kgroup_method:
        logger.info(
            "Ignoring kt_expert_checkpoint_path for KGroup SFT; TP=1 packed backward reads KGroup weights directly."
        )

    use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path

    logger.debug(
        f"Weight source: kt_weight_path={kt_weight_path!r}, "
        f"kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}, "
        f"checkpoint_files count={len(checkpoint_files) if checkpoint_files else 0}, "
        f"use_kt_weight_path={use_kt_weight_path}, use_checkpoint_files={use_checkpoint_files}"
    )

    if use_checkpoint_files and not is_kgroup_method:
        logger.info("Loading expert weights from checkpoint files (online conversion).")
    elif use_checkpoint_files and is_kgroup_method:
        logger.info("KGroup SFT uses packed tensors directly; BF16 checkpoint files will not be loaded for backward.")
    elif use_kt_weight_path and bool(checkpoint_files) and not is_kgroup_method:
        logger.info("BF16 checkpoint files available for backward gradient computation.")
    elif use_kt_weight_path and bool(checkpoint_files) and is_kgroup_method:
        logger.info("KGroup SFT packed backward uses packed weights directly; BF16 checkpoint files are ignored.")
    elif (not use_kt_weight_path) and bool(getattr(cfg, "kt_skip_expert_loading", False)):
        # If HF expert weights were skipped during `from_pretrained`, we must source expert weights externally.
        model_name_or_path = getattr(getattr(model, "config", None), "name_or_path", None)
        if model_name_or_path:
            resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=model_name_or_path)
            if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
                checkpoint_files = resolved_files
                sharded_metadata = resolved_meta
                cfg.kt_checkpoint_files = checkpoint_files
                cfg.kt_sharded_metadata = sharded_metadata
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

    from .arch import detect_fused_experts as _detect_fused

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        _layer_experts = getattr(moe_module, moe_config.experts_attr, None)
        _layer_is_fused = _detect_fused(_layer_experts)

        _use_fused_expert_lora = _layer_is_fused or force_fused_expert_lora

        logger.debug(
            f"Wrapping MoE layer {layer_idx} "
            f"(method={kt_method}, fused={_layer_is_fused}, force_fused_lora={force_fused_expert_lora})"
        )

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        kgroup_weights = None
        bwd_gate_proj, bwd_up_proj, bwd_down_proj = None, None, None
        wrapper = None

        if is_rank_0:
            # Get block_size from quantization_config if available (for FP8 dequant)
            _quant_cfg = getattr(model.config, "quantization_config", None)
            _block_size = None
            if _quant_cfg is not None:
                _block_size = getattr(_quant_cfg, "weight_block_size", None)

            if is_kgroup_method and use_kt_weight_path and has_kgroup_experts_in_kt_weight_path(
                kt_weight_path,
                _get_layers_prefix(model.config),
                moe_config,
                layer_idx,
            ):
                logger.debug(f"Layer {layer_idx}: loading KGroup compressed tensors from kt_weight_path")
                kgroup_weights = load_kgroup_experts_from_kt_weight_path(
                    kt_weight_path=kt_weight_path,
                    layers_prefix=_get_layers_prefix(model.config),
                    moe_config=moe_config,
                    layer_idx=layer_idx,
                    hidden_size=hidden_size,
                    group_size=group_size,
                )
                logger.info("Layer %s: loaded KGroup compressed tensors from kt_weight_path.", layer_idx)
                _log_kgroup_backward_capability(layer_idx, threadpool_count)
            elif is_kgroup_method and not use_kt_weight_path:
                kgroup_weights = extract_kgroup_moe_weights(
                    moe_module=moe_module,
                    moe_config=moe_config,
                    hidden_size=hidden_size,
                    group_size=group_size,
                )
                if any(
                    shadow is not None
                    for shadow in (
                        kgroup_weights.gate_bwd_shadow,
                        kgroup_weights.up_bwd_shadow,
                        kgroup_weights.down_bwd_shadow,
                    )
                ):
                    logger.debug(
                        "Layer %s: ignoring KGroup BF16 shadow tensors; packed backward is the training path.",
                        layer_idx,
                    )
            elif use_kt_weight_path:
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

        chunked_prefill_size = getattr(cfg, "kt_model_max_length", None)
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
                gpu_experts_mask=None,
                num_gpu_experts=0,
                cpuinfer_threads=getattr(cfg, "kt_num_threads", 1),
                threadpool_count=threadpool_count,
                weight_path=kt_weight_path or "",
                chunked_prefill_size=chunked_prefill_size,
                method=kt_method,
                mode="sft",
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_cache_depth=getattr(cfg, "kt_max_cache_depth", 2),
                group_size=group_size,
                zero_point=zero_point,
            )

            # Set share_backward_bb and share_cache_pool BEFORE load_weights (config is built during load)
            wrapper.share_backward_bb = cfg.kt_share_backward_bb
            wrapper.share_cache_pool = cfg.kt_share_cache_pool

            if bwd_gate_proj is not None and "KGroup" not in kt_method:
                wrapper.set_backward_shadow_weights(
                    gate_proj=bwd_gate_proj,
                    up_proj=bwd_up_proj,
                    down_proj=bwd_down_proj,
                )

            physical_to_logical_map = torch.arange(moe_config.expert_num, dtype=torch.int64, device="cpu")

            if kgroup_weights is not None:
                logger.debug(
                    f"Layer {layer_idx}: calling wrapper.load_kgroup_weights_from_tensors() "
                    f"(packed KGroup tensor path, gate_proj numel={kgroup_weights.gate_proj.numel()})"
                )
                wrapper.load_kgroup_weights_from_tensors(
                    gate_proj=kgroup_weights.gate_proj,
                    gate_scale=kgroup_weights.gate_scale,
                    up_proj=kgroup_weights.up_proj,
                    up_scale=kgroup_weights.up_scale,
                    down_proj=kgroup_weights.down_proj,
                    down_scale=kgroup_weights.down_scale,
                    physical_to_logical_map_cpu=physical_to_logical_map,
                )
            elif use_kt_weight_path:
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
        layer_wrapper._fused_experts = _use_fused_expert_lora
        layer_wrapper._lora_rank = lora_rank
        layer_wrapper._skip_expert_lora_adaptation = skip_expert_lora_adaptation or "SkipLoRA" in kt_method

        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)
        # Base weights have been copied into the C++ kernel's internal BufferB format.
        # Do not hold a Python-side reference --- it wastes ~1 GB/layer.
        del gate_proj, up_proj, down_proj, kgroup_weights, bwd_gate_proj, bwd_up_proj, bwd_down_proj

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
    _sync_after_kt_wrap(cfg)
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
        kt_backend=getattr(model_args, "kt_backend", None),
        kt_num_threads=getattr(model_args, "kt_num_threads", None),
        kt_tp_enabled=getattr(model_args, "kt_tp_enabled", None),
        kt_threadpool_count=getattr(model_args, "kt_threadpool_count", None),
        kt_max_cache_depth=getattr(model_args, "kt_max_cache_depth", None),
        kt_num_gpu_experts=getattr(model_args, "kt_num_gpu_experts", None),
        kt_weight_path=getattr(model_args, "kt_weight_path", None),
        kt_expert_checkpoint_path=getattr(model_args, "kt_expert_checkpoint_path", None),
        kt_group_size=getattr(model_args, "kt_group_size", None),
        kt_zero_point=getattr(model_args, "kt_zero_point", None),
        kt_use_lora_experts=getattr(model_args, "kt_use_lora_experts", None),
        kt_lora_expert_num=getattr(model_args, "kt_lora_expert_num", None),
        kt_lora_expert_intermediate_size=getattr(model_args, "kt_lora_expert_intermediate_size", None),
        kt_lora_rank=getattr(finetuning_args, "lora_rank", None) if finetuning_args else None,
        kt_lora_alpha=getattr(finetuning_args, "lora_alpha", None) if finetuning_args else None,
        kt_model_max_length=getattr(model_args, "model_max_length", None),
        kt_sync_after_wrap=getattr(model_args, "kt_sync_after_wrap", None),
        kt_text_only_sft=getattr(model_args, "kt_text_only_sft", None),
        kt_skip_expert_lora_adaptation=getattr(model_args, "kt_skip_expert_lora_adaptation", None),
        kt_force_fused_expert_lora=getattr(model_args, "kt_force_fused_expert_lora", None),
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

    if getattr(cfg, "kt_skip_expert_loading", None) is None:
        checkpoint_files, sharded_metadata = _resolve_checkpoint_files(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir, revision=revision,
            token=token, trust_remote_code=trust_remote_code,
        )
        if checkpoint_files and all(f.endswith(".safetensors") for f in checkpoint_files):
            if getattr(cfg, "kt_weight_path", None) is None:
                cfg.kt_skip_expert_loading = True
            else:
                cfg.kt_skip_expert_loading = False
            cfg.kt_checkpoint_files = checkpoint_files
            cfg.kt_sharded_metadata = sharded_metadata
        else:
            cfg.kt_skip_expert_loading = False

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
    model._kt_tp_enabled = bool(getattr(cfg, "kt_tp_enabled", False))
    model._kt_use_lora_experts = bool(getattr(cfg, "kt_use_lora_experts", False))

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model
