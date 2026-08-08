# Model wrapping entry points for SFT
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util as _u
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .arch import (
    KTAMXConfigError,
    KTAMXNotAvailableError,
    _get_layers_prefix,
    _get_model_container_and_layers,
    get_moe_arch_config,
    get_moe_module,
)
from .layer import KTMoELayerWrapper
from .lora import LoRAExperts
from .base import _supports_authoritative_optimizer_grads
from .backend import FP8_BACKEND, INT8_BACKEND, get_fp8_runtime, get_int8_runtime
from .checkpoint import load_full_weight_layer, resolve_full_weight_checkpoint
from .dist_utils import _distributed_rank_world_size
from .weights import (
    _clear_original_expert_weights,
    extract_moe_weights,
    load_block_fp8_experts_from_checkpoint_files,
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


def _supports_checkpoint_forward_reuse(full_weight_grad: bool, lora_rank: int) -> bool:
    return (full_weight_grad and lora_rank == 0) or (not full_weight_grad and lora_rank > 0)


def _native_fp8_block_size(model_config: Any) -> tuple[int, int]:
    """Read and strictly validate the checkpoint's native FP8 contract."""

    text_config = getattr(model_config, "text_config", None)
    quant_config = getattr(model_config, "quantization_config", None)
    if quant_config is None and text_config is not None:
        quant_config = getattr(text_config, "quantization_config", None)
    if quant_config is None:
        raise KTAMXConfigError(
            "native FP8 SFT requires model quantization_config metadata"
        )

    def read(name: str):
        if isinstance(quant_config, dict):
            return quant_config.get(name)
        return getattr(quant_config, name, None)

    quant_method = str(read("quant_method") or "").lower()
    if "fp8" not in quant_method:
        raise KTAMXConfigError(
            "native FP8 SFT requires an FP8 checkpoint, got "
            f"quant_method={quant_method!r}"
        )
    block_size = read("weight_block_size")
    if block_size is None:
        block_size = read("weight_block_shape")
    if block_size is None or tuple(block_size) != (128, 128):
        raise KTAMXConfigError(
            "native FP8 SFT requires weight_block_size=[128, 128], "
            f"got {block_size!r}"
        )
    return (128, 128)


def _resolve_native_fp8_checkpoint_files(
    model_name_or_path: str,
) -> tuple[list[str] | None, dict | None]:
    """Resolve raw safetensors, including a dependency-light local fallback."""

    resolved_files, resolved_metadata = _resolve_checkpoint_files(
        model_name_or_path=model_name_or_path
    )
    if resolved_files and all(str(path).endswith(".safetensors") for path in resolved_files):
        return [str(path) for path in resolved_files], resolved_metadata

    checkpoint_path = Path(model_name_or_path)
    if checkpoint_path.is_file() and checkpoint_path.suffix == ".safetensors":
        return [str(checkpoint_path)], None
    if not checkpoint_path.is_dir():
        return None, None

    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        weight_map = metadata.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise KTAMXConfigError(
                f"invalid safetensors index without weight_map: {index_path}"
            )
        files = sorted(
            {str(checkpoint_path / filename) for filename in weight_map.values()}
        )
        missing = [path for path in files if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(
                f"safetensors index references missing shard: {missing[0]}"
            )
        return files, metadata

    files = sorted(str(path) for path in checkpoint_path.glob("*.safetensors"))
    return (files, None) if files else (None, None)


def _sync_rank0_wrap_error(
    error: BaseException | None,
    *,
    context: str,
    rank: int,
    world_size: int,
) -> None:
    """Make a rank-0-only wrapping stage fail coherently on every rank."""
    if world_size <= 1:
        if error is not None:
            raise error
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        if error is not None:
            raise error
        return

    payload = [
        None
        if error is None
        else f"{type(error).__name__}: {error}"
    ]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        return
    if rank == 0 and error is not None:
        raise error
    raise RuntimeError(f"{context}: rank 0 failed: {payload[0]}")


# =============================================================================
# Device-map builders
# =============================================================================


def _get_kt_config(kt_plugin: Any):
    """Resolve a private KTConfig from a compatible public container."""
    from .config import KTConfig

    if isinstance(kt_plugin, KTConfig):
        return kt_plugin
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

    logger.info(f"Built KT device_map: {num_gpu_experts} GPU experts, {num_experts - num_gpu_experts} CPU experts")

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
    if not KT_KERNEL_AVAILABLE:
        raise KTAMXNotAvailableError("kt_kernel not found. Please install kt_kernel to enable KT MoE support.")

    # Only global rank 0 initializes KT. Launcher env fallback matters when
    # model construction happens before init_process_group().
    distributed_rank, distributed_world_size = _distributed_rank_world_size()
    is_rank_0 = distributed_rank == 0

    moe_config = get_moe_arch_config(model.config)
    _text_cfg = getattr(model.config, "text_config", model.config)
    hidden_size = _text_cfg.hidden_size

    cfg = _get_kt_config(kt_plugin)
    activation_policy = cfg.kt_activation_policy

    # Read lora_rank/lora_alpha for C++ wrapper initialization (buffer allocation only)
    # Use explicit None checks: lora_rank=0 is a valid value (full mode, no LoRA),
    # but `or` pattern would treat 0 as falsy and replace it with 1.
    _raw_rank = getattr(cfg, "kt_lora_rank", None)
    lora_rank = _raw_rank if _raw_rank is not None else 1
    _raw_alpha = getattr(cfg, "kt_lora_alpha", None)
    lora_alpha = _raw_alpha if _raw_alpha is not None else 1.0
    _raw_dropout = getattr(cfg, "kt_lora_dropout", None)
    lora_dropout = _raw_dropout if _raw_dropout is not None else 0.0

    # Read full_weight_grad mode
    _raw_fwg = getattr(cfg, "kt_full_weight_grad", None)
    full_weight_grad = _raw_fwg if _raw_fwg is not None else False
    train_mode = getattr(cfg, "kt_train_mode", "lora")

    # Full and hybrid are explicit modes.  LlamaFactory exposes a default
    # lora_rank even for full tuning, which must not silently turn Full into
    # Hybrid.  Preserve the legacy fallback for callers without train_mode.
    if train_mode == "full":
        lora_rank = 0
    elif full_weight_grad and train_mode != "hybrid" and lora_rank > 0:
        _has_explicit_lora_rank = getattr(cfg, "kt_lora_rank", None) is not None
        if not _has_explicit_lora_rank:
            lora_rank = 0

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
        if full_weight_grad:
            logger.info(f"Full weight gradient mode enabled (lora_rank={lora_rank})")

    wrappers: list[KTMoELayerWrapper] = []
    moe_layer_count = 0

    kt_backend_map = {
        "AMXBF16": "AMXBF16_SFT",
        FP8_BACKEND: "AMXFP8_SFT",
        "AMXFP8": "AMXFP8_SFT",
        INT8_BACKEND: "INT8_SFT",
        "AMXINT8": "INT8_SFT",
        "AMXINT4": "AMXINT4_SFT",
        "AMXBF16_SkipLoRA": "AMXBF16_SFT_SkipLoRA",
        "AMXINT8_SkipLoRA": "AMXINT8_SFT_SkipLoRA",
        "AMXINT4_SkipLoRA": "AMXINT4_SFT_SkipLoRA",
    }
    # Case-insensitive matching remains for compatibility, but an unknown
    # backend must never silently turn a requested quantized run into BF16.
    _kt_backend_map_lower = {k.lower(): v for k, v in kt_backend_map.items()}
    kt_backend = str(getattr(cfg, "kt_backend", "AMXBF16"))
    kt_method = kt_backend_map.get(kt_backend) or _kt_backend_map_lower.get(kt_backend.lower())
    if kt_method is None:
        raise KTAMXConfigError(
            f"Unknown kt_backend {kt_backend!r}; expected one of {sorted(kt_backend_map)}"
        )
    if kt_backend not in kt_backend_map:
        logger.warning(
            f"kt_backend '{kt_backend}' matched via case-insensitive lookup -> '{kt_method}'. "
            f"Please use the exact name from: {list(kt_backend_map.keys())}"
        )

    if "SkipLoRA" in kt_method:
        logger.info(f"Using SkipLoRA backend: {kt_method} (MoE LoRA gradients will be skipped)")
    force_fused_expert_lora = bool(
        getattr(cfg, "kt_force_fused_expert_lora", False)
    )
    if force_fused_expert_lora and "SkipLoRA" in kt_method:
        raise KTAMXConfigError(
            "kt_force_fused_expert_lora is incompatible with SkipLoRA backends"
        )
    requested_num_gpu_experts = int(getattr(cfg, "kt_num_gpu_experts", 0) or 0)
    expert_weight_format = getattr(cfg, "kt_expert_weight_format", None)
    if expert_weight_format == "int8":
        if kt_method != "INT8_SFT":
            raise KTAMXConfigError(
                "kt_expert_weight_format='int8' requires kt_backend='auto' or 'INT8'"
            )
        if full_weight_grad or train_mode != "lora" or lora_rank <= 0:
            raise KTAMXConfigError(
                "INT8 SFT supports frozen-base LoRA only; Full and Hybrid are not supported"
            )
        if requested_num_gpu_experts != 0 or use_lora_experts:
            raise KTAMXConfigError(
                "INT8 SFT requires all base experts and LoRA execution on CPU"
            )
        if not bool(getattr(cfg, "kt_share_backward_bb", False)):
            raise KTAMXConfigError("INT8 SFT requires kt_share_backward_bb=true")
    if expert_weight_format == "fp8":
        if kt_method != "AMXFP8_SFT":
            raise KTAMXConfigError(
                "kt_expert_weight_format='fp8' requires kt_backend='auto' or 'FP8'"
            )
        if full_weight_grad or train_mode != "lora" or lora_rank <= 0:
            raise KTAMXConfigError(
                "FP8 SFT supports frozen-base LoRA only; Full and Hybrid are not supported"
            )
        if requested_num_gpu_experts != 0 or use_lora_experts:
            raise KTAMXConfigError(
                "FP8 SFT requires all base experts and LoRA execution on CPU"
            )
        if not bool(getattr(cfg, "kt_share_backward_bb", False)):
            raise KTAMXConfigError("FP8 SFT requires kt_share_backward_bb=true")
        if getattr(cfg, "kt_weight_lifecycle", "persistent") != "persistent":
            raise KTAMXConfigError("FP8 SFT requires persistent checkpoint weights")
    cpu_activation_retain = activation_policy.cpu == "retain"
    reuse_checkpoint_forward = cpu_activation_retain and activation_policy.gpu == "recompute"
    if cpu_activation_retain and (
        not _supports_checkpoint_forward_reuse(full_weight_grad, lora_rank)
        or kt_method not in {"AMXBF16_SFT", "AMXFP8_SFT", "INT8_SFT"}
        or requested_num_gpu_experts != 0
        or use_lora_experts
    ):
        raise KTAMXConfigError(
            "activation_policy.cpu=retain requires CPU-only AMXBF16 Full/LoRA "
            "or frozen-base INT8/FP8 LoRA; Hybrid, GPU-expert, LoRA-expert, "
            "INT4, and SkipLoRA paths are not supported"
        )
    if is_rank_0:
        logger.warning(
            "KT activation policy: cpu=%s, gpu=%s, "
            "checkpoint_forward_reuse=%s, share_cache_pool=%s, world_size=%d",
            activation_policy.cpu,
            activation_policy.gpu,
            reuse_checkpoint_forward,
            not cpu_activation_retain,
            distributed_world_size,
        )
    uses_authoritative_optimizer_grads = _supports_authoritative_optimizer_grads(
        kt_method,
        requested_num_gpu_experts,
        full_weight_grad=full_weight_grad,
        lora_rank=lora_rank,
    )
    int8_runtime = None
    int8_runtime_error = None
    if expert_weight_format == "int8" and is_rank_0:
        try:
            int8_runtime = get_int8_runtime()
        except RuntimeError as exc:
            int8_runtime_error = KTAMXNotAvailableError(str(exc))
    if expert_weight_format == "int8":
        _sync_rank0_wrap_error(
            int8_runtime_error,
            context="selecting the INT8 SFT kernel",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
    if int8_runtime is not None:
        logger.info(
            "KT INT8 SFT dispatch: configured_backend=%s, logical_backend=%s, "
            "cpu_variant=%s, effective_kernel=%s, weight_layout=%s",
            kt_backend,
            INT8_BACKEND,
            int8_runtime.cpu_variant,
            int8_runtime.kernel,
            int8_runtime.weight_layout,
        )
    fp8_runtime = None
    fp8_runtime_error = None
    if expert_weight_format == "fp8" and is_rank_0:
        try:
            fp8_runtime = get_fp8_runtime()
        except RuntimeError as exc:
            fp8_runtime_error = KTAMXNotAvailableError(str(exc))
    if expert_weight_format == "fp8":
        _sync_rank0_wrap_error(
            fp8_runtime_error,
            context="selecting the native FP8 SFT kernel",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )
    if fp8_runtime is not None:
        logger.info(
            "KT FP8 SFT dispatch: configured_backend=%s, logical_backend=%s, "
            "cpu_variant=%s, effective_kernel=%s, weight_layout=%s",
            kt_backend,
            FP8_BACKEND,
            fp8_runtime.cpu_variant,
            fp8_runtime.kernel,
            fp8_runtime.weight_layout,
        )

    threadpool_count = getattr(cfg, "kt_threadpool_count", 1) if getattr(cfg, "kt_tp_enabled", False) else 1
    fp8_block_size = None
    if expert_weight_format == "fp8":
        fp8_block_size = _native_fp8_block_size(model.config)
        if hidden_size % 128 or moe_config.intermediate_size % 128:
            raise KTAMXConfigError(
                "FP8 SFT requires hidden and routed intermediate dimensions divisible by 128"
            )
        if (
            threadpool_count < 1
            or moe_config.intermediate_size % threadpool_count
            or (moe_config.intermediate_size // threadpool_count) % 128
        ):
            raise KTAMXConfigError(
                "FP8 SFT requires each TP intermediate slice divisible by 128; "
                f"intermediate_size={moe_config.intermediate_size}, "
                f"threadpool_count={threadpool_count}"
            )

    kt_weight_path = getattr(cfg, "kt_weight_path", None)
    # For FP8 the frontend forwards model_name_or_path through kt_weight_path.
    # It is checkpoint provenance, not a pre-packed .kt directory.
    use_kt_weight_path = kt_weight_path is not None and expert_weight_format != "fp8"
    if use_kt_weight_path:
        logger.info(
            "Loading %s weights from kt_weight_path: %s",
            expert_weight_format or "pre-quantized",
            kt_weight_path,
        )

    checkpoint_files = getattr(cfg, "kt_checkpoint_files", None)
    sharded_metadata = getattr(cfg, "kt_sharded_metadata", None)

    # When kt_expert_checkpoint_path is set, always resolve from it (overrides any existing
    # checkpoint_files which may come from AttnOnlyBf16 and lack expert weights).
    kt_expert_checkpoint_path = getattr(cfg, "kt_expert_checkpoint_path", None)
    full_weight_checkpoint = resolve_full_weight_checkpoint(kt_expert_checkpoint_path)
    use_full_weight_checkpoint = full_weight_checkpoint is not None
    if use_full_weight_checkpoint and use_kt_weight_path:
        raise KTAMXConfigError(
            "A KT Full checkpoint and kt_weight_path were both selected. "
            "Choose exactly one authoritative expert weight source."
        )
    if use_full_weight_checkpoint:
        logger.info(f"Loading expert weights from KT Full checkpoint: {full_weight_checkpoint}")
    elif kt_expert_checkpoint_path:
        logger.info(f"Resolving expert checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}")
        resolved_files, resolved_meta = _resolve_checkpoint_files(model_name_or_path=kt_expert_checkpoint_path)
        if resolved_files and all(f.endswith(".safetensors") for f in resolved_files):
            checkpoint_files = resolved_files
            sharded_metadata = resolved_meta
            cfg.kt_checkpoint_files = checkpoint_files
            cfg.kt_sharded_metadata = sharded_metadata
            logger.info(f"Resolved {len(checkpoint_files)} checkpoint files from kt_expert_checkpoint_path")
        else:
            logger.warning(
                f"Failed to resolve checkpoint files from kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}"
            )

    if expert_weight_format == "fp8" and not checkpoint_files and kt_weight_path:
        logger.info(
            "Resolving native FP8 checkpoint files from kt_weight_path=%r",
            kt_weight_path,
        )
        resolved_files, resolved_meta = _resolve_native_fp8_checkpoint_files(
            kt_weight_path
        )
        if resolved_files:
            checkpoint_files = resolved_files
            sharded_metadata = resolved_meta
            cfg.kt_checkpoint_files = checkpoint_files
            cfg.kt_sharded_metadata = sharded_metadata

    use_checkpoint_files = bool(checkpoint_files) and not use_kt_weight_path and not use_full_weight_checkpoint
    if expert_weight_format == "int8":
        if not use_kt_weight_path:
            raise KTAMXConfigError(
                "INT8 SFT requires kt_weight_path with pre-quantized .kt weights"
            )
        if use_full_weight_checkpoint or kt_expert_checkpoint_path:
            raise KTAMXConfigError(
                "INT8 SFT does not support Full checkpoints or online expert conversion"
            )
    if expert_weight_format == "fp8":
        if use_full_weight_checkpoint:
            raise KTAMXConfigError("FP8 SFT does not support KT Full checkpoints")

    logger.debug(
        f"Weight source: kt_weight_path={kt_weight_path!r}, "
        f"kt_expert_checkpoint_path={kt_expert_checkpoint_path!r}, "
        f"full_weight_checkpoint={full_weight_checkpoint!r}, "
        f"checkpoint_files count={len(checkpoint_files) if checkpoint_files else 0}, "
        f"use_kt_weight_path={use_kt_weight_path}, use_full_weight_checkpoint={use_full_weight_checkpoint}, "
        f"use_checkpoint_files={use_checkpoint_files}"
    )

    if use_full_weight_checkpoint:
        logger.info("Loading expert weights from a KT Full checkpoint.")
    elif use_checkpoint_files:
        logger.info("Loading expert weights from checkpoint files (online conversion).")
    elif use_kt_weight_path and bool(checkpoint_files):
        logger.info("BF16 checkpoint files available for backward gradient computation.")
    elif (
        not use_kt_weight_path
        and not use_full_weight_checkpoint
        and bool(getattr(cfg, "kt_skip_expert_loading", False))
    ):
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

    if expert_weight_format == "fp8" and not use_checkpoint_files:
        raise KTAMXConfigError(
            "FP8 SFT requires raw safetensors checkpoint files. Point "
            "kt_weight_path at model_name_or_path or set kt_expert_checkpoint_path."
        )

    model_container, layers = _get_model_container_and_layers(model, purpose="wrapping")
    logger.info(f"Total layers={len(layers)}, is_rank_0={is_rank_0}")

    from .arch import detect_fused_experts as _detect_fused

    expert_layer_indices = [
        layer_idx
        for layer_idx, layer in enumerate(layers)
        if get_moe_module(layer, moe_config) is not None
    ]
    ephemeral_store = None
    ephemeral_requested = (
        getattr(cfg, "kt_weight_lifecycle", "persistent") == "ephemeral"
    )
    ephemeral_open_error = None
    if is_rank_0 and ephemeral_requested:
        try:
            from .ephemeral import EphemeralKTWeightStore

            ephemeral_store = EphemeralKTWeightStore.open(
                kt_weight_path,
                layer_indices=expert_layer_indices,
                numa_count=threadpool_count,
                expert_num=moe_config.expert_num,
                hidden_size=hidden_size,
                intermediate_size=moe_config.intermediate_size,
            )
        except BaseException as exc:
            ephemeral_open_error = exc
    if ephemeral_requested:
        _sync_rank0_wrap_error(
            ephemeral_open_error,
            context="opening ephemeral INT8 weights",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

    persistent_manifest_error = None
    if (
        is_rank_0
        and expert_weight_format == "int8"
        and not ephemeral_requested
    ):
        try:
            from .weight_manifest import validate_persistent_int8_weights

            validated_manifest = validate_persistent_int8_weights(
                kt_weight_path,
                layer_indices=expert_layer_indices,
                numa_count=threadpool_count,
                expert_num=moe_config.expert_num,
                hidden_size=hidden_size,
                intermediate_size=moe_config.intermediate_size,
            )
            logger.info(
                "Validated persistent INT8 weights: manifest=%s, schema=%d%s, "
                "layout=%s, layers=%d, files=%d, bytes=%d",
                validated_manifest.path,
                validated_manifest.schema_version,
                " (legacy compatibility)" if validated_manifest.is_legacy else "",
                validated_manifest.layout,
                len(validated_manifest.layer_indices),
                validated_manifest.file_count,
                validated_manifest.size_bytes,
            )
        except BaseException as exc:
            persistent_manifest_error = exc
    if expert_weight_format == "int8" and not ephemeral_requested:
        _sync_rank0_wrap_error(
            persistent_manifest_error,
            context="validating persistent INT8 weights",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

    for layer_idx, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue

        _layer_experts = getattr(moe_module, moe_config.experts_attr, None)
        _layer_is_fused = _detect_fused(_layer_experts)
        if (
            expert_weight_format in {"int8", "fp8"}
            and not _layer_is_fused
            and not force_fused_expert_lora
        ):
            raise KTAMXConfigError(
                f"{expert_weight_format.upper()} LoRA with non-fused runtime experts requires "
                "kt_force_fused_expert_lora=true"
            )
        _use_fused_expert_lora = _layer_is_fused or force_fused_expert_lora

        logger.debug(
            "Wrapping MoE layer %s (method=%s, fused=%s, force_fused_lora=%s)",
            layer_idx,
            kt_method,
            _layer_is_fused,
            force_fused_expert_lora,
        )

        # Only rank 0 loads weights and initializes KT kernel
        gate_proj, up_proj, down_proj = None, None, None
        block_fp8_weights = None
        wrapper = None
        weight_source_error = None

        if is_rank_0:
            try:
                # Get block_size from quantization_config if available (for legacy FP8 dequant)
                _quant_cfg = getattr(model.config, "quantization_config", None)
                _block_size = None
                if _quant_cfg is not None:
                    _block_size = (
                        _quant_cfg.get("weight_block_size")
                        if isinstance(_quant_cfg, dict)
                        else getattr(_quant_cfg, "weight_block_size", None)
                    )

                if use_full_weight_checkpoint:
                    expected_shapes = {
                        "gate_proj": (
                            int(moe_config.expert_num),
                            int(moe_config.intermediate_size),
                            int(hidden_size),
                        ),
                        "up_proj": (
                            int(moe_config.expert_num),
                            int(moe_config.intermediate_size),
                            int(hidden_size),
                        ),
                        "down_proj": (
                            int(moe_config.expert_num),
                            int(hidden_size),
                            int(moe_config.intermediate_size),
                        ),
                    }
                    gate_proj, up_proj, down_proj = load_full_weight_layer(
                        full_weight_checkpoint,
                        layer_idx=layer_idx,
                        expected_shapes=expected_shapes,
                    )
                elif use_kt_weight_path:
                    logger.debug(f"Layer {layer_idx}: forward + backward from kt_weight_path (.kt files)")
                elif expert_weight_format == "fp8":
                    layers_prefix = _get_layers_prefix(model.config)
                    block_fp8_weights = load_block_fp8_experts_from_checkpoint_files(
                        checkpoint_files=checkpoint_files,
                        sharded_metadata=sharded_metadata,
                        layers_prefix=layers_prefix,
                        moe_config=moe_config,
                        layer_idx=layer_idx,
                        hidden_size=hidden_size,
                        block_size=fp8_block_size,
                    )
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
            except BaseException as exc:
                weight_source_error = exc

        _sync_rank0_wrap_error(
            weight_source_error,
            context=f"resolving expert weights for layer {layer_idx}",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

        chunked_prefill_size = getattr(cfg, "kt_model_max_length", None)
        if chunked_prefill_size is None:
            chunked_prefill_size = getattr(model.config, "max_position_embeddings", 4096)
        # Rank 0 receives the concatenation of every rank's local rows.  Model
        # configs are homogeneous across ranks, so the sum of local maxima is
        # the per-rank capacity multiplied by world size.
        rank0_chunked_prefill_size = int(chunked_prefill_size) * distributed_world_size

        # Only rank 0 creates KTMoEWrapper and loads weights
        construct_error = None
        if is_rank_0:
            try:
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
                    chunked_prefill_size=rank0_chunked_prefill_size,
                    method=kt_method,
                    mode="sft",
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    max_cache_depth=getattr(cfg, "kt_max_cache_depth", 2),
                    full_weight_grad=full_weight_grad,
                )
            except BaseException as exc:
                try:
                    if ephemeral_store is not None:
                        ephemeral_store.cleanup()
                except BaseException:
                    logger.exception(
                        "Cleanup failed after constructing ephemeral INT8 layer %s",
                        layer_idx,
                    )
                construct_error = exc

        _sync_rank0_wrap_error(
            construct_error,
            context=f"constructing KT layer {layer_idx}",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

        load_error = None
        if is_rank_0:
            try:
                # The current SFT wrapping path routes all experts through KT even
                # when the loading config requested GPU experts. Preserve that
                # configuration's legacy gradient lifecycle until the hybrid
                # routed-expert path supports authoritative buffers end to end.
                wrapper._uses_authoritative_optimizer_grads = uses_authoritative_optimizer_grads

                # These flags are consumed while the C++ config is built.
                wrapper.share_backward_bb = cfg.kt_share_backward_bb
                wrapper.reuse_checkpoint_forward = reuse_checkpoint_forward
                wrapper.activation_policy = activation_policy
                wrapper.share_cache_pool = not cpu_activation_retain

                physical_to_logical_map = torch.arange(
                    moe_config.expert_num,
                    dtype=torch.int64,
                    device="cpu",
                )

                if expert_weight_format == "fp8":
                    logger.debug(
                        "Layer %s: packing raw per-expert FP8 checkpoint tensors",
                        layer_idx,
                    )
                    wrapper.load_block_fp8_weights(
                        block_fp8_weights,
                        physical_to_logical_map,
                    )
                    block_fp8_weights = None
                elif use_kt_weight_path:
                    logger.debug(
                        f"Layer {layer_idx}: calling wrapper.load_weights() "
                        "(C++ direct .kt load)"
                    )
                    wrapper.load_weights(physical_to_logical_map)
                    if ephemeral_store is not None:
                        # load_weights() returns only after C++ copied every file
                        # into its owned BufferB storage.
                        ephemeral_store.consume_layer(layer_idx)
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

                if full_weight_grad:
                    wrapper.init_full_weight_grad_buffers(
                        gate_proj=wrapper.gate_proj if wrapper.gate_proj is not None else gate_proj,
                        up_proj=wrapper.up_proj if wrapper.up_proj is not None else up_proj,
                        down_proj=wrapper.down_proj if wrapper.down_proj is not None else down_proj,
                    )
                else:
                    wrapper.gate_proj = None
                    wrapper.up_proj = None
                    wrapper.down_proj = None
            except BaseException as exc:
                if ephemeral_store is not None:
                    try:
                        ephemeral_store.cleanup()
                    except BaseException:
                        logger.exception(
                            "Cleanup failed after loading ephemeral INT8 layer %s",
                            layer_idx,
                        )
                load_error = exc

        _sync_rank0_wrap_error(
            load_error,
            context=f"loading KT layer {layer_idx}",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

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
            full_weight_grad=full_weight_grad,
            uses_authoritative_optimizer_grads=uses_authoritative_optimizer_grads,
            activation_policy=activation_policy,
        )
        layer_wrapper._fused_experts = _layer_is_fused
        layer_wrapper._use_fused_expert_lora = _use_fused_expert_lora
        layer_wrapper._force_fused_expert_lora = force_fused_expert_lora
        layer_wrapper._lora_rank = lora_rank
        layer_wrapper._lora_alpha = float(lora_alpha)
        layer_wrapper._kt_owner_rank = 0
        layer_wrapper._kt_world_size_at_wrap = distributed_world_size
        layer_wrapper._kt_expert_weight_format = expert_weight_format or "bf16"

        setattr(layer, moe_config.moe_layer_attr, layer_wrapper)
        # Base weights have been copied into the C++ kernel's internal BufferB format.
        # In full_weight_grad mode, the authoritative copies are gate_proj_buf etc.
        # Always release local references to save ~1 GB/layer.
        del gate_proj, up_proj, down_proj, block_fp8_weights

        wrappers.append(layer_wrapper)
        moe_layer_count += 1

        # Replace original expert weights with zero-storage placeholders.
        # Experts remain in the model tree (via wrapper.experts) so PEFT can discover them.
        # Rank 0 already copied weights to C++ kernel via load_weights_from_tensors.
        # gate_proj_buf serves as the authoritative copy in full_weight_grad mode.
        _clear_original_expert_weights(
            moe_module,
            moe_config,
            full_weight_grad=full_weight_grad,
            empty_placeholders=expert_weight_format in {"int8", "fp8"},
        )

    ephemeral_finish_error = None
    if ephemeral_store is not None:
        try:
            ephemeral_store.finish()
        except BaseException as exc:
            try:
                ephemeral_store.cleanup()
            except BaseException:
                logger.exception("Cleanup failed while finishing ephemeral INT8 weights")
            ephemeral_finish_error = exc
    if ephemeral_requested:
        _sync_rank0_wrap_error(
            ephemeral_finish_error,
            context="finishing ephemeral INT8 weights",
            rank=distributed_rank,
            world_size=distributed_world_size,
        )

    logger.info(f"Wrapped {moe_layer_count} MoE layers with KTMoEWrapper")
    model._kt_expert_weight_format = expert_weight_format or "bf16"

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

    # Map LlamaFactory finetuning_type to kt_train_mode
    finetuning_type = getattr(finetuning_args, "finetuning_type", None) if finetuning_args else None
    kt_train_mode_map = {
        "full": "full",
        "freeze": "hybrid",
        "lora": "lora",
        "galore": "full",
        "badam": "full",
    }
    kt_train_mode = kt_train_mode_map.get(finetuning_type, None) if finetuning_type else None

    configured_lora_rank = getattr(finetuning_args, "lora_rank", None) if finetuning_args else None
    configured_lora_alpha = getattr(finetuning_args, "lora_alpha", None) if finetuning_args else None
    configured_lora_dropout = getattr(finetuning_args, "lora_dropout", None) if finetuning_args else None
    if kt_train_mode == "full":
        configured_lora_rank = None
        configured_lora_alpha = None
        configured_lora_dropout = None

    kt_config = KTConfig(
        kt_backend=getattr(model_args, "kt_backend", None),
        kt_num_threads=getattr(model_args, "kt_num_threads", None),
        kt_tp_enabled=getattr(model_args, "kt_tp_enabled", None),
        kt_threadpool_count=getattr(model_args, "kt_threadpool_count", None),
        kt_max_cache_depth=getattr(model_args, "kt_max_cache_depth", None),
        kt_num_gpu_experts=getattr(model_args, "kt_num_gpu_experts", None),
        kt_weight_path=getattr(model_args, "kt_weight_path", None),
        kt_expert_weight_format=getattr(model_args, "kt_expert_weight_format", None),
        kt_weight_lifecycle=getattr(model_args, "kt_weight_lifecycle", None),
        kt_expert_checkpoint_path=getattr(model_args, "kt_expert_checkpoint_path", None),
        kt_force_fused_expert_lora=getattr(model_args, "kt_force_fused_expert_lora", None),
        kt_use_lora_experts=getattr(model_args, "kt_use_lora_experts", None),
        kt_lora_expert_num=getattr(model_args, "kt_lora_expert_num", None),
        kt_lora_expert_intermediate_size=getattr(model_args, "kt_lora_expert_intermediate_size", None),
        kt_lora_rank=configured_lora_rank,
        kt_lora_alpha=configured_lora_alpha,
        kt_lora_dropout=configured_lora_dropout,
        kt_model_max_length=getattr(model_args, "model_max_length", None),
        kt_train_mode=kt_train_mode,
        kt_activation_policy=getattr(model_args, "activation_policy", None),
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
        import inspect

        from transformers.modeling_utils import _get_resolved_checkpoint_files
    except Exception:
        return None, None
    try:
        common = {
            "pretrained_model_name_or_path": model_name_or_path,
            "variant": None,
            "gguf_file": None,
            "use_safetensors": None,
            "user_agent": {"file_type": "model", "framework": "pytorch"},
            "is_remote_code": bool(trust_remote_code),
            "transformers_explicit_filename": None,
        }
        if "download_kwargs" in inspect.signature(_get_resolved_checkpoint_files).parameters:
            common["download_kwargs"] = {
                "cache_dir": cache_dir,
                "force_download": False,
                "local_files_only": False,
                "token": token,
                "revision": revision or "main",
                "subfolder": "",
            }
            checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(**common)
        else:
            checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
                **common,
                subfolder="",
                from_tf=False,
                from_flax=False,
                cache_dir=cache_dir,
                force_download=False,
                proxies=None,
                local_files_only=False,
                token=token,
                revision=revision or "main",
                commit_hash=None,
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
    from .arch import (
        get_moe_arch_config,
        move_non_experts_to_gpu,
        KTAMXConfigError,
    )

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
        config,
        kt_plugin,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if model_args is not None:
        for key in ("cache_dir", "revision"):
            value = getattr(model_args, key, None)
            if value is not None:
                loading_kwargs[key] = value
    loading_kwargs.update(kwargs)

    cfg = _get_kt_config(kt_plugin)
    auto_full_weight_checkpoint = resolve_full_weight_checkpoint(model_name_or_path)
    if auto_full_weight_checkpoint is not None and getattr(cfg, "kt_expert_checkpoint_path", None) is None:
        cfg.kt_expert_checkpoint_path = auto_full_weight_checkpoint
        plugin_config = getattr(kt_plugin, "kt_config", None)
        if isinstance(plugin_config, dict):
            plugin_config["kt_expert_checkpoint_path"] = auto_full_weight_checkpoint
        elif plugin_config is not None:
            setattr(plugin_config, "kt_expert_checkpoint_path", auto_full_weight_checkpoint)
        else:
            setattr(kt_plugin, "kt_expert_checkpoint_path", auto_full_weight_checkpoint)
        logger.info("Detected KT Full checkpoint in model directory: %s", auto_full_weight_checkpoint)

    native_fp8_experts = getattr(cfg, "kt_expert_weight_format", None) == "fp8"
    if native_fp8_experts:
        # FP8 kt_weight_path is raw-checkpoint provenance; KT loads routed experts.
        cfg.kt_skip_expert_loading = True

    skip_expert_loading = getattr(cfg, "kt_skip_expert_loading", None)
    needs_checkpoint_resolution = (
        (native_fp8_experts and not getattr(cfg, "kt_checkpoint_files", None))
        or skip_expert_loading is None
        or (
            bool(skip_expert_loading)
            and not getattr(cfg, "kt_checkpoint_files", None)
            and not getattr(cfg, "kt_weight_path", None)
        )
    )
    if needs_checkpoint_resolution:
        checkpoint_files, sharded_metadata = _resolve_checkpoint_files(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            trust_remote_code=trust_remote_code,
        )
        if checkpoint_files and all(f.endswith(".safetensors") for f in checkpoint_files):
            if native_fp8_experts or getattr(cfg, "kt_weight_path", None) is None:
                cfg.kt_skip_expert_loading = True
            else:
                cfg.kt_skip_expert_loading = False
            cfg.kt_checkpoint_files = checkpoint_files
            cfg.kt_sharded_metadata = sharded_metadata
        else:
            if not native_fp8_experts:
                cfg.kt_skip_expert_loading = False

    if native_fp8_experts:
        checkpoint_files = getattr(cfg, "kt_checkpoint_files", None)
        if not checkpoint_files or not all(
            str(path).endswith(".safetensors") for path in checkpoint_files
        ):
            raise KTAMXConfigError(
                "native FP8 SFT requires raw safetensors checkpoint files; "
                "no Transformers expert-materialization fallback is supported"
            )
        cfg.kt_skip_expert_loading = True

    # Transformers consumes the resolved fields while from_pretrained runs;
    # Accelerate deliberately keeps those fields opaque under plugin.kt_config.
    set_kt_config(cfg)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **loading_kwargs)
    finally:
        unset_kt_config()

    moe_config = get_moe_arch_config(config)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    move_non_experts_to_gpu(model, moe_config, device=f"cuda:{local_rank}")

    existing_wrappers = getattr(model, "_kt_wrappers", None)
    if existing_wrappers:
        logger.info(f"MoE layers already wrapped ({len(existing_wrappers)} layers), skipping re-wrap")
        wrappers = existing_wrappers
    else:
        wrappers = wrap_moe_layers_with_kt_wrapper(model, kt_plugin)

    model._kt_wrappers = wrappers
    model._kt_tp_enabled = bool(getattr(cfg, "kt_tp_enabled", False))
    model._kt_use_lora_experts = bool(getattr(cfg, "kt_use_lora_experts", False))
    model._kt_full_weight_grad = bool(getattr(cfg, "kt_full_weight_grad", False))
    model._kt_train_mode = getattr(cfg, "kt_train_mode", "lora")
    model._kt_expert_weight_format = getattr(cfg, "kt_expert_weight_format", None) or "bf16"

    logger.info("Model loaded with KTMoEWrapper backend successfully")
    return model
