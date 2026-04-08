# MoE architecture configuration and model utilities
# SPDX-License-Identifier: Apache-2.0

"""
MoE architecture detection and model navigation utilities.

This is a leaf module — no imports from other sft/ submodules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class KTAMXError(Exception):
    """Base exception for KT AMX errors."""


class KTAMXNotAvailableError(KTAMXError):
    """kt_kernel not installed or AMX not supported."""


class KTAMXModelNotSupportedError(KTAMXError):
    """Model architecture not supported."""


class KTAMXConfigError(KTAMXError):
    """Configuration error."""


# =============================================================================
# MoE Configuration
# =============================================================================


@dataclass
class MOEArchConfig:
    """MoE architecture configuration for different model types."""

    moe_layer_attr: str
    router_attr: str
    experts_attr: str
    weight_names: tuple[str, str, str]
    expert_num: int
    intermediate_size: int
    num_experts_per_tok: int
    has_shared_experts: bool = False
    router_type: str = "linear"


def get_moe_arch_config(config) -> MOEArchConfig:
    """
    Get MoE architecture configuration based on model type.

    Args:
        config: HuggingFace model configuration

    Returns:
        MOEArchConfig for the model

    Raises:
        KTAMXModelNotSupportedError: If model architecture is not supported
    """
    arch = config.architectures[0] if getattr(config, "architectures", None) else ""

    if "DeepseekV2" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",
        )
    if "DeepseekV3" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.n_routed_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "n_shared_experts", 0) > 0,
            router_type="deepseek_gate",
        )
    if "Qwen2Moe" in arch or "Qwen3Moe" in arch:
        return MOEArchConfig(
            moe_layer_attr="mlp",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("gate_proj", "up_proj", "down_proj"),
            expert_num=config.num_experts,
            intermediate_size=config.moe_intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=getattr(config, "shared_expert_intermediate_size", 0) > 0,
        )
    if "Mixtral" in arch:
        return MOEArchConfig(
            moe_layer_attr="block_sparse_moe",
            router_attr="gate",
            experts_attr="experts",
            weight_names=("w1", "w3", "w2"),
            expert_num=config.num_local_experts,
            intermediate_size=config.intermediate_size,
            num_experts_per_tok=config.num_experts_per_tok,
            has_shared_experts=False,
        )

    raise KTAMXModelNotSupportedError(
        f"Model architecture {arch} not supported for KT AMX. "
        "Supported architectures: DeepseekV2, DeepseekV3, Qwen2Moe, Qwen3Moe, Mixtral"
    )


def get_moe_module(layer: nn.Module, moe_config: MOEArchConfig) -> nn.Module | None:
    """Get MoE module from transformer layer."""
    moe_module = getattr(layer, moe_config.moe_layer_attr, None)
    if moe_module is None:
        return None
    if not hasattr(moe_module, moe_config.experts_attr):
        return None
    return moe_module


def _get_layers_prefix(config) -> str:
    arch = config.architectures[0] if getattr(config, "architectures", None) else ""
    if any(x in arch for x in ["Deepseek", "Qwen", "Mixtral", "Llama"]):
        return "model.layers"
    return "model.layers"


def _get_model_container_and_layers(model: nn.Module, *, purpose: str) -> tuple[nn.Module, any]:
    """
    Resolve the transformer layer container for KT integration.

    KT expects the transformer block stack to be accessible as `<container>.layers`.
    Handles PEFT PeftModel, TRL value-head models, DDP wrappers.
    """
    to_visit: list[nn.Module] = [model]
    visited: set[int] = set()
    visited_types: list[str] = []

    while to_visit:
        current = to_visit.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        visited_types.append(type(current).__name__)

        layers = getattr(current, "layers", None)
        if layers is not None and isinstance(layers, (list, tuple, nn.ModuleList)):
            return current, layers

        for attr in ("model", "base_model", "pretrained_model", "module"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module) and child is not current:
                to_visit.append(child)

        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            try:
                base = get_base_model()
            except Exception:
                base = None
            if isinstance(base, nn.Module) and base is not current:
                to_visit.append(base)

    visited_preview = ", ".join(visited_types[:6])
    if len(visited_types) > 6:
        visited_preview += ", ..."

    raise KTAMXConfigError(
        f"Model does not expose a .model.layers or .layers attribute for KT {purpose}. "
        "Tried unwrapping via model/base_model/pretrained_model/module/get_base_model; "
        f"visited: {visited_preview}"
    )


def move_non_experts_to_gpu(
    model: nn.Module,
    moe_config: MOEArchConfig | None = None,
    device: str = "cuda:0",
) -> None:
    """Move non-expert parameters to GPU after loading (experts stay on CPU)."""
    if moe_config is None:
        config = getattr(model, "config", None)
        if config is None:
            raise KTAMXConfigError("Model config is required to infer MoE architecture.")
        moe_config = get_moe_arch_config(config)

    container, layers = _get_model_container_and_layers(model, purpose="placement")

    if hasattr(container, "embed_tokens"):
        container.embed_tokens.to(device)
    if hasattr(container, "norm"):
        container.norm.to(device)
    if hasattr(model, "lm_head"):
        model.lm_head.to(device)

    for layer in layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.to(device)

        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm.to(device)
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm.to(device)

        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None or not hasattr(moe_module, moe_config.experts_attr):
            if hasattr(layer, "mlp"):
                layer.mlp.to(device)
            continue

        router = getattr(moe_module, moe_config.router_attr, None)
        if router is not None:
            router.to(device)

        if hasattr(moe_module, "shared_experts") and moe_module.shared_experts is not None:
            moe_module.shared_experts.to(device)

    logger.info(f"Moved non-expert parameters to {device}")


def get_expert_device(model: nn.Module, moe_config: MOEArchConfig | None = None) -> str:
    """Get the device type of MoE experts."""
    if moe_config is None:
        config = getattr(model, "config", None)
        if config is None:
            return "unknown"
        moe_config = get_moe_arch_config(config)

    try:
        _, layers = _get_model_container_and_layers(model, purpose="expert device probing")
    except KTAMXConfigError:
        return "unknown"

    for layer in layers:
        moe_module = getattr(layer, moe_config.moe_layer_attr, None)
        if moe_module is None:
            continue
        experts = getattr(moe_module, moe_config.experts_attr, None)
        if not experts:
            continue
        first_expert = experts[0]
        gate_name = moe_config.weight_names[0]
        gate_proj = getattr(first_expert, gate_name, None)
        if gate_proj is not None:
            return str(gate_proj.weight.device.type)

    return "unknown"
