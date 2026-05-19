# SFT (Supervised Fine-Tuning) submodule for kt-kernel
# SPDX-License-Identifier: Apache-2.0

"""
SFT training support for KT-Kernel MoE.

This submodule adds training capabilities (forward/backward, LoRA, autograd,
distributed) on top of the inference-only kt_kernel base package.

Additional dependencies beyond base kt_kernel: torch.nn, torch.distributed, peft (optional).
"""

from .config import KTConfig
from .base import BaseSFTMoEWrapper, KExpertsSFTBuffer
from .amx import AMXSFTMoEWrapper
from .arch import (
    MOEArchConfig, get_moe_arch_config, get_moe_module, move_non_experts_to_gpu, get_expert_device,
    KTAMXError, KTAMXNotAvailableError, KTAMXModelNotSupportedError, KTAMXConfigError,
)
from .autograd import KTMoEFunction
from .layer import KTMoELayerWrapper
from .weights import (
    extract_moe_weights,
    load_experts_from_checkpoint_files,
    load_experts_from_kt_weight_path,
    INT8ExpertWeights,
)
from .lora import (
    kt_adapt_peft_lora,
    get_kt_lora_params,
    update_kt_lora_pointers,
    sync_kt_lora_gradients,
    save_lora_experts_to_adapter,
    save_kt_moe_to_adapter,
    load_lora_experts_from_adapter,
    load_kt_moe_from_adapter,
    LoRAExpertMLP,
    LoRAExperts,
)
from .wrapper import (
    wrap_moe_layers_with_kt_wrapper,
    build_kt_device_map,
    build_kt_device_map_simplified,
    get_kt_loading_kwargs,
    load_kt_model,
)

__all__ = [
    "KTConfig",
    "BaseSFTMoEWrapper",
    "KExpertsSFTBuffer",
    "AMXSFTMoEWrapper",
    "MOEArchConfig",
    "get_moe_arch_config",
    "get_moe_module",
    "move_non_experts_to_gpu",
    "get_expert_device",
    "KTAMXError",
    "KTAMXNotAvailableError",
    "KTAMXModelNotSupportedError",
    "KTAMXConfigError",
    "KTMoEFunction",
    "KTMoELayerWrapper",
    "extract_moe_weights",
    "load_experts_from_checkpoint_files",
    "load_experts_from_kt_weight_path",
    "INT8ExpertWeights",
    "kt_adapt_peft_lora",
    "get_kt_lora_params",
    "update_kt_lora_pointers",
    "sync_kt_lora_gradients",
    "save_lora_experts_to_adapter",
    "save_kt_moe_to_adapter",
    "load_lora_experts_from_adapter",
    "load_kt_moe_from_adapter",
    "LoRAExpertMLP",
    "LoRAExperts",
    "wrap_moe_layers_with_kt_wrapper",
    "build_kt_device_map",
    "build_kt_device_map_simplified",
    "get_kt_loading_kwargs",
    "load_kt_model",
]
