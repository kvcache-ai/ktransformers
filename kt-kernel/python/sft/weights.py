# Weight extraction and loading utilities for SFT
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn

from .arch import MOEArchConfig
from .dist_utils import _maybe_zero3_gathered_parameters

logger = logging.getLogger(__name__)

try:
    from safetensors import safe_open

    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    safe_open = None


# =============================================================================
# Weight Extraction
# =============================================================================


def extract_moe_weights(
    moe_module: nn.Module, moe_config: MOEArchConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract MoE expert weights from the module.

    Returns (gate_proj, up_proj, down_proj) with shape
    [expert_num, out_features, in_features].

    Supports two formats:
    - ModuleList of Linear experts (transformers v4 style)
    - Fused Parameters (transformers v5 style): single module with
      ``gate_up_proj`` [E, 2*I, H] and ``down_proj`` [E, H, I] tensors.
    """
    from .arch import detect_fused_experts

    experts = getattr(moe_module, moe_config.experts_attr)

    # Fused format (transformers v5): a single nn.Module with gate_up_proj/down_proj tensors
    if detect_fused_experts(experts):
        gate_up = getattr(experts, "gate_up_proj").data
        down_fused = getattr(experts, "down_proj").data
        # gate_up_proj is [E, 2*I, H], split into gate [E, I, H] and up [E, I, H]
        intermediate = gate_up.shape[1] // 2
        gate_proj = gate_up[:, :intermediate, :].contiguous()
        up_proj = gate_up[:, intermediate:, :].contiguous()
        # down_proj is already [E, H, I]
        down_proj = down_fused.contiguous()
        return gate_proj, up_proj, down_proj

    gate_name, up_name, down_name = moe_config.weight_names

    gather_params: list[torch.nn.Parameter] = []
    for expert in experts:
        for weight_name in (gate_name, up_name, down_name):
            proj = getattr(expert, weight_name, None)
            if proj is not None and hasattr(proj, "weight"):
                # Handle PEFT LoRA wrapped modules
                weight = proj.weight
                if isinstance(weight, torch.Tensor):
                    gather_params.append(weight)
                elif hasattr(weight, "data"):
                    gather_params.append(weight.data)

    with _maybe_zero3_gathered_parameters(gather_params):
        gate_weights = []
        up_weights = []
        down_weights = []

        for expert in experts:
            # Handle PEFT LoRA wrapped modules - get weight tensor properly
            gate_proj = getattr(expert, gate_name)
            up_proj_mod = getattr(expert, up_name)
            down_proj_mod = getattr(expert, down_name)

            # Get weight tensors, handling both regular Linear and PEFT LoRA wrapped
            def get_weight_tensor(mod):
                weight = mod.weight
                if isinstance(weight, torch.Tensor):
                    return weight.data
                elif hasattr(weight, "data"):
                    return weight.data
                else:
                    raise ValueError(f"Cannot extract weight from {type(mod)}, weight type={type(weight)}")

            gate_weights.append(get_weight_tensor(gate_proj))
            up_weights.append(get_weight_tensor(up_proj_mod))
            down_weights.append(get_weight_tensor(down_proj_mod))

    gate_proj = torch.stack(gate_weights, dim=0)
    up_proj = torch.stack(up_weights, dim=0)
    down_proj = torch.stack(down_weights, dim=0)

    return gate_proj, up_proj, down_proj


def _clear_original_expert_weights(moe_module: nn.Module, moe_config: MOEArchConfig) -> None:
    """
    Clear original expert weights to free memory after KT weights are loaded.
    """
    from .arch import detect_fused_experts

    experts = getattr(moe_module, moe_config.experts_attr, None)
    if experts is None:
        return

    # Fused format: replace gate_up_proj/down_proj tensors with zero-storage placeholders
    if detect_fused_experts(experts):
        for name in ("gate_up_proj", "down_proj"):
            param = getattr(experts, name, None)
            if not isinstance(param, torch.nn.Parameter):
                continue
            original_dtype = param.dtype
            tiny_storage = torch.UntypedStorage(1, device="cpu")
            fake_tensor = torch.tensor([], dtype=original_dtype, device="cpu").set_(
                tiny_storage, storage_offset=0, size=param.shape,
                stride=[0] * len(param.shape),
            )
            experts._parameters[name] = nn.Parameter(fake_tensor, requires_grad=False)
        return

    def _iter_weight_params():
        for expert in experts:
            for weight_name in moe_config.weight_names:
                proj = getattr(expert, weight_name, None)
                if proj is None or not hasattr(proj, "weight"):
                    continue

                parametrizations = getattr(proj, "parametrizations", None)
                parametrized_weight = getattr(parametrizations, "weight", None) if parametrizations is not None else None
                if parametrized_weight is not None:
                    original = getattr(parametrized_weight, "original", None)
                    if isinstance(original, torch.nn.Parameter):
                        yield proj, parametrized_weight, "original", original
                        continue

                direct_weight = getattr(proj, "_parameters", {}).get("weight")
                if isinstance(direct_weight, torch.nn.Parameter):
                    yield proj, proj, "weight", direct_weight
                    continue

                # Fallback: `weight` can be a non-settable property (e.g. parametrizations) or a non-Parameter.
                weight_attr = getattr(proj, "weight", None)
                if isinstance(weight_attr, torch.nn.Parameter):
                    yield proj, proj, "weight", weight_attr

    gather_params: list[torch.nn.Parameter] = []
    for _, _, _, weight_param in _iter_weight_params():
        gather_params.append(weight_param)

    replaced_count = 0

    with _maybe_zero3_gathered_parameters(gather_params):
        for proj, container, param_name, weight_param in _iter_weight_params():
            original_dtype = weight_param.dtype

            # Create a CPU tensor with the correct shape but NO physical memory.
            # torch.empty(shape, device="cpu") unfortunately touches pages via the
            # allocator, consuming real RSS.  Instead, allocate a 1-byte storage and
            # use set_ to give it the original shape with zero strides.  The tensor
            # is "valid" (correct dtype, device, shape) so PEFT can discover
            # in/out features, but its storage is essentially zero-cost.
            # NOTE: reading element values from this tensor is undefined -- it is
            # only used for shape/dtype discovery by PEFT.
            tiny_storage = torch.UntypedStorage(1, device="cpu")
            fake_tensor = torch.tensor([], dtype=original_dtype, device="cpu").set_(
                tiny_storage, storage_offset=0, size=weight_param.shape,
                stride=[0] * len(weight_param.shape),
            )
            new_param = nn.Parameter(fake_tensor, requires_grad=False)
            replaced_count += 1

            # Avoid `KeyError: attribute 'weight' already exists` for parametrized modules
            # where `weight` is a property and the real parameter lives elsewhere.
            container_params = getattr(container, "_parameters", {})
            if isinstance(container_params, dict) and param_name in container_params:
                container_params[param_name] = new_param
                continue

            if hasattr(container, param_name):
                logger.debug(
                    f"Skipping clearing expert weight {type(proj).__name__}.{param_name}: "
                    "attribute exists but is not a registered parameter."
                )
                continue

            try:
                setattr(container, param_name, new_param)
            except Exception as exc:
                logger.warning(
                    f"Failed to clear expert weight {type(proj).__name__}.{param_name}: {exc}"
                )

    logger.info(f"Replaced {replaced_count} expert weight params")


# =============================================================================
# kt_weight_path Loading Functions
# =============================================================================


@dataclass
class INT8ExpertWeights:
    """Container for INT8 expert weights with scales."""

    gate_proj: torch.Tensor
    gate_scale: torch.Tensor
    up_proj: torch.Tensor
    up_scale: torch.Tensor
    down_proj: torch.Tensor
    down_scale: torch.Tensor


def _find_safetensor_files(kt_weight_path: str) -> list[str]:
    if not os.path.isdir(kt_weight_path):
        raise FileNotFoundError(f"kt_weight_path directory not found: {kt_weight_path}")

    safetensor_files = []
    for file in sorted(os.listdir(kt_weight_path)):
        if file.endswith(".safetensors"):
            safetensor_files.append(os.path.join(kt_weight_path, file))

    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {kt_weight_path}")

    return safetensor_files


def _load_kt_weight_index(kt_weight_path: str) -> dict[str, str]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    index = {}
    safetensor_files = _find_safetensor_files(kt_weight_path)

    for file_path in safetensor_files:
        with safe_open(file_path, framework="pt") as f:
            for key in f.keys():
                index[key] = file_path

    logger.info(f"Indexed {len(index)} tensors from {len(safetensor_files)} safetensors files")
    return index


def _dequant_fp8_experts(weights: list[torch.Tensor], scales: list[torch.Tensor | None], block_size: tuple[int, int]) -> torch.Tensor:
    """Dequantize a list of FP8 expert weights and stack them (batched, vectorized).

    Args:
        weights: list of [out, in] float8_e4m3fn tensors (one per expert)
        scales: list of [out//bs_m, in//bs_n] scale_inv tensors (one per expert, may be None)
        block_size: (bs_m, bs_n)

    Returns:
        Stacked BF16 tensor of shape [num_experts, out, in]
    """
    has_scales = scales[0] is not None
    if not has_scales:
        return torch.stack(weights, dim=0).to(torch.bfloat16).cpu().contiguous()

    bs_m, bs_n = block_size
    n = len(weights)
    out_features, in_features = weights[0].shape

    # Stack all experts: [N, out, in] fp8 -> reshape to blocks -> bf16
    w = torch.stack(weights, dim=0)  # [N, out, in] fp8
    w = w.reshape(n, out_features // bs_m, bs_m, in_features // bs_n, bs_n)
    w = w.to(torch.bfloat16)

    # Stack all scales: [N, out//bs_m, in//bs_n] -> bf16, broadcast multiply
    s = torch.stack(scales, dim=0).to(torch.bfloat16)  # [N, out//bs_m, in//bs_n]
    w = w * s[:, :, None, :, None]

    return w.reshape(n, out_features, in_features).contiguous()


def load_experts_from_checkpoint_files(
    checkpoint_files: list[str],
    sharded_metadata: dict | None,
    layers_prefix: str,
    moe_config: MOEArchConfig,
    layer_idx: int,
    block_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading experts from checkpoint files")

    if not checkpoint_files:
        raise FileNotFoundError("checkpoint_files is empty")

    t0 = time.time()

    weight_map = None
    base_dir = os.path.dirname(checkpoint_files[0])
    if sharded_metadata is not None:
        weight_map = sharded_metadata.get("weight_map", None)

    gate_name, up_name, down_name = moe_config.weight_names
    keys = []
    for expert_idx in range(moe_config.expert_num):
        base = f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"
        keys.append(f"{base}.{gate_name}.weight")
        keys.append(f"{base}.{gate_name}.weight_scale_inv")
        keys.append(f"{base}.{up_name}.weight")
        keys.append(f"{base}.{up_name}.weight_scale_inv")
        keys.append(f"{base}.{down_name}.weight")
        keys.append(f"{base}.{down_name}.weight_scale_inv")

    keys_by_file: dict[str, list[str]] = {}
    mapped_count = 0
    unmapped_count = 0
    for key in keys:
        if weight_map is not None:
            filename = weight_map.get(key)
            if filename is None:
                unmapped_count += 1
                continue
            mapped_count += 1
            file_path = os.path.join(base_dir, filename)
        else:
            file_path = checkpoint_files[0]
        keys_by_file.setdefault(file_path, []).append(key)

    print(
        f"[kt_moe] Layer {layer_idx}: key mapping done in {time.time()-t0:.1f}s — "
        f"total_keys={len(keys)}, mapped={mapped_count}, unmapped={unmapped_count}, "
        f"files_to_open={len(keys_by_file)}",
        flush=True,
    )

    t1 = time.time()
    tensor_map: dict[str, torch.Tensor] = {}
    for file_idx, (file_path, file_keys) in enumerate(keys_by_file.items()):
        with safe_open(file_path, framework="pt") as f:
            available_keys = set(f.keys())
            for key in file_keys:
                if key in available_keys:
                    tensor_map[key] = f.get_tensor(key)
        if file_idx == 0:
            print(
                f"[kt_moe] Layer {layer_idx}: first file loaded ({os.path.basename(file_path)}, "
                f"{len(file_keys)} keys) in {time.time()-t1:.1f}s",
                flush=True,
            )

    print(
        f"[kt_moe] Layer {layer_idx}: all files loaded in {time.time()-t1:.1f}s — "
        f"tensor_map has {len(tensor_map)} tensors",
        flush=True,
    )

    gate_weights = []
    up_weights = []
    down_weights = []
    gate_scales = []
    up_scales = []
    down_scales = []
    for expert_idx in range(moe_config.expert_num):
        base = f"{layers_prefix}.{layer_idx}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}.{expert_idx}"
        gate_key = f"{base}.{gate_name}.weight"
        up_key = f"{base}.{up_name}.weight"
        down_key = f"{base}.{down_name}.weight"
        if gate_key not in tensor_map or up_key not in tensor_map or down_key not in tensor_map:
            raise FileNotFoundError(f"Missing expert weights for layer {layer_idx}, expert {expert_idx}")
        gate_weights.append(tensor_map[gate_key])
        up_weights.append(tensor_map[up_key])
        down_weights.append(tensor_map[down_key])
        gate_scales.append(tensor_map.get(f"{base}.{gate_name}.weight_scale_inv"))
        up_scales.append(tensor_map.get(f"{base}.{up_name}.weight_scale_inv"))
        down_scales.append(tensor_map.get(f"{base}.{down_name}.weight_scale_inv"))

    # Check if weights are FP8 and need dequantization
    t2 = time.time()
    is_fp8 = gate_weights[0].dtype == torch.float8_e4m3fn
    if is_fp8:
        if block_size is None:
            block_size = (128, 128)
        print(
            f"[kt_moe] Layer {layer_idx}: FP8 expert weights detected, "
            f"dequantizing with block_size={block_size} "
            f"(has_scales={gate_scales[0] is not None})",
            flush=True,
        )
        gate_proj = _dequant_fp8_experts(gate_weights, gate_scales, block_size)
        up_proj = _dequant_fp8_experts(up_weights, up_scales, block_size)
        down_proj = _dequant_fp8_experts(down_weights, down_scales, block_size)
    else:
        gate_proj = torch.stack(gate_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
        up_proj = torch.stack(up_weights, dim=0).cpu().to(torch.bfloat16).contiguous()
        down_proj = torch.stack(down_weights, dim=0).cpu().to(torch.bfloat16).contiguous()

    print(
        f"[kt_moe] Layer {layer_idx}: done — dtype={gate_proj.dtype}, shape={gate_proj.shape}, "
        f"dequant={time.time()-t2:.1f}s, total={time.time()-t0:.1f}s",
        flush=True,
    )
    return gate_proj, up_proj, down_proj


def load_experts_from_kt_weight_path(
    kt_weight_path: str,
    layer_idx: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> INT8ExpertWeights:
    """Load INT8 preprocessed expert weights from kt_weight_path for a specific layer."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors is required for loading kt_weight_path")

    index = _load_kt_weight_index(kt_weight_path)

    numa_count = 0
    test_key_prefix = f"blk.{layer_idx}.ffn_gate_exps.0.numa."
    for key in index.keys():
        if key.startswith(test_key_prefix) and key.endswith(".weight"):
            numa_idx = int(key.split("numa.")[1].split(".")[0])
            numa_count = max(numa_count, numa_idx + 1)

    if numa_count == 0:
        raise FileNotFoundError(
            f"No weights found for layer {layer_idx} in {kt_weight_path}. "
            f"Expected keys like 'blk.{layer_idx}.ffn_gate_exps.0.numa.0.weight'"
        )

    logger.info(
        f"Loading INT8 weights for layer {layer_idx}: {num_experts} experts, {numa_count} NUMA partitions"
    )

    gate_weights_list = []
    gate_scales_list = []
    up_weights_list = []
    up_scales_list = []
    down_weights_list = []
    down_scales_list = []

    for expert_idx in range(num_experts):
        gate_w_parts = []
        gate_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_gate_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                gate_w_parts.append(f.get_tensor(w_key))
                gate_s_parts.append(f.get_tensor(s_key))

        gate_w = torch.cat(gate_w_parts, dim=0)
        gate_s = torch.cat(gate_s_parts, dim=0)
        gate_w = gate_w.view(intermediate_size, hidden_size)

        gate_weights_list.append(gate_w)
        gate_scales_list.append(gate_s)

        up_w_parts = []
        up_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_up_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                up_w_parts.append(f.get_tensor(w_key))
                up_s_parts.append(f.get_tensor(s_key))

        up_w = torch.cat(up_w_parts, dim=0)
        up_s = torch.cat(up_s_parts, dim=0)
        up_w = up_w.view(intermediate_size, hidden_size)

        up_weights_list.append(up_w)
        up_scales_list.append(up_s)

        down_w_parts = []
        down_s_parts = []
        for numa_idx in range(numa_count):
            w_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.weight"
            s_key = f"blk.{layer_idx}.ffn_down_exps.{expert_idx}.numa.{numa_idx}.scale"

            if w_key not in index:
                raise FileNotFoundError(f"Weight key not found: {w_key}")

            with safe_open(index[w_key], framework="pt") as f:
                down_w_parts.append(f.get_tensor(w_key))
                down_s_parts.append(f.get_tensor(s_key))

        down_w = torch.cat(down_w_parts, dim=0)
        down_s = torch.cat(down_s_parts, dim=0)
        down_w = down_w.view(hidden_size, intermediate_size)

        down_weights_list.append(down_w)
        down_scales_list.append(down_s)

    gate_proj = torch.stack(gate_weights_list, dim=0)
    gate_scale = torch.stack(gate_scales_list, dim=0)
    up_proj = torch.stack(up_weights_list, dim=0)
    up_scale = torch.stack(up_scales_list, dim=0)
    down_proj = torch.stack(down_weights_list, dim=0)
    down_scale = torch.stack(down_scales_list, dim=0)

    return INT8ExpertWeights(
        gate_proj=gate_proj,
        gate_scale=gate_scale,
        up_proj=up_proj,
        up_scale=up_scale,
        down_proj=down_proj,
        down_scale=down_scale,
    )
