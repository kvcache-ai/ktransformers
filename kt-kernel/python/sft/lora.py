# PEFT LoRA adaptation utilities for SFT
# SPDX-License-Identifier: Apache-2.0

"""
PEFT LoRA integration for KT-Kernel MoE training.

Handles:
- LoRA Expert modules (LoRAExpertMLP, LoRAExperts)
- PEFT LoRA adaptation onto KT wrappers (contiguous buffer views, grad buffers)
- LoRA parameter collection for optimizer injection
- Checkpoint save/load for lora_experts
"""

from __future__ import annotations

import logging
import math
import os
import re

import torch
import torch.nn as nn

from .arch import MOEArchConfig

logger = logging.getLogger(__name__)


# =============================================================================
# LoRA Experts Modules
# =============================================================================


class LoRAExpertMLP(nn.Module):
    """Single LoRA Expert with SwiGLU activation structure."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.le_gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.le_up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.le_down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

        nn.init.zeros_(self.le_down.weight)
        nn.init.kaiming_uniform_(self.le_gate.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.le_up.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.le_down(self.act_fn(self.le_gate(x)) * self.le_up(x))


class LoRAExperts(nn.Module):
    """LoRA Experts module containing multiple LoRA Expert MLPs."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [LoRAExpertMLP(hidden_size, intermediate_size, device, dtype) for _ in range(num_experts)]
        )
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden_states)
        for expert in self.experts:
            output = output + expert(hidden_states)
        return output / self.num_experts


# =============================================================================
# LoRA Parameter Collection
# =============================================================================


def _find_kt_wrappers(model: nn.Module):
    """Find _kt_wrappers on model, unwrapping PEFT/other wrappers if needed."""
    wrappers = getattr(model, "_kt_wrappers", None)
    if wrappers is None:
        base_model = model
        for attr in ("base_model", "model"):
            if hasattr(base_model, attr):
                base_model = getattr(base_model, attr)
                wrappers = getattr(base_model, "_kt_wrappers", None)
                if wrappers:
                    break
    return wrappers


def get_kt_lora_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all MoE LoRA parameters from KT model.

    Returns PEFT LoRA parameters from expert modules and lora_experts parameters.
    """
    params: list[nn.Parameter] = []

    wrappers = _find_kt_wrappers(model)

    if wrappers:
        for wrapper in wrappers:
            # PEFT LoRA parameters (from _peft_lora_modules)
            peft_lora_modules = getattr(wrapper, "_peft_lora_modules", None)
            if peft_lora_modules is not None:
                for expert_loras in peft_lora_modules.values():
                    for lora_A, lora_B in expert_loras.values():
                        if hasattr(lora_A, 'weight') and lora_A.weight.requires_grad:
                            params.append(lora_A.weight)
                        if hasattr(lora_B, 'weight') and lora_B.weight.requires_grad:
                            params.append(lora_B.weight)
            # Fused expert LoRA parameters (KT-managed, not PEFT)
            fused_params = getattr(wrapper, "_fused_expert_lora_params", None)
            if fused_params is not None:
                params.extend(fused_params)
            # lora_experts parameters (separate feature)
            if getattr(wrapper, "lora_experts", None) is not None:
                params.extend(wrapper.lora_experts.parameters())

    return params


# =============================================================================
# PEFT LoRA Adaptation
# =============================================================================


def kt_adapt_peft_lora(model: nn.Module) -> None:
    """
    Adapt PEFT LoRA on expert modules for KT kernel.

    After PEFT injects LoRA adapters onto expert Linear modules, this function:
    1. Detects PEFT LoRA presence and rank on each wrapper's experts
    2. Stores references to PEFT LoRA modules on the wrapper (for backward gradient writing)
    3. Syncs initial PEFT LoRA weights to the C++ KT kernel (rank 0 only)

    PEFT LoRA remains active and is managed by PEFT. No separate KT lora_params created.
    Optimizer updates PEFT LoRA directly, and KT kernel reads from PEFT LoRA on each forward.

    Should be called after PEFT LoRA injection and before create_optimizer.
    """
    import torch.distributed as dist

    wrappers = _find_kt_wrappers(model)

    if not wrappers:
        logger.info("[kt_adapt_peft_lora] No _kt_wrappers found, skipping")
        return

    is_rank_0 = True
    if dist.is_initialized():
        is_rank_0 = dist.get_rank() == 0

    adapted_count = 0
    for wrapper in wrappers:
        moe_config = wrapper.moe_config
        layer_idx = wrapper.layer_idx
        experts_attr = getattr(wrapper, "_experts_attr", "experts")
        experts = getattr(wrapper, experts_attr, None)

        if experts is None:
            continue

        # Fused experts (transformers v5): PEFT cannot auto-attach LoRA to packed
        # nn.Parameter tensors. Create KT-managed LoRA buffers with proper init,
        # wrap as nn.Parameter for optimizer, and pre-assign .grad for C++ backward.
        if getattr(wrapper, "_fused_experts", False):
            lora_rank = getattr(wrapper, "_lora_rank", 1)
            lora_buffers, lora_grad_buffers, lora_params = _create_fused_expert_lora_buffers(
                wrapper, moe_config, lora_rank, torch.bfloat16,
            )

            if is_rank_0 and wrapper.wrapper is not None:
                all_buffers = {}
                all_buffers.update(lora_buffers)
                all_buffers.update(lora_grad_buffers)
                wrapper.wrapper.init_lora_weights(**all_buffers)
                logger.info(
                    f"[kt_adapt_peft_lora] Layer {layer_idx}: fused expert LoRA "
                    f"(r={lora_rank}, E={moe_config.expert_num})"
                )

            wrapper._fused_expert_lora_params = lora_params
            wrapper._peft_lora_modules = None
            adapted_count += 1
            continue

        if len(experts) == 0:
            continue

        # Collect references to PEFT LoRA modules for each expert
        # Structure: {expert_idx: {proj_name: (lora_A_module, lora_B_module)}}
        peft_lora_modules = {}
        gate_name, up_name, down_name = moe_config.weight_names

        for expert_idx, expert in enumerate(experts):
            expert_loras = {}
            for proj_name in (gate_name, up_name, down_name):
                proj = getattr(expert, proj_name, None)
                if proj is None:
                    continue
                lora_A = getattr(proj, "lora_A", None)
                lora_B = getattr(proj, "lora_B", None)
                if lora_A is not None and lora_B is not None:
                    # Get the actual Linear modules (inside ModuleDict if using adapters)
                    if isinstance(lora_A, nn.ModuleDict):
                        adapter_name = "default"
                        active = getattr(proj, "active_adapter", ["default"])
                        if isinstance(active, (list, tuple)) and active:
                            adapter_name = active[0]
                        # ModuleDict doesn't have .get(), use [] with in check
                        lora_A = lora_A[adapter_name] if adapter_name in lora_A else None
                        lora_B = lora_B[adapter_name] if adapter_name in lora_B else None
                    if lora_A is not None and lora_B is not None:
                        expert_loras[proj_name] = (lora_A, lora_B)
            if expert_loras:
                peft_lora_modules[expert_idx] = expert_loras

        # Store PEFT LoRA references on wrapper
        wrapper._peft_lora_modules = peft_lora_modules

        if not peft_lora_modules:
            raise RuntimeError(
                f"[kt_adapt_peft_lora] Layer {layer_idx}: No PEFT LoRA found on any expert. "
                f"Check that PEFT lora_target includes expert modules."
            )

        # Allocate contiguous bf16 buffers and populate with initial PEFT values (all ranks)
        lora_buffers = _create_lora_view_buffers(peft_lora_modules, moe_config, torch.bfloat16)
        lora_grad_buffers = _create_lora_grad_buffers(peft_lora_modules, moe_config)

        # Rank 0: pass buffers to C++ wrapper (init_lora_weights stores them via .contiguous() no-op)
        if is_rank_0 and wrapper.wrapper is not None:
            # concat lora_buffers and lora_grad_buffers into single dict
            lora_buffers.update(lora_grad_buffers)
            wrapper.wrapper.init_lora_weights(**lora_buffers)
            logger.info(f"[kt_adapt_peft_lora] Layer {layer_idx}: synced PEFT LoRA to C++ kernel")

        # All ranks: replace PEFT weights with views into the contiguous buffers
        _replace_peft_weights_with_views(peft_lora_modules, lora_buffers, lora_grad_buffers, moe_config)

        adapted_count += 1

    # After collecting all LoRA references, shrink expert base weight parameters
    # from their original shape (e.g. [768, 2048]) to scalar (1,).
    # These base weights were already replaced with tiny-storage stride=[0] placeholders
    # by _clear_original_expert_weights(). They have correct shape but serve no purpose
    # after PEFT injection. FSDP2 broadcasts ALL non-DTensor params, and uses
    # torch.empty(param.size()) on non-rank-0 — with the original shape this wastes
    # ~28GB+. Shrinking to (1,) reduces broadcast cost to ~30KB total.
    shrunk_count = 0
    shrunk_saved_bytes = 0
    for wrapper in wrappers:
        experts_attr = getattr(wrapper, "_experts_attr", "experts")
        experts = getattr(wrapper, experts_attr, None)
        if experts is None:
            continue
        if getattr(wrapper, "_fused_experts", False):
            continue
        for expert in experts:
            for param_name, param in list(expert.named_parameters()):
                if param.requires_grad:
                    continue  # Skip trainable params (LoRA weights)
                try:
                    storage_bytes = param.data.untyped_storage().nbytes()
                except Exception:
                    continue
                if storage_bytes > 2:
                    continue  # Skip non-placeholder params

                # This is a tiny-storage placeholder (base weight) — replace with
                # a scalar (1,) parameter so FSDP broadcasts only 1 element.
                original_numel = param.nelement()
                parts = param_name.split(".")
                container = expert
                for p in parts[:-1]:
                    container = getattr(container, p)
                local_name = parts[-1]
                container_params = getattr(container, "_parameters", {})
                if isinstance(container_params, dict) and local_name in container_params:
                    scalar_param = nn.Parameter(
                        torch.empty(1, dtype=param.dtype, device="cpu"),
                        requires_grad=False,
                    )
                    container_params[local_name] = scalar_param
                    shrunk_count += 1
                    shrunk_saved_bytes += (original_numel - 1) * param.element_size()

    if shrunk_count > 0:
        logger.info(
            f"[kt_adapt_peft_lora] Shrunk {shrunk_count} expert base weight params "
            f"to shape (1,), FSDP broadcast savings={shrunk_saved_bytes / 1024 / 1024:.1f} MB"
        )

    logger.info(f"[kt_adapt_peft_lora] Adapted {adapted_count} layers (PEFT LoRA mode)")


# =============================================================================
# Contiguous Buffer Creation
# =============================================================================


def _create_lora_view_buffers(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    moe_config: MOEArchConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """
    Allocate contiguous buffers and populate with initial PEFT LoRA values.

    Returns dict with gate_lora_a, gate_lora_b, up_lora_a, up_lora_b,
    down_lora_a, down_lora_b — each shape [num_experts, ...].
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    first_expert_loras = peft_lora_modules.get(0, {})
    if not first_expert_loras:
        raise RuntimeError("No PEFT LoRA found on expert 0")
    gate_lora = first_expert_loras.get(gate_name)
    if gate_lora is None:
        raise RuntimeError(f"No PEFT LoRA found on expert 0 {gate_name}")

    lora_rank = gate_lora[0].weight.shape[0]
    hidden_size = gate_lora[0].weight.shape[1]
    intermediate_size = gate_lora[1].weight.shape[0]

    buffers = {
        "gate_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "gate_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "up_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "up_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "down_lora_a": torch.zeros(num_experts, lora_rank, intermediate_size, dtype=dtype, device="cpu"),
        "down_lora_b": torch.zeros(num_experts, hidden_size, lora_rank, dtype=dtype, device="cpu"),
    }

    proj_to_keys = {
        gate_name: ("gate_lora_a", "gate_lora_b"),
        up_name: ("up_lora_a", "up_lora_b"),
        down_name: ("down_lora_a", "down_lora_b"),
    }
    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})
        for proj_name, (key_a, key_b) in proj_to_keys.items():
            if proj_name in expert_loras:
                lora_A, lora_B = expert_loras[proj_name]
                buffers[key_a][expert_idx].copy_(lora_A.weight.data.to(dtype=dtype))
                buffers[key_b][expert_idx].copy_(lora_B.weight.data.to(dtype=dtype))

    return buffers


def _create_lora_grad_buffers(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    moe_config: MOEArchConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """
    Allocate contiguous gradient buffers for PEFT LoRA.

    Returns dict with grad_gate_lora_a, grad_gate_lora_b, etc. — each shape [num_experts, ...].
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    first_expert_loras = peft_lora_modules.get(0, {})
    if not first_expert_loras:
        raise RuntimeError("No PEFT LoRA found on expert 0")
    gate_lora = first_expert_loras.get(gate_name)
    if gate_lora is None:
        raise RuntimeError(f"No PEFT LoRA found on expert 0 {gate_name}")

    lora_rank = gate_lora[0].weight.shape[0]
    hidden_size = gate_lora[0].weight.shape[1]
    intermediate_size = gate_lora[1].weight.shape[0]

    buffers = {
        "grad_gate_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "grad_gate_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "grad_up_lora_a": torch.zeros(num_experts, lora_rank, hidden_size, dtype=dtype, device="cpu"),
        "grad_up_lora_b": torch.zeros(num_experts, intermediate_size, lora_rank, dtype=dtype, device="cpu"),
        "grad_down_lora_a": torch.zeros(num_experts, lora_rank, intermediate_size, dtype=dtype, device="cpu"),
        "grad_down_lora_b": torch.zeros(num_experts, hidden_size, lora_rank, dtype=dtype, device="cpu"),
    }

    return buffers


def _create_fused_expert_lora_buffers(
    wrapper,
    moe_config: MOEArchConfig,
    lora_rank: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], list[nn.Parameter]]:
    """
    Create KT-managed LoRA buffers for fused expert modules.

    Fused experts store weights as 3D parameters (gate_up_proj [E, 2I, H], down_proj [E, H, I])
    rather than per-expert nn.Linear modules. PEFT can't attach per-expert LoRA to these,
    so we create our own LoRA buffers that the C++ kernel reads/writes directly.

    Returns:
        (lora_buffers, lora_grad_buffers, lora_params):
        - lora_buffers: dict of weight buffers for C++ init_lora_weights()
        - lora_grad_buffers: dict of grad buffers for C++ backward
        - lora_params: list of nn.Parameter wrappers for the optimizer
    """
    E = moe_config.expert_num
    I = moe_config.intermediate_size
    H = wrapper.hidden_size
    r = lora_rank

    logger.info(f"[_create_fused_expert_lora_buffers] E={E}, I={I}, H={H}, r={r}")

    lora_buffers = {
        "gate_lora_a": torch.zeros(E, r, H, dtype=dtype, device="cpu"),
        "gate_lora_b": torch.zeros(E, I, r, dtype=dtype, device="cpu"),
        "up_lora_a": torch.zeros(E, r, H, dtype=dtype, device="cpu"),
        "up_lora_b": torch.zeros(E, I, r, dtype=dtype, device="cpu"),
        "down_lora_a": torch.zeros(E, r, I, dtype=dtype, device="cpu"),
        "down_lora_b": torch.zeros(E, H, r, dtype=dtype, device="cpu"),
    }

    for key in ("gate_lora_a", "up_lora_a", "down_lora_a"):
        nn.init.kaiming_uniform_(lora_buffers[key].view(E * r, -1), a=math.sqrt(5))

    lora_grad_buffers = {
        "grad_gate_lora_a": torch.zeros(E, r, H, dtype=dtype, device="cpu"),
        "grad_gate_lora_b": torch.zeros(E, I, r, dtype=dtype, device="cpu"),
        "grad_up_lora_a": torch.zeros(E, r, H, dtype=dtype, device="cpu"),
        "grad_up_lora_b": torch.zeros(E, I, r, dtype=dtype, device="cpu"),
        "grad_down_lora_a": torch.zeros(E, r, I, dtype=dtype, device="cpu"),
        "grad_down_lora_b": torch.zeros(E, H, r, dtype=dtype, device="cpu"),
    }

    lora_params = []
    for key in ("gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b"):
        param = nn.Parameter(lora_buffers[key], requires_grad=True)
        param.grad = lora_grad_buffers[f"grad_{key}"]
        lora_params.append(param)

    return lora_buffers, lora_grad_buffers, lora_params


# =============================================================================
# PEFT Weight View Replacement
# =============================================================================


def _replace_peft_weights_with_views(
    peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]],
    buffers: dict[str, torch.Tensor],
    grad_buffers: dict[str, torch.Tensor],
    moe_config: MOEArchConfig,
) -> None:
    """
    Replace each PEFT LoRA module's .weight with a view into the contiguous buffer.

    After this, optimizer.step() updates the buffer in-place via the view —
    no copy needed to sync with C++.
    """
    gate_name, up_name, down_name = moe_config.weight_names
    num_experts = moe_config.expert_num

    proj_to_keys = {
        gate_name: ("gate_lora_a", "gate_lora_b"),
        up_name: ("up_lora_a", "up_lora_b"),
        down_name: ("down_lora_a", "down_lora_b"),
    }

    _replaced = 0
    _first_logged = False
    for expert_idx in range(num_experts):
        expert_loras = peft_lora_modules.get(expert_idx, {})
        for proj_name, (key_a, key_b) in proj_to_keys.items():
            if proj_name not in expert_loras:
                continue
            lora_A, lora_B = expert_loras[proj_name]

            # Log before/after for first replacement to verify .data assignment
            if not _first_logged:
                _old_id_a = id(lora_A.weight)
                _old_ptr_a = lora_A.weight.data_ptr()

            # Use .data assignment to keep the same Parameter objects.
            # This preserves optimizer references (which point to these objects).
            # Creating new nn.Parameter() would break the optimizer link.
            lora_A.weight.data = buffers[key_a][expert_idx]
            lora_B.weight.data = buffers[key_b][expert_idx]
            lora_A.weight.requires_grad_(True)
            lora_B.weight.requires_grad_(True)
            lora_A.weight.grad = grad_buffers["grad_" + key_a][expert_idx]
            lora_B.weight.grad = grad_buffers["grad_" + key_b][expert_idx]

            if not _first_logged:
                _new_id_a = id(lora_A.weight)
                _new_ptr_a = lora_A.weight.data_ptr()
                _buf_ptr_a = buffers[key_a][expert_idx].data_ptr()
                _has_grad = lora_A.weight.grad is not None
                logger.info(
                    "[_replace_peft_weights_with_views] first param: "
                    "id %s->%s (same=%s) data_ptr %s->%s buf_ptr=%s (match=%s) "
                    "has_grad=%s requires_grad=%s shape=%s",
                    _old_id_a, _new_id_a, _old_id_a == _new_id_a,
                    _old_ptr_a, _new_ptr_a, _buf_ptr_a, _new_ptr_a == _buf_ptr_a,
                    _has_grad, lora_A.weight.requires_grad, tuple(lora_A.weight.shape),
                )
                _first_logged = True
            _replaced += 1

    logger.info("[_replace_peft_weights_with_views] replaced %d param pairs", _replaced)


# =============================================================================
# Runtime LoRA Pointer Updates
# =============================================================================


def update_kt_lora_pointers(model: nn.Module):
    """Mark KT wrapper LoRA pointers as dirty after optimizer.step()."""
    wrappers = _find_kt_wrappers(model)

    if wrappers:
        for wrapper in wrappers:
            wrapper._lora_pointers_dirty = True


# =============================================================================
# Cross-Rank Gradient Synchronization
# =============================================================================


def sync_kt_lora_gradients(model: nn.Module) -> None:
    """
    Synchronize KT-managed LoRA gradients across ranks.

    KT computes expert LoRA gradients only on rank 0 (gather/scatter path). This function broadcasts the
    per-layer contiguous grad buffers from rank 0 to all ranks so that:
      - gradient clipping sees identical grads on every rank
      - optimizer.step() applies identical updates
    """
    import torch.distributed as dist

    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return

    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    params = get_kt_lora_params(model)
    if not params:
        return

    for param in params:
        if param.grad is not None:
            # Move grad to the same device as the parameter for all-reduce
            # Then move back to CPU
            original_device = param.grad.device
            if original_device.type == "cpu":
                # All-reduce on CPU might be slow; consider using a GPU buffer
                grad_gpu = param.grad.cuda()
                dist.all_reduce(grad_gpu, op=dist.ReduceOp.SUM)
                grad_gpu.div_(world_size)
                param.grad.copy_(grad_gpu.cpu())
            else:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world_size)


# =============================================================================
# Checkpoint Save/Load
# =============================================================================


def save_lora_experts_to_adapter(model: nn.Module, output_dir: str) -> None:
    """
    Save LoRA Experts weights to adapter file by merging with existing Attention LoRA.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    wrappers = _find_kt_wrappers(model) or []
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts saving")
        return

    adapter_file = os.path.join(output_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file_bin = os.path.join(output_dir, "adapter_model.bin")
        if os.path.exists(adapter_file_bin):
            state_dict = torch.load(adapter_file_bin, map_location="cpu", weights_only=True)
        else:
            logger.warning(f"No existing adapter file found at {output_dir}, creating new one")
            state_dict = {}
    else:
        state_dict = {}
        with safe_open(adapter_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    lora_expert_count = 0
    for wrapper in wrappers:
        if wrapper.lora_experts is None:
            continue

        layer_idx = wrapper.layer_idx
        for expert_idx, expert in enumerate(wrapper.lora_experts.experts):
            base_key = f"base_model.model.model.layers.{layer_idx}.mlp.lora_experts.{expert_idx}"
            state_dict[f"{base_key}.le_gate.weight"] = expert.le_gate.weight.data.cpu().clone()
            state_dict[f"{base_key}.le_up.weight"] = expert.le_up.weight.data.cpu().clone()
            state_dict[f"{base_key}.le_down.weight"] = expert.le_down.weight.data.cpu().clone()
            lora_expert_count += 3

        logger.debug(f"Added LoRA Experts for layer {layer_idx} ({len(wrapper.lora_experts.experts)} experts)")

    output_file = os.path.join(output_dir, "adapter_model.safetensors")
    save_file(state_dict, output_file, metadata={"format": "pt"})

    logger.info(
        f"Saved LoRA Experts to {output_file}: "
        f"{len(wrappers)} layers, {lora_expert_count} LoRA Expert tensors added, "
        f"{len(state_dict)} total tensors"
    )


def save_kt_moe_to_adapter(model: nn.Module, output_dir: str) -> None:
    """
    Unified function to save KT MoE weights to adapter file.
    Note: Per-expert PEFT LoRA is saved by PEFT directly, not here.
    This function only handles lora_experts (a separate feature).
    """
    wrappers = _find_kt_wrappers(model) or []
    if not wrappers:
        logger.info("[save_kt_moe] No KT wrappers found, skipping")
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    has_fused_lora = any(getattr(w, "_fused_expert_lora_params", None) is not None for w in wrappers)

    if has_lora_experts:
        save_lora_experts_to_adapter(model, output_dir)

    if has_fused_lora:
        _save_fused_expert_lora(wrappers, output_dir)

    if not has_lora_experts and not has_fused_lora:
        logger.info("[save_kt_moe] No lora_experts or fused expert LoRA in KT wrappers")


def _save_fused_expert_lora(wrappers: list, output_dir: str) -> None:
    """Save fused expert LoRA params to a safetensors file."""
    from safetensors.torch import save_file

    names = ["gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b"]
    tensors = {}
    for w in wrappers:
        fused = getattr(w, "_fused_expert_lora_params", None)
        if fused is None:
            continue
        for param, name in zip(fused, names):
            key = f"layers.{w.layer_idx}.experts.{name}"
            tensors[key] = param.data.clone()

    if tensors:
        path = os.path.join(output_dir, "fused_expert_lora.safetensors")
        save_file(tensors, path)
        logger.info(f"[save_kt_moe] Saved {len(tensors)} fused expert LoRA tensors to {path}")


def _load_fused_expert_lora(wrappers: list, adapter_path: str) -> None:
    """Load fused expert LoRA params from a safetensors file into existing wrapper buffers."""
    path = os.path.join(adapter_path, "fused_expert_lora.safetensors")
    if not os.path.isfile(path):
        logger.warning(f"No fused_expert_lora.safetensors found at {adapter_path}")
        return

    from safetensors.torch import load_file

    saved = load_file(path)
    names = ["gate_lora_a", "gate_lora_b", "up_lora_a", "up_lora_b", "down_lora_a", "down_lora_b"]
    wrapper_map = {w.layer_idx: w for w in wrappers}
    loaded_count = 0

    for key, tensor in saved.items():
        parts = key.split(".")
        if len(parts) != 4 or parts[0] != "layers" or parts[2] != "experts":
            logger.warning(f"Unexpected key in fused_expert_lora.safetensors: {key}")
            continue
        layer_idx = int(parts[1])
        name = parts[3]
        if name not in names:
            continue

        wrapper = wrapper_map.get(layer_idx)
        if wrapper is None:
            continue
        fused = getattr(wrapper, "_fused_expert_lora_params", None)
        if fused is None:
            continue

        param_idx = names.index(name)
        fused[param_idx].data.copy_(tensor)
        loaded_count += 1

    logger.info(f"[_load_fused_expert_lora] Loaded {loaded_count} tensors from {path}")


def load_lora_experts_from_adapter(model: nn.Module, adapter_path: str) -> None:
    """
    Load LoRA Experts weights from adapter file into KT wrappers.
    """
    from safetensors import safe_open

    wrappers = _find_kt_wrappers(model) or []
    if not wrappers:
        logger.warning("No KT wrappers found, skipping LoRA Experts loading")
        return

    wrapper_map = {w.layer_idx: w for w in wrappers if w.lora_experts is not None}
    if not wrapper_map:
        logger.warning("No LoRA Experts found in KT wrappers, skipping")
        return

    # Prefer dedicated lora_experts file, fallback to adapter file
    adapter_file = os.path.join(adapter_path, "lora_experts.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            adapter_file = os.path.join(adapter_path, "adapter_model.bin")
            if not os.path.exists(adapter_file):
                logger.warning(f"No lora_experts or adapter file found at {adapter_path}")
                return

    logger.info(f"Loading LoRA Experts from {adapter_file}")

    lora_expert_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.mlp\.lora_experts\.(\d+)\.(le_gate|le_up|le_down)\.weight"
    )

    layer_weights = {}
    with safe_open(adapter_file, framework="pt") as f:
        for key in f.keys():
            match = lora_expert_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_name = match.group(3)
                layer_weights.setdefault(layer_idx, {}).setdefault(expert_idx, {})[proj_name] = f.get_tensor(key)

    loaded_count = 0
    for layer_idx, experts_dict in layer_weights.items():
        if layer_idx not in wrapper_map:
            logger.warning(f"No LoRA Experts for layer {layer_idx}, skipping")
            continue

        wrapper = wrapper_map[layer_idx]
        for expert_idx, proj_dict in experts_dict.items():
            if expert_idx >= len(wrapper.lora_experts.experts):
                continue
            expert = wrapper.lora_experts.experts[expert_idx]
            if "le_gate" in proj_dict:
                expert.le_gate.weight.data.copy_(proj_dict["le_gate"].to(expert.le_gate.weight.device))
            if "le_up" in proj_dict:
                expert.le_up.weight.data.copy_(proj_dict["le_up"].to(expert.le_up.weight.device))
            if "le_down" in proj_dict:
                expert.le_down.weight.data.copy_(proj_dict["le_down"].to(expert.le_down.weight.device))
            loaded_count += 1

    logger.info(f"Loaded LoRA Experts for {loaded_count} experts from {adapter_path}")


def load_kt_moe_from_adapter(model: nn.Module, adapter_path: str) -> None:
    """
    Unified function to load KT MoE weights from adapter file.
    Note: Per-expert PEFT LoRA is loaded by PEFT directly, not here.
    This function only handles lora_experts (a separate feature).
    """
    wrappers = _find_kt_wrappers(model) or []
    if not wrappers:
        logger.warning("No KT wrappers found, skipping KT MoE loading")
        return

    has_lora_experts = any(w.lora_experts is not None for w in wrappers)
    has_fused_lora = any(getattr(w, "_fused_expert_lora_params", None) is not None for w in wrappers)

    if has_lora_experts:
        load_lora_experts_from_adapter(model, adapter_path)

    if has_fused_lora:
        _load_fused_expert_lora(wrappers, adapter_path)

    if not has_lora_experts and not has_fused_lora:
        logger.info("No lora_experts or fused expert LoRA in KT wrappers")
