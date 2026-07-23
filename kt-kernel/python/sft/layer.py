# KTMoELayerWrapper — nn.Module replacing HF MoE layers for SFT
# SPDX-License-Identifier: Apache-2.0

"""
KTMoELayerWrapper: drop-in nn.Module replacement for HuggingFace MoE layers.

Delegates expert computation to the C++ KTMoEWrapper backend, with support
for gradient checkpointing, PEFT LoRA on experts, LoRA Experts (separate
small MLPs on GPU), shared experts, and multi-GPU rank-0-only execution.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .arch import MOEArchConfig
from .autograd import KTMoEFunction
from .dist_utils import (
    _all_gather_qlens,
    _checkpoint_hook_mode,
    _dist_gather_varlen_to_rank0,
    _dist_scatter_varlen_from_rank0,
    _qlen_offsets,
)

logger = logging.getLogger(__name__)
_KT_SFT_DEBUG = os.environ.get("KT_SFT_DEBUG", "0") == "1"


def _strip_kt_zero_storage_from_state_dict(module, state_dict, prefix, local_metadata) -> None:
    """Never serialize expert placeholders that do not contain real weights."""
    del local_metadata
    for name, param in module.named_parameters():
        if getattr(param, "_kt_zero_storage", False):
            state_dict.pop(f"{prefix}{name}", None)


def _supply_kt_zero_storage_for_state_dict_load(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
) -> None:
    """Keep placeholder keys out of checkpoints without reporting them missing."""
    del local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for name, param in module.named_parameters():
        if getattr(param, "_kt_zero_storage", False):
            state_dict[f"{prefix}{name}"] = param


class KTMoELayerWrapper(nn.Module):
    """Wrapper for MoE layer using KTMoEWrapper."""

    def __init__(
        self,
        original_moe: nn.Module,
        wrapper: Any,
        lora_params: dict[str, nn.Parameter] | None,  # Kept for backward compatibility, but ignored
        moe_config: MOEArchConfig,
        hidden_size: int,
        layer_idx: int,
        lora_experts: "LoRAExperts | None" = None,
        full_weight_grad: bool | None = None,
        uses_authoritative_optimizer_grads: bool | None = None,
    ):
        super().__init__()
        self._is_kt_moe_wrapper = True

        self.wrapper = wrapper
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.router_type = moe_config.router_type

        # IMPORTANT: Register submodules in the SAME ORDER as original MoE module
        # so that PEFT's named_modules() traversal order matches baseline.
        # This ensures kaiming_uniform_ calls happen in the same sequence.
        # Qwen3MoeSparseMoeBlock order: gate FIRST, then experts.

        # 1. gate/router FIRST - keep original attribute name for PEFT compatibility
        router_attr = moe_config.router_attr  # "gate" for Qwen3/DeepSeek
        original_router = getattr(original_moe, router_attr, None)
        self._original_router = None  # Set when router is not nn.Linear (e.g. TopKRouter)

        if original_router is not None and isinstance(original_router, nn.Linear):
            # transformers <=4.x / some models: gate is nn.Linear - register directly.
            setattr(self, router_attr, original_router)
        elif original_router is not None and hasattr(original_router, "weight") and isinstance(
            getattr(original_router, "weight"), nn.Parameter
        ):
            # transformers v5+: gate is a TopKRouter with nn.Parameter weight.
            # Wrap it in nn.Linear so PEFT can discover and inject LoRA.
            # The nn.Linear shares the same weight tensor - LoRA applied to it
            # is equivalent to LoRA on the original gate.
            router_weight = original_router.weight
            router_linear = nn.Linear(
                router_weight.shape[1], router_weight.shape[0], bias=False,
            )
            router_linear.weight = router_weight  # share the same parameter
            setattr(self, router_attr, router_linear)
            # Keep the original router for forward (top-k selection logic)
            self._original_router = original_router
        else:
            setattr(self, router_attr, original_router)
        self._router_attr = router_attr

        # 2. experts SECOND (this is what PEFT targets for LoRA)
        experts_attr = moe_config.experts_attr  # typically "experts"
        setattr(self, experts_attr, getattr(original_moe, experts_attr, None))
        self._experts_attr = experts_attr

        # 3. Shared expert (if any). DeepSeek/GLM use ``shared_experts`` while
        # Qwen2-MoE/Qwen3.5 use ``shared_expert`` plus a sigmoid gate. Preserve
        # the original attribute names so state_dict/checkpoint keys stay
        # compatible with the Hugging Face model.
        self._shared_expert_attr: str | None = None
        self._shared_expert_gate_attr: str | None = None
        if moe_config.has_shared_experts:
            for shared_expert_attr in ("shared_experts", "shared_expert"):
                shared_expert = getattr(original_moe, shared_expert_attr, None)
                if shared_expert is not None:
                    setattr(self, shared_expert_attr, shared_expert)
                    self._shared_expert_attr = shared_expert_attr
                    break

            if self._shared_expert_attr is None:
                raise ValueError(
                    f"Layer {layer_idx}: architecture declares shared experts, "
                    "but the MoE module has neither 'shared_experts' nor 'shared_expert'"
                )

            gate_candidates = (
                ("shared_expert_gate", "shared_experts_gate")
                if self._shared_expert_attr == "shared_expert"
                else ("shared_experts_gate", "shared_expert_gate")
            )
            for shared_expert_gate_attr in gate_candidates:
                shared_expert_gate = getattr(original_moe, shared_expert_gate_attr, None)
                if shared_expert_gate is not None:
                    setattr(self, shared_expert_gate_attr, shared_expert_gate)
                    self._shared_expert_gate_attr = shared_expert_gate_attr
                    break
            if self._shared_expert_attr == "shared_expert" and self._shared_expert_gate_attr is None:
                raise ValueError(
                    f"Layer {layer_idx}: singular 'shared_expert' requires "
                    "'shared_expert_gate' for Qwen-style gated output"
                )

        # 4. lora_experts (separate LoRA expert MLPs, different from PEFT LoRA on experts)
        self.lora_experts = lora_experts

        # PEFT LoRA tracking (set by kt_adapt_peft_lora)
        # _peft_lora_modules: {expert_idx: {proj_name: (lora_A, lora_B)}}
        self._peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]] | None = None
        self._lora_pointers_dirty = False
        self._kt_managed_lora_enabled = False

        # Training-mode flags must be identical on every distributed rank even
        # though only rank 0 owns the backend object.
        if full_weight_grad is None:
            full_weight_grad = getattr(wrapper, "_full_weight_grad", False) if wrapper is not None else False
        if uses_authoritative_optimizer_grads is None:
            uses_authoritative_optimizer_grads = bool(
                wrapper is not None and getattr(wrapper, "_uses_authoritative_optimizer_grads", False)
            )
        self._full_weight_grad = bool(full_weight_grad)
        self._uses_authoritative_optimizer_grads = bool(uses_authoritative_optimizer_grads)
        self.register_state_dict_post_hook(_strip_kt_zero_storage_from_state_dict)
        self.register_load_state_dict_pre_hook(_supply_kt_zero_storage_for_state_dict_load)

    def _compute_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self._shared_expert_attr is None:
            return None

        shared_expert = getattr(self, self._shared_expert_attr)
        shared_output = shared_expert(hidden_states)
        if self._shared_expert_gate_attr is not None:
            shared_expert_gate = getattr(self, self._shared_expert_gate_attr)
            shared_output = torch.sigmoid(shared_expert_gate(hidden_states)) * shared_output
        return shared_output

    def _apply(self, fn, recurse=True):
        # Protect experts from device transfer (PEFT LoRA should stay on CPU for KT)
        saved_experts = None
        experts_attr = getattr(self, "_experts_attr", None)

        if experts_attr is not None and getattr(self, experts_attr, None) is not None:
            saved_experts = getattr(self, experts_attr)
            self._modules.pop(experts_attr, None)

        result = super()._apply(fn, recurse)

        if saved_experts is not None:
            self._modules[experts_attr] = saved_experts

        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        import torch.distributed as dist

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Check if we need to use distributed broadcast (only rank 0 has KT kernel)
        use_broadcast = dist_on and self.wrapper is None

        with torch.profiler.record_function("kt.sft.routing"):
            topk_ids, topk_weights = self._compute_routing(hidden_states)

        train_lora = bool(
            self._kt_managed_lora_enabled
            or (self._peft_lora_modules is not None and len(self._peft_lora_modules) > 0)
            or getattr(self, "_fused_expert_lora_params", None)
        )
        full_weight_grad = self._full_weight_grad

        save_for_backward = (
            self.training
            and torch.is_grad_enabled()
            and (hidden_states.requires_grad or topk_weights.requires_grad or train_lora or full_weight_grad)
        )
        use_autograd_path = save_for_backward
        checkpoint_mode = _checkpoint_hook_mode()
        reuse_checkpoint_forward = (
            not dist_on
            and self.wrapper is not None
            and getattr(self.wrapper, "reuse_checkpoint_forward", False)
        )
        reuse_cached_forward = (
            reuse_checkpoint_forward
            and checkpoint_mode == "recompute"
            and getattr(self.wrapper, "_kt_has_cached_forward", False)
        )
        cache_checkpoint_forward = reuse_checkpoint_forward and checkpoint_mode == "first_forward"
        save_for_backward_submit = use_autograd_path
        if checkpoint_mode == "first_forward" and not cache_checkpoint_forward:
            save_for_backward_submit = False

        if train_lora and self._lora_pointers_dirty:
            self.update_lora_pointers()
            self._lora_pointers_dirty = False

        # In full_weight_grad mode, sync base weights after optimizer step
        if full_weight_grad and getattr(self.wrapper, "_kt_full_checkpoint_load_failed", False):
            raise RuntimeError(
                f"Layer {self.layer_idx}: a previous KT Full checkpoint load failed; "
                "reload a valid checkpoint before running forward"
            )
        if full_weight_grad and getattr(self.wrapper, "_base_weights_dirty", False):
            with torch.profiler.record_function("kt.sft.base_weight_reload"):
                self.wrapper.update_base_weights()
            self.wrapper._base_weights_dirty = False

        with torch.profiler.record_function("kt.sft.submit_and_gpu_experts"):
            gpu_output, all_qlens = self._submit_and_compute_gpu(
                hidden_states,
                topk_ids,
                topk_weights,
                save_for_backward_submit,
                reuse_cached_forward,
            )

        # Use KTMoEFunction whenever backward is needed so KT backward and LoRA
        # gradient paths remain connected.
        if use_autograd_path:
            # A requires-grad sentinel keeps the custom autograd node alive on
            # non-rank-0 fused/full ranks that intentionally own no KT params.
            lora_ref = hidden_states.new_empty((), requires_grad=(train_lora or full_weight_grad))
            if train_lora and self._peft_lora_modules:
                found_lora_ref = False
                for expert_loras in self._peft_lora_modules.values():
                    for lora_A, lora_B in expert_loras.values():
                        if hasattr(lora_A, "weight") and lora_A.weight.requires_grad:
                            lora_ref = lora_A.weight
                            found_lora_ref = True
                            break
                    if found_lora_ref:
                        break
            elif train_lora and getattr(self, "_fused_expert_lora_params", None):
                lora_ref = self._fused_expert_lora_params[0]
            elif full_weight_grad and self.wrapper is not None:
                # In full mode, use base weight param as autograd sentinel
                if self.wrapper.gate_proj_buf is not None:
                    lora_ref = self.wrapper.gate_proj_buf

            with torch.profiler.record_function("kt.sft.autograd_apply_and_cpu_sync"):
                moe_output = KTMoEFunction.apply(
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    self.wrapper,
                    lora_ref,
                    self.hidden_size,
                    self.moe_config.num_experts_per_tok,
                    self.layer_idx,
                    save_for_backward,
                    train_lora,
                    all_qlens,
                    cache_checkpoint_forward,
                    reuse_cached_forward,
                    # Base weight params for full mode gradient flow
                    self.wrapper.gate_proj_buf if full_weight_grad and self.wrapper is not None else None,
                    self.wrapper.up_proj_buf if full_weight_grad and self.wrapper is not None else None,
                    self.wrapper.down_proj_buf if full_weight_grad and self.wrapper is not None else None,
                )
        else:
            moe_output = self._sync_forward_output_no_autograd(
                hidden_states=hidden_states,
                all_qlens=all_qlens,
            )

        if gpu_output is not None:
            moe_output = moe_output + gpu_output

        return moe_output

    def _sync_forward_output_no_autograd(
        self,
        hidden_states: torch.Tensor,
        all_qlens: list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:
        """Sync CPU expert output without creating KTMoEFunction autograd nodes."""
        import torch.distributed as dist

        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        qlen = batch_size * seq_len

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        if dist_on:
            if all_qlens is None:
                all_qlens_list = _all_gather_qlens(qlen, original_device, world_size)
            else:
                all_qlens_list = [int(q) for q in all_qlens]
                if len(all_qlens_list) != world_size:
                    raise RuntimeError(f"all_qlens length mismatch: got {len(all_qlens_list)}, expected {world_size}")
            if int(all_qlens_list[rank]) != qlen:
                raise RuntimeError(f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens_list[rank]}")
            total_qlen = sum(all_qlens_list)

            if rank == 0:
                if self.wrapper is None:
                    raise RuntimeError("Rank0 wrapper is required in distributed KT overlap path.")
                cpu_output = self.wrapper.sync_forward(output_device=original_device)
                cpu_output = cpu_output.to(dtype=original_dtype).view(total_qlen, self.hidden_size)
                offsets = _qlen_offsets(all_qlens_list)
                scatter_list = [cpu_output[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
            else:
                scatter_list = None

            output_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_list,
                all_qlens=all_qlens_list,
                rank=rank,
                world_size=world_size,
                feature_shape=(self.hidden_size,),
                device=original_device,
                dtype=original_dtype,
            )
            output = output_flat.view(batch_size, seq_len, self.hidden_size)
            del output_flat
            return output

        if self.wrapper is not None:
            cpu_output = self.wrapper.sync_forward(output_device=original_device)
            output = cpu_output.view(batch_size, seq_len, self.hidden_size).to(dtype=original_dtype)
            return output

        return torch.empty(batch_size, seq_len, self.hidden_size, device=original_device, dtype=original_dtype)

    def _compute_routing(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router = getattr(self, self._router_attr)
        router_grad_enabled = self.training and torch.is_grad_enabled() and any(
            parameter.requires_grad for parameter in router.parameters()
        )
        routing_context = nullcontext() if router_grad_enabled else torch.no_grad()

        def finish(
            topk_ids: torch.Tensor,
            topk_weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if topk_weights.is_floating_point():
                topk_weights = topk_weights.to(torch.bfloat16)
            if router_grad_enabled and not topk_weights.requires_grad:
                raise RuntimeError(
                    f"Layer {self.layer_idx}: trainable router produced detached routing weights"
                )
            return topk_ids, topk_weights

        with routing_context:
            if self.router_type == "deepseek_gate":
                # DeepSeek V3's MoEGate has `assert not self.training` in its noaux_tc
                # routing path because the HF model is an inference-only port.
                # For LoRA fine-tuning the router is frozen, so eval() is safe.
                was_training = router.training
                if was_training:
                    router.eval()
                router_output = router(hidden_states)
                if was_training:
                    router.train()
                if len(router_output) == 2:
                    topk_ids, topk_weights = router_output
                else:
                    topk_ids, topk_weights = router_output[0], router_output[1]
                return finish(topk_ids, topk_weights)

            # When _original_router is set, self.gate is an nn.Linear wrapper
            # around the TopKRouter's weight.  Use it (with PEFT LoRA if
            # applied) for the linear projection, then replicate top-k logic.
            if self._original_router is not None:
                orig_router = self._original_router
                router_logits = router(hidden_states.view(-1, self.hidden_size))
                if self.router_type == "glm4_moe_gate":
                    router_probs = torch.sigmoid(router_logits.float())
                    correction_bias = getattr(orig_router, "e_score_correction_bias", None)
                    if correction_bias is None:
                        router_logits_for_choice = router_probs
                    else:
                        router_logits_for_choice = router_probs + correction_bias.to(
                            device=router_probs.device,
                            dtype=router_probs.dtype,
                        )
                    n_group = getattr(orig_router, "n_group", 1)
                    topk_group = getattr(orig_router, "topk_group", n_group)
                    expert_num = self.moe_config.expert_num
                    group_scores = (
                        router_logits_for_choice.view(-1, n_group, expert_num // n_group)
                        .topk(2, dim=-1)[0]
                        .sum(dim=-1)
                    )
                    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
                    group_mask = torch.zeros_like(group_scores)
                    group_mask.scatter_(1, group_idx, 1)
                    score_mask = (
                        group_mask.unsqueeze(-1)
                        .expand(-1, n_group, expert_num // n_group)
                        .reshape(-1, expert_num)
                    )
                    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
                    top_k = getattr(orig_router, "top_k", self.moe_config.num_experts_per_tok)
                    topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
                    topk_weights = router_probs.gather(1, topk_ids)
                    if getattr(orig_router, "norm_topk_prob", True):
                        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
                    topk_weights = topk_weights * getattr(orig_router, "routed_scaling_factor", 1.0)
                    return finish(topk_ids, topk_weights)

                router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
                top_k = getattr(orig_router, "top_k", self.moe_config.num_experts_per_tok)
                norm_topk_prob = getattr(orig_router, "norm_topk_prob", True)
                topk_weights, topk_ids = torch.topk(router_probs, top_k, dim=-1)
                if norm_topk_prob:
                    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
                topk_weights = topk_weights.to(router_logits.dtype)
                return finish(topk_ids, topk_weights)

            router_output = router(hidden_states.view(-1, self.hidden_size))
            # transformers v5 TopKRouter returns (router_logits, router_scores, router_indices)
            # directly — scores/indices are already topk-normalized.
            if isinstance(router_output, tuple):
                if len(router_output) >= 3:
                    _logits, topk_weights, topk_ids = router_output[0], router_output[1], router_output[2]
                    return finish(topk_ids, topk_weights)
                router_output = router_output[0]

            router_logits = router_output
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(routing_weights, self.moe_config.num_experts_per_tok, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            return finish(topk_ids, topk_weights)

    def _submit_and_compute_gpu(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        save_for_backward: bool,
        reuse_cached_forward: bool = False,
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        import torch.distributed as dist

        batch_size, seq_len, _ = hidden_states.shape
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        if dist_on and self._uses_authoritative_optimizer_grads:
            wrapped_world_size = int(getattr(self, "_kt_world_size_at_wrap", world_size))
            if wrapped_world_size != world_size:
                raise RuntimeError(
                    f"Layer {self.layer_idx}: KT wrapper was created for world_size={wrapped_world_size}, "
                    f"but the active process group has world_size={world_size}"
                )
            if rank == 0 and self.wrapper is None:
                raise RuntimeError(f"Layer {self.layer_idx}: rank 0 does not own the authoritative KT backend")
            if rank != 0 and self.wrapper is not None:
                raise RuntimeError(
                    f"Layer {self.layer_idx}: rank {rank} unexpectedly owns an authoritative KT backend"
                )

        qlen = batch_size * seq_len

        if dist_on:
            all_qlens = _all_gather_qlens(qlen, original_device, world_size)
            if int(all_qlens[rank]) != qlen:
                raise RuntimeError(f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens[rank]}")
            total_qlen = sum(all_qlens)

            if not reuse_cached_forward:
                hs_flat = hidden_states.view(qlen, self.hidden_size).contiguous()
                expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok).contiguous()
                weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok).contiguous()

                gathered_hs = _dist_gather_varlen_to_rank0(
                    hs_flat.detach(),
                    all_qlens=all_qlens,
                    rank=rank,
                    world_size=world_size,
                )
                gathered_ids = _dist_gather_varlen_to_rank0(
                    expert_ids.detach(),
                    all_qlens=all_qlens,
                    rank=rank,
                    world_size=world_size,
                )
                gathered_wts = _dist_gather_varlen_to_rank0(
                    weights.detach(),
                    all_qlens=all_qlens,
                    rank=rank,
                    world_size=world_size,
                )

                if rank == 0:
                    self.wrapper.submit_forward(
                        torch.cat(gathered_hs, dim=0),
                        torch.cat(gathered_ids, dim=0),
                        torch.cat(gathered_wts, dim=0),
                        save_for_backward=save_for_backward,
                    )

            # Keep shared/lora experts local to avoid qlen_max-style amplification.
            gpu_output = self._compute_shared_expert(hidden_states)
            if gpu_output is not None:
                gpu_output = gpu_output.to(dtype=original_dtype)

            if self.lora_experts is not None:
                lora_out = self.lora_experts(hidden_states)
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out

            return gpu_output, all_qlens

        else:
            # ---- Single-GPU path: submit + GPU compute ----
            input_flat = hidden_states.view(qlen, self.hidden_size)
            expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok)
            weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok)

            # Avoid passing graph-attached tensors into C++ cache.
            submit_hs = input_flat.detach()
            submit_ids = expert_ids.detach()
            submit_wts = weights.detach()
            if not reuse_cached_forward:
                self.wrapper.submit_forward(
                    submit_hs,
                    submit_ids,
                    submit_wts,
                    save_for_backward=save_for_backward,
                )

            # GPU compute: shared_experts + lora_experts
            gpu_output = self._compute_shared_expert(hidden_states)
            if self.lora_experts is not None:
                lora_out = self.lora_experts(hidden_states)
                gpu_output = lora_out if gpu_output is None else gpu_output + lora_out

            return gpu_output, None

    def update_lora_pointers(self):
        """Sync PEFT LoRA weights to C++ kernel after optimizer update."""
        # Skip if wrapper is None (non-rank-0 processes)
        if self.wrapper is None:
            return
        # Skip if wrapper is not properly initialized
        if not getattr(self.wrapper, "_weights_loaded", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - weights not loaded")
            return
        if not getattr(self.wrapper, "_lora_initialized", False):
            logger.warning(f"Layer {self.layer_idx}: Skipping update_lora_pointers - LoRA not initialized")
            return

        # PEFT weights are views into wrapper's contiguous buffers —
        # optimizer.step() already updated them in-place, just re-sync to C++.
        self.wrapper.update_lora_weights()
