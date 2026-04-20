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
        setattr(self, router_attr, getattr(original_moe, router_attr, None))
        self._router_attr = router_attr

        # 2. experts SECOND (this is what PEFT targets for LoRA)
        experts_attr = moe_config.experts_attr  # typically "experts"
        setattr(self, experts_attr, getattr(original_moe, experts_attr, None))
        self._experts_attr = experts_attr

        # 3. shared_experts (if any)
        if moe_config.has_shared_experts and hasattr(original_moe, "shared_experts"):
            self.shared_experts = original_moe.shared_experts
        else:
            self.shared_experts = None

        # 4. lora_experts (separate LoRA expert MLPs, different from PEFT LoRA on experts)
        self.lora_experts = lora_experts

        # PEFT LoRA tracking (set by kt_adapt_peft_lora)
        # _peft_lora_modules: {expert_idx: {proj_name: (lora_A, lora_B)}}
        self._peft_lora_modules: dict[int, dict[str, tuple[nn.Module, nn.Module]]] | None = None
        self._lora_pointers_dirty = False

    def _apply(self, fn, recurse=True):
        # Protect experts from device transfer (PEFT LoRA should stay on CPU for KT)
        saved_experts = None
        experts_attr = getattr(self, '_experts_attr', None)

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

        topk_ids, topk_weights = self._compute_routing(hidden_states)

        train_lora = self._peft_lora_modules is not None and len(self._peft_lora_modules) > 0

        save_for_backward = (
            self.training
            and torch.is_grad_enabled()
            and (hidden_states.requires_grad or topk_weights.requires_grad or train_lora)
        )
        use_autograd_path = save_for_backward
        save_for_backward_submit = use_autograd_path
        if _checkpoint_hook_mode() == "first_forward":
            save_for_backward_submit = False

        if train_lora and self._lora_pointers_dirty:
            self.update_lora_pointers()
            self._lora_pointers_dirty = False

        gpu_output, all_qlens = self._submit_and_compute_gpu(
            hidden_states,
            topk_ids,
            topk_weights,
            save_for_backward_submit,
        )

        # Use KTMoEFunction whenever backward is needed so KT backward and LoRA
        # gradient paths remain connected.
        if use_autograd_path:
            lora_ref = hidden_states.new_empty(())
            if train_lora and self._peft_lora_modules:
                for expert_loras in self._peft_lora_modules.values():
                    for lora_A, lora_B in expert_loras.values():
                        if hasattr(lora_A, 'weight') and lora_A.weight.requires_grad:
                            lora_ref = lora_A.weight
                            break
                    if lora_ref.numel() > 0:
                        break

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
                    raise RuntimeError(
                        f"all_qlens length mismatch: got {len(all_qlens_list)}, expected {world_size}"
                    )
            if int(all_qlens_list[rank]) != qlen:
                raise RuntimeError(
                    f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens_list[rank]}"
                )
            total_qlen = sum(all_qlens_list)

            if rank == 0:
                if self.wrapper is None:
                    raise RuntimeError("Rank0 wrapper is required in distributed KT overlap path.")
                cpu_output = self.wrapper.sync_forward_sft(output_device=original_device)
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
            cpu_output = self.wrapper.sync_forward_sft(output_device=original_device)
            output = cpu_output.view(batch_size, seq_len, self.hidden_size).to(dtype=original_dtype)
            return output

        return torch.empty(batch_size, seq_len, self.hidden_size, device=original_device, dtype=original_dtype)

    def _compute_routing(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Run routing under no_grad to avoid creating autograd nodes whose
        # SavedVariables become orphan holders inside gradient checkpoint.
        # The gate is frozen during LoRA fine-tuning and the main gradient
        # flows through KTMoEFunction.backward()'s grad_input, so the
        # routing gradient contribution to hidden_states can be safely dropped.
        with torch.no_grad():
            router = getattr(self, self._router_attr)
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
                if topk_weights.is_floating_point():
                    topk_weights = topk_weights.to(torch.bfloat16)
                return topk_ids, topk_weights

            router_output = router(hidden_states.view(-1, self.hidden_size))
            # transformers v5 TopKRouter returns (router_logits, router_scores, router_indices)
            # directly — scores/indices are already topk-normalized.
            if isinstance(router_output, tuple):
                if len(router_output) >= 3:
                    _logits, topk_weights, topk_ids = router_output[0], router_output[1], router_output[2]
                    if topk_weights.is_floating_point():
                        topk_weights = topk_weights.to(torch.bfloat16)
                    return topk_ids, topk_weights
                router_output = router_output[0]

            router_logits = router_output
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(routing_weights, self.moe_config.num_experts_per_tok, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(torch.bfloat16)
            return topk_ids, topk_weights

    def _submit_and_compute_gpu(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        save_for_backward: bool,
    ) -> tuple[torch.Tensor | None, list[int] | None]:
        import torch.distributed as dist

        batch_size, seq_len, _ = hidden_states.shape
        original_device = hidden_states.device
        original_dtype = hidden_states.dtype

        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        qlen = batch_size * seq_len

        if dist_on:
            all_qlens = _all_gather_qlens(qlen, original_device, world_size)
            if int(all_qlens[rank]) != qlen:
                raise RuntimeError(
                    f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens[rank]}"
                )
            total_qlen = sum(all_qlens)

            hs_flat = hidden_states.view(qlen, self.hidden_size).contiguous()
            expert_ids = topk_ids.view(qlen, self.moe_config.num_experts_per_tok).contiguous()
            weights = topk_weights.view(qlen, self.moe_config.num_experts_per_tok).contiguous()

            submit_hs = hs_flat.detach()
            submit_ids = expert_ids.detach()
            submit_wts = weights.detach()

            gathered_hs = _dist_gather_varlen_to_rank0(
                submit_hs,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            gathered_ids = _dist_gather_varlen_to_rank0(
                submit_ids,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            gathered_wts = _dist_gather_varlen_to_rank0(
                submit_wts,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )

            if rank == 0:
                all_hs = torch.cat(gathered_hs, dim=0)
                all_ids = torch.cat(gathered_ids, dim=0)
                all_wts = torch.cat(gathered_wts, dim=0)
                self.wrapper.submit_forward_sft(
                    all_hs,
                    all_ids,
                    all_wts,
                    save_for_backward=save_for_backward,
                )

            # Keep shared/lora experts local to avoid qlen_max-style amplification.
            gpu_output = None
            if self.shared_experts is not None:
                gpu_output = self.shared_experts(hidden_states)
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
            self.wrapper.submit_forward_sft(
                submit_hs,
                submit_ids,
                submit_wts,
                save_for_backward=save_for_backward,
            )

            # GPU compute: shared_experts + lora_experts
            gpu_output = None
            if self.shared_experts is not None:
                gpu_output = self.shared_experts(hidden_states)
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
