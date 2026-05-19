# Autograd function for KT MoE SFT training
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .dist_utils import (
    _all_gather_qlens,
    _qlen_offsets,
    _dist_gather_varlen_to_rank0,
    _dist_scatter_varlen_from_rank0,
)

_KT_SFT_DEBUG = os.environ.get("KT_SFT_DEBUG", "0") == "1"

logger = logging.getLogger(__name__)


class KTMoEFunction(torch.autograd.Function):
    """Unified autograd function for KTMoE forward/backward."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        wrapper: Any,
        lora_ref: torch.Tensor,
        hidden_size: int,
        num_experts_per_tok: int,
        layer_idx: int,
        training: bool,
        train_lora: bool,
        all_qlens: list[int] | tuple[int, ...] | None,
    ) -> torch.Tensor:

        if _KT_SFT_DEBUG:
            logging.debug(
                "KTMoEFunction.forward: layer=%d training=%s train_lora=%s",
                layer_idx, training, train_lora,
            )

        original_device = hidden_states.device
        original_dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        qlen = batch_size * seq_len

        import torch.distributed as dist
        dist_on = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist_on else 1

        ctx.use_broadcast = wrapper is None

        # ---- Sync CPU expert result and distribute ----
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

            # Rank 0: sync CPU result and split by real lengths
            if rank == 0:
                cpu_output = wrapper.sync_forward(output_device=original_device)
                cpu_output = cpu_output.to(dtype=original_dtype).view(total_qlen, hidden_size)
                offsets = _qlen_offsets(all_qlens_list)
                scatter_list = [cpu_output[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
            else:
                scatter_list = None

            output_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_list,
                all_qlens=all_qlens_list,
                rank=rank,
                world_size=world_size,
                feature_shape=(hidden_size,),
                device=original_device,
                dtype=original_dtype,
            )
            output = output_flat.view(batch_size, seq_len, hidden_size)
            del output_flat
        elif wrapper is not None:
            # Single-GPU: sync directly
            cpu_output = wrapper.sync_forward(output_device=original_device)
            output = cpu_output.view(batch_size, seq_len, hidden_size).to(dtype=original_dtype)
        else:
            # Broadcast-only rank (no wrapper)
            output = torch.empty(
                batch_size, seq_len, hidden_size, device=original_device, dtype=original_dtype
            )

        ctx.wrapper = wrapper
        ctx.hidden_size = hidden_size
        ctx.qlen = qlen
        ctx.batch_size = batch_size
        ctx.seq_len = seq_len
        ctx.original_device = original_device
        ctx.original_dtype = original_dtype
        ctx.weights_shape = topk_weights.shape
        ctx.weights_dtype = topk_weights.dtype
        ctx.weights_device = topk_weights.device
        ctx.dist_on = dist_on
        ctx.world_size = world_size
        ctx.all_qlens = all_qlens_list if dist_on else None
        ctx.num_experts_per_tok = num_experts_per_tok
        ctx.layer_idx = layer_idx

        # Save a sentinel tensor so non-reentrant checkpoint's saved_tensors
        # hooks can intercept it.  When backward accesses ctx.saved_tensors,
        # the checkpoint unpack hook triggers a full recompute of the decoder
        # layer — which re-runs the MoE forward with save_for_backward=True,
        # populating the C++ cache BEFORE this backward proceeds.
        # Without this, MoE backward runs before the recompute (MoE comes
        # after attention in forward order → its backward runs first), and
        # the C++ cache is empty when first-forward cache-skip is active.
        ctx.save_for_backward(hidden_states.new_empty(()))

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Wait for any in-flight async repack before recompute forward uses the pool
        if getattr(ctx.wrapper, 'share_backward_bb', False):
            ctx.wrapper.wait_backward_repack()

        # Access saved_tensors FIRST — under non-reentrant checkpoint this
        # triggers the unpack hook which runs a full decoder-layer recompute,
        # populating the C++ cache before we call wrapper.backward().
        _ = ctx.saved_tensors

        qlen = ctx.qlen
        hidden_size = ctx.hidden_size
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        dist_on = ctx.dist_on
        world_size = ctx.world_size
        num_experts_per_tok = ctx.num_experts_per_tok

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        if _KT_SFT_DEBUG:
            logging.debug(
                "KTMoEFunction.backward: layer=%d dist_on=%s qlen=%d",
                getattr(ctx, "layer_idx", -1), dist_on, qlen,
            )

        if dist_on:
            all_qlens = getattr(ctx, "all_qlens", None)
            if all_qlens is None or len(all_qlens) != world_size:
                all_qlens = _all_gather_qlens(qlen, ctx.original_device, world_size)
            else:
                all_qlens = [int(q) for q in all_qlens]
            if int(all_qlens[rank]) != qlen:
                raise RuntimeError(
                    f"Backward qlen mismatch on rank {rank}: local={qlen}, all_qlens[{rank}]={all_qlens[rank]}"
                )

            grad_out_flat = grad_output.view(qlen, hidden_size).contiguous()

            gathered_go = _dist_gather_varlen_to_rank0(
                grad_out_flat,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                all_go = torch.cat(gathered_go, dim=0)
                total_qlen = int(all_go.shape[0])

                backward_out = ctx.wrapper.backward(
                    all_go,
                    output_device=ctx.original_device,
                )
                if isinstance(backward_out, tuple) and len(backward_out) == 2:
                    all_grad_input, all_grad_weights = backward_out
                elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                    all_grad_input, _, all_grad_weights = backward_out
                else:
                    raise ValueError("KTMoEWrapper.backward returned unexpected format.")

                all_grad_input = all_grad_input.to(dtype=ctx.original_dtype).view(total_qlen, hidden_size)
                all_grad_weights = all_grad_weights.to(dtype=torch.bfloat16).view(total_qlen, num_experts_per_tok)

                offsets = _qlen_offsets(all_qlens)
                scatter_gi = [all_grad_input[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
                scatter_gw = [all_grad_weights[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)]
            else:
                scatter_gi = None
                scatter_gw = None

            grad_input_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_gi,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                feature_shape=(hidden_size,),
                device=ctx.original_device,
                dtype=ctx.original_dtype,
            )
            grad_weights_flat = _dist_scatter_varlen_from_rank0(
                rank0_chunks=scatter_gw,
                all_qlens=all_qlens,
                rank=rank,
                world_size=world_size,
                feature_shape=(num_experts_per_tok,),
                device=ctx.weights_device,
                dtype=torch.bfloat16,
            )
            grad_input = grad_input_flat.view(batch_size, seq_len, hidden_size)
            grad_weights = grad_weights_flat.view(ctx.weights_shape).to(dtype=ctx.weights_dtype)

        elif not ctx.use_broadcast:
            # ---- Single-GPU path ----
            grad_output_flat = grad_output.view(qlen, hidden_size)
            backward_out = ctx.wrapper.backward(
                grad_output_flat,
                output_device=ctx.original_device,
            )
            ctx.wrapper._kt_has_cached_forward = False
            if isinstance(backward_out, tuple) and len(backward_out) == 2:
                grad_input, grad_weights = backward_out
            elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                grad_input, _, grad_weights = backward_out
            else:
                raise ValueError("KTMoEWrapper.backward returned unexpected format.")
            grad_input = grad_input.view(batch_size, seq_len, hidden_size).to(dtype=ctx.original_dtype)
            grad_weights = grad_weights.to(dtype=torch.bfloat16)
        else:
            # No wrapper, no dist — shouldn't happen in normal flow
            grad_input = torch.zeros(batch_size, seq_len, hidden_size, device=ctx.original_device, dtype=ctx.original_dtype)
            grad_weights = torch.zeros(ctx.weights_shape, device=ctx.weights_device, dtype=ctx.weights_dtype)

        # Trigger async repack for next MoE layer in backward order
        next_bwd = getattr(ctx.wrapper, '_next_backward_wrapper', None)
        if next_bwd is not None and getattr(next_bwd, 'share_backward_bb', False):
            next_bwd.submit_backward_repack()

        return grad_input, None, grad_weights, None, None, None, None, None, None, None, None
