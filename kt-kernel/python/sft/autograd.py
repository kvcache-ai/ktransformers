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


def _poison_checkpoint_cache(wrapper: Any, error: BaseException | str) -> None:
    if wrapper is None:
        return
    poison = getattr(wrapper, "poison_checkpoint_output", None)
    if poison is not None:
        poison(error)


def _clear_checkpoint_cache(wrapper: Any) -> None:
    clear = getattr(wrapper, "clear_checkpoint_output", None)
    if clear is not None:
        clear()


def _sync_rank0_exception(
    error: BaseException | None,
    *,
    device: torch.device,
    context: str,
) -> None:
    """Broadcast a rank-0-only failure before peers enter the next collective."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        if error is not None:
            raise RuntimeError(f"{context}: {error}") from error
        return

    rank = dist.get_rank()
    encoded = (
        f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
        if rank == 0 and error is not None
        else b""
    )
    length = torch.tensor([len(encoded)], device=device, dtype=torch.int64)
    dist.broadcast(length, src=0)
    payload_length = int(length.item())
    if payload_length == 0:
        return

    if rank == 0:
        payload = torch.tensor(list(encoded), device=device, dtype=torch.uint8)
    else:
        payload = torch.empty(payload_length, device=device, dtype=torch.uint8)
    dist.broadcast(payload, src=0)
    message = bytes(payload.cpu().tolist()).decode("utf-8", errors="replace")
    synced_error = RuntimeError(f"{context}: {message}")
    if error is not None:
        raise synced_error from error
    raise synced_error


def _sync_any_rank_exception(
    error: BaseException | None,
    *,
    device: torch.device,
    context: str,
) -> None:
    """Raise the lowest-rank failure coherently on every distributed peer."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        if error is not None:
            raise RuntimeError(f"{context}: {error}") from error
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    encoded = (
        f"{type(error).__name__}: {error}".encode("utf-8", errors="replace")
        if error is not None
        else b""
    )
    local_length = torch.tensor([len(encoded)], device=device, dtype=torch.int64)
    lengths = [torch.empty_like(local_length) for _ in range(world_size)]
    dist.all_gather(lengths, local_length)
    failing_ranks = [idx for idx, value in enumerate(lengths) if int(value.item()) > 0]
    if not failing_ranks:
        return

    source = failing_ranks[0]
    payload_length = int(lengths[source].item())
    if rank == source:
        payload = torch.tensor(list(encoded), device=device, dtype=torch.uint8)
    else:
        payload = torch.empty(payload_length, device=device, dtype=torch.uint8)
    dist.broadcast(payload, src=source)
    message = bytes(payload.cpu().tolist()).decode("utf-8", errors="replace")
    synced_error = RuntimeError(f"{context} on rank {source}: {message}")
    if rank == source and error is not None:
        raise synced_error from error
    raise synced_error


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
        cache_checkpoint_forward: bool = False,
        reuse_cached_forward: bool = False,
        gate_proj_param: torch.Tensor | None = None,
        up_proj_param: torch.Tensor | None = None,
        down_proj_param: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if _KT_SFT_DEBUG:
            logging.debug(
                "KTMoEFunction.forward: layer=%d training=%s train_lora=%s",
                layer_idx,
                training,
                train_lora,
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
                    raise RuntimeError(f"all_qlens length mismatch: got {len(all_qlens_list)}, expected {world_size}")
            if int(all_qlens_list[rank]) != qlen:
                raise RuntimeError(f"Rank {rank} qlen mismatch: local={qlen}, all_qlens[{rank}]={all_qlens_list[rank]}")
            total_qlen = sum(all_qlens_list)

            # Rank 0 owns the backend. Synchronize its failure before peers
            # enter the output scatter.
            forward_error = None
            if rank == 0:
                try:
                    if wrapper is None:
                        raise RuntimeError("Rank 0 does not own the KT backend")
                    if reuse_cached_forward:
                        with torch.profiler.record_function("kt.sft.checkpoint_cached_cpu_moe"):
                            cpu_output = wrapper.get_checkpoint_output(total_qlen, output_device=original_device)
                    elif cache_checkpoint_forward:
                        with torch.profiler.record_function("kt.sft.cpu_forward_sync"):
                            cached_output = wrapper.sync_forward(output_device=None)
                        wrapper.cache_checkpoint_output(cached_output, total_qlen)
                        cpu_output = cached_output.to(device=original_device, non_blocking=True)
                    else:
                        with torch.profiler.record_function("kt.sft.cpu_forward_sync"):
                            cpu_output = wrapper.sync_forward(output_device=original_device)
                    cpu_output = cpu_output.to(dtype=original_dtype).view(total_qlen, hidden_size)
                    offsets = _qlen_offsets(all_qlens_list)
                    scatter_list = [
                        cpu_output[offsets[i] : offsets[i + 1]].contiguous()
                        for i in range(world_size)
                    ]
                except Exception as exc:
                    forward_error = exc
                    if cache_checkpoint_forward or reuse_cached_forward:
                        _poison_checkpoint_cache(wrapper, exc)
            else:
                scatter_list = None
            _sync_rank0_exception(
                forward_error,
                device=original_device,
                context=f"Layer {layer_idx} CPU forward synchronization failed",
            )

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
            try:
                if reuse_cached_forward:
                    with torch.profiler.record_function("kt.sft.checkpoint_cached_cpu_moe"):
                        cpu_output = wrapper.get_checkpoint_output(qlen, output_device=original_device)
                elif cache_checkpoint_forward:
                    with torch.profiler.record_function("kt.sft.cpu_forward_sync"):
                        cached_output = wrapper.sync_forward(output_device=None)
                    wrapper.cache_checkpoint_output(cached_output, qlen)
                    cpu_output = cached_output.to(device=original_device, non_blocking=True)
                else:
                    with torch.profiler.record_function("kt.sft.cpu_forward_sync"):
                        cpu_output = wrapper.sync_forward(output_device=original_device)
                output = cpu_output.view(batch_size, seq_len, hidden_size).to(dtype=original_dtype)
            except Exception as exc:
                if cache_checkpoint_forward or reuse_cached_forward:
                    _poison_checkpoint_cache(wrapper, exc)
                raise
        else:
            # Broadcast-only rank (no wrapper)
            output = torch.empty(batch_size, seq_len, hidden_size, device=original_device, dtype=original_dtype)

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
        ctx.cache_checkpoint_forward = bool(cache_checkpoint_forward)
        ctx.reuse_cached_forward = bool(reuse_cached_forward)

        # Store base weight param references for gradient flow in full mode
        ctx.full_weight_grad = (
            wrapper is not None and getattr(wrapper, "_full_weight_grad", False) and gate_proj_param is not None
        )
        ctx.authoritative_optimizer_grads = bool(
            wrapper is not None and getattr(wrapper, "_uses_authoritative_optimizer_grads", False)
        )

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
        import torch.distributed as dist

        dist_on = ctx.dist_on
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Wait for any in-flight async repack before recompute forward uses the pool
        repack_wait_error = None
        if (not dist_on or rank == 0) and getattr(ctx.wrapper, "share_backward_bb", False):
            try:
                with torch.profiler.record_function("kt.sft.wait_backward_repack"):
                    ctx.wrapper.wait_backward_repack()
            except Exception as exc:
                repack_wait_error = exc
                if ctx.cache_checkpoint_forward or ctx.reuse_cached_forward:
                    _poison_checkpoint_cache(ctx.wrapper, exc)
        if dist_on:
            _sync_rank0_exception(
                repack_wait_error,
                device=ctx.original_device,
                context=f"Layer {ctx.layer_idx} backward repack wait failed",
            )
        elif repack_wait_error is not None:
            raise repack_wait_error

        # Access saved_tensors FIRST — under non-reentrant checkpoint this
        # triggers the unpack hook which runs a full decoder-layer recompute,
        # populating the C++ cache before we call wrapper.backward().
        recompute_error = None
        try:
            with torch.profiler.record_function("kt.sft.checkpoint_recompute"):
                _ = ctx.saved_tensors
        except Exception as exc:
            recompute_error = exc

        retained_cache = ctx.cache_checkpoint_forward or ctx.reuse_cached_forward
        if retained_cache and dist_on:
            try:
                _sync_any_rank_exception(
                    recompute_error,
                    device=ctx.original_device,
                    context=f"Layer {ctx.layer_idx} checkpoint recompute failed",
                )
            except Exception as exc:
                _poison_checkpoint_cache(ctx.wrapper, exc)
                raise
        elif recompute_error is not None:
            if retained_cache:
                _poison_checkpoint_cache(ctx.wrapper, recompute_error)
            raise recompute_error

        qlen = ctx.qlen
        hidden_size = ctx.hidden_size
        batch_size = ctx.batch_size
        seq_len = ctx.seq_len
        world_size = ctx.world_size
        num_experts_per_tok = ctx.num_experts_per_tok

        if _KT_SFT_DEBUG:
            logging.debug(
                "KTMoEFunction.backward: layer=%d dist_on=%s qlen=%d",
                getattr(ctx, "layer_idx", -1),
                dist_on,
                qlen,
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
            authoritative_grad_published = False

            def close_published_authoritative_window() -> None:
                nonlocal authoritative_grad_published
                if not (rank == 0 and authoritative_grad_published and ctx.wrapper is not None):
                    return
                try:
                    ctx.wrapper.release_authoritative_optimizer_grads()
                except Exception:
                    logger.exception(
                        "Failed to close authoritative optimizer-gradient window after distributed backward error"
                    )
                finally:
                    authoritative_grad_published = False

            backward_error = None
            if rank == 0:
                try:
                    if ctx.wrapper is None:
                        raise RuntimeError("Rank 0 does not own the KT backend")
                    all_go = torch.cat(gathered_go, dim=0)
                    total_qlen = int(all_go.shape[0])

                    with torch.profiler.record_function("kt.sft.cpu_backward"):
                        # Rank 0 computes one dWeight over every rank's gathered
                        # rows. Match DDP averaging at the C++ gradient producer
                        # so grad clipping and GAS both observe normalized values.
                        backward_out = ctx.wrapper.backward(
                            all_go,
                            output_device=ctx.original_device,
                            optimizer_grad_scale=1.0 / world_size,
                        )
                    authoritative_grad_published = ctx.authoritative_optimizer_grads
                    if isinstance(backward_out, tuple) and len(backward_out) == 2:
                        all_grad_input, all_grad_weights = backward_out
                    elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                        all_grad_input, _, all_grad_weights = backward_out
                    else:
                        raise ValueError("KTMoEWrapper.backward returned unexpected format.")

                    all_grad_input = all_grad_input.to(dtype=ctx.original_dtype).view(total_qlen, hidden_size)
                    all_grad_weights = all_grad_weights.to(dtype=torch.bfloat16).view(
                        total_qlen, num_experts_per_tok
                    )

                    offsets = _qlen_offsets(all_qlens)
                    scatter_gi = [
                        all_grad_input[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)
                    ]
                    scatter_gw = [
                        all_grad_weights[offsets[i] : offsets[i + 1]].contiguous() for i in range(world_size)
                    ]
                except Exception as exc:
                    backward_error = exc
                    close_published_authoritative_window()
                    if ctx.cache_checkpoint_forward or ctx.reuse_cached_forward:
                        _poison_checkpoint_cache(ctx.wrapper, exc)
                finally:
                    try:
                        _clear_checkpoint_cache(ctx.wrapper)
                    except Exception as clear_exc:
                        if backward_error is None:
                            backward_error = clear_exc
                        _poison_checkpoint_cache(ctx.wrapper, clear_exc)
                        close_published_authoritative_window()
            else:
                scatter_gi = None
                scatter_gw = None
            _sync_rank0_exception(
                backward_error,
                device=ctx.original_device,
                context=f"Layer {ctx.layer_idx} CPU backward failed",
            )

            try:
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
            except Exception:
                close_published_authoritative_window()
                raise

        elif not ctx.use_broadcast:
            # ---- Single-GPU path ----
            grad_output_flat = grad_output.view(qlen, hidden_size)
            try:
                with torch.profiler.record_function("kt.sft.cpu_backward"):
                    if ctx.authoritative_optimizer_grads:
                        backward_out = ctx.wrapper.backward(
                            grad_output_flat,
                            output_device=ctx.original_device,
                            optimizer_grad_scale=1.0,
                        )
                    else:
                        backward_out = ctx.wrapper.backward(
                            grad_output_flat,
                            output_device=ctx.original_device,
                        )
                if isinstance(backward_out, tuple) and len(backward_out) == 2:
                    grad_input, grad_weights = backward_out
                elif isinstance(backward_out, tuple) and len(backward_out) == 3:
                    grad_input, _, grad_weights = backward_out
                else:
                    raise ValueError("KTMoEWrapper.backward returned unexpected format.")
                grad_input = grad_input.view(batch_size, seq_len, hidden_size).to(dtype=ctx.original_dtype)
                grad_weights = grad_weights.to(dtype=torch.bfloat16)
            except Exception as exc:
                if ctx.cache_checkpoint_forward or ctx.reuse_cached_forward:
                    _poison_checkpoint_cache(ctx.wrapper, exc)
                if ctx.authoritative_optimizer_grads and ctx.wrapper is not None:
                    try:
                        ctx.wrapper.release_authoritative_optimizer_grads()
                    except Exception:
                        logger.exception(
                            "Failed to close authoritative optimizer-gradient window after local backward error"
                        )
                raise
            finally:
                try:
                    _clear_checkpoint_cache(ctx.wrapper)
                except Exception:
                    if ctx.authoritative_optimizer_grads and ctx.wrapper is not None:
                        try:
                            ctx.wrapper.release_authoritative_optimizer_grads()
                        except Exception:
                            logger.exception(
                                "Failed to close authoritative optimizer-gradient window "
                                "after checkpoint cache cleanup error"
                            )
                    raise
        else:
            # No wrapper, no dist — shouldn't happen in normal flow
            grad_input = torch.zeros(
                batch_size, seq_len, hidden_size, device=ctx.original_device, dtype=ctx.original_dtype
            )
            grad_weights = torch.zeros(ctx.weights_shape, device=ctx.weights_device, dtype=ctx.weights_dtype)

        # Trigger async repack for next MoE layer in backward order
        next_bwd = getattr(ctx.wrapper, "_next_backward_wrapper", None)
        repack_submit_error = None
        if (not dist_on or rank == 0) and next_bwd is not None and getattr(next_bwd, "share_backward_bb", False):
            try:
                with torch.profiler.record_function("kt.sft.submit_backward_repack"):
                    next_bwd.submit_backward_repack()
            except Exception as exc:
                repack_submit_error = exc
                if ctx.authoritative_optimizer_grads and ctx.wrapper is not None:
                    try:
                        ctx.wrapper.release_authoritative_optimizer_grads()
                    except Exception:
                        logger.exception(
                            "Failed to close authoritative optimizer-gradient window after repack submission error"
                        )
        if dist_on:
            _sync_rank0_exception(
                repack_submit_error,
                device=ctx.original_device,
                context=f"Layer {ctx.layer_idx} backward repack submission failed",
            )
        elif repack_submit_error is not None:
            raise repack_submit_error

        # Legacy backends still use PyTorch AccumulateGrad.  AMXBF16_SFT
        # publishes the C++ buffers directly as Parameter.grad, so returning
        # them here would create a second giant copy and an aten::add_.
        if ctx.full_weight_grad and ctx.wrapper is not None and not ctx.authoritative_optimizer_grads:
            grad_gate_proj = ctx.wrapper.grad_gate_proj_buf
            grad_up_proj = ctx.wrapper.grad_up_proj_buf
            grad_down_proj = ctx.wrapper.grad_down_proj_buf
        else:
            grad_gate_proj = None
            grad_up_proj = None
            grad_down_proj = None

        return (
            grad_input,
            None,
            grad_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_gate_proj,
            grad_up_proj,
            grad_down_proj,
        )
