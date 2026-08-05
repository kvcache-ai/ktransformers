# SPDX-License-Identifier: Apache-2.0
"""Expert placement planning utilities for heterogeneous MoE inference.

This module contains pure-Python / PyTorch helpers for turning expert
activation statistics into GPU expert masks.

The default frequency strategy intentionally preserves the existing
generate_gpu_experts_masks behavior: select the globally hottest experts.

The score_aware_layer_balanced strategy is a small, practical placement policy
inspired by hybrid CPU-GPU MoE scheduling work: smooth activation scores across
profiling windows and avoid concentrating all GPU experts in only a few layers.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch


SUPPORTED_EXPERT_PLACEMENT_STRATEGIES = frozenset(
    {
        "frequency",
        "score_aware_layer_balanced",
    }
)


def plan_gpu_expert_placement(
    activation_freq: torch.Tensor,
    num_gpu_experts: int,
    strategy: str = "frequency",
    *,
    min_experts_per_layer: int = 0,
    max_experts_per_layer: Optional[int] = None,
    previous_scores: Optional[torch.Tensor] = None,
    alpha: float = 0.8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Plan which MoE experts should be placed on GPU.

    Args:
        activation_freq:
            Activation frequency or score table with shape
            ``(num_layers, num_experts_per_layer)``.
        num_gpu_experts:
            Total number of experts to place on GPU across all layers.
        strategy:
            Placement strategy. ``frequency`` preserves the existing global
            top-k behavior. ``score_aware_layer_balanced`` uses EMA-smoothed
            scores and optional per-layer bounds.
        min_experts_per_layer:
            Minimum experts to place on GPU for each layer. Only meaningful for
            ``score_aware_layer_balanced``.
        max_experts_per_layer:
            Maximum experts to place on GPU for each layer. Defaults to all
            experts in a layer.
        previous_scores:
            Optional previous activation score table for EMA smoothing.
        alpha:
            EMA coefficient for current activation scores when previous_scores
            is provided: ``alpha * current + (1 - alpha) * previous``.

    Returns:
        A tuple ``(gpu_experts_mask, report)`` where mask is a CPU bool tensor
        with the same shape as activation_freq.

    Raises:
        ValueError:
            If inputs are malformed or constraints are impossible.
    """
    scores = _prepare_scores(
        activation_freq=activation_freq,
        previous_scores=previous_scores,
        alpha=alpha,
    )
    num_layers, num_experts_per_layer = scores.shape
    total_experts = num_layers * num_experts_per_layer
    budget = _clamp_int(num_gpu_experts, 0, total_experts)

    if strategy not in SUPPORTED_EXPERT_PLACEMENT_STRATEGIES:
        raise ValueError(
            f"Unknown expert placement strategy: {strategy!r}. "
            f"Supported strategies: {sorted(SUPPORTED_EXPERT_PLACEMENT_STRATEGIES)}"
        )

    if strategy == "frequency":
        mask = _global_topk_mask(scores, budget)
    else:
        mask = _score_aware_layer_balanced_mask(
            scores=scores,
            budget=budget,
            min_experts_per_layer=min_experts_per_layer,
            max_experts_per_layer=max_experts_per_layer,
        )

    return mask, _build_report(
        scores=scores,
        mask=mask,
        requested_num_gpu_experts=num_gpu_experts,
        strategy=strategy,
        min_experts_per_layer=min_experts_per_layer,
        max_experts_per_layer=max_experts_per_layer,
        alpha=alpha,
        previous_scores_provided=previous_scores is not None,
    )


def _prepare_scores(
    activation_freq: torch.Tensor,
    previous_scores: Optional[torch.Tensor],
    alpha: float,
) -> torch.Tensor:
    if not isinstance(activation_freq, torch.Tensor):
        raise TypeError("activation_freq must be a torch.Tensor")

    if activation_freq.ndim != 2:
        raise ValueError(
            f"activation_freq must be a 2D tensor of shape "
            f"(num_layers, num_experts), got shape {tuple(activation_freq.shape)}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    current = activation_freq.detach().to(device="cpu", dtype=torch.float32)

    if previous_scores is None:
        return current.contiguous()

    if not isinstance(previous_scores, torch.Tensor):
        raise TypeError("previous_scores must be a torch.Tensor when provided")

    if previous_scores.shape != activation_freq.shape:
        raise ValueError(
            f"previous_scores shape {tuple(previous_scores.shape)} must match "
            f"activation_freq shape {tuple(activation_freq.shape)}"
        )

    previous = previous_scores.detach().to(device="cpu", dtype=torch.float32)
    return (alpha * current + (1.0 - alpha) * previous).contiguous()


def _global_topk_mask(scores: torch.Tensor, budget: int) -> torch.Tensor:
    num_layers, num_experts_per_layer = scores.shape
    total_experts = num_layers * num_experts_per_layer

    mask = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    if budget == 0:
        return mask.view(num_layers, num_experts_per_layer)

    flat_scores = scores.reshape(-1)
    _, top_indices = torch.topk(flat_scores, k=budget, largest=True, sorted=False)
    mask[top_indices] = True
    return mask.view(num_layers, num_experts_per_layer)


def _score_aware_layer_balanced_mask(
    scores: torch.Tensor,
    budget: int,
    min_experts_per_layer: int,
    max_experts_per_layer: Optional[int],
) -> torch.Tensor:
    num_layers, num_experts_per_layer = scores.shape

    if min_experts_per_layer < 0:
        raise ValueError(
            f"min_experts_per_layer must be non-negative, got {min_experts_per_layer}"
        )

    if max_experts_per_layer is None:
        max_experts_per_layer = num_experts_per_layer

    if max_experts_per_layer < 0:
        raise ValueError(
            f"max_experts_per_layer must be non-negative, got {max_experts_per_layer}"
        )

    if min_experts_per_layer > max_experts_per_layer:
        raise ValueError(
            f"min_experts_per_layer ({min_experts_per_layer}) cannot exceed "
            f"max_experts_per_layer ({max_experts_per_layer})"
        )

    if max_experts_per_layer > num_experts_per_layer:
        raise ValueError(
            f"max_experts_per_layer ({max_experts_per_layer}) cannot exceed "
            f"num_experts_per_layer ({num_experts_per_layer})"
        )

    min_required = min_experts_per_layer * num_layers
    max_allowed = max_experts_per_layer * num_layers

    if budget < min_required:
        raise ValueError(
            f"num_gpu_experts={budget} is too small for "
            f"min_experts_per_layer={min_experts_per_layer} across "
            f"{num_layers} layers; need at least {min_required}"
        )

    if budget > max_allowed:
        raise ValueError(
            f"num_gpu_experts={budget} exceeds max_experts_per_layer="
            f"{max_experts_per_layer} across {num_layers} layers; "
            f"at most {max_allowed} experts can be selected"
        )

    mask = torch.zeros_like(scores, dtype=torch.bool, device="cpu")
    per_layer_count = [0 for _ in range(num_layers)]

    # First satisfy the minimum layer coverage constraint.
    if min_experts_per_layer > 0:
        for layer_idx in range(num_layers):
            layer_scores = scores[layer_idx]
            _, expert_indices = torch.topk(
                layer_scores,
                k=min_experts_per_layer,
                largest=True,
                sorted=False,
            )
            mask[layer_idx, expert_indices] = True
            per_layer_count[layer_idx] = min_experts_per_layer

    remaining = budget - int(mask.sum().item())
    if remaining == 0:
        return mask

    # Deterministic global ordering with layer cap enforcement.
    # Tie-breaks by layer index then expert index so tests and profiles are stable.
    candidates: list[tuple[float, int, int]] = []
    for layer_idx in range(num_layers):
        for expert_idx in range(num_experts_per_layer):
            if not bool(mask[layer_idx, expert_idx].item()):
                candidates.append(
                    (float(scores[layer_idx, expert_idx].item()), layer_idx, expert_idx)
                )

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    for _, layer_idx, expert_idx in candidates:
        if remaining == 0:
            break
        if per_layer_count[layer_idx] >= max_experts_per_layer:
            continue

        mask[layer_idx, expert_idx] = True
        per_layer_count[layer_idx] += 1
        remaining -= 1

    if remaining != 0:
        raise RuntimeError(
            f"Failed to allocate requested GPU expert budget; {remaining} experts "
            f"were left unassigned. This indicates an internal placement bug."
        )

    return mask


def _build_report(
    scores: torch.Tensor,
    mask: torch.Tensor,
    requested_num_gpu_experts: int,
    strategy: str,
    min_experts_per_layer: int,
    max_experts_per_layer: Optional[int],
    alpha: float,
    previous_scores_provided: bool,
) -> dict[str, Any]:
    selected_scores = scores[mask]
    total_score = float(scores.sum().item())
    selected_score = float(selected_scores.sum().item()) if selected_scores.numel() else 0.0
    per_layer_counts_tensor = mask.sum(dim=1).to(dtype=torch.float32)
    per_layer_counts = [int(v) for v in mask.sum(dim=1).tolist()]

    if per_layer_counts_tensor.numel() > 1:
        layer_std = float(per_layer_counts_tensor.std(unbiased=False).item())
    else:
        layer_std = 0.0

    if math.isclose(total_score, 0.0):
        expected_hit_rate = 0.0
    else:
        expected_hit_rate = selected_score / total_score

    return {
        "strategy": strategy,
        "num_layers": int(scores.shape[0]),
        "num_experts_per_layer": int(scores.shape[1]),
        "requested_num_gpu_experts": int(requested_num_gpu_experts),
        "actual_num_gpu_experts": int(mask.sum().item()),
        "expected_hit_rate": expected_hit_rate,
        "selected_score": selected_score,
        "total_score": total_score,
        "per_layer_counts": per_layer_counts,
        "layer_min": int(min(per_layer_counts)) if per_layer_counts else 0,
        "layer_max": int(max(per_layer_counts)) if per_layer_counts else 0,
        "layer_mean": float(per_layer_counts_tensor.mean().item())
        if per_layer_counts_tensor.numel()
        else 0.0,
        "layer_std": layer_std,
        "min_experts_per_layer": int(min_experts_per_layer),
        "max_experts_per_layer": max_experts_per_layer,
        "alpha": float(alpha),
        "previous_scores_provided": previous_scores_provided,
    }


def _clamp_int(value: int, min_value: int, max_value: int) -> int:
    return min(max(int(value), min_value), max_value)