# Co-activation-aware GPU expert placement planner
# SPDX-License-Identifier: Apache-2.0

"""
Co-activation-aware GPU expert placement for MoE inference.

The default placement helper (:func:`kt_kernel.generate_gpu_experts_masks`)
selects the globally most-frequently-activated experts. That objective scores
every expert *independently*: it answers "which experts are hot on their own?"
but is blind to which experts fire *together* on the same token.

For CPU/GPU hybrid MoE serving this matters. Every token routes to ``top_k``
experts, and the throughput-relevant quantity is how many of a token's routed
experts are GPU-resident. If two experts almost always co-fire but only one is
placed on the GPU, that token still pays the slow CPU path for the other. A
placement that keeps *co-firing clusters* together therefore raises the
per-token GPU hit rate at the same GPU-expert budget.

This module builds pairwise co-occurrence statistics from per-token expert
selection traces and plans placement to maximise expected co-resident hits per
token, rather than the sum of marginal frequencies. When only marginal
activation counts are available (no per-token traces), it degrades to exactly
the frequency top-k behaviour, so it never regresses against the existing
strategy.

The planner is pure Python/torch, performs no routing and no weight movement,
and produces the same ``(num_layers, num_experts)`` boolean mask that the rest
of the pipeline already consumes.

Example:
    >>> import torch
    >>> from kt_kernel.expert_placement import plan_gpu_expert_placement
    >>> # Two layers, four experts, per-token routed-expert traces per layer.
    >>> traces = {
    ...     0: torch.tensor([[0, 1], [0, 1], [2, 3]]),
    ...     1: torch.tensor([[1, 2], [1, 2], [0, 3]]),
    ... }
    >>> mask = plan_gpu_expert_placement(
    ...     num_layers=2, num_experts=4, num_gpu_experts=4, token_traces=traces
    ... )
    >>> mask.shape
    torch.Size([2, 4])
"""

from __future__ import annotations

import torch
from typing import Dict, Optional


def build_coactivation_matrix(
    token_traces: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Build a symmetric pairwise co-occurrence matrix from per-token traces.

    ``C[i][j]`` counts the number of tokens in which experts ``i`` and ``j``
    were both routed. The diagonal ``C[i][i]`` holds the marginal activation
    count of expert ``i`` (the number of tokens that routed to it).

    Args:
        token_traces: Long/int tensor of shape ``(num_tokens, top_k)`` holding
                      the routed expert ids for each token in one layer. Values
                      outside ``[0, num_experts)`` (e.g. ``-1`` padding) are
                      ignored.
        num_experts: Total number of experts in the layer.

    Returns:
        Float tensor of shape ``(num_experts, num_experts)`` on CPU.
    """
    coactivation = torch.zeros(num_experts, num_experts, dtype=torch.float32, device="cpu")

    if token_traces.numel() == 0:
        return coactivation

    traces = token_traces.to(device="cpu", dtype=torch.long)
    if traces.dim() == 1:
        traces = traces.unsqueeze(0)

    num_tokens, top_k = traces.shape
    for token_idx in range(num_tokens):
        ids = traces[token_idx]
        valid = ids[(ids >= 0) & (ids < num_experts)]
        unique_ids = torch.unique(valid)
        if unique_ids.numel() == 0:
            continue
        # Outer product over the present experts increments every ordered pair,
        # including the (i, i) diagonal, giving marginal counts for free.
        rows = unique_ids.unsqueeze(1).expand(-1, unique_ids.numel()).reshape(-1)
        cols = unique_ids.unsqueeze(0).expand(unique_ids.numel(), -1).reshape(-1)
        coactivation[rows, cols] += 1.0

    return coactivation


def _greedy_cluster_select(
    coactivation: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """
    Greedily select ``budget`` experts that form a tight co-firing cluster.

    The objective is to keep experts that fire *together* on the GPU, so that
    whole tokens become GPU-resident and skip the CPU path. The selection is
    seeded with the single strongest co-firing pair (largest off-diagonal
    entry), then grown by adding, at each step, the expert with the highest
    total co-activation with the experts already chosen.

    Seeding from the strongest pair rather than the single hottest expert is
    what distinguishes this from frequency top-k: an individually-hot expert
    that never co-fires with a resident partner does not help whole-token
    residency, whereas a co-firing pair does.

    Args:
        coactivation: ``(num_experts, num_experts)`` co-occurrence matrix whose
                      diagonal holds marginal counts and whose off-diagonal
                      holds pairwise co-firing counts.
        budget: Number of experts to select for this layer.

    Returns:
        Long tensor of selected expert indices (length ``min(budget, num_experts)``).
    """
    num_experts = coactivation.shape[0]
    budget = max(0, min(int(budget), num_experts))
    if budget == 0:
        return torch.empty(0, dtype=torch.long)

    marginal = torch.diagonal(coactivation)
    if budget == 1:
        # No pair to keep together; fall back to the hottest single expert.
        return torch.tensor([int(torch.argmax(marginal).item())], dtype=torch.long)

    # Off-diagonal only: zero the diagonal so seeding picks a genuine pair.
    off_diagonal = coactivation - torch.diag(marginal)
    selected = torch.zeros(num_experts, dtype=torch.bool)

    # Seed with the strongest co-firing pair. argmax over the flattened matrix
    # yields the (i, j) with the largest co-activation; ties break toward the
    # lowest flattened index for determinism.
    pair_flat = int(torch.argmax(off_diagonal).item())
    seed_i, seed_j = divmod(pair_flat, num_experts)
    if off_diagonal[seed_i, seed_j] <= 0:
        # No expert ever co-fires with another (degenerate trace): fall back to
        # marginal top-k so behaviour matches the frequency strategy.
        return torch.topk(marginal, k=budget, largest=True, sorted=False).indices
    selected[seed_i] = True
    selected[seed_j] = True

    # Affinity of every expert to the current selected set.
    affinity = off_diagonal[seed_i].clone() + off_diagonal[seed_j].clone()

    while int(selected.sum().item()) < budget:
        candidate_score = affinity.clone()
        candidate_score[selected] = float("-inf")
        # Tie-break toward the hotter marginal expert, then lowest index.
        best_score = torch.max(candidate_score)
        if best_score == float("-inf"):
            break
        tied = (candidate_score == best_score).nonzero(as_tuple=False).flatten()
        best = int(tied[torch.argmax(marginal[tied])].item())
        selected[best] = True
        affinity += off_diagonal[best]

    chosen = int(selected.sum().item())
    if chosen < budget:
        # Cluster smaller than budget (isolated experts remain): fill the rest
        # by marginal frequency so the GPU budget is fully used.
        remaining = budget - chosen
        fill_scores = marginal.clone()
        fill_scores[selected] = float("-inf")
        extra = torch.topk(fill_scores, k=remaining, largest=True, sorted=False).indices
        selected[extra] = True

    return torch.nonzero(selected, as_tuple=False).flatten()


def _coverage_select(
    token_traces: torch.Tensor,
    num_experts: int,
    budget: int,
    core_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Select experts to maximise the number of *fully-resident tokens* (coverage).

    A token skips the CPU path only when *every* one of its routed experts is on
    the GPU. The objective is therefore not to place the tightest cluster nor
    the individually-hottest experts, but to cover as many whole tokens as the
    budget allows. This is a weighted maximum-coverage / set-cover problem.

    Two-phase greedy:

    1. **Core.** Experts that fire in at least ``core_threshold`` of tokens are
       placed first. A stable core (e.g. the 6 experts present in nearly every
       token of a Qwen-style top-8 route) is always worth resident space.
    2. **Residual set-cover.** Each token is reduced to its non-core experts.
       The planner repeatedly adds the group of residual experts that covers the
       most currently-uncovered tokens per extra slot spent
       (``tokens_gained / new_experts_needed``), until the budget is exhausted.

    Crucially, co-occurrence is scored *among residual (non-core) experts*, not
    against the always-on core. Scoring against the core is misleading: every
    expert co-occurs with an always-on core, so an expert whose true partner can
    never be afforded would look attractive yet complete zero tokens. Reducing
    to residuals removes that decoy signal.

    Args:
        token_traces: ``(num_tokens, top_k)`` routed expert ids for one layer.
        num_experts: Total experts in the layer.
        budget: GPU-expert budget for this layer.
        core_threshold: Fraction of tokens an expert must appear in to be treated
                        as always-on core. Defaults to 0.9.

    Returns:
        Long tensor of selected expert indices (length ``<= budget``).
    """
    budget = max(0, min(int(budget), num_experts))
    if budget == 0:
        return torch.empty(0, dtype=torch.long)

    traces = token_traces.to(device="cpu", dtype=torch.long)
    if traces.dim() == 1:
        traces = traces.unsqueeze(0)

    # Reduce each token to the set of valid, unique routed experts.
    token_sets = []
    for token_ids in traces:
        valid = token_ids[(token_ids >= 0) & (token_ids < num_experts)]
        if valid.numel() > 0:
            token_sets.append(set(int(e) for e in valid.tolist()))
    num_tokens = len(token_sets)
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.long)

    # Marginal presence per expert.
    presence = torch.zeros(num_experts, dtype=torch.float32)
    for s in token_sets:
        for e in s:
            presence[e] += 1.0

    selected = torch.zeros(num_experts, dtype=torch.bool)

    # Phase 1: place the always-on core (subject to budget).
    core = (presence >= core_threshold * num_tokens) & (presence > 0)
    core_ids = torch.nonzero(core, as_tuple=False).flatten()
    # If the core alone exceeds budget, keep the most frequent core experts.
    if core_ids.numel() > budget:
        keep = torch.topk(presence[core_ids], k=budget, largest=True, sorted=False).indices
        core_ids = core_ids[keep]
    selected[core_ids] = True

    # Phase 2: weighted set-cover on residuals (non-core experts per token).
    selected_set = set(int(e) for e in torch.nonzero(selected, as_tuple=False).flatten().tolist())
    while int(selected.sum().item()) < budget:
        remaining_budget = budget - int(selected.sum().item())

        # For each not-yet-selected expert, tally the residual it would help
        # complete: tokens whose only missing experts include it. We score whole
        # residual groups by counting, per uncovered token, the extra experts it
        # needs, and crediting candidate groups by tokens-gained per slot.
        # Aggregate candidate residual groups across uncovered tokens.
        group_gain: Dict[frozenset, int] = {}
        for s in token_sets:
            missing = s - selected_set
            if not missing:
                continue  # already fully covered
            if len(missing) > remaining_budget:
                continue  # cannot be completed within remaining budget
            key = frozenset(missing)
            group_gain[key] = group_gain.get(key, 0) + 1

        if not group_gain:
            break  # no token can be completed with the remaining budget

        # Pick the residual group with the best tokens-gained-per-extra-slot,
        # breaking ties toward more tokens covered, then fewer experts.
        best_key = None
        best_ratio = -1.0
        best_tokens = -1
        for key, tokens_gained in group_gain.items():
            cost = len(key)
            ratio = tokens_gained / cost
            if ratio > best_ratio or (ratio == best_ratio and tokens_gained > best_tokens):
                best_ratio = ratio
                best_tokens = tokens_gained
                best_key = key

        if best_key is None:
            break
        for e in best_key:
            selected[e] = True
            selected_set.add(e)

    # If budget remains and every token is covered, fill by marginal frequency.
    if int(selected.sum().item()) < budget:
        remaining = budget - int(selected.sum().item())
        fill_scores = presence.clone()
        fill_scores[selected] = float("-inf")
        extra = torch.topk(fill_scores, k=remaining, largest=True, sorted=False).indices
        selected[extra] = True

    return torch.nonzero(selected, as_tuple=False).flatten()


def plan_gpu_expert_placement(
    num_layers: int,
    num_experts: int,
    num_gpu_experts: int,
    token_traces: Optional[Dict[int, torch.Tensor]] = None,
    activation_freq: Optional[torch.Tensor] = None,
    strategy: str = "coverage",
    core_threshold: float = 0.9,
) -> torch.Tensor:
    """
    Plan GPU expert placement to maximise whole-token GPU residency.

    The GPU-expert budget is distributed evenly across layers (matching the
    ``num_gpu_experts`` semantics used elsewhere: it is the per-layer count).

    Strategies (used only when per-layer ``token_traces`` are available):
        - ``"coverage"`` (default): weighted maximum-coverage. Places a stable
          core, then greedily adds the residual expert groups that complete the
          most tokens per slot. Maximises the number of fully-resident tokens,
          which is the quantity that governs CPU-path avoidance. See
          :func:`_coverage_select`.
        - ``"cluster"``: single greedy co-firing cluster grown from the strongest
          pair. See :func:`_greedy_cluster_select`.

    Fallback order when a layer has no traces (independent of ``strategy``):
        1. ``activation_freq`` present -> marginal frequency top-k (identical to
           :func:`generate_gpu_experts_masks` restricted per layer).
        2. Neither -> uniform (first ``num_gpu_experts`` experts per layer).

    Args:
        num_layers: Number of MoE layers.
        num_experts: Experts per layer.
        num_gpu_experts: Experts to place on GPU *per layer*.
        token_traces: Optional mapping ``layer_idx -> (num_tokens, top_k)`` tensor
                      of routed expert ids. Layers absent from the map fall back
                      to ``activation_freq``/uniform for that layer.
        activation_freq: Optional ``(num_layers, num_experts)`` marginal counts,
                         used as a fallback when a layer has no traces.
        strategy: ``"coverage"`` or ``"cluster"``. Defaults to ``"coverage"``.
        core_threshold: Core-membership fraction for the coverage strategy.

    Returns:
        Boolean mask of shape ``(num_layers, num_experts)`` on CPU. ``True``
        means the expert should be placed on GPU.
    """
    if strategy not in ("coverage", "cluster"):
        raise ValueError(f"unknown strategy {strategy!r}; expected 'coverage' or 'cluster'")

    per_layer_budget = max(0, min(int(num_gpu_experts), num_experts))
    mask = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")

    if per_layer_budget == 0:
        return mask

    for layer_idx in range(num_layers):
        layer_trace = None if token_traces is None else token_traces.get(layer_idx)

        if layer_trace is not None and layer_trace.numel() > 0:
            if strategy == "coverage":
                selected = _coverage_select(layer_trace, num_experts, per_layer_budget, core_threshold)
            else:
                coactivation = build_coactivation_matrix(layer_trace, num_experts)
                selected = _greedy_cluster_select(coactivation, per_layer_budget)
        elif activation_freq is not None:
            freq = activation_freq[layer_idx].to(device="cpu")
            selected = torch.topk(freq, k=per_layer_budget, largest=True, sorted=False).indices
        else:
            selected = torch.arange(per_layer_budget, dtype=torch.long)

        mask[layer_idx, selected] = True

    return mask


class EmaHotnessTracker:
    """Online per-layer EMA expert-hotness tracker for adaptive GPU placement.

    The planners above choose a *static* placement from a fixed trace. In serving
    the workload shifts -- a coding request routes to different experts than a
    math one -- so a mask frozen at load time drifts out of date and the per-token
    GPU hit rate falls. This tracker instead maintains a per-layer exponential
    moving average of expert activation and can emit an up-to-date residency mask
    at any point, adapting to the live workload with only ``O(top_k)`` work per
    layer per observed token.

    It is workload- and model-agnostic: no offline profile and no per-domain
    tables are required, so the same tracker applies unchanged across models,
    quantizations, and deployments. Cold start optionally warms from marginal
    activation frequencies (e.g. the same ``activation_freq`` that
    :func:`generate_gpu_experts_masks` consumes); with no prior it begins uniform
    and converges as tokens arrive. Because the warm-started mask before any
    observation is exactly frequency top-k, the tracker never regresses against
    the existing static strategy at cold start.

    The update rule matches a standard EMA of the per-token activation indicator:
    each observed token decays every score by ``decay`` and adds ``1 - decay`` to
    the experts it routed to. Higher ``decay`` means longer memory (closer to
    global frequency); lower ``decay`` adapts faster to the current request.

    This class performs no routing and moves no weights. It only tracks statistics
    and produces the same ``(num_layers, num_experts)`` boolean mask the rest of
    the pipeline already consumes, so integrating it is a drop-in replacement for a
    static mask.

    Example:
        >>> tracker = EmaHotnessTracker(num_layers=2, num_experts=4,
        ...                             num_gpu_experts=2, decay=0.9)
        >>> tracker.observe(0, torch.tensor([0, 1]))  # a token routed to 0,1
        >>> tracker.observe(0, torch.tensor([0, 2]))  # then to 0,2
        >>> mask = tracker.mask()
        >>> mask.shape
        torch.Size([2, 4])
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        num_gpu_experts: int,
        decay: float = 0.9,
        activation_freq: Optional[torch.Tensor] = None,
    ) -> None:
        if num_layers <= 0 or num_experts <= 0:
            raise ValueError("num_layers and num_experts must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0.0, 1.0); got {decay}")

        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.num_gpu_experts = max(0, min(int(num_gpu_experts), int(num_experts)))
        self.decay = float(decay)
        self._hotness = torch.zeros(self.num_layers, self.num_experts, dtype=torch.float32, device="cpu")

        if activation_freq is not None:
            freq = activation_freq.to(device="cpu", dtype=torch.float32)
            if freq.shape != (self.num_layers, self.num_experts):
                raise ValueError(
                    f"activation_freq shape {tuple(freq.shape)} != " f"{(self.num_layers, self.num_experts)}"
                )
            # Normalise each layer to a proper starting hotness so the warm-start
            # mask equals frequency top-k. Rows that are all-zero stay uniform.
            row_max = freq.amax(dim=1, keepdim=True)
            self._hotness = torch.where(row_max > 0, freq / row_max, freq)

    def observe(self, layer_idx: int, routed_expert_ids: torch.Tensor) -> None:
        """Update one layer's EMA with the experts a single token routed to.

        Args:
            layer_idx: Layer index in ``[0, num_layers)``. Out-of-range is ignored.
            routed_expert_ids: 1-D tensor of the token's routed expert ids. Values
                               outside ``[0, num_experts)`` (e.g. ``-1`` padding)
                               are ignored.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        ids = routed_expert_ids.to(device="cpu", dtype=torch.long).flatten()
        ids = ids[(ids >= 0) & (ids < self.num_experts)]
        row = self._hotness[layer_idx]
        row.mul_(self.decay)
        if ids.numel() > 0:
            row.index_add_(0, ids, torch.full((ids.numel(),), 1.0 - self.decay, dtype=torch.float32))

    def observe_batch(self, token_traces: Dict[int, torch.Tensor]) -> None:
        """Feed a batch of per-layer traces token-by-token, preserving order.

        Args:
            token_traces: Mapping ``layer_idx -> (num_tokens, top_k)`` routed ids.
                          Tokens are consumed in row order so recency is respected.
        """
        for layer_idx, traces in token_traces.items():
            ids = traces.to(device="cpu", dtype=torch.long)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            for token_ids in ids:
                self.observe(layer_idx, token_ids)

    def scores(self) -> torch.Tensor:
        """Return a copy of the current ``(num_layers, num_experts)`` EMA scores."""
        return self._hotness.clone()

    def mask(self) -> torch.Tensor:
        """Emit the current GPU-residency mask: per-layer top-``num_gpu_experts``.

        Returns:
            Boolean ``(num_layers, num_experts)`` tensor on CPU. ``True`` means the
            expert should be GPU-resident given activation observed so far.
        """
        mask = torch.zeros(self.num_layers, self.num_experts, dtype=torch.bool, device="cpu")
        if self.num_gpu_experts == 0:
            return mask
        for layer_idx in range(self.num_layers):
            selected = torch.topk(self._hotness[layer_idx], k=self.num_gpu_experts, largest=True, sorted=False).indices
            mask[layer_idx, selected] = True
        return mask


def gpu_hit_rate(
    mask: torch.Tensor,
    token_traces: Dict[int, torch.Tensor],
) -> float:
    """
    Compute the mean fraction of a token's routed experts that are GPU-resident.

    This per-selection hit rate counts individual experts. Note that it is
    maximised exactly by marginal frequency top-k, since it equals the summed
    marginal counts of the resident experts; it does not reward keeping
    co-firing experts together. Use :func:`token_resident_rate` for the metric
    that actually reflects CPU-path avoidance.

    Args:
        mask: ``(num_layers, num_experts)`` boolean GPU-residency mask.
        token_traces: Mapping ``layer_idx -> (num_tokens, top_k)`` routed ids.

    Returns:
        Mean per-selection GPU hit rate in ``[0.0, 1.0]``. Returns ``0.0`` when
        there are no valid routed selections.
    """
    total_hits = 0
    total_selections = 0
    num_layers, num_experts = mask.shape

    for layer_idx, traces in token_traces.items():
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        ids = traces.to(device="cpu", dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        valid = (ids >= 0) & (ids < num_experts)
        layer_mask = mask[layer_idx]
        # Gather residency only for valid ids; clamp to keep indexing in range.
        resident = layer_mask[ids.clamp(min=0, max=num_experts - 1)]
        total_hits += int((resident & valid).sum().item())
        total_selections += int(valid.sum().item())

    if total_selections == 0:
        return 0.0
    return total_hits / total_selections


def token_resident_rate(
    mask: torch.Tensor,
    token_traces: Dict[int, torch.Tensor],
) -> float:
    """
    Compute the fraction of tokens whose routed experts are *all* GPU-resident.

    This is the placement-quality metric that reflects hybrid throughput: a
    token only skips the slow CPU expert path when *every* one of its routed
    experts is resident on the GPU. A single non-resident expert forces the CPU
    path for that whole token, so keeping co-firing experts together (rather
    than scattering individually-hot ones) is what raises this rate.

    Unlike :func:`gpu_hit_rate`, this metric rewards co-activation-aware
    placement and is not maximised by marginal frequency top-k.

    Args:
        mask: ``(num_layers, num_experts)`` boolean GPU-residency mask.
        token_traces: Mapping ``layer_idx -> (num_tokens, top_k)`` routed ids.

    Returns:
        Mean fraction of fully-resident tokens in ``[0.0, 1.0]``. Returns ``0.0``
        when there are no valid tokens.
    """
    total_tokens = 0
    fully_resident = 0
    num_layers, num_experts = mask.shape

    for layer_idx, traces in token_traces.items():
        if layer_idx < 0 or layer_idx >= num_layers:
            continue
        ids = traces.to(device="cpu", dtype=torch.long)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        layer_mask = mask[layer_idx]
        for token_ids in ids:
            valid = token_ids[(token_ids >= 0) & (token_ids < num_experts)]
            if valid.numel() == 0:
                continue
            total_tokens += 1
            if bool(layer_mask[valid].all()):
                fully_resident += 1

    if total_tokens == 0:
        return 0.0
    return fully_resident / total_tokens
