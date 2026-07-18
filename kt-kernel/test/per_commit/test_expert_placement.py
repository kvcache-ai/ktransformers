"""Tests for co-activation-aware GPU expert placement.

These tests are pure Python/torch and do not touch the compiled ``kt_kernel_ext``
extension, so they run in CPU CI. The placement module is imported directly from
the ``python`` directory (added to ``sys.path``) to avoid importing the package
``__init__``, which pulls in the native extension.

The central claim under test: when experts fire in correlated clusters, keeping a
cluster together on the GPU yields a higher per-token GPU hit rate than selecting
the globally most-frequent experts independently. ``test_coactivation_beats_frequency``
constructs a trace with a planted cluster and asserts exactly that.
"""

import os
import sys
import unittest

import torch

# Register this test for CPU CI.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
import expert_placement

build_coactivation_matrix = expert_placement.build_coactivation_matrix
plan_gpu_expert_placement = expert_placement.plan_gpu_expert_placement
gpu_hit_rate = expert_placement.gpu_hit_rate
token_resident_rate = expert_placement.token_resident_rate
EmaHotnessTracker = expert_placement.EmaHotnessTracker


def _frequency_mask(activation_freq: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Per-layer frequency top-k, the baseline the new planner must beat."""
    num_layers, num_experts = activation_freq.shape
    budget = max(0, min(int(num_gpu_experts), num_experts))
    mask = torch.zeros(num_layers, num_experts, dtype=torch.bool)
    if budget == 0:
        return mask
    for layer_idx in range(num_layers):
        top = torch.topk(activation_freq[layer_idx], k=budget, largest=True, sorted=False).indices
        mask[layer_idx, top] = True
    return mask


class TestCoactivationMatrix(unittest.TestCase):
    def test_diagonal_holds_marginal_counts(self):
        """The diagonal must equal each expert's per-token activation count."""
        # Expert 0 fires in 3 tokens, expert 1 in 2, expert 2 in 1.
        traces = torch.tensor([[0, 1], [0, 1], [0, 2]])
        coact = build_coactivation_matrix(traces, num_experts=3)
        self.assertEqual(coact[0, 0].item(), 3.0)
        self.assertEqual(coact[1, 1].item(), 2.0)
        self.assertEqual(coact[2, 2].item(), 1.0)

    def test_offdiagonal_is_symmetric_cooccurrence(self):
        """Off-diagonal entries count co-firing and are symmetric."""
        traces = torch.tensor([[0, 1], [0, 1], [0, 2]])
        coact = build_coactivation_matrix(traces, num_experts=3)
        self.assertEqual(coact[0, 1].item(), 2.0)  # 0 and 1 co-fire twice
        self.assertEqual(coact[1, 0].item(), 2.0)  # symmetric
        self.assertEqual(coact[0, 2].item(), 1.0)  # 0 and 2 co-fire once
        self.assertEqual(coact[1, 2].item(), 0.0)  # 1 and 2 never co-fire

    def test_padding_ids_are_ignored(self):
        """Out-of-range ids (e.g. -1 padding) must not be counted."""
        traces = torch.tensor([[0, -1], [0, 1]])
        coact = build_coactivation_matrix(traces, num_experts=2)
        self.assertEqual(coact[0, 0].item(), 2.0)
        self.assertEqual(coact[1, 1].item(), 1.0)

    def test_empty_traces_return_zero_matrix(self):
        coact = build_coactivation_matrix(torch.empty(0, dtype=torch.long), num_experts=4)
        self.assertEqual(coact.shape, (4, 4))
        self.assertEqual(coact.sum().item(), 0.0)


class TestCoactivationBeatsFrequency(unittest.TestCase):
    def test_coactivation_beats_frequency(self):
        """The core claim: cluster placement wins on whole-token residency.

        Six experts, top-2 routing, budget of 2 GPU experts for one layer.

        - Cluster {0, 1} co-fires on 10 tokens (their pairwise count is 10).
        - Experts 2 and 3 are individually hotter (12 tokens each) but each
          spreads across four *rotating* cold partners (4..7), so no single
          pair involving 2 or 3 co-fires more than 3 times, and 2 and 3 never
          co-fire with each other.

        Frequency top-2 picks {2, 3} (highest marginals: 12, 12 vs 10 each for
        the pair). But 2 and 3 never appear in the same token, so *no* token is
        ever fully resident: every one of their tokens still hits the CPU path.
        Co-activation seeds from the strongest co-firing pair {0, 1} and keeps
        them together, so all 10 of its tokens are fully GPU-resident. On the
        whole-token metric that governs CPU-path avoidance, cluster placement
        wins decisively.
        """
        tokens = []
        tokens += [[0, 1]] * 10  # tight co-firing cluster, pairwise count 10
        # Expert 2 is hot (marginal 12) but spread over rotating cold partners.
        for partner in (4, 5, 6, 7):
            tokens += [[2, partner]] * 3
        # Expert 3 is hot (marginal 12) but spread over the same cold partners,
        # and never co-fires with expert 2.
        for partner in (4, 5, 6, 7):
            tokens += [[3, partner]] * 3
        traces = {0: torch.tensor(tokens)}

        num_experts = 8
        # Marginal counts for the frequency baseline: 2 and 3 are hottest.
        activation_freq = torch.zeros(1, num_experts)
        flat = torch.tensor(tokens).flatten()
        for e in range(num_experts):
            activation_freq[0, e] = (flat == e).sum()

        freq_mask = _frequency_mask(activation_freq, num_gpu_experts=2)
        coact_mask = plan_gpu_expert_placement(
            num_layers=1,
            num_experts=num_experts,
            num_gpu_experts=2,
            token_traces=traces,
            strategy="cluster",
        )

        # Sanity: frequency really did pick the individually-hot, non-co-firing pair.
        self.assertTrue(bool(freq_mask[0, 2]) and bool(freq_mask[0, 3]))
        # Co-activation keeps the co-firing cluster together.
        self.assertTrue(bool(coact_mask[0, 0]) and bool(coact_mask[0, 1]))

        # Whole-token residency: the metric that reflects CPU-path avoidance.
        freq_tok = token_resident_rate(freq_mask, traces)
        coact_tok = token_resident_rate(coact_mask, traces)
        self.assertEqual(freq_tok, 0.0)  # 2 and 3 never co-fire -> no full token
        self.assertGreater(coact_tok, freq_tok)

        # And the per-selection metric confirms frequency's known advantage there,
        # documenting exactly why whole-token residency is the right objective.
        self.assertGreaterEqual(gpu_hit_rate(freq_mask, traces), gpu_hit_rate(coact_mask, traces))


class TestCoverageStrategy(unittest.TestCase):
    def test_coverage_beats_frequency_and_cluster_with_stable_core(self):
        """Your scenario: 6 stable core + 2 rotating slots, plus a decoy.

        Realistic Qwen-style routing: top-8 of many experts, where 6 experts
        form a nearly-always-on core and 2 slots rotate. A token is a hit only
        when *all 8* of its experts are resident. Budget is 10 GPU experts, so
        we can afford the 6-core plus 4 rotating experts (two rotating pairs).

        Layout:
        - Core = {0,1,2,3,4,5}, present in every token.
        - Rotating pair A = {6,7} on 30 tokens.
        - Rotating pair B = {8,9} on 25 tokens.
        - Decoy expert 10 is the *single hottest* rotating expert (30 tokens),
          but its 8th slot is a *unique* cold expert each time (11,12,...), so
          completing any decoy token is impossible within budget.

        Frequency top-10 spends its slots on the highest marginals: the 6 core,
        then 6, 7, 10 (all 30) and only *one* of {8,9} (25). That wastes a slot
        on the un-completable decoy 10 and breaks pair B (only 8 resident, 9
        evicted), so pair B's tokens all miss. Coverage reduces to residuals,
        sees {6,7} and {8,9} complete many tokens per slot while {10, cold}
        completes only ~0.5, and packs both real pairs.
        """
        core = [0, 1, 2, 3, 4, 5]
        tokens = []
        tokens += [core + [6, 7]] * 30  # rotating pair A
        tokens += [core + [8, 9]] * 25  # rotating pair B
        # Decoy: expert 10 is the hottest rotating expert but its partner is a
        # unique cold expert every time, so its tokens can never be completed.
        for i in range(30):
            tokens += [core + [10, 11 + i]]  # 11..40 each appear once
        traces = {0: torch.tensor(tokens)}

        num_experts = 48
        budget = 10

        activation_freq = torch.zeros(1, num_experts)
        flat = torch.tensor(tokens).flatten()
        for e in range(num_experts):
            activation_freq[0, e] = (flat == e).sum()

        freq_mask = _frequency_mask(activation_freq, num_gpu_experts=budget)
        cluster_mask = plan_gpu_expert_placement(
            num_layers=1, num_experts=num_experts, num_gpu_experts=budget, token_traces=traces, strategy="cluster"
        )
        coverage_mask = plan_gpu_expert_placement(
            num_layers=1, num_experts=num_experts, num_gpu_experts=budget, token_traces=traces, strategy="coverage"
        )

        freq_cov = token_resident_rate(freq_mask, traces)
        cluster_cov = token_resident_rate(cluster_mask, traces)
        coverage_cov = token_resident_rate(coverage_mask, traces)

        # Coverage must win the whole-token metric outright.
        self.assertGreater(coverage_cov, freq_cov)
        self.assertGreater(coverage_cov, cluster_cov)
        # It should keep the core plus both real rotating pairs.
        for e in core + [6, 7, 8, 9]:
            self.assertTrue(bool(coverage_mask[0, e]), f"expert {e} should be resident")
        # And it should not waste a slot on the un-completable decoy.
        self.assertFalse(bool(coverage_mask[0, 10]), "decoy expert 10 should not be placed")
        # All 55 rotating-pair tokens are covered; the 30 decoy tokens cannot be.
        self.assertAlmostEqual(coverage_cov, 55 / 85, places=6)

    def test_coverage_takes_all_rotating_when_budget_allows(self):
        """When budget covers core + every rotating variant, coverage hits 100%."""
        core = [0, 1, 2, 3]
        tokens = []
        tokens += [core + [4, 5]] * 10
        tokens += [core + [6, 7]] * 10
        traces = {0: torch.tensor(tokens)}
        # Budget 8 = 4 core + 4 rotating -> everything fits.
        mask = plan_gpu_expert_placement(
            num_layers=1, num_experts=16, num_gpu_experts=8, token_traces=traces, strategy="coverage"
        )
        self.assertEqual(token_resident_rate(mask, traces), 1.0)


class TestPlanFallbacks(unittest.TestCase):
    def test_falls_back_to_frequency_without_traces(self):
        """With no traces, planning matches per-layer frequency top-k exactly."""
        activation_freq = torch.tensor([[0.1, 0.9, 0.3, 0.7], [0.5, 0.2, 0.8, 0.4]])
        mask = plan_gpu_expert_placement(
            num_layers=2,
            num_experts=4,
            num_gpu_experts=2,
            token_traces=None,
            activation_freq=activation_freq,
        )
        expected = _frequency_mask(activation_freq, num_gpu_experts=2)
        self.assertTrue(torch.equal(mask, expected))

    def test_per_layer_trace_absence_falls_back(self):
        """A layer missing from the trace map uses activation_freq for that layer."""
        traces = {0: torch.tensor([[0, 1], [0, 1]])}
        activation_freq = torch.tensor([[9.0, 9.0, 0.0, 0.0], [0.0, 0.0, 5.0, 6.0]])
        mask = plan_gpu_expert_placement(
            num_layers=2,
            num_experts=4,
            num_gpu_experts=2,
            token_traces=traces,
            activation_freq=activation_freq,
        )
        # Layer 1 has no trace: falls back to frequency top-2 -> experts 2, 3.
        self.assertTrue(bool(mask[1, 2]) and bool(mask[1, 3]))

    def test_zero_budget_returns_empty_mask(self):
        mask = plan_gpu_expert_placement(num_layers=3, num_experts=8, num_gpu_experts=0)
        self.assertEqual(mask.shape, (3, 8))
        self.assertFalse(bool(mask.any()))

    def test_budget_clamped_to_num_experts(self):
        mask = plan_gpu_expert_placement(num_layers=1, num_experts=4, num_gpu_experts=99)
        self.assertEqual(int(mask.sum().item()), 4)

    def test_uniform_fallback_without_any_stats(self):
        """No traces and no freq -> first num_gpu_experts per layer."""
        mask = plan_gpu_expert_placement(num_layers=2, num_experts=5, num_gpu_experts=2)
        self.assertTrue(bool(mask[0, 0]) and bool(mask[0, 1]))
        self.assertFalse(bool(mask[0, 2]))


class TestGpuHitRate(unittest.TestCase):
    def test_all_resident_is_one(self):
        mask = torch.tensor([[True, True, False, False]])
        traces = {0: torch.tensor([[0, 1], [0, 1]])}
        self.assertEqual(gpu_hit_rate(mask, traces), 1.0)

    def test_none_resident_is_zero(self):
        mask = torch.tensor([[False, False, True, True]])
        traces = {0: torch.tensor([[0, 1], [0, 1]])}
        self.assertEqual(gpu_hit_rate(mask, traces), 0.0)

    def test_half_resident(self):
        mask = torch.tensor([[True, False, False, False]])
        traces = {0: torch.tensor([[0, 1]])}  # one hit (0), one miss (1)
        self.assertEqual(gpu_hit_rate(mask, traces), 0.5)

    def test_no_selections_returns_zero(self):
        mask = torch.tensor([[True, False]])
        self.assertEqual(gpu_hit_rate(mask, {}), 0.0)


class TestTokenResidentRate(unittest.TestCase):
    def test_all_tokens_fully_resident(self):
        mask = torch.tensor([[True, True, False, False]])
        traces = {0: torch.tensor([[0, 1], [0, 1]])}
        self.assertEqual(token_resident_rate(mask, traces), 1.0)

    def test_partial_token_is_not_resident(self):
        """A token with one resident and one non-resident expert counts as a miss."""
        mask = torch.tensor([[True, False, False, False]])
        traces = {0: torch.tensor([[0, 1]])}  # expert 1 on CPU -> whole token misses
        self.assertEqual(token_resident_rate(mask, traces), 0.0)

    def test_mixed_tokens(self):
        mask = torch.tensor([[True, True, False, False]])
        traces = {0: torch.tensor([[0, 1], [0, 2]])}  # first fully resident, second not
        self.assertEqual(token_resident_rate(mask, traces), 0.5)

    def test_no_tokens_returns_zero(self):
        mask = torch.tensor([[True, True]])
        self.assertEqual(token_resident_rate(mask, {}), 0.0)


class TestEmaHotnessTracker(unittest.TestCase):
    """Online EMA tracker: adapts placement to the live workload.

    The central claim: when the workload shifts, an online tracker that follows
    recent activation keeps a higher whole-token GPU residency than a mask frozen
    from the earlier (now stale) traffic. ``test_adapts_to_workload_shift`` plants
    a shift and asserts exactly that.
    """

    def test_mask_shape_and_budget(self):
        t = EmaHotnessTracker(num_layers=3, num_experts=8, num_gpu_experts=4)
        t.observe(0, torch.tensor([0, 1, 2, 3]))
        mask = t.mask()
        self.assertEqual(mask.shape, (3, 8))
        self.assertEqual(int(mask[0].sum()), 4)

    def test_hot_experts_selected(self):
        """Experts that fire every token outrank experts that never fire."""
        t = EmaHotnessTracker(num_layers=1, num_experts=6, num_gpu_experts=2)
        for _ in range(10):
            t.observe(0, torch.tensor([2, 5]))
        mask = t.mask()
        self.assertTrue(bool(mask[0, 2]) and bool(mask[0, 5]))
        self.assertEqual(int(mask[0].sum()), 2)

    def test_padding_ids_ignored(self):
        t = EmaHotnessTracker(num_layers=1, num_experts=4, num_gpu_experts=2)
        t.observe(0, torch.tensor([0, -1, 99, 1]))  # -1 and 99 are out of range
        scores = t.scores()
        self.assertGreater(float(scores[0, 0]), 0.0)
        self.assertGreater(float(scores[0, 1]), 0.0)
        self.assertEqual(float(scores[0, 2]), 0.0)
        self.assertEqual(float(scores[0, 3]), 0.0)

    def test_out_of_range_layer_ignored(self):
        t = EmaHotnessTracker(num_layers=2, num_experts=4, num_gpu_experts=2)
        t.observe(5, torch.tensor([0, 1]))  # no such layer; must not raise
        self.assertEqual(float(t.scores().sum()), 0.0)

    def test_recency_beats_staleness(self):
        """Lower decay weights recent tokens more, so a switch flips the ranking."""
        t = EmaHotnessTracker(num_layers=1, num_experts=4, num_gpu_experts=1, decay=0.5)
        for _ in range(5):
            t.observe(0, torch.tensor([0]))  # expert 0 hot for a while
        for _ in range(5):
            t.observe(0, torch.tensor([1]))  # workload switches to expert 1
        mask = t.mask()
        self.assertTrue(bool(mask[0, 1]))
        self.assertFalse(bool(mask[0, 0]))

    def test_warm_start_matches_frequency_top_k(self):
        """Before any observation, a freq-warm-started mask equals frequency top-k."""
        freq = torch.tensor([[10.0, 1.0, 5.0, 0.0]])  # hottest: expert 0, then 2
        t = EmaHotnessTracker(num_layers=1, num_experts=4, num_gpu_experts=2, activation_freq=freq)
        mask = t.mask()
        expected = torch.zeros(1, 4, dtype=torch.bool)
        expected[0, torch.topk(freq[0], k=2).indices] = True
        self.assertTrue(bool((mask == expected).all()))

    def test_observe_batch_order_preserved(self):
        """observe_batch consumes rows in order; equivalent to sequential observe."""
        traces = {0: torch.tensor([[0, 1], [0, 2], [0, 3]])}
        batch = EmaHotnessTracker(1, 4, 2, decay=0.8)
        batch.observe_batch(traces)
        seq = EmaHotnessTracker(1, 4, 2, decay=0.8)
        for row in traces[0]:
            seq.observe(0, row)
        self.assertTrue(bool(torch.allclose(batch.scores(), seq.scores())))

    def test_adapts_to_workload_shift(self):
        """Online tracker beats a mask frozen on stale traffic after a shift.

        Phase 1 routes to experts {0,1}; phase 2 switches to {2,3}. A static mask
        planned on phase-1 traffic misses every phase-2 token. The online tracker,
        having observed the shift, places {2,3} and stays fully resident.
        """
        num_experts, budget = 4, 2
        phase1 = torch.tensor([[0, 1]] * 8)
        phase2 = torch.tensor([[2, 3]] * 8)

        # Static placement fit on phase-1 traffic only.
        static_mask = plan_gpu_expert_placement(1, num_experts, budget, token_traces={0: phase1}, strategy="cluster")
        static_rate = token_resident_rate(static_mask, {0: phase2})

        # Online tracker observes phase 1 then phase 2, then emits its mask.
        tracker = EmaHotnessTracker(1, num_experts, budget, decay=0.7)
        tracker.observe_batch({0: phase1})
        tracker.observe_batch({0: phase2})
        online_rate = token_resident_rate(tracker.mask(), {0: phase2})

        self.assertEqual(static_rate, 0.0)  # stale mask misses the new workload
        self.assertEqual(online_rate, 1.0)  # adaptive mask fully covers it
        self.assertGreater(online_rate, static_rate)

    def test_invalid_decay_rejected(self):
        with self.assertRaises(ValueError):
            EmaHotnessTracker(1, 4, 2, decay=1.0)
        with self.assertRaises(ValueError):
            EmaHotnessTracker(1, 4, 2, decay=-0.1)

    def test_zero_budget_empty_mask(self):
        t = EmaHotnessTracker(1, 4, 0)
        t.observe(0, torch.tensor([0, 1]))
        self.assertEqual(int(t.mask().sum()), 0)


if __name__ == "__main__":
    unittest.main()
