"""Pure-Python tests for tiered MoE weight management helpers."""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np


MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python", "utils", "weight_provider.py")
SPEC = importlib.util.spec_from_file_location("weight_provider", MODULE_PATH)
weight_provider = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(weight_provider)

TieredWeightProvider = weight_provider.TieredWeightProvider
backend_supports_tiered_strategy = weight_provider.backend_supports_tiered_strategy
create_residency_policy = weight_provider.create_residency_policy
resolve_backend_weight_strategy = weight_provider.resolve_backend_weight_strategy
resolve_weight_strategy = weight_provider.resolve_weight_strategy
compute_max_tier0_experts = weight_provider.compute_max_tier0_experts
resolve_auto_tier0_budget_bytes = weight_provider.resolve_auto_tier0_budget_bytes
get_cgroup_memory_limit_current_bytes = weight_provider.get_cgroup_memory_limit_current_bytes
get_available_ram_bytes = weight_provider.get_available_ram_bytes
get_total_ram_bytes = weight_provider.get_total_ram_bytes
constrain_tier0_memory_bytes = weight_provider.constrain_tier0_memory_bytes
normalize_residency_policy_name = weight_provider.normalize_residency_policy_name


class DummyMoe:
    """Minimal MOE stub for promotion/demotion tests."""

    def __init__(self):
        self.promoted = set()
        self.promote_calls = []
        self.demote_calls = []

    def promote_expert(self, expert_id: int):
        self.promoted.add(expert_id)
        self.promote_calls.append(expert_id)

    def demote_expert(self, expert_id: int):
        self.promoted.discard(expert_id)
        self.demote_calls.append(expert_id)

    def is_expert_promoted(self, expert_id: int) -> bool:
        return expert_id in self.promoted


class DummyRegion:
    """Simple mmap-region stub that records prefetch requests."""

    def __init__(self):
        self.prefetch_calls = 0

    def prefetch(self):
        self.prefetch_calls += 1


def test_resolve_weight_strategy_auto_switches_between_legacy_and_tiered():
    """Auto mode should choose a concrete strategy from the RAM estimate."""
    tiered, model_bytes, total_ram = resolve_weight_strategy(
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=32 * 1024**3,
    )
    assert tiered == "tiered"
    assert model_bytes > 0
    assert total_ram == 64 * 1024**3

    legacy, _, _ = resolve_weight_strategy(
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=512 * 1024**3,
        available_ram_bytes=512 * 1024**3,
    )
    assert legacy == "legacy"


def test_backend_strategy_support_is_backend_specific():
    """Only backends with mmap support should expose tiered semantics."""
    assert backend_supports_tiered_strategy("LLAMAFILE") is True
    assert backend_supports_tiered_strategy("AMXINT4") is True
    assert backend_supports_tiered_strategy("AMXINT8") is True
    assert backend_supports_tiered_strategy("MOE_INT4") is True
    assert backend_supports_tiered_strategy("MOE_INT8") is True
    assert backend_supports_tiered_strategy("BF16") is True
    assert backend_supports_tiered_strategy("RAWINT4") is False


def test_amx_backends_can_resolve_into_tiered_mode():
    """AMX mmap-capable backends should resolve auto into tiered mode under memory pressure."""
    resolved, model_bytes, total_ram = resolve_backend_weight_strategy(
        "AMXINT4",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert resolved == "tiered"
    assert model_bytes > 0
    assert total_ram == 64 * 1024**3

    moe_resolved, moe_model_bytes, _ = resolve_backend_weight_strategy(
        "MOE_INT8",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert moe_resolved == "tiered"
    assert moe_model_bytes > 0

    bf16_resolved, bf16_model_bytes, _ = resolve_backend_weight_strategy(
        "BF16",
        "auto",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=256 * 1024**3,
        available_ram_bytes=64 * 1024**3,
    )
    assert bf16_resolved == "tiered"
    assert bf16_model_bytes > moe_model_bytes


def test_non_mmap_backends_still_fall_back_to_legacy():
    """Backends without file-backed weight support must stay on resident loading."""
    forced, _, _ = resolve_backend_weight_strategy(
        "RAWINT4",
        "tiered",
        num_layers=60,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        total_ram_bytes=64 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    assert forced == "legacy"


def test_auto_tier0_budget_shrinks_as_memory_pressure_rises():
    """Auto Tier0 budget should reserve less NUMA hotset when the model pressure is high."""
    low_pressure = resolve_auto_tier0_budget_bytes(
        model_bytes=40 * 1024**3,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=96 * 1024**3,
    )
    high_pressure = resolve_auto_tier0_budget_bytes(
        model_bytes=200 * 1024**3,
        total_ram_bytes=128 * 1024**3,
        available_ram_bytes=96 * 1024**3,
    )

    assert low_pressure > high_pressure > 0


def test_cgroup_v2_memory_limit_is_detected_from_process_scope():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        scoped = root / "system.slice" / "run-test.scope"
        scoped.mkdir(parents=True)
        (scoped / "memory.max").write_text(str(64 * 1024**3))
        (scoped / "memory.current").write_text(str(12 * 1024**3))

        limit, current = get_cgroup_memory_limit_current_bytes(
            proc_self_cgroup_text="0::/system.slice/run-test.scope\n",
            mountinfo_text="29 23 0:28 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n",
            mount_root_override=root,
        )

    assert limit == 64 * 1024**3
    assert current == 12 * 1024**3


def test_ram_queries_prefer_cgroup_over_host_memory():
    original = weight_provider.get_cgroup_memory_limit_current_bytes
    try:
        weight_provider.get_cgroup_memory_limit_current_bytes = lambda **_: (
            64 * 1024**3,
            12 * 1024**3,
        )
        assert get_total_ram_bytes() == 64 * 1024**3
        assert get_available_ram_bytes() == 52 * 1024**3
    finally:
        weight_provider.get_cgroup_memory_limit_current_bytes = original


def test_explicit_tier0_budget_is_clamped_to_effective_scope():
    constrained = constrain_tier0_memory_bytes(
        60 * 1024**3,
        available_ram_bytes=32 * 1024**3,
    )
    assert constrained == 28 * 1024**3


def test_provider_skips_gpu_experts_for_prefetch_and_promotion():
    """GPU-routed experts must not consume Tier0 promotion or mmap prefetch budget."""
    provider = TieredWeightProvider(
        num_experts=4,
        num_layers=2,
        max_tier0_experts=2,
        residency_policy="baseline",
    )
    provider.start_promotion_thread = lambda: None

    moe0 = DummyMoe()
    moe1 = DummyMoe()
    provider.register_moe(0, moe0, gpu_experts_mask=np.array([False, True, False, False], dtype=np.bool_))
    provider.register_moe(1, moe1, gpu_experts_mask=np.array([True, False, False, False], dtype=np.bool_))

    regions = {}
    for expert_id in range(4):
        region = DummyRegion()
        regions[expert_id] = region
        provider.register_mmap_region(0, "gate", expert_id, region)

    provider.prefetch_layer(0, np.array([[0, 1, 2, -1]], dtype=np.int64))
    assert regions[0].prefetch_calls == 1
    assert regions[1].prefetch_calls == 0
    assert regions[2].prefetch_calls == 1
    assert regions[3].prefetch_calls == 0

    provider.record_activations(0, np.array([[1, 2, 2, 3]], dtype=np.int64))
    policy0 = provider.policy_by_layer[0]
    assert policy0.counts[1] == 0.0
    assert policy0.counts[2] > 0.0
    assert policy0.counts[3] > 0.0

    policy1 = provider.policy_by_layer[1]
    policy0.counts[:] = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
    policy0._resident = [0, 1]
    policy1.counts[:] = np.array([0.9, 0.8, 0.1, 0.0], dtype=np.float64)
    policy1._resident = [0, 1]
    provider._maybe_promote()

    assert moe0.promote_calls == [0]
    assert moe1.promote_calls == [1]


def test_provider_prefetches_all_registered_regions_for_an_expert():
    """TP-aware registration may attach multiple mmap slices to the same expert."""
    provider = TieredWeightProvider(num_experts=2, num_layers=1, max_tier0_experts=1)

    region_a = DummyRegion()
    region_b = DummyRegion()
    provider.register_mmap_region(0, "gate", 0, region_a)
    provider.register_mmap_region(0, "gate", 0, region_b)

    provider.prefetch_layer(0, np.array([[0]], dtype=np.int64))

    assert region_a.prefetch_calls == 1
    assert region_b.prefetch_calls == 1


def test_zero_tier0_budget_disables_background_promotion():
    """A zero Tier0 budget must disable provider-managed promotion entirely."""
    assert (
        compute_max_tier0_experts(
            tier0_memory_bytes=0,
            num_layers=60,
            num_experts=256,
            hidden_size=7168,
            intermediate_size=2048,
        )
        == 0
    )

    provider = TieredWeightProvider(num_experts=4, num_layers=1, max_tier0_experts=0)
    start_calls = []
    provider.start_promotion_thread = lambda: start_calls.append(True)

    moe = DummyMoe()
    provider.register_moe(0, moe, gpu_experts_mask=np.zeros(4, dtype=np.bool_))
    provider._maybe_promote()

    assert start_calls == []
    assert moe.promote_calls == []
    assert moe.demote_calls == []


def test_policy_factory_supports_all_configured_algorithms():
    expected = {
        "baseline": "baseline",
        "default": "baseline",
        "current_ema": "baseline",
        "current": "baseline",
        "ema": "baseline",
        "ema_hotset": "baseline",
        "lru": "lru",
        "2q": "2q",
        "twoq": "2q",
        "two-q": "2q",
        "slru": "slru",
        "sieve": "sieve",
        "s3fifo": "s3fifo",
        "s3-fifo": "s3fifo",
        "w_tinylfu": "w_tinylfu",
        "wtinylfu": "w_tinylfu",
        "w-tinylfu": "w_tinylfu",
    }

    for raw_name, normalized in expected.items():
        assert normalize_residency_policy_name(raw_name) == normalized
        policy = create_residency_policy(raw_name, num_experts=8, capacity=3)
        assert policy.policy_name == normalized


def test_lru_policy_evicts_oldest_resident():
    policy = create_residency_policy("lru", num_experts=8, capacity=2)
    policy.record_accesses([0, 1])
    policy.record_accesses([0])
    policy.record_accesses([2])
    snapshot = policy.snapshot()

    assert snapshot["resident"] == [0, 2]
    assert snapshot["stats"]["hits"] == 1
    assert snapshot["stats"]["misses"] == 3


def test_2q_policy_promotes_second_hit_into_am_queue():
    policy = create_residency_policy("2q", num_experts=8, capacity=4, config={"2q_a1_ratio": 0.5})
    policy.record_accesses([0, 1, 0])
    snapshot = policy.snapshot()

    assert list(policy._a1.keys()) == [1]
    assert list(policy._am.keys()) == [0]
    assert snapshot["resident"] == [1, 0]
    assert snapshot["stats"]["hits"] == 1


def test_2q_policy_evicts_a1_before_am():
    policy = create_residency_policy("2q", num_experts=8, capacity=3, config={"2q_a1_ratio": 1 / 3})
    policy.record_accesses([0, 0, 1, 2, 3])
    snapshot = policy.snapshot()

    assert list(policy._a1.keys()) == [3]
    assert list(policy._am.keys()) == [0]
    assert 0 in snapshot["resident"]
    assert 3 in snapshot["resident"]
    assert 1 not in snapshot["resident"]


def test_slru_policy_promotes_second_hit_into_protected_segment():
    policy = create_residency_policy("slru", num_experts=8, capacity=4, config={"slru_protected_ratio": 0.5})
    policy.record_accesses([0, 1, 0])
    snapshot = policy.snapshot()

    assert 0 in snapshot["resident"]
    assert 1 in snapshot["resident"]
    assert snapshot["stats"]["hits"] == 1


def test_slru_protected_overflow_demotes_oldest_protected_entry():
    policy = create_residency_policy("slru", num_experts=8, capacity=3, config={"slru_protected_ratio": 2 / 3})
    policy.record_accesses([0, 0, 1, 1])
    policy.record_accesses([2, 2])
    snapshot = policy.snapshot()

    assert list(policy._protected.keys()) == [1, 2]
    assert list(policy._probationary.keys()) == [0]
    assert snapshot["resident"] == [0, 1, 2]


def test_sieve_policy_gives_referenced_items_a_second_chance():
    policy = create_residency_policy("sieve", num_experts=8, capacity=2)
    policy.record_accesses([0, 1, 0, 2])
    snapshot = policy.snapshot()

    assert 0 in snapshot["resident"]
    assert 2 in snapshot["resident"]
    assert 1 not in snapshot["resident"]


def test_sieve_policy_inserts_new_entries_at_head_like_reference_impl():
    policy = create_residency_policy("sieve", num_experts=8, capacity=3)
    policy.record_accesses([0, 1, 2])
    snapshot = policy.snapshot()

    # libCacheSim's Sieve prepends new objects to the queue head.
    assert snapshot["resident"] == [2, 1, 0]


def test_s3fifo_uses_ghost_references_to_bypass_small_queue():
    policy = create_residency_policy(
        "s3fifo",
        num_experts=8,
        capacity=2,
        config={"s3fifo_small_ratio": 0.5, "s3fifo_ghost_ratio": 1.0, "s3fifo_move_to_main_threshold": 1},
    )
    policy.record_accesses([0, 1, 2, 0])

    assert policy._resident_queue.get(0) == "main"
    assert 0 in policy.snapshot()["resident"]


def test_s3fifo_main_queue_entry_gets_second_chance_before_eviction():
    policy = create_residency_policy(
        "s3fifo",
        num_experts=8,
        capacity=2,
        config={"s3fifo_small_ratio": 0.5, "s3fifo_ghost_ratio": 1.0, "s3fifo_move_to_main_threshold": 1},
    )
    policy.record_accesses([0, 0, 1, 2, 1])
    snapshot = policy.snapshot()

    assert policy._resident_queue.get(1) == "main"
    assert 1 in snapshot["resident"]
    assert 0 not in snapshot["resident"]


def test_w_tinylfu_prefers_frequent_candidate_over_probationary_victim():
    policy = create_residency_policy(
        "w_tinylfu",
        num_experts=8,
        capacity=2,
        config={"wtinylfu_window_ratio": 0.5, "wtinylfu_protected_ratio": 0.5, "wtinylfu_decay_interval": 100},
    )
    policy.record_accesses([0, 1, 1, 2])
    snapshot = policy.snapshot()

    assert 1 in snapshot["resident"]
    assert 0 not in snapshot["resident"]


def test_w_tinylfu_promotes_probation_hit_and_demotes_oldest_protected():
    policy = create_residency_policy(
        "w_tinylfu",
        num_experts=8,
        capacity=3,
        config={"wtinylfu_window_ratio": 1 / 3, "wtinylfu_protected_ratio": 0.5, "wtinylfu_decay_interval": 100},
    )
    policy.record_accesses([0, 1, 0, 2, 1])
    snapshot = policy.snapshot()

    assert list(policy._window.keys()) == [2]
    assert list(policy._probationary.keys()) == [0]
    assert list(policy._protected.keys()) == [1]
    assert snapshot["resident"] == [2, 0, 1]


def test_provider_respects_residency_policy_env_override():
    original = os.environ.get("KT_RESIDENCY_POLICY")
    try:
        os.environ["KT_RESIDENCY_POLICY"] = "sieve"
        provider = TieredWeightProvider(num_experts=8, num_layers=1, max_tier0_experts=2, residency_policy="lru")
        provider.start_promotion_thread = lambda: None
        provider.register_moe(0, DummyMoe(), gpu_experts_mask=np.zeros(8, dtype=np.bool_))
        assert provider.residency_policy_name == "sieve"
        assert provider.policy_by_layer[0].policy_name == "sieve"
    finally:
        if original is None:
            os.environ.pop("KT_RESIDENCY_POLICY", None)
        else:
            os.environ["KT_RESIDENCY_POLICY"] = original
