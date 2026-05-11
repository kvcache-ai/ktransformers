"""Golden-trace behavior checks for non-baseline residency policies.

These tests intentionally verify step-by-step queue evolution instead of only
asserting the final resident set. The expected states are derived from the
reference algorithms we aligned to:

- Caffeine `SegmentedLruPolicy`
- Caffeine `WindowTinyLfuPolicy`
- libCacheSim `S3FIFO`
"""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from pathlib import Path


WEIGHT_PROVIDER_PATH = Path(__file__).resolve().parents[2] / "python" / "utils" / "weight_provider.py"
SPEC = importlib.util.spec_from_file_location("weight_provider", WEIGHT_PROVIDER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Failed to load weight_provider from {WEIGHT_PROVIDER_PATH}")
weight_provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weight_provider)

create_residency_policy = weight_provider.create_residency_policy


def snapshot_trace(policy, accesses):
    trace = []
    for event in accesses:
        policy.record_accesses(event)
        snap = policy.snapshot()
        trace.append(
            {
                "resident": snap["resident"],
                "stats": snap["stats"],
            }
        )
    return trace


def reference_2q_trace(*, capacity, a1_ratio, accesses):
    if capacity <= 1:
        a1_capacity = capacity
        am_capacity = 0
    else:
        a1_capacity = max(1, min(capacity - 1, int(round(capacity * a1_ratio))))
        am_capacity = capacity - a1_capacity

    a1 = OrderedDict()
    am = OrderedDict()
    trace = []
    for event in accesses:
        for key in event:
            if key in am:
                am.move_to_end(key)
            elif key in a1:
                a1.pop(key)
                if am_capacity > 0:
                    am[key] = None
                else:
                    a1[key] = None
            else:
                a1[key] = None

            while len(a1) > a1_capacity:
                a1.popitem(last=False)
            while len(am) > am_capacity:
                am.popitem(last=False)
            while len(a1) + len(am) > capacity:
                if a1:
                    a1.popitem(last=False)
                elif am:
                    am.popitem(last=False)
                else:
                    break

        trace.append(
            {
                "resident": list(a1.keys()) + list(am.keys()),
                "a1": list(a1.keys()),
                "am": list(am.keys()),
            }
        )
    return trace


def reference_segmented_lru_trace(*, capacity, protected_ratio, accesses):
    if capacity <= 1:
        max_protected = 0
    else:
        max_protected = max(1, min(capacity - 1, int(round(capacity * protected_ratio))))

    probationary = OrderedDict()
    protected = OrderedDict()
    trace = []
    for event in accesses:
        for key in event:
            if key in protected:
                protected.move_to_end(key)
            elif key in probationary:
                if len(protected) >= max_protected and max_protected > 0:
                    demote = next(iter(protected))
                    protected.pop(demote)
                    probationary[demote] = None
                probationary.pop(key)
                if max_protected > 0:
                    protected[key] = None
                else:
                    probationary[key] = None
            else:
                probationary[key] = None
                if len(probationary) + len(protected) > capacity:
                    probationary.popitem(last=False)
        trace.append(
            {
                "resident": list(probationary.keys()) + list(protected.keys()),
                "probationary": list(probationary.keys()),
                "protected": list(protected.keys()),
            }
        )
    return trace


def reference_window_tiny_lfu_trace(*, capacity, window_ratio, protected_ratio, accesses):
    if capacity <= 1:
        max_window = capacity
        max_main = 0
    else:
        max_window = max(1, min(capacity - 1, int(round(capacity * window_ratio))))
        max_main = capacity - max_window
    max_protected = max(0, min(max_main, int(max_main * protected_ratio)))

    window = OrderedDict()
    probationary = OrderedDict()
    protected = OrderedDict()
    freq = {}
    trace = []

    def admit(candidate, victim):
        return freq.get(candidate, 0) >= freq.get(victim, 0)

    for event in accesses:
        for key in event:
            freq[key] = freq.get(key, 0) + 1
            if key in window:
                window.move_to_end(key)
            elif key in probationary:
                probationary.pop(key)
                protected[key] = None
                if len(protected) > max_protected:
                    demote, _ = protected.popitem(last=False)
                    probationary[demote] = None
            elif key in protected:
                protected.move_to_end(key)
            else:
                window[key] = None
                if len(window) > max_window:
                    candidate, _ = window.popitem(last=False)
                    probationary[candidate] = None
                    total = len(window) + len(probationary) + len(protected)
                    if total > capacity:
                        victim = next(iter(probationary))
                        evict = victim if admit(candidate, victim) else candidate
                        probationary.pop(evict, None)
        trace.append(
            {
                "resident": list(window.keys()) + list(probationary.keys()) + list(protected.keys()),
                "window": list(window.keys()),
                "probationary": list(probationary.keys()),
                "protected": list(protected.keys()),
            }
        )
    return trace


def test_slru_golden_trace_matches_segmented_lru_behavior():
    policy = create_residency_policy("slru", num_experts=8, capacity=3, config={"slru_protected_ratio": 2 / 3})
    trace = snapshot_trace(policy, [[0], [1], [0], [1], [2], [2], [3]])

    assert trace[0]["resident"] == [0]
    assert trace[1]["resident"] == [0, 1]
    assert trace[2]["resident"] == [1, 0]
    assert trace[3]["resident"] == [0, 1]
    assert trace[4]["resident"] == [2, 0, 1]
    assert trace[5]["resident"] == [0, 1, 2]
    assert trace[6]["resident"] == [3, 1, 2]

    assert list(policy._probationary.keys()) == [3]
    assert list(policy._protected.keys()) == [1, 2]


def test_2q_golden_trace_matches_a1_then_am_flow():
    policy = create_residency_policy("2q", num_experts=8, capacity=4, config={"2q_a1_ratio": 0.5})
    trace = snapshot_trace(policy, [[0], [1], [0], [2], [3], [2], [4]])

    assert trace[0]["resident"] == [0]
    assert trace[1]["resident"] == [0, 1]
    assert trace[2]["resident"] == [1, 0]
    assert trace[3]["resident"] == [1, 2, 0]
    assert trace[4]["resident"] == [2, 3, 0]
    assert trace[5]["resident"] == [3, 0, 2]
    assert trace[6]["resident"] == [3, 4, 0, 2]

    assert list(policy._a1.keys()) == [3, 4]
    assert list(policy._am.keys()) == [0, 2]


def test_s3fifo_golden_trace_matches_small_main_ghost_flow():
    policy = create_residency_policy(
        "s3fifo",
        num_experts=8,
        capacity=3,
        config={
            "s3fifo_small_ratio": 1 / 3,
            "s3fifo_ghost_ratio": 1.0,
            "s3fifo_move_to_main_threshold": 1,
        },
    )
    trace = snapshot_trace(policy, [[0], [0], [1], [2], [1]])

    assert trace[0]["resident"] == [0]
    assert trace[1]["resident"] == [0]
    assert trace[2]["resident"] == [1, 0]
    assert trace[3]["resident"] == [2, 0]
    assert trace[4]["resident"] == [2, 0, 1]

    assert list(policy._small) == [2]
    assert list(policy._main) == [0, 1]
    assert list(policy._ghost.keys()) == []


def test_wtinylfu_golden_trace_matches_window_probation_protected_flow():
    policy = create_residency_policy(
        "w_tinylfu",
        num_experts=8,
        capacity=3,
        config={
            "wtinylfu_window_ratio": 1 / 3,
            "wtinylfu_protected_ratio": 0.5,
            "wtinylfu_decay_interval": 100,
        },
    )
    trace = snapshot_trace(policy, [[0], [1], [0], [2], [1], [3]])

    assert trace[0]["resident"] == [0]
    assert trace[1]["resident"] == [1, 0]
    assert trace[2]["resident"] == [1, 0]
    assert trace[3]["resident"] == [2, 1, 0]
    assert trace[4]["resident"] == [2, 0, 1]
    assert trace[5]["resident"] == [3, 0, 1]

    assert list(policy._window.keys()) == [3]
    assert list(policy._probationary.keys()) == [0]
    assert list(policy._protected.keys()) == [1]


def test_slru_reference_trace_matches_caffeine_style_segmented_lru():
    accesses = [[0], [1], [2], [0], [3], [1], [4], [0], [5]]
    policy = create_residency_policy("slru", num_experts=8, capacity=4, config={"slru_protected_ratio": 0.5})

    actual = snapshot_trace(policy, accesses)
    expected = reference_segmented_lru_trace(capacity=4, protected_ratio=0.5, accesses=accesses)

    for index, (act, exp) in enumerate(zip(actual, expected)):
        assert act["resident"] == exp["resident"], f"resident mismatch at step {index}"

    assert list(policy._probationary.keys()) == expected[-1]["probationary"]
    assert list(policy._protected.keys()) == expected[-1]["protected"]


def test_2q_reference_trace_matches_expected_queue_behavior():
    accesses = [[0], [1], [0], [2], [3], [2], [4], [2], [5]]
    policy = create_residency_policy("2q", num_experts=8, capacity=4, config={"2q_a1_ratio": 0.5})

    actual = snapshot_trace(policy, accesses)
    expected = reference_2q_trace(capacity=4, a1_ratio=0.5, accesses=accesses)

    for index, (act, exp) in enumerate(zip(actual, expected)):
        assert act["resident"] == exp["resident"], f"resident mismatch at step {index}"

    assert list(policy._a1.keys()) == expected[-1]["a1"]
    assert list(policy._am.keys()) == expected[-1]["am"]


def test_wtinylfu_reference_trace_matches_caffeine_window_main_flow():
    accesses = [[0], [1], [0], [2], [3], [0], [4], [1], [5], [0]]
    policy = create_residency_policy(
        "w_tinylfu",
        num_experts=8,
        capacity=4,
        config={
            "wtinylfu_window_ratio": 0.25,
            "wtinylfu_protected_ratio": 0.5,
            "wtinylfu_decay_interval": 1000,
        },
    )

    actual = snapshot_trace(policy, accesses)
    expected = reference_window_tiny_lfu_trace(
        capacity=4,
        window_ratio=0.25,
        protected_ratio=0.5,
        accesses=accesses,
    )

    for index, (act, exp) in enumerate(zip(actual, expected)):
        assert act["resident"] == exp["resident"], f"resident mismatch at step {index}"

    assert list(policy._window.keys()) == expected[-1]["window"]
    assert list(policy._probationary.keys()) == expected[-1]["probationary"]
    assert list(policy._protected.keys()) == expected[-1]["protected"]
