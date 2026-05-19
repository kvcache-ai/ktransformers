# Tiered Residency Policy Design

## Goal

Make tiered expert residency policy selectable at runtime and comparable offline with a shared implementation core.

## Scope

- Keep `weight_strategy` semantics unchanged: `auto/legacy/tiered` still decide whether mmap-tiered mode is used.
- Inside tiered mode, make Tier0 pinned-expert selection pluggable.
- Compare the following policies under the same trace format:
  - `baseline`
  - `lru`
  - `slru`
  - `sieve`
  - `s3fifo`
  - `tinylfu`
  - `w_tinylfu`

## Design

### Shared policy core

`kt-kernel/python/utils/weight_provider.py` owns a `ResidencyPolicy` interface.

Each policy instance is per-layer and exposes:

- `record_accesses(expert_ids)`
- `resident_ids()`
- `snapshot()`

This keeps online provider behavior and offline replay behavior aligned.

### Online integration

`TieredWeightProvider` now owns `policy_by_layer` instead of a hard-coded EMA tracker.

- `record_activations()` updates the configured policy with CPU expert accesses.
- `_maybe_promote()` diffs the policy’s desired resident set against actual pinned experts and calls `promote_expert()` / `demote_expert()`.
- `KT_RESIDENCY_POLICY` selects the policy at runtime.
- `KT_RESIDENCY_TRACE_PATH` emits JSONL access traces for offline replay.

### Offline comparison

`kt-kernel/scripts/compare_residency_policies.py` replays a JSONL trace against multiple policies and reports:

- hits / misses / hit rate
- promotions / demotions
- prefetch candidates
- final resident set

## Non-goals

- Replacing the C++ request-path resident-cache eviction logic in this change.
- Exposing every per-policy tuning knob in the user-facing CLI.

## Follow-up

- If a non-EMA policy wins clearly under real traces, move the same policy family into the C++ resident-cache path too.
