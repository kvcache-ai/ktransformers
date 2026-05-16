# Expert residency policy helpers for MESH.
# SPDX-License-Identifier: Apache-2.0

"""
Small Python-side helpers shared by the MESH runtime and policy tests.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

def _read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, OSError):
        return None


def _parse_mountinfo(mountinfo_text: Optional[str] = None) -> List[Tuple[str, str, str]]:
    if mountinfo_text is None:
        mountinfo_text = _read_text_file(Path("/proc/self/mountinfo")) or ""

    mounts: List[Tuple[str, str, str]] = []
    for line in mountinfo_text.splitlines():
        left, sep, right = line.partition(" - ")
        if not sep:
            continue
        left_parts = left.split()
        right_parts = right.split()
        if len(left_parts) < 5 or len(right_parts) < 3:
            continue
        mount_point = left_parts[4]
        fs_type = right_parts[0]
        super_opts = right_parts[2]
        mounts.append((fs_type, mount_point, super_opts))
    return mounts


def _parse_self_cgroup(proc_self_cgroup_text: Optional[str] = None) -> List[Tuple[str, List[str], str]]:
    if proc_self_cgroup_text is None:
        proc_self_cgroup_text = _read_text_file(Path("/proc/self/cgroup")) or ""

    entries: List[Tuple[str, List[str], str]] = []
    for line in proc_self_cgroup_text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy_id, controllers_raw, rel_path = parts
        controllers = [c for c in controllers_raw.split(",") if c]
        entries.append((hierarchy_id, controllers, rel_path))
    return entries


def get_cgroup_memory_limit_current_bytes(
    *,
    proc_self_cgroup_text: Optional[str] = None,
    mountinfo_text: Optional[str] = None,
    mount_root_override: Optional[Path] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (limit_bytes, current_bytes) for the current process cgroup when available.

    Supports cgroup v2 first, then falls back to cgroup v1 memory controller layouts.
    Returns (None, None) when no finite cgroup limit can be determined.
    """

    entries = _parse_self_cgroup(proc_self_cgroup_text)
    mounts = _parse_mountinfo(mountinfo_text)

    def _read_limit_current(base: Path, limit_name: str, current_name: str) -> Tuple[Optional[int], Optional[int]]:
        limit_text = _read_text_file(base / limit_name)
        current_text = _read_text_file(base / current_name)
        if limit_text is None or current_text is None:
            return None, None
        if limit_text == "max":
            return None, None
        try:
            limit = int(limit_text)
            current = int(current_text)
        except ValueError:
            return None, None
        # Some v1 setups expose a huge sentinel instead of "max".
        if limit <= 0 or limit >= (1 << 60):
            return None, None
        return limit, max(0, current)

    rel_v2 = None
    for hierarchy_id, controllers, rel_path in entries:
        if hierarchy_id == "0" and not controllers:
            rel_v2 = rel_path
            break

    mount_v2 = mount_root_override
    if mount_v2 is None:
        for fs_type, mount_point, _super_opts in mounts:
            if fs_type == "cgroup2":
                mount_v2 = Path(mount_point)
                break
    if mount_v2 is not None:
        candidates = []
        if rel_v2 and rel_v2 != "/":
            candidates.append(mount_v2 / rel_v2.lstrip("/"))
        candidates.append(mount_v2)
        for candidate in candidates:
            limit, current = _read_limit_current(candidate, "memory.max", "memory.current")
            if limit is not None and current is not None:
                return limit, current

    rel_v1 = None
    for _hierarchy_id, controllers, rel_path in entries:
        if "memory" in controllers:
            rel_v1 = rel_path
            break

    mount_v1 = mount_root_override
    if mount_v1 is None:
        for fs_type, mount_point, super_opts in mounts:
            if fs_type == "cgroup" and "memory" in super_opts.split(","):
                mount_v1 = Path(mount_point)
                break
    if mount_v1 is not None:
        candidates = []
        if rel_v1 and rel_v1 != "/":
            candidates.append(mount_v1 / rel_v1.lstrip("/"))
        candidates.append(mount_v1)
        for candidate in candidates:
            limit, current = _read_limit_current(candidate, "memory.limit_in_bytes", "memory.usage_in_bytes")
            if limit is not None and current is not None:
                return limit, current

    return None, None


def get_available_ram_bytes() -> int:
    """Get available RAM in bytes, preferring the current process cgroup when finite."""
    cgroup_limit, cgroup_current = get_cgroup_memory_limit_current_bytes()
    if cgroup_limit is not None and cgroup_current is not None:
        return max(0, cgroup_limit - cgroup_current)

    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass
    # Fallback: read from /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except (FileNotFoundError, ValueError):
        pass
    # Last resort: assume 64GB
    return 64 * 1024**3


def get_total_ram_bytes() -> int:
    """Get total RAM in bytes, preferring the current process cgroup when finite."""
    cgroup_limit, _cgroup_current = get_cgroup_memory_limit_current_bytes()
    if cgroup_limit is not None:
        return cgroup_limit

    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        pass
    return 64 * 1024**3


def estimate_model_weight_bytes(
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,  # Q4 = 0.5 bytes/element
) -> int:
    """Estimate total MoE expert weight size in bytes."""
    # Each expert has 3 projections: gate, up (hidden×intermediate), down (intermediate×hidden)
    elements_per_expert = 2 * hidden_size * intermediate_size + intermediate_size * hidden_size
    total_elements = num_layers * num_experts * elements_per_expert
    return int(total_elements * bytes_per_element)


def method_bytes_per_element(method: Optional[str]) -> float:
    """Return an approximate bytes/element ratio for a backend's expert weights."""
    if method is None:
        return 0.5

    normalized = method.upper()
    if normalized in {"LLAMAFILE", "AMXINT4", "MOE_INT4", "RAWINT4"}:
        return 0.5
    if normalized in {"AMXINT8", "MOE_INT8", "FP8", "FP8_PERCHANNEL"}:
        return 1.0
    if normalized == "BF16":
        return 2.0
    return 0.5


def compute_max_tier0_experts(
    tier0_memory_bytes: int,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,  # Q4_K ≈ 0.5 bytes/element
) -> int:
    """
    Compute how many experts can fit in the Tier 0 NUMA-local memory budget.

    Tier 0 promotes the same expert IDs across ALL layers, so total memory is:
        max_tier0_experts * num_layers * per_expert_per_layer_bytes

    Args:
        tier0_memory_bytes: Total memory budget for Tier 0 in bytes
        num_layers: Number of MoE layers in the model
        num_experts: Total experts per layer (for clamping)
        hidden_size: Model hidden size (e.g., 7168)
        intermediate_size: Expert intermediate size (e.g., 2048)
        bytes_per_element: Quantization ratio (Q4_K≈0.5, Q8≈1.0, BF16≈2.0)

    Returns:
        Maximum number of experts that can be promoted to Tier 0
    """
    # Each expert has 3 projections: gate(H×I), up(H×I), down(I×H)
    per_expert_per_layer = int(3 * hidden_size * intermediate_size * bytes_per_element)
    if per_expert_per_layer <= 0 or num_layers <= 0:
        return 0
    if tier0_memory_bytes <= 0:
        return 0
    per_expert_total = per_expert_per_layer * num_layers
    max_experts = int(tier0_memory_bytes / per_expert_total)
    return max(0, min(max_experts, num_experts))


RESIDENCY_POLICY_ALIASES = {
    "baseline": "baseline",
    "default": "baseline",
    "current": "baseline",
    "current_ema": "baseline",
    "ema": "baseline",
    "ema_hotset": "baseline",
    "legacy": "baseline",
    "lru": "lru",
    "2q": "2q",
    "twoq": "2q",
    "two-q": "2q",
    "slru": "slru",
    "sieve": "sieve",
    "s3fifo": "s3fifo",
    "s3-fifo": "s3fifo",
    "wtinylfu": "w_tinylfu",
    "w-tinylfu": "w_tinylfu",
    "w_tinylfu": "w_tinylfu",
}


def normalize_residency_policy_name(name: Optional[str]) -> str:
    normalized = (name or "baseline").strip().lower()
    try:
        return RESIDENCY_POLICY_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(set(RESIDENCY_POLICY_ALIASES.values())))
        raise ValueError(f"Unsupported residency policy {name!r}. Expected one of: {choices}") from exc


def load_residency_policy_config(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid KT_RESIDENCY_POLICY_CONFIG JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("KT_RESIDENCY_POLICY_CONFIG must decode to a JSON object")
    return parsed


def _filter_valid_expert_ids(expert_ids: Iterable[int], num_experts: int) -> List[int]:
    valid: List[int] = []
    for expert_id in expert_ids:
        try:
            value = int(expert_id)
        except (TypeError, ValueError):
            continue
        if 0 <= value < num_experts:
            valid.append(value)
    return valid


class ResidencyPolicy:
    """Common interface for MESH resident-cache policies."""

    policy_name = "base"

    def __init__(self, num_experts: int, capacity: int):
        self.num_experts = num_experts
        self.capacity = max(0, min(int(capacity), int(num_experts)))
        self._lock = threading.RLock()
        self.stats: Dict[str, float] = {
            "accesses": 0,
            "unique_accesses": 0,
            "hits": 0,
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
            "prefetch_candidates": 0,
        }

    def record_accesses(self, expert_ids: Sequence[int]) -> Dict[str, Any]:
        accesses = _filter_valid_expert_ids(expert_ids, self.num_experts)
        with self._lock:
            before = self._resident_order_locked()
            hits, misses = self._record_accesses_locked(accesses)
            after = self._resident_order_locked()
            before_set = set(before)
            after_set = set(after)
            promotions = len(after_set - before_set)
            demotions = len(before_set - after_set)
            unique_accesses = len(set(accesses))
            self.stats["accesses"] += len(accesses)
            self.stats["unique_accesses"] += unique_accesses
            self.stats["hits"] += hits
            self.stats["misses"] += misses
            self.stats["promotions"] += promotions
            self.stats["demotions"] += demotions
            self.stats["prefetch_candidates"] += sum(1 for eid in set(accesses) if eid not in before_set)
            return {
                "policy": self.policy_name,
                "accesses": list(accesses),
                "unique_accesses": unique_accesses,
                "hits": hits,
                "misses": misses,
                "promotions": promotions,
                "demotions": demotions,
                "resident_before": before,
                "resident_after": after,
            }

    def resident_ids(self) -> List[int]:
        with self._lock:
            return self._resident_order_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            hits = float(self.stats["hits"])
            misses = float(self.stats["misses"])
            accesses = hits + misses
            return {
                "policy": self.policy_name,
                "capacity": self.capacity,
                "resident": self._resident_order_locked(),
                "stats": dict(self.stats),
                "hit_rate": 0.0 if accesses <= 0 else hits / accesses,
            }

    def _resident_order_locked(self) -> List[int]:
        raise NotImplementedError

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        raise NotImplementedError


class BaselineHotsetPolicy(ResidencyPolicy):
    """EMA hotness policy for selecting resident expert IDs."""

    policy_name = "baseline"

    def __init__(self, num_experts: int, capacity: int, ema_alpha: float = 0.01):
        super().__init__(num_experts=num_experts, capacity=capacity)
        self.ema_alpha = float(ema_alpha)
        self.counts = np.zeros(num_experts, dtype=np.float64)
        self._resident: List[int] = []

    def _resident_order_locked(self) -> List[int]:
        return list(self._resident)

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        before = set(self._resident)
        hits = sum(1 for expert_id in expert_ids if expert_id in before)
        misses = len(expert_ids) - hits
        if expert_ids:
            self.counts *= 1 - self.ema_alpha
            hits_vec = np.zeros(self.num_experts, dtype=np.float64)
            np.add.at(hits_vec, np.asarray(expert_ids, dtype=np.int64), 1.0)
            self.counts[hits_vec > 0] += self.ema_alpha
        if self.capacity <= 0:
            self._resident = []
        elif not np.any(self.counts > 0):
            self._resident = []
        else:
            top = np.argsort(self.counts)[-self.capacity :][::-1]
            self._resident = [int(expert_id) for expert_id in top.tolist()]
        return hits, misses


class LRUPolicy(ResidencyPolicy):
    policy_name = "lru"

    def __init__(self, num_experts: int, capacity: int):
        super().__init__(num_experts=num_experts, capacity=capacity)
        self._cache: "OrderedDict[int, None]" = OrderedDict()

    def _resident_order_locked(self) -> List[int]:
        return list(self._cache.keys())

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            if expert_id in self._cache:
                hits += 1
                self._cache.move_to_end(expert_id)
                continue
            misses += 1
            if self.capacity <= 0:
                continue
            self._cache[expert_id] = None
            self._cache.move_to_end(expert_id)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return hits, misses


class TwoQPolicy(ResidencyPolicy):
    policy_name = "2q"

    def __init__(self, num_experts: int, capacity: int, a1_ratio: float = 0.25):
        super().__init__(num_experts=num_experts, capacity=capacity)
        if self.capacity <= 1:
            self.a1_capacity = self.capacity
            self.am_capacity = 0
        else:
            self.a1_capacity = max(1, min(self.capacity - 1, int(round(self.capacity * a1_ratio))))
            self.am_capacity = self.capacity - self.a1_capacity
        self._a1: "OrderedDict[int, None]" = OrderedDict()
        self._am: "OrderedDict[int, None]" = OrderedDict()

    def _resident_order_locked(self) -> List[int]:
        return list(self._a1.keys()) + list(self._am.keys())

    def _ensure_limits_locked(self) -> None:
        while len(self._a1) > self.a1_capacity:
            self._a1.popitem(last=False)
        while len(self._am) > self.am_capacity:
            self._am.popitem(last=False)
        while len(self._a1) + len(self._am) > self.capacity:
            if self._a1:
                self._a1.popitem(last=False)
            elif self._am:
                self._am.popitem(last=False)
            else:
                break

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            if expert_id in self._am:
                hits += 1
                self._am.move_to_end(expert_id)
                continue
            if expert_id in self._a1:
                hits += 1
                self._a1.pop(expert_id, None)
                if self.am_capacity > 0:
                    self._am[expert_id] = None
                else:
                    self._a1[expert_id] = None
                self._ensure_limits_locked()
                continue

            misses += 1
            if self.capacity <= 0:
                continue
            self._a1[expert_id] = None
            self._ensure_limits_locked()
        return hits, misses


class SLRUPolicy(ResidencyPolicy):
    policy_name = "slru"

    def __init__(self, num_experts: int, capacity: int, protected_ratio: float = 0.8):
        super().__init__(num_experts=num_experts, capacity=capacity)
        if self.capacity <= 1:
            self.protected_capacity = 0
        else:
            self.protected_capacity = max(1, min(self.capacity - 1, int(round(self.capacity * protected_ratio))))
        self._probationary: "OrderedDict[int, None]" = OrderedDict()
        self._protected: "OrderedDict[int, None]" = OrderedDict()

    def _resident_order_locked(self) -> List[int]:
        return list(self._probationary.keys()) + list(self._protected.keys())

    def _ensure_limits_locked(self) -> None:
        while len(self._protected) > self.protected_capacity and self.protected_capacity >= 0:
            demoted, _ = self._protected.popitem(last=False)
            self._probationary[demoted] = None
        while len(self._probationary) + len(self._protected) > self.capacity:
            self._probationary.popitem(last=False)

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            if expert_id in self._protected:
                hits += 1
                self._protected.move_to_end(expert_id)
                continue
            if expert_id in self._probationary:
                hits += 1
                self._probationary.pop(expert_id, None)
                if self.protected_capacity > 0:
                    self._protected[expert_id] = None
                else:
                    self._probationary[expert_id] = None
                self._ensure_limits_locked()
                continue

            misses += 1
            if self.capacity <= 0:
                continue
            self._probationary[expert_id] = None
            self._ensure_limits_locked()
        return hits, misses


class SIEVEPolicy(ResidencyPolicy):
    """One-bit lazy promotion policy aligned with libCacheSim's Sieve."""

    policy_name = "sieve"

    def __init__(self, num_experts: int, capacity: int):
        super().__init__(num_experts=num_experts, capacity=capacity)
        self._queue: List[int] = []
        self._present: Dict[int, bool] = {}
        self._visited: Dict[int, bool] = {}
        self._hand: Optional[int] = None

    def _resident_order_locked(self) -> List[int]:
        return list(self._queue)

    def _append_new_head_locked(self, expert_id: int) -> None:
        # libCacheSim's Sieve prepends new objects to the queue head.
        self._queue.insert(0, expert_id)
        self._present[expert_id] = True
        self._visited[expert_id] = False
        if self._hand is None:
            self._hand = len(self._queue) - 1
        elif self._hand >= 0:
            # Existing pointer should keep referring to the same logical victim
            # candidate after a head insertion.
            self._hand += 1

    def _prev_index_locked(self, idx: int) -> int:
        if not self._queue:
            return 0
        if idx <= 0:
            return len(self._queue) - 1
        return idx - 1

    def _evict_one_locked(self) -> None:
        if not self._queue:
            return
        if self._hand is None or self._hand >= len(self._queue):
            self._hand = len(self._queue) - 1
        while self._queue:
            victim = self._queue[self._hand]
            if self._visited.get(victim, False):
                self._visited[victim] = False
                self._hand = self._prev_index_locked(self._hand)
                continue
            self._queue.pop(self._hand)
            self._present.pop(victim, None)
            self._visited.pop(victim, None)
            if self._queue:
                if self._hand >= len(self._queue):
                    self._hand = len(self._queue) - 1
            else:
                self._hand = None
            return

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            if self._present.get(expert_id, False):
                hits += 1
                self._visited[expert_id] = True
                continue
            misses += 1
            if self.capacity <= 0:
                continue
            if len(self._queue) >= self.capacity:
                self._evict_one_locked()
            self._append_new_head_locked(expert_id)
        return hits, misses


class S3FIFOPolicy(ResidencyPolicy):
    """Count-based adaptation of libCacheSim's S3-FIFO."""

    policy_name = "s3fifo"

    def __init__(
        self,
        num_experts: int,
        capacity: int,
        small_ratio: float = 0.1,
        ghost_ratio: float = 1.0,
        move_to_main_threshold: int = 2,
        main_freq_cap: int = 3,
    ):
        super().__init__(num_experts=num_experts, capacity=capacity)
        if self.capacity <= 1:
            self.small_capacity = self.capacity
            self.main_capacity = 0
        else:
            self.small_capacity = max(1, min(self.capacity - 1, int(round(self.capacity * small_ratio))))
            self.main_capacity = self.capacity - self.small_capacity
        self.ghost_capacity = max(1, int(max(self.capacity, 1) * ghost_ratio))
        self.move_to_main_threshold = max(1, int(move_to_main_threshold))
        self.main_freq_cap = max(1, int(main_freq_cap))
        self._small: Deque[int] = deque()
        self._main: Deque[int] = deque()
        self._resident_queue: Dict[int, str] = {}
        self._ghost: "OrderedDict[int, None]" = OrderedDict()
        self._freq: Dict[int, int] = {}

    def _resident_order_locked(self) -> List[int]:
        return list(self._small) + list(self._main)

    def _remember_ghost_locked(self, expert_id: int) -> None:
        self._ghost.pop(expert_id, None)
        self._ghost[expert_id] = None
        while len(self._ghost) > self.ghost_capacity:
            self._ghost.popitem(last=False)

    def _insert_main_locked(self, expert_id: int, freq: int) -> None:
        if self.main_capacity <= 0:
            return
        self._main.append(expert_id)
        self._resident_queue[expert_id] = "main"
        self._freq[expert_id] = min(self.main_freq_cap, max(0, int(freq)))

    def _insert_small_locked(self, expert_id: int) -> None:
        if self.small_capacity <= 0:
            self._insert_main_locked(expert_id, 0)
            return
        self._small.append(expert_id)
        self._resident_queue[expert_id] = "small"
        self._freq[expert_id] = 0

    def _evict_from_main_locked(self) -> bool:
        while self._main:
            victim = self._main.popleft()
            freq = self._freq.get(victim, 0)
            if freq >= 1:
                self._freq[victim] = freq - 1
                self._main.append(victim)
                continue
            self._resident_queue.pop(victim, None)
            self._freq.pop(victim, None)
            return True
        return False

    def _evict_from_small_locked(self) -> bool:
        while self._small:
            victim = self._small.popleft()
            freq = self._freq.get(victim, 0)
            self._resident_queue.pop(victim, None)
            self._freq.pop(victim, None)
            if freq >= self.move_to_main_threshold and self.main_capacity > 0:
                self._insert_main_locked(victim, freq)
                return self._ensure_capacity_locked()
            self._remember_ghost_locked(victim)
            return True
        return False

    def _ensure_capacity_locked(self) -> bool:
        progress = True
        while progress:
            progress = False
            if len(self._small) > self.small_capacity:
                progress = self._evict_from_small_locked()
                continue
            if len(self._main) > self.main_capacity:
                progress = self._evict_from_main_locked()
                continue
            if len(self._small) + len(self._main) > self.capacity:
                if self._small:
                    progress = self._evict_from_small_locked()
                else:
                    progress = self._evict_from_main_locked()
        return len(self._small) <= self.small_capacity and len(self._main) <= self.main_capacity and (
            len(self._small) + len(self._main) <= self.capacity
        )

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            location = self._resident_queue.get(expert_id)
            if location is not None:
                hits += 1
                self._freq[expert_id] = min(self.main_freq_cap, self._freq.get(expert_id, 0) + 1)
                continue
            misses += 1
            if self.capacity <= 0:
                continue
            if expert_id in self._ghost:
                self._ghost.pop(expert_id, None)
                self._insert_main_locked(expert_id, self.move_to_main_threshold)
            else:
                self._insert_small_locked(expert_id)
            self._ensure_capacity_locked()
        return hits, misses


class WTinyLFUPolicy(ResidencyPolicy):
    """A small-cache W-TinyLFU variant aligned with Caffeine's queue structure."""

    policy_name = "w_tinylfu"

    def __init__(
        self,
        num_experts: int,
        capacity: int,
        window_ratio: float = 0.1,
        protected_ratio: float = 0.8,
        decay_interval: int = 1000,
    ):
        super().__init__(num_experts=num_experts, capacity=capacity)
        if self.capacity <= 1:
            self.window_capacity = self.capacity
            self.main_capacity = 0
        else:
            self.window_capacity = max(1, min(self.capacity - 1, int(round(self.capacity * window_ratio))))
            self.main_capacity = self.capacity - self.window_capacity
        self.protected_capacity = max(0, min(self.main_capacity, int(self.main_capacity * protected_ratio)))
        self.decay_interval = max(1, int(decay_interval))
        self._window: "OrderedDict[int, None]" = OrderedDict()
        self._probationary: "OrderedDict[int, None]" = OrderedDict()
        self._protected: "OrderedDict[int, None]" = OrderedDict()
        self._freq: Dict[int, int] = {}
        self._total_accesses = 0

    def _resident_order_locked(self) -> List[int]:
        return list(self._window.keys()) + list(self._probationary.keys()) + list(self._protected.keys())

    def _increment_freq_locked(self, expert_id: int) -> None:
        self._total_accesses += 1
        self._freq[expert_id] = self._freq.get(expert_id, 0) + 1
        if self._total_accesses % self.decay_interval == 0:
            for key in list(self._freq.keys()):
                decayed = self._freq[key] // 2
                if decayed <= 0:
                    self._freq.pop(key, None)
                else:
                    self._freq[key] = decayed

    def _ensure_main_limits_locked(self) -> None:
        while len(self._protected) > self.protected_capacity and self.protected_capacity >= 0:
            demoted, _ = self._protected.popitem(last=False)
            self._probationary[demoted] = None

    def _admit_candidate_locked(self, candidate: int) -> None:
        if self.main_capacity <= 0:
            return
        self._probationary[candidate] = None
        if len(self._window) + len(self._probationary) + len(self._protected) <= self.capacity:
            return

        victim = next(iter(self._probationary))
        candidate_freq = self._freq.get(candidate, 0)
        victim_freq = self._freq.get(victim, 0)
        if candidate_freq >= victim_freq:
            self._probationary.pop(victim, None)
        else:
            self._probationary.pop(candidate, None)

    def _record_accesses_locked(self, expert_ids: Sequence[int]) -> Tuple[int, int]:
        hits = 0
        misses = 0
        for expert_id in expert_ids:
            self._increment_freq_locked(expert_id)

            if expert_id in self._window:
                hits += 1
                self._window.move_to_end(expert_id)
                continue
            if expert_id in self._protected:
                hits += 1
                self._protected.move_to_end(expert_id)
                continue
            if expert_id in self._probationary:
                hits += 1
                self._probationary.pop(expert_id, None)
                self._protected[expert_id] = None
                self._ensure_main_limits_locked()
                continue

            misses += 1
            if self.capacity <= 0:
                continue

            self._window[expert_id] = None
            self._window.move_to_end(expert_id)
            if len(self._window) > self.window_capacity:
                candidate, _ = self._window.popitem(last=False)
                self._admit_candidate_locked(candidate)
        return hits, misses


def create_residency_policy(
    policy_name: Optional[str],
    *,
    num_experts: int,
    capacity: int,
    config: Optional[Dict[str, Any]] = None,
) -> ResidencyPolicy:
    config = dict(config or {})
    normalized = normalize_residency_policy_name(policy_name)
    if normalized == "baseline":
        return BaselineHotsetPolicy(
            num_experts=num_experts,
            capacity=capacity,
            ema_alpha=float(config.get("ema_alpha", 0.01)),
        )
    if normalized == "lru":
        return LRUPolicy(num_experts=num_experts, capacity=capacity)
    if normalized == "2q":
        return TwoQPolicy(
            num_experts=num_experts,
            capacity=capacity,
            a1_ratio=float(config.get("2q_a1_ratio", config.get("twoq_a1_ratio", 0.25))),
        )
    if normalized == "slru":
        return SLRUPolicy(
            num_experts=num_experts,
            capacity=capacity,
            protected_ratio=float(config.get("slru_protected_ratio", 0.8)),
        )
    if normalized == "sieve":
        return SIEVEPolicy(num_experts=num_experts, capacity=capacity)
    if normalized == "s3fifo":
        return S3FIFOPolicy(
            num_experts=num_experts,
            capacity=capacity,
            small_ratio=float(config.get("s3fifo_small_ratio", 0.1)),
            ghost_ratio=float(config.get("s3fifo_ghost_ratio", 1.0)),
            move_to_main_threshold=int(config.get("s3fifo_move_to_main_threshold", 2)),
            main_freq_cap=int(config.get("s3fifo_main_freq_cap", 3)),
        )
    if normalized == "w_tinylfu":
        return WTinyLFUPolicy(
            num_experts=num_experts,
            capacity=capacity,
            window_ratio=float(config.get("wtinylfu_window_ratio", 0.1)),
            protected_ratio=float(config.get("wtinylfu_protected_ratio", 0.8)),
            decay_interval=int(config.get("wtinylfu_decay_interval", 1000)),
        )
    raise ValueError(f"Unhandled residency policy {policy_name!r}")
