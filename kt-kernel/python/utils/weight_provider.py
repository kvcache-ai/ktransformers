# Weight provider abstraction for tiered MoE weight management
# SPDX-License-Identifier: Apache-2.0

"""
Tiered weight provider for MoE expert weights.

Implements a three-tier caching strategy:
  Tier 0: NUMA-local malloc buffers (hottest experts, ~80ns access)
          Managed by C++ promote_expert/demote_expert on the MOE object.
  Tier 1: mmap pages resident in RAM (warm experts, ~80-150ns, OS-managed)
  Tier 2: mmap pages on disk (cold experts, ~100us on page fault)

This is essential when model_size >= physical_ram to avoid swap thrashing.
"""

from __future__ import annotations

import ctypes
import json
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

# Try to import madvise support
_libc = None
MADV_WILLNEED = 3
TIERED_BACKENDS = frozenset({"LLAMAFILE", "AMXINT4", "AMXINT8", "MOE_INT4", "MOE_INT8", "BF16"})

try:
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    try:
        _libc = ctypes.CDLL("libc.dylib", use_errno=True)
    except OSError:
        pass


def _madvise_willneed(addr: int, length: int):
    """Hint to the OS that the given address range will be needed soon."""
    if _libc is None or addr == 0 or length <= 0:
        return

    page_size = os.sysconf("SC_PAGESIZE")
    aligned_addr = addr & ~(page_size - 1)
    aligned_end = (addr + length + page_size - 1) & ~(page_size - 1)
    aligned_length = aligned_end - aligned_addr
    _libc.madvise(ctypes.c_void_p(aligned_addr), ctypes.c_size_t(aligned_length), ctypes.c_int(MADV_WILLNEED))


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


def should_use_tiered_strategy(
    model_bytes: int,
    total_ram: Optional[int] = None,
    available_ram: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> bool:
    """Determine whether to use tiered (mmap) or legacy (malloc) strategy."""
    if available_ram is None:
        available_ram = get_available_ram_bytes()
    if model_bytes > max(0, available_ram - safety_bytes):
        return True
    if total_ram is None:
        total_ram = get_total_ram_bytes()
    return model_bytes >= total_ram * threshold


def backend_supports_tiered_strategy(method: Optional[str]) -> bool:
    """Return whether a backend supports mmap-backed expert residency control."""
    if method is None:
        return False
    return method.upper() in TIERED_BACKENDS


def resolve_weight_strategy(
    requested_strategy: Optional[str],
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    bytes_per_element: float = 0.5,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> Tuple[str, int, int]:
    """Resolve "auto" into a concrete strategy using estimated model size."""
    strategy = requested_strategy or "tiered"
    if strategy != "auto":
        total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
        return strategy, 0, total_ram

    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    available_ram = get_available_ram_bytes() if available_ram_bytes is None else available_ram_bytes
    model_bytes = estimate_model_weight_bytes(
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bytes_per_element=bytes_per_element,
    )
    resolved = (
        "tiered"
        if should_use_tiered_strategy(
            model_bytes,
            total_ram=total_ram,
            available_ram=available_ram,
            threshold=threshold,
            safety_bytes=safety_bytes,
        )
        else "legacy"
    )
    return resolved, model_bytes, total_ram


def resolve_backend_weight_strategy(
    method: Optional[str],
    requested_strategy: Optional[str],
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    threshold: float = 0.7,
    safety_bytes: int = 4 * 1024**3,
) -> Tuple[str, int, int]:
    """Resolve a requested strategy while respecting backend residency capabilities."""
    strategy = requested_strategy or "auto"
    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    if not backend_supports_tiered_strategy(method):
        if strategy in {"auto", "tiered"}:
            return "legacy", 0, total_ram
        return strategy, 0, total_ram

    return resolve_weight_strategy(
        strategy,
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        bytes_per_element=method_bytes_per_element(method),
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram_bytes,
        threshold=threshold,
        safety_bytes=safety_bytes,
    )


def resolve_auto_tier0_budget_bytes(
    *,
    model_bytes: int,
    total_ram_bytes: Optional[int] = None,
    available_ram_bytes: Optional[int] = None,
    safety_bytes: int = 4 * 1024**3,
) -> int:
    """
    Pick a Tier0 NUMA budget automatically.

    The baseline assumption is:
      - Tier1/Tier2 live in mmap
      - Tier0 is a bounded NUMA-local hotset

    More free headroom allows a larger hotset; higher model pressure shrinks it.
    """
    total_ram = get_total_ram_bytes() if total_ram_bytes is None else total_ram_bytes
    available_ram = get_available_ram_bytes() if available_ram_bytes is None else available_ram_bytes
    headroom_bytes = max(0, available_ram - safety_bytes)
    if headroom_bytes <= 0:
        return 0

    pressure = model_bytes / max(float(total_ram), 1.0)
    if pressure <= 0.70:
        tier0_fraction = 1.0
    elif pressure <= 1.00:
        tier0_fraction = 0.50
    elif pressure <= 1.50:
        tier0_fraction = 0.25
    else:
        tier0_fraction = 0.10

    return int(max(0.0, min(float(headroom_bytes), headroom_bytes * tier0_fraction)))


def constrain_tier0_memory_bytes(
    requested_bytes: int,
    *,
    available_ram_bytes: Optional[int] = None,
    safety_bytes: int = 4 * 1024**3,
) -> int:
    """
    Clamp an explicit Tier0 budget to the effective memory scope.

    This makes a caller-provided Tier0 budget respect the current cgroup when one
    is active, rather than assuming host-wide RAM is available.
    """
    if requested_bytes <= 0:
        return 0
    if available_ram_bytes is None:
        available_ram_bytes = get_available_ram_bytes()
    return min(requested_bytes, max(0, available_ram_bytes - safety_bytes))


class ExpertHotnessTracker:
    """
    Track expert activation frequency to identify hot experts for Tier 0 promotion.

    Uses an exponential moving average to adapt to changing access patterns.
    Thread-safe for concurrent recording from inference threads.
    """

    def __init__(self, num_experts: int, ema_alpha: float = 0.01):
        self.num_experts = num_experts
        self.ema_alpha = ema_alpha
        self.counts = np.zeros(num_experts, dtype=np.float64)
        self._lock = threading.Lock()

    def record(self, expert_ids: np.ndarray):
        """
        Record expert activations from a forward pass (vectorized).

        expert_ids: flat array of activated expert IDs (may contain duplicates).
        Uses np.add.at for O(n) vectorized accumulation instead of Python for-loop.
        """
        if expert_ids.size == 0:
            return
        # Filter valid IDs
        valid = expert_ids[(expert_ids >= 0) & (expert_ids < self.num_experts)]
        if valid.size == 0:
            return

        # Build per-expert hit counts in one vectorized pass
        hits = np.zeros(self.num_experts, dtype=np.float64)
        np.add.at(hits, valid, 1.0)
        # Normalize: each activated expert gets alpha, regardless of how many tokens hit it
        mask = hits > 0

        with self._lock:
            # EMA update: decay all, then boost activated ones
            self.counts *= 1 - self.ema_alpha
            self.counts[mask] += self.ema_alpha

    def get_top_k(self, k: int) -> List[int]:
        """Return indices of the top-k hottest experts."""
        with self._lock:
            if k <= 0:
                return []
            k = min(k, self.num_experts)
            return np.argsort(self.counts)[-k:][::-1].tolist()

    def decay(self):
        """Decay all counts (call periodically to let cold experts fade)."""
        with self._lock:
            self.counts *= 1 - self.ema_alpha


class MmapWeightRegion:
    """
    Manages a zero-copy mmap view into a GGUF weight file.

    Keeps the numpy mmap handle alive and provides raw pointer access.
    """

    def __init__(self, mmap_data: np.ndarray, offset: int, n_bytes: int):
        # Create a view into the mmap region (no copy)
        self._view = np.frombuffer(mmap_data[offset : offset + n_bytes], dtype=np.uint8)
        self.ptr = self._view.ctypes.data
        self.n_bytes = n_bytes

    def prefetch(self):
        """Hint to OS that this region will be needed soon."""
        _madvise_willneed(self.ptr, self.n_bytes)


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
    """Common interface for provider-managed expert residency policies."""

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
    """Current provider behavior: EMA hotness + top-k pinned experts."""

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


class TieredWeightProvider:
    """
    Three-tier weight manager for MoE experts.

    Tier 0 is managed entirely by C++ (NUMA-local malloc + pointer swap).
    This Python class orchestrates which experts to promote/demote by calling
    moe.promote_expert(eid) / moe.demote_expert(eid) on the C++ MOE objects.

    Usage:
        provider = TieredWeightProvider(num_experts=256, num_layers=60)
        # After creating MOE object in load_weights():
        provider.register_moe(layer_idx, moe_object)
        # During inference:
        provider.prefetch_layer(layer_idx, topk_ids)
        provider.record_activations(layer_idx, topk_ids)
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        max_tier0_experts: int = 30,
        promotion_interval_sec: float = 5.0,
        residency_policy: Optional[str] = None,
        policy_config: Optional[Dict[str, Any]] = None,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.max_tier0_experts = max_tier0_experts
        env_policy = os.environ.get("KT_RESIDENCY_POLICY")
        self.residency_policy_name = normalize_residency_policy_name(env_policy or residency_policy)
        self.policy_config = dict(policy_config or load_residency_policy_config(os.environ.get("KT_RESIDENCY_POLICY_CONFIG")))
        env_interval = os.environ.get("KT_PROMOTION_INTERVAL_SEC")
        if env_interval is not None:
            try:
                promotion_interval_sec = float(env_interval)
            except ValueError:
                print(
                    f"[TieredWeightProvider] ignoring invalid KT_PROMOTION_INTERVAL_SEC={env_interval!r}; "
                    f"using default {promotion_interval_sec}"
                )
        self.promotion_interval_sec = promotion_interval_sec
        self.trace_enabled = os.environ.get("KT_PROVIDER_TRACE") == "1"
        self.trace_hot_k = int(os.environ.get("KT_PROVIDER_TRACE_HOT_K", "16"))
        self.trace_path = os.environ.get("KT_RESIDENCY_TRACE_PATH")
        self._trace_lock = threading.Lock()

        # Per-layer mmap regions: layer_idx → proj_name → list[expert_id] → list[MmapWeightRegion]
        self.mmap_regions: Dict[int, Dict[str, List[List[MmapWeightRegion]]]] = {}

        # C++ MOE object references: layer_idx → MOE object (has promote/demote methods)
        self.moe_refs: Dict[int, object] = {}
        self.layer_gpu_expert_masks: Dict[int, np.ndarray] = {}

        # Each layer gets its own policy instance so one layer's routing pattern
        # does not bias another layer's pinned set.
        self.policy_by_layer: Dict[int, ResidencyPolicy] = {}

        # Background promotion thread
        self._running = False
        self._promotion_thread: Optional[threading.Thread] = None

    def register_moe(self, layer_idx: int, moe: object, gpu_experts_mask: Optional[np.ndarray] = None):
        """
        Register a C++ MOE object for a layer.

        The MOE object must have promote_expert(eid), demote_expert(eid),
        and is_expert_promoted(eid) methods (exposed via pybind11).
        """
        self.moe_refs[layer_idx] = moe
        if gpu_experts_mask is None:
            self.layer_gpu_expert_masks[layer_idx] = np.zeros(self.num_experts, dtype=np.bool_)
        else:
            mask = np.asarray(gpu_experts_mask, dtype=np.bool_).reshape(-1)
            if mask.size != self.num_experts:
                raise ValueError(
                    f"gpu_experts_mask for layer {layer_idx} has size {mask.size}, expected {self.num_experts}"
                )
            self.layer_gpu_expert_masks[layer_idx] = mask.copy()
        self.policy_by_layer.setdefault(
            layer_idx,
            create_residency_policy(
                self.residency_policy_name,
                num_experts=self.num_experts,
                capacity=self.max_tier0_experts,
                config=self.policy_config,
            ),
        )

        # Lazy start: only launch promotion thread once at least one MOE is registered.
        # This avoids running the thread when no backends support promote/demote (e.g., AMX).
        if self.max_tier0_experts > 0 and not self._running:
            self.start_promotion_thread()

    def unregister_moe(self, layer_idx: int):
        """Remove all state associated with a layer's MOE object and mmap regions."""
        self.moe_refs.pop(layer_idx, None)
        self.layer_gpu_expert_masks.pop(layer_idx, None)
        self.mmap_regions.pop(layer_idx, None)
        self.policy_by_layer.pop(layer_idx, None)
        if not self.moe_refs:
            self.stop_promotion_thread()

    def _get_policy(self, layer_idx: int) -> ResidencyPolicy:
        return self.policy_by_layer.setdefault(
            layer_idx,
            create_residency_policy(
                self.residency_policy_name,
                num_experts=self.num_experts,
                capacity=self.max_tier0_experts,
                config=self.policy_config,
            ),
        )

    def clear_layer_regions(self, layer_idx: int):
        """Drop mmap-region metadata for a layer before re-registering fresh slices."""
        self.mmap_regions.pop(layer_idx, None)

    def register_mmap_region(self, layer_idx: int, proj_name: str, expert_id: int, region: MmapWeightRegion):
        """Register an mmap region for a specific expert weight."""
        if layer_idx not in self.mmap_regions:
            self.mmap_regions[layer_idx] = {}
        if proj_name not in self.mmap_regions[layer_idx]:
            self.mmap_regions[layer_idx][proj_name] = [[] for _ in range(self.num_experts)]
        self.mmap_regions[layer_idx][proj_name][expert_id].append(region)

    def _filter_cpu_expert_ids(self, layer_idx: int, expert_ids: np.ndarray) -> np.ndarray:
        """Remove invalid IDs and experts that are assigned to GPU on this layer."""
        valid = expert_ids[(expert_ids >= 0) & (expert_ids < self.num_experts)]
        if valid.size == 0:
            return valid
        gpu_mask = self.layer_gpu_expert_masks.get(layer_idx)
        if gpu_mask is None:
            return valid
        return valid[~gpu_mask[valid]]

    def _is_gpu_expert(self, layer_idx: int, expert_id: int) -> bool:
        """Check whether an expert is assigned to GPU for the given layer."""
        gpu_mask = self.layer_gpu_expert_masks.get(layer_idx)
        if gpu_mask is None or expert_id < 0 or expert_id >= self.num_experts:
            return False
        return bool(gpu_mask[expert_id])

    def record_activations(self, layer_idx: int, topk_ids: np.ndarray):
        """Record expert activations into the layer's residency policy."""
        if self.max_tier0_experts <= 0:
            return
        flat = self._filter_cpu_expert_ids(layer_idx, topk_ids.flatten())
        event = self._get_policy(layer_idx).record_accesses(flat.tolist())
        if self.trace_path:
            self._append_trace(
                {
                    "ts": time.time(),
                    "layer_idx": int(layer_idx),
                    "policy": self.residency_policy_name,
                    "capacity": int(self.max_tier0_experts),
                    "num_experts": int(self.num_experts),
                    **event,
                }
            )

    def prefetch_layer(self, layer_idx: int, topk_ids: np.ndarray):
        """
        Issue madvise(MADV_WILLNEED) for a specific layer's upcoming experts.

        Skips experts already promoted to Tier 0 (they're in NUMA-local malloc,
        no page fault possible). This avoids the madvise syscall storm (PERF-3).
        """
        if layer_idx not in self.mmap_regions:
            return

        moe = self.moe_refs.get(layer_idx)
        unique_ids = np.unique(self._filter_cpu_expert_ids(layer_idx, topk_ids.flatten()))
        layer_regions = self.mmap_regions[layer_idx]

        for eid in unique_ids:
            if eid < 0:
                continue
            # Skip Tier 0 experts — already in NUMA-local malloc, no prefetch needed
            if moe is not None and moe.is_expert_promoted(int(eid)):
                continue
            for proj_name, regions in layer_regions.items():
                del proj_name
                for region in regions[eid]:
                    region.prefetch()

    def start_promotion_thread(self):
        """Start background thread that promotes hot experts to Tier 0."""
        if self._running or self.max_tier0_experts <= 0:
            return
        self._running = True
        self._promotion_thread = threading.Thread(target=self._promotion_loop, daemon=True)
        self._promotion_thread.start()

    def stop_promotion_thread(self):
        """Stop the background promotion thread."""
        self._running = False
        if self._promotion_thread is not None:
            self._promotion_thread.join(timeout=10)
            self._promotion_thread = None

    def _promotion_loop(self):
        """Background loop: periodically promote hot experts to Tier 0."""
        while self._running:
            time.sleep(self.promotion_interval_sec)
            try:
                self._maybe_promote()
            except Exception as e:
                print(f"[TieredWeightProvider] promotion error: {e}")

    def _append_trace(self, payload: Dict[str, Any]) -> None:
        trace_path = self.trace_path
        if not trace_path:
            return
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._trace_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _maybe_promote(self):
        """
        Promote/demote experts via C++ MOE objects.

        Calls moe.promote_expert(eid) which allocates NUMA-local buffers,
        copies weight data, and atomically swaps the live pointers.
        Calls moe.demote_expert(eid) to restore baseline pointers and free NUMA buffers.
        """
        if self.max_tier0_experts <= 0 or not self.moe_refs:
            return
        for layer_idx, moe in list(self.moe_refs.items()):
            hot_ids = set(self._get_policy(layer_idx).resident_ids())
            promoted = []
            demoted = []
            # Promote hot experts not yet in Tier 0
            for eid in hot_ids:
                if self._is_gpu_expert(layer_idx, eid):
                    continue
                if not moe.is_expert_promoted(eid):
                    moe.promote_expert(eid)
                    promoted.append(int(eid))

            # Demote cold experts back to baseline (mmap or legacy)
            for eid in range(self.num_experts):
                if self._is_gpu_expert(layer_idx, eid):
                    if moe.is_expert_promoted(eid):
                        moe.demote_expert(eid)
                        demoted.append(int(eid))
                    continue
                if eid not in hot_ids and moe.is_expert_promoted(eid):
                    moe.demote_expert(eid)
                    demoted.append(int(eid))

            if self.trace_enabled:
                hot_preview = sorted(int(x) for x in hot_ids)[: self.trace_hot_k]
                if promoted or demoted or hot_preview:
                    print(
                        "[TieredWeightProviderTrace] "
                        f"layer={layer_idx} policy={self.residency_policy_name} hot_count={len(hot_ids)} "
                        f"hot_preview={hot_preview} "
                        f"promote_count={len(promoted)} promote={promoted[:self.trace_hot_k]} "
                        f"demote_count={len(demoted)} demote={demoted[:self.trace_hot_k]}"
                    )
