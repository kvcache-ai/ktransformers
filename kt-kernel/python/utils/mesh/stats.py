"""
MESH 统计信息收集。

读取 C++ 侧 MeshStats，供 benchmark 用。
包括：hit rate、io_uring read GiB、驱逐次数等。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MeshStats:
    """MESH 运行时统计。"""

    # 命中率
    cache_hit_count: int = 0
    cache_miss_count: int = 0

    # io_uring 读取量
    io_uring_read_bytes: int = 0
    io_uring_read_count: int = 0

    # 驱逐统计
    eviction_count: int = 0
    eviction_blocked_wait_us: int = 0  # 驱逐时等待 reader 归零的累计微秒

    # defer 统计
    defer_count: int = 0
    defer_overflow_count: int = 0  # overflow 通道触发次数

    # prefill 统计
    prefill_layer_count: int = 0
    prefill_temporal_swap_count: int = 0  # 指针 swap 次数

    # decode 统计
    decode_token_count: int = 0
    decode_immediate_count: int = 0
    decode_deferred_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hit_count + self.cache_miss_count
        if total == 0:
            return 0.0
        return self.cache_hit_count / total

    @property
    def io_uring_read_gib(self) -> float:
        return self.io_uring_read_bytes / (1024 ** 3)

    def summary(self) -> str:
        """返回可读的统计摘要。"""
        return (
            f"MESH Stats:\n"
            f"  cache hit rate: {self.cache_hit_rate:.2%} "
            f"({self.cache_hit_count}/{self.cache_hit_count + self.cache_miss_count})\n"
            f"  io_uring read: {self.io_uring_read_gib:.2f} GiB "
            f"({self.io_uring_read_count} requests)\n"
            f"  evictions: {self.eviction_count} "
            f"(blocked {self.eviction_blocked_wait_us} us)\n"
            f"  defer: {self.defer_count} (overflow {self.defer_overflow_count})\n"
            f"  prefill: {self.prefill_layer_count} layers, "
            f"{self.prefill_temporal_swap_count} swaps\n"
            f"  decode: {self.decode_token_count} tokens, "
            f"{self.decode_immediate_count} immediate, "
            f"{self.decode_deferred_count} deferred"
        )

    def to_kv(self) -> str:
        """返回单行 key=value 格式，方便从日志 grep 解析。"""
        return (
            f"hit_rate={self.cache_hit_rate:.4f} "
            f"hit_count={self.cache_hit_count} "
            f"miss_count={self.cache_miss_count} "
            f"iouring_read_gib={self.io_uring_read_gib:.2f} "
            f"iouring_read_count={self.io_uring_read_count} "
            f"eviction_count={self.eviction_count} "
            f"defer_count={self.defer_count} "
            f"defer_overflow={self.defer_overflow_count} "
            f"prefill_layers={self.prefill_layer_count} "
            f"prefill_swaps={self.prefill_temporal_swap_count} "
            f"decode_tokens={self.decode_token_count} "
            f"decode_immediate={self.decode_immediate_count} "
            f"decode_deferred={self.decode_deferred_count}"
        )


class MeshStatsCollector:
    """从 C++ 侧读取统计信息。"""

    def __init__(self, residency_manager):
        """Args:
            residency_manager: MeshResidencyManager 实例
        """
        self._mgr = residency_manager

    def collect(self) -> MeshStats:
        """收集当前统计信息。"""
        stats = MeshStats()
        try:
            # B9: C++ 侧已绑定 stats() 方法，返回 MeshStats 引用
            cpp_stats = self._mgr.raw.stats()
            stats.cache_hit_count = getattr(cpp_stats, "cache_hit_count", 0)
            stats.cache_miss_count = getattr(cpp_stats, "cache_miss_count", 0)
            stats.io_uring_read_bytes = getattr(cpp_stats, "io_uring_read_bytes", 0)
            stats.io_uring_read_count = getattr(cpp_stats, "io_uring_read_count", 0)
            stats.eviction_count = getattr(cpp_stats, "eviction_count", 0)
            stats.eviction_blocked_wait_us = getattr(cpp_stats, "eviction_blocked_wait_us", 0)
            stats.defer_count = getattr(cpp_stats, "defer_count", 0)
            stats.defer_overflow_count = getattr(cpp_stats, "defer_overflow_count", 0)
            stats.prefill_layer_count = getattr(cpp_stats, "prefill_layer_count", 0)
            stats.prefill_temporal_swap_count = getattr(cpp_stats, "prefill_temporal_swap_count", 0)
            stats.decode_token_count = getattr(cpp_stats, "decode_token_count", 0)
            stats.decode_immediate_count = getattr(cpp_stats, "decode_immediate_count", 0)
            stats.decode_deferred_count = getattr(cpp_stats, "decode_deferred_count", 0)
        except (AttributeError, RuntimeError) as e:
            logger.warning("Failed to collect MESH stats: %s", e)
        return stats

    def log_summary(self) -> None:
        """打印统计摘要。"""
        stats = self.collect()
        logger.info(stats.summary())
