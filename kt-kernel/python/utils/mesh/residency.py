"""
MESH ResidencyManager Python 侧包装。

封装 C++ 侧 mesh::MeshResidencyManager，提供 Python 友好的接口。
负责：
- 初始化 slot 池、io_uring、调度器等组件
- 注入文件布局（safetensors 偏移）
- 注入 GPU expert mask
- 提供 forward 回调入口
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional, Tuple

from .config import MeshConfig

logger = logging.getLogger(__name__)


class MeshResidencyManager:
    """C++ 侧 MeshResidencyManager 的 Python 包装。"""

    def __init__(self):
        try:
            from kt_kernel_ext.mesh import ResidencyManager as CppManager
        except ImportError as e:
            raise RuntimeError(
                "MESH C++ extension not available. "
                "Please build kt-kernel with CPUINFER_ENABLE_MESH=ON"
            ) from e
        self._mgr = CppManager()
        self._initialized = False
        self._layouts_set = False

    def init(self, config: MeshConfig, numa_nodes: List[int]) -> None:
        """初始化 MESH。

        Args:
            config: MESH 配置
            numa_nodes: 每个 TP 分片对应的 NUMA 节点 ID
        """
        config.validate()
        cpp_cfg = config.to_cpp()
        self._mgr.init(cpp_cfg, numa_nodes)
        self._initialized = True
        logger.info(
            "MESH initialized: cap=%d, GE=%d, layers=%d, tp=%d",
            config.cap, config.num_gpu_experts,
            config.total_layers, config.tp_count,
        )

    def set_file_layout(
        self,
        tp_part_idx: int,
        expert_id: int,
        fd: int,
        gate_offset: int,
        up_offset: int,
        down_offset: int,
        gate_bytes: int,
        up_bytes: int,
        down_bytes: int,
        gate_scale_offset: int = 0,
        up_scale_offset: int = 0,
        down_scale_offset: int = 0,
        gate_scale_bytes: int = 0,
        up_scale_bytes: int = 0,
        down_scale_bytes: int = 0,
        gate_mins_offset: int = 0,
        up_mins_offset: int = 0,
        down_mins_offset: int = 0,
        gate_mins_bytes: int = 0,
        up_mins_bytes: int = 0,
        down_mins_bytes: int = 0,
    ) -> None:
        """注入单个专家在某个 TP 分片上的文件布局。

        Args:
            tp_part_idx: TP 分片索引
            expert_id: 专家 ID
            fd: O_DIRECT 打开的文件描述符
            *_offset: 各矩阵在文件中的偏移
            *_bytes: 各矩阵的字节数
        """
        if not self._initialized:
            raise RuntimeError("Call init() before set_file_layout()")
        self._mgr.set_file_layout(
            tp_part_idx, expert_id, fd,
            gate_offset, up_offset, down_offset,
            gate_bytes, up_bytes, down_bytes,
            gate_scale_offset, up_scale_offset, down_scale_offset,
            gate_scale_bytes, up_scale_bytes, down_scale_bytes,
            gate_mins_offset, up_mins_offset, down_mins_offset,
            gate_mins_bytes, up_mins_bytes, down_mins_bytes,
        )

    def set_gpu_experts_mask(self, mask: List[int]) -> None:
        """注入 GPU expert mask。

        Args:
            mask: 长度 = expert_num 的列表，1 = GPU expert，0 = CPU expert
        """
        if not self._initialized:
            raise RuntimeError("Call init() before set_gpu_experts_mask()")
        self._mgr.set_gpu_experts_mask(mask)
        logger.info("GPU expert mask set: %d GPU experts", sum(mask))

    def bootstrap(self) -> None:
        """启动阶段：读前 cap 个专家进 slot。"""
        if not self._initialized:
            raise RuntimeError("Call init() before bootstrap()")
        if not self._layouts_set:
            raise RuntimeError("Call set_file_layout() before bootstrap()")
        logger.info("MESH bootstrap: loading first %d experts per layer", self._mgr.config().cap)
        self._mgr.bootstrap()
        logger.info("MESH bootstrap complete")

    def mark_layouts_set(self) -> None:
        """标记文件布局已全部注入。"""
        self._layouts_set = True

    # ===== 权重指针查询（KT 计算用）=====

    def get_gate_ptr(self, layer: int, tp: int, expert_id: int) -> int:
        """获取 gate 矩阵指针。"""
        return self._mgr.get_gate_ptr(layer, tp, expert_id)

    def get_up_ptr(self, layer: int, tp: int, expert_id: int) -> int:
        """获取 up 矩阵指针。"""
        return self._mgr.get_up_ptr(layer, tp, expert_id)

    def get_down_ptr(self, layer: int, tp: int, expert_id: int) -> int:
        """获取 down 矩阵指针。"""
        return self._mgr.get_down_ptr(layer, tp, expert_id)

    # ===== Prefill 阶段回调 =====

    def on_prefill_layer_start(self, layer_idx: int, qlen: int,
                                active_experts: List[int]) -> None:
        self._mgr.on_prefill_layer_start(layer_idx, qlen, active_experts)

    def on_prefill_layer_done(self, layer_idx: int) -> None:
        self._mgr.on_prefill_layer_done(layer_idx)

    # ===== 过渡阶段 =====

    def on_prefill_to_decode(self) -> None:
        """Prefill→Decode 过渡。

        在第一个 decode token 的第一层前调用。
        """
        logger.info("MESH prefill→decode handoff")
        self._mgr.on_prefill_to_decode()

    # ===== Decode 阶段回调 =====

    def on_decode_token_start(self) -> None:
        self._mgr.on_decode_token_start()

    def on_decode_layer(self, layer_idx: int, topk: List[int],
                        scores: List[float], tp_part_idx: int):
        """Decode 每层处理，返回 (immediate, deferred) 分组。"""
        return self._mgr.on_decode_layer(layer_idx, topk, scores, tp_part_idx)

    def on_decode_token_end(self, all_layers_topk: List[List[int]],
                            all_layers_scores: List[List[float]]) -> None:
        """单 token 结束后批量更新 Heat 和 Markov。"""
        self._mgr.on_decode_token_end(all_layers_topk, all_layers_scores)

    # ===== 访问器 =====

    @property
    def config(self):
        return self._mgr.config()

    @property
    def raw(self):
        """获取底层 C++ 对象指针（用于 hook 注册）。"""
        return self._mgr
