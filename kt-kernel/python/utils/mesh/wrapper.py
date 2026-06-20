"""
MESH 模式的 MoE Wrapper。

不继承 AMXMoEWrapper，而是持有它的实例并注入 mesh_enabled=True。
AMX 内核代码完全复用，只在权重加载和 forward 指针重定向时走 mesh 路径。

职责：
1. 创建 AMXMoEWrapper（复用全部 AMX 内核）
2. 创建 ResidencyManager 并注入文件布局 + gpu_experts_mask
3. 在 forward 时调用 ResidencyManager 的回调
"""

from __future__ import annotations

import logging
import os
import torch
from typing import List, Optional

from .config import MeshConfig
from .residency import MeshResidencyManager

logger = logging.getLogger(__name__)


class MeshMoEWrapper:
    """MESH 模式的 MoE Wrapper。

    委托给 AMXMoEWrapper 进行实际计算，MESH 只负责权重驻留管理。

    ResidencyManager 是进程级单例：所有层共享同一个 manager，
    避免 40× 内存重复（每层都创建完整 40 层 pool）。
    """

    # 进程级单例：key=numa_nodes tuple, value=MeshResidencyManager
    _shared_residency = None
    _shared_bootstrap_done = False

    def __init__(
        self,
        layer_idx: int,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        moe_intermediate_size: int,
        gpu_experts_mask: Optional[torch.Tensor],
        cpuinfer_threads: int,
        threadpool_count: int,
        weight_path: str,
        chunked_prefill_size: int,
        mesh_config: MeshConfig,
        method: str = "AMXINT4",
        numa_nodes: Optional[List[int]] = None,
        cpu_save: bool = False,
        max_deferred_experts_per_token: Optional[int] = None,
    ):
        """初始化 MESH MoE Wrapper。

        Args:
            layer_idx: 层索引
            num_experts: 专家总数
            num_experts_per_tok: 每 token 的 top-k
            hidden_size: 隐藏维度
            moe_intermediate_size: MoE 中间维度
            gpu_experts_mask: GPU expert mask
            cpuinfer_threads: CPU 推理线程数
            threadpool_count: NUMA 子池数（TP 数）
            weight_path: 权重路径
            chunked_prefill_size: prefill chunk 大小
            mesh_config: MESH 配置
            method: 后端方法（AMXINT4 / BF16）
            numa_nodes: NUMA 节点列表
            cpu_save: 是否保存权重到 CPU 内存
            max_deferred_experts_per_token: 每 token defer 数
        """
        # 注入模型维度到 mesh_config
        mesh_config.hidden_size = hidden_size
        mesh_config.intermediate_size = moe_intermediate_size
        mesh_config.expert_num = num_experts
        mesh_config.tp_count = threadpool_count
        if mesh_config.total_layers <= 0:
            # 如果未设置，需要外部补充（通常由调用者设置）
            logger.warning("mesh_config.total_layers not set, MESH schedule_key may be incorrect")

        # 1. 创建 AMXMoEWrapper，注入 mesh_enabled=True
        #    AMX 内核代码完全复用，load_weights() 会走 mesh 分支（直接 return）
        from ..amx import AMXMoEWrapper
        self._inner = AMXMoEWrapper(
            layer_idx=layer_idx,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            gpu_experts_mask=gpu_experts_mask,
            cpuinfer_threads=cpuinfer_threads,
            threadpool_count=threadpool_count,
            weight_path=weight_path,
            chunked_prefill_size=chunked_prefill_size,
            cpu_save=cpu_save,
            max_deferred_experts_per_token=max_deferred_experts_per_token,
            method=method,
            numa_nodes=numa_nodes,
        )

        # 2. 创建/复用 ResidencyManager（进程级单例，避免 40× 内存重复）
        if numa_nodes is None:
            numa_nodes = list(range(threadpool_count))

        if MeshMoEWrapper._shared_residency is None:
            # 第一层：创建并初始化共享 ResidencyManager
            self._residency = MeshResidencyManager()
            self._residency.init(mesh_config, numa_nodes)

            # 注入 GPU expert mask
            if gpu_experts_mask is not None:
                mask_list = gpu_experts_mask.cpu().tolist()
                mask_int = [1 if x else 0 for x in mask_list]
                self._residency.set_gpu_experts_mask(mask_int)

            # 注入文件布局（从 SafeTensorLoader 获取）
            self._inject_file_layouts(weight_path, threadpool_count, num_experts, mesh_config)

            # 启动阶段：读前 cap 个专家进 slot
            self._residency.mark_layouts_set()
            self._residency.bootstrap()
            MeshMoEWrapper._shared_residency = self._residency
            MeshMoEWrapper._shared_bootstrap_done = True
            logger.info(
                "MeshMoEWrapper: shared ResidencyManager created at layer=%d, cap=%d, method=%s",
                layer_idx, mesh_config.cap, method,
            )
        else:
            # 后续层：复用共享 ResidencyManager
            self._residency = MeshMoEWrapper._shared_residency
            logger.info(
                "MeshMoEWrapper: layer=%d reuses shared ResidencyManager",
                layer_idx,
            )

        self._mesh_config = mesh_config
        self._layer_idx = layer_idx

    def _inject_file_layouts(
        self,
        weight_path: str,
        tp_count: int,
        expert_num: int,
        mesh_config: MeshConfig,
    ) -> None:
        """从 SafeTensorLoader 获取文件布局并注入 ResidencyManager。

        这里需要根据实际权重文件格式解析偏移。
        简化实现：假设权重已按 TP 切分存储，每个 TP 一个文件。
        """
        # TODO: 实际实现需要根据 SafeTensorLoader 的接口获取每个专家的文件偏移
        # 这里只提供框架，具体偏移计算依赖权重文件格式
        #
        # for tp in range(tp_count):
        #     for eid in range(expert_num):
        #         layout = loader.get_expert_layout(tp, eid)
        #         fd = os.open(layout.path, os.O_DIRECT | os.O_RDONLY)
        #         self._residency.set_file_layout(
        #             tp, eid, fd,
        #             layout.gate_offset, layout.up_offset, layout.down_offset,
        #             layout.gate_bytes, layout.up_bytes, layout.down_bytes,
        #             ...
        #         )
        logger.warning(
            "File layout injection not yet implemented. "
            "Please implement _inject_file_layouts based on SafeTensorLoader."
        )

    # ===== 属性委托 =====

    def __getattr__(self, name: str):
        """未在 MeshMoEWrapper 中定义的属性，委托给内部 AMXMoEWrapper。

        确保 submit_forward / sync_forward 等基类方法自动透传，
        无需逐个手写委托。__getattr__ 仅在正常属性查找失败时调用，
        不会覆盖本类已显式定义的方法（forward / load_weights 等）。
        """
        inner = self.__dict__.get("_inner")
        if inner is not None:
            return getattr(inner, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ===== forward 委托 =====

    def forward(self, *args, **kwargs):
        """forward 委托给 AMXMoEWrapper。

        MESH 通过 hook 在 AMX 内核中重定向权重指针，
        Python 侧无需特殊处理。
        """
        return self._inner.forward(*args, **kwargs)

    def load_weights(self, physical_to_logical_map_cpu=None):
        """load_weights：委托给内部 AMXMoEWrapper 加载全量权重。

        TODO: MESH 模式的最终目标是跳过全量加载，仅由 ResidencyManager 的
        slot pool 管理 cap 个专家（io_uring 按需加载）。但当前 io_uring 按需
        加载和 AMX 内核 hook 尚未实现，self.moe 为 None 会导致 submit_forward
        崩溃。暂时先正常加载权重以保证推理可用，slot pool 作为额外开销。
        """
        return self._inner.load_weights(physical_to_logical_map_cpu)

    # ===== MESH 专属接口 =====

    @property
    def residency(self) -> MeshResidencyManager:
        """获取 ResidencyManager。"""
        return self._residency

    @property
    def mesh_config(self) -> MeshConfig:
        return self._mesh_config

    @property
    def inner(self):
        """获取内部 AMXMoEWrapper。"""
        return self._inner

    # ===== 生命周期回调 =====

    def on_prefill_layer_start(self, qlen: int, active_experts: List[int]) -> None:
        self._residency.on_prefill_layer_start(self._layer_idx, qlen, active_experts)

    def on_prefill_layer_done(self) -> None:
        self._residency.on_prefill_layer_done(self._layer_idx)

    def on_prefill_to_decode(self) -> None:
        self._residency.on_prefill_to_decode()

    def on_decode_token_start(self) -> None:
        self._residency.on_decode_token_start()

    def on_decode_layer(self, topk: List[int], scores: List[float], tp_part_idx: int):
        return self._residency.on_decode_layer(self._layer_idx, topk, scores, tp_part_idx)

    def on_decode_token_end(self, all_layers_topk: List[List[int]],
                            all_layers_scores: List[List[float]]) -> None:
        self._residency.on_decode_token_end(all_layers_topk, all_layers_scores)
