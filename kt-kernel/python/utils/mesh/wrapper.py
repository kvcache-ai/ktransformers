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
            # A2: mesh_enabled 暂不开启（避免 load_weights 跳过导致 gate_bb_ 为空崩溃）
            # 仅注入 mesh_residency 指针，让 do_gate_up_gemm 的 hook 可被调用
            mesh_enabled=False,
            mesh_residency_ptr=0,
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

        # A2: 注入 mesh_residency 指针到 inner AMXMoEWrapper
        # 这样 do_gate_up_gemm/do_down_gemm 中的 mesh hook 可以调用 ResidencyManager
        self._inner.mesh_residency_ptr = self._residency.raw_ptr()

    def _inject_file_layouts(
        self,
        weight_path: str,
        tp_count: int,
        expert_num: int,
        mesh_config: MeshConfig,
    ) -> None:
        """从 safetensors 文件解析专家权重偏移并注入 ResidencyManager。

        A4 实现：解析 safetensors 文件头获取每个 tensor 的绝对文件偏移，
        为每层每 TP 每专家构造 ExpertFileLayout 并注入。

        支持两种格式：
        - AMXINT4 NUMA-sharded: blk.{L}.ffn_{gate,up,down}_exps.{E}.numa.{N}.{weight,scale}
        - BF16 packed: model.layers.{L}.mlp.experts.{gate_up_proj,down_proj}
        """
        import struct
        import json

        total_layers = mesh_config.total_layers
        weight_type = mesh_config.weight_type

        # 1. 遍历所有 safetensors 文件，构建 tensor_name -> (file_path, abs_offset, bytes) 映射
        tensor_map: dict[str, tuple[str, int, int]] = {}
        for root, _, files in os.walk(weight_path):
            for fname in sorted(files):
                if not fname.endswith('.safetensors'):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, 'rb') as f:
                    header_len_bytes = f.read(8)
                    header_len = struct.unpack('<Q', header_len_bytes)[0]
                    header_raw = f.read(header_len).decode('utf-8')
                header = json.loads(header_raw)
                data_start = 8 + header_len
                for name, info in header.items():
                    if name == '__metadata__':
                        continue
                    if not isinstance(info, dict) or 'data_offsets' not in info:
                        continue
                    abs_offset = data_start + info['data_offsets'][0]
                    nbytes = info['data_offsets'][1] - info['data_offsets'][0]
                    tensor_map[name] = (fpath, abs_offset, nbytes)

        if not tensor_map:
            logger.warning(
                "_inject_file_layouts: no safetensors tensors found under %s, "
                "bootstrap will run in memory-test mode", weight_path,
            )
            return

        logger.info(
            "_inject_file_layouts: parsed %d tensors from safetensors under %s",
            len(tensor_map), weight_path,
        )

        # 2. 根据 weight_type 注入偏移
        if weight_type == "amxint4":
            self._inject_amxint4_layouts(tensor_map, total_layers, tp_count, expert_num)
        elif weight_type == "bf16":
            self._inject_bf16_layouts(
                tensor_map, total_layers, tp_count, expert_num, mesh_config,
            )
        else:
            logger.warning("_inject_file_layouts: unknown weight_type=%s, skipping", weight_type)

    def _inject_amxint4_layouts(
        self,
        tensor_map: dict,
        total_layers: int,
        tp_count: int,
        expert_num: int,
    ) -> None:
        """注入 AMXINT4 NUMA-sharded 格式的文件布局。

        Tensor 命名：
        - blk.{L}.ffn_gate_exps.{E}.numa.{N}.weight / .scale
        - blk.{L}.ffn_up_exps.{E}.numa.{N}.weight / .scale
        - blk.{L}.ffn_down_exps.{E}.numa.{N}.weight / .scale
        """
        # 收集所有需要打开的文件，用 O_DIRECT 打开
        file_fds: dict[str, int] = {}

        def get_fd(fpath: str) -> int:
            if fpath not in file_fds:
                # O_DIRECT 需要对齐读取，但 safetensors 偏移可能不对齐
                # 先用 O_RDONLY 打开，O_DIRECT 对齐问题在 io_uring 层处理
                fd = os.open(fpath, os.O_RDONLY)
                file_fds[fpath] = fd
                logger.debug("_inject_amxint4_layouts: opened %s as fd=%d", fpath, fd)
            return file_fds[fpath]

        injected = 0
        missing = 0
        for layer in range(total_layers):
            for tp in range(tp_count):
                for expert in range(expert_num):
                    # AMXINT4 NUMA-sharded tensor 名
                    gate_w_key = f"blk.{layer}.ffn_gate_exps.{expert}.numa.{tp}.weight"
                    gate_s_key = f"blk.{layer}.ffn_gate_exps.{expert}.numa.{tp}.scale"
                    up_w_key = f"blk.{layer}.ffn_up_exps.{expert}.numa.{tp}.weight"
                    up_s_key = f"blk.{layer}.ffn_up_exps.{expert}.numa.{tp}.scale"
                    down_w_key = f"blk.{layer}.ffn_down_exps.{expert}.numa.{tp}.weight"
                    down_s_key = f"blk.{layer}.ffn_down_exps.{expert}.numa.{tp}.scale"

                    # 检查必需的 tensor 是否存在
                    required = [gate_w_key, up_w_key, down_w_key]
                    if not all(k in tensor_map for k in required):
                        missing += 1
                        continue

                    # 获取 gate 布局
                    g_fpath, g_off, g_bytes = tensor_map[gate_w_key]
                    fd = get_fd(g_fpath)

                    # 获取 up 布局
                    u_fpath, u_off, u_bytes = tensor_map[up_w_key]
                    if u_fpath != g_fpath:
                        fd = get_fd(u_fpath)  # 不同文件需要不同 fd

                    # 获取 down 布局
                    d_fpath, d_off, d_bytes = tensor_map[down_w_key]
                    if d_fpath not in file_fds:
                        fd = get_fd(d_fpath)

                    # scale 偏移（AMXINT4 专用）
                    g_s_off = g_s_bytes = 0
                    u_s_off = u_s_bytes = 0
                    d_s_off = d_s_bytes = 0
                    if gate_s_key in tensor_map:
                        _, g_s_off, g_s_bytes = tensor_map[gate_s_key]
                    if up_s_key in tensor_map:
                        _, u_s_off, u_s_bytes = tensor_map[up_s_key]
                    if down_s_key in tensor_map:
                        _, d_s_off, d_s_bytes = tensor_map[down_s_key]

                    # 注意：fd 取 gate 所在文件的 fd。如果 up/down 在不同文件，
                    # io_uring 的 submit_load 会用 layout.fd 读取所有三个矩阵，
                    # 这要求三个矩阵在同一文件。safetensors 通常如此（同层同 NUMA 在同一文件）。
                    # 如果跨文件，需要后续拆分为多次 submit_load。
                    self._residency.set_file_layout(
                        layer_idx=layer,
                        tp_part_idx=tp,
                        expert_id=expert,
                        fd=fd,
                        gate_offset=g_off,
                        up_offset=u_off,
                        down_offset=d_off,
                        gate_bytes=g_bytes,
                        up_bytes=u_bytes,
                        down_bytes=d_bytes,
                        gate_scale_offset=g_s_off,
                        up_scale_offset=u_s_off,
                        down_scale_offset=d_s_off,
                        gate_scale_bytes=g_s_bytes,
                        up_scale_bytes=u_s_bytes,
                        down_scale_bytes=d_s_bytes,
                        # AMXINT4 对称量化无 mins，保持 0
                    )
                    injected += 1

        logger.info(
            "_inject_amxint4_layouts: injected %d layouts (missing %d), "
            "opened %d files with O_RDONLY",
            injected, missing, len(file_fds),
        )

    def _inject_bf16_layouts(
        self,
        tensor_map: dict,
        total_layers: int,
        tp_count: int,
        expert_num: int,
        mesh_config: MeshConfig,
    ) -> None:
        """注入 BF16 packed 格式的文件布局。

        Packed 格式：所有专家打包成 3D tensor
        - model.layers.{L}.mlp.experts.gate_up_proj  [E, 2*I, H]
        - model.layers.{L}.mlp.experts.down_proj     [E, H, I]

        每个 TP 分片读取 intermediate_size/tp_count 的切片。
        """
        hidden = mesh_config.hidden_size
        inter = mesh_config.intermediate_size
        inter_per_tp = inter // tp_count
        # bf16 = 2 bytes
        elem_bytes = 2

        file_fds: dict[str, int] = {}

        def get_fd(fpath: str) -> int:
            if fpath not in file_fds:
                fd = os.open(fpath, os.O_RDONLY)
                file_fds[fpath] = fd
            return file_fds[fpath]

        injected = 0
        missing = 0
        for layer in range(total_layers):
            gate_up_key = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            down_key = f"model.layers.{layer}.mlp.experts.down_proj"

            if gate_up_key not in tensor_map or down_key not in tensor_map:
                missing += 1
                continue

            gu_fpath, gu_base, _ = tensor_map[gate_up_key]
            d_fpath, d_base, _ = tensor_map[down_key]
            fd = get_fd(gu_fpath)

            # gate_up_proj: [E, 2*I, H]，每个专家占 2*I*H*elem_bytes
            expert_stride_gu = 2 * inter * hidden * elem_bytes
            gate_bytes = inter_per_tp * hidden * elem_bytes
            up_bytes = inter_per_tp * hidden * elem_bytes

            # down_proj: [E, H, I]，每个专家占 H*I*elem_bytes
            expert_stride_d = hidden * inter * elem_bytes
            down_bytes = hidden * inter_per_tp * elem_bytes
            # Bug 1 fix: BF16 down_proj [E,H,I] 行主序，TP 沿 I 切不连续
            # down_stride = 完整行字节数（I * elem_bytes），down_rows = H
            # tp_count > 1 时 down_stride > row_bytes，需逐行读取
            down_stride = inter * elem_bytes  # 完整行步长
            down_rows = hidden

            for tp in range(tp_count):
                for expert in range(expert_num):
                    # gate: expert 的 gate 部分起始偏移 + TP 切片偏移
                    gate_off = gu_base + expert * expert_stride_gu + tp * gate_bytes
                    # up: expert 的 up 部分起始偏移（跳过 gate） + TP 切片偏移
                    up_off = gu_base + expert * expert_stride_gu + inter * hidden * elem_bytes + tp * up_bytes
                    # down: expert 的 down 起始偏移 + TP 切片偏移
                    down_off = d_base + expert * expert_stride_d + tp * down_bytes

                    self._residency.set_file_layout(
                        layer_idx=layer,
                        tp_part_idx=tp,
                        expert_id=expert,
                        fd=fd,
                        gate_offset=gate_off,
                        up_offset=up_off,
                        down_offset=down_off,
                        gate_bytes=gate_bytes,
                        up_bytes=up_bytes,
                        down_bytes=down_bytes,
                        # BF16 无 scale/mins
                        # Bug 1 fix: 传入 down_stride/down_rows 支持不连续读取
                        down_stride=down_stride,
                        down_rows=down_rows,
                    )
                    injected += 1

        logger.info(
            "_inject_bf16_layouts: injected %d layouts (missing %d), "
            "opened %d files",
            injected, missing, len(file_fds),
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
