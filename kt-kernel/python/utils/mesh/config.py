"""
MESH 插件 Python 侧配置。

从环境变量读取 MESH 配置，转换为 C++ 侧 MeshConfig。
环境变量：
  KT_ENABLE_MESH         : 是否启用 MESH（"1"/"0"）
  KT_MESH_CAP            : 单层单 TP 的 slot 数
  KT_NUM_GPU_EXPERTS     : GE，过渡阶段搬运的 GPU 专家数
  KT_MAX_DEFERRED_EXPERTS_PER_TOKEN : 每 token 最多 defer 的专家数
  KT_MESH_DECODE_FRONT_LAYERS : decode 前 N 层满配，默认 5
  KT_MESH_DECODE_FRONT_LAYER_CAP : 前 N 层 cap，-1 = 拉满
  KT_MESH_TOTAL_LAYERS   : 模型总层数
  KT_MESH_PREFILL_WINDOW : prefill 窗口宽度，默认 1
  KT_MESH_HEAT_GAMMA     : Heat 层内 EMA 衰减率，默认 0.7
  KT_MESH_HEAT_BETA      : Heat 跨 token 衰减率，默认 0.5
  KT_MESH_MARKOV_ALPHA   : Markov 转移矩阵更新率，默认 0.5
  KT_MESH_MARKOV_TOPK    : Markov 每行稀疏保留数，默认 16
  KT_MESH_LOOKAHEAD_WEIGHT : 驱逐评分权重，默认 1.0
  KT_MESH_WEIGHT_TYPE    : 权重类型 "amxint4" / "bf16"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeshConfig:
    """MESH 配置，对应 C++ 侧 mesh::MeshConfig。"""

    # 基本开关
    enabled: bool = False

    # 容量配置
    cap: int = 0  # 单层单 TP 的 slot 数

    # GPU expert 配置
    num_gpu_experts: int = 0  # GE

    # Expert Defer 配置
    max_deferred_per_token: int = 3

    # Decode 前 N 层满配
    decode_front_layers: int = 5
    decode_front_layer_cap: int = -1  # -1 = 拉满

    # 时序配置
    total_layers: int = 0
    prefill_window: int = 1

    # Heat EMA 参数
    heat_gamma: float = 0.7
    heat_beta: float = 0.5

    # Markov 转移矩阵参数
    markov_alpha: float = 0.5
    markov_topk: int = 16

    # 驱逐评分权重
    lookahead_weight: float = 1.0

    # 权重类型
    weight_type: str = "amxint4"  # "amxint4" / "bf16"

    # 模型维度（由 wrapper 注入，不从 env 读）
    hidden_size: int = 0
    intermediate_size: int = 0
    expert_num: int = 0
    tp_count: int = 1

    @classmethod
    def from_env(cls) -> "MeshConfig":
        """从环境变量读取配置。"""
        def _env_bool(key: str, default: bool = False) -> bool:
            val = os.environ.get(key, "")
            return val.lower() in ("1", "true", "yes", "on")

        def _env_int(key: str, default: int = 0) -> int:
            try:
                return int(os.environ.get(key, str(default)))
            except ValueError:
                return default

        def _env_float(key: str, default: float = 0.0) -> float:
            try:
                return float(os.environ.get(key, str(default)))
            except ValueError:
                return default

        return cls(
            enabled=_env_bool("KT_ENABLE_MESH"),
            cap=_env_int("KT_MESH_CAP"),
            num_gpu_experts=_env_int("KT_NUM_GPU_EXPERTS"),
            max_deferred_per_token=_env_int("KT_MAX_DEFERRED_EXPERTS_PER_TOKEN", 3),
            decode_front_layers=_env_int("KT_MESH_DECODE_FRONT_LAYERS", 5),
            decode_front_layer_cap=_env_int("KT_MESH_DECODE_FRONT_LAYER_CAP", -1),
            total_layers=_env_int("KT_MESH_TOTAL_LAYERS"),
            prefill_window=_env_int("KT_MESH_PREFILL_WINDOW", 1),
            heat_gamma=_env_float("KT_MESH_HEAT_GAMMA", 0.7),
            heat_beta=_env_float("KT_MESH_HEAT_BETA", 0.5),
            markov_alpha=_env_float("KT_MESH_MARKOV_ALPHA", 0.5),
            markov_topk=_env_int("KT_MESH_MARKOV_TOPK", 16),
            lookahead_weight=_env_float("KT_MESH_LOOKAHEAD_WEIGHT", 1.0),
            weight_type=os.environ.get("KT_MESH_WEIGHT_TYPE", "amxint4").lower(),
        )

    def validate(self) -> None:
        """校验配置合法性。"""
        if not self.enabled:
            return
        if self.cap <= 0:
            raise ValueError("KT_MESH_CAP must be positive when MESH is enabled")
        if self.total_layers <= 0:
            raise ValueError("KT_MESH_TOTAL_LAYERS must be positive when MESH is enabled")
        if self.expert_num <= 0:
            raise ValueError("expert_num must be positive when MESH is enabled")
        if self.weight_type not in ("amxint4", "bf16"):
            raise ValueError(f"Unknown weight_type: {self.weight_type}")
        if self.max_deferred_per_token < 0:
            raise ValueError("max_deferred_per_token must be non-negative")

    def to_cpp(self):
        """转换为 C++ 侧 MeshConfig 对象。

        需要 kt_kernel_ext.mesh.MeshConfig 已绑定。
        """
        try:
            from kt_kernel_ext.mesh import MeshConfig as CppMeshConfig
        except ImportError as e:
            raise RuntimeError(
                "MESH C++ extension not available. "
                "Please build kt-kernel with CPUINFER_ENABLE_MESH=ON"
            ) from e

        cpp_cfg = CppMeshConfig()
        cpp_cfg.enabled = self.enabled
        cpp_cfg.cap = self.cap
        cpp_cfg.num_gpu_experts = self.num_gpu_experts
        cpp_cfg.max_deferred_per_token = self.max_deferred_per_token
        cpp_cfg.decode_front_layers = self.decode_front_layers
        cpp_cfg.decode_front_layer_cap = self.decode_front_layer_cap
        cpp_cfg.total_layers = self.total_layers
        cpp_cfg.prefill_window = self.prefill_window
        cpp_cfg.heat_gamma = self.heat_gamma
        cpp_cfg.heat_beta = self.heat_beta
        cpp_cfg.markov_alpha = self.markov_alpha
        cpp_cfg.markov_topk = self.markov_topk
        cpp_cfg.lookahead_weight = self.lookahead_weight
        # weight_type 枚举
        if self.weight_type == "amxint4":
            cpp_cfg.weight_type = 0  # WeightType::AMXINT4
        else:
            cpp_cfg.weight_type = 1  # WeightType::BF16
        cpp_cfg.hidden_size = self.hidden_size
        cpp_cfg.intermediate_size = self.intermediate_size
        cpp_cfg.expert_num = self.expert_num
        cpp_cfg.tp_count = self.tp_count
        return cpp_cfg
