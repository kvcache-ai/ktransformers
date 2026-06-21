/**
 * @file mesh_config.hpp
 * @brief MESH 配置结构体
 *
 * 纯配置 POD，由 Python 侧通过 pybind11 注入。
 * GeneralMOEConfig 持有 MeshConfig* 指针，默认 nullptr。
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mesh {

// 权重类型，决定 slot 内存大小
enum class WeightType : uint8_t {
  AMXINT4,  // 量化格式：weight(int4) + scale + mins
  BF16,     // 全精度 bf16
};

// 单个专家在某个 TP 分片上的文件布局
struct ExpertFileLayout {
  int fd = -1;            // O_DIRECT 打开的 safetensors 文件描述符
  off_t gate_offset = 0;  // gate 矩阵在文件中的偏移
  off_t up_offset = 0;    // up 矩阵偏移
  off_t down_offset = 0;  // down 矩阵偏移
  off_t gate_scale_offset = 0;  // AMXINT4 专用：gate scale 偏移
  off_t up_scale_offset = 0;    // AMXINT4 专用：up scale 偏移
  off_t down_scale_offset = 0;  // AMXINT4 专用：down scale 偏移
  off_t gate_mins_offset = 0;   // AMXINT4 专用：gate mins 偏移
  off_t up_mins_offset = 0;     // AMXINT4 专用：up mins 偏移
  off_t down_mins_offset = 0;   // AMXINT4 专用：down mins 偏移
  size_t gate_bytes = 0;        // gate 权重字节数
  size_t up_bytes = 0;          // up 权重字节数
  size_t down_bytes = 0;        // down 权重字节数
  size_t gate_scale_bytes = 0;  // scale 字节数（很小，几 KB）
  size_t up_scale_bytes = 0;
  size_t down_scale_bytes = 0;
  size_t gate_mins_bytes = 0;
  size_t up_mins_bytes = 0;
  size_t down_mins_bytes = 0;
};

// MESH 配置
struct MeshConfig {
  // ===== 基本开关 =====
  bool enabled = false;  // 是否启用 MESH，false 时走原版 KT 路径

  // ===== 容量配置 =====
  int cap = 0;  // 单层单 TP 的 slot 数（该层该 TP 最多驻留的 CPU expert shard 数）

  // ===== GPU expert 配置 =====
  int num_gpu_experts = 0;  // GE：过渡阶段要搬运到 GPU 的专家数

  // ===== Expert Defer 配置 =====
  int max_deferred_per_token = 3;  // 每 token 最多 defer 的专家数，对齐 KT 的 --kt-max-deferred-experts-per-token

  // ===== Decode 前 N 层满配 =====
  int decode_front_layers = 5;       // 前 N 层接近满配，默认 5
  int decode_front_layer_cap = -1;   // 前 N 层的 cap，-1 = 该层 CPU 专家总数（拉满）

  // ===== 时序配置 =====
  int total_layers = 0;      // 模型总层数，用于 schedule_key 计算
  int prefill_window = 1;    // prefill 窗口宽度，默认 1（只预取下一层）

  // ===== Heat EMA 参数 =====
  float heat_gamma = 0.7f;  // token 内层内 EMA：heat_new = gamma * heat_old + (1-gamma) * score
  float heat_beta = 0.5f;   // 跨 token 全局 EMA：几何衰减

  // ===== Markov 转移矩阵参数 =====
  float markov_alpha = 0.5f;  // 转移矩阵更新率
  int markov_topk = 16;       // 每行稀疏保留的目标专家数

  // ===== 驱逐评分权重 =====
  float lookahead_weight = 1.0f;  // score = policy_rank + lookahead_weight * heat

  // ===== 权重类型 =====
  WeightType weight_type = WeightType::AMXINT4;

  // ===== 模型维度（用于计算 slot 内存大小）=====
  int hidden_size = 0;            // 如 Qwen3.5-35B: 2048
  int intermediate_size = 0;      // 如 512
  int expert_num = 0;             // 单层专家总数
  int tp_count = 1;               // TP 分片数（= NUMA 节点数）

  // ===== 文件布局 =====
  // [tp_part_idx][expert_id] -> ExpertFileLayout
  // 由 Python 侧通过 set_file_layout 注入
  // 实际存储在 MeshResidencyManager 中，这里只放声明
};

// B9: 运行时统计结构体
struct MeshStats {
  // 命中率
  uint64_t cache_hit_count = 0;
  uint64_t cache_miss_count = 0;

  // io_uring 读取量
  uint64_t io_uring_read_bytes = 0;
  uint64_t io_uring_read_count = 0;

  // 驱逐统计
  uint64_t eviction_count = 0;
  uint64_t eviction_blocked_wait_us = 0;

  // defer 统计
  uint64_t defer_count = 0;
  uint64_t defer_overflow_count = 0;

  // prefill 统计
  uint64_t prefill_layer_count = 0;
  uint64_t prefill_temporal_swap_count = 0;

  // decode 统计
  uint64_t decode_token_count = 0;
  uint64_t decode_immediate_count = 0;
  uint64_t decode_deferred_count = 0;
};

}  // namespace mesh
