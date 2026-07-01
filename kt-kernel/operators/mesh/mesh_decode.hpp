/**
 * @file mesh_decode.hpp
 * @brief Decode 策略：immediate/deferred 分组 + 在线驱逐 + 前 5 层满配
 *
 * 每 token 每层即时收集 CPU 侧 top-k 专家，按 defer 参数分组：
 * - immediate 组：CACHED 状态，本层立即计算
 * - deferred 组：BASELINE/LOADING 状态，异步 io_uring 读取，推迟到下一层
 *
 * 分组逻辑（按 defer 参数对齐）：
 * - 目标 immediate 数 = k - defer
 * - 若 immediate数 > k-defer：按 router score 排序取前 k-defer 个
 * - 若 immediate数 < k-defer：阻塞等待异步读，直到 immediate 数量够
 *
 * schedule_key：
 * - immediate: timeline_step × total_layers + layer_idx
 * - deferred: timeline_step × total_layers + layer_idx + 1（跟下一层同优先级）
 *
 * 前 5 层允许更大容量（接近全量常驻），默认拉满。
 */
#pragma once

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_eviction.hpp"
#include "mesh_scheduler.hpp"
#include "mesh_slot_pool.hpp"

namespace mesh {

/**
 * @brief Decode 策略
 */
class MeshDecode {
 public:
  MeshDecode(const MeshConfig& config)
      : defer_count_(config.max_deferred_per_token),
        front_layers_(config.decode_front_layers),
        front_layer_cap_(config.decode_front_layer_cap),
        total_layers_(config.total_layers),
        expert_num_(config.expert_num) {}

  // immediate/deferred 分组结果
  struct SplitResult {
    std::vector<int> immediate;           // 本层立即计算（CACHED + GPU + 阻塞读完成的 missing）
    std::vector<int> deferred;            // 推迟到下一层计算（cached + missing）
    std::vector<int> blocking_missing;    // SKILL.md 第 282 行：需要阻塞读补充 immediate 组的 missing 专家
    std::vector<int> deferred_missing;    // SKILL.md 第 284 行：deferred 组中需要提交异步 io_uring 的 missing 专家
  };

  /**
   * @brief 对 top-k 专家进行 immediate/deferred 分组（SKILL.md 第 280-284 行）
   *
   * 分组逻辑（与 KTransformers select_deferred_experts 一致）：
   * - target_immediate = original_k - defer_count（= KTransformers 的 protected_k）
   * - 按 score 降序排序整个 topk，取 score 最高的 target_immediate 个作为 immediate 候选
   *   （与 KTransformers 的 torch.topk(expert_scores, k=protected_k) 一致）
   * - immediate 候选中：GPU 专家直接进 immediate；CPU 已 CACHED 进 immediate；
   *   CPU 未 CACHED 加入 blocking_missing 阻塞读后进 immediate（SKILL.md 第 282 行）
   * - 其余 topk 走 deferred，其中未 CACHED 的提交异步 io_uring（SKILL.md 第 284 行）
   *
   * 一致性保证：MESH 的 immediate 组 = KTransformers 的 immediate 组，
   * GEMM 计算的 CPU 专家都已 CACHED（未 CACHED 的会被阻塞读补充），
   * 不会走同步 SSD 加载路径。
   *
   * 注意：split 只负责分组，不提交 io_uring。blocking_missing 由 on_decode_layer 阻塞读处理，
   * deferred_missing 由 on_decode_layer 异步提交 io_uring。
   *
   * @param topk effective_topk（含上一层层 deferred 已 CACHED 的专家）
   * @param original_k 原始 top-k 数（当前 token 路由选出的，不含 prev_layer_deferred_）
   */
  SplitResult split(const std::vector<int>& topk,
                    const std::vector<float>& scores,
                    MeshSlotPool& slot_pool,
                    int layer_idx,
                    int original_k,
                    std::function<bool(int)> is_gpu_expert = nullptr) {
    // SKILL.md 第 280 行：target_immediate 基于原始 top-k 数，不是 effective_topk.size()
    // = KTransformers 的 protected_k = num_experts_per_tok - max_deferred_experts_per_token
    int target_immediate = std::max(0, original_k - defer_count_);

    // 过滤掉 -1（KTransformers select_deferred_experts 把 deferred 位置填 -1）
    // on_decode_layer hook 接收 GEMM 的 expert_ids，defer 模式下含 -1
    // -1 不是有效专家 ID，提交 io_uring 会报 Bad file descriptor
    std::vector<int> valid_topk;
    valid_topk.reserve(topk.size());
    for (int eid : topk) {
      if (eid >= 0) {
        valid_topk.push_back(eid);
      }
    }

    // 与 KTransformers select_deferred_experts 一致：按 score 降序排序整个 topk，
    // 取 score 最高的 target_immediate 个作为 immediate 候选。
    // KTransformers 用 torch.topk(expert_scores, k=protected_k) 选 score 最高的位置，
    // MESH 用 scores[expert_id] 排序（top-k 无重复时两者等价）。
    std::sort(valid_topk.begin(), valid_topk.end(),
              [&scores](int a, int b) {
                float sa = (a < (int)scores.size()) ? scores[a] : 0.0f;
                float sb = (b < (int)scores.size()) ? scores[b] : 0.0f;
                return sa > sb;  // 降序
              });

    int n_immediate = std::min(static_cast<int>(valid_topk.size()), target_immediate);

    SplitResult result;
    // 遍历 immediate 候选，区分 GPU/CPU 和 CACHED/missing
    for (int i = 0; i < n_immediate; i++) {
      int eid = valid_topk[i];
      if (is_gpu_expert && is_gpu_expert(eid)) {
        // GPU 专家直接进 immediate（由 GPU 计算，不需 slot）
        result.immediate.push_back(eid);
      } else if (slot_pool.is_cached(eid)) {
        // CPU 专家已 CACHED，进 immediate
        result.immediate.push_back(eid);
      } else {
        // CPU 专家未 CACHED，阻塞读后进 immediate（SKILL.md 第 282 行）
        result.blocking_missing.push_back(eid);
        result.immediate.push_back(eid);  // 阻塞读完成后会 CACHED
      }
    }

    // 其余 topk 走 deferred（SKILL.md 第 284 行）
    for (int i = n_immediate; i < static_cast<int>(valid_topk.size()); i++) {
      int eid = valid_topk[i];
      result.deferred.push_back(eid);
      if (!is_gpu_expert || !is_gpu_expert(eid)) {
        if (!slot_pool.is_cached(eid)) {
          result.deferred_missing.push_back(eid);
        }
      }
    }

    return result;
  }

  /**
   * @brief 在线驱逐：slot 满时选分数最低的覆盖
   *
   * B1 fix: 实现完整的驱逐逻辑
   * 1. 找 victim 专家（分数最低的 CACHED 专家）
   * 2. 查 victim 的 slot_idx
   * 3. 调用 overwrite(slot_idx, new_expert_id)
   *
   * 引用计数保护：overwrite 内部 CV 等待 active_readers 归零，
   * GEMM 时 acquire_reader 持有的专家不会被驱逐（SKILL.md 第 62/125 行）。
   * immediate 组是 score 最高的 CACHED 专家，evict 选 score 最低的，不会选 immediate 组。
   *
   * @param slot_pool 该层 slot 池
   * @param scorer 驱逐评分器
   * @param layer_idx 当前层
   * @param new_expert_id 要加载的新专家
   * @return int 被驱逐释放的 slot_idx，-1 表示无需驱逐或无法驱逐
   */
  int evict_for_new_expert(MeshSlotPool& slot_pool,
                           const EvictionScorer& scorer,
                           int layer_idx, int new_expert_id) {
    // 前五层满配检查：用实际 slot 数量判断
    // 当 slot_pool.cap() < expert_num_ 时，前 5 层也无法容纳所有专家，必须允许驱逐
    if (is_front_layer(layer_idx)) {
      if (slot_pool.cap() >= expert_num_) {
        return -1;  // 实际 slot 数 >= 专家总数，满配无需驱逐
      }
    }

    // 找可驱逐的 slot
    auto cached = slot_pool.cached_experts();
    if (cached.empty()) return -1;

    // 选分数最低的
    int victim_expert = scorer.select_victim(cached, layer_idx);
    if (victim_expert < 0) return -1;

    // B1 fix: 查找 victim 对应的 slot_idx
    int victim_slot_idx = slot_pool.expert_to_slot_idx(victim_expert);
    if (victim_slot_idx < 0) return -1;

    // 检查 victim 是否有活跃 reader
    // overwrite 内部会 spin wait 等 reader 归零
    // 调用 overwrite 覆盖 slot
    slot_pool.overwrite(victim_slot_idx, new_expert_id);

    return victim_slot_idx;
  }

  // ===== 前 5 层满配 =====

  bool is_front_layer(int layer_idx) const {
    return layer_idx < front_layers_;
  }

  int get_effective_cap(int layer_idx, int default_cap) const {
    if (is_front_layer(layer_idx)) {
      return (front_layer_cap_ < 0) ? expert_num_ : front_layer_cap_;
    }
    return default_cap;
  }

  // ===== 访问器 =====
  int defer_count() const { return defer_count_; }
  int front_layers() const { return front_layers_; }

 private:
  int defer_count_;
  int front_layers_;
  int front_layer_cap_;  // -1 = 拉满
  int total_layers_;
  int expert_num_;
};

}  // namespace mesh
