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
    std::vector<int> immediate;  // 本层立即计算
    std::vector<int> deferred;   // 推迟到下一层计算
  };

  /**
   * @brief 对 top-k 专家进行 immediate/deferred 分组
   *
   * B6 fix: 入口先过滤 GPU 专家（GPU 专家不参与 slot 调度）
   *
   * @param topk 当前层 top-k 专家 ID
   * @param scores 各专家的 router score（全长 expert_num）
   * @param slot_pool 该层 slot 池
   * @param scheduler 调度器
   * @param layer_idx 当前层
   * @param tp_part_idx TP 分片
   * @param get_dst_ptrs 获取 slot buffer 指针的回调
   * @param is_gpu_expert 判断专家是否在 GPU 上的回调
   * @return SplitResult
   */
  SplitResult split(const std::vector<int>& topk,
                    const std::vector<float>& scores,
                    MeshSlotPool& slot_pool,
                    MeshScheduler& scheduler,
                    int layer_idx, int tp_part_idx,
                    std::function<std::vector<void*>(int)> get_dst_ptrs,
                    std::function<bool(int)> is_gpu_expert = nullptr) {
    int k = static_cast<int>(topk.size());
    int target_immediate = std::max(0, k - defer_count_);

    // B6 fix: 过滤 GPU 专家，GPU 专家直接进 immediate（由 GPU 计算，不需 slot）
    std::vector<int> cpu_topk;
    std::vector<int> gpu_experts;
    for (int eid : topk) {
      if (is_gpu_expert && is_gpu_expert(eid)) {
        gpu_experts.push_back(eid);
      } else {
        cpu_topk.push_back(eid);
      }
    }

    // 区分已缓存和缺失（仅 CPU 专家）
    std::vector<int> cached, missing;
    for (int eid : cpu_topk) {
      if (slot_pool.is_cached(eid)) {
        cached.push_back(eid);
      } else {
        missing.push_back(eid);
      }
    }

    SplitResult result;
    // GPU 专家直接进 immediate
    result.immediate = gpu_experts;

    if (static_cast<int>(cached.size()) > target_immediate) {
      // immediate 数 > k-defer：按 router score 排序取前 target_immediate 个
      std::sort(cached.begin(), cached.end(),
                [&scores](int a, int b) {
                  float sa = (a < (int)scores.size()) ? scores[a] : 0.0f;
                  float sb = (b < (int)scores.size()) ? scores[b] : 0.0f;
                  return sa > sb;  // 降序
                });
      result.immediate.insert(result.immediate.end(), cached.begin(), cached.begin() + target_immediate);
      result.deferred.assign(cached.begin() + target_immediate, cached.end());
      // 缺失专家全部进 deferred
      result.deferred.insert(result.deferred.end(), missing.begin(), missing.end());

      // 为缺失专家提交 deferred 异步读
      for (int eid : missing) {
        auto ptrs = get_dst_ptrs(eid);
        scheduler.submit_decode_deferred(layer_idx, eid, tp_part_idx,
                                         ptrs[0], ptrs[1], ptrs[2]);
      }
    } else {
      // immediate 数 < k-defer：阻塞读缺失专家直到 immediate 数量够
      result.immediate.insert(result.immediate.end(), cached.begin(), cached.end());
      int need = target_immediate - static_cast<int>(cached.size());

      for (int i = 0; i < need && i < static_cast<int>(missing.size()); i++) {
        int eid = missing[i];
        auto ptrs = get_dst_ptrs(eid);
        // 提交 immediate 异步读（schedule_key = 当前层）
        scheduler.submit_decode_immediate(layer_idx, eid, tp_part_idx,
                                          ptrs[0], ptrs[1], ptrs[2]);
        // 阻塞等待读取完成
        // 实际实现需要与 MeshIoUring 配合等待 CQE
        // io_.wait_expert(eid, n_reqs);
        result.immediate.push_back(eid);
      }

      // 剩余缺失专家走 deferred
      for (int i = need; i < static_cast<int>(missing.size()); i++) {
        int eid = missing[i];
        auto ptrs = get_dst_ptrs(eid);
        scheduler.submit_decode_deferred(layer_idx, eid, tp_part_idx,
                                         ptrs[0], ptrs[1], ptrs[2]);
        result.deferred.push_back(eid);
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
   * @param slot_pool 该层 slot 池
   * @param scorer 驱逐评分器
   * @param layer_idx 当前层
   * @param new_expert_id 要加载的新专家
   * @return int 被驱逐释放的 slot_idx，-1 表示无需驱逐或无法驱逐
   */
  int evict_for_new_expert(MeshSlotPool& slot_pool,
                           const EvictionScorer& scorer,
                           int layer_idx, int new_expert_id) {
    // 前五层满配检查
    if (is_front_layer(layer_idx)) {
      int cap = get_effective_cap(layer_idx, slot_pool.cap());
      if (cap >= expert_num_) {
        return -1;  // 满配，无需驱逐
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
