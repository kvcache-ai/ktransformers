/**
 * @file mesh_prefill.hpp
 * @brief Prefill 策略：temporal 双缓冲 + 双 embedding 空间 + layer-major 调度
 *
 * 流程（layer-major）：
 *   for layer L in 0..N:
 *     for chunk C in 0..M:
 *       a. wait temporal_A[L] 就绪
 *       b. submit temporal_B[L+1] 异步预取 (schedule_key = L)
 *       c. AMX 计算 (temporal_A + slot 常驻)
 *       d. 指针 swap (slot 低频 ↔ temporal_A 高频)
 *       e. 保存 chunk 产物到 embedding 缓冲
 *       f. 下一个 chunk → 回到 c
 *     下一层
 *
 * 双 embedding 空间：奇偶层交替写入 buf_a / buf_b
 * temporal 双缓冲：A 用于当前层计算，B 用于下一层预取，每层角色互换
 *
 * temporal 长度 = 单层专家总数 - slot cap
 * 只在 prefill→decode 切换时清除 temporal 内存，其余全是覆盖
 */
#pragma once

#include <cstring>
#include <functional>
#include <numa.h>
#include <stdexcept>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_scheduler.hpp"
#include "mesh_slot_pool.hpp"

namespace mesh {

/**
 * @brief Prefill 策略
 *
 * MESH 模式下接管 prefill 调度（layer-major + 双 embedding 空间）。
 * KT 原版只加 if(mesh_enabled) { mesh_prefill.run(); return; } 一个分支。
 */
class MeshPrefill {
 public:
  enum class TemporalRole { COMPUTE, PREFETCH };

  MeshPrefill(int num_layers, int expert_num, int cap, int tp_count,
              size_t slot_bytes, int numa_node)
      : num_layers_(num_layers),
        expert_num_(expert_num),
        cap_(cap),
        tp_count_(tp_count),
        slot_bytes_(slot_bytes),
        numa_node_(numa_node) {
    // temporal 长度 = 单层专家总数 - slot cap
    temporal_size_ = (expert_num > cap) ? (expert_num - cap) : 0;
  }

  ~MeshPrefill() { release_temporal(); }

  // 初始化 temporal 双缓冲（每 TP 一对 A/B）
  void init_temporal() {
    if (temporal_a_ != nullptr) return;  // 已初始化
    size_t bytes = static_cast<size_t>(temporal_size_) * slot_bytes_ * tp_count_;
    role_a_ = TemporalRole::COMPUTE;
    role_b_ = TemporalRole::PREFETCH;
    if (bytes == 0) {
      // cap >= expert_num: 所有专家常驻 slot，不需要 temporal buffer
      return;
    }
    temporal_a_ = numa_alloc_onnode(bytes, numa_node_);
    temporal_b_ = numa_alloc_onnode(bytes, numa_node_);
    if (!temporal_a_ || !temporal_b_) {
      throw std::runtime_error("MeshPrefill: numa_alloc_onnode for temporal failed");
    }
    temporal_bytes_ = bytes;
  }

  // 释放 temporal 内存（仅 prefill→decode 切换时调用）
  void release_temporal() {
    if (temporal_a_) {
      numa_free(temporal_a_, temporal_bytes_);
      temporal_a_ = nullptr;
    }
    if (temporal_b_) {
      numa_free(temporal_b_, temporal_bytes_);
      temporal_b_ = nullptr;
    }
  }

  // ===== Prefill 主流程 =====

  /**
   * @brief 处理某一层的所有 chunk
   *
   * @param layer_idx 当前层 L
   * @param total_chunks 总 chunk 数
   * @param active_experts_per_chunk [chunk] -> 该 chunk 活跃的专家列表
   * @param slot_pool 该层的 slot 池
   * @param scheduler 调度器
   * @param amx_forward AMX 计算回调（layer_idx, chunk_idx, expert_ids）
   * @param get_temporal_ptrs 获取 temporal buffer 指针的回调
   */
  void run_layer(int layer_idx, int total_chunks,
                 const std::vector<std::vector<int>>& active_experts_per_chunk,
                 MeshSlotPool& slot_pool, MeshScheduler& scheduler,
                 std::function<void(int, int, const std::vector<int>&)> amx_forward,
                 std::function<std::vector<void*>(int, int)> get_temporal_ptrs) {
    for (int c = 0; c < total_chunks; c++) {
      const auto& active = active_experts_per_chunk[c];

      // a. 等待 temporal_A 内第 L 层所需专家全部就绪
      //    （由上一层的异步预取完成；第一层阻塞等待）
      wait_temporal_ready(layer_idx, active, slot_pool);

      // b. 对 temporal_B 发起第 L+1 层的异步预取（覆盖另一个 temporal）
      if (layer_idx + 1 < num_layers_) {
        submit_next_layer_prefetch(layer_idx + 1, active, scheduler,
                                   get_temporal_ptrs);
      }

      // c. 本层计算使用 temporal_A 中的临时专家 + slot 池常驻专家
      //    具体计算逻辑交给 ktransformers
      amx_forward(layer_idx, c, active);

      // d. 计算完成后，根据本层专家调用频次，指针 swap
      //    slot 池内低频专家 ↔ temporal_A 中高频专家
      swap_high_freq_to_slot(layer_idx, active, slot_pool);

      // e. 保存当前 chunk 临时产物到 embedding 缓冲
      //    （由 KT 侧通过双 embedding 空间管理，MESH 只感知奇偶层）
      save_chunk_output(layer_idx, c);

      // f. 推进窗口，计算下一个 chunk → 回到 c
    }
    // 本层所有 chunk 计算完毕，来到下一层
  }

  // ===== temporal 角色切换 =====

  // 每过一层，A/B 角色互换
  void swap_temporal_roles() {
    TemporalRole tmp = role_a_;
    role_a_ = role_b_;
    role_b_ = tmp;
  }

  // 获取当前 COMPUTE 角色的 temporal buffer
  void* compute_temporal() const {
    return role_a_ == TemporalRole::COMPUTE ? temporal_a_ : temporal_b_;
  }

  // 获取当前 PREFETCH 角色的 temporal buffer
  void* prefetch_temporal() const {
    return role_a_ == TemporalRole::PREFETCH ? temporal_a_ : temporal_b_;
  }

  // ===== 访问器 =====
  int temporal_size() const { return temporal_size_; }
  bool temporal_ready() const { return temporal_a_ != nullptr; }

 private:
  int num_layers_;
  int expert_num_;
  int cap_;
  int tp_count_;
  size_t slot_bytes_;
  int numa_node_;
  size_t temporal_bytes_ = 0;

  // temporal 双缓冲
  void* temporal_a_ = nullptr;
  void* temporal_b_ = nullptr;
  TemporalRole role_a_ = TemporalRole::COMPUTE;
  TemporalRole role_b_ = TemporalRole::PREFETCH;
  int temporal_size_ = 0;  // = expert_num - cap

  // 等待 temporal 内专家就绪
  void wait_temporal_ready(int layer_idx, const std::vector<int>& active,
                           MeshSlotPool& slot_pool) {
    // 对于第一层，阻塞直到全部读取完毕
    // 对于后续层，由上一层预取完成，这里检查状态
    for (int eid : active) {
      if (slot_pool.is_cached(eid)) continue;
      // 等待 io_uring CQE
      // 实际实现需要与 MeshIoUring 配合
    }
  }

  // 提交下一层的异步预取
  void submit_next_layer_prefetch(int next_layer,
                                  const std::vector<int>& active,
                                  MeshScheduler& scheduler,
                                  std::function<std::vector<void*>(int, int)> get_ptrs) {
    // 对下一层需要的专家提交异步读，schedule_key = next_layer
    void* prefetch_buf = prefetch_temporal();
    for (int tp = 0; tp < tp_count_; tp++) {
      auto ptrs = get_ptrs(next_layer, tp);
      for (int eid : active) {
        scheduler.submit_prefill(next_layer, eid, tp,
                                  ptrs[eid * 3], ptrs[eid * 3 + 1], ptrs[eid * 3 + 2],
                                  ReadPriority::Prefetch);
      }
    }
  }

  // 指针 swap：slot 低频 ↔ temporal 高频
  void swap_high_freq_to_slot(int layer_idx, const std::vector<int>& active,
                              MeshSlotPool& slot_pool) {
    // 统计本层专家调用频次
    std::vector<int> freq(expert_num_, 0);
    for (int eid : active) {
      if (eid >= 0 && eid < expert_num_) freq[eid]++;
    }

    // 找出 temporal 中高频但不在 slot 中的专家
    std::vector<int> high_freq_in_temporal;
    for (int e = 0; e < expert_num_; e++) {
      if (freq[e] > 0 && !slot_pool.is_cached(e)) {
        high_freq_in_temporal.push_back(e);
      }
    }

    // 找出 slot 中低频的专家
    auto cached = slot_pool.cached_experts();
    std::vector<int> low_freq_in_slot;
    for (int eid : cached) {
      if (eid >= 0 && eid < expert_num_ && freq[eid] == 0) {
        low_freq_in_slot.push_back(eid);
      }
    }

    // 指针 swap：把池子指针指向 temporal 目标专家，temporal 目标专家指针指向原 slot 专家
    // 不操作内存本身，只交换指针映射
    size_t swap_count = std::min(high_freq_in_temporal.size(), low_freq_in_slot.size());
    for (size_t i = 0; i < swap_count; i++) {
      // 实际实现需要通过 slot_pool 的接口完成指针交换
      // slot_pool.swap_pointers(low_freq_in_slot[i], high_freq_in_temporal[i]);
    }
  }

  // 保存 chunk 产物到 embedding 缓冲
  // 双 embedding 空间：奇偶层交替写入
  void save_chunk_output(int layer_idx, int chunk_idx) {
    // 由 KT 侧管理 embedding 空间，MESH 只感知奇偶层
    // embedding_buf_a_ / embedding_buf_b_ 由 KT 持有
  }
};

}  // namespace mesh
