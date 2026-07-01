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

#include <cstdio>
#include <cstring>
#include <functional>
#include <numa.h>
#include <stdexcept>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_io_uring.hpp"
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
              size_t slot_bytes, const std::vector<int>& numa_nodes)
      : num_layers_(num_layers),
        expert_num_(expert_num),
        cap_(cap),
        tp_count_(tp_count),
        slot_bytes_(slot_bytes),
        numa_nodes_(numa_nodes) {
    // temporal 长度 = 单层专家总数 - slot cap
    temporal_size_ = (expert_num > cap) ? (expert_num - cap) : 0;
  }

  ~MeshPrefill() { release_temporal(); }

  // 初始化 temporal 双缓冲（每 TP 一对 A/B，按 TP 对应 NUMA 节点分配）
  void init_temporal() {
    if (!temporal_a_vec_.empty()) return;  // 已初始化
    // 辅因3: 每 TP 独立分配，单 TP 字节数 = temporal_size * slot_bytes
    size_t per_tp_bytes = static_cast<size_t>(temporal_size_) * slot_bytes_;
    role_a_ = TemporalRole::COMPUTE;
    role_b_ = TemporalRole::PREFETCH;
    // B4 fix: 初始化 expert_id -> temporal_offset 映射
    // temporal 中装的是 cap ~ expert_num-1 的专家，offset = expert_id - cap
    temporal_expert_map_.assign(tp_count_, std::vector<int>(expert_num_, -1));
    for (int tp = 0; tp < tp_count_; tp++) {
      for (int e = cap_; e < expert_num_; e++) {
        temporal_expert_map_[tp][e] = e - cap_;
      }
    }
    if (per_tp_bytes == 0) {
      // cap >= expert_num: 所有专家常驻 slot，不需要 temporal buffer
      return;
    }
    temporal_a_vec_.assign(tp_count_, nullptr);
    temporal_b_vec_.assign(tp_count_, nullptr);
    for (int tp = 0; tp < tp_count_; tp++) {
      int numa_node = numa_nodes_[tp % numa_nodes_.size()];  // 辅因3: 按 TP 选 NUMA
      temporal_a_vec_[tp] = numa_alloc_onnode(per_tp_bytes, numa_node);
      temporal_b_vec_[tp] = numa_alloc_onnode(per_tp_bytes, numa_node);
      if (!temporal_a_vec_[tp] || !temporal_b_vec_[tp]) {
        throw std::runtime_error("MeshPrefill: numa_alloc_onnode for temporal failed");
      }
    }
    temporal_bytes_ = per_tp_bytes;
  }

  // 释放 temporal 内存（仅 prefill→decode 切换时调用）
  void release_temporal() {
    for (void* p : temporal_a_vec_) {
      if (p) numa_free(p, temporal_bytes_);
    }
    for (void* p : temporal_b_vec_) {
      if (p) numa_free(p, temporal_bytes_);
    }
    temporal_a_vec_.clear();
    temporal_b_vec_.clear();
  }

  // ===== Prefill 主流程 =====

  /**
   * @brief 处理某一层的所有 chunk
   *
   * B3 fix: wait_temporal_ready 接入 io_uring 等待
   * B4 fix: submit_next_layer_prefetch 用 temporal_ptr 正确寻址
   *
   * @param layer_idx 当前层 L
   * @param total_chunks 总 chunk 数
   * @param active_experts_per_chunk [chunk] -> 该 chunk 活跃的专家列表
   * @param slot_pool 该层的 slot 池
   * @param scheduler 调度器
   * @param io io_uring 读取器（B3: 用于等待 CQE）
   * @param amx_forward AMX 计算回调（layer_idx, chunk_idx, expert_ids）
   */
  void run_layer(int layer_idx, int total_chunks,
                 const std::vector<std::vector<int>>& active_experts_per_chunk,
                 MeshSlotPool& slot_pool, MeshScheduler& scheduler,
                 MeshIoUring* io,
                 std::function<void(int, int, const std::vector<int>&)> amx_forward) {
    // 辅因3: temporal buffer 按 TP 分片，compute/prefetch 通过 tp 索引访问
    for (int c = 0; c < total_chunks; c++) {
      const auto& active = active_experts_per_chunk[c];

      // a. 等待 temporal_A 内第 L 层所需专家全部就绪
      //    （由上一层的异步预取完成；第一层阻塞等待）
      wait_temporal_ready(layer_idx, active, slot_pool, io);

      // b. 对 temporal_B 发起第 L+1 层的异步预取（覆盖另一个 temporal）
      if (layer_idx + 1 < num_layers_) {
        submit_next_layer_prefetch(layer_idx + 1, active, scheduler);
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

  // 获取当前 COMPUTE 角色的 temporal buffer（辅因3: 按 TP 索引）
  void* compute_temporal(int tp) const {
    if (tp < 0 || tp >= tp_count_) return nullptr;
    const auto& vec = (role_a_ == TemporalRole::COMPUTE) ? temporal_a_vec_ : temporal_b_vec_;
    if (tp >= (int)vec.size()) return nullptr;
    return vec[tp];
  }

  // 获取当前 PREFETCH 角色的 temporal buffer（辅因3: 按 TP 索引）
  void* prefetch_temporal(int tp) const {
    if (tp < 0 || tp >= tp_count_) return nullptr;
    const auto& vec = (role_a_ == TemporalRole::PREFETCH) ? temporal_a_vec_ : temporal_b_vec_;
    if (tp >= (int)vec.size()) return nullptr;
    return vec[tp];
  }

  /**
   * @brief B4 fix: 获取 temporal buffer 中某专家某矩阵的指针
   *
   * @param buf 某 TP 的 temporal buffer 基地址（compute_temporal(tp) 或 prefetch_temporal(tp)）
   * @param expert_id 专家 ID
   * @param matrix_idx 0=gate, 1=up, 2=down
   * @return void* 指针，未映射返回 nullptr
   *
   * 辅因3: buf 已是 per-TP 指针，无需再按 tp 偏移
   */
  void* temporal_ptr(void* buf, int expert_id, int matrix_idx) const {
    if (expert_id < 0 || expert_id >= expert_num_) return nullptr;
    // expert_id -> offset 映射对所有 TP 相同（cap ~ expert_num-1）
    int offset = (temporal_expert_map_.empty() || temporal_expert_map_[0].empty())
                 ? -1 : temporal_expert_map_[0][expert_id];
    if (offset < 0) return nullptr;
    // 布局：[offset] 每个 slot_bytes_，内含 gate + up + down
    char* base = static_cast<char*>(buf) +
                 static_cast<size_t>(offset) * slot_bytes_;
    // gate 在偏移 0，up 在偏移 gate_up_bytes_，down 在偏移 gate_up_bytes_*2
    size_t matrix_offset = (matrix_idx == 0) ? 0 :
                           (matrix_idx == 1) ? gate_up_bytes_ : gate_up_bytes_ * 2;
    return base + matrix_offset;
  }

  // B4 fix: 设置 gate/up 单块字节数（用于 temporal_ptr 偏移计算）
  void set_gate_up_bytes(size_t bytes) { gate_up_bytes_ = bytes; }

  // ===== 访问器 =====
  int temporal_size() const { return temporal_size_; }
  bool temporal_ready() const { return !temporal_a_vec_.empty(); }

 private:
  int num_layers_;
  int expert_num_;
  int cap_;
  int tp_count_;
  size_t slot_bytes_;
  size_t gate_up_bytes_ = 0;  // B4: gate/up 单块字节数
  std::vector<int> numa_nodes_;  // 辅因3: 每 TP 对应的 NUMA 节点
  size_t temporal_bytes_ = 0;    // 辅因3: 单 TP temporal 字节数

  // temporal 双缓冲（辅因3: 每 TP 独立分配，按 NUMA 节点定位）
  std::vector<void*> temporal_a_vec_;  // size == tp_count_
  std::vector<void*> temporal_b_vec_;  // size == tp_count_
  TemporalRole role_a_ = TemporalRole::COMPUTE;
  TemporalRole role_b_ = TemporalRole::PREFETCH;
  int temporal_size_ = 0;  // = expert_num - cap

  // B4 fix: expert_id -> temporal_offset 映射 [tp][expert_id]
  std::vector<std::vector<int>> temporal_expert_map_;

  // 等待 temporal 内专家就绪
  // B3 fix: 调用 io_uring process_cqes 推进 CQE，直到所有活跃专家 CACHED
  void wait_temporal_ready(int layer_idx, const std::vector<int>& active,
                           MeshSlotPool& slot_pool, MeshIoUring* io) {
    // Bug 10 fix: 加超时保护，避免 expert 未被调度时死循环
    const int kMaxWaitIters = 1000000;  // ~几秒
    int wait_iters = 0;
    for (int eid : active) {
      if (slot_pool.is_cached(eid)) continue;
      // 阻塞等待 io_uring CQE 到达
      // Bug 10 fix: 只用 process_cqes，不用 submit_and_wait
      // submit_and_wait 内部也处理 CQE 并 delete pending，
      // 与 process_cqes 同时调用会导致 double free
      if (io) {
        while (!slot_pool.is_cached(eid)) {
          io->process_cqes();
          if (++wait_iters > kMaxWaitIters) {
            fprintf(stderr,
                    "[MESH] wait_temporal_ready: expert %d never cached at layer %d, "
                    "possible scheduler deadlock\n",
                    eid, layer_idx);
            break;  // 避免死循环，返回后 AMX 会读到未就绪数据（nullptr 检查）
          }
        }
      }
    }
  }

  // 提交下一层的异步预取
  // B4 fix: 用 temporal_ptr 正确寻址，不再用 ptrs[eid*3] 越界
  // 辅因3: prefetch_buf 按 TP 从 prefetch_temporal(tp) 获取
  void submit_next_layer_prefetch(int next_layer,
                                  const std::vector<int>& active,
                                  MeshScheduler& scheduler) {
    for (int tp = 0; tp < tp_count_; tp++) {
      void* prefetch_buf = prefetch_temporal(tp);
      if (!prefetch_buf) continue;
      for (int eid : active) {
        // 只预取不在 slot 中的专家（在 temporal 中的）
        // expert_id -> offset 映射对所有 TP 相同
        int offset = (!temporal_expert_map_.empty() &&
                      tp < (int)temporal_expert_map_.size() &&
                      eid < (int)temporal_expert_map_[tp].size())
                     ? temporal_expert_map_[tp][eid] : -1;
        if (offset < 0) continue;
        void* gate = temporal_ptr(prefetch_buf, eid, 0);
        void* up = temporal_ptr(prefetch_buf, eid, 1);
        void* down = temporal_ptr(prefetch_buf, eid, 2);
        if (!gate || !up || !down) continue;
        scheduler.submit_prefill(next_layer, eid, tp, gate, up, down,
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
