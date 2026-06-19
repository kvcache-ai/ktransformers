/**
 * @file mesh_handoff.hpp
 * @brief Prefill→Decode 过渡阶段
 *
 * 触发时机：第一个 decode token 的第一层前
 *
 * 流程：
 * 1. 保留 slot 池（prefill 留下的种子专家给 decode 继承）
 * 2. 释放两个 prefill temporal 缓冲区
 * 3. 搬运 GPU 专家权重到 slot 池填补因裁剪留下的槽位
 *    - GPU expert 从 CPU slot 搬到 GPU（不读盘，纯内存→GPU 拷贝）
 *    - 释放的 GE 个 slot 用 io_uring 读 (cap, cap+GE] 区间的专家填充
 * 4. 清空优先队列
 * 5. 切换到 decode 模式
 */
#pragma once

#include <functional>
#include <stdexcept>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_eviction.hpp"
#include "mesh_io_uring.hpp"
#include "mesh_prefill.hpp"
#include "mesh_scheduler.hpp"
#include "mesh_slot_pool.hpp"

namespace mesh {

/**
 * @brief Prefill→Decode 过渡处理器
 */
class MeshHandoff {
 public:
  /**
   * @brief 执行过渡
   *
   * @param config MESH 配置
   * @param prefill prefill 策略对象
   * @param scheduler 调度器
   * @param io io_uring 读取器
   * @param scorer 驱逐评分器
   * @param pools [layer][tp] 的 slot 池
   * @param layouts [tp][expert] 的文件布局
   * @param move_gpu_expert_to_gpu 搬运 GPU 专家到 GPU 的回调
   * @param get_dst_ptrs 获取 slot buffer 指针的回调
   */
  void transition(const MeshConfig& config,
                  MeshPrefill& prefill,
                  MeshScheduler& scheduler,
                  MeshIoUring& io,
                  EvictionScorer& scorer,
                  std::vector<std::vector<MeshSlotPool>>& pools,
                  const std::vector<std::vector<ExpertFileLayout>>& layouts,
                  std::function<void(int, int)> move_gpu_expert_to_gpu,
                  std::function<std::vector<void*>(int, int, int)> get_dst_ptrs) {
    int ge = config.num_gpu_experts;
    int cap = config.cap;

    // 1. 保留 slot 池（无需操作，prefill 留下的种子专家直接继承）

    // 2. 释放两个 prefill temporal 缓冲区
    prefill.release_temporal();

    // 3. 搬运 GPU 专家权重到 GPU，释放的 slot 用 io_uring 读新专家填充
    //    GPU expert 在 slot 前茅（GE 个），搬到 GPU 后释放这些 slot
    //    然后读 (cap, cap+GE] 区间的专家填充这些 slot
    if (ge > 0) {
      move_gpu_experts_and_refill(config, pools, layouts, scorer, io,
                                  move_gpu_expert_to_gpu, get_dst_ptrs);
    }

    // 4. 清空优先队列
    scheduler.clear_queue();

    // 5. 切换到 decode 模式
    scheduler.switch_to_decode();
  }

 private:
  /**
   * @brief 搬运 GPU 专家到 GPU，并用 io_uring 读新专家填充释放的 slot
   *
   * GPU expert 在 slot 前茅（GE 个位置）：
   * - 标记这些 slot 的 active_readers（防止驱逐）
   * - 搬运到 GPU（不读盘）
   * - 释放这些 slot
   * - 用 io_uring 读 (cap, cap+GE] 区间的专家填充
   */
  void move_gpu_experts_and_refill(
      const MeshConfig& config,
      std::vector<std::vector<MeshSlotPool>>& pools,
      const std::vector<std::vector<ExpertFileLayout>>& layouts,
      EvictionScorer& scorer,
      MeshIoUring& io,
      std::function<void(int, int)> move_gpu_expert_to_gpu,
      std::function<std::vector<void*>(int, int, int)> get_dst_ptrs) {
    int ge = config.num_gpu_experts;
    int cap = config.cap;
    int num_layers = config.total_layers;
    int tp_count = config.tp_count;

    // (cap, cap+GE] 区间的专家 ID
    std::vector<int> refill_experts;
    for (int e = cap; e < cap + ge && e < config.expert_num; e++) {
      refill_experts.push_back(e);
    }

    for (int layer = 0; layer < num_layers; layer++) {
      for (int tp = 0; tp < tp_count; tp++) {
        MeshSlotPool& pool = pools[layer][tp];

        // GPU expert 在 slot 前茅（GE 个）
        for (int slot_idx = 0; slot_idx < ge && slot_idx < pool.cap(); slot_idx++) {
          int expert_id = slot_idx;  // 假设 GPU expert 占据前 GE 个 slot
          if (expert_id >= config.expert_num) break;

          // 标记 active_readers 防止驱逐
          pool.acquire_reader(expert_id);

          // 搬运到 GPU（不读盘，纯内存→GPU 拷贝）
          move_gpu_expert_to_gpu(layer, expert_id);

          // 释放 reader
          pool.release_reader(expert_id);

          // 释放这个 slot，用 io_uring 读新专家填充
          int new_expert_id = cap + slot_idx;
          if (new_expert_id < config.expert_num &&
              slot_idx < static_cast<int>(refill_experts.size())) {
            new_expert_id = refill_experts[slot_idx];
            auto ptrs = get_dst_ptrs(layer, tp, new_expert_id);
            // 提交 io_uring 读
            const auto& layout = layouts[tp][new_expert_id];
            io.submit_load(new_expert_id, tp, layout,
                           ptrs[0], ptrs[1], ptrs[2],
                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                           ReadPriority::Demand);
            // 覆盖 slot
            pool.overwrite(slot_idx, new_expert_id);
          }
        }
      }
    }

    // 等待所有 io_uring 读完成
    io.submit_and_wait();
  }
};

}  // namespace mesh
