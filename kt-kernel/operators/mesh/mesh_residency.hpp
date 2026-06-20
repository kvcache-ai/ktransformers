/**
 * @file mesh_residency.hpp
 * @brief MESH 顶层 Manager
 *
 * 协调 slot 池、io_uring、调度器、驱逐评分、prefill/decode 策略、过渡处理。
 * 由 ext_bindings.cpp 暴露给 Python，是 MESH 对外的唯一入口。
 *
 * 生命周期：
 * 1. init()：初始化所有组件
 * 2. bootstrap()：启动阶段，读前 cap 个专家进 slot
 * 3. prefill 阶段：on_prefill_layer_start/done
 * 4. 过渡阶段：on_prefill_to_decode()
 * 5. decode 阶段：on_decode_token / on_decode_layer
 */
#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mesh_config.hpp"
#include "mesh_decode.hpp"
#include "mesh_eviction.hpp"
#include "mesh_handoff.hpp"
#include "mesh_io_uring.hpp"
#include "mesh_prefill.hpp"
#include "mesh_scheduler.hpp"
#include "mesh_slot_pool.hpp"

namespace mesh {

/**
 * @brief MESH 顶层 Manager
 */
class MeshResidencyManager {
 public:
  MeshResidencyManager() = default;
  ~MeshResidencyManager() = default;

  // 禁止拷贝
  MeshResidencyManager(const MeshResidencyManager&) = delete;
  MeshResidencyManager& operator=(const MeshResidencyManager&) = delete;

  /**
   * @brief 初始化 MESH
   *
   * @param config 配置
   * @param numa_nodes 每个 TP 分片对应的 NUMA 节点
   */
  void init(const MeshConfig& config, const std::vector<int>& numa_nodes) {
    config_ = config;
    numa_nodes_ = numa_nodes;

    // 计算 slot 字节数（根据权重类型和模型维度）
    slot_bytes_ = compute_slot_bytes(config);
    size_t gate_up_bytes = compute_gate_up_bytes(config);
    size_t down_bytes = compute_down_bytes(config);

    // 诊断日志：确认 config 值和内存分配量
    size_t total_slot_bytes = static_cast<size_t>(config.total_layers) *
                              config.tp_count * config.cap * slot_bytes_;
    fprintf(stderr,
            "[MESH init DIAG] hidden=%d, inter=%d, expert_num=%d, tp=%d, "
            "cap=%d, layers=%d, weight_type=%d\n",
            config.hidden_size, config.intermediate_size,
            config.expert_num, config.tp_count,
            config.cap, config.total_layers,
            static_cast<int>(config.weight_type));
    fprintf(stderr,
            "[MESH init DIAG] gate_up_bytes=%zu, down_bytes=%zu, slot_bytes=%zu, "
            "total_slot_pool=%zu MB\n",
            gate_up_bytes, down_bytes, slot_bytes_,
            total_slot_bytes / (1024 * 1024));

    // 创建 slot 池 [layer][tp]
    pools_.resize(config.total_layers);
    for (int l = 0; l < config.total_layers; l++) {
      pools_[l].reserve(config.tp_count);
      for (int tp = 0; tp < config.tp_count; tp++) {
        pools_[l].emplace_back(l, tp, numa_nodes[tp], config.cap, slot_bytes_);
        pools_[l][tp].set_gate_up_bytes(compute_gate_up_bytes(config));
        pools_[l][tp].init_expert_map(config.expert_num);
      }
    }

    // 初始化各组件
    io_ = std::make_unique<MeshIoUring>();
    scheduler_ = std::make_unique<MeshScheduler>(config.total_layers);
    scorer_ = std::make_unique<EvictionScorer>(config.total_layers, config.expert_num, config);
    prefill_ = std::make_unique<MeshPrefill>(config.total_layers, config.expert_num,
                                             config.cap, config.tp_count,
                                             slot_bytes_, numa_nodes[0]);
    decode_ = std::make_unique<MeshDecode>(config);
    handoff_ = std::make_unique<MeshHandoff>();

    // 初始化 temporal 双缓冲
    prefill_->init_temporal();

    fprintf(stderr, "[MESH init DIAG] init complete, pools_ size=%zu\n",
            pools_.size());
  }

  // ===== 文件布局注入 =====

  // A4 fix: layouts_ 改为 [layer][tp][expert] 3D，
  // 因为每层的专家权重在 safetensors 文件中的偏移不同。
  void set_file_layout(int layer_idx, int tp_part_idx, int expert_id,
                       const ExpertFileLayout& layout) {
    if (layouts_.empty()) {
      layouts_.resize(config_.total_layers,
                      std::vector<std::vector<ExpertFileLayout>>(
                          config_.tp_count,
                          std::vector<ExpertFileLayout>(config_.expert_num)));
    }
    if (layer_idx >= 0 && layer_idx < (int)layouts_.size() &&
        tp_part_idx >= 0 && tp_part_idx < (int)layouts_[layer_idx].size() &&
        expert_id >= 0 && expert_id < (int)layouts_[layer_idx][tp_part_idx].size()) {
      layouts_[layer_idx][tp_part_idx][expert_id] = layout;
    }
  }

  // ===== GPU expert mask 注入 =====

  void set_gpu_experts_mask(const uint8_t* mask, int n) {
    gpu_experts_mask_.assign(mask, mask + n);
  }

  bool is_gpu_expert(int expert_id) const {
    if (expert_id < 0 || expert_id >= (int)gpu_experts_mask_.size()) return false;
    return gpu_experts_mask_[expert_id] != 0;
  }

  // ===== 启动阶段 =====

  /**
   * @brief 启动阶段：读前 cap 个专家进 slot
   *
   * 每层 Slot 池按编号顺序填满。
   */
  void bootstrap() {
    // 文件布局未注入时跳过实际权重加载（仅用于内存测试模式）
    if (layouts_.empty()) {
      fprintf(stderr, "[MESH] bootstrap: layouts_ empty, skipping weight preload "
              "(memory-test mode, slot pools already allocated)\n");
      return;
    }

    // 预加载 Scale Cache（AMXINT4 专用）— 每层独立预加载
    if (config_.weight_type == WeightType::AMXINT4) {
      for (int l = 0; l < config_.total_layers; l++) {
        io_->preload_scale_cache(config_.expert_num, config_.tp_count,
                                 numa_nodes_[0], layouts_[l], l);
      }
    }

    // 每层每 TP 读前 cap 个专家
    for (int l = 0; l < config_.total_layers; l++) {
      for (int tp = 0; tp < config_.tp_count; tp++) {
        MeshSlotPool& pool = pools_[l][tp];
        for (int e = 0; e < config_.cap && e < config_.expert_num; e++) {
          if (is_gpu_expert(e)) continue;  // GPU expert 跳过

          const auto& layout = layouts_[l][tp][e];
          void* gate_dst = pool.gate_ptr(e);
          void* up_dst = pool.up_ptr(e);
          void* down_dst = pool.down_ptr(e);

          // 绑定 slot
          pool.bind(e, e);

          // 提交 io_uring 读
          // A6: 传入 layer_idx 和 on_complete 回调，完成后 mark_cached
          io_->submit_load(e, tp, layout, gate_dst, up_dst, down_dst,
                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                           ReadPriority::Demand,
                           /*layer_idx=*/l,
                           /*on_complete=*/[this, l, tp, e](int, int, int) {
                             pools_[l][tp].mark_cached(e);
                           });
        }
      }
    }

    // 等待所有读完成
    io_->submit_and_wait();

    // 标记所有 slot 为 CACHED
    for (int l = 0; l < config_.total_layers; l++) {
      for (int tp = 0; tp < config_.tp_count; tp++) {
        MeshSlotPool& pool = pools_[l][tp];
        for (int e = 0; e < config_.cap && e < config_.expert_num; e++) {
          if (is_gpu_expert(e)) continue;
          pool.mark_cached(e);
        }
      }
    }
  }

  // ===== 权重指针查询（KT 计算用）=====

  void* get_gate_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].expert_gate_ptr(expert_id);
  }

  void* get_up_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].expert_up_ptr(expert_id);
  }

  void* get_down_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].expert_down_ptr(expert_id);
  }

  // ===== Prefill 阶段回调 =====

  void on_prefill_layer_start(int layer_idx, int qlen,
                              const std::vector<int>& active_experts) {
    // 由 prefill 策略处理
  }

  void on_prefill_layer_done(int layer_idx) {
    // 每过一层，temporal 角色互换
    prefill_->swap_temporal_roles();
  }

  // ===== 过渡阶段 =====

  void on_prefill_to_decode() {
    handoff_->transition(config_, *prefill_, *scheduler_, *io_, *scorer_,
                         pools_, layouts_,
                         /*move_gpu_expert_to_gpu=*/[](int, int) {},
                         /*get_dst_ptrs=*/[this](int l, int tp, int e) {
                           // A5 fix: 用 expert_*_ptr 按 expert_id 查 slot，
                           // 不能把 expert_id 当 slot_idx 传给 gate_ptr（eid>=cap 时越界）
                           return std::vector<void*>{
                               pools_[l][tp].expert_gate_ptr(e),
                               pools_[l][tp].expert_up_ptr(e),
                               pools_[l][tp].expert_down_ptr(e)};
                         });
  }

  // ===== Decode 阶段回调 =====

  void on_decode_token_start() {
    scheduler_->inc_timeline_step();
  }

  /**
   * @brief Decode 每层处理
   *
   * @param layer_idx 当前层
   * @param topk 当前 token 的 top-k 专家
   * @param scores 各专家 router score（全长）
   * @param tp_part_idx TP 分片
   * @return MeshDecode::SplitResult immediate/deferred 分组
   */
  MeshDecode::SplitResult on_decode_layer(int layer_idx,
                                          const std::vector<int>& topk,
                                          const std::vector<float>& scores,
                                          int tp_part_idx) {
    // A7: 先处理已完成的 io_uring CQE，触发 mark_cached 回调
    io_->process_cqes();

    MeshSlotPool& pool = pools_[layer_idx][tp_part_idx];
    auto result = decode_->split(topk, scores, pool, *scheduler_, layer_idx, tp_part_idx,
                          [this, &pool, layer_idx, tp_part_idx](int eid) {
                            // A5 fix: 用 expert_*_ptr 按 expert_id 查 slot，
                            // 不能把 expert_id 当 slot_idx 传给 gate_ptr（eid>=cap 时越界）
                            return std::vector<void*>{
                                pool.expert_gate_ptr(eid),
                                pool.expert_up_ptr(eid),
                                pool.expert_down_ptr(eid)};
                          });

    // A7: 排空调度器队列，提交 io_uring 读取
    drain_and_submit();

    return result;
  }

  /**
   * @brief A7: 排空调度器队列并提交 io_uring 读取
   *
   * 把 MeshScheduler 优先队列中的 ScheduledRequest 取出，
   * 为每个请求找到对应的 ExpertFileLayout 并提交给 MeshIoUring。
   * 完成后触发 mark_cached + 原始 on_complete 回调。
   */
  void drain_and_submit() {
    auto requests = scheduler_->drain_all();
    for (auto& req : requests) {
      // 边界检查
      if (req.layer_idx < 0 || req.layer_idx >= (int)layouts_.size()) continue;
      if (req.tp_part_idx < 0 || req.tp_part_idx >= (int)layouts_[req.layer_idx].size()) continue;
      if (req.expert_id < 0 || req.expert_id >= (int)layouts_[req.layer_idx][req.tp_part_idx].size()) continue;
      // layouts_ 为空（memory-test 模式）时跳过
      if (layouts_.empty()) continue;

      const auto& layout = layouts_[req.layer_idx][req.tp_part_idx][req.expert_id];

      // 捕获原始 on_complete 回调
      auto orig_on_complete = std::move(req.on_complete);
      int layer = req.layer_idx;
      int tp = req.tp_part_idx;
      int expert = req.expert_id;

      // 绑定 slot（如果尚未绑定）
      MeshSlotPool& pool = pools_[layer][tp];
      if (!pool.is_cached(expert)) {
        // 需要先驱逐一个 slot（如果池满）
        // B1: 在线驱逐尚未完全实现，这里先尝试 bind
        // 如果池未满，bind 到空闲 slot
        // 如果池满，需要驱逐（B1 修复后完善）
      }

      io_->submit_load(
          req.expert_id, req.tp_part_idx, layout,
          req.gate_dst, req.up_dst, req.down_dst,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // scale 从 cache memcpy
          req.priority,
          /*layer_idx=*/layer,
          /*on_complete=*/[this, layer, tp, expert, orig_on_complete](int, int, int) {
            pools_[layer][tp].mark_cached(expert);
            if (orig_on_complete) orig_on_complete();
          });
    }
  }

  /**
   * @brief 单 token 结束后批量更新 Heat 和 Markov
   *
   * @param all_layers_topk [layer] -> top-k expert ids
   * @param all_layers_scores [layer] -> normalized router scores (全长)
   */
  void on_decode_token_end(const std::vector<std::vector<int>>& all_layers_topk,
                           const std::vector<std::vector<float>>& all_layers_scores) {
    scorer_->commit_token(all_layers_topk, all_layers_scores);
  }

  // ===== 访问器 =====
  const MeshConfig& config() const { return config_; }
  MeshScheduler& scheduler() { return *scheduler_; }
  MeshIoUring& io() { return *io_; }
  MeshPrefill& prefill() { return *prefill_; }
  MeshDecode& decode() { return *decode_; }
  EvictionScorer& scorer() { return *scorer_; }
  MeshSlotPool& pool(int layer, int tp) { return pools_[layer][tp]; }
  // A4 fix: layouts_ 改为 [layer][tp][expert] 3D
  const std::vector<std::vector<std::vector<ExpertFileLayout>>>& layouts() const { return layouts_; }

 private:
  MeshConfig config_;
  std::vector<int> numa_nodes_;
  size_t slot_bytes_ = 0;

  // [layer][tp] 的 slot 池
  std::vector<std::vector<MeshSlotPool>> pools_;

  // A4 fix: [layer][tp][expert] 的文件布局（每层偏移不同）
  std::vector<std::vector<std::vector<ExpertFileLayout>>> layouts_;

  // GPU expert mask
  std::vector<uint8_t> gpu_experts_mask_;

  // 组件
  std::unique_ptr<MeshIoUring> io_;
  std::unique_ptr<MeshScheduler> scheduler_;
  std::unique_ptr<EvictionScorer> scorer_;
  std::unique_ptr<MeshPrefill> prefill_;
  std::unique_ptr<MeshDecode> decode_;
  std::unique_ptr<MeshHandoff> handoff_;

  // 计算 slot 字节数（gate + up + down 三个矩阵）
  size_t compute_slot_bytes(const MeshConfig& config) const {
    // 根据 SKILL 第 8 节的双 NUMA 拆分：
    // gate: [hidden, intermediate/tp_count]
    // up:   [hidden, intermediate/tp_count]
    // down: [intermediate/tp_count, hidden]
    size_t gate_up_bytes = compute_gate_up_bytes(config);
    size_t down_bytes = compute_down_bytes(config);
    return gate_up_bytes * 2 + down_bytes;
  }

  size_t compute_gate_up_bytes(const MeshConfig& config) const {
    // gate 或 up 单块字节数
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;  // TP 切分后
    size_t elements = static_cast<size_t>(h) * i;
    switch (config.weight_type) {
      case WeightType::AMXINT4:
        return elements / 2;  // int4 = 0.5 byte
      case WeightType::BF16:
        return elements * 2;  // bf16 = 2 bytes
      default:
        return elements * 2;
    }
  }

  size_t compute_down_bytes(const MeshConfig& config) const {
    // down 单块字节数
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;
    size_t elements = static_cast<size_t>(i) * h;
    switch (config.weight_type) {
      case WeightType::AMXINT4:
        return elements / 2;
      case WeightType::BF16:
        return elements * 2;
      default:
        return elements * 2;
    }
  }
};

}  // namespace mesh
