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

#include <algorithm>
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
        // AMXINT4: 设置纯权重大小（不含 scale），用于 slot 内 scale 指针偏移
        if (config.weight_type == WeightType::AMXINT4) {
          pools_[l][tp].set_gate_up_weights_bytes(compute_gate_up_weights_bytes(config));
          pools_[l][tp].set_down_weights_bytes(compute_down_weights_bytes(config));
        }
        pools_[l][tp].init_expert_map(config.expert_num);
      }
    }

    // 初始化各组件
    io_ = std::make_unique<MeshIoUring>();
    scheduler_ = std::make_unique<MeshScheduler>(config.total_layers);
    scorer_ = std::make_unique<EvictionScorer>(config.total_layers, config.expert_num, config);
    prefill_ = std::make_unique<MeshPrefill>(config.total_layers, config.expert_num,
                                             config.cap, config.tp_count,
                                             slot_bytes_, numa_nodes);
    decode_ = std::make_unique<MeshDecode>(config);
    handoff_ = std::make_unique<MeshHandoff>();

    // 初始化 temporal 双缓冲
    prefill_->init_temporal();
    // B4 fix: 注入 gate_up_bytes 用于 temporal_ptr 偏移计算
    prefill_->set_gate_up_bytes(compute_gate_up_bytes(config));

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
                                 numa_nodes_, layouts_[l], l);
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
          // 辅因1 fix: on_complete 中调用 copy_scale_from_cache 把 scale 写入 slot
          io_->submit_load(e, tp, layout, gate_dst, up_dst, down_dst,
                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                           ReadPriority::Demand,
                           /*layer_idx=*/l,
                           /*on_complete=*/[this, l, tp, e](int, int, int) {
                             // Bug 4 fix: 先 copy scale，再 mark_cached
                             // mark_cached 后状态为 CACHED，AMX kernel 可能立即读取
                             // 若先 mark_cached 后 copy scale，AMX 会读到零 scale → NaN
                             if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(l)) {
                               // Bug 3 fix: bootstrap 中 bind(e, e) 使 slot_idx == e
                               int slot_idx = pools_[l][tp].expert_to_slot_idx(e);
                               if (slot_idx >= 0) {
                                 void* gs = pools_[l][tp].gate_scale_ptr(slot_idx);
                                 void* us = pools_[l][tp].up_scale_ptr(slot_idx);
                                 void* ds = pools_[l][tp].down_scale_ptr(slot_idx);
                                 io_->copy_scale_from_cache(l, tp, e, gs, us, ds,
                                                            nullptr, nullptr, nullptr,
                                                            layouts_[l][tp]);
                               }
                             }
                             // Bug 3 fix: mark_cached 参数是 slot_idx，bootstrap 中 slot_idx == e
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
    // B5: 每 token 开始时清空跨层 defer 队列
    prev_layer_deferred_.clear();
    total_deferred_ = 0;
    // B9: 统计
    stats_.decode_token_count++;
  }

  /**
   * @brief Decode 每层处理
   *
   * B5: 跨层 defer 累积 + overflow 阻塞
   * - 上一层的 deferred 专家在当前层转为 immediate 候选
   * - 如果 deferred 累积超过 max_deferred_per_token，阻塞等待 io_uring 完成
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

    // B5: 把上一层的 deferred 专家加入当前层的 immediate 候选
    // 它们应该已经通过 io_uring 读取完成
    std::vector<int> effective_topk = topk;
    if (!prev_layer_deferred_.empty()) {
      // 检查 deferred 专家是否已 CACHED
      MeshSlotPool& pool = pools_[layer_idx][tp_part_idx];
      for (int eid : prev_layer_deferred_) {
        if (pool.is_cached(eid)) {
          // 已缓存，加入 immediate 候选
          if (std::find(effective_topk.begin(), effective_topk.end(), eid) == effective_topk.end()) {
            effective_topk.push_back(eid);
          }
        }
        // 未缓存的 deferred 专家在本层继续等待（会在 split 中被当作 missing 处理）
      }
      // 清空上一层的 defer 队列
      prev_layer_deferred_.clear();
    }

    MeshSlotPool& pool = pools_[layer_idx][tp_part_idx];
    auto result = decode_->split(effective_topk, scores, pool, *scheduler_, layer_idx, tp_part_idx,
                          [this, &pool, layer_idx, tp_part_idx](int eid) {
                            // A5 fix: 用 expert_*_ptr 按 expert_id 查 slot，
                            // 不能把 expert_id 当 slot_idx 传给 gate_ptr（eid>=cap 时越界）
                            return std::vector<void*>{
                                pool.expert_gate_ptr(eid),
                                pool.expert_up_ptr(eid),
                                pool.expert_down_ptr(eid)};
                          },
                          // B6: 传入 is_gpu_expert 回调，过滤 GPU 专家
                          [this](int eid) { return is_gpu_expert(eid); });

    // B5: 更新跨层 defer 队列
    prev_layer_deferred_ = result.deferred;
    total_deferred_ += static_cast<int>(result.deferred.size());

    // B9: 统计
    stats_.decode_immediate_count += static_cast<uint64_t>(result.immediate.size());
    stats_.decode_deferred_count += static_cast<uint64_t>(result.deferred.size());
    stats_.defer_count += static_cast<uint64_t>(result.deferred.size());
    // hit/miss 统计：effective_topk 中已缓存的是 hit，未缓存的是 miss
    for (int eid : effective_topk) {
      if (is_gpu_expert(eid)) continue;  // GPU 专家不计入 hit/miss
      if (pool.is_cached(eid)) {
        stats_.cache_hit_count++;
      } else {
        stats_.cache_miss_count++;
      }
    }

    // B5: overflow 检查 — 如果 defer 累积超过阈值，阻塞等待 io_uring 完成
    if (total_deferred_ > config_.max_deferred_per_token * config_.total_layers) {
      // B9: 统计
      stats_.defer_overflow_count++;
      // 阻塞等待所有 inflight io_uring 完成
      io_->submit_and_wait();
      io_->process_cqes();
      total_deferred_ = 0;  // 重置计数
    }

    // A7: 排空调度器队列，提交 io_uring 读取
    drain_and_submit();

    // B2: 用当前层的实际路由（原始 topk + scores）预测下一层的 cross_layer_prior
    // 放在 drain_and_submit 之后：当前层 IO 已提交，为下一层驱逐评分做准备
    // 注意：用原始 topk/scores，不用 effective_topk（后者混入了上一层 deferred，不是真实路由）
    scorer_->predict_next_layer(layer_idx, topk, scores);

    return result;
  }

  /**
   * @brief A7: 排空调度器队列并提交 io_uring 读取
   *
   * 把 MeshScheduler 优先队列中的 ScheduledRequest 取出，
   * 为每个请求找到对应的 ExpertFileLayout 并提交给 MeshIoUring。
   * 完成后触发 mark_cached + 原始 on_complete 回调。
   *
   * B1: 如果专家未缓存且 slot 池满，先驱逐一个 victim 再 bind
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

      MeshSlotPool& pool = pools_[layer][tp];

      // B1: 如果专家未缓存，需要分配 slot
      void* gate_dst = req.gate_dst;
      void* up_dst = req.up_dst;
      void* down_dst = req.down_dst;

      if (!pool.is_cached(expert) && !is_gpu_expert(expert)) {
        // 专家未缓存，需要分配 slot
        int slot_idx = pool.expert_to_slot_idx(expert);

        if (slot_idx < 0) {
          // 专家没有绑定 slot，需要分配
          // Bug 7 fix: 用 find_free_slot 替代线性扫描 + 双重映射
          slot_idx = pool.find_free_slot();

          if (slot_idx < 0) {
            // 没有空闲 slot，需要驱逐
            slot_idx = decode_->evict_for_new_expert(pool, *scorer_, layer, expert);
            if (slot_idx >= 0) {
              // B9: 统计驱逐
              stats_.eviction_count++;
            }
          }

          if (slot_idx >= 0) {
            // bind slot
            pool.bind(slot_idx, expert);
            // 获取 slot 指针
            gate_dst = pool.gate_ptr(slot_idx);
            up_dst = pool.up_ptr(slot_idx);
            down_dst = pool.down_ptr(slot_idx);
          }
          // slot_idx < 0 表示无法分配，跳过此专家
        }
      }

      if (!gate_dst) continue;  // 无法分配 slot，跳过

      // Bug 3 fix: 查询 expert 绑定的 slot_idx，传给 lambda
      // mark_cached 参数是 slot_idx，不是 expert_id
      int slot_idx = pool.expert_to_slot_idx(expert);
      if (slot_idx < 0) continue;  // 未绑定 slot，跳过

      // B9: 统计 io_uring 读取
      stats_.io_uring_read_count++;
      stats_.io_uring_read_bytes += static_cast<uint64_t>(
          layout.gate_bytes + layout.up_bytes + layout.down_bytes);

      io_->submit_load(
          req.expert_id, req.tp_part_idx, layout,
          gate_dst, up_dst, down_dst,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // scale 从 cache memcpy
          req.priority,
          /*layer_idx=*/layer,
          /*on_complete=*/[this, layer, tp, expert, slot_idx, orig_on_complete](int, int, int) {
            // Bug 4 fix: 先 copy scale，再 mark_cached
            // mark_cached 后状态为 CACHED，AMX kernel 可能立即读取
            // 若先 mark_cached 后 copy scale，AMX 会读到零 scale → NaN
            if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(layer)) {
              void* gs = pools_[layer][tp].gate_scale_ptr(slot_idx);
              void* us = pools_[layer][tp].up_scale_ptr(slot_idx);
              void* ds = pools_[layer][tp].down_scale_ptr(slot_idx);
              io_->copy_scale_from_cache(layer, tp, expert, gs, us, ds,
                                         nullptr, nullptr, nullptr,
                                         layouts_[layer][tp]);
            }
            // Bug 3 fix: mark_cached 参数是 slot_idx，不是 expert_id
            pools_[layer][tp].mark_cached(slot_idx);
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

  // B9: 统计接口
  const MeshStats& stats() const { return stats_; }
  MeshStats& stats_mut() { return stats_; }

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

  // B5: 跨层 defer 队列状态
  std::vector<int> prev_layer_deferred_;  // 上一层的 deferred 专家
  int total_deferred_ = 0;                // 跨层累积 defer 计数

  // B9: 运行时统计
  MeshStats stats_;

  // 组件
  std::unique_ptr<MeshIoUring> io_;
  std::unique_ptr<MeshScheduler> scheduler_;
  std::unique_ptr<EvictionScorer> scorer_;
  std::unique_ptr<MeshPrefill> prefill_;
  std::unique_ptr<MeshDecode> decode_;
  std::unique_ptr<MeshHandoff> handoff_;

  // 计算 slot 字节数（gate + up + down 三个矩阵，AMXINT4 含 scale）
  size_t compute_slot_bytes(const MeshConfig& config) const {
    // 根据 SKILL 第 8 节的双 NUMA 拆分：
    // gate: [hidden, intermediate/tp_count]  BufferB n=intermediate/tp, k=hidden
    // up:   [hidden, intermediate/tp_count]  BufferB n=intermediate/tp, k=hidden
    // down: [intermediate/tp_count, hidden]  BufferB n=hidden, k=intermediate/tp
    // AMXINT4 BufferB 期望: n*k/2 (权重) + n*sizeof(float) (scale) 连续存放
    size_t gate_up_bytes = compute_gate_up_bytes(config);
    size_t down_bytes = compute_down_bytes(config);
    return gate_up_bytes * 2 + down_bytes;
  }

  // gate 或 up 单块字节数（权重 + scale）
  // BufferB 构造: n=intermediate/tp, k=hidden → 权重=n*k/2, scale=n*4
  size_t compute_gate_up_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;  // TP 切分后
    size_t elements = static_cast<size_t>(h) * i;
    switch (config.weight_type) {
      case WeightType::AMXINT4:
        return elements / 2 + static_cast<size_t>(i) * sizeof(float);  // int4 权重 + scale (n=i)
      case WeightType::BF16:
        return elements * 2;  // bf16 = 2 bytes
      default:
        return elements * 2;
    }
  }

  // down 单块字节数（权重 + scale）
  // BufferB 构造: n=hidden, k=intermediate/tp → 权重=n*k/2, scale=n*4
  size_t compute_down_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;
    size_t elements = static_cast<size_t>(i) * h;
    switch (config.weight_type) {
      case WeightType::AMXINT4:
        return elements / 2 + static_cast<size_t>(h) * sizeof(float);  // int4 权重 + scale (n=h)
      case WeightType::BF16:
        return elements * 2;
      default:
        return elements * 2;
    }
  }

  // AMXINT4: gate/up 的纯权重大小（不含 scale），用于 slot 内 scale 指针偏移
  size_t compute_gate_up_weights_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;
    return static_cast<size_t>(h) * i / 2;
  }

  // AMXINT4: down 的纯权重大小（不含 scale）
  size_t compute_down_weights_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;
    return static_cast<size_t>(i) * h / 2;
  }
};

}  // namespace mesh
