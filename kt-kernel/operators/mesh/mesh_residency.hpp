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
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <unistd.h>
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

    // 初始化 per-(layer,tp) mutex 数组
    load_mutexes_.resize(config.total_layers);
    for (int l = 0; l < config.total_layers; l++) {
      load_mutexes_[l].reserve(config.tp_count);
      for (int tp = 0; tp < config.tp_count; tp++) {
        load_mutexes_[l].push_back(std::make_unique<std::mutex>());
      }
    }

    // 初始化 temporal 双缓冲
    prefill_->init_temporal();
    // B4 fix: 注入 gate_up_bytes 用于 temporal_ptr 偏移计算
    prefill_->set_gate_up_bytes(compute_gate_up_bytes(config));

    // per-tp defer 队列状态初始化（不同 TP 的 forward_decode 并发调用 on_decode_layer）
    prev_layer_deferred_.resize(config.tp_count);
    total_deferred_.assign(config.tp_count, 0);
    prev_layer_topk_.resize(config.tp_count);
    prev_layer_scores_.resize(config.tp_count);

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
      fprintf(stderr, "[MESH bootstrap] preload_scale_cache start, layers=%d\n", config_.total_layers);
      for (int l = 0; l < config_.total_layers; l++) {
        io_->preload_scale_cache(config_.expert_num, config_.tp_count,
                                 numa_nodes_, layouts_[l], l);
      }
      fprintf(stderr, "[MESH bootstrap] preload_scale_cache done\n");
    }

    // pread 重试 lambda（带 EINTR 重试）
    auto pread_retry = [](int fd, void* dst, size_t bytes, off_t offset) -> bool {
      if (bytes == 0) return true;
      size_t done = 0;
      while (done < bytes) {
        ssize_t n = pread(fd, (char*)dst + done, bytes - done, offset + done);
        if (n < 0) {
          if (errno == EINTR) continue;
          fprintf(stderr, "[MESH] bootstrap pread failed: errno=%d (%s), bytes=%zu offset=%ld\n",
                  errno, strerror(errno), bytes, (long)offset);
          return false;
        }
        if (n == 0) {
          fprintf(stderr, "[MESH] bootstrap pread EOF: bytes=%zu done=%zu offset=%ld\n",
                  bytes, done, (long)offset);
          return false;
        }
        done += (size_t)n;
      }
      return true;
    };

    // 每层每 TP 读前 cap 个 CPU 专家
    // SKILL.md 第 39 行：读取所有专家的前 cap 个进入 slot
    // Bug fix: 原代码 `for (e = 0; e < cap; e++)` 只遍历 e=0~cap-1，跳过 GPU 专家后
    // 实际加载的 CPU 专家 < cap，ID >= cap 的 CPU 专家从未加载 → hit_rate < 100%
    // 修复：遍历所有专家，加载前 cap 个 CPU 专家，slot_idx 从 0 递增
    // slot 存 TP 切分后的权重（intermediate_size / tp_count），
    // 与 AMX_MOE_BASE::config_.intermediate_size（被 TP_MOE_Common 切分）匹配。
    // 每个 pool 只读取自己 tp 对应的文件切片，不合并。
    for (int l = 0; l < config_.total_layers; l++) {
      for (int tp = 0; tp < config_.tp_count; tp++) {
        MeshSlotPool& pool = pools_[l][tp];
        int slot_idx = 0;
        for (int e = 0; e < config_.expert_num && slot_idx < config_.cap; e++) {
          if (is_gpu_expert(e)) continue;  // GPU expert 跳过

          // 绑定 slot（slot_idx 从 0 递增，expert_id = e）
          pool.bind(slot_idx, e);

          const auto& layout = layouts_[l][tp][e];

          // 读取 gate 权重
          if (!pread_retry(layout.fd, pool.gate_ptr(slot_idx), layout.gate_bytes, layout.gate_offset)) {
            fprintf(stderr, "[MESH] bootstrap: gate pread failed expert=%d layer=%d tp=%d\n", e, l, tp);
            pool.unbind(slot_idx);  // 加载失败，释放 slot
            slot_idx--;
            continue;
          }
          // 读取 up 权重
          if (!pread_retry(layout.fd, pool.up_ptr(slot_idx), layout.up_bytes, layout.up_offset)) {
            fprintf(stderr, "[MESH] bootstrap: up pread failed expert=%d layer=%d tp=%d\n", e, l, tp);
            pool.unbind(slot_idx);
            slot_idx--;
            continue;
          }
          // 读取 down 权重
          if (layout.down_stride > 0) {
            // BF16 down_proj [E,H,I] 行主序，TP 沿 I 切不连续，逐行读取
            size_t row_bytes = layout.down_bytes / layout.down_rows;
            off_t src_off = layout.down_offset;
            bool down_ok = true;
            for (int r = 0; r < layout.down_rows; r++) {
              if (!pread_retry(layout.fd, static_cast<char*>(pool.down_ptr(slot_idx)) + r * row_bytes, row_bytes, src_off)) {
                fprintf(stderr, "[MESH] bootstrap: down row %d pread failed expert=%d\n", r, e);
                down_ok = false;
                break;
              }
              src_off += layout.down_stride;
            }
            if (!down_ok) {
              pool.unbind(slot_idx);
              slot_idx--;
              continue;
            }
          } else {
            if (!pread_retry(layout.fd, pool.down_ptr(slot_idx), layout.down_bytes, layout.down_offset)) {
              fprintf(stderr, "[MESH] bootstrap: down pread failed expert=%d layer=%d tp=%d\n", e, l, tp);
              pool.unbind(slot_idx);
              slot_idx--;
              continue;
            }
          }

          // AMXINT4: 从 scale cache 拷贝 scale 数据到 slot
          // 每个 tp 的 scale 对应自己的权重切片，直接复制
          if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(l)) {
            io_->copy_scale_from_cache(l, tp, e,
                                       pool.gate_scale_ptr(slot_idx),
                                       pool.up_scale_ptr(slot_idx),
                                       pool.down_scale_ptr(slot_idx),
                                       nullptr, nullptr, nullptr,
                                       layouts_[l][tp]);
          }

          // 标记 slot 为 CACHED
          pool.mark_cached(slot_idx);

          stats_.io_uring_read_count++;
          stats_.io_uring_read_bytes += layout.gate_bytes + layout.up_bytes + layout.down_bytes;
          slot_idx++;
        }
      }
    }

    fprintf(stderr, "[MESH] bootstrap: preloaded experts per layer per tp (cap=%d)\n",
            config_.cap);
  }

  // ===== 权重指针查询（KT 计算用）=====
  // 以下方法均内联 acquire_reader：返回非 nullptr 时 reader 已持有，
  // 调用者 GEMM 完后必须调用 release_reader。

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

  // 带 acquire_reader 的版本（hook 用）：返回非 nullptr 时 reader 已持有
  // 原子化：lookup + state check + reader + pointer 在同一把 slot_mtx_ 锁内完成，
  // 消除 TOCTOU 竞争（指针和 reader 保证在同一个 slot 上）
  void* get_gate_ptr_with_reader(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].acquire_gate_ptr(expert_id);
  }

  void* get_up_ptr_with_reader(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].acquire_up_ptr(expert_id);
  }

  void* get_down_ptr_with_reader(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;
    return pools_[layer][tp].acquire_down_ptr(expert_id);
  }

  // ===== 同步加载（slot 未命中时阻塞加载）=====
  // MESH 模式下 gate_bb_ 为空，slot 未命中不能回退，必须同步从 SSD 加载到 slot
  // 返回非 nullptr 时 reader 已持有，调用者 GEMM 完后必须调用 release_reader。

  void* get_or_load_gate_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;

    // 快速路径：原子化 acquire（lookup + state + reader + pointer 在同一锁内）
    void* ptr = pools_[layer][tp].acquire_gate_ptr(expert_id);
    if (ptr) return ptr;

    // 未缓存或 acquire 失败（被驱逐）：同步加载
    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;

    // 加载完成后再原子化 acquire（可能又被其他线程驱逐，重试一次）
    ptr = pools_[layer][tp].acquire_gate_ptr(expert_id);
    if (ptr) return ptr;

    // 极端竞争下被驱逐，再加载一次
    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;
    ptr = pools_[layer][tp].acquire_gate_ptr(expert_id);
    return ptr;
  }

  void* get_or_load_up_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;

    void* ptr = pools_[layer][tp].acquire_up_ptr(expert_id);
    if (ptr) return ptr;

    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;

    ptr = pools_[layer][tp].acquire_up_ptr(expert_id);
    if (ptr) return ptr;

    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;
    ptr = pools_[layer][tp].acquire_up_ptr(expert_id);
    return ptr;
  }

  void* get_or_load_down_ptr(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return nullptr;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return nullptr;

    void* ptr = pools_[layer][tp].acquire_down_ptr(expert_id);
    if (ptr) return ptr;

    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;

    ptr = pools_[layer][tp].acquire_down_ptr(expert_id);
    if (ptr) return ptr;

    if (!load_expert_sync(layer, tp, expert_id)) return nullptr;
    ptr = pools_[layer][tp].acquire_down_ptr(expert_id);
    return ptr;
  }

  // ===== 引用计数（防止 AMX 计算期间 slot 被驱逐）=====
  // acquire_reader 已内联到 get/get_or_load 方法中，外部无需调用。
  // release_reader 仍由 do_gate_up_gemm/do_down_gemm 在 GEMM 完后调用。

  void release_reader(int layer, int tp, int expert_id) {
    if (layer < 0 || layer >= (int)pools_.size()) return;
    if (tp < 0 || tp >= (int)pools_[layer].size()) return;
    pools_[layer][tp].release_reader(expert_id);
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
    // B5: 每 token 开始时清空所有 tp 的跨层 defer 队列
    // on_decode_token_start 由 Python 侧调用（非并发），安全清空所有 tp
    for (int tp = 0; tp < (int)prev_layer_deferred_.size(); tp++) {
      prev_layer_deferred_[tp].clear();
      total_deferred_[tp] = 0;
    }
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
    // 检测 double call：KTransformers defer 机制每层调用 GEMM 两次
    // （experts_base.py:427-454 两次 submit_with_cuda_stream）
    // 第一次（immediate GEMM）：topk 中 -1 数量 = defer_count_（deferred 位置填 -1）
    // 第二次（deferred GEMM）：topk 中 -1 数量 = num_experts_per_tok - defer_count_（immediate 位置填 -1）
    //
    // SKILL.md 第 286 行：当前层计算只执行 immediate 部分，合并上一层 deferred 输出由 KT 原版负责。
    // 第二次 GEMM 计算 deferred 组专家。
    //
    // 根因修复：第一次 on_decode_layer 的 topk 中 deferred 位置是 -1，看不到真实 expert_id，
    // 无法为 deferred 专家提交 io_uring 预取。第二次调用时必须 blocking_load_experts，
    // 否则 GEMM 会走 load_expert_sync 的同步 pread 路径（SLOT MISS + 串行阻塞读）。
    // blocking_load_experts 会提交 io_uring（并行 I/O）并 spin-wait 直到 CACHED，
    // 比让 GEMM 逐个 pread（串行同步 I/O）快得多。
    if (decode_->defer_count() > 0) {
      int neg_one_count = 0;
      for (int eid : topk) {
        if (eid < 0) neg_one_count++;
      }
      if (neg_one_count > decode_->defer_count()) {
        // 第二次调用（deferred GEMM）：处理 deferred 组专家
        io_->process_cqes();

        // 提取 deferred 专家（topk 中非 -1 的）
        std::vector<int> deferred_experts;
        deferred_experts.reserve(topk.size() - neg_one_count);
        for (int eid : topk) {
          if (eid >= 0) deferred_experts.push_back(eid);
        }

        // 对未 CACHED 的 deferred 专家做 blocking_load（提交 io_uring + spin-wait）
        // 避免后续 GEMM 走 load_expert_sync 的同步 pread 路径
        if (!deferred_experts.empty()) {
          blocking_load_experts(layer_idx, tp_part_idx, deferred_experts);
        }

        return MeshDecode::SplitResult();
      }
    }

    // A7: 先处理已完成的 io_uring CQE，触发 mark_cached 回调
    auto _t0 = std::chrono::steady_clock::now();
    io_->process_cqes();
    auto _t1 = std::chrono::steady_clock::now();

    // on_decode_token_start 未被 Python 侧调用，在 layer_idx==0 时自动执行
    // CAS 确保每 token 只有一个 TP 执行 inc_timeline_step
    if (layer_idx == 0) {
      int expected = decode_token_seq_.load(std::memory_order_acquire);
      if (decode_token_seq_.compare_exchange_strong(expected, expected + 1,
                                                    std::memory_order_acq_rel)) {
        scheduler_->inc_timeline_step();
        stats_.decode_token_count++;
      }
      // 每 token 第一层清空自己的 total_deferred_（无竞争，每个 TP 只操作自己的）
      total_deferred_[tp_part_idx] = 0;
    }

    MeshSlotPool& pool = pools_[layer_idx][tp_part_idx];

    // B5: 上一层的 deferred 专家在当前层必须被计算处理完毕（SKILL.md 第 100 行）
    // 已 cached 的：KTransformers 原版 defer 机制会在当前层合并它们的输出，MESH 不需要管
    // 未 cached 的：需要阻塞读，确保它们在当前层被计算
    std::vector<int> prev_deferred_blocking;
    if (!prev_layer_deferred_[tp_part_idx].empty()) {
      for (int eid : prev_layer_deferred_[tp_part_idx]) {
        if (!pool.is_cached(eid) && !is_gpu_expert(eid)) {
          prev_deferred_blocking.push_back(eid);
        }
      }
      prev_layer_deferred_[tp_part_idx].clear();
    }

    // SKILL.md 第 280 行：split 只对原始 topk 分组
    // target_immediate = k - defer_count，deferred 最多 defer_count 个
    // 不把上一层的 deferred 加入 effective_topk（否则 deferred 指数级增长）
    auto result = decode_->split(topk, scores, pool, layer_idx,
                          static_cast<int>(topk.size()),
                          [this](int eid) { return is_gpu_expert(eid); });
    auto _t2 = std::chrono::steady_clock::now();

    // 上一层的 deferred 未 cached 的加入 blocking_missing（阻塞读）
    result.blocking_missing.insert(result.blocking_missing.end(),
                                   prev_deferred_blocking.begin(),
                                   prev_deferred_blocking.end());

    // SKILL.md 第 282 行：blocking_missing 阻塞读补充 immediate 组
    // 为 blocking_missing 分配 slot + 提交 io_uring + 阻塞等待完成
    blocking_load_experts(layer_idx, tp_part_idx, result.blocking_missing);
    auto _t3 = std::chrono::steady_clock::now();
    // 阻塞读完成后，只把 prev_deferred_blocking 加入 immediate 组
    // （split 返回的 blocking_missing 已在 immediate 中，不重复加入）
    result.immediate.insert(result.immediate.end(),
                            prev_deferred_blocking.begin(),
                            prev_deferred_blocking.end());

    // SKILL.md 第 284 行：deferred_missing 异步 io_uring 读取，推迟到下一层
    for (int eid : result.deferred_missing) {
      // 获取 slot 指针（可能为 nullptr，drain_and_submit 会分配 slot）
      auto ptrs = std::vector<void*>{
          pool.expert_gate_ptr(eid),
          pool.expert_up_ptr(eid),
          pool.expert_down_ptr(eid)};
      scheduler_->submit_decode_deferred(layer_idx, eid, tp_part_idx,
                                         ptrs[0], ptrs[1], ptrs[2]);
    }

    // B5: 更新跨层 defer 队列（只包含当前层的 deferred，最多 defer_count_ 个）
    prev_layer_deferred_[tp_part_idx] = result.deferred;
    total_deferred_[tp_part_idx] += static_cast<int>(result.deferred.size());

    // B9: 统计
    stats_.decode_immediate_count += static_cast<uint64_t>(result.immediate.size());
    stats_.decode_deferred_count += static_cast<uint64_t>(result.deferred.size());
    stats_.defer_count += static_cast<uint64_t>(result.deferred.size());
    // hit/miss 统计：只对原始 topk（不含上一层 deferred）
    // 跳过 -1（KTransformers defer 机制产生的无效位置，不是真实 miss）
    for (int eid : topk) {
      if (eid < 0) continue;  // 跳过 -1
      if (is_gpu_expert(eid)) continue;  // GPU 专家不计入 hit/miss
      if (pool.is_cached(eid)) {
        stats_.cache_hit_count++;
      } else {
        stats_.cache_miss_count++;
      }
    }

    // B5: overflow 检查 — 如果 defer 累积超过阈值，处理已完成的 CQE 并重置计数
    // 不用 submit_and_wait（持锁阻塞等 CQE，大量 inflight 时死锁）。
    // 引用计数保护下，deferred_missing 提交的 io_uring 会在后续层 process_cqes 时完成，
    // 不需要在这里阻塞等待。
    if (total_deferred_[tp_part_idx] > config_.max_deferred_per_token * config_.total_layers) {
      stats_.defer_overflow_count++;
      io_->process_cqes();  // 只处理已完成的 CQE，不阻塞等待
      total_deferred_[tp_part_idx] = 0;  // 重置计数
    }

    // A7: 排空当前 TP 的调度器队列，提交 io_uring 读取
    // SKILL.md 第 54 行：只处理本 TP 的请求，不跨 NUMA 操作
    drain_and_submit(tp_part_idx);
    auto _t4 = std::chrono::steady_clock::now();

    // SKILL.md 第 135 行：满驻留自动跳过
    // 当 slot 容量 ≥ CPU 专家数时，驱逐不可能发生，Heat/Markov 信号无价值
    int cpu_expert_num = config_.expert_num - config_.num_gpu_experts;
    if (!scorer_->should_skip_layer(pool.cap(), cpu_expert_num)) {
      // 方案 B: 只 tp_part_idx==0 更新 scorer，避免 NUMA 间 mtx_ 锁竞争
      // 安全性: TP_MOE_Common 按 intermediate_size 切分，各 NUMA 看到的 topk/scores 相同，
      // 所以 NUMA 1 重复提交是冗余的，跳过后结果完全一致。
      if (tp_part_idx == 0) {
        scorer_->commit_layer(layer_idx, topk, scores, prev_layer_topk_[tp_part_idx], prev_layer_scores_[tp_part_idx]);
        // B2: 用当前层的实际路由预测下一层的 cross_layer_prior
        scorer_->predict_next_layer(layer_idx, topk, scores);
      }
      prev_layer_topk_[tp_part_idx] = topk;
      prev_layer_scores_[tp_part_idx] = scores;
    }
    auto _t5 = std::chrono::steady_clock::now();

    // 时间统计（每 200 层打印一次）
    {
      static thread_local uint64_t stat_n = 0;
      static thread_local double stat_total = 0, stat_cqes = 0, stat_split = 0;
      static thread_local double stat_block = 0, stat_drain = 0, stat_scorer = 0;
      stat_n++;
      stat_total += std::chrono::duration<double, std::milli>(_t5 - _t0).count();
      stat_cqes  += std::chrono::duration<double, std::milli>(_t1 - _t0).count();
      stat_split += std::chrono::duration<double, std::milli>(_t2 - _t1).count();
      stat_block += std::chrono::duration<double, std::milli>(_t3 - _t2).count();
      stat_drain += std::chrono::duration<double, std::milli>(_t4 - _t3).count();
      stat_scorer+= std::chrono::duration<double, std::milli>(_t5 - _t4).count();
      if (stat_n % 200 == 0) {
        fprintf(stderr, "[MESH TIME] L%d tp%d avg: total=%.3fms cqes=%.3fms split=%.3fms "
                "block=%.3fms drain=%.3fms scorer=%.3fms (n=%lu)\n",
                layer_idx, tp_part_idx,
                stat_total / stat_n, stat_cqes / stat_n, stat_split / stat_n,
                stat_block / stat_n, stat_drain / stat_n, stat_scorer / stat_n,
                (unsigned long)stat_n);
      }
    }

    return result;
  }

  /**
   * @brief SKILL.md 第 282 行：阻塞读 missing 专家补充 immediate 组
   *
   * 为每个 missing 专家分配 slot + 提交 io_uring + 阻塞等待完成。
   * 完成后专家状态为 CACHED，可加入 immediate 组。
   *
   * 引用计数保护：overwrite CV 等待 active_readers 归零，
   * GEMM 时 acquire_reader 持有的专家不会被驱逐（SKILL.md 第 62/125 行）。
   */
  void blocking_load_experts(int layer_idx, int tp_part_idx,
                             const std::vector<int>& experts) {
    if (experts.empty()) return;
    if (layouts_.empty()) return;

    MeshSlotPool& pool = pools_[layer_idx][tp_part_idx];

    int submitted = 0;
    int skipped = 0;
    bool needs_wait = false;  // 是否有专家需要 spin-wait（未 CACHED 的）
    // fix4: 记录需要 spin-wait 的专家及其 slot，供 stale LOADING 重试使用
    std::vector<std::pair<int,int>> wait_list;  // (eid, slot_idx)
    for (int eid : experts) {
      if (pool.is_cached(eid)) continue;  // 已缓存，跳过
      if (is_gpu_expert(eid)) continue;   // GPU 专家，跳过
      needs_wait = true;  // 至少有一个专家未 CACHED，需要 spin-wait

      // 分配 slot
      int slot_idx = pool.expert_to_slot_idx(eid);
      bool just_bound = false;  // fix4: 区分"本次刚 bind"和"之前调用遗留的 LOADING"
      if (slot_idx < 0) {
        slot_idx = pool.find_free_slot();
        if (slot_idx >= 0) {
          // fix5: 空闲 slot，需要 bind
          pool.bind(slot_idx, eid);  // bind() 会设置 state=LOADING
          just_bound = true;
        } else {
          // fix5: 没有空闲 slot，需要驱逐
          // evict_for_new_expert 调用 overwrite，overwrite 内部已持锁完成 bind
          // （设置 state=LOADING, bound_expert_id, expert_to_slot_）
          // 不能再调用 pool.bind，否则不持锁重复设置可能与 overwrite 的持锁操作竞争
          slot_idx = decode_->evict_for_new_expert(pool, *scorer_, layer_idx, eid);
          if (slot_idx >= 0) {
            stats_.eviction_count++;
            just_bound = true;
          }
        }
      }

      if (slot_idx < 0) {
        fprintf(stderr, "[MESH] blocking_load: slot alloc FAILED eid=%d layer=%d tp=%d\n",
                eid, layer_idx, tp_part_idx);
        skipped++;
        continue;  // 无法分配 slot，跳过
      }

      // fix4: 防止 double-submission，但区分两种 LOADING：
      // 1. just_bound=true：本次调用刚 bind，必须提交 io_uring（旧 fix3b 的 bug 就是这里误跳过）
      // 2. just_bound=false：之前调用遗留的 LOADING（io_uring 可能在飞行中）
      //    - inflight>0：有 io_uring 在飞行中，跳过提交，spin-wait 会处理
      //    - inflight==0：io_uring 丢失（提交失败/CQE 错误），重置并重新提交
      ExpertState state = pool.slot_state(slot_idx);
      if (state == ExpertState::LOADING && !just_bound) {
        if (io_->inflight_count() > 0) {
          // io_uring 可能在飞行中，跳过提交，spin-wait 会处理
          wait_list.emplace_back(eid, slot_idx);
          continue;
        }
        // inflight==0 但 LOADING：io_uring 丢失，重置并重新提交
        fprintf(stderr, "[MESH] blocking_load: stale LOADING (inflight=0), resubmit eid=%d slot=%d layer=%d tp=%d\n",
                eid, slot_idx, layer_idx, tp_part_idx);
        pool.unbind(slot_idx);
        pool.bind(slot_idx, eid);
        // 继续执行下面的 io_uring 提交逻辑
      }

      // 获取 slot 指针
      void* gate_dst = pool.gate_ptr(slot_idx);
      void* up_dst = pool.up_ptr(slot_idx);
      void* down_dst = pool.down_ptr(slot_idx);

      const auto& layout = layouts_[layer_idx][tp_part_idx][eid];

      stats_.io_uring_read_count++;
      stats_.io_uring_read_bytes += static_cast<uint64_t>(
          layout.gate_bytes + layout.up_bytes + layout.down_bytes);

      io_->submit_load(
          eid, tp_part_idx, layout,
          gate_dst, up_dst, down_dst,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  // scale 从 cache memcpy
          ReadPriority::Demand,
          /*layer_idx=*/layer_idx,
          /*on_complete=*/[this, layer_idx, tp_part_idx, eid, slot_idx](int, int, int) {
            // Bug 4 fix: 先 copy scale，再 mark_cached
            if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(layer_idx)) {
              void* gs = pools_[layer_idx][tp_part_idx].gate_scale_ptr(slot_idx);
              void* us = pools_[layer_idx][tp_part_idx].up_scale_ptr(slot_idx);
              void* ds = pools_[layer_idx][tp_part_idx].down_scale_ptr(slot_idx);
              io_->copy_scale_from_cache(layer_idx, tp_part_idx, eid, gs, us, ds,
                                         nullptr, nullptr, nullptr,
                                         layouts_[layer_idx][tp_part_idx]);
            }
            pools_[layer_idx][tp_part_idx].mark_cached(slot_idx);
          });
      submitted++;
      wait_list.emplace_back(eid, slot_idx);  // fix4: 记录已提交的专家
    }

    if (!needs_wait) return;  // 所有专家已 CACHED/GPU，无需等待
    if (submitted == 0 && skipped > 0 && wait_list.empty()) {
      // 所有未 CACHED 的专家都分配 slot 失败，且没有 LOADING 中的专家
      // 无法 spin-wait，让 GEMM 的 load_expert_sync 处理
      fprintf(stderr, "[MESH] blocking_load: all %d experts skipped (no slot) layer=%d tp=%d\n",
              skipped, layer_idx, tp_part_idx);
      return;
    }
    // 注意：如果 submitted==0 但 needs_wait==true 且 wait_list 非空，
    // 说明所有未 CACHED 的专家都在 LOADING 状态（io_uring 可能在飞行中），
    // 必须继续 spin-wait，不能提前返回

    // SKILL.md 第 282 行：阻塞等待 blocking_missing 的 io_uring 完成
    // 不用 submit_and_wait（会等待所有 inflight，包括之前层 deferred_missing 提交的 io_uring，
    // 如果那些读取慢会死锁）。改用 process_cqes + spin-wait 等 blocking_missing CACHED。
    // 加超时检查（120秒），超时打印警告并返回。
    // fix4: 加 stale LOADING 检测 — 等待 >5s 后，如果仍有专家 LOADING 且 inflight==0，
    // 说明 io_uring 丢失，重置 slot 并重新提交 io_uring
    auto t0 = std::chrono::steady_clock::now();
    int resubmit_count = 0;
    while (true) {
      // 处理已完成的 CQE，触发 mark_cached 回调
      io_->process_cqes();

      // 检查所有 blocking_missing 是否已 CACHED
      bool all_cached = true;
      for (int eid : experts) {
        if (is_gpu_expert(eid)) continue;
        if (!pool.is_cached(eid)) {
          all_cached = false;
          break;
        }
      }
      if (all_cached) break;

      // fix6: 检测所有未 CACHED 专家的 slot 状态（不只是 wait_list）
      // fix5 只检测 wait_list，漏掉了"提交时已 CACHED 但 spin-wait 期间被其他线程 evict"的专家
      // 这些专家被 evict 后 is_cached 返回 false，all_cached=false，但 fix5 不检测它们
      // 导致 spin-wait 永远等待已不存在的专家 → 120s timeout
      // fix6: 遍历 experts（输入参数），检测所有未 CACHED 专家的 slot 解绑/覆盖
      for (int eid : experts) {
        if (is_gpu_expert(eid)) continue;
        if (pool.is_cached(eid)) continue;
        int cur_slot = pool.expert_to_slot_idx(eid);
        if (cur_slot < 0) {
          // slot 被解绑（被 evict_for_new_expert 的 unbind/erase 清除）
          // 让 GEMM 的 load_expert_sync（持有 load_mutexes_ 锁）重新加载
          fprintf(stderr, "[MESH] blocking_load: slot unbound during spin-wait, eid=%d "
                  "layer=%d tp=%d inflight=%d — was cached at submit, evicted by concurrent thread, "
                  "returning to let GEMM load_expert_sync handle\n",
                  eid, layer_idx, tp_part_idx, io_->inflight_count());
          return;
        }
      }

      // 超时检查（120秒）
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed > 120) {
        // 诊断：打印未 CACHED 的专家和 inflight 状态
        fprintf(stderr, "[MESH] blocking_load_experts: timeout layer=%d tp=%d elapsed=%lds "
                "submitted=%d skipped=%d inflight=%d\n",
                layer_idx, tp_part_idx, (long)elapsed, submitted, skipped,
                io_->inflight_count());
        for (int eid : experts) {
          if (is_gpu_expert(eid)) continue;
          if (!pool.is_cached(eid)) {
            int sidx = pool.expert_to_slot_idx(eid);
            fprintf(stderr, "[MESH]   NOT CACHED: eid=%d slot=%d state=%d\n",
                    eid, sidx, sidx >= 0 ? (int)pool.slot_state(sidx) : -1);
          }
        }
        break;
      }

      // fix4: stale LOADING 检测 — 等待 >5s 后，如果仍有专家 LOADING 且 inflight==0，
      // 说明 io_uring 丢失（提交失败/CQE 错误/被其他线程误处理）
      // 重置 slot 状态并重新提交 io_uring
      if (elapsed > 5 && io_->inflight_count() == 0 && resubmit_count < 3) {
        for (auto& wl : wait_list) {
          int eid = wl.first;
          int sidx = wl.second;
          if (is_gpu_expert(eid)) continue;
          if (pool.is_cached(eid)) continue;
          // fix5: 再次检查 slot 是否仍绑定到 eid（防御性，修改2已检测但中间可能有变化）
          int cur_slot = pool.expert_to_slot_idx(eid);
          if (cur_slot != sidx) {
            fprintf(stderr, "[MESH] blocking_load: slot reassigned before stale LOADING resubmit, "
                    "eid=%d orig_slot=%d cur_slot=%d layer=%d tp=%d — skipping\n",
                    eid, sidx, cur_slot, layer_idx, tp_part_idx);
            continue;
          }
          ExpertState st = pool.slot_state(sidx);
          if (st == ExpertState::LOADING) {
            fprintf(stderr, "[MESH] blocking_load: stale LOADING after %lds, resubmit eid=%d slot=%d "
                    "layer=%d tp=%d (resubmit #%d)\n",
                    (long)elapsed, eid, sidx, layer_idx, tp_part_idx, resubmit_count + 1);
            pool.unbind(sidx);
            pool.bind(sidx, eid);
            const auto& layout = layouts_[layer_idx][tp_part_idx][eid];
            void* gate_dst = pool.gate_ptr(sidx);
            void* up_dst = pool.up_ptr(sidx);
            void* down_dst = pool.down_ptr(sidx);
            stats_.io_uring_read_count++;
            stats_.io_uring_read_bytes += static_cast<uint64_t>(
                layout.gate_bytes + layout.up_bytes + layout.down_bytes);
            io_->submit_load(
                eid, tp_part_idx, layout,
                gate_dst, up_dst, down_dst,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                ReadPriority::Demand,
                layer_idx,
                [this, layer_idx, tp_part_idx, eid, sidx](int, int, int) {
                  if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(layer_idx)) {
                    void* gs = pools_[layer_idx][tp_part_idx].gate_scale_ptr(sidx);
                    void* us = pools_[layer_idx][tp_part_idx].up_scale_ptr(sidx);
                    void* ds = pools_[layer_idx][tp_part_idx].down_scale_ptr(sidx);
                    io_->copy_scale_from_cache(layer_idx, tp_part_idx, eid, gs, us, ds,
                                               nullptr, nullptr, nullptr,
                                               layouts_[layer_idx][tp_part_idx]);
                  }
                  pools_[layer_idx][tp_part_idx].mark_cached(sidx);
                });
            submitted++;
            resubmit_count++;
          }
        }
      }

      // 短暂 sleep，避免 100% CPU
      usleep(100);  // 100us
    }
  }

  // Bug 1+2 fix: 简化版 on_decode_layer（供 hook 调用，接收原始指针参数）
  // 在 forward_decode 的 GEMM 之前调用，更新 Heat/Markov 使驱逐策略生效
  // 注意：异步 io_uring 预取暂时禁用（CQE 在 GEMM 期间无法可靠处理），
  // 改由 load_expert_sync 的并行 pread（Bug 3 fix）处理 missing 专家
  void on_decode_layer_simple(int layer_idx, int tp_part_idx,
                              const int* topk_ptr, int k,
                              const float* weights, int expert_num) {
    if (!config_.enabled) return;

    // 构造 topk 和 scores（全长，top-k 位置填入 weights，其余 0）
    std::vector<int> topk(topk_ptr, topk_ptr + k);
    std::vector<float> scores(expert_num, 0.0f);
    for (int i = 0; i < k; i++) {
      int eid = topk_ptr[i];
      if (eid >= 0 && eid < expert_num) {
        scores[eid] = weights[i];
      }
    }

    // 方案 B: 只 tp_part_idx==0 更新 scorer，避免 NUMA 间 mtx_ 锁竞争
    if (tp_part_idx == 0) {
      scorer_->commit_layer(layer_idx, topk, scores, prev_layer_topk_[tp_part_idx], prev_layer_scores_[tp_part_idx]);
      scorer_->predict_next_layer(layer_idx, topk, scores);
    }
    prev_layer_topk_[tp_part_idx] = topk;
    prev_layer_scores_[tp_part_idx] = scores;
  }

  /**
   * @brief A7: 排空当前 TP 的调度器队列并提交 io_uring 读取
   *
   * SKILL.md 第 54 行：驱逐和覆盖只影响本 TP 的本地 shard，不跨 NUMA 操作
   * 只取走 tp_part_idx 对应的请求，其余 TP 的请求留在队列中由各自 TP 处理
   *
   * 把 MeshScheduler 优先队列中的 ScheduledRequest 取出，
   * 为每个请求找到对应的 ExpertFileLayout 并提交给 MeshIoUring。
   * 完成后触发 mark_cached + 原始 on_complete 回调。
   *
   * B1: 如果专家未缓存且 slot 池满，先驱逐一个 victim 再 bind
   */
  void drain_and_submit(int tp_part_idx = -1) {
    std::vector<ScheduledRequest> requests;
    if (tp_part_idx >= 0) {
      requests = scheduler_->drain_all_for_tp(tp_part_idx);
    } else {
      requests = scheduler_->drain_all();
    }
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
            // 引用计数保护：overwrite CV 等待 reader 归零，GEMM 中的专家不会被驱逐
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

  // B5: 跨层 defer 队列状态 — per-tp，因为不同 TP 的 forward_decode 并发调用 on_decode_layer
  // do_numa_job 并发执行 tps[numa_id]->forward()，不同 TP 同时操作这些变量会导致
  // std::vector 的并发 clear()/operator=/iterate → use-after-free/堆损坏/段错误
  std::vector<std::vector<int>> prev_layer_deferred_;  // [tp] 上一层的 deferred 专家
  std::vector<int> total_deferred_;                    // [tp] 跨层累积 defer 计数

  // Bug 2 fix: 逐层 Heat/Markov 更新所需的上一层 topk/scores — per-tp
  std::vector<std::vector<int>> prev_layer_topk_;      // [tp]
  std::vector<std::vector<float>> prev_layer_scores_;  // [tp]

  // on_decode_token_start 未被 Python 侧调用，改在 on_decode_layer layer_idx==0 时自动执行
  // 用 atomic + CAS 确保每 token 只有一个 TP 执行 inc_timeline_step
  std::atomic<int> decode_token_seq_{0};  // 已处理的 decode token 序号

  // 并发保护：per-(layer,tp) mutex，保护 load_expert_sync 的 slot 分配和加载
  // 防止多个线程同时为同一/不同 expert 分配 slot 导致竞争
  // 用 unique_ptr 因为 std::mutex 不可移动
  std::vector<std::vector<std::unique_ptr<std::mutex>>> load_mutexes_;

  // B9: 运行时统计
  MeshStats stats_;

  // 组件
  std::unique_ptr<MeshIoUring> io_;
  std::unique_ptr<MeshScheduler> scheduler_;
  std::unique_ptr<EvictionScorer> scorer_;
  std::unique_ptr<MeshPrefill> prefill_;
  std::unique_ptr<MeshDecode> decode_;
  std::unique_ptr<MeshHandoff> handoff_;

  // ===== 同步加载专家到 slot（slot 未命中时阻塞调用）=====
  // 返回 true 表示加载成功，false 表示失败（layouts 未注入或无法分配 slot）
  // 根因修复：用 pread 直接读取，替代 io_uring。
  // io_uring 的 CQE 在 EINTR 时 res<0，但旧代码仍调用 on_complete 标记 slot 为 CACHED，
  // 导致 slot 内是垃圾数据，AMX kernel 读取后内存损坏 → double free。
  // Bug 3 fix: 减小锁粒度 — pread 移出锁外，允许同层同 TP 并行 I/O
  // 锁内只做 slot 分配（find_free_slot / evict + bind/overwrite），锁外做 pread。
  // 同一 expert 的并发线程：发现 LOADING 状态后释放锁，spin-wait 等 CACHED。
  bool load_expert_sync(int layer, int tp, int expert_id) {
    if (is_gpu_expert(expert_id)) return false;
    if (layouts_.empty()) return false;
    if (layer < 0 || layer >= (int)layouts_.size()) return false;
    if (tp < 0 || tp >= (int)layouts_[layer].size()) return false;
    if (expert_id < 0 || expert_id >= (int)layouts_[layer][tp].size()) return false;

    MeshSlotPool& pool = pools_[layer][tp];

    // 快速路径：已缓存直接返回（无需加锁）
    if (pool.is_cached(expert_id)) return true;

    // 外层 retry loop：处理 slot 被 evict 后需要重新分配的情况
    // Race condition: Thread A sets LOADING + pread, Thread B spin-waits,
    // Thread C evicts the slot (overwrite) after pread completes,
    // Thread B never sees CACHED → 必须检测并 retry
    auto t0 = std::chrono::steady_clock::now();
    for (int outer_retry = 0; ; outer_retry++) {
      if (pool.is_cached(expert_id)) return true;

      // 超时检查（120秒）— 避免超过 NCCL watchdog 的 600 秒超时
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - t0).count();
      if (elapsed > 120) {
        fprintf(stderr, "[MESH] load_expert_sync: timeout expert=%d layer=%d tp=%d elapsed=%lds\n",
                expert_id, layer, tp, (long)elapsed);
        return false;
      }

      int slot_idx = -1;
      bool wait_for_loading = false;

      // 内层 retry loop: 当所有 slot 都在 LOADING 时，evict_for_new_expert 找不到
      // CACHED slot 来驱逐。等待某个 pread 完成后 slot 变为 CACHED，再重试。
      for (int alloc_retry = 0; alloc_retry < 100000; alloc_retry++) {
        if (pool.is_cached(expert_id)) return true;
        {
          std::lock_guard<std::mutex> lock(*load_mutexes_[layer][tp]);
          if (pool.is_cached(expert_id)) return true;

          slot_idx = pool.expert_to_slot_idx(expert_id);
          if (slot_idx >= 0) {
            ExpertState state = pool.slot_state(slot_idx);
            if (state == ExpertState::LOADING) {
              wait_for_loading = true;
            }
            break;  // 跳出 alloc retry loop
          } else {
            slot_idx = pool.find_free_slot();
            if (slot_idx >= 0) {
              pool.bind(slot_idx, expert_id);  // 设置 LOADING
              break;
            } else {
              slot_idx = decode_->evict_for_new_expert(pool, *scorer_, layer, expert_id);
              if (slot_idx >= 0) {
                break;
              }
              // slot_idx < 0: 所有 slot 都在 LOADING，无法驱逐
            }
          }
        }
        // 锁已释放 — spin-wait 等待 slot 释放
        if ((alloc_retry & 63) == 0) io_->process_cqes();
        if ((alloc_retry % 1000) == 0) usleep(1);
        sched_yield();
      }

      if (wait_for_loading) {
        // 等待其他线程完成 pread 加载
        // 关键修复：检测 slot 是否被 evict（overwrite），如果是则 retry
        while (true) {
          if (pool.is_cached(expert_id)) return true;
          // 检查 expert 是否仍绑定在 LOADING 状态的 slot 上
          int sidx = pool.expert_to_slot_idx(expert_id);
          if (sidx < 0) break;  // expert 不再绑定任何 slot — 被 evict，retry
          ExpertState st = pool.slot_state(sidx);
          if (st != ExpertState::LOADING) break;  // 状态变了（CACHED 或被 evict），retry
          io_->process_cqes();
          usleep(100);  // 100us
        }
        continue;  // 外层 retry
      }

      if (slot_idx < 0) continue;  // alloc 失败，外层 retry

      // 锁已释放 — 以下 pread 在锁外执行，允许同层同 TP 的其他 expert 并行 I/O
      const auto& layout = layouts_[layer][tp][expert_id];
      void* gate_dst = pool.gate_ptr(slot_idx);
      void* up_dst = pool.up_ptr(slot_idx);
      void* down_dst = pool.down_ptr(slot_idx);

      stats_.io_uring_read_count++;
      stats_.io_uring_read_bytes += layout.gate_bytes + layout.up_bytes + layout.down_bytes;

      // 用 pread 同步读取权重到 slot（带 EINTR 重试）
      auto pread_retry = [](int fd, void* dst, size_t bytes, off_t offset) -> bool {
        if (bytes == 0) return true;
        size_t done = 0;
        while (done < bytes) {
          ssize_t n = pread(fd, (char*)dst + done, bytes - done, offset + done);
          if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "[MESH] pread failed: errno=%d (%s), bytes=%zu offset=%ld\n",
                    errno, strerror(errno), bytes, (long)offset);
            return false;
          }
          if (n == 0) {
            fprintf(stderr, "[MESH] pread unexpected EOF: bytes=%zu done=%zu offset=%ld\n",
                    bytes, done, (long)offset);
            return false;
          }
          done += (size_t)n;
        }
        return true;
      };

      // 读取 gate 权重（单个 tp 切片）
      if (!pread_retry(layout.fd, gate_dst, layout.gate_bytes, layout.gate_offset)) {
        fprintf(stderr, "[MESH] load_expert_sync: gate pread failed, expert=%d layer=%d tp=%d\n",
                expert_id, layer, tp);
        return false;
      }
      // 读取 up 权重
      if (!pread_retry(layout.fd, up_dst, layout.up_bytes, layout.up_offset)) {
        fprintf(stderr, "[MESH] load_expert_sync: up pread failed, expert=%d layer=%d tp=%d\n",
                expert_id, layer, tp);
        return false;
      }
      // 读取 down 权重
      if (layout.down_stride > 0) {
        size_t row_bytes = layout.down_bytes / layout.down_rows;
        off_t src_off = layout.down_offset;
        for (int r = 0; r < layout.down_rows; r++) {
          if (!pread_retry(layout.fd, static_cast<char*>(down_dst) + r * row_bytes, row_bytes, src_off)) {
            fprintf(stderr, "[MESH] load_expert_sync: down row %d pread failed, expert=%d\n",
                    r, expert_id);
            return false;
          }
          src_off += layout.down_stride;
        }
      } else {
        if (!pread_retry(layout.fd, down_dst, layout.down_bytes, layout.down_offset)) {
          fprintf(stderr, "[MESH] load_expert_sync: down pread failed, expert=%d layer=%d tp=%d\n",
                  expert_id, layer, tp);
          return false;
        }
      }

      // AMXINT4: 从 scale cache 拷贝 scale 数据到 slot（单个 tp 切片）
      if (config_.weight_type == WeightType::AMXINT4 && io_->scale_cache_loaded(layer)) {
        io_->copy_scale_from_cache(layer, tp, expert_id,
                                   pool.gate_scale_ptr(slot_idx),
                                   pool.up_scale_ptr(slot_idx),
                                   pool.down_scale_ptr(slot_idx),
                                   nullptr, nullptr, nullptr,
                                   layouts_[layer][tp]);
      }

      // 所有读取成功，标记 slot 为 CACHED（原子操作，无需锁）
      pool.mark_cached(slot_idx);
      return true;
    }  // 外层 for loop
    return false;  // unreachable
  }

  // 计算 slot 字节数（gate + up + down 三个矩阵，AMXINT4 含 scale）
  // slot 存 TP 切分后的权重（intermediate_size / tp_count），
  // 与 AMX_MOE_BASE::config_.intermediate_size（被 TP_MOE_Common 切分）匹配。
  // GEMM 的 BufferB 用 n=intermediate_size/tp 计算 scale 偏移 = n*k/2，
  // slot 的 gate_up_weights_bytes 必须等于这个值，否则 scale 指针错位。
  size_t compute_slot_bytes(const MeshConfig& config) const {
    size_t gate_up_bytes = compute_gate_up_bytes(config);
    size_t down_bytes = compute_down_bytes(config);
    return gate_up_bytes * 2 + down_bytes;
  }

  // gate 或 up 单块字节数（权重 + scale）
  // BufferB 构造: n=intermediate/tp, k=hidden → 权重=n*k/2, scale=n*4
  // 除以 tp_count：与 do_gate_up_gemm 中 config_.intermediate_size（已被 TP 切分）匹配
  size_t compute_gate_up_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;  // TP 切分
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
    int i = config.intermediate_size / config.tp_count;  // TP 切分
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
    int i = config.intermediate_size / config.tp_count;  // TP 切分
    return static_cast<size_t>(h) * i / 2;
  }

  // AMXINT4: down 的纯权重大小（不含 scale）
  size_t compute_down_weights_bytes(const MeshConfig& config) const {
    int h = config.hidden_size;
    int i = config.intermediate_size / config.tp_count;  // TP 切分
    return static_cast<size_t>(i) * h / 2;
  }
};

}  // namespace mesh
