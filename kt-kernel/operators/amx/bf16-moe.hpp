/**
 * @Description  : BF16 AMX MoE operator for native BF16 inference
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-01-06
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements BF16 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * BF16 weights are stored without quantization (no scales).
 **/
#ifndef CPUINFER_OPERATOR_AMX_BF16_MOE_H
#define CPUINFER_OPERATOR_AMX_BF16_MOE_H

// #define DEBUG_BF16_MOE

#ifndef _WIN32
#include <numa.h>
#include <sys/mman.h>
#endif

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "la/amx_kernels.hpp"  // For vec_mul/mat_mul
#include "la/amx_raw_buffers.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/amx_utils.hpp"  // For transpose_16x16_32bit
#include "moe_base.hpp"

/**
 * @brief BF16 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to GemmKernel224BF16
 *
 * This class provides BF16-specific implementations:
 * - do_gate_up_gemm, do_down_gemm: BF16 weight mat mul (no quantization)
 * - load_weights: Load native BF16 weights (no scales)
 */
template <class T = amx::GemmKernel224BF16>
class AMX_BF16_MOE_TP : public AMX_MOE_BASE<T, AMX_BF16_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_BF16_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  static constexpr bool kUsesLazyMmapPacking = true;

#ifndef _WIN32
 private:
  enum ExpertState : uint8_t {
    EXPERT_BASELINE = 0,
    EXPERT_PACKING = 1,
    EXPERT_CACHED = 2,
    EXPERT_PINNED = 3,
    EXPERT_DEMOTING = 4,
  };

  std::unique_ptr<const ggml_bf16_t*[]> baseline_gate_src_;
  std::unique_ptr<const ggml_bf16_t*[]> baseline_up_src_;
  std::unique_ptr<const ggml_bf16_t*[]> baseline_down_src_;

  std::unique_ptr<void*[]> packed_gate_owner_;
  std::unique_ptr<void*[]> packed_up_owner_;
  std::unique_ptr<void*[]> packed_down_owner_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;

  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  ResidentCachePolicyState resident_policy_;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  int full_intermediate_size_ = 0;
#endif

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_BF16_MOE_TP() = default;

  AMX_BF16_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    this->derived_init();
  }

  void set_weight_buffers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  void derived_init() {
    // BF16 has no quantization, no need to check quant_config
    printf("Created AMX_BF16_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
#ifndef _WIN32
    if (config_.use_mmap) {
      initialize_lazy_mmap_state();
    }
#endif
  }

  ~AMX_BF16_MOE_TP() {
#ifndef _WIN32
    if (config_.use_mmap) {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        free_packed_expert(expert_id);
      }
    } else {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        std::free(gate_bb_[expert_id]->b);
        std::free(up_bb_[expert_id]->b);
        std::free(down_bb_[expert_id]->b);
        gate_bb_[expert_id]->b = nullptr;
        up_bb_[expert_id]->b = nullptr;
        down_bb_[expert_id]->b = nullptr;
      }
    }
#endif
  }

  // ============================================================================
  // CRTP buffer creation - without group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }

  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k);  // 2 parameters - no group_size
  }

  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }

  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);  // 2 parameters - no group_size
  }

  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Use vec_mul/mat_mul (no group_size)
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    }
  }

#ifndef _WIN32
  void initialize_lazy_mmap_state() {
    const int en = config_.expert_num;
    baseline_gate_src_ = std::make_unique<const ggml_bf16_t*[]>(en);
    std::fill_n(baseline_gate_src_.get(), en, nullptr);
    baseline_up_src_ = std::make_unique<const ggml_bf16_t*[]>(en);
    std::fill_n(baseline_up_src_.get(), en, nullptr);
    baseline_down_src_ = std::make_unique<const ggml_bf16_t*[]>(en);
    std::fill_n(baseline_down_src_.get(), en, nullptr);
    packed_gate_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_gate_owner_.get(), en, nullptr);
    packed_up_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_up_owner_.get(), en, nullptr);
    packed_down_owner_ = std::make_unique<void*[]>(en);
    std::fill_n(packed_down_owner_.get(), en, nullptr);
    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    resident_expert_count_.store(0, std::memory_order_relaxed);
    eviction_cursor_.store(0, std::memory_order_relaxed);
    gate_weight_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    up_weight_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    down_weight_bytes_ = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    const int configured_resident = config_.max_resident_experts > 0 ? config_.max_resident_experts : config_.max_tier0_experts;
    cache_capacity_ = configured_resident <= 0
                          ? 0
                          : std::min(config_.expert_num, std::max(configured_resident, config_.num_experts_per_tok));
    resident_policy_.reset(en, config_.resident_cache_policy);
    full_intermediate_size_ =
        config_.intermediate_size * std::max(1, config_.pool != nullptr ? config_.pool->config.subpool_count : 1);

    if (config_.pool != nullptr && tp_part_idx < (int)config_.pool->config.subpool_numa_map.size()) {
      numa_node_ = config_.pool->config.subpool_numa_map[tp_part_idx];
    }

    for (int expert_id = 0; expert_id < en; ++expert_id) {
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_relaxed);
      active_readers_[expert_id].store(0, std::memory_order_relaxed);
      std::free(gate_bb_[expert_id]->b);
      std::free(up_bb_[expert_id]->b);
      std::free(down_bb_[expert_id]->b);
      gate_bb_[expert_id]->b = nullptr;
      up_bb_[expert_id]->b = nullptr;
      down_bb_[expert_id]->b = nullptr;
    }
  }

  void set_mmap_source_ptrs(int expert_id,
                            const ggml_bf16_t* gate_src,
                            const ggml_bf16_t* up_src,
                            const ggml_bf16_t* down_src) override {
    baseline_gate_src_[expert_id] = gate_src;
    baseline_up_src_[expert_id] = up_src;
    baseline_down_src_[expert_id] = down_src;
    expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
  }

  void acquire_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_add(1, std::memory_order_acq_rel);
  }

  void release_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_sub(1, std::memory_order_acq_rel);
  }

  class ExpertReadScope {
   public:
    explicit ExpertReadScope(AMX_BF16_MOE_TP* owner, size_t reserve_count = 0) : owner_(owner) {
      experts_.reserve(reserve_count);
    }

    void add_expert(int expert_id) {
      owner_->acquire_expert_read(expert_id);
      experts_.push_back(expert_id);
    }

    ~ExpertReadScope() {
      for (int expert_id : experts_) {
        owner_->release_expert_read(expert_id);
      }
    }

   private:
    AMX_BF16_MOE_TP* owner_;
    std::vector<int> experts_;
  };

  void free_packed_expert(int expert_id) {
    void* gate_owner = packed_gate_owner_[expert_id];
    void* up_owner = packed_up_owner_[expert_id];
    void* down_owner = packed_down_owner_[expert_id];
    packed_gate_owner_[expert_id] = nullptr;
    packed_up_owner_[expert_id] = nullptr;
    packed_down_owner_[expert_id] = nullptr;
    gate_bb_[expert_id]->b = nullptr;
    up_bb_[expert_id]->b = nullptr;
    down_bb_[expert_id]->b = nullptr;

    if (gate_owner) {
      numa_free(gate_owner, gate_weight_bytes_);
    }
    if (up_owner) {
      numa_free(up_owner, up_weight_bytes_);
    }
    if (down_owner) {
      numa_free(down_owner, down_weight_bytes_);
    }
  }

  void drop_baseline_cache_for_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    if (baseline_gate_src_[expert_id] != nullptr) {
      madvise((void*)baseline_gate_src_[expert_id], gate_weight_bytes_, MADV_DONTNEED);
    }
    if (baseline_up_src_[expert_id] != nullptr) {
      madvise((void*)baseline_up_src_[expert_id], up_weight_bytes_, MADV_DONTNEED);
    }
    if (baseline_down_src_[expert_id] != nullptr) {
      madvise((void*)baseline_down_src_[expert_id],
              sizeof(ggml_bf16_t) * (size_t)config_.hidden_size * (size_t)full_intermediate_size_, MADV_DONTNEED);
    }
#else
    (void)expert_id;
#endif
  }

  void note_expert_access(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
    resident_policy_.note_access(expert_id, state == EXPERT_CACHED || state == EXPERT_PINNED);
  }

  void note_expert_insert(int expert_id, bool pinned) { resident_policy_.on_insert(expert_id, pinned); }

  void note_expert_pin(int expert_id) { resident_policy_.on_pin(expert_id); }

  void note_expert_demote(int expert_id) { resident_policy_.on_demote(expert_id); }

  int select_eviction_victim(int exclude_expert_id) {
    return resident_policy_.pick_victim(
        config_.expert_num, exclude_expert_id, static_cast<uint8_t>(EXPERT_CACHED),
        [this](int expert_id) { return expert_states_[expert_id].load(std::memory_order_acquire); },
        [this](int expert_id) { return active_readers_[expert_id].load(std::memory_order_acquire); });
  }

  bool evict_one_cached_expert(int exclude_expert_id) {
    if (config_.expert_num <= 1) return false;

    for (int attempt = 0; attempt < config_.expert_num; ++attempt) {
      const int victim = select_eviction_victim(exclude_expert_id);
      if (victim < 0) {
        return false;
      }
      if (victim == exclude_expert_id) {
        continue;
      }

      uint8_t expected = EXPERT_CACHED;
      if (!expert_states_[victim].compare_exchange_strong(
              expected, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }

      if (active_readers_[victim].load(std::memory_order_acquire) != 0) {
        expert_states_[victim].store(EXPERT_CACHED, std::memory_order_release);
        continue;
      }

      free_packed_expert(victim);
      note_expert_demote(victim);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[victim].store(EXPERT_BASELINE, std::memory_order_release);
      return true;
    }

    return false;
  }

  bool allocate_and_pack_expert(int expert_id, bool pin_after_pack) {
    const size_t resident_bytes_per_expert = gate_weight_bytes_ + up_weight_bytes_ + down_weight_bytes_;
    const size_t promotion_temp_bytes =
        sizeof(ggml_bf16_t) * (size_t)config_.hidden_size * (size_t)config_.intermediate_size;
    if (cache_capacity_ > 0) {
      while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        if (!evict_one_cached_expert(expert_id)) {
          break;
        }
      }
      if (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        int cached_count = 0;
        int pinned_count = 0;
        int packing_count = 0;
        int demoting_count = 0;
        int busy_count = 0;
        for (int i = 0; i < config_.expert_num; ++i) {
          const uint8_t state = expert_states_[i].load(std::memory_order_acquire);
          if (state == EXPERT_CACHED) {
            ++cached_count;
            if (active_readers_[i].load(std::memory_order_acquire) != 0) ++busy_count;
          } else if (state == EXPERT_PINNED) {
            ++pinned_count;
          } else if (state == EXPERT_PACKING) {
            ++packing_count;
          } else if (state == EXPERT_DEMOTING) {
            ++demoting_count;
          }
        }
        std::fprintf(stderr,
                     "[BF16_PROMOTION_FAIL] reason=cache_full expert=%d resident=%d capacity=%d cached=%d pinned=%d packing=%d demoting=%d busy_cached=%d resident_bytes_per_expert=%.3fMiB resident_total_est=%.3fGiB promotion_temp=%.3fMiB\n",
                     expert_id,
                     (int)resident_expert_count_.load(std::memory_order_acquire),
                     cache_capacity_,
                     cached_count,
                     pinned_count,
                     packing_count,
                     demoting_count,
                     busy_count,
                     (double)resident_bytes_per_expert / (1024.0 * 1024.0),
                     (double)(resident_bytes_per_expert * (size_t)resident_expert_count_.load(std::memory_order_acquire)) /
                         (1024.0 * 1024.0 * 1024.0),
                     (double)promotion_temp_bytes / (1024.0 * 1024.0));
        return false;
      }
    }

    void* gate_owner = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_weight_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      std::fprintf(stderr,
                   "[BF16_PROMOTION_FAIL] reason=numa_alloc expert=%d gate_bytes=%zu up_bytes=%zu down_bytes=%zu gate_ptr=%p up_ptr=%p down_ptr=%p numa=%d\n",
                   expert_id,
                   gate_weight_bytes_,
                   up_weight_bytes_,
                   down_weight_bytes_,
                   gate_owner,
                   up_owner,
                   down_owner,
                   numa_node_);
      if (gate_owner) numa_free(gate_owner, gate_weight_bytes_);
      if (up_owner) numa_free(up_owner, up_weight_bytes_);
      if (down_owner) numa_free(down_owner, down_weight_bytes_);
      return false;
    }

    if (baseline_gate_src_[expert_id] == nullptr || baseline_up_src_[expert_id] == nullptr ||
        baseline_down_src_[expert_id] == nullptr) {
      std::fprintf(stderr,
                   "[BF16_PROMOTION_FAIL] reason=missing_baseline expert=%d gate_src=%p up_src=%p down_src=%p\n",
                   expert_id,
                   baseline_gate_src_[expert_id],
                   baseline_up_src_[expert_id],
                   baseline_down_src_[expert_id]);
      numa_free(gate_owner, gate_weight_bytes_);
      numa_free(up_owner, up_weight_bytes_);
      numa_free(down_owner, down_weight_bytes_);
      return false;
    }

    madvise((void*)baseline_gate_src_[expert_id], gate_weight_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_up_src_[expert_id], up_weight_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_down_src_[expert_id],
            sizeof(ggml_bf16_t) * (size_t)config_.hidden_size * (size_t)full_intermediate_size_, MADV_WILLNEED);

    gate_bb_[expert_id]->set_data(gate_owner);
    up_bb_[expert_id]->set_data(up_owner);
    down_bb_[expert_id]->set_data(down_owner);

    // Tiered BF16 currently runs during early scheduler warmup/capture where nested
    // work-stealing dispatch has shown to be brittle. Pack the expert
    // synchronously here and keep parallelism in the hot GEMM path.
    gate_bb_[expert_id]->from_mat(const_cast<ggml_bf16_t*>(baseline_gate_src_[expert_id]), 0, 1);
    up_bb_[expert_id]->from_mat(const_cast<ggml_bf16_t*>(baseline_up_src_[expert_id]), 0, 1);

    std::vector<ggml_bf16_t> down_contiguous((size_t)config_.hidden_size * (size_t)config_.intermediate_size);
    const size_t down_col_offset = (size_t)tp_part_idx * (size_t)config_.intermediate_size;
    for (int row = 0; row < config_.hidden_size; ++row) {
      const ggml_bf16_t* src_row = baseline_down_src_[expert_id] + (size_t)row * (size_t)full_intermediate_size_ + down_col_offset;
      ggml_bf16_t* dst_row = down_contiguous.data() + (size_t)row * (size_t)config_.intermediate_size;
      std::memcpy(dst_row, src_row, sizeof(ggml_bf16_t) * (size_t)config_.intermediate_size);
    }
    down_bb_[expert_id]->from_mat(down_contiguous.data(), 0, 1);

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_pack ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_pack);
    const char* trace = std::getenv("KT_TRACE_CPU_MEM_BREAKDOWN");
    if (trace != nullptr &&
        (std::strcmp(trace, "1") == 0 || std::strcmp(trace, "true") == 0 || std::strcmp(trace, "TRUE") == 0)) {
      std::fprintf(stderr,
                   "[KT_CPU_MEM_RESIDENT] layer=%d tp=%d expert=%d resident_count=%d capacity=%d resident_bytes_per_expert=%.3fMiB resident_total_est=%.3fGiB promotion_temp=%.3fMiB state=%s\n",
                   config_.layer_idx,
                   tp_part_idx,
                   expert_id,
                   (int)resident_expert_count_.load(std::memory_order_acquire),
                   cache_capacity_,
                   (double)resident_bytes_per_expert / (1024.0 * 1024.0),
                   (double)(resident_bytes_per_expert * (size_t)resident_expert_count_.load(std::memory_order_acquire)) /
                       (1024.0 * 1024.0 * 1024.0),
                   (double)promotion_temp_bytes / (1024.0 * 1024.0),
                   pin_after_pack ? "PINNED" : "CACHED");
    }
    // Once the NUMA-local packed copy is ready, immediately drop the
    // file-backed baseline pages for this expert instead of waiting for
    // global pressure reclaim. The mmap mapping remains valid and can fault
    // back in later if the resident copy is demoted.
    drop_baseline_cache_for_expert(expert_id);
    return true;
  }

  void ensure_expert_ready(int expert_id, bool pin = false) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        return;
      }
      if (state == EXPERT_CACHED) {
        if (!pin) {
          return;
        }
        uint8_t expected = EXPERT_CACHED;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PINNED, std::memory_order_acq_rel, std::memory_order_acquire)) {
          note_expert_pin(expert_id);
          drop_baseline_cache_for_expert(expert_id);
          return;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
        uint8_t expected = EXPERT_BASELINE;
        if (!expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        if (!allocate_and_pack_expert(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("BF16 lazy-pack promotion failed");
        }
        continue;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        std::this_thread::yield();
        continue;
      }
    }
  }
#endif

#ifdef DEBUG_BF16_MOE
  // Function to dump Buffer B data for debugging
  inline void dump_buffer_b(int expert_idx, const std::string& matrix_type, typename T::BufferB* buffer) {
    printf("[DUMP_BUFFER_B] TP%d BF16 Expert%d %s:\n", tp_part_idx, expert_idx, matrix_type.c_str());

    // Calculate dimensions based on matrix type
    int rows, cols;
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
    }

    // Dump BF16 weights
    size_t weight_size = (size_t)rows * cols;
    ggml_bf16_t* weight_ptr = buffer->b;

    printf("  BF16 Weights[first 16]: ");
    for (int i = 0; i < std::min(16, (int)weight_size); i++) {
      printf("%.6f ", ggml_bf16_to_fp32(weight_ptr[i]));
    }
    printf("\n");

    if (weight_size > 16) {
      printf("  BF16 Weights[last 16]: ");
      int start_idx = std::max(0, (int)weight_size - 16);
      for (int i = start_idx; i < (int)weight_size; i++) {
        printf("%.6f ", ggml_bf16_to_fp32(weight_ptr[i]));
      }
      printf("\n");
    }

    printf("  Matrix dimensions: %dx%d (n x k)\n", rows, cols);
  }
#endif

  /**
   * @brief Load BF16 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj (no scales).
   */
  void load_weights() {
    if (config_.use_mmap) {
      const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
      const bool use_per_expert_ptrs = !config_.gate_projs.empty();
      const size_t tp_weight_elems = (size_t)config_.intermediate_size * config_.hidden_size;
      const size_t gate_up_weight_src_offset = (size_t)tp_part_idx * tp_weight_elems;
      const size_t full_weight_elems = (size_t)full_intermediate_size_ * (size_t)config_.hidden_size;

      for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
        const size_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
        const ggml_bf16_t* gate_src;
        const ggml_bf16_t* up_src;
        const ggml_bf16_t* down_src;

        if (use_per_expert_ptrs) {
          gate_src = (const ggml_bf16_t*)config_.gate_projs[0][logical_expert_id] + gate_up_weight_src_offset;
          up_src = (const ggml_bf16_t*)config_.up_projs[0][logical_expert_id] + gate_up_weight_src_offset;
          down_src = (const ggml_bf16_t*)config_.down_projs[0][logical_expert_id];
        } else {
          gate_src = (const ggml_bf16_t*)config_.gate_proj + logical_expert_id * full_weight_elems +
                     gate_up_weight_src_offset;
          up_src = (const ggml_bf16_t*)config_.up_proj + logical_expert_id * full_weight_elems + gate_up_weight_src_offset;
          down_src = (const ggml_bf16_t*)config_.down_proj + logical_expert_id * full_weight_elems;
        }

        set_mmap_source_ptrs(expert_idx, gate_src, up_src, down_src);
      }
      return;
    }

    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_proj == nullptr) {
      throw std::runtime_error("BF16 MOE requires native BF16 weight.");
    }

    // Load gate + up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Gate: from BF16 data (no scale)
          gate_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.gate_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);  // 3 parameters: (bf16*, ith, nth)

          // Up: same
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.up_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Down
          down_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.down_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);
        },
        nullptr);

#ifdef DEBUG_BF16_MOE
    dump_buffer_b(0, "gate", gate_bb_[0].get());
    dump_buffer_b(0, "down", down_bb_[0].get());
#endif
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
               void* output) override {
#ifndef _WIN32
    std::vector<int> active_experts;
    std::vector<int> expert_counts;
    if (config_.use_mmap) {
      active_experts.reserve(qlen * k);
      expert_counts.assign(config_.expert_num, 0);
      std::vector<uint8_t> seen(config_.expert_num, 0);

      for (int i = 0; i < qlen * k; ++i) {
        const int expert_id = (int)expert_ids[i];
        if (config_.should_skip_expert(expert_id)) {
          continue;
        }
        ++expert_counts[expert_id];
        if (!seen[expert_id]) {
          seen[expert_id] = 1;
          active_experts.push_back(expert_id);
        }
      }

      const bool enable_wave_mode = []() {
        const char* raw = std::getenv("KT_ENABLE_BF16_WAVE_RESIDENT");
        if (raw == nullptr) return false;
        return std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 || std::strcmp(raw, "TRUE") == 0;
      }();
      const int wave_capacity = cache_capacity_ > 0 ? cache_capacity_ : (int)active_experts.size();
      if (enable_wave_mode && qlen > 1 && wave_capacity > 0 && (int)active_experts.size() > wave_capacity) {
        auto resident_rank = [this](int expert_id) {
          const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
          if (state == EXPERT_PINNED) return 2;
          if (state == EXPERT_CACHED) return 1;
          return 0;
        };
        std::stable_sort(active_experts.begin(), active_experts.end(), [&](int lhs, int rhs) {
          const int lhs_rank = resident_rank(lhs);
          const int rhs_rank = resident_rank(rhs);
          if (lhs_rank != rhs_rank) {
            return lhs_rank > rhs_rank;
          }
          if (expert_counts[lhs] != expert_counts[rhs]) {
            return expert_counts[lhs] > expert_counts[rhs];
          }
          return lhs < rhs;
        });

        const size_t output_elems = (size_t)qlen * (size_t)config_.hidden_size;
        auto free_aligned_f32 = [](void* ptr) {
#ifdef _WIN32
          _aligned_free(ptr);
#else
          std::free(ptr);
#endif
        };
        auto alloc_aligned_f32 = [&](size_t elem_count) {
          void* ptr = nullptr;
#ifdef _WIN32
          ptr = _aligned_malloc(sizeof(float) * elem_count, 64);
          if (ptr == nullptr) {
            throw std::bad_alloc();
          }
#else
          if (posix_memalign(&ptr, 64, sizeof(float) * elem_count) != 0 || ptr == nullptr) {
            throw std::bad_alloc();
          }
#endif
          return std::unique_ptr<float, decltype(free_aligned_f32)>(static_cast<float*>(ptr), free_aligned_f32);
        };

        auto accum_output_holder = alloc_aligned_f32(output_elems);
        auto wave_output_holder = alloc_aligned_f32(output_elems);
        float* accum_output = accum_output_holder.get();
        float* wave_output = wave_output_holder.get();
        std::fill(accum_output, accum_output + output_elems, 0.0f);
        std::vector<int64_t> wave_expert_ids((size_t)qlen * (size_t)k, -1);
        std::vector<uint8_t> in_wave(config_.expert_num, 0);

        for (int wave_begin = 0; wave_begin < (int)active_experts.size(); wave_begin += wave_capacity) {
          const int wave_end = std::min<int>(wave_begin + wave_capacity, (int)active_experts.size());
          std::fill(in_wave.begin(), in_wave.end(), 0);
          for (int idx = wave_begin; idx < wave_end; ++idx) {
            in_wave[active_experts[idx]] = 1;
          }

          std::fill(wave_expert_ids.begin(), wave_expert_ids.end(), static_cast<int64_t>(-1));
          for (int i = 0; i < qlen * k; ++i) {
            const int expert_id = (int)expert_ids[i];
            if (!config_.should_skip_expert(expert_id) && in_wave[expert_id]) {
              wave_expert_ids[i] = expert_id;
            }
          }

          auto expert_read_scope = std::make_unique<ExpertReadScope>(this, wave_end - wave_begin);
          for (int idx = wave_begin; idx < wave_end; ++idx) {
            const int expert_id = active_experts[idx];
            note_expert_access(expert_id);
            ensure_expert_ready(expert_id, false);
            expert_read_scope->add_expert(expert_id);
          }

          std::fill(wave_output, wave_output + output_elems, 0.0f);
          Base::forward(qlen, k, wave_expert_ids.data(), weights, input, wave_output);
          for (size_t i = 0; i < output_elems; ++i) {
            accum_output[i] += wave_output[i];
          }

          if (config_.max_tier0_experts <= 0) {
            expert_read_scope.reset();
            for (int idx = wave_begin; idx < wave_end; ++idx) {
              demote_expert(active_experts[idx]);
            }
          }
        }

        std::memcpy(output, accum_output, sizeof(float) * output_elems);
        return;
      }

      auto expert_read_scope = std::make_unique<ExpertReadScope>(this, active_experts.size());
      for (int expert_id : active_experts) {
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope->add_expert(expert_id);
      }

      Base::forward(qlen, k, expert_ids, weights, input, output);
      if (config_.max_tier0_experts <= 0 && !active_experts.empty()) {
        // With zero Tier0 budget, keep mmap-backed experts only for the current
        // forward pass and drop them immediately afterwards instead of retaining
        // a cross-token resident cache.
        expert_read_scope.reset();
        for (int expert_id : active_experts) {
          demote_expert(expert_id);
        }
      }
      return;
    }
#endif
    Base::forward(qlen, k, expert_ids, weights, input, output);
  }

  void promote_expert(int expert_id) override {
#ifndef _WIN32
    ensure_expert_ready(expert_id, true);
#else
    (void)expert_id;
#endif
  }

  void demote_expert(int expert_id) override {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_BASELINE) {
        return;
      }
      if (state == EXPERT_PACKING) {
        std::this_thread::yield();
        continue;
      }
      if (state != EXPERT_CACHED && state != EXPERT_PINNED) {
        return;
      }

      uint8_t expected = state;
      if (!expert_states_[expert_id].compare_exchange_strong(
              expected, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }

      while (active_readers_[expert_id].load(std::memory_order_acquire) != 0) {
        std::this_thread::yield();
      }

      free_packed_expert(expert_id);
      note_expert_demote(expert_id);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return;
    }
#else
    (void)expert_id;
#endif
  }

  bool is_expert_promoted(int expert_id) const override {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  // Fast 64-byte (512-bit) memcpy using AVX512
  static inline void fast_memcpy_64(void* __restrict dst, const void* __restrict src) {
    __m512i data = _mm512_loadu_si512(src);
    _mm512_storeu_si512(dst, data);
  }

  // Fast 64-byte non-temporal store (bypass cache for write-only patterns)
  static inline void fast_stream_64(void* __restrict dst, const void* __restrict src) {
    __m512i data = _mm512_loadu_si512(src);
    _mm512_stream_si512((__m512i*)dst, data);
  }

  // Fast memcpy for arbitrary sizes using AVX512
  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      fast_memcpy_64(d, s);
      d += 64;
      s += 64;
    }
    bytes -= chunks * 64;
    if (bytes > 0) {
      std::memcpy(d, s, bytes);
    }
  }

  /**
   * @brief Unpack a single N_STEP x K_STEP block from packed BufferB format to n-major format (BF16 version)
   *
   * This is the inverse of the packing done in BufferBBF16Impl::from_mat.
   * BF16 elements are 2 bytes, and the packed format includes 16x16 32-bit transpose.
   *
   * @param src Pointer to packed data (N_STEP * K_STEP * 2 bytes in packed layout)
   * @param dst Pointer to destination in n-major layout
   * @param dst_row_stride Row stride in destination buffer (number of BF16 elements per row)
   */
  static inline void unpack_nk_block_bf16(const ggml_bf16_t* src, ggml_bf16_t* dst, size_t dst_row_stride) {
    constexpr int N_STEP = T::N_STEP;  // 32
    constexpr int K_STEP = T::K_STEP;  // 32
    constexpr int TILE_N = T::TILE_N;  // 16

    // The packed format has two 16x16 blocks (32-bit view) that were transposed
    // We need to reverse the transpose first, then copy to n-major layout

    // Create aligned temporary buffers for transpose
    alignas(64) __m512i temp_block1[TILE_N];
    alignas(64) __m512i temp_block2[TILE_N];

    // Copy source data to temporary buffers
    const __m512i* src_vec = reinterpret_cast<const __m512i*>(src);
    for (int i = 0; i < TILE_N; i++) {
      temp_block1[i] = src_vec[i];
      temp_block2[i] = src_vec[TILE_N + i];
    }

    // Reverse transpose (transpose is self-inverse)
    amx::transpose_16x16_32bit(temp_block1);
    amx::transpose_16x16_32bit(temp_block2);

    // Copy transposed data to destination in n-major layout using non-temporal stores
    // First 16 rows (block 1)
    for (int i = 0; i < TILE_N; i++) {
      fast_stream_64(dst + i * dst_row_stride, &temp_block1[i]);
    }

    // Next 16 rows (block 2)
    for (int i = 0; i < TILE_N; i++) {
      fast_stream_64(dst + (TILE_N + i) * dst_row_stride, &temp_block2[i]);
    }

    // Ensure all stores complete before returning
    _mm_sfence();
  }

  /**
   * @brief Reconstruct weights for a single expert to the output buffers
   *
   * Directly unpacks from packed BufferB format to n-major GPU buffers without intermediate storage.
   * BF16 version - no scales needed.
   *
   * @param gpu_tp_count Number of GPU TP parts (1, 2, 4, or 8)
   * @param cpu_tp_count Number of CPU TP parts
   * @param expert_id Expert index to process
   * @param full_config Full configuration (before CPU TP split)
   * @param w13_weight_ptrs Pointers to gate+up weight buffers (one per GPU TP)
   * @param w13_scale_ptrs Pointers to gate+up scale buffers (unused for BF16, kept for interface compatibility)
   * @param w2_weight_ptrs Pointers to down weight buffers (one per GPU TP)
   * @param w2_scale_ptrs Pointers to down scale buffers (unused for BF16, kept for interface compatibility)
   */
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);

    constexpr int N_STEP = T::N_STEP;
    constexpr int K_STEP = T::K_STEP;
    constexpr int N_BLOCK = T::N_BLOCK;
    constexpr int K_BLOCK = T::K_BLOCK;

    // ========= W13 (gate+up): Shape [intermediate, hidden], split by N only =========
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;

    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;

    // ========= W2 (down): Shape [hidden, intermediate], split by K =========
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_n_w2 = full_config.hidden_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;

    // ========= Optimized job layout =========
    constexpr int NUM_W13_TASKS = 32;  // Per matrix (gate or up), total 64 for w13
    constexpr int NUM_W2_TASKS = 32;   // For down matrix

    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    // Calculate N_STEP blocks per task
    const int w13_n_steps = div_up(cpu_n_w13, N_STEP);
    const int w13_steps_per_task = div_up(w13_n_steps, NUM_W13_TASKS);
    const int w2_n_steps = div_up(cpu_n_w2, N_STEP);
    const int w2_steps_per_task = div_up(w2_n_steps, NUM_W2_TASKS);

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w2_weight_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // ========= W13 weight task: process chunk of rows x full K =========
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int step_start = chunk_idx * w13_steps_per_task;
            const int step_end = std::min(step_start + w13_steps_per_task, w13_n_steps);
            if (step_start >= w13_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w13);

            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              const int global_n = global_n_offset_w13 + local_n_start;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              ggml_bf16_t* weight_base = (ggml_bf16_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_weight_off = is_up ? gpu_w13_weight_per_mat : 0;

              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w13 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              for (int k_block_begin = 0; k_block_begin < cpu_k_w13; k_block_begin += K_BLOCK) {
                const int k_block_size = std::min(K_BLOCK, cpu_k_w13 - k_block_begin);

                for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
                  const ggml_bf16_t* src = bb->b + (size_t)n_block_begin * cpu_k_w13 +
                                           (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                           (size_t)k_begin * N_STEP;
                  ggml_bf16_t* dst =
                      weight_base + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13 + k_block_begin + k_begin;
                  unpack_nk_block_bf16(src, dst, gpu_k_w13);
                }
              }
            }

          } else {
            // ========= W2 weight task: process chunk of rows x all K slices =========
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int step_start = chunk_idx * w2_steps_per_task;
            const int step_end = std::min(step_start + w2_steps_per_task, w2_n_steps);
            if (step_start >= w2_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w2);

            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w2 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              for (int k_slice_start = 0; k_slice_start < cpu_k_w2; k_slice_start += gpu_k_w2) {
                const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

                const int global_k_start = global_k_offset_w2 + k_slice_start;
                const int target_gpu = global_k_start / gpu_k_w2;
                const int k_in_gpu_base = global_k_start % gpu_k_w2;

                ggml_bf16_t* weight_base = (ggml_bf16_t*)w2_weight_ptrs[target_gpu];

                for (int k_abs = k_slice_start; k_abs < k_slice_end; k_abs += K_STEP) {
                  const int k_block_idx = k_abs / K_BLOCK;
                  const int k_block_begin = k_block_idx * K_BLOCK;
                  const int k_block_size = std::min(K_BLOCK, cpu_k_w2 - k_block_begin);
                  const int k_in_block = k_abs - k_block_begin;
                  const int k_in_gpu = k_in_gpu_base + (k_abs - k_slice_start);

                  const ggml_bf16_t* src = bb->b + (size_t)n_block_begin * cpu_k_w2 +
                                           (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                           (size_t)k_in_block * N_STEP;
                  ggml_bf16_t* dst = weight_base + (size_t)local_n_start * gpu_k_w2 + k_in_gpu;
                  unpack_nk_block_bf16(src, dst, gpu_k_w2);
                }
              }
            }
          }
        },
        nullptr);
  }
};

template <typename K>
class TP_MOE<AMX_BF16_MOE_TP<K>> : public TP_MOE_Common<AMX_BF16_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AMX_BF16_MOE_TP<K>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    // BF16 has no quantization check needed
    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }

    const bool use_per_expert_ptrs = !config.gate_projs.empty();
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;

    if (config.use_mmap) {
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
    } else {
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;

        // Allocate BF16 weights (2 bytes/element)
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];

        const size_t gate_up_weight_src_offset = (size_t)i * tp_weight_elems;
        const size_t down_weight_src_col_offset = (size_t)i * (size_t)tpc.intermediate_size;

        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&](int expert_id) {
              const size_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);

              ggml_bf16_t* gate_dst = (ggml_bf16_t*)tpc.gate_proj + expert_id * tp_weight_elems;
              ggml_bf16_t* up_dst = (ggml_bf16_t*)tpc.up_proj + expert_id * tp_weight_elems;
              ggml_bf16_t* down_dst = (ggml_bf16_t*)tpc.down_proj + expert_id * tp_weight_elems;

              const ggml_bf16_t* gate_src;
              const ggml_bf16_t* up_src;
              const ggml_bf16_t* down_src;

              if (use_per_expert_ptrs) {
                gate_src = (const ggml_bf16_t*)config.gate_projs[0][logical_expert_id] + gate_up_weight_src_offset;
                up_src = (const ggml_bf16_t*)config.up_projs[0][logical_expert_id] + gate_up_weight_src_offset;
                down_src = (const ggml_bf16_t*)config.down_projs[0][logical_expert_id];
              } else {
                gate_src = (const ggml_bf16_t*)config.gate_proj + logical_expert_id * full_weight_elems +
                           gate_up_weight_src_offset;
                up_src =
                    (const ggml_bf16_t*)config.up_proj + logical_expert_id * full_weight_elems + gate_up_weight_src_offset;
                down_src = (const ggml_bf16_t*)config.down_proj + logical_expert_id * full_weight_elems;
              }

              std::memcpy(gate_dst, gate_src, tp_weight_elems * sizeof(ggml_bf16_t));
              std::memcpy(up_dst, up_src, tp_weight_elems * sizeof(ggml_bf16_t));

              for (int row = 0; row < config.hidden_size; row++) {
                const size_t src_row_offset =
                    (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
                const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
                std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset,
                            (size_t)tpc.intermediate_size * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      });

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        delete[] (ggml_bf16_t*)tpc.gate_proj;
        delete[] (ggml_bf16_t*)tpc.up_proj;
        delete[] (ggml_bf16_t*)tpc.down_proj;
      });
    }

    this->weights_loaded = true;
  }

  /**
   * @brief Write weights to GPU buffer for all TP parts
   *
   * BF16 version - no scales needed, scale_ptrs parameters are kept for interface compatibility.
   */
  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Weight pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }

  void promote_expert(int expert_id) {
    if (this->tp_count <= 1) {
      this->tps[0]->promote_expert(expert_id);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, expert_id](int tp_id) { this->tps[tp_id]->promote_expert(expert_id); });
  }

  void demote_expert(int expert_id) {
    for (int i = 0; i < this->tp_count; ++i) {
      this->tps[i]->demote_expert(expert_id);
    }
  }

  bool is_expert_promoted(int expert_id) const {
    return !this->tps.empty() && this->tps[0]->is_expert_promoted(expert_id);
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };

    auto pool = config.pool;
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_BF16_MOE_H
