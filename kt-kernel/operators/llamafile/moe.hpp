#ifndef LLAMAFILE_MOE_HPP
#define LLAMAFILE_MOE_HPP
#ifdef FORWARD_TIME_PROFILE
#include <fmt/format.h>
#endif
#ifndef _WIN32
#include <numa.h>
#include <numaif.h>
#endif

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../moe-tp.hpp"
#include "conversion.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

// Bind mmap pages to a specific NUMA node using mbind().
// This ensures mmap'd weight pages are NUMA-local for the TP instance that uses them.
// Without this, pages land on whichever NUMA node first faults them (first-touch policy),
// which is typically the Python main thread's node — wrong for TP instances on other nodes.
//
// Uses MPOL_MF_MOVE to migrate already-faulted pages (e.g., pages touched by GGUFLoader
// during parsing) to the target node. Pages not yet faulted will be placed on the
// target node on first access.
static inline void bind_pages_to_numa([[maybe_unused]] void* addr,
                                      [[maybe_unused]] size_t len,
                                      [[maybe_unused]] int node) {
#if defined(__linux__)
  const char* disable_numa_binding = std::getenv("KT_DISABLE_NUMA_BINDING");
  if (disable_numa_binding && disable_numa_binding[0] == '1') return;
  if (len == 0 || addr == nullptr || node < 0) return;
  static const size_t PAGE_SIZE = (size_t)sysconf(_SC_PAGESIZE);

  // mbind requires page-aligned address and length
  uintptr_t start = (uintptr_t)addr & ~(PAGE_SIZE - 1);
  uintptr_t end = ((uintptr_t)addr + len + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
  size_t page_len = end - start;

  const size_t bits_per_ulong = sizeof(unsigned long) * 8;
  std::vector<unsigned long> nodemask((size_t)node / bits_per_ulong + 1, 0);
  nodemask[(size_t)node / bits_per_ulong] |= 1UL << (node % bits_per_ulong);
  // max_node must be greater than the highest NUMA node bit we touch.
  unsigned long max_node = node + 1;

  long ret = mbind((void*)start, page_len, MPOL_BIND, nodemask.data(), max_node, MPOL_MF_MOVE);
  if (ret != 0 && errno != EPERM) {
    // EPERM is expected when some pages are locked; silently ignore.
    // Other errors (ENOMEM, EIO) are logged but non-fatal.
    printf("[bind_pages_to_numa] mbind failed for node %d: %s (addr=%p, len=%zu)\n",
           node, strerror(errno), addr, len);
  }
#endif
}

inline void debug_quant(void* input, ggml_type type) {
  std::vector<float> output(ggml_blck_size(type));
  to_float(input, output.data(), ggml_blck_size(type), type);
  for (size_t i = 0; i < 10; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
}

class LLAMA_MOE_TP {
 public:
  int get_numa_node() const { return numa_node_; }

 private:
  GeneralMOEConfig config_;
  int tp_part_idx;

  // Per-expert weight pointers (live pointers used in forward, swappable for tiered cache)
  std::unique_ptr<std::atomic<uint8_t*>[]> gate_expert_ptrs_;  // [expert_num]
  std::unique_ptr<std::atomic<uint8_t*>[]> up_expert_ptrs_;    // [expert_num]
  std::unique_ptr<std::atomic<uint8_t*>[]> down_expert_ptrs_;  // [expert_num]

  // Baseline pointers (mmap or legacy storage) for demotion fallback
  std::vector<uint8_t*> baseline_gate_ptrs_; // [expert_num]
  std::vector<uint8_t*> baseline_up_ptrs_;   // [expert_num]
  std::vector<uint8_t*> baseline_down_ptrs_; // [expert_num]

  // Tier 0 NUMA-local buffers (non-null when promoted)
  std::unique_ptr<std::atomic<uint8_t*>[]> tier0_gate_;        // [expert_num]
  std::unique_ptr<std::atomic<uint8_t*>[]> tier0_up_;          // [expert_num]
  std::unique_ptr<std::atomic<uint8_t*>[]> tier0_down_;        // [expert_num]
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;    // [expert_num]
  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;      // [expert_num]

  enum : uint8_t {
    EXPERT_BASELINE = 0,
    EXPERT_PROMOTING = 1,
    EXPERT_PROMOTED = 2,
    EXPERT_DEMOTING = 3,
  };

  // Backing storage ownership (delete[] in destructor)
  uint8_t* gate_storage_ = nullptr;
  uint8_t* up_storage_ = nullptr;
  uint8_t* down_storage_ = nullptr;

  // Per-expert byte sizes (for promote/demote allocation)
  size_t gate_expert_bytes_ = 0;
  size_t up_expert_bytes_ = 0;
  size_t down_expert_bytes_ = 0;
  int numa_node_ = 0;
  bool down_baseline_is_mmap_ = false;

  float* s_input_fp32_;    // [hidden_size]
  uint8_t* s_gate_input_;  // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* s_up_input_;    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
  std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
  std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
  std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size *
                                             // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                             // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
  float* s_output_fp32_;                     // [hidden_size]

  std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
  std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size *
                                        // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                        // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  std::vector<uint8_t*>
      m_up_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type)
                    // / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  uint8_t* m_local_gate_input_;        // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* m_local_up_input_;          // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  float* m_local_gate_output_;         // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_up_output_;           // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_intermediate_fp32_;   // [routed_expert_num * group_max_len * intermediate_size]
  uint8_t* m_local_down_input_;        // [routed_expert_num * group_max_len * intermediate_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  float* m_local_down_output_;         // [routed_expert_num * group_max_len * hidden_size]
  std::vector<float*> m_output_fp32_;  // [group_max_len, hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
  std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
  std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
  std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
  std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
  std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
  std::vector<float*> m_local_down_output_ptr_;        // [expert_num]

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
    template <typename ExpertIdT>
    ExpertReadScope(LLAMA_MOE_TP* owner, const ExpertIdT* expert_ids, int count) : owner_(owner) {
      experts_.reserve(count);
      for (int i = 0; i < count; ++i) {
        int expert_id = (int)expert_ids[i];
        if (expert_id < 0) continue;
        experts_.push_back(expert_id);
        owner_->acquire_expert_read(expert_id);
      }
    }

    ~ExpertReadScope() {
      for (int expert_id : experts_) {
        owner_->release_expert_read(expert_id);
      }
    }

   private:
    LLAMA_MOE_TP* owner_;
    std::vector<int> experts_;
  };
 public:
  using input_t = ggml_bf16_t;
  using output_t = float;

  LLAMA_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : config_(config), tp_part_idx(tp_part_idx) {
    MemoryRequest mem_requests;
    mem_requests.append_pointer(&s_input_fp32_, sizeof(float) * config_.hidden_size);
    mem_requests.append_pointer(
        &s_gate_input_, config_.hidden_size *
                            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
    mem_requests.append_pointer(
        &s_up_input_, config_.hidden_size *
                          ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                          ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    s_gate_output_.resize(config_.num_experts_per_tok);
    s_up_output_.resize(config_.num_experts_per_tok);
    s_intermediate_fp32_.resize(config_.num_experts_per_tok);
    s_down_input_.resize(config_.num_experts_per_tok);
    s_down_output_.resize(config_.num_experts_per_tok);
    for (int i = 0; i < config_.num_experts_per_tok; i++) {
      mem_requests.append_pointer(&s_gate_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_up_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(
          &s_down_input_[i],
          config_.intermediate_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type));
      mem_requests.append_pointer(&s_down_output_[i], sizeof(float) * config_.hidden_size);
    }
    mem_requests.append_pointer(&s_output_fp32_, sizeof(float) * config_.hidden_size);
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, mem_requests);

    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_input_fp32_[i], sizeof(float) * config_.hidden_size);
      mem_requests.append_pointer(
          &m_gate_input_[i],
          config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
      mem_requests.append_pointer(
          &m_up_input_[i], config_.hidden_size *
                               ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                               ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    }
    mem_requests.append_pointer(
        &m_local_gate_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
    mem_requests.append_pointer(
        &m_local_up_input_, config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
                                ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                                ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(float) * config_.num_experts_per_tok *
                                                         config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_intermediate_fp32_, sizeof(float) * config_.num_experts_per_tok *
                                                                 config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(
        &m_local_down_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.intermediate_size *
            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type));
    mem_requests.append_pointer(&m_local_down_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.hidden_size);
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_output_fp32_[i], sizeof(float) * config_.hidden_size);
    }
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    // Initialize per-expert pointer vectors
    int en = config.expert_num;
    gate_expert_ptrs_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    up_expert_ptrs_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    down_expert_ptrs_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    baseline_gate_ptrs_.resize(en, nullptr);
    baseline_up_ptrs_.resize(en, nullptr);
    baseline_down_ptrs_.resize(en, nullptr);
    tier0_gate_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    tier0_up_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    tier0_down_ = std::make_unique<std::atomic<uint8_t*>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    for (int i = 0; i < en; ++i) {
      gate_expert_ptrs_[i].store(nullptr, std::memory_order_relaxed);
      up_expert_ptrs_[i].store(nullptr, std::memory_order_relaxed);
      down_expert_ptrs_[i].store(nullptr, std::memory_order_relaxed);
      tier0_gate_[i].store(nullptr, std::memory_order_relaxed);
      tier0_up_[i].store(nullptr, std::memory_order_relaxed);
      tier0_down_[i].store(nullptr, std::memory_order_relaxed);
      active_readers_[i].store(0, std::memory_order_relaxed);
      expert_states_[i].store(EXPERT_BASELINE, std::memory_order_relaxed);
    }

    // Compute per-expert byte sizes
    auto gr = ggml_type_size((ggml_type)config.gate_type) / ggml_blck_size((ggml_type)config.gate_type);
    auto ur = ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type);
    auto dr = ggml_type_size((ggml_type)config.down_type) / ggml_blck_size((ggml_type)config.down_type);
    gate_expert_bytes_ = (size_t)config.intermediate_size * config.hidden_size * gr;
    up_expert_bytes_ = (size_t)config.intermediate_size * config.hidden_size * ur;
    down_expert_bytes_ = (size_t)config.hidden_size * config.intermediate_size * dr;
    if (config.pool != nullptr && tp_part_idx >= 0 &&
        tp_part_idx < (int)config.pool->config.subpool_numa_map.size()) {
      numa_node_ = config.pool->config.subpool_numa_map[tp_part_idx];
    } else {
      numa_node_ = tp_part_idx;
    }

    if (!config.use_mmap) {
      // Legacy mode: allocate flat storage buffers
      gate_storage_ = new uint8_t[(size_t)en * gate_expert_bytes_];
      up_storage_ = new uint8_t[(size_t)en * up_expert_bytes_];
      down_storage_ = new uint8_t[(size_t)en * down_expert_bytes_];
    }
    // Per-expert pointers are populated in load_weights()
  }

  void load_weights(int complete_intermediate_size, int offset) {
    auto& config = config_;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    static bool printed_types = false;
    if (!printed_types) {
      printed_types = true;
      printf(
          "[LLAMA_TYPES] gate_type=%d gate_vec=%d up_type=%d up_vec=%d down_type=%d down_vec=%d hidden_type=%d\n",
          (int)config.gate_type,
          (int)ggml_internal_get_type_traits((ggml_type)config.gate_type).vec_dot_type,
          (int)config.up_type,
          (int)ggml_internal_get_type_traits((ggml_type)config.up_type).vec_dot_type,
          (int)config.down_type,
          (int)ggml_internal_get_type_traits((ggml_type)config.down_type).vec_dot_type,
          (int)config.hidden_type
      );
    }
    // we need to make sure the blck size is correct for size.
    if (config.intermediate_size % ggml_blck_size((ggml_type)config.down_type) != 0) {
      printf("intermediate_size: %d, down_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.down_type));
      throw std::runtime_error("intermediate_size must be a multiple of gate_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.up_type) != 0) {
      printf("intermediate_size: %d, up_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.up_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of up_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.gate_type) != 0) {
      printf("intermediate_size: %d, gate_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.gate_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of gate_type blck size");
    }

    auto gate_ratio = ggml_type_size((ggml_type)config.gate_type) / ggml_blck_size((ggml_type)config.gate_type);
    auto up_ratio = ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type);
    auto down_ratio = ggml_type_size((ggml_type)config.down_type) / ggml_blck_size((ggml_type)config.down_type);
    bool is_tp_split = (config.intermediate_size != complete_intermediate_size);
    down_baseline_is_mmap_ = config.use_mmap && !is_tp_split;

    if (config.use_mmap) {
      // mmap mode: set per-expert pointers into mmap region
      // gate/up: [expert, intermediate, hidden] - contiguous per-expert even with TP
      for (int eid = 0; eid < config.expert_num; eid++) {
        uint64_t logical_eid = expert_map(physical_to_logical_map, eid);
        uint8_t* gate_ptr = (uint8_t*)config.gate_proj +
            ((size_t)logical_eid * complete_intermediate_size + offset) * config.hidden_size * gate_ratio;
        uint8_t* up_ptr = (uint8_t*)config.up_proj +
            ((size_t)logical_eid * complete_intermediate_size + offset) * config.hidden_size * up_ratio;
        gate_expert_ptrs_[eid].store(gate_ptr, std::memory_order_release);
        up_expert_ptrs_[eid].store(up_ptr, std::memory_order_release);
        baseline_gate_ptrs_[eid] = gate_ptr;
        baseline_up_ptrs_[eid] = up_ptr;
      }

      if (!is_tp_split) {
        // No TP split: down_proj is also contiguous per-expert, zero-copy
        for (int eid = 0; eid < config.expert_num; eid++) {
          uint64_t logical_eid = expert_map(physical_to_logical_map, eid);
          uint8_t* down_ptr = (uint8_t*)config.down_proj +
              (size_t)logical_eid * config.hidden_size * complete_intermediate_size * down_ratio;
          down_expert_ptrs_[eid].store(down_ptr, std::memory_order_release);
          baseline_down_ptrs_[eid] = down_ptr;
        }
      } else {
        // TP split: down_proj [expert, hidden, complete_intermediate] needs stride-copy
        // because the TP slice along intermediate is not contiguous per row
        down_storage_ = new uint8_t[(size_t)config.expert_num * down_expert_bytes_];
        uint8_t* src_down_base = (uint8_t*)config.down_proj;
        for (int eid = 0; eid < config.expert_num; eid++) {
          uint64_t logical_eid = expert_map(physical_to_logical_map, eid);
          uint8_t* dst = down_storage_ + (size_t)eid * down_expert_bytes_;
          for (int j = 0; j < config.hidden_size; j++) {
            uint8_t* src_row = src_down_base +
                ((size_t)logical_eid * config.hidden_size + j) * complete_intermediate_size * down_ratio +
                (size_t)offset * down_ratio;
            memcpy(dst + (size_t)j * config.intermediate_size * down_ratio,
                   src_row,
                   (size_t)config.intermediate_size * down_ratio);
          }
          down_expert_ptrs_[eid].store(dst, std::memory_order_release);
          baseline_down_ptrs_[eid] = dst;
        }
      }

      // Bind mmap pages to this TP's NUMA node for locality.
      // This runs inside do_numa_job() on a NUMA-bound worker thread.
      // Without mbind, pages would land on whichever NUMA node first faulted them
      // (often Python's main thread on node 0), causing cross-NUMA reads during forward.
      for (int eid = 0; eid < config.expert_num; eid++) {
        bind_pages_to_numa(baseline_gate_ptrs_[eid], gate_expert_bytes_, numa_node_);
        bind_pages_to_numa(baseline_up_ptrs_[eid], up_expert_bytes_, numa_node_);
        if (!is_tp_split) {
          // Zero-copy down_proj pages also need binding
          bind_pages_to_numa(baseline_down_ptrs_[eid], down_expert_bytes_, numa_node_);
        }
        // TP-split down_proj: stride-copied into down_storage_ (new[])
        // which already respects the thread's NUMA policy via set_memory_to_numa().
      }

    } else {
      // Legacy mode: copy from source into per-expert storage
      uint8_t* gate_base = (uint8_t*)config.gate_proj;
      uint8_t* up_base = (uint8_t*)config.up_proj;
      uint8_t* down_base = (uint8_t*)config.down_proj;

      for (int eid = 0; eid < config.expert_num; eid++) {
        uint64_t logical_eid = expert_map(physical_to_logical_map, eid);
        uint8_t* src_gate =
            gate_base + ((size_t)logical_eid * complete_intermediate_size + offset) * config.hidden_size * gate_ratio;
        uint8_t* src_up =
            up_base + ((size_t)logical_eid * complete_intermediate_size + offset) * config.hidden_size * up_ratio;

        // Gate/up: contiguous per-expert in source
        uint8_t* gate_ptr = gate_storage_ + (size_t)eid * gate_expert_bytes_;
        uint8_t* up_ptr = up_storage_ + (size_t)eid * up_expert_bytes_;
        gate_expert_ptrs_[eid].store(gate_ptr, std::memory_order_release);
        up_expert_ptrs_[eid].store(up_ptr, std::memory_order_release);
        memcpy(gate_ptr, src_gate, gate_expert_bytes_);
        memcpy(up_ptr, src_up, up_expert_bytes_);

        // Down: stride-copy [hidden_size rows, each intermediate_size from complete_intermediate_size]
        uint8_t* down_ptr = down_storage_ + (size_t)eid * down_expert_bytes_;
        down_expert_ptrs_[eid].store(down_ptr, std::memory_order_release);
        for (int j = 0; j < config.hidden_size; j++) {
          uint8_t* src_down =
              down_base + ((size_t)logical_eid * config.hidden_size + j) * complete_intermediate_size * down_ratio +
              (size_t)offset * down_ratio;
          memcpy(down_ptr + (size_t)j * config.intermediate_size * down_ratio,
                 src_down,
                 (size_t)config.intermediate_size * down_ratio);
        }

        baseline_gate_ptrs_[eid] = gate_ptr;
        baseline_up_ptrs_[eid] = up_ptr;
        baseline_down_ptrs_[eid] = down_ptr;
      }
    }
  }

  void warm_up() {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                               ggml_blck_size((ggml_type)config_.hidden_type));
    std::vector<float> output(config_.hidden_size);
    for (int i = 0; i < config_.hidden_size; i++) {
      input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, (ggml_type)config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
      int64_t expert_ids = i;
      float weights = 0;
      forward_one(1, &expert_ids, &weights, input.data(), output.data());
    }
  }

  static float act_fn(float x) { return x / (1.0f + expf(-x)); }

  static inline bool use_row_dot_debug_fallback() {
    const char* env = std::getenv("KT_LLAMA_USE_ROW_DOT");
    return env && env[0] == '1';
  }

  static inline void rowwise_vec_dot(int n, int rows, const void* weights, size_t weight_row_bytes, ggml_type weight_type,
                                     const void* input, float* output) {
    auto vec_dot = ggml_internal_get_type_traits(weight_type).vec_dot;
    for (int row = 0; row < rows; ++row) {
      vec_dot(
          n,
          output + row,
          0,
          (const uint8_t*)weights + (size_t)row * weight_row_bytes,
          0,
          input,
          0,
          1
      );
    }
  }

  void forward_one(int k, const int64_t* expert_ids, const float* weights, const void* input, float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if ((ggml_type)config_.hidden_type == ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type &&
        (ggml_type)config_.hidden_type == ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
      gate_input_ptr = up_input_ptr = input;
    } else {
      to_float(input, s_input_fp32_, config_.hidden_size, (ggml_type)config_.hidden_type);
      if (ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type ==
          ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
        from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                   ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
        gate_input_ptr = up_input_ptr = s_gate_input_;
      } else {
        if ((ggml_type)config_.hidden_type !=
            ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) {
          from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                     ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
          gate_input_ptr = s_gate_input_;
        } else {
          gate_input_ptr = input;
        }
        if ((ggml_type)config_.hidden_type != ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
          from_float(s_input_fp32_, s_up_input_, config_.hidden_size,
                     ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
          up_input_ptr = s_up_input_;
        } else {
          up_input_ptr = input;
        }
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("gate_input: ");
    // debug_quant(const_cast<void *>(gate_input_ptr),
    // ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
    // printf("up_input: ");
    // debug_quant(const_cast<void *>(up_input_ptr),
    // ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, convert time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

#endif

    int activated_expert = 0;
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      activated_expert++;
    }

    int nth = config_.intermediate_size / config_.m_block;
    const bool use_row_dot = use_row_dot_debug_fallback();

    // Only process activated (CPU) experts; skip GPU experts entirely to keep buffers aligned.
    const char* debug_env = std::getenv("KT_DEBUG_LLAMAFILE_EXPERT_ID");
    int debug_expert_id = debug_env ? std::atoi(debug_env) : -1;
    static std::atomic<bool> printed_once{false};
    std::unique_ptr<ExpertReadScope> expert_read_scope;
    if (activated_expert > 0) {
      expert_read_scope.reset(new ExpertReadScope(this, m_expert_id_map_.data(), activated_expert));

    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int act_idx = task_id / nth;
          int64_t expert_id = m_expert_id_map_[act_idx];
            if (expert_id == -1) {
              return;
            }
            int ith = task_id % nth;

            uint8_t* gate_base_ptr = gate_expert_ptrs_[expert_id].load(std::memory_order_acquire);
            void* gate_proj_ptr =
                gate_base_ptr + (size_t)ith * config_.m_block *
                                    config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                    ggml_blck_size((ggml_type)config_.gate_type);

            float* gate_output_ptr = s_gate_output_[act_idx] + ith * config_.m_block;
            if (use_row_dot) {
              rowwise_vec_dot(
                  config_.hidden_size,
                  config_.m_block,
                  gate_proj_ptr,
                  (size_t)config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                      ggml_blck_size((ggml_type)config_.gate_type),
                  (ggml_type)config_.gate_type,
                  gate_input_ptr,
                  gate_output_ptr
              );
            } else {
              auto ok = llamafile_sgemm(
                  config_.m_block, 1, config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                  config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                  config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr, config_.m_block, 0,
                  1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                  ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type, GGML_TYPE_F32,
                  GGML_PREC_DEFAULT);
              if (ok == false) [[unlikely]] {
                throw std::runtime_error("llamafile not supported");
              }
            }

            uint8_t* up_base_ptr = up_expert_ptrs_[expert_id].load(std::memory_order_acquire);
            void* up_proj_ptr =
                up_base_ptr + (size_t)ith * config_.m_block *
                                  config_.hidden_size * ggml_type_size((ggml_type)config_.up_type) /
                                  ggml_blck_size((ggml_type)config_.up_type);

            float* up_output_ptr = s_up_output_[act_idx] + ith * config_.m_block;
            if (use_row_dot) {
              rowwise_vec_dot(
                  config_.hidden_size,
                  config_.m_block,
                  up_proj_ptr,
                  (size_t)config_.hidden_size * ggml_type_size((ggml_type)config_.up_type) /
                      ggml_blck_size((ggml_type)config_.up_type),
                  (ggml_type)config_.up_type,
                  up_input_ptr,
                  up_output_ptr
              );
            } else {
              llamafile_sgemm(config_.m_block, 1, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
                              up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
                              config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
                              config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
                              ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type, GGML_TYPE_F32,
                              GGML_PREC_DEFAULT);
            }

            for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
              s_intermediate_fp32_[act_idx][i] = act_fn(s_gate_output_[act_idx][i]) * s_up_output_[act_idx][i];
            }
            if (!printed_once.load(std::memory_order_acquire) && expert_id == debug_expert_id && ith == 0) {
              printed_once.store(true, std::memory_order_release);
              fprintf(stderr, "[LLAMA_DEBUG] expert=%ld gate_head=", (long)expert_id);
              for (int t = 0; t < 8; ++t) fprintf(stderr, "%f ", s_gate_output_[act_idx][t]);
              fprintf(stderr, "\n[LLAMA_DEBUG] expert=%ld up_head=", (long)expert_id);
              for (int t = 0; t < 8; ++t) fprintf(stderr, "%f ", s_up_output_[act_idx][t]);
              fprintf(stderr, "\n[LLAMA_DEBUG] expert=%ld inter_head=", (long)expert_id);
              for (int t = 0; t < 8; ++t) fprintf(stderr, "%f ", s_intermediate_fp32_[act_idx][t]);
              fprintf(stderr, "\n");
              fflush(stderr);
            }
            if (config_.m_block %
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) ==
                0) {
              float* intermediate_fp32_ptr = s_intermediate_fp32_[act_idx] + ith * config_.m_block;
              void* down_input_ptr =
                  s_down_input_[act_idx] +
                  ith * config_.m_block *
                      ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                      ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
              from_float(intermediate_fp32_ptr, down_input_ptr, config_.m_block,
                         ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
            }
          },
          nullptr);
    }

    if (config_.m_block % ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) !=
        0) {
      for (int i = 0; i < activated_expert; i++) {
        from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size,
                   ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("sinter:");
    // debug_f32(s_intermediate_fp32_[expert_ids[0]]);
    auto t2 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, gate/up time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
#endif

    nth = config_.hidden_size / config_.m_block;
    pool->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
            output[i] = 0;
          }
          for (int expert_idx = 0; expert_idx < activated_expert; expert_idx++) {
            int64_t expert_id = m_expert_id_map_[expert_idx];
            if (expert_id == -1) {
              continue;
            }

            auto expert_offset = expert_id * config_.hidden_size * config_.intermediate_size;
            auto m_block_offset = ith * config_.m_block * config_.intermediate_size;
            uint8_t* down_base_ptr = down_expert_ptrs_[expert_id].load(std::memory_order_acquire);
            void* down_proj_ptr = down_base_ptr + (size_t)m_block_offset *
                                                      ggml_type_size((ggml_type)config_.down_type) /
                                                      ggml_blck_size((ggml_type)config_.down_type);

            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.m_block;
            if (use_row_dot) {
              rowwise_vec_dot(
                  config_.intermediate_size,
                  config_.m_block,
                  down_proj_ptr,
                  (size_t)config_.intermediate_size * ggml_type_size((ggml_type)config_.down_type) /
                      ggml_blck_size((ggml_type)config_.down_type),
                  (ggml_type)config_.down_type,
                  s_down_input_[expert_idx],
                  down_output_ptr
              );
            } else {
              llamafile_sgemm(
                  config_.m_block, 1, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                  down_proj_ptr, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                  s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                  down_output_ptr, config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                  ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type, GGML_TYPE_F32,
                  GGML_PREC_DEFAULT);
            }

            float expert_weight = 0.0f;
            for (int j = 0; j < k; j++) {
              if (expert_ids[j] == expert_id) {
                expert_weight = weights[j];
                break;
              }
            }

            for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
              output[i] += s_down_output_[expert_idx][i] * expert_weight;
            }
            if (expert_id == debug_expert_id && ith == 0) {
              fprintf(stderr, "[LLAMA_DEBUG] expert=%ld down_head=", (long)expert_id);
              for (int t = 0; t < 8; ++t) fprintf(stderr, "%f ", s_down_output_[expert_idx][t]);
              fprintf(stderr, "\n");
              fflush(stderr);
            }
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    auto t3 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, down time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    fmt::print("numa_node: {}, total time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count());
#endif
  }

  void forward_many(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                    float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_gate_input_ptr_[i] =
          m_local_gate_input_ +
          offset * config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
      m_local_up_input_ptr_[i] =
          m_local_up_input_ +
          offset * config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
      m_local_down_input_ptr_[i] =
          m_local_down_input_ +
          offset * config_.intermediate_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          const void* gate_input_ptr;
          const void* up_input_ptr;
          if ((ggml_type)config_.hidden_type ==
                  ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type &&
              (ggml_type)config_.hidden_type ==
                  ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                                  ggml_type_size((ggml_type)config_.hidden_type) /
                                                                  ggml_blck_size((ggml_type)config_.hidden_type);
          } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                           ggml_blck_size((ggml_type)config_.hidden_type),
                     m_input_fp32_[i], config_.hidden_size, (ggml_type)config_.hidden_type);
            if (ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type ==
                ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
              from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                         ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
              gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
              if ((ggml_type)config_.hidden_type !=
                  ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                           ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
                gate_input_ptr = m_gate_input_[i];
              } else {
                gate_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                       ggml_type_size((ggml_type)config_.hidden_type) /
                                                       ggml_blck_size((ggml_type)config_.hidden_type);
              }
              if ((ggml_type)config_.hidden_type !=
                  ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size,
                           ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
                up_input_ptr = m_up_input_[i];
              } else {
                up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                     ggml_type_size((ggml_type)config_.hidden_type) /
                                                     ggml_blck_size((ggml_type)config_.hidden_type);
              }
            }
          }
          for (int j = 0; j < k; j++) {
            if (config_.should_skip_expert(expert_ids[i * k + j])) {
              continue;
            }
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                           ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type),
                   gate_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                       ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                           ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type),
                   up_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                       ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int m_block = QK_K;
    int nth = config_.intermediate_size / m_block;
    // printf("nth: %d, m_block: %d, activated_expert: %d\n", nth, m_block, activated_expert);
    // printf("config_.hidden_size: %d, config_.intermediate_size: %d\n", config_.hidden_size,
    // config_.intermediate_size);
    ExpertReadScope expert_read_scope(this, m_expert_id_map_.data(), activated_expert);
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

          uint8_t* gate_base_ptr = gate_expert_ptrs_[expert_idx].load(std::memory_order_acquire);
          void* gate_proj_ptr =
              gate_base_ptr + (size_t)ith * m_block *
                                  config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                  ggml_blck_size((ggml_type)config_.gate_type);

          float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * m_block;

          // if (ith == 0) {
          //   printf("matrix size: m:%d, n:%d, k:%d\n", m_block, m_local_num_[expert_idx],
          //          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type));
          // }
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr,
                          config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                          ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type, GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
          void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

          uint8_t* up_base_ptr = up_expert_ptrs_[expert_idx].load(std::memory_order_acquire);
          void* up_proj_ptr = up_base_ptr + (size_t)ith * m_block *
                                                config_.hidden_size *
                                                ggml_type_size((ggml_type)config_.up_type) /
                                                ggml_blck_size((ggml_type)config_.up_type);

          float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(
              m_block, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
              up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
              config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
              config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
              ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * m_block; j < (ith + 1) * m_block; j++) {
              m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] =
                  act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) *
                  m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr =
                m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * m_block;
            void* down_input_ptr =
                m_local_down_input_ptr_[expert_idx] +
                i * config_.intermediate_size *
                    ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) +
                ith * m_block *
                    ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, m_block,
                       ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    m_block = QK_K;
    nth = config_.hidden_size / m_block;
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* down_input_ptr = m_local_down_input_ptr_[expert_idx];

          auto m_block_offset = ith * m_block * config_.intermediate_size;

          uint8_t* down_base_ptr = down_expert_ptrs_[expert_idx].load(std::memory_order_acquire);
          void* down_proj_ptr = down_base_ptr + (size_t)m_block_offset *
                                                    ggml_type_size((ggml_type)config_.down_type) /
                                                    ggml_blck_size((ggml_type)config_.down_type);

          float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_proj_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_input_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_output_ptr,
                          config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                          ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type, GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
          }
          for (int j = 0; j < k; j++) {
            if (config_.should_skip_expert(expert_ids[i * k + j])) {
              continue;
            }
            for (int e = 0; e < config_.hidden_size; e++) {
              m_output_fp32_[i][e] +=
                  m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] *
                  weights[i * k + j];
            }
          }
          for (int e = 0; e < config_.hidden_size; e++) {
            output[i * config_.hidden_size + e] = m_output_fp32_[i][e];
          }
        },
        nullptr);
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output_in) {
    auto output = (float*)output_in;
    if (qlen < config_.group_min_len) {
      for (int i = 0; i < qlen; i++) {
        forward_one(k, expert_ids + i * k, weights + i * k,
                    (uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                          ggml_blck_size((ggml_type)config_.hidden_type),
                    output + i * config_.hidden_size);
      }
      return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k,
            (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                  ggml_blck_size((ggml_type)config_.hidden_type),
            output + forward_len * config_.hidden_size);
  }

  ~LLAMA_MOE_TP() {
#ifndef _WIN32
    for (int i = 0; i < config_.expert_num; i++) {
      uint8_t* gate_ptr = tier0_gate_[i].load(std::memory_order_acquire);
      uint8_t* up_ptr = tier0_up_[i].load(std::memory_order_acquire);
      uint8_t* down_ptr = tier0_down_[i].load(std::memory_order_acquire);
      if (gate_ptr) numa_free(gate_ptr, gate_expert_bytes_);
      if (up_ptr) numa_free(up_ptr, up_expert_bytes_);
      if (down_ptr) numa_free(down_ptr, down_expert_bytes_);
    }
#endif
    delete[] gate_storage_;
    delete[] up_storage_;
    delete[] down_storage_;
  }

  void promote_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    uint8_t expected_state = EXPERT_BASELINE;
    if (!expert_states_[expert_id].compare_exchange_strong(
            expected_state, EXPERT_PROMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }

    // Allocate NUMA-local buffers
    uint8_t* gate_tier0 = (uint8_t*)numa_alloc_onnode(gate_expert_bytes_, numa_node_);
    uint8_t* up_tier0 = (uint8_t*)numa_alloc_onnode(up_expert_bytes_, numa_node_);
    uint8_t* down_tier0 = (uint8_t*)numa_alloc_onnode(down_expert_bytes_, numa_node_);

    if (!gate_tier0 || !up_tier0 || !down_tier0) {
      // Allocation failed, clean up
      if (gate_tier0) numa_free(gate_tier0, gate_expert_bytes_);
      if (up_tier0) numa_free(up_tier0, up_expert_bytes_);
      if (down_tier0) numa_free(down_tier0, down_expert_bytes_);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return;
    }

    // Prefetch source pages before memcpy to reduce page-fault stalls.
    // When baseline pointers are in mmap, pages may have been evicted under memory pressure.
    // madvise(MADV_WILLNEED) triggers asynchronous readahead so pages are resident by the
    // time memcpy reaches them, avoiding synchronous page faults (~100us each).
    if (config_.use_mmap) {
      madvise(baseline_gate_ptrs_[expert_id], gate_expert_bytes_, MADV_WILLNEED);
      madvise(baseline_up_ptrs_[expert_id], up_expert_bytes_, MADV_WILLNEED);
      if (down_baseline_is_mmap_) {
        madvise(baseline_down_ptrs_[expert_id], down_expert_bytes_, MADV_WILLNEED);
      }
    }

    // Copy data from current live pointers (mmap or legacy)
	    memcpy(gate_tier0, gate_expert_ptrs_[expert_id].load(std::memory_order_acquire), gate_expert_bytes_);
	    memcpy(up_tier0, up_expert_ptrs_[expert_id].load(std::memory_order_acquire), up_expert_bytes_);
	    memcpy(down_tier0, down_expert_ptrs_[expert_id].load(std::memory_order_acquire), down_expert_bytes_);

    tier0_gate_[expert_id].store(gate_tier0, std::memory_order_release);
    tier0_up_[expert_id].store(up_tier0, std::memory_order_release);
    tier0_down_[expert_id].store(down_tier0, std::memory_order_release);

    // Publish live pointers to NUMA-local buffers without serializing the hot path.
	    gate_expert_ptrs_[expert_id].store(gate_tier0, std::memory_order_release);
	    up_expert_ptrs_[expert_id].store(up_tier0, std::memory_order_release);
	    down_expert_ptrs_[expert_id].store(down_tier0, std::memory_order_release);
	    expert_states_[expert_id].store(EXPERT_PROMOTED, std::memory_order_release);

    if (config_.use_mmap) {
      madvise(baseline_gate_ptrs_[expert_id], gate_expert_bytes_, MADV_DONTNEED);
      madvise(baseline_up_ptrs_[expert_id], up_expert_bytes_, MADV_DONTNEED);
      if (down_baseline_is_mmap_) {
        madvise(baseline_down_ptrs_[expert_id], down_expert_bytes_, MADV_DONTNEED);
      }
    }
#endif
  }

  void demote_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    uint8_t expected_state = EXPERT_PROMOTED;
    if (!expert_states_[expert_id].compare_exchange_strong(
            expected_state, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return;
    }
    uint8_t* gate_tier0 = tier0_gate_[expert_id].load(std::memory_order_acquire);
    uint8_t* up_tier0 = tier0_up_[expert_id].load(std::memory_order_acquire);
    uint8_t* down_tier0 = tier0_down_[expert_id].load(std::memory_order_acquire);
    if (gate_tier0 == nullptr || up_tier0 == nullptr || down_tier0 == nullptr) {
      tier0_gate_[expert_id].store(nullptr, std::memory_order_release);
      tier0_up_[expert_id].store(nullptr, std::memory_order_release);
      tier0_down_[expert_id].store(nullptr, std::memory_order_release);
      gate_expert_ptrs_[expert_id].store(baseline_gate_ptrs_[expert_id], std::memory_order_release);
      up_expert_ptrs_[expert_id].store(baseline_up_ptrs_[expert_id], std::memory_order_release);
      down_expert_ptrs_[expert_id].store(baseline_down_ptrs_[expert_id], std::memory_order_release);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return;
    }

    // Restore live pointers to baseline (mmap or legacy storage)
    gate_expert_ptrs_[expert_id].store(baseline_gate_ptrs_[expert_id], std::memory_order_release);
    up_expert_ptrs_[expert_id].store(baseline_up_ptrs_[expert_id], std::memory_order_release);
    down_expert_ptrs_[expert_id].store(baseline_down_ptrs_[expert_id], std::memory_order_release);

    while (active_readers_[expert_id].load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }

    // Free NUMA-local buffers
    numa_free(gate_tier0, gate_expert_bytes_);
    numa_free(up_tier0, up_expert_bytes_);
    numa_free(down_tier0, down_expert_bytes_);

    tier0_gate_[expert_id].store(nullptr, std::memory_order_release);
    tier0_up_[expert_id].store(nullptr, std::memory_order_release);
    tier0_down_[expert_id].store(nullptr, std::memory_order_release);
    expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
#endif
  }

  bool is_expert_promoted(int expert_id) const {
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_PROMOTED;
  }
};

template <>
class TP_MOE<LLAMA_MOE_TP> : public TP_MOE_Common<LLAMA_MOE_TP> {
 public:
  using TP_MOE_Common<LLAMA_MOE_TP>::TP_MOE_Common;

  void load_weights() {
    auto pool = this->config.pool;

    std::vector<int> tp_offsets(this->tp_count);
    int accumulated_offset = 0;
    for (int i = 0; i < this->tp_count; i++) {
      tp_offsets[i] = accumulated_offset;
      accumulated_offset += this->tp_configs[i].intermediate_size;
    }

    pool->dispense_backend()->do_numa_job([this, pool, tp_offsets](int tp_id) {
      this->tps[tp_id]->load_weights(this->config.intermediate_size, tp_offsets[tp_id]);
    });
    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }

  void merge_results(int qlen, void* output, bool incremental) {
    auto pool = this->config.pool;
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, incremental](int token_nth) {
          if (incremental) {
            to_float((uint8_t*)output + token_nth * config.hidden_size * ggml_type_size((ggml_type)config.hidden_type) /
                                            ggml_blck_size((ggml_type)config.hidden_type),
                     local_output + token_nth * config.hidden_size, config.hidden_size, (ggml_type)config.hidden_type);
            for (int e = 0; e < config.hidden_size; e++) {
              local_output_numa[0][token_nth * config.hidden_size + e] +=
                  local_output[token_nth * config.hidden_size + e];
            }
          }
          auto& tp_count = this->tp_count;
          for (int i = 1; i < tp_count; i++) {
            for (int e = 0; e < config.hidden_size; e++) {
              local_output_numa[0][token_nth * config.hidden_size + e] +=
                  local_output_numa[i][token_nth * config.hidden_size + e];
            }
          }
          from_float(local_output_numa[0] + token_nth * config.hidden_size,
                     (uint8_t*)output + token_nth * config.hidden_size * ggml_type_size((ggml_type)config.hidden_type) /
                                            ggml_blck_size((ggml_type)config.hidden_type),
                     config.hidden_size, (ggml_type)config.hidden_type);
        },
        nullptr);
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
    // Demote is lightweight (pointer swap + numa_free), no memcpy.
    // NUMA locality doesn't matter for demote, keep simple sequential dispatch.
    for (int i = 0; i < this->tp_count; i++) {
      this->tps[i]->demote_expert(expert_id);
    }
  }

  bool is_expert_promoted(int expert_id) const {
    return this->tps[0]->is_expert_promoted(expert_id);
  }
};
#endif
