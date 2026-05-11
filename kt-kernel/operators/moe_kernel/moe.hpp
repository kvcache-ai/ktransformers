#ifndef MOE_KERNEL_HPP
#define MOE_KERNEL_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#ifndef _WIN32
#include <numa.h>
#include <sys/mman.h>
#endif
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "api/common.h"
#include "api/mat_kernel.h"
#include "llama.cpp/ggml.h"
template <class T, bool PLAIN = true>
class MOE_KERNEL_TP
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  int tp_part_idx;
  std::filesystem::path prefix;

  void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if
                     // quantized)]

  ggml_bf16_t* m_local_input_;  // [routed_expert_num * max_len * hidden_size]
  float* m_local_gate_output_;  // [routed_expert_num * max_len * intermediate_size]
  float* m_local_up_output_;    // [routed_expert_num * max_len * intermediate_size]
  float* m_local_down_output_;  // [routed_expert_num * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;    // [max_len, routed_expert_num]
  std::vector<int> m_local_num_;                 // [expert_num]
  std::vector<int> m_expert_id_map_;             // [expert_num]
  std::vector<ggml_bf16_t*> m_local_input_ptr_;  // [expert_num]
  std::vector<float*> m_local_gate_output_ptr_;  // [expert_num]
  std::vector<float*> m_local_up_output_ptr_;    // [expert_num]
  std::vector<float*> m_local_down_output_ptr_;  // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  std::vector<void*> gate_up_owner_ptr_;
  std::vector<void*> down_owner_ptr_;

#ifndef _WIN32
  enum ExpertState : uint8_t {
    EXPERT_BASELINE = 0,
    EXPERT_PACKING = 1,
    EXPERT_CACHED = 2,
    EXPERT_PINNED = 3,
    EXPERT_DEMOTING = 4,
  };

  std::vector<void*> baseline_gate_weight_src_;
  std::vector<void*> baseline_up_weight_src_;
  std::vector<void*> baseline_down_weight_src_;
  std::vector<float*> baseline_gate_scale_src_;
  std::vector<float*> baseline_up_scale_src_;
  std::vector<float*> baseline_down_scale_src_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;
  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  ResidentCachePolicyState resident_policy_;

  size_t gate_total_bytes_ = 0;
  size_t up_total_bytes_ = 0;
  size_t down_total_bytes_ = 0;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  size_t gate_scale_bytes_ = 0;
  size_t up_scale_bytes_ = 0;
  size_t down_scale_bytes_ = 0;
#endif

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                            size_t scale_size) {
    // printf("expert %d, size %ld, scale size %ld\n", expert_idx, size, scale_size);
    // std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file: %s", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                            std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                     .c_str());
      // throw std::runtime_error("No such file");
    }
    of.write((char*)bb, size - scale_size);
    of.close();
    // of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file\n");
      // throw std::runtime_error("No such file");
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    // std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * (size - scale_size) / mat_split);
    f.read(((char*)bb) + mat_split_idex * (size - scale_size) / mat_split, (size - scale_size) / mat_split);
    f.close();
    // f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                     "_scale_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(scale_size) + "Byte" + "_scale_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * scale_size / mat_split);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_size / mat_split, scale_size / mat_split);
  }

 public:
  using input_t = ggml_bf16_t;
  using output_t = float;

  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  MOE_KERNEL_TP(GeneralMOEConfig config, int tp_part_idx) {
    printf("  Creating AMD_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
    auto& load = config.load;
    auto& save = config.save;
    if (load && config.path == "") {
      load = false;
    }

    prefix = config.path;
    prefix = prefix / ("_layer_" + std::to_string(config.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
    if (save) {
      std::cout << "Creating " << prefix << std::endl;
      std::filesystem::create_directories(prefix);
    }
    if (load) {
      if (std::filesystem::exists(prefix)) {
        std::cout << "Loading from " << prefix << std::endl;
      } else {
        throw std::runtime_error("Path not found: " + prefix.string());
      }
    }

    this->tp_part_idx = tp_part_idx;
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    MemoryRequest mem_requests;
    mem_requests.append_pointer(&m_local_input_,
                                sizeof(input_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);

    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * config_.num_experts_per_tok * config_.max_len *
                                                           config_.intermediate_size);
    mem_requests.append_pointer(
        &m_local_up_output_, sizeof(float) * config_.num_experts_per_tok * config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_,
                                sizeof(float) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    // printf("tp part %d alloc layer %d, %f GB, on numa %d\n", tp_part_idx, config_.layer_idx,
    //        1e-9 * config_.expert_num *
    //            (T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) * 2 +
    //             T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)),
    //        numa_node_of_cpu(sched_getcpu()));
    // 统一分配一块巨大的内存用于权重：
    size_t gate_up_exp_size =
        T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN) +
        T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, nullptr));
      void* gate_up_down_per_exp_ptr = std::aligned_alloc(64, gate_up_exp_size);
      gate_up_owner_ptr_.push_back(gate_up_down_per_exp_ptr);

      gate_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size,
                                                               gate_up_down_per_exp_ptr, PACKED, 'u', PLAIN));
      up_bb_.push_back(std::make_shared<typename T::BufferB>(
          config_.intermediate_size, config_.hidden_size,
          offset_pointer(gate_up_down_per_exp_ptr,
                         T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN)),
          PACKED, 'u', PLAIN));

      void* down_bb_ptr = std::aligned_alloc(
          64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, PACKED, 'd', PLAIN));
      down_owner_ptr_.push_back(down_bb_ptr);
      down_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size,
                                                               down_bb_ptr, PACKED, 'd', PLAIN));
    }

    for (int i = 0; i < config_.expert_num; i++) {
      mem_requests.append_function([this, i](void* new_ptr) { gate_up_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.hidden_size));
      mem_requests.append_function([this, i](void* new_ptr) { gate_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { up_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.hidden_size));
    }

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

#ifndef _WIN32
    if (config_.use_mmap) {
      initialize_lazy_mmap_state();
    }
#endif
  }

  MOE_KERNEL_TP(const MOE_KERNEL_TP&) = delete;
  MOE_KERNEL_TP& operator=(const MOE_KERNEL_TP&) = delete;
  MOE_KERNEL_TP(MOE_KERNEL_TP&&) = delete;
  MOE_KERNEL_TP& operator=(MOE_KERNEL_TP&&) = delete;

  ~MOE_KERNEL_TP() {
    // printf("  Destroying KML_MOE_TP %lx\n", (intptr_t)(this));
    for (void* ptr : gate_up_owner_ptr_) {
      if (ptr) std::free(ptr);
    }
    for (void* ptr : down_owner_ptr_) {
      if (ptr) std::free(ptr);
    }
  }

#ifndef _WIN32
  void initialize_lazy_mmap_state() {
    const int en = config_.expert_num;
    baseline_gate_weight_src_.assign(en, nullptr);
    baseline_up_weight_src_.assign(en, nullptr);
    baseline_down_weight_src_.assign(en, nullptr);
    baseline_gate_scale_src_.assign(en, nullptr);
    baseline_up_scale_src_.assign(en, nullptr);
    baseline_down_scale_src_.assign(en, nullptr);

    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    resident_expert_count_.store(0, std::memory_order_relaxed);
    eviction_cursor_.store(0, std::memory_order_relaxed);

    gate_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
    up_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
    down_total_bytes_ = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, PACKED, 'd', PLAIN);
    gate_scale_bytes_ = config_.intermediate_size * sizeof(float);
    up_scale_bytes_ = config_.intermediate_size * sizeof(float);
    down_scale_bytes_ = config_.hidden_size * sizeof(float);
    gate_weight_bytes_ = gate_total_bytes_ - gate_scale_bytes_;
    up_weight_bytes_ = up_total_bytes_ - up_scale_bytes_;
    down_weight_bytes_ = down_total_bytes_ - down_scale_bytes_;
    const int configured_resident = config_.max_resident_experts > 0 ? config_.max_resident_experts : config_.max_tier0_experts;
    cache_capacity_ = configured_resident <= 0
                          ? 0
                          : std::min(config_.expert_num, std::max(configured_resident, config_.num_experts_per_tok));
    resident_policy_.reset(en, config_.resident_cache_policy);

    if (config_.pool != nullptr && tp_part_idx < (int)config_.pool->config.subpool_numa_map.size()) {
      numa_node_ = config_.pool->config.subpool_numa_map[tp_part_idx];
    }

    for (int expert_id = 0; expert_id < en; ++expert_id) {
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_relaxed);
      active_readers_[expert_id].store(0, std::memory_order_relaxed);
      std::free(gate_up_owner_ptr_[expert_id]);
      std::free(down_owner_ptr_[expert_id]);
      gate_up_owner_ptr_[expert_id] = nullptr;
      down_owner_ptr_[expert_id] = nullptr;
      gate_bb_[expert_id]->b = nullptr;
      up_bb_[expert_id]->b = nullptr;
      down_bb_[expert_id]->b = nullptr;
      gate_bb_[expert_id]->d = nullptr;
      up_bb_[expert_id]->d = nullptr;
      down_bb_[expert_id]->d = nullptr;
      if constexpr (!PLAIN) {
        std::fill(gate_bb_[expert_id]->b_pack.begin(), gate_bb_[expert_id]->b_pack.end(), nullptr);
        std::fill(up_bb_[expert_id]->b_pack.begin(), up_bb_[expert_id]->b_pack.end(), nullptr);
        std::fill(down_bb_[expert_id]->b_pack.begin(), down_bb_[expert_id]->b_pack.end(), nullptr);
      }
    }
  }

  void apply_baseline_ptrs(int expert_id) {
    if constexpr (PLAIN) {
      gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(baseline_gate_weight_src_[expert_id]);
      up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(baseline_up_weight_src_[expert_id]);
      down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(baseline_down_weight_src_[expert_id]);
    } else {
      gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(baseline_gate_weight_src_[expert_id]);
      up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(baseline_up_weight_src_[expert_id]);
      down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(baseline_down_weight_src_[expert_id]);
      for (size_t i = 0; i < gate_bb_[expert_id]->nth_B; ++i) {
        gate_bb_[expert_id]->b_pack[i] =
            reinterpret_cast<int8_t*>(baseline_gate_weight_src_[expert_id]) + i * gate_bb_[expert_id]->reorder_B_size;
        up_bb_[expert_id]->b_pack[i] =
            reinterpret_cast<int8_t*>(baseline_up_weight_src_[expert_id]) + i * up_bb_[expert_id]->reorder_B_size;
        down_bb_[expert_id]->b_pack[i] =
            reinterpret_cast<int8_t*>(baseline_down_weight_src_[expert_id]) + i * down_bb_[expert_id]->reorder_B_size;
      }
    }
    gate_bb_[expert_id]->d = baseline_gate_scale_src_[expert_id];
    up_bb_[expert_id]->d = baseline_up_scale_src_[expert_id];
    down_bb_[expert_id]->d = baseline_down_scale_src_[expert_id];
  }

  void apply_owned_ptrs(int expert_id) {
    gate_bb_[expert_id]->set_data(gate_up_owner_ptr_[expert_id], PLAIN);
    up_bb_[expert_id]->set_data(moe_kernel::offset_pointer(gate_up_owner_ptr_[expert_id], gate_total_bytes_), PLAIN);
    down_bb_[expert_id]->set_data(down_owner_ptr_[expert_id], PLAIN);
  }

  void set_mmap_source_ptrs(int expert_id, void* gate_weight_src, void* up_weight_src, void* down_weight_src,
                            float* gate_scale_src, float* up_scale_src, float* down_scale_src) {
    baseline_gate_weight_src_[expert_id] = gate_weight_src;
    baseline_up_weight_src_[expert_id] = up_weight_src;
    baseline_down_weight_src_[expert_id] = down_weight_src;
    baseline_gate_scale_src_[expert_id] = gate_scale_src;
    baseline_up_scale_src_[expert_id] = up_scale_src;
    baseline_down_scale_src_[expert_id] = down_scale_src;
    apply_baseline_ptrs(expert_id);
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
    explicit ExpertReadScope(MOE_KERNEL_TP* owner, size_t reserve_count = 0) : owner_(owner) {
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
    MOE_KERNEL_TP* owner_;
    std::vector<int> experts_;
  };

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

      std::free(gate_up_owner_ptr_[victim]);
      std::free(down_owner_ptr_[victim]);
      gate_up_owner_ptr_[victim] = nullptr;
      down_owner_ptr_[victim] = nullptr;
      apply_baseline_ptrs(victim);
      note_expert_demote(victim);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[victim].store(EXPERT_BASELINE, std::memory_order_release);
      return true;
    }

    return false;
  }

  void drop_baseline_cache_for_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    if (baseline_gate_weight_src_[expert_id] != nullptr) {
      madvise(baseline_gate_weight_src_[expert_id], gate_weight_bytes_, MADV_DONTNEED);
    }
    if (baseline_up_weight_src_[expert_id] != nullptr) {
      madvise(baseline_up_weight_src_[expert_id], up_weight_bytes_, MADV_DONTNEED);
    }
    if (baseline_down_weight_src_[expert_id] != nullptr) {
      madvise(baseline_down_weight_src_[expert_id], down_weight_bytes_, MADV_DONTNEED);
    }
    if (baseline_gate_scale_src_[expert_id] != nullptr) {
      madvise((void*)baseline_gate_scale_src_[expert_id], gate_scale_bytes_, MADV_DONTNEED);
    }
    if (baseline_up_scale_src_[expert_id] != nullptr) {
      madvise((void*)baseline_up_scale_src_[expert_id], up_scale_bytes_, MADV_DONTNEED);
    }
    if (baseline_down_scale_src_[expert_id] != nullptr) {
      madvise((void*)baseline_down_scale_src_[expert_id], down_scale_bytes_, MADV_DONTNEED);
    }
#else
    (void)expert_id;
#endif
  }

  bool allocate_and_copy_expert(int expert_id, bool pin_after_copy) {
    if (cache_capacity_ > 0) {
      while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        if (!evict_one_cached_expert(expert_id)) {
          break;
        }
      }
      if (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        return false;
      }
    }

    void* gate_up_owner = numa_alloc_onnode(gate_total_bytes_ + up_total_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_total_bytes_, numa_node_);
    if (gate_up_owner == nullptr || down_owner == nullptr) {
      if (gate_up_owner) numa_free(gate_up_owner, gate_total_bytes_ + up_total_bytes_);
      if (down_owner) numa_free(down_owner, down_total_bytes_);
      return false;
    }

    if (baseline_gate_weight_src_[expert_id] == nullptr || baseline_up_weight_src_[expert_id] == nullptr ||
        baseline_down_weight_src_[expert_id] == nullptr || baseline_gate_scale_src_[expert_id] == nullptr ||
        baseline_up_scale_src_[expert_id] == nullptr || baseline_down_scale_src_[expert_id] == nullptr) {
      numa_free(gate_up_owner, gate_total_bytes_ + up_total_bytes_);
      numa_free(down_owner, down_total_bytes_);
      return false;
    }

    madvise(baseline_gate_weight_src_[expert_id], gate_weight_bytes_, MADV_WILLNEED);
    madvise(baseline_up_weight_src_[expert_id], up_weight_bytes_, MADV_WILLNEED);
    madvise(baseline_down_weight_src_[expert_id], down_weight_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_gate_scale_src_[expert_id], gate_scale_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_up_scale_src_[expert_id], up_scale_bytes_, MADV_WILLNEED);
    madvise((void*)baseline_down_scale_src_[expert_id], down_scale_bytes_, MADV_WILLNEED);

    std::memcpy(gate_up_owner, baseline_gate_weight_src_[expert_id], gate_weight_bytes_);
    std::memcpy(moe_kernel::offset_pointer(gate_up_owner, gate_weight_bytes_), baseline_gate_scale_src_[expert_id], gate_scale_bytes_);
    std::memcpy(moe_kernel::offset_pointer(gate_up_owner, gate_total_bytes_), baseline_up_weight_src_[expert_id], up_weight_bytes_);
    std::memcpy(moe_kernel::offset_pointer(gate_up_owner, gate_total_bytes_ + up_weight_bytes_),
                baseline_up_scale_src_[expert_id], up_scale_bytes_);
    std::memcpy(down_owner, baseline_down_weight_src_[expert_id], down_weight_bytes_);
    std::memcpy(moe_kernel::offset_pointer(down_owner, down_weight_bytes_), baseline_down_scale_src_[expert_id], down_scale_bytes_);

    gate_up_owner_ptr_[expert_id] = gate_up_owner;
    down_owner_ptr_[expert_id] = down_owner;
    apply_owned_ptrs(expert_id);
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_copy ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_copy);
    // After creating the NUMA-local resident copy, immediately release the
    // file-backed baseline pages for this expert. The mapping stays as the
    // fallback source if the resident copy is later evicted.
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
        if (!allocate_and_copy_expert(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          throw std::runtime_error("moe_kernel lazy-copy promotion failed");
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

  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (config_.gate_projs.size()) {
      printf("load from safetensor");
      if (config_.use_mmap) {
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              set_mmap_source_ptrs(
                  expert_id, config_.gate_projs[tp_part_idx][logical_expert_id],
                  config_.up_projs[tp_part_idx][logical_expert_id], config_.down_projs[tp_part_idx][logical_expert_id],
                  (float*)config_.gate_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.up_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.down_scales[tp_part_idx][logical_expert_id]);
            },
            nullptr);
      } else {
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              {
                size_t scale_size = config_.intermediate_size * sizeof(float);
                size_t whole_size_ =
                    T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
                size_t size = whole_size_ - scale_size;
                void* dst_ = PLAIN ? gate_bb_[expert_id]->b : gate_bb_[expert_id]->b_pack[0];

                memcpy(dst_, config_.gate_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
                }

                whole_size_ =
                    T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
                size = whole_size_ - scale_size;
                dst_ = PLAIN ? up_bb_[expert_id]->b : up_bb_[expert_id]->b_pack[0];
                memcpy(dst_, config_.up_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }

              {
                size_t scale_size = config_.hidden_size * sizeof(float);
                size_t whole_size_ =
                    T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, PACKED, 'd', PLAIN);
                size_t size = whole_size_ - scale_size;
                void* dst_ = PLAIN ? down_bb_[expert_id]->b : down_bb_[expert_id]->b_pack[0];
                memcpy(dst_, config_.down_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }
            },
            nullptr);
      }

    } else {
      static uint8_t mat_type_all = 3, mat_split = 1;
      if (config_.load) {
        std::cout << "Loading from " << prefix << std::endl;
        for (int task_id = 0; task_id < config_.expert_num * mat_type_all * mat_split; task_id++) {
          int64_t expert_idx = task_id / (mat_type_all * mat_split);
          uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
          uint8_t mat_split_idex = task_id % mat_split;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          void* src_;
          if (mat_class == 0) {  // the up matrix
            src_ = PLAIN ? up_bb_[expert_idx]->b : up_bb_[expert_idx]->b_pack[0];
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_up_", (char*)src_, logical_expert_id, size, scale_size, mat_split, mat_split_idex);
          } else if (mat_class == 1) {
            void* src_ = PLAIN ? gate_bb_[expert_idx]->b : gate_bb_[expert_idx]->b_pack[0];
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_gate_", (char*)src_, logical_expert_id, size, scale_size, mat_split, mat_split_idex);
          } else {
            void* src_ = PLAIN ? down_bb_[expert_idx]->b : down_bb_[expert_idx]->b_pack[0];
            size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, PACKED, 'd', PLAIN);
            size_t scale_size = config_.hidden_size * sizeof(float);
            read_weights(prefix, "_down_", (char*)src_, logical_expert_id, size, scale_size, mat_split, mat_split_idex);
          }
        }
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else
#endif
      {
        if (tp_part_idx == 0) {
          std::cout << "  online quant from bf16" << std::endl;
        }
        int nth = T::recommended_nth_up_gate(config_.intermediate_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // gate part
              gate_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth, -1, PACKED, PLAIN);
              // up part
              up_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth, -1, PACKED, PLAIN);
            },
            nullptr);

        nth = T::recommended_nth_down(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              int ith = task_id % nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              // down part
              down_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.down_proj + logical_expert_id * config_.hidden_size * config_.intermediate_size,
                  ith, nth, -1, PACKED, PLAIN);
            },
            nullptr);
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              expert_idx = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size =
                    T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b_pack[0], expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size =
                    T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, PACKED, 'u', PLAIN);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b_pack[0], expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size =
                    T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, PACKED, 'd', PLAIN);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b_pack[0], expert_idx, size, scale_size);
              }
            },
            nullptr);
      }
    }
  }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(input_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(output_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

#define MOE_DIRECT_OR_POOL_BY_VAR(var, fn)                       \
  do {                                                           \
    if (var < 5) {                                               \
      for (int i = 0; i < (var); i++) {                          \
        (fn)(i);                                                 \
      }                                                          \
    } else {                                                     \
      pool->do_work_stealing_job((var), nullptr, (fn), nullptr); \
    }                                                            \
  } while (0)
  static float act_fn(float x) { return x / (1.0f + expf(-x)); }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
#ifndef _WIN32
    std::unique_ptr<ExpertReadScope> expert_read_scope;
    std::vector<int> active_experts;
    if (config_.use_mmap) {
      active_experts.reserve(qlen * k);
      std::vector<uint8_t> seen(config_.expert_num, 0);

      for (int i = 0; i < qlen * k; ++i) {
        const int expert_id = (int)expert_ids[i];
        if (config_.should_skip_expert(expert_id) || seen[expert_id]) {
          continue;
        }
        seen[expert_id] = 1;
        active_experts.push_back(expert_id);
      }

      expert_read_scope = std::make_unique<ExpertReadScope>(this, active_experts.size());
      for (int expert_id : active_experts) {
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope->add_expert(expert_id);
      }
    }
#endif
    // Unified forward path: 'd' for decode (qlen<=1), 'p' for prefill (qlen>1)
    char mode = (qlen <= 1) ? 'd' : 'p';
    forward_unified(mode, qlen, k, expert_ids, weights, input, output);
#ifndef _WIN32
    if (config_.use_mmap && config_.max_tier0_experts <= 0 && !active_experts.empty()) {
      expert_read_scope.reset();
      for (int expert_id : active_experts) {
        demote_expert(expert_id);
      }
    }
#endif
  }

  void promote_expert(int expert_id) {
#ifndef _WIN32
    ensure_expert_ready(expert_id, true);
#else
    (void)expert_id;
#endif
  }

  void demote_expert(int expert_id) {
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

      std::free(gate_up_owner_ptr_[expert_id]);
      std::free(down_owner_ptr_[expert_id]);
      gate_up_owner_ptr_[expert_id] = nullptr;
      down_owner_ptr_[expert_id] = nullptr;
      apply_baseline_ptrs(expert_id);
      note_expert_demote(expert_id);
      resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return;
    }
#else
    (void)expert_id;
#endif
  }

  bool is_expert_promoted(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  // Helper to select B pointer for up or gate mat based on packing
  inline int8_t* select_up_or_gate_B_ptr_(bool do_up, int expert_idx, int ith, int devide_elements_size) {
    if constexpr (PLAIN) {
      int8_t* base = do_up ? (int8_t*)up_bb_[expert_idx]->b : (int8_t*)gate_bb_[expert_idx]->b;
      return base + ith * config_.hidden_size * T::N_BLOCK_UP_GATE / devide_elements_size;
    } else {
      return do_up ? (int8_t*)up_bb_[expert_idx]->b_pack[ith] : (int8_t*)gate_bb_[expert_idx]->b_pack[ith];
    }
  }

  // Helper to select B pointer for down mat based on packing
  inline int8_t* select_down_B_ptr_(int expert_idx, int ith, int devide_elements_size) {
    if constexpr (PLAIN) {
      return ((int8_t*)down_bb_[expert_idx]->b) +
             ith * config_.intermediate_size * T::N_BLOCK_DOWN / devide_elements_size;
    } else {
      return (int8_t*)down_bb_[expert_idx]->b_pack[ith];
    }
  }

  // Unified implementation for decode/prefill using mode 'd' or 'p'
  void forward_unified(char mode, int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    MatKernelVariant var = (mode == 'p') ? MatKernelVariant::Prefill : MatKernelVariant::Decode;
    MatKernelSelection kernel = select_mat_kernel<T>(var);
    GemmFn cblas_gemm_s8s8s32 = kernel.fn;
    int devide_elements_size = kernel.divide_elements_size;

#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif
    int max_local_num = 0;

    auto pool = config_.pool->get_subpool(tp_part_idx);

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

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        max_local_num = std::max(max_local_num, m_local_num_[i]);
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("prepare");
#endif

    // Copy inputs into expert-local buffers
    MOE_DIRECT_OR_POOL_BY_VAR(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (input_t*)input + i * config_.hidden_size, sizeof(input_t) * config_.hidden_size);
      }
    });

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("copy_input");
#endif

    // Quantize expert inputs (row-wise)
    {
      size_t mth = T::recommended_mth(max_local_num);
      MOE_DIRECT_OR_POOL_BY_VAR(activated_expert * mth, [&](int task_id) {
        int task_id_expert = task_id / mth;
        int ith = task_id % mth;
        int expert_idx = m_expert_id_map_[task_id_expert];
        if (ith * T::M_BLOCK >= m_local_num_[expert_idx]) return;
        gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], ith, mth);
      });
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("quant_input");
#endif

    int nth_up = T::recommended_nth_up_gate(config_.intermediate_size, mode);
    int mth = T::recommended_mth(max_local_num);
    int32_t oc = 0;

    // Up and Gate GEMMs + dequant scale
    pool->do_work_stealing_job(
        mth * nth_up * activated_expert * 2, nullptr,
        [this, qlen, nth_up, oc, &cblas_gemm_s8s8s32, devide_elements_size, mth](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / (nth_up * mth)];
          task_id = task_id % (nth_up * mth);
          int ith = task_id % nth_up;
          int jth = task_id / nth_up;
          if (jth * T::M_BLOCK >= m_local_num_[expert_idx]) return;
          int m_block = T::M_BLOCK;
          if ((jth + 1) * T::M_BLOCK > m_local_num_[expert_idx]) {
            m_block = m_local_num_[expert_idx] - jth * T::M_BLOCK;
          }
          int8_t* a_ptr = (int8_t*)gate_up_ba_[expert_idx]->a + jth * T::M_BLOCK * config_.hidden_size;
          int8_t* b_ptr = select_up_or_gate_B_ptr_(do_up, expert_idx, ith, devide_elements_size);
          int32_t* c_ptr = (do_up ? (int32_t*)up_bc_[expert_idx]->c : (int32_t*)gate_bc_[expert_idx]->c) +
                           ith * T::N_BLOCK_UP_GATE + jth * T::M_BLOCK * config_.intermediate_size;

          cblas_gemm_s8s8s32(KernelCblasRowMajor, KernelCblasNoTrans, KernelCblasTrans, KernelCblasFixOffset, m_block,
                             T::N_BLOCK_UP_GATE, config_.hidden_size, 1.0, a_ptr, config_.hidden_size, 0, b_ptr,
                             config_.hidden_size, 0, 0.0, c_ptr, config_.intermediate_size, &oc);

          if (do_up) {
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), up_bb_[expert_idx].get(), up_bc_[expert_idx].get(), ith,
                           nth_up, T::N_BLOCK_UP_GATE, jth);
          } else {
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), gate_bb_[expert_idx].get(), gate_bc_[expert_idx].get(), ith,
                           nth_up, T::N_BLOCK_UP_GATE, jth);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("up_gate");
#endif

    // Activate gate and multiply by up
    {
      int nth = T::recommended_nth(config_.intermediate_size);
      auto up_gate_fn = [this, nth](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
          float* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
          float* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
          for (int j = n_start; j < n_end; j++) {
            gate_output_ptr[j] = act_fn(gate_output_ptr[j]) * up_output_ptr[j];
          }
        }
      };
      MOE_DIRECT_OR_POOL_BY_VAR(nth * activated_expert, up_gate_fn);
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("act");
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx]);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("quant_down_input");
#endif

    int nth_down = T::recommended_nth_down(config_.hidden_size, mode);
    pool->do_work_stealing_job(
        mth * nth_down * activated_expert, nullptr,
        [this, qlen, nth_down, oc, &cblas_gemm_s8s8s32, devide_elements_size, mth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / (nth_down * mth)];
          task_id = task_id % (nth_down * mth);
          int ith = task_id % nth_down;
          int jth = task_id / nth_down;
          if (jth * T::M_BLOCK >= m_local_num_[expert_idx]) return;
          int m_block = T::M_BLOCK;
          if ((jth + 1) * T::M_BLOCK > m_local_num_[expert_idx]) {
            m_block = m_local_num_[expert_idx] - jth * T::M_BLOCK;
          }
          int8_t* a_ptr = ((int8_t*)down_ba_[expert_idx]->a) + jth * T::M_BLOCK * config_.intermediate_size;
          int8_t* b_ptr = select_down_B_ptr_(expert_idx, ith, devide_elements_size);
          int32_t* c_ptr =
              ((int32_t*)down_bc_[expert_idx]->c) + ith * T::N_BLOCK_DOWN + jth * T::M_BLOCK * config_.hidden_size;
          cblas_gemm_s8s8s32(KernelCblasRowMajor, KernelCblasNoTrans, KernelCblasTrans, KernelCblasFixOffset, m_block,
                             T::N_BLOCK_DOWN, config_.intermediate_size, 1.0, a_ptr, config_.intermediate_size, 0,
                             b_ptr, config_.intermediate_size, 0, 0.0, c_ptr, config_.hidden_size, &oc);

          T::apply_scale(m_local_num_[expert_idx], config_.hidden_size, m_local_down_output_ptr_[expert_idx],
                         down_ba_[expert_idx].get(), down_bb_[expert_idx].get(), down_bc_[expert_idx].get(), ith,
                         nth_down, T::N_BLOCK_DOWN, jth);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("down");
#endif

    // Merge k experts per token with weights
    size_t block_dim = 512;
    size_t block_num = (config_.hidden_size + block_dim - 1) / block_dim;
    pool->do_work_stealing_job(
        qlen * block_num, nullptr,
        [this, k, expert_ids, weights, output, block_dim, block_num](int i) {
          int q_idx = i / block_num;
          int block_idx = i % block_num;
          int e_start = block_idx * block_dim;
          int e_end =
              ((block_idx + 1) * block_dim) < config_.hidden_size ? ((block_idx + 1) * block_dim) : config_.hidden_size;
          for (int e = e_start; e < e_end; e++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[q_idx * k + j])) {
                continue;
              }
              sum += weights[q_idx * k + j] * ((float*)m_local_down_output_ptr_[expert_ids[q_idx * k + j]])
                                                  [m_local_pos_[q_idx][j] * config_.hidden_size + e];
            }
            ((float*)output)[q_idx * config_.hidden_size + e] = sum;
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    time_perf_name = std::string("[moe] ") + ((mode == 'p') ? "layer prefill" : "decode layer ") +
                     std::to_string(config_.layer_idx) + " tp_part_idx: " + std::to_string(tp_part_idx);
    perf_report();
#endif
  }

  /* merged into forward_unified */
  void forward_decode(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                      void* output) {
    forward_unified('d', qlen, k, expert_ids, weights, input, output);
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    forward_unified('p', qlen, k, expert_ids, weights, input, output);
  }
};

template <typename K, bool T>
class TP_MOE<MOE_KERNEL_TP<K, T>> : public TP_MOE_Common<MOE_KERNEL_TP<K, T>> {
 public:
  using TP_MOE_Common<MOE_KERNEL_TP<K, T>>::TP_MOE_Common;

  void load_weights() {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                             col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }
      if (config.save) {
        // free the bf16 weights after saving
        tps.clear();
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file\n");
      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
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

  void merge_results(int qlen, void* output, bool incremental) {
    // #ifdef FORWARD_TIME_PROFILE
    //     forward_perf_start();
    // #endif
    auto pool = this->config.pool;
    auto merge_fn = [this, output, incremental](int token_nth) {
      auto& local_output_numa = this->local_output_numa;
      auto& tp_configs = this->tp_configs;
      auto& tp_count = this->tp_count;
      auto& config = this->config;
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e++) {
          merge_to[e] += ggml_bf16_to_fp32(((ggml_bf16_t*)output + token_nth * config.hidden_size)[e]);
        }
      }

      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        // TODO: 后续用 SVE 来加速
        // for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
        //   *((__m512 *)(merge_to + e)) = _mm512_add_ps(*((__m512 *)(merge_to + e)), *((__m512 *)(merge_from + e)));
        // }
        // CHECK: 目前用普通的纯 C++ 来实现
        for (int e = 0; e < tp_configs[i].hidden_size; e++) {
          merge_to[e] += merge_from[e];
        }
      }

      convert_or_copy((ggml_bf16_t*)output + token_nth * config.hidden_size, merge_to, config.hidden_size);

      // for (int e = 0; e < config.hidden_size; e += 32) {
      // TODO: 这里需要用 SVE 来加速，实现 fp32 到 bf16 的转换
      // __m512 x0 = *(__m512 *)(merge_to + e);
      // __m512 x1 = *(__m512 *)(merge_to + e + 16);
      // avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)((ggml_bf16_t *)output + token_nth * config.hidden_size + e));

      // CHECK: 目前用普通的纯 C++ 来实现 fp32 到 bf16 的转换

      // convert_32fp32_to_32bf16_pure_c(merge_to + e,
      // (uint16_t *)((ggml_bf16_t *)output + token_nth * config.hidden_size + e));

      // }
    };
    MOE_DIRECT_OR_POOL_BY_VAR(qlen, merge_fn);
    // #ifdef FORWARD_TIME_PROFILE
    //     PROFILE_RECORD_TIME_STAMP("moe merge done");
    // #endif
    // #ifdef FORWARD_TIME_PROFILE
    //     time_perf_name = "[moe merge] decode layer " + std::to_string(this->config.layer_idx);
    //     perf_report();
    // #endif
  }

  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }
};

#endif
