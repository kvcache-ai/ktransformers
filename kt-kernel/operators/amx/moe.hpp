/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_H
#define CPUINFER_OPERATOR_AMX_MOE_H

// #define CHECK
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#ifndef _WIN32
#include <numa.h>
#include <sys/mman.h>
#endif

#include <cstdio>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <thread>

#include "moe_base.hpp"
#include "../../cpu_backend/async_io.hpp"

template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 protected:
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
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

#ifndef _WIN32
  enum ExpertState : uint8_t {
    EXPERT_BASELINE = 0,
    EXPERT_PACKING = 1,
    EXPERT_CACHED = 2,
    EXPERT_PINNED = 3,
    EXPERT_DEMOTING = 4,
    EXPERT_PREFETCHING = 5,
  };

  enum SlotState : uint8_t {
    SLOT_EMPTY = 0,
    SLOT_LOADING = 1,
    SLOT_READY = 2,
  };

  enum MeshSlotMode : uint8_t {
    MESH_SLOT_DECODE_CACHE = 0,
    MESH_SLOT_PREFILL_LAYER = 1,
  };

  struct PendingPrefetch {
    int slot_index = -1;
    std::vector<uint64_t> requests;
  };

  struct BatchPromotion {
    int expert_id = -1;
    int slot = -1;
    void* gate_owner = nullptr;
    void* up_owner = nullptr;
    void* down_owner = nullptr;
    std::vector<uint64_t> requests;
  };

  std::vector<void*> baseline_gate_weight_src_;
  std::vector<void*> baseline_up_weight_src_;
  std::vector<void*> baseline_down_weight_src_;
  std::vector<float*> baseline_gate_scale_src_;
  std::vector<float*> baseline_up_scale_src_;
  std::vector<float*> baseline_down_scale_src_;
  std::vector<float*> baseline_gate_mins_src_;
  std::vector<float*> baseline_up_mins_src_;
  std::vector<float*> baseline_down_mins_src_;

  std::vector<void*> packed_gate_owner_;
  std::vector<void*> packed_up_owner_;
  std::vector<void*> packed_down_owner_;
  std::vector<void*> slot_gate_owner_;
  std::vector<void*> slot_up_owner_;
  std::vector<void*> slot_down_owner_;
  std::vector<int> slot_to_expert_;
  std::vector<int> expert_to_slot_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;
  std::unique_ptr<std::atomic<uint8_t>[]> slot_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> slot_active_readers_;
  std::vector<PendingPrefetch> pending_prefetches_;

  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  bool resident_slot_pool_allocated_ = false;
  MeshSlotMode mesh_slot_mode_ = MESH_SLOT_DECODE_CACHE;
  uint64_t memory_guard_tick_ = 0;
  std::string cgroup_memory_current_path_;
  std::string cgroup_memory_max_path_;
  ResidentCachePolicyState resident_policy_;
  const std::vector<uint8_t>* eviction_protected_mask_ = nullptr;
  bool prefill_streaming_active_ = false;
  size_t gate_total_bytes_ = 0;
  size_t up_total_bytes_ = 0;
  size_t down_total_bytes_ = 0;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  size_t gate_scale_bytes_ = 0;
  size_t up_scale_bytes_ = 0;
  size_t down_scale_bytes_ = 0;
  size_t gate_mins_bytes_ = 0;
  size_t up_mins_bytes_ = 0;
  size_t down_mins_bytes_ = 0;
  std::vector<float> lookahead_heat_;
#endif

#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
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
#ifdef CHECK
  inline void load_check() {
    memcpy(check_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
  }

  void verify_load_right() {
    // printf("varify down bb_0 %d\n", tp_part_idx);
    memcpy(verify_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
    // check if verify_bb_0 equal to check_bb_0
    if (memcmp(verify_bb, check_bb, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)) != 0) {
      printf("verify error\n");
      for (size_t i = 0; i < T::BufferB::required_size(config_.hidden_size, config_.intermediate_size); ++i) {
        if (verify_bb[i] != check_bb[i]) {
          printf("Difference at byte %zu: verify_bb_%d[%zu] = %02x, check_bb[%zu] = %02x\n", i, compare_expers, i,
                 (unsigned char)verify_bb[i], i, (unsigned char)check_bb[i]);
          break;  // find the first difference and exit
        }
      }
      assert(0);
    } else {
      printf("pass verify\n");
      // pick out the 100th~150th byte of scale to see
      printf("numa %d, verify_bb_%d:\n", tp_part_idx, compare_expers);
      size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      size_t scale_size = config_.hidden_size * sizeof(float);
      for (size_t i = size - scale_size; i < size - scale_size + 50; ++i) {
        printf("%02x ", (unsigned char)verify_bb[i]);
      }
      printf("\n");
    }
  }
#endif

#ifdef FORWARD_TIME_REPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> last_now;
#endif

 public:
  AMX_MOE_TP() = default;

  AMX_MOE_TP(GeneralMOEConfig config, int tp_part_idx = 0) : Base(config, tp_part_idx) {
    this->derived_init();
  }

  static constexpr bool buffer_b_has_scale_ptr() {
    return requires(typename T::BufferB* bb) { bb->d; };
  }

  bool iouring_enabled() const { return config_.io_backend == IOBackend::IOURING; }
  bool mmap_enabled() const { return config_.use_mmap && !iouring_enabled(); }
  bool resident_io_enabled() const { return buffer_b_has_scale_ptr() && (mmap_enabled() || iouring_enabled()); }

  void set_weight_buffers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  void derived_init() {
    printf("Creating AMX_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
#ifndef _WIN32
    if (resident_io_enabled()) {
      initialize_resident_io_state();
    }
#endif
    auto& load = config_.load;
    auto& save = config_.save;

    std::filesystem::path prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
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
  }

  ~AMX_MOE_TP() {
#ifndef _WIN32
    if (resident_io_enabled()) {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        if (iouring_enabled() && config_.async_reader != nullptr && !pending_prefetches_[expert_id].requests.empty()) {
          (void)config_.async_reader->wait_for_requests(pending_prefetches_[expert_id].requests, 1000);
        }
        pending_prefetches_[expert_id].requests.clear();
        pending_prefetches_[expert_id].slot_index = -1;
        free_packed_expert(expert_id);
      }
      (void)release_resident_slot_pool(false);
    }
#endif
  }

#ifndef _WIN32
  void initialize_resident_io_state() {
    if constexpr (!buffer_b_has_scale_ptr()) {
      cache_capacity_ = 0;
      return;
    } else {
    const int en = config_.expert_num;
    baseline_gate_weight_src_.assign(en, nullptr);
    baseline_up_weight_src_.assign(en, nullptr);
    baseline_down_weight_src_.assign(en, nullptr);
    baseline_gate_scale_src_.assign(en, nullptr);
    baseline_up_scale_src_.assign(en, nullptr);
    baseline_down_scale_src_.assign(en, nullptr);
    baseline_gate_mins_src_.assign(en, nullptr);
    baseline_up_mins_src_.assign(en, nullptr);
    baseline_down_mins_src_.assign(en, nullptr);
    packed_gate_owner_.assign(en, nullptr);
    packed_up_owner_.assign(en, nullptr);
    packed_down_owner_.assign(en, nullptr);
    pending_prefetches_.assign(en, PendingPrefetch{});
    expert_states_ = std::make_unique<std::atomic<uint8_t>[]>(en);
    active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(en);
    resident_expert_count_.store(0, std::memory_order_relaxed);
    eviction_cursor_.store(0, std::memory_order_relaxed);
    resident_slot_pool_allocated_ = false;
    mesh_slot_mode_ = MESH_SLOT_DECODE_CACHE;

    gate_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    up_total_bytes_ = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    down_total_bytes_ = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    gate_weight_bytes_ = reinterpret_cast<const char*>(gate_bb_[0]->d) - reinterpret_cast<const char*>(gate_bb_[0]->b);
    up_weight_bytes_ = reinterpret_cast<const char*>(up_bb_[0]->d) - reinterpret_cast<const char*>(up_bb_[0]->b);
    down_weight_bytes_ = reinterpret_cast<const char*>(down_bb_[0]->d) - reinterpret_cast<const char*>(down_bb_[0]->b);
    if constexpr (requires { gate_bb_[0]->mins; }) {
      gate_scale_bytes_ = reinterpret_cast<const char*>(gate_bb_[0]->mins) - reinterpret_cast<const char*>(gate_bb_[0]->d);
      up_scale_bytes_ = reinterpret_cast<const char*>(up_bb_[0]->mins) - reinterpret_cast<const char*>(up_bb_[0]->d);
      down_scale_bytes_ = reinterpret_cast<const char*>(down_bb_[0]->mins) - reinterpret_cast<const char*>(down_bb_[0]->d);
      gate_mins_bytes_ = gate_total_bytes_ - gate_weight_bytes_ - gate_scale_bytes_;
      up_mins_bytes_ = up_total_bytes_ - up_weight_bytes_ - up_scale_bytes_;
      down_mins_bytes_ = down_total_bytes_ - down_weight_bytes_ - down_scale_bytes_;
    } else {
      gate_scale_bytes_ = gate_total_bytes_ - gate_weight_bytes_;
      up_scale_bytes_ = up_total_bytes_ - up_weight_bytes_;
      down_scale_bytes_ = down_total_bytes_ - down_weight_bytes_;
    }
    const int configured_resident = config_.max_resident_experts > 0 ? config_.max_resident_experts : config_.max_tier0_experts;
    cache_capacity_ = configured_resident <= 0
                          ? 0
                          : std::min(config_.expert_num, std::max(configured_resident, config_.num_experts_per_tok));
    if (config_.mesh_prefill_layer_mode_enabled && cache_capacity_ > 0) {
      cache_capacity_ = config_.expert_num;
    }
    if (cache_capacity_ > 0) {
      slot_gate_owner_.assign(cache_capacity_, nullptr);
      slot_up_owner_.assign(cache_capacity_, nullptr);
      slot_down_owner_.assign(cache_capacity_, nullptr);
      slot_to_expert_.assign(cache_capacity_, -1);
      expert_to_slot_.assign(en, -1);
      slot_states_ = std::make_unique<std::atomic<uint8_t>[]>(cache_capacity_);
      slot_active_readers_ = std::make_unique<std::atomic<uint32_t>[]>(cache_capacity_);
    } else {
      slot_gate_owner_.clear();
      slot_up_owner_.clear();
      slot_down_owner_.clear();
      slot_to_expert_.clear();
      expert_to_slot_.assign(en, -1);
      slot_states_.reset();
      slot_active_readers_.reset();
    }
    resident_policy_.reset(en, config_.resident_cache_policy);
    lookahead_heat_.assign(en, 0.0f);

    if (config_.pool != nullptr && tp_part_idx < (int)config_.pool->config.subpool_numa_map.size()) {
      numa_node_ = config_.pool->config.subpool_numa_map[tp_part_idx];
    }

    for (int slot = 0; slot < cache_capacity_; ++slot) {
      slot_states_[slot].store(SLOT_EMPTY, std::memory_order_relaxed);
      slot_active_readers_[slot].store(0, std::memory_order_relaxed);
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
      gate_bb_[expert_id]->d = nullptr;
      up_bb_[expert_id]->d = nullptr;
      down_bb_[expert_id]->d = nullptr;
      if constexpr (requires { gate_bb_[expert_id]->mins; }) {
        gate_bb_[expert_id]->mins = nullptr;
        up_bb_[expert_id]->mins = nullptr;
        down_bb_[expert_id]->mins = nullptr;
      }
    }

    if (cache_capacity_ > 0 && !allocate_resident_slot_pool()) {
      throw std::runtime_error("Failed to allocate MESH resident slot pool");
    }
    }
  }

  void apply_baseline_ptrs(int expert_id) {
    gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(baseline_gate_weight_src_[expert_id]);
    up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(baseline_up_weight_src_[expert_id]);
    down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(baseline_down_weight_src_[expert_id]);
    if constexpr (buffer_b_has_scale_ptr()) {
    gate_bb_[expert_id]->d = baseline_gate_scale_src_[expert_id];
    up_bb_[expert_id]->d = baseline_up_scale_src_[expert_id];
    down_bb_[expert_id]->d = baseline_down_scale_src_[expert_id];
    }
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      gate_bb_[expert_id]->mins = baseline_gate_mins_src_[expert_id];
      up_bb_[expert_id]->mins = baseline_up_mins_src_[expert_id];
      down_bb_[expert_id]->mins = baseline_down_mins_src_[expert_id];
    }
  }

  void clear_expert_ptrs(int expert_id) {
    gate_bb_[expert_id]->b = nullptr;
    up_bb_[expert_id]->b = nullptr;
    down_bb_[expert_id]->b = nullptr;
    if constexpr (buffer_b_has_scale_ptr()) {
    gate_bb_[expert_id]->d = nullptr;
    up_bb_[expert_id]->d = nullptr;
    down_bb_[expert_id]->d = nullptr;
    }
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      gate_bb_[expert_id]->mins = nullptr;
      up_bb_[expert_id]->mins = nullptr;
      down_bb_[expert_id]->mins = nullptr;
    }
  }

  void apply_cold_ptrs(int expert_id) {
    if (mmap_enabled()) {
      apply_baseline_ptrs(expert_id);
    } else {
      clear_expert_ptrs(expert_id);
    }
  }

  void apply_owned_ptrs(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    gate_bb_[expert_id]->b = reinterpret_cast<decltype(gate_bb_[expert_id]->b)>(gate_owner);
    up_bb_[expert_id]->b = reinterpret_cast<decltype(up_bb_[expert_id]->b)>(up_owner);
    down_bb_[expert_id]->b = reinterpret_cast<decltype(down_bb_[expert_id]->b)>(down_owner);
    if constexpr (buffer_b_has_scale_ptr()) {
    gate_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_);
    up_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(up_owner) + up_weight_bytes_);
    down_bb_[expert_id]->d = reinterpret_cast<float*>(reinterpret_cast<char*>(down_owner) + down_weight_bytes_);
    }
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      gate_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_ + gate_scale_bytes_);
      up_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(up_owner) + up_weight_bytes_ + up_scale_bytes_);
      down_bb_[expert_id]->mins = reinterpret_cast<float*>(reinterpret_cast<char*>(down_owner) + down_weight_bytes_ + down_scale_bytes_);
    }
  }

  void set_mmap_source_ptrs_quantized(int expert_id, void* gate_weight_src, void* up_weight_src, void* down_weight_src,
                                      float* gate_scale_src, float* up_scale_src, float* down_scale_src,
                                      float* gate_mins_src = nullptr, float* up_mins_src = nullptr,
                                      float* down_mins_src = nullptr) {
    baseline_gate_weight_src_[expert_id] = gate_weight_src;
    baseline_up_weight_src_[expert_id] = up_weight_src;
    baseline_down_weight_src_[expert_id] = down_weight_src;
    baseline_gate_scale_src_[expert_id] = gate_scale_src;
    baseline_up_scale_src_[expert_id] = up_scale_src;
    baseline_down_scale_src_[expert_id] = down_scale_src;
    baseline_gate_mins_src_[expert_id] = gate_mins_src;
    baseline_up_mins_src_[expert_id] = up_mins_src;
    baseline_down_mins_src_[expert_id] = down_mins_src;
    apply_baseline_ptrs(expert_id);
    expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
  }

  void acquire_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_add(1, std::memory_order_acq_rel);
    if (cache_capacity_ > 0 && expert_id < static_cast<int>(expert_to_slot_.size())) {
      const int slot = expert_to_slot_[expert_id];
      if (slot >= 0 && slot < cache_capacity_) {
        slot_active_readers_[slot].fetch_add(1, std::memory_order_acq_rel);
      }
    }
  }

  void release_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_sub(1, std::memory_order_acq_rel);
    if (cache_capacity_ > 0 && expert_id < static_cast<int>(expert_to_slot_.size())) {
      const int slot = expert_to_slot_[expert_id];
      if (slot >= 0 && slot < cache_capacity_) {
        slot_active_readers_[slot].fetch_sub(1, std::memory_order_acq_rel);
      }
    }
  }

  class ExpertReadScope {
   public:
    explicit ExpertReadScope(AMX_MOE_TP* owner, size_t reserve_count = 0) : owner_(owner) {
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
    AMX_MOE_TP* owner_;
    std::vector<int> experts_;
  };

  bool slot_pool_configured() const {
    return cache_capacity_ > 0 && static_cast<int>(slot_to_expert_.size()) == cache_capacity_ &&
           static_cast<int>(expert_to_slot_.size()) >= config_.expert_num;
  }

  bool slot_pool_enabled() const {
    return slot_pool_configured() && resident_slot_pool_allocated_;
  }

  bool lazy_slot_buffers_enabled() const {
    return config_.mesh_prefill_layer_mode_enabled && iouring_enabled();
  }

  bool slot_has_buffers(int slot) const {
    return slot >= 0 && slot < cache_capacity_ && slot_gate_owner_[slot] != nullptr &&
           slot_up_owner_[slot] != nullptr && slot_down_owner_[slot] != nullptr;
  }

  int allocated_slot_count() const {
    int count = 0;
    for (int slot = 0; slot < cache_capacity_; ++slot) {
      if (slot_has_buffers(slot)) count += 1;
    }
    return count;
  }

  void release_expert_buffers(void* gate_owner, void* up_owner, void* down_owner) {
    if (gate_owner != nullptr) numa_free(gate_owner, gate_total_bytes_);
    if (up_owner != nullptr) numa_free(up_owner, up_total_bytes_);
    if (down_owner != nullptr) numa_free(down_owner, down_total_bytes_);
  }

  bool allocate_expert_buffers(void** gate_owner, void** up_owner, void** down_owner) {
    *gate_owner = numa_alloc_onnode(gate_total_bytes_, numa_node_);
    *up_owner = numa_alloc_onnode(up_total_bytes_, numa_node_);
    *down_owner = numa_alloc_onnode(down_total_bytes_, numa_node_);
    if (*gate_owner == nullptr || *up_owner == nullptr || *down_owner == nullptr) {
      release_expert_buffers(*gate_owner, *up_owner, *down_owner);
      *gate_owner = nullptr;
      *up_owner = nullptr;
      *down_owner = nullptr;
      return false;
    }
    return true;
  }

  bool allocate_slot_buffers(int slot) {
    if (slot < 0 || slot >= cache_capacity_) return false;
    if (slot_has_buffers(slot)) return true;
    return allocate_expert_buffers(&slot_gate_owner_[slot], &slot_up_owner_[slot], &slot_down_owner_[slot]);
  }

  void release_slot_buffers(int slot) {
    if (slot < 0 || slot >= cache_capacity_) return;
    release_expert_buffers(slot_gate_owner_[slot], slot_up_owner_[slot], slot_down_owner_[slot]);
    slot_gate_owner_[slot] = nullptr;
    slot_up_owner_[slot] = nullptr;
    slot_down_owner_[slot] = nullptr;
  }

  bool allocate_resident_slot_pool() {
    if (!slot_pool_configured()) return false;
    if (resident_slot_pool_allocated_) return true;
    resident_expert_count_.store(0, std::memory_order_release);
    for (int slot = 0; slot < cache_capacity_; ++slot) {
      if (!lazy_slot_buffers_enabled()) {
        if (!allocate_slot_buffers(slot)) {
          (void)release_resident_slot_pool(false);
          return false;
        }
      }
      slot_to_expert_[slot] = -1;
      slot_states_[slot].store(SLOT_EMPTY, std::memory_order_release);
      slot_active_readers_[slot].store(0, std::memory_order_release);
    }
    resident_slot_pool_allocated_ = true;
    return true;
  }

  bool ensure_resident_slot_pool_allocated() {
    if (!slot_pool_configured()) return false;
    return allocate_resident_slot_pool();
  }

  bool release_resident_slot_pool(bool count_stats = false) {
    if (!slot_pool_configured()) return true;
    for (int slot = 0; slot < static_cast<int>(slot_gate_owner_.size()); ++slot) {
      const int old_expert = slot < static_cast<int>(slot_to_expert_.size()) ? slot_to_expert_[slot] : -1;
      if (old_expert >= 0) {
        const uint8_t old_state = expert_states_[old_expert].load(std::memory_order_acquire);
        if (old_state == EXPERT_PREFETCHING || old_state == EXPERT_PACKING || old_state == EXPERT_DEMOTING) {
          return false;
        }
        if (active_readers_[old_expert].load(std::memory_order_acquire) != 0 ||
            slot_active_readers_[slot].load(std::memory_order_acquire) != 0) {
          return false;
        }
        packed_gate_owner_[old_expert] = nullptr;
        packed_up_owner_[old_expert] = nullptr;
        packed_down_owner_[old_expert] = nullptr;
        expert_to_slot_[old_expert] = -1;
        apply_cold_ptrs(old_expert);
        if (count_stats) {
          note_expert_demote(old_expert);
        }
        expert_states_[old_expert].store(EXPERT_BASELINE, std::memory_order_release);
      }
      release_slot_buffers(slot);
      if (slot < static_cast<int>(slot_to_expert_.size())) slot_to_expert_[slot] = -1;
      slot_states_[slot].store(SLOT_EMPTY, std::memory_order_release);
    }
    for (int expert_id = 0; expert_id < static_cast<int>(expert_to_slot_.size()); ++expert_id) {
      expert_to_slot_[expert_id] = -1;
    }
    resident_expert_count_.store(0, std::memory_order_release);
    resident_slot_pool_allocated_ = false;
    mesh_slot_mode_ = MESH_SLOT_DECODE_CACHE;
    return true;
  }

  void bind_slot_to_expert(int slot, int expert_id, bool pin_after_copy) {
    if (slot < 0 || slot >= cache_capacity_ || expert_id < 0 || expert_id >= config_.expert_num) {
      throw std::runtime_error("Invalid MESH slot bind");
    }
    const int old_expert = slot_to_expert_[slot];
    if (old_expert >= 0 && old_expert != expert_id) {
      throw std::runtime_error("MESH slot bind attempted on an occupied slot");
    }
    slot_to_expert_[slot] = expert_id;
    expert_to_slot_[expert_id] = slot;
    packed_gate_owner_[expert_id] = slot_gate_owner_[slot];
    packed_up_owner_[expert_id] = slot_up_owner_[slot];
    packed_down_owner_[expert_id] = slot_down_owner_[slot];
    apply_owned_ptrs(expert_id, slot_gate_owner_[slot], slot_up_owner_[slot], slot_down_owner_[slot]);
    const bool hard_pin = pin_after_copy && !slot_pool_enabled();
    slot_states_[slot].store(SLOT_READY, std::memory_order_release);
    expert_states_[expert_id].store(hard_pin ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, hard_pin);
    drop_baseline_cache_for_expert(expert_id);
  }

  bool unbind_slot(int slot, bool release_occupancy, bool count_stats = true) {
    if (slot < 0 || slot >= cache_capacity_) return false;
    const int old_expert = slot_to_expert_[slot];
    if (old_expert >= 0) {
      if (active_readers_[old_expert].load(std::memory_order_acquire) != 0 ||
          slot_active_readers_[slot].load(std::memory_order_acquire) != 0) {
        return false;
      }
      packed_gate_owner_[old_expert] = nullptr;
      packed_up_owner_[old_expert] = nullptr;
      packed_down_owner_[old_expert] = nullptr;
      expert_to_slot_[old_expert] = -1;
      apply_cold_ptrs(old_expert);
      if (count_stats) {
        note_expert_demote(old_expert);
      }
      expert_states_[old_expert].store(EXPERT_BASELINE, std::memory_order_release);
      slot_to_expert_[slot] = -1;
    }
    if (release_occupancy) {
      const uint8_t prev = slot_states_[slot].exchange(SLOT_EMPTY, std::memory_order_acq_rel);
      if (prev != SLOT_EMPTY) {
        resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
      }
    }
    return true;
  }

  int find_empty_slot() const {
    if (!slot_pool_enabled()) return -1;
    for (int slot = 0; slot < cache_capacity_; ++slot) {
      if (slot_states_[slot].load(std::memory_order_acquire) == SLOT_EMPTY && slot_to_expert_[slot] < 0) {
        return slot;
      }
    }
    return -1;
  }

  bool reserve_slot_for_loading(int slot, int expert_id) {
    if (slot < 0 || slot >= cache_capacity_ || expert_id < 0 || expert_id >= config_.expert_num) return false;
    const uint8_t state = slot_states_[slot].load(std::memory_order_acquire);
    if (state == SLOT_READY && !unbind_slot(slot, false)) {
      return false;
    }
    if (slot_states_[slot].load(std::memory_order_acquire) == SLOT_EMPTY) {
      if (!allocate_slot_buffers(slot)) {
        return false;
      }
      resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    }
    slot_states_[slot].store(SLOT_LOADING, std::memory_order_release);
    return true;
  }

  int slot_for_expert_or_empty(int expert_id) const {
    if (!slot_pool_enabled() || expert_id < 0 || expert_id >= static_cast<int>(expert_to_slot_.size())) return -1;
    const int slot = expert_to_slot_[expert_id];
    if (slot >= 0 && slot < cache_capacity_) return slot;
    return find_empty_slot();
  }

  std::string resident_debug_summary() const {
    std::ostringstream oss;
    int base = 0, packing = 0, cached = 0, pinned = 0, demoting = 0, prefetching = 0, other = 0;
    int readers = 0;
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      switch (expert_states_[expert_id].load(std::memory_order_acquire)) {
        case EXPERT_BASELINE:
          base++;
          break;
        case EXPERT_PACKING:
          packing++;
          break;
        case EXPERT_CACHED:
          cached++;
          break;
        case EXPERT_PINNED:
          pinned++;
          break;
        case EXPERT_DEMOTING:
          demoting++;
          break;
        case EXPERT_PREFETCHING:
          prefetching++;
          break;
        default:
          other++;
          break;
      }
      if (active_readers_[expert_id].load(std::memory_order_acquire) != 0) readers++;
    }
    oss << " states={base:" << base << ",packing:" << packing << ",cached:" << cached
        << ",pinned:" << pinned << ",demoting:" << demoting << ",prefetching:" << prefetching
        << ",other:" << other << ",readers:" << readers << "}";
    if (slot_pool_enabled()) {
      int empty = 0, loading = 0, ready = 0, slot_readers = 0;
      oss << " slots=[";
      for (int slot = 0; slot < cache_capacity_; ++slot) {
        const uint8_t ss = slot_states_[slot].load(std::memory_order_acquire);
        if (ss == SLOT_EMPTY) empty++;
        else if (ss == SLOT_LOADING) loading++;
        else if (ss == SLOT_READY) ready++;
        if (slot_active_readers_[slot].load(std::memory_order_acquire) != 0) slot_readers++;
        if (slot < 16) {
          if (slot > 0) oss << ",";
          const int expert_id = slot_to_expert_[slot];
          const int estate = expert_id >= 0 ? static_cast<int>(expert_states_[expert_id].load(std::memory_order_acquire)) : -1;
          oss << slot << ":" << expert_id << ":s" << static_cast<int>(ss) << ":e" << estate
              << ":r" << slot_active_readers_[slot].load(std::memory_order_acquire);
        }
      }
      oss << "] slot_counts={empty:" << empty << ",loading:" << loading << ",ready:" << ready
          << ",readers:" << slot_readers << "}";
    }
    return oss.str();
  }

  void free_packed_expert(int expert_id) {
    if (slot_pool_enabled() && expert_id >= 0 && expert_id < static_cast<int>(expert_to_slot_.size())) {
      const int slot = expert_to_slot_[expert_id];
      if (slot >= 0) {
        (void)unbind_slot(slot, true, false);
        return;
      }
    }
    void* gate_owner = packed_gate_owner_[expert_id];
    void* up_owner = packed_up_owner_[expert_id];
    void* down_owner = packed_down_owner_[expert_id];
    packed_gate_owner_[expert_id] = nullptr;
    packed_up_owner_[expert_id] = nullptr;
    packed_down_owner_[expert_id] = nullptr;

    if (gate_owner) {
      numa_free(gate_owner, gate_total_bytes_);
    }
    if (up_owner) {
      numa_free(up_owner, up_total_bytes_);
    }
    if (down_owner) {
      numa_free(down_owner, down_total_bytes_);
    }
  }

  void drop_baseline_cache_for_expert(int expert_id) {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    if (!mmap_enabled()) return;
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
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      if (baseline_gate_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_gate_mins_src_[expert_id], gate_mins_bytes_, MADV_DONTNEED);
      }
      if (baseline_up_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_up_mins_src_[expert_id], up_mins_bytes_, MADV_DONTNEED);
      }
      if (baseline_down_mins_src_[expert_id] != nullptr) {
        madvise((void*)baseline_down_mins_src_[expert_id], down_mins_bytes_, MADV_DONTNEED);
      }
    }
#else
    (void)expert_id;
#endif
  }

  int logical_expert_id_for_slot(int expert_id) const {
    if (expert_id < 0 || expert_id >= config_.expert_num) {
      throw std::runtime_error("Invalid expert id for io_uring slot lookup");
    }
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    const int logical_expert_id =
        physical_to_logical_map == nullptr ? expert_id : static_cast<int>(physical_to_logical_map[expert_id]);
    if (logical_expert_id < 0 || logical_expert_id >= config_.expert_num) {
      std::ostringstream oss;
      oss << "Invalid physical_to_logical_map entry for layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " expert=" << expert_id << " logical=" << logical_expert_id;
      throw std::runtime_error(oss.str());
    }
    return logical_expert_id;
  }

  const ExpertFileSlot& iouring_slot_at(const std::vector<std::vector<ExpertFileSlot>>& slots,
                                        const char* name,
                                        int expert_id) const {
    if (tp_part_idx < 0 || tp_part_idx >= static_cast<int>(slots.size())) {
      std::ostringstream oss;
      oss << "io_uring file slots for " << name << " do not contain tp=" << tp_part_idx
          << " rows=" << slots.size();
      throw std::runtime_error(oss.str());
    }
    const int logical_expert_id = logical_expert_id_for_slot(expert_id);
    const auto& row = slots[tp_part_idx];
    if (logical_expert_id >= static_cast<int>(row.size())) {
      std::ostringstream oss;
      oss << "io_uring file slots for " << name << " do not contain logical expert=" << logical_expert_id
          << " row_size=" << row.size();
      throw std::runtime_error(oss.str());
    }
    return row[logical_expert_id];
  }

  void validate_iouring_slot(const char* name, const ExpertFileSlot& slot, size_t expected_size) const {
    if (slot.fd < 0 || slot.size == 0) {
      std::ostringstream oss;
      oss << "Invalid io_uring slot for " << name << " layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " fd=" << slot.fd << " offset=" << slot.offset << " size=" << slot.size;
      throw std::runtime_error(oss.str());
    }
    if (slot.size != expected_size) {
      std::ostringstream oss;
      oss << "Unexpected io_uring slot size for " << name << " layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " expected=" << expected_size << " actual=" << slot.size << " offset=" << slot.offset;
      throw std::runtime_error(oss.str());
    }
    if (config_.iouring_direct_io && ((slot.offset % 512) != 0 || (slot.size % 512) != 0)) {
      std::ostringstream oss;
      oss << "io_uring O_DIRECT slot for " << name << " is not 512-byte aligned layer=" << config_.layer_idx
          << " tp=" << tp_part_idx << " offset=" << slot.offset << " size=" << slot.size;
      throw std::runtime_error(oss.str());
    }
  }

  void validate_iouring_slot_matrix(const char* name,
                                    const std::vector<std::vector<ExpertFileSlot>>& slots,
                                    size_t expected_size) const {
    if (tp_part_idx < 0 || tp_part_idx >= static_cast<int>(slots.size())) {
      std::ostringstream oss;
      oss << "io_uring backend requires " << name << " slots for tp=" << tp_part_idx
          << " rows=" << slots.size();
      throw std::runtime_error(oss.str());
    }
    if (static_cast<int>(slots[tp_part_idx].size()) < config_.expert_num) {
      std::ostringstream oss;
      oss << "io_uring backend requires " << name << " slots for every expert layer=" << config_.layer_idx
          << " tp=" << tp_part_idx << " experts=" << config_.expert_num
          << " row_size=" << slots[tp_part_idx].size();
      throw std::runtime_error(oss.str());
    }
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      validate_iouring_slot(name, slots[tp_part_idx][expert_id], expected_size);
    }
  }

  void validate_iouring_config() const {
    if (!iouring_enabled()) return;
#ifdef HAVE_LIBURING
    if (config_.async_reader == nullptr) {
      throw std::runtime_error("io_uring backend requires a non-null AsyncExpertReader");
    }
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      (void)logical_expert_id_for_slot(expert_id);
    }
    validate_iouring_slot_matrix("gate.weight", config_.gate_file_slots, gate_weight_bytes_);
    validate_iouring_slot_matrix("gate.scale", config_.gate_scale_file_slots, gate_scale_bytes_);
    validate_iouring_slot_matrix("up.weight", config_.up_file_slots, up_weight_bytes_);
    validate_iouring_slot_matrix("up.scale", config_.up_scale_file_slots, up_scale_bytes_);
    validate_iouring_slot_matrix("down.weight", config_.down_file_slots, down_weight_bytes_);
    validate_iouring_slot_matrix("down.scale", config_.down_scale_file_slots, down_scale_bytes_);
    if constexpr (requires { gate_bb_[0]->mins; }) {
      if (!config_.gate_mins_file_slots.empty()) {
        validate_iouring_slot_matrix("gate.mins", config_.gate_mins_file_slots, gate_mins_bytes_);
      }
      if (!config_.up_mins_file_slots.empty()) {
        validate_iouring_slot_matrix("up.mins", config_.up_mins_file_slots, up_mins_bytes_);
      }
      if (!config_.down_mins_file_slots.empty()) {
        validate_iouring_slot_matrix("down.mins", config_.down_mins_file_slots, down_mins_bytes_);
      }
    }
    std::fprintf(stderr,
                 "[MESHIO] layer=%d tp=%d backend=iouring direct_io=%s mmap_baseline=false capacity=%d policy=%s "
                 "decode_capacity=%d prefill_layer_mode=%s lookahead=%s topk_fallback=%s "
                 "w=%.3f gamma=%.3f beta=%.3f transition=%.3f "
                 "prefetch=%d coldstart=%s coldstart_limit=%d "
                 "mem_guard=%s high=%.3f target=%.3f interval=%d demotes=%d "
                 "gate=%zu+%zu up=%zu+%zu down=%zu+%zu\n",
                 config_.layer_idx,
                 tp_part_idx,
                 config_.iouring_direct_io ? "true" : "false",
                 cache_capacity_,
                 config_.resident_cache_policy.c_str(),
                 config_.mesh_decode_resident_experts,
                 config_.mesh_prefill_layer_mode_enabled ? "true" : "false",
                 config_.mesh_lookahead_enabled ? "true" : "false",
                 config_.mesh_topk_fallback_enabled ? "true" : "false",
                 config_.mesh_lookahead_weight,
                 config_.mesh_heat_gamma,
                 config_.mesh_heat_beta,
                 config_.mesh_transition_alpha,
                 config_.mesh_prefetch_budget,
                 config_.mesh_coldstart_prefill_enabled ? "true" : "false",
                 config_.mesh_coldstart_prefill_limit,
                 config_.mesh_memory_guard_enabled ? "true" : "false",
                 config_.mesh_memory_high_watermark,
                 config_.mesh_memory_target_watermark,
                 config_.mesh_memory_check_interval,
                 config_.mesh_memory_max_demotes_per_check,
                 gate_weight_bytes_,
                 gate_scale_bytes_,
                 up_weight_bytes_,
                 up_scale_bytes_,
                 down_weight_bytes_,
                 down_scale_bytes_);
#else
    throw std::runtime_error("io_uring backend requested but kt_kernel_ext was built without liburing support");
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

  bool mesh_lookahead_active() const {
    return resident_io_enabled() && config_.mesh_lookahead_enabled && config_.mesh_lookahead_weight > 0.0f &&
           static_cast<int>(lookahead_heat_.size()) >= config_.expert_num;
  }

  bool mesh_prefill_stream_trace_enabled() const {
    const char* trace = std::getenv("KT_MESH_PREFILL_STREAM_TRACE");
    return trace != nullptr && trace[0] != '\0' && trace[0] != '0';
  }

  static uint64_t elapsed_us(std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }

  uint64_t iouring_read_bytes_for_expert(int expert_id) const {
    if (!iouring_enabled() || expert_id < 0 || expert_id >= config_.expert_num) return 0;
    uint64_t total = 0;
    auto add_slot = [&](const std::vector<std::vector<ExpertFileSlot>>& slots, const char* name) {
      if (!slots.empty()) {
        const auto& slot = iouring_slot_at(slots, name, expert_id);
        if (slot.fd >= 0 && slot.size > 0) {
          total += static_cast<uint64_t>(slot.size);
        }
      }
    };
    add_slot(config_.gate_file_slots, "gate.weight");
    add_slot(config_.gate_scale_file_slots, "gate.scale");
    add_slot(config_.up_file_slots, "up.weight");
    add_slot(config_.up_scale_file_slots, "up.scale");
    add_slot(config_.down_file_slots, "down.weight");
    add_slot(config_.down_scale_file_slots, "down.scale");
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      add_slot(config_.gate_mins_file_slots, "gate.mins");
      add_slot(config_.up_mins_file_slots, "up.mins");
      add_slot(config_.down_mins_file_slots, "down.mins");
    }
    return total;
  }

  bool is_eviction_protected(int expert_id, const std::vector<uint8_t>* protected_mask) const {
    return protected_mask != nullptr && expert_id >= 0 && expert_id < static_cast<int>(protected_mask->size()) &&
           (*protected_mask)[expert_id] != 0;
  }

  int select_eviction_victim(int exclude_expert_id, const std::vector<uint8_t>* protected_mask = nullptr) {
    const std::vector<uint8_t>* effective_protected_mask =
        protected_mask != nullptr ? protected_mask : eviction_protected_mask_;
    auto is_legal_prefetch_victim = [this, protected_mask](int expert_id) {
      const std::vector<uint8_t>* effective_mask =
          protected_mask != nullptr ? protected_mask : eviction_protected_mask_;
      return expert_id >= 0 && expert_id < config_.expert_num && !is_eviction_protected(expert_id, effective_mask) &&
             expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_CACHED &&
             active_readers_[expert_id].load(std::memory_order_acquire) == 0;
    };

    if (!prefill_streaming_active_ && mesh_lookahead_active()) {
      std::vector<int> order = resident_policy_.build_reclaim_order(
          config_.expert_num, exclude_expert_id, static_cast<uint8_t>(EXPERT_CACHED),
          static_cast<uint8_t>(EXPERT_CACHED),
          [this](int expert_id) { return expert_states_[expert_id].load(std::memory_order_acquire); });
      double best_score = std::numeric_limits<double>::infinity();
      int best = -1;
      const double denom = order.empty() ? 1.0 : static_cast<double>(order.size());
      for (size_t rank = 0; rank < order.size(); ++rank) {
        const int candidate = order[rank];
        if (candidate < 0 || candidate >= config_.expert_num) continue;
        if (candidate == exclude_expert_id) continue;
        if (is_eviction_protected(candidate, effective_protected_mask)) continue;
        if (expert_states_[candidate].load(std::memory_order_acquire) != EXPERT_CACHED) continue;
        if (active_readers_[candidate].load(std::memory_order_acquire) != 0) continue;
        const double policy_score = static_cast<double>(rank) / denom;
        const double heat = std::max(0.0f, lookahead_heat_[candidate]);
        const double score = policy_score + static_cast<double>(config_.mesh_lookahead_weight) * heat;
        if (score < best_score) {
          best_score = score;
          best = candidate;
        }
      }
      if (best >= 0) {
        return best;
      }
    }
    if (effective_protected_mask != nullptr) {
      std::vector<int> order = resident_policy_.build_reclaim_order(
          config_.expert_num, exclude_expert_id, static_cast<uint8_t>(EXPERT_CACHED),
          static_cast<uint8_t>(EXPERT_CACHED),
          [this](int expert_id) { return expert_states_[expert_id].load(std::memory_order_acquire); });
      for (int candidate : order) {
        if (candidate == exclude_expert_id) continue;
        if (is_legal_prefetch_victim(candidate)) return candidate;
      }
      if (protected_mask != nullptr || !prefill_streaming_active_) {
        return -1;
      }
    }
    return resident_policy_.pick_victim(
        config_.expert_num, exclude_expert_id, static_cast<uint8_t>(EXPERT_CACHED),
        [this](int expert_id) { return expert_states_[expert_id].load(std::memory_order_acquire); },
        [this](int expert_id) { return active_readers_[expert_id].load(std::memory_order_acquire); });
  }

  bool evict_one_cached_expert(int exclude_expert_id, const std::vector<uint8_t>* protected_mask = nullptr) {
    if (config_.expert_num <= 1) return false;
    const std::vector<uint8_t>* effective_protected_mask =
        protected_mask != nullptr ? protected_mask : eviction_protected_mask_;

    for (int attempt = 0; attempt < config_.expert_num; ++attempt) {
      const int victim = select_eviction_victim(exclude_expert_id, protected_mask);
      if (victim < 0) {
        return false;
      }
      if (victim == exclude_expert_id) {
        continue;
      }
      if (!(prefill_streaming_active_ && protected_mask == nullptr) &&
          is_eviction_protected(victim, effective_protected_mask)) {
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
      apply_cold_ptrs(victim);
      note_expert_demote(victim);
      if (!slot_pool_enabled()) {
        resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
        expert_states_[victim].store(EXPERT_BASELINE, std::memory_order_release);
      }

      // Record eviction and demote
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->eviction_count.fetch_add(1, std::memory_order_relaxed);
        config_.cache_stats->demote_count.fetch_add(1, std::memory_order_relaxed);
      }

      return true;
    }

    return false;
  }

  bool evict_one_slot_fallback(int exclude_expert_id, const std::vector<uint8_t>* protected_mask = nullptr) {
    if (!slot_pool_enabled()) return false;
    for (int slot = 0; slot < cache_capacity_; ++slot) {
      if (slot_states_[slot].load(std::memory_order_acquire) != SLOT_READY) continue;
      const int victim = slot_to_expert_[slot];
      if (victim < 0 || victim == exclude_expert_id) continue;
      if (is_eviction_protected(victim, protected_mask)) continue;
      if (active_readers_[victim].load(std::memory_order_acquire) != 0 ||
          slot_active_readers_[slot].load(std::memory_order_acquire) != 0) {
        continue;
      }
      uint8_t state = expert_states_[victim].load(std::memory_order_acquire);
      if (state != EXPERT_CACHED && state != EXPERT_PINNED) continue;
      if (!expert_states_[victim].compare_exchange_strong(
              state, EXPERT_DEMOTING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }
      if (!unbind_slot(slot, true, false)) {
        expert_states_[victim].store(state, std::memory_order_release);
        continue;
      }
      note_expert_demote(victim);
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->eviction_count.fetch_add(1, std::memory_order_relaxed);
        config_.cache_stats->demote_count.fetch_add(1, std::memory_order_relaxed);
      }
      return true;
    }
    return false;
  }

  bool reserve_loading_slot_for_expert(int expert_id, int* slot_out,
                                       const std::vector<uint8_t>* protected_mask = nullptr) {
    if (slot_out == nullptr || !slot_pool_enabled()) return false;
    int slot = -1;
    int wait_attempts = 0;
    while ((slot = find_empty_slot()) < 0) {
      complete_ready_prefetches();
      if ((slot = find_empty_slot()) >= 0) {
        break;
      }
      if (!evict_one_cached_expert(expert_id, protected_mask) &&
          !evict_one_slot_fallback(expert_id, protected_mask)) {
        if (++wait_attempts > 2000000) {
          return false;
        }
        if ((wait_attempts & 1023) == 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        std::this_thread::yield();
        continue;
      }
      wait_attempts = 0;
    }
    if (!reserve_slot_for_loading(slot, expert_id)) {
      return false;
    }
    *slot_out = slot;
    return true;
  }

  void finalize_expert_buffers(int expert_id, void* gate_owner, void* up_owner, void* down_owner, bool pin_after_copy) {
    if (slot_pool_enabled()) {
      int slot = -1;
      for (int i = 0; i < cache_capacity_; ++i) {
        if (slot_gate_owner_[i] == gate_owner && slot_up_owner_[i] == up_owner && slot_down_owner_[i] == down_owner) {
          slot = i;
          break;
        }
      }
      if (slot < 0) {
        throw std::runtime_error("MESH finalize could not map owner buffers to a resident slot");
      }
      bind_slot_to_expert(slot, expert_id, pin_after_copy);
      return;
    }
    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    apply_owned_ptrs(expert_id, gate_owner, up_owner, down_owner);
    expert_states_[expert_id].store(pin_after_copy ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_copy);
    // Keep the newly materialized NUMA-local copy as the authoritative hot
    // representation for this expert and immediately drop the mmap-backed
    // baseline pages. They can fault back in later if this expert is demoted.
    drop_baseline_cache_for_expert(expert_id);
  }

#ifdef HAVE_LIBURING
  std::vector<uint64_t> submit_iouring_reads_for_expert(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    if (config_.async_reader == nullptr) {
      return {};
    }

    const auto& gate_slot = iouring_slot_at(config_.gate_file_slots, "gate.weight", expert_id);
    const auto& up_slot = iouring_slot_at(config_.up_file_slots, "up.weight", expert_id);
    const auto& down_slot = iouring_slot_at(config_.down_file_slots, "down.weight", expert_id);
    const auto& gate_scale_slot = iouring_slot_at(config_.gate_scale_file_slots, "gate.scale", expert_id);
    const auto& up_scale_slot = iouring_slot_at(config_.up_scale_file_slots, "up.scale", expert_id);
    const auto& down_scale_slot = iouring_slot_at(config_.down_scale_file_slots, "down.scale", expert_id);

    if (gate_slot.fd < 0 || up_slot.fd < 0 || down_slot.fd < 0 || gate_scale_slot.fd < 0 ||
        up_scale_slot.fd < 0 || down_scale_slot.fd < 0) {
      std::ostringstream oss;
      oss << "AMX io_uring promotion found invalid fd layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " expert=" << expert_id << " gate_fd=" << gate_slot.fd << " up_fd=" << up_slot.fd
          << " down_fd=" << down_slot.fd << " gate_scale_fd=" << gate_scale_slot.fd
          << " up_scale_fd=" << up_scale_slot.fd << " down_scale_fd=" << down_scale_slot.fd;
      throw std::runtime_error(oss.str());
    }

    std::vector<uint64_t> read_requests;
    read_requests.reserve(9);
    auto submit_slot = [&](const ExpertFileSlot& slot, void* dst) {
      if (slot.fd >= 0 && slot.size > 0) {
        read_requests.push_back(config_.async_reader->submit_read(slot.fd, dst, slot.size, slot.offset, expert_id));
        if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->iouring_read_request_count.fetch_add(1, std::memory_order_relaxed);
          config_.cache_stats->iouring_read_bytes.fetch_add(static_cast<uint64_t>(slot.size), std::memory_order_relaxed);
        }
      }
    };

    submit_slot(gate_slot, gate_owner);
    submit_slot(gate_scale_slot, reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_);
    submit_slot(up_slot, up_owner);
    submit_slot(up_scale_slot, reinterpret_cast<char*>(up_owner) + up_weight_bytes_);
    submit_slot(down_slot, down_owner);
    submit_slot(down_scale_slot, reinterpret_cast<char*>(down_owner) + down_weight_bytes_);

    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      if (!config_.gate_mins_file_slots.empty()) {
        submit_slot(iouring_slot_at(config_.gate_mins_file_slots, "gate.mins", expert_id),
                    reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_ + gate_scale_bytes_);
      }
      if (!config_.up_mins_file_slots.empty()) {
        submit_slot(iouring_slot_at(config_.up_mins_file_slots, "up.mins", expert_id),
                    reinterpret_cast<char*>(up_owner) + up_weight_bytes_ + up_scale_bytes_);
      }
      if (!config_.down_mins_file_slots.empty()) {
        submit_slot(iouring_slot_at(config_.down_mins_file_slots, "down.mins", expert_id),
                    reinterpret_cast<char*>(down_owner) + down_weight_bytes_ + down_scale_bytes_);
      }
    }
    return read_requests;
  }
#endif

  bool allocate_and_copy_expert(int expert_id, bool pin_after_copy) {
    if (cache_capacity_ > 0 && !slot_pool_enabled()) {
      (void)ensure_resident_slot_pool_allocated();
    }
    const bool use_slot_pool = slot_pool_enabled();
    int slot = -1;
    void* gate_owner = nullptr;
    void* up_owner = nullptr;
    void* down_owner = nullptr;

    if (use_slot_pool) {
      int wait_attempts = 0;
      while ((slot = find_empty_slot()) < 0) {
        complete_ready_prefetches();
        if ((slot = find_empty_slot()) >= 0) {
          break;
        }
        if (!evict_one_cached_expert(expert_id) && !evict_one_slot_fallback(expert_id)) {
          if (++wait_attempts > 2000000) {
            return false;
          }
          if ((wait_attempts & 1023) == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
          }
          std::this_thread::yield();
          continue;
        }
        wait_attempts = 0;
      }
      if (!reserve_slot_for_loading(slot, expert_id)) {
        return false;
      }
      gate_owner = slot_gate_owner_[slot];
      up_owner = slot_up_owner_[slot];
      down_owner = slot_down_owner_[slot];
    } else if (cache_capacity_ > 0) {
      while (resident_expert_count_.load(std::memory_order_acquire) >= cache_capacity_) {
        if (!evict_one_cached_expert(expert_id)) {
          break;
        }
      }
      // A single MoE forward can touch more unique experts than the steady-state
      // cache capacity. Those experts are protected by ExpertReadScope while the
      // AMX kernel is using them, so there may be no legal victim. Allow
      // temporary over-capacity residency and trim after the forward.
    }

    if (!use_slot_pool && !allocate_expert_buffers(&gate_owner, &up_owner, &down_owner)) {
      return false;
    }

    auto release_on_failure = [&]() {
      if (use_slot_pool) {
        if (slot >= 0) {
          (void)unbind_slot(slot, true, false);
        }
      } else {
        release_expert_buffers(gate_owner, up_owner, down_owner);
      }
    };

    if (!iouring_enabled() &&
        (baseline_gate_weight_src_[expert_id] == nullptr || baseline_up_weight_src_[expert_id] == nullptr ||
         baseline_down_weight_src_[expert_id] == nullptr || baseline_gate_scale_src_[expert_id] == nullptr ||
         baseline_up_scale_src_[expert_id] == nullptr || baseline_down_scale_src_[expert_id] == nullptr)) {
      release_on_failure();
      return false;
    }

    // Load expert weights: io_uring (direct I/O) or mmap (page cache)
    if (iouring_enabled()) {
#ifdef HAVE_LIBURING
      // io_uring path: direct read from SSD to NUMA buffer
      if (config_.async_reader == nullptr) {
        release_on_failure();
        return false;
      }

      std::vector<uint64_t> read_requests = submit_iouring_reads_for_expert(expert_id, gate_owner, up_owner, down_owner);

      // Wait for every tensor fragment (weight, scale, optional mins) to complete.
      const int timeout_ms = 60000;
      if (!config_.async_reader->wait_for_requests(read_requests, timeout_ms)) {
        const auto& gate_slot = iouring_slot_at(config_.gate_file_slots, "gate.weight", expert_id);
        const auto& up_slot = iouring_slot_at(config_.up_file_slots, "up.weight", expert_id);
        const auto& down_slot = iouring_slot_at(config_.down_file_slots, "down.weight", expert_id);
        std::ostringstream oss;
        oss << "AMX io_uring promotion failed layer=" << config_.layer_idx << " tp=" << tp_part_idx
            << " expert=" << expert_id << " logical=" << logical_expert_id_for_slot(expert_id)
            << " requests=" << read_requests.size() << " inflight=" << config_.async_reader->get_inflight_count()
            << " timeout_ms=" << timeout_ms << " gate=(" << gate_slot.fd << "," << gate_slot.offset << ","
            << gate_slot.size << ") up=(" << up_slot.fd << "," << up_slot.offset << "," << up_slot.size
            << ") down=(" << down_slot.fd << "," << down_slot.offset << "," << down_slot.size << ") detail="
            << config_.async_reader->describe_requests(read_requests);
        release_on_failure();
        throw std::runtime_error(oss.str());
      }
#else
      // io_uring not available, fallback to error
      release_on_failure();
      return false;
#endif
    } else {
      // mmap path: prefetch + memcpy from page cache
      madvise(baseline_gate_weight_src_[expert_id], gate_weight_bytes_, MADV_WILLNEED);
      madvise(baseline_up_weight_src_[expert_id], up_weight_bytes_, MADV_WILLNEED);
      madvise(baseline_down_weight_src_[expert_id], down_weight_bytes_, MADV_WILLNEED);
      madvise((void*)baseline_gate_scale_src_[expert_id], gate_scale_bytes_, MADV_WILLNEED);
      madvise((void*)baseline_up_scale_src_[expert_id], up_scale_bytes_, MADV_WILLNEED);
      madvise((void*)baseline_down_scale_src_[expert_id], down_scale_bytes_, MADV_WILLNEED);
      if constexpr (requires { gate_bb_[expert_id]->mins; }) {
        if (baseline_gate_mins_src_[expert_id] != nullptr) {
          madvise((void*)baseline_gate_mins_src_[expert_id], gate_mins_bytes_, MADV_WILLNEED);
        }
        if (baseline_up_mins_src_[expert_id] != nullptr) {
          madvise((void*)baseline_up_mins_src_[expert_id], up_mins_bytes_, MADV_WILLNEED);
        }
        if (baseline_down_mins_src_[expert_id] != nullptr) {
          madvise((void*)baseline_down_mins_src_[expert_id], down_mins_bytes_, MADV_WILLNEED);
        }
      }

      std::memcpy(gate_owner, baseline_gate_weight_src_[expert_id], gate_weight_bytes_);
      std::memcpy(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_, baseline_gate_scale_src_[expert_id], gate_scale_bytes_);
      std::memcpy(up_owner, baseline_up_weight_src_[expert_id], up_weight_bytes_);
      std::memcpy(reinterpret_cast<char*>(up_owner) + up_weight_bytes_, baseline_up_scale_src_[expert_id], up_scale_bytes_);
      std::memcpy(down_owner, baseline_down_weight_src_[expert_id], down_weight_bytes_);
      std::memcpy(reinterpret_cast<char*>(down_owner) + down_weight_bytes_, baseline_down_scale_src_[expert_id], down_scale_bytes_);

      if constexpr (requires { gate_bb_[expert_id]->mins; }) {
        if (gate_mins_bytes_ > 0 && baseline_gate_mins_src_[expert_id] != nullptr) {
          std::memcpy(reinterpret_cast<char*>(gate_owner) + gate_weight_bytes_ + gate_scale_bytes_,
                      baseline_gate_mins_src_[expert_id], gate_mins_bytes_);
        }
        if (up_mins_bytes_ > 0 && baseline_up_mins_src_[expert_id] != nullptr) {
          std::memcpy(reinterpret_cast<char*>(up_owner) + up_weight_bytes_ + up_scale_bytes_,
                      baseline_up_mins_src_[expert_id], up_mins_bytes_);
        }
        if (down_mins_bytes_ > 0 && baseline_down_mins_src_[expert_id] != nullptr) {
          std::memcpy(reinterpret_cast<char*>(down_owner) + down_weight_bytes_ + down_scale_bytes_,
                      baseline_down_mins_src_[expert_id], down_mins_bytes_);
        }
      }
    }

    if (!use_slot_pool) {
      resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    }
    finalize_expert_buffers(expert_id, gate_owner, up_owner, down_owner, pin_after_copy);
    return true;
  }

  bool batch_ensure_experts_ready(const std::vector<int>& expert_ids,
                                  bool pin = false,
                                  const std::vector<uint8_t>* protected_mask = nullptr) {
    if (expert_ids.empty()) return true;
    if (cache_capacity_ > 0 && !slot_pool_enabled()) {
      (void)ensure_resident_slot_pool_allocated();
    }
    if (!resident_io_enabled() || !iouring_enabled() || config_.async_reader == nullptr || !slot_pool_enabled()) {
      for (int expert_id : expert_ids) {
        ensure_expert_ready(expert_id, pin);
      }
      return true;
    }

#ifndef HAVE_LIBURING
    for (int expert_id : expert_ids) {
      ensure_expert_ready(expert_id, pin);
    }
    return true;
#else
    const bool batch_trace = mesh_prefill_stream_trace_enabled();
    const auto batch_start = std::chrono::steady_clock::now();
    const uint64_t trace_read_reqs_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
            : 0;
    const uint64_t trace_read_bytes_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
            : 0;
    int trace_hits = 0;
    int trace_cold = 0;
    int trace_inflight = 0;
    int trace_unique = 0;
    uint64_t trace_expected_cold_read_bytes = 0;
    uint64_t trace_wait_us = 0;

    complete_ready_prefetches();

    std::vector<uint8_t> local_protected_mask;
    const std::vector<uint8_t>* effective_protected_mask = protected_mask;
    if (effective_protected_mask == nullptr) {
      local_protected_mask.assign(config_.expert_num, 0);
      for (int expert_id : expert_ids) {
        if (expert_id >= 0 && expert_id < config_.expert_num && !config_.should_skip_expert(expert_id)) {
          local_protected_mask[expert_id] = 1;
        }
      }
      effective_protected_mask = &local_protected_mask;
    }

    std::vector<uint8_t> seen(config_.expert_num, 0);
    std::vector<BatchPromotion> cold_promotions;
    std::vector<int> pending_prefetch_experts;
    std::vector<uint64_t> all_requests;

    auto record_total_access = [&](int expert_id) {
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->total_access_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_access(expert_id);
        }
      }
    };
    auto record_hit = [&](int expert_id) {
      trace_hits += 1;
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->hit_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_hit(expert_id);
        }
      }
    };
    auto record_cold_miss = [&](int expert_id) {
      trace_cold += 1;
      trace_expected_cold_read_bytes += iouring_read_bytes_for_expert(expert_id);
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->miss_count.fetch_add(1, std::memory_order_relaxed);
        config_.cache_stats->cold_miss_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_miss(expert_id);
          config_.cache_stats->note_expert_cold_miss(expert_id);
        }
      }
    };
    auto record_inflight_miss = [&](int expert_id) {
      trace_inflight += 1;
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->miss_count.fetch_add(1, std::memory_order_relaxed);
        config_.cache_stats->in_flight_miss_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_miss(expert_id);
          config_.cache_stats->note_expert_in_flight_miss(expert_id);
        }
      }
    };

    for (int expert_id : expert_ids) {
      if (expert_id < 0 || expert_id >= config_.expert_num || config_.should_skip_expert(expert_id)) {
        continue;
      }
      if (seen[expert_id]) {
        continue;
      }
      seen[expert_id] = 1;
      trace_unique += 1;
      record_total_access(expert_id);

      for (;;) {
        const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
        if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
          record_hit(expert_id);
          break;
        }
        if (state == EXPERT_PREFETCHING) {
          record_inflight_miss(expert_id);
          if (!pending_prefetches_[expert_id].requests.empty()) {
            pending_prefetch_experts.push_back(expert_id);
            all_requests.insert(all_requests.end(),
                                pending_prefetches_[expert_id].requests.begin(),
                                pending_prefetches_[expert_id].requests.end());
            break;
          }
          complete_ready_prefetches();
          std::this_thread::yield();
          continue;
        }
        if (state == EXPERT_BASELINE) {
          uint8_t expected = EXPERT_BASELINE;
          if (!expert_states_[expert_id].compare_exchange_strong(
                  expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
          }

          record_cold_miss(expert_id);
          BatchPromotion promotion;
          promotion.expert_id = expert_id;
          if (!reserve_loading_slot_for_expert(expert_id, &promotion.slot, effective_protected_mask)) {
            expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
            std::ostringstream oss;
            oss << "AMX batch promotion could not reserve slot layer=" << config_.layer_idx
                << " tp=" << tp_part_idx << " expert=" << expert_id
                << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
                << " capacity=" << cache_capacity_ << resident_debug_summary();
            throw std::runtime_error(oss.str());
          }
          promotion.gate_owner = slot_gate_owner_[promotion.slot];
          promotion.up_owner = slot_up_owner_[promotion.slot];
          promotion.down_owner = slot_down_owner_[promotion.slot];
          try {
            promotion.requests = submit_iouring_reads_for_expert(expert_id,
                                                                 promotion.gate_owner,
                                                                 promotion.up_owner,
                                                                 promotion.down_owner);
          } catch (...) {
            (void)unbind_slot(promotion.slot, true, false);
            expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
            throw;
          }
          if (promotion.requests.empty()) {
            (void)unbind_slot(promotion.slot, true, false);
            expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
            std::ostringstream oss;
            oss << "AMX batch promotion submitted no io_uring reads layer=" << config_.layer_idx
                << " tp=" << tp_part_idx << " expert=" << expert_id;
            throw std::runtime_error(oss.str());
          }
          all_requests.insert(all_requests.end(), promotion.requests.begin(), promotion.requests.end());
          cold_promotions.push_back(std::move(promotion));
          break;
        }
        if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
          std::this_thread::yield();
          continue;
        }
      }
    }

    if (!all_requests.empty()) {
      const int timeout_ms = 60000;
      const auto wait_start = std::chrono::steady_clock::now();
      const bool wait_ok = config_.async_reader->wait_for_requests(all_requests, timeout_ms);
      const auto wait_end = std::chrono::steady_clock::now();
      trace_wait_us = elapsed_us(wait_start, wait_end);
      if (!wait_ok) {
        std::ostringstream oss;
        oss << "AMX io_uring batch promotion failed layer=" << config_.layer_idx << " tp=" << tp_part_idx
            << " experts=" << cold_promotions.size() << " pending=" << pending_prefetch_experts.size()
            << " requests=" << all_requests.size() << " inflight=" << config_.async_reader->get_inflight_count()
            << " timeout_ms=" << timeout_ms << " detail="
            << config_.async_reader->describe_requests(all_requests);
        for (auto& promotion : cold_promotions) {
          if (promotion.slot >= 0) {
            (void)unbind_slot(promotion.slot, true, false);
          }
          if (promotion.expert_id >= 0 && promotion.expert_id < config_.expert_num) {
            expert_states_[promotion.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          }
        }
        for (int expert_id : pending_prefetch_experts) {
          clear_pending_prefetch(expert_id, true);
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        }
        throw std::runtime_error(oss.str());
      }
    }

    for (auto& promotion : cold_promotions) {
      finalize_expert_buffers(promotion.expert_id,
                              promotion.gate_owner,
                              promotion.up_owner,
                              promotion.down_owner,
                              pin);
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->promote_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_promote(promotion.expert_id);
        }
      }
    }

    for (int expert_id : pending_prefetch_experts) {
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->prefetch_hit_count.fetch_add(1, std::memory_order_relaxed);
        if (tp_part_idx == 0) {
          config_.cache_stats->note_expert_prefetch_hit(expert_id);
        }
      }
      finalize_pending_prefetch(expert_id, pin);
    }

    if (batch_trace) {
      const auto batch_end = std::chrono::steady_clock::now();
      const uint64_t trace_read_reqs_after =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
              : trace_read_reqs_before;
      const uint64_t trace_read_bytes_after =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
              : trace_read_bytes_before;
      std::fprintf(stderr,
                   "[MESH_BATCH_ENSURE_TRACE] layer=%d tp=%d requested=%zu unique=%d hits=%d cold=%d inflight=%d "
                   "cold_promotions=%zu pending_prefetch=%zu wait_us=%llu total_us=%llu "
                   "read_req_delta=%llu read_bytes_delta=%llu expected_cold_read_bytes=%llu resident=%d capacity=%d\n",
                   config_.layer_idx,
                   tp_part_idx,
                   expert_ids.size(),
                   trace_unique,
                   trace_hits,
                   trace_cold,
                   trace_inflight,
                   cold_promotions.size(),
                   pending_prefetch_experts.size(),
                   static_cast<unsigned long long>(trace_wait_us),
                   static_cast<unsigned long long>(elapsed_us(batch_start, batch_end)),
                   static_cast<unsigned long long>(trace_read_reqs_after - trace_read_reqs_before),
                   static_cast<unsigned long long>(trace_read_bytes_after - trace_read_bytes_before),
                   static_cast<unsigned long long>(trace_expected_cold_read_bytes),
                   resident_expert_count_.load(std::memory_order_acquire),
                   cache_capacity_);
    }

    return true;
#endif
  }

  void trim_cache_to_capacity() {
    if (cache_capacity_ <= 0) return;
    const int target_capacity =
        mesh_slot_mode_ == MESH_SLOT_DECODE_CACHE ? decode_cache_capacity() : cache_capacity_;
    for (int attempt = 0; attempt < config_.expert_num * 2 &&
                          resident_expert_count_.load(std::memory_order_acquire) > target_capacity;
         ++attempt) {
      if (!evict_one_cached_expert(-1)) {
        break;
      }
    }
  }

  int cpu_managed_expert_count() const {
    int count = 0;
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (!config_.should_skip_expert(expert_id)) count += 1;
    }
    return count;
  }

  int decode_cache_capacity() const {
    const int configured =
        config_.mesh_decode_resident_experts > 0 ? config_.mesh_decode_resident_experts : cache_capacity_;
    if (configured <= 0) return 0;
    return std::min(cpu_managed_expert_count(), std::min(config_.expert_num, std::max(configured, 0)));
  }

  void release_empty_slot_buffers_to_limit(int allocated_limit) {
    if (!lazy_slot_buffers_enabled()) return;
    allocated_limit = std::max(0, std::min(allocated_limit, cache_capacity_));
    int allocated = allocated_slot_count();
    if (allocated <= allocated_limit) return;
    for (int slot = cache_capacity_ - 1; slot >= 0 && allocated > allocated_limit; --slot) {
      if (slot_states_[slot].load(std::memory_order_acquire) != SLOT_EMPTY) continue;
      if (slot_to_expert_[slot] >= 0) continue;
      if (!slot_has_buffers(slot)) continue;
      release_slot_buffers(slot);
      allocated -= 1;
    }
  }

  std::vector<int> cpu_managed_experts_by_heat() {
    refresh_lookahead_heat();
    std::vector<int> experts;
    experts.reserve(config_.expert_num);
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (!config_.should_skip_expert(expert_id)) {
        experts.push_back(expert_id);
      }
    }
    std::sort(experts.begin(), experts.end(), [this](int lhs, int rhs) {
      const float lh = lhs < static_cast<int>(lookahead_heat_.size()) ? lookahead_heat_[lhs] : 0.0f;
      const float rh = rhs < static_cast<int>(lookahead_heat_.size()) ? lookahead_heat_[rhs] : 0.0f;
      if (lh != rh) return lh > rh;
      return lhs < rhs;
    });
    return experts;
  }

  int submit_decode_heat_prefetches(int target_capacity, int fill_limit) {
    if (!iouring_enabled() || config_.async_reader == nullptr || target_capacity <= 0) return 0;
    if (!slot_pool_enabled() && !ensure_resident_slot_pool_allocated()) return 0;
    const int resident_now = resident_expert_count_.load(std::memory_order_acquire);
    int remaining = std::max(0, target_capacity - resident_now);
    if (fill_limit > 0) {
      remaining = std::min(remaining, fill_limit);
    }
    if (remaining <= 0) return 0;

    std::vector<int> ranked = cpu_managed_experts_by_heat();
    std::vector<uint8_t> protected_mask(config_.expert_num, 0);
    for (int i = 0; i < static_cast<int>(ranked.size()) && i < target_capacity; ++i) {
      protected_mask[ranked[i]] = 1;
    }

    int submitted = 0;
    for (int expert_id : ranked) {
      if (submitted >= remaining) break;
      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state != EXPERT_BASELINE) continue;
      if (submit_async_prefetch(expert_id, &protected_mask)) {
        submitted += 1;
      }
    }
    if (submitted > 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->prefetch_count.fetch_add(static_cast<uint64_t>(submitted), std::memory_order_relaxed);
    }
    return submitted;
  }

  int prepare_prefill_layer_window() {
    if (!config_.mesh_prefill_layer_mode_enabled || !resident_io_enabled() || cache_capacity_ <= 0) return 0;
    if (!slot_pool_enabled() && !ensure_resident_slot_pool_allocated()) return 0;
    mesh_slot_mode_ = MESH_SLOT_PREFILL_LAYER;

    std::vector<int> cpu_experts;
    cpu_experts.reserve(config_.expert_num);
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (!config_.should_skip_expert(expert_id)) {
        cpu_experts.push_back(expert_id);
      }
    }
    batch_ensure_experts_ready(cpu_experts, false);
    return static_cast<int>(cpu_experts.size());
  }

  bool release_prefill_layer_window() {
    if (!config_.mesh_prefill_layer_mode_enabled || !resident_io_enabled()) return true;
    complete_ready_prefetches();
    return release_resident_slot_pool(true);
  }

  int transition_to_decode_cache(int requested_decode_capacity, int fill_limit) {
    if (!resident_io_enabled() || cache_capacity_ <= 0) return 0;
    if (!slot_pool_enabled() && !ensure_resident_slot_pool_allocated()) return 0;
    complete_ready_prefetches();
    mesh_slot_mode_ = MESH_SLOT_DECODE_CACHE;
    if (requested_decode_capacity > 0) {
      config_.mesh_decode_resident_experts = requested_decode_capacity;
    }
    const int target_capacity = decode_cache_capacity();
    trim_cache_to_capacity();
    release_empty_slot_buffers_to_limit(target_capacity);
    const int submitted = submit_decode_heat_prefetches(target_capacity, fill_limit);
    return submitted;
  }

  bool init_cgroup_memory_paths() {
    if (!cgroup_memory_current_path_.empty() && !cgroup_memory_max_path_.empty()) return true;

    std::ifstream cgroup_file("/proc/self/cgroup");
    std::string line;
    while (std::getline(cgroup_file, line)) {
      const std::string marker = "0::";
      if (line.rfind(marker, 0) != 0) continue;
      std::string rel = line.substr(marker.size());
      if (rel.empty() || rel[0] != '/') rel = "/" + rel;
      const std::string base = "/sys/fs/cgroup" + rel;
      cgroup_memory_current_path_ = base + "/memory.current";
      cgroup_memory_max_path_ = base + "/memory.max";
      return true;
    }
    return false;
  }

  bool read_uint64_file(const std::string& path, uint64_t* value, bool allow_max = false) {
    if (value == nullptr || path.empty()) return false;
    std::ifstream in(path);
    std::string text;
    if (!(in >> text)) return false;
    if (allow_max && text == "max") {
      *value = 0;
      return true;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || parsed == 0ULL) return false;
    *value = static_cast<uint64_t>(parsed);
    return true;
  }

  bool read_cgroup_memory_scope(uint64_t* current, uint64_t* limit) {
    if (!init_cgroup_memory_paths()) return false;
    uint64_t cur = 0;
    uint64_t max = 0;
    if (!read_uint64_file(cgroup_memory_current_path_, &cur)) return false;
    if (!read_uint64_file(cgroup_memory_max_path_, &max, true)) return false;
    if (max == 0) return false;
    if (current != nullptr) *current = cur;
    if (limit != nullptr) *limit = max;
    return true;
  }

  bool memory_guard_pressure_active() {
    if (!config_.mesh_memory_guard_enabled || cache_capacity_ <= 0) return false;
    uint64_t current = 0;
    uint64_t limit = 0;
    if (!read_cgroup_memory_scope(&current, &limit)) return false;
    const double high = std::max(0.0, std::min(1.0, static_cast<double>(config_.mesh_memory_high_watermark)));
    return static_cast<double>(current) >= static_cast<double>(limit) * high;
  }

  void trim_cache_for_memory_pressure() {
    if (!config_.mesh_memory_guard_enabled || cache_capacity_ <= 0) return;
    const int interval = std::max(1, config_.mesh_memory_check_interval);
    memory_guard_tick_ += 1;
    if ((memory_guard_tick_ % static_cast<uint64_t>(interval)) != 0) return;

    uint64_t current = 0;
    uint64_t limit = 0;
    if (!read_cgroup_memory_scope(&current, &limit)) return;

    const double high = std::max(0.0, std::min(1.0, static_cast<double>(config_.mesh_memory_high_watermark)));
    const double target = std::max(0.0, std::min(high, static_cast<double>(config_.mesh_memory_target_watermark)));
    if (static_cast<double>(current) < static_cast<double>(limit) * high) return;

    const int max_demotes = std::max(1, config_.mesh_memory_max_demotes_per_check);
    int demoted = 0;
    for (int attempt = 0; attempt < max_demotes; ++attempt) {
      if (!evict_one_cached_expert(-1)) break;
      demoted += 1;
      if ((attempt & 3) == 3) {
        if (!read_cgroup_memory_scope(&current, &limit)) break;
        if (static_cast<double>(current) <= static_cast<double>(limit) * target) break;
      }
    }
    if (demoted > 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->memory_guard_demote_count.fetch_add(static_cast<uint64_t>(demoted),
                                                               std::memory_order_relaxed);
    }
  }

  bool pending_prefetch_finished(int expert_id, bool* failed) const {
    if (failed != nullptr) *failed = false;
    const auto& pending = pending_prefetches_[expert_id];
    if (pending.requests.empty() || config_.async_reader == nullptr) return false;
    for (uint64_t request_id : pending.requests) {
      const int result = config_.async_reader->get_request_result(request_id);
      if (result == std::numeric_limits<int>::min()) {
        return false;
      }
      if (!config_.async_reader->request_succeeded(request_id)) {
        if (failed != nullptr) *failed = true;
        return false;
      }
    }
    return true;
  }

  void clear_pending_prefetch(int expert_id, bool release_buffers) {
    auto& pending = pending_prefetches_[expert_id];
    if (release_buffers && pending.slot_index >= 0) {
      (void)unbind_slot(pending.slot_index, true, false);
    }
    pending.slot_index = -1;
    pending.requests.clear();
  }

  bool finalize_pending_prefetch(int expert_id, bool pin_after_wait) {
    auto& pending = pending_prefetches_[expert_id];
    const int slot = pending.slot_index;
    if (slot < 0 || slot >= cache_capacity_) {
      clear_pending_prefetch(expert_id, true);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }
    void* gate_owner = slot_gate_owner_[slot];
    void* up_owner = slot_up_owner_[slot];
    void* down_owner = slot_down_owner_[slot];
    pending.slot_index = -1;
    pending.requests.clear();
    finalize_expert_buffers(expert_id, gate_owner, up_owner, down_owner, pin_after_wait);
    return true;
  }

  bool wait_for_pending_prefetch(int expert_id, bool pin_after_wait) {
    if (!iouring_enabled() || config_.async_reader == nullptr) return false;
    const auto& requests = pending_prefetches_[expert_id].requests;
    if (requests.empty()) return false;
    const int timeout_ms = 60000;
    if (!config_.async_reader->wait_for_requests(requests, timeout_ms)) {
      std::ostringstream oss;
      oss << "AMX io_uring prefetch wait failed layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " expert=" << expert_id << " requests=" << requests.size()
          << " detail=" << config_.async_reader->describe_requests(requests);
      clear_pending_prefetch(expert_id, true);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      throw std::runtime_error(oss.str());
    }
    if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->prefetch_hit_count.fetch_add(1, std::memory_order_relaxed);
      if (tp_part_idx == 0) {
        config_.cache_stats->note_expert_prefetch_hit(expert_id);
      }
    }
    return finalize_pending_prefetch(expert_id, pin_after_wait);
  }

  void complete_ready_prefetches() {
    if (!iouring_enabled() || config_.async_reader == nullptr) return;
#ifdef HAVE_LIBURING
    (void)config_.async_reader->poll_completions();
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      if (expert_states_[expert_id].load(std::memory_order_acquire) != EXPERT_PREFETCHING) continue;
      bool failed = false;
      if (pending_prefetch_finished(expert_id, &failed)) {
        finalize_pending_prefetch(expert_id, false);
      } else if (failed) {
        clear_pending_prefetch(expert_id, true);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      }
    }
#endif
  }

  bool submit_async_prefetch(int expert_id, const std::vector<uint8_t>* protected_mask = nullptr) {
    if (!iouring_enabled() || config_.async_reader == nullptr) return false;
    if (cache_capacity_ > 0 && !slot_pool_enabled()) {
      if (!ensure_resident_slot_pool_allocated()) return false;
    }
    if (!slot_pool_enabled()) return false;
#ifndef HAVE_LIBURING
    return false;
#else
    if (expert_id < 0 || expert_id >= config_.expert_num || config_.should_skip_expert(expert_id)) return false;
    int slot = find_empty_slot();
    while (slot < 0) {
      if (!evict_one_cached_expert(expert_id, protected_mask)) {
        return false;
      }
      slot = find_empty_slot();
    }

    uint8_t expected = EXPERT_BASELINE;
    if (!expert_states_[expert_id].compare_exchange_strong(
            expected, EXPERT_PREFETCHING, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return false;
    }

    if (!reserve_slot_for_loading(slot, expert_id)) {
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      return false;
    }
    void* gate_owner = slot_gate_owner_[slot];
    void* up_owner = slot_up_owner_[slot];
    void* down_owner = slot_down_owner_[slot];

    try {
      std::vector<uint64_t> requests = submit_iouring_reads_for_expert(expert_id, gate_owner, up_owner, down_owner);
      if (requests.empty()) {
        (void)unbind_slot(slot, true, false);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        return false;
      }
      auto& pending = pending_prefetches_[expert_id];
      pending.slot_index = slot;
      pending.requests = std::move(requests);
      if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->async_prefetch_count.fetch_add(1, std::memory_order_relaxed);
      }
      return true;
    } catch (...) {
      (void)unbind_slot(slot, true, false);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      throw;
    }
#endif
  }

  void ensure_expert_ready(int expert_id, bool pin = false) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;

    // Record total access
    if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->total_access_count.fetch_add(1, std::memory_order_relaxed);
      if (tp_part_idx == 0) {
        config_.cache_stats->note_expert_access(expert_id);
      }
    }

    bool counted_miss_for_this_access = false;
    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        // Cache hit
        if (!counted_miss_for_this_access && config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->hit_count.fetch_add(1, std::memory_order_relaxed);
          if (tp_part_idx == 0) {
            config_.cache_stats->note_expert_hit(expert_id);
          }
        }
        return;
      }
      if (state == EXPERT_CACHED) {
        // Cache hit
        if (!counted_miss_for_this_access && config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->hit_count.fetch_add(1, std::memory_order_relaxed);
          if (tp_part_idx == 0) {
            config_.cache_stats->note_expert_hit(expert_id);
          }
        }
        if (!pin) {
          return;
        }
        if (slot_pool_enabled()) {
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
        // Cache miss - need to promote
        if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->miss_count.fetch_add(1, std::memory_order_relaxed);
          config_.cache_stats->cold_miss_count.fetch_add(1, std::memory_order_relaxed);
          if (tp_part_idx == 0) {
            config_.cache_stats->note_expert_miss(expert_id);
            config_.cache_stats->note_expert_cold_miss(expert_id);
          }
        }
        counted_miss_for_this_access = true;
        uint8_t expected = EXPERT_BASELINE;
        if (!expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          continue;
        }
        if (!allocate_and_copy_expert(expert_id, pin)) {
          expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
          std::ostringstream oss;
          oss << "AMX lazy-copy promotion failed layer=" << config_.layer_idx << " tp=" << tp_part_idx
              << " expert=" << expert_id << " state=" << static_cast<int>(state)
              << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
              << " capacity=" << cache_capacity_ << " gate_bytes=" << gate_total_bytes_
              << " up_bytes=" << up_total_bytes_ << " down_bytes=" << down_total_bytes_
              << " iouring=" << (iouring_enabled() ? "true" : "false")
              << resident_debug_summary();
          throw std::runtime_error(oss.str());
        }
        // Record promote
        if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->promote_count.fetch_add(1, std::memory_order_relaxed);
          if (tp_part_idx == 0) {
            config_.cache_stats->note_expert_promote(expert_id);
          }
        }
        continue;
      }
      if (state == EXPERT_PREFETCHING) {
        if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
          config_.cache_stats->miss_count.fetch_add(1, std::memory_order_relaxed);
          config_.cache_stats->in_flight_miss_count.fetch_add(1, std::memory_order_relaxed);
          if (tp_part_idx == 0) {
            config_.cache_stats->note_expert_miss(expert_id);
            config_.cache_stats->note_expert_in_flight_miss(expert_id);
          }
        }
        counted_miss_for_this_access = true;
        wait_for_pending_prefetch(expert_id, pin);
        continue;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        std::this_thread::yield();
        continue;
      }
    }
  }

  void refresh_lookahead_heat() {
    if (!resident_io_enabled() || !config_.mesh_lookahead_enabled) return;
    lookahead_heat_ = mesh_lookahead_registry().snapshot_for_layer(config_.layer_idx, config_.expert_num);
  }

  void update_lookahead_heat_from_forward(int qlen, int k, const int64_t* expert_ids, const float* weights) {
    if (!resident_io_enabled() || !config_.mesh_lookahead_enabled) return;
    if (!config_.mesh_topk_fallback_enabled) return;
    if (tp_part_idx != 0) return;
    mesh_lookahead_registry().observe_topk(config_.layer_idx,
                                           config_.expert_num,
                                           expert_ids,
                                           weights,
                                           qlen * k,
                                           config_.gpu_experts_mask,
                                           config_.mesh_heat_gamma,
                                           config_.mesh_heat_beta,
                                           config_.mesh_transition_alpha);
    if (config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->lookahead_update_count.fetch_add(1, std::memory_order_relaxed);
      if (config_.mesh_transition_alpha > 0.0f) {
        config_.cache_stats->transition_update_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

	  int prefetch_experts(int count,
                       const int64_t* expert_ids,
                       int protect_count = 0,
                       const int64_t* protect_ids = nullptr,
                       int max_to_submit = 0,
                       int prefetch_kind = 0) {
    if (!resident_io_enabled() || !iouring_enabled() || expert_ids == nullptr || count <= 0 || cache_capacity_ <= 0) {
      return 0;
    }
    complete_ready_prefetches();
    const bool is_bootstrap_prefetch = prefetch_kind == 1;
    uint64_t bootstrap_candidate_count = 0;
    uint64_t bootstrap_skip_gpu_count = 0;
    uint64_t bootstrap_skip_resident_count = 0;

    std::vector<uint8_t> protected_mask(config_.expert_num, 0);
    auto mark_protected = [&](const int64_t* ids, int n) {
      if (ids == nullptr || n <= 0) return;
      for (int i = 0; i < n; ++i) {
        const int expert_id = static_cast<int>(ids[i]);
        if (expert_id >= 0 && expert_id < config_.expert_num && !config_.should_skip_expert(expert_id)) {
          protected_mask[expert_id] = 1;
        }
      }
    };
    mark_protected(expert_ids, count);
    mark_protected(protect_ids, protect_count);

    std::vector<uint8_t> seen(config_.expert_num, 0);
    const int limit = max_to_submit > 0 ? max_to_submit : count;
    int submitted = 0;
    for (int i = 0; i < count && submitted < limit; ++i) {
      const int expert_id = static_cast<int>(expert_ids[i]);
      if (expert_id < 0 || expert_id >= config_.expert_num) continue;
      if (config_.should_skip_expert(expert_id)) {
        if (is_bootstrap_prefetch && tp_part_idx == 0) {
          bootstrap_skip_gpu_count += 1;
        }
        continue;
      }
      if (seen[expert_id]) continue;
      seen[expert_id] = 1;
      if (is_bootstrap_prefetch && tp_part_idx == 0) {
        bootstrap_candidate_count += 1;
      }

      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state != EXPERT_BASELINE) {
        if (is_bootstrap_prefetch && tp_part_idx == 0) {
          bootstrap_skip_resident_count += 1;
        }
        continue;
      }
      if (submit_async_prefetch(expert_id, &protected_mask)) {
        submitted += 1;
      }
    }
    if (submitted > 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->prefetch_count.fetch_add(static_cast<uint64_t>(submitted), std::memory_order_relaxed);
    }
    if (is_bootstrap_prefetch && tp_part_idx == 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->bootstrap_prefetch_candidate_count.fetch_add(bootstrap_candidate_count,
                                                                        std::memory_order_relaxed);
      config_.cache_stats->bootstrap_prefetch_submit_count.fetch_add(static_cast<uint64_t>(submitted),
                                                                     std::memory_order_relaxed);
      config_.cache_stats->bootstrap_prefetch_skip_gpu_count.fetch_add(bootstrap_skip_gpu_count,
                                                                       std::memory_order_relaxed);
      config_.cache_stats->bootstrap_prefetch_skip_resident_count.fetch_add(bootstrap_skip_resident_count,
                                                                            std::memory_order_relaxed);
      config_.cache_stats->coldstart_prefill_count.fetch_add(static_cast<uint64_t>(submitted),
                                                             std::memory_order_relaxed);
    }
    return submitted;
  }

  void prefetch_experts_binding(intptr_t expert_ids,
                                int count,
                                intptr_t protect_ids,
                                int protect_count,
                                int max_to_submit,
                                int prefetch_kind = 0) {
    prefetch_experts(count,
                     reinterpret_cast<const int64_t*>(expert_ids),
                     protect_count,
                     reinterpret_cast<const int64_t*>(protect_ids),
                     max_to_submit,
                     prefetch_kind);
  }
#endif

  // ============================================================================
  // CRTP buffer creation - no group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
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

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = down_ba_[expert_idx];
    auto& bb = down_bb_[expert_idx];
    auto& bc = down_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }
  }
  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (iouring_enabled()) {
      validate_iouring_config();
      return;
    }

    if (config_.gate_projs.size()) {
      if (config_.use_mmap) {
        // mmap baseline mode: experts start on file-backed pointers and can be
        // promoted into NUMA-local resident buffers on demand.
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              float* gate_mins = nullptr;
              float* up_mins = nullptr;
              float* down_mins = nullptr;
              if constexpr (requires { gate_bb_[expert_id]->mins; }) {
                if (!config_.gate_zeros.empty()) gate_mins = (float*)(config_.gate_zeros[tp_part_idx][logical_expert_id]);
                if (!config_.up_zeros.empty()) up_mins = (float*)(config_.up_zeros[tp_part_idx][logical_expert_id]);
                if (!config_.down_zeros.empty()) down_mins = (float*)(config_.down_zeros[tp_part_idx][logical_expert_id]);
              }
              set_mmap_source_ptrs_quantized(
                  expert_id, config_.gate_projs[tp_part_idx][logical_expert_id],
                  config_.up_projs[tp_part_idx][logical_expert_id], config_.down_projs[tp_part_idx][logical_expert_id],
                  (float*)config_.gate_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.up_scales[tp_part_idx][logical_expert_id],
                  (float*)config_.down_scales[tp_part_idx][logical_expert_id], gate_mins, up_mins, down_mins);
            },
            nullptr);
      } else {
        // Legacy copy mode
        pool->do_work_stealing_job(
            config_.expert_num, nullptr,
            [this, physical_to_logical_map](int expert_id) {
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
              {
                size_t scale_size = config_.intermediate_size * sizeof(float);
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;

                memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
                }

                memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }

              {
                size_t scale_size = config_.hidden_size * sizeof(float);
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;

                memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][logical_expert_id], size);

                if constexpr (T::BufferB::SCALE) {
                  memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
                }
              }
            },
            nullptr);
      }

    } else {
      int nth = T::recommended_nth(config_.intermediate_size);
      static uint8_t mat_type_all = 3, mat_split = 1;
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));

      if (config_.load) {
        std::cout << "Loading from \"" << prefix << "\"" << std::endl;
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all * mat_split,
            [this, physical_to_logical_map, prefix, mat_type_all, mat_split](int task_id) {
              int64_t expert_idx = task_id / (mat_type_all * mat_split);
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
              uint8_t mat_split_idex = task_id % mat_split;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                read_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              }
            });
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
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // gate part
              gate_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
              // up part
              up_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                  ith, nth);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              // down part
              down_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.down_proj + logical_expert_id * config_.hidden_size * config_.intermediate_size,
                  ith, nth);
              // printf("load idown, expert %ld, ith %d, total nth %d\n", expert_idx, ith, nth);
            },
            nullptr);
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        std::filesystem::create_directories(prefix);
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this, physical_to_logical_map, prefix](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              expert_idx = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size);
              }
            },
            nullptr);
      }
    }
  }

  bool forward_prefill_streaming_slots(int qlen, int k, const int64_t* expert_ids, const float* weights,
                                       const void* input, void* output) {
#ifndef _WIN32
    if (!resident_io_enabled() || !slot_pool_enabled() || qlen <= 1) return false;

    const bool stream_trace = mesh_prefill_stream_trace_enabled();
    const auto forward_start = std::chrono::steady_clock::now();
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = 0;
    std::fill(this->m_local_num_.begin(), this->m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        const int expert_id = static_cast<int>(expert_ids[i * k + j]);
        if (config_.should_skip_expert(expert_id)) {
          continue;
        }
        this->m_local_pos_[i][j] = this->m_local_num_[expert_id]++;
      }
    }

    std::vector<int> active_experts;
    active_experts.reserve(config_.expert_num);
    for (int expert_id = 0; expert_id < config_.expert_num; expert_id++) {
      if (this->m_local_num_[expert_id] > 0) {
        this->m_expert_id_map_[activated_expert] = expert_id;
        active_experts.push_back(expert_id);
        activated_expert++;
      }
    }

    if (activated_expert == 0) {
      std::memset(output, 0, sizeof(float) * static_cast<size_t>(qlen) * config_.hidden_size);
      return true;
    }

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = this->gate_up_ba_pool_;
    void* gate_bc_pool_ptr = this->gate_bc_pool_;
    void* up_bc_pool_ptr = this->up_bc_pool_;
    void* down_ba_pool_ptr = this->down_ba_pool_;
    void* down_bc_pool_ptr = this->down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0,
           used_pool_bytes_ba_down = 0, used_pool_bytes_bc_down = 0;

    for (int expert_id = 0; expert_id < config_.expert_num; expert_id++) {
      this->m_local_input_ptr_[expert_id] = this->m_local_input_ + offset * config_.hidden_size;
      this->m_local_gate_output_ptr_[expert_id] = this->m_local_gate_output_ + offset * config_.intermediate_size;
      this->m_local_up_output_ptr_[expert_id] = this->m_local_up_output_ + offset * config_.intermediate_size;
      this->m_local_down_output_ptr_[expert_id] = this->m_local_down_output_ + offset * config_.hidden_size;
      offset += this->m_local_num_[expert_id];

      if (this->m_local_num_[expert_id] == 0) {
        continue;
      }

      size_t max_m = (this->m_local_num_[expert_id] + M_STEP - 1) / M_STEP * M_STEP;
      this->gate_up_ba_[expert_id]->max_m = max_m;
      this->gate_up_ba_[expert_id]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(this->buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      this->gate_bc_[expert_id]->max_m = max_m;
      this->gate_bc_[expert_id]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(this->buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      this->up_bc_[expert_id]->max_m = max_m;
      this->up_bc_[expert_id]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(this->buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      this->down_ba_[expert_id]->max_m = max_m;
      this->down_ba_[expert_id]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(this->buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      this->down_bc_[expert_id]->max_m = max_m;
      this->down_bc_[expert_id]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(this->buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }

    assert(used_pool_m <= this->pool_count_);
    assert(used_pool_bytes_a <= this->gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= this->gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= this->up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= this->down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= this->down_bc_pool_bytes_);

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    const auto copy_input_start = std::chrono::steady_clock::now();
    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        const int expert_id = static_cast<int>(expert_ids[i * k + j]);
        if (config_.should_skip_expert(expert_id)) {
          continue;
        }
        std::memcpy(this->m_local_input_ptr_[expert_id] + this->m_local_pos_[i][j] * config_.hidden_size,
                    (ggml_bf16_t*)input + i * config_.hidden_size,
                    sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });
    const auto copy_input_end = std::chrono::steady_clock::now();

    std::vector<int> ordered_experts;
    ordered_experts.reserve(active_experts.size());
    for (int expert_id : active_experts) {
      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
        ordered_experts.push_back(expert_id);
      }
    }
    for (int expert_id : active_experts) {
      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state != EXPERT_CACHED && state != EXPERT_PINNED) {
        ordered_experts.push_back(expert_id);
      }
    }

    const int window = std::max(1, std::min(cache_capacity_, activated_expert));
    const int total_windows = (activated_expert + window - 1) / window;
    std::vector<uint8_t> protected_mask(config_.expert_num, 0);
    prefill_streaming_active_ = true;
    struct PrefillEvictionGuard {
      AMX_MOE_TP* owner;
      ~PrefillEvictionGuard() {
        owner->eviction_protected_mask_ = nullptr;
        owner->prefill_streaming_active_ = false;
      }
    } prefill_guard{this};

    for (int start = 0; start < activated_expert; start += window) {
      const int count = std::min(window, activated_expert - start);
      const int window_index = start / window;
      std::fill(protected_mask.begin(), protected_mask.end(), 0);
      for (int i = start; i < activated_expert; ++i) {
        protected_mask[ordered_experts[i]] = 1;
      }
      eviction_protected_mask_ = &protected_mask;

      int hit_before = 0;
      int cold_before = 0;
      int inflight_before = 0;
      int other_before = 0;
      uint64_t expected_cold_read_bytes = 0;
      for (int i = 0; i < count; ++i) {
        const int expert_id = ordered_experts[start + i];
        const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
        if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
          hit_before += 1;
        } else if (state == EXPERT_BASELINE) {
          cold_before += 1;
          expected_cold_read_bytes += iouring_read_bytes_for_expert(expert_id);
        } else if (state == EXPERT_PREFETCHING) {
          inflight_before += 1;
        } else {
          other_before += 1;
        }
      }

      const uint64_t read_reqs_before =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
              : 0;
      const uint64_t read_bytes_before =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
              : 0;
      const int resident_before = resident_expert_count_.load(std::memory_order_acquire);
      const auto ensure_start = std::chrono::steady_clock::now();
      ExpertReadScope expert_read_scope(this, count);
      for (int i = 0; i < count; ++i) {
        const int expert_id = ordered_experts[start + i];
        this->m_expert_id_map_[i] = expert_id;
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope.add_expert(expert_id);
      }
      const auto ensure_end = std::chrono::steady_clock::now();
      const uint64_t read_reqs_after =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
              : read_reqs_before;
      const uint64_t read_bytes_after =
          config_.cache_stats != nullptr
              ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
              : read_bytes_before;
      const int resident_after_ensure = resident_expert_count_.load(std::memory_order_acquire);

      const auto compute_start = std::chrono::steady_clock::now();
      direct_or_pool(count, [this](int task_id) {
        int expert_idx = this->m_expert_id_map_[task_id];
        this->gate_up_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                this->m_local_input_ptr_[expert_idx], 0, 1);
      });

      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * count * 2, [](int _) { T::config(); },
          [this, nth, qlen](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
            if (do_up) {
              this->up_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_up_output_ptr_[expert_idx], ith, nth);
            } else {
              this->gate_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], ith, nth);
            }
          },
          nullptr);

      this->apply_activation(count, nth, qlen);

      pool->do_work_stealing_job(
          count, nullptr,
          [this](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id];
            this->down_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);

      nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * count, [](int _) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_down_gemm(expert_idx, ith, nth, qlen);
            this->down_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
      const auto compute_end = std::chrono::steady_clock::now();

      if (stream_trace) {
        std::fprintf(stderr,
                     "[MESH_PREFILL_STREAM_TRACE] layer=%d tp=%d qlen=%d active=%d window=%d windows=%d "
                     "start=%d count=%d hit_before=%d cold_before=%d inflight_before=%d other_before=%d "
                     "ensure_us=%llu compute_us=%llu read_req_delta=%llu read_bytes_delta=%llu "
                     "expected_cold_read_bytes=%llu resident_before=%d resident_after_ensure=%d resident_after_compute=%d\n",
                     config_.layer_idx,
                     tp_part_idx,
                     qlen,
                     activated_expert,
                     window_index,
                     total_windows,
                     start,
                     count,
                     hit_before,
                     cold_before,
                     inflight_before,
                     other_before,
                     static_cast<unsigned long long>(elapsed_us(ensure_start, ensure_end)),
                     static_cast<unsigned long long>(elapsed_us(compute_start, compute_end)),
                     static_cast<unsigned long long>(read_reqs_after - read_reqs_before),
                     static_cast<unsigned long long>(read_bytes_after - read_bytes_before),
                     static_cast<unsigned long long>(expected_cold_read_bytes),
                     resident_before,
                     resident_after_ensure,
                     resident_expert_count_.load(std::memory_order_acquire));
      }
    }

    eviction_protected_mask_ = nullptr;
    prefill_streaming_active_ = false;

    const auto merge_start = std::chrono::steady_clock::now();
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              const int expert_id = static_cast<int>(expert_ids[i * k + j]);
              if (config_.should_skip_expert(expert_id)) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(this->m_local_down_output_ptr_[expert_id] +
                             this->m_local_pos_[i][j] * config_.hidden_size + e),
                  &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);
    const auto merge_end = std::chrono::steady_clock::now();

    update_lookahead_heat_from_forward(qlen, k, expert_ids, weights);
    refresh_lookahead_heat();
    trim_cache_to_capacity();
    trim_cache_for_memory_pressure();
    if (tp_part_idx == 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
      config_.cache_stats->maybe_dump_jsonl();
    }
    if (stream_trace) {
      const auto forward_end = std::chrono::steady_clock::now();
      std::fprintf(stderr,
                   "[MESH_PREFILL_STREAM_SUMMARY] layer=%d tp=%d qlen=%d active=%d window_size=%d windows=%d "
                   "copy_input_us=%llu merge_us=%llu total_us=%llu resident_final=%d allocated_slots=%d\n",
                   config_.layer_idx,
                   tp_part_idx,
                   qlen,
                   activated_expert,
                   window,
                   total_windows,
                   static_cast<unsigned long long>(elapsed_us(copy_input_start, copy_input_end)),
                   static_cast<unsigned long long>(elapsed_us(merge_start, merge_end)),
                   static_cast<unsigned long long>(elapsed_us(forward_start, forward_end)),
                   resident_expert_count_.load(std::memory_order_acquire),
                   allocated_slot_count());
    }
    return true;
#else
    (void)qlen;
    (void)k;
    (void)expert_ids;
    (void)weights;
    (void)input;
    (void)output;
    return false;
#endif
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
               void* output) override {
#ifndef _WIN32
    std::unique_ptr<ExpertReadScope> expert_read_scope;
    std::vector<int> active_experts;
    if (resident_io_enabled()) {
      complete_ready_prefetches();
      refresh_lookahead_heat();
      if (qlen > 1 && forward_prefill_streaming_slots(qlen, k, expert_ids, weights, input, output)) {
        return;
      }
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
      }
      batch_ensure_experts_ready(active_experts, false);
      for (int expert_id : active_experts) {
        expert_read_scope->add_expert(expert_id);
      }
    }
#endif
    Base::forward(qlen, k, expert_ids, weights, input, output);
#ifndef _WIN32
    if (resident_io_enabled() && !active_experts.empty()) {
      update_lookahead_heat_from_forward(qlen, k, expert_ids, weights);
      refresh_lookahead_heat();
      expert_read_scope.reset();
      if (config_.max_tier0_experts <= 0) {
        for (int expert_id : active_experts) {
          demote_expert(expert_id);
        }
      } else {
        trim_cache_to_capacity();
        trim_cache_for_memory_pressure();
      }
      if (tp_part_idx == 0 && config_.enable_cache_stats && config_.cache_stats != nullptr) {
        config_.cache_stats->maybe_dump_jsonl();
      }
    }
#endif
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
      apply_cold_ptrs(expert_id);
      note_expert_demote(expert_id);
      if (!slot_pool_enabled()) {
        resident_expert_count_.fetch_sub(1, std::memory_order_acq_rel);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      }
      return;
    }
#else
    (void)expert_id;
#endif
  }

  bool is_expert_promoted(int expert_id) const override {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
    if (slot_pool_enabled()) {
      return state == EXPERT_CACHED || state == EXPERT_PINNED || state == EXPERT_PREFETCHING;
    }
    return state == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  int expert_state_code(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return static_cast<int>(EXPERT_BASELINE);
    if (!resident_io_enabled() || expert_states_ == nullptr) return static_cast<int>(EXPERT_CACHED);
    return static_cast<int>(expert_states_[expert_id].load(std::memory_order_acquire));
#else
    (void)expert_id;
    return 0;
#endif
  }

  bool is_expert_ready_resident(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    if (!resident_io_enabled() || expert_states_ == nullptr) return true;
    const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
    return state == EXPERT_CACHED || state == EXPERT_PINNED;
#else
    (void)expert_id;
    return false;
#endif
  }

  bool is_expert_cold_baseline(int expert_id) const {
#ifndef _WIN32
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    if (!resident_io_enabled() || expert_states_ == nullptr) return false;
    return expert_states_[expert_id].load(std::memory_order_acquire) == EXPERT_BASELINE;
#else
    (void)expert_id;
    return false;
#endif
  }

  // forward, forward_prefill, forward_decode, warm_up are inherited from Base
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_TP
// Holds concrete AMX_MOE_TP instances directly. Inheriting through
// TP_MOE<AMX_MOE_BASE<...>> would allocate base objects into `tps`, which
// breaks derived-only tiered logic for AMXINT4/AMXINT8.
// ============================================================================

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE_Common<AMX_MOE_TP<K>> {
 public:
  using Base = TP_MOE_Common<AMX_MOE_TP<K>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.io_backend == IOBackend::IOURING) {
      printf("TP Load from io_uring file slots\n");
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
      this->weights_loaded = true;
    } else if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = this->tp_configs[i];
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tpc.load == false) {
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

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = this->tp_configs[i];
        this->tps[i]->set_weight_buffers(tpc.gate_proj, tpc.up_proj, tpc.down_proj);
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = this->tp_configs[i];
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file %s\n", config.path.c_str());
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        this->tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
        this->tps[i]->load_weights();
      });
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

  int expert_state_code(int expert_id) const {
    return !this->tps.empty() ? this->tps[0]->expert_state_code(expert_id) : 0;
  }

  void split_deferred_experts_binding(intptr_t source_ids,
                                      intptr_t immediate_ids,
                                      intptr_t deferred_ids,
                                      int count,
                                      int k,
                                      int max_deferred_per_token) {
    split_deferred_experts(reinterpret_cast<const int64_t*>(source_ids),
                           reinterpret_cast<int64_t*>(immediate_ids),
                           reinterpret_cast<int64_t*>(deferred_ids),
                           count,
                           k,
                           max_deferred_per_token);
  }

  void split_deferred_experts(const int64_t* source_ids,
                              int64_t* immediate_ids,
                              int64_t* deferred_ids,
                              int count,
                              int k,
                              int max_deferred_per_token) {
    if (source_ids == nullptr || immediate_ids == nullptr || deferred_ids == nullptr || count <= 0 || k <= 0) return;

    const int defer_limit = std::max(0, std::min(max_deferred_per_token, k));
    std::vector<int64_t> ids(count);
    for (int i = 0; i < count; ++i) {
      ids[i] = source_ids[i];
      immediate_ids[i] = -1;
      deferred_ids[i] = -1;
    }

    if (!this->weights_loaded || this->tps.empty()) {
      for (int i = 0; i < count; ++i) {
        if (!this->config.should_skip_expert(ids[i])) {
          immediate_ids[i] = ids[i];
        }
      }
      return;
    }

    if (this->tp_count <= 1) {
      this->tps[0]->complete_ready_prefetches();
    } else {
      this->config.pool->dispense_backend()->do_numa_job(
          [this](int tp_id) { this->tps[tp_id]->complete_ready_prefetches(); });
    }

    std::vector<int64_t> cold_deferred;
    std::vector<int64_t> protected_immediate;
    cold_deferred.reserve(count);
    protected_immediate.reserve(count);

    const int rows = (count + k - 1) / k;
    uint64_t cpu_topk_count = 0;
    uint64_t gpu_skip_count = 0;
    uint64_t nonready_count = 0;
    uint64_t deferred_count = 0;
    uint64_t overflow_immediate_count = 0;
    uint64_t overflow_token_count = 0;
    for (int row = 0; row < rows; ++row) {
      int deferred_in_row = 0;
      int overflow_in_row = 0;
      const int row_begin = row * k;
      const int row_end = std::min(count, row_begin + k);
      for (int idx = row_begin; idx < row_end; ++idx) {
        const int expert_id = static_cast<int>(ids[idx]);
        if (this->config.should_skip_expert(expert_id)) {
          gpu_skip_count += 1;
          continue;
        }
        cpu_topk_count += 1;

        const bool ready = this->tps[0]->is_expert_ready_resident(expert_id);
        if (!ready) {
          nonready_count += 1;
        }
        if (!ready && deferred_in_row < defer_limit) {
          deferred_ids[idx] = expert_id;
          deferred_in_row += 1;
          deferred_count += 1;
          if (this->tps[0]->is_expert_cold_baseline(expert_id)) {
            cold_deferred.push_back(expert_id);
          }
          continue;
        }

        immediate_ids[idx] = expert_id;
        protected_immediate.push_back(expert_id);
        if (!ready) {
          overflow_immediate_count += 1;
          overflow_in_row += 1;
        }
      }
      if (overflow_in_row > 0) {
        overflow_token_count += 1;
      }
    }

    if (this->config.enable_cache_stats && this->config.cache_stats != nullptr) {
      this->config.cache_stats->state_defer_token_count.fetch_add(static_cast<uint64_t>(rows),
                                                                  std::memory_order_relaxed);
      this->config.cache_stats->state_defer_cpu_topk_count.fetch_add(cpu_topk_count,
                                                                     std::memory_order_relaxed);
      this->config.cache_stats->state_defer_gpu_skip_count.fetch_add(gpu_skip_count,
                                                                    std::memory_order_relaxed);
      this->config.cache_stats->state_defer_nonready_count.fetch_add(nonready_count,
                                                                     std::memory_order_relaxed);
      this->config.cache_stats->state_defer_deferred_count.fetch_add(deferred_count,
                                                                    std::memory_order_relaxed);
      this->config.cache_stats->state_defer_overflow_immediate_count.fetch_add(
          overflow_immediate_count, std::memory_order_relaxed);
      this->config.cache_stats->state_defer_overflow_token_count.fetch_add(overflow_token_count,
                                                                          std::memory_order_relaxed);
      this->config.cache_stats->maybe_dump_jsonl();
    }

    if (!cold_deferred.empty()) {
      prefetch_experts(static_cast<int>(cold_deferred.size()),
                       cold_deferred.data(),
                       static_cast<int>(protected_immediate.size()),
                       protected_immediate.empty() ? nullptr : protected_immediate.data(),
                       static_cast<int>(cold_deferred.size()),
                       0);
    }
  }

  void prefetch_experts_binding(intptr_t expert_ids,
                                int count,
                                intptr_t protect_ids,
                                int protect_count,
                                int max_to_submit,
                                int prefetch_kind = 0) {
    prefetch_experts(count,
                     reinterpret_cast<const int64_t*>(expert_ids),
                     protect_count,
                     reinterpret_cast<const int64_t*>(protect_ids),
                     max_to_submit,
                     prefetch_kind);
  }

  void prefetch_experts(int count,
                        const int64_t* expert_ids,
                        int protect_count = 0,
                        const int64_t* protect_ids = nullptr,
                        int max_to_submit = 0,
                        int prefetch_kind = 0) {
    if (!this->weights_loaded || expert_ids == nullptr || count <= 0) return;
    if (this->tp_count <= 1) {
      this->tps[0]->prefetch_experts(count, expert_ids, protect_count, protect_ids, max_to_submit, prefetch_kind);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, count, expert_ids, protect_count, protect_ids, max_to_submit, prefetch_kind](int tp_id) {
          this->tps[tp_id]->prefetch_experts(count, expert_ids, protect_count, protect_ids, max_to_submit,
                                             prefetch_kind);
        });
  }

  void mesh_prepare_prefill_layer_binding() {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      this->tps[0]->prepare_prefill_layer_window();
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this](int tp_id) { this->tps[tp_id]->prepare_prefill_layer_window(); });
  }

  void mesh_release_prefill_layer_binding() {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      (void)this->tps[0]->release_prefill_layer_window();
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this](int tp_id) { (void)this->tps[tp_id]->release_prefill_layer_window(); });
  }

  void mesh_transition_decode_cache_binding(int decode_capacity, int fill_limit) {
    if (!this->weights_loaded) return;
    if (this->tp_count <= 1) {
      this->tps[0]->transition_to_decode_cache(decode_capacity, fill_limit);
      return;
    }
    this->config.pool->dispense_backend()->do_numa_job(
        [this, decode_capacity, fill_limit](int tp_id) {
          this->tps[tp_id]->transition_to_decode_cache(decode_capacity, fill_limit);
        });
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

#endif
