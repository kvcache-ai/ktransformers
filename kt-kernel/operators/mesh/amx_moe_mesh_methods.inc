// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  void set_weight_buffers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

#ifndef _WIN32
  static constexpr bool buffer_b_has_scale_ptr() {
    return requires(typename T::BufferB* bb) { bb->d; };
  }

  bool iouring_enabled() const { return mesh::iouring_enabled(config_); }
  bool resident_io_enabled() const { return buffer_b_has_scale_ptr() && iouring_enabled(); }

  bool mesh_load_weights_from_resident_sources(const uint64_t* physical_to_logical_map) {
    (void)physical_to_logical_map;
    if (iouring_enabled()) {
      validate_iouring_config();
      return true;
    }
    return false;
  }

  void cleanup_resident_io_state() {
    if (!resident_io_enabled()) return;
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

  void initialize_resident_io_state() {
    if constexpr (!buffer_b_has_scale_ptr()) {
      cache_capacity_ = 0;
      return;
    } else {
    const int en = config_.expert_num;
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
    prefill_static_capacity_ = 0;
    prefill_static_slot_count_ = 0;
    prefill_static_expert_mask_.assign(en, 0);
    prefill_static_slot_for_expert_.assign(en, -1);
    prefill_static_experts_.clear();
    expert_frequency_score_.assign(en, 0);
    expert_frequency_total_ = 0;

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
    const mesh::ResidentCapacityPlan capacity_plan = mesh::build_resident_capacity_plan(config_);
    cache_capacity_ = capacity_plan.cache_capacity;
    prefill_static_capacity_ = capacity_plan.prefill_static_capacity;
    mesh::resize_resident_slot_storage(cache_capacity_,
                                       en,
                                       slot_gate_owner_,
                                       slot_up_owner_,
                                       slot_down_owner_,
                                       slot_to_expert_,
                                       expert_to_slot_,
                                       slot_states_,
                                       slot_active_readers_);
    resident_policy_.reset(en, config_.resident_cache_policy);
    lookahead_heat_.assign(en, 0.0f);

    if (config_.pool != nullptr && tp_part_idx < (int)config_.pool->config.subpool_numa_map.size()) {
      numa_node_ = config_.pool->config.subpool_numa_map[tp_part_idx];
    }

    rebuild_prefill_static_expert_set();

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
    clear_expert_ptrs(expert_id);
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

  void acquire_expert_read(int expert_id) {
    mesh::acquire_expert_read(resident_slot_pool_view(), expert_id);
  }

  void release_expert_read(int expert_id) {
    mesh::release_expert_read(resident_slot_pool_view(), expert_id);
  }

  using ExpertReadScope = mesh::ExpertReadScope<AMX_MOE_TP<T>>;

  mesh::ResidentSlotPoolView resident_slot_pool_view() {
    return mesh::ResidentSlotPoolView{cache_capacity_,
                                      config_.expert_num,
                                      &slot_gate_owner_,
                                      &slot_up_owner_,
                                      &slot_down_owner_,
                                      &slot_to_expert_,
                                      &expert_to_slot_,
                                      expert_states_.get(),
                                      active_readers_.get(),
                                      slot_states_.get(),
                                      slot_active_readers_.get(),
                                      &resident_expert_count_,
                                      &resident_slot_pool_allocated_,
                                      &mesh_slot_mode_};
  }

  mesh::ResidentSlotPoolConstView resident_slot_pool_view() const {
    return mesh::ResidentSlotPoolConstView{cache_capacity_,
                                           config_.expert_num,
                                           &slot_gate_owner_,
                                           &slot_up_owner_,
                                           &slot_down_owner_,
                                           &slot_to_expert_,
                                           &expert_to_slot_,
                                           expert_states_.get(),
                                           active_readers_.get(),
                                           slot_states_.get(),
                                           slot_active_readers_.get(),
                                           &resident_expert_count_,
                                           &resident_slot_pool_allocated_};
  }

  struct ResidentSlotPoolHooks {
    AMX_MOE_TP* owner = nullptr;

    bool allocate_slot_buffers(int slot) { return owner->allocate_slot_buffers(slot); }

    void release_slot_buffers(int slot) { owner->release_slot_buffers(slot); }

    void clear_packed_owners(int expert_id) {
      owner->packed_gate_owner_[expert_id] = nullptr;
      owner->packed_up_owner_[expert_id] = nullptr;
      owner->packed_down_owner_[expert_id] = nullptr;
    }

    void set_packed_owners(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
      owner->packed_gate_owner_[expert_id] = gate_owner;
      owner->packed_up_owner_[expert_id] = up_owner;
      owner->packed_down_owner_[expert_id] = down_owner;
    }

    void apply_cold_ptrs(int expert_id) { owner->apply_cold_ptrs(expert_id); }

    void apply_owned_ptrs(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
      owner->apply_owned_ptrs(expert_id, gate_owner, up_owner, down_owner);
    }

    void note_expert_insert(int expert_id, bool hard_pin) { owner->note_expert_insert(expert_id, hard_pin); }

    void note_expert_demote(int expert_id) { owner->note_expert_demote(expert_id); }

    std::string make_slot_bind_error(int slot, int old_expert, int new_expert) const {
      return mesh::slot_bind_error_message(owner->config_,
                                           owner->tp_part_idx,
                                           mesh::const_view(owner->resident_slot_pool_view()),
                                           slot,
                                           old_expert,
                                           new_expert,
                                           owner->resident_debug_summary());
    }
  };

  ResidentSlotPoolHooks resident_slot_pool_hooks() { return ResidentSlotPoolHooks{this}; }

  bool slot_pool_configured() const {
    return mesh::resident_slot_pool_configured(resident_slot_pool_view());
  }

  bool slot_pool_enabled() const {
    return mesh::resident_slot_pool_enabled(resident_slot_pool_view());
  }

  bool lazy_slot_buffers_enabled() const {
    return config_.mesh_prefill_layer_mode_enabled && iouring_enabled();
  }

  bool slot_has_buffers(int slot) const {
    return mesh::slot_has_buffers(resident_slot_pool_view(), slot);
  }

  int allocated_slot_count() const {
    return mesh::allocated_slot_count(resident_slot_pool_view());
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
    auto hooks = resident_slot_pool_hooks();
    return mesh::allocate_resident_slot_pool(resident_slot_pool_view(), hooks, lazy_slot_buffers_enabled());
  }

  bool ensure_resident_slot_pool_allocated() {
    if (!slot_pool_configured()) return false;
    return allocate_resident_slot_pool();
  }

  bool release_resident_slot_pool(bool count_stats = false) {
    auto hooks = resident_slot_pool_hooks();
    return mesh::release_resident_slot_pool(resident_slot_pool_view(), hooks, count_stats);
  }

  void bind_slot_to_expert(int slot, int expert_id, bool pin_after_copy) {
    auto hooks = resident_slot_pool_hooks();
    mesh::bind_slot_to_expert(resident_slot_pool_view(), hooks, slot, expert_id, pin_after_copy, slot_pool_enabled());
  }

  bool unbind_slot(int slot, bool release_occupancy, bool count_stats = true) {
    auto hooks = resident_slot_pool_hooks();
    return mesh::unbind_slot(resident_slot_pool_view(), hooks, slot, release_occupancy, count_stats);
  }

  int find_empty_slot() const {
    return mesh::find_empty_slot(resident_slot_pool_view(), slot_pool_enabled());
  }

  bool reserve_slot_for_loading(int slot, int expert_id, PrefillPromotionTiming* timing = nullptr) {
    auto hooks = resident_slot_pool_hooks();
    return mesh::reserve_slot_for_loading(resident_slot_pool_view(), hooks, slot, expert_id, timing);
  }

  int slot_for_expert_or_empty(int expert_id) const {
    return mesh::slot_for_expert_or_empty(resident_slot_pool_view(), slot_pool_enabled(), expert_id);
  }

  std::string resident_debug_summary() const {
    return mesh::resident_debug_summary(config_.expert_num,
                                        cache_capacity_,
                                        slot_pool_enabled(),
                                        expert_states_.get(),
                                        active_readers_.get(),
                                        slot_states_.get(),
                                        slot_active_readers_.get(),
                                        slot_to_expert_);
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

  void validate_iouring_config() const {
    if (!iouring_enabled()) return;
#ifdef HAVE_LIBURING
    bool include_mins = false;
    if constexpr (requires { gate_bb_[0]->mins; }) {
      include_mins = true;
    }
    mesh::validate_amx_iouring_config(config_,
                                      tp_part_idx,
                                      cache_capacity_,
                                      gate_weight_bytes_,
                                      gate_scale_bytes_,
                                      up_weight_bytes_,
                                      up_scale_bytes_,
                                      down_weight_bytes_,
                                      down_scale_bytes_,
                                      gate_mins_bytes_,
                                      up_mins_bytes_,
                                      down_mins_bytes_,
                                      include_mins);
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
    return mesh::prefill_stream_trace_enabled();
  }

  void maybe_dump_prefill_expert_frequency(int qlen, int k, const int64_t* expert_ids,
                                           const std::vector<int>& active_experts) const {
    mesh::maybe_dump_prefill_expert_frequency(
        config_, tp_part_idx, qlen, k, expert_ids, active_experts, this->m_local_num_);
  }

  static uint64_t elapsed_us(std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }

  uint64_t iouring_read_bytes_for_expert(int expert_id) const {
    if (!iouring_enabled() || expert_id < 0 || expert_id >= config_.expert_num) return 0;
    bool include_mins = false;
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      include_mins = true;
    }
    return mesh::amx_iouring_read_bytes_for_expert(config_, tp_part_idx, expert_id, include_mins);
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
      mesh::record_cache_eviction_demote(config_);

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
      mesh::record_cache_eviction_demote(config_);
      return true;
    }
    return false;
  }

  bool evict_one_decode_scratch_expert(const std::vector<uint8_t>* protected_mask = nullptr) {
    if (!decode_frequency_policy_active()) return false;
    const int scratch_begin = decode_main_slot_count();
    for (int slot = scratch_begin; slot < cache_capacity_; ++slot) {
      if (slot_states_[slot].load(std::memory_order_acquire) != SLOT_READY || slot_to_expert_[slot] < 0) continue;
      if (!slot_has_rebindable_expert(slot, protected_mask)) continue;
      if (!unbind_slot(slot, true, true)) continue;
      mesh::record_cache_eviction_demote(config_);
      return true;
    }
    return false;
  }

  bool reserve_loading_slot_for_expert(int expert_id, int* slot_out,
                                       const std::vector<uint8_t>* protected_mask = nullptr) {
    if (slot_out == nullptr || !slot_pool_enabled()) return false;
    int slot = -1;
    if (decode_frequency_policy_active()) {
      slot = find_decode_frequency_loading_slot(expert_id, protected_mask);
      if (slot < 0) return false;
      if (!reserve_slot_for_loading(slot, expert_id)) {
        return false;
      }
      *slot_out = slot;
      return true;
    }

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
      const int slot =
          mesh::find_slot_by_owners(mesh::const_view(resident_slot_pool_view()), gate_owner, up_owner, down_owner);
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
    // representation for this expert.
  }

#ifdef HAVE_LIBURING
  void append_iouring_read_requests_for_expert(int expert_id,
                                               void* gate_owner,
                                               void* up_owner,
                                               void* down_owner,
                                               std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch) {
    bool include_mins = false;
    if constexpr (requires { gate_bb_[expert_id]->mins; }) {
      include_mins = true;
    }
    mesh::append_amx_iouring_read_requests_for_expert(config_,
                                                      tp_part_idx,
                                                      expert_id,
                                                      gate_owner,
                                                      up_owner,
                                                      down_owner,
                                                      gate_weight_bytes_,
                                                      gate_scale_bytes_,
                                                      up_weight_bytes_,
                                                      up_scale_bytes_,
                                                      down_weight_bytes_,
                                                      down_scale_bytes_,
                                                      include_mins,
                                                      read_batch);
  }

  void record_iouring_read_stats(const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch) {
    mesh::record_iouring_read_stats(config_.cache_stats, config_.enable_cache_stats, read_batch);
  }

  std::vector<uint64_t> submit_iouring_reads_for_expert(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    if (config_.async_reader == nullptr) {
      return {};
    }

    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    read_batch.reserve(9);
    append_iouring_read_requests_for_expert(expert_id, gate_owner, up_owner, down_owner, &read_batch);
    std::vector<uint64_t> read_requests = config_.async_reader->submit_reads(read_batch);
    record_iouring_read_stats(read_batch);
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

    if (!iouring_enabled()) {
      release_on_failure();
      return false;
    }

    // Load expert weights through io_uring direct I/O.
#ifdef HAVE_LIBURING
      if (config_.async_reader == nullptr) {
        release_on_failure();
        return false;
      }

      std::vector<uint64_t> read_requests = submit_iouring_reads_for_expert(expert_id, gate_owner, up_owner, down_owner);

      // Wait for every tensor fragment (weight, scale, optional mins) to complete.
      const int timeout_ms = 60000;
      if (!config_.async_reader->wait_for_requests(read_requests, timeout_ms)) {
        const std::string detail = mesh::amx_iouring_promotion_failure_message(
            config_, tp_part_idx, expert_id, read_requests, config_.async_reader, timeout_ms);
        release_on_failure();
        throw std::runtime_error(detail);
      }
#else
      // io_uring not available, fallback to error
      release_on_failure();
      return false;
#endif

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
    if (decode_frequency_policy_active()) {
      record_decode_frequency_counts(expert_ids);
    }

    std::vector<uint8_t> local_protected_mask;
    const std::vector<uint8_t>* effective_protected_mask = protected_mask;
    if (effective_protected_mask == nullptr) {
      local_protected_mask = mesh::build_protected_mask(config_, expert_ids);
      effective_protected_mask = &local_protected_mask;
    }

    std::vector<uint8_t> seen(config_.expert_num, 0);
    std::vector<BatchPromotion> cold_promotions;
    std::vector<int> pending_prefetch_experts;
    std::vector<uint64_t> all_requests;

    auto record_total_access = [&](int expert_id) {
      mesh::record_cache_total_access(config_, tp_part_idx, expert_id);
    };
    auto record_hit = [&](int expert_id) {
      trace_hits += 1;
      mesh::record_cache_hit(config_, tp_part_idx, expert_id);
    };
    auto record_cold_miss = [&](int expert_id) {
      trace_cold += 1;
      trace_expected_cold_read_bytes += iouring_read_bytes_for_expert(expert_id);
      mesh::record_cache_cold_miss(config_, tp_part_idx, expert_id);
    };
    auto record_inflight_miss = [&](int expert_id) {
      trace_inflight += 1;
      mesh::record_cache_inflight_miss(config_, tp_part_idx, expert_id);
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
          (void)maybe_promote_scratch_expert_by_frequency(expert_id, effective_protected_mask);
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
      mesh::record_cache_promote(config_, tp_part_idx, promotion.expert_id);
    }

    for (int expert_id : pending_prefetch_experts) {
      mesh::record_cache_prefetch_hit(config_, tp_part_idx, expert_id);
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
      mesh::log_batch_ensure_trace(config_,
                                    tp_part_idx,
                                    expert_ids.size(),
                                    trace_unique,
                                    trace_hits,
                                    trace_cold,
                                    trace_inflight,
                                    cold_promotions.size(),
                                    pending_prefetch_experts.size(),
                                    trace_wait_us,
                                    elapsed_us(batch_start, batch_end),
                                    trace_read_reqs_after - trace_read_reqs_before,
                                    trace_read_bytes_after - trace_read_bytes_before,
                                    trace_expected_cold_read_bytes,
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
    if (decode_frequency_policy_active()) {
      for (int attempt = 0; attempt < config_.expert_num * 2 &&
                            resident_expert_count_.load(std::memory_order_acquire) > target_capacity;
           ++attempt) {
        if (!evict_one_decode_scratch_expert()) {
          break;
        }
      }
    }
    for (int attempt = 0; attempt < config_.expert_num * 2 &&
                          resident_expert_count_.load(std::memory_order_acquire) > target_capacity;
         ++attempt) {
      if (!evict_one_cached_expert(-1)) {
        break;
      }
    }
  }

  int decode_cache_capacity() const {
    return mesh::decode_cache_capacity(config_, cache_capacity_);
  }

  int decode_main_slot_count() const {
    if (!slot_pool_enabled()) return 0;
    return std::max(0, std::min(decode_cache_capacity(), cache_capacity_));
  }

  bool decode_frequency_policy_active() const {
    return config_.mesh_prefill_layer_mode_enabled && slot_pool_enabled() &&
           mesh_slot_mode_ == MESH_SLOT_DECODE_CACHE && expert_frequency_total_ > 0 &&
           decode_main_slot_count() > 0;
  }

  uint64_t expert_frequency_score(int expert_id) const {
    return mesh::expert_frequency_score(expert_frequency_score_, expert_id);
  }

  void record_prefill_frequency_counts() {
    mesh::record_prefill_frequency_counts(
        config_, this->m_local_num_, expert_frequency_score_, expert_frequency_total_);
  }

  void record_decode_frequency_counts(const std::vector<int>& expert_ids) {
    mesh::record_decode_frequency_counts(config_, expert_ids, expert_frequency_score_, expert_frequency_total_);
  }

  std::vector<int> cpu_managed_experts_by_frequency() const {
    return mesh::cpu_managed_experts_by_frequency(config_, expert_frequency_score_);
  }

  void rebuild_prefill_static_expert_set() {
    mesh::rebuild_prefill_static_expert_set(config_,
                                            prefill_static_capacity_,
                                            nullptr,
                                            prefill_static_slot_count_,
                                            prefill_static_expert_mask_,
                                            prefill_static_slot_for_expert_,
                                            prefill_static_experts_);
  }

  void rebuild_prefill_static_expert_set_by_frequency(int requested_capacity) {
    mesh::rebuild_prefill_static_expert_set(config_,
                                            requested_capacity,
                                            &expert_frequency_score_,
                                            prefill_static_slot_count_,
                                            prefill_static_expert_mask_,
                                            prefill_static_slot_for_expert_,
                                            prefill_static_experts_);
  }

  int prefill_static_slot_for_expert(int expert_id) const {
    if (expert_id < 0 || expert_id >= static_cast<int>(prefill_static_slot_for_expert_.size())) return -1;
    return prefill_static_slot_for_expert_[expert_id];
  }

  bool slot_has_rebindable_expert(int slot, const std::vector<uint8_t>* protected_mask) const {
    return mesh::slot_has_rebindable_expert(resident_slot_pool_view(), config_, slot, protected_mask);
  }

  bool swap_slot_bindings(int lhs, int rhs) {
    if (!slot_pool_enabled()) return false;
    if (lhs == rhs) return true;
    if (lhs < 0 || lhs >= cache_capacity_ || rhs < 0 || rhs >= cache_capacity_) return false;
    if (slot_active_readers_[lhs].load(std::memory_order_acquire) != 0 ||
        slot_active_readers_[rhs].load(std::memory_order_acquire) != 0) {
      return false;
    }
    const int lhs_expert = slot_to_expert_[lhs];
    const int rhs_expert = slot_to_expert_[rhs];
    if (lhs_expert >= 0 && active_readers_[lhs_expert].load(std::memory_order_acquire) != 0) return false;
    if (rhs_expert >= 0 && active_readers_[rhs_expert].load(std::memory_order_acquire) != 0) return false;

    const uint8_t lhs_state = slot_states_[lhs].load(std::memory_order_acquire);
    const uint8_t rhs_state = slot_states_[rhs].load(std::memory_order_acquire);
    if (lhs_state == SLOT_LOADING || rhs_state == SLOT_LOADING) return false;

    std::swap(slot_gate_owner_[lhs], slot_gate_owner_[rhs]);
    std::swap(slot_up_owner_[lhs], slot_up_owner_[rhs]);
    std::swap(slot_down_owner_[lhs], slot_down_owner_[rhs]);
    std::swap(slot_to_expert_[lhs], slot_to_expert_[rhs]);
    slot_states_[lhs].store(rhs_state, std::memory_order_release);
    slot_states_[rhs].store(lhs_state, std::memory_order_release);

    auto refresh_expert_mapping = [this](int slot) {
      const int expert_id = slot_to_expert_[slot];
      if (expert_id < 0 || expert_id >= config_.expert_num) return;
      expert_to_slot_[expert_id] = slot;
      packed_gate_owner_[expert_id] = slot_gate_owner_[slot];
      packed_up_owner_[expert_id] = slot_up_owner_[slot];
      packed_down_owner_[expert_id] = slot_down_owner_[slot];
      apply_owned_ptrs(expert_id, slot_gate_owner_[slot], slot_up_owner_[slot], slot_down_owner_[slot]);
    };
    refresh_expert_mapping(lhs);
    refresh_expert_mapping(rhs);
    return true;
  }

  int find_frequency_main_replacement_slot(int expert_id,
                                           const std::vector<uint8_t>* protected_mask) const {
    return mesh::find_frequency_main_replacement_slot(resident_slot_pool_view(),
                                                      config_,
                                                      decode_main_slot_count(),
                                                      expert_frequency_score_,
                                                      expert_id,
                                                      protected_mask);
  }

  int find_frequency_scratch_slot(const std::vector<uint8_t>* protected_mask) const {
    return mesh::find_frequency_scratch_slot(
        resident_slot_pool_view(), config_, decode_main_slot_count(), protected_mask);
  }

  int find_decode_frequency_loading_slot(int expert_id, const std::vector<uint8_t>* protected_mask) const {
    const int main_slot = find_frequency_main_replacement_slot(expert_id, protected_mask);
    if (main_slot >= 0) return main_slot;
    return find_frequency_scratch_slot(protected_mask);
  }

  int find_main_slot_outside_frequency_set(const std::vector<uint8_t>& desired_mask) const {
    return mesh::find_main_slot_outside_frequency_set(
        resident_slot_pool_view(), config_, decode_main_slot_count(), desired_mask);
  }

  bool maybe_promote_scratch_expert_by_frequency(int expert_id, const std::vector<uint8_t>* protected_mask) {
    if (!decode_frequency_policy_active()) return false;
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;
    const int main_slots = decode_main_slot_count();
    const int current_slot = expert_to_slot_[expert_id];
    if (current_slot < main_slots || current_slot >= cache_capacity_) return false;
    if (!slot_has_rebindable_expert(current_slot, nullptr)) return false;
    const int target_slot = find_frequency_main_replacement_slot(expert_id, protected_mask);
    if (target_slot < 0) return false;
    return swap_slot_bindings(target_slot, current_slot);
  }

  bool release_prefill_scratch_slots(bool release_buffers) {
    if (!slot_pool_enabled()) return true;
    bool ok = true;
    const int scratch_begin = std::max(0, std::min(prefill_static_slot_count_, cache_capacity_));
    for (int slot = scratch_begin; slot < cache_capacity_; ++slot) {
      const uint8_t state = slot_states_[slot].load(std::memory_order_acquire);
      if (state == SLOT_LOADING) {
        ok = false;
        continue;
      }
      if (state == SLOT_READY && !unbind_slot(slot, true, false)) {
        ok = false;
        continue;
      }
      if (release_buffers && slot_states_[slot].load(std::memory_order_acquire) == SLOT_EMPTY &&
          slot_to_expert_[slot] < 0 && slot_has_buffers(slot)) {
        release_slot_buffers(slot);
      }
    }
    return ok;
  }

  bool submit_prefill_promotion_to_slot(int expert_id,
                                        int slot,
                                        std::vector<BatchPromotion>* promotions,
                                        std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch,
                                        PrefillPromotionTiming* timing = nullptr) {
    if (promotions == nullptr || read_batch == nullptr) return false;
    if (!iouring_enabled() || config_.async_reader == nullptr || !slot_pool_enabled()) return false;
    if (expert_id < 0 || expert_id >= config_.expert_num || config_.should_skip_expert(expert_id)) return true;
    if (slot < 0 || slot >= cache_capacity_) return false;
    if (timing != nullptr) {
      timing->promote_calls += 1;
    }

#ifndef HAVE_LIBURING
    return false;
#else
    for (;;) {
      const auto state_start = std::chrono::steady_clock::now();
      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
        if (timing != nullptr) {
          timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
        }
        const int old_slot = expert_to_slot_[expert_id];
        if (old_slot == slot || old_slot < 0) {
          return true;
        }
        const auto slot_start = std::chrono::steady_clock::now();
        if (!unbind_slot(old_slot, true, false)) {
          if (timing != nullptr) {
            timing->slot_us += elapsed_us(slot_start, std::chrono::steady_clock::now());
          }
          return false;
        }
        if (timing != nullptr) {
          timing->slot_us += elapsed_us(slot_start, std::chrono::steady_clock::now());
        }
        continue;
      }
      if (state == EXPERT_PREFETCHING) {
        if (timing != nullptr) {
          timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
        }
        wait_for_pending_prefetch(expert_id, false);
        return true;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        if (timing != nullptr) {
          timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
        }
        std::this_thread::yield();
        continue;
      }
      if (state != EXPERT_BASELINE) {
        if (timing != nullptr) {
          timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
        }
        return false;
      }

      uint8_t expected = EXPERT_BASELINE;
      if (!expert_states_[expert_id].compare_exchange_strong(
              expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
        if (timing != nullptr) {
          timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
        }
        continue;
      }
      if (timing != nullptr) {
        timing->state_us += elapsed_us(state_start, std::chrono::steady_clock::now());
      }

      const auto slot_start = std::chrono::steady_clock::now();
      const int old_slot = expert_to_slot_[expert_id];
      if (old_slot >= 0 && old_slot != slot && !unbind_slot(old_slot, true, false)) {
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        if (timing != nullptr) {
          timing->slot_us += elapsed_us(slot_start, std::chrono::steady_clock::now());
        }
        return false;
      }

      if (!reserve_slot_for_loading(slot, expert_id, timing)) {
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        if (timing != nullptr) {
          timing->slot_us += elapsed_us(slot_start, std::chrono::steady_clock::now());
        }
        return false;
      }
      if (timing != nullptr) {
        timing->slot_us += elapsed_us(slot_start, std::chrono::steady_clock::now());
      }

      BatchPromotion promotion;
      promotion.expert_id = expert_id;
      promotion.slot = slot;
      promotion.gate_owner = slot_gate_owner_[slot];
      promotion.up_owner = slot_up_owner_[slot];
      promotion.down_owner = slot_down_owner_[slot];
      const size_t request_begin = read_batch->size();
      const auto request_build_start = std::chrono::steady_clock::now();
      try {
        append_iouring_read_requests_for_expert(expert_id,
                                                promotion.gate_owner,
                                                promotion.up_owner,
                                                promotion.down_owner,
                                                read_batch);
      } catch (...) {
        if (timing != nullptr) {
          timing->request_build_us += elapsed_us(request_build_start, std::chrono::steady_clock::now());
        }
        (void)unbind_slot(slot, true, false);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        throw;
      }
      if (timing != nullptr) {
        timing->request_build_us += elapsed_us(request_build_start, std::chrono::steady_clock::now());
      }
      promotion.request_count = read_batch->size() - request_begin;
      if (promotion.request_count == 0) {
        (void)unbind_slot(slot, true, false);
        expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        return false;
      }
      promotions->push_back(std::move(promotion));
      return true;
    }
#endif
  }

  void rollback_prefill_promotions(std::vector<BatchPromotion>* promotions) {
    if (promotions == nullptr) return;
    for (auto& promotion : *promotions) {
      if (promotion.slot >= 0) {
        (void)unbind_slot(promotion.slot, true, false);
      }
      if (promotion.expert_id >= 0 && promotion.expert_id < config_.expert_num) {
        expert_states_[promotion.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      }
    }
  }

  void submit_prefill_promotion_reads(std::vector<BatchPromotion>* promotions,
                                      const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch,
                                      std::vector<uint64_t>* all_requests,
                                      PrefillPromotionTiming* timing = nullptr) {
#ifndef HAVE_LIBURING
    (void)promotions;
    (void)read_batch;
    (void)all_requests;
    (void)timing;
    return;
#else
    mesh::submit_prefill_promotion_reads(config_, tp_part_idx, promotions, read_batch, all_requests, timing);
#endif
  }

  void wait_and_finalize_prefill_promotions(std::vector<BatchPromotion>* promotions,
                                            const std::vector<uint64_t>& all_requests,
                                            const char* label) {
    if (promotions == nullptr || all_requests.empty()) return;
    const int timeout_ms = 60000;
    if (!config_.async_reader->wait_for_requests(all_requests, timeout_ms)) {
      std::ostringstream oss;
      oss << "AMX io_uring prefill promotion failed layer=" << config_.layer_idx << " tp=" << tp_part_idx
          << " label=" << (label != nullptr ? label : "unknown") << " experts=" << promotions->size()
          << " requests=" << all_requests.size() << " inflight=" << config_.async_reader->get_inflight_count()
          << " timeout_ms=" << timeout_ms << " detail="
          << config_.async_reader->describe_requests(all_requests);
      for (auto& promotion : *promotions) {
        if (promotion.slot >= 0) {
          (void)unbind_slot(promotion.slot, true, false);
        }
        if (promotion.expert_id >= 0 && promotion.expert_id < config_.expert_num) {
          expert_states_[promotion.expert_id].store(EXPERT_BASELINE, std::memory_order_release);
        }
      }
      throw std::runtime_error(oss.str());
    }

    for (auto& promotion : *promotions) {
      finalize_expert_buffers(promotion.expert_id,
                              promotion.gate_owner,
                              promotion.up_owner,
                              promotion.down_owner,
                              false);
      mesh::record_cache_promote(config_, tp_part_idx, promotion.expert_id);
    }
  }

  int ensure_prefill_static_residents() {
    last_prefill_static_timing_ = PrefillPromotionTiming{};
    if (!config_.mesh_prefill_layer_mode_enabled || !resident_io_enabled() || !iouring_enabled() ||
        cache_capacity_ <= 0) {
      return 0;
    }
    if (!slot_pool_enabled() && !ensure_resident_slot_pool_allocated()) return 0;
    mesh_slot_mode_ = MESH_SLOT_PREFILL_LAYER;
    if (prefill_static_experts_.empty() && prefill_static_capacity_ > 0) {
      rebuild_prefill_static_expert_set();
    }

    std::vector<BatchPromotion> promotions;
    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    std::vector<uint64_t> all_requests;
    promotions.reserve(prefill_static_experts_.size());
    read_batch.reserve(prefill_static_experts_.size() * 9);
    for (int expert_id : prefill_static_experts_) {
      const int slot = prefill_static_slot_for_expert(expert_id);
      if (!submit_prefill_promotion_to_slot(
              expert_id, slot, &promotions, &read_batch, &last_prefill_static_timing_)) {
        std::ostringstream oss;
        oss << "AMX prefill static promotion could not reserve slot layer=" << config_.layer_idx
            << " tp=" << tp_part_idx << " expert=" << expert_id << " slot=" << slot
            << " static_capacity=" << prefill_static_capacity_
            << " static_slots=" << prefill_static_slot_count_
            << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
            << " capacity=" << cache_capacity_ << resident_debug_summary();
        throw std::runtime_error(oss.str());
      }
    }
    try {
      submit_prefill_promotion_reads(&promotions, read_batch, &all_requests, &last_prefill_static_timing_);
    } catch (...) {
      rollback_prefill_promotions(&promotions);
      throw;
    }
    wait_and_finalize_prefill_promotions(&promotions, all_requests, "static");
    return static_cast<int>(prefill_static_experts_.size());
  }

  void release_empty_slot_buffers_to_limit(int allocated_limit) {
    auto hooks = resident_slot_pool_hooks();
    mesh::release_empty_slot_buffers_to_limit(
        resident_slot_pool_view(), hooks, lazy_slot_buffers_enabled(), allocated_limit);
  }

  std::vector<int> cpu_managed_experts_by_heat() {
    refresh_lookahead_heat();
    return mesh::cpu_managed_experts_by_heat(config_, lookahead_heat_);
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
    mesh::record_cache_prefetch_submit(config_, static_cast<uint64_t>(submitted));
    return submitted;
  }

  int transition_to_decode_frequency_cache(int target_capacity, int fill_limit) {
    if (!config_.mesh_prefill_layer_mode_enabled || target_capacity <= 0 || !slot_pool_enabled()) return 0;
    rebuild_prefill_static_expert_set_by_frequency(target_capacity);
    if (prefill_static_experts_.empty()) return 0;

    const std::vector<uint8_t> desired_mask = prefill_static_expert_mask_;
    const int main_slots = decode_main_slot_count();

    for (int expert_id : prefill_static_experts_) {
      const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state != EXPERT_CACHED && state != EXPERT_PINNED) continue;
      const int slot = expert_to_slot_[expert_id];
      if (slot >= 0 && slot < main_slots) continue;
      if (slot < main_slots || slot >= cache_capacity_) continue;
      const int target_slot = find_main_slot_outside_frequency_set(desired_mask);
      if (target_slot >= 0) {
        (void)swap_slot_bindings(target_slot, slot);
      }
    }

    std::vector<BatchPromotion> promotions;
    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    std::vector<uint64_t> all_requests;
    promotions.reserve(prefill_static_experts_.size());
    read_batch.reserve(prefill_static_experts_.size() * 9);

    int submitted = 0;
    const int submit_limit = fill_limit > 0 ? fill_limit : std::numeric_limits<int>::max();
    for (int expert_id : prefill_static_experts_) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PREFETCHING) {
        wait_for_pending_prefetch(expert_id, false);
        state = expert_states_[expert_id].load(std::memory_order_acquire);
      }
      if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
        const int slot = expert_to_slot_[expert_id];
        if (slot >= 0 && slot < main_slots) {
          continue;
        }
        if (slot >= main_slots && slot < cache_capacity_) {
          const int target_slot = find_main_slot_outside_frequency_set(desired_mask);
          if (target_slot >= 0 && swap_slot_bindings(target_slot, slot)) {
            continue;
          }
        }
      }

      if (submitted >= submit_limit) break;
      const int target_slot = find_main_slot_outside_frequency_set(desired_mask);
      if (target_slot < 0) {
        std::ostringstream oss;
        oss << "MESH frequency decode transition has no replaceable main slot layer=" << config_.layer_idx
            << " tp=" << tp_part_idx << " expert=" << expert_id
            << " target_capacity=" << target_capacity
            << " main_slots=" << main_slots
            << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
            << " capacity=" << cache_capacity_ << resident_debug_summary();
        throw std::runtime_error(oss.str());
      }
      if (!submit_prefill_promotion_to_slot(expert_id, target_slot, &promotions, &read_batch, nullptr)) {
        std::ostringstream oss;
        oss << "MESH frequency decode transition could not reserve main slot layer=" << config_.layer_idx
            << " tp=" << tp_part_idx << " expert=" << expert_id
            << " slot=" << target_slot
            << " target_capacity=" << target_capacity
            << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
            << " capacity=" << cache_capacity_ << resident_debug_summary();
        throw std::runtime_error(oss.str());
      }
      submitted += 1;
    }

    try {
      submit_prefill_promotion_reads(&promotions, read_batch, &all_requests, nullptr);
    } catch (...) {
      rollback_prefill_promotions(&promotions);
      throw;
    }
    wait_and_finalize_prefill_promotions(&promotions, all_requests, "frequency_decode");
    trim_cache_to_capacity();
    release_empty_slot_buffers_to_limit(target_capacity);
    return submitted;
  }

  int prepare_prefill_layer_window() {
    if (!config_.mesh_prefill_layer_mode_enabled || !resident_io_enabled() || cache_capacity_ <= 0) return 0;
    return ensure_prefill_static_residents();
  }

  bool release_prefill_layer_window() {
    if (!config_.mesh_prefill_layer_mode_enabled || !resident_io_enabled()) return true;
    complete_ready_prefetches();
    return release_prefill_scratch_slots(true);
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
    if (config_.mesh_prefill_layer_mode_enabled && expert_frequency_total_ > 0) {
      return transition_to_decode_frequency_cache(target_capacity, fill_limit);
    }
    trim_cache_to_capacity();
    release_empty_slot_buffers_to_limit(target_capacity);
    const int submitted = submit_decode_heat_prefetches(target_capacity, fill_limit);
    return submitted;
  }

  void trim_cache_for_memory_pressure() {
    mesh::MemoryGuardTrimCheck check;
    if (!mesh::prepare_memory_guard_trim_check(
            config_, cache_capacity_, memory_guard_tick_, cgroup_memory_current_path_, cgroup_memory_max_path_, &check)) {
      return;
    }
    int demoted = 0;
    for (int attempt = 0; attempt < check.max_demotes; ++attempt) {
      if (!evict_one_cached_expert(-1)) break;
      demoted += 1;
      if ((attempt & 3) == 3) {
        if (!mesh::refresh_memory_guard_trim_check(
                cgroup_memory_current_path_, cgroup_memory_max_path_, &check)) {
          break;
        }
        if (mesh::memory_guard_trim_target_reached(check)) break;
      }
    }
    mesh::record_memory_guard_demotes(config_, static_cast<uint64_t>(demoted));
  }

  bool pending_prefetch_finished(int expert_id, bool* failed) const {
    return mesh::pending_prefetch_finished(pending_prefetches_[expert_id], config_.async_reader, failed);
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
      const std::string detail =
          mesh::amx_iouring_prefetch_wait_failure_message(config_, tp_part_idx, expert_id, requests, config_.async_reader);
      clear_pending_prefetch(expert_id, true);
      expert_states_[expert_id].store(EXPERT_BASELINE, std::memory_order_release);
      throw std::runtime_error(detail);
    }
    mesh::record_cache_prefetch_hit(config_, tp_part_idx, expert_id);
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
      mesh::record_cache_async_prefetch(config_);
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

    mesh::record_cache_total_access(config_, tp_part_idx, expert_id);

    bool counted_miss_for_this_access = false;
    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        if (!counted_miss_for_this_access) {
          mesh::record_cache_hit(config_, tp_part_idx, expert_id);
        }
        return;
      }
      if (state == EXPERT_CACHED) {
        if (!counted_miss_for_this_access) {
          mesh::record_cache_hit(config_, tp_part_idx, expert_id);
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
          return;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
        mesh::record_cache_cold_miss(config_, tp_part_idx, expert_id);
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
        mesh::record_cache_promote(config_, tp_part_idx, expert_id);
        continue;
      }
      if (state == EXPERT_PREFETCHING) {
        mesh::record_cache_inflight_miss(config_, tp_part_idx, expert_id);
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
    mesh::record_lookahead_update(config_);
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

    std::vector<uint8_t> protected_mask =
        mesh::build_protected_mask(config_, expert_ids, count, protect_ids, protect_count);

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
    mesh::record_cache_prefetch_submit(config_, static_cast<uint64_t>(submitted));
    mesh::record_bootstrap_prefetch_stats(config_,
                                          tp_part_idx,
                                          is_bootstrap_prefetch,
                                          bootstrap_candidate_count,
                                          static_cast<uint64_t>(submitted),
                                          bootstrap_skip_gpu_count,
                                          bootstrap_skip_resident_count);
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
