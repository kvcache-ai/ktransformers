// This file is included inside AMX_BF16_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  void set_weight_buffers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

#ifndef _WIN32
  bool mesh_load_weights_from_lazy_sources() {
    if (!lazy_weight_enabled()) return false;
    validate_iouring_config();
    return true;
  }

  void cleanup_lazy_weight_state() {
    if (lazy_weight_enabled()) {
      for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
        free_packed_expert(expert_id);
      }
      return;
    }
    for (int expert_id = 0; expert_id < config_.expert_num; ++expert_id) {
      std::free(gate_bb_[expert_id]->b);
      std::free(up_bb_[expert_id]->b);
      std::free(down_bb_[expert_id]->b);
      gate_bb_[expert_id]->b = nullptr;
      up_bb_[expert_id]->b = nullptr;
      down_bb_[expert_id]->b = nullptr;
    }
  }

  void initialize_lazy_weight_state() {
    const int en = config_.expert_num;
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

  void validate_iouring_config() const {
    if (!iouring_enabled()) return;
    mesh::validate_bf16_iouring_config(config_,
                                       tp_part_idx,
                                       cache_capacity_,
                                       gate_weight_bytes_,
                                       up_weight_bytes_,
                                       full_down_weight_bytes(),
                                       down_weight_bytes_);
  }

  void acquire_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_add(1, std::memory_order_acq_rel);
  }

  void release_expert_read(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    active_readers_[expert_id].fetch_sub(1, std::memory_order_acq_rel);
  }

  using ExpertReadScope = mesh::ExpertReadScope<AMX_BF16_MOE_TP<T>>;

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

  void record_iouring_read_stats(const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch) {
    mesh::record_iouring_read_stats(config_.cache_stats, config_.enable_cache_stats, read_batch);
  }

  int cache_capacity_for_debug() const { return cache_capacity_; }

  bool read_and_pack_iouring_expert(int expert_id, void* gate_owner, void* up_owner, void* down_owner) {
    const size_t down_full_bytes = full_down_weight_bytes();
    void* gate_raw = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_raw = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_full_raw = numa_alloc_onnode(down_full_bytes, numa_node_);
    if (gate_raw == nullptr || up_raw == nullptr || down_full_raw == nullptr) {
      if (gate_raw) numa_free(gate_raw, gate_weight_bytes_);
      if (up_raw) numa_free(up_raw, up_weight_bytes_);
      if (down_full_raw) numa_free(down_full_raw, down_full_bytes);
      mesh::log_bf16_iouring_temp_alloc_failure(expert_id, gate_raw, up_raw, down_full_raw);
      return false;
    }

    std::vector<ktransformers::AsyncExpertReader::ReadRequest> read_batch;
    read_batch.reserve(3);
    mesh::append_bf16_iouring_read_requests_for_expert(config_,
                                                       tp_part_idx,
                                                       expert_id,
                                                       gate_raw,
                                                       up_raw,
                                                       down_full_raw,
                                                       gate_weight_bytes_,
                                                       up_weight_bytes_,
                                                       down_full_bytes,
                                                       &read_batch);

    record_iouring_read_stats(read_batch);
    std::vector<uint64_t> request_ids = config_.async_reader->submit_reads(read_batch);
    const bool ok = config_.async_reader->wait_for_requests(request_ids, 30000);
    if (!ok) {
      mesh::log_bf16_iouring_read_failure(expert_id, request_ids, config_.async_reader);
      numa_free(gate_raw, gate_weight_bytes_);
      numa_free(up_raw, up_weight_bytes_);
      numa_free(down_full_raw, down_full_bytes);
      return false;
    }

    gate_bb_[expert_id]->set_data(gate_owner);
    up_bb_[expert_id]->set_data(up_owner);
    down_bb_[expert_id]->set_data(down_owner);

    gate_bb_[expert_id]->from_mat(static_cast<ggml_bf16_t*>(gate_raw), 0, 1);
    up_bb_[expert_id]->from_mat(static_cast<ggml_bf16_t*>(up_raw), 0, 1);

    if (full_intermediate_size_ == config_.intermediate_size) {
      down_bb_[expert_id]->from_mat(static_cast<ggml_bf16_t*>(down_full_raw), 0, 1);
    } else {
      std::vector<ggml_bf16_t> down_contiguous((size_t)config_.hidden_size * (size_t)config_.intermediate_size);
      const size_t down_col_offset = (size_t)tp_part_idx * (size_t)config_.intermediate_size;
      const auto* src = static_cast<const ggml_bf16_t*>(down_full_raw);
      for (int row = 0; row < config_.hidden_size; ++row) {
        const ggml_bf16_t* src_row = src + (size_t)row * (size_t)full_intermediate_size_ + down_col_offset;
        ggml_bf16_t* dst_row = down_contiguous.data() + (size_t)row * (size_t)config_.intermediate_size;
        std::memcpy(dst_row, src_row, sizeof(ggml_bf16_t) * (size_t)config_.intermediate_size);
      }
      down_bb_[expert_id]->from_mat(down_contiguous.data(), 0, 1);
    }

    numa_free(gate_raw, gate_weight_bytes_);
    numa_free(up_raw, up_weight_bytes_);
    numa_free(down_full_raw, down_full_bytes);
    return true;
  }

  bool begin_external_iouring_pack(int expert_id, bool pin_after_pack) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return false;

    for (;;) {
      uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
      if (state == EXPERT_PINNED) {
        return false;
      }
      if (state == EXPERT_CACHED) {
        if (!pin_after_pack) {
          return false;
        }
        uint8_t expected = EXPERT_CACHED;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PINNED, std::memory_order_acq_rel, std::memory_order_acquire)) {
          note_expert_pin(expert_id);
          return false;
        }
        continue;
      }
      if (state == EXPERT_BASELINE) {
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

        uint8_t expected = EXPERT_BASELINE;
        if (expert_states_[expert_id].compare_exchange_strong(
                expected, EXPERT_PACKING, std::memory_order_acq_rel, std::memory_order_acquire)) {
          return true;
        }
        continue;
      }
      if (state == EXPERT_PACKING || state == EXPERT_DEMOTING) {
        std::this_thread::yield();
        continue;
      }
      return false;
    }
  }

  void cancel_external_iouring_pack(int expert_id) {
    if (expert_id < 0 || expert_id >= config_.expert_num) return;
    uint8_t expected = EXPERT_PACKING;
    (void)expert_states_[expert_id].compare_exchange_strong(
        expected, EXPERT_BASELINE, std::memory_order_acq_rel, std::memory_order_acquire);
  }

  bool pack_iouring_expert_from_raw_slices(int expert_id,
                                           const ggml_bf16_t* gate_raw,
                                           const ggml_bf16_t* up_raw,
                                           const ggml_bf16_t* down_full_raw,
                                           bool pin_after_pack) {
    void* gate_owner = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_weight_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      mesh::log_bf16_numa_alloc_failure(expert_id,
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
      cancel_external_iouring_pack(expert_id);
      return false;
    }

    gate_bb_[expert_id]->set_data(gate_owner);
    up_bb_[expert_id]->set_data(up_owner);
    down_bb_[expert_id]->set_data(down_owner);

    gate_bb_[expert_id]->from_mat(const_cast<ggml_bf16_t*>(gate_raw), 0, 1);
    up_bb_[expert_id]->from_mat(const_cast<ggml_bf16_t*>(up_raw), 0, 1);

    if (full_intermediate_size_ == config_.intermediate_size) {
      down_bb_[expert_id]->from_mat(const_cast<ggml_bf16_t*>(down_full_raw), 0, 1);
    } else {
      std::vector<ggml_bf16_t> down_contiguous((size_t)config_.hidden_size * (size_t)config_.intermediate_size);
      const size_t down_col_offset = (size_t)tp_part_idx * (size_t)config_.intermediate_size;
      for (int row = 0; row < config_.hidden_size; ++row) {
        const ggml_bf16_t* src_row =
            down_full_raw + (size_t)row * (size_t)full_intermediate_size_ + down_col_offset;
        ggml_bf16_t* dst_row = down_contiguous.data() + (size_t)row * (size_t)config_.intermediate_size;
        std::memcpy(dst_row, src_row, sizeof(ggml_bf16_t) * (size_t)config_.intermediate_size);
      }
      down_bb_[expert_id]->from_mat(down_contiguous.data(), 0, 1);
    }

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_pack ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_pack);
    return true;
  }

  bool allocate_and_pack_expert(int expert_id, bool pin_after_pack) {
    if (!iouring_enabled()) {
      return false;
    }
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
        mesh::log_bf16_cache_full_failure(config_,
                                          expert_id,
                                          resident_expert_count_.load(std::memory_order_acquire),
                                          cache_capacity_,
                                          cached_count,
                                          pinned_count,
                                          packing_count,
                                          demoting_count,
                                          busy_count,
                                          resident_bytes_per_expert,
                                          promotion_temp_bytes);
        return false;
      }
    }

    void* gate_owner = numa_alloc_onnode(gate_weight_bytes_, numa_node_);
    void* up_owner = numa_alloc_onnode(up_weight_bytes_, numa_node_);
    void* down_owner = numa_alloc_onnode(down_weight_bytes_, numa_node_);
    if (gate_owner == nullptr || up_owner == nullptr || down_owner == nullptr) {
      mesh::log_bf16_numa_alloc_failure(expert_id,
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

    if (!read_and_pack_iouring_expert(expert_id, gate_owner, up_owner, down_owner)) {
      numa_free(gate_owner, gate_weight_bytes_);
      numa_free(up_owner, up_weight_bytes_);
      numa_free(down_owner, down_weight_bytes_);
      return false;
    }

    packed_gate_owner_[expert_id] = gate_owner;
    packed_up_owner_[expert_id] = up_owner;
    packed_down_owner_[expert_id] = down_owner;
    resident_expert_count_.fetch_add(1, std::memory_order_acq_rel);
    expert_states_[expert_id].store(pin_after_pack ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
    note_expert_insert(expert_id, pin_after_pack);
    mesh::maybe_log_cpu_mem_resident(config_,
                                     tp_part_idx,
                                     expert_id,
                                     resident_expert_count_.load(std::memory_order_acquire),
                                     cache_capacity_,
                                     resident_bytes_per_expert,
                                     promotion_temp_bytes,
                                     pin_after_pack);
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
