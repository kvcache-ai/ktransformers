// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

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
      mesh::collect_unique_cpu_experts(config_, expert_ids, qlen * k, active_experts);

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
