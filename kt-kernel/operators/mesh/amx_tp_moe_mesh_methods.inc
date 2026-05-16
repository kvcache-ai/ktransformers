// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

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
    mesh::StateDeferStats defer_stats;
    defer_stats.token_count = static_cast<uint64_t>(rows);
    for (int row = 0; row < rows; ++row) {
      int deferred_in_row = 0;
      int overflow_in_row = 0;
      const int row_begin = row * k;
      const int row_end = std::min(count, row_begin + k);
      for (int idx = row_begin; idx < row_end; ++idx) {
        const int expert_id = static_cast<int>(ids[idx]);
        if (this->config.should_skip_expert(expert_id)) {
          defer_stats.gpu_skip_count += 1;
          continue;
        }
        defer_stats.cpu_topk_count += 1;

        const bool ready = this->tps[0]->is_expert_ready_resident(expert_id);
        if (!ready) {
          defer_stats.nonready_count += 1;
        }
        if (!ready && deferred_in_row < defer_limit) {
          deferred_ids[idx] = expert_id;
          deferred_in_row += 1;
          defer_stats.deferred_count += 1;
          if (this->tps[0]->is_expert_cold_baseline(expert_id)) {
            cold_deferred.push_back(expert_id);
          }
          continue;
        }

        immediate_ids[idx] = expert_id;
        protected_immediate.push_back(expert_id);
        if (!ready) {
          defer_stats.overflow_immediate_count += 1;
          overflow_in_row += 1;
        }
      }
      if (overflow_in_row > 0) {
        defer_stats.overflow_token_count += 1;
      }
    }

    mesh::record_state_defer_stats(this->config, defer_stats);

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

