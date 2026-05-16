// This file is included inside AMX_BF16_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
               void* output) override {
#ifndef _WIN32
    std::vector<int> active_experts;
    std::vector<int> expert_counts;
    if (lazy_weight_enabled()) {
      mesh::collect_unique_cpu_experts(config_, expert_ids, qlen * k, active_experts, &expert_counts);

      const bool enable_wave_mode = mesh::bf16_wave_resident_enabled();
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
