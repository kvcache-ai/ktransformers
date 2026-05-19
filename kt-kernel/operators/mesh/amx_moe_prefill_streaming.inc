// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

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
    record_prefill_frequency_counts();
    maybe_dump_prefill_expert_frequency(qlen, k, expert_ids, active_experts);

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

    if (config_.mesh_prefill_layer_mode_enabled && expert_frequency_total_ > 0 && prefill_static_capacity_ > 0) {
      rebuild_prefill_static_expert_set_by_frequency(prefill_static_capacity_);
    }
    ensure_prefill_static_residents();
    // Unbind scratch slots from the previous scheduling round. Pool-backed
    // buffers return to IDLE with metadata preserved for warm restore; fallback
    // numa_alloc buffers are freed immediately.
    if (!release_prefill_scratch_slots()) {
      std::ostringstream oss;
      oss << "AMX prefill scratch slots are still busy before scheduling layer=" << config_.layer_idx
          << " tp=" << tp_part_idx << " static_slots=" << prefill_static_slot_count_
          << " capacity=" << cache_capacity_ << resident_debug_summary();
      throw std::runtime_error(oss.str());
    }

    std::vector<int> static_active_experts;
    std::vector<int> scratch_active_experts;
    mesh::split_prefill_static_and_scratch_experts(
        active_experts, prefill_static_expert_mask_, static_active_experts, scratch_active_experts);

    std::vector<uint8_t> protected_mask = mesh::build_protected_mask(config_, active_experts);
    eviction_protected_mask_ = &protected_mask;
    prefill_streaming_active_ = true;
    struct PrefillEvictionGuard {
      AMX_MOE_TP* owner;
      ~PrefillEvictionGuard() {
        owner->eviction_protected_mask_ = nullptr;
        owner->prefill_streaming_active_ = false;
      }
    } prefill_guard{this};

    bool include_mins = false;
    if constexpr (requires { gate_bb_[0]->mins; }) {
      include_mins = true;
    }

    // Cross-chunk warm restore: probe the global scratch pool for any active
    // scratch expert whose weight is still resident in a pool block (placed
    // there by a prior layer / SGLang prefill chunk and not yet covered by a
    // newer acquire). On hit we bind the pool block to a scratch slot and
    // promote the expert to EXPERT_CACHED — this is what closes the
    // "memory reused but expert content NOT reused" gap that drove the
    // measured 3x read amplification.
    //
    // Must run BEFORE count_prefill_scratch_states so the stat snapshot
    // reflects the post-restore state, and BEFORE submit_scratch_chunk so
    // restored experts are skipped by its EXPERT_CACHED state check.
    const int warm_restored = try_warm_restore_scratch_experts(
        scratch_active_experts,
        std::max(0, std::min(prefill_static_slot_count_, cache_capacity_)),
        protected_mask);
    (void)warm_restored;  // counter consumed via standard cache_stats path

    const mesh::PrefillScratchStateCounts scratch_state_before = mesh::count_prefill_scratch_states(
        config_, tp_part_idx, scratch_active_experts, expert_states_.get(), include_mins);

    const uint64_t read_reqs_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
            : 0;
    const uint64_t read_bytes_before =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
            : 0;
    const int resident_before = resident_expert_count_.load(std::memory_order_acquire);

    std::vector<BatchPromotion> scratch_promotions;
    std::vector<ktransformers::AsyncExpertReader::ReadRequest> scratch_read_batch;
    std::vector<uint64_t> scratch_requests;
    const int scratch_begin_slot = std::max(0, std::min(prefill_static_slot_count_, cache_capacity_));
    const int scratch_slot_capacity = cache_capacity_ - scratch_begin_slot;
    if (!scratch_active_experts.empty() && scratch_slot_capacity <= 0) {
      std::ostringstream oss;
      oss << "AMX prefill scratch has no slot capacity layer=" << config_.layer_idx
          << " tp=" << tp_part_idx
          << " static_slots=" << prefill_static_slot_count_
          << " scratch_active=" << scratch_active_experts.size()
          << " capacity=" << cache_capacity_ << resident_debug_summary();
      throw std::runtime_error(oss.str());
    }
    scratch_promotions.reserve(std::min<size_t>(scratch_active_experts.size(), scratch_slot_capacity));
    scratch_read_batch.reserve(std::min<size_t>(scratch_active_experts.size(), scratch_slot_capacity) * 9);
    PrefillPromotionTiming scratch_timing;
    uint64_t scratch_submit_us = 0;
    auto submit_scratch_chunk = [&](size_t begin,
                                    size_t end,
                                    std::vector<BatchPromotion>* promotions,
                                    std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch,
                                    std::vector<uint64_t>* requests) {
      promotions->clear();
      read_batch->clear();
      requests->clear();
      int next_scratch_slot = scratch_begin_slot;
      // Slot picker: from the current cursor, skip past any scratch slot that
      // currently holds an expert in protected_mask (i.e. an expert this layer's
      // active set still needs). This is what keeps chunk-2's already-cached
      // experts from being evicted by chunk-2's own new submissions.
      auto advance_to_evictable_slot = [&]() {
        while (next_scratch_slot < cache_capacity_) {
          const int slot_expert = slot_to_expert_[next_scratch_slot];
          if (slot_expert >= 0 && slot_expert < static_cast<int>(protected_mask.size()) &&
              protected_mask[slot_expert]) {
            next_scratch_slot += 1;
            continue;
          }
          return;
        }
      };
      for (size_t idx = begin; idx < end; ++idx) {
        const int expert_id = scratch_active_experts[idx];
        const uint8_t state = expert_states_[expert_id].load(std::memory_order_acquire);
        if (state != EXPERT_CACHED && state != EXPERT_PINNED && state != EXPERT_PREFETCHING) {
          advance_to_evictable_slot();
          if (next_scratch_slot >= cache_capacity_) {
            std::ostringstream oss;
            oss << "AMX prefill scratch has no free slot layer=" << config_.layer_idx
                << " tp=" << tp_part_idx << " expert=" << expert_id
                << " static_slots=" << prefill_static_slot_count_
                << " scratch_active=" << scratch_active_experts.size()
                << " chunk_begin=" << begin
                << " chunk_end=" << end
                << " capacity=" << cache_capacity_ << resident_debug_summary();
            throw std::runtime_error(oss.str());
          }
          if (!submit_prefill_promotion_to_slot(
                  expert_id, next_scratch_slot, promotions, read_batch, &scratch_timing)) {
            std::ostringstream oss;
            oss << "AMX prefill scratch promotion could not reserve slot layer=" << config_.layer_idx
                << " tp=" << tp_part_idx << " expert=" << expert_id
                << " slot=" << next_scratch_slot
                << " static_slots=" << prefill_static_slot_count_
                << " resident=" << resident_expert_count_.load(std::memory_order_acquire)
                << " capacity=" << cache_capacity_ << resident_debug_summary();
            throw std::runtime_error(oss.str());
          }
          next_scratch_slot += 1;
        }
      }
      try {
        submit_prefill_promotion_reads(promotions, *read_batch, requests, &scratch_timing);
      } catch (...) {
        rollback_prefill_promotions(promotions);
        throw;
      }
    };

    const auto scratch_submit_start = std::chrono::steady_clock::now();
    size_t scratch_next = 0;
    const size_t first_scratch_end =
        std::min(scratch_active_experts.size(), static_cast<size_t>(std::max(0, scratch_slot_capacity)));
    std::vector<int> scratch_compute_group;
    if (first_scratch_end > 0) {
      scratch_compute_group.assign(scratch_active_experts.begin(), scratch_active_experts.begin() + first_scratch_end);
      submit_scratch_chunk(0, first_scratch_end, &scratch_promotions, &scratch_read_batch, &scratch_requests);
      scratch_next = first_scratch_end;
    }
    const auto scratch_submit_end = std::chrono::steady_clock::now();
    scratch_submit_us += elapsed_us(scratch_submit_start, scratch_submit_end);

    auto compute_expert_group = [&](const std::vector<int>& group) -> uint64_t {
      if (group.empty()) return 0;
      ExpertReadScope expert_read_scope(this, group.size());
      for (int i = 0; i < static_cast<int>(group.size()); ++i) {
        const int expert_id = group[i];
        this->m_expert_id_map_[i] = expert_id;
        note_expert_access(expert_id);
        ensure_expert_ready(expert_id, false);
        expert_read_scope.add_expert(expert_id);
      }

      const auto compute_start = std::chrono::steady_clock::now();
      direct_or_pool(static_cast<int>(group.size()), [this](int task_id) {
        int expert_idx = this->m_expert_id_map_[task_id];
        this->gate_up_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                this->m_local_input_ptr_[expert_idx], 0, 1);
      });

      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * static_cast<int>(group.size()) * 2, [](int _) { T::config(); },
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

      this->apply_activation(static_cast<int>(group.size()), nth, qlen);

      pool->do_work_stealing_job(
          static_cast<int>(group.size()), nullptr,
          [this](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id];
            this->down_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);

      nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * static_cast<int>(group.size()), [](int _) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_down_gemm(expert_idx, ith, nth, qlen);
            this->down_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
      const auto compute_end = std::chrono::steady_clock::now();
      return elapsed_us(compute_start, compute_end);
    };

    const uint64_t static_compute_us = compute_expert_group(static_active_experts);
    uint64_t scratch_wait_us = 0;
    uint64_t scratch_compute_us = 0;
    auto finish_scratch_chunk = [&](const std::vector<int>& group,
                                    std::vector<BatchPromotion>* promotions,
                                    const std::vector<uint64_t>& requests) {
      if (group.empty()) return;
      const auto scratch_wait_start = std::chrono::steady_clock::now();
      wait_and_finalize_prefill_promotions(promotions, requests, "scratch");
      const auto scratch_wait_end = std::chrono::steady_clock::now();
      scratch_wait_us += elapsed_us(scratch_wait_start, scratch_wait_end);
      scratch_compute_us += compute_expert_group(group);
      for (int expert_id : group) {
        (void)maybe_promote_prefill_scratch_expert_by_frequency(expert_id, &protected_mask);
      }
      // INTENTIONALLY do NOT call release_prefill_scratch_slots here.
      //
      // Why: each chunk's experts often overlap heavily with the next chunk's
      // active set (same prompt, adjacent tokens hit similar router top-k).
      // Unbinding on every chunk forces the next chunk to re-read every expert
      // from disk even if it's already in a slot. Measured 3x read amplification
      // (25 GiB vs historical 8 GiB) when this was active.
      //
      // Instead let reserve_slot_for_loading do natural eviction on the next
      // chunk: protected_mask already pins this chunk's still-needed experts,
      // and the chunk-2 active set's protected_mask also pins what chunk-2
      // needs. Overlapping experts stay resident, non-overlapping ones get
      // evicted only when the new chunk actually needs the slot.
      //
      // Final teardown still runs through release_prefill_layer_window().
    };

    finish_scratch_chunk(scratch_compute_group, &scratch_promotions, scratch_requests);
    while (scratch_next < scratch_active_experts.size()) {
      const size_t chunk_end =
          std::min(scratch_active_experts.size(), scratch_next + static_cast<size_t>(scratch_slot_capacity));
      scratch_compute_group.assign(scratch_active_experts.begin() + scratch_next,
                                   scratch_active_experts.begin() + chunk_end);
      const auto chunk_submit_start = std::chrono::steady_clock::now();
      submit_scratch_chunk(scratch_next, chunk_end, &scratch_promotions, &scratch_read_batch, &scratch_requests);
      const auto chunk_submit_end = std::chrono::steady_clock::now();
      scratch_submit_us += elapsed_us(chunk_submit_start, chunk_submit_end);
      finish_scratch_chunk(scratch_compute_group, &scratch_promotions, scratch_requests);
      scratch_next = chunk_end;
    }

    const uint64_t read_reqs_after =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_request_count.load(std::memory_order_relaxed)
            : read_reqs_before;
    const uint64_t read_bytes_after =
        config_.cache_stats != nullptr
            ? config_.cache_stats->iouring_read_bytes.load(std::memory_order_relaxed)
            : read_bytes_before;

    if (stream_trace) {
      mesh::PrefillStaticScratchTraceData trace_data;
      trace_data.qlen = qlen;
      trace_data.active = activated_expert;
      trace_data.static_active = static_active_experts.size();
      trace_data.scratch_active = scratch_active_experts.size();
      trace_data.scratch_state = scratch_state_before;
      trace_data.static_slots = prefill_static_slot_count_;
      trace_data.scratch_slots = cache_capacity_ - prefill_static_slot_count_;
      trace_data.scratch_submit_us = scratch_submit_us;
      trace_data.scratch_wait_us = scratch_wait_us;
      trace_data.static_compute_us = static_compute_us;
      trace_data.scratch_compute_us = scratch_compute_us;
      trace_data.read_req_delta = read_reqs_after - read_reqs_before;
      trace_data.read_bytes_delta = read_bytes_after - read_bytes_before;
      trace_data.resident_before = resident_before;
      trace_data.resident_after = resident_expert_count_.load(std::memory_order_acquire);
      trace_data.allocated_slots = allocated_slot_count();
      trace_data.scratch_timing = scratch_timing;
      trace_data.static_timing = last_prefill_static_timing_;
      mesh::log_prefill_static_scratch_trace(config_, tp_part_idx, trace_data);
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
      mesh::log_prefill_stream_summary(config_,
                                        tp_part_idx,
                                        qlen,
                                        activated_expert,
                                        static_active_experts.size(),
                                        scratch_active_experts.size(),
                                        prefill_static_slot_count_,
                                        cache_capacity_ - prefill_static_slot_count_,
                                        elapsed_us(copy_input_start, copy_input_end),
                                        elapsed_us(merge_start, merge_end),
                                        elapsed_us(forward_start, forward_end),
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
