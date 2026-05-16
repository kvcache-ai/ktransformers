// This file is included inside TP_MOE_Common<T>.
// Keep declarations free of namespace wrappers and include guards.

  void forward_binding_with_scores(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input,
                                   intptr_t output, bool incremental, intptr_t router_scores, int score_rows,
                                   int score_cols, int score_transform) {
    observe_router_scores_binding(router_scores, score_rows, score_cols, score_transform);
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  std::map<std::string, double> cache_stats_snapshot() const {
    const ExpertCacheStats* stats = config.cache_stats;
    if (stats == nullptr) {
      return {
          {"enabled", 0.0},
          {"promote_count", 0.0},
          {"demote_count", 0.0},
          {"hit_count", 0.0},
          {"miss_count", 0.0},
          {"cold_miss_count", 0.0},
          {"in_flight_miss_count", 0.0},
          {"eviction_count", 0.0},
          {"total_access_count", 0.0},
          {"lookahead_update_count", 0.0},
          {"prefetch_count", 0.0},
          {"full_score_update_count", 0.0},
          {"async_prefetch_count", 0.0},
          {"prefetch_hit_count", 0.0},
          {"iouring_read_request_count", 0.0},
          {"iouring_read_bytes", 0.0},
          {"transition_update_count", 0.0},
          {"coldstart_prefill_count", 0.0},
          {"bootstrap_prefetch_candidate_count", 0.0},
          {"bootstrap_prefetch_submit_count", 0.0},
          {"bootstrap_prefetch_skip_gpu_count", 0.0},
          {"bootstrap_prefetch_skip_resident_count", 0.0},
          {"memory_guard_demote_count", 0.0},
          {"state_defer_token_count", 0.0},
          {"state_defer_cpu_topk_count", 0.0},
          {"state_defer_gpu_skip_count", 0.0},
          {"state_defer_nonready_count", 0.0},
          {"state_defer_deferred_count", 0.0},
          {"state_defer_overflow_immediate_count", 0.0},
          {"state_defer_overflow_token_count", 0.0},
          {"hit_rate", 0.0},
      };
    }
    return {
        {"enabled", config.enable_cache_stats ? 1.0 : 0.0},
        {"promote_count", static_cast<double>(stats->promote_count.load(std::memory_order_relaxed))},
        {"demote_count", static_cast<double>(stats->demote_count.load(std::memory_order_relaxed))},
        {"hit_count", static_cast<double>(stats->hit_count.load(std::memory_order_relaxed))},
        {"miss_count", static_cast<double>(stats->miss_count.load(std::memory_order_relaxed))},
        {"cold_miss_count", static_cast<double>(stats->cold_miss_count.load(std::memory_order_relaxed))},
        {"in_flight_miss_count", static_cast<double>(stats->in_flight_miss_count.load(std::memory_order_relaxed))},
        {"eviction_count", static_cast<double>(stats->eviction_count.load(std::memory_order_relaxed))},
        {"total_access_count", static_cast<double>(stats->total_access_count.load(std::memory_order_relaxed))},
        {"lookahead_update_count", static_cast<double>(stats->lookahead_update_count.load(std::memory_order_relaxed))},
        {"prefetch_count", static_cast<double>(stats->prefetch_count.load(std::memory_order_relaxed))},
        {"full_score_update_count", static_cast<double>(stats->full_score_update_count.load(std::memory_order_relaxed))},
        {"async_prefetch_count", static_cast<double>(stats->async_prefetch_count.load(std::memory_order_relaxed))},
        {"prefetch_hit_count", static_cast<double>(stats->prefetch_hit_count.load(std::memory_order_relaxed))},
        {"iouring_read_request_count", static_cast<double>(stats->iouring_read_request_count.load(std::memory_order_relaxed))},
        {"iouring_read_bytes", static_cast<double>(stats->iouring_read_bytes.load(std::memory_order_relaxed))},
        {"transition_update_count", static_cast<double>(stats->transition_update_count.load(std::memory_order_relaxed))},
        {"coldstart_prefill_count", static_cast<double>(stats->coldstart_prefill_count.load(std::memory_order_relaxed))},
        {"bootstrap_prefetch_candidate_count", static_cast<double>(stats->bootstrap_prefetch_candidate_count.load(std::memory_order_relaxed))},
        {"bootstrap_prefetch_submit_count", static_cast<double>(stats->bootstrap_prefetch_submit_count.load(std::memory_order_relaxed))},
        {"bootstrap_prefetch_skip_gpu_count", static_cast<double>(stats->bootstrap_prefetch_skip_gpu_count.load(std::memory_order_relaxed))},
        {"bootstrap_prefetch_skip_resident_count", static_cast<double>(stats->bootstrap_prefetch_skip_resident_count.load(std::memory_order_relaxed))},
        {"memory_guard_demote_count", static_cast<double>(stats->memory_guard_demote_count.load(std::memory_order_relaxed))},
        {"state_defer_token_count", static_cast<double>(stats->state_defer_token_count.load(std::memory_order_relaxed))},
        {"state_defer_cpu_topk_count", static_cast<double>(stats->state_defer_cpu_topk_count.load(std::memory_order_relaxed))},
        {"state_defer_gpu_skip_count", static_cast<double>(stats->state_defer_gpu_skip_count.load(std::memory_order_relaxed))},
        {"state_defer_nonready_count", static_cast<double>(stats->state_defer_nonready_count.load(std::memory_order_relaxed))},
        {"state_defer_deferred_count", static_cast<double>(stats->state_defer_deferred_count.load(std::memory_order_relaxed))},
        {"state_defer_overflow_immediate_count", static_cast<double>(stats->state_defer_overflow_immediate_count.load(std::memory_order_relaxed))},
        {"state_defer_overflow_token_count", static_cast<double>(stats->state_defer_overflow_token_count.load(std::memory_order_relaxed))},
        {"hit_rate", stats->hit_rate()},
    };
  }

  void observe_router_scores_binding(intptr_t scores, int rows, int cols, int score_transform) {
    if (!config.mesh_lookahead_enabled || scores == 0 || rows <= 0 || cols <= 0) return;
    mesh_lookahead_registry().observe_scores(config.layer_idx,
                                             config.expert_num,
                                             reinterpret_cast<const float*>(scores),
                                             rows,
                                             cols,
                                             config.gpu_experts_mask,
                                             config.mesh_heat_gamma,
                                             config.mesh_heat_beta,
                                             score_transform,
                                             config.mesh_transition_alpha);
    if (config.enable_cache_stats && config.cache_stats != nullptr) {
      config.cache_stats->lookahead_update_count.fetch_add(1, std::memory_order_relaxed);
      config.cache_stats->full_score_update_count.fetch_add(1, std::memory_order_relaxed);
      if (config.mesh_transition_alpha > 0.0f) {
        config.cache_stats->transition_update_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  void observe_router_scores_batch_binding(intptr_t scores,
                                           int score_stride,
                                           int cols,
                                           intptr_t layer_indices,
                                           intptr_t score_transforms,
                                           intptr_t gpu_experts_masks,
                                           int layer_count) {
    if (!config.mesh_lookahead_enabled || scores == 0 || score_stride <= 0 || cols <= 0 || layer_count <= 0) return;
    const float* score_base = reinterpret_cast<const float*>(scores);
    const int32_t* layers = reinterpret_cast<const int32_t*>(layer_indices);
    const int32_t* transforms = reinterpret_cast<const int32_t*>(score_transforms);
    const uint8_t* masks = reinterpret_cast<const uint8_t*>(gpu_experts_masks);

    int observed_layers = 0;
    for (int i = 0; i < layer_count; ++i) {
      const int layer_idx = layers == nullptr ? i : static_cast<int>(layers[i]);
      if (layer_idx < 0) continue;
      const int score_transform = transforms == nullptr ? 0 : static_cast<int>(transforms[i]);
      const float* row_scores = score_base + static_cast<size_t>(layer_idx) * static_cast<size_t>(score_stride);
      const uint8_t* layer_mask = masks == nullptr ? config.gpu_experts_mask
                                                   : masks + static_cast<size_t>(i) * static_cast<size_t>(cols);
      mesh_lookahead_registry().observe_scores(layer_idx,
                                               config.expert_num,
                                               row_scores,
                                               1,
                                               cols,
                                               layer_mask,
                                               config.mesh_heat_gamma,
                                               config.mesh_heat_beta,
                                               score_transform,
                                               config.mesh_transition_alpha);
      observed_layers++;
    }

    if (observed_layers > 0 && config.enable_cache_stats && config.cache_stats != nullptr) {
      config.cache_stats->lookahead_update_count.fetch_add(static_cast<uint64_t>(observed_layers),
                                                           std::memory_order_relaxed);
      config.cache_stats->full_score_update_count.fetch_add(static_cast<uint64_t>(observed_layers),
                                                            std::memory_order_relaxed);
      if (config.mesh_transition_alpha > 0.0f) {
        config.cache_stats->transition_update_count.fetch_add(static_cast<uint64_t>(observed_layers),
                                                              std::memory_order_relaxed);
      }
    }
  }

  void reset_cache_stats() {
    if (config.cache_stats != nullptr) {
      config.cache_stats->reset();
    }
  }
