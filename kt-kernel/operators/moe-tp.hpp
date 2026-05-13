#ifndef CPUINFER_OPERATOR_MOE_HPP
#define CPUINFER_OPERATOR_MOE_HPP

// #define CHECK

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "../cpu_backend/shared_mem_buffer.h"
#include "common.hpp"

// Forward declaration for Llamafile backend type checking
class LLAMA_MOE_TP;

template <typename T>
concept MOE_TP_PART = requires(T t, int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                               void* output, GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;
  { t.forward(qlen, k, expert_ids, weights, input, output) } -> std::same_as<void>;
  // { t.load_weights() } -> std::same_as<void>;
};

template <MOE_TP_PART T>
class TP_MOE_Common : public MoE_Interface {
 protected:
  std::vector<GeneralMOEConfig> tp_configs;
  int tp_count;
  int me_numa_id;
  std::vector<std::unique_ptr<T>> tps;

  std::vector<typename T::output_t*> local_output_numa;
  typename T::output_t* local_output = nullptr;

  bool weights_loaded = false;

#ifdef FORWARD_TIME_REPORT
  size_t forward_time_sum_ns = 0;
  size_t forward_count = 0;
#endif
 public:
  GeneralMOEConfig config;
  using input_t = typename T::input_t;
  TP_MOE_Common(const GeneralMOEConfig& config) : config(config) {
    if (this->config.enable_cache_stats && this->config.cache_stats == nullptr) {
      this->config.cache_stats_owner = std::make_shared<ExpertCacheStats>();
      this->config.cache_stats = this->config.cache_stats_owner.get();
    }
    if (this->config.enable_cache_stats && this->config.cache_stats != nullptr) {
      this->config.cache_stats->configure_experts(this->config.layer_idx, this->config.expert_num);
    }
    printf("TP MOE layer %d, pool: 0x%lx, expert num: %d, num_experts_per_tok: %d\n", config.layer_idx,
           (intptr_t)config.pool, config.expert_num, config.num_experts_per_tok);
    if (config.pool == nullptr) {
      printf("TP MOE layer %d, no worker pool\n", config.layer_idx);
      throw std::runtime_error("no worker pool");
    }

    tp_count = config.pool->config.subpool_count;
    if (config.intermediate_size % tp_count != 0) {
      printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
      throw std::runtime_error(
          "For TP, intermediate_size must be a "
          "multiple of NUMA node count");
    }

    // Check if this is Llamafile backend using compile-time type checking
    constexpr bool is_llamafile = std::is_same<T, LLAMA_MOE_TP>::value;
#ifndef QK_K
#define QK_K 256
#endif

    if (is_llamafile) {
      // For Llamafile backend: use QK_K-aligned TP splitting
      if (config.intermediate_size % QK_K != 0) {
        printf("intermediate_size %d must be divisible by QK_K %d for Llamafile backend\n", config.intermediate_size,
               QK_K);
        throw std::runtime_error("intermediate_size must be divisible by QK_K (256) for Llamafile backend");
      }

      int num_blocks = config.intermediate_size / QK_K;
      int base_blocks = num_blocks / tp_count;
      int extra_blocks = num_blocks % tp_count;

      if (base_blocks == 0) {
        printf("intermediate_size %d is too small for tp_count %d (num_blocks=%d)\n", config.intermediate_size,
               tp_count, num_blocks);
        throw std::runtime_error("intermediate_size too small: cannot distribute blocks to all TP instances");
      }

      printf("Llamafile TP splitting: intermediate_size=%d, tp_count=%d, QK_K=%d\n", config.intermediate_size, tp_count,
             QK_K);
      printf("  num_blocks=%d, base_blocks=%d, extra_blocks=%d\n", num_blocks, base_blocks, extra_blocks);

      int current_offset = 0;
      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;

        // First extra_blocks TPs get one more block
        int num_blocks_for_this_tp = base_blocks + (i < extra_blocks ? 1 : 0);
        tp_config.intermediate_size = num_blocks_for_this_tp * QK_K;

        printf("  TP %d: intermediate_size=%d, offset=%d, blocks=%d\n", i, tp_config.intermediate_size, current_offset,
               num_blocks_for_this_tp);

        tp_configs.push_back(tp_config);
        current_offset += tp_config.intermediate_size;
      }
    } else {
      // For non-Llamafile backends: use simple equal division
      if (config.intermediate_size % tp_count != 0) {
        printf("intermediate_size %d, tp count %d\n", config.intermediate_size, tp_count);
        throw std::runtime_error(
            "For TP, intermediate_size must be a "
            "multiple of NUMA node count");
      }

      for (auto i = 0; i < tp_count; i++) {
        tps.push_back(nullptr);
        GeneralMOEConfig tp_config = config;
        tp_config.intermediate_size /= tp_count;
        tp_configs.push_back(tp_config);
      }
    }

    config.pool->dispense_backend()->do_numa_job(
        [this, config](int i) { tps[i] = std::move(std::unique_ptr<T>(new T(tp_configs[i], i))); });

    local_output_numa.resize(tp_count, nullptr);
    MemoryRequest mem_requests;
    for (auto i = 0; i < tp_count; i++) {
      mem_requests.append_pointer(
          &local_output_numa[i],
          (size_t)sizeof(typename T::output_t) * tp_configs[i].max_possible_qlen() * tp_configs[i].hidden_size);
    }
    mem_requests.append_pointer(
        (void**)&local_output,
        sizeof(typename T::output_t) * tp_configs[0].max_possible_qlen() * tp_configs[0].hidden_size);
    // printf("local output tp, %d,\n", tp_configs[0].max_possible_qlen());
    shared_mem_buffer.alloc(this, mem_requests);
  }

  void warm_up() {
    int qlen = config.max_possible_qlen();
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config.num_experts_per_tok);
    std::vector<float> weights(qlen * config.num_experts_per_tok);
    for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
      expert_ids[i] = i % config.expert_num;
      weights[i] = 0.01;
    }
    forward(&qlen, config.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data(), false);
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental = false) {
    int qlen_local = qlen;
    forward(&qlen_local, k, expert_ids, weights, input, output, incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    forward(qlen_ptr, k, expert_ids, weights, input, output, false);
  }

  void forward_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input, intptr_t output,
                       bool incremental) {
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  void forward_binding_with_scores(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input,
                                   intptr_t output, bool incremental, intptr_t router_scores, int score_rows,
                                   int score_cols, int score_transform) {
    observe_router_scores_binding(router_scores, score_rows, score_cols, score_transform);
    forward((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
            incremental);
  }

  void forward(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
               bool incremental) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Not Loaded");
    }
#ifdef FORWARD_TIME_REPORT
    auto start = std::chrono::high_resolution_clock::now();
#endif
    int qlen = *qlen_ptr;

    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, pool, qlen, k, expert_ids, input, weights](int numa_id) {
      tps[numa_id]->forward(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id]);
    });

    merge_results(qlen, output, incremental);
#ifdef FORWARD_TIME_REPORT
    auto end = std::chrono::high_resolution_clock::now();
    auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    int unique_experts = 0;
    {
      std::unordered_set<int64_t> expert_set;
      for (int i = 0; i < qlen * config.num_experts_per_tok; i++) {
        expert_set.insert(expert_ids[i]);
      }
      unique_experts = expert_set.size();
    }
    auto band_width =
        (1.0 * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) / (1.0 * forward_time / 1e6);
    auto GFLOPS =
        (1.0 * config.hidden_size * config.intermediate_size * qlen * 3 * config.num_experts_per_tok * 2 / 1e9) /
        (1.0 * forward_time / 1e6);
    if (qlen <= 10) {
      forward_time_sum_ns += forward_time;
      forward_count++;
    }
    auto average_bandwidth =
        (1.0 * forward_count * unique_experts * config.hidden_size * config.intermediate_size * 3 / 1e9) /
        (1.0 * forward_time_sum_ns / 1e6);
    printf(
        "forward time %ld, time stamp:%ld, band width %f GElement/s, ave bandwidth %f GElement/s (only "
        "decode), %f GFLOPS, me numa: %d\n",
        forward_time, end.time_since_epoch().count() / 1000 % 100000000, band_width, average_bandwidth, GFLOPS,
        numa_node_of_cpu(sched_getcpu()));
#endif
  }

  virtual void load_weights() = 0;

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

  virtual void merge_results(int qlen, void* output) = 0;

  virtual void merge_results(int qlen, void* output, bool incremental) {
    if (incremental == false) {
      merge_results(qlen, output);
    } else {
      throw std::runtime_error("Not Implemented");
    }
  };
};

template <MOE_TP_PART T>
class TP_MOE : public TP_MOE_Common<T> {
 public:
  using TP_MOE_Common<T>::TP_MOE_Common;
  void load_weights(const uint64_t* physical_to_logical_map) { throw std::runtime_error("Not Implemented"); }
  // void merge_results(int qlen, void *output, bool incremental) { throw std::runtime_error("Not Implemented"); }
};

#endif
