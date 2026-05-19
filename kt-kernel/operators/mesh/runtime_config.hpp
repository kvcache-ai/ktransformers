#ifndef CPUINFER_OPERATOR_MESH_RUNTIME_CONFIG_HPP
#define CPUINFER_OPERATOR_MESH_RUNTIME_CONFIG_HPP

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace ktransformers {
class AsyncExpertReader;
}

enum class ResidentCachePolicyKind : uint8_t {
  RoundRobin = 0,
  LRU = 1,
  TwoQ = 2,
  SLRU = 3,
  SIEVE = 4,
  S3FIFO = 5,
  WTinyLFU = 6,
};

// I/O backend for expert weight loading.
//   FULL    : ordinary KT path — Python loader reads every expert into memory
//             up front and hands raw pointers to C++ via `gate_projs` etc.
//             Picked up by `amx_tp_moe_runtime.inc` "TP Load from loader"
//             branch. Default for compatibility with callers unaware of
//             io_uring.
//   IOURING : MESH lazy path — Python supplies file slots and an
//             AsyncExpertReader, C++ reads experts on demand via io_uring.
// The historical mmap-based resident loader was removed; FULL is *not* mmap,
// it's the original full-preload path.
enum class IOBackend : uint8_t {
  FULL = 0,
  IOURING = 1,
};

// File slot for io_uring direct I/O
struct ExpertFileSlot {
  int fd = -1;        // File descriptor (opened with O_DIRECT)
  off_t offset = 0;   // Byte offset in file
  size_t size = 0;    // Number of bytes to read
};

struct ExpertCacheStats;

struct MeshMOEConfigExtension {
  int max_tier0_experts = 0;
  // Optional resident cache limit for request-path on-demand expert copies.
  // When 0, backends fall back to max_tier0_experts-compatible behavior.
  int max_resident_experts = 0;
  // Heuristic policy used by the request-path resident cache.
  // baseline maps to the historical round-robin resident eviction behavior.
  std::string resident_cache_policy = "baseline";

  // I/O backend selection. Defaults to FULL (ordinary KT preload via
  // gate_projs/up_projs/down_projs). MESH's lazy path flips this to IOURING
  // when Python calls set_iouring_file_slots_for_readers — see
  // python_bindings.hpp.
  IOBackend io_backend = IOBackend::FULL;
  bool iouring_direct_io = true;

  // File slots for io_uring direct I/O: [numa_node][expert_id]
  std::vector<std::vector<ExpertFileSlot>> gate_file_slots;
  std::vector<std::vector<ExpertFileSlot>> up_file_slots;
  std::vector<std::vector<ExpertFileSlot>> down_file_slots;
  std::vector<std::vector<ExpertFileSlot>> gate_scale_file_slots;
  std::vector<std::vector<ExpertFileSlot>> up_scale_file_slots;
  std::vector<std::vector<ExpertFileSlot>> down_scale_file_slots;
  std::vector<std::vector<ExpertFileSlot>> gate_mins_file_slots;
  std::vector<std::vector<ExpertFileSlot>> up_mins_file_slots;
  std::vector<std::vector<ExpertFileSlot>> down_mins_file_slots;

  // io_uring reader used by this TP shard. Older callers set only
  // async_reader; MESH can also provide one reader per TP in async_readers.
  ktransformers::AsyncExpertReader* async_reader = nullptr;
  std::vector<ktransformers::AsyncExpertReader*> async_readers;

  // Cache statistics (optional, disabled by default for performance)
  bool enable_cache_stats = false;
  std::shared_ptr<ExpertCacheStats> cache_stats_owner;
  ExpertCacheStats* cache_stats = nullptr;

  // MESH lookahead eviction. Heat is tracked per (layer, expert_id), updated
  // online from this layer's router scores, and used as a soft eviction bonus.
  bool mesh_lookahead_enabled = false;
  bool mesh_topk_fallback_enabled = true;
  float mesh_lookahead_weight = 1.0f;
  float mesh_heat_gamma = 0.7f;
  float mesh_heat_beta = 0.5f;
  float mesh_transition_alpha = 0.5f;
  int mesh_prefetch_budget = 0;
  bool mesh_coldstart_prefill_enabled = false;
  int mesh_coldstart_prefill_limit = 0;
  bool mesh_prefill_layer_mode_enabled = false;
  int mesh_prefill_static_experts = 0;
  int mesh_decode_resident_experts = 0;
  bool mesh_memory_guard_enabled = false;
  float mesh_memory_high_watermark = 0.95f;
  float mesh_memory_target_watermark = 0.90f;
  int mesh_memory_check_interval = 64;
  int mesh_memory_max_demotes_per_check = 8;

};

// Expert cache statistics for hit rate analysis
struct ExpertCacheStats {
  std::mutex expert_mu;
  int layer_idx = -1;
  int expert_num = 0;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_access_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_hit_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_miss_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_cold_miss_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_in_flight_miss_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_promote_count;
  std::unique_ptr<std::atomic<uint64_t>[]> expert_prefetch_hit_count;
  std::atomic<uint64_t> dump_tick{0};
  std::string dump_path;
  uint64_t dump_every = 1;

  std::atomic<uint64_t> promote_count{0};      // Number of promote operations
  std::atomic<uint64_t> demote_count{0};       // Number of demote operations
  std::atomic<uint64_t> hit_count{0};          // Cache hit (expert already in NUMA)
  std::atomic<uint64_t> miss_count{0};         // Cache miss (need to promote)
  std::atomic<uint64_t> cold_miss_count{0};    // Demand access found expert fully cold
  std::atomic<uint64_t> in_flight_miss_count{0}; // Demand access waited on pending async read
  std::atomic<uint64_t> eviction_count{0};     // Number of evictions
  std::atomic<uint64_t> total_access_count{0}; // Total access count
  std::atomic<uint64_t> lookahead_update_count{0}; // Online Heat updates
  std::atomic<uint64_t> prefetch_count{0};      // Lookahead prefetch promotions
  std::atomic<uint64_t> full_score_update_count{0}; // Full router-vector Heat updates
  std::atomic<uint64_t> async_prefetch_count{0}; // Non-blocking io_uring prefetch submissions
  std::atomic<uint64_t> prefetch_hit_count{0};   // Demand accesses satisfied by pending prefetch
  std::atomic<uint64_t> iouring_read_request_count{0}; // Submitted io_uring read requests
  std::atomic<uint64_t> iouring_read_bytes{0};          // Submitted io_uring read bytes
  std::atomic<uint64_t> transition_update_count{0};     // Cross-layer transition-prior updates
  std::atomic<uint64_t> coldstart_prefill_count{0};     // Cold-start fill prefetch submissions
  std::atomic<uint64_t> bootstrap_prefetch_candidate_count{0}; // First decode-token warm-fill candidates
  std::atomic<uint64_t> bootstrap_prefetch_submit_count{0};    // First decode-token warm-fill submissions
  std::atomic<uint64_t> bootstrap_prefetch_skip_gpu_count{0};  // Warm-fill candidates skipped by GPU mask
  std::atomic<uint64_t> bootstrap_prefetch_skip_resident_count{0}; // Warm-fill candidates already resident/pending
  std::atomic<uint64_t> memory_guard_demote_count{0};   // Memory-pressure demotions
  std::atomic<uint64_t> state_defer_token_count{0};      // Decode rows split by resident-state defer
  std::atomic<uint64_t> state_defer_cpu_topk_count{0};   // CPU-managed top-k entries seen by state defer
  std::atomic<uint64_t> state_defer_gpu_skip_count{0};   // GPU experts skipped by state defer
  std::atomic<uint64_t> state_defer_nonready_count{0};   // CPU top-k entries not resident at split time
  std::atomic<uint64_t> state_defer_deferred_count{0};   // Non-ready entries absorbed by defer window
  std::atomic<uint64_t> state_defer_overflow_immediate_count{0}; // Non-ready entries left immediate
  std::atomic<uint64_t> state_defer_overflow_token_count{0};     // Rows with any immediate overflow

  double hit_rate() const {
    uint64_t total = hit_count.load() + miss_count.load();
    return total > 0 ? static_cast<double>(hit_count.load()) / total : 0.0;
  }

  void configure_experts(int layer, int n) {
    if (n <= 0) return;
    std::lock_guard<std::mutex> guard(expert_mu);
    if (layer_idx == layer && expert_num == n && expert_access_count != nullptr) {
      return;
    }
    layer_idx = layer;
    expert_num = n;
    dump_path.clear();
    if (const char* path = std::getenv("KT_EXPERT_STATS_PATH")) {
      dump_path = path;
    }
    dump_every = 1;
    if (const char* raw_every = std::getenv("KT_EXPERT_STATS_DUMP_EVERY")) {
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(raw_every, &end, 10);
      if (end != raw_every && parsed > 0) {
        dump_every = static_cast<uint64_t>(parsed);
      }
    }
    expert_access_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_hit_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_miss_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_cold_miss_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_in_flight_miss_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_promote_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    expert_prefetch_hit_count = std::make_unique<std::atomic<uint64_t>[]>(n);
    for (int i = 0; i < n; ++i) {
      expert_access_count[i].store(0, std::memory_order_relaxed);
      expert_hit_count[i].store(0, std::memory_order_relaxed);
      expert_miss_count[i].store(0, std::memory_order_relaxed);
      expert_cold_miss_count[i].store(0, std::memory_order_relaxed);
      expert_in_flight_miss_count[i].store(0, std::memory_order_relaxed);
      expert_promote_count[i].store(0, std::memory_order_relaxed);
      expert_prefetch_hit_count[i].store(0, std::memory_order_relaxed);
    }
  }

  void note_expert_access(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_access_count == nullptr) return;
    expert_access_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_hit(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_hit_count == nullptr) return;
    expert_hit_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_miss(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_miss_count == nullptr) return;
    expert_miss_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_cold_miss(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_cold_miss_count == nullptr) return;
    expert_cold_miss_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_in_flight_miss(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_in_flight_miss_count == nullptr) return;
    expert_in_flight_miss_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_promote(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_promote_count == nullptr) return;
    expert_promote_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void note_expert_prefetch_hit(int expert_id) {
    if (expert_id < 0 || expert_id >= expert_num || expert_prefetch_hit_count == nullptr) return;
    expert_prefetch_hit_count[expert_id].fetch_add(1, std::memory_order_relaxed);
  }

  void reset() {
    promote_count.store(0);
    demote_count.store(0);
    hit_count.store(0);
    miss_count.store(0);
    cold_miss_count.store(0);
    in_flight_miss_count.store(0);
    eviction_count.store(0);
    total_access_count.store(0);
    lookahead_update_count.store(0);
    prefetch_count.store(0);
    full_score_update_count.store(0);
    async_prefetch_count.store(0);
    prefetch_hit_count.store(0);
    iouring_read_request_count.store(0);
    iouring_read_bytes.store(0);
    transition_update_count.store(0);
    coldstart_prefill_count.store(0);
    bootstrap_prefetch_candidate_count.store(0);
    bootstrap_prefetch_submit_count.store(0);
    bootstrap_prefetch_skip_gpu_count.store(0);
    bootstrap_prefetch_skip_resident_count.store(0);
    memory_guard_demote_count.store(0);
    state_defer_token_count.store(0);
    state_defer_cpu_topk_count.store(0);
    state_defer_gpu_skip_count.store(0);
    state_defer_nonready_count.store(0);
    state_defer_deferred_count.store(0);
    state_defer_overflow_immediate_count.store(0);
    state_defer_overflow_token_count.store(0);
    dump_tick.store(0);
    std::lock_guard<std::mutex> guard(expert_mu);
    for (int i = 0; i < expert_num; ++i) {
      if (expert_access_count != nullptr) expert_access_count[i].store(0, std::memory_order_relaxed);
      if (expert_hit_count != nullptr) expert_hit_count[i].store(0, std::memory_order_relaxed);
      if (expert_miss_count != nullptr) expert_miss_count[i].store(0, std::memory_order_relaxed);
      if (expert_cold_miss_count != nullptr) expert_cold_miss_count[i].store(0, std::memory_order_relaxed);
      if (expert_in_flight_miss_count != nullptr) expert_in_flight_miss_count[i].store(0, std::memory_order_relaxed);
      if (expert_promote_count != nullptr) expert_promote_count[i].store(0, std::memory_order_relaxed);
      if (expert_prefetch_hit_count != nullptr) expert_prefetch_hit_count[i].store(0, std::memory_order_relaxed);
    }
  }

  void maybe_dump_jsonl() {
    if (dump_path.empty() || expert_num <= 0 || expert_access_count == nullptr) return;
    const uint64_t tick = dump_tick.fetch_add(1, std::memory_order_relaxed) + 1;
    if (tick % dump_every != 0) return;

    static std::mutex dump_mu;
    std::lock_guard<std::mutex> dump_guard(dump_mu);
    std::ofstream out(dump_path, std::ios::app);
    if (!out.good()) return;

    auto append_counter_array = [&](const char* name, const std::unique_ptr<std::atomic<uint64_t>[]>& counters) {
      out << ",\"" << name << "\":[";
      for (int i = 0; i < expert_num; ++i) {
        if (i > 0) out << ",";
        out << (counters == nullptr ? 0 : counters[i].load(std::memory_order_relaxed));
      }
      out << "]";
    };

    out << "{\"layer\":" << layer_idx << ",\"expert_num\":" << expert_num << ",\"dump_tick\":" << tick
        << ",\"promote_count\":" << promote_count.load(std::memory_order_relaxed)
        << ",\"demote_count\":" << demote_count.load(std::memory_order_relaxed)
        << ",\"hit_count\":" << hit_count.load(std::memory_order_relaxed)
        << ",\"miss_count\":" << miss_count.load(std::memory_order_relaxed)
        << ",\"cold_miss_count\":" << cold_miss_count.load(std::memory_order_relaxed)
        << ",\"in_flight_miss_count\":" << in_flight_miss_count.load(std::memory_order_relaxed)
        << ",\"eviction_count\":" << eviction_count.load(std::memory_order_relaxed)
        << ",\"total_access_count\":" << total_access_count.load(std::memory_order_relaxed)
        << ",\"lookahead_update_count\":" << lookahead_update_count.load(std::memory_order_relaxed)
        << ",\"prefetch_count\":" << prefetch_count.load(std::memory_order_relaxed)
        << ",\"full_score_update_count\":" << full_score_update_count.load(std::memory_order_relaxed)
        << ",\"async_prefetch_count\":" << async_prefetch_count.load(std::memory_order_relaxed)
        << ",\"prefetch_hit_count\":" << prefetch_hit_count.load(std::memory_order_relaxed)
        << ",\"iouring_read_request_count\":" << iouring_read_request_count.load(std::memory_order_relaxed)
        << ",\"iouring_read_bytes\":" << iouring_read_bytes.load(std::memory_order_relaxed)
        << ",\"transition_update_count\":" << transition_update_count.load(std::memory_order_relaxed)
        << ",\"coldstart_prefill_count\":" << coldstart_prefill_count.load(std::memory_order_relaxed)
        << ",\"bootstrap_prefetch_candidate_count\":"
        << bootstrap_prefetch_candidate_count.load(std::memory_order_relaxed)
        << ",\"bootstrap_prefetch_submit_count\":"
        << bootstrap_prefetch_submit_count.load(std::memory_order_relaxed)
        << ",\"bootstrap_prefetch_skip_gpu_count\":"
        << bootstrap_prefetch_skip_gpu_count.load(std::memory_order_relaxed)
        << ",\"bootstrap_prefetch_skip_resident_count\":"
        << bootstrap_prefetch_skip_resident_count.load(std::memory_order_relaxed)
        << ",\"memory_guard_demote_count\":" << memory_guard_demote_count.load(std::memory_order_relaxed)
        << ",\"state_defer_token_count\":" << state_defer_token_count.load(std::memory_order_relaxed)
        << ",\"state_defer_cpu_topk_count\":" << state_defer_cpu_topk_count.load(std::memory_order_relaxed)
        << ",\"state_defer_gpu_skip_count\":" << state_defer_gpu_skip_count.load(std::memory_order_relaxed)
        << ",\"state_defer_nonready_count\":" << state_defer_nonready_count.load(std::memory_order_relaxed)
        << ",\"state_defer_deferred_count\":" << state_defer_deferred_count.load(std::memory_order_relaxed)
        << ",\"state_defer_overflow_immediate_count\":"
        << state_defer_overflow_immediate_count.load(std::memory_order_relaxed)
        << ",\"state_defer_overflow_token_count\":"
        << state_defer_overflow_token_count.load(std::memory_order_relaxed);
    append_counter_array("expert_access", expert_access_count);
    append_counter_array("expert_hit", expert_hit_count);
    append_counter_array("expert_miss", expert_miss_count);
    append_counter_array("expert_cold_miss", expert_cold_miss_count);
    append_counter_array("expert_in_flight_miss", expert_in_flight_miss_count);
    append_counter_array("expert_promote", expert_promote_count);
    append_counter_array("expert_prefetch_hit", expert_prefetch_hit_count);
    out << "}\n";
  }
};

namespace mesh {

template <typename Config>
inline void configure_cache_stats(Config& config) {
  if (config.enable_cache_stats && config.cache_stats == nullptr) {
    config.cache_stats_owner = std::make_shared<ExpertCacheStats>();
    config.cache_stats = config.cache_stats_owner.get();
  }
  if (config.enable_cache_stats && config.cache_stats != nullptr) {
    config.cache_stats->configure_experts(config.layer_idx, config.expert_num);
  }
}

template <typename Config>
inline void assign_tp_async_reader(Config& root_config, Config& tp_config, int tp_idx) {
  if (tp_config.io_backend != IOBackend::IOURING || tp_config.async_readers.empty()) return;
  const int reader_idx = std::min<int>(tp_idx, static_cast<int>(tp_config.async_readers.size()) - 1);
  tp_config.async_reader = tp_config.async_readers[reader_idx];
  if (tp_config.async_reader == nullptr) {
    throw std::runtime_error("io_uring TP reader is null");
  }
  if (root_config.async_reader == nullptr && tp_idx == 0) {
    root_config.async_reader = tp_config.async_reader;
  }
}

}  // namespace mesh

struct MeshLookaheadLayerState {
  int expert_num = 0;
  bool observed = false;
  std::vector<float> token_heat;
  std::vector<float> global_ema;
  std::vector<float> cross_layer_prior;
  std::vector<float> last_observed;

  void ensure_size(int n) {
    if (n <= 0) return;
    if (expert_num == n && static_cast<int>(token_heat.size()) == n &&
        static_cast<int>(global_ema.size()) == n &&
        static_cast<int>(cross_layer_prior.size()) == n &&
        static_cast<int>(last_observed.size()) == n) {
      return;
    }
    expert_num = n;
    observed = false;
    token_heat.assign(n, 0.0f);
    global_ema.assign(n, 0.0f);
    cross_layer_prior.assign(n, 0.0f);
    last_observed.assign(n, 0.0f);
  }
};

struct MeshTransitionEdgeState {
  int expert_num = 0;
  std::vector<float> matrix;

  void ensure_size(int n) {
    if (n <= 0) return;
    if (expert_num == n && static_cast<int>(matrix.size()) == n * n) {
      return;
    }
    expert_num = n;
    matrix.assign(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0f);
  }
};

struct MeshLookaheadRegistry {
  std::mutex mu;
  std::map<int, MeshLookaheadLayerState> layers;
  std::map<int, MeshTransitionEdgeState> transition_edges;

  MeshLookaheadLayerState& ensure_layer_locked(int layer_idx, int n) {
    MeshLookaheadLayerState& state = layers[layer_idx];
    state.ensure_size(n);
    return state;
  }

  void update_layer_locked(MeshLookaheadLayerState& state,
                           const std::vector<float>& observed,
                           float gamma,
                           float beta) {
    if (state.expert_num <= 0) return;
    const int n = state.expert_num;
    for (int expert_id = 0; expert_id < n; ++expert_id) {
      const float prior = std::max(state.global_ema[expert_id], state.cross_layer_prior[expert_id]);
      state.token_heat[expert_id] = gamma * observed[expert_id] + (1.0f - gamma) * prior;
      state.global_ema[expert_id] = beta * state.token_heat[expert_id] + (1.0f - beta) * state.global_ema[expert_id];
      state.cross_layer_prior[expert_id] *= (1.0f - gamma);
    }
    state.observed = true;
  }

  MeshTransitionEdgeState& ensure_edge_locked(int source_layer_idx, int n) {
    MeshTransitionEdgeState& edge = transition_edges[source_layer_idx];
    edge.ensure_size(n);
    return edge;
  }

  std::vector<std::pair<int, float>> top_distribution_locked(const std::vector<float>& values,
                                                             int n,
                                                             const uint8_t* gpu_experts_mask,
                                                             int max_items = 16) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(n);
    for (int expert_id = 0; expert_id < n; ++expert_id) {
      if (gpu_experts_mask != nullptr && gpu_experts_mask[expert_id]) continue;
      const float value = expert_id < static_cast<int>(values.size()) ? values[expert_id] : 0.0f;
      if (!std::isfinite(value) || value <= 0.0f) continue;
      ranked.emplace_back(value, expert_id);
    }
    if (ranked.empty()) return {};
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
      if (lhs.first != rhs.first) return lhs.first > rhs.first;
      return lhs.second < rhs.second;
    });
    if (static_cast<int>(ranked.size()) > max_items) {
      ranked.resize(max_items);
    }
    float sum = 0.0f;
    for (const auto& item : ranked) sum += item.first;
    if (sum <= 0.0f || !std::isfinite(sum)) return {};
    std::vector<std::pair<int, float>> dist;
    dist.reserve(ranked.size());
    for (const auto& item : ranked) {
      dist.emplace_back(item.second, item.first / sum);
    }
    return dist;
  }

  void update_transition_edge_locked(int source_layer_idx,
                                     int n,
                                     const std::vector<float>& source_observed,
                                     const std::vector<float>& target_observed,
                                     const uint8_t* gpu_experts_mask,
                                     float transition_alpha) {
    if (n <= 0 || source_observed.empty() || target_observed.empty() || transition_alpha <= 0.0f) return;
    transition_alpha = std::max(0.0f, std::min(1.0f, transition_alpha));
    const auto source_dist = top_distribution_locked(source_observed, n, gpu_experts_mask);
    const auto target_dist = top_distribution_locked(target_observed, n, gpu_experts_mask);
    if (source_dist.empty() || target_dist.empty()) return;

    MeshTransitionEdgeState& edge = ensure_edge_locked(source_layer_idx, n);
    for (const auto& source : source_dist) {
      const int source_id = source.first;
      float* row = edge.matrix.data() + static_cast<size_t>(source_id) * static_cast<size_t>(n);
      for (int target_id = 0; target_id < n; ++target_id) {
        row[target_id] *= (1.0f - transition_alpha);
      }
      for (const auto& target : target_dist) {
        row[target.first] += transition_alpha * target.second;
      }
    }
  }

  void propagate_transition_to_next_locked(int layer_idx,
                                           int n,
                                           const std::vector<float>& observed,
                                           const uint8_t* gpu_experts_mask,
                                           float transition_alpha) {
    if (n <= 0 || observed.empty() || transition_alpha <= 0.0f) return;
    transition_alpha = std::max(0.0f, std::min(1.0f, transition_alpha));
    const auto source_dist = top_distribution_locked(observed, n, gpu_experts_mask);
    if (source_dist.empty()) return;

    MeshTransitionEdgeState& edge = ensure_edge_locked(layer_idx, n);
    MeshLookaheadLayerState& next = ensure_layer_locked(layer_idx + 1, n);
    std::vector<float> predicted(n, 0.0f);
    for (const auto& source : source_dist) {
      const int source_id = source.first;
      const float source_weight = source.second;
      const float* row = edge.matrix.data() + static_cast<size_t>(source_id) * static_cast<size_t>(n);
      float row_sum = 0.0f;
      for (int target_id = 0; target_id < n; ++target_id) {
        row_sum += std::max(0.0f, row[target_id]);
      }
      if (row_sum > 0.0f && std::isfinite(row_sum)) {
        for (int target_id = 0; target_id < n; ++target_id) {
          if (gpu_experts_mask != nullptr && gpu_experts_mask[target_id]) continue;
          predicted[target_id] += source_weight * std::max(0.0f, row[target_id]) / row_sum;
        }
      } else {
        predicted[source_id] = std::max(predicted[source_id], source_weight);
      }
    }

    for (int target_id = 0; target_id < n; ++target_id) {
      if (gpu_experts_mask != nullptr && gpu_experts_mask[target_id]) continue;
      next.cross_layer_prior[target_id] =
          transition_alpha * predicted[target_id] + (1.0f - transition_alpha) * next.cross_layer_prior[target_id];
    }
  }

  std::vector<float> snapshot_for_layer(int layer_idx, int n) {
    std::lock_guard<std::mutex> guard(mu);
    MeshLookaheadLayerState& state = ensure_layer_locked(layer_idx, n);
    std::vector<float> heat = state.token_heat;
    for (int expert_id = 0; expert_id < n; ++expert_id) {
      heat[expert_id] = std::max(heat[expert_id], state.cross_layer_prior[expert_id]);
    }
    return heat;
  }

  void observe_topk(int layer_idx,
                    int n,
                    const int64_t* expert_ids,
                    const float* weights,
                    int count,
                    const uint8_t* gpu_experts_mask,
                    float gamma,
                    float beta,
                    float transition_alpha = 0.0f) {
    if (expert_ids == nullptr || count <= 0 || n <= 0) return;
    gamma = std::max(0.0f, std::min(1.0f, gamma));
    beta = std::max(0.0f, std::min(1.0f, beta));

    std::lock_guard<std::mutex> guard(mu);
    MeshLookaheadLayerState& state = ensure_layer_locked(layer_idx, n);

    std::vector<float> observed(n, 0.0f);
    for (int i = 0; i < count; ++i) {
      const int expert_id = static_cast<int>(expert_ids[i]);
      if (expert_id < 0 || expert_id >= n) continue;
      if (gpu_experts_mask != nullptr && gpu_experts_mask[expert_id]) continue;
      const float w = weights == nullptr ? 1.0f : std::max(0.0f, weights[i]);
      observed[expert_id] = std::max(observed[expert_id], w);
    }

    if (layer_idx > 0) {
      MeshLookaheadLayerState& prev = ensure_layer_locked(layer_idx - 1, n);
      update_transition_edge_locked(layer_idx - 1, n, prev.last_observed, observed, gpu_experts_mask, transition_alpha);
    }
    update_layer_locked(state, observed, gamma, beta);
    state.last_observed = observed;
    propagate_transition_to_next_locked(layer_idx, n, observed, gpu_experts_mask, transition_alpha);
  }

  void observe_scores(int layer_idx,
                      int n,
                      const float* scores,
                      int rows,
                      int cols,
                      const uint8_t* gpu_experts_mask,
                      float gamma,
                      float beta,
                      int score_transform,
                      float transition_alpha = 0.0f) {
    if (scores == nullptr || rows <= 0 || cols <= 0 || n <= 0) return;
    gamma = std::max(0.0f, std::min(1.0f, gamma));
    beta = std::max(0.0f, std::min(1.0f, beta));

    std::lock_guard<std::mutex> guard(mu);
    MeshLookaheadLayerState& state = ensure_layer_locked(layer_idx, n);

    std::vector<float> observed(n, 0.0f);
    const int usable_cols = std::min(n, cols);
    for (int row = 0; row < rows; ++row) {
      const float* row_scores = scores + static_cast<size_t>(row) * static_cast<size_t>(cols);
      float row_max = 0.0f;
      float row_sum = 0.0f;
      if (score_transform == 1) {
        row_max = -std::numeric_limits<float>::infinity();
        for (int expert_id = 0; expert_id < usable_cols; ++expert_id) {
          if (gpu_experts_mask != nullptr && gpu_experts_mask[expert_id]) continue;
          row_max = std::max(row_max, row_scores[expert_id]);
        }
        for (int expert_id = 0; expert_id < usable_cols; ++expert_id) {
          if (gpu_experts_mask != nullptr && gpu_experts_mask[expert_id]) continue;
          row_sum += std::exp(row_scores[expert_id] - row_max);
        }
        if (row_sum <= 0.0f || !std::isfinite(row_sum)) {
          row_sum = 1.0f;
        }
      }

      for (int expert_id = 0; expert_id < usable_cols; ++expert_id) {
        if (gpu_experts_mask != nullptr && gpu_experts_mask[expert_id]) continue;
        float value = row_scores[expert_id];
        if (score_transform == 1) {
          value = std::exp(value - row_max) / row_sum;
        } else if (score_transform == 2) {
          value = 1.0f / (1.0f + std::exp(-value));
        }
        if (!std::isfinite(value) || value < 0.0f) {
          value = 0.0f;
        }
        observed[expert_id] = std::max(observed[expert_id], value);
      }
    }

    if (layer_idx > 0) {
      MeshLookaheadLayerState& prev = ensure_layer_locked(layer_idx - 1, n);
      update_transition_edge_locked(layer_idx - 1, n, prev.last_observed, observed, gpu_experts_mask, transition_alpha);
    }
    update_layer_locked(state, observed, gamma, beta);
    state.last_observed = observed;
    propagate_transition_to_next_locked(layer_idx, n, observed, gpu_experts_mask, transition_alpha);
  }

  void reset() {
    std::lock_guard<std::mutex> guard(mu);
    layers.clear();
    transition_edges.clear();
  }
};

inline MeshLookaheadRegistry& mesh_lookahead_registry() {
  static MeshLookaheadRegistry registry;
  return registry;
}

inline std::string normalize_resident_cache_policy_name(std::string name) {
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (name.empty() || name == "baseline" || name == "default" || name == "current" || name == "current_ema" ||
      name == "ema" || name == "ema_hotset" || name == "legacy") {
    return "baseline";
  }
  if (name == "lru" || name == "2q" || name == "slru" || name == "sieve" || name == "s3fifo" || name == "w_tinylfu") {
    return name;
  }
  if (name == "twoq" || name == "two-q") {
    return "2q";
  }
  if (name == "s3-fifo") {
    return "s3fifo";
  }
  if (name == "w-tinylfu" || name == "wtinylfu") {
    return "w_tinylfu";
  }
  return "baseline";
}

inline ResidentCachePolicyKind parse_resident_cache_policy_kind(const std::string& raw_name) {
  const std::string name = normalize_resident_cache_policy_name(raw_name);
  if (name == "lru") return ResidentCachePolicyKind::LRU;
  if (name == "2q") return ResidentCachePolicyKind::TwoQ;
  if (name == "slru") return ResidentCachePolicyKind::SLRU;
  if (name == "sieve") return ResidentCachePolicyKind::SIEVE;
  if (name == "s3fifo") return ResidentCachePolicyKind::S3FIFO;
  if (name == "w_tinylfu") return ResidentCachePolicyKind::WTinyLFU;
  return ResidentCachePolicyKind::RoundRobin;
}

struct ResidentCachePolicyState {
  ResidentCachePolicyKind kind = ResidentCachePolicyKind::RoundRobin;
  std::vector<uint64_t> last_access_seq;
  std::vector<uint64_t> insert_seq;
  std::vector<uint32_t> access_freq;
  std::vector<uint8_t> segment;
  std::vector<uint8_t> ref_bit;
  std::vector<uint8_t> ghost_hint;
  uint64_t logical_clock = 1;
  uint64_t hand = 0;
  mutable std::mutex mu;

  void reset(int num_experts, const std::string& policy_name) {
    std::lock_guard<std::mutex> guard(mu);
    kind = parse_resident_cache_policy_kind(policy_name);
    last_access_seq.assign(num_experts, 0);
    insert_seq.assign(num_experts, 0);
    access_freq.assign(num_experts, 0);
    segment.assign(num_experts, 0);
    ref_bit.assign(num_experts, 0);
    ghost_hint.assign(num_experts, 0);
    logical_clock = 1;
    hand = 0;
  }

  void note_access(int expert_id, bool resident) {
    if (expert_id < 0 || expert_id >= static_cast<int>(last_access_seq.size())) return;
    std::lock_guard<std::mutex> guard(mu);
    const uint64_t tick = logical_clock++;
    last_access_seq[expert_id] = tick;
    if (access_freq[expert_id] < std::numeric_limits<uint32_t>::max()) {
      access_freq[expert_id] += 1;
    }
    ref_bit[expert_id] = 1;

    if (!resident) {
      return;
    }

    switch (kind) {
      case ResidentCachePolicyKind::TwoQ:
      case ResidentCachePolicyKind::SLRU:
      case ResidentCachePolicyKind::S3FIFO:
      case ResidentCachePolicyKind::WTinyLFU:
        if (segment[expert_id] == 0 && access_freq[expert_id] >= 2) {
          segment[expert_id] = 1;
        }
        break;
      default:
        break;
    }
  }

  void on_insert(int expert_id, bool pinned) {
    if (expert_id < 0 || expert_id >= static_cast<int>(insert_seq.size())) return;
    std::lock_guard<std::mutex> guard(mu);
    const uint64_t tick = logical_clock++;
    insert_seq[expert_id] = tick;
    last_access_seq[expert_id] = tick;
    if (access_freq[expert_id] == 0) {
      access_freq[expert_id] = 1;
    }

    switch (kind) {
      case ResidentCachePolicyKind::TwoQ:
        segment[expert_id] = pinned ? 1 : 0;
        ghost_hint[expert_id] = 0;
        break;
      case ResidentCachePolicyKind::SLRU:
        segment[expert_id] = pinned ? 1 : 0;
        break;
      case ResidentCachePolicyKind::S3FIFO:
        segment[expert_id] = (pinned || ghost_hint[expert_id]) ? 1 : 0;
        ghost_hint[expert_id] = 0;
        break;
      case ResidentCachePolicyKind::WTinyLFU:
        segment[expert_id] = (pinned || access_freq[expert_id] >= 2) ? 1 : 0;
        ghost_hint[expert_id] = 0;
        break;
      default:
        segment[expert_id] = pinned ? 1 : 0;
        break;
    }
    ref_bit[expert_id] = 0;
  }

  void on_pin(int expert_id) {
    if (expert_id < 0 || expert_id >= static_cast<int>(segment.size())) return;
    std::lock_guard<std::mutex> guard(mu);
    segment[expert_id] = 1;
    ref_bit[expert_id] = 1;
    last_access_seq[expert_id] = logical_clock++;
  }

  void on_demote(int expert_id) {
    if (expert_id < 0 || expert_id >= static_cast<int>(segment.size())) return;
    std::lock_guard<std::mutex> guard(mu);
    if (kind == ResidentCachePolicyKind::S3FIFO || kind == ResidentCachePolicyKind::WTinyLFU) {
      ghost_hint[expert_id] = 1;
    }
    segment[expert_id] = 0;
    ref_bit[expert_id] = 0;
    insert_seq[expert_id] = 0;
    last_access_seq[expert_id] = 0;
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_victim(int expert_num,
                  int exclude_expert_id,
                  uint8_t cached_state,
                  StateGetter state_getter,
                  ReaderGetter reader_getter) {
    std::lock_guard<std::mutex> guard(mu);
    switch (kind) {
      case ResidentCachePolicyKind::LRU:
        return pick_oldest_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter, false, false);
      case ResidentCachePolicyKind::TwoQ: {
        const int a1 = pick_oldest_by_insert_locked(expert_num, exclude_expert_id, cached_state, state_getter,
                                                    reader_getter, true, false);
        return a1 >= 0 ? a1
                       : pick_oldest_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter,
                                            true, true);
      }
      case ResidentCachePolicyKind::SLRU: {
        const int probationary = pick_oldest_locked(expert_num, exclude_expert_id, cached_state, state_getter,
                                                    reader_getter, true, false);
        return probationary >= 0 ? probationary
                                 : pick_oldest_locked(expert_num, exclude_expert_id, cached_state, state_getter,
                                                      reader_getter, true, true);
      }
      case ResidentCachePolicyKind::SIEVE:
        return pick_sieve_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter);
      case ResidentCachePolicyKind::S3FIFO: {
        const int small = pick_oldest_by_insert_locked(expert_num, exclude_expert_id, cached_state, state_getter,
                                                       reader_getter, true, false);
        return small >= 0 ? small
                          : pick_oldest_by_insert_locked(expert_num, exclude_expert_id, cached_state, state_getter,
                                                         reader_getter, true, true);
      }
      case ResidentCachePolicyKind::WTinyLFU:
        return pick_lowest_freq_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter);
      case ResidentCachePolicyKind::RoundRobin:
      default:
        return pick_round_robin_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter);
    }
  }

  template <typename StateGetter>
  std::vector<int> build_reclaim_order(int expert_num,
                                       int preferred_expert_id,
                                       uint8_t cached_state,
                                       uint8_t pinned_state,
                                       StateGetter state_getter) {
    std::lock_guard<std::mutex> guard(mu);
    std::vector<int> order;
    order.reserve(expert_num);

    auto is_resident = [&](int expert_id) {
      if (expert_id < 0 || expert_id >= static_cast<int>(last_access_seq.size())) return false;
      if (expert_id == preferred_expert_id) return false;
      const uint8_t state = state_getter(expert_id);
      return state == cached_state || state == pinned_state;
    };
    auto append_if = [&](int expert_id) {
      if (!is_resident(expert_id)) return;
      if (std::find(order.begin(), order.end(), expert_id) != order.end()) return;
      order.push_back(expert_id);
    };

    switch (kind) {
      case ResidentCachePolicyKind::LRU:
        append_sorted_oldest_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, false, false, &order);
        break;
      case ResidentCachePolicyKind::TwoQ:
        append_sorted_by_insert_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, false, &order);
        append_sorted_oldest_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, true, &order);
        break;
      case ResidentCachePolicyKind::SLRU:
        append_sorted_oldest_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, false, &order);
        append_sorted_oldest_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, true, &order);
        break;
      case ResidentCachePolicyKind::SIEVE:
        append_sieve_reclaim_order_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, &order);
        break;
      case ResidentCachePolicyKind::S3FIFO:
        append_sorted_by_insert_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, false, &order);
        append_sorted_by_insert_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, true, true, &order);
        break;
      case ResidentCachePolicyKind::WTinyLFU:
        append_lowest_freq_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, &order);
        break;
      case ResidentCachePolicyKind::RoundRobin:
      default:
        append_round_robin_locked(expert_num, preferred_expert_id, cached_state, pinned_state, state_getter, &order);
        break;
    }

    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      append_if(expert_id);
    }
    return order;
  }

 private:
  template <typename StateGetter>
  bool is_reclaim_candidate_locked(int expert_id,
                                   int preferred_expert_id,
                                   uint8_t cached_state,
                                   uint8_t pinned_state,
                                   StateGetter state_getter) const {
    if (expert_id < 0 || expert_id >= static_cast<int>(last_access_seq.size())) return false;
    if (expert_id == preferred_expert_id) return false;
    const uint8_t state = state_getter(expert_id);
    return state == cached_state || state == pinned_state;
  }

  template <typename StateGetter, typename ReaderGetter>
  bool is_candidate_locked(int expert_id,
                           int exclude_expert_id,
                           uint8_t cached_state,
                           StateGetter state_getter,
                           ReaderGetter reader_getter) const {
    if (expert_id < 0 || expert_id >= static_cast<int>(last_access_seq.size())) return false;
    if (expert_id == exclude_expert_id) return false;
    if (state_getter(expert_id) != cached_state) return false;
    if (reader_getter(expert_id) != 0) return false;
    return true;
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_round_robin_locked(int expert_num,
                              int exclude_expert_id,
                              uint8_t cached_state,
                              StateGetter state_getter,
                              ReaderGetter reader_getter) {
    for (int attempt = 0; attempt < expert_num; ++attempt) {
      const int victim = static_cast<int>((hand + attempt) % static_cast<uint64_t>(expert_num));
      if (!is_candidate_locked(victim, exclude_expert_id, cached_state, state_getter, reader_getter)) {
        continue;
      }
      hand = static_cast<uint64_t>(victim + 1);
      return victim;
    }
    return -1;
  }

  template <typename StateGetter>
  void append_round_robin_locked(int expert_num,
                                 int preferred_expert_id,
                                 uint8_t cached_state,
                                 uint8_t pinned_state,
                                 StateGetter state_getter,
                                 std::vector<int>* out) {
    if (out == nullptr) return;
    for (int attempt = 0; attempt < expert_num; ++attempt) {
      const int candidate = static_cast<int>((hand + attempt) % static_cast<uint64_t>(expert_num));
      if (!is_reclaim_candidate_locked(candidate, preferred_expert_id, cached_state, pinned_state, state_getter)) {
        continue;
      }
      out->push_back(candidate);
    }
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_oldest_locked(int expert_num,
                         int exclude_expert_id,
                         uint8_t cached_state,
                         StateGetter state_getter,
                         ReaderGetter reader_getter,
                         bool use_segment_filter,
                         bool required_segment) {
    uint64_t best_tick = std::numeric_limits<uint64_t>::max();
    int best = -1;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_candidate_locked(expert_id, exclude_expert_id, cached_state, state_getter, reader_getter)) continue;
      if (use_segment_filter && static_cast<bool>(segment[expert_id]) != required_segment) continue;
      const uint64_t tick = last_access_seq[expert_id] == 0 ? insert_seq[expert_id] : last_access_seq[expert_id];
      if (tick < best_tick) {
        best_tick = tick;
        best = expert_id;
      }
    }
    return best;
  }

  template <typename StateGetter>
  void append_sorted_oldest_locked(int expert_num,
                                   int preferred_expert_id,
                                   uint8_t cached_state,
                                   uint8_t pinned_state,
                                   StateGetter state_getter,
                                   bool use_segment_filter,
                                   bool required_segment,
                                   std::vector<int>* out) {
    if (out == nullptr) return;
    std::vector<std::pair<uint64_t, int>> ranked;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_reclaim_candidate_locked(expert_id, preferred_expert_id, cached_state, pinned_state, state_getter)) continue;
      if (use_segment_filter && static_cast<bool>(segment[expert_id]) != required_segment) continue;
      const uint64_t tick = last_access_seq[expert_id] == 0 ? insert_seq[expert_id] : last_access_seq[expert_id];
      ranked.emplace_back(tick, expert_id);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    for (const auto& item : ranked) {
      out->push_back(item.second);
    }
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_oldest_by_insert_locked(int expert_num,
                                   int exclude_expert_id,
                                   uint8_t cached_state,
                                   StateGetter state_getter,
                                   ReaderGetter reader_getter,
                                   bool use_segment_filter,
                                   bool required_segment) {
    uint64_t best_tick = std::numeric_limits<uint64_t>::max();
    int best = -1;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_candidate_locked(expert_id, exclude_expert_id, cached_state, state_getter, reader_getter)) continue;
      if (use_segment_filter && static_cast<bool>(segment[expert_id]) != required_segment) continue;
      const uint64_t tick = insert_seq[expert_id];
      if (tick < best_tick) {
        best_tick = tick;
        best = expert_id;
      }
    }
    return best;
  }

  template <typename StateGetter>
  void append_sorted_by_insert_locked(int expert_num,
                                      int preferred_expert_id,
                                      uint8_t cached_state,
                                      uint8_t pinned_state,
                                      StateGetter state_getter,
                                      bool use_segment_filter,
                                      bool required_segment,
                                      std::vector<int>* out) {
    if (out == nullptr) return;
    std::vector<std::pair<uint64_t, int>> ranked;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_reclaim_candidate_locked(expert_id, preferred_expert_id, cached_state, pinned_state, state_getter)) continue;
      if (use_segment_filter && static_cast<bool>(segment[expert_id]) != required_segment) continue;
      ranked.emplace_back(insert_seq[expert_id], expert_id);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
    for (const auto& item : ranked) {
      out->push_back(item.second);
    }
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_sieve_locked(int expert_num,
                        int exclude_expert_id,
                        uint8_t cached_state,
                        StateGetter state_getter,
                        ReaderGetter reader_getter) {
    for (int attempt = 0; attempt < expert_num * 2; ++attempt) {
      const int victim = static_cast<int>(hand % static_cast<uint64_t>(expert_num));
      hand = static_cast<uint64_t>(victim + 1);
      if (!is_candidate_locked(victim, exclude_expert_id, cached_state, state_getter, reader_getter)) {
        continue;
      }
      if (ref_bit[victim]) {
        ref_bit[victim] = 0;
        continue;
      }
      return victim;
    }
    return pick_round_robin_locked(expert_num, exclude_expert_id, cached_state, state_getter, reader_getter);
  }

  template <typename StateGetter>
  void append_sieve_reclaim_order_locked(int expert_num,
                                         int preferred_expert_id,
                                         uint8_t cached_state,
                                         uint8_t pinned_state,
                                         StateGetter state_getter,
                                         std::vector<int>* out) {
    if (out == nullptr) return;
    for (int pass = 0; pass < 2; ++pass) {
      for (int attempt = 0; attempt < expert_num; ++attempt) {
        const int candidate = static_cast<int>((hand + attempt) % static_cast<uint64_t>(expert_num));
        if (!is_reclaim_candidate_locked(candidate, preferred_expert_id, cached_state, pinned_state, state_getter)) {
          continue;
        }
        const bool referenced = ref_bit[candidate] != 0;
        if ((pass == 0 && referenced) || (pass == 1 && !referenced)) {
          continue;
        }
        if (std::find(out->begin(), out->end(), candidate) == out->end()) {
          out->push_back(candidate);
        }
      }
    }
  }

  template <typename StateGetter, typename ReaderGetter>
  int pick_lowest_freq_locked(int expert_num,
                              int exclude_expert_id,
                              uint8_t cached_state,
                              StateGetter state_getter,
                              ReaderGetter reader_getter) {
    uint32_t best_freq = std::numeric_limits<uint32_t>::max();
    uint64_t best_tick = std::numeric_limits<uint64_t>::max();
    int best = -1;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_candidate_locked(expert_id, exclude_expert_id, cached_state, state_getter, reader_getter)) continue;
      const uint32_t freq = access_freq[expert_id];
      const uint64_t tick = last_access_seq[expert_id] == 0 ? insert_seq[expert_id] : last_access_seq[expert_id];
      const bool better = best < 0 || (freq < best_freq) ||
                          (freq == best_freq && segment[expert_id] < segment[best]) ||
                          (freq == best_freq && segment[expert_id] == segment[best] && tick < best_tick);
      if (better) {
        best = expert_id;
        best_freq = freq;
        best_tick = tick;
      }
    }
    return best;
  }

  template <typename StateGetter>
  void append_lowest_freq_locked(int expert_num,
                                 int preferred_expert_id,
                                 uint8_t cached_state,
                                 uint8_t pinned_state,
                                 StateGetter state_getter,
                                 std::vector<int>* out) {
    if (out == nullptr) return;
    struct RankedExpert {
      uint32_t freq;
      uint8_t seg;
      uint64_t tick;
      int expert_id;
    };
    std::vector<RankedExpert> ranked;
    for (int expert_id = 0; expert_id < expert_num; ++expert_id) {
      if (!is_reclaim_candidate_locked(expert_id, preferred_expert_id, cached_state, pinned_state, state_getter)) continue;
      ranked.push_back(RankedExpert{
          access_freq[expert_id],
          segment[expert_id],
          last_access_seq[expert_id] == 0 ? insert_seq[expert_id] : last_access_seq[expert_id],
          expert_id,
      });
    }
    std::sort(ranked.begin(), ranked.end(), [](const RankedExpert& lhs, const RankedExpert& rhs) {
      if (lhs.freq != rhs.freq) return lhs.freq < rhs.freq;
      if (lhs.seg != rhs.seg) return lhs.seg < rhs.seg;
      if (lhs.tick != rhs.tick) return lhs.tick < rhs.tick;
      return lhs.expert_id < rhs.expert_id;
    });
    for (const auto& item : ranked) {
      out->push_back(item.expert_id);
    }
  }
};

#endif  // CPUINFER_OPERATOR_MESH_RUNTIME_CONFIG_HPP
