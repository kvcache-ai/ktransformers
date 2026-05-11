#ifndef CPUINFER_OPERATOR_COMMON_HPP
#define CPUINFER_OPERATOR_COMMON_HPP

#include <map>

#include "../cpu_backend/worker_pool.h"
#include "ggml.h"

#if defined(__aarch64__) && defined(CPU_USE_KML)
#include <arm_sve.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Forward declarations
namespace ktransformers {
class AsyncExpertReader;
}

// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#define ASSERT_RELEASE(x, text)                                                            \
  do {                                                                                     \
    if (!(x)) {                                                                            \
      fprintf(stderr, "Assertion failed: %s, file %s, line %d\n", #x, __FILE__, __LINE__); \
      fprintf(stderr, "Error message: %s\n", (text));                                      \
      throw std::runtime_error((text));                                                    \
    }                                                                                      \
  } while (0)

#define PUSH_MEM_REQ(ptr, size) mem_requests.append_pointer(&(ptr), (size))

#define PROFILE_RECORD_TIME_STAMP(name)                                                             \
  do {                                                                                              \
    auto end_time = std::chrono::high_resolution_clock::now();                                      \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - last).count(); \
    time_map[(name)] = duration;                                                                    \
    last = end_time;                                                                                \
  } while (0)

#define DO_TPS_LOAD_WEIGHTS(pool)                                                         \
  (pool)->dispense_backend()->do_numa_job([this, pool, config](int numa_id) {             \
    this->tps[numa_id]->config_.physical_to_logical_map = config.physical_to_logical_map; \
    this->tps[numa_id]->load_weights();                                                   \
  })

#define expert_map(m, x) (m != nullptr ? m[(x)] : (x))

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline T div_up(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
T* offset_pointer(T* ptr, size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
size_t pointer_offset(T* ptr, T* b) {
  return reinterpret_cast<size_t>(b) - reinterpret_cast<size_t>(ptr);
}

template <typename T>
const T* offset_pointer(const T* ptr, size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

template <typename T>
T* offset_pointer_row_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, row * ld) + col;
}

template <typename T>
T* offset_pointer_col_major(T* t, int row, int col, size_t ld) {
  return offset_pointer(t, col * ld) + row;
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

// I/O backend for expert weight loading
enum class IOBackend : uint8_t {
  MMAP = 0,      // mmap-based loading (current default, uses OS page cache)
  IOURING = 1,   // io_uring-based async I/O (Linux 5.1+, zero page cache dependency)
};

// File slot for io_uring direct I/O
struct ExpertFileSlot {
  int fd = -1;        // File descriptor (opened with O_DIRECT)
  off_t offset = 0;   // Byte offset in file
  size_t size = 0;    // Number of bytes to read
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
    const char* path = std::getenv("KT_EXPERT_STATS_PATH");
    if (path == nullptr || path[0] == '\0' || expert_num <= 0 || expert_access_count == nullptr) return;

    uint64_t every = 1;
    if (const char* raw_every = std::getenv("KT_EXPERT_STATS_DUMP_EVERY")) {
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(raw_every, &end, 10);
      if (end != raw_every && parsed > 0) {
        every = static_cast<uint64_t>(parsed);
      }
    }
    const uint64_t tick = dump_tick.fetch_add(1, std::memory_order_relaxed) + 1;
    if (tick % every != 0) return;

    static std::mutex dump_mu;
    std::lock_guard<std::mutex> dump_guard(dump_mu);
    std::ofstream out(path, std::ios::app);
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
        << ",\"memory_guard_demote_count\":" << memory_guard_demote_count.load(std::memory_order_relaxed);
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

class TimePerf {
 protected:
  std::string time_perf_name;
  std::map<std::string, long> time_map;
  std::chrono::time_point<std::chrono::high_resolution_clock> last;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

  void forward_perf_start() {
    start_time = std::chrono::high_resolution_clock::now();
    last = start_time;
  }

  void perf_report() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::string output = time_perf_name + ", forward time: " + std::to_string(duration.count()) + " us";
    // for (auto [name, t] : time_map) {
    //   double p = 100.0 * t / duration.count();
    //   // if (p < 1.0) {
    //   //   continue; // Skip if the percentage is less than 1%
    //   // }
    //   output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    // }
    // 反向遍历
    for (auto it = time_map.rbegin(); it != time_map.rend(); ++it) {
      const std::string& name = it->first;
      long t = it->second;
      double p = 100.0 * t / duration.count();
      // if (p < 1.0) {
      //   continue; // Skip if the percentage is less than 1%
      // }
      output += ", " + name + ": " + std::to_string(t) + " us(" + std::to_string(size_t(round(p))) + "%)";
    }
    printf("%s\n", output.c_str());
  }
};

struct TaskCounter {
  std::vector<size_t> fold = {}, card = {};

  TaskCounter(std::initializer_list<size_t> i) {
    card.push_back(1);
    for (auto j : i) {
      push_back(j);
    }
  }

  void push_back(size_t i) {
    fold.push_back(i);
    for (auto& c : card) {
      c *= i;
    }
    card.push_back(1);
  }
  void push_back(std::vector<size_t> i) {
    for (auto j : i) {
      push_back(j);
    }
  }
  size_t count() { return card[0]; }
  size_t at(size_t id, size_t which) { return id % card.at(which) / card.at(which + 1); }
};

struct GeneralConfig {
  size_t vocab_size;
  size_t hidden_size;

  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_shared_experts;
  size_t max_qlen = 4096;

  void* lm_heads_ptr;
  ggml_type lm_heads_type;
  void* norm_weights_ptr;
  ggml_type norm_weights_type;
  void* token_embd_ptr;
  ggml_type token_embd_type;
  WorkerPool* pool = nullptr;
  GeneralConfig() {}
};

struct GeneralMLAConfig {
  size_t hidden_size;
  size_t q_lora_rank;
  size_t num_heads;
  size_t nope_size;
  size_t rope_size;
  size_t kv_lora_rank;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;
  size_t token_count_in_page = 256;  // token count in a page
  size_t max_qlen = 1024;
  size_t max_kvlen = 4096;

  // rope
  size_t max_position_embeddings;
  double rope_scaling_factor = 1.0;
  double rope_theta = 10000.0;
  double rope_scaling_beta_fast;
  double rope_scaling_beta_slow;
  double rope_scaling_mscale;
  double rope_scaling_mscale_all_dim;
  double rope_scaling_original_max_position_embeddings;

  void* q_a_proj;
  void* q_a_norm = nullptr;
  void* q_b_proj;
  void* kv_a_proj_with_mqa;
  void* kv_a_norm = nullptr;
  void* kv_b_proj;
  void* o_proj;

  // for llamafile
  ggml_type q_a_proj_type;
  ggml_type q_a_norm_type;
  ggml_type q_b_proj_type;
  ggml_type kv_a_proj_with_mqa_type;
  ggml_type kv_a_norm_type;
  ggml_type kv_b_proj_type;
  ggml_type w_o_type;

  ggml_type input_type = GGML_TYPE_F32;
  ggml_type output_type = GGML_TYPE_F32;

  size_t m_block = 4;
  size_t n_block = 4;
  // for kvcache
  size_t page_count = 200;  // page count for kv cache

  GeneralMLAConfig() {}
  GeneralMLAConfig(size_t hidden_size, size_t q_lora_rank, size_t kv_lora_rank, size_t num_heads, size_t nope_size,
                   size_t rope_size)
      : hidden_size(hidden_size),
        q_lora_rank(q_lora_rank),
        kv_lora_rank(kv_lora_rank),
        num_heads(num_heads),
        nope_size(nope_size),
        rope_size(rope_size) {}
};

struct QuantConfig {
  std::string quant_method = "";
  int bits = 0;
  int group_size = 0;
  bool zero_point = false;
  bool per_channel = false;  // Per-channel quantization (GLM-4.7-FP8 style)
};

struct GeneralMOEConfig {
  // Basic Config
  int expert_num;
  int num_experts_per_tok;
  int hidden_size;
  int intermediate_size;

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  // SGLang offload
  int num_gpu_experts = 0;              // Computed from gpu_experts_mask
  uint8_t* gpu_experts_mask = nullptr;  // Bool mask: true = expert on GPU
  void* physical_to_logical_map = nullptr;

  // Compute num_gpu_experts from gpu_experts_mask
  void compute_num_gpu_experts() {
    num_gpu_experts = 0;
    if (gpu_experts_mask) {
      for (int i = 0; i < expert_num; i++) {
        if (gpu_experts_mask[i]) num_gpu_experts++;
      }
    }
  }

  // Check if expert should be skipped (invalid, out of range, or on GPU)
  inline bool should_skip_expert(int64_t expert_id) const {
    return expert_id < 0 || expert_id >= expert_num || (gpu_experts_mask && gpu_experts_mask[expert_id]);
  }

  void* gate_proj = nullptr;
  void* up_proj = nullptr;
  void* down_proj = nullptr;

  void* gate_scale = nullptr;
  void* up_scale = nullptr;
  void* down_scale = nullptr;

  void* gate_zero = nullptr;
  void* up_zero = nullptr;
  void* down_zero = nullptr;

  QuantConfig quant_config;

  // for amx
  int max_len = 0;
  std::vector<std::vector<void*>> gate_projs;
  std::vector<std::vector<void*>> up_projs;
  std::vector<std::vector<void*>> down_projs;
  std::vector<std::vector<void*>> gate_scales;
  std::vector<std::vector<void*>> up_scales;
  std::vector<std::vector<void*>> down_scales;
  std::vector<std::vector<void*>> gate_zeros;
  std::vector<std::vector<void*>> up_zeros;
  std::vector<std::vector<void*>> down_zeros;

  // Pre-quantized backward weights (transposed, in BufferB format) [tp_count][expert_id]
  std::vector<std::vector<void*>> gate_bwd_projs;
  std::vector<std::vector<void*>> up_bwd_projs;
  std::vector<std::vector<void*>> down_bwd_projs;
  std::vector<std::vector<void*>> gate_bwd_scales;
  std::vector<std::vector<void*>> up_bwd_scales;
  std::vector<std::vector<void*>> down_bwd_scales;

  std::string path;
  bool save = false;
  bool load = false;
  bool share_backward_bb = false;
  bool share_cache_pool = false;

  // mmap mode: when true, weight pointers point directly into mmap'd regions.
  // load_weights() should skip memcpy and use the pointers as-is.
  // This avoids double-buffering when model size approaches physical RAM.
  bool use_mmap = false;
  int max_tier0_experts = 0;
  // Optional resident cache limit for request-path on-demand expert copies.
  // When 0, backends fall back to max_tier0_experts-compatible behavior.
  int max_resident_experts = 0;
  // Heuristic policy used by the request-path resident cache.
  // baseline maps to the historical round-robin resident eviction behavior.
  std::string resident_cache_policy = "baseline";

  // I/O backend selection (mmap vs io_uring)
  IOBackend io_backend = IOBackend::MMAP;
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

  // Shared io_uring reader instance (one per process)
  ktransformers::AsyncExpertReader* async_reader = nullptr;

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
  bool mesh_memory_guard_enabled = false;
  float mesh_memory_high_watermark = 0.95f;
  float mesh_memory_target_watermark = 0.90f;
  int mesh_memory_check_interval = 64;
  int mesh_memory_max_demotes_per_check = 8;

  // mmap baseline page reclaim parameters
  float mmap_file_reclaim_trigger_ratio = 0.90f;  // Trigger reclaim at 90% memory usage
  float mmap_file_target_file_ratio = 0.15f;      // Target 15% file-backed pages after reclaim

  // for llamafile
  int m_block = 4;
  int group_min_len = 0;
  int group_max_len = 0;
  int gate_type;
  int up_type;
  int down_type;
  int hidden_type;

  int max_cache_depth = 1;

  // SwiGLU asymmetric clamp applied to gate/up before silu*up. 0.0f =
  // disabled (default for all non-MXFP4 paths). Set to e.g. 10.0f for
  // DeepSeek V4-Flash 2604B routed experts, matching the trtllm
  // `gemm1_clamp_limit` and the sglang deep_gemm path's
  // `_apply_swiglu_limit`:
  //   gate = clamp(gate, max=limit)            // one-sided (silu input)
  //   up   = clamp(up, min=-limit, max=limit)  // symmetric
  // Read by `act_fn` in la/amx.hpp; non-zero only for MXFP4 today.
  // Origin: kt-sglang 耦合 (carries the V4-2604B limit set by sglang side).
  float swiglu_limit = 0.0f;

  GeneralMOEConfig() {}

  GeneralMOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : expert_num(expert_num),
        num_experts_per_tok(routed_expert_num),
        hidden_size(hidden_size),
        intermediate_size(intermediate_size) {}

  int max_possible_qlen() { return std::max(max_len, group_max_len); }
};

// SFT (Supervised Fine-Tuning) configuration for MoE with LoRA
struct MOESFTConfig : public GeneralMOEConfig {
  // LoRA configuration
  int lora_rank = 16;
  float lora_alpha = 32.0f;
  float lora_scaling() const { return lora_alpha / lora_rank; }

  // LoRA weight pointers (directly pointing to Python tensor memory, zero-copy)
  // Layout: [expert_num, lora_rank, in_dim] for A, [expert_num, out_dim, lora_rank] for B
  void* gate_lora_a = nullptr;  // [expert_num, lora_rank, hidden_size]
  void* gate_lora_b = nullptr;  // [expert_num, intermediate_size, lora_rank]
  void* up_lora_a = nullptr;    // [expert_num, lora_rank, hidden_size]
  void* up_lora_b = nullptr;    // [expert_num, intermediate_size, lora_rank]
  void* down_lora_a = nullptr;  // [expert_num, lora_rank, intermediate_size]
  void* down_lora_b = nullptr;  // [expert_num, hidden_size, lora_rank]

  MOESFTConfig() : GeneralMOEConfig() {}

  MOESFTConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size) {}

  // Conversion constructor from GeneralMOEConfig (for MOE_TP_PART concept satisfaction)
  explicit MOESFTConfig(const GeneralMOEConfig& base) : GeneralMOEConfig(base) {
    // LoRA fields use default values (already initialized in struct definition)
  }
};

struct GeneralGateConfig {
  size_t hidden_size;
  size_t num_experts_per_tok;
  size_t n_routed_experts;
  size_t n_group;
  size_t topk_group;

  bool norm_topk_prob = true;
  float routed_scaling_factor = 2.5f;

  std::string scoring_func = "sigmoid";
  std::string topk_method = "noaux_tc";

  int layer_idx = 0;
  WorkerPool* pool = nullptr;

  void* weight = nullptr;
  ggml_type weight_type;
  void* e_score_correction_bias = nullptr;
  ggml_type e_score_correction_bias_type;

  size_t max_seqlen = 25600;

  GeneralGateConfig() = default;

  GeneralGateConfig(int hidden_size, int num_experts_per_tok, int n_routed_experts, int n_group, int topk_group)
      : hidden_size(hidden_size),
        num_experts_per_tok(num_experts_per_tok),
        n_routed_experts(n_routed_experts),
        n_group(n_group),
        topk_group(topk_group) {}
};

class MLA_Interface {
 public:
  virtual void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
                       const void* input, void* output) = 0;
};

class MoE_Interface {
 public:
  virtual void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output, bool incremental = false) = 0;
};
inline void init_ggml() {
  static bool inited = false;
  if (inited) {
    return;
  }
  struct ggml_init_params params = {
      0,
      NULL,
      true,
  };

  auto ctx_eval = ggml_init(params);

  if (!ctx_eval) {
    throw std::runtime_error("Failed to create ggml context");
  }
  inited = true;
}

template <typename A, typename B>
void convert_or_copy(A* dst, const B* src, size_t count) {
  if constexpr (std::is_same_v<A, B>) {
    // printf("Direct copy\n");
    memcpy(dst, src, sizeof(A) * count);
  } else {
    if constexpr (std::is_same_v<A, float>) {
      if constexpr (std::is_same_v<B, ggml_bf16_t>) {
        // printf("Converting ggml_bf16_t to float\n");
        ggml_bf16_to_fp32_row(src, dst, count);
      } else if constexpr (std::is_same_v<B, ggml_fp16_t>) {
        ggml_fp16_to_fp32_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    } else if constexpr (std::is_same_v<A, ggml_bf16_t>) {
      if constexpr (std::is_same_v<B, float>) {
        // printf("Converting float to ggml_bf16_t\n");
        ggml_fp32_to_bf16_row(src, dst, count);
      } else {
        throw std::runtime_error("Unsupported conversion");
      }
    }

    else {
      throw std::runtime_error("Unsupported conversion");
    }
  }
}

template <typename A>
void convert_or_copy(A* dst, void* src, ggml_type type, size_t count) {
  switch (type) {
    case GGML_TYPE_BF16: {
      auto src_bf16 = (ggml_bf16_t*)src;
      convert_or_copy(dst, src_bf16, count);
      break;
    }
    case GGML_TYPE_F16: {
#if defined(__aarch64__) && defined(CPU_USE_KML)
      auto src_fp16 = (float16_t*)src;
      convert_or_copy(dst, src_fp16, count);
#else
      throw std::runtime_error("GGML_TYPE_F16 is not supported on this platform");
#endif
      break;
    }
    case GGML_TYPE_F32: {
      auto src_f32 = (float*)src;
      convert_or_copy(dst, src_f32, count);
      break;
    }
    default:
      throw std::runtime_error("Unsupported type for conversion");
  }
}

template <typename A>
void check_numerics(A* data, size_t count) {
  for (size_t i = 0; i < count; i++) {
    if (std::isnan(data[i]) || std::isinf(data[i])) {
      printf("Numerics check failed at index %zu: value = %f\n", i, data[i]);
      throw std::runtime_error("Numerics check failed");
    }
  }
  printf("Numerics check passed for %zu elements.\n", count);
}

inline void debug_bf16(ggml_bf16_t* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", ggml_bf16_to_fp32(x[i]));
  }
  printf("\n");
}
inline void debug_f32(float* x) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", x[i]);
  }
  printf("\n");
}

inline void debug_f32(float* x, size_t count) {
  if (count < 10) {
    for (size_t i = 0; i < count; i++) {
      printf("%f ", x[i]);
    }
  } else {
    for (size_t i = 0; i < 3; i++) {
      printf("%f ", x[i]);
    }
    printf("...");
    for (size_t i = count - 3; i < count; i++) {
      printf("%f ", x[i]);
    }
    printf("\n");
  }
}

#endif
