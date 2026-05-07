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
  std::atomic<uint64_t> promote_count{0};      // Number of promote operations
  std::atomic<uint64_t> demote_count{0};       // Number of demote operations
  std::atomic<uint64_t> hit_count{0};          // Cache hit (expert already in NUMA)
  std::atomic<uint64_t> miss_count{0};         // Cache miss (need to promote)
  std::atomic<uint64_t> eviction_count{0};     // Number of evictions
  std::atomic<uint64_t> total_access_count{0}; // Total access count

  double hit_rate() const {
    uint64_t total = hit_count.load() + miss_count.load();
    return total > 0 ? static_cast<double>(hit_count.load()) / total : 0.0;
  }

  void reset() {
    promote_count.store(0);
    demote_count.store(0);
    hit_count.store(0);
    miss_count.store(0);
    eviction_count.store(0);
    total_access_count.store(0);
  }
};

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

  void* gate_proj;
  void* up_proj;
  void* down_proj;

  void* gate_scale;
  void* up_scale;
  void* down_scale;

  void* gate_zero;
  void* up_zero;
  void* down_zero;

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

  std::string path;
  bool save = false;
  bool load = false;

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
  ExpertCacheStats* cache_stats = nullptr;

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

  GeneralMOEConfig() {}

  GeneralMOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : expert_num(expert_num),
        num_experts_per_tok(routed_expert_num),
        hidden_size(hidden_size),
        intermediate_size(intermediate_size) {}

  int max_possible_qlen() { return std::max(max_len, group_max_len); }
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
