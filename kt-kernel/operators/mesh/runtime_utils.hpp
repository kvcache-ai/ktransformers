#ifndef CPUINFER_OPERATOR_MESH_RUNTIME_UTILS_HPP
#define CPUINFER_OPERATOR_MESH_RUNTIME_UTILS_HPP

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "expert_residency.hpp"

#ifndef _WIN32

namespace mesh {

inline bool prefill_stream_trace_enabled() {
  const char* trace = std::getenv("KT_MESH_PREFILL_STREAM_TRACE");
  return trace != nullptr && trace[0] != '\0' && trace[0] != '0';
}

inline void maybe_dump_prefill_expert_frequency(const GeneralMOEConfig& config,
                                                int tp_part_idx,
                                                int qlen,
                                                int k,
                                                const int64_t* expert_ids,
                                                const std::vector<int>& active_experts,
                                                const std::vector<int>& local_counts) {
  const char* path = std::getenv("KT_MESH_PREFILL_EXPERT_FREQ_PATH");
  if (path == nullptr || path[0] == '\0' || path[0] == '0' || tp_part_idx != 0 || qlen <= 0 || k <= 0 ||
      expert_ids == nullptr || config.expert_num <= 0) {
    return;
  }

  std::vector<uint64_t> router_counts(static_cast<size_t>(config.expert_num), 0);
  uint64_t router_routes = 0;
  uint64_t cpu_routes = 0;
  uint64_t gpu_skipped_routes = 0;
  uint64_t invalid_routes = 0;
  for (int i = 0; i < qlen; ++i) {
    for (int j = 0; j < k; ++j) {
      const int expert_id = static_cast<int>(expert_ids[i * k + j]);
      if (expert_id < 0 || expert_id >= config.expert_num) {
        invalid_routes += 1;
        continue;
      }
      router_counts[static_cast<size_t>(expert_id)] += 1;
      router_routes += 1;
      if (config.should_skip_expert(expert_id)) {
        gpu_skipped_routes += 1;
      } else {
        cpu_routes += 1;
      }
    }
  }

  int router_active = 0;
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (router_counts[static_cast<size_t>(expert_id)] > 0) {
      router_active += 1;
    }
  }

  static std::mutex dump_mu;
  std::lock_guard<std::mutex> guard(dump_mu);
  std::ofstream out(path, std::ios::app);
  if (!out.good()) return;

  out << "{\"layer\":" << config.layer_idx << ",\"tp\":" << tp_part_idx << ",\"qlen\":" << qlen
      << ",\"top_k\":" << k << ",\"expert_num\":" << config.expert_num
      << ",\"router_routes\":" << router_routes << ",\"cpu_routes\":" << cpu_routes
      << ",\"gpu_skipped_routes\":" << gpu_skipped_routes << ",\"invalid_routes\":" << invalid_routes
      << ",\"router_active_experts\":" << router_active << ",\"cpu_active_experts\":" << active_experts.size()
      << ",\"router_counts\":[";
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (expert_id > 0) out << ",";
    out << router_counts[static_cast<size_t>(expert_id)];
  }
  out << "],\"cpu_counts\":[";
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (expert_id > 0) out << ",";
    const int count = expert_id < static_cast<int>(local_counts.size()) ? local_counts[expert_id] : 0;
    out << count;
  }
  out << "],\"active_experts\":[";
  for (size_t i = 0; i < active_experts.size(); ++i) {
    if (i > 0) out << ",";
    out << active_experts[i];
  }
  out << "]}\n";
}

inline bool init_cgroup_memory_paths(std::string& memory_current_path, std::string& memory_max_path) {
  if (!memory_current_path.empty() && !memory_max_path.empty()) return true;

  std::ifstream cgroup_file("/proc/self/cgroup");
  std::string line;
  while (std::getline(cgroup_file, line)) {
    const std::string marker = "0::";
    if (line.rfind(marker, 0) != 0) continue;
    std::string rel = line.substr(marker.size());
    if (rel.empty() || rel[0] != '/') rel = "/" + rel;
    const std::string base = "/sys/fs/cgroup" + rel;
    memory_current_path = base + "/memory.current";
    memory_max_path = base + "/memory.max";
    return true;
  }
  return false;
}

inline bool read_uint64_file(const std::string& path, uint64_t* value, bool allow_max = false) {
  if (value == nullptr || path.empty()) return false;
  std::ifstream in(path);
  std::string text;
  if (!(in >> text)) return false;
  if (allow_max && text == "max") {
    *value = 0;
    return true;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
  if (errno != 0 || end == text.c_str() || parsed == 0ULL) return false;
  *value = static_cast<uint64_t>(parsed);
  return true;
}

inline bool read_cgroup_memory_scope(std::string& memory_current_path,
                                     std::string& memory_max_path,
                                     uint64_t* current,
                                     uint64_t* limit) {
  if (!init_cgroup_memory_paths(memory_current_path, memory_max_path)) return false;
  uint64_t cur = 0;
  uint64_t max = 0;
  if (!read_uint64_file(memory_current_path, &cur)) return false;
  if (!read_uint64_file(memory_max_path, &max, true)) return false;
  if (max == 0) return false;
  if (current != nullptr) *current = cur;
  if (limit != nullptr) *limit = max;
  return true;
}

inline bool memory_guard_pressure_active(const GeneralMOEConfig& config,
                                         int cache_capacity,
                                         std::string& memory_current_path,
                                         std::string& memory_max_path) {
  if (!config.mesh_memory_guard_enabled || cache_capacity <= 0) return false;
  uint64_t current = 0;
  uint64_t limit = 0;
  if (!read_cgroup_memory_scope(memory_current_path, memory_max_path, &current, &limit)) return false;
  const double high = std::max(0.0, std::min(1.0, static_cast<double>(config.mesh_memory_high_watermark)));
  return static_cast<double>(current) >= static_cast<double>(limit) * high;
}

struct MemoryGuardTrimCheck {
  uint64_t current = 0;
  uint64_t limit = 0;
  double high = 0.0;
  double target = 0.0;
  int max_demotes = 0;
};

inline bool prepare_memory_guard_trim_check(const GeneralMOEConfig& config,
                                            int cache_capacity,
                                            uint64_t& memory_guard_tick,
                                            std::string& memory_current_path,
                                            std::string& memory_max_path,
                                            MemoryGuardTrimCheck* check) {
  if (check == nullptr || !config.mesh_memory_guard_enabled || cache_capacity <= 0) return false;
  const int interval = std::max(1, config.mesh_memory_check_interval);
  memory_guard_tick += 1;
  if ((memory_guard_tick % static_cast<uint64_t>(interval)) != 0) return false;

  if (!read_cgroup_memory_scope(memory_current_path, memory_max_path, &check->current, &check->limit)) return false;
  check->high = std::max(0.0, std::min(1.0, static_cast<double>(config.mesh_memory_high_watermark)));
  check->target = std::max(0.0, std::min(check->high, static_cast<double>(config.mesh_memory_target_watermark)));
  check->max_demotes = std::max(1, config.mesh_memory_max_demotes_per_check);
  return static_cast<double>(check->current) >= static_cast<double>(check->limit) * check->high;
}

inline bool refresh_memory_guard_trim_check(std::string& memory_current_path,
                                            std::string& memory_max_path,
                                            MemoryGuardTrimCheck* check) {
  if (check == nullptr) return false;
  return read_cgroup_memory_scope(memory_current_path, memory_max_path, &check->current, &check->limit);
}

inline bool memory_guard_trim_target_reached(const MemoryGuardTrimCheck& check) {
  return static_cast<double>(check.current) <= static_cast<double>(check.limit) * check.target;
}

inline void log_amx_iouring_config(const GeneralMOEConfig& config,
                                   int tp_part_idx,
                                   int cache_capacity,
                                   size_t gate_weight_bytes,
                                   size_t gate_scale_bytes,
                                   size_t up_weight_bytes,
                                   size_t up_scale_bytes,
                                   size_t down_weight_bytes,
                                   size_t down_scale_bytes) {
  std::fprintf(stderr,
               "[MESHIO] layer=%d tp=%d backend=iouring direct_io=%s mmap_baseline=false capacity=%d policy=%s "
               "decode_capacity=%d prefill_static=%d prefill_layer_mode=%s lookahead=%s topk_fallback=%s "
               "w=%.3f gamma=%.3f beta=%.3f transition=%.3f "
               "prefetch=%d coldstart=%s coldstart_limit=%d "
               "mem_guard=%s high=%.3f target=%.3f interval=%d demotes=%d "
               "gate=%zu+%zu up=%zu+%zu down=%zu+%zu\n",
               config.layer_idx,
               tp_part_idx,
               config.iouring_direct_io ? "true" : "false",
               cache_capacity,
               config.resident_cache_policy.c_str(),
               config.mesh_decode_resident_experts,
               config.mesh_prefill_static_experts,
               config.mesh_prefill_layer_mode_enabled ? "true" : "false",
               config.mesh_lookahead_enabled ? "true" : "false",
               config.mesh_topk_fallback_enabled ? "true" : "false",
               config.mesh_lookahead_weight,
               config.mesh_heat_gamma,
               config.mesh_heat_beta,
               config.mesh_transition_alpha,
               config.mesh_prefetch_budget,
               config.mesh_coldstart_prefill_enabled ? "true" : "false",
               config.mesh_coldstart_prefill_limit,
               config.mesh_memory_guard_enabled ? "true" : "false",
               config.mesh_memory_high_watermark,
               config.mesh_memory_target_watermark,
               config.mesh_memory_check_interval,
               config.mesh_memory_max_demotes_per_check,
               gate_weight_bytes,
               gate_scale_bytes,
               up_weight_bytes,
               up_scale_bytes,
               down_weight_bytes,
               down_scale_bytes);
}

inline void validate_amx_iouring_config(const GeneralMOEConfig& config,
                                        int tp_part_idx,
                                        int cache_capacity,
                                        size_t gate_weight_bytes,
                                        size_t gate_scale_bytes,
                                        size_t up_weight_bytes,
                                        size_t up_scale_bytes,
                                        size_t down_weight_bytes,
                                        size_t down_scale_bytes,
                                        size_t gate_mins_bytes,
                                        size_t up_mins_bytes,
                                        size_t down_mins_bytes,
                                        bool include_mins) {
  if (config.async_reader == nullptr) {
    throw std::runtime_error("io_uring backend requires a non-null AsyncExpertReader");
  }
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    (void)logical_expert_id_for_slot(config, tp_part_idx, expert_id);
  }
  validate_file_slot_matrix(config, tp_part_idx, "gate.weight", config.gate_file_slots, gate_weight_bytes, false);
  validate_file_slot_matrix(config, tp_part_idx, "gate.scale", config.gate_scale_file_slots, gate_scale_bytes, false);
  validate_file_slot_matrix(config, tp_part_idx, "up.weight", config.up_file_slots, up_weight_bytes, false);
  validate_file_slot_matrix(config, tp_part_idx, "up.scale", config.up_scale_file_slots, up_scale_bytes, false);
  validate_file_slot_matrix(config, tp_part_idx, "down.weight", config.down_file_slots, down_weight_bytes, false);
  validate_file_slot_matrix(config, tp_part_idx, "down.scale", config.down_scale_file_slots, down_scale_bytes, false);
  if (include_mins) {
    if (!config.gate_mins_file_slots.empty()) {
      validate_file_slot_matrix(config, tp_part_idx, "gate.mins", config.gate_mins_file_slots, gate_mins_bytes, false);
    }
    if (!config.up_mins_file_slots.empty()) {
      validate_file_slot_matrix(config, tp_part_idx, "up.mins", config.up_mins_file_slots, up_mins_bytes, false);
    }
    if (!config.down_mins_file_slots.empty()) {
      validate_file_slot_matrix(config, tp_part_idx, "down.mins", config.down_mins_file_slots, down_mins_bytes, false);
    }
  }
  log_amx_iouring_config(config,
                         tp_part_idx,
                         cache_capacity,
                         gate_weight_bytes,
                         gate_scale_bytes,
                         up_weight_bytes,
                         up_scale_bytes,
                         down_weight_bytes,
                         down_scale_bytes);
}

inline void log_bf16_iouring_config(const GeneralMOEConfig& config,
                                    int tp_part_idx,
                                    int cache_capacity,
                                    size_t gate_weight_bytes,
                                    size_t up_weight_bytes,
                                    size_t down_full_weight_bytes,
                                    size_t down_local_weight_bytes) {
  std::fprintf(stderr,
               "[BF16_MESHIO] layer=%d tp=%d backend=iouring direct_io=%s capacity=%d policy=%s "
               "gate_bytes=%.3fMiB up_bytes=%.3fMiB down_full_bytes=%.3fMiB down_local_bytes=%.3fMiB\n",
               config.layer_idx,
               tp_part_idx,
               config.iouring_direct_io ? "true" : "false",
               cache_capacity,
               config.resident_cache_policy.c_str(),
               static_cast<double>(gate_weight_bytes) / (1024.0 * 1024.0),
               static_cast<double>(up_weight_bytes) / (1024.0 * 1024.0),
               static_cast<double>(down_full_weight_bytes) / (1024.0 * 1024.0),
               static_cast<double>(down_local_weight_bytes) / (1024.0 * 1024.0));
}

inline void validate_bf16_iouring_config(const GeneralMOEConfig& config,
                                         int tp_part_idx,
                                         int cache_capacity,
                                         size_t gate_weight_bytes,
                                         size_t up_weight_bytes,
                                         size_t down_full_weight_bytes,
                                         size_t down_local_weight_bytes) {
  if (config.async_reader == nullptr) {
    throw std::runtime_error("BF16 io_uring requires a non-null AsyncExpertReader");
  }
  validate_file_slot_matrix(config, tp_part_idx, "gate.weight", config.gate_file_slots, gate_weight_bytes, true);
  validate_file_slot_matrix(config, tp_part_idx, "up.weight", config.up_file_slots, up_weight_bytes, true);
  validate_file_slot_matrix(config, tp_part_idx, "down.weight", config.down_file_slots, down_full_weight_bytes, true);
  log_bf16_iouring_config(
      config, tp_part_idx, cache_capacity, gate_weight_bytes, up_weight_bytes, down_full_weight_bytes,
      down_local_weight_bytes);
}

inline void log_bf16_iouring_temp_alloc_failure(int expert_id,
                                                const void* gate_raw,
                                                const void* up_raw,
                                                const void* down_full_raw) {
  std::fprintf(stderr,
               "[BF16_PROMOTION_FAIL] reason=iouring_temp_alloc expert=%d gate_raw=%p up_raw=%p down_full_raw=%p\n",
               expert_id,
               gate_raw,
               up_raw,
               down_full_raw);
}

inline void log_bf16_iouring_read_failure(int expert_id,
                                          const std::vector<uint64_t>& request_ids,
                                          const ktransformers::AsyncExpertReader* async_reader) {
  std::fprintf(stderr,
               "[BF16_PROMOTION_FAIL] reason=iouring_read expert=%d requests=%s\n",
               expert_id,
               async_reader == nullptr ? "" : async_reader->describe_requests(request_ids).c_str());
}

inline void log_bf16_cache_full_failure(const GeneralMOEConfig& config,
                                        int expert_id,
                                        int resident,
                                        int capacity,
                                        int cached_count,
                                        int pinned_count,
                                        int packing_count,
                                        int demoting_count,
                                        int busy_count,
                                        size_t resident_bytes_per_expert,
                                        size_t promotion_temp_bytes) {
  std::fprintf(stderr,
               "[BF16_PROMOTION_FAIL] reason=cache_full expert=%d resident=%d capacity=%d cached=%d pinned=%d "
               "packing=%d demoting=%d busy_cached=%d resident_bytes_per_expert=%.3fMiB "
               "resident_total_est=%.3fGiB promotion_temp=%.3fMiB\n",
               expert_id,
               resident,
               capacity,
               cached_count,
               pinned_count,
               packing_count,
               demoting_count,
               busy_count,
               static_cast<double>(resident_bytes_per_expert) / (1024.0 * 1024.0),
               static_cast<double>(resident_bytes_per_expert * static_cast<size_t>(resident)) /
                   (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(promotion_temp_bytes) / (1024.0 * 1024.0));
  (void)config;
}

inline void log_bf16_numa_alloc_failure(int expert_id,
                                        size_t gate_weight_bytes,
                                        size_t up_weight_bytes,
                                        size_t down_weight_bytes,
                                        const void* gate_owner,
                                        const void* up_owner,
                                        const void* down_owner,
                                        int numa_node) {
  std::fprintf(stderr,
               "[BF16_PROMOTION_FAIL] reason=numa_alloc expert=%d gate_bytes=%zu up_bytes=%zu down_bytes=%zu "
               "gate_ptr=%p up_ptr=%p down_ptr=%p numa=%d\n",
               expert_id,
               gate_weight_bytes,
               up_weight_bytes,
               down_weight_bytes,
               gate_owner,
               up_owner,
               down_owner,
               numa_node);
}

inline void log_bf16_missing_baseline_failure(int expert_id,
                                              const void* gate_src,
                                              const void* up_src,
                                              const void* down_src) {
  std::fprintf(stderr,
               "[BF16_PROMOTION_FAIL] reason=missing_baseline expert=%d gate_src=%p up_src=%p down_src=%p\n",
               expert_id,
               gate_src,
               up_src,
               down_src);
}

inline bool trace_cpu_mem_breakdown_enabled() {
  const char* trace = std::getenv("KT_TRACE_CPU_MEM_BREAKDOWN");
  return trace != nullptr &&
         (std::strcmp(trace, "1") == 0 || std::strcmp(trace, "true") == 0 || std::strcmp(trace, "TRUE") == 0);
}

inline bool bf16_wave_resident_enabled() {
  const char* raw = std::getenv("KT_ENABLE_BF16_WAVE_RESIDENT");
  return raw != nullptr &&
         (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 || std::strcmp(raw, "TRUE") == 0);
}

inline void maybe_log_cpu_mem_resident(const GeneralMOEConfig& config,
                                       int tp_part_idx,
                                       int expert_id,
                                       int resident_count,
                                       int cache_capacity,
                                       size_t resident_bytes_per_expert,
                                       size_t promotion_temp_bytes,
                                       bool pin_after_pack) {
  if (!trace_cpu_mem_breakdown_enabled()) return;
  std::fprintf(stderr,
               "[KT_CPU_MEM_RESIDENT] layer=%d tp=%d expert=%d resident_count=%d capacity=%d "
               "resident_bytes_per_expert=%.3fMiB resident_total_est=%.3fGiB promotion_temp=%.3fMiB state=%s\n",
               config.layer_idx,
               tp_part_idx,
               expert_id,
               resident_count,
               cache_capacity,
               static_cast<double>(resident_bytes_per_expert) / (1024.0 * 1024.0),
               static_cast<double>(resident_bytes_per_expert * static_cast<size_t>(resident_count)) /
                   (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(promotion_temp_bytes) / (1024.0 * 1024.0),
               pin_after_pack ? "PINNED" : "CACHED");
}

inline uint64_t runtime_elapsed_us(std::chrono::steady_clock::time_point start,
                                   std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

inline void submit_prefill_promotion_reads(
    const GeneralMOEConfig& config,
    int tp_part_idx,
    std::vector<BatchPromotion>* promotions,
    const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch,
    std::vector<uint64_t>* all_requests,
    PrefillPromotionTiming* timing = nullptr) {
  if (promotions == nullptr || all_requests == nullptr || read_batch.empty()) return;
  ktransformers::AsyncExpertReader::SubmitStats submit_stats;
  const auto reader_submit_start = std::chrono::steady_clock::now();
  std::vector<uint64_t> request_ids =
      config.async_reader->submit_reads(read_batch, timing != nullptr ? &submit_stats : nullptr);
  const auto reader_submit_end = std::chrono::steady_clock::now();
  if (timing != nullptr) {
    timing->reader_submit_us += runtime_elapsed_us(reader_submit_start, reader_submit_end);
    timing->reader_total_us += submit_stats.total_us;
    timing->reader_lock_wait_us += submit_stats.lock_wait_us;
    timing->reader_sqe_prep_us += submit_stats.sqe_prep_us;
    timing->reader_bookkeeping_us += submit_stats.request_bookkeeping_us;
    timing->ring_submit_us += submit_stats.ring_submit_us;
    timing->read_requests += submit_stats.request_count;
    timing->ring_flushes += submit_stats.flush_count;
  }
  if (request_ids.size() != read_batch.size()) {
    std::ostringstream oss;
    oss << "AMX io_uring prefill batched submit returned wrong request count layer=" << config.layer_idx
        << " tp=" << tp_part_idx << " expected=" << read_batch.size() << " actual=" << request_ids.size();
    throw std::runtime_error(oss.str());
  }
  const auto stats_start = std::chrono::steady_clock::now();
  record_iouring_read_stats(config.cache_stats, config.enable_cache_stats, read_batch);
  if (timing != nullptr) {
    timing->reader_stats_us += runtime_elapsed_us(stats_start, std::chrono::steady_clock::now());
  }

  const auto slice_start = std::chrono::steady_clock::now();
  size_t offset = 0;
  for (auto& promotion : *promotions) {
    const size_t count = promotion.request_count;
    if (count == 0) continue;
    if (offset + count > request_ids.size()) {
      throw std::runtime_error("AMX io_uring prefill batched submit request slicing overflow");
    }
    promotion.requests.assign(request_ids.begin() + static_cast<std::ptrdiff_t>(offset),
                              request_ids.begin() + static_cast<std::ptrdiff_t>(offset + count));
    all_requests->insert(all_requests->end(), promotion.requests.begin(), promotion.requests.end());
    offset += count;
  }
  if (timing != nullptr) {
    timing->request_slice_us += runtime_elapsed_us(slice_start, std::chrono::steady_clock::now());
  }
}

struct PrefillScratchStateCounts {
  int hit = 0;
  int cold = 0;
  int inflight = 0;
  int other = 0;
  uint64_t expected_read_bytes = 0;
};

inline PrefillScratchStateCounts count_prefill_scratch_states(const GeneralMOEConfig& config,
                                                              int tp_part_idx,
                                                              const std::vector<int>& scratch_active_experts,
                                                              const std::atomic<uint8_t>* expert_states,
                                                              bool include_mins) {
  PrefillScratchStateCounts counts;
  if (expert_states == nullptr) return counts;
  for (int expert_id : scratch_active_experts) {
    if (expert_id < 0 || expert_id >= config.expert_num) {
      counts.other += 1;
      continue;
    }
    const uint8_t state = expert_states[expert_id].load(std::memory_order_acquire);
    if (state == EXPERT_CACHED || state == EXPERT_PINNED) {
      counts.hit += 1;
    } else if (state == EXPERT_BASELINE) {
      counts.cold += 1;
      counts.expected_read_bytes += amx_iouring_read_bytes_for_expert(config, tp_part_idx, expert_id, include_mins);
    } else if (state == EXPERT_PREFETCHING) {
      counts.inflight += 1;
    } else {
      counts.other += 1;
    }
  }
  return counts;
}

struct PrefillStaticScratchTraceData {
  int qlen = 0;
  int active = 0;
  size_t static_active = 0;
  size_t scratch_active = 0;
  PrefillScratchStateCounts scratch_state;
  int static_slots = 0;
  int scratch_slots = 0;
  uint64_t scratch_submit_us = 0;
  uint64_t scratch_wait_us = 0;
  uint64_t static_compute_us = 0;
  uint64_t scratch_compute_us = 0;
  uint64_t read_req_delta = 0;
  uint64_t read_bytes_delta = 0;
  int resident_before = 0;
  int resident_after = 0;
  int allocated_slots = 0;
  PrefillPromotionTiming scratch_timing;
  PrefillPromotionTiming static_timing;
};

inline void log_prefill_static_scratch_trace(const GeneralMOEConfig& config,
                                             int tp_part_idx,
                                             const PrefillStaticScratchTraceData& data) {
  std::fprintf(stderr,
               "[MESH_PREFILL_STATIC_SCRATCH_TRACE] layer=%d tp=%d qlen=%d active=%d static_active=%zu "
               "scratch_active=%zu scratch_hit_before=%d scratch_cold_before=%d "
               "scratch_inflight_before=%d scratch_other_before=%d static_slots=%d scratch_slots=%d "
               "scratch_submit_us=%llu scratch_wait_us=%llu static_compute_us=%llu scratch_compute_us=%llu "
               "read_req_delta=%llu read_bytes_delta=%llu expected_scratch_read_bytes=%llu "
               "resident_before=%d resident_after=%d allocated_slots=%d "
               "scratch_state_us=%llu scratch_slot_us=%llu scratch_slot_alloc_us=%llu "
               "scratch_request_build_us=%llu scratch_reader_submit_us=%llu "
               "scratch_reader_total_us=%llu scratch_reader_lock_wait_us=%llu "
               "scratch_reader_sqe_prep_us=%llu scratch_reader_bookkeeping_us=%llu "
               "scratch_ring_submit_us=%llu scratch_reader_stats_us=%llu scratch_request_slice_us=%llu "
               "scratch_promote_calls=%llu scratch_allocated_slots=%llu scratch_reused_slots=%llu "
               "scratch_read_requests=%llu scratch_ring_flushes=%llu "
               "static_state_us=%llu static_slot_us=%llu static_slot_alloc_us=%llu "
               "static_request_build_us=%llu static_reader_submit_us=%llu "
               "static_reader_total_us=%llu static_reader_lock_wait_us=%llu "
               "static_reader_sqe_prep_us=%llu static_reader_bookkeeping_us=%llu "
               "static_ring_submit_us=%llu static_reader_stats_us=%llu static_request_slice_us=%llu "
               "static_promote_calls=%llu static_allocated_slots=%llu static_reused_slots=%llu "
               "static_read_requests=%llu static_ring_flushes=%llu\n",
               config.layer_idx,
               tp_part_idx,
               data.qlen,
               data.active,
               data.static_active,
               data.scratch_active,
               data.scratch_state.hit,
               data.scratch_state.cold,
               data.scratch_state.inflight,
               data.scratch_state.other,
               data.static_slots,
               data.scratch_slots,
               static_cast<unsigned long long>(data.scratch_submit_us),
               static_cast<unsigned long long>(data.scratch_wait_us),
               static_cast<unsigned long long>(data.static_compute_us),
               static_cast<unsigned long long>(data.scratch_compute_us),
               static_cast<unsigned long long>(data.read_req_delta),
               static_cast<unsigned long long>(data.read_bytes_delta),
               static_cast<unsigned long long>(data.scratch_state.expected_read_bytes),
               data.resident_before,
               data.resident_after,
               data.allocated_slots,
               static_cast<unsigned long long>(data.scratch_timing.state_us),
               static_cast<unsigned long long>(data.scratch_timing.slot_us),
               static_cast<unsigned long long>(data.scratch_timing.slot_alloc_us),
               static_cast<unsigned long long>(data.scratch_timing.request_build_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_submit_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_total_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_lock_wait_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_sqe_prep_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_bookkeeping_us),
               static_cast<unsigned long long>(data.scratch_timing.ring_submit_us),
               static_cast<unsigned long long>(data.scratch_timing.reader_stats_us),
               static_cast<unsigned long long>(data.scratch_timing.request_slice_us),
               static_cast<unsigned long long>(data.scratch_timing.promote_calls),
               static_cast<unsigned long long>(data.scratch_timing.allocated_slots),
               static_cast<unsigned long long>(data.scratch_timing.reused_slots),
               static_cast<unsigned long long>(data.scratch_timing.read_requests),
               static_cast<unsigned long long>(data.scratch_timing.ring_flushes),
               static_cast<unsigned long long>(data.static_timing.state_us),
               static_cast<unsigned long long>(data.static_timing.slot_us),
               static_cast<unsigned long long>(data.static_timing.slot_alloc_us),
               static_cast<unsigned long long>(data.static_timing.request_build_us),
               static_cast<unsigned long long>(data.static_timing.reader_submit_us),
               static_cast<unsigned long long>(data.static_timing.reader_total_us),
               static_cast<unsigned long long>(data.static_timing.reader_lock_wait_us),
               static_cast<unsigned long long>(data.static_timing.reader_sqe_prep_us),
               static_cast<unsigned long long>(data.static_timing.reader_bookkeeping_us),
               static_cast<unsigned long long>(data.static_timing.ring_submit_us),
               static_cast<unsigned long long>(data.static_timing.reader_stats_us),
               static_cast<unsigned long long>(data.static_timing.request_slice_us),
               static_cast<unsigned long long>(data.static_timing.promote_calls),
               static_cast<unsigned long long>(data.static_timing.allocated_slots),
               static_cast<unsigned long long>(data.static_timing.reused_slots),
               static_cast<unsigned long long>(data.static_timing.read_requests),
               static_cast<unsigned long long>(data.static_timing.ring_flushes));
}

inline void log_prefill_stream_summary(const GeneralMOEConfig& config,
                                       int tp_part_idx,
                                       int qlen,
                                       int active,
                                       size_t static_active,
                                       size_t scratch_active,
                                       int static_slots,
                                       int scratch_slots,
                                       uint64_t copy_input_us,
                                       uint64_t merge_us,
                                       uint64_t total_us,
                                       int resident_final,
                                       int allocated_slots) {
  std::fprintf(stderr,
               "[MESH_PREFILL_STREAM_SUMMARY] layer=%d tp=%d qlen=%d active=%d static_active=%zu "
               "scratch_active=%zu static_slots=%d scratch_slots=%d "
               "copy_input_us=%llu merge_us=%llu total_us=%llu resident_final=%d allocated_slots=%d\n",
               config.layer_idx,
               tp_part_idx,
               qlen,
               active,
               static_active,
               scratch_active,
               static_slots,
               scratch_slots,
               static_cast<unsigned long long>(copy_input_us),
               static_cast<unsigned long long>(merge_us),
               static_cast<unsigned long long>(total_us),
               resident_final,
               allocated_slots);
}

inline void log_batch_ensure_trace(const GeneralMOEConfig& config,
                                   int tp_part_idx,
                                   size_t requested,
                                   int unique,
                                   int hits,
                                   int cold,
                                   int inflight,
                                   size_t cold_promotions,
                                   size_t pending_prefetch,
                                   uint64_t wait_us,
                                   uint64_t total_us,
                                   uint64_t read_req_delta,
                                   uint64_t read_bytes_delta,
                                   uint64_t expected_cold_read_bytes,
                                   int resident,
                                   int capacity) {
  std::fprintf(stderr,
               "[MESH_BATCH_ENSURE_TRACE] layer=%d tp=%d requested=%zu unique=%d hits=%d cold=%d inflight=%d "
               "cold_promotions=%zu pending_prefetch=%zu wait_us=%llu total_us=%llu "
               "read_req_delta=%llu read_bytes_delta=%llu expected_cold_read_bytes=%llu resident=%d capacity=%d\n",
               config.layer_idx,
               tp_part_idx,
               requested,
               unique,
               hits,
               cold,
               inflight,
               cold_promotions,
               pending_prefetch,
               static_cast<unsigned long long>(wait_us),
               static_cast<unsigned long long>(total_us),
               static_cast<unsigned long long>(read_req_delta),
               static_cast<unsigned long long>(read_bytes_delta),
               static_cast<unsigned long long>(expected_cold_read_bytes),
               resident,
               capacity);
}

struct StateDeferStats {
  uint64_t token_count = 0;
  uint64_t cpu_topk_count = 0;
  uint64_t gpu_skip_count = 0;
  uint64_t nonready_count = 0;
  uint64_t deferred_count = 0;
  uint64_t overflow_immediate_count = 0;
  uint64_t overflow_token_count = 0;
};

inline void record_state_defer_stats(const GeneralMOEConfig& config, const StateDeferStats& stats) {
  ExpertCacheStats* cache_stats = enabled_cache_stats(config);
  if (cache_stats == nullptr) return;
  cache_stats->state_defer_token_count.fetch_add(stats.token_count, std::memory_order_relaxed);
  cache_stats->state_defer_cpu_topk_count.fetch_add(stats.cpu_topk_count, std::memory_order_relaxed);
  cache_stats->state_defer_gpu_skip_count.fetch_add(stats.gpu_skip_count, std::memory_order_relaxed);
  cache_stats->state_defer_nonready_count.fetch_add(stats.nonready_count, std::memory_order_relaxed);
  cache_stats->state_defer_deferred_count.fetch_add(stats.deferred_count, std::memory_order_relaxed);
  cache_stats->state_defer_overflow_immediate_count.fetch_add(stats.overflow_immediate_count,
                                                              std::memory_order_relaxed);
  cache_stats->state_defer_overflow_token_count.fetch_add(stats.overflow_token_count, std::memory_order_relaxed);
  cache_stats->maybe_dump_jsonl();
}

inline void record_bootstrap_prefetch_stats(const GeneralMOEConfig& config,
                                            int tp_part_idx,
                                            bool is_bootstrap_prefetch,
                                            uint64_t candidate_count,
                                            uint64_t submitted,
                                            uint64_t skip_gpu_count,
                                            uint64_t skip_resident_count) {
  ExpertCacheStats* cache_stats = enabled_cache_stats(config);
  if (!is_bootstrap_prefetch || tp_part_idx != 0 || cache_stats == nullptr) return;
  cache_stats->bootstrap_prefetch_candidate_count.fetch_add(candidate_count, std::memory_order_relaxed);
  cache_stats->bootstrap_prefetch_submit_count.fetch_add(submitted, std::memory_order_relaxed);
  cache_stats->bootstrap_prefetch_skip_gpu_count.fetch_add(skip_gpu_count, std::memory_order_relaxed);
  cache_stats->bootstrap_prefetch_skip_resident_count.fetch_add(skip_resident_count, std::memory_order_relaxed);
  cache_stats->coldstart_prefill_count.fetch_add(submitted, std::memory_order_relaxed);
}

inline void record_memory_guard_demotes(const GeneralMOEConfig& config, uint64_t demoted) {
  ExpertCacheStats* cache_stats = enabled_cache_stats(config);
  if (cache_stats == nullptr || demoted == 0) return;
  cache_stats->memory_guard_demote_count.fetch_add(demoted, std::memory_order_relaxed);
}

inline void record_lookahead_update(const GeneralMOEConfig& config) {
  ExpertCacheStats* cache_stats = enabled_cache_stats(config);
  if (cache_stats == nullptr) return;
  cache_stats->lookahead_update_count.fetch_add(1, std::memory_order_relaxed);
  if (config.mesh_transition_alpha > 0.0f) {
    cache_stats->transition_update_count.fetch_add(1, std::memory_order_relaxed);
  }
}

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_RUNTIME_UTILS_HPP
