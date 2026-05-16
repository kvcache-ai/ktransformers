#ifndef CPUINFER_OPERATOR_MESH_EXPERT_RESIDENCY_HPP
#define CPUINFER_OPERATOR_MESH_EXPERT_RESIDENCY_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "async_io.hpp"
#include "../common.hpp"

#ifndef _WIN32

enum ExpertState : uint8_t {
  EXPERT_BASELINE = 0,
  EXPERT_PACKING = 1,
  EXPERT_CACHED = 2,
  EXPERT_PINNED = 3,
  EXPERT_DEMOTING = 4,
  EXPERT_PREFETCHING = 5,
};

enum SlotState : uint8_t {
  SLOT_EMPTY = 0,
  SLOT_LOADING = 1,
  SLOT_READY = 2,
};

enum MeshSlotMode : uint8_t {
  MESH_SLOT_DECODE_CACHE = 0,
  MESH_SLOT_PREFILL_LAYER = 1,
};

struct PendingPrefetch {
  int slot_index = -1;
  std::vector<uint64_t> requests;
};

struct BatchPromotion {
  int expert_id = -1;
  int slot = -1;
  void* gate_owner = nullptr;
  void* up_owner = nullptr;
  void* down_owner = nullptr;
  size_t request_count = 0;
  std::vector<uint64_t> requests;
};

struct PrefillPromotionTiming {
  uint64_t state_us = 0;
  uint64_t slot_us = 0;
  uint64_t slot_alloc_us = 0;
  uint64_t request_build_us = 0;
  uint64_t reader_submit_us = 0;
  uint64_t reader_total_us = 0;
  uint64_t reader_lock_wait_us = 0;
  uint64_t reader_sqe_prep_us = 0;
  uint64_t reader_bookkeeping_us = 0;
  uint64_t ring_submit_us = 0;
  uint64_t reader_stats_us = 0;
  uint64_t request_slice_us = 0;
  uint64_t promote_calls = 0;
  uint64_t allocated_slots = 0;
  uint64_t reused_slots = 0;
  uint64_t read_requests = 0;
  uint64_t ring_flushes = 0;
};

namespace mesh {

template <class Owner>
class ExpertReadScope {
 public:
  explicit ExpertReadScope(Owner* owner, size_t reserve_count = 0) : owner_(owner) {
    experts_.reserve(reserve_count);
  }

  void add_expert(int expert_id) {
    owner_->acquire_expert_read(expert_id);
    experts_.push_back(expert_id);
  }

  ~ExpertReadScope() {
    for (int expert_id : experts_) {
      owner_->release_expert_read(expert_id);
    }
  }

 private:
  Owner* owner_ = nullptr;
  std::vector<int> experts_;
};

inline bool iouring_enabled(const GeneralMOEConfig& config) {
  return config.io_backend == IOBackend::IOURING;
}

inline bool lazy_weight_enabled(const GeneralMOEConfig& config) {
  return iouring_enabled(config);
}

inline int logical_expert_id_for_slot(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  if (expert_id < 0 || expert_id >= config.expert_num) {
    throw std::runtime_error("Invalid expert id for io_uring slot lookup");
  }
  const auto* physical_to_logical_map = reinterpret_cast<const uint64_t*>(config.physical_to_logical_map);
  const int logical_expert_id =
      physical_to_logical_map == nullptr ? expert_id : static_cast<int>(physical_to_logical_map[expert_id]);
  if (logical_expert_id < 0 || logical_expert_id >= config.expert_num) {
    std::ostringstream oss;
    oss << "Invalid physical_to_logical_map entry for layer=" << config.layer_idx << " tp=" << tp_part_idx
        << " expert=" << expert_id << " logical=" << logical_expert_id;
    throw std::runtime_error(oss.str());
  }
  return logical_expert_id;
}

inline const ExpertFileSlot& file_slot_at(const GeneralMOEConfig& config,
                                          int tp_part_idx,
                                          const std::vector<std::vector<ExpertFileSlot>>& slots,
                                          const char* name,
                                          int expert_id,
                                          bool apply_physical_to_logical_map = true) {
  if (tp_part_idx < 0 || tp_part_idx >= static_cast<int>(slots.size())) {
    std::ostringstream oss;
    oss << "io_uring file slots for " << name << " do not contain tp=" << tp_part_idx
        << " rows=" << slots.size() << " layer=" << config.layer_idx;
    throw std::runtime_error(oss.str());
  }

  const int slot_expert_id =
      apply_physical_to_logical_map ? logical_expert_id_for_slot(config, tp_part_idx, expert_id) : expert_id;
  const auto& row = slots[tp_part_idx];
  if (slot_expert_id < 0 || slot_expert_id >= static_cast<int>(row.size())) {
    std::ostringstream oss;
    oss << "io_uring file slots for " << name << " do not contain expert=" << slot_expert_id
        << " row_size=" << row.size() << " layer=" << config.layer_idx << " tp=" << tp_part_idx;
    throw std::runtime_error(oss.str());
  }
  return row[slot_expert_id];
}

inline void validate_file_slot(const GeneralMOEConfig& config,
                               int tp_part_idx,
                               const char* name,
                               const ExpertFileSlot& slot,
                               size_t expected_size) {
  if (slot.fd < 0 || slot.size == 0) {
    std::ostringstream oss;
    oss << "Invalid io_uring slot for " << name << " layer=" << config.layer_idx << " tp=" << tp_part_idx
        << " fd=" << slot.fd << " offset=" << slot.offset << " size=" << slot.size;
    throw std::runtime_error(oss.str());
  }
  if (slot.size != expected_size) {
    std::ostringstream oss;
    oss << "Unexpected io_uring slot size for " << name << " layer=" << config.layer_idx << " tp=" << tp_part_idx
        << " expected=" << expected_size << " actual=" << slot.size << " offset=" << slot.offset;
    throw std::runtime_error(oss.str());
  }
  if (config.iouring_direct_io && ((slot.offset % 512) != 0 || (slot.size % 512) != 0)) {
    std::ostringstream oss;
    oss << "io_uring O_DIRECT slot for " << name << " is not 512-byte aligned layer=" << config.layer_idx
        << " tp=" << tp_part_idx << " offset=" << slot.offset << " size=" << slot.size;
    throw std::runtime_error(oss.str());
  }
}

inline void validate_file_slot_matrix(const GeneralMOEConfig& config,
                                      int tp_part_idx,
                                      const char* name,
                                      const std::vector<std::vector<ExpertFileSlot>>& slots,
                                      size_t expected_size,
                                      bool apply_physical_to_logical_map = false) {
  if (tp_part_idx < 0 || tp_part_idx >= static_cast<int>(slots.size())) {
    std::ostringstream oss;
    oss << "io_uring backend requires " << name << " slots for tp=" << tp_part_idx
        << " rows=" << slots.size() << " layer=" << config.layer_idx;
    throw std::runtime_error(oss.str());
  }
  if (static_cast<int>(slots[tp_part_idx].size()) < config.expert_num) {
    std::ostringstream oss;
    oss << "io_uring backend requires " << name << " slots for every expert layer=" << config.layer_idx
        << " tp=" << tp_part_idx << " experts=" << config.expert_num
        << " row_size=" << slots[tp_part_idx].size();
    throw std::runtime_error(oss.str());
  }
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    const auto& slot = file_slot_at(config, tp_part_idx, slots, name, expert_id, apply_physical_to_logical_map);
    validate_file_slot(config, tp_part_idx, name, slot, expected_size);
  }
}

inline void record_iouring_read_stats(ExpertCacheStats* cache_stats,
                                      bool enabled,
                                      const std::vector<ktransformers::AsyncExpertReader::ReadRequest>& read_batch) {
  if (!enabled || cache_stats == nullptr || read_batch.empty()) return;
  cache_stats->iouring_read_request_count.fetch_add(static_cast<uint64_t>(read_batch.size()),
                                                    std::memory_order_relaxed);
  uint64_t read_bytes = 0;
  for (const auto& req : read_batch) {
    read_bytes += static_cast<uint64_t>(req.size);
  }
  cache_stats->iouring_read_bytes.fetch_add(read_bytes, std::memory_order_relaxed);
}

inline ExpertCacheStats* enabled_cache_stats(const GeneralMOEConfig& config) {
  return config.enable_cache_stats ? config.cache_stats : nullptr;
}

inline void record_cache_total_access(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->total_access_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_access(expert_id);
  }
}

inline void record_cache_hit(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->hit_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_hit(expert_id);
  }
}

inline void record_cache_cold_miss(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->miss_count.fetch_add(1, std::memory_order_relaxed);
  stats->cold_miss_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_miss(expert_id);
    stats->note_expert_cold_miss(expert_id);
  }
}

inline void record_cache_inflight_miss(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->miss_count.fetch_add(1, std::memory_order_relaxed);
  stats->in_flight_miss_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_miss(expert_id);
    stats->note_expert_in_flight_miss(expert_id);
  }
}

inline void record_cache_promote(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->promote_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_promote(expert_id);
  }
}

inline void record_cache_prefetch_hit(const GeneralMOEConfig& config, int tp_part_idx, int expert_id) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->prefetch_hit_count.fetch_add(1, std::memory_order_relaxed);
  if (tp_part_idx == 0) {
    stats->note_expert_prefetch_hit(expert_id);
  }
}

inline void record_cache_eviction_demote(const GeneralMOEConfig& config) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->eviction_count.fetch_add(1, std::memory_order_relaxed);
  stats->demote_count.fetch_add(1, std::memory_order_relaxed);
}

inline void record_cache_prefetch_submit(const GeneralMOEConfig& config, uint64_t submitted) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr || submitted == 0) return;
  stats->prefetch_count.fetch_add(submitted, std::memory_order_relaxed);
}

inline void record_cache_async_prefetch(const GeneralMOEConfig& config) {
  ExpertCacheStats* stats = enabled_cache_stats(config);
  if (stats == nullptr) return;
  stats->async_prefetch_count.fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t amx_iouring_read_bytes_for_expert(const GeneralMOEConfig& config,
                                                  int tp_part_idx,
                                                  int expert_id,
                                                  bool include_mins) {
  if (expert_id < 0 || expert_id >= config.expert_num) return 0;
  uint64_t total = 0;
  auto add_slot = [&](const std::vector<std::vector<ExpertFileSlot>>& slots, const char* name) {
    if (!slots.empty()) {
      const auto& slot = file_slot_at(config, tp_part_idx, slots, name, expert_id, true);
      if (slot.fd >= 0 && slot.size > 0) {
        total += static_cast<uint64_t>(slot.size);
      }
    }
  };
  add_slot(config.gate_file_slots, "gate.weight");
  add_slot(config.gate_scale_file_slots, "gate.scale");
  add_slot(config.up_file_slots, "up.weight");
  add_slot(config.up_scale_file_slots, "up.scale");
  add_slot(config.down_file_slots, "down.weight");
  add_slot(config.down_scale_file_slots, "down.scale");
  if (include_mins) {
    add_slot(config.gate_mins_file_slots, "gate.mins");
    add_slot(config.up_mins_file_slots, "up.mins");
    add_slot(config.down_mins_file_slots, "down.mins");
  }
  return total;
}

inline void append_amx_iouring_read_requests_for_expert(
    const GeneralMOEConfig& config,
    int tp_part_idx,
    int expert_id,
    void* gate_owner,
    void* up_owner,
    void* down_owner,
    size_t gate_weight_bytes,
    size_t gate_scale_bytes,
    size_t up_weight_bytes,
    size_t up_scale_bytes,
    size_t down_weight_bytes,
    size_t down_scale_bytes,
    bool include_mins,
    std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch) {
  if (read_batch == nullptr) return;
  const auto& gate_slot = file_slot_at(config, tp_part_idx, config.gate_file_slots, "gate.weight", expert_id, true);
  const auto& up_slot = file_slot_at(config, tp_part_idx, config.up_file_slots, "up.weight", expert_id, true);
  const auto& down_slot = file_slot_at(config, tp_part_idx, config.down_file_slots, "down.weight", expert_id, true);
  const auto& gate_scale_slot =
      file_slot_at(config, tp_part_idx, config.gate_scale_file_slots, "gate.scale", expert_id, true);
  const auto& up_scale_slot =
      file_slot_at(config, tp_part_idx, config.up_scale_file_slots, "up.scale", expert_id, true);
  const auto& down_scale_slot =
      file_slot_at(config, tp_part_idx, config.down_scale_file_slots, "down.scale", expert_id, true);

  if (gate_slot.fd < 0 || up_slot.fd < 0 || down_slot.fd < 0 || gate_scale_slot.fd < 0 ||
      up_scale_slot.fd < 0 || down_scale_slot.fd < 0) {
    std::ostringstream oss;
    oss << "AMX io_uring promotion found invalid fd layer=" << config.layer_idx << " tp=" << tp_part_idx
        << " expert=" << expert_id << " gate_fd=" << gate_slot.fd << " up_fd=" << up_slot.fd
        << " down_fd=" << down_slot.fd << " gate_scale_fd=" << gate_scale_slot.fd
        << " up_scale_fd=" << up_scale_slot.fd << " down_scale_fd=" << down_scale_slot.fd;
    throw std::runtime_error(oss.str());
  }

  auto queue_slot = [&](const ExpertFileSlot& slot, void* dst) {
    if (slot.fd >= 0 && slot.size > 0) {
      read_batch->push_back(
          ktransformers::AsyncExpertReader::ReadRequest{expert_id, slot.fd, dst, slot.size, slot.offset, 0});
    }
  };

  queue_slot(gate_slot, gate_owner);
  queue_slot(gate_scale_slot, reinterpret_cast<char*>(gate_owner) + gate_weight_bytes);
  queue_slot(up_slot, up_owner);
  queue_slot(up_scale_slot, reinterpret_cast<char*>(up_owner) + up_weight_bytes);
  queue_slot(down_slot, down_owner);
  queue_slot(down_scale_slot, reinterpret_cast<char*>(down_owner) + down_weight_bytes);

  if (include_mins) {
    if (!config.gate_mins_file_slots.empty()) {
      queue_slot(file_slot_at(config, tp_part_idx, config.gate_mins_file_slots, "gate.mins", expert_id, true),
                 reinterpret_cast<char*>(gate_owner) + gate_weight_bytes + gate_scale_bytes);
    }
    if (!config.up_mins_file_slots.empty()) {
      queue_slot(file_slot_at(config, tp_part_idx, config.up_mins_file_slots, "up.mins", expert_id, true),
                 reinterpret_cast<char*>(up_owner) + up_weight_bytes + up_scale_bytes);
    }
    if (!config.down_mins_file_slots.empty()) {
      queue_slot(file_slot_at(config, tp_part_idx, config.down_mins_file_slots, "down.mins", expert_id, true),
                 reinterpret_cast<char*>(down_owner) + down_weight_bytes + down_scale_bytes);
    }
  }
}

inline void append_bf16_iouring_read_requests_for_expert(
    const GeneralMOEConfig& config,
    int tp_part_idx,
    int expert_id,
    void* gate_raw,
    void* up_raw,
    void* down_full_raw,
    size_t gate_weight_bytes,
    size_t up_weight_bytes,
    size_t down_full_bytes,
    std::vector<ktransformers::AsyncExpertReader::ReadRequest>* read_batch) {
  if (read_batch == nullptr) return;
  const auto& gate_slot = file_slot_at(config, tp_part_idx, config.gate_file_slots, "gate.weight", expert_id, true);
  const auto& up_slot = file_slot_at(config, tp_part_idx, config.up_file_slots, "up.weight", expert_id, true);
  const auto& down_slot = file_slot_at(config, tp_part_idx, config.down_file_slots, "down.weight", expert_id, true);
  read_batch->push_back(ktransformers::AsyncExpertReader::ReadRequest{
      expert_id, gate_slot.fd, gate_raw, gate_weight_bytes, gate_slot.offset, 0});
  read_batch->push_back(ktransformers::AsyncExpertReader::ReadRequest{
      expert_id, up_slot.fd, up_raw, up_weight_bytes, up_slot.offset, 0});
  read_batch->push_back(ktransformers::AsyncExpertReader::ReadRequest{
      expert_id, down_slot.fd, down_full_raw, down_full_bytes, down_slot.offset, 0});
}

inline std::string amx_iouring_promotion_failure_message(
    const GeneralMOEConfig& config,
    int tp_part_idx,
    int expert_id,
    const std::vector<uint64_t>& read_requests,
    const ktransformers::AsyncExpertReader* async_reader,
    int timeout_ms) {
  const auto& gate_slot = file_slot_at(config, tp_part_idx, config.gate_file_slots, "gate.weight", expert_id, true);
  const auto& up_slot = file_slot_at(config, tp_part_idx, config.up_file_slots, "up.weight", expert_id, true);
  const auto& down_slot = file_slot_at(config, tp_part_idx, config.down_file_slots, "down.weight", expert_id, true);
  std::ostringstream oss;
  oss << "AMX io_uring promotion failed layer=" << config.layer_idx << " tp=" << tp_part_idx
      << " expert=" << expert_id << " logical=" << logical_expert_id_for_slot(config, tp_part_idx, expert_id)
      << " requests=" << read_requests.size() << " inflight=" << async_reader->get_inflight_count()
      << " timeout_ms=" << timeout_ms << " gate=(" << gate_slot.fd << "," << gate_slot.offset << ","
      << gate_slot.size << ") up=(" << up_slot.fd << "," << up_slot.offset << "," << up_slot.size
      << ") down=(" << down_slot.fd << "," << down_slot.offset << "," << down_slot.size
      << ") detail=" << async_reader->describe_requests(read_requests);
  return oss.str();
}

inline bool pending_prefetch_finished(const PendingPrefetch& pending,
                                      const ktransformers::AsyncExpertReader* async_reader,
                                      bool* failed) {
  if (failed != nullptr) *failed = false;
  if (pending.requests.empty() || async_reader == nullptr) return false;
  for (uint64_t request_id : pending.requests) {
    const int result = async_reader->get_request_result(request_id);
    if (result == std::numeric_limits<int>::min()) {
      return false;
    }
    if (!async_reader->request_succeeded(request_id)) {
      if (failed != nullptr) *failed = true;
      return false;
    }
  }
  return true;
}

inline std::string amx_iouring_prefetch_wait_failure_message(
    const GeneralMOEConfig& config,
    int tp_part_idx,
    int expert_id,
    const std::vector<uint64_t>& requests,
    const ktransformers::AsyncExpertReader* async_reader) {
  std::ostringstream oss;
  oss << "AMX io_uring prefetch wait failed layer=" << config.layer_idx << " tp=" << tp_part_idx
      << " expert=" << expert_id << " requests=" << requests.size()
      << " detail=" << async_reader->describe_requests(requests);
  return oss.str();
}

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_EXPERT_RESIDENCY_HPP
