#ifndef CPUINFER_OPERATOR_MESH_RESIDENT_SLOT_POOL_HPP
#define CPUINFER_OPERATOR_MESH_RESIDENT_SLOT_POOL_HPP

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "expert_residency.hpp"

#ifndef _WIN32

namespace mesh {

struct ResidentSlotPoolConstView {
  int cache_capacity = 0;
  int expert_count = 0;
  const std::vector<void*>* gate_owner = nullptr;
  const std::vector<void*>* up_owner = nullptr;
  const std::vector<void*>* down_owner = nullptr;
  const std::vector<int>* slot_to_expert = nullptr;
  const std::vector<int>* expert_to_slot = nullptr;
  const std::atomic<uint8_t>* expert_states = nullptr;
  const std::atomic<uint32_t>* active_readers = nullptr;
  const std::atomic<uint8_t>* slot_states = nullptr;
  const std::atomic<uint32_t>* slot_active_readers = nullptr;
  const std::atomic<int>* resident_expert_count = nullptr;
  const bool* resident_slot_pool_allocated = nullptr;
};

struct ResidentSlotPoolView {
  int cache_capacity = 0;
  int expert_count = 0;
  std::vector<void*>* gate_owner = nullptr;
  std::vector<void*>* up_owner = nullptr;
  std::vector<void*>* down_owner = nullptr;
  std::vector<int>* slot_to_expert = nullptr;
  std::vector<int>* expert_to_slot = nullptr;
  std::atomic<uint8_t>* expert_states = nullptr;
  std::atomic<uint32_t>* active_readers = nullptr;
  std::atomic<uint8_t>* slot_states = nullptr;
  std::atomic<uint32_t>* slot_active_readers = nullptr;
  std::atomic<int>* resident_expert_count = nullptr;
  bool* resident_slot_pool_allocated = nullptr;
  MeshSlotMode* slot_mode = nullptr;
};

inline ResidentSlotPoolConstView const_view(const ResidentSlotPoolView& view) {
  return ResidentSlotPoolConstView{view.cache_capacity,
                                   view.expert_count,
                                   view.gate_owner,
                                   view.up_owner,
                                   view.down_owner,
                                   view.slot_to_expert,
                                   view.expert_to_slot,
                                   view.expert_states,
                                   view.active_readers,
                                   view.slot_states,
                                   view.slot_active_readers,
                                   view.resident_expert_count,
                                   view.resident_slot_pool_allocated};
}

inline void resize_resident_slot_storage(int cache_capacity,
                                         int expert_count,
                                         std::vector<void*>& gate_owner,
                                         std::vector<void*>& up_owner,
                                         std::vector<void*>& down_owner,
                                         std::vector<int>& slot_to_expert,
                                         std::vector<int>& expert_to_slot,
                                         std::unique_ptr<std::atomic<uint8_t>[]>& slot_states,
                                         std::unique_ptr<std::atomic<uint32_t>[]>& slot_active_readers) {
  if (cache_capacity > 0) {
    gate_owner.assign(cache_capacity, nullptr);
    up_owner.assign(cache_capacity, nullptr);
    down_owner.assign(cache_capacity, nullptr);
    slot_to_expert.assign(cache_capacity, -1);
    expert_to_slot.assign(expert_count, -1);
    slot_states = std::make_unique<std::atomic<uint8_t>[]>(cache_capacity);
    slot_active_readers = std::make_unique<std::atomic<uint32_t>[]>(cache_capacity);
    for (int slot = 0; slot < cache_capacity; ++slot) {
      slot_states[slot].store(SLOT_EMPTY, std::memory_order_relaxed);
      slot_active_readers[slot].store(0, std::memory_order_relaxed);
    }
    return;
  }

  gate_owner.clear();
  up_owner.clear();
  down_owner.clear();
  slot_to_expert.clear();
  expert_to_slot.assign(expert_count, -1);
  slot_states.reset();
  slot_active_readers.reset();
}

inline bool resident_slot_storage_configured(int cache_capacity,
                                             int expert_count,
                                             const std::vector<int>& slot_to_expert,
                                             const std::vector<int>& expert_to_slot) {
  return cache_capacity > 0 && static_cast<int>(slot_to_expert.size()) == cache_capacity &&
         static_cast<int>(expert_to_slot.size()) >= expert_count;
}

inline bool resident_slot_pool_configured(const ResidentSlotPoolConstView& view) {
  if (view.slot_to_expert == nullptr || view.expert_to_slot == nullptr) return false;
  return resident_slot_storage_configured(
      view.cache_capacity, view.expert_count, *view.slot_to_expert, *view.expert_to_slot);
}

inline bool resident_slot_pool_configured(const ResidentSlotPoolView& view) {
  return resident_slot_pool_configured(const_view(view));
}

inline bool resident_slot_pool_enabled(const ResidentSlotPoolConstView& view) {
  return resident_slot_pool_configured(view) && view.resident_slot_pool_allocated != nullptr &&
         *view.resident_slot_pool_allocated;
}

inline bool resident_slot_pool_enabled(const ResidentSlotPoolView& view) {
  return resident_slot_pool_enabled(const_view(view));
}

inline bool slot_has_buffers(int slot,
                             int cache_capacity,
                             const std::vector<void*>& gate_owner,
                             const std::vector<void*>& up_owner,
                             const std::vector<void*>& down_owner) {
  return slot >= 0 && slot < cache_capacity && slot < static_cast<int>(gate_owner.size()) &&
         slot < static_cast<int>(up_owner.size()) && slot < static_cast<int>(down_owner.size()) &&
         gate_owner[slot] != nullptr && up_owner[slot] != nullptr && down_owner[slot] != nullptr;
}

inline bool slot_has_buffers(const ResidentSlotPoolConstView& view, int slot) {
  if (view.gate_owner == nullptr || view.up_owner == nullptr || view.down_owner == nullptr) return false;
  return slot_has_buffers(slot, view.cache_capacity, *view.gate_owner, *view.up_owner, *view.down_owner);
}

inline bool slot_has_buffers(const ResidentSlotPoolView& view, int slot) {
  return slot_has_buffers(const_view(view), slot);
}

inline int allocated_slot_count(int cache_capacity,
                                const std::vector<void*>& gate_owner,
                                const std::vector<void*>& up_owner,
                                const std::vector<void*>& down_owner) {
  int count = 0;
  for (int slot = 0; slot < cache_capacity; ++slot) {
    if (slot_has_buffers(slot, cache_capacity, gate_owner, up_owner, down_owner)) {
      count += 1;
    }
  }
  return count;
}

inline int allocated_slot_count(const ResidentSlotPoolConstView& view) {
  if (view.gate_owner == nullptr || view.up_owner == nullptr || view.down_owner == nullptr) return 0;
  return allocated_slot_count(view.cache_capacity, *view.gate_owner, *view.up_owner, *view.down_owner);
}

inline int allocated_slot_count(const ResidentSlotPoolView& view) {
  return allocated_slot_count(const_view(view));
}

inline std::string slot_bind_error_message(const GeneralMOEConfig& config,
                                           int tp_part_idx,
                                           const ResidentSlotPoolConstView& view,
                                           int slot,
                                           int old_expert,
                                           int new_expert,
                                           const std::string& resident_summary) {
  const int slot_state =
      view.slot_states != nullptr && slot >= 0 && slot < view.cache_capacity
          ? static_cast<int>(view.slot_states[slot].load(std::memory_order_acquire))
          : -1;
  const int old_state =
      view.expert_states != nullptr && old_expert >= 0 && old_expert < view.expert_count
          ? static_cast<int>(view.expert_states[old_expert].load(std::memory_order_acquire))
          : -1;
  const int new_state =
      view.expert_states != nullptr && new_expert >= 0 && new_expert < view.expert_count
          ? static_cast<int>(view.expert_states[new_expert].load(std::memory_order_acquire))
          : -1;
  std::ostringstream oss;
  oss << "MESH slot bind attempted on an occupied slot layer=" << config.layer_idx
      << " tp=" << tp_part_idx << " slot=" << slot << " old_expert=" << old_expert
      << " new_expert=" << new_expert << " slot_state=" << slot_state
      << " old_state=" << old_state << " new_state=" << new_state << resident_summary;
  return oss.str();
}

inline int find_slot_by_owners(const ResidentSlotPoolConstView& view,
                               const void* gate_owner,
                               const void* up_owner,
                               const void* down_owner) {
  if (view.gate_owner == nullptr || view.up_owner == nullptr || view.down_owner == nullptr) return -1;
  for (int slot = 0; slot < view.cache_capacity; ++slot) {
    if (slot < static_cast<int>(view.gate_owner->size()) && slot < static_cast<int>(view.up_owner->size()) &&
        slot < static_cast<int>(view.down_owner->size()) && (*view.gate_owner)[slot] == gate_owner &&
        (*view.up_owner)[slot] == up_owner && (*view.down_owner)[slot] == down_owner) {
      return slot;
    }
  }
  return -1;
}

inline int find_empty_slot(const ResidentSlotPoolConstView& view, bool slot_pool_enabled) {
  if (!slot_pool_enabled || view.slot_states == nullptr || view.slot_to_expert == nullptr) return -1;
  for (int slot = 0; slot < view.cache_capacity; ++slot) {
    if (view.slot_states[slot].load(std::memory_order_acquire) == SLOT_EMPTY &&
        slot < static_cast<int>(view.slot_to_expert->size()) && (*view.slot_to_expert)[slot] < 0) {
      return slot;
    }
  }
  return -1;
}

inline int slot_for_expert_or_empty(const ResidentSlotPoolConstView& view,
                                    bool slot_pool_enabled,
                                    int expert_id) {
  if (!slot_pool_enabled || expert_id < 0 || view.expert_to_slot == nullptr ||
      expert_id >= static_cast<int>(view.expert_to_slot->size())) {
    return -1;
  }
  const int slot = (*view.expert_to_slot)[expert_id];
  if (slot >= 0 && slot < view.cache_capacity) return slot;
  return find_empty_slot(view, slot_pool_enabled);
}

inline void acquire_expert_read(const ResidentSlotPoolView& view, int expert_id) {
  if (expert_id < 0 || expert_id >= view.expert_count || view.active_readers == nullptr) return;
  view.active_readers[expert_id].fetch_add(1, std::memory_order_acq_rel);
  if (view.cache_capacity <= 0 || view.expert_to_slot == nullptr || view.slot_active_readers == nullptr ||
      expert_id >= static_cast<int>(view.expert_to_slot->size())) {
    return;
  }
  const int slot = (*view.expert_to_slot)[expert_id];
  if (slot >= 0 && slot < view.cache_capacity) {
    view.slot_active_readers[slot].fetch_add(1, std::memory_order_acq_rel);
  }
}

inline void release_expert_read(const ResidentSlotPoolView& view, int expert_id) {
  if (expert_id < 0 || expert_id >= view.expert_count || view.active_readers == nullptr) return;
  view.active_readers[expert_id].fetch_sub(1, std::memory_order_acq_rel);
  if (view.cache_capacity <= 0 || view.expert_to_slot == nullptr || view.slot_active_readers == nullptr ||
      expert_id >= static_cast<int>(view.expert_to_slot->size())) {
    return;
  }
  const int slot = (*view.expert_to_slot)[expert_id];
  if (slot >= 0 && slot < view.cache_capacity) {
    view.slot_active_readers[slot].fetch_sub(1, std::memory_order_acq_rel);
  }
}

inline uint64_t elapsed_us(std::chrono::steady_clock::time_point start,
                           std::chrono::steady_clock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

template <class Hooks>
bool release_resident_slot_pool(ResidentSlotPoolView view, Hooks& hooks, bool count_stats = false);

template <class Hooks>
bool allocate_resident_slot_pool(ResidentSlotPoolView view, Hooks& hooks, bool lazy_slot_buffers_enabled) {
  if (!resident_slot_pool_configured(view)) return false;
  if (view.resident_slot_pool_allocated != nullptr && *view.resident_slot_pool_allocated) return true;
  if (view.resident_expert_count != nullptr) {
    view.resident_expert_count->store(0, std::memory_order_release);
  }
  for (int slot = 0; slot < view.cache_capacity; ++slot) {
    if (!lazy_slot_buffers_enabled && !hooks.allocate_slot_buffers(slot)) {
      (void)release_resident_slot_pool(view, hooks, false);
      return false;
    }
    if (view.slot_to_expert != nullptr && slot < static_cast<int>(view.slot_to_expert->size())) {
      (*view.slot_to_expert)[slot] = -1;
    }
    if (view.slot_states != nullptr) {
      view.slot_states[slot].store(SLOT_EMPTY, std::memory_order_release);
    }
    if (view.slot_active_readers != nullptr) {
      view.slot_active_readers[slot].store(0, std::memory_order_release);
    }
  }
  if (view.resident_slot_pool_allocated != nullptr) {
    *view.resident_slot_pool_allocated = true;
  }
  return true;
}

template <class Hooks>
bool release_resident_slot_pool(ResidentSlotPoolView view, Hooks& hooks, bool count_stats) {
  if (!resident_slot_pool_configured(view)) return true;
  const int slot_count = view.gate_owner == nullptr ? view.cache_capacity : static_cast<int>(view.gate_owner->size());
  for (int slot = 0; slot < slot_count; ++slot) {
    const int old_expert =
        view.slot_to_expert != nullptr && slot < static_cast<int>(view.slot_to_expert->size())
            ? (*view.slot_to_expert)[slot]
            : -1;
    if (old_expert >= 0) {
      const uint8_t old_state =
          view.expert_states == nullptr ? static_cast<uint8_t>(EXPERT_BASELINE)
                                        : view.expert_states[old_expert].load(std::memory_order_acquire);
      if (old_state == EXPERT_PREFETCHING || old_state == EXPERT_PACKING || old_state == EXPERT_DEMOTING) {
        return false;
      }
      if ((view.active_readers != nullptr &&
           view.active_readers[old_expert].load(std::memory_order_acquire) != 0) ||
          (view.slot_active_readers != nullptr &&
           view.slot_active_readers[slot].load(std::memory_order_acquire) != 0)) {
        return false;
      }
      hooks.clear_packed_owners(old_expert);
      if (view.expert_to_slot != nullptr && old_expert < static_cast<int>(view.expert_to_slot->size())) {
        (*view.expert_to_slot)[old_expert] = -1;
      }
      hooks.apply_cold_ptrs(old_expert);
      if (count_stats) {
        hooks.note_expert_demote(old_expert);
      }
      if (view.expert_states != nullptr) {
        view.expert_states[old_expert].store(EXPERT_BASELINE, std::memory_order_release);
      }
    }
    hooks.release_slot_buffers(slot);
    if (view.slot_to_expert != nullptr && slot < static_cast<int>(view.slot_to_expert->size())) {
      (*view.slot_to_expert)[slot] = -1;
    }
    if (view.slot_states != nullptr) {
      view.slot_states[slot].store(SLOT_EMPTY, std::memory_order_release);
    }
  }
  if (view.expert_to_slot != nullptr) {
    for (int expert_id = 0; expert_id < static_cast<int>(view.expert_to_slot->size()); ++expert_id) {
      (*view.expert_to_slot)[expert_id] = -1;
    }
  }
  if (view.resident_expert_count != nullptr) {
    view.resident_expert_count->store(0, std::memory_order_release);
  }
  if (view.resident_slot_pool_allocated != nullptr) {
    *view.resident_slot_pool_allocated = false;
  }
  if (view.slot_mode != nullptr) {
    *view.slot_mode = MESH_SLOT_DECODE_CACHE;
  }
  return true;
}

template <class Hooks>
void bind_slot_to_expert(ResidentSlotPoolView view,
                         Hooks& hooks,
                         int slot,
                         int expert_id,
                         bool pin_after_copy,
                         bool slot_pool_enabled) {
  if (slot < 0 || slot >= view.cache_capacity || expert_id < 0 || expert_id >= view.expert_count ||
      view.slot_to_expert == nullptr || view.expert_to_slot == nullptr || view.gate_owner == nullptr ||
      view.up_owner == nullptr || view.down_owner == nullptr || view.slot_states == nullptr ||
      view.expert_states == nullptr) {
    throw std::runtime_error("Invalid MESH slot bind");
  }
  const int old_expert = (*view.slot_to_expert)[slot];
  if (old_expert >= 0 && old_expert != expert_id) {
    throw std::runtime_error(hooks.make_slot_bind_error(slot, old_expert, expert_id));
  }
  (*view.slot_to_expert)[slot] = expert_id;
  (*view.expert_to_slot)[expert_id] = slot;
  hooks.set_packed_owners(expert_id, (*view.gate_owner)[slot], (*view.up_owner)[slot], (*view.down_owner)[slot]);
  hooks.apply_owned_ptrs(expert_id, (*view.gate_owner)[slot], (*view.up_owner)[slot], (*view.down_owner)[slot]);
  const bool hard_pin = pin_after_copy && !slot_pool_enabled;
  view.slot_states[slot].store(SLOT_READY, std::memory_order_release);
  view.expert_states[expert_id].store(hard_pin ? EXPERT_PINNED : EXPERT_CACHED, std::memory_order_release);
  hooks.note_expert_insert(expert_id, hard_pin);
  hooks.drop_baseline_cache_for_expert(expert_id);
}

template <class Hooks>
bool unbind_slot(ResidentSlotPoolView view, Hooks& hooks, int slot, bool release_occupancy, bool count_stats = true) {
  if (slot < 0 || slot >= view.cache_capacity || view.slot_to_expert == nullptr) return false;
  const int old_expert = (*view.slot_to_expert)[slot];
  if (old_expert >= 0) {
    if ((view.active_readers != nullptr &&
         view.active_readers[old_expert].load(std::memory_order_acquire) != 0) ||
        (view.slot_active_readers != nullptr &&
         view.slot_active_readers[slot].load(std::memory_order_acquire) != 0)) {
      return false;
    }
    hooks.clear_packed_owners(old_expert);
    if (view.expert_to_slot != nullptr && old_expert < static_cast<int>(view.expert_to_slot->size())) {
      (*view.expert_to_slot)[old_expert] = -1;
    }
    hooks.apply_cold_ptrs(old_expert);
    if (count_stats) {
      hooks.note_expert_demote(old_expert);
    }
    if (view.expert_states != nullptr) {
      view.expert_states[old_expert].store(EXPERT_BASELINE, std::memory_order_release);
    }
    (*view.slot_to_expert)[slot] = -1;
  }
  if (release_occupancy && view.slot_states != nullptr && view.resident_expert_count != nullptr) {
    const uint8_t prev = view.slot_states[slot].exchange(SLOT_EMPTY, std::memory_order_acq_rel);
    if (prev != SLOT_EMPTY) {
      view.resident_expert_count->fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  return true;
}

template <class Hooks>
bool reserve_slot_for_loading(ResidentSlotPoolView view,
                              Hooks& hooks,
                              int slot,
                              int expert_id,
                              PrefillPromotionTiming* timing = nullptr) {
  if (slot < 0 || slot >= view.cache_capacity || expert_id < 0 || expert_id >= view.expert_count ||
      view.slot_states == nullptr || view.slot_to_expert == nullptr) {
    return false;
  }
  const uint8_t state = view.slot_states[slot].load(std::memory_order_acquire);
  const int old_expert = (*view.slot_to_expert)[slot];
  if (old_expert >= 0 && old_expert != expert_id) {
    if (state == SLOT_LOADING) {
      return false;
    }
    if (!unbind_slot(view, hooks, slot, state != SLOT_READY)) {
      return false;
    }
  } else if (state == SLOT_READY && !unbind_slot(view, hooks, slot, false)) {
    return false;
  } else if (state == SLOT_LOADING) {
    return false;
  }
  if (view.slot_states[slot].load(std::memory_order_acquire) == SLOT_EMPTY) {
    const bool had_buffers = slot_has_buffers(view, slot);
    const auto alloc_start = std::chrono::steady_clock::now();
    if (!hooks.allocate_slot_buffers(slot)) {
      return false;
    }
    const auto alloc_end = std::chrono::steady_clock::now();
    if (timing != nullptr) {
      timing->slot_alloc_us += elapsed_us(alloc_start, alloc_end);
      if (had_buffers) {
        timing->reused_slots += 1;
      } else {
        timing->allocated_slots += 1;
      }
    }
    if (view.resident_expert_count != nullptr) {
      view.resident_expert_count->fetch_add(1, std::memory_order_acq_rel);
    }
  }
  view.slot_states[slot].store(SLOT_LOADING, std::memory_order_release);
  return true;
}

template <class Hooks>
void release_empty_slot_buffers_to_limit(ResidentSlotPoolView view,
                                         Hooks& hooks,
                                         bool enabled,
                                         int allocated_limit) {
  if (!enabled) return;
  allocated_limit = std::max(0, std::min(allocated_limit, view.cache_capacity));
  int allocated = allocated_slot_count(view);
  if (allocated <= allocated_limit) return;
  for (int slot = view.cache_capacity - 1; slot >= 0 && allocated > allocated_limit; --slot) {
    if (view.slot_states == nullptr || view.slot_to_expert == nullptr) return;
    if (view.slot_states[slot].load(std::memory_order_acquire) != SLOT_EMPTY) continue;
    if (slot < static_cast<int>(view.slot_to_expert->size()) && (*view.slot_to_expert)[slot] >= 0) continue;
    if (!slot_has_buffers(view, slot)) continue;
    hooks.release_slot_buffers(slot);
    allocated -= 1;
  }
}

inline std::string resident_debug_summary(int expert_count,
                                          int cache_capacity,
                                          bool slot_pool_enabled,
                                          const std::atomic<uint8_t>* expert_states,
                                          const std::atomic<uint32_t>* active_readers,
                                          const std::atomic<uint8_t>* slot_states,
                                          const std::atomic<uint32_t>* slot_active_readers,
                                          const std::vector<int>& slot_to_expert) {
  std::ostringstream oss;
  int base = 0, packing = 0, cached = 0, pinned = 0, demoting = 0, prefetching = 0, other = 0;
  int readers = 0;
  for (int expert_id = 0; expert_id < expert_count; ++expert_id) {
    const uint8_t state =
        expert_states == nullptr ? static_cast<uint8_t>(EXPERT_BASELINE)
                                 : expert_states[expert_id].load(std::memory_order_acquire);
    switch (state) {
      case EXPERT_BASELINE:
        base++;
        break;
      case EXPERT_PACKING:
        packing++;
        break;
      case EXPERT_CACHED:
        cached++;
        break;
      case EXPERT_PINNED:
        pinned++;
        break;
      case EXPERT_DEMOTING:
        demoting++;
        break;
      case EXPERT_PREFETCHING:
        prefetching++;
        break;
      default:
        other++;
        break;
    }
    if (active_readers != nullptr && active_readers[expert_id].load(std::memory_order_acquire) != 0) {
      readers++;
    }
  }
  oss << " states={base:" << base << ",packing:" << packing << ",cached:" << cached << ",pinned:" << pinned
      << ",demoting:" << demoting << ",prefetching:" << prefetching << ",other:" << other
      << ",readers:" << readers << "}";

  if (slot_pool_enabled && slot_states != nullptr && slot_active_readers != nullptr) {
    int empty = 0, loading = 0, ready = 0, slot_readers = 0;
    oss << " slots=[";
    for (int slot = 0; slot < cache_capacity; ++slot) {
      const uint8_t slot_state = slot_states[slot].load(std::memory_order_acquire);
      if (slot_state == SLOT_EMPTY) {
        empty++;
      } else if (slot_state == SLOT_LOADING) {
        loading++;
      } else if (slot_state == SLOT_READY) {
        ready++;
      }
      if (slot_active_readers[slot].load(std::memory_order_acquire) != 0) {
        slot_readers++;
      }
      if (slot < 16) {
        if (slot > 0) oss << ",";
        const int expert_id = slot < static_cast<int>(slot_to_expert.size()) ? slot_to_expert[slot] : -1;
        const int expert_state =
            expert_id >= 0 && expert_id < expert_count && expert_states != nullptr
                ? static_cast<int>(expert_states[expert_id].load(std::memory_order_acquire))
                : -1;
        oss << slot << ":" << expert_id << ":s" << static_cast<int>(slot_state) << ":e" << expert_state
            << ":r" << slot_active_readers[slot].load(std::memory_order_acquire);
      }
    }
    oss << "] slot_counts={empty:" << empty << ",loading:" << loading << ",ready:" << ready
        << ",readers:" << slot_readers << "}";
  }
  return oss.str();
}

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_RESIDENT_SLOT_POOL_HPP
