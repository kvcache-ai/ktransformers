#ifndef CPUINFER_OPERATOR_MESH_PREFILL_POLICY_HPP
#define CPUINFER_OPERATOR_MESH_PREFILL_POLICY_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <vector>

#include "resident_slot_pool.hpp"

#ifndef _WIN32

namespace mesh {

struct ResidentCapacityPlan {
  int configured_resident = 0;
  int requested_decode_resident = 0;
  int cache_capacity = 0;
  int prefill_static_capacity = 0;
};

inline int cpu_managed_expert_count(const GeneralMOEConfig& config) {
  int count = 0;
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (!config.should_skip_expert(expert_id)) count += 1;
  }
  return count;
}

inline int decode_cache_capacity(const GeneralMOEConfig& config, int cache_capacity) {
  const int configured = config.mesh_decode_resident_experts > 0 ? config.mesh_decode_resident_experts : cache_capacity;
  if (configured <= 0) return 0;
  return std::min(cpu_managed_expert_count(config), std::min(config.expert_num, std::max(configured, 0)));
}

inline ResidentCapacityPlan build_resident_capacity_plan(const GeneralMOEConfig& config) {
  ResidentCapacityPlan plan;
  plan.configured_resident =
      config.max_resident_experts > 0 ? config.max_resident_experts : config.max_tier0_experts;
  plan.requested_decode_resident =
      config.mesh_decode_resident_experts > 0 ? config.mesh_decode_resident_experts : plan.configured_resident;
  plan.cache_capacity =
      plan.configured_resident <= 0
          ? 0
          : std::min(config.expert_num, std::max(plan.configured_resident, config.num_experts_per_tok));
  if (config.mesh_prefill_layer_mode_enabled && plan.cache_capacity > 0) {
    const int cpu_capacity = cpu_managed_expert_count(config);
    const int requested_prefill_static =
        config.mesh_prefill_static_experts > 0 ? config.mesh_prefill_static_experts : plan.requested_decode_resident;
    plan.prefill_static_capacity =
        requested_prefill_static <= 0
            ? 0
            : std::min(cpu_capacity, std::max(requested_prefill_static, config.num_experts_per_tok));
    plan.cache_capacity = cpu_capacity;
  }
  return plan;
}

inline uint64_t expert_frequency_score(const std::vector<uint64_t>& frequency_scores, int expert_id) {
  if (expert_id < 0 || expert_id >= static_cast<int>(frequency_scores.size())) return 0;
  return frequency_scores[expert_id];
}

inline void ensure_frequency_storage(const GeneralMOEConfig& config,
                                     std::vector<uint64_t>& frequency_scores,
                                     uint64_t& frequency_total) {
  if (static_cast<int>(frequency_scores.size()) == config.expert_num) return;
  frequency_scores.assign(config.expert_num, 0);
  frequency_total = 0;
}

inline void record_prefill_frequency_counts(const GeneralMOEConfig& config,
                                            const std::vector<int>& local_counts,
                                            std::vector<uint64_t>& frequency_scores,
                                            uint64_t& frequency_total) {
  if (!config.mesh_prefill_layer_mode_enabled || config.expert_num <= 0) return;
  ensure_frequency_storage(config, frequency_scores, frequency_total);
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (config.should_skip_expert(expert_id)) continue;
    const int count = expert_id < static_cast<int>(local_counts.size()) ? local_counts[expert_id] : 0;
    if (count <= 0) continue;
    frequency_scores[expert_id] += static_cast<uint64_t>(count);
    frequency_total += static_cast<uint64_t>(count);
  }
}

inline void record_decode_frequency_counts(const GeneralMOEConfig& config,
                                           const std::vector<int>& expert_ids,
                                           std::vector<uint64_t>& frequency_scores,
                                           uint64_t& frequency_total) {
  if (!config.mesh_prefill_layer_mode_enabled || expert_ids.empty() || config.expert_num <= 0) return;
  ensure_frequency_storage(config, frequency_scores, frequency_total);
  for (int expert_id : expert_ids) {
    if (expert_id < 0 || expert_id >= config.expert_num || config.should_skip_expert(expert_id)) continue;
    frequency_scores[expert_id] += 1;
    frequency_total += 1;
  }
}

inline std::vector<int> cpu_managed_experts_by_frequency(const GeneralMOEConfig& config,
                                                         const std::vector<uint64_t>& frequency_scores) {
  std::vector<int> experts;
  experts.reserve(config.expert_num);
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (!config.should_skip_expert(expert_id)) {
      experts.push_back(expert_id);
    }
  }
  std::sort(experts.begin(), experts.end(), [&frequency_scores](int lhs, int rhs) {
    const uint64_t lh = expert_frequency_score(frequency_scores, lhs);
    const uint64_t rh = expert_frequency_score(frequency_scores, rhs);
    if (lh != rh) return lh > rh;
    return lhs < rhs;
  });
  return experts;
}

inline std::vector<int> cpu_managed_experts_by_heat(const GeneralMOEConfig& config,
                                                    const std::vector<float>& lookahead_heat) {
  std::vector<int> experts;
  experts.reserve(config.expert_num);
  for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
    if (!config.should_skip_expert(expert_id)) {
      experts.push_back(expert_id);
    }
  }
  std::sort(experts.begin(), experts.end(), [&lookahead_heat](int lhs, int rhs) {
    const float lh = lhs < static_cast<int>(lookahead_heat.size()) ? lookahead_heat[lhs] : 0.0f;
    const float rh = rhs < static_cast<int>(lookahead_heat.size()) ? lookahead_heat[rhs] : 0.0f;
    if (lh != rh) return lh > rh;
    return lhs < rhs;
  });
  return experts;
}

inline void rebuild_prefill_static_expert_set(const GeneralMOEConfig& config,
                                              int requested_capacity,
                                              const std::vector<uint64_t>* frequency_scores,
                                              int& static_slot_count,
                                              std::vector<uint8_t>& static_expert_mask,
                                              std::vector<int>& static_slot_for_expert,
                                              std::vector<int>& static_experts) {
  static_slot_count = 0;
  static_experts.clear();
  static_expert_mask.assign(config.expert_num, 0);
  static_slot_for_expert.assign(config.expert_num, -1);
  if (!config.mesh_prefill_layer_mode_enabled || requested_capacity <= 0) return;

  const int target = std::min(requested_capacity, cpu_managed_expert_count(config));
  std::vector<int> ranked;
  if (frequency_scores != nullptr) {
    ranked = cpu_managed_experts_by_frequency(config, *frequency_scores);
  } else {
    ranked.reserve(config.expert_num);
    for (int expert_id = 0; expert_id < config.expert_num; ++expert_id) {
      if (!config.should_skip_expert(expert_id)) {
        ranked.push_back(expert_id);
      }
    }
  }

  for (int expert_id : ranked) {
    if (static_slot_count >= target) break;
    static_expert_mask[expert_id] = 1;
    static_slot_for_expert[expert_id] = static_slot_count;
    static_experts.push_back(expert_id);
    static_slot_count += 1;
  }
}

inline bool is_eviction_protected(int expert_id, const std::vector<uint8_t>* protected_mask) {
  return protected_mask != nullptr && expert_id >= 0 && expert_id < static_cast<int>(protected_mask->size()) &&
         (*protected_mask)[expert_id] != 0;
}

inline void mark_protected_expert(const GeneralMOEConfig& config, std::vector<uint8_t>& mask, int expert_id) {
  if (expert_id >= 0 && expert_id < config.expert_num && !config.should_skip_expert(expert_id)) {
    mask[expert_id] = 1;
  }
}

inline std::vector<uint8_t> build_protected_mask(const GeneralMOEConfig& config,
                                                 const int64_t* expert_ids,
                                                 int count,
                                                 const int64_t* protect_ids = nullptr,
                                                 int protect_count = 0) {
  std::vector<uint8_t> mask(config.expert_num, 0);
  if (expert_ids != nullptr) {
    for (int i = 0; i < count; ++i) {
      mark_protected_expert(config, mask, static_cast<int>(expert_ids[i]));
    }
  }
  if (protect_ids != nullptr) {
    for (int i = 0; i < protect_count; ++i) {
      mark_protected_expert(config, mask, static_cast<int>(protect_ids[i]));
    }
  }
  return mask;
}

inline std::vector<uint8_t> build_protected_mask(const GeneralMOEConfig& config,
                                                 const std::vector<int>& expert_ids) {
  std::vector<uint8_t> mask(config.expert_num, 0);
  for (int expert_id : expert_ids) {
    mark_protected_expert(config, mask, expert_id);
  }
  return mask;
}

inline void collect_unique_cpu_experts(const GeneralMOEConfig& config,
                                       const int64_t* expert_ids,
                                       int count,
                                       std::vector<int>& active_experts,
                                       std::vector<int>* expert_counts = nullptr) {
  active_experts.clear();
  active_experts.reserve(std::max(0, count));
  std::vector<uint8_t> seen(config.expert_num, 0);
  if (expert_counts != nullptr) {
    expert_counts->assign(config.expert_num, 0);
  }
  for (int i = 0; i < count; ++i) {
    const int expert_id = static_cast<int>(expert_ids[i]);
    if (config.should_skip_expert(expert_id)) {
      continue;
    }
    if (expert_counts != nullptr) {
      ++(*expert_counts)[expert_id];
    }
    if (!seen[expert_id]) {
      seen[expert_id] = 1;
      active_experts.push_back(expert_id);
    }
  }
}

inline bool slot_has_rebindable_expert(const ResidentSlotPoolConstView& view,
                                       const GeneralMOEConfig& config,
                                       int slot,
                                       const std::vector<uint8_t>* protected_mask) {
  if (slot < 0 || slot >= view.cache_capacity || view.slot_states == nullptr || view.slot_to_expert == nullptr ||
      view.slot_active_readers == nullptr || view.expert_states == nullptr || view.active_readers == nullptr) {
    return false;
  }
  if (view.slot_active_readers[slot].load(std::memory_order_acquire) != 0) return false;
  const uint8_t slot_state = view.slot_states[slot].load(std::memory_order_acquire);
  const int expert_id = slot < static_cast<int>(view.slot_to_expert->size()) ? (*view.slot_to_expert)[slot] : -1;
  if (slot_state == SLOT_EMPTY && expert_id < 0) return true;
  if (slot_state != SLOT_READY) return false;
  if (expert_id < 0 || expert_id >= config.expert_num) return false;
  if (is_eviction_protected(expert_id, protected_mask)) return false;
  const uint8_t expert_state = view.expert_states[expert_id].load(std::memory_order_acquire);
  if (expert_state != EXPERT_CACHED && expert_state != EXPERT_PINNED) return false;
  return view.active_readers[expert_id].load(std::memory_order_acquire) == 0;
}

inline int find_frequency_main_replacement_slot(const ResidentSlotPoolConstView& view,
                                                const GeneralMOEConfig& config,
                                                int main_slots,
                                                const std::vector<uint64_t>& frequency_scores,
                                                int expert_id,
                                                const std::vector<uint8_t>* protected_mask) {
  if (main_slots <= 0 || view.slot_states == nullptr || view.slot_to_expert == nullptr) return -1;
  const uint64_t incoming_score = expert_frequency_score(frequency_scores, expert_id);

  for (int slot = 0; slot < main_slots; ++slot) {
    const int resident_expert =
        slot < static_cast<int>(view.slot_to_expert->size()) ? (*view.slot_to_expert)[slot] : -1;
    if (view.slot_states[slot].load(std::memory_order_acquire) == SLOT_EMPTY && resident_expert < 0) {
      return incoming_score > 0 ? slot : -1;
    }
  }

  int victim_slot = -1;
  uint64_t victim_score = std::numeric_limits<uint64_t>::max();
  for (int slot = 0; slot < main_slots; ++slot) {
    if (!slot_has_rebindable_expert(view, config, slot, protected_mask)) continue;
    const int resident_expert = (*view.slot_to_expert)[slot];
    if (resident_expert == expert_id) continue;
    const uint64_t score = expert_frequency_score(frequency_scores, resident_expert);
    if (score < victim_score) {
      victim_score = score;
      victim_slot = slot;
    }
  }
  if (victim_slot < 0) return -1;
  return incoming_score > victim_score ? victim_slot : -1;
}

inline int find_frequency_scratch_slot(const ResidentSlotPoolConstView& view,
                                       const GeneralMOEConfig& config,
                                       int scratch_begin,
                                       const std::vector<uint8_t>* protected_mask) {
  if (view.slot_states == nullptr || view.slot_to_expert == nullptr || scratch_begin >= view.cache_capacity) return -1;
  for (int slot = scratch_begin; slot < view.cache_capacity; ++slot) {
    const int resident_expert =
        slot < static_cast<int>(view.slot_to_expert->size()) ? (*view.slot_to_expert)[slot] : -1;
    if (view.slot_states[slot].load(std::memory_order_acquire) == SLOT_EMPTY && resident_expert < 0) {
      return slot;
    }
  }
  for (int slot = scratch_begin; slot < view.cache_capacity; ++slot) {
    if (slot_has_rebindable_expert(view, config, slot, protected_mask)) {
      return slot;
    }
  }
  return -1;
}

inline int find_main_slot_outside_frequency_set(const ResidentSlotPoolConstView& view,
                                                const GeneralMOEConfig& config,
                                                int main_slots,
                                                const std::vector<uint8_t>& desired_mask) {
  if (view.slot_states == nullptr || view.slot_to_expert == nullptr) return -1;
  for (int slot = 0; slot < main_slots; ++slot) {
    const int expert_id = slot < static_cast<int>(view.slot_to_expert->size()) ? (*view.slot_to_expert)[slot] : -1;
    if (view.slot_states[slot].load(std::memory_order_acquire) == SLOT_EMPTY && expert_id < 0) {
      return slot;
    }
    if (expert_id >= 0 && expert_id < static_cast<int>(desired_mask.size()) && desired_mask[expert_id] != 0) {
      continue;
    }
    if (slot_has_rebindable_expert(view, config, slot, nullptr)) {
      return slot;
    }
  }
  return -1;
}

inline void split_prefill_static_and_scratch_experts(const std::vector<int>& active_experts,
                                                     const std::vector<uint8_t>& static_expert_mask,
                                                     std::vector<int>& static_active_experts,
                                                     std::vector<int>& scratch_active_experts) {
  static_active_experts.clear();
  scratch_active_experts.clear();
  static_active_experts.reserve(active_experts.size());
  scratch_active_experts.reserve(active_experts.size());
  for (int expert_id : active_experts) {
    if (expert_id >= 0 && expert_id < static_cast<int>(static_expert_mask.size()) &&
        static_expert_mask[expert_id] != 0) {
      static_active_experts.push_back(expert_id);
    } else {
      scratch_active_experts.push_back(expert_id);
    }
  }
}

}  // namespace mesh

#endif  // _WIN32

#endif  // CPUINFER_OPERATOR_MESH_PREFILL_POLICY_HPP
