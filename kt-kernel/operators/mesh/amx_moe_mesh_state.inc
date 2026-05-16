// This file is included inside AMX_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

#ifndef _WIN32
  std::vector<void*> baseline_gate_weight_src_;
  std::vector<void*> baseline_up_weight_src_;
  std::vector<void*> baseline_down_weight_src_;
  std::vector<float*> baseline_gate_scale_src_;
  std::vector<float*> baseline_up_scale_src_;
  std::vector<float*> baseline_down_scale_src_;
  std::vector<float*> baseline_gate_mins_src_;
  std::vector<float*> baseline_up_mins_src_;
  std::vector<float*> baseline_down_mins_src_;

  std::vector<void*> packed_gate_owner_;
  std::vector<void*> packed_up_owner_;
  std::vector<void*> packed_down_owner_;
  std::vector<void*> slot_gate_owner_;
  std::vector<void*> slot_up_owner_;
  std::vector<void*> slot_down_owner_;
  std::vector<int> slot_to_expert_;
  std::vector<int> expert_to_slot_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;
  std::unique_ptr<std::atomic<uint8_t>[]> slot_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> slot_active_readers_;
  std::vector<PendingPrefetch> pending_prefetches_;

  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  int prefill_static_capacity_ = 0;
  int prefill_static_slot_count_ = 0;
  bool resident_slot_pool_allocated_ = false;
  MeshSlotMode mesh_slot_mode_ = MESH_SLOT_DECODE_CACHE;
  uint64_t memory_guard_tick_ = 0;
  std::string cgroup_memory_current_path_;
  std::string cgroup_memory_max_path_;
  ResidentCachePolicyState resident_policy_;
  const std::vector<uint8_t>* eviction_protected_mask_ = nullptr;
  bool prefill_streaming_active_ = false;
  std::vector<uint8_t> prefill_static_expert_mask_;
  std::vector<int> prefill_static_slot_for_expert_;
  std::vector<int> prefill_static_experts_;
  std::vector<uint64_t> expert_frequency_score_;
  uint64_t expert_frequency_total_ = 0;
  PrefillPromotionTiming last_prefill_static_timing_;
  size_t gate_total_bytes_ = 0;
  size_t up_total_bytes_ = 0;
  size_t down_total_bytes_ = 0;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  size_t gate_scale_bytes_ = 0;
  size_t up_scale_bytes_ = 0;
  size_t down_scale_bytes_ = 0;
  size_t gate_mins_bytes_ = 0;
  size_t up_mins_bytes_ = 0;
  size_t down_mins_bytes_ = 0;
  std::vector<float> lookahead_heat_;
#endif
