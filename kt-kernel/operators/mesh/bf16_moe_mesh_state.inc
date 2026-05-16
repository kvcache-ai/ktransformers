// This file is included inside AMX_BF16_MOE_TP / TP_MOE class definitions.
// Keep declarations free of namespace wrappers and include guards.

 public:
  static constexpr bool kUsesLazyWeightPacking = true;

#ifndef _WIN32
 private:
  std::unique_ptr<void*[]> packed_gate_owner_;
  std::unique_ptr<void*[]> packed_up_owner_;
  std::unique_ptr<void*[]> packed_down_owner_;

  std::unique_ptr<std::atomic<uint8_t>[]> expert_states_;
  std::unique_ptr<std::atomic<uint32_t>[]> active_readers_;

  std::atomic<int> resident_expert_count_{0};
  std::atomic<int> eviction_cursor_{0};
  int numa_node_ = 0;
  int cache_capacity_ = 0;
  ResidentCachePolicyState resident_policy_;
  size_t gate_weight_bytes_ = 0;
  size_t up_weight_bytes_ = 0;
  size_t down_weight_bytes_ = 0;
  int full_intermediate_size_ = 0;

  bool iouring_enabled() const { return mesh::iouring_enabled(config_); }

  bool lazy_weight_enabled() const { return mesh::lazy_weight_enabled(config_); }

  size_t full_down_weight_bytes() const {
    return sizeof(ggml_bf16_t) * (size_t)config_.hidden_size * (size_t)full_intermediate_size_;
  }
#endif
