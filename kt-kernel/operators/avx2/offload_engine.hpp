/**
 * @Description  : CPU Offloading Engine for Large Model Inference
 * @Author       : CPU Offloading Implementation
 * @Date         : 2026
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Enables running models larger than CPU cache/RAM by streaming weights from
 * system RAM to CPU cache on-demand. Keeps original quantized weights in RAM,
 * pre-decodes only active experts, prefetches next expert weights during compute.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_OFFLOAD_ENGINE_H
#define CPUINFER_OPERATOR_AVX2_OFFLOAD_ENGINE_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "llama.cpp/ggml.h"
#include "fp8_dequant.hpp"

namespace avx2 {

// ============================================================================
// Offload Configuration
// ============================================================================
struct OffloadConfig {
  // Expert weight configuration
  size_t expert_num = 0;
  size_t hidden_size = 0;
  size_t intermediate_size = 0;
  int block_size = 128;        // For FP8 block-wise quantization
  int k_group_size = 32;       // For RAWINT4 K-group scales
  bool is_rawint4 = false;     // Set to true for RAWINT4 variant

  // Offloading policy
  size_t max_cached_experts = 4;      // Max experts to keep pre-decoded in cache
  bool enable_prefetch = true;        // Enable async prefetch of next expert
  size_t prefetch_distance = 1;       // How many experts ahead to prefetch

  // Thread pool for async operations
  void* worker_pool = nullptr;

  // Pointers to original quantized weights in RAM (not owned)
  const uint8_t* gate_proj = nullptr;
  const uint8_t* up_proj = nullptr;
  const uint8_t* down_proj = nullptr;
  const float* gate_scale = nullptr;
  const float* up_scale = nullptr;
  const float* down_scale = nullptr;
  const uint64_t* physical_to_logical_map = nullptr;
};

// ============================================================================
// Expert Weight Cache Entry
// ============================================================================
struct ExpertCacheEntry {
  size_t expert_id = SIZE_MAX;  // Logical expert ID
  int last_used = 0;            // LRU timestamp
  bool is_loading = false;
  bool is_predecoded = false;

  // Pre-decoded buffers (owned by cache)
  std::unique_ptr<float[]> gate_predecoded;   // [intermediate_size, hidden_size]
  std::unique_ptr<float[]> up_predecoded;     // [intermediate_size, hidden_size]
  std::unique_ptr<float[]> down_predecoded;   // [hidden_size, intermediate_size]

  // Scales are not pre-decoded (small, kept in RAM)
  const float* gate_scale_ptr = nullptr;
  const float* up_scale_ptr = nullptr;
  const float* down_scale_ptr = nullptr;

  ExpertCacheEntry() = default;
  ExpertCacheEntry(ExpertCacheEntry&&) = default;
  ExpertCacheEntry& operator=(ExpertCacheEntry&&) = default;
};

// ============================================================================
// CPU Offloading Engine
// ============================================================================
class OffloadEngine {
 public:
  explicit OffloadEngine(const OffloadConfig& config);
  ~OffloadEngine();

  // Initialize the engine (allocates cache, starts prefetch thread)
  void initialize();

  // Request expert weights for computation (blocks until available)
  // Returns pointers to pre-decoded weights and scales
  struct ExpertWeights {
    const float* gate_predecoded = nullptr;
    const float* up_predecoded = nullptr;
    const float* down_predecoded = nullptr;
    const float* gate_scale = nullptr;
    const float* up_scale = nullptr;
    const float* down_scale = nullptr;
    size_t expert_id = SIZE_MAX;
  };

  ExpertWeights get_expert_weights(size_t logical_expert_id);

  // Hint that expert will be needed soon (starts async prefetch)
  void prefetch_expert(size_t logical_expert_id);

  // Evict least recently used expert if cache is full
  void evict_if_needed();

  // Get cache statistics
  struct Stats {
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t evictions = 0;
    size_t prefetches = 0;
  };
  Stats get_stats() const;

  // Shutdown engine
  void shutdown();

 private:
  OffloadConfig config_;
  std::vector<ExpertCacheEntry> cache_;
  std::mutex cache_mutex_;
  std::condition_variable cache_cv_;

  // Prefetch thread
  std::thread prefetch_thread_;
  std::atomic<bool> running_{false};
  std::queue<size_t> prefetch_queue_;
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_;

  // LRU timestamp
  std::atomic<int> lru_counter_{0};

  // Statistics
  mutable std::mutex stats_mutex_;
  Stats stats_;

  // Helper: decode expert weights from quantized RAM to cache entry
  void decode_expert_to_cache(size_t logical_expert_id, ExpertCacheEntry& entry);

  // Prefetch thread main loop
  void prefetch_loop();

  // Get or create cache entry for expert
  ExpertCacheEntry& get_or_create_entry(size_t logical_expert_id);

  // Update LRU for entry
  void touch_entry(ExpertCacheEntry& entry);
};

// ============================================================================
// Scoped Expert Weight Access (RAII)
// ============================================================================
class ScopedExpertWeights {
 public:
  ScopedExpertWeights(OffloadEngine& engine, size_t logical_expert_id)
      : engine_(engine), weights_(engine.get_expert_weights(logical_expert_id)) {}

  ~ScopedExpertWeights() = default;

  const OffloadEngine::ExpertWeights& operator*() const { return weights_; }
  const OffloadEngine::ExpertWeights* operator->() const { return &weights_; }

  // Get raw pointers for GEMM
  const float* gate_predecoded() const { return weights_.gate_predecoded; }
  const float* up_predecoded() const { return weights_.up_predecoded; }
  const float* down_predecoded() const { return weights_.down_predecoded; }
  const float* gate_scale() const { return weights_.gate_scale; }
  const float* up_scale() const { return weights_.up_scale; }
  const float* down_scale() const { return weights_.down_scale; }

 private:
  OffloadEngine& engine_;
  OffloadEngine::ExpertWeights weights_;
};

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_OFFLOAD_ENGINE_H