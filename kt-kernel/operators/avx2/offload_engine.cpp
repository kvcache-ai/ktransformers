/**
 * @Description  : CPU Offloading Engine Implementation
 * @Author       : CPU Offloading Implementation
 * @Date         : 2026
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "offload_engine.hpp"

#include <algorithm>
#include <cstring>

namespace avx2 {

// Utility function for ceiling division
inline int div_up(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// FP8 Weight Matrix Decoding (internal helper)
// ============================================================================
static void decode_fp8_weight_matrix(
    const uint8_t* src, const float* scales,
    float* dst, size_t rows, size_t cols, int block_size) {
  // src is row-major [rows, cols] (as stored by BufferB::from_mat memcpy)
  // scales is block-wise [rows/block_size, cols/block_size]
  // dst is row-major [rows, cols] (FP32 dequantized with scales applied)
  ensure_fp8_lut_initialized();

  size_t n_block_rows = div_up(rows, block_size);
  size_t n_block_cols = div_up(cols, block_size);

  for (size_t br = 0; br < n_block_rows; ++br) {
    size_t row_start = br * block_size;
    size_t row_end = std::min(row_start + block_size, rows);
    size_t block_row_size = row_end - row_start;

    for (size_t r = 0; r < block_row_size; ++r) {
      size_t row = row_start + r;
      const uint8_t* src_row = src + row * cols;
      float* dst_row = dst + row * cols;

      for (size_t bc = 0; bc < n_block_cols; ++bc) {
        size_t col_start = bc * block_size;
        size_t col_end = std::min(col_start + block_size, cols);
        size_t block_col_size = col_end - col_start;

        // Scale is NOT applied here — gemm_fp8_with_predecoded applies it once
        // at sum += block_sum * scale. Applying it here too would double-scale.

        const uint8_t* src_block_row = src_row + col_start;
        float* dst_block_row = dst_row + col_start;

        size_t c = 0;
        for (; c + 8 <= block_col_size; c += 8) {
          __m256 dec = fp8x8_to_fp32x8(src_block_row + c);
          _mm256_storeu_ps(dst_block_row + c, dec);
        }
        for (; c < block_col_size; ++c) {
          dst_block_row[c] = fp8_to_fp32_scalar(src_block_row[c]);
        }
      }
    }
  }
}

// ============================================================================
// RAWINT4 Weight Matrix Decoding (internal helper)
// ============================================================================
static inline __m256 rawint4_dequant_8x4bit_unscaled(uint32_t packed_weight) {
  const __m256i shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  __m256i packed_v = _mm256_set1_epi32(packed_weight);
  __m256i nibbles = _mm256_and_si256(_mm256_srlv_epi32(packed_v, shifts), _mm256_set1_epi32(0xF));
  __m256 w = _mm256_cvtepi32_ps(nibbles);
  return _mm256_sub_ps(w, _mm256_set1_ps(8.0f));
}

static void decode_rawint4_weight_matrix(
    const uint8_t* src, const float* scales,
    float* dst, size_t rows, size_t cols, int k_group_size) {
  // src is packed INT4 [rows, cols/2] (row-major)
  // scales is [div_up(rows, k_group_size), div_up(cols, k_group_size)] (row-major block-wise, float32)
  // dst is row-major [rows, cols] (float32, dequantized with scales applied)

  size_t k_group_count = cols / k_group_size;
  size_t row_bytes = cols / 2;

  for (size_t r = 0; r < rows; ++r) {
    const uint8_t* src_row = src + r * row_bytes;
    // Scales are per K-group block, not per row. Rows within the same K-group
    // share the same scale row, so index by r / k_group_size.
    const float* scales_row = scales + (r / k_group_size) * k_group_count;
    float* dst_row = dst + r * cols;

    int c = 0;
    for (size_t g = 0; g < k_group_count; ++g) {
      float scale = scales_row[g];
      int k_base = g * k_group_size;
      int group_end = k_base + k_group_size;

      for (; c + 8 <= group_end; c += 8) {
        uint32_t packed;
        std::memcpy(&packed, src_row + c / 2, sizeof(uint32_t));
        __m256 dec = rawint4_dequant_8x4bit_unscaled(packed);
        // Apply scale during decode
        dec = _mm256_mul_ps(dec, _mm256_set1_ps(scale));
        _mm256_storeu_ps(dst_row + c, dec);
      }
      for (; c < group_end; ++c) {
        const uint8_t packed = src_row[c / 2];
        const int nibble = (c & 1) ? (packed >> 4) : (packed & 0x0F);
        dst_row[c] = (float)(nibble - 8) * scale;
      }
    }
  }
}

// ============================================================================
// OffloadEngine Implementation
// ============================================================================

OffloadEngine::OffloadEngine(const OffloadConfig& config) : config_(config) {}

OffloadEngine::~OffloadEngine() {
  shutdown();
}

void OffloadEngine::initialize() {
  if (running_.load()) return;

  // Allocate cache entries
  cache_.resize(config_.max_cached_experts);

  // Start prefetch thread
  running_.store(true);
  prefetch_thread_ = std::thread(&OffloadEngine::prefetch_loop, this);
}

void OffloadEngine::shutdown() {
  if (!running_.load()) return;

  running_.store(false);
  prefetch_cv_.notify_all();

  if (prefetch_thread_.joinable()) {
    prefetch_thread_.join();
  }

  // Clear cache
  std::lock_guard<std::mutex> lock(cache_mutex_);
  for (auto& entry : cache_) {
    entry.gate_predecoded.reset();
    entry.up_predecoded.reset();
    entry.down_predecoded.reset();
  }
  cache_.clear();
}

OffloadEngine::ExpertWeights OffloadEngine::get_expert_weights(size_t logical_expert_id) {
  std::unique_lock<std::mutex> lock(cache_mutex_);

  // Find or create entry for this expert
  ExpertCacheEntry& entry = get_or_create_entry(logical_expert_id);

  // Wait if currently loading
  while (entry.is_loading) {
    cache_cv_.wait(lock);
  }

  // If not predecoded yet, decode now
  if (!entry.is_predecoded) {
    entry.is_loading = true;
    lock.unlock();

    decode_expert_to_cache(logical_expert_id, entry);

    lock.lock();
    entry.is_loading = false;
    entry.is_predecoded = true;
    cache_cv_.notify_all();
  }

  // Update LRU
  touch_entry(entry);

  // Update stats
  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.cache_hits++;
  }

  // Return weights
  ExpertWeights weights;
  weights.gate_predecoded = entry.gate_predecoded.get();
  weights.up_predecoded = entry.up_predecoded.get();
  weights.down_predecoded = entry.down_predecoded.get();
  weights.gate_scale = entry.gate_scale_ptr;
  weights.up_scale = entry.up_scale_ptr;
  weights.down_scale = entry.down_scale_ptr;
  weights.expert_id = entry.expert_id;

  return weights;
}

void OffloadEngine::prefetch_expert(size_t logical_expert_id) {
  if (!config_.enable_prefetch) return;

  std::lock_guard<std::mutex> lock(prefetch_mutex_);
  prefetch_queue_.push(logical_expert_id);
  prefetch_cv_.notify_one();

  {
    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.prefetches++;
  }
}

void OffloadEngine::evict_if_needed() {
  // Find LRU entry that is not currently in use
  size_t oldest_idx = 0;
  int oldest_lru = INT_MAX;

  for (size_t i = 0; i < cache_.size(); ++i) {
    if (!cache_[i].is_loading && cache_[i].last_used < oldest_lru) {
      oldest_lru = cache_[i].last_used;
      oldest_idx = i;
    }
  }

  // Evict
  ExpertCacheEntry& entry = cache_[oldest_idx];
  if (entry.is_predecoded) {
    entry.gate_predecoded.reset();
    entry.up_predecoded.reset();
    entry.down_predecoded.reset();
    entry.is_predecoded = false;
    entry.expert_id = SIZE_MAX;
    entry.gate_scale_ptr = nullptr;
    entry.up_scale_ptr = nullptr;
    entry.down_scale_ptr = nullptr;

    std::lock_guard<std::mutex> stats_lock(stats_mutex_);
    stats_.evictions++;
  }
}

OffloadEngine::Stats OffloadEngine::get_stats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

void OffloadEngine::decode_expert_to_cache(size_t logical_expert_id, ExpertCacheEntry& entry) {
  const size_t hidden_size = config_.hidden_size;
  const size_t intermediate_size = config_.intermediate_size;
  const size_t block_size = config_.block_size;
  const size_t k_group_size = config_.k_group_size;

  // Determine format: RAWINT4 uses k_group_size (32), FP8 uses block_size (128)
  // RAWINT4 has packed weights (cols/2 elements per row)
  const bool is_rawint4 = (k_group_size > 0 && k_group_size != block_size);

  // Get physical expert index
  size_t physical_expert = logical_expert_id;
  if (config_.physical_to_logical_map) {
    physical_expert = config_.physical_to_logical_map[logical_expert_id];
  }

  // Calculate weight offsets for this expert
  // Weights are stored in separate arrays (not contiguous)
  const size_t gate_up_size = is_rawint4 ? intermediate_size * hidden_size / 2 : intermediate_size * hidden_size;
  const size_t down_size = is_rawint4 ? hidden_size * intermediate_size / 2 : hidden_size * intermediate_size;

  const uint8_t* gate_src = config_.gate_proj + physical_expert * gate_up_size;
  const uint8_t* up_src = config_.up_proj + physical_expert * gate_up_size;
  const uint8_t* down_src = config_.down_proj + physical_expert * down_size;

  // Scales
  const float* gate_scale_src;
  const float* up_scale_src;
  const float* down_scale_src;

  if (is_rawint4) {
    // RAWINT4: K-group scales [div_up(rows, k_group_size), div_up(cols, k_group_size)]
    const size_t gate_up_scale_elems = div_up(intermediate_size, k_group_size) * div_up(hidden_size, k_group_size);
    const size_t down_scale_elems = div_up(hidden_size, k_group_size) * div_up(intermediate_size, k_group_size);
    gate_scale_src = config_.gate_scale + physical_expert * gate_up_scale_elems;
    up_scale_src = config_.up_scale + physical_expert * gate_up_scale_elems;
    down_scale_src = config_.down_scale + physical_expert * down_scale_elems;
  } else {
    // FP8: block-wise scales [rows/block_size, cols/block_size]
    const size_t scale_blocks_per_expert = div_up(hidden_size, block_size) * div_up(intermediate_size, block_size);
    const size_t down_scale_blocks_per_expert = div_up(intermediate_size, block_size) * div_up(hidden_size, block_size);
    gate_scale_src = config_.gate_scale + physical_expert * scale_blocks_per_expert;
    up_scale_src = config_.up_scale + physical_expert * scale_blocks_per_expert;
    down_scale_src = config_.down_scale + physical_expert * down_scale_blocks_per_expert;
  }

  // Allocate pre-decoded buffers (always row-major float32 [rows, cols])
  entry.gate_predecoded = std::make_unique<float[]>(intermediate_size * hidden_size);
  entry.up_predecoded = std::make_unique<float[]>(intermediate_size * hidden_size);
  entry.down_predecoded = std::make_unique<float[]>(hidden_size * intermediate_size);

  if (is_rawint4) {
    // RAWINT4 decode
    decode_rawint4_weight_matrix(
        gate_src, gate_scale_src,
        entry.gate_predecoded.get(),
        intermediate_size, hidden_size, k_group_size
    );

    decode_rawint4_weight_matrix(
        up_src, up_scale_src,
        entry.up_predecoded.get(),
        intermediate_size, hidden_size, k_group_size
    );

    decode_rawint4_weight_matrix(
        down_src, down_scale_src,
        entry.down_predecoded.get(),
        hidden_size, intermediate_size, k_group_size
    );
  } else {
    // FP8 decode
    decode_fp8_weight_matrix(
        gate_src, gate_scale_src,
        entry.gate_predecoded.get(),
        intermediate_size, hidden_size, block_size
    );

    decode_fp8_weight_matrix(
        up_src, up_scale_src,
        entry.up_predecoded.get(),
        intermediate_size, hidden_size, block_size
    );

    decode_fp8_weight_matrix(
        down_src, down_scale_src,
        entry.down_predecoded.get(),
        hidden_size, intermediate_size, block_size
    );
  }

  // Set scale pointers
  entry.gate_scale_ptr = gate_scale_src;
  entry.up_scale_ptr = up_scale_src;
  entry.down_scale_ptr = down_scale_src;
}

void OffloadEngine::prefetch_loop() {
  while (running_.load()) {
    size_t expert_id = SIZE_MAX;

    {
      std::unique_lock<std::mutex> lock(prefetch_mutex_);
      prefetch_cv_.wait(lock, [this] { return !prefetch_queue_.empty() || !running_.load(); });

      if (!running_.load() && prefetch_queue_.empty()) break;

      if (!prefetch_queue_.empty()) {
        expert_id = prefetch_queue_.front();
        prefetch_queue_.pop();
      }
    }

    if (expert_id == SIZE_MAX) continue;

    // Check if already cached or loading
    {
      std::lock_guard<std::mutex> lock(cache_mutex_);
      bool found = false;
      for (auto& entry : cache_) {
        if (entry.expert_id == expert_id && (entry.is_predecoded || entry.is_loading)) {
          found = true;
          break;
        }
      }
      if (found) continue;
    }

    // Prefetch: decode expert weights
    std::unique_lock<std::mutex> lock(cache_mutex_);
    ExpertCacheEntry& entry = get_or_create_entry(expert_id);

    if (!entry.is_predecoded && !entry.is_loading) {
      entry.is_loading = true;
      lock.unlock();

      decode_expert_to_cache(expert_id, entry);

      lock.lock();
      entry.is_loading = false;
      entry.is_predecoded = true;
      cache_cv_.notify_all();
    }
  }
}

ExpertCacheEntry& OffloadEngine::get_or_create_entry(size_t logical_expert_id) {
  // First, check if already exists
  for (auto& entry : cache_) {
    if (entry.expert_id == logical_expert_id) {
      return entry;
    }
  }

  // Find empty slot
  for (auto& entry : cache_) {
    if (entry.expert_id == SIZE_MAX) {
      entry.expert_id = logical_expert_id;
      return entry;
    }
  }

  // Cache full, evict LRU
  evict_if_needed();

  // Find empty slot again (should have one now)
  for (auto& entry : cache_) {
    if (entry.expert_id == SIZE_MAX) {
      entry.expert_id = logical_expert_id;
      return entry;
    }
  }

  // Should not reach here
  return cache_[0];
}

void OffloadEngine::touch_entry(ExpertCacheEntry& entry) {
  entry.last_used = lru_counter_.fetch_add(1);
}

}  // namespace avx2