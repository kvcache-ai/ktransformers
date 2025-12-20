/**
 * @Description  : FP8 AMX MoE operator for DeepSeek V3.2 native inference
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements FP8 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * FP8 weights are stored with 128x128 block-wise scales.
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP8_MOE_H
#define CPUINFER_OPERATOR_AMX_FP8_MOE_H

// #define DEBUG_FP8_MOE

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "la/amx_raw_buffers.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/amx_raw_utils.hpp"
#include "moe_base.hpp"

/**
 * @brief FP8 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to GemmKernel224FP8
 *
 * This class provides FP8-specific implementations:
 * - do_gate_up_gemm, do_down_gemm : FP8 weight -> BF16 conversion mat mul
 * - load_weights: Load FP8 weights with 128x128 block scales
 */
template <class T = amx::GemmKernel224FP8>
class AMX_FP8_MOE_TP : public AMX_MOE_BASE<T, AMX_FP8_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP8_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP8_MOE_TP() = default;

  AMX_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("KT-Kernel fp8 MoE only support block-wise FP8. group_size = %d, zero_point = %d",
                               quant_config.group_size, quant_config.zero_point);
    }
    printf("Created AMX_FP8_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_FP8_MOE_TP() = default;
  // ============================================================================
  // CRTP buffer creation - with group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
  }
  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                        down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
  }

#ifdef DEBUG_FP8_MOE
  // Function to dump Buffer B data for debugging FP8 quantization results
  inline void dump_buffer_b(const std::string& quantization_type, int expert_idx, const std::string& matrix_type,
                            typename T::BufferB* buffer) {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;

    printf("[DUMP_BUFFER_B] TP%d %s Expert%d %s:\n", tp_part_idx, quantization_type.c_str(), expert_idx,
           matrix_type.c_str());

    // Calculate dimensions based on matrix type
    int rows, cols;
    size_t scale_elem_count;
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
    }
    int n_blocks_n = (rows + group_size - 1) / group_size;
    int n_blocks_k = (cols + group_size - 1) / group_size;
    scale_elem_count = n_blocks_n * n_blocks_k;

    // Dump scales (as BF16 converted to float)
    printf("  Scales[first 16]: ");
    for (int i = 0; i < std::min(16, (int)scale_elem_count); i++) {
      printf("%.6f ", buffer->d[i]);
    }
    printf("\n");

    if (scale_elem_count > 16) {
      printf("  Scales[last 16]: ");
      int start_idx = std::max(0, (int)scale_elem_count - 16);
      for (int i = start_idx; i < (int)scale_elem_count; i++) {
        printf("%.6f ", buffer->d[i]);
      }
      printf("\n");
    }

    // Dump FP8 weights (as hex uint8)
    size_t weight_size = (size_t)rows * cols;  // FP8 is 1 byte per element
    uint8_t* weight_ptr = (uint8_t*)buffer->b;

    printf("  FP8 Weights[first 32 bytes]: ");
    for (int i = 0; i < std::min(32, (int)weight_size); i++) {
      printf("%02x ", weight_ptr[i]);
    }
    printf("\n");

    if (weight_size > 32) {
      printf("  FP8 Weights[last 32 bytes]: ");
      int start_idx = std::max(32, (int)weight_size - 32);
      for (int i = start_idx; i < (int)weight_size; i++) {
        printf("%02x ", weight_ptr[i]);
      }
      printf("\n");
    }

    printf("  Matrix dimensions: %dx%d (n x k), Scale blocks: %dx%d, Group size: %d, Scale elements: %zu\n", rows, cols,
           n_blocks_n, n_blocks_k, group_size, scale_elem_count);
  }
#endif

  /**
   * @brief Load FP8 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj with scales
   * from config_.gate_scale, up_scale, down_scale.
   */
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("FP8 AVX MOE only support native weight.");
    }

    // load weight
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // gate part
          gate_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.gate_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              (float*)config_.gate_scale +
                  (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
              ith, nth);
          // up part
          up_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.up_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              (float*)config_.up_scale +
                  (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // down part
          down_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.down_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              (float*)config_.down_scale +
                  (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
              ith, nth);
        },
        nullptr);
#ifdef DEBUG_FP8_MOE
    dump_buffer_b("Native FP8", 0, "gate", gate_bb_[0].get());
    dump_buffer_b("Native FP8", 0, "down", down_bb_[0].get());
#endif
  }

  // Fast 64-byte (512-bit) memcpy using AVX512
  static inline void fast_memcpy_64(void* __restrict dst, const void* __restrict src) {
    __m512i data = _mm512_loadu_si512(src);
    _mm512_storeu_si512(dst, data);
  }

  // Fast memcpy for arbitrary sizes using AVX512
  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      fast_memcpy_64(d, s);
      d += 64;
      s += 64;
    }
    bytes -= chunks * 64;
    if (bytes > 0) {
      std::memcpy(d, s, bytes);
    }
  }

  /**
   * @brief Unpack a single N_STEP x K_STEP block from packed BufferB format to n-major format
   *
   * This is the inverse of the packing done in BufferBFP8Impl::from_mat.
   * Optimized with AVX512 gather for efficient non-contiguous reads.
   *
   * @param src Pointer to packed data (N_STEP * K_STEP bytes in packed layout)
   * @param dst Pointer to destination in n-major layout
   * @param dst_row_stride Row stride in destination buffer (number of columns in full matrix)
   */
  static inline void unpack_nk_block(const uint8_t* src, uint8_t* dst, size_t dst_row_stride) {
    // row_map[packed_i] gives the base row for packed index packed_i
    static constexpr int row_map[8] = {0, 16, 4, 20, 8, 24, 12, 28};
    const uint64_t* src64 = reinterpret_cast<const uint64_t*>(src);

    // Gather indices: src64[8*j + packed_i] for j = 0..7
    // Offsets in uint64 units: 0, 8, 16, 24, 32, 40, 48, 56 (+ packed_i for each group)
    const __m512i gather_offsets = _mm512_set_epi64(56, 48, 40, 32, 24, 16, 8, 0);

    // Process each packed group (8 groups of 4 rows each = 32 rows total)
    for (int packed_i = 0; packed_i < 8; packed_i++) {
      const int base_row = row_map[packed_i];
      const uint64_t* base_src = src64 + packed_i;

      // Gather 8 values for j=0..7 and j=8..15
      __m512i vals_0_7 = _mm512_i64gather_epi64(gather_offsets, base_src, 8);
      __m512i vals_8_15 = _mm512_i64gather_epi64(gather_offsets, base_src + 64, 8);

      // Extract 4 rows from each set of 8 values
      // Row 0: bits 0-15
      __m128i row0_lo = _mm512_cvtepi64_epi16(_mm512_and_si512(vals_0_7, _mm512_set1_epi64(0xFFFF)));
      __m128i row0_hi = _mm512_cvtepi64_epi16(_mm512_and_si512(vals_8_15, _mm512_set1_epi64(0xFFFF)));
      // Row 1: bits 16-31
      __m128i row1_lo =
          _mm512_cvtepi64_epi16(_mm512_and_si512(_mm512_srli_epi64(vals_0_7, 16), _mm512_set1_epi64(0xFFFF)));
      __m128i row1_hi =
          _mm512_cvtepi64_epi16(_mm512_and_si512(_mm512_srli_epi64(vals_8_15, 16), _mm512_set1_epi64(0xFFFF)));
      // Row 2: bits 32-47
      __m128i row2_lo =
          _mm512_cvtepi64_epi16(_mm512_and_si512(_mm512_srli_epi64(vals_0_7, 32), _mm512_set1_epi64(0xFFFF)));
      __m128i row2_hi =
          _mm512_cvtepi64_epi16(_mm512_and_si512(_mm512_srli_epi64(vals_8_15, 32), _mm512_set1_epi64(0xFFFF)));
      // Row 3: bits 48-63
      __m128i row3_lo = _mm512_cvtepi64_epi16(_mm512_srli_epi64(vals_0_7, 48));
      __m128i row3_hi = _mm512_cvtepi64_epi16(_mm512_srli_epi64(vals_8_15, 48));

      // Store 32 bytes (16 x uint16) to each row
      // Combine two 128-bit values into 256-bit for more efficient stores
      uint8_t* row0_dst = dst + (size_t)base_row * dst_row_stride;
      uint8_t* row1_dst = dst + (size_t)(base_row + 1) * dst_row_stride;
      uint8_t* row2_dst = dst + (size_t)(base_row + 2) * dst_row_stride;
      uint8_t* row3_dst = dst + (size_t)(base_row + 3) * dst_row_stride;

      // Combine lo and hi into 256-bit and store
      __m256i row0_256 = _mm256_set_m128i(row0_hi, row0_lo);
      __m256i row1_256 = _mm256_set_m128i(row1_hi, row1_lo);
      __m256i row2_256 = _mm256_set_m128i(row2_hi, row2_lo);
      __m256i row3_256 = _mm256_set_m128i(row3_hi, row3_lo);

      _mm256_storeu_si256((__m256i*)row0_dst, row0_256);
      _mm256_storeu_si256((__m256i*)row1_dst, row1_256);
      _mm256_storeu_si256((__m256i*)row2_dst, row2_256);
      _mm256_storeu_si256((__m256i*)row3_dst, row3_256);
    }
  }

  /**
   * @brief Unpack 4 consecutive N_STEP x K_STEP blocks to maximize cache line utilization
   *
   * Processing 4 blocks together means each row write is 128 bytes = 2 cache lines,
   * which greatly improves write efficiency compared to 32 bytes per row.
   *
   * @param src Array of 4 source pointers (each pointing to a 32x32 packed block)
   * @param dst Destination pointer in n-major layout
   * @param dst_row_stride Row stride in destination buffer
   */
  static inline void unpack_4nk_blocks(const uint8_t* src[4], uint8_t* dst, size_t dst_row_stride) {
    static constexpr int row_map[8] = {0, 16, 4, 20, 8, 24, 12, 28};
    constexpr int K_STEP = T::K_STEP;  // 32

    // Reinterpret as uint64 arrays for efficient access
    const uint64_t* src0 = reinterpret_cast<const uint64_t*>(src[0]);
    const uint64_t* src1 = reinterpret_cast<const uint64_t*>(src[1]);
    const uint64_t* src2 = reinterpret_cast<const uint64_t*>(src[2]);
    const uint64_t* src3 = reinterpret_cast<const uint64_t*>(src[3]);

    // Process all 32 rows, writing 128 bytes (4 x 32) per row
    for (int packed_i = 0; packed_i < 8; packed_i++) {
      const int base_row = row_map[packed_i];

      // Process 4 rows at a time
      for (int r = 0; r < 4; r++) {
        uint16_t* row_dst = reinterpret_cast<uint16_t*>(dst + (size_t)(base_row + r) * dst_row_stride);
        const int shift = r * 16;

        // Unroll: process all 4 blocks x 16 columns = 64 uint16 values
        // Block 0: columns 0-15
        for (int j = 0; j < 16; j++) {
          row_dst[j] = static_cast<uint16_t>(src0[8 * j + packed_i] >> shift);
        }
        // Block 1: columns 16-31
        for (int j = 0; j < 16; j++) {
          row_dst[16 + j] = static_cast<uint16_t>(src1[8 * j + packed_i] >> shift);
        }
        // Block 2: columns 32-47
        for (int j = 0; j < 16; j++) {
          row_dst[32 + j] = static_cast<uint16_t>(src2[8 * j + packed_i] >> shift);
        }
        // Block 3: columns 48-63
        for (int j = 0; j < 16; j++) {
          row_dst[48 + j] = static_cast<uint16_t>(src3[8 * j + packed_i] >> shift);
        }
      }
    }
  }

  /**
   * @brief Reconstruct weights for a single expert to the output buffers (no temp buffer version)
   *
   * Directly unpacks from packed BufferB format to n-major GPU buffers without intermediate storage.
   * Optimized version with coarse-grained task splitting for better cache utilization.
   *
   * Key optimizations:
   * - Reduced task count (~40 vs ~350) to minimize scheduling overhead
   * - Larger chunks per task for better cache line utilization
   * - Process multiple N_STEPs per task for better write locality
   *
   * @param gpu_tp_count Number of GPU TP parts (1, 2, 4, or 8)
   * @param cpu_tp_count Number of CPU TP parts
   * @param expert_id Expert index to process
   * @param full_config Full configuration (before CPU TP split)
   * @param w13_weight_ptrs Pointers to gate+up weight buffers (one per GPU TP)
   * @param w13_scale_ptrs Pointers to gate+up scale buffers (one per GPU TP)
   * @param w2_weight_ptrs Pointers to down weight buffers (one per GPU TP)
   * @param w2_scale_ptrs Pointers to down scale buffers (one per GPU TP)
   */
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    const int group_size = config.quant_config.group_size;
    auto pool = config.pool->get_subpool(tp_part_idx);

    constexpr int N_STEP = T::N_STEP;
    constexpr int K_STEP = T::K_STEP;
    constexpr int N_BLOCK = T::N_BLOCK;
    constexpr int K_BLOCK = T::K_BLOCK;

    // ========= W13 (gate+up): Shape [intermediate, hidden], split by N only =========
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;

    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;
    const size_t gpu_w13_scale_per_mat = (size_t)div_up(gpu_n_w13, group_size) * div_up(gpu_k_w13, group_size);
    const int cpu_scale_k_blocks_w13 = div_up(cpu_k_w13, group_size);
    const int gpu_scale_k_blocks_w13 = div_up(gpu_k_w13, group_size);

    // ========= W2 (down): Shape [hidden, intermediate], split by K =========
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_n_w2 = full_config.hidden_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;

    const size_t gpu_w2_weight_per_mat = (size_t)gpu_n_w2 * gpu_k_w2;
    const size_t gpu_w2_scale_per_mat = (size_t)div_up(gpu_n_w2, group_size) * div_up(gpu_k_w2, group_size);
    const int cpu_scale_k_blocks_w2 = div_up(cpu_k_w2, group_size);
    const int gpu_scale_k_blocks_w2 = div_up(gpu_k_w2, group_size);

    // ========= Scale dimensions =========
    const int cpu_scale_n_blocks_w13 = div_up(cpu_n_w13, group_size);
    const int gpu_scale_n_blocks_w13 = div_up(gpu_n_w13, group_size);
    const int cpu_scale_n_blocks_w2 = div_up(cpu_n_w2, group_size);

    // ========= Optimized job layout =========
    // Use task count slightly above CPU core count for good work stealing
    // For 80-core system, ~100 tasks provides good balance
    constexpr int NUM_W13_TASKS = 32;  // Per matrix (gate or up), total 64 for w13
    constexpr int NUM_W2_TASKS = 32;   // For down matrix
    constexpr int SCALE_TASKS = 3;     // gate_scale, up_scale, down_scale

    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS + SCALE_TASKS;

    // Calculate N_STEP blocks per task (must be N_STEP aligned for correct BufferB addressing)
    const int w13_n_steps = div_up(cpu_n_w13, N_STEP);
    const int w13_steps_per_task = div_up(w13_n_steps, NUM_W13_TASKS);
    const int w2_n_steps = div_up(cpu_n_w2, N_STEP);
    const int w2_steps_per_task = div_up(w2_n_steps, NUM_W2_TASKS);

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // ========= W13 weight task: process chunk of rows x full K =========
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            // Calculate row range for this task (N_STEP aligned)
            const int step_start = chunk_idx * w13_steps_per_task;
            const int step_end = std::min(step_start + w13_steps_per_task, w13_n_steps);
            if (step_start >= w13_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w13);

            // Process each N_STEP within this chunk
            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              // Calculate GPU target and offset for each N_STEP (may cross GPU TP boundaries)
              const int global_n = global_n_offset_w13 + local_n_start;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              uint8_t* weight_base = (uint8_t*)w13_weight_ptrs[target_gpu];
              // Pointer already points to current expert's location, only add offset for up matrix
              const size_t expert_weight_off = is_up ? gpu_w13_weight_per_mat : 0;

              // Calculate N_BLOCK info for source addressing
              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w13 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              // Process all K in groups of 4 K_STEPs when possible for cache efficiency
              for (int k_block_begin = 0; k_block_begin < cpu_k_w13; k_block_begin += K_BLOCK) {
                const int k_block_size = std::min(K_BLOCK, cpu_k_w13 - k_block_begin);

                // Try to process 4 K_STEPs at once (128 columns = 2 cache lines per row)
                int k_begin = 0;
                for (; k_begin + 4 * K_STEP <= k_block_size; k_begin += 4 * K_STEP) {
                  const uint8_t* src_ptrs[4];
                  for (int i = 0; i < 4; i++) {
                    src_ptrs[i] = bb->b + (size_t)n_block_begin * cpu_k_w13 + (size_t)k_block_begin * n_block_size +
                                  (size_t)n_in_block * k_block_size + (size_t)(k_begin + i * K_STEP) * N_STEP;
                  }
                  uint8_t* dst =
                      weight_base + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13 + k_block_begin + k_begin;
                  unpack_4nk_blocks(src_ptrs, dst, gpu_k_w13);
                }

                // Handle remaining K_STEPs one by one
                for (; k_begin < k_block_size; k_begin += K_STEP) {
                  const uint8_t* src = bb->b + (size_t)n_block_begin * cpu_k_w13 +
                                       (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                       (size_t)k_begin * N_STEP;
                  uint8_t* dst =
                      weight_base + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13 + k_block_begin + k_begin;
                  unpack_nk_block(src, dst, gpu_k_w13);
                }
              }
            }

          } else if (task_id < NUM_W13_TASKS * 2 + NUM_W2_TASKS) {
            // ========= W2 weight task: process chunk of rows x all K slices =========
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            // Calculate row range for this task (N_STEP aligned)
            const int step_start = chunk_idx * w2_steps_per_task;
            const int step_end = std::min(step_start + w2_steps_per_task, w2_n_steps);
            if (step_start >= w2_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w2);

            // Process each N_STEP within this chunk
            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              // Calculate N_BLOCK info for source addressing
              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w2 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              // Process all K slices (each slice goes to a different GPU TP)
              for (int k_slice_start = 0; k_slice_start < cpu_k_w2; k_slice_start += gpu_k_w2) {
                const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

                const int global_k_start = global_k_offset_w2 + k_slice_start;
                const int target_gpu = global_k_start / gpu_k_w2;
                const int k_in_gpu_base = global_k_start % gpu_k_w2;

                uint8_t* weight_base = (uint8_t*)w2_weight_ptrs[target_gpu];
                // Pointer already points to current expert's location
                const size_t expert_weight_off = 0;

                // Process K within this slice, trying 4 K_STEPs at once when aligned
                for (int k_abs = k_slice_start; k_abs < k_slice_end;) {
                  const int k_block_idx = k_abs / K_BLOCK;
                  const int k_block_begin = k_block_idx * K_BLOCK;
                  const int k_block_size = std::min(K_BLOCK, cpu_k_w2 - k_block_begin);
                  const int k_in_block = k_abs - k_block_begin;
                  const int k_in_gpu = k_in_gpu_base + (k_abs - k_slice_start);

                  // Check if we can process 4 K_STEPs at once
                  const int remaining_in_block = k_block_size - k_in_block;
                  const int remaining_in_slice = k_slice_end - k_abs;

                  if (remaining_in_block >= 4 * K_STEP && remaining_in_slice >= 4 * K_STEP) {
                    const uint8_t* src_ptrs[4];
                    for (int i = 0; i < 4; i++) {
                      src_ptrs[i] = bb->b + (size_t)n_block_begin * cpu_k_w2 + (size_t)k_block_begin * n_block_size +
                                    (size_t)n_in_block * k_block_size + (size_t)(k_in_block + i * K_STEP) * N_STEP;
                    }
                    uint8_t* dst = weight_base + expert_weight_off + (size_t)local_n_start * gpu_k_w2 + k_in_gpu;
                    unpack_4nk_blocks(src_ptrs, dst, gpu_k_w2);
                    k_abs += 4 * K_STEP;
                  } else {
                    const uint8_t* src = bb->b + (size_t)n_block_begin * cpu_k_w2 +
                                         (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                         (size_t)k_in_block * N_STEP;
                    uint8_t* dst = weight_base + expert_weight_off + (size_t)local_n_start * gpu_k_w2 + k_in_gpu;
                    unpack_nk_block(src, dst, gpu_k_w2);
                    k_abs += K_STEP;
                  }
                }
              }
            }

          } else {
            // ========= Scale copy task: simple linear copy with fast_memcpy =========
            const int scale_task_id = task_id - NUM_W13_TASKS * 2 - NUM_W2_TASKS;

            if (scale_task_id < 2) {
              // Gate (0) or Up (1) scale copy
              const bool is_up = scale_task_id == 1;
              const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

              // W13 scales: copy N blocks corresponding to this CPU TP
              // Note: when gpu_tp > cpu_tp, scale blocks may span multiple GPU TPs
              const int bn_start_global = global_n_offset_w13 / group_size;

              for (int bn = 0; bn < cpu_scale_n_blocks_w13; bn++) {
                const int global_bn = bn_start_global + bn;
                const int target_gpu = global_bn / gpu_scale_n_blocks_w13;
                const int gpu_bn = global_bn % gpu_scale_n_blocks_w13;

                float* scale_dst = (float*)w13_scale_ptrs[target_gpu];
                // Pointer already points to current expert's location, only add offset for up matrix
                const size_t expert_scale_off = is_up ? gpu_w13_scale_per_mat : 0;

                fast_memcpy(scale_dst + expert_scale_off + (size_t)gpu_bn * gpu_scale_k_blocks_w13,
                            bb->d + (size_t)bn * cpu_scale_k_blocks_w13, cpu_scale_k_blocks_w13 * sizeof(float));
              }
            } else {
              // Down scale copy (scale_task_id == 2)
              const auto& bb = down_bb_[expert_id];

              // W2 scales: K dimension is split, copy to each GPU TP
              for (int k_slice_idx = 0; k_slice_idx < div_up(cpu_k_w2, gpu_k_w2); k_slice_idx++) {
                const int k_slice_start = k_slice_idx * gpu_k_w2;
                const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

                const int global_k_start = global_k_offset_w2 + k_slice_start;
                const int target_gpu = global_k_start / gpu_k_w2;
                const int bk_gpu_base = (global_k_start % gpu_k_w2) / group_size;

                float* scale_dst = (float*)w2_scale_ptrs[target_gpu];
                // Pointer already points to current expert's location
                const size_t expert_scale_off = 0;

                const int bk_start = k_slice_start / group_size;
                const int bk_end = div_up(k_slice_end, group_size);
                const int bk_count = bk_end - bk_start;

                for (int bn = 0; bn < cpu_scale_n_blocks_w2; bn++) {
                  fast_memcpy(scale_dst + expert_scale_off + (size_t)bn * gpu_scale_k_blocks_w2 + bk_gpu_base,
                              bb->d + (size_t)bn * cpu_scale_k_blocks_w2 + bk_start, bk_count * sizeof(float));
                }
              }
            }
          }
        },
        nullptr);
  }
};

template <typename K>
class TP_MOE<AMX_FP8_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0 || config.quant_config.zero_point) {
      throw std::runtime_error("FP8 MoE only supports have group_size, zero_point=false");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems =
        (size_t)div_up(config.hidden_size, group_size) * div_up(config.intermediate_size, group_size);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems =
          (size_t)div_up(tpc.intermediate_size, group_size) * div_up(tpc.hidden_size, group_size);

      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];

      tpc.gate_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_scale_elems];

      const size_t tp_idx = (size_t)i;
      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = i * tp_scale_elems;

      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems;

            float* gate_scale_dst = (float*)tpc.gate_scale + expert_id * tp_scale_elems;
            float* up_scale_dst = (float*)tpc.up_scale + expert_id * tp_scale_elems;
            float* down_scale_dst = (float*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const float* gate_scale_src;
            const float* up_scale_src;
            const float* down_scale_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_projs[0][expert_id];

              gate_scale_src = (const float*)config.gate_scales[0][expert_id] + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scales[0][expert_id] + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scales[0][expert_id];
            } else {
              gate_src = (const uint8_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems;

              gate_scale_src =
                  (const float*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + expert_id * full_scale_elems;
            }

            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_scale_elems);

            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            const int n_blocks_n = div_up(config.hidden_size, group_size);
            const int full_n_blocks_k = div_up(config.intermediate_size, group_size);
            const int tp_n_blocks_k = div_up(tpc.intermediate_size, group_size);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const float* src = down_scale_src + (size_t)bn * (size_t)full_n_blocks_k + down_scale_src_block_k_offset;
              float* dst = down_scale_dst + (size_t)bn * (size_t)tp_n_blocks_k;
              std::memcpy(dst, src, sizeof(float) * (size_t)tp_n_blocks_k);
            }
          },
          nullptr);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
      delete[] (float*)tpc.gate_scale;
      delete[] (float*)tpc.up_scale;
      delete[] (float*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w13_scale_ptrs.size() != gpu_tp_count ||
        (int)w2_weight_ptrs.size() != gpu_tp_count || (int)w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP8_MOE_H
