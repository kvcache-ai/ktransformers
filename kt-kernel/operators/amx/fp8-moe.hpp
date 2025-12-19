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

  /**
   * @brief Unpack a single N_STEP x K_STEP block from packed BufferB format to n-major format
   *
   * This is the inverse of the packing done in BufferBFP8Impl::from_mat.
   * Optimized: process by row groups to keep writes more localized.
   *
   * @param src Pointer to packed data (N_STEP * K_STEP bytes in packed layout)
   * @param dst Pointer to destination in n-major layout
   * @param dst_row_stride Row stride in destination buffer (number of columns in full matrix)
   */
  static inline void unpack_nk_block(const uint16_t* src16, uint8_t* dst, int dst_row_stride) {
    // inv_mat_offset: packed position -> original row group index
    // mat_offset = {0, 2, 4, 6, 1, 3, 5, 7}, inv = {0, 4, 1, 5, 2, 6, 3, 7}
    // So row_map[packed_i] = inv_mat_offset[packed_i] * 4
    static constexpr int row_map[8] = {0, 16, 4, 20, 8, 24, 12, 28};
    const uint64_t* src = reinterpret_cast<const uint64_t*>(src16);

    // Packed format: 128 uint64_t values arranged as src[8*j + packed_i]
    // Each uint64_t contains 4 uint16 values for 4 consecutive rows
    for (int packed_i = 0; packed_i < 8; packed_i++) {
      const int base_row = row_map[packed_i];
      uint16_t* row0 = reinterpret_cast<uint16_t*>(dst + (size_t)base_row * dst_row_stride);
      uint16_t* row1 = reinterpret_cast<uint16_t*>(dst + (size_t)(base_row + 1) * dst_row_stride);
      uint16_t* row2 = reinterpret_cast<uint16_t*>(dst + (size_t)(base_row + 2) * dst_row_stride);
      uint16_t* row3 = reinterpret_cast<uint16_t*>(dst + (size_t)(base_row + 3) * dst_row_stride);

      for (int j = 0; j < 16; j++) {
        uint64_t val = src[8 * j + packed_i];
        row0[j] = static_cast<uint16_t>(val);
        row1[j] = static_cast<uint16_t>(val >> 16);
        row2[j] = static_cast<uint16_t>(val >> 32);
        row3[j] = static_cast<uint16_t>(val >> 48);
      }
    }
  }

  /**
   * @brief Reconstruct weights for a single expert to the output buffers (no temp buffer version)
   *
   * Directly unpacks from packed BufferB format to n-major GPU buffers without intermediate storage.
   * Optimized version: merged w13/w2 jobs, coarse-grained splitting, uint16 unpack.
   *
   * Key insight:
   * - w13 (gate+up): Shape [intermediate, hidden], split by N (intermediate) only
   *   K dimension is not split across GPU TPs, so process full K per job.
   * - w2 (down): Shape [hidden, intermediate], split by K (intermediate)
   *   Use gpu_k_w2 as K_SLICE so each slice maps to exactly one GPU TP.
   *
   * @param gpu_tp_count Number of GPU TP parts (1, 2, 4, or 8)
   * @param cpu_tp_count Number of CPU TP parts
   * @param expert_idx Expert index to process
   * @param full_config Full configuration (before CPU TP split)
   * @param w13_weight_ptrs Pointers to gate+up weight buffers (one per GPU TP)
   * @param w13_scale_ptrs Pointers to gate+up scale buffers (one per GPU TP)
   * @param w2_weight_ptrs Pointers to down weight buffers (one per GPU TP)
   * @param w2_scale_ptrs Pointers to down scale buffers (one per GPU TP)
   */
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_idx,
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

    // ========= Job counts =========
    // W13: split by N_STEP (each job processes N_STEP rows x full K, contiguous src and dst)
    // W2: split by (N_STEP, gpu_k_w2) for balanced workload
    const int n_steps_w13 = div_up(cpu_n_w13, N_STEP);
    const int n_steps_w2 = div_up(cpu_n_w2, N_STEP);
    const int k_slices_w2 = div_up(cpu_k_w2, gpu_k_w2);

    const int w13_weight_jobs = n_steps_w13 * 2;  // x2 for gate and up
    const int w2_weight_jobs = n_steps_w2 * k_slices_w2;

    // Scale jobs: 3 simple copy tasks (gate_scale, up_scale, down_scale)
    const int scale_jobs = 3;

    const int total_jobs = w13_weight_jobs + w2_weight_jobs + scale_jobs;

    pool->do_work_stealing_job(
        total_jobs, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < w13_weight_jobs) {
            // ========= W13 weight task: process N_STEP rows x full K (contiguous memory) =========
            const bool is_up = task_id >= n_steps_w13;
            const int n_step_idx = task_id % n_steps_w13;

            const auto& bb = is_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];

            const int local_n_start = n_step_idx * N_STEP;
            if (local_n_start >= cpu_n_w13) return;  // Out of bounds

            const int global_n = global_n_offset_w13 + local_n_start;
            const int target_gpu = global_n / gpu_n_w13;
            const int n_in_gpu = global_n % gpu_n_w13;

            uint8_t* weight_base = (uint8_t*)w13_weight_ptrs[target_gpu];
            const size_t expert_weight_off =
                expert_idx * 2 * gpu_w13_weight_per_mat + (is_up ? gpu_w13_weight_per_mat : 0);

            // Which N_BLOCK does this N_STEP belong to?
            const int n_block_idx = local_n_start / N_BLOCK;
            const int n_block_begin = n_block_idx * N_BLOCK;
            const int n_block_size = std::min(N_BLOCK, cpu_n_w13 - n_block_begin);
            const int n_in_block = local_n_start - n_block_begin;

            // Process all K_BLOCKs for this N_STEP (contiguous in src)
            for (int k_block_begin = 0; k_block_begin < cpu_k_w13; k_block_begin += K_BLOCK) {
              const int k_block_size = std::min(K_BLOCK, cpu_k_w13 - k_block_begin);

              for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
                const uint16_t* src = reinterpret_cast<const uint16_t*>(
                    bb->b + (size_t)n_block_begin * cpu_k_w13 + (size_t)k_block_begin * n_block_size +
                    (size_t)n_in_block * k_block_size + (size_t)k_begin * N_STEP);
                uint8_t* dst = weight_base + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13 + k_block_begin + k_begin;
                unpack_nk_block(src, dst, gpu_k_w13);
              }
            }
          } else if (task_id < w13_weight_jobs + w2_weight_jobs) {
            // ========= W2 weight task: process N_STEP rows x gpu_k_w2 slice =========
            const int w2_task_id = task_id - w13_weight_jobs;
            const int n_step_idx = w2_task_id / k_slices_w2;
            const int k_slice_idx = w2_task_id % k_slices_w2;
            const auto& bb = down_bb_[expert_idx];

            const int local_n_start = n_step_idx * N_STEP;
            if (local_n_start >= cpu_n_w2) return;  // Out of bounds

            const int k_slice_start = k_slice_idx * gpu_k_w2;
            const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

            const int global_k_start = global_k_offset_w2 + k_slice_start;
            const int target_gpu = global_k_start / gpu_k_w2;
            const int k_in_gpu_base = global_k_start % gpu_k_w2;

            uint8_t* weight_base = (uint8_t*)w2_weight_ptrs[target_gpu];
            const size_t expert_weight_off = expert_idx * gpu_w2_weight_per_mat;

            // Which N_BLOCK does this N_STEP belong to?
            const int n_block_idx = local_n_start / N_BLOCK;
            const int n_block_begin = n_block_idx * N_BLOCK;
            const int n_block_size = std::min(N_BLOCK, cpu_n_w2 - n_block_begin);
            const int n_in_block = local_n_start - n_block_begin;

            const int k_block_begin = (k_slice_start / K_BLOCK) * K_BLOCK;
            const int k_block_size = std::min(K_BLOCK, cpu_k_w2 - k_block_begin);

            for (int k_abs = k_slice_start; k_abs < k_slice_end; k_abs += K_STEP) {
              const int k_in_block = k_abs - k_block_begin;
              const int k_in_gpu = k_in_gpu_base + (k_abs - k_slice_start);

              const uint16_t* src = reinterpret_cast<const uint16_t*>(
                  bb->b + (size_t)n_block_begin * cpu_k_w2 + (size_t)k_block_begin * n_block_size +
                  (size_t)n_in_block * k_block_size + (size_t)k_in_block * N_STEP);
              uint8_t* dst = weight_base + expert_weight_off + (size_t)local_n_start * gpu_k_w2 + k_in_gpu;
              unpack_nk_block(src, dst, gpu_k_w2);
            }
          } else {
            // ========= Scale copy task: simple linear copy =========
            const int scale_task_id = task_id - w13_weight_jobs - w2_weight_jobs;

            if (scale_task_id < 2) {
              // Gate (0) or Up (1) scale copy
              const bool is_up = scale_task_id == 1;
              const auto& bb = is_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];

              // W13 scales: copy N blocks corresponding to this CPU TP
              const int bn_start = global_n_offset_w13 / group_size;
              const int bn_end = div_up(global_n_offset_w13 + cpu_n_w13, group_size);
              const int target_gpu = bn_start / gpu_scale_n_blocks_w13;
              const int gpu_bn_start = bn_start % gpu_scale_n_blocks_w13;

              float* scale_dst = (float*)w13_scale_ptrs[target_gpu];
              const size_t expert_scale_off =
                  expert_idx * 2 * gpu_w13_scale_per_mat + (is_up ? gpu_w13_scale_per_mat : 0);

              for (int bn = 0; bn < cpu_scale_n_blocks_w13; bn++) {
                const int gpu_bn = gpu_bn_start + bn;
                memcpy(scale_dst + expert_scale_off + (size_t)gpu_bn * gpu_scale_k_blocks_w13,
                       bb->d + (size_t)bn * cpu_scale_k_blocks_w13, cpu_scale_k_blocks_w13 * sizeof(float));
              }
            } else {
              // Down scale copy (scale_task_id == 2)
              const auto& bb = down_bb_[expert_idx];

              // W2 scales: K dimension is split, copy to each GPU TP
              for (int k_slice_idx = 0; k_slice_idx < div_up(cpu_k_w2, gpu_k_w2); k_slice_idx++) {
                const int k_slice_start = k_slice_idx * gpu_k_w2;
                const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

                const int global_k_start = global_k_offset_w2 + k_slice_start;
                const int target_gpu = global_k_start / gpu_k_w2;
                const int bk_gpu_base = (global_k_start % gpu_k_w2) / group_size;

                float* scale_dst = (float*)w2_scale_ptrs[target_gpu];
                const size_t expert_scale_off = expert_idx * gpu_w2_scale_per_mat;

                const int bk_start = k_slice_start / group_size;
                const int bk_end = div_up(k_slice_end, group_size);
                const int bk_count = bk_end - bk_start;

                for (int bn = 0; bn < cpu_scale_n_blocks_w2; bn++) {
                  memcpy(scale_dst + expert_scale_off + (size_t)bn * gpu_scale_k_blocks_w2 + bk_gpu_base,
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

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_idx, const std::vector<uintptr_t>& w13_weight_ptrs,
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
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_idx, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP8_MOE_H
