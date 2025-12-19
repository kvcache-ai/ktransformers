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
   * @brief Reconstruct weights for a single expert to the output buffers
   *
   * This function handles the TP-specific portion of the reconstruction for one expert.
   * Used for GPU offloading scenarios. Unlike K2 MoE which processes all experts,
   * FP8 MoE processes one expert at a time.
   *
   * Optimized version using work_stealing_job for parallelism:
   * - Gate/Up: Direct write to destination (no temp buffer) since each N_BLOCK maps to one GPU TP
   * - Down: Use work_stealing for both unpack and copy phases
   *
   * @param gpu_tp_count Number of GPU TP parts
   * @param cpu_tp_count Number of CPU TP parts
   * @param expert_idx Expert index to process
   * @param full_config Full configuration (before CPU TP split)
   * @param w13_weight_ptrs Pointers to gate+up weight buffers (one per GPU TP)
   * @param w13_scale_ptrs Pointers to gate+up scale buffers (one per GPU TP)
   * @param w2_weight_ptrs Pointers to down weight buffers (one per GPU TP)
   * @param w2_scale_ptrs Pointers to down scale buffers (one per GPU TP)
   */
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_idx, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    const int group_size = config.quant_config.group_size;
    auto pool = config.pool->get_subpool(tp_part_idx);

    // Calculate sizes for CPU TP part (this instance)
    size_t cpu_tp_weight_elem_count = (size_t)config.intermediate_size * config.hidden_size;
    size_t cpu_tp_n_blocks_gate_up = div_up(config.intermediate_size, group_size);
    size_t cpu_tp_k_blocks = div_up(config.hidden_size, group_size);
    size_t cpu_tp_scale_elem_count_gate_up = cpu_tp_n_blocks_gate_up * cpu_tp_k_blocks;

    // For down: n=hidden_size, k=intermediate_size
    size_t cpu_tp_n_blocks_down = cpu_tp_k_blocks;
    size_t cpu_tp_k_blocks_down = cpu_tp_n_blocks_gate_up;
    size_t cpu_tp_scale_elem_count_down = cpu_tp_n_blocks_down * cpu_tp_k_blocks_down;

    // Calculate sizes for GPU TP part
    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_n_blocks_gate_up = div_up(full_config.intermediate_size / gpu_tp_count, group_size);
    size_t gpu_tp_k_blocks = div_up(full_config.hidden_size, group_size);
    size_t gpu_tp_scale_elem_count_gate_up = gpu_tp_n_blocks_gate_up * gpu_tp_k_blocks;

    size_t gpu_tp_n_blocks_down = gpu_tp_k_blocks;
    size_t gpu_tp_k_blocks_down = gpu_tp_n_blocks_gate_up;
    size_t gpu_tp_scale_elem_count_down = gpu_tp_n_blocks_down * gpu_tp_k_blocks_down;

    if (cpu_tp_count >= gpu_tp_count) {
      // Multiple CPU TPs map to one GPU TP
      int cpu_tps_per_gpu_tp = cpu_tp_count / gpu_tp_count;
      int target_gpu_tp = tp_part_idx / cpu_tps_per_gpu_tp;
      int local_idx = tp_part_idx % cpu_tps_per_gpu_tp;

      // Get pointers for this GPU TP part
      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      float* w13_scale_dst = (float*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      float* w2_scale_dst = (float*)w2_scale_ptrs[target_gpu_tp];

      // Calculate offset within the GPU TP buffer for gate/up
      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_elem_count;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count_gate_up;

      // Expert base offsets in destination buffers
      size_t w13_expert_base_weight = expert_idx * 2 * gpu_tp_weight_elem_count;
      size_t w13_expert_base_scale = expert_idx * 2 * gpu_tp_scale_elem_count_gate_up;
      size_t w2_expert_base_weight = expert_idx * gpu_tp_weight_elem_count;
      size_t w2_expert_base_scale = expert_idx * gpu_tp_scale_elem_count_down;

      // Gate and Up destinations - direct write without temp buffer
      uint8_t* gate_weight_dst = w13_weight_dst + w13_expert_base_weight + offset_in_gpu_weight;
      float* gate_scale_dst = w13_scale_dst + w13_expert_base_scale + offset_in_gpu_scale;
      uint8_t* up_weight_dst =
          w13_weight_dst + w13_expert_base_weight + gpu_tp_weight_elem_count + offset_in_gpu_weight;
      float* up_scale_dst =
          w13_scale_dst + w13_expert_base_scale + gpu_tp_scale_elem_count_gate_up + offset_in_gpu_scale;

      // Gate and Up: write directly using work_stealing
      // With nth = recommended_nth(intermediate_size), each thread handles one N_BLOCK (128 rows)
      int nth_gate_up = T::recommended_nth(config.intermediate_size);
      pool->do_work_stealing_job(
          nth_gate_up * 2, nullptr,
          [this, expert_idx, gate_weight_dst, gate_scale_dst, up_weight_dst, up_scale_dst, nth_gate_up](int task_id) {
            bool is_up = task_id >= nth_gate_up;
            int ith = task_id % nth_gate_up;
            if (is_up) {
              up_bb_[expert_idx]->to_mat(up_weight_dst, up_scale_dst, ith, nth_gate_up);
            } else {
              gate_bb_[expert_idx]->to_mat(gate_weight_dst, gate_scale_dst, ith, nth_gate_up);
            }
          },
          nullptr);

      // Down matrix handling
      size_t gpu_intermediate_per_tp = full_config.intermediate_size / gpu_tp_count;
      size_t cpu_intermediate = config.intermediate_size;

      if (cpu_tps_per_gpu_tp == 1) {
        // Direct write: cpu_tp_count == gpu_tp_count, no column interleaving needed
        uint8_t* down_weight_dst = w2_weight_dst + w2_expert_base_weight;
        float* down_scale_dst = w2_scale_dst + w2_expert_base_scale;

        int nth_down = T::recommended_nth(config.hidden_size);
        pool->do_work_stealing_job(
            nth_down, nullptr,
            [this, expert_idx, down_weight_dst, down_scale_dst, nth_down](int ith) {
              down_bb_[expert_idx]->to_mat(down_weight_dst, down_scale_dst, ith, nth_down);
            },
            nullptr);
      } else {
        // Need column interleaving: unpack to temp buffer, then copy with slicing
        std::vector<uint8_t> down_tmp(cpu_tp_weight_elem_count);
        std::vector<float> down_scale_tmp(cpu_tp_scale_elem_count_down);

        int nth_down = T::recommended_nth(config.hidden_size);
        pool->do_work_stealing_job(
            nth_down, nullptr,
            [this, expert_idx, &down_tmp, &down_scale_tmp, nth_down](int ith) {
              down_bb_[expert_idx]->to_mat(down_tmp.data(), down_scale_tmp.data(), ith, nth_down);
            },
            nullptr);

        // Copy with column interleaving - parallelize by row chunks (N_BLOCK = 128 rows)
        size_t slice_offset_weight = local_idx * cpu_intermediate;
        size_t slice_offset_scale = local_idx * cpu_tp_k_blocks_down;
        int n_row_chunks = div_up(config.hidden_size, T::N_BLOCK);

        pool->do_work_stealing_job(
            n_row_chunks, nullptr,
            [&config, &down_tmp, &down_scale_tmp, w2_weight_dst, w2_scale_dst, w2_expert_base_weight,
             w2_expert_base_scale, gpu_intermediate_per_tp, cpu_intermediate, slice_offset_weight, slice_offset_scale,
             cpu_tp_k_blocks_down, gpu_tp_k_blocks_down, group_size](int chunk_id) {
              int row_start = chunk_id * T::N_BLOCK;
              int row_end = std::min(row_start + T::N_BLOCK, config.hidden_size);

              for (int row = row_start; row < row_end; row++) {
                size_t gpu_row_offset = row * gpu_intermediate_per_tp;
                size_t cpu_row_offset = row * cpu_intermediate;

                std::memcpy(w2_weight_dst + w2_expert_base_weight + gpu_row_offset + slice_offset_weight,
                            down_tmp.data() + cpu_row_offset, cpu_intermediate);
              }

              // Scale: copy once per scale block row
              int scale_row = row_start / group_size;
              if (row_start % group_size == 0 && scale_row * (int)group_size < config.hidden_size) {
                std::memcpy(w2_scale_dst + w2_expert_base_scale + scale_row * gpu_tp_k_blocks_down + slice_offset_scale,
                            down_scale_tmp.data() + scale_row * cpu_tp_k_blocks_down,
                            sizeof(float) * cpu_tp_k_blocks_down);
              }
            },
            nullptr);
      }
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_elem_count / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale_gate_up = cpu_tp_scale_elem_count_gate_up / gpu_tps_per_cpu_tp;

      // Gate and Up: write directly using work_stealing
      // Each thread handles one N_BLOCK, which maps to exactly one GPU TP
      int nth_gate_up = T::recommended_nth(config.intermediate_size);
      int n_blocks_per_gpu_tp = nth_gate_up / gpu_tps_per_cpu_tp;

      pool->do_work_stealing_job(
          nth_gate_up * 2, nullptr,
          [this, expert_idx, &w13_weight_ptrs, &w13_scale_ptrs, start_gpu_tp, n_blocks_per_gpu_tp,
           gpu_tp_weight_elem_count, gpu_tp_scale_elem_count_gate_up, nth_gate_up](int task_id) {
            bool is_up = task_id >= nth_gate_up;
            int ith = task_id % nth_gate_up;

            // Which GPU TP does this N_BLOCK belong to?
            int local_gpu_idx = ith / n_blocks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;
            int ith_within_gpu = ith % n_blocks_per_gpu_tp;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            float* w13_scale_dst = (float*)w13_scale_ptrs[gpu_tp_idx];

            size_t w13_gpu_expert_offset_weight = expert_idx * 2 * gpu_tp_weight_elem_count;
            size_t w13_gpu_expert_offset_scale = expert_idx * 2 * gpu_tp_scale_elem_count_gate_up;

            if (is_up) {
              uint8_t* up_dst = w13_weight_dst + w13_gpu_expert_offset_weight + gpu_tp_weight_elem_count;
              float* up_scale_dst = w13_scale_dst + w13_gpu_expert_offset_scale + gpu_tp_scale_elem_count_gate_up;
              up_bb_[expert_idx]->to_mat(up_dst, up_scale_dst, ith_within_gpu, n_blocks_per_gpu_tp);
            } else {
              uint8_t* gate_dst = w13_weight_dst + w13_gpu_expert_offset_weight;
              float* gate_scale_dst = w13_scale_dst + w13_gpu_expert_offset_scale;
              gate_bb_[expert_idx]->to_mat(gate_dst, gate_scale_dst, ith_within_gpu, n_blocks_per_gpu_tp);
            }
          },
          nullptr);

      // Down: need temp buffer for column slicing
      std::vector<uint8_t> down_tmp(cpu_tp_weight_elem_count);
      std::vector<float> down_scale_tmp(cpu_tp_scale_elem_count_down);

      int nth_down = T::recommended_nth(config.hidden_size);
      pool->do_work_stealing_job(
          nth_down, nullptr,
          [this, expert_idx, &down_tmp, &down_scale_tmp, nth_down](int ith) {
            down_bb_[expert_idx]->to_mat(down_tmp.data(), down_scale_tmp.data(), ith, nth_down);
          },
          nullptr);

      // Copy down with column slicing for each GPU TP - parallelize by (gpu_tp, row_chunk)
      size_t gpu_intermediate_per_tp = full_config.intermediate_size / gpu_tp_count;
      size_t cpu_intermediate_per_gpu_tp = config.intermediate_size / gpu_tps_per_cpu_tp;
      int n_row_chunks = div_up(config.hidden_size, T::N_BLOCK);

      pool->do_work_stealing_job(
          gpu_tps_per_cpu_tp * n_row_chunks, nullptr,
          [&config, &down_tmp, &down_scale_tmp, &w2_weight_ptrs, &w2_scale_ptrs, start_gpu_tp, gpu_tps_per_cpu_tp,
           n_row_chunks, expert_idx, gpu_tp_weight_elem_count, gpu_tp_scale_elem_count_down, gpu_intermediate_per_tp,
           cpu_intermediate_per_gpu_tp, cpu_tp_k_blocks_down, gpu_tp_k_blocks_down, group_size](int task_id) {
            int local_gpu_idx = task_id / n_row_chunks;
            int chunk_id = task_id % n_row_chunks;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            float* w2_scale_dst = (float*)w2_scale_ptrs[gpu_tp_idx];

            size_t w2_gpu_expert_offset_weight = expert_idx * gpu_tp_weight_elem_count;
            size_t w2_gpu_expert_offset_scale = expert_idx * gpu_tp_scale_elem_count_down;

            int row_start = chunk_id * T::N_BLOCK;
            int row_end = std::min(row_start + T::N_BLOCK, config.hidden_size);

            for (int row = row_start; row < row_end; row++) {
              size_t col_offset_weight = row * config.intermediate_size + local_gpu_idx * cpu_intermediate_per_gpu_tp;
              size_t gpu_row_offset = row * gpu_intermediate_per_tp;

              std::memcpy(w2_weight_dst + w2_gpu_expert_offset_weight + gpu_row_offset,
                          down_tmp.data() + col_offset_weight, cpu_intermediate_per_gpu_tp);
            }

            // Scale: copy once per scale block row
            int scale_row = row_start / group_size;
            if (row_start % (int)group_size == 0 && scale_row * (int)group_size < config.hidden_size) {
              size_t col_offset_scale =
                  scale_row * cpu_tp_k_blocks_down + local_gpu_idx * (cpu_tp_k_blocks_down / gpu_tps_per_cpu_tp);
              size_t gpu_scale_row_offset = scale_row * gpu_tp_k_blocks_down;

              std::memcpy(w2_scale_dst + w2_gpu_expert_offset_scale + gpu_scale_row_offset,
                          down_scale_tmp.data() + col_offset_scale,
                          sizeof(float) * (cpu_tp_k_blocks_down / gpu_tps_per_cpu_tp));
            }
          },
          nullptr);
    }
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
