/**
 * @Description  : NVFP4 MoE operator using CRTP pattern, inheriting from moe_base.hpp
 * @Author       : Claude & KVCache.AI Team
 * @Date         : 2025-01-17
 * @Version      : 0.2.0
 * @Copyright (c) 2025 by KVCache.AI, All Rights Reserved.
 *
 * This file implements NVFP4 MoE using CRTP pattern, similar to fp8-moe.hpp.
 * NVFP4 (E2M1) weights use dual-level scaling: FP8 block scales + FP32 tensor scale.
 **/
#ifndef CPUINFER_OPERATOR_AMX_NVFP4_MOE_H
#define CPUINFER_OPERATOR_AMX_NVFP4_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "la/nvfp4_kernel.hpp"
#include "moe_base.hpp"

// ============================================================================
// NVFP4 Kernel Wrapper for AMX_MOE_BASE compatibility
// ============================================================================

/**
 * @brief Wrapper for NVFP4 kernel to provide type aliases required by AMX_MOE_BASE
 *
 * This wrapper adapts nvfp4::GemmKernelNVFP4 and its buffers to the interface
 * expected by AMX_MOE_BASE template.
 */
struct GemmKernelNVFP4Wrapper {
  using BufferA = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>;
  using BufferB = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>;
  using BufferC = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>;

  static constexpr int M_STEP = nvfp4::GemmKernelNVFP4::M_STEP;
  static constexpr int N_STEP = nvfp4::GemmKernelNVFP4::N_STEP;
  static constexpr int K_STEP = nvfp4::GemmKernelNVFP4::K_STEP;
  static constexpr int BLOCK_SIZE = nvfp4::GemmKernelNVFP4::BLOCK_SIZE;

  static constexpr double ELEMENT_SIZE = 0.5625;  // FP4: 4 bits + scale overhead

  static void config() { nvfp4::GemmKernelNVFP4::config(); }

  static int recommended_nth(int n) { return nvfp4::GemmKernelNVFP4::recommended_nth(n); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return nvfp4::GemmKernelNVFP4::split_range_n(n, ith, nth);
  }
};

// ============================================================================
// AMX_NVFP4_MOE_TP - NVFP4 MoE operator using CRTP
// ============================================================================

/**
 * @brief NVFP4 MoE operator using CRTP pattern
 * @tparam T Kernel wrapper type, defaults to GemmKernelNVFP4Wrapper
 *
 * This class provides NVFP4-specific implementations:
 * - do_gate_up_gemm, do_down_gemm: NVFP4 LUT-based matrix multiplication
 * - load_weights: Load NVFP4 weights with FP8 block scales
 */
template <class T = GemmKernelNVFP4Wrapper>
class AMX_NVFP4_MOE_TP : public AMX_MOE_BASE<T, AMX_NVFP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_NVFP4_MOE_TP<T>>;
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

  AMX_NVFP4_MOE_TP() = default;

  AMX_NVFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    auto& quant_config = config_.quant_config;

    if (quant_config.group_size != T::BLOCK_SIZE) {
      printf("Warning: NVFP4 requires group_size=%d, but got %d. Forcing to %d.\n", T::BLOCK_SIZE,
             quant_config.group_size, T::BLOCK_SIZE);
      quant_config.group_size = T::BLOCK_SIZE;
    }

    if (quant_config.zero_point) {
      throw std::runtime_error("NVFP4 MoE does not support zero-point quantization");
    }

    printf("Created AMX_NVFP4_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_NVFP4_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - NVFP4 specific
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }

  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }

  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }

  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }

  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch using NVFP4 LUT multiplication
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Use opt5 with vectorized scale application (SIMD gather + reduce)
    nvfp4::nvfp4_matmul_opt5(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];

    // Use opt5 with vectorized scale application (SIMD gather + reduce)
    nvfp4::nvfp4_matmul_opt5(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                             down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
  }

  // ============================================================================
  // Weight loading - NVFP4 format with FP8 E4M3 block scales
  // ============================================================================

  /**
   * @brief Load NVFP4 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj with scales
   * from config_.gate_scale, up_scale, down_scale.
   *
   * Weight format: packed FP4 (E2M1), 2 values per byte
   * Scale format: FP8 E4M3, one per block of 16 elements
   */
  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("NVFP4 MoE requires FP8 E4M3 scale tensors");
    }

    // Load gate and up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Calculate sizes
          size_t weight_elements = config_.intermediate_size * config_.hidden_size;
          size_t scale_elements = weight_elements / T::BLOCK_SIZE;

          // Load gate weights and scales
          gate_bb_[expert_idx]->from_raw_nvfp4(
              (uint8_t*)config_.gate_proj + ((logical_expert_id * weight_elements) >> 1),
              (uint8_t*)config_.gate_scale + (logical_expert_id * scale_elements),
              1.0f,  // Tensor scale
              ith, nth);

          // Load up weights and scales
          up_bb_[expert_idx]->from_raw_nvfp4((uint8_t*)config_.up_proj + ((logical_expert_id * weight_elements) >> 1),
                                             (uint8_t*)config_.up_scale + (logical_expert_id * scale_elements),
                                             1.0f,  // Tensor scale
                                             ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_elements = config_.hidden_size * config_.intermediate_size;
          size_t scale_elements = weight_elements / T::BLOCK_SIZE;

          // Load down weights and scales
          down_bb_[expert_idx]->from_raw_nvfp4(
              (uint8_t*)config_.down_proj + ((logical_expert_id * weight_elements) >> 1),
              (uint8_t*)config_.down_scale + (logical_expert_id * scale_elements),
              1.0f,  // Tensor scale
              ith, nth);
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization for AMX_NVFP4_MOE_TP
// ============================================================================

template <typename K>
class TP_MOE<AMX_NVFP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_NVFP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_NVFP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    constexpr int BLOCK_SIZE = K::BLOCK_SIZE;

    if (config.gate_scale == nullptr) {
      throw std::runtime_error("NVFP4 MoE requires packed FP4 with FP8 E4M3 scales");
    }

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems = full_weight_elems / BLOCK_SIZE;

    // Allocate and copy weights for each TP part
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = tp_weight_elems / BLOCK_SIZE;

      // Allocate buffers for packed FP4 weights (2 values per byte)
      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems / 2];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems / 2];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems / 2];

      // Allocate buffers for FP8 scales
      tpc.gate_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new uint8_t[tpc.expert_num * tp_scale_elems];

      const size_t tp_idx = (size_t)i;
      const size_t gate_up_weight_src_offset = i * tp_weight_elems / 2;  // Byte offset
      const size_t gate_up_scale_src_offset = i * tp_scale_elems;

      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)BLOCK_SIZE;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems / 2;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems / 2;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems / 2;

            uint8_t* gate_scale_dst = (uint8_t*)tpc.gate_scale + expert_id * tp_scale_elems;
            uint8_t* up_scale_dst = (uint8_t*)tpc.up_scale + expert_id * tp_scale_elems;
            uint8_t* down_scale_dst = (uint8_t*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src =
                (const uint8_t*)config.gate_proj + expert_id * full_weight_elems / 2 + gate_up_weight_src_offset;
            const uint8_t* up_src =
                (const uint8_t*)config.up_proj + expert_id * full_weight_elems / 2 + gate_up_weight_src_offset;
            const uint8_t* down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems / 2;

            const uint8_t* gate_scale_src =
                (const uint8_t*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
            const uint8_t* up_scale_src =
                (const uint8_t*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
            const uint8_t* down_scale_src = (const uint8_t*)config.down_scale + expert_id * full_scale_elems;

            // Copy gate and up weights/scales (simple contiguous copy for row-split)
            std::memcpy(gate_dst, gate_src, tp_weight_elems / 2);
            std::memcpy(up_dst, up_src, tp_weight_elems / 2);
            std::memcpy(gate_scale_dst, gate_scale_src, tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, tp_scale_elems);

            // Copy down weights (column-split, need per-row copy)
            // Each row has config.intermediate_size / 2 bytes for FP4
            const size_t full_row_bytes = config.intermediate_size / 2;
            const size_t tp_row_bytes = tpc.intermediate_size / 2;
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * full_row_bytes + down_weight_src_col_offset / 2;
              const size_t dst_row_offset = (size_t)row * tp_row_bytes;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, tp_row_bytes);
            }

            // Copy down scales (column-split)
            const int n_blocks_n = div_up(config.hidden_size, BLOCK_SIZE);
            const int full_n_blocks_k = div_up(config.intermediate_size, BLOCK_SIZE);
            const int tp_n_blocks_k = div_up(tpc.intermediate_size, BLOCK_SIZE);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const uint8_t* src =
                  down_scale_src + (size_t)bn * (size_t)full_n_blocks_k + down_scale_src_block_k_offset;
              uint8_t* dst = down_scale_dst + (size_t)bn * (size_t)tp_n_blocks_k;
              std::memcpy(dst, src, (size_t)tp_n_blocks_k);
            }
          },
          nullptr);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    // Cleanup temporary buffers
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (uint8_t*)(tpc.gate_scale);
      delete[] (uint8_t*)(tpc.up_scale);
      delete[] (uint8_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }
};

#endif  // CPUINFER_OPERATOR_AMX_NVFP4_MOE_H
