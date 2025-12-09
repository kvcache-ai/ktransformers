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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "moe_base.hpp"
#include "la/amx_raw_utils.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/amx_raw_buffers.hpp"

/**
 * @brief FP8 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to GemmKernel224FP8
 *
 * This class provides FP8-specific GEMM implementations:
 * - do_gate_up_gemm: FP8 weight -> BF16 conversion + AMX GEMM
 * - do_down_gemm: Same FP8->BF16 conversion
 * - load_weights: Load FP8 weights with 128x128 block scales
 */
template <class T = amx::GemmKernel224FP8>
class AMX_FP8_MOE_TP : public AMX_MOE_BASE<T, AMX_FP8_MOE_TP<T>> {
 public:
  using Base = AMX_MOE_BASE<T, AMX_FP8_MOE_TP<T>>;
  using typename Base::input_t;
  using typename Base::output_t;

  // FP8 weight buffer structure
  struct FP8WeightBuffer {
    uint8_t* weight = nullptr;      // FP8 weights [N x K] or [K x N]
    float* scales = nullptr;      // Block scales [n_blocks_n * n_blocks_k]
    int n = 0;
    int k = 0;
    int n_blocks_n = 0;
    int n_blocks_k = 0;

    size_t weight_bytes() const { return (size_t)n * k; }
    size_t scale_count() const { return (size_t)n_blocks_n * n_blocks_k; }
  };

 private:
  // FP8 weight buffers per expert
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;

  // Raw FP8 weights (for layerwise prefill)
  std::vector<FP8WeightBuffer> gate_weights_;
  std::vector<FP8WeightBuffer> up_weights_;
  std::vector<FP8WeightBuffer> down_weights_;

  static constexpr int BLOCK_SIZE = amx::fp8::FP8_BLOCK_SIZE;  // 128

 public:
  AMX_FP8_MOE_TP() = default;

  AMX_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    init_buffer_b_impl();
  }

  // Initialize FP8 weight buffers (called by base class)
  void init_buffer_b() {
    init_buffer_b_impl();
  }

 private:
  void init_buffer_b_impl() {
    auto& config = Base::config_;
    int expert_num = config.expert_num;
    int hidden_size = config.hidden_size;
    int intermediate_size = config.intermediate_size;

    // Calculate block counts
    int n_blocks_gate = (intermediate_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int k_blocks_gate = (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int n_blocks_down = (hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int k_blocks_down = (intermediate_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gate_bb_.resize(expert_num);
    up_bb_.resize(expert_num);
    down_bb_.resize(expert_num);
    gate_weights_.resize(expert_num);
    up_weights_.resize(expert_num);
    down_weights_.resize(expert_num);

    for (int i = 0; i < expert_num; i++) {
      // Allocate BufferB for each expert
      gate_bb_[i] = std::make_shared<typename T::BufferB>(
          intermediate_size, hidden_size, nullptr);
      up_bb_[i] = std::make_shared<typename T::BufferB>(
          intermediate_size, hidden_size, nullptr);
      down_bb_[i] = std::make_shared<typename T::BufferB>(
          hidden_size, intermediate_size, nullptr);

      // Initialize weight buffer metadata
      gate_weights_[i].n = intermediate_size;
      gate_weights_[i].k = hidden_size;
      gate_weights_[i].n_blocks_n = n_blocks_gate;
      gate_weights_[i].n_blocks_k = k_blocks_gate;

      up_weights_[i].n = intermediate_size;
      up_weights_[i].k = hidden_size;
      up_weights_[i].n_blocks_n = n_blocks_gate;
      up_weights_[i].n_blocks_k = k_blocks_gate;

      down_weights_[i].n = hidden_size;
      down_weights_[i].k = intermediate_size;
      down_weights_[i].n_blocks_n = n_blocks_down;
      down_weights_[i].n_blocks_k = k_blocks_down;
    }
  }

 public:
  /**
   * @brief Load FP8 weights from external pointers
   *
   * Supports two loading modes:
   * 1. Per-expert pointers: gate_projs[expert_num], up_projs[expert_num], down_projs[expert_num]
   * 2. Contiguous memory: gate_proj (all experts), up_proj, down_proj
   *
   * @param gate_projs Array of FP8 gate weight pointers per expert
   * @param up_projs Array of FP8 up weight pointers per expert
   * @param down_projs Array of FP8 down weight pointers per expert
   * @param gate_scales Array of gate scale pointers per expert
   * @param up_scales Array of up scale pointers per expert
   * @param down_scales Array of down scale pointers per expert
   */
  void load_weights(void** gate_projs, void** up_projs, void** down_projs,
                    void** gate_scales, void** up_scales, void** down_scales) {
    auto& config = Base::config_;
    int expert_num = config.expert_num;

    for (int i = 0; i < expert_num; i++) {
      // Store raw pointers
      gate_weights_[i].weight = (uint8_t*)gate_projs[i];
      gate_weights_[i].scales = (float*)gate_scales[i];
      up_weights_[i].weight = (uint8_t*)up_projs[i];
      up_weights_[i].scales = (float*)up_scales[i];
      down_weights_[i].weight = (uint8_t*)down_projs[i];
      down_weights_[i].scales = (float*)down_scales[i];

      // Prepare BufferB from FP8 weights
      gate_bb_[i]->from_mat(gate_weights_[i].weight, gate_weights_[i].scales,
                            gate_weights_[i].n, gate_weights_[i].k, 0, 1);
      up_bb_[i]->from_mat(up_weights_[i].weight, up_weights_[i].scales,
                          up_weights_[i].n, up_weights_[i].k, 0, 1);
      down_bb_[i]->from_mat(down_weights_[i].weight, down_weights_[i].scales,
                            down_weights_[i].n, down_weights_[i].k, 0, 1);
    }
  }

  /**
   * @brief Load weights from contiguous memory layout
   *
   * @param gate_proj Contiguous FP8 gate weights [expert_num * N * K]
   * @param up_proj Contiguous FP8 up weights [expert_num * N * K]
   * @param down_proj Contiguous FP8 down weights [expert_num * N * K]
   * @param gate_scale Contiguous gate scales
   * @param up_scale Contiguous up scales
   * @param down_scale Contiguous down scales
   */
  void load_weights_contiguous(void* gate_proj, void* up_proj, void* down_proj,
                               void* gate_scale, void* up_scale, void* down_scale) {
    auto& config = Base::config_;
    int expert_num = config.expert_num;
    int hidden_size = config.hidden_size;
    int intermediate_size = config.intermediate_size;

    size_t gate_weight_per_expert = (size_t)intermediate_size * hidden_size;
    size_t down_weight_per_expert = (size_t)hidden_size * intermediate_size;
    size_t gate_scale_per_expert = gate_weights_[0].scale_count();
    size_t down_scale_per_expert = down_weights_[0].scale_count();

    for (int i = 0; i < expert_num; i++) {
      gate_weights_[i].weight = (uint8_t*)gate_proj + i * gate_weight_per_expert;
      gate_weights_[i].scales = (float*)gate_scale + i * gate_scale_per_expert;
      up_weights_[i].weight = (uint8_t*)up_proj + i * gate_weight_per_expert;
      up_weights_[i].scales = (float*)up_scale + i * gate_scale_per_expert;
      down_weights_[i].weight = (uint8_t*)down_proj + i * down_weight_per_expert;
      down_weights_[i].scales = (float*)down_scale + i * down_scale_per_expert;

      gate_bb_[i]->from_fp8(gate_weights_[i].weight, gate_weights_[i].scales,
                            gate_weights_[i].n, gate_weights_[i].k, 0, 1);
      up_bb_[i]->from_fp8(up_weights_[i].weight, up_weights_[i].scales,
                          up_weights_[i].n, up_weights_[i].k, 0, 1);
      down_bb_[i]->from_fp8(down_weights_[i].weight, down_weights_[i].scales,
                            down_weights_[i].n, down_weights_[i].k, 0, 1);
    }
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& config = Base::config_;
    int m = Base::m_local_num_[expert_idx];
    auto& ba = Base::gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? Base::up_bc_[expert_idx] : Base::gate_bc_[expert_idx];

    // Dispatch based on qlen threshold
    if (qlen > 4 * config.expert_num / config.num_experts_per_tok) {
      amx::mat_mul_fp8(m, config.intermediate_size, config.hidden_size,
                       ba.get(), bb.get(), bc.get(), ith, nth);
    } else {
      amx::vec_mul_fp8(m, config.intermediate_size, config.hidden_size,
                       ba.get(), bb.get(), bc.get(), ith, nth);
    }
  }
  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& config = Base::config_;
    int m = Base::m_local_num_[expert_idx];

    if (qlen > 4 * config.expert_num / config.num_experts_per_tok) {
      amx::mat_mul_fp8(m, config.hidden_size, config.intermediate_size,
                       Base::down_ba_[expert_idx].get(), down_bb_[expert_idx].get(),
                       Base::down_bc_[expert_idx].get(), ith, nth);
    } else {
      amx::vec_mul_fp8(m, config.hidden_size, config.intermediate_size,
                       Base::down_ba_[expert_idx].get(), down_bb_[expert_idx].get(),
                       Base::down_bc_[expert_idx].get(), ith, nth);
    }
  }
  void write_weights_to_buffer(void** w13_weight_ptrs, void** w13_scale_ptrs,
                               void** w2_weight_ptrs, void** w2_scale_ptrs) {
  }

  // Accessors for weight buffers
  const std::vector<std::shared_ptr<typename T::BufferB>>& get_gate_bb() const { return gate_bb_; }
  const std::vector<std::shared_ptr<typename T::BufferB>>& get_up_bb() const { return up_bb_; }
  const std::vector<std::shared_ptr<typename T::BufferB>>& get_down_bb() const { return down_bb_; }
};

#endif  // CPUINFER_OPERATOR_AMX_FP8_MOE_H
