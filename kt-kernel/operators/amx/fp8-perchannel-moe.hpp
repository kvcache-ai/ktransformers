/**
 * @Description  : FP8 Per-Channel AMX MoE operator for GLM-4.7-FP8 native inference
 * @Author       : Claude
 * @Date         : 2025-01-12
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements FP8 MoE with per-channel quantization using CRTP pattern.
 * Per-channel quantization: each output channel (row) has one scale factor.
 * This is different from block-wise quantization where each 128x128 block has one scale.
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP8_PERCHANNEL_MOE_H
#define CPUINFER_OPERATOR_AMX_FP8_PERCHANNEL_MOE_H

#include "la/amx_raw_buffers.hpp"
#include "la/amx_raw_kernels.hpp"
#include "moe_base.hpp"

/**
 * @brief FP8 Per-Channel MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to GemmKernel224FP8PerChannel
 *
 * This class provides FP8 per-channel specific implementations:
 * - do_gate_up_gemm, do_down_gemm : FP8 weight -> BF16 conversion mat mul with per-channel scale
 * - load_weights: Load FP8 weights with per-channel scales (shape: [n])
 */
template <class T = amx::GemmKernel224FP8PerChannel>
class AMX_FP8_PERCHANNEL_MOE_TP : public AMX_MOE_BASE<T, AMX_FP8_PERCHANNEL_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP8_PERCHANNEL_MOE_TP<T>>;
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

  AMX_FP8_PERCHANNEL_MOE_TP() = default;

  AMX_FP8_PERCHANNEL_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    // Initialization now happens in derived_init() which is called by base constructor
  }

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (!quant_config.per_channel) {
      throw std::runtime_error("KT-Kernel FP8 Per-Channel MoE requires per_channel=true");
    }
    printf("Created AMX_FP8_PERCHANNEL_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_FP8_PERCHANNEL_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - per-channel (no group_size needed)
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    // Per-channel: weight size + n scales (no group_size)
    return T::BufferB::required_size(n, k);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    // Per-channel BufferB doesn't need group_size
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch (per-channel)
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Per-channel: use vec_mul_perchannel instead of vec_mul_kgroup
    amx::float_mat_vec_perchannel<T>(m, config_.intermediate_size, config_.hidden_size, ba.get(), bb.get(), bc.get(),
                                     ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];

    amx::float_mat_vec_perchannel<T>(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx].get(),
                                     down_bb_[expert_idx].get(), down_bc_[expert_idx].get(), ith, nth);
  }

  /**
   * @brief Load FP8 weights from contiguous memory layout with per-channel scales
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj with scales
   * from config_.gate_scale, up_scale, down_scale.
   *
   * Per-channel scale shape: [n] (one scale per output channel)
   */
  void load_weights() {
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("FP8 Per-Channel MoE requires scale pointers.");
    }

    // load gate and up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Per-channel scale: shape [intermediate_size] for gate/up
          const size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          const size_t scale_offset = logical_expert_id * config_.intermediate_size;

          // gate part
          gate_bb_[expert_idx]->from_mat((uint8_t*)config_.gate_proj + weight_offset,
                                         (float*)config_.gate_scale + scale_offset, ith, nth);
          // up part
          up_bb_[expert_idx]->from_mat((uint8_t*)config_.up_proj + weight_offset,
                                       (float*)config_.up_scale + scale_offset, ith, nth);
        },
        nullptr);

    // load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Per-channel scale: shape [hidden_size] for down
          const size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          const size_t scale_offset = logical_expert_id * config_.hidden_size;

          // down part
          down_bb_[expert_idx]->from_mat((uint8_t*)config_.down_proj + weight_offset,
                                         (float*)config_.down_scale + scale_offset, ith, nth);
        },
        nullptr);
  }
};

/**
 * @brief TP_MOE specialization for FP8 Per-Channel MoE
 */
template <typename K>
class TP_MOE<AMX_FP8_PERCHANNEL_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP8_PERCHANNEL_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP8_PERCHANNEL_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (!config.quant_config.per_channel) {
      throw std::runtime_error("FP8 Per-Channel MoE requires per_channel=true");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    // Per-channel: scale count = output dimension
    const size_t gate_up_scale_elems = (size_t)config.intermediate_size;
    const size_t down_scale_elems = (size_t)config.hidden_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      // Per-channel scales for TP part
      const size_t tp_gate_up_scale_elems = (size_t)tpc.intermediate_size;
      const size_t tp_down_scale_elems = (size_t)tpc.hidden_size;

      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];

      tpc.gate_scale = new float[tpc.expert_num * tp_gate_up_scale_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_gate_up_scale_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_down_scale_elems];

      const size_t tp_idx = (size_t)i;
      // gate/up: split by N (intermediate_size)
      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = i * tp_gate_up_scale_elems;

      // down: split by K (intermediate_size)
      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems;

            float* gate_scale_dst = (float*)tpc.gate_scale + expert_id * tp_gate_up_scale_elems;
            float* up_scale_dst = (float*)tpc.up_scale + expert_id * tp_gate_up_scale_elems;
            float* down_scale_dst = (float*)tpc.down_scale + expert_id * tp_down_scale_elems;

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
                  (const float*)config.gate_scale + expert_id * gate_up_scale_elems + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scale + expert_id * gate_up_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + expert_id * down_scale_elems;
            }

            // Copy gate/up weights and scales (N dimension split)
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_gate_up_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_gate_up_scale_elems);

            // Copy down weights (K dimension split) - row by row
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            // Copy down scales (N dimension = hidden_size, full copy for each TP)
            std::memcpy(down_scale_dst, down_scale_src, sizeof(float) * tp_down_scale_elems);
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
};

#endif  // CPUINFER_OPERATOR_AMX_FP8_PERCHANNEL_MOE_H
