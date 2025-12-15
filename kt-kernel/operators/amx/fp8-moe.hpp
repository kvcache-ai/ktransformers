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
  using Base = AMX_MOE_BASE<T, AMX_FP8_MOE_TP<T>>;
  using Base::config_;
  using Base::tp_part_idx;
  using Base::gate_bb_;
  using Base::up_bb_;
  using Base::down_bb_;
  using Base::gate_up_ba_;
  using Base::gate_bc_;
  using Base::up_bc_;
  using Base::down_ba_;
  using Base::down_bc_;
  using Base::m_local_num_;
  
 public:

  AMX_FP8_MOE_TP() = default;

  AMX_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("KT-Kernel fp8 MoE only support block-wise FP8. group_size = %d, zero_point = %d",
                               quant_config.group_size, quant_config.zero_point);
    }
    printf("Created AMX_FP8_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));
  }
  // ============================================================================
  // CRTP buffer creation - with group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const {
    return T::BufferC::required_size(m, n);
  }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
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

    // Dispatch based on qlen threshold
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_fp8(m, config_.intermediate_size, config_.hidden_size,
                       ba.get(), bb.get(), bc.get(), ith, nth);
    } else {
      amx::vec_mul_fp8(m, config_.intermediate_size, config_.hidden_size,
                       ba.get(), bb.get(), bc.get(), ith, nth);
    }
  }
  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_fp8(m, config_.hidden_size, config_.intermediate_size,
                       down_ba_[expert_idx].get(), down_bb_[expert_idx].get(),
                       down_bc_[expert_idx].get(), ith, nth);
    } else {
      amx::vec_mul_fp8(m, config_.hidden_size, config_.intermediate_size,
                       down_ba_[expert_idx].get(), down_bb_[expert_idx].get(),
                       down_bc_[expert_idx].get(), ith, nth);
    }
  }

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
      [this, nth, physical_to_logical_map](int task_id) {
        uint64_t expert_idx = task_id / nth;
        uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
        int ith = task_id % nth;
        // gate part
        gate_bb_[expert_idx]->from_mat(
            (uint8_t*)config_.gate_proj +
                (logical_expert_id * config_.intermediate_size * config_.hidden_size),
            (ggml_bf16_t*)config_.gate_scale +
                (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
            ith, nth);
        // up part
        up_bb_[expert_idx]->from_mat(
            (uint8_t*)config_.up_proj +
                (logical_expert_id * config_.intermediate_size * config_.hidden_size),
            (ggml_bf16_t*)config_.up_scale +
                (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
            ith, nth);
      },
      nullptr);
      
      nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // down part
          down_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.down_proj +
                  (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              (ggml_bf16_t*)config_.down_scale +
                  (logical_expert_id * (config_.hidden_size / group_size) * (config_.intermediate_size / group_size)),
            ith, nth);
        },
        nullptr);
  #ifdef DEBUG_K2_MOE
    dump_buffer_b("Native FP8", 0, "gate", gate_bb_[0].get());
    dump_buffer_b("Native FP8", 0, "down", down_bb_[0].get());
  #endif
  }
  
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int num_experts, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    // TODO: write by AI, wrong!
    if (gpu_tp_count != cpu_tp_count || gpu_tp_count != full_config.pool->config.subpool_count) {
      throw std::runtime_error("FP8 write_weights_to_buffer currently requires gpu_tp_count == cpu_tp_count");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w13_scale_ptrs.size() != gpu_tp_count ||
        (int)w2_weight_ptrs.size() != gpu_tp_count || (int)w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

    auto& config = Base::config_;
    const int group_size = config.quant_config.group_size;
    if (group_size != 128) {
      throw std::runtime_error("FP8 write_weights_to_buffer requires group_size=128");
    }

    auto pool = config.pool->get_subpool(Base::tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const size_t gpu_tp_intermediate = (size_t)full_config.intermediate_size / (size_t)gpu_tp_count;
    const size_t gpu_tp_gate_up_weight_bytes = gpu_tp_intermediate * (size_t)full_config.hidden_size;
    const size_t gpu_tp_gate_up_scale_elems =
        (size_t)div_up((int)gpu_tp_intermediate, group_size) * div_up(full_config.hidden_size, group_size);
    const size_t gpu_tp_down_weight_bytes = (size_t)full_config.hidden_size * gpu_tp_intermediate;
    const size_t gpu_tp_down_scale_elems =
        (size_t)div_up(full_config.hidden_size, group_size) * div_up((int)gpu_tp_intermediate, group_size);

    uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[Base::tp_part_idx];
    ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[Base::tp_part_idx];
    uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[Base::tp_part_idx];
    ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[Base::tp_part_idx];

    pool->do_work_stealing_job(
        num_experts, nullptr,
        [&, this](int expert_id_) {
          size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

          const size_t w13_gpu_expert_offset_weight = expert_id * 2 * gpu_tp_gate_up_weight_bytes;
          const size_t w13_gpu_expert_offset_scale = expert_id * 2 * gpu_tp_gate_up_scale_elems;
          const size_t w2_gpu_expert_offset_weight = expert_id * gpu_tp_down_weight_bytes;
          const size_t w2_gpu_expert_offset_scale = expert_id * gpu_tp_down_scale_elems;

          std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight,
                      (const void*)Base::gate_bb_[expert_id_]->b, gpu_tp_gate_up_weight_bytes);
          std::memcpy(w13_scale_dst + w13_gpu_expert_offset_scale,
                      (const void*)Base::gate_bb_[expert_id_]->d, sizeof(ggml_bf16_t) * gpu_tp_gate_up_scale_elems);

          std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight + gpu_tp_gate_up_weight_bytes,
                      (const void*)Base::up_bb_[expert_id_]->b, gpu_tp_gate_up_weight_bytes);
          std::memcpy(w13_scale_dst + w13_gpu_expert_offset_scale + gpu_tp_gate_up_scale_elems,
                      (const void*)Base::up_bb_[expert_id_]->d, sizeof(ggml_bf16_t) * gpu_tp_gate_up_scale_elems);

          std::memcpy(w2_weight_dst + w2_gpu_expert_offset_weight,
                      (const void*)Base::down_bb_[expert_id_]->b, gpu_tp_down_weight_bytes);
          std::memcpy(w2_scale_dst + w2_gpu_expert_offset_scale,
                      (const void*)Base::down_bb_[expert_id_]->d, sizeof(ggml_bf16_t) * gpu_tp_down_scale_elems);
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
    const size_t full_scale_elems = (size_t)div_up(config.hidden_size, group_size) * div_up(config.intermediate_size, group_size);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = (size_t)div_up(tpc.intermediate_size, group_size) * div_up(tpc.hidden_size, group_size);

      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];

      tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new ggml_bf16_t[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new ggml_bf16_t[tpc.expert_num * tp_scale_elems];

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

          ggml_bf16_t* gate_scale_dst = (ggml_bf16_t*)tpc.gate_scale + expert_id * tp_scale_elems;
          ggml_bf16_t* up_scale_dst = (ggml_bf16_t*)tpc.up_scale + expert_id * tp_scale_elems;
          ggml_bf16_t* down_scale_dst = (ggml_bf16_t*)tpc.down_scale + expert_id * tp_scale_elems;

          const uint8_t* gate_src;
          const uint8_t* up_src;
          const uint8_t* down_src;
          const ggml_bf16_t* gate_scale_src;
          const ggml_bf16_t* up_scale_src;
          const ggml_bf16_t* down_scale_src;

          if (use_per_expert_ptrs) {
            gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
            up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
            down_src = (const uint8_t*)config.down_projs[0][expert_id];

            gate_scale_src = (const ggml_bf16_t*)config.gate_scales[0][expert_id] + gate_up_scale_src_offset;
            up_scale_src = (const ggml_bf16_t*)config.up_scales[0][expert_id] + gate_up_scale_src_offset;
            down_scale_src = (const ggml_bf16_t*)config.down_scales[0][expert_id];
          } else {
            gate_src = (const uint8_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
            up_src = (const uint8_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
            down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems;

            gate_scale_src = (const ggml_bf16_t*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
            up_scale_src = (const ggml_bf16_t*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
            down_scale_src = (const ggml_bf16_t*)config.down_scale + expert_id * full_scale_elems;
          }

          std::memcpy(gate_dst, gate_src, tp_weight_elems);
          std::memcpy(up_dst, up_src, tp_weight_elems);
          std::memcpy(gate_scale_dst, gate_scale_src, sizeof(ggml_bf16_t) * tp_scale_elems);
          std::memcpy(up_scale_dst, up_scale_src, sizeof(ggml_bf16_t) * tp_scale_elems);

          for (int row = 0; row < config.hidden_size; row++) {
            const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
            const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
            std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
          }

          const int n_blocks_n = div_up(config.hidden_size, group_size);
          const int full_n_blocks_k = div_up(config.intermediate_size, group_size);
          const int tp_n_blocks_k = div_up(tpc.intermediate_size, group_size);
          for (int bn = 0; bn < n_blocks_n; bn++) {
            const ggml_bf16_t* src = down_scale_src + (size_t)bn * (size_t)full_n_blocks_k + down_scale_src_block_k_offset;
            ggml_bf16_t* dst = down_scale_dst + (size_t)bn * (size_t)tp_n_blocks_k;
            std::memcpy(dst, src, sizeof(ggml_bf16_t) * (size_t)tp_n_blocks_k);
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
      delete[] (ggml_bf16_t*)tpc.gate_scale;
      delete[] (ggml_bf16_t*)tpc.up_scale;
      delete[] (ggml_bf16_t*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
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
      this->tps[i]->write_weights_to_buffer(
          gpu_tp_count, this->tp_count,
          gpu_experts_num, this->config,
          w13_weight_ptrs, w13_scale_ptrs,
          w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP8_MOE_H
