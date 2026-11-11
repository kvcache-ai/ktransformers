/**
 * @Description  : AVX2 FP8 MoE with CPU Offloading Engine
 * @Author       : CPU Offloading Implementation
 * @Date         : 2026
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This variant enables running models larger than CPU cache by:
 * 1. Keeping original quantized weights in system RAM
 * 2. Pre-decoding only active experts on-demand
 * 3. Prefetching next expert weights during compute
 * 4. LRU eviction of unused experts from cache
 **/
#ifndef CPUINFER_OPERATOR_AVX2_FP8_MOE_OFFLOAD_H
#define CPUINFER_OPERATOR_AVX2_FP8_MOE_OFFLOAD_H

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "../common.hpp"
#include "../moe-tp.hpp"
#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "fp8_dequant.hpp"
#include "fp8-moe.hpp"
#include "llama.cpp/ggml.h"
#include "offload_engine.hpp"

// Platform-specific NUMA/CPU affinity
#if defined(__linux__)
#include <numa.h>
#include <sched.h>
#elif defined(__APPLE__)
// macOS fallback - no NUMA support
inline int sched_getcpu(void) { return 0; }
inline int numa_node_of_cpu(int cpu) { return 0; }
#else
inline int sched_getcpu(void) { return 0; }
inline int numa_node_of_cpu(int cpu) { return 0; }
#endif

namespace avx2 {

// ============================================================================
// Offloaded FP8 MoE Template (per-TP part)
// ============================================================================
template <class T>
class AVX2_FP8_MOE_OFFLOAD_TP : public AVX2_MOE_BASE<T, AVX2_FP8_MOE_OFFLOAD_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_FP8_MOE_OFFLOAD_TP<T>>;

 public:
  using Base::config_;
  using Base::gate_bb_;
  using Base::up_bb_;
  using Base::down_bb_;
  using Base::gate_up_ba_;
  using Base::gate_bc_;
  using Base::up_bc_;
  using Base::down_ba_;
  using Base::down_bc_;
  using Base::m_local_num_;
  using Base::tp_part_idx;

  std::unique_ptr<OffloadEngine> offload_engine_;

  AVX2_FP8_MOE_OFFLOAD_TP(GeneralMOEConfig config, int tp_part_idx_)
      : Base(config, tp_part_idx_) {
    // Don't initialize offload_engine_ here - config_ pointers not set yet
    // init_offload_engine() will be called in load_weights() after TP_MOE specialization sets up pointers
    Base::derived()->derived_init();
    printf("Created AVX2_FP8_MOE_OFFLOAD_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AVX2_FP8_MOE_OFFLOAD_TP() {
    if (offload_engine_) {
      offload_engine_->shutdown();
    }
  }

  void init_offload_engine() {
    OffloadConfig offload_config;
    offload_config.expert_num = config_.expert_num;
    offload_config.hidden_size = config_.hidden_size;
    offload_config.intermediate_size = config_.intermediate_size;
    offload_config.block_size = config_.quant_config.group_size;
    offload_config.k_group_size = 0;  // FP8 doesn't use k_group_size (only RAWINT4)
    offload_config.max_cached_experts = 8;  // Keep up to 8 experts pre-decoded (match expert_num for test)
    offload_config.enable_prefetch = true;
    offload_config.prefetch_distance = 1;
    offload_config.worker_pool = config_.pool;

    // Original quantized weights in RAM (now config_ pointers are valid)
    offload_config.gate_proj = (const uint8_t*)config_.gate_proj;
    offload_config.up_proj = (const uint8_t*)config_.up_proj;
    offload_config.down_proj = (const uint8_t*)config_.down_proj;
    offload_config.gate_scale = (const float*)config_.gate_scale;
    offload_config.up_scale = (const float*)config_.up_scale;
    offload_config.down_scale = (const float*)config_.down_scale;
    offload_config.physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;

    offload_engine_ = std::make_unique<OffloadEngine>(offload_config);
    offload_engine_->initialize();
  }

  // CRTP buffer creation - smaller BufferB (no pre-decoded)
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    // BufferB only needs space for weights and scales (no pre-decoded)
    // since we use offload_engine_ for pre-decoded weights
    size_t n_blocks_n = div_up((int)n, config_.quant_config.group_size);
    size_t n_blocks_k = div_up((int)k, config_.quant_config.group_size);
    return n * k + n_blocks_n * n_blocks_k * sizeof(float);
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

  // GEMM dispatch with offloaded weights
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Get pre-decoded weights from offload engine
    int logical_expert_id = expert_idx;  // TODO: handle TP mapping
    auto weights = offload_engine_->get_expert_weights(logical_expert_id);

    if (do_up) {
      avx2::gemm_fp8_with_predecoded(
          m, config_.intermediate_size, config_.hidden_size,
          *ba, *bb, *bc,
          weights.up_predecoded, weights.up_scale,
          ith, nth);
    } else {
      avx2::gemm_fp8_with_predecoded(
          m, config_.intermediate_size, config_.hidden_size,
          *ba, *bb, *bc,
          weights.gate_predecoded, weights.gate_scale,
          ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = down_ba_[expert_idx];
    auto& bb = down_bb_[expert_idx];
    auto& bc = down_bc_[expert_idx];

    int logical_expert_id = expert_idx;
    auto weights = offload_engine_->get_expert_weights(logical_expert_id);

    avx2::gemm_fp8_with_predecoded(
        m, config_.hidden_size, config_.intermediate_size,
        *ba, *bb, *bc,
        weights.down_predecoded, weights.down_scale,
        ith, nth);
  }

  // Load weights - only store quantized weights and scales (no pre-decoding)
  // We override from_mat behavior to avoid pre-decoding into non-existent buffer
  void load_weights() {
    // Initialize offload_engine now that config_ pointers are set by TP_MOE specialization
    if (!offload_engine_) {
      init_offload_engine();
    }

    auto& quant_config = config_.quant_config;
    int group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("FP8 MOE requires scale pointers.");
    }

    // Helper lambda to copy weights and scales without pre-decoding
    auto copy_weights_scales = [this](std::shared_ptr<typename T::BufferB>& bb,
                                      const uint8_t* src_weights, const float* src_scales,
                                      int ith, int nth) {
      // Copy weights (split by N)
      auto [n_start, n_end] = avx2::split_range((int)bb->n, ith, nth);
      std::memcpy(bb->b + n_start * bb->k, src_weights + n_start * bb->k,
                  (size_t)(n_end - n_start) * bb->k);

      // Copy scales (split by N blocks)
      int n_blocks_k = div_up((int)bb->k, bb->block_size);
      int nb_start = n_start / bb->block_size;
      int nb_end = div_up(n_end, bb->block_size);
      std::memcpy(bb->d + nb_start * n_blocks_k, src_scales + nb_start * n_blocks_k,
                  (size_t)(nb_end - nb_start) * n_blocks_k * sizeof(float));
      // NOTE: No pre-decoding - offload_engine handles that on-demand
    };

    // Load gate + up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size, copy_weights_scales](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id *
              avx2::div_up(config_.hidden_size, group_size) *
              avx2::div_up(config_.intermediate_size, group_size);

          copy_weights_scales(gate_bb_[expert_idx],
              (uint8_t*)config_.gate_proj + weight_offset,
              (float*)config_.gate_scale + scale_offset,
              ith, nth);

          copy_weights_scales(up_bb_[expert_idx],
              (uint8_t*)config_.up_proj + weight_offset,
              (float*)config_.up_scale + scale_offset,
              ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size, copy_weights_scales](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id *
              avx2::div_up(config_.hidden_size, group_size) *
              avx2::div_up(config_.intermediate_size, group_size);

          copy_weights_scales(down_bb_[expert_idx],
              (uint8_t*)config_.down_proj + weight_offset,
              (float*)config_.down_scale + scale_offset,
              ith, nth);
        },
        nullptr);
  }

  void derived_init() {}
};

}  // namespace avx2

// ============================================================================
// TP_MOE specialization for offloaded FP8 MoE
// Must be at global scope (same namespace as primary template)
// Inherits from TP_MOE<AVX2_MOE_BASE<...>> specialization in moe_base.hpp
// ============================================================================
template <typename K>
class TP_MOE<avx2::AVX2_FP8_MOE_OFFLOAD_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, avx2::AVX2_FP8_MOE_OFFLOAD_TP<K>>> {
  using Base = TP_MOE<AVX2_MOE_BASE<K, avx2::AVX2_FP8_MOE_OFFLOAD_TP<K>>>;
  using OffloadPart = avx2::AVX2_FP8_MOE_OFFLOAD_TP<K>;

 public:
  using Base::Base;
  using Base::config;
  using Base::tps;
  using Base::tp_configs;
  using Base::tp_count;
  using Base::weights_loaded;

  // Store quantized weights persistently for offload engine (NOT freed after load)
  std::vector<uint8_t*> tp_gate_proj_;
  std::vector<uint8_t*> tp_up_proj_;
  std::vector<uint8_t*> tp_down_proj_;
  std::vector<float*> tp_gate_scale_;
  std::vector<float*> tp_up_scale_;
  std::vector<float*> tp_down_scale_;

  ~TP_MOE() {
    // Free persistent quantized weight buffers
    for (size_t i = 0; i < tp_gate_proj_.size(); ++i) {
      delete[] tp_gate_proj_[i];
      delete[] tp_up_proj_[i];
      delete[] tp_down_proj_[i];
      delete[] tp_gate_scale_[i];
      delete[] tp_up_scale_[i];
      delete[] tp_down_scale_[i];
    }
  }

  void load_weights() override {
    auto& pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0 || config.quant_config.zero_point) {
      throw std::runtime_error("FP8 MoE only supports block-wise (group_size > 0, zero_point=false)");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems =
        (size_t)avx2::div_up(config.hidden_size, group_size) * avx2::div_up(config.intermediate_size, group_size);

    // Resize persistent storage vectors
    tp_gate_proj_.resize(tp_count);
    tp_up_proj_.resize(tp_count);
    tp_down_proj_.resize(tp_count);
    tp_gate_scale_.resize(tp_count);
    tp_up_scale_.resize(tp_count);
    tp_down_scale_.resize(tp_count);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems =
          (size_t)avx2::div_up(tpc.intermediate_size, group_size) * avx2::div_up(tpc.hidden_size, group_size);

      // Allocate PERSISTENT buffers (NOT freed after load - needed by offload_engine)
      tp_gate_proj_[i] = new uint8_t[tpc.expert_num * tp_weight_elems];
      tp_up_proj_[i] = new uint8_t[tpc.expert_num * tp_weight_elems];
      tp_down_proj_[i] = new uint8_t[tpc.expert_num * tp_weight_elems];
      tp_gate_scale_[i] = new float[tpc.expert_num * tp_scale_elems];
      tp_up_scale_[i] = new float[tpc.expert_num * tp_scale_elems];
      tp_down_scale_[i] = new float[tpc.expert_num * tp_scale_elems];

      // Point tpc to persistent buffers
      tpc.gate_proj = tp_gate_proj_[i];
      tpc.up_proj = tp_up_proj_[i];
      tpc.down_proj = tp_down_proj_[i];
      tpc.gate_scale = tp_gate_scale_[i];
      tpc.up_scale = tp_up_scale_[i];
      tpc.down_scale = tp_down_scale_[i];

      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = i * tp_scale_elems;
      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = tp_gate_proj_[i] + expert_id * tp_weight_elems;
            uint8_t* up_dst = tp_up_proj_[i] + expert_id * tp_weight_elems;
            uint8_t* down_dst = tp_down_proj_[i] + expert_id * tp_weight_elems;
            float* gate_scale_dst = tp_gate_scale_[i] + expert_id * tp_scale_elems;
            float* up_scale_dst = tp_up_scale_[i] + expert_id * tp_scale_elems;
            float* down_scale_dst = tp_down_scale_[i] + expert_id * tp_scale_elems;

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
              gate_scale_src = (const float*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + expert_id * full_scale_elems;
            }

            // Copy gate/up weights + scales (column slice)
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_scale_elems);

            // Copy down weights (row-wise split)
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            // Copy down scales (block-row-wise split)
            const int n_blocks_n = avx2::div_up(config.hidden_size, group_size);
            const int full_n_blocks_k = avx2::div_up(config.intermediate_size, group_size);
            const int tp_n_blocks_k = avx2::div_up(tpc.intermediate_size, group_size);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const float* src = down_scale_src + (size_t)bn * full_n_blocks_k + down_scale_src_block_k_offset;
              float* dst = down_scale_dst + (size_t)bn * tp_n_blocks_k;
              std::memcpy(dst, src, sizeof(float) * tp_n_blocks_k);
            }
          },
          nullptr);
    });

    // Call per-TP load_weights (which loads quantized weights into BufferB, no pre-decoding)
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      tps[i]->load_weights();
    });

    // DON'T free persistent buffers - offload_engine needs them!
    // They are freed in destructor.

    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output) override {
    // Use base class merge_results
    Base::merge_results(qlen, output);
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_FP8_MOE_OFFLOAD_H