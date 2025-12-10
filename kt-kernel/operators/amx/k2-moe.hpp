/**
 * @Description  : K2 AMX MoE operator for Kimi-K2 native inference
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements K2 Int4 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * K2 weights are stored with group-wise scales (KGroup Int4).
 **/
#ifndef CPUINFER_OPERATOR_AMX_K2_MOE_H
#define CPUINFER_OPERATOR_AMX_K2_MOE_H

// #define LOAD_TIME_PROFILE

#include "moe_base.hpp"

/**
 * @brief K2 Int4 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to amx::GemmKernel224Int4SmallKGroup
 *
 * This class provides K2-specific GEMM implementations:
 * - do_gate_up_gemm: Int4 weight with KGroup scale + AMX GEMM
 * - do_down_gemm: Same Int4 KGroup GEMM
 * - load_weights: Load Int4 weights with group-wise scales
 */
template <class T = amx::GemmKernel224Int4SmallKGroup>
class AMX_K2_MOE_TP : public AMX_MOE_BASE<T, AMX_K2_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_K2_MOE_TP<T>>;
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
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_K2_MOE_TP() = default;

  AMX_K2_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("Kimi-K2 MoE only support KGroup Int4");
    }
    printf("Creating AMX_K2_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_K2_MOE_TP() = default;

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
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size,
                          ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size,
                          ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size,
                          down_ba_[expert_idx], down_bb_[expert_idx],
                          down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size,
                          down_ba_[expert_idx], down_bb_[expert_idx],
                          down_bc_[expert_idx], ith, nth);
    }
  }

  /**
   * @brief Load Int4 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj with scales
   * from config_.gate_scale, up_scale, down_scale.
   */
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("Kimi AVX MOE only support KGroup Int4.");
    }
    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("Kimi AVX MOE only support load native weight.");
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
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          // up part
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
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
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count =
              (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;

          // convert scales from BF16 to FP32
          convert_or_copy(gate_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count),
                          scale_elem_count);
          convert_or_copy(up_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.up_scale + (logical_expert_id * scale_elem_count),
                          scale_elem_count);
          convert_or_copy(down_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.down_scale + (logical_expert_id * scale_elem_count),
                          scale_elem_count);
        },
        nullptr);
#ifdef DEBUG_K2_MOE
    dump_buffer_b("native", 0, "down", down_bb_[0].get());
#endif
  }

  /**
   * @brief Reconstruct weights for all experts to the output buffers
   *
   * This function handles the TP-specific portion of the reconstruction for all experts.
   * Used for GPU offloading scenarios.
   */
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int num_experts, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    const int group_size = config.quant_config.group_size;
    auto pool = config.pool->get_subpool(tp_part_idx);

    // Calculate sizes for CPU TP part (this instance)
    size_t cpu_tp_weight_elem_count = (size_t)config.intermediate_size * config.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;  // int4 packing
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    // Calculate sizes for GPU TP part
    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;  // int4 packing
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      // Multiple CPU TPs map to one GPU TP
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      // Get pointers for this GPU TP part
      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      // Calculate offset within the GPU TP buffer
      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      int nth = 1;
      pool->do_work_stealing_job(
          nth * num_experts, nullptr,
          [&, this](int task_id) {
            int expert_id = task_id / nth;

            size_t w13_expert_base_weight = expert_id * 2 * gpu_tp_weight_bytes;
            size_t w13_expert_base_scale = expert_id * 2 * gpu_tp_scale_elem_count;
            size_t w2_expert_base_weight = expert_id * gpu_tp_weight_bytes;
            size_t w2_expert_base_scale = expert_id * gpu_tp_scale_elem_count;

            // Gate
            uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b;
            float* gate_scale_src = gate_bb_[expert_id]->d;
            std::memcpy(w13_weight_dst + w13_expert_base_weight + offset_in_gpu_weight,
                        gate_weight_src, cpu_tp_weight_bytes);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_expert_base_scale + offset_in_gpu_scale),
                            gate_scale_src, cpu_tp_scale_elem_count);

            // Up
            uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b;
            float* up_scale_src = up_bb_[expert_id]->d;
            std::memcpy(w13_weight_dst + w13_expert_base_weight + offset_in_gpu_weight + gpu_tp_weight_bytes,
                        up_weight_src, cpu_tp_weight_bytes);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_expert_base_scale + offset_in_gpu_scale + gpu_tp_scale_elem_count),
                            up_scale_src, cpu_tp_scale_elem_count);

            // Down - column-wise slicing
            for (size_t col = 0; col < config.hidden_size; col++) {
              size_t gpu_col_offset = col * ((full_config.intermediate_size / gpu_tp_count) >> 1);
              size_t cpu_col_offset = col * (config.intermediate_size >> 1);
              size_t gpu_col_slice_offset = local_idx * (config.intermediate_size >> 1);

              std::memcpy(w2_weight_dst + w2_expert_base_weight + gpu_col_offset + gpu_col_slice_offset,
                          (uint8_t*)down_bb_[expert_id]->b + cpu_col_offset,
                          config.intermediate_size / 2);

              size_t gpu_scale_col_offset = col * ((full_config.intermediate_size / gpu_tp_count) / group_size);
              size_t cpu_scale_col_offset = col * (config.intermediate_size / group_size);
              size_t gpu_scale_slice_offset = local_idx * (config.intermediate_size / group_size);

              convert_or_copy((ggml_bf16_t*)(w2_scale_dst + w2_expert_base_scale + gpu_scale_col_offset + gpu_scale_slice_offset),
                              down_bb_[expert_id]->d + cpu_scale_col_offset,
                              config.intermediate_size / group_size);
            }
          },
          nullptr);
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      pool->do_work_stealing_job(
          gpu_tps_per_cpu_tp * num_experts, nullptr,
          [&, this](int task_id) {
            int expert_id = task_id % num_experts;
            int local_gpu_idx = task_id / num_experts;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            size_t w13_gpu_expert_offset_weight = expert_id * 2 * gpu_tp_weight_bytes;
            size_t w13_gpu_expert_offset_scale = expert_id * 2 * gpu_tp_scale_elem_count;
            size_t w2_gpu_expert_offset_weight = expert_id * gpu_tp_weight_bytes;
            size_t w2_gpu_expert_offset_scale = expert_id * gpu_tp_scale_elem_count;

            // Gate
            uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight;
            float* gate_scale_src = gate_bb_[expert_id]->d + cpu_offset_scale;
            std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight,
                        gate_weight_src, data_per_gpu_tp_weight);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_gpu_expert_offset_scale),
                            gate_scale_src, data_per_gpu_tp_scale);

            // Up
            uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight;
            float* up_scale_src = up_bb_[expert_id]->d + cpu_offset_scale;
            std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight + gpu_tp_weight_bytes,
                        up_weight_src, data_per_gpu_tp_weight);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_gpu_expert_offset_scale + gpu_tp_scale_elem_count),
                            up_scale_src, data_per_gpu_tp_scale);

            // Down - column-wise slicing
            for (size_t col = 0; col < config.hidden_size; col++) {
              size_t col_offset_weight = (col * config.intermediate_size / 2) + (local_gpu_idx * data_per_gpu_tp_weight / config.hidden_size);
              size_t col_offset_scale = (col * (config.intermediate_size / group_size)) + (local_gpu_idx * data_per_gpu_tp_scale / config.hidden_size);

              std::memcpy(w2_weight_dst + w2_gpu_expert_offset_weight + (col * (config.intermediate_size / gpu_tps_per_cpu_tp) / 2),
                          (uint8_t*)down_bb_[expert_id]->b + col_offset_weight,
                          (config.intermediate_size / gpu_tps_per_cpu_tp) / 2);

              convert_or_copy((ggml_bf16_t*)(w2_scale_dst + w2_gpu_expert_offset_scale + col * ((config.intermediate_size / gpu_tps_per_cpu_tp) / group_size)),
                              down_bb_[expert_id]->d + col_offset_scale,
                              (config.intermediate_size / gpu_tps_per_cpu_tp) / group_size);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_K2_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation
// ============================================================================

template <typename K>
class TP_MOE<AMX_K2_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

#ifdef LOAD_TIME_PROFILE
    auto load_start_time = std::chrono::high_resolution_clock::now();
    auto load_last = load_start_time;
    long alloc_and_tp_slice_time = 0, tps_load_time = 0, cleanup_time = 0;
#endif

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (!use_per_expert_ptrs && config.gate_scale == nullptr) {
      throw std::runtime_error("K2 MoE only supports Packed Int4 with KGroup Scale");
    }

    if (use_per_expert_ptrs) {
      printf("From per-expert pointers (gate_projs)\n");
    } else {
      printf("From Packed Int4 with KGroup Scale\n");
    }

    int& group_size = config.quant_config.group_size;

    if (use_per_expert_ptrs) {
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

        tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.gate_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
        tpc.up_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
        tpc.down_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];

        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1),
                     (weight_elem_count >> 1));

              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1),
                     (weight_elem_count >> 1));

              memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count),
                     sizeof(ggml_bf16_t) * scales_elem_count);

              memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count),
                     sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale + (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale + (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
        printf("TP %d load weight done.\n", i);
      }
    } else {
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];

        size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

        tpc.gate_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
        tpc.up_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
        tpc.down_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];

        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (sizeof(uint8_t) * tpc.intermediate_size) >> 1);
                  memcpy((ggml_bf16_t*)tpc.down_scale + (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (ggml_bf16_t*)config.down_scale + ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                                                           col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
        printf("TP %d load weight done.\n", i);
      }
    }

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      alloc_and_tp_slice_time = std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
      load_last = load_now_time;
    }
#endif

    DO_TPS_LOAD_WEIGHTS(pool);

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      tps_load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
      load_last = load_now_time;
    }
#endif

    for (auto i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);

      delete[] (ggml_bf16_t*)(tpc.gate_scale);
      delete[] (ggml_bf16_t*)(tpc.up_scale);
      delete[] (ggml_bf16_t*)(tpc.down_scale);
    }

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      cleanup_time = std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
    }
    auto load_end_time = std::chrono::high_resolution_clock::now();
    auto load_total_time = std::chrono::duration_cast<std::chrono::microseconds>(load_end_time - load_start_time).count();
    printf(
        "[K2 MoE Load Weights] tp_count: %d, alloc_and_tp_slice: %ld us, tps_load_weights: %ld us, cleanup: %ld us, total: %ld us\n",
        tp_count, alloc_and_tp_slice_time, tps_load_time, cleanup_time, load_total_time);
#endif

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

    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count) {
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

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>>
};

#endif  // CPUINFER_OPERATOR_AMX_K2_MOE_H
