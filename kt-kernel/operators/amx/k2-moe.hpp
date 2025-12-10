/**
 * @Description  : Skeleton for K2 AMX MoE operator.
 * @Author       : Codex
 * @Date         : 2024-07-22
 * @Version      : 0.1.0
 * @LastEditors  : Codex
 * @LastEditTime : 2024-07-22
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_K2_MOE_H
#define CPUINFER_OPERATOR_AMX_K2_MOE_H

// #define DEBUG_K2_MOE

#include <cstddef>
#include <cstdint>
#include <cstring>
// #define FORWARD_TIME_PROFILE
#define LOAD_TIME_PROFILE

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "la/amx.hpp"
#include "llama.cpp/ggml.h"

template <class T>
class AMX_K2_MOE_TP {
 private:
  int tp_part_idx = 0;

  void* gate_proj_ = nullptr;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void* up_proj_ = nullptr;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void* down_proj_ = nullptr;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

  ggml_bf16_t* m_local_input_ = nullptr;        // [num_experts_per_tok * max_len * hidden_size]
  ggml_bf16_t* m_local_gate_output_ = nullptr;  // [num_experts_per_tok * max_len * intermediate_size]
  ggml_bf16_t* m_local_up_output_ = nullptr;    // [num_experts_per_tok * max_len * intermediate_size]
  ggml_bf16_t* m_local_down_output_ = nullptr;  // [num_experts_per_tok * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [max_len, num_experts_per_tok]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<ggml_bf16_t*> m_local_input_ptr_;        // [expert_num]
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;  // [expert_num]
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;    // [expert_num]
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;  // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;  // rows reserved in each scratch pool
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;
#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
#endif

#ifdef CHECK
  inline void load_check() {
    // TODO: implement load_check for verification.
  }

  void verify_load_right() {
    // TODO: implement verification helpers.
  }
#endif

  inline void dump_buffer_b(const std::string &quantization_type, int expert_idx, const std::string &matrix_type,
                            typename T::BufferB *buffer) {
    auto &quant_config = config_.quant_config;
    int &group_size = quant_config.group_size;

    printf("[DUMP_BUFFER_B] TP%d %s Expert%d %s:\n", tp_part_idx, quantization_type.c_str(), expert_idx,
           matrix_type.c_str());

    // Calculate dimensions based on matrix type
    int rows, cols, num_groups;
    size_t scale_elem_count;
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    } else { // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    }

    // Dump scales (as float)
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
    // Dump quantized weights (as hex uint8)
    size_t weight_size = (rows * cols) / 2; // INT4 packed
    uint8_t *weight_ptr = (uint8_t *)buffer->b;

    printf("  Weights[first 32 bytes]: ");
    for (int i = 0; i < std::min(32, (int)weight_size); i++) {
      printf("%02x ", weight_ptr[i]);
    }
    printf("\n");

    if (weight_size > 32) {
      printf("  Weights[last 32 bytes]: ");
      int start_idx = std::max(32, (int)weight_size - 32);
      for (int i = start_idx; i < (int)weight_size; i++) {
        printf("%02x ", weight_ptr[i]);
      }
      printf("\n");
    }

    printf("  Matrix dimensions: %dx%d, Groups: %d, Group size: %d, Scale elements: %zu\n", rows, cols, num_groups,
           group_size, scale_elem_count);
    printf("\n");
    fflush(stdout);
  }

 public:
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_K2_MOE_TP(GeneralMOEConfig config, int tp_part_idx_) {
    auto& quant_config = config.quant_config;
    int& group_size = quant_config.group_size;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("Kimi-K2 MoE only support KGroup Int4");
    }
    printf("Creating AMX_K2_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));
    auto& load = config.load;
    auto& save = config.save;
    if (load && config.path == "") {
      load = false;
    }

    this->tp_part_idx = tp_part_idx_;
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    MemoryRequest mem_requests;
    mem_requests.append_pointer(
        &m_local_input_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                         config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.hidden_size);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, group_size, nullptr));
      gate_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, group_size, nullptr));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, group_size));
      gate_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size,
                                                               group_size, gate_bb_ptr));

      void* up_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, group_size));
      up_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, group_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, group_size));
      down_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size,
                                                               group_size, down_bb_ptr));
    }
    assert(T::BufferA::M_STEP == T::BufferC::M_STEP);
    // TODO: need update to all *.hpp
    // (config_.expert_num * T::BufferA::M_STEP) in pool_count_ is to ensure padding for each experts.
    pool_count_ = config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::BufferA::M_STEP;

    gate_up_ba_pool_bytes_ = (T::BufferA::required_size(pool_count_, config_.hidden_size, group_size)) + pool_count_ * 64;
    gate_bc_pool_bytes_ = (T::BufferC::required_size(pool_count_, config_.intermediate_size)) + pool_count_ * 64;
    up_bc_pool_bytes_ = (T::BufferC::required_size(pool_count_, config_.intermediate_size)) + pool_count_ * 64;
    down_ba_pool_bytes_ = (T::BufferA::required_size(pool_count_, config_.intermediate_size, group_size)) + pool_count_ * 64;
    down_bc_pool_bytes_ = (T::BufferC::required_size(pool_count_, config_.hidden_size)) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AMX_K2_MOE_TP() = default;

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
    // dump_buffer_b("native", 0, "down", down_bb_[0].get());
  }

  // Reconstruct weights for all experts to the output buffers
  // This function handles the TP-specific portion of the reconstruction for all experts
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int num_experts, const GeneralMOEConfig& full_config,
                                const std::vector<uintptr_t>& w13_weight_ptrs,
                                const std::vector<uintptr_t>& w13_scale_ptrs,
                                const std::vector<uintptr_t>& w2_weight_ptrs,
                                const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Calculate sizes for CPU TP part (this instance)
    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;  // int4 packing
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    // Calculate sizes for GPU TP part
    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;  // int4 packing
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    // Determine mapping: which GPU TP parts should this CPU TP part write to?
    // Since weights are col-major and we slice directly by memory order:
    // - If cpu_tp_count >= gpu_tp_count: multiple(or one) CPU TPs write to one GPU TP
    // - If cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
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

      // Process only the first num_experts experts (GPU experts)
      int nth = T::recommended_nth(config_.intermediate_size);
      nth = 1;
      pool->do_work_stealing_job(
          nth * num_experts, nullptr,
          [&, this](int task_id) {
            int expert_id = task_id / nth;
            // int ith = task_id % nth;
            // auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);

            // Calculate base offsets for this expert in the GPU buffers
            // For w13: each expert has gate+up, so the offset needs to account for 2x size
            size_t w13_expert_base_weight = expert_id * 2 * gpu_tp_weight_bytes;
            size_t w13_expert_base_scale = expert_id * 2 * gpu_tp_scale_elem_count;
            size_t w2_expert_base_weight = expert_id * gpu_tp_weight_bytes;
            size_t w2_expert_base_scale = expert_id * gpu_tp_scale_elem_count;

            // Gate (first part of w13 for this expert)
            uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b;
            float* gate_scale_src = gate_bb_[expert_id]->d;
            std::memcpy(w13_weight_dst + w13_expert_base_weight + offset_in_gpu_weight,
                       gate_weight_src, cpu_tp_weight_bytes);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_expert_base_scale + offset_in_gpu_scale),
                           gate_scale_src, cpu_tp_scale_elem_count);

            // Up (second part of w13 for this expert, immediately after gate)
            uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b;
            float* up_scale_src = up_bb_[expert_id]->d;
            std::memcpy(w13_weight_dst + w13_expert_base_weight + offset_in_gpu_weight + gpu_tp_weight_bytes,
                       up_weight_src, cpu_tp_weight_bytes);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_expert_base_scale + offset_in_gpu_scale + gpu_tp_scale_elem_count),
                           up_scale_src, cpu_tp_scale_elem_count);

            // Down (w2) - need to handle column-wise slicing
            // The down matrix is transposed compared to gate/up, so we need to extract by columns
            // When multiple CPU TPs map to one GPU TP, each CPU TP has a slice of intermediate dimension
            // CPU TP internal layout: each column has config_.intermediate_size elements
            // GPU expects: each column has full_config.intermediate_size elements
            size_t cpu_tps_per_gpu = cpu_tp_count / gpu_tp_count;

            for (size_t col = 0; col < config_.hidden_size; col++) {
              // GPU buffer column width is full_config.intermediate_size / gpu_tp_count
              size_t gpu_col_offset = col * ((full_config.intermediate_size / gpu_tp_count) >> 1);
              size_t cpu_col_offset = col * (config_.intermediate_size >> 1);
              size_t gpu_col_slice_offset = local_idx * (config_.intermediate_size >> 1);

              std::memcpy(w2_weight_dst + w2_expert_base_weight + gpu_col_offset + gpu_col_slice_offset,
                         (uint8_t*)down_bb_[expert_id]->b + cpu_col_offset,
                         config_.intermediate_size / 2);

              // Same for scales
              size_t gpu_scale_col_offset = col * ((full_config.intermediate_size / gpu_tp_count) / group_size);
              size_t cpu_scale_col_offset = col * (config_.intermediate_size / group_size);
              size_t gpu_scale_slice_offset = local_idx * (config_.intermediate_size / group_size);

              convert_or_copy((ggml_bf16_t*)(w2_scale_dst + w2_expert_base_scale + gpu_scale_col_offset + gpu_scale_slice_offset),
                             down_bb_[expert_id]->d + cpu_scale_col_offset,
                             config_.intermediate_size / group_size);
            }
          },
          nullptr);
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
      // Each CPU TP part contains data for multiple GPU TP parts
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;

      // This CPU TP part writes to GPU TP indices: [start_gpu_tp, start_gpu_tp + gpu_tps_per_cpu_tp)
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      // Size of data per GPU TP within this CPU TP
      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      // Process all experts for this GPU TP
      pool->do_work_stealing_job(
          gpu_tps_per_cpu_tp * num_experts, nullptr,
          [&, this](int task_id) {
            int expert_id = task_id % num_experts;
            int local_gpu_idx = task_id / num_experts;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            // Get pointers for this GPU TP part
            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            // Calculate offsets within CPU TP buffers
            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            // Calculate offsets for this expert in GPU buffers
            // For w13: each expert has gate+up, so the offset needs to account for 2x size
            size_t w13_gpu_expert_offset_weight = expert_id * 2 * gpu_tp_weight_bytes;
            size_t w13_gpu_expert_offset_scale = expert_id * 2 * gpu_tp_scale_elem_count;
            size_t w2_gpu_expert_offset_weight = expert_id * gpu_tp_weight_bytes;
            size_t w2_gpu_expert_offset_scale = expert_id * gpu_tp_scale_elem_count;

            // Gate (first part of w13 for this expert)
            uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight;
            float* gate_scale_src = gate_bb_[expert_id]->d + cpu_offset_scale;
            std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight,
                        gate_weight_src, data_per_gpu_tp_weight);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_gpu_expert_offset_scale),
                            gate_scale_src, data_per_gpu_tp_scale);

            // Up (second part of w13 for this expert, immediately after gate)
            uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight;
            float* up_scale_src = up_bb_[expert_id]->d + cpu_offset_scale;
            std::memcpy(w13_weight_dst + w13_gpu_expert_offset_weight + gpu_tp_weight_bytes,
                        up_weight_src, data_per_gpu_tp_weight);
            convert_or_copy((ggml_bf16_t*)(w13_scale_dst + w13_gpu_expert_offset_scale + gpu_tp_scale_elem_count),
                            up_scale_src, data_per_gpu_tp_scale);

            // Down (w2) - need to handle column-wise slicing
            // The down matrix is transposed compared to gate/up, so we need to extract by columns
            for (size_t col = 0; col < config_.hidden_size; col++) {
              // Calculate the offset within the column for this GPU TP part
              size_t col_offset_weight = (col * config_.intermediate_size / 2) + (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
              size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) + (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

              // Copy weights column by column
              std::memcpy(w2_weight_dst + w2_gpu_expert_offset_weight + (col * (config_.intermediate_size / gpu_tps_per_cpu_tp) / 2),
                          (uint8_t*)down_bb_[expert_id]->b + col_offset_weight,
                          (config_.intermediate_size / gpu_tps_per_cpu_tp) / 2);

              // Copy scales column by column
              convert_or_copy((ggml_bf16_t*)(w2_scale_dst + w2_gpu_expert_offset_scale + col * ((config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size)),
                              down_bb_[expert_id]->d + col_offset_scale,
                              (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size);
            }
          },
          nullptr);
    }
  }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

#ifndef DIRECT_OR_POOL_BY_QLEN
#define DIRECT_OR_POOL_BY_QLEN(var, fn)                          \
  do {                                                           \
    if (qlen < 10) {                                             \
      for (int i = 0; i < (var); i++) {                          \
        (fn)(i);                                                 \
      }                                                          \
    } else {                                                     \
      pool->do_work_stealing_job((var), nullptr, (fn), nullptr); \
    }                                                            \
  } while (0)
#endif

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    // activated_expert 已经统计完成

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::BufferA::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0)
        continue;
      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(T::BufferA::required_size(max_m, config_.hidden_size, group_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(T::BufferC::required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);
      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(T::BufferC::required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);
      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(T::BufferA::required_size(max_m, config_.intermediate_size, group_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);
      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(T::BufferC::required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);
      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_a <= gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    DIRECT_OR_POOL_BY_QLEN(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    DIRECT_OR_POOL_BY_QLEN(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int& group_size = config_.quant_config.group_size;
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          if (do_up) {
            MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                            group_size, gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx],
                                            ith, nth);
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                            group_size, gate_up_ba_[expert_idx], gate_bb_[expert_idx],
                                            gate_bc_[expert_idx], ith, nth);
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    auto up_gate_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0);
          __m512 result1 = amx::act_fn(gate_val1, up_val1);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
        }
      }
    };
    DIRECT_OR_POOL_BY_QLEN(nth * activated_expert, up_gate_fn);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int& group_size = config_.quant_config.group_size;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                                          group_size, down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx],
                                          ith, nth);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, nth, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i*)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                   m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
    // for (int i = 0; i < qlen; i ++)
    //   forward_decode(k, expert_ids + i * k, weights + i * k, (ggml_bf16_t*)input + i * config_.hidden_size, (float*)output + i * config_.hidden_size);
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < k; i++) {
      if (expert_ids[i] < config_.num_gpu_experts || expert_ids[i] >= config_.expert_num) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      activated_expert++;
    }
    
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::BufferA::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(T::BufferC::required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(T::BufferC::required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(T::BufferA::required_size(max_m, config_.intermediate_size, group_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(T::BufferC::required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

    gate_up_ba_[0]->max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
    gate_up_ba_[0]->set_data(gate_up_ba_pool_);
    gate_up_ba_[0]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    // calc gate & up
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int& group_size = config_.quant_config.group_size;
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          if (do_up) {
            amx::vec_mul_kgroup(qlen, config_.intermediate_size, config_.hidden_size, group_size, gate_up_ba_[0],
                                up_bb_[expert_idx], up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            amx::vec_mul_kgroup(qlen, config_.intermediate_size, config_.hidden_size, group_size, gate_up_ba_[0],
                                gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth);
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
      }
    },
    nullptr);

#ifdef DEBUG_K2_MOE
    if (activated_expert > 0) {
      int print_elems = std::min(config_.intermediate_size, 16);
      for (int dbg = 0; dbg < activated_expert; ++dbg) {
        int sample_expert = m_expert_id_map_[dbg];
        ggml_bf16_t* gate_ptr = m_local_gate_output_ptr_[sample_expert];
        if (gate_ptr == nullptr) {
          continue;
        }

        printf("[K2][TP %d] gate_out (expert %d, first %d elems): ", tp_part_idx, sample_expert, print_elems);
        for (int idx = 0; idx < print_elems; idx++) {
          float val = ggml_bf16_to_fp32(gate_ptr[idx]);
          printf("%.6f ", val);
        }
        printf("\n");

        int tail_start = config_.intermediate_size > print_elems ? config_.intermediate_size - print_elems : 0;
        printf("[K2][TP %d] gate_out (expert %d, last %d elems): ", tp_part_idx, sample_expert, print_elems);
        for (int idx = 0; idx < print_elems; idx++) {
          float val = ggml_bf16_to_fp32(gate_ptr[tail_start + idx]);
          printf("%.6f ", val);
        }
        printf("\n");
      }
    }
#endif

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    // act
    for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < qlen; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0);
          __m512 result1 = amx::act_fn(gate_val1, up_val1);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
        }
      }
    }

    
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    // quant, get down a
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    // * down
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int& group_size = config_.quant_config.group_size;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          amx::vec_mul_kgroup(qlen, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                              down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

#ifdef DEBUG_K2_MOE
    if (activated_expert > 0) {
      int print_elems = std::min(config_.hidden_size, 16);
      for (int dbg = 0; dbg < activated_expert; ++dbg) {
        int sample_expert = m_expert_id_map_[dbg];
        ggml_bf16_t* down_ptr = m_local_down_output_ptr_[sample_expert];
        if (down_ptr == nullptr) {
          continue;
        }

        printf("[K2][TP %d] down_out (expert %d, first %d elems): ", tp_part_idx, sample_expert, print_elems);
        for (int idx = 0; idx < print_elems; idx++) {
          float val = ggml_bf16_to_fp32(down_ptr[idx]);
          printf("%.6f ", val);
        }
        printf("\n");

        int tail_start = config_.hidden_size > print_elems ? config_.hidden_size - print_elems : 0;
        printf("[K2][TP %d] down_out (expert %d, last %d elems): ", tp_part_idx, sample_expert, print_elems);
        for (int idx = 0; idx < print_elems; idx++) {
          float val = ggml_bf16_to_fp32(down_ptr[tail_start + idx]);
          printf("%.6f ", val);
        }
        printf("\n");
      }
    }
#endif

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    // get output
    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (expert_ids[j] < config_.num_gpu_experts || expert_ids[j] >= config_.expert_num) {
          continue;
        }
        __m512 weight = _mm512_set1_ps(weights[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32((__m512i*)(m_local_down_output_ptr_[expert_ids[j]] +
                                              m_local_pos_[0][j] * config_.hidden_size + e),
                                  &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }
      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }
};

template <typename K>
class TP_MOE<AMX_K2_MOE_TP<K>> : public TP_MOE_Common<AMX_K2_MOE_TP<K>> {
 public:
  using TP_MOE_Common<AMX_K2_MOE_TP<K>>::TP_MOE_Common;

  void load_weights() {
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

    // Check if using per-expert pointers (gate_projs) or contiguous memory (gate_proj + gate_scale)
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
      // Load from per-expert pointers - no need to allocate intermediate buffers
      // gate_projs[numa_id][expert_id] -> pointer to expert weight
      // For RAWINT4, numa dimension is 1 (index 0)
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

        // Allocate per-TP buffers
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

              // Source pointers from per-expert pointer arrays
              // gate_projs[0][expert_id] since numa dimension is 1
              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              // TP-slicing for gate and up (row-major slicing)
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

              // TP-slicing for down (by column)
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
      // Original path: load from contiguous memory with gate_proj/gate_scale
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

        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) { // weight and scale are all in col majored.
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                // weight and scale TP-slicing for gate and up
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

                // weight and scale TP-slicing for down (by column)
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

    // Validate input vector sizes
    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

    auto& config = this->config;
    auto pool = config.pool;
    // Each TP part writes to its corresponding buffer
    pool->dispense_backend()->do_numa_job([this, pool, gpu_tp_count, gpu_experts_num,
      w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs](int numa_id) {
      // Note: w13 combines gate and up projections
      // Split w13 pointers for gate and up
      this->tps[numa_id]->write_weights_to_buffer(
          gpu_tp_count, this->tp_count,
          gpu_experts_num, this->config,
          w13_weight_ptrs, w13_scale_ptrs, //gate + up use w13
          w2_weight_ptrs, w2_scale_ptrs);    // down uses w2
    });
  }

  void merge_results(int qlen, void* output, bool incremental) {
    auto pool = this->config.pool;
    auto merge_fn = [this, output, incremental](int token_nth) {
      auto& local_output_numa = this->local_output_numa;
      auto& tp_configs = this->tp_configs;
      auto& tp_count = this->tp_count;
      auto& config = this->config;
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };
    for (int i = 0; i < qlen; i++) {
      merge_fn(i);
    }
  }

  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_K2_MOE_H
