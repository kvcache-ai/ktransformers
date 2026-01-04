/**
 * @Description  : NVFP4 MoE operator for CPU inference
 * @Author       : Claude & KVCache.AI Team
 * @Date         : 2025-01-17
 * @Version      : 0.1.0
 * @Copyright (c) 2025 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_NVFP4_MOE_H
#define CPUINFER_OPERATOR_AMX_NVFP4_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "la/amx.hpp"
#include "la/nvfp4_kernel.hpp"
#include "llama.cpp/ggml.h"

template <class T>
class NVFP4_MOE_TP {
 private:
  int tp_part_idx = 0;

  void* gate_proj_ = nullptr;  // [expert_num * intermediate_size * hidden_size / 2] (packed FP4)
  void* up_proj_ = nullptr;    // [expert_num * intermediate_size * hidden_size / 2] (packed FP4)
  void* down_proj_ = nullptr;  // [expert_num * hidden_size * intermediate_size / 2] (packed FP4)

  void* gate_scale_ = nullptr;  // FP8 E4M3 block scales for gate
  void* up_scale_ = nullptr;    // FP8 E4M3 block scales for up
  void* down_scale_ = nullptr;  // FP8 E4M3 block scales for down

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

  std::vector<std::shared_ptr<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>> gate_up_ba_;
  std::vector<std::shared_ptr<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>> gate_bb_;
  std::vector<std::shared_ptr<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>> gate_bc_;
  std::vector<std::shared_ptr<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>> up_bb_;
  std::vector<std::shared_ptr<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>> up_bc_;
  std::vector<std::shared_ptr<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>> down_ba_;
  std::vector<std::shared_ptr<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>> down_bb_;
  std::vector<std::shared_ptr<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>> down_bc_;

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

 public:
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = 0.5625;  // FP4: 4 bits + scale overhead
  static constexpr int BLOCK_SIZE = 16;           // NVFP4 fixed block size

  NVFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_) {
    auto& quant_config = config.quant_config;

    if (quant_config.group_size != BLOCK_SIZE) {
      printf("Warning: NVFP4 requires group_size=16, but got %d. Forcing to 16.\n", quant_config.group_size);
      quant_config.group_size = BLOCK_SIZE;
    }

    if (quant_config.zero_point) {
      throw std::runtime_error("NVFP4 MoE does not support zero-point quantization");
    }

    printf("Creating NVFP4_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));

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

    // Initialize buffers for each expert
    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr = std::aligned_alloc(64, nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(
                                                     config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr = std::aligned_alloc(64, nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(
                                                   config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr = std::aligned_alloc(64, nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(
                                                     config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(
          config_.hidden_size, config_.intermediate_size, down_bb_ptr));
    }

    assert(nvfp4::GemmKernelNVFP4::M_STEP == 32);  // Ensure M_STEP matches our kernel design
    pool_count_ = config_.max_len * config_.num_experts_per_tok + config_.expert_num * nvfp4::GemmKernelNVFP4::M_STEP;

    gate_up_ba_pool_bytes_ =
        (nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(pool_count_, config_.hidden_size)) +
        pool_count_ * 64;
    gate_bc_pool_bytes_ =
        (nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(pool_count_, config_.intermediate_size)) +
        pool_count_ * 64;
    up_bc_pool_bytes_ =
        (nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(pool_count_, config_.intermediate_size)) +
        pool_count_ * 64;
    down_ba_pool_bytes_ =
        (nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(pool_count_, config_.intermediate_size)) +
        pool_count_ * 64;
    down_bc_pool_bytes_ =
        (nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(pool_count_, config_.hidden_size)) +
        pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~NVFP4_MOE_TP() = default;

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("NVFP4 MoE requires FP8 E4M3 scale tensors");
    }

    gate_scale_ = config_.gate_scale;
    up_scale_ = config_.up_scale;
    down_scale_ = config_.down_scale;

    // Load weights - copy packed FP4 data and FP8 scales
    int nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.intermediate_size);

    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Calculate sizes
          size_t weight_elements = config_.intermediate_size * config_.hidden_size;
          size_t scale_elements = weight_elements / BLOCK_SIZE;

          // Load gate weights and scales
          gate_bb_[expert_idx]->from_raw_nvfp4(
              (uint8_t*)config_.gate_proj + ((logical_expert_id * weight_elements) >> 1),
              (uint8_t*)gate_scale_ + (logical_expert_id * scale_elements),
              1.0f,  // Tensor scale (should be extracted from config if available)
              ith, nth);

          // Load up weights and scales
          up_bb_[expert_idx]->from_raw_nvfp4((uint8_t*)config_.up_proj + ((logical_expert_id * weight_elements) >> 1),
                                             (uint8_t*)up_scale_ + (logical_expert_id * scale_elements), 1.0f, ith,
                                             nth);
        },
        nullptr);

    nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_elements = config_.hidden_size * config_.intermediate_size;
          size_t scale_elements = weight_elements / BLOCK_SIZE;

          // Load down weights and scales
          down_bb_[expert_idx]->from_raw_nvfp4(
              (uint8_t*)config_.down_proj + ((logical_expert_id * weight_elements) >> 1),
              (uint8_t*)down_scale_ + (logical_expert_id * scale_elements), 1.0f, ith, nth);
        },
        nullptr);
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

    // Count tokens per expert
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
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    // Setup buffer pointers
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = nvfp4::GemmKernelNVFP4::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) continue;

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size =
          align64(nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size =
          align64(nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);
    }

    // Copy input data
    DIRECT_OR_POOL_BY_QLEN(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

    // Quantize input for each expert (using BF16 → NVFP4)
    DIRECT_OR_POOL_BY_QLEN(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_bf16(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    // Compute gate and up projections
    int nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { nvfp4::GemmKernelNVFP4::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          if (do_up) {
            nvfp4::nvfp4_matmul_opt4(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                     gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            nvfp4::nvfp4_matmul_opt4(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                     gate_up_ba_[expert_idx], gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth);
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    // Apply SiLU activation: gate / (1 + exp(-gate)) * up
    auto act_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = nvfp4::GemmKernelNVFP4::split_range_n(config_.intermediate_size, ith, nth);

      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];

        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);

          // SiLU: x / (1 + exp(-x)) * up
          __m512 result0 = amx::act_fn(gate_val0, up_val0);
          __m512 result1 = amx::act_fn(gate_val1, up_val1);

          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
        }
      }
    };
    DIRECT_OR_POOL_BY_QLEN(nth * activated_expert, act_fn);

    // Quantize intermediate results for down projection (BF16 → NVFP4)
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_bf16(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Compute down projection
    nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { nvfp4::GemmKernelNVFP4::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          nvfp4::nvfp4_matmul_opt4(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                                   down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Combine expert outputs with weights
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
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Count activated experts and setup mappings
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);

    for (int i = 0; i < k; i++) {
      if (expert_ids[i] < config_.num_gpu_experts || expert_ids[i] >= config_.expert_num) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    if (activated_expert == 0) {
      // No experts activated, zero output
      memset(output, 0, sizeof(float) * config_.hidden_size);
      return;
    }

    // Setup buffer pointers
    size_t offset = 0;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    constexpr size_t M_STEP = nvfp4::GemmKernelNVFP4::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;

      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size =
          align64(nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size =
          align64(nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);
    }

    // Quantize input for each activated expert
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size =
          align64(nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      gate_up_ba_[expert_idx]->from_bf16(qlen, (ggml_bf16_t*)input, 0, 1);
    }

    // Compute gate and up projections
    int nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { nvfp4::GemmKernelNVFP4::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          if (do_up) {
            nvfp4::nvfp4_matmul_opt4(qlen, config_.intermediate_size, config_.hidden_size, gate_up_ba_[expert_idx],
                                     up_bb_[expert_idx], up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            nvfp4::nvfp4_matmul_opt4(qlen, config_.intermediate_size, config_.hidden_size, gate_up_ba_[expert_idx],
                                     gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth);
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    // Apply SiLU activation: gate * sigmoid(gate) * up
    auto act_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = nvfp4::GemmKernelNVFP4::split_range_n(config_.intermediate_size, ith, nth);

      ggml_bf16_t* gate_output_ptr = m_local_gate_output_ptr_[expert_idx];
      ggml_bf16_t* up_output_ptr = m_local_up_output_ptr_[expert_idx];

      for (int j = n_start; j < n_end; j += 32) {
        __m512 gate_val0, gate_val1, up_val0, up_val1;
        avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
        avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);

        // SiLU: x / (1 + exp(-x)) * up
        __m512 result0 = amx::act_fn(gate_val0, up_val0);
        __m512 result1 = amx::act_fn(gate_val1, up_val1);

        avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
      }
    };

    for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
      act_fn(task_id);
    }

    // Quantize intermediate for down projection
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_bf16(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Compute down projection
    nth = nvfp4::GemmKernelNVFP4::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { nvfp4::GemmKernelNVFP4::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          nvfp4::nvfp4_matmul_opt4(qlen, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                                   down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Combine expert outputs with weights
    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();

      for (int j = 0; j < k; j++) {
        if (expert_ids[j] < config_.num_gpu_experts || expert_ids[j] >= config_.expert_num) {
          continue;
        }

        __m512 weight = _mm512_set1_ps(weights[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32(
            (__m512i*)(m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e),
            &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }

      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }
  }
};

// Template specialization for TP_MOE
template <typename K>
class TP_MOE<NVFP4_MOE_TP<K>> : public TP_MOE_Common<NVFP4_MOE_TP<K>> {
 public:
  using TP_MOE_Common<NVFP4_MOE_TP<K>>::TP_MOE_Common;

  void load_weights() {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    // Check for per-expert pointers or contiguous memory
    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (!use_per_expert_ptrs && config.gate_scale == nullptr) {
      throw std::runtime_error("NVFP4 MoE requires packed FP4 with FP8 E4M3 scales");
    }

    constexpr int BLOCK_SIZE = 16;  // NVFP4 fixed block size

    // Load weights using TP slicing (similar to k2-moe.hpp)
    for (auto i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];

      size_t scales_elem_count = weight_elem_count / BLOCK_SIZE;

      tpc.gate_scale = new uint8_t[(tpc.expert_num * scales_elem_count)];  // FP8 scales
      tpc.up_scale = new uint8_t[(tpc.expert_num * scales_elem_count)];
      tpc.down_scale = new uint8_t[(tpc.expert_num * scales_elem_count)];

      // Copy weights and scales
      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            // TP-slicing for gate and up (similar to k2-moe.hpp)
            memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                   (uint8_t*)config.gate_proj +
                       ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                   ((sizeof(uint8_t) * weight_elem_count) >> 1));

            memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                   (uint8_t*)config.up_proj +
                       ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                   ((sizeof(uint8_t) * weight_elem_count) >> 1));

            memcpy(
                (uint8_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                (uint8_t*)config.gate_scale +
                    (expert_id * (config.hidden_size * config.intermediate_size / BLOCK_SIZE) + i * scales_elem_count),
                sizeof(uint8_t) * scales_elem_count);

            memcpy(
                (uint8_t*)tpc.up_scale + (expert_id * scales_elem_count),
                (uint8_t*)config.up_scale +
                    (expert_id * (config.hidden_size * config.intermediate_size / BLOCK_SIZE) + i * scales_elem_count),
                sizeof(uint8_t) * scales_elem_count);

            // TP-slicing for down (by column)
            for (size_t col = 0; col < config.hidden_size; col++) {
              memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                     (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                    col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                   1),
                     (sizeof(uint8_t) * tpc.intermediate_size) >> 1);

              size_t col_scales = tpc.intermediate_size / BLOCK_SIZE;
              memcpy((uint8_t*)tpc.down_scale + (expert_id * scales_elem_count + col * col_scales),
                     (uint8_t*)config.down_scale +
                         ((expert_id * (config.intermediate_size * config.hidden_size / BLOCK_SIZE)) +
                          col * (config.intermediate_size / BLOCK_SIZE) + i * col_scales),
                     sizeof(uint8_t) * col_scales);
            }
          },
          nullptr);
    }

    DO_TPS_LOAD_WEIGHTS(pool);

    // Cleanup temporary buffers
    for (auto i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);
      delete[] (uint8_t*)(tpc.gate_scale);
      delete[] (uint8_t*)(tpc.up_scale);
      delete[] (uint8_t*)(tpc.down_scale);
    }

    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;

      // If incremental, add from existing output (BF16) to merge_to
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }

      // Sum results from all TP parts (each produces partial sum due to intermediate_size slicing)
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }

      // Convert to BF16 and store in output
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };

    auto pool = config.pool;
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_NVFP4_MOE_H
