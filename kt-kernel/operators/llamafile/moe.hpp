#ifndef LLAMAFILE_MOE_HPP
#define LLAMAFILE_MOE_HPP
#ifdef FORWARD_TIME_PROFILE
#include <fmt/format.h>
#endif
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../moe-tp.hpp"
#include "conversion.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

inline void debug_quant(void* input, ggml_type type) {
  std::vector<float> output(ggml_blck_size(type));
  to_float(input, output.data(), ggml_blck_size(type), type);
  for (size_t i = 0; i < 10; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
}

class LLAMA_MOE_TP {
 private:
  GeneralMOEConfig config_;
  int tp_part_idx;

  uint8_t* m_local_gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  uint8_t* m_local_up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  uint8_t* m_local_down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

  float* s_input_fp32_;    // [hidden_size]
  uint8_t* s_gate_input_;  // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* s_up_input_;    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
  std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
  std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
  std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size *
                                             // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                             // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
  float* s_output_fp32_;                     // [hidden_size]

  std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
  std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size *
                                        // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                        // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  std::vector<uint8_t*>
      m_up_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type)
                    // / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  uint8_t* m_local_gate_input_;        // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* m_local_up_input_;          // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  float* m_local_gate_output_;         // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_up_output_;           // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_intermediate_fp32_;   // [routed_expert_num * group_max_len * intermediate_size]
  uint8_t* m_local_down_input_;        // [routed_expert_num * group_max_len * intermediate_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  float* m_local_down_output_;         // [routed_expert_num * group_max_len * hidden_size]
  std::vector<float*> m_output_fp32_;  // [group_max_len, hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
  std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
  std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
  std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
  std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
  std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
  std::vector<float*> m_local_down_output_ptr_;        // [expert_num]
 public:
  using input_t = ggml_bf16_t;
  using output_t = float;

  LLAMA_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : config_(config), tp_part_idx(tp_part_idx) {
    MemoryRequest mem_requests;
    mem_requests.append_pointer(&s_input_fp32_, sizeof(float) * config_.hidden_size);
    mem_requests.append_pointer(
        &s_gate_input_, config_.hidden_size *
                            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
    mem_requests.append_pointer(
        &s_up_input_, config_.hidden_size *
                          ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                          ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    s_gate_output_.resize(config_.num_experts_per_tok);
    s_up_output_.resize(config_.num_experts_per_tok);
    s_intermediate_fp32_.resize(config_.num_experts_per_tok);
    s_down_input_.resize(config_.num_experts_per_tok);
    s_down_output_.resize(config_.num_experts_per_tok);
    for (int i = 0; i < config_.num_experts_per_tok; i++) {
      mem_requests.append_pointer(&s_gate_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_up_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(
          &s_down_input_[i],
          config_.intermediate_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type));
      mem_requests.append_pointer(&s_down_output_[i], sizeof(float) * config_.hidden_size);
    }
    mem_requests.append_pointer(&s_output_fp32_, sizeof(float) * config_.hidden_size);
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, mem_requests);

    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_input_fp32_[i], sizeof(float) * config_.hidden_size);
      mem_requests.append_pointer(
          &m_gate_input_[i],
          config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
      mem_requests.append_pointer(
          &m_up_input_[i], config_.hidden_size *
                               ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                               ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    }
    mem_requests.append_pointer(
        &m_local_gate_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
    mem_requests.append_pointer(
        &m_local_up_input_, config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
                                ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                                ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(float) * config_.num_experts_per_tok *
                                                         config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_intermediate_fp32_, sizeof(float) * config_.num_experts_per_tok *
                                                                 config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(
        &m_local_down_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.intermediate_size *
            ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
            ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type));
    mem_requests.append_pointer(&m_local_down_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.hidden_size);
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_output_fp32_[i], sizeof(float) * config_.hidden_size);
    }
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    auto size = 1ll * config.expert_num * config.intermediate_size * config.hidden_size;
    m_local_up_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type)];

    m_local_gate_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.gate_type) / ggml_blck_size((ggml_type)config.gate_type)];
    m_local_down_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.down_type) / ggml_blck_size((ggml_type)config.down_type)];
  }

  void load_weights(int complete_intermediate_size, int offset) {
    auto local_gate_proj = m_local_gate_proj_;
    auto local_up_proj = m_local_up_proj_;
    auto local_down_proj = m_local_down_proj_;
    auto& config = config_;
    // printf("gate load weights:");
    // debug_quant(config.gate_proj, (ggml_type)config.gate_type);
    // we need to make sure the blck size is correct for size.
    if (config.intermediate_size % ggml_blck_size((ggml_type)config.down_type) != 0) {
      printf("intermediate_size: %d, down_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.down_type));
      throw std::runtime_error("intermediate_size must be a multiple of gate_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.up_type) != 0) {
      printf("intermediate_size: %d, up_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.up_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of up_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.gate_type) != 0) {
      printf("intermediate_size: %d, gate_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.gate_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of gate_type blck size");
    }
    uint8_t* gate_proj = (uint8_t*)config.gate_proj + offset * config.hidden_size *
                                                          ggml_type_size((ggml_type)config.gate_type) /
                                                          ggml_blck_size((ggml_type)config.gate_type);
    uint8_t* up_proj = (uint8_t*)config.up_proj + offset * config.hidden_size *
                                                      ggml_type_size((ggml_type)config.up_type) /
                                                      ggml_blck_size((ggml_type)config.up_type);
    uint8_t* down_proj = (uint8_t*)config.down_proj + offset * ggml_type_size((ggml_type)config.down_type) /
                                                          ggml_blck_size((ggml_type)config.down_type);

    for (int i = 0; i < config.expert_num; ++i) {
      memcpy(local_gate_proj, gate_proj,
             config.intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.gate_type) /
                 ggml_blck_size((ggml_type)config.gate_type));
      memcpy(local_up_proj, up_proj,
             config.intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.up_type) /
                 ggml_blck_size((ggml_type)config.up_type));
      for (int j = 0; j < config.hidden_size; ++j) {
        memcpy(local_down_proj, down_proj,
               config.intermediate_size * ggml_type_size((ggml_type)config.down_type) /
                   ggml_blck_size((ggml_type)config.down_type));
        local_down_proj += config.intermediate_size * ggml_type_size((ggml_type)config.down_type) /
                           ggml_blck_size((ggml_type)config.down_type);
        down_proj += complete_intermediate_size * ggml_type_size((ggml_type)config.down_type) /
                     ggml_blck_size((ggml_type)config.down_type);
      }
      local_gate_proj += config.intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.gate_type) /
                         ggml_blck_size((ggml_type)config.gate_type);
      local_up_proj += config.intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.up_type) /
                       ggml_blck_size((ggml_type)config.up_type);
      gate_proj += complete_intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.gate_type) /
                   ggml_blck_size((ggml_type)config.gate_type);
      up_proj += complete_intermediate_size * config.hidden_size * ggml_type_size((ggml_type)config.up_type) /
                 ggml_blck_size((ggml_type)config.up_type);
    }
  }

  void warm_up() {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                               ggml_blck_size((ggml_type)config_.hidden_type));
    std::vector<float> output(config_.hidden_size);
    for (int i = 0; i < config_.hidden_size; i++) {
      input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, (ggml_type)config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
      int64_t expert_ids = i;
      float weights = 0;
      forward_one(1, &expert_ids, &weights, input.data(), output.data());
    }
  }

  static float act_fn(float x) { return x / (1.0f + expf(-x)); }

  void forward_one(int k, const int64_t* expert_ids, const float* weights, const void* input, float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if ((ggml_type)config_.hidden_type == ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type &&
        (ggml_type)config_.hidden_type == ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
      gate_input_ptr = up_input_ptr = input;
    } else {
      to_float(input, s_input_fp32_, config_.hidden_size, (ggml_type)config_.hidden_type);
      if (ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type ==
          ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
        from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                   ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
        gate_input_ptr = up_input_ptr = s_gate_input_;
      } else {
        if ((ggml_type)config_.hidden_type !=
            ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) {
          from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                     ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
          gate_input_ptr = s_gate_input_;
        } else {
          gate_input_ptr = input;
        }
        if ((ggml_type)config_.hidden_type != ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
          from_float(s_input_fp32_, s_up_input_, config_.hidden_size,
                     ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
          up_input_ptr = s_up_input_;
        } else {
          up_input_ptr = input;
        }
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("gate_input: ");
    // debug_quant(const_cast<void *>(gate_input_ptr),
    // ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
    // printf("up_input: ");
    // debug_quant(const_cast<void *>(up_input_ptr),
    // ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, convert time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

#endif

    int nth = config_.intermediate_size / config_.m_block;

    pool->do_work_stealing_job(
        nth * k, nullptr,
        [&](int task_id) {
          int expert_idx = task_id / nth;
          int64_t expert_id = expert_ids[expert_idx];
          if (expert_id == -1) {
            return;
          }
          int ith = task_id % nth;

          void* gate_proj_ptr =
              (uint8_t*)m_local_gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.m_block) *
                                                 config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                                 ggml_blck_size((ggml_type)config_.gate_type);

          float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.m_block;
          auto ok = llamafile_sgemm(config_.m_block, 1,
                                    config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                                    config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                                    config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr,
                                    config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                                    ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type,
                                    GGML_TYPE_F32, GGML_PREC_DEFAULT);
          if (ok == false) [[unlikely]] {
            throw std::runtime_error("llamafile not supported");
          }
          // printf("gate output: ");
          // debug_f32(gate_output_ptr);

          void* up_proj_ptr =
              (uint8_t*)m_local_up_proj_ + (expert_id * config_.intermediate_size + ith * config_.m_block) *
                                               config_.hidden_size * ggml_type_size((ggml_type)config_.up_type) /
                                               ggml_blck_size((ggml_type)config_.up_type);

          float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.m_block;
          llamafile_sgemm(config_.m_block, 1, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
                          up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
                          config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
                          ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type, GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
          // printf("up output: ");
          // debug_f32(up_output_ptr);

          for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
          }
          if (config_.m_block %
                  ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) ==
              0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.m_block;
            void* down_input_ptr =
                s_down_input_[expert_idx] +
                ith * config_.m_block *
                    ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.m_block,
                       ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
          }
        },
        nullptr);

    if (config_.m_block % ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) !=
        0) {
      for (int i = 0; i < k; i++) {
        from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size,
                   ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("sinter:");
    // debug_f32(s_intermediate_fp32_[expert_ids[0]]);
    auto t2 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, gate/up time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
#endif

    nth = config_.hidden_size / config_.m_block;
    pool->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
            output[i] = 0;
          }
          for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            int64_t expert_id = expert_ids[expert_idx];
            if (expert_id == -1) {
              continue;
            }

            auto expert_offset = expert_id * config_.hidden_size * config_.intermediate_size;
            auto m_block_offset = ith * config_.m_block * config_.intermediate_size;
            void* down_proj_ptr = (uint8_t*)m_local_down_proj_ + (expert_offset + m_block_offset) *
                                                                     ggml_type_size((ggml_type)config_.down_type) /
                                                                     ggml_blck_size((ggml_type)config_.down_type);

            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.m_block;
            llamafile_sgemm(
                config_.m_block, 1, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                down_proj_ptr, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                down_output_ptr, config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type, GGML_TYPE_F32,
                GGML_PREC_DEFAULT);

            for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
              output[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    auto t3 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, down time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    fmt::print("numa_node: {}, total time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count());
#endif
  }

  void forward_many(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                    float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
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
        if (expert_ids[i * k + j] == -1) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_gate_input_ptr_[i] =
          m_local_gate_input_ +
          offset * config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
      m_local_up_input_ptr_[i] =
          m_local_up_input_ +
          offset * config_.hidden_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
      m_local_down_input_ptr_[i] =
          m_local_down_input_ +
          offset * config_.intermediate_size *
              ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
              ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          const void* gate_input_ptr;
          const void* up_input_ptr;
          if ((ggml_type)config_.hidden_type ==
                  ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type &&
              (ggml_type)config_.hidden_type ==
                  ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                                  ggml_type_size((ggml_type)config_.hidden_type) /
                                                                  ggml_blck_size((ggml_type)config_.hidden_type);
          } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                           ggml_blck_size((ggml_type)config_.hidden_type),
                     m_input_fp32_[i], config_.hidden_size, (ggml_type)config_.hidden_type);
            if (ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type ==
                ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
              from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                         ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
              gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
              if ((ggml_type)config_.hidden_type !=
                  ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                           ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type);
                gate_input_ptr = m_gate_input_[i];
              } else {
                gate_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                       ggml_type_size((ggml_type)config_.hidden_type) /
                                                       ggml_blck_size((ggml_type)config_.hidden_type);
              }
              if ((ggml_type)config_.hidden_type !=
                  ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size,
                           ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type);
                up_input_ptr = m_up_input_[i];
              } else {
                up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                     ggml_type_size((ggml_type)config_.hidden_type) /
                                                     ggml_blck_size((ggml_type)config_.hidden_type);
              }
            }
          }
          for (int j = 0; j < k; j++) {
            if (expert_ids[i * k + j] == -1) {
              continue;
            }
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                           ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type),
                   gate_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type) /
                       ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                           ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type),
                   up_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type) /
                       ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type));
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int m_block = QK_K;
    int nth = config_.intermediate_size / m_block;
    // printf("nth: %d, m_block: %d, activated_expert: %d\n", nth, m_block, activated_expert);
    // printf("config_.hidden_size: %d, config_.intermediate_size: %d\n", config_.hidden_size,
    // config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

          void* gate_proj_ptr =
              (uint8_t*)m_local_gate_proj_ + (expert_idx * config_.intermediate_size + ith * m_block) *
                                                 config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                                 ggml_blck_size((ggml_type)config_.gate_type);

          float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * m_block;

          // if (ith == 0) {
          //   printf("matrix size: m:%d, n:%d, k:%d\n", m_block, m_local_num_[expert_idx],
          //          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type));
          // }
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr,
                          config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                          ggml_internal_get_type_traits((ggml_type)config_.gate_type).vec_dot_type, GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
          void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

          void* up_proj_ptr = (uint8_t*)m_local_up_proj_ + (expert_idx * config_.intermediate_size + ith * m_block) *
                                                               config_.hidden_size *
                                                               ggml_type_size((ggml_type)config_.up_type) /
                                                               ggml_blck_size((ggml_type)config_.up_type);

          float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(
              m_block, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
              up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
              config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
              config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
              ggml_internal_get_type_traits((ggml_type)config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * m_block; j < (ith + 1) * m_block; j++) {
              m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] =
                  act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) *
                  m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr =
                m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * m_block;
            void* down_input_ptr =
                m_local_down_input_ptr_[expert_idx] +
                i * config_.intermediate_size *
                    ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) +
                ith * m_block *
                    ggml_type_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type) /
                    ggml_blck_size(ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, m_block,
                       ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type);
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

    m_block = QK_K;
    nth = config_.hidden_size / m_block;
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* down_input_ptr = m_local_down_input_ptr_[expert_idx];

          auto expert_offset = expert_idx * config_.hidden_size * config_.intermediate_size;
          auto m_block_offset = ith * m_block * config_.intermediate_size;

          void* down_proj_ptr = (uint8_t*)m_local_down_proj_ + (expert_offset + m_block_offset) *
                                                                   ggml_type_size((ggml_type)config_.down_type) /
                                                                   ggml_blck_size((ggml_type)config_.down_type);

          float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_proj_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_input_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_output_ptr,
                          config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                          ggml_internal_get_type_traits((ggml_type)config_.down_type).vec_dot_type, GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
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
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
          }
          for (int j = 0; j < k; j++) {
            if (expert_ids[i * k + j] == -1) {
              continue;
            }
            for (int e = 0; e < config_.hidden_size; e++) {
              m_output_fp32_[i][e] +=
                  m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] *
                  weights[i * k + j];
            }
          }
          for (int e = 0; e < config_.hidden_size; e++) {
            output[i * config_.hidden_size + e] = m_output_fp32_[i][e];
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
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output_in) {
    auto output = (float*)output_in;
    if (qlen < config_.group_min_len) {
      for (int i = 0; i < qlen; i++) {
        forward_one(k, expert_ids + i * k, weights + i * k,
                    (uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                          ggml_blck_size((ggml_type)config_.hidden_type),
                    output + i * config_.hidden_size);
      }
      return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k,
            (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                  ggml_blck_size((ggml_type)config_.hidden_type),
            output + forward_len * config_.hidden_size);
  }
};

template <>
class TP_MOE<LLAMA_MOE_TP> : public TP_MOE_Common<LLAMA_MOE_TP> {
 public:
  using TP_MOE_Common<LLAMA_MOE_TP>::TP_MOE_Common;

  void load_weights() {
    auto pool = this->config.pool;

    std::vector<int> tp_offsets(this->tp_count);
    int accumulated_offset = 0;
    for (int i = 0; i < this->tp_count; i++) {
      tp_offsets[i] = accumulated_offset;
      accumulated_offset += this->tp_configs[i].intermediate_size;
    }

    pool->dispense_backend()->do_numa_job([this, pool, tp_offsets](int tp_id) {
      this->tps[tp_id]->load_weights(this->config.intermediate_size, tp_offsets[tp_id]);
    });
    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }

  void merge_results(int qlen, void *output, bool incremental) {
    auto pool = this->config.pool;
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, incremental](int token_nth) {
          if (incremental) {
            to_float((uint8_t *)output + token_nth * config.hidden_size *
                                             ggml_type_size((ggml_type)config.hidden_type) /
                                             ggml_blck_size((ggml_type)config.hidden_type),
                     local_output + token_nth * config.hidden_size, config.hidden_size, (ggml_type)config.hidden_type);
            for (int e = 0; e < config.hidden_size; e++) {
              local_output_numa[0][token_nth * config.hidden_size + e] +=
                  local_output[token_nth * config.hidden_size + e];
            }
          }
          auto &tp_count = this->tp_count;
          for (int i = 1; i < tp_count; i++) {
            for (int e = 0; e < config.hidden_size; e++) {
              local_output_numa[0][token_nth * config.hidden_size + e] +=
                  local_output_numa[i][token_nth * config.hidden_size + e];
            }
          }
          from_float(local_output_numa[0] + token_nth * config.hidden_size,
                     (uint8_t *)output + token_nth * config.hidden_size *
                                             ggml_type_size((ggml_type)config.hidden_type) /
                                             ggml_blck_size((ggml_type)config.hidden_type),
                     config.hidden_size, (ggml_type)config.hidden_type);
        },
        nullptr);
  }
};
#endif