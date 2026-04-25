/**
 * @Description  : Mainly used for dev debug, with no numa version
 * @Author       : chenht2022
 * @Date         : 2025-04-25 18:28:12
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2025-04-25 18:28:12
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_SFT_AMX_MOE_H
#define CPUINFER_OPERATOR_SFT_AMX_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include <fstream>
#include <filesystem>

#include "debug_sft_moe.hpp"

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

#include "la/amx.hpp"

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
void *numa_alloc_aligned(size_t size, int node, size_t alignment) {
  void *ptr = numa_alloc_onnode(size, node);
  assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
  return ptr;
}
#endif

// static inline __m512 exp_avx512(__m512 x) {
//   const __m512 log2e = _mm512_set1_ps(1.44269504089f);
//   const __m512 c1 = _mm512_set1_ps(0.69314718056f);

//   __m512 y = _mm512_mul_ps(x, log2e);
//   __m512i int_part = _mm512_cvtps_epi32(y);
//   __m512 frac_part = _mm512_sub_ps(y, _mm512_cvtepi32_ps(int_part));

//   const __m512 poly_1 = _mm512_set1_ps(0.9999999995f);
//   const __m512 poly_2 = _mm512_set1_ps(0.6931471805f);
//   const __m512 poly_3 = _mm512_set1_ps(0.2402265069f);
//   const __m512 poly_4 = _mm512_set1_ps(0.0555041087f);
//   const __m512 poly_5 = _mm512_set1_ps(0.0096181291f);
//   const __m512 poly_6 = _mm512_set1_ps(0.0013333558f);

//   __m512 frac_exp = _mm512_fmadd_ps(
//       frac_part, poly_6,
//       _mm512_fmadd_ps(frac_part, poly_5,
//                       _mm512_fmadd_ps(frac_part, poly_4,
//                                       _mm512_fmadd_ps(frac_part, poly_3, _mm512_fmadd_ps(frac_part, poly_2, poly_1)))));

//   __m512 two_pow_i = _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(int_part));
//   return _mm512_mul_ps(two_pow_i, frac_exp);
// }

// static inline __m512 act_fn(__m512 gate_val, __m512 up_val) {
//   __m512 neg_gate_val = _mm512_sub_ps(_mm512_setzero_ps(), gate_val);
//   __m512 exp_neg_gate = exp_avx512(neg_gate_val);
//   __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_neg_gate);
//   __m512 act_val = _mm512_div_ps(gate_val, denom);

//   return _mm512_mul_ps(act_val, up_val);
// }

static inline __m512 sigmoid(__m512 x) {
  __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
  __m512 e = exp_avx512(neg);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), e);
  return _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
}

static inline __m512 act_fn_1(__m512 x) {
  __m512 sigmoid_val = sigmoid(x);
  return _mm512_mul_ps(sigmoid_val, x);
}

static inline __m512 act_fn_grad(__m512 x) {
  // sigmoid(x) * (1 + x * (1 - sigmoid(x)))
  __m512 sigmoid_val = sigmoid(x);
  __m512 one_minus_sigmoid = _mm512_sub_ps(_mm512_set1_ps(1.0f), sigmoid_val);
  __m512 x_term = _mm512_mul_ps(x, one_minus_sigmoid);
  __m512 one_plus_x_term = _mm512_add_ps(_mm512_set1_ps(1.0f), x_term);
  return _mm512_mul_ps(sigmoid_val, one_plus_x_term);
}

// static inline float bf16_to_fp32(ggml_bf16_t v) {
//     uint16_t lo16;
//     std::memcpy(&lo16, &v, sizeof(lo16));   // 取出 16 bit 数据
//     uint32_t tmp = uint32_t(lo16) << 16; // 放到高 16 位
//     float out;
//     std::memcpy(&out, &tmp, sizeof(float));
//     return out;
// }

// 把 ggml_bf16_t 数组转换成可读字符串（逗号分隔）
std::string int8_row_to_string(const int8_t* row, int len) {
    std::string s;
    for (int i = 0; i < len; ++i) {
        if (i) s += ", ";
        s += std::to_string(row[i]);
    }
    return s;
}

struct SFT_AMX_MOEConfig {
  int expert_num;
  int routed_expert_num;
  int hidden_size;
  int intermediate_size;
  int max_len;
  void *gate_proj;
  void *up_proj;
  void *down_proj;

  SFT_AMX_MOEConfig() {}

  SFT_AMX_MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int max_len,
                void *gate_proj, void *up_proj, void *down_proj)
      : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size),
        intermediate_size(intermediate_size), max_len(max_len), gate_proj(gate_proj), up_proj(up_proj),
        down_proj(down_proj) {}
};

template <class T> class SFT_AMX_MOE {
private:
  SFT_AMX_MOEConfig config_;
  void *gate_proj_; // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void *up_proj_;   // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  void *down_proj_; // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

  void *gate_proj_t_; // [expert_num * intermediate_size * hidden_size]
  void *up_proj_t_;   // [expert_num * intermediate_size * hidden_size]
  void *down_proj_t_; // [expert_num * hidden_size * intermediate_size]

  ggml_bf16_t *m_local_input_;       // [routed_expert_num * max_len * hidden_size]
  ggml_bf16_t *m_local_gate_output_; // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_up_output_;   // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_down_output_; // [routed_expert_num * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [max_len, routed_expert_num]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<ggml_bf16_t *> m_local_input_ptr_;       // [expert_num]
  std::vector<ggml_bf16_t *> m_local_gate_output_ptr_; // [expert_num]
  std::vector<ggml_bf16_t *> m_local_up_output_ptr_;   // [expert_num]
  std::vector<ggml_bf16_t *> m_local_down_output_ptr_; // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
#endif

  ggml_bf16_t *m_local_down_output_grad_;       // [routed_expert_num * max_len * hidden_size]
  ggml_bf16_t *m_local_down_input_grad_;        // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_gate_output_grad_;       // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_up_output_grad_;         // [routed_expert_num * max_len * intermediate_size]
  ggml_bf16_t *m_local_gate_input_grad_;        // [routed_expert_num * max_len * hidden_size]
  ggml_bf16_t *m_local_up_input_grad_;          // [routed_expert_num * max_len * hidden_size]

  std::vector<ggml_bf16_t *> m_local_down_output_grad_ptr_;       // [expert_num]
  std::vector<ggml_bf16_t *> m_local_down_input_grad_ptr_;        // [expert_num]
  std::vector<ggml_bf16_t *> m_local_gate_output_grad_ptr_;       // [expert_num]
  std::vector<ggml_bf16_t *> m_local_up_output_grad_ptr_;         // [expert_num]
  std::vector<ggml_bf16_t *> m_local_gate_input_grad_ptr_;        // [expert_num]
  std::vector<ggml_bf16_t *> m_local_up_input_grad_ptr_;          // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_t_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> up_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_t_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_t_bc_;

  // TODO: NUMA
  std::vector<std::shared_ptr<typename T::BufferB>> gate_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_t_bb_;

  int* m_local_token_indices_;                                   // [routed_expert_num * max_len]
  int* m_local_expert_positions_;                               // [routed_expert_num * max_len]
  std::vector<int *> m_local_token_indices_ptr_;                // [expert_num]
  std::vector<int *> m_local_expert_positions_ptr_;             // [expert_num]

public:
  SFT_AMX_MOE(SFT_AMX_MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    std::vector<std::pair<void **, uint64_t>> m_mem_requests;
    m_mem_requests.push_back({(void **)&m_local_input_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests.push_back({(void **)&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.routed_expert_num *
                                                                  config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_up_output_, sizeof(ggml_bf16_t) * config_.routed_expert_num *
                                                                config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_down_output_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    std::vector<void *> gate_up_ba_ptr(config_.expert_num);
    std::vector<void *> gate_bc_ptr(config_.expert_num);
    std::vector<void *> up_bc_ptr(config_.expert_num);
    std::vector<void *> down_ba_ptr(config_.expert_num);
    std::vector<void *> down_bc_ptr(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
      m_mem_requests.push_back(
          {(void **)&gate_up_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests.push_back(
          {(void **)&gate_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&up_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&down_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&down_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.hidden_size)});
    }

    m_mem_requests.push_back({(void **)&gate_proj_t_,
                              sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size});
    m_mem_requests.push_back({(void **)&up_proj_t_,
                              sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size});
    m_mem_requests.push_back({(void **)&down_proj_t_,
                              sizeof(ggml_bf16_t) * config_.expert_num * config_.hidden_size * config_.intermediate_size});
    
    m_mem_requests.push_back({(void **)&m_local_down_output_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests.push_back({(void **)&m_local_down_input_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_gate_output_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_up_output_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void **)&m_local_gate_input_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests.push_back({(void **)&m_local_up_input_grad_,
                              sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests.push_back({(void **)&m_local_token_indices_,
                              sizeof(int) * config_.routed_expert_num * config_.max_len});
    m_mem_requests.push_back({(void **)&m_local_expert_positions_,
                              sizeof(int) * config_.routed_expert_num * config_.max_len});
    std::vector<void *> gate_t_ba_ptr(config_.expert_num);
    std::vector<void *> gate_t_bc_ptr(config_.expert_num);
    std::vector<void *> up_t_ba_ptr(config_.expert_num);
    std::vector<void *> up_t_bc_ptr(config_.expert_num);
    std::vector<void *> down_t_ba_ptr(config_.expert_num);
    std::vector<void *> down_t_bc_ptr(config_.expert_num);
    for (int i = 0; i < config_.expert_num; i++) {
      m_mem_requests.push_back(
          {(void **)&gate_t_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&gate_t_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests.push_back(
          {(void **)&up_t_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests.push_back(
          {(void **)&up_t_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests.push_back(
          {(void **)&down_t_ba_ptr[i], T::BufferA::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests.push_back(
          {(void **)&down_t_bc_ptr[i], T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
    }

    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
    m_local_down_output_grad_ptr_.resize(config_.expert_num);
    m_local_down_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_grad_ptr_.resize(config_.expert_num);

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
      gate_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
      down_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, down_ba_ptr[i]));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, down_bc_ptr[i]));

#ifdef USE_NUMA
      int numa_nodes = numa_num_configured_nodes();
      gate_bb_numa_.resize(numa_nodes);
      up_bb_numa_.resize(numa_nodes);
      down_bb_numa_.resize(numa_nodes);
      for (int j = 0; j < numa_nodes; j++) {
        void *gate_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.intermediate_size, config_.hidden_size), j, 64);
        gate_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));
        void *up_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.intermediate_size, config_.hidden_size), j, 64);
        up_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr));
        void *down_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.hidden_size, config_.intermediate_size), j, 64);
        down_bb_numa_[j].push_back(  
            std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
      }
#else
      void *gate_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void *up_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void *down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
#endif
    }

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, gate_t_ba_ptr[i]));
      gate_t_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, gate_t_bc_ptr[i]));
      up_t_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, up_t_ba_ptr[i]));
      up_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, up_t_bc_ptr[i]));
      down_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, down_t_ba_ptr[i]));
      down_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, down_t_bc_ptr[i]));

      // TODO: NUMA
      void *gate_t_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      gate_t_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, gate_t_bb_ptr));

      void *up_t_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      up_t_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, up_t_bb_ptr));

      void *down_t_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      down_t_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, down_t_bb_ptr));
    }

    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);
  }

  ~SFT_AMX_MOE() { shared_mem_buffer.dealloc(this); }

  void transpose_expert(const void* src, void* dst, int R, int C, Backend* backend) {
    backend->do_work_stealing_job(
        config_.expert_num, nullptr,
        [&](uint64_t expert_idx) {
          for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                memcpy(
                    (uint8_t*)dst + (expert_idx * R * C + (c * R + r)) * sizeof(ggml_bf16_t),
                    (uint8_t*)src + (expert_idx * R * C + (r * C + c)) * sizeof(ggml_bf16_t),
                    sizeof(ggml_bf16_t));
            }
          }
        },
        nullptr);
  }
  
  void load_weights(Backend *backend) {
    transpose_expert(config_.gate_proj, gate_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(config_.up_proj, up_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(config_.down_proj, down_proj_t_, config_.hidden_size, config_.intermediate_size, backend);

    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            gate_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.gate_proj +
                                                       expert_idx * config_.intermediate_size * config_.hidden_size,
                                                   ith, nth);
            up_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.up_proj +
                                                     expert_idx * config_.intermediate_size * config_.hidden_size,
                                                 ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat((ggml_bf16_t *)config_.gate_proj +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)config_.up_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
#endif
		},
        nullptr);
    nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
          down_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)down_proj_t_ +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
        },
        nullptr);

	// if constexpr (std::is_same_v<typename T::dt, ggml_bf16_t>) {
	// 	// 确保 debug/ 目录存在
	// 	std::filesystem::create_directories("debug");

	// 	int tail_cols = 1024;
	// 	for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
	// 		auto buf = down_t_bb_[expert_idx].get();
	// 		std::cout << "k: " << buf->k << "; n: " << buf->n << std::endl;
	// 		// 打开对应 expert 的文件
	// 		std::string path = "debug/" + std::to_string(expert_idx) + "_down_bb_t_debug.txt";
	// 		std::ofstream ofs(path, std::ios::out);
	// 		if (!ofs) {
	// 			std::cerr << "Failed to open " << path << " for writing\n";
	// 			continue;
	// 		}

	// 		ofs << "==== Expert " << expert_idx << " ====\n";
	// 		for (int n_idx = 0; n_idx < buf->k; ++n_idx) {
	// 			// 明确当作 int8 读
	// 			const int8_t* row = reinterpret_cast<const int8_t*>(buf->b) + n_idx * buf->n;

	// 			// 写整行
	// 			ofs << "row[" << n_idx << "] = { "
	// 				<< int8_row_to_string(row, buf->n)
	// 				<< " }\n";
	// 		}

	// 		ofs.close();
	// 	}
	// }

	// if constexpr (std::is_same_v<typename T::dt, int8_t>) {
	// 	for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
	// 		auto buf = down_t_bb_[expert_idx].get();

	// 		// 打开对应 expert 的文件
	// 		std::string path = "debug/" + std::to_string(expert_idx) + "_down_bb_t_debug3.bin";
	// 		std::ofstream ofs(path, std::ios::binary);
	// 		for (int n_idx = 0; n_idx < buf->k; ++n_idx) {
	// 			const int8_t* row = reinterpret_cast<const int8_t*>(buf->b) + n_idx * buf->n;
	// 			for (int j = 0; j < buf->n; ++j) {
	// 				float v = row[j];
	// 				ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
	// 			}
	// 		}
	// 		ofs.close();
	// 	}
	// }

	
	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	// 打开对应 expert 的文件
	// 	std::string path = "debug/" + std::to_string(expert_idx) + "_up_proj_t.bin";
	// 	std::ofstream ofs(path, std::ios::binary);
	// 	std::cout << "config_.hidden_size: " << config_.hidden_size << std::endl;
	// 	std::cout << "config_.intermediate_size: " << config_.intermediate_size << std::endl;
	// 	for (int n_idx = 0; n_idx < config_.intermediate_size; ++n_idx) {
	// 		const int8_t* row = reinterpret_cast<const int8_t*>(config_.down_proj + expert_idx * n_idx * config_.hidden_size);
	// 		for (int j = 0; j < config_.hidden_size; ++j) {
	// 			float v = row[j];
	// 			ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
	// 		}
	// 	}
	// 	ofs.close();
	// }

    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)config_.down_proj +
                                                       expert_idx * config_.hidden_size * config_.intermediate_size,
                                                   ith, nth);
          }
#else
          down_bb_[expert_idx]->from_mat((ggml_bf16_t *)config_.down_proj +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
#endif
          up_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)up_proj_t_ +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
          gate_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)gate_proj_t_ +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
        },
        nullptr);
  }

  void warm_up(Backend *backend) {}

  void forward(int qlen, int k, const uint64_t *expert_ids, const float *weights, const void *input, void *output, Backend *backend) {
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
                   (ggml_bf16_t *)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
          }
        },
        nullptr);
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];

          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], gate_bb_numa_[Backend::numa_node][expert_idx], gate_bc_[expert_idx],
                       ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], up_bb_numa_[Backend::numa_node][expert_idx], up_bc_[expert_idx], ith,
                       nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                       gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx], ith, nth, use_amx);
#endif
          gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
            for (int j = n_start; j < n_end; j += 32) {
              __m512 gate_val0, gate_val1, up_val0, up_val1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
              __m512 result0 = act_fn(gate_val0, up_val0);
              __m512 result1 = act_fn(gate_val1, up_val1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
            }
          }
        },
        nullptr);
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);
	
	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	dump_grad_bin("cpp_layer0_E_End"+std::to_string(expert_idx)+"_down_ba_ori_", (ggml_bf16_t*)m_local_gate_output_ptr_[expert_idx], config_.intermediate_size * m_local_num_[expert_idx], GGML_TYPE_BF16);
	// }
	// if constexpr (std::is_same_v<typename T::dt, int8_t>) {
	// 	std::cout << "GO INTO forward output" << std::endl;
	// 	// 确保 debug/ 目录存在
	// 	std::filesystem::create_directories("debug");

	// 	for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
	// 		auto buf = down_ba_[expert_idx].get();
	// 		// std::cout << "k: " << buf->k << "; n: " << buf->n << std::endl;
	// 		// 打开对应 expert 的文件
	// 		std::string path = "debug/" + std::to_string(expert_idx) + "_down_ba_debug.txt";
	// 		std::ofstream ofs(path, std::ios::out);
	// 		if (!ofs) {
	// 			std::cerr << "Failed to open " << path << " for writing\n";
	// 			continue;
	// 		}

	// 		ofs << "==== Expert " << expert_idx << " ====\n";
	// 		ofs << "buf_k: " << buf->k << "\n";
	// 		for (int n_idx = 0; n_idx < m_local_num_[expert_idx]; ++n_idx) {
	// 			// 明确当作 bfloat16 读
	// 			const int8_t* row = reinterpret_cast<const int8_t*>(buf->a) + n_idx * buf->k;

	// 			// 写整行
	// 			ofs << "row[" << n_idx << "] = { "
	// 				<< int8_row_to_string(row, buf->k)
	// 				<< " }\n";
	// 		}

	// 		ofs.close();
	// 	}
	// 	std::cout << "OUT INTO forward output" << std::endl;
	// }
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_numa_[Backend::numa_node][expert_idx], down_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx],
                       down_bb_[expert_idx], down_bc_[expert_idx], ith, nth, use_amx);
#endif
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                    m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)((ggml_bf16_t *)output + i * config_.hidden_size + e));
          }
        },
        nullptr);
  }

  void backward(int qlen, int k, const uint64_t *expert_ids, const float *weights, const void* input, const void *output_grad, void *input_grad, Backend *backend) {
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;

      m_local_down_output_grad_ptr_[i] = m_local_down_output_grad_ + offset * config_.hidden_size;
      m_local_down_input_grad_ptr_[i] = m_local_down_input_grad_ + offset * config_.intermediate_size;
      m_local_gate_output_grad_ptr_[i] = m_local_gate_output_grad_ + offset * config_.intermediate_size;
      m_local_up_output_grad_ptr_[i] = m_local_up_output_grad_ + offset * config_.intermediate_size;
      m_local_gate_input_grad_ptr_[i] = m_local_gate_input_grad_ + offset * config_.hidden_size;
      m_local_up_input_grad_ptr_[i] = m_local_up_input_grad_ + offset * config_.hidden_size;
      m_local_token_indices_ptr_[i] = m_local_token_indices_ + offset;
      m_local_expert_positions_ptr_[i] = m_local_expert_positions_ + offset;
      offset += m_local_num_[i];
    }

    // TODO: cache
    backend->do_work_stealing_job(
        qlen, nullptr, 
        [&](int i) {
          for (int j = 0; j < k; j++) {
            uint64_t expert_id = expert_ids[i * k + j];
            int local_row = m_local_pos_[i][j];
            memcpy(m_local_input_ptr_[expert_id] + local_row * config_.hidden_size,
              (ggml_bf16_t *)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size); // TODO: cache
            memcpy(m_local_down_output_grad_ptr_[expert_id] + local_row * config_.hidden_size,
              (ggml_bf16_t *)output_grad + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
            m_local_token_indices_ptr_[expert_id][local_row] = i;
            m_local_expert_positions_ptr_[expert_id][local_row] = j;
          }
        }, 
        nullptr);

    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1); // TODO: cache
          down_t_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_down_output_grad_ptr_[expert_idx], 0, 1);
        },
        nullptr);
		
	// // for debug
	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	dump_grad_bin("cpp_layer0_E_End"+std::to_string(expert_idx)+"_down_output_grad_", (ggml_bf16_t*)m_local_down_output_grad_ptr_[expert_idx], config_.hidden_size * m_local_num_[expert_idx], GGML_TYPE_BF16);
	// }
	
	// if constexpr (std::is_same_v<typename T::dt, int8_t>) {
	// 	// 确保 debug/ 目录存在
	// 	std::filesystem::create_directories("debug");

	// 	for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
	// 		auto buf = down_t_ba_[expert_idx].get();
	// 		// std::cout << "k: " << buf->k << "; n: " << buf->n << std::endl;
	// 		// 打开对应 expert 的文件
	// 		std::string path = "debug/" + std::to_string(expert_idx) + "_down_ba_t_debug.txt";
	// 		std::ofstream ofs(path, std::ios::out);
	// 		if (!ofs) {
	// 			std::cerr << "Failed to open " << path << " for writing\n";
	// 			continue;
	// 		}

	// 		ofs << "==== Expert " << expert_idx << " ====\n";
	// 		for (int n_idx = 0; n_idx < m_local_num_[expert_idx]; ++n_idx) {
	// 			// 明确当作 bfloat16 读
	// 			const int8_t* row = reinterpret_cast<const int8_t*>(buf->a) + n_idx * buf->k;

	// 			// 写整行
	// 			ofs << "row[" << n_idx << "] = { "
	// 				<< int8_row_to_string(row, buf->k)
	// 				<< " }\n";
	// 		}

	// 		ofs.close();
	// 	}
	// }

    int nth = T::recommended_nth(config_.intermediate_size);  
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

        //   // TODO: cache
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      gate_up_ba_[expert_idx], gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx], ith, nth, use_amx);
          gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);

          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      down_t_ba_[expert_idx], down_t_bb_[expert_idx], down_t_bc_[expert_idx], ith, nth, use_amx);
          down_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_input_grad_ptr_[expert_idx], ith, nth);


          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *down_input_grad_ptr = &m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *gate_output_grad_ptr = &m_local_gate_output_grad_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_grad_ptr = &m_local_up_output_grad_ptr_[expert_idx][i * config_.intermediate_size];
            
            int token_idx = m_local_token_indices_ptr_[expert_idx][i];
            int expert_pos = m_local_expert_positions_ptr_[expert_idx][i];
            __m512 weight = _mm512_set1_ps(weights[token_idx * k + expert_pos]);
            
            for (int j = n_start; j < n_end; j += 32) {
              __m512 gate_val0, gate_val1, up_val0, up_val1, down_input_grad0, down_input_grad1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(down_input_grad_ptr + j), &down_input_grad0, &down_input_grad1);
              
              down_input_grad0 = _mm512_mul_ps(down_input_grad0, weight);
              down_input_grad1 = _mm512_mul_ps(down_input_grad1, weight);
              
              // gate_output_grad = δ_zji ⊙ v_ji ⊙ σ'(u_ji)
              __m512 gate_grad0 = _mm512_mul_ps(down_input_grad0, 
                                               _mm512_mul_ps(up_val0, act_fn_grad(gate_val0)));
              __m512 gate_grad1 = _mm512_mul_ps(down_input_grad1, 
                                               _mm512_mul_ps(up_val1, act_fn_grad(gate_val1)));
              
              // up_output_grad = δ_zji ⊙ σ(u_ji)
              __m512 up_grad0 = _mm512_mul_ps(down_input_grad0, act_fn_1(gate_val0));
              __m512 up_grad1 = _mm512_mul_ps(down_input_grad1, act_fn_1(gate_val1));
              
              avx512_32xfp32_to_32xbf16(&gate_grad0, &gate_grad1, (__m512i *)(gate_output_grad_ptr + j));
              avx512_32xfp32_to_32xbf16(&up_grad0, &up_grad1, (__m512i *)(up_output_grad_ptr + j));
            }
          }
        },
        nullptr);

	// for debug
	// if constexpr (std::is_same_v<typename T::dt, ggml_bf16_t>) {	
	// 	for (int expert_idx = 0; expert_idx < config_.expert_num; ++expert_idx) {
	// 		auto buf = down_t_ba_[expert_idx].get();

	// 		// 打开对应 expert 的文件
	// 		std::string path = "debug/" + std::to_string(expert_idx) + "_down_ba_t_debug3.bin";
	// 		std::ofstream ofs(path, std::ios::binary);
	// 		for (int n_idx = 0; n_idx < m_local_num_[expert_idx]; ++n_idx) {
	// 			const ggml_bf16_t* row = reinterpret_cast<const ggml_bf16_t*>(buf->a) + n_idx * buf->k;
	// 			for (int j = 0; j < buf->k; ++j) {
	// 				float v = row[j];
	// 				ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
	// 			}
	// 		}
	// 		ofs.close();
	// 	}
	// }
	
	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	dump_grad_bin("cpp_layer0_E_End"+std::to_string(expert_idx)+"_down_t_ba_", (ggml_bf16_t*)m_local_down_output_grad_ptr_[expert_idx], config_.hidden_size * m_local_num_[expert_idx], GGML_TYPE_BF16);
	// }
	
	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	dump_grad_bin("cpp_layer0_E_End"+std::to_string(expert_idx)+"_down_t_bb_", (ggml_bf16_t *)down_proj_t_ + expert_idx * config_.intermediate_size * config_.hidden_size, config_.hidden_size * config_.intermediate_size, GGML_TYPE_BF16);
	// }

	// for (uint64_t expert_idx = 0; expert_idx < (uint64_t)config_.expert_num; ++expert_idx) {
	// 	dump_grad_bin("cpp_layer0_E_End"+std::to_string(expert_idx)+"_down_t_bc_", (ggml_bf16_t*)m_local_down_input_grad_ptr_[expert_idx], config_.intermediate_size * m_local_num_[expert_idx], GGML_TYPE_BF16);
	// }

    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_t_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_grad_ptr_[expert_idx], 0, 1);
          up_t_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_up_output_grad_ptr_[expert_idx], 0, 1);
        },
        nullptr);
    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      gate_t_ba_[expert_idx], gate_t_bb_[expert_idx], gate_t_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      up_t_ba_[expert_idx], up_t_bb_[expert_idx], up_t_bc_[expert_idx], ith, nth, use_amx);
          gate_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_input_grad_ptr_[expert_idx], ith, nth);
          up_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_input_grad_ptr_[expert_idx], ith, nth);
        },
        nullptr);
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              __m512 gate_input_grad0, gate_input_grad1, up_input_grad0, up_input_grad1;
              avx512_32xbf16_to_32xfp32((__m512i *)(m_local_gate_input_grad_ptr_[expert_ids[i * k + j]] +
                                                    m_local_pos_[i][j] * config_.hidden_size + e),
                                        &gate_input_grad0, &gate_input_grad1);
              avx512_32xbf16_to_32xfp32((__m512i *)(m_local_up_input_grad_ptr_[expert_ids[i * k + j]] +
                                                    m_local_pos_[i][j] * config_.hidden_size + e),
                                        &up_input_grad0, &up_input_grad1);
              x0 = _mm512_add_ps(gate_input_grad0, x0);
              x1 = _mm512_add_ps(gate_input_grad1, x1);
              x0 = _mm512_add_ps(up_input_grad0, x0);
              x1 = _mm512_add_ps(up_input_grad1, x1);
            }
            avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)((ggml_bf16_t *)input_grad + i * config_.hidden_size + e));
          }
        },
        nullptr);
  }
};
#endif