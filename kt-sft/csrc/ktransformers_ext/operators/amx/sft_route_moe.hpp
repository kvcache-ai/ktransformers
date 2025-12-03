/**
 * @Description  : SFT Routed Experts MoE with LoRA Fine-tuning Support
 * @Author       : KT-SFT Team
 * @Date         : 2025-01-25
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_SFT_AMX_ROUTE_MOE_H
#define CPUINFER_OPERATOR_SFT_AMX_ROUTE_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>
#include <fstream>
#include <filesystem>
#include <omp.h>

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
#endif

// Reuse activation functions from sft_moe.hpp
static inline __m512 sigmoid_route(__m512 x) {
  __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), x);
  __m512 e = exp_avx512(neg);
  __m512 denom = _mm512_add_ps(_mm512_set1_ps(1.0f), e);
  return _mm512_div_ps(_mm512_set1_ps(1.0f), denom);
}

static inline __m512 act_fn_route(__m512 x) {
  __m512 sigmoid_val = sigmoid_route(x);
  return _mm512_mul_ps(sigmoid_val, x);
}

static inline __m512 act_fn_grad_route(__m512 x) {
  __m512 sigmoid_val = sigmoid_route(x);
  __m512 one_minus_sigmoid = _mm512_sub_ps(_mm512_set1_ps(1.0f), sigmoid_val);
  __m512 x_term = _mm512_mul_ps(x, one_minus_sigmoid);
  __m512 one_plus_x_term = _mm512_add_ps(_mm512_set1_ps(1.0f), x_term);
  return _mm512_mul_ps(sigmoid_val, one_plus_x_term);
}

/**
 * Configuration for SFT Routed MoE with LoRA
 * This differs from regular MoE by supporting LoRA adapters
 */
struct SFT_ROUTE_MOEConfig {
  int expert_num;          // Total number of routed experts
  int routed_expert_num;   // Number of experts routed per token
  int hidden_size;         // Model hidden dimension
  int intermediate_size;   // Expert intermediate dimension
  int max_len;             // Maximum sequence length

  // Base weights (frozen during LoRA training)
  void *gate_proj_base;
  void *up_proj_base;
  void *down_proj_base;

  // LoRA adapters (trainable)
  void *gate_lora_A;  // [expert_num, lora_rank, hidden_size]
  void *gate_lora_B;  // [expert_num, intermediate_size, lora_rank]
  void *up_lora_A;
  void *up_lora_B;
  void *down_lora_A;  // [expert_num, lora_rank, intermediate_size]
  void *down_lora_B;  // [expert_num, hidden_size, lora_rank]

  // LoRA gradients (output from backward pass)
  void *grad_gate_lora_A;  // [expert_num, lora_rank, hidden_size]
  void *grad_gate_lora_B;  // [expert_num, intermediate_size, lora_rank]
  void *grad_up_lora_A;    // [expert_num, lora_rank, hidden_size]
  void *grad_up_lora_B;    // [expert_num, intermediate_size, lora_rank]
  void *grad_down_lora_A;  // [expert_num, lora_rank, intermediate_size]
  void *grad_down_lora_B;  // [expert_num, hidden_size, lora_rank]

  int lora_rank;      // LoRA rank
  float lora_scaling; // LoRA scaling factor (alpha / rank)

  SFT_ROUTE_MOEConfig() {}

  SFT_ROUTE_MOEConfig(int expert_num, int routed_expert_num, int hidden_size,
                      int intermediate_size, int max_len,
                      void *gate_proj_base, void *up_proj_base, void *down_proj_base,
                      void *gate_lora_A, void *gate_lora_B,
                      void *up_lora_A, void *up_lora_B,
                      void *down_lora_A, void *down_lora_B,
                      int lora_rank, float lora_scaling,
                      void *grad_gate_lora_A, void *grad_gate_lora_B,
                      void *grad_up_lora_A, void *grad_up_lora_B,
                      void *grad_down_lora_A, void *grad_down_lora_B)
      : expert_num(expert_num), routed_expert_num(routed_expert_num),
        hidden_size(hidden_size), intermediate_size(intermediate_size), max_len(max_len),
        gate_proj_base(gate_proj_base), up_proj_base(up_proj_base), down_proj_base(down_proj_base),
        gate_lora_A(gate_lora_A), gate_lora_B(gate_lora_B),
        up_lora_A(up_lora_A), up_lora_B(up_lora_B),
        down_lora_A(down_lora_A), down_lora_B(down_lora_B),
        lora_rank(lora_rank), lora_scaling(lora_scaling),
        grad_gate_lora_A(grad_gate_lora_A), grad_gate_lora_B(grad_gate_lora_B),
        grad_up_lora_A(grad_up_lora_A), grad_up_lora_B(grad_up_lora_B),
        grad_down_lora_A(grad_down_lora_A), grad_down_lora_B(grad_down_lora_B) {}
};

/**
 * SFT Routed MoE with LoRA Support
 * Optimized for CPU inference with AMX acceleration
 */
template <class T> class SFT_ROUTE_MOE {
private:
  SFT_ROUTE_MOEConfig config_;

  // Base weights (frozen)
  void *gate_proj_base_;
  void *up_proj_base_;
  void *down_proj_base_;

  // LoRA adapters
  void *gate_lora_A_;
  void *gate_lora_B_;
  void *up_lora_A_;
  void *up_lora_B_;
  void *down_lora_A_;
  void *down_lora_B_;

  // Merged weights (base + LoRA) for forward pass
  void *gate_proj_merged_;
  void *up_proj_merged_;
  void *down_proj_merged_;

  // Transposed weights for backward pass
  void *gate_proj_t_;
  void *up_proj_t_;
  void *down_proj_t_;

  // Local buffers for token packing
  ggml_bf16_t *m_local_input_;
  ggml_bf16_t *m_local_gate_output_;
  ggml_bf16_t *m_local_up_output_;
  ggml_bf16_t *m_local_down_output_;

  // Gradient buffers for backward pass
  ggml_bf16_t *m_local_down_output_grad_;
  ggml_bf16_t *m_local_down_input_grad_;
  ggml_bf16_t *m_local_gate_output_grad_;
  ggml_bf16_t *m_local_up_output_grad_;
  ggml_bf16_t *m_local_gate_input_grad_;
  ggml_bf16_t *m_local_up_input_grad_;

  // Expert routing metadata
  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t *> m_local_input_ptr_;
  std::vector<ggml_bf16_t *> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t *> m_local_down_output_ptr_;

  std::vector<ggml_bf16_t *> m_local_down_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_down_input_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_gate_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_gate_input_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_input_grad_ptr_;

  // Token indices for backward pass
  int* m_local_token_indices_;
  int* m_local_expert_positions_;
  std::vector<int *> m_local_token_indices_ptr_;
  std::vector<int *> m_local_expert_positions_ptr_;

  // Track all allocated buffers for cleanup in destructor
  std::vector<void*> allocated_buffers_;

  // AMX buffers for matrix multiplication
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

  // Backward pass buffers
  std::vector<std::shared_ptr<typename T::BufferA>> gate_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_t_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> up_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_t_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_t_ba_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_t_bc_;

#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_t_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_t_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_t_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_t_bb_;
#endif

  // LoRA gradient computation buffers (per expert)
  // NOTE: lora_rank is padded to 32 for AMX alignment (actual rank may be smaller)
  int padded_lora_rank_;  // Padded lora_rank to meet AMX 32-alignment requirement

  // Gate projection LoRA buffers
  std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_input_ba_;       // [num_tokens, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_inter_bc_;       // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_grad_ba_;        // [num_tokens, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_temp_grad_bc_;   // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_A_bb_;           // [padded_rank, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_B_bb_;           // [intermediate_size, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_B_t_bb_;         // [padded_rank, intermediate_size] - transposed for grad_A computation
  std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_lora_A_bc_;      // [padded_rank, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_lora_B_bc_;      // [intermediate_size, padded_rank]

  // Up projection LoRA buffers
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_inter_bc_;         // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferA>> lora_up_grad_ba_;          // [num_tokens, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_temp_grad_bc_;     // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_A_bb_;             // [padded_rank, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_B_bb_;             // [intermediate_size, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_B_t_bb_;           // [padded_rank, intermediate_size] - transposed for grad_A computation
  std::vector<std::shared_ptr<typename T::BufferC>> grad_up_lora_A_bc_;        // [padded_rank, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_up_lora_B_bc_;        // [intermediate_size, padded_rank]

  // Down projection LoRA buffers
  std::vector<std::shared_ptr<typename T::BufferA>> lora_down_inter_ba_;       // [num_tokens, intermediate_size] (intermediate = silu(gate) * up)
  std::vector<std::shared_ptr<typename T::BufferC>> lora_down_lora_inter_bc_;  // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferA>> lora_down_grad_ba_;        // [num_tokens, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_down_temp_grad_bc_;   // [num_tokens, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_A_bb_;           // [padded_rank, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_B_bb_;           // [hidden_size, padded_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_B_t_bb_;         // [padded_rank, hidden_size] - transposed for grad_A computation
  std::vector<std::shared_ptr<typename T::BufferC>> grad_down_lora_A_bc_;      // [padded_rank, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_down_lora_B_bc_;      // [hidden_size, padded_rank]

public:
  SFT_ROUTE_MOE(SFT_ROUTE_MOEConfig config) {
    config_ = config;
    gate_proj_base_ = config_.gate_proj_base;
    up_proj_base_ = config_.up_proj_base;
    down_proj_base_ = config_.down_proj_base;

    gate_lora_A_ = config_.gate_lora_A;
    gate_lora_B_ = config_.gate_lora_B;
    up_lora_A_ = config_.up_lora_A;
    up_lora_B_ = config_.up_lora_B;
    down_lora_A_ = config_.down_lora_A;
    down_lora_B_ = config_.down_lora_B;

    // Calculate padded lora_rank for AMX alignment (must be multiple of 32)
    padded_lora_rank_ = (config_.lora_rank + 31) / 32 * 32;

    // Allocate memory for all buffers using aligned_alloc (64-byte alignment for AMX)
    // NOTE: We use independent allocation instead of shared_mem_buffer because
    // multiple SFT_ROUTE_MOE objects (different layers) exist simultaneously
    // and cannot share the same memory space.

    // Helper lambda to allocate and track buffers
    auto alloc_buffer = [this](size_t size, const char* name) -> void* {
      // Align size to 64 bytes
      size_t aligned_size = (size + 63) & ~63ULL;
      void* ptr = std::aligned_alloc(64, aligned_size);
      if (!ptr) {
        throw std::bad_alloc();
      }
      allocated_buffers_.push_back(ptr);
      return ptr;
    };

    // Local buffers
    m_local_input_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size,
        "m_local_input_");
    m_local_gate_output_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size,
        "m_local_gate_output_");
    m_local_up_output_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size,
        "m_local_up_output_");
    m_local_down_output_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size,
        "m_local_down_output_");

    // Gradient buffers
    m_local_down_output_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size,
        "m_local_down_output_grad_");
    m_local_down_input_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size,
        "m_local_down_input_grad_");
    m_local_gate_output_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size,
        "m_local_gate_output_grad_");
    m_local_up_output_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size,
        "m_local_up_output_grad_");
    m_local_gate_input_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size,
        "m_local_gate_input_grad_");
    m_local_up_input_grad_ = (ggml_bf16_t*)alloc_buffer(
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size,
        "m_local_up_input_grad_");

    // Token indices
    m_local_token_indices_ = (int*)alloc_buffer(
        sizeof(int) * config_.routed_expert_num * config_.max_len,
        "m_local_token_indices_");
    m_local_expert_positions_ = (int*)alloc_buffer(
        sizeof(int) * config_.routed_expert_num * config_.max_len,
        "m_local_expert_positions_");

    // Merged weights
    gate_proj_merged_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size,
        "gate_proj_merged_");
    up_proj_merged_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size,
        "up_proj_merged_");
    down_proj_merged_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.hidden_size * config_.intermediate_size,
        "down_proj_merged_");

    // Transposed weights
    gate_proj_t_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size,
        "gate_proj_t_");
    up_proj_t_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size,
        "up_proj_t_");
    down_proj_t_ = alloc_buffer(
        sizeof(ggml_bf16_t) * config_.expert_num * config_.hidden_size * config_.intermediate_size,
        "down_proj_t_");

    // AMX buffers - allocate independently for each expert
    std::vector<void *> gate_up_ba_ptr(config_.expert_num);
    std::vector<void *> gate_bc_ptr(config_.expert_num);
    std::vector<void *> up_bc_ptr(config_.expert_num);
    std::vector<void *> down_ba_ptr(config_.expert_num);
    std::vector<void *> down_bc_ptr(config_.expert_num);
    std::vector<void *> gate_t_ba_ptr(config_.expert_num);
    std::vector<void *> gate_t_bc_ptr(config_.expert_num);
    std::vector<void *> up_t_ba_ptr(config_.expert_num);
    std::vector<void *> up_t_bc_ptr(config_.expert_num);
    std::vector<void *> down_t_ba_ptr(config_.expert_num);
    std::vector<void *> down_t_bc_ptr(config_.expert_num);

    // LoRA gradient buffers - allocate only if lora_rank > 0
    std::vector<void *> lora_gate_input_ba_ptr(config_.expert_num);
    std::vector<void *> lora_gate_inter_bc_ptr(config_.expert_num);
    std::vector<void *> lora_gate_grad_ba_ptr(config_.expert_num);
    std::vector<void *> lora_gate_temp_grad_bc_ptr(config_.expert_num);
    std::vector<void *> lora_gate_A_bb_ptr(config_.expert_num);
    std::vector<void *> lora_gate_B_bb_ptr(config_.expert_num);
    std::vector<void *> lora_gate_B_t_bb_ptr(config_.expert_num);  // transposed for grad_A computation
    std::vector<void *> grad_gate_lora_A_bc_ptr(config_.expert_num);
    std::vector<void *> grad_gate_lora_B_bc_ptr(config_.expert_num);

    std::vector<void *> lora_up_inter_bc_ptr(config_.expert_num);
    std::vector<void *> lora_up_grad_ba_ptr(config_.expert_num);
    std::vector<void *> lora_up_temp_grad_bc_ptr(config_.expert_num);
    std::vector<void *> lora_up_A_bb_ptr(config_.expert_num);
    std::vector<void *> lora_up_B_bb_ptr(config_.expert_num);
    std::vector<void *> lora_up_B_t_bb_ptr(config_.expert_num);  // transposed for grad_A computation
    std::vector<void *> grad_up_lora_A_bc_ptr(config_.expert_num);
    std::vector<void *> grad_up_lora_B_bc_ptr(config_.expert_num);

    std::vector<void *> lora_down_inter_ba_ptr(config_.expert_num);
    std::vector<void *> lora_down_lora_inter_bc_ptr(config_.expert_num);
    std::vector<void *> lora_down_grad_ba_ptr(config_.expert_num);
    std::vector<void *> lora_down_temp_grad_bc_ptr(config_.expert_num);
    std::vector<void *> lora_down_A_bb_ptr(config_.expert_num);
    std::vector<void *> lora_down_B_bb_ptr(config_.expert_num);
    std::vector<void *> lora_down_B_t_bb_ptr(config_.expert_num);  // transposed for grad_A computation
    std::vector<void *> grad_down_lora_A_bc_ptr(config_.expert_num);
    std::vector<void *> grad_down_lora_B_bc_ptr(config_.expert_num);

    for (int i = 0; i < config_.expert_num; i++) {
      // Forward pass buffers
      gate_up_ba_ptr[i] = alloc_buffer(
          T::BufferA::required_size(config_.max_len, config_.hidden_size),
          "gate_up_ba");
      gate_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.intermediate_size),
          "gate_bc");
      up_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.intermediate_size),
          "up_bc");
      down_ba_ptr[i] = alloc_buffer(
          T::BufferA::required_size(config_.max_len, config_.intermediate_size),
          "down_ba");
      down_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.hidden_size),
          "down_bc");

      // Backward pass buffers
      gate_t_ba_ptr[i] = alloc_buffer(
          T::BufferA::required_size(config_.max_len, config_.intermediate_size),
          "gate_t_ba");
      gate_t_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.hidden_size),
          "gate_t_bc");
      up_t_ba_ptr[i] = alloc_buffer(
          T::BufferA::required_size(config_.max_len, config_.intermediate_size),
          "up_t_ba");
      up_t_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.hidden_size),
          "up_t_bc");
      down_t_ba_ptr[i] = alloc_buffer(
          T::BufferA::required_size(config_.max_len, config_.hidden_size),
          "down_t_ba");
      down_t_bc_ptr[i] = alloc_buffer(
          T::BufferC::required_size(config_.max_len, config_.intermediate_size),
          "down_t_bc");

      // LoRA gradient computation buffers (only if lora_rank > 0)
      if (config_.lora_rank > 0) {
        // Gate projection LoRA buffers
        lora_gate_input_ba_ptr[i] = alloc_buffer(
            T::BufferA::required_size(config_.max_len, config_.hidden_size),
            "lora_gate_input_ba");
        lora_gate_inter_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_gate_inter_bc");
        lora_gate_grad_ba_ptr[i] = alloc_buffer(
            T::BufferA::required_size(config_.max_len, config_.intermediate_size),
            "lora_gate_grad_ba");
        lora_gate_temp_grad_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_gate_temp_grad_bc");
        lora_gate_A_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size),
            "lora_gate_A_bb");
        lora_gate_B_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_),
            "lora_gate_B_bb");
        lora_gate_B_t_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size),
            "lora_gate_B_t_bb");  // transposed: [padded_rank, intermediate]
        grad_gate_lora_A_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(padded_lora_rank_, config_.hidden_size),
            "grad_gate_lora_A_bc");
        grad_gate_lora_B_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.intermediate_size, padded_lora_rank_),
            "grad_gate_lora_B_bc");

        // Up projection LoRA buffers
        lora_up_inter_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_up_inter_bc");
        lora_up_grad_ba_ptr[i] = alloc_buffer(
            T::BufferA::required_size(config_.max_len, config_.intermediate_size),
            "lora_up_grad_ba");
        lora_up_temp_grad_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_up_temp_grad_bc");
        lora_up_A_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size),
            "lora_up_A_bb");
        lora_up_B_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_),
            "lora_up_B_bb");
        lora_up_B_t_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size),
            "lora_up_B_t_bb");  // transposed: [padded_rank, intermediate]
        grad_up_lora_A_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(padded_lora_rank_, config_.hidden_size),
            "grad_up_lora_A_bc");
        grad_up_lora_B_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.intermediate_size, padded_lora_rank_),
            "grad_up_lora_B_bc");

        // Down projection LoRA buffers
        lora_down_inter_ba_ptr[i] = alloc_buffer(
            T::BufferA::required_size(config_.max_len, config_.intermediate_size),
            "lora_down_inter_ba");
        lora_down_lora_inter_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_down_lora_inter_bc");
        lora_down_grad_ba_ptr[i] = alloc_buffer(
            T::BufferA::required_size(config_.max_len, config_.hidden_size),
            "lora_down_grad_ba");
        lora_down_temp_grad_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.max_len, padded_lora_rank_),
            "lora_down_temp_grad_bc");
        lora_down_A_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size),
            "lora_down_A_bb");
        lora_down_B_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(config_.hidden_size, padded_lora_rank_),
            "lora_down_B_bb");
        lora_down_B_t_bb_ptr[i] = alloc_buffer(
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size),
            "lora_down_B_t_bb");  // transposed: [padded_rank, hidden]
        grad_down_lora_A_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(padded_lora_rank_, config_.intermediate_size),
            "grad_down_lora_A_bc");
        grad_down_lora_B_bc_ptr[i] = alloc_buffer(
            T::BufferC::required_size(config_.hidden_size, padded_lora_rank_),
            "grad_down_lora_B_bc");
      }
    }

    // Initialize metadata structures
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
    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);

    // Initialize AMX buffers
    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
      gate_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
      down_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, down_ba_ptr[i]));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, down_bc_ptr[i]));

      gate_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, gate_t_ba_ptr[i]));
      gate_t_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, gate_t_bc_ptr[i]));
      up_t_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, up_t_ba_ptr[i]));
      up_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, up_t_bc_ptr[i]));
      down_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, down_t_ba_ptr[i]));
      down_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, down_t_bc_ptr[i]));

#ifdef USE_NUMA
      int numa_nodes = numa_num_configured_nodes();
      gate_bb_numa_.resize(numa_nodes);
      up_bb_numa_.resize(numa_nodes);
      down_bb_numa_.resize(numa_nodes);
      gate_t_bb_numa_.resize(numa_nodes);
      up_t_bb_numa_.resize(numa_nodes);
      down_t_bb_numa_.resize(numa_nodes);

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

        void *gate_t_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.hidden_size, config_.intermediate_size), j, 64);
        gate_t_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, gate_t_bb_ptr));
        void *up_t_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.hidden_size, config_.intermediate_size), j, 64);
        up_t_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, up_t_bb_ptr));
        void *down_t_bb_ptr =
            numa_alloc_aligned(T::BufferB::required_size(config_.intermediate_size, config_.hidden_size), j, 64);
        down_t_bb_numa_[j].push_back(
            std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, down_t_bb_ptr));
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
#endif

      // Initialize LoRA gradient buffers (only if lora_rank > 0)
      if (config_.lora_rank > 0) {
        // Gate projection LoRA buffers
        lora_gate_input_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.hidden_size, lora_gate_input_ba_ptr[i]));
        lora_gate_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_gate_inter_bc_ptr[i]));
        lora_gate_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_gate_grad_ba_ptr[i]));
        lora_gate_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_gate_temp_grad_bc_ptr[i]));
        lora_gate_A_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.hidden_size, lora_gate_A_bb_ptr[i]));
        lora_gate_B_bb_.push_back(std::make_shared<typename T::BufferB>(
            config_.intermediate_size, padded_lora_rank_, lora_gate_B_bb_ptr[i]));
        lora_gate_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.intermediate_size, lora_gate_B_t_bb_ptr[i]));  // transposed
        grad_gate_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.hidden_size, grad_gate_lora_A_bc_ptr[i]));
        grad_gate_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.intermediate_size, padded_lora_rank_, grad_gate_lora_B_bc_ptr[i]));

        // Up projection LoRA buffers
        lora_up_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_up_inter_bc_ptr[i]));
        lora_up_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_up_grad_ba_ptr[i]));
        lora_up_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_up_temp_grad_bc_ptr[i]));
        lora_up_A_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.hidden_size, lora_up_A_bb_ptr[i]));
        lora_up_B_bb_.push_back(std::make_shared<typename T::BufferB>(
            config_.intermediate_size, padded_lora_rank_, lora_up_B_bb_ptr[i]));
        lora_up_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.intermediate_size, lora_up_B_t_bb_ptr[i]));  // transposed
        grad_up_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.hidden_size, grad_up_lora_A_bc_ptr[i]));
        grad_up_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.intermediate_size, padded_lora_rank_, grad_up_lora_B_bc_ptr[i]));

        // Down projection LoRA buffers
        lora_down_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_down_inter_ba_ptr[i]));
        lora_down_lora_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_lora_inter_bc_ptr[i]));
        lora_down_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.hidden_size, lora_down_grad_ba_ptr[i]));
        lora_down_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_temp_grad_bc_ptr[i]));
        lora_down_A_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.intermediate_size, lora_down_A_bb_ptr[i]));
        lora_down_B_bb_.push_back(std::make_shared<typename T::BufferB>(
            config_.hidden_size, padded_lora_rank_, lora_down_B_bb_ptr[i]));
        lora_down_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(
            padded_lora_rank_, config_.hidden_size, lora_down_B_t_bb_ptr[i]));  // transposed
        grad_down_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.intermediate_size, grad_down_lora_A_bc_ptr[i]));
        grad_down_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.hidden_size, padded_lora_rank_, grad_down_lora_B_bc_ptr[i]));
      }
    }
  }

  ~SFT_ROUTE_MOE() {
    for (void* ptr : allocated_buffers_) {
      if (ptr) {
        free(ptr);
      }
    }
  }

  /**
   * Transpose expert weights
   */
  void transpose_expert(const void* src, void* dst, int R, int C, Backend* backend) {
    backend->do_work_stealing_job(
        config_.expert_num, nullptr,
        [&](int expert_idx) {
          // NO cout inside lambda - it's not thread-safe!
          size_t expert_offset = expert_idx * R * C;
          for (int r = 0; r < R; ++r) {
            for (int c = 0; c < C; ++c) {
                size_t src_idx = expert_offset + r * C + c;
                size_t dst_idx = expert_offset + c * R + r;

                memcpy(
                    (uint8_t*)dst + dst_idx * sizeof(ggml_bf16_t),
                    (uint8_t*)src + src_idx * sizeof(ggml_bf16_t),
                    sizeof(ggml_bf16_t));
            }
          }
        },
        nullptr);
  }

  /**
   * Merge LoRA adapters with base weights: W = W_base + scaling * B @ A
   */
  void merge_lora_weights(Backend *backend) {
    backend->do_work_stealing_job(
        config_.expert_num, nullptr,
        [&](int expert_idx) {
          // NO cout inside lambda - it's not thread-safe!
          // Merge gate_proj
          ggml_bf16_t *gate_base = (ggml_bf16_t *)config_.gate_proj_base + expert_idx * config_.intermediate_size * config_.hidden_size;
          ggml_bf16_t *gate_merged = (ggml_bf16_t *)gate_proj_merged_ + expert_idx * config_.intermediate_size * config_.hidden_size;
          ggml_bf16_t *gate_A = (ggml_bf16_t *)config_.gate_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;
          ggml_bf16_t *gate_B = (ggml_bf16_t *)config_.gate_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

          // Copy base weight
          memcpy(gate_merged, gate_base, config_.intermediate_size * config_.hidden_size * sizeof(ggml_bf16_t));

          // Add LoRA: W += scaling * B @ A
          for (int i = 0; i < config_.intermediate_size; i++) {
            for (int j = 0; j < config_.hidden_size; j++) {
              float lora_delta = 0.0f;
              for (int r = 0; r < config_.lora_rank; r++) {
                float b_val = ggml_bf16_to_fp32(gate_B[i * config_.lora_rank + r]);
                float a_val = ggml_bf16_to_fp32(gate_A[r * config_.hidden_size + j]);
                lora_delta += b_val * a_val;
              }
              float base_val = ggml_bf16_to_fp32(gate_merged[i * config_.hidden_size + j]);
              gate_merged[i * config_.hidden_size + j] = GGML_FP32_TO_BF16(base_val + config_.lora_scaling * lora_delta);
            }
          }

          // Merge up_proj
          ggml_bf16_t *up_base = (ggml_bf16_t *)config_.up_proj_base + expert_idx * config_.intermediate_size * config_.hidden_size;
          ggml_bf16_t *up_merged = (ggml_bf16_t *)up_proj_merged_ + expert_idx * config_.intermediate_size * config_.hidden_size;
          ggml_bf16_t *up_A = (ggml_bf16_t *)config_.up_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;
          ggml_bf16_t *up_B = (ggml_bf16_t *)config_.up_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

          memcpy(up_merged, up_base, config_.intermediate_size * config_.hidden_size * sizeof(ggml_bf16_t));

          for (int i = 0; i < config_.intermediate_size; i++) {
            for (int j = 0; j < config_.hidden_size; j++) {
              float lora_delta = 0.0f;
              for (int r = 0; r < config_.lora_rank; r++) {
                float b_val = ggml_bf16_to_fp32(up_B[i * config_.lora_rank + r]);
                float a_val = ggml_bf16_to_fp32(up_A[r * config_.hidden_size + j]);
                lora_delta += b_val * a_val;
              }
              float base_val = ggml_bf16_to_fp32(up_merged[i * config_.hidden_size + j]);
              up_merged[i * config_.hidden_size + j] = GGML_FP32_TO_BF16(base_val + config_.lora_scaling * lora_delta);
            }
          }

          // Merge down_proj
          ggml_bf16_t *down_base = (ggml_bf16_t *)config_.down_proj_base + expert_idx * config_.hidden_size * config_.intermediate_size;
          ggml_bf16_t *down_merged = (ggml_bf16_t *)down_proj_merged_ + expert_idx * config_.hidden_size * config_.intermediate_size;
          ggml_bf16_t *down_A = (ggml_bf16_t *)config_.down_lora_A + expert_idx * config_.lora_rank * config_.intermediate_size;
          ggml_bf16_t *down_B = (ggml_bf16_t *)config_.down_lora_B + expert_idx * config_.hidden_size * config_.lora_rank;

          memcpy(down_merged, down_base, config_.hidden_size * config_.intermediate_size * sizeof(ggml_bf16_t));

          for (int i = 0; i < config_.hidden_size; i++) {
            for (int j = 0; j < config_.intermediate_size; j++) {
              float lora_delta = 0.0f;
              for (int r = 0; r < config_.lora_rank; r++) {
                float b_val = ggml_bf16_to_fp32(down_B[i * config_.lora_rank + r]);
                float a_val = ggml_bf16_to_fp32(down_A[r * config_.intermediate_size + j]);
                lora_delta += b_val * a_val;
              }
              float base_val = ggml_bf16_to_fp32(down_merged[i * config_.intermediate_size + j]);
              down_merged[i * config_.intermediate_size + j] = GGML_FP32_TO_BF16(base_val + config_.lora_scaling * lora_delta);
            }
          }
        },
        nullptr);
  }

  /**
   * Load and prepare weights for inference
   */
  void load_weights(Backend *backend) {
    // Merge LoRA with base weights
    merge_lora_weights(backend);

    // Transpose merged weights for backward pass
    transpose_expert(gate_proj_merged_, gate_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(up_proj_merged_, up_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(down_proj_merged_, down_proj_t_, config_.hidden_size, config_.intermediate_size, backend);

    // Load weights into AMX buffers
    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            gate_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)gate_proj_merged_ +
                                                       expert_idx * config_.intermediate_size * config_.hidden_size,
                                                   ith, nth);
            up_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)up_proj_merged_ +
                                                     expert_idx * config_.intermediate_size * config_.hidden_size,
                                                 ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat((ggml_bf16_t *)gate_proj_merged_ +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)up_proj_merged_ + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
#endif
        },
        nullptr);

    nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_t_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)down_proj_t_ +
                                                         expert_idx * config_.intermediate_size * config_.hidden_size,
                                                     ith, nth);
          }
#else
          down_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)down_proj_t_ +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
#endif
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    backend->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [&](int task_id) {
          uint64_t expert_idx = task_id / nth;
          int ith = task_id % nth;
#ifdef USE_NUMA
          int numa_nodes = numa_num_configured_nodes();
          for (int j = 0; j < numa_nodes; j++) {
            down_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)down_proj_merged_ +
                                                       expert_idx * config_.hidden_size * config_.intermediate_size,
                                                   ith, nth);
            gate_t_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)gate_proj_t_ +
                                                         expert_idx * config_.hidden_size * config_.intermediate_size,
                                                     ith, nth);
            up_t_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)up_proj_t_ +
                                                       expert_idx * config_.hidden_size * config_.intermediate_size,
                                                   ith, nth);
          }
#else
          down_bb_[expert_idx]->from_mat((ggml_bf16_t *)down_proj_merged_ +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
          gate_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)gate_proj_t_ +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
          up_t_bb_[expert_idx]->from_mat((ggml_bf16_t *)up_proj_t_ +
                                             expert_idx * config_.hidden_size * config_.intermediate_size,
                                         ith, nth);
#endif
        },
        nullptr);
  }

  void warm_up(Backend *backend) {}

  /**
   * Forward pass: compute MoE output with LoRA-adapted weights
   * Same interface as SFT_AMX_MOE for compatibility
   */
  void forward(int qlen, int k, const uint64_t *expert_ids, const float *weights,
               const void *input, void *output, Backend *backend) {
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;

    // Count tokens per expert
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

    // Setup local pointers
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

    // Pack tokens by expert
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
                   (ggml_bf16_t *)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
          }
        },
        nullptr);

    // Prepare input buffers
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Compute gate and up projections
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

          // Apply activation: gate * up
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
            for (int j = n_start; j < n_end; j += 32) {
              __m512 gate_val0, gate_val1, up_val0, up_val1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);
              __m512 result0 = _mm512_mul_ps(act_fn_route(gate_val0), up_val0);
              __m512 result1 = _mm512_mul_ps(act_fn_route(gate_val1), up_val1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
            }
          }
        },
        nullptr);

    // Prepare down projection input
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Compute down projection
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

    // Unpack and apply routing weights
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

  /**
   * Backward pass: compute gradients for LoRA fine-tuning
   * Same interface as SFT_AMX_MOE for compatibility
   */
  void backward(int qlen, int k, const uint64_t *expert_ids, const float *weights, const void* input,
                const void *output_grad, void *input_grad, Backend *backend) {
    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;

    // Count tokens per expert (same as forward)
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

    // Setup local pointers
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

    // Pack input and output gradients
    backend->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            uint64_t expert_id = expert_ids[i * k + j];
            int local_row = m_local_pos_[i][j];
            memcpy(m_local_input_ptr_[expert_id] + local_row * config_.hidden_size,
              (ggml_bf16_t *)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
            memcpy(m_local_down_output_grad_ptr_[expert_id] + local_row * config_.hidden_size,
              (ggml_bf16_t *)output_grad + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
            m_local_token_indices_ptr_[expert_id][local_row] = i;
            m_local_expert_positions_ptr_[expert_id][local_row] = j;
          }
        },
        nullptr);

    // Recompute forward pass (cache could be added for optimization)
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
          down_t_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_down_output_grad_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    int nth = T::recommended_nth(config_.intermediate_size);
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // Recompute forward
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

          // Compute down input gradient
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      down_t_ba_[expert_idx], down_t_bb_numa_[Backend::numa_node][expert_idx], down_t_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      down_t_ba_[expert_idx], down_t_bb_[expert_idx], down_t_bc_[expert_idx], ith, nth, use_amx);
#endif
          down_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_input_grad_ptr_[expert_idx], ith, nth);

          // Compute gate and up output gradients
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

              // gate_output_grad = δ * up * σ'(gate)
              __m512 gate_grad0 = _mm512_mul_ps(down_input_grad0,
                                               _mm512_mul_ps(up_val0, act_fn_grad_route(gate_val0)));
              __m512 gate_grad1 = _mm512_mul_ps(down_input_grad1,
                                               _mm512_mul_ps(up_val1, act_fn_grad_route(gate_val1)));

              // up_output_grad = δ * σ(gate)
              __m512 up_grad0 = _mm512_mul_ps(down_input_grad0, act_fn_route(gate_val0));
              __m512 up_grad1 = _mm512_mul_ps(down_input_grad1, act_fn_route(gate_val1));

              avx512_32xfp32_to_32xbf16(&gate_grad0, &gate_grad1, (__m512i *)(gate_output_grad_ptr + j));
              avx512_32xfp32_to_32xbf16(&up_grad0, &up_grad1, (__m512i *)(up_output_grad_ptr + j));
            }
          }
        },
        nullptr);

    // Compute input gradients
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
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      gate_t_ba_[expert_idx], gate_t_bb_numa_[Backend::numa_node][expert_idx], gate_t_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      up_t_ba_[expert_idx], up_t_bb_numa_[Backend::numa_node][expert_idx], up_t_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      gate_t_ba_[expert_idx], gate_t_bb_[expert_idx], gate_t_bc_[expert_idx], ith, nth, use_amx);
          amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                      up_t_ba_[expert_idx], up_t_bb_[expert_idx], up_t_bc_[expert_idx], ith, nth, use_amx);
#endif
          gate_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_input_grad_ptr_[expert_idx], ith, nth);
          up_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_input_grad_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // ==================== LoRA Gradient Computation (Optimized with AMX) ====================
    if (config_.lora_rank > 0) {
      // Compute LoRA gradients using optimized AMX matrix multiplications
      // This replaces the previous scalar loop implementation (Lines 1245-1374)

      // Step 1: Compute intermediate = silu(gate) * up for down_proj LoRA gradients
      // (Vectorized, keep as is)
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t *intermediate = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * config_.intermediate_size);

            for (int t = 0; t < num_tokens; t++) {
              ggml_bf16_t *gate_ptr = m_local_gate_output_ptr_[expert_idx] + t * config_.intermediate_size;
              ggml_bf16_t *up_ptr = m_local_up_output_ptr_[expert_idx] + t * config_.intermediate_size;
              ggml_bf16_t *inter_ptr = intermediate + t * config_.intermediate_size;

              for (int i = 0; i < config_.intermediate_size; i += 32) {
                __m512 gate0, gate1, up0, up1;
                avx512_32xbf16_to_32xfp32((__m512i *)(gate_ptr + i), &gate0, &gate1);
                avx512_32xbf16_to_32xfp32((__m512i *)(up_ptr + i), &up0, &up1);

                __m512 inter0 = _mm512_mul_ps(act_fn_route(gate0), up0);
                __m512 inter1 = _mm512_mul_ps(act_fn_route(gate1), up1);

                avx512_32xfp32_to_32xbf16(&inter0, &inter1, (__m512i *)(inter_ptr + i));
              }
            }

            // Store intermediate for later use
            m_local_down_output_ptr_[expert_idx] = intermediate;
          },
          nullptr);

      // Step 2: Load LoRA weights to BufferB (per expert, one-time setup)
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];

            // Gate LoRA weights: prepare padded versions with zeros
            ggml_bf16_t *gate_lora_A_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            ggml_bf16_t *gate_lora_B_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * config_.intermediate_size * padded_lora_rank_);

            memset(gate_lora_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            memset(gate_lora_B_padded, 0, sizeof(ggml_bf16_t) * config_.intermediate_size * padded_lora_rank_);

            ggml_bf16_t *gate_lora_A_src = (ggml_bf16_t *)config_.gate_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;
            ggml_bf16_t *gate_lora_B_src = (ggml_bf16_t *)config_.gate_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            // Copy actual rank data (8) into padded buffer (32), row by row
            for (int r = 0; r < config_.lora_rank; r++) {
              memcpy(gate_lora_A_padded + r * config_.hidden_size,
                     gate_lora_A_src + r * config_.hidden_size,
                     sizeof(ggml_bf16_t) * config_.hidden_size);
            }
            for (int i = 0; i < config_.intermediate_size; i++) {
              memcpy(gate_lora_B_padded + i * padded_lora_rank_,
                     gate_lora_B_src + i * config_.lora_rank,
                     sizeof(ggml_bf16_t) * config_.lora_rank);
            }

            // Load to BufferB (will be transposed internally)
            lora_gate_A_bb_[expert_idx]->from_mat(gate_lora_A_padded, 0, 1);
            lora_gate_B_bb_[expert_idx]->from_mat(gate_lora_B_padded, 0, 1);

            // Create and load transposed lora_B for grad_A computation
            // gate_lora_B is [intermediate_size, padded_rank], need [padded_rank, intermediate_size]
            ggml_bf16_t *gate_lora_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            memset(gate_lora_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            // Transpose: B[i, r] -> B_t[r, i]
            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                gate_lora_B_t_padded[r * config_.intermediate_size + i] = gate_lora_B_src[i * config_.lora_rank + r];
              }
            }
            lora_gate_B_t_bb_[expert_idx]->from_mat(gate_lora_B_t_padded, 0, 1);
            free(gate_lora_B_t_padded);

            free(gate_lora_A_padded);
            free(gate_lora_B_padded);

            // Up LoRA weights
            ggml_bf16_t *up_lora_A_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            ggml_bf16_t *up_lora_B_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * config_.intermediate_size * padded_lora_rank_);

            memset(up_lora_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            memset(up_lora_B_padded, 0, sizeof(ggml_bf16_t) * config_.intermediate_size * padded_lora_rank_);

            ggml_bf16_t *up_lora_A_src = (ggml_bf16_t *)config_.up_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;
            ggml_bf16_t *up_lora_B_src = (ggml_bf16_t *)config_.up_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            for (int r = 0; r < config_.lora_rank; r++) {
              memcpy(up_lora_A_padded + r * config_.hidden_size,
                     up_lora_A_src + r * config_.hidden_size,
                     sizeof(ggml_bf16_t) * config_.hidden_size);
            }
            for (int i = 0; i < config_.intermediate_size; i++) {
              memcpy(up_lora_B_padded + i * padded_lora_rank_,
                     up_lora_B_src + i * config_.lora_rank,
                     sizeof(ggml_bf16_t) * config_.lora_rank);
            }

            lora_up_A_bb_[expert_idx]->from_mat(up_lora_A_padded, 0, 1);
            lora_up_B_bb_[expert_idx]->from_mat(up_lora_B_padded, 0, 1);

            // Create and load transposed lora_B for grad_A computation
            // up_lora_B is [intermediate_size, padded_rank], need [padded_rank, intermediate_size]
            ggml_bf16_t *up_lora_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            memset(up_lora_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            // Transpose: B[i, r] -> B_t[r, i]
            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                up_lora_B_t_padded[r * config_.intermediate_size + i] = up_lora_B_src[i * config_.lora_rank + r];
              }
            }
            lora_up_B_t_bb_[expert_idx]->from_mat(up_lora_B_t_padded, 0, 1);
            free(up_lora_B_t_padded);

            free(up_lora_A_padded);
            free(up_lora_B_padded);

            // Down LoRA weights
            ggml_bf16_t *down_lora_A_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            ggml_bf16_t *down_lora_B_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * config_.hidden_size * padded_lora_rank_);

            memset(down_lora_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.intermediate_size);
            memset(down_lora_B_padded, 0, sizeof(ggml_bf16_t) * config_.hidden_size * padded_lora_rank_);

            ggml_bf16_t *down_lora_A_src = (ggml_bf16_t *)config_.down_lora_A + expert_idx * config_.lora_rank * config_.intermediate_size;
            ggml_bf16_t *down_lora_B_src = (ggml_bf16_t *)config_.down_lora_B + expert_idx * config_.hidden_size * config_.lora_rank;

            for (int r = 0; r < config_.lora_rank; r++) {
              memcpy(down_lora_A_padded + r * config_.intermediate_size,
                     down_lora_A_src + r * config_.intermediate_size,
                     sizeof(ggml_bf16_t) * config_.intermediate_size);
            }
            for (int h = 0; h < config_.hidden_size; h++) {
              memcpy(down_lora_B_padded + h * padded_lora_rank_,
                     down_lora_B_src + h * config_.lora_rank,
                     sizeof(ggml_bf16_t) * config_.lora_rank);
            }

            lora_down_A_bb_[expert_idx]->from_mat(down_lora_A_padded, 0, 1);
            lora_down_B_bb_[expert_idx]->from_mat(down_lora_B_padded, 0, 1);

            // Create and load transposed lora_B for grad_A computation
            // down_lora_B is [hidden_size, padded_rank], need [padded_rank, hidden_size]
            ggml_bf16_t *down_lora_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            memset(down_lora_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * config_.hidden_size);
            // Transpose: B[h, r] -> B_t[r, h]
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                down_lora_B_t_padded[r * config_.hidden_size + h] = down_lora_B_src[h * config_.lora_rank + r];
              }
            }
            lora_down_B_t_bb_[expert_idx]->from_mat(down_lora_B_t_padded, 0, 1);
            free(down_lora_B_t_padded);

            free(down_lora_A_padded);
            free(down_lora_B_padded);
          },
          nullptr);

      // Step 3: Load inputs and gradients to BufferA
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Load input, gate_grad, up_grad to BufferA
            lora_gate_input_ba_[expert_idx]->from_mat(num_tokens, m_local_input_ptr_[expert_idx], 0, 1);
            lora_gate_grad_ba_[expert_idx]->from_mat(num_tokens, m_local_gate_output_grad_ptr_[expert_idx], 0, 1);
            lora_up_grad_ba_[expert_idx]->from_mat(num_tokens, m_local_up_output_grad_ptr_[expert_idx], 0, 1);

            // For down_grad, need to apply routing weights first
            ggml_bf16_t *down_grad_weighted = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * config_.hidden_size);
            for (int t = 0; t < num_tokens; t++) {
              int token_idx = m_local_token_indices_ptr_[expert_idx][t];
              int expert_pos = m_local_expert_positions_ptr_[expert_idx][t];
              float token_weight = weights[token_idx * k + expert_pos];

              ggml_bf16_t *src = m_local_down_output_grad_ptr_[expert_idx] + t * config_.hidden_size;
              ggml_bf16_t *dst = down_grad_weighted + t * config_.hidden_size;

              for (int h = 0; h < config_.hidden_size; h += 32) {
                __m512 grad0, grad1;
                avx512_32xbf16_to_32xfp32((__m512i *)(src + h), &grad0, &grad1);
                grad0 = _mm512_mul_ps(grad0, _mm512_set1_ps(token_weight));
                grad1 = _mm512_mul_ps(grad1, _mm512_set1_ps(token_weight));
                avx512_32xfp32_to_32xbf16(&grad0, &grad1, (__m512i *)(dst + h));
              }
            }
            lora_down_grad_ba_[expert_idx]->from_mat(num_tokens, down_grad_weighted, 0, 1);
            free(down_grad_weighted);

            // Load intermediate (for down_proj)
            lora_down_inter_ba_[expert_idx]->from_mat(num_tokens, m_local_down_output_ptr_[expert_idx], 0, 1);
          },
          nullptr);

      // Step 4-6: Compute LoRA gradients using AMX matrix multiplication
      int nth = T::recommended_nth(padded_lora_rank_);

      // Gate projection gradients
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // lora_inter_gate = input @ gate_lora_A.T
            // [num_tokens, hidden_size] @ [hidden_size, padded_rank] -> [num_tokens, padded_rank]
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                        lora_gate_input_ba_[expert_idx], lora_gate_A_bb_[expert_idx],
                        lora_gate_inter_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      // Compute grad_gate_lora_B = gate_grad.T @ lora_inter_gate
      // This requires transposed gate_grad, but we can compute it as:
      // grad_B^T = lora_inter^T @ gate_grad, then transpose result
      // For simplicity, we'll use a manual outer product accumulation (still faster than scalar)
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Access intermediate result from BufferC
            float *lora_inter = lora_gate_inter_bc_[expert_idx]->c;
            ggml_bf16_t *gate_grad = m_local_gate_output_grad_ptr_[expert_idx];
            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_gate_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            float scaling = config_.lora_scaling;

            // grad_B[i, r] += sum_t(gate_grad[t,i] * lora_inter[t,r]) * scaling
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  float grad_val = ggml_bf16_to_fp32(gate_grad[t * config_.intermediate_size + i]);
                  float inter_val = lora_inter[t * padded_lora_rank_ + r];  // BufferC stores in row-major
                  sum += grad_val * inter_val;
                }
                float current = ggml_bf16_to_fp32(grad_B_dst[i * config_.lora_rank + r]);
                grad_B_dst[i * config_.lora_rank + r] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      // Compute temp_grad = gate_grad @ gate_lora_B for grad_gate_lora_A
      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // temp_grad = gate_grad @ gate_lora_B
            // [num_tokens, intermediate_size] @ [intermediate_size, padded_rank] -> [num_tokens, padded_rank]
            // Use transposed BufferB: mat_mul computes C = A @ B.T, so we need B_t where B_t = lora_B.T
            // This gives us: C = gate_grad @ B_t.T = gate_grad @ (lora_B.T).T = gate_grad @ lora_B
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.intermediate_size,
                        lora_gate_grad_ba_[expert_idx], lora_gate_B_t_bb_[expert_idx],
                        lora_gate_temp_grad_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      // Compute grad_gate_lora_A = temp_grad.T @ input
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            float *temp_grad = lora_gate_temp_grad_bc_[expert_idx]->c;
            ggml_bf16_t *input = m_local_input_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_gate_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;

            float scaling = config_.lora_scaling;

            // grad_A[r, h] += sum_t(temp_grad[t,r] * input[t,h]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  float temp_val = temp_grad[t * padded_lora_rank_ + r];
                  float input_val = ggml_bf16_to_fp32(input[t * config_.hidden_size + h]);
                  sum += temp_val * input_val;
                }
                float current = ggml_bf16_to_fp32(grad_A_dst[r * config_.hidden_size + h]);
                grad_A_dst[r * config_.hidden_size + h] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      // Up projection gradients (identical pattern to gate)
      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // lora_inter_up = input @ up_lora_A.T
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                        lora_gate_input_ba_[expert_idx], lora_up_A_bb_[expert_idx],
                        lora_up_inter_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            float *lora_inter = lora_up_inter_bc_[expert_idx]->c;
            ggml_bf16_t *up_grad = m_local_up_output_grad_ptr_[expert_idx];
            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_up_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            float scaling = config_.lora_scaling;

            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  sum += ggml_bf16_to_fp32(up_grad[t * config_.intermediate_size + i]) * lora_inter[t * padded_lora_rank_ + r];
                }
                float current = ggml_bf16_to_fp32(grad_B_dst[i * config_.lora_rank + r]);
                grad_B_dst[i * config_.lora_rank + r] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // temp_grad = up_grad @ up_lora_B
            // Use transposed BufferB for correct dimension matching
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.intermediate_size,
                        lora_up_grad_ba_[expert_idx], lora_up_B_t_bb_[expert_idx],
                        lora_up_temp_grad_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            float *temp_grad = lora_up_temp_grad_bc_[expert_idx]->c;
            ggml_bf16_t *input = m_local_input_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_up_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;

            float scaling = config_.lora_scaling;

            // grad_A[r, h] += sum_t(temp_grad[t,r] * input[t,h]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  sum += temp_grad[t * padded_lora_rank_ + r] * ggml_bf16_to_fp32(input[t * config_.hidden_size + h]);
                }
                float current = ggml_bf16_to_fp32(grad_A_dst[r * config_.hidden_size + h]);
                grad_A_dst[r * config_.hidden_size + h] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      // Down projection gradients (uses intermediate instead of input)
      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // lora_inter_down = intermediate @ down_lora_A.T
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.intermediate_size,
                        lora_down_inter_ba_[expert_idx], lora_down_A_bb_[expert_idx],
                        lora_down_lora_inter_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            float *lora_inter = lora_down_lora_inter_bc_[expert_idx]->c;
            ggml_bf16_t *down_grad_weighted = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * config_.hidden_size);

            // Reconstruct weighted down_grad (we loaded weighted version to BufferA)
            // But we need it again for grad_B computation
            for (int t = 0; t < num_tokens; t++) {
              int token_idx = m_local_token_indices_ptr_[expert_idx][t];
              int expert_pos = m_local_expert_positions_ptr_[expert_idx][t];
              float token_weight = weights[token_idx * k + expert_pos];

              ggml_bf16_t *src = m_local_down_output_grad_ptr_[expert_idx] + t * config_.hidden_size;
              ggml_bf16_t *dst = down_grad_weighted + t * config_.hidden_size;

              for (int h = 0; h < config_.hidden_size; h += 32) {
                __m512 grad0, grad1;
                avx512_32xbf16_to_32xfp32((__m512i *)(src + h), &grad0, &grad1);
                grad0 = _mm512_mul_ps(grad0, _mm512_set1_ps(token_weight));
                grad1 = _mm512_mul_ps(grad1, _mm512_set1_ps(token_weight));
                avx512_32xfp32_to_32xbf16(&grad0, &grad1, (__m512i *)(dst + h));
              }
            }

            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_down_lora_B + expert_idx * config_.hidden_size * config_.lora_rank;

            float scaling = config_.lora_scaling;
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  sum += ggml_bf16_to_fp32(down_grad_weighted[t * config_.hidden_size + h]) * lora_inter[t * padded_lora_rank_ + r];
                }
                float current = ggml_bf16_to_fp32(grad_B_dst[h * config_.lora_rank + r]);
                grad_B_dst[h * config_.lora_rank + r] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }

            free(down_grad_weighted);
          },
          nullptr);

      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // temp_grad = down_grad_weighted @ down_lora_B
            // Use transposed BufferB for correct dimension matching
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                        lora_down_grad_ba_[expert_idx], lora_down_B_t_bb_[expert_idx],
                        lora_down_temp_grad_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            float *temp_grad = lora_down_temp_grad_bc_[expert_idx]->c;
            ggml_bf16_t *intermediate = m_local_down_output_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_down_lora_A + expert_idx * config_.lora_rank * config_.intermediate_size;

            float scaling = config_.lora_scaling;

            // grad_A[r, i] += sum_t(temp_grad[t,r] * intermediate[t,i]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                #pragma omp simd reduction(+:sum)
                for (int t = 0; t < num_tokens; t++) {
                  sum += temp_grad[t * padded_lora_rank_ + r] * ggml_bf16_to_fp32(intermediate[t * config_.intermediate_size + i]);
                }
                float current = ggml_bf16_to_fp32(grad_A_dst[r * config_.intermediate_size + i]);
                grad_A_dst[r * config_.intermediate_size + i] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      // Cleanup intermediate buffers
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            if (m_local_down_output_ptr_[expert_idx]) {
              free(m_local_down_output_ptr_[expert_idx]);
              m_local_down_output_ptr_[expert_idx] = nullptr;
            }
          },
          nullptr);
    }
    // ==================== End LoRA Gradient Computation ====================

    // Unpack and accumulate gradients
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

#endif // CPUINFER_OPERATOR_SFT_AMX_ROUTE_MOE_H
