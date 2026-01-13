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
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <fstream>
#include <filesystem>
#include <type_traits>
#include <omp.h>

#include "debug_sft_moe.hpp"

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

#include "la/amx.hpp"

#include "moe.hpp"


#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

// Bit-level NaN/Inf detection (works with -ffast-math)
static inline bool is_nan_or_inf_bitwise(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(float));
  // NaN/Inf have all exponent bits set (0x7F800000)
  // NaN also has non-zero mantissa
  uint32_t exp_mask = 0x7F800000;
  return (bits & exp_mask) == exp_mask;
}

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

// ============================================================================
// Step-by-step dump utilities for debugging LoRA calculations
// Enable with environment variable: SFT_ROUTE_MOE_DUMP=1
// Dump directory: SFT_ROUTE_MOE_DUMP_DIR (default: ./cpp_dump)
// ============================================================================

static int g_sft_dump_enabled = -1;
static std::string g_sft_dump_dir = "";

static inline bool is_dump_enabled() {
  if (g_sft_dump_enabled < 0) {
    const char* env = std::getenv("SFT_ROUTE_MOE_DUMP");
    g_sft_dump_enabled = env ? std::atoi(env) : 0;
    if (g_sft_dump_enabled) {
      const char* dir_env = std::getenv("SFT_ROUTE_MOE_DUMP_DIR");
      g_sft_dump_dir = dir_env ? dir_env : "./cpp_dump";
      // Create directory if it doesn't exist
      std::filesystem::create_directories(g_sft_dump_dir);
      printf("[DUMP] Dumping enabled, output dir: %s\n", g_sft_dump_dir.c_str());
    }
  }
  return g_sft_dump_enabled > 0;
}

// Dump bf16 matrix to binary file
// Format: rows(int32), cols(int32), data(float32 converted from bf16)
static void dump_bf16_matrix(const char* name, int expert_id,
                              const ggml_bf16_t* data, int rows, int cols) {
  if (!is_dump_enabled()) return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s_e%d.bin",
           g_sft_dump_dir.c_str(), name, expert_id);

  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("[DUMP] Failed to open %s for writing\n", filename);
    return;
  }

  // Write header
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

  // Convert bf16 to f32 and check for NaN
  std::vector<float> f32_data(rows * cols);
  std::vector<std::pair<int, int>> nan_positions;
  for (int i = 0; i < rows * cols; i++) {
    f32_data[i] = GGML_BF16_TO_FP32(data[i]);
    if (is_nan_or_inf_bitwise(f32_data[i])) {
      int row = i / cols;
      int col = i % cols;
      nan_positions.push_back({row, col});
    }
  }
  file.write(reinterpret_cast<const char*>(f32_data.data()),
             sizeof(float) * rows * cols);

  printf("[DUMP] Wrote %s: [%d x %d]\n", filename, rows, cols);

  // Check for NaN/Inf and throw exception
  if (!nan_positions.empty()) {
    printf("[DUMP] ERROR: Found %zu NaN/Inf values in %s!\n", nan_positions.size(), filename);
    printf("[DUMP] First 10 NaN/Inf positions:\n");
    for (size_t i = 0; i < std::min(nan_positions.size(), (size_t)10); i++) {
      printf("  [%d, %d] = %f\n", nan_positions[i].first, nan_positions[i].second,
             f32_data[nan_positions[i].first * cols + nan_positions[i].second]);
    }
    // cpptrace::generate_trace().print();
    throw std::runtime_error(std::string("[DUMP] NaN/Inf detected in ") + filename);
  }
}

// Dump BufferA (AMX layout) to bf16 matrix
// Performs reverse conversion from AMX tiled layout to row-major bf16
template<typename T>
static void dump_buffer_a(const char* name, int expert_id,
                          std::shared_ptr<typename T::BufferA> buffer_a,
                          int m, int k) {
  if (!is_dump_enabled()) return;

  using K = T;
  constexpr int M_STEP = K::M_STEP;
  constexpr int K_STEP = K::K_STEP;
  constexpr int K_BLOCK = K::K_BLOCK;

  // Allocate temporary buffer for conversion
  ggml_bf16_t* tmp = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * m * k);
  memset(tmp, 0, sizeof(ggml_bf16_t) * m * k);

  // Reverse conversion from AMX layout to row-major
  int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
  for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
    for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512i *d = (__m512i *)(tmp + (m_begin + i) * k + k_block_begin + k_begin);
          __m512i *s = (__m512i *)(buffer_a->a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP);
          *d = *s;
        }
      }
    }
  }

  // Dump using existing function
  dump_bf16_matrix(name, expert_id, tmp, m, k);
  free(tmp);
}

// Dump BufferB (AMX layout) to bf16 matrix
// Performs reverse conversion from AMX tiled layout to row-major bf16
// Handles both BF16 and Int8 BufferB types correctly
template<typename T>
static void dump_buffer_b(const char* name, int expert_id,
                          std::shared_ptr<typename T::BufferB> buffer_b,
                          int n, int k) {
  if (!is_dump_enabled()) return;

  using K = T;
  constexpr int N_STEP = K::N_STEP;
  constexpr int K_STEP = K::K_STEP;
  constexpr int K_BLOCK = K::K_BLOCK;
  constexpr int N_BLOCK = K::N_BLOCK;
  constexpr int TILE_N = K::TILE_N;

  // Allocate output buffer
  ggml_bf16_t* tmp = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * n * k);
  memset(tmp, 0, sizeof(ggml_bf16_t) * n * k);

  // Allocate work buffer for one tile (N_STEP rows x K_STEP elements)
  // Size: N_STEP * K_STEP * sizeof(element) = 32 * K_STEP * element_size bytes
  void* work_buf = aligned_alloc(64, N_STEP * K_STEP * sizeof(typename T::dt));

  // Process by N_BLOCK to match from_mat layout
  for (int n_block_begin = 0; n_block_begin < n; n_block_begin += N_BLOCK) {
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);

    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);

        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          // Get pointer to this tile in BufferB
          typename T::dt* tile_ptr = buffer_b->b + n_block_begin * k +
                                     k_block_begin * n_block_size +
                                     n_begin * k_block_size +
                                     k_begin * N_STEP;

          // Copy tile to work buffer
          memcpy(work_buf, tile_ptr, N_STEP * K_STEP * sizeof(typename T::dt));

          // Reverse the two 16x16 32-bit transposes (transpose is self-inverse)
          amx::transpose_16x16_32bit((__m512i*)work_buf);
          amx::transpose_16x16_32bit((__m512i*)((char*)work_buf + TILE_N * K_STEP * sizeof(typename T::dt)));

          // Now extract rows from work buffer to output
          if constexpr (std::is_same_v<typename T::dt, ggml_bf16_t>) {
            // BF16 case: direct copy
            ggml_bf16_t* work_bf16 = (ggml_bf16_t*)work_buf;
            for (int i = 0; i < N_STEP; i++) {
              int row = n_block_begin + n_begin + i;
              if (row < n) {
                memcpy(tmp + row * k + k_block_begin + k_begin,
                       work_bf16 + i * K_STEP,
                       K_STEP * sizeof(ggml_bf16_t));
              }
            }
          } else {
            // Int8 case: dequantize q * d[row] -> bf16
            int8_t* work_int8 = (int8_t*)work_buf;
            for (int i = 0; i < N_STEP; i++) {
              int row = n_block_begin + n_begin + i;
              if (row < n) {
                float scale = buffer_b->d[row];
                for (int j = 0; j < K_STEP; j++) {
                  float val = (float)work_int8[i * K_STEP + j] * scale;
                  tmp[row * k + k_block_begin + k_begin + j] = GGML_FP32_TO_BF16(val);
                }
              }
            }
          }
        }
      }
    }
  }

  free(work_buf);

  // Dump using existing function
  dump_bf16_matrix(name, expert_id, tmp, n, k);
  free(tmp);
}

// Dump f32 matrix to binary file
// Format: rows(int32), cols(int32), data(float32)
static void dump_f32_matrix(const char* name, int expert_id,
                             const float* data, int rows, int cols) {
  if (!is_dump_enabled()) return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s_e%d.bin",
           g_sft_dump_dir.c_str(), name, expert_id);

  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("[DUMP] Failed to open %s for writing\n", filename);
    return;
  }

  // Write header
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));

  // Write data
  file.write(reinterpret_cast<const char*>(data), sizeof(float) * rows * cols);

  printf("[DUMP] Wrote %s: [%d x %d]\n", filename, rows, cols);
}

// Dump routing info
static void dump_routing_info(int qlen, int k, const uint64_t* expert_ids,
                               const float* weights, const std::vector<int>& m_local_num_) {
  if (!is_dump_enabled()) return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/routing_info.bin", g_sft_dump_dir.c_str());

  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("[DUMP] Failed to open %s for writing\n", filename);
    return;
  }

  // Write qlen, k
  file.write(reinterpret_cast<const char*>(&qlen), sizeof(int));
  file.write(reinterpret_cast<const char*>(&k), sizeof(int));

  // Write expert_ids as int64
  file.write(reinterpret_cast<const char*>(expert_ids), sizeof(uint64_t) * qlen * k);

  // Write weights
  file.write(reinterpret_cast<const char*>(weights), sizeof(float) * qlen * k);

  // Write m_local_num_ (tokens per expert)
  int num_experts = m_local_num_.size();
  file.write(reinterpret_cast<const char*>(&num_experts), sizeof(int));
  file.write(reinterpret_cast<const char*>(m_local_num_.data()), sizeof(int) * num_experts);

  printf("[DUMP] Wrote %s: qlen=%d, k=%d\n", filename, qlen, k);
}

// Dump final output
static void dump_final_output(const void* output, int qlen, int hidden_size) {
  if (!is_dump_enabled()) return;

  char filename[256];
  snprintf(filename, sizeof(filename), "%s/final_output.bin", g_sft_dump_dir.c_str());

  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("[DUMP] Failed to open %s for writing\n", filename);
    return;
  }

  // Write header
  file.write(reinterpret_cast<const char*>(&qlen), sizeof(int));
  file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));

  // Convert bf16 to f32 and write
  const ggml_bf16_t* bf16_data = static_cast<const ggml_bf16_t*>(output);
  std::vector<float> f32_data(qlen * hidden_size);
  for (int i = 0; i < qlen * hidden_size; i++) {
    f32_data[i] = GGML_BF16_TO_FP32(bf16_data[i]);
  }
  file.write(reinterpret_cast<const char*>(f32_data.data()),
             sizeof(float) * qlen * hidden_size);

  printf("[DUMP] Wrote %s: [%d x %d]\n", filename, qlen, hidden_size);
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

  // Local buffers for token packing
  ggml_bf16_t *m_local_input_;
  ggml_bf16_t *m_local_gate_output_;
  ggml_bf16_t *m_local_up_output_;
  ggml_bf16_t *m_local_down_output_;
  ggml_bf16_t *m_local_gate_output_lora_;
  ggml_bf16_t *m_local_up_output_lora_;
  ggml_bf16_t *m_local_down_output_lora_;
  
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

  // IN
  std::vector<ggml_bf16_t *> m_local_input_ptr_;

  // GATE UP DOWN OUT
  std::vector<ggml_bf16_t *> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t *> m_local_down_output_ptr_;

  // GATE UP OUTPUT LORA
  std::vector<ggml_bf16_t *> m_local_gate_output_lora_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_output_lora_ptr_;
  std::vector<ggml_bf16_t *> m_local_down_output_lora_ptr_;


  std::vector<ggml_bf16_t *> m_local_down_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_down_input_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_gate_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_output_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_gate_input_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_input_grad_ptr_;

  // LoRA input gradients (for separate base + lora computation)
  ggml_bf16_t *m_local_gate_input_lora_grad_;
  ggml_bf16_t *m_local_up_input_lora_grad_;
  std::vector<ggml_bf16_t *> m_local_gate_input_lora_grad_ptr_;
  std::vector<ggml_bf16_t *> m_local_up_input_lora_grad_ptr_;

  // Cached weighted down gradient (avoid redundant computation)
  std::vector<ggml_bf16_t *> m_local_down_grad_weighted_ptr_;

  // Token indices for backward pass
  int* m_local_token_indices_;
  int* m_local_expert_positions_;
  std::vector<int *> m_local_token_indices_ptr_;
  std::vector<int *> m_local_expert_positions_ptr_;

  // Track all allocated buffers for cleanup in destructor
  std::vector<void*> allocated_buffers_;

    // forward buffer requests
  std::vector<std::pair<void**, uint64_t>> m_mem_requests_fwd;
  std::vector<std::pair<void**, uint64_t>> m_mem_requests_bak;

  // AMX buffers for matrix multiplication

#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;

  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_lora_A_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_lora_B_;

  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_lora_A_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_lora_B_;

  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_lora_A_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_lora_B_;
#endif

  // Backward pass buffers
#ifdef USE_NUMA
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> gate_t_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> up_t_bb_numa_;
  std::vector<std::vector<std::shared_ptr<typename T::BufferB>>> down_t_bb_numa_;
#else
  std::vector<std::shared_ptr<typename T::BufferB>> gate_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_t_bb_;
#endif

  // Backward LoRA BufferB vectors (used in backward pass)
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_A_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_A_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_B_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_gate_B_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_A_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_A_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_B_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_up_B_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_A_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_B_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_A_t_bb_;
  std::vector<std::shared_ptr<typename T::BufferB>> lora_down_B_t_bb_;

  // LoRA gradient computation buffers (per expert)
  // NOTE: lora_rank is padded to 32 for AMX alignment (actual rank may be smaller)
  int padded_lora_rank_;  // Padded lora_rank to meet AMX 32-alignment requirement

  std::vector<void *> gate_up_ba_ptr;
  std::vector<void *> gate_bc_ptr;
  std::vector<void *> up_bc_ptr;
  std::vector<void *> down_ba_ptr;
  std::vector<void *> down_bc_ptr;
  std::vector<void *> gate_t_ba_ptr;
  std::vector<void *> gate_t_bc_ptr;
  std::vector<void *> up_t_ba_ptr;
  std::vector<void *> up_t_bc_ptr;
  std::vector<void *> down_t_ba_ptr;
  std::vector<void *> down_t_bc_ptr;

  // LoRA buffer pointers (allocated via shared_mem_buffer like gate_up_ba_ptr etc.)
  std::vector<void *> lora_gate_input_ba_ptr;
  std::vector<void *> lora_gate_inter_bc_ptr;
  std::vector<void *> lora_gate_grad_ba_ptr;
  std::vector<void *> lora_gate_temp_grad_bc_ptr;
  std::vector<void *> lora_gate_A_bb_ptr;
  std::vector<void *> lora_gate_A_t_bb_ptr;  // Transposed A for input grad LoRA
  std::vector<void *> lora_gate_B_bb_ptr;
  std::vector<void *> lora_gate_B_t_bb_ptr;
  std::vector<void *> grad_gate_lora_A_bc_ptr;
  std::vector<void *> grad_gate_lora_B_bc_ptr;

  std::vector<void *> lora_up_inter_bc_ptr;
  std::vector<void *> lora_up_grad_ba_ptr;
  std::vector<void *> lora_up_temp_grad_bc_ptr;
  std::vector<void *> lora_up_A_bb_ptr;
  std::vector<void *> lora_up_A_t_bb_ptr;  // Transposed A for input grad LoRA
  std::vector<void *> lora_up_B_bb_ptr;
  std::vector<void *> lora_up_B_t_bb_ptr;
  std::vector<void *> grad_up_lora_A_bc_ptr;
  std::vector<void *> grad_up_lora_B_bc_ptr;

  std::vector<void *> lora_down_inter_ba_ptr;
  std::vector<void *> lora_down_lora_inter_bc_ptr;
  std::vector<void *> lora_down_grad_ba_ptr;
  std::vector<void *> lora_down_temp_grad_bc_ptr;
  std::vector<void *> lora_down_temp_grad_inter_bc_ptr;  // [num_tokens, intermediate_size] for mat_mul output
  std::vector<void *> lora_down_A_bb_ptr;
  std::vector<void *> lora_down_B_bb_ptr;
  std::vector<void *> lora_down_A_t_bb_ptr;
  std::vector<void *> lora_down_B_t_bb_ptr;
  std::vector<void *> grad_down_lora_A_bc_ptr;
  std::vector<void *> grad_down_lora_B_bc_ptr;

  // Forward LoRA buffer pointers
  std::vector<void *> lora_gate_inter_bc_fwd_ptr;
  std::vector<void *> lora_up_inter_bc_fwd_ptr;
  std::vector<void *> lora_gate_inter_ba_fwd_ptr;
  std::vector<void *> lora_up_inter_ba_fwd_ptr;
  std::vector<void *> lora_gate_output_bc_fwd_ptr;
  std::vector<void *> lora_up_output_bc_fwd_ptr;
  std::vector<void *> lora_down_inter_bc_fwd_ptr;
  std::vector<void *> lora_down_inter_ba_fwd_ptr;
  std::vector<void *> lora_down_output_bc_fwd_ptr;


  std::vector<void *> lora_gate_inter_bc_bak_fwd_ptr;
  std::vector<void *> lora_up_inter_bc_bak_fwd_ptr;
  std::vector<void *> lora_gate_inter_ba_bak_fwd_ptr;
  std::vector<void *> lora_up_inter_ba_bak_fwd_ptr;
  std::vector<void *> lora_gate_output_bc_bak_fwd_ptr;
  std::vector<void *> lora_up_output_bc_bak_fwd_ptr;
  std::vector<void *> lora_down_inter_bc_bak_fwd_ptr;
  std::vector<void *> lora_down_inter_ba_bak_fwd_ptr;
  std::vector<void *> lora_down_inter_ba_inter_ptr;  // [num_tokens, intermediate_size] for mat_mul with k=intermediate_size
  std::vector<void *> lora_down_output_bc_bak_fwd_ptr;

  // Backward temporary buffer pointers
  // Forward recomputation buffers
  std::vector<void *> bak_gate_inter_bf16_ptr;      // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_up_inter_bf16_ptr;        // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_gate_lora_bf16_ptr;       // [max_len, intermediate_size] per expert
  std::vector<void *> bak_up_lora_bf16_ptr;         // [max_len, intermediate_size] per expert
  std::vector<void *> bak_down_inter_bf16_ptr;      // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_down_lora_grad_bf16_ptr;  // [max_len, intermediate_size] per expert
  std::vector<void *> bak_temp_gate_ptr;            // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_temp_up_ptr;              // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_intermediate_ptr;         // [max_len, intermediate_size] per expert
  std::vector<void *> bak_down_grad_weighted_ptr;   // [max_len, hidden_size] per expert

  // Gate LoRA gradient computation buffers
  std::vector<void *> bak_gate_lora_inter_bf16_ptr;  // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_gate_lora_inter_f32_ptr;   // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_gate_grad_f32_ptr;         // [max_len, intermediate_size] per expert
  std::vector<void *> bak_gate_temp_grad_bf16_ptr;   // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_gate_temp_grad_f32_ptr;    // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_gate_input_f32_ptr;        // [max_len, hidden_size] per expert

  // Up LoRA gradient computation buffers
  std::vector<void *> bak_up_lora_inter_bf16_ptr;    // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_up_lora_inter_f32_ptr;     // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_up_grad_f32_ptr;           // [max_len, intermediate_size] per expert
  std::vector<void *> bak_up_temp_grad_bf16_ptr;     // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_up_temp_grad_f32_ptr;      // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_up_input_f32_ptr;          // [max_len, hidden_size] per expert

  // Down LoRA gradient computation buffers
  std::vector<void *> bak_down_lora_inter_bf16_ptr;     // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_down_lora_inter_f32_ptr;      // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_down_grad_weighted_f32_ptr;   // [max_len, hidden_size] per expert
  std::vector<void *> bak_down_temp_grad_bf16_ptr;      // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_down_temp_grad_f32_ptr;       // [max_len, padded_lora_rank] per expert
  std::vector<void *> bak_down_intermediate_f32_ptr;    // [max_len, intermediate_size] per expert

public:
  SFT_ROUTE_MOE(SFT_ROUTE_MOEConfig config)
  : gate_up_ba_ptr(config.expert_num, 0), gate_bc_ptr(config.expert_num, 0),
    up_bc_ptr(config.expert_num, 0), down_ba_ptr(config.expert_num, 0),
    down_bc_ptr(config.expert_num, 0), gate_t_ba_ptr(config.expert_num, 0), gate_t_bc_ptr(config.expert_num, 0),
    up_t_ba_ptr(config.expert_num, 0), up_t_bc_ptr(config.expert_num, 0),
    down_t_ba_ptr(config.expert_num, 0), down_t_bc_ptr(config.expert_num, 0),
    // LoRA buffer pointers initialization
    lora_gate_input_ba_ptr(config.expert_num, 0), lora_gate_inter_bc_ptr(config.expert_num, 0),
    lora_gate_grad_ba_ptr(config.expert_num, 0), lora_gate_temp_grad_bc_ptr(config.expert_num, 0),
    lora_gate_A_bb_ptr(config.expert_num, 0), lora_gate_A_t_bb_ptr(config.expert_num, 0),
    lora_gate_B_bb_ptr(config.expert_num, 0), lora_gate_B_t_bb_ptr(config.expert_num, 0),
    grad_gate_lora_A_bc_ptr(config.expert_num, 0),
    grad_gate_lora_B_bc_ptr(config.expert_num, 0),
    lora_up_inter_bc_ptr(config.expert_num, 0), lora_up_grad_ba_ptr(config.expert_num, 0),
    lora_up_temp_grad_bc_ptr(config.expert_num, 0), lora_up_A_bb_ptr(config.expert_num, 0),
    lora_up_A_t_bb_ptr(config.expert_num, 0), lora_up_B_bb_ptr(config.expert_num, 0),
    lora_up_B_t_bb_ptr(config.expert_num, 0),
    grad_up_lora_A_bc_ptr(config.expert_num, 0), grad_up_lora_B_bc_ptr(config.expert_num, 0),
    lora_down_inter_ba_ptr(config.expert_num, 0), lora_down_lora_inter_bc_ptr(config.expert_num, 0),
    lora_down_grad_ba_ptr(config.expert_num, 0), lora_down_temp_grad_bc_ptr(config.expert_num, 0),
    lora_down_temp_grad_inter_bc_ptr(config.expert_num, 0),
    lora_down_A_bb_ptr(config.expert_num, 0), lora_down_B_bb_ptr(config.expert_num, 0),
    lora_down_A_t_bb_ptr(config.expert_num, 0),
    lora_down_B_t_bb_ptr(config.expert_num, 0), grad_down_lora_A_bc_ptr(config.expert_num, 0),
    grad_down_lora_B_bc_ptr(config.expert_num, 0),
    // Forward LoRA buffer pointers initialization
    lora_gate_inter_bc_fwd_ptr(config.expert_num, 0), lora_up_inter_bc_fwd_ptr(config.expert_num, 0),
    lora_gate_inter_ba_fwd_ptr(config.expert_num, 0), lora_up_inter_ba_fwd_ptr(config.expert_num, 0),
    lora_gate_output_bc_fwd_ptr(config.expert_num, 0), lora_up_output_bc_fwd_ptr(config.expert_num, 0),
    lora_down_inter_bc_fwd_ptr(config.expert_num, 0), lora_down_inter_ba_fwd_ptr(config.expert_num, 0),
    lora_down_output_bc_fwd_ptr(config.expert_num, 0),
    lora_gate_inter_bc_bak_fwd_ptr(config.expert_num, 0), lora_up_inter_bc_bak_fwd_ptr(config.expert_num, 0),
    lora_gate_inter_ba_bak_fwd_ptr(config.expert_num, 0), lora_up_inter_ba_bak_fwd_ptr(config.expert_num, 0),
    lora_gate_output_bc_bak_fwd_ptr(config.expert_num, 0), lora_up_output_bc_bak_fwd_ptr(config.expert_num, 0),
    lora_down_inter_bc_bak_fwd_ptr(config.expert_num, 0), lora_down_inter_ba_bak_fwd_ptr(config.expert_num, 0),
    lora_down_inter_ba_inter_ptr(config.expert_num, 0),
    lora_down_output_bc_bak_fwd_ptr(config.expert_num, 0),
    // Backward temporary buffer pointers initialization
    bak_gate_inter_bf16_ptr(config.expert_num, 0), bak_up_inter_bf16_ptr(config.expert_num, 0),
    bak_gate_lora_bf16_ptr(config.expert_num, 0), bak_up_lora_bf16_ptr(config.expert_num, 0),
    bak_down_inter_bf16_ptr(config.expert_num, 0), bak_down_lora_grad_bf16_ptr(config.expert_num, 0),
    bak_temp_gate_ptr(config.expert_num, 0), bak_temp_up_ptr(config.expert_num, 0),
    bak_intermediate_ptr(config.expert_num, 0), bak_down_grad_weighted_ptr(config.expert_num, 0),
    // Gate LoRA gradient computation buffers
    bak_gate_lora_inter_bf16_ptr(config.expert_num, 0), bak_gate_lora_inter_f32_ptr(config.expert_num, 0),
    bak_gate_grad_f32_ptr(config.expert_num, 0), bak_gate_temp_grad_bf16_ptr(config.expert_num, 0),
    bak_gate_temp_grad_f32_ptr(config.expert_num, 0), bak_gate_input_f32_ptr(config.expert_num, 0),
    // Up LoRA gradient computation buffers
    bak_up_lora_inter_bf16_ptr(config.expert_num, 0), bak_up_lora_inter_f32_ptr(config.expert_num, 0),
    bak_up_grad_f32_ptr(config.expert_num, 0), bak_up_temp_grad_bf16_ptr(config.expert_num, 0),
    bak_up_temp_grad_f32_ptr(config.expert_num, 0), bak_up_input_f32_ptr(config.expert_num, 0),
    // Down LoRA gradient computation buffers
    bak_down_lora_inter_bf16_ptr(config.expert_num, 0), bak_down_lora_inter_f32_ptr(config.expert_num, 0),
    bak_down_grad_weighted_f32_ptr(config.expert_num, 0), bak_down_temp_grad_bf16_ptr(config.expert_num, 0),
    bak_down_temp_grad_f32_ptr(config.expert_num, 0), bak_down_intermediate_f32_ptr(config.expert_num, 0)  {
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
    padded_lora_rank_ = (config_.lora_rank + T::K_STEP - 1) / T::K_STEP * T::K_STEP;


    // Local buffers - forward pass
    m_mem_requests_fwd.push_back({(void **)&m_local_input_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_gate_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_up_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_down_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_down_output_lora_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_gate_output_lora_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_fwd.push_back({(void **)&m_local_up_output_lora_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});

    // Local buffers - backward pass (reuse some forward buffers)
    m_mem_requests_bak.push_back({(void **)&m_local_input_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_bak.push_back({(void **)&m_local_gate_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_bak.push_back({(void **)&m_local_up_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_bak.push_back({(void **)&m_local_down_output_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});

    // Gradient buffers - backward pass only
    m_mem_requests_bak.push_back({(void **)&m_local_down_output_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_bak.push_back({(void **)&m_local_down_input_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_bak.push_back({(void **)&m_local_gate_output_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_bak.push_back({(void **)&m_local_up_output_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.intermediate_size});
    m_mem_requests_bak.push_back({(void **)&m_local_gate_input_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_bak.push_back({(void **)&m_local_up_input_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    // LoRA input gradients (for separate base + lora computation)
    m_mem_requests_bak.push_back({(void **)&m_local_gate_input_lora_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});
    m_mem_requests_bak.push_back({(void **)&m_local_up_input_lora_grad_,
        sizeof(ggml_bf16_t) * config_.routed_expert_num * config_.max_len * config_.hidden_size});

    // Token indices - backward pass only
    m_mem_requests_bak.push_back({(void **)&m_local_token_indices_,
        sizeof(int) * config_.routed_expert_num * config_.max_len});
    m_mem_requests_bak.push_back({(void **)&m_local_expert_positions_,
        sizeof(int) * config_.routed_expert_num * config_.max_len});

    for (int i = 0; i < config_.expert_num; i++) {
      // Forward pass buffers
      m_mem_requests_fwd.push_back({(void **)&gate_up_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests_fwd.push_back({(void **)&gate_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests_fwd.push_back({(void **)&up_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests_fwd.push_back({(void **)&down_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests_fwd.push_back({(void **)&down_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.hidden_size)});

      // Forward LoRA buffers (only if lora_rank > 0)
      if (config_.lora_rank > 0) {
        // Gate projection forward LoRA buffers
        // inter_bc_gate: x @ lora_A -> [num_tokens, padded_rank]
        m_mem_requests_fwd.push_back({(void **)&lora_gate_inter_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        // inter_ba_gate: converted from inter_bc_gate for second matmul
        m_mem_requests_fwd.push_back({(void **)&lora_gate_inter_ba_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        // output_bc_gate: inter_ba @ lora_B -> [num_tokens, intermediate_size]
        m_mem_requests_fwd.push_back({(void **)&lora_gate_output_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.intermediate_size)});

        // Up projection forward LoRA buffers
        m_mem_requests_fwd.push_back({(void **)&lora_up_inter_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_fwd.push_back({(void **)&lora_up_inter_ba_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_fwd.push_back({(void **)&lora_up_output_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.intermediate_size)});

        // Down projection forward LoRA buffers
        m_mem_requests_fwd.push_back({(void **)&lora_down_inter_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_fwd.push_back({(void **)&lora_down_inter_ba_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_fwd.push_back({(void **)&lora_down_output_bc_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.hidden_size)});
      }

      m_mem_requests_bak.push_back({(void **)&gate_up_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.hidden_size)});
      m_mem_requests_bak.push_back({(void **)&gate_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests_bak.push_back({(void **)&up_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.intermediate_size)});


      // Backward pass buffers
      m_mem_requests_bak.push_back({(void **)&down_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
      m_mem_requests_bak.push_back({(void **)&down_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.hidden_size)});


      m_mem_requests_bak.push_back({(void **)&gate_t_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.intermediate_size)});

      m_mem_requests_bak.push_back({(void **)&gate_t_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.hidden_size)});

      m_mem_requests_bak.push_back({(void **)&up_t_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.intermediate_size)
      });

      m_mem_requests_bak.push_back({(void **)&up_t_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.hidden_size)
      });

      m_mem_requests_bak.push_back({(void **)&down_t_ba_ptr[i],
          T::BufferA::required_size(config_.max_len, config_.hidden_size)
      });
      
      m_mem_requests_bak.push_back({(void **)&down_t_bc_ptr[i],
          T::BufferC::required_size(config_.max_len, config_.intermediate_size)
      });

      // LoRA gradient computation buffers (only if lora_rank > 0)
      if (config_.lora_rank > 0) {
        // Gate projection LoRA buffers
        m_mem_requests_bak.push_back({(void **)&lora_gate_input_ba_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_inter_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_grad_ba_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_temp_grad_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_A_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_A_t_bb_ptr[i],
            T::BufferB::required_size(config_.hidden_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_B_bb_ptr[i],
            T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_gate_B_t_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&grad_gate_lora_A_bc_ptr[i],
            T::BufferC::required_size(padded_lora_rank_, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&grad_gate_lora_B_bc_ptr[i],
            T::BufferC::required_size(config_.intermediate_size, padded_lora_rank_)});

        // Up projection LoRA buffers
        m_mem_requests_bak.push_back({(void **)&lora_up_inter_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_grad_ba_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_up_temp_grad_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_A_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&lora_up_A_t_bb_ptr[i],
            T::BufferB::required_size(config_.hidden_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_B_bb_ptr[i],
            T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_B_t_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&grad_up_lora_A_bc_ptr[i],
            T::BufferC::required_size(padded_lora_rank_, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&grad_up_lora_B_bc_ptr[i],
            T::BufferC::required_size(config_.intermediate_size, padded_lora_rank_)});

        // Down projection LoRA buffers
        m_mem_requests_bak.push_back({(void **)&lora_down_inter_ba_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_down_lora_inter_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_grad_ba_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&lora_down_temp_grad_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_temp_grad_inter_bc_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_down_A_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_down_B_bb_ptr[i],
            T::BufferB::required_size(config_.hidden_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_A_t_bb_ptr[i],
            T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_B_t_bb_ptr[i],
            T::BufferB::required_size(padded_lora_rank_, config_.hidden_size)});
        m_mem_requests_bak.push_back({(void **)&grad_down_lora_A_bc_ptr[i],
            T::BufferC::required_size(padded_lora_rank_, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&grad_down_lora_B_bc_ptr[i],
            T::BufferC::required_size(config_.hidden_size, padded_lora_rank_)});


        m_mem_requests_bak.push_back({(void **)&lora_gate_inter_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        // inter_ba_gate: converted from inter_bc_gate for second matmul
        m_mem_requests_bak.push_back({(void **)&lora_gate_inter_ba_bak_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        // output_bc_gate: inter_ba @ lora_B -> [num_tokens, intermediate_size]
        m_mem_requests_bak.push_back({(void **)&lora_gate_output_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.intermediate_size)});

        // Up projection forward LoRA buffers
        m_mem_requests_bak.push_back({(void **)&lora_up_inter_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_inter_ba_bak_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_up_output_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.intermediate_size)});

        // Down projection forward LoRA buffers
        m_mem_requests_bak.push_back({(void **)&lora_down_inter_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_inter_ba_bak_fwd_ptr[i],
            T::BufferA::required_size(config_.max_len, padded_lora_rank_)});
        m_mem_requests_bak.push_back({(void **)&lora_down_inter_ba_inter_ptr[i],
            T::BufferA::required_size(config_.max_len, config_.intermediate_size)});
        m_mem_requests_bak.push_back({(void **)&lora_down_output_bc_bak_fwd_ptr[i],
            T::BufferC::required_size(config_.max_len, config_.hidden_size)});

        // Backward temporary buffers (for avoiding aligned_alloc in backward)
        // Forward recomputation buffers
        m_mem_requests_bak.push_back({(void **)&bak_gate_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_up_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_gate_lora_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_up_lora_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_down_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_down_lora_grad_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_temp_gate_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_temp_up_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_intermediate_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_down_grad_weighted_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * config_.hidden_size});

        // Gate LoRA gradient computation buffers
        m_mem_requests_bak.push_back({(void **)&bak_gate_lora_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_gate_lora_inter_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_gate_grad_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_gate_temp_grad_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_gate_temp_grad_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_gate_input_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.hidden_size});

        // Up LoRA gradient computation buffers
        m_mem_requests_bak.push_back({(void **)&bak_up_lora_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_up_lora_inter_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_up_grad_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.intermediate_size});
        m_mem_requests_bak.push_back({(void **)&bak_up_temp_grad_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_up_temp_grad_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_up_input_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.hidden_size});

        // Down LoRA gradient computation buffers
        m_mem_requests_bak.push_back({(void **)&bak_down_lora_inter_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_down_lora_inter_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_down_grad_weighted_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.hidden_size});
        m_mem_requests_bak.push_back({(void **)&bak_down_temp_grad_bf16_ptr[i],
            sizeof(ggml_bf16_t) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_down_temp_grad_f32_ptr[i],
            sizeof(float) * config_.max_len * padded_lora_rank_});
        m_mem_requests_bak.push_back({(void **)&bak_down_intermediate_f32_ptr[i],
            sizeof(float) * config_.max_len * config_.intermediate_size});
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

    m_local_down_output_lora_ptr_.resize(config_.expert_num);
    m_local_gate_output_lora_ptr_.resize(config_.expert_num);
    m_local_up_output_lora_ptr_.resize(config_.expert_num);

    m_local_down_output_grad_ptr_.resize(config_.expert_num);
    m_local_down_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_lora_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_lora_grad_ptr_.resize(config_.expert_num);
    m_local_down_grad_weighted_ptr_.resize(config_.expert_num);
    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);

    // Initialize AMX buffers
    for (uint64_t i = 0; i < config_.expert_num; i++) {
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

      void *gate_bb_lora_A_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.hidden_size));
      gate_bb_lora_A_.push_back(
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, gate_bb_lora_A_ptr));
      void *gate_bb_lora_B_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_));
      gate_bb_lora_B_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, gate_bb_lora_B_ptr));
      void *up_bb_lora_A_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.hidden_size));
      up_bb_lora_A_.push_back(
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, up_bb_lora_A_ptr));
      void *up_bb_lora_B_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_));
      up_bb_lora_B_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, up_bb_lora_B_ptr));
      void *down_bb_lora_A_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size));
      down_bb_lora_A_.push_back(
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, down_bb_lora_A_ptr));
      void *down_bb_lora_B_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, padded_lora_rank_));
      down_bb_lora_B_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, down_bb_lora_B_ptr));

      // Backward LoRA BufferB vectors (for update_lora() to load weights)
      // Gate LoRA backward buffers
      void *lora_gate_A_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.hidden_size));
      lora_gate_A_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, lora_gate_A_bb_mem));
      void *lora_gate_A_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, padded_lora_rank_));
      lora_gate_A_t_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, lora_gate_A_t_bb_mem));
      void *lora_gate_B_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_));
      lora_gate_B_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, lora_gate_B_bb_mem));
      void *lora_gate_B_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size));
      lora_gate_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, lora_gate_B_t_bb_mem));

      // Up LoRA backward buffers
      void *lora_up_A_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.hidden_size));
      lora_up_A_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, lora_up_A_bb_mem));
      void *lora_up_A_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, padded_lora_rank_));
      lora_up_A_t_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, lora_up_A_t_bb_mem));
      void *lora_up_B_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_));
      lora_up_B_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, lora_up_B_bb_mem));
      void *lora_up_B_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size));
      lora_up_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, lora_up_B_t_bb_mem));

      // Down LoRA backward buffers
      void *lora_down_A_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size));
      lora_down_A_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, lora_down_A_bb_mem));
      void *lora_down_B_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, padded_lora_rank_));
      lora_down_B_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, lora_down_B_bb_mem));
      void *lora_down_A_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_));
      lora_down_A_t_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, lora_down_A_t_bb_mem));
      void *lora_down_B_t_bb_mem = std::aligned_alloc(64, T::BufferB::required_size(padded_lora_rank_, config_.hidden_size));
      lora_down_B_t_bb_.push_back(std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, lora_down_B_t_bb_mem));
#endif
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
   * Load and prepare weights for inference
   */
  void load_weights(Backend *backend) {
    // Merge LoRA with base weights
    // merge_lora_weights(backend);

    void *gate_proj_t_ = std::aligned_alloc(
        64, sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size);
    void *up_proj_t_ = std::aligned_alloc(
        64, sizeof(ggml_bf16_t) * config_.expert_num * config_.intermediate_size * config_.hidden_size);
    void *down_proj_t_ = std::aligned_alloc(
        64, sizeof(ggml_bf16_t) * config_.expert_num * config_.hidden_size * config_.intermediate_size);
        
    // Transpose merged weights for backward pass
    transpose_expert(gate_proj_base_, gate_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(up_proj_base_, up_proj_t_, config_.intermediate_size, config_.hidden_size, backend);
    transpose_expert(down_proj_base_, down_proj_t_, config_.hidden_size, config_.intermediate_size, backend);

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
            gate_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)gate_proj_base_ +
                                                       expert_idx * config_.intermediate_size * config_.hidden_size,
                                                   ith, nth);
            up_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)up_proj_base_ +
                                                     expert_idx * config_.intermediate_size * config_.hidden_size,
                                                 ith, nth);
          }
#else
          gate_bb_[expert_idx]->from_mat((ggml_bf16_t *)gate_proj_base_ +
                                             expert_idx * config_.intermediate_size * config_.hidden_size,
                                         ith, nth);
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t *)up_proj_base_ + expert_idx * config_.intermediate_size * config_.hidden_size, ith, nth);
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
            down_bb_numa_[j][expert_idx]->from_mat((ggml_bf16_t *)down_proj_base_ +
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
          down_bb_[expert_idx]->from_mat((ggml_bf16_t *)down_proj_base_ +
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

        free(gate_proj_t_);
        free(up_proj_t_);
        free(down_proj_t_);
  }

  void update_lora(Backend *backend) {
    const int lora_rank = config_.lora_rank;
    const int hidden = config_.hidden_size;
    const int inter = config_.intermediate_size;

    // Process each expert sequentially (padding requires allocation)
    for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
      // Source pointers for this expert
      ggml_bf16_t* gate_A_src = (ggml_bf16_t*)config_.gate_lora_A + expert_idx * lora_rank * hidden;
      ggml_bf16_t* gate_B_src = (ggml_bf16_t*)config_.gate_lora_B + expert_idx * inter * lora_rank;
      ggml_bf16_t* up_A_src = (ggml_bf16_t*)config_.up_lora_A + expert_idx * lora_rank * hidden;
      ggml_bf16_t* up_B_src = (ggml_bf16_t*)config_.up_lora_B + expert_idx * inter * lora_rank;
      ggml_bf16_t* down_A_src = (ggml_bf16_t*)config_.down_lora_A + expert_idx * lora_rank * inter;
      ggml_bf16_t* down_B_src = (ggml_bf16_t*)config_.down_lora_B + expert_idx * hidden * lora_rank;

      // Allocate padded buffers
      // A matrices: [lora_rank, dim] -> [padded_lora_rank, dim]
      // B matrices: [dim, lora_rank] -> [dim, padded_lora_rank]
      ggml_bf16_t* gate_A_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      ggml_bf16_t* gate_B_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      ggml_bf16_t* up_A_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      ggml_bf16_t* up_B_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      ggml_bf16_t* down_A_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      ggml_bf16_t* down_B_padded = (ggml_bf16_t*)aligned_alloc(64, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);

      // Zero out padded buffers
      memset(gate_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      memset(gate_B_padded, 0, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      memset(up_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      memset(up_B_padded, 0, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      memset(down_A_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      memset(down_B_padded, 0, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);


      memcpy(gate_A_padded, gate_A_src, sizeof(ggml_bf16_t) * lora_rank * hidden);
      // up_A: [lora_rank, hidden] -> [padded_lora_rank, hidden]
      memcpy(up_A_padded, up_A_src, sizeof(ggml_bf16_t) * lora_rank * hidden);
      // down_A: [lora_rank, inter] -> [padded_lora_rank, inter]
      memcpy(down_A_padded, down_A_src, sizeof(ggml_bf16_t) * lora_rank * inter);

      // Copy B matrices: [dim, lora_rank] -> [dim, padded_lora_rank]
      // Need row-by-row copy since we're padding columns
      // gate_B: [inter, lora_rank] -> [inter, padded_lora_rank]
      for (int row = 0; row < inter; row++) {
        memcpy(gate_B_padded + row * padded_lora_rank_,
               gate_B_src + row * lora_rank,
               sizeof(ggml_bf16_t) * lora_rank);
      }
      // up_B: [inter, lora_rank] -> [inter, padded_lora_rank]
      for (int row = 0; row < inter; row++) {
        memcpy(up_B_padded + row * padded_lora_rank_,
               up_B_src + row * lora_rank,
               sizeof(ggml_bf16_t) * lora_rank);
      }
      // down_B: [hidden, lora_rank] -> [hidden, padded_lora_rank]
      for (int row = 0; row < hidden; row++) {
        memcpy(down_B_padded + row * padded_lora_rank_,
               down_B_src + row * lora_rank,
               sizeof(ggml_bf16_t) * lora_rank);
      }

      int nth;

      // Load padded weights into AMX buffers
#ifdef USE_NUMA
      int numa_nodes = numa_num_configured_nodes();
      for (int j = 0; j < numa_nodes; j++) {
        // lora_gate_A: N = padded_lora_rank_ <= 256, safe with (0, 1)
        lora_gate_A_bb_numa_[j][expert_idx]->from_mat(gate_A_padded, 0, 1);
        // lora_gate_B: N = intermediate_size, may exceed N_BLOCK(256)
        int nth_inter = T::recommended_nth(inter);
        for (int ith = 0; ith < nth_inter; ith++) {
          lora_gate_B_bb_numa_[j][expert_idx]->from_mat(gate_B_padded, ith, nth_inter);
        }
        // lora_up_A: N = padded_lora_rank_ <= 256, safe with (0, 1)
        lora_up_A_bb_numa_[j][expert_idx]->from_mat(up_A_padded, 0, 1);
        // lora_up_B: N = intermediate_size, may exceed N_BLOCK(256)
        for (int ith = 0; ith < nth_inter; ith++) {
          lora_up_B_bb_numa_[j][expert_idx]->from_mat(up_B_padded, ith, nth_inter);
        }
        // lora_down_A: N = padded_lora_rank_ <= 256, safe with (0, 1)
        lora_down_A_bb_numa_[j][expert_idx]->from_mat(down_A_padded, 0, 1);
        // lora_down_B: N = hidden_size, may exceed N_BLOCK(256)
        int nth_hidden = T::recommended_nth(hidden);
        for (int ith = 0; ith < nth_hidden; ith++) {
          lora_down_B_bb_numa_[j][expert_idx]->from_mat(down_B_padded, ith, nth_hidden);
        }
      }
#else
      nth = T::recommended_nth(inter);
      backend->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          up_bb_lora_B_[expert_idx]->from_mat(up_B_padded, ith, nth);
          gate_bb_lora_B_[expert_idx]->from_mat(gate_B_padded, ith, nth);
        }, nullptr);

      // dump_buffer_b<T>("updatelora_up_bb_lora_B", expert_idx, up_bb_lora_B_[expert_idx], inter, padded_lora_rank_);
      // dump_buffer_b<T>("updatelora_gate_bb_lora_B", expert_idx, gate_bb_lora_B_[expert_idx], inter, padded_lora_rank_);

      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          up_bb_lora_A_[expert_idx]->from_mat(up_A_padded, ith, nth);
          gate_bb_lora_A_[expert_idx]->from_mat(gate_A_padded, ith, nth);
          down_bb_lora_A_[expert_idx]->from_mat(down_A_padded, ith, nth);
        }, nullptr);

      // dump_buffer_b<T>("updatelora_down_bb_lora_A", expert_idx, down_bb_lora_A_[expert_idx], padded_lora_rank_, hidden);
      // dump_buffer_b<T>("updatelora_up_bb_lora_A", expert_idx, up_bb_lora_A_[expert_idx], padded_lora_rank_, hidden);
      // dump_buffer_b<T>("updatelora_gate_bb_lora_A", expert_idx, gate_bb_lora_A_[expert_idx], padded_lora_rank_, hidden);

      nth = T::recommended_nth(hidden);
      backend->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          down_bb_lora_B_[expert_idx]->from_mat(down_B_padded, ith, nth);
        }, nullptr);

      // dump_buffer_b<T>("updatelora_down_bb_lora_B", expert_idx, down_bb_lora_B_[expert_idx], hidden, padded_lora_rank_);

      // === Backward LoRA BufferB loading ===
      // Load weights into backward LoRA BufferB (lora_gate_A_bb_, etc.)
      // These are used in backward pass for gradient computation

      // Gate LoRA backward buffers
      lora_gate_A_bb_[expert_idx]->from_mat(gate_A_padded, 0, 1);
      nth = T::recommended_nth(inter);
      for (int ith = 0; ith < nth; ith++) {
        lora_gate_B_bb_[expert_idx]->from_mat(gate_B_padded, ith, nth);
      }

      // Gate transposed buffers for grad computation
      // gate_lora_B is [inter, padded_rank], need [padded_rank, inter] for gate_B_t
      ggml_bf16_t *gate_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      memset(gate_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      for (int i = 0; i < inter; i++) {
        for (int r = 0; r < lora_rank; r++) {
          gate_B_t_padded[r * inter + i] = gate_B_src[i * lora_rank + r];
        }
      }
      lora_gate_B_t_bb_[expert_idx]->from_mat(gate_B_t_padded, 0, 1);
      free(gate_B_t_padded);

      // gate_lora_A is [padded_rank, hidden], need [hidden, padded_rank] for gate_A_t
      ggml_bf16_t *gate_A_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);
      memset(gate_A_t_padded, 0, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);
      for (int r = 0; r < lora_rank; r++) {
        for (int h = 0; h < hidden; h++) {
          gate_A_t_padded[h * padded_lora_rank_ + r] = gate_A_src[r * hidden + h];
        }
      }
      nth = T::recommended_nth(hidden);
      for (int ith = 0; ith < nth; ith++) {
        lora_gate_A_t_bb_[expert_idx]->from_mat(gate_A_t_padded, ith, nth);
      }
      free(gate_A_t_padded);

      // Up LoRA backward buffers
      nth = T::recommended_nth(padded_lora_rank_);
      for (int ith = 0; ith < nth; ith++) {
        lora_up_A_bb_[expert_idx]->from_mat(up_A_padded, ith, nth);
      }
      nth = T::recommended_nth(inter);
      for (int ith = 0; ith < nth; ith++) {
        lora_up_B_bb_[expert_idx]->from_mat(up_B_padded, ith, nth);
      }

      // Up transposed buffers
      ggml_bf16_t *up_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      memset(up_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * inter);
      for (int i = 0; i < inter; i++) {
        for (int r = 0; r < lora_rank; r++) {
          up_B_t_padded[r * inter + i] = up_B_src[i * lora_rank + r];
        }
      }
      nth = T::recommended_nth(padded_lora_rank_);
      for (int ith = 0; ith < nth; ith++) {
        lora_up_B_t_bb_[expert_idx]->from_mat(up_B_t_padded, ith, nth);
      }
      free(up_B_t_padded);

      ggml_bf16_t *up_A_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);
      memset(up_A_t_padded, 0, sizeof(ggml_bf16_t) * hidden * padded_lora_rank_);
      for (int r = 0; r < lora_rank; r++) {
        for (int h = 0; h < hidden; h++) {
          up_A_t_padded[h * padded_lora_rank_ + r] = up_A_src[r * hidden + h];
        }
      }
      nth = T::recommended_nth(hidden);
      for (int ith = 0; ith < nth; ith++) {
        lora_up_A_t_bb_[expert_idx]->from_mat(up_A_t_padded, ith, nth);
      }
      free(up_A_t_padded);

      // Down LoRA backward buffers
      nth = T::recommended_nth(padded_lora_rank_);
      for (int ith = 0; ith < nth; ith++) {
        lora_down_A_bb_[expert_idx]->from_mat(down_A_padded, ith, nth);
      }
      nth = T::recommended_nth(hidden);
      for (int ith = 0; ith < nth; ith++) {
        lora_down_B_bb_[expert_idx]->from_mat(down_B_padded, ith, nth);
      }

      // Down transposed buffers
      // down_lora_A is [padded_rank, inter], need [inter, padded_rank] for down_A_t
      ggml_bf16_t *down_A_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      memset(down_A_t_padded, 0, sizeof(ggml_bf16_t) * inter * padded_lora_rank_);
      for (int i = 0; i < inter; ++i) {
        for (int r = 0; r < lora_rank; ++r) {
          down_A_t_padded[i * padded_lora_rank_ + r] = down_A_src[r * inter + i];
        }
      }
      nth = T::recommended_nth(inter);
      for (int ith = 0; ith < nth; ith++) {
        lora_down_A_t_bb_[expert_idx]->from_mat(down_A_t_padded, ith, nth);
      }
      free(down_A_t_padded);

      // down_lora_B is [hidden, padded_rank], need [padded_rank, hidden] for down_B_t
      ggml_bf16_t *down_B_t_padded = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      memset(down_B_t_padded, 0, sizeof(ggml_bf16_t) * padded_lora_rank_ * hidden);
      for (int h = 0; h < hidden; h++) {
        for (int r = 0; r < lora_rank; r++) {
          down_B_t_padded[r * hidden + h] = down_B_src[h * lora_rank + r];
        }
      }
      lora_down_B_t_bb_[expert_idx]->from_mat(down_B_t_padded, 0, 1);
      free(down_B_t_padded);
#endif

      // Free temporary padded buffers
      free(gate_A_padded);
      free(gate_B_padded);
      free(up_A_padded);
      free(up_B_padded);
      free(down_A_padded);
      free(down_B_padded);
    }
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

    // printf("SFT_ROUTE_MOE %p forward %d pass: qlen=%d, k=%d, use_amx=%d: start\n", this, cot++, qlen, k, use_amx);

    // Note: update_lora() must be called externally before forward() when LoRA weights change
    shared_mem_buffer.alloc((void*)((uint64_t)this+1), m_mem_requests_fwd);

    std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

    // Forward LoRA buffers
    std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_output_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_up_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_up_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_up_output_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_down_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_down_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_down_output_bc_;

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
      // printf("Allocated gate_up_ba_ for expert %lu at ptr %p\n", i, gate_up_ba_ptr[i]);
      gate_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
      // printf("Allocated gate_bc_ for expert %lu at ptr %p\n", i, gate_bc_ptr[i]);
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
      // printf("Allocated up/_bc_ for expert %lu at ptr %p\n", i, up_bc_ptr[i]);
      down_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, down_ba_ptr[i]));
      // printf("Allocated down_ba_ for expert %lu at ptr %p\n", i, down_ba_ptr[i]);
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, down_bc_ptr[i]));
      // printf("Allocated down_bc_ for expert %lu at ptr %p\n", i, down_bc_ptr[i]);

      // Initialize forward LoRA buffers (only if lora_rank > 0)
      if (config_.lora_rank > 0) {
        // Gate projection forward LoRA buffers
        lora_gate_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_gate_inter_bc_fwd_ptr[i]));
        lora_gate_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_gate_inter_ba_fwd_ptr[i]));
        lora_gate_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, lora_gate_output_bc_fwd_ptr[i]));

        // Up projection forward LoRA buffers
        lora_up_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_up_inter_bc_fwd_ptr[i]));
        lora_up_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_up_inter_ba_fwd_ptr[i]));
        lora_up_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, lora_up_output_bc_fwd_ptr[i]));

        // Down projection forward LoRA buffers
        lora_down_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_inter_bc_fwd_ptr[i]));
        lora_down_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_down_inter_ba_fwd_ptr[i]));
        lora_down_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.hidden_size, lora_down_output_bc_fwd_ptr[i]));
      }
    }
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
      m_local_down_output_lora_ptr_[i] = m_local_down_output_lora_ + offset * config_.hidden_size;
      m_local_gate_output_lora_ptr_[i] = m_local_gate_output_lora_ + offset * config_.intermediate_size;
      m_local_up_output_lora_ptr_[i] = m_local_up_output_lora_ + offset * config_.intermediate_size;
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

    // // DUMP: routing info and packed input
    // dump_routing_info(qlen, k, expert_ids, weights, m_local_num_);
    // for (int e = 0; e < config_.expert_num; e++) {
    //   if (m_local_num_[e] > 0) {
    //     dump_bf16_matrix("packed_input", e, m_local_input_ptr_[e],
    //                      m_local_num_[e], config_.hidden_size);
    //   }
    // }

    // Prepare input buffers
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);


    // for (int e = 0; e < config_.expert_num; e++) {
    //   if (m_local_num_[e] > 0) {
    //     dump_bf16_matrix("fwd_input", e, m_local_input_ptr_[e],
    //                     m_local_num_[e], config_.hidden_size);
    //     auto test = std::aligned_alloc(64, sizeof(ggml_bf16_t) * config_.max_len * config_.hidden_size);
    //     auto testa = std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, test);
    //     memset(test, 0, sizeof(ggml_bf16_t) * config_.max_len * config_.hidden_size);
    //     testa->from_mat(m_local_num_[e], m_local_input_ptr_[e], 0, 1);
    //     // dump_buffer_a<T>("fwd_gate_up_ba_test_", e, testa, m_local_num_[e], config_.hidden_size);
        
    //     // dump_buffer_a<T>("fwd_gate_up_ba_AAA_", e, gate_up_ba_[e], m_local_num_[e], config_.hidden_size);
    //   }
    // }

    // Compute gate and up projections
    int nth = T::recommended_nth(config_.intermediate_size);


    // x[m_local_num_[expert_idx], hidden_size] * gate_bb_lora_A_[hidden_size, r] -> inter_bc_gate[m_local_num_[expert_idx], r]
    // inter_bc_gate (BufferC) -> inter_gate_ptr_f32 (align_alloc) -> inter_gate_ptr_bf16 (align_alloc) -> inter_ba_gate (BufferA)
    // inter_ba[m_local_num_[expert_idx], r] * gate_bb_lora_B_[r, intermediate_size] -> m_local_gate_output_lora_ptr_[m_local_num_[expert_idx], intermediate_size] 

    // x[m_local_num_[expert_idx], hidden_size] * up_bb_lora_A_[hidden_size, r] -> inter_bc_up[m_local_num_[expert_idx], r]
    // inter_bc_up (BufferC) -> inter_up_ptr_f32 (align_alloc) -> inter_up_ptr_bf16 (align_alloc) -> inter_ba_up (BufferA)
    // inter_ba[m_local_num_[expert_idx], r] * up_bb_lora_B_[r, intermediate_size] -> m_local_up_output_lora_ptr_[m_local_num_[expert_idx], intermediate_size] 

    // m_local_gate_output_ptr_ += m_local_gate_output_lora_ptr_
    // m_local_up_output_ptr_ += m_local_up_output_lora_ptr_

    // do the same as down

    // LoRA forward stage 1: x @ lora_A -> inter (for gate and up projections)
    if (config_.lora_rank > 0) {
      int nth_lora = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(nth_lora * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth_lora];
        int ith = task_id % nth_lora;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Gate LoRA stage 1: x @ gate_lora_A -> lora_gate_inter_bc_
        // [num_tokens, hidden_size] @ [hidden_size, padded_rank] -> [num_tokens, padded_rank]
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                     gate_up_ba_[expert_idx], gate_bb_lora_A_[expert_idx],
                     lora_gate_inter_bc_[expert_idx], ith, nth_lora, use_amx);

        // Up LoRA stage 1: x @ up_lora_A -> lora_up_inter_bc_
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                     gate_up_ba_[expert_idx], up_bb_lora_A_[expert_idx],
                     lora_up_inter_bc_[expert_idx], ith, nth_lora, use_amx);
      }, nullptr);

      // Convert BufferC to BufferA for LoRA stage 2
      backend->do_work_stealing_job(activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        ggml_bf16_t *gate_inter_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * padded_lora_rank_);
        ggml_bf16_t *up_inter_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * padded_lora_rank_);

        lora_gate_inter_bc_[expert_idx]->to_mat(num_tokens, gate_inter_bf16, 0, 1);
        lora_up_inter_bc_[expert_idx]->to_mat(num_tokens, up_inter_bf16, 0, 1);

        lora_gate_inter_ba_[expert_idx]->from_mat(num_tokens, gate_inter_bf16, 0, 1);
        lora_up_inter_ba_[expert_idx]->from_mat(num_tokens, up_inter_bf16, 0, 1);

        free(gate_inter_bf16);
        free(up_inter_bf16);
      }, nullptr);

      // LoRA forward stage 2: inter @ lora_B -> lora_output (for gate and up projections)
      backend->do_work_stealing_job(nth * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Gate LoRA stage 2: lora_gate_inter_ba_ @ gate_lora_B -> lora_gate_output_bc_
        // [num_tokens, padded_rank] @ [padded_rank, intermediate_size] -> [num_tokens, intermediate_size]
        amx::mat_mul(num_tokens, config_.intermediate_size, padded_lora_rank_,
                     lora_gate_inter_ba_[expert_idx], gate_bb_lora_B_[expert_idx],
                     lora_gate_output_bc_[expert_idx], ith, nth, use_amx);

        // Up LoRA stage 2: lora_up_inter_ba_ @ up_lora_B -> lora_up_output_bc_
        amx::mat_mul(num_tokens, config_.intermediate_size, padded_lora_rank_,
                     lora_up_inter_ba_[expert_idx], up_bb_lora_B_[expert_idx],
                     lora_up_output_bc_[expert_idx], ith, nth, use_amx);
      }, nullptr);

      // Convert LoRA outputs from BufferC tile-blocked format to linear bf16 format
      backend->do_work_stealing_job(nth * activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Convert gate LoRA output to linear format
        lora_gate_output_bc_[expert_idx]->to_mat(num_tokens, m_local_gate_output_lora_ptr_[expert_idx], ith, nth);
        // Convert up LoRA output to linear format
        lora_up_output_bc_[expert_idx]->to_mat(num_tokens, m_local_up_output_lora_ptr_[expert_idx], ith, nth);
      }, nullptr);

      // DUMP: gate and up LoRA outputs (after conversion to linear bf16 format)
      // for (int e = 0; e < config_.expert_num; e++) {
      //   if (m_local_num_[e] > 0) {
      //     dump_bf16_matrix("gate_lora_output", e, m_local_gate_output_lora_ptr_[e],
      //                      m_local_num_[e], config_.intermediate_size);
      //     dump_bf16_matrix("up_lora_output", e, m_local_up_output_lora_ptr_[e],
      //                      m_local_num_[e], config_.intermediate_size);
      //   }
      // }
    }


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
          }, nullptr);

    // for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("gate_base_output", e, m_local_gate_output_ptr_[e],
    //                        m_local_num_[e], config_.intermediate_size);
    //       dump_bf16_matrix("up_base_output", e, m_local_up_output_ptr_[e],
    //                        m_local_num_[e], config_.intermediate_size);
    //   }
    // }


    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          // Apply activation: gate * up
          auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            ggml_bf16_t *gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
            ggml_bf16_t *up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
            for (int j = n_start; j < n_end; j += 32) {
              __m512 gate_val0, gate_val1, up_val0, up_val1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_output_ptr + j), &gate_val0, &gate_val1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_output_ptr + j), &up_val0, &up_val1);

              // Add LoRA output if enabled
              if (config_.lora_rank > 0) {
                ggml_bf16_t *gate_lora_bf16_ptr = &m_local_gate_output_lora_ptr_[expert_idx][i * config_.intermediate_size + j];
                ggml_bf16_t *up_lora_bf16_ptr = &m_local_up_output_lora_ptr_[expert_idx][i * config_.intermediate_size + j];
                __m512 scaling = _mm512_set1_ps(config_.lora_scaling);

                // Convert LoRA bf16 to f32
                __m512 gate_lora0, gate_lora1, up_lora0, up_lora1;
                avx512_32xbf16_to_32xfp32((__m512i *)gate_lora_bf16_ptr, &gate_lora0, &gate_lora1);
                avx512_32xbf16_to_32xfp32((__m512i *)up_lora_bf16_ptr, &up_lora0, &up_lora1);

                // gate_output += gate_lora_output * scaling
                gate_val0 = _mm512_fmadd_ps(gate_lora0, scaling, gate_val0);
                gate_val1 = _mm512_fmadd_ps(gate_lora1, scaling, gate_val1);

                // up_output += up_lora_output * scaling
                up_val0 = _mm512_fmadd_ps(up_lora0, scaling, up_val0);
                up_val1 = _mm512_fmadd_ps(up_lora1, scaling, up_val1);
              }

              __m512 result0 = _mm512_mul_ps(act_fn_route(gate_val0), up_val0);
              __m512 result1 = _mm512_mul_ps(act_fn_route(gate_val1), up_val1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_output_ptr + j));
            }
          }
        },
        nullptr);

    // DUMP: activation output (stored in m_local_gate_output_ptr_ as bf16)
    // Note: BufferC data is tile-blocked, can't dump directly. But m_local_gate/up_output_ptr_
    // contain linearized data after to_mat(), before activation they have base output only.
    // After activation loop, m_local_gate_output_ptr_ contains silu(gate_total) * up_total
    // for (int e = 0; e < config_.expert_num; e++) {
    //   if (m_local_num_[e] > 0) {
    //     dump_bf16_matrix("activation_output", e, m_local_gate_output_ptr_[e],
    //                      m_local_num_[e], config_.intermediate_size);
    //   }
    // }

    // Prepare down projection input
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Down LoRA forward (if enabled)
    if (config_.lora_rank > 0) {
      // Down LoRA stage 1: intermediate @ down_lora_A -> inter
      int nth_lora = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(nth_lora * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth_lora];
        int ith = task_id % nth_lora;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Down LoRA stage 1: down_input @ down_lora_A -> lora_down_inter_bc_
        // [num_tokens, intermediate_size] @ [intermediate_size, padded_rank] -> [num_tokens, padded_rank]
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.intermediate_size,
                     down_ba_[expert_idx], down_bb_lora_A_[expert_idx],
                     lora_down_inter_bc_[expert_idx], ith, nth_lora, use_amx);
      }, nullptr);

      // Convert BufferC to BufferA for down LoRA stage 2
      backend->do_work_stealing_job(activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Allocate temporary bf16 buffer for conversion
        ggml_bf16_t *down_inter_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * num_tokens * padded_lora_rank_);
        lora_down_inter_bc_[expert_idx]->to_mat(num_tokens, down_inter_bf16, 0, 1);
        lora_down_inter_ba_[expert_idx]->from_mat(num_tokens, down_inter_bf16, 0, 1);
                  
        // dump_bf16_matrix("down_lora_inter", expert_idx, down_inter_bf16,
        //                 m_local_num_[expert_idx], padded_lora_rank_);

        free(down_inter_bf16);
      }, nullptr);

      // Down LoRA stage 2: inter @ down_lora_B -> lora_down_output
      nth = T::recommended_nth(config_.hidden_size);
      backend->do_work_stealing_job(nth * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Down LoRA stage 2: lora_down_inter_ba_ @ down_lora_B -> lora_down_output_bc_
        // [num_tokens, padded_rank] @ [padded_rank, hidden_size] -> [num_tokens, hidden_size]
        amx::mat_mul(num_tokens, config_.hidden_size, padded_lora_rank_,
                     lora_down_inter_ba_[expert_idx], down_bb_lora_B_[expert_idx],
                     lora_down_output_bc_[expert_idx], ith, nth, use_amx);

        lora_down_output_bc_[expert_idx]->to_mat(num_tokens, m_local_down_output_lora_ptr_[expert_idx], ith, nth);
      }, nullptr);

        // DUMP: lora_down_inter_ba_ and down_bb_lora_B_ before mat_mul
        // dump_buffer_a<T>("lora_down_inter_ba", 0, lora_down_inter_ba_[0],
        //                  m_local_num_[0], padded_lora_rank_);
        // dump_buffer_b<T>("down_bb_lora_B", 0, down_bb_lora_B_[0],
        //                  config_.hidden_size, padded_lora_rank_);

        // amx::mat_mul(m_local_num_[0], config_.hidden_size, padded_lora_rank_,
        //              lora_down_inter_ba_[0], down_bb_lora_B_[0],
        //              lora_down_output_bc_[0], 0, 2, use_amx);

        // amx::mat_mul(m_local_num_[0], config_.hidden_size, padded_lora_rank_,
        //              lora_down_inter_ba_[0], down_bb_lora_B_[0],
        //              lora_down_output_bc_[0], 1, 2, use_amx);

        // lora_down_output_bc_[0]->to_mat(m_local_num_[0], m_local_down_output_lora_ptr_[0], 0, 2);
        // lora_down_output_bc_[0]->to_mat(m_local_num_[0], m_local_down_output_lora_ptr_[0], 1, 2);

      // // DUMP: down LoRA output (before adding to base)
      // for (int e = 0; e < config_.expert_num; e++) {
      //   if (m_local_num_[e] > 0) {

      //     dump_bf16_matrix("down_lora_output", e, m_local_down_output_lora_ptr_[e],
      //                     m_local_num_[e], config_.hidden_size);
      //   }
      // }
    }
    

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
          
    }, nullptr);

    // for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("down_base_output", e, m_local_down_output_ptr_[e],
    //                       m_local_num_[e], config_.hidden_size);
    //     }
    // }

    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          // Add down LoRA output if enabled
          // Use m_local_down_output_lora_ptr_ which was converted to linear bf16 format via to_mat
          if (config_.lora_rank > 0) {
            __m512 scaling = _mm512_set1_ps(config_.lora_scaling);

            auto [n_start, n_end] = T::split_range_n(config_.hidden_size, ith, nth);
            for (int i = 0; i < m_local_num_[expert_idx]; i++) {
              ggml_bf16_t *down_output_ptr = &m_local_down_output_ptr_[expert_idx][i * config_.hidden_size];
              ggml_bf16_t *down_lora_ptr = &m_local_down_output_lora_ptr_[expert_idx][i * config_.hidden_size];
              for (int j = n_start; j < n_end; j += 32) {
                __m512 down_val0, down_val1;
                avx512_32xbf16_to_32xfp32((__m512i *)(down_output_ptr + j), &down_val0, &down_val1);

                // Load down LoRA output as bf16 and convert to f32
                __m512 down_lora0, down_lora1;
                avx512_32xbf16_to_32xfp32((__m512i *)(down_lora_ptr + j), &down_lora0, &down_lora1);

                // down_output += down_lora_output * scaling
                down_val0 = _mm512_fmadd_ps(down_lora0, scaling, down_val0);
                down_val1 = _mm512_fmadd_ps(down_lora1, scaling, down_val1);

                avx512_32xfp32_to_32xbf16(&down_val0, &down_val1, (__m512i *)(down_output_ptr + j));
              }
            }
          }
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
              // printf("loading weight for token %d, choice %d: %f, i=%d k=%d j=%d\n", i, j, weights[i * k + j], i, k, j);
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

    // // DUMP: final output
    // dump_final_output(output, qlen, config_.hidden_size);

    shared_mem_buffer.dealloc((void*)((uint64_t)this+1));
    // printf("SFT_ROUTE_MOE %p forward %d pass: qlen=%d, k=%d, use_amx=%d: end\n", this, cot++, qlen, k, use_amx);

  }

  /**
   * Backward pass: compute gradients for LoRA fine-tuning
   * Same interface as SFT_AMX_MOE for compatibility
   */
  void backward(int qlen, int k, const uint64_t *expert_ids, const float *weights, const void* input,
                const void *output_grad, void *input_grad, void *grad_weights, Backend *backend) {
    // printf("SFT_ROUTE_MOE %p backward %d pass: qlen=%d, k=%d, use_amx=%d: start\n", this, cot++, qlen, k);

    bool use_amx = (qlen > 4 * config_.expert_num / config_.routed_expert_num);
    int activated_expert = 0;

    // Note: update_lora() must be called externally before backward() when LoRA weights change

    std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;

    std::vector<std::shared_ptr<typename T::BufferA>> gate_t_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> gate_t_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> up_t_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> up_t_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> down_t_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> down_t_bc_;

    // Gate projection LoRA buffers (BufferA/C are local, BufferB are now member variables)
    std::vector<std::shared_ptr<typename T::BufferA>>
        lora_gate_input_ba_; // [num_tokens, hidden_size]
    std::vector<std::shared_ptr<typename T::BufferA>>
        lora_gate_grad_ba_; // [num_tokens, intermediate_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        lora_gate_temp_grad_bc_; // [num_tokens, padded_rank]
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_gate_lora_A_bc_; // [padded_rank, hidden_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_gate_lora_B_bc_; // [intermediate_size, padded_rank]

    // Up projection LoRA buffers
    std::vector<std::shared_ptr<typename T::BufferA>>
        lora_up_grad_ba_; // [num_tokens, intermediate_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        lora_up_temp_grad_bc_; // [num_tokens, padded_rank]
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_up_lora_A_bc_; // [padded_rank, hidden_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_up_lora_B_bc_; // [intermediate_size, padded_rank]

    // Down projection LoRA buffers
    std::vector<std::shared_ptr<typename T::BufferC>>
        lora_down_lora_inter_bc_; // [num_tokens, padded_rank]
    std::vector<std::shared_ptr<typename T::BufferA>>
        lora_down_grad_ba_; // [num_tokens, hidden_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        lora_down_temp_grad_bc_; // [num_tokens, padded_rank]
    std::vector<std::shared_ptr<typename T::BufferC>>
        lora_down_temp_grad_inter_bc_; // [num_tokens, intermediate_size] for mat_mul output
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_down_lora_A_bc_; // [padded_rank, intermediate_size]
    std::vector<std::shared_ptr<typename T::BufferC>>
        grad_down_lora_B_bc_; // [hidden_size, padded_rank]


    std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_output_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_up_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_up_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_up_output_bc_;
    std::vector<std::shared_ptr<typename T::BufferC>> lora_down_inter_bc_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_down_inter_ba_;
    std::vector<std::shared_ptr<typename T::BufferA>> lora_down_inter_ba_inter_;  // [num_tokens, intermediate_size]
    std::vector<std::shared_ptr<typename T::BufferC>> lora_down_output_bc_;


    shared_mem_buffer.alloc((void*)((uint64_t)this+1), m_mem_requests_bak);

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, gate_up_ba_ptr[i]));
      // printf("Allocated gate_up_ba_ for expert %lu at ptr %p\n", i, gate_up_ba_ptr[i]);
      gate_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, gate_bc_ptr[i]));
      // printf("Allocated gate_bc_ for expert %lu at ptr %p\n", i, gate_bc_ptr[i]);
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, up_bc_ptr[i]));
      // printf("Allocated up_bc_ for expert %lu at ptr %p\n", i, up_bc_ptr[i]);

      gate_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, gate_t_ba_ptr[i]));
      gate_t_bc_.push_back(
          std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, gate_t_bc_ptr[i]));
      up_t_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, up_t_ba_ptr[i]));
      up_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, up_t_bc_ptr[i]));
      down_t_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, down_t_ba_ptr[i]));
      down_t_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, down_t_bc_ptr[i]));

      // Initialize LoRA gradient buffers (only if lora_rank > 0)
      // Note: BufferB objects (lora_*_A_bb_, lora_*_B_bb_, etc.) are now member variables
      //       initialized in constructor and loaded in update_lora()
      if (config_.lora_rank > 0) {
        // Gate projection LoRA buffers (BufferA/C only - BufferB are members)
        lora_gate_input_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.hidden_size, lora_gate_input_ba_ptr[i]));
        lora_gate_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_gate_grad_ba_ptr[i]));
        lora_gate_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_gate_temp_grad_bc_ptr[i]));
        grad_gate_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.hidden_size, grad_gate_lora_A_bc_ptr[i]));
        grad_gate_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.intermediate_size, padded_lora_rank_, grad_gate_lora_B_bc_ptr[i]));

        // Up projection LoRA buffers
        lora_up_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_up_grad_ba_ptr[i]));
        lora_up_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_up_temp_grad_bc_ptr[i]));
        grad_up_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.hidden_size, grad_up_lora_A_bc_ptr[i]));
        grad_up_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.intermediate_size, padded_lora_rank_, grad_up_lora_B_bc_ptr[i]));

        // Down projection LoRA buffers
        lora_down_lora_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_lora_inter_bc_ptr[i]));
        lora_down_grad_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.hidden_size, lora_down_grad_ba_ptr[i]));
        lora_down_temp_grad_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_temp_grad_bc_ptr[i]));
        lora_down_temp_grad_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, lora_down_temp_grad_inter_bc_ptr[i]));
        grad_down_lora_A_bc_.push_back(std::make_shared<typename T::BufferC>(
            padded_lora_rank_, config_.intermediate_size, grad_down_lora_A_bc_ptr[i]));
        grad_down_lora_B_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.hidden_size, padded_lora_rank_, grad_down_lora_B_bc_ptr[i]));

        // Gate projection forward LoRA buffers
        lora_gate_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_gate_inter_bc_bak_fwd_ptr[i]));
        lora_gate_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_gate_inter_ba_bak_fwd_ptr[i]));
        lora_gate_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, lora_gate_output_bc_bak_fwd_ptr[i]));

        // Up projection forward LoRA buffers
        lora_up_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_up_inter_bc_bak_fwd_ptr[i]));
        lora_up_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_up_inter_ba_bak_fwd_ptr[i]));
        lora_up_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.intermediate_size, lora_up_output_bc_bak_fwd_ptr[i]));
        // Down projection forward LoRA buffers
        lora_down_inter_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, padded_lora_rank_, lora_down_inter_bc_bak_fwd_ptr[i]));
        lora_down_inter_ba_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, padded_lora_rank_, lora_down_inter_ba_bak_fwd_ptr[i]));
        lora_down_inter_ba_inter_.push_back(std::make_shared<typename T::BufferA>(
            config_.max_len, config_.intermediate_size, lora_down_inter_ba_inter_ptr[i]));
        lora_down_output_bc_.push_back(std::make_shared<typename T::BufferC>(
            config_.max_len, config_.hidden_size, lora_down_output_bc_bak_fwd_ptr[i]));
      }
    }
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
      m_local_gate_input_lora_grad_ptr_[i] = m_local_gate_input_lora_grad_ + offset * config_.hidden_size;
      m_local_up_input_lora_grad_ptr_[i] = m_local_up_input_lora_grad_ + offset * config_.hidden_size;
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

    // // DUMP: After packing input and output_grad
    // if (is_dump_enabled()) {
    //   dump_bf16_matrix("bwd_input", -1, (ggml_bf16_t*)input, qlen, config_.hidden_size);
    //   dump_bf16_matrix("bwd_output_grad", -1, (ggml_bf16_t*)output_grad, qlen, config_.hidden_size);
    //   for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("bwd_packed_input", e, m_local_input_ptr_[e], m_local_num_[e], config_.hidden_size);
    //       dump_bf16_matrix("bwd_packed_output_grad", e, m_local_down_output_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
    //     }
    //   }
    // }

    // Recompute forward pass (cache could be added for optimization)
    backend->do_work_stealing_job(
        activated_expert, nullptr,
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
          down_t_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_down_output_grad_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // for (int e = 0; e < config_.expert_num; e++) {
    //   if (m_local_num_[e] > 0) {
    //     dump_bf16_matrix("bwd_recompute_input", e, m_local_input_ptr_[e],
    //                     m_local_num_[e], config_.hidden_size);
    //     dump_bf16_matrix("bwd_recompute_down_output_grad", e,
    //                     m_local_down_output_grad_ptr_[e], m_local_num_[e],
    //                     config_.hidden_size);
    //     // dump_buffer_a<T>("gate_up_ba_AAA_", e, gate_up_ba_[e], m_local_num_[e], config_.hidden_size);
    //   }
    // }

    
    int nth = T::recommended_nth(config_.intermediate_size);

    // Step 1: Compute base gate/up projections
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // Recompute forward (base only first)
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

          // Compute down input gradient (base part): dL/dz_base = output_grad @ down_base.T
#ifdef USE_NUMA
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      down_t_ba_[expert_idx], down_t_bb_numa_[Backend::numa_node][expert_idx], down_t_bc_[expert_idx], ith, nth, use_amx);
#else
          amx::mat_mul(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                      down_t_ba_[expert_idx], down_t_bb_[expert_idx], down_t_bc_[expert_idx], ith, nth, use_amx);
#endif
          down_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_input_grad_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // // DUMP: After Step 1 - base gate/up outputs and down_input_grad (base part)
    // if (is_dump_enabled()) {
    //   for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("bwd_step1_gate_base", e, m_local_gate_output_ptr_[e], m_local_num_[e], config_.intermediate_size);
    //       dump_bf16_matrix("bwd_step1_up_base", e, m_local_up_output_ptr_[e], m_local_num_[e], config_.intermediate_size);
    //       dump_bf16_matrix("bwd_step1_down_input_grad_base", e, m_local_down_input_grad_ptr_[e], m_local_num_[e], config_.intermediate_size);
    //     }
    //   }
    // }

    // Note: LoRA weights are now loaded in update_lora() which must be called before backward()

    // Step 2: If LoRA is enabled, compute LoRA contributions and add to base
    if (config_.lora_rank > 0) {
      // LoRA forward stage 1: x @ lora_A -> inter (for gate and up projections)
      int nth_lora = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(nth_lora * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth_lora];
        int ith = task_id % nth_lora;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Gate LoRA stage 1: x @ gate_lora_A -> lora_gate_inter_bc_
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                     gate_up_ba_[expert_idx], gate_bb_lora_A_[expert_idx],
                     lora_gate_inter_bc_[expert_idx], ith, nth_lora, use_amx);

        // Up LoRA stage 1: x @ up_lora_A -> lora_up_inter_bc_
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                     gate_up_ba_[expert_idx], up_bb_lora_A_[expert_idx],
                     lora_up_inter_bc_[expert_idx], ith, nth_lora, use_amx);
      }, nullptr);

      // for (int e = 0; e < config_.expert_num; e++) {
      //   dump_buffer_b<T>("gate_bb_lora_A",e , gate_bb_lora_A_[e], padded_lora_rank_, config_.hidden_size);
      //   // dump_buffer_a<T>("fwd_gate_up_ba", e, gate_up_ba_[e], m_local_num_[e], config_.hidden_size);
      // }


      // Convert BufferC to BufferA for LoRA stage 2
      backend->do_work_stealing_job(activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        ggml_bf16_t *gate_inter_bf16 = (ggml_bf16_t *)bak_gate_inter_bf16_ptr[expert_idx];
        ggml_bf16_t *up_inter_bf16 = (ggml_bf16_t *)bak_up_inter_bf16_ptr[expert_idx];

        lora_gate_inter_bc_[expert_idx]->to_mat(num_tokens, gate_inter_bf16, 0, 1);
        lora_up_inter_bc_[expert_idx]->to_mat(num_tokens, up_inter_bf16, 0, 1);

        lora_gate_inter_ba_[expert_idx]->from_mat(num_tokens, gate_inter_bf16, 0, 1);
        lora_up_inter_ba_[expert_idx]->from_mat(num_tokens, up_inter_bf16, 0, 1);
      }, nullptr);

      // LoRA forward stage 2: inter @ lora_B -> lora_output
      backend->do_work_stealing_job(nth * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Gate LoRA stage 2: lora_gate_inter @ gate_lora_B -> lora_gate_output_bc_
        // Using lora_gate_grad_ba_ which now holds gate_inter
        amx::mat_mul(num_tokens, config_.intermediate_size, padded_lora_rank_,
                     lora_gate_inter_ba_[expert_idx], gate_bb_lora_B_[expert_idx],
                     lora_gate_output_bc_[expert_idx], ith, nth, use_amx);

        // Up LoRA stage 2: lora_up_inter @ up_lora_B -> lora_up_output_bc_
        amx::mat_mul(num_tokens, config_.intermediate_size, padded_lora_rank_,
                     lora_up_inter_ba_[expert_idx], up_bb_lora_B_[expert_idx],
                     lora_up_output_bc_[expert_idx], ith, nth, use_amx);
      }, nullptr);


      // DUMP: Step 2.1 - gate/up LoRA forward outputs (before adding to base)
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       ggml_bf16_t *gate_lora_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       ggml_bf16_t *up_lora_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       memset(gate_lora_bf16, 0, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       memset(up_lora_bf16, 0, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       // N_BLOCK=256, need to call to_mat for each block to export all columns
      //       int n_blocks = (config_.intermediate_size + T::N_BLOCK - 1) / T::N_BLOCK;
      //       for (int b = 0; b < n_blocks; b++) {
      //         lora_gate_output_bc_[e]->to_mat(m_local_num_[e], gate_lora_bf16, b, n_blocks);
      //         lora_up_output_bc_[e]->to_mat(m_local_num_[e], up_lora_bf16, b, n_blocks);
      //       }
      //       // dump_bf16_matrix("bwd_step2_gate_lora_output", e, gate_lora_bf16, m_local_num_[e], config_.intermediate_size);
      //       // dump_bf16_matrix("bwd_step2_up_lora_output", e, up_lora_bf16, m_local_num_[e], config_.intermediate_size);
      //       // dump_buffer_b<T>("gate_bb_lora_B", e, gate_bb_lora_B_[e], config_.intermediate_size, padded_lora_rank_);
      //       // dump_buffer_a<T>("lora_gate_inter_ba_", e, lora_gate_inter_ba_[e], m_local_num_[e], padded_lora_rank_);

      //       free(gate_lora_bf16);
      //       free(up_lora_bf16);
      //     }
      //   }
      // }

      // Add LoRA output to base output (gate_total = gate_base + scaling * gate_lora)
      backend->do_work_stealing_job(nth * activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // Convert LoRA outputs to bf16 and add to base
        ggml_bf16_t *gate_lora_bf16 = (ggml_bf16_t *)bak_gate_lora_bf16_ptr[expert_idx];
        ggml_bf16_t *up_lora_bf16 = (ggml_bf16_t *)bak_up_lora_bf16_ptr[expert_idx];

        lora_gate_output_bc_[expert_idx]->to_mat(num_tokens, gate_lora_bf16, ith, nth);
        lora_up_output_bc_[expert_idx]->to_mat(num_tokens, up_lora_bf16, ith, nth);

        auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
        __m512 scaling = _mm512_set1_ps(config_.lora_scaling);

        for (int t = 0; t < num_tokens; t++) {
          for (int j = n_start; j < n_end; j += 32) {
            // gate_total = gate_base + scaling * gate_lora
            __m512 gate_base0, gate_base1, gate_lora0, gate_lora1;
            avx512_32xbf16_to_32xfp32((__m512i *)(m_local_gate_output_ptr_[expert_idx] + t * config_.intermediate_size + j), &gate_base0, &gate_base1);
            avx512_32xbf16_to_32xfp32((__m512i *)(gate_lora_bf16 + t * config_.intermediate_size + j), &gate_lora0, &gate_lora1);
            gate_base0 = _mm512_fmadd_ps(gate_lora0, scaling, gate_base0);
            gate_base1 = _mm512_fmadd_ps(gate_lora1, scaling, gate_base1);
            avx512_32xfp32_to_32xbf16(&gate_base0, &gate_base1, (__m512i *)(m_local_gate_output_ptr_[expert_idx] + t * config_.intermediate_size + j));

            // up_total = up_base + scaling * up_lora
            __m512 up_base0, up_base1, up_lora0, up_lora1;
            avx512_32xbf16_to_32xfp32((__m512i *)(m_local_up_output_ptr_[expert_idx] + t * config_.intermediate_size + j), &up_base0, &up_base1);
            avx512_32xbf16_to_32xfp32((__m512i *)(up_lora_bf16 + t * config_.intermediate_size + j), &up_lora0, &up_lora1);
            up_base0 = _mm512_fmadd_ps(up_lora0, scaling, up_base0);
            up_base1 = _mm512_fmadd_ps(up_lora1, scaling, up_base1);
            avx512_32xfp32_to_32xbf16(&up_base0, &up_base1, (__m512i *)(m_local_up_output_ptr_[expert_idx] + t * config_.intermediate_size + j));
          }
        }
      }, nullptr);

      // Compute down LoRA contribution to down_input_grad
      // down_input_grad_lora = output_grad @ (B @ A).T = (output_grad @ B.T) @ A.T
      // Stage 1: output_grad @ down_lora_B.T -> temp [num_tokens, padded_rank]
      backend->do_work_stealing_job(nth_lora * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth_lora];
        int ith = task_id % nth_lora;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // output_grad @ down_lora_B.T -> temp
        // down_lora_B is [hidden_size, padded_rank], so B.T is [padded_rank, hidden_size]
        // We use lora_down_B_t_bb_ which is already [padded_rank, hidden_size]
        amx::mat_mul(num_tokens, padded_lora_rank_, config_.hidden_size,
                     down_t_ba_[expert_idx], lora_down_B_t_bb_[expert_idx],
                     lora_down_lora_inter_bc_[expert_idx], ith, nth_lora, use_amx);
      }, nullptr);

      // DUMP: Step 2.2 - down LoRA stage 1 (output_grad @ down_lora_B.T)
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       ggml_bf16_t *down_inter_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * m_local_num_[e] * padded_lora_rank_);
      //       lora_down_lora_inter_bc_[e]->to_mat(m_local_num_[e], down_inter_bf16, 0, 1);
      //       free(down_inter_bf16);
      //     }
      //   }
      // }

      // Convert BufferC to BufferA for down LoRA stage 2
      backend->do_work_stealing_job(activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        ggml_bf16_t *down_inter_bf16 = (ggml_bf16_t *)bak_down_inter_bf16_ptr[expert_idx];
        lora_down_lora_inter_bc_[expert_idx]->to_mat(num_tokens, down_inter_bf16, 0, 1);
        lora_down_inter_ba_[expert_idx]->from_mat(num_tokens, down_inter_bf16, 0, 1);
        // dump_bf16_matrix("bwd_step2_down_lora_inter", expert_idx, down_inter_bf16, num_tokens, padded_lora_rank_);
      }, nullptr);

      // Stage 2: temp @ down_lora_A.T -> down_input_grad_lora [num_tokens, intermediate_size]
      // down_lora_A is [padded_rank, intermediate_size], so A.T is [intermediate_size, padded_rank]
      // We need to create A.T BufferB or compute differently
      // Actually: temp @ A.T where A is [rank, inter], A.T is [inter, rank]
      // mat_mul computes C = A @ B where B is stored in BufferB
      // So we need BufferB to hold A.T = [inter, rank]
      nth = T::recommended_nth(config_.intermediate_size);
      backend->do_work_stealing_job(nth * activated_expert, [&](int _) { T::config(); }, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        // temp @ down_lora_A.T -> down_lora_grad
        // We use lora_down_A_bb_ which is [padded_rank, intermediate_size]
        // mat_mul: [num_tokens, padded_rank] @ [padded_rank, intermediate_size] -> [num_tokens, intermediate_size]
        amx::mat_mul(num_tokens, config_.intermediate_size, padded_lora_rank_,
                     lora_down_inter_ba_[expert_idx], lora_down_A_t_bb_[expert_idx],
                     lora_down_temp_grad_inter_bc_[expert_idx], ith, nth, use_amx);
      }, nullptr);

      // // DUMP: Step 2.3 - down LoRA stage 2 result (before adding to base)
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       ggml_bf16_t *down_lora_grad_bf16 = (ggml_bf16_t *)aligned_alloc(64, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       memset(down_lora_grad_bf16, 0, sizeof(ggml_bf16_t) * m_local_num_[e] * config_.intermediate_size);
      //       // N_BLOCK=256, need to call to_mat for each block to export all columns
      //       int n_blocks = (config_.intermediate_size + T::N_BLOCK - 1) / T::N_BLOCK;
      //       for (int b = 0; b < n_blocks; b++) {
      //         lora_down_temp_grad_inter_bc_[e]->to_mat(m_local_num_[e], down_lora_grad_bf16, b, n_blocks);
      //       }
      //       dump_bf16_matrix("bwd_step2_down_lora_grad", e, down_lora_grad_bf16, m_local_num_[e], config_.intermediate_size);
      //       dump_buffer_b<T>("lora_down_A_t_bb", e, lora_down_A_t_bb_[e],config_.intermediate_size, padded_lora_rank_);
      //       dump_buffer_a<T>("lora_down_inter_ba_", e, lora_down_inter_ba_[e], m_local_num_[e], padded_lora_rank_);
      //       free(down_lora_grad_bf16);
      //     }
      //   }
      // }

      // Add down LoRA contribution to down_input_grad
      backend->do_work_stealing_job(nth * activated_expert, nullptr, [&](int task_id) {
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        int num_tokens = m_local_num_[expert_idx];
        if (num_tokens == 0) return;

        ggml_bf16_t *down_lora_grad_bf16 = (ggml_bf16_t *)bak_down_lora_grad_bf16_ptr[expert_idx];
        lora_down_temp_grad_inter_bc_[expert_idx]->to_mat(num_tokens, down_lora_grad_bf16, ith, nth);

        auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
        __m512 scaling = _mm512_set1_ps(config_.lora_scaling);

        for (int t = 0; t < num_tokens; t++) {
          for (int j = n_start; j < n_end; j += 32) {
            // down_input_grad_total = down_input_grad_base + scaling * down_input_grad_lora
            __m512 base0, base1, lora0, lora1;
            avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_input_grad_ptr_[expert_idx] + t * config_.intermediate_size + j), &base0, &base1);
            avx512_32xbf16_to_32xfp32((__m512i *)(down_lora_grad_bf16 + t * config_.intermediate_size + j), &lora0, &lora1);
            base0 = _mm512_fmadd_ps(lora0, scaling, base0);
            base1 = _mm512_fmadd_ps(lora1, scaling, base1);
            avx512_32xfp32_to_32xbf16(&base0, &base1, (__m512i *)(m_local_down_input_grad_ptr_[expert_idx] + t * config_.intermediate_size + j));
          }
        }
      }, nullptr);

      // // DUMP: After Step 2 - LoRA contributions added (gate_total, up_total, down_input_grad_total)
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       dump_bf16_matrix("bwd_step2_gate_total", e, m_local_gate_output_ptr_[e], m_local_num_[e], config_.intermediate_size);
      //       dump_bf16_matrix("bwd_step2_up_total", e, m_local_up_output_ptr_[e], m_local_num_[e], config_.intermediate_size);
      //       dump_bf16_matrix("bwd_step2_down_input_grad_total", e, m_local_down_input_grad_ptr_[e], m_local_num_[e], config_.intermediate_size);
      //     }
      //   }
      // }
    }

    // Step 3: Compute gate and up output gradients using TOTAL values (gate_total, up_total, down_input_grad_total)
    backend->do_work_stealing_job(
        nth * activated_expert, [&](int _) { T::config(); },
        [&](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

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

              // gate_output_grad = δ * up_total * σ'(gate_total)
              __m512 gate_grad0 = _mm512_mul_ps(down_input_grad0,
                                               _mm512_mul_ps(up_val0, act_fn_grad_route(gate_val0)));
              __m512 gate_grad1 = _mm512_mul_ps(down_input_grad1,
                                               _mm512_mul_ps(up_val1, act_fn_grad_route(gate_val1)));

              // up_output_grad = δ * σ(gate_total)
              __m512 up_grad0 = _mm512_mul_ps(down_input_grad0, act_fn_route(gate_val0));
              __m512 up_grad1 = _mm512_mul_ps(down_input_grad1, act_fn_route(gate_val1));

              avx512_32xfp32_to_32xbf16(&gate_grad0, &gate_grad1, (__m512i *)(gate_output_grad_ptr + j));
              avx512_32xfp32_to_32xbf16(&up_grad0, &up_grad1, (__m512i *)(up_output_grad_ptr + j));
            }
          }
        },
        nullptr);

    // // DUMP: After Step 3 - gate_output_grad and up_output_grad
    // if (is_dump_enabled()) {
    //   for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("bwd_step3_gate_output_grad", e, m_local_gate_output_grad_ptr_[e], m_local_num_[e], config_.intermediate_size);
    //       dump_bf16_matrix("bwd_step3_up_output_grad", e, m_local_up_output_grad_ptr_[e], m_local_num_[e], config_.intermediate_size);
    //     }
    //   }
    // }

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

    // // DUMP: Base input gradients (before LoRA merge)
    // if (is_dump_enabled() && config_.lora_rank > 0) {
    //   for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("bwd_gate_input_grad_base", e, m_local_gate_input_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
    //       dump_bf16_matrix("bwd_up_input_grad_base", e, m_local_up_input_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
    //     }
    //   }
    // }

    // ==================== LoRA Input Gradient Computation ====================
    // Compute: input_grad_lora = output_grad @ lora_B @ lora_A * scaling
    // This is computed in two stages:
    //   Stage 1: temp = output_grad @ lora_B.T  (reuse lora_gate_B_t_bb_)
    //   Stage 2: lora_input_grad = temp @ lora_A.T  (use lora_gate_A_t_bb_)

    if (config_.lora_rank > 0) {
      // Stage 1: output_grad @ lora_B -> temp [M, padded_rank]
      // gate_t_ba_ already contains gate_output_grad, up_t_ba_ contains up_output_grad
      nth = T::recommended_nth(padded_lora_rank_);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            // gate: output_grad [M, inter] @ B.T [rank, inter]^T = [M, rank]
            amx::mat_mul(m_local_num_[expert_idx], padded_lora_rank_, config_.intermediate_size,
                        gate_t_ba_[expert_idx], lora_gate_B_t_bb_[expert_idx],
                        lora_gate_temp_grad_bc_[expert_idx], ith, nth, use_amx);
            // up: output_grad [M, inter] @ B.T [rank, inter]^T = [M, rank]
            amx::mat_mul(m_local_num_[expert_idx], padded_lora_rank_, config_.intermediate_size,
                        up_t_ba_[expert_idx], lora_up_B_t_bb_[expert_idx],
                        lora_up_temp_grad_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      // // DUMP: Stage 1 output (output_grad @ B)
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       // Temp buffer for Stage 1 output
      //       std::vector<ggml_bf16_t> temp_gate(m_local_num_[e] * padded_lora_rank_);
      //       std::vector<ggml_bf16_t> temp_up(m_local_num_[e] * padded_lora_rank_);
      //       lora_gate_temp_grad_bc_[e]->to_mat(m_local_num_[e], temp_gate.data(), 0, 1);
      //       lora_up_temp_grad_bc_[e]->to_mat(m_local_num_[e], temp_up.data(), 0, 1);
      //       dump_bf16_matrix("bwd_gate_input_grad_lora_stage1", e, temp_gate.data(), m_local_num_[e], padded_lora_rank_);
      //       dump_bf16_matrix("bwd_up_input_grad_lora_stage1", e, temp_up.data(), m_local_num_[e], padded_lora_rank_);
      //     }
      //   }
      // }

      // Convert Stage 1 output (BufferC) to BufferA for Stage 2
      // NOTE: Must use temporary buffers because BufferA's internal storage is the same as
      // the bak_fwd_ptr. In-place from_mat would corrupt data during tile layout transformation.
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Use pre-allocated temporary buffers for linear data
            ggml_bf16_t *temp_gate = (ggml_bf16_t *)bak_temp_gate_ptr[expert_idx];
            ggml_bf16_t *temp_up = (ggml_bf16_t *)bak_temp_up_ptr[expert_idx];

            // Convert lora_gate_temp_grad_bc_ to lora_gate_inter_ba_
            lora_gate_temp_grad_bc_[expert_idx]->to_mat(num_tokens, temp_gate, 0, 1);
            lora_gate_inter_ba_[expert_idx]->from_mat(num_tokens, temp_gate, 0, 1);

            // Convert lora_up_temp_grad_bc_ to lora_up_inter_ba_
            lora_up_temp_grad_bc_[expert_idx]->to_mat(num_tokens, temp_up, 0, 1);
            lora_up_inter_ba_[expert_idx]->from_mat(num_tokens, temp_up, 0, 1);
          },
          nullptr);

      // Stage 2: temp @ lora_A.T -> lora_input_grad [M, hidden]
      nth = T::recommended_nth(config_.hidden_size);
      backend->do_work_stealing_job(
          nth * activated_expert, [&](int _) { T::config(); },
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            // gate: temp [M, rank] @ A.T [hidden, rank]^T = [M, hidden]
            amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, padded_lora_rank_,
                        lora_gate_inter_ba_[expert_idx], lora_gate_A_t_bb_[expert_idx],
                        gate_t_bc_[expert_idx], ith, nth, use_amx);
            // up: temp [M, rank] @ A.T [hidden, rank]^T = [M, hidden]
            amx::mat_mul(m_local_num_[expert_idx], config_.hidden_size, padded_lora_rank_,
                        lora_up_inter_ba_[expert_idx], lora_up_A_t_bb_[expert_idx],
                        up_t_bc_[expert_idx], ith, nth, use_amx);
            // Write LoRA input grad to buffer
            gate_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_input_lora_grad_ptr_[expert_idx], ith, nth);
            up_t_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_input_lora_grad_ptr_[expert_idx], ith, nth);
          },
          nullptr);

      // // DUMP: Stage 2 output (before merge) - LoRA input grad without scaling
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       dump_bf16_matrix("bwd_gate_input_grad_lora", e, m_local_gate_input_lora_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
      //       dump_bf16_matrix("bwd_up_input_grad_lora", e, m_local_up_input_lora_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
      //     }
      //   }
      // }

      // Merge: input_grad = input_grad_base + input_grad_lora * scaling
      __m512 scaling_vec = _mm512_set1_ps(config_.lora_scaling);
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t *gate_base = m_local_gate_input_grad_ptr_[expert_idx];
            ggml_bf16_t *gate_lora = m_local_gate_input_lora_grad_ptr_[expert_idx];
            ggml_bf16_t *up_base = m_local_up_input_grad_ptr_[expert_idx];
            ggml_bf16_t *up_lora = m_local_up_input_lora_grad_ptr_[expert_idx];

            int total_elements = num_tokens * config_.hidden_size;
            for (int d = 0; d + 32 <= total_elements; d += 32) {
              // Gate: base + lora * scaling
              __m512 base0, base1, lora0, lora1;
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_base + d), &base0, &base1);
              avx512_32xbf16_to_32xfp32((__m512i *)(gate_lora + d), &lora0, &lora1);
              __m512 result0 = _mm512_fmadd_ps(lora0, scaling_vec, base0);
              __m512 result1 = _mm512_fmadd_ps(lora1, scaling_vec, base1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(gate_base + d));

              // Up: base + lora * scaling
              avx512_32xbf16_to_32xfp32((__m512i *)(up_base + d), &base0, &base1);
              avx512_32xbf16_to_32xfp32((__m512i *)(up_lora + d), &lora0, &lora1);
              result0 = _mm512_fmadd_ps(lora0, scaling_vec, base0);
              result1 = _mm512_fmadd_ps(lora1, scaling_vec, base1);
              avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i *)(up_base + d));
            }
            // Handle remaining elements (if any)
            int remaining_start = (total_elements / 32) * 32;
            for (int d = remaining_start; d < total_elements; d++) {
              float gb = GGML_BF16_TO_FP32(gate_base[d]);
              float gl = GGML_BF16_TO_FP32(gate_lora[d]);
              gate_base[d] = GGML_FP32_TO_BF16(gb + gl * config_.lora_scaling);

              float ub = GGML_BF16_TO_FP32(up_base[d]);
              float ul = GGML_BF16_TO_FP32(up_lora[d]);
              up_base[d] = GGML_FP32_TO_BF16(ub + ul * config_.lora_scaling);
            }
          },
          nullptr);
    }

    // // DUMP: After input gradients computation (gate_input_grad, up_input_grad)
    // if (is_dump_enabled()) {
    //   for (int e = 0; e < config_.expert_num; e++) {
    //     if (m_local_num_[e] > 0) {
    //       dump_bf16_matrix("bwd_gate_input_grad", e, m_local_gate_input_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
    //       dump_bf16_matrix("bwd_up_input_grad", e, m_local_up_input_grad_ptr_[e], m_local_num_[e], config_.hidden_size);
    //     }
    //   }
    // }
  backend->do_work_stealing_job(
    activated_expert, nullptr,
    [&](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      int num_tokens = m_local_num_[expert_idx];
      if (num_tokens == 0) return;

      for (int t = 0; t < num_tokens; ++t) {
        int token_idx  = m_local_token_indices_ptr_[expert_idx][t];
        int expert_pos = m_local_expert_positions_ptr_[expert_idx][t];

        ggml_bf16_t *p_ptr    = &m_local_down_input_grad_ptr_[expert_idx][t * config_.intermediate_size];
        ggml_bf16_t *gate_ptr = &m_local_gate_output_ptr_[expert_idx][t * config_.intermediate_size];
        ggml_bf16_t *up_ptr   = &m_local_up_output_ptr_[expert_idx][t * config_.intermediate_size];

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();

        
        for (int d = 0; d + 32 <= config_.intermediate_size; d += 32) {
          __m512 g0, g1, u0, u1, p0, p1;
          avx512_32xbf16_to_32xfp32((__m512i *)(gate_ptr + d), &g0, &g1);
          avx512_32xbf16_to_32xfp32((__m512i *)(up_ptr   + d), &u0, &u1);
          avx512_32xbf16_to_32xfp32((__m512i *)(p_ptr    + d), &p0, &p1);

          // z = silu(gate) * up
          __m512 z0 = _mm512_mul_ps(act_fn_route(g0), u0);
          __m512 z1 = _mm512_mul_ps(act_fn_route(g1), u1);

          // dot += p * z
          acc0 = _mm512_fmadd_ps(p0, z0, acc0);
          acc1 = _mm512_fmadd_ps(p1, z1, acc1);
        }

        ((float*)grad_weights)[token_idx * k + expert_pos] =  _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1);
      }
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

            ggml_bf16_t *intermediate = (ggml_bf16_t *)bak_intermediate_ptr[expert_idx];
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

      // // DUMP: intermediate (silu(gate) * up) for down LoRA gradients
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     if (m_local_num_[e] > 0) {
      //       dump_bf16_matrix("bwd_intermediate", e, m_local_down_output_ptr_[e], m_local_num_[e], config_.intermediate_size);
      //     }
      //   }
      // }

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
            // Cache the result to avoid redundant computation later
            ggml_bf16_t *down_grad_weighted = (ggml_bf16_t *)bak_down_grad_weighted_ptr[expert_idx];
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
            // Save for later use in grad_B computation (avoid redundant computation)
            m_local_down_grad_weighted_ptr_[expert_idx] = down_grad_weighted;

            // Load intermediate (for down_proj) - use inter buffer with k=intermediate_size
            lora_down_inter_ba_inter_[expert_idx]->from_mat(num_tokens, m_local_down_output_ptr_[expert_idx], 0, 1);
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

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            // BufferC is tile-blocked layout, cannot be accessed directly as row-major
            ggml_bf16_t *lora_inter_bf16 = (ggml_bf16_t *)bak_gate_lora_inter_bf16_ptr[expert_idx];
            lora_gate_inter_bc_[expert_idx]->to_mat(num_tokens, lora_inter_bf16, 0, 1);

            // Convert to f32 for computation
            float *lora_inter = (float *)bak_gate_lora_inter_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                lora_inter[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(lora_inter_bf16[t * padded_lora_rank_ + r]);
              }
            }

            ggml_bf16_t *gate_grad = m_local_gate_output_grad_ptr_[expert_idx];
            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_gate_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            // Pre-convert gate_grad to fp32 to avoid redundant conversions in inner loop
            float *gate_grad_f32 = (float *)bak_gate_grad_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                gate_grad_f32[t * config_.intermediate_size + i] = ggml_bf16_to_fp32(gate_grad[t * config_.intermediate_size + i]);
              }
            }

            float scaling = config_.lora_scaling;

            // grad_B[i, r] += sum_t(gate_grad[t,i] * lora_inter[t,r]) * scaling
            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += gate_grad_f32[t * config_.intermediate_size + i] * lora_inter[t * padded_lora_rank_ + r];
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

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            ggml_bf16_t *temp_grad_bf16 = (ggml_bf16_t *)bak_gate_temp_grad_bf16_ptr[expert_idx];
            lora_gate_temp_grad_bc_[expert_idx]->to_mat(num_tokens, temp_grad_bf16, 0, 1);

            float *temp_grad = (float *)bak_gate_temp_grad_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                temp_grad[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(temp_grad_bf16[t * padded_lora_rank_ + r]);
              }
            }

            ggml_bf16_t *input = m_local_input_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_gate_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;

            // Pre-convert input to fp32 to avoid redundant conversions in inner loop
            float *input_f32 = (float *)bak_gate_input_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                input_f32[t * config_.hidden_size + h] = ggml_bf16_to_fp32(input[t * config_.hidden_size + h]);
              }
            }

            float scaling = config_.lora_scaling;

            // grad_A[r, h] += sum_t(temp_grad[t,r] * input[t,h]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += temp_grad[t * padded_lora_rank_ + r] * input_f32[t * config_.hidden_size + h];
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

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            ggml_bf16_t *lora_inter_bf16 = (ggml_bf16_t *)bak_up_lora_inter_bf16_ptr[expert_idx];
            lora_up_inter_bc_[expert_idx]->to_mat(num_tokens, lora_inter_bf16, 0, 1);

            float *lora_inter = (float *)bak_up_lora_inter_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                lora_inter[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(lora_inter_bf16[t * padded_lora_rank_ + r]);
              }
            }

            ggml_bf16_t *up_grad = m_local_up_output_grad_ptr_[expert_idx];
            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_up_lora_B + expert_idx * config_.intermediate_size * config_.lora_rank;

            // Pre-convert up_grad to fp32 to avoid redundant conversions in inner loop
            float *up_grad_f32 = (float *)bak_up_grad_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                up_grad_f32[t * config_.intermediate_size + i] = ggml_bf16_to_fp32(up_grad[t * config_.intermediate_size + i]);
              }
            }

            float scaling = config_.lora_scaling;

            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += up_grad_f32[t * config_.intermediate_size + i] * lora_inter[t * padded_lora_rank_ + r];
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

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            ggml_bf16_t *temp_grad_bf16 = (ggml_bf16_t *)bak_up_temp_grad_bf16_ptr[expert_idx];
            lora_up_temp_grad_bc_[expert_idx]->to_mat(num_tokens, temp_grad_bf16, 0, 1);

            float *temp_grad = (float *)bak_up_temp_grad_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                temp_grad[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(temp_grad_bf16[t * padded_lora_rank_ + r]);
              }
            }

            ggml_bf16_t *input = m_local_input_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_up_lora_A + expert_idx * config_.lora_rank * config_.hidden_size;

            // Pre-convert input to fp32 to avoid redundant conversions in inner loop
            float *input_f32 = (float *)bak_up_input_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                input_f32[t * config_.hidden_size + h] = ggml_bf16_to_fp32(input[t * config_.hidden_size + h]);
              }
            }

            float scaling = config_.lora_scaling;

            // grad_A[r, h] += sum_t(temp_grad[t,r] * input[t,h]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += temp_grad[t * padded_lora_rank_ + r] * input_f32[t * config_.hidden_size + h];
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
            // Use lora_down_inter_ba_inter_ with k=intermediate_size
            amx::mat_mul(num_tokens, padded_lora_rank_, config_.intermediate_size,
                        lora_down_inter_ba_inter_[expert_idx], lora_down_A_bb_[expert_idx],
                        lora_down_lora_inter_bc_[expert_idx], ith, nth, use_amx);
          },
          nullptr);

      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            ggml_bf16_t *lora_inter_bf16 = (ggml_bf16_t *)bak_down_lora_inter_bf16_ptr[expert_idx];
            lora_down_lora_inter_bc_[expert_idx]->to_mat(num_tokens, lora_inter_bf16, 0, 1);

            // // DUMP: down_lora_inter (intermediate @ down_lora_A.T) - before freeing
            // if (is_dump_enabled()) {
            //   dump_bf16_matrix("bwd_down_lora_inter", expert_idx, lora_inter_bf16, num_tokens, padded_lora_rank_);
            // }

            float *lora_inter = (float *)bak_down_lora_inter_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                lora_inter[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(lora_inter_bf16[t * padded_lora_rank_ + r]);
              }
            }

            // Use cached weighted down_grad (computed earlier, avoid redundant computation)
            ggml_bf16_t *down_grad_weighted = m_local_down_grad_weighted_ptr_[expert_idx];

            // Pre-convert down_grad_weighted to fp32 to avoid redundant conversions in inner loop
            float *down_grad_weighted_f32 = (float *)bak_down_grad_weighted_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                down_grad_weighted_f32[t * config_.hidden_size + h] = ggml_bf16_to_fp32(down_grad_weighted[t * config_.hidden_size + h]);
              }
            }

            ggml_bf16_t *grad_B_dst = (ggml_bf16_t *)config_.grad_down_lora_B + expert_idx * config_.hidden_size * config_.lora_rank;

            float scaling = config_.lora_scaling;
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < config_.lora_rank; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += down_grad_weighted_f32[t * config_.hidden_size + h] * lora_inter[t * padded_lora_rank_ + r];
                }
                float current = ggml_bf16_to_fp32(grad_B_dst[h * config_.lora_rank + r]);
                grad_B_dst[h * config_.lora_rank + r] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }

            m_local_down_grad_weighted_ptr_[expert_idx] = nullptr;
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

            // Export BufferC to linear bf16 format using to_mat(), then convert to f32
            ggml_bf16_t *temp_grad_bf16 = (ggml_bf16_t *)bak_down_temp_grad_bf16_ptr[expert_idx];
            lora_down_temp_grad_bc_[expert_idx]->to_mat(num_tokens, temp_grad_bf16, 0, 1);

            float *temp_grad = (float *)bak_down_temp_grad_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < padded_lora_rank_; r++) {
                temp_grad[t * padded_lora_rank_ + r] = ggml_bf16_to_fp32(temp_grad_bf16[t * padded_lora_rank_ + r]);
              }
            }

            ggml_bf16_t *intermediate = m_local_down_output_ptr_[expert_idx];
            ggml_bf16_t *grad_A_dst = (ggml_bf16_t *)config_.grad_down_lora_A + expert_idx * config_.lora_rank * config_.intermediate_size;

            // Pre-convert intermediate to fp32 to avoid redundant conversions in inner loop
            float *intermediate_f32 = (float *)bak_down_intermediate_f32_ptr[expert_idx];
            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                intermediate_f32[t * config_.intermediate_size + i] = ggml_bf16_to_fp32(intermediate[t * config_.intermediate_size + i]);
              }
            }

            float scaling = config_.lora_scaling;

            // grad_A[r, i] += sum_t(temp_grad[t,r] * intermediate[t,i]) * scaling
            // Note: Only multiply by scaling once (not scaling²)
            for (int r = 0; r < config_.lora_rank; r++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += temp_grad[t * padded_lora_rank_ + r] * intermediate_f32[t * config_.intermediate_size + i];
                }
                float current = ggml_bf16_to_fp32(grad_A_dst[r * config_.intermediate_size + i]);
                grad_A_dst[r * config_.intermediate_size + i] = ggml_fp32_to_bf16(current + sum * scaling);
              }
            }
          },
          nullptr);

      // // DUMP: After LoRA gradient computation - dump all LoRA gradients
      // if (is_dump_enabled()) {
      //   for (int e = 0; e < config_.expert_num; e++) {
      //     ggml_bf16_t *grad_gate_A = (ggml_bf16_t *)config_.grad_gate_lora_A + e * config_.lora_rank * config_.hidden_size;
      //     ggml_bf16_t *grad_gate_B = (ggml_bf16_t *)config_.grad_gate_lora_B + e * config_.intermediate_size * config_.lora_rank;
      //     ggml_bf16_t *grad_up_A = (ggml_bf16_t *)config_.grad_up_lora_A + e * config_.lora_rank * config_.hidden_size;
      //     ggml_bf16_t *grad_up_B = (ggml_bf16_t *)config_.grad_up_lora_B + e * config_.intermediate_size * config_.lora_rank;
      //     ggml_bf16_t *grad_down_A = (ggml_bf16_t *)config_.grad_down_lora_A + e * config_.lora_rank * config_.intermediate_size;
      //     ggml_bf16_t *grad_down_B = (ggml_bf16_t *)config_.grad_down_lora_B + e * config_.hidden_size * config_.lora_rank;

      //     dump_bf16_matrix("bwd_grad_gate_lora_A", e, grad_gate_A, config_.lora_rank, config_.hidden_size);
      //     dump_bf16_matrix("bwd_grad_gate_lora_B", e, grad_gate_B, config_.intermediate_size, config_.lora_rank);
      //     dump_bf16_matrix("bwd_grad_up_lora_A", e, grad_up_A, config_.lora_rank, config_.hidden_size);
      //     dump_bf16_matrix("bwd_grad_up_lora_B", e, grad_up_B, config_.intermediate_size, config_.lora_rank);
      //     dump_bf16_matrix("bwd_grad_down_lora_A", e, grad_down_A, config_.lora_rank, config_.intermediate_size);
      //     dump_bf16_matrix("bwd_grad_down_lora_B", e, grad_down_B, config_.hidden_size, config_.lora_rank);
      //   }
      // }

      // Reset intermediate buffer pointers (buffers are pre-allocated, no need to free)
      backend->do_work_stealing_job(
          activated_expert, nullptr,
          [&](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            m_local_down_output_ptr_[expert_idx] = nullptr;
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

    // // DUMP: Final output - input_grad and grad_weights
    // if (is_dump_enabled()) {
    //   dump_bf16_matrix("bwd_final_input_grad", -1, (ggml_bf16_t*)input_grad, qlen, config_.hidden_size);
    //   dump_f32_matrix("bwd_final_grad_weights", -1, (float*)grad_weights, qlen, k);
    // }
    // printf("SFT_ROUTE_MOE %p backward %d pass: qlen=%d, k=%d, use_amx=%d: end\n", this, cot++, qlen, k);

    shared_mem_buffer.dealloc((void*)((uint64_t)this+1));
  }
};

#endif // CPUINFER_OPERATOR_SFT_AMX_ROUTE_MOE_H