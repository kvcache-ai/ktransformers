/**
 * @Description  : AMX MoE SFT (Supervised Fine-Tuning) implementation with LoRA support.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_SFT_MOE_H
#define CPUINFER_OPERATOR_AMX_SFT_MOE_H

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../cpu_backend/worker_pool.h"
#include "moe.hpp"

// =====================================================
// Backward timing macros for profiling
// Uses counter to skip warmup iterations and print at specific call
// Default: print at 8th call (warmup=5 + backward_test step 0,1,2)
// Can be overridden via SFT_MOE_PRINT_AT_CALL environment variable
// =====================================================
inline int get_print_at_call() {
  static int print_at = -1;
  if (print_at < 0) {
    const char* env = getenv("SFT_MOE_PRINT_AT_CALL");
    print_at = env ? atoi(env) : 8;  // Default: 8th call
  }
  return print_at;
}

#define BACKWARD_TIMER_START()                                       \
  static int _bwd_call_count = 0;                                    \
  _bwd_call_count++;                                                 \
  bool _bwd_should_print = (_bwd_call_count == get_print_at_call()); \
  auto _bwd_start = std::chrono::high_resolution_clock::now();       \
  auto _bwd_last = _bwd_start;

#define BACKWARD_TIMER_CHECKPOINT(name)                                                          \
  do {                                                                                           \
    auto _bwd_now = std::chrono::high_resolution_clock::now();                                   \
    if (_bwd_should_print) {                                                                     \
      double _elapsed = std::chrono::duration<double, std::milli>(_bwd_now - _bwd_last).count(); \
      double _total = std::chrono::duration<double, std::milli>(_bwd_now - _bwd_start).count();  \
      printf("[BWD TIMER] %s: %.3f ms (total: %.3f ms)\n", name, _elapsed, _total);              \
    }                                                                                            \
    _bwd_last = _bwd_now;                                                                        \
  } while (0)

#define BACKWARD_TIMER_END() (void)0

// =====================================================
// BUG-010: NaN Diagnostic Helper Functions
// =====================================================
struct NaNCheckResult {
  int nan_count = 0;
  int inf_count = 0;
  int first_nan_idx = -1;
  float first_nan_input_val = 0.0f;
};

struct Bf16Stats {
  double abs_mean = 0.0;
  double abs_max = 0.0;
  double norm = 0.0;
};

inline Bf16Stats compute_bf16_stats(const ggml_bf16_t* buf, size_t size) {
  Bf16Stats stats;
  if (size == 0 || buf == nullptr) {
    return stats;
  }
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double max_abs = 0.0;
  for (size_t i = 0; i < size; i++) {
    float v = GGML_BF16_TO_FP32(buf[i]);
    double a = std::fabs(static_cast<double>(v));
    sum_abs += a;
    sum_sq += static_cast<double>(v) * static_cast<double>(v);
    if (a > max_abs) {
      max_abs = a;
    }
  }
  stats.abs_mean = sum_abs / static_cast<double>(size);
  stats.abs_max = max_abs;
  stats.norm = std::sqrt(sum_sq);
  return stats;
}

// Check BF16 buffer for NaN/Inf
inline NaNCheckResult check_bf16_buffer_for_nan(const ggml_bf16_t* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    float val = GGML_BF16_TO_FP32(buf[i]);
    if (std::isnan(val)) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
    if (std::isinf(val)) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf("[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d\n", label, result.nan_count, result.inf_count,
           result.first_nan_idx);
  }
  return result;
}

// Check FP32 buffer for NaN/Inf
inline NaNCheckResult check_fp32_buffer_for_nan(const float* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    if (std::isnan(buf[i])) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
    if (std::isinf(buf[i])) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf("[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d\n", label, result.nan_count, result.inf_count,
           result.first_nan_idx);
  }
  return result;
}

// =====================================================
// Dump Utility Functions for debugging
// Controlled by SFT_MOE_DUMP environment variable
// =====================================================
inline bool is_dump_enabled() {
  return false;
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = getenv("SFT_MOE_DUMP");
    enabled = (env && env[0] != '0') ? 1 : 0;
  }
  return enabled == 1;
}

inline const char* get_dump_dir() {
  static const char* dir = nullptr;
  if (dir == nullptr) {
    dir = getenv("SFT_MOE_DUMP_DIR");
    if (dir == nullptr) {
      dir = "./cpp_dump";
    }
  }
  return dir;
}

// Dump BF16 matrix to binary file (format: rows(int32), cols(int32), data(float32))
// tp_idx: TP partition index (-1 for no TP suffix)
// expert_id: Expert index (-1 for no expert suffix)
inline void dump_bf16_matrix(const ggml_bf16_t* data, int rows, int cols, const char* name, int tp_idx = -1,
                             int expert_id = -1) {
  if (!is_dump_enabled()) return;

  char filename[512];
  if (tp_idx >= 0 && expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d_e%d.bin", get_dump_dir(), name, tp_idx, expert_id);
  } else if (tp_idx >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d.bin", get_dump_dir(), name, tp_idx);
  } else if (expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_e%d.bin", get_dump_dir(), name, expert_id);
  } else {
    snprintf(filename, sizeof(filename), "%s/%s.bin", get_dump_dir(), name);
  }

  // Create directory if needed
  char mkdir_cmd[600];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", get_dump_dir());
  system(mkdir_cmd);

  FILE* f = fopen(filename, "wb");
  if (!f) {
    printf("[DUMP ERROR] Cannot open file: %s\n", filename);
    return;
  }

  // Write header
  int32_t dims[2] = {rows, cols};
  fwrite(dims, sizeof(int32_t), 2, f);

  // Convert BF16 to FP32 and write
  for (int i = 0; i < rows * cols; i++) {
    float val = GGML_BF16_TO_FP32(data[i]);
    fwrite(&val, sizeof(float), 1, f);
  }

  fclose(f);
  printf("[CPP DUMP] Saved %s: [%d x %d]\n", filename, rows, cols);
}

// Dump BF16 matrix with scaling factor (for LoRA contributions that need lora_scaling applied)
inline void dump_bf16_matrix_scaled(const ggml_bf16_t* data, int rows, int cols, float scale, const char* name,
                                    int tp_idx = -1, int expert_id = -1) {
  if (!is_dump_enabled()) return;

  char filename[512];
  if (tp_idx >= 0 && expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d_e%d.bin", get_dump_dir(), name, tp_idx, expert_id);
  } else if (tp_idx >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d.bin", get_dump_dir(), name, tp_idx);
  } else if (expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_e%d.bin", get_dump_dir(), name, expert_id);
  } else {
    snprintf(filename, sizeof(filename), "%s/%s.bin", get_dump_dir(), name);
  }

  // Create directory if needed
  char mkdir_cmd[600];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", get_dump_dir());
  system(mkdir_cmd);

  FILE* f = fopen(filename, "wb");
  if (!f) {
    printf("[DUMP ERROR] Cannot open file: %s\n", filename);
    return;
  }

  // Write header
  int32_t dims[2] = {rows, cols};
  fwrite(dims, sizeof(int32_t), 2, f);

  // Convert BF16 to FP32, apply scale, and write
  for (int i = 0; i < rows * cols; i++) {
    float val = GGML_BF16_TO_FP32(data[i]) * scale;
    fwrite(&val, sizeof(float), 1, f);
  }

  fclose(f);
  printf("[CPP DUMP] Saved %s: [%d x %d] (scaled by %.2f)\n", filename, rows, cols, scale);
}

// Dump FP32 matrix to binary file
inline void dump_fp32_matrix(const float* data, int rows, int cols, const char* name, int tp_idx = -1,
                             int expert_id = -1) {
  if (!is_dump_enabled()) return;

  char filename[512];
  if (tp_idx >= 0 && expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d_e%d.bin", get_dump_dir(), name, tp_idx, expert_id);
  } else if (tp_idx >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_tp%d.bin", get_dump_dir(), name, tp_idx);
  } else if (expert_id >= 0) {
    snprintf(filename, sizeof(filename), "%s/%s_e%d.bin", get_dump_dir(), name, expert_id);
  } else {
    snprintf(filename, sizeof(filename), "%s/%s.bin", get_dump_dir(), name);
  }

  // Create directory if needed
  char mkdir_cmd[600];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", get_dump_dir());
  system(mkdir_cmd);

  FILE* f = fopen(filename, "wb");
  if (!f) {
    printf("[DUMP ERROR] Cannot open file: %s\n", filename);
    return;
  }

  // Write header
  int32_t dims[2] = {rows, cols};
  fwrite(dims, sizeof(int32_t), 2, f);

  // Write data
  fwrite(data, sizeof(float), rows * cols, f);

  fclose(f);
  printf("[CPP DUMP] Saved %s: [%d x %d]\n", filename, rows, cols);
}

// Dump routing info to binary file
inline void dump_routing_info(int qlen, int k, const int64_t* expert_ids, const float* weights, int num_experts,
                              const std::vector<int>& m_local_num, int tp_idx = -1) {
  if (!is_dump_enabled()) return;

  char filename[512];
  if (tp_idx >= 0) {
    snprintf(filename, sizeof(filename), "%s/routing_info_tp%d.bin", get_dump_dir(), tp_idx);
  } else {
    snprintf(filename, sizeof(filename), "%s/routing_info.bin", get_dump_dir());
  }

  // Create directory if needed
  char mkdir_cmd[600];
  snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", get_dump_dir());
  system(mkdir_cmd);

  FILE* f = fopen(filename, "wb");
  if (!f) {
    printf("[DUMP ERROR] Cannot open file: %s\n", filename);
    return;
  }

  // Write qlen, k
  int32_t dims[2] = {qlen, k};
  fwrite(dims, sizeof(int32_t), 2, f);

  // Write expert_ids [qlen * k]
  fwrite(expert_ids, sizeof(int64_t), qlen * k, f);

  // Write weights [qlen * k]
  fwrite(weights, sizeof(float), qlen * k, f);

  // Write num_experts and m_local_num
  int32_t ne = num_experts;
  fwrite(&ne, sizeof(int32_t), 1, f);
  for (int i = 0; i < num_experts; i++) {
    int32_t cnt = m_local_num[i];
    fwrite(&cnt, sizeof(int32_t), 1, f);
  }

  fclose(f);
  printf("[CPP DUMP] Saved %s: qlen=%d, k=%d\n", filename, qlen, k);
}

// =====================================================
// Type trait to detect if kernel supports standard mat_mul API
// Only these kernels have the standard amx::mat_mul(m,n,k,ba,bb,bc,ith,nth) overload
// KGroup kernels use mat_mul_kgroup() with different BufferB interface
// =====================================================
template <typename T>
struct supports_standard_mat_mul : std::false_type {};

template <>
struct supports_standard_mat_mul<amx::GemmKernel224BF> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int8> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4_1> : std::true_type {};

template <typename T>
inline constexpr bool supports_standard_mat_mul_v = supports_standard_mat_mul<T>::value;

/**
 * @brief Forward cache structure for gradient checkpointing.
 *
 * Stores intermediate values from forward pass needed for backward computation.
 * Supports multiple cache slots for gradient checkpointing (multiple forwards before backward).
 */
struct ForwardCache {
  // Intermediate values (need to be copied as next layer's forward will overwrite)
  ggml_bf16_t* input_cache = nullptr;         // [qlen, hidden_size]
  ggml_bf16_t* gate_output_cache = nullptr;   // [tokens_total, intermediate_size]
  ggml_bf16_t* up_output_cache = nullptr;     // [tokens_total, intermediate_size]
  ggml_bf16_t* intermediate_cache = nullptr;  // [tokens_total, intermediate_size] (after activation)
  ggml_bf16_t* down_output_cache = nullptr;   // [tokens_total, hidden_size] (for grad_weights)

  // Routing information
  std::vector<int64_t> expert_ids_cache;
  std::vector<float> weights_cache;
  std::vector<int> m_local_num_cache;
  std::vector<std::vector<int>> m_local_pos_cache;
  std::vector<int> m_expert_id_map_cache;
  int qlen_cache = 0;
  int k_cache = 0;
  int activated_expert_cache = 0;

  bool valid = false;
};

/**
 * @brief AMX SFT MoE implementation with LoRA support.
 *
 * Inherits from AMX_MOE_TP and adds:
 * - LoRA computation for gate/up/down projections
 * - Forward cache for gradient checkpointing
 * - Backward pass implementation
 *
 * @tparam T The GEMM kernel type (e.g., GemmKernel224BF, GemmKernel224Int8)
 * @tparam BaseMOE The base MOE class template (default: AMX_MOE_TP, can be AMX_AWQ_MOE_TP or AMX_K2_MOE_TP)
 * @tparam SkipLoRA If true, skip all LoRA computation in backward pass,
 *                  only compute base weight contribution to grad_input. (default: false)
 */
template <class T, template <class> class BaseMOE = AMX_MOE_TP, bool SkipLoRA = false>
class AMX_SFT_MOE_TP : public BaseMOE<T> {
 protected:
  using Base = BaseMOE<T>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_gate_output_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_input_;
  using Base::m_local_input_ptr_;
  using Base::m_local_num_;
  using Base::m_local_pos_;
  using Base::m_local_up_output_;
  using Base::m_local_up_output_ptr_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 private:
  // SFT configuration
  MOESFTConfig sft_config_;

  // LoRA configuration (from MOESFTConfig)
  int lora_rank_;
  float lora_scaling_;

  // LoRA weight pointers (directly pointing to Python tensors)
  ggml_bf16_t* gate_lora_a_;  // [expert_num, lora_rank, hidden_size]
  ggml_bf16_t* gate_lora_b_;  // [expert_num, intermediate_size, lora_rank]
  ggml_bf16_t* up_lora_a_;
  ggml_bf16_t* up_lora_b_;
  ggml_bf16_t* down_lora_a_;
  ggml_bf16_t* down_lora_b_;

  // LoRA intermediate buffer (using shared_mem_buffer pool allocation)
  // For lora_A @ x results
  ggml_bf16_t* lora_intermediate_;  // [max_len * k, lora_rank] - kept for compatibility but not used
  void* lora_intermediate_pool_;
  size_t lora_intermediate_pool_bytes_;

  // Forward cache stack (for gradient checkpointing)
  std::vector<ForwardCache> cache_stack_;
  int cache_stack_top_ = 0;  // Stack top pointer
  int max_cache_depth_;

  // Last backward expert token distribution (for load balancing analysis)
  std::vector<int> last_backward_expert_tokens_;

  // Cache buffer pools
  void* cache_input_pool_ = nullptr;
  void* cache_gate_output_pool_ = nullptr;
  void* cache_up_output_pool_ = nullptr;
  void* cache_intermediate_pool_ = nullptr;
  void* cache_down_output_pool_ = nullptr;  // For grad_weights computation
  size_t cache_slot_bytes_input_;
  size_t cache_slot_bytes_intermediate_;

  // Gradient intermediate buffers
  ggml_bf16_t* grad_intermediate_;  // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_gate_output_;   // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_up_output_;     // [max_len * k, intermediate_size]
  void* grad_intermediate_pool_;
  void* grad_gate_output_pool_;
  void* grad_up_output_pool_;

  // =====================================================
  // AMX-optimized LoRA GEMM buffers (performance optimization)
  // =====================================================

  // Padded lora_rank for AMX alignment (must be multiple of K_STEP=32)
  int padded_lora_rank_;

  // LoRA weight BufferB for AMX GEMM
  // Step 1 weights: lora_A matrices [padded_lora_rank, hidden_size or intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_a_bb_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_a_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> down_lora_a_bb_;  // [expert_num]

  // Step 2 weights: lora_B matrices [output_dim, padded_lora_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_b_bb_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_b_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> down_lora_b_bb_;  // [expert_num]
  // Transposed weights for backward GEMM
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_a_t_bb_;  // [expert_num] [hidden_size, padded_lora_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_a_t_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>>
      gate_lora_b_t_bb_;  // [expert_num] [padded_lora_rank, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_b_t_bb_;  // [expert_num]

  // LoRA intermediate BufferA and BufferC
  // For step 1 output / step 2 input: [num_tokens, padded_lora_rank]
  // Gate and Up need SEPARATE buffers to avoid race condition in parallel execution
  std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_intermediate_ba_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferA>> lora_up_intermediate_ba_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_intermediate_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_intermediate_bc_;    // [expert_num]

  // LoRA step 2 output BufferC (for accumulation before adding to main output)
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_out_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_out_bc_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_down_out_bc_;  // [expert_num]

  // LoRA intermediate output pointers (for step 1 -> step 2)
  // Gate and Up need SEPARATE pointers to avoid race condition in parallel execution
  std::vector<ggml_bf16_t*> lora_gate_intermediate_ptr_;  // [expert_num]
  std::vector<ggml_bf16_t*> lora_up_intermediate_ptr_;    // [expert_num]

  // LoRA buffer pools
  void* lora_bb_pool_ = nullptr;                 // All LoRA weight BufferB
  void* lora_ba_pool_ = nullptr;                 // LoRA intermediate BufferA
  void* lora_bc_inter_pool_ = nullptr;           // LoRA step 1 output BufferC
  void* lora_bc_out_pool_ = nullptr;             // LoRA step 2 output BufferC
  void* lora_intermediate_bf16_pool_ = nullptr;  // BF16 intermediate for step 1->step 2

  // Buffer pool sizes
  size_t lora_bb_pool_bytes_ = 0;
  size_t lora_ba_pool_bytes_ = 0;
  size_t lora_bc_inter_pool_bytes_ = 0;
  size_t lora_bc_out_pool_bytes_ = 0;
  size_t lora_intermediate_bf16_pool_bytes_ = 0;

  // =====================================================
  // Backward pass AMX buffers
  // =====================================================

  // BufferA for grad_output (scattered to per-expert)
  std::vector<std::shared_ptr<typename T::BufferA>> grad_output_ba_;  // [expert_num]

  // BufferC for backward GEMM outputs
  std::vector<std::shared_ptr<typename T::BufferC>> grad_intermediate_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_up_bc_;       // [expert_num]

  // BF16 buffer for scattered grad_output (before quantization to BufferA)
  std::vector<ggml_bf16_t*> grad_output_bf16_ptr_;  // [expert_num]

  // Backward buffer pools
  void* backward_ba_pool_ = nullptr;
  void* backward_bc_pool_ = nullptr;
  void* grad_output_bf16_pool_ = nullptr;

  // Backward buffer pool sizes
  size_t backward_ba_pool_bytes_ = 0;
  size_t backward_bc_pool_bytes_ = 0;
  size_t grad_output_bf16_pool_bytes_ = 0;

  // =====================================================
  // Backward pass BufferB for transposed base weights
  // =====================================================
  // For backward GEMM, we need transposed versions of the base weights:
  // - Forward gate/up: input @ W^T uses gate_bb_[intermediate_size, hidden_size]
  // - Backward gate/up: grad @ W uses BufferB[hidden_size, intermediate_size]
  // - Forward down: intermediate @ W^T uses down_bb_[hidden_size, intermediate_size]
  // - Backward down: grad_output @ W uses BufferB[intermediate_size, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_backward_bb_;  // [hidden_size, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> up_backward_bb_;    // [hidden_size, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> down_backward_bb_;  // [intermediate_size, hidden_size]

  // Backward BufferB pool
  void* backward_bb_pool_ = nullptr;
  size_t backward_bb_pool_bytes_ = 0;

  // Flag to track if backward weights have been prepared
  bool backward_weights_prepared_ = false;

  // Flag to track if LoRA weights have been converted to BufferB format
  bool lora_weights_prepared_ = false;

 public:
  AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    printf("Creating AMX_SFT_MOE_TP layer=%d tp_part=%d at numa %d\n", config.layer_idx, tp_part_idx,
           numa_node_of_cpu(sched_getcpu()));

    // Initialize LoRA configuration
    lora_rank_ = config.lora_rank;
    lora_scaling_ = config.lora_scaling();
    max_cache_depth_ = config.max_cache_depth;

    // Get LoRA weight pointers
    gate_lora_a_ = (ggml_bf16_t*)config.gate_lora_a;
    gate_lora_b_ = (ggml_bf16_t*)config.gate_lora_b;
    up_lora_a_ = (ggml_bf16_t*)config.up_lora_a;
    up_lora_b_ = (ggml_bf16_t*)config.up_lora_b;
    down_lora_a_ = (ggml_bf16_t*)config.down_lora_a;
    down_lora_b_ = (ggml_bf16_t*)config.down_lora_b;

    // Initialize all buffers in a single alloc() to avoid memory overlap
    // (Bug #15: SharedMemBuffer assigns all alloc() calls from same base address)
    init_all_buffers();
  }

  // Constructor to satisfy MOE_TP_PART concept (takes GeneralMOEConfig)
  AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}

  ~AMX_SFT_MOE_TP() {
    // Free LoRA buffers allocated with aligned_alloc
    if (lora_bb_pool_) free(lora_bb_pool_);
    if (lora_ba_pool_) free(lora_ba_pool_);
    if (lora_bc_inter_pool_) free(lora_bc_inter_pool_);
    if (lora_bc_out_pool_) free(lora_bc_out_pool_);
    if (lora_intermediate_bf16_pool_) free(lora_intermediate_bf16_pool_);
    // Free cache buffers allocated with aligned_alloc (Bug #18 fix)
    if (cache_input_pool_) free(cache_input_pool_);
    if (cache_gate_output_pool_) free(cache_gate_output_pool_);
    if (cache_up_output_pool_) free(cache_up_output_pool_);
    if (cache_intermediate_pool_) free(cache_intermediate_pool_);
    if (cache_down_output_pool_) free(cache_down_output_pool_);
    // Free gradient buffers allocated with aligned_alloc (Bug #18c fix)
    if (grad_intermediate_pool_) free(grad_intermediate_pool_);
    if (grad_gate_output_pool_) free(grad_gate_output_pool_);
    if (grad_up_output_pool_) free(grad_up_output_pool_);
    // Free backward pass buffers allocated with aligned_alloc (Bug #18d fix)
    if (backward_ba_pool_) free(backward_ba_pool_);
    if (backward_bc_pool_) free(backward_bc_pool_);
    if (grad_output_bf16_pool_) free(grad_output_bf16_pool_);
    if (backward_bb_pool_) free(backward_bb_pool_);
  }

  /**
   * @brief Set LoRA parameters after construction (Bug #007 fix).
   *
   * This is needed because TP_MOE base class uses GeneralMOEConfig which
   * doesn't have lora_rank/lora_alpha fields, causing object slicing.
   * The TP_MOE_SFT wrapper calls this method to propagate correct values.
   *
   * @param rank LoRA rank (typically 8 or 16)
   * @param alpha LoRA alpha for scaling (lora_scaling = alpha / rank)
   */
  void set_lora_params(int rank, float alpha) {
    lora_rank_ = rank;
    lora_scaling_ = alpha / rank;
  }

  /**
   * @brief SFT Forward pass with optional caching for backward.
   *
   * Computes: output = Σ weights[i] * down_proj(silu(gate_proj(x) + gate_lora(x)) * (up_proj(x) + up_lora(x))) +
   * down_lora(...)
   *
   * @param qlen Number of tokens
   * @param k Number of experts per token
   * @param expert_ids Expert indices [qlen, k]
   * @param weights Expert weights [qlen, k]
   * @param input Input tensor [qlen, hidden_size]
   * @param output Output tensor [qlen, hidden_size]
   * @param save_for_backward Whether to save intermediate values for backward pass
   */
  void forward_sft(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
                   bool save_for_backward) {
    uint64_t _fwd_start_cycles = __rdtsc();
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Step 1: Expert routing (reuse base class logic)
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
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

    // Step 2: Buffer pool allocation (reuse base class logic)
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = Base::gate_up_ba_pool_;
    void* gate_bc_pool_ptr = Base::gate_bc_pool_;
    void* up_bc_pool_ptr = Base::up_bc_pool_;
    void* down_ba_pool_ptr = Base::down_ba_pool_;
    void* down_bc_pool_ptr = Base::down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) {
        continue;
      }

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr =
          (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(Base::buffer_a_required_size(max_m, config_.hidden_size)));

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr +
                                 align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr =
          (void*)((uintptr_t)up_bc_pool_ptr + align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr +
                                 align64(Base::buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr =
          (void*)((uintptr_t)down_bc_pool_ptr + align64(Base::buffer_c_required_size(max_m, config_.hidden_size)));
    }

    // Step 3: Copy input to expert buffers
    auto direct_or_pool = [&](int count, auto&& fn, const char* task_name, int block_size) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr, task_name, block_size);
      }
    };

    direct_or_pool(
        qlen,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
              continue;
            }
            memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
                   (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
          }
        },
        "fwd_pack_input", 1);

    // DUMP: Routing info and packed input
    if (is_dump_enabled()) {
      dump_routing_info(qlen, k, expert_ids, weights, config_.expert_num, m_local_num_, tp_part_idx);
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          dump_bf16_matrix(m_local_input_ptr_[expert_idx], m_local_num_[expert_idx], config_.hidden_size,
                           "packed_input", tp_part_idx, expert_idx);
        }
      }
    }

    // Step 4: Quantize input
    direct_or_pool(
        activated_expert,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
        },
        "fwd_quantize_in", 1);

    // Step 5: Gate + Up GEMM (base projection)
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr, "fwd_gate_up_gemm", 1);

    // DUMP: Gate/Up base output (before LoRA)
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          dump_bf16_matrix(m_local_gate_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "gate_base_output", tp_part_idx, expert_idx);
          dump_bf16_matrix(m_local_up_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "up_base_output", tp_part_idx, expert_idx);
        }
      }
    }

    // Step 5.5: Gate + Up LoRA
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      if constexpr (supports_standard_mat_mul_v<T>) {
        compute_lora_gate_up_amx(qlen, activated_expert);  // AMX-optimized path
      } else {
        compute_lora_gate_up(qlen, activated_expert);  // For-loop fallback for KGroup kernels
      }
    }

    // DUMP: Gate/Up output (after LoRA, before activation)
    if (is_dump_enabled() && gate_lora_a_ != nullptr) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          // Note: After LoRA, gate/up outputs have been updated in-place
          // These now include base + lora
          dump_bf16_matrix(m_local_gate_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "gate_lora_output", tp_part_idx, expert_idx);
          dump_bf16_matrix(m_local_up_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "up_lora_output", tp_part_idx, expert_idx);
        }
      }
    }

    // Save gate/up outputs before activation (for backward)
    if (save_for_backward) {
      ForwardCache& cache = push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert, input);
    }

    // DUMP: Activation input (gate_out and up_out before activation)
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          dump_bf16_matrix(m_local_gate_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "activation_input_gate", tp_part_idx, expert_idx);
          dump_bf16_matrix(m_local_up_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "activation_input_up", tp_part_idx, expert_idx);
        }
      }
    }

    // Step 6: Activation (silu(gate) * up)
    Base::apply_activation(activated_expert, nth, qlen);

    // DUMP: Activation output (silu(gate) * up)
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          // After activation, result is stored in m_local_gate_output_ptr_
          dump_bf16_matrix(m_local_gate_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.intermediate_size,
                           "activation_output", tp_part_idx, expert_idx);
        }
      }
    }

    // Save intermediate AFTER activation for backward_down (Bug #17c fix)
    if (save_for_backward) {
      ForwardCache& cache = cache_stack_[cache_stack_top_ - 1];  // Get the cache we just pushed
      save_intermediate_to_cache(cache, activated_expert);
    }

    // Step 7: Quantize intermediate for down projection
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr, "fwd_down_quantize");

    // Step 8: Down GEMM
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          this->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr, "fwd_down_gemm", 1);

    // DUMP: Down base output (before LoRA)
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          dump_bf16_matrix(m_local_down_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.hidden_size,
                           "down_base_output", tp_part_idx, expert_idx);
        }
      }
    }

    // Step 8.5: Down LoRA
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      if constexpr (supports_standard_mat_mul_v<T>) {
        compute_lora_down_amx(qlen, activated_expert);  // AMX-optimized path
      } else {
        compute_lora_down(qlen, activated_expert);  // For-loop fallback for KGroup kernels
      }
    }

    // DUMP: Down output (after LoRA, before merge)
    if (is_dump_enabled() && down_lora_a_ != nullptr) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          dump_bf16_matrix(m_local_down_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.hidden_size,
                           "down_lora_output", tp_part_idx, expert_idx);
          // down_total_output is same as down_lora_output (lora is added in-place)
          dump_bf16_matrix(m_local_down_output_ptr_[expert_idx], m_local_num_[expert_idx], config_.hidden_size,
                           "down_total_output", tp_part_idx, expert_idx);
        }
      }
    }

    // Save down_output for grad_weights computation
    if (save_for_backward) {
      ForwardCache& cache = cache_stack_[cache_stack_top_ - 1];  // Get the cache we just pushed
      save_down_output_to_cache(cache, activated_expert);
    }

    // Step 9: Weighted merge
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
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
        nullptr, "fwd_merge");

    // DUMP: Final output (after weighted merge)
    // Note: Each TP partition outputs a partial result that gets summed later
    if (is_dump_enabled()) {
      dump_fp32_matrix((const float*)output, qlen, config_.hidden_size, "final_output", tp_part_idx);
    }
  }

  /**
   * @brief Backward pass for SFT.
   *
   * Computes gradients for LoRA weights using cached intermediate values.
   * When SkipLoRA template parameter is true, skips all LoRA computation
   * and only computes base weight contribution to grad_input.
   *
   * @param grad_output Gradient of loss w.r.t. output [qlen, hidden_size] (BF16)
   * @param grad_input Gradient of loss w.r.t. input [qlen, hidden_size] (BF16, output)
   * @param grad_gate_lora_a Gradient for gate LoRA A [expert_num, lora_rank, hidden_size] (BF16, ignored if
   * SkipLoRA=true)
   * @param grad_gate_lora_b Gradient for gate LoRA B [expert_num, intermediate_size, lora_rank] (ignored if
   * SkipLoRA=true)
   * @param grad_up_lora_a Gradient for up LoRA A (BF16, ignored if SkipLoRA=true)
   * @param grad_up_lora_b Gradient for up LoRA B (BF16, ignored if SkipLoRA=true)
   * @param grad_down_lora_a Gradient for down LoRA A (BF16, ignored if SkipLoRA=true)
   * @param grad_down_lora_b Gradient for down LoRA B (BF16, ignored if SkipLoRA=true)
   * @param grad_weights Gradient for routing weights [qlen, k] (FP32, output)
   */
  void backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b,
                void* grad_weights) {
    uint64_t _bwd_start_cycles = __rdtsc();
    BACKWARD_TIMER_START();

    // Pop cache from stack
    ForwardCache cache = pop_cache();
    if (!cache.valid) {
      throw std::runtime_error("No valid forward cache for backward");
    }

    int qlen = cache.qlen_cache;
    int k = cache.k_cache;
    int activated_expert = cache.activated_expert_cache;

    // auto print_lora_stats = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
    //   if (ptr == nullptr) {
    //     printf("KT MoE param stats (layer %d, %s): null\n", config_.layer_idx, name);
    //     return;
    //   }
    //   Bf16Stats stats = compute_bf16_stats(ptr, elems);
    //   printf("cpp KT MoE param stats (layer %d, %s): abs_mean=%.6e abs_max=%.6e norm=%.6e\n", config_.layer_idx,
    //   name,
    //          stats.abs_mean, stats.abs_max, stats.norm);
    // };

    // size_t gate_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
    // size_t gate_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
    // size_t up_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
    // size_t up_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
    // size_t down_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
    // size_t down_b_elems = static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_;

    // print_lora_stats("gate_lora_a", gate_lora_a_, gate_a_elems);
    // print_lora_stats("gate_lora_b", gate_lora_b_, gate_b_elems);
    // print_lora_stats("up_lora_a", up_lora_a_, up_a_elems);
    // print_lora_stats("up_lora_b", up_lora_b_, up_b_elems);
    // print_lora_stats("down_lora_a", down_lora_a_, down_a_elems);
    // print_lora_stats("down_lora_b", down_lora_b_, down_b_elems);

    // Restore routing information
    m_local_num_ = cache.m_local_num_cache;
    m_local_pos_ = cache.m_local_pos_cache;
    m_expert_id_map_ = cache.m_expert_id_map_cache;

    // Recompute pointer offsets
    size_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

    // // Compute total tokens for debug
    // size_t total_tokens = 0;
    // for (int i = 0; i < activated_expert; i++) {
    //   total_tokens += m_local_num_[m_expert_id_map_[i]];
    // }

    // printf("[BACKWARD DEBUG] qlen=%d, k=%d, activated_expert=%d, total_tokens=%zu\n", qlen, k, activated_expert,
    //        total_tokens);
    // printf("[BACKWARD DEBUG] grad_output norm: %f\n",
    //        compute_bf16_norm((const ggml_bf16_t*)grad_output, qlen * config_.hidden_size));

    // Step 1: Down projection backward
    if constexpr (supports_standard_mat_mul_v<T>) {
      backward_down_amx(cache, grad_output, grad_down_lora_a, grad_down_lora_b);
    } else {
      // backward_down(cache, grad_output, grad_down_lora_a, grad_down_lora_b);
    }
    BACKWARD_TIMER_CHECKPOINT("backward_down");

    // // DEBUG: Check m_local_input_ptr_ after backward_down (should be populated from cache)
    // {
    //   bool has_nan = false, has_large = false;
    //   float max_val = 0.0f;
    //   int activated_expert_dbg = cache.activated_expert_cache;
    //   for (int task_id = 0; task_id < activated_expert_dbg && !has_nan; task_id++) {
    //     int expert_idx = m_expert_id_map_[task_id];
    //     int m = m_local_num_[expert_idx];
    //     if (m == 0) continue;
    //     ggml_bf16_t* input_ptr = m_local_input_ptr_[expert_idx];
    //     for (int i = 0; i < m * config_.hidden_size && !has_nan; i++) {
    //       float v = GGML_BF16_TO_FP32(input_ptr[i]);
    //       if (std::isnan(v) || std::isinf(v)) has_nan = true;
    //       float av = std::abs(v);
    //       if (av > max_val) max_val = av;
    //       if (av > 1e10f) has_large = true;
    //     }
    //   }
    //   if (has_nan || has_large) {
    //     printf("[NaN DEBUG L%d] m_local_input AFTER backward_down: has_nan=%d has_large=%d max=%.6e\n",
    //            config_.layer_idx, has_nan, has_large, max_val);
    //   }
    // }

    // // DEBUG: Check for NaN after backward_down
    // {
    //   size_t grad_inter_size = qlen * k * config_.intermediate_size;
    //   bool has_nan = false;
    //   for (size_t i = 0; i < grad_inter_size && !has_nan; i++) {
    //     float val = GGML_BF16_TO_FP32(grad_intermediate_[i]);
    //     if (std::isnan(val) || std::isinf(val)) has_nan = true;
    //   }
    //   if (has_nan) {
    //     printf("[NaN DEBUG L%d] NaN detected in grad_intermediate after backward_down!\n", config_.layer_idx);
    //   }
    // }

    // Step 2: Activation backward
    backward_activation(cache);
    BACKWARD_TIMER_CHECKPOINT("backward_activation");

    // DEBUG: Check for NaN after backward_activation
    // {
    //   size_t grad_size = qlen * k * config_.intermediate_size;
    //   bool gate_nan = false, up_nan = false;
    //   for (size_t i = 0; i < grad_size && (!gate_nan || !up_nan); i++) {
    //     float g = GGML_BF16_TO_FP32(grad_gate_output_[i]);
    //     float u = GGML_BF16_TO_FP32(grad_up_output_[i]);
    //     if (std::isnan(g) || std::isinf(g)) gate_nan = true;
    //     if (std::isnan(u) || std::isinf(u)) up_nan = true;
    //   }
    //   if (gate_nan || up_nan) {
    //     printf("[NaN DEBUG L%d] NaN after backward_activation: grad_gate=%s, grad_up=%s\n",
    //            config_.layer_idx, gate_nan ? "NaN" : "OK", up_nan ? "NaN" : "OK");
    //   }
    // }

    // // DEBUG: Check m_local_input_ptr_ BEFORE backward_gate_up (after backward_activation)
    // {
    //   bool has_nan = false, has_large = false;
    //   float max_val = 0.0f;
    //   int activated_expert_dbg = cache.activated_expert_cache;
    //   for (int task_id = 0; task_id < activated_expert_dbg && !has_nan; task_id++) {
    //     int expert_idx = m_expert_id_map_[task_id];
    //     int m = m_local_num_[expert_idx];
    //     if (m == 0) continue;
    //     ggml_bf16_t* input_ptr = m_local_input_ptr_[expert_idx];
    //     for (int i = 0; i < m * config_.hidden_size && !has_nan; i++) {
    //       float v = GGML_BF16_TO_FP32(input_ptr[i]);
    //       if (std::isnan(v) || std::isinf(v)) has_nan = true;
    //       float av = std::abs(v);
    //       if (av > max_val) max_val = av;
    //       if (av > 1e10f) has_large = true;
    //     }
    //   }
    //   if (has_nan || has_large) {
    //     printf("[NaN DEBUG L%d] m_local_input BEFORE backward_gate_up: has_nan=%d has_large=%d max=%.6e\n",
    //            config_.layer_idx, has_nan, has_large, max_val);
    //   }
    // }

    // Step 3: Gate + Up projection backward
    if constexpr (supports_standard_mat_mul_v<T>) {
      backward_gate_up_amx(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
    } else {
      // backward_gate_up(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
    }
    BACKWARD_TIMER_CHECKPOINT("backward_gate_up");

    // DEBUG: Check for NaN after backward_gate_up
    // {
    //   size_t grad_input_size = qlen * config_.hidden_size;
    //   bool has_nan = false;
    //   const ggml_bf16_t* gi = (const ggml_bf16_t*)grad_input;
    //   for (size_t i = 0; i < grad_input_size && !has_nan; i++) {
    //     float val = GGML_BF16_TO_FP32(gi[i]);
    //     if (std::isnan(val) || std::isinf(val)) has_nan = true;
    //   }
    //   if (has_nan) {
    //     printf("[NaN DEBUG L%d] NaN detected in grad_input after backward_gate_up!\n", config_.layer_idx);
    //   }
    // }

    // Step 4: Compute grad_weights (gradient for routing weights)
    // grad_weights[token_idx, expert_pos] = dot(grad_output[token_idx], down_output[token, expert])
    if (grad_weights != nullptr) {
      auto pool = config_.pool->get_subpool(tp_part_idx);
      float* grad_w = (float*)grad_weights;
      const ggml_bf16_t* grad_out = (const ggml_bf16_t*)grad_output;

      // Compute offset mapping for down_output_cache (same layout as other caches)
      std::vector<size_t> expert_cache_offset(config_.expert_num, 0);
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = cache.m_expert_id_map_cache[i];
        expert_cache_offset[expert_idx] = offset;
        offset += cache.m_local_num_cache[expert_idx];
      }

      // Zero out grad_weights first
      memset(grad_w, 0, qlen * k * sizeof(float));

      // Compute grad_weights for each token-expert pair
      pool->do_work_stealing_job(
          qlen, nullptr,
          [&](int token_idx) {
            for (int j = 0; j < k; j++) {
              int64_t expert_idx = cache.expert_ids_cache[token_idx * k + j];
              if (expert_idx < config_.num_gpu_experts || expert_idx >= config_.expert_num) {
                continue;  // Skip GPU experts or invalid experts
              }

              int local_pos = cache.m_local_pos_cache[token_idx][j];
              size_t down_offset = expert_cache_offset[expert_idx] + local_pos;

              // dot(grad_output[token_idx], down_output_cache[down_offset])
              const ggml_bf16_t* grad_out_ptr = grad_out + token_idx * config_.hidden_size;
              const ggml_bf16_t* down_out_ptr = cache.down_output_cache + down_offset * config_.hidden_size;

              __m512 acc0 = _mm512_setzero_ps();
              __m512 acc1 = _mm512_setzero_ps();

              for (int h = 0; h + 32 <= config_.hidden_size; h += 32) {
                __m512 g0, g1, d0, d1;
                avx512_32xbf16_to_32xfp32((__m512i*)(grad_out_ptr + h), &g0, &g1);
                avx512_32xbf16_to_32xfp32((__m512i*)(down_out_ptr + h), &d0, &d1);
                acc0 = _mm512_fmadd_ps(g0, d0, acc0);
                acc1 = _mm512_fmadd_ps(g1, d1, acc1);
              }

              grad_w[token_idx * k + j] = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1);
            }
          },
          nullptr, "bwd_grad_weights");
    }
    BACKWARD_TIMER_CHECKPOINT("backward_grad_weights");
    BACKWARD_TIMER_END();
    // printf("[BACKWARD DEBUG] After backward_gate_up - grad_input norm: %f\n",
    //        compute_bf16_norm((const ggml_bf16_t*)grad_input, qlen * config_.hidden_size));

    // Mark cache as invalid
    cache.valid = false;
  }

  /**
   * @brief Get qlen from the top of the forward cache stack.
   *
   * Bug #22 fix: This is needed by TP_MOE_SFT::backward() to allocate
   * separate grad_input buffers for each NUMA node before calling backward.
   */
  int get_cache_qlen() const {
    if (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid) {
      return cache_stack_[cache_stack_top_ - 1].qlen_cache;
    }
    return 0;  // No valid cache
  }

  /**
   * @brief Get expert token distribution from last backward for load balancing analysis.
   * @return Vector of token counts per activated expert
   */
  const std::vector<int>& get_expert_token_distribution() const { return last_backward_expert_tokens_; }

  /**
   * @brief Update LoRA weight pointers (call when Python tensors are reallocated).
   */
  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    gate_lora_a_ = (ggml_bf16_t*)gate_lora_a;
    gate_lora_b_ = (ggml_bf16_t*)gate_lora_b;
    up_lora_a_ = (ggml_bf16_t*)up_lora_a;
    up_lora_b_ = (ggml_bf16_t*)up_lora_b;
    down_lora_a_ = (ggml_bf16_t*)down_lora_a;
    down_lora_b_ = (ggml_bf16_t*)down_lora_b;

    // auto print_lora_stats = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
    //   if (ptr == nullptr) {
    //     printf("KT MoE update stats (layer %d, %s): null\n", config_.layer_idx, name);
    //     return;
    //   }
    //   Bf16Stats stats = compute_bf16_stats(ptr, elems);
    //   printf("KT MoE update stats (layer %d, %s): abs_mean=%.6e abs_max=%.6e norm=%.6e\n", config_.layer_idx, name,
    //          stats.abs_mean, stats.abs_max, stats.norm);
    // };

    // size_t gate_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
    // size_t gate_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
    // size_t up_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
    // size_t up_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
    // size_t down_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
    // size_t down_b_elems = static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_;

    // print_lora_stats("gate_lora_a", gate_lora_a_, gate_a_elems);
    // print_lora_stats("gate_lora_b", gate_lora_b_, gate_b_elems);
    // print_lora_stats("up_lora_a", up_lora_a_, up_a_elems);
    // print_lora_stats("up_lora_b", up_lora_b_, up_b_elems);
    // print_lora_stats("down_lora_a", down_lora_a_, down_a_elems);
    // print_lora_stats("down_lora_b", down_lora_b_, down_b_elems);

    // Mark weights as needing re-conversion to BufferB format
    lora_weights_prepared_ = false;
  }

  /**
   * @brief Prepare LoRA weights for AMX GEMM.
   *
   * Converts BF16 LoRA weights from Python tensors to AMX BufferB format.
   * This includes padding to K_STEP multiples for AMX alignment.
   * Must be called before forward_sft() if lora_weights_prepared_ is false.
   */
  void prepare_lora_weights() {
    // Only prepare weights for kernels that support standard mat_mul
    if constexpr (!supports_standard_mat_mul_v<T>) {
      return;  // KGroup kernels use for-loop implementation
    }

    if (lora_weights_prepared_) {
      return;
    }
    if (gate_lora_a_ == nullptr) {
      return;  // No LoRA weights to prepare
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel conversion of all LoRA weights to BufferB format
    // 10 matrices per expert: gate/up (A, B, A^T, B^T) + down (A, B)
    pool->do_work_stealing_job(
        config_.expert_num * 10, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 10;
          int lora_type = task_id % 10;

          switch (lora_type) {
            case 0:  // gate_lora_a [lora_rank, hidden_size] -> [padded_lora_rank, hidden_size]
              convert_lora_a_to_buffer_b(gate_lora_a_, gate_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.hidden_size, padded_lora_rank_, config_.hidden_size);
              break;
            case 1:  // up_lora_a [lora_rank, hidden_size]
              convert_lora_a_to_buffer_b(up_lora_a_, up_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.hidden_size, padded_lora_rank_, config_.hidden_size);
              break;
            case 2:  // gate_lora_b [intermediate_size, lora_rank] -> [intermediate_size, padded_lora_rank]
              convert_lora_b_to_buffer_b(gate_lora_b_, gate_lora_b_bb_[expert_idx], expert_idx,
                                         config_.intermediate_size, lora_rank_, config_.intermediate_size,
                                         padded_lora_rank_);
              break;
            case 3:  // up_lora_b [intermediate_size, lora_rank]
              convert_lora_b_to_buffer_b(up_lora_b_, up_lora_b_bb_[expert_idx], expert_idx, config_.intermediate_size,
                                         lora_rank_, config_.intermediate_size, padded_lora_rank_);
              break;
            case 4:  // down_lora_a [lora_rank, intermediate_size] -> [padded_lora_rank, intermediate_size]
              convert_lora_a_to_buffer_b(down_lora_a_, down_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.intermediate_size, padded_lora_rank_, config_.intermediate_size);
              break;
            case 5:  // down_lora_b [hidden_size, lora_rank] -> [hidden_size, padded_lora_rank]
              convert_lora_b_to_buffer_b(down_lora_b_, down_lora_b_bb_[expert_idx], expert_idx, config_.hidden_size,
                                         lora_rank_, config_.hidden_size, padded_lora_rank_);
              break;
            case 6:  // gate_lora_a^T [hidden_size, lora_rank] -> [hidden_size, padded_lora_rank]
              convert_lora_a_transposed_to_buffer_b(gate_lora_a_, gate_lora_a_t_bb_[expert_idx], expert_idx, lora_rank_,
                                                    config_.hidden_size, config_.hidden_size, padded_lora_rank_);
              break;
            case 7:  // up_lora_a^T
              convert_lora_a_transposed_to_buffer_b(up_lora_a_, up_lora_a_t_bb_[expert_idx], expert_idx, lora_rank_,
                                                    config_.hidden_size, config_.hidden_size, padded_lora_rank_);
              break;
            case 8:  // gate_lora_b^T [lora_rank, intermediate_size] -> [padded_lora_rank, intermediate_size]
              convert_lora_b_transposed_to_buffer_b(gate_lora_b_, gate_lora_b_t_bb_[expert_idx], expert_idx,
                                                    config_.intermediate_size, lora_rank_, padded_lora_rank_,
                                                    config_.intermediate_size);
              break;
            case 9:  // up_lora_b^T
              convert_lora_b_transposed_to_buffer_b(up_lora_b_, up_lora_b_t_bb_[expert_idx], expert_idx,
                                                    config_.intermediate_size, lora_rank_, padded_lora_rank_,
                                                    config_.intermediate_size);
              break;
          }
        },
        nullptr, "fwd_lora_prep");

    lora_weights_prepared_ = true;
  }

  // Debug getter for LoRA pointer verification
  void* get_gate_lora_a() const { return (void*)gate_lora_a_; }

  /**
   * @brief Prepare backward weights for AMX GEMM.
   *
   * Converts base weights to transposed BufferB format for backward pass.
   * For backward GEMM, we need:
   * - gate_backward_bb_: gate_proj transposed [hidden_size, intermediate_size]
   * - up_backward_bb_: up_proj transposed [hidden_size, intermediate_size]
   * - down_backward_bb_: down_proj transposed [intermediate_size, hidden_size]
   *
   * Must be called before backward_down/backward_gate_up if backward_weights_prepared_ is false.
   */
  void prepare_backward_weights() {
    // Only prepare weights for kernels that support standard mat_mul
    if constexpr (!supports_standard_mat_mul_v<T>) {
      return;  // KGroup kernels use for-loop implementation
    }

    if (backward_weights_prepared_) return;
    if (config_.gate_proj == nullptr) return;  // No base weights to prepare

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel conversion of all base weights to transposed BufferB format
    // 3 matrices per expert: gate, up, down
    pool->do_work_stealing_job(
        config_.expert_num * 3, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 3;
          int weight_type = task_id % 3;

          switch (weight_type) {
            case 0: {  // gate_proj: [intermediate_size, hidden_size] -> transposed [hidden_size, intermediate_size]
              const ggml_bf16_t* gate_proj = (const ggml_bf16_t*)config_.gate_proj;
              size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

              // Create transposed matrix: [hidden_size, intermediate_size]
              // Original: [intermediate_size, hidden_size] stored row-major
              // Transposed: [hidden_size, intermediate_size] stored row-major
              std::vector<ggml_bf16_t> transposed(config_.hidden_size * config_.intermediate_size);
              for (int i = 0; i < config_.intermediate_size; i++) {
                for (int h = 0; h < config_.hidden_size; h++) {
                  transposed[h * config_.intermediate_size + i] =
                      gate_proj[expert_offset + i * config_.hidden_size + h];
                }
              }
              // FIX: Use the same nth as mat_mul will use, to ensure BufferB layout matches.
              // mat_mul uses nth = recommended_nth(hidden_size) which partitions by N_BLOCK.
              int nth = T::recommended_nth(config_.hidden_size);
              for (int ith = 0; ith < nth; ith++) {
                gate_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
              }
              break;
            }
            case 1: {  // up_proj: same as gate
              const ggml_bf16_t* up_proj = (const ggml_bf16_t*)config_.up_proj;
              size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

              std::vector<ggml_bf16_t> transposed(config_.hidden_size * config_.intermediate_size);
              for (int i = 0; i < config_.intermediate_size; i++) {
                for (int h = 0; h < config_.hidden_size; h++) {
                  transposed[h * config_.intermediate_size + i] = up_proj[expert_offset + i * config_.hidden_size + h];
                }
              }
              // FIX: Use the same nth as mat_mul will use, to ensure BufferB layout matches.
              // mat_mul uses nth = recommended_nth(hidden_size) which partitions by N_BLOCK.
              int nth = T::recommended_nth(config_.hidden_size);
              for (int ith = 0; ith < nth; ith++) {
                up_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
              }
              break;
            }
            case 2: {  // down_proj: [hidden_size, intermediate_size] -> transposed [intermediate_size, hidden_size]
              const ggml_bf16_t* down_proj = (const ggml_bf16_t*)config_.down_proj;
              size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;

              // Create transposed matrix: [intermediate_size, hidden_size]
              // Original: [hidden_size, intermediate_size] stored row-major
              // Transposed: [intermediate_size, hidden_size] stored row-major
              std::vector<ggml_bf16_t> transposed(config_.intermediate_size * config_.hidden_size);
              for (int h = 0; h < config_.hidden_size; h++) {
                for (int i = 0; i < config_.intermediate_size; i++) {
                  transposed[i * config_.hidden_size + h] =
                      down_proj[expert_offset + h * config_.intermediate_size + i];
                }
              }
              // FIX: Use the same nth as mat_mul will use, to ensure BufferB layout matches.
              // mat_mul uses nth = recommended_nth(intermediate_size) which partitions by N_BLOCK.
              int nth = T::recommended_nth(config_.intermediate_size);
              for (int ith = 0; ith < nth; ith++) {
                down_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
              }
              break;
            }
          }
        },
        nullptr, "bwd_prep");

    backward_weights_prepared_ = true;
  }

  /**
   * @brief Set base weight pointers for TP partitioning.
   * Used by TP_MOE_SFT::load_weights() to set partitioned weights before calling load_weights().
   */
  void set_base_weight_pointers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  /**
   * @brief Set physical to logical expert mapping.
   */
  void set_physical_to_logical_map(const void* map) { config_.physical_to_logical_map = const_cast<void*>(map); }

 private:
  /**
   * @brief Initialize all buffers in a single alloc() call.
   *
   * IMPORTANT: SharedMemBuffer is designed to let multiple callers share the same memory pool.
   * Each alloc() call assigns pointers starting from the SAME base address, which means:
   * - Multiple alloc() calls will OVERLAP in memory!
   * - This is intentional for temporary buffers that are not used simultaneously.
   * - But for SFT, cache and grad buffers ARE used simultaneously (cache written during forward,
   *   grad written during backward, both needed in backward_activation).
   *
   * Solution: Combine all buffer requests into a SINGLE alloc() call, so they get
   * consecutive, non-overlapping addresses.
   *
   * Bug #15 root cause: Three separate alloc() calls caused grad_intermediate_ to overlap
   * with cache_gate_output_pool_, and memset in backward_down() zeroed the cache data.
   */
  void init_all_buffers() {
    // =====================================================
    // Calculate padded_lora_rank for AMX alignment
    // AMX requires K dimension to be multiple of K_STEP (32 for BF16)
    // =====================================================
    constexpr int K_STEP = T::K_STEP;
    constexpr int N_STEP = T::N_STEP;
    constexpr int M_STEP = T::M_STEP;
    padded_lora_rank_ = ((lora_rank_ + K_STEP - 1) / K_STEP) * K_STEP;
    // Also need N dimension aligned for BufferB output dimension
    int padded_lora_rank_n = ((lora_rank_ + N_STEP - 1) / N_STEP) * N_STEP;
    // Use the larger of the two for consistency
    padded_lora_rank_ = std::max(padded_lora_rank_, padded_lora_rank_n);

    // Calculate all buffer sizes
    lora_intermediate_pool_bytes_ = sizeof(ggml_bf16_t) * config_.max_len * config_.num_experts_per_tok * lora_rank_;

    cache_slot_bytes_input_ = config_.max_len * config_.hidden_size * sizeof(ggml_bf16_t);
    cache_slot_bytes_intermediate_ =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    size_t grad_buffer_bytes =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    // =====================================================
    // Calculate LoRA AMX buffer sizes
    // Only for kernels that support standard mat_mul API
    // =====================================================
    // Max tokens per expert (with M_STEP alignment)
    // Bug-C Fix: Each expert processes at most max_len tokens (worst case: all tokens select this expert)
    // Previously used max_len * num_experts_per_tok which is incorrect and wastes 8x memory
    size_t max_m = ((config_.max_len + M_STEP - 1) / M_STEP) * M_STEP;

    // Variables for buffer sizes (used in init_lora_amx_buffers)
    size_t lora_a_gate_up_bb_size = 0;
    size_t lora_b_gate_up_bb_size = 0;
    size_t lora_a_gate_up_t_bb_size = 0;
    size_t lora_b_gate_up_t_bb_size = 0;
    size_t lora_a_down_bb_size = 0;
    size_t lora_b_down_bb_size = 0;
    size_t lora_intermediate_ba_size = 0;
    size_t lora_intermediate_bc_size = 0;
    size_t lora_gate_up_out_bc_size = 0;
    size_t lora_down_out_bc_size = 0;
    size_t grad_output_ba_size = 0;
    size_t grad_intermediate_bc_size = 0;
    size_t grad_gate_up_bc_size = 0;
    size_t gate_up_backward_bb_size = 0;
    size_t down_backward_bb_size = 0;

    if constexpr (supports_standard_mat_mul_v<T>) {
      // BufferB sizes for LoRA weights (need to be aligned)
      // gate/up lora_A: [padded_lora_rank, hidden_size] per expert
      lora_a_gate_up_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.hidden_size);
      // gate/up lora_B: [intermediate_size, padded_lora_rank] per expert
      lora_b_gate_up_bb_size = T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_);
      // Transposed weights for backward LoRA GEMM
      // gate/up lora_A^T: [hidden_size, padded_lora_rank] per expert
      lora_a_gate_up_t_bb_size = T::BufferB::required_size(config_.hidden_size, padded_lora_rank_);
      // gate/up lora_B^T: [padded_lora_rank, intermediate_size] per expert
      lora_b_gate_up_t_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size);
      // down lora_A: [padded_lora_rank, intermediate_size] per expert
      lora_a_down_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size);
      // down lora_B: [hidden_size, padded_lora_rank] per expert
      lora_b_down_bb_size = T::BufferB::required_size(config_.hidden_size, padded_lora_rank_);

      // Total BufferB pool size for all experts (10 matrices per expert)
      lora_bb_pool_bytes_ = config_.expert_num * (lora_a_gate_up_bb_size * 2 +    // gate_a, up_a
                                                  lora_b_gate_up_bb_size * 2 +    // gate_b, up_b
                                                  lora_a_gate_up_t_bb_size * 2 +  // gate_a^T, up_a^T
                                                  lora_b_gate_up_t_bb_size * 2 +  // gate_b^T, up_b^T
                                                  lora_a_down_bb_size +           // down_a
                                                  lora_b_down_bb_size);           // down_b

      // Bug-C Fix Step 2: Use shared buffer pool instead of per-expert allocation
      // Max total tokens across all activated experts per forward pass
      // (each of max_len tokens selects num_experts_per_tok experts)
      size_t max_total_tokens = ((config_.max_len * config_.num_experts_per_tok + M_STEP - 1) / M_STEP) * M_STEP;

      // BufferA for LoRA intermediate: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_ba_size = T::BufferA::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_ba_pool_bytes_ = T::BufferA::required_size(max_total_tokens, padded_lora_rank_) * 2;  // gate + up

      // BufferC for LoRA step 1 output: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bc_size = T::BufferC::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_bc_inter_pool_bytes_ = T::BufferC::required_size(max_total_tokens, padded_lora_rank_) * 2;  // gate + up

      // BufferC for LoRA step 2 output (gate, up, down): shared pool for all activated experts
      lora_gate_up_out_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      lora_down_out_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);           // per-expert size
      lora_bc_out_pool_bytes_ = T::BufferC::required_size(max_total_tokens, config_.intermediate_size) * 2 +
                                T::BufferC::required_size(max_total_tokens, config_.hidden_size);

      // BF16 intermediate buffer for step 1 -> step 2 conversion
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bf16_pool_bytes_ = max_total_tokens * padded_lora_rank_ * sizeof(ggml_bf16_t) * 2;  // gate + up

      // =====================================================
      // Calculate Backward pass AMX buffer sizes
      // =====================================================
      // BufferA for scattered grad_output: shared pool for all activated experts
      grad_output_ba_size = T::BufferA::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_ba_pool_bytes_ = T::BufferA::required_size(max_total_tokens, config_.hidden_size);

      // BufferC for backward GEMM outputs: shared pool for all activated experts
      // grad_intermediate: [max_total_tokens, intermediate_size]
      grad_intermediate_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      // grad_gate_up: [max_total_tokens, hidden_size]
      grad_gate_up_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_bc_pool_bytes_ = T::BufferC::required_size(max_total_tokens, config_.intermediate_size) +
                                T::BufferC::required_size(max_total_tokens, config_.hidden_size);

      // BF16 buffer for scattered grad_output
      grad_output_bf16_pool_bytes_ = max_total_tokens * config_.hidden_size * sizeof(ggml_bf16_t);

      // =====================================================
      // Calculate Backward pass BufferB sizes (transposed base weights)
      // =====================================================
      // For backward GEMM, we need transposed versions of base weights:
      // - gate/up backward: BufferB[hidden_size, intermediate_size] per expert
      // - down backward: BufferB[intermediate_size, hidden_size] per expert
      gate_up_backward_bb_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      down_backward_bb_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
      backward_bb_pool_bytes_ = config_.expert_num * (gate_up_backward_bb_size * 2 + down_backward_bb_size);
    } else {
      // For unsupported kernels (KGroup kernels), set all AMX buffer sizes to 0
      // These kernels will use the original for-loop implementation
      lora_bb_pool_bytes_ = 0;
      lora_ba_pool_bytes_ = 0;
      lora_bc_inter_pool_bytes_ = 0;
      lora_bc_out_pool_bytes_ = 0;
      lora_intermediate_bf16_pool_bytes_ = 0;
      backward_ba_pool_bytes_ = 0;
      backward_bc_pool_bytes_ = 0;
      grad_output_bf16_pool_bytes_ = 0;
      backward_bb_pool_bytes_ = 0;
    }

    // ★ Bug #18 fix: Cache buffers use aligned_alloc instead of shared_mem_buffer_numa ★
    // The base class AMX_MOE_BASE::init() also calls shared_mem_buffer_numa.alloc(), and
    // SharedMemBuffer is designed to let multiple callers share the same memory pool.
    // This causes cache buffers to overlap with base class buffers like m_local_gate_output_,
    // which corrupts the cache when apply_activation() writes to m_local_gate_output_.
    // Solution: Use aligned_alloc for cache pools so they have dedicated memory.

    // Cache buffers (5 pools × max_cache_depth) - use aligned_alloc for independent memory
    size_t cache_input_bytes = cache_slot_bytes_input_ * max_cache_depth_;
    size_t cache_intermediate_bytes = cache_slot_bytes_intermediate_ * max_cache_depth_;

    if (cache_input_bytes > 0) {
      cache_input_pool_ = aligned_alloc(64, cache_input_bytes);
      memset(cache_input_pool_, 0, cache_input_bytes);
    }
    if (cache_intermediate_bytes > 0) {
      cache_gate_output_pool_ = aligned_alloc(64, cache_intermediate_bytes);
      cache_up_output_pool_ = aligned_alloc(64, cache_intermediate_bytes);
      cache_intermediate_pool_ = aligned_alloc(64, cache_intermediate_bytes);

      memset(cache_gate_output_pool_, 0, cache_intermediate_bytes);
      memset(cache_up_output_pool_, 0, cache_intermediate_bytes);
      memset(cache_intermediate_pool_, 0, cache_intermediate_bytes);
    }
    if (cache_input_bytes > 0) {
      cache_down_output_pool_ = aligned_alloc(64, cache_input_bytes);  // [tokens, hidden_size]
      memset(cache_down_output_pool_, 0, cache_input_bytes);
    }

    // Gradient buffers (3 pools) - Bug #18c fix: also use aligned_alloc to avoid overlap
    // with base class buffers that use shared_mem_buffer_numa
    if (grad_buffer_bytes > 0) {
      grad_intermediate_pool_ = aligned_alloc(64, grad_buffer_bytes);
      grad_gate_output_pool_ = aligned_alloc(64, grad_buffer_bytes);
      grad_up_output_pool_ = aligned_alloc(64, grad_buffer_bytes);
      // Note: These are zeroed in backward_down() before use, so no need to memset here
    }

    MemoryRequest mem_requests;

    // LoRA buffers (legacy, kept for compatibility)
    mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

    // LoRA AMX buffers - use aligned_alloc for independent allocation
    // (shared_mem_buffer would cause memory sharing issues)
    if (lora_bb_pool_bytes_ > 0) {
      lora_bb_pool_ = aligned_alloc(64, lora_bb_pool_bytes_);
      memset(lora_bb_pool_, 0, lora_bb_pool_bytes_);
    }
    if (lora_ba_pool_bytes_ > 0) {
      lora_ba_pool_ = aligned_alloc(64, lora_ba_pool_bytes_);
      memset(lora_ba_pool_, 0, lora_ba_pool_bytes_);
    }
    if (lora_bc_inter_pool_bytes_ > 0) {
      lora_bc_inter_pool_ = aligned_alloc(64, lora_bc_inter_pool_bytes_);
      memset(lora_bc_inter_pool_, 0, lora_bc_inter_pool_bytes_);
    }
    if (lora_bc_out_pool_bytes_ > 0) {
      lora_bc_out_pool_ = aligned_alloc(64, lora_bc_out_pool_bytes_);
      memset(lora_bc_out_pool_, 0, lora_bc_out_pool_bytes_);
    }
    if (lora_intermediate_bf16_pool_bytes_ > 0) {
      lora_intermediate_bf16_pool_ = aligned_alloc(64, lora_intermediate_bf16_pool_bytes_);
      memset(lora_intermediate_bf16_pool_, 0, lora_intermediate_bf16_pool_bytes_);
    }

    // ★ Bug #18d fix: Backward pass buffers use aligned_alloc instead of shared_mem_buffer_numa ★
    // These buffers are used during backward_down_amx and backward_gate_up_amx, and need to
    // coexist with base class buffers (m_local_input_, etc.). SharedMemBuffer causes overlap.
    if (backward_ba_pool_bytes_ > 0) {
      backward_ba_pool_ = aligned_alloc(64, backward_ba_pool_bytes_);
      memset(backward_ba_pool_, 0, backward_ba_pool_bytes_);
    }
    if (backward_bc_pool_bytes_ > 0) {
      backward_bc_pool_ = aligned_alloc(64, backward_bc_pool_bytes_);
      memset(backward_bc_pool_, 0, backward_bc_pool_bytes_);
    }
    if (grad_output_bf16_pool_bytes_ > 0) {
      grad_output_bf16_pool_ = aligned_alloc(64, grad_output_bf16_pool_bytes_);
      memset(grad_output_bf16_pool_, 0, grad_output_bf16_pool_bytes_);
    }
    if (backward_bb_pool_bytes_ > 0) {
      backward_bb_pool_ = aligned_alloc(64, backward_bb_pool_bytes_);
      memset(backward_bb_pool_, 0, backward_bb_pool_bytes_);
    }

    // Single allocation for remaining buffers (only lora_intermediate_pool_ uses SharedMemBuffer now)
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    // Initialize LoRA and gradient pointers
    lora_intermediate_ = (ggml_bf16_t*)lora_intermediate_pool_;
    grad_intermediate_ = (ggml_bf16_t*)grad_intermediate_pool_;
    grad_gate_output_ = (ggml_bf16_t*)grad_gate_output_pool_;
    grad_up_output_ = (ggml_bf16_t*)grad_up_output_pool_;

    // Initialize cache stack
    cache_stack_.resize(max_cache_depth_);
    for (int i = 0; i < max_cache_depth_; i++) {
      cache_stack_[i].input_cache = (ggml_bf16_t*)cache_input_pool_ + i * config_.max_len * config_.hidden_size;
      cache_stack_[i].gate_output_cache = (ggml_bf16_t*)cache_gate_output_pool_ +
                                          i * config_.max_len * config_.num_experts_per_tok * config_.intermediate_size;
      cache_stack_[i].up_output_cache = (ggml_bf16_t*)cache_up_output_pool_ +
                                        i * config_.max_len * config_.num_experts_per_tok * config_.intermediate_size;
      cache_stack_[i].intermediate_cache = (ggml_bf16_t*)cache_intermediate_pool_ + i * config_.max_len *
                                                                                        config_.num_experts_per_tok *
                                                                                        config_.intermediate_size;
      cache_stack_[i].down_output_cache = (ggml_bf16_t*)cache_down_output_pool_ +
                                          i * config_.max_len * config_.num_experts_per_tok * config_.hidden_size;
      cache_stack_[i].m_local_num_cache.resize(config_.expert_num);
      cache_stack_[i].m_local_pos_cache.resize(config_.max_len);
      for (int j = 0; j < config_.max_len; j++) {
        cache_stack_[i].m_local_pos_cache[j].resize(config_.num_experts_per_tok);
      }
      cache_stack_[i].m_expert_id_map_cache.resize(config_.expert_num);
    }

    // =====================================================
    // Initialize LoRA AMX buffer objects (only for supported kernels)
    // =====================================================
    if constexpr (supports_standard_mat_mul_v<T>) {
      init_lora_amx_buffers(max_m, lora_a_gate_up_bb_size, lora_b_gate_up_bb_size, lora_a_gate_up_t_bb_size,
                            lora_b_gate_up_t_bb_size, lora_a_down_bb_size, lora_b_down_bb_size,
                            lora_intermediate_ba_size, lora_intermediate_bc_size, lora_gate_up_out_bc_size,
                            lora_down_out_bc_size, grad_output_ba_size, grad_intermediate_bc_size, grad_gate_up_bc_size,
                            gate_up_backward_bb_size, down_backward_bb_size);
    }

    // Bug-C Debug: Memory allocation summary (commented out after verification)
    // Uncomment to see memory allocation details
    /*
    printf("\n========== Memory Allocation Summary ==========\n");
    printf("Config: expert_num=%d, hidden_size=%d, intermediate_size=%d\n",
           config_.expert_num, config_.hidden_size, config_.intermediate_size);
    printf("Config: max_len=%d, num_experts_per_tok=%d, lora_rank=%d, padded_lora_rank=%d\n",
           config_.max_len, config_.num_experts_per_tok, lora_rank_, padded_lora_rank_);
    printf("Calculated max_m=%zu, max_total_tokens=%zu\n",
           max_m, (size_t)config_.max_len * config_.num_experts_per_tok);
    printf("\n--- LoRA Buffers (aligned_alloc) ---\n");
    printf("  lora_bb_pool_bytes_:              %12zu bytes (%6.2f MB)\n", lora_bb_pool_bytes_, lora_bb_pool_bytes_ /
    1024.0 / 1024.0); printf("  lora_ba_pool_bytes_:              %12zu bytes (%6.2f MB)\n", lora_ba_pool_bytes_,
    lora_ba_pool_bytes_ / 1024.0 / 1024.0); printf("  lora_bc_inter_pool_bytes_:        %12zu bytes (%6.2f MB)\n",
    lora_bc_inter_pool_bytes_, lora_bc_inter_pool_bytes_ / 1024.0 / 1024.0); printf("  lora_bc_out_pool_bytes_: %12zu
    bytes (%6.2f GB)\n", lora_bc_out_pool_bytes_, lora_bc_out_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("
    lora_intermediate_bf16_pool_bytes_:%12zu bytes (%6.2f MB)\n", lora_intermediate_bf16_pool_bytes_,
    lora_intermediate_bf16_pool_bytes_ / 1024.0 / 1024.0); printf("\n--- Backward Buffers (shared_mem_buffer) ---\n");
    printf("  backward_ba_pool_bytes_:          %12zu bytes (%6.2f GB)\n", backward_ba_pool_bytes_,
    backward_ba_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("  backward_bc_pool_bytes_:          %12zu bytes (%6.2f
    GB)\n", backward_bc_pool_bytes_, backward_bc_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("
    grad_output_bf16_pool_bytes_:     %12zu bytes (%6.2f GB)\n", grad_output_bf16_pool_bytes_,
    grad_output_bf16_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("  backward_bb_pool_bytes_:          %12zu bytes
    (%6.2f GB)\n", backward_bb_pool_bytes_, backward_bb_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("\n--- Other
    Buffers (shared_mem_buffer) ---\n"); printf("  lora_intermediate_pool_bytes_:    %12zu bytes (%6.2f GB)\n",
    lora_intermediate_pool_bytes_, lora_intermediate_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf(" grad_buffer_bytes
    (×3):           %12zu bytes (%6.2f GB)\n", grad_buffer_bytes * 3, grad_buffer_bytes * 3 / 1024.0 / 1024.0 / 1024.0);
    size_t cache_total = (cache_slot_bytes_input_ + cache_slot_bytes_intermediate_ * 3) * max_cache_depth_;
    printf("  cache_total (depth=%d):           %12zu bytes (%6.2f GB)\n", max_cache_depth_, cache_total, cache_total /
    1024.0 / 1024.0 / 1024.0); size_t total_aligned = lora_bb_pool_bytes_ + lora_ba_pool_bytes_ +
    lora_bc_inter_pool_bytes_ + lora_bc_out_pool_bytes_ + lora_intermediate_bf16_pool_bytes_; size_t total_shared =
    backward_ba_pool_bytes_ + backward_bc_pool_bytes_ + grad_output_bf16_pool_bytes_ + backward_bb_pool_bytes_ +
    lora_intermediate_pool_bytes_ + grad_buffer_bytes * 3 + cache_total; printf("\n--- Summary ---\n"); printf("  Total
    aligned_alloc:              %12zu bytes (%6.2f GB)\n", total_aligned, total_aligned / 1024.0 / 1024.0 / 1024.0);
    printf("  Total shared_mem_buffer:          %12zu bytes (%6.2f GB)\n", total_shared, total_shared / 1024.0 / 1024.0
    / 1024.0); printf("  GRAND TOTAL:                      %12zu bytes (%6.2f GB)\n", total_aligned + total_shared,
    (total_aligned + total_shared) / 1024.0 / 1024.0 / 1024.0);
    printf("===============================================\n\n");
    */
  }

  /**
   * @brief Initialize LoRA AMX buffer objects (including backward pass buffers).
   */
  void init_lora_amx_buffers(size_t max_m, size_t lora_a_gate_up_bb_size, size_t lora_b_gate_up_bb_size,
                             size_t lora_a_gate_up_t_bb_size, size_t lora_b_gate_up_t_bb_size,
                             size_t lora_a_down_bb_size, size_t lora_b_down_bb_size, size_t lora_intermediate_ba_size,
                             size_t lora_intermediate_bc_size, size_t lora_gate_up_out_bc_size,
                             size_t lora_down_out_bc_size, size_t grad_output_ba_size, size_t grad_intermediate_bc_size,
                             size_t grad_gate_up_bc_size, size_t gate_up_backward_bb_size,
                             size_t down_backward_bb_size) {
    // Resize vectors - forward pass
    gate_lora_a_bb_.resize(config_.expert_num);
    up_lora_a_bb_.resize(config_.expert_num);
    down_lora_a_bb_.resize(config_.expert_num);
    gate_lora_b_bb_.resize(config_.expert_num);
    up_lora_b_bb_.resize(config_.expert_num);
    down_lora_b_bb_.resize(config_.expert_num);
    gate_lora_a_t_bb_.resize(config_.expert_num);
    up_lora_a_t_bb_.resize(config_.expert_num);
    gate_lora_b_t_bb_.resize(config_.expert_num);
    up_lora_b_t_bb_.resize(config_.expert_num);
    // Separate buffers for gate and up to avoid race condition
    lora_gate_intermediate_ba_.resize(config_.expert_num);
    lora_up_intermediate_ba_.resize(config_.expert_num);
    lora_gate_intermediate_bc_.resize(config_.expert_num);
    lora_up_intermediate_bc_.resize(config_.expert_num);
    lora_gate_out_bc_.resize(config_.expert_num);
    lora_up_out_bc_.resize(config_.expert_num);
    lora_down_out_bc_.resize(config_.expert_num);
    lora_gate_intermediate_ptr_.resize(config_.expert_num);
    lora_up_intermediate_ptr_.resize(config_.expert_num);

    // Resize vectors - backward pass
    grad_output_ba_.resize(config_.expert_num);
    grad_intermediate_bc_.resize(config_.expert_num);
    grad_gate_up_bc_.resize(config_.expert_num);
    grad_output_bf16_ptr_.resize(config_.expert_num);

    // Resize vectors - backward BufferB (transposed base weights)
    gate_backward_bb_.resize(config_.expert_num);
    up_backward_bb_.resize(config_.expert_num);
    down_backward_bb_.resize(config_.expert_num);

    // Calculate offsets and create buffer objects
    // Bug-C Fix Step 2: BufferA/BufferC use shared pools, data will be assigned in forward/backward
    char* bb_ptr = (char*)lora_bb_pool_;

    for (int i = 0; i < config_.expert_num; i++) {
      // BufferB for LoRA weights (still per-expert, as weights are different for each expert)
      gate_lora_a_bb_[i] = std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_bb_size;

      up_lora_a_bb_[i] = std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_bb_size;

      gate_lora_b_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_bb_size;

      up_lora_b_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_bb_size;

      gate_lora_a_t_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_t_bb_size;

      up_lora_a_t_bb_[i] = std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_t_bb_size;

      gate_lora_b_t_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_t_bb_size;

      up_lora_b_t_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_t_bb_size;

      down_lora_a_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_a_down_bb_size;

      down_lora_b_bb_[i] = std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_down_bb_size;

      // BufferA for LoRA intermediate: create with nullptr, will set_data in forward
      lora_gate_intermediate_ba_[i] = std::make_shared<typename T::BufferA>(max_m, padded_lora_rank_, nullptr);
      lora_up_intermediate_ba_[i] = std::make_shared<typename T::BufferA>(max_m, padded_lora_rank_, nullptr);

      // BufferC for LoRA step 1 output: create with nullptr, will set_data in forward
      lora_gate_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, padded_lora_rank_, nullptr);
      lora_up_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, padded_lora_rank_, nullptr);

      // BufferC for LoRA step 2 output: create with nullptr, will set_data in forward
      lora_gate_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);
      lora_up_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);
      lora_down_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.hidden_size, nullptr);

      // BF16 intermediate pointer: will be assigned in forward
      lora_gate_intermediate_ptr_[i] = nullptr;
      lora_up_intermediate_ptr_[i] = nullptr;
    }

    // =====================================================
    // Initialize backward pass buffer objects
    // Bug-C Fix Step 2: Use shared pools, data will be assigned in backward
    // =====================================================
    for (int i = 0; i < config_.expert_num; i++) {
      // BufferA for grad_output: create with nullptr, will set_data in backward
      grad_output_ba_[i] = std::make_shared<typename T::BufferA>(max_m, config_.hidden_size, nullptr);

      // BufferC for grad_intermediate: create with nullptr, will set_data in backward
      grad_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);

      // BufferC for grad_gate_up: create with nullptr, will set_data in backward
      grad_gate_up_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.hidden_size, nullptr);

      // BF16 pointer: will be assigned in backward
      grad_output_bf16_ptr_[i] = nullptr;
    }

    // =====================================================
    // Initialize backward BufferB objects (transposed base weights)
    // =====================================================
    char* backward_bb_ptr = (char*)backward_bb_pool_;
    for (int i = 0; i < config_.expert_num; i++) {
      // BufferB for gate backward: [hidden_size, intermediate_size]
      gate_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      // BufferB for up backward: [hidden_size, intermediate_size]
      up_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      // BufferB for down backward: [intermediate_size, hidden_size]
      down_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, (void*)backward_bb_ptr);
      backward_bb_ptr += down_backward_bb_size;
    }

    lora_weights_prepared_ = false;
    backward_weights_prepared_ = false;
  }

  /**
   * @brief Convert LoRA A matrix to BufferB format with padding.
   *
   * LoRA A shape: [expert_num, lora_rank, k_dim]
   * Padded shape: [expert_num, padded_lora_rank, k_dim]
   * BufferB expects: [n_dim, k_dim] where n_dim = padded_lora_rank
   *
   * Padding rows with zeros for lora_rank < padded_lora_rank.
   */
  void convert_lora_a_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb, int expert_idx,
                                  int src_n, int src_k, int dst_n, int dst_k) {
    // Create temporary padded matrix
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));

    // Copy source data (with potential padding)
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;
    for (int r = 0; r < src_n && r < dst_n; r++) {
      for (int c = 0; c < src_k && c < dst_k; c++) {
        padded[r * dst_k + c] = expert_src[r * src_k + c];
      }
    }

    // Convert to BufferB format using from_mat
    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Convert LoRA B matrix to BufferB format with padding.
   *
   * LoRA B shape: [expert_num, output_dim, lora_rank]
   * Padded shape: [expert_num, output_dim, padded_lora_rank]
   * BufferB expects: [n_dim, k_dim] where n_dim = output_dim, k_dim = padded_lora_rank
   *
   * Padding columns with zeros for lora_rank < padded_lora_rank.
   */
  void convert_lora_b_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb, int expert_idx,
                                  int src_n, int src_k, int dst_n, int dst_k) {
    // Create temporary padded matrix
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));

    // Copy source data (with potential padding on K dimension)
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int r = 0; r < src_n && r < dst_n; r++) {
      for (int c = 0; c < src_k && c < dst_k; c++) {
        padded[r * dst_k + c] = expert_src[r * src_k + c];
      }
    }

    // Convert to BufferB format using from_mat
    // NOTE: from_mat with (ith, nth) only processes one N_BLOCK chunk.
    // For dst_n > N_BLOCK, we need to loop over all N_BLOCKs.
    int num_n_blocks = (dst_n + T::N_BLOCK - 1) / T::N_BLOCK;
    for (int ith = 0; ith < num_n_blocks; ith++) {
      dst_bb->from_mat(padded.data(), ith, num_n_blocks);
    }
  }

  /**
   * @brief Convert LoRA A^T matrix to BufferB format with padding on rank dimension.
   *
   * Input shape: [expert_num, lora_rank, hidden_size]
   * Output shape: [expert_num, hidden_size, padded_lora_rank]
   */
  void convert_lora_a_transposed_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb,
                                             int expert_idx, int src_n, int src_k, int dst_n, int dst_k) {
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int h = 0; h < src_k && h < dst_n; h++) {
      for (int r = 0; r < src_n && r < dst_k; r++) {
        padded[h * dst_k + r] = expert_src[r * src_k + h];
      }
    }

    // NOTE: from_mat with (ith, nth) only processes one N_BLOCK chunk.
    // For dst_n > N_BLOCK (hidden_size is typically 7168), we need to loop over all N_BLOCKs.
    int num_n_blocks = (dst_n + T::N_BLOCK - 1) / T::N_BLOCK;
    for (int ith = 0; ith < num_n_blocks; ith++) {
      dst_bb->from_mat(padded.data(), ith, num_n_blocks);
    }
  }

  /**
   * @brief Convert LoRA B^T matrix to BufferB format with padding on rank dimension.
   *
   * Input shape: [expert_num, intermediate_size, lora_rank]
   * Output shape: [expert_num, padded_lora_rank, intermediate_size]
   */
  void convert_lora_b_transposed_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb,
                                             int expert_idx, int src_n, int src_k, int dst_n, int dst_k) {
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int r = 0; r < src_k && r < dst_n; r++) {
      for (int i = 0; i < src_n && i < dst_k; i++) {
        padded[r * dst_k + i] = expert_src[i * src_k + r];
      }
    }

    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Compute LoRA for gate and up projections using AMX GEMM.
   *
   * gate_lora_out = (input @ gate_lora_A^T) @ gate_lora_B^T * scaling
   * gate_output += gate_lora_out
   * (similar for up)
   *
   * This is the AMX-optimized version replacing the naive for-loop implementation.
   */
  void compute_lora_gate_up_amx(int qlen, int activated_expert) {
    if (gate_lora_a_ == nullptr || gate_lora_b_ == nullptr) {
      return;
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Ensure LoRA weights are prepared
    prepare_lora_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate LoRA buffers from shared pool
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Pool pointers for forward LoRA buffers
    char* lora_ba_ptr = (char*)lora_ba_pool_;
    char* lora_bc_inter_ptr = (char*)lora_bc_inter_pool_;
    char* lora_bc_out_ptr = (char*)lora_bc_out_pool_;
    char* bf16_inter_ptr = (char*)lora_intermediate_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferA for intermediate (gate and up)
      lora_gate_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      lora_up_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // Allocate BufferC for intermediate (gate and up)
      lora_gate_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      lora_up_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // Allocate BufferC for output (gate, up, down - but down is done in compute_lora_down_amx)
      lora_gate_out_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_out_bc_[expert_idx]->set_data(lora_bc_out_ptr);
      lora_bc_out_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      lora_up_out_bc_[expert_idx]->max_m = local_max_m;
      lora_up_out_bc_[expert_idx]->set_data(lora_bc_out_ptr);
      lora_bc_out_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      // Allocate BF16 intermediate buffer (gate and up)
      lora_gate_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));

      lora_up_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));
    }

    // =====================================================
    // Step 1: input @ lora_A^T -> lora_intermediate
    // Uses gate_up_ba_ (already quantized input)
    // Gate and Up use SEPARATE intermediate buffers to avoid race condition
    // =====================================================
    int nth = T::recommended_nth(padded_lora_rank_);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = gate_up_ba_[expert_idx];  // Reuse quantized input
          auto& bb = do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx];
          // Use separate BufferC for gate and up to avoid race condition
          auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];

          // GEMM: [m, hidden_size] @ [padded_lora_rank, hidden_size]^T -> [m, padded_lora_rank]
          amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, ba, bb, bc, ith, nth);

          // Convert BufferC to BF16 for step 2 input (separate for gate and up)
          ggml_bf16_t* inter_ptr =
              do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
          bc->to_mat(m, inter_ptr, ith, nth);
        },
        nullptr, "fwd_lora_gu_a");

    // DUMP: LoRA intermediate (input @ lora_A^T) for gate and up
    // Note: Use padded_lora_rank_ as stride since to_mat writes with this stride
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          dump_bf16_matrix(lora_gate_intermediate_ptr_[expert_idx], m, padded_lora_rank_, "gate_lora_intermediate",
                           tp_part_idx, expert_idx);
          dump_bf16_matrix(lora_up_intermediate_ptr_[expert_idx], m, padded_lora_rank_, "up_lora_intermediate",
                           tp_part_idx, expert_idx);
        }
      }
    }

    // =====================================================
    // Step 2: Quantize lora_intermediate to BufferA
    // Need to quantize BOTH gate and up intermediates separately
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,  // 2x tasks for gate and up
        [this](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          // Use separate BufferA and BF16 pointer for gate and up
          auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
          ggml_bf16_t* ptr = do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
          ba->from_mat(m, ptr, 0, 1);
        },
        nullptr, "fwd_lora_gu_quant");

    // =====================================================
    // Step 3a: lora_intermediate @ lora_B^T -> lora_output (GEMM only)
    // =====================================================
    nth = T::recommended_nth(config_.intermediate_size);
    if (is_dump_enabled()) {
      printf("[DEBUG] Step 3a GEMM: nth=%d, activated_expert=%d, total_tasks=%d\n", nth, activated_expert,
             nth * activated_expert * 2);
    }
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          // Use separate BufferA for gate and up
          auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
          auto& bb = do_up ? up_lora_b_bb_[expert_idx] : gate_lora_b_bb_[expert_idx];
          auto& bc = do_up ? lora_up_out_bc_[expert_idx] : lora_gate_out_bc_[expert_idx];

          if (is_dump_enabled() && !do_up && expert_idx == 0) {
            printf("[DEBUG] GEMM task START: expert=%d, ith=%d, nth=%d, m=%d, n=%d, k=%d\n", expert_idx, ith, nth, m,
                   config_.intermediate_size, padded_lora_rank_);
          }

          // GEMM: [m, padded_lora_rank] @ [intermediate_size, padded_lora_rank]^T -> [m, intermediate_size]
          amx::mat_mul(m, config_.intermediate_size, padded_lora_rank_, ba, bb, bc, ith, nth);

          if (is_dump_enabled() && !do_up && expert_idx == 0) {
            // Check raw BufferC data immediately after this GEMM task
            float* raw_c = bc->get_submat(m, config_.intermediate_size, 0, ith * T::N_BLOCK);
            printf("[DEBUG] GEMM task DONE: expert=%d, ith=%d, raw_c[0]=%.6f, raw_c[1]=%.6f\n", expert_idx, ith,
                   raw_c[0], raw_c[1]);
          }
        },
        nullptr, "fwd_lora_gu_gemm");

    // DUMP: Pure gate/up LoRA GEMM output (before scaling and add)
    // Note: to_mat with (ith, nth) only reads one N_BLOCK chunk, so we need to loop
    if (is_dump_enabled()) {
      int dump_nth = T::recommended_nth(config_.intermediate_size);
      printf("[DEBUG] gate/up GEMM dump: intermediate_size=%d, N_BLOCK=%d, dump_nth=%d\n", config_.intermediate_size,
             T::N_BLOCK, dump_nth);
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          printf("[DEBUG] expert=%d, m=%d, BufferC.n=%d\n", expert_idx, m, lora_gate_out_bc_[expert_idx]->n);
          // Convert BufferC to FP32 and dump for gate
          std::vector<float> gate_lora_fp32(m * config_.intermediate_size);
          std::vector<ggml_bf16_t> gate_lora_bf16(m * config_.intermediate_size);
          // Initialize to a known pattern to detect if to_mat writes anything
          for (size_t idx = 0; idx < gate_lora_bf16.size(); idx++) {
            gate_lora_bf16[idx] = GGML_FP32_TO_BF16(999.0f);
          }
          for (int ith = 0; ith < dump_nth; ith++) {
            printf("[DEBUG] calling to_mat with ith=%d, dump_nth=%d\n", ith, dump_nth);
            lora_gate_out_bc_[expert_idx]->to_mat(m, gate_lora_bf16.data(), ith, dump_nth);
            // Check what was written
            float val_at_0 = GGML_BF16_TO_FP32(gate_lora_bf16[0]);
            float val_at_256 = GGML_BF16_TO_FP32(gate_lora_bf16[256]);
            float val_at_512 = (m > 1) ? GGML_BF16_TO_FP32(gate_lora_bf16[512]) : 0;
            float val_at_768 = (m > 1) ? GGML_BF16_TO_FP32(gate_lora_bf16[768]) : 0;
            printf("[DEBUG] after ith=%d: buf[0]=%.6f, buf[256]=%.6f, buf[512]=%.6f, buf[768]=%.6f\n", ith, val_at_0,
                   val_at_256, val_at_512, val_at_768);
          }
          for (int j = 0; j < m * config_.intermediate_size; j++) {
            gate_lora_fp32[j] = GGML_BF16_TO_FP32(gate_lora_bf16[j]);
          }
          dump_fp32_matrix(gate_lora_fp32.data(), m, config_.intermediate_size, "gate_lora_gemm_output", tp_part_idx,
                           expert_idx);

          // Convert BufferC to FP32 and dump for up
          std::vector<float> up_lora_fp32(m * config_.intermediate_size);
          std::vector<ggml_bf16_t> up_lora_bf16(m * config_.intermediate_size);
          for (int ith = 0; ith < dump_nth; ith++) {
            lora_up_out_bc_[expert_idx]->to_mat(m, up_lora_bf16.data(), ith, dump_nth);
          }
          for (int j = 0; j < m * config_.intermediate_size; j++) {
            up_lora_fp32[j] = GGML_BF16_TO_FP32(up_lora_bf16[j]);
          }
          dump_fp32_matrix(up_lora_fp32.data(), m, config_.intermediate_size, "up_lora_gemm_output", tp_part_idx,
                           expert_idx);
        }
      }
    }

    // =====================================================
    // Step 3b: Add LoRA output to main output with scaling
    // =====================================================
    double gate_lora_sum = 0.0;
    double up_lora_sum = 0.0;

    pool->do_work_stealing_job(
        nth * activated_expert * 2, nullptr,
        [this, nth, &gate_lora_sum, &up_lora_sum](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& bc = do_up ? lora_up_out_bc_[expert_idx] : lora_gate_out_bc_[expert_idx];
          ggml_bf16_t* main_output = do_up ? m_local_up_output_ptr_[expert_idx] : m_local_gate_output_ptr_[expert_idx];
          double* lora_sum_ptr = do_up ? &up_lora_sum : &gate_lora_sum;
          add_lora_output_to_main(bc.get(), main_output, m, config_.intermediate_size, lora_scaling_, ith, nth,
                                  lora_sum_ptr);
        },
        nullptr, "fwd_lora_gu_add");
  }

  /**
   * @brief Compute LoRA for down projection using AMX GEMM.
   */
  void compute_lora_down_amx(int qlen, int activated_expert) {
    if (down_lora_a_ == nullptr || down_lora_b_ == nullptr) return;

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Ensure LoRA weights are prepared
    prepare_lora_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate lora_down_out_bc_ from shared pool
    // Note: lora_gate_intermediate_bc_ and lora_gate_intermediate_ba_ are reused
    // from compute_lora_gate_up_amx (they are not used simultaneously)
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Use offset after gate and up output buffers in lora_bc_out_pool_
    // Pool layout: [gate_out × N] [up_out × N] [down_out × N]
    // But since we allocate dynamically, we need to track the offset
    // Actually, we can reuse the lora_bc_out_pool_ starting position since
    // gate/up outputs are already consumed by this point

    // For simplicity, allocate from the end of the pool (after gate+up)
    // Calculate gate+up total size first
    size_t gate_up_total = 0;
    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      size_t local_max_m = ((m_local_num_[expert_idx] + M_STEP - 1) / M_STEP) * M_STEP;
      gate_up_total += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size)) * 2;  // gate + up
    }

    char* lora_down_bc_ptr = (char*)lora_bc_out_pool_ + gate_up_total;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      lora_down_out_bc_[expert_idx]->max_m = local_max_m;
      lora_down_out_bc_[expert_idx]->set_data(lora_down_bc_ptr);
      lora_down_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));
    }

    // =====================================================
    // Step 1: intermediate @ down_lora_A^T -> lora_intermediate
    // Uses down_ba_ (already quantized intermediate after activation)
    // =====================================================
    int nth = T::recommended_nth(padded_lora_rank_);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = down_ba_[expert_idx];  // Reuse quantized intermediate
          auto& bb = down_lora_a_bb_[expert_idx];
          // Reuse gate intermediate buffer (no race condition for down projection)
          auto& bc = lora_gate_intermediate_bc_[expert_idx];

          // GEMM: [m, intermediate_size] @ [padded_lora_rank, intermediate_size]^T -> [m, padded_lora_rank]
          amx::mat_mul(m, padded_lora_rank_, config_.intermediate_size, ba, bb, bc, ith, nth);

          // Convert BufferC to BF16 for step 2 input
          bc->to_mat(m, lora_gate_intermediate_ptr_[expert_idx], ith, nth);
        },
        nullptr, "fwd_lora_down_a");

    // DUMP: LoRA intermediate (intermediate @ down_lora_A^T) for down
    // Note: Use padded_lora_rank_ as stride since to_mat writes with this stride
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          // Down reuses lora_gate_intermediate_ptr_
          dump_bf16_matrix(lora_gate_intermediate_ptr_[expert_idx], m, padded_lora_rank_, "down_lora_intermediate",
                           tp_part_idx, expert_idx);
        }
      }
    }

    // =====================================================
    // Step 2: Quantize lora_intermediate to BufferA
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          // Reuse gate intermediate buffer (no race condition for down projection)
          lora_gate_intermediate_ba_[expert_idx]->from_mat(m, lora_gate_intermediate_ptr_[expert_idx], 0, 1);
        },
        nullptr, "fwd_lora_down_quant");

    // =====================================================
    // Step 3a: lora_intermediate @ down_lora_B^T -> lora_output (GEMM only)
    // =====================================================
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          // Reuse gate intermediate buffer (no race condition for down projection)
          auto& ba = lora_gate_intermediate_ba_[expert_idx];
          auto& bb = down_lora_b_bb_[expert_idx];
          auto& bc = lora_down_out_bc_[expert_idx];

          // GEMM: [m, padded_lora_rank] @ [hidden_size, padded_lora_rank]^T -> [m, hidden_size]
          amx::mat_mul(m, config_.hidden_size, padded_lora_rank_, ba, bb, bc, ith, nth);
        },
        nullptr, "fwd_lora_down_gemm", 1);

    // DUMP: Pure down LoRA GEMM output (before scaling and add)
    // Note: to_mat with (ith, nth) only reads one N_BLOCK chunk, so we need to loop
    if (is_dump_enabled()) {
      int dump_nth = T::recommended_nth(config_.hidden_size);
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          // Convert BufferC to FP32 matrix and dump
          std::vector<float> lora_out_fp32(m * config_.hidden_size);
          auto& bc = lora_down_out_bc_[expert_idx];
          // Use to_mat to convert, but we need BF16 temp buffer
          std::vector<ggml_bf16_t> lora_out_bf16(m * config_.hidden_size);
          // Loop over all N_BLOCK chunks
          for (int ith = 0; ith < dump_nth; ith++) {
            bc->to_mat(m, lora_out_bf16.data(), ith, dump_nth);
          }
          for (int j = 0; j < m * config_.hidden_size; j++) {
            lora_out_fp32[j] = GGML_BF16_TO_FP32(lora_out_bf16[j]);
          }
          dump_fp32_matrix(lora_out_fp32.data(), m, config_.hidden_size, "down_lora_gemm_output", tp_part_idx,
                           expert_idx);
        }
      }
    }

    // =====================================================
    // Step 3b: Add LoRA output to main output with scaling
    // =====================================================
    double down_lora_sum = 0.0;

    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [this, nth, &down_lora_sum](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& bc = lora_down_out_bc_[expert_idx];

          // Add LoRA output to main output with scaling and collect statistics
          add_lora_output_to_main(bc.get(), m_local_down_output_ptr_[expert_idx], m, config_.hidden_size, lora_scaling_,
                                  ith, nth, &down_lora_sum);
        },
        nullptr, "fwd_lora_down_add");

    // // Print LoRA contribution statistics
    // size_t total_elements = 0;
    // for (int i = 0; i < activated_expert; i++) {
    //   total_elements += m_local_num_[m_expert_id_map_[i]];
    // }
    // total_elements *= config_.hidden_size;

    // if (total_elements > 0) {
    //   double down_lora_mean = down_lora_sum / total_elements;
    //   printf("[LoRA] layer=%d down_mean=%.6e\n", config_.layer_idx, down_lora_mean);
    // }
  }

  /**
   * @brief Add LoRA BufferC output to main BF16 output with scaling.
   *
   * main_output[i] += lora_bc_output[i] * scaling
   * @param lora_sum Optional pointer to accumulate sum of absolute LoRA contributions for statistics
   */
  void add_lora_output_to_main(typename T::BufferC* bc, ggml_bf16_t* main_output, int m, int n, float scaling, int ith,
                               int nth, double* lora_sum = nullptr) {
    // BUG FIX: BufferC uses tiled layout [n_blocks][m_blocks][n_steps][M_STEP][N_STEP]
    // We must iterate over tiles (m_begin in M_STEP steps) and rows within tiles (i)
    // to correctly compute the offset into the tiled buffer.
    constexpr int M_STEP = T::M_STEP;
    constexpr int N_STEP = T::N_STEP;
    constexpr int N_BLOCK = T::N_BLOCK;

    auto [n_start, n_end] = T::split_range_n(n, ith, nth);
    double local_sum = 0.0;

    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;

    __m512 scale = _mm512_set1_ps(scaling);

    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          // Compute correct offset into tiled BufferC (same formula as BufferC::to_mat)
          float* c_ptr = bc->c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP;

          // Load from main output (BF16)
          int row = m_begin + i;
          int col = n_block_begin + n_begin;
          __m512 main0, main1;
          avx512_32xbf16_to_32xfp32((__m512i*)(main_output + row * n + col), &main0, &main1);

          // Load LoRA output from BufferC (FP32)
          __m512 lora0 = _mm512_load_ps(c_ptr);
          __m512 lora1 = _mm512_load_ps(c_ptr + 16);

          // // Accumulate absolute LoRA contribution for statistics
          // if (lora_sum != nullptr) {
          //   for (int j = 0; j < 16; j++) {
          //     local_sum += std::abs(c_ptr[j] * scaling);
          //     local_sum += std::abs(c_ptr[j + 16] * scaling);
          //   }
          // }

          // Add with scaling: main = main + lora * scale
          main0 = _mm512_fmadd_ps(lora0, scale, main0);
          main1 = _mm512_fmadd_ps(lora1, scale, main1);

          // Store back to main output (BF16)
          avx512_32xfp32_to_32xbf16(&main0, &main1, (__m512i*)(main_output + row * n + col));
        }
      }
    }

    // if (lora_sum != nullptr) {
    //   #pragma omp atomic
    //   *lora_sum += local_sum;
    // }
  }

  /**
   * @brief Compute LoRA for gate and up projections.
   *
   * gate_lora_out = (input @ gate_lora_A^T) @ gate_lora_B^T * scaling
   * gate_output += gate_lora_out
   * (similar for up)
   */
  void compute_lora_gate_up(int qlen, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    double gate_lora_sum = 0.0;
    double up_lora_sum = 0.0;

    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,
        [this, &gate_lora_sum, &up_lora_sum](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Get weight pointers
          ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          ggml_bf16_t* lora_b = do_up ? up_lora_b_ : gate_lora_b_;
          ggml_bf16_t* input = m_local_input_ptr_[expert_idx];
          ggml_bf16_t* output = do_up ? m_local_up_output_ptr_[expert_idx] : m_local_gate_output_ptr_[expert_idx];

          if (lora_a == nullptr || lora_b == nullptr) return;

          // Offset to current expert's weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
          size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          ggml_bf16_t* expert_lora_b = lora_b + lora_b_offset;

          // Use thread-local intermediate buffer to avoid race conditions
          // This completely eliminates the shared buffer contention issue
          std::vector<float> local_intermediate(num_tokens * lora_rank_);
          double local_lora_sum = 0.0;

          // Step 1: intermediate = input @ lora_A^T
          // [num_tokens, hidden_size] @ [lora_rank, hidden_size]^T → [num_tokens, lora_rank]
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int h = 0; h < config_.hidden_size; h++) {
                float inp = GGML_BF16_TO_FP32(input[t * config_.hidden_size + h]);
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                sum += inp * w;
              }
              local_intermediate[t * lora_rank_ + r] = sum;
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [intermediate_size, lora_rank]^T → [num_tokens, intermediate_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int i = 0; i < config_.intermediate_size; i++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = local_intermediate[t * lora_rank_ + r];
                float w = GGML_BF16_TO_FP32(expert_lora_b[i * lora_rank_ + r]);
                sum += inter * w;
              }
              // Add to output with scaling and accumulate statistics
              float lora_contrib = sum * lora_scaling_;
              local_lora_sum += std::abs(lora_contrib);

              float out_val = GGML_BF16_TO_FP32(output[t * config_.intermediate_size + i]);
              out_val += lora_contrib;
              output[t * config_.intermediate_size + i] = GGML_FP32_TO_BF16(out_val);
            }
          }

          // // Accumulate to global sum
          // if (do_up) {
          //   #pragma omp atomic
          //   up_lora_sum += local_lora_sum;
          // } else {
          //   #pragma omp atomic
          //   gate_lora_sum += local_lora_sum;
          // }
        },
        nullptr, "fwd_lora_gu_fallback");

    // // Print LoRA contribution statistics
    // size_t total_elements = 0;
    // for (int i = 0; i < activated_expert; i++) {
    //   total_elements += m_local_num_[m_expert_id_map_[i]];
    // }
    // total_elements *= config_.intermediate_size;

    // if (total_elements > 0) {
    //   double gate_lora_mean = gate_lora_sum / total_elements;
    //   double up_lora_mean = up_lora_sum / total_elements;
    //   printf("[LoRA] layer=%d gate_mean=%.6e up_mean=%.6e\n", config_.layer_idx, gate_lora_mean, up_lora_mean);
    // }
  }

  /**
   * @brief Compute LoRA for down projection.
   */
  void compute_lora_down(int qlen, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    double down_lora_sum = 0.0;

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &down_lora_sum](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          ggml_bf16_t* input = m_local_gate_output_ptr_[expert_idx];  // After activation
          ggml_bf16_t* output = m_local_down_output_ptr_[expert_idx];

          if (down_lora_a_ == nullptr || down_lora_b_ == nullptr) return;

          // Offset to current expert's weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
          size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
          ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

          // Use thread-local intermediate buffer to avoid race conditions
          std::vector<float> local_intermediate(num_tokens * lora_rank_);
          double local_lora_sum = 0.0;

          // Step 1: intermediate = input @ lora_A^T
          // [num_tokens, intermediate_size] @ [lora_rank, intermediate_size]^T → [num_tokens, lora_rank]
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int i = 0; i < config_.intermediate_size; i++) {
                float inp = GGML_BF16_TO_FP32(input[t * config_.intermediate_size + i]);
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
                sum += inp * w;
              }
              local_intermediate[t * lora_rank_ + r] = sum;
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [hidden_size, lora_rank]^T → [num_tokens, hidden_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int h = 0; h < config_.hidden_size; h++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = local_intermediate[t * lora_rank_ + r];
                float w = GGML_BF16_TO_FP32(expert_lora_b[h * lora_rank_ + r]);
                sum += inter * w;
              }
              // Add to output with scaling and accumulate statistics
              float lora_contrib = sum * lora_scaling_;
              local_lora_sum += std::abs(lora_contrib);

              float out_val = GGML_BF16_TO_FP32(output[t * config_.hidden_size + h]);
              out_val += lora_contrib;
              output[t * config_.hidden_size + h] = GGML_FP32_TO_BF16(out_val);
            }
          }

// Accumulate to global sum
#pragma omp atomic
          down_lora_sum += local_lora_sum;
        },
        nullptr, "fwd_lora_down_fallback");

    // Print LoRA contribution statistics
    size_t total_elements = 0;
    for (int i = 0; i < activated_expert; i++) {
      total_elements += m_local_num_[m_expert_id_map_[i]];
    }
    total_elements *= config_.hidden_size;

    if (total_elements > 0) {
      double down_lora_mean = down_lora_sum / total_elements;
      printf("[LoRA] layer=%d down_mean=%.6e\n", config_.layer_idx, down_lora_mean);
    }
  }

  ForwardCache& push_cache() {
    if (cache_stack_top_ >= max_cache_depth_) {
      std::cerr << "[KT-MOE ERROR] Forward cache stack overflow!" << std::endl;
      std::cerr << "  cache_stack_top_ = " << cache_stack_top_ << std::endl;
      std::cerr << "  max_cache_depth_ = " << max_cache_depth_ << std::endl;
      std::cerr << "  Hint: If you are doing inference (forward only without backward)," << std::endl;
      std::cerr << "        set save_for_backward=False in forward_sft() call." << std::endl;
      std::cerr << "        Or increase max_cache_depth in MOESFTConfig." << std::endl;
      throw std::runtime_error("Forward cache stack overflow");
    }
    return cache_stack_[cache_stack_top_++];
  }

  ForwardCache pop_cache() {
    if (cache_stack_top_ <= 0) {
      std::cerr << "[KT-MOE ERROR] Forward cache stack underflow!" << std::endl;
      std::cerr << "  cache_stack_top_ = " << cache_stack_top_ << std::endl;
      std::cerr << "  Hint: Calling backward() without corresponding forward(save_for_backward=True)." << std::endl;
      throw std::runtime_error("Forward cache stack underflow");
    }
    return cache_stack_[--cache_stack_top_];
  }

  void save_to_cache(ForwardCache& cache, int qlen, int k, const int64_t* expert_ids, const float* weights,
                     int activated_expert, const void* input) {
    cache.qlen_cache = qlen;
    cache.k_cache = k;
    cache.activated_expert_cache = activated_expert;

    // Copy routing information
    cache.expert_ids_cache.resize(qlen * k);
    cache.weights_cache.resize(qlen * k);
    std::copy(expert_ids, expert_ids + qlen * k, cache.expert_ids_cache.begin());
    std::copy(weights, weights + qlen * k, cache.weights_cache.begin());

    cache.m_local_num_cache = m_local_num_;
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        cache.m_local_pos_cache[i][j] = m_local_pos_[i][j];
      }
    }
    for (int i = 0; i < activated_expert; i++) {
      cache.m_expert_id_map_cache[i] = m_expert_id_map_[i];
    }

    // Copy intermediate values
    size_t total_tokens = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      total_tokens += m_local_num_[expert_idx];
    }

    // Copy input (original input, not m_local_input_ which is expert-sorted)
    // Bug #17b fix: backward_gate_up expects original token order
    memcpy(cache.input_cache, input, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

    // Copy gate and up outputs (before activation)
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      int num_tokens = m_local_num_[expert_idx];
      memcpy(cache.gate_output_cache + offset * config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      memcpy(cache.up_output_cache + offset * config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      offset += num_tokens;
    }

    // Debug code commented out - Bug #15 verified fixed
    // printf("[DEBUG save_to_cache] total_tokens=%zu\n", offset);
    // printf("[DEBUG ADDR] cache.gate_output_cache = %p, cache.up_output_cache = %p\n", (void*)cache.gate_output_cache,
    //        (void*)cache.up_output_cache);
    // printf("[DEBUG save_to_cache] gate_output_cache[0..7] = ");
    // for (int i = 0; i < 8 && i < (int)(offset * config_.intermediate_size); i++) {
    //   printf("%.4f ", GGML_BF16_TO_FP32(cache.gate_output_cache[i]));
    // }
    // printf("\n");

    cache.valid = true;
  }

  /**
   * @brief Save intermediate values AFTER activation for backward_down.
   *
   * Must be called after apply_activation() since m_local_gate_output_ptr_
   * now contains silu(gate) * up (the intermediate value).
   */
  void save_intermediate_to_cache(ForwardCache& cache, int activated_expert) {
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      int num_tokens = m_local_num_[expert_idx];
      // m_local_gate_output_ptr_ now contains intermediate (after activation: silu(gate) * up)
      memcpy(cache.intermediate_cache + offset * config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      offset += num_tokens;
    }
  }

  /**
   * @brief Save down projection output for grad_weights computation.
   *
   * Must be called after down projection (and LoRA) but before weighted merge.
   */
  void save_down_output_to_cache(ForwardCache& cache, int activated_expert) {
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      int num_tokens = m_local_num_[expert_idx];
      memcpy(cache.down_output_cache + offset * config_.hidden_size, m_local_down_output_ptr_[expert_idx],
             num_tokens * config_.hidden_size * sizeof(ggml_bf16_t));
      offset += num_tokens;
    }
  }

  void backward_down(const ForwardCache& cache, const void* grad_output, void* grad_down_lora_a,
                     void* grad_down_lora_b) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Debug code commented out - Bug #15 verified fixed
    // printf("[DEBUG ADDR backward_down] grad_intermediate_ = %p\n", (void*)grad_intermediate_);
    // printf("[DEBUG ADDR backward_down] cache.gate_output_cache = %p\n", (void*)cache.gate_output_cache);
    // printf("[DEBUG ADDR backward_down] cache.up_output_cache = %p\n", (void*)cache.up_output_cache);
    // printf("[DEBUG BEFORE memset] gate_cache[0..3] = %.4f %.4f %.4f %.4f\n",
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[0]), GGML_BF16_TO_FP32(cache.gate_output_cache[1]),
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[2]), GGML_BF16_TO_FP32(cache.gate_output_cache[3]));

    // Initialize gradient intermediate buffer
    memset(grad_intermediate_, 0,
           config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));

    // printf("[DEBUG AFTER memset] gate_cache[0..3] = %.4f %.4f %.4f %.4f\n",
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[0]), GGML_BF16_TO_FP32(cache.gate_output_cache[1]),
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[2]), GGML_BF16_TO_FP32(cache.gate_output_cache[3]));

    // Scatter grad_output to per-expert buffers and compute gradients
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, grad_output, grad_down_a, grad_down_b, qlen, k](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Collect gradients for this expert from grad_output
          // grad_output is [qlen, hidden_size] in bf16, need to scatter based on routing
          const ggml_bf16_t* grad_out = (const ggml_bf16_t*)grad_output;
          std::vector<float> expert_grad_out(num_tokens * config_.hidden_size, 0.0f);

          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                float w = cache.weights_cache[i * k + j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  expert_grad_out[pos * config_.hidden_size + h] +=
                      GGML_BF16_TO_FP32(grad_out[i * config_.hidden_size + h]) * w;
                }
              }
            }
          }

          // Get cached intermediate (after activation)
          ggml_bf16_t* intermediate = cache.intermediate_cache;  // Will use gate_output_cache after activation saved

          // Compute grad w.r.t. intermediate: grad_intermediate = grad_output @ down_proj
          // down_proj layout: [expert_num, hidden_size, intermediate_size]
          // grad_output: [num_tokens, hidden_size], grad_intermediate: [num_tokens, intermediate_size]
          // grad_intermediate[t, i] = sum_h grad_output[t, h] * down_proj[h, i]
          {
            const ggml_bf16_t* down_proj = (const ggml_bf16_t*)config_.down_proj;
            size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;

            // Compute offset into grad_intermediate_ for this expert
            size_t grad_inter_offset = 0;
            for (int e = 0; e < task_id; e++) {
              grad_inter_offset += m_local_num_[m_expert_id_map_[e]];
            }
            grad_inter_offset *= config_.intermediate_size;

            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int h = 0; h < config_.hidden_size; h++) {
                  float grad_out_val = expert_grad_out[t * config_.hidden_size + h];
                  float down_val = GGML_BF16_TO_FP32(down_proj[expert_offset + h * config_.intermediate_size + i]);
                  sum += grad_out_val * down_val;
                }
                grad_intermediate_[grad_inter_offset + t * config_.intermediate_size + i] = GGML_FP32_TO_BF16(sum);
              }
            }
          }

          // Skip LoRA gradient computation when SkipLoRA is true
          if (!SkipLoRA && down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
            // Get expert's LoRA weights
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
            size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
            ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

            // Bug #17c fix: Use cached intermediate (after activation), not gate_output_cache (before activation)
            // The cache is stored in task order (activated expert order), need to compute offset
            size_t cache_offset = 0;
            for (int e = 0; e < task_id; e++) {
              cache_offset += m_local_num_[m_expert_id_map_[e]];
            }
            const ggml_bf16_t* cached_intermediate =
                cache.intermediate_cache + cache_offset * config_.intermediate_size;

            // Gradient for LoRA B: grad_B = grad_output^T @ (intermediate @ lora_A^T) * scaling
            // = (grad_output^T @ intermediate @ lora_A^T) * scaling
            // Shape: [hidden_size, num_tokens] @ [num_tokens, lora_rank] → [hidden_size, lora_rank]

            // First compute intermediate @ lora_A^T → [num_tokens, lora_rank]
            std::vector<float> inter_proj(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  // Use cached intermediate (gate_output after activation)
                  float inp = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
                  float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
                  sum += inp * w;
                }
                inter_proj[t * lora_rank_ + r] = sum;
              }
            }

            // grad_B = grad_output^T @ inter_proj * scaling
            // [hidden_size, num_tokens] @ [num_tokens, lora_rank] → [hidden_size, lora_rank]
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += expert_grad_out[t * config_.hidden_size + h] * inter_proj[t * lora_rank_ + r];
                }
                // Accumulate gradient
                size_t idx = lora_b_offset + h * lora_rank_ + r;
                float cur = GGML_BF16_TO_FP32(grad_down_b[idx]);
                cur += sum * lora_scaling_;
                grad_down_b[idx] = GGML_FP32_TO_BF16(cur);
              }
            }

            // Gradient for LoRA A: more complex, involves backprop through lora_B
            // grad_A = (lora_B^T @ grad_output^T @ intermediate)^T * scaling
            // = intermediate^T @ grad_output @ lora_B * scaling
            // Shape: [intermediate_size, num_tokens] @ [num_tokens, hidden_size] @ [hidden_size, lora_rank]
            //      = [intermediate_size, lora_rank]
            // First: grad_output @ lora_B → [num_tokens, lora_rank]
            std::vector<float> grad_times_b(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int h = 0; h < config_.hidden_size; h++) {
                  float g = expert_grad_out[t * config_.hidden_size + h];
                  float b = GGML_BF16_TO_FP32(expert_lora_b[h * lora_rank_ + r]);
                  sum += g * b;
                }
                grad_times_b[t * lora_rank_ + r] = sum;
              }
            }

            // grad_A = intermediate^T @ grad_times_b * scaling
            // [intermediate_size, num_tokens] @ [num_tokens, lora_rank] → [intermediate_size, lora_rank]
            // But A is stored as [lora_rank, intermediate_size], so we compute for that layout
            for (int r = 0; r < lora_rank_; r++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  // Bug #17a fix: Use cached_intermediate instead of m_local_gate_output_ptr_
                  float inter = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
                  sum += inter * grad_times_b[t * lora_rank_ + r];
                }
                size_t idx_a = lora_a_offset + r * config_.intermediate_size + i;
                float cur = GGML_BF16_TO_FP32(grad_down_a[idx_a]);
                cur += sum * lora_scaling_;
                grad_down_a[idx_a] = GGML_FP32_TO_BF16(cur);
              }
            }
          }
        },
        nullptr, "bwd_down");
  }

  /**
   * @brief AMX-optimized backward pass for down projection.
   *
   * Optimizes the main GEMM: grad_intermediate = grad_output @ down_proj
   * Using AMX mat_mul with down_backward_bb_ (transposed weight).
   *
   * LoRA gradient computation is kept as for-loop for now due to complexity
   * and small matrix sizes involved.
   */
  void backward_down_amx(const ForwardCache& cache, const void* grad_output, void* grad_down_lora_a,
                         void* grad_down_lora_b) {
    // Timing for backward_down_amx substeps
    static int _down_call_count = 0;
    _down_call_count++;
    bool _down_should_print = (_down_call_count == get_print_at_call());
    auto _down_start = std::chrono::high_resolution_clock::now();
    auto _down_last = _down_start;

#define DOWN_CHECKPOINT(name)                                                                                       \
  do {                                                                                                              \
    auto _now = std::chrono::high_resolution_clock::now();                                                          \
    if (_down_should_print) {                                                                                       \
      printf("  [DOWN] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _down_last).count()); \
    }                                                                                                               \
    _down_last = _now;                                                                                              \
  } while (0)

    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Ensure backward weights are prepared
    prepare_backward_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate backward buffers from shared pool
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    char* backward_ba_ptr = (char*)backward_ba_pool_;
    char* backward_bc_ptr = (char*)backward_bc_pool_;
    char* grad_output_bf16_ptr = (char*)grad_output_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferA for grad_output
      grad_output_ba_[expert_idx]->max_m = local_max_m;
      grad_output_ba_[expert_idx]->set_data(backward_ba_ptr);
      backward_ba_ptr += align64(T::BufferA::required_size(local_max_m, config_.hidden_size));

      // Allocate BufferC for grad_intermediate
      grad_intermediate_bc_[expert_idx]->max_m = local_max_m;
      grad_intermediate_bc_[expert_idx]->set_data(backward_bc_ptr);
      backward_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      // Allocate BF16 buffer for scattered grad_output
      grad_output_bf16_ptr_[expert_idx] = (ggml_bf16_t*)grad_output_bf16_ptr;
      grad_output_bf16_ptr += align64(local_max_m * config_.hidden_size * sizeof(ggml_bf16_t));
    }

    // Initialize gradient intermediate buffer
    memset(grad_intermediate_, 0,
           config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));

    DOWN_CHECKPOINT("D0_prepare+memset");

    // =====================================================
    // Step 1: Scatter grad_output to per-expert BF16 buffers
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, grad_output, qlen, k](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          const ggml_bf16_t* grad_out = (const ggml_bf16_t*)grad_output;
          ggml_bf16_t* expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];

          // Zero the buffer first
          memset(expert_grad_bf16, 0, num_tokens * config_.hidden_size * sizeof(ggml_bf16_t));

          // Scatter grad_output to expert's buffer (with routing weights)
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                float w = cache.weights_cache[i * k + j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  float val = GGML_BF16_TO_FP32(expert_grad_bf16[pos * config_.hidden_size + h]);
                  val += GGML_BF16_TO_FP32(grad_out[i * config_.hidden_size + h]) * w;
                  expert_grad_bf16[pos * config_.hidden_size + h] = GGML_FP32_TO_BF16(val);
                }
              }
            }
          }
        },
        nullptr, "bwd_down_scatter");

    DOWN_CHECKPOINT("D1_scatter");

    // =====================================================
    // Step 2: Quantize scattered grad_output to BufferA
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Convert scattered grad_output to BufferA
          grad_output_ba_[expert_idx]->from_mat(num_tokens, grad_output_bf16_ptr_[expert_idx], 0, 1);
        },
        nullptr, "bwd_down_quantize");

    DOWN_CHECKPOINT("D2_quantize");

    // =====================================================
    // Step 3+4: AMX GEMM + to_mat (merged to use same ith/nth)
    // grad_intermediate = grad_output @ down_proj
    // Using: A @ B^T where A = grad_output, B = down_proj^T (stored in down_backward_bb_)
    // m = num_tokens, n = intermediate_size, k = hidden_size
    //
    // BUG FIX: Previously Step 3 used (ith, nth) for mat_mul but Step 4 used (0, 1) for to_mat,
    // which only output the first N_BLOCK columns. Now merged to use same (ith, nth).
    // =====================================================
    int nth = T::recommended_nth(config_.intermediate_size);

    // Pre-compute offsets for each expert (needed for to_mat output location)
    std::vector<size_t> expert_offsets(activated_expert);
    {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        expert_offsets[i] = offset * config_.intermediate_size;
        offset += m_local_num_[m_expert_id_map_[i]];
      }
    }

    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, &expert_offsets](int task_id) {
          int task_idx = task_id / nth;  // Which expert (0 to activated_expert-1)
          int expert_idx = m_expert_id_map_[task_idx];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = grad_output_ba_[expert_idx];
          auto& bb = down_backward_bb_[expert_idx];
          auto& bc = grad_intermediate_bc_[expert_idx];

          // mat_mul: [m, hidden_size] @ [intermediate_size, hidden_size]^T = [m, intermediate_size]
          amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);

          // to_mat: Convert BufferC to BF16 - use same ith, nth as mat_mul!
          bc->to_mat(m, grad_intermediate_ + expert_offsets[task_idx], ith, nth);
        },
        nullptr, "bwd_down_gemm", 1);

    DOWN_CHECKPOINT("D3_gemm");

    // =====================================================
    // Step 3.5: Add LoRA contribution to grad_intermediate
    // grad_intermediate += grad_output @ down_lora_B @ down_lora_A * scaling
    // This is needed for correct backward through activation to gate/up
    // =====================================================
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      const int hidden = config_.hidden_size;
      const int inter_size = config_.intermediate_size;
      const int rank = lora_rank_;

      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [this, &cache, &expert_offsets, hidden, inter_size, rank](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Get expert's LoRA weights
            // down_lora_a: [expert_num, lora_rank, intermediate_size]
            // down_lora_b: [expert_num, hidden_size, lora_rank]
            size_t lora_a_offset = (size_t)expert_idx * rank * inter_size;
            size_t lora_b_offset = (size_t)expert_idx * hidden * rank;
            const ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            const ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;
            const ggml_bf16_t* expert_grad = grad_output_bf16_ptr_[expert_idx];

            // Output location in grad_intermediate_
            // expert_offsets[task_id] is already (token_offset * intermediate_size)
            ggml_bf16_t* grad_inter = grad_intermediate_ + expert_offsets[task_id];

            // Step 1: grad_output @ down_lora_B -> [num_tokens, rank]
            std::vector<float> grad_times_b(num_tokens * rank, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < rank; r++) {
                float sum = 0.0f;
                for (int h = 0; h < hidden; h++) {
                  float g = GGML_BF16_TO_FP32(expert_grad[t * hidden + h]);
                  float b = GGML_BF16_TO_FP32(expert_lora_b[h * rank + r]);
                  sum += g * b;
                }
                grad_times_b[t * rank + r] = sum;
              }
            }

            // Step 2: grad_times_b @ down_lora_A -> [num_tokens, inter_size]
            // Accumulate to grad_intermediate with scaling
            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < inter_size; i++) {
                float sum = 0.0f;
                for (int r = 0; r < rank; r++) {
                  float gtb = grad_times_b[t * rank + r];
                  float a = GGML_BF16_TO_FP32(expert_lora_a[r * inter_size + i]);
                  sum += gtb * a;
                }
                // Accumulate with scaling
                float cur = GGML_BF16_TO_FP32(grad_inter[t * inter_size + i]);
                cur += sum * lora_scaling_;
                grad_inter[t * inter_size + i] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          nullptr, "bwd_down_lora_to_inter");
    }

    DOWN_CHECKPOINT("D3.5_lora_to_inter");

    // DUMP: backward grad_output and grad_intermediate (base) after GEMM
    if (is_dump_enabled()) {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          // Dump scattered grad_output
          dump_bf16_matrix(grad_output_bf16_ptr_[expert_idx], m, config_.hidden_size, "backward_grad_output",
                           tp_part_idx, expert_idx);
          // Dump grad_intermediate (base, before LoRA)
          dump_bf16_matrix(grad_intermediate_ + offset, m, config_.intermediate_size, "backward_down_base", tp_part_idx,
                           expert_idx);
        }
        offset += m * config_.intermediate_size;
      }
    }

    // =====================================================
    // Step 5: LoRA gradient computation (parallelized across blocks)
    // Skip when SkipLoRA is true (only compute grad_input, not LoRA weight gradients)
    // =====================================================
    if (!SkipLoRA && down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      struct LoraGradBuf {
        int expert_idx = -1;
        int num_tokens = 0;
        size_t lora_a_offset = 0;
        size_t lora_b_offset = 0;
        const ggml_bf16_t* cached_intermediate = nullptr;
        const ggml_bf16_t* expert_lora_a = nullptr;
        const ggml_bf16_t* expert_lora_b = nullptr;
        const ggml_bf16_t* expert_grad_bf16 = nullptr;
        std::vector<float> grad_out;
        std::vector<float> inter_proj;
        std::vector<float> grad_times_b;
      };

      const int hidden = config_.hidden_size;
      const int inter_size = config_.intermediate_size;
      const int rank = lora_rank_;
      constexpr int kPrecomputeTokenBlock = 4;
      constexpr int kGradBBlock = 128;
      constexpr int kGradABlock = 256;

      std::vector<LoraGradBuf> lora_grad_bufs(activated_expert);

      auto load_bf16_16 = [](const ggml_bf16_t* src, __mmask16 mask) -> __m512 {
        __m256i bf16 = _mm256_maskz_loadu_epi16(mask, src);
        __m512i i32 = _mm512_cvtepu16_epi32(bf16);
        return _mm512_castsi512_ps(_mm512_slli_epi32(i32, 16));
      };
      auto store_bf16_16 = [](ggml_bf16_t* dst, __m512 v) {
        __m256i bf16 = (__m256i)_mm512_cvtneps_pbh(v);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), bf16);
      };

      // Initialize per-expert buffers (metadata + allocations)
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, hidden, inter_size, rank](int task_id) {
            LoraGradBuf& buf = lora_grad_bufs[task_id];
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];

            buf.expert_idx = expert_idx;
            buf.num_tokens = num_tokens;
            if (num_tokens == 0) return;

            buf.lora_a_offset = static_cast<size_t>(expert_idx) * rank * inter_size;
            buf.lora_b_offset = static_cast<size_t>(expert_idx) * hidden * rank;
            buf.expert_lora_a = down_lora_a_ + buf.lora_a_offset;
            buf.expert_lora_b = down_lora_b_ + buf.lora_b_offset;
            buf.expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];

            size_t token_offset = expert_offsets[task_id] / inter_size;
            buf.cached_intermediate = cache.intermediate_cache + token_offset * inter_size;

            buf.grad_out.resize(num_tokens * hidden);
            buf.inter_proj.resize(num_tokens * rank);
            buf.grad_times_b.assign(num_tokens * rank, 0.0f);
          },
          nullptr, "bwd_down_lora_init");

      // Precompute grad_out, inter_proj, grad_times_b per token block
      std::vector<int> precompute_offsets(activated_expert + 1, 0);
      for (int i = 0; i < activated_expert; i++) {
        int num_tokens = lora_grad_bufs[i].num_tokens;
        int blocks = (num_tokens + kPrecomputeTokenBlock - 1) / kPrecomputeTokenBlock;
        precompute_offsets[i + 1] = precompute_offsets[i] + blocks;
      }

      int precompute_tasks = precompute_offsets[activated_expert];
      if (precompute_tasks > 0) {
        pool->do_work_stealing_job(
            precompute_tasks, nullptr,
            [&, hidden, inter_size, rank](int task_id) {
              int expert_task =
                  static_cast<int>(std::upper_bound(precompute_offsets.begin(), precompute_offsets.end(), task_id) -
                                   precompute_offsets.begin() - 1);
              LoraGradBuf& buf = lora_grad_bufs[expert_task];
              int num_tokens = buf.num_tokens;
              if (num_tokens == 0) return;

              int block_idx = task_id - precompute_offsets[expert_task];
              int t_begin = block_idx * kPrecomputeTokenBlock;
              int t_end = std::min(num_tokens, t_begin + kPrecomputeTokenBlock);
              if (t_begin >= t_end) return;

              int hidden_vec_end = hidden & ~31;
              int inter_vec_end = inter_size & ~31;
              int rank_vec_end = rank & ~15;

              for (int t = t_begin; t < t_end; t++) {
                // Convert grad_output to fp32
                const ggml_bf16_t* grad_row = buf.expert_grad_bf16 + t * hidden;
                float* grad_out_row = buf.grad_out.data() + t * hidden;
                int h = 0;
                for (; h < hidden_vec_end; h += 32) {
                  __m512 g0, g1;
                  avx512_32xbf16_to_32xfp32((__m512i*)(grad_row + h), &g0, &g1);
                  _mm512_storeu_ps(grad_out_row + h, g0);
                  _mm512_storeu_ps(grad_out_row + h + 16, g1);
                }
                for (; h < hidden; h++) {
                  grad_out_row[h] = GGML_BF16_TO_FP32(grad_row[h]);
                }

                // intermediate @ lora_A^T -> inter_proj
                const ggml_bf16_t* inter_row = buf.cached_intermediate + t * inter_size;
                for (int r = 0; r < rank; r++) {
                  __m512 sum0 = _mm512_setzero_ps();
                  __m512 sum1 = _mm512_setzero_ps();
                  int i = 0;
                  for (; i < inter_vec_end; i += 32) {
                    __m512 x0, x1, w0, w1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(inter_row + i), &x0, &x1);
                    avx512_32xbf16_to_32xfp32((__m512i*)(buf.expert_lora_a + r * inter_size + i), &w0, &w1);
                    sum0 = _mm512_fmadd_ps(x0, w0, sum0);
                    sum1 = _mm512_fmadd_ps(x1, w1, sum1);
                  }
                  float sum = _mm512_reduce_add_ps(sum0) + _mm512_reduce_add_ps(sum1);
                  for (; i < inter_size; i++) {
                    float inp = GGML_BF16_TO_FP32(inter_row[i]);
                    float w = GGML_BF16_TO_FP32(buf.expert_lora_a[r * inter_size + i]);
                    sum += inp * w;
                  }
                  buf.inter_proj[t * rank + r] = sum;
                }

                // grad_output @ lora_B -> grad_times_b
                float* out_row = buf.grad_times_b.data() + t * rank;
                for (int h2 = 0; h2 < hidden; h2++) {
                  float g = grad_out_row[h2];
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const ggml_bf16_t* b_row = buf.expert_lora_b + h2 * rank;
                  int r = 0;
                  for (; r < rank_vec_end; r += 16) {
                    __m512 acc = _mm512_loadu_ps(out_row + r);
                    __m512 b = load_bf16_16(b_row + r, 0xFFFF);
                    acc = _mm512_fmadd_ps(b, g_vec, acc);
                    _mm512_storeu_ps(out_row + r, acc);
                  }
                  if (r < rank) {
                    __mmask16 mask = static_cast<__mmask16>((1u << (rank - r)) - 1);
                    __m512 acc = _mm512_maskz_loadu_ps(mask, out_row + r);
                    __m512 b = load_bf16_16(b_row + r, mask);
                    acc = _mm512_fmadd_ps(b, g_vec, acc);
                    _mm512_mask_storeu_ps(out_row + r, mask, acc);
                  }
                }
              }
            },
            nullptr, "bwd_down_lora_precompute", 1);
      }

      // grad_B = grad_output^T @ inter_proj * scaling
      int h_blocks = (hidden + kGradBBlock - 1) / kGradBBlock;
      pool->do_work_stealing_job(
          activated_expert * h_blocks, nullptr,
          [&, hidden, rank, h_blocks](int task_id) {
            int expert_task = task_id / h_blocks;
            int block_idx = task_id % h_blocks;
            LoraGradBuf& buf = lora_grad_bufs[expert_task];
            if (buf.num_tokens == 0) return;

            int h_start = block_idx * kGradBBlock;
            int h_end = std::min(hidden, h_start + kGradBBlock);
            int rank_vec32 = rank & ~31;
            int rank_vec16 = rank & ~15;
            const float scale = lora_scaling_;
            const __m512 scale_vec = _mm512_set1_ps(scale);
            ggml_bf16_t* grad_b_base = grad_down_b + buf.lora_b_offset;

            for (int h = h_start; h < h_end; h++) {
              int r = 0;
              for (; r < rank_vec32; r += 32) {
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();
                for (int t = 0; t < buf.num_tokens; t++) {
                  float g = buf.grad_out[t * hidden + h];
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const float* u_row = buf.inter_proj.data() + t * rank + r;
                  __m512 u0 = _mm512_loadu_ps(u_row);
                  __m512 u1 = _mm512_loadu_ps(u_row + 16);
                  acc0 = _mm512_fmadd_ps(u0, g_vec, acc0);
                  acc1 = _mm512_fmadd_ps(u1, g_vec, acc1);
                }
                ggml_bf16_t* out = grad_b_base + h * rank + r;
                __m512 cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)out, &cur0, &cur1);
                cur0 = _mm512_fmadd_ps(acc0, scale_vec, cur0);
                cur1 = _mm512_fmadd_ps(acc1, scale_vec, cur1);
                avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)out);
              }
              for (; r < rank_vec16; r += 16) {
                __m512 acc = _mm512_setzero_ps();
                for (int t = 0; t < buf.num_tokens; t++) {
                  float g = buf.grad_out[t * hidden + h];
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const float* u_row = buf.inter_proj.data() + t * rank + r;
                  __m512 u = _mm512_loadu_ps(u_row);
                  acc = _mm512_fmadd_ps(u, g_vec, acc);
                }
                ggml_bf16_t* out = grad_b_base + h * rank + r;
                __m512 cur = load_bf16_16(out, 0xFFFF);
                cur = _mm512_fmadd_ps(acc, scale_vec, cur);
                store_bf16_16(out, cur);
              }
              if (r < rank) {
                __mmask16 mask = static_cast<__mmask16>((1u << (rank - r)) - 1);
                __m512 acc = _mm512_setzero_ps();
                for (int t = 0; t < buf.num_tokens; t++) {
                  float g = buf.grad_out[t * hidden + h];
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const float* u_row = buf.inter_proj.data() + t * rank + r;
                  __m512 u = _mm512_maskz_loadu_ps(mask, u_row);
                  acc = _mm512_fmadd_ps(u, g_vec, acc);
                }
                float acc_vals[16];
                _mm512_storeu_ps(acc_vals, acc);
                ggml_bf16_t* out = grad_b_base + h * rank + r;
                int lanes = rank - r;
                for (int rr = 0; rr < lanes; rr++) {
                  float cur = GGML_BF16_TO_FP32(out[rr]);
                  cur += acc_vals[rr] * scale;
                  out[rr] = GGML_FP32_TO_BF16(cur);
                }
              }
            }
          },
          nullptr, "bwd_down_lora_gradB", 1);

      // grad_A = intermediate^T @ grad_times_b * scaling
      int i_blocks = (inter_size + kGradABlock - 1) / kGradABlock;
      pool->do_work_stealing_job(
          activated_expert * i_blocks, nullptr,
          [&, inter_size, rank, i_blocks](int task_id) {
            int expert_task = task_id / i_blocks;
            int block_idx = task_id % i_blocks;
            LoraGradBuf& buf = lora_grad_bufs[expert_task];
            if (buf.num_tokens == 0) return;

            int i_start = block_idx * kGradABlock;
            int i_end = std::min(inter_size, i_start + kGradABlock);
            int block_len = i_end - i_start;
            if (block_len <= 0) return;

            int block_vec_end = block_len & ~31;
            std::vector<float> inter_accum(block_len);
            __m512 scale_vec = _mm512_set1_ps(lora_scaling_);
            ggml_bf16_t* grad_a_base = grad_down_a + buf.lora_a_offset + i_start;

            for (int r = 0; r < rank; r++) {
              std::fill(inter_accum.begin(), inter_accum.end(), 0.0f);
              for (int t = 0; t < buf.num_tokens; t++) {
                float g = buf.grad_times_b[t * rank + r];
                if (g == 0.0f) {
                  continue;
                }
                __m512 g_vec = _mm512_set1_ps(g);
                const ggml_bf16_t* inter_row = buf.cached_intermediate + t * inter_size + i_start;
                int i = 0;
                for (; i < block_vec_end; i += 32) {
                  __m512 acc0 = _mm512_loadu_ps(inter_accum.data() + i);
                  __m512 acc1 = _mm512_loadu_ps(inter_accum.data() + i + 16);
                  __m512 x0, x1;
                  avx512_32xbf16_to_32xfp32((__m512i*)(inter_row + i), &x0, &x1);
                  acc0 = _mm512_fmadd_ps(x0, g_vec, acc0);
                  acc1 = _mm512_fmadd_ps(x1, g_vec, acc1);
                  _mm512_storeu_ps(inter_accum.data() + i, acc0);
                  _mm512_storeu_ps(inter_accum.data() + i + 16, acc1);
                }
                for (; i < block_len; i++) {
                  float inter = GGML_BF16_TO_FP32(inter_row[i]);
                  inter_accum[i] += inter * g;
                }
              }

              ggml_bf16_t* grad_row = grad_a_base + r * inter_size;
              int i = 0;
              for (; i < block_vec_end; i += 32) {
                __m512 sum0 = _mm512_loadu_ps(inter_accum.data() + i);
                __m512 sum1 = _mm512_loadu_ps(inter_accum.data() + i + 16);
                __m512 cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)(grad_row + i), &cur0, &cur1);
                cur0 = _mm512_fmadd_ps(sum0, scale_vec, cur0);
                cur1 = _mm512_fmadd_ps(sum1, scale_vec, cur1);
                avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)(grad_row + i));
              }
              for (; i < block_len; i++) {
                float cur = GGML_BF16_TO_FP32(grad_row[i]);
                cur += inter_accum[i] * lora_scaling_;
                grad_row[i] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          nullptr, "bwd_down_lora_gradA");
      DOWN_CHECKPOINT("D4_lora_grad");
    }

#undef DOWN_CHECKPOINT
  }

  void backward_activation(const ForwardCache& cache) {
    // Timing for backward_activation
    static int _act_call_count = 0;
    _act_call_count++;
    bool _act_should_print = (_act_call_count == get_print_at_call());
    auto _act_start = std::chrono::high_resolution_clock::now();

    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;

    // // DEBUG: Check cache values for NaN at the beginning
    // {
    //   bool gate_nan = false, up_nan = false;
    //   size_t total_elems = 0;
    //   for (int i = 0; i < activated_expert; i++) {
    //     total_elems += m_local_num_[m_expert_id_map_[i]] * config_.intermediate_size;
    //   }
    //   for (size_t i = 0; i < total_elems && (!gate_nan || !up_nan); i++) {
    //     float g = GGML_BF16_TO_FP32(cache.gate_output_cache[i]);
    //     float u = GGML_BF16_TO_FP32(cache.up_output_cache[i]);
    //     if (std::isnan(g) || std::isinf(g)) gate_nan = true;
    //     if (std::isnan(u) || std::isinf(u)) up_nan = true;
    //   }
    //   if (gate_nan || up_nan) {
    //     printf("[NaN DEBUG L%d] Cache has NaN BEFORE backward_activation: gate=%s, up=%s\n",
    //            config_.layer_idx, gate_nan ? "NaN" : "OK", up_nan ? "NaN" : "OK");
    //   }
    // }

    // SiLU backward:
    // y = silu(gate) * up = gate * sigmoid(gate) * up
    // dy/d(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))) * up
    // dy/d(up) = silu(gate) = gate * sigmoid(gate)

    size_t cache_offset = 0;
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, &cache_offset](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Get cached gate and up outputs (before activation)
          // Need to compute offset into cache
          size_t offset = 0;
          for (int i = 0; i < task_id; i++) {
            offset += m_local_num_[m_expert_id_map_[i]];
          }

          ggml_bf16_t* gate_output = cache.gate_output_cache + offset * config_.intermediate_size;
          ggml_bf16_t* up_output = cache.up_output_cache + offset * config_.intermediate_size;
          ggml_bf16_t* grad_inter = grad_intermediate_ + offset * config_.intermediate_size;
          ggml_bf16_t* grad_gate = grad_gate_output_ + offset * config_.intermediate_size;
          ggml_bf16_t* grad_up = grad_up_output_ + offset * config_.intermediate_size;

          // Debug code commented out - Bug #15 verified fixed
          // if (task_id == 0) {
          //   printf("[DEBUG backward_activation] task_id=0, expert_idx=%d, num_tokens=%d, offset=%zu\n", expert_idx,
          //          num_tokens, offset);
          //   printf("[DEBUG] gate_output[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(gate_output[dbg]));
          //   }
          //   printf("\n");
          //   printf("[DEBUG] up_output[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(up_output[dbg]));
          //   }
          //   printf("\n");
          //   printf("[DEBUG] grad_inter[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(grad_inter[dbg]));
          //   }
          //   printf("\n");
          // }

          for (int i = 0; i < num_tokens * config_.intermediate_size; i++) {
            float g = GGML_BF16_TO_FP32(gate_output[i]);
            float u = GGML_BF16_TO_FP32(up_output[i]);
            float sigmoid_g = 1.0f / (1.0f + expf(-g));
            float silu_g = g * sigmoid_g;

            float grad_i = GGML_BF16_TO_FP32(grad_inter[i]);

            // Compute gradients
            float grad_gate_val = grad_i * u * sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
            float grad_up_val = grad_i * silu_g;

            grad_gate[i] = GGML_FP32_TO_BF16(grad_gate_val);
            grad_up[i] = GGML_FP32_TO_BF16(grad_up_val);
          }
        },
        nullptr, "bwd_act_silu");

    // DUMP: backward activation inputs and outputs
    // Bug #18b fix: offset is accumulated as elements (tokens * intermediate_size),
    // so don't multiply by intermediate_size again when accessing cache
    if (is_dump_enabled()) {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          // Dump cached gate_output and up_output used in activation backward
          // Note: offset is already in elements, no need to multiply by intermediate_size
          ggml_bf16_t* gate_out_cached = cache.gate_output_cache + offset;
          ggml_bf16_t* up_out_cached = cache.up_output_cache + offset;
          dump_bf16_matrix(gate_out_cached, m, config_.intermediate_size, "backward_act_gate_cache", tp_part_idx,
                           expert_idx);
          dump_bf16_matrix(up_out_cached, m, config_.intermediate_size, "backward_act_up_cache", tp_part_idx,
                           expert_idx);
          // Dump grad_intermediate (input to activation backward)
          dump_bf16_matrix(grad_intermediate_ + offset, m, config_.intermediate_size, "backward_grad_intermediate",
                           tp_part_idx, expert_idx);
          // Dump grad_gate_out
          dump_bf16_matrix(grad_gate_output_ + offset, m, config_.intermediate_size, "backward_grad_gate_out",
                           tp_part_idx, expert_idx);
          // Dump grad_up_out
          dump_bf16_matrix(grad_up_output_ + offset, m, config_.intermediate_size, "backward_grad_up_out", tp_part_idx,
                           expert_idx);
        }
        offset += m * config_.intermediate_size;
      }
    }

    if (_act_should_print) {
      auto _act_end = std::chrono::high_resolution_clock::now();
      printf("  [ACT] silu_backward: %.3f ms\n",
             std::chrono::duration<double, std::milli>(_act_end - _act_start).count());
    }
  }

  void backward_gate_up(const ForwardCache& cache, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                        void* grad_up_lora_a, void* grad_up_lora_b) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
    ggml_bf16_t* grad_gate_b = (ggml_bf16_t*)grad_gate_lora_b;
    ggml_bf16_t* grad_up_a = (ggml_bf16_t*)grad_up_lora_a;
    ggml_bf16_t* grad_up_b = (ggml_bf16_t*)grad_up_lora_b;

    // Initialize grad_input to zero (bf16)
    memset(grad_input, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,
        [this, &cache, grad_input, grad_gate_a, grad_gate_b, grad_up_a, grad_up_b, qlen, k](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Compute offset into gradient buffers
          size_t offset = 0;
          for (int i = 0; i < task_id / 2; i++) {
            offset += m_local_num_[m_expert_id_map_[i]];
          }

          ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                    : (grad_gate_output_ + offset * config_.intermediate_size);
          ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          ggml_bf16_t* lora_b = do_up ? up_lora_b_ : gate_lora_b_;
          ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
          ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;

          // First, compute base weight contribution to grad_input (always, regardless of LoRA)
          // grad_input += grad @ W^T (for gate or up, depending on do_up)
          // W layout: [expert_num, intermediate_size, hidden_size]
          // grad: [num_tokens, intermediate_size]
          // grad_input[t, h] += sum_i grad[t, i] * W[i, h]
          {
            ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
            const ggml_bf16_t* base_proj =
                do_up ? (const ggml_bf16_t*)config_.up_proj : (const ggml_bf16_t*)config_.gate_proj;
            size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

            // Pre-compute grad_input contribution per token, then scatter
            std::vector<float> token_grad_input(num_tokens * config_.hidden_size, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  float w = GGML_BF16_TO_FP32(base_proj[expert_offset + i * config_.hidden_size + h]);
                  sum += g * w;
                }
                token_grad_input[t * config_.hidden_size + h] = sum;
              }
            }

            // Scatter back to grad_input
            for (int i = 0; i < qlen; i++) {
              for (int j = 0; j < k; j++) {
                if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                  int pos = cache.m_local_pos_cache[i][j];
                  for (int h = 0; h < config_.hidden_size; h++) {
                    size_t idx = (size_t)i * config_.hidden_size + h;
                    float cur = GGML_BF16_TO_FP32(grad_input_bf16[idx]);
                    cur += token_grad_input[pos * config_.hidden_size + h];
                    grad_input_bf16[idx] = GGML_FP32_TO_BF16(cur);
                  }
                }
              }
            }
          }

          // Skip all LoRA computation when SkipLoRA is true
          // (only base weight contribution to grad_input is computed above)
          if (SkipLoRA || lora_a == nullptr || lora_b == nullptr) return;

          // Get cached input
          ggml_bf16_t* input = cache.input_cache;

          // Get expert's LoRA weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
          size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          ggml_bf16_t* expert_lora_b = lora_b + lora_b_offset;

          // Collect expert's input from cached input using routing
          std::vector<float> expert_input(num_tokens * config_.hidden_size, 0.0f);
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  expert_input[pos * config_.hidden_size + h] = GGML_BF16_TO_FP32(input[i * config_.hidden_size + h]);
                }
              }
            }
          }

          // Gradient for LoRA B: grad_B = grad^T @ (input @ lora_A^T) * scaling
          // First compute input @ lora_A^T → [num_tokens, lora_rank]
          std::vector<float> input_proj(num_tokens * lora_rank_, 0.0f);
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int h = 0; h < config_.hidden_size; h++) {
                float inp = expert_input[t * config_.hidden_size + h];
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                sum += inp * w;
              }
              input_proj[t * lora_rank_ + r] = sum;
            }
          }

          // grad_B = grad^T @ input_proj * scaling
          for (int i = 0; i < config_.intermediate_size; i++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int t = 0; t < num_tokens; t++) {
                float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                sum += g * input_proj[t * lora_rank_ + r];
              }
              size_t idx = lora_b_offset + i * lora_rank_ + r;
              float cur = GGML_BF16_TO_FP32(grad_lora_b[idx]);
              cur += sum * lora_scaling_;
              grad_lora_b[idx] = GGML_FP32_TO_BF16(cur);
            }
          }

          // Gradient for LoRA A
          // First: grad @ lora_B → [num_tokens, lora_rank]
          std::vector<float> grad_times_b(num_tokens * lora_rank_, 0.0f);
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int i = 0; i < config_.intermediate_size; i++) {
                float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                float b = GGML_BF16_TO_FP32(expert_lora_b[i * lora_rank_ + r]);
                sum += g * b;
              }
              grad_times_b[t * lora_rank_ + r] = sum;
            }
          }

          // grad_A = input^T @ grad_times_b * scaling
          for (int r = 0; r < lora_rank_; r++) {
            for (int h = 0; h < config_.hidden_size; h++) {
              float sum = 0.0f;
              for (int t = 0; t < num_tokens; t++) {
                sum += expert_input[t * config_.hidden_size + h] * grad_times_b[t * lora_rank_ + r];
              }
              size_t idx = lora_a_offset + r * config_.hidden_size + h;
              float cur = GGML_BF16_TO_FP32(grad_lora_a[idx]);
              cur += sum * lora_scaling_;
              grad_lora_a[idx] = GGML_FP32_TO_BF16(cur);
            }
          }

          // Compute grad_input contribution from this expert's LoRA
          // grad_input += grad @ lora_B @ lora_A * scaling
          // = grad_times_b @ lora_A * scaling
          // Need to scatter back to original positions
          ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  float sum = 0.0f;
                  for (int r = 0; r < lora_rank_; r++) {
                    sum += grad_times_b[pos * lora_rank_ + r] *
                           GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                  }
                  size_t idx = (size_t)i * config_.hidden_size + h;
                  float cur = GGML_BF16_TO_FP32(grad_input_bf16[idx]);
                  cur += sum * lora_scaling_;
                  grad_input_bf16[idx] = GGML_FP32_TO_BF16(cur);
                }
              }
            }
          }
        },
        nullptr, "bwd_gate_up");
  }

  /**
   * @brief AMX-optimized backward pass for gate and up projections.
   *
   * Uses AMX GEMM for base weight contribution and LoRA grad_input. LoRA weight gradients
   * remain small for-loops.
   */
  void backward_gate_up_amx(const ForwardCache& cache, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                            void* grad_up_lora_a, void* grad_up_lora_b) {
    // Timing for backward_gate_up_amx substeps (counter-based)
    static int _gu_call_count = 0;
    _gu_call_count++;
    bool _gu_should_print = (_gu_call_count == get_print_at_call());
    auto _gu_start = std::chrono::high_resolution_clock::now();
    auto _gu_last = _gu_start;

#define GU_CHECKPOINT(name)                                                                                     \
  do {                                                                                                          \
    auto _now = std::chrono::high_resolution_clock::now();                                                      \
    if (_gu_should_print) {                                                                                     \
      printf("  [GU] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _gu_last).count()); \
    }                                                                                                           \
    _gu_last = _now;                                                                                            \
  } while (0)

    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
    ggml_bf16_t* grad_gate_b = (ggml_bf16_t*)grad_gate_lora_b;
    ggml_bf16_t* grad_up_a = (ggml_bf16_t*)grad_up_lora_a;
    ggml_bf16_t* grad_up_b = (ggml_bf16_t*)grad_up_lora_b;

    prepare_backward_weights();
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      prepare_lora_weights();
    }

    // =====================================================
    // Bug-C Fix Step 2: Allocate backward buffers from shared pool
    // Note: backward_down_amx already allocated grad_output_ba_ and grad_intermediate_bc_
    // Here we need grad_gate_up_bc_ which uses the remaining part of backward_bc_pool_
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Calculate offset after grad_intermediate_bc_ allocations
    size_t grad_intermediate_total = 0;
    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      size_t local_max_m = ((m_local_num_[expert_idx] + M_STEP - 1) / M_STEP) * M_STEP;
      grad_intermediate_total += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));
    }

    char* grad_gate_up_bc_ptr = (char*)backward_bc_pool_ + grad_intermediate_total;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferC for grad_gate_up
      grad_gate_up_bc_[expert_idx]->max_m = local_max_m;
      grad_gate_up_bc_[expert_idx]->set_data(grad_gate_up_bc_ptr);
      grad_gate_up_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));
    }

    // Allocate LoRA intermediate buffers from shared pools (for LoRA backward pass)
    char* lora_ba_ptr = (char*)lora_ba_pool_;
    char* lora_bc_inter_ptr = (char*)lora_bc_inter_pool_;
    char* bf16_inter_ptr = (char*)lora_intermediate_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // BufferA for LoRA intermediate (gate)
      lora_gate_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // BufferA for LoRA intermediate (up)
      lora_up_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // BufferC for LoRA step 1 output (gate)
      lora_gate_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // BufferC for LoRA step 1 output (up)
      lora_up_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // BF16 intermediate pointers (gate)
      lora_gate_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));

      // BF16 intermediate pointers (up)
      lora_up_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));
    }

    memset(grad_input, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

    GU_CHECKPOINT("GU0_prepare+memset");

    // Offsets into contiguous grad_gate/up buffers
    std::vector<size_t> expert_offsets(activated_expert);
    {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        expert_offsets[i] = offset;
        offset += m_local_num_[m_expert_id_map_[i]];
      }
    }

    bool dump_enabled = is_dump_enabled();

    // Accumulation buffers for per-expert grad_input dump (only when dump is enabled)
    // Maps expert_idx -> FP32 accumulation buffer [m_local_num x hidden_size]
    std::unordered_map<int, std::vector<float>> expert_grad_accum;
    if (dump_enabled) {
      for (int task_id = 0; task_id < activated_expert; task_id++) {
        int expert_idx = m_expert_id_map_[task_id];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          expert_grad_accum[expert_idx].resize(m * config_.hidden_size, 0.0f);
        }
      }
    }

    auto scatter_to_grad_input = [&](float scale, const char* task_name) {
      ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
      pool->do_work_stealing_job(
          qlen, nullptr,
          [&, scale](int token_id) {
            ggml_bf16_t* dst = grad_input_bf16 + token_id * config_.hidden_size;
            for (int j = 0; j < k; j++) {
              int expert_idx = cache.expert_ids_cache[token_id * k + j];
              if (expert_idx < config_.num_gpu_experts || expert_idx >= config_.expert_num) {
                continue;
              }
              if (m_local_num_[expert_idx] == 0) {
                continue;
              }
              int pos = cache.m_local_pos_cache[token_id][j];
              ggml_bf16_t* contrib = grad_output_bf16_ptr_[expert_idx] + pos * config_.hidden_size;

              // Accumulate per-expert grad_input for dumps (no routing weights)
              if (dump_enabled) {
                auto it = expert_grad_accum.find(expert_idx);
                if (it != expert_grad_accum.end()) {
                  float* accum = it->second.data() + pos * config_.hidden_size;
                  for (int h = 0; h < config_.hidden_size; h++) {
                    accum[h] += GGML_BF16_TO_FP32(contrib[h]) * scale;
                  }
                }
              }

              for (int h = 0; h < config_.hidden_size; h++) {
                float add = GGML_BF16_TO_FP32(contrib[h]) * scale;
                float cur = GGML_BF16_TO_FP32(dst[h]);
                cur += add;
                dst[h] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          nullptr, task_name);
    };

    auto base_pass = [&](bool do_up) {
      auto _bp_start = std::chrono::high_resolution_clock::now();
      const char* quant_name = do_up ? "bwd_gu_base_q_up" : "bwd_gu_base_q_gate";
      const char* gemm_name = do_up ? "bwd_gu_base_gemm_up" : "bwd_gu_base_gemm_gate";

      // Quantize grad to BufferA
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            down_ba_[expert_idx]->from_mat(m, grad, 0, 1);
          },
          nullptr, quant_name);

      int nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth](int task_id) {
            int task_idx = task_id / nth;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = down_ba_[expert_idx];
            auto& bb = do_up ? up_backward_bb_[expert_idx] : gate_backward_bb_[expert_idx];
            auto& bc = grad_gate_up_bc_[expert_idx];

            amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
            bc->to_mat(m, grad_output_bf16_ptr_[expert_idx], ith, nth);
          },
          nullptr, gemm_name, 1);

      // DUMP: base backward output before scatter
      if (is_dump_enabled()) {
        for (int i = 0; i < activated_expert; i++) {
          int expert_idx = m_expert_id_map_[i];
          int m = m_local_num_[expert_idx];
          if (m > 0) {
            const char* name = do_up ? "backward_up_base" : "backward_gate_base";
            dump_bf16_matrix(grad_output_bf16_ptr_[expert_idx], m, config_.hidden_size, name, tp_part_idx, expert_idx);
          }
        }
      }

      scatter_to_grad_input(1.0f, "bwd_gu_scatter_base");

      if (_gu_should_print) {
        auto _bp_end = std::chrono::high_resolution_clock::now();
        printf("  [GU] base_pass(%s): %.3f ms\n", do_up ? "up" : "gate",
               std::chrono::duration<double, std::milli>(_bp_end - _bp_start).count());
      }
    };

    base_pass(false);  // gate
    base_pass(true);   // up
    GU_CHECKPOINT("GU1_base_passes_total");

    // // DEBUG: Check m_local_input_ptr_ AFTER base_pass (before LoRA)
    // {
    //   bool has_nan = false, has_large = false;
    //   float max_val = 0.0f;
    //   for (int task_id = 0; task_id < activated_expert && !has_nan; task_id++) {
    //     int expert_idx = m_expert_id_map_[task_id];
    //     int m = m_local_num_[expert_idx];
    //     if (m == 0) continue;
    //     ggml_bf16_t* input_ptr = m_local_input_ptr_[expert_idx];
    //     for (int i = 0; i < m * config_.hidden_size && !has_nan; i++) {
    //       float v = GGML_BF16_TO_FP32(input_ptr[i]);
    //       if (std::isnan(v) || std::isinf(v)) has_nan = true;
    //       float av = std::abs(v);
    //       if (av > max_val) max_val = av;
    //       if (av > 1e10f) has_large = true;
    //     }
    //   }
    //   if (has_nan || has_large) {
    //     printf("[NaN DEBUG L%d] m_local_input AFTER base_pass: has_nan=%d has_large=%d max=%.6e\n",
    //            config_.layer_idx, has_nan, has_large, max_val);
    //   }
    // }

    // Skip all LoRA computation when SkipLoRA is true
    if (SkipLoRA || gate_lora_a_ == nullptr || gate_lora_b_ == nullptr) {
#undef GU_CHECKPOINT
      return;
    }

// Re-define GU_CHECKPOINT and GU_PROFILE_OP for the rest of the function
#define GU_CHECKPOINT(name)                                                                                     \
  do {                                                                                                          \
    auto _now = std::chrono::high_resolution_clock::now();                                                      \
    if (_gu_should_print) {                                                                                     \
      printf("  [GU] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _gu_last).count()); \
    }                                                                                                           \
    _gu_last = _now;                                                                                            \
  } while (0)

    // Re-quantize inputs for LoRA path
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          gate_up_ba_[expert_idx]->from_mat(m, m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr, "bwd_gu_lora_requant");

    GU_CHECKPOINT("GU2_requantize_for_lora");

    auto lora_pass = [&](bool do_up) {
      auto _lp_start = std::chrono::high_resolution_clock::now();
      const char* u_name = do_up ? "bwd_gu_lora_u_up" : "bwd_gu_lora_u_gate";
      const char* gradb_name = do_up ? "bwd_gu_lora_gradb_up" : "bwd_gu_lora_gradb_gate";
      const char* gb_quant_name = do_up ? "bwd_gu_lora_gb_quant_up" : "bwd_gu_lora_gb_quant_gate";
      const char* gb_gemm_name = do_up ? "bwd_gu_lora_gb_gemm_up" : "bwd_gu_lora_gb_gemm_gate";
      const char* gb_out_quant_name = do_up ? "bwd_gu_lora_gb_out_q_up" : "bwd_gu_lora_gb_out_q_gate";
      const char* grad_in_name = do_up ? "bwd_gu_lora_gradin_up" : "bwd_gu_lora_gradin_gate";
      const char* grad_a_name = do_up ? "bwd_gu_lora_gradA_up" : "bwd_gu_lora_gradA_gate";

      // // DEBUG: Check input values BEFORE Step 1
      // {
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   for (int task_id = 0; task_id < activated_expert && !has_nan; task_id++) {
      //     int expert_idx = m_expert_id_map_[task_id];
      //     int m = m_local_num_[expert_idx];
      //     if (m == 0) continue;
      //     ggml_bf16_t* input_ptr = m_local_input_ptr_[expert_idx];
      //     for (int i = 0; i < m * config_.hidden_size && !has_nan; i++) {
      //       float v = GGML_BF16_TO_FP32(input_ptr[i]);
      //       if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //       float av = std::abs(v);
      //       if (av > max_val) max_val = av;
      //       if (av > 1e10f) has_large = true;
      //     }
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s input BEFORE Step1: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // }

      // // DEBUG: Check lora_A weights BEFORE Step 1
      // {
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
      //   size_t total = config_.expert_num * lora_rank_ * config_.hidden_size;
      //   for (size_t i = 0; i < total && !has_nan; i++) {
      //     float v = GGML_BF16_TO_FP32(lora_a[i]);
      //     if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //     float av = std::abs(v);
      //     if (av > max_val) max_val = av;
      //     if (av > 1e10f) has_large = true;
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s lora_A BEFORE Step1: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // }

      // Step 1: input @ lora_A^T -> U
      int nth_inter = T::recommended_nth(padded_lora_rank_);
      pool->do_work_stealing_job(
          nth_inter * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_inter](int task_id) {
            int task_idx = task_id / nth_inter;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_inter;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = gate_up_ba_[expert_idx];
            auto& bb = do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx];
            auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];
            ggml_bf16_t* inter_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, ba, bb, bc, ith, nth_inter);
            bc->to_mat(m, inter_ptr, ith, nth_inter);
          },
          nullptr, u_name);

      // // DEBUG: Check U values after Step 1
      // {
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   for (int task_id = 0; task_id < activated_expert && !has_nan; task_id++) {
      //     int expert_idx = m_expert_id_map_[task_id];
      //     int m = m_local_num_[expert_idx];
      //     if (m == 0) continue;
      //     ggml_bf16_t* u_ptr = do_up ? lora_up_intermediate_ptr_[expert_idx] :
      //     lora_gate_intermediate_ptr_[expert_idx]; for (int i = 0; i < m * padded_lora_rank_ && !has_nan; i++) {
      //       float v = GGML_BF16_TO_FP32(u_ptr[i]);
      //       if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //       float av = std::abs(v);
      //       if (av > max_val) max_val = av;
      //       if (av > 1e10f) has_large = true;
      //     }
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s U after Step1: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // // }

      // // DEBUG: Check grad values before Step 2
      // {
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   size_t total = 0;
      //   for (int i = 0; i < activated_expert; i++) {
      //     total += m_local_num_[m_expert_id_map_[i]];
      //   }
      //   total *= config_.intermediate_size;
      //   ggml_bf16_t* grad = do_up ? grad_up_output_ : grad_gate_output_;
      //   for (size_t i = 0; i < total && !has_nan; i++) {
      //     float v = GGML_BF16_TO_FP32(grad[i]);
      //     if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //     float av = std::abs(v);
      //     if (av > max_val) max_val = av;
      //     if (av > 1e10f) has_large = true;
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s grad before Step2: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // }

      // Step 2: grad_B = grad^T @ U (AVX512 across lora_rank)
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;
            size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
            ggml_bf16_t* u_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            auto load_bf16_16 = [](const ggml_bf16_t* src, __mmask16 mask) -> __m512 {
              __m256i bf16 = _mm256_maskz_loadu_epi16(mask, src);
              __m512i i32 = _mm512_cvtepu16_epi32(bf16);
              return _mm512_castsi512_ps(_mm512_slli_epi32(i32, 16));
            };
            auto store_bf16_16 = [](ggml_bf16_t* dst, __m512 v) {
              __m256i bf16 = (__m256i)_mm512_cvtneps_pbh(v);
              _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), bf16);
            };

            const int rank_vec32 = lora_rank_ & ~31;
            const int rank_vec16 = lora_rank_ & ~15;
            const float scale = lora_scaling_;
            const __m512 scale_vec = _mm512_set1_ps(scale);

            for (int i = 0; i < config_.intermediate_size; i++) {
              int r = 0;
              for (; r < rank_vec32; r += 32) {
                __m512 acc0 = _mm512_setzero_ps();
                __m512 acc1 = _mm512_setzero_ps();
                for (int t = 0; t < num_tokens; t++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const ggml_bf16_t* u_row = u_ptr + t * padded_lora_rank_ + r;
                  __m512 u0 = load_bf16_16(u_row, 0xFFFF);
                  __m512 u1 = load_bf16_16(u_row + 16, 0xFFFF);
                  acc0 = _mm512_fmadd_ps(u0, g_vec, acc0);
                  acc1 = _mm512_fmadd_ps(u1, g_vec, acc1);
                }

                ggml_bf16_t* out = grad_lora_b + lora_b_offset + i * lora_rank_ + r;
                __m512 cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)out, &cur0, &cur1);
                cur0 = _mm512_fmadd_ps(acc0, scale_vec, cur0);
                cur1 = _mm512_fmadd_ps(acc1, scale_vec, cur1);
                avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)out);
              }

              for (; r < rank_vec16; r += 16) {
                __m512 acc = _mm512_setzero_ps();
                for (int t = 0; t < num_tokens; t++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const ggml_bf16_t* u_row = u_ptr + t * padded_lora_rank_ + r;
                  __m512 u = load_bf16_16(u_row, 0xFFFF);
                  acc = _mm512_fmadd_ps(u, g_vec, acc);
                }

                ggml_bf16_t* out = grad_lora_b + lora_b_offset + i * lora_rank_ + r;
                __m512 cur = load_bf16_16(out, 0xFFFF);
                cur = _mm512_fmadd_ps(acc, scale_vec, cur);
                store_bf16_16(out, cur);
              }

              if (r < lora_rank_) {
                __mmask16 mask = static_cast<__mmask16>((1u << (lora_rank_ - r)) - 1);
                __m512 acc = _mm512_setzero_ps();
                for (int t = 0; t < num_tokens; t++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  if (g == 0.0f) {
                    continue;
                  }
                  __m512 g_vec = _mm512_set1_ps(g);
                  const ggml_bf16_t* u_row = u_ptr + t * padded_lora_rank_ + r;
                  __m512 u = load_bf16_16(u_row, mask);
                  acc = _mm512_fmadd_ps(u, g_vec, acc);
                }

                float acc_vals[16];
                _mm512_storeu_ps(acc_vals, acc);
                ggml_bf16_t* out = grad_lora_b + lora_b_offset + i * lora_rank_ + r;
                int lanes = lora_rank_ - r;
                for (int rr = 0; rr < lanes; rr++) {
                  float cur = GGML_BF16_TO_FP32(out[rr]);
                  cur += acc_vals[rr] * scale;
                  out[rr] = GGML_FP32_TO_BF16(cur);
                }
              }
            }
          },
          nullptr, gradb_name);

      // DEBUG: Check grad_B values after Step 2
      // {
      //   ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   size_t total = config_.expert_num * config_.intermediate_size * lora_rank_;
      //   for (size_t i = 0; i < total && !has_nan; i++) {
      //     float v = GGML_BF16_TO_FP32(grad_lora_b[i]);
      //     if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //     float av = std::abs(v);
      //     if (av > max_val) max_val = av;
      //     if (av > 1e10f) has_large = true;
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s grad_B after Step2: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // }

      // Step 3: grad @ lora_B -> G_B
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            down_ba_[expert_idx]->from_mat(m, grad, 0, 1);
          },
          nullptr, gb_quant_name);

      int nth_gb = T::recommended_nth(padded_lora_rank_);
      pool->do_work_stealing_job(
          nth_gb * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_gb](int task_id) {
            int task_idx = task_id / nth_gb;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_gb;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = down_ba_[expert_idx];
            auto& bb = do_up ? up_lora_b_t_bb_[expert_idx] : gate_lora_b_t_bb_[expert_idx];
            auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];
            ggml_bf16_t* inter_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            amx::mat_mul(m, padded_lora_rank_, config_.intermediate_size, ba, bb, bc, ith, nth_gb);
            bc->to_mat(m, inter_ptr, ith, nth_gb);
          },
          nullptr, gb_gemm_name);

      // Step 4: Quantize G_B
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            ggml_bf16_t* g_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
            ba->from_mat(m, g_ptr, 0, 1);
          },
          nullptr, gb_out_quant_name);

      // Step 5: G_B @ lora_A -> grad_input (LoRA part)
      int nth_input = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth_input * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_input](int task_id) {
            int task_idx = task_id / nth_input;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_input;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
            auto& bb = do_up ? up_lora_a_t_bb_[expert_idx] : gate_lora_a_t_bb_[expert_idx];
            auto& bc = grad_gate_up_bc_[expert_idx];

            amx::mat_mul(m, config_.hidden_size, padded_lora_rank_, ba, bb, bc, ith, nth_input);
            bc->to_mat(m, grad_output_bf16_ptr_[expert_idx], ith, nth_input);
          },
          nullptr, grad_in_name);

      // DUMP: LoRA contribution before scatter (will be scaled by lora_scaling_)
      if (is_dump_enabled()) {
        for (int i = 0; i < activated_expert; i++) {
          int expert_idx = m_expert_id_map_[i];
          int m = m_local_num_[expert_idx];
          if (m > 0) {
            // Dump LoRA intermediate (grad @ lora_B result)
            ggml_bf16_t* inter_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            const char* inter_name = do_up ? "backward_up_lora_inter" : "backward_gate_lora_inter";
            dump_bf16_matrix(inter_ptr, m, padded_lora_rank_, inter_name, tp_part_idx, expert_idx);

            // Dump LoRA output (G_B @ lora_A result, WITH lora_scaling applied for comparison)
            const char* lora_name = do_up ? "backward_up_lora" : "backward_gate_lora";
            dump_bf16_matrix_scaled(grad_output_bf16_ptr_[expert_idx], m, config_.hidden_size, lora_scaling_, lora_name,
                                    tp_part_idx, expert_idx);
          }
        }
      }

      scatter_to_grad_input(lora_scaling_, "bwd_gu_scatter_lora");

      // Step 6: grad_A = G_B^T @ X (AVX512 accumulation across hidden_size)
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t* g_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
            ggml_bf16_t* expert_input = m_local_input_ptr_[expert_idx];

            const int hidden = config_.hidden_size;
            constexpr int kVecWidth = 32;
            int vec_end = hidden & ~(kVecWidth - 1);
            __m512 scale_vec = _mm512_set1_ps(lora_scaling_);
            std::vector<float> accum(hidden);

            for (int r = 0; r < lora_rank_; r++) {
              std::fill(accum.begin(), accum.end(), 0.0f);

              for (int t = 0; t < num_tokens; t++) {
                float gb = GGML_BF16_TO_FP32(g_ptr[t * padded_lora_rank_ + r]);
                if (gb == 0.0f) {
                  continue;
                }
                __m512 gb_vec = _mm512_set1_ps(gb);
                const ggml_bf16_t* input_row = expert_input + t * hidden;

                int h = 0;
                for (; h < vec_end; h += kVecWidth) {
                  __m512 acc0 = _mm512_loadu_ps(accum.data() + h);
                  __m512 acc1 = _mm512_loadu_ps(accum.data() + h + 16);
                  __m512 x0, x1;
                  avx512_32xbf16_to_32xfp32((__m512i*)(input_row + h), &x0, &x1);
                  acc0 = _mm512_fmadd_ps(x0, gb_vec, acc0);
                  acc1 = _mm512_fmadd_ps(x1, gb_vec, acc1);
                  _mm512_storeu_ps(accum.data() + h, acc0);
                  _mm512_storeu_ps(accum.data() + h + 16, acc1);
                }
                for (; h < hidden; h++) {
                  float inp = GGML_BF16_TO_FP32(input_row[h]);
                  accum[h] += inp * gb;
                }
              }

              ggml_bf16_t* grad_row = grad_lora_a + lora_a_offset + r * hidden;
              int h = 0;
              for (; h < vec_end; h += kVecWidth) {
                __m512 sum0 = _mm512_loadu_ps(accum.data() + h);
                __m512 sum1 = _mm512_loadu_ps(accum.data() + h + 16);
                __m512 cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)(grad_row + h), &cur0, &cur1);
                cur0 = _mm512_fmadd_ps(sum0, scale_vec, cur0);
                cur1 = _mm512_fmadd_ps(sum1, scale_vec, cur1);
                avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)(grad_row + h));
              }
              for (; h < hidden; h++) {
                float cur = GGML_BF16_TO_FP32(grad_row[h]);
                cur += accum[h] * lora_scaling_;
                grad_row[h] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          nullptr, grad_a_name);

      // // DEBUG: Check grad_A values after Step 6
      // {
      //   ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
      //   bool has_nan = false, has_large = false;
      //   float max_val = 0.0f;
      //   size_t total = config_.expert_num * lora_rank_ * config_.hidden_size;
      //   for (size_t i = 0; i < total && !has_nan; i++) {
      //     float v = GGML_BF16_TO_FP32(grad_lora_a[i]);
      //     if (std::isnan(v) || std::isinf(v)) has_nan = true;
      //     float av = std::abs(v);
      //     if (av > max_val) max_val = av;
      //     if (av > 1e10f) has_large = true;
      //   }
      //   if (has_nan || has_large) {
      //     printf("[NaN DEBUG L%d] %s grad_A after Step6: has_nan=%d has_large=%d max=%.6e\n",
      //            config_.layer_idx, do_up ? "up" : "gate", has_nan, has_large, max_val);
      //   }
      // }

      if (_gu_should_print) {
        auto _lp_end = std::chrono::high_resolution_clock::now();
        printf("  [GU] lora_pass(%s): %.3f ms\n", do_up ? "up" : "gate",
               std::chrono::duration<double, std::milli>(_lp_end - _lp_start).count());
      }
    };

    lora_pass(false);  // gate
    lora_pass(true);   // up
    GU_CHECKPOINT("GU3_lora_passes_total");

    // DUMP: backward grad_input per expert (accumulated sum of gate_base + gate_lora + up_base + up_lora)
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          auto it = expert_grad_accum.find(expert_idx);
          if (it != expert_grad_accum.end()) {
            // Dump accumulated per-expert grad_input (sum of all 4 contributions)
            dump_fp32_matrix(it->second.data(), m, config_.hidden_size, "backward_grad_input_expert", tp_part_idx,
                             expert_idx);
          }
        }
      }
    }

#undef GU_CHECKPOINT
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_MOE_H
