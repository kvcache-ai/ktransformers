/**
 * @Description  : AMX MoE SFT (Supervised Fine-Tuning) implementation with LoRA support.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_SFT_MOE_H
#define CPUINFER_OPERATOR_AMX_SFT_MOE_H

// Bump on every change to this file. Printed at construction time.
static constexpr int kSftMoeVersion = 5;

// =====================================================
// AVX vectorized BF16 LoRA grad write-back helpers
// =====================================================

// Vectorized BF16 read-modify-write for rank=8:
//   out[i*rank..i*rank+7] += acc[i*rank..i*rank+7] * scale
// Processes `count` rows of `rank` BF16 values each.
static inline void bf16_fma_writeback_rank8(ggml_bf16_t* out, const float* acc,
                                            int count, int rank, float scale) {
  (void)rank;  // always 8
  __m256 scale_vec = _mm256_set1_ps(scale);
  for (int i = 0; i < count; i++) {
    ggml_bf16_t* row_out = out + static_cast<size_t>(i) * 8;
    const float* row_acc = acc + static_cast<size_t>(i) * 8;
    // BF16 → FP32: zero-extend 16→32 bit, shift left 16
    __m128i bf16_in = _mm_loadu_si128((const __m128i*)row_out);
    __m256 fp32_cur = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_in), 16));
    // FMA: cur += acc * scale
    __m256 acc_val = _mm256_loadu_ps(row_acc);
    fp32_cur = _mm256_fmadd_ps(acc_val, scale_vec, fp32_cur);
    // FP32 → BF16 (AVX512_BF16)
    __m128i bf16_out = (__m128i)_mm256_cvtneps_pbh(fp32_cur);
    _mm_storeu_si128((__m128i*)row_out, bf16_out);
  }
}

// Vectorized FP32 scale-write for rank=8:
//   out[i*rank..i*rank+7] = acc[i*rank..i*rank+7] * scale
static inline void fp32_scale_write_rank8(float* out, const float* acc,
                                          int count, int rank, float scale) {
  (void)rank;
  __m256 scale_vec = _mm256_set1_ps(scale);
  for (int i = 0; i < count; i++) {
    float* row_out = out + static_cast<size_t>(i) * 8;
    const float* row_acc = acc + static_cast<size_t>(i) * 8;
    __m256 val = _mm256_mul_ps(_mm256_loadu_ps(row_acc), scale_vec);
    _mm256_storeu_ps(row_out, val);
  }
}

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../../cpu_backend/worker_pool.h"
#include "ggml.h"
#include "la/amx_kernels.hpp"
#include "la/avx_kernels.hpp"
#include "moe.hpp"

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
    double dv = static_cast<double>(v);
    double a = std::fabs(dv);
    sum_abs += a;
    sum_sq += dv * dv;
    if (a > max_abs || std::isnan(a)) {
      max_abs = a;
    }
  }
  stats.abs_mean = sum_abs / static_cast<double>(size);
  stats.abs_max = max_abs;
  stats.norm = std::sqrt(sum_sq);
  return stats;
}

// ANSI color codes for terminal output
#define ANSI_COLOR_RED "\033[1;31m"
#define ANSI_COLOR_YELLOW "\033[1;33m"
#define ANSI_COLOR_GREEN "\033[1;32m"
#define ANSI_COLOR_RESET "\033[0m"
#define ANSI_BG_YELLOW "\033[43m"
#define ANSI_BG_RED "\033[41m"
#define ANSI_BG_BLUE "\033[44m"

// Robust NaN/Inf check (v != v is true only for NaN)
inline bool is_nan_value(float v) { return v != v; }
inline bool is_inf_value(float v) {
  return !is_nan_value(v) &&
         (v == std::numeric_limits<float>::infinity() || v == -std::numeric_limits<float>::infinity());
}

// Threshold for "large value" warning (yellow)
constexpr double NAN_CHECK_LARGE_THRESHOLD = 1e4;

// Check BF16 buffer for NaN/Inf (using robust v != v check)
inline NaNCheckResult check_bf16_buffer_for_nan(const ggml_bf16_t* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    float val = GGML_BF16_TO_FP32(buf[i]);
    // Use val != val for robust NaN detection
    if (val != val) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
        result.first_nan_input_val = val;
      }
    }
    if (!(val != val) && is_inf_value(val)) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf(ANSI_COLOR_RED "[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d" ANSI_COLOR_RESET "\n", label,
           result.nan_count, result.inf_count, result.first_nan_idx);
  }
  return result;
}

// Check FP32 buffer for NaN/Inf (using robust v != v check)
inline NaNCheckResult check_fp32_buffer_for_nan(const float* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    float val = buf[i];
    // Use val != val for robust NaN detection
    if (val != val) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
        result.first_nan_input_val = val;
      }
    }
    if (!(val != val) && is_inf_value(val)) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf(ANSI_COLOR_RED "[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d" ANSI_COLOR_RESET "\n", label,
           result.nan_count, result.inf_count, result.first_nan_idx);
  }
  return result;
}

// Check if NaN checking is enabled via environment variable
inline bool is_nan_check_enabled() {
  return false;
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = getenv("SFT_MOE_NAN_CHECK");
    enabled = (env && env[0] != '0') ? 1 : 0;
  }
  return enabled == 1;
}

// =====================================================
// Dump Utility Functions for debugging
// Controlled by SFT_MOE_DUMP environment variable
// =====================================================
inline bool is_dump_enabled() {
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
// Pool Memory Logger — writes per-call alloc/free events to file
// Enable: set SFT_POOL_LOG=1 (or any non-zero)
// Output: sft_pool_log.txt in current directory (append mode)
// Disable: return false; at the top of is_pool_log_enabled()
// =====================================================
inline bool is_pool_log_enabled() {
  // return false;
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = getenv("SFT_POOL_LOG");
    enabled = (env && env[0] != '0') ? 1 : 0;
  }
  return enabled == 1;
}

inline FILE* get_pool_log_file() {
  static FILE* f = nullptr;
  if (f == nullptr) {
    const char* path = getenv("SFT_POOL_LOG_FILE");
    if (!path) path = "sft_pool_log.txt";
    f = fopen(path, "a");
    if (f) {
      fprintf(f, "# event | layer | numa | qlen | cache_stack_top | "
                 "fwd_work_bytes | cache_pool_bytes | bwd_pool_bytes | "
                 "alloc_request_bytes | detail\n");
      fflush(f);
    }
  }
  return f;
}

// Printf-style pool log: writes one line per event
// event: "fwd_alloc", "fwd_cache_alloc", "bwd_alloc", "cache_free", "fwd_enter", "bwd_enter", etc.
#define SFT_POOL_LOG(event, layer, numa, qlen, cache_top,                  \
                     fwd_bytes, cache_bytes, bwd_bytes, req_bytes, ...)     \
  do {                                                                     \
    if (is_pool_log_enabled()) {                                           \
      FILE* _pf = get_pool_log_file();                                     \
      if (_pf) {                                                           \
        fprintf(_pf, "%-16s | L%02d | N%d | q%-5d | cst=%-2d | "           \
                     "fwd=%10zu | cache=%10zu | bwd=%10zu | req=%10zu | ",  \
                event, layer, numa, qlen, cache_top,                       \
                (size_t)(fwd_bytes), (size_t)(cache_bytes),                \
                (size_t)(bwd_bytes), (size_t)(req_bytes));                 \
        fprintf(_pf, __VA_ARGS__);                                         \
        fprintf(_pf, "\n");                                                \
        fflush(_pf);                                                       \
      }                                                                    \
    }                                                                      \
  } while (0)

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

// =====================================================
// Type trait: kernel has direct BB→BB transposed repack (from_bb_transposed)
// INT4 lacks this, so it falls back to to_mat + from_mat_transposed.
// =====================================================
template <typename T>
struct has_bb_transposed_repack : std::false_type {};
template <>
struct has_bb_transposed_repack<amx::GemmKernel224BF> : std::true_type {};
template <>
struct has_bb_transposed_repack<amx::GemmKernel224Int8> : std::true_type {};
template <typename T>
inline constexpr bool has_bb_transposed_repack_v = has_bb_transposed_repack<T>::value;

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
  float* down_lora_u_cache = nullptr;         // [tokens_total, lora_rank] FP32, reused by backward grad_B

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
 * @brief Singleton holding shared forward/backward working pools (one per NUMA node).
 *
 * In this training path, each NUMA partition executes layer forward/backward sequentially,
 * so seqlen-dependent working buffers can be reused across all MoE layers on that partition.
 * The shared pools are process-lifetime (freed on static destruction).
 */
struct SFTSharedPools {
  struct PerNuma {
    void* fwd_work = nullptr;
    size_t fwd_work_bytes = 0;
    void* bwd_work = nullptr;
    size_t bwd_work_bytes = 0;
    void* bwd_bb = nullptr;
    size_t bwd_bb_bytes = 0;
    int bwd_bb_owner_layer = -1;  // layer_idx that last repacked into this pool
    void* cache = nullptr;
    size_t cache_bytes = 0;
  };
  std::vector<PerNuma> pools;
  std::mutex mu;

  static SFTSharedPools& instance() {
    static SFTSharedPools inst;
    return inst;
  }

  void ensure_numa_count(int n) {
    if ((int)pools.size() < n) pools.resize(n);
  }

  static void* acquire(void*& ptr, size_t& cur_bytes, size_t required, size_t align) {
    required = (required + align - 1) / align * align;
    if (required <= cur_bytes) return ptr;
    if (ptr) {
      free(ptr);
      ptr = nullptr;
      cur_bytes = 0;
    }
    int rc = posix_memalign(&ptr, align, required);
    if (rc != 0 || !ptr) throw std::runtime_error("SFTSharedPools: posix_memalign failed");
    cur_bytes = required;
    return ptr;
  }

  ~SFTSharedPools() {
    for (auto& p : pools) {
      if (p.fwd_work) {
        free(p.fwd_work);
        p.fwd_work = nullptr;
      }
      if (p.bwd_work) {
        free(p.bwd_work);
        p.bwd_work = nullptr;
      }
      if (p.bwd_bb) {
        free(p.bwd_bb);
        p.bwd_bb = nullptr;
      }
      if (p.cache) {
        free(p.cache);
        p.cache = nullptr;
      }
    }
  }

 private:
  SFTSharedPools() = default;
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
 public:
  static constexpr bool kSkipLoRA = SkipLoRA;

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
  static constexpr size_t kAmxAlignment = 64;
  static inline size_t round_up(size_t x, size_t align) { return (x + align - 1) / align * align; }

  static inline void* alloc_aligned(size_t align, size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, align, bytes);
    if (rc != 0 || !ptr) {
      errno = rc;  // posix_memalign returns error code instead of setting errno
      perror("posix_memalign");
      throw std::runtime_error("posix_memalign failed");
    }
    return ptr;
  }

  void alloc_or_resize_forward_pool(size_t required_bytes) {
    auto& shared = SFTSharedPools::instance();
    std::lock_guard<std::mutex> guard(shared.mu);
    shared.ensure_numa_count(tp_part_idx + 1);
    auto& p = shared.pools[tp_part_idx];
    forward_pool_ = SFTSharedPools::acquire(p.fwd_work, p.fwd_work_bytes,
                                             required_bytes, kAmxAlignment);
    forward_pool_bytes_ = p.fwd_work_bytes;
  }

  void alloc_or_resize_cache_pool(size_t required_bytes) {
    required_bytes = round_up(required_bytes, kAmxAlignment);
    if (required_bytes == 0) return;
    if (config_.share_cache_pool) {
      // Shared mode: all layers share one cache pool via SFTSharedPools.
      // Safe only with gradient checkpoint (one layer at a time).
      auto& shared = SFTSharedPools::instance();
      std::lock_guard<std::mutex> guard(shared.mu);
      shared.ensure_numa_count(tp_part_idx + 1);
      auto& p = shared.pools[tp_part_idx];
      cache_pool_ = SFTSharedPools::acquire(p.cache, p.cache_bytes,
                                             required_bytes, kAmxAlignment);
      cache_pool_bytes_ = p.cache_bytes;
      cache_locally_owned_ = false;
    } else {
      // Per-layer mode: each layer has its own cache pool.
      if (required_bytes <= cache_pool_bytes_) return;
      if (cache_pool_ && cache_locally_owned_) {
        free(cache_pool_);
        cache_pool_ = nullptr;
        cache_pool_bytes_ = 0;
      }
      cache_pool_ = alloc_aligned(kAmxAlignment, required_bytes);
      cache_pool_bytes_ = required_bytes;
      cache_locally_owned_ = true;
    }
  }

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

  ggml_bf16_t* gate_lora_b_transposed_ = nullptr;  // [expert_num, lora_rank, intermediate_size]
  ggml_bf16_t* up_lora_b_transposed_ = nullptr;    // [expert_num, lora_rank, intermediate_size]
  ggml_bf16_t* down_lora_b_transposed_ = nullptr;  // [expert_num, lora_rank, hidden_size]

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
  // Experts that had non-zero contributions in last backward (for selective zeroing)
  std::vector<int> last_backward_active_experts_;
  bool grad_outputs_initialized_ = false;

  // Cache buffer pools
  void* cache_input_pool_ = nullptr;
  void* cache_gate_output_pool_ = nullptr;
  void* cache_up_output_pool_ = nullptr;
  void* cache_intermediate_pool_ = nullptr;
  void* cache_down_output_pool_ = nullptr;  // For grad_weights computation
  void* cache_down_lora_u_pool_ = nullptr;  // For down LoRA backward grad_B reuse
  size_t cache_slot_bytes_input_;
  size_t cache_slot_bytes_intermediate_;
  size_t cache_slot_bytes_down_lora_u_;

  // Forward pooled buffers (shared across layers via SFTSharedPools singleton)
  void* forward_pool_ = nullptr;
  size_t forward_pool_bytes_ = 0;

  // Cache pool (per-instance or shared via SFTSharedPools when share_cache_pool=true)
  void* cache_pool_ = nullptr;
  size_t cache_pool_bytes_ = 0;
  bool cache_locally_owned_ = true;  // false when shared via SFTSharedPools

  // Gradient intermediate buffers
  ggml_bf16_t* grad_intermediate_ = nullptr;  // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_gate_output_ = nullptr;   // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_up_output_ = nullptr;     // [max_len * k, intermediate_size]
  void* grad_intermediate_pool_ = nullptr;
  void* grad_gate_output_pool_ = nullptr;
  void* grad_up_output_pool_ = nullptr;

  // Buffer sizes for dynamic allocation
  size_t grad_buffer_bytes_ = 0;
  size_t cache_down_output_bytes_ = 0;

  // Precomputed offsets for cache operations (avoid repeated heap allocation)
  std::vector<size_t> cache_offsets_;

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
  std::vector<std::shared_ptr<typename T::BufferB>>
      down_lora_a_t_bb_;  // [expert_num] [intermediate_size, padded_lora_rank]
  std::vector<std::shared_ptr<typename T::BufferB>>
      down_lora_b_t_bb_;  // [expert_num] [padded_lora_rank, hidden_size]

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
  void* backward_pool_ = nullptr;
  size_t backward_pool_bytes_ = 0;

  // Duplicate pools for up-path in merged gate+up GEMM
  void* gu_up_ba_pool_ = nullptr;         // BufferA for up quantized grad
  void* gu_up_bc_pool_ = nullptr;         // BufferC for up GEMM output
  void* gu_up_bf16_pool_ = nullptr;       // BF16 output buffer for up
  size_t gu_up_ba_pool_bytes_ = 0;
  size_t gu_up_bc_pool_bytes_ = 0;
  size_t gu_up_bf16_pool_bytes_ = 0;


  // Per-expert buffers for up-path (gate uses down_ba_, grad_gate_up_bc_, grad_output_bf16_ptr_)
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_up_;             // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_up_bc_up_;     // [expert_num]
  std::vector<ggml_bf16_t*> grad_output_bf16_up_ptr_;                        // [expert_num]

  // Pre-allocated FP32 accumulators for gate+up LoRA grad_B
  // Avoids 5ms std::vector alloc+zero on every backward call
  float* gu_gate_gradb_accum_ = nullptr;  // [expert_num * intermediate_size * lora_rank]
  float* gu_up_gradb_accum_ = nullptr;    // [expert_num * intermediate_size * lora_rank]
  size_t gu_gradb_accum_pool_bytes_ = 0;
  void* gu_gradb_accum_pool_ = nullptr;

  // Backward buffer pool sizes
  size_t backward_ba_pool_bytes_ = 0;
  size_t backward_bc_pool_bytes_ = 0;
  size_t grad_output_bf16_pool_bytes_ = 0;

  // LoRA gradient computation pools (FP32, used in bwd_down_lora_precompute and grad computation)
  float* lora_grad_out_pool_ = nullptr;      // [max_len * num_experts_per_tok * hidden_size]
  float* lora_inter_proj_pool_ = nullptr;    // [max_len * num_experts_per_tok * lora_rank]
  float* lora_grad_times_b_pool_ = nullptr;  // [max_len * num_experts_per_tok * lora_rank]
  float* down_lora_grad_b_accum_pool_ = nullptr;   // [expert_num * hidden_size * lora_rank]
  float* down_lora_grad_a_accum_pool_ = nullptr;   // [expert_num * intermediate_size * lora_rank]
  size_t lora_grad_out_pool_bytes_ = 0;
  size_t lora_inter_proj_pool_bytes_ = 0;
  size_t lora_grad_times_b_pool_bytes_ = 0;
  size_t down_lora_grad_b_accum_pool_bytes_ = 0;
  size_t down_lora_grad_a_accum_pool_bytes_ = 0;
  std::unique_ptr<std::mutex[]> down_lora_grad_mutexes_;
  std::vector<uint8_t> down_lora_grad_accum_initialized_;

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

  // true = per-instance alloc, false = shared pool or nullptr
  bool backward_bb_locally_owned_ = false;

  // Flag to track if LoRA weights have been converted to BufferB format
  bool lora_weights_prepared_ = false;
  bool lora_backward_weights_prepared_ = false;

  bool lora_b_transposed_ = false;   // For transpose_lora_b_weights (used in forward)
  bool lora_a_bb_prepared_ = false;  // For gate_lora_a_bb_ and up_lora_a_bb_ (used in backward)

 private:
  void alloc_or_resize_backward_pool(size_t required_bytes) {
    auto& shared = SFTSharedPools::instance();
    std::lock_guard<std::mutex> guard(shared.mu);
    shared.ensure_numa_count(tp_part_idx + 1);
    auto& p = shared.pools[tp_part_idx];
    backward_pool_ = SFTSharedPools::acquire(p.bwd_work, p.bwd_work_bytes,
                                             required_bytes, kAmxAlignment);
    backward_pool_bytes_ = p.bwd_work_bytes;
  }

  void alloc_or_resize_backward_bb(size_t required_bytes) {
    auto& shared = SFTSharedPools::instance();
    std::lock_guard<std::mutex> guard(shared.mu);
    shared.ensure_numa_count(tp_part_idx + 1);
    auto& p = shared.pools[tp_part_idx];
    backward_bb_pool_ = SFTSharedPools::acquire(p.bwd_bb, p.bwd_bb_bytes,
                                                required_bytes, kAmxAlignment);
    backward_bb_pool_bytes_ = p.bwd_bb_bytes;
  }

 public:
  AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    printf("Creating AMX_SFT_MOE_TP layer=%d tp_part=%d at numa %d skiplora %s share_backward_bb %s share_cache_pool %s\n",
           config.layer_idx, tp_part_idx, numa_node_of_cpu(sched_getcpu()),
           SkipLoRA ? "true" : "false", config.share_backward_bb ? "true" : "false",
           config.share_cache_pool ? "true" : "false");

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
    down_lora_grad_mutexes_ = std::make_unique<std::mutex[]>(config.expert_num);
    down_lora_grad_accum_initialized_.assign(config.expert_num, 0);

    // Allocate pre-transposed LoRA B weight buffers (once, in constructor)
    alloc_transposed_lora_weights();

    // Initialize all buffers in a single alloc() to avoid memory overlap
    // (Bug #15: SharedMemBuffer assigns all alloc() calls from same base address)
    init_all_buffers();
  }

  // Constructor to satisfy MOE_TP_PART concept (takes GeneralMOEConfig)
  AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}

  ~AMX_SFT_MOE_TP() {
    // forward_pool_ → shared (singleton-owned, process-lifetime), do NOT free
    // backward_pool_ → shared (singleton-owned, process-lifetime), do NOT free
    // Cache pool: only free if locally owned (not shared via SFTSharedPools)
    if (cache_locally_owned_ && cache_pool_) free(cache_pool_);
    // Persistent buffers (allocated in constructor)
    if (lora_bb_pool_) free(lora_bb_pool_);
    if (backward_bb_locally_owned_ && backward_bb_pool_) free(backward_bb_pool_);
    // Pre-transposed LoRA weights
    free_transposed_lora_weights();
  }

  /**
   * @brief Allocate forward-phase buffers.
   * Called at the start of forward_sft.
   * - LoRA working buffers: always allocated (needed for forward LoRA computation)
   * - Cache buffers: only allocated when save_for_backward=true
   *
   * @param alloc_cache Whether to allocate cache buffers (for backward pass)
   */
  void alloc_forward_buffers(bool alloc_cache) {
    // 1. Working buffers → shared pool (across all layers on same NUMA)
    size_t work_required = 0;
    work_required += round_up(lora_ba_pool_bytes_, kAmxAlignment);
    work_required += round_up(lora_bc_inter_pool_bytes_, kAmxAlignment);
    work_required += round_up(lora_bc_out_pool_bytes_, kAmxAlignment);
    work_required += round_up(lora_intermediate_bf16_pool_bytes_, kAmxAlignment);

    alloc_or_resize_forward_pool(work_required);

    SFT_POOL_LOG("fwd_work", config_.layer_idx, tp_part_idx, 0, cache_stack_top_,
                 forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, work_required,
                 "shared_pool alloc_cache=%d", (int)alloc_cache);

    auto* work_base = static_cast<uint8_t*>(forward_pool_);
    size_t offset = 0;
    auto assign = [&](void** ptr, size_t bytes) {
      if (bytes == 0) {
        *ptr = nullptr;
        return;
      }
      *ptr = work_base + offset;
      offset += round_up(bytes, kAmxAlignment);
    };

    // LoRA working buffers (always needed for forward, even for inference)
    assign(&lora_ba_pool_, lora_ba_pool_bytes_);
    assign(&lora_bc_inter_pool_, lora_bc_inter_pool_bytes_);
    assign(&lora_bc_out_pool_, lora_bc_out_pool_bytes_);
    assign(&lora_intermediate_bf16_pool_, lora_intermediate_bf16_pool_bytes_);

    // 2. Cache buffers → per-instance pool
    if (alloc_cache) {
      const size_t cache_input_bytes = cache_slot_bytes_input_ * max_cache_depth_;
      const size_t cache_intermediate_bytes = cache_slot_bytes_intermediate_ * max_cache_depth_;
      const size_t cache_down_lora_u_bytes = cache_slot_bytes_down_lora_u_ * max_cache_depth_;

      size_t cache_required = 0;
      cache_required += round_up(cache_input_bytes, kAmxAlignment);
      cache_required += round_up(cache_intermediate_bytes, kAmxAlignment) * 3;
      cache_required += round_up(cache_down_lora_u_bytes, kAmxAlignment);
      cache_required += round_up(cache_down_output_bytes_, kAmxAlignment);

      alloc_or_resize_cache_pool(cache_required);

      SFT_POOL_LOG("fwd_cache", config_.layer_idx, tp_part_idx, 0, cache_stack_top_,
                   forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, cache_required,
                   "cache_pool alloc");

      auto* cache_base = static_cast<uint8_t*>(cache_pool_);
      size_t cache_offset = 0;
      auto cache_assign = [&](void** ptr, size_t bytes) {
        if (bytes == 0) {
          *ptr = nullptr;
          return;
        }
        *ptr = cache_base + cache_offset;
        cache_offset += round_up(bytes, kAmxAlignment);
      };

      cache_assign(&cache_input_pool_, cache_input_bytes);
      cache_assign(&cache_gate_output_pool_, cache_intermediate_bytes);
      cache_assign(&cache_up_output_pool_, cache_intermediate_bytes);
      cache_assign(&cache_intermediate_pool_, cache_intermediate_bytes);
      cache_assign(&cache_down_lora_u_pool_, cache_down_lora_u_bytes);
      cache_assign(&cache_down_output_pool_, cache_down_output_bytes_);

      // Initialize cache stack pointers (use size_t to prevent int overflow)
      for (int i = 0; i < max_cache_depth_; i++) {
        const size_t si = i;
        const size_t ml = config_.max_len;
        const size_t k_tok = config_.num_experts_per_tok;
        const size_t H = config_.hidden_size;
        const size_t I = config_.intermediate_size;
        cache_stack_[i].input_cache = (ggml_bf16_t*)cache_input_pool_ + si * ml * H;
        cache_stack_[i].gate_output_cache = (ggml_bf16_t*)cache_gate_output_pool_ + si * ml * k_tok * I;
        cache_stack_[i].up_output_cache = (ggml_bf16_t*)cache_up_output_pool_ + si * ml * k_tok * I;
        cache_stack_[i].intermediate_cache = (ggml_bf16_t*)cache_intermediate_pool_ + si * ml * k_tok * I;
        cache_stack_[i].down_lora_u_cache =
            (float*)cache_down_lora_u_pool_ + si * ml * k_tok * lora_rank_;
        cache_stack_[i].down_output_cache = (ggml_bf16_t*)cache_down_output_pool_ + si * ml * k_tok * H;
      }
    } else {
      cache_input_pool_ = nullptr;
      cache_gate_output_pool_ = nullptr;
      cache_up_output_pool_ = nullptr;
      cache_intermediate_pool_ = nullptr;
      cache_down_lora_u_pool_ = nullptr;
      cache_down_output_pool_ = nullptr;
    }
  }

  /**
   * @brief Free LoRA working buffers (for inference mode).
   * Called at the end of forward_sft when save_for_backward=false.
   */
  void free_lora_working_buffers() {
    // Intentionally keep pooled buffers to avoid frequent alloc/free in inference loops.
  }

  /**
   * @brief Allocate backward-phase buffers.
   * Called at the start of backward.
   * Includes: gradient buffers + backward working buffers
   */
  void alloc_backward_buffers() {
    // Allocate backward-phase buffers from a single resizable pool (like forward_pool_).
    size_t required = 0;
    required += round_up(grad_buffer_bytes_, kAmxAlignment) * 3;  // grad_intermediate, grad_gate_output, grad_up_output
    required += round_up(backward_ba_pool_bytes_, kAmxAlignment);
    required += round_up(backward_bc_pool_bytes_, kAmxAlignment);
    required += round_up(grad_output_bf16_pool_bytes_, kAmxAlignment);
    required += round_up(gu_up_ba_pool_bytes_, kAmxAlignment);
    required += round_up(gu_up_bc_pool_bytes_, kAmxAlignment);
    required += round_up(gu_up_bf16_pool_bytes_, kAmxAlignment);
    required += round_up(gu_gradb_accum_pool_bytes_, kAmxAlignment);
    required += round_up(lora_grad_out_pool_bytes_, kAmxAlignment);
    required += round_up(lora_inter_proj_pool_bytes_, kAmxAlignment);
    required += round_up(lora_grad_times_b_pool_bytes_, kAmxAlignment);
    required += round_up(down_lora_grad_b_accum_pool_bytes_, kAmxAlignment);
    required += round_up(down_lora_grad_a_accum_pool_bytes_, kAmxAlignment);

    alloc_or_resize_backward_pool(required);

    SFT_POOL_LOG("bwd_alloc", config_.layer_idx, tp_part_idx, 0, cache_stack_top_,
                 forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, required,
                 "backward_pool alloc");

    auto* base = static_cast<uint8_t*>(backward_pool_);
    size_t offset = 0;
    auto assign = [&](void** ptr, size_t bytes) {
      if (bytes == 0) {
        *ptr = nullptr;
        return;
      }
      *ptr = base + offset;
      offset += round_up(bytes, kAmxAlignment);
    };

    assign(&grad_intermediate_pool_, grad_buffer_bytes_);
    assign(&grad_gate_output_pool_, grad_buffer_bytes_);
    assign(&grad_up_output_pool_, grad_buffer_bytes_);
    grad_intermediate_ = (ggml_bf16_t*)grad_intermediate_pool_;
    grad_gate_output_ = (ggml_bf16_t*)grad_gate_output_pool_;
    grad_up_output_ = (ggml_bf16_t*)grad_up_output_pool_;

    assign(&backward_ba_pool_, backward_ba_pool_bytes_);
    assign(&backward_bc_pool_, backward_bc_pool_bytes_);
    assign(&grad_output_bf16_pool_, grad_output_bf16_pool_bytes_);
    assign(&gu_up_ba_pool_, gu_up_ba_pool_bytes_);
    assign(&gu_up_bc_pool_, gu_up_bc_pool_bytes_);
    assign(&gu_up_bf16_pool_, gu_up_bf16_pool_bytes_);
    assign(&gu_gradb_accum_pool_, gu_gradb_accum_pool_bytes_);
    if (gu_gradb_accum_pool_ && gu_gradb_accum_pool_bytes_ > 0) {
      size_t half = gu_gradb_accum_pool_bytes_ / 2;
      gu_gate_gradb_accum_ = (float*)gu_gradb_accum_pool_;
      gu_up_gradb_accum_ = (float*)((char*)gu_gradb_accum_pool_ + round_up(half, kAmxAlignment));
    }

    assign((void**)&lora_grad_out_pool_, lora_grad_out_pool_bytes_);
    assign((void**)&lora_inter_proj_pool_, lora_inter_proj_pool_bytes_);
    assign((void**)&lora_grad_times_b_pool_, lora_grad_times_b_pool_bytes_);
    assign((void**)&down_lora_grad_b_accum_pool_, down_lora_grad_b_accum_pool_bytes_);
    assign((void**)&down_lora_grad_a_accum_pool_, down_lora_grad_a_accum_pool_bytes_);
  }

  /**
   * @brief Free seqlen-dependent buffers after backward.
   * Called at the end of backward.
   */
  void free_seqlen_buffers() {
    SFT_POOL_LOG("cache_free", config_.layer_idx, tp_part_idx, 0, cache_stack_top_,
                 forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, cache_pool_bytes_,
                 "freeing cache_pool");

    // Hard check: all cache entries must have been popped before freeing.
    // A non-zero cache_stack_top_ means backward didn't consume all pushes,
    // and freeing would leave dangling pointers in the cache stack.
    if (cache_stack_top_ != 0) {
      fprintf(stderr,
              "[KT-MOE BUG] free_seqlen_buffers called with cache_stack_top_=%d "
              "(expected 0) on layer %d numa %d. Skipping cache free.\n",
              cache_stack_top_, config_.layer_idx, tp_part_idx);
      return;  // Do NOT free — better to leak than corrupt
    }
    if (cache_locally_owned_ && cache_pool_) {
      free(cache_pool_);
    }
    cache_pool_ = nullptr;
    cache_pool_bytes_ = 0;
    cache_input_pool_ = nullptr;
    cache_gate_output_pool_ = nullptr;
    cache_up_output_pool_ = nullptr;
    cache_intermediate_pool_ = nullptr;
    cache_down_lora_u_pool_ = nullptr;
    cache_down_output_pool_ = nullptr;
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

    SFT_POOL_LOG("fwd_enter", config_.layer_idx, tp_part_idx, qlen, cache_stack_top_,
                 forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, 0,
                 "save_bwd=%d", (int)save_for_backward);

    // =====================================================
    // Bounds Check: Verify qlen doesn't exceed max_len
    // =====================================================
    if (is_nan_check_enabled() && qlen > config_.max_len) {
      printf(ANSI_BG_RED "[OVERFLOW L%d] qlen=%d EXCEEDS max_len=%d! Buffer overflow will occur!" ANSI_COLOR_RESET "\n",
             config_.layer_idx, qlen, config_.max_len);
    }

    // NaN Check: Input
    if (is_nan_check_enabled()) {
      char label[128];
      snprintf(label, sizeof(label), "[FWD L%d] Input", config_.layer_idx);
      check_bf16_buffer_for_nan((const ggml_bf16_t*)input, qlen * config_.hidden_size, label);
    }

    // ★ Allocate forward-phase buffers ★
    // LoRA working buffers are always needed for forward (even for inference)
    // Cache buffers are only needed when save_for_backward=true
    alloc_forward_buffers(save_for_backward);

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Lazy preparation: transpose LoRA B weights for AVX512 fused_add kernel
    if (!lora_b_transposed_ && gate_lora_b_ != nullptr) {
      transpose_lora_b_weights();
      lora_b_transposed_ = true;
    }

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

    // =====================================================
    // Bounds Check: Verify base class pool allocation didn't overflow
    // =====================================================
    if (is_nan_check_enabled()) {
      char* gate_up_ba_pool_end = (char*)Base::gate_up_ba_pool_ + Base::gate_up_ba_pool_bytes_;
      char* gate_bc_pool_end = (char*)Base::gate_bc_pool_ + Base::gate_bc_pool_bytes_;
      char* up_bc_pool_end = (char*)Base::up_bc_pool_ + Base::up_bc_pool_bytes_;
      char* down_ba_pool_end = (char*)Base::down_ba_pool_ + Base::down_ba_pool_bytes_;
      char* down_bc_pool_end = (char*)Base::down_bc_pool_ + Base::down_bc_pool_bytes_;

      bool overflow = false;
      if ((char*)gate_up_ba_pool_ptr > gate_up_ba_pool_end) {
        size_t used = (char*)gate_up_ba_pool_ptr - (char*)Base::gate_up_ba_pool_;
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] gate_up_ba_pool: used=%zu, allocated=%zu, OVERFLOW by %zu bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, used, Base::gate_up_ba_pool_bytes_, used - Base::gate_up_ba_pool_bytes_);
        overflow = true;
      }
      if ((char*)gate_bc_pool_ptr > gate_bc_pool_end) {
        size_t used = (char*)gate_bc_pool_ptr - (char*)Base::gate_bc_pool_;
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] gate_bc_pool: used=%zu, allocated=%zu, OVERFLOW by %zu bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, used, Base::gate_bc_pool_bytes_, used - Base::gate_bc_pool_bytes_);
        overflow = true;
      }
      if ((char*)up_bc_pool_ptr > up_bc_pool_end) {
        size_t used = (char*)up_bc_pool_ptr - (char*)Base::up_bc_pool_;
        printf(ANSI_BG_RED "[OVERFLOW L%d] up_bc_pool: used=%zu, allocated=%zu, OVERFLOW by %zu bytes" ANSI_COLOR_RESET
                           "\n",
               config_.layer_idx, used, Base::up_bc_pool_bytes_, used - Base::up_bc_pool_bytes_);
        overflow = true;
      }
      if ((char*)down_ba_pool_ptr > down_ba_pool_end) {
        size_t used = (char*)down_ba_pool_ptr - (char*)Base::down_ba_pool_;
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] down_ba_pool: used=%zu, allocated=%zu, OVERFLOW by %zu bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, used, Base::down_ba_pool_bytes_, used - Base::down_ba_pool_bytes_);
        overflow = true;
      }
      if ((char*)down_bc_pool_ptr > down_bc_pool_end) {
        size_t used = (char*)down_bc_pool_ptr - (char*)Base::down_bc_pool_;
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] down_bc_pool: used=%zu, allocated=%zu, OVERFLOW by %zu bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, used, Base::down_bc_pool_bytes_, used - Base::down_bc_pool_bytes_);
        overflow = true;
      }

      if (overflow) {
        printf("[OVERFLOW DEBUG L%d] qlen=%d, k=%d, max_len=%d, pool_count=%zu, activated_expert=%d\n",
               config_.layer_idx, qlen, k, config_.max_len, Base::pool_count_, activated_expert);
        printf("[OVERFLOW DEBUG L%d] Total tokens processed: %zu (offset after loop)\n", config_.layer_idx, offset);
      }
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

    // NaN Check: Step 3 - Packed input
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step3 packed_input expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_input_ptr_[expert_idx], m_local_num_[expert_idx] * config_.hidden_size,
                                    label);
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

    // NaN Check: Step 5 - Gate/Up GEMM output (before LoRA)
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step5 gate_base_output expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_gate_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.intermediate_size, label);
          snprintf(label, sizeof(label), "[FWD L%d] Step5 up_base_output expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_up_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.intermediate_size, label);
        }
      }
    }

    // Step 5.5: Gate + Up LoRA (AVX512 BF16 - no BufferB conversion needed)
    if (!SkipLoRA) {
      compute_lora_gate_up(qlen, activated_expert);
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

    // NaN Check: Step 5.5 - Gate/Up output (after LoRA)
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step5.5 gate_after_lora expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_gate_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.intermediate_size, label);
          snprintf(label, sizeof(label), "[FWD L%d] Step5.5 up_after_lora expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_up_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.intermediate_size, label);
        }
      }
    }

    // Save gate/up outputs before activation (for backward)
    if (save_for_backward) {
      // If a cache entry already exists (checkpoint recompute scenario),
      // overwrite it instead of pushing a new one.  This keeps the cache
      // consistent with the current forward's buffer state (max_m, routing)
      // and avoids cache stack overflow from duplicate pushes.
      ForwardCache& cache = (cache_stack_top_ > 0) ? cache_stack_[cache_stack_top_ - 1] : push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert, input);

      // NaN Check: Forward Cache - input, gate_output, up_output
      if (is_nan_check_enabled()) {
        auto check_cache_bf16 = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
          if (ptr == nullptr || elems == 0) return;
          double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
          int nan_count = 0, inf_count = 0;
          for (size_t i = 0; i < elems; i++) {
            float v = GGML_BF16_TO_FP32(ptr[i]);
            if (v != v) nan_count++;
            if (!(v != v) && is_inf_value(v)) inf_count++;
            double dv = static_cast<double>(v);
            double a = std::fabs(dv);
            sum_sq += dv * dv;
            sum_abs += a;
            if (a > max_abs || a != a) max_abs = a;
          }
          double norm = std::sqrt(sum_sq);
          double abs_mean = sum_abs / static_cast<double>(elems);
          bool has_nan_inf = (nan_count > 0 || inf_count > 0);
          bool computed_nan = (norm != norm) || (abs_mean != abs_mean);
          const char* bg = (has_nan_inf || computed_nan) ? ANSI_BG_RED : ANSI_BG_BLUE;
          printf(
              "%s[CACHE SAVE L%d] %s: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d (total=%zu)" ANSI_COLOR_RESET
              "\n",
              bg, config_.layer_idx, name, norm, abs_mean, max_abs, nan_count, inf_count, elems);
        };

        size_t total_tokens = 0;
        for (int i = 0; i < activated_expert; i++) {
          total_tokens += m_local_num_[m_expert_id_map_[i]];
        }
        check_cache_bf16("input_cache", cache.input_cache, qlen * config_.hidden_size);
        check_cache_bf16("gate_output_cache", cache.gate_output_cache, total_tokens * config_.intermediate_size);
        check_cache_bf16("up_output_cache", cache.up_output_cache, total_tokens * config_.intermediate_size);
      }
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
    {
      uint64_t act_start = sft_timer::get_trace_timestamp();
      Base::apply_activation(activated_expert, nth, qlen);
      uint64_t act_end = sft_timer::get_trace_timestamp();
      sft_timer::add_kernel_trace("apply_activation", act_start, act_end, tp_part_idx, 0);
    }

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

    // NaN Check: Step 6 - Activation output (silu(gate) * up)
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step6 activation_output expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_gate_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.intermediate_size, label);
        }
      }
    }

    // Save intermediate AFTER activation for backward_down (Bug #17c fix)
    if (save_for_backward) {
      ForwardCache& cache = cache_stack_[cache_stack_top_ - 1];  // Get the cache we just pushed
      save_intermediate_to_cache(cache, activated_expert);

      // NaN Check: Forward Cache - intermediate_cache
      if (is_nan_check_enabled()) {
        size_t total_tokens = 0;
        for (int i = 0; i < activated_expert; i++) {
          total_tokens += m_local_num_[m_expert_id_map_[i]];
        }
        size_t elems = total_tokens * config_.intermediate_size;
        if (cache.intermediate_cache != nullptr && elems > 0) {
          double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
          int nan_count = 0, inf_count = 0;
          for (size_t i = 0; i < elems; i++) {
            float v = GGML_BF16_TO_FP32(cache.intermediate_cache[i]);
            if (v != v) nan_count++;
            if (!(v != v) && is_inf_value(v)) inf_count++;
            double dv = static_cast<double>(v);
            double a = std::fabs(dv);
            sum_sq += dv * dv;
            sum_abs += a;
            if (a > max_abs || a != a) max_abs = a;
          }
          double norm = std::sqrt(sum_sq);
          double abs_mean = sum_abs / static_cast<double>(elems);
          bool has_nan_inf = (nan_count > 0 || inf_count > 0);
          bool computed_nan = (norm != norm) || (abs_mean != abs_mean);
          const char* bg = (has_nan_inf || computed_nan) ? ANSI_BG_RED : ANSI_BG_BLUE;
          printf(
              "%s[CACHE SAVE L%d] intermediate_cache: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d "
              "(total=%zu)" ANSI_COLOR_RESET "\n",
              bg, config_.layer_idx, norm, abs_mean, max_abs, nan_count, inf_count, elems);
        }
      }
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

    // NaN Check: Step 8 - Down GEMM output (before LoRA)
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step8 down_base_output expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_down_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.hidden_size, label);
        }
      }
    }

    // Step 8.5: Down LoRA (AVX512 BF16 - no BufferB conversion needed)
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      ForwardCache* cache_ptr = save_for_backward ? &cache_stack_[cache_stack_top_ - 1] : nullptr;
      compute_lora_down(qlen, activated_expert, cache_ptr);
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

    // NaN Check: Step 8.5 - Down output (after LoRA)
    if (is_nan_check_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        if (m_local_num_[expert_idx] > 0) {
          char label[128];
          snprintf(label, sizeof(label), "[FWD L%d] Step8.5 down_after_lora expert=%d tokens=%d", config_.layer_idx,
                   expert_idx, m_local_num_[expert_idx]);
          check_bf16_buffer_for_nan(m_local_down_output_ptr_[expert_idx],
                                    m_local_num_[expert_idx] * config_.hidden_size, label);
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

    // NaN Check: Step 9 - Final output (after weighted merge)
    if (is_nan_check_enabled()) {
      char label[128];
      snprintf(label, sizeof(label), "[FWD L%d] Step9 final_output", config_.layer_idx);
      check_fp32_buffer_for_nan((const float*)output, qlen * config_.hidden_size, label);
    }

    // ★ Inference mode cleanup ★
    // LoRA working buffers are pooled (kept) to avoid frequent alloc/free overhead.
    if (!save_for_backward) {
      free_lora_working_buffers();
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
                void* grad_weights,
                int full_intermediate_size = 0,
                float* fp32_grad_down_lora_b = nullptr,
                float* fp32_grad_gate_lora_a = nullptr,
                float* fp32_grad_up_lora_a = nullptr) {
    // If full_intermediate_size not provided, use local (non-TP mode)
    if (full_intermediate_size == 0) full_intermediate_size = config_.intermediate_size;
    SFT_POOL_LOG("bwd_enter", config_.layer_idx, tp_part_idx, 0, cache_stack_top_,
                 forward_pool_bytes_, cache_pool_bytes_, backward_pool_bytes_, 0,
                 "backward entry");

    // Pop cache from stack
    ForwardCache cache = pop_cache();
    if (!cache.valid) {
      throw std::runtime_error("No valid forward cache for backward");
    }

    int qlen = cache.qlen_cache;
    int k = cache.k_cache;
    int activated_expert = cache.activated_expert_cache;
    constexpr int kSmallBwdDirectQlen = 0;
    constexpr int kSmallBwdDirectMaxTasks = 16;
    auto trace_phase = [this](const char* name, auto&& fn) {
      uint64_t start = sft_timer::get_trace_timestamp();
      fn();
      uint64_t end = sft_timer::get_trace_timestamp();
      sft_timer::add_kernel_trace(name, start, end, tp_part_idx, 0);
    };

    // NaN Check: grad_output input
    if (is_nan_check_enabled()) {
      char label[128];
      snprintf(label, sizeof(label), "[BWD L%d] Input grad_output", config_.layer_idx);
      check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_output, qlen * config_.hidden_size, label);
    }

    // NaN Check: Forward Cache (read from cache)
    if (is_nan_check_enabled()) {
      auto check_cache_bf16 = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
        if (ptr == nullptr) {
          printf(ANSI_BG_RED "[CACHE READ L%d] %s: NULL pointer!" ANSI_COLOR_RESET "\n", config_.layer_idx, name);
          return;
        }
        if (elems == 0) {
          printf(ANSI_BG_BLUE "[CACHE READ L%d] %s: empty (elems=0)" ANSI_COLOR_RESET "\n", config_.layer_idx, name);
          return;
        }
        double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < elems; i++) {
          float v = GGML_BF16_TO_FP32(ptr[i]);
          // Use v != v for robust NaN detection
          if (v != v) nan_count++;
          if (!is_nan_value(v) && is_inf_value(v)) inf_count++;
          double dv = static_cast<double>(v);
          double a = std::fabs(dv);
          sum_sq += dv * dv;
          sum_abs += a;
          if (a > max_abs || a != a) max_abs = a;
        }
        double norm = std::sqrt(sum_sq);
        double abs_mean = sum_abs / static_cast<double>(elems);
        bool has_nan_inf = (nan_count > 0 || inf_count > 0);
        // Also check if computed values are NaN/Inf
        bool computed_nan = (norm != norm) || (abs_mean != abs_mean) || (max_abs != max_abs);
        bool has_large = (!is_nan_value(max_abs) && !is_inf_value(max_abs) && max_abs > NAN_CHECK_LARGE_THRESHOLD);
        const char* bg = (has_nan_inf || computed_nan) ? ANSI_BG_RED : ANSI_BG_BLUE;
        printf("%s[CACHE READ L%d] %s: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d (total=%zu)" ANSI_COLOR_RESET
               "\n",
               bg, config_.layer_idx, name, norm, abs_mean, max_abs, nan_count, inf_count, elems);
      };

      // Compute total tokens
      size_t total_tokens = 0;
      for (int i = 0; i < activated_expert; i++) {
        total_tokens += cache.m_local_num_cache[cache.m_expert_id_map_cache[i]];
      }

      check_cache_bf16("input_cache", cache.input_cache, qlen * config_.hidden_size);
      check_cache_bf16("gate_output_cache", cache.gate_output_cache, total_tokens * config_.intermediate_size);
      check_cache_bf16("up_output_cache", cache.up_output_cache, total_tokens * config_.intermediate_size);
      check_cache_bf16("intermediate_cache", cache.intermediate_cache, total_tokens * config_.intermediate_size);
      check_cache_bf16("down_output_cache", cache.down_output_cache, total_tokens * config_.hidden_size);
    }

    // ★ Allocate backward-phase buffers ★
    trace_phase("bwd_setup_alloc", [&] { alloc_backward_buffers(); });

    trace_phase("bwd_setup_state", [&] {
      // ★ share_backward_bb: check if async repack already prepared this layer ★
      if (config_.share_backward_bb) {
        auto& shared = SFTSharedPools::instance();
        shared.ensure_numa_count(tp_part_idx + 1);
        if (shared.pools[tp_part_idx].bwd_bb_owner_layer != config_.layer_idx) {
          // Pool was overwritten by another layer or not yet repacked — sync fallback
          prepare_backward_bb_for_async();
        }
      }

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
    });

    // Restore input data from cache into m_local_input_ (shared_mem_buffer may have been
    // overwritten by subsequent layers' forward passes). This is needed for gate/up LoRA
    // gradient computation which reads from m_local_input_ptr_.
    trace_phase("bwd_phase_down", [&] {
      auto pool_local = config_.pool->get_subpool(tp_part_idx);
      auto restore_input = [&](int i) {
        for (int j = 0; j < k; j++) {
          int eid = cache.expert_ids_cache[i * k + j];
          if (eid < config_.num_gpu_experts || eid >= config_.expert_num) {
            continue;
          }
          if (m_local_num_[eid] == 0) continue;
          int pos = cache.m_local_pos_cache[i][j];
          memcpy(m_local_input_ptr_[eid] + pos * config_.hidden_size,
                 (const ggml_bf16_t*)cache.input_cache + i * config_.hidden_size,
                 sizeof(ggml_bf16_t) * config_.hidden_size);
        }
      };
      if (qlen <= kSmallBwdDirectQlen && qlen <= kSmallBwdDirectMaxTasks) {
        for (int i = 0; i < qlen; i++) {
          restore_input(i);
        }
      } else {
        pool_local->do_work_stealing_job(qlen, nullptr, restore_input, nullptr, "bwd_restore_input", 1);
      }

      // Step 1: Down projection backward
      if constexpr (supports_standard_mat_mul_v<T>) {
        backward_down_amx(cache, grad_output, grad_down_lora_a, grad_down_lora_b,
                          full_intermediate_size, fp32_grad_down_lora_b);
      } else {
        // backward_down(cache, grad_output, grad_down_lora_a, grad_down_lora_b);
      }
    });

    // // Compute total tokens for debug
    // size_t total_tokens = 0;
    // for (int i = 0; i < activated_expert; i++) {
    //   total_tokens += m_local_num_[m_expert_id_map_[i]];
    // }

    // printf("[BACKWARD DEBUG] qlen=%d, k=%d, activated_expert=%d, total_tokens=%zu\n", qlen, k, activated_expert,
    //        total_tokens);
    // printf("[BACKWARD DEBUG] grad_output norm: %f\n",
    //        compute_bf16_norm((const ggml_bf16_t*)grad_output, qlen * config_.hidden_size));

    // NaN Check: Step 1 - After backward_down
    if (is_nan_check_enabled()) {
      char label[128];
      // Check grad_intermediate
      size_t grad_inter_size = 0;
      for (int i = 0; i < activated_expert; i++) {
        grad_inter_size += m_local_num_[m_expert_id_map_[i]];
      }
      grad_inter_size *= config_.intermediate_size;
      snprintf(label, sizeof(label), "[BWD L%d] Step1 grad_intermediate", config_.layer_idx);
      check_bf16_buffer_for_nan(grad_intermediate_, grad_inter_size, label);

      // Check grad_down_lora_a
      if (grad_down_lora_a != nullptr) {
        size_t down_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
        snprintf(label, sizeof(label), "[BWD L%d] Step1 grad_down_lora_a", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_down_lora_a, down_a_elems, label);
      }
      // Check grad_down_lora_b
      if (grad_down_lora_b != nullptr) {
        size_t down_b_elems = static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_;
        snprintf(label, sizeof(label), "[BWD L%d] Step1 grad_down_lora_b", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_down_lora_b, down_b_elems, label);
      }
    }

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

    trace_phase("bwd_phase_act", [&] { backward_activation(cache); });

    // NaN Check: Step 2 - After backward_activation
    if (is_nan_check_enabled()) {
      char label[128];
      size_t grad_size = 0;
      for (int i = 0; i < activated_expert; i++) {
        grad_size += m_local_num_[m_expert_id_map_[i]];
      }
      grad_size *= config_.intermediate_size;
      snprintf(label, sizeof(label), "[BWD L%d] Step2 grad_gate_output", config_.layer_idx);
      check_bf16_buffer_for_nan(grad_gate_output_, grad_size, label);
      snprintf(label, sizeof(label), "[BWD L%d] Step2 grad_up_output", config_.layer_idx);
      check_bf16_buffer_for_nan(grad_up_output_, grad_size, label);
    }

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

    trace_phase("bwd_phase_gate_up", [&] {
      if constexpr (supports_standard_mat_mul_v<T>) {
        backward_gate_up_amx(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b,
                            full_intermediate_size, fp32_grad_gate_lora_a, fp32_grad_up_lora_a);
      } else {
        // backward_gate_up(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
      }
    });

    // NaN Check: Step 3 - After backward_gate_up
    if (is_nan_check_enabled()) {
      char label[128];
      // Check grad_input
      snprintf(label, sizeof(label), "[BWD L%d] Step3 grad_input", config_.layer_idx);
      check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_input, qlen * config_.hidden_size, label);

      // Check grad_gate_lora_a
      if (grad_gate_lora_a != nullptr) {
        size_t gate_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
        snprintf(label, sizeof(label), "[BWD L%d] Step3 grad_gate_lora_a", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_gate_lora_a, gate_a_elems, label);
      }
      // Check grad_gate_lora_b
      if (grad_gate_lora_b != nullptr) {
        size_t gate_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
        snprintf(label, sizeof(label), "[BWD L%d] Step3 grad_gate_lora_b", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_gate_lora_b, gate_b_elems, label);
      }
      // Check grad_up_lora_a
      if (grad_up_lora_a != nullptr) {
        size_t up_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
        snprintf(label, sizeof(label), "[BWD L%d] Step3 grad_up_lora_a", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_up_lora_a, up_a_elems, label);
      }
      // Check grad_up_lora_b
      if (grad_up_lora_b != nullptr) {
        size_t up_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
        snprintf(label, sizeof(label), "[BWD L%d] Step3 grad_up_lora_b", config_.layer_idx);
        check_bf16_buffer_for_nan((const ggml_bf16_t*)grad_up_lora_b, up_b_elems, label);
      }
    }

    // Step 4: Compute grad_weights (gradient for routing weights)
    // grad_weights[token_idx, expert_pos] = dot(grad_output[token_idx], down_output[token, expert])
    if (grad_weights != nullptr) {
      trace_phase("bwd_phase_gradw", [&] {
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

        // Compute grad_weights for each token-expert pair
        auto compute_grad_weight = [&](int token_idx) {
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
        };
        if (qlen <= kSmallBwdDirectQlen && qlen <= kSmallBwdDirectMaxTasks) {
          for (int token_idx = 0; token_idx < qlen; token_idx++) {
            compute_grad_weight(token_idx);
          }
        } else {
          pool->do_work_stealing_job(qlen, nullptr, compute_grad_weight, nullptr, "bwd_grad_weights");
        }
      });
    }

    // NaN Check: Step 4 - After grad_weights computation
    if (is_nan_check_enabled() && grad_weights != nullptr) {
      char label[128];
      snprintf(label, sizeof(label), "[BWD L%d] Step4 grad_weights", config_.layer_idx);
      check_fp32_buffer_for_nan((const float*)grad_weights, qlen * k, label);
    }

    // NaN Check & Norm: Final output gradients summary
    if (is_nan_check_enabled()) {
      auto print_grad_stats = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
        if (ptr == nullptr) {
          printf(ANSI_COLOR_RED "[BWD L%d OUTPUT] %s: NULL pointer!" ANSI_COLOR_RESET "\n", config_.layer_idx, name);
          return;
        }
        if (elems == 0) {
          printf(ANSI_COLOR_YELLOW "[BWD L%d OUTPUT] %s: empty (elems=0)" ANSI_COLOR_RESET "\n", config_.layer_idx,
                 name);
          return;
        }
        // Compute stats and NaN check in one pass - DO NOT skip NaN/Inf
        double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < elems; i++) {
          float v = GGML_BF16_TO_FP32(ptr[i]);
          // Use v != v for robust NaN detection
          if (v != v) {
            nan_count++;
          }
          if (!(v != v) && is_inf_value(v)) {
            inf_count++;
          }
          double dv = static_cast<double>(v);
          double a = std::fabs(dv);
          sum_sq += dv * dv;
          sum_abs += a;
          if (a > max_abs || a != a) max_abs = a;
        }
        double norm = std::sqrt(sum_sq);
        double abs_mean = sum_abs / static_cast<double>(elems);
        bool has_nan_inf = (nan_count > 0 || inf_count > 0);
        // Also check if computed values are NaN
        bool computed_nan = (norm != norm) || (abs_mean != abs_mean);
        bool has_large = (!(max_abs != max_abs) && !is_inf_value(max_abs) && max_abs > NAN_CHECK_LARGE_THRESHOLD);
        const char* color = (has_nan_inf || computed_nan) ? ANSI_COLOR_RED : (has_large ? ANSI_COLOR_YELLOW : "");
        const char* reset = (has_nan_inf || computed_nan || has_large) ? ANSI_COLOR_RESET : "";
        printf("%s[BWD L%d OUTPUT] %s: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d (total=%zu)%s\n", color,
               config_.layer_idx, name, norm, abs_mean, max_abs, nan_count, inf_count, elems, reset);
      };

      auto print_grad_stats_fp32 = [&](const char* name, const float* ptr, size_t elems) {
        if (ptr == nullptr) {
          printf(ANSI_COLOR_RED "[BWD L%d OUTPUT] %s: NULL pointer!" ANSI_COLOR_RESET "\n", config_.layer_idx, name);
          return;
        }
        if (elems == 0) {
          printf(ANSI_COLOR_YELLOW "[BWD L%d OUTPUT] %s: empty (elems=0)" ANSI_COLOR_RESET "\n", config_.layer_idx,
                 name);
          return;
        }
        // DO NOT skip NaN/Inf - include them in computation
        double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < elems; i++) {
          float fv = ptr[i];
          // Use fv != fv for robust NaN detection
          if (fv != fv) {
            nan_count++;
          }
          if (!(fv != fv) && is_inf_value(fv)) {
            inf_count++;
          }
          double v = static_cast<double>(fv);
          double a = std::fabs(v);
          sum_sq += v * v;
          sum_abs += a;
          if (a > max_abs || a != a) max_abs = a;
        }
        double norm = std::sqrt(sum_sq);
        double abs_mean = sum_abs / static_cast<double>(elems);
        bool has_nan_inf = (nan_count > 0 || inf_count > 0);
        // Also check if computed values are NaN
        bool computed_nan = (norm != norm) || (abs_mean != abs_mean);
        bool has_large = (!(max_abs != max_abs) && !is_inf_value(max_abs) && max_abs > NAN_CHECK_LARGE_THRESHOLD);
        const char* color = (has_nan_inf || computed_nan) ? ANSI_COLOR_RED : (has_large ? ANSI_COLOR_YELLOW : "");
        const char* reset = (has_nan_inf || computed_nan || has_large) ? ANSI_COLOR_RESET : "";
        printf("%s[BWD L%d OUTPUT] %s: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d (total=%zu)%s\n", color,
               config_.layer_idx, name, norm, abs_mean, max_abs, nan_count, inf_count, elems, reset);
      };

      // grad_input
      print_grad_stats("grad_input", (const ggml_bf16_t*)grad_input, qlen * config_.hidden_size);

      // LoRA gradient sizes
      size_t gate_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
      size_t gate_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
      size_t up_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
      size_t up_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
      size_t down_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
      size_t down_b_elems = static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_;

      // Gate LoRA gradients
      print_grad_stats("grad_gate_lora_a", (const ggml_bf16_t*)grad_gate_lora_a, gate_a_elems);
      print_grad_stats("grad_gate_lora_b", (const ggml_bf16_t*)grad_gate_lora_b, gate_b_elems);

      // Up LoRA gradients
      print_grad_stats("grad_up_lora_a", (const ggml_bf16_t*)grad_up_lora_a, up_a_elems);
      print_grad_stats("grad_up_lora_b", (const ggml_bf16_t*)grad_up_lora_b, up_b_elems);

      // Down LoRA gradients
      print_grad_stats("grad_down_lora_a", (const ggml_bf16_t*)grad_down_lora_a, down_a_elems);
      print_grad_stats("grad_down_lora_b", (const ggml_bf16_t*)grad_down_lora_b, down_b_elems);

      // Routing weights gradient
      print_grad_stats_fp32("grad_weights", (const float*)grad_weights, qlen * k);
    }

    // ★ Cache pool is NOT freed here — kept for reuse across steps.
    // alloc_or_resize_cache_pool() is grow-only, so same-seqlen steps
    // reuse the existing allocation without malloc/free overhead.
    // Previously: free_seqlen_buffers() was called here, costing ~3.6ms per TP.

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

  int get_cache_activated_expert_count() const {
    return (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
        ? cache_stack_[cache_stack_top_ - 1].activated_expert_cache : 0;
  }

  const int* get_cache_expert_id_map() const {
    return (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
        ? cache_stack_[cache_stack_top_ - 1].m_expert_id_map_cache.data() : nullptr;
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

    // NaN Check and Norm printing for LoRA weights
    if (is_nan_check_enabled()) {
      auto print_lora_stats = [&](const char* name, const ggml_bf16_t* ptr, size_t elems) {
        if (ptr == nullptr) {
          printf("[LoRA L%d] %s: null\n", config_.layer_idx, name);
          return;
        }
        if (elems == 0) {
          printf(ANSI_COLOR_YELLOW "[LoRA L%d] %s: empty (elems=0)" ANSI_COLOR_RESET "\n", config_.layer_idx, name);
          return;
        }
        // DO NOT skip NaN/Inf - include them in computation
        double sum_sq = 0.0, sum_abs = 0.0, max_abs = 0.0;
        int nan_count = 0, inf_count = 0;
        for (size_t i = 0; i < elems; i++) {
          float v = GGML_BF16_TO_FP32(ptr[i]);
          // Use v != v for robust NaN detection
          if (v != v) {
            nan_count++;
          }
          if (!(v != v) && is_inf_value(v)) {
            inf_count++;
          }
          double dv = static_cast<double>(v);
          double a = std::fabs(dv);
          sum_sq += dv * dv;
          sum_abs += a;
          if (a > max_abs || a != a) max_abs = a;
        }
        double norm = std::sqrt(sum_sq);
        double abs_mean = sum_abs / static_cast<double>(elems);
        bool has_nan_inf = (nan_count > 0 || inf_count > 0);
        // Also check if computed values are NaN
        bool computed_nan = (norm != norm) || (abs_mean != abs_mean);
        bool has_large = (!(max_abs != max_abs) && !is_inf_value(max_abs) && max_abs > NAN_CHECK_LARGE_THRESHOLD);
        const char* color = (has_nan_inf || computed_nan) ? ANSI_COLOR_RED : (has_large ? ANSI_COLOR_YELLOW : "");
        const char* reset = (has_nan_inf || computed_nan || has_large) ? ANSI_COLOR_RESET : "";
        printf("%s[LoRA L%d] %s: norm=%.6e abs_mean=%.6e abs_max=%.6e nan=%d inf=%d (total=%zu)%s\n", color,
               config_.layer_idx, name, norm, abs_mean, max_abs, nan_count, inf_count, elems, reset);
      };

      size_t gate_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
      size_t gate_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
      size_t up_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;
      size_t up_b_elems = static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_;
      size_t down_a_elems = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
      size_t down_b_elems = static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_;

      print_lora_stats("gate_lora_a", gate_lora_a_, gate_a_elems);
      print_lora_stats("gate_lora_b", gate_lora_b_, gate_b_elems);
      print_lora_stats("up_lora_a", up_lora_a_, up_a_elems);
      print_lora_stats("up_lora_b", up_lora_b_, up_b_elems);
      print_lora_stats("down_lora_a", down_lora_a_, down_a_elems);
      print_lora_stats("down_lora_b", down_lora_b_, down_b_elems);
    }

    // Mark weights as needing re-conversion (lazy preparation in forward/backward)
    lora_weights_prepared_ = false;
    lora_backward_weights_prepared_ = false;
    lora_b_transposed_ = false;   // Will be prepared lazily in forward_sft
    lora_a_bb_prepared_ = false;  // Will be prepared lazily in backward_gate_up_amx
  }

  /**
   * @brief Allocate buffers for pre-transposed LoRA B weights.
   *
   * Pre-transposed weights enable contiguous memory access for 16 outputs at a time,
   * providing ~5x speedup for small LoRA ranks (8-16).
   */
  void alloc_transposed_lora_weights() {
    if (lora_rank_ <= 0) return;
    if (gate_lora_b_transposed_ != nullptr) return;  // Already allocated

    size_t gate_up_b_size = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.intermediate_size;
    size_t down_b_size = static_cast<size_t>(config_.expert_num) * lora_rank_ * config_.hidden_size;

    // Allocate all transposed buffers at once
    gate_lora_b_transposed_ = (ggml_bf16_t*)aligned_alloc(64, gate_up_b_size * sizeof(ggml_bf16_t));
    up_lora_b_transposed_ = (ggml_bf16_t*)aligned_alloc(64, gate_up_b_size * sizeof(ggml_bf16_t));
    down_lora_b_transposed_ = (ggml_bf16_t*)aligned_alloc(64, down_b_size * sizeof(ggml_bf16_t));
  }

  /**
   * @brief Free pre-transposed LoRA weight buffers.
   */
  void free_transposed_lora_weights() {
    if (gate_lora_b_transposed_) {
      free(gate_lora_b_transposed_);
      gate_lora_b_transposed_ = nullptr;
    }
    if (up_lora_b_transposed_) {
      free(up_lora_b_transposed_);
      up_lora_b_transposed_ = nullptr;
    }
    if (down_lora_b_transposed_) {
      free(down_lora_b_transposed_);
      down_lora_b_transposed_ = nullptr;
    }
  }

  /**
   * @brief Transpose LoRA B weights for optimized AVX512 fused_add.
   *
   * Transposes weight from [output_dim][rank] to [rank][output_dim] for each expert.
   */
  void transpose_lora_b_weights() {
    if (lora_rank_ <= 0) return;
    if (gate_lora_b_transposed_ == nullptr) return;  // Not allocated yet

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel transpose for all experts and all LoRA B matrices
    pool->do_work_stealing_job(
        config_.expert_num * 3, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 3;
          int lora_type = task_id % 3;

          switch (lora_type) {
            case 0:  // gate_lora_b: [intermediate_size][rank] -> [rank][intermediate_size]
              if (gate_lora_b_ && gate_lora_b_transposed_) {
                size_t src_offset = static_cast<size_t>(expert_idx) * config_.intermediate_size * lora_rank_;
                size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.intermediate_size;
                avx::transpose_lora_weight(gate_lora_b_ + src_offset, gate_lora_b_transposed_ + dst_offset,
                                           config_.intermediate_size, lora_rank_);
              }
              break;
            case 1:  // up_lora_b: [intermediate_size][rank] -> [rank][intermediate_size]
              if (up_lora_b_ && up_lora_b_transposed_) {
                size_t src_offset = static_cast<size_t>(expert_idx) * config_.intermediate_size * lora_rank_;
                size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.intermediate_size;
                avx::transpose_lora_weight(up_lora_b_ + src_offset, up_lora_b_transposed_ + dst_offset,
                                           config_.intermediate_size, lora_rank_);
              }
              break;
            case 2:  // down_lora_b: [hidden_size][rank] -> [rank][hidden_size]
              if (down_lora_b_ && down_lora_b_transposed_) {
                size_t src_offset = static_cast<size_t>(expert_idx) * config_.hidden_size * lora_rank_;
                size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.hidden_size;
                avx::transpose_lora_weight(down_lora_b_ + src_offset, down_lora_b_transposed_ + dst_offset,
                                           config_.hidden_size, lora_rank_);
              }
              break;
          }
        },
        nullptr, "transpose_lora_b_weights");
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

    // Parallel conversion of forward LoRA weights to BufferB format
    // 6 matrices per expert: gate/up/down (A, B) - only for forward pass
    pool->do_work_stealing_job(
        config_.expert_num * 6, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 6;
          int lora_type = task_id % 6;

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
          }
        },
        nullptr, "fwd_lora_prep");

    lora_weights_prepared_ = true;
  }

  /**
   * @brief Prepare transposed LoRA weights needed by the backward pass.
   *
   * Gate/up backward now uses direct AVX kernels on the raw/transposed BF16
   * weights, so only the down-path AMX BufferB weights still need lazy prep.
   */
  void prepare_lora_backward_weights() {
    if constexpr (!supports_standard_mat_mul_v<T>) {
      return;
    }

    if (lora_backward_weights_prepared_) {
      return;
    }
    if (gate_lora_a_ == nullptr) {
      return;
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Only down-path LoRA backward still consumes BufferB weights.
    pool->do_work_stealing_job(
        config_.expert_num * 2, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 2;
          int lora_type = task_id % 2;

          switch (lora_type) {
            case 0:  // down_lora_a^T [rank, inter_size] -> [inter_size, padded_rank]
              convert_lora_a_transposed_to_buffer_b(down_lora_a_, down_lora_a_t_bb_[expert_idx], expert_idx, lora_rank_,
                                                    config_.intermediate_size, config_.intermediate_size,
                                                    padded_lora_rank_);
              break;
            case 1:  // down_lora_b^T [hidden_size, rank] -> [padded_rank, hidden_size]
              convert_lora_b_transposed_to_buffer_b(down_lora_b_, down_lora_b_t_bb_[expert_idx], expert_idx,
                                                    config_.hidden_size, lora_rank_, padded_lora_rank_,
                                                    config_.hidden_size);
              break;
          }
        },
        nullptr, "bwd_lora_prep");

    lora_backward_weights_prepared_ = true;
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

    // Fine-grained parallelism: nth_gate_up * expert_num * 2 + nth_down * expert_num tasks
    int nth_gate_up = T::recommended_nth(config_.hidden_size);
    int nth_down = T::recommended_nth(config_.intermediate_size);

    // Phase 1: gate + up backward (both have same dimensions)
    // gate/up_proj: [intermediate_size, hidden_size] -> transposed BufferB [hidden_size, intermediate_size]
    pool->do_work_stealing_job(
        nth_gate_up * config_.expert_num * 2, nullptr,
        [this, nth_gate_up](int task_id) {
          int proj_idx = task_id / (nth_gate_up * config_.expert_num);  // 0=gate, 1=up
          int remaining = task_id % (nth_gate_up * config_.expert_num);
          int expert_idx = remaining / nth_gate_up;
          int ith = remaining % nth_gate_up;

          const ggml_bf16_t* src =
              (proj_idx == 0) ? (const ggml_bf16_t*)config_.gate_proj : (const ggml_bf16_t*)config_.up_proj;
          auto& dst_bb = (proj_idx == 0) ? gate_backward_bb_[expert_idx] : up_backward_bb_[expert_idx];

          // source: [intermediate_size, hidden_size], target: [hidden_size, intermediate_size]
          size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;
          dst_bb->from_mat_transposed((ggml_bf16_t*)(src + expert_offset), config_.intermediate_size,
                                      config_.hidden_size, ith, nth_gate_up);
        },
        nullptr, "bwd_prep_gate_up");

    // Phase 2: down backward
    // down_proj: [hidden_size, intermediate_size] -> transposed BufferB [intermediate_size, hidden_size]
    pool->do_work_stealing_job(
        nth_down * config_.expert_num, nullptr,
        [this, nth_down](int task_id) {
          int expert_idx = task_id / nth_down;
          int ith = task_id % nth_down;

          const ggml_bf16_t* src = (const ggml_bf16_t*)config_.down_proj;
          // source: [hidden_size, intermediate_size], target: [intermediate_size, hidden_size]
          size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;
          down_backward_bb_[expert_idx]->from_mat_transposed((ggml_bf16_t*)(src + expert_offset), config_.hidden_size,
                                                             config_.intermediate_size, ith, nth_down);
        },
        nullptr, "bwd_prep_down");

    backward_weights_prepared_ = true;
  }

  /**
   * @brief Dynamically repack backward BufferB from forward weights using to_mat() + from_mat_transposed().
   * Used in share_backward_bb mode (Mode 1) to avoid persistent backward_bb_pool_ per instance.
   */
  void prepare_backward_weights_from_forward() {
    if constexpr (!supports_standard_mat_mul_v<T>) return;

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Phase 1: gate + up (both use [intermediate_size, hidden_size] -> [hidden_size, intermediate_size])
    pool->do_work_stealing_job(
        config_.expert_num * 2, nullptr,
        [this](int task_id) {
          int proj = task_id / config_.expert_num;
          int expert_idx = task_id % config_.expert_num;
          auto& src_bb = (proj == 0) ? gate_bb_[expert_idx] : up_bb_[expert_idx];
          auto& dst_bb = (proj == 0) ? gate_backward_bb_[expert_idx] : up_backward_bb_[expert_idx];

          if constexpr (has_bb_transposed_repack_v<T>) {
            int nth = T::recommended_nth(dst_bb->n);
            for (int p = 0; p < nth; p++)
              dst_bb->from_bb_transposed(*src_bb, p, nth);
          } else {
            thread_local std::vector<ggml_bf16_t> workspace;
            workspace.resize((size_t)src_bb->n * src_bb->k);
            int src_nth = T::recommended_nth(src_bb->n);
            for (int p = 0; p < src_nth; p++)
              src_bb->to_mat(workspace.data(), p, src_nth);
            int dst_nth = T::recommended_nth(dst_bb->n);
            for (int p = 0; p < dst_nth; p++)
              dst_bb->from_mat_transposed(workspace.data(), src_bb->n, src_bb->k, p, dst_nth);
          }
        },
        nullptr, "bwd_repack_gate_up");

    // Phase 2: down (uses [hidden_size, intermediate_size] -> [intermediate_size, hidden_size])
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this](int task_id) {
          auto& src_bb = down_bb_[task_id];
          auto& dst_bb = down_backward_bb_[task_id];

          if constexpr (has_bb_transposed_repack_v<T>) {
            int nth = T::recommended_nth(dst_bb->n);
            for (int p = 0; p < nth; p++)
              dst_bb->from_bb_transposed(*src_bb, p, nth);
          } else {
            thread_local std::vector<ggml_bf16_t> workspace;
            workspace.resize((size_t)src_bb->n * src_bb->k);
            int src_nth = T::recommended_nth(src_bb->n);
            for (int p = 0; p < src_nth; p++)
              src_bb->to_mat(workspace.data(), p, src_nth);
            int dst_nth = T::recommended_nth(dst_bb->n);
            for (int p = 0; p < dst_nth; p++)
              dst_bb->from_mat_transposed(workspace.data(), src_bb->n, src_bb->k, p, dst_nth);
          }
        },
        nullptr, "bwd_repack_down");

    backward_weights_prepared_ = true;
  }

  /**
   * @brief Standalone method for async backward BB repack (Phase 2).
   * Called from TP_MOE_SFT::submit_backward_repack() on a separate thread.
   * Allocates/resizes the shared backward_bb pool, repacks from forward weights,
   * and sets the owner layer on the shared pool.
   */
  void prepare_backward_bb_for_async() {
    if constexpr (!supports_standard_mat_mul_v<T>) return;
    if (backward_bb_pool_bytes_ == 0) return;

    // Free any locally-allocated pool before switching to shared
    if (backward_bb_locally_owned_ && backward_bb_pool_ != nullptr) {
      free(backward_bb_pool_);
      backward_bb_pool_ = nullptr;
      backward_bb_locally_owned_ = false;
    }

    alloc_or_resize_backward_bb(backward_bb_pool_bytes_);
    backward_bb_locally_owned_ = false;
    init_backward_bb_pointers();
    backward_weights_prepared_ = false;
    prepare_backward_weights_from_forward();
    // backward_weights_prepared_ = true is set inside prepare_backward_weights_from_forward()

    auto& shared = SFTSharedPools::instance();
    shared.ensure_numa_count(tp_part_idx + 1);
    shared.pools[tp_part_idx].bwd_bb_owner_layer = config_.layer_idx;
  }

  /**
   * @brief Set base weight pointers for TP partitioning.
   * Used by TP_MOE_SFT::load_weights() to set partitioned weights before calling load_weights().
   * Unlike prepare_bwd, this does NOT call prepare_backward_weights() and does NOT reset pointers.
   */
  void set_weight_pointers_for_forward(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  /**
   * @brief Clear base weight pointers after forward path initialization.
   */
  void clear_weight_pointers() {
    config_.gate_proj = nullptr;
    config_.up_proj = nullptr;
    config_.down_proj = nullptr;
  }

  /**
   * @brief Set base weight pointers for TP partitioning (backward path).
   * Used by TP_MOE_SFT::load_weights() to set partitioned weights and prepare backward weights.
   */
  void prepare_bwd(void* gate_proj, void* up_proj, void* down_proj) {
    // If pool not yet allocated (Mode 1 init), allocate per-instance for save/load path
    if (backward_bb_pool_ == nullptr && backward_bb_pool_bytes_ > 0) {
      backward_bb_pool_ = aligned_alloc(64, backward_bb_pool_bytes_);
      init_backward_bb_pointers();
      backward_bb_locally_owned_ = true;
    }

    // Try loading pre-quantized backward weights from disk first
    if (!config_.path.empty()) {
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
      if (load_backward_weights(prefix)) {
        printf("  [BWD] Loaded pre-quantized backward weights from disk (layer %d, numa %d)\n", config_.layer_idx,
               tp_part_idx);
        return;
      }
    }

    // Fall back to online transpose + quantize
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
    prepare_backward_weights();

    // Save to disk for next time if save mode is enabled
    if (config_.save && !config_.path.empty()) {
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
      save_backward_weights(prefix);
    }

    config_.gate_proj = 0;
    config_.up_proj = 0;
    config_.down_proj = 0;
  }

  /**
   * @brief Write backward weights to disk (reuses forward weight save pattern from moe.hpp).
   */
  void write_bwd_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                         size_t scale_size) {
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (!of.is_open()) {
      printf("write_bwd_weights: cannot open file: %s\n",
             (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(size - scale_size) +
                        "Byte" + "_quant_" + ".kt"))
                 .c_str());
    }
    of.write(bb, size - scale_size);
    of.close();
    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (!of.is_open()) {
      printf("write_bwd_weights: cannot open scale file\n");
    }
    of.write(bb + (size - scale_size), scale_size);
    of.close();
  }

  /**
   * @brief Save pre-quantized backward weights to disk.
   * Must be called after prepare_backward_weights().
   */
  void save_backward_weights(const std::filesystem::path& prefix) {
    if constexpr (!supports_standard_mat_mul_v<T>) return;
    if (!backward_weights_prepared_) return;

    std::filesystem::create_directories(prefix);

    for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
      // gate_bwd: [hidden_size, intermediate_size]
      size_t gu_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      size_t gu_scale = T::BufferB::SCALE ? config_.hidden_size * sizeof(float) : 0;
      write_bwd_weights(prefix, "_gate_bwd_", (char*)gate_backward_bb_[expert_idx]->b, expert_idx, gu_size, gu_scale);
      write_bwd_weights(prefix, "_up_bwd_", (char*)up_backward_bb_[expert_idx]->b, expert_idx, gu_size, gu_scale);
      // down_bwd: [intermediate_size, hidden_size]
      size_t d_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
      size_t d_scale = T::BufferB::SCALE ? config_.intermediate_size * sizeof(float) : 0;
      write_bwd_weights(prefix, "_down_bwd_", (char*)down_backward_bb_[expert_idx]->b, expert_idx, d_size, d_scale);
    }
  }

  /**
   * @brief Load pre-quantized backward weights from disk.
   * @return true if files exist and loading succeeds, false otherwise.
   */
  bool load_backward_weights(const std::filesystem::path& prefix) {
    if constexpr (!supports_standard_mat_mul_v<T>) return false;
    if (backward_weights_prepared_) return true;

    // Check if files exist for the first expert
    size_t gu_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    size_t gu_scale = T::BufferB::SCALE ? config_.hidden_size * sizeof(float) : 0;
    std::string test_file = T::name() + "_gate_bwd_0_" + std::to_string(gu_size - gu_scale) + "Byte_quant_.kt";
    if (!std::filesystem::exists(prefix / test_file)) return false;

    size_t d_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
    size_t d_scale = T::BufferB::SCALE ? config_.intermediate_size * sizeof(float) : 0;

    // mat_class: 0=gate_bwd, 1=up_bwd, 2=down_bwd
    static constexpr int mat_type_all = 3;
    std::atomic<bool> ok{true};
    auto pool = config_.pool->get_subpool(tp_part_idx);

    auto read_one = [&](int expert_idx, const char* proj_name, char* dst_b, size_t size, size_t scale_size,
                        auto* bb_ptr /* only used when SCALE */) {
      std::ifstream f(prefix / (T::name() + proj_name + std::to_string(expert_idx) + "_" +
                                std::to_string(size - scale_size) + "Byte_quant_.kt"));
      if (!f.is_open()) {
        ok.store(false, std::memory_order_relaxed);
        return;
      }
      f.read(dst_b, size - scale_size);
      f.close();

      if constexpr (T::BufferB::SCALE) {
        f.open(prefix / (T::name() + proj_name + std::to_string(expert_idx) + "_" + std::to_string(scale_size) +
                         "Byte_scale_.kt"));
        if (!f.is_open()) {
          ok.store(false, std::memory_order_relaxed);
          return;
        }
        f.read((char*)bb_ptr->d, scale_size);
      }
    };

    pool->do_work_stealing_job(
        config_.expert_num * mat_type_all, nullptr,
        [&](int task_id) {
          if (!ok.load(std::memory_order_relaxed)) return;
          int expert_idx = task_id / mat_type_all;
          int mat_class = task_id % mat_type_all;

          if (mat_class == 0) {
            read_one(expert_idx, "_gate_bwd_", (char*)gate_backward_bb_[expert_idx]->b, gu_size, gu_scale,
                     gate_backward_bb_[expert_idx].get());
          } else if (mat_class == 1) {
            read_one(expert_idx, "_up_bwd_", (char*)up_backward_bb_[expert_idx]->b, gu_size, gu_scale,
                     up_backward_bb_[expert_idx].get());
          } else {
            read_one(expert_idx, "_down_bwd_", (char*)down_backward_bb_[expert_idx]->b, d_size, d_scale,
                     down_backward_bb_[expert_idx].get());
          }
        },
        nullptr, "load_bwd_kt");

    if (!ok.load()) return false;
    backward_weights_prepared_ = true;
    return true;
  }

  /**
   * @brief Load backward weights from pre-quantized per-NUMA buffers (memcpy path).
   * Uses gate_bwd_projs/scales etc. from GeneralMOEConfig.
   */
  void load_backward_weights_from_projs() {
    if constexpr (!supports_standard_mat_mul_v<T>) return;
    if (backward_weights_prepared_) return;

    auto pool = config_.pool->get_subpool(tp_part_idx);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this](int expert_idx) {
          const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);

          // gate_bwd: [hidden_size, intermediate_size]
          {
            size_t scale_size = T::BufferB::SCALE ? config_.hidden_size * sizeof(float) : 0;
            size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;
            memcpy(gate_backward_bb_[expert_idx]->b, config_.gate_bwd_projs[tp_part_idx][logical_expert_id], size);
            if constexpr (T::BufferB::SCALE) {
              memcpy(gate_backward_bb_[expert_idx]->d, config_.gate_bwd_scales[tp_part_idx][logical_expert_id],
                     scale_size);
            }
          }
          // up_bwd
          {
            size_t scale_size = T::BufferB::SCALE ? config_.hidden_size * sizeof(float) : 0;
            size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;
            memcpy(up_backward_bb_[expert_idx]->b, config_.up_bwd_projs[tp_part_idx][logical_expert_id], size);
            if constexpr (T::BufferB::SCALE) {
              memcpy(up_backward_bb_[expert_idx]->d, config_.up_bwd_scales[tp_part_idx][logical_expert_id], scale_size);
            }
          }
          // down_bwd: [intermediate_size, hidden_size]
          {
            size_t scale_size = T::BufferB::SCALE ? config_.intermediate_size * sizeof(float) : 0;
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;
            memcpy(down_backward_bb_[expert_idx]->b, config_.down_bwd_projs[tp_part_idx][logical_expert_id], size);
            if constexpr (T::BufferB::SCALE) {
              memcpy(down_backward_bb_[expert_idx]->d, config_.down_bwd_scales[tp_part_idx][logical_expert_id],
                     scale_size);
            }
          }
        },
        nullptr, "load_bwd_projs");

    backward_weights_prepared_ = true;
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

    // Calculate all buffer sizes (cast to size_t to prevent int overflow with large max_len)
    const size_t ml = config_.max_len;
    const size_t k_tok = config_.num_experts_per_tok;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;

    lora_intermediate_pool_bytes_ = sizeof(ggml_bf16_t) * ml * k_tok * lora_rank_;

    cache_slot_bytes_input_ = ml * H * sizeof(ggml_bf16_t);
    cache_slot_bytes_intermediate_ = ml * k_tok * I * sizeof(ggml_bf16_t);
    cache_slot_bytes_down_lora_u_ = ml * k_tok * lora_rank_ * sizeof(float);
    cache_down_output_bytes_ = (size_t)max_cache_depth_ * ml * k_tok * H * sizeof(ggml_bf16_t);

    grad_buffer_bytes_ = ml * k_tok * I * sizeof(ggml_bf16_t);

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
    size_t lora_a_down_t_bb_size = 0;
    size_t lora_b_down_t_bb_size = 0;
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
      // down lora_A^T: [intermediate_size, padded_lora_rank] per expert (for backward)
      lora_a_down_t_bb_size = T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_);
      // down lora_B^T: [padded_lora_rank, hidden_size] per expert (for backward)
      lora_b_down_t_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.hidden_size);

      // Total BufferB pool size for all experts (12 matrices per expert)
      lora_bb_pool_bytes_ = config_.expert_num * (lora_a_gate_up_bb_size * 2 +    // gate_a, up_a
                                                  lora_b_gate_up_bb_size * 2 +    // gate_b, up_b
                                                  lora_a_gate_up_t_bb_size * 2 +  // gate_a^T, up_a^T
                                                  lora_b_gate_up_t_bb_size * 2 +  // gate_b^T, up_b^T
                                                  lora_a_down_bb_size +           // down_a
                                                  lora_b_down_bb_size +           // down_b
                                                  lora_a_down_t_bb_size +         // down_a^T
                                                  lora_b_down_t_bb_size);         // down_b^T

      size_t raw_total_tokens = (size_t)config_.max_len * config_.num_experts_per_tok;
      size_t safe_alloc_tokens = raw_total_tokens + (config_.expert_num * M_STEP);

      // Ensure global alignment too
      safe_alloc_tokens = ((safe_alloc_tokens + M_STEP - 1) / M_STEP) * M_STEP;

      // Add extra bytes for "align64" calls inside the loops (64 bytes per expert per buffer)
      size_t align_overhead = config_.expert_num * 64;

      // BufferA for LoRA intermediate: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_ba_size = T::BufferA::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_ba_pool_bytes_ = T::BufferA::required_size(safe_alloc_tokens, padded_lora_rank_) * 2 + align_overhead * 2;

      // BufferC for LoRA step 1 output: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bc_size = T::BufferC::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_bc_inter_pool_bytes_ =
          T::BufferC::required_size(safe_alloc_tokens, padded_lora_rank_) * 2 + align_overhead * 2;

      // BufferC for LoRA step 2 output (gate, up, down): shared pool for all activated experts
      lora_gate_up_out_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      lora_down_out_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);           // per-expert size
      // Note: bc_out needs space for Gate, Up AND Down
      lora_bc_out_pool_bytes_ = T::BufferC::required_size(safe_alloc_tokens, config_.intermediate_size) * 2 +
                                T::BufferC::required_size(safe_alloc_tokens, config_.hidden_size) + align_overhead * 3;

      // BF16 intermediate buffer for step 1 -> step 2 conversion
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bf16_pool_bytes_ =
          safe_alloc_tokens * padded_lora_rank_ * sizeof(ggml_bf16_t) * 2 + align_overhead * 2;

      // =====================================================
      // Calculate Backward pass AMX buffer sizes
      // =====================================================
      // BufferA for scattered grad_output: shared pool for all activated experts
      grad_output_ba_size = T::BufferA::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_ba_pool_bytes_ = T::BufferA::required_size(safe_alloc_tokens, config_.hidden_size) + align_overhead;

      // BufferC for backward GEMM outputs: shared pool for all activated experts
      // grad_intermediate: [safe_alloc_tokens, intermediate_size]
      grad_intermediate_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      // grad_gate_up: [safe_alloc_tokens, hidden_size]
      grad_gate_up_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_bc_pool_bytes_ = T::BufferC::required_size(safe_alloc_tokens, config_.intermediate_size) +
                                T::BufferC::required_size(safe_alloc_tokens, config_.hidden_size) + align_overhead * 2;

      // BF16 buffer for scattered grad_output
      grad_output_bf16_pool_bytes_ = safe_alloc_tokens * config_.hidden_size * sizeof(ggml_bf16_t) + align_overhead;

      // Duplicate pools for up-path in merged gate+up GEMM
      // BufferA: intermediate_size dimension (same as down_ba_)
      gu_up_ba_pool_bytes_ = T::BufferA::required_size(safe_alloc_tokens, config_.intermediate_size) + align_overhead;
      // BufferC: hidden_size dimension (same as grad_gate_up_bc_)
      gu_up_bc_pool_bytes_ = T::BufferC::required_size(safe_alloc_tokens, config_.hidden_size) + align_overhead;
      // BF16: hidden_size dimension (same as grad_output_bf16_)
      gu_up_bf16_pool_bytes_ = safe_alloc_tokens * config_.hidden_size * sizeof(ggml_bf16_t) + align_overhead;

      // FP32 accumulators for gate+up LoRA grad_B: 2 × expert_num × intermediate_size × lora_rank
      gu_gradb_accum_pool_bytes_ =
          2 * static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_ * sizeof(float) +
          align_overhead;

      // LoRA gradient computation FP32 pools (used in bwd_down_lora_precompute and grad computation)
      // Total tokens across all activated experts = safe_alloc_tokens
      lora_grad_out_pool_bytes_ = safe_alloc_tokens * config_.hidden_size * sizeof(float) + align_overhead;
      lora_inter_proj_pool_bytes_ = safe_alloc_tokens * lora_rank_ * sizeof(float) + align_overhead;
      lora_grad_times_b_pool_bytes_ = safe_alloc_tokens * lora_rank_ * sizeof(float) + align_overhead;
      down_lora_grad_b_accum_pool_bytes_ =
          static_cast<size_t>(config_.expert_num) * config_.hidden_size * lora_rank_ * sizeof(float) + align_overhead;
      down_lora_grad_a_accum_pool_bytes_ =
          static_cast<size_t>(config_.expert_num) * config_.intermediate_size * lora_rank_ * sizeof(float) +
          align_overhead;

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
      gu_up_ba_pool_bytes_ = 0;
      gu_up_bc_pool_bytes_ = 0;
      gu_up_bf16_pool_bytes_ = 0;
      gu_gradb_accum_pool_bytes_ = 0;
      backward_bb_pool_bytes_ = 0;
      lora_grad_out_pool_bytes_ = 0;
      lora_inter_proj_pool_bytes_ = 0;
      lora_grad_times_b_pool_bytes_ = 0;
      down_lora_grad_b_accum_pool_bytes_ = 0;
      down_lora_grad_a_accum_pool_bytes_ = 0;
    }

    // ★ Bug #18 fix: Cache buffers use aligned_alloc instead of shared_mem_buffer_numa ★
    // The base class AMX_MOE_BASE::init() also calls shared_mem_buffer_numa.alloc(), and
    // SharedMemBuffer is designed to let multiple callers share the same memory pool.
    // This causes cache buffers to overlap with base class buffers like m_local_gate_output_,
    // which corrupts the cache when apply_activation() writes to m_local_gate_output_.
    // Solution: Use aligned_alloc for cache pools so they have dedicated memory.

    // ★ seqlen-dependent buffers are allocated on-demand ★
    // Forward/cache buffers are pooled to avoid frequent alloc/free overhead:
    // - Cache buffers: allocated in forward_sft() when save_for_backward=true (kept in per-instance cache_pool_)
    // - LoRA working buffers (ba/bc/bf16): allocated in forward_sft() (kept in shared forward_pool_)
    // - Backward working buffers: allocated in backward() (kept in shared backward_pool_)
    //
    // Only persistent buffers are allocated here:
    // - lora_bb_pool_: LoRA weights in BufferB format (not seqlen-dependent)
    // - backward_bb_pool_: transposed base weights in BufferB format (not seqlen-dependent)

    MemoryRequest mem_requests;

    // LoRA buffers (legacy, kept for compatibility) - still uses SharedMemBuffer
    mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

    // LoRA BB pool (persistent - stores converted LoRA weights, not seqlen-dependent)
    if (lora_bb_pool_bytes_ > 0) {
      lora_bb_pool_ = aligned_alloc(64, lora_bb_pool_bytes_);
    }

    // ★ Backward pass working buffers are allocated on-demand in backward() and freed after use ★
    // This saves memory when not training (inference mode).
    // backward_ba_pool_, backward_bc_pool_, grad_output_bf16_pool_ are allocated at the start of backward()
    // and freed at the end.
    //
    // backward_bb_pool_ is different: it stores transposed base weights (BufferB format) that need to be
    // initialized once and persist. So it's allocated here in the constructor.
    // In share_backward_bb mode (Mode 1), skip per-instance allocation — backward() will use a shared pool
    // and dynamically repack from forward weights each step.
    if (config_.share_backward_bb) {
      backward_bb_pool_ = nullptr;
      backward_bb_locally_owned_ = false;
    } else {
      if (backward_bb_pool_bytes_ > 0) {
        backward_bb_pool_ = aligned_alloc(64, backward_bb_pool_bytes_);
      }
      backward_bb_locally_owned_ = true;
    }

    // Single allocation for remaining buffers (only lora_intermediate_pool_ uses SharedMemBuffer now)
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    // Initialize LoRA pointer (only lora_intermediate_pool_ is allocated via SharedMemBuffer)
    lora_intermediate_ = (ggml_bf16_t*)lora_intermediate_pool_;
    // Note: grad_intermediate_, grad_gate_output_, grad_up_output_ are set in alloc_backward_buffers()

    // Initialize cache stack (only vectors, pointers are set in alloc_forward_buffers())
    cache_stack_.resize(max_cache_depth_);
    // Preallocate cache offsets to avoid heap allocation in hot path
    cache_offsets_.resize(config_.expert_num + 1);
    for (int i = 0; i < max_cache_depth_; i++) {
      // Note: cache pointers (input_cache, gate_output_cache, etc.) are set in alloc_forward_buffers()
      cache_stack_[i].input_cache = nullptr;
      cache_stack_[i].gate_output_cache = nullptr;
      cache_stack_[i].up_output_cache = nullptr;
      cache_stack_[i].intermediate_cache = nullptr;
      cache_stack_[i].down_output_cache = nullptr;
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
                            lora_a_down_t_bb_size, lora_b_down_t_bb_size,
                            lora_intermediate_ba_size, lora_intermediate_bc_size, lora_gate_up_out_bc_size,
                            lora_down_out_bc_size, grad_output_ba_size, grad_intermediate_bc_size, grad_gate_up_bc_size,
                            gate_up_backward_bb_size, down_backward_bb_size);
    }

    // Pool logger: static allocation summary (printed once per instance at init)
    SFT_POOL_LOG("init_static", config_.layer_idx, tp_part_idx, config_.max_len, 0,
                 lora_bb_pool_bytes_, backward_bb_pool_bytes_, 0, backward_bb_pool_bytes_ + lora_bb_pool_bytes_,
                 "static_alloc: expert_num=%d hidden=%d inter=%d lora_bb=%.2fGB bwd_bb=%.2fGB",
                 config_.expert_num, config_.hidden_size, config_.intermediate_size,
                 lora_bb_pool_bytes_ / 1024.0 / 1024.0 / 1024.0,
                 backward_bb_pool_bytes_ / 1024.0 / 1024.0 / 1024.0);
  }

  /**
   * @brief Initialize LoRA AMX buffer objects (including backward pass buffers).
   */
  void init_lora_amx_buffers(size_t max_m, size_t lora_a_gate_up_bb_size, size_t lora_b_gate_up_bb_size,
                             size_t lora_a_gate_up_t_bb_size, size_t lora_b_gate_up_t_bb_size,
                             size_t lora_a_down_bb_size, size_t lora_b_down_bb_size,
                             size_t lora_a_down_t_bb_size, size_t lora_b_down_t_bb_size,
                             size_t lora_intermediate_ba_size,
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
    down_lora_a_t_bb_.resize(config_.expert_num);
    down_lora_b_t_bb_.resize(config_.expert_num);
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
    // Up-path duplicates for merged gate+up GEMM
    down_ba_up_.resize(config_.expert_num);
    grad_gate_up_bc_up_.resize(config_.expert_num);
    grad_output_bf16_up_ptr_.resize(config_.expert_num);

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

      down_lora_a_t_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_a_down_t_bb_size;

      down_lora_b_t_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
      bb_ptr += lora_b_down_t_bb_size;

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

      // Up-path duplicates for merged gate+up GEMM
      down_ba_up_[i] = std::make_shared<typename T::BufferA>(max_m, config_.intermediate_size, nullptr);
      grad_gate_up_bc_up_[i] = std::make_shared<typename T::BufferC>(max_m, config_.hidden_size, nullptr);
      grad_output_bf16_up_ptr_[i] = nullptr;
    }

    // =====================================================
    // Initialize backward BufferB objects (transposed base weights)
    // =====================================================
    if (backward_bb_pool_ != nullptr) {
      init_backward_bb_pointers();
    }
    // If nullptr (Mode 1 at init), vectors stay with nullptr shared_ptrs — safe.

    lora_weights_prepared_ = false;
    lora_backward_weights_prepared_ = false;
    backward_weights_prepared_ = false;
  }

  /**
   * @brief Point backward BufferB objects at the current backward_bb_pool_.
   * Requires backward_bb_pool_ != nullptr and backward_bb_pool_bytes_ > 0.
   */
  void init_backward_bb_pointers() {
    size_t gate_up_backward_bb_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
    size_t down_backward_bb_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);

    char* backward_bb_ptr = (char*)backward_bb_pool_;
    for (int i = 0; i < config_.expert_num; i++) {
      gate_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      up_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      down_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, (void*)backward_bb_ptr);
      backward_bb_ptr += down_backward_bb_size;
    }
  }

  /**
   * @brief Get thread-local buffer for LoRA weight conversion.
   *
   * Uses thread_local storage to avoid repeated memory allocation.
   * The buffer is resized only when a larger size is needed.
   */
  static ggml_bf16_t* get_lora_convert_buffer(size_t required_size) {
    thread_local std::vector<ggml_bf16_t> tl_buffer;
    if (tl_buffer.size() < required_size) {
      tl_buffer.resize(required_size);
    }
    return tl_buffer.data();
  }

  /**
   * @brief Get thread-local FP32 buffer for LoRA intermediate results.
   *
   * Used by AVX512 LoRA computation to store intermediate FP32 values.
   */
  static float* get_lora_fp32_buffer(size_t required_size) {
    thread_local std::vector<float> tl_fp32_buffer;
    if (tl_fp32_buffer.size() < required_size) {
      tl_fp32_buffer.resize(required_size);
    }
    return tl_fp32_buffer.data();
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
    // Use thread-local buffer to avoid allocation
    size_t buf_size = static_cast<size_t>(dst_n) * dst_k;
    ggml_bf16_t* padded = get_lora_convert_buffer(buf_size);

    // Zero-initialize the buffer
    const ggml_bf16_t zero = GGML_FP32_TO_BF16(0.0f);
    std::fill(padded, padded + buf_size, zero);

    // Copy source data (with potential padding)
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
      dst_bb->from_mat(padded, ith, num_n_blocks);
    }
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
    // Use thread-local buffer to avoid allocation
    size_t buf_size = static_cast<size_t>(dst_n) * dst_k;
    ggml_bf16_t* padded = get_lora_convert_buffer(buf_size);

    // Zero-initialize the buffer
    const ggml_bf16_t zero = GGML_FP32_TO_BF16(0.0f);
    std::fill(padded, padded + buf_size, zero);

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
      dst_bb->from_mat(padded, ith, num_n_blocks);
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
    // Use thread-local buffer to avoid allocation
    size_t buf_size = static_cast<size_t>(dst_n) * dst_k;
    ggml_bf16_t* padded = get_lora_convert_buffer(buf_size);

    // Zero-initialize the buffer
    const ggml_bf16_t zero = GGML_FP32_TO_BF16(0.0f);
    std::fill(padded, padded + buf_size, zero);

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
      dst_bb->from_mat(padded, ith, num_n_blocks);
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
    // Use thread-local buffer to avoid allocation
    size_t buf_size = static_cast<size_t>(dst_n) * dst_k;
    ggml_bf16_t* padded = get_lora_convert_buffer(buf_size);

    // Zero-initialize the buffer
    const ggml_bf16_t zero = GGML_FP32_TO_BF16(0.0f);
    std::fill(padded, padded + buf_size, zero);

    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int r = 0; r < src_k && r < dst_n; r++) {
      for (int i = 0; i < src_n && i < dst_k; i++) {
        padded[r * dst_k + i] = expert_src[i * src_k + r];
      }
    }

    // NOTE: from_mat with (ith, nth) only processes one N_BLOCK chunk.
    // For dst_n > N_BLOCK, we need to loop over all N_BLOCKs.
    int num_n_blocks = (dst_n + T::N_BLOCK - 1) / T::N_BLOCK;
    for (int ith = 0; ith < num_n_blocks; ith++) {
      dst_bb->from_mat(padded, ith, num_n_blocks);
    }
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
    // Bounds Check: Verify pool allocation didn't overflow
    // =====================================================
    if (is_nan_check_enabled()) {
      char* lora_ba_pool_end = (char*)lora_ba_pool_ + lora_ba_pool_bytes_;
      char* lora_bc_inter_pool_end = (char*)lora_bc_inter_pool_ + lora_bc_inter_pool_bytes_;
      char* lora_bc_out_pool_end = (char*)lora_bc_out_pool_ + lora_bc_out_pool_bytes_;
      char* lora_bf16_pool_end = (char*)lora_intermediate_bf16_pool_ + lora_intermediate_bf16_pool_bytes_;

      size_t ba_used = lora_ba_ptr - (char*)lora_ba_pool_;
      size_t bc_inter_used = lora_bc_inter_ptr - (char*)lora_bc_inter_pool_;
      size_t bc_out_used = lora_bc_out_ptr - (char*)lora_bc_out_pool_;
      size_t bf16_used = bf16_inter_ptr - (char*)lora_intermediate_bf16_pool_;

      bool overflow = false;
      if (lora_ba_ptr > lora_ba_pool_end) {
        printf(
            ANSI_BG_RED
            "[OVERFLOW L%d] lora_ba_pool: used=%zu bytes, allocated=%zu bytes, OVERFLOW by %zu bytes" ANSI_COLOR_RESET
            "\n",
            config_.layer_idx, ba_used, lora_ba_pool_bytes_, ba_used - lora_ba_pool_bytes_);
        overflow = true;
      }
      if (lora_bc_inter_ptr > lora_bc_inter_pool_end) {
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] lora_bc_inter_pool: used=%zu bytes, allocated=%zu bytes, OVERFLOW by %zu "
               "bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, bc_inter_used, lora_bc_inter_pool_bytes_, bc_inter_used - lora_bc_inter_pool_bytes_);
        overflow = true;
      }
      if (lora_bc_out_ptr > lora_bc_out_pool_end) {
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] lora_bc_out_pool: used=%zu bytes, allocated=%zu bytes, OVERFLOW by %zu "
               "bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, bc_out_used, lora_bc_out_pool_bytes_, bc_out_used - lora_bc_out_pool_bytes_);
        overflow = true;
      }
      if (bf16_inter_ptr > lora_bf16_pool_end) {
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] lora_intermediate_bf16_pool: used=%zu bytes, allocated=%zu bytes, OVERFLOW by %zu "
               "bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, bf16_used, lora_intermediate_bf16_pool_bytes_,
               bf16_used - lora_intermediate_bf16_pool_bytes_);
        overflow = true;
      }

      if (overflow) {
        // Print detailed per-expert allocation info
        printf("[OVERFLOW DEBUG L%d] activated_expert=%d, M_STEP=%zu, padded_lora_rank=%d\n", config_.layer_idx,
               activated_expert, M_STEP, padded_lora_rank_);
        size_t sum_tokens = 0, sum_padded_tokens = 0;
        for (int task_id = 0; task_id < activated_expert; task_id++) {
          int expert_idx = m_expert_id_map_[task_id];
          int m = m_local_num_[expert_idx];
          if (m > 0) {
            size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;
            sum_tokens += m;
            sum_padded_tokens += local_max_m;
            printf("  expert=%d tokens=%d padded=%zu\n", expert_idx, m, local_max_m);
          }
        }
        printf("[OVERFLOW DEBUG L%d] sum_tokens=%zu, sum_padded_tokens=%zu, padding_overhead=%zu\n", config_.layer_idx,
               sum_tokens, sum_padded_tokens, sum_padded_tokens - sum_tokens);
        printf("[OVERFLOW DEBUG L%d] config: max_len=%d, num_experts_per_tok=%d, expert_num=%d\n", config_.layer_idx,
               config_.max_len, config_.num_experts_per_tok, config_.expert_num);
        printf("[OVERFLOW DEBUG L%d] expected raw_total_tokens=%zu, safe_alloc_tokens estimate=%zu\n", config_.layer_idx,
               (size_t)config_.max_len * config_.num_experts_per_tok,
               (size_t)config_.max_len * config_.num_experts_per_tok + config_.expert_num * (size_t)M_STEP);
      }

      // Always print summary for debugging token distribution
      size_t sum_tokens = 0, max_expert_tokens = 0;
      int max_expert_idx = -1;
      for (int task_id = 0; task_id < activated_expert; task_id++) {
        int expert_idx = m_expert_id_map_[task_id];
        int m = m_local_num_[expert_idx];
        sum_tokens += m;
        if ((size_t)m > max_expert_tokens) {
          max_expert_tokens = m;
          max_expert_idx = expert_idx;
        }
      }
      // Check if any single expert has extremely high token count
      size_t expected_per_expert = sum_tokens / (activated_expert > 0 ? activated_expert : 1);
      if (max_expert_tokens > expected_per_expert * 10 && max_expert_tokens > 1000) {
        printf(ANSI_COLOR_YELLOW
               "[WARN L%d] Expert %d has %zu tokens (%.1fx average), activated_expert=%d, total=%zu" ANSI_COLOR_RESET
               "\n",
               config_.layer_idx, max_expert_idx, max_expert_tokens,
               (double)max_expert_tokens / (expected_per_expert > 0 ? expected_per_expert : 1), activated_expert,
               sum_tokens);
      }
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
    // Bounds Check: Verify pool allocation didn't overflow (gate+up+down)
    // =====================================================
    if (is_nan_check_enabled()) {
      char* lora_bc_out_pool_end = (char*)lora_bc_out_pool_ + lora_bc_out_pool_bytes_;
      size_t bc_out_used = lora_down_bc_ptr - (char*)lora_bc_out_pool_;

      if (lora_down_bc_ptr > lora_bc_out_pool_end) {
        printf(ANSI_BG_RED
               "[OVERFLOW L%d] lora_bc_out_pool (gate+up+down): used=%zu bytes, allocated=%zu bytes, OVERFLOW by %zu "
               "bytes" ANSI_COLOR_RESET "\n",
               config_.layer_idx, bc_out_used, lora_bc_out_pool_bytes_, bc_out_used - lora_bc_out_pool_bytes_);
        printf("[OVERFLOW DEBUG L%d] gate_up_total=%zu bytes\n", config_.layer_idx, gate_up_total);
      }
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
   * @brief Compute LoRA for gate and up projections (AVX512 BF16 optimized).
   *
   * gate_lora_out = (input @ gate_lora_A^T) @ gate_lora_B^T * scaling
   * gate_output += gate_lora_out
   * (similar for up)
   *
   * Optimized with:
   * - Native _mm512_dpbf16_ps for BF16 dot-accumulate (no BF16->FP32 conversion)
   * - Token-blocking (T_BLOCK=4): process 4 tokens per weight load
   * - Rank-blocking (R_BLOCK=4): process 4 ranks in parallel
   * - Arithmetic intensity: 2.0 FLOP/byte
   */
  void compute_lora_gate_up(int qlen, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int rank = lora_rank_;
    const float scale = lora_scaling_;
    const int nth = 2;

    pool->do_work_stealing_job(
        activated_expert * 2 * nth, nullptr,
        [this, hidden, inter_size, rank, scale, nth](int task_id) {
          bool do_up = (task_id / nth) % 2;
          int expert_task = task_id / (2 * nth);
          int ith = task_id % nth;
          int expert_idx = m_expert_id_map_[expert_task];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Divide tokens among threads
          int tokens_per_thread = (num_tokens + nth - 1) / nth;
          int t_start = ith * tokens_per_thread;
          int t_end = std::min(t_start + tokens_per_thread, num_tokens);
          if (t_start >= num_tokens) return;

          // Get weight pointers
          ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          ggml_bf16_t* lora_b_t = do_up ? up_lora_b_transposed_ : gate_lora_b_transposed_;
          ggml_bf16_t* input = m_local_input_ptr_[expert_idx];
          ggml_bf16_t* output = do_up ? m_local_up_output_ptr_[expert_idx] : m_local_gate_output_ptr_[expert_idx];

          if (lora_a == nullptr || lora_b_t == nullptr) return;

          size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
          // Transposed layout: [expert_num][rank][intermediate_size]
          size_t lora_b_t_offset = expert_idx * lora_rank_ * config_.intermediate_size;
          ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          ggml_bf16_t* expert_lora_b_t = lora_b_t + lora_b_t_offset;

          int local_num_tokens = t_end - t_start;
          float* local_intermediate = get_lora_fp32_buffer(local_num_tokens * rank);

          // Step 1: intermediate = input @ lora_A^T (optimized with T_BLOCK=4, R_BLOCK=4)
          avx::lora_bf16_matmul_t4r4(input + t_start * hidden,  // input for this thread's tokens
                                     expert_lora_a,             // lora_A weight [rank, hidden]
                                     local_intermediate,        // output [local_num_tokens, rank]
                                     local_num_tokens, hidden, rank);

          // Step 2: output += scale * (intermediate @ lora_B_transposed)
          // Using optimized kernel with pre-transposed weight layout [rank][inter_size]
          avx::lora_fp32_bf16_fused_add_transposed(
              local_intermediate,             // intermediate [local_num_tokens, rank]
              expert_lora_b_t,                // lora_B transposed [rank, inter_size]
              output + t_start * inter_size,  // output [local_num_tokens, inter_size]
              local_num_tokens, rank, inter_size, scale);
        },
        nullptr, "fwd_lora_gu");
  }

  /**
   * @brief Compute LoRA for down projection (AVX512 BF16 optimized).
   *
   * Optimized with:
   * - Native _mm512_dpbf16_ps for BF16 dot-accumulate (no BF16->FP32 conversion)
   * - Token-blocking (T_BLOCK=4): process 4 tokens per weight load
   * - Rank-blocking (R_BLOCK=4): process 4 ranks in parallel
   * - Arithmetic intensity: 2.0 FLOP/byte
   */
  void compute_lora_down(int qlen, int activated_expert, ForwardCache* cache = nullptr) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (down_lora_a_ == nullptr || down_lora_b_ == nullptr) return;

    const int inter_size = config_.intermediate_size;
    const int hidden = config_.hidden_size;
    const int rank = lora_rank_;
    const float scale = lora_scaling_;
    const int nth = 2;

    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [this, cache, inter_size, hidden, rank, scale, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int num_tokens = m_local_num_[expert_idx];
          if (num_tokens == 0) return;

          int tokens_per_thread = (num_tokens + nth - 1) / nth;
          int t_start = ith * tokens_per_thread;
          int t_end = std::min(t_start + tokens_per_thread, num_tokens);
          if (t_start >= num_tokens) return;

          ggml_bf16_t* input = m_local_gate_output_ptr_[expert_idx];
          ggml_bf16_t* output = m_local_down_output_ptr_[expert_idx];
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
          // Transposed layout: [expert_num][rank][hidden_size]
          size_t lora_b_t_offset = expert_idx * lora_rank_ * config_.hidden_size;
          ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
          ggml_bf16_t* expert_lora_b_t = down_lora_b_transposed_ + lora_b_t_offset;

          int local_num_tokens = t_end - t_start;
          float* local_intermediate = get_lora_fp32_buffer(local_num_tokens * rank);

          // Step 1: intermediate = input @ lora_A^T (optimized with T_BLOCK=4, R_BLOCK=4)
          avx::lora_bf16_matmul_t4r4(input + t_start * inter_size,  // input for this thread's tokens
                                     expert_lora_a,                 // lora_A weight [rank, inter_size]
                                     local_intermediate,            // output [local_num_tokens, rank]
                                     local_num_tokens, inter_size, rank);

          if (cache != nullptr && cache->down_lora_u_cache != nullptr) {
            float* cache_u = cache->down_lora_u_cache + (cache_offsets_[task_id / nth] + t_start) * rank;
            memcpy(cache_u, local_intermediate, static_cast<size_t>(local_num_tokens) * rank * sizeof(float));
          }

          // Step 2: output += scale * (intermediate @ lora_B_transposed)
          // Using optimized kernel with pre-transposed weight layout [rank][hidden]
          avx::lora_fp32_bf16_fused_add_transposed(local_intermediate,         // intermediate [local_num_tokens, rank]
                                                   expert_lora_b_t,            // lora_B transposed [rank, hidden]
                                                   output + t_start * hidden,  // output [local_num_tokens, hidden]
                                                   local_num_tokens, rank, hidden, scale);
        },
        nullptr, "fwd_lora_down");
  }

  ForwardCache& push_cache() {
    if (cache_stack_top_ >= max_cache_depth_) {
      // std::cerr << "[KT-MOE ERROR] Forward cache stack overflow!" << std::endl;
      // std::cerr << "  cache_stack_top_ = " << cache_stack_top_ << std::endl;
      // std::cerr << "  max_cache_depth_ = " << max_cache_depth_ << std::endl;
      // std::cerr << "  Hint: If you are doing inference (forward only without backward)," << std::endl;
      // std::cerr << "        set save_for_backward=False in forward_sft() call." << std::endl;
      // std::cerr << "        Or increase max_cache_depth in MOESFTConfig." << std::endl;
      // throw std::runtime_error("Forward cache stack overflow");
      cache_stack_top_ = 0;  // Wrap around (for inference only)
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
    auto pool = config_.pool->get_subpool(tp_part_idx);

    cache.qlen_cache = qlen;
    cache.k_cache = k;
    cache.activated_expert_cache = activated_expert;

    // Copy routing information (small data, keep serial)
    cache.expert_ids_cache.resize(qlen * k);
    cache.weights_cache.resize(qlen * k);
    std::copy(expert_ids, expert_ids + qlen * k, cache.expert_ids_cache.begin());
    std::copy(weights, weights + qlen * k, cache.weights_cache.begin());

    cache.m_local_num_cache = m_local_num_;
    // Optimized: use memcpy for inner vector instead of scalar loop
    for (int i = 0; i < qlen; i++) {
      memcpy(cache.m_local_pos_cache[i].data(), m_local_pos_[i].data(), k * sizeof(int));
    }
    for (int i = 0; i < activated_expert; i++) {
      cache.m_expert_id_map_cache[i] = m_expert_id_map_[i];
    }

    // Compute offsets using preallocated buffer (avoid heap allocation)
    cache_offsets_[0] = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      cache_offsets_[i + 1] = cache_offsets_[i] + m_local_num_[expert_idx];
    }

    // Parallel copy: input(1 task) + gate(N tasks) + up(N tasks) = 1 + 2N tasks
    // This parallelizes the ~1.8MB input copy that was previously serial
    int total_tasks = 1 + activated_expert * 2;
    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [this, &cache, input, qlen, activated_expert](int task_id) {
          if (task_id == 0) {
            // Task 0: copy input (~1.8MB for qlen=128, hidden=7168)
            memcpy(cache.input_cache, input, qlen * config_.hidden_size * sizeof(ggml_bf16_t));
          } else {
            // Tasks 1..2N: copy gate and up outputs
            int idx = task_id - 1;
            bool do_up = idx % 2;
            int i = idx / 2;
            int expert_idx = m_expert_id_map_[i];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            size_t offset = cache_offsets_[i];
            if (do_up) {
              memcpy(cache.up_output_cache + offset * config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
                     num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
            } else {
              memcpy(cache.gate_output_cache + offset * config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
                     num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
            }
          }
        },
        nullptr, "save_cache");

    cache.valid = true;
  }

  /**
   * @brief Save intermediate values AFTER activation for backward_down.
   *
   * Must be called after apply_activation() since m_local_gate_output_ptr_
   * now contains silu(gate) * up (the intermediate value).
   *
   * Note: Uses cache_offsets_ computed by save_to_cache() - must be called after it.
   */
  void save_intermediate_to_cache(ForwardCache& cache, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel memcpy (reuse cache_offsets_ from save_to_cache)
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache](int i) {
          int expert_idx = m_expert_id_map_[i];
          int num_tokens = m_local_num_[expert_idx];
          if (num_tokens == 0) return;
          // m_local_gate_output_ptr_ now contains intermediate (after activation: silu(gate) * up)
          memcpy(cache.intermediate_cache + cache_offsets_[i] * config_.intermediate_size,
                 m_local_gate_output_ptr_[expert_idx], num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
        },
        nullptr, "save_inter_cache");
  }

  /**
   * @brief Save down projection output for grad_weights computation.
   *
   * Must be called after down projection (and LoRA) but before weighted merge.
   *
   * Note: Uses cache_offsets_ computed by save_to_cache() - must be called after it.
   */
  void save_down_output_to_cache(ForwardCache& cache, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Expert-level parallelism: each task copies one expert's contiguous data block
    // This maintains memory locality and cache efficiency
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache](int i) {
          int expert_idx = m_expert_id_map_[i];
          int num_tokens = m_local_num_[expert_idx];
          if (num_tokens == 0) return;
          ggml_bf16_t* src_ptr = m_local_down_output_ptr_[expert_idx];
          memcpy(cache.down_output_cache + cache_offsets_[i] * config_.hidden_size, src_ptr,
                 num_tokens * config_.hidden_size * sizeof(ggml_bf16_t));
        },
        nullptr, "save_down_cache");
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
    // Initialize gradient intermediate buffer (parallelized)
    {
      size_t total_size =
          (size_t)config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);
      const int num_chunks = 8;
      size_t chunk_size = (total_size + num_chunks - 1) / num_chunks;
      pool->do_work_stealing_job(
          num_chunks, nullptr,
          [this, total_size, chunk_size](int i) {
            size_t offset = i * chunk_size;
            size_t size = std::min(chunk_size, total_size - offset);
            if (size > 0) {
              memset(reinterpret_cast<char*>(grad_intermediate_) + offset, 0, size);
            }
          },
          nullptr, "bwd_down_memset");
    }

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
                         void* grad_down_lora_b,
                         int full_intermediate_size = 0,
                         float* fp32_grad_down_lora_b = nullptr) {
    if (full_intermediate_size == 0) full_intermediate_size = config_.intermediate_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;
    constexpr int kSmallBwdDirectQlen = 0;
    constexpr int kSmallBwdDirectMaxTasks = 16;
    auto direct_or_pool = [&](int count, auto&& fn, const char* task_name, int block_size = 1) {
      if (qlen <= kSmallBwdDirectQlen && count <= kSmallBwdDirectMaxTasks) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr, task_name, block_size);
      }
    };

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Ensure backward weights are prepared
    assert(backward_weights_prepared_);

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

    // NOTE: no full-buffer memset here; grad_intermediate_ is overwritten by to_mat() for active tokens.

    // =====================================================
    // Step 1: Zero per-expert grad_output buffers
    // =====================================================
    direct_or_pool(
        activated_expert,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];
          if (num_tokens == 0) return;
          memset(grad_output_bf16_ptr_[expert_idx], 0, num_tokens * config_.hidden_size * sizeof(ggml_bf16_t));
        },
        "bwd_down_zero");

    // =====================================================
    // Step 2: Scatter grad_output to per-expert BF16 buffers
    // =====================================================
    {
      const int hidden = config_.hidden_size;
      const int hidden_vec_end = hidden & ~31;

      direct_or_pool(
          qlen,
          [this, &cache, grad_output, k, hidden, hidden_vec_end](int token_id) {
            const ggml_bf16_t* src_row = (const ggml_bf16_t*)grad_output + token_id * hidden;

            for (int j = 0; j < k; j++) {
              int expert_idx = cache.expert_ids_cache[token_id * k + j];
              if (expert_idx < config_.num_gpu_experts || expert_idx >= config_.expert_num) {
                continue;
              }
              if (m_local_num_[expert_idx] == 0) {
                continue;
              }

              // Each token-route pair owns one unique local position within an expert buffer.
              int pos = cache.m_local_pos_cache[token_id][j];
              float w = cache.weights_cache[token_id * k + j];
              ggml_bf16_t* dst_row = grad_output_bf16_ptr_[expert_idx] + pos * hidden;

              __m512 w_vec = _mm512_set1_ps(w);
              int h = 0;
              for (; h < hidden_vec_end; h += 32) {
                __m512 x0, x1, cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)(src_row + h), &x0, &x1);
                avx512_32xbf16_to_32xfp32((__m512i*)(dst_row + h), &cur0, &cur1);
                x0 = _mm512_mul_ps(x0, w_vec);
                x1 = _mm512_mul_ps(x1, w_vec);
                x0 = _mm512_add_ps(x0, cur0);
                x1 = _mm512_add_ps(x1, cur1);
                avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)(dst_row + h));
              }
              for (; h < hidden; h++) {
                float cur = GGML_BF16_TO_FP32(dst_row[h]);
                cur += GGML_BF16_TO_FP32(src_row[h]) * w;
                dst_row[h] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          "bwd_down_scatter");
    }

    // =====================================================
    // Step 3: Quantize scattered grad_output to BufferA
    // =====================================================
    direct_or_pool(
        activated_expert,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];
          if (num_tokens == 0) return;
          grad_output_ba_[expert_idx]->from_mat(num_tokens, grad_output_bf16_ptr_[expert_idx], 0, 1);
        },
        "bwd_down_quantize");

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

    // Pre-compute offsets for each expert in both token units and BF16 matrix units.
    std::vector<size_t> expert_offsets(activated_expert);
    std::vector<size_t> expert_token_offsets(activated_expert);
    {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        expert_token_offsets[i] = offset;
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

    // =====================================================
    // Step 3.5: Add LoRA contribution to grad_intermediate (AVX512)
    // grad_intermediate += grad_output @ down_lora_B @ down_lora_A * scaling
    // This is needed for correct backward through activation to gate/up
    // =====================================================
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr && down_lora_b_transposed_ != nullptr) {
      const int hidden = config_.hidden_size;
      const int inter_size = config_.intermediate_size;
      const int rank = lora_rank_;
      const float scale = lora_scaling_;
      const int nth = 4;

      direct_or_pool(
          nth * activated_expert,
          [this, &expert_offsets, &expert_token_offsets, hidden, inter_size, rank, scale, nth](int task_id) {
            int expert_idx = m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            // Divide tokens among threads
            int tokens_per_thread = (num_tokens + nth - 1) / nth;
            int t_start = ith * tokens_per_thread;
            int t_end = std::min(t_start + tokens_per_thread, num_tokens);
            if (t_start >= num_tokens) return;

            // Get expert's LoRA weights (use transposed layout for lora_B)
            size_t lora_a_offset = (size_t)expert_idx * rank * inter_size;
            size_t lora_b_t_offset = (size_t)expert_idx * rank * hidden;  // Transposed: [rank, hidden]
            const ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            const ggml_bf16_t* expert_lora_b_t = down_lora_b_transposed_ + lora_b_t_offset;
            const ggml_bf16_t* expert_grad = grad_output_bf16_ptr_[expert_idx];
            ggml_bf16_t* grad_inter = grad_intermediate_ + expert_offsets[task_id / nth];
            float* grad_times_b =
                lora_grad_times_b_pool_ + (expert_token_offsets[task_id / nth] + t_start) * rank;

            int local_num_tokens = t_end - t_start;

            // Step 1: grad_output @ down_lora_B_transposed -> [local_num_tokens, rank]
            // Using optimized kernel with transposed weight layout [rank, hidden]
            avx::lora_backward_matmul_transposed(expert_grad + t_start * hidden,  // [local_num_tokens, hidden] BF16
                                                 expert_lora_b_t,                 // [rank, hidden] BF16 (transposed)
                                                 grad_times_b,                    // [local_num_tokens, rank] FP32
                                                 local_num_tokens, hidden, rank);

            // Step 2: grad_times_b @ down_lora_A -> [local_num_tokens, inter_size] (AVX512)
            // Using optimized kernel with weight layout [rank, inter_size]
            avx::lora_fp32_bf16_fused_add_wt(grad_times_b,                       // [local_num_tokens, rank] FP32
                                             expert_lora_a,                      // [rank, inter_size] BF16
                                             grad_inter + t_start * inter_size,  // [local_num_tokens, inter_size] BF16
                                             local_num_tokens, rank, inter_size, scale);
          },
          "bwd_down_lora_to_inter");
    }

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
    if (!SkipLoRA) {
      struct LoraGradBuf {
        int expert_idx = -1;
        int num_tokens = 0;
        size_t lora_a_offset = 0;  // copy-type: expert_idx * rank * full_I (stride uses full_intermediate_size)
        size_t lora_b_offset = 0;  // reduce-type: task_id * hidden * rank (sparse FP32)
        const ggml_bf16_t* cached_intermediate = nullptr;
        const float* cached_down_lora_u = nullptr;
        const ggml_bf16_t* expert_grad_bf16 = nullptr;
        const float* grad_times_b = nullptr;
      };

      const int hidden = config_.hidden_size;
      const int inter_size = config_.intermediate_size;
      const int rank = lora_rank_;
      const int down_a_row_stride = full_intermediate_size;  // full I for copy-type direct write
      const size_t grad_b_elems = static_cast<size_t>(hidden) * rank;
      const size_t grad_a_elems = static_cast<size_t>(inter_size) * rank;
      const bool use_fp32_down_b = (fp32_grad_down_lora_b != nullptr);

      std::vector<LoraGradBuf> lora_grad_bufs(activated_expert);
      int max_down_lora_tokens = 0;

      // Initialize per-expert buffers (sequential - cheaper than parallel job dispatch)
      for (int task_id = 0; task_id < activated_expert; task_id++) {
        LoraGradBuf& buf = lora_grad_bufs[task_id];
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];

        buf.expert_idx = expert_idx;
        buf.num_tokens = num_tokens;
        if (num_tokens == 0) continue;

        buf.lora_a_offset = static_cast<size_t>(expert_idx) * rank * down_a_row_stride;  // copy-type: full_I stride
        buf.lora_b_offset = use_fp32_down_b
            ? static_cast<size_t>(task_id) * hidden * rank   // sparse FP32 indexing
            : static_cast<size_t>(expert_idx) * hidden * rank;  // legacy dense BF16
        buf.expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];
        buf.grad_times_b = lora_grad_times_b_pool_ + expert_token_offsets[task_id] * rank;

        size_t token_offset = expert_token_offsets[task_id];
        buf.cached_intermediate = cache.intermediate_cache + token_offset * inter_size;
        buf.cached_down_lora_u = cache.down_lora_u_cache + token_offset * rank;
        max_down_lora_tokens = std::max(max_down_lora_tokens, num_tokens);
      }

      // Step-1 note: FP32 sparse buffers are NOT pre-zeroed.
      // First write uses '=' (overwrite), subsequent writes use '+=' (accumulate).

      const float scale = lora_scaling_;
      constexpr int kDownGradABBlockedThreshold = 4096;

      if (max_down_lora_tokens >= kDownGradABBlockedThreshold) {
        struct GradABBlockTask {
          int expert_task = -1;
          int start = 0;
          int end = 0;
          bool is_grad_b = false;
        };
        constexpr int kDownGradBTile = 256;
        constexpr int kDownGradATile = 256;
        constexpr int kMaxDownGradTile = kDownGradBTile > kDownGradATile ? kDownGradBTile : kDownGradATile;
        std::vector<GradABBlockTask> grad_ab_tasks;
        grad_ab_tasks.reserve(static_cast<size_t>(activated_expert) *
                              (((hidden + kDownGradBTile - 1) / kDownGradBTile) +
                               ((inter_size + kDownGradATile - 1) / kDownGradATile)));

        for (int task_id = 0; task_id < activated_expert; task_id++) {
          const LoraGradBuf& buf = lora_grad_bufs[task_id];
          if (buf.num_tokens == 0) continue;
          for (int h = 0; h < hidden; h += kDownGradBTile) {
            grad_ab_tasks.push_back({task_id, h, std::min(h + kDownGradBTile, hidden), true});
          }
          for (int i = 0; i < inter_size; i += kDownGradATile) {
            grad_ab_tasks.push_back({task_id, i, std::min(i + kDownGradATile, inter_size), false});
          }
        }

        pool->do_work_stealing_job(
            static_cast<int>(grad_ab_tasks.size()), nullptr,
            [&, hidden, inter_size, rank, scale](int task_id) {
              const GradABBlockTask& task = grad_ab_tasks[task_id];
              const LoraGradBuf& buf = lora_grad_bufs[task.expert_task];
              if (buf.num_tokens == 0) return;

              const int block_len = task.end - task.start;
              float* accum = get_lora_fp32_buffer(static_cast<size_t>(kMaxDownGradTile) * rank);
              memset(accum, 0, static_cast<size_t>(block_len) * rank * sizeof(float));

              if (task.is_grad_b) {
                for (int t = 0; t < buf.num_tokens; t++) {
                  const ggml_bf16_t* grad_row_bf16 = buf.expert_grad_bf16 + static_cast<size_t>(t) * hidden + task.start;
                  const float* inter_proj = buf.cached_down_lora_u + static_cast<size_t>(t) * rank;

                  if (rank == 8) {
                    __m256 inter_proj_vec = _mm256_loadu_ps(inter_proj);
                    for (int hh = 0; hh < block_len; hh++) {
                      float g = GGML_BF16_TO_FP32(grad_row_bf16[hh]);
                      if (g == 0.0f) continue;
                      float* out = accum + static_cast<size_t>(hh) * rank;
                      __m256 acc = _mm256_loadu_ps(out);
                      acc = _mm256_fmadd_ps(_mm256_set1_ps(g), inter_proj_vec, acc);
                      _mm256_storeu_ps(out, acc);
                    }
                  } else {
                    for (int hh = 0; hh < block_len; hh++) {
                      float g = GGML_BF16_TO_FP32(grad_row_bf16[hh]);
                      if (g == 0.0f) continue;
                      float* out = accum + static_cast<size_t>(hh) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += g * inter_proj[r];
                      }
                    }
                  }
                }

                if (use_fp32_down_b) {
                  // Step-1 opt: '=' instead of '+=' — each (expert, h_block) written by exactly one task.
                  // No pre-zeroing needed.
                  for (int hh = 0; hh < block_len; hh++) {
                    float* fp32_out = fp32_grad_down_lora_b + buf.lora_b_offset + static_cast<size_t>(task.start + hh) * rank;
                    float* acc_row = accum + static_cast<size_t>(hh) * rank;
                    for (int r = 0; r < rank; r++) {
                      fp32_out[r] = acc_row[r] * scale;
                    }
                  }
                } else {
                  for (int hh = 0; hh < block_len; hh++) {
                    ggml_bf16_t* out = grad_down_b + buf.lora_b_offset + static_cast<size_t>(task.start + hh) * rank;
                    float* acc_row = accum + static_cast<size_t>(hh) * rank;
                    for (int r = 0; r < rank; r++) {
                      float cur = GGML_BF16_TO_FP32(out[r]);
                      cur += acc_row[r] * scale;
                      out[r] = GGML_FP32_TO_BF16(cur);
                    }
                  }
                }
              } else {
                for (int t = 0; t < buf.num_tokens; t++) {
                  const ggml_bf16_t* inter_row_bf16 =
                      buf.cached_intermediate + static_cast<size_t>(t) * inter_size + task.start;
                  const float* grad_times_b = buf.grad_times_b + static_cast<size_t>(t) * rank;

                  if (rank == 8) {
                    __m256 grad_times_b_vec = _mm256_loadu_ps(grad_times_b);
                    for (int ii = 0; ii < block_len; ii++) {
                      float x = GGML_BF16_TO_FP32(inter_row_bf16[ii]);
                      if (x == 0.0f) continue;
                      float* out = accum + static_cast<size_t>(ii) * rank;
                      __m256 acc = _mm256_loadu_ps(out);
                      acc = _mm256_fmadd_ps(_mm256_set1_ps(x), grad_times_b_vec, acc);
                      _mm256_storeu_ps(out, acc);
                    }
                  } else {
                    for (int ii = 0; ii < block_len; ii++) {
                      float x = GGML_BF16_TO_FP32(inter_row_bf16[ii]);
                      if (x == 0.0f) continue;
                      float* out = accum + static_cast<size_t>(ii) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += x * grad_times_b[r];
                      }
                    }
                  }
                }

                for (int r = 0; r < rank; r++) {
                  ggml_bf16_t* grad_row =
                      grad_down_a + buf.lora_a_offset + static_cast<size_t>(r) * down_a_row_stride + task.start;
                  float* acc_row = accum + static_cast<size_t>(r);
                  for (int ii = 0; ii < block_len; ii++) {
                    float cur = GGML_BF16_TO_FP32(grad_row[ii]);
                    cur += acc_row[static_cast<size_t>(ii) * rank] * scale;
                    grad_row[ii] = GGML_FP32_TO_BF16(cur);
                  }
                }
              }
            },
            nullptr, "bwd_down_lora_grad_AB");
      } else {
        struct LoraGradTask {
          int expert_task = -1;
          int t_start = 0;
          int t_end = 0;
        };
        std::vector<LoraGradTask> lora_grad_tasks;

        uint64_t grad_setup_start = sft_timer::get_trace_timestamp();
        float* grad_b_accum_all = down_lora_grad_b_accum_pool_;
        float* grad_a_accum_all = down_lora_grad_a_accum_pool_;
        if (activated_expert > 0) {
          std::memset(down_lora_grad_accum_initialized_.data(), 0, static_cast<size_t>(activated_expert));
        }

        int total_task_count = 0;
        for (int task_id = 0; task_id < activated_expert; task_id++) {
          const LoraGradBuf& buf = lora_grad_bufs[task_id];
          const int token_tile = buf.num_tokens <= 128 ? 32 : 64;
          total_task_count += (buf.num_tokens + token_tile - 1) / token_tile;
        }
        lora_grad_tasks.reserve(total_task_count);
        for (int task_id = 0; task_id < activated_expert; task_id++) {
          const LoraGradBuf& buf = lora_grad_bufs[task_id];
          const int token_tile = buf.num_tokens <= 128 ? 32 : 64;
          for (int t = 0; t < buf.num_tokens; t += token_tile) {
            lora_grad_tasks.push_back({task_id, t, std::min(t + token_tile, buf.num_tokens)});
          }
        }
        uint64_t grad_setup_end = sft_timer::get_trace_timestamp();
        sft_timer::add_kernel_trace("bwd_down_lora_grad_setup", grad_setup_start, grad_setup_end, tp_part_idx, 0);

        if (!lora_grad_tasks.empty()) {
          direct_or_pool(
              static_cast<int>(lora_grad_tasks.size()),
              [&, hidden, inter_size, rank, grad_b_elems, grad_a_elems](int task_id) {
                const LoraGradTask& task = lora_grad_tasks[task_id];
                LoraGradBuf& buf = lora_grad_bufs[task.expert_task];
                if (buf.num_tokens == 0) return;

                const int hidden_vec_end = hidden & ~31;
                const int inter_vec_end = inter_size & ~31;
                size_t scratch_elems = grad_b_elems + grad_a_elems + hidden + inter_size;
                float* scratch = get_lora_fp32_buffer(scratch_elems);
                float* grad_b_local = scratch;
                float* grad_a_local = grad_b_local + grad_b_elems;
                float* grad_row_fp32 = grad_a_local + grad_a_elems;
                float* inter_row_fp32 = grad_row_fp32 + hidden;

                memset(grad_b_local, 0, (grad_b_elems + grad_a_elems) * sizeof(float));

                for (int t = task.t_start; t < task.t_end; t++) {
                  const ggml_bf16_t* grad_row_bf16 = buf.expert_grad_bf16 + static_cast<size_t>(t) * hidden;
                  const ggml_bf16_t* inter_row_bf16 = buf.cached_intermediate + static_cast<size_t>(t) * inter_size;

                  int h = 0;
                  for (; h < hidden_vec_end; h += 32) {
                    __m512 g0, g1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(grad_row_bf16 + h), &g0, &g1);
                    _mm512_storeu_ps(grad_row_fp32 + h, g0);
                    _mm512_storeu_ps(grad_row_fp32 + h + 16, g1);
                  }
                  for (; h < hidden; h++) {
                    grad_row_fp32[h] = GGML_BF16_TO_FP32(grad_row_bf16[h]);
                  }

                  int i = 0;
                  for (; i < inter_vec_end; i += 32) {
                    __m512 x0, x1;
                    avx512_32xbf16_to_32xfp32((__m512i*)(inter_row_bf16 + i), &x0, &x1);
                    _mm512_storeu_ps(inter_row_fp32 + i, x0);
                    _mm512_storeu_ps(inter_row_fp32 + i + 16, x1);
                  }
                  for (; i < inter_size; i++) {
                    inter_row_fp32[i] = GGML_BF16_TO_FP32(inter_row_bf16[i]);
                  }

                  const float* inter_proj = buf.cached_down_lora_u + static_cast<size_t>(t) * rank;
                  const float* grad_times_b = buf.grad_times_b + static_cast<size_t>(t) * rank;

                  if (rank == 8) {
                    __m256 inter_proj_vec = _mm256_loadu_ps(inter_proj);
                    for (int hh = 0; hh < hidden; hh++) {
                      float g = grad_row_fp32[hh];
                      if (g == 0.0f) continue;
                      float* out = grad_b_local + static_cast<size_t>(hh) * rank;
                      __m256 acc = _mm256_loadu_ps(out);
                      acc = _mm256_fmadd_ps(_mm256_set1_ps(g), inter_proj_vec, acc);
                      _mm256_storeu_ps(out, acc);
                    }

                    __m256 grad_times_b_vec = _mm256_loadu_ps(grad_times_b);
                    for (int ii = 0; ii < inter_size; ii++) {
                      float x = inter_row_fp32[ii];
                      if (x == 0.0f) continue;
                      float* out = grad_a_local + static_cast<size_t>(ii) * rank;
                      __m256 acc = _mm256_loadu_ps(out);
                      acc = _mm256_fmadd_ps(_mm256_set1_ps(x), grad_times_b_vec, acc);
                      _mm256_storeu_ps(out, acc);
                    }
                  } else {
                    for (int hh = 0; hh < hidden; hh++) {
                      float g = grad_row_fp32[hh];
                      if (g == 0.0f) continue;
                      float* out = grad_b_local + static_cast<size_t>(hh) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += g * inter_proj[r];
                      }
                    }

                    for (int ii = 0; ii < inter_size; ii++) {
                      float x = inter_row_fp32[ii];
                      if (x == 0.0f) continue;
                      float* out = grad_a_local + static_cast<size_t>(ii) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += x * grad_times_b[r];
                      }
                    }
                  }
                }

                std::lock_guard<std::mutex> lock(down_lora_grad_mutexes_[task.expert_task]);
                float* grad_b_global = grad_b_accum_all + static_cast<size_t>(task.expert_task) * grad_b_elems;
                float* grad_a_global = grad_a_accum_all + static_cast<size_t>(task.expert_task) * grad_a_elems;
                if (!down_lora_grad_accum_initialized_[task.expert_task]) {
                  std::memcpy(grad_b_global, grad_b_local, grad_b_elems * sizeof(float));
                  std::memcpy(grad_a_global, grad_a_local, grad_a_elems * sizeof(float));
                  down_lora_grad_accum_initialized_[task.expert_task] = 1;
                } else if (rank == 8) {
                  for (size_t off = 0; off < grad_b_elems; off += rank) {
                    __m256 acc = _mm256_loadu_ps(grad_b_global + off);
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(grad_b_local + off));
                    _mm256_storeu_ps(grad_b_global + off, acc);
                  }
                  for (size_t off = 0; off < grad_a_elems; off += rank) {
                    __m256 acc = _mm256_loadu_ps(grad_a_global + off);
                    acc = _mm256_add_ps(acc, _mm256_loadu_ps(grad_a_local + off));
                    _mm256_storeu_ps(grad_a_global + off, acc);
                  }
                } else {
                  for (size_t off = 0; off < grad_b_elems; off++) {
                    grad_b_global[off] += grad_b_local[off];
                  }
                  for (size_t off = 0; off < grad_a_elems; off++) {
                    grad_a_global[off] += grad_a_local[off];
                  }
                }
              },
              "bwd_down_lora_grad_AB");

          constexpr int kDownGradBTile = 512;
          constexpr int kDownGradATile = 512;
          int grad_b_blocks = (hidden + kDownGradBTile - 1) / kDownGradBTile;
          int grad_a_blocks = (inter_size + kDownGradATile - 1) / kDownGradATile;

          // Merged: write grad_B and grad_A in one dispatch (was two separate dispatches).
          int total_write_tasks = activated_expert * (grad_b_blocks + grad_a_blocks);
          int ab_boundary = activated_expert * grad_b_blocks;
          pool->do_work_stealing_job(
              total_write_tasks, nullptr,
              [&, hidden, inter_size, rank, scale, grad_b_elems, grad_a_elems,
               grad_b_blocks, grad_a_blocks, ab_boundary, use_fp32_down_b, down_a_row_stride](int task_id) {
                if (task_id < ab_boundary) {
                  // === Write grad_B ===
                  int expert_task = task_id / grad_b_blocks;
                  int block_idx = task_id % grad_b_blocks;
                  LoraGradBuf& buf = lora_grad_bufs[expert_task];
                  if (buf.num_tokens == 0) return;

                  int h_start = block_idx * kDownGradBTile;
                  int h_end = std::min(hidden, h_start + kDownGradBTile);
                  float* grad_b_global = grad_b_accum_all + static_cast<size_t>(expert_task) * grad_b_elems;

                  if (use_fp32_down_b) {
                    if (rank == 8) {
                      fp32_scale_write_rank8(
                          fp32_grad_down_lora_b + buf.lora_b_offset + static_cast<size_t>(h_start) * rank,
                          grad_b_global + static_cast<size_t>(h_start) * rank,
                          h_end - h_start, rank, scale);
                    } else {
                      for (int hh = h_start; hh < h_end; hh++) {
                        float* fp32_out = fp32_grad_down_lora_b + buf.lora_b_offset + static_cast<size_t>(hh) * rank;
                        float* acc_row = grad_b_global + static_cast<size_t>(hh) * rank;
                        for (int r = 0; r < rank; r++) {
                          fp32_out[r] = acc_row[r] * scale;
                        }
                      }
                    }
                  } else {
                    if (rank == 8) {
                      bf16_fma_writeback_rank8(
                          grad_down_b + buf.lora_b_offset + static_cast<size_t>(h_start) * rank,
                          grad_b_global + static_cast<size_t>(h_start) * rank,
                          h_end - h_start, rank, scale);
                    } else {
                      for (int hh = h_start; hh < h_end; hh++) {
                        ggml_bf16_t* out = grad_down_b + buf.lora_b_offset + static_cast<size_t>(hh) * rank;
                        float* acc_row = grad_b_global + static_cast<size_t>(hh) * rank;
                        for (int r = 0; r < rank; r++) {
                          float cur = GGML_BF16_TO_FP32(out[r]);
                          cur += acc_row[r] * scale;
                          out[r] = GGML_FP32_TO_BF16(cur);
                        }
                      }
                    }
                  }
                } else {
                  // === Write grad_A ===
                  int local_id = task_id - ab_boundary;
                  int expert_task = local_id / grad_a_blocks;
                  int block_idx = local_id % grad_a_blocks;
                  LoraGradBuf& buf = lora_grad_bufs[expert_task];
                  if (buf.num_tokens == 0) return;

                  int i_start = block_idx * kDownGradATile;
                  int i_end = std::min(inter_size, i_start + kDownGradATile);
                  float* grad_a_global = grad_a_accum_all + static_cast<size_t>(expert_task) * grad_a_elems;

                  for (int r = 0; r < rank; r++) {
                    ggml_bf16_t* grad_row =
                        grad_down_a + buf.lora_a_offset + static_cast<size_t>(r) * down_a_row_stride + i_start;
                    int len = i_end - i_start;
                    int ii = 0;
                    // AVX512: 16 BF16 at a time, gather from strided grad_a_global
                    __m512 scale_v = _mm512_set1_ps(scale);
                    __m512i stride_idx = _mm512_mullo_epi32(
                        _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15),
                        _mm512_set1_epi32(rank));
                    for (; ii + 16 <= len; ii += 16) {
                      // Load 16 BF16 → 16 FP32
                      __m256i bf16_in = _mm256_loadu_si256((const __m256i*)(grad_row + ii));
                      __m512 fp32_cur = _mm512_castsi512_ps(
                          _mm512_slli_epi32(_mm512_cvtepu16_epi32(bf16_in), 16));
                      // Gather 16 FP32 from grad_a_global with stride=rank
                      __m512i base_idx = _mm512_add_epi32(
                          _mm512_set1_epi32(static_cast<int>((i_start + ii) * rank + r)),
                          stride_idx);
                      __m512 acc_val = _mm512_i32gather_ps(base_idx, grad_a_global, 4);
                      // FMA + convert back
                      fp32_cur = _mm512_fmadd_ps(acc_val, scale_v, fp32_cur);
                      __m256i bf16_out = (__m256i)_mm512_cvtneps_pbh(fp32_cur);
                      _mm256_storeu_si256((__m256i*)(grad_row + ii), bf16_out);
                    }
                    // Scalar remainder
                    for (; ii < len; ii++) {
                      float cur = GGML_BF16_TO_FP32(grad_row[ii]);
                      cur += grad_a_global[static_cast<size_t>(i_start + ii) * rank + r] * scale;
                      grad_row[ii] = GGML_FP32_TO_BF16(cur);
                    }
                  }
                }
              },
              nullptr, "bwd_down_lora_write_AB");
        }
      }
    }
  }

  void backward_activation(const ForwardCache& cache) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    constexpr int kSmallBwdDirectQlen = 0;
    constexpr int kSmallBwdDirectMaxTasks = 16;
    auto direct_or_pool = [&](int count, auto&& fn, const char* task_name, int block_size = 1) {
      if (qlen <= kSmallBwdDirectQlen && count <= kSmallBwdDirectMaxTasks) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr, task_name, block_size);
      }
    };

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
    direct_or_pool(
        activated_expert,
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

          int total = num_tokens * config_.intermediate_size;
          int i = 0;

          // AVX512: process 32 BF16 elements (2×16 FP32) per iteration
          __m512 one = _mm512_set1_ps(1.0f);
          for (; i + 32 <= total; i += 32) {
            __m512 g0, g1, u0, u1, gi0, gi1;
            avx512_32xbf16_to_32xfp32((__m512i*)(gate_output + i), &g0, &g1);
            avx512_32xbf16_to_32xfp32((__m512i*)(up_output + i), &u0, &u1);
            avx512_32xbf16_to_32xfp32((__m512i*)(grad_inter + i), &gi0, &gi1);

            // First 16: sigmoid, silu derivative, gradients
            __m512 exp0 = avx512_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g0));
            __m512 sig0 = _mm512_div_ps(one, _mm512_add_ps(one, exp0));
            __m512 silu0 = _mm512_mul_ps(g0, sig0);
            __m512 dsilu0 = _mm512_mul_ps(sig0, _mm512_fmadd_ps(g0, _mm512_sub_ps(one, sig0), one));
            __m512 gg0 = _mm512_mul_ps(_mm512_mul_ps(gi0, u0), dsilu0);
            __m512 gu0 = _mm512_mul_ps(gi0, silu0);

            // Second 16: same computation
            __m512 exp1 = avx512_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g1));
            __m512 sig1 = _mm512_div_ps(one, _mm512_add_ps(one, exp1));
            __m512 silu1 = _mm512_mul_ps(g1, sig1);
            __m512 dsilu1 = _mm512_mul_ps(sig1, _mm512_fmadd_ps(g1, _mm512_sub_ps(one, sig1), one));
            __m512 gg1 = _mm512_mul_ps(_mm512_mul_ps(gi1, u1), dsilu1);
            __m512 gu1 = _mm512_mul_ps(gi1, silu1);

            avx512_32xfp32_to_32xbf16(&gg0, &gg1, (__m512i*)(grad_gate + i));
            avx512_32xfp32_to_32xbf16(&gu0, &gu1, (__m512i*)(grad_up + i));
          }

          // Scalar tail
          for (; i < total; i++) {
            float g_val = GGML_BF16_TO_FP32(gate_output[i]);
            float u_val = GGML_BF16_TO_FP32(up_output[i]);
            float sigmoid_val = 1.0f / (1.0f + expf(-g_val));
            float silu_val = g_val * sigmoid_val;
            float grad_i_val = GGML_BF16_TO_FP32(grad_inter[i]);
            grad_gate[i] = GGML_FP32_TO_BF16(
                grad_i_val * u_val * sigmoid_val * (1.0f + g_val * (1.0f - sigmoid_val)));
            grad_up[i] = GGML_FP32_TO_BF16(grad_i_val * silu_val);
          }
        },
        "bwd_act_silu");

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
  }

  /**
   * @brief AMX-optimized backward pass for gate and up projections.
   *
   * Uses AMX GEMM for base weight contribution and LoRA grad_input. LoRA weight gradients
   * remain small for-loops.
   */
  void backward_gate_up_amx(const ForwardCache& cache, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                            void* grad_up_lora_a, void* grad_up_lora_b,
                            int full_intermediate_size = 0,
                            float* fp32_grad_gate_lora_a = nullptr,
                            float* fp32_grad_up_lora_a = nullptr) {
    if (full_intermediate_size == 0) full_intermediate_size = config_.intermediate_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;
    constexpr int kSmallBwdDirectQlen = 0;
    constexpr int kSmallBwdDirectMaxTasks = 16;
    auto direct_or_pool = [&](int count, auto&& fn, const char* task_name, int block_size = 1) {
      if (qlen <= kSmallBwdDirectQlen && count <= kSmallBwdDirectMaxTasks) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr, task_name, block_size);
      }
    };

    ggml_bf16_t* grad_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
    ggml_bf16_t* grad_gate_b = (ggml_bf16_t*)grad_gate_lora_b;
    ggml_bf16_t* grad_up_a = (ggml_bf16_t*)grad_up_lora_a;
    ggml_bf16_t* grad_up_b = (ggml_bf16_t*)grad_up_lora_b;

    assert(backward_weights_prepared_);
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      prepare_lora_backward_weights();
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

      // Allocate BufferC for grad_gate_up (gate path)
      grad_gate_up_bc_[expert_idx]->max_m = local_max_m;
      grad_gate_up_bc_[expert_idx]->set_data(grad_gate_up_bc_ptr);
      grad_gate_up_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));
    }

    // Allocate up-path duplicate buffers for merged gate+up GEMM
    {
      char* up_ba_ptr = (char*)gu_up_ba_pool_;
      char* up_bc_ptr = (char*)gu_up_bc_pool_;
      char* up_bf16_ptr = (char*)gu_up_bf16_pool_;
      for (int task_id = 0; task_id < activated_expert; task_id++) {
        int expert_idx = m_expert_id_map_[task_id];
        int m = m_local_num_[expert_idx];
        if (m == 0) continue;
        size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

        down_ba_up_[expert_idx]->max_m = local_max_m;
        down_ba_up_[expert_idx]->set_data(up_ba_ptr);
        up_ba_ptr += align64(T::BufferA::required_size(local_max_m, config_.intermediate_size));

        grad_gate_up_bc_up_[expert_idx]->max_m = local_max_m;
        grad_gate_up_bc_up_[expert_idx]->set_data(up_bc_ptr);
        up_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));

        grad_output_bf16_up_ptr_[expert_idx] = (ggml_bf16_t*)up_bf16_ptr;
        up_bf16_ptr += align64(local_max_m * config_.hidden_size * sizeof(ggml_bf16_t));
      }
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
      const int hidden = config_.hidden_size;
      const int hidden_vec_end = hidden & ~31;
      const __m512 scale_vec = _mm512_set1_ps(scale);
      direct_or_pool(
          qlen,
          [&, scale, hidden, hidden_vec_end, scale_vec](int token_id) {
            ggml_bf16_t* dst = grad_input_bf16 + token_id * hidden;
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
                  float* accum = it->second.data() + pos * hidden;
                  for (int h = 0; h < hidden; h++) {
                    accum[h] += GGML_BF16_TO_FP32(contrib[h]) * scale;
                  }
                }
              }

              int h = 0;
              for (; h < hidden_vec_end; h += 32) {
                __m512 x0, x1, cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)(contrib + h), &x0, &x1);
                avx512_32xbf16_to_32xfp32((__m512i*)(dst + h), &cur0, &cur1);
                x0 = _mm512_fmadd_ps(x0, scale_vec, cur0);
                x1 = _mm512_fmadd_ps(x1, scale_vec, cur1);
                avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)(dst + h));
              }
              for (; h < hidden; h++) {
                float add = GGML_BF16_TO_FP32(contrib[h]) * scale;
                float cur = GGML_BF16_TO_FP32(dst[h]);
                cur += add;
                dst[h] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          task_name);
    };

    // =====================================================
    // Merged gate+up base pass: quantize + GEMM in one dispatch
    // Gate uses: down_ba_, grad_gate_up_bc_, grad_output_bf16_ptr_
    // Up uses:   down_ba_up_, grad_gate_up_bc_up_, grad_output_bf16_up_ptr_
    // No buffer conflicts → gate and up experts can run in parallel.
    // =====================================================
    {
      // Step 1: Quantize both gate and up grads to their respective BufferA
      // Interleaved: task_id/2=expert, task_id%2=do_up (same thread does gate+up for same expert)
      direct_or_pool(
          activated_expert * 2,
          [&](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            auto& ba = do_up ? down_ba_up_[expert_idx] : down_ba_[expert_idx];
            ba->from_mat(m, grad, 0, 1);
          },
          "bwd_gu_base_q");

      // Step 2: GEMM both gate and up in one dispatch
      // Use interleaved layout (same as forward fwd_gate_up_gemm):
      // task_id2 / 2 = task_id (expert × nth), task_id2 % 2 = do_up
      // This ensures gate and up for same expert run on same thread sequentially,
      // avoiding cache thrashing between gate_backward_bb_ and up_backward_bb_.
      int nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * activated_expert * 2, [](int _) { T::config(); },
          [&, nth](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int task_idx = task_id / nth;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = do_up ? down_ba_up_[expert_idx] : down_ba_[expert_idx];
            auto& bb = do_up ? up_backward_bb_[expert_idx] : gate_backward_bb_[expert_idx];
            auto& bc = do_up ? grad_gate_up_bc_up_[expert_idx] : grad_gate_up_bc_[expert_idx];
            ggml_bf16_t* out = do_up ? grad_output_bf16_up_ptr_[expert_idx] : grad_output_bf16_ptr_[expert_idx];

            amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
            bc->to_mat(m, out, ith, nth);
          },
          nullptr, "bwd_gu_base_gemm", 1);

      // scatter_base removed: lora_compute_gradin now accumulates scaled lora gradin
      // on top of base gradin in-place. A single scatter pass (bwd_gu_scatter_combined)
      // at the end handles both base + lora contributions with scale=1.0.
    }

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
      return;
    }

    const bool use_fp32_lora_a = (fp32_grad_gate_lora_a != nullptr);

    // =====================================================
    // Step 6a: Repack input → gate_up_ba_ (AMX BufferA format)
    // gate_up_ba_ is free during backward (only used in forward).
    // =====================================================
    {
      auto align64_fn = [](size_t v) { return (v + 63) & (~(size_t)63); };
      void* ba_ptr = Base::gate_up_ba_pool_;
      for (int task_id = 0; task_id < activated_expert; task_id++) {
        int expert_idx = m_expert_id_map_[task_id];
        int m = m_local_num_[expert_idx];
        if (m == 0) continue;
        size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;
        gate_up_ba_[expert_idx]->max_m = local_max_m;
        gate_up_ba_[expert_idx]->set_data(ba_ptr);
        ba_ptr = (void*)((uintptr_t)ba_ptr +
            align64_fn(T::BufferA::required_size(local_max_m, config_.hidden_size)));
      }
      direct_or_pool(activated_expert,
          [this](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            if (m_local_num_[expert_idx] == 0) return;
            gate_up_ba_[expert_idx]->from_mat(
                m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
          }, "bwd_gu_repack_ba", 1);
    }

    // =====================================================
    // Step 6b: AMX GEMM input @ lora_a^T → BF16 intermediate
    // Replaces dpbf16 lora_bf16_matmul_t4r4 in the fused kernel.
    // =====================================================
    {
      int amx_nth = T::recommended_nth(padded_lora_rank_);
      pool->do_work_stealing_job(
          amx_nth * activated_expert * 2,
          [](int) { T::config(); },
          [this, amx_nth](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = m_expert_id_map_[task_id / amx_nth];
            int ith = task_id % amx_nth;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;
            amx::mat_mul(m, padded_lora_rank_, config_.hidden_size,
                         gate_up_ba_[expert_idx],
                         do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx],
                         do_up ? lora_up_intermediate_bc_[expert_idx]
                               : lora_gate_intermediate_bc_[expert_idx],
                         ith, amx_nth);
            (do_up ? lora_up_intermediate_bc_[expert_idx]
                   : lora_gate_intermediate_bc_[expert_idx])
                ->to_mat(m, do_up ? lora_up_intermediate_ptr_[expert_idx]
                                   : lora_gate_intermediate_ptr_[expert_idx],
                         ith, amx_nth);
          }, nullptr, "bwd_gu_lora_amx", 1);
    }

    // =====================================================
    // Fused LoRA gradb:
    //   u_gate/u_up already in BF16 intermediate (from Step 6b AMX GEMM)
    //   grad_gate/grad_up + u_gate/u_up -> grad_B
    // =====================================================
    struct GuLoraFusedBuf {
        int expert_idx = -1;
        int num_tokens = 0;
        size_t token_offset = 0;
        size_t lora_b_offset = 0;        // copy-type: expert_idx * full_I * rank
        size_t lora_a_sparse_offset = 0; // reduce-type: task_id * rank * hidden (sparse FP32)
        size_t lora_a_dense_offset = 0;  // legacy: expert_idx * rank * hidden (dense BF16)
        const ggml_bf16_t* input = nullptr;
        const ggml_bf16_t* gate_lora_a = nullptr;
        const ggml_bf16_t* up_lora_a = nullptr;
        ggml_bf16_t* gate_inter = nullptr;
        ggml_bf16_t* up_inter = nullptr;
        ggml_bf16_t* gate_grad = nullptr;
        ggml_bf16_t* up_grad = nullptr;
      };
      struct GuLoraFusedTask {
        int expert_task = -1;
        int t_start = 0;
        int t_end = 0;
      };

      const int hidden = config_.hidden_size;
      const int inter_size = config_.intermediate_size;
      const int rank = lora_rank_;
      const int lora_b_expert_stride = full_intermediate_size * rank;  // copy-type: full_I * rank
      const size_t gradb_elems = static_cast<size_t>(inter_size) * rank;
      std::vector<GuLoraFusedBuf> fused_bufs(activated_expert);
      std::vector<GuLoraFusedTask> fused_tasks;
      // Use pre-allocated pool (no alloc or zero-init on hot path).
      // First write uses memcpy (=), subsequent writes use += (same as backward_down pattern).
      float* gate_gradb_all = gu_gate_gradb_accum_;
      float* up_gradb_all = gu_up_gradb_accum_;
      std::vector<std::mutex> gradb_mutexes(activated_expert);
      std::vector<uint8_t> gu_gradb_initialized(activated_expert, 0);

      for (int task_id = 0; task_id < activated_expert; task_id++) {
        GuLoraFusedBuf& buf = fused_bufs[task_id];
        int expert_idx = m_expert_id_map_[task_id];
        int num_tokens = m_local_num_[expert_idx];

        buf.expert_idx = expert_idx;
        buf.num_tokens = num_tokens;
        if (num_tokens == 0) continue;

        buf.token_offset = expert_offsets[task_id];
        buf.lora_b_offset = static_cast<size_t>(expert_idx) * lora_b_expert_stride;  // copy-type: full_I * rank
        buf.lora_a_sparse_offset = static_cast<size_t>(task_id) * rank * hidden;       // sparse FP32
        buf.lora_a_dense_offset = static_cast<size_t>(expert_idx) * rank * hidden;     // legacy dense BF16
        buf.input = m_local_input_ptr_[expert_idx];
        buf.gate_lora_a = gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden;
        buf.up_lora_a = up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden;
        buf.gate_inter = lora_gate_intermediate_ptr_[expert_idx];
        buf.up_inter = lora_up_intermediate_ptr_[expert_idx];
        buf.gate_grad = grad_gate_output_ + buf.token_offset * inter_size;
        buf.up_grad = grad_up_output_ + buf.token_offset * inter_size;

        constexpr int kGuLoraTokenTile = 1024;
        for (int t = 0; t < num_tokens; t += kGuLoraTokenTile) {
          fused_tasks.push_back({task_id, t, std::min(t + kGuLoraTokenTile, num_tokens)});
        }
      }

      // Step-1 note: FP32 sparse lora_a buffers are NOT pre-zeroed.
      // First write uses '=' (overwrite), subsequent writes use '+=' (accumulate).

      if (!fused_tasks.empty()) {
        direct_or_pool(
            static_cast<int>(fused_tasks.size()),
            [&, hidden, inter_size, rank, gradb_elems](int task_id) {
              const GuLoraFusedTask& task = fused_tasks[task_id];
              GuLoraFusedBuf& buf = fused_bufs[task.expert_task];
              if (buf.num_tokens == 0) return;

              int local_tokens = task.t_end - task.t_start;
              // u values already computed by AMX GEMM (Step 6b) → read from buf.gate_inter / buf.up_inter
              size_t scratch_elems = gradb_elems * 2;
              float* scratch = get_lora_fp32_buffer(scratch_elems);
              float* gate_gradb_local = scratch;
              float* up_gradb_local = gate_gradb_local + gradb_elems;
              memset(gate_gradb_local, 0, gradb_elems * 2 * sizeof(float));

              for (int t = 0; t < local_tokens; t++) {
                const ggml_bf16_t* gate_inter_row = buf.gate_inter + static_cast<size_t>(task.t_start + t) * padded_lora_rank_;
                const ggml_bf16_t* up_inter_row = buf.up_inter + static_cast<size_t>(task.t_start + t) * padded_lora_rank_;
                const ggml_bf16_t* gate_grad_row = buf.gate_grad + static_cast<size_t>(task.t_start + t) * inter_size;
                const ggml_bf16_t* up_grad_row = buf.up_grad + static_cast<size_t>(task.t_start + t) * inter_size;

                if (rank == 8) {
                  // Load u values from BF16 intermediate → FP32
                  __m128i g_raw = _mm_loadu_si128((__m128i*)gate_inter_row);
                  __m256 gate_u_vec = _mm256_castsi256_ps(
                      _mm256_slli_epi32(_mm256_cvtepu16_epi32(g_raw), 16));
                  __m128i u_raw = _mm_loadu_si128((__m128i*)up_inter_row);
                  __m256 up_u_vec = _mm256_castsi256_ps(
                      _mm256_slli_epi32(_mm256_cvtepu16_epi32(u_raw), 16));

                  for (int i = 0; i < inter_size; i++) {
                    float gg = GGML_BF16_TO_FP32(gate_grad_row[i]);
                    float* g_out = gate_gradb_local + static_cast<size_t>(i) * 8;
                    __m256 g_acc = _mm256_loadu_ps(g_out);
                    g_acc = _mm256_fmadd_ps(_mm256_set1_ps(gg), gate_u_vec, g_acc);
                    _mm256_storeu_ps(g_out, g_acc);

                    float ug = GGML_BF16_TO_FP32(up_grad_row[i]);
                    float* u_out = up_gradb_local + static_cast<size_t>(i) * 8;
                    __m256 u_acc = _mm256_loadu_ps(u_out);
                    u_acc = _mm256_fmadd_ps(_mm256_set1_ps(ug), up_u_vec, u_acc);
                    _mm256_storeu_ps(u_out, u_acc);
                  }
                } else {
                  for (int i = 0; i < inter_size; i++) {
                    float gg = GGML_BF16_TO_FP32(gate_grad_row[i]);
                    if (gg != 0.0f) {
                      float* out = gate_gradb_local + static_cast<size_t>(i) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += gg * GGML_BF16_TO_FP32(gate_inter_row[r]);
                      }
                    }
                    float ug = GGML_BF16_TO_FP32(up_grad_row[i]);
                    if (ug != 0.0f) {
                      float* out = up_gradb_local + static_cast<size_t>(i) * rank;
                      for (int r = 0; r < rank; r++) {
                        out[r] += ug * GGML_BF16_TO_FP32(up_inter_row[r]);
                      }
                    }
                  }
                }
              }

              std::lock_guard<std::mutex> lock(gradb_mutexes[task.expert_task]);
              float* gate_gradb_global =
                  gate_gradb_all + static_cast<size_t>(task.expert_task) * gradb_elems;
              float* up_gradb_global = up_gradb_all + static_cast<size_t>(task.expert_task) * gradb_elems;
              if (!gu_gradb_initialized[task.expert_task]) {
                // First write: memcpy (no zero-init needed)
                std::memcpy(gate_gradb_global, gate_gradb_local, gradb_elems * sizeof(float));
                std::memcpy(up_gradb_global, up_gradb_local, gradb_elems * sizeof(float));
                gu_gradb_initialized[task.expert_task] = 1;
              } else if (rank == 8) {
                for (size_t off = 0; off < gradb_elems; off += rank) {
                  __m256 gate_acc = _mm256_loadu_ps(gate_gradb_global + off);
                  gate_acc = _mm256_add_ps(gate_acc, _mm256_loadu_ps(gate_gradb_local + off));
                  _mm256_storeu_ps(gate_gradb_global + off, gate_acc);

                  __m256 up_acc = _mm256_loadu_ps(up_gradb_global + off);
                  up_acc = _mm256_add_ps(up_acc, _mm256_loadu_ps(up_gradb_local + off));
                  _mm256_storeu_ps(up_gradb_global + off, up_acc);
                }
              } else {
                for (size_t off = 0; off < gradb_elems; off++) {
                  gate_gradb_global[off] += gate_gradb_local[off];
                  up_gradb_global[off] += up_gradb_local[off];
                }
              }
            },
            "bwd_gu_lora_u_gradb_fused");

        constexpr int kGuGradBBlock = 256;
        int gradb_blocks = (inter_size + kGuGradBBlock - 1) / kGuGradBBlock;
        const float scale = lora_scaling_;
        pool->do_work_stealing_job(
            activated_expert * 2 * gradb_blocks, nullptr,
            [&, grad_gate_b, grad_up_b, activated_expert, gradb_blocks, inter_size, rank, gradb_elems,
             scale](int task_id) {
              int half_tasks = activated_expert * gradb_blocks;
              bool do_up = task_id >= half_tasks;
              int local_task_id = do_up ? (task_id - half_tasks) : task_id;
              int expert_task = local_task_id / gradb_blocks;
              int block_idx = local_task_id % gradb_blocks;
              GuLoraFusedBuf& buf = fused_bufs[expert_task];
              if (buf.num_tokens == 0) return;

              int i_start = block_idx * kGuGradBBlock;
              int i_end = std::min(inter_size, i_start + kGuGradBBlock);
              float* gradb_global =
                  (do_up ? up_gradb_all : gate_gradb_all) + static_cast<size_t>(expert_task) * gradb_elems;
              ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;

              if (rank == 8) {
                bf16_fma_writeback_rank8(
                    grad_lora_b + buf.lora_b_offset + static_cast<size_t>(i_start) * rank,
                    gradb_global + static_cast<size_t>(i_start) * rank,
                    i_end - i_start, rank, scale);
              } else {
                for (int i = i_start; i < i_end; i++) {
                  ggml_bf16_t* out = grad_lora_b + buf.lora_b_offset + static_cast<size_t>(i) * rank;
                  float* acc_row = gradb_global + static_cast<size_t>(i) * rank;
                  for (int r = 0; r < rank; r++) {
                    float cur = GGML_BF16_TO_FP32(out[r]);
                    cur += acc_row[r] * scale;
                    out[r] = GGML_FP32_TO_BF16(cur);
                  }
                }
              }
            },
            nullptr, "bwd_gu_lora_gradb_fused_write");
      }

    // =====================================================
    // Remaining LoRA steps:
    //   grad @ B^T -> G_B
    //   G_B @ A -> grad_input contribution
    //   scatter + grad_A
    // Gate writes to grad_output_bf16_ptr_, up writes to grad_output_bf16_up_ptr_
    // → no buffer conflict → gb_gradin runs in parallel, scatter merges both.
    // =====================================================
    auto lora_compute_gradin = [&](bool do_up) {
      const char* gb_gradin_name = do_up ? "bwd_gu_lora_gb_gradin_fused_up" : "bwd_gu_lora_gb_gradin_fused_gate";

      struct GuLoraGradInTask {
        int expert_task = -1;
        int t_start = 0;
        int t_end = 0;
      };
      std::vector<GuLoraGradInTask> gradin_tasks;
      gradin_tasks.reserve(activated_expert * 16);
      for (int expert_task = 0; expert_task < activated_expert; expert_task++) {
        int expert_idx = m_expert_id_map_[expert_task];
        int m = m_local_num_[expert_idx];
        constexpr int kGuGradInTile = 512;
        for (int t = 0; t < m; t += kGuGradInTile) {
          gradin_tasks.push_back({expert_task, t, std::min(t + kGuGradInTile, m)});
        }
      }

      if (!gradin_tasks.empty()) {
        direct_or_pool(
            static_cast<int>(gradin_tasks.size()),
            [&, do_up](int task_id) {
              const GuLoraGradInTask& task = gradin_tasks[task_id];
              int expert_task = task.expert_task;
              int expert_idx = m_expert_id_map_[expert_task];
              int local_tokens = task.t_end - task.t_start;
              if (local_tokens <= 0) return;

              const int hidden = config_.hidden_size;
              const int inter_size = config_.intermediate_size;
              const size_t offset = expert_offsets[expert_task] + task.t_start;
              ggml_bf16_t* grad =
                  do_up ? (grad_up_output_ + offset * inter_size) : (grad_gate_output_ + offset * inter_size);
              ggml_bf16_t* inter_ptr_base =
                  do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
              ggml_bf16_t* inter_ptr = inter_ptr_base + static_cast<size_t>(task.t_start) * padded_lora_rank_;
              // Gate → grad_output_bf16_ptr_, Up → grad_output_bf16_up_ptr_ (separate buffers)
              ggml_bf16_t* grad_out = do_up
                  ? (grad_output_bf16_up_ptr_[expert_idx] + static_cast<size_t>(task.t_start) * hidden)
                  : (grad_output_bf16_ptr_[expert_idx] + static_cast<size_t>(task.t_start) * hidden);
              const ggml_bf16_t* lora_b_t =
                  (do_up ? up_lora_b_transposed_ : gate_lora_b_transposed_) +
                  static_cast<size_t>(expert_idx) * lora_rank_ * inter_size;
              const ggml_bf16_t* lora_a =
                  (do_up ? up_lora_a_ : gate_lora_a_) + static_cast<size_t>(expert_idx) * lora_rank_ * hidden;

              float* gb = get_lora_fp32_buffer(static_cast<size_t>(local_tokens) * lora_rank_);
              avx::lora_backward_matmul_transposed(grad, lora_b_t, gb, local_tokens, inter_size, lora_rank_);

              memset(inter_ptr, 0, static_cast<size_t>(local_tokens) * padded_lora_rank_ * sizeof(ggml_bf16_t));
              for (int t = 0; t < local_tokens; t++) {
                ggml_bf16_t* inter_row = inter_ptr + static_cast<size_t>(t) * padded_lora_rank_;
                const float* gb_row = gb + static_cast<size_t>(t) * lora_rank_;
                for (int r = 0; r < lora_rank_; r++) {
                  inter_row[r] = GGML_FP32_TO_BF16(gb_row[r]);
                }
              }

              // Accumulate scaled lora gradin on top of existing base gradin in-place.
              // (base gradin was written here by bwd_gu_base_gemm → to_mat; we skip scatter_base
              //  and do a single merged scatter at the end with scale=1.0)
              avx::lora_fp32_bf16_fused_add_transposed(gb, lora_a, grad_out, local_tokens, lora_rank_, hidden, lora_scaling_);
            },
            gb_gradin_name);
      }
    };

    auto lora_gradA = [&](bool do_up) {
      // Step 6: grad_A = G_B^T @ X
      ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
      float* fp32_grad_lora_a = do_up ? fp32_grad_up_lora_a : fp32_grad_gate_lora_a;
      constexpr int kGuGradATile = 512;
      int grad_a_blocks = (config_.hidden_size + kGuGradATile - 1) / kGuGradATile;
      pool->do_work_stealing_job(
          activated_expert * grad_a_blocks, nullptr,
          [this, do_up, grad_lora_a, fp32_grad_lora_a, use_fp32_lora_a, grad_a_blocks, &fused_bufs](int task_id) {
            int expert_task = task_id / grad_a_blocks;
            int block_idx = task_id % grad_a_blocks;
            int expert_idx = m_expert_id_map_[expert_task];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t* g_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            const GuLoraFusedBuf& buf = fused_bufs[expert_task];
            ggml_bf16_t* expert_input = m_local_input_ptr_[expert_idx];

            const int hidden = config_.hidden_size;
            constexpr int kVecWidth = 32;
            int h_start = block_idx * kGuGradATile;
            int h_end = std::min(hidden, h_start + kGuGradATile);
            int tile_len = h_end - h_start;
            if (tile_len <= 0) return;
            int tile_vec_end = tile_len & ~(kVecWidth - 1);
            __m512 scale_vec = _mm512_set1_ps(lora_scaling_);
            const int lora_r = lora_rank_;

            std::vector<float> accum(lora_r * tile_len, 0.0f);

            for (int t = 0; t < num_tokens; t++) {
              const ggml_bf16_t* g_row = g_ptr + t * padded_lora_rank_;
              const ggml_bf16_t* input_row = expert_input + t * hidden + h_start;
              // Pre-load g values for all ranks (tiny: 8 floats)
              float g_vals[8];
              for (int r = 0; r < lora_r && r < 8; r++)
                g_vals[r] = GGML_BF16_TO_FP32(g_row[r]);
              // Swap r/h loops: convert input BF16→FP32 once, FMA into all 8 rank accumulators
              int h = 0;
              for (; h < tile_vec_end; h += kVecWidth) {
                __m512 x0, x1;
                avx512_32xbf16_to_32xfp32((__m512i*)(input_row + h), &x0, &x1);
                for (int r = 0; r < lora_r; r++) {
                  if (g_vals[r] == 0.0f) continue;
                  __m512 gb_vec = _mm512_set1_ps(g_vals[r]);
                  float* acc_row = accum.data() + r * tile_len;
                  __m512 acc0 = _mm512_loadu_ps(acc_row + h);
                  __m512 acc1 = _mm512_loadu_ps(acc_row + h + 16);
                  acc0 = _mm512_fmadd_ps(x0, gb_vec, acc0);
                  acc1 = _mm512_fmadd_ps(x1, gb_vec, acc1);
                  _mm512_storeu_ps(acc_row + h, acc0);
                  _mm512_storeu_ps(acc_row + h + 16, acc1);
                }
              }
              for (; h < tile_len; h++) {
                float inp = GGML_BF16_TO_FP32(input_row[h]);
                for (int r = 0; r < lora_r; r++) {
                  if (g_vals[r] == 0.0f) continue;
                  accum[r * tile_len + h] += inp * g_vals[r];
                }
              }
            }

            // Write back
            if (use_fp32_lora_a) {
              for (int r = 0; r < lora_r; r++) {
                float* fp32_row = fp32_grad_lora_a + buf.lora_a_sparse_offset + r * hidden + h_start;
                float* acc_row = accum.data() + r * tile_len;
                int h = 0;
                for (; h + 16 <= tile_len; h += 16) {
                  __m512 val = _mm512_loadu_ps(acc_row + h);
                  __m512 result = _mm512_mul_ps(val, scale_vec);
                  _mm512_storeu_ps(fp32_row + h, result);
                }
                for (; h < tile_len; h++) {
                  fp32_row[h] = acc_row[h] * lora_scaling_;
                }
              }
            } else {
              for (int r = 0; r < lora_r; r++) {
                ggml_bf16_t* grad_row = grad_lora_a + buf.lora_a_dense_offset + r * hidden + h_start;
                float* acc_row = accum.data() + r * tile_len;
                int h = 0;
                for (; h + kVecWidth <= tile_len; h += kVecWidth) {
                  __m512 sum0 = _mm512_loadu_ps(acc_row + h);
                  __m512 sum1 = _mm512_loadu_ps(acc_row + h + 16);
                  __m512 cur0, cur1;
                  avx512_32xbf16_to_32xfp32((__m512i*)(grad_row + h), &cur0, &cur1);
                  cur0 = _mm512_fmadd_ps(sum0, scale_vec, cur0);
                  cur1 = _mm512_fmadd_ps(sum1, scale_vec, cur1);
                  avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)(grad_row + h));
                }
                for (; h < tile_len; h++) {
                  float cur = GGML_BF16_TO_FP32(grad_row[h]);
                  cur += acc_row[h] * lora_scaling_;
                  grad_row[h] = GGML_FP32_TO_BF16(cur);
                }
              }
            }
          },
          nullptr, do_up ? "bwd_gu_lora_gradA_up" : "bwd_gu_lora_gradA_gate");
    };

    // Phase 1: Compute gate+up LoRA grad_input into separate buffers (no conflict → parallel)
    lora_compute_gradin(false);  // gate → grad_output_bf16_ptr_
    lora_compute_gradin(true);   // up → grad_output_bf16_up_ptr_

    // DUMP: LoRA intermediate + combined gradin before scatter
    // NOTE: grad_output_bf16_ptr_ now contains base_gradin + scaled_lora_gradin (not lora-only).
    // The dump of "backward_gate_lora" / "backward_up_lora" will reflect this combined value.
    if (is_dump_enabled()) {
      for (int i = 0; i < activated_expert; i++) {
        int expert_idx = m_expert_id_map_[i];
        int m = m_local_num_[expert_idx];
        if (m > 0) {
          dump_bf16_matrix(lora_gate_intermediate_ptr_[expert_idx], m, padded_lora_rank_,
                           "backward_gate_lora_inter", tp_part_idx, expert_idx);
          // Buffer contains base + scaled lora; dump as-is (scale=1.0)
          dump_bf16_matrix_scaled(grad_output_bf16_ptr_[expert_idx], m, config_.hidden_size, 1.0f,
                                  "backward_gate_combined_gradin", tp_part_idx, expert_idx);
          dump_bf16_matrix(lora_up_intermediate_ptr_[expert_idx], m, padded_lora_rank_,
                           "backward_up_lora_inter", tp_part_idx, expert_idx);
          dump_bf16_matrix_scaled(grad_output_bf16_up_ptr_[expert_idx], m, config_.hidden_size, 1.0f,
                                  "backward_up_combined_gradin", tp_part_idx, expert_idx);
        }
      }
    }

    // Combined scatter: buffers now contain base_gradin + scaled_lora_gradin.
    // Scatter with scale=1.0 to grad_input.
    {
      ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
      const int hidden = config_.hidden_size;
      const int hidden_vec_end = hidden & ~31;
      direct_or_pool(
          qlen,
          [&, hidden, hidden_vec_end](int token_id) {
            ggml_bf16_t* dst = grad_input_bf16 + token_id * hidden;
            for (int j = 0; j < k; j++) {
              int expert_idx = cache.expert_ids_cache[token_id * k + j];
              if (expert_idx < config_.num_gpu_experts || expert_idx >= config_.expert_num) continue;
              if (m_local_num_[expert_idx] == 0) continue;
              int pos = cache.m_local_pos_cache[token_id][j];

              ggml_bf16_t* gate_contrib = grad_output_bf16_ptr_[expert_idx] + pos * hidden;
              ggml_bf16_t* up_contrib = grad_output_bf16_up_ptr_[expert_idx] + pos * hidden;

              // Accumulate for dump (buffer already contains base + scaled lora)
              if (dump_enabled) {
                auto it = expert_grad_accum.find(expert_idx);
                if (it != expert_grad_accum.end()) {
                  float* accum = it->second.data() + pos * hidden;
                  for (int h = 0; h < hidden; h++) {
                    accum[h] += GGML_BF16_TO_FP32(gate_contrib[h]) + GGML_BF16_TO_FP32(up_contrib[h]);
                  }
                }
              }

              int h = 0;
              for (; h < hidden_vec_end; h += 32) {
                __m512 g0, g1, u0, u1, cur0, cur1;
                avx512_32xbf16_to_32xfp32((__m512i*)(gate_contrib + h), &g0, &g1);
                avx512_32xbf16_to_32xfp32((__m512i*)(up_contrib + h), &u0, &u1);
                avx512_32xbf16_to_32xfp32((__m512i*)(dst + h), &cur0, &cur1);
                cur0 = _mm512_add_ps(cur0, _mm512_add_ps(g0, u0));
                cur1 = _mm512_add_ps(cur1, _mm512_add_ps(g1, u1));
                avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)(dst + h));
              }
              for (; h < hidden; h++) {
                float cur = GGML_BF16_TO_FP32(dst[h]);
                cur += GGML_BF16_TO_FP32(gate_contrib[h]) + GGML_BF16_TO_FP32(up_contrib[h]);
                dst[h] = GGML_FP32_TO_BF16(cur);
              }
            }
          },
          "bwd_gu_scatter_combined");
    }

    // Phase 3: grad_A (gate and up independently)
    lora_gradA(false);  // gate
    lora_gradA(true);   // up

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
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_MOE_H
