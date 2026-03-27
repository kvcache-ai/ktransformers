/**
 * @Description  : AVX2 BF16 GEMM kernel with trivial Buffer abstractions
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Unlike AMX kernels that use packed tile layouts (BufferB with 16x16 transpose),
 * the AVX2 kernel uses row-major storage for all buffers.
 * BufferA/B/C are thin wrappers over raw memory with trivial from_mat/to_mat.
 *
 * GEMM: C[m,n] = sum_k A[m,k] * B[n,k]
 *   A: [M, K] row-major BF16 (input activations)
 *   B: [N, K] row-major BF16 (weights, each row is one output neuron)
 *   C: [M, N] row-major FP32 (output)
 **/
#ifndef CPUINFER_OPERATOR_AVX2_BF16_GEMM_H
#define CPUINFER_OPERATOR_AVX2_BF16_GEMM_H

#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>

#include "avx2_bf16_utils.hpp"

namespace avx2 {

// Split range [0, total) among nth threads, return [start, end) for thread ith
static inline std::pair<int, int> split_range(int total, int ith, int nth) {
  int per = total / nth;
  int rem = total % nth;
  int start = ith * per + std::min(ith, rem);
  int end = start + per + (ith < rem ? 1 : 0);
  return {start, end};
}

struct GemmKernelAVX2BF16 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;      // No M-direction padding needed (vs AMX 16)
  static constexpr int N_STEP = 8;      // 8-wide FP32 AVX2 (vs AMX 32)
  static constexpr int K_STEP = 8;      // Process 8 K elements at a time
  static constexpr int N_BLOCK = 64;    // N blocking for cache
  static constexpr int K_BLOCK = 256;   // K blocking for cache
  static constexpr double ELEMENT_SIZE = 2.0;  // BF16 = 2 bytes

  // No AMX tile configuration needed
  static void config() {}

  // Thread count for N-dimension parallelism
  // Must return >= 1 to avoid division by zero in moe_base task dispatch
  static int recommended_nth(int n) {
    return std::max(1, n / N_STEP);
  }

  // Split N range for multi-threaded GEMM
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return split_range(n, ith, nth);
  }

  // ========================================================================
  // BufferA: Input activations [M, K] row-major BF16
  // from_mat() = memcpy (no packing needed for AVX2)
  // ========================================================================
  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : max_m(m), k(k_), data((ggml_bf16_t*)ptr) {}

    static size_t required_size(size_t m, size_t k) {
      return m * k * sizeof(ggml_bf16_t);
    }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    // Copy input rows into buffer (trivial memcpy)
    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        // Multi-threaded: split by rows
        auto [m_start, m_end] = split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  // ========================================================================
  // BufferB: Weight matrix [N, K] row-major BF16
  // from_mat() = memcpy (no transpose/packing needed)
  // ========================================================================
  struct BufferB {
    ggml_bf16_t* b = nullptr;
    size_t n = 0;
    size_t k = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, void* ptr) : n(n_), k(k_), b((ggml_bf16_t*)ptr) {}

    static size_t required_size(size_t n, size_t k) {
      return n * k * sizeof(ggml_bf16_t);
    }

    // Copy weight data (multi-threaded by N dimension)
    void from_mat(const ggml_bf16_t* src, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      std::memcpy(b + n_start * k, src + n_start * k,
                  (size_t)(n_end - n_start) * k * sizeof(ggml_bf16_t));
    }
  };

  // ========================================================================
  // BufferC: Output matrix [M, N] row-major FP32
  // to_mat() converts FP32 -> BF16 and writes out
  // ========================================================================
  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : max_m(m), n(n_), data((float*)ptr) {}

    static size_t required_size(size_t m, size_t n) {
      return m * n * sizeof(float);
    }

    void set_data(void* ptr) { data = (float*)ptr; }

    // Convert FP32 output to BF16 and write to destination
    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range_n((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          __m256 v = _mm256_loadu_ps(src_row + j);
          store_fp32_to_bf16(dst_row + j, v);
        }
        // Scalar tail
        for (; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

// ============================================================================
// AVX2 BF16 GEMM functions
// C[m,n] = sum_k A[m,k] * B[n,k]
// ============================================================================

// General GEMM (works for both vec_mul m=1 and mat_mul m>1)
static inline void gemm_bf16(
    int m, int n, int k,
    GemmKernelAVX2BF16::BufferA& a,
    GemmKernelAVX2BF16::BufferB& b,
    GemmKernelAVX2BF16::BufferC& c,
    int ith, int nth) {

  auto [n_start, n_end] = split_range(n, ith, nth);

  for (int ni = n_start; ni < n_end; ni++) {
    const ggml_bf16_t* b_row = b.b + (size_t)ni * k;

    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;

      // AVX2 BF16 dot product (matches ggml_vec_dot_bf16 AVX2 path)
      __m256 c1 = _mm256_setzero_ps();
      __m256 c2 = _mm256_setzero_ps();
      __m256 c3 = _mm256_setzero_ps();
      __m256 c4 = _mm256_setzero_ps();

      int ki = 0;
      for (; ki + 32 <= k; ki += 32) {
        c1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki),      load_bf16_to_fp32(b_row + ki),      c1);
        c2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 8),  load_bf16_to_fp32(b_row + ki + 8),  c2);
        c3 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 16), load_bf16_to_fp32(b_row + ki + 16), c3);
        c4 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + ki + 24), load_bf16_to_fp32(b_row + ki + 24), c4);
      }

      float sum = hsum_avx2(_mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4)));

      // Scalar tail
      for (; ki < k; ki++) {
        sum += GGML_BF16_TO_FP32(a_row[ki]) * GGML_BF16_TO_FP32(b_row[ki]);
      }

      c.data[mi * n + ni] = sum;
    }
  }
}

// vec_mul: dispatch to gemm_bf16
static inline void vec_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelAVX2BF16::BufferA>& a,
    std::shared_ptr<GemmKernelAVX2BF16::BufferB>& b,
    std::shared_ptr<GemmKernelAVX2BF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

// mat_mul: dispatch to gemm_bf16
static inline void mat_mul(
    int m, int n, int k,
    std::shared_ptr<GemmKernelAVX2BF16::BufferA>& a,
    std::shared_ptr<GemmKernelAVX2BF16::BufferB>& b,
    std::shared_ptr<GemmKernelAVX2BF16::BufferC>& c,
    int ith, int nth) {
  gemm_bf16(m, n, k, *a, *b, *c, ith, nth);
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_BF16_GEMM_H
