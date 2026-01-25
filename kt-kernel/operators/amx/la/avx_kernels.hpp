#ifndef AVX_KERNELS_HPP
#define AVX_KERNELS_HPP

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "../../../cpu_backend/worker_pool.h"
#include "llama.cpp/ggml-impl.h"
#include "utils.hpp"

namespace avx {

// Enable/disable kernel tracing (can be controlled at compile time)
#ifndef AVX_KERNEL_TRACE_ENABLED
#define AVX_KERNEL_TRACE_ENABLED 0
#endif

// ============================================================================
// AVX512 BF16 LoRA Kernels
//
// Optimized kernels for LoRA computations using AVX512 with native BF16 support.
// These kernels use token-blocking and rank-blocking to maximize arithmetic
// intensity and reduce memory bandwidth pressure.
//
// Key optimizations:
// 1. Native _mm512_dpbf16_ps for BF16 dot-accumulate (no BF16->FP32 conversion)
// 2. Token-blocking: process multiple tokens per weight load
// 3. Rank-blocking: process multiple ranks in parallel
// ============================================================================

/**
 * @brief BF16 input × BF16 weight → FP32 output matmul
 *
 * Computes: output[t, r] = sum_k(input[t, k] * weight[r, k])
 *
 * Optimized with T_BLOCK=4, R_BLOCK=4 for high arithmetic intensity.
 * Uses native _mm512_dpbf16_ps instruction.
 *
 * @param input   Input tensor [num_tokens, k_dim] in BF16
 * @param weight  Weight tensor [rank, k_dim] in BF16
 * @param output  Output tensor [num_tokens, rank] in FP32
 * @param num_tokens Number of tokens to process
 * @param k_dim   Inner dimension (hidden size)
 * @param rank    LoRA rank (output dimension)
 */
inline void lora_bf16_matmul_t4r4(const ggml_bf16_t* __restrict input, const ggml_bf16_t* __restrict weight,
                                  float* __restrict output, int num_tokens, int k_dim, int rank) {
  // #if AVX_KERNEL_TRACE_ENABLED
  //   uint64_t trace_start = sft_timer::get_trace_timestamp();
  // #endif

  constexpr int T_BLOCK = 4;
  constexpr int R_BLOCK = 4;

  int t = 0;
  // Process 4 tokens at a time
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const ggml_bf16_t* inp0 = input + (t + 0) * k_dim;
    const ggml_bf16_t* inp1 = input + (t + 1) * k_dim;
    const ggml_bf16_t* inp2 = input + (t + 2) * k_dim;
    const ggml_bf16_t* inp3 = input + (t + 3) * k_dim;
    float* out0 = output + (t + 0) * rank;
    float* out1 = output + (t + 1) * rank;
    float* out2 = output + (t + 2) * rank;
    float* out3 = output + (t + 3) * rank;

    int r = 0;
    // Process 4 ranks at a time
    for (; r + R_BLOCK <= rank; r += R_BLOCK) {
      // 16 accumulators: 4 tokens × 4 ranks
      __m512 acc_t0_r0 = _mm512_setzero_ps(), acc_t0_r1 = _mm512_setzero_ps();
      __m512 acc_t0_r2 = _mm512_setzero_ps(), acc_t0_r3 = _mm512_setzero_ps();
      __m512 acc_t1_r0 = _mm512_setzero_ps(), acc_t1_r1 = _mm512_setzero_ps();
      __m512 acc_t1_r2 = _mm512_setzero_ps(), acc_t1_r3 = _mm512_setzero_ps();
      __m512 acc_t2_r0 = _mm512_setzero_ps(), acc_t2_r1 = _mm512_setzero_ps();
      __m512 acc_t2_r2 = _mm512_setzero_ps(), acc_t2_r3 = _mm512_setzero_ps();
      __m512 acc_t3_r0 = _mm512_setzero_ps(), acc_t3_r1 = _mm512_setzero_ps();
      __m512 acc_t3_r2 = _mm512_setzero_ps(), acc_t3_r3 = _mm512_setzero_ps();

      const ggml_bf16_t* w0 = weight + (r + 0) * k_dim;
      const ggml_bf16_t* w1 = weight + (r + 1) * k_dim;
      const ggml_bf16_t* w2 = weight + (r + 2) * k_dim;
      const ggml_bf16_t* w3 = weight + (r + 3) * k_dim;

      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        // Load weights once (4 cache lines), reuse for 4 tokens
        __m512bh wv0 = (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + k));
        __m512bh wv1 = (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + k));
        __m512bh wv2 = (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + k));
        __m512bh wv3 = (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + k));

        // Token 0
        __m512bh iv0 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k));
        acc_t0_r0 = _mm512_dpbf16_ps(acc_t0_r0, iv0, wv0);
        acc_t0_r1 = _mm512_dpbf16_ps(acc_t0_r1, iv0, wv1);
        acc_t0_r2 = _mm512_dpbf16_ps(acc_t0_r2, iv0, wv2);
        acc_t0_r3 = _mm512_dpbf16_ps(acc_t0_r3, iv0, wv3);

        // Token 1
        __m512bh iv1 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k));
        acc_t1_r0 = _mm512_dpbf16_ps(acc_t1_r0, iv1, wv0);
        acc_t1_r1 = _mm512_dpbf16_ps(acc_t1_r1, iv1, wv1);
        acc_t1_r2 = _mm512_dpbf16_ps(acc_t1_r2, iv1, wv2);
        acc_t1_r3 = _mm512_dpbf16_ps(acc_t1_r3, iv1, wv3);

        // Token 2
        __m512bh iv2 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp2 + k));
        acc_t2_r0 = _mm512_dpbf16_ps(acc_t2_r0, iv2, wv0);
        acc_t2_r1 = _mm512_dpbf16_ps(acc_t2_r1, iv2, wv1);
        acc_t2_r2 = _mm512_dpbf16_ps(acc_t2_r2, iv2, wv2);
        acc_t2_r3 = _mm512_dpbf16_ps(acc_t2_r3, iv2, wv3);

        // Token 3
        __m512bh iv3 = (__m512bh)_mm512_loadu_si512((__m512i*)(inp3 + k));
        acc_t3_r0 = _mm512_dpbf16_ps(acc_t3_r0, iv3, wv0);
        acc_t3_r1 = _mm512_dpbf16_ps(acc_t3_r1, iv3, wv1);
        acc_t3_r2 = _mm512_dpbf16_ps(acc_t3_r2, iv3, wv2);
        acc_t3_r3 = _mm512_dpbf16_ps(acc_t3_r3, iv3, wv3);
      }

      // Horizontal reduce and store
      out0[r + 0] = _mm512_reduce_add_ps(acc_t0_r0);
      out0[r + 1] = _mm512_reduce_add_ps(acc_t0_r1);
      out0[r + 2] = _mm512_reduce_add_ps(acc_t0_r2);
      out0[r + 3] = _mm512_reduce_add_ps(acc_t0_r3);
      out1[r + 0] = _mm512_reduce_add_ps(acc_t1_r0);
      out1[r + 1] = _mm512_reduce_add_ps(acc_t1_r1);
      out1[r + 2] = _mm512_reduce_add_ps(acc_t1_r2);
      out1[r + 3] = _mm512_reduce_add_ps(acc_t1_r3);
      out2[r + 0] = _mm512_reduce_add_ps(acc_t2_r0);
      out2[r + 1] = _mm512_reduce_add_ps(acc_t2_r1);
      out2[r + 2] = _mm512_reduce_add_ps(acc_t2_r2);
      out2[r + 3] = _mm512_reduce_add_ps(acc_t2_r3);
      out3[r + 0] = _mm512_reduce_add_ps(acc_t3_r0);
      out3[r + 1] = _mm512_reduce_add_ps(acc_t3_r1);
      out3[r + 2] = _mm512_reduce_add_ps(acc_t3_r2);
      out3[r + 3] = _mm512_reduce_add_ps(acc_t3_r3);

      // Scalar tail for k
      for (int rr = 0; rr < R_BLOCK; rr++) {
        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
        for (int kk = k; kk < k_dim; kk++) {
          float w = GGML_BF16_TO_FP32(weight[(r + rr) * k_dim + kk]);
          sum0 += GGML_BF16_TO_FP32(inp0[kk]) * w;
          sum1 += GGML_BF16_TO_FP32(inp1[kk]) * w;
          sum2 += GGML_BF16_TO_FP32(inp2[kk]) * w;
          sum3 += GGML_BF16_TO_FP32(inp3[kk]) * w;
        }
        out0[r + rr] += sum0;
        out1[r + rr] += sum1;
        out2[r + rr] += sum2;
        out3[r + rr] += sum3;
      }
    }

    // Remainder ranks
    for (; r < rank; r++) {
      const ggml_bf16_t* w_row = weight + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k)), wv);
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k)), wv);
        acc2 = _mm512_dpbf16_ps(acc2, (__m512bh)_mm512_loadu_si512((__m512i*)(inp2 + k)), wv);
        acc3 = _mm512_dpbf16_ps(acc3, (__m512bh)_mm512_loadu_si512((__m512i*)(inp3 + k)), wv);
      }
      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      float sum2 = _mm512_reduce_add_ps(acc2);
      float sum3 = _mm512_reduce_add_ps(acc3);
      for (; k < k_dim; k++) {
        float w = GGML_BF16_TO_FP32(w_row[k]);
        sum0 += GGML_BF16_TO_FP32(inp0[k]) * w;
        sum1 += GGML_BF16_TO_FP32(inp1[k]) * w;
        sum2 += GGML_BF16_TO_FP32(inp2[k]) * w;
        sum3 += GGML_BF16_TO_FP32(inp3[k]) * w;
      }
      out0[r] = sum0;
      out1[r] = sum1;
      out2[r] = sum2;
      out3[r] = sum3;
    }
  }

  // Handle remaining tokens with 2-token kernel
  for (; t + 2 <= num_tokens; t += 2) {
    const ggml_bf16_t* inp0 = input + t * k_dim;
    const ggml_bf16_t* inp1 = input + (t + 1) * k_dim;
    float* out0 = output + t * rank;
    float* out1 = output + (t + 1) * rank;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = weight + r * k_dim;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k));
        acc0 = _mm512_dpbf16_ps(acc0, (__m512bh)_mm512_loadu_si512((__m512i*)(inp0 + k)), wv);
        acc1 = _mm512_dpbf16_ps(acc1, (__m512bh)_mm512_loadu_si512((__m512i*)(inp1 + k)), wv);
      }
      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      for (; k < k_dim; k++) {
        float w = GGML_BF16_TO_FP32(w_row[k]);
        sum0 += GGML_BF16_TO_FP32(inp0[k]) * w;
        sum1 += GGML_BF16_TO_FP32(inp1[k]) * w;
      }
      out0[r] = sum0;
      out1[r] = sum1;
    }
  }

  // Handle remaining single token
  for (; t < num_tokens; t++) {
    const ggml_bf16_t* inp_row = input + t * k_dim;
    float* out_row = output + t * rank;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* w_row = weight + r * k_dim;
      __m512 acc = _mm512_setzero_ps();
      int k = 0;
      for (; k + 32 <= k_dim; k += 32) {
        acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512((__m512i*)(inp_row + k)),
                               (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + k)));
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; k < k_dim; k++) {
        sum += GGML_BF16_TO_FP32(inp_row[k]) * GGML_BF16_TO_FP32(w_row[k]);
      }
      out_row[r] = sum;
    }
  }

  // #if AVX_KERNEL_TRACE_ENABLED
  //   uint64_t trace_end = sft_timer::get_trace_timestamp();
  //   char args_buf[128];
  //   snprintf(args_buf, sizeof(args_buf), "{\"T\":%d,\"K\":%d,\"R\":%d}", num_tokens, k_dim, rank);
  //   sft_timer::add_kernel_trace("lora_bf16_matmul_t4r4", trace_start, trace_end, 0, WorkerPool::thread_local_id,
  //                               args_buf);
  // #endif
}

/**
 * @brief FP32 intermediate × BF16 weight → BF16 output with scale and add
 *
 * Computes: output[t, i] += scale * sum_r(intermediate[t, r] * weight[i, r])
 *
 * Highly optimized version with:
 * - T_BLOCK=4, O_BLOCK=8 for maximum register utilization
 * - Interleaved load/FMA pattern for better pipelining
 * - Vectorized BF16 load/store (8 outputs at a time)
 * - Masked tail handling (no scalar fallback)
 * - Software prefetching for weight data
 *
 * Performance: ~6.6 GFLOPS for R=8, ~38.5 GFLOPS for R=64 (single thread)
 *
 * @param intermediate  Intermediate tensor [num_tokens, rank] in FP32
 * @param weight        Weight tensor [output_dim, rank] in BF16
 * @param output        Output tensor [num_tokens, output_dim] in BF16 (accumulated)
 * @param num_tokens    Number of tokens to process
 * @param rank          LoRA rank (inner dimension)
 * @param output_dim    Output dimension
 * @param scale         Scaling factor for LoRA
 */
inline void lora_fp32_bf16_fused_add(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                                     ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim,
                                     float scale) {
#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_start = sft_timer::get_trace_timestamp();
#endif

  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;
  constexpr int PREFETCH_DISTANCE = 16;

  const __m256 scale_vec = _mm256_set1_ps(scale);
  const int rank_tail = rank & 15;
  const __mmask16 tail_mask = rank_tail ? ((__mmask16)1 << rank_tail) - 1 : 0;

  int t = 0;
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const float* inter0 = intermediate + (t + 0) * rank;
    const float* inter1 = intermediate + (t + 1) * rank;
    const float* inter2 = intermediate + (t + 2) * rank;
    const float* inter3 = intermediate + (t + 3) * rank;
    ggml_bf16_t* out0 = output + (t + 0) * output_dim;
    ggml_bf16_t* out1 = output + (t + 1) * output_dim;
    ggml_bf16_t* out2 = output + (t + 2) * output_dim;
    ggml_bf16_t* out3 = output + (t + 3) * output_dim;

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      // Prefetch weight rows for future iterations
      if (i + O_BLOCK + PREFETCH_DISTANCE * O_BLOCK <= output_dim) {
        _mm_prefetch((const char*)(weight + (i + PREFETCH_DISTANCE * O_BLOCK) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + PREFETCH_DISTANCE * O_BLOCK + 1) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + PREFETCH_DISTANCE * O_BLOCK + 2) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + PREFETCH_DISTANCE * O_BLOCK + 3) * rank), _MM_HINT_T0);
      }

      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;
      const ggml_bf16_t* w4 = weight + (i + 4) * rank;
      const ggml_bf16_t* w5 = weight + (i + 5) * rank;
      const ggml_bf16_t* w6 = weight + (i + 6) * rank;
      const ggml_bf16_t* w7 = weight + (i + 7) * rank;

      // 32 accumulators: 4 tokens × 8 outputs
      __m512 acc_t0_o0 = _mm512_setzero_ps(), acc_t0_o1 = _mm512_setzero_ps();
      __m512 acc_t0_o2 = _mm512_setzero_ps(), acc_t0_o3 = _mm512_setzero_ps();
      __m512 acc_t0_o4 = _mm512_setzero_ps(), acc_t0_o5 = _mm512_setzero_ps();
      __m512 acc_t0_o6 = _mm512_setzero_ps(), acc_t0_o7 = _mm512_setzero_ps();

      __m512 acc_t1_o0 = _mm512_setzero_ps(), acc_t1_o1 = _mm512_setzero_ps();
      __m512 acc_t1_o2 = _mm512_setzero_ps(), acc_t1_o3 = _mm512_setzero_ps();
      __m512 acc_t1_o4 = _mm512_setzero_ps(), acc_t1_o5 = _mm512_setzero_ps();
      __m512 acc_t1_o6 = _mm512_setzero_ps(), acc_t1_o7 = _mm512_setzero_ps();

      __m512 acc_t2_o0 = _mm512_setzero_ps(), acc_t2_o1 = _mm512_setzero_ps();
      __m512 acc_t2_o2 = _mm512_setzero_ps(), acc_t2_o3 = _mm512_setzero_ps();
      __m512 acc_t2_o4 = _mm512_setzero_ps(), acc_t2_o5 = _mm512_setzero_ps();
      __m512 acc_t2_o6 = _mm512_setzero_ps(), acc_t2_o7 = _mm512_setzero_ps();

      __m512 acc_t3_o0 = _mm512_setzero_ps(), acc_t3_o1 = _mm512_setzero_ps();
      __m512 acc_t3_o2 = _mm512_setzero_ps(), acc_t3_o3 = _mm512_setzero_ps();
      __m512 acc_t3_o4 = _mm512_setzero_ps(), acc_t3_o5 = _mm512_setzero_ps();
      __m512 acc_t3_o6 = _mm512_setzero_ps(), acc_t3_o7 = _mm512_setzero_ps();

      int r = 0;
      // Main loop with interleaved loads and FMAs for better pipelining
      for (; r + 16 <= rank; r += 16) {
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

        // Interleave weight loads and FMAs
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r))), 16));
        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);

        __m512 wv1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r))), 16));
        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);

        __m512 wv2 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r))), 16));
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);

        __m512 wv3 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r))), 16));
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);

        __m512 wv4 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w4 + r))), 16));
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
        acc_t0_o4 = _mm512_fmadd_ps(iv0, wv4, acc_t0_o4);
        acc_t1_o4 = _mm512_fmadd_ps(iv1, wv4, acc_t1_o4);

        __m512 wv5 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w5 + r))), 16));
        acc_t2_o4 = _mm512_fmadd_ps(iv2, wv4, acc_t2_o4);
        acc_t3_o4 = _mm512_fmadd_ps(iv3, wv4, acc_t3_o4);
        acc_t0_o5 = _mm512_fmadd_ps(iv0, wv5, acc_t0_o5);
        acc_t1_o5 = _mm512_fmadd_ps(iv1, wv5, acc_t1_o5);

        __m512 wv6 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w6 + r))), 16));
        acc_t2_o5 = _mm512_fmadd_ps(iv2, wv5, acc_t2_o5);
        acc_t3_o5 = _mm512_fmadd_ps(iv3, wv5, acc_t3_o5);
        acc_t0_o6 = _mm512_fmadd_ps(iv0, wv6, acc_t0_o6);
        acc_t1_o6 = _mm512_fmadd_ps(iv1, wv6, acc_t1_o6);

        __m512 wv7 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w7 + r))), 16));
        acc_t2_o6 = _mm512_fmadd_ps(iv2, wv6, acc_t2_o6);
        acc_t3_o6 = _mm512_fmadd_ps(iv3, wv6, acc_t3_o6);
        acc_t0_o7 = _mm512_fmadd_ps(iv0, wv7, acc_t0_o7);
        acc_t1_o7 = _mm512_fmadd_ps(iv1, wv7, acc_t1_o7);
        acc_t2_o7 = _mm512_fmadd_ps(iv2, wv7, acc_t2_o7);
        acc_t3_o7 = _mm512_fmadd_ps(iv3, wv7, acc_t3_o7);
      }

      // Masked tail handling
      if (tail_mask) {
        __m512 iv0 = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1 = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);
        __m512 iv2 = _mm512_maskz_loadu_ps(tail_mask, inter2 + r);
        __m512 iv3 = _mm512_maskz_loadu_ps(tail_mask, inter3 + r);

#define LOAD_W_MASK(ptr) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, ptr + r)), 16))

        __m512 wv0 = LOAD_W_MASK(w0);
        __m512 wv1 = LOAD_W_MASK(w1);
        __m512 wv2 = LOAD_W_MASK(w2);
        __m512 wv3 = LOAD_W_MASK(w3);
        __m512 wv4 = LOAD_W_MASK(w4);
        __m512 wv5 = LOAD_W_MASK(w5);
        __m512 wv6 = LOAD_W_MASK(w6);
        __m512 wv7 = LOAD_W_MASK(w7);

        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);
        acc_t0_o4 = _mm512_fmadd_ps(iv0, wv4, acc_t0_o4);
        acc_t0_o5 = _mm512_fmadd_ps(iv0, wv5, acc_t0_o5);
        acc_t0_o6 = _mm512_fmadd_ps(iv0, wv6, acc_t0_o6);
        acc_t0_o7 = _mm512_fmadd_ps(iv0, wv7, acc_t0_o7);

        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);
        acc_t1_o4 = _mm512_fmadd_ps(iv1, wv4, acc_t1_o4);
        acc_t1_o5 = _mm512_fmadd_ps(iv1, wv5, acc_t1_o5);
        acc_t1_o6 = _mm512_fmadd_ps(iv1, wv6, acc_t1_o6);
        acc_t1_o7 = _mm512_fmadd_ps(iv1, wv7, acc_t1_o7);

        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);
        acc_t2_o4 = _mm512_fmadd_ps(iv2, wv4, acc_t2_o4);
        acc_t2_o5 = _mm512_fmadd_ps(iv2, wv5, acc_t2_o5);
        acc_t2_o6 = _mm512_fmadd_ps(iv2, wv6, acc_t2_o6);
        acc_t2_o7 = _mm512_fmadd_ps(iv2, wv7, acc_t2_o7);

        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
        acc_t3_o4 = _mm512_fmadd_ps(iv3, wv4, acc_t3_o4);
        acc_t3_o5 = _mm512_fmadd_ps(iv3, wv5, acc_t3_o5);
        acc_t3_o6 = _mm512_fmadd_ps(iv3, wv6, acc_t3_o6);
        acc_t3_o7 = _mm512_fmadd_ps(iv3, wv7, acc_t3_o7);

#undef LOAD_W_MASK
      }

      // Reduce 8 accumulators to __m256 (8 floats) for each token
      // Token 0
      __m256 sum_t0 = _mm256_set_ps(_mm512_reduce_add_ps(acc_t0_o7), _mm512_reduce_add_ps(acc_t0_o6),
                                    _mm512_reduce_add_ps(acc_t0_o5), _mm512_reduce_add_ps(acc_t0_o4),
                                    _mm512_reduce_add_ps(acc_t0_o3), _mm512_reduce_add_ps(acc_t0_o2),
                                    _mm512_reduce_add_ps(acc_t0_o1), _mm512_reduce_add_ps(acc_t0_o0));
      // Token 1
      __m256 sum_t1 = _mm256_set_ps(_mm512_reduce_add_ps(acc_t1_o7), _mm512_reduce_add_ps(acc_t1_o6),
                                    _mm512_reduce_add_ps(acc_t1_o5), _mm512_reduce_add_ps(acc_t1_o4),
                                    _mm512_reduce_add_ps(acc_t1_o3), _mm512_reduce_add_ps(acc_t1_o2),
                                    _mm512_reduce_add_ps(acc_t1_o1), _mm512_reduce_add_ps(acc_t1_o0));
      // Token 2
      __m256 sum_t2 = _mm256_set_ps(_mm512_reduce_add_ps(acc_t2_o7), _mm512_reduce_add_ps(acc_t2_o6),
                                    _mm512_reduce_add_ps(acc_t2_o5), _mm512_reduce_add_ps(acc_t2_o4),
                                    _mm512_reduce_add_ps(acc_t2_o3), _mm512_reduce_add_ps(acc_t2_o2),
                                    _mm512_reduce_add_ps(acc_t2_o1), _mm512_reduce_add_ps(acc_t2_o0));
      // Token 3
      __m256 sum_t3 = _mm256_set_ps(_mm512_reduce_add_ps(acc_t3_o7), _mm512_reduce_add_ps(acc_t3_o6),
                                    _mm512_reduce_add_ps(acc_t3_o5), _mm512_reduce_add_ps(acc_t3_o4),
                                    _mm512_reduce_add_ps(acc_t3_o3), _mm512_reduce_add_ps(acc_t3_o2),
                                    _mm512_reduce_add_ps(acc_t3_o1), _mm512_reduce_add_ps(acc_t3_o0));

      // Apply scale
      sum_t0 = _mm256_mul_ps(sum_t0, scale_vec);
      sum_t1 = _mm256_mul_ps(sum_t1, scale_vec);
      sum_t2 = _mm256_mul_ps(sum_t2, scale_vec);
      sum_t3 = _mm256_mul_ps(sum_t3, scale_vec);

      // Vectorized load/add/store for output (8 BF16 values at a time)
      __m256 out_t0 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out0 + i))), 16));
      __m256 out_t1 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out1 + i))), 16));
      __m256 out_t2 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out2 + i))), 16));
      __m256 out_t3 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out3 + i))), 16));

      out_t0 = _mm256_add_ps(out_t0, sum_t0);
      out_t1 = _mm256_add_ps(out_t1, sum_t1);
      out_t2 = _mm256_add_ps(out_t2, sum_t2);
      out_t3 = _mm256_add_ps(out_t3, sum_t3);

      // Convert FP32 -> BF16 and store
      __m128bh bf16_t0 = _mm256_cvtneps_pbh(out_t0);
      __m128bh bf16_t1 = _mm256_cvtneps_pbh(out_t1);
      __m128bh bf16_t2 = _mm256_cvtneps_pbh(out_t2);
      __m128bh bf16_t3 = _mm256_cvtneps_pbh(out_t3);

      _mm_storeu_si128((__m128i*)(out0 + i), (__m128i)bf16_t0);
      _mm_storeu_si128((__m128i*)(out1 + i), (__m128i)bf16_t1);
      _mm_storeu_si128((__m128i*)(out2 + i), (__m128i)bf16_t2);
      _mm_storeu_si128((__m128i*)(out3 + i), (__m128i)bf16_t3);
    }

    // Remainder outputs (< O_BLOCK)
    for (; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r), wv, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r), wv, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(inter2 + r), wv, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(inter3 + r), wv, acc3);
      }
      if (tail_mask) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w_row + r)), 16));
        acc0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter0 + r), wv, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter1 + r), wv, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter2 + r), wv, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter3 + r), wv, acc3);
      }

      float s0 = _mm512_reduce_add_ps(acc0) * scale;
      float s1 = _mm512_reduce_add_ps(acc1) * scale;
      float s2 = _mm512_reduce_add_ps(acc2) * scale;
      float s3 = _mm512_reduce_add_ps(acc3) * scale;

      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3);
    }
  }

  // Handle remaining tokens (< T_BLOCK)
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    for (int i = 0; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc = _mm512_setzero_ps();
      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r), wv, acc);
      }
      if (tail_mask) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w_row + r)), 16));
        acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter_row + r), wv, acc);
      }
      float sum = _mm512_reduce_add_ps(acc) * scale;
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum);
    }
  }

#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_end = sft_timer::get_trace_timestamp();
  char args_buf[128];
  snprintf(args_buf, sizeof(args_buf), "{\"T\":%d,\"R\":%d,\"O\":%d}", num_tokens, rank, output_dim);
  sft_timer::add_kernel_trace("lora_fp32_bf16_fused_add", trace_start, trace_end, 0, WorkerPool::thread_local_id,
                              args_buf);
#endif
}

/**
 * @brief FP32 intermediate × BF16 weight (transposed layout) → BF16 output with scale and add
 *
 * Computes: output[t, i] += scale * sum_r(intermediate[t, r] * weight[r, i])
 *
 * This variant handles weight in [rank, output_dim] layout (transposed from standard).
 * The weight access pattern is contiguous along output_dim, enabling efficient vectorized loads.
 *
 * Optimizations:
 * - T_BLOCK=4, O_BLOCK=32 for maximum register utilization
 * - R_UNROLL=4: unroll 4 ranks per iteration for better pipelining
 * - Contiguous weight loads (32 outputs at once)
 * - Vectorized BF16 load/store
 *
 * Performance: ~68-87 GFLOPS (2x speedup vs baseline), tested on R=8-64
 *
 * @param intermediate  Intermediate tensor [num_tokens, rank] in FP32
 * @param weight        Weight tensor [rank, output_dim] in BF16 (transposed layout)
 * @param output        Output tensor [num_tokens, output_dim] in BF16 (accumulated)
 * @param num_tokens    Number of tokens to process
 * @param rank          LoRA rank (inner dimension)
 * @param output_dim    Output dimension
 * @param scale         Scaling factor for LoRA
 */
inline void lora_fp32_bf16_fused_add_wt(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                                        ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim,
                                        float scale) {
#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_start = sft_timer::get_trace_timestamp();
#endif

  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 32;
  constexpr int R_UNROLL = 4;

  const __m512 scale_vec = _mm512_set1_ps(scale);
  const int rank_main = (rank / R_UNROLL) * R_UNROLL;

  int t = 0;
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const float* inter0 = intermediate + (t + 0) * rank;
    const float* inter1 = intermediate + (t + 1) * rank;
    const float* inter2 = intermediate + (t + 2) * rank;
    const float* inter3 = intermediate + (t + 3) * rank;
    ggml_bf16_t* out0 = output + (t + 0) * output_dim;
    ggml_bf16_t* out1 = output + (t + 1) * output_dim;
    ggml_bf16_t* out2 = output + (t + 2) * output_dim;
    ggml_bf16_t* out3 = output + (t + 3) * output_dim;

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      __m512 acc_t0_0 = _mm512_setzero_ps(), acc_t0_1 = _mm512_setzero_ps();
      __m512 acc_t1_0 = _mm512_setzero_ps(), acc_t1_1 = _mm512_setzero_ps();
      __m512 acc_t2_0 = _mm512_setzero_ps(), acc_t2_1 = _mm512_setzero_ps();
      __m512 acc_t3_0 = _mm512_setzero_ps(), acc_t3_1 = _mm512_setzero_ps();

      // Main loop: 4 ranks per iteration for better pipelining
      int r = 0;
      for (; r < rank_main; r += R_UNROLL) {
        __m512 iv0_r0 = _mm512_set1_ps(inter0[r + 0]), iv0_r1 = _mm512_set1_ps(inter0[r + 1]);
        __m512 iv0_r2 = _mm512_set1_ps(inter0[r + 2]), iv0_r3 = _mm512_set1_ps(inter0[r + 3]);
        __m512 iv1_r0 = _mm512_set1_ps(inter1[r + 0]), iv1_r1 = _mm512_set1_ps(inter1[r + 1]);
        __m512 iv1_r2 = _mm512_set1_ps(inter1[r + 2]), iv1_r3 = _mm512_set1_ps(inter1[r + 3]);
        __m512 iv2_r0 = _mm512_set1_ps(inter2[r + 0]), iv2_r1 = _mm512_set1_ps(inter2[r + 1]);
        __m512 iv2_r2 = _mm512_set1_ps(inter2[r + 2]), iv2_r3 = _mm512_set1_ps(inter2[r + 3]);
        __m512 iv3_r0 = _mm512_set1_ps(inter3[r + 0]), iv3_r1 = _mm512_set1_ps(inter3[r + 1]);
        __m512 iv3_r2 = _mm512_set1_ps(inter3[r + 2]), iv3_r3 = _mm512_set1_ps(inter3[r + 3]);

        const ggml_bf16_t* w0 = weight + (r + 0) * output_dim + i;
        const ggml_bf16_t* w1 = weight + (r + 1) * output_dim + i;
        const ggml_bf16_t* w2 = weight + (r + 2) * output_dim + i;
        const ggml_bf16_t* w3 = weight + (r + 3) * output_dim + i;

        __m512 wv0_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w0)), 16));
        __m512 wv0_1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + 16))), 16));
        __m512 wv1_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w1)), 16));
        __m512 wv1_1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + 16))), 16));
        __m512 wv2_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w2)), 16));
        __m512 wv2_1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + 16))), 16));
        __m512 wv3_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w3)), 16));
        __m512 wv3_1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + 16))), 16));

        // Token 0
        acc_t0_0 = _mm512_fmadd_ps(iv0_r0, wv0_0, acc_t0_0);
        acc_t0_1 = _mm512_fmadd_ps(iv0_r0, wv0_1, acc_t0_1);
        acc_t0_0 = _mm512_fmadd_ps(iv0_r1, wv1_0, acc_t0_0);
        acc_t0_1 = _mm512_fmadd_ps(iv0_r1, wv1_1, acc_t0_1);
        acc_t0_0 = _mm512_fmadd_ps(iv0_r2, wv2_0, acc_t0_0);
        acc_t0_1 = _mm512_fmadd_ps(iv0_r2, wv2_1, acc_t0_1);
        acc_t0_0 = _mm512_fmadd_ps(iv0_r3, wv3_0, acc_t0_0);
        acc_t0_1 = _mm512_fmadd_ps(iv0_r3, wv3_1, acc_t0_1);
        // Token 1
        acc_t1_0 = _mm512_fmadd_ps(iv1_r0, wv0_0, acc_t1_0);
        acc_t1_1 = _mm512_fmadd_ps(iv1_r0, wv0_1, acc_t1_1);
        acc_t1_0 = _mm512_fmadd_ps(iv1_r1, wv1_0, acc_t1_0);
        acc_t1_1 = _mm512_fmadd_ps(iv1_r1, wv1_1, acc_t1_1);
        acc_t1_0 = _mm512_fmadd_ps(iv1_r2, wv2_0, acc_t1_0);
        acc_t1_1 = _mm512_fmadd_ps(iv1_r2, wv2_1, acc_t1_1);
        acc_t1_0 = _mm512_fmadd_ps(iv1_r3, wv3_0, acc_t1_0);
        acc_t1_1 = _mm512_fmadd_ps(iv1_r3, wv3_1, acc_t1_1);
        // Token 2
        acc_t2_0 = _mm512_fmadd_ps(iv2_r0, wv0_0, acc_t2_0);
        acc_t2_1 = _mm512_fmadd_ps(iv2_r0, wv0_1, acc_t2_1);
        acc_t2_0 = _mm512_fmadd_ps(iv2_r1, wv1_0, acc_t2_0);
        acc_t2_1 = _mm512_fmadd_ps(iv2_r1, wv1_1, acc_t2_1);
        acc_t2_0 = _mm512_fmadd_ps(iv2_r2, wv2_0, acc_t2_0);
        acc_t2_1 = _mm512_fmadd_ps(iv2_r2, wv2_1, acc_t2_1);
        acc_t2_0 = _mm512_fmadd_ps(iv2_r3, wv3_0, acc_t2_0);
        acc_t2_1 = _mm512_fmadd_ps(iv2_r3, wv3_1, acc_t2_1);
        // Token 3
        acc_t3_0 = _mm512_fmadd_ps(iv3_r0, wv0_0, acc_t3_0);
        acc_t3_1 = _mm512_fmadd_ps(iv3_r0, wv0_1, acc_t3_1);
        acc_t3_0 = _mm512_fmadd_ps(iv3_r1, wv1_0, acc_t3_0);
        acc_t3_1 = _mm512_fmadd_ps(iv3_r1, wv1_1, acc_t3_1);
        acc_t3_0 = _mm512_fmadd_ps(iv3_r2, wv2_0, acc_t3_0);
        acc_t3_1 = _mm512_fmadd_ps(iv3_r2, wv2_1, acc_t3_1);
        acc_t3_0 = _mm512_fmadd_ps(iv3_r3, wv3_0, acc_t3_0);
        acc_t3_1 = _mm512_fmadd_ps(iv3_r3, wv3_1, acc_t3_1);
      }

      // Remainder ranks
      for (; r < rank; r++) {
        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);
        const ggml_bf16_t* w_ptr = weight + r * output_dim + i;
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr)), 16));
        __m512 wv1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr + 16))), 16));
        acc_t0_0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_0);
        acc_t0_1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_1);
        acc_t1_0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_0);
        acc_t1_1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_1);
        acc_t2_0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_0);
        acc_t2_1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_1);
        acc_t3_0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_0);
        acc_t3_1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_1);
      }

      // Apply scale
      acc_t0_0 = _mm512_mul_ps(acc_t0_0, scale_vec);
      acc_t0_1 = _mm512_mul_ps(acc_t0_1, scale_vec);
      acc_t1_0 = _mm512_mul_ps(acc_t1_0, scale_vec);
      acc_t1_1 = _mm512_mul_ps(acc_t1_1, scale_vec);
      acc_t2_0 = _mm512_mul_ps(acc_t2_0, scale_vec);
      acc_t2_1 = _mm512_mul_ps(acc_t2_1, scale_vec);
      acc_t3_0 = _mm512_mul_ps(acc_t3_0, scale_vec);
      acc_t3_1 = _mm512_mul_ps(acc_t3_1, scale_vec);

      // Load current output, add, store (32 values per token)
      // Token 0
      __m512 cur0_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i))), 16));
      __m512 cur0_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i + 16))), 16));
      cur0_0 = _mm512_add_ps(cur0_0, acc_t0_0);
      cur0_1 = _mm512_add_ps(cur0_1, acc_t0_1);
      _mm256_storeu_si256((__m256i*)(out0 + i), (__m256i)_mm512_cvtneps_pbh(cur0_0));
      _mm256_storeu_si256((__m256i*)(out0 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur0_1));

      // Token 1
      __m512 cur1_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i))), 16));
      __m512 cur1_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i + 16))), 16));
      cur1_0 = _mm512_add_ps(cur1_0, acc_t1_0);
      cur1_1 = _mm512_add_ps(cur1_1, acc_t1_1);
      _mm256_storeu_si256((__m256i*)(out1 + i), (__m256i)_mm512_cvtneps_pbh(cur1_0));
      _mm256_storeu_si256((__m256i*)(out1 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur1_1));

      // Token 2
      __m512 cur2_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i))), 16));
      __m512 cur2_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i + 16))), 16));
      cur2_0 = _mm512_add_ps(cur2_0, acc_t2_0);
      cur2_1 = _mm512_add_ps(cur2_1, acc_t2_1);
      _mm256_storeu_si256((__m256i*)(out2 + i), (__m256i)_mm512_cvtneps_pbh(cur2_0));
      _mm256_storeu_si256((__m256i*)(out2 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur2_1));

      // Token 3
      __m512 cur3_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i))), 16));
      __m512 cur3_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i + 16))), 16));
      cur3_0 = _mm512_add_ps(cur3_0, acc_t3_0);
      cur3_1 = _mm512_add_ps(cur3_1, acc_t3_1);
      _mm256_storeu_si256((__m256i*)(out3 + i), (__m256i)_mm512_cvtneps_pbh(cur3_0));
      _mm256_storeu_si256((__m256i*)(out3 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur3_1));
    }

    // Handle remaining outputs (< O_BLOCK, process 16 at a time)
    for (; i + 16 <= output_dim; i += 16) {
      __m512 acc_t0 = _mm512_setzero_ps();
      __m512 acc_t1 = _mm512_setzero_ps();
      __m512 acc_t2 = _mm512_setzero_ps();
      __m512 acc_t3 = _mm512_setzero_ps();

      for (int r = 0; r < rank; r++) {
        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);

        const ggml_bf16_t* w_ptr = weight + r * output_dim + i;
        __m512 wv =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr)), 16));

        acc_t0 = _mm512_fmadd_ps(iv0, wv, acc_t0);
        acc_t1 = _mm512_fmadd_ps(iv1, wv, acc_t1);
        acc_t2 = _mm512_fmadd_ps(iv2, wv, acc_t2);
        acc_t3 = _mm512_fmadd_ps(iv3, wv, acc_t3);
      }

      acc_t0 = _mm512_mul_ps(acc_t0, scale_vec);
      acc_t1 = _mm512_mul_ps(acc_t1, scale_vec);
      acc_t2 = _mm512_mul_ps(acc_t2, scale_vec);
      acc_t3 = _mm512_mul_ps(acc_t3, scale_vec);

      __m512 cur0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i))), 16));
      __m512 cur1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i))), 16));
      __m512 cur2 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i))), 16));
      __m512 cur3 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i))), 16));

      cur0 = _mm512_add_ps(cur0, acc_t0);
      cur1 = _mm512_add_ps(cur1, acc_t1);
      cur2 = _mm512_add_ps(cur2, acc_t2);
      cur3 = _mm512_add_ps(cur3, acc_t3);

      _mm256_storeu_si256((__m256i*)(out0 + i), (__m256i)_mm512_cvtneps_pbh(cur0));
      _mm256_storeu_si256((__m256i*)(out1 + i), (__m256i)_mm512_cvtneps_pbh(cur1));
      _mm256_storeu_si256((__m256i*)(out2 + i), (__m256i)_mm512_cvtneps_pbh(cur2));
      _mm256_storeu_si256((__m256i*)(out3 + i), (__m256i)_mm512_cvtneps_pbh(cur3));
    }

    // Scalar remainder for tail outputs
    for (; i < output_dim; i++) {
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      for (int r = 0; r < rank; r++) {
        float w = GGML_BF16_TO_FP32(weight[r * output_dim + i]);
        sum0 += inter0[r] * w;
        sum1 += inter1[r] * w;
        sum2 += inter2[r] * w;
        sum3 += inter3[r] * w;
      }
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + sum0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + sum1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + sum2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + sum3 * scale);
    }
  }

  // Handle remaining tokens (< T_BLOCK)
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    int i = 0;
    for (; i + 16 <= output_dim; i += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int r = 0; r < rank; r++) {
        __m512 iv = _mm512_set1_ps(inter_row[r]);
        const ggml_bf16_t* w_ptr = weight + r * output_dim + i;
        __m512 wv =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr)), 16));
        acc = _mm512_fmadd_ps(iv, wv, acc);
      }
      acc = _mm512_mul_ps(acc, scale_vec);

      __m512 cur = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out_row + i))), 16));
      cur = _mm512_add_ps(cur, acc);
      _mm256_storeu_si256((__m256i*)(out_row + i), (__m256i)_mm512_cvtneps_pbh(cur));
    }

    for (; i < output_dim; i++) {
      float sum = 0.0f;
      for (int r = 0; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(weight[r * output_dim + i]);
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }

#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_end = sft_timer::get_trace_timestamp();
  char args_buf[128];
  snprintf(args_buf, sizeof(args_buf), "{\"T\":%d,\"R\":%d,\"O\":%d}", num_tokens, rank, output_dim);
  sft_timer::add_kernel_trace("lora_fp32_bf16_fused_add_wt", trace_start, trace_end, 0, WorkerPool::thread_local_id,
                              args_buf);
#endif
}

// ============================================================================
// Pre-transposed weight utilities and optimized kernel
//
// Transpose weight from [output_dim][rank] to [rank][output_dim]
// This enables contiguous memory access in the inner loop for better cache efficiency.
// ============================================================================

// ============================================================================
// AVX-512 In-Register Transpose Kernels (4x4, 8x8, 16x16)
// ============================================================================

/**
 * @brief Transpose 4x4 BF16 block in-register
 */
inline void transpose_4x4_bf16(const ggml_bf16_t* src, int src_stride, ggml_bf16_t* dst, int dst_stride) {
  __m128i r0 = _mm_loadl_epi64((__m128i*)(src + 0 * src_stride));
  __m128i r1 = _mm_loadl_epi64((__m128i*)(src + 1 * src_stride));
  __m128i r2 = _mm_loadl_epi64((__m128i*)(src + 2 * src_stride));
  __m128i r3 = _mm_loadl_epi64((__m128i*)(src + 3 * src_stride));

  __m128i t0 = _mm_unpacklo_epi16(r0, r1);
  __m128i t1 = _mm_unpacklo_epi16(r2, r3);

  __m128i s0 = _mm_unpacklo_epi32(t0, t1);
  __m128i s1 = _mm_unpackhi_epi32(t0, t1);

  _mm_storel_epi64((__m128i*)(dst + 0 * dst_stride), s0);
  _mm_storel_epi64((__m128i*)(dst + 1 * dst_stride), _mm_srli_si128(s0, 8));
  _mm_storel_epi64((__m128i*)(dst + 2 * dst_stride), s1);
  _mm_storel_epi64((__m128i*)(dst + 3 * dst_stride), _mm_srli_si128(s1, 8));
}

/**
 * @brief Transpose 8x8 BF16 block in-register using SSE
 */
inline void transpose_8x8_bf16(const ggml_bf16_t* src, int src_stride, ggml_bf16_t* dst, int dst_stride) {
  __m128i r0 = _mm_loadu_si128((__m128i*)(src + 0 * src_stride));
  __m128i r1 = _mm_loadu_si128((__m128i*)(src + 1 * src_stride));
  __m128i r2 = _mm_loadu_si128((__m128i*)(src + 2 * src_stride));
  __m128i r3 = _mm_loadu_si128((__m128i*)(src + 3 * src_stride));
  __m128i r4 = _mm_loadu_si128((__m128i*)(src + 4 * src_stride));
  __m128i r5 = _mm_loadu_si128((__m128i*)(src + 5 * src_stride));
  __m128i r6 = _mm_loadu_si128((__m128i*)(src + 6 * src_stride));
  __m128i r7 = _mm_loadu_si128((__m128i*)(src + 7 * src_stride));

  // Step 1: Interleave 16-bit
  __m128i t0 = _mm_unpacklo_epi16(r0, r1);
  __m128i t1 = _mm_unpackhi_epi16(r0, r1);
  __m128i t2 = _mm_unpacklo_epi16(r2, r3);
  __m128i t3 = _mm_unpackhi_epi16(r2, r3);
  __m128i t4 = _mm_unpacklo_epi16(r4, r5);
  __m128i t5 = _mm_unpackhi_epi16(r4, r5);
  __m128i t6 = _mm_unpacklo_epi16(r6, r7);
  __m128i t7 = _mm_unpackhi_epi16(r6, r7);

  // Step 2: Interleave 32-bit
  r0 = _mm_unpacklo_epi32(t0, t2);
  r1 = _mm_unpackhi_epi32(t0, t2);
  r2 = _mm_unpacklo_epi32(t1, t3);
  r3 = _mm_unpackhi_epi32(t1, t3);
  r4 = _mm_unpacklo_epi32(t4, t6);
  r5 = _mm_unpackhi_epi32(t4, t6);
  r6 = _mm_unpacklo_epi32(t5, t7);
  r7 = _mm_unpackhi_epi32(t5, t7);

  // Step 3: Interleave 64-bit
  t0 = _mm_unpacklo_epi64(r0, r4);
  t1 = _mm_unpackhi_epi64(r0, r4);
  t2 = _mm_unpacklo_epi64(r1, r5);
  t3 = _mm_unpackhi_epi64(r1, r5);
  t4 = _mm_unpacklo_epi64(r2, r6);
  t5 = _mm_unpackhi_epi64(r2, r6);
  t6 = _mm_unpacklo_epi64(r3, r7);
  t7 = _mm_unpackhi_epi64(r3, r7);

  _mm_storeu_si128((__m128i*)(dst + 0 * dst_stride), t0);
  _mm_storeu_si128((__m128i*)(dst + 1 * dst_stride), t1);
  _mm_storeu_si128((__m128i*)(dst + 2 * dst_stride), t2);
  _mm_storeu_si128((__m128i*)(dst + 3 * dst_stride), t3);
  _mm_storeu_si128((__m128i*)(dst + 4 * dst_stride), t4);
  _mm_storeu_si128((__m128i*)(dst + 5 * dst_stride), t5);
  _mm_storeu_si128((__m128i*)(dst + 6 * dst_stride), t6);
  _mm_storeu_si128((__m128i*)(dst + 7 * dst_stride), t7);
}

/**
 * @brief Transpose 16x16 BF16 block in-register using AVX2
 */
inline void transpose_16x16_bf16(const ggml_bf16_t* src, int src_stride, ggml_bf16_t* dst, int dst_stride) {
  __m256i r0 = _mm256_loadu_si256((__m256i*)(src + 0 * src_stride));
  __m256i r1 = _mm256_loadu_si256((__m256i*)(src + 1 * src_stride));
  __m256i r2 = _mm256_loadu_si256((__m256i*)(src + 2 * src_stride));
  __m256i r3 = _mm256_loadu_si256((__m256i*)(src + 3 * src_stride));
  __m256i r4 = _mm256_loadu_si256((__m256i*)(src + 4 * src_stride));
  __m256i r5 = _mm256_loadu_si256((__m256i*)(src + 5 * src_stride));
  __m256i r6 = _mm256_loadu_si256((__m256i*)(src + 6 * src_stride));
  __m256i r7 = _mm256_loadu_si256((__m256i*)(src + 7 * src_stride));
  __m256i r8 = _mm256_loadu_si256((__m256i*)(src + 8 * src_stride));
  __m256i r9 = _mm256_loadu_si256((__m256i*)(src + 9 * src_stride));
  __m256i r10 = _mm256_loadu_si256((__m256i*)(src + 10 * src_stride));
  __m256i r11 = _mm256_loadu_si256((__m256i*)(src + 11 * src_stride));
  __m256i r12 = _mm256_loadu_si256((__m256i*)(src + 12 * src_stride));
  __m256i r13 = _mm256_loadu_si256((__m256i*)(src + 13 * src_stride));
  __m256i r14 = _mm256_loadu_si256((__m256i*)(src + 14 * src_stride));
  __m256i r15 = _mm256_loadu_si256((__m256i*)(src + 15 * src_stride));

  // Step 1: Interleave 16-bit
  __m256i t0 = _mm256_unpacklo_epi16(r0, r1);
  __m256i t1 = _mm256_unpackhi_epi16(r0, r1);
  __m256i t2 = _mm256_unpacklo_epi16(r2, r3);
  __m256i t3 = _mm256_unpackhi_epi16(r2, r3);
  __m256i t4 = _mm256_unpacklo_epi16(r4, r5);
  __m256i t5 = _mm256_unpackhi_epi16(r4, r5);
  __m256i t6 = _mm256_unpacklo_epi16(r6, r7);
  __m256i t7 = _mm256_unpackhi_epi16(r6, r7);
  __m256i t8 = _mm256_unpacklo_epi16(r8, r9);
  __m256i t9 = _mm256_unpackhi_epi16(r8, r9);
  __m256i t10 = _mm256_unpacklo_epi16(r10, r11);
  __m256i t11 = _mm256_unpackhi_epi16(r10, r11);
  __m256i t12 = _mm256_unpacklo_epi16(r12, r13);
  __m256i t13 = _mm256_unpackhi_epi16(r12, r13);
  __m256i t14 = _mm256_unpacklo_epi16(r14, r15);
  __m256i t15 = _mm256_unpackhi_epi16(r14, r15);

  // Step 2: Interleave 32-bit
  r0 = _mm256_unpacklo_epi32(t0, t2);
  r1 = _mm256_unpackhi_epi32(t0, t2);
  r2 = _mm256_unpacklo_epi32(t1, t3);
  r3 = _mm256_unpackhi_epi32(t1, t3);
  r4 = _mm256_unpacklo_epi32(t4, t6);
  r5 = _mm256_unpackhi_epi32(t4, t6);
  r6 = _mm256_unpacklo_epi32(t5, t7);
  r7 = _mm256_unpackhi_epi32(t5, t7);
  r8 = _mm256_unpacklo_epi32(t8, t10);
  r9 = _mm256_unpackhi_epi32(t8, t10);
  r10 = _mm256_unpacklo_epi32(t9, t11);
  r11 = _mm256_unpackhi_epi32(t9, t11);
  r12 = _mm256_unpacklo_epi32(t12, t14);
  r13 = _mm256_unpackhi_epi32(t12, t14);
  r14 = _mm256_unpacklo_epi32(t13, t15);
  r15 = _mm256_unpackhi_epi32(t13, t15);

  // Step 3: Interleave 64-bit
  t0 = _mm256_unpacklo_epi64(r0, r4);
  t1 = _mm256_unpackhi_epi64(r0, r4);
  t2 = _mm256_unpacklo_epi64(r1, r5);
  t3 = _mm256_unpackhi_epi64(r1, r5);
  t4 = _mm256_unpacklo_epi64(r2, r6);
  t5 = _mm256_unpackhi_epi64(r2, r6);
  t6 = _mm256_unpacklo_epi64(r3, r7);
  t7 = _mm256_unpackhi_epi64(r3, r7);
  t8 = _mm256_unpacklo_epi64(r8, r12);
  t9 = _mm256_unpackhi_epi64(r8, r12);
  t10 = _mm256_unpacklo_epi64(r9, r13);
  t11 = _mm256_unpackhi_epi64(r9, r13);
  t12 = _mm256_unpacklo_epi64(r10, r14);
  t13 = _mm256_unpackhi_epi64(r10, r14);
  t14 = _mm256_unpacklo_epi64(r11, r15);
  t15 = _mm256_unpackhi_epi64(r11, r15);

  // Step 4: Permute 128-bit lanes
  r0 = _mm256_permute2x128_si256(t0, t8, 0x20);
  r8 = _mm256_permute2x128_si256(t0, t8, 0x31);
  r1 = _mm256_permute2x128_si256(t1, t9, 0x20);
  r9 = _mm256_permute2x128_si256(t1, t9, 0x31);
  r2 = _mm256_permute2x128_si256(t2, t10, 0x20);
  r10 = _mm256_permute2x128_si256(t2, t10, 0x31);
  r3 = _mm256_permute2x128_si256(t3, t11, 0x20);
  r11 = _mm256_permute2x128_si256(t3, t11, 0x31);
  r4 = _mm256_permute2x128_si256(t4, t12, 0x20);
  r12 = _mm256_permute2x128_si256(t4, t12, 0x31);
  r5 = _mm256_permute2x128_si256(t5, t13, 0x20);
  r13 = _mm256_permute2x128_si256(t5, t13, 0x31);
  r6 = _mm256_permute2x128_si256(t6, t14, 0x20);
  r14 = _mm256_permute2x128_si256(t6, t14, 0x31);
  r7 = _mm256_permute2x128_si256(t7, t15, 0x20);
  r15 = _mm256_permute2x128_si256(t7, t15, 0x31);

  _mm256_storeu_si256((__m256i*)(dst + 0 * dst_stride), r0);
  _mm256_storeu_si256((__m256i*)(dst + 1 * dst_stride), r1);
  _mm256_storeu_si256((__m256i*)(dst + 2 * dst_stride), r2);
  _mm256_storeu_si256((__m256i*)(dst + 3 * dst_stride), r3);
  _mm256_storeu_si256((__m256i*)(dst + 4 * dst_stride), r4);
  _mm256_storeu_si256((__m256i*)(dst + 5 * dst_stride), r5);
  _mm256_storeu_si256((__m256i*)(dst + 6 * dst_stride), r6);
  _mm256_storeu_si256((__m256i*)(dst + 7 * dst_stride), r7);
  _mm256_storeu_si256((__m256i*)(dst + 8 * dst_stride), r8);
  _mm256_storeu_si256((__m256i*)(dst + 9 * dst_stride), r9);
  _mm256_storeu_si256((__m256i*)(dst + 10 * dst_stride), r10);
  _mm256_storeu_si256((__m256i*)(dst + 11 * dst_stride), r11);
  _mm256_storeu_si256((__m256i*)(dst + 12 * dst_stride), r12);
  _mm256_storeu_si256((__m256i*)(dst + 13 * dst_stride), r13);
  _mm256_storeu_si256((__m256i*)(dst + 14 * dst_stride), r14);
  _mm256_storeu_si256((__m256i*)(dst + 15 * dst_stride), r15);
}

/**
 * @brief Transpose LoRA B weight from [output_dim][rank] to [rank][output_dim]
 *
 * Uses AVX2/SSE in-register transpose kernels (16x16, 8x8, 4x4) for high performance.
 * Achieves 2-6x speedup over naive implementation.
 *
 * @param src     Source weight [output_dim][rank]
 * @param dst     Destination weight [rank][output_dim]
 * @param output_dim Output dimension
 * @param rank    LoRA rank
 */
inline void transpose_lora_weight(const ggml_bf16_t* __restrict src, ggml_bf16_t* __restrict dst, int output_dim,
                                  int rank) {
  int r = 0;

  // Process 16x16 blocks
  for (; r + 16 <= rank; r += 16) {
    int i = 0;
    for (; i + 16 <= output_dim; i += 16) {
      transpose_16x16_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
    }
    // 8x8 for remainder columns
    for (; i + 8 <= output_dim; i += 8) {
      transpose_8x8_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
      transpose_8x8_bf16(src + i * rank + r + 8, rank, dst + (r + 8) * output_dim + i, output_dim);
    }
    // 4x4 for remainder columns
    for (; i + 4 <= output_dim; i += 4) {
      transpose_4x4_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
      transpose_4x4_bf16(src + i * rank + r + 4, rank, dst + (r + 4) * output_dim + i, output_dim);
      transpose_4x4_bf16(src + i * rank + r + 8, rank, dst + (r + 8) * output_dim + i, output_dim);
      transpose_4x4_bf16(src + i * rank + r + 12, rank, dst + (r + 12) * output_dim + i, output_dim);
    }
    // Scalar remainder
    for (; i < output_dim; i++) {
      for (int rr = 0; rr < 16; rr++) {
        dst[(r + rr) * output_dim + i] = src[i * rank + (r + rr)];
      }
    }
  }

  // Process 8x8 blocks for remaining rows
  for (; r + 8 <= rank; r += 8) {
    int i = 0;
    for (; i + 8 <= output_dim; i += 8) {
      transpose_8x8_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
    }
    for (; i + 4 <= output_dim; i += 4) {
      transpose_4x4_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
      transpose_4x4_bf16(src + i * rank + r + 4, rank, dst + (r + 4) * output_dim + i, output_dim);
    }
    for (; i < output_dim; i++) {
      for (int rr = 0; rr < 8; rr++) {
        dst[(r + rr) * output_dim + i] = src[i * rank + (r + rr)];
      }
    }
  }

  // Process 4x4 blocks for remaining rows
  for (; r + 4 <= rank; r += 4) {
    int i = 0;
    for (; i + 4 <= output_dim; i += 4) {
      transpose_4x4_bf16(src + i * rank + r, rank, dst + r * output_dim + i, output_dim);
    }
    for (; i < output_dim; i++) {
      for (int rr = 0; rr < 4; rr++) {
        dst[(r + rr) * output_dim + i] = src[i * rank + (r + rr)];
      }
    }
  }

  // Scalar remainder rows
  for (; r < rank; r++) {
    for (int i = 0; i < output_dim; i++) {
      dst[r * output_dim + i] = src[i * rank + r];
    }
  }
}

/**
 * @brief Fused LoRA add with pre-transposed weight (optimized version)
 *
 * Computes: output[t, i] += scale * sum_r(intermediate[t, r] * weight_t[r, i])
 *
 * Key optimization: weight_t is pre-transposed to [rank][output_dim], allowing
 * contiguous memory access for 16 outputs at a time in the inner loop.
 * This eliminates the horizontal reduction overhead and maximizes cache efficiency.
 *
 * @param intermediate FP32 input [num_tokens, rank]
 * @param weight_t     Pre-transposed BF16 weight [rank][output_dim]
 * @param output       BF16 output [num_tokens, output_dim] (accumulated)
 * @param num_tokens   Number of tokens
 * @param rank         LoRA rank
 * @param output_dim   Output dimension
 * @param scale        LoRA scaling factor
 */
inline void lora_fp32_bf16_fused_add_transposed(const float* __restrict intermediate,
                                                const ggml_bf16_t* __restrict weight_t, ggml_bf16_t* __restrict output,
                                                int num_tokens, int rank, int output_dim, float scale) {
#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_start = sft_timer::get_trace_timestamp();
#endif

  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 16;  // Process 16 outputs at a time (one AVX-512 vector)

  const __m512 scale_vec = _mm512_set1_ps(scale);
  const int output_tail = output_dim & 15;
  const __mmask16 output_tail_mask = output_tail ? ((__mmask16)1 << output_tail) - 1 : 0;

  int t = 0;
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const float* inter0 = intermediate + (t + 0) * rank;
    const float* inter1 = intermediate + (t + 1) * rank;
    const float* inter2 = intermediate + (t + 2) * rank;
    const float* inter3 = intermediate + (t + 3) * rank;
    ggml_bf16_t* out0 = output + (t + 0) * output_dim;
    ggml_bf16_t* out1 = output + (t + 1) * output_dim;
    ggml_bf16_t* out2 = output + (t + 2) * output_dim;
    ggml_bf16_t* out3 = output + (t + 3) * output_dim;

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      // 4 accumulators for 4 tokens, each accumulating 16 outputs
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      // Inner loop over rank - weights are contiguous for each rank position
      for (int r = 0; r < rank; r++) {
        // Load 16 consecutive weights: weight_t[r][i:i+16]
        __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(weight_t + r * output_dim + i))), 16));

        // Broadcast intermediate values and FMA
        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);

        acc0 = _mm512_fmadd_ps(iv0, wv, acc0);
        acc1 = _mm512_fmadd_ps(iv1, wv, acc1);
        acc2 = _mm512_fmadd_ps(iv2, wv, acc2);
        acc3 = _mm512_fmadd_ps(iv3, wv, acc3);
      }

      // Scale
      acc0 = _mm512_mul_ps(acc0, scale_vec);
      acc1 = _mm512_mul_ps(acc1, scale_vec);
      acc2 = _mm512_mul_ps(acc2, scale_vec);
      acc3 = _mm512_mul_ps(acc3, scale_vec);

      // Load output, add, convert, store
      __m512 out_v0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i))), 16));
      __m512 out_v1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i))), 16));
      __m512 out_v2 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i))), 16));
      __m512 out_v3 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i))), 16));

      out_v0 = _mm512_add_ps(out_v0, acc0);
      out_v1 = _mm512_add_ps(out_v1, acc1);
      out_v2 = _mm512_add_ps(out_v2, acc2);
      out_v3 = _mm512_add_ps(out_v3, acc3);

      __m256bh bf16_0 = _mm512_cvtneps_pbh(out_v0);
      __m256bh bf16_1 = _mm512_cvtneps_pbh(out_v1);
      __m256bh bf16_2 = _mm512_cvtneps_pbh(out_v2);
      __m256bh bf16_3 = _mm512_cvtneps_pbh(out_v3);

      _mm256_storeu_si256((__m256i*)(out0 + i), (__m256i)bf16_0);
      _mm256_storeu_si256((__m256i*)(out1 + i), (__m256i)bf16_1);
      _mm256_storeu_si256((__m256i*)(out2 + i), (__m256i)bf16_2);
      _mm256_storeu_si256((__m256i*)(out3 + i), (__m256i)bf16_3);
    }

    // Handle remaining outputs (< 16)
    if (output_tail_mask) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      for (int r = 0; r < rank; r++) {
        __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, weight_t + r * output_dim + i)), 16));

        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);

        acc0 = _mm512_fmadd_ps(iv0, wv, acc0);
        acc1 = _mm512_fmadd_ps(iv1, wv, acc1);
        acc2 = _mm512_fmadd_ps(iv2, wv, acc2);
        acc3 = _mm512_fmadd_ps(iv3, wv, acc3);
      }

      acc0 = _mm512_mul_ps(acc0, scale_vec);
      acc1 = _mm512_mul_ps(acc1, scale_vec);
      acc2 = _mm512_mul_ps(acc2, scale_vec);
      acc3 = _mm512_mul_ps(acc3, scale_vec);

      __m512 out_v0 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, out0 + i)), 16));
      __m512 out_v1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, out1 + i)), 16));
      __m512 out_v2 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, out2 + i)), 16));
      __m512 out_v3 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, out3 + i)), 16));

      out_v0 = _mm512_add_ps(out_v0, acc0);
      out_v1 = _mm512_add_ps(out_v1, acc1);
      out_v2 = _mm512_add_ps(out_v2, acc2);
      out_v3 = _mm512_add_ps(out_v3, acc3);

      _mm256_mask_storeu_epi16(out0 + i, output_tail_mask, (__m256i)_mm512_cvtneps_pbh(out_v0));
      _mm256_mask_storeu_epi16(out1 + i, output_tail_mask, (__m256i)_mm512_cvtneps_pbh(out_v1));
      _mm256_mask_storeu_epi16(out2 + i, output_tail_mask, (__m256i)_mm512_cvtneps_pbh(out_v2));
      _mm256_mask_storeu_epi16(out3 + i, output_tail_mask, (__m256i)_mm512_cvtneps_pbh(out_v3));
    }
  }

  // Remaining tokens (< T_BLOCK)
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      __m512 acc = _mm512_setzero_ps();
      for (int r = 0; r < rank; r++) {
        __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(weight_t + r * output_dim + i))), 16));
        __m512 iv = _mm512_set1_ps(inter_row[r]);
        acc = _mm512_fmadd_ps(iv, wv, acc);
      }
      acc = _mm512_mul_ps(acc, scale_vec);
      __m512 out_v = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out_row + i))), 16));
      out_v = _mm512_add_ps(out_v, acc);
      _mm256_storeu_si256((__m256i*)(out_row + i), (__m256i)_mm512_cvtneps_pbh(out_v));
    }

    if (output_tail_mask) {
      __m512 acc = _mm512_setzero_ps();
      for (int r = 0; r < rank; r++) {
        __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, weight_t + r * output_dim + i)), 16));
        __m512 iv = _mm512_set1_ps(inter_row[r]);
        acc = _mm512_fmadd_ps(iv, wv, acc);
      }
      acc = _mm512_mul_ps(acc, scale_vec);
      __m512 out_v = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(output_tail_mask, out_row + i)), 16));
      out_v = _mm512_add_ps(out_v, acc);
      _mm256_mask_storeu_epi16(out_row + i, output_tail_mask, (__m256i)_mm512_cvtneps_pbh(out_v));
    }
  }

#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_end = sft_timer::get_trace_timestamp();
  char args_buf[128];
  snprintf(args_buf, sizeof(args_buf), "{\"T\":%d,\"R\":%d,\"O\":%d}", num_tokens, rank, output_dim);
  sft_timer::add_kernel_trace("lora_fp32_bf16_fused_add_transposed", trace_start, trace_end, 0,
                              WorkerPool::thread_local_id, args_buf);
#endif
}

/**
 * @brief Optimized matmul for backward: grad @ lora_B_transposed -> result
 *
 * Computes result[t, r] = Σ_h grad[t, h] * lora_B_t[r, h]
 * Using pre-transposed lora_B with layout [rank, hidden] for contiguous access.
 *
 * @param grad Input gradient [num_tokens, hidden] BF16
 * @param lora_b_t Pre-transposed lora_B [rank, hidden] BF16
 * @param result Output [num_tokens, rank] FP32
 * @param num_tokens Number of tokens
 * @param hidden Hidden dimension (input dim)
 * @param rank LoRA rank (output dim)
 */
inline void lora_backward_matmul_transposed(const ggml_bf16_t* __restrict grad, const ggml_bf16_t* __restrict lora_b_t,
                                            float* __restrict result, int num_tokens, int hidden, int rank) {
#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_start = sft_timer::get_trace_timestamp();
#endif

  constexpr int H_BLOCK = 32;

  for (int t = 0; t < num_tokens; t++) {
    const ggml_bf16_t* g_row = grad + t * hidden;

    for (int r = 0; r < rank; r++) {
      const ggml_bf16_t* b_row = lora_b_t + r * hidden;

      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();

      int h = 0;
      for (; h + H_BLOCK <= hidden; h += H_BLOCK) {
        // Load 32 weights (contiguous access from transposed layout)
        __m512 b0, b1;
        avx512_32xbf16_to_32xfp32((__m512i*)(b_row + h), &b0, &b1);

        // Load 32 gradient values
        __m512 g0, g1;
        avx512_32xbf16_to_32xfp32((__m512i*)(g_row + h), &g0, &g1);

        acc0 = _mm512_fmadd_ps(g0, b0, acc0);
        acc1 = _mm512_fmadd_ps(g1, b1, acc1);
      }

      float sum = _mm512_reduce_add_ps(acc0) + _mm512_reduce_add_ps(acc1);

      // Handle remaining elements
      for (; h < hidden; h++) {
        sum += GGML_BF16_TO_FP32(g_row[h]) * GGML_BF16_TO_FP32(b_row[h]);
      }

      result[t * rank + r] = sum;
    }
  }

#if AVX_KERNEL_TRACE_ENABLED
  uint64_t trace_end = sft_timer::get_trace_timestamp();
  char args_buf[128];
  snprintf(args_buf, sizeof(args_buf), "{\"T\":%d,\"H\":%d,\"R\":%d}", num_tokens, hidden, rank);
  sft_timer::add_kernel_trace("lora_backward_matmul_transposed", trace_start, trace_end, 0, WorkerPool::thread_local_id,
                              args_buf);
#endif
}

}  // namespace avx

#endif  // AVX_KERNELS_HPP
