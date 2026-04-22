/**
 * Unit test and benchmark for lora_fp32_bf16_fused_add_wt kernel
 * (Weight layout: [rank, output_dim] - transposed from standard)
 *
 * Computes: output[t, i] += scale * sum_r(intermediate[t, r] * weight[r, i])
 *
 * Build:
 *   g++ -O3 -march=native -mavx512f -mavx512bw -mavx512bf16 \
 *       -I/home/star/hxx/ktransformers/kt-kernel \
 *       -I/home/star/hxx/ktransformers/third_party/llama.cpp \
 *       test_lora_fused_add_wt.cpp -o test_lora_fused_add_wt
 *
 * Run:
 *   ./test_lora_fused_add_wt
 *   ./test_lora_fused_add_wt --profile --impl <name> --rank <n> --tokens <n> --output <n>
 */

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "llama.cpp/ggml-impl.h"

// ============================================================================
// Reference implementation (scalar)
// Weight layout: [rank, output_dim]
// ============================================================================
void lora_fused_add_wt_reference(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                                 ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim,
                                 float scale) {
  for (int t = 0; t < num_tokens; t++) {
    for (int i = 0; i < output_dim; i++) {
      float sum = 0.0f;
      for (int r = 0; r < rank; r++) {
        // weight[r, i] = weight[r * output_dim + i]
        sum += intermediate[t * rank + r] * GGML_BF16_TO_FP32(weight[r * output_dim + i]);
      }
      float out_val = GGML_BF16_TO_FP32(output[t * output_dim + i]);
      out_val += sum * scale;
      output[t * output_dim + i] = GGML_FP32_TO_BF16(out_val);
    }
  }
}

// ============================================================================
// Baseline: Original implementation from sft_moe.hpp backward pass
// Weight layout: [rank, output_dim]
// ============================================================================
inline void avx512_32xbf16_to_32xfp32(const __m512i* src, __m512* dst0, __m512* dst1) {
  __m512i raw = _mm512_loadu_si512(src);
  __m256i lo = _mm512_extracti32x8_epi32(raw, 0);
  __m256i hi = _mm512_extracti32x8_epi32(raw, 1);
  *dst0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(lo), 16));
  *dst1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16));
}

inline void avx512_32xfp32_to_32xbf16(const __m512* src0, const __m512* src1, __m512i* dst) {
  __m256i lo = (__m256i)_mm512_cvtneps_pbh(*src0);
  __m256i hi = (__m256i)_mm512_cvtneps_pbh(*src1);
  __m512i result = _mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1);
  _mm512_storeu_si512(dst, result);
}

void lora_fused_add_wt_baseline(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                                ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  __m512 scale_vec = _mm512_set1_ps(scale);

  for (int t = 0; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    int i = 0;
    for (; i + 32 <= output_dim; i += 32) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();

      for (int r = 0; r < rank; r++) {
        __m512 gtb_vec = _mm512_set1_ps(inter_row[r]);
        const ggml_bf16_t* a_ptr = weight + r * output_dim + i;
        __m512 a0, a1;
        avx512_32xbf16_to_32xfp32((__m512i*)a_ptr, &a0, &a1);
        acc0 = _mm512_fmadd_ps(gtb_vec, a0, acc0);
        acc1 = _mm512_fmadd_ps(gtb_vec, a1, acc1);
      }

      // Load current, add scaled result, store
      __m512 cur0, cur1;
      avx512_32xbf16_to_32xfp32((__m512i*)(out_row + i), &cur0, &cur1);
      cur0 = _mm512_fmadd_ps(acc0, scale_vec, cur0);
      cur1 = _mm512_fmadd_ps(acc1, scale_vec, cur1);
      avx512_32xfp32_to_32xbf16(&cur0, &cur1, (__m512i*)(out_row + i));
    }
    // Scalar remainder
    for (; i < output_dim; i++) {
      float sum = 0.0f;
      for (int r = 0; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(weight[r * output_dim + i]);
      }
      float cur = GGML_BF16_TO_FP32(out_row[i]);
      cur += sum * scale;
      out_row[i] = GGML_FP32_TO_BF16(cur);
    }
  }
}

// ============================================================================
// Optimized v1: T_BLOCK=4, O_BLOCK=32 (contiguous weight access)
// ============================================================================
void lora_fused_add_wt_opt1(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                            ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 32;

  const __m512 scale_vec = _mm512_set1_ps(scale);

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
      // 8 accumulators per token: 4 tokens × 2 (for 32 outputs = 2×16)
      __m512 acc_t0_0 = _mm512_setzero_ps(), acc_t0_1 = _mm512_setzero_ps();
      __m512 acc_t1_0 = _mm512_setzero_ps(), acc_t1_1 = _mm512_setzero_ps();
      __m512 acc_t2_0 = _mm512_setzero_ps(), acc_t2_1 = _mm512_setzero_ps();
      __m512 acc_t3_0 = _mm512_setzero_ps(), acc_t3_1 = _mm512_setzero_ps();

      for (int r = 0; r < rank; r++) {
        // Broadcast intermediate values for each token
        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);

        // Load 32 contiguous weight values: weight[r, i:i+32]
        const ggml_bf16_t* w_ptr = weight + r * output_dim + i;
        __m512i w_i32_0 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr));
        __m512i w_i32_1 = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr + 16)));
        __m512 wv0 = _mm512_castsi512_ps(_mm512_slli_epi32(w_i32_0, 16));
        __m512 wv1 = _mm512_castsi512_ps(_mm512_slli_epi32(w_i32_1, 16));

        // FMA for all 4 tokens
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
}

// ============================================================================
// Optimized v2: Loop unrolling over rank (4 ranks per iteration)
// ============================================================================
void lora_fused_add_wt_opt2(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                            ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
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

      // Main loop: 4 ranks per iteration
      int r = 0;
      for (; r < rank_main; r += R_UNROLL) {
        // Load intermediate values (4 per token, 16 total)
        __m512 iv0_r0 = _mm512_set1_ps(inter0[r + 0]);
        __m512 iv0_r1 = _mm512_set1_ps(inter0[r + 1]);
        __m512 iv0_r2 = _mm512_set1_ps(inter0[r + 2]);
        __m512 iv0_r3 = _mm512_set1_ps(inter0[r + 3]);

        __m512 iv1_r0 = _mm512_set1_ps(inter1[r + 0]);
        __m512 iv1_r1 = _mm512_set1_ps(inter1[r + 1]);
        __m512 iv1_r2 = _mm512_set1_ps(inter1[r + 2]);
        __m512 iv1_r3 = _mm512_set1_ps(inter1[r + 3]);

        __m512 iv2_r0 = _mm512_set1_ps(inter2[r + 0]);
        __m512 iv2_r1 = _mm512_set1_ps(inter2[r + 1]);
        __m512 iv2_r2 = _mm512_set1_ps(inter2[r + 2]);
        __m512 iv2_r3 = _mm512_set1_ps(inter2[r + 3]);

        __m512 iv3_r0 = _mm512_set1_ps(inter3[r + 0]);
        __m512 iv3_r1 = _mm512_set1_ps(inter3[r + 1]);
        __m512 iv3_r2 = _mm512_set1_ps(inter3[r + 2]);
        __m512 iv3_r3 = _mm512_set1_ps(inter3[r + 3]);

        // Load weights for 4 ranks × 32 outputs
        const ggml_bf16_t* w_ptr0 = weight + (r + 0) * output_dim + i;
        const ggml_bf16_t* w_ptr1 = weight + (r + 1) * output_dim + i;
        const ggml_bf16_t* w_ptr2 = weight + (r + 2) * output_dim + i;
        const ggml_bf16_t* w_ptr3 = weight + (r + 3) * output_dim + i;

        __m512 wv0_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr0)), 16));
        __m512 wv0_1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr0 + 16))), 16));
        __m512 wv1_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr1)), 16));
        __m512 wv1_1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr1 + 16))), 16));
        __m512 wv2_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr2)), 16));
        __m512 wv2_1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr2 + 16))), 16));
        __m512 wv3_0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr3)), 16));
        __m512 wv3_1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr3 + 16))), 16));

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

      // Apply scale and store
      acc_t0_0 = _mm512_mul_ps(acc_t0_0, scale_vec);
      acc_t0_1 = _mm512_mul_ps(acc_t0_1, scale_vec);
      acc_t1_0 = _mm512_mul_ps(acc_t1_0, scale_vec);
      acc_t1_1 = _mm512_mul_ps(acc_t1_1, scale_vec);
      acc_t2_0 = _mm512_mul_ps(acc_t2_0, scale_vec);
      acc_t2_1 = _mm512_mul_ps(acc_t2_1, scale_vec);
      acc_t3_0 = _mm512_mul_ps(acc_t3_0, scale_vec);
      acc_t3_1 = _mm512_mul_ps(acc_t3_1, scale_vec);

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

    // Remainder outputs
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

  // Handle remaining tokens (< T_BLOCK) - use baseline
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    int i = 0;
    for (; i + 32 <= output_dim; i += 32) {
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();

      for (int r = 0; r < rank; r++) {
        __m512 iv = _mm512_set1_ps(inter_row[r]);
        const ggml_bf16_t* w_ptr = weight + r * output_dim + i;
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w_ptr)), 16));
        __m512 wv1 = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_ptr + 16))), 16));
        acc0 = _mm512_fmadd_ps(iv, wv0, acc0);
        acc1 = _mm512_fmadd_ps(iv, wv1, acc1);
      }

      acc0 = _mm512_mul_ps(acc0, scale_vec);
      acc1 = _mm512_mul_ps(acc1, scale_vec);

      __m512 cur0 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out_row + i))), 16));
      __m512 cur1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out_row + i + 16))), 16));
      cur0 = _mm512_add_ps(cur0, acc0);
      cur1 = _mm512_add_ps(cur1, acc1);
      _mm256_storeu_si256((__m256i*)(out_row + i), (__m256i)_mm512_cvtneps_pbh(cur0));
      _mm256_storeu_si256((__m256i*)(out_row + i + 16), (__m256i)_mm512_cvtneps_pbh(cur1));
    }

    for (; i < output_dim; i++) {
      float sum = 0.0f;
      for (int r = 0; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(weight[r * output_dim + i]);
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v3: Specialized for rank=8 with preloaded intermediate
// Pre-broadcast all intermediate values before inner loop
// ============================================================================
void lora_fused_add_wt_opt3_r8(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                               ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  if (rank != 8) {
    // Fallback to opt2 for other ranks
    lora_fused_add_wt_opt2(intermediate, weight, output, num_tokens, rank, output_dim, scale);
    return;
  }

  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 32;

  const __m512 scale_vec = _mm512_set1_ps(scale);

  int t = 0;
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const float* inter0 = intermediate + (t + 0) * 8;
    const float* inter1 = intermediate + (t + 1) * 8;
    const float* inter2 = intermediate + (t + 2) * 8;
    const float* inter3 = intermediate + (t + 3) * 8;
    ggml_bf16_t* out0 = output + (t + 0) * output_dim;
    ggml_bf16_t* out1 = output + (t + 1) * output_dim;
    ggml_bf16_t* out2 = output + (t + 2) * output_dim;
    ggml_bf16_t* out3 = output + (t + 3) * output_dim;

    // Pre-broadcast all intermediate values (8 ranks × 4 tokens = 32 vectors)
    __m512 iv0_r0 = _mm512_set1_ps(inter0[0]), iv0_r1 = _mm512_set1_ps(inter0[1]);
    __m512 iv0_r2 = _mm512_set1_ps(inter0[2]), iv0_r3 = _mm512_set1_ps(inter0[3]);
    __m512 iv0_r4 = _mm512_set1_ps(inter0[4]), iv0_r5 = _mm512_set1_ps(inter0[5]);
    __m512 iv0_r6 = _mm512_set1_ps(inter0[6]), iv0_r7 = _mm512_set1_ps(inter0[7]);

    __m512 iv1_r0 = _mm512_set1_ps(inter1[0]), iv1_r1 = _mm512_set1_ps(inter1[1]);
    __m512 iv1_r2 = _mm512_set1_ps(inter1[2]), iv1_r3 = _mm512_set1_ps(inter1[3]);
    __m512 iv1_r4 = _mm512_set1_ps(inter1[4]), iv1_r5 = _mm512_set1_ps(inter1[5]);
    __m512 iv1_r6 = _mm512_set1_ps(inter1[6]), iv1_r7 = _mm512_set1_ps(inter1[7]);

    __m512 iv2_r0 = _mm512_set1_ps(inter2[0]), iv2_r1 = _mm512_set1_ps(inter2[1]);
    __m512 iv2_r2 = _mm512_set1_ps(inter2[2]), iv2_r3 = _mm512_set1_ps(inter2[3]);
    __m512 iv2_r4 = _mm512_set1_ps(inter2[4]), iv2_r5 = _mm512_set1_ps(inter2[5]);
    __m512 iv2_r6 = _mm512_set1_ps(inter2[6]), iv2_r7 = _mm512_set1_ps(inter2[7]);

    __m512 iv3_r0 = _mm512_set1_ps(inter3[0]), iv3_r1 = _mm512_set1_ps(inter3[1]);
    __m512 iv3_r2 = _mm512_set1_ps(inter3[2]), iv3_r3 = _mm512_set1_ps(inter3[3]);
    __m512 iv3_r4 = _mm512_set1_ps(inter3[4]), iv3_r5 = _mm512_set1_ps(inter3[5]);
    __m512 iv3_r6 = _mm512_set1_ps(inter3[6]), iv3_r7 = _mm512_set1_ps(inter3[7]);

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      // Weight pointers for 8 ranks
      const ggml_bf16_t* w0 = weight + 0 * output_dim + i;
      const ggml_bf16_t* w1 = weight + 1 * output_dim + i;
      const ggml_bf16_t* w2 = weight + 2 * output_dim + i;
      const ggml_bf16_t* w3 = weight + 3 * output_dim + i;
      const ggml_bf16_t* w4 = weight + 4 * output_dim + i;
      const ggml_bf16_t* w5 = weight + 5 * output_dim + i;
      const ggml_bf16_t* w6 = weight + 6 * output_dim + i;
      const ggml_bf16_t* w7 = weight + 7 * output_dim + i;

      // Load all 8 weight rows × 2 (for 32 outputs)
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
      __m512 wv4_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w4)), 16));
      __m512 wv4_1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w4 + 16))), 16));
      __m512 wv5_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w5)), 16));
      __m512 wv5_1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w5 + 16))), 16));
      __m512 wv6_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w6)), 16));
      __m512 wv6_1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w6 + 16))), 16));
      __m512 wv7_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)w7)), 16));
      __m512 wv7_1 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w7 + 16))), 16));

      // Token 0: fully unrolled accumulation
      __m512 acc_t0_0 = _mm512_mul_ps(iv0_r0, wv0_0);
      __m512 acc_t0_1 = _mm512_mul_ps(iv0_r0, wv0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r1, wv1_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r1, wv1_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r2, wv2_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r2, wv2_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r3, wv3_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r3, wv3_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r4, wv4_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r4, wv4_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r5, wv5_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r5, wv5_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r6, wv6_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r6, wv6_1, acc_t0_1);
      acc_t0_0 = _mm512_fmadd_ps(iv0_r7, wv7_0, acc_t0_0);
      acc_t0_1 = _mm512_fmadd_ps(iv0_r7, wv7_1, acc_t0_1);

      // Token 1
      __m512 acc_t1_0 = _mm512_mul_ps(iv1_r0, wv0_0);
      __m512 acc_t1_1 = _mm512_mul_ps(iv1_r0, wv0_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r1, wv1_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r1, wv1_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r2, wv2_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r2, wv2_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r3, wv3_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r3, wv3_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r4, wv4_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r4, wv4_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r5, wv5_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r5, wv5_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r6, wv6_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r6, wv6_1, acc_t1_1);
      acc_t1_0 = _mm512_fmadd_ps(iv1_r7, wv7_0, acc_t1_0);
      acc_t1_1 = _mm512_fmadd_ps(iv1_r7, wv7_1, acc_t1_1);

      // Token 2
      __m512 acc_t2_0 = _mm512_mul_ps(iv2_r0, wv0_0);
      __m512 acc_t2_1 = _mm512_mul_ps(iv2_r0, wv0_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r1, wv1_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r1, wv1_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r2, wv2_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r2, wv2_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r3, wv3_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r3, wv3_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r4, wv4_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r4, wv4_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r5, wv5_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r5, wv5_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r6, wv6_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r6, wv6_1, acc_t2_1);
      acc_t2_0 = _mm512_fmadd_ps(iv2_r7, wv7_0, acc_t2_0);
      acc_t2_1 = _mm512_fmadd_ps(iv2_r7, wv7_1, acc_t2_1);

      // Token 3
      __m512 acc_t3_0 = _mm512_mul_ps(iv3_r0, wv0_0);
      __m512 acc_t3_1 = _mm512_mul_ps(iv3_r0, wv0_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r1, wv1_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r1, wv1_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r2, wv2_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r2, wv2_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r3, wv3_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r3, wv3_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r4, wv4_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r4, wv4_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r5, wv5_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r5, wv5_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r6, wv6_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r6, wv6_1, acc_t3_1);
      acc_t3_0 = _mm512_fmadd_ps(iv3_r7, wv7_0, acc_t3_0);
      acc_t3_1 = _mm512_fmadd_ps(iv3_r7, wv7_1, acc_t3_1);

      // Apply scale and store
      acc_t0_0 = _mm512_mul_ps(acc_t0_0, scale_vec);
      acc_t0_1 = _mm512_mul_ps(acc_t0_1, scale_vec);
      acc_t1_0 = _mm512_mul_ps(acc_t1_0, scale_vec);
      acc_t1_1 = _mm512_mul_ps(acc_t1_1, scale_vec);
      acc_t2_0 = _mm512_mul_ps(acc_t2_0, scale_vec);
      acc_t2_1 = _mm512_mul_ps(acc_t2_1, scale_vec);
      acc_t3_0 = _mm512_mul_ps(acc_t3_0, scale_vec);
      acc_t3_1 = _mm512_mul_ps(acc_t3_1, scale_vec);

      // Load, add, store for each token
      __m512 cur0_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i))), 16));
      __m512 cur0_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i + 16))), 16));
      cur0_0 = _mm512_add_ps(cur0_0, acc_t0_0);
      cur0_1 = _mm512_add_ps(cur0_1, acc_t0_1);
      _mm256_storeu_si256((__m256i*)(out0 + i), (__m256i)_mm512_cvtneps_pbh(cur0_0));
      _mm256_storeu_si256((__m256i*)(out0 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur0_1));

      __m512 cur1_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i))), 16));
      __m512 cur1_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i + 16))), 16));
      cur1_0 = _mm512_add_ps(cur1_0, acc_t1_0);
      cur1_1 = _mm512_add_ps(cur1_1, acc_t1_1);
      _mm256_storeu_si256((__m256i*)(out1 + i), (__m256i)_mm512_cvtneps_pbh(cur1_0));
      _mm256_storeu_si256((__m256i*)(out1 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur1_1));

      __m512 cur2_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i))), 16));
      __m512 cur2_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out2 + i + 16))), 16));
      cur2_0 = _mm512_add_ps(cur2_0, acc_t2_0);
      cur2_1 = _mm512_add_ps(cur2_1, acc_t2_1);
      _mm256_storeu_si256((__m256i*)(out2 + i), (__m256i)_mm512_cvtneps_pbh(cur2_0));
      _mm256_storeu_si256((__m256i*)(out2 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur2_1));

      __m512 cur3_0 =
          _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i))), 16));
      __m512 cur3_1 = _mm512_castsi512_ps(
          _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out3 + i + 16))), 16));
      cur3_0 = _mm512_add_ps(cur3_0, acc_t3_0);
      cur3_1 = _mm512_add_ps(cur3_1, acc_t3_1);
      _mm256_storeu_si256((__m256i*)(out3 + i), (__m256i)_mm512_cvtneps_pbh(cur3_0));
      _mm256_storeu_si256((__m256i*)(out3 + i + 16), (__m256i)_mm512_cvtneps_pbh(cur3_1));
    }

    // Handle remaining outputs
    for (; i < output_dim; i++) {
      float sum0 = inter0[0] * GGML_BF16_TO_FP32(weight[0 * output_dim + i]) +
                   inter0[1] * GGML_BF16_TO_FP32(weight[1 * output_dim + i]) +
                   inter0[2] * GGML_BF16_TO_FP32(weight[2 * output_dim + i]) +
                   inter0[3] * GGML_BF16_TO_FP32(weight[3 * output_dim + i]) +
                   inter0[4] * GGML_BF16_TO_FP32(weight[4 * output_dim + i]) +
                   inter0[5] * GGML_BF16_TO_FP32(weight[5 * output_dim + i]) +
                   inter0[6] * GGML_BF16_TO_FP32(weight[6 * output_dim + i]) +
                   inter0[7] * GGML_BF16_TO_FP32(weight[7 * output_dim + i]);
      float sum1 = inter1[0] * GGML_BF16_TO_FP32(weight[0 * output_dim + i]) +
                   inter1[1] * GGML_BF16_TO_FP32(weight[1 * output_dim + i]) +
                   inter1[2] * GGML_BF16_TO_FP32(weight[2 * output_dim + i]) +
                   inter1[3] * GGML_BF16_TO_FP32(weight[3 * output_dim + i]) +
                   inter1[4] * GGML_BF16_TO_FP32(weight[4 * output_dim + i]) +
                   inter1[5] * GGML_BF16_TO_FP32(weight[5 * output_dim + i]) +
                   inter1[6] * GGML_BF16_TO_FP32(weight[6 * output_dim + i]) +
                   inter1[7] * GGML_BF16_TO_FP32(weight[7 * output_dim + i]);
      float sum2 = inter2[0] * GGML_BF16_TO_FP32(weight[0 * output_dim + i]) +
                   inter2[1] * GGML_BF16_TO_FP32(weight[1 * output_dim + i]) +
                   inter2[2] * GGML_BF16_TO_FP32(weight[2 * output_dim + i]) +
                   inter2[3] * GGML_BF16_TO_FP32(weight[3 * output_dim + i]) +
                   inter2[4] * GGML_BF16_TO_FP32(weight[4 * output_dim + i]) +
                   inter2[5] * GGML_BF16_TO_FP32(weight[5 * output_dim + i]) +
                   inter2[6] * GGML_BF16_TO_FP32(weight[6 * output_dim + i]) +
                   inter2[7] * GGML_BF16_TO_FP32(weight[7 * output_dim + i]);
      float sum3 = inter3[0] * GGML_BF16_TO_FP32(weight[0 * output_dim + i]) +
                   inter3[1] * GGML_BF16_TO_FP32(weight[1 * output_dim + i]) +
                   inter3[2] * GGML_BF16_TO_FP32(weight[2 * output_dim + i]) +
                   inter3[3] * GGML_BF16_TO_FP32(weight[3 * output_dim + i]) +
                   inter3[4] * GGML_BF16_TO_FP32(weight[4 * output_dim + i]) +
                   inter3[5] * GGML_BF16_TO_FP32(weight[5 * output_dim + i]) +
                   inter3[6] * GGML_BF16_TO_FP32(weight[6 * output_dim + i]) +
                   inter3[7] * GGML_BF16_TO_FP32(weight[7 * output_dim + i]);
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + sum0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + sum1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + sum2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + sum3 * scale);
    }
  }

  // Handle remaining tokens - fallback to opt2
  if (t < num_tokens) {
    lora_fused_add_wt_opt2(intermediate + t * 8, weight, output + t * output_dim, num_tokens - t, 8, output_dim, scale);
  }
}

// ============================================================================
// Test infrastructure
// ============================================================================
void fill_random_fp32(float* data, size_t count, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.1f);
  for (size_t i = 0; i < count; i++) {
    data[i] = dist(rng);
  }
}

void fill_random_bf16(ggml_bf16_t* data, size_t count, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.1f);
  for (size_t i = 0; i < count; i++) {
    data[i] = GGML_FP32_TO_BF16(dist(rng));
  }
}

float max_abs_diff(const ggml_bf16_t* a, const ggml_bf16_t* b, size_t count) {
  float max_diff = 0.0f;
  for (size_t i = 0; i < count; i++) {
    float diff = std::abs(GGML_BF16_TO_FP32(a[i]) - GGML_BF16_TO_FP32(b[i]));
    max_diff = std::max(max_diff, diff);
  }
  return max_diff;
}

using KernelFn = void (*)(const float*, const ggml_bf16_t*, ggml_bf16_t*, int, int, int, float);

struct ImplInfo {
  const char* name;
  KernelFn fn;
};

ImplInfo impls[] = {
    {"reference", lora_fused_add_wt_reference}, {"baseline", lora_fused_add_wt_baseline},
    {"opt1", lora_fused_add_wt_opt1},           {"opt2", lora_fused_add_wt_opt2},
    {"opt3_r8", lora_fused_add_wt_opt3_r8},
};

void run_correctness_test(int num_tokens, int rank, int output_dim) {
  printf("\n=== Correctness Test: T=%d, R=%d, O=%d ===\n", num_tokens, rank, output_dim);

  float scale = 0.5f;

  // Allocate buffers (ensure enough size for alignment)
  size_t inter_size = (size_t)num_tokens * rank;
  size_t weight_size = (size_t)rank * output_dim;
  size_t out_size = (size_t)num_tokens * output_dim;

  // Add padding for vector loads
  size_t inter_alloc = ((inter_size + 31) / 32) * 32;
  size_t weight_alloc = ((weight_size + 31) / 32) * 32;
  size_t out_alloc = ((out_size + 31) / 32) * 32;

  float* intermediate = (float*)aligned_alloc(64, inter_alloc * sizeof(float));
  ggml_bf16_t* weight = (ggml_bf16_t*)aligned_alloc(64, weight_alloc * sizeof(ggml_bf16_t));
  ggml_bf16_t* output_ref = (ggml_bf16_t*)aligned_alloc(64, out_alloc * sizeof(ggml_bf16_t));
  ggml_bf16_t* output_test = (ggml_bf16_t*)aligned_alloc(64, out_alloc * sizeof(ggml_bf16_t));
  ggml_bf16_t* output_init = (ggml_bf16_t*)aligned_alloc(64, out_alloc * sizeof(ggml_bf16_t));

  // Zero padding areas
  memset(intermediate, 0, inter_alloc * sizeof(float));
  memset(weight, 0, weight_alloc * sizeof(ggml_bf16_t));
  memset(output_init, 0, out_alloc * sizeof(ggml_bf16_t));

  std::mt19937 rng(42);
  fill_random_fp32(intermediate, inter_size, rng);
  fill_random_bf16(weight, weight_size, rng);
  fill_random_bf16(output_init, out_size, rng);

  // Run reference
  memcpy(output_ref, output_init, out_alloc * sizeof(ggml_bf16_t));
  lora_fused_add_wt_reference(intermediate, weight, output_ref, num_tokens, rank, output_dim, scale);

  // Test each implementation
  for (int impl_idx = 1; impl_idx < (int)(sizeof(impls) / sizeof(impls[0])); impl_idx++) {
    memcpy(output_test, output_init, out_alloc * sizeof(ggml_bf16_t));
    impls[impl_idx].fn(intermediate, weight, output_test, num_tokens, rank, output_dim, scale);

    float max_diff = max_abs_diff(output_ref, output_test, out_size);
    // BF16 has ~3 decimal digits precision, allow larger error for larger accumulations
    float threshold = 1e-3f * (1 + rank / 8.0f);
    bool pass = max_diff < threshold;
    printf("  %12s: max_diff=%.6e (thresh=%.1e) %s\n", impls[impl_idx].name, max_diff, threshold,
           pass ? "PASS" : "FAIL");
  }

  free(intermediate);
  free(weight);
  free(output_ref);
  free(output_test);
  free(output_init);
}

void run_benchmark(int num_tokens, int rank, int output_dim, int warmup, int iters, const char* impl_name = nullptr) {
  printf("\n=== Benchmark: T=%d, R=%d, O=%d ===\n", num_tokens, rank, output_dim);

  std::mt19937 rng(42);
  float scale = 0.5f;

  size_t inter_size = (size_t)num_tokens * rank;
  size_t weight_size = (size_t)rank * output_dim;
  size_t out_size = (size_t)num_tokens * output_dim;

  // Add padding for vector loads
  size_t inter_alloc = ((inter_size + 31) / 32) * 32;
  size_t weight_alloc = ((weight_size + 31) / 32) * 32;
  size_t out_alloc = ((out_size + 31) / 32) * 32;

  float* intermediate = (float*)aligned_alloc(64, inter_alloc * sizeof(float));
  ggml_bf16_t* weight = (ggml_bf16_t*)aligned_alloc(64, weight_alloc * sizeof(ggml_bf16_t));
  ggml_bf16_t* output = (ggml_bf16_t*)aligned_alloc(64, out_alloc * sizeof(ggml_bf16_t));

  memset(intermediate, 0, inter_alloc * sizeof(float));
  memset(weight, 0, weight_alloc * sizeof(ggml_bf16_t));
  memset(output, 0, out_alloc * sizeof(ggml_bf16_t));

  fill_random_fp32(intermediate, inter_size, rng);
  fill_random_bf16(weight, weight_size, rng);
  fill_random_bf16(output, out_size, rng);

  // FLOPs: 2 * num_tokens * output_dim * rank (multiply-add)
  double flops = 2.0 * num_tokens * output_dim * rank;

  for (int impl_idx = 0; impl_idx < (int)(sizeof(impls) / sizeof(impls[0])); impl_idx++) {
    if (impl_name && strcmp(impls[impl_idx].name, impl_name) != 0) continue;

    // Warmup
    for (int i = 0; i < warmup; i++) {
      impls[impl_idx].fn(intermediate, weight, output, num_tokens, rank, output_dim, scale);
    }

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
      impls[impl_idx].fn(intermediate, weight, output, num_tokens, rank, output_dim, scale);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(end - start).count();
    double gflops = (flops * iters) / elapsed_s / 1e9;

    printf("  %12s: %.3f ms/iter, %.2f GFLOPS\n", impls[impl_idx].name, elapsed_s * 1000.0 / iters, gflops);
  }

  free(intermediate);
  free(weight);
  free(output);
}

void run_profile_mode(int num_tokens, int rank, int output_dim, const char* impl_name) {
  printf("Profile mode: T=%d, R=%d, O=%d, impl=%s\n", num_tokens, rank, output_dim, impl_name);
  printf("Running infinite loop for profiling (Ctrl+C to stop)...\n");

  std::mt19937 rng(42);
  float scale = 0.5f;

  size_t inter_size = (size_t)num_tokens * rank;
  size_t weight_size = (size_t)rank * output_dim;
  size_t out_size = (size_t)num_tokens * output_dim;

  // Add padding for vector loads
  size_t inter_alloc = ((inter_size + 31) / 32) * 32;
  size_t weight_alloc = ((weight_size + 31) / 32) * 32;
  size_t out_alloc = ((out_size + 31) / 32) * 32;

  float* intermediate = (float*)aligned_alloc(64, inter_alloc * sizeof(float));
  ggml_bf16_t* weight = (ggml_bf16_t*)aligned_alloc(64, weight_alloc * sizeof(ggml_bf16_t));
  ggml_bf16_t* output = (ggml_bf16_t*)aligned_alloc(64, out_alloc * sizeof(ggml_bf16_t));

  memset(intermediate, 0, inter_alloc * sizeof(float));
  memset(weight, 0, weight_alloc * sizeof(ggml_bf16_t));
  memset(output, 0, out_alloc * sizeof(ggml_bf16_t));

  fill_random_fp32(intermediate, inter_size, rng);
  fill_random_bf16(weight, weight_size, rng);
  fill_random_bf16(output, out_size, rng);

  KernelFn fn = nullptr;
  for (auto& impl : impls) {
    if (strcmp(impl.name, impl_name) == 0) {
      fn = impl.fn;
      break;
    }
  }

  if (!fn) {
    printf("Unknown implementation: %s\n", impl_name);
    printf("Available: ");
    for (auto& impl : impls) printf("%s ", impl.name);
    printf("\n");
    exit(1);
  }

  while (true) {
    fn(intermediate, weight, output, num_tokens, rank, output_dim, scale);
  }
}

int main(int argc, char** argv) {
  bool profile_mode = false;
  const char* impl_name = nullptr;
  int tokens = 128;
  int rank = 8;
  int output_dim = 14336;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--profile") == 0) {
      profile_mode = true;
    } else if (strcmp(argv[i], "--impl") == 0 && i + 1 < argc) {
      impl_name = argv[++i];
    } else if (strcmp(argv[i], "--rank") == 0 && i + 1 < argc) {
      rank = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
      tokens = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output_dim = atoi(argv[++i]);
    }
  }

  if (profile_mode) {
    if (!impl_name) impl_name = "opt2";
    run_profile_mode(tokens, rank, output_dim, impl_name);
    return 0;
  }

  printf("lora_fp32_bf16_fused_add_wt Benchmark\n");
  printf("Weight layout: [rank, output_dim] (transposed)\n");
  printf("=====================================\n");

  // Correctness tests
  run_correctness_test(4, 8, 64);
  run_correctness_test(17, 8, 100);
  run_correctness_test(128, 8, 14336);
  run_correctness_test(128, 64, 14336);

  // Benchmarks - typical backward pass dimensions
  printf("\n\n===== Performance Benchmarks =====\n");

  // Different ranks
  for (int r : {8, 16, 32, 64}) {
    run_benchmark(128, r, 14336, 10, 100);
  }

  // Different token counts
  for (int t : {32, 64, 128, 256}) {
    run_benchmark(t, 8, 14336, 10, 100);
  }

  return 0;
}
