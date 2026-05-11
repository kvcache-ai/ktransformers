/**
 * Unit test and benchmark for lora_fp32_bf16_fused_add kernel
 *
 * Computes: output[t, i] += scale * sum_r(intermediate[t, r] * weight[i, r])
 *
 * Build:
 *   g++ -O3 -march=native -mavx512f -mavx512bw -mavx512bf16 \
 *       -I/home/star/hxx/ktransformers/kt-kernel \
 *       -I/home/star/hxx/ktransformers/third_party/llama.cpp \
 *       test_lora_fused_add.cpp -o test_lora_fused_add
 *
 * Run:
 *   ./test_lora_fused_add
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
// ============================================================================
void lora_fused_add_reference(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                              ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  for (int t = 0; t < num_tokens; t++) {
    for (int i = 0; i < output_dim; i++) {
      float sum = 0.0f;
      for (int r = 0; r < rank; r++) {
        sum += intermediate[t * rank + r] * GGML_BF16_TO_FP32(weight[i * rank + r]);
      }
      float out_val = GGML_BF16_TO_FP32(output[t * output_dim + i]);
      out_val += sum * scale;
      output[t * output_dim + i] = GGML_FP32_TO_BF16(out_val);
    }
  }
}

// ============================================================================
// Current implementation (from avx_kernels.hpp)
// ============================================================================
void lora_fused_add_current(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                            ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  for (int t = 0; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;

    // Vectorize over output dimension with unrolling
    int i = 0;
    for (; i + 4 <= output_dim; i += 4) {
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;

      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 inter_vec = _mm512_loadu_ps(inter_row + r);

        // Convert BF16 weights to FP32
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r))), 16));
        __m512 wv1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r))), 16));
        __m512 wv2 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r))), 16));
        __m512 wv3 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r))), 16));

        acc0 = _mm512_fmadd_ps(inter_vec, wv0, acc0);
        acc1 = _mm512_fmadd_ps(inter_vec, wv1, acc1);
        acc2 = _mm512_fmadd_ps(inter_vec, wv2, acc2);
        acc3 = _mm512_fmadd_ps(inter_vec, wv3, acc3);
      }

      float sum0 = _mm512_reduce_add_ps(acc0);
      float sum1 = _mm512_reduce_add_ps(acc1);
      float sum2 = _mm512_reduce_add_ps(acc2);
      float sum3 = _mm512_reduce_add_ps(acc3);

      // Scalar tail for rank
      for (; r < rank; r++) {
        float inter_val = inter_row[r];
        sum0 += inter_val * GGML_BF16_TO_FP32(w0[r]);
        sum1 += inter_val * GGML_BF16_TO_FP32(w1[r]);
        sum2 += inter_val * GGML_BF16_TO_FP32(w2[r]);
        sum3 += inter_val * GGML_BF16_TO_FP32(w3[r]);
      }

      // Scale and add to output
      float out_val0 = GGML_BF16_TO_FP32(out_row[i + 0]) + sum0 * scale;
      float out_val1 = GGML_BF16_TO_FP32(out_row[i + 1]) + sum1 * scale;
      float out_val2 = GGML_BF16_TO_FP32(out_row[i + 2]) + sum2 * scale;
      float out_val3 = GGML_BF16_TO_FP32(out_row[i + 3]) + sum3 * scale;
      out_row[i + 0] = GGML_FP32_TO_BF16(out_val0);
      out_row[i + 1] = GGML_FP32_TO_BF16(out_val1);
      out_row[i + 2] = GGML_FP32_TO_BF16(out_val2);
      out_row[i + 3] = GGML_FP32_TO_BF16(out_val3);
    }

    // Remainder output dimensions
    for (; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc = _mm512_setzero_ps();
      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 inter_vec = _mm512_loadu_ps(inter_row + r);
        __m512 w_vec = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc = _mm512_fmadd_ps(inter_vec, w_vec, acc);
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(w_row[r]);
      }
      float out_val = GGML_BF16_TO_FP32(out_row[i]) + sum * scale;
      out_row[i] = GGML_FP32_TO_BF16(out_val);
    }
  }
}

// ============================================================================
// Optimized v1: Token-blocking (T_BLOCK=4) + Output-blocking (O_BLOCK=4)
// Reuse weight loads across multiple tokens
// ============================================================================
void lora_fused_add_opt1(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 4;

  int t = 0;
  // Process T_BLOCK tokens at a time
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
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;

      // 16 accumulators: 4 tokens × 4 outputs
      __m512 acc_t0_o0 = _mm512_setzero_ps(), acc_t0_o1 = _mm512_setzero_ps();
      __m512 acc_t0_o2 = _mm512_setzero_ps(), acc_t0_o3 = _mm512_setzero_ps();
      __m512 acc_t1_o0 = _mm512_setzero_ps(), acc_t1_o1 = _mm512_setzero_ps();
      __m512 acc_t1_o2 = _mm512_setzero_ps(), acc_t1_o3 = _mm512_setzero_ps();
      __m512 acc_t2_o0 = _mm512_setzero_ps(), acc_t2_o1 = _mm512_setzero_ps();
      __m512 acc_t2_o2 = _mm512_setzero_ps(), acc_t2_o3 = _mm512_setzero_ps();
      __m512 acc_t3_o0 = _mm512_setzero_ps(), acc_t3_o1 = _mm512_setzero_ps();
      __m512 acc_t3_o2 = _mm512_setzero_ps(), acc_t3_o3 = _mm512_setzero_ps();

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        // Load weights once, reuse for 4 tokens
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r))), 16));
        __m512 wv1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r))), 16));
        __m512 wv2 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r))), 16));
        __m512 wv3 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r))), 16));

        // Token 0
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);

        // Token 1
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);

        // Token 2
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);

        // Token 3
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);
        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
      }

      // Reduce accumulators
      float s_t0_o0 = _mm512_reduce_add_ps(acc_t0_o0);
      float s_t0_o1 = _mm512_reduce_add_ps(acc_t0_o1);
      float s_t0_o2 = _mm512_reduce_add_ps(acc_t0_o2);
      float s_t0_o3 = _mm512_reduce_add_ps(acc_t0_o3);
      float s_t1_o0 = _mm512_reduce_add_ps(acc_t1_o0);
      float s_t1_o1 = _mm512_reduce_add_ps(acc_t1_o1);
      float s_t1_o2 = _mm512_reduce_add_ps(acc_t1_o2);
      float s_t1_o3 = _mm512_reduce_add_ps(acc_t1_o3);
      float s_t2_o0 = _mm512_reduce_add_ps(acc_t2_o0);
      float s_t2_o1 = _mm512_reduce_add_ps(acc_t2_o1);
      float s_t2_o2 = _mm512_reduce_add_ps(acc_t2_o2);
      float s_t2_o3 = _mm512_reduce_add_ps(acc_t2_o3);
      float s_t3_o0 = _mm512_reduce_add_ps(acc_t3_o0);
      float s_t3_o1 = _mm512_reduce_add_ps(acc_t3_o1);
      float s_t3_o2 = _mm512_reduce_add_ps(acc_t3_o2);
      float s_t3_o3 = _mm512_reduce_add_ps(acc_t3_o3);

      // Scalar tail for rank
      for (; r < rank; r++) {
        float w0v = GGML_BF16_TO_FP32(w0[r]);
        float w1v = GGML_BF16_TO_FP32(w1[r]);
        float w2v = GGML_BF16_TO_FP32(w2[r]);
        float w3v = GGML_BF16_TO_FP32(w3[r]);
        s_t0_o0 += inter0[r] * w0v;
        s_t0_o1 += inter0[r] * w1v;
        s_t0_o2 += inter0[r] * w2v;
        s_t0_o3 += inter0[r] * w3v;
        s_t1_o0 += inter1[r] * w0v;
        s_t1_o1 += inter1[r] * w1v;
        s_t1_o2 += inter1[r] * w2v;
        s_t1_o3 += inter1[r] * w3v;
        s_t2_o0 += inter2[r] * w0v;
        s_t2_o1 += inter2[r] * w1v;
        s_t2_o2 += inter2[r] * w2v;
        s_t2_o3 += inter2[r] * w3v;
        s_t3_o0 += inter3[r] * w0v;
        s_t3_o1 += inter3[r] * w1v;
        s_t3_o2 += inter3[r] * w2v;
        s_t3_o3 += inter3[r] * w3v;
      }

      // Scale and add to output
      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
    }

    // Remainder outputs
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
      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);
      for (; r < rank; r++) {
        float wv = GGML_BF16_TO_FP32(w_row[r]);
        s0 += inter0[r] * wv;
        s1 += inter1[r] * wv;
        s2 += inter2[r] * wv;
        s3 += inter3[r] * wv;
      }
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
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
      float sum = _mm512_reduce_add_ps(acc);
      for (; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(w_row[r]);
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v2: Convert FP32 intermediate to BF16 and use dpbf16_ps
// This allows native BF16 dot product for better throughput
// ============================================================================
void lora_fused_add_opt2(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 4;

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
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;

      // 16 accumulators
      __m512 acc_t0_o0 = _mm512_setzero_ps(), acc_t0_o1 = _mm512_setzero_ps();
      __m512 acc_t0_o2 = _mm512_setzero_ps(), acc_t0_o3 = _mm512_setzero_ps();
      __m512 acc_t1_o0 = _mm512_setzero_ps(), acc_t1_o1 = _mm512_setzero_ps();
      __m512 acc_t1_o2 = _mm512_setzero_ps(), acc_t1_o3 = _mm512_setzero_ps();
      __m512 acc_t2_o0 = _mm512_setzero_ps(), acc_t2_o1 = _mm512_setzero_ps();
      __m512 acc_t2_o2 = _mm512_setzero_ps(), acc_t2_o3 = _mm512_setzero_ps();
      __m512 acc_t3_o0 = _mm512_setzero_ps(), acc_t3_o1 = _mm512_setzero_ps();
      __m512 acc_t3_o2 = _mm512_setzero_ps(), acc_t3_o3 = _mm512_setzero_ps();

      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        // Load BF16 weights (32 elements = 64 bytes)
        __m512bh wv0 = (__m512bh)_mm512_loadu_si512((__m512i*)(w0 + r));
        __m512bh wv1 = (__m512bh)_mm512_loadu_si512((__m512i*)(w1 + r));
        __m512bh wv2 = (__m512bh)_mm512_loadu_si512((__m512i*)(w2 + r));
        __m512bh wv3 = (__m512bh)_mm512_loadu_si512((__m512i*)(w3 + r));

        // Convert FP32 intermediate to BF16 (32 elements)
        // Load 32 FP32 values (2 x 16), convert to BF16
        __m512 fp32_lo0 = _mm512_loadu_ps(inter0 + r);
        __m512 fp32_hi0 = _mm512_loadu_ps(inter0 + r + 16);
        __m512bh iv0 = _mm512_cvtne2ps_pbh(fp32_hi0, fp32_lo0);

        __m512 fp32_lo1 = _mm512_loadu_ps(inter1 + r);
        __m512 fp32_hi1 = _mm512_loadu_ps(inter1 + r + 16);
        __m512bh iv1 = _mm512_cvtne2ps_pbh(fp32_hi1, fp32_lo1);

        __m512 fp32_lo2 = _mm512_loadu_ps(inter2 + r);
        __m512 fp32_hi2 = _mm512_loadu_ps(inter2 + r + 16);
        __m512bh iv2 = _mm512_cvtne2ps_pbh(fp32_hi2, fp32_lo2);

        __m512 fp32_lo3 = _mm512_loadu_ps(inter3 + r);
        __m512 fp32_hi3 = _mm512_loadu_ps(inter3 + r + 16);
        __m512bh iv3 = _mm512_cvtne2ps_pbh(fp32_hi3, fp32_lo3);

        // Native BF16 dot product
        acc_t0_o0 = _mm512_dpbf16_ps(acc_t0_o0, iv0, wv0);
        acc_t0_o1 = _mm512_dpbf16_ps(acc_t0_o1, iv0, wv1);
        acc_t0_o2 = _mm512_dpbf16_ps(acc_t0_o2, iv0, wv2);
        acc_t0_o3 = _mm512_dpbf16_ps(acc_t0_o3, iv0, wv3);

        acc_t1_o0 = _mm512_dpbf16_ps(acc_t1_o0, iv1, wv0);
        acc_t1_o1 = _mm512_dpbf16_ps(acc_t1_o1, iv1, wv1);
        acc_t1_o2 = _mm512_dpbf16_ps(acc_t1_o2, iv1, wv2);
        acc_t1_o3 = _mm512_dpbf16_ps(acc_t1_o3, iv1, wv3);

        acc_t2_o0 = _mm512_dpbf16_ps(acc_t2_o0, iv2, wv0);
        acc_t2_o1 = _mm512_dpbf16_ps(acc_t2_o1, iv2, wv1);
        acc_t2_o2 = _mm512_dpbf16_ps(acc_t2_o2, iv2, wv2);
        acc_t2_o3 = _mm512_dpbf16_ps(acc_t2_o3, iv2, wv3);

        acc_t3_o0 = _mm512_dpbf16_ps(acc_t3_o0, iv3, wv0);
        acc_t3_o1 = _mm512_dpbf16_ps(acc_t3_o1, iv3, wv1);
        acc_t3_o2 = _mm512_dpbf16_ps(acc_t3_o2, iv3, wv2);
        acc_t3_o3 = _mm512_dpbf16_ps(acc_t3_o3, iv3, wv3);
      }

      // Reduce
      float s_t0_o0 = _mm512_reduce_add_ps(acc_t0_o0);
      float s_t0_o1 = _mm512_reduce_add_ps(acc_t0_o1);
      float s_t0_o2 = _mm512_reduce_add_ps(acc_t0_o2);
      float s_t0_o3 = _mm512_reduce_add_ps(acc_t0_o3);
      float s_t1_o0 = _mm512_reduce_add_ps(acc_t1_o0);
      float s_t1_o1 = _mm512_reduce_add_ps(acc_t1_o1);
      float s_t1_o2 = _mm512_reduce_add_ps(acc_t1_o2);
      float s_t1_o3 = _mm512_reduce_add_ps(acc_t1_o3);
      float s_t2_o0 = _mm512_reduce_add_ps(acc_t2_o0);
      float s_t2_o1 = _mm512_reduce_add_ps(acc_t2_o1);
      float s_t2_o2 = _mm512_reduce_add_ps(acc_t2_o2);
      float s_t2_o3 = _mm512_reduce_add_ps(acc_t2_o3);
      float s_t3_o0 = _mm512_reduce_add_ps(acc_t3_o0);
      float s_t3_o1 = _mm512_reduce_add_ps(acc_t3_o1);
      float s_t3_o2 = _mm512_reduce_add_ps(acc_t3_o2);
      float s_t3_o3 = _mm512_reduce_add_ps(acc_t3_o3);

      // Scalar tail
      for (; r < rank; r++) {
        float w0v = GGML_BF16_TO_FP32(w0[r]);
        float w1v = GGML_BF16_TO_FP32(w1[r]);
        float w2v = GGML_BF16_TO_FP32(w2[r]);
        float w3v = GGML_BF16_TO_FP32(w3[r]);
        s_t0_o0 += inter0[r] * w0v;
        s_t0_o1 += inter0[r] * w1v;
        s_t0_o2 += inter0[r] * w2v;
        s_t0_o3 += inter0[r] * w3v;
        s_t1_o0 += inter1[r] * w0v;
        s_t1_o1 += inter1[r] * w1v;
        s_t1_o2 += inter1[r] * w2v;
        s_t1_o3 += inter1[r] * w3v;
        s_t2_o0 += inter2[r] * w0v;
        s_t2_o1 += inter2[r] * w1v;
        s_t2_o2 += inter2[r] * w2v;
        s_t2_o3 += inter2[r] * w3v;
        s_t3_o0 += inter3[r] * w0v;
        s_t3_o1 += inter3[r] * w1v;
        s_t3_o2 += inter3[r] * w2v;
        s_t3_o3 += inter3[r] * w3v;
      }

      // Scale and store
      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
    }

    // Remainder outputs
    for (; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + r));
        __m512bh iv0 = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(inter0 + r + 16), _mm512_loadu_ps(inter0 + r));
        __m512bh iv1 = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(inter1 + r + 16), _mm512_loadu_ps(inter1 + r));
        __m512bh iv2 = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(inter2 + r + 16), _mm512_loadu_ps(inter2 + r));
        __m512bh iv3 = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(inter3 + r + 16), _mm512_loadu_ps(inter3 + r));
        acc0 = _mm512_dpbf16_ps(acc0, iv0, wv);
        acc1 = _mm512_dpbf16_ps(acc1, iv1, wv);
        acc2 = _mm512_dpbf16_ps(acc2, iv2, wv);
        acc3 = _mm512_dpbf16_ps(acc3, iv3, wv);
      }
      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);
      for (; r < rank; r++) {
        float wv = GGML_BF16_TO_FP32(w_row[r]);
        s0 += inter0[r] * wv;
        s1 += inter1[r] * wv;
        s2 += inter2[r] * wv;
        s3 += inter3[r] * wv;
      }
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Handle remaining tokens
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;
    for (int i = 0; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc = _mm512_setzero_ps();
      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        __m512bh wv = (__m512bh)_mm512_loadu_si512((__m512i*)(w_row + r));
        __m512bh iv = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(inter_row + r + 16), _mm512_loadu_ps(inter_row + r));
        acc = _mm512_dpbf16_ps(acc, iv, wv);
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(w_row[r]);
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v5: O_BLOCK=8, rank step=32 with 2 accumulators, masked tail, optimized reduce
// ============================================================================
void lora_fused_add_opt5(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;  // Increased from 4 to 8

  // Precompute tail mask for rank
  const int rank_tail = rank & 15;  // rank % 16
  const __mmask16 tail_mask = rank_tail ? ((__mmask16)1 << rank_tail) - 1 : 0;
  const int rank_aligned = rank & ~15;  // rank rounded down to 16

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
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;
      const ggml_bf16_t* w4 = weight + (i + 4) * rank;
      const ggml_bf16_t* w5 = weight + (i + 5) * rank;
      const ggml_bf16_t* w6 = weight + (i + 6) * rank;
      const ggml_bf16_t* w7 = weight + (i + 7) * rank;

      // 32 accumulators: 4 tokens × 8 outputs, each with 2 accumulators for latency hiding
      __m512 acc_t0_o0_a = _mm512_setzero_ps(), acc_t0_o0_b = _mm512_setzero_ps();
      __m512 acc_t0_o1_a = _mm512_setzero_ps(), acc_t0_o1_b = _mm512_setzero_ps();
      __m512 acc_t0_o2_a = _mm512_setzero_ps(), acc_t0_o2_b = _mm512_setzero_ps();
      __m512 acc_t0_o3_a = _mm512_setzero_ps(), acc_t0_o3_b = _mm512_setzero_ps();
      __m512 acc_t0_o4_a = _mm512_setzero_ps(), acc_t0_o4_b = _mm512_setzero_ps();
      __m512 acc_t0_o5_a = _mm512_setzero_ps(), acc_t0_o5_b = _mm512_setzero_ps();
      __m512 acc_t0_o6_a = _mm512_setzero_ps(), acc_t0_o6_b = _mm512_setzero_ps();
      __m512 acc_t0_o7_a = _mm512_setzero_ps(), acc_t0_o7_b = _mm512_setzero_ps();

      __m512 acc_t1_o0_a = _mm512_setzero_ps(), acc_t1_o0_b = _mm512_setzero_ps();
      __m512 acc_t1_o1_a = _mm512_setzero_ps(), acc_t1_o1_b = _mm512_setzero_ps();
      __m512 acc_t1_o2_a = _mm512_setzero_ps(), acc_t1_o2_b = _mm512_setzero_ps();
      __m512 acc_t1_o3_a = _mm512_setzero_ps(), acc_t1_o3_b = _mm512_setzero_ps();
      __m512 acc_t1_o4_a = _mm512_setzero_ps(), acc_t1_o4_b = _mm512_setzero_ps();
      __m512 acc_t1_o5_a = _mm512_setzero_ps(), acc_t1_o5_b = _mm512_setzero_ps();
      __m512 acc_t1_o6_a = _mm512_setzero_ps(), acc_t1_o6_b = _mm512_setzero_ps();
      __m512 acc_t1_o7_a = _mm512_setzero_ps(), acc_t1_o7_b = _mm512_setzero_ps();

      __m512 acc_t2_o0_a = _mm512_setzero_ps(), acc_t2_o0_b = _mm512_setzero_ps();
      __m512 acc_t2_o1_a = _mm512_setzero_ps(), acc_t2_o1_b = _mm512_setzero_ps();
      __m512 acc_t2_o2_a = _mm512_setzero_ps(), acc_t2_o2_b = _mm512_setzero_ps();
      __m512 acc_t2_o3_a = _mm512_setzero_ps(), acc_t2_o3_b = _mm512_setzero_ps();
      __m512 acc_t2_o4_a = _mm512_setzero_ps(), acc_t2_o4_b = _mm512_setzero_ps();
      __m512 acc_t2_o5_a = _mm512_setzero_ps(), acc_t2_o5_b = _mm512_setzero_ps();
      __m512 acc_t2_o6_a = _mm512_setzero_ps(), acc_t2_o6_b = _mm512_setzero_ps();
      __m512 acc_t2_o7_a = _mm512_setzero_ps(), acc_t2_o7_b = _mm512_setzero_ps();

      __m512 acc_t3_o0_a = _mm512_setzero_ps(), acc_t3_o0_b = _mm512_setzero_ps();
      __m512 acc_t3_o1_a = _mm512_setzero_ps(), acc_t3_o1_b = _mm512_setzero_ps();
      __m512 acc_t3_o2_a = _mm512_setzero_ps(), acc_t3_o2_b = _mm512_setzero_ps();
      __m512 acc_t3_o3_a = _mm512_setzero_ps(), acc_t3_o3_b = _mm512_setzero_ps();
      __m512 acc_t3_o4_a = _mm512_setzero_ps(), acc_t3_o4_b = _mm512_setzero_ps();
      __m512 acc_t3_o5_a = _mm512_setzero_ps(), acc_t3_o5_b = _mm512_setzero_ps();
      __m512 acc_t3_o6_a = _mm512_setzero_ps(), acc_t3_o6_b = _mm512_setzero_ps();
      __m512 acc_t3_o7_a = _mm512_setzero_ps(), acc_t3_o7_b = _mm512_setzero_ps();

      // Main loop: step by 32 (2×16)
      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        // Load intermediate values for 4 tokens, 2 chunks
        __m512 iv0_a = _mm512_loadu_ps(inter0 + r);
        __m512 iv0_b = _mm512_loadu_ps(inter0 + r + 16);
        __m512 iv1_a = _mm512_loadu_ps(inter1 + r);
        __m512 iv1_b = _mm512_loadu_ps(inter1 + r + 16);
        __m512 iv2_a = _mm512_loadu_ps(inter2 + r);
        __m512 iv2_b = _mm512_loadu_ps(inter2 + r + 16);
        __m512 iv3_a = _mm512_loadu_ps(inter3 + r);
        __m512 iv3_b = _mm512_loadu_ps(inter3 + r + 16);

// Load and convert weights for 8 outputs, 2 chunks each
#define LOAD_W_A(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w##idx + r))), 16))
#define LOAD_W_B(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w##idx + r + 16))), 16))

        __m512 wv0_a = LOAD_W_A(0);
        __m512 wv0_b = LOAD_W_B(0);
        __m512 wv1_a = LOAD_W_A(1);
        __m512 wv1_b = LOAD_W_B(1);
        __m512 wv2_a = LOAD_W_A(2);
        __m512 wv2_b = LOAD_W_B(2);
        __m512 wv3_a = LOAD_W_A(3);
        __m512 wv3_b = LOAD_W_B(3);
        __m512 wv4_a = LOAD_W_A(4);
        __m512 wv4_b = LOAD_W_B(4);
        __m512 wv5_a = LOAD_W_A(5);
        __m512 wv5_b = LOAD_W_B(5);
        __m512 wv6_a = LOAD_W_A(6);
        __m512 wv6_b = LOAD_W_B(6);
        __m512 wv7_a = LOAD_W_A(7);
        __m512 wv7_b = LOAD_W_B(7);

        // FMA for token 0
        acc_t0_o0_a = _mm512_fmadd_ps(iv0_a, wv0_a, acc_t0_o0_a);
        acc_t0_o0_b = _mm512_fmadd_ps(iv0_b, wv0_b, acc_t0_o0_b);
        acc_t0_o1_a = _mm512_fmadd_ps(iv0_a, wv1_a, acc_t0_o1_a);
        acc_t0_o1_b = _mm512_fmadd_ps(iv0_b, wv1_b, acc_t0_o1_b);
        acc_t0_o2_a = _mm512_fmadd_ps(iv0_a, wv2_a, acc_t0_o2_a);
        acc_t0_o2_b = _mm512_fmadd_ps(iv0_b, wv2_b, acc_t0_o2_b);
        acc_t0_o3_a = _mm512_fmadd_ps(iv0_a, wv3_a, acc_t0_o3_a);
        acc_t0_o3_b = _mm512_fmadd_ps(iv0_b, wv3_b, acc_t0_o3_b);
        acc_t0_o4_a = _mm512_fmadd_ps(iv0_a, wv4_a, acc_t0_o4_a);
        acc_t0_o4_b = _mm512_fmadd_ps(iv0_b, wv4_b, acc_t0_o4_b);
        acc_t0_o5_a = _mm512_fmadd_ps(iv0_a, wv5_a, acc_t0_o5_a);
        acc_t0_o5_b = _mm512_fmadd_ps(iv0_b, wv5_b, acc_t0_o5_b);
        acc_t0_o6_a = _mm512_fmadd_ps(iv0_a, wv6_a, acc_t0_o6_a);
        acc_t0_o6_b = _mm512_fmadd_ps(iv0_b, wv6_b, acc_t0_o6_b);
        acc_t0_o7_a = _mm512_fmadd_ps(iv0_a, wv7_a, acc_t0_o7_a);
        acc_t0_o7_b = _mm512_fmadd_ps(iv0_b, wv7_b, acc_t0_o7_b);

        // FMA for token 1
        acc_t1_o0_a = _mm512_fmadd_ps(iv1_a, wv0_a, acc_t1_o0_a);
        acc_t1_o0_b = _mm512_fmadd_ps(iv1_b, wv0_b, acc_t1_o0_b);
        acc_t1_o1_a = _mm512_fmadd_ps(iv1_a, wv1_a, acc_t1_o1_a);
        acc_t1_o1_b = _mm512_fmadd_ps(iv1_b, wv1_b, acc_t1_o1_b);
        acc_t1_o2_a = _mm512_fmadd_ps(iv1_a, wv2_a, acc_t1_o2_a);
        acc_t1_o2_b = _mm512_fmadd_ps(iv1_b, wv2_b, acc_t1_o2_b);
        acc_t1_o3_a = _mm512_fmadd_ps(iv1_a, wv3_a, acc_t1_o3_a);
        acc_t1_o3_b = _mm512_fmadd_ps(iv1_b, wv3_b, acc_t1_o3_b);
        acc_t1_o4_a = _mm512_fmadd_ps(iv1_a, wv4_a, acc_t1_o4_a);
        acc_t1_o4_b = _mm512_fmadd_ps(iv1_b, wv4_b, acc_t1_o4_b);
        acc_t1_o5_a = _mm512_fmadd_ps(iv1_a, wv5_a, acc_t1_o5_a);
        acc_t1_o5_b = _mm512_fmadd_ps(iv1_b, wv5_b, acc_t1_o5_b);
        acc_t1_o6_a = _mm512_fmadd_ps(iv1_a, wv6_a, acc_t1_o6_a);
        acc_t1_o6_b = _mm512_fmadd_ps(iv1_b, wv6_b, acc_t1_o6_b);
        acc_t1_o7_a = _mm512_fmadd_ps(iv1_a, wv7_a, acc_t1_o7_a);
        acc_t1_o7_b = _mm512_fmadd_ps(iv1_b, wv7_b, acc_t1_o7_b);

        // FMA for token 2
        acc_t2_o0_a = _mm512_fmadd_ps(iv2_a, wv0_a, acc_t2_o0_a);
        acc_t2_o0_b = _mm512_fmadd_ps(iv2_b, wv0_b, acc_t2_o0_b);
        acc_t2_o1_a = _mm512_fmadd_ps(iv2_a, wv1_a, acc_t2_o1_a);
        acc_t2_o1_b = _mm512_fmadd_ps(iv2_b, wv1_b, acc_t2_o1_b);
        acc_t2_o2_a = _mm512_fmadd_ps(iv2_a, wv2_a, acc_t2_o2_a);
        acc_t2_o2_b = _mm512_fmadd_ps(iv2_b, wv2_b, acc_t2_o2_b);
        acc_t2_o3_a = _mm512_fmadd_ps(iv2_a, wv3_a, acc_t2_o3_a);
        acc_t2_o3_b = _mm512_fmadd_ps(iv2_b, wv3_b, acc_t2_o3_b);
        acc_t2_o4_a = _mm512_fmadd_ps(iv2_a, wv4_a, acc_t2_o4_a);
        acc_t2_o4_b = _mm512_fmadd_ps(iv2_b, wv4_b, acc_t2_o4_b);
        acc_t2_o5_a = _mm512_fmadd_ps(iv2_a, wv5_a, acc_t2_o5_a);
        acc_t2_o5_b = _mm512_fmadd_ps(iv2_b, wv5_b, acc_t2_o5_b);
        acc_t2_o6_a = _mm512_fmadd_ps(iv2_a, wv6_a, acc_t2_o6_a);
        acc_t2_o6_b = _mm512_fmadd_ps(iv2_b, wv6_b, acc_t2_o6_b);
        acc_t2_o7_a = _mm512_fmadd_ps(iv2_a, wv7_a, acc_t2_o7_a);
        acc_t2_o7_b = _mm512_fmadd_ps(iv2_b, wv7_b, acc_t2_o7_b);

        // FMA for token 3
        acc_t3_o0_a = _mm512_fmadd_ps(iv3_a, wv0_a, acc_t3_o0_a);
        acc_t3_o0_b = _mm512_fmadd_ps(iv3_b, wv0_b, acc_t3_o0_b);
        acc_t3_o1_a = _mm512_fmadd_ps(iv3_a, wv1_a, acc_t3_o1_a);
        acc_t3_o1_b = _mm512_fmadd_ps(iv3_b, wv1_b, acc_t3_o1_b);
        acc_t3_o2_a = _mm512_fmadd_ps(iv3_a, wv2_a, acc_t3_o2_a);
        acc_t3_o2_b = _mm512_fmadd_ps(iv3_b, wv2_b, acc_t3_o2_b);
        acc_t3_o3_a = _mm512_fmadd_ps(iv3_a, wv3_a, acc_t3_o3_a);
        acc_t3_o3_b = _mm512_fmadd_ps(iv3_b, wv3_b, acc_t3_o3_b);
        acc_t3_o4_a = _mm512_fmadd_ps(iv3_a, wv4_a, acc_t3_o4_a);
        acc_t3_o4_b = _mm512_fmadd_ps(iv3_b, wv4_b, acc_t3_o4_b);
        acc_t3_o5_a = _mm512_fmadd_ps(iv3_a, wv5_a, acc_t3_o5_a);
        acc_t3_o5_b = _mm512_fmadd_ps(iv3_b, wv5_b, acc_t3_o5_b);
        acc_t3_o6_a = _mm512_fmadd_ps(iv3_a, wv6_a, acc_t3_o6_a);
        acc_t3_o6_b = _mm512_fmadd_ps(iv3_b, wv6_b, acc_t3_o6_b);
        acc_t3_o7_a = _mm512_fmadd_ps(iv3_a, wv7_a, acc_t3_o7_a);
        acc_t3_o7_b = _mm512_fmadd_ps(iv3_b, wv7_b, acc_t3_o7_b);

#undef LOAD_W_A
#undef LOAD_W_B
      }

      // Handle 16-element chunk if remaining
      if (r + 16 <= rank) {
        __m512 iv0_a = _mm512_loadu_ps(inter0 + r);
        __m512 iv1_a = _mm512_loadu_ps(inter1 + r);
        __m512 iv2_a = _mm512_loadu_ps(inter2 + r);
        __m512 iv3_a = _mm512_loadu_ps(inter3 + r);

#define LOAD_W_A(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w##idx + r))), 16))
        __m512 wv0_a = LOAD_W_A(0);
        __m512 wv1_a = LOAD_W_A(1);
        __m512 wv2_a = LOAD_W_A(2);
        __m512 wv3_a = LOAD_W_A(3);
        __m512 wv4_a = LOAD_W_A(4);
        __m512 wv5_a = LOAD_W_A(5);
        __m512 wv6_a = LOAD_W_A(6);
        __m512 wv7_a = LOAD_W_A(7);

        acc_t0_o0_a = _mm512_fmadd_ps(iv0_a, wv0_a, acc_t0_o0_a);
        acc_t0_o1_a = _mm512_fmadd_ps(iv0_a, wv1_a, acc_t0_o1_a);
        acc_t0_o2_a = _mm512_fmadd_ps(iv0_a, wv2_a, acc_t0_o2_a);
        acc_t0_o3_a = _mm512_fmadd_ps(iv0_a, wv3_a, acc_t0_o3_a);
        acc_t0_o4_a = _mm512_fmadd_ps(iv0_a, wv4_a, acc_t0_o4_a);
        acc_t0_o5_a = _mm512_fmadd_ps(iv0_a, wv5_a, acc_t0_o5_a);
        acc_t0_o6_a = _mm512_fmadd_ps(iv0_a, wv6_a, acc_t0_o6_a);
        acc_t0_o7_a = _mm512_fmadd_ps(iv0_a, wv7_a, acc_t0_o7_a);

        acc_t1_o0_a = _mm512_fmadd_ps(iv1_a, wv0_a, acc_t1_o0_a);
        acc_t1_o1_a = _mm512_fmadd_ps(iv1_a, wv1_a, acc_t1_o1_a);
        acc_t1_o2_a = _mm512_fmadd_ps(iv1_a, wv2_a, acc_t1_o2_a);
        acc_t1_o3_a = _mm512_fmadd_ps(iv1_a, wv3_a, acc_t1_o3_a);
        acc_t1_o4_a = _mm512_fmadd_ps(iv1_a, wv4_a, acc_t1_o4_a);
        acc_t1_o5_a = _mm512_fmadd_ps(iv1_a, wv5_a, acc_t1_o5_a);
        acc_t1_o6_a = _mm512_fmadd_ps(iv1_a, wv6_a, acc_t1_o6_a);
        acc_t1_o7_a = _mm512_fmadd_ps(iv1_a, wv7_a, acc_t1_o7_a);

        acc_t2_o0_a = _mm512_fmadd_ps(iv2_a, wv0_a, acc_t2_o0_a);
        acc_t2_o1_a = _mm512_fmadd_ps(iv2_a, wv1_a, acc_t2_o1_a);
        acc_t2_o2_a = _mm512_fmadd_ps(iv2_a, wv2_a, acc_t2_o2_a);
        acc_t2_o3_a = _mm512_fmadd_ps(iv2_a, wv3_a, acc_t2_o3_a);
        acc_t2_o4_a = _mm512_fmadd_ps(iv2_a, wv4_a, acc_t2_o4_a);
        acc_t2_o5_a = _mm512_fmadd_ps(iv2_a, wv5_a, acc_t2_o5_a);
        acc_t2_o6_a = _mm512_fmadd_ps(iv2_a, wv6_a, acc_t2_o6_a);
        acc_t2_o7_a = _mm512_fmadd_ps(iv2_a, wv7_a, acc_t2_o7_a);

        acc_t3_o0_a = _mm512_fmadd_ps(iv3_a, wv0_a, acc_t3_o0_a);
        acc_t3_o1_a = _mm512_fmadd_ps(iv3_a, wv1_a, acc_t3_o1_a);
        acc_t3_o2_a = _mm512_fmadd_ps(iv3_a, wv2_a, acc_t3_o2_a);
        acc_t3_o3_a = _mm512_fmadd_ps(iv3_a, wv3_a, acc_t3_o3_a);
        acc_t3_o4_a = _mm512_fmadd_ps(iv3_a, wv4_a, acc_t3_o4_a);
        acc_t3_o5_a = _mm512_fmadd_ps(iv3_a, wv5_a, acc_t3_o5_a);
        acc_t3_o6_a = _mm512_fmadd_ps(iv3_a, wv6_a, acc_t3_o6_a);
        acc_t3_o7_a = _mm512_fmadd_ps(iv3_a, wv7_a, acc_t3_o7_a);

#undef LOAD_W_A
        r += 16;
      }

      // Masked tail: handle remaining elements with mask
      if (tail_mask) {
        __m512 iv0_a = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1_a = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);
        __m512 iv2_a = _mm512_maskz_loadu_ps(tail_mask, inter2 + r);
        __m512 iv3_a = _mm512_maskz_loadu_ps(tail_mask, inter3 + r);

#define LOAD_W_MASK(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w##idx + r)), 16))
        __m512 wv0_a = LOAD_W_MASK(0);
        __m512 wv1_a = LOAD_W_MASK(1);
        __m512 wv2_a = LOAD_W_MASK(2);
        __m512 wv3_a = LOAD_W_MASK(3);
        __m512 wv4_a = LOAD_W_MASK(4);
        __m512 wv5_a = LOAD_W_MASK(5);
        __m512 wv6_a = LOAD_W_MASK(6);
        __m512 wv7_a = LOAD_W_MASK(7);

        acc_t0_o0_a = _mm512_fmadd_ps(iv0_a, wv0_a, acc_t0_o0_a);
        acc_t0_o1_a = _mm512_fmadd_ps(iv0_a, wv1_a, acc_t0_o1_a);
        acc_t0_o2_a = _mm512_fmadd_ps(iv0_a, wv2_a, acc_t0_o2_a);
        acc_t0_o3_a = _mm512_fmadd_ps(iv0_a, wv3_a, acc_t0_o3_a);
        acc_t0_o4_a = _mm512_fmadd_ps(iv0_a, wv4_a, acc_t0_o4_a);
        acc_t0_o5_a = _mm512_fmadd_ps(iv0_a, wv5_a, acc_t0_o5_a);
        acc_t0_o6_a = _mm512_fmadd_ps(iv0_a, wv6_a, acc_t0_o6_a);
        acc_t0_o7_a = _mm512_fmadd_ps(iv0_a, wv7_a, acc_t0_o7_a);

        acc_t1_o0_a = _mm512_fmadd_ps(iv1_a, wv0_a, acc_t1_o0_a);
        acc_t1_o1_a = _mm512_fmadd_ps(iv1_a, wv1_a, acc_t1_o1_a);
        acc_t1_o2_a = _mm512_fmadd_ps(iv1_a, wv2_a, acc_t1_o2_a);
        acc_t1_o3_a = _mm512_fmadd_ps(iv1_a, wv3_a, acc_t1_o3_a);
        acc_t1_o4_a = _mm512_fmadd_ps(iv1_a, wv4_a, acc_t1_o4_a);
        acc_t1_o5_a = _mm512_fmadd_ps(iv1_a, wv5_a, acc_t1_o5_a);
        acc_t1_o6_a = _mm512_fmadd_ps(iv1_a, wv6_a, acc_t1_o6_a);
        acc_t1_o7_a = _mm512_fmadd_ps(iv1_a, wv7_a, acc_t1_o7_a);

        acc_t2_o0_a = _mm512_fmadd_ps(iv2_a, wv0_a, acc_t2_o0_a);
        acc_t2_o1_a = _mm512_fmadd_ps(iv2_a, wv1_a, acc_t2_o1_a);
        acc_t2_o2_a = _mm512_fmadd_ps(iv2_a, wv2_a, acc_t2_o2_a);
        acc_t2_o3_a = _mm512_fmadd_ps(iv2_a, wv3_a, acc_t2_o3_a);
        acc_t2_o4_a = _mm512_fmadd_ps(iv2_a, wv4_a, acc_t2_o4_a);
        acc_t2_o5_a = _mm512_fmadd_ps(iv2_a, wv5_a, acc_t2_o5_a);
        acc_t2_o6_a = _mm512_fmadd_ps(iv2_a, wv6_a, acc_t2_o6_a);
        acc_t2_o7_a = _mm512_fmadd_ps(iv2_a, wv7_a, acc_t2_o7_a);

        acc_t3_o0_a = _mm512_fmadd_ps(iv3_a, wv0_a, acc_t3_o0_a);
        acc_t3_o1_a = _mm512_fmadd_ps(iv3_a, wv1_a, acc_t3_o1_a);
        acc_t3_o2_a = _mm512_fmadd_ps(iv3_a, wv2_a, acc_t3_o2_a);
        acc_t3_o3_a = _mm512_fmadd_ps(iv3_a, wv3_a, acc_t3_o3_a);
        acc_t3_o4_a = _mm512_fmadd_ps(iv3_a, wv4_a, acc_t3_o4_a);
        acc_t3_o5_a = _mm512_fmadd_ps(iv3_a, wv5_a, acc_t3_o5_a);
        acc_t3_o6_a = _mm512_fmadd_ps(iv3_a, wv6_a, acc_t3_o6_a);
        acc_t3_o7_a = _mm512_fmadd_ps(iv3_a, wv7_a, acc_t3_o7_a);

#undef LOAD_W_MASK
      }

// Optimized reduce: first add a+b, then hsum
#define REDUCE_AND_STORE(t, o) _mm512_reduce_add_ps(_mm512_add_ps(acc_t##t##_o##o##_a, acc_t##t##_o##o##_b))

      float s_t0_o0 = REDUCE_AND_STORE(0, 0);
      float s_t0_o1 = REDUCE_AND_STORE(0, 1);
      float s_t0_o2 = REDUCE_AND_STORE(0, 2);
      float s_t0_o3 = REDUCE_AND_STORE(0, 3);
      float s_t0_o4 = REDUCE_AND_STORE(0, 4);
      float s_t0_o5 = REDUCE_AND_STORE(0, 5);
      float s_t0_o6 = REDUCE_AND_STORE(0, 6);
      float s_t0_o7 = REDUCE_AND_STORE(0, 7);

      float s_t1_o0 = REDUCE_AND_STORE(1, 0);
      float s_t1_o1 = REDUCE_AND_STORE(1, 1);
      float s_t1_o2 = REDUCE_AND_STORE(1, 2);
      float s_t1_o3 = REDUCE_AND_STORE(1, 3);
      float s_t1_o4 = REDUCE_AND_STORE(1, 4);
      float s_t1_o5 = REDUCE_AND_STORE(1, 5);
      float s_t1_o6 = REDUCE_AND_STORE(1, 6);
      float s_t1_o7 = REDUCE_AND_STORE(1, 7);

      float s_t2_o0 = REDUCE_AND_STORE(2, 0);
      float s_t2_o1 = REDUCE_AND_STORE(2, 1);
      float s_t2_o2 = REDUCE_AND_STORE(2, 2);
      float s_t2_o3 = REDUCE_AND_STORE(2, 3);
      float s_t2_o4 = REDUCE_AND_STORE(2, 4);
      float s_t2_o5 = REDUCE_AND_STORE(2, 5);
      float s_t2_o6 = REDUCE_AND_STORE(2, 6);
      float s_t2_o7 = REDUCE_AND_STORE(2, 7);

      float s_t3_o0 = REDUCE_AND_STORE(3, 0);
      float s_t3_o1 = REDUCE_AND_STORE(3, 1);
      float s_t3_o2 = REDUCE_AND_STORE(3, 2);
      float s_t3_o3 = REDUCE_AND_STORE(3, 3);
      float s_t3_o4 = REDUCE_AND_STORE(3, 4);
      float s_t3_o5 = REDUCE_AND_STORE(3, 5);
      float s_t3_o6 = REDUCE_AND_STORE(3, 6);
      float s_t3_o7 = REDUCE_AND_STORE(3, 7);

#undef REDUCE_AND_STORE

      // Store results
      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out0[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 4]) + s_t0_o4 * scale);
      out0[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 5]) + s_t0_o5 * scale);
      out0[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 6]) + s_t0_o6 * scale);
      out0[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 7]) + s_t0_o7 * scale);

      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out1[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 4]) + s_t1_o4 * scale);
      out1[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 5]) + s_t1_o5 * scale);
      out1[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 6]) + s_t1_o6 * scale);
      out1[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 7]) + s_t1_o7 * scale);

      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out2[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 4]) + s_t2_o4 * scale);
      out2[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 5]) + s_t2_o5 * scale);
      out2[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 6]) + s_t2_o6 * scale);
      out2[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 7]) + s_t2_o7 * scale);

      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
      out3[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 4]) + s_t3_o4 * scale);
      out3[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 5]) + s_t3_o5 * scale);
      out3[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 6]) + s_t3_o6 * scale);
      out3[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 7]) + s_t3_o7 * scale);
    }

    // Remainder outputs (one at a time)
    for (; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc0_a = _mm512_setzero_ps(), acc0_b = _mm512_setzero_ps();
      __m512 acc1_a = _mm512_setzero_ps(), acc1_b = _mm512_setzero_ps();
      __m512 acc2_a = _mm512_setzero_ps(), acc2_b = _mm512_setzero_ps();
      __m512 acc3_a = _mm512_setzero_ps(), acc3_b = _mm512_setzero_ps();

      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        __m512 wv_a = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        __m512 wv_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r + 16))), 16));
        acc0_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r), wv_a, acc0_a);
        acc0_b = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r + 16), wv_b, acc0_b);
        acc1_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r), wv_a, acc1_a);
        acc1_b = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r + 16), wv_b, acc1_b);
        acc2_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter2 + r), wv_a, acc2_a);
        acc2_b = _mm512_fmadd_ps(_mm512_loadu_ps(inter2 + r + 16), wv_b, acc2_b);
        acc3_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter3 + r), wv_a, acc3_a);
        acc3_b = _mm512_fmadd_ps(_mm512_loadu_ps(inter3 + r + 16), wv_b, acc3_b);
      }
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc0_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r), wv, acc0_a);
        acc1_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r), wv, acc1_a);
        acc2_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter2 + r), wv, acc2_a);
        acc3_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter3 + r), wv, acc3_a);
      }
      if (tail_mask) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w_row + r)), 16));
        acc0_a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter0 + r), wv, acc0_a);
        acc1_a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter1 + r), wv, acc1_a);
        acc2_a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter2 + r), wv, acc2_a);
        acc3_a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter3 + r), wv, acc3_a);
      }

      float s0 = _mm512_reduce_add_ps(_mm512_add_ps(acc0_a, acc0_b));
      float s1 = _mm512_reduce_add_ps(_mm512_add_ps(acc1_a, acc1_b));
      float s2 = _mm512_reduce_add_ps(_mm512_add_ps(acc2_a, acc2_b));
      float s3 = _mm512_reduce_add_ps(_mm512_add_ps(acc3_a, acc3_b));

      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Handle remaining tokens (one at a time)
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;
    for (int i = 0; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc_a = _mm512_setzero_ps(), acc_b = _mm512_setzero_ps();
      int r = 0;
      for (; r + 32 <= rank; r += 32) {
        __m512 wv_a = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        __m512 wv_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r + 16))), 16));
        acc_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r), wv_a, acc_a);
        acc_b = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r + 16), wv_b, acc_b);
      }
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc_a = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r), wv, acc_a);
      }
      if (tail_mask) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w_row + r)), 16));
        acc_a = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter_row + r), wv, acc_a);
      }
      float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc_a, acc_b));
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v6: O_BLOCK=8 with single accumulator, masked tail, step=32 unroll
// Balances register pressure with better inter_vec reuse
// ============================================================================
void lora_fused_add_opt6(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;

  // Precompute tail mask
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
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;
      const ggml_bf16_t* w4 = weight + (i + 4) * rank;
      const ggml_bf16_t* w5 = weight + (i + 5) * rank;
      const ggml_bf16_t* w6 = weight + (i + 6) * rank;
      const ggml_bf16_t* w7 = weight + (i + 7) * rank;

      // 32 accumulators: 4 tokens × 8 outputs (single acc per output)
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

      // Main loop: step by 16
      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        // Load intermediate for 4 tokens
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

// Load and convert weights for 8 outputs
#define LOAD_W(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w##idx + r))), 16))
        __m512 wv0 = LOAD_W(0);
        __m512 wv1 = LOAD_W(1);
        __m512 wv2 = LOAD_W(2);
        __m512 wv3 = LOAD_W(3);
        __m512 wv4 = LOAD_W(4);
        __m512 wv5 = LOAD_W(5);
        __m512 wv6 = LOAD_W(6);
        __m512 wv7 = LOAD_W(7);

        // FMA for token 0
        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);
        acc_t0_o4 = _mm512_fmadd_ps(iv0, wv4, acc_t0_o4);
        acc_t0_o5 = _mm512_fmadd_ps(iv0, wv5, acc_t0_o5);
        acc_t0_o6 = _mm512_fmadd_ps(iv0, wv6, acc_t0_o6);
        acc_t0_o7 = _mm512_fmadd_ps(iv0, wv7, acc_t0_o7);

        // FMA for token 1
        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);
        acc_t1_o4 = _mm512_fmadd_ps(iv1, wv4, acc_t1_o4);
        acc_t1_o5 = _mm512_fmadd_ps(iv1, wv5, acc_t1_o5);
        acc_t1_o6 = _mm512_fmadd_ps(iv1, wv6, acc_t1_o6);
        acc_t1_o7 = _mm512_fmadd_ps(iv1, wv7, acc_t1_o7);

        // FMA for token 2
        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);
        acc_t2_o4 = _mm512_fmadd_ps(iv2, wv4, acc_t2_o4);
        acc_t2_o5 = _mm512_fmadd_ps(iv2, wv5, acc_t2_o5);
        acc_t2_o6 = _mm512_fmadd_ps(iv2, wv6, acc_t2_o6);
        acc_t2_o7 = _mm512_fmadd_ps(iv2, wv7, acc_t2_o7);

        // FMA for token 3
        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
        acc_t3_o4 = _mm512_fmadd_ps(iv3, wv4, acc_t3_o4);
        acc_t3_o5 = _mm512_fmadd_ps(iv3, wv5, acc_t3_o5);
        acc_t3_o6 = _mm512_fmadd_ps(iv3, wv6, acc_t3_o6);
        acc_t3_o7 = _mm512_fmadd_ps(iv3, wv7, acc_t3_o7);

#undef LOAD_W
      }

      // Masked tail
      if (tail_mask) {
        __m512 iv0 = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1 = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);
        __m512 iv2 = _mm512_maskz_loadu_ps(tail_mask, inter2 + r);
        __m512 iv3 = _mm512_maskz_loadu_ps(tail_mask, inter3 + r);

#define LOAD_W_MASK(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w##idx + r)), 16))
        __m512 wv0 = LOAD_W_MASK(0);
        __m512 wv1 = LOAD_W_MASK(1);
        __m512 wv2 = LOAD_W_MASK(2);
        __m512 wv3 = LOAD_W_MASK(3);
        __m512 wv4 = LOAD_W_MASK(4);
        __m512 wv5 = LOAD_W_MASK(5);
        __m512 wv6 = LOAD_W_MASK(6);
        __m512 wv7 = LOAD_W_MASK(7);

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

      // Reduce and store
      float s_t0_o0 = _mm512_reduce_add_ps(acc_t0_o0);
      float s_t0_o1 = _mm512_reduce_add_ps(acc_t0_o1);
      float s_t0_o2 = _mm512_reduce_add_ps(acc_t0_o2);
      float s_t0_o3 = _mm512_reduce_add_ps(acc_t0_o3);
      float s_t0_o4 = _mm512_reduce_add_ps(acc_t0_o4);
      float s_t0_o5 = _mm512_reduce_add_ps(acc_t0_o5);
      float s_t0_o6 = _mm512_reduce_add_ps(acc_t0_o6);
      float s_t0_o7 = _mm512_reduce_add_ps(acc_t0_o7);

      float s_t1_o0 = _mm512_reduce_add_ps(acc_t1_o0);
      float s_t1_o1 = _mm512_reduce_add_ps(acc_t1_o1);
      float s_t1_o2 = _mm512_reduce_add_ps(acc_t1_o2);
      float s_t1_o3 = _mm512_reduce_add_ps(acc_t1_o3);
      float s_t1_o4 = _mm512_reduce_add_ps(acc_t1_o4);
      float s_t1_o5 = _mm512_reduce_add_ps(acc_t1_o5);
      float s_t1_o6 = _mm512_reduce_add_ps(acc_t1_o6);
      float s_t1_o7 = _mm512_reduce_add_ps(acc_t1_o7);

      float s_t2_o0 = _mm512_reduce_add_ps(acc_t2_o0);
      float s_t2_o1 = _mm512_reduce_add_ps(acc_t2_o1);
      float s_t2_o2 = _mm512_reduce_add_ps(acc_t2_o2);
      float s_t2_o3 = _mm512_reduce_add_ps(acc_t2_o3);
      float s_t2_o4 = _mm512_reduce_add_ps(acc_t2_o4);
      float s_t2_o5 = _mm512_reduce_add_ps(acc_t2_o5);
      float s_t2_o6 = _mm512_reduce_add_ps(acc_t2_o6);
      float s_t2_o7 = _mm512_reduce_add_ps(acc_t2_o7);

      float s_t3_o0 = _mm512_reduce_add_ps(acc_t3_o0);
      float s_t3_o1 = _mm512_reduce_add_ps(acc_t3_o1);
      float s_t3_o2 = _mm512_reduce_add_ps(acc_t3_o2);
      float s_t3_o3 = _mm512_reduce_add_ps(acc_t3_o3);
      float s_t3_o4 = _mm512_reduce_add_ps(acc_t3_o4);
      float s_t3_o5 = _mm512_reduce_add_ps(acc_t3_o5);
      float s_t3_o6 = _mm512_reduce_add_ps(acc_t3_o6);
      float s_t3_o7 = _mm512_reduce_add_ps(acc_t3_o7);

      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out0[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 4]) + s_t0_o4 * scale);
      out0[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 5]) + s_t0_o5 * scale);
      out0[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 6]) + s_t0_o6 * scale);
      out0[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 7]) + s_t0_o7 * scale);

      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out1[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 4]) + s_t1_o4 * scale);
      out1[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 5]) + s_t1_o5 * scale);
      out1[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 6]) + s_t1_o6 * scale);
      out1[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 7]) + s_t1_o7 * scale);

      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out2[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 4]) + s_t2_o4 * scale);
      out2[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 5]) + s_t2_o5 * scale);
      out2[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 6]) + s_t2_o6 * scale);
      out2[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 7]) + s_t2_o7 * scale);

      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
      out3[i + 4] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 4]) + s_t3_o4 * scale);
      out3[i + 5] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 5]) + s_t3_o5 * scale);
      out3[i + 6] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 6]) + s_t3_o6 * scale);
      out3[i + 7] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 7]) + s_t3_o7 * scale);
    }

    // Remainder outputs
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

      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);

      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Handle remaining tokens
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
      float sum = _mm512_reduce_add_ps(acc);
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v7: Vectorized reduce + store, reduce Port 0 pressure
// Key changes:
// 1. Pack 8 reduce results into __m256 for vectorized store
// 2. Use transpose to convert row-major accumulator to column-major for reduce
// 3. Vectorized BF16 load/store for output
// ============================================================================

// Helper: horizontal sum of 8 __m512 accumulators, return as __m256
inline __m256 hsum_8x512_to_256(__m512 a0, __m512 a1, __m512 a2, __m512 a3, __m512 a4, __m512 a5, __m512 a6,
                                __m512 a7) {
  // Reduce each 512-bit to 256-bit by adding high and low halves
  __m256 h0 = _mm256_add_ps(_mm512_castps512_ps256(a0), _mm512_extractf32x8_ps(a0, 1));
  __m256 h1 = _mm256_add_ps(_mm512_castps512_ps256(a1), _mm512_extractf32x8_ps(a1, 1));
  __m256 h2 = _mm256_add_ps(_mm512_castps512_ps256(a2), _mm512_extractf32x8_ps(a2, 1));
  __m256 h3 = _mm256_add_ps(_mm512_castps512_ps256(a3), _mm512_extractf32x8_ps(a3, 1));
  __m256 h4 = _mm256_add_ps(_mm512_castps512_ps256(a4), _mm512_extractf32x8_ps(a4, 1));
  __m256 h5 = _mm256_add_ps(_mm512_castps512_ps256(a5), _mm512_extractf32x8_ps(a5, 1));
  __m256 h6 = _mm256_add_ps(_mm512_castps512_ps256(a6), _mm512_extractf32x8_ps(a6, 1));
  __m256 h7 = _mm256_add_ps(_mm512_castps512_ps256(a7), _mm512_extractf32x8_ps(a7, 1));

  // Now each h0-h7 is 256-bit (8 floats), need to reduce to 1 float each
  // Reduce 256 -> 128
  __m128 q0 = _mm_add_ps(_mm256_castps256_ps128(h0), _mm256_extractf128_ps(h0, 1));
  __m128 q1 = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));
  __m128 q2 = _mm_add_ps(_mm256_castps256_ps128(h2), _mm256_extractf128_ps(h2, 1));
  __m128 q3 = _mm_add_ps(_mm256_castps256_ps128(h3), _mm256_extractf128_ps(h3, 1));
  __m128 q4 = _mm_add_ps(_mm256_castps256_ps128(h4), _mm256_extractf128_ps(h4, 1));
  __m128 q5 = _mm_add_ps(_mm256_castps256_ps128(h5), _mm256_extractf128_ps(h5, 1));
  __m128 q6 = _mm_add_ps(_mm256_castps256_ps128(h6), _mm256_extractf128_ps(h6, 1));
  __m128 q7 = _mm_add_ps(_mm256_castps256_ps128(h7), _mm256_extractf128_ps(h7, 1));

  // Reduce 128 -> 64 (2 floats)
  q0 = _mm_add_ps(q0, _mm_movehl_ps(q0, q0));
  q1 = _mm_add_ps(q1, _mm_movehl_ps(q1, q1));
  q2 = _mm_add_ps(q2, _mm_movehl_ps(q2, q2));
  q3 = _mm_add_ps(q3, _mm_movehl_ps(q3, q3));
  q4 = _mm_add_ps(q4, _mm_movehl_ps(q4, q4));
  q5 = _mm_add_ps(q5, _mm_movehl_ps(q5, q5));
  q6 = _mm_add_ps(q6, _mm_movehl_ps(q6, q6));
  q7 = _mm_add_ps(q7, _mm_movehl_ps(q7, q7));

  // Reduce 64 -> 32 (1 float)
  q0 = _mm_add_ss(q0, _mm_shuffle_ps(q0, q0, 1));
  q1 = _mm_add_ss(q1, _mm_shuffle_ps(q1, q1, 1));
  q2 = _mm_add_ss(q2, _mm_shuffle_ps(q2, q2, 1));
  q3 = _mm_add_ss(q3, _mm_shuffle_ps(q3, q3, 1));
  q4 = _mm_add_ss(q4, _mm_shuffle_ps(q4, q4, 1));
  q5 = _mm_add_ss(q5, _mm_shuffle_ps(q5, q5, 1));
  q6 = _mm_add_ss(q6, _mm_shuffle_ps(q6, q6, 1));
  q7 = _mm_add_ss(q7, _mm_shuffle_ps(q7, q7, 1));

  // Pack 8 scalar results into __m256
  // q0-q3 -> low 128 bits, q4-q7 -> high 128 bits
  __m128 lo = _mm_unpacklo_ps(q0, q1);   // [s0, s1, ?, ?]
  __m128 lo2 = _mm_unpacklo_ps(q2, q3);  // [s2, s3, ?, ?]
  lo = _mm_movelh_ps(lo, lo2);           // [s0, s1, s2, s3]

  __m128 hi = _mm_unpacklo_ps(q4, q5);   // [s4, s5, ?, ?]
  __m128 hi2 = _mm_unpacklo_ps(q6, q7);  // [s6, s7, ?, ?]
  hi = _mm_movelh_ps(hi, hi2);           // [s4, s5, s6, s7]

  return _mm256_set_m128(hi, lo);  // [s0, s1, s2, s3, s4, s5, s6, s7]
}

void lora_fused_add_opt7(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;

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
      for (; r + 16 <= rank; r += 16) {
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

#define LOAD_W(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w##idx + r))), 16))
        __m512 wv0 = LOAD_W(0);
        __m512 wv1 = LOAD_W(1);
        __m512 wv2 = LOAD_W(2);
        __m512 wv3 = LOAD_W(3);
        __m512 wv4 = LOAD_W(4);
        __m512 wv5 = LOAD_W(5);
        __m512 wv6 = LOAD_W(6);
        __m512 wv7 = LOAD_W(7);

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

#undef LOAD_W
      }

      if (tail_mask) {
        __m512 iv0 = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1 = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);
        __m512 iv2 = _mm512_maskz_loadu_ps(tail_mask, inter2 + r);
        __m512 iv3 = _mm512_maskz_loadu_ps(tail_mask, inter3 + r);

#define LOAD_W_MASK(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w##idx + r)), 16))
        __m512 wv0 = LOAD_W_MASK(0);
        __m512 wv1 = LOAD_W_MASK(1);
        __m512 wv2 = LOAD_W_MASK(2);
        __m512 wv3 = LOAD_W_MASK(3);
        __m512 wv4 = LOAD_W_MASK(4);
        __m512 wv5 = LOAD_W_MASK(5);
        __m512 wv6 = LOAD_W_MASK(6);
        __m512 wv7 = LOAD_W_MASK(7);

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

      // Vectorized reduce: 8 accumulators -> 1 __m256 (8 floats)
      __m256 sum_t0 =
          hsum_8x512_to_256(acc_t0_o0, acc_t0_o1, acc_t0_o2, acc_t0_o3, acc_t0_o4, acc_t0_o5, acc_t0_o6, acc_t0_o7);
      __m256 sum_t1 =
          hsum_8x512_to_256(acc_t1_o0, acc_t1_o1, acc_t1_o2, acc_t1_o3, acc_t1_o4, acc_t1_o5, acc_t1_o6, acc_t1_o7);
      __m256 sum_t2 =
          hsum_8x512_to_256(acc_t2_o0, acc_t2_o1, acc_t2_o2, acc_t2_o3, acc_t2_o4, acc_t2_o5, acc_t2_o6, acc_t2_o7);
      __m256 sum_t3 =
          hsum_8x512_to_256(acc_t3_o0, acc_t3_o1, acc_t3_o2, acc_t3_o3, acc_t3_o4, acc_t3_o5, acc_t3_o6, acc_t3_o7);

      // Apply scale
      sum_t0 = _mm256_mul_ps(sum_t0, scale_vec);
      sum_t1 = _mm256_mul_ps(sum_t1, scale_vec);
      sum_t2 = _mm256_mul_ps(sum_t2, scale_vec);
      sum_t3 = _mm256_mul_ps(sum_t3, scale_vec);

      // Load existing output, convert BF16->FP32, add, convert back, store
      // Load 8 BF16 values -> convert to FP32
      __m256 out_t0 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out0 + i))), 16));
      __m256 out_t1 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out1 + i))), 16));
      __m256 out_t2 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out2 + i))), 16));
      __m256 out_t3 =
          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(out3 + i))), 16));

      // Add
      out_t0 = _mm256_add_ps(out_t0, sum_t0);
      out_t1 = _mm256_add_ps(out_t1, sum_t1);
      out_t2 = _mm256_add_ps(out_t2, sum_t2);
      out_t3 = _mm256_add_ps(out_t3, sum_t3);

      // Convert FP32 -> BF16 and store
      // Use VCVTNEPS2BF16, cast __m128bh to __m128i for store
      __m128bh bf16_t0 = _mm256_cvtneps_pbh(out_t0);
      __m128bh bf16_t1 = _mm256_cvtneps_pbh(out_t1);
      __m128bh bf16_t2 = _mm256_cvtneps_pbh(out_t2);
      __m128bh bf16_t3 = _mm256_cvtneps_pbh(out_t3);

      _mm_storeu_si128((__m128i*)(out0 + i), (__m128i)bf16_t0);
      _mm_storeu_si128((__m128i*)(out1 + i), (__m128i)bf16_t1);
      _mm_storeu_si128((__m128i*)(out2 + i), (__m128i)bf16_t2);
      _mm_storeu_si128((__m128i*)(out3 + i), (__m128i)bf16_t3);
    }

    // Remainder outputs
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

      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);

      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Handle remaining tokens
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
      float sum = _mm512_reduce_add_ps(acc);
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v8: 512-bit reduce, software prefetch, reduced shuffles
// ============================================================================

// Reduce 8 x __m512 to __m256 (8 floats) using 512-bit operations
// Approach: pair-wise reduction in 512-bit, then extract
inline __m256 hsum_8x512_to_256_v2(__m512 a0, __m512 a1, __m512 a2, __m512 a3, __m512 a4, __m512 a5, __m512 a6,
                                   __m512 a7) {
  // Step 1: Reduce 512 -> 256 by adding high/low halves (8 ops)
  // Use shuffle within 512-bit to move high 256 to low, then add
  const __m512i idx_hi = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 8);

  __m512 t0 = _mm512_add_ps(a0, _mm512_permutexvar_ps(idx_hi, a0));
  __m512 t1 = _mm512_add_ps(a1, _mm512_permutexvar_ps(idx_hi, a1));
  __m512 t2 = _mm512_add_ps(a2, _mm512_permutexvar_ps(idx_hi, a2));
  __m512 t3 = _mm512_add_ps(a3, _mm512_permutexvar_ps(idx_hi, a3));
  __m512 t4 = _mm512_add_ps(a4, _mm512_permutexvar_ps(idx_hi, a4));
  __m512 t5 = _mm512_add_ps(a5, _mm512_permutexvar_ps(idx_hi, a5));
  __m512 t6 = _mm512_add_ps(a6, _mm512_permutexvar_ps(idx_hi, a6));
  __m512 t7 = _mm512_add_ps(a7, _mm512_permutexvar_ps(idx_hi, a7));

  // Now each t[i] has valid data in low 256 bits (8 floats)
  // Step 2: Pack pairs into single 512-bit vectors
  // t0,t1 -> pack low 256 of t0 and t1 into one 512
  // Use mask blend or shuffle
  __m512 p01 = _mm512_shuffle_f32x4(t0, t1, 0x44);  // [t0_lo, t1_lo, t0_lo, t1_lo] -> need [t0_lo, t1_lo]
  __m512 p23 = _mm512_shuffle_f32x4(t2, t3, 0x44);
  __m512 p45 = _mm512_shuffle_f32x4(t4, t5, 0x44);
  __m512 p67 = _mm512_shuffle_f32x4(t6, t7, 0x44);

  // Step 3: Reduce 256 -> 128 within each pair
  const __m512i idx_128_hi = _mm512_set_epi32(7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4);
  p01 = _mm512_add_ps(p01, _mm512_permutexvar_ps(idx_128_hi, p01));
  p23 = _mm512_add_ps(p23, _mm512_permutexvar_ps(idx_128_hi, p23));
  p45 = _mm512_add_ps(p45, _mm512_permutexvar_ps(idx_128_hi, p45));
  p67 = _mm512_add_ps(p67, _mm512_permutexvar_ps(idx_128_hi, p67));

  // Step 4: Reduce 128 -> 64 -> 32 within each
  // hadd pattern: [a,b,c,d] + [b,a,d,c] with mask
  const __m512i idx_64 = _mm512_set_epi32(3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2);
  p01 = _mm512_add_ps(p01, _mm512_permutexvar_ps(idx_64, p01));
  p23 = _mm512_add_ps(p23, _mm512_permutexvar_ps(idx_64, p23));
  p45 = _mm512_add_ps(p45, _mm512_permutexvar_ps(idx_64, p45));
  p67 = _mm512_add_ps(p67, _mm512_permutexvar_ps(idx_64, p67));

  const __m512i idx_32 = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  p01 = _mm512_add_ps(p01, _mm512_permutexvar_ps(idx_32, p01));
  p23 = _mm512_add_ps(p23, _mm512_permutexvar_ps(idx_32, p23));
  p45 = _mm512_add_ps(p45, _mm512_permutexvar_ps(idx_32, p45));
  p67 = _mm512_add_ps(p67, _mm512_permutexvar_ps(idx_32, p67));

  // Now p01[0] = sum(a0), p01[8] = sum(a1), etc.
  // Extract and pack into __m256
  // p01: [sum0, ?, ?, ?, ?, ?, ?, ?, sum1, ?, ?, ?, ?, ?, ?, ?]
  // p23: [sum2, ?, ?, ?, ?, ?, ?, ?, sum3, ?, ?, ?, ?, ?, ?, ?]

  float s0 = _mm512_cvtss_f32(p01);
  float s1 = _mm_cvtss_f32(_mm512_extractf32x4_ps(p01, 2));
  float s2 = _mm512_cvtss_f32(p23);
  float s3 = _mm_cvtss_f32(_mm512_extractf32x4_ps(p23, 2));
  float s4 = _mm512_cvtss_f32(p45);
  float s5 = _mm_cvtss_f32(_mm512_extractf32x4_ps(p45, 2));
  float s6 = _mm512_cvtss_f32(p67);
  float s7 = _mm_cvtss_f32(_mm512_extractf32x4_ps(p67, 2));

  return _mm256_set_ps(s7, s6, s5, s4, s3, s2, s1, s0);
}

// Simpler approach: reduce in pairs, more parallelism
inline __m256 hsum_8x512_fast(__m512 a0, __m512 a1, __m512 a2, __m512 a3, __m512 a4, __m512 a5, __m512 a6, __m512 a7) {
  // Reduce each 512 to scalar using the built-in, but do 8 in parallel
  // The compiler should pipeline these well
  __m256 result;

  // Use inline asm or let compiler optimize
  // Reduce in pairs to allow more ILP
  __m512 sum01 = _mm512_add_ps(a0, a1);
  __m512 sum23 = _mm512_add_ps(a2, a3);
  __m512 sum45 = _mm512_add_ps(a4, a5);
  __m512 sum67 = _mm512_add_ps(a6, a7);

  // Now reduce each pair
  float r0 = _mm512_reduce_add_ps(a0);
  float r1 = _mm512_reduce_add_ps(a1);
  float r2 = _mm512_reduce_add_ps(a2);
  float r3 = _mm512_reduce_add_ps(a3);
  float r4 = _mm512_reduce_add_ps(a4);
  float r5 = _mm512_reduce_add_ps(a5);
  float r6 = _mm512_reduce_add_ps(a6);
  float r7 = _mm512_reduce_add_ps(a7);

  return _mm256_set_ps(r7, r6, r5, r4, r3, r2, r1, r0);
}

void lora_fused_add_opt8(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;
  constexpr int PREFETCH_DISTANCE = 16;  // Prefetch 16 output rows ahead

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

      // 32 accumulators
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

      // Main loop with software pipelining
      // Prefetch intermediate data for next iteration
      for (; r + 16 <= rank; r += 16) {
        // Load intermediate
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

        // Load weights - interleave loads and FMAs for better pipelining
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

      // Tail handling
      if (tail_mask) {
        __m512 iv0 = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1 = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);
        __m512 iv2 = _mm512_maskz_loadu_ps(tail_mask, inter2 + r);
        __m512 iv3 = _mm512_maskz_loadu_ps(tail_mask, inter3 + r);

#define LOAD_W_MASK(idx) \
  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w##idx + r)), 16))
        __m512 wv0 = LOAD_W_MASK(0);
        __m512 wv1 = LOAD_W_MASK(1);
        __m512 wv2 = LOAD_W_MASK(2);
        __m512 wv3 = LOAD_W_MASK(3);
        __m512 wv4 = LOAD_W_MASK(4);
        __m512 wv5 = LOAD_W_MASK(5);
        __m512 wv6 = LOAD_W_MASK(6);
        __m512 wv7 = LOAD_W_MASK(7);

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

      // Vectorized reduce using hsum_8x512_fast (simpler, lets compiler optimize)
      __m256 sum_t0 =
          hsum_8x512_fast(acc_t0_o0, acc_t0_o1, acc_t0_o2, acc_t0_o3, acc_t0_o4, acc_t0_o5, acc_t0_o6, acc_t0_o7);
      __m256 sum_t1 =
          hsum_8x512_fast(acc_t1_o0, acc_t1_o1, acc_t1_o2, acc_t1_o3, acc_t1_o4, acc_t1_o5, acc_t1_o6, acc_t1_o7);
      __m256 sum_t2 =
          hsum_8x512_fast(acc_t2_o0, acc_t2_o1, acc_t2_o2, acc_t2_o3, acc_t2_o4, acc_t2_o5, acc_t2_o6, acc_t2_o7);
      __m256 sum_t3 =
          hsum_8x512_fast(acc_t3_o0, acc_t3_o1, acc_t3_o2, acc_t3_o3, acc_t3_o4, acc_t3_o5, acc_t3_o6, acc_t3_o7);

      // Scale
      sum_t0 = _mm256_mul_ps(sum_t0, scale_vec);
      sum_t1 = _mm256_mul_ps(sum_t1, scale_vec);
      sum_t2 = _mm256_mul_ps(sum_t2, scale_vec);
      sum_t3 = _mm256_mul_ps(sum_t3, scale_vec);

      // Load output, add, convert, store
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

      __m128bh bf16_t0 = _mm256_cvtneps_pbh(out_t0);
      __m128bh bf16_t1 = _mm256_cvtneps_pbh(out_t1);
      __m128bh bf16_t2 = _mm256_cvtneps_pbh(out_t2);
      __m128bh bf16_t3 = _mm256_cvtneps_pbh(out_t3);

      _mm_storeu_si128((__m128i*)(out0 + i), (__m128i)bf16_t0);
      _mm_storeu_si128((__m128i*)(out1 + i), (__m128i)bf16_t1);
      _mm_storeu_si128((__m128i*)(out2 + i), (__m128i)bf16_t2);
      _mm_storeu_si128((__m128i*)(out3 + i), (__m128i)bf16_t3);
    }

    // Remainder outputs
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

      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);

      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Remaining tokens
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
      float sum = _mm512_reduce_add_ps(acc);
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v9: T_BLOCK=2, O_BLOCK=16 - better weight reuse
// 32 accumulators = 2 tokens × 16 outputs
// ============================================================================
void lora_fused_add_opt9(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 2;
  constexpr int O_BLOCK = 16;

  const int rank_tail = rank & 15;
  const __mmask16 tail_mask = rank_tail ? ((__mmask16)1 << rank_tail) - 1 : 0;

  int t = 0;
  for (; t + T_BLOCK <= num_tokens; t += T_BLOCK) {
    const float* inter0 = intermediate + (t + 0) * rank;
    const float* inter1 = intermediate + (t + 1) * rank;
    ggml_bf16_t* out0 = output + (t + 0) * output_dim;
    ggml_bf16_t* out1 = output + (t + 1) * output_dim;

    int i = 0;
    for (; i + O_BLOCK <= output_dim; i += O_BLOCK) {
      // 32 accumulators: 2 tokens × 16 outputs
      __m512 acc0[16], acc1[16];
      for (int j = 0; j < 16; j++) {
        acc0[j] = _mm512_setzero_ps();
        acc1[j] = _mm512_setzero_ps();
      }

      const ggml_bf16_t* w[16];
      for (int j = 0; j < 16; j++) {
        w[j] = weight + (i + j) * rank;
      }

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);

// Unroll weight loads and FMAs
#pragma unroll
        for (int j = 0; j < 16; j++) {
          __m512 wv = _mm512_castsi512_ps(
              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w[j] + r))), 16));
          acc0[j] = _mm512_fmadd_ps(iv0, wv, acc0[j]);
          acc1[j] = _mm512_fmadd_ps(iv1, wv, acc1[j]);
        }
      }

      if (tail_mask) {
        __m512 iv0 = _mm512_maskz_loadu_ps(tail_mask, inter0 + r);
        __m512 iv1 = _mm512_maskz_loadu_ps(tail_mask, inter1 + r);

#pragma unroll
        for (int j = 0; j < 16; j++) {
          __m512 wv = _mm512_castsi512_ps(
              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w[j] + r)), 16));
          acc0[j] = _mm512_fmadd_ps(iv0, wv, acc0[j]);
          acc1[j] = _mm512_fmadd_ps(iv1, wv, acc1[j]);
        }
      }

      // Reduce and store - use 512-bit for 16 outputs
      // First token
      {
        // Reduce 16 accumulators
        float sums0[16];
#pragma unroll
        for (int j = 0; j < 16; j++) {
          sums0[j] = _mm512_reduce_add_ps(acc0[j]) * scale;
        }

        // Load 16 BF16 outputs, convert, add, convert back, store
        __m512 out_v =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out0 + i))), 16));
        __m512 sum_v = _mm512_loadu_ps(sums0);
        out_v = _mm512_add_ps(out_v, sum_v);

        // Convert FP32 -> BF16 (16 values -> 256 bits)
        __m256bh bf16_out = _mm512_cvtneps_pbh(out_v);
        _mm256_storeu_si256((__m256i*)(out0 + i), (__m256i)bf16_out);
      }

      // Second token
      {
        float sums1[16];
#pragma unroll
        for (int j = 0; j < 16; j++) {
          sums1[j] = _mm512_reduce_add_ps(acc1[j]) * scale;
        }

        __m512 out_v =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(out1 + i))), 16));
        __m512 sum_v = _mm512_loadu_ps(sums1);
        out_v = _mm512_add_ps(out_v, sum_v);

        __m256bh bf16_out = _mm512_cvtneps_pbh(out_v);
        _mm256_storeu_si256((__m256i*)(out1 + i), (__m256i)bf16_out);
      }
    }

    // Remainder outputs
    for (; i < output_dim; i++) {
      const ggml_bf16_t* w_row = weight + i * rank;
      __m512 acc0_v = _mm512_setzero_ps();
      __m512 acc1_v = _mm512_setzero_ps();

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
        acc0_v = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r), wv, acc0_v);
        acc1_v = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r), wv, acc1_v);
      }
      if (tail_mask) {
        __m512 wv = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(tail_mask, w_row + r)), 16));
        acc0_v = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter0 + r), wv, acc0_v);
        acc1_v = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail_mask, inter1 + r), wv, acc1_v);
      }

      float s0 = _mm512_reduce_add_ps(acc0_v);
      float s1 = _mm512_reduce_add_ps(acc1_v);
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
    }
  }

  // Remaining tokens
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
      float sum = _mm512_reduce_add_ps(acc);
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Optimized v10: Pre-transposed weight layout [rank][output_dim]
// This allows contiguous memory access for output dimension in inner loop
// Benefits: Better cache locality, vectorized output accumulation
// T_BLOCK=4, O_BLOCK=16, with 4 accumulators per output (total 64 -> use 2 passes)
// ============================================================================

// Transpose weight from [output_dim][rank] to [rank][output_dim]
void transpose_weight_bf16(const ggml_bf16_t* __restrict weight, ggml_bf16_t* __restrict weight_t, int output_dim,
                           int rank) {
  // Simple transpose: weight[i][r] -> weight_t[r][i]
  for (int r = 0; r < rank; r++) {
    for (int i = 0; i < output_dim; i++) {
      weight_t[r * output_dim + i] = weight[i * rank + r];
    }
  }
}

// Optimized transpose using AVX-512
void transpose_weight_bf16_fast(const ggml_bf16_t* __restrict weight, ggml_bf16_t* __restrict weight_t, int output_dim,
                                int rank) {
  // Process 16x16 blocks for efficient transpose
  constexpr int BLOCK = 16;

  int r = 0;
  for (; r + BLOCK <= rank; r += BLOCK) {
    int i = 0;
    for (; i + BLOCK <= output_dim; i += BLOCK) {
      // Load 16x16 block from weight[i:i+16][r:r+16]
      // and store as weight_t[r:r+16][i:i+16]
      for (int rr = 0; rr < BLOCK; rr++) {
        for (int ii = 0; ii < BLOCK; ii++) {
          weight_t[(r + rr) * output_dim + (i + ii)] = weight[(i + ii) * rank + (r + rr)];
        }
      }
    }
    // Remainder columns
    for (; i < output_dim; i++) {
      for (int rr = 0; rr < BLOCK; rr++) {
        weight_t[(r + rr) * output_dim + i] = weight[i * rank + (r + rr)];
      }
    }
  }
  // Remainder rows
  for (; r < rank; r++) {
    for (int i = 0; i < output_dim; i++) {
      weight_t[r * output_dim + i] = weight[i * rank + r];
    }
  }
}

// Kernel using pre-transposed weights: weight_t[rank][output_dim]
// For each rank position, weights for all outputs are contiguous
void lora_fused_add_opt10(const float* __restrict intermediate,
                          const ggml_bf16_t* __restrict weight_t,  // Transposed: [rank][output_dim]
                          ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 16;  // Process 16 outputs at a time (fits in one AVX-512 register)

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
      // 4 accumulators per token for 16 outputs
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();

      // Inner loop over rank - weights are now contiguous for each rank position
      for (int r = 0; r < rank; r++) {
        // Load 16 consecutive weights for this rank position
        // weight_t[r][i:i+16] - contiguous!
        __m512 wv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(weight_t + r * output_dim + i))), 16));

        // Broadcast intermediate values
        __m512 iv0 = _mm512_set1_ps(inter0[r]);
        __m512 iv1 = _mm512_set1_ps(inter1[r]);
        __m512 iv2 = _mm512_set1_ps(inter2[r]);
        __m512 iv3 = _mm512_set1_ps(inter3[r]);

        // FMA: accumulate weighted contributions
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

      __m256bh bf16_0 = _mm512_cvtneps_pbh(out_v0);
      __m256bh bf16_1 = _mm512_cvtneps_pbh(out_v1);
      __m256bh bf16_2 = _mm512_cvtneps_pbh(out_v2);
      __m256bh bf16_3 = _mm512_cvtneps_pbh(out_v3);

      _mm256_mask_storeu_epi16(out0 + i, output_tail_mask, (__m256i)bf16_0);
      _mm256_mask_storeu_epi16(out1 + i, output_tail_mask, (__m256i)bf16_1);
      _mm256_mask_storeu_epi16(out2 + i, output_tail_mask, (__m256i)bf16_2);
      _mm256_mask_storeu_epi16(out3 + i, output_tail_mask, (__m256i)bf16_3);
    }
  }

  // Remaining tokens
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
}

// ============================================================================
// AMX support detection and initialization
// ============================================================================
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
#define AMX_AVAILABLE_LORA 1
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define XFEATURE_XTILEDATA 18
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

static bool amx_init_lora = false;

bool init_amx_lora() {
  if (amx_init_lora) return true;

  unsigned long bitmask = 0;
  if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) {
    return false;
  }

  if (!(bitmask & (1UL << XFEATURE_XTILEDATA))) {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
      return false;
    }
  }

  amx_init_lora = true;
  return true;
}

// AMX tile configuration structure
struct TileConfigLora {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};

  void set_row_col(int tile, int rows_, int colsb_) {
    rows[tile] = rows_;
    colsb[tile] = colsb_;
  }

  void set_config() { _tile_loadconfig(this); }
};

// Configure AMX for BF16 matmul
// A tile: [16 rows, 32 BF16] = [16, 64 bytes]
// B tile (VNNI): [16 rows, 32 BF16] = [16, 64 bytes]
// C tile: [16 rows, 16 FP32] = [16, 64 bytes]
void configure_amx_lora() {
  TileConfigLora cfg;
  cfg.set_row_col(0, 16, 64);  // A: 16 rows x 64 bytes
  cfg.set_row_col(1, 16, 64);  // B: 16 rows x 64 bytes (VNNI)
  cfg.set_row_col(2, 16, 64);  // C: 16 rows x 64 bytes
  cfg.set_config();
}

#else
#define AMX_AVAILABLE_LORA 0
bool init_amx_lora() { return false; }
void configure_amx_lora() {}
#endif

// ============================================================================
// Pre-pack weight into VNNI format for AMX
// Input: weight_t[rank][output_dim] (transposed BF16)
// Output: weight_vnni - VNNI packed format for direct AMX tile load
//
// VNNI format for AMX BF16:
//   For each output tile (16 outputs), for each rank pair (2 ranks):
//   store [out0_r0, out0_r1, out1_r0, out1_r1, ..., out15_r0, out15_r1]
//
// Layout: [num_output_tiles][padded_rank/2][32] where 32 = 16 outputs * 2 ranks
// ============================================================================
constexpr int AMX_TILE_N = 16;  // outputs per tile
constexpr int AMX_TILE_K = 32;  // rank per tile (padded)

size_t get_vnni_weight_size(int rank, int output_dim) {
  int padded_rank = ((rank + AMX_TILE_K - 1) / AMX_TILE_K) * AMX_TILE_K;
  int num_output_tiles = (output_dim + AMX_TILE_N - 1) / AMX_TILE_N;
  return (size_t)num_output_tiles * (padded_rank / 2) * (AMX_TILE_N * 2);
}

void pack_weight_vnni(const ggml_bf16_t* __restrict weight_t,  // [rank][output_dim]
                      ggml_bf16_t* __restrict weight_vnni, int rank, int output_dim) {
  int padded_rank = ((rank + AMX_TILE_K - 1) / AMX_TILE_K) * AMX_TILE_K;
  int num_output_tiles = (output_dim + AMX_TILE_N - 1) / AMX_TILE_N;

  // Zero initialize for padding
  memset(weight_vnni, 0, get_vnni_weight_size(rank, output_dim) * sizeof(ggml_bf16_t));

  // Pack into VNNI format
  // For each output tile
  for (int ot = 0; ot < num_output_tiles; ot++) {
    int o_begin = ot * AMX_TILE_N;
    int o_end = std::min(o_begin + AMX_TILE_N, output_dim);

    // For each rank pair
    for (int rp = 0; rp < padded_rank / 2; rp++) {
      int r0 = rp * 2;
      int r1 = rp * 2 + 1;

      // Destination: weight_vnni[ot][rp][0..31]
      ggml_bf16_t* dst = weight_vnni + (size_t)ot * (padded_rank / 2) * (AMX_TILE_N * 2) + rp * (AMX_TILE_N * 2);

      // Pack 16 outputs, 2 ranks each
      for (int oi = 0; oi < AMX_TILE_N; oi++) {
        int o = o_begin + oi;
        if (o < output_dim) {
          // weight_t is [rank][output_dim]
          dst[oi * 2 + 0] = (r0 < rank) ? weight_t[r0 * output_dim + o] : ggml_bf16_t{0};
          dst[oi * 2 + 1] = (r1 < rank) ? weight_t[r1 * output_dim + o] : ggml_bf16_t{0};
        } else {
          dst[oi * 2 + 0] = ggml_bf16_t{0};
          dst[oi * 2 + 1] = ggml_bf16_t{0};
        }
      }
    }
  }
}

// ============================================================================
// Optimized v11: AMX BF16 with pre-packed VNNI weights
// weight_vnni: pre-packed in VNNI format
// ============================================================================
#if AMX_AVAILABLE_LORA
void lora_fused_add_opt11_amx(const float* __restrict intermediate,
                              const ggml_bf16_t* __restrict weight_vnni,  // Pre-packed VNNI format
                              ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int TILE_M = 16;  // tokens per tile
  constexpr int TILE_K = 32;  // rank per tile
  constexpr int TILE_N = 16;  // outputs per tile

  int padded_rank = ((rank + TILE_K - 1) / TILE_K) * TILE_K;
  int num_output_tiles = (output_dim + TILE_N - 1) / TILE_N;
  size_t vnni_tile_stride = (size_t)(padded_rank / 2) * (TILE_N * 2);

  // Temporary buffers (aligned)
  alignas(64) ggml_bf16_t tile_a[TILE_M * TILE_K];
  alignas(64) float tile_c[TILE_M * TILE_N];

  const __m512 scale_vec = _mm512_set1_ps(scale);

  // Process tokens in blocks of TILE_M
  for (int t_begin = 0; t_begin < num_tokens; t_begin += TILE_M) {
    int t_end = std::min(t_begin + TILE_M, num_tokens);
    int t_count = t_end - t_begin;

    // Process output tiles
    for (int ot = 0; ot < num_output_tiles; ot++) {
      int o_begin = ot * TILE_N;
      int o_end = std::min(o_begin + TILE_N, output_dim);
      int o_count = o_end - o_begin;

      // Zero the C tile
      _tile_zero(2);

      // Pointer to VNNI weight for this output tile
      const ggml_bf16_t* weight_tile = weight_vnni + ot * vnni_tile_stride;

      // Accumulate over rank dimension
      for (int r_begin = 0; r_begin < padded_rank; r_begin += TILE_K) {
        int r_end = std::min(r_begin + TILE_K, padded_rank);
        int actual_r_end = std::min(r_end, rank);

        // Pack A tile: convert intermediate from FP32 to BF16
        memset(tile_a, 0, sizeof(tile_a));
        for (int ti = 0; ti < t_count; ti++) {
          for (int ri = 0; ri < actual_r_end - r_begin; ri++) {
            tile_a[ti * TILE_K + ri] = GGML_FP32_TO_BF16(intermediate[(t_begin + ti) * rank + r_begin + ri]);
          }
        }

        // B tile is already in VNNI format - load directly
        // weight_tile[r_begin/2 * 32 ... ]
        const ggml_bf16_t* b_ptr = weight_tile + (r_begin / 2) * (TILE_N * 2);

        _tile_loadd(0, tile_a, TILE_K * sizeof(ggml_bf16_t));
        _tile_loadd(1, b_ptr, TILE_N * 2 * sizeof(ggml_bf16_t));
        _tile_dpbf16ps(2, 0, 1);
      }

      // Store C tile
      _tile_stored(2, tile_c, TILE_N * sizeof(float));

      // Apply scale and accumulate to output
      for (int ti = 0; ti < t_count; ti++) {
        int t_idx = t_begin + ti;

        if (o_count == TILE_N) {
          __m512 result = _mm512_loadu_ps(&tile_c[ti * TILE_N]);
          result = _mm512_mul_ps(result, scale_vec);

          __m512 out_fp32 = _mm512_castsi512_ps(_mm512_slli_epi32(
              _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(output + t_idx * output_dim + o_begin))), 16));
          out_fp32 = _mm512_add_ps(out_fp32, result);
          __m256bh out_bf16 = _mm512_cvtneps_pbh(out_fp32);
          _mm256_storeu_si256((__m256i*)(output + t_idx * output_dim + o_begin), (__m256i)out_bf16);
        } else {
          for (int oi = 0; oi < o_count; oi++) {
            float result = tile_c[ti * TILE_N + oi] * scale;
            float out_val = GGML_BF16_TO_FP32(output[t_idx * output_dim + o_begin + oi]);
            output[t_idx * output_dim + o_begin + oi] = GGML_FP32_TO_BF16(out_val + result);
          }
        }
      }
    }
  }
}
#else
void lora_fused_add_opt11_amx(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight_t,
                              ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  // Fallback to opt10 when AMX not available
  lora_fused_add_opt10(intermediate, weight_t, output, num_tokens, rank, output_dim, scale);
}
#endif

// ============================================================================
// FP32 weight version for comparison (no BF16 conversion overhead)
// ============================================================================
void lora_fused_add_fp32_weight(const float* __restrict intermediate,
                                const float* __restrict weight_fp32,  // FP32 weight instead of BF16
                                ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 4;

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
      const float* w0 = weight_fp32 + (i + 0) * rank;
      const float* w1 = weight_fp32 + (i + 1) * rank;
      const float* w2 = weight_fp32 + (i + 2) * rank;
      const float* w3 = weight_fp32 + (i + 3) * rank;

      __m512 acc_t0_o0 = _mm512_setzero_ps(), acc_t0_o1 = _mm512_setzero_ps();
      __m512 acc_t0_o2 = _mm512_setzero_ps(), acc_t0_o3 = _mm512_setzero_ps();
      __m512 acc_t1_o0 = _mm512_setzero_ps(), acc_t1_o1 = _mm512_setzero_ps();
      __m512 acc_t1_o2 = _mm512_setzero_ps(), acc_t1_o3 = _mm512_setzero_ps();
      __m512 acc_t2_o0 = _mm512_setzero_ps(), acc_t2_o1 = _mm512_setzero_ps();
      __m512 acc_t2_o2 = _mm512_setzero_ps(), acc_t2_o3 = _mm512_setzero_ps();
      __m512 acc_t3_o0 = _mm512_setzero_ps(), acc_t3_o1 = _mm512_setzero_ps();
      __m512 acc_t3_o2 = _mm512_setzero_ps(), acc_t3_o3 = _mm512_setzero_ps();

      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        // Direct FP32 load - no conversion needed!
        __m512 wv0 = _mm512_loadu_ps(w0 + r);
        __m512 wv1 = _mm512_loadu_ps(w1 + r);
        __m512 wv2 = _mm512_loadu_ps(w2 + r);
        __m512 wv3 = _mm512_loadu_ps(w3 + r);

        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);
        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);
        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);
        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
      }

      float s_t0_o0 = _mm512_reduce_add_ps(acc_t0_o0);
      float s_t0_o1 = _mm512_reduce_add_ps(acc_t0_o1);
      float s_t0_o2 = _mm512_reduce_add_ps(acc_t0_o2);
      float s_t0_o3 = _mm512_reduce_add_ps(acc_t0_o3);
      float s_t1_o0 = _mm512_reduce_add_ps(acc_t1_o0);
      float s_t1_o1 = _mm512_reduce_add_ps(acc_t1_o1);
      float s_t1_o2 = _mm512_reduce_add_ps(acc_t1_o2);
      float s_t1_o3 = _mm512_reduce_add_ps(acc_t1_o3);
      float s_t2_o0 = _mm512_reduce_add_ps(acc_t2_o0);
      float s_t2_o1 = _mm512_reduce_add_ps(acc_t2_o1);
      float s_t2_o2 = _mm512_reduce_add_ps(acc_t2_o2);
      float s_t2_o3 = _mm512_reduce_add_ps(acc_t2_o3);
      float s_t3_o0 = _mm512_reduce_add_ps(acc_t3_o0);
      float s_t3_o1 = _mm512_reduce_add_ps(acc_t3_o1);
      float s_t3_o2 = _mm512_reduce_add_ps(acc_t3_o2);
      float s_t3_o3 = _mm512_reduce_add_ps(acc_t3_o3);

      for (; r < rank; r++) {
        float w0v = w0[r], w1v = w1[r], w2v = w2[r], w3v = w3[r];
        s_t0_o0 += inter0[r] * w0v;
        s_t0_o1 += inter0[r] * w1v;
        s_t0_o2 += inter0[r] * w2v;
        s_t0_o3 += inter0[r] * w3v;
        s_t1_o0 += inter1[r] * w0v;
        s_t1_o1 += inter1[r] * w1v;
        s_t1_o2 += inter1[r] * w2v;
        s_t1_o3 += inter1[r] * w3v;
        s_t2_o0 += inter2[r] * w0v;
        s_t2_o1 += inter2[r] * w1v;
        s_t2_o2 += inter2[r] * w2v;
        s_t2_o3 += inter2[r] * w3v;
        s_t3_o0 += inter3[r] * w0v;
        s_t3_o1 += inter3[r] * w1v;
        s_t3_o2 += inter3[r] * w2v;
        s_t3_o3 += inter3[r] * w3v;
      }

      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
    }

    // Remainder outputs
    for (; i < output_dim; i++) {
      const float* w_row = weight_fp32 + i * rank;
      __m512 acc0 = _mm512_setzero_ps();
      __m512 acc1 = _mm512_setzero_ps();
      __m512 acc2 = _mm512_setzero_ps();
      __m512 acc3 = _mm512_setzero_ps();
      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_loadu_ps(w_row + r);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(inter0 + r), wv, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(inter1 + r), wv, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(inter2 + r), wv, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(inter3 + r), wv, acc3);
      }
      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);
      for (; r < rank; r++) {
        float wv = w_row[r];
        s0 += inter0[r] * wv;
        s1 += inter1[r] * wv;
        s2 += inter2[r] * wv;
        s3 += inter3[r] * wv;
      }
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Remainder tokens
  for (; t < num_tokens; t++) {
    const float* inter_row = intermediate + t * rank;
    ggml_bf16_t* out_row = output + t * output_dim;
    for (int i = 0; i < output_dim; i++) {
      const float* w_row = weight_fp32 + i * rank;
      __m512 acc = _mm512_setzero_ps();
      int r = 0;
      for (; r + 16 <= rank; r += 16) {
        __m512 wv = _mm512_loadu_ps(w_row + r);
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r), wv, acc);
      }
      float sum = _mm512_reduce_add_ps(acc);
      for (; r < rank; r++) {
        sum += inter_row[r] * w_row[r];
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Test utilities
// ============================================================================
void init_random_bf16(ggml_bf16_t* buf, size_t size, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < size; i++) {
    buf[i] = GGML_FP32_TO_BF16(dist(gen));
  }
}

void init_random_fp32(float* buf, size_t size, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < size; i++) {
    buf[i] = dist(gen);
  }
}

bool compare_bf16_buffers(const ggml_bf16_t* a, const ggml_bf16_t* b, size_t size, float rtol = 1e-2f,
                          float atol = 1e-2f) {
  int mismatch_count = 0;
  float max_diff = 0.0f;
  for (size_t i = 0; i < size; i++) {
    float va = GGML_BF16_TO_FP32(a[i]);
    float vb = GGML_BF16_TO_FP32(b[i]);
    float diff = std::fabs(va - vb);
    float tol = atol + rtol * std::fabs(vb);
    if (diff > tol) {
      if (mismatch_count < 5) {
        printf("  Mismatch at %zu: ref=%.6f got=%.6f diff=%.6f\n", i, vb, va, diff);
      }
      mismatch_count++;
    }
    max_diff = std::max(max_diff, diff);
  }
  if (mismatch_count > 0) {
    printf("  Total mismatches: %d / %zu, max_diff: %.6f\n", mismatch_count, size, max_diff);
    return false;
  }
  return true;
}

// ============================================================================
// Optimized v3: Output tiling for better cache locality
// Process output in tiles to keep working set in L2 cache
// ============================================================================
void lora_fused_add_opt3(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 8;   // Process 8 outputs at a time for better register utilization
  constexpr int O_TILE = 256;  // Tile output dimension for cache locality

  // Process output in tiles
  for (int o_tile = 0; o_tile < output_dim; o_tile += O_TILE) {
    int o_tile_end = std::min(o_tile + O_TILE, output_dim);

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

      int i = o_tile;
      for (; i + O_BLOCK <= o_tile_end; i += O_BLOCK) {
        // 32 accumulators: 4 tokens × 8 outputs
        __m512 acc_t0[8], acc_t1[8], acc_t2[8], acc_t3[8];
        for (int j = 0; j < 8; j++) {
          acc_t0[j] = _mm512_setzero_ps();
          acc_t1[j] = _mm512_setzero_ps();
          acc_t2[j] = _mm512_setzero_ps();
          acc_t3[j] = _mm512_setzero_ps();
        }

        const ggml_bf16_t* w[8];
        for (int j = 0; j < 8; j++) {
          w[j] = weight + (i + j) * rank;
        }

        int r = 0;
        for (; r + 16 <= rank; r += 16) {
          // Load weights (8 rows × 16 elements)
          __m512 wv[8];
          for (int j = 0; j < 8; j++) {
            wv[j] = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w[j] + r))), 16));
          }

          // Load intermediate (4 tokens)
          __m512 iv0 = _mm512_loadu_ps(inter0 + r);
          __m512 iv1 = _mm512_loadu_ps(inter1 + r);
          __m512 iv2 = _mm512_loadu_ps(inter2 + r);
          __m512 iv3 = _mm512_loadu_ps(inter3 + r);

          // Accumulate
          for (int j = 0; j < 8; j++) {
            acc_t0[j] = _mm512_fmadd_ps(iv0, wv[j], acc_t0[j]);
            acc_t1[j] = _mm512_fmadd_ps(iv1, wv[j], acc_t1[j]);
            acc_t2[j] = _mm512_fmadd_ps(iv2, wv[j], acc_t2[j]);
            acc_t3[j] = _mm512_fmadd_ps(iv3, wv[j], acc_t3[j]);
          }
        }

        // Reduce and store
        for (int j = 0; j < 8; j++) {
          float s0 = _mm512_reduce_add_ps(acc_t0[j]);
          float s1 = _mm512_reduce_add_ps(acc_t1[j]);
          float s2 = _mm512_reduce_add_ps(acc_t2[j]);
          float s3 = _mm512_reduce_add_ps(acc_t3[j]);

          // Scalar tail
          for (int rr = r; rr < rank; rr++) {
            float wv = GGML_BF16_TO_FP32(w[j][rr]);
            s0 += inter0[rr] * wv;
            s1 += inter1[rr] * wv;
            s2 += inter2[rr] * wv;
            s3 += inter3[rr] * wv;
          }

          out0[i + j] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + j]) + s0 * scale);
          out1[i + j] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + j]) + s1 * scale);
          out2[i + j] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + j]) + s2 * scale);
          out3[i + j] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + j]) + s3 * scale);
        }
      }

      // Remainder outputs in tile
      for (; i < o_tile_end; i++) {
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
        float s0 = _mm512_reduce_add_ps(acc0);
        float s1 = _mm512_reduce_add_ps(acc1);
        float s2 = _mm512_reduce_add_ps(acc2);
        float s3 = _mm512_reduce_add_ps(acc3);
        for (; r < rank; r++) {
          float wv = GGML_BF16_TO_FP32(w_row[r]);
          s0 += inter0[r] * wv;
          s1 += inter1[r] * wv;
          s2 += inter2[r] * wv;
          s3 += inter3[r] * wv;
        }
        out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
        out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
        out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
        out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
      }
    }

    // Remainder tokens
    for (; t < num_tokens; t++) {
      const float* inter_row = intermediate + t * rank;
      ggml_bf16_t* out_row = output + t * output_dim;
      for (int i = o_tile; i < o_tile_end; i++) {
        const ggml_bf16_t* w_row = weight + i * rank;
        __m512 acc = _mm512_setzero_ps();
        int r = 0;
        for (; r + 16 <= rank; r += 16) {
          __m512 wv = _mm512_castsi512_ps(
              _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w_row + r))), 16));
          acc = _mm512_fmadd_ps(_mm512_loadu_ps(inter_row + r), wv, acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; r < rank; r++) {
          sum += inter_row[r] * GGML_BF16_TO_FP32(w_row[r]);
        }
        out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
      }
    }
  }
}

// ============================================================================
// Optimized v4: Full unroll with explicit registers + prefetching
// ============================================================================
void lora_fused_add_opt4(const float* __restrict intermediate, const ggml_bf16_t* __restrict weight,
                         ggml_bf16_t* __restrict output, int num_tokens, int rank, int output_dim, float scale) {
  constexpr int T_BLOCK = 4;
  constexpr int O_BLOCK = 4;

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
      const ggml_bf16_t* w0 = weight + (i + 0) * rank;
      const ggml_bf16_t* w1 = weight + (i + 1) * rank;
      const ggml_bf16_t* w2 = weight + (i + 2) * rank;
      const ggml_bf16_t* w3 = weight + (i + 3) * rank;

      // Prefetch next weight rows
      if (i + O_BLOCK < output_dim) {
        _mm_prefetch((const char*)(weight + (i + O_BLOCK + 0) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + O_BLOCK + 1) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + O_BLOCK + 2) * rank), _MM_HINT_T0);
        _mm_prefetch((const char*)(weight + (i + O_BLOCK + 3) * rank), _MM_HINT_T0);
      }

      // 16 accumulators fully unrolled
      __m512 acc_t0_o0 = _mm512_setzero_ps(), acc_t0_o1 = _mm512_setzero_ps();
      __m512 acc_t0_o2 = _mm512_setzero_ps(), acc_t0_o3 = _mm512_setzero_ps();
      __m512 acc_t1_o0 = _mm512_setzero_ps(), acc_t1_o1 = _mm512_setzero_ps();
      __m512 acc_t1_o2 = _mm512_setzero_ps(), acc_t1_o3 = _mm512_setzero_ps();
      __m512 acc_t2_o0 = _mm512_setzero_ps(), acc_t2_o1 = _mm512_setzero_ps();
      __m512 acc_t2_o2 = _mm512_setzero_ps(), acc_t2_o3 = _mm512_setzero_ps();
      __m512 acc_t3_o0 = _mm512_setzero_ps(), acc_t3_o1 = _mm512_setzero_ps();
      __m512 acc_t3_o2 = _mm512_setzero_ps(), acc_t3_o3 = _mm512_setzero_ps();

      int r = 0;
      // Unroll by 2 in rank dimension
      for (; r + 32 <= rank; r += 32) {
        // First 16 elements
        __m512 wv0_a =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r))), 16));
        __m512 wv1_a =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r))), 16));
        __m512 wv2_a =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r))), 16));
        __m512 wv3_a =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r))), 16));

        __m512 iv0_a = _mm512_loadu_ps(inter0 + r);
        __m512 iv1_a = _mm512_loadu_ps(inter1 + r);
        __m512 iv2_a = _mm512_loadu_ps(inter2 + r);
        __m512 iv3_a = _mm512_loadu_ps(inter3 + r);

        acc_t0_o0 = _mm512_fmadd_ps(iv0_a, wv0_a, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0_a, wv1_a, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0_a, wv2_a, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0_a, wv3_a, acc_t0_o3);
        acc_t1_o0 = _mm512_fmadd_ps(iv1_a, wv0_a, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1_a, wv1_a, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1_a, wv2_a, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1_a, wv3_a, acc_t1_o3);
        acc_t2_o0 = _mm512_fmadd_ps(iv2_a, wv0_a, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2_a, wv1_a, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2_a, wv2_a, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2_a, wv3_a, acc_t2_o3);
        acc_t3_o0 = _mm512_fmadd_ps(iv3_a, wv0_a, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3_a, wv1_a, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3_a, wv2_a, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3_a, wv3_a, acc_t3_o3);

        // Second 16 elements
        __m512 wv0_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r + 16))), 16));
        __m512 wv1_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r + 16))), 16));
        __m512 wv2_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r + 16))), 16));
        __m512 wv3_b = _mm512_castsi512_ps(
            _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r + 16))), 16));

        __m512 iv0_b = _mm512_loadu_ps(inter0 + r + 16);
        __m512 iv1_b = _mm512_loadu_ps(inter1 + r + 16);
        __m512 iv2_b = _mm512_loadu_ps(inter2 + r + 16);
        __m512 iv3_b = _mm512_loadu_ps(inter3 + r + 16);

        acc_t0_o0 = _mm512_fmadd_ps(iv0_b, wv0_b, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0_b, wv1_b, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0_b, wv2_b, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0_b, wv3_b, acc_t0_o3);
        acc_t1_o0 = _mm512_fmadd_ps(iv1_b, wv0_b, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1_b, wv1_b, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1_b, wv2_b, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1_b, wv3_b, acc_t1_o3);
        acc_t2_o0 = _mm512_fmadd_ps(iv2_b, wv0_b, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2_b, wv1_b, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2_b, wv2_b, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2_b, wv3_b, acc_t2_o3);
        acc_t3_o0 = _mm512_fmadd_ps(iv3_b, wv0_b, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3_b, wv1_b, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3_b, wv2_b, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3_b, wv3_b, acc_t3_o3);
      }

      // Handle remaining 16-element chunk
      for (; r + 16 <= rank; r += 16) {
        __m512 wv0 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w0 + r))), 16));
        __m512 wv1 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w1 + r))), 16));
        __m512 wv2 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w2 + r))), 16));
        __m512 wv3 =
            _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)(w3 + r))), 16));

        __m512 iv0 = _mm512_loadu_ps(inter0 + r);
        __m512 iv1 = _mm512_loadu_ps(inter1 + r);
        __m512 iv2 = _mm512_loadu_ps(inter2 + r);
        __m512 iv3 = _mm512_loadu_ps(inter3 + r);

        acc_t0_o0 = _mm512_fmadd_ps(iv0, wv0, acc_t0_o0);
        acc_t0_o1 = _mm512_fmadd_ps(iv0, wv1, acc_t0_o1);
        acc_t0_o2 = _mm512_fmadd_ps(iv0, wv2, acc_t0_o2);
        acc_t0_o3 = _mm512_fmadd_ps(iv0, wv3, acc_t0_o3);
        acc_t1_o0 = _mm512_fmadd_ps(iv1, wv0, acc_t1_o0);
        acc_t1_o1 = _mm512_fmadd_ps(iv1, wv1, acc_t1_o1);
        acc_t1_o2 = _mm512_fmadd_ps(iv1, wv2, acc_t1_o2);
        acc_t1_o3 = _mm512_fmadd_ps(iv1, wv3, acc_t1_o3);
        acc_t2_o0 = _mm512_fmadd_ps(iv2, wv0, acc_t2_o0);
        acc_t2_o1 = _mm512_fmadd_ps(iv2, wv1, acc_t2_o1);
        acc_t2_o2 = _mm512_fmadd_ps(iv2, wv2, acc_t2_o2);
        acc_t2_o3 = _mm512_fmadd_ps(iv2, wv3, acc_t2_o3);
        acc_t3_o0 = _mm512_fmadd_ps(iv3, wv0, acc_t3_o0);
        acc_t3_o1 = _mm512_fmadd_ps(iv3, wv1, acc_t3_o1);
        acc_t3_o2 = _mm512_fmadd_ps(iv3, wv2, acc_t3_o2);
        acc_t3_o3 = _mm512_fmadd_ps(iv3, wv3, acc_t3_o3);
      }

      // Reduce
      float s_t0_o0 = _mm512_reduce_add_ps(acc_t0_o0);
      float s_t0_o1 = _mm512_reduce_add_ps(acc_t0_o1);
      float s_t0_o2 = _mm512_reduce_add_ps(acc_t0_o2);
      float s_t0_o3 = _mm512_reduce_add_ps(acc_t0_o3);
      float s_t1_o0 = _mm512_reduce_add_ps(acc_t1_o0);
      float s_t1_o1 = _mm512_reduce_add_ps(acc_t1_o1);
      float s_t1_o2 = _mm512_reduce_add_ps(acc_t1_o2);
      float s_t1_o3 = _mm512_reduce_add_ps(acc_t1_o3);
      float s_t2_o0 = _mm512_reduce_add_ps(acc_t2_o0);
      float s_t2_o1 = _mm512_reduce_add_ps(acc_t2_o1);
      float s_t2_o2 = _mm512_reduce_add_ps(acc_t2_o2);
      float s_t2_o3 = _mm512_reduce_add_ps(acc_t2_o3);
      float s_t3_o0 = _mm512_reduce_add_ps(acc_t3_o0);
      float s_t3_o1 = _mm512_reduce_add_ps(acc_t3_o1);
      float s_t3_o2 = _mm512_reduce_add_ps(acc_t3_o2);
      float s_t3_o3 = _mm512_reduce_add_ps(acc_t3_o3);

      // Scalar tail
      for (; r < rank; r++) {
        float w0v = GGML_BF16_TO_FP32(w0[r]);
        float w1v = GGML_BF16_TO_FP32(w1[r]);
        float w2v = GGML_BF16_TO_FP32(w2[r]);
        float w3v = GGML_BF16_TO_FP32(w3[r]);
        s_t0_o0 += inter0[r] * w0v;
        s_t0_o1 += inter0[r] * w1v;
        s_t0_o2 += inter0[r] * w2v;
        s_t0_o3 += inter0[r] * w3v;
        s_t1_o0 += inter1[r] * w0v;
        s_t1_o1 += inter1[r] * w1v;
        s_t1_o2 += inter1[r] * w2v;
        s_t1_o3 += inter1[r] * w3v;
        s_t2_o0 += inter2[r] * w0v;
        s_t2_o1 += inter2[r] * w1v;
        s_t2_o2 += inter2[r] * w2v;
        s_t2_o3 += inter2[r] * w3v;
        s_t3_o0 += inter3[r] * w0v;
        s_t3_o1 += inter3[r] * w1v;
        s_t3_o2 += inter3[r] * w2v;
        s_t3_o3 += inter3[r] * w3v;
      }

      // Store
      out0[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 0]) + s_t0_o0 * scale);
      out0[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 1]) + s_t0_o1 * scale);
      out0[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 2]) + s_t0_o2 * scale);
      out0[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i + 3]) + s_t0_o3 * scale);
      out1[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 0]) + s_t1_o0 * scale);
      out1[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 1]) + s_t1_o1 * scale);
      out1[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 2]) + s_t1_o2 * scale);
      out1[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i + 3]) + s_t1_o3 * scale);
      out2[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 0]) + s_t2_o0 * scale);
      out2[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 1]) + s_t2_o1 * scale);
      out2[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 2]) + s_t2_o2 * scale);
      out2[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i + 3]) + s_t2_o3 * scale);
      out3[i + 0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 0]) + s_t3_o0 * scale);
      out3[i + 1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 1]) + s_t3_o1 * scale);
      out3[i + 2] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 2]) + s_t3_o2 * scale);
      out3[i + 3] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i + 3]) + s_t3_o3 * scale);
    }

    // Remainder outputs
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
      float s0 = _mm512_reduce_add_ps(acc0);
      float s1 = _mm512_reduce_add_ps(acc1);
      float s2 = _mm512_reduce_add_ps(acc2);
      float s3 = _mm512_reduce_add_ps(acc3);
      for (; r < rank; r++) {
        float wv = GGML_BF16_TO_FP32(w_row[r]);
        s0 += inter0[r] * wv;
        s1 += inter1[r] * wv;
        s2 += inter2[r] * wv;
        s3 += inter3[r] * wv;
      }
      out0[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out0[i]) + s0 * scale);
      out1[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out1[i]) + s1 * scale);
      out2[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out2[i]) + s2 * scale);
      out3[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out3[i]) + s3 * scale);
    }
  }

  // Handle remaining tokens
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
      float sum = _mm512_reduce_add_ps(acc);
      for (; r < rank; r++) {
        sum += inter_row[r] * GGML_BF16_TO_FP32(w_row[r]);
      }
      out_row[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_row[i]) + sum * scale);
    }
  }
}

// ============================================================================
// Print usage
// ============================================================================
void print_usage(const char* prog) {
  printf("Usage: %s [options]\n", prog);
  printf("Options:\n");
  printf("  --impl <name>    Implementation to use: current, opt1, opt2, opt3, opt4, opt5, opt6 (default: all)\n");
  printf("  --tokens <n>     Number of tokens (default: 128)\n");
  printf("  --rank <n>       Rank (default: 8)\n");
  printf("  --output <n>     Output dimension (default: 14336)\n");
  printf("  --iters <n>      Number of iterations for profiling (default: 100)\n");
  printf("  --profile        Run in profile mode (single impl, many iterations)\n");
  printf("  --help           Print this help\n");
  printf("\nExamples:\n");
  printf("  %s                                    # Run all tests\n", prog);
  printf("  %s --profile --impl opt6              # Profile opt6 with default params\n", prog);
  printf("  %s --profile --impl opt6 --rank 64    # Profile opt6 with rank=64\n", prog);
  printf("  vtune -collect hotspots -- %s --profile --impl opt6\n", prog);
}

// ============================================================================
// Main test
// ============================================================================
int main(int argc, char** argv) {
  // Parse command line arguments
  std::string impl_name = "all";
  int num_tokens = 128;
  int rank = 8;
  int output_dim = 14336;
  int profile_iters = 100;
  bool profile_mode = false;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--impl" && i + 1 < argc) {
      impl_name = argv[++i];
    } else if (arg == "--tokens" && i + 1 < argc) {
      num_tokens = std::atoi(argv[++i]);
    } else if (arg == "--rank" && i + 1 < argc) {
      rank = std::atoi(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_dim = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      profile_iters = std::atoi(argv[++i]);
    } else if (arg == "--profile") {
      profile_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  // Profile mode: run single implementation many times
  if (profile_mode) {
    printf("=== Profile Mode ===\n");
    printf("Implementation: %s\n", impl_name.c_str());
    printf("Tokens: %d, Rank: %d, Output: %d\n", num_tokens, rank, output_dim);
    printf("Iterations: %d\n\n", profile_iters);

    size_t inter_size = num_tokens * rank;
    size_t weight_size = output_dim * rank;
    size_t output_size = num_tokens * output_dim;

    std::mt19937 gen(42);
    float* intermediate = (float*)aligned_alloc(64, inter_size * sizeof(float));
    ggml_bf16_t* weight = (ggml_bf16_t*)aligned_alloc(64, weight_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_init = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));

    init_random_fp32(intermediate, inter_size, gen);
    init_random_bf16(weight, weight_size, gen);
    init_random_bf16(output_init, output_size, gen);

    float scale = 0.5f;

    // Select implementation
    using KernelFn = void (*)(const float*, const ggml_bf16_t*, ggml_bf16_t*, int, int, int, float);
    KernelFn kernel = nullptr;

    if (impl_name == "current")
      kernel = lora_fused_add_current;
    else if (impl_name == "opt1")
      kernel = lora_fused_add_opt1;
    else if (impl_name == "opt2")
      kernel = lora_fused_add_opt2;
    else if (impl_name == "opt3")
      kernel = lora_fused_add_opt3;
    else if (impl_name == "opt4")
      kernel = lora_fused_add_opt4;
    else if (impl_name == "opt5")
      kernel = lora_fused_add_opt5;
    else if (impl_name == "opt6")
      kernel = lora_fused_add_opt6;
    else if (impl_name == "opt7")
      kernel = lora_fused_add_opt7;
    else if (impl_name == "opt8")
      kernel = lora_fused_add_opt8;
    else if (impl_name == "opt9")
      kernel = lora_fused_add_opt9;
    else {
      printf("Unknown implementation: %s\n", impl_name.c_str());
      printf("Available: current, opt1, opt2, opt3, opt4, opt5, opt6, opt7, opt8, opt9\n");
      return 1;
    }

    // Warmup
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
      memcpy(output, output_init, output_size * sizeof(ggml_bf16_t));
      kernel(intermediate, weight, output, num_tokens, rank, output_dim, scale);
    }

    // Profile run
    printf("Running %d iterations for profiling...\n", profile_iters);
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < profile_iters; i++) {
      memcpy(output, output_init, output_size * sizeof(ggml_bf16_t));
      kernel(intermediate, weight, output, num_tokens, rank, output_dim, scale);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = elapsed_ms / profile_iters;
    double flops = 2.0 * num_tokens * rank * output_dim;
    double gflops = (flops / 1e9) / (avg_ms / 1000.0);

    printf("\nResults:\n");
    printf("  Total time: %.2f ms\n", elapsed_ms);
    printf("  Avg per iter: %.3f ms\n", avg_ms);
    printf("  Performance: %.1f GFLOPS\n", gflops);

    free(intermediate);
    free(weight);
    free(output);
    free(output_init);
    return 0;
  }

  // Normal test mode
  printf("=== lora_fp32_bf16_fused_add Unit Test ===\n\n");

  std::mt19937 gen(42);

  // Test configurations: {num_tokens, rank, output_dim}
  struct TestConfig {
    int num_tokens;
    int rank;
    int output_dim;
  };

  std::vector<TestConfig> configs = {
      {1, 8, 14336},     // Single token, typical LoRA
      {4, 8, 14336},     // Small batch
      {32, 8, 14336},    // Medium batch
      {128, 8, 14336},   // Large batch
      {256, 8, 14336},   // Very large batch
      {128, 16, 14336},  // Larger rank
      {128, 32, 14336},  // Even larger rank
      {128, 64, 14336},  // Max typical rank
      {128, 8, 7168},    // Smaller output (down projection)
  };

  float scale = 0.5f;

  for (const auto& cfg : configs) {
    printf("Testing T=%d, R=%d, O=%d\n", cfg.num_tokens, cfg.rank, cfg.output_dim);

    size_t inter_size = cfg.num_tokens * cfg.rank;
    size_t weight_size = cfg.output_dim * cfg.rank;
    size_t output_size = cfg.num_tokens * cfg.output_dim;

    // Allocate aligned buffers
    float* intermediate = (float*)aligned_alloc(64, inter_size * sizeof(float));
    ggml_bf16_t* weight = (ggml_bf16_t*)aligned_alloc(64, weight_size * sizeof(ggml_bf16_t));
    float* weight_fp32 = (float*)aligned_alloc(64, weight_size * sizeof(float));  // FP32 weight for comparison
    ggml_bf16_t* output_ref = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_cur = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt1 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt2 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt3 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt4 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt5 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt6 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt7 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt8 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt9 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt10 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* output_opt11 = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));
    ggml_bf16_t* weight_t = (ggml_bf16_t*)aligned_alloc(64, weight_size * sizeof(ggml_bf16_t));  // Transposed weight
    size_t vnni_size = get_vnni_weight_size(cfg.rank, cfg.output_dim);
    ggml_bf16_t* weight_vnni = (ggml_bf16_t*)aligned_alloc(64, vnni_size * sizeof(ggml_bf16_t));  // VNNI packed weight
    ggml_bf16_t* output_fp32w =
        (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));  // For FP32 weight test
    ggml_bf16_t* output_init = (ggml_bf16_t*)aligned_alloc(64, output_size * sizeof(ggml_bf16_t));

    // Initialize data
    init_random_fp32(intermediate, inter_size, gen);
    init_random_bf16(weight, weight_size, gen);
    init_random_bf16(output_init, output_size, gen);

    // Transpose weight for opt10: [output_dim][rank] -> [rank][output_dim]
    transpose_weight_bf16_fast(weight, weight_t, cfg.output_dim, cfg.rank);

    // Pack weight into VNNI format for AMX opt11
    pack_weight_vnni(weight_t, weight_vnni, cfg.rank, cfg.output_dim);

    // Convert BF16 weights to FP32 for comparison test
    for (size_t i = 0; i < weight_size; i++) {
      weight_fp32[i] = GGML_BF16_TO_FP32(weight[i]);
    }

    // Copy initial output for each test
    memcpy(output_ref, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_cur, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt1, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt2, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt3, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt4, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt5, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt6, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt7, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt8, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt9, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt10, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_opt11, output_init, output_size * sizeof(ggml_bf16_t));
    memcpy(output_fp32w, output_init, output_size * sizeof(ggml_bf16_t));

    // Run reference
    lora_fused_add_reference(intermediate, weight, output_ref, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);

    // Test current implementation
    lora_fused_add_current(intermediate, weight, output_cur, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool cur_ok = compare_bf16_buffers(output_cur, output_ref, output_size);
    printf("  current: %s\n", cur_ok ? "PASS" : "FAIL");

    // Test opt1
    lora_fused_add_opt1(intermediate, weight, output_opt1, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt1_ok = compare_bf16_buffers(output_opt1, output_ref, output_size);
    printf("  opt1:    %s\n", opt1_ok ? "PASS" : "FAIL");

    // Test opt2
    lora_fused_add_opt2(intermediate, weight, output_opt2, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt2_ok = compare_bf16_buffers(output_opt2, output_ref, output_size);
    printf("  opt2:    %s\n", opt2_ok ? "PASS" : "FAIL");

    // Test opt3
    lora_fused_add_opt3(intermediate, weight, output_opt3, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt3_ok = compare_bf16_buffers(output_opt3, output_ref, output_size);
    printf("  opt3:    %s\n", opt3_ok ? "PASS" : "FAIL");

    // Test opt4
    lora_fused_add_opt4(intermediate, weight, output_opt4, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt4_ok = compare_bf16_buffers(output_opt4, output_ref, output_size);
    printf("  opt4:    %s\n", opt4_ok ? "PASS" : "FAIL");

    // Test opt5
    lora_fused_add_opt5(intermediate, weight, output_opt5, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt5_ok = compare_bf16_buffers(output_opt5, output_ref, output_size);
    printf("  opt5:    %s\n", opt5_ok ? "PASS" : "FAIL");

    // Test opt6
    lora_fused_add_opt6(intermediate, weight, output_opt6, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt6_ok = compare_bf16_buffers(output_opt6, output_ref, output_size);
    printf("  opt6:    %s\n", opt6_ok ? "PASS" : "FAIL");

    // Test opt7
    lora_fused_add_opt7(intermediate, weight, output_opt7, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt7_ok = compare_bf16_buffers(output_opt7, output_ref, output_size);
    printf("  opt7:    %s\n", opt7_ok ? "PASS" : "FAIL");

    // Test opt8
    lora_fused_add_opt8(intermediate, weight, output_opt8, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt8_ok = compare_bf16_buffers(output_opt8, output_ref, output_size);
    printf("  opt8:    %s\n", opt8_ok ? "PASS" : "FAIL");

    // Test opt9
    lora_fused_add_opt9(intermediate, weight, output_opt9, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt9_ok = compare_bf16_buffers(output_opt9, output_ref, output_size);
    printf("  opt9:    %s\n", opt9_ok ? "PASS" : "FAIL");

    // Test opt10 (pre-transposed weight)
    lora_fused_add_opt10(intermediate, weight_t, output_opt10, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
    bool opt10_ok = compare_bf16_buffers(output_opt10, output_ref, output_size);
    printf("  opt10:   %s\n", opt10_ok ? "PASS" : "FAIL");

    // Test opt11 (AMX with pre-transposed weight)
#if AMX_AVAILABLE_LORA
    static bool amx_configured = false;
    if (!amx_configured && init_amx_lora()) {
      configure_amx_lora();
      amx_configured = true;
    }
    if (amx_configured) {
      lora_fused_add_opt11_amx(intermediate, weight_vnni, output_opt11, cfg.num_tokens, cfg.rank, cfg.output_dim,
                               scale);
      bool opt11_ok = compare_bf16_buffers(output_opt11, output_ref, output_size);
      printf("  opt11:   %s  (AMX)\n", opt11_ok ? "PASS" : "FAIL");
    } else {
      printf("  opt11:   SKIP (AMX not available)\n");
    }
#else
    printf("  opt11:   SKIP (AMX not compiled)\n");
#endif

    // Test FP32 weight version
    lora_fused_add_fp32_weight(intermediate, weight_fp32, output_fp32w, cfg.num_tokens, cfg.rank, cfg.output_dim,
                               scale);
    bool fp32w_ok = compare_bf16_buffers(output_fp32w, output_ref, output_size);
    printf("  fp32w:   %s\n", fp32w_ok ? "PASS" : "FAIL");

    // Benchmark
    const int warmup = 3;
    const int iters = 10;

    auto benchmark = [&](auto kernel_fn, const char* name) {
      // Reset output
      memcpy(output_cur, output_init, output_size * sizeof(ggml_bf16_t));

      // Warmup
      for (int i = 0; i < warmup; i++) {
        memcpy(output_cur, output_init, output_size * sizeof(ggml_bf16_t));
        kernel_fn(intermediate, weight, output_cur, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
      }

      // Benchmark
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
        memcpy(output_cur, output_init, output_size * sizeof(ggml_bf16_t));
        kernel_fn(intermediate, weight, output_cur, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = elapsed_ms / iters;

      // Calculate GFLOPS: 2 * T * R * O (multiply-add)
      double flops = 2.0 * cfg.num_tokens * cfg.rank * cfg.output_dim;
      double gflops = (flops / 1e9) / (avg_ms / 1000.0);

      printf("  %8s: %.3f ms, %.1f GFLOPS\n", name, avg_ms, gflops);
    };

    benchmark(lora_fused_add_current, "current");
    benchmark(lora_fused_add_opt1, "opt1");
    benchmark(lora_fused_add_opt2, "opt2");
    benchmark(lora_fused_add_opt3, "opt3");
    benchmark(lora_fused_add_opt4, "opt4");
    benchmark(lora_fused_add_opt5, "opt5");
    benchmark(lora_fused_add_opt6, "opt6");
    benchmark(lora_fused_add_opt7, "opt7");
    benchmark(lora_fused_add_opt8, "opt8");
    benchmark(lora_fused_add_opt9, "opt9");

    // Benchmark opt10 separately (uses transposed weight)
    {
      // Warmup
      for (int i = 0; i < warmup; i++) {
        memcpy(output_opt10, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_opt10(intermediate, weight_t, output_opt10, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
      }

      // Benchmark
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
        memcpy(output_opt10, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_opt10(intermediate, weight_t, output_opt10, cfg.num_tokens, cfg.rank, cfg.output_dim, scale);
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = elapsed_ms / iters;
      double flops = 2.0 * cfg.num_tokens * cfg.rank * cfg.output_dim;
      double gflops = (flops / 1e9) / (avg_ms / 1000.0);

      printf("  %8s: %.3f ms, %.1f GFLOPS  (pre-transposed weight)\n", "opt10", avg_ms, gflops);
    }

    // Benchmark opt11 (AMX) separately
#if AMX_AVAILABLE_LORA
    if (amx_configured) {
      // Warmup
      for (int i = 0; i < warmup; i++) {
        memcpy(output_opt11, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_opt11_amx(intermediate, weight_vnni, output_opt11, cfg.num_tokens, cfg.rank, cfg.output_dim,
                                 scale);
      }

      // Benchmark
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
        memcpy(output_opt11, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_opt11_amx(intermediate, weight_vnni, output_opt11, cfg.num_tokens, cfg.rank, cfg.output_dim,
                                 scale);
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = elapsed_ms / iters;
      double flops = 2.0 * cfg.num_tokens * cfg.rank * cfg.output_dim;
      double gflops = (flops / 1e9) / (avg_ms / 1000.0);

      printf("  %8s: %.3f ms, %.1f GFLOPS  (AMX BF16)\n", "opt11", avg_ms, gflops);
    }
#endif

    // Benchmark FP32 weight version separately (different weight type)
    {
      // Warmup
      for (int i = 0; i < warmup; i++) {
        memcpy(output_fp32w, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_fp32_weight(intermediate, weight_fp32, output_fp32w, cfg.num_tokens, cfg.rank, cfg.output_dim,
                                   scale);
      }

      // Benchmark
      auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
        memcpy(output_fp32w, output_init, output_size * sizeof(ggml_bf16_t));
        lora_fused_add_fp32_weight(intermediate, weight_fp32, output_fp32w, cfg.num_tokens, cfg.rank, cfg.output_dim,
                                   scale);
      }
      auto end = std::chrono::high_resolution_clock::now();

      double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = elapsed_ms / iters;

      // Calculate GFLOPS: 2 * T * R * O (multiply-add)
      double flops = 2.0 * cfg.num_tokens * cfg.rank * cfg.output_dim;
      double gflops = (flops / 1e9) / (avg_ms / 1000.0);

      printf("  %8s: %.3f ms, %.1f GFLOPS  (FP32 weights, no BF16 conversion)\n", "fp32w", avg_ms, gflops);
    }

    printf("\n");

    free(intermediate);
    free(weight);
    free(weight_fp32);
    free(output_ref);
    free(output_cur);
    free(output_opt1);
    free(output_opt2);
    free(output_opt3);
    free(output_opt4);
    free(output_opt5);
    free(output_opt6);
    free(output_opt7);
    free(output_opt8);
    free(output_opt9);
    free(output_opt10);
    free(output_opt11);
    free(weight_t);
    free(weight_vnni);
    free(output_fp32w);
    free(output_init);
  }

  printf("=== All tests completed ===\n");
  return 0;
}
