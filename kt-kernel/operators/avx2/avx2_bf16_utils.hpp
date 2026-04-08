/**
 * @Description  : AVX2 BF16 utility functions (bf16<->fp32 conversion, activation)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * AVX2 ports of the AVX512 utilities in amx/la/utils.hpp and amx/la/amx.hpp.
 * Uses 256-bit SIMD (8 floats) instead of 512-bit (16 floats).
 **/
#ifndef CPUINFER_OPERATOR_AVX2_BF16_UTILS_H
#define CPUINFER_OPERATOR_AVX2_BF16_UTILS_H

#include <immintrin.h>
#include <cmath>
#include "llama.cpp/ggml.h"

namespace avx2 {

// ============================================================================
// BF16 <-> FP32 conversion
// ============================================================================

// Load 8 BF16 values and convert to 8 FP32 values
// BF16 is the upper 16 bits of FP32, so shift left by 16
static inline __m256 load_bf16_to_fp32(const ggml_bf16_t* src) {
  __m128i bf16 = _mm_loadu_si128((const __m128i*)src);
  __m256i i32 = _mm256_cvtepu16_epi32(bf16);
  return _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

// Convert 8 FP32 values to 8 BF16 values with round-to-nearest-even
// Matches ggml_compute_fp32_to_bf16 semantics (ggml-impl.h:87)
// and amx/la/utils.hpp:24 tie-bit correction
static inline void store_fp32_to_bf16(ggml_bf16_t* dst, __m256 src) {
  __m256i i32 = _mm256_castps_si256(src);
  // Round-to-nearest-even: add 0x7FFF + ((val >> 16) & 1)
  __m256i tie_bit = _mm256_and_si256(_mm256_srli_epi32(i32, 16), _mm256_set1_epi32(1));
  __m256i round = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), tie_bit);
  __m256i rounded = _mm256_add_epi32(i32, round);
  __m256i shifted = _mm256_srli_epi32(rounded, 16);
  // Pack 32-bit -> 16-bit
  // _mm_packus_epi32 processes 128-bit lanes: packs [lo0..lo3, hi0..hi3] -> [lo0..lo3, hi0..hi3]
  __m128i lo = _mm256_castsi256_si128(shifted);
  __m128i hi = _mm256_extracti128_si256(shifted, 1);
  __m128i packed = _mm_packus_epi32(lo, hi);
  _mm_storeu_si128((__m128i*)dst, packed);
}

// Load 16 BF16 -> 2x8 FP32 (corresponds to avx512_32xbf16_to_32xfp32)
static inline void load_16xbf16_to_2x8xfp32(const ggml_bf16_t* src, __m256* out0, __m256* out1) {
  *out0 = load_bf16_to_fp32(src);
  *out1 = load_bf16_to_fp32(src + 8);
}

// Store 2x8 FP32 -> 16 BF16 (corresponds to avx512_32xfp32_to_32xbf16)
static inline void store_2x8xfp32_to_16xbf16(__m256* in0, __m256* in1, ggml_bf16_t* dst) {
  store_fp32_to_bf16(dst, *in0);
  store_fp32_to_bf16(dst + 8, *in1);
}

// ============================================================================
// Horizontal sum for __m256 (8 floats -> 1 float)
// ============================================================================

static inline float hsum_avx2(__m256 v) {
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
  return _mm_cvtss_f32(sum);
}

// ============================================================================
// Fast exp approximation (AVX2 port of amx::exp_avx512)
// ============================================================================

static inline __m256 exp_avx2(__m256 x) {
  const __m256 log2e = _mm256_set1_ps(1.44269504089f);

  __m256 y = _mm256_mul_ps(x, log2e);
  __m256i int_part = _mm256_cvtps_epi32(y);
  __m256 frac_part = _mm256_sub_ps(y, _mm256_cvtepi32_ps(int_part));

  const __m256 poly_1 = _mm256_set1_ps(0.9999999995f);
  const __m256 poly_2 = _mm256_set1_ps(0.6931471805f);
  const __m256 poly_3 = _mm256_set1_ps(0.2402265069f);
  const __m256 poly_4 = _mm256_set1_ps(0.0555041087f);
  const __m256 poly_5 = _mm256_set1_ps(0.0096181291f);
  const __m256 poly_6 = _mm256_set1_ps(0.0013333558f);

  __m256 frac_exp = _mm256_fmadd_ps(
      _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(poly_6, frac_part, poly_5), frac_part, poly_4),
                                      frac_part, poly_3),
                      frac_part, poly_2),
      frac_part, poly_1);

  // 2^int_part: AVX2 doesn't have _mm256_scalef_ps, use manual construction
  // 2^n = reinterpret((n + 127) << 23) for float
  // Clamp int_part to [-126, 127] to avoid invalid bit patterns:
  //   int_part < -126 → biased < 1 → denorm/zero (scalef_ps would give 0)
  //   int_part > 127  → biased > 254 → inf (scalef_ps would give inf)
  __m256i clamped = _mm256_max_epi32(_mm256_min_epi32(int_part, _mm256_set1_epi32(127)),
                                      _mm256_set1_epi32(-126));
  __m256i biased = _mm256_add_epi32(clamped, _mm256_set1_epi32(127));
  __m256i shifted = _mm256_slli_epi32(biased, 23);
  __m256 two_pow_i = _mm256_castsi256_ps(shifted);

  return _mm256_mul_ps(two_pow_i, frac_exp);
}

// ============================================================================
// SiLU activation: silu(gate) * up = gate * sigmoid(gate) * up
// AVX2 port of amx::act_fn
// ============================================================================

static inline __m256 act_fn(__m256 gate_val, __m256 up_val) {
  __m256 neg_gate_val = _mm256_sub_ps(_mm256_setzero_ps(), gate_val);
  // Clamp to avoid exp overflow
  const __m256 max_exp_input = _mm256_set1_ps(88.0f);
  neg_gate_val = _mm256_min_ps(neg_gate_val, max_exp_input);
  __m256 exp_neg_gate = exp_avx2(neg_gate_val);
  __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_gate);
  __m256 act_val = _mm256_div_ps(gate_val, denom);

  return _mm256_mul_ps(act_val, up_val);
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_BF16_UTILS_H
