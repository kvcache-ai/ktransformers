/**
 * @Description  : FP8 E4M3 dequantization for AVX2
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * AVX512 uses _mm512_permutex2var_epi8 (VBMI) for FP8→BF16 LUT conversion.
 * AVX2 vector conversion uses exact bit arithmetic; scalar tails retain the LUT.
 *
 * FP8 E4M3 format: sign(1) + exponent(4) + mantissa(3)
 * Reference: examples/test_fp8_moe.py:103-116
 **/
#ifndef CPUINFER_OPERATOR_AVX2_FP8_DEQUANT_H
#define CPUINFER_OPERATOR_AVX2_FP8_DEQUANT_H

#include <immintrin.h>

#include <cmath>
#include <cstdint>

namespace avx2 {

// Precomputed FP8 E4M3 → FP32 lookup table (256 entries)
// Used for scalar fallback and tail elements
struct FP8LUT {
  alignas(32) float table[256];
  bool initialized = false;

  void init() {
    if (initialized) return;
    for (int i = 0; i < 256; i++) {
      int sign = (i >> 7) & 1;
      int exp = (i >> 3) & 0xF;   // 4-bit exponent (bits 3-6)
      int man = i & 0x7;           // 3-bit mantissa (bits 0-2)

      float val;
      if (exp == 0 && man == 0) {
        val = 0.0f;                                   // zero
      } else if (exp == 0) {
        val = std::ldexp((float)man / 8.0f, -6);      // subnormal: 2^(-6) * (0.man)
      } else if (exp == 15 && man == 7) {
        val = 0.0f;  // Only 0x7F is NaN in E4M3. Treat as 0 to avoid propagation.
        // E4M3 has no Inf. exp=15 with man=0-6 are valid finite values (256-448).
      } else {
        val = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);  // normal: 2^(exp-7) * (1.man)
      }
      table[i] = sign ? -val : val;
    }
    initialized = true;
  }
};

// Global LUT instance
inline FP8LUT& get_fp8_lut() {
  static FP8LUT lut;
  return lut;
}

// Ensure LUT is initialized (call once at startup)
inline void ensure_fp8_lut_initialized() {
  get_fp8_lut().init();
}

// ============================================================================
// AVX2 FP8→FP32 dequantization: 8 FP8 bytes → 8 FP32 values
// Exact arithmetic avoids costly AVX2 gathers and per-vector LUT initialization guards.
// ============================================================================

static inline __m256 fp8x8_to_fp32x8(const uint8_t* src) {
  const __m256i raw = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)src));
  const __m256i magnitude = _mm256_and_si256(raw, _mm256_set1_epi32(127));
  const __m256i sign = _mm256_slli_epi32(
      _mm256_and_si256(raw, _mm256_set1_epi32(128)), 24);

  // Normal FP8 mantissa/exponent bits occupy FP32 bits 20..26; bias changes by 120.
  __m256i bits = _mm256_add_epi32(_mm256_slli_epi32(magnitude, 20),
                                 _mm256_set1_epi32(0x3c000000));
  // FP8 subnormals are integer mantissas times 2^-9, exactly representable in FP32.
  const __m256 subnormal = _mm256_mul_ps(_mm256_cvtepi32_ps(magnitude),
                                       _mm256_set1_ps(1.0f / 512.0f));
  bits = _mm256_blendv_epi8(bits, _mm256_castps_si256(subnormal),
                            _mm256_cmpgt_epi32(_mm256_set1_epi32(8), magnitude));
  // Preserve the existing native NaN-to-zero contract, including the sign bit.
  bits = _mm256_andnot_si256(
      _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(127)), bits);
  return _mm256_castsi256_ps(_mm256_or_si256(bits, sign));
}

// Scalar fallback for non-aligned or tail elements
static inline float fp8_to_fp32_scalar(uint8_t val) {
  return get_fp8_lut().table[val];
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_FP8_DEQUANT_H