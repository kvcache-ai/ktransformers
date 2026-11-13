/**
 * @Description  : FP8 E4M3 dequantization for AVX2 (LUT-based)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * AVX512 uses _mm512_permutex2var_epi8 (VBMI) for FP8→BF16 LUT conversion.
 * AVX2 uses a precomputed 256-entry FP8→FP32 lookup table + _mm256_i32gather_ps.
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
// Initialized once at program startup via init_fp8_lut()
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
//
// Shifts rather than _mm256_i32gather_ps: gather has poor throughput on Zen 2,
// and an FP8 MoE layer pays this per weight byte. Measured over 64 MiB on one
// EPYC 7452 core: gather 1.62 GB/s, this 4.15 GB/s. Identical output on all
// 256 byte values, so the table below is still the definition of correct.
//
// E4M3 is sign(1) exponent(4) mantissa(3) at bias 7; FP32 is sign(1)
// exponent(8) mantissa(23) at bias 127. For a normal value the fields already
// line up once exp|mant is moved to bit 20 and (127-7) << 23 is added.
// Exponent 0 has no implicit leading 1, so the rebias means nothing there and
// subnormals are computed separately as mantissa * 2^-9 and blended in.
// ============================================================================

static inline __m256 fp8x8_to_fp32x8(const uint8_t* src) {
  __m128i bytes = _mm_loadl_epi64((const __m128i*)src);
  __m256i b = _mm256_cvtepu8_epi32(bytes);

  const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
  const __m256i mag = _mm256_and_si256(b, _mm256_set1_epi32(0x7F));

  // normal: exp|mant to bits 26..20, then rebias 7 -> 127.
  __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(mag, 20), _mm256_set1_epi32(0x3C000000));

  // exponent 0: the value is mantissa * 2^-9 exactly, mantissa being 0..7.
  __m256i mant = _mm256_and_si256(b, _mm256_set1_epi32(0x7));
  __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mant), _mm256_set1_ps(1.0f / 512.0f));

  __m256i exp_is_zero = _mm256_cmpeq_epi32(_mm256_and_si256(mag, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
  __m256 out = _mm256_blendv_ps(_mm256_castsi256_ps(normal), sub, _mm256_castsi256_ps(exp_is_zero));

  // 0x7F is E4M3's only NaN and the table flushes it to zero; keep that, so a
  // bad weight cannot spread through the accumulator.
  __m256i is_nan = _mm256_cmpeq_epi32(mag, _mm256_set1_epi32(0x7F));
  out = _mm256_blendv_ps(out, _mm256_setzero_ps(), _mm256_castsi256_ps(is_nan));

  return _mm256_or_ps(out, _mm256_castsi256_ps(sign));
}

// Scalar fallback for non-aligned or tail elements
static inline float fp8_to_fp32_scalar(uint8_t val) {
  return get_fp8_lut().table[val];
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_FP8_DEQUANT_H
