#ifndef UTILS_HPP
#define UTILS_HPP
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

static inline void avx512_copy_32xbf16(__m512i* src, __m512i* dst) {
  _mm512_storeu_si512(dst, _mm512_loadu_si512(src));
}

// FP32 to BF16 conversion (32 floats -> 32 bf16)
// This requires AVX512BF16 for the fast path, with a fallback for CPUs without it
static inline void avx512_32xfp32_to_32xbf16(__m512* src0, __m512* src1, __m512i* dst) {
#if defined(HAVE_AVX512BF16) || defined(__AVX512BF16__)
  // Fast path: use native AVX512BF16 instruction
  _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
#else
  // Fallback: manual BF16 conversion using bit manipulation
  // BF16 is the upper 16 bits of FP32 (with rounding)
  __m512i i0 = _mm512_castps_si512(*src0);
  __m512i i1 = _mm512_castps_si512(*src1);

  // Round to nearest even: add 0x7FFF + ((val >> 16) & 1)
  __m512i round0 = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                                    _mm512_and_epi32(_mm512_srli_epi32(i0, 16), _mm512_set1_epi32(1)));
  __m512i round1 = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                                    _mm512_and_epi32(_mm512_srli_epi32(i1, 16), _mm512_set1_epi32(1)));

  i0 = _mm512_add_epi32(i0, round0);
  i1 = _mm512_add_epi32(i1, round1);

  // Extract upper 16 bits (BF16)
  i0 = _mm512_srli_epi32(i0, 16);
  i1 = _mm512_srli_epi32(i1, 16);

  // Pack 32-bit values to 16-bit
  __m512i result = _mm512_packus_epi32(i0, i1);
  // Fix the interleaving from packus
  result = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), result);

  _mm512_storeu_si512(dst, result);
#endif
}

// BF16 to FP32 conversion (32 bf16 -> 32 floats)
// This does NOT require AVX512BF16 - uses basic AVX512 bit manipulation
static inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
  _mm512_storeu_ps(dst0, _mm512_castsi512_ps(
                             _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src))), 16)));
  _mm512_storeu_ps(dst1, _mm512_castsi512_ps(_mm512_slli_epi32(
                             _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src) + 1)), 16)));
}

static inline __m512 vector_abs_max(__m512 a, __m512 b) {
  __m512 a_abs = _mm512_abs_ps(a);
  __m512 b_abs = _mm512_abs_ps(b);

  __mmask16 mask = _mm512_cmp_ps_mask(a_abs, b_abs, _CMP_GT_OS);

  return _mm512_mask_blend_ps(mask, b_abs, a_abs);
}

#endif  // UTILS_HPP