#ifndef UTILS_HPP
#define UTILS_HPP
#include <cstddef>
#include <cstdint>

static inline void avx512_copy_32xbf16(__m512i* src, __m512i* dst) {
  _mm512_storeu_si512(dst, _mm512_loadu_si512(src));
}

static inline void avx512_32xfp32_to_32xbf16(__m512* src0, __m512* src1, __m512i* dst) {
  _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
}

static inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
  _mm512_storeu_ps(dst0, _mm512_castsi512_ps(
                             _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src))), 16)));
  _mm512_storeu_ps(dst1, _mm512_castsi512_ps(_mm512_slli_epi32(
                             _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(src) + 1)), 16)));
}

#endif  // UTILS_HPP