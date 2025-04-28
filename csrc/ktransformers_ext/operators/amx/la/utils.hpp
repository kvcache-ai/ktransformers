/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2025-04-25 18:28:12
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2025-04-25 18:28:12
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#pragma once
#include <cstdint>


template <typename T>
T* offset_pointer(T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + byte_offset);
}

template <typename T>
const T* offset_pointer(const T* ptr, std::size_t byte_offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(ptr) + byte_offset);
}

template <typename T>
T* offset_pointer_row_major(T* t, int row, int col, std::size_t ld) {
  return offset_pointer(t, row * ld) + col;
}

template <typename T>
T* offset_pointer_col_major(T* t, int row, int col, std::size_t ld) {
  return offset_pointer(t, col * ld) + row;
}

static inline void avx512_copy_32xbf16(__m512i* src, __m512i* dst) {
  _mm512_storeu_si512(dst, _mm512_loadu_si512(src));
}

static inline void avx512_32xfp32_to_32xbf16(__m512* src0, __m512* src1, __m512i* dst) {
  _mm512_storeu_si512(dst, __m512i(_mm512_cvtne2ps_pbh(*src1, *src0)));
}

static inline void avx512_32xbf16_to_32xfp32(__m512i* src, __m512* dst0, __m512* dst1) {
  _mm512_storeu_ps(dst0, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src))), 16)));
  _mm512_storeu_ps(dst1, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(src) + 1)), 16)));
}