#ifndef AMX_BUFFERS_HPP
#define AMX_BUFFERS_HPP
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "amx_config.hpp"
#include "amx_utils.hpp"
#include "llama.cpp/ggml-impl.h"
#include "pack.hpp"
#include "utils.hpp"

namespace amx {

template <typename K>
struct BufferAImpl {
  int8_t* a;
  float* d;
  int max_m, k;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k) { return sizeof(int8_t) * max_m * k + sizeof(float) * max_m; }

  BufferAImpl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    if (max_m % M_STEP || k % K_STEP) {
      printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d\n", max_m, k, M_STEP, K_STEP);
      throw std::runtime_error("BufferAImpl: max_m and k must be multiple of M_STEP and K_STEP");
    }
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    a = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(a + max_m * k);
  }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float amax = 0.0f;
        for (int j = 0; j < k; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
        }
        d[m_begin + i] = amax / ((1 << 7) - 1);
      }
    }
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512 id = _mm512_set1_ps(d[m_begin + i] ? 1.0f / d[m_begin + i] : 0.0f);
            int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
            __m512 f0, f1, f2, f3;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
            __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
            __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
            __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
            __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
            __m128i s0 = _mm512_cvtsepi32_epi8(i0);
            __m128i s1 = _mm512_cvtsepi32_epi8(i1);
            __m128i s2 = _mm512_cvtsepi32_epi8(i2);
            __m128i s3 = _mm512_cvtsepi32_epi8(i3);
            _mm_store_si128((__m128i*)dst, s0);
            _mm_store_si128((__m128i*)(dst + 16), s1);
            _mm_store_si128((__m128i*)(dst + 32), s2);
            _mm_store_si128((__m128i*)(dst + 48), s3);
          }
        }
      }
    }
  }

  int8_t* get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }

  float* get_scale(int m, int m_begin) { return d + m_begin; }
};

template <typename K>
struct BufferAWithSumImpl {
  int8_t* a;
  float* d;
  float* sum;
  int max_m, k;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k) { return sizeof(int8_t) * max_m * k + sizeof(float) * max_m * 2; }

  BufferAWithSumImpl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    if (max_m % M_STEP || k % K_STEP) {
      printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d\n", max_m, k, M_STEP, K_STEP);
      throw std::runtime_error("BufferAWithSumImpl: max_m and k must be multiple of M_STEP and K_STEP");
    }
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    a = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(a + max_m * k);
    sum = d + max_m;
  }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float amax = 0.0f;
        float row_sum = 0.0f;
        for (int j = 0; j < k; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          row_sum += _mm512_reduce_add_ps(f0);
          row_sum += _mm512_reduce_add_ps(f1);
        }
        d[m_begin + i] = amax / ((1 << 7) - 1);
        sum[m_begin + i] = row_sum;
      }
    }
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512 id = _mm512_set1_ps(d[m_begin + i] ? 1.0f / d[m_begin + i] : 0.0f);
            int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
            __m512 f0, f1, f2, f3;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
            __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
            __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
            __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
            __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
            __m128i s0 = _mm512_cvtsepi32_epi8(i0);
            __m128i s1 = _mm512_cvtsepi32_epi8(i1);
            __m128i s2 = _mm512_cvtsepi32_epi8(i2);
            __m128i s3 = _mm512_cvtsepi32_epi8(i3);
            _mm_store_si128((__m128i*)dst, s0);
            _mm_store_si128((__m128i*)(dst + 16), s1);
            _mm_store_si128((__m128i*)(dst + 32), s2);
            _mm_store_si128((__m128i*)(dst + 48), s3);
          }
        }
      }
    }
  }

  int8_t* get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }

  float* get_scale(int m, int m_begin) { return d + m_begin; }
  float* get_sum(int m, int m_begin) { return sum + m_begin; }
};

template <typename K>
struct BufferAWithSumKGroupImpl {
  int8_t* a;
  float* d;
  float* sum;
  int max_m, k, k_group_size, k_group_count;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k, int k_group_size) {
    return sizeof(int8_t) * max_m * k + sizeof(float) * max_m * (k / k_group_size) * 2;
  }

  BufferAWithSumKGroupImpl(int max_m, int k, int k_group_size, void* ptr)
      : max_m(max_m), k(k), k_group_size(k_group_size) {
    if (max_m % M_STEP || k % K_STEP || k % k_group_size) {
      printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d, k_group_size = %d\n", max_m, k, M_STEP, K_STEP,
             k_group_size);
      throw std::runtime_error("BufferAWithSumImpl: max_m and k must be multiple of M_STEP and K_STEP");
    }
    k_group_count = k / k_group_size;
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    a = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(a + max_m * k);
    sum = d + max_m * k_group_count;
  }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    // for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
    //   for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
    //     for(int kg = 0; kg < k_group_count; kg++){
    //       float amax = 0.0f;
    //       float row_sum = 0.0f;
    //       for (int j = 0; j < k; j += 32) {
    //         __m512 f0, f1;
    //         avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k + j), &f0, &f1);
    //         amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
    //         amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
    //         row_sum += _mm512_reduce_add_ps(f0);
    //         row_sum += _mm512_reduce_add_ps(f1);
    //       }
    //       d[(m_begin + i) * k_group_count + kg] = amax / ((1 << 7) - 1);
    //       sum[(m_begin + i) * k_group_count + kg] = row_sum;
    //     }
    //   }
    // }
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int kg = 0; kg < k_group_count; kg++) {
        float amax = 0.0f;
        float row_sum = 0.0f;
        int k_start = kg * k_group_size;
        int k_end = k_start + k_group_size;
        for (int j = k_start; j < k_end; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_idx)*k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          row_sum += _mm512_reduce_add_ps(f0);
          row_sum += _mm512_reduce_add_ps(f1);
        }
        d[(m_idx)*k_group_count + kg] = amax / ((1 << 7) - 1);
        sum[(m_idx)*k_group_count + kg] = row_sum;
      }
    }
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            int k_group_idx = (k_block_begin + k_begin) / k_group_size;
            float scale = d[(m_begin + i) * k_group_count + k_group_idx];
            __m512 id = _mm512_set1_ps(scale ? 1.0f / scale : 0.0f);
            // __m512 id = _mm512_set1_ps(d[m_begin + i] ? 1.0f / d[m_begin + i] : 0.0f);
            int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
            __m512 f0, f1, f2, f3;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
            __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
            __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
            __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
            __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
            __m128i s0 = _mm512_cvtsepi32_epi8(i0);
            __m128i s1 = _mm512_cvtsepi32_epi8(i1);
            __m128i s2 = _mm512_cvtsepi32_epi8(i2);
            __m128i s3 = _mm512_cvtsepi32_epi8(i3);
            _mm_store_si128((__m128i*)dst, s0);
            _mm_store_si128((__m128i*)(dst + 16), s1);
            _mm_store_si128((__m128i*)(dst + 32), s2);
            _mm_store_si128((__m128i*)(dst + 48), s3);
          }
        }
      }
    }
  }

  int8_t* get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }

  float* get_scale(int m, int m_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return d + m_begin * k_group_count + k_group_idx;
  }
  float* get_sum(int m, int m_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return sum + m_begin * k_group_count + k_group_idx;
  }
};

template <typename K>
struct BufferAKGroupImpl {
  int8_t* a;
  float* d;
  int max_m, k, k_group_size, k_group_count;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  using index_t = Packed2DLayout::index_t;
  Packed2DLayout pack;

  static size_t required_size(int max_m, int k, int k_group_size) {
    ASSERT_RELEASE(k % k_group_size == 0, "k must be multiple of k_group_size");
    return sizeof(int8_t) * max_m * k + sizeof(float) * max_m * (k / k_group_size);
  }

  BufferAKGroupImpl(int max_m, int k, int k_group_size, void* ptr)
      : max_m(max_m),
        k(k),
        k_group_size(k_group_size),
        pack({{static_cast<index_t>(K_STEP), 'c'},
              {static_cast<index_t>(M_STEP), 'r'},
              {static_cast<index_t>(k_group_size / K_STEP), 'c'},
              {static_cast<index_t>(K_BLOCK / k_group_size), 'c'},
              {static_cast<index_t>(max_m / M_STEP), 'r'},
              {static_cast<index_t>(k / K_BLOCK), 'c'}}) {
    ASSERT_RELEASE(k % k_group_size == 0, "k must be multiple of k_group_size");
    ASSERT_RELEASE(max_m % M_STEP == 0, "max_m must be multiple of M_STEP");
    ASSERT_RELEASE(k % K_STEP == 0, "k must be multiple of K_STEP");
    ASSERT_RELEASE(K_BLOCK % k_group_size == 0, "K_BLOCK must be multiple of k_group_size");
    ASSERT_RELEASE(k % K_BLOCK == 0, "k must be multiple of K_BLOCK");
    k_group_count = k / k_group_size;

    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    a = reinterpret_cast<int8_t*>(ptr);
    d = reinterpret_cast<float*>(a + max_m * k);
  }

  int8_t* get_submat(int m, int k, int m_begin, int k_begin) {
    // Follow BufferAImpl pattern
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }

  void from_mat(int m, ggml_bf16_t* src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);

    // 计算每个 k_group 的 scale
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int kg = 0; kg < k_group_count; kg++) {
        float amax = 0.0f;
        int k_start = kg * k_group_size;
        int k_end = k_start + k_group_size;
        // 32 -> M_STEP
        for (int j = k_start; j < k_end; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + m_idx * k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
        }
        d[m_idx * k_group_count + kg] = amax / ((1 << 7) - 1);
      }
    }

    // Simplified quantization following BufferAImpl pattern but with k-group support
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            // Get the scale for this k_group
            int k_group_idx = (k_block_begin + k_begin) / k_group_size;
            float scale = d[(m_begin + i) * k_group_count + k_group_idx];
            __m512 id = _mm512_set1_ps(scale ? 1.0f / scale : 0.0f);

            // Calculate destination similar to BufferAImpl but accounting for k-groups
            int8_t* dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;

            __m512 f0, f1, f2, f3;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
            __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
            __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
            __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
            __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
            __m128i s0 = _mm512_cvtsepi32_epi8(i0);
            __m128i s1 = _mm512_cvtsepi32_epi8(i1);
            __m128i s2 = _mm512_cvtsepi32_epi8(i2);
            __m128i s3 = _mm512_cvtsepi32_epi8(i3);
            _mm_store_si128((__m128i*)dst, s0);
            _mm_store_si128((__m128i*)(dst + 16), s1);
            _mm_store_si128((__m128i*)(dst + 32), s2);
            _mm_store_si128((__m128i*)(dst + 48), s3);
          }
        }
      }
    }
  }

  float* get_scale(int m, int m_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return d + m_begin * k_group_count + k_group_idx;
  }
};

template <typename K>
struct BufferBInt4Impl {
  using dt = typename K::dt;
  dt* b;
  float* d;
  int n, k;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static const int B_K_STEP = 2 * K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }

  BufferBInt4Impl(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % B_K_STEP == 0);
    if (n % N_STEP || k % B_K_STEP) {
      printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d\n", n, k, N_STEP, B_K_STEP);
      throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
    }
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, n * k / 2));
  }

  static __m128i round_4bit_s8(__m128i x) {
    __m128i s = _mm_and_si128(x, _mm_set1_epi8(0x80));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 1));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 2));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 4));

    x = _mm_abs_epi8(x);
    x = _mm_add_epi8(x, _mm_set1_epi8(0x08));
    x = _mm_and_si128(x, _mm_set1_epi8(0xF0));
    x = _mm_xor_si128(x, s);
    x = _mm_sub_epi8(x, s);
    return x;
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int i = 0; i < N_STEP; i++) {
        float amax = 0.0f;
        for (int j = 0; j < k; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
          amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
        }
        d[n_block_begin + n_begin + i] = amax / 112.0;  // 7*16
      }
    }
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            __m512 id = _mm512_set1_ps(d[n_block_begin + n_begin + i] ? 1.0f / d[n_block_begin + n_begin + i] : 0.0f);
            dt* dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                         k_begin * N_STEP + i * B_K_STEP) /
                                            2);
            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              s0 = _mm_srli_epi16(round_4bit_s8(s0), 4);
              s1 = _mm_srli_epi16(round_4bit_s8(s1), 4);
              s2 = _mm_srli_epi16(round_4bit_s8(s2), 4);
              s3 = _mm_srli_epi16(round_4bit_s8(s3), 4);
              // s0 = _mm_or_si128(round_up4(s0), _mm_srli_epi16(round_up4(s1), 4));
              // s2 = _mm_or_si128(round_up4(s2), _mm_srli_epi16(round_up4(s3), 4));
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)), s1);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)), s2);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)), s3);
            }

            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              s0 = round_4bit_s8(s0);
              s1 = round_4bit_s8(s1);
              s2 = round_4bit_s8(s2);
              s3 = round_4bit_s8(s3);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 0)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 0))), s0));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 16))), s1));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 32))), s2));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 48))), s3));
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return offset_pointer(
        b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
  }

  float* get_scale(int n, int n_begin) { return d + n_begin; }
};

template <typename K>
struct BufferBKGroupImpl {
  using dt = typename K::dt;
  dt* b;
  float* d;
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static const int B_K_STEP = 2 * K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k, int k_group_size) {
    ASSERT_RELEASE(k % k_group_size == 0, "k must be multiple of k_group_size");
    return sizeof(int8_t) * n * k / 2 + sizeof(float) * n * (k / k_group_size);
  }

  BufferBKGroupImpl(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % B_K_STEP == 0);
    ASSERT_RELEASE(k % k_group_size == 0, "k must be multiple of k_group_size");
    ASSERT_RELEASE(K_BLOCK % k_group_size == 0, "K_BLOCK must be multiple of k_group_size");
    if (n % N_STEP || k % B_K_STEP) {
      printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d\n", n, k, N_STEP, B_K_STEP);
      throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, n * k / 2));
  }

  static __m128i round_4bit_s8(__m128i x) {
    __m128i s = _mm_and_si128(x, _mm_set1_epi8(0x80));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 1));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 2));
    s = _mm_or_si128(s, _mm_srai_epi16(s, 4));

    x = _mm_abs_epi8(x);
    x = _mm_add_epi8(x, _mm_set1_epi8(0x08));
    x = _mm_and_si128(x, _mm_set1_epi8(0xF0));
    x = _mm_xor_si128(x, s);
    x = _mm_sub_epi8(x, s);
    return x;
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;

    // Compute scales per k-group for each n
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int i = 0; i < N_STEP; i++) {
        for (int kg = 0; kg < k_group_count; kg++) {
          float amax = 0.0f;
          int k_start = kg * k_group_size;
          int k_end = k_start + k_group_size;

          for (int j = k_start; j < k_end; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[kg * n + (n_block_begin + n_begin + i)] = amax / 112.0;  // 7*16
          // d[(n_block_begin + n_begin + i) * k_group_count + kg] = amax / 112.0; // 7*16
        }
      }
    }

    // Quantize with per k-group scaling
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            // Get the scale for this k_group
            int k_group_idx0 = (k_block_begin + k_begin) / k_group_size;
            int k_group_idx1 = (k_block_begin + k_begin + K_STEP) / k_group_size;
            float scale0 = d[k_group_idx0 * n + (n_block_begin + n_begin + i)];
            float scale1 = d[k_group_idx1 * n + (n_block_begin + n_begin + i)];
            // float scale = d[(n_block_begin + n_begin + i) * k_group_count + k_group_idx];
            __m512 id0 = _mm512_set1_ps(scale0 ? 1.0f / scale0 : 0.0f);
            __m512 id1 = _mm512_set1_ps(scale1 ? 1.0f / scale1 : 0.0f);

            dt* dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                         k_begin * N_STEP + i * B_K_STEP) /
                                            2);
            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id0));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id0));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id0));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id0));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              s0 = _mm_srli_epi16(round_4bit_s8(s0), 4);
              s1 = _mm_srli_epi16(round_4bit_s8(s1), 4);
              s2 = _mm_srli_epi16(round_4bit_s8(s2), 4);
              s3 = _mm_srli_epi16(round_4bit_s8(s3), 4);
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)), s1);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)), s2);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)), s3);
            }

            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id1));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id1));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id1));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id1));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              s0 = round_4bit_s8(s0);
              s1 = round_4bit_s8(s1);
              s2 = round_4bit_s8(s2);
              s3 = round_4bit_s8(s3);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 0)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 0))), s0));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 16))), s1));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 32))), s2));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 48))), s3));
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return offset_pointer(
        b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
  }

  float* get_scale(int n, int n_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return d + k_group_idx * n + n_begin;
    // return d + n_begin * k_group_count + k_group_idx;
  }
};

template <typename K>
struct BufferBInt4WithZeroImpl {
  using dt = typename K::dt;
  dt* b;
  float *d, *mins;  // scale, mins
  int n, k;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static const int B_K_STEP = 2 * K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n * 2; }

  BufferBInt4WithZeroImpl(int n, int k, void* ptr) : n(n), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % B_K_STEP == 0);
    if (n % N_STEP || k % B_K_STEP) {
      printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d\n", n, k, N_STEP, B_K_STEP);
      throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
    }
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, n * k / 2));
    mins = d + n;
  }

  // 对 uint8_t 批量四舍五入到最接近的 16 倍数
  static __m128i round_4bit_u8(__m128i x) {
    // 加 8 做四舍五入，使用 Saturate
    x = _mm_adds_epi8(x, _mm_set1_epi8(0x08));
    // 清除低 4 位（即对 16 对齐）
    x = _mm_and_si128(x, _mm_set1_epi8(0xF0));
    return x;
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int i = 0; i < N_STEP; i++) {
        float amax = std::numeric_limits<float>::lowest();
        float amin = std::numeric_limits<float>::max();
        for (int j = 0; j < k; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
          amax = MAX(amax, _mm512_reduce_max_ps(f0));
          amax = MAX(amax, _mm512_reduce_max_ps(f1));
          amin = MIN(amin, _mm512_reduce_min_ps(f0));
          amin = MIN(amin, _mm512_reduce_min_ps(f1));
        }
        d[n_block_begin + n_begin + i] = (amax - amin) / 240.0;  // 15*16
        mins[n_block_begin + n_begin + i] = amin;
      }
    }
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            __m512 id = _mm512_set1_ps(d[n_block_begin + n_begin + i] ? 1.0f / d[n_block_begin + n_begin + i] : 0.0f);
            __m512 zps = _mm512_set1_ps(-mins[n_block_begin + n_begin + i]);
            dt* dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                         k_begin * N_STEP + i * B_K_STEP) /
                                            2);
            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps), id));
              __m512i i1 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps), id));
              __m512i i2 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps), id));
              __m512i i3 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps), id));
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              s0 = _mm_srli_epi16(round_4bit_u8(s0), 4);
              s1 = _mm_srli_epi16(round_4bit_u8(s1), 4);
              s2 = _mm_srli_epi16(round_4bit_u8(s2), 4);
              s3 = _mm_srli_epi16(round_4bit_u8(s3), 4);
              // s0 = _mm_or_si128(round_up4(s0), _mm_srli_epi16(round_up4(s1), 4));
              // s2 = _mm_or_si128(round_up4(s2), _mm_srli_epi16(round_up4(s3), 4));
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)), s1);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)), s2);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)), s3);
            }

            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps), id));
              __m512i i1 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps), id));
              __m512i i2 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps), id));
              __m512i i3 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps), id));
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              s0 = round_4bit_u8(s0);
              s1 = round_4bit_u8(s1);
              s2 = round_4bit_u8(s2);
              s3 = round_4bit_u8(s3);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 0)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 0))), s0));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 16))), s1));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 32))), s2));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 48))), s3));
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return offset_pointer(
        b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
  }

  float* get_scale(int n, int n_begin) { return d + n_begin; }
  float* get_min(int n, int n_begin) { return mins + n_begin; }
};

template <typename K>
struct BufferBInt4WithZeroKGroupImpl {
  using dt = typename K::dt;
  dt* b;
  float *d, *mins;  // scale, mins
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static const int B_K_STEP = 2 * K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k, int k_group_size) {
    return sizeof(int8_t) * n * k / 2 + sizeof(float) * n * (k / k_group_size) * 2;
  }

  BufferBInt4WithZeroKGroupImpl(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % B_K_STEP == 0);
    if (n % N_STEP || k % B_K_STEP || k % k_group_size) {
      printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d, k_group_size: %d\n", n, k, N_STEP, B_K_STEP, k_group_size);
      throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, n * k / 2));
    mins = d + n * k_group_count;
  }

  // 对 uint8_t 批量四舍五入到最接近的 16 倍数
  static __m128i round_4bit_u8(__m128i x) {
    // 加 8 做四舍五入，使用 Saturate
    x = _mm_adds_epi8(x, _mm_set1_epi8(0x08));
    // 清除低 4 位（即对 16 对齐）
    x = _mm_and_si128(x, _mm_set1_epi8(0xF0));
    return x;
  }

  void from_raw_mat(uint8_t* proj, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;

    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            uint8_t* dst = (uint8_t*)offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size +
                                                        n_begin * k_block_size + k_begin * N_STEP + i * B_K_STEP) >>
                                                           1);
            uint32_t* src =
                (uint32_t*)offset_pointer(proj, ((n_block_begin + n_begin + i) * k + k_block_begin + k_begin) >> 1);
            for (int a0 = 0; a0 < 8; a0++) {
              uint32_t src0 = src[a0], src1 = src[a0 + 8];
              for (int a1 = 0; a1 < 8; a1++) {
                uint8_t cur_src0 = src0 & 0x0F, cur_src1 = src1 & 0x0F;
                dst[(a0 * 8) + a1] = (cur_src0 | (cur_src1 << 4));
                src0 = src0 >> 4;
                src1 = src1 >> 4;
              }
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int i = 0; i < N_STEP; i++) {
        for (int kg = 0; kg < k_group_count; kg++) {
          int k_start = kg * k_group_size;
          int k_end = k_start + k_group_size;

          float amax = std::numeric_limits<float>::lowest();
          float amin = std::numeric_limits<float>::max();
          for (int j = k_start; j < k_end; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(f0));
            amax = MAX(amax, _mm512_reduce_max_ps(f1));
            amin = MIN(amin, _mm512_reduce_min_ps(f0));
            amin = MIN(amin, _mm512_reduce_min_ps(f1));
          }
          d[kg * n + n_block_begin + n_begin + i] = (amax - amin) / 240.0;  // 15*16
          // d[n_block_begin + n_begin + i] = (amax - amin) / 240.0; // 15*16
          mins[kg * n + n_block_begin + n_begin + i] = amin;
        }
      }
    }
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            int k_group_idx0 = (k_block_begin + k_begin) / k_group_size;
            int k_group_idx1 = (k_block_begin + k_begin + K_STEP) / k_group_size;
            float scale0 = d[k_group_idx0 * n + n_block_begin + n_begin + i];
            float scale1 = d[k_group_idx1 * n + n_block_begin + n_begin + i];
            __m512 id0 = _mm512_set1_ps(scale0 ? 1.0f / scale0 : 0.0f);
            __m512 id1 = _mm512_set1_ps(scale1 ? 1.0f / scale1 : 0.0f);
            __m512 zps0 = _mm512_set1_ps(-mins[k_group_idx0 * n + n_block_begin + n_begin + i]);
            __m512 zps1 = _mm512_set1_ps(-mins[k_group_idx1 * n + n_block_begin + n_begin + i]);
            // __m512 zps = _mm512_set1_ps(-mins[n_block_begin + n_begin + i]);
            dt* dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                         k_begin * N_STEP + i * B_K_STEP) /
                                            2);
            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps0), id0));
              __m512i i1 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps0), id0));
              __m512i i2 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps0), id0));
              __m512i i3 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps0), id0));
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              s0 = _mm_srli_epi16(round_4bit_u8(s0), 4);
              s1 = _mm_srli_epi16(round_4bit_u8(s1), 4);
              s2 = _mm_srli_epi16(round_4bit_u8(s2), 4);
              s3 = _mm_srli_epi16(round_4bit_u8(s3), 4);
              // s0 = _mm_or_si128(round_up4(s0), _mm_srli_epi16(round_up4(s1), 4));
              // s2 = _mm_or_si128(round_up4(s2), _mm_srli_epi16(round_up4(s3), 4));
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)), s1);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)), s2);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)), s3);
            }

            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps1), id1));
              __m512i i1 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps1), id1));
              __m512i i2 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps1), id1));
              __m512i i3 = _mm512_cvtps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps1), id1));
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              s0 = round_4bit_u8(s0);
              s1 = round_4bit_u8(s1);
              s2 = round_4bit_u8(s2);
              s3 = round_4bit_u8(s3);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 0)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 0))), s0));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 16))), s1));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 32))), s2));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 48))), s3));
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return offset_pointer(
        b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
  }

  float* get_scale(int n, int n_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return d + k_group_idx * n + n_begin;
  }
  float* get_min(int n, int n_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return mins + k_group_idx * n + n_begin;
  }
};

template <typename K>
struct BufferBInt4WithZeroLowKGroupImpl {
  using dt = typename K::dt;
  dt* b;
  float *d, *mins;  // scale, mins
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;
  static constexpr int TILE_N = K::TILE_N;

  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;
  static const int B_K_STEP = 2 * K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k, int k_group_size) {
    return sizeof(int8_t) * n * k / 2 + sizeof(float) * n * (k / k_group_size) * 2;
  }

  BufferBInt4WithZeroLowKGroupImpl(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(n % N_STEP == 0);
    assert(k % B_K_STEP == 0);
    if (n % N_STEP || k % B_K_STEP || k % k_group_size) {
      printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d, k_group_size: %d\n", n, k, N_STEP, B_K_STEP, k_group_size);
      throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<float*>(offset_pointer(b, n * k / 2));
    mins = d + n * k_group_count;
  }

  // 对 uint8_t 批量四舍五入到最接近的 16 倍数
  static __m128i round_4bit_u8(__m128i x) {
    // 加 8 做四舍五入，使用 Saturate
    x = _mm_adds_epi8(x, _mm_set1_epi8(0x08));
    // 清除低 4 位（即对 16 对齐）
    x = _mm_and_si128(x, _mm_set1_epi8(0xF0));
    return x;
  }

  void from_raw_mat(uint8_t* proj, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;

    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            uint8_t* dst = (uint8_t*)offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size +
                                                        n_begin * k_block_size + k_begin * N_STEP + i * B_K_STEP) >>
                                                           1);
            uint32_t* src =
                (uint32_t*)offset_pointer(proj, ((n_block_begin + n_begin + i) * k + k_block_begin + k_begin) >> 1);
            for (int a0 = 0; a0 < 8; a0++) {
              uint32_t src0 = src[a0], src1 = src[a0 + 8];
              for (int a1 = 0; a1 < 8; a1++) {
                uint8_t cur_src0 = src0 & 0x0F, cur_src1 = src1 & 0x0F;
                dst[(a0 * 8) + a1] = (cur_src0 | (cur_src1 << 4));
                src0 = src0 >> 4;
                src1 = src1 >> 4;
              }
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  void from_mat(ggml_bf16_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int i = 0; i < N_STEP; i++) {
        for (int kg = 0; kg < k_group_count; kg++) {
          int k_start = kg * k_group_size;
          int k_end = k_start + k_group_size;

          float amax = std::numeric_limits<float>::lowest();
          float amin = std::numeric_limits<float>::max();
          for (int j = k_start; j < k_end; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(f0));
            amax = MAX(amax, _mm512_reduce_max_ps(f1));
            amin = MIN(amin, _mm512_reduce_min_ps(f0));
            amin = MIN(amin, _mm512_reduce_min_ps(f1));
          }
          d[kg * n + n_block_begin + n_begin + i] = (amax - amin) / 15.0;  // 15*16
          // d[n_block_begin + n_begin + i] = (amax - amin) / 240.0; // 15*16
          mins[kg * n + n_block_begin + n_begin + i] = amin;
        }
      }
    }

    for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
      for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
        int k_block_size = std::min(K_BLOCK, k - k_block_begin);
        for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
          for (int i = 0; i < N_STEP; i++) {
            int k_group_idx0 = (k_block_begin + k_begin) / k_group_size;
            int k_group_idx1 = (k_block_begin + k_begin + K_STEP) / k_group_size;
            float scale0 = d[k_group_idx0 * n + n_block_begin + n_begin + i];
            float scale1 = d[k_group_idx1 * n + n_block_begin + n_begin + i];
            __m512 id0 = _mm512_set1_ps(scale0 ? 1.0f / scale0 : 0.0f);
            __m512 id1 = _mm512_set1_ps(scale1 ? 1.0f / scale1 : 0.0f);
            __m512 zps0 = _mm512_set1_ps(-mins[k_group_idx0 * n + n_block_begin + n_begin + i]);
            __m512 zps1 = _mm512_set1_ps(-mins[k_group_idx1 * n + n_block_begin + n_begin + i]);
            // __m512 zps = _mm512_set1_ps(-mins[n_block_begin + n_begin + i]);
            dt* dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                         k_begin * N_STEP + i * B_K_STEP) /
                                            2);
            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps0), id0),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i1 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps0), id0),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i2 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps0), id0),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i3 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps0), id0),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              // s0 = _mm_srli_epi16(s0, 4);
              // s1 = _mm_srli_epi16(s1, 4);
              // s2 = _mm_srli_epi16(s2, 4);
              // s3 = _mm_srli_epi16(s3, 4);
              // s0 = _mm_or_si128(round_up4(s0), _mm_srli_epi16(round_up4(s1), 4));
              // s2 = _mm_or_si128(round_up4(s2), _mm_srli_epi16(round_up4(s3), 4));
              _mm_store_si128((__m128i*)dst, s0);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)), s1);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)), s2);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)), s3);
            }

            {
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i*)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
              __m512i i0 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f0, zps1), id1),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i1 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f1, zps1), id1),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i2 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f2, zps1), id1),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m512i i3 = _mm512_cvt_roundps_epu32(_mm512_mul_ps(_mm512_add_ps(f3, zps1), id1),
                                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
              __m128i s0 = _mm512_cvtusepi32_epi8(i0);
              __m128i s1 = _mm512_cvtusepi32_epi8(i1);
              __m128i s2 = _mm512_cvtusepi32_epi8(i2);
              __m128i s3 = _mm512_cvtusepi32_epi8(i3);
              s0 = _mm_slli_epi16(s0, 4);
              s1 = _mm_slli_epi16(s1, 4);
              s2 = _mm_slli_epi16(s2, 4);
              s3 = _mm_slli_epi16(s3, 4);
              _mm_store_si128((__m128i*)(offset_pointer(dst, 0)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 0))), s0));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 16)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 16))), s1));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 32)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 32))), s2));
              _mm_store_si128((__m128i*)(offset_pointer(dst, 48)),
                              _mm_or_si128(_mm_loadu_si128((__m128i*)(offset_pointer(dst, 48))), s3));
            }
          }
          transpose_16x16_32bit((__m512i*)(offset_pointer(
              b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
          transpose_16x16_32bit(
              (__m512i*)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                            k_begin * N_STEP + TILE_N * B_K_STEP) /
                                               2)));
        }
      }
    }
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    n_begin -= n_block_begin;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return offset_pointer(
        b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
  }

  float* get_scale(int n, int n_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return d + k_group_idx * n + n_begin;
  }
  float* get_min(int n, int n_begin, int k, int k_begin) {
    int k_group_idx = k_begin / k_group_size;
    return mins + k_group_idx * n + n_begin;
  }
};

template <typename K>
struct BufferCImpl {
  float* c;
  int max_m, n;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

  BufferCImpl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    if (max_m % M_STEP || n % N_STEP) {
      printf("max_m = %d, n = %d, M_STEP = %d, N_STEP = %d\n", max_m, n, M_STEP, N_STEP);
      throw std::runtime_error("BufferCImpl: max_m and n must be multiple of M_STEP and N_STEP");
    }
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    c = reinterpret_cast<float*>(ptr);
  }

  void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512* x0 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
          __m512* x1 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP + 16);
          avx512_32xfp32_to_32xbf16(x0, x1, (__m512i*)(dst + (m_begin + i) * n + n_block_begin + n_begin));
        }
      }
    }
  }

  float* get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }
};

template <typename K>
struct BufferCReduceImpl {
  float* c;
  int32_t* int_c;  // Additional int32_t buffer, same size as c
  int max_m, n;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static size_t required_size(int max_m, int n) {
    // Need space for both float* c and int32_t* int_c
    return sizeof(float) * max_m * n + sizeof(int32_t) * max_m * n;
  }

  BufferCReduceImpl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    if (max_m % M_STEP || n % N_STEP) {
      printf("max_m = %d, n = %d, M_STEP = %d, N_STEP = %d\n", max_m, n, M_STEP, N_STEP);
      throw std::runtime_error("BufferCReduceImpl: max_m and n must be multiple of M_STEP and N_STEP");
    }
    set_data(ptr);
  }

  void set_data(void* ptr) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    c = reinterpret_cast<float*>(ptr);
    // int_c starts after the float buffer
    int_c = reinterpret_cast<int32_t*>(c + max_m * n);
  }

  void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512* x0 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
          __m512* x1 =
              (__m512*)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP + 16);
          avx512_32xfp32_to_32xbf16(x0, x1, (__m512i*)(dst + (m_begin + i) * n + n_block_begin + n_begin));
        }
      }
    }
  }

  float* get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }

  int32_t* get_int_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return int_c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }

  // Clear the int_c buffer
  void clear_int_buffer() { std::memset(int_c, 0, sizeof(int32_t) * max_m * n); }

  // Convert int32_t results to float
  void convert_int_to_float(int m) {
    assert(m <= max_m);
    for (int i = 0; i < m * n; i++) {
      c[i] = static_cast<float>(int_c[i]);
    }
  }
};

}  // namespace amx

#endif  // AMX_BUFFERS_HPP