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
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "utils.hpp"
#include <memory>

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

namespace amx {

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

const int TMMCount = 8;
const int MaxTileHeight = 16;
const int MaxTileWidth = 64;

const int AMX_BLK_SIZE = 32;

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

inline bool enable_amx() {
  static thread_local bool initialized = false;
  if (initialized) {
    return true;
  }
  initialized = true;

  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
    return false;
  } else {
    // printf("\n TILE DATA USE SET - OK \n\n");
    return true;
  }
  return true;
}

struct alignas(64) TileConfig {
  uint8_t palette;
  uint8_t start_row;
  std::array<uint8_t, 14> __0 = {};
  std::array<uint16_t, 8> colsb;
  std::array<uint8_t, 16> __1 = {};
  std::array<uint8_t, 8> rows;
  std::array<uint8_t, 8> __2 = {};

  TileConfig() {
    palette = 1;
    start_row = 0;
    for (int i = 0; i < 8; i++) {
      set_row_col(i, 0, 0);
    }
  }

  void set_row_col(int i, uint8_t row, uint16_t col) {
    colsb[i] = col;
    rows[i] = row;
  }

  void set_config() { _tile_loadconfig(this); }

  static void load_data(int to, void *from, size_t stride) {
    switch (to) {
    case 0:
      _tile_loadd(0, from, stride);
      break;
    case 1:
      _tile_loadd(1, from, stride);
      break;
    case 2:
      _tile_loadd(2, from, stride);
      break;
    case 3:
      _tile_loadd(3, from, stride);
      break;
    case 4:
      _tile_loadd(4, from, stride);
      break;
    case 5:
      _tile_loadd(5, from, stride);
      break;
    case 6:
      _tile_loadd(6, from, stride);
      break;
    case 7:
      _tile_loadd(7, from, stride);
      break;
    default:
      throw std::runtime_error("no such tile");
    }
  }

  static void store_data(int from, void *to, size_t stride) {
    switch (from) {
    case 0:
      _tile_stored(0, to, stride);
      break;
    case 1:
      _tile_stored(1, to, stride);
      break;
    case 2:
      _tile_stored(2, to, stride);
      break;
    case 3:
      _tile_stored(3, to, stride);
      break;
    case 4:
      _tile_stored(4, to, stride);
      break;
    case 5:
      _tile_stored(5, to, stride);
      break;
    case 6:
      _tile_stored(6, to, stride);
      break;
    case 7:
      _tile_stored(7, to, stride);
      break;
    default:
      throw std::runtime_error("no such tile");
    }
  }
};

static_assert(sizeof(TileConfig) == 64);

inline void debug_tile(int t) {
  printf("Tile %d\n", t);
  uint8_t data[16][64] = {};
  TileConfig::store_data(t, data, 64);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 64; j++) {
      printf("%3d ", data[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

inline void debug_tiles(int to = 8) {
  for (int i = 0; i < to; i++) {
    debug_tile(i);
  }
}

inline void debug_m512(__m512 x) {
  float data[16];
  _mm512_storeu_ps(data, x);
  for (int i = 0; i < 16; i++) {
    printf("%f ", data[i]);
  }
  printf("\n");
}

// transpose utils
inline void transpose_16x16_32bit(__m512i *v) {
  __m512i v1[16];
  v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
  v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
  v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
  v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
  v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
  v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
  v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
  v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
  v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

  v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
  v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
  v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
  v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
  v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
  v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
  v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
  v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
  v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
  v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
  v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
  v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
  v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
  v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
  v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
  v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
  v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
  v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
  v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
  v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
  v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
  v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
  v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
  v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
  v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
  v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
  v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
  v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
  v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
  v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
  v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

  v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

struct GemmKernel224BF {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 2;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 3584;
  static std::string name() { return "BF16"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 32
    for (int i = 0; i < 2; i++)
      tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 32
    for (int i = 2; i < 4; i++)
      tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
    for (int i = 4; i < 8; i++)
      tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
  }

  static void load_a(dt *a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b(dt *b, size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t *c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void store_c(output_t *c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void run_tile() {
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(5, 0, 3);
    _tile_dpbf16ps(6, 1, 2);
    _tile_dpbf16ps(7, 1, 3);
  }

  struct BufferA {
    ggml_bf16_t *a;
    int max_m, k;

    static size_t required_size(int max_m, int k) { return sizeof(ggml_bf16_t) * max_m * k; }

    BufferA(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(k % K_STEP == 0);
      a = reinterpret_cast<ggml_bf16_t *>(ptr);
    }

    void from_mat(int m, ggml_bf16_t *src, int ith, int nth) {
      assert(m <= max_m);
      assert(ith == 0 && nth == 1);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
              __m512i *s = (__m512i *)(src + (m_begin + i) * k + k_block_begin + k_begin);
              __m512i *d = (__m512i *)(a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP +
                                       i * K_STEP);
              avx512_copy_32xbf16(s, d);
            }
          }
        }
      }
    }

    ggml_bf16_t *get_submat(int m, int k, int m_begin, int k_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
    }
  };

  struct BufferB {
    ggml_bf16_t *b;
    int n, k;
    static constexpr bool SCALE = false;

    static size_t required_size(int n, int k) { return sizeof(ggml_bf16_t) * n * k; }

    BufferB(int n, int k, void *ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      b = reinterpret_cast<ggml_bf16_t *>(ptr);
    }

    void from_mat(ggml_bf16_t *src, int ith, int nth) {
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              __m512i *s = (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin);
              __m512i *d = (__m512i *)(b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                       k_begin * N_STEP + i * K_STEP);
              avx512_copy_32xbf16(s, d);
            }
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k + k_block_begin * n_block_size +
                                              n_begin * k_block_size + k_begin * N_STEP));
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k + k_block_begin * n_block_size +
                                              n_begin * k_block_size + k_begin * N_STEP + TILE_N * K_STEP));
          }
        }
      }
    }

    ggml_bf16_t *get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP;
    }
  };

  struct BufferC {
    float *c;
    int max_m, n;

    static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

    BufferC(int max_m, int n, void *ptr) : max_m(max_m), n(n) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(n % N_STEP == 0);
      c = reinterpret_cast<float *>(ptr);
    }

    void to_mat(int m, ggml_bf16_t *dst, int ith, int nth) {
      assert(m <= max_m);
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
        for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
          for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
            __m512 *x0 =
                (__m512 *)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
            __m512 *x1 = (__m512 *)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP +
                                    i * N_STEP + 16);
            avx512_32xfp32_to_32xbf16(x0, x1, (__m512i *)(dst + (m_begin + i) * n + n_block_begin + n_begin));
          }
        }
      }
    }

    float *get_submat(int m, int n, int m_begin, int n_begin) {
      int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      n_begin -= n_block_begin;
      return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
    }
  };

  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    using K = GemmKernel224BF;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c(c, K::N_STEP * sizeof(float));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(ggml_bf16_t));
      K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::K_STEP * sizeof(ggml_bf16_t));
      K::run_tile();
    }
    K::store_c(c, K::N_STEP * sizeof(float));
  }

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    using K = GemmKernel224BF;
    __m512 *c512 = (__m512 *)c;
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        c512[m_i * 2] = _mm512_setzero_ps();
        c512[m_i * 2 + 1] = _mm512_setzero_ps();
      }
    }

    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
      int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512bh *b512 = (__m512bh *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            c512[m_i * 2 + n_i] = _mm512_dpbf16_ps(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
          }
        }
      }
    }
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float *c, BufferA *ba, BufferB *bb) {}
};

template <typename K> struct BufferAImpl {
  int8_t *a;
  float *d;
  int max_m, k;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int K_BLOCK = K::K_BLOCK;

  static size_t required_size(int max_m, int k) { return sizeof(int8_t) * max_m * k + sizeof(float) * max_m; }

  BufferAImpl(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(k % K_STEP == 0);
    if (max_m % M_STEP || k % K_STEP) {
      printf("max_m = %d, k = %d, M_STEP = %d, K_STEP = %d\n", max_m, k, M_STEP, K_STEP);
      throw std::runtime_error("BufferAImpl: max_m and k must be multiple of M_STEP and K_STEP");
    }
    a = reinterpret_cast<int8_t *>(ptr);
    d = reinterpret_cast<float *>(a + max_m * k);
  }

  void from_mat(int m, ggml_bf16_t *src, int ith, int nth) {
    assert(m <= max_m);
    assert(ith == 0 && nth == 1);
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
        float amax = 0.0f;
        for (int j = 0; j < k; j += 32) {
          __m512 f0, f1;
          avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k + j), &f0, &f1);
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
            int8_t *dst = a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP + i * K_STEP;
            __m512 f0, f1, f2, f3;
            avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
            avx512_32xbf16_to_32xfp32((__m512i *)(src + (m_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
            __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
            __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
            __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
            __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
            __m128i s0 = _mm512_cvtsepi32_epi8(i0);
            __m128i s1 = _mm512_cvtsepi32_epi8(i1);
            __m128i s2 = _mm512_cvtsepi32_epi8(i2);
            __m128i s3 = _mm512_cvtsepi32_epi8(i3);
            _mm_store_si128((__m128i *)dst, s0);
            _mm_store_si128((__m128i *)(dst + 16), s1);
            _mm_store_si128((__m128i *)(dst + 32), s2);
            _mm_store_si128((__m128i *)(dst + 48), s3);
          }
        }
      }
    }
  }

  int8_t *get_submat(int m, int k, int m_begin, int k_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
    k_begin -= k_block_begin;
    int k_block_size = std::min(K_BLOCK, k - k_block_begin);
    return a + k_block_begin * m_block_size + m_begin * k_block_size + k_begin * M_STEP;
  }

  float *get_scale(int m, int m_begin) { return d + m_begin; }
};

template <typename K> struct BufferCImpl {
  float *c;
  int max_m, n;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;
  static constexpr int N_BLOCK = K::N_BLOCK;

  static size_t required_size(int max_m, int n) { return sizeof(float) * max_m * n; }

  BufferCImpl(int max_m, int n, void *ptr) : max_m(max_m), n(n) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    assert(max_m % M_STEP == 0);
    assert(n % N_STEP == 0);
    if (max_m % M_STEP || n % N_STEP) {
      printf("max_m = %d, n = %d, M_STEP = %d, N_STEP = %d\n", max_m, n, M_STEP, N_STEP);
      throw std::runtime_error("BufferCImpl: max_m and n must be multiple of M_STEP and N_STEP");
    }
    c = reinterpret_cast<float *>(ptr);
  }

  void to_mat(int m, ggml_bf16_t *dst, int ith, int nth) {
    assert(m <= max_m);
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_start;
    int n_block_size = n_end - n_block_begin;
    for (int m_begin = 0; m_begin < m; m_begin += M_STEP) {
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < M_STEP && m_begin + i < m; i++) {
          __m512 *x0 =
              (__m512 *)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP + i * N_STEP);
          __m512 *x1 = (__m512 *)(c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP +
                                  i * N_STEP + 16);
          avx512_32xfp32_to_32xbf16(x0, x1, (__m512i *)(dst + (m_begin + i) * n + n_block_begin + n_begin));
        }
      }
    }
  }

  float *get_submat(int m, int n, int m_begin, int n_begin) {
    int m_block_size = (m + M_STEP - 1) / M_STEP * M_STEP;
    int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
    int n_block_size = std::min(N_BLOCK, n - n_block_begin);
    n_begin -= n_block_begin;
    return c + m_block_size * n_block_begin + m_begin * n_block_size + n_begin * M_STEP;
  }
};

inline __m512i _mm512_dpbssd_epi32(__m512i src, __m512i a, __m512i b) {
  __m256i a_lo = _mm512_extracti64x4_epi64(a, 0);
  __m256i a_hi = _mm512_extracti64x4_epi64(a, 1);
  __m256i b_lo = _mm512_extracti64x4_epi64(b, 0);
  __m256i b_hi = _mm512_extracti64x4_epi64(b, 1);

  b_lo = _mm256_sign_epi8(b_lo, a_lo);
  b_hi = _mm256_sign_epi8(b_hi, a_hi);

  b = _mm512_inserti64x4(b, b_lo, 0);
  b = _mm512_inserti64x4(b, b_hi, 1);

  a = _mm512_abs_epi8(a);

  return _mm512_dpbusd_epi32(src, a, b);
}

struct GemmKernel224Int8 {
  using dt = int8_t;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 1;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 3584;
  static std::string name() { return "INT8"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 64
    for (int i = 0; i < 2; i++)
      tile_config.set_row_col(i, TILE_M, TILE_K * sizeof(dt));

    // size is 16 x 64
    for (int i = 2; i < 4; i++)
      tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK * sizeof(dt));

    // size is 16 x 16
    for (int i = 4; i < 8; i++)
      tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
  }

  static void load_a(dt *a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  static void load_b(dt *b, size_t ldb) {
    _tile_loadd(2, b, ldb);
    _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t *c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void store_c(output_t *c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void run_tile() {
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
  }

  using BufferA = BufferAImpl<GemmKernel224Int8>;
  using BufferC = BufferCImpl<GemmKernel224Int8>;

  struct BufferB {
    int8_t *b;
    float *d;
    int n, k;
    static constexpr bool SCALE = true;

    static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k + sizeof(float) * n; }

    BufferB(int n, int k, void *ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      if (n % N_STEP || k % K_STEP) {
        printf("n: %d, k: %d, N_STEP: %d, K_STEP: %d\n", n, k, N_STEP, K_STEP);
        throw std::runtime_error("BufferB: n and k must be multiples of N_STEP and K_STEP");
      }
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + n * k);
    }

    void from_mat(ggml_bf16_t *src, int ith, int nth) { // CHECK: nth has no usage
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP; i++) {
          float amax = 0.0f;
          for (int j = 0; j < k; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i *)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[n_block_begin + n_begin + i] = amax / ((1 << 7) - 1);
        }
      }
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              __m512 id = _mm512_set1_ps(d[n_block_begin + n_begin + i] ? 1.0f / d[n_block_begin + n_begin + i] : 0.0f);
              int8_t *dst = b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                            k_begin * N_STEP + i * K_STEP;
              __m512 f0, f1, f2, f3;
              avx512_32xbf16_to_32xfp32((__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin),
                                        &f0, &f1);
              avx512_32xbf16_to_32xfp32(
                  (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
              __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
              __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
              __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
              __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
              __m128i s0 = _mm512_cvtsepi32_epi8(i0);
              __m128i s1 = _mm512_cvtsepi32_epi8(i1);
              __m128i s2 = _mm512_cvtsepi32_epi8(i2);
              __m128i s3 = _mm512_cvtsepi32_epi8(i3);
              _mm_store_si128((__m128i *)dst, s0);
              _mm_store_si128((__m128i *)(dst + 16), s1);
              _mm_store_si128((__m128i *)(dst + 32), s2);
              _mm_store_si128((__m128i *)(dst + 48), s3);
            }
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k + k_block_begin * n_block_size +
                                              n_begin * k_block_size + k_begin * N_STEP));
            transpose_16x16_32bit((__m512i *)(b + n_block_begin * k + k_block_begin * n_block_size +
                                              n_begin * k_block_size + k_begin * N_STEP + TILE_N * K_STEP));
          }
        }
      }
    }

    int8_t *get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return b + n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP;
    }

    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };

  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    using K = GemmKernel224Int8;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::load_b(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::run_tile();
    }
    K::store_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
  }

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    __m512i *c512 = (__m512i *)c;
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }

    for (int k_begin = 0; k_begin < K_BLOCK && k_block_begin + k_begin < k; k_begin += K_STEP) {
      static_assert(K_STEP * sizeof(int8_t) == sizeof(__m512i));
      static_assert(N_STEP / TILE_N == 2, "Must be lke this");

      int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      __m512i *b512 = (__m512i *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma = _mm512_set1_epi32(a32[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
          }
        }
      }
    }
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float *c, BufferA *ba, BufferB *bb) {
    using K = GemmKernel224Int8;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i));
      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
      __m512i now = _mm512_load_si512((__m512i *)(c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512 *)(c + i * K::N_STEP), result);
      bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i *)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512 *)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

struct GemmKernel224Int4 {
  using dt = void;
  using output_t = int32_t;
  static constexpr double ELEMENT_SIZE = 0.5;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static const int M_STEP = TILE_M * 2;
  static const int N_STEP = TILE_N * 2;
  static const int K_STEP = TILE_K;

  static inline const int N_BLOCK = 128;
  static inline const int K_BLOCK = 3584;
  static std::string name() { return "INT4"; }

  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }

  static void config() {
    enable_amx();
    TileConfig tile_config;

    // size is 16 x 64
    for (int i = 0; i < 2; i++)
      tile_config.set_row_col(i, TILE_M, TILE_K);

    // size is 16 x 64
    for (int i = 2; i < 4; i++)
      tile_config.set_row_col(i, TILE_K / VNNI_BLK, TILE_N * VNNI_BLK);

    // size is 16 x 16
    for (int i = 4; i < 8; i++)
      tile_config.set_row_col(i, TILE_M, TILE_N * sizeof(output_t));

    tile_config.set_config();
  }

  alignas(64) static constexpr uint8_t hi_mask_arr[64] = {
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0,
      0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0, 0xF0};

  alignas(64) static constexpr uint8_t lo_mask_arr[64] = {
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F,
      0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F};

  alignas(64) static constexpr uint8_t sign_mask_arr[64] = {
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
      0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
  };

  static __m512i hi_mask() { return *((__m512i *)(&hi_mask_arr[0])); }
  static __m128i hi_mask_128() { return *((__m128i *)(&hi_mask_arr[0])); }
  static __m512i lo_mask() { return *((__m512i *)(&lo_mask_arr[0])); }
  static __m128i lo_mask_128() { return *((__m128i *)(&lo_mask_arr[0])); }
  static __m128i si_mask_128() { return *((__m128i *)(&sign_mask_arr[0])); }

  // static void load_a_hi(dt *a, size_t lda) {
  //   // 在函数内部分配一个局部(栈上)对齐缓冲区
  //   alignas(64) int8_t local_buffer[TILE_M * TILE_K];
  //   // 用 db 指向这块缓冲区，方便后面使用
  //   __m512i *db = reinterpret_cast<__m512i *>(local_buffer);

  //   // 先加载前半部分 (i 从 0 到 TILE_M-1)
  //   for (size_t i = 0; i < TILE_M; i++) {
  //     __m512i tmp = _mm512_and_si512(hi_mask(), *static_cast<__m512i *>(offset_pointer(a, lda * i)));
  //     _mm512_store_si512(&db[i], tmp);
  //   }
  //   asm volatile("" ::: "memory");
  //   // 将 db 对应的数据加载到 tile 寄存器 0
  //   _tile_loadd(0, db, TILE_K);

  //   // 加载后半部分 (从 a 的下一个 TILE_M 行开始)
  //   for (size_t i = 0; i < TILE_M; i++) {
  //     db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i *>(offset_pointer(a, lda * (i + TILE_M))));
  //   }
  //   asm volatile("" ::: "memory");
  //   // 将新的内容加载到 tile 寄存器 1
  //   _tile_loadd(1, db, TILE_K);
  // }

  // static void load_a_lo(dt *a, size_t lda) {
  //   // 在函数内部分配一个局部(栈上)对齐缓冲区
  //   alignas(64) int8_t local_buffer[TILE_M * TILE_K];
  //   __m512i *db = reinterpret_cast<__m512i *>(local_buffer);

  //   for (size_t i = 0; i < TILE_M; i++) {
  //     __m512i tmp =
  //         _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(a, lda * i))), 4);
  //     _mm512_store_si512(&db[i], tmp);
  //   }
  //   asm volatile("" ::: "memory");
  //   _tile_loadd(0, db, TILE_K);

  //   for (size_t i = 0; i < TILE_M; i++) {
  //     db[i] = _mm512_slli_epi32(
  //         _mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(a, lda * (i + TILE_M)))), 4);
  //   }
  //   asm volatile("" ::: "memory");
  //   _tile_loadd(1, db, TILE_K);
  // }

  static void load_b_hi(dt *b, size_t ldb) {
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i *db = reinterpret_cast<__m512i *>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * i)));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_and_si512(hi_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * (i + TILE_N))));
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
  }

  static void load_b_lo(dt *b, size_t ldb) {
    // 在函数内部分配一个局部(栈上)对齐缓冲区
    alignas(64) int8_t local_buffer[TILE_N * TILE_K];
    __m512i *db = reinterpret_cast<__m512i *>(local_buffer);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(_mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * i))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(2, db, TILE_K);

    for (size_t i = 0; i < TILE_N; i++) {
      db[i] = _mm512_slli_epi32(
          _mm512_and_si512(lo_mask(), *static_cast<__m512i *>(offset_pointer(b, ldb * (i + TILE_N)))), 4);
    }
    asm volatile("" ::: "memory");
    _tile_loadd(3, db, TILE_K);
  }

  static void load_a(dt *a, size_t lda) {
    _tile_loadd(0, a, lda);
    _tile_loadd(1, offset_pointer(a, lda * TILE_M), lda);
  }

  // static void load_b(dt* b, size_t ldb) {
  //   _tile_loadd(2, b, ldb);
  //   _tile_loadd(3, offset_pointer(b, ldb * TILE_N), ldb);
  // }

  static void clean_c() {
    _tile_zero(4);
    _tile_zero(5);
    _tile_zero(6);
    _tile_zero(7);
  }

  static void load_c(output_t *c, size_t ldc) {
    _tile_loadd(4, c, ldc);
    _tile_loadd(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_loadd(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_loadd(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void store_c(output_t *c, size_t ldc) {
    _tile_stored(4, c, ldc);
    _tile_stored(5, offset_pointer(c, TILE_N * sizeof(output_t)), ldc);
    _tile_stored(6, offset_pointer(c, ldc * TILE_M), ldc);
    _tile_stored(7, offset_pointer(c, ldc * TILE_M + TILE_N * sizeof(output_t)), ldc);
  }

  static void run_tile() {
    _tile_dpbssd(4, 0, 2);
    _tile_dpbssd(5, 0, 3);
    _tile_dpbssd(6, 1, 2);
    _tile_dpbssd(7, 1, 3);
  }

  using BufferA = BufferAImpl<GemmKernel224Int4>;
  using BufferC = BufferCImpl<GemmKernel224Int4>;

  struct BufferB {
    dt *b;
    float *d;
    int n, k;
    static const int B_K_STEP = 2 * K_STEP;
    static constexpr bool SCALE = true;

    static size_t required_size(int n, int k) { return sizeof(int8_t) * n * k / 2 + sizeof(float) * n; }

    BufferB(int n, int k, void *ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % B_K_STEP == 0);
      if (n % N_STEP || k % B_K_STEP) {
        printf("n: %d, k: %d, N_STEP: %d, B_K_STEP: %d\n", n, k, N_STEP, B_K_STEP);
        throw std::runtime_error("n or k is not aligned to N_STEP or B_K_STEP");
      }
      b = reinterpret_cast<dt *>(ptr);
      d = reinterpret_cast<float *>(offset_pointer(b, n * k / 2));
    }

    static __m128i round_up4(__m128i x) {
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

    void from_mat(ggml_bf16_t *src, int ith, int nth) {
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      int n_block_begin = n_start;
      int n_block_size = n_end - n_block_begin;
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int i = 0; i < N_STEP; i++) {
          float amax = 0.0f;
          for (int j = 0; j < k; j += 32) {
            __m512 f0, f1;
            avx512_32xbf16_to_32xfp32((__m512i *)(src + (n_block_begin + n_begin + i) * k + j), &f0, &f1);
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f0)));
            amax = MAX(amax, _mm512_reduce_max_ps(_mm512_abs_ps(f1)));
          }
          d[n_block_begin + n_begin + i] = amax / 112.0; // 7*16
        }
      }
      for (int n_begin = 0; n_begin < n_block_size; n_begin += N_STEP) {
        for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K_BLOCK) {
          int k_block_size = std::min(K_BLOCK, k - k_block_begin);
          for (int k_begin = 0; k_begin < k_block_size; k_begin += B_K_STEP) {
            for (int i = 0; i < N_STEP; i++) {
              __m512 id = _mm512_set1_ps(d[n_block_begin + n_begin + i] ? 1.0f / d[n_block_begin + n_begin + i] : 0.0f);
              dt *dst = offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size +
                                           k_begin * N_STEP + i * B_K_STEP) /
                                              2);
              {
                __m512 f0, f1, f2, f3;
                avx512_32xbf16_to_32xfp32(
                    (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin), &f0, &f1);
                avx512_32xbf16_to_32xfp32(
                    (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 1, &f2, &f3);
                __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
                __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
                __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
                __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
                __m128i s0 = _mm512_cvtsepi32_epi8(i0);
                __m128i s1 = _mm512_cvtsepi32_epi8(i1);
                __m128i s2 = _mm512_cvtsepi32_epi8(i2);
                __m128i s3 = _mm512_cvtsepi32_epi8(i3);
                s0 = _mm_srli_epi16(round_up4(s0), 4);
                s1 = _mm_srli_epi16(round_up4(s1), 4);
                s2 = _mm_srli_epi16(round_up4(s2), 4);
                s3 = _mm_srli_epi16(round_up4(s3), 4);
                // s0 = _mm_or_si128(round_up4(s0), _mm_srli_epi16(round_up4(s1), 4));
                // s2 = _mm_or_si128(round_up4(s2), _mm_srli_epi16(round_up4(s3), 4));
                _mm_store_si128((__m128i *)dst, s0);
                _mm_store_si128((__m128i *)(offset_pointer(dst, 16)), s1);
                _mm_store_si128((__m128i *)(offset_pointer(dst, 32)), s2);
                _mm_store_si128((__m128i *)(offset_pointer(dst, 48)), s3);
              }

              {
                __m512 f0, f1, f2, f3;
                avx512_32xbf16_to_32xfp32(
                    (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 2, &f0, &f1);
                avx512_32xbf16_to_32xfp32(
                    (__m512i *)(src + (n_block_begin + n_begin + i) * k + k_block_begin + k_begin) + 3, &f2, &f3);
                __m512i i0 = _mm512_cvtps_epi32(_mm512_mul_ps(f0, id));
                __m512i i1 = _mm512_cvtps_epi32(_mm512_mul_ps(f1, id));
                __m512i i2 = _mm512_cvtps_epi32(_mm512_mul_ps(f2, id));
                __m512i i3 = _mm512_cvtps_epi32(_mm512_mul_ps(f3, id));
                __m128i s0 = _mm512_cvtsepi32_epi8(i0);
                __m128i s1 = _mm512_cvtsepi32_epi8(i1);
                __m128i s2 = _mm512_cvtsepi32_epi8(i2);
                __m128i s3 = _mm512_cvtsepi32_epi8(i3);
                s0 = round_up4(s0);
                s1 = round_up4(s1);
                s2 = round_up4(s2);
                s3 = round_up4(s3);
                _mm_store_si128((__m128i *)(offset_pointer(dst, 0)),
                                _mm_or_si128(_mm_loadu_si128((__m128i *)(offset_pointer(dst, 0))), s0));
                _mm_store_si128((__m128i *)(offset_pointer(dst, 16)),
                                _mm_or_si128(_mm_loadu_si128((__m128i *)(offset_pointer(dst, 16))), s1));
                _mm_store_si128((__m128i *)(offset_pointer(dst, 32)),
                                _mm_or_si128(_mm_loadu_si128((__m128i *)(offset_pointer(dst, 32))), s2));
                _mm_store_si128((__m128i *)(offset_pointer(dst, 48)),
                                _mm_or_si128(_mm_loadu_si128((__m128i *)(offset_pointer(dst, 48))), s3));
              }
            }
            transpose_16x16_32bit((__m512i *)(offset_pointer(
                b,
                (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2)));
            transpose_16x16_32bit(
                (__m512i *)(offset_pointer(b, (n_block_begin * k + k_block_begin * n_block_size +
                                               n_begin * k_block_size + k_begin * N_STEP + TILE_N * B_K_STEP) /
                                                  2)));
          }
        }
      }
    }

    dt *get_submat(int n, int k, int n_begin, int k_begin) {
      int n_block_begin = n_begin / N_BLOCK * N_BLOCK;
      n_begin -= n_block_begin;
      int n_block_size = std::min(N_BLOCK, n - n_block_begin);
      int k_block_begin = k_begin / K_BLOCK * K_BLOCK;
      k_begin -= k_block_begin;
      int k_block_size = std::min(K_BLOCK, k - k_block_begin);
      return offset_pointer(
          b, (n_block_begin * k + k_block_begin * n_block_size + n_begin * k_block_size + k_begin * N_STEP) / 2);
    }

    float *get_scale(int n, int n_begin) { return d + n_begin; }
  };

  static void amx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    using K = GemmKernel224Int4;
    if (k_block_begin == 0) {
      K::clean_c();
    } else {
      K::load_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
    }
    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      // printf("offset a %ld\n", pointer_offset(ba->get_submat(m, k, m_begin, k_block_begin + k_begin),
      // ba->a)); printf("offset b %ld\n", pointer_offset(bb->get_submat(n, k, n_begin, k_block_begin +
      // k_begin), bb->b));
      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin), K::K_STEP * sizeof(int8_t));
      K::load_b_lo(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();

      K::load_a(ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP), K::K_STEP * sizeof(int8_t));
      K::load_b_hi(bb->get_submat(n, k, n_begin, k_block_begin + k_begin), K::BufferB::B_K_STEP / 2);
      K::run_tile();
    }

    // debug_tiles_224();
    K::store_c((int32_t *)c, K::N_STEP * sizeof(int32_t));
  }

  static void avx_kernel(int m, int n, int k, int m_begin, int n_begin, int k_block_begin, float *c, BufferA *ba,
                         BufferB *bb) {
    using K = GemmKernel224Int4;
    __m512i *c512 = (__m512i *)c;
    if (k_block_begin == 0) {
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        c512[m_i * 2] = _mm512_setzero_si512();
        c512[m_i * 2 + 1] = _mm512_setzero_si512();
      }
    }

    for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::BufferB::B_K_STEP) {
      int32_t *a32_lo = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
      int32_t *a32_hi = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin + K::K_STEP);
      __m512i *b512 = (__m512i *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
      for (int m_i = 0; m_i < m && m_i < M_STEP; m_i++) {
        for (int k_i = 0; k_i < 16; k_i++) {
          __m512i ma_lo = _mm512_set1_epi32(a32_lo[m_i * 16 + k_i]);
          __m512i ma_hi = _mm512_set1_epi32(a32_hi[m_i * 16 + k_i]);
          for (int n_i = 0; n_i < 2; n_i++) {
            __m512i b512_lo = _mm512_slli_epi32(_mm512_and_si512(K::lo_mask(), b512[n_i * 16 + k_i]), 4);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_lo, b512_lo);
            __m512i b512_hi = _mm512_and_si512(K::hi_mask(), b512[n_i * 16 + k_i]);
            c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma_hi, b512_hi);
          }
        }
      }
    }
  }

  static void apply_scale(int m, int n, int m_begin, int n_begin, float *c, BufferA *ba, BufferB *bb) {
    using K = GemmKernel224Int4;
    int to = m - m_begin;
    if (m - m_begin > K::M_STEP) {
      to = K::M_STEP;
    }
    for (int i = 0; i < to; i++) {
      __m512 as = _mm512_set1_ps(*ba->get_scale(m, m_begin + i));
      __m512 bs = _mm512_load_ps(bb->get_scale(n, n_begin));
      __m512i now = _mm512_load_epi32((__m512i *)(c + i * K::N_STEP));
      __m512 result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512 *)(c + i * K::N_STEP), result);
      bs = _mm512_load_ps(bb->get_scale(n, n_begin) + K::TILE_N);
      now = _mm512_load_si512((__m512i *)(c + i * K::N_STEP + K::TILE_N));
      result = _mm512_mul_ps(_mm512_mul_ps(as, bs), _mm512_cvtepi32_ps(now));
      _mm512_store_ps((__m512 *)(c + i * K::N_STEP + K::TILE_N), result);
    }
  }
};

// This is for breakdown analysis
static const int USE_AMX_THRESHOLD = []() {
  const char *env_val = std::getenv("USE_AMX_THRESHOLD");
  if (env_val) {
    return std::atoi(env_val);
  }
  return 5;
}();

template <typename K>
void mat_mul(int m, int n, int k, typename K::BufferA *ba, typename K::BufferB *bb, typename K::BufferC *bc, int ith,
             int nth) {
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float *c = bc->get_submat(m, n, m_begin, n_begin);
        if (m >= USE_AMX_THRESHOLD) {
          K::amx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        } else {
          K::avx_kernel(m, n, k, m_begin, n_begin, k_block_begin, c, ba, bb);
        }

        if (k_block_begin + K::K_BLOCK >= k) {
          K::apply_scale(m, n, m_begin, n_begin, c, ba, bb);
        }
      }
    }
  }
}

} // namespace amx