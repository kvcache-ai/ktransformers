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

/*
  Transpose 16x16 32-bit elements
  Note that v must be 64 byte aligned
*/
inline void transpose_16x16_32bit(__m512i *v, size_t stride) {
  assert(reinterpret_cast<intptr_t>(v) % 64 == 0 && "v must be 64 aligned");

  auto stride_v = [=](int i) { return offset_pointer(v, i * stride); };
  __m512i v1[16];

  v1[0] = _mm512_unpacklo_epi32(*stride_v(0), *stride_v(1));
  v1[1] = _mm512_unpackhi_epi32(*stride_v(0), *stride_v(1));
  v1[2] = _mm512_unpacklo_epi32(*stride_v(2), *stride_v(3));
  v1[3] = _mm512_unpackhi_epi32(*stride_v(2), *stride_v(3));
  v1[4] = _mm512_unpacklo_epi32(*stride_v(4), *stride_v(5));
  v1[5] = _mm512_unpackhi_epi32(*stride_v(4), *stride_v(5));
  v1[6] = _mm512_unpacklo_epi32(*stride_v(6), *stride_v(7));
  v1[7] = _mm512_unpackhi_epi32(*stride_v(6), *stride_v(7));
  v1[8] = _mm512_unpacklo_epi32(*stride_v(8), *stride_v(9));
  v1[9] = _mm512_unpackhi_epi32(*stride_v(8), *stride_v(9));
  v1[10] = _mm512_unpacklo_epi32(*stride_v(10), *stride_v(11));
  v1[11] = _mm512_unpackhi_epi32(*stride_v(10), *stride_v(11));
  v1[12] = _mm512_unpacklo_epi32(*stride_v(12), *stride_v(13));
  v1[13] = _mm512_unpackhi_epi32(*stride_v(12), *stride_v(13));
  v1[14] = _mm512_unpacklo_epi32(*stride_v(14), *stride_v(15));
  v1[15] = _mm512_unpackhi_epi32(*stride_v(14), *stride_v(15));

  *stride_v(0) = _mm512_unpacklo_epi64(v1[0], v1[2]);
  *stride_v(1) = _mm512_unpackhi_epi64(v1[0], v1[2]);
  *stride_v(2) = _mm512_unpacklo_epi64(v1[1], v1[3]);
  *stride_v(3) = _mm512_unpackhi_epi64(v1[1], v1[3]);
  *stride_v(4) = _mm512_unpacklo_epi64(v1[4], v1[6]);
  *stride_v(5) = _mm512_unpackhi_epi64(v1[4], v1[6]);
  *stride_v(6) = _mm512_unpacklo_epi64(v1[5], v1[7]);
  *stride_v(7) = _mm512_unpackhi_epi64(v1[5], v1[7]);
  *stride_v(8) = _mm512_unpacklo_epi64(v1[8], v1[10]);
  *stride_v(9) = _mm512_unpackhi_epi64(v1[8], v1[10]);
  *stride_v(10) = _mm512_unpacklo_epi64(v1[9], v1[11]);
  *stride_v(11) = _mm512_unpackhi_epi64(v1[9], v1[11]);
  *stride_v(12) = _mm512_unpacklo_epi64(v1[12], v1[14]);
  *stride_v(13) = _mm512_unpackhi_epi64(v1[12], v1[14]);
  *stride_v(14) = _mm512_unpacklo_epi64(v1[13], v1[15]);
  *stride_v(15) = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(*stride_v(0), *stride_v(4), 0x88);
  v1[1] = _mm512_shuffle_i32x4(*stride_v(1), *stride_v(5), 0x88);
  v1[2] = _mm512_shuffle_i32x4(*stride_v(2), *stride_v(6), 0x88);
  v1[3] = _mm512_shuffle_i32x4(*stride_v(3), *stride_v(7), 0x88);
  v1[4] = _mm512_shuffle_i32x4(*stride_v(0), *stride_v(4), 0xdd);
  v1[5] = _mm512_shuffle_i32x4(*stride_v(1), *stride_v(5), 0xdd);
  v1[6] = _mm512_shuffle_i32x4(*stride_v(2), *stride_v(6), 0xdd);
  v1[7] = _mm512_shuffle_i32x4(*stride_v(3), *stride_v(7), 0xdd);
  v1[8] = _mm512_shuffle_i32x4(*stride_v(8), *stride_v(12), 0x88);
  v1[9] = _mm512_shuffle_i32x4(*stride_v(9), *stride_v(13), 0x88);
  v1[10] = _mm512_shuffle_i32x4(*stride_v(10), *stride_v(14), 0x88);
  v1[11] = _mm512_shuffle_i32x4(*stride_v(11), *stride_v(15), 0x88);
  v1[12] = _mm512_shuffle_i32x4(*stride_v(8), *stride_v(12), 0xdd);
  v1[13] = _mm512_shuffle_i32x4(*stride_v(9), *stride_v(13), 0xdd);
  v1[14] = _mm512_shuffle_i32x4(*stride_v(10), *stride_v(14), 0xdd);
  v1[15] = _mm512_shuffle_i32x4(*stride_v(11), *stride_v(15), 0xdd);

  *stride_v(0) = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  *stride_v(1) = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  *stride_v(2) = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  *stride_v(3) = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  *stride_v(4) = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  *stride_v(5) = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  *stride_v(6) = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  *stride_v(7) = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  *stride_v(8) = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  *stride_v(9) = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  *stride_v(10) = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  *stride_v(11) = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  *stride_v(12) = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  *stride_v(13) = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  *stride_v(14) = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  *stride_v(15) = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

struct GemmKernel224BF {
  using dt = ggml_bf16_t;
  using output_t = float;
  static const int TILE_M = 16;
  static const int TILE_K = 32;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 2;

  static inline constexpr int M_STEP = TILE_M * 2;
  static inline constexpr int N_STEP = TILE_N * 2;
  static inline constexpr int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 1792;

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

    static size_t required_size(int max_m, int k) { return max_m * k * sizeof(ggml_bf16_t); }

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

    static size_t required_size(int n, int k) { return n * k * sizeof(ggml_bf16_t); }

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

    static size_t required_size(int max_m, int n) { return max_m * n * sizeof(float); }

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
};

struct GemmKernel224Int8 {
  using dt = int8_t;
  using output_t = int32_t;
  static const int TILE_M = 16;
  static const int TILE_K = 64;
  static const int TILE_N = 16;
  static const int VNNI_BLK = 4;

  static inline constexpr int M_STEP = TILE_M * 2;
  static inline constexpr int N_STEP = TILE_N * 2;
  static inline constexpr int K_STEP = TILE_K;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 3584;

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

  struct BufferA {
    int8_t *a;
    float *d;
    int max_m, k;

    static size_t required_size(int max_m, int k) { return max_m * k * sizeof(int8_t) + max_m * sizeof(float); }

    BufferA(int max_m, int k, void *ptr) : max_m(max_m), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(max_m % M_STEP == 0);
      assert(k % K_STEP == 0);
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
              _mm_storeu_si128((__m128i *)dst, s0);
              _mm_storeu_si128((__m128i *)(dst + 16), s1);
              _mm_storeu_si128((__m128i *)(dst + 32), s2);
              _mm_storeu_si128((__m128i *)(dst + 48), s3);
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

  struct BufferB {
    int8_t *b;
    float *d;
    int n, k;

    static size_t required_size(int n, int k) { return n * k * sizeof(int8_t) + n * sizeof(float); }

    BufferB(int n, int k, void *ptr) : n(n), k(k) {
      assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
      assert(n % N_STEP == 0);
      assert(k % K_STEP == 0);
      b = reinterpret_cast<int8_t *>(ptr);
      d = reinterpret_cast<float *>(b + n * k);
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
              _mm_storeu_si128((__m128i *)dst, s0);
              _mm_storeu_si128((__m128i *)(dst + 16), s1);
              _mm_storeu_si128((__m128i *)(dst + 32), s2);
              _mm_storeu_si128((__m128i *)(dst + 48), s3);
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

  struct BufferC {
    float *c;
    int max_m, n;

    static size_t required_size(int max_m, int n) { return max_m * n * sizeof(float); }

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
};

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224BF::BufferA> ba,
                    std::shared_ptr<GemmKernel224BF::BufferB> bb, std::shared_ptr<GemmKernel224BF::BufferC> bc, int ith,
                    int nth, bool use_amx) {
//   std::cout << "mat_mul in BF16!!!!" << std::endl;
  using K = GemmKernel224BF;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {

        float *c = bc->get_submat(m, n, m_begin, n_begin);
        if (!use_amx) {
          __m512 *c512 = (__m512 *)c;
          if (k_block_begin == 0) {
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              c512[m_i * 2] = _mm512_setzero_ps();
              c512[m_i * 2 + 1] = _mm512_setzero_ps();
            }
          }

          for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
            int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
            __m512bh *b512 = (__m512bh *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              for (int k_i = 0; k_i < 16; k_i++) {
                __m512bh ma = (__m512bh)_mm512_set1_epi32(a32[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                  c512[m_i * 2 + n_i] = _mm512_dpbf16_ps(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
                }
              }
            }
          }

        } else {
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
      }
    }
  }
}

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

inline void mat_mul(int m, int n, int k, std::shared_ptr<GemmKernel224Int8::BufferA> ba,
                    std::shared_ptr<GemmKernel224Int8::BufferB> bb, std::shared_ptr<GemmKernel224Int8::BufferC> bc,
                    int ith, int nth, bool use_amx) {
//   std::cout << "mat_mul in INT8!!!!" << std::endl;
  using K = GemmKernel224Int8;
  assert(n % K::N_STEP == 0);
  assert(k % K::K_STEP == 0);

  auto [n_start, n_end] = K::split_range_n(n, ith, nth);

  for (int k_block_begin = 0; k_block_begin < k; k_block_begin += K::K_BLOCK) {
    for (int m_begin = 0; m_begin < m; m_begin += K::M_STEP) {
      for (int n_begin = n_start; n_begin < n_end; n_begin += K::N_STEP) {
        float *c = bc->get_submat(m, n, m_begin, n_begin);

        if (!use_amx) {
          __m512i *c512 = (__m512i *)c;
          if (k_block_begin == 0) {
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              c512[m_i * 2] = _mm512_setzero_si512();
              c512[m_i * 2 + 1] = _mm512_setzero_si512();
            }
          }

          for (int k_begin = 0; k_begin < K::K_BLOCK && k_block_begin + k_begin < k; k_begin += K::K_STEP) {
            static_assert(K::K_STEP * sizeof(int8_t) == sizeof(__m512i));
            static_assert(K::N_STEP / K::TILE_N == 2, "Must be lke this");

            int32_t *a32 = (int32_t *)ba->get_submat(m, k, m_begin, k_block_begin + k_begin);
            __m512i *b512 = (__m512i *)bb->get_submat(n, k, n_begin, k_block_begin + k_begin);
            for (int m_i = 0; m_i < m && m_i < K::M_STEP; m_i++) {
              for (int k_i = 0; k_i < 16; k_i++) {
                __m512i ma = _mm512_set1_epi32(a32[m_i * 16 + k_i]);
                for (int n_i = 0; n_i < 2; n_i++) {
                  c512[m_i * 2 + n_i] = _mm512_dpbssd_epi32(c512[m_i * 2 + n_i], ma, b512[n_i * 16 + k_i]);
                }
              }
            }
          }
        } else {
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

        if (k_block_begin + K::K_BLOCK >= k) {
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
      }
    }
  }
}

} // namespace amx