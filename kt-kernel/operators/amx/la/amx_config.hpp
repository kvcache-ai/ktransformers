#ifndef AMX_CONFIG_HPP
#define AMX_CONFIG_HPP
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
#include <immintrin.h>

#if !defined(__AVX512BF16__)
#ifndef __m512bh
#define __m512bh __m512i
#endif

// BF16 emulation via AVX512F
static ALWAYS_INLINE __m512 _mm512_dpbf16_ps_emulated(__m512 src, __m512i a, __m512i b) {
  __m512 a_low = _mm512_castsi512_ps(_mm512_slli_epi32(a, 16));
  __m512 b_low = _mm512_castsi512_ps(_mm512_slli_epi32(b, 16));
  __m512i mask = _mm512_set1_epi32(0xFFFF0000);
  __m512 a_high = _mm512_castsi512_ps(_mm512_and_si512(a, mask));
  __m512 b_high = _mm512_castsi512_ps(_mm512_and_si512(b, mask));
  __m512 res = _mm512_fmadd_ps(a_low, b_low, src);
  res = _mm512_fmadd_ps(a_high, b_high, res);
  return res;
}

#ifndef _mm512_dpbf16_ps
#define _mm512_dpbf16_ps _mm512_dpbf16_ps_emulated
#endif
#endif

#if !defined(__AVX512VBMI__)
// VBMI emulation via AVX512F+BW
static ALWAYS_INLINE __m512i _mm512_permutex2var_epi8_emulated(__m512i a, __m512i idx, __m512i b) {
  __m256i idx_lo_256 = _mm512_castsi512_si256(idx);
  __m256i idx_hi_256 = _mm512_extracti64x4_epi64(idx, 1);
  __m512i idx_lo_w = _mm512_cvtepu8_epi16(idx_lo_256);
  __m512i idx_hi_w = _mm512_cvtepu8_epi16(idx_hi_256);
  __m512i p_lo = _mm512_srli_epi16(idx_lo_w, 1);
  __m512i p_hi = _mm512_srli_epi16(idx_hi_w, 1);
  __m512i w_lo = _mm512_permutex2var_epi16(a, p_lo, b);
  __m512i w_hi = _mm512_permutex2var_epi16(a, p_hi, b);
  __m512i shift_lo = _mm512_slli_epi16(_mm512_and_si512(idx_lo_w, _mm512_set1_epi16(1)), 3);
  __m512i shift_hi = _mm512_slli_epi16(_mm512_and_si512(idx_hi_w, _mm512_set1_epi16(1)), 3);
  __m512i res_lo_w = _mm512_srlv_epi16(w_lo, shift_lo);
  __m512i res_hi_w = _mm512_srlv_epi16(w_hi, shift_hi);
  __m256i res_lo = _mm512_cvtepi16_epi8(res_lo_w);
  __m256i res_hi = _mm512_cvtepi16_epi8(res_hi_w);
  return _mm512_inserti64x4(_mm512_castsi256_si512(res_lo), res_hi, 1);
}

#ifndef _mm512_permutex2var_epi8
#define _mm512_permutex2var_epi8 _mm512_permutex2var_epi8_emulated
#endif
#endif

#if defined(__AMX__) || defined(__AMXINT8__) || defined(__AMXBF16__) || defined(__AMX_TILE__) || defined(HAVE_AMX)
#ifndef HAVE_AMX
#define HAVE_AMX
#endif
#include <emmintrin.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <tmmintrin.h>
#include <unistd.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace amx {

#define ARCH_GET_XCOMP_SUPP 0x1021
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE ((1 << XFEATURE_XTILECFG) | (1 << XFEATURE_XTILEDATA))

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
  // CHECK: whether this can be removed?
  // static thread_local bool initialized = false;
  // if (initialized) {
  //   return true;
  // }
  // initialized = true;

  // if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
  //   printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
  //   return false;
  // } else {
  //   // printf("\n TILE DATA USE SET - OK \n\n");
  //   return true;
  // }
  // return true;
  unsigned long features;
  long rc;
  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_SUPP, &features);

  if (!rc && (features & XFEATURE_MASK_XTILE) == XFEATURE_MASK_XTILE) {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) return false;  // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    // printf("\n TILE DATA USE SET - OK \n\n");
    return true;
  }
  return false;
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

  static void load_data(int to, void* from, size_t stride) {
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

  static void store_data(int from, void* to, size_t stride) {
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

}  // namespace amx
#endif  // defined(__AMX__)
#endif  // AMX_CONFIG_HPP
