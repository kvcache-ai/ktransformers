#ifndef AMX_UTILS_HPP
#define AMX_UTILS_HPP

#include <cstdio>
#include <iostream>

#include "../../common.hpp"
#include "amx_config.hpp"

namespace amx {
#if defined(HAVE_AMX)
// Debug functions
inline void debug_tile(int t) {
  printf("Tile %d\n", t);
  int8_t data[16][64] = {};
  TileConfig::store_data(t, data, 64);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 64; j++) {
      printf("%4d ", data[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

inline void debug_tile_int32(int t) {
  printf("Tile %d\n", t);
  int32_t data[16][16] = {};
  TileConfig::store_data(t, data, 64);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      printf("%10d ", data[i][j]);
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

inline void debug_tiles_int32(int to = 8) {
  for (int i = 0; i < to; i++) {
    debug_tile_int32(i);
  }
}

inline void debug_tiles_224() {
  for (int i = 0; i < 4; i++) {
    debug_tile(i);
  }
  for (int i = 4; i < 8; i++) {
    debug_tile_int32(i);
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

inline void debug_m512i(__m512i x) {
  int32_t data[16];
  _mm512_storeu_epi32(data, x);
  for (int i = 0; i < 16; i++) {
    printf("0x%08x ", data[i]);
  }
  printf("\n");
}

inline void debug_m128i(__m128i x) {
  int32_t data[16];
  _mm_storeu_epi32(data, x);
  for (int i = 0; i < 4; i++) {
    printf("0x%08x ", data[i]);
  }
  printf("\n");
}
#endif
// transpose utils
#define SHUFFLE_EPI32(a, b, mask) \
  _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))

inline void transpose_8x8_32bit(__m256i* v, __m256i* v1) {
  // unpacking and 32-bit elements
  v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
  v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
  v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
  v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
  v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
  v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
  v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
  v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

  // shuffling the 32-bit elements
  v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
  v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
  v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
  v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
  v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
  v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
  v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
  v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

  // shuffling 128-bit elements
  v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
  v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
  v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
  v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
  v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
  v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
  v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
  v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

inline void transpose_8x8_32bit(__m256i* v) {
  __m256i v1[8];
  transpose_8x8_32bit(v, v1);

  v[0] = v1[0];
  v[1] = v1[1];
  v[2] = v1[2];
  v[3] = v1[3];
  v[4] = v1[4];
  v[5] = v1[5];
  v[6] = v1[6];
  v[7] = v1[7];
}

inline void transpose_16x4_32bit(__m512i* r, __m512i* d) {
  static const __m512i index1 =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02, 0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);

  d[0] = _mm512_permutexvar_epi32(index1, r[0]);
  d[1] = _mm512_permutexvar_epi32(index1, r[1]);
  d[2] = _mm512_permutexvar_epi32(index1, r[2]);
  d[3] = _mm512_permutexvar_epi32(index1, r[3]);

  r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
  r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
  r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
  r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

  d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
  d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
  d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
  d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);
}

inline void transpose_16x16_32bit(__m512i* v) {
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

inline void transpose_16x8_32bit(__m256i* v) {
  transpose_8x8_32bit(v);
  transpose_8x8_32bit(v + 8);
  __m256i v1[16];
  for (int i = 0; i < 16; i++) v1[i] = v[i];

  for (int i = 0; i < 8; i++) {
    v[i * 2] = v1[i];
    v[i * 2 + 1] = v1[8 + i];
  }
}

/*
  Transpose 16x16 32-bit elements
  Note that v must be 64 byte aligned
*/
inline void transpose_16x16_32bit(__m512i* v, size_t stride) {
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

}  // namespace amx

#endif  // AMX_UTILS_HPP