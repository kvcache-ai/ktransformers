// SPDX-License-Identifier: Apache-2.0
// Transposition changes the MX scaling axis. Split each 32x32 tile into one
// plane per distinct exponent, retaining only the nibbles that use that scale.
#include <emmintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

static int get_exponents(const uint8_t* start, int stride, uint8_t* exponents) {
  uint64_t bits[4]{};
  for (int row = 0; row < 32; ++row) {
    const uint8_t value = start[row * stride];
    bits[value >> 6] |= uint64_t(1) << (value & 63);
  }
  int count = 0;
  for (int group = 0; group < 4; ++group) {
    uint64_t word = bits[group];
    while (word) {
      exponents[count++] = group * 64 + __builtin_ctzll(word);
      word &= word - 1;
    }
  }
  return count;
}

extern "C" int planes(const uint8_t* scales, int n, int k) {
  int maximum = 0;
#pragma omp parallel for num_threads(4) reduction(max : maximum)
  for (int a = 0; a < n / 32; ++a) {
    for (int b = 0; b < k / 32; ++b) {
      uint8_t sorted[32];
      maximum = std::max(maximum, get_exponents(scales + (a * 32) * (k / 32) + b, k / 32, sorted));
    }
  }
  return maximum;
}

static inline void transpose16(__m128i* x) {
  __m128i t[16], u[16], v[16];
  for (int i = 0; i < 16; i += 2) {
    t[i] = _mm_unpacklo_epi8(x[i], x[i + 1]);
    t[i + 1] = _mm_unpackhi_epi8(x[i], x[i + 1]);
  }
  for (int i = 0; i < 16; i += 4)
    for (int j = 0; j < 2; ++j) {
      u[i + 2 * j] = _mm_unpacklo_epi16(t[i + j], t[i + 2 + j]);
      u[i + 2 * j + 1] = _mm_unpackhi_epi16(t[i + j], t[i + 2 + j]);
    }
  for (int i = 0; i < 16; i += 8)
    for (int j = 0; j < 4; ++j) {
      v[i + 2 * j] = _mm_unpacklo_epi32(u[i + j], u[i + 4 + j]);
      v[i + 2 * j + 1] = _mm_unpackhi_epi32(u[i + j], u[i + 4 + j]);
    }
  for (int j = 0; j < 8; ++j) {
    x[2 * j] = _mm_unpacklo_epi64(v[j], v[8 + j]);
    x[2 * j + 1] = _mm_unpackhi_epi64(v[j], v[8 + j]);
  }
}

extern "C" void pack(const uint8_t* w, const uint8_t* scales, int n, int k, int planes, uint8_t* out,
                     uint8_t* result_scales) {
  const __m128i low = _mm_set1_epi8(15);
#pragma omp parallel for num_threads(4)
  for (int b = 0; b < k / 32; ++b)
    for (int a = 0; a < n / 32; ++a) {
      const uint8_t* s = scales + a * 32 * (k / 32) + b;
      uint8_t sorted[32];
      const int used = get_exponents(s, k / 32, sorted);
      __m128i even[16], odd[16];
      alignas(16) uint8_t even_sc[16], odd_sc[16];
      for (int i = 0; i < 16; ++i) {
        const size_t offset = size_t(32 * a + 2 * i) * (k / 2) + 16 * b;
        __m128i p = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + offset));
        __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + offset + k / 2));
        even[i] = _mm_or_si128(_mm_and_si128(p, low), _mm_slli_epi16(_mm_and_si128(q, low), 4));
        odd[i] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(p, 4), low), _mm_andnot_si128(low, q));
        even_sc[i] = s[2 * i * (k / 32)];
        odd_sc[i] = s[(2 * i + 1) * (k / 32)];
      }
      transpose16(even);
      transpose16(odd);
      const auto even_exps = _mm_load_si128(reinterpret_cast<const __m128i*>(even_sc));
      const auto odd_exps = _mm_load_si128(reinterpret_cast<const __m128i*>(odd_sc));
      for (int p = 0; p < planes; ++p) {
        const uint8_t exp = p < used ? sorted[p] : 127;
        auto code = _mm_set1_epi8(exp);
        auto mask = p >= used ? _mm_setzero_si128()
                              : _mm_or_si128(_mm_and_si128(low, _mm_cmpeq_epi8(even_exps, code)),
                                             _mm_andnot_si128(low, _mm_cmpeq_epi8(odd_exps, code)));
        for (int j = 0; j < 16; ++j) {
          const size_t col = size_t(p) * k + b * 32 + 2 * j;
          _mm_storeu_si128(reinterpret_cast<__m128i*>(out + col * (n / 2) + a * 16), _mm_and_si128(even[j], mask));
          _mm_storeu_si128(reinterpret_cast<__m128i*>(out + (col + 1) * (n / 2) + a * 16), _mm_and_si128(odd[j], mask));
          result_scales[col * (n / 32) + a] = exp;
          result_scales[(col + 1) * (n / 32) + a] = exp;
        }
      }
    }
}
