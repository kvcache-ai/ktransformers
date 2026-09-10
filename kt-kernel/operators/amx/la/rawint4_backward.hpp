#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "amx_config.hpp"

#if defined(__AVX512BF16__)
namespace rawint4 {

// Pair order matches gate/up AVX dot accumulation: g0,g1,u0,u1,...
inline void pack_gate_up_coeff(const float* gate, const float* up, uint16_t* out, size_t size) {
  const __m512i order = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
  size_t i = 0;
  for (; i + 16 <= size; i += 16) {
    const __m512bh values = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(up + i), _mm512_loadu_ps(gate + i));
    _mm512_storeu_si512(out + 2 * i, _mm512_permutexvar_epi32(order, (__m512i)values));
  }
  if (i < size) {
    const __mmask16 mask = (1u << (size - i)) - 1;
    const __m512bh values =
        _mm512_cvtne2ps_pbh(_mm512_maskz_loadu_ps(mask, up + i), _mm512_maskz_loadu_ps(mask, gate + i));
    _mm512_mask_storeu_epi32(out + 2 * i, mask, _mm512_permutexvar_epi32(order, (__m512i)values));
  }
}

inline bool use_amx() {
#if defined(__AMX_BF16__)
  static const bool requested = [] {
    const char* value = std::getenv("KT_K2_SFT_BWD_AMX");
    return value == nullptr || value[0] != '0';
  }();
  static thread_local const bool available = requested && __builtin_cpu_supports("amx-bf16") && amx::enable_amx();
  return available;
#else
  return false;
#endif
}

inline void store_result(float* output, const float* value, int width, bool accumulate) {
  for (int col = 0; col < width; col += 16) {
    __m512 result = _mm512_loadu_ps(value + col);
    if (accumulate) result = _mm512_add_ps(_mm512_loadu_ps(output + col), result);
    _mm512_storeu_ps(output + col, result);
  }
}

template <class Decode>
inline void vector_matmul(const uint16_t* input, const uint16_t* panel, int m, int k, int n, int begin, int width,
                          const int* row_to_token, float* output, Decode&& decode) {
  for (int mb = 0; mb < m; mb += 8) {
    const int count = std::min(8, m - mb);
    for (int col = 0; col < width; col += 32) {
      __m512 lo[8], hi[8];
      for (int i = 0; i < 8; ++i) lo[i] = hi[i] = _mm512_setzero_ps();
      for (int kk = 0; kk < k; kk += 2) {
        __m512bh wl, wh;
        if (panel != nullptr) {
          const uint16_t* src = panel + size_t(kk) * width + col * 2;
          wl = (__m512bh)_mm512_loadu_si512(src);
          wh = (__m512bh)_mm512_loadu_si512(src + 32);
        } else {
          decode(kk / 2, begin + col, wl, wh);
        }
#pragma GCC unroll 8
        for (int row = 0; row < count; ++row) {
          uint32_t pair;
          std::memcpy(&pair, input + size_t(mb + row) * k + kk, sizeof(pair));
          const __m512bh coeff = (__m512bh)_mm512_set1_epi32(static_cast<int>(pair));
          lo[row] = _mm512_dpbf16_ps(lo[row], coeff, wl);
          hi[row] = _mm512_dpbf16_ps(hi[row], coeff, wh);
        }
      }
      for (int row = 0; row < count; ++row) {
        const int dst_row = row_to_token == nullptr ? mb + row : row_to_token[mb + row];
        float* dst = output + size_t(dst_row) * n + begin + col;
        if (row_to_token != nullptr) {
          lo[row] = _mm512_add_ps(_mm512_loadu_ps(dst), lo[row]);
          hi[row] = _mm512_add_ps(_mm512_loadu_ps(dst + 16), hi[row]);
        }
        _mm512_storeu_ps(dst, lo[row]);
        _mm512_storeu_ps(dst + 16, hi[row]);
      }
    }
  }
}

#if defined(__AMX_BF16__)
inline void tile_matmul(const uint16_t* input, const uint16_t* panel, int m, int k, int n, int begin, int width,
                        const int* row_to_token, float* output) {
  amx::TileConfig config;
  for (int tile = 0; tile < 8; ++tile) config.set_row_col(tile, 16, 64);
  int configured_rows = 0;
  alignas(64) float result[32 * 32];
  for (int mb = 0; mb < m; mb += 32) {
    const int count = std::min(32, m - mb);
    const int first = std::min(16, count);
    const int second = std::max(0, count - 16);
    if (configured_rows != count) {
      config.set_row_col(0, first, 64);
      config.set_row_col(1, first, 64);
      config.set_row_col(4, first, 64);
      config.set_row_col(2, second, second ? 64 : 0);
      config.set_row_col(3, second, second ? 64 : 0);
      config.set_row_col(5, second, second ? 64 : 0);
      config.set_config();
      configured_rows = count;
    }
    for (int col = 0; col < width; col += 32) {
      _tile_zero(0);
      _tile_zero(1);
      if (second) {
        _tile_zero(2);
        _tile_zero(3);
      }
      for (int kk = 0; kk < k; kk += 32) {
        _tile_loadd(4, input + size_t(mb) * k + kk, k * sizeof(uint16_t));
        if (second) _tile_loadd(5, input + size_t(mb + 16) * k + kk, k * sizeof(uint16_t));
        _tile_loadd(6, panel + size_t(kk) * width + col * 2, width * 2 * sizeof(uint16_t));
        _tile_loadd(7, panel + size_t(kk) * width + col * 2 + 32, width * 2 * sizeof(uint16_t));
        _tile_dpbf16ps(0, 4, 6);
        _tile_dpbf16ps(1, 4, 7);
        if (second) {
          _tile_dpbf16ps(2, 5, 6);
          _tile_dpbf16ps(3, 5, 7);
        }
      }
      _tile_stored(0, result, 32 * sizeof(float));
      _tile_stored(1, result + 16, 32 * sizeof(float));
      if (second) {
        _tile_stored(2, result + 16 * 32, 32 * sizeof(float));
        _tile_stored(3, result + 16 * 32 + 16, 32 * sizeof(float));
      }
      for (int row = 0; row < count; ++row) {
        const int dst_row = row_to_token == nullptr ? mb + row : row_to_token[mb + row];
        store_result(output + size_t(dst_row) * n + begin + col, result + row * 32, 32, row_to_token != nullptr);
      }
    }
  }
  _tile_release();
}
#endif

// One bounded BF16 panel per worker, never a persistent dequantized expert copy.
template <class Decode>
inline void backward(const uint16_t* input, int m, int k, int n, int begin, int end, const int* row_to_token,
                     float* output, Decode&& decode) {
  const bool tile = m >= 4 && k % 32 == 0 && use_amx();
  if (!tile && m < 8) {
    vector_matmul(input, nullptr, m, k, n, begin, end - begin, row_to_token, output, decode);
    return;
  }
  constexpr int kPanelColumns = 64;
  thread_local std::vector<uint16_t> panel;
  panel.resize(size_t(k) * kPanelColumns);
  for (int col = begin; col < end; col += kPanelColumns) {
    const int width = std::min(kPanelColumns, end - col);
    for (int kk = 0; kk < k; kk += 2) {
      for (int offset = 0; offset < width; offset += 32) {
        __m512bh lo, hi;
        decode(kk / 2, col + offset, lo, hi);
        uint16_t* dst = panel.data() + size_t(kk) * width + offset * 2;
        _mm512_storeu_si512(dst, (__m512i)lo);
        _mm512_storeu_si512(dst + 32, (__m512i)hi);
      }
    }
#if defined(__AMX_BF16__)
    if (tile) {
      tile_matmul(input, panel.data(), m, k, n, col, width, row_to_token, output);
      continue;
    }
#endif
    vector_matmul(input, panel.data(), m, k, n, col, width, row_to_token, output, decode);
  }
}

}  // namespace rawint4
#endif
