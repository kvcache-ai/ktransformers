// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arm_neon.h>

#include <cstring>

namespace kt::arm {

template <class BF16>
inline float32x4_t load_lora_bf16(const BF16* input) {
  static_assert(sizeof(BF16) == 2);
  uint16x4_t bits;
  std::memcpy(&bits, input, sizeof(bits));
  return vreinterpretq_f32_u32(vshll_n_u16(bits, 16));
}

inline uint16x4_t lora_bf16_rne(float32x4_t value) {
  const auto bits = vreinterpretq_u32_f32(value);
  const auto magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
  const auto odd = vandq_u32(vshrq_n_u32(bits, 16), vdupq_n_u32(1));
  auto rounded = vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7fff), odd));
  // Match the BF16 conversion in the existing epilogue, including signed zero and quiet NaN.
  rounded = vbslq_u32(vcltq_u32(magnitude, vdupq_n_u32(0x00800000)), vandq_u32(bits, vdupq_n_u32(0x80000000)), rounded);
  rounded = vbslq_u32(vcgtq_u32(magnitude, vdupq_n_u32(0x7f800000)), vorrq_u32(bits, vdupq_n_u32(0x00400000)), rounded);
  return vshrn_n_u32(rounded, 16);
}

template <int Rows, class BF16>
inline void lora_rank8_rows(const float* input, const BF16* weight, BF16* output, int width, float scale) {
  for (int col = 0; col < width; col += 16) {
    float32x4_t accum[Rows][4] = {};
    for (int k = 0; k < 8; ++k) {
      for (int v = 0; v < 4; ++v) {
        auto w = load_lora_bf16(weight + k * width + col + 4 * v);
        for (int row = 0; row < Rows; ++row) accum[row][v] = vfmaq_n_f32(accum[row][v], w, input[row * 8 + k]);
      }
    }
    for (int row = 0; row < Rows; ++row) {
      for (int v = 0; v < 4; ++v) {
        BF16* dst = output + row * width + col + 4 * v;
        auto result = vaddq_f32(load_lora_bf16(dst), vmulq_n_f32(accum[row][v], scale));
        auto bits = lora_bf16_rne(result);
        std::memcpy(dst, &bits, sizeof(bits));
      }
    }
  }
}

template <class BF16>
inline bool try_lora_rank8(const float* input, const BF16* weight, BF16* output, int tokens, int rank, int width,
                           float scale) {
  if (rank != 8 || width % 16 != 0) return false;
  int row = 0;
  for (; row + 1 < tokens; row += 2) lora_rank8_rows<2>(input + row * 8, weight, output + row * width, width, scale);
  if (row < tokens) lora_rank8_rows<1>(input + row * 8, weight, output + row * width, width, scale);
  return true;
}

}  // namespace kt::arm
