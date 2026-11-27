// SPDX-License-Identifier: Apache-2.0
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "../../operators/arm/lora_rank8.hpp"

static uint16_t narrow(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0) return (bits >> 16) & 0x8000u;
  if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x0040u;
  return (bits + 0x7fffu + ((bits >> 16) & 1)) >> 16;
}

static float widen(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

int main() {
  const uint32_t special[][4] = {
      {0x00000001, 0x80000001, 0x7f800000, 0xff800000},
      {0x3f808000, 0x3f818000, 0x7f800001, 0xff808000},
  };
  for (const auto& bits : special) {
    uint16_t result[4];
    vst1_u16(result, kt::arm::lora_bf16_rne(vreinterpretq_f32_u32(vld1q_u32(bits))));
    for (int i = 0; i < 4; ++i) {
      float value;
      std::memcpy(&value, &bits[i], sizeof(value));
      assert(result[i] == narrow(value));
    }
  }
  std::mt19937 rng(42);
  std::normal_distribution<float> sample;
  for (int rows : {1, 2, 3, 17}) {
    for (int cols : {16, 80, 512, 2048, 7168}) {
      std::vector<float> x(rows * 8);
      std::vector<uint16_t> w(8 * cols), y(rows * cols);
      for (auto& v : x) v = sample(rng);
      for (auto& v : w) v = narrow(sample(rng));
      for (auto& v : y) v = narrow(sample(rng));
      auto expected = y;
      assert(!kt::arm::try_lora_rank8(x.data(), w.data(), y.data(), rows, 4, cols, 0.75f));
      assert(!kt::arm::try_lora_rank8(x.data(), w.data(), y.data(), rows, 8, cols - 1, 0.75f));
      assert(y == expected);
      for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          float sum = 0;
          for (int k = 0; k < 8; ++k) sum = std::fma(x[row * 8 + k], widen(w[k * cols + col]), sum);
          expected[row * cols + col] = narrow(widen(y[row * cols + col]) + 0.75f * sum);
        }
      }
      assert(kt::arm::try_lora_rank8(x.data(), w.data(), y.data(), rows, 8, cols, 0.75f));
      assert(y == expected);
    }
  }
}
