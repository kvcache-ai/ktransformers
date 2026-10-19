// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -O2 test_vector_projection.cpp -o test_vector_projection
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "../../operators/sycl/gptq_int4_vector-kernels.hpp"

using namespace sycl_xmx;

void check_dp4a() {
  USM<uint32_t> w;
  USM<int32_t> a, output;
  w.reserve(16);
  a.reserve(16);
  output.reserve(16);
  std::mt19937 random(20261014);
  for (int j = 0; j < 16; ++j) {
    w.data[j] = random();
    a.data[j] = random();
  }
  auto* wp = w.data;
  auto* ap = a.data;
  auto* op = output.data;
  auto event = sycl_int4::queue().single_task([=]() SYCL_ESIMD_KERNEL {
    es::simd<uint32_t, 16> packed;
    packed.copy_from(wp);
    es::simd<int32_t, 16> activations;
    activations.copy_from(ap);
    es::simd<int32_t, 16> zero = 0;
    es::dp4a<int32_t>(zero, unpack_s4_bytes(packed), activations).copy_to(op);
  });
  sycl_int4::wait_and_throw(event);
  for (int j = 0; j < 16; ++j) {
    int expected = 0;
    for (int r = 0; r < 4; ++r) {
      int weight = static_cast<int>((w.data[j] >> (4 * r)) & 15) - 8;
      int activation = static_cast<int8_t>((static_cast<uint32_t>(a.data[j]) >> (8 * r)) & 255);
      expected += weight * activation;
    }
    if (output.data[j] != expected) {
      std::cerr << "lane=" << j << " w=" << w.data[j] << " a=" << a.data[j] << " expected=" << expected
                << " actual=" << output.data[j] << '\n';
      throw std::runtime_error("Signed nibble/DP4A mismatch");
    }
  }
}

template <bool Q8, bool GateUp>
void check_projection(int group_size, int splits) {
  constexpr int K = 256, N = 32, R = 3;
  std::mt19937 random(123);
  const int groups = K / group_size;
  std::vector<uint32_t> qw(K / 8 * N), up(K / 8 * N);
  std::vector<float> scales(groups * N), up_scales(groups * N);
  for (auto& x : qw) x = random();
  for (auto& x : up) x = random();
  for (int g = 0; g < groups; ++g)
    for (int n = 0; n < N; ++n) {
      scales[g * N + n] = (g + 1) * 0.003f;
      up_scales[g * N + n] = (g + 2) * 0.001f;
    }
  Weight w, uw;
  w.init(1, N, K, group_size);
  uw.init(1, N, K, group_size);
  w.from_mat(0, qw.data(), scales.data());
  uw.from_mat(0, up.data(), up_scales.data());
  USM<uint16_t> input, activated;
  USM<int8_t> qi;
  USM<float> sa, output;
  USM<int> tokens;
  USM<Tile> tiles;
  input.reserve(R * K);
  activated.reserve(R * N);
  qi.reserve(R * K);
  sa.reserve(R * groups);
  output.reserve(R * N);
  tiles.reserve(1);
  tokens.reserve(R);
  tiles.data[0] = {0, 0, R};
  for (int r = 0; r < R; ++r) {
    tokens.data[r] = R - r - 1;
    for (int k = 0; k < K; ++k) {
      float value = (static_cast<int>(random() % 255) - 127) * 0.125f;
      input.data[r * K + k] = sycl_int4::fp32_to_bf16(value);
      qi.data[r * K + k] = static_cast<int8_t>(random() % 255 - 127);
    }
    for (int g = 0; g < groups; ++g) sa.data[r * groups + g] = (g + 1) * 0.03f;
  }
  auto event = vector_dispatch<Q8, GateUp>(splits, tiles.data, 1, 8, tokens.data, input.data, qi.data, sa.data, w, &uw,
                                           activated.data, output.data, 0.5f, 1.7f);
  sycl_int4::wait_and_throw(event);
  for (int r = 0; r < R; ++r)
    for (int n = 0; n < N; ++n) {
      const int row = GateUp ? tokens.data[r] : r;
      double gate = 0, upper = 0;
      for (int g = 0; g < groups; ++g) {
        double pg = 0, pu = 0;
        for (int j = 0; j < group_size; ++j) {
          int k = g * group_size + j;
          double av = Q8 ? qi.data[row * K + k] : sycl_int4::bf16_to_fp32(input.data[row * K + k]);
          pg += av * (static_cast<int>((qw[(k / 8) * N + n] >> (4 * (k % 8))) & 15) - 8);
          pu += av * (static_cast<int>((up[(k / 8) * N + n] >> (4 * (k % 8))) & 15) - 8);
        }
        double activation_scale = Q8 ? sa.data[row * groups + g] : 1.0;
        gate += pg * scales[g * N + n] * activation_scale;
        upper += pu * up_scales[g * N + n] * activation_scale;
      }
      if constexpr (GateUp) {
        gate = std::clamp(gate, -0.5, 0.5);
        upper = std::clamp(upper, -0.5, 0.5);
        gate = gate / (1.0 + std::exp(-gate * 1.7)) * (upper + 1.0);
      }
      float expected = sycl_int4::bf16_to_fp32(sycl_int4::fp32_to_bf16(static_cast<float>(gate)));
      float actual = GateUp ? sycl_int4::bf16_to_fp32(activated.data[r * N + n]) : output.data[r * N + n];
      if (std::fabs(expected - actual) > std::max(0.004f * std::fabs(expected), 1e-5f))
        throw std::runtime_error("Vector projection/reference mismatch");
    }
}

int main() {
  try {
    check_dp4a();
    for (int group : {32, 128})
      for (int splits : {1, 2, 4, 8}) {
        check_projection<false, false>(group, splits);
        check_projection<false, true>(group, splits);
        check_projection<true, false>(group, splits);
        check_projection<true, true>(group, splits);
        std::cout << "vector/DP4A gate/down group=" << group << " splits=" << splits << " passed\n";
      }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
