// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -O2 test_xmx_dpas.cpp -o test_xmx_dpas
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "../../operators/sycl/gptq_int4_xmx-kernels.hpp"

namespace es = sycl::ext::intel::esimd;
namespace xm = es::xmx;

template <int M>
void check_signed_dpas() {
  auto& q = sycl_int4::queue();
  sycl_xmx::USM<int8_t> a;
  sycl_xmx::USM<uint32_t> b;
  sycl_xmx::USM<int> out;
  a.reserve(M * 32);
  b.reserve(64);
  out.reserve(M * 16);
  int8_t weights[32 * 16];
  std::mt19937 gen(20261014 + M);
  for (int i = 0; i < M * 32; ++i) a.data[i] = static_cast<int>(gen() % 255) - 127;
  std::fill_n(b.data, 64, 0u);
  for (int k = 0; k < 32; ++k)
    for (int n = 0; n < 16; ++n) {
      int v = static_cast<int>(gen() % 16) - 8;
      weights[k * 16 + n] = v;
      // Convert the GPTQ offset-binary nibble into a two's-complement S4 nibble.
      b.data[(k / 8) * 16 + n] |= static_cast<uint32_t>((v + 8) ^ 8) << ((k % 8) * 4);
    }
  auto* ap = a.data;
  auto* bp = b.data;
  auto* op = out.data;
  q.single_task([=]() SYCL_ESIMD_KERNEL {
     es::simd<int8_t, M * 32> av;
     av.copy_from(ap);
     es::simd<uint32_t, 64> bv;
     bv.copy_from(bp);
     es::simd<int, M * 16> c = 0;
     auto value =
         xm::dpas<8, M, int, int, uint32_t, int8_t, xm::dpas_argument_type::s4, xm::dpas_argument_type::s8>(c, bv, av);
     value.copy_to(op);
   }).wait_and_throw();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < 16; ++n) {
      int expected = 0;
      for (int k = 0; k < 32; ++k) expected += a.data[m * 32 + k] * weights[k * 16 + n];
      if (out.data[m * 16 + n] != expected) throw std::runtime_error("Signed S8xS4 DPAS mismatch");
    }
  std::cout << "signed DPAS M=" << M << " passed\n";
}

template <bool Q8, int Splits = 1>
void check_projection() {
  constexpr int K = 128, N = 32, GS = 32, R = 3;
  std::mt19937 gen(123);
  std::vector<uint32_t> qw(K / 8 * N, 0);
  std::vector<float> scales(K / GS * N);
  for (auto& word : qw) word = gen();
  for (int g = 0; g < K / GS; ++g)
    for (int n = 0; n < N; ++n) scales[g * N + n] = (g + 1) * 0.007f;
  sycl_xmx::Weight weight;
  weight.init(1, N, K, GS);
  weight.from_mat(0, qw.data(), scales.data());
  sycl_xmx::USM<uint16_t> input;
  sycl_xmx::USM<int8_t> qi;
  sycl_xmx::USM<float> sa, output;
  sycl_xmx::USM<sycl_xmx::Tile> tile;
  input.reserve(R * K);
  qi.reserve(R * K);
  sa.reserve(R * K / GS);
  output.reserve(R * N);
  tile.reserve(1);
  tile.data[0] = {0, 0, R};
  for (int r = 0; r < R; ++r)
    for (int k = 0; k < K; ++k) {
      float v = static_cast<int>(gen() % 15) - 7;
      input.data[r * K + k] = sycl_int4::fp32_to_bf16(v);
      qi.data[r * K + k] = static_cast<int8_t>(v);
    }
  for (int r = 0; r < R; ++r)
    for (int g = 0; g < K / GS; ++g) sa.data[r * K / GS + g] = (g + 1) * 0.2f;
  auto event = sycl_xmx::projection<8, Q8, false, false, Splits>(tile.data, 1, nullptr, input.data, qi.data, sa.data,
                                                                 weight, nullptr, nullptr, output.data, 0, 0);
  sycl_int4::wait_and_throw(event);
  for (int r = 0; r < R; ++r)
    for (int n = 0; n < N; ++n) {
      float expected = 0;
      for (int g = 0; g < K / GS; ++g) {
        int partial = 0;
        for (int j = 0; j < GS; ++j) {
          int k = g * GS + j;
          int w = static_cast<int>((qw[(k / 8) * N + n] >> (4 * (k % 8))) & 15) - 8;
          if constexpr (Q8)
            partial += qi.data[r * K + k] * w;
          else
            expected +=
                sycl_int4::bf16_to_fp32(input.data[r * K + k]) * static_cast<float>(sycl::half(w * scales[g * N + n]));
        }
        if constexpr (Q8) expected += partial * scales[g * N + n] * sa.data[r * K / GS + g];
      }
      expected = sycl_int4::bf16_to_fp32(sycl_int4::fp32_to_bf16(expected));
      if (std::fabs(output.data[r * N + n] - expected) > 0.02f)
        throw std::runtime_error("Projection/tail/group-scale mismatch");
    }
  std::cout << (Q8 ? "W4A8" : "W4A16") << " projection, partial tile and distinct group scales passed\n";
}

int main() {
  try {
    check_signed_dpas<1>();
    check_signed_dpas<2>();
    check_signed_dpas<4>();
    check_signed_dpas<8>();
    check_projection<false>();
    check_projection<true>();
    check_projection<false, 2>();
    check_projection<true, 2>();
    check_projection<false, 4>();
    check_projection<true, 4>();
    check_projection<false, 8>();
    check_projection<true, 8>();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
