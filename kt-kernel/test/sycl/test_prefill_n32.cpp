// SPDX-License-Identifier: Apache-2.0
// Build: icpx -fsycl -fp-model=precise -O2 test_prefill_n32.cpp -o test_prefill_n32
#include <iostream>
#include <random>
#include <vector>

#include "../../operators/sycl/gptq_int4_xmx-prefill.hpp"
using namespace sycl_xmx;

template <int Rows, bool Gate, bool Fused = false, PrefillLoad Load = PrefillLoad::Block>
void check(int gs) {
  constexpr int K = 256, N = 256, R = 33, E = 2;
  const int groups = K / gs;
  std::mt19937 random(258);
  Weight w, u;
  w.init(E, N, K, gs);
  u.init(E, N, K, gs);
  std::vector<uint32_t> packed(K / 8 * N);
  std::vector<float> scales(groups * N);
  for (int e = 0; e < E; ++e) {
    for (auto* weight : {&w, &u}) {
      for (auto& value : packed) value = random();
      for (auto& value : scales) value = .001f + (random() % 64) * .0001f;
      weight->from_mat(e, packed.data(), scales.data());
    }
  }
  USM<int8_t> input, qref, qtest;
  USM<float> sa, oref, otest, sref, stest;
  USM<uint16_t> aref, atest;
  USM<int> tokens;
  USM<Tile> old_tiles, new_tiles;
  input.reserve(R * E * K);
  sa.reserve(R * E * groups);
  tokens.reserve(R * E);
  oref.reserve(R * E * N);
  otest.reserve(R * E * N);
  aref.reserve(R * E * N);
  atest.reserve(R * E * N);
  qref.reserve(R * E * N);
  qtest.reserve(R * E * N);
  sref.reserve(R * E * (N / 128));
  stest.reserve(R * E * (N / 128));
  for (int r = 0; r < R * E; ++r) {
    tokens.data[r] = R * E - r - 1;
    for (int k = 0; k < K; ++k) input.data[r * K + k] = int(random() % 255) - 127;
    for (int g = 0; g < groups; ++g) sa.data[r * groups + g] = .0001f * (1 + random() % 32);
  }
  old_tiles.reserve(E * ((R + 7) / 8));
  new_tiles.reserve(E * ((R + Rows - 1) / Rows));
  int a = 0, b = 0;
  for (int e = 0; e < E; ++e) {
    for (int r = 0; r < R; r += 8) old_tiles.data[a++] = {e, e * R + r, std::min(8, R - r)};
    for (int r = 0; r < R; r += Rows) new_tiles.data[b++] = {e, e * R + r, std::min(Rows, R - r)};
  }
  auto first = projection<8, true, Gate, Fused>(old_tiles.data, a, tokens.data, nullptr, input.data, sa.data, w, &u,
                                                aref.data, oref.data, .5f, 1.702f, nullptr, qref.data, sref.data);
  sycl_int4::wait_and_throw(first);
  auto second =
      prefill_n32<Rows, Gate, Fused, Load>(new_tiles.data, b, tokens.data, input.data, sa.data, w, &u, atest.data,
                                           otest.data, .5f, 1.702f, nullptr, qtest.data, stest.data);
  sycl_int4::wait_and_throw(second);
  for (int j = 0; j < R * E * N; ++j) {
    bool equal;
    if constexpr (Fused)
      equal = qref.data[j] == qtest.data[j];
    else if constexpr (Gate)
      equal = aref.data[j] == atest.data[j];
    else
      equal = oref.data[j] == otest.data[j];
    if (!equal) {
      std::cerr << "M=" << Rows << " gate=" << Gate << " fused=" << Fused << " group=" << gs << " index=" << j;
      if constexpr (Fused)
        std::cerr << " values=" << int(qref.data[j]) << '/' << int(qtest.data[j]);
      else if constexpr (Gate)
        std::cerr << " values=" << sycl_int4::bf16_to_fp32(aref.data[j]) << '/'
                  << sycl_int4::bf16_to_fp32(atest.data[j]);
      else
        std::cerr << " values=" << oref.data[j] << '/' << otest.data[j];
      std::cerr << '\n';
      throw std::runtime_error("N32 projection/activation/quantization mismatch");
    }
  }
  if constexpr (Fused)
    for (int j = 0; j < R * E * N / 128; ++j)
      if (sref.data[j] != stest.data[j]) throw std::runtime_error("N32 quantization scale mismatch");
}

template <int Rows>
void suite() {
  for (int group : {32, 128}) {
    check<Rows, true>(group);
    check<Rows, false>(group);
  }
  check<Rows, true, true>(128);
  for (int group : {32, 128}) {
    check<Rows, true, false, PrefillLoad::Gather>(group);
    check<Rows, false, false, PrefillLoad::Gather>(group);
  }
  check<Rows, true, true, PrefillLoad::Gather>(128);
  std::cout << "N32 M=" << Rows
            << " gate/down, distinct experts/groups/scales, reverse row mapping, tail33, block/gather, fused quant "
               "exact passed\n";
}
int main() {
  try {
    suite<1>();
    suite<2>();
    suite<4>();
    suite<8>();
    suite<16>();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
