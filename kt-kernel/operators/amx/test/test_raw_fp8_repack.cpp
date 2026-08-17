#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../la/amx_kernels.hpp"
#include "../la/amx_raw_kernels.hpp"

namespace {

using Kernel = amx::GemmKernel224FP8;
using BufferA = Kernel::BufferA;
using BufferB = Kernel::BufferB;
using BufferC = Kernel::BufferC;
constexpr int kGroupSize = 128;

void* alloc_buffer(size_t bytes) {
  const size_t aligned_bytes = (bytes + 63) / 64 * 64;
  void* pointer = std::aligned_alloc(64, aligned_bytes);
  if (pointer == nullptr) std::abort();
  std::memset(pointer, 0, aligned_bytes);
  return pointer;
}

void fill_raw(std::vector<uint8_t>& weights) {
  for (size_t i = 0; i < weights.size(); ++i) weights[i] = static_cast<uint8_t>(i);
}

void fill_scale_bits(std::vector<float>& scales) {
  for (size_t i = 0; i < scales.size(); ++i) {
    const uint32_t bits = (i % 5 == 0) ? (0x7fc00000u + static_cast<uint32_t>(i & 0x003fffffu))
                                        : (0x3f000000u + static_cast<uint32_t>(i * 0x00010101u));
    std::memcpy(scales.data() + i, &bits, sizeof(bits));
  }
}

void from_mat(BufferB& buffer, const uint8_t* weights, const float* scales) {
  const int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ++ith) buffer.from_mat(weights, scales, ith, nth);
}

void to_mat(const BufferB& buffer, uint8_t* weights, float* scales) {
  const int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ++ith) buffer.to_mat(weights, scales, ith, nth);
}

void transpose_raw(const std::vector<uint8_t>& source, std::vector<uint8_t>& destination, int n, int k) {
  for (int row = 0; row < n; ++row) {
    for (int column = 0; column < k; ++column) {
      destination[(size_t)column * n + row] = source[(size_t)row * k + column];
    }
  }
}

void transpose_scales(const std::vector<float>& source, std::vector<float>& destination, int n, int k) {
  const int source_n_blocks = n / kGroupSize;
  const int source_k_blocks = k / kGroupSize;
  for (int bn = 0; bn < source_n_blocks; ++bn) {
    for (int bk = 0; bk < source_k_blocks; ++bk) {
      std::memcpy(destination.data() + (size_t)bk * source_n_blocks + bn,
                  source.data() + (size_t)bn * source_k_blocks + bk, sizeof(float));
    }
  }
}

bool run_repack_case(int n, int k, int transpose_threads) {
  const size_t weight_count = (size_t)n * k;
  const size_t scale_count = (size_t)(n / kGroupSize) * (k / kGroupSize);
  std::vector<uint8_t> source(weight_count);
  std::vector<float> source_scales(scale_count);
  std::vector<uint8_t> roundtrip(weight_count);
  std::vector<float> roundtrip_scales(scale_count);
  std::vector<uint8_t> expected_weights(weight_count);
  std::vector<float> expected_scales(scale_count);
  std::vector<uint8_t> actual_weights(weight_count);
  std::vector<float> actual_scales(scale_count);
  fill_raw(source);
  fill_scale_bits(source_scales);
  transpose_raw(source, expected_weights, n, k);
  transpose_scales(source_scales, expected_scales, n, k);

  void* source_memory = alloc_buffer(BufferB::required_size(n, k, kGroupSize));
  void* expected_memory = alloc_buffer(BufferB::required_size(k, n, kGroupSize));
  void* direct_memory = alloc_buffer(BufferB::required_size(k, n, kGroupSize));
  void* twice_memory = alloc_buffer(BufferB::required_size(n, k, kGroupSize));
  BufferB packed_source(n, k, kGroupSize, source_memory);
  BufferB packed_expected(k, n, kGroupSize, expected_memory);
  BufferB packed_direct(k, n, kGroupSize, direct_memory);
  BufferB packed_twice(n, k, kGroupSize, twice_memory);

  from_mat(packed_source, source.data(), source_scales.data());
  to_mat(packed_source, roundtrip.data(), roundtrip_scales.data());
  from_mat(packed_expected, expected_weights.data(), expected_scales.data());
  for (int ith = 0; ith < transpose_threads; ++ith) {
    packed_direct.from_bb_transposed(packed_source, ith, transpose_threads);
  }
  to_mat(packed_direct, actual_weights.data(), actual_scales.data());
  for (int ith = 0; ith < transpose_threads; ++ith) {
    packed_twice.from_bb_transposed(packed_direct, ith, transpose_threads);
  }

  const bool roundtrip_ok = std::memcmp(source.data(), roundtrip.data(), weight_count) == 0 &&
                            std::memcmp(source_scales.data(), roundtrip_scales.data(), scale_count * sizeof(float)) == 0;
  const bool logical_ok = std::memcmp(expected_weights.data(), actual_weights.data(), weight_count) == 0 &&
                          std::memcmp(expected_scales.data(), actual_scales.data(), scale_count * sizeof(float)) == 0;
  const bool packed_ok =
      std::memcmp(expected_memory, direct_memory, BufferB::required_size(k, n, kGroupSize)) == 0;
  const bool twice_ok =
      std::memcmp(source_memory, twice_memory, BufferB::required_size(n, k, kGroupSize)) == 0;

  std::free(source_memory);
  std::free(expected_memory);
  std::free(direct_memory);
  std::free(twice_memory);
  std::printf("raw FP8 repack %dx%d threads=%d: roundtrip=%s logical=%s packed=%s twice=%s\n", n, k,
              transpose_threads, roundtrip_ok ? "PASS" : "FAIL", logical_ok ? "PASS" : "FAIL",
              packed_ok ? "PASS" : "FAIL", twice_ok ? "PASS" : "FAIL");
  return roundtrip_ok && logical_ok && packed_ok && twice_ok;
}

bool run_alignment_rejection() {
  void* source_memory = alloc_buffer(BufferB::required_size(96, 128, kGroupSize));
  void* destination_memory = alloc_buffer(BufferB::required_size(128, 96, kGroupSize));
  BufferB source(96, 128, kGroupSize, source_memory);
  BufferB destination(128, 96, kGroupSize, destination_memory);
  bool rejected = false;
  try {
    destination.from_bb_transposed(source, 0, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  std::free(source_memory);
  std::free(destination_memory);
  std::printf("raw FP8 repack unaligned rejection: %s\n", rejected ? "PASS" : "FAIL");
  return rejected;
}

bool run_group_rejection() {
  constexpr int invalid_group = 64;
  void* source_memory = alloc_buffer(BufferB::required_size(128, 128, invalid_group));
  void* destination_memory = alloc_buffer(BufferB::required_size(128, 128, invalid_group));
  BufferB source(128, 128, invalid_group, source_memory);
  BufferB destination(128, 128, invalid_group, destination_memory);
  bool rejected = false;
  try {
    destination.from_bb_transposed(source, 0, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  std::free(source_memory);
  std::free(destination_memory);
  std::printf("raw FP8 repack group rejection: %s\n", rejected ? "PASS" : "FAIL");
  return rejected;
}

bool run_pool_alignment_case() {
  constexpr int n = 128;
  constexpr int k = 128;
  const size_t one_buffer = BufferB::required_size(n, k, kGroupSize);
  void* pool = alloc_buffer(one_buffer * 2);
  auto* first = static_cast<uint8_t*>(pool);
  auto* second = first + one_buffer;
  const bool aligned = one_buffer % 64 == 0 && reinterpret_cast<uintptr_t>(first) % 64 == 0 &&
                       reinterpret_cast<uintptr_t>(second) % 64 == 0;
  BufferB first_buffer(n, k, kGroupSize, first);
  BufferB second_buffer(n, k, kGroupSize, second);
  const bool scale_offsets = first_buffer.d == reinterpret_cast<float*>(first + (size_t)n * k) &&
                             second_buffer.d == reinterpret_cast<float*>(second + (size_t)n * k);
  std::free(pool);
  std::printf("raw FP8 contiguous pool alignment: aligned=%s scale_offsets=%s\n", aligned ? "PASS" : "FAIL",
              scale_offsets ? "PASS" : "FAIL");
  return aligned && scale_offsets;
}

bool run_backward_gemm_case() {
  constexpr int source_n = 128;
  constexpr int source_k = 256;
  constexpr int m = 32;
  constexpr int n = source_k;
  constexpr int k = source_n;
  std::vector<uint8_t> source((size_t)source_n * source_k, 0x38);  // E4M3 1.0
  std::vector<float> source_scales((size_t)(source_n / kGroupSize) * (source_k / kGroupSize), 1.0f);
  std::vector<uint8_t> transposed(source.size());
  std::vector<float> transposed_scales(source_scales.size());
  transpose_raw(source, transposed, source_n, source_k);
  transpose_scales(source_scales, transposed_scales, source_n, source_k);

  void* source_memory = alloc_buffer(BufferB::required_size(source_n, source_k, kGroupSize));
  void* expected_memory = alloc_buffer(BufferB::required_size(n, k, kGroupSize));
  void* direct_memory = alloc_buffer(BufferB::required_size(n, k, kGroupSize));
  BufferB packed_source(source_n, source_k, kGroupSize, source_memory);
  auto expected = std::make_shared<BufferB>(n, k, kGroupSize, expected_memory);
  auto direct = std::make_shared<BufferB>(n, k, kGroupSize, direct_memory);
  from_mat(packed_source, source.data(), source_scales.data());
  from_mat(*expected, transposed.data(), transposed_scales.data());
  const int transpose_nth = Kernel::recommended_nth(n);
  for (int ith = 0; ith < transpose_nth; ++ith) direct->from_bb_transposed(packed_source, ith, transpose_nth);

  std::vector<ggml_bf16_t> input((size_t)m * k, GGML_FP32_TO_BF16(1.0f));
  std::vector<ggml_bf16_t> expected_output((size_t)m * n);
  std::vector<ggml_bf16_t> direct_output((size_t)m * n);
  void* a_memory = alloc_buffer(BufferA::required_size(m, k));
  void* expected_c_memory = alloc_buffer(BufferC::required_size(m, n));
  void* direct_c_memory = alloc_buffer(BufferC::required_size(m, n));
  auto a = std::make_shared<BufferA>(m, k, a_memory);
  auto expected_c = std::make_shared<BufferC>(m, n, expected_c_memory);
  auto direct_c = std::make_shared<BufferC>(m, n, direct_c_memory);
  a->from_mat(m, input.data(), 0, 1);
  const int gemm_nth = Kernel::recommended_nth(n);
  for (int ith = 0; ith < gemm_nth; ++ith) {
    amx::mat_mul_kgroup(m, n, k, kGroupSize, a, expected, expected_c, ith, gemm_nth);
    amx::mat_mul_kgroup(m, n, k, kGroupSize, a, direct, direct_c, ith, gemm_nth);
    expected_c->to_mat(m, expected_output.data(), ith, gemm_nth);
    direct_c->to_mat(m, direct_output.data(), ith, gemm_nth);
  }

  const bool equal = std::memcmp(expected_output.data(), direct_output.data(),
                                 expected_output.size() * sizeof(ggml_bf16_t)) == 0;
  bool nonzero = false;
  for (const auto value : direct_output) nonzero = nonzero || value.bits != 0;
  std::printf("raw FP8 backward mat_mul_kgroup: equal=%s nonzero=%s\n", equal ? "PASS" : "FAIL",
              nonzero ? "PASS" : "FAIL");

  std::free(source_memory);
  std::free(expected_memory);
  std::free(direct_memory);
  std::free(a_memory);
  std::free(expected_c_memory);
  std::free(direct_c_memory);
  return equal && nonzero;
}

}  // namespace

int main() {
  bool passed = true;
  passed = run_repack_case(128, 128, 1) && passed;
  passed = run_repack_case(128, 256, 1) && passed;
  passed = run_repack_case(256, 128, 2) && passed;
  passed = run_repack_case(256, 384, 3) && passed;
  passed = run_alignment_rejection() && passed;
  passed = run_group_rejection() && passed;
  passed = run_pool_alignment_case() && passed;
  passed = run_backward_gemm_case() && passed;
  return passed ? 0 : 1;
}
