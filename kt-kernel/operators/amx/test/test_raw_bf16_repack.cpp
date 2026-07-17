#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../la/amx_kernels.hpp"
#include "../la/amx_raw_kernels.hpp"

namespace {

using Kernel = amx::GemmKernel224BF16;
using BufferB = Kernel::BufferB;

void* alloc_buffer(size_t bytes) {
  void* ptr = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
  if (ptr == nullptr) std::abort();
  std::memset(ptr, 0, bytes);
  return ptr;
}

void fill_random(std::vector<ggml_bf16_t>& values, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (auto& value : values) value = GGML_FP32_TO_BF16(distribution(generator));
}

void from_mat(BufferB& buffer, ggml_bf16_t* source) {
  int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ith++) buffer.from_mat(source, ith, nth);
}

void from_mat_strided(BufferB& buffer, ggml_bf16_t* source, int source_stride) {
  int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ith++) buffer.from_mat_strided(source, source_stride, ith, nth);
}

void to_mat(const BufferB& buffer, ggml_bf16_t* destination) {
  int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ith++) buffer.to_mat(destination, ith, nth);
}

void from_mat_transposed(BufferB& buffer, ggml_bf16_t* source, int source_n, int source_k) {
  int nth = Kernel::recommended_nth(buffer.n);
  for (int ith = 0; ith < nth; ith++) buffer.from_mat_transposed(source, source_n, source_k, ith, nth);
}

void from_bb_transposed(BufferB& destination, const BufferB& source) {
  int nth = Kernel::recommended_nth(destination.n);
  for (int ith = 0; ith < nth; ith++) destination.from_bb_transposed(source, ith, nth);
}

bool run_case(int n, int k) {
  const size_t count = (size_t)n * k;
  std::vector<ggml_bf16_t> source(count);
  std::vector<ggml_bf16_t> roundtrip(count);
  std::vector<ggml_bf16_t> transposed(count);
  std::vector<ggml_bf16_t> direct_transposed(count);
  fill_random(source, (unsigned)(n * 31 + k));

  void* forward_memory = alloc_buffer(BufferB::required_size(n, k));
  void* expected_memory = alloc_buffer(BufferB::required_size(k, n));
  void* direct_memory = alloc_buffer(BufferB::required_size(k, n));
  BufferB forward(n, k, forward_memory);
  BufferB expected(k, n, expected_memory);
  BufferB direct(k, n, direct_memory);

  from_mat(forward, source.data());
  to_mat(forward, roundtrip.data());
  from_mat_transposed(expected, source.data(), n, k);
  from_bb_transposed(direct, forward);
  to_mat(expected, transposed.data());
  to_mat(direct, direct_transposed.data());

  bool roundtrip_ok = std::memcmp(source.data(), roundtrip.data(), count * sizeof(ggml_bf16_t)) == 0;
  bool direct_ok = std::memcmp(transposed.data(), direct_transposed.data(), count * sizeof(ggml_bf16_t)) == 0;
  bool transpose_ok = true;
  for (int row = 0; row < n && transpose_ok; row++) {
    for (int column = 0; column < k; column++) {
      if (source[(size_t)row * k + column].bits != transposed[(size_t)column * n + row].bits) {
        transpose_ok = false;
        break;
      }
    }
  }

  std::free(forward_memory);
  std::free(expected_memory);
  std::free(direct_memory);
  std::printf("raw BF16 repack %dx%d: roundtrip=%s transpose=%s direct=%s\n", n, k, roundtrip_ok ? "PASS" : "FAIL",
              transpose_ok ? "PASS" : "FAIL", direct_ok ? "PASS" : "FAIL");
  return roundtrip_ok && transpose_ok && direct_ok;
}

bool run_strided_case(int n, int k, int source_stride, size_t source_offset) {
  const size_t source_count = source_offset + (size_t)(n - 1) * source_stride + k;
  const size_t output_count = (size_t)n * k;
  std::vector<ggml_bf16_t> source(source_count);
  std::vector<ggml_bf16_t> output(output_count);
  fill_random(source, (unsigned)(n * 17 + k * 13 + source_stride));

  void* memory = alloc_buffer(BufferB::required_size(n, k));
  BufferB packed(n, k, memory);
  from_mat_strided(packed, source.data() + source_offset, source_stride);
  to_mat(packed, output.data());

  bool passed = true;
  for (int row = 0; row < n && passed; row++) {
    for (int column = 0; column < k; column++) {
      if (source[source_offset + (size_t)row * source_stride + column].bits != output[(size_t)row * k + column].bits) {
        passed = false;
        break;
      }
    }
  }

  std::free(memory);
  std::printf("raw BF16 strided repack %dx%d stride=%d offset=%zu: %s\n", n, k, source_stride, source_offset,
              passed ? "PASS" : "FAIL");
  return passed;
}

}  // namespace

int main() {
  bool passed = true;
  passed = run_case(64, 64) && passed;
  passed = run_case(768, 2048) && passed;
  passed = run_case(2048, 768) && passed;
  passed = run_strided_case(64, 64, 64, 32 * 64) && passed;
  passed = run_strided_case(64, 64, 96, 17) && passed;
  return passed ? 0 : 1;
}
