// SPDX-License-Identifier: Apache-2.0
// Check the shared production dispatch, without importing tuning prototypes.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "../la/amx_kernels.hpp"
#include "../la/amx_raw_kernels.hpp"

namespace {
using Kernel = amx::GemmKernel224FP8;

// Preserve the pre-optimization row kernel and K/M/N traversal as the reference.
struct ReferenceKernel : Kernel {
  static void avx_kernel(int m, int n, int k, int mb, int nb, int kb, float* c, BufferA* a, BufferB* b, int group) {
    avx_kernel_rows(m, n, k, mb, nb, kb, c, a, b, group);
  }
};

using Memory = std::unique_ptr<void, decltype(&std::free)>;

Memory allocate(std::size_t bytes) {
  Memory memory(std::aligned_alloc(64, (bytes + 63) / 64 * 64), &std::free);
  if (!memory) throw std::bad_alloc();
  std::memset(memory.get(), 0, bytes);
  return memory;
}

bool check_shape(int m, int n, int k, int seed) {
  const int padded_m = (m + Kernel::M_STEP - 1) / Kernel::M_STEP * Kernel::M_STEP;
  auto a_memory = allocate(Kernel::BufferA::required_size(padded_m, k));
  auto b_memory = allocate(Kernel::BufferB::required_size(n, k, 128));
  auto c_memory = allocate(Kernel::BufferC::required_size(padded_m, n));
  Kernel::BufferA a(padded_m, k, a_memory.get());
  Kernel::BufferB b(n, k, 128, b_memory.get());
  Kernel::BufferC c(padded_m, n, c_memory.get());

  std::vector<ggml_bf16_t> inputs(std::size_t(m) * k);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const float value = (int((i * 17 + seed) % 127) - 63) / 512.0f;
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    inputs[i] = static_cast<ggml_bf16_t>(bits >> 16);
  }
  a.from_mat(m, inputs.data(), 0, 1);
  std::vector<std::uint8_t> weights(std::size_t(n) * k);
  std::uint32_t state = 0x12345678u + seed;
  for (auto& weight : weights) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    weight = static_cast<std::uint8_t>(state);
    if ((weight & 0x7f) == 0x7f) weight ^= 1;  // Finite E4M3, including subnormals.
  }
  std::vector<float> scales(std::size_t(n / 128) * (k / 128));
  for (std::size_t i = 0; i < scales.size(); ++i) scales[i] = (1 + (i + seed) % 13) / 4096.0f;
  const int parts = Kernel::recommended_nth(n);
  for (int part = 0; part < parts; ++part) b.from_mat(weights.data(), scales.data(), part, parts);
  for (int part = 0; part < parts; ++part)
    amx::float_mat_vec_kgroup<ReferenceKernel, false>(m, n, k, 128, &a, &b, &c, part, parts);

  const std::size_t count = std::size_t(padded_m) * n;
  std::vector<float> reference(count);
  std::memcpy(reference.data(), c.c, count * sizeof(float));
  // Poison valid output rows, retaining the zero-initialized padding. This
  // checks that the first scale group overwrites previous invocation contents.
  for (int mb = 0; mb < m; mb += Kernel::M_STEP)
    for (int nb = 0; nb < n; nb += Kernel::N_STEP)
      std::memset(c.get_submat(m, n, mb, nb), 0x7f, std::min(m - mb, Kernel::M_STEP) * Kernel::N_STEP * sizeof(float));
  for (int part = 0; part < parts; ++part) Kernel::mat_vec_kgroup(m, n, k, 128, &a, &b, &c, part, parts);
  if (std::memcmp(reference.data(), c.c, count * sizeof(float)) != 0) {
    std::fprintf(stderr, "FP32 mismatch: m=%d n=%d k=%d seed=%d\n", m, n, k, seed);
    return false;
  }
  return true;
}
}  // namespace

int main() {
  int checks = 0;
  for (int seed : {0, 42}) {
    for (int m : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 255, 256, 257}) {
      for (int k : {128, 256, 1024, 7168, 7296}) {
        if (!check_shape(m, 128, k, seed)) return 1;
        ++checks;
      }
    }
    for (int m : {65, 256}) {
      if (!check_shape(m, 1024, 7168, seed) || !check_shape(m, 7168, 1024, seed)) return 1;
      checks += 2;
    }
    for (int m : {1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192}) {
      if (!check_shape(m, 128, 256, seed)) return 1;
      ++checks;
    }
  }
  std::printf("PASS: %d production FP8 GEMM bitwise comparisons\n", checks);
}
