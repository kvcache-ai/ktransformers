#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#include "../la/amx_kernels.hpp"
#include "../la/rawint4_backward.hpp"

#if defined(__AVX512BF16__)
namespace forward_test {
using Kernel = amx::GemmKernel224Int4SmallKGroup;

static void* storage(size_t bytes) {
  void* pointer = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
  if (!pointer) std::abort();
  return pointer;
}

static uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

__attribute__((optimize("no-fast-math"), noinline)) static double scalar_dot(Kernel::BufferA& a, Kernel::BufferB& b,
                                                                             int row, int col, int m, int k) {
  double sum = 0;
  for (int group = 0; group < k / 32; ++group) {
    int dot = 0;
    for (int offset = 0; offset < 32; ++offset) {
      const int kk = group * 32 + offset;
      const uint8_t packed = b.b[size_t(col) * k / 2 + kk / 2];
      const int weight = ((packed >> ((kk % 2) * 4)) & 15) - 8;
      dot += int(*a.get_submat(m, k, row, kk)) * weight;
    }
    const float scale = a.d[size_t(row) * (k / 32) + group] * b.d[size_t(col) * (k / 32) + group];
    sum += double(dot) * scale;
  }
  return sum;
}

template <bool BlockedA>
static void reference_rows(int m, int n, int k, int first, int last, Kernel::BufferA& a, Kernel::BufferB& b,
                           Kernel::BufferC& c) {
  int row = 0;
  for (; row + 8 <= m; row += 8) Kernel::matmat_rows<8, BlockedA>(m, n, k, row, first, last, &a, &b, &c);
  if (row + 4 <= m) {
    Kernel::matmat_rows<4, BlockedA>(m, n, k, row, first, last, &a, &b, &c);
    row += 4;
  }
  if (row + 2 <= m) {
    Kernel::matmat_rows<2, BlockedA>(m, n, k, row, first, last, &a, &b, &c);
    row += 2;
  }
  if (row < m) Kernel::matmat_rows<1, BlockedA>(m, n, k, row, first, last, &a, &b, &c);
}

int run() {
  std::mt19937 rng(1979);
  size_t cases = 0, fp32_differences = 0;
  double maximum_error = 0;
  for (int k : {32, 96, 1024, 1056, 2048, 7168, 7232}) {
    for (int m : {1, 7, 15, 16, 17, 31, 43, 127, 128, 129, 255, 256, 257, 513}) {
      constexpr int n = 288;
      const size_t asize = Kernel::BufferA::required_size(m, k, 32);
      const size_t bsize = Kernel::BufferB::required_size(n, k, 32);
      const size_t csize = Kernel::BufferC::required_size(m, n);
      void *ap = storage(asize), *bp = storage(bsize), *cp = storage(csize + 64);
      Kernel::BufferA a(m, k, 32, ap);
      Kernel::BufferB b(n, k, 32, bp);
      Kernel::BufferC c(m, n, cp);
      for (int i = 0; i < m * k; ++i) a.a[i] = int(rng() % 255) - 127;
      for (int i = 0; i < m * k / 32; ++i) a.d[i] = float(1 + rng() % 256) / 63331;
      for (int i = 0; i < n * k / 2; ++i) b.b[i] = rng() % 256;
      for (int i = 0; i < n * k / 32; ++i) b.d[i] = float(1 + rng() % 256) / 64839;
      for (int first : {0, 32}) {
        const int last = first ? 96 : n;
        std::memset(cp, 0x5a, csize + 64);
        if (k <= Kernel::K_BLOCK)
          reference_rows<false>(m, n, k, first, last, a, b, c);
        else
          reference_rows<true>(m, n, k, first, last, a, b, c);
        const std::vector<float> reference(c.c, c.c + csize / sizeof(float));
        for (int partition : {0, 8, 256}) {
          std::memset(cp, 0x5a, csize + 64);
          const int block = partition ? partition : m;
          for (int row = 0; row < m; row += block) {
            const int end = std::min(m, row + block);
            if (k <= Kernel::K_BLOCK)
              Kernel::matmat_avx512<false>(m, n, k, first, last, &a, &b, &c, row, end);
            else
              Kernel::matmat_avx512<true>(m, n, k, first, last, &a, &b, &c, row, end);
          }
          for (size_t i = 0; i < reference.size(); ++i) {
            fp32_differences += reference[i] != c.c[i];
            if (bf16(reference[i]) != bf16(c.c[i])) {
              std::fprintf(stderr, "BF16 mismatch m=%d n=%d k=%d first=%d partition=%d index=%zu %.9g %.9g\n", m, n, k,
                           first, partition, i, reference[i], c.c[i]);
              return 3;
            }
          }
          const auto* guard = static_cast<const uint8_t*>(cp) + csize;
          for (int i = 0; i < 64; ++i)
            if (guard[i] != 0x5a) return 4;
          for (int sample = 0; sample < 8; ++sample) {
            const int row = rng() % m, col = first + rng() % (last - first);
            const double gold = scalar_dot(a, b, row, col, m, k);
            const double error = std::abs(double(*c.get_submat(m, n, row, col)) - gold);
            maximum_error = std::max(maximum_error, error);
            if (error > 1e-6 + 1e-5 * std::abs(gold)) return 5;
          }
          ++cases;
        }
      }
      std::free(ap);
      std::free(bp);
      std::free(cp);
    }
  }
  if (fp32_differences != 0) return 6;
  std::printf("{\"passed\":true,\"cases\":%zu,\"fp32_differences\":%zu,\"fp64_max_error\":%.9g}\n", cases,
              fp32_differences, maximum_error);
  return 0;
}
}  // namespace forward_test

namespace backward_test {
static uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

static float fp32(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int run() {
  std::mt19937 rng(817);
  std::uniform_real_distribution<float> random(-0.3f, 0.3f);
  int cases = 0;
  double max_error = 0;
  for (int size : {2, 6, 16, 18, 32, 50}) {
    std::vector<float> gate(size), up(size);
    std::vector<uint16_t> out(2 * size + 16, 0x7fc1);
    for (int i = 0; i < size; ++i) {
      gate[i] = random(rng);
      up[i] = random(rng);
    }
    rawint4::pack_gate_up_coeff(gate.data(), up.data(), out.data(), size);
    for (int i = 0; i < size; ++i) {
      if (out[(i / 2) * 4 + i % 2] != bf16(gate[i]) || out[(i / 2) * 4 + 2 + i % 2] != bf16(up[i]))
        throw std::runtime_error("coefficient packing");
    }
    for (size_t i = 2 * size; i < out.size(); ++i)
      if (out[i] != 0x7fc1) throw std::runtime_error("coefficient overrun");
  }
  for (int m : {1, 3, 4, 7, 8, 16, 17, 31, 32, 33, 65}) {
    for (int k : {2, 32, 96, 1024}) {
      for (int width : {32, 64, 96}) {
        for (bool scatter : {false, true}) {
          constexpr int n = 224, begin = 32;
          const int output_rows = scatter ? 2 * m + 1 : m;
          std::vector<uint16_t> a(size_t(m) * k), b(size_t(k) * n);
          std::vector<float> out(size_t(output_rows) * n, 0.123f);
          std::vector<int> mapping(m);
          std::vector<bool> touched(out.size(), false);
          for (auto& value : a) value = bf16(random(rng));
          for (auto& value : b) value = bf16(random(rng));
          for (int row = 0; row < m; ++row) mapping[row] = 2 * (m - row - 1) + 1;
          auto decode = [&](int pair, int col, __m512bh& lo, __m512bh& hi) {
            alignas(64) uint16_t packed[64];
            for (int i = 0; i < 32; ++i) {
              packed[2 * i] = b[size_t(pair * 2) * n + col + i];
              packed[2 * i + 1] = b[size_t(pair * 2 + 1) * n + col + i];
            }
            lo = (__m512bh)_mm512_load_si512(packed);
            hi = (__m512bh)_mm512_load_si512(packed + 32);
          };
          rawint4::backward(a.data(), m, k, n, begin, begin + width, scatter ? mapping.data() : nullptr, out.data(),
                            decode);
          for (int row = 0; row < m; ++row) {
            for (int col = begin; col < begin + width; ++col) {
              const size_t index = size_t(scatter ? mapping[row] : row) * n + col;
              double reference = scatter ? double(0.123f) : 0.0;
              for (int inner = 0; inner < k; ++inner)
                reference += double(fp32(a[size_t(row) * k + inner])) * fp32(b[size_t(inner) * n + col]);
              const double error = std::abs(double(out[index]) - reference);
              max_error = std::max(max_error, error);
              if (!std::isfinite(out[index]) || error > 1e-5 + 2e-5 * std::abs(reference))
                throw std::runtime_error("matrix product or scatter");
              touched[index] = true;
            }
          }
          for (size_t i = 0; i < out.size(); ++i)
            if (!touched[i] && out[i] != 0.123f) throw std::runtime_error("output overrun");
          ++cases;
        }
      }
    }
  }
  std::printf("{\"passed\":true,\"cases\":%d,\"amx\":%s,\"max_error\":%.9g}\n", cases,
              rawint4::use_amx() ? "true" : "false", max_error);
  return 0;
}
}  // namespace backward_test

int main() {
  const int status = forward_test::run();
  if (status != 0) return status;
  return backward_test::run();
}
#else
int main() {
  std::puts("SKIP: RAWINT4 SFT panel tests require AVX512-BF16");
  return 0;
}
#endif
