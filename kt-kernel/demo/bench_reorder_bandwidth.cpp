#include <blis.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
constexpr int kM = 1;
constexpr int kK = 7168;
constexpr int kN = 512;
constexpr int kIters = 10000;

void fill_random(int8_t* ptr, size_t count) {
  std::srand(47);
  for (size_t i = 0; i < count; ++i) {
    ptr[i] = static_cast<int8_t>(std::rand() % 30);
  }
}

void fill_zero(int32_t* ptr, size_t count) { std::memset(ptr, 0, count * sizeof(int32_t)); }

bool verify(const int8_t* a, const int8_t* b, const int32_t* c) {
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int32_t ref = 0;
      for (int k = 0; k < kK; ++k) {
        ref += static_cast<int32_t>(a[m * kK + k]) * static_cast<int32_t>(b[n * kK + k]);
      }
      if (ref != c[m * kN + n]) {
        std::printf("Mismatch at (%d, %d): got %d, expect %d\n", m, n, c[m * kN + n], ref);
        return false;
      }
    }
  }
  return true;
}
}  // namespace

int main() {
  int8_t* a = static_cast<int8_t*>(std::aligned_alloc(64, kM * kK));
  int8_t* b = static_cast<int8_t*>(std::aligned_alloc(64, kK * kN));
  int32_t* c = static_cast<int32_t*>(std::aligned_alloc(64, kM * kN * sizeof(int32_t)));
  int32_t* c_tmp = static_cast<int32_t*>(std::aligned_alloc(64, kM * kN * sizeof(int32_t)));

  if (!a || !b || !c || !c_tmp) {
    std::fprintf(stderr, "Allocation failed.\n");
    std::free(a);
    std::free(b);
    std::free(c);
    std::free(c_tmp);
    return EXIT_FAILURE;
  }

  fill_random(a, kM * kK);
  fill_random(b, kK * kN);
  fill_zero(c, kM * kN);
  fill_zero(c_tmp, kM * kN);

  const dim_t reorder_size = aocl_get_reorder_buf_size_s8s8s32os32('r', 't', 'B', kK, kN);
  int8_t* b_reordered = static_cast<int8_t*>(std::aligned_alloc(64, reorder_size));
  if (!b_reordered) {
    std::fprintf(stderr, "Reorder buffer allocation failed.\n");
    std::free(a);
    std::free(b);
    std::free(c);
    return EXIT_FAILURE;
  }

  aocl_reorder_s8s8s32os32('r', 't', 'B', b, b_reordered, kK, kN, kK);

  // Warm-up GEMM to load kernels.
  aocl_gemm_s8s8s32os32('r', 'n', 't', kM, kN, kK, 1, a, kK, 'n', b_reordered, kK, 'r', 0, c_tmp, kN, nullptr);
  fill_zero(c, kM * kN);

  const double bytes_per_mul = static_cast<double>(kM) * kK * sizeof(int8_t) +  // A matrix read
                               static_cast<double>(kK) * kN * sizeof(int8_t);   // original B read

  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < kIters; ++iter) {
    aocl_gemm_s8s8s32os32('r', 'n', 't', kM, kN, kK, 1, a, kK, 'n', b_reordered, kK, 'r', 0, c, kN, nullptr);
  }
  auto end = std::chrono::high_resolution_clock::now();

  const double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  const double total_bytes = bytes_per_mul * kIters;
  const double bandwidth_gbps = total_bytes / elapsed_seconds / 1e9;
  const double ops_per_mul = static_cast<double>(kM) * kN * kK * 2.0;
  const double tflops = (ops_per_mul * kIters) / elapsed_seconds / 1e12;

  std::printf("Reorder buffer size: %ld bytes\n", static_cast<long>(reorder_size));
  std::printf("Iterations: %d\n", kIters);
  std::printf("Elapsed time: %.4f s\n", elapsed_seconds);
  std::printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbps);
  std::printf("Int8 GEMM throughput: %.2f TOPS\n", tflops * 1e3);

  if (!verify(a, b, c)) {
    std::fprintf(stderr, "Verification failed.\n");
  } else {
    std::puts("Verification passed.");
  }

  std::free(a);
  std::free(b);
  std::free(b_reordered);
  std::free(c);
  return 0;
}
