
#include <immintrin.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

constexpr size_t DATA_SIZE = 100ULL * 1024 * 1024 * 1024;  // 100 GB
constexpr size_t ALIGNMENT = 64;                           // alignment for AVX-512
constexpr int TEST_ITERATIONS = 100;
constexpr int INNER_TEST_ITERATIONS = 100;

void generate_data(uint8_t* data, size_t size) {
  size_t size_int64 = size / sizeof(int64_t);

#pragma omp parallel
  {
    std::mt19937_64 engine(omp_get_thread_num());
    std::uniform_int_distribution<int64_t> dist;

    int64_t* data64 = reinterpret_cast<int64_t*>(data);

#pragma omp for
    for (size_t i = 0; i < size_int64; ++i) {
      data64[i] = dist(engine);
    }
  }
}

void dpbusd_test(const uint8_t* data_a, const uint8_t* data_b, int32_t* result, size_t size) {
  constexpr size_t simd_width = 64;  // 512 bits = 64 bytes
  size_t vec_count = size / simd_width;

#pragma omp parallel for
  for (size_t x = 0; x < vec_count * INNER_TEST_ITERATIONS; ++x) {
    auto i = x % vec_count;
    __m512i va = _mm512_load_si512(reinterpret_cast<const __m512i*>(data_a + i * simd_width));
    __m512i vb = _mm512_load_si512(reinterpret_cast<const __m512i*>(data_b + i * simd_width));
    __m512i vc = _mm512_setzero_si512();

    vc = _mm512_dpbusd_epi32(vc, va, vb);

    _mm512_store_si512(reinterpret_cast<__m512i*>(result + i * (simd_width / 4)), vc);
  }
}

int main() {
  std::cout << "Allocating aligned memory...\n";
  uint8_t* data_a = reinterpret_cast<uint8_t*>(aligned_alloc(ALIGNMENT, DATA_SIZE));
  uint8_t* data_b = reinterpret_cast<uint8_t*>(aligned_alloc(ALIGNMENT, DATA_SIZE));
  int32_t* result = reinterpret_cast<int32_t*>(aligned_alloc(ALIGNMENT, DATA_SIZE));

  std::cout << "Generating random data...\n";
  generate_data(data_a, DATA_SIZE);
  generate_data(data_b, DATA_SIZE);

  for (int iter = 0; iter < TEST_ITERATIONS; ++iter) {
    std::cout << "Starting computation iteration " << iter + 1 << "...\n";
    auto start = std::chrono::high_resolution_clock::now();

    dpbusd_test(data_a, data_b, result, DATA_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double bandwidth = (3 * DATA_SIZE * INNER_TEST_ITERATIONS) / (1e9) / diff.count();  // GB/s

    std::cout << "Iteration " << iter + 1 << " execution time: " << diff.count() << " s\n";
    std::cout << "Iteration " << iter + 1 << " estimated memory bandwidth: " << bandwidth << " GB/s\n";
  }

  free(data_a);
  free(data_b);
  free(result);

  return 0;
}
