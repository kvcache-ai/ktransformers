#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../la/bf16_dweight.hpp"

namespace {

using DWeightKernel = amx::BF16DWeightKernel;
using Kernel = DWeightKernel::Kernel;

void* alloc_buffer(size_t bytes) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, 64, (bytes + 63) / 64 * 64) != 0 || pointer == nullptr) std::abort();
  std::memset(pointer, 0, bytes);
  return pointer;
}

void fill_random(std::vector<ggml_bf16_t>& values, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
  for (auto& value : values) value = GGML_FP32_TO_BF16(distribution(generator));
}

bool run_case(int routes, int rows, int columns) {
  constexpr int source_column = 2;
  constexpr int destination_column = 3;
  const int lhs_stride = rows + source_column + 3;
  const int rhs_stride = columns + source_column + 5;
  const int destination_stride = columns + destination_column + 7;
  const int padded_k = DWeightKernel::padded_k(routes);

  std::vector<ggml_bf16_t> lhs(static_cast<size_t>(routes) * lhs_stride);
  std::vector<ggml_bf16_t> rhs(static_cast<size_t>(routes) * rhs_stride);
  std::vector<ggml_bf16_t> actual(static_cast<size_t>(rows) * destination_stride);
  std::vector<ggml_bf16_t> expected(static_cast<size_t>(rows) * columns);
  fill_random(lhs, static_cast<unsigned>(routes * 17 + rows));
  fill_random(rhs, static_cast<unsigned>(routes * 31 + columns));

  void* a_memory = alloc_buffer(Kernel::BufferA::required_size(Kernel::M_STEP, padded_k));
  void* b_memory = alloc_buffer(Kernel::BufferB::required_size(Kernel::N_STEP, padded_k));
  Kernel::BufferA a(Kernel::M_STEP, padded_k, a_memory);
  Kernel::BufferB b(Kernel::N_STEP, padded_k, b_memory);
  alignas(64) float accumulator[Kernel::M_STEP * Kernel::N_STEP];

  DWeightKernel::pack_a_transposed(a, lhs.data(), lhs_stride, source_column, rows, routes);
  DWeightKernel::pack_b_transposed(b, rhs.data(), rhs_stride, source_column, columns, routes);
  DWeightKernel::multiply(padded_k, accumulator, a, b);
  DWeightKernel::store_bf16(accumulator, actual.data() + destination_column, destination_stride, rows, columns);

  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float sum = 0.0f;
      for (int route = 0; route < routes; ++route) {
        sum += GGML_BF16_TO_FP32(lhs[static_cast<size_t>(route) * lhs_stride + source_column + row]) *
               GGML_BF16_TO_FP32(rhs[static_cast<size_t>(route) * rhs_stride + source_column + column]);
      }
      expected[static_cast<size_t>(row) * columns + column] = GGML_FP32_TO_BF16(sum);
    }
  }

  double difference_sq = 0.0;
  double expected_sq = 0.0;
  double actual_sq = 0.0;
  double dot = 0.0;
  float max_abs = 0.0f;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const float expected_value = GGML_BF16_TO_FP32(expected[static_cast<size_t>(row) * columns + column]);
      const float actual_value =
          GGML_BF16_TO_FP32(actual[static_cast<size_t>(row) * destination_stride + destination_column + column]);
      const double difference = static_cast<double>(actual_value) - expected_value;
      difference_sq += difference * difference;
      expected_sq += static_cast<double>(expected_value) * expected_value;
      actual_sq += static_cast<double>(actual_value) * actual_value;
      dot += static_cast<double>(expected_value) * actual_value;
      max_abs = std::max(max_abs, std::fabs(actual_value - expected_value));
    }
  }

  const double relative_l2 = std::sqrt(difference_sq / std::max(expected_sq, 1e-30));
  const double cosine = dot / std::sqrt(std::max(expected_sq * actual_sq, 1e-30));
  const bool passed = relative_l2 <= 0.01 && cosine >= 0.999;
  std::printf("BF16 dWeight routes=%d shape=%dx%d: rel_l2=%.6e cosine=%.9f max_abs=%.6e %s\n", routes, rows, columns,
              relative_l2, cosine, max_abs, passed ? "PASS" : "FAIL");

  std::free(a_memory);
  std::free(b_memory);
  return passed;
}

}  // namespace

int main() {
  DWeightKernel::configure_worker();
  bool passed = true;
  for (int routes : {1, 31, 32, 33, 65, 1792, 1825}) {
    passed = run_case(routes, 32, 32) && passed;
  }
  passed = run_case(33, 17, 29) && passed;
  passed = run_case(65, 31, 7) && passed;
  return passed ? 0 : 1;
}
