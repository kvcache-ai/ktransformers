#include <algorithm>
#include <chrono>
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

bool run_shared_panel_case(int routes, int rows, int columns) {
  const int padded_k = DWeightKernel::padded_k(routes);
  const int padded_rows = (rows + Kernel::M_STEP - 1) / Kernel::M_STEP * Kernel::M_STEP;
  const int padded_columns = (columns + Kernel::N_STEP - 1) / Kernel::N_STEP * Kernel::N_STEP;
  std::vector<ggml_bf16_t> lhs(static_cast<size_t>(routes) * rows);
  std::vector<ggml_bf16_t> rhs(static_cast<size_t>(routes) * columns);
  std::vector<ggml_bf16_t> actual(static_cast<size_t>(rows) * columns);
  fill_random(lhs, static_cast<unsigned>(routes * 41 + rows));
  fill_random(rhs, static_cast<unsigned>(routes * 43 + columns));

  void* a_memory = alloc_buffer(Kernel::BufferA::required_size(padded_rows, padded_k));
  void* b_memory = alloc_buffer(Kernel::BufferB::required_size(padded_columns, padded_k));
  Kernel::BufferA a(padded_rows, padded_k, a_memory);
  Kernel::BufferB b(padded_columns, padded_k, b_memory);
  for (int row = 0; row < rows; row += Kernel::M_STEP) {
    DWeightKernel::pack_a_transposed(a, lhs.data(), rows, row, std::min(Kernel::M_STEP, rows - row), routes, row);
  }
  for (int column = 0; column < columns; column += Kernel::N_STEP) {
    DWeightKernel::pack_b_transposed(b, rhs.data(), columns, column,
                                     std::min(Kernel::N_STEP, columns - column), routes, column);
  }

  alignas(64) float accumulator[Kernel::M_STEP * Kernel::N_STEP];
  for (int row = 0; row < rows; row += Kernel::M_STEP) {
    const int row_count = std::min(Kernel::M_STEP, rows - row);
    for (int column = 0; column < columns; column += Kernel::N_STEP) {
      const int column_count = std::min(Kernel::N_STEP, columns - column);
      DWeightKernel::multiply(padded_k, accumulator, a, b, row, column);
      DWeightKernel::store_bf16(accumulator, actual.data() + static_cast<size_t>(row) * columns + column, columns,
                                row_count, column_count);
    }
  }

  double difference_sq = 0.0;
  double expected_sq = 0.0;
  double actual_sq = 0.0;
  double dot = 0.0;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float reference = 0.0f;
      for (int route = 0; route < routes; ++route) {
        reference += GGML_BF16_TO_FP32(lhs[static_cast<size_t>(route) * rows + row]) *
                     GGML_BF16_TO_FP32(rhs[static_cast<size_t>(route) * columns + column]);
      }
      const float expected = GGML_BF16_TO_FP32(GGML_FP32_TO_BF16(reference));
      const float value = GGML_BF16_TO_FP32(actual[static_cast<size_t>(row) * columns + column]);
      const double difference = static_cast<double>(value) - expected;
      difference_sq += difference * difference;
      expected_sq += static_cast<double>(expected) * expected;
      actual_sq += static_cast<double>(value) * value;
      dot += static_cast<double>(expected) * value;
    }
  }
  const double relative_l2 = std::sqrt(difference_sq / std::max(expected_sq, 1e-30));
  const double cosine = dot / std::sqrt(std::max(expected_sq * actual_sq, 1e-30));
  const bool passed = relative_l2 <= 0.01 && cosine >= 0.999;
  std::printf("BF16 dWeight shared panel routes=%d shape=%dx%d: rel_l2=%.6e cosine=%.9f %s\n", routes, rows,
              columns, relative_l2, cosine, passed ? "PASS" : "FAIL");

  std::free(a_memory);
  std::free(b_memory);
  return passed;
}

bool run_store_mode_case(int rows, int columns, int destination_stride, bool accumulate, float scale) {
  constexpr int prefix_guard_elements = 5;
  constexpr int suffix_guard_elements = 11;
  const ggml_bf16_t guard = GGML_FP32_TO_BF16(-123.0f);

  alignas(64) float source[Kernel::M_STEP * Kernel::N_STEP];
  for (int row = 0; row < Kernel::M_STEP; ++row) {
    for (int column = 0; column < Kernel::N_STEP; ++column) {
      source[row * Kernel::N_STEP + column] =
          static_cast<float>((row * Kernel::N_STEP + column) % 41 - 20) * 0.03125f;
    }
  }

  std::vector<ggml_bf16_t> storage(
      prefix_guard_elements + static_cast<size_t>(rows) * destination_stride + suffix_guard_elements, guard);
  std::vector<ggml_bf16_t> expected(static_cast<size_t>(rows) * columns);
  ggml_bf16_t* destination = storage.data() + prefix_guard_elements;

  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const float initial = static_cast<float>((row * columns + column) % 17 - 8) * 0.125f;
      destination[static_cast<size_t>(row) * destination_stride + column] = GGML_FP32_TO_BF16(initial);
      const float old = GGML_BF16_TO_FP32(destination[static_cast<size_t>(row) * destination_stride + column]);
      const float value = source[row * Kernel::N_STEP + column] * scale + (accumulate ? old : 0.0f);
      expected[static_cast<size_t>(row) * columns + column] = GGML_FP32_TO_BF16(value);
    }
  }

  DWeightKernel::store_bf16(source, destination, destination_stride, rows, columns, accumulate, scale);

  bool passed = true;
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const ggml_bf16_t actual = destination[static_cast<size_t>(row) * destination_stride + column];
      if (actual.bits != expected[static_cast<size_t>(row) * columns + column].bits) passed = false;
    }
    for (int column = columns; column < destination_stride; ++column) {
      if (destination[static_cast<size_t>(row) * destination_stride + column].bits != guard.bits) passed = false;
    }
  }
  for (int i = 0; i < prefix_guard_elements; ++i) {
    if (storage[i].bits != guard.bits) passed = false;
  }
  const size_t suffix_begin = prefix_guard_elements + static_cast<size_t>(rows) * destination_stride;
  for (int i = 0; i < suffix_guard_elements; ++i) {
    if (storage[suffix_begin + i].bits != guard.bits) passed = false;
  }

  std::printf("BF16 dWeight store mode=%s scale=%.4f shape=%dx%d stride=%d guards=%s %s\n",
              accumulate ? "accumulate" : "overwrite", scale, rows, columns, destination_stride,
              passed ? "intact" : "CORRUPTED", passed ? "PASS" : "FAIL");
  return passed;
}

bool run_amx_benchmark() {
  if constexpr (!amx::AMX_AVAILABLE) {
    std::printf("BF16 dWeight AMX benchmark: SKIP (AMX unavailable)\n");
    return true;
  }

  constexpr int routes = 1024;
  constexpr int iterations = 20000;
  constexpr int rounds = 7;
  const int padded_k = DWeightKernel::padded_k(routes);
  std::vector<ggml_bf16_t> lhs(static_cast<size_t>(routes) * Kernel::M_STEP);
  std::vector<ggml_bf16_t> rhs(static_cast<size_t>(routes) * Kernel::N_STEP);
  fill_random(lhs, 20260717);
  fill_random(rhs, 20260718);

  void* a_memory = alloc_buffer(Kernel::BufferA::required_size(Kernel::M_STEP, padded_k));
  void* b_memory = alloc_buffer(Kernel::BufferB::required_size(Kernel::N_STEP, padded_k));
  Kernel::BufferA a(Kernel::M_STEP, padded_k, a_memory);
  Kernel::BufferB b(Kernel::N_STEP, padded_k, b_memory);
  alignas(64) float accumulator[Kernel::M_STEP * Kernel::N_STEP];
  DWeightKernel::pack_a_transposed(a, lhs.data(), Kernel::M_STEP, 0, Kernel::M_STEP, routes);
  DWeightKernel::pack_b_transposed(b, rhs.data(), Kernel::N_STEP, 0, Kernel::N_STEP, routes);

  auto legacy_tile_loop = [&] {
    Kernel::clean_c();
    for (int k_begin = 0; k_begin < padded_k; k_begin += Kernel::K_STEP) {
      Kernel::load_b(b.get_submat(Kernel::N_STEP, padded_k, 0, k_begin),
                     Kernel::K_STEP * sizeof(ggml_bf16_t));
      Kernel::load_a(a.get_submat(Kernel::M_STEP, padded_k, 0, k_begin),
                     Kernel::K_STEP * sizeof(ggml_bf16_t));
      Kernel::run_tile();
    }
    Kernel::store_c(accumulator, Kernel::N_STEP * sizeof(float));
  };
  auto common_driver = [&] { DWeightKernel::multiply(padded_k, accumulator, a, b); };

  for (int warmup = 0; warmup < 200; ++warmup) {
    legacy_tile_loop();
    common_driver();
  }

  auto measure = [&](auto&& operation) {
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) operation();
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count() / iterations;
  };
  std::vector<double> legacy_ns;
  std::vector<double> common_ns;
  for (int round = 0; round < rounds; ++round) {
    if (round % 2 == 0) {
      legacy_ns.push_back(measure(legacy_tile_loop));
      common_ns.push_back(measure(common_driver));
    } else {
      common_ns.push_back(measure(common_driver));
      legacy_ns.push_back(measure(legacy_tile_loop));
    }
  }
  std::sort(legacy_ns.begin(), legacy_ns.end());
  std::sort(common_ns.begin(), common_ns.end());
  const double legacy_median = legacy_ns[rounds / 2];
  const double common_median = common_ns[rounds / 2];
  const double ratio = common_median / legacy_median;
  const bool passed = ratio <= 1.05;
  std::printf("BF16 dWeight AMX kernel routes=%d: legacy=%.1f ns common=%.1f ns ratio=%.4f %s\n", routes,
              legacy_median, common_median, ratio, passed ? "PASS" : "FAIL");

  std::free(a_memory);
  std::free(b_memory);
  return passed;
}

bool run_avx_benchmark() {
  if constexpr (amx::AMX_AVAILABLE) {
    std::printf("BF16 dWeight AVX benchmark: SKIP (AMX enabled)\n");
    return true;
  }

  constexpr int routes = 64;
  constexpr int iterations = 50000;
  constexpr int rounds = 7;
  const int padded_k = DWeightKernel::padded_k(routes);
  std::vector<ggml_bf16_t> lhs(static_cast<size_t>(routes) * Kernel::M_STEP);
  std::vector<ggml_bf16_t> rhs(static_cast<size_t>(routes) * Kernel::N_STEP);
  fill_random(lhs, 20260717);
  fill_random(rhs, 20260718);

  void* a_memory = alloc_buffer(Kernel::BufferA::required_size(Kernel::M_STEP, padded_k));
  void* b_memory = alloc_buffer(Kernel::BufferB::required_size(Kernel::N_STEP, padded_k));
  Kernel::BufferA a(Kernel::M_STEP, padded_k, a_memory);
  Kernel::BufferB b(Kernel::N_STEP, padded_k, b_memory);
  alignas(64) float accumulator[Kernel::M_STEP * Kernel::N_STEP];
  DWeightKernel::pack_a_transposed(a, lhs.data(), Kernel::M_STEP, 0, Kernel::M_STEP, routes);
  DWeightKernel::pack_b_transposed(b, rhs.data(), Kernel::N_STEP, 0, Kernel::N_STEP, routes);

  auto generic_driver = [&] {
    for (int k_block_begin = 0; k_block_begin < padded_k; k_block_begin += Kernel::K_BLOCK) {
      Kernel::avx_kernel_4(Kernel::M_STEP, Kernel::N_STEP, padded_k, 0, 0, k_block_begin, accumulator, &a, &b);
    }
  };
  auto register_blocked_driver = [&] { DWeightKernel::multiply(padded_k, accumulator, a, b); };

  for (int warmup = 0; warmup < 200; ++warmup) {
    generic_driver();
    register_blocked_driver();
  }
  auto measure = [&](auto&& operation) {
    const auto begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) operation();
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - begin).count() / iterations;
  };

  std::vector<double> generic_ns;
  std::vector<double> register_blocked_ns;
  for (int round = 0; round < rounds; ++round) {
    if (round % 2 == 0) {
      generic_ns.push_back(measure(generic_driver));
      register_blocked_ns.push_back(measure(register_blocked_driver));
    } else {
      register_blocked_ns.push_back(measure(register_blocked_driver));
      generic_ns.push_back(measure(generic_driver));
    }
  }
  std::sort(generic_ns.begin(), generic_ns.end());
  std::sort(register_blocked_ns.begin(), register_blocked_ns.end());
  const double generic_median = generic_ns[rounds / 2];
  const double register_blocked_median = register_blocked_ns[rounds / 2];
  const double ratio = register_blocked_median / generic_median;
  const bool passed = ratio <= 1.05;
  std::printf("BF16 dWeight AVX kernel routes=%d: generic=%.1f ns register_blocked=%.1f ns ratio=%.4f %s\n", routes,
              generic_median, register_blocked_median, ratio, passed ? "PASS" : "FAIL");

  std::free(a_memory);
  std::free(b_memory);
  return passed;
}

}  // namespace

int main(int argc, char** argv) {
  DWeightKernel::configure_worker();
  if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
    return (run_amx_benchmark() && run_avx_benchmark()) ? 0 : 1;
  }
  bool passed = true;
  for (int routes : {1, 31, 32, 33, 65, 1792, 1825}) {
    passed = run_case(routes, 32, 32) && passed;
  }
  passed = run_case(33, 17, 29) && passed;
  passed = run_case(65, 31, 7) && passed;
  passed = run_shared_panel_case(33, 45, 77) && passed;
  passed = run_shared_panel_case(65, 64, 96) && passed;
  passed = run_store_mode_case(32, 32, 41, false, 1.0f) && passed;
  passed = run_store_mode_case(32, 32, 39, true, 0.5f) && passed;
  passed = run_store_mode_case(17, 29, 37, false, -0.25f) && passed;
  passed = run_store_mode_case(31, 7, 19, true, 0.125f) && passed;
  return passed ? 0 : 1;
}
