#include <omp.h>

#include "../la/amx.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

void debug_simple_multiplication() {
  std::cout << "=== Debug Simple K-Group Multiplication ===" << std::endl;

  // Very small test case for debugging
  const int m = 32;   // 1 M_STEP
  const int n = 32;   // 1 N_STEP
  const int k = 512;  // Must be at least K_BLOCK (512)
  const int k_group_size = 128;

  std::cout << fmt::format("Parameters: m={}, n={}, k={}, k_group_size={}\n", m, n, k, k_group_size);

  using Kernel = amx::GemmKernel224Int4KGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;

  // Allocate buffers
  void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
  void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));
  void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  // Create identity-like matrices for easy verification
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  // Initialize A as mostly zeros with a few ones
  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(0.0f);
  }
  // Set A[0,0] = 1
  input_a[0] = ggml_compute_fp32_to_bf16(1.0f);

  // Initialize B as mostly zeros with a few ones
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(0.0f);
  }
  // Set B[0,0] = 1
  input_b[0] = ggml_compute_fp32_to_bf16(1.0f);

  // Expected result: C[0,0] = 1*1 = 1, rest = 0
  std::cout << "\nExpected result: C[0,0] = 1.0, rest = 0.0\n" << std::endl;

  // Quantize inputs
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  // Print scales for debugging
  std::cout << "BufferA scales for row 0:" << std::endl;
  for (int kg = 0; kg < k / k_group_size; kg++) {
    float scale = *ba->get_scale(m, 0, k, kg * k_group_size);
    std::cout << fmt::format("  k_group[{}]: scale = {:.6f}\n", kg, scale);
  }

  std::cout << "\nBufferB scales for col 0:" << std::endl;
  for (int kg = 0; kg < k / k_group_size; kg++) {
    float scale = *bb->get_scale(n, 0, k, kg * k_group_size);
    std::cout << fmt::format("  k_group[{}]: scale = {:.6f}\n", kg, scale);
  }

  // Configure AMX
  Kernel::config();

  // Run matrix multiplication
  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  // Get output
  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Print results
  std::cout << "\nActual result (first 5x5):" << std::endl;
  for (int i = 0; i < std::min(5, m); i++) {
    for (int j = 0; j < std::min(5, n); j++) {
      float val = ggml_compute_bf16_to_fp32(output[i * n + j]);
      std::cout << fmt::format("{:8.4f} ", val);
    }
    std::cout << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

void debug_pattern_multiplication() {
  std::cout << "\n=== Debug Pattern Multiplication ===" << std::endl;

  const int m = 32;
  const int n = 32;
  const int k = 512;  // Must be at least K_BLOCK (512)
  const int k_group_size = 128;

  using Kernel = amx::GemmKernel224Int4KGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;

  void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
  void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));
  void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  // Create constant matrices
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  // Fill A with 0.1 and B with 0.1
  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(0.1f);
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(0.1f);
  }

  // Expected: Each element should be 0.1 * 0.1 * k = 0.01 * 512 = 5.12
  float expected = 0.1f * 0.1f * k;
  std::cout << fmt::format("\nExpected result: all elements = {:.4f}\n", expected);

  // Quantize
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  // Run
  Kernel::config();
  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  // Get output
  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Check results
  float max_error = 0.0f;
  float avg_error = 0.0f;
  for (int i = 0; i < m * n; i++) {
    float actual = ggml_compute_bf16_to_fp32(output[i]);
    float error = std::abs(actual - expected);
    max_error = std::max(max_error, error);
    avg_error += error;
  }
  avg_error /= (m * n);

  std::cout << fmt::format("Max error: {:.6f}\n", max_error);
  std::cout << fmt::format("Avg error: {:.6f}\n", avg_error);
  std::cout << fmt::format("Relative error: {:.2f}%\n", (max_error / expected) * 100);

  // Print sample values
  std::cout << "\nSample values (first 5x5):" << std::endl;
  for (int i = 0; i < std::min(5, m); i++) {
    for (int j = 0; j < std::min(5, n); j++) {
      float val = ggml_compute_bf16_to_fp32(output[i * n + j]);
      std::cout << fmt::format("{:8.4f} ", val);
    }
    std::cout << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

void compare_with_regular_int4() {
  std::cout << "\n=== Compare K-Group vs Regular INT4 ===" << std::endl;

  const int m = 32;
  const int n = 32;
  const int k = 512;
  const int k_group_size = 128;

  // Create test data
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Test with regular INT4
  {
    using Kernel = amx::GemmKernel224Int4;
    using BufferA = Kernel::BufferA;
    using BufferB = Kernel::BufferB;
    using BufferC = Kernel::BufferC;

    void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k));
    void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k));  // Fixed: n, k not k, n
    void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

    auto ba = std::make_shared<BufferA>(m, k, buffer_a);
    auto bb = std::make_shared<BufferB>(n, k, buffer_b);  // Fixed: n, k not k, n
    auto bc = std::make_shared<BufferC>(m, n, buffer_c);

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    Kernel::config();
    amx::mat_mul(m, n, k, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output_regular(m * n);
    bc->to_mat(m, output_regular.data(), 0, 1);

    std::cout << "Regular INT4 results (first 3x3):" << std::endl;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        float val = ggml_compute_bf16_to_fp32(output_regular[i * n + j]);
        std::cout << fmt::format("{:8.4f} ", val);
      }
      std::cout << std::endl;
    }

    free(buffer_a);
    free(buffer_b);
    free(buffer_c);
  }

  // Test with K-Group INT4
  {
    using Kernel = amx::GemmKernel224Int4KGroup;
    using BufferA = Kernel::BufferA;
    using BufferB = Kernel::BufferB;
    using BufferC = Kernel::BufferC;

    void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
    void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));
    void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

    auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
    auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);
    auto bc = std::make_shared<BufferC>(m, n, buffer_c);

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    Kernel::config();
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output_kgroup(m * n);
    bc->to_mat(m, output_kgroup.data(), 0, 1);

    std::cout << "\nK-Group INT4 results (first 3x3):" << std::endl;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        float val = ggml_compute_bf16_to_fp32(output_kgroup[i * n + j]);
        std::cout << fmt::format("{:8.4f} ", val);
      }
      std::cout << std::endl;
    }

    free(buffer_a);
    free(buffer_b);
    free(buffer_c);
  }
}

int main() {
  std::cout << "Starting K-Group Debugging\n" << std::endl;

  debug_simple_multiplication();
  debug_pattern_multiplication();
  compare_with_regular_int4();

  return 0;
}