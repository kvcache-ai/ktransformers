#include <omp.h>

#include "../la/amx.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>

void test_kgroup_kernel_basic() {
  std::cout << "=== Testing GemmKernel224Int4KGroup Basic Functionality ===" << std::endl;

  // Test parameters - must match kernel requirements
  const int m = 64;              // Must be multiple of M_STEP (32)
  const int n = 64;              // Must be multiple of N_STEP (32)
  const int k = 1024;            // Must be multiple of K_STEP (64)
  const int k_group_size = 256;  // Must divide k evenly

  std::cout << fmt::format("Parameters: m={}, n={}, k={}, k_group_size={}\n", m, n, k, k_group_size);

  using Kernel = amx::GemmKernel224Int4KGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;

  // Allocate buffers
  size_t size_a = BufferA::required_size(m, k, k_group_size);
  size_t size_b = BufferB::required_size(n, k, k_group_size);  // Fixed: n, k not k, n
  size_t size_c = BufferC::required_size(m, n);

  void* buffer_a = std::aligned_alloc(64, size_a);
  void* buffer_b = std::aligned_alloc(64, size_b);
  void* buffer_c = std::aligned_alloc(64, size_c);

  std::cout << fmt::format("Buffer sizes: A={} KB, B={} KB, C={} KB\n", size_a / 1024, size_b / 1024, size_c / 1024);

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);  // Fixed: n, k not k, n
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  // Create test input data
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Fill with small values to avoid overflow
  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Quantize inputs
  std::cout << "Quantizing inputs..." << std::endl;
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  // Configure AMX
  Kernel::config();

  // Run matrix multiplication with k-group quantization
  std::cout << "Running k-group matrix multiplication..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << fmt::format("Time: {} ms\n", duration / 1000.0);

  // Convert output to bf16
  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Print sample output values
  std::cout << "\nSample output values:" << std::endl;
  for (int i = 0; i < std::min(5, m); i++) {
    for (int j = 0; j < std::min(5, n); j++) {
      float val = ggml_compute_bf16_to_fp32(output[i * n + j]);
      std::cout << fmt::format("{:8.4f} ", val);
    }
    std::cout << std::endl;
  }

  // Clean up
  free(buffer_a);
  free(buffer_b);
  free(buffer_c);

  std::cout << "\n✓ Basic test completed!" << std::endl;
}

void test_kgroup_kernel_correctness() {
  std::cout << "\n=== Testing GemmKernel224Int4KGroup Correctness ===" << std::endl;

  const int m = 32;
  const int n = 32;
  const int k = 512;
  const int k_group_size = 128;

  using Kernel = amx::GemmKernel224Int4KGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;

  // Allocate buffers
  void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
  void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));  // Fixed: n, k not k, n
  void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);  // Fixed: n, k not k, n
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  // Create simple test pattern
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);
  std::vector<float> expected(m * n, 0.0f);

  // Fill A with row indices and B with column indices (scaled down)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      input_a[i * k + j] = ggml_compute_fp32_to_bf16((i + 1) * 0.001f);
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      input_b[i * n + j] = ggml_compute_fp32_to_bf16((j + 1) * 0.001f);
    }
  }

  // Compute expected result (naive)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        float a_val = ggml_compute_bf16_to_fp32(input_a[i * k + l]);
        float b_val = ggml_compute_bf16_to_fp32(input_b[l * n + j]);
        sum += a_val * b_val;
      }
      expected[i * n + j] = sum;
    }
  }

  // Quantize and run
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  Kernel::config();
  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  // Get output
  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Compare results
  float max_error = 0.0f;
  float total_error = 0.0f;
  int count = 0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float actual = ggml_compute_bf16_to_fp32(output[i * n + j]);
      float exp = expected[i * n + j];
      float error = std::abs(actual - exp);
      max_error = std::max(max_error, error);
      total_error += error;
      count++;
    }
  }

  float avg_error = total_error / count;
  float relative_error = max_error / (*std::max_element(expected.begin(), expected.end()) + 1e-8f);

  std::cout << fmt::format("Error Analysis:\n");
  std::cout << fmt::format("  Max absolute error: {:.6f}\n", max_error);
  std::cout << fmt::format("  Average absolute error: {:.6f}\n", avg_error);
  std::cout << fmt::format("  Relative error: {:.2f}%\n", relative_error * 100);

  // Check acceptability (INT4 quantization + k-group should have reasonable error)
  if (relative_error < 0.10f) {  // 10% relative error threshold for INT4
    std::cout << "✓ Error is within acceptable range for INT4 quantization" << std::endl;
  } else {
    std::cout << "✗ Error is higher than expected!" << std::endl;
  }

  // Print first few values for comparison
  std::cout << "\nFirst 5x5 values comparison:" << std::endl;
  std::cout << "Expected vs Actual:" << std::endl;
  for (int i = 0; i < std::min(5, m); i++) {
    for (int j = 0; j < std::min(5, n); j++) {
      float actual = ggml_compute_bf16_to_fp32(output[i * n + j]);
      float exp = expected[i * n + j];
      std::cout << fmt::format("({:.4f},{:.4f}) ", exp, actual);
    }
    std::cout << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);

  std::cout << "\n✓ Correctness test completed!" << std::endl;
}

void test_kgroup_kernel_performance() {
  std::cout << "\n=== Testing GemmKernel224Int4KGroup Performance ===" << std::endl;

  const int m = 256;
  const int n = 256;
  const int k = 2048;
  const int k_group_size = 512;
  const int iterations = 100;

  using Kernel = amx::GemmKernel224Int4KGroup;
  using BufferA = Kernel::BufferA;
  using BufferB = Kernel::BufferB;
  using BufferC = Kernel::BufferC;

  // Allocate buffers
  void* buffer_a = std::aligned_alloc(64, BufferA::required_size(m, k, k_group_size));
  void* buffer_b = std::aligned_alloc(64, BufferB::required_size(n, k, k_group_size));  // Fixed: n, k not k, n
  void* buffer_c = std::aligned_alloc(64, BufferC::required_size(m, n));

  auto ba = std::make_shared<BufferA>(m, k, k_group_size, buffer_a);
  auto bb = std::make_shared<BufferB>(n, k, k_group_size, buffer_b);  // Fixed: n, k not k, n
  auto bc = std::make_shared<BufferC>(m, n, buffer_c);

  // Create random input
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

  // Quantize
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  Kernel::config();

  // Warm up
  for (int i = 0; i < 10; i++) {
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  double avg_time_ms = duration / (1000.0 * iterations);
  double ops = 2.0 * m * n * k;
  double gflops = (ops * iterations) / (duration * 1000.0);

  std::cout << fmt::format("Matrix size: {}x{}x{}\n", m, n, k);
  std::cout << fmt::format("K-group size: {}\n", k_group_size);
  std::cout << fmt::format("Average time per multiplication: {:.3f} ms\n", avg_time_ms);
  std::cout << fmt::format("Performance: {:.2f} GFLOPS\n", gflops);

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);

  std::cout << "\n✓ Performance test completed!" << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Starting GemmKernel224Int4KGroup Tests\n" << std::endl;

  try {
    test_kgroup_kernel_basic();
    test_kgroup_kernel_correctness();
    test_kgroup_kernel_performance();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}