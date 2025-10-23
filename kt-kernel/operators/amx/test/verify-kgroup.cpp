#include <cmath>
#include <iostream>
#include <memory>
#include <random>

#include "../la/amx.hpp"

void verify_kgroup_accuracy() {
  std::cout << "=== Verifying K-Group Accuracy ===" << std::endl;

  const int m = 32;
  const int n = 32;
  const int k = 1024;
  const int k_group_size = 256;

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

  // Create input matrices with values in the quantization sweet spot
  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  std::mt19937 gen(12345);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Compute reference result with float32
  std::vector<float> ref_result(m * n, 0.0f);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        float a_val = ggml_compute_bf16_to_fp32(input_a[i * k + l]);
        float b_val = ggml_compute_bf16_to_fp32(input_b[l * n + j]);
        sum += a_val * b_val;
      }
      ref_result[i * n + j] = sum;
    }
  }

  // Quantize and compute with k-group
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  Kernel::config();
  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Compute errors
  float max_abs_error = 0.0f;
  float total_abs_error = 0.0f;
  float max_ref_value = 0.0f;

  for (int i = 0; i < m * n; i++) {
    float actual = ggml_compute_bf16_to_fp32(output[i]);
    float ref = ref_result[i];
    float error = std::abs(actual - ref);

    max_abs_error = std::max(max_abs_error, error);
    total_abs_error += error;
    max_ref_value = std::max(max_ref_value, std::abs(ref));
  }

  float avg_abs_error = total_abs_error / (m * n);
  float relative_error = max_abs_error / (max_ref_value + 1e-8f);

  std::cout << "Matrix dimensions: " << m << "x" << n << "x" << k << std::endl;
  std::cout << "K-group size: " << k_group_size << std::endl;
  std::cout << "Max absolute error: " << max_abs_error << std::endl;
  std::cout << "Average absolute error: " << avg_abs_error << std::endl;
  std::cout << "Max reference value: " << max_ref_value << std::endl;
  std::cout << "Relative error: " << (relative_error * 100) << "%" << std::endl;

  // Check if accuracy is acceptable for INT4
  // INT4 quantization typically has 5-10% error
  if (relative_error < 0.15f) {
    std::cout << "✓ Accuracy is acceptable for INT4 quantization" << std::endl;
  } else {
    std::cout << "✗ Accuracy needs improvement" << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  verify_kgroup_accuracy();
  return 0;
}