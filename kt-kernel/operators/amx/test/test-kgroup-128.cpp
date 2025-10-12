#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../la/amx.hpp"

void test_kgroup_128() {
  std::cout << "=== Testing K-Group with k_group_size = 128 ===\n" << std::endl;

  const int m = 32;  // Simple case
  const int n = 32;
  const int k = 512;  // Multiple of 128
  const int k_group_size = 128;

  std::cout << "Matrix dimensions: " << m << " x " << n << " x " << k << std::endl;
  std::cout << "K-group size: " << k_group_size << std::endl;
  std::cout << "Number of k-groups: " << k / k_group_size << std::endl;

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

  Kernel::config();

  // Test 1: All ones
  std::cout << "\n--- Test 1: All ones (expected = " << k << ") ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(1.0f);
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(1.0f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float actual = ggml_compute_bf16_to_fp32(output[0]);
    float error = std::abs(actual - k) / k * 100;
    std::cout << "Result[0,0]: " << actual << " (error: " << error << "%)" << std::endl;
  }

  // Test 2: Values in quantization sweet spot (0.5)
  std::cout << "\n--- Test 2: All 0.5 (expected = " << 0.5f * 0.5f * k << ") ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(0.5f);
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.5f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float expected = 0.5f * 0.5f * k;
    float actual = ggml_compute_bf16_to_fp32(output[0]);
    float error = std::abs(actual - expected) / expected * 100;
    std::cout << "Result[0,0]: " << actual << " (expected: " << expected << ", error: " << error << "%)" << std::endl;
  }

  // Test 3: Different values per k-group
  std::cout << "\n--- Test 3: Different values per k-group ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Each k-group has different value
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        int kg = j / k_group_size;
        float val = (kg + 1) * 0.25f;  // 0.25, 0.5, 0.75, 1.0
        input_a[i * k + j] = ggml_compute_fp32_to_bf16(val);
      }
    }

    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.5f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Expected: sum of (kg+1)*0.25 * 0.5 * k_group_size for all k-groups
    float expected = 0.0f;
    for (int kg = 0; kg < k / k_group_size; kg++) {
      expected += (kg + 1) * 0.25f * 0.5f * k_group_size;
    }

    float actual = ggml_compute_bf16_to_fp32(output[0]);
    float error = std::abs(actual - expected) / expected * 100;
    std::cout << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cout << "Error: " << error << "%" << std::endl;
  }

  // Test 4: Pattern test
  std::cout << "\n--- Test 4: Pattern with alternating values ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Alternating pattern in A
    for (int i = 0; i < m * k; i++) {
      float val = (i % 2 == 0) ? 0.25f : 0.75f;
      input_a[i] = ggml_compute_fp32_to_bf16(val);
    }

    // Constant in B
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.4f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Expected: average of 0.25 and 0.75 is 0.5, so 0.5 * 0.4 * k
    float expected = 0.5f * 0.4f * k;
    float actual = ggml_compute_bf16_to_fp32(output[0]);
    float error = std::abs(actual - expected) / expected * 100;
    std::cout << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cout << "Error: " << error << "%" << std::endl;
  }

  // Test 5: Check all output elements
  std::cout << "\n--- Test 5: Verify all output elements (0.1 × 0.1) ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(0.1f);
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.1f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float expected = 0.1f * 0.1f * k;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;

    for (int i = 0; i < m * n; i++) {
      float actual = ggml_compute_bf16_to_fp32(output[i]);
      float error = std::abs(actual - expected) / expected * 100;
      max_error = std::max(max_error, error);
      avg_error += error;
      if (error > 5.0f) error_count++;
    }
    avg_error /= (m * n);

    std::cout << "Expected value: " << expected << std::endl;
    std::cout << "Max error: " << max_error << "%" << std::endl;
    std::cout << "Average error: " << avg_error << "%" << std::endl;
    std::cout << "Elements with >5% error: " << error_count << "/" << m * n << std::endl;
  }

  // Test 6: Random normal distribution (like real model weights)
  std::cout << "\n--- Test 6: Random normal distribution ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
    }

    // Compute reference with float32
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

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Compute errors
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float avg_rel_error = 0.0f;
    int large_error_count = 0;

    for (int i = 0; i < m * n; i++) {
      float actual = ggml_compute_bf16_to_fp32(output[i]);
      float ref = ref_result[i];
      float abs_error = std::abs(actual - ref);
      float rel_error = std::abs(ref) > 1e-6 ? abs_error / std::abs(ref) : 0.0f;

      max_abs_error = std::max(max_abs_error, abs_error);
      max_rel_error = std::max(max_rel_error, rel_error);
      avg_rel_error += rel_error;

      if (rel_error > 0.2f) {  // 20% error
        large_error_count++;
        if (large_error_count <= 5) {
          std::cout << "  [" << i / n << "," << i % n << "]: actual=" << actual << ", ref=" << ref
                    << ", rel_error=" << (rel_error * 100) << "%" << std::endl;
        }
      }
    }
    avg_rel_error /= (m * n);

    std::cout << "Max absolute error: " << max_abs_error << std::endl;
    std::cout << "Max relative error: " << (max_rel_error * 100) << "%" << std::endl;
    std::cout << "Average relative error: " << (avg_rel_error * 100) << "%" << std::endl;
    std::cout << "Elements with >20% error: " << large_error_count << "/" << m * n << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  test_kgroup_128();
  return 0;
}