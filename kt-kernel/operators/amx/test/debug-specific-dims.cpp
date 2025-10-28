#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../la/amx.hpp"

void debug_specific_dimensions() {
  std::cout << "=== Debugging Specific Dimensions Issue ===\n" << std::endl;

  const int m_original = 200;
  const int n = 2048;
  const int k = 7168;
  const int k_group_size = 128;

  const int M_STEP = 32;
  const int m = ((m_original + M_STEP - 1) / M_STEP) * M_STEP;  // Round up to 224

  std::cout << "Original dimensions: " << m_original << " x " << n << " x " << k << std::endl;
  std::cout << "Padded dimensions: " << m << " x " << n << " x " << k << std::endl;
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

  // Test 1: Simple pattern - all ones
  std::cout << "\n--- Test 1: All ones (should give k = 7168) ---" << std::endl;
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

    // Check some scales
    std::cout << "A scales (first 3 k-groups): ";
    for (int kg = 0; kg < 3; kg++) {
      float scale = *ba->get_scale(m, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    std::cout << "B scales (first 3 k-groups): ";
    for (int kg = 0; kg < 3; kg++) {
      float scale = *bb->get_scale(n, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    Kernel::config();
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float expected = 7168.0f;
    float actual = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cout << "Error: " << std::abs(actual - expected) / expected * 100 << "%" << std::endl;
  }

  // Test 2: Small values
  std::cout << "\n--- Test 2: Small values (0.01) ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(0.01f);
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.01f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    Kernel::config();
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float expected = 0.01f * 0.01f * 7168.0f;  // 0.7168
    float actual = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cout << "Error: " << std::abs(actual - expected) / expected * 100 << "%" << std::endl;
  }

  // Test 3: Identity-like pattern
  std::cout << "\n--- Test 3: Identity pattern ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Initialize to zeros
    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(0.0f);
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.0f);
    }

    // Set diagonal to 1
    int min_dim = std::min(std::min(m, n), k);
    for (int i = 0; i < min_dim; i++) {
      input_a[i * k + i] = ggml_compute_fp32_to_bf16(1.0f);
      input_b[i * n + i] = ggml_compute_fp32_to_bf16(1.0f);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    Kernel::config();
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Check diagonal elements
    std::cout << "Diagonal elements (should be 1): ";
    for (int i = 0; i < std::min(5, min_dim); i++) {
      float val = ggml_compute_bf16_to_fp32(output[i * n + i]);
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  // Test 4: Pattern with different values per k-group
  std::cout << "\n--- Test 4: Different values per k-group ---" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Each k-group has different value
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        int kg = j / k_group_size;
        float val = (kg + 1) * 0.1f;  // 0.1, 0.2, 0.3, ...
        input_a[i * k + j] = ggml_compute_fp32_to_bf16(val);
      }
    }

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        input_b[i * n + j] = ggml_compute_fp32_to_bf16(0.1f);
      }
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    // Check scales for different k-groups
    std::cout << "A scales (first 5 k-groups): ";
    for (int kg = 0; kg < std::min(5, k / k_group_size); kg++) {
      float scale = *ba->get_scale(m, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    Kernel::config();
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Expected: sum of (kg+1)*0.1 * 0.1 * k_group_size for all k-groups
    float expected = 0.0f;
    for (int kg = 0; kg < k / k_group_size; kg++) {
      expected += (kg + 1) * 0.1f * 0.1f * k_group_size;
    }

    float actual = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "Expected: " << expected << ", Actual: " << actual << std::endl;
    std::cout << "Error: " << std::abs(actual - expected) / expected * 100 << "%" << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  debug_specific_dimensions();
  return 0;
}