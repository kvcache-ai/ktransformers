#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../la/amx.hpp"

void analyze_error_patterns() {
  std::cout << "=== Analyzing Error Patterns in K-Group Quantization ===" << std::endl;

  const int m = 32;
  const int n = 32;
  const int k = 512;
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

  Kernel::config();

  std::cout << "\n1. Testing with very small values (prone to quantization loss):" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Very small values - will mostly quantize to 0
    for (int i = 0; i < m * k; i++) {
      input_a[i] = ggml_compute_fp32_to_bf16(0.0001f * (i % 10));
    }
    for (int i = 0; i < k * n; i++) {
      input_b[i] = ggml_compute_fp32_to_bf16(0.0001f * (i % 10));
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    // Check scales
    float a_scale = *ba->get_scale(m, 0, k, 0);
    float b_scale = *bb->get_scale(n, 0, k, 0);
    std::cout << "  A scale: " << a_scale << ", B scale: " << b_scale << std::endl;

    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    float first_val = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "  Result[0,0]: " << first_val << std::endl;
  }

  std::cout << "\n2. Testing with values near quantization boundaries:" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Values at quantization boundaries (multiples of 1/127 for int8)
    for (int i = 0; i < m * k; i++) {
      float val = (i % 16) / 127.0f;  // INT4 has 16 levels
      input_a[i] = ggml_compute_fp32_to_bf16(val);
    }
    for (int i = 0; i < k * n; i++) {
      float val = (i % 16) / 127.0f;
      input_b[i] = ggml_compute_fp32_to_bf16(val);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    std::cout << "  First row results: ";
    for (int j = 0; j < 5; j++) {
      float val = ggml_compute_bf16_to_fp32(output[j]);
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\n3. Testing with different scale ranges per k-group:" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Different magnitude for each k-group
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        int kg = j / k_group_size;
        float scale = std::pow(10.0f, -kg);  // 1.0, 0.1, 0.01, 0.001
        input_a[i * k + j] = ggml_compute_fp32_to_bf16(scale * 0.5f);
      }
    }

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        int kg = i / k_group_size;
        float scale = std::pow(10.0f, -kg);
        input_b[i * n + j] = ggml_compute_fp32_to_bf16(scale * 0.5f);
      }
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    // Print scales for each k-group
    std::cout << "  A scales per k-group: ";
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba->get_scale(m, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    std::cout << "  B scales per k-group: ";
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *bb->get_scale(n, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Compute reference
    float ref = 0.0f;
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = std::pow(10.0f, -kg);
      ref += k_group_size * scale * scale * 0.25f;  // 0.5 * 0.5
    }

    float actual = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "  Expected: " << ref << ", Actual: " << actual << std::endl;
    std::cout << "  Error: " << std::abs(ref - actual) / ref * 100 << "%" << std::endl;
  }

  std::cout << "\n4. Testing with sparse patterns (many zeros):" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Sparse pattern - 90% zeros
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < m * k; i++) {
      float val = (dist(gen) < 0.1f) ? 0.5f : 0.0f;
      input_a[i] = ggml_compute_fp32_to_bf16(val);
    }
    for (int i = 0; i < k * n; i++) {
      float val = (dist(gen) < 0.1f) ? 0.5f : 0.0f;
      input_b[i] = ggml_compute_fp32_to_bf16(val);
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);
    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Compute statistics
    float max_val = 0.0f;
    float avg_val = 0.0f;
    int non_zero = 0;

    for (int i = 0; i < m * n; i++) {
      float val = std::abs(ggml_compute_bf16_to_fp32(output[i]));
      max_val = std::max(max_val, val);
      avg_val += val;
      if (val > 1e-6) non_zero++;
    }
    avg_val /= (m * n);

    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Avg value: " << avg_val << std::endl;
    std::cout << "  Non-zero outputs: " << non_zero << "/" << m * n << std::endl;
  }

  std::cout << "\n5. Testing with gradual value changes (worst case for k-group):" << std::endl;
  {
    std::vector<ggml_bf16_t> input_a(m * k);
    std::vector<ggml_bf16_t> input_b(k * n);

    // Gradual increase across k dimension - worst case for k-group quantization
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        float val = j * 0.001f;  // Gradual increase
        input_a[i * k + j] = ggml_compute_fp32_to_bf16(val);
      }
    }

    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        float val = 0.1f;  // Constant
        input_b[i * n + j] = ggml_compute_fp32_to_bf16(val);
      }
    }

    ba->from_mat(m, input_a.data(), 0, 1);
    bb->from_mat(input_b.data(), 0, 1);

    // Check how scales vary
    std::cout << "  A scales (should increase): ";
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba->get_scale(m, 0, k, kg * k_group_size);
      std::cout << scale << " ";
    }
    std::cout << std::endl;

    amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

    std::vector<ggml_bf16_t> output(m * n);
    bc->to_mat(m, output.data(), 0, 1);

    // Reference calculation
    float ref = 0.0f;
    for (int j = 0; j < k; j++) {
      ref += j * 0.001f * 0.1f;
    }

    float actual = ggml_compute_bf16_to_fp32(output[0]);
    std::cout << "  Expected: " << ref << ", Actual: " << actual << std::endl;
    std::cout << "  Error: " << std::abs(ref - actual) / ref * 100 << "%" << std::endl;
  }

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  analyze_error_patterns();
  return 0;
}