#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "../la/amx.hpp"

void debug_kgroup_details() {
  std::cout << "=== Debugging K-Group Details ===\n" << std::endl;

  const int m = 32;  // Minimum size for AMX
  const int n = 32;
  const int k = 512;  // 4 k-groups, must be >= K_BLOCK
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

  // Test with specific values to debug quantization
  std::cout << "Test: Specific values with normal distribution\n" << std::endl;

  std::vector<ggml_bf16_t> input_a(m * k);
  std::vector<ggml_bf16_t> input_b(k * n);

  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 0.1f);

  // Fill with random normal values and print some
  std::cout << "Sample A values (first 8):" << std::endl;
  for (int i = 0; i < 8; i++) {
    float val = dist(gen);
    input_a[i] = ggml_compute_fp32_to_bf16(val);
    std::cout << "  A[" << i << "] = " << val << std::endl;
  }

  // Fill rest of A
  for (int i = 8; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  std::cout << "\nSample B values (first 8):" << std::endl;
  for (int i = 0; i < 8; i++) {
    float val = dist(gen);
    input_b[i] = ggml_compute_fp32_to_bf16(val);
    std::cout << "  B[" << i << "] = " << val << std::endl;
  }

  // Fill rest of B
  for (int i = 8; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Quantize
  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  // Print scales for debugging
  std::cout << "\nA scales (per k-group):" << std::endl;
  for (int row = 0; row < m; row++) {
    std::cout << "  Row " << row << ": ";
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *ba->get_scale(m, row, k, kg * k_group_size);
      std::cout << "kg" << kg << "=" << scale << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "\nB scales (per k-group):" << std::endl;
  for (int col = 0; col < n; col++) {
    std::cout << "  Col " << col << ": ";
    for (int kg = 0; kg < k / k_group_size; kg++) {
      float scale = *bb->get_scale(n, col, k, kg * k_group_size);
      std::cout << "kg" << kg << "=" << scale << " ";
    }
    std::cout << std::endl;
  }

  // Test dequantization to check if quantization is working
  std::cout << "\nDequantization test (first row of A):" << std::endl;
  // We need to manually dequantize to check
  // Get quantized values and scale
  int8_t* a_data = (int8_t*)ba->get_submat(m, k, 0, 0);
  float scale0 = *ba->get_scale(m, 0, k, 0);

  std::cout << "  First 8 quantized values: ";
  for (int i = 0; i < 8; i++) {
    std::cout << (int)a_data[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "  Dequantized (q * scale): ";
  for (int i = 0; i < 8; i++) {
    float dequant = a_data[i] * scale0;
    float original = ggml_compute_bf16_to_fp32(input_a[i]);
    std::cout << dequant << " (orig=" << original << ") ";
  }
  std::cout << std::endl;

  // Compute reference
  std::cout << "\nComputing reference result..." << std::endl;
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

  // Run k-group multiplication
  Kernel::config();
  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

  std::vector<ggml_bf16_t> output(m * n);
  bc->to_mat(m, output.data(), 0, 1);

  // Compare results
  std::cout << "\nResults comparison:" << std::endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int idx = i * n + j;
      float actual = ggml_compute_bf16_to_fp32(output[idx]);
      float ref = ref_result[idx];
      float error = std::abs(actual - ref) / (std::abs(ref) + 1e-8) * 100;
      std::cout << "  [" << i << "," << j << "]: actual=" << actual << ", ref=" << ref << ", error=" << error << "%"
                << std::endl;
    }
  }

  // Test a simple case to verify the mechanism
  std::cout << "\n--- Simple test with k_group boundaries ---" << std::endl;

  // Clear buffers
  for (int i = 0; i < m * k; i++) {
    input_a[i] = ggml_compute_fp32_to_bf16(0.0f);
  }
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(0.0f);
  }

  // Set specific values for each k-group
  for (int i = 0; i < m; i++) {
    // First k-group (0-127): value = 0.5
    for (int j = 0; j < 128; j++) {
      input_a[i * k + j] = ggml_compute_fp32_to_bf16(0.5f);
    }
    // Second k-group (128-255): value = 0.25
    for (int j = 128; j < 256; j++) {
      input_a[i * k + j] = ggml_compute_fp32_to_bf16(0.25f);
    }
    // Remaining k-groups: value = 0.1
    for (int j = 256; j < k; j++) {
      input_a[i * k + j] = ggml_compute_fp32_to_bf16(0.1f);
    }
  }

  // B matrix: all 0.4
  for (int i = 0; i < k * n; i++) {
    input_b[i] = ggml_compute_fp32_to_bf16(0.4f);
  }

  ba->from_mat(m, input_a.data(), 0, 1);
  bb->from_mat(input_b.data(), 0, 1);

  // Expected: 0.5 * 0.4 * 128 + 0.25 * 0.4 * 128 + 0.1 * 0.4 * 256 = 25.6 + 12.8 + 10.24 = 48.64
  float expected = 0.5f * 0.4f * 128 + 0.25f * 0.4f * 128 + 0.1f * 0.4f * 256;
  std::cout << "Expected value: " << expected << std::endl;

  amx::mat_mul_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);
  bc->to_mat(m, output.data(), 0, 1);

  float actual = ggml_compute_bf16_to_fp32(output[0]);
  std::cout << "Actual value: " << actual << std::endl;
  std::cout << "Error: " << std::abs(actual - expected) / expected * 100 << "%" << std::endl;

  free(buffer_a);
  free(buffer_b);
  free(buffer_c);
}

int main() {
  debug_kgroup_details();
  return 0;
}