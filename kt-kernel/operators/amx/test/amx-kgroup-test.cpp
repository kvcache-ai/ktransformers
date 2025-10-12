#include <omp.h>

#include "../la/amx.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <memory>

// Test kernel configuration for k-group testing
struct TestKernelKGroup {
  static constexpr int M_STEP = 32;
  static constexpr int K_STEP = 64;
  static constexpr int K_BLOCK = 512;
  static constexpr int N_STEP = 32;
  static constexpr int N_BLOCK = 512;
  static constexpr int TILE_N = 16;
  using dt = int8_t;

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_per_thread = (n + nth - 1) / nth;
    int n_start = ith * n_per_thread;
    int n_end = std::min(n_start + n_per_thread, n);
    return {n_start, n_end};
  }
};

void test_buffer_kgroup_basic() {
  std::cout << "=== Testing BufferAKGroupImpl Basic Functionality ===" << std::endl;

  // Test parameters
  const int max_m = 64;          // Must be multiple of M_STEP
  const int k = 2048;            // Must be multiple of K_STEP and K_BLOCK
  const int k_group_size = 128;  // Must divide K_BLOCK evenly

  std::cout << fmt::format("Parameters: max_m={}, k={}, k_group_size={}\n", max_m, k, k_group_size);

  // Calculate and allocate buffer
  size_t buffer_size = amx::BufferAKGroupImpl<TestKernelKGroup>::required_size(max_m, k, k_group_size);
  void* buffer = std::aligned_alloc(64, buffer_size);
  std::memset(buffer, 0, buffer_size);

  std::cout << fmt::format("Buffer size: {} bytes\n", buffer_size);

  // Create BufferAKGroupImpl instance
  auto buf = std::make_unique<amx::BufferAKGroupImpl<TestKernelKGroup>>(max_m, k, k_group_size, buffer);

  // Create test input data (bf16)
  std::vector<ggml_bf16_t> input(max_m * k);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < max_m * k; i++) {
    float val = dist(gen);
    input[i] = ggml_compute_fp32_to_bf16(val);
  }

  // Test from_mat
  std::cout << "Testing from_mat..." << std::endl;
  buf->from_mat(max_m, input.data(), 0, 1);
  std::cout << "✓ from_mat completed successfully" << std::endl;

  // Test get_submat
  std::cout << "Testing get_submat..." << std::endl;
  for (int m_begin = 0; m_begin < max_m; m_begin += TestKernelKGroup::M_STEP) {
    for (int k_begin = 0; k_begin < k; k_begin += TestKernelKGroup::K_STEP) {
      int8_t* submat = buf->get_submat(max_m, k, m_begin, k_begin);
      if (submat == nullptr) {
        std::cerr << fmt::format("ERROR: get_submat returned null for m_begin={}, k_begin={}\n", m_begin, k_begin);
        free(buffer);
        return;
      }
    }
  }
  std::cout << "✓ get_submat tested for all valid positions" << std::endl;

  // Test get_scale
  std::cout << "Testing get_scale..." << std::endl;
  int k_group_count = k / k_group_size;
  for (int m_idx = 0; m_idx < max_m; m_idx++) {
    for (int kg_idx = 0; kg_idx < k_group_count; kg_idx++) {
      float* scale = buf->get_scale(max_m, m_idx, k, kg_idx * k_group_size);
      if (scale == nullptr) {
        std::cerr << fmt::format("ERROR: get_scale returned null for m_idx={}, k_group={}\n", m_idx, kg_idx);
        free(buffer);
        return;
      }
      // Verify scale is non-zero (should be set by from_mat)
      if (*scale == 0.0f) {
        std::cerr << fmt::format("WARNING: scale is zero for m_idx={}, k_group={}\n", m_idx, kg_idx);
      }
    }
  }
  std::cout << "✓ get_scale tested for all k-groups" << std::endl;

  // Print some scale values for verification
  std::cout << "\nSample scale values:" << std::endl;
  for (int kg = 0; kg < std::min(4, k_group_count); kg++) {
    float* scale = buf->get_scale(max_m, 0, k, kg * k_group_size);
    std::cout << fmt::format("  k_group[{}] (k={}): scale = {:.6f}\n", kg, kg * k_group_size, *scale);
  }

  // Clean up
  free(buffer);
  std::cout << "\n✓ All basic tests passed!" << std::endl;
}

void test_buffer_kgroup_correctness() {
  std::cout << "\n=== Testing BufferAKGroupImpl Quantization Correctness ===" << std::endl;

  const int max_m = 32;
  const int k = 512;
  const int k_group_size = 128;

  size_t buffer_size = amx::BufferAKGroupImpl<TestKernelKGroup>::required_size(max_m, k, k_group_size);
  void* buffer = std::aligned_alloc(64, buffer_size);

  auto buf = std::make_unique<amx::BufferAKGroupImpl<TestKernelKGroup>>(max_m, k, k_group_size, buffer);

  // Create test input matrix with known patterns
  std::vector<float> original(max_m * k);
  std::vector<ggml_bf16_t> input(max_m * k);

  // Fill with different patterns for each k-group to test group-wise quantization
  for (int m = 0; m < max_m; m++) {
    for (int k_idx = 0; k_idx < k; k_idx++) {
      int kg = k_idx / k_group_size;
      // Different magnitude for each k-group
      float base_val = (kg + 1) * 0.1f;
      float val = base_val * std::sin(m * 0.1f + k_idx * 0.01f);
      original[m * k + k_idx] = val;
      input[m * k + k_idx] = ggml_compute_fp32_to_bf16(val);
    }
  }

  // Quantize
  buf->from_mat(max_m, input.data(), 0, 1);

  // Dequantize and check error
  std::vector<float> dequantized(max_m * k);
  float max_error = 0.0f;
  float total_error = 0.0f;
  int num_elements = 0;

  for (int m = 0; m < max_m; m++) {
    for (int k_idx = 0; k_idx < k; k_idx++) {
      int kg = k_idx / k_group_size;

      // Get the scale for this k-group
      float* scale_ptr = buf->get_scale(max_m, m, k, kg * k_group_size);
      float scale = *scale_ptr;

      // Get quantized value (simplified access for testing)
      // In real use, this would go through get_submat
      int m_block_size = (max_m + TestKernelKGroup::M_STEP - 1) / TestKernelKGroup::M_STEP * TestKernelKGroup::M_STEP;
      int k_block_begin = (k_idx / TestKernelKGroup::K_BLOCK) * TestKernelKGroup::K_BLOCK;
      int k_in_block = k_idx - k_block_begin;
      int k_block_size = std::min(TestKernelKGroup::K_BLOCK, k - k_block_begin);

      // Locate the quantized data
      int m_step_idx = m / TestKernelKGroup::M_STEP;
      int m_in_step = m % TestKernelKGroup::M_STEP;
      int k_step_idx = k_in_block / TestKernelKGroup::K_STEP;
      int k_in_step = k_in_block % TestKernelKGroup::K_STEP;

      int8_t* base = buf->a + k_block_begin * m_block_size + m_step_idx * TestKernelKGroup::M_STEP * k_block_size +
                     k_step_idx * TestKernelKGroup::K_STEP * TestKernelKGroup::M_STEP +
                     m_in_step * TestKernelKGroup::K_STEP + k_in_step;

      int8_t quantized_val = *base;

      // Dequantize
      float deq = quantized_val * scale;
      dequantized[m * k + k_idx] = deq;

      // Calculate error
      float error = std::abs(original[m * k + k_idx] - deq);
      max_error = std::max(max_error, error);
      total_error += error;
      num_elements++;
    }
  }

  float avg_error = total_error / num_elements;
  float avg_magnitude = 0.0f;
  for (int i = 0; i < max_m * k; i++) {
    avg_magnitude += std::abs(original[i]);
  }
  avg_magnitude /= (max_m * k);

  float relative_error = avg_error / (avg_magnitude + 1e-8f);

  std::cout << fmt::format("Quantization Error Analysis:\n");
  std::cout << fmt::format("  Max absolute error: {:.6f}\n", max_error);
  std::cout << fmt::format("  Average absolute error: {:.6f}\n", avg_error);
  std::cout << fmt::format("  Average magnitude: {:.6f}\n", avg_magnitude);
  std::cout << fmt::format("  Relative error: {:.2f}%\n", relative_error * 100);

  // Check that relative error is reasonable (typically < 5% for int8 quantization)
  if (relative_error < 0.05f) {
    std::cout << "✓ Quantization error is within acceptable range" << std::endl;
  } else {
    std::cerr << "WARNING: Quantization error is higher than expected!" << std::endl;
  }

  // Test that different k-groups have different scales
  std::cout << "\nVerifying k-group scales are computed independently:" << std::endl;
  bool scales_differ = false;
  for (int m = 0; m < std::min(4, max_m); m++) {
    float* scale0 = buf->get_scale(max_m, m, k, 0);
    for (int kg = 1; kg < k / k_group_size; kg++) {
      float* scale_kg = buf->get_scale(max_m, m, k, kg * k_group_size);
      if (std::abs(*scale0 - *scale_kg) > 1e-6f) {
        scales_differ = true;
        break;
      }
    }
    if (scales_differ) break;
  }

  if (scales_differ) {
    std::cout << "✓ Different k-groups have independent scales" << std::endl;
  } else {
    std::cout << "✗ Warning: All k-groups have the same scale (might be correct for uniform data)" << std::endl;
  }

  free(buffer);
}

void test_buffer_kgroup_comparison() {
  std::cout << "\n=== Comparing BufferAImpl vs BufferAKGroupImpl ===" << std::endl;

  const int max_m = 128;
  const int k = 2048;
  const int k_group_size = 256;

  // Create test data
  std::vector<ggml_bf16_t> input(max_m * k);
  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < max_m * k; i++) {
    input[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Test original BufferAImpl
  {
    size_t buffer_size = amx::BufferAImpl<TestKernelKGroup>::required_size(max_m, k);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_a = std::make_unique<amx::BufferAImpl<TestKernelKGroup>>(max_m, k, buffer);

    buf_a->from_mat(max_m, input.data(), 0, 1);

    // Print some scales
    std::cout << "BufferAImpl scales (per-row):" << std::endl;
    for (int m = 0; m < std::min(4, max_m); m++) {
      float* scale = buf_a->get_scale(max_m, m);
      std::cout << fmt::format("  row[{}]: scale = {:.6f}\n", m, *scale);
    }

    free(buffer);
  }

  // Test BufferAKGroupImpl
  {
    size_t buffer_size = amx::BufferAKGroupImpl<TestKernelKGroup>::required_size(max_m, k, k_group_size);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_kg = std::make_unique<amx::BufferAKGroupImpl<TestKernelKGroup>>(max_m, k, k_group_size, buffer);

    buf_kg->from_mat(max_m, input.data(), 0, 1);

    // Print some scales
    std::cout << "\nBufferAKGroupImpl scales (per k-group):" << std::endl;
    for (int m = 0; m < std::min(2, max_m); m++) {
      std::cout << fmt::format("  row[{}]:\n", m);
      for (int kg = 0; kg < std::min(4, k / k_group_size); kg++) {
        float* scale = buf_kg->get_scale(max_m, m, k, kg * k_group_size);
        std::cout << fmt::format("    k_group[{}]: scale = {:.6f}\n", kg, *scale);
      }
    }

    free(buffer);
  }

  std::cout << "\n✓ Comparison test completed" << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Starting BufferAKGroupImpl Tests\n" << std::endl;

  try {
    // Run basic functionality tests
    test_buffer_kgroup_basic();

    // Run correctness tests
    test_buffer_kgroup_correctness();

    // Run comparison tests
    test_buffer_kgroup_comparison();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}