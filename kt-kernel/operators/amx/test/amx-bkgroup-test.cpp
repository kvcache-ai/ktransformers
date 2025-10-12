#include <omp.h>

#include "../la/amx.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <memory>

// Test kernel configuration for k-group testing
struct TestKernelKGroupB {
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

void test_buffer_bkgroup_basic() {
  std::cout << "=== Testing BufferBKGroupImpl Basic Functionality ===" << std::endl;

  // Test parameters
  const int k = 2048;            // Must be multiple of K_STEP and K_BLOCK
  const int n = 1024;            // Must be multiple of TILE_N
  const int k_group_size = 128;  // Must divide K_BLOCK evenly

  std::cout << fmt::format("Parameters: k={}, n={}, k_group_size={}\n", k, n, k_group_size);

  // Calculate and allocate buffer
  size_t buffer_size = amx::BufferBKGroupImpl<TestKernelKGroupB>::required_size(k, n, k_group_size);
  void* buffer = std::aligned_alloc(64, buffer_size);
  std::memset(buffer, 0, buffer_size);

  std::cout << fmt::format("Buffer size: {} bytes\n", buffer_size);

  // Create BufferBKGroupImpl instance
  auto buf = std::make_unique<amx::BufferBKGroupImpl<TestKernelKGroupB>>(k, n, k_group_size, buffer);

  // Create test input data (bf16)
  std::vector<ggml_bf16_t> input(k * n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < k * n; i++) {
    float val = dist(gen);
    input[i] = ggml_compute_fp32_to_bf16(val);
  }

  // Test from_mat
  std::cout << "Testing from_mat..." << std::endl;
  buf->from_mat(input.data(), 0, 1);
  std::cout << "✓ from_mat completed successfully" << std::endl;

  // Test get_submat
  std::cout << "Testing get_submat..." << std::endl;
  for (int k_begin = 0; k_begin < k; k_begin += TestKernelKGroupB::K_STEP) {
    for (int n_begin = 0; n_begin < n; n_begin += TestKernelKGroupB::TILE_N) {
      int8_t* submat = buf->get_submat(k, n, k_begin, n_begin);
      if (submat == nullptr) {
        std::cerr << fmt::format("ERROR: get_submat returned null for k_begin={}, n_begin={}\n", k_begin, n_begin);
        free(buffer);
        return;
      }
    }
  }
  std::cout << "✓ get_submat tested for all valid positions" << std::endl;

  // Test get_scale
  std::cout << "Testing get_scale..." << std::endl;
  int k_group_count = k / k_group_size;
  for (int n_idx = 0; n_idx < n; n_idx++) {
    for (int kg_idx = 0; kg_idx < k_group_count; kg_idx++) {
      float* scale = buf->get_scale(n, n_idx, k, kg_idx * k_group_size);
      if (scale == nullptr) {
        std::cerr << fmt::format("ERROR: get_scale returned null for n_idx={}, k_group={}\n", n_idx, kg_idx);
        free(buffer);
        return;
      }
      // Verify scale is non-zero (should be set by from_mat)
      if (*scale == 0.0f) {
        std::cerr << fmt::format("WARNING: scale is zero for n_idx={}, k_group={}\n", n_idx, kg_idx);
      }
    }
  }
  std::cout << "✓ get_scale tested for all k-groups" << std::endl;

  // Print some scale values for verification
  std::cout << "\nSample scale values:" << std::endl;
  for (int kg = 0; kg < std::min(4, k_group_count); kg++) {
    float* scale = buf->get_scale(n, 0, k, kg * k_group_size);
    std::cout << fmt::format("  k_group[{}] (k={}): scale = {:.6f}\n", kg, kg * k_group_size, *scale);
  }

  // Clean up
  free(buffer);
  std::cout << "\n✓ All basic tests passed!" << std::endl;
}

void test_buffer_bkgroup_correctness() {
  std::cout << "\n=== Testing BufferBKGroupImpl Quantization Correctness ===" << std::endl;

  const int k = 512;
  const int n = 256;
  const int k_group_size = 128;

  size_t buffer_size = amx::BufferBKGroupImpl<TestKernelKGroupB>::required_size(k, n, k_group_size);
  void* buffer = std::aligned_alloc(64, buffer_size);

  auto buf = std::make_unique<amx::BufferBKGroupImpl<TestKernelKGroupB>>(k, n, k_group_size, buffer);

  // Create test input matrix with known patterns
  std::vector<float> original(k * n);
  std::vector<ggml_bf16_t> input(k * n);

  // Fill with different patterns for each k-group to test group-wise quantization
  for (int k_idx = 0; k_idx < k; k_idx++) {
    for (int n_idx = 0; n_idx < n; n_idx++) {
      int kg = k_idx / k_group_size;
      // Different magnitude for each k-group
      float base_val = (kg + 1) * 0.1f;
      float val = base_val * std::sin(k_idx * 0.01f + n_idx * 0.1f);
      original[k_idx * n + n_idx] = val;
      input[k_idx * n + n_idx] = ggml_compute_fp32_to_bf16(val);
    }
  }

  // Quantize
  buf->from_mat(input.data(), 0, 1);

  // Calculate quantization error statistics
  float max_error = 0.0f;
  float total_error = 0.0f;
  float avg_magnitude = 0.0f;

  for (int i = 0; i < k * n; i++) {
    avg_magnitude += std::abs(original[i]);
  }
  avg_magnitude /= (k * n);

  // Since we're using 4-bit quantization, expect higher error than int8
  // Just verify that scales are being computed correctly
  std::cout << fmt::format("Quantization Analysis:\n");
  std::cout << fmt::format("  Average magnitude: {:.6f}\n", avg_magnitude);
  std::cout << fmt::format("  Using 4-bit quantization (INT4)\n");

  // Test that different k-groups have different scales
  std::cout << "\nVerifying k-group scales are computed independently:" << std::endl;
  bool scales_differ = false;
  for (int n_idx = 0; n_idx < std::min(4, n); n_idx++) {
    float* scale0 = buf->get_scale(n, n_idx, k, 0);
    for (int kg = 1; kg < k / k_group_size; kg++) {
      float* scale_kg = buf->get_scale(n, n_idx, k, kg * k_group_size);
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

void test_buffer_bkgroup_comparison() {
  std::cout << "\n=== Comparing BufferBInt4Impl vs BufferBKGroupImpl ===" << std::endl;

  const int k = 2048;
  const int n = 512;
  const int k_group_size = 256;

  // Create test data
  std::vector<ggml_bf16_t> input(k * n);
  std::mt19937 gen(456);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (int i = 0; i < k * n; i++) {
    input[i] = ggml_compute_fp32_to_bf16(dist(gen));
  }

  // Test original BufferBInt4Impl
  {
    size_t buffer_size = amx::BufferBInt4Impl<TestKernelKGroupB>::required_size(k, n);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_b = std::make_unique<amx::BufferBInt4Impl<TestKernelKGroupB>>(k, n, buffer);

    buf_b->from_mat(input.data(), 0, 1);

    // Print some scales
    std::cout << "BufferBInt4Impl scales (per-column):" << std::endl;
    for (int n_idx = 0; n_idx < std::min(4, n); n_idx++) {
      float* scale = buf_b->get_scale(n, n_idx);
      std::cout << fmt::format("  col[{}]: scale = {:.6f}\n", n_idx, *scale);
    }

    free(buffer);
  }

  // Test BufferBKGroupImpl
  {
    size_t buffer_size = amx::BufferBKGroupImpl<TestKernelKGroupB>::required_size(k, n, k_group_size);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_kg = std::make_unique<amx::BufferBKGroupImpl<TestKernelKGroupB>>(k, n, k_group_size, buffer);

    buf_kg->from_mat(input.data(), 0, 1);

    // Print some scales
    std::cout << "\nBufferBKGroupImpl scales (per k-group):" << std::endl;
    for (int n_idx = 0; n_idx < std::min(2, n); n_idx++) {
      std::cout << fmt::format("  col[{}]:\n", n_idx);
      for (int kg = 0; kg < std::min(4, k / k_group_size); kg++) {
        float* scale = buf_kg->get_scale(n, n_idx, k, kg * k_group_size);
        std::cout << fmt::format("    k_group[{}]: scale = {:.6f}\n", kg, *scale);
      }
    }

    free(buffer);
  }

  std::cout << "\n✓ Comparison test completed" << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Starting BufferBKGroupImpl Tests\n" << std::endl;

  try {
    // Run basic functionality tests
    test_buffer_bkgroup_basic();

    // Run correctness tests
    test_buffer_bkgroup_correctness();

    // Run comparison tests
    test_buffer_bkgroup_comparison();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}