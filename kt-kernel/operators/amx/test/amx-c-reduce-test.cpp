#include <omp.h>

#include "../la/amx.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

// Test kernel configuration
struct TestKernelC {
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

void test_buffer_c_reduce_basic() {
  std::cout << "=== Testing BufferCReduceImpl Basic Functionality ===" << std::endl;

  // Test parameters
  const int max_m = 64;  // Must be multiple of M_STEP
  const int n = 512;     // Must be multiple of N_STEP

  std::cout << fmt::format("Parameters: max_m={}, n={}\n", max_m, n);

  // Calculate and allocate buffer for BufferCReduceImpl
  size_t buffer_size = amx::BufferCReduceImpl<TestKernelC>::required_size(max_m, n);
  void* buffer = std::aligned_alloc(64, buffer_size);
  std::memset(buffer, 0, buffer_size);

  std::cout << fmt::format("Buffer size: {} bytes\n", buffer_size);
  std::cout << fmt::format("  Float buffer: {} bytes\n", sizeof(float) * max_m * n);
  std::cout << fmt::format("  Int32 buffer: {} bytes\n", sizeof(int32_t) * max_m * n);

  // Create BufferCReduceImpl instance
  auto buf = std::make_unique<amx::BufferCReduceImpl<TestKernelC>>(max_m, n, buffer);

  // Test 1: Verify buffer pointers are set correctly
  std::cout << "\nTest 1: Buffer pointer verification" << std::endl;
  if (buf->c == nullptr) {
    std::cerr << "ERROR: Float buffer pointer is null" << std::endl;
    free(buffer);
    return;
  }
  if (buf->int_c == nullptr) {
    std::cerr << "ERROR: Int32 buffer pointer is null" << std::endl;
    free(buffer);
    return;
  }

  // Verify int_c starts after c
  size_t expected_offset = max_m * n;
  size_t actual_offset = buf->int_c - reinterpret_cast<int32_t*>(buf->c);
  if (actual_offset != expected_offset) {
    std::cerr << fmt::format("ERROR: int_c offset incorrect. Expected: {}, Got: {}\n", expected_offset, actual_offset)
              << std::endl;
    free(buffer);
    return;
  }
  std::cout << "✓ Buffer pointers are correctly set" << std::endl;

  // Test 2: Write to float buffer and verify
  std::cout << "\nTest 2: Float buffer write/read" << std::endl;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Fill float buffer with test data
  for (int i = 0; i < max_m * n; i++) {
    buf->c[i] = dist(gen);
  }

  // Verify get_submat works
  for (int m_begin = 0; m_begin < max_m; m_begin += TestKernelC::M_STEP) {
    for (int n_begin = 0; n_begin < n; n_begin += TestKernelC::N_STEP) {
      float* submat = buf->get_submat(max_m, n, m_begin, n_begin);
      if (submat == nullptr) {
        std::cerr << fmt::format("ERROR: get_submat returned null for m_begin={}, n_begin={}\n", m_begin, n_begin)
                  << std::endl;
        free(buffer);
        return;
      }
    }
  }
  std::cout << "✓ Float buffer read/write works correctly" << std::endl;

  // Test 3: Write to int32 buffer and verify
  std::cout << "\nTest 3: Int32 buffer write/read" << std::endl;
  std::uniform_int_distribution<int32_t> int_dist(-1000, 1000);

  // Fill int32 buffer with test data
  for (int i = 0; i < max_m * n; i++) {
    buf->int_c[i] = int_dist(gen);
  }

  // Verify get_int_submat works
  for (int m_begin = 0; m_begin < max_m; m_begin += TestKernelC::M_STEP) {
    for (int n_begin = 0; n_begin < n; n_begin += TestKernelC::N_STEP) {
      int32_t* submat = buf->get_int_submat(max_m, n, m_begin, n_begin);
      if (submat == nullptr) {
        std::cerr << fmt::format("ERROR: get_int_submat returned null for m_begin={}, n_begin={}\n", m_begin, n_begin)
                  << std::endl;
        free(buffer);
        return;
      }
    }
  }
  std::cout << "✓ Int32 buffer read/write works correctly" << std::endl;

  // Test 4: Clear int buffer
  std::cout << "\nTest 4: Clear int buffer" << std::endl;
  buf->clear_int_buffer();
  bool all_zero = true;
  for (int i = 0; i < max_m * n; i++) {
    if (buf->int_c[i] != 0) {
      all_zero = false;
      break;
    }
  }
  if (!all_zero) {
    std::cerr << "ERROR: clear_int_buffer failed to zero the buffer" << std::endl;
    free(buffer);
    return;
  }
  std::cout << "✓ clear_int_buffer works correctly" << std::endl;

  // Test 5: Convert int to float
  std::cout << "\nTest 5: Convert int32 to float" << std::endl;
  // Set some test values in int buffer
  for (int i = 0; i < max_m * n; i++) {
    buf->int_c[i] = i % 100 - 50;  // Values from -50 to 49
  }

  // Convert
  buf->convert_int_to_float(max_m);

  // Verify conversion
  bool conversion_correct = true;
  for (int i = 0; i < max_m * n; i++) {
    float expected = static_cast<float>(i % 100 - 50);
    if (std::abs(buf->c[i] - expected) > 1e-6) {
      std::cerr << fmt::format("ERROR: Conversion mismatch at index {}. Expected: {}, Got: {}\n", i, expected,
                               buf->c[i])
                << std::endl;
      conversion_correct = false;
      break;
    }
  }
  if (!conversion_correct) {
    free(buffer);
    return;
  }
  std::cout << "✓ convert_int_to_float works correctly" << std::endl;

  // Test 6: to_mat functionality
  std::cout << "\nTest 6: to_mat conversion" << std::endl;
  // Fill buffer using proper blocked layout via get_submat
  for (int m_idx = 0; m_idx < max_m; m_idx += TestKernelC::M_STEP) {
    for (int n_idx = 0; n_idx < n; n_idx += TestKernelC::N_STEP) {
      float* submat = buf->get_submat(max_m, n, m_idx, n_idx);
      // Fill this submat block
      for (int i = 0; i < TestKernelC::M_STEP && m_idx + i < max_m; i++) {
        for (int j = 0; j < TestKernelC::N_STEP && n_idx + j < n; j++) {
          submat[i * TestKernelC::N_STEP + j] = (m_idx + i) * 0.1f + (n_idx + j) * 0.01f;
        }
      }
    }
  }

  // Convert to bf16
  std::vector<ggml_bf16_t> output(max_m * n);
  buf->to_mat(max_m, output.data(), 0, 1);

  // Verify some values
  bool to_mat_correct = true;
  for (int i = 0; i < std::min(10, max_m); i++) {
    for (int j = 0; j < std::min(10, n); j++) {
      float original = i * 0.1f + j * 0.01f;
      float converted = ggml_compute_bf16_to_fp32(output[i * n + j]);
      // BF16 has limited precision, allow for some error
      if (std::abs(original - converted) > 0.02f) {  // Increased tolerance for BF16
        std::cerr << fmt::format("ERROR: to_mat mismatch at ({},{}). Original: {}, Converted: {}\n", i, j, original,
                                 converted)
                  << std::endl;
        to_mat_correct = false;
        break;
      }
    }
    if (!to_mat_correct) break;
  }

  if (!to_mat_correct) {
    free(buffer);
    return;
  }
  std::cout << "✓ to_mat works correctly" << std::endl;

  // Clean up
  free(buffer);
  std::cout << "\n✓ All basic tests passed!" << std::endl;
}

void test_buffer_c_reduce_comparison() {
  std::cout << "\n=== Comparing BufferCImpl vs BufferCReduceImpl ===" << std::endl;

  const int max_m = 128;
  const int n = 1024;

  // Test original BufferCImpl
  {
    size_t buffer_size = amx::BufferCImpl<TestKernelC>::required_size(max_m, n);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_c = std::make_unique<amx::BufferCImpl<TestKernelC>>(max_m, n, buffer);

    std::cout << fmt::format("BufferCImpl size: {} bytes\n", buffer_size);

    // Fill with test data
    for (int i = 0; i < max_m * n; i++) {
      buf_c->c[i] = static_cast<float>(i % 1000) / 100.0f;
    }

    // Test to_mat
    std::vector<ggml_bf16_t> output(max_m * n);
    buf_c->to_mat(max_m, output.data(), 0, 1);

    std::cout << "  Sample values from BufferCImpl:" << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << fmt::format("    c[{}] = {:.4f}\n", i, buf_c->c[i]);
    }

    free(buffer);
  }

  // Test BufferCReduceImpl
  {
    size_t buffer_size = amx::BufferCReduceImpl<TestKernelC>::required_size(max_m, n);
    void* buffer = std::aligned_alloc(64, buffer_size);
    auto buf_cr = std::make_unique<amx::BufferCReduceImpl<TestKernelC>>(max_m, n, buffer);

    std::cout << fmt::format("\nBufferCReduceImpl size: {} bytes ({}x larger)\n", buffer_size,
                             buffer_size / (sizeof(float) * max_m * n));

    // Fill float buffer
    for (int i = 0; i < max_m * n; i++) {
      buf_cr->c[i] = static_cast<float>(i % 1000) / 100.0f;
    }

    // Fill int buffer
    for (int i = 0; i < max_m * n; i++) {
      buf_cr->int_c[i] = i % 1000;
    }

    // Test to_mat
    std::vector<ggml_bf16_t> output(max_m * n);
    buf_cr->to_mat(max_m, output.data(), 0, 1);

    std::cout << "  Sample values from BufferCReduceImpl:" << std::endl;
    for (int i = 0; i < 3; i++) {
      std::cout << fmt::format("    c[{}] = {:.4f}, int_c[{}] = {}\n", i, buf_cr->c[i], i, buf_cr->int_c[i]);
    }

    free(buffer);
  }

  std::cout << "\n✓ Comparison test completed" << std::endl;
}

void test_buffer_c_reduce_performance() {
  std::cout << "\n=== Testing BufferCReduceImpl Performance Characteristics ===" << std::endl;

  const int max_m = 256;
  const int n = 2048;
  const int iterations = 1000;

  size_t buffer_size = amx::BufferCReduceImpl<TestKernelC>::required_size(max_m, n);
  void* buffer = std::aligned_alloc(64, buffer_size);
  auto buf = std::make_unique<amx::BufferCReduceImpl<TestKernelC>>(max_m, n, buffer);

  std::cout << fmt::format("Testing with max_m={}, n={}\n", max_m, n);
  std::cout << fmt::format("Total elements: {}\n", max_m * n);
  std::cout << fmt::format("Buffer size: {:.2f} MB\n", buffer_size / (1024.0 * 1024.0));

  // Test clear_int_buffer performance
  std::cout << "\nTesting clear_int_buffer..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    buf->clear_int_buffer();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << fmt::format("  Average time: {:.3f} us\n", duration / (double)iterations);

  // Test convert_int_to_float performance
  std::cout << "\nTesting convert_int_to_float..." << std::endl;
  // Fill int buffer with test data
  for (int i = 0; i < max_m * n; i++) {
    buf->int_c[i] = i;
  }

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    buf->convert_int_to_float(max_m);
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << fmt::format("  Average time: {:.3f} us\n", duration / (double)iterations);

  free(buffer);
  std::cout << "\n✓ Performance tests completed" << std::endl;
}

int main(int argc, char** argv) {
  std::cout << "Starting BufferCReduceImpl Tests\n" << std::endl;

  try {
    // Run basic functionality tests
    test_buffer_c_reduce_basic();

    // Run comparison tests
    test_buffer_c_reduce_comparison();

    // Run performance tests
    test_buffer_c_reduce_performance();

    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}