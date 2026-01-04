/**
 * @Description  : Test for NVFP4 MoE operator
 * @Author       : Claude & KVCache.AI Team
 * @Date         : 2025-01-17
 * @Version      : 0.1.0
 * @Copyright (c) 2025 by KVCache.AI, All Rights Reserved.
 **/

#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "../la/nvfp4_kernel.hpp"
#include "../la/nvfp4_utils.hpp"

// Test LUT multiplication correctness
void test_lut_multiplication() {
  std::cout << "=== Testing LUT Multiplication ===" << std::endl;

  // E2M1 values
  const float E2M1_VALUES[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  int errors = 0;
  // Test all 64 unsigned combinations
  for (int a = 0; a < 8; a++) {
    for (int b = 0; b < 8; b++) {
      float expected = E2M1_VALUES[a] * E2M1_VALUES[b];

      // Lookup via INDEX_LUT
      int idx_6bit = (a << 3) | b;
      int result_idx = nvfp4::nvfp4_lut::INDEX_LUT[idx_6bit];

      // Lookup result value (scaled by 4)
      int result_scaled = nvfp4::nvfp4_lut::RESULT_TABLE[result_idx];
      float result_f32 = result_scaled / 4.0f;

      if (std::abs(expected - result_f32) > 1e-6) {
        printf("  ERROR: %d×%d = %.2f, but LUT gives idx=%d, val=%d (%.2f)\n", a, b, expected, result_idx,
               result_scaled, result_f32);
        errors++;
      }
    }
  }

  if (errors == 0) {
    std::cout << "✓ All 64 LUT combinations verified correct" << std::endl;
  } else {
    std::cout << "✗ Found " << errors << " LUT errors" << std::endl;
  }

  // Test full AVX512 multiplication function with signs
  std::cout << "Testing AVX512 multiplication with signs..." << std::endl;

  alignas(64) uint8_t a_vals[64];
  alignas(64) uint8_t b_vals[64];
  alignas(64) int16_t results[64];

  // Test pattern: pairs of (2, 3) with various signs
  // 2 × 3 = 6, scaled by 4 = 24
  for (int i = 0; i < 64; i++) {
    uint8_t sign_a = (i & 1) ? 0x08 : 0x00;  // Alternating signs
    uint8_t sign_b = (i & 2) ? 0x08 : 0x00;
    a_vals[i] = 0x04 | sign_a;  // value=4 (which is E2M1 encoding for 2.0)
    b_vals[i] = 0x05 | sign_b;  // value=5 (which is E2M1 encoding for 3.0)
  }

  __m512i a_vec = _mm512_load_si512((const __m512i*)a_vals);
  __m512i b_vec = _mm512_load_si512((const __m512i*)b_vals);
  nvfp4::nvfp4_mul_64pairs_avx512(a_vec, b_vec, results);

  int sign_errors = 0;
  for (int i = 0; i < 64; i++) {
    bool neg_a = (i & 1) != 0;
    bool neg_b = (i & 2) != 0;
    bool expected_neg = neg_a != neg_b;          // XOR
    int16_t expected = expected_neg ? -24 : 24;  // 2 × 3 × 4 = 24

    if (results[i] != expected) {
      printf("  ERROR at [%d]: expected %d, got %d (neg_a=%d, neg_b=%d)\n", i, expected, results[i], neg_a, neg_b);
      sign_errors++;
    }
  }

  if (sign_errors == 0) {
    std::cout << "✓ AVX512 sign handling verified correct" << std::endl;
  } else {
    std::cout << "✗ Found " << sign_errors << " sign handling errors" << std::endl;
  }

  std::cout << std::endl;
}

// Test basic E2M1 quantization and dequantization
void test_fp4_quantization() {
  std::cout << "=== Testing E2M1 FP4 Quantization ===" << std::endl;

  std::vector<float> test_values = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f, 6.0f,
                                    -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

  std::cout << "Testing float <-> E2M1 conversion:" << std::endl;
  for (float val : test_values) {
    uint8_t e2m1 = nvfp4::float_to_e2m1(val);
    float recovered = nvfp4::e2m1_to_float(e2m1);
    float error = std::abs(val - recovered);

    printf("  %.2f -> E2M1(%02x) -> %.2f, error: %.4f\n", val, e2m1, recovered, error);

    // Check if error is reasonable (should be exact for the base values)
    if (error > 1e-6) {
      std::cerr << "ERROR: Conversion error too large!" << std::endl;
    }
  }

  std::cout << "✓ E2M1 quantization test passed" << std::endl << std::endl;
}

// Test NVFP4 block quantization
void test_block_quantization() {
  std::cout << "=== Testing NVFP4 Block Quantization ===" << std::endl;

  const int k = 64;  // 4 blocks of 16
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  // Create test data
  std::vector<float> test_data(k);
  for (int i = 0; i < k; i++) {
    test_data[i] = dist(gen);
  }

  // Quantize using NVFP4Block
  const int num_blocks = k / 16;
  std::vector<nvfp4::NVFP4Block> blocks(num_blocks);
  for (int blk = 0; blk < num_blocks; blk++) {
    blocks[blk].quantize(test_data.data() + blk * 16);
  }

  // Dequantize
  std::vector<float> dequant_data(k);
  for (int blk = 0; blk < num_blocks; blk++) {
    blocks[blk].dequantize(dequant_data.data() + blk * 16);
  }

  // Check error
  float max_error = 0.0f;
  float sum_error = 0.0f;
  for (int i = 0; i < k; i++) {
    float error = std::abs(test_data[i] - dequant_data[i]);
    max_error = std::max(max_error, error);
    sum_error += error;
  }

  float avg_error = sum_error / k;
  printf("Max error: %.6f, Avg error: %.6f\n", max_error, avg_error);

  // Print some values
  std::cout << "Sample values (original -> dequant):" << std::endl;
  for (int i = 0; i < std::min(8, k); i++) {
    printf("  [%2d] %.4f -> %.4f\n", i, test_data[i], dequant_data[i]);
  }

  std::cout << "✓ NVFP4 block quantization test passed" << std::endl << std::endl;
}

// Test BufferB loading
void test_buffer_b_loading() {
  std::cout << "=== Testing BufferB Loading ===" << std::endl;

  const int n = 128;
  const int k = 256;

  size_t buffer_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(n, k);
  void* buffer = std::aligned_alloc(64, buffer_size);

  auto buf = std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(n, k, buffer);

  printf("BufferB created: n=%d, k=%d, buffer_size=%zu bytes\n", n, k, buffer_size);

  // Create random quantized weight data
  std::vector<uint8_t> packed_weights(n * k / 2);
  std::vector<uint8_t> scales(n * k / 16);
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint8_t> dist(0, 255);

  for (size_t i = 0; i < packed_weights.size(); i++) {
    packed_weights[i] = dist(gen);
  }
  for (size_t i = 0; i < scales.size(); i++) {
    scales[i] = dist(gen);
  }

  // Load weights
  buf->from_raw_nvfp4(packed_weights.data(), scales.data(), 1.0f, 0, 1);

  std::cout << "✓ BufferB loading test passed" << std::endl << std::endl;
  std::free(buffer);
}

// Test BufferA quantization from BF16
void test_buffer_a_quantization() {
  std::cout << "=== Testing BufferA Quantization ===" << std::endl;

  const int max_m = 32;
  const int k = 256;

  size_t buffer_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(max_m, k);
  void* buffer = std::aligned_alloc(64, buffer_size);

  auto buf = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(max_m, k, buffer);

  printf("BufferA created: max_m=%d, k=%d, buffer_size=%zu bytes\n", max_m, k, buffer_size);

  // Create random BF16 input data
  std::vector<ggml_bf16_t> input(max_m * k);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (int i = 0; i < max_m * k; i++) {
    float val = dist(gen);
    input[i] = ggml_compute_fp32_to_bf16(val);
  }

  // Quantize from BF16
  buf->from_bf16(max_m, input.data(), 0, 1);

  printf("Tensor scale: %.6f\n", buf->tensor_scale);

  // Verify block scales
  std::cout << "Sample block scale values (FP8 E4M3):" << std::endl;
  for (int i = 0; i < std::min(4, max_m * k / 16); i++) {
    uint8_t scale_fp8 = buf->block_scales[i];
    float scale_f32 = nvfp4::fp8_e4m3_to_float(scale_fp8);
    printf("  block[%d]: fp8=%02x, f32=%.6f\n", i, scale_fp8, scale_f32);
  }

  std::free(buffer);
  std::cout << "✓ BufferA quantization test passed" << std::endl << std::endl;
}

// Test opt4 correctness by comparing with baseline
void test_opt4_correctness() {
  std::cout << "=== Testing opt4 Kernel Correctness ===" << std::endl;

  const int m = 1;
  const int n = 512;
  const int k = 1024;

  // Allocate buffers
  size_t ba_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, k);
  size_t bb_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(n, k);
  size_t bc_size = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, n);
  size_t bc_size2 = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, n);

  void* ba_buffer = std::aligned_alloc(64, ba_size);
  void* bb_buffer = std::aligned_alloc(64, bb_size);
  void* bc_buffer = std::aligned_alloc(64, bc_size);
  void* bc_buffer2 = std::aligned_alloc(64, bc_size2);

  auto ba = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, k, ba_buffer);
  auto bb = std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(n, k, bb_buffer);
  auto bc = std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, n, bc_buffer);
  auto bc_opt4 = std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, n, bc_buffer2);

  printf("Testing opt4 vs baseline: M=%d, N=%d, K=%d\n", m, n, k);

  // Create random test data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Create activation (BF16)
  std::vector<ggml_bf16_t> a_bf16(m * k);
  std::vector<float> a_f32(m * k);
  for (int i = 0; i < m * k; i++) {
    a_f32[i] = dist(gen);
    a_bf16[i] = ggml_compute_fp32_to_bf16(a_f32[i]);
  }

  // Create weights
  std::vector<float> b_f32(n * k);
  for (int i = 0; i < n * k; i++) {
    b_f32[i] = dist(gen);
  }

  // Quantize weights
  float b_global_max = 0.0f;
  for (int i = 0; i < n * k; i++) {
    b_global_max = std::max(b_global_max, std::abs(b_f32[i]));
  }
  float b_tensor_scale = b_global_max / 448.0f;
  float b_ts_inv = (b_tensor_scale > 1e-10f) ? (1.0f / b_tensor_scale) : 1.0f;

  const int k_group_size = 16;
  int k_group_count = k / k_group_size;
  std::vector<uint8_t> b_fp4(n * k / 2);
  std::vector<uint8_t> b_scales_fp8(n * k_group_count);

  for (int n_i = 0; n_i < n; n_i++) {
    for (int kg = 0; kg < k_group_count; kg++) {
      int k_start = kg * k_group_size;
      float max_val = 0.0f;
      for (int k_i = 0; k_i < k_group_size; k_i++) {
        float val = b_f32[n_i * k + k_start + k_i] * b_ts_inv;
        max_val = std::max(max_val, std::abs(val));
      }
      float block_scale = max_val / 6.0f;
      b_scales_fp8[n_i * k_group_count + kg] = nvfp4::float_to_fp8_e4m3(block_scale);
      float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

      for (int k_i = 0; k_i < k_group_size; k_i += 2) {
        float val0 = b_f32[n_i * k + k_start + k_i] * b_ts_inv * scale_inv;
        float val1 = b_f32[n_i * k + k_start + k_i + 1] * b_ts_inv * scale_inv;
        uint8_t q0 = nvfp4::float_to_e2m1(val0);
        uint8_t q1 = nvfp4::float_to_e2m1(val1);
        b_fp4[n_i * k / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
      }
    }
  }

  // Load buffers
  ba->from_bf16(m, a_bf16.data(), 0, 1);
  bb->from_raw_nvfp4(b_fp4.data(), b_scales_fp8.data(), b_tensor_scale, 0, 1);

  // Run baseline
  bc->clear(m);
  nvfp4::nvfp4_matmul(m, n, k, ba, bb, bc, 0, 1);

  // Run opt4
  bc_opt4->clear(m);
  nvfp4::nvfp4_matmul_opt4(m, n, k, ba, bb, bc_opt4, 0, 1);

  // Compare results
  float max_error = 0.0f;
  float sum_error = 0.0f;
  int errors = 0;

  for (int i = 0; i < m * n; i++) {
    float error = std::abs(bc->c_fp32[i] - bc_opt4->c_fp32[i]);
    max_error = std::max(max_error, error);
    sum_error += error;
    if (error > 1e-4) {
      errors++;
      if (errors <= 5) {
        printf("  ERROR at [%d]: baseline=%.6f, opt4=%.6f, diff=%.6f\n", i, bc->c_fp32[i], bc_opt4->c_fp32[i], error);
      }
    }
  }

  float avg_error = sum_error / (m * n);
  printf("Max error: %.6f, Avg error: %.6f, Errors: %d/%d\n", max_error, avg_error, errors, m * n);

  // Also compare with FP32 reference
  std::vector<float> c_ref(m * n, 0.0f);
  for (int m_i = 0; m_i < m; m_i++) {
    for (int n_i = 0; n_i < n; n_i++) {
      float sum = 0.0f;
      for (int k_i = 0; k_i < k; k_i++) {
        sum += a_f32[m_i * k + k_i] * b_f32[n_i * k + k_i];
      }
      c_ref[m_i * n + n_i] = sum;
    }
  }

  float max_ref_error = 0.0f;
  for (int i = 0; i < m * n; i++) {
    float error = std::abs(bc_opt4->c_fp32[i] - c_ref[i]);
    max_ref_error = std::max(max_ref_error, error);
  }
  printf("Max error vs FP32 reference: %.6f (expected due to quantization)\n", max_ref_error);

  std::free(ba_buffer);
  std::free(bb_buffer);
  std::free(bc_buffer);
  std::free(bc_buffer2);

  if (errors == 0) {
    std::cout << "✓ opt4 kernel matches baseline exactly" << std::endl;
  } else {
    std::cout << "✗ opt4 kernel has " << errors << " differences from baseline" << std::endl;
  }
  std::cout << std::endl;
}

// Test matrix multiplication kernel
void test_matrix_multiplication() {
  std::cout << "=== Testing Matrix Multiplication ===" << std::endl;

  const int m = 1;  // Single row (vector multiplication)
  const int n = 128;
  const int k = 256;

  // Allocate buffers
  size_t ba_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, k);
  size_t bb_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(n, k);
  size_t bc_size = nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(m, n);

  void* ba_buffer = std::aligned_alloc(64, ba_size);
  void* bb_buffer = std::aligned_alloc(64, bb_size);
  void* bc_buffer = std::aligned_alloc(64, bc_size);

  auto ba = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, k, ba_buffer);
  auto bb = std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(n, k, bb_buffer);
  auto bc = std::make_shared<nvfp4::BufferCNVFP4Impl<nvfp4::GemmKernelNVFP4>>(m, n, bc_buffer);

  printf("Matrix sizes: M=%d, N=%d, K=%d\n", m, n, k);

  // Create test data
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Create activation (BF16)
  std::vector<ggml_bf16_t> a_bf16(m * k);
  std::vector<float> a_f32(m * k);
  for (int i = 0; i < m * k; i++) {
    a_f32[i] = dist(gen);
    a_bf16[i] = ggml_compute_fp32_to_bf16(a_f32[i]);
  }

  // Create weights (FP4, quantized)
  std::vector<float> b_f32(n * k);
  for (int i = 0; i < n * k; i++) {
    b_f32[i] = dist(gen);
  }

  // Quantize weights to NVFP4 block format
  // First compute tensor_scale for B (global max)
  float b_global_max = 0.0f;
  for (int i = 0; i < n * k; i++) {
    b_global_max = std::max(b_global_max, std::abs(b_f32[i]));
  }
  float b_tensor_scale = b_global_max / 448.0f;
  float b_ts_inv = (b_tensor_scale > 1e-10f) ? (1.0f / b_tensor_scale) : 1.0f;

  const int k_group_size = 16;
  int k_group_count = k / k_group_size;
  std::vector<uint8_t> b_fp4(n * k / 2);
  std::vector<uint8_t> b_scales_fp8(n * k_group_count);

  for (int n_i = 0; n_i < n; n_i++) {
    for (int kg = 0; kg < k_group_count; kg++) {
      int k_start = kg * k_group_size;

      // Find max value in group (after applying tensor scale)
      float max_val = 0.0f;
      for (int k_i = 0; k_i < k_group_size; k_i++) {
        float val = b_f32[n_i * k + k_start + k_i] * b_ts_inv;
        max_val = std::max(max_val, std::abs(val));
      }

      float block_scale = max_val / 6.0f;  // Map to max E2M1 value
      b_scales_fp8[n_i * k_group_count + kg] = nvfp4::float_to_fp8_e4m3(block_scale);

      float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

      // Quantize group (apply tensor scale inverse, then block scale inverse)
      for (int k_i = 0; k_i < k_group_size; k_i += 2) {
        float val0 = b_f32[n_i * k + k_start + k_i] * b_ts_inv * scale_inv;
        float val1 = b_f32[n_i * k + k_start + k_i + 1] * b_ts_inv * scale_inv;

        uint8_t q0 = nvfp4::float_to_e2m1(val0);
        uint8_t q1 = nvfp4::float_to_e2m1(val1);

        b_fp4[n_i * k / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
      }
    }
  }

  // Load buffers
  ba->from_bf16(m, a_bf16.data(), 0, 1);
  bb->from_raw_nvfp4(b_fp4.data(), b_scales_fp8.data(), b_tensor_scale, 0, 1);

  // Clear output
  bc->clear(m);

  // Perform multiplication
  std::cout << "Performing matrix multiplication..." << std::endl;
  nvfp4::nvfp4_matmul(m, n, k, ba, bb, bc, 0, 1);

  // Compute reference result
  std::cout << "Computing reference result..." << std::endl;
  std::vector<float> c_ref(m * n, 0.0f);
  for (int m_i = 0; m_i < m; m_i++) {
    for (int n_i = 0; n_i < n; n_i++) {
      float sum = 0.0f;
      for (int k_i = 0; k_i < k; k_i++) {
        sum += a_f32[m_i * k + k_i] * b_f32[n_i * k + k_i];
      }
      c_ref[m_i * n + n_i] = sum;
    }
  }

  // Compare results
  std::cout << "Comparing results..." << std::endl;
  float max_error = 0.0f;
  float sum_error = 0.0f;
  int max_error_idx = -1;

  for (int i = 0; i < m * n; i++) {
    float error = std::abs(bc->c_fp32[i] - c_ref[i]);
    if (error > max_error) {
      max_error = error;
      max_error_idx = i;
    }
    sum_error += error;
  }

  float avg_error = sum_error / (m * n);
  printf("Max error: %.6f at index %d, Avg error: %.6f\n", max_error, max_error_idx, avg_error);

  // Print some sample results
  std::cout << "Sample results (NVFP4 vs Reference):" << std::endl;
  for (int i = 0; i < std::min(8, m * n); i++) {
    printf("  [%2d] %.6f vs %.6f\n", i, bc->c_fp32[i], c_ref[i]);
  }

  // Cleanup
  std::free(ba_buffer);
  std::free(bb_buffer);
  std::free(bc_buffer);

  std::cout << "✓ Matrix multiplication test completed" << std::endl << std::endl;
}

// Test MoE single token forward
void test_moe_single_token() {
  std::cout << "=== Testing MoE Single Token Forward ===" << std::endl;

  const int hidden_dim = 256;
  const int expert_dim = 512;
  const int num_experts = 8;
  const int top_k = 2;

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  // Create input activation
  size_t ba_size = nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(1, hidden_dim);
  void* ba_buffer = std::aligned_alloc(64, ba_size);
  auto input = std::make_shared<nvfp4::BufferANVFP4Impl<nvfp4::GemmKernelNVFP4>>(1, hidden_dim, ba_buffer);

  std::vector<ggml_bf16_t> input_bf16(hidden_dim);
  std::vector<float> input_f32(hidden_dim);
  for (int i = 0; i < hidden_dim; i++) {
    input_f32[i] = dist(gen);
    input_bf16[i] = ggml_compute_fp32_to_bf16(input_f32[i]);
  }
  input->from_bf16(1, input_bf16.data(), 0, 1);

  // Create experts
  nvfp4::MoEBufferBNVFP4Impl<nvfp4::GemmKernelNVFP4> experts(num_experts, expert_dim, hidden_dim);

  std::vector<void*> expert_buffers(num_experts);
  std::vector<std::vector<float>> expert_weights_f32(num_experts);
  std::vector<std::vector<uint8_t>> expert_fp4(num_experts);
  std::vector<std::vector<uint8_t>> expert_scales(num_experts);
  std::vector<float> expert_tensor_scales(num_experts);

  const int k_group_size = 16;
  int k_group_count = hidden_dim / k_group_size;

  for (int e = 0; e < num_experts; e++) {
    size_t bb_size = nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>::required_size(expert_dim, hidden_dim);
    expert_buffers[e] = std::aligned_alloc(64, bb_size);
    auto expert_buf =
        std::make_shared<nvfp4::BufferBNVFP4Impl<nvfp4::GemmKernelNVFP4>>(expert_dim, hidden_dim, expert_buffers[e]);

    // Generate random weights
    expert_weights_f32[e].resize(expert_dim * hidden_dim);
    for (int i = 0; i < expert_dim * hidden_dim; i++) {
      expert_weights_f32[e][i] = dist(gen);
    }

    // Quantize
    float b_global_max = 0.0f;
    for (int i = 0; i < expert_dim * hidden_dim; i++) {
      b_global_max = std::max(b_global_max, std::abs(expert_weights_f32[e][i]));
    }
    expert_tensor_scales[e] = b_global_max / 448.0f;
    float b_ts_inv = (expert_tensor_scales[e] > 1e-10f) ? (1.0f / expert_tensor_scales[e]) : 1.0f;

    expert_fp4[e].resize(expert_dim * hidden_dim / 2);
    expert_scales[e].resize(expert_dim * k_group_count);

    for (int n_i = 0; n_i < expert_dim; n_i++) {
      for (int kg = 0; kg < k_group_count; kg++) {
        int k_start = kg * k_group_size;
        float max_val = 0.0f;
        for (int k_i = 0; k_i < k_group_size; k_i++) {
          float val = expert_weights_f32[e][n_i * hidden_dim + k_start + k_i] * b_ts_inv;
          max_val = std::max(max_val, std::abs(val));
        }
        float block_scale = max_val / 6.0f;
        expert_scales[e][n_i * k_group_count + kg] = nvfp4::float_to_fp8_e4m3(block_scale);
        float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

        for (int k_i = 0; k_i < k_group_size; k_i += 2) {
          float val0 = expert_weights_f32[e][n_i * hidden_dim + k_start + k_i] * b_ts_inv * scale_inv;
          float val1 = expert_weights_f32[e][n_i * hidden_dim + k_start + k_i + 1] * b_ts_inv * scale_inv;
          uint8_t q0 = nvfp4::float_to_e2m1(val0);
          uint8_t q1 = nvfp4::float_to_e2m1(val1);
          expert_fp4[e][n_i * hidden_dim / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
        }
      }
    }

    expert_buf->from_raw_nvfp4(expert_fp4[e].data(), expert_scales[e].data(), expert_tensor_scales[e], 0, 1);
    experts.set_expert(e, expert_buf);
  }

  // Create gate logits (random)
  std::vector<float> gate_logits(num_experts);
  for (int e = 0; e < num_experts; e++) {
    gate_logits[e] = dist(gen) * 2.0f;  // Scale up for more variance
  }

  printf("Gate logits: ");
  for (int e = 0; e < num_experts; e++) {
    printf("%.3f ", gate_logits[e]);
  }
  printf("\n");

  // Run MoE forward
  std::vector<float> output(expert_dim, 0.0f);
  nvfp4::nvfp4_moe_single_token(hidden_dim, expert_dim, num_experts, top_k, input, experts, gate_logits.data(),
                                output.data());

  // Compute reference using FP32
  // First compute softmax and top-K
  std::vector<std::pair<float, int>> scores(num_experts);
  float max_logit = gate_logits[0];
  for (int e = 1; e < num_experts; e++) {
    max_logit = std::max(max_logit, gate_logits[e]);
  }
  float sum_exp = 0.0f;
  for (int e = 0; e < num_experts; e++) {
    scores[e] = {std::exp(gate_logits[e] - max_logit), e};
    sum_exp += scores[e].first;
  }
  for (int e = 0; e < num_experts; e++) {
    scores[e].first /= sum_exp;
  }
  std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  float top_k_sum = 0.0f;
  for (int i = 0; i < top_k; i++) {
    top_k_sum += scores[i].first;
  }

  printf("Top-%d experts selected:\n", top_k);
  for (int i = 0; i < top_k; i++) {
    float weight = scores[i].first / top_k_sum;
    int expert_idx = scores[i].second;
    printf("  Expert %d: weight=%.4f\n", expert_idx, weight);
  }

  // Compute reference output
  std::vector<float> ref_output(expert_dim, 0.0f);
  for (int i = 0; i < top_k; i++) {
    float weight = scores[i].first / top_k_sum;
    int expert_idx = scores[i].second;

    for (int n_i = 0; n_i < expert_dim; n_i++) {
      float sum = 0.0f;
      for (int k_i = 0; k_i < hidden_dim; k_i++) {
        sum += input_f32[k_i] * expert_weights_f32[expert_idx][n_i * hidden_dim + k_i];
      }
      ref_output[n_i] += weight * sum;
    }
  }

  // Compare
  float max_error = 0.0f;
  for (int i = 0; i < expert_dim; i++) {
    float error = std::abs(output[i] - ref_output[i]);
    max_error = std::max(max_error, error);
  }

  printf("Max error vs FP32 reference: %.6f (expected due to quantization)\n", max_error);

  printf("Sample output values:\n");
  for (int i = 0; i < std::min(8, expert_dim); i++) {
    printf("  [%2d] NVFP4=%.6f, Ref=%.6f\n", i, output[i], ref_output[i]);
  }

  // Cleanup
  std::free(ba_buffer);
  for (int e = 0; e < num_experts; e++) {
    std::free(expert_buffers[e]);
  }

  std::cout << "✓ MoE single token test completed" << std::endl << std::endl;
}

int main() {
  std::cout << "NVFP4 MoE Operator Test Suite" << std::endl;
  std::cout << "==============================" << std::endl << std::endl;

  try {
    test_lut_multiplication();
    test_fp4_quantization();
    test_block_quantization();
    test_buffer_b_loading();
    test_buffer_a_quantization();
    test_opt4_correctness();
    test_matrix_multiplication();
    test_moe_single_token();

    std::cout << std::endl << "All tests passed! ✓" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Test failed with exception: " << e.what() << std::endl;
    return 1;
  }
}
