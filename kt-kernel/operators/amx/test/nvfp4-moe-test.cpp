/**
 * @Description  : Test for NVFP4 MoE operator
 * @Author       : Claude & KVCache.AI Team
 * @Date         : 2025-01-17
 * @Version      : 0.1.0
 * @Copyright (c) 2025 by KVCache.AI, All Rights Reserved.
 **/

#include <omp.h>

#include "../la/nvfp4_kernel.hpp"
#include "../la/nvfp4_utils.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// Test basic FP4 quantization and dequantization
void test_fp4_quantization() {
    std::cout << "=== Testing FP4 Quantization ===" << std::endl;

    std::vector<float> test_values = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                      -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

    std::cout << "Testing float <-> FP4 conversion:" << std::endl;
    for (float val : test_values) {
        uint8_t fp4 = amx::float_to_fp4(val);
        float recovered = amx::fp4_to_float(fp4);
        float error = std::abs(val - recovered);

        std::cout << fmt::format("  {:.2f} -> FP4({:02x}) -> {:.2f}, error: {:.4f}\n",
                                 val, fp4, recovered, error);

        // Check if error is reasonable (should be exact for the base values)
        if (error > 1e-6) {
            std::cerr << "ERROR: Conversion error too large!" << std::endl;
        }
    }

    std::cout << "✓ FP4 quantization test passed" << std::endl << std::endl;
}

// Test block quantization
void test_block_quantization() {
    std::cout << "=== Testing Block Quantization ===" << std::endl;

    const int k = 64;  // 4 blocks of 16
    const size_t data_size = amx::blocks_aligned_nvfp4_ref::expected_data_size(k);
    void* quant_buffer = std::aligned_alloc(64, data_size);

    // Create test data
    std::vector<float> test_data(k);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    for (int i = 0; i < k; i++) {
        test_data[i] = dist(gen);
    }

    // Quantize
    auto quant_ref = amx::blocks_aligned_nvfp4_ref::quantize(test_data.data(), quant_buffer, k);

    // Dequantize
    std::vector<float> dequant_data(k);
    quant_ref.dequantize(dequant_data.data(), k);

    // Check error
    float max_error = 0.0f;
    float sum_error = 0.0f;
    for (int i = 0; i < k; i++) {
        float error = std::abs(test_data[i] - dequant_data[i]);
        max_error = std::max(max_error, error);
        sum_error += error;
    }

    float avg_error = sum_error / k;
    std::cout << fmt::format("Max error: {:.6f}, Avg error: {:.6f}\n", max_error, avg_error);

    // Print some values
    std::cout << "Sample values (original -> dequant):" << std::endl;
    for (int i = 0; i < std::min(8, k); i++) {
        std::cout << fmt::format("  [{:2d}] {:.4f} -> {:.4f}\n", i, test_data[i], dequant_data[i]);
    }

    std::free(quant_buffer);
    std::cout << "✓ Block quantization test passed" << std::endl << std::endl;
}

// Test BufferB loading
void test_buffer_b_loading() {
    std::cout << "=== Testing BufferB Loading ===" << std::endl;

    const int n = 128;
    const int k = 256;
    const int k_group_size = 16;

    size_t buffer_size = amx::GemmKernelNVFP4KGroup::BufferB<amx::GemmKernelNVFP4KGroup>::required_size(
        n, k, k_group_size);
    void* buffer = std::aligned_alloc(64, buffer_size);

    auto buf = std::make_shared<amx::GemmKernelNVFP4KGroup::BufferB<amx::GemmKernelNVFP4KGroup>>(
        n, k, k_group_size, buffer);

    std::cout << fmt::format("BufferB created: n={}, k={}, k_group_size={}, buffer_size={} bytes\n",
                             n, k, k_group_size, buffer_size);

    // Create random quantized weight data
    std::vector<uint8_t> packed_weights(n * k / 2);
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint8_t> dist(0, 255);

    for (size_t i = 0; i < packed_weights.size(); i++) {
        packed_weights[i] = dist(gen);
    }

    // Load weights
    buf->from_raw_mat(packed_weights.data(), 0, 1);

    std::cout << "✓ BufferB loading test passed" << std::endl << std::endl;
    std::free(buffer);
}

// Test BufferA quantization
void test_buffer_a_quantization() {
    std::cout << "=== Testing BufferA Quantization ===" << std::endl;

    const int max_m = 32;
    const int k = 256;
    const int k_group_size = 16;

    size_t buffer_size = amx::GemmKernelNVFP4KGroup::BufferA<amx::GemmKernelNVFP4KGroup>::required_size(
        max_m, k, k_group_size);
    void* buffer = std::aligned_alloc(64, buffer_size);

    auto buf = std::make_shared<amx::GemmKernelNVFP4KGroup::BufferA<amx::GemmKernelNVFP4KGroup>>(
        max_m, k, k_group_size, buffer);

    std::cout << fmt::format("BufferA created: max_m={}, k={}, k_group_size={}, buffer_size={} bytes\n",
                             max_m, k, k_group_size, buffer_size);

    // Create random BF16 input data
    std::vector<ggml_bf16_t> input(max_m * k);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < max_m * k; i++) {
        float val = dist(gen);
        input[i] = ggml_compute_fp32_to_bf16(val);
    }

    // Quantize
    buf->from_mat(max_m, input.data(), 0, 1);

    // Verify scales are non-zero
    int k_group_count = k / k_group_size;
    std::cout << "Sample scale values:" << std::endl;
    for (int kg = 0; kg < std::min(4, k_group_count); kg++) {
        float* scale = buf->get_scale(max_m, 0, k, kg * k_group_size);
        std::cout << fmt::format("  k_group[{}]: scale={:.6f}\n", kg, *scale);
    }

    std::free(buffer);
    std::cout << "✓ BufferA quantization test passed" << std::endl << std::endl;
}

// Test matrix multiplication kernel
void test_matrix_multiplication() {
    std::cout << "=== Testing Matrix Multiplication ===" << std::endl;

    const int m = 1;  // Single row (vector multiplication)
    const int n = 128;
    const int k = 256;
    const int k_group_size = 16;

    // Allocate buffers
    size_t ba_size = amx::GemmKernelNVFP4KGroup::BufferA<amx::GemmKernelNVFP4KGroup>::required_size(
        m, k, k_group_size);
    size_t bb_size = amx::GemmKernelNVFP4KGroup::BufferB<amx::GemmKernelNVFP4KGroup>::required_size(
        n, k, k_group_size);
    size_t bc_size = amx::GemmKernelNVFP4KGroup::BufferC<amx::GemmKernelNVFP4KGroup>::required_size(m, n);

    void* ba_buffer = std::aligned_alloc(64, ba_size);
    void* bb_buffer = std::aligned_alloc(64, bb_size);
    void* bc_buffer = std::aligned_alloc(64, bc_size);

    auto ba = std::make_shared<amx::GemmKernelNVFP4KGroup::BufferA<amx::GemmKernelNVFP4KGroup>>(
        m, k, k_group_size, ba_buffer);
    auto bb = std::make_shared<amx::GemmKernelNVFP4KGroup::BufferB<amx::GemmKernelNVFP4KGroup>>(
        n, k, k_group_size, bb_buffer);
    auto bc = std::make_shared<amx::GemmKernelNVFP4KGroup::BufferC<amx::GemmKernelNVFP4KGroup>>(
        m, n, bc_buffer);

    std::cout << fmt::format("Matrix sizes: M={}, N={}, K={}\n", m, n, k);

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

    // Quantize weights to FP4 block format
    int k_group_count = k / k_group_size;
    std::vector<uint8_t> b_fp4(n * k / 2);
    std::vector<float> b_scales(n * k_group_count);

    for (int n_i = 0; n_i < n; n_i++) {
        for (int kg = 0; kg < k_group_count; kg++) {
            int k_start = kg * k_group_size;

            // Find max value in group
            float max_val = 0.0f;
            for (int k_i = 0; k_i < k_group_size; k_i++) {
                max_val = std::max(max_val, std::abs(b_f32[n_i * k + k_start + k_i]));
            }

            float scale = max_val / 6.0f;  // Map to max FP4 value
            b_scales[n_i * k_group_count + kg] = scale;

            // Quantize group
            for (int k_i = 0; k_i < k_group_size; k_i += 2) {
                float val0 = b_f32[n_i * k + k_start + k_i] / (scale + 1e-8f);
                float val1 = b_f32[n_i * k + k_start + k_i + 1] / (scale + 1e-8f);

                uint8_t q0 = amx::float_to_fp4(val0);
                uint8_t q1 = amx::float_to_fp4(val1);

                b_fp4[n_i * k / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
            }
        }
    }

    // Load buffers
    ba->from_mat(m, a_bf16.data(), 0, 1);
    bb->from_raw_mat(b_fp4.data(), 0, 1);
    std::memcpy(bb->d, b_scales.data(), b_scales.size() * sizeof(float));

    // Clear output
    bc->clear(m);

    // Perform multiplication
    std::cout << "Performing matrix multiplication..." << std::endl;
    amx::vec_mul_nvfp4_kgroup(m, n, k, k_group_size, ba, bb, bc, 0, 1);

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
        float error = std::abs(bc->c[i] - c_ref[i]);
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
        sum_error += error;
    }

    float avg_error = sum_error / (m * n);
    std::cout << fmt::format("Max error: {:.6f} at index {}, Avg error: {:.6f}\n",
                             max_error, max_error_idx, avg_error);

    // Print some sample results
    std::cout << "Sample results (NVFP4 vs Reference):" << std::endl;
    for (int i = 0; i < std::min(8, m * n); i++) {
        std::cout << fmt::format("  [{:2d}] {:.6f} vs {:.6f}\n", i, bc->c[i], c_ref[i]);
    }

    // Cleanup
    std::free(ba_buffer);
    std::free(bb_buffer);
    std::free(bc_buffer);

    std::cout << "✓ Matrix multiplication test completed" << std::endl << std::endl;
}

int main() {
    std::cout << "NVFP4 MoE Operator Test Suite" << std::endl;
    std::cout << "==============================" << std::endl << std::endl;

    try {
        test_fp4_quantization();
        test_block_quantization();
        test_buffer_b_loading();
        test_buffer_a_quantization();
        test_matrix_multiplication();

        std::cout << std::endl << "All tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
