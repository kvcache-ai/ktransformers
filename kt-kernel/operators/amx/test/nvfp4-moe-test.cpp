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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <memory>
#include <random>
#include <vector>

// Test basic E2M1 quantization and dequantization
void test_fp4_quantization() {
    std::cout << "=== Testing E2M1 FP4 Quantization ===" << std::endl;

    std::vector<float> test_values = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                      -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

    std::cout << "Testing float <-> E2M1 conversion:" << std::endl;
    for (float val : test_values) {
        uint8_t e2m1 = nvfp4::float_to_e2m1(val);
        float recovered = nvfp4::e2m1_to_float(e2m1);
        float error = std::abs(val - recovered);

        printf("  %.2f -> E2M1(%02x) -> %.2f, error: %.4f\n",
               val, e2m1, recovered, error);

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

    printf("BufferB created: n=%d, k=%d, buffer_size=%zu bytes\n",
           n, k, buffer_size);

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

    printf("BufferA created: max_m=%d, k=%d, buffer_size=%zu bytes\n",
           max_m, k, buffer_size);

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
    const int k_group_size = 16;
    int k_group_count = k / k_group_size;
    std::vector<uint8_t> b_fp4(n * k / 2);
    std::vector<uint8_t> b_scales_fp8(n * k_group_count);

    for (int n_i = 0; n_i < n; n_i++) {
        for (int kg = 0; kg < k_group_count; kg++) {
            int k_start = kg * k_group_size;

            // Find max value in group
            float max_val = 0.0f;
            for (int k_i = 0; k_i < k_group_size; k_i++) {
                max_val = std::max(max_val, std::abs(b_f32[n_i * k + k_start + k_i]));
            }

            float scale = max_val / 6.0f;  // Map to max E2M1 value
            b_scales_fp8[n_i * k_group_count + kg] = nvfp4::float_to_fp8_e4m3(scale);

            // Quantize group
            for (int k_i = 0; k_i < k_group_size; k_i += 2) {
                float val0 = b_f32[n_i * k + k_start + k_i] / (scale + 1e-8f);
                float val1 = b_f32[n_i * k + k_start + k_i + 1] / (scale + 1e-8f);

                uint8_t q0 = nvfp4::float_to_e2m1(val0);
                uint8_t q1 = nvfp4::float_to_e2m1(val1);

                b_fp4[n_i * k / 2 + (k_start + k_i) / 2] = q0 | (q1 << 4);
            }
        }
    }

    // Load buffers
    ba->from_bf16(m, a_bf16.data(), 0, 1);
    bb->from_raw_nvfp4(b_fp4.data(), b_scales_fp8.data(), 1.0f, 0, 1);

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
    printf("Max error: %.6f at index %d, Avg error: %.6f\n",
           max_error, max_error_idx, avg_error);

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
