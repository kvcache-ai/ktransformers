#ifndef NVFP4_KERNEL_HPP
#define NVFP4_KERNEL_HPP

#include <immintrin.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

#include "amx_config.hpp"
#include "amx_utils.hpp"
#include "nvfp4_utils.hpp"
#include "utils.hpp"

namespace nvfp4 {

/**
 * NVFP4 × NVFP4 Matrix Multiplication Kernel
 *
 * Implements efficient CPU-side NVFP4 matrix multiplication using AVX512 lookup tables
 * Based on the plan in nvfp4.md
 *
 * Key features:
 * - E2M1 FP4 format with dual-level scaling (FP8 block + FP32 tensor)
 * - AVX512 lookup table multiplication
 * - Process 64 FP4 pairs per AVX512 register
 * - FP32 accumulation, BF16 output
 */

// ============================================================================
// NVFP4 Multiplication Lookup Tables (Two AVX512 registers)
// ============================================================================

namespace nvfp4_lut {

// E2M1 × E2M1 multiplication results (scaled by 4 for INT16 storage)
// 19 unique values: 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 3, 4, 4.5, 6, 8, 9, 12, 16, 18, 24, 36
// Stored as INT16 (×4): 0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144

// Positive results table (indices 0-18, padded to 32)
alignas(64) static const int16_t RESULT_TABLE_POS[32] = {
    0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64,      // 0-15
    72, 96, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0           // 16-31 (padding)
};

// Negative results table
alignas(64) static const int16_t RESULT_TABLE_NEG[32] = {
    0, -1, -2, -3, -4, -6, -8, -9, -12, -16, -18, -24, -32, -36, -48, -64,
    -72, -96, -144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// Index lookup table: (val_a[3 bits], val_b[3 bits]) → result_index[0-18]
// 8×8 = 64 entries for E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
alignas(64) static const uint8_t INDEX_LUT[64] = {
    // Row 0: 0 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 0, 0, 0, 0, 0, 0, 0,

    // Row 1: 0.5 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 1, 2, 3, 4, 5, 6, 7,      // →results: 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3

    // Row 2: 1 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 2, 4, 5, 6, 7, 8, 9,      // →results: 0, 0.5, 1, 1.5, 2, 3, 4, 6

    // Row 3: 1.5 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 3, 5, 10, 7, 11, 9, 12,   // →results: 0, 0.75, 1.5, 2.25, 3, 4.5, 6, 9

    // Row 4: 2 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 4, 6, 7, 8, 9, 13, 14,    // →results: 0, 1, 2, 3, 4, 6, 8, 12

    // Row 5: 3 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 5, 7, 11, 9, 12, 14, 17,  // →results: 0, 1.5, 3, 4.5, 6, 9, 12, 18

    // Row 6: 4 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 6, 8, 9, 13, 14, 15, 16,  // →results: 0, 2, 4, 6, 8, 12, 16, 24

    // Row 7: 6 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 7, 9, 12, 14, 17, 16, 18  // →results: 0, 3, 6, 9, 12, 18, 24, 36
};

// For debugging: actual float multiplication results
const float MUL_RESULTS[19] = {
    0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 2.25f, 3.0f,
    4.0f, 4.5f, 6.0f, 8.0f, 9.0f, 12.0f, 16.0f, 18.0f, 24.0f, 36.0f
};

} // namespace nvfp4_lut

// ============================================================================
// Core NVFP4 Multiplication Function (64 pairs using AVX512 lookup)
// ============================================================================

/**
 * Multiply 64 pairs of NVFP4 values using AVX512 lookup tables
 *
 * @param a_fp4: 64 packed FP4 values from matrix A (in __m512i register)
 * @param b_fp4: 64 packed FP4 values from matrix B (in __m512i register)
 * @param out_int16: Output array for 64 INT16 results (scaled by 4)
 */
inline void nvfp4_mul_64pairs_avx512(
    __m512i a_fp4,
    __m512i b_fp4,
    int16_t* out_int16  // 64 INT16 results
) {
    using namespace nvfp4_lut;

    // Masks for sign and value extraction
    const __m512i SIGN_MASK = _mm512_set1_epi8(0x08);
    const __m512i VALUE_MASK = _mm512_set1_epi8(0x07);

    // Step 1: Extract signs and compute result sign
    __m512i sign_a = _mm512_and_si512(a_fp4, SIGN_MASK);
    __m512i sign_b = _mm512_and_si512(b_fp4, SIGN_MASK);
    __m512i sign_result = _mm512_xor_si512(sign_a, sign_b);  // XOR for multiplication

    // Step 2: Extract value indices (3 bits each)
    __m512i val_a = _mm512_and_si512(a_fp4, VALUE_MASK);
    __m512i val_b = _mm512_and_si512(b_fp4, VALUE_MASK);

    // Step 3: Combine into 6-bit index: (val_a << 3) | val_b
    __m512i combined_idx = _mm512_or_si512(
        _mm512_slli_epi32(val_a, 3),  // Shift left by 3
        val_b
    );
    combined_idx = _mm512_and_si512(combined_idx, _mm512_set1_epi8(0x3F));  // Mask to 6 bits

    // Step 4: Lookup result indices using permutexvar
    __m512i lut = _mm512_load_si512((const __m512i*)INDEX_LUT);
    __m512i result_indices = _mm512_permutexvar_epi8(combined_idx, lut);

    // Step 5: Convert result to INT16 by looking up in result tables
    // We need to process this in chunks since we're going from bytes to int16

    // Split 64 bytes into two halves for processing
    alignas(64) uint8_t indices_array[64];
    alignas(64) uint8_t signs_array[64];
    _mm512_store_si512((__m512i*)indices_array, result_indices);
    _mm512_store_si512((__m512i*)signs_array, sign_result);

    // Lookup INT16 values
    for (int i = 0; i < 64; i++) {
        uint8_t idx = indices_array[i] & 0x1F;  // Clamp to valid range
        bool is_negative = (signs_array[i] & 0x08) != 0;

        if (idx < 19) {
            out_int16[i] = is_negative ? RESULT_TABLE_NEG[idx] : RESULT_TABLE_POS[idx];
        } else {
            out_int16[i] = 0;  // Should not happen with correct LUT
        }
    }
}

// ============================================================================
// NVFP4 Kernel Configuration
// ============================================================================

struct GemmKernelNVFP4 {
    using dt = void;
    using output_t = float;

    static constexpr int M_STEP = 32;
    static constexpr int N_STEP = 64;
    static constexpr int K_STEP = 64;  // 4 groups of 16

    static constexpr int BLOCK_SIZE = 16;  // FP4 group size
    static constexpr int GROUPS_PER_K_STEP = K_STEP / BLOCK_SIZE;  // = 4

    static void config() {
        // No special configuration needed for AVX512-only implementation
    }

    static int recommended_nth(int n) {
        return std::max(1, n / N_STEP);
    }

    static std::pair<int, int> split_range_n(int n, int ith, int nth) {
        int n_per_thread = (n + nth - 1) / nth;
        n_per_thread = (n_per_thread + N_STEP - 1) / N_STEP * N_STEP;
        int n_start = std::min(ith * n_per_thread, n);
        int n_end = std::min(n_start + n_per_thread, n);
        return {n_start, n_end};
    }
};

// ============================================================================
// BufferA - NVFP4 Activation (with BF16 quantization support)
// ============================================================================

template <typename K>
struct BufferANVFP4Impl {
    uint8_t* fp4_data;       // Packed FP4 (E2M1), 2 per byte
    uint8_t* block_scales;   // FP8 E4M3 block scales
    float tensor_scale;      // FP32 tensor scale

    int max_m, k;
    int block_count;

    static constexpr int M_STEP = K::M_STEP;
    static constexpr int K_STEP = K::K_STEP;
    static constexpr int BLOCK_SIZE = 16;

    static size_t required_size(int max_m, int k) {
        int total_elems = max_m * k;
        assert(total_elems % BLOCK_SIZE == 0);
        return total_elems / 2              // FP4 packed
             + total_elems / BLOCK_SIZE     // FP8 block scales
             + 64;                          // Alignment
    }

    BufferANVFP4Impl(int max_m, int k, void* ptr) : max_m(max_m), k(k) {
        assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
        assert(k % BLOCK_SIZE == 0);

        block_count = (max_m * k) / BLOCK_SIZE;
        set_data(ptr);
    }

    void set_data(void* ptr) {
        fp4_data = (uint8_t*)ptr;
        block_scales = fp4_data + (max_m * k) / 2;
        // tensor_scale stored in member variable
    }

    // Quantize from BF16 activation
    void from_bf16(int m, const ggml_bf16_t* src, int ith, int nth) {
        assert(m <= max_m);
        assert(ith == 0 && nth == 1);  // Single-threaded for now

        // Step 1: Compute tensor scale (global max)
        float global_max = compute_global_max_abs_bf16(m * k, src);
        tensor_scale = global_max / 448.0f;  // Map to FP8 E4M3 range

        // Step 2: Quantize each block
        for (int m_idx = 0; m_idx < m; m_idx++) {
            for (int k_block = 0; k_block < k; k_block += BLOCK_SIZE) {
                int block_idx = m_idx * (k / BLOCK_SIZE) + k_block / BLOCK_SIZE;

                // Find block max
                float block_max = 0.0f;
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    float val = bf16_to_f32_inline(src[m_idx * k + k_block + i]);
                    block_max = std::max(block_max, std::abs(val));
                }

                // Compute block scale
                float block_scale = block_max / 6.0f;  // Map to E2M1 max
                block_scales[block_idx] = float_to_fp8_e4m3(block_scale);

                // Quantize 16 values
                float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

                for (int i = 0; i < BLOCK_SIZE; i += 2) {
                    float val0 = bf16_to_f32_inline(src[m_idx * k + k_block + i]);
                    float val1 = bf16_to_f32_inline(src[m_idx * k + k_block + i + 1]);

                    float norm0 = val0 * scale_inv;
                    float norm1 = val1 * scale_inv;

                    uint8_t q0 = float_to_e2m1(norm0);
                    uint8_t q1 = float_to_e2m1(norm1);

                    int byte_idx = (m_idx * k + k_block + i) / 2;
                    fp4_data[byte_idx] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
                }
            }
        }
    }

    // Load from pre-quantized data (GPU quantized)
    void from_quantized(
        const uint8_t* pre_fp4,
        const uint8_t* pre_scales,
        float pre_tensor_scale,
        int m
    ) {
        int data_size = (m * k) / 2;
        int scale_count = (m * k) / BLOCK_SIZE;

        memcpy(fp4_data, pre_fp4, data_size);
        memcpy(block_scales, pre_scales, scale_count);
        tensor_scale = pre_tensor_scale;
    }

    // Get submatrix pointers
    uint8_t* get_fp4_data(int m_begin, int k_begin) {
        return fp4_data + (m_begin * k + k_begin) / 2;
    }

    uint8_t* get_block_scale(int m_begin, int k_begin) {
        int block_idx = (m_begin * k + k_begin) / BLOCK_SIZE;
        return block_scales + block_idx;
    }
};

// ============================================================================
// BufferB - NVFP4 Weight
// ============================================================================

template <typename K>
struct BufferBNVFP4Impl {
    uint8_t* fp4_data;
    uint8_t* block_scales;
    float tensor_scale;

    int n, k;
    int block_count;

    static constexpr int N_STEP = K::N_STEP;
    static constexpr int K_STEP = K::K_STEP;
    static constexpr int BLOCK_SIZE = 16;

    static size_t required_size(int n, int k) {
        int total_elems = n * k;
        return total_elems / 2
             + total_elems / BLOCK_SIZE
             + 64;
    }

    BufferBNVFP4Impl(int n, int k, void* ptr) : n(n), k(k) {
        assert(k % BLOCK_SIZE == 0);
        block_count = (n * k) / BLOCK_SIZE;
        set_data(ptr);
    }

    void set_data(void* ptr) {
        fp4_data = (uint8_t*)ptr;
        block_scales = fp4_data + (n * k) / 2;
    }

    // Load from raw NVFP4 weight data
    void from_raw_nvfp4(
        const uint8_t* raw_fp4,
        const uint8_t* raw_scales,
        float raw_tensor_scale,
        int ith, int nth
    ) {
        auto [n_start, n_end] = K::split_range_n(n, ith, nth);

        int row_bytes = k / 2;
        int row_blocks = k / BLOCK_SIZE;

        for (int n_idx = n_start; n_idx < n_end; n_idx++) {
            memcpy(fp4_data + n_idx * row_bytes,
                   raw_fp4 + n_idx * row_bytes,
                   row_bytes);
            memcpy(block_scales + n_idx * row_blocks,
                   raw_scales + n_idx * row_blocks,
                   row_blocks);
        }

        tensor_scale = raw_tensor_scale;
    }

    uint8_t* get_fp4_data(int n_begin, int k_begin) {
        return fp4_data + (n_begin * k + k_begin) / 2;
    }

    uint8_t* get_block_scale(int n_begin, int k_begin) {
        int block_idx = (n_begin * k + k_begin) / BLOCK_SIZE;
        return block_scales + block_idx;
    }
};

// ============================================================================
// BufferC - Output (FP32 accumulation, BF16 output)
// ============================================================================

template <typename K>
struct BufferCNVFP4Impl {
    float* c_fp32;       // FP32 accumulator
    ggml_bf16_t* c_bf16; // BF16 final output

    int max_m, n;

    static constexpr int M_STEP = K::M_STEP;
    static constexpr int N_STEP = K::N_STEP;

    static size_t required_size(int max_m, int n) {
        return max_m * n * sizeof(float)
             + max_m * n * sizeof(ggml_bf16_t)
             + 128;
    }

    BufferCNVFP4Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n) {
        set_data(ptr);
    }

    void set_data(void* ptr) {
        c_fp32 = (float*)ptr;
        c_bf16 = (ggml_bf16_t*)(c_fp32 + max_m * n);
    }

    void clear(int m) {
        memset(c_fp32, 0, m * n * sizeof(float));
    }

    // Accumulate INT16 results with scales
    void accumulate_scaled(
        int m_idx, int n_idx,
        const int16_t* int16_vals,
        int count,
        float scale_combined
    ) {
        float scale_factor = scale_combined / 4.0f;  // Divide by 4 (INT16 scaling)

        for (int i = 0; i < count; i++) {
            float val = int16_vals[i] * scale_factor;
            c_fp32[m_idx * n + n_idx + i] += val;
        }
    }

    // Finalize to BF16 output
    void to_bf16(int m) {
        for (int i = 0; i < m * n; i += 32) {
            int remaining = std::min(32, m * n - i);
            if (remaining >= 32) {
                __m512 f0 = _mm512_loadu_ps(c_fp32 + i);
                __m512 f1 = _mm512_loadu_ps(c_fp32 + i + 16);
                avx512_32xfp32_to_32xbf16(&f0, &f1, (__m512i*)(c_bf16 + i));
            } else {
                for (int j = 0; j < remaining; j++) {
                    c_bf16[i + j] = ggml_compute_fp32_to_bf16(c_fp32[i + j]);
                }
            }
        }
    }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
        to_bf16(m);
        memcpy(dst, c_bf16, m * n * sizeof(ggml_bf16_t));
    }
};

// ============================================================================
// NVFP4 Matrix Multiplication Kernel
// ============================================================================

/**
 * NVFP4 × NVFP4 matrix multiplication
 * A: [m × k] NVFP4
 * B: [n × k] NVFP4 (transposed view: B[k × n])
 * C: [m × n] FP32/BF16
 */
void nvfp4_matmul(
    int m, int n, int k,
    std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
    std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
    std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc,
    int ith, int nth
) {
    auto [n_start, n_end] = GemmKernelNVFP4::split_range_n(n, ith, nth);

    constexpr int BLOCK_SIZE = 16;
    constexpr int K_STEP = GemmKernelNVFP4::K_STEP;  // 64 = 4 blocks

    // Combined tensor scales
    float tensor_scale_combined = ba->tensor_scale * bb->tensor_scale;

    // Process each output element
    for (int m_idx = 0; m_idx < m; m_idx++) {
        for (int n_idx = n_start; n_idx < n_end; n_idx++) {
            float acc = 0.0f;

            // Process K dimension in steps of 64 (4 blocks)
            for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
                // Load 64 FP4 from A (row)
                uint8_t* a_fp4_ptr = ba->get_fp4_data(m_idx, k_begin);

                // Unpack to get individual FP4 values (64 values in 32 bytes)
                // Each byte contains 2 FP4 values
                alignas(64) uint8_t a_fp4[64];
                for (int i = 0; i < 32; i++) {
                    uint8_t packed = a_fp4_ptr[i];
                    a_fp4[i * 2] = packed & 0x0F;
                    a_fp4[i * 2 + 1] = (packed >> 4) & 0x0F;
                }

                // Load 64 FP4 from B (column)
                uint8_t* b_fp4_ptr = bb->get_fp4_data(n_idx, k_begin);
                alignas(64) uint8_t b_fp4[64];
                for (int i = 0; i < 32; i++) {
                    uint8_t packed = b_fp4_ptr[i];
                    b_fp4[i * 2] = packed & 0x0F;
                    b_fp4[i * 2 + 1] = (packed >> 4) & 0x0F;
                }

                // Perform lookup multiplication for 64 pairs
                alignas(64) int16_t results_int16[64];
                __m512i a_vec = _mm512_load_si512((const __m512i*)a_fp4);
                __m512i b_vec = _mm512_load_si512((const __m512i*)b_fp4);
                nvfp4_mul_64pairs_avx512(a_vec, b_vec, results_int16);

                // Apply scales and accumulate
                // Process each of the 4 blocks
                for (int blk = 0; blk < 4; blk++) {
                    int k_offset = k_begin + blk * BLOCK_SIZE;

                    // Get block scales
                    float scale_a = fp8_e4m3_to_float(ba->get_block_scale(m_idx, k_offset)[0]);
                    float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx, k_offset)[0]);
                    float scale_combined = scale_a * scale_b * tensor_scale_combined / 4.0f;

                    // Accumulate 16 results from this block
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        acc += results_int16[blk * BLOCK_SIZE + i] * scale_combined;
                    }
                }
            }

            bc->c_fp32[m_idx * n + n_idx] = acc;
        }
    }
}

} // namespace nvfp4

#endif  // NVFP4_KERNEL_HPP
