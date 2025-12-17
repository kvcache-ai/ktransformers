#ifndef NVFP4_UTILS_HPP
#define NVFP4_UTILS_HPP

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "amx_config.hpp"
#include "llama.cpp/ggml-impl.h"
#include "utils.hpp"

namespace amx {

/**
 * NVFP4 (NVIDIA FP4) format
 *
 * FP4 has 8 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
 * With sign bit, we have 16 values total (including negative)
 *
 * Storage format:
 * - Packed 4-bit values (2 FP4 values per byte)
 * - Group-based scales (one scale per group of 16 FP4 values)
 *
 * Multiplication approach:
 * - Use lookup table for FP4 x FP4 multiplication
 * - Results stored as INT16 (scaled by 4 to preserve 0.25 precision)
 * - Use _mm512_permutexvar_epi8 for table lookup
 */

// FP4 value encoding (4 bits: 1 sign + 3 mantissa/exponent)
// Positive values: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
// Negative values: 8=-0.0, 9=-0.5, 10=-1.0, 11=-1.5, 12=-2.0, 13=-3.0, 14=-4.0, 15=-6.0
constexpr float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Multiplication result lookup table (scaled by 4)
// Results are unique values from FP4 x FP4 multiplication
// 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 3, 4, 4.5, 6, 8, 9, 12, 16, 18, 24, 36
// Scaled by 4: 0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144
constexpr int16_t FP4_MUL_RESULTS_SCALED[19] = {
    0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144
};

// Mapping from (fp4_a, fp4_b) to result index (0-36, includes sign bit)
// This is a 8x8 table for positive values, sign is handled separately
// Result index 0-18: positive results
// Result index 19-36: negative results (19 = -results[0], 20 = -results[1], etc.)
constexpr uint8_t FP4_MUL_INDEX[8][8] = {
    // Row: fp4_a (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    // Col: fp4_b (0, 0.5, 1, 1.5, 2, 3, 4, 6)
    {0, 0, 0, 0, 0, 0, 0, 0},      // 0 * x = 0
    {0, 1, 2, 3, 4, 5, 6, 7},      // 0.5 * x
    {0, 2, 4, 5, 6, 7, 8, 9},      // 1 * x
    {0, 3, 5, 10, 7, 11, 9, 12},   // 1.5 * x
    {0, 4, 6, 7, 8, 9, 10, 11},    // 2 * x
    {0, 5, 7, 11, 9, 12, 13, 14},  // 3 * x
    {0, 6, 8, 9, 10, 13, 15, 16},  // 4 * x
    {0, 7, 9, 12, 11, 14, 16, 17}  // 6 * x
};

// Actual multiplication result values (unscaled, for reference)
const float FP4_MUL_TABLE[8][8] = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f},
    {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f},
    {0.0f, 0.75f, 1.5f, 2.25f, 3.0f, 4.5f, 6.0f, 9.0f},
    {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f},
    {0.0f, 1.5f, 3.0f, 4.5f, 6.0f, 9.0f, 12.0f, 18.0f},
    {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 12.0f, 16.0f, 24.0f},
    {0.0f, 3.0f, 6.0f, 9.0f, 12.0f, 18.0f, 24.0f, 36.0f}
};

// Convert float to FP4 (4-bit encoding)
inline uint8_t float_to_fp4(float x) {
    const float abs_x = fabsf(x);
    uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;

    // Find the closest FP4 value
    uint8_t best_idx = 0;
    float best_diff = abs_x;

    for (int i = 0; i < 8; i++) {
        float diff = fabsf(abs_x - FP4_LUT[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }

    return sign | best_idx;
}

// Convert FP4 to float
inline float fp4_to_float(uint8_t fp4) {
    return FP4_LUT[fp4 & 0x0F];
}

/**
 * AVX512 implementation of FP4 multiplication using lookup table
 *
 * Input: 64 pairs of FP4 values (packed in __m512i registers)
 * Output: 64 INT16 results (in two __m512i registers)
 *
 * Process:
 * 1. Extract sign bits and compute result sign
 * 2. Combine mantissa/exponent bits (6 bits total)
 * 3. Use _mm512_permutexvar_epi8 to lookup result index
 * 4. Lookup scaled INT16 result values
 * 5. Apply sign to results
 */

// Lookup table for result indices (6-bit input: 3-bit a + 3-bit b)
// Returns index into FP4_MUL_RESULTS_SCALED array
alignas(64) static const uint8_t FP4_MUL_INDEX_FLAT[64] = {
    // Flattened version of FP4_MUL_INDEX
    // Index = (fp4_a & 0x7) * 8 + (fp4_b & 0x7)
    0, 0, 0, 0, 0, 0, 0, 0,      // 0 * x
    0, 1, 2, 3, 4, 5, 6, 7,      // 0.5 * x
    0, 2, 4, 5, 6, 7, 8, 9,      // 1 * x
    0, 3, 5, 10, 7, 11, 9, 12,   // 1.5 * x
    0, 4, 6, 7, 8, 9, 10, 11,    // 2 * x
    0, 5, 7, 11, 9, 12, 13, 14,  // 3 * x
    0, 6, 8, 9, 10, 13, 15, 16,  // 4 * x
    0, 7, 9, 12, 11, 14, 16, 17  // 6 * x
};

// Lookup table for scaled INT16 results (extended to 64 bytes for alignment)
// First 19 entries are positive results, next 19 are negative (actually stored as negative values)
alignas(64) static const int16_t FP4_MUL_RESULTS_LUT[64] = {
    // Positive results (indices 0-18)
    0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144,
    // Padding
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/**
 * Multiply 64 pairs of FP4 values using AVX512
 *
 * @param a_lo Low 4 bits of 64 FP4 values (32 bytes, each byte contains 2 FP4 values)
 * @param a_hi High 4 bits if needed (for sign extension, usually same as a_lo for packed)
 * @param b_lo Low 4 bits of 64 FP4 values
 * @param b_hi High 4 bits of 64 FP4 values
 * @return Two __m512i registers containing 64 INT16 results (scaled by 4)
 *
 * Note: This is a simplified implementation. For actual use, you may need to
 * handle the packed format differently based on your data layout.
 */
inline void fp4_mul_64pairs_avx512(
    __m512i fp4_a,  // 64 FP4 values from activation (packed)
    __m512i fp4_b,  // 64 FP4 values from weight (packed)
    __m512i& result_lo,  // Output: low 32 INT16 results
    __m512i& result_hi   // Output: high 32 INT16 results
) {
    // Masks
    const __m512i mask_0x07 = _mm512_set1_epi8(0x07);  // Mask for 3-bit value (mantissa/exponent)
    const __m512i mask_0x08 = _mm512_set1_epi8(0x08);  // Mask for sign bit

    // Extract sign bits
    __m512i sign_a = _mm512_and_si512(fp4_a, mask_0x08);
    __m512i sign_b = _mm512_and_si512(fp4_b, mask_0x08);

    // XOR signs to get result sign (0 = positive, 0x08 = negative)
    __m512i sign_result = _mm512_xor_si512(sign_a, sign_b);

    // Extract magnitude (3-bit value)
    __m512i mag_a = _mm512_and_si512(fp4_a, mask_0x07);
    __m512i mag_b = _mm512_and_si512(fp4_b, mask_0x07);

    // Combine magnitudes into 6-bit index: (mag_a << 3) | mag_b
    __m512i mag_a_shifted = _mm512_slli_epi16(mag_a, 3);
    __m512i index_6bit = _mm512_or_si512(mag_a_shifted, mag_b);

    // Mask to 6 bits
    __m512i mask_0x3F = _mm512_set1_epi8(0x3F);
    index_6bit = _mm512_and_si512(index_6bit, mask_0x3F);

    // Load lookup table
    __m512i lut_indices = _mm512_load_si512((const __m512i*)FP4_MUL_INDEX_FLAT);

    // Lookup result indices (this gives us indices 0-18 for magnitude results)
    __m512i result_indices = _mm512_permutexvar_epi8(index_6bit, lut_indices);

    // Adjust indices based on sign: if negative, add 19 to get negative results
    // But since we store only positive values, we'll handle sign separately

    // Now we need to convert byte indices to INT16 values
    // This requires a second lookup into FP4_MUL_RESULTS_LUT

    // For simplicity, we'll use a gather operation or multiple lookups
    // This is a placeholder - actual implementation may need optimization

    // For now, let's create the result by loading from the lookup table
    // We need to expand result_indices from bytes to 16-bit indices

    // Split into low and high 32 bytes
    __m256i indices_lo = _mm512_extracti64x4_epi64(result_indices, 0);
    __m256i indices_hi = _mm512_extracti64x4_epi64(result_indices, 1);

    // Convert to 16-bit indices (zero-extend)
    __m512i indices_16_lo = _mm512_cvtepu8_epi16(indices_lo);
    __m512i indices_16_hi = _mm512_cvtepu8_epi16(indices_hi);

    // Load result values using gather (scale by sizeof(int16_t) = 2)
    result_lo = _mm512_i32gather_epi32(indices_16_lo, FP4_MUL_RESULTS_LUT, 2);
    result_hi = _mm512_i32gather_epi32(indices_16_hi, FP4_MUL_RESULTS_LUT, 2);

    // Apply sign: if sign_result has bit 0x08 set, negate the result
    // Convert sign to mask
    __m256i sign_lo = _mm512_extracti64x4_epi64(sign_result, 0);
    __m256i sign_hi = _mm512_extracti64x4_epi64(sign_result, 1);

    __m512i sign_16_lo = _mm512_cvtepi8_epi16(sign_lo);
    __m512i sign_16_hi = _mm512_cvtepi8_epi16(sign_hi);

    // Create mask: 0xFFFF if negative, 0x0000 if positive
    __m512i neg_mask_lo = _mm512_slli_epi16(sign_16_lo, 12);  // Move bit 3 to bit 15
    __m512i neg_mask_hi = _mm512_slli_epi16(sign_16_hi, 12);
    neg_mask_lo = _mm512_srai_epi16(neg_mask_lo, 15);  // Arithmetic shift to fill with sign
    neg_mask_hi = _mm512_srai_epi16(neg_mask_hi, 15);

    // Apply negation: result = (result ^ mask) - mask
    result_lo = _mm512_sub_epi16(_mm512_xor_si512(result_lo, neg_mask_lo), neg_mask_lo);
    result_hi = _mm512_sub_epi16(_mm512_xor_si512(result_hi, neg_mask_hi), neg_mask_hi);
}

/**
 * Quantization structure for NVFP4 format
 *
 * Group size: 16 FP4 values = 8 bytes (packed)
 * Each group has one scale factor
 */
struct blocks_aligned_nvfp4_ref {
    static constexpr int block_size = 16;  // 16 FP4 values per block
    static constexpr double bytes_per_element = double(sizeof(ggml_half) + double(block_size) / 2) / block_size;

    ggml_half* d;   // Scale factors
    uint8_t* qs;    // Quantized FP4 values (packed, 2 per byte)

    blocks_aligned_nvfp4_ref offset(size_t blck_cnt) const {
        blocks_aligned_nvfp4_ref re;
        re.d = &d[blck_cnt];
        re.qs = &qs[blck_cnt * block_size / 2];
        return re;
    }

    static size_t expected_data_size(int64_t k) {
        assert(k % block_size == 0);
        return (sizeof(ggml_half) + block_size / 2) * (k / block_size);
    }

    uint8_t* get_qs(int block_idx) {
        return offset_pointer(qs, block_idx * (block_size / 2));
    }

    // Quantize float array to NVFP4 format
    static blocks_aligned_nvfp4_ref quantize(const float* RESTRICT x, void* RESTRICT data, int64_t k) {
        assert(reinterpret_cast<intptr_t>(data) % 64 == 0);

        blocks_aligned_nvfp4_ref re;
        re.qs = reinterpret_cast<uint8_t*>(data);
        re.d = reinterpret_cast<ggml_half*>(offset_pointer(re.qs, k / 2));

        static const int qk = block_size;
        assert(k % qk == 0);

        const int nb = k / qk;

        for (int i = 0; i < nb; i++) {
            // Find max abs value in block
            float amax = 0.0f;
            for (int j = 0; j < qk; j++) {
                amax = MAX(amax, fabsf(x[i * qk + j]));
            }

            // Scale: map max value to largest FP4 value (6.0)
            const float d = amax / 6.0f;
            const float id = d ? 1.0f / d : 0.0f;

            re.d[i] = GGML_FP32_TO_FP16(d);

            // Quantize each value
            for (int j = 0; j < qk / 2; j++) {
                const float x0 = x[i * qk + j * 2 + 0] * id;
                const float x1 = x[i * qk + j * 2 + 1] * id;

                const uint8_t q0 = float_to_fp4(x0);
                const uint8_t q1 = float_to_fp4(x1);

                // Pack two FP4 values into one byte
                re.get_qs(i)[j] = q0 | (q1 << 4);
            }
        }

        return re;
    }

    // Dequantize NVFP4 to float array
    void dequantize(float* y, int64_t k) {
        static const int qk = block_size;
        assert(k % qk == 0);

        const int nb = k / qk;

        for (int i = 0; i < nb; i++) {
            const float d = GGML_FP16_TO_FP32(this->d[i]);

            for (int j = 0; j < qk / 2; j++) {
                const uint8_t packed = get_qs(i)[j];
                const uint8_t q0 = packed & 0x0F;
                const uint8_t q1 = packed >> 4;

                y[i * qk + j * 2 + 0] = fp4_to_float(q0) * d;
                y[i * qk + j * 2 + 1] = fp4_to_float(q1) * d;
            }
        }
    }
};

}  // namespace amx

#endif  // NVFP4_UTILS_HPP
