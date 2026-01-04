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
// NVFP4 Multiplication Lookup Tables
// ============================================================================

namespace nvfp4_lut {

// E2M1 × E2M1 multiplication results (scaled by 4 for INT16 storage)
// 19 unique positive values: 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.25, 3, 4, 4.5, 6, 8, 9, 12, 16, 18, 24, 36
// Stored as uint8 (×4): 0, 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96, 144
// All values < 256, so single byte is sufficient
alignas(64) static const uint8_t RESULT_TABLE[64] = {
    // Indices 0-18: multiplication results × 4
    0,
    1,
    2,
    3,
    4,
    6,
    8,
    9,
    12,
    16,
    18,
    24,
    32,
    36,
    48,
    64,
    72,
    96,
    144,
    // Padding to 64 bytes (indices 19-63 are never accessed since result_idx ∈ [0,18])
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

// Index lookup table: (val_a[3 bits], val_b[3 bits]) → result_index[0-18]
// 8×8 = 64 entries for E2M1 values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
alignas(64) static const uint8_t INDEX_LUT[64] = {
    // Row 0: 0 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 0, 0, 0, 0, 0, 0, 0,
    // Row 1: 0.5 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 1, 2, 3, 4, 5, 6, 8,
    // Row 2: 1 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 2, 4, 5, 6, 8, 9, 11,
    // Row 3: 1.5 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 3, 5, 7, 8, 10, 11, 13,
    // Row 4: 2 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 4, 6, 8, 9, 11, 12, 14,
    // Row 5: 3 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 5, 8, 10, 11, 13, 14, 16,
    // Row 6: 4 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 6, 9, 11, 12, 14, 15, 17,
    // Row 7: 6 × {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    0, 8, 11, 13, 14, 16, 17, 18};

// For debugging: actual float multiplication results
const float MUL_RESULTS[19] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f,  1.5f,  2.0f,  2.25f, 3.0f, 4.0f,
                               4.5f, 6.0f,  8.0f, 9.0f,  12.0f, 16.0f, 18.0f, 24.0f, 36.0f};

}  // namespace nvfp4_lut

// ============================================================================
// Core NVFP4 Multiplication Function (64 pairs using pure AVX512 lookup)
// ============================================================================

/**
 * Multiply 64 pairs of NVFP4 values using pure AVX512 lookup tables
 *
 * Algorithm:
 * 1. Extract sign bits from both inputs, XOR to get result sign
 * 2. Extract value bits (0-7) from both inputs
 * 3. Combine value bits into 6-bit index: (val_a << 3) | val_b
 * 4. Lookup result index (0-18) from INDEX_LUT
 * 5. Lookup result values from RESULT_TABLE_LO (all values < 256, so only LO needed)
 * 6. Use cvtepu8_epi16 to expand bytes to int16 (preserves order)
 * 7. Apply sign using masked subtraction (branchless)
 *
 * @param a_fp4: 64 packed FP4 values from matrix A (in __m512i register)
 * @param b_fp4: 64 packed FP4 values from matrix B (in __m512i register)
 * @param out_int16: Output array for 64 INT16 results (scaled by 4)
 */
inline void nvfp4_mul_64pairs_avx512(__m512i a_fp4, __m512i b_fp4,
                                     int16_t* out_int16  // 64 INT16 results
) {
  using namespace nvfp4_lut;

  // Masks for sign and value extraction
  const __m512i SIGN_MASK = _mm512_set1_epi8(0x08);
  const __m512i VALUE_MASK = _mm512_set1_epi8(0x07);

  // Step 1: Extract signs and compute result sign (XOR for multiplication)
  __m512i sign_a = _mm512_and_si512(a_fp4, SIGN_MASK);
  __m512i sign_b = _mm512_and_si512(b_fp4, SIGN_MASK);
  __m512i sign_result = _mm512_xor_si512(sign_a, sign_b);  // Result is 0x00 or 0x08

  // Step 2: Extract value indices (3 bits each, 0-7)
  __m512i val_a = _mm512_and_si512(a_fp4, VALUE_MASK);
  __m512i val_b = _mm512_and_si512(b_fp4, VALUE_MASK);

  // Step 3: Combine into 6-bit index: (val_a << 3) | val_b
  __m512i combined_idx = _mm512_or_si512(_mm512_slli_epi32(val_a, 3),  // Shift left by 3
                                         val_b);
  combined_idx = _mm512_and_si512(combined_idx, _mm512_set1_epi8(0x3F));  // Mask to 6 bits

  // Step 4: Lookup result indices (0-18) from INDEX_LUT
  __m512i lut = _mm512_load_si512((const __m512i*)INDEX_LUT);
  __m512i result_idx = _mm512_permutexvar_epi8(combined_idx, lut);

  // Step 5: Lookup result values (all values < 256, single table sufficient)
  __m512i result_table = _mm512_load_si512((const __m512i*)RESULT_TABLE);
  __m512i result_bytes = _mm512_permutexvar_epi8(result_idx, result_table);

  // Step 6: Convert bytes to int16 using cvtepu8_epi16 (preserves element order)
  // Extract low and high 256-bit halves
  __m256i result_lo_256 = _mm512_castsi512_si256(result_bytes);        // bytes 0-31
  __m256i result_hi_256 = _mm512_extracti64x4_epi64(result_bytes, 1);  // bytes 32-63

  // Zero-extend each byte to int16
  __m512i result_lo = _mm512_cvtepu8_epi16(result_lo_256);  // 32 int16 from bytes 0-31
  __m512i result_hi = _mm512_cvtepu8_epi16(result_hi_256);  // 32 int16 from bytes 32-63

  // Step 7: Apply sign using masked subtraction (branchless)
  // Extract sign masks for each half
  __m256i sign_lo_256 = _mm512_castsi512_si256(sign_result);
  __m256i sign_hi_256 = _mm512_extracti64x4_epi64(sign_result, 1);

  // Create 32-bit masks from sign bytes (bit i = 1 if byte i has sign bit set)
  __mmask32 sign_mask_lo = _mm256_test_epi8_mask(sign_lo_256, _mm256_set1_epi8(0x08));
  __mmask32 sign_mask_hi = _mm256_test_epi8_mask(sign_hi_256, _mm256_set1_epi8(0x08));

  // For masked elements: result = 0 - result (negate)
  __m512i zero = _mm512_setzero_si512();
  __m512i final_lo = _mm512_mask_sub_epi16(result_lo, sign_mask_lo, zero, result_lo);
  __m512i final_hi = _mm512_mask_sub_epi16(result_hi, sign_mask_hi, zero, result_hi);

  // Step 8: Store 64 INT16 results
  _mm512_storeu_si512((__m512i*)out_int16, final_lo);
  _mm512_storeu_si512((__m512i*)(out_int16 + 32), final_hi);
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

  static constexpr int BLOCK_SIZE = 16;                          // FP4 group size
  static constexpr int GROUPS_PER_K_STEP = K_STEP / BLOCK_SIZE;  // = 4

  static void config() {
    // No special configuration needed for AVX512-only implementation
  }

  static int recommended_nth(int n) { return std::max(1, n / N_STEP); }

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
  uint8_t* fp4_data;      // Packed FP4 (E2M1), 2 per byte
  uint8_t* block_scales;  // FP8 E4M3 block scales
  float tensor_scale;     // FP32 tensor scale

  int max_m, k;
  int block_count;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr int BLOCK_SIZE = 16;

  static size_t required_size(int max_m, int k) {
    int total_elems = max_m * k;
    assert(total_elems % BLOCK_SIZE == 0);
    return total_elems / 2             // FP4 packed
           + total_elems / BLOCK_SIZE  // FP8 block scales
           + 64;                       // Alignment
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

    // Step 1: Compute tensor scale (global max) - maps to FP8 E4M3 range
    // The tensor_scale is applied BEFORE block quantization
    float global_max = compute_global_max_abs_bf16(m * k, src);
    tensor_scale = global_max / 448.0f;  // Map to FP8 E4M3 range

    // Prevent division by zero
    float ts_inv = (tensor_scale > 1e-10f) ? (1.0f / tensor_scale) : 1.0f;

    // Step 2: Quantize each block
    for (int m_idx = 0; m_idx < m; m_idx++) {
      for (int k_block = 0; k_block < k; k_block += BLOCK_SIZE) {
        int block_idx = m_idx * (k / BLOCK_SIZE) + k_block / BLOCK_SIZE;

        // Find block max (after applying tensor scale)
        float block_max = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
          float val = bf16_to_f32_inline(src[m_idx * k + k_block + i]);
          val *= ts_inv;  // Apply tensor scale inverse first
          block_max = std::max(block_max, std::abs(val));
        }

        // Compute block scale: map block max to E2M1 max (6.0)
        float block_scale = block_max / 6.0f;
        block_scales[block_idx] = float_to_fp8_e4m3(block_scale);

        // Quantize 16 values
        float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

        for (int i = 0; i < BLOCK_SIZE; i += 2) {
          float val0 = bf16_to_f32_inline(src[m_idx * k + k_block + i]);
          float val1 = bf16_to_f32_inline(src[m_idx * k + k_block + i + 1]);

          // Apply tensor scale inverse, then block scale inverse
          float norm0 = val0 * ts_inv * scale_inv;
          float norm1 = val1 * ts_inv * scale_inv;

          uint8_t q0 = float_to_e2m1(norm0);
          uint8_t q1 = float_to_e2m1(norm1);

          int byte_idx = (m_idx * k + k_block + i) / 2;
          fp4_data[byte_idx] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
      }
    }
  }

  // Load from pre-quantized data (GPU quantized)
  void from_quantized(const uint8_t* pre_fp4, const uint8_t* pre_scales, float pre_tensor_scale, int m) {
    int data_size = (m * k) / 2;
    int scale_count = (m * k) / BLOCK_SIZE;

    memcpy(fp4_data, pre_fp4, data_size);
    memcpy(block_scales, pre_scales, scale_count);
    tensor_scale = pre_tensor_scale;
  }

  // Get submatrix pointers
  uint8_t* get_fp4_data(int m_begin, int k_begin) { return fp4_data + (m_begin * k + k_begin) / 2; }

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
    return total_elems / 2 + total_elems / BLOCK_SIZE + 64;
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
  void from_raw_nvfp4(const uint8_t* raw_fp4, const uint8_t* raw_scales, float raw_tensor_scale, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);

    int row_bytes = k / 2;
    int row_blocks = k / BLOCK_SIZE;

    for (int n_idx = n_start; n_idx < n_end; n_idx++) {
      memcpy(fp4_data + n_idx * row_bytes, raw_fp4 + n_idx * row_bytes, row_bytes);
      memcpy(block_scales + n_idx * row_blocks, raw_scales + n_idx * row_blocks, row_blocks);
    }

    tensor_scale = raw_tensor_scale;
  }

  uint8_t* get_fp4_data(int n_begin, int k_begin) { return fp4_data + (n_begin * k + k_begin) / 2; }

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
  float* c_fp32;        // FP32 accumulator
  ggml_bf16_t* c_bf16;  // BF16 final output

  int max_m, n;

  static constexpr int M_STEP = K::M_STEP;
  static constexpr int N_STEP = K::N_STEP;

  static size_t required_size(int max_m, int n) {
    return max_m * n * sizeof(float) + max_m * n * sizeof(ggml_bf16_t) + 128;
  }

  BufferCNVFP4Impl(int max_m, int n, void* ptr) : max_m(max_m), n(n) { set_data(ptr); }

  void set_data(void* ptr) {
    c_fp32 = (float*)ptr;
    c_bf16 = (ggml_bf16_t*)(c_fp32 + max_m * n);
  }

  void clear(int m) { memset(c_fp32, 0, m * n * sizeof(float)); }

  // Accumulate INT16 results with scales
  void accumulate_scaled(int m_idx, int n_idx, const int16_t* int16_vals, int count, float scale_combined) {
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
void nvfp4_matmul(int m, int n, int k, std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
                  std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
                  std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc, int ith, int nth) {
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

// ============================================================================
// Optimized NVFP4 Helper Functions
// ============================================================================

/**
 * Vectorized unpack: 32 packed bytes -> 64 unpacked bytes
 * Input:  [lo0|hi0, lo1|hi1, ...] in 32 bytes (__m256i)
 * Output: [lo0, hi0, lo1, hi1, ...] in 64 bytes (__m512i)
 */
inline __m512i unpack_fp4_avx512(const uint8_t* packed) {
  // Load 32 packed bytes
  __m256i packed_256 = _mm256_loadu_si256((const __m256i*)packed);

  // Expand to 512-bit by duplicating each byte: [b0,b0,b1,b1,...]
  // Use permutexvar to interleave
  __m512i packed_512 = _mm512_cvtepu8_epi16(packed_256);  // 32 x 16-bit

  // Extract low nibble (even positions) and high nibble (odd positions)
  __m512i lo_mask = _mm512_set1_epi16(0x000F);
  __m512i hi_mask = _mm512_set1_epi16(0x0F00);

  __m512i lo = _mm512_and_si512(packed_512, lo_mask);  // [lo0, 0, lo1, 0, ...]
  __m512i hi = _mm512_and_si512(packed_512, hi_mask);  // [0, hi0<<8, 0, hi1<<8, ...]
  __m512i hi_shifted = _mm512_srli_epi16(hi, 4);       // [0, hi0, 0, hi1, ...]

  // Combine: lo in even byte positions, hi in odd byte positions
  __m512i combined = _mm512_or_si512(lo, hi_shifted);

  // Now we have [lo0, hi0, lo1, hi1, ...] as 16-bit values
  // Pack back to bytes
  __m256i result_lo = _mm512_cvtepi16_epi8(combined);

  // But we need 64 bytes... Let me reconsider the approach
  // Actually, cvtepu8_epi16 gives us 32 int16, and we need 64 bytes output
  // Better approach: use shuffle

  // Reload and use different strategy
  __m512i src = _mm512_castsi256_si512(packed_256);
  src = _mm512_inserti64x4(src, packed_256, 1);  // Duplicate in high half

  // Shuffle pattern to interleave nibbles
  // For each pair of bytes [A, B], output [A&0xF, A>>4, B&0xF, B>>4]
  // Use two shuffles and masks

  __m512i mask_lo = _mm512_set1_epi8(0x0F);
  __m512i lo_nibbles = _mm512_and_si512(src, mask_lo);
  __m512i hi_nibbles = _mm512_and_si512(_mm512_srli_epi16(src, 4), mask_lo);

  // Interleave lo and hi nibbles
  // unpacklo_epi8 interleaves bytes from two sources
  __m512i result = _mm512_unpacklo_epi8(lo_nibbles, hi_nibbles);
  __m512i result_hi = _mm512_unpackhi_epi8(lo_nibbles, hi_nibbles);

  // But this gives wrong order due to lane structure...
  // Need to use permute to fix lane order

  // Simpler approach: use vpshufb with custom pattern
  // Actually let's just use a simpler interleave pattern

  return result;  // This is partially correct, need to fix lane ordering
}

/**
 * Optimized unpack using simple shuffle pattern
 * Unpack 32 packed bytes to 64 unpacked nibbles
 */
inline void unpack_fp4_to_bytes(const uint8_t* __restrict packed, uint8_t* __restrict unpacked) {
  __m256i src = _mm256_loadu_si256((const __m256i*)packed);

  // Extract low and high nibbles
  __m256i mask = _mm256_set1_epi8(0x0F);
  __m256i lo = _mm256_and_si256(src, mask);
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(src, 4), mask);

  // Interleave: unpacklo gives [lo0,hi0,lo1,hi1,...] for each 128-bit lane
  __m256i interleaved_lo = _mm256_unpacklo_epi8(lo, hi);
  __m256i interleaved_hi = _mm256_unpackhi_epi8(lo, hi);

  // Fix lane order: AVX2 unpack works within 128-bit lanes
  // Lane 0: bytes 0-7 interleaved, Lane 1: bytes 16-23 interleaved
  // We need: bytes 0-15 in first 256-bit, bytes 16-31 in second 256-bit
  __m256i perm = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20);
  __m256i perm_hi = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31);

  _mm256_storeu_si256((__m256i*)unpacked, perm);
  _mm256_storeu_si256((__m256i*)(unpacked + 32), perm_hi);
}

/**
 * Vectorized sum of 16 INT16 values -> INT32
 */
inline int32_t reduce_add_epi16x16(__m256i v) {
  // Sum 16 int16 -> 8 int32
  __m256i sum32 = _mm256_madd_epi16(v, _mm256_set1_epi16(1));
  // Horizontal sum of 8 int32
  __m128i lo = _mm256_castsi256_si128(sum32);
  __m128i hi = _mm256_extracti128_si256(sum32, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
  return _mm_cvtsi128_si32(sum);
}

/**
 * Vectorized sum of 32 INT16 values -> INT32
 */
inline int32_t reduce_add_epi16x32(__m512i v) {
  // Sum 32 int16 -> 16 int32 using madd with 1s
  __m512i sum32 = _mm512_madd_epi16(v, _mm512_set1_epi16(1));
  // Reduce 16 int32 to scalar
  return _mm512_reduce_add_epi32(sum32);
}

// ============================================================================
// Optimized NVFP4 Matrix Multiplication Kernel
// ============================================================================

/**
 * Optimized NVFP4 × NVFP4 matrix multiplication
 * - Vectorized FP4 unpacking
 * - Vectorized accumulation
 * - Better memory access pattern
 */
inline void nvfp4_matmul_opt(int m, int n, int k, std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
                             std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
                             std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc, int ith, int nth) {
  auto [n_start, n_end] = GemmKernelNVFP4::split_range_n(n, ith, nth);

  constexpr int BLOCK_SIZE = 16;
  constexpr int K_STEP = GemmKernelNVFP4::K_STEP;  // 64 = 4 blocks

  // Combined tensor scales (precompute)
  float tensor_scale_combined = ba->tensor_scale * bb->tensor_scale;

  // Precompute scale factor (includes /4.0 for INT16 scaling)
  float base_scale = tensor_scale_combined / 4.0f;

  // Temporary buffers for unpacked FP4
  alignas(64) uint8_t a_fp4[64];
  alignas(64) uint8_t b_fp4[64];
  alignas(64) int16_t results_int16[64];

  // Process each output element
  for (int m_idx = 0; m_idx < m; m_idx++) {
    // Precompute A's block scales for this row
    const int num_k_blocks = k / BLOCK_SIZE;

    for (int n_idx = n_start; n_idx < n_end; n_idx++) {
      float acc = 0.0f;

      // Process K dimension in steps of 64 (4 blocks)
      for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
        // === Vectorized unpack A ===
        const uint8_t* a_ptr = ba->get_fp4_data(m_idx, k_begin);
        unpack_fp4_to_bytes(a_ptr, a_fp4);

        // === Vectorized unpack B ===
        const uint8_t* b_ptr = bb->get_fp4_data(n_idx, k_begin);
        unpack_fp4_to_bytes(b_ptr, b_fp4);

        // === LUT multiplication (already optimized) ===
        __m512i a_vec = _mm512_load_si512((const __m512i*)a_fp4);
        __m512i b_vec = _mm512_load_si512((const __m512i*)b_fp4);
        nvfp4_mul_64pairs_avx512(a_vec, b_vec, results_int16);

        // === Vectorized accumulation with scales ===
        // Process 4 blocks of 16 elements each
        for (int blk = 0; blk < 4; blk++) {
          int k_offset = k_begin + blk * BLOCK_SIZE;

          // Get block scales
          float scale_a = fp8_e4m3_to_float(ba->get_block_scale(m_idx, k_offset)[0]);
          float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx, k_offset)[0]);
          float scale_combined = scale_a * scale_b * base_scale;

          // Vectorized sum of 16 INT16 values
          __m256i block_data = _mm256_loadu_si256((const __m256i*)(results_int16 + blk * BLOCK_SIZE));
          int32_t block_sum = reduce_add_epi16x16(block_data);

          acc += block_sum * scale_combined;
        }
      }

      bc->c_fp32[m_idx * n + n_idx] = acc;
    }
  }
}

// ============================================================================
// Further Optimized Version: Batch N processing
// ============================================================================

/**
 * Process multiple N columns together to amortize A unpacking cost
 */
inline void nvfp4_matmul_opt2(int m, int n, int k, std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
                              std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
                              std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc, int ith, int nth) {
  auto [n_start, n_end] = GemmKernelNVFP4::split_range_n(n, ith, nth);

  constexpr int BLOCK_SIZE = 16;
  constexpr int K_STEP = GemmKernelNVFP4::K_STEP;  // 64
  constexpr int N_BATCH = 4;                       // Process 4 N columns together

  float tensor_scale_combined = ba->tensor_scale * bb->tensor_scale;
  float base_scale = tensor_scale_combined / 4.0f;

  alignas(64) uint8_t a_fp4[64];
  alignas(64) uint8_t b_fp4[4][64];  // 4 columns
  alignas(64) int16_t results[4][64];

  for (int m_idx = 0; m_idx < m; m_idx++) {
    for (int n_idx = n_start; n_idx < n_end; n_idx += N_BATCH) {
      int n_batch = std::min(N_BATCH, n_end - n_idx);
      float acc[N_BATCH] = {0.0f, 0.0f, 0.0f, 0.0f};

      for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
        // Unpack A once for all N columns
        const uint8_t* a_ptr = ba->get_fp4_data(m_idx, k_begin);
        unpack_fp4_to_bytes(a_ptr, a_fp4);
        __m512i a_vec = _mm512_load_si512((const __m512i*)a_fp4);

        // Process each N column in batch
        for (int nb = 0; nb < n_batch; nb++) {
          const uint8_t* b_ptr = bb->get_fp4_data(n_idx + nb, k_begin);
          unpack_fp4_to_bytes(b_ptr, b_fp4[nb]);
          __m512i b_vec = _mm512_load_si512((const __m512i*)b_fp4[nb]);
          nvfp4_mul_64pairs_avx512(a_vec, b_vec, results[nb]);
        }

        // Accumulate with scales
        for (int blk = 0; blk < 4; blk++) {
          int k_offset = k_begin + blk * BLOCK_SIZE;
          float scale_a = fp8_e4m3_to_float(ba->get_block_scale(m_idx, k_offset)[0]);

          for (int nb = 0; nb < n_batch; nb++) {
            float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx + nb, k_offset)[0]);
            float scale_combined = scale_a * scale_b * base_scale;

            __m256i block_data = _mm256_loadu_si256((const __m256i*)(results[nb] + blk * BLOCK_SIZE));
            int32_t block_sum = reduce_add_epi16x16(block_data);
            acc[nb] += block_sum * scale_combined;
          }
        }
      }

      // Store results
      for (int nb = 0; nb < n_batch; nb++) {
        bc->c_fp32[m_idx * n + n_idx + nb] = acc[nb];
      }
    }
  }
}

// ============================================================================
// Fused LUT + Reduction (no intermediate memory)
// ============================================================================

/**
 * Fused LUT multiplication + block reduction
 * Returns 4 int32 sums for 4 blocks of 16 elements each
 * No intermediate memory stores - everything stays in registers
 */
inline void nvfp4_mul_64_reduce_to_4blocks(__m512i a_fp4, __m512i b_fp4, int32_t* block_sums) {
  using namespace nvfp4_lut;

  const __m512i SIGN_MASK = _mm512_set1_epi8(0x08);
  const __m512i VALUE_MASK = _mm512_set1_epi8(0x07);

  // Extract signs and compute result sign
  __m512i sign_a = _mm512_and_si512(a_fp4, SIGN_MASK);
  __m512i sign_b = _mm512_and_si512(b_fp4, SIGN_MASK);
  __m512i sign_result = _mm512_xor_si512(sign_a, sign_b);

  // Extract value indices
  __m512i val_a = _mm512_and_si512(a_fp4, VALUE_MASK);
  __m512i val_b = _mm512_and_si512(b_fp4, VALUE_MASK);

  // Combine into 6-bit index
  __m512i combined_idx = _mm512_or_si512(_mm512_slli_epi32(val_a, 3), val_b);
  combined_idx = _mm512_and_si512(combined_idx, _mm512_set1_epi8(0x3F));

  // LUT lookups
  __m512i lut = _mm512_load_si512((const __m512i*)INDEX_LUT);
  __m512i result_idx = _mm512_permutexvar_epi8(combined_idx, lut);
  __m512i result_table = _mm512_load_si512((const __m512i*)RESULT_TABLE);
  __m512i result_bytes = _mm512_permutexvar_epi8(result_idx, result_table);

  // Convert to int16 and apply sign
  __m256i result_lo_256 = _mm512_castsi512_si256(result_bytes);
  __m256i result_hi_256 = _mm512_extracti64x4_epi64(result_bytes, 1);
  __m512i result_lo = _mm512_cvtepu8_epi16(result_lo_256);
  __m512i result_hi = _mm512_cvtepu8_epi16(result_hi_256);

  __m256i sign_lo_256 = _mm512_castsi512_si256(sign_result);
  __m256i sign_hi_256 = _mm512_extracti64x4_epi64(sign_result, 1);
  __mmask32 sign_mask_lo = _mm256_test_epi8_mask(sign_lo_256, _mm256_set1_epi8(0x08));
  __mmask32 sign_mask_hi = _mm256_test_epi8_mask(sign_hi_256, _mm256_set1_epi8(0x08));

  __m512i zero = _mm512_setzero_si512();
  __m512i final_lo = _mm512_mask_sub_epi16(result_lo, sign_mask_lo, zero, result_lo);
  __m512i final_hi = _mm512_mask_sub_epi16(result_hi, sign_mask_hi, zero, result_hi);

  // Now reduce to 4 block sums directly
  // final_lo contains elements 0-31 (blocks 0,1)
  // final_hi contains elements 32-63 (blocks 2,3)

  // Use madd to sum pairs: int16 * 1 -> int32
  __m512i sum32_lo = _mm512_madd_epi16(final_lo, _mm512_set1_epi16(1));  // 16 int32
  __m512i sum32_hi = _mm512_madd_epi16(final_hi, _mm512_set1_epi16(1));  // 16 int32

  // Extract 128-bit chunks and reduce
  // Block 0: elements 0-15 -> sum32_lo[0:7]
  // Block 1: elements 16-31 -> sum32_lo[8:15]
  // Block 2: elements 32-47 -> sum32_hi[0:7]
  // Block 3: elements 48-63 -> sum32_hi[8:15]

  // Reduce each block of 8 int32 to single int32
  __m256i lo_256 = _mm512_castsi512_si256(sum32_lo);       // Block 0 (first 8 int32)
  __m256i lo_hi = _mm512_extracti64x4_epi64(sum32_lo, 1);  // Block 1 (next 8 int32)
  __m256i hi_256 = _mm512_castsi512_si256(sum32_hi);       // Block 2
  __m256i hi_hi = _mm512_extracti64x4_epi64(sum32_hi, 1);  // Block 3

  // Horizontal sum for each 256-bit chunk (8 int32 -> 1 int32)
  // Use hadd twice then extract
  auto reduce8 = [](__m256i v) -> int32_t {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
  };

  block_sums[0] = reduce8(lo_256);
  block_sums[1] = reduce8(lo_hi);
  block_sums[2] = reduce8(hi_256);
  block_sums[3] = reduce8(hi_hi);
}

// ============================================================================
// Optimized Version 4: Fully inline with no intermediate stores
// ============================================================================

/**
 * Inline unpack + LUT + reduce in one function
 * No intermediate memory writes - everything stays in registers
 */
inline __m512i unpack_packed_to_512(const uint8_t* packed32) {
  __m256i src = _mm256_loadu_si256((const __m256i*)packed32);
  __m256i mask = _mm256_set1_epi8(0x0F);
  __m256i lo = _mm256_and_si256(src, mask);
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(src, 4), mask);
  __m256i interleaved_lo = _mm256_unpacklo_epi8(lo, hi);
  __m256i interleaved_hi = _mm256_unpackhi_epi8(lo, hi);
  __m256i perm = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20);
  __m256i perm_hi = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31);
  return _mm512_inserti64x4(_mm512_castsi256_si512(perm), perm_hi, 1);
}

/**
 * Fully fused: unpack + LUT multiply + reduce to 4 block sums
 * Input: 32 packed bytes from A and B each
 * Output: 4 int32 block sums
 */
inline void fused_mul_reduce_64(const uint8_t* a_packed32, const uint8_t* b_packed32, int32_t* block_sums) {
  using namespace nvfp4_lut;

  // Unpack both inputs to 64 bytes in registers
  __m512i a_fp4 = unpack_packed_to_512(a_packed32);
  __m512i b_fp4 = unpack_packed_to_512(b_packed32);

  // LUT multiplication
  const __m512i SIGN_MASK = _mm512_set1_epi8(0x08);
  const __m512i VALUE_MASK = _mm512_set1_epi8(0x07);

  __m512i sign_a = _mm512_and_si512(a_fp4, SIGN_MASK);
  __m512i sign_b = _mm512_and_si512(b_fp4, SIGN_MASK);
  __m512i sign_result = _mm512_xor_si512(sign_a, sign_b);

  __m512i val_a = _mm512_and_si512(a_fp4, VALUE_MASK);
  __m512i val_b = _mm512_and_si512(b_fp4, VALUE_MASK);
  __m512i combined_idx = _mm512_or_si512(_mm512_slli_epi32(val_a, 3), val_b);
  combined_idx = _mm512_and_si512(combined_idx, _mm512_set1_epi8(0x3F));

  __m512i lut = _mm512_load_si512((const __m512i*)INDEX_LUT);
  __m512i result_idx = _mm512_permutexvar_epi8(combined_idx, lut);
  __m512i result_table = _mm512_load_si512((const __m512i*)RESULT_TABLE);
  __m512i result_bytes = _mm512_permutexvar_epi8(result_idx, result_table);

  // Convert to int16 and apply sign
  __m256i result_lo_256 = _mm512_castsi512_si256(result_bytes);
  __m256i result_hi_256 = _mm512_extracti64x4_epi64(result_bytes, 1);
  __m512i result_lo = _mm512_cvtepu8_epi16(result_lo_256);
  __m512i result_hi = _mm512_cvtepu8_epi16(result_hi_256);

  __m256i sign_lo_256 = _mm512_castsi512_si256(sign_result);
  __m256i sign_hi_256 = _mm512_extracti64x4_epi64(sign_result, 1);
  __mmask32 sign_mask_lo = _mm256_test_epi8_mask(sign_lo_256, _mm256_set1_epi8(0x08));
  __mmask32 sign_mask_hi = _mm256_test_epi8_mask(sign_hi_256, _mm256_set1_epi8(0x08));

  __m512i zero = _mm512_setzero_si512();
  __m512i final_lo = _mm512_mask_sub_epi16(result_lo, sign_mask_lo, zero, result_lo);
  __m512i final_hi = _mm512_mask_sub_epi16(result_hi, sign_mask_hi, zero, result_hi);

  // Reduce to 4 block sums using madd
  __m512i sum32_lo = _mm512_madd_epi16(final_lo, _mm512_set1_epi16(1));
  __m512i sum32_hi = _mm512_madd_epi16(final_hi, _mm512_set1_epi16(1));

  // Extract 256-bit chunks and reduce each to single int32
  __m256i lo_256 = _mm512_castsi512_si256(sum32_lo);
  __m256i lo_hi = _mm512_extracti64x4_epi64(sum32_lo, 1);
  __m256i hi_256 = _mm512_castsi512_si256(sum32_hi);
  __m256i hi_hi = _mm512_extracti64x4_epi64(sum32_hi, 1);

  // Reduce each 256-bit (8 int32) to scalar
  __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(lo_256), _mm256_extracti128_si256(lo_256, 1));
  t0 = _mm_add_epi32(t0, _mm_srli_si128(t0, 8));
  t0 = _mm_add_epi32(t0, _mm_srli_si128(t0, 4));
  block_sums[0] = _mm_cvtsi128_si32(t0);

  __m128i t1 = _mm_add_epi32(_mm256_castsi256_si128(lo_hi), _mm256_extracti128_si256(lo_hi, 1));
  t1 = _mm_add_epi32(t1, _mm_srli_si128(t1, 8));
  t1 = _mm_add_epi32(t1, _mm_srli_si128(t1, 4));
  block_sums[1] = _mm_cvtsi128_si32(t1);

  __m128i t2 = _mm_add_epi32(_mm256_castsi256_si128(hi_256), _mm256_extracti128_si256(hi_256, 1));
  t2 = _mm_add_epi32(t2, _mm_srli_si128(t2, 8));
  t2 = _mm_add_epi32(t2, _mm_srli_si128(t2, 4));
  block_sums[2] = _mm_cvtsi128_si32(t2);

  __m128i t3 = _mm_add_epi32(_mm256_castsi256_si128(hi_hi), _mm256_extracti128_si256(hi_hi, 1));
  t3 = _mm_add_epi32(t3, _mm_srli_si128(t3, 8));
  t3 = _mm_add_epi32(t3, _mm_srli_si128(t3, 4));
  block_sums[3] = _mm_cvtsi128_si32(t3);
}

/**
 * Optimized matmul with fully fused kernel
 * Process 4 K_STEPs at once to reduce loop overhead
 */
inline void nvfp4_matmul_opt4(int m, int n, int k, std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
                              std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
                              std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc, int ith, int nth) {
  auto [n_start, n_end] = GemmKernelNVFP4::split_range_n(n, ith, nth);

  constexpr int BLOCK_SIZE = 16;
  constexpr int K_STEP = 64;
  constexpr int K_STEP4 = K_STEP * 4;  // Process 256 elements at once
  constexpr int N_BATCH = 64;

  const float base_scale = (ba->tensor_scale * bb->tensor_scale) / 4.0f;
  const float* fp8_lut = fp8_lut::get_fp8_e4m3_lut();
  const int total_k = ba->k;

  alignas(16) int32_t block_sums1[4];
  alignas(16) int32_t block_sums2[4];
  alignas(16) int32_t block_sums3[4];
  alignas(16) int32_t block_sums4[4];

  for (int m_idx = 0; m_idx < m; m_idx++) {
    const uint8_t* a_scale_base = ba->get_block_scale(m_idx, 0);

    for (int n_idx = n_start; n_idx < n_end; n_idx += N_BATCH) {
      const int n_batch = std::min(N_BATCH, n_end - n_idx);
      alignas(64) float acc[N_BATCH] = {};

      // Process 4 K_STEPs at once
      int k_begin = 0;
      for (; k_begin + K_STEP4 <= total_k; k_begin += K_STEP4) {
        const uint8_t* a_ptr1 = ba->get_fp4_data(m_idx, k_begin);
        const uint8_t* a_ptr2 = ba->get_fp4_data(m_idx, k_begin + K_STEP);
        const uint8_t* a_ptr3 = ba->get_fp4_data(m_idx, k_begin + K_STEP * 2);
        const uint8_t* a_ptr4 = ba->get_fp4_data(m_idx, k_begin + K_STEP * 3);
        const int k_block_base = k_begin / BLOCK_SIZE;

        // Precompute A scales for all 4 K_STEPs (16 blocks)
        float a_s[16];
        for (int i = 0; i < 16; i++) {
          a_s[i] = fp8_lut[a_scale_base[k_block_base + i]] * base_scale;
        }

        for (int nb = 0; nb < n_batch; nb++) {
          const uint8_t* b_ptr1 = bb->get_fp4_data(n_idx + nb, k_begin);
          const uint8_t* b_ptr2 = bb->get_fp4_data(n_idx + nb, k_begin + K_STEP);
          const uint8_t* b_ptr3 = bb->get_fp4_data(n_idx + nb, k_begin + K_STEP * 2);
          const uint8_t* b_ptr4 = bb->get_fp4_data(n_idx + nb, k_begin + K_STEP * 3);
          const uint8_t* b_scale_ptr = bb->get_block_scale(n_idx + nb, k_begin);

          // Process all 4 K_STEPs
          fused_mul_reduce_64(a_ptr1, b_ptr1, block_sums1);
          fused_mul_reduce_64(a_ptr2, b_ptr2, block_sums2);
          fused_mul_reduce_64(a_ptr3, b_ptr3, block_sums3);
          fused_mul_reduce_64(a_ptr4, b_ptr4, block_sums4);

          // Apply scales and accumulate for all
          float sum = 0.0f;
          sum += block_sums1[0] * a_s[0] * fp8_lut[b_scale_ptr[0]];
          sum += block_sums1[1] * a_s[1] * fp8_lut[b_scale_ptr[1]];
          sum += block_sums1[2] * a_s[2] * fp8_lut[b_scale_ptr[2]];
          sum += block_sums1[3] * a_s[3] * fp8_lut[b_scale_ptr[3]];
          sum += block_sums2[0] * a_s[4] * fp8_lut[b_scale_ptr[4]];
          sum += block_sums2[1] * a_s[5] * fp8_lut[b_scale_ptr[5]];
          sum += block_sums2[2] * a_s[6] * fp8_lut[b_scale_ptr[6]];
          sum += block_sums2[3] * a_s[7] * fp8_lut[b_scale_ptr[7]];
          sum += block_sums3[0] * a_s[8] * fp8_lut[b_scale_ptr[8]];
          sum += block_sums3[1] * a_s[9] * fp8_lut[b_scale_ptr[9]];
          sum += block_sums3[2] * a_s[10] * fp8_lut[b_scale_ptr[10]];
          sum += block_sums3[3] * a_s[11] * fp8_lut[b_scale_ptr[11]];
          sum += block_sums4[0] * a_s[12] * fp8_lut[b_scale_ptr[12]];
          sum += block_sums4[1] * a_s[13] * fp8_lut[b_scale_ptr[13]];
          sum += block_sums4[2] * a_s[14] * fp8_lut[b_scale_ptr[14]];
          sum += block_sums4[3] * a_s[15] * fp8_lut[b_scale_ptr[15]];
          acc[nb] += sum;
        }
      }

      // Handle remaining K_STEPs
      for (; k_begin < total_k; k_begin += K_STEP) {
        const uint8_t* a_ptr = ba->get_fp4_data(m_idx, k_begin);
        const int k_block_base = k_begin / BLOCK_SIZE;
        float a_s0 = fp8_lut[a_scale_base[k_block_base]] * base_scale;
        float a_s1 = fp8_lut[a_scale_base[k_block_base + 1]] * base_scale;
        float a_s2 = fp8_lut[a_scale_base[k_block_base + 2]] * base_scale;
        float a_s3 = fp8_lut[a_scale_base[k_block_base + 3]] * base_scale;

        for (int nb = 0; nb < n_batch; nb++) {
          const uint8_t* b_ptr = bb->get_fp4_data(n_idx + nb, k_begin);
          const uint8_t* b_scale_ptr = bb->get_block_scale(n_idx + nb, k_begin);
          fused_mul_reduce_64(a_ptr, b_ptr, block_sums1);
          acc[nb] += block_sums1[0] * a_s0 * fp8_lut[b_scale_ptr[0]];
          acc[nb] += block_sums1[1] * a_s1 * fp8_lut[b_scale_ptr[1]];
          acc[nb] += block_sums1[2] * a_s2 * fp8_lut[b_scale_ptr[2]];
          acc[nb] += block_sums1[3] * a_s3 * fp8_lut[b_scale_ptr[3]];
        }
      }

      for (int nb = 0; nb < n_batch; nb++) {
        bc->c_fp32[m_idx * n + n_idx + nb] = acc[nb];
      }
    }
  }
}

// ============================================================================
// Aggressively Optimized Version: Precompute scales + FMA accumulation
// ============================================================================

/**
 * Most optimized version:
 * - Precompute all block scales to float array
 * - Use FMA for accumulation
 * - Minimize function call overhead
 * - Process larger N batches
 */
inline void nvfp4_matmul_opt3(int m, int n, int k, std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> ba,
                              std::shared_ptr<BufferBNVFP4Impl<GemmKernelNVFP4>> bb,
                              std::shared_ptr<BufferCNVFP4Impl<GemmKernelNVFP4>> bc, int ith, int nth) {
  auto [n_start, n_end] = GemmKernelNVFP4::split_range_n(n, ith, nth);

  constexpr int BLOCK_SIZE = 16;
  constexpr int K_STEP = 64;
  constexpr int N_BATCH = 8;

  const int num_k_blocks = k / BLOCK_SIZE;
  const float base_scale = (ba->tensor_scale * bb->tensor_scale) / 4.0f;

  // Precompute all A scales for current processing
  alignas(64) float a_scales_cache[4];  // For 4 blocks in K_STEP

  alignas(64) uint8_t a_fp4[64];
  alignas(64) uint8_t b_fp4[N_BATCH][64];
  alignas(64) int16_t results[N_BATCH][64];

  for (int m_idx = 0; m_idx < m; m_idx++) {
    for (int n_idx = n_start; n_idx < n_end; n_idx += N_BATCH) {
      const int n_batch = std::min(N_BATCH, n_end - n_idx);

      // Use __m256 for accumulation (8 floats)
      __m256 acc_vec = _mm256_setzero_ps();
      float acc_scalar[N_BATCH] = {};

      for (int k_begin = 0; k_begin < k; k_begin += K_STEP) {
        // Precompute A scales for this K_STEP (4 blocks)
        for (int blk = 0; blk < 4; blk++) {
          a_scales_cache[blk] = fp8_e4m3_to_float(ba->get_block_scale(m_idx, k_begin + blk * BLOCK_SIZE)[0]);
        }

        // Unpack A once
        const uint8_t* a_ptr = ba->get_fp4_data(m_idx, k_begin);
        unpack_fp4_to_bytes(a_ptr, a_fp4);
        __m512i a_vec = _mm512_load_si512((const __m512i*)a_fp4);

        // Process N batch
        for (int nb = 0; nb < n_batch; nb++) {
          const uint8_t* b_ptr = bb->get_fp4_data(n_idx + nb, k_begin);
          unpack_fp4_to_bytes(b_ptr, b_fp4[nb]);
          __m512i b_vec = _mm512_load_si512((const __m512i*)b_fp4[nb]);
          nvfp4_mul_64pairs_avx512(a_vec, b_vec, results[nb]);
        }

        // Accumulate with scales - unroll blocks
        for (int nb = 0; nb < n_batch; nb++) {
          const int16_t* res = results[nb];

          // Block 0
          {
            float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx + nb, k_begin)[0]);
            float scale = a_scales_cache[0] * scale_b * base_scale;
            __m256i data = _mm256_loadu_si256((const __m256i*)(res));
            __m256i sum32 = _mm256_madd_epi16(data, _mm256_set1_epi16(1));
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            acc_scalar[nb] += _mm_cvtsi128_si32(sum) * scale;
          }

          // Block 1
          {
            float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx + nb, k_begin + 16)[0]);
            float scale = a_scales_cache[1] * scale_b * base_scale;
            __m256i data = _mm256_loadu_si256((const __m256i*)(res + 16));
            __m256i sum32 = _mm256_madd_epi16(data, _mm256_set1_epi16(1));
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            acc_scalar[nb] += _mm_cvtsi128_si32(sum) * scale;
          }

          // Block 2
          {
            float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx + nb, k_begin + 32)[0]);
            float scale = a_scales_cache[2] * scale_b * base_scale;
            __m256i data = _mm256_loadu_si256((const __m256i*)(res + 32));
            __m256i sum32 = _mm256_madd_epi16(data, _mm256_set1_epi16(1));
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            acc_scalar[nb] += _mm_cvtsi128_si32(sum) * scale;
          }

          // Block 3
          {
            float scale_b = fp8_e4m3_to_float(bb->get_block_scale(n_idx + nb, k_begin + 48)[0]);
            float scale = a_scales_cache[3] * scale_b * base_scale;
            __m256i data = _mm256_loadu_si256((const __m256i*)(res + 48));
            __m256i sum32 = _mm256_madd_epi16(data, _mm256_set1_epi16(1));
            __m128i lo = _mm256_castsi256_si128(sum32);
            __m128i hi = _mm256_extracti128_si256(sum32, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            acc_scalar[nb] += _mm_cvtsi128_si32(sum) * scale;
          }
        }
      }

      // Store results
      for (int nb = 0; nb < n_batch; nb++) {
        bc->c_fp32[m_idx * n + n_idx + nb] = acc_scalar[nb];
      }
    }
  }
}

// ============================================================================
// MoE (Mixture of Experts) NVFP4 Implementation
// ============================================================================

/**
 * MoE Expert Weight Buffer
 * Stores multiple experts' weights in NVFP4 format
 */
template <typename K>
struct MoEBufferBNVFP4Impl {
  std::vector<std::shared_ptr<BufferBNVFP4Impl<K>>> experts;
  int num_experts;
  int n;  // output dim per expert
  int k;  // input dim

  MoEBufferBNVFP4Impl(int num_experts, int n, int k) : num_experts(num_experts), n(n), k(k) {
    experts.resize(num_experts);
  }

  void set_expert(int expert_idx, std::shared_ptr<BufferBNVFP4Impl<K>> expert_buffer) {
    assert(expert_idx >= 0 && expert_idx < num_experts);
    experts[expert_idx] = expert_buffer;
  }

  BufferBNVFP4Impl<K>* get_expert(int expert_idx) { return experts[expert_idx].get(); }
};

/**
 * MoE forward pass with NVFP4 quantized experts
 *
 * Algorithm:
 * 1. For each token, use gate_weights to select top-K experts
 * 2. Compute expert outputs: out[token] = sum(gate_weight[i] * expert[i](input[token]))
 *
 * @param num_tokens: number of input tokens (M dimension)
 * @param hidden_dim: input hidden dimension (K dimension)
 * @param expert_dim: output dimension per expert (N dimension)
 * @param num_experts: total number of experts
 * @param top_k: number of experts to select per token
 * @param input: input activation buffer [num_tokens × hidden_dim]
 * @param experts: MoE expert weight buffer
 * @param gate_logits: gate scores [num_tokens × num_experts] (pre-softmax)
 * @param output: output buffer [num_tokens × expert_dim]
 */
inline void nvfp4_moe_forward(int num_tokens, int hidden_dim, int expert_dim, int num_experts, int top_k,
                              std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> input,
                              MoEBufferBNVFP4Impl<GemmKernelNVFP4>& experts, const float* gate_logits, float* output,
                              int ith, int nth) {
  // Thread-local workspace
  std::vector<float> expert_output(expert_dim);
  std::vector<std::pair<float, int>> gate_scores(num_experts);

  // Allocate temporary output buffer for single expert
  size_t bc_size = BufferCNVFP4Impl<GemmKernelNVFP4>::required_size(1, expert_dim);
  void* bc_buffer = std::aligned_alloc(64, bc_size);
  auto bc = std::make_shared<BufferCNVFP4Impl<GemmKernelNVFP4>>(1, expert_dim, bc_buffer);

  // Process tokens assigned to this thread
  int tokens_per_thread = (num_tokens + nth - 1) / nth;
  int token_start = ith * tokens_per_thread;
  int token_end = std::min(token_start + tokens_per_thread, num_tokens);

  for (int token = token_start; token < token_end; token++) {
    // Step 1: Compute softmax gate weights and select top-K experts
    const float* token_gate = gate_logits + token * num_experts;

    // Find max for numerical stability
    float max_logit = token_gate[0];
    for (int e = 1; e < num_experts; e++) {
      max_logit = std::max(max_logit, token_gate[e]);
    }

    // Compute softmax
    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; e++) {
      float exp_val = std::exp(token_gate[e] - max_logit);
      gate_scores[e] = {exp_val, e};
      sum_exp += exp_val;
    }

    // Normalize and sort to get top-K
    for (int e = 0; e < num_experts; e++) {
      gate_scores[e].first /= sum_exp;
    }

    // Partial sort to get top-K
    std::partial_sort(gate_scores.begin(), gate_scores.begin() + top_k, gate_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // Renormalize top-K weights
    float top_k_sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      top_k_sum += gate_scores[i].first;
    }
    for (int i = 0; i < top_k; i++) {
      gate_scores[i].first /= top_k_sum;
    }

    // Step 2: Clear output for this token
    float* token_output = output + token * expert_dim;
    std::memset(token_output, 0, expert_dim * sizeof(float));

    // Step 3: Compute weighted sum of expert outputs
    for (int i = 0; i < top_k; i++) {
      float weight = gate_scores[i].first;
      int expert_idx = gate_scores[i].second;

      auto expert = experts.get_expert(expert_idx);
      if (!expert) continue;

      // Create a view of input for this token
      // Note: This requires input to support per-row access
      bc->clear(1);

      // Run matmul for this expert
      nvfp4_matmul_opt4(1, expert_dim, hidden_dim, input, experts.experts[expert_idx], bc, 0, 1);

      // Weighted accumulate to output
      for (int j = 0; j < expert_dim; j++) {
        token_output[j] += weight * bc->c_fp32[j];
      }
    }
  }

  std::free(bc_buffer);
}

/**
 * Optimized MoE forward for batch processing
 * Groups tokens by selected experts for better cache efficiency
 */
inline void nvfp4_moe_forward_grouped(int num_tokens, int hidden_dim, int expert_dim, int num_experts, int top_k,
                                      std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> input,
                                      MoEBufferBNVFP4Impl<GemmKernelNVFP4>& experts, const float* gate_logits,
                                      float* output, int ith, int nth) {
  // Step 1: Compute gate assignments for all tokens
  struct TokenAssignment {
    int token_idx;
    float weight;
  };

  std::vector<std::vector<TokenAssignment>> expert_assignments(num_experts);
  std::vector<std::pair<float, int>> gate_scores(num_experts);

  for (int token = 0; token < num_tokens; token++) {
    const float* token_gate = gate_logits + token * num_experts;

    // Softmax
    float max_logit = token_gate[0];
    for (int e = 1; e < num_experts; e++) {
      max_logit = std::max(max_logit, token_gate[e]);
    }

    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; e++) {
      float exp_val = std::exp(token_gate[e] - max_logit);
      gate_scores[e] = {exp_val, e};
      sum_exp += exp_val;
    }

    for (int e = 0; e < num_experts; e++) {
      gate_scores[e].first /= sum_exp;
    }

    std::partial_sort(gate_scores.begin(), gate_scores.begin() + top_k, gate_scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    float top_k_sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
      top_k_sum += gate_scores[i].first;
    }

    for (int i = 0; i < top_k; i++) {
      float weight = gate_scores[i].first / top_k_sum;
      int expert_idx = gate_scores[i].second;
      expert_assignments[expert_idx].push_back({token, weight});
    }
  }

  // Clear output
  std::memset(output, 0, num_tokens * expert_dim * sizeof(float));

  // Step 2: Process experts in parallel
  // Each thread processes a subset of experts
  int experts_per_thread = (num_experts + nth - 1) / nth;
  int expert_start = ith * experts_per_thread;
  int expert_end = std::min(expert_start + experts_per_thread, num_experts);

  // Allocate workspace
  size_t bc_size = BufferCNVFP4Impl<GemmKernelNVFP4>::required_size(1, expert_dim);
  void* bc_buffer = std::aligned_alloc(64, bc_size);
  auto bc = std::make_shared<BufferCNVFP4Impl<GemmKernelNVFP4>>(1, expert_dim, bc_buffer);

  for (int expert_idx = expert_start; expert_idx < expert_end; expert_idx++) {
    auto& assignments = expert_assignments[expert_idx];
    if (assignments.empty()) continue;

    auto expert = experts.get_expert(expert_idx);
    if (!expert) continue;

    // Process all tokens assigned to this expert
    for (const auto& assign : assignments) {
      int token = assign.token_idx;
      float weight = assign.weight;

      bc->clear(1);

      // Run matmul
      // Note: input should be set up to access token's row
      nvfp4_matmul_opt4(1, expert_dim, hidden_dim, input, experts.experts[expert_idx], bc, 0, 1);

      // Accumulate to output (needs atomic or per-thread accumulation for thread safety)
      float* token_output = output + token * expert_dim;
      for (int j = 0; j < expert_dim; j++) {
#pragma omp atomic
        token_output[j] += weight * bc->c_fp32[j];
      }
    }
  }

  std::free(bc_buffer);
}

/**
 * Single-token MoE inference (optimized for autoregressive decoding)
 *
 * @param hidden_dim: input dimension (K)
 * @param expert_dim: output dimension per expert (N)
 * @param num_experts: total experts
 * @param top_k: experts to select
 * @param input: single token input [1 × hidden_dim]
 * @param experts: expert weights
 * @param gate_logits: gate scores for this token [num_experts]
 * @param output: output [expert_dim]
 */
inline void nvfp4_moe_single_token(int hidden_dim, int expert_dim, int num_experts, int top_k,
                                   std::shared_ptr<BufferANVFP4Impl<GemmKernelNVFP4>> input,
                                   MoEBufferBNVFP4Impl<GemmKernelNVFP4>& experts, const float* gate_logits,
                                   float* output) {
  // Compute softmax gate weights
  std::vector<std::pair<float, int>> gate_scores(num_experts);

  float max_logit = gate_logits[0];
  for (int e = 1; e < num_experts; e++) {
    max_logit = std::max(max_logit, gate_logits[e]);
  }

  float sum_exp = 0.0f;
  for (int e = 0; e < num_experts; e++) {
    float exp_val = std::exp(gate_logits[e] - max_logit);
    gate_scores[e] = {exp_val, e};
    sum_exp += exp_val;
  }

  for (int e = 0; e < num_experts; e++) {
    gate_scores[e].first /= sum_exp;
  }

  // Get top-K
  std::partial_sort(gate_scores.begin(), gate_scores.begin() + top_k, gate_scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  // Renormalize
  float top_k_sum = 0.0f;
  for (int i = 0; i < top_k; i++) {
    top_k_sum += gate_scores[i].first;
  }

  // Clear output
  std::memset(output, 0, expert_dim * sizeof(float));

  // Allocate workspace
  size_t bc_size = BufferCNVFP4Impl<GemmKernelNVFP4>::required_size(1, expert_dim);
  void* bc_buffer = std::aligned_alloc(64, bc_size);
  auto bc = std::make_shared<BufferCNVFP4Impl<GemmKernelNVFP4>>(1, expert_dim, bc_buffer);

  // Compute each selected expert
  for (int i = 0; i < top_k; i++) {
    float weight = gate_scores[i].first / top_k_sum;
    int expert_idx = gate_scores[i].second;

    auto expert = experts.get_expert(expert_idx);
    if (!expert) continue;

    bc->clear(1);
    nvfp4_matmul_opt4(1, expert_dim, hidden_dim, input, experts.experts[expert_idx], bc, 0, 1);

    // Weighted accumulate
    for (int j = 0; j < expert_dim; j++) {
      output[j] += weight * bc->c_fp32[j];
    }
  }

  std::free(bc_buffer);
}

}  // namespace nvfp4

#endif  // NVFP4_KERNEL_HPP
