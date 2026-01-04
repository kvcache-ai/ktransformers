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

namespace nvfp4 {

// Inline BF16 conversion helper (to avoid linking issues)
static inline float bf16_to_f32_inline(ggml_bf16_t x) {
  union {
    uint32_t i;
    float f;
  } u;
  u.i = ((uint32_t)x.bits) << 16;
  return u.f;
}

/**
 * NVFP4 (NVIDIA FP4) Format Implementation
 *
 * Format: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
 * Values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
 *
 * Dual-Level Scaling:
 * - Block Scale: FP8 E4M3 (1 per 16 FP4 values)
 * - Tensor Scale: FP32 (1 per tensor)
 *
 * References:
 * - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
 * - https://arxiv.org/html/2509.23202v1
 */

// ============================================================================
// E2M1 FP4 Format Utilities
// ============================================================================

// E2M1 encodable values (positive only, sign bit handled separately)
constexpr float E2M1_VALUES[8] = {
    0.0f,  // 000: 0
    0.5f,  // 001: 2^-1
    1.0f,  // 010: 2^0
    1.5f,  // 011: 2^0 * 1.5
    2.0f,  // 100: 2^1
    3.0f,  // 101: 2^1 * 1.5
    4.0f,  // 110: 2^2
    6.0f   // 111: 2^2 * 1.5
};

// Decode E2M1 4-bit value to float
inline float e2m1_to_float(uint8_t e2m1) {
  uint8_t sign = (e2m1 >> 3) & 0x1;
  uint8_t value_idx = e2m1 & 0x7;
  float val = E2M1_VALUES[value_idx];
  return sign ? -val : val;
}

// Encode float to E2M1 4-bit value (round to nearest)
inline uint8_t float_to_e2m1(float x) {
  if (x == 0.0f) return 0;

  uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;
  float abs_x = std::abs(x);

  // Clamp to representable range
  abs_x = std::min(abs_x, 6.0f);

  // Find nearest E2M1 value using round-to-nearest
  int best_idx = 0;
  float best_diff = std::abs(abs_x - E2M1_VALUES[0]);

  for (int i = 1; i < 8; i++) {
    float diff = std::abs(abs_x - E2M1_VALUES[i]);
    if (diff < best_diff) {
      best_diff = diff;
      best_idx = i;
    }
  }

  return sign | best_idx;
}

// ============================================================================
// FP8 E4M3 Format Utilities
// ============================================================================

// FP8 E4M3: 1 sign, 4 exponent (bias=7), 3 mantissa
// Range: [-448, 448], more uniform than E8M0

// LUT for FP8 E4M3 -> Float conversion (256 entries, positive values only)
// Negative values computed by negation
namespace fp8_lut {
inline float compute_fp8_e4m3_value(uint8_t fp8_unsigned) {
  if (fp8_unsigned == 0) return 0.0f;
  uint8_t exp = (fp8_unsigned >> 3) & 0x0F;
  uint8_t mantissa = fp8_unsigned & 0x07;
  if (exp == 0) {
    return std::ldexp((float)mantissa / 8.0f, -6);
  } else if (exp == 0x0F) {
    return 448.0f;
  } else {
    return std::ldexp(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
  }
}

// Generate LUT at compile time or static init
inline const float* get_fp8_e4m3_lut() {
  static float lut[256] = {};
  static bool initialized = false;
  if (!initialized) {
    for (int i = 0; i < 128; i++) {
      lut[i] = compute_fp8_e4m3_value(i);         // Positive
      lut[i + 128] = -compute_fp8_e4m3_value(i);  // Negative
    }
    initialized = true;
  }
  return lut;
}
}  // namespace fp8_lut

inline float fp8_e4m3_to_float(uint8_t fp8) { return fp8_lut::get_fp8_e4m3_lut()[fp8]; }

// Slow version for reference
inline float fp8_e4m3_to_float_slow(uint8_t fp8) {
  if (fp8 == 0) return 0.0f;

  uint8_t sign = (fp8 >> 7) & 0x1;
  uint8_t exp = (fp8 >> 3) & 0x0F;
  uint8_t mantissa = fp8 & 0x07;

  // Special cases
  if (exp == 0) {
    // Subnormal: (-1)^sign * 2^-6 * (mantissa / 8)
    float val = std::ldexp((float)mantissa / 8.0f, -6);
    return sign ? -val : val;
  } else if (exp == 0x0F) {
    // NaN or Inf - clamp to max
    return sign ? -448.0f : 448.0f;
  } else {
    // Normal: (-1)^sign * 2^(exp-7) * (1 + mantissa/8)
    float val = std::ldexp(1.0f + (float)mantissa / 8.0f, (int)exp - 7);
    return sign ? -val : val;
  }
}

inline uint8_t float_to_fp8_e4m3(float x) {
  if (x == 0.0f) return 0;

  // Clamp to E4M3 range
  x = std::max(-448.0f, std::min(448.0f, x));

  uint8_t sign = (x < 0.0f) ? 0x1 : 0x0;
  float abs_x = std::abs(x);

  // Get exponent and mantissa using frexp
  int exp;
  float mantissa_f = std::frexp(abs_x, &exp);  // abs_x = mantissa_f * 2^exp, mantissa_f in [0.5, 1)

  // Convert to E4M3 representation
  // E4M3 stores: 2^(exp_stored - 7) * (1 + m/8)
  // We have: 2^exp * mantissa_f = 2^exp * mantissa_f

  // Adjust: mantissa_f in [0.5, 1) → [1, 2) by exp -= 1
  exp -= 1;
  mantissa_f *= 2.0f;  // Now mantissa_f in [1, 2)

  // Bias exponent (E4M3 bias = 7)
  int exp_biased = exp + 7;

  // Clamp exponent to valid range [0, 14] (15 is reserved for NaN/Inf)
  if (exp_biased < 0) {
    // Subnormal
    exp_biased = 0;
    // Adjust mantissa for subnormal
    mantissa_f = abs_x * std::ldexp(1.0f, 6);  // Scale to subnormal range
  } else if (exp_biased > 14) {
    // Overflow - clamp to max
    exp_biased = 14;
    mantissa_f = 1.875f;  // 1 + 7/8
  }

  // Extract 3-bit mantissa (fractional part)
  uint8_t mantissa_bits = (uint8_t)((mantissa_f - 1.0f) * 8.0f + 0.5f);  // Round to nearest
  mantissa_bits = std::min((uint8_t)7, mantissa_bits);

  return (sign << 7) | ((exp_biased & 0x0F) << 3) | (mantissa_bits & 0x07);
}

// ============================================================================
// NVFP4 Quantization Block Format
// ============================================================================

struct NVFP4Block {
  static constexpr int SIZE = 16;  // 16 FP4 values per block

  uint8_t fp4_data[SIZE / 2];  // Packed: 2 FP4 per byte
  uint8_t scale_fp8;           // FP8 E4M3 block scale

  // Quantize 16 float values to NVFP4 block
  void quantize(const float* values) {
    // Find max absolute value in block
    float max_abs = 0.0f;
    for (int i = 0; i < SIZE; i++) {
      max_abs = std::max(max_abs, std::abs(values[i]));
    }

    // Compute block scale: map max to E2M1 max (6.0)
    float block_scale = max_abs / 6.0f;
    scale_fp8 = float_to_fp8_e4m3(block_scale);

    // Quantize each value
    float scale_inv = (block_scale > 1e-10f) ? (1.0f / block_scale) : 0.0f;

    for (int i = 0; i < SIZE; i += 2) {
      float normalized0 = values[i] * scale_inv;
      float normalized1 = values[i + 1] * scale_inv;

      uint8_t q0 = float_to_e2m1(normalized0);
      uint8_t q1 = float_to_e2m1(normalized1);

      fp4_data[i / 2] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
  }

  // Dequantize NVFP4 block to float values
  void dequantize(float* values, float tensor_scale = 1.0f) const {
    float block_scale = fp8_e4m3_to_float(scale_fp8);
    float combined_scale = block_scale * tensor_scale;

    for (int i = 0; i < SIZE; i += 2) {
      uint8_t packed = fp4_data[i / 2];
      uint8_t q0 = packed & 0x0F;
      uint8_t q1 = (packed >> 4) & 0x0F;

      values[i] = e2m1_to_float(q0) * combined_scale;
      values[i + 1] = e2m1_to_float(q1) * combined_scale;
    }
  }
};

// ============================================================================
// NVFP4 Tensor Quantization
// ============================================================================

struct NVFP4Tensor {
  uint8_t* fp4_data;      // Packed FP4 values
  uint8_t* block_scales;  // FP8 E4M3 block scales
  float tensor_scale;     // FP32 tensor scale

  int rows, cols;
  int block_count;

  static constexpr int BLOCK_SIZE = 16;

  NVFP4Tensor(int rows, int cols, void* buffer) : rows(rows), cols(cols) {
    int total_elems = rows * cols;
    assert(total_elems % BLOCK_SIZE == 0);

    block_count = total_elems / BLOCK_SIZE;

    fp4_data = (uint8_t*)buffer;
    block_scales = fp4_data + total_elems / 2;
  }

  static size_t required_size(int rows, int cols) {
    int total_elems = rows * cols;
    return total_elems / 2             // FP4 packed data
           + total_elems / BLOCK_SIZE  // FP8 block scales
           + sizeof(float);            // FP32 tensor scale (stored separately)
  }

  // Quantize from float array
  void quantize_from_float(const float* src) {
    // Step 1: Find global max for tensor scale
    float global_max = 0.0f;
    int total_elems = rows * cols;
    for (int i = 0; i < total_elems; i++) {
      global_max = std::max(global_max, std::abs(src[i]));
    }

    // Tensor scale: map global max to E4M3 range (448)
    tensor_scale = global_max / 448.0f;

    // Step 2: Quantize each block
    for (int block_idx = 0; block_idx < block_count; block_idx++) {
      const float* block_src = src + block_idx * BLOCK_SIZE;

      NVFP4Block block;
      block.quantize(block_src);

      // Copy to tensor storage
      memcpy(fp4_data + block_idx * BLOCK_SIZE / 2, block.fp4_data, BLOCK_SIZE / 2);
      block_scales[block_idx] = block.scale_fp8;
    }
  }

  // Dequantize to float array
  void dequantize_to_float(float* dst) const {
    for (int block_idx = 0; block_idx < block_count; block_idx++) {
      NVFP4Block block;
      memcpy(block.fp4_data, fp4_data + block_idx * BLOCK_SIZE / 2, BLOCK_SIZE / 2);
      block.scale_fp8 = block_scales[block_idx];

      float* block_dst = dst + block_idx * BLOCK_SIZE;
      block.dequantize(block_dst, tensor_scale);
    }
  }
};

// ============================================================================
// Helper Functions
// ============================================================================

// Compute global max absolute value
inline float compute_global_max_abs(int size, const float* data) {
  float max_val = 0.0f;

  // Use AVX512 for acceleration
  for (int i = 0; i < size; i += 16) {
    int remaining = std::min(16, size - i);
    if (remaining == 16) {
      __m512 vals = _mm512_loadu_ps(data + i);
      __m512 abs_vals = _mm512_abs_ps(vals);
      float local_max = _mm512_reduce_max_ps(abs_vals);
      max_val = std::max(max_val, local_max);
    } else {
      for (int j = 0; j < remaining; j++) {
        max_val = std::max(max_val, std::abs(data[i + j]));
      }
    }
  }

  return max_val;
}

// Compute global max absolute value from BF16
inline float compute_global_max_abs_bf16(int size, const ggml_bf16_t* data) {
  float max_val = 0.0f;

  for (int i = 0; i < size; i += 32) {
    int remaining = std::min(32, size - i);
    if (remaining >= 32) {
      __m512 f0, f1;
      avx512_32xbf16_to_32xfp32((__m512i*)(data + i), &f0, &f1);

      __m512 abs_f0 = _mm512_abs_ps(f0);
      __m512 abs_f1 = _mm512_abs_ps(f1);

      max_val = std::max(max_val, _mm512_reduce_max_ps(abs_f0));
      max_val = std::max(max_val, _mm512_reduce_max_ps(abs_f1));
    } else {
      for (int j = 0; j < remaining; j++) {
        float val = bf16_to_f32_inline(data[i + j]);
        max_val = std::max(max_val, std::abs(val));
      }
    }
  }

  return max_val;
}

}  // namespace nvfp4

#endif  // NVFP4_UTILS_HPP
