/**
 * @Description  : GPTQ-Int4 symmetric dequantization for AVX2
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * GPTQ symmetric quantization (sym=true):
 *   dequant[k,n] = (((qweight[k/8,n] >> ((k%8)*4)) & 0xF) - 8) * scale[k/gs, n]
 *
 * qweight layout: [K/8, N] int32, packing 8 x 4-bit values along K dimension
 * scales layout: [K/group_size, N] fp16 (converted to fp32 at load time)
 * qzeros: not needed (symmetric, zero_point = 8 for all)
 **/
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT4_DEQUANT_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT4_DEQUANT_H

#include <immintrin.h>
#include <cstdint>

namespace avx2 {

// Dequantize 8 x 4-bit values from a packed int32 (symmetric, zero_point=8)
// packed_weight contains 8 nibbles: bits [0:3]=val0, [4:7]=val1, ..., [28:31]=val7
// Result: ((nibble - 8) * scale) for each of the 8 values
static inline __m256 gptq_sym_dequant_8x4bit(uint32_t packed_weight, float scale) {
  // Variable shift: extract each 4-bit nibble into its own 32-bit lane
  // GPTQ packing: bit 0-3 = k_offset 0, bit 4-7 = k_offset 1, ...
  // _mm256_set_epi32 sets lanes in reverse order (lane 7 first), so:
  // lane 0 = shift 0 (k_offset 0), lane 1 = shift 4 (k_offset 1), ...
  const __m256i shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  __m256i packed_v = _mm256_set1_epi32(packed_weight);
  __m256i nibbles = _mm256_and_si256(_mm256_srlv_epi32(packed_v, shifts),
                                     _mm256_set1_epi32(0xF));

  // (nibble - 8) * scale
  __m256 w = _mm256_cvtepi32_ps(nibbles);
  return _mm256_mul_ps(_mm256_sub_ps(w, _mm256_set1_ps(8.0f)),
                        _mm256_set1_ps(scale));
}

// Scalar version for verification
static inline float gptq_sym_dequant_scalar(uint32_t packed_weight, int k_in_pack, float scale) {
  int nibble = (packed_weight >> (k_in_pack * 4)) & 0xF;
  return (float)(nibble - 8) * scale;
}

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT4_DEQUANT_H
