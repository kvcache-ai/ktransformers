/**
 * @file dequant.hpp
 * @brief GGUF (GGML quantized block) -> BF16 strip dequantization for the
 *        AMXINT8 online-quant path.
 *
 * Option B: `--kt-weight-path <gguf-dir> --kt-method AMXINT8` works directly.
 * The AMXINT8 loader dequantizes just the 64-row strip each worker needs
 * (`N_BLOCK` of `GemmKernel224Int8`), feeds it through `BufferB::from_mat_strip`,
 * and writes the existing `.kt` cache format. There is never a full BF16 (or
 * FP32) copy of a layer - f32 intermediates live only in registers, or in a
 * single-row scratch for the exotic-type fallback.
 *
 * Bit-exactness contract: for Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16/F32 the output BF16
 * values are bit-identical to `ggml_internal_get_type_traits(type).to_float()`
 * followed by `ggml_fp32_to_bf16` (round-to-nearest-even, subnormal flush,
 * NaN quieting). The AVX-512 kernels replicate the scalar operation order of
 * ggml-quants.c exactly (no FMA contraction where ggml rounds twice).
 **/
#ifndef CPUINFER_OPERATOR_GGUF_DEQUANT_HPP
#define CPUINFER_OPERATOR_GGUF_DEQUANT_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "ggml.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "../llamafile/conversion.h"

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace kt::gguf {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Bytes occupied by one row of `k` elements of `type`.
inline int64_t gguf_row_bytes(ggml_type type, int64_t k) { return (int64_t)ggml_row_size(type, k); }

// Elements per quantization block.
inline int64_t gguf_block_size(ggml_type type) { return ggml_blck_size(type); }

// ---------------------------------------------------------------------------
// f32 -> bf16 with exact ggml_compute_fp32_to_bf16 semantics
// (RNE rounding, subnormal flush to signed zero, NaN quieting).
// ---------------------------------------------------------------------------

// Self-contained fp16 -> fp32 (bit manipulation, exact). Deliberately does
// NOT use GGML_FP16_TO_FP32: on x86 that macro reads ggml's global
// ggml_table_f32_f16, which is all zeros until ggml_init() runs — a freshly
// imported module would silently convert every fp16 to 0.0. This bit-trick
// produces identical values to the initialized table / F16C conversion.
static inline float kt_fp32_from_bits(uint32_t w) {
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

static inline uint32_t kt_fp32_to_bits(float f) {
  uint32_t w;
  std::memcpy(&w, &f, 4);
  return w;
}

static inline float fp16_to_fp32_bits(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value = kt_fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = kt_fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? kt_fp32_to_bits(denormalized_value)
                                          : kt_fp32_to_bits(normalized_value));
  return kt_fp32_from_bits(result);
}

static inline ggml_bf16_t fp32_to_bf16_ggml(float s) {
  uint32_t i;
  std::memcpy(&i, &s, 4);
  if ((i & 0x7fffffff) > 0x7f800000) { /* nan */
    ggml_bf16_t h;
    h.bits = (uint16_t)((i >> 16) | 64);
    return h;
  }
  if (!(i & 0x7f800000)) { /* subnormal */
    ggml_bf16_t h;
    h.bits = (uint16_t)((i & 0x80000000) >> 16);
    return h;
  }
  ggml_bf16_t h;
  h.bits = (uint16_t)((i + (0x7fff + ((i >> 16) & 1))) >> 16);
  return h;
}

#if defined(__x86_64__) && defined(__AVX512F__)
// 32 f32 -> 32 bf16 (low 16 bits of each 32-bit lane), exact ggml semantics.
static inline __m512i fp32v32_to_bf16v32_ggml(__m512 f) {
  const __m512i i = _mm512_castps_si512(f);
  const __m512i absmask = _mm512_set1_epi32(0x7fffffff);
  const __m512i expmask = _mm512_set1_epi32(0x7f800000);
  const __m512i signmask = _mm512_set1_epi32(0x80000000);
  const __mmask16 is_nan = _mm512_cmpgt_epi32_mask(_mm512_and_si512(i, absmask), expmask);
  const __mmask16 is_sub = _mm512_cmpeq_epi32_mask(_mm512_and_si512(i, expmask), _mm512_setzero_si512());
  const __m512i round =
      _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(_mm512_srli_epi32(i, 16), _mm512_set1_epi32(1)));
  __m512i r = _mm512_srli_epi32(_mm512_add_epi32(i, round), 16);
  const __m512i sub = _mm512_srli_epi32(_mm512_and_si512(i, signmask), 16);
  r = _mm512_mask_mov_epi32(r, is_sub, sub);
  r = _mm512_mask_or_epi32(r, is_nan, r, _mm512_set1_epi32(64));
  return r;
}

// Pack two 16-lane bf16 registers (low 16 bits of each lane) into one
// 32-lane bf16 register ready for a 64-byte store.
static inline __m512i pack_bf16_pairs(__m512i lo16, __m512i hi16) {
  const __m512i p = _mm512_packus_epi32(lo16, hi16);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), p);
}

// Expand 32 bytes to two 32-lane f32 vectors (lanes 0-15 from bytes 0-15).
static inline void bytes32_to_2xf32(__m256i bytes, __m512* lo, __m512* hi) {
  const __m512i lo_i = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(bytes));
  const __m512i hi_i = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(bytes, 1));
  *lo = _mm512_cvtepi32_ps(lo_i);
  *hi = _mm512_cvtepi32_ps(hi_i);
}

// 32 int8 -> two 32-lane f32 vectors (lanes 0-15 from bytes 0-15).
static inline void bytes32_i8_to_2xf32(__m256i bytes, __m512* lo, __m512* hi) {
  const __m512i lo_i = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(bytes));
  const __m512i hi_i = _mm512_cvtepi8_epi32(_mm256_extracti128_si256(bytes, 1));
  *lo = _mm512_cvtepi32_ps(lo_i);
  *hi = _mm512_cvtepi32_ps(hi_i);
}

// d1*q - m1 for two 32-lane f32 quants, stored as 32 bf16.
// (mul and sub are separate roundings, matching ggml's `d1 * q - m1`.)
static inline void fmsub32_store(ggml_bf16_t* dst, __m512 q_lo, __m512 q_hi, float d1, float m1) {
  const __m512 vd = _mm512_set1_ps(d1);
  const __m512 vm = _mm512_set1_ps(m1);
  const __m512 x0 = _mm512_sub_ps(_mm512_mul_ps(q_lo, vd), vm);
  const __m512 x1 = _mm512_sub_ps(_mm512_mul_ps(q_hi, vd), vm);
  const __m512i out = pack_bf16_pairs(fp32v32_to_bf16v32_ggml(x0), fp32v32_to_bf16v32_ggml(x1));
  _mm512_storeu_si512((__m512i*)dst, out);
}

// dsc * q for two 16-lane f32 quants, stored as 32 bf16.
// Each 16-lane half gets its own scalar (d*sc[is] with is=l/16), matching
// ggml's `d * sc[is] * q` with its two separate roundings.
static inline void fmscalar_store_q6(ggml_bf16_t* dst, __m512 q_lo, __m512 q_hi, float dsc_lo, float dsc_hi) {
  const __m512 x0 = _mm512_mul_ps(q_lo, _mm512_set1_ps(dsc_lo));
  const __m512 x1 = _mm512_mul_ps(q_hi, _mm512_set1_ps(dsc_hi));
  const __m512i out = pack_bf16_pairs(fp32v32_to_bf16v32_ggml(x0), fp32v32_to_bf16v32_ggml(x1));
  _mm512_storeu_si512((__m512i*)dst, out);
}

// Convert 64 contiguous f32 (two 16-lane halves) to 32 bf16 and store.
static inline void store_f32x32_as_32xbf16(__m512 f0, __m512 f1, ggml_bf16_t* dst) {
  const __m512i out = pack_bf16_pairs(fp32v32_to_bf16v32_ggml(f0), fp32v32_to_bf16v32_ggml(f1));
  _mm512_storeu_si512((__m512i*)dst, out);
}

// 32 lanes where lanes 0-15 = a, lanes 16-31 = b.
static inline __m512 half16_broadcast(float a, float b) {
  const __m512 x = _mm512_set1_ps(a);
  return _mm512_insertf32x8(x, _mm256_set1_ps(b), 1);
}
#endif  // __x86_64__ && __AVX512F__

// ---------------------------------------------------------------------------
// scalar reference kernels (mirror ggml-quants.c operation order exactly)
// ---------------------------------------------------------------------------

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Q4_K: y = d1*q - m1 with d1 = d*sc (rounded), m1 = min*m (rounded).
static void dequant_q4_k_scalar(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q4_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    for (int64_t b = 0; b < k; b += 256) {
      const block_q4_K* blk = (const block_q4_K*)(row + (b / 256) * sizeof(block_q4_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const float min = fp16_to_fp32_bits(blk->dmin);
      const uint8_t* q = blk->qs;
      int is = 0;
      for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; l++) {
          const int64_t c = b + j + l;
          if (c >= col_begin && c < col_end) drow[c - col_begin] = fp32_to_bf16_ggml(d1 * (q[l] & 0xF) - m1);
        }
        for (int l = 0; l < 32; l++) {
          const int64_t c = b + j + 32 + l;
          if (c >= col_begin && c < col_end) drow[c - col_begin] = fp32_to_bf16_ggml(d2 * (q[l] >> 4) - m2);
        }
        q += 32;
        is += 2;
      }
    }
  }
}

// Q5_K: y = d1*(ql + 16*qh_bit) - m1.
static void dequant_q5_k_scalar(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q5_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    for (int64_t b = 0; b < k; b += 256) {
      const block_q5_K* blk = (const block_q5_K*)(row + (b / 256) * sizeof(block_q5_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const float min = fp16_to_fp32_bits(blk->dmin);
      const uint8_t* ql = blk->qs;
      const uint8_t* qh = blk->qh;
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int l = 0; l < 32; l++) {
          const int64_t c = b + j + l;
          if (c >= col_begin && c < col_end)
            drow[c - col_begin] = fp32_to_bf16_ggml(d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
        }
        for (int l = 0; l < 32; l++) {
          const int64_t c = b + j + 32 + l;
          if (c >= col_begin && c < col_end)
            drow[c - col_begin] = fp32_to_bf16_ggml(d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
      }
    }
  }
}

// Q6_K: y = d * sc * q  (two roundings: d*sc, then *q).
static void dequant_q6_k_scalar(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q6_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    for (int64_t b = 0; b < k; b += 256) {
      const block_q6_K* blk = (const block_q6_K*)(row + (b / 256) * sizeof(block_q6_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const uint8_t* ql = blk->ql;
      const uint8_t* qh = blk->qh;
      const int8_t* sc = blk->scales;
      for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
          const int is = l / 16;
          const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          const float dsc0 = d * sc[is + 0];
          const float dsc1 = d * sc[is + 2];
          const float dsc2 = d * sc[is + 4];
          const float dsc3 = d * sc[is + 6];
          const int64_t c0 = b + n + l;
          if (c0 >= col_begin && c0 < col_end) drow[c0 - col_begin] = fp32_to_bf16_ggml(dsc0 * q1);
          if (c0 + 32 >= col_begin && c0 + 32 < col_end) drow[c0 + 32 - col_begin] = fp32_to_bf16_ggml(dsc1 * q2);
          if (c0 + 64 >= col_begin && c0 + 64 < col_end) drow[c0 + 64 - col_begin] = fp32_to_bf16_ggml(dsc2 * q3);
          if (c0 + 96 >= col_begin && c0 + 96 < col_end) drow[c0 + 96 - col_begin] = fp32_to_bf16_ggml(dsc3 * q4);
        }
        ql += 64;
        qh += 32;
        sc += 8;
      }
    }
  }
}

// Q8_0: y = d * q (single rounding).
static void dequant_q8_0_scalar(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q8_0, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    for (int64_t b = 0; b < k; b += 32) {
      const block_q8_0* blk = (const block_q8_0*)(row + (b / 32) * sizeof(block_q8_0));
      const float d = fp16_to_fp32_bits(blk->d);
      for (int l = 0; l < 32; l++) {
        const int64_t c = b + l;
        if (c >= col_begin && c < col_end) drow[c - col_begin] = fp32_to_bf16_ggml(d * blk->qs[l]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// AVX-512 kernels (bit-exact with the scalar references above)
// ---------------------------------------------------------------------------

#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__)

// Q4_K AVX-512: processes whole 256-element super-blocks that lie inside
// [col_begin, col_end). The caller guarantees block-aligned column ranges.
static void dequant_q4_k_avx512(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t col_off = col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q4_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    const int64_t blk_begin = col_begin / 256;
    const int64_t blk_end = col_end / 256;
    for (int64_t bi = blk_begin; bi < blk_end; bi++) {
      const block_q4_K* blk = (const block_q4_K*)(row + bi * sizeof(block_q4_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const float min = fp16_to_fp32_bits(blk->dmin);
      const uint8_t* q = blk->qs;
      int is = 0;
      for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        // 32 bytes of packed nibbles -> sub-block 1 (low) and sub-block 2 (high)
        const __m256i bytes = _mm256_loadu_si256((const __m256i*)q);
        const __m256i lo_nib = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
        const __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));
        __m512 q1_lo, q1_hi, q2_lo, q2_hi;
        bytes32_to_2xf32(lo_nib, &q1_lo, &q1_hi);
        bytes32_to_2xf32(hi_nib, &q2_lo, &q2_hi);
        ggml_bf16_t* dp = drow + (bi * 256 + j - col_off);
        fmsub32_store(dp, q1_lo, q1_hi, d1, m1);
        fmsub32_store(dp + 32, q2_lo, q2_hi, d2, m2);
        q += 32;
        is += 2;
      }
    }
  }
}

// Q5_K AVX-512.
static void dequant_q5_k_avx512(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t col_off = col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q5_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    const int64_t blk_begin = col_begin / 256;
    const int64_t blk_end = col_end / 256;
    for (int64_t bi = blk_begin; bi < blk_end; bi++) {
      const block_q5_K* blk = (const block_q5_K*)(row + bi * sizeof(block_q5_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const float min = fp16_to_fp32_bits(blk->dmin);
      const uint8_t* ql = blk->qs;
      const uint8_t* qh = blk->qh;
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d * sc;
        const float m2 = min * m;
        const __m256i ql_bytes = _mm256_loadu_si256((const __m256i*)ql);
        const __m256i qh_bytes = _mm256_loadu_si256((const __m256i*)qh);
        const __m256i lo_nib = _mm256_and_si256(ql_bytes, _mm256_set1_epi8(0x0F));
        const __m256i hi_nib = _mm256_and_si256(_mm256_srli_epi16(ql_bytes, 4), _mm256_set1_epi8(0x0F));
        // 5th bit: 16 where (qh & u) != 0
        const __m256i sel1 = _mm256_andnot_si256(
            _mm256_cmpeq_epi8(_mm256_and_si256(qh_bytes, _mm256_set1_epi8((int8_t)u1)), _mm256_setzero_si256()),
            _mm256_set1_epi8(16));
        const __m256i sel2 = _mm256_andnot_si256(
            _mm256_cmpeq_epi8(_mm256_and_si256(qh_bytes, _mm256_set1_epi8((int8_t)u2)), _mm256_setzero_si256()),
            _mm256_set1_epi8(16));
        __m512 q1_lo, q1_hi, q2_lo, q2_hi;
        bytes32_to_2xf32(_mm256_add_epi8(lo_nib, sel1), &q1_lo, &q1_hi);
        bytes32_to_2xf32(_mm256_add_epi8(hi_nib, sel2), &q2_lo, &q2_hi);
        ggml_bf16_t* dp = drow + (bi * 256 + j - col_off);
        fmsub32_store(dp, q1_lo, q1_hi, d1, m1);
        fmsub32_store(dp + 32, q2_lo, q2_hi, d2, m2);
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
      }
    }
  }
}

// Q6_K AVX-512.
static void dequant_q6_k_avx512(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t col_off = col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q6_K, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    const int64_t blk_begin = col_begin / 256;
    const int64_t blk_end = col_end / 256;
    for (int64_t bi = blk_begin; bi < blk_end; bi++) {
      const block_q6_K* blk = (const block_q6_K*)(row + bi * sizeof(block_q6_K));
      const float d = fp16_to_fp32_bits(blk->d);
      const uint8_t* ql = blk->ql;
      const uint8_t* qh = blk->qh;
      const int8_t* sc = blk->scales;
      for (int n = 0; n < 256; n += 128) {
        const __m256i ql0 = _mm256_loadu_si256((const __m256i*)ql);
        const __m256i ql32 = _mm256_loadu_si256((const __m256i*)(ql + 32));
        const __m256i qh_v = _mm256_loadu_si256((const __m256i*)qh);
        const __m256i lo0 = _mm256_and_si256(ql0, _mm256_set1_epi8(0x0F));
        const __m256i hi0 = _mm256_and_si256(_mm256_srli_epi16(ql0, 4), _mm256_set1_epi8(0x0F));
        const __m256i lo32 = _mm256_and_si256(ql32, _mm256_set1_epi8(0x0F));
        const __m256i hi32 = _mm256_and_si256(_mm256_srli_epi16(ql32, 4), _mm256_set1_epi8(0x0F));
        const __m256i h1 = _mm256_and_si256(qh_v, _mm256_set1_epi8(0x03));
        const __m256i h2 = _mm256_and_si256(_mm256_srli_epi16(qh_v, 2), _mm256_set1_epi8(0x03));
        const __m256i h3 = _mm256_and_si256(_mm256_srli_epi16(qh_v, 4), _mm256_set1_epi8(0x03));
        const __m256i h4 = _mm256_and_si256(_mm256_srli_epi16(qh_v, 6), _mm256_set1_epi8(0x03));
        // q = (lo | (h<<4)) - 32, as signed int8 then f32
        __m512 q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi, q4_lo, q4_hi;
        bytes32_i8_to_2xf32(_mm256_sub_epi8(_mm256_or_si256(lo0, _mm256_slli_epi16(h1, 4)), _mm256_set1_epi8(32)),
                            &q1_lo, &q1_hi);
        bytes32_i8_to_2xf32(_mm256_sub_epi8(_mm256_or_si256(lo32, _mm256_slli_epi16(h2, 4)), _mm256_set1_epi8(32)),
                            &q2_lo, &q2_hi);
        bytes32_i8_to_2xf32(_mm256_sub_epi8(_mm256_or_si256(hi0, _mm256_slli_epi16(h3, 4)), _mm256_set1_epi8(32)),
                            &q3_lo, &q3_hi);
        bytes32_i8_to_2xf32(_mm256_sub_epi8(_mm256_or_si256(hi32, _mm256_slli_epi16(h4, 4)), _mm256_set1_epi8(32)),
                            &q4_lo, &q4_hi);
        // scales: is = l/16 -> the 16-lane halves use sc[i] and sc[i+1]
        ggml_bf16_t* dp = drow + (bi * 256 + n - col_off);
        fmscalar_store_q6(dp + 0, q1_lo, q1_hi, d * sc[0], d * sc[1]);
        fmscalar_store_q6(dp + 32, q2_lo, q2_hi, d * sc[2], d * sc[3]);
        fmscalar_store_q6(dp + 64, q3_lo, q3_hi, d * sc[4], d * sc[5]);
        fmscalar_store_q6(dp + 96, q4_lo, q4_hi, d * sc[6], d * sc[7]);
        ql += 64;
        qh += 32;
        sc += 8;
      }
    }
  }
}

// Q8_0 AVX-512.
static void dequant_q8_0_avx512(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t col_off = col_begin;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* row = src + r * gguf_row_bytes(GGML_TYPE_Q8_0, k);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    const int64_t blk_begin = col_begin / 32;
    const int64_t blk_end = col_end / 32;
    for (int64_t bi = blk_begin; bi < blk_end; bi++) {
      const block_q8_0* blk = (const block_q8_0*)(row + bi * sizeof(block_q8_0));
      const float d = fp16_to_fp32_bits(blk->d);
      __m512 q_lo, q_hi;
      bytes32_i8_to_2xf32(_mm256_loadu_si256((const __m256i*)blk->qs), &q_lo, &q_hi);
      const __m512 vd = _mm512_set1_ps(d);
      const __m512i out = pack_bf16_pairs(fp32v32_to_bf16v32_ggml(_mm512_mul_ps(q_lo, vd)),
                                          fp32v32_to_bf16v32_ggml(_mm512_mul_ps(q_hi, vd)));
      _mm512_storeu_si512((__m512i*)(drow + (bi * 32 - col_off)), out);
    }
  }
}

#endif  // AVX512

// ---------------------------------------------------------------------------
// passthrough types
// ---------------------------------------------------------------------------

static void dequant_bf16_passthrough(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end,
                                     int64_t col_begin, int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t row_bytes = k * 2;
  for (int64_t r = row_begin; r < row_end; r++) {
    std::memcpy(dst + (r - row_begin) * dcol, src + r * row_bytes + col_begin * 2, (size_t)(dcol * 2));
  }
}

static void dequant_f16_to_bf16(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t row_bytes = k * 2;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* srow = src + r * row_bytes;
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    int64_t c = col_begin;
#if defined(__x86_64__) && defined(__AVX512F__)
    for (; c + 32 <= col_end; c += 32) {
      const __m512 f0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(srow + c * 2)));
      const __m512 f1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(srow + c * 2 + 32)));
      store_f32x32_as_32xbf16(f0, f1, drow + c - col_begin);
    }
#endif
    for (; c < col_end; c++) {
      drow[c - col_begin] = fp32_to_bf16_ggml(fp16_to_fp32_bits(*(const ggml_fp16_t*)(srow + c * 2)));
    }
  }
}

static void dequant_f32_to_bf16(const uint8_t* src, int64_t k, int64_t row_begin, int64_t row_end, int64_t col_begin,
                                int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t row_bytes = k * 4;
  for (int64_t r = row_begin; r < row_end; r++) {
    const uint8_t* srow = src + r * row_bytes;
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    int64_t c = col_begin;
#if defined(__x86_64__) && defined(__AVX512F__)
    for (; c + 32 <= col_end; c += 32) {
      const __m512 f0 = _mm512_loadu_ps((const float*)(srow + c * 4));
      const __m512 f1 = _mm512_loadu_ps((const float*)(srow + c * 4 + 64));
      store_f32x32_as_32xbf16(f0, f1, drow + c - col_begin);
    }
#endif
    for (; c < col_end; c++) {
      float v;
      std::memcpy(&v, srow + c * 4, 4);
      drow[c - col_begin] = fp32_to_bf16_ggml(v);
    }
  }
}

// ---------------------------------------------------------------------------
// generic fallback: ggml to_float -> bf16, one row at a time (single-row f32
// scratch, never a full-tensor FP32 materialization)
// ---------------------------------------------------------------------------

static void dequant_generic(const uint8_t* src, ggml_type type, int64_t k, int64_t row_begin, int64_t row_end,
                            int64_t col_begin, int64_t col_end, ggml_bf16_t* dst) {
  const int64_t dcol = col_end - col_begin;
  const int64_t row_bytes = gguf_row_bytes(type, k);
  thread_local std::vector<float> row_f32;
  if ((int64_t)row_f32.size() < k) row_f32.resize((size_t)k);
  for (int64_t r = row_begin; r < row_end; r++) {
    to_float(src + r * row_bytes, row_f32.data(), (int)k, type);
    ggml_bf16_t* drow = dst + (r - row_begin) * dcol;
    int64_t c = col_begin;
#if defined(__x86_64__) && defined(__AVX512F__)
    for (; c + 32 <= col_end; c += 32) {
      const __m512 f0 = _mm512_loadu_ps(row_f32.data() + c);
      const __m512 f1 = _mm512_loadu_ps(row_f32.data() + c + 16);
      store_f32x32_as_32xbf16(f0, f1, drow + c - col_begin);
    }
#endif
    for (; c < col_end; c++) {
      drow[c - col_begin] = fp32_to_bf16_ggml(row_f32[c]);
    }
  }
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/**
 * Dequantize rows [row_begin, row_end) of a row-major, block-aligned GGUF
 * tensor of `k` columns, taking only columns [col_begin, col_end), into `dst`
 * as BF16 (row-major, (row_end-row_begin) x (col_end-col_begin)).
 *
 * `src` must point at row 0 of the tensor (e.g. expert 0 of an ffn_*_exps
 * tensor). Rows are block-aligned by construction (k % block_size == 0).
 * Column ranges need not be block-aligned; non-aligned ranges take the
 * generic ggml path.
 *
 * Supported fast types: Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32. Everything
 * else falls back to ggml `to_float` (slower first boot, never a hard
 * failure).
 */
inline void dequant_rows_bf16(const void* src, ggml_type type, int64_t k, int64_t row_begin, int64_t row_end,
                              int64_t col_begin, int64_t col_end, ggml_bf16_t* dst) {
  if (row_begin >= row_end || col_begin >= col_end) return;
  const uint8_t* s = (const uint8_t*)src;
  switch (type) {
    case GGML_TYPE_BF16:
      dequant_bf16_passthrough(s, k, row_begin, row_end, col_begin, col_end, dst);
      return;
    case GGML_TYPE_F16:
      dequant_f16_to_bf16(s, k, row_begin, row_end, col_begin, col_end, dst);
      return;
    case GGML_TYPE_F32:
      dequant_f32_to_bf16(s, k, row_begin, row_end, col_begin, col_end, dst);
      return;
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_0: {
      const int64_t B = gguf_block_size(type);
      const bool aligned = (k % B == 0) && (col_begin % B == 0) && (col_end % B == 0);
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512BW__)
      if (aligned) {
        switch (type) {
          case GGML_TYPE_Q4_K:
            dequant_q4_k_avx512(s, k, row_begin, row_end, col_begin, col_end, dst);
            return;
          case GGML_TYPE_Q5_K:
            dequant_q5_k_avx512(s, k, row_begin, row_end, col_begin, col_end, dst);
            return;
          case GGML_TYPE_Q6_K:
            dequant_q6_k_avx512(s, k, row_begin, row_end, col_begin, col_end, dst);
            return;
          default:
            dequant_q8_0_avx512(s, k, row_begin, row_end, col_begin, col_end, dst);
            return;
        }
      }
#endif
      // scalar (bit-exact with ggml) or non-aligned range
      switch (type) {
        case GGML_TYPE_Q4_K:
          dequant_q4_k_scalar(s, k, row_begin, row_end, col_begin, col_end, dst);
          return;
        case GGML_TYPE_Q5_K:
          dequant_q5_k_scalar(s, k, row_begin, row_end, col_begin, col_end, dst);
          return;
        case GGML_TYPE_Q6_K:
          dequant_q6_k_scalar(s, k, row_begin, row_end, col_begin, col_end, dst);
          return;
        default:
          dequant_q8_0_scalar(s, k, row_begin, row_end, col_begin, col_end, dst);
          return;
      }
    }
    default:
      dequant_generic(s, type, k, row_begin, row_end, col_begin, col_end, dst);
      return;
  }
}

// Convenience: full columns.
inline void dequant_rows_bf16(const void* src, ggml_type type, int64_t k, int64_t row_begin, int64_t row_end,
                              ggml_bf16_t* dst) {
  dequant_rows_bf16(src, type, k, row_begin, row_end, 0, k, dst);
}

// ============================================================================
// Per-row INT4 requant round-trip error (AMXINT4_SMART layer routing).
//
// Measures the relative RMS error introduced by re-quantizing the GGUF
// dequant (the double-quant source baseline, NOT the original fp32) to
// per-row signed INT4 (scale = amax/7, 7 levels each side):
//
//   err = rms(row - requant(row)) / rms(row)
//
// For a gated-MoE layer the output error from quantizing an attribute tensor
// scales with this quantity times the flowing activation norms (first-order
// approximation, router weights ignored — conservative), so a per-attribute
// threshold on err is a sound static routing signal: err <= theta keeps the
// fast per-row INT4 kernel; err > theta re-quantizes the layer to INT8.
// Only the first `sample_rows` rows of expert 0 are measured (single expert,
// small fixed sample — never a full-tensor FP32/BF16 materialization).
// Returns -1 if the sample rows cannot be dequantized.
inline float per_row_int4_roundtrip_err(const void* src, ggml_type type, int64_t k, int64_t sample_rows) {
  if (sample_rows <= 0 || k <= 0) return -1.0f;
  const int64_t rows = std::min<int64_t>(sample_rows, 256);
  thread_local std::vector<ggml_bf16_t> buf;
  buf.resize((size_t)rows * k);
  try {
    dequant_rows_bf16(src, type, k, 0, rows, buf.data());
  } catch (...) {
    return -1.0f;
  }
  double sum_sq = 0.0, sum_err_sq = 0.0;
  for (int64_t r = 0; r < rows; r++) {
    const ggml_bf16_t* row = buf.data() + r * k;
    float amax = 0.0f;
    for (int64_t j = 0; j < k; j++) {
      amax = std::max(amax, std::fabs(ggml_bf16_to_fp32(row[j])));
    }
    const float scale = amax > 0 ? amax / 7.0f : 0.0f;
    double row_ss = 0.0, row_es = 0.0;
    for (int64_t j = 0; j < k; j++) {
      const float v = ggml_bf16_to_fp32(row[j]);
      row_ss += (double)v * v;
      float q;
      if (scale > 0) {
        float x = std::round(v / scale);
        x = std::max(-7.0f, std::min(7.0f, x));
        q = x * scale;
      } else {
        q = 0.0f;
      }
      const float e = v - q;
      row_es += (double)e * e;
    }
    sum_sq += row_ss;
    sum_err_sq += row_es;
  }
  if (sum_sq <= 0) return 0.0f;
  return (float)std::sqrt(sum_err_sq / sum_sq);
}

}  // namespace kt::gguf

#endif  // CPUINFER_OPERATOR_GGUF_DEQUANT_HPP