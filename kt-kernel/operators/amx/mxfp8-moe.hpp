/**
 * @Description  : MXFP8 MoE operator — FP8 E4M3fn weights × BF16 activations
 * @Author       : yyj and Claude
 * @Date         : 2026-06-02
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Serves MiniMax M3 Preview (MXFP8 quantized checkpoint).
 * Based on fp4-moe.hpp (MXFP4). Key differences from MXFP4:
 *   Weight:   FP8 E4M3fn (1 byte/element, no nibble packing)
 *   Act:      BF16 direct (BufferABF16Impl, same as MXFP4)
 *   Decode:   Reuse GemmKernel224FP8 VBMI LUT tables for FP8→BF16
 *   Dot prod: _mm512_dpbf16_ps (BF16×BF16→FP32, same as MXFP4)
 *   Scale:    ue8m0 per-group (group_size=32), converted to FP32 on load
 *
 * Activation: "swigluoai" — gate * sigmoid(gate * alpha) * (up + 1)
 *   Handled by act_fn(gate, up, swiglu_limit, swiglu_alpha) in la/amx.hpp.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MXFP8_MOE_H
#define CPUINFER_OPERATOR_AMX_MXFP8_MOE_H

#include "la/amx_raw_buffers.hpp"   // BufferABF16Impl
#include "la/amx_raw_kernels.hpp"   // GemmKernel224FP8 LUT tables
#include "moe_base.hpp"

namespace amx {

// ============================================================================
// BufferBMXFP8KGroupImpl — FP8 E4M3fn weight buffer with per-group ue8m0 scale
//
// Memory layout: [FP8 weights (n*k bytes)] [FP32 scales (n * k/gs floats)]
// Weights are row-major, 1 byte per element (no nibble packing).
// Scales are row-major, one FP32 value per group of `k_group_size` elements.
// The ue8m0→FP32 conversion happens during load_weights, not here.
// ============================================================================
template <typename K>
struct BufferBMXFP8KGroupImpl {
  using dt = uint8_t;
  uint8_t* b;   // FP8 E4M3fn weights, row-major [n, k], 1 byte per element
  float* d;     // FP32 per-group scales, row-major [n, k/k_group_size]
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr bool SCALE = true;

  static size_t required_size(int n, int k, int k_group_size) {
    // FP8: 1 byte per element (vs FP4's 0.5 byte)
    return static_cast<size_t>(n) * k + sizeof(float) * n * (k / k_group_size);
  }

  BufferBMXFP8KGroupImpl(int n, int k, int k_group_size, void* ptr)
      : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    if (n % N_STEP || k % K_STEP || k % k_group_size) {
      printf("BufferBMXFP8KGroupImpl: n=%d k=%d N_STEP=%d K_STEP=%d gs=%d\n",
             n, k, N_STEP, K_STEP, k_group_size);
      throw std::runtime_error("n or k not aligned to N_STEP/K_STEP/group_size");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<uint8_t*>(ptr);
    // Scale region starts after weight data
    d = reinterpret_cast<float*>(b + static_cast<size_t>(n) * k);
  }

  // Copy raw FP8 bytes from checkpoint. No repacking needed (already 1 byte/element).
  void from_raw_mat(const uint8_t* proj, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const size_t row_bytes = static_cast<size_t>(k);  // 1 byte per element
    const size_t rows = static_cast<size_t>(n_end - n_start);
    std::memcpy(b + n_start * row_bytes, proj + n_start * row_bytes, rows * row_bytes);
  }

  // Pointer to FP8 data at row n_begin, column k_begin
  uint8_t* get_submat(int /*n*/, int /*k*/, int n_begin, int k_begin) {
    return b + static_cast<size_t>(n_begin) * k + k_begin;
  }

  // Pointer to FP32 scale for row n_begin at k-group starting at k_begin
  float* get_scale(int /*n*/, int n_begin, int /*k*/, int k_begin) {
    return d + static_cast<size_t>(n_begin) * k_group_count + k_begin / k_group_size;
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
// GemmKernel224MXFP8SmallKGroup — FP8 E4M3fn × BF16 → FP32 (AVX512 + VBMI)
//
// Per-group scale (group_size=32): each 32 FP8 elements share one ue8m0 scale.
// Inner loop: load 32 FP8 bytes → VBMI LUT decode → 32 BF16 → dpbf16_ps → fmadd(scale).
// ============================================================================
struct GemmKernel224MXFP8SmallKGroup {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 1.0;  // 1 byte per FP8 value

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;   // = group_size

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 6144;  // M3 hidden=6144, fits in one pass

  static std::string name() { return "MXFP8_KGROUP"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  // --------------------------------------------------------------------------
  // FP8 E4M3fn → BF16 decode for 32 values
  //
  // Reuses the VBMI LUT tables from GemmKernel224FP8 (amx_raw_kernels.hpp).
  // The 4 tables (bf16_hi_0/1, bf16_lo_0/1) map the 7-bit FP8 magnitude
  // (128 entries across two 64-byte tables) to BF16 hi/lo bytes.
  //
  // Strategy for 32 values (half of what fp8x64_to_bf16x64 handles):
  //   1. Zero-extend 32 × uint8 → 32 × uint16 in a __m512i
  //   2. Extract the original bytes into lower 32 bytes of a __m512i for LUT
  //   3. VBMI permutex2var_epi8 for hi/lo byte lookup (same as fp8x64_to_bf16x64)
  //   4. Combine hi (with sign) and lo into BF16 pairs
  //
  // The tricky part: fp8x64_to_bf16x64 uses unpacklo/hi which interleaves
  // within 128-bit lanes, scattering 32 values across 2 registers. Instead,
  // we do the lookup on the raw 32 bytes and manually zip lo+hi into uint16.
  // --------------------------------------------------------------------------

  // ActivationBF16: wraps a __m512bh (32 BF16 values = one k-group of activation)
  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
    inline static const __m512i odd_mask = _mm512_set1_epi32(0xFFFF0000);
#endif
    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, odd_mask));
#endif
    }
  };

  // DequantizedWeight: decodes 32 FP8 E4M3fn bytes → 32 BF16 values
  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d;
#else
    __m512 w_even;
    __m512 w_odd;
#endif

    __attribute__((always_inline)) DequantizedWeight(__m256i fp8_32) {
      // Pad 32 bytes into lower half of __m512i for VBMI LUT lookup
      __m512i fp8_64 = _mm512_castsi256_si512(fp8_32);
      // zeros in upper 32 bytes → LUT maps them to BF16(0), harmless

      // VBMI lookup: same tables as GemmKernel224FP8
      __m512i b_hi = _mm512_permutex2var_epi8(
          GemmKernel224FP8::bf16_hi_0_mask(), fp8_64, GemmKernel224FP8::bf16_hi_1_mask());
      __m512i b_lo = _mm512_permutex2var_epi8(
          GemmKernel224FP8::bf16_lo_0_mask(), fp8_64, GemmKernel224FP8::bf16_lo_1_mask());
      // Apply sign from original FP8 bytes
      b_hi = _mm512_or_si512(_mm512_and_si512(GemmKernel224FP8::sign_mask(), fp8_64), b_hi);

      // Now b_lo[i] and b_hi[i] are the low/high bytes of BF16 for FP8 byte i.
      // We need: bf16[i] = (b_hi[i] << 8) | b_lo[i] as a uint16 at position i.
      //
      // unpacklo/hi interleave within 128-bit lanes which scrambles the order.
      // Instead, use the 32 bytes we care about (positions 0..31) directly:
      //   - Extract lower 256 bits of b_lo and b_hi (our 32 valid bytes)
      //   - Interleave with _mm256_unpacklo/hi → 4 chunks of 8 BF16 each
      //   - Assemble into one __m512i of 32 BF16 values

      __m256i lo_256 = _mm512_castsi512_si256(b_lo);   // b_lo bytes 0..31
      __m256i hi_256 = _mm512_castsi512_si256(b_hi);   // b_hi bytes 0..31

      // _mm256_unpacklo/hi_epi8 interleaves within each 128-bit LANE (not whole 256 bits)
      // Lane 0 (bytes 0..15):  unpacklo → [lo[0],hi[0], lo[1],hi[1], ..., lo[7],hi[7]]    = 8 BF16
      //                        unpackhi → [lo[8],hi[8], ..., lo[15],hi[15]]                  = 8 BF16
      // Lane 1 (bytes 16..31): unpacklo → [lo[16],hi[16], ..., lo[23],hi[23]]               = 8 BF16
      //                        unpackhi → [lo[24],hi[24], ..., lo[31],hi[31]]                = 8 BF16
      __m256i bf16_0_2 = _mm256_unpacklo_epi8(lo_256, hi_256);  // lane0: BF16[0..7],  lane1: BF16[16..23]
      __m256i bf16_1_3 = _mm256_unpackhi_epi8(lo_256, hi_256);  // lane0: BF16[8..15], lane1: BF16[24..31]

      // Reorder lanes to get BF16[0..31] in order:
      // bf16_0_2 = [BF16[0..7] | BF16[16..23]]  (128-bit lanes)
      // bf16_1_3 = [BF16[8..15] | BF16[24..31]]
      // Target:    [BF16[0..7] | BF16[8..15] | BF16[16..23] | BF16[24..31]]
      __m512i result = _mm512_inserti64x4(_mm512_castsi256_si512(bf16_0_2), bf16_1_3, 1);
      // Now: [BF16[0..7], BF16[16..23], BF16[8..15], BF16[24..31]]
      // Need to swap 128-bit lanes 1 and 2:
      result = _mm512_permutexvar_epi64(
          _mm512_setr_epi64(0, 1, 4, 5, 2, 3, 6, 7), result);
      // Now: [BF16[0..7], BF16[8..15], BF16[16..23], BF16[24..31]] ✓

#if defined(__AVX512BF16__)
      d = (__m512bh)result;
#else
      w_even = _mm512_castsi512_ps(_mm512_slli_epi32(result, 16));
      w_odd = _mm512_castsi512_ps(_mm512_and_si512(result, _mm512_set1_epi32(0xFFFF0000)));
#endif
    }
  };

  __attribute__((always_inline)) static inline __m512 mxfp8_dot_bf16(
      const DequantizedWeight& w, const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(_mm512_setzero_ps(), act.a, w.d);
#else
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
#endif
  }

  // Buffer type aliases
  using BufferA = BufferABF16Impl<GemmKernel224MXFP8SmallKGroup>;
  using BufferB = BufferBMXFP8KGroupImpl<GemmKernel224MXFP8SmallKGroup>;
  using BufferC = BufferCReduceImpl<GemmKernel224MXFP8SmallKGroup>;

  __attribute__((always_inline)) static inline void reduce4(
      __m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // --------------------------------------------------------------------------
  // mat-vec: M tokens × N output rows, 4-wide N unroll.
  // Each k-group: load 32 FP8 bytes → decode → BF16 dot → fmadd(scale, dot, acc)
  // --------------------------------------------------------------------------
  static void mxfp8_mat_vec_kgroup(int m, int n, int k, int k_group_size,
                                   BufferA* ba, BufferB* bb, BufferC* bc,
                                   int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        // FP8 weights: 32 bytes per k-group per N-row (vs 16 bytes for FP4)
        __m256i* w0 = (__m256i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m256i* w1 = (__m256i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m256i* w2 = (__m256i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m256i* w3 = (__m256i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp8_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp8_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp8_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp8_dot_bf16(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      // N tail: single-row fallback
      for (; n_pos < n_end; n_pos++) {
        __m256i* w = (__m256i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp8_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

  // --------------------------------------------------------------------------
  // mat-mat: 4×4 register tile (MB=4 tokens, NB=4 N-rows → 16 accumulators).
  // Weight decode shared across MB tokens → amortize VBMI cost by 4×.
  // --------------------------------------------------------------------------
  static void mxfp8_mat_mat_kgroup(int m, int n, int k, int k_group_size,
                                   BufferA* ba, BufferB* bb, BufferC* bc,
                                   int ith, int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      __m512bh* a_rows[MB] = {
          (__m512bh*)ba->get_submat(m, k, m_pos + 0, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 1, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 2, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m256i* w0 = (__m256i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m256i* w1 = (__m256i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m256i* w2 = (__m256i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m256i* w3 = (__m256i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = _mm512_set1_ps(s0[g]);
          const __m512 sv1 = _mm512_set1_ps(s1[g]);
          const __m512 sv2 = _mm512_set1_ps(s2[g]);
          const __m512 sv3 = _mm512_set1_ps(s3[g]);

#define MXFP8_FMA_ROW(M_I)                                                          \
  do {                                                                               \
    const ActivationBF16 a(a_rows[M_I][g]);                                          \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp8_dot_bf16(d0, a), acc[M_I][0]);         \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp8_dot_bf16(d1, a), acc[M_I][1]);         \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp8_dot_bf16(d2, a), acc[M_I][2]);         \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp8_dot_bf16(d3, a), acc[M_I][3]);         \
  } while (0)
          MXFP8_FMA_ROW(0);
          MXFP8_FMA_ROW(1);
          MXFP8_FMA_ROW(2);
          MXFP8_FMA_ROW(3);
#undef MXFP8_FMA_ROW
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      // N tail
      for (; n_pos < n_end; n_pos++) {
        __m256i* w = (__m256i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp8_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
        }
      }
    }
    // M tail: remaining tokens that don't fill a 4-token tile
    for (int mi = m_pos; mi < m; mi++) {
      float* c_row = bc->get_submat(m, n, mi, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, mi, 0);
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m256i* w0 = (__m256i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m256i* w1 = (__m256i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m256i* w2 = (__m256i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m256i* w3 = (__m256i*)bb->get_submat(n, k, n_pos + 3, 0);
        const float* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const float* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const float* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const float* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(),
               a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          a0 = _mm512_fmadd_ps(_mm512_set1_ps(s0[g]), mxfp8_dot_bf16(d0, a), a0);
          a1 = _mm512_fmadd_ps(_mm512_set1_ps(s1[g]), mxfp8_dot_bf16(d1, a), a1);
          a2 = _mm512_fmadd_ps(_mm512_set1_ps(s2[g]), mxfp8_dot_bf16(d2, a), a2);
          a3 = _mm512_fmadd_ps(_mm512_set1_ps(s3[g]), mxfp8_dot_bf16(d3, a), a3);
        }
        reduce4(a0, a1, a2, a3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m256i* w = (__m256i*)bb->get_submat(n, k, n_pos, 0);
        const float* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(s[g]), mxfp8_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }
};

// Dispatch functions
inline void mxfp8_vec_mul_kgroup(int m, int n, int k, int k_group_size,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferA> ba,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferB> bb,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferC> bc,
                                 int ith, int nth) {
  GemmKernel224MXFP8SmallKGroup::mxfp8_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mxfp8_mat_mul_kgroup(int m, int n, int k, int k_group_size,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferA> ba,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferB> bb,
                                 std::shared_ptr<GemmKernel224MXFP8SmallKGroup::BufferC> bc,
                                 int ith, int nth) {
  GemmKernel224MXFP8SmallKGroup::mxfp8_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// AMX_MXFP8_MOE_TP — CRTP class for MXFP8 MoE (MiniMax M3)
// ============================================================================
template <class T = amx::GemmKernel224MXFP8SmallKGroup>
class AMX_MXFP8_MOE_TP : public AMX_MOE_BASE<T, AMX_MXFP8_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_MXFP8_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_MXFP8_MOE_TP() = default;
  AMX_MXFP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error("MXFP8 MoE requires group_size == 32 and no zero_point");
    }
    if (config_.hidden_size % quant_config.group_size != 0 ||
        config_.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error(
          "MXFP8 MoE: hidden_size and intermediate_size must be divisible by group_size");
    }
    printf("Creating AMX_MXFP8_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_MXFP8_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mxfp8_mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::mxfp8_vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mxfp8_mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size,
                                down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::mxfp8_vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size,
                                down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  // --------------------------------------------------------------------------
  // ue8m0 → FP32 scale conversion (vectorized)
  //
  // ue8m0 is an unsigned 8-bit exponent: value = 2^(byte - 127).
  // IEEE754 FP32 bit layout: [sign:1][exp:8][mantissa:23].
  // Setting exp field = byte and mantissa = 0 gives 2^(byte - 127) directly.
  // Edge case: byte=0 → 2^(-127) ≈ 5.9e-39, but (0 << 23) = 0.0f.
  // In practice M3 weights don't have ue8m0=0 scales; we accept 0.0f here.
  // --------------------------------------------------------------------------
  static inline void convert_ue8m0_to_fp32(float* __restrict dst,
                                           const uint8_t* __restrict src,
                                           size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
      __m128i bytes = _mm_loadl_epi64((__m128i const*)(src + i));
      __m256i dwords = _mm256_cvtepu8_epi32(bytes);
      __m256 floats = _mm256_castsi256_ps(_mm256_slli_epi32(dwords, 23));
      _mm256_storeu_ps(dst + i, floats);
    }
    for (; i < count; i++) {
      uint32_t bits = static_cast<uint32_t>(src[i]) << 23;
      std::memcpy(dst + i, &bits, sizeof(float));
    }
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("MXFP8 MoE requires group_size > 0 and no zero_point");
    if (config_.gate_scale == nullptr)
      throw std::runtime_error("MXFP8 MoE requires native MXFP8 weights with ue8m0 scales");

    // --- Load FP8 weights (1 byte per element, no nibble packing) ---
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // FP8: no >> 1 (1 byte/element, not nibble-packed)
          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          gate_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.gate_proj + weight_offset, ith, nth);
          up_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.up_proj + weight_offset, ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          size_t weight_offset = logical_expert_id * config_.hidden_size * config_.intermediate_size;
          down_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.down_proj + weight_offset, ith, nth);
        },
        nullptr);

    // --- Convert ue8m0 scales to FP32 ---
    // M3 scale dtype is uint8 (ue8m0), not bf16 like V4. Use bit-shift conversion.
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_count =
              (static_cast<size_t>(config_.intermediate_size) * config_.hidden_size) / config_.quant_config.group_size;
          // gate and up scales: [intermediate_size, hidden_size / group_size]
          convert_ue8m0_to_fp32(gate_bb_[expert_idx]->d,
                                (const uint8_t*)config_.gate_scale + logical_expert_id * scale_count, scale_count);
          convert_ue8m0_to_fp32(up_bb_[expert_idx]->d,
                                (const uint8_t*)config_.up_scale + logical_expert_id * scale_count, scale_count);
          // down scale: [hidden_size, intermediate_size / group_size] — same total count
          convert_ue8m0_to_fp32(down_bb_[expert_idx]->d,
                                (const uint8_t*)config_.down_scale + logical_expert_id * scale_count, scale_count);
        },
        nullptr);
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      __m512i data = _mm512_loadu_si512((__m512i*)s);
      _mm512_storeu_si512((__m512i*)d, data);
      d += 64;
      s += 64;
    }
    if (bytes -= chunks * 64) std::memcpy(d, s, bytes);
  }

  static inline void fast_fp32_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
      __m512 v0 = _mm512_loadu_ps(src + i);
      __m512 v1 = _mm512_loadu_ps(src + i + 16);
      __m512i i0 = _mm512_srli_epi32(_mm512_castps_si512(v0), 16);
      __m512i i1 = _mm512_srli_epi32(_mm512_castps_si512(v1), 16);
      __m512i packed = _mm512_packus_epi32(i0, i1);
      __m512i permuted = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);
      _mm512_storeu_si512((__m512i*)(dst + i), permuted);
    }
    for (; i < count; i++) dst[i] = ggml_fp32_to_bf16(src[i]);
  }

  // FP32 -> ue8m0 (uint8) — AVX-512 sibling of avx2::fast_fp32_to_ue8m0.
  // Used by write_weights_to_buffer for layerwise prefill: CPU stores FP32
  // scales in BufferB.d for fast FMA in inner loop, but GPU MXFP8 buffer
  // expects ue8m0 uint8 (1 byte/scale). Extract bits 23-30 of each FP32
  // (the exponent field) and store as uint8. Bit-exact iff the FP32 came
  // from convert_ue8m0_to_fp32 (mantissa=0), which is the case for
  // layerwise prefill since CPU forward never modifies BufferB.d.
  static inline void fast_fp32_to_ue8m0(uint8_t* __restrict dst,
                                         const float* __restrict src,
                                         size_t count) {
    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
      __m512 v = _mm512_loadu_ps(src + i);
      __m512i bits = _mm512_castps_si512(v);
      // Extract bits 23-30 (FP32 exponent) -> 16 x uint32 in [0,255]
      __m512i shifted = _mm512_srli_epi32(bits, 23);
      // Truncate 32 -> 8 (AVX512-BW): values already in [0,255]
      __m128i p8 = _mm512_cvtepi32_epi8(shifted);
      _mm_storeu_si128((__m128i*)(dst + i), p8);
    }
    for (; i < count; i++) {
      uint32_t b;
      std::memcpy(&b, &src[i], sizeof(uint32_t));
      dst[i] = (uint8_t)((b >> 23) & 0xFF);
    }
  }

  // write_weights_to_buffer: copies CPU expert weights to GPU pinned buffer
  // for the GPU prefill fallback (sglang/kt_ep_wrapper.py:_prepare_weight_mxfp8).
  //
  // Unified per-row scatter — works for any (cpu_tp_count, gpu_tp_count)
  // relationship. Mirrors avx2/mxfp8-moe.hpp and avx2/fp8-moe.hpp:347-451.
  //
  // Replaces the earlier `cpu_tp_count >= gpu_tp_count` direct-write branch
  // that (a) silently no-op'd for the typical 2-NUMA x tp>=4 deployment and
  // (b) wrote bf16 (2 bytes/scale) into a uint8 (1 byte/scale) GPU buffer,
  // corrupting adjacent expert slots whenever it did execute.
  //
  // GPU MXFP8 buffer layout (set by Fp8MoEMethod.create_weights when
  // use_mxfp8=True in kt-sglang/fp8.py:872-893):
  //   w13_weight_scale_inv: (E, 2*intermediate, hidden/group_size) uint8 ue8m0
  //   w2_weight_scale_inv:  (E, hidden, intermediate/group_size)   uint8 ue8m0
  // CPU storage (gate_bb_[e]->d / up_bb_[e]->d / down_bb_[e]->d) is fp32
  // (pre-decoded ue8m0 from disk, for fast FMA in CPU forward); we re-encode
  // to ue8m0 via fast_fp32_to_ue8m0 on write.
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);
    const int group_size = config.quant_config.group_size;

    // ========= W13 (gate+up): Shape [intermediate, hidden], split by N only =========
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;
    const int scales_per_row_w13 = cpu_k_w13 / group_size;
    const size_t gpu_w13_scale_per_mat = (size_t)gpu_n_w13 * scales_per_row_w13;

    // ========= W2 (down): Shape [hidden, intermediate], split by K =========
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;
    const int cpu_scales_per_row_w2 = cpu_k_w2 / group_size;
    const int gpu_scales_per_row_w2 = gpu_k_w2 / group_size;

    constexpr int NUM_W13_TASKS = 32;  // per matrix (gate or up); total 64 W13 tasks
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // ---- W13 weight + scale: per-row scatter (one target_gpu per row) ----
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int rows_per_task = (cpu_n_w13 + NUM_W13_TASKS - 1) / NUM_W13_TASKS;
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w13);
            if (row_start >= cpu_n_w13) return;

            for (int row = row_start; row < row_end; row++) {
              const int global_n = global_n_offset_w13 + row;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              // Weight row: full K (cpu_k_w13 == gpu_k_w13 for W13).
              uint8_t* w_dst = (uint8_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_w_off = is_up ? gpu_w13_weight_per_mat : 0;
              fast_memcpy(w_dst + expert_w_off + (size_t)n_in_gpu * gpu_k_w13,
                          bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13);

              // Scale row: full K/group_size ue8m0 bytes (fp32 -> ue8m0).
              uint8_t* s_dst = (uint8_t*)w13_scale_ptrs[target_gpu];
              const size_t expert_s_off = is_up ? gpu_w13_scale_per_mat : 0;
              fast_fp32_to_ue8m0(
                  s_dst + expert_s_off + (size_t)n_in_gpu * scales_per_row_w13,
                  bb->d + (size_t)row * scales_per_row_w13,
                  scales_per_row_w13);
            }
          } else {
            // ---- W2 weight + scale: per-row + per-k-slice scatter ----
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int rows_per_task = (cpu_n_w2 + NUM_W2_TASKS - 1) / NUM_W2_TASKS;
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w2);
            if (row_start >= cpu_n_w2) return;

            for (int row = row_start; row < row_end; row++) {
              // Each CPU TP's K range = [global_k_offset_w2, +cpu_k_w2).
              // Iterate gpu_k_w2-aligned k-slices within this range; each
              // slice goes to its own target_gpu. cpu < gpu case: one slice
              // (length cpu_k_w2). cpu == gpu: one slice (length gpu_k_w2).
              // cpu > gpu: multiple slices.
              for (int k_start = 0; k_start < cpu_k_w2; k_start += gpu_k_w2) {
                const int k_slice_len = std::min(gpu_k_w2, cpu_k_w2 - k_start);
                const int global_k = global_k_offset_w2 + k_start;
                const int target_gpu = global_k / gpu_k_w2;
                const int k_in_gpu = global_k % gpu_k_w2;

                // Weight K-slice
                uint8_t* w_dst = (uint8_t*)w2_weight_ptrs[target_gpu];
                fast_memcpy(w_dst + (size_t)row * gpu_k_w2 + k_in_gpu,
                            bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len);

                // Scale K-slice (k_slice_len/group_size ue8m0 bytes)
                const int scale_slice_len = k_slice_len / group_size;
                const int k_in_gpu_scale = k_in_gpu / group_size;
                const int k_start_scale = k_start / group_size;
                uint8_t* s_dst = (uint8_t*)w2_scale_ptrs[target_gpu];
                fast_fp32_to_ue8m0(
                    s_dst + (size_t)row * gpu_scales_per_row_w2 + k_in_gpu_scale,
                    bb->d + (size_t)row * cpu_scales_per_row_w2 + k_start_scale,
                    scale_slice_len);
              }
            }
          }
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization for AMX_MXFP8_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AMX_MXFP8_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_MXFP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_MXFP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP8 MoE requires FP8 weights with ue8m0 KGroup Scale");

    printf("MXFP8 MoE: loading from %s\n",
           use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "flat arrays with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      // FP8: 1 byte per element (no / 2)
      tpc.gate_proj = new uint8_t[tpc.expert_num * weight_elem_count];
      tpc.up_proj = new uint8_t[tpc.expert_num * weight_elem_count];
      tpc.down_proj = new uint8_t[tpc.expert_num * weight_elem_count];
      // Scales: uint8 ue8m0 (will be converted to FP32 inside per-TP load_weights)
      tpc.gate_scale = new uint8_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new uint8_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new uint8_t[tpc.expert_num * scales_elem_count];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              uint8_t* src_gate_scale = (uint8_t*)config.gate_scales[0][expert_id];
              uint8_t* src_up_scale = (uint8_t*)config.up_scales[0][expert_id];
              uint8_t* src_down_scale = (uint8_t*)config.down_scales[0][expert_id];

              // gate/up: row-major [intermediate_size, hidden_size], TP splits along hidden
              memcpy((uint8_t*)tpc.gate_proj + expert_id * weight_elem_count,
                     src_gate + i * weight_elem_count, weight_elem_count);
              memcpy((uint8_t*)tpc.up_proj + expert_id * weight_elem_count,
                     src_up + i * weight_elem_count, weight_elem_count);
              memcpy((uint8_t*)tpc.gate_scale + expert_id * scales_elem_count,
                     src_gate_scale + i * scales_elem_count, scales_elem_count);
              memcpy((uint8_t*)tpc.up_scale + expert_id * scales_elem_count,
                     src_up_scale + i * scales_elem_count, scales_elem_count);

              // down: row-major [hidden_size, intermediate_size], TP splits along intermediate
              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + expert_id * weight_elem_count + col * tpc.intermediate_size,
                       src_down + col * config.intermediate_size + i * tpc.intermediate_size,
                       tpc.intermediate_size);
                memcpy((uint8_t*)tpc.down_scale + expert_id * scales_elem_count +
                           col * (tpc.intermediate_size / group_size),
                       src_down_scale + col * (config.intermediate_size / group_size) +
                           i * (tpc.intermediate_size / group_size),
                       tpc.intermediate_size / group_size);
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                // gate/up: no >> 1 (FP8 = 1 byte/element)
                memcpy((uint8_t*)tpc.gate_proj + expert_id * weight_elem_count,
                       (uint8_t*)config.gate_proj +
                           expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count,
                       weight_elem_count);
                memcpy((uint8_t*)tpc.up_proj + expert_id * weight_elem_count,
                       (uint8_t*)config.up_proj +
                           expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count,
                       weight_elem_count);
                // scales (uint8, not bf16)
                memcpy((uint8_t*)tpc.gate_scale + expert_id * scales_elem_count,
                       (uint8_t*)config.gate_scale +
                           expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                           i * scales_elem_count,
                       scales_elem_count);
                memcpy((uint8_t*)tpc.up_scale + expert_id * scales_elem_count,
                       (uint8_t*)config.up_scale +
                           expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                           i * scales_elem_count,
                       scales_elem_count);

                // down: column-wise TP split
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + expert_id * weight_elem_count + col * tpc.intermediate_size,
                         (uint8_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         tpc.intermediate_size);
                  memcpy((uint8_t*)tpc.down_scale + expert_id * scales_elem_count +
                             col * (tpc.intermediate_size / group_size),
                         (uint8_t*)config.down_scale +
                             expert_id * (config.intermediate_size / group_size) * config.hidden_size +
                             col * (config.intermediate_size / group_size) +
                             i * (tpc.intermediate_size / group_size),
                         tpc.intermediate_size / group_size);
                }
              },
              nullptr);
        }
      }
      printf("MXFP8 TP %d load weight done.\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[](uint8_t*)(tpc.gate_proj);
      delete[](uint8_t*)(tpc.up_proj);
      delete[](uint8_t*)(tpc.down_proj);
      delete[](uint8_t*)(tpc.gate_scale);
      delete[](uint8_t*)(tpc.up_scale);
      delete[](uint8_t*)(tpc.down_scale);
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != (size_t)gpu_tp_count || w13_scale_ptrs.size() != (size_t)gpu_tp_count ||
        w2_weight_ptrs.size() != (size_t)gpu_tp_count || w2_scale_ptrs.size() != (size_t)gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config,
                                            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_MXFP8_MOE_H
