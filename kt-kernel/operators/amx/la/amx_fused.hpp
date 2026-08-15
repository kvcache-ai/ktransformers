/**
 * @file amx_fused.hpp
 * @brief Task-specific fused two-stage MoE decode kernels (qlen=1 shape).
 *
 * AMXINT4_SMART stage-pair kernels (F4x8, F8x16, F4x16), selected by the
 * (upstream, downstream) precision pair stored on the layer:
 *   F4x8  = (0,1): gate/up on the per-row INT4 node, down on INT8
 *   F8x16 = (1,2): gate/up on the INT8 node, down on BF16
 *   F4x16 = (0,2): gate/up on the per-row INT4 node, down on BF16
 *
 * Substrate (as specified): anything below INT8 is expressed through
 * AVX512-VNNI-INT8 (the upstream kernel KA's own nibble->dpbssd decode);
 * the BF16 side runs the AVX512-FP32 emulation (KT_DOT_BF16 fallback).
 *
 * One entry point computes both stages serially per expert-row: stage 1
 * (gate + up gemms on KA) -> fused silu(gate)*up hadamard in a compact
 * fp32 intermediate (I floats per row, cache-hot) -> stage 2 (down gemm on
 * KB), with no pool round-trips or re-dispatch between stages. The kernel
 * bodies reuse the validated per-node entries (integer_mat_mul /
 * float_mat_vec) so the decode math is identical to the single-node paths.
 */
#ifndef CPUINFER_OPERATOR_AMX_LA_AMX_FUSED_HPP
#define CPUINFER_OPERATOR_AMX_LA_AMX_FUSED_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "amx_kernels.hpp"
#include "amx_raw_kernels.hpp"
#include "../../gguf/dequant.hpp"  // fp32_to_bf16_ggml for the fused intermediate

namespace amx {

// Stage-pair codes (must match the python config values).
static constexpr int FUSED_NODE_INT4 = 0;
static constexpr int FUSED_NODE_INT8 = 1;
static constexpr int FUSED_NODE_BF16 = 2;

// Two-stage fused decode. Raw pointers (no ownership); the caller supplies
// the packed weight buffers (in the node kernels' production layouts) and the
// scratch vectors g/u/h (each m*I floats).
//
// Accumulator spec (AVX512-VNNI-INT8 definitions):
//   F4x8  (0->1): stage-1 widens the 4-bit codes into an INT16 staging
//                 buffer (signed bytes of nibble<<4), then folds them into
//                 the VNNI-int8 dot (dpbssd, INT32 accumulator) with the
//                 per-row scale applied ONCE at the end; stage-2 is the
//                 plain int8 dpbssd.
//   F8x16 (1->2): stage-1 plain int8 dpbssd; stage-2 runs the BF16 buffer
//                 dot (KT_DOT_BF16, AVX512-FP32 emulation) with the int8
//                 scale folded in as the bounding factor.
//   F4x16 (0->2): the int4 stage-1 as F4x8 + the BF16 stage-2 as F8x16.
template <class KA, class KB>
struct FusedTwoStage {
  using UA = typename KA::BufferA;
  using UB = typename KA::BufferB;
  using UC = typename KA::BufferC;
  using DA = typename KB::BufferA;
  using DB = typename KB::BufferB;
  using DC = typename KB::BufferC;

  // ==========================================================================
  // The VNNI accumulation definition, implemented self-contained (CLX-safe,
  // no VBMI):
  //   int8 x int8  -> int16 staging (cvtepi8 widen)
  //   int16 x int16 -> int32 accumulate (madd_epi16)
  // The 4-bit stage converts the packed nibbles to the signed value bytes
  // (nibble<<4) first, then uses the same madd chain — the int4 values must
  // never be multiplied raw at int8 precision (the historic failure mode).
  // ==========================================================================

  // 32 packed nibble bytes -> 64 signed value bytes (nibble<<4).
  static inline __m512i nibbles_to_values(__m512i packed) {
    const __m512i m = _mm512_set1_epi16(0x0F0F);
    __m512i even = _mm512_and_si512(packed, m);                     // lo nibbles per byte
    __m512i odd = _mm512_and_si512(_mm512_srli_epi16(packed, 4), m);  // hi nibbles per byte
    __m512i ve = _mm512_slli_epi16(even, 4);                        // value bytes (lo)
    __m512i vo = _mm512_slli_epi16(odd, 4);                         // value bytes (hi)
    // unpacklo/hi interleave the 16-byte groups as (0-7, 16-23, 32-39,
    // 48-55) and (8-15, 24-31, 40-47, 56-63). The 2-source permute indexes
    // b as lanes 8-15, so the sequential v[2j]=lo[j], v[2j+1]=hi[j] is
    // [l0: 0-3, 4-7 | h0: 8-11, 12-15 | l0: 16-19, 20-23 | h0: 24-27, 28-31].
    __m512i l0 = _mm512_unpacklo_epi8(ve, vo);
    __m512i h0 = _mm512_unpackhi_epi8(ve, vo);
    return _mm512_permutex2var_epi64(l0, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), h0);
  }

  // 64 int8 x 64 int8 -> 32 int32 lane-partials via the int16 staging.
  // The two 16-lane madds accumulate separately: acc0 = the low-32 pairs'
  // lanes, acc1 = the high-32 pairs' lanes (concatenation would truncate).
  static inline void madd_dot64(__m512i a64, __m512i b64, __m512i& acc0, __m512i& acc1) {
    __m512i al = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(a64));
    __m512i ah = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a64, 1));
    __m512i bl = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(b64));
    __m512i bh = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b64, 1));
    acc0 = _mm512_add_epi32(acc0, _mm512_madd_epi16(al, bl));
    acc1 = _mm512_add_epi32(acc1, _mm512_madd_epi16(ah, bh));
  }

  // Fused INT4 stage-1 decode (self-contained): packed nibbles -> value bytes
  // -> the int16-staged madd chain -> int32, the per-row scale once at the
  // end. A = plain int8 row-major activations (per-row d); B = the packed
  // per-row-int4 (per-row d). Outputs = [m][I] bf16 rows.
  static inline void int4_stage1(int m, int I, int H, UA* a, UB* b, UC* c, ggml_bf16_t* out) {
    (void)c;
    const int nth = KA::recommended_nth(I);
    for (int ith = 0; ith < nth; ith++) {
      auto [n_start, n_end] = KA::split_range_n(I, ith, nth);
      if (n_start >= n_end) continue;
      for (int m_i = 0; m_i < m; m_i++) {
        const float a_d = *a->get_scale(m, m_i);
        for (int n_out = n_start; n_out < n_end; n_out += 32) {
          __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
          for (int k_begin = 0; k_begin < H; k_begin += 64) {
            const int8_t* av = (const int8_t*)a->get_submat(m, H, m_i, k_begin);
            const uint8_t* packed = (const uint8_t*)b->get_submat(I, H, n_out, k_begin);
            __m512i b64 = nibbles_to_values(_mm512_loadu_si512(packed));
            __m512i a64 = _mm512_loadu_si512(av);
            madd_dot64(a64, b64, acc0, acc1);
          }
          const __m512 s = _mm512_set1_ps(a_d * b->d[n_out]);
          __m512 v0 = _mm512_mul_ps(s, _mm512_cvtepi32_ps(acc0));
          __m512 v1 = _mm512_mul_ps(s, _mm512_cvtepi32_ps(acc1));
          avx512_32xfp32_to_32xbf16(&v0, &v1, (__m512i*)(out + (size_t)m_i * I + n_out));
        }
      }
    }
  }

  // Fused BF16 stage-2 (KT_DOT_BF16, AVX512-FP32 emulation).
  static inline void bf16_stage2(int m, int I, int H, DA* a, DB* b, DC* c, ggml_bf16_t* out) {
    const int nth = KB::recommended_nth(H);
    for (int ith = 0; ith < nth; ith++) {
      float_mat_vec<KB, false>(m, H, I, a, b, c, ith, nth);
      c->to_mat(m, out, ith, nth);
    }
  }

  static void run(int m, int I, int H, ggml_bf16_t* x, UA* gate_a, UB* gate_b, UC* gate_c, UA* up_a, UB* up_b,
                  UC* up_c, DA* down_a, DB* down_b, DC* down_c, ggml_bf16_t* g, ggml_bf16_t* u, ggml_bf16_t* h,
                  ggml_bf16_t* out) {
    // ---- stage 1: gate + up on the upstream node KA ----
    // Maximum reuse: every stage runs the EXISTING kernel of its precision
    // (the wrapper only orchestrates). The int-family nodes use the
    // production integer_mat_mul + apply_scale on the correctly loaded
    // buffers; the BF16 node the float_mat_vec path. The dispatch is decided
    // at entry, one step before the GEMM loops begin.
    gate_a->from_mat(m, x, 0, 1);
    {
      const int nth = KA::recommended_nth(I);
      for (int ith = 0; ith < nth; ith++) {
        if constexpr (std::is_same_v<KA, amx::GemmKernel224BF16>) {
          float_mat_vec<KA, false>(m, I, H, gate_a, gate_b, gate_c, ith, nth);
        } else {
          integer_mat_mul<KA, false>(m, I, H, gate_a, gate_b, gate_c, ith, nth);
        }
        gate_c->to_mat(m, g, ith, nth);
      }
    }

    up_a->from_mat(m, x, 0, 1);
    {
      const int nth = KA::recommended_nth(I);
      for (int ith = 0; ith < nth; ith++) {
        if constexpr (std::is_same_v<KA, amx::GemmKernel224BF16>) {
          float_mat_vec<KA, false>(m, I, H, up_a, up_b, up_c, ith, nth);
        } else {
          integer_mat_mul<KA, false>(m, I, H, up_a, up_b, up_c, ith, nth);
        }
        up_c->to_mat(m, u, ith, nth);
      }
    }

    // ---- fused silu(gate) * up ----
    for (int i = 0; i < m * I; i++) {
      // silu(v) = v * sigmoid(v); fp32 math, bf16 intermediate (as production)
      const uint32_t gb = ((uint32_t)g[i].bits & 0xFFFFu) << 16;
      const uint32_t ub = ((uint32_t)u[i].bits & 0xFFFFu) << 16;
      float gv, uv;
      memcpy(&gv, &gb, 4);
      memcpy(&uv, &ub, 4);
      h[i] = kt::gguf::fp32_to_bf16_ggml(gv / (1.0f + std::exp(-gv)) * uv);
    }

    // ---- stage 2: down on the downstream node KB ----
    down_a->from_mat(m, h, 0, 1);
    {
      const int nth = KB::recommended_nth(H);
      for (int ith = 0; ith < nth; ith++) {
        if constexpr (std::is_same_v<KB, amx::GemmKernel224BF16>) {
          float_mat_vec<KB, false>(m, H, I, down_a, down_b, down_c, ith, nth);
        } else {
          integer_mat_mul<KB, false>(m, H, I, down_a, down_b, down_c, ith, nth);
        }
        down_c->to_mat(m, out, ith, nth);
      }
    }
  }
};

}  // namespace amx

#endif  // CPUINFER_OPERATOR_AMX_LA_AMX_FUSED_HPP