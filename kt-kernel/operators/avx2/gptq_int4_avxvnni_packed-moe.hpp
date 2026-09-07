/**
 * @Description  : AVX-VNNI-256 packed GPTQ-Int4 MoE operator (int4 weights stay resident)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Same GEMM/MoE structure as gptq_int4_avxvnni-moe.hpp: the base-class
 * pipeline runs gate, up and down as three separate GEMMs, and SwiGLU is
 * applied by the base class (plain silu / swiglu_limit / swigluoai all come
 * from there). The difference is weight residency and the inner kernel:
 *
 *   The pre-unpacked backends expand every weight into a persistent int8
 *   [N, K] copy at load time (~1.05 B/weight incl. scales and sums). This
 *   backend keeps the GPTQ int4 nibbles packed for the whole lifetime of the
 *   model — ~0.56 B/weight (incl. scales + c2), about half the footprint:
 *
 *     BufferB per projection: qweight [N, K/8] int32 (transposed view of the
 *     GPTQ source, still 0.5 B/weight) + scales [N, num_groups] fp32 + c2
 *     [N, num_groups] fp32 correction table (c2 = 128 * scale * rowsum,
 *     computed once at load time with integer nibble sums).
 *
 *   The nibble -> (nibble - 8) int8 unpacking happens inside the GEMM: each
 *   8-column block is unpacked once into a staging buffer that is reused
 *   across all activation rows (unpack once, use m times) — L1-resident
 *   16KB stack for k <= 2048, auto-sized heap (L2-resident) beyond, so no
 *   model dimension is hard-limited. For Qwen3-30B-A3B the expert weights
 *   shrink from ~30.4 GB (pre-unpacked) to ~16.3 GB, fitting where the
 *   pre-unpacked VNNI backend runs out of memory.
 *
 * dpbusd math: activations are quantized per group to biased uint8 (zp=128)
 * exactly once per expert projection — when the base packs the input into
 * BufferA (biased u8 + per-group fp32 scales), so the GEMM consumes
 * pre-quantized rows for decode and prefill alike; the +128 bias in the
 * dpbusd result is removed exactly with the per-group weight row-sum
 * carried in c2, and the per-group int32 partial sums are rescaled into
 * fp32 accumulators right after each group (scales differ per group, so
 * accumulation must not straddle groups).
 **/
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_PACKED_MOE_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_PACKED_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_AVXVNNI256_PACKED_TARGET __attribute__((target("avx2,avxvnni,fma,f16c")))
#else
#define KT_AVXVNNI256_PACKED_TARGET
#endif

namespace avxvnni_packed {

static constexpr int MAX_SUPPORTED_GROUP_SIZE = 2048;
static constexpr int MAX_TMP_ROW = 2048;  // staging threshold: k <= 2048 stages in L1 stack buffers, larger k in heap

KT_AVXVNNI256_PACKED_TARGET
static inline float hmax_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 m = _mm_max_ps(lo, hi);
  m = _mm_max_ps(m, _mm_permute_ps(m, 0x4E));  // swap halves
  m = _mm_max_ps(m, _mm_permute_ps(m, 0xB1));  // swap pairs
  return _mm_cvtss_f32(m);
}

// Unpack 4 consecutive int32 words (32 nibbles, GPTQ LSB-first) into one __m256i
// of 32 s8 lanes = (nibble - 8), ready for dpbusd's b operand. No row-sum
// accumulation here (the load-time c2 table carries it).
KT_AVXVNNI256_PACKED_TARGET
static inline __m256i unpack_4words_to_s8_nosad(const int32_t* qwords) {
  const __m128i MASK = _mm_set1_epi8(0x0F);
  __m128i w = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qwords));
  __m128i even = _mm_and_si128(w, MASK);
  __m128i odd = _mm_and_si128(_mm_srli_epi16(w, 4), MASK);
  __m128i lo16 = _mm_unpacklo_epi8(even, odd);
  __m128i hi16 = _mm_unpackhi_epi8(even, odd);
  __m256i nibs = _mm256_inserti128_si256(_mm256_castsi128_si256(lo16), hi16, 1);
  return _mm256_sub_epi8(nibs, _mm256_set1_epi8(8));
}

KT_AVXVNNI256_PACKED_TARGET
static inline float hsum_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}

// Per-group activation quantization for a bf16 source row (run once per
// expert projection when the base packs the input row into BufferA):
// per-group scale = max_abs/127 (ties-to-even, zp=128, clamp 0-255, all-zero
// group -> memset zp128 + scale 1.0f; the kernel's c2 correction cancels the
// 128*bias term exactly, so the all-zero case needs no special handling in
// the GEMM). Contract: k % group_size == 0 (no tail).
KT_AVXVNNI256_PACKED_TARGET
static inline void quantize_bf16_to_u8_pg(const ggml_bf16_t* x, uint8_t* out_u8, float* scales_out, int k,
                                          int group_size) {
  const __m256 ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  const int num_groups = k / group_size;
  for (int g = 0; g < num_groups; g++) {
    const ggml_bf16_t* sg = x + (int64_t)g * group_size;
    float max_abs = 0.0f;
    int i = 0;
    for (; i + 8 <= group_size; i += 8) {
      __m256 v = avx2::load_bf16_to_fp32(sg + i);
      max_abs = std::max(max_abs, hmax_ps(_mm256_and_ps(v, ABS_MASK)));
    }
    for (; i < group_size; i++) {
      max_abs = std::max(max_abs, std::fabs(GGML_BF16_TO_FP32(sg[i])));
    }
    if (max_abs <= std::numeric_limits<float>::min()) {
      std::memset(out_u8 + (int64_t)g * group_size, 128, group_size);
      scales_out[g] = 1.0f;
      continue;
    }
    const float scale = max_abs / 127.0f;
    scales_out[g] = scale;
    const float inv = 1.0f / scale;
    const __m256 inv_v = _mm256_set1_ps(inv);
    __m256i zp = _mm256_set1_epi32(128);
    i = 0;
    uint8_t* dg = out_u8 + (int64_t)g * group_size;
    for (; i + 8 <= group_size; i += 8) {
      __m256 v = avx2::load_bf16_to_fp32(sg + i);
      __m256 scaled = _mm256_mul_ps(v, inv_v);
      __m256 rounded = _mm256_round_ps(scaled, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      __m256i vi = _mm256_cvtps_epi32(rounded);
      vi = _mm256_add_epi32(vi, zp);
      vi = _mm256_max_epi32(vi, _mm256_setzero_si256());
      vi = _mm256_min_epi32(vi, _mm256_set1_epi32(255));
      __m128i lo32 = _mm256_castsi256_si128(vi);
      __m128i hi32 = _mm256_extracti128_si256(vi, 1);
      __m128i packed16 = _mm_packus_epi32(lo32, hi32);
      __m128i packed8 = _mm_packus_epi16(packed16, packed16);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dg + i), packed8);
    }
    for (; i < group_size; i++) {
      int q = (int)std::lrint(GGML_BF16_TO_FP32(sg[i]) * inv) + 128;
      dg[i] = (uint8_t)std::clamp(q, 0, 255);
    }
  }
}

constexpr int IKJ_N8_NR = 8;

// Per-jb staging: unpack the 8 weight rows of one output block into tmp (8
// rows of k bytes each, row stride tmp_stride) once, reused by all m
// activation rows.
KT_AVXVNNI256_PACKED_TARGET
static inline void unpack_jb_to_tmp(const int32_t* const* qwr, uint8_t* tmp_base, size_t tmp_stride, int num_groups,
                                    int lanes_per_group, int group_size, int words_per_group) {
  for (int j = 0; j < IKJ_N8_NR; j++) {
    const int32_t* qwj = qwr[j];
    uint8_t* tmpj = tmp_base + (size_t)j * tmp_stride;
    for (int g_idx = 0; g_idx < num_groups; g_idx++) {
      for (int lane = 0; lane < lanes_per_group; lane++) {
        __m256i u = unpack_4words_to_s8_nosad(qwj + g_idx * words_per_group + lane * 4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmpj + g_idx * group_size + lane * 32), u);
      }
    }
  }
}

// Split [0, n) into nth tasks at 8-column jb-block granularity instead of
// avx2::split_range's element granularity. n % 8 == 0 is guaranteed by the
// BufferB k-alignment contract (hidden_size and intermediate_size each serve
// as the k of one projection), and nth <= n/8 for the recommended_nth this
// backend uses — so every task boundary lands on a jb-block edge, no block
// ever straddles tasks, and every task owns at least one full block. The
// GEMM's per-column stores are therefore unconditional.
static inline std::pair<int, int> split_range_jb(int n, int ith, int nth) {
  const int jb_total = n / IKJ_N8_NR;
  const int per = jb_total / nth;
  const int rem = jb_total % nth;
  const int start = (ith * per + std::min(ith, rem)) * IKJ_N8_NR;
  return {start, start + (per + (ith < rem ? 1 : 0)) * IKJ_N8_NR};
}

struct GemmKernelAVXVNNI256PackedGPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 8;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static int recommended_nth(int n) { return std::max(1, n / N_BLOCK); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range_jb(n, ith, nth); }

  // Quantized activations: the base packs each expert's input rows into
  // BufferA with a single from_mat call per expert (before any GEMM task is
  // dispatched — the work-stealing jobs are synchronous), so the per-group
  // quantization runs exactly once per expert projection, for decode and
  // prefill alike. The GEMM consumes the pre-quantized rows.
  struct BufferA {
    uint8_t* data = nullptr;  // [max_m, k] biased u8 (zp=128)
    float* scales = nullptr;  // [max_m, k/group_size] per-group fp32
    size_t max_m = 0;
    size_t k = 0;
    int group_size = 128;

    BufferA() = default;
    BufferA(size_t m, size_t k_, int gs, void* ptr) { init(m, k_, gs, ptr); }

    static size_t required_size(size_t m, size_t k, int gs) { return m * k + m * (k / gs) * sizeof(float); }

    void init(size_t m, size_t k_, int gs, void* ptr) {
      k = k_;
      group_size = gs;
      max_m = m;
      if (group_size <= 0 || (k % group_size) != 0) {
        throw std::runtime_error(
            "AVX-VNNI-256 packed GPTQ INT4 requires k to be a positive multiple of group_size "
            "(per-group activation quantization)");
      }
      set_data(ptr);
    }

    void set_data(void* ptr) {
      data = (uint8_t*)ptr;
      scales = ptr == nullptr ? nullptr : (float*)((uint8_t*)ptr + max_m * k);
    }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      int m_start = 0, m_end = m;
      if (!(ith == 0 && nth == 1)) {
        auto [s, e] = avx2::split_range(m, ith, nth);
        m_start = s;
        m_end = e;
      }
      const int ng = (int)(k / group_size);
      for (int mi = m_start; mi < m_end; ++mi) {
        quantize_bf16_to_u8_pg(src + (size_t)mi * k, data + (size_t)mi * k, scales + (size_t)mi * ng, (int)k,
                               group_size);
      }
    }
  };

  // Packed-int4 resident weights (0.5 B/weight + scales + c2, no int8 copy).
  struct BufferB {
    int32_t* qweight = nullptr;  // [N, K/8] int32 (transposed GPTQ source view)
    float* scales = nullptr;     // [N, num_groups] fp32
    float* c2 = nullptr;         // [N, num_groups] fp32 (128 * scale * rowsum)
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;
    int k_packed = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) { init(n_, k_, gs, ptr); }

    static size_t required_size(size_t n, size_t k, int gs) {
      const size_t num_groups = k / gs;
      return (k / 8) * n * sizeof(int32_t) + 2 * num_groups * n * sizeof(float);
    }

    void init(size_t n_, size_t k_, int gs, void* ptr) {
      n = (int)n_;
      k = (int)k_;
      group_size = gs;
      if (group_size <= 0 || (group_size % 32) != 0) {
        throw std::runtime_error("AVX-VNNI-256 packed GPTQ INT4 requires group_size to be a positive multiple of 32");
      }
      if (group_size > MAX_SUPPORTED_GROUP_SIZE) {
        throw std::runtime_error("AVX-VNNI-256 packed GPTQ INT4 requires group_size <= 2048");
      }
      if ((k % 8) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX-VNNI-256 packed GPTQ INT4 requires k to be divisible by both 8 and group_size");
      }
      k_packed = k / 8;
      num_groups = k / group_size;
      qweight = (int32_t*)ptr;
      scales = (float*)((uint8_t*)ptr + (size_t)k_packed * n * sizeof(int32_t));
      c2 = (float*)((uint8_t*)scales + (size_t)num_groups * n * sizeof(float));
    }

    // Transposing gather from the GPTQ source layout for this task's column
    // slice [n_start, n_end): for each local row ni, read the k_packed int32
    // source words at src_qweight[w * n + ni] (source [K/8, N]) and the fp32
    // scales at src_scales[g * n + ni], storing row-major here (nibbles stay
    // packed — a pure int32 copy, no format conversion). The c2 correction
    // table is computed per row: rowsum = sum(nibble) - 8*group_size over the
    // raw nibbles (int64), c2 = 128 * scale * rowsum.
    void from_mat(const uint32_t* src_qweight, const float* src_scales, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);
      const int words_per_group = group_size / 8;
      for (int ni = n_start; ni < n_end; ++ni) {
        int32_t* dst_row = qweight + (size_t)ni * k_packed;
        for (int w = 0; w < k_packed; ++w) {
          dst_row[w] = (int32_t)src_qweight[(size_t)w * n + ni];
        }
        float* sc_row = scales + (size_t)ni * num_groups;
        float* c2_row = c2 + (size_t)ni * num_groups;
        int word = 0;
        for (int g = 0; g < num_groups; ++g) {
          int64_t nibble_sum = 0;
          for (int wr = 0; wr < words_per_group; ++wr, ++word) {
            const uint32_t packed = (uint32_t)dst_row[word];
            for (int nb = 0; nb < 8; ++nb) {
              nibble_sum += (packed >> (nb * 4)) & 0xF;
            }
          }
          const int64_t rowsum = nibble_sum - 8 * (int64_t)group_size;  // sum(nibble - 8)
          const float sc = src_scales[(size_t)g * n + ni];
          sc_row[g] = sc;
          c2_row[g] = 128.0f * sc * (float)rowsum;
        }
      }
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) { return m * n * sizeof(float); }

    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; ++mi) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          avx2::store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        }
        for (; j < n_end; ++j) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

// ---------------------------------------------------------------------------
// Core GEMM — the packed counterpart of gemm_gptq_sym_int4_avxvnni256.
// Task = one column block [n_start, n_end) from (ith, nth). Activations
// arrive pre-quantized in BufferA (biased u8 + per-group fp32 scales,
// quantized exactly once per expert projection when the base packs the
// input). Per 8-column jb block the 8 weight rows are unpacked into the
// staging buffer once and reused across all m activation rows; 8 dpbusd
// chains share one activation load, each group's int32 partials are
// rescaled into fp32 accumulators immediately, and the +128 weight bias is
// removed exactly with the c2 table:
//   out[m, col] = sum_g a_scale[m, g] * w_scale[col, g] * dot_g
//                 - sum_g a_scale[m, g] * c2[col, g]
// Task ranges are split at 8-column jb-block granularity (split_range_jb), so
// blocks never straddle tasks and every task owns whole blocks — the inner
// stores are unconditional.
// ---------------------------------------------------------------------------
KT_AVXVNNI256_PACKED_TARGET
static inline void gemm_gptq_sym_int4_packed_avxvnni256(int m, int n, int k,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferA& a,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferB& b,
                                                        GemmKernelAVXVNNI256PackedGPTQInt4::BufferC& c, int ith,
                                                        int nth) {
  auto [n_start, n_end] = split_range_jb(n, ith, nth);
  const int num_groups = b.num_groups;
  const int group_size = b.group_size;
  const int lanes_per_group = group_size / 32;
  const int words_per_group = group_size / 8;

  // Weight staging scales with k: k <= MAX_TMP_ROW takes the fast stack path
  // (16KB staging, L1-resident); larger k falls back to a per-thread grow-only
  // heap buffer (L2-resident, one allocation per worker thread, reused across
  // tasks — no per-task malloc), so no model dimension is hard-limited. The
  // inner loops are identical for both paths. Activations arrive pre-quantized
  // in BufferA (quantized once per expert projection at pack time — no
  // per-block re-quantization).
  alignas(64) uint8_t tmp_stack[IKJ_N8_NR][MAX_TMP_ROW];
  const size_t tmp_stride = (k <= MAX_TMP_ROW) ? MAX_TMP_ROW : (size_t)k;
  uint8_t* tmp_base;
  if (k <= MAX_TMP_ROW) {
    tmp_base = &tmp_stack[0][0];
  } else {
    static thread_local std::vector<uint8_t> heap_staging;
    const size_t needed = IKJ_N8_NR * (size_t)k;
    if (heap_staging.size() < needed) {
      heap_staging.resize(needed);
    }
    tmp_base = heap_staging.data();
  }
  const int sx_stride = k / group_size;
  const int32_t* qwr[IKJ_N8_NR];
  const float* scr[IKJ_N8_NR];
  uint8_t* tmpj[IKJ_N8_NR];

  for (int jb = n_start / IKJ_N8_NR; jb < n_end / IKJ_N8_NR; ++jb) {
    const int j_blk = jb * IKJ_N8_NR;
    for (int j = 0; j < IKJ_N8_NR; j++) {
      qwr[j] = b.qweight + (int64_t)(j_blk + j) * b.k_packed;
      scr[j] = b.scales + (int64_t)(j_blk + j) * num_groups;
      tmpj[j] = tmp_base + (size_t)j * tmp_stride;
    }
    unpack_jb_to_tmp(qwr, tmp_base, tmp_stride, num_groups, lanes_per_group, group_size, words_per_group);
    for (int mi = 0; mi < m; ++mi) {
      const float* sxa = a.scales + (size_t)mi * sx_stride;
      const uint8_t* xm = a.data + (size_t)mi * k;
      __m256 fp0 = _mm256_setzero_ps(), fp1 = _mm256_setzero_ps(), fp2 = _mm256_setzero_ps(), fp3 = _mm256_setzero_ps(),
             fp4 = _mm256_setzero_ps(), fp5 = _mm256_setzero_ps(), fp6 = _mm256_setzero_ps(), fp7 = _mm256_setzero_ps();
      float rc0 = 0.0f, rc1 = 0.0f, rc2 = 0.0f, rc3 = 0.0f, rc4 = 0.0f, rc5 = 0.0f, rc6 = 0.0f, rc7 = 0.0f;
      for (int g = 0; g < num_groups; ++g) {
        const float sg = sxa[g];
        __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256(),
                acc3 = _mm256_setzero_si256(), acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256(),
                acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();
        const uint8_t* xg = xm + g * group_size;
        for (int lane = 0; lane < lanes_per_group; lane++) {
          const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xg + lane * 32));
          const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[0] + g * group_size + lane * 32));
          acc0 = _mm256_dpbusd_epi32(acc0, av, b0);
          const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[1] + g * group_size + lane * 32));
          acc1 = _mm256_dpbusd_epi32(acc1, av, b1);
          const __m256i b2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[2] + g * group_size + lane * 32));
          acc2 = _mm256_dpbusd_epi32(acc2, av, b2);
          const __m256i b3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[3] + g * group_size + lane * 32));
          acc3 = _mm256_dpbusd_epi32(acc3, av, b3);
          const __m256i b4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[4] + g * group_size + lane * 32));
          acc4 = _mm256_dpbusd_epi32(acc4, av, b4);
          const __m256i b5 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[5] + g * group_size + lane * 32));
          acc5 = _mm256_dpbusd_epi32(acc5, av, b5);
          const __m256i b6 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[6] + g * group_size + lane * 32));
          acc6 = _mm256_dpbusd_epi32(acc6, av, b6);
          const __m256i b7 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tmpj[7] + g * group_size + lane * 32));
          acc7 = _mm256_dpbusd_epi32(acc7, av, b7);
        }
        fp0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0), _mm256_set1_ps(sg * scr[0][g]), fp0);
        fp1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1), _mm256_set1_ps(sg * scr[1][g]), fp1);
        fp2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc2), _mm256_set1_ps(sg * scr[2][g]), fp2);
        fp3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc3), _mm256_set1_ps(sg * scr[3][g]), fp3);
        fp4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc4), _mm256_set1_ps(sg * scr[4][g]), fp4);
        fp5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc5), _mm256_set1_ps(sg * scr[5][g]), fp5);
        fp6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc6), _mm256_set1_ps(sg * scr[6][g]), fp6);
        fp7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc7), _mm256_set1_ps(sg * scr[7][g]), fp7);
        rc0 += sg * b.c2[(int64_t)(j_blk + 0) * num_groups + g];
        rc1 += sg * b.c2[(int64_t)(j_blk + 1) * num_groups + g];
        rc2 += sg * b.c2[(int64_t)(j_blk + 2) * num_groups + g];
        rc3 += sg * b.c2[(int64_t)(j_blk + 3) * num_groups + g];
        rc4 += sg * b.c2[(int64_t)(j_blk + 4) * num_groups + g];
        rc5 += sg * b.c2[(int64_t)(j_blk + 5) * num_groups + g];
        rc6 += sg * b.c2[(int64_t)(j_blk + 6) * num_groups + g];
        rc7 += sg * b.c2[(int64_t)(j_blk + 7) * num_groups + g];
      }
      // One hsum + correction per (row, column); task ranges are jb-block
      // aligned (split_range_jb), so every block is full and stores are
      // unconditional.
      float* c_row = c.data + (size_t)mi * n;
      c_row[j_blk + 0] = hsum_ps(fp0) - rc0;
      c_row[j_blk + 1] = hsum_ps(fp1) - rc1;
      c_row[j_blk + 2] = hsum_ps(fp2) - rc2;
      c_row[j_blk + 3] = hsum_ps(fp3) - rc3;
      c_row[j_blk + 4] = hsum_ps(fp4) - rc4;
      c_row[j_blk + 5] = hsum_ps(fp5) - rc5;
      c_row[j_blk + 6] = hsum_ps(fp6) - rc6;
      c_row[j_blk + 7] = hsum_ps(fp7) - rc7;
    }
  }
}

}  // namespace avxvnni_packed

template <class T = avxvnni_packed::GemmKernelAVXVNNI256PackedGPTQInt4>
class AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP<T>>;
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

  AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP() = default;
  AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("AVX-VNNI-256 packed GPTQ_INT4 backend requires CPU support for avx_vnni");
    }
#endif
    const auto& qc = config_.quant_config;
    if (qc.group_size == 0 || (qc.group_size % 32) != 0) {
      throw std::runtime_error("AVX-VNNI-256 packed GPTQ_INT4 requires group_size to be a positive multiple of 32");
    }
    if (qc.group_size > avxvnni_packed::MAX_SUPPORTED_GROUP_SIZE) {
      throw std::runtime_error("AVX-VNNI-256 packed GPTQ_INT4 requires group_size <= 2048");
    }
    // No hidden_size / intermediate_size limit: the kernel's staging buffers
    // scale with k (L1-resident stack up to 2048, heap beyond — see the core
    // GEMM).
    printf("Created AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP %d at numa %d (group_size=%d, packed int4 resident)\n",
           tp_part_idx, numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }

  ~AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP() = default;

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avxvnni_packed::gemm_gptq_sym_int4_packed_avxvnni256(m, config_.intermediate_size, config_.hidden_size, *ba, *bb,
                                                         *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avxvnni_packed::gemm_gptq_sym_int4_packed_avxvnni256(m, config_.hidden_size, config_.intermediate_size,
                                                         *down_ba_[expert_idx], *down_bb_[expert_idx],
                                                         *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("GPTQ INT4 MOE requires scale pointers.");
    }

    int gate_up_k = config_.hidden_size;
    int gate_up_n = config_.intermediate_size;
    size_t qw_elems = (size_t)(gate_up_k / 8) * gate_up_n;
    size_t sc_elems = (size_t)(gate_up_k / group_size) * gate_up_n;

    int nth = T::recommended_nth(gate_up_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, qw_elems, sc_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          gate_bb_[expert_idx]->from_mat((uint32_t*)config_.gate_proj + logical * qw_elems,
                                         (float*)config_.gate_scale + logical * sc_elems, ith, nth);

          up_bb_[expert_idx]->from_mat((uint32_t*)config_.up_proj + logical * qw_elems,
                                       (float*)config_.up_scale + logical * sc_elems, ith, nth);
        },
        nullptr);

    int down_k = config_.intermediate_size;
    int down_n = config_.hidden_size;
    size_t down_qw_elems = (size_t)(down_k / 8) * down_n;
    size_t down_sc_elems = (size_t)(down_k / group_size) * down_n;

    nth = T::recommended_nth(down_n);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, down_qw_elems, down_sc_elems](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          down_bb_[expert_idx]->from_mat((uint32_t*)config_.down_proj + logical * down_qw_elems,
                                         (float*)config_.down_scale + logical * down_sc_elems, ith, nth);
        },
        nullptr);
  }

  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    (void)gpu_tp_count;
    (void)expert_id;
    (void)full_config;
    (void)w13_weight_ptrs;
    (void)w2_weight_ptrs;
    throw std::runtime_error("AVX-VNNI-256 packed GPTQ INT4 write_weights_to_buffer not yet implemented");
  }
};

template <typename K>
class TP_MOE<AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP<K>>
    : public TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_GPTQ_INT4_PACKED_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0) {
      throw std::runtime_error("GPTQ INT4 requires group_size > 0");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const int full_intermediate = config.intermediate_size;
    const int full_hidden = config.hidden_size;

    const int gate_up_k_packed = full_hidden / 8;
    const int gate_up_num_groups = full_hidden / group_size;
    const size_t full_gate_up_qw_elems = (size_t)gate_up_k_packed * full_intermediate;
    const size_t full_gate_up_sc_elems = (size_t)gate_up_num_groups * full_intermediate;

    const int down_k_packed = full_intermediate / 8;
    const int down_num_groups = full_intermediate / group_size;
    const size_t full_down_qw_elems = (size_t)down_k_packed * full_hidden;
    const size_t full_down_sc_elems = (size_t)down_num_groups * full_hidden;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const int tp_intermediate = tpc.intermediate_size;

      const size_t tp_gate_up_qw_elems = (size_t)gate_up_k_packed * tp_intermediate;
      const size_t tp_gate_up_sc_elems = (size_t)gate_up_num_groups * tp_intermediate;

      tpc.gate_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.up_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];

      const int tp_down_k_packed = tp_intermediate / 8;
      const int tp_down_num_groups = tp_intermediate / group_size;
      const size_t tp_down_qw_elems = (size_t)tp_down_k_packed * full_hidden;
      const size_t tp_down_sc_elems = (size_t)tp_down_num_groups * full_hidden;

      tpc.down_proj = new uint32_t[tpc.expert_num * tp_down_qw_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_down_sc_elems];

      const int gate_up_n_offset = i * tp_intermediate;
      const int down_k_offset_packed = i * tp_down_k_packed;
      const int down_group_offset = i * tp_down_num_groups;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            const uint32_t* gate_qw_src;
            const uint32_t* up_qw_src;
            const uint32_t* down_qw_src;
            const float* gate_sc_src;
            const float* up_sc_src;
            const float* down_sc_src;

            if (use_per_expert_ptrs) {
              gate_qw_src = (const uint32_t*)config.gate_projs[0][expert_id];
              up_qw_src = (const uint32_t*)config.up_projs[0][expert_id];
              down_qw_src = (const uint32_t*)config.down_projs[0][expert_id];
              gate_sc_src = (const float*)config.gate_scales[0][expert_id];
              up_sc_src = (const float*)config.up_scales[0][expert_id];
              down_sc_src = (const float*)config.down_scales[0][expert_id];
            } else {
              gate_qw_src = (const uint32_t*)config.gate_proj + expert_id * full_gate_up_qw_elems;
              up_qw_src = (const uint32_t*)config.up_proj + expert_id * full_gate_up_qw_elems;
              down_qw_src = (const uint32_t*)config.down_proj + expert_id * full_down_qw_elems;
              gate_sc_src = (const float*)config.gate_scale + expert_id * full_gate_up_sc_elems;
              up_sc_src = (const float*)config.up_scale + expert_id * full_gate_up_sc_elems;
              down_sc_src = (const float*)config.down_scale + expert_id * full_down_sc_elems;
            }

            uint32_t* gate_qw_dst = (uint32_t*)tpc.gate_proj + expert_id * tp_gate_up_qw_elems;
            uint32_t* up_qw_dst = (uint32_t*)tpc.up_proj + expert_id * tp_gate_up_qw_elems;
            float* gate_sc_dst = (float*)tpc.gate_scale + expert_id * tp_gate_up_sc_elems;
            float* up_sc_dst = (float*)tpc.up_scale + expert_id * tp_gate_up_sc_elems;

            for (int kr = 0; kr < gate_up_k_packed; ++kr) {
              std::memcpy(gate_qw_dst + kr * tp_intermediate, gate_qw_src + kr * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(uint32_t));
              std::memcpy(up_qw_dst + kr * tp_intermediate, up_qw_src + kr * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(uint32_t));
            }

            for (int g = 0; g < gate_up_num_groups; ++g) {
              std::memcpy(gate_sc_dst + g * tp_intermediate, gate_sc_src + g * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(float));
              std::memcpy(up_sc_dst + g * tp_intermediate, up_sc_src + g * full_intermediate + gate_up_n_offset,
                          (size_t)tp_intermediate * sizeof(float));
            }

            uint32_t* down_qw_dst = (uint32_t*)tpc.down_proj + expert_id * tp_down_qw_elems;
            for (int kr = 0; kr < tp_down_k_packed; ++kr) {
              std::memcpy(down_qw_dst + kr * full_hidden, down_qw_src + (down_k_offset_packed + kr) * full_hidden,
                          (size_t)full_hidden * sizeof(uint32_t));
            }

            float* down_sc_dst = (float*)tpc.down_scale + expert_id * tp_down_sc_elems;
            for (int g = 0; g < tp_down_num_groups; ++g) {
              std::memcpy(down_sc_dst + g * full_hidden, down_sc_src + (down_group_offset + g) * full_hidden,
                          (size_t)full_hidden * sizeof(float));
            }
          },
          nullptr);
    });

    pool->dispense_backend()->do_numa_job([&, this](int i) { tps[i]->load_weights(); });

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint32_t*)tpc.gate_proj;
      delete[] (uint32_t*)tpc.up_proj;
      delete[] (uint32_t*)tpc.down_proj;
      delete[] (float*)tpc.gate_scale;
      delete[] (float*)tpc.up_scale;
      delete[] (float*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    (void)gpu_tp_count;
    (void)expert_id;
    (void)w13_weight_ptrs;
    (void)w13_scale_ptrs;
    (void)w2_weight_ptrs;
    (void)w2_scale_ptrs;
    throw std::runtime_error("AVX-VNNI-256 packed GPTQ INT4 write_weight_scale_to_buffer not yet implemented");
  }
};

#undef KT_AVXVNNI256_PACKED_TARGET

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT4_AVXVNNI_PACKED_MOE_H
