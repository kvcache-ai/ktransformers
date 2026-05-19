/**
 * @Description  : AVX2 MXFP4 MoE operator for DeepSeek V4 native inference
 * SPDX-License-Identifier: Apache-2.0
 *
 * MXFP4 stores FP4 E2M1 weights nibble-packed (2 values per byte), plus
 * per-group-32 FP32 scales. This AVX2 backend dequantizes FP4→FP32 via SSSE3
 * PSHUFB lookup tables, then accumulates with BF16 activations using
 * _mm256_fmadd_ps. No AVX-512 required.
 *
 * FP4 E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
#define CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H

#include <immintrin.h>
#include <tmmintrin.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"

namespace avx2 {

// ============================================================================
// AVX2 MXFP4 GemmKernel
// Weights: FP4 E2M1 nibble-packed, per-group-32 FP32 scales
// Activations: BF16; Output: FP32
// ============================================================================
struct GemmKernelAVX2MXFP4 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}
  static int recommended_nth(int n) { return std::max(1, div_up(n, N_BLOCK)); }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }
  static std::string name() { return "AVX2_MXFP4"; }

  // FP4 E2M1 → BF16 lookup tables (low byte and high byte of BF16)
  // Identical to amx/fp4-moe.hpp GemmKernel224MXFP4SmallKGroup LUTs
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,
                                                          0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,
                                                          0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};

  // Dequantize 4 packed bytes (= 8 FP4 E2M1 nibbles) → 8 FP32 values.
  //
  // Nibble order: packed[j] = {lo_nib = column 2j, hi_nib = column 2j+1}
  // Output order: columns 0, 1, 2, ..., 7 (sequential).
  //
  // Algorithm: SSSE3 PSHUFB table lookup → 8 BF16 values → BF16-to-FP32
  // (zero-extend uint16 to uint32, shift left 16).
  __attribute__((always_inline)) static inline __m256 fp4x8_to_fp32(const uint8_t* packed, __m128i lut_lo,
                                                                    __m128i lut_hi) {
    int32_t raw;
    std::memcpy(&raw, packed, 4);
    __m128i data = _mm_cvtsi32_si128(raw);
    __m128i lo_mask = _mm_set1_epi8(0x0F);

    // Extract lo nibbles (bits 0-3) and hi nibbles (bits 4-7)
    __m128i lo_nib = _mm_and_si128(data, lo_mask);
    __m128i hi_nib = _mm_and_si128(_mm_srli_epi16(data, 4), lo_mask);

    // PSHUFB: BF16 byte lookup for each set of nibbles
    __m128i lo_bf16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut_lo, lo_nib),
                                        _mm_shuffle_epi8(lut_hi, lo_nib));  // 4 BF16: cols {0,2,4,6}
    __m128i hi_bf16 = _mm_unpacklo_epi8(_mm_shuffle_epi8(lut_lo, hi_nib),
                                        _mm_shuffle_epi8(lut_hi, hi_nib));  // 4 BF16: cols {1,3,5,7}

    // Interleave: {col0,col1,col2,...,col7} in sequential order
    __m128i bf16_8 = _mm_unpacklo_epi16(lo_bf16, hi_bf16);  // 8 × BF16 in 128 bits

    // BF16 → FP32: zero-extend u16 → u32, then shift left 16
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_8), 16));
  }

  // Scalar dequant of a single nibble index (for tail handling)
  static inline float fp4_scalar(uint8_t nib) {
    const uint32_t bf16_bits = (uint32_t)fp4_bf16_lo[nib] | ((uint32_t)fp4_bf16_hi[nib] << 8);
    const uint32_t fp32_bits = bf16_bits << 16;
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
  }

  // ---- Buffer types --------------------------------------------------------

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0, k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, int /*group_size*/, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k, int /*group_size*/) { return m * k * sizeof(ggml_bf16_t); }
    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        auto [m_start, m_end] = split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k, (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  struct BufferB {
    uint8_t* b = nullptr;  // nibble-packed FP4 (may be nullptr in scale-only mode)
    float* d = nullptr;    // FP32 group scales
    int n = 0, k = 0, k_group_size = 0, k_group_count = 0;

    BufferB() = default;

    // Full allocation: b and d packed into a single aligned block.
    BufferB(int n_, int k_, int k_group_size_, void* ptr)
        : b((uint8_t*)ptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0)
        throw std::runtime_error("MXFP4 AVX2 requires k % group_size == 0 and k % 8 == 0");
      k_group_count = k / k_group_size;
      d = (float*)((uint8_t*)ptr + (size_t)n * k / 2);
    }

    // Scale-only: b is set externally from mmap'd safetensor data.
    BufferB(int n_, int k_, int k_group_size_, void* scale_ptr, std::nullptr_t)
        : b(nullptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0)
        throw std::runtime_error("MXFP4 AVX2 requires k % group_size == 0 and k % 8 == 0");
      k_group_count = k / k_group_size;
      d = (float*)scale_ptr;
    }

    static size_t required_size(size_t n, size_t k, int k_group_size) {
      return n * k / 2 + n * (k / k_group_size) * sizeof(float);
    }
    static size_t required_size_scale_only(size_t n, size_t k, int k_group_size) {
      return n * (k / k_group_size) * sizeof(float);
    }

    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      if (b == nullptr) return;
      auto [n_start, n_end] = split_range(n, ith, nth);
      const size_t row_bytes = (size_t)k / 2;
      std::memcpy(b + n_start * row_bytes, proj + n_start * row_bytes, (size_t)(n_end - n_start) * row_bytes);
    }
  };

  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0, n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : data((float*)ptr), max_m(m), n(n_) {}

    static size_t required_size(size_t m, size_t n) { return m * n * sizeof(float); }
    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        for (; j < n_end; j++) dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
      }
    }
  };
};

// ============================================================================
// Compute: M tokens × N output neurons, K inner dimension, group_size scale
//
// N-outer loop: each weight row loaded once and shared across M tokens.
// 4-token M-blocking for prefill: weight decode amortized over 4 activations.
// ============================================================================
static void gemm_mxfp4(int m, int n, int k, GemmKernelAVX2MXFP4::BufferA& a, GemmKernelAVX2MXFP4::BufferB& b,
                       GemmKernelAVX2MXFP4::BufferC& c, int ith, int nth) {
  // H3: Check weight buffer is not null
  if (b.b == nullptr) {
    throw std::runtime_error("gemm_mxfp4: weight buffer (b.b) is null");
  }
  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_count = b.k_group_count;
  const int group_size = b.k_group_size;
  const size_t row_bytes = (size_t)k / 2;

  const __m128i lut_lo = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_lo);
  const __m128i lut_hi = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_hi);

  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
    const float* b_scales = b.d + (size_t)ni * group_count;

    if (ni + 1 < n_end) {
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 64), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 128), _MM_HINT_T0);
      _mm_prefetch((const char*)(b.b + (size_t)(ni + 1) * row_bytes + 192), _MM_HINT_T0);
    }

    // 4-token blocked path: decode weight once, fmadd into 4 token accumulators
    int mi = 0;
    for (; mi + 4 <= m; mi += 4) {
      const ggml_bf16_t* a0 = a.data + (size_t)(mi + 0) * a.k;
      const ggml_bf16_t* a1 = a.data + (size_t)(mi + 1) * a.k;
      const ggml_bf16_t* a2 = a.data + (size_t)(mi + 2) * a.k;
      const ggml_bf16_t* a3 = a.data + (size_t)(mi + 3) * a.k;
      __m256 tot0 = _mm256_setzero_ps(), tot1 = _mm256_setzero_ps();
      __m256 tot2 = _mm256_setzero_ps(), tot3 = _mm256_setzero_ps();

      for (int g = 0; g < group_count; g++) {
        const int k_base = g * group_size;
        __m256 g0 = _mm256_setzero_ps(), g1 = _mm256_setzero_ps();
        __m256 g2 = _mm256_setzero_ps(), g3 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= group_size; ki += 32) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          __m256 wv0 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 0, lut_lo, lut_hi);
          __m256 wv1 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 4, lut_lo, lut_hi);
          __m256 wv2 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 8, lut_lo, lut_hi);
          __m256 wv3 = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 12, lut_lo, lut_hi);

          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki), wv0, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 8), wv1, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 16), wv2, g0);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki + 24), wv3, g0);

          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki), wv0, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 8), wv1, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 16), wv2, g1);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki + 24), wv3, g1);

          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki), wv0, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 8), wv1, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 16), wv2, g2);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki + 24), wv3, g2);

          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki), wv0, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 8), wv1, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 16), wv2, g3);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki + 24), wv3, g3);
        }
        for (; ki + 8 <= group_size; ki += 8) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          __m256 wv = GemmKernelAVX2MXFP4::fp4x8_to_fp32(w, lut_lo, lut_hi);
          g0 = _mm256_fmadd_ps(load_bf16_to_fp32(a0 + k_base + ki), wv, g0);
          g1 = _mm256_fmadd_ps(load_bf16_to_fp32(a1 + k_base + ki), wv, g1);
          g2 = _mm256_fmadd_ps(load_bf16_to_fp32(a2 + k_base + ki), wv, g2);
          g3 = _mm256_fmadd_ps(load_bf16_to_fp32(a3 + k_base + ki), wv, g3);
        }

        __m256 sv = _mm256_broadcast_ss(&b_scales[g]);
        tot0 = _mm256_fmadd_ps(g0, sv, tot0);
        tot1 = _mm256_fmadd_ps(g1, sv, tot1);
        tot2 = _mm256_fmadd_ps(g2, sv, tot2);
        tot3 = _mm256_fmadd_ps(g3, sv, tot3);
      }
      c.data[(size_t)(mi + 0) * n + ni] = hsum_avx2(tot0);
      c.data[(size_t)(mi + 1) * n + ni] = hsum_avx2(tot1);
      c.data[(size_t)(mi + 2) * n + ni] = hsum_avx2(tot2);
      c.data[(size_t)(mi + 3) * n + ni] = hsum_avx2(tot3);
    }

    // M-tail: single token fallback
    for (; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      __m256 total_acc = _mm256_setzero_ps();
      float scalar_tail = 0.0f;

      for (int g = 0; g < group_count; g++) {
        const float scale = b_scales[g];
        const int k_base = g * group_size;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= group_size; ki += 32) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          acc0 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 0, lut_lo, lut_hi), acc0);
          acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 8),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 4, lut_lo, lut_hi), acc1);
          acc2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 16),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 8, lut_lo, lut_hi), acc2);
          acc3 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 24),
                                 GemmKernelAVX2MXFP4::fp4x8_to_fp32(w + 12, lut_lo, lut_hi), acc3);
        }
        __m256 g_acc = _mm256_add_ps(_mm256_add_ps(acc0, acc2), _mm256_add_ps(acc1, acc3));

        for (; ki + 8 <= group_size; ki += 8) {
          const uint8_t* w = b_row + (k_base + ki) / 2;
          g_acc = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki),
                                  GemmKernelAVX2MXFP4::fp4x8_to_fp32(w, lut_lo, lut_hi), g_acc);
        }

        total_acc = _mm256_fmadd_ps(g_acc, _mm256_broadcast_ss(&scale), total_acc);

        for (; ki < group_size; ki++) {
          const int pos = k_base + ki;
          const uint8_t packed = b_row[pos / 2];
          const uint8_t nib = (pos & 1) ? (packed >> 4) : (packed & 0x0F);
          scalar_tail += GGML_BF16_TO_FP32(a_row[pos]) * GemmKernelAVX2MXFP4::fp4_scalar(nib) * scale;
        }
      }

      c.data[(size_t)mi * n + ni] = hsum_avx2(total_acc) + scalar_tail;
    }
  }
}

}  // namespace avx2

// ============================================================================
// CRTP MoE wrapper — mirrors AVX2_RAW_INT4_MOE_TP
// ============================================================================
template <class T = avx2::GemmKernelAVX2MXFP4>
class AVX2_MXFP4_MOE_TP : public AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>>;
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

  AVX2_MXFP4_MOE_TP() = default;
  AVX2_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    if (config_.quant_config.group_size == 0 || config_.quant_config.zero_point)
      throw std::runtime_error("MXFP4 AVX2 MoE requires KGroup FP4 without zero point");
    printf("Created AVX2_MXFP4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()),
           config_.quant_config.group_size);
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    if (!config_.gate_projs.empty()) return T::BufferB::required_size_scale_only(n, k, config_.quant_config.group_size);
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    if (!config_.gate_projs.empty())
      return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data, nullptr);
    return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_mxfp4(m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx], *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_mxfp4(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx], *down_bb_[expert_idx],
                     *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    const bool use_per_expert = !config_.gate_projs.empty();
    if (!use_per_expert && config_.gate_proj == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE requires weight pointers");
    if (!use_per_expert && config_.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE requires scale pointers");

    if (use_per_expert) {
      // C1: Per-expert mode with TP > 1 must use TP_MOE wrapper which has full_intermediate_size.
      // Direct use of AVX2_MXFP4_MOE_TP with tp_part_idx > 0 in per-expert mode is not supported
      // because we cannot compute correct offsets without knowing full_intermediate_size.
      if (tp_part_idx > 0) {
        throw std::runtime_error(
            "AVX2_MXFP4_MOE_TP::load_weights() per-expert mode with tp_part_idx > 0 is not supported. "
            "Use TP_MOE<AVX2_MXFP4_MOE_TP> wrapper for multi-TP scenarios.");
      }

      // Direct-pointer mode: BufferB.b points into mmap'd safetensor data.
      // For tp_part_idx == 0, offset is 0, so config_.intermediate_size value doesn't matter.
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_idx) {
            if (expert_idx < 0 || expert_idx >= config_.expert_num || gate_bb_[expert_idx] == nullptr) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            // H1 & H2: Validate source pointers and lid bounds
            if (lid >= config_.gate_projs[0].size()) {
              throw std::runtime_error("load_weights: lid " + std::to_string(lid) +
                                       " out of bounds (size=" + std::to_string(config_.gate_projs[0].size()) + ")");
            }
            if (config_.gate_projs[0][lid] == nullptr || config_.up_projs[0][lid] == nullptr ||
                config_.down_projs[0][lid] == nullptr) {
              throw std::runtime_error("load_weights: null weight pointer for expert " + std::to_string(lid));
            }
            // tp_part_idx == 0 guaranteed here, so offset is 0
            gate_bb_[expert_idx]->b = (uint8_t*)config_.gate_projs[0][lid];
            up_bb_[expert_idx]->b = (uint8_t*)config_.up_projs[0][lid];
            down_bb_[expert_idx]->b = (uint8_t*)config_.down_projs[0][lid];
          },
          nullptr);

      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (expert_idx >= (uint64_t)config_.expert_num || gate_bb_[expert_idx] == nullptr) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            // H2: Bounds check (already validated in weight loading, but be safe)
            if (lid >= config_.gate_scales[0].size()) return;
            if (config_.gate_scales[0][lid] == nullptr || config_.up_scales[0][lid] == nullptr ||
                config_.down_scales[0][lid] == nullptr) return;
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            // tp_part_idx == 0 guaranteed here, so offset is 0
            convert_or_copy(gate_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.gate_scales[0][lid], scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.up_scales[0][lid], scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d, (const ggml_bf16_t*)config_.down_scales[0][lid], scale_elem_count);
          },
          nullptr);
    } else {
      // Flat-buffer mode: copy TP-sliced weights, convert BF16 scales → FP32.
      int nth_gate = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth_gate * config_.expert_num, nullptr,
          [this, nth_gate, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth_gate;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth_gate;
            size_t weight_offset = ((size_t)lid * config_.intermediate_size * config_.hidden_size) / 2;
            gate_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.gate_proj + weight_offset, ith, nth_gate);
            up_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.up_proj + weight_offset, ith, nth_gate);
          },
          nullptr);

      int nth_down = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth_down * config_.expert_num, nullptr,
          [this, nth_down, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth_down;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth_down;
            size_t weight_offset = ((size_t)lid * config_.hidden_size * config_.intermediate_size) / 2;
            down_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.down_proj + weight_offset, ith, nth_down);
          },
          nullptr);

      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t lid = expert_map(physical_to_logical_map, expert_idx);
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            convert_or_copy(gate_bb_[expert_idx]->d, (ggml_bf16_t*)config_.gate_scale + lid * scale_elem_count,
                            scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d, (ggml_bf16_t*)config_.up_scale + lid * scale_elem_count,
                            scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d, (ggml_bf16_t*)config_.down_scale + lid * scale_elem_count,
                            scale_elem_count);
          },
          nullptr);
    }
  }

  static inline void fp32_to_bf16(ggml_bf16_t* dst, const float* src, size_t count) {
    convert_or_copy(dst, src, count);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    if (expert_id < 0 || expert_id >= config_.expert_num || gate_bb_[expert_id] == nullptr ||
        up_bb_[expert_id] == nullptr || down_bb_[expert_id] == nullptr)
      throw std::runtime_error("MXFP4 AVX2 write_weights_to_buffer: invalid expert");

    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_bytes = (size_t)config_.intermediate_size * config_.hidden_size / 2;
    size_t cpu_tp_scale_elem_count = (size_t)config_.intermediate_size * config_.hidden_size / group_size;
    size_t gpu_tp_weight_bytes = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count / 2;
    size_t gpu_tp_scale_elem_count =
        (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);
      uint8_t* w13_wdst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_sdst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_wdst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_sdst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];
      size_t w_off = (size_t)local_idx * cpu_tp_weight_bytes;
      size_t s_off = (size_t)local_idx * cpu_tp_scale_elem_count;

      constexpr int NWT = 8;
      int ndt = std::min(std::max(1, config_.hidden_size / 128), 32);
      size_t wchunk = ((cpu_tp_weight_bytes + NWT - 1) / NWT + 63) & ~63ULL;
      pool->do_work_stealing_job(
          NWT * 2 + ndt + 2, nullptr,
          [=, this](int tid) {
            if (tid < NWT) {
              size_t s = (size_t)tid * wchunk, e = std::min(s + wchunk, cpu_tp_weight_bytes);
              if (s < e) std::memcpy(w13_wdst + w_off + s, gate_bb_[expert_id]->b + s, e - s);
            } else if (tid < NWT * 2) {
              size_t s = (size_t)(tid - NWT) * wchunk, e = std::min(s + wchunk, cpu_tp_weight_bytes);
              if (s < e) std::memcpy(w13_wdst + w_off + gpu_tp_weight_bytes + s, up_bb_[expert_id]->b + s, e - s);
            } else if (tid < NWT * 2 + ndt) {
              size_t cols_per = (config_.hidden_size + ndt - 1) / ndt;
              size_t cs = (size_t)(tid - NWT * 2) * cols_per;
              size_t ce = std::min(cs + cols_per, (size_t)config_.hidden_size);
              size_t wpc = config_.intermediate_size >> 1;
              size_t spc = config_.intermediate_size / group_size;
              size_t gws = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gss = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gwo = (size_t)local_idx * wpc, gso = (size_t)local_idx * spc;
              for (size_t col = cs; col < ce; col++) {
                std::memcpy(w2_wdst + col * gws + gwo, down_bb_[expert_id]->b + col * wpc, wpc);
                fp32_to_bf16(w2_sdst + col * gss + gso, down_bb_[expert_id]->d + col * spc, spc);
              }
            } else if (tid == NWT * 2 + ndt) {
              fp32_to_bf16(w13_sdst + s_off, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fp32_to_bf16(w13_sdst + s_off + gpu_tp_scale_elem_count, up_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU partition feeds multiple GPU TPs
      int gpu_per_cpu = gpu_tp_count / cpu_tp_count;
      int start_gtp = tp_part_idx * gpu_per_cpu;
      size_t dw = cpu_tp_weight_bytes / gpu_per_cpu;
      size_t ds = cpu_tp_scale_elem_count / gpu_per_cpu;
      constexpr int NWT = 8;
      int ndt = std::min(std::max(1, config_.hidden_size / 128), 32);
      size_t wchunk = ((dw + NWT - 1) / NWT + 63) & ~63ULL;
      int tpg = NWT * 2 + ndt + 2;
      pool->do_work_stealing_job(
          tpg * gpu_per_cpu, nullptr,
          [=, this, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs](int task_id) {
            int lg = task_id / tpg;
            int tt = task_id % tpg;
            int gtp = start_gtp + lg;
            uint8_t* w13_wdst = (uint8_t*)w13_weight_ptrs[gtp];
            ggml_bf16_t* w13_sdst = (ggml_bf16_t*)w13_scale_ptrs[gtp];
            uint8_t* w2_wdst = (uint8_t*)w2_weight_ptrs[gtp];
            ggml_bf16_t* w2_sdst = (ggml_bf16_t*)w2_scale_ptrs[gtp];
            size_t cow = (size_t)lg * dw, cos = (size_t)lg * ds;
            if (tt < NWT) {
              size_t s = (size_t)tt * wchunk, e = std::min(s + wchunk, dw);
              if (s < e) std::memcpy(w13_wdst + s, gate_bb_[expert_id]->b + cow + s, e - s);
            } else if (tt < NWT * 2) {
              size_t s = (size_t)(tt - NWT) * wchunk, e = std::min(s + wchunk, dw);
              if (s < e) std::memcpy(w13_wdst + gpu_tp_weight_bytes + s, up_bb_[expert_id]->b + cow + s, e - s);
            } else if (tt < NWT * 2 + ndt) {
              size_t cols_per = (config_.hidden_size + ndt - 1) / ndt;
              size_t cs = (size_t)(tt - NWT * 2) * cols_per;
              size_t ce = std::min(cs + cols_per, (size_t)config_.hidden_size);
              size_t wgc = (config_.intermediate_size / gpu_per_cpu) >> 1;
              size_t sgc = (config_.intermediate_size / gpu_per_cpu) / group_size;
              for (size_t col = cs; col < ce; col++) {
                size_t cwoff = col * config_.intermediate_size / 2 + lg * dw / config_.hidden_size;
                size_t csoff = col * (config_.intermediate_size / group_size) + lg * ds / config_.hidden_size;
                std::memcpy(w2_wdst + col * wgc, down_bb_[expert_id]->b + cwoff, wgc);
                fp32_to_bf16(w2_sdst + col * sgc, down_bb_[expert_id]->d + csoff, sgc);
              }
            } else if (tt == NWT * 2 + ndt) {
              fp32_to_bf16(w13_sdst, gate_bb_[expert_id]->d + cos, ds);
            } else {
              fp32_to_bf16(w13_sdst + gpu_tp_scale_elem_count, up_bb_[expert_id]->d + cos, ds);
            }
          },
          nullptr);
    }
  }
};

// TP_MOE specialization (boilerplate, identical to rawint4 pattern)
template <typename K>
class TP_MOE<AVX2_MXFP4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>> {
  std::vector<void*> tp_owned_down_bufs_;

 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP4_MOE_TP<K>>>;
  using Base::Base;

  ~TP_MOE() {
    for (void* p : tp_owned_down_bufs_)
      if (p) std::free(p);
  }

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    bool use_per_expert = !config.gate_projs.empty();
    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 AVX2 MoE only supports packed FP4 with KGroup scales");

    if (use_per_expert) {
      int group_size = config.quant_config.group_size;
      int full_interm = config.intermediate_size;

      // Allocate per-partition down_proj repack buffers (gate/up use direct pointers).
      // down is [hidden, intermediate] with TP along K — rows are non-contiguous.
      tp_owned_down_bufs_.resize(this->tp_count, nullptr);
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto* tp = tps[i].get();
        auto& tpc = tp->config_;
        tpc.physical_to_logical_map = config.physical_to_logical_map;
        int per_tp_interm = tpc.intermediate_size;
        if (per_tp_interm % 2 != 0)
          throw std::runtime_error("MXFP4 TP: per_tp_interm must be even for nibble-aligned addressing, got " +
                                   std::to_string(per_tp_interm));
        size_t down_wt_per_expert = (size_t)tpc.hidden_size * per_tp_interm / 2;
        size_t alloc_size = ((size_t)tpc.expert_num * down_wt_per_expert + 63) & ~(size_t)63;
        uint8_t* down_buf = (uint8_t*)std::aligned_alloc(64, alloc_size);
        if (!down_buf) throw std::runtime_error("aligned_alloc failed for MXFP4 down_buf");
        tp_owned_down_bufs_[i] = down_buf;

        auto subpool = pool->get_subpool(i);
        subpool->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i, per_tp_interm, full_interm, down_buf, down_wt_per_expert](int eid) {
              if (tpc.should_skip_expert(eid)) return;
              uint64_t lid = expert_map(physical_to_logical_map, eid);

              // H1 & H2: Validate source pointers and lid bounds
              if (lid >= config.gate_projs[0].size()) {
                throw std::runtime_error("TP_MOE load_weights: lid " + std::to_string(lid) +
                                         " out of bounds (size=" + std::to_string(config.gate_projs[0].size()) + ")");
              }
              if (config.gate_projs[0][lid] == nullptr || config.up_projs[0][lid] == nullptr ||
                  config.down_projs[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null weight pointer for expert " + std::to_string(lid));
              }

              // C2 FIX: gate/up weights: N-split, contiguous rows → direct pointer
              // Use full_interm (not per_tp_interm) to compute offset into full safetensor data
              size_t n_byte_off = (size_t)i * full_interm * tpc.hidden_size / 2;
              tp->gate_bb_[eid]->b = (uint8_t*)config.gate_projs[0][lid] + n_byte_off;
              tp->up_bb_[eid]->b = (uint8_t*)config.up_projs[0][lid] + n_byte_off;

              // C4 FIX: gate/up scales: contiguous → convert BF16→FP32
              // Use full_interm (not per_tp_interm) for scale offset calculation
              size_t scale_count = (size_t)(tpc.hidden_size / group_size) * per_tp_interm;
              size_t scale_off = (size_t)i * full_interm * (tpc.hidden_size / group_size);
              if (config.gate_scales[0][lid] == nullptr || config.up_scales[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null scale pointer for expert " + std::to_string(lid));
              }
              convert_or_copy(tp->gate_bb_[eid]->d, (const ggml_bf16_t*)config.gate_scales[0][lid] + scale_off,
                              scale_count);
              convert_or_copy(tp->up_bb_[eid]->d, (const ggml_bf16_t*)config.up_scales[0][lid] + scale_off,
                              scale_count);

              // down weights: K-split, non-contiguous → per-row memcpy into repack buf
              uint8_t* src_down = (uint8_t*)config.down_projs[0][lid];
              uint8_t* dst_down = down_buf + (size_t)eid * down_wt_per_expert;
              for (int row = 0; row < tpc.hidden_size; row++) {
                std::memcpy(dst_down + (size_t)row * per_tp_interm / 2,
                            src_down + (size_t)row * full_interm / 2 + (size_t)i * per_tp_interm / 2,
                            per_tp_interm / 2);
              }
              tp->down_bb_[eid]->b = dst_down;

              // down scales: K-split, non-contiguous → per-row convert
              if (config.down_scales[0][lid] == nullptr) {
                throw std::runtime_error("TP_MOE load_weights: null down_scale pointer for expert " + std::to_string(lid));
              }
              int full_interm_groups = full_interm / group_size;
              int per_tp_groups = per_tp_interm / group_size;
              const ggml_bf16_t* src_ds = (const ggml_bf16_t*)config.down_scales[0][lid];
              float* dst_ds = tp->down_bb_[eid]->d;
              for (int row = 0; row < tpc.hidden_size; row++) {
                convert_or_copy(dst_ds + (size_t)row * per_tp_groups,
                                src_ds + (size_t)row * full_interm_groups + (size_t)i * per_tp_groups, per_tp_groups);
              }
            },
            nullptr);
      });
    } else {
      int group_size = config.quant_config.group_size;
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        if (tpc.intermediate_size % 2 != 0)
          throw std::runtime_error("MXFP4 TP flat-buffer: intermediate_size must be even for nibble-aligned addressing, got " +
                                   std::to_string(tpc.intermediate_size));
        size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = ((size_t)tpc.hidden_size / group_size) * tpc.intermediate_size;
        tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
        tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
        tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      });

      // Fill the TP-sliced flat buffers from full-model data, then load.
      // TP slicing is along the intermediate dimension. gate/up are
      // [intermediate × hidden] with intermediate as outer (N) dim — contiguous
      // memcpy slice works. down is [hidden × intermediate] with intermediate
      // as inner (K) dim — must loop per hidden row.
      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
        size_t scales_elem_count = ((size_t)tpc.hidden_size / group_size) * tpc.intermediate_size;
        size_t n_per_tp = tpc.intermediate_size;
        size_t hidden_groups_per_row = (size_t)tpc.hidden_size / group_size;
        size_t interm_groups_per_row = (size_t)tpc.intermediate_size / group_size;
        size_t full_interm_groups_per_row = (size_t)config.intermediate_size / group_size;
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int eid) {
              uint64_t lid = expert_map(physical_to_logical_map, eid);
              uint8_t* src_gate = (uint8_t*)config.gate_proj +
                                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * config.hidden_size / 2;
              uint8_t* src_up = (uint8_t*)config.up_proj +
                                (lid * (size_t)config.intermediate_size + i * n_per_tp) * config.hidden_size / 2;
              std::memcpy((uint8_t*)tpc.gate_proj + eid * weight_elem_count / 2, src_gate, weight_elem_count / 2);
              std::memcpy((uint8_t*)tpc.up_proj + eid * weight_elem_count / 2, src_up, weight_elem_count / 2);
              ggml_bf16_t* src_gate_scale =
                  (ggml_bf16_t*)config.gate_scale +
                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * hidden_groups_per_row;
              ggml_bf16_t* src_up_scale =
                  (ggml_bf16_t*)config.up_scale +
                  (lid * (size_t)config.intermediate_size + i * n_per_tp) * hidden_groups_per_row;
              std::memcpy((ggml_bf16_t*)tpc.gate_scale + eid * scales_elem_count, src_gate_scale,
                          scales_elem_count * sizeof(ggml_bf16_t));
              std::memcpy((ggml_bf16_t*)tpc.up_scale + eid * scales_elem_count, src_up_scale,
                          scales_elem_count * sizeof(ggml_bf16_t));
              uint8_t* src_down =
                  (uint8_t*)config.down_proj + lid * (size_t)config.hidden_size * config.intermediate_size / 2;
              ggml_bf16_t* src_down_scale =
                  (ggml_bf16_t*)config.down_scale + lid * (size_t)config.hidden_size * full_interm_groups_per_row;
              for (size_t col = 0; col < (size_t)tpc.hidden_size; col++) {
                std::memcpy((uint8_t*)tpc.down_proj + eid * weight_elem_count / 2 + col * n_per_tp / 2,
                            src_down + (col * config.intermediate_size + i * n_per_tp) / 2, n_per_tp / 2);
                std::memcpy((ggml_bf16_t*)tpc.down_scale + eid * scales_elem_count + col * interm_groups_per_row,
                            src_down_scale + col * full_interm_groups_per_row + i * interm_groups_per_row,
                            interm_groups_per_row * sizeof(ggml_bf16_t));
              }
            },
            nullptr);
      });

      DO_TPS_LOAD_WEIGHTS(pool);

      pool->dispense_backend()->do_numa_job([&, this](int i) {
        auto& tpc = tps[i]->config_;
        delete[] (uint8_t*)tpc.gate_proj;
        delete[] (uint8_t*)tpc.up_proj;
        delete[] (uint8_t*)tpc.down_proj;
        delete[] (ggml_bf16_t*)tpc.gate_scale;
        delete[] (ggml_bf16_t*)tpc.up_scale;
        delete[] (ggml_bf16_t*)tpc.down_scale;
      });
    }

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w13_scale_ptrs.size() != gpu_tp_count ||
        (int)w2_weight_ptrs.size() != gpu_tp_count || (int)w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }
    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_MXFP4_MOE_H
