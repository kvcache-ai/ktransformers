/**
 * @Description  : AVX2 MXFP8 MoE operator (AVX2 sibling of amx/mxfp8-moe.hpp)
 * @Author       : yyj and Claude
 * @Date         : 2026-06-09
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Serves MiniMax M3 Preview (MXFP8 quantized checkpoint) on AVX2-only hardware
 * (Haswell+ / Zen+, no AVX-512 / VBMI / dpbf16_ps required).
 *
 * Algorithm parity with amx/mxfp8-moe.hpp:
 *   Weight:   FP8 E4M3fn (1 byte/element, row-major [n, k])
 *   Scale:    ue8m0 per-group (group_size=32), converted to FP32 on load
 *   Act:      BF16 (row-major [m, k]), promoted to FP32 per 8-lane chunk
 *   Decode:   256-entry FP32 LUT + _mm256_i32gather_ps  (vs VBMI permutex2var)
 *   Dot prod: BF16→FP32 promote + _mm256_fmadd_ps       (vs _mm512_dpbf16_ps)
 *   Activation: swiglu_oai via avx2_bf16_utils.hpp::act_fn (alpha/limit aware)
 *
 * Inner-loop pattern: **deferred-hsum** — accumulate scale·v_g into a vector
 * row accumulator across all k-groups, hsum once per output row. ~10× fewer
 * hsums than per-group hsum used by gptq_int4-moe.hpp (which can't defer
 * because its scale is baked into dequant). MXFP8 has scale separate from
 * dequant, so deferring is math-equivalent: hsum(sum_g s[g]*v_g) =
 * sum_g s[g]*hsum(v_g) since hsum is linear.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_MXFP8_MOE_H
#define CPUINFER_OPERATOR_AVX2_MXFP8_MOE_H

#include <cassert>
#include <cstdint>
#include <cstring>

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "fp8_dequant.hpp"
#include "moe_base.hpp"

namespace avx2 {

// ============================================================================
// ue8m0 → FP32 scale conversion (vectorized)
//
// ue8m0 byte b ∈ [0, 255] → FP32 value 2^(b - 127).
// IEEE754 FP32 layout: [sign:1][exp:8][mantissa:23]. Setting exp = b and
// mantissa = 0 (with sign = 0) gives exactly 2^(b - 127).
// Edge case: b=0 → (0 << 23) = 0.0f (technically 2^-127 ≈ 5.9e-39, accepted).
// Verbatim copy of amx/mxfp8-moe.hpp::convert_ue8m0_to_fp32 — already pure AVX2.
// ============================================================================
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

// ============================================================================
// FP32 -> ue8m0 (uint8) — reverse of convert_ue8m0_to_fp32 (AVX2 version)
//
// Used by write_weights_to_buffer for layerwise prefill: CPU stores FP32 scales
// internally (in BufferB.d for fast FMA in inner loop), but GPU's MXFP8 buffer
// expects ue8m0 uint8 (1 byte/scale). We extract bits 23-30 of each FP32 (the
// exponent field) and store as uint8. Bit-exact iff the FP32 came from
// convert_ue8m0_to_fp32 (mantissa=0), which is the case for layerwise prefill
// since CPU forward never modifies BufferB.d.
// ============================================================================
static inline void fast_fp32_to_ue8m0(uint8_t* __restrict dst,
                                       const float* __restrict src,
                                       size_t count) {
  size_t i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    __m256i bits = _mm256_castps_si256(v);
    // Extract bits 23-30 (the FP32 exponent) -> 8 uint32 values in [0,255]
    __m256i shifted = _mm256_srli_epi32(bits, 23);
    // Pack 8 uint32 -> 8 uint16 -> 8 uint8 (saturating; values are <= 255 so no-op)
    __m128i lo = _mm256_castsi256_si128(shifted);
    __m128i hi = _mm256_extracti128_si256(shifted, 1);
    __m128i p16 = _mm_packus_epi32(lo, hi);            // 8 x uint16
    __m128i p8 = _mm_packus_epi16(p16, p16);           // 8 x uint8 in low 64 bits
    _mm_storel_epi64((__m128i*)(dst + i), p8);
  }
  for (; i < count; i++) {
    uint32_t bits;
    std::memcpy(&bits, &src[i], sizeof(uint32_t));
    dst[i] = (uint8_t)((bits >> 23) & 0xFF);
  }
}

// ============================================================================
// GemmKernelAVX2MXFP8 — kernel descriptor
//
//   M_STEP=1, N_STEP=8, K_STEP=K_GROUP_SIZE=32
//   BufferA: BF16 activations [m, k] (raw, no pre-promotion)
//   BufferB: FP8 weights [n, k] + FP32 scales [n, k/group_size]
//   BufferC: FP32 output [m, n]
// ============================================================================
struct GemmKernelAVX2MXFP8 {
  using dt = ggml_bf16_t;
  using output_t = float;

  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 32;          // == K_GROUP_SIZE
  static constexpr int K_GROUP_SIZE = 32;    // MXFP8 group size per OCP standard
  static constexpr int N_BLOCK = 64;         // ~8 N_STEP tiles per task
  static constexpr int K_BLOCK = 6144;       // M3 hidden=6144 in one pass
  static constexpr double ELEMENT_SIZE = 1.0;  // FP8 = 1 byte/element

  static void config() {}

  static int recommended_nth(int n) {
    return std::max(1, (n + N_BLOCK - 1) / N_BLOCK);
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return ::avx2::split_range(n, ith, nth);
  }

  // --------------------------------------------------------------------------
  // BufferA: BF16 activations [m, k] — identical to AVX2 BF16/FP8 backends
  // --------------------------------------------------------------------------
  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : max_m(m), k(k_), data((ggml_bf16_t*)ptr) {}

    static size_t required_size(size_t m, size_t k) {
      return m * k * sizeof(ggml_bf16_t);
    }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        auto [m_start, m_end] = ::avx2::split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  // --------------------------------------------------------------------------
  // BufferB: FP8 [n, k] + FP32 scales [n, k/group_size]
  //   b = ptr;           (n*k FP8 bytes)
  //   d = ptr + n*k;     (n * (k/group_size) FP32 scales)
  // Identical memory layout to amx::BufferBMXFP8KGroupImpl, so the M3 checkpoint
  // loader can target both AMX and AVX2 sides with the same pointer arithmetic.
  // --------------------------------------------------------------------------
  struct BufferB {
    uint8_t* b = nullptr;
    float* d = nullptr;
    int n = 0;
    int k = 0;
    int k_group_size = K_GROUP_SIZE;
    int k_group_count = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr)
        : n((int)n_), k((int)k_), k_group_size(gs) {
      if (k % gs != 0) {
        printf("BufferB(MXFP8 AVX2): k=%d not divisible by group_size=%d\n", k, gs);
        throw std::runtime_error("MXFP8 AVX2: k must be divisible by group_size");
      }
      k_group_count = k / gs;
      b = (uint8_t*)ptr;
      d = (float*)((uint8_t*)ptr + (size_t)n * k);
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      return n * k + n * (k / gs) * sizeof(float);
    }

    // Copy raw FP8 bytes from checkpoint. 1 byte/element, row-major.
    void from_raw_mat(const uint8_t* src_weights, int ith, int nth) {
      auto [n_start, n_end] = ::avx2::split_range(n, ith, nth);
      if (n_start >= n_end) return;
      const size_t row_bytes = (size_t)k;
      std::memcpy(b + (size_t)n_start * row_bytes,
                  src_weights + (size_t)n_start * row_bytes,
                  (size_t)(n_end - n_start) * row_bytes);
    }

    // Returns pointer to FP8 byte at (n_begin, k_begin).
    uint8_t* get_submat(int n_begin, int k_begin) {
      return b + (size_t)n_begin * k + k_begin;
    }

    // Returns pointer to FP32 scale for row n_begin starting at k-group k_begin/group_size.
    float* get_scale(int n_begin, int k_begin) {
      return d + (size_t)n_begin * k_group_count + k_begin / k_group_size;
    }
  };

  // --------------------------------------------------------------------------
  // BufferC: FP32 output [m, n] — identical to AVX2 BF16/FP8 backends
  // --------------------------------------------------------------------------
  struct BufferC {
    float* data = nullptr;
    size_t max_m = 0;
    size_t n = 0;

    BufferC() = default;
    BufferC(size_t m, size_t n_, void* ptr) : max_m(m), n(n_), data((float*)ptr) {}

    static size_t required_size(size_t m, size_t n) {
      return m * n * sizeof(float);
    }

    void set_data(void* ptr) { data = (float*)ptr; }

    void to_mat(int m, ggml_bf16_t* dst, int ith, int nth) {
      auto [n_start, n_end] = ::avx2::split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          __m256 v = _mm256_loadu_ps(src_row + j);
          store_fp32_to_bf16(dst_row + j, v);
        }
        for (; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

// ============================================================================
// gemm_mxfp8_kgroup — AVX2 MXFP8 GEMM (BF16 act × FP8 weight + per-K-group scale)
//
//   C[m, n] = sum_g scale[n, g] * sum_{i in group g} A[m, k_base+i] * dequant(B[n, k_base+i])
//
// Per output row (ni): one vector accumulator `row_acc` lives across all groups;
// per group, do 4× FMA into a fresh `v` (8 lanes), then `row_acc += s[g] * v`
// via broadcast+FMA. After all groups, hsum_avx2(row_acc) once for the scalar.
// ============================================================================
static inline void gemm_mxfp8_kgroup(
    int m, int n, int k,
    GemmKernelAVX2MXFP8::BufferA& a,
    GemmKernelAVX2MXFP8::BufferB& b,
    GemmKernelAVX2MXFP8::BufferC& c,
    int ith, int nth) {

  ensure_fp8_lut_initialized();  // idempotent; cheap after first call

  auto [n_start, n_end] = ::avx2::split_range(n, ith, nth);
  if (n_start >= n_end) return;
  const int kg_count = b.k_group_count;

  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* w_base = b.get_submat(ni, 0);
    const float* s = b.get_scale(ni, 0);

    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_base = a.data + (size_t)mi * a.k;
      __m256 row_acc = _mm256_setzero_ps();
      const uint8_t* w = w_base;
      const ggml_bf16_t* a_row = a_base;

      for (int g = 0; g < kg_count; g++) {
        // 4-chunk × 8-lane unroll: 32 elements per k-group
        __m256 v = _mm256_setzero_ps();
        v = _mm256_fmadd_ps(fp8x8_to_fp32x8(w +  0), load_bf16_to_fp32(a_row +  0), v);
        v = _mm256_fmadd_ps(fp8x8_to_fp32x8(w +  8), load_bf16_to_fp32(a_row +  8), v);
        v = _mm256_fmadd_ps(fp8x8_to_fp32x8(w + 16), load_bf16_to_fp32(a_row + 16), v);
        v = _mm256_fmadd_ps(fp8x8_to_fp32x8(w + 24), load_bf16_to_fp32(a_row + 24), v);
        // Defer hsum: broadcast scale, FMA into vector row accumulator.
        // Math: hsum(Σ_g s[g] * v_g) = Σ_g s[g] * hsum(v_g) (hsum is linear).
        row_acc = _mm256_fmadd_ps(_mm256_set1_ps(s[g]), v, row_acc);
        w += 32;
        a_row += 32;
      }
      c.data[mi * n + ni] = hsum_avx2(row_acc);
    }
  }
}

}  // namespace avx2

// ============================================================================
// AVX2_MXFP8_MOE_TP — CRTP wrapper, mirrors amx::AMX_MXFP8_MOE_TP
//
// Activation (swigluoai) is handled by AVX2_MOE_BASE::apply_activation, which
// dispatches via avx2::act_fn(g, u, swiglu_limit, swiglu_alpha). Our derived
// class just needs derived_init / buffer factories / GEMM dispatch / load.
// ============================================================================
template <class T = avx2::GemmKernelAVX2MXFP8>
class AVX2_MXFP8_MOE_TP : public AVX2_MOE_BASE<T, AVX2_MXFP8_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_MXFP8_MOE_TP<T>>;
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

  AVX2_MXFP8_MOE_TP() = default;
  AVX2_MXFP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    avx2::ensure_fp8_lut_initialized();  // single-threaded init before any forward
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error("AVX2 MXFP8 MoE requires group_size == 32 and no zero_point");
    }
    if (config_.hidden_size % quant_config.group_size != 0 ||
        config_.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error("AVX2 MXFP8 MoE: hidden_size and intermediate_size must be divisible by group_size");
    }
    printf("Created AVX2_MXFP8_MOE_TP %d at numa %d (group_size=%d, swiglu_alpha=%.4f, swiglu_limit=%.4f)\n",
           tp_part_idx, numa_node_of_cpu(sched_getcpu()),
           quant_config.group_size, config_.swiglu_alpha, config_.swiglu_limit);
  }

  ~AVX2_MXFP8_MOE_TP() = default;

  // CRTP buffer creation
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

  // GEMM dispatch
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_mxfp8_kgroup(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, [[maybe_unused]] int qlen) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_mxfp8_kgroup(m, config_.hidden_size, config_.intermediate_size,
                            *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // Load FP8 weights + ue8m0 scales from checkpoint.
  //   gate_proj/up_proj/down_proj: uint8_t [E, N, K] (FP8 E4M3fn bytes)
  //   gate_scale/up_scale/down_scale: uint8_t [E, N, K/group_size] (ue8m0)
  // We memcpy weights and bit-shift convert scales to FP32 in-place.
  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point)
      throw std::runtime_error("AVX2 MXFP8 MoE requires group_size > 0 and no zero_point");
    if (config_.gate_scale == nullptr)
      throw std::runtime_error("AVX2 MXFP8 MoE requires native MXFP8 weights with ue8m0 scales");

    // --- Load FP8 weights (1 byte/element) — copy into BufferB.b via from_raw_mat ---
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          size_t weight_offset = (size_t)logical_expert_id * config_.intermediate_size * config_.hidden_size;
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
          size_t weight_offset = (size_t)logical_expert_id * config_.hidden_size * config_.intermediate_size;
          down_bb_[expert_idx]->from_raw_mat((uint8_t*)config_.down_proj + weight_offset, ith, nth);
        },
        nullptr);

    // --- Convert ue8m0 scales → FP32 in BufferB.d ---
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_count =
              ((size_t)config_.intermediate_size * config_.hidden_size) / config_.quant_config.group_size;
          avx2::convert_ue8m0_to_fp32(
              gate_bb_[expert_idx]->d,
              (const uint8_t*)config_.gate_scale + logical_expert_id * scale_count, scale_count);
          avx2::convert_ue8m0_to_fp32(
              up_bb_[expert_idx]->d,
              (const uint8_t*)config_.up_scale + logical_expert_id * scale_count, scale_count);
          avx2::convert_ue8m0_to_fp32(
              down_bb_[expert_idx]->d,
              (const uint8_t*)config_.down_scale + logical_expert_id * scale_count, scale_count);
        },
        nullptr);
  }

  // --------------------------------------------------------------------------
  // write_weights_to_buffer: copies CPU expert weights to GPU pinned host buffer
  // for layerwise prefill (sglang full-GPU fallback at large prefill token count).
  //
  // Mirrors amx::AMX_MXFP8_MOE_TP::write_weights_to_buffer (amx/mxfp8-moe.hpp:662)
  // with two substitutions for AVX2:
  //   fast_memcpy        → std::memcpy   (libc memcpy is AVX2-optimized on Haswell+)
  //   fast_fp32_to_bf16  → avx2::fast_fp32_to_ue8m0 (writes 1 byte/scale matching
  //                                                  GPU's torch.uint8 ue8m0 layout)
  //
  // Scale dtype on the GPU side is torch.uint8 (ue8m0), see
  // kt-sglang/python/sglang/srt/layers/quantization/fp8.py:872.
  // --------------------------------------------------------------------------
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    // Unified per-row scatter that works for any (cpu_tp_count, gpu_tp_count)
    // relationship (mirrors avx2/fp8-moe.hpp:347-451). Replaces the earlier
    // `cpu_tp_count >= gpu_tp_count` direct-write branch which was a silent
    // no-op for the realistic 2-NUMA × tp>=4 case (typical M3 deployment:
    // 2 NUMA sockets × tp=8). Each task processes a row chunk of this CPU
    // TP's slice; per row we compute target_gpu = global_n / gpu_n_w13 (W13)
    // or scatter across multiple gpu_tp k-slices (W2) and write to the
    // corresponding GPU TP staging buffer at the right offset.
    //
    // MXFP8 specifics vs FP8 block:
    //   - Scale dtype on GPU side is uint8 ue8m0 (1 byte per `group_size`
    //     weights, per row, not per block).
    //   - Source scale (bb->d) is stored as FP32 in kt-kernel BufferB and is
    //     converted to ue8m0 via avx2::fast_fp32_to_ue8m0 at write time.
    //   - Scale layout matches GPU: (E, 2*intermediate, hidden/group_size)
    //     for W13 and (E, hidden, intermediate/group_size) for W2.

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
              std::memcpy(w_dst + expert_w_off + (size_t)n_in_gpu * gpu_k_w13,
                          bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13);

              // Scale row: full K/group_size ue8m0 bytes (fp32 → ue8m0 conversion).
              uint8_t* s_dst = (uint8_t*)w13_scale_ptrs[target_gpu];
              const size_t expert_s_off = is_up ? gpu_w13_scale_per_mat : 0;
              avx2::fast_fp32_to_ue8m0(
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
              // CPU's K range = [global_k_offset_w2, global_k_offset_w2 + cpu_k_w2).
              // Each gpu_k_w2-aligned slice within this range goes to its own
              // target_gpu. Loop covers cpu_k_w2/gpu_k_w2 GPU TPs (or 1 if
              // cpu_k_w2 < gpu_k_w2 — i.e. cpu_tp_count > gpu_tp_count case).
              for (int k_start = 0; k_start < cpu_k_w2; k_start += gpu_k_w2) {
                const int k_slice_len = std::min(gpu_k_w2, cpu_k_w2 - k_start);
                const int global_k = global_k_offset_w2 + k_start;
                const int target_gpu = global_k / gpu_k_w2;
                const int k_in_gpu = global_k % gpu_k_w2;

                // Weight K-slice
                uint8_t* w_dst = (uint8_t*)w2_weight_ptrs[target_gpu];
                std::memcpy(w_dst + (size_t)row * gpu_k_w2 + k_in_gpu,
                            bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len);

                // Scale K-slice (k_slice_len/group_size ue8m0 bytes)
                const int scale_slice_len = k_slice_len / group_size;
                const int k_in_gpu_scale = k_in_gpu / group_size;
                const int k_start_scale = k_start / group_size;
                uint8_t* s_dst = (uint8_t*)w2_scale_ptrs[target_gpu];
                avx2::fast_fp32_to_ue8m0(
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
// TP_MOE specialization — handles per-expert FP8 weight + ue8m0 scale loading
// across TP parts. Mirrors AVX2_FP8_MOE_TP's TP_MOE but with MXFP8 layout:
//   weights: [E, N, K] FP8 bytes (no nibble pack, no block scale)
//   scales:  [E, N, K/group_size] ue8m0 bytes (NOT pre-converted)
//
// Inside per-TP load_weights() we point config to the staged TP-sliced bytes,
// then derived_class::load_weights() reads from there and runs ue8m0→FP32.
// ============================================================================
template <typename K>
class TP_MOE<AVX2_MXFP8_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_MXFP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0 || config.quant_config.zero_point) {
      throw std::runtime_error("MXFP8 MoE only supports group-wise (group_size > 0, zero_point=false)");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    // Full dimensions
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems = full_weight_elems / (size_t)group_size;  // ue8m0 = 1 byte each

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems = tp_weight_elems / (size_t)group_size;

      // Allocate temporary buffers for TP-sliced FP8 + ue8m0 scales
      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.gate_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new uint8_t[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new uint8_t[tpc.expert_num * tp_scale_elems];

      // gate/up: split N=intermediate, each expert is [intermediate, hidden] FP8 + [intermediate, hidden/gs] ue8m0
      const size_t gate_up_w_src_off = i * tp_weight_elems;  // bytes
      const size_t gate_up_s_src_off = i * tp_scale_elems;   // bytes (ue8m0=1B)

      // down: split K=intermediate (columns of [hidden, intermediate] FP8 + [hidden, intermediate/gs] ue8m0)
      const size_t down_col_off = (size_t)i * tpc.intermediate_size;
      const size_t down_scale_col_off = down_col_off / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems;
            uint8_t* gate_s_dst = (uint8_t*)tpc.gate_scale + expert_id * tp_scale_elems;
            uint8_t* up_s_dst = (uint8_t*)tpc.up_scale + expert_id * tp_scale_elems;
            uint8_t* down_s_dst = (uint8_t*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const uint8_t* gate_s_src;
            const uint8_t* up_s_src;
            const uint8_t* down_s_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_w_src_off;
              up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_w_src_off;
              down_src = (const uint8_t*)config.down_projs[0][expert_id];
              gate_s_src = (const uint8_t*)config.gate_scales[0][expert_id] + gate_up_s_src_off;
              up_s_src = (const uint8_t*)config.up_scales[0][expert_id] + gate_up_s_src_off;
              down_s_src = (const uint8_t*)config.down_scales[0][expert_id];
            } else {
              gate_src = (const uint8_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_w_src_off;
              up_src = (const uint8_t*)config.up_proj + expert_id * full_weight_elems + gate_up_w_src_off;
              down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems;
              gate_s_src = (const uint8_t*)config.gate_scale + expert_id * full_scale_elems + gate_up_s_src_off;
              up_s_src = (const uint8_t*)config.up_scale + expert_id * full_scale_elems + gate_up_s_src_off;
              down_s_src = (const uint8_t*)config.down_scale + expert_id * full_scale_elems;
            }

            // gate/up weights + scales: contiguous N slice
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_s_dst, gate_s_src, tp_scale_elems);
            std::memcpy(up_s_dst, up_s_src, tp_scale_elems);

            // down weights: column slice within each of `hidden` rows
            // src row = [hidden_row, full_intermediate]; dst row = [hidden_row, tp_intermediate]
            for (int row = 0; row < config.hidden_size; row++) {
              std::memcpy(down_dst + (size_t)row * tpc.intermediate_size,
                          down_src + (size_t)row * config.intermediate_size + down_col_off,
                          tpc.intermediate_size);
            }
            // down scales: column slice within each of `hidden` scale rows
            const int full_kg = config.intermediate_size / group_size;
            const int tp_kg = tpc.intermediate_size / group_size;
            for (int row = 0; row < config.hidden_size; row++) {
              std::memcpy(down_s_dst + (size_t)row * tp_kg,
                          down_s_src + (size_t)row * full_kg + down_scale_col_off,
                          tp_kg);
            }
          },
          nullptr);
    });

    // Call per-TP load_weights (does FP8 memcpy into BufferB.b + ue8m0→FP32 into BufferB.d)
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      tps[i]->load_weights();
    });

    // Free temporary buffers
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
      delete[] (uint8_t*)tpc.gate_scale;
      delete[] (uint8_t*)tpc.up_scale;
      delete[] (uint8_t*)tpc.down_scale;
    });

    this->weights_loaded = true;
  }

  // --------------------------------------------------------------------------
  // write_weight_scale_to_buffer: orchestrator for layerwise prefill.
  // Called once per expert by Python (kt-sglang/.../kt_ep_wrapper.py:_prepare_weight_fp8).
  // Dispatches across all NUMA TP parts; each part runs its own
  // AVX2_MXFP8_MOE_TP::write_weights_to_buffer to mirror its slice of the expert
  // into the pre-allocated GPU pinned staging buffer.
  //
  // Mirrors amx/mxfp8-moe.hpp:897-912.
  // SFINAE in ext_bindings.cpp:435 auto-detects this method and exposes
  // `moe.write_weight_scale_to_buffer_task(...)` to Python — no manual binding needed.
  // --------------------------------------------------------------------------
  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != (size_t)gpu_tp_count ||
        w13_scale_ptrs.size()  != (size_t)gpu_tp_count ||
        w2_weight_ptrs.size()  != (size_t)gpu_tp_count ||
        w2_scale_ptrs.size()   != (size_t)gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config,
                                            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_MXFP8_MOE_H
