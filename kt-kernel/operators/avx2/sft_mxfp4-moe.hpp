/**
 * @file sft_mxfp4-moe.hpp
 * @brief AVX2 tier of the MXFP4 routed-expert LoRA SFT part (frozen native
 *        E2M1/UE8M0 base, LoRA on gate/up/down), the per-NUMA class behind
 *        TP_MOE_SFT<...> on machines without AVX-512 (Zen 2/3, older Xeons).
 *
 * Mirrors the semantics of operators/amx/sft_moe.hpp for the MXFP4 backend:
 * base projections through the AVX2 MXFP4 GEMM of mxfp4-moe.hpp, LoRA added
 * before the DeepSeek-V4 asymmetric SwiGLU clamp, base dX straight from the
 * packed weights (no transposed copy), authoritative LoRA gradients written in
 * the layouts TP_MOE_SFT expects (bf16 slices for gate/up B and down A, sparse
 * FP32 partials for gate/up A and down B).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef CPUINFER_OPERATOR_AVX2_SFT_MXFP4_MOE_H
#define CPUINFER_OPERATOR_AVX2_SFT_MXFP4_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../sft_profile.hpp"
#include "avx2_bf16_utils.hpp"
#include "mxfp4-moe.hpp"

namespace avx2 {

// ---------------------------------------------------------------------------
// Small dense kernels (FP32 accumulate) used by the LoRA path and the glue.
// All take an (ith, nth) split of their leading output dimension so they run
// under the worker pool.
// ---------------------------------------------------------------------------

// out[m, r] = scale * X[m, k](bf16, row stride ldx) @ A[r, k](bf16)^T
static inline void sft_gemm_x_at(int m, int k, int r, const ggml_bf16_t* x, size_t ldx, const ggml_bf16_t* a,
                                 float scale, float* out, int ith, int nth) {
  auto [m0, m1] = split_range(m, ith, nth);
  for (int mi = m0; mi < m1; mi++) {
    const ggml_bf16_t* xr = x + (size_t)mi * ldx;
    for (int j = 0; j < r; j++) {
      const ggml_bf16_t* ar = a + (size_t)j * k;
      __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
      int ki = 0;
      for (; ki + 16 <= k; ki += 16) {
        c0 = _mm256_fmadd_ps(load_bf16_to_fp32(xr + ki), load_bf16_to_fp32(ar + ki), c0);
        c1 = _mm256_fmadd_ps(load_bf16_to_fp32(xr + ki + 8), load_bf16_to_fp32(ar + ki + 8), c1);
      }
      float s = hsum_avx2(_mm256_add_ps(c0, c1));
      for (; ki < k; ki++) s += GGML_BF16_TO_FP32(xr[ki]) * GGML_BF16_TO_FP32(ar[ki]);
      out[(size_t)mi * r + j] = scale * s;
    }
  }
}

// out[m, r] = G[m, n](fp32) @ B[n, r](bf16)  computed as dot(G[i], Bt[j]) with Bt[r, n] fp32
static inline void sft_gemm_g_bt(int m, int n, int r, const float* g, const float* bt, float* out, int ith, int nth) {
  auto [m0, m1] = split_range(m, ith, nth);
  for (int mi = m0; mi < m1; mi++) {
    const float* gr = g + (size_t)mi * n;
    for (int j = 0; j < r; j++) {
      const float* br = bt + (size_t)j * n;
      __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
      int ni = 0;
      for (; ni + 16 <= n; ni += 16) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(gr + ni), _mm256_loadu_ps(br + ni), c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(gr + ni + 8), _mm256_loadu_ps(br + ni + 8), c1);
      }
      float s = hsum_avx2(_mm256_add_ps(c0, c1));
      for (; ni < n; ni++) s += gr[ni] * br[ni];
      out[(size_t)mi * r + j] = s;
    }
  }
}

// out[m, n](fp32) += scale * U[m, r](fp32) @ Bt[r, n](fp32); split over n
static inline void sft_gemm_u_bt_acc(int m, int r, int n, const float* u, const float* bt, float scale, float* out,
                                     int ith, int nth) {
  auto [n0, n1] = split_range(n, ith, nth);
  for (int mi = 0; mi < m; mi++) {
    const float* ur = u + (size_t)mi * r;
    float* orow = out + (size_t)mi * n;
    int ni = n0;
    for (; ni + 8 <= n1; ni += 8) {
      __m256 acc = _mm256_loadu_ps(orow + ni);
      for (int j = 0; j < r; j++) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scale * ur[j]), _mm256_loadu_ps(bt + (size_t)j * n + ni), acc);
      }
      _mm256_storeu_ps(orow + ni, acc);
    }
    for (; ni < n1; ni++) {
      float s = orow[ni];
      for (int j = 0; j < r; j++) s += scale * ur[j] * bt[(size_t)j * n + ni];
      orow[ni] = s;
    }
  }
}

// out[m, k](fp32) += scale * V[m, r](fp32) @ A[r, k](bf16); split over k
static inline void sft_gemm_v_a_acc(int m, int r, int k, const float* v, const ggml_bf16_t* a, float scale, float* out,
                                    int ith, int nth) {
  auto [k0, k1] = split_range(k, ith, nth);
  for (int mi = 0; mi < m; mi++) {
    const float* vr = v + (size_t)mi * r;
    float* orow = out + (size_t)mi * k;
    int ki = k0;
    for (; ki + 8 <= k1; ki += 8) {
      __m256 acc = _mm256_loadu_ps(orow + ki);
      for (int j = 0; j < r; j++) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scale * vr[j]), load_bf16_to_fp32(a + (size_t)j * k + ki), acc);
      }
      _mm256_storeu_ps(orow + ki, acc);
    }
    for (; ki < k1; ki++) {
      float s = orow[ki];
      for (int j = 0; j < r; j++) s += scale * vr[j] * GGML_BF16_TO_FP32(a[(size_t)j * k + ki]);
      orow[ki] = s;
    }
  }
}

// dB[n, r](fp32) += scale * G[m, n](fp32)^T @ U[m, r](fp32); split over n
static inline void sft_gemm_gt_u_acc(int m, int n, int r, const float* g, const float* u, float scale, float* db,
                                     int ith, int nth) {
  auto [n0, n1] = split_range(n, ith, nth);
  for (int ni = n0; ni < n1; ni++) {
    float* row = db + (size_t)ni * r;
    for (int j = 0; j < r; j++) {
      float s = 0.f;
      for (int mi = 0; mi < m; mi++) s += g[(size_t)mi * n + ni] * u[(size_t)mi * r + j];
      row[j] += scale * s;
    }
  }
}

// dA[r, k](fp32) += scale * V[m, r](fp32)^T @ X[m, k](bf16, row stride ldx); split over k
static inline void sft_gemm_vt_x_acc(int m, int r, int k, const float* v, const ggml_bf16_t* x, size_t ldx, float scale,
                                     float* da, int ith, int nth) {
  auto [k0, k1] = split_range(k, ith, nth);
  for (int j = 0; j < r; j++) {
    float* row = da + (size_t)j * k;
    int ki = k0;
    for (; ki + 8 <= k1; ki += 8) {
      __m256 acc = _mm256_loadu_ps(row + ki);
      for (int mi = 0; mi < m; mi++) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scale * v[(size_t)mi * r + j]), load_bf16_to_fp32(x + (size_t)mi * ldx + ki),
                              acc);
      }
      _mm256_storeu_ps(row + ki, acc);
    }
    for (; ki < k1; ki++) {
      float s = row[ki];
      for (int mi = 0; mi < m; mi++) s += scale * v[(size_t)mi * r + j] * GGML_BF16_TO_FP32(x[(size_t)mi * ldx + ki]);
      row[ki] = s;
    }
  }
}

// Bt[r, n] fp32 <- B[n, r] bf16 (transpose, small)
static inline void sft_transpose_b(int n, int r, const ggml_bf16_t* b, float* bt) {
  for (int ni = 0; ni < n; ni++)
    for (int j = 0; j < r; j++) bt[(size_t)j * n + ni] = GGML_BF16_TO_FP32(b[(size_t)ni * r + j]);
}

// ---------------------------------------------------------------------------
// Base dX straight from the packed MXFP4 forward weight:
//   dX[m, k] = dY[m, n](fp32) @ W[n, k]   (W row-major packed E2M1, fp32 group scales)
// Each worker owns whole 32-column groups of k; the decode of #2175's fast
// path is reused, so the accumulators live in the same permuted order and are
// un-permuted once on store.  Tokens are blocked by 8 so one decoded group
// feeds 8 x 4 accumulators (32 ymm total with the 4 weight vectors: the
// compiler spills a little; 8 is still ~2x fewer decodes than 4).
// ---------------------------------------------------------------------------
static inline void sft_dx_mxfp4(int m, int n, int k, const float* dy, GemmKernelAVX2MXFP4::BufferB& w, float* dx,
                                int ith, int nth) {
  if (w.b == nullptr) throw std::runtime_error("sft_dx_mxfp4: packed weight is null");
  if (w.k_group_size != 32 || (k % 32) != 0) throw std::runtime_error("sft_dx_mxfp4 needs group-32 weights");
  static constexpr int kPerm[32] = {0,  2,  4,  6,  1,  3,  5,  7,  8,  10, 12, 14, 9,  11, 13, 15,
                                    16, 18, 20, 22, 17, 19, 21, 23, 24, 26, 28, 30, 25, 27, 29, 31};
  const int group_count = k / 32;
  auto [g0, g1] = split_range(group_count, ith, nth);
  if (g0 >= g1) return;
  const size_t row_bytes = (size_t)k / 2;
  const __m128i lut_lo = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_lo);
  const __m128i lut_hi = _mm_load_si128((const __m128i*)GemmKernelAVX2MXFP4::fp4_bf16_hi);
  const __m256i lut_lo256 = _mm256_broadcastsi128_si256(lut_lo);
  const __m256i lut_hi256 = _mm256_broadcastsi128_si256(lut_hi);
  const __m256i zero256 = _mm256_setzero_si256();
  const __m128i nib_mask = _mm_set1_epi8(0x0F);
  constexpr int TB = 8;
  alignas(32) float tmp[TB][32];

  for (int mb = 0; mb < m; mb += TB) {
    const int tcount = std::min(TB, m - mb);
    for (int g = g0; g < g1; g++) {
      __m256 acc[TB][4];
      for (int t = 0; t < TB; t++)
        for (int q = 0; q < 4; q++) acc[t][q] = _mm256_setzero_ps();
      for (int ni = 0; ni < n; ni++) {
        const uint8_t* b_row = w.b + (size_t)ni * row_bytes + (size_t)g * 16;
        const float scale = w.d[(size_t)ni * group_count + g];
        const __m128i raw = _mm_loadu_si128((const __m128i*)b_row);
        const __m128i lo = _mm_and_si128(raw, nib_mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nib_mask);
        const __m256i v = _mm256_set_m128i(hi, lo);
        const __m256i bl = _mm256_shuffle_epi8(lut_lo256, v);
        const __m256i bh = _mm256_shuffle_epi8(lut_hi256, v);
        const __m256i u16a = _mm256_unpacklo_epi8(bl, bh);
        const __m256i u16b = _mm256_unpackhi_epi8(bl, bh);
        const __m256 w0 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16a));
        const __m256 w1 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16a));
        const __m256 w2 = _mm256_castsi256_ps(_mm256_unpacklo_epi16(zero256, u16b));
        const __m256 w3 = _mm256_castsi256_ps(_mm256_unpackhi_epi16(zero256, u16b));
        for (int t = 0; t < tcount; t++) {
          const __m256 c = _mm256_set1_ps(dy[(size_t)(mb + t) * n + ni] * scale);
          acc[t][0] = _mm256_fmadd_ps(c, w0, acc[t][0]);
          acc[t][1] = _mm256_fmadd_ps(c, w1, acc[t][1]);
          acc[t][2] = _mm256_fmadd_ps(c, w2, acc[t][2]);
          acc[t][3] = _mm256_fmadd_ps(c, w3, acc[t][3]);
        }
      }
      // un-permute: decoded lane j holds column kPerm[j] of the group
      for (int t = 0; t < tcount; t++) {
        _mm256_store_ps(tmp[t], acc[t][0]);
        _mm256_store_ps(tmp[t] + 8, acc[t][1]);
        _mm256_store_ps(tmp[t] + 16, acc[t][2]);
        _mm256_store_ps(tmp[t] + 24, acc[t][3]);
        float* dst = dx + (size_t)(mb + t) * k + (size_t)g * 32;
        for (int j = 0; j < 32; j++) dst[kPerm[j]] = tmp[t][j];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// The part
// ---------------------------------------------------------------------------
template <class T = avx2::GemmKernelAVX2MXFP4>
class AVX2_SFT_MXFP4_MOE_TP : public AVX2_MXFP4_MOE_TP<T> {
  using Base = AVX2_MXFP4_MOE_TP<T>;
  // AVX2_MXFP4_MOE_TP re-declares these privately; take them from the root base.
  using Root = AVX2_MOE_BASE<T, AVX2_MXFP4_MOE_TP<T>>;
  using Root::config_;
  using Root::down_ba_;
  using Root::down_bb_;
  using Root::down_bc_;
  using Root::gate_bb_;
  using Root::gate_bc_;
  using Root::gate_up_ba_;
  using Root::m_expert_id_map_;
  using Root::m_local_down_output_ptr_;
  using Root::m_local_gate_output_ptr_;
  using Root::m_local_input_ptr_;
  using Root::m_local_num_;
  using Root::m_local_pos_;
  using Root::m_local_up_output_ptr_;
  using Root::tp_part_idx;
  using Root::up_bb_;
  using Root::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;
  static constexpr bool kSkipLoRA = false;
  static constexpr bool kIsInt8Backend = false;
  static constexpr bool kIsFP8Backend = false;
  static constexpr bool kIsMXFP4Backend = true;
  static constexpr bool kSupportsDirectBf16Reload = false;
  static constexpr bool kSupportsAuthoritativeBaseGrads = false;
  static constexpr bool kSupportsAuthoritativeLoraGrads = true;

  MOESFTConfig sft_config_;

  AVX2_SFT_MXFP4_MOE_TP(MOESFTConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_), sft_config_(config) {
    if (config.quant_config.group_size != 32 || config.quant_config.zero_point)
      throw std::invalid_argument("MXFP4 SFT (AVX2) requires zero-point-free group-32 E2M1 weights");
    if (config.hidden_size % 32 != 0 || config.intermediate_size % 32 != 0)
      throw std::invalid_argument("MXFP4 SFT (AVX2) requires 32-aligned hidden and intermediate sizes");
    if (!std::isfinite(config.swiglu_limit) || config.swiglu_limit <= 0.0f || config.swiglu_alpha != 0.0f)
      throw std::invalid_argument("MXFP4 SFT requires the DeepSeek-V4 asymmetric SwiGLU clamp (swiglu_limit > 0)");
    if (config.full_weight_grad) throw std::invalid_argument("MXFP4 SFT supports frozen-base LoRA only");
    if (config.lora_dropout != 0.0f)
      throw std::invalid_argument("MXFP4 SFT (AVX2) does not implement lora_dropout yet");
    rank_ = config.lora_rank;
    scaling_ = config.lora_scaling();
    update_lora_weights(config.gate_lora_a, config.gate_lora_b, config.up_lora_a, config.up_lora_b,
                        config.down_lora_a, config.down_lora_b);
    cache_stack_.resize(std::max(1, config.max_cache_depth));
    printf("Created AVX2_SFT_MXFP4_MOE_TP %d (rank %d, alpha %.1f, limit %.1f)\n", tp_part_idx, rank_,
           config.lora_alpha, config.swiglu_limit);
  }
  AVX2_SFT_MXFP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_)
      : AVX2_SFT_MXFP4_MOE_TP(MOESFTConfig(config), tp_part_idx_) {}


  // ---- weights ------------------------------------------------------------
  void set_staged_weight_pointers(void* gate, void* up, void* down, void* gate_scale, void* up_scale,
                                  void* down_scale) {
    config_.gate_proj = gate;
    config_.up_proj = up;
    config_.down_proj = down;
    config_.gate_scale = gate_scale;
    config_.up_scale = up_scale;
    config_.down_scale = down_scale;
    // The TP wrapper stages this part's intermediate slice into flat buffers even when the
    // checkpoint arrived as per-expert pointers; the base loader picks per-expert mode whenever
    // gate_projs is non-empty (and rejects tp_part_idx > 0 there), so drop the per-expert view.
    if (gate != nullptr) {
      config_.gate_projs.clear();
      config_.up_projs.clear();
      config_.down_projs.clear();
      config_.gate_scales.clear();
      config_.up_scales.clear();
      config_.down_scales.clear();
    }
  }
  void clear_staged_weight_pointers() { set_staged_weight_pointers(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr); }
  void set_physical_to_logical_map(const void* map) { config_.physical_to_logical_map = const_cast<void*>(map); }

  void validate_mxfp4_scales() const {
    const size_t count = (size_t)config_.hidden_size * config_.intermediate_size / 32;
    for (int e = 0; e < config_.expert_num; e++) {
      for (const auto* bb : {gate_bb_[e].get(), up_bb_[e].get(), down_bb_[e].get()}) {
        if (bb == nullptr || bb->d == nullptr) continue;
        for (size_t i = 0; i < count; i++) {
          if (!std::isfinite(bb->d[i])) throw std::runtime_error("MXFP4 SFT requires finite native UE8M0 group scales");
        }
      }
    }
  }

  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    gate_lora_a_ = (const ggml_bf16_t*)gate_lora_a;
    gate_lora_b_ = (const ggml_bf16_t*)gate_lora_b;
    up_lora_a_ = (const ggml_bf16_t*)up_lora_a;
    up_lora_b_ = (const ggml_bf16_t*)up_lora_b;
    down_lora_a_ = (const ggml_bf16_t*)down_lora_a;
    down_lora_b_ = (const ggml_bf16_t*)down_lora_b;
    bt_valid_ = false;
  }

  // Frozen-base stubs (the wrapper only calls these for bf16/full-grad backends).
  void prepare_bwd(void*, void*, void*) {}
  void prepare_backward_weights_from_forward() {}
  void prepare_backward_bb_for_async() {}
  void load_backward_weights_from_projs() {}
  void load_forward_weights_from_full_bf16(void*, void*, void*, int, int) {
    throw std::logic_error("MXFP4 SFT base weights are frozen packed tensors");
  }
  void set_weight_pointers_for_forward(void*, void*, void*) {
    throw std::logic_error("MXFP4 SFT base weights are frozen packed tensors");
  }
  void save_backward_weights(const std::string&) {
    throw std::logic_error("MXFP4 SFT backward reads the packed forward weights; there is no bf16 backward copy");
  }
  void set_full_weight_grad(bool enabled) {
    if (enabled) throw std::invalid_argument("MXFP4 SFT keeps routed-expert base weights frozen");
  }

  // ---- cache accessors (the wrapper reads the top before backward pops) ----
  int get_cache_qlen() const { return top_cache().qlen; }
  int get_cache_activated_expert_count() const { return top_cache().activated; }
  const int* get_cache_expert_id_map() const { return top_cache().expert_id_map.data(); }
  const std::vector<int>& get_expert_token_distribution() const { return last_backward_expert_tokens_; }

  void append_profile_stats(std::map<std::string, double>& out, const std::string& prefix, bool reset_after = false) {
    profiler_.append(out, prefix, reset_after);
  }
  void reset_profile_stats() { profiler_.reset(); }

  // =========================================================================
  // forward
  // =========================================================================
  void forward_sft(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
                   bool save_for_backward) {
    SFTProfileScope total_scope(profiler_, SFTProfileStage::FwdTotal);
    if (qlen > config_.max_len)
      throw std::runtime_error("MXFP4 SFT forward: qlen " + std::to_string(qlen) + " exceeds max_len " +
                               std::to_string(config_.max_len));
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int K = config_.hidden_size, N = config_.intermediate_size, r = rank_;
    ensure_lora_bt();

    // 1. route
    int activated = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++)
      for (int j = 0; j < k; j++) {
        const int64_t e = expert_ids[i * k + j];
        if (e < config_.num_gpu_experts || e >= config_.expert_num) continue;
        m_local_pos_[i][j] = m_local_num_[e]++;
      }
    for (int e = 0; e < config_.expert_num; e++)
      if (m_local_num_[e] > 0) m_expert_id_map_[activated++] = e;
    carve_pools();
    const int tokens_total = row_offset_[config_.expert_num];
    ensure_work(tokens_total);

    // 2. gather input rows
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            const int64_t e = expert_ids[i * k + j];
            if (e < config_.num_gpu_experts || e >= config_.expert_num) continue;
            std::memcpy(m_local_input_ptr_[e] + (size_t)m_local_pos_[i][j] * K, (const ggml_bf16_t*)input + (size_t)i * K,
                        sizeof(ggml_bf16_t) * K);
          }
        },
        nullptr);
    pool->do_work_stealing_job(
        activated, nullptr,
        [&](int t) {
          const int e = m_expert_id_map_[t];
          gate_up_ba_[e]->from_mat(m_local_num_[e], m_local_input_ptr_[e], 0, 1);
        },
        nullptr);

    // 3. base gate/up GEMM -> fp32 BufferC
    {
      const int nth = T::recommended_nth(N);
      pool->do_work_stealing_job(
          nth * activated * 2, nullptr,
          [&](int task) {
            const bool do_up = task & 1;
            const int t = (task >> 1) / nth, ith = (task >> 1) % nth;
            Base::do_gate_up_gemm(do_up, m_expert_id_map_[t], ith, nth, qlen);
          },
          nullptr);
    }

    // 4. LoRA gate/up: U = scaling * x @ A^T (fp32 [m, r]); out += U @ Bt
    if (has_lora()) {
      const int nth = std::max(1, std::min(16, tokens_total / 32 + 1));
      pool->do_work_stealing_job(
          activated * nth * 2, nullptr,
          [&](int task) {
            const bool up = task & 1;
            const int t = (task >> 1) / nth, ith = (task >> 1) % nth;
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            const ggml_bf16_t* a = (up ? up_lora_a_ : gate_lora_a_) + (size_t)e * r * K;
            float* u = (up ? u_up_ : u_gate_) + (size_t)row_offset_[e] * r;
            sft_gemm_x_at(m, K, r, m_local_input_ptr_[e], K, a, scaling_, u, ith, nth);
          },
          nullptr);
      const int nthn = T::recommended_nth(N);
      pool->do_work_stealing_job(
          activated * nthn * 2, nullptr,
          [&](int task) {
            const bool up = task & 1;
            const int t = (task >> 1) / nthn, ith = (task >> 1) % nthn;
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            const float* u = (up ? u_up_ : u_gate_) + (size_t)row_offset_[e] * r;
            const float* bt = (up ? bt_up_ : bt_gate_) + (size_t)e * r * N;
            float* out = (up ? up_bc_[e] : gate_bc_[e])->data;
            sft_gemm_u_bt_acc(m, r, N, u, bt, 1.0f, out, ith, nthn);
          },
          nullptr);
    }

    // 5. cache gate/up (pre-clamp) as bf16, routing, input
    ForwardCache* cache = nullptr;
    if (save_for_backward) {
      cache = &push_cache();
      cache->qlen = qlen;
      cache->k = k;
      cache->tokens_total = tokens_total;
      cache->activated = activated;
      cache->expert_ids.assign(expert_ids, expert_ids + (size_t)qlen * k);
      cache->weights.assign(weights, weights + (size_t)qlen * k);
      cache->local_num.assign(m_local_num_.begin(), m_local_num_.end());
      cache->local_pos.assign(qlen, std::vector<int>(k, 0));
      for (int i = 0; i < qlen; i++)
        for (int j = 0; j < k; j++) cache->local_pos[i][j] = m_local_pos_[i][j];
      cache->expert_id_map.assign(m_expert_id_map_.begin(), m_expert_id_map_.begin() + activated);
      cache->row_offset = row_offset_;
      cache->input.resize((size_t)qlen * K);
      std::memcpy(cache->input.data(), input, sizeof(ggml_bf16_t) * (size_t)qlen * K);
      cache->gate_out.resize((size_t)tokens_total * N);
      cache->up_out.resize((size_t)tokens_total * N);
      cache->inter.resize((size_t)tokens_total * N);
      cache->down_out.resize((size_t)tokens_total * K);
      cache->u_down.assign((size_t)tokens_total * std::max(r, 1), 0.f);
      pool->do_work_stealing_job(
          activated, nullptr,
          [&](int t) {
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            fp32_to_bf16(cache->gate_out.data() + (size_t)row_offset_[e] * N, gate_bc_[e]->data, (size_t)m * N);
            fp32_to_bf16(cache->up_out.data() + (size_t)row_offset_[e] * N, up_bc_[e]->data, (size_t)m * N);
          },
          nullptr);
    }

    // 6. activation with the V4 clamp -> bf16 intermediate (down GEMM input)
    {
      const float L = config_.swiglu_limit;
      pool->do_work_stealing_job(
          activated, nullptr,
          [&](int t) {
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            const float* g = gate_bc_[e]->data;
            const float* u = up_bc_[e]->data;
            ggml_bf16_t* h = m_local_gate_output_ptr_[e];  // reuse the base's bf16 intermediate buffer
            const size_t total = (size_t)m * N;
            const __m256 lim = _mm256_set1_ps(L), nlim = _mm256_set1_ps(-L), one = _mm256_set1_ps(1.f);
            size_t i = 0;
            for (; i + 8 <= total; i += 8) {
              __m256 gv = _mm256_min_ps(_mm256_loadu_ps(g + i), lim);
              __m256 uv = _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(u + i), nlim), lim);
              __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), gv))));
              store_fp32_to_bf16(h + i, _mm256_mul_ps(_mm256_mul_ps(gv, sig), uv));
            }
            for (; i < total; i++) {
              float gv = std::min(g[i], L), uv = std::min(std::max(u[i], -L), L);
              h[i] = GGML_FP32_TO_BF16(gv * sigmoid_scalar(gv) * uv);
            }
            if (cache) std::memcpy(cache->inter.data() + (size_t)row_offset_[e] * N, h, sizeof(ggml_bf16_t) * total);
            down_ba_[e]->from_mat(m, h, 0, 1);
          },
          nullptr);
    }

    // 7. base down GEMM -> fp32 BufferC; LoRA down
    {
      const int nth = T::recommended_nth(K);
      pool->do_work_stealing_job(
          nth * activated, nullptr,
          [&](int task) { Base::do_down_gemm(m_expert_id_map_[task / nth], task % nth, nth, qlen); }, nullptr);
    }
    if (has_lora()) {
      const int nth = std::max(1, std::min(16, tokens_total / 32 + 1));
      pool->do_work_stealing_job(
          activated * nth, nullptr,
          [&](int task) {
            const int t = task / nth, ith = task % nth;
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            float* u = (cache ? cache->u_down.data() : u_down_) + (size_t)row_offset_[e] * r;
            sft_gemm_x_at(m, N, r, m_local_gate_output_ptr_[e], N, down_lora_a_ + (size_t)e * r * N, scaling_, u, ith,
                          nth);
          },
          nullptr);
      const int nthk = T::recommended_nth(K);
      pool->do_work_stealing_job(
          activated * nthk, nullptr,
          [&](int task) {
            const int t = task / nthk, ith = task % nthk;
            const int e = m_expert_id_map_[t];
            const int m = m_local_num_[e];
            const float* u = (cache ? cache->u_down.data() : u_down_) + (size_t)row_offset_[e] * r;
            sft_gemm_u_bt_acc(m, r, K, u, bt_down_ + (size_t)e * r * K, 1.0f, down_bc_[e]->data, ith, nthk);
          },
          nullptr);
    }
    if (cache) {
      pool->do_work_stealing_job(
          activated, nullptr,
          [&](int t) {
            const int e = m_expert_id_map_[t];
            fp32_to_bf16(cache->down_out.data() + (size_t)row_offset_[e] * K, down_bc_[e]->data,
                         (size_t)m_local_num_[e] * K);
          },
          nullptr);
    }

    // 8. weighted merge -> fp32 part output (the TP wrapper sums the parts and rounds to bf16)
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          float* out = (float*)output + (size_t)i * K;
          for (int c = 0; c < K; c += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int j = 0; j < k; j++) {
              const int64_t e = expert_ids[i * k + j];
              if (e < config_.num_gpu_experts || e >= config_.expert_num) continue;
              const float* row = down_bc_[e]->data + (size_t)m_local_pos_[i][j] * K + c;
              acc = _mm256_fmadd_ps(_mm256_set1_ps(weights[i * k + j]), _mm256_loadu_ps(row), acc);
            }
            _mm256_storeu_ps(out + c, acc);
          }
        },
        nullptr);
  }

  // =========================================================================
  // backward
  // =========================================================================
  void backward(const void* grad_output, void* grad_input, void* /*grad_gate_lora_a*/, void* grad_gate_lora_b,
                void* /*grad_up_lora_a*/, void* grad_up_lora_b, void* grad_down_lora_a, void* /*grad_down_lora_b*/,
                void* grad_weights, int full_intermediate_size = 0, float* fp32_grad_down_lora_b = nullptr,
                float* fp32_grad_gate_lora_a = nullptr, float* fp32_grad_up_lora_a = nullptr,
                void* grad_gate_proj = nullptr, void* grad_up_proj = nullptr, void* grad_down_proj = nullptr,
                bool accumulate_optimizer_grads = false, float optimizer_grad_scale = 1.0f) {
    SFTProfileScope total_scope(profiler_, SFTProfileStage::BwdTotal);
    if (grad_gate_proj || grad_up_proj || grad_down_proj)
      throw std::invalid_argument("MXFP4 SFT does not accept routed-expert base-gradient outputs");
    ForwardCache cache = pop_cache();
    if (!cache.valid) throw std::runtime_error("No valid forward cache for backward");
    if (full_intermediate_size == 0) full_intermediate_size = config_.intermediate_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int K = config_.hidden_size, N = config_.intermediate_size, r = rank_;
    const int qlen = cache.qlen, k = cache.k, activated = cache.activated, tokens_total = cache.tokens_total;
    const int64_t* expert_ids = cache.expert_ids.data();
    const float* weights = cache.weights.data();
    const auto& row_offset = cache.row_offset;
    const bool want_lora_grads = has_lora() && (fp32_grad_gate_lora_a || grad_gate_lora_b || grad_down_lora_a ||
                                                fp32_grad_down_lora_b || fp32_grad_up_lora_a || grad_up_lora_b);
    last_backward_expert_tokens_.assign(cache.local_num.begin(), cache.local_num.end());
    ensure_lora_bt();
    ensure_bwd(tokens_total);
    // Gradient outputs follow the AMX part's convention: the sparse fp32
    // partials ([active, ...], in cache.expert_id_map order) and the bf16
    // slices are always *added to*; TP_MOE_SFT clears them beforehand when the
    // optimizer window starts (overwrite) and leaves them when accumulating.
    (void)accumulate_optimizer_grads;
    // 0. the cached input rows in per-expert order (LoRA A gradients need them)
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            const int64_t e = expert_ids[i * k + j];
            if (e < config_.num_gpu_experts || e >= config_.expert_num) continue;
            std::memcpy(x_rows_ + ((size_t)row_offset[e] + cache.local_pos[i][j]) * K, cache.input.data() + (size_t)i * K,
                        sizeof(ggml_bf16_t) * K);
          }
        },
        nullptr);

    // 1. gy rows = w * grad_output ; grad_weights partial
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          const ggml_bf16_t* go = (const ggml_bf16_t*)grad_output + (size_t)i * K;
          for (int j = 0; j < k; j++) {
            const int64_t e = expert_ids[i * k + j];
            if (e < config_.num_gpu_experts || e >= config_.expert_num) {
              if (grad_weights) ((float*)grad_weights)[i * k + j] = 0.f;
              continue;
            }
            const size_t row = (size_t)row_offset[e] + cache.local_pos[i][j];
            float* gy = gy_ + row * K;
            const __m256 wv = _mm256_set1_ps(weights[i * k + j]);
            __m256 dot = _mm256_setzero_ps();
            const ggml_bf16_t* dout = cache.down_out.data() + row * K;
            for (int c = 0; c < K; c += 8) {
              const __m256 gv = load_bf16_to_fp32(go + c);
              _mm256_storeu_ps(gy + c, _mm256_mul_ps(gv, wv));
              if (grad_weights) dot = _mm256_fmadd_ps(gv, load_bf16_to_fp32(dout + c), dot);
            }
            if (grad_weights) ((float*)grad_weights)[i * k + j] = hsum_avx2(dot);
          }
        },
        nullptr);

    // 2. down backward: grad_h = gy @ Wd + scaling * (gy @ Bd) @ Ad ; dBd, dAd
    {
      const int nth = T::recommended_nth(N);
      pool->do_work_stealing_job(
          activated * nth, nullptr,
          [&](int task) {
            const int t = task / nth, ith = task % nth;
            const int e = cache.expert_id_map[t];
            const int m = cache.local_num[e];
            sft_dx_mxfp4(m, K, N, gy_ + (size_t)row_offset[e] * K, *down_bb_[e], gh_ + (size_t)row_offset[e] * N, ith,
                         nth);
          },
          nullptr);
      if (has_lora()) {
        const int nthm = std::max(1, std::min(16, tokens_total / 32 + 1));
        pool->do_work_stealing_job(
            activated * nthm, nullptr,
            [&](int task) {
              const int t = task / nthm, ith = task % nthm;
              const int e = cache.expert_id_map[t];
              const int m = cache.local_num[e];
              sft_gemm_g_bt(m, K, r, gy_ + (size_t)row_offset[e] * K, bt_down_ + (size_t)e * r * K,
                            v_ + (size_t)row_offset[e] * r, ith, nthm);
            },
            nullptr);
        pool->do_work_stealing_job(
            activated * nth, nullptr,
            [&](int task) {
              const int t = task / nth, ith = task % nth;
              const int e = cache.expert_id_map[t];
              const int m = cache.local_num[e];
              sft_gemm_v_a_acc(m, r, N, v_ + (size_t)row_offset[e] * r, down_lora_a_ + (size_t)e * r * N, scaling_,
                               gh_ + (size_t)row_offset[e] * N, ith, nth);
            },
            nullptr);
        if (want_lora_grads) {
          // dBd (sparse fp32 [active, K, r]) += gy^T @ U_d   (U_d already carries scaling)
          const int nthk = T::recommended_nth(K);
          if (fp32_grad_down_lora_b)
            pool->do_work_stealing_job(
                activated * nthk, nullptr,
                [&](int task) {
                  const int t = task / nthk, ith = task % nthk;
                  const int e = cache.expert_id_map[t];
                  const int m = cache.local_num[e];
                  sft_gemm_gt_u_acc(m, K, r, gy_ + (size_t)row_offset[e] * K, cache.u_down.data() + (size_t)row_offset[e] * r,
                                    optimizer_grad_scale, fp32_grad_down_lora_b + (size_t)t * K * r, ith, nthk);
                },
                nullptr);
          // dAd (bf16 direct, this TP's slice of [E, r, N_full]) += scaling * V^T @ h
          if (grad_down_lora_a)
            pool->do_work_stealing_job(
                activated * nth, nullptr,
                [&](int task) {
                  const int t = task / nth, ith = task % nth;
                  const int e = cache.expert_id_map[t];
                  const int m = cache.local_num[e];
                  float* tmp = da_tmp_ + (size_t)t * r * N;
                  auto [n0, n1] = split_range(N, ith, nth);
                  for (int j = 0; j < r; j++) std::fill(tmp + (size_t)j * N + n0, tmp + (size_t)j * N + n1, 0.f);
                  sft_gemm_vt_x_acc(m, r, N, v_ + (size_t)row_offset[e] * r, cache.inter.data() + (size_t)row_offset[e] * N,
                                    N, scaling_ * optimizer_grad_scale, tmp, ith, nth);
                  for (int j = 0; j < r; j++) {
                    ggml_bf16_t* dst = (ggml_bf16_t*)grad_down_lora_a + ((size_t)e * r + j) * full_intermediate_size;
                    write_bf16_slice(dst + n0, tmp + (size_t)j * N + n0, n1 - n0, true);
                  }
                },
                nullptr);
        }
      }
    }

    // 3. activation backward -> gg, gu (fp32 [tokens, N])
    {
      const float L = config_.swiglu_limit;
      pool->do_work_stealing_job(
          activated, nullptr,
          [&](int t) {
            const int e = cache.expert_id_map[t];
            const size_t base = (size_t)row_offset[e] * N;
            const size_t total = (size_t)cache.local_num[e] * N;
            const ggml_bf16_t* g = cache.gate_out.data() + base;
            const ggml_bf16_t* u = cache.up_out.data() + base;
            const float* gh = gh_ + base;
            float* gg = gg_ + base;
            float* gu = gu_ + base;
            for (size_t i = 0; i < total; i++) {
              const float graw = GGML_BF16_TO_FP32(g[i]), uraw = GGML_BF16_TO_FP32(u[i]);
              const float gv = std::min(graw, L), uv = std::min(std::max(uraw, -L), L);
              const float sig = sigmoid_scalar(gv);
              const float silu = gv * sig;
              const float dsilu = sig * (1.f + gv * (1.f - sig));
              gg[i] = (graw <= L) ? gh[i] * uv * dsilu : 0.f;
              gu[i] = (uraw >= -L && uraw <= L) ? gh[i] * silu : 0.f;
            }
          },
          nullptr);
    }

    // 4. gate/up backward: grad_x = gg @ Wg + gu @ Wu + scaling * (Vg @ Ag + Vu @ Au); dB, dA
    {
      const int nth = T::recommended_nth(K);
      pool->do_work_stealing_job(
          activated * nth, nullptr,
          [&](int task) {
            const int t = task / nth, ith = task % nth;
            const int e = cache.expert_id_map[t];
            const int m = cache.local_num[e];
            float* gx = gx_ + (size_t)row_offset[e] * K;
            sft_dx_mxfp4(m, N, K, gg_ + (size_t)row_offset[e] * N, *gate_bb_[e], gx, ith, nth);
            sft_dx_mxfp4(m, N, K, gu_ + (size_t)row_offset[e] * N, *up_bb_[e], gx2_ + (size_t)row_offset[e] * K, ith, nth);
            auto [g0, g1] = split_range(K / 32, ith, nth);  // the dX kernel's own column split
            for (int mi = 0; mi < m; mi++)
              for (int c = g0 * 32; c < g1 * 32; c++) gx[(size_t)mi * K + c] += gx2_[((size_t)row_offset[e] + mi) * K + c];
          },
          nullptr);
      if (has_lora()) {
        const int nthm = std::max(1, std::min(16, tokens_total / 32 + 1));
        pool->do_work_stealing_job(
            activated * nthm * 2, nullptr,
            [&](int task) {
              const bool up = task & 1;
              const int t = (task >> 1) / nthm, ith = (task >> 1) % nthm;
              const int e = cache.expert_id_map[t];
              const int m = cache.local_num[e];
              sft_gemm_g_bt(m, N, r, (up ? gu_ : gg_) + (size_t)row_offset[e] * N, (up ? bt_up_ : bt_gate_) + (size_t)e * r * N,
                            (up ? vu_ : vg_) + (size_t)row_offset[e] * r, ith, nthm);
            },
            nullptr);
        pool->do_work_stealing_job(
            activated * nth, nullptr,
            [&](int task) {
              const int t = task / nth, ith = task % nth;
              const int e = cache.expert_id_map[t];
              const int m = cache.local_num[e];
              float* gx = gx_ + (size_t)row_offset[e] * K;
              sft_gemm_v_a_acc(m, r, K, vg_ + (size_t)row_offset[e] * r, gate_lora_a_ + (size_t)e * r * K, scaling_, gx, ith, nth);
              sft_gemm_v_a_acc(m, r, K, vu_ + (size_t)row_offset[e] * r, up_lora_a_ + (size_t)e * r * K, scaling_, gx, ith, nth);
            },
            nullptr);
        if (want_lora_grads) {
          // U_g / U_u = scaling * x @ A^T (recomputed from the cached input rows)
          pool->do_work_stealing_job(
              activated * nthm * 2, nullptr,
              [&](int task) {
                const bool up = task & 1;
                const int t = (task >> 1) / nthm, ith = (task >> 1) % nthm;
                const int e = cache.expert_id_map[t];
                const int m = cache.local_num[e];
                sft_gemm_x_at(m, K, r, x_rows_ + (size_t)row_offset[e] * K, K,
                              (up ? up_lora_a_ : gate_lora_a_) + (size_t)e * r * K, scaling_,
                              (up ? u_up_ : u_gate_) + (size_t)row_offset[e] * r, ith, nthm);
              },
              nullptr);
          // dB (bf16 direct, this TP's slice of [E, N_full, r]) += g^T @ U
          const int nthn = T::recommended_nth(N);
          pool->do_work_stealing_job(
              activated * nthn * 2, nullptr,
              [&](int task) {
                const bool up = task & 1;
                const int t = (task >> 1) / nthn, ith = (task >> 1) % nthn;
                void* dst_base = up ? grad_up_lora_b : grad_gate_lora_b;
                if (dst_base == nullptr) return;
                const int e = cache.expert_id_map[t];
                const int m = cache.local_num[e];
                float* tmp = (up ? dbu_tmp_ : dbg_tmp_) + (size_t)t * N * r;
                auto [n0, n1] = split_range(N, ith, nthn);
                std::fill(tmp + (size_t)n0 * r, tmp + (size_t)n1 * r, 0.f);
                sft_gemm_gt_u_acc(m, N, r, (up ? gu_ : gg_) + (size_t)row_offset[e] * N,
                                  (up ? u_up_ : u_gate_) + (size_t)row_offset[e] * r, optimizer_grad_scale, tmp, ith, nthn);
                ggml_bf16_t* dst = (ggml_bf16_t*)dst_base + ((size_t)e * full_intermediate_size + n0) * r;
                write_bf16_slice(dst, tmp + (size_t)n0 * r, (size_t)(n1 - n0) * r, true);
              },
              nullptr);
          // dA (sparse fp32 [active, r, K]) += scaling * V^T @ x
          pool->do_work_stealing_job(
              activated * nth * 2, nullptr,
              [&](int task) {
                const bool up = task & 1;
                const int t = (task >> 1) / nth, ith = (task >> 1) % nth;
                float* dst_base = up ? fp32_grad_up_lora_a : fp32_grad_gate_lora_a;
                if (dst_base == nullptr) return;
                const int e = cache.expert_id_map[t];
                const int m = cache.local_num[e];
                sft_gemm_vt_x_acc(m, r, K, (up ? vu_ : vg_) + (size_t)row_offset[e] * r, x_rows_ + (size_t)row_offset[e] * K, K,
                                  scaling_ * optimizer_grad_scale, dst_base + (size_t)t * r * K, ith, nth);
              },
              nullptr);
        }
      }
    }

    // 5. scatter grad_x rows into grad_input (bf16 [qlen, K], this part's partial)
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          ggml_bf16_t* out = (ggml_bf16_t*)grad_input + (size_t)i * K;
          for (int c = 0; c < K; c += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int j = 0; j < k; j++) {
              const int64_t e = expert_ids[i * k + j];
              if (e < config_.num_gpu_experts || e >= config_.expert_num) continue;
              acc = _mm256_add_ps(acc, _mm256_loadu_ps(gx_ + ((size_t)row_offset[e] + cache.local_pos[i][j]) * K + c));
            }
            store_fp32_to_bf16(out + c, acc);
          }
        },
        nullptr);
  }

 private:
  struct ForwardCache {
    bool valid = false;
    int qlen = 0, k = 0, tokens_total = 0, activated = 0;
    std::vector<int64_t> expert_ids;
    std::vector<float> weights;
    std::vector<int> local_num;
    std::vector<std::vector<int>> local_pos;
    std::vector<int> expert_id_map;
    std::vector<int> row_offset;
    std::vector<ggml_bf16_t> input, gate_out, up_out, inter, down_out;
    std::vector<float> u_down;
  };

  int rank_ = 0;
  float scaling_ = 1.f;
  const ggml_bf16_t* gate_lora_a_ = nullptr;
  const ggml_bf16_t* gate_lora_b_ = nullptr;
  const ggml_bf16_t* up_lora_a_ = nullptr;
  const ggml_bf16_t* up_lora_b_ = nullptr;
  const ggml_bf16_t* down_lora_a_ = nullptr;
  const ggml_bf16_t* down_lora_b_ = nullptr;
  // transposed LoRA B (fp32): gate/up [E, r, N], down [E, r, K]
  std::vector<float> bt_gate_s_, bt_up_s_, bt_down_s_;
  float *bt_gate_ = nullptr, *bt_up_ = nullptr, *bt_down_ = nullptr;
  bool bt_valid_ = false;
  // per-forward working buffers
  std::vector<int> row_offset_;
  std::vector<float> u_gate_s_, u_up_s_, u_down_s_;
  float *u_gate_ = nullptr, *u_up_ = nullptr, *u_down_ = nullptr;
  size_t work_tokens_ = 0;
  // backward working buffers
  std::vector<float> gy_s_, gh_s_, gg_s_, gu_s_, gx_s_, gx2_s_, v_s_, vg_s_, vu_s_, dbg_s_, dbu_s_, da_s_;
  std::vector<ggml_bf16_t> x_rows_s_;
  float *gy_ = nullptr, *gh_ = nullptr, *gg_ = nullptr, *gu_ = nullptr, *gx_ = nullptr, *gx2_ = nullptr, *v_ = nullptr,
        *vg_ = nullptr, *vu_ = nullptr, *dbg_tmp_ = nullptr, *dbu_tmp_ = nullptr, *da_tmp_ = nullptr;
  ggml_bf16_t* x_rows_ = nullptr;
  size_t bwd_tokens_ = 0;
  std::vector<ForwardCache> cache_stack_;
  int cache_top_ = 0;
  std::vector<int> last_backward_expert_tokens_;
  SFTProfiler profiler_;

  bool has_lora() const { return rank_ > 0 && gate_lora_a_ && gate_lora_b_ && up_lora_a_ && up_lora_b_ && down_lora_a_ && down_lora_b_; }

  const ForwardCache& top_cache() const {
    if (cache_top_ == 0) throw std::runtime_error("MXFP4 SFT: no forward cache");
    return cache_stack_[cache_top_ - 1];
  }
  ForwardCache& push_cache() {
    if (cache_top_ >= (int)cache_stack_.size()) throw std::runtime_error("MXFP4 SFT: forward cache stack overflow");
    ForwardCache& c = cache_stack_[cache_top_++];
    c.valid = true;
    return c;
  }
  ForwardCache pop_cache() {
    if (cache_top_ == 0) return ForwardCache{};
    ForwardCache c = std::move(cache_stack_[--cache_top_]);
    cache_stack_[cache_top_] = ForwardCache{};
    return c;
  }

  // Carve the base's per-expert pools exactly as AVX2_MOE_BASE::forward_prefill does.
  void carve_pools() {
    const int K = config_.hidden_size, N = config_.intermediate_size;
    row_offset_.assign(config_.expert_num + 1, 0);
    size_t offset = 0;
    void* ba = Root::gate_up_ba_pool_;
    void* gbc = Root::gate_bc_pool_;
    void* ubc = Root::up_bc_pool_;
    void* dba = Root::down_ba_pool_;
    void* dbc = Root::down_bc_pool_;
    auto align64 = [](size_t v) { return (v + 63) & ~(size_t)63; };
    for (int e = 0; e < config_.expert_num; e++) {
      row_offset_[e] = (int)offset;
      m_local_input_ptr_[e] = Root::m_local_input_ + offset * K;
      m_local_gate_output_ptr_[e] = Base::m_local_gate_output_ + offset * N;
      m_local_up_output_ptr_[e] = Base::m_local_up_output_ + offset * N;
      m_local_down_output_ptr_[e] = Base::m_local_down_output_ + offset * K;
      offset += m_local_num_[e];
      if (m_local_num_[e] == 0) continue;
      const size_t max_m = m_local_num_[e];
      gate_up_ba_[e]->max_m = max_m;
      gate_up_ba_[e]->set_data(ba);
      ba = (void*)((uintptr_t)ba + align64(Base::buffer_a_required_size(max_m, K)));
      gate_bc_[e]->max_m = max_m;
      gate_bc_[e]->set_data(gbc);
      gbc = (void*)((uintptr_t)gbc + align64(Base::buffer_c_required_size(max_m, N)));
      up_bc_[e]->max_m = max_m;
      up_bc_[e]->set_data(ubc);
      ubc = (void*)((uintptr_t)ubc + align64(Base::buffer_c_required_size(max_m, N)));
      down_ba_[e]->max_m = max_m;
      down_ba_[e]->set_data(dba);
      dba = (void*)((uintptr_t)dba + align64(Base::buffer_a_required_size(max_m, N)));
      down_bc_[e]->max_m = max_m;
      down_bc_[e]->set_data(dbc);
      dbc = (void*)((uintptr_t)dbc + align64(Base::buffer_c_required_size(max_m, K)));
    }
    row_offset_[config_.expert_num] = (int)offset;
  }

  void ensure_lora_bt() {
    if (bt_valid_ || !has_lora()) return;
    const int K = config_.hidden_size, N = config_.intermediate_size, r = rank_, E = config_.expert_num;
    bt_gate_s_.resize((size_t)E * r * N);
    bt_up_s_.resize((size_t)E * r * N);
    bt_down_s_.resize((size_t)E * r * K);
    bt_gate_ = bt_gate_s_.data();
    bt_up_ = bt_up_s_.data();
    bt_down_ = bt_down_s_.data();
    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        E, nullptr,
        [&](int e) {
          sft_transpose_b(N, r, gate_lora_b_ + (size_t)e * N * r, bt_gate_ + (size_t)e * r * N);
          sft_transpose_b(N, r, up_lora_b_ + (size_t)e * N * r, bt_up_ + (size_t)e * r * N);
          sft_transpose_b(K, r, down_lora_b_ + (size_t)e * K * r, bt_down_ + (size_t)e * r * K);
        },
        nullptr);
    bt_valid_ = true;
  }

  void ensure_work(int tokens_total) {
    const size_t need = (size_t)std::max(tokens_total, 1) * std::max(rank_, 1);
    if (work_tokens_ >= (size_t)tokens_total && u_gate_) return;
    u_gate_s_.assign(need, 0.f);
    u_up_s_.assign(need, 0.f);
    u_down_s_.assign(need, 0.f);
    u_gate_ = u_gate_s_.data();
    u_up_ = u_up_s_.data();
    u_down_ = u_down_s_.data();
    work_tokens_ = tokens_total;
  }

  void ensure_bwd(int tokens_total) {
    const int K = config_.hidden_size, N = config_.intermediate_size, r = std::max(rank_, 1), E = config_.expert_num;
    const size_t tt = std::max(tokens_total, 1);
    if (bwd_tokens_ >= tt && gy_) return;
    gy_s_.resize(tt * K);
    gh_s_.resize(tt * N);
    gg_s_.resize(tt * N);
    gu_s_.resize(tt * N);
    gx_s_.resize(tt * K);
    gx2_s_.resize(tt * K);
    v_s_.resize(tt * r);
    vg_s_.resize(tt * r);
    vu_s_.resize(tt * r);
    x_rows_s_.resize(tt * K);
    // per-active-expert temporaries for the bf16 slices (bounded by the expert count)
    const size_t active_max = std::min<size_t>(E, tt);
    dbg_s_.resize(active_max * N * r);
    dbu_s_.resize(active_max * N * r);
    da_s_.resize(active_max * r * N);
    gy_ = gy_s_.data(); gh_ = gh_s_.data(); gg_ = gg_s_.data(); gu_ = gu_s_.data();
    gx_ = gx_s_.data(); gx2_ = gx2_s_.data(); v_ = v_s_.data(); vg_ = vg_s_.data(); vu_ = vu_s_.data();
    dbg_tmp_ = dbg_s_.data(); dbu_tmp_ = dbu_s_.data(); da_tmp_ = da_s_.data();
    x_rows_ = x_rows_s_.data();
    bwd_tokens_ = tt;
    // the gathered input rows for the LoRA grads come from the cache's input on demand
  }

  static void fp32_to_bf16(ggml_bf16_t* dst, const float* src, size_t count) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) store_fp32_to_bf16(dst + i, _mm256_loadu_ps(src + i));
    for (; i < count; i++) dst[i] = GGML_FP32_TO_BF16(src[i]);
  }

  // dst[i] = (accumulate ? dst[i] : 0) + src[i], bf16 storage
  static void write_bf16_slice(ggml_bf16_t* dst, const float* src, size_t count, bool accumulate) {
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
      __m256 v = _mm256_loadu_ps(src + i);
      if (accumulate) v = _mm256_add_ps(v, load_bf16_to_fp32(dst + i));
      store_fp32_to_bf16(dst + i, v);
    }
    for (; i < count; i++) {
      float v = src[i];
      if (accumulate) v += GGML_BF16_TO_FP32(dst[i]);
      dst[i] = GGML_FP32_TO_BF16(v);
    }
  }

  // sigmoid for the scalar tails.  The build uses -ffast-math (finite-math-only), so exp() of an
  // argument past the float range is undefined rather than inf: clamp like exp256_ps does.
  static inline float sigmoid_scalar(float x) {
    const float z = std::min(std::max(-x, -87.f), 88.f);
    return 1.f / (1.f + std::exp(z));
  }
  // exp() for 8 lanes: cephes-style polynomial, adequate for the sigmoid here
  static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.f)), _mm256_set1_ps(88.f));
    const __m256 log2e = _mm256_set1_ps(1.44269504f);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.f)));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
  }
};

}  // namespace avx2

#endif  // CPUINFER_OPERATOR_AVX2_SFT_MXFP4_MOE_H
