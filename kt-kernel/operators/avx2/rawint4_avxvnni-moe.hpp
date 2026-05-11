/**
 * @Description  : AVX-VNNI-256 RAWINT4 MoE operator (Kimi native INT4 layout)
 * SPDX-License-Identifier: Apache-2.0
 *
 * This backend consumes the same RAWINT4 source layout as the plain AVX2
 * backend in rawint4-moe.hpp:
 *   weights [N, K/2] uint8  (two signed int4 values per byte, value = nibble - 8,
 *                            low nibble = even k, high nibble = odd k)
 *   scales  [N, K/group_size] bf16
 *
 * To use AVX-VNNI-256 effectively, activations are quantized on the fly to
 * group-wise int8 (biased to uint8 for dpbusd), and the packed signed int4
 * weights are unpacked into signed int8 [N, K] once at load time. dpbusd then
 * does the group-wise inner product, followed by compensation (128 * weight_sum)
 * and rescale by (a_scale * w_scale).
 **/
#ifndef CPUINFER_OPERATOR_AVX2_RAW_INT4_AVXVNNI_MOE_H
#define CPUINFER_OPERATOR_AVX2_RAW_INT4_AVXVNNI_MOE_H

#include <immintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

#include "avx2_bf16_utils.hpp"
#include "moe_base.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define KT_AVXVNNI256_RAWINT4_TARGET __attribute__((target("avx2,avxvnni,fma")))
#else
#define KT_AVXVNNI256_RAWINT4_TARGET
#endif

namespace avxvnni_rawint4 {

static constexpr int MAX_SUPPORTED_GROUP_SIZE = 256;

static inline int hsum_epi32_avx2(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i sum = _mm_add_epi32(lo, hi);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

// Quantize one activation group to biased uint8 for dpbusd.
// Returns the activation scale. A return value of 0 means the group is all zero.
static inline float quantize_activation_group_u8(const ggml_bf16_t* src, int group_size, uint8_t* dst) {
  float absmax = 0.0f;

  for (int i = 0; i < group_size; ++i) {
    absmax = std::max(absmax, std::fabs(GGML_BF16_TO_FP32(src[i])));
  }

  if (absmax <= std::numeric_limits<float>::min()) {
    std::memset(dst, 0x80, (size_t)group_size);
    return 0.0f;
  }

  const float scale = absmax / 127.0f;
  const float inv_scale = 1.0f / scale;
  for (int i = 0; i < group_size; ++i) {
    int q = (int)std::lrint(GGML_BF16_TO_FP32(src[i]) * inv_scale);
    q = std::clamp(q, -127, 127);
    dst[i] = (uint8_t)(((uint8_t)(int8_t)q) ^ 0x80);
  }
  return scale;
}

struct GemmKernelAVXVNNI256RawInt4 {
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

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return avx2::split_range(n, ith, nth); }

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k) { return m * k * sizeof(ggml_bf16_t); }

    void set_data(void* ptr) { data = (ggml_bf16_t*)ptr; }

    void from_mat(int m, const ggml_bf16_t* src, int ith, int nth) {
      if (ith == 0 && nth == 1) {
        std::memcpy(data, src, (size_t)m * k * sizeof(ggml_bf16_t));
      } else {
        auto [m_start, m_end] = avx2::split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k, (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  struct BufferB {
    int8_t* qweight_s8 = nullptr;    // [N, K] unpacked signed int8 weights
    float* scales = nullptr;         // [N, num_groups] float32 (converted from bf16)
    int16_t* weight_sums = nullptr;  // [N, num_groups] per-group sum of unpacked int8 weights
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr) : n((int)n_), k((int)k_), group_size(gs) {
      if (group_size <= 0 || (group_size % 32) != 0) {
        throw std::runtime_error("AVX-VNNI RAWINT4 requires group_size to be a positive multiple of 32");
      }
      if (group_size > MAX_SUPPORTED_GROUP_SIZE) {
        throw std::runtime_error("AVX-VNNI RAWINT4 requires group_size <= 256");
      }
      if ((k % 8) != 0 || (k % group_size) != 0) {
        throw std::runtime_error("AVX-VNNI RAWINT4 requires k to be divisible by both 8 and group_size");
      }
      num_groups = k / group_size;
      qweight_s8 = (int8_t*)ptr;
      scales = (float*)((uint8_t*)ptr + (size_t)k * n * sizeof(int8_t));
      weight_sums = (int16_t*)((uint8_t*)scales + (size_t)num_groups * n * sizeof(float));
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      const size_t num_groups = k / gs;
      return k * n * sizeof(int8_t) + num_groups * n * sizeof(float) + num_groups * n * sizeof(int16_t);
    }

    // Unpack packed RAWINT4 weights [n_len, k/2] (row-major, low nibble = even k) into
    // signed int8 [n_len, k], convert bf16 scales [n_len, num_groups] to float32,
    // and compute int16 weight_sums [n_len, num_groups].
    void from_raw_mat(const uint8_t* src_packed, const ggml_bf16_t* src_scales, int ith, int nth) {
      auto [n_start, n_end] = avx2::split_range(n, ith, nth);
      const size_t row_bytes = (size_t)k / 2;

      for (int ni = n_start; ni < n_end; ++ni) {
        const uint8_t* src_row = src_packed + (size_t)ni * row_bytes;
        int8_t* dst_col = qweight_s8 + (size_t)ni * k;
        const ggml_bf16_t* src_sc_row = src_scales + (size_t)ni * num_groups;
        float* dst_sc_row = scales + (size_t)ni * num_groups;
        int16_t* dst_ws_row = weight_sums + (size_t)ni * num_groups;

        for (int g = 0; g < num_groups; ++g) {
          const int k_base = g * group_size;
          int sum = 0;
          for (int kk = 0; kk < group_size; kk += 2) {
            const uint8_t packed = src_row[(k_base + kk) / 2];
            const int8_t v0 = (int8_t)((packed & 0x0F) - 8);         // even k
            const int8_t v1 = (int8_t)(((packed >> 4) & 0x0F) - 8);  // odd k
            dst_col[k_base + kk] = v0;
            dst_col[k_base + kk + 1] = v1;
            sum += v0 + v1;
          }
          dst_ws_row[g] = (int16_t)sum;
          dst_sc_row[g] = GGML_BF16_TO_FP32(src_sc_row[g]);
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

KT_AVXVNNI256_RAWINT4_TARGET
static inline void gemm_rawint4_avxvnni256(int m, int n, int k, GemmKernelAVXVNNI256RawInt4::BufferA& a,
                                           GemmKernelAVXVNNI256RawInt4::BufferB& b,
                                           GemmKernelAVXVNNI256RawInt4::BufferC& c, int ith, int nth) {
  (void)k;
  auto [n_start, n_end] = avx2::split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  alignas(32) std::array<uint8_t, MAX_SUPPORTED_GROUP_SIZE> a_u8{};

  for (int mi = 0; mi < m; ++mi) {
    const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
    float* c_row = c.data + (size_t)mi * n;
    std::fill(c_row + n_start, c_row + n_end, 0.0f);

    for (int g = 0; g < num_groups; ++g) {
      const int k_base = g * group_size;
      const float a_scale = quantize_activation_group_u8(a_row + k_base, group_size, a_u8.data());
      if (a_scale == 0.0f) {
        continue;
      }

      for (int ni = n_start; ni < n_end; ++ni) {
        __m256i acc = _mm256_setzero_si256();
        const int8_t* w_col = b.qweight_s8 + (size_t)ni * b.k + k_base;
        for (int kk = 0; kk < group_size; kk += 32) {
          const __m256i a_vec = _mm256_load_si256((const __m256i*)(a_u8.data() + kk));
          const __m256i w_vec = _mm256_loadu_si256((const __m256i*)(w_col + kk));
          acc = _mm256_dpbusd_avx_epi32(acc, a_vec, w_vec);
        }

        const int dot = hsum_epi32_avx2(acc) - 128 * (int)b.weight_sums[(size_t)ni * num_groups + g];
        c_row[ni] += (float)dot * a_scale * b.scales[(size_t)ni * num_groups + g];
      }
    }
  }
}

}  // namespace avxvnni_rawint4

template <class T = avxvnni_rawint4::GemmKernelAVXVNNI256RawInt4>
class AVXVNNI256_RAW_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVXVNNI256_RAW_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVXVNNI256_RAW_INT4_MOE_TP<T>>;
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

  AVXVNNI256_RAW_INT4_MOE_TP() = default;
  AVXVNNI256_RAW_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avxvnni")) {
      throw std::runtime_error("AVX-VNNI-256 RAWINT4 backend requires CPU support for avx_vnni");
    }
#endif
    auto& qc = config_.quant_config;
    if (qc.group_size == 0 || (qc.group_size % 32) != 0) {
      throw std::runtime_error("AVX-VNNI-256 RAWINT4 requires group_size to be a positive multiple of 32");
    }
    if (qc.group_size > avxvnni_rawint4::MAX_SUPPORTED_GROUP_SIZE) {
      throw std::runtime_error("AVX-VNNI-256 RAWINT4 requires group_size <= 256");
    }
    if (qc.zero_point) {
      throw std::runtime_error("AVX-VNNI-256 RAWINT4 only supports signed INT4 without zero point");
    }
    printf("Created AVXVNNI256_RAW_INT4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }

  ~AVXVNNI256_RAW_INT4_MOE_TP() = default;

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
    (void)qlen;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avxvnni_rawint4::gemm_rawint4_avxvnni256(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith,
                                             nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    (void)qlen;
    int m = m_local_num_[expert_idx];
    avxvnni_rawint4::gemm_rawint4_avxvnni256(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx],
                                             *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_proj == nullptr || config_.gate_scale == nullptr) {
      throw std::runtime_error("AVX-VNNI RAWINT4 MoE requires flat packed weight and bf16 scale pointers");
    }

    // Gate/Up: source per expert is packed int4 [N_tp, K/2] uint8 + bf16 scales [N_tp, K/gs].
    {
      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id / nth;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;

            const size_t weight_bytes_per_expert = ((size_t)config_.intermediate_size * config_.hidden_size) / 2;
            const size_t scale_elems_per_expert =
                (size_t)config_.intermediate_size * (config_.hidden_size / group_size);

            const uint8_t* gate_src = (const uint8_t*)config_.gate_proj + logical * weight_bytes_per_expert;
            const uint8_t* up_src = (const uint8_t*)config_.up_proj + logical * weight_bytes_per_expert;
            const ggml_bf16_t* gate_sc_src = (const ggml_bf16_t*)config_.gate_scale + logical * scale_elems_per_expert;
            const ggml_bf16_t* up_sc_src = (const ggml_bf16_t*)config_.up_scale + logical * scale_elems_per_expert;

            gate_bb_[expert_idx]->from_raw_mat(gate_src, gate_sc_src, ith, nth);
            up_bb_[expert_idx]->from_raw_mat(up_src, up_sc_src, ith, nth);
          },
          nullptr);
    }

    // Down: source per expert is packed int4 [hidden_size, intermediate_size_tp/2] uint8
    //       + bf16 scales [hidden_size, intermediate_size_tp/gs].
    {
      int nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id / nth;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t logical = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;

            const size_t weight_bytes_per_expert = ((size_t)config_.hidden_size * config_.intermediate_size) / 2;
            const size_t scale_elems_per_expert =
                (size_t)config_.hidden_size * (config_.intermediate_size / group_size);

            const uint8_t* down_src = (const uint8_t*)config_.down_proj + logical * weight_bytes_per_expert;
            const ggml_bf16_t* down_sc_src = (const ggml_bf16_t*)config_.down_scale + logical * scale_elems_per_expert;

            down_bb_[expert_idx]->from_raw_mat(down_src, down_sc_src, ith, nth);
          },
          nullptr);
    }
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
    throw std::runtime_error("AVX-VNNI-256 RAWINT4 write_weights_to_buffer not yet implemented");
  }
};

template <typename K>
class TP_MOE<AVXVNNI256_RAW_INT4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_RAW_INT4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVXVNNI256_RAW_INT4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (config.gate_proj == nullptr || config.gate_scale == nullptr) {
      throw std::runtime_error("AVX-VNNI RAWINT4 MoE only supports flat packed INT4 with KGroup bf16 scales");
    }

    const int group_size = config.quant_config.group_size;
    if (group_size == 0) {
      throw std::runtime_error("AVX-VNNI RAWINT4 requires group_size > 0");
    }

    // Build TP-sliced flat buffers per NUMA partition (source layout mirrors rawint4-moe.hpp).
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = (size_t)tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = ((size_t)tpc.hidden_size / group_size) * tpc.intermediate_size;
      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.up_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];
      tpc.down_scale = new ggml_bf16_t[tpc.expert_num * scales_elem_count];

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, i](int expert_id_) {
            size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
            uint8_t* src_gate =
                (uint8_t*)config.gate_proj + ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
            uint8_t* src_up =
                (uint8_t*)config.up_proj + ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
            uint8_t* src_down =
                (uint8_t*)config.down_proj + ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
            ggml_bf16_t* src_gate_scale =
                (ggml_bf16_t*)config.gate_scale +
                expert_id * ((size_t)config.hidden_size / group_size) * config.intermediate_size;
            ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scale + expert_id *
                                                                            ((size_t)config.hidden_size / group_size) *
                                                                            config.intermediate_size;
            ggml_bf16_t* src_down_scale =
                (ggml_bf16_t*)config.down_scale +
                expert_id * ((size_t)config.intermediate_size / group_size) * config.hidden_size;

            // Gate/Up: contiguous slice along N (intermediate).
            std::memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                        src_gate + ((i * weight_elem_count) >> 1), weight_elem_count >> 1);
            std::memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                        src_up + ((i * weight_elem_count) >> 1), weight_elem_count >> 1);
            std::memcpy((ggml_bf16_t*)tpc.gate_scale + expert_id * scales_elem_count,
                        src_gate_scale + i * scales_elem_count, sizeof(ggml_bf16_t) * scales_elem_count);
            std::memcpy((ggml_bf16_t*)tpc.up_scale + expert_id * scales_elem_count,
                        src_up_scale + i * scales_elem_count, sizeof(ggml_bf16_t) * scales_elem_count);

            // Down: column-gather across N (hidden) rows for the TP-sliced K (intermediate).
            for (size_t col = 0; col < (size_t)config.hidden_size; col++) {
              std::memcpy(
                  (uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                  src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                  tpc.intermediate_size >> 1);
              std::memcpy((ggml_bf16_t*)tpc.down_scale +
                              (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                          src_down_scale + (col * (config.intermediate_size / group_size) +
                                            i * (tpc.intermediate_size / group_size)),
                          sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
            }
          },
          nullptr);
      printf("AVX-VNNI-256 RAWINT4 TP %d load weight done.\n", i);
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
    throw std::runtime_error("AVX-VNNI-256 RAWINT4 write_weight_scale_to_buffer not yet implemented");
  }
};

#undef KT_AVXVNNI256_RAWINT4_TARGET

#endif  // CPUINFER_OPERATOR_AVX2_RAW_INT4_AVXVNNI_MOE_H
