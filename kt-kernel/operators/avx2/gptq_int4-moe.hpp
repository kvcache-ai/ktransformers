/**
 * @Description  : AVX2 GPTQ-Int4 MoE operator (symmetric quantization)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Supports GPTQ symmetric (sym=true, desc_act=false) INT4 quantization.
 * qweight [K/8, N] int32 + scales [K/gs, N] fp32. No qzeros needed.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_GPTQ_INT4_MOE_H
#define CPUINFER_OPERATOR_AVX2_GPTQ_INT4_MOE_H

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "gptq_int4_dequant.hpp"
#include "moe_base.hpp"

namespace avx2 {

struct GemmKernelAVX2GPTQInt4 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 8;     // 8 INT4 values per int32
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;  // = group_size typically
  static constexpr double ELEMENT_SIZE = 0.5;  // INT4 = 0.5 byte

  static void config() {}

  static int recommended_nth(int n) {
    return std::max(1, n / N_BLOCK);
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return split_range(n, ith, nth);
  }

  // ========================================================================
  // BufferA: BF16 activations [M, K] — same as BF16/FP8
  // ========================================================================
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
        auto [m_start, m_end] = split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  // ========================================================================
  // BufferB: GPTQ INT4 weights [K/8, N] int32 + scales [num_groups, N] fp32
  // ========================================================================
  struct BufferB {
    uint32_t* qweight = nullptr;  // [K/8, N] packed int32
    float* scales = nullptr;      // [num_groups, N] fp32
    int n = 0;
    int k = 0;
    int group_size = 128;
    int num_groups = 0;
    int k_packed = 0;  // = K/8

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int gs, void* ptr)
        : n(n_), k(k_), group_size(gs) {
      k_packed = k / 8;
      num_groups = k / gs;
      qweight = (uint32_t*)ptr;
      scales = (float*)((uint8_t*)ptr + (size_t)k_packed * n * sizeof(uint32_t));
    }

    static size_t required_size(size_t n, size_t k, int gs) {
      size_t k_packed = k / 8;
      size_t num_groups = k / gs;
      return k_packed * n * sizeof(uint32_t) + num_groups * n * sizeof(float);
    }

    // Load qweight and scales from separate source pointers
    void from_mat(const uint32_t* src_qweight, const float* src_scales, int ith, int nth) {
      // Split by N dimension
      auto [n_start, n_end] = split_range(n, ith, nth);
      int n_len = n_end - n_start;

      // Copy qweight rows [K/8 rows, each row = N int32]
      for (int kr = 0; kr < k_packed; kr++) {
        std::memcpy(qweight + kr * n + n_start,
                    src_qweight + kr * n + n_start,
                    n_len * sizeof(uint32_t));
      }

      // Copy scales rows [num_groups rows, each row = N float]
      for (int g = 0; g < num_groups; g++) {
        std::memcpy(scales + g * n + n_start,
                    src_scales + g * n + n_start,
                    n_len * sizeof(float));
      }
    }
  };

  // ========================================================================
  // BufferC: FP32 output — same as BF16/FP8
  // ========================================================================
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
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + mi * n;
        ggml_bf16_t* dst_row = dst + mi * n;
        int j = n_start;
        for (; j + 8 <= n_end; j += 8) {
          store_fp32_to_bf16(dst_row + j, _mm256_loadu_ps(src_row + j));
        }
        for (; j < n_end; j++) {
          dst_row[j] = GGML_FP32_TO_BF16(src_row[j]);
        }
      }
    }
  };
};

// ============================================================================
// AVX2 GPTQ INT4 GEMM (symmetric)
// C[m,n] = sum_k A_bf16[m,k] * dequant(B_int4[k,n])
// ============================================================================

static inline void gemm_gptq_sym_int4(
    int m, int n, int k,
    GemmKernelAVX2GPTQInt4::BufferA& a,
    GemmKernelAVX2GPTQInt4::BufferB& b,
    GemmKernelAVX2GPTQInt4::BufferC& c,
    int ith, int nth) {

  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_size = b.group_size;
  const int num_groups = b.num_groups;

  for (int ni = n_start; ni < n_end; ni++) {
    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      float sum = 0.0f;

      for (int g = 0; g < num_groups; g++) {
        float scale = b.scales[g * n + ni];
        int k_base = g * group_size;

        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();

        // group_size/8 iterations (e.g., 128/8 = 16)
        for (int ki = 0; ki < group_size; ki += 8) {
          int k_abs = k_base + ki;
          __m256 a_val = load_bf16_to_fp32(a_row + k_abs);
          uint32_t packed = b.qweight[(k_abs / 8) * n + ni];
          __m256 w_val = gptq_sym_dequant_8x4bit(packed, scale);
          acc1 = _mm256_fmadd_ps(a_val, w_val, acc1);
        }

        sum += hsum_avx2(acc1);
      }

      c.data[mi * n + ni] = sum;
    }
  }
}

}  // namespace avx2

// ============================================================================
// AVX2 GPTQ INT4 MoE operator
// ============================================================================

template <class T = avx2::GemmKernelAVX2GPTQInt4>
class AVX2_GPTQ_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVX2_GPTQ_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_GPTQ_INT4_MOE_TP<T>>;
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

  AVX2_GPTQ_INT4_MOE_TP() = default;
  AVX2_GPTQ_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& qc = config_.quant_config;
    if (qc.group_size == 0) {
      throw std::runtime_error("GPTQ INT4 requires group_size > 0");
    }
    printf("Created AVX2_GPTQ_INT4_MOE_TP %d at numa %d (group_size=%d)\n",
           tp_part_idx, numa_node_of_cpu(sched_getcpu()), qc.group_size);
  }

  ~AVX2_GPTQ_INT4_MOE_TP() = default;

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
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_gptq_sym_int4(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_gptq_sym_int4(m, config_.hidden_size, config_.intermediate_size,
                              *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // Load weights from contiguous qweight + scales pointers
  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("GPTQ INT4 MOE requires scale pointers.");
    }

    // gate + up: qweight [K/8, N=intermediate], scales [K/gs, N=intermediate]
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

          gate_bb_[expert_idx]->from_mat(
              (uint32_t*)config_.gate_proj + logical * qw_elems,
              (float*)config_.gate_scale + logical * sc_elems,
              ith, nth);

          up_bb_[expert_idx]->from_mat(
              (uint32_t*)config_.up_proj + logical * qw_elems,
              (float*)config_.up_scale + logical * sc_elems,
              ith, nth);
        },
        nullptr);

    // down: qweight [K/8, N=hidden] where K=intermediate
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

          down_bb_[expert_idx]->from_mat(
              (uint32_t*)config_.down_proj + logical * down_qw_elems,
              (float*)config_.down_scale + logical * down_sc_elems,
              ith, nth);
        },
        nullptr);
  }

  // write_weights_to_buffer for layerwise prefill / GPU expert offload
  // Note: GPTQ INT4 GPU offload requires the GPU to support INT4 dequant.
  // For now, this is a placeholder that copies raw packed data.
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    // TODO: Implement GPTQ INT4 GPU offload when needed
    // For now, layerwise prefill with GPTQ INT4 is not supported
    throw std::runtime_error("GPTQ INT4 write_weights_to_buffer not yet implemented");
  }
};

// ============================================================================
// TP_MOE specialization for AVX2_GPTQ_INT4_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AVX2_GPTQ_INT4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_GPTQ_INT4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_GPTQ_INT4_MOE_TP<K>>>;
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

    // Full dimensions
    const int full_intermediate = config.intermediate_size;
    const int full_hidden = config.hidden_size;

    // gate/up: shape [K=hidden, N=intermediate]
    // qweight: [hidden/8, intermediate], scales: [hidden/gs, intermediate]
    const int gate_up_k_packed = full_hidden / 8;
    const int gate_up_num_groups = full_hidden / group_size;
    const size_t full_gate_up_qw_elems = (size_t)gate_up_k_packed * full_intermediate;
    const size_t full_gate_up_sc_elems = (size_t)gate_up_num_groups * full_intermediate;

    // down: shape [K=intermediate, N=hidden]
    // qweight: [intermediate/8, hidden], scales: [intermediate/gs, hidden]
    const int down_k_packed = full_intermediate / 8;
    const int down_num_groups = full_intermediate / group_size;
    const size_t full_down_qw_elems = (size_t)down_k_packed * full_hidden;
    const size_t full_down_sc_elems = (size_t)down_num_groups * full_hidden;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const int tp_intermediate = tpc.intermediate_size;

      // gate/up TP: N=intermediate is split
      const size_t tp_gate_up_qw_elems = (size_t)gate_up_k_packed * tp_intermediate;
      const size_t tp_gate_up_sc_elems = (size_t)gate_up_num_groups * tp_intermediate;

      tpc.gate_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.up_proj = new uint32_t[tpc.expert_num * tp_gate_up_qw_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_gate_up_sc_elems];

      // down TP: K=intermediate is split
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
          [&, &tpc](int expert_id_) {
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

            // gate/up qweight: [K/8, N] → slice N columns
            for (int kr = 0; kr < gate_up_k_packed; kr++) {
              std::memcpy(gate_qw_dst + kr * tp_intermediate,
                          gate_qw_src + kr * full_intermediate + gate_up_n_offset,
                          tp_intermediate * sizeof(uint32_t));
              std::memcpy(up_qw_dst + kr * tp_intermediate,
                          up_qw_src + kr * full_intermediate + gate_up_n_offset,
                          tp_intermediate * sizeof(uint32_t));
            }

            // gate/up scales: [num_groups, N] → slice N columns
            for (int g = 0; g < gate_up_num_groups; g++) {
              std::memcpy(gate_sc_dst + g * tp_intermediate,
                          gate_sc_src + g * full_intermediate + gate_up_n_offset,
                          tp_intermediate * sizeof(float));
              std::memcpy(up_sc_dst + g * tp_intermediate,
                          up_sc_src + g * full_intermediate + gate_up_n_offset,
                          tp_intermediate * sizeof(float));
            }

            // down qweight: [K/8, N=hidden] row-major → slice contiguous rows (K/8 dim)
            uint32_t* down_qw_dst = (uint32_t*)tpc.down_proj + expert_id * tp_down_qw_elems;
            for (int kr = 0; kr < tp_down_k_packed; kr++) {
              std::memcpy(down_qw_dst + kr * full_hidden,
                          down_qw_src + (down_k_offset_packed + kr) * full_hidden,
                          full_hidden * sizeof(uint32_t));
            }

            // down scales: [K/gs, N=hidden] row-major → slice contiguous rows (K/gs dim)
            float* down_sc_dst = (float*)tpc.down_scale + expert_id * tp_down_sc_elems;
            for (int g = 0; g < tp_down_num_groups; g++) {
              std::memcpy(down_sc_dst + g * full_hidden,
                          down_sc_src + (down_group_offset + g) * full_hidden,
                          full_hidden * sizeof(float));
            }
          },
          nullptr);
    });

    // Call per-TP load_weights
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      tps[i]->load_weights();
    });

    // Free temporary buffers
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
    // GPTQ INT4 GPU offload not yet supported
    throw std::runtime_error("GPTQ INT4 write_weight_scale_to_buffer not yet implemented");
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_GPTQ_INT4_MOE_H
