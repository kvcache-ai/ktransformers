/**
 * @Description  : AVX2 FP8 MoE operator (ported from amx/fp8-moe.hpp)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * FP8 E4M3 weights with 128×128 block-wise float32 scales.
 * Dequantization: FP8→FP32 via precomputed 256-entry LUT + AVX2 gather.
 * GEMM: BF16 input × FP32 dequantized weight → FP32 output.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_FP8_MOE_H
#define CPUINFER_OPERATOR_AVX2_FP8_MOE_H

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "fp8_dequant.hpp"
#include "moe_base.hpp"

namespace avx2 {

inline int div_up(int a, int b) { return (a + b - 1) / b; }

struct GemmKernelAVX2FP8 {
  using dt = ggml_bf16_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 8;
  static constexpr int BLOCK_SIZE = 128;  // 128×128 block quantization
  static constexpr int N_BLOCK = 128;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 1.0;  // FP8 = 1 byte

  static void config() {}

  static int recommended_nth(int n) {
    return std::max(1, div_up(n, N_BLOCK));
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    return avx2::split_range(n, ith, nth);
  }

  // ========================================================================
  // BufferA: BF16 activations [M, K] — same as BF16 backend
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
        auto [m_start, m_end] = avx2::split_range(m, ith, nth);
        std::memcpy(data + m_start * k, src + m_start * k,
                    (size_t)(m_end - m_start) * k * sizeof(ggml_bf16_t));
      }
    }
  };

  // ========================================================================
  // BufferB: FP8 weights [N, K] + float32 scales [N/BS, K/BS]
  // Row-major, no packing. from_mat = memcpy.
  // ========================================================================
  struct BufferB {
    uint8_t* b = nullptr;  // FP8 weights
    float* d = nullptr;    // Block-wise scales
    size_t n = 0;
    size_t k = 0;
    int block_size = BLOCK_SIZE;

    BufferB() = default;
    BufferB(size_t n_, size_t k_, int bs, void* ptr) : n(n_), k(k_), block_size(bs) {
      b = (uint8_t*)ptr;
      size_t weight_bytes = n * k;
      d = (float*)((uint8_t*)ptr + weight_bytes);
    }

    static size_t required_size(size_t n, size_t k, int bs) {
      size_t n_blocks_n = div_up((int)n, bs);
      size_t n_blocks_k = div_up((int)k, bs);
      return n * k + n_blocks_n * n_blocks_k * sizeof(float);
    }

    void from_mat(const uint8_t* src_weights, const float* src_scales, int ith, int nth) {
      // Copy weights (split by N)
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
      std::memcpy(b + n_start * k, src_weights + n_start * k,
                  (size_t)(n_end - n_start) * k);

      // Copy scales (split by N blocks)
      int n_blocks_k = div_up((int)k, block_size);
      int nb_start = n_start / block_size;
      int nb_end = div_up(n_end, block_size);
      std::memcpy(d + nb_start * n_blocks_k, src_scales + nb_start * n_blocks_k,
                  (size_t)(nb_end - nb_start) * n_blocks_k * sizeof(float));
    }
  };

  // ========================================================================
  // BufferC: FP32 output — same as BF16 backend
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
      auto [n_start, n_end] = avx2::split_range((int)n, ith, nth);
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
// AVX2 FP8 GEMM: C[m,n] = sum_k (A[m,k] * dequant(B[n,k])) * scale[n/BS, k/BS]
// ============================================================================

static inline void gemm_fp8(
    int m, int n, int k,
    GemmKernelAVX2FP8::BufferA& a,
    GemmKernelAVX2FP8::BufferB& b,
    GemmKernelAVX2FP8::BufferC& c,
    int ith, int nth) {

  ensure_fp8_lut_initialized();

  auto [n_start, n_end] = split_range(n, ith, nth);
  const int block_size = b.block_size;
  const int n_blocks_k = div_up(k, block_size);

  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* b_row = b.b + (size_t)ni * k;
    const int n_block_idx = ni / block_size;

    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;
      float sum = 0.0f;

      for (int kb = 0; kb < k; kb += block_size) {
        int k_len = std::min(block_size, k - kb);
        int k_block_idx = kb / block_size;
        float scale = b.d[n_block_idx * n_blocks_k + k_block_idx];

        // Accumulate within this block
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= k_len; ki += 32) {
          acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + kb + ki),
                                  fp8x8_to_fp32x8(b_row + kb + ki), acc1);
          acc2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + kb + ki + 8),
                                  fp8x8_to_fp32x8(b_row + kb + ki + 8), acc2);
          acc3 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + kb + ki + 16),
                                  fp8x8_to_fp32x8(b_row + kb + ki + 16), acc3);
          acc4 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + kb + ki + 24),
                                  fp8x8_to_fp32x8(b_row + kb + ki + 24), acc4);
        }
        for (; ki + 8 <= k_len; ki += 8) {
          acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + kb + ki),
                                  fp8x8_to_fp32x8(b_row + kb + ki), acc1);
        }

        float block_sum = hsum_avx2(_mm256_add_ps(_mm256_add_ps(acc1, acc3),
                                                   _mm256_add_ps(acc2, acc4)));

        // Scalar tail
        for (; ki < k_len; ki++) {
          block_sum += GGML_BF16_TO_FP32(a_row[kb + ki]) * fp8_to_fp32_scalar(b_row[kb + ki]);
        }

        sum += block_sum * scale;
      }

      c.data[mi * n + ni] = sum;
    }
  }
}

}  // namespace avx2

// ============================================================================
// AVX2 FP8 MoE operator (CRTP derived from AVX2_MOE_BASE)
// ============================================================================

template <class T = avx2::GemmKernelAVX2FP8>
class AVX2_FP8_MOE_TP : public AVX2_MOE_BASE<T, AVX2_FP8_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_FP8_MOE_TP<T>>;
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

  AVX2_FP8_MOE_TP() = default;

  AVX2_FP8_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    avx2::ensure_fp8_lut_initialized();
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("AVX2 FP8 MoE only supports block-wise FP8 (group_size > 0, no zero_point)");
    }
    printf("Created AVX2_FP8_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AVX2_FP8_MOE_TP() = default;

  // CRTP buffer creation — with group_size for BufferB
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
    avx2::gemm_fp8(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_fp8(m, config_.hidden_size, config_.intermediate_size,
                   *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  // Load FP8 weights + scales from contiguous memory
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("FP8 MOE requires scale pointers.");
    }

    // Load gate + up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id *
              avx2::div_up(config_.hidden_size, group_size) *
              avx2::div_up(config_.intermediate_size, group_size);

          gate_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.gate_proj + weight_offset,
              (float*)config_.gate_scale + scale_offset,
              ith, nth);

          up_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.up_proj + weight_offset,
              (float*)config_.up_scale + scale_offset,
              ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map, group_size](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          size_t weight_offset = logical_expert_id * config_.intermediate_size * config_.hidden_size;
          size_t scale_offset = logical_expert_id *
              avx2::div_up(config_.hidden_size, group_size) *
              avx2::div_up(config_.intermediate_size, group_size);

          down_bb_[expert_idx]->from_mat(
              (uint8_t*)config_.down_proj + weight_offset,
              (float*)config_.down_scale + scale_offset,
              ith, nth);
        },
        nullptr);
  }

  // Write weights to GPU buffer (for dynamic expert offload / layerwise prefill)
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);
    int group_size = config.quant_config.group_size;

    // W13 (gate+up)
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;
    const int gpu_n_blocks_k_w13 = avx2::div_up(gpu_k_w13, group_size);
    const size_t gpu_w13_scale_per_mat = (size_t)avx2::div_up(gpu_n_w13, group_size) * gpu_n_blocks_k_w13;

    // W2 (down)
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;
    const int cpu_n_blocks_k_w2 = avx2::div_up(cpu_k_w2, group_size);

    constexpr int NUM_W13_TASKS = 32;
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int rows_per_task = avx2::div_up(cpu_n_w13, NUM_W13_TASKS);
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w13);
            if (row_start >= cpu_n_w13) return;

            for (int row = row_start; row < row_end; row++) {
              const int global_n = global_n_offset_w13 + row;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              // Copy weight row
              uint8_t* w_dst = (uint8_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_w_off = is_up ? gpu_w13_weight_per_mat : 0;
              std::memcpy(w_dst + expert_w_off + (size_t)n_in_gpu * gpu_k_w13,
                          bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13);

              // Copy scale row (if at block boundary)
              if (row % group_size == 0) {
                int n_block = row / group_size;
                int gpu_n_block = n_in_gpu / group_size;
                float* s_dst = (float*)w13_scale_ptrs[target_gpu];
                const size_t expert_s_off = is_up ? gpu_w13_scale_per_mat : 0;
                std::memcpy(s_dst + expert_s_off + gpu_n_block * gpu_n_blocks_k_w13,
                            bb->d + n_block * avx2::div_up(cpu_k_w13, group_size),
                            avx2::div_up(cpu_k_w13, group_size) * sizeof(float));
              }
            }
          } else {
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int rows_per_task = avx2::div_up(cpu_n_w2, NUM_W2_TASKS);
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w2);
            if (row_start >= cpu_n_w2) return;

            for (int row = row_start; row < row_end; row++) {
              // Iterate over all gpu_k_w2-sized slices within this CPU TP's K range
              for (int k_start = 0; k_start < cpu_k_w2; k_start += gpu_k_w2) {
                const int k_slice_len = std::min(gpu_k_w2, cpu_k_w2 - k_start);
                const int global_k = global_k_offset_w2 + k_start;
                const int target_gpu = global_k / gpu_k_w2;
                const int k_in_gpu = global_k % gpu_k_w2;

                uint8_t* w_dst = (uint8_t*)w2_weight_ptrs[target_gpu];
                std::memcpy(w_dst + (size_t)row * gpu_k_w2 + k_in_gpu,
                            bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len);

                // Copy scales for down (at block boundaries)
                if (row % group_size == 0) {
                  int n_block = row / group_size;
                  float* s_dst = (float*)w2_scale_ptrs[target_gpu];
                  int gpu_n_blocks_k_w2 = avx2::div_up(gpu_k_w2, group_size);
                  int k_block_start = k_in_gpu / group_size;
                  int n_blocks_to_copy = std::min(cpu_n_blocks_k_w2, gpu_n_blocks_k_w2 - k_block_start);
                  std::memcpy(s_dst + n_block * gpu_n_blocks_k_w2 + k_block_start,
                              bb->d + n_block * cpu_n_blocks_k_w2 + k_start / group_size,
                              n_blocks_to_copy * sizeof(float));
                }
              }  // end k_start loop
            }  // end row loop
          }
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization — ported from amx/fp8-moe.hpp:628-738
// Handles per-expert pointer loading + TP weight/scale splitting
// ============================================================================
template <typename K>
class TP_MOE<AVX2_FP8_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_FP8_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_FP8_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    const int group_size = config.quant_config.group_size;
    if (group_size == 0 || config.quant_config.zero_point) {
      throw std::runtime_error("FP8 MoE only supports block-wise (group_size > 0, zero_point=false)");
    }

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }
    const bool use_per_expert_ptrs = !config.gate_projs.empty();

    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;
    const size_t full_scale_elems =
        (size_t)avx2::div_up(config.hidden_size, group_size) * avx2::div_up(config.intermediate_size, group_size);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;
      const size_t tp_scale_elems =
          (size_t)avx2::div_up(tpc.intermediate_size, group_size) * avx2::div_up(tpc.hidden_size, group_size);

      // Allocate temporary buffers
      tpc.gate_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new uint8_t[tpc.expert_num * tp_weight_elems];
      tpc.gate_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.up_scale = new float[tpc.expert_num * tp_scale_elems];
      tpc.down_scale = new float[tpc.expert_num * tp_scale_elems];

      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t gate_up_scale_src_offset = i * tp_scale_elems;
      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;
      const size_t down_scale_src_block_k_offset = down_weight_src_col_offset / (size_t)group_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            uint8_t* gate_dst = (uint8_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            uint8_t* up_dst = (uint8_t*)tpc.up_proj + expert_id * tp_weight_elems;
            uint8_t* down_dst = (uint8_t*)tpc.down_proj + expert_id * tp_weight_elems;
            float* gate_scale_dst = (float*)tpc.gate_scale + expert_id * tp_scale_elems;
            float* up_scale_dst = (float*)tpc.up_scale + expert_id * tp_scale_elems;
            float* down_scale_dst = (float*)tpc.down_scale + expert_id * tp_scale_elems;

            const uint8_t* gate_src;
            const uint8_t* up_src;
            const uint8_t* down_src;
            const float* gate_scale_src;
            const float* up_scale_src;
            const float* down_scale_src;

            if (use_per_expert_ptrs) {
              gate_src = (const uint8_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_projs[0][expert_id];
              gate_scale_src = (const float*)config.gate_scales[0][expert_id] + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scales[0][expert_id] + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scales[0][expert_id];
            } else {
              gate_src = (const uint8_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src = (const uint8_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const uint8_t*)config.down_proj + expert_id * full_weight_elems;
              gate_scale_src = (const float*)config.gate_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              up_scale_src = (const float*)config.up_scale + expert_id * full_scale_elems + gate_up_scale_src_offset;
              down_scale_src = (const float*)config.down_scale + expert_id * full_scale_elems;
            }

            // Copy gate/up weights + scales (column slice)
            std::memcpy(gate_dst, gate_src, tp_weight_elems);
            std::memcpy(up_dst, up_src, tp_weight_elems);
            std::memcpy(gate_scale_dst, gate_scale_src, sizeof(float) * tp_scale_elems);
            std::memcpy(up_scale_dst, up_scale_src, sizeof(float) * tp_scale_elems);

            // Copy down weights (row-wise split)
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset, (size_t)tpc.intermediate_size);
            }

            // Copy down scales (block-row-wise split)
            const int n_blocks_n = avx2::div_up(config.hidden_size, group_size);
            const int full_n_blocks_k = avx2::div_up(config.intermediate_size, group_size);
            const int tp_n_blocks_k = avx2::div_up(tpc.intermediate_size, group_size);
            for (int bn = 0; bn < n_blocks_n; bn++) {
              const float* src = down_scale_src + (size_t)bn * full_n_blocks_k + down_scale_src_block_k_offset;
              float* dst = down_scale_dst + (size_t)bn * tp_n_blocks_k;
              std::memcpy(dst, src, sizeof(float) * tp_n_blocks_k);
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
      delete[] (uint8_t*)tpc.gate_proj;
      delete[] (uint8_t*)tpc.up_proj;
      delete[] (uint8_t*)tpc.down_proj;
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
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config,
                                            w13_weight_ptrs, w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_FP8_MOE_H
