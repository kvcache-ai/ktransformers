/**
 * @Description  : AVX2 RAWINT4 MoE operator for Kimi native INT4 weights
 * SPDX-License-Identifier: Apache-2.0
 *
 * RAWINT4 stores signed int4 weights packed two values per byte, plus BF16
 * K-group scales. This AVX2 backend keeps the native layout so GPU layerwise
 * prefill can reuse the same weight buffers, and uses BF16 activations with
 * FP32 accumulation for CPU decode.
 **/
#ifndef CPUINFER_OPERATOR_AVX2_RAW_INT4_MOE_H
#define CPUINFER_OPERATOR_AVX2_RAW_INT4_MOE_H

#include "avx2_bf16_gemm.hpp"
#include "avx2_bf16_utils.hpp"
#include "gptq_int4_dequant.hpp"
#include "moe_base.hpp"

namespace avx2 {

// Local, RAWINT4-only variant of gptq_sym_dequant_8x4bit that skips the
// per-group scale multiply. Callers fold the scale in via a single FMA after
// the K-loop so the inner path avoids a broadcast+mul per packed int32.
static inline __m256 rawint4_dequant_8x4bit_unscaled(uint32_t packed_weight) {
  const __m256i shifts = _mm256_set_epi32(28, 24, 20, 16, 12, 8, 4, 0);
  __m256i packed_v = _mm256_set1_epi32(packed_weight);
  __m256i nibbles = _mm256_and_si256(_mm256_srlv_epi32(packed_v, shifts), _mm256_set1_epi32(0xF));
  __m256 w = _mm256_cvtepi32_ps(nibbles);
  return _mm256_sub_ps(w, _mm256_set1_ps(8.0f));
}

struct GemmKernelAVX2RawInt4 {
  using dt = uint8_t;
  using output_t = float;
  static constexpr int M_STEP = 1;
  static constexpr int N_STEP = 8;
  static constexpr int K_STEP = 8;
  static constexpr int N_BLOCK = 64;
  static constexpr int K_BLOCK = 128;
  static constexpr double ELEMENT_SIZE = 0.5;

  static void config() {}

  static int recommended_nth(int n) { return std::max(1, div_up(n, N_BLOCK)); }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) { return split_range(n, ith, nth); }

  struct BufferA {
    ggml_bf16_t* data = nullptr;
    size_t max_m = 0;
    size_t k = 0;

    BufferA() = default;
    BufferA(size_t m, size_t k_, int, void* ptr) : data((ggml_bf16_t*)ptr), max_m(m), k(k_) {}

    static size_t required_size(size_t m, size_t k, int) { return m * k * sizeof(ggml_bf16_t); }

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
    uint8_t* b = nullptr;
    float* d = nullptr;
    int n = 0;
    int k = 0;
    int k_group_size = 0;
    int k_group_count = 0;

    BufferB() = default;

    // Full allocation: b and d packed into a single aligned block.
    BufferB(int n_, int k_, int k_group_size_, void* ptr)
        : b((uint8_t*)ptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0) {
        throw std::runtime_error("RAWINT4 requires k aligned to group_size and 8");
      }
      k_group_count = k / k_group_size;
      d = (float*)((uint8_t*)ptr + ((size_t)n * k / 2));
    }

    // Scale-only allocation: b points to external (mmap'd) weight data; d owns scale_ptr.
    // Used when weights are consumed directly from safetensor mmap without copying.
    BufferB(int n_, int k_, int k_group_size_, void* scale_ptr, std::nullptr_t /*scale_only*/)
        : b(nullptr), n(n_), k(k_), k_group_size(k_group_size_) {
      if (k_group_size <= 0 || k % k_group_size != 0 || k % 8 != 0) {
        throw std::runtime_error("RAWINT4 requires k aligned to group_size and 8");
      }
      k_group_count = k / k_group_size;
      d = (float*)scale_ptr;
    }

    // Full size: packed INT4 weights + float32 scales.
    static size_t required_size(size_t n, size_t k, int k_group_size) {
      return n * k / 2 + n * (k / k_group_size) * sizeof(float);
    }

    // Scale-only size: only float32 scales (b will point to external weight data).
    static size_t required_size_scale_only(size_t n, size_t k, int k_group_size) {
      return n * (k / k_group_size) * sizeof(float);
    }

    void from_raw_mat(const uint8_t* proj, int ith, int nth) {
      if (b == nullptr) return;  // scale-only mode: b is an external pointer set later
      auto [n_start, n_end] = split_range_n(n, ith, nth);
      const size_t row_bytes = (size_t)k / 2;
      std::memcpy(b + (size_t)n_start * row_bytes, proj + (size_t)n_start * row_bytes,
                  (size_t)(n_end - n_start) * row_bytes);
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
      auto [n_start, n_end] = split_range((int)n, ith, nth);
      for (int mi = 0; mi < m; mi++) {
        float* src_row = data + (size_t)mi * n;
        ggml_bf16_t* dst_row = dst + (size_t)mi * n;
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

static inline void gemm_rawint4(int m, int n, int k, GemmKernelAVX2RawInt4::BufferA& a,
                                GemmKernelAVX2RawInt4::BufferB& b, GemmKernelAVX2RawInt4::BufferC& c, int ith,
                                int nth) {
  auto [n_start, n_end] = split_range(n, ith, nth);
  const int group_size = b.k_group_size;
  const int group_count = b.k_group_count;
  const size_t row_bytes = (size_t)k / 2;

  for (int ni = n_start; ni < n_end; ni++) {
    const uint8_t* b_row = b.b + (size_t)ni * row_bytes;
    const float* b_scales = b.d + (size_t)ni * group_count;

    // Prefetch the head of the next B row while we chew through this one.
    // Streams the decoupled weight pages for large N without blocking.
    if (ni + 1 < n_end) {
      const uint8_t* b_next = b.b + (size_t)(ni + 1) * row_bytes;
      _mm_prefetch((const char*)b_next, _MM_HINT_T0);
      _mm_prefetch((const char*)(b_next + 64), _MM_HINT_T0);
    }

    for (int mi = 0; mi < m; mi++) {
      const ggml_bf16_t* a_row = a.data + (size_t)mi * a.k;

      // Running vector total: each group folds in via a single FMA with its
      // scale broadcast, so we only do one horizontal reduction per (mi, ni).
      __m256 total_acc = _mm256_setzero_ps();
      float scalar_tail = 0.0f;

      for (int g = 0; g < group_count; g++) {
        const float scale = b_scales[g];
        const int k_base = g * group_size;

        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();

        int ki = 0;
        for (; ki + 32 <= group_size; ki += 32) {
          uint32_t p0, p1, p2, p3;
          std::memcpy(&p0, b_row + (k_base + ki) / 2, sizeof(uint32_t));
          std::memcpy(&p1, b_row + (k_base + ki + 8) / 2, sizeof(uint32_t));
          std::memcpy(&p2, b_row + (k_base + ki + 16) / 2, sizeof(uint32_t));
          std::memcpy(&p3, b_row + (k_base + ki + 24) / 2, sizeof(uint32_t));
          acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki), rawint4_dequant_8x4bit_unscaled(p0), acc1);
          acc2 = _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 8), rawint4_dequant_8x4bit_unscaled(p1), acc2);
          acc3 =
              _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 16), rawint4_dequant_8x4bit_unscaled(p2), acc3);
          acc4 =
              _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki + 24), rawint4_dequant_8x4bit_unscaled(p3), acc4);
        }
        __m256 g_acc = _mm256_add_ps(_mm256_add_ps(acc1, acc3), _mm256_add_ps(acc2, acc4));

        for (; ki + 8 <= group_size; ki += 8) {
          uint32_t packed;
          std::memcpy(&packed, b_row + (k_base + ki) / 2, sizeof(uint32_t));
          g_acc =
              _mm256_fmadd_ps(load_bf16_to_fp32(a_row + k_base + ki), rawint4_dequant_8x4bit_unscaled(packed), g_acc);
        }

        // Fold this group's unscaled accumulator into the running total with
        // one scale-broadcast FMA — saves (group_count - 1) hsum reductions
        // compared to reducing per group.
        total_acc = _mm256_fmadd_ps(g_acc, _mm256_broadcast_ss(&scale), total_acc);

        for (; ki < group_size; ki++) {
          const uint8_t packed = b_row[(k_base + ki) / 2];
          const int nibble = ((k_base + ki) & 1) ? (packed >> 4) : (packed & 0x0F);
          scalar_tail += GGML_BF16_TO_FP32(a_row[k_base + ki]) * (float)(nibble - 8) * scale;
        }
      }

      c.data[(size_t)mi * n + ni] = hsum_avx2(total_acc) + scalar_tail;
    }
  }
}

}  // namespace avx2

template <class T = avx2::GemmKernelAVX2RawInt4>
class AVX2_RAW_INT4_MOE_TP : public AVX2_MOE_BASE<T, AVX2_RAW_INT4_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_RAW_INT4_MOE_TP<T>>;
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

  AVX2_RAW_INT4_MOE_TP() = default;
  AVX2_RAW_INT4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    if (config_.quant_config.group_size == 0 || config_.quant_config.zero_point) {
      throw std::runtime_error("RAWINT4 AVX2 MoE only supports KGroup signed INT4 without zero point");
    }
    printf("Created AVX2_RAW_INT4_MOE_TP %d at numa %d (group_size=%d)\n", tp_part_idx,
           numa_node_of_cpu(sched_getcpu()), config_.quant_config.group_size);
  }

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    // When per-expert source pointers are available, only allocate float32 scales.
    // Weights will be served directly from the mmap'd safetensor data (no copy).
    if (!config_.gate_projs.empty()) {
      return T::BufferB::required_size_scale_only(n, k, config_.quant_config.group_size);
    }
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    // Scale-only mode: b is nullptr here; set externally in load_weights().
    if (!config_.gate_projs.empty()) {
      return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data, nullptr);
    }
    return std::make_shared<typename T::BufferB>((int)n, (int)k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    avx2::gemm_rawint4(m, config_.intermediate_size, config_.hidden_size, *gate_up_ba_[expert_idx], *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_rawint4(m, config_.hidden_size, config_.intermediate_size, *down_ba_[expert_idx], *down_bb_[expert_idx],
                       *down_bc_[expert_idx], ith, nth);
  }

  void load_weights() {
    int group_size = config_.quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    const bool use_per_expert = !config_.gate_projs.empty();
    if (!use_per_expert && config_.gate_proj == nullptr) {
      throw std::runtime_error("RAWINT4 AVX2 MoE requires weight pointers");
    }
    if (!use_per_expert && config_.gate_scale == nullptr) {
      throw std::runtime_error("RAWINT4 AVX2 MoE requires scale pointers");
    }

    if (use_per_expert) {
      // Direct-pointer mode: BufferB.b is set to point into the mmap'd safetensor data
      // (no weight copy). Only float32 scales are allocated and converted.
      //
      // For gate/up: source shape [intermediate_size_full, hidden_size/2] row-major.
      //   TP partition tp_part_idx handles rows [tp_part_idx * n_per_tp, (tp_part_idx+1) * n_per_tp).
      //   These are contiguous in memory → simple byte offset: tp_part_idx * n_per_tp * (k/2).
      //   With kt_threadpool_count=1 (tp_part_idx=0): offset = 0.
      //
      // For down: source shape [hidden_size, intermediate_size_full/2] row-major.
      //   TP partition tp_part_idx handles columns [tp_part_idx * n_per_tp/2, ...) per row.
      //   These are NOT contiguous across rows for tp_count > 1.
      //   This mode therefore requires kt_threadpool_count=1 (enforced in outer TP wrapper).
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_idx) {
            if (expert_idx < 0 || expert_idx >= config_.expert_num || gate_bb_[expert_idx] == nullptr ||
                up_bb_[expert_idx] == nullptr || down_bb_[expert_idx] == nullptr) {
              return;
            }
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            // Gate/Up row offset for this TP partition (bytes).
            size_t gate_tp_byte_offset = (size_t)tp_part_idx * config_.intermediate_size * config_.hidden_size / 2;
            gate_bb_[expert_idx]->b = (uint8_t*)config_.gate_projs[0][logical_expert_id] + gate_tp_byte_offset;
            up_bb_[expert_idx]->b = (uint8_t*)config_.up_projs[0][logical_expert_id] + gate_tp_byte_offset;
            // Down column offset (bytes per row start). Correct only for tp_count=1.
            size_t down_tp_byte_offset = (size_t)tp_part_idx * config_.intermediate_size / 2;
            down_bb_[expert_idx]->b = (uint8_t*)config_.down_projs[0][logical_expert_id] + down_tp_byte_offset;
          },
          nullptr);

      // Scale conversion: BF16 → float32.
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (expert_idx >= (uint64_t)config_.expert_num || gate_bb_[expert_idx] == nullptr ||
                up_bb_[expert_idx] == nullptr || down_bb_[expert_idx] == nullptr) {
              return;
            }
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            // Gate/Up scale offset: rows [tp_part_idx * n_per_tp, ...) of scale[n_total, k/gs].
            size_t gate_scale_tp_offset =
                (size_t)tp_part_idx * config_.intermediate_size * (config_.hidden_size / group_size);
            // Down scale offset: cols [tp_part_idx * (n_per_tp/gs), ...) per row. Correct for tp_count=1.
            size_t down_scale_tp_offset = (size_t)tp_part_idx * (config_.intermediate_size / group_size);
            convert_or_copy(gate_bb_[expert_idx]->d,
                            (const ggml_bf16_t*)config_.gate_scales[0][logical_expert_id] + gate_scale_tp_offset,
                            scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d,
                            (const ggml_bf16_t*)config_.up_scales[0][logical_expert_id] + gate_scale_tp_offset,
                            scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d,
                            (const ggml_bf16_t*)config_.down_scales[0][logical_expert_id] + down_scale_tp_offset,
                            scale_elem_count);
          },
          nullptr);
    } else {
      // Flat-buffer mode: copy TP-sliced weights from flat buffer into allocated BufferB.b.
      int nth = T::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;
            size_t weight_offset = ((size_t)logical_expert_id * config_.intermediate_size * config_.hidden_size) / 2;
            gate_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.gate_proj + weight_offset, ith, nth);
            up_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.up_proj + weight_offset, ith, nth);
          },
          nullptr);

      int nth_down = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth_down * config_.expert_num, nullptr,
          [this, nth_down, physical_to_logical_map](int task_id) {
            uint64_t expert_idx = task_id / nth_down;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth_down;
            size_t weight_offset = ((size_t)logical_expert_id * config_.hidden_size * config_.intermediate_size) / 2;
            down_bb_[expert_idx]->from_raw_mat((const uint8_t*)config_.down_proj + weight_offset, ith, nth_down);
          },
          nullptr);

      // Scale conversion in flat-buffer mode.
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map, group_size](int task_id) {
            uint64_t expert_idx = task_id;
            if (config_.should_skip_expert(expert_idx)) return;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            size_t scale_elem_count = ((size_t)config_.hidden_size * config_.intermediate_size) / group_size;
            convert_or_copy(gate_bb_[expert_idx]->d,
                            (ggml_bf16_t*)config_.gate_scale + logical_expert_id * scale_elem_count, scale_elem_count);
            convert_or_copy(up_bb_[expert_idx]->d,
                            (ggml_bf16_t*)config_.up_scale + logical_expert_id * scale_elem_count, scale_elem_count);
            convert_or_copy(down_bb_[expert_idx]->d,
                            (ggml_bf16_t*)config_.down_scale + logical_expert_id * scale_elem_count, scale_elem_count);
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
        up_bb_[expert_id] == nullptr || down_bb_[expert_id] == nullptr) {
      throw std::runtime_error("RAWINT4 write_weights_to_buffer requested an expert without loaded weights");
    }
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;
    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);
      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];
      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::min(std::max(1, config_.hidden_size / MIN_COLS_PER_TASK), 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [=, this](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              size_t start = (size_t)task_id * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                std::memcpy(w13_weight_dst + offset_in_gpu_weight + start, gate_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = (size_t)chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                std::memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start,
                            up_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = (size_t)chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);
              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;
              for (size_t col = col_start; col < col_end; col++) {
                std::memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);
                fp32_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                             down_bb_[expert_id]->d + col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, up_bb_[expert_id]->d,
                           cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;
      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::min(std::max(1, config_.hidden_size / MIN_COLS_PER_TASK), 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;
      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [=, this, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;
            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];
            size_t cpu_offset_weight = (size_t)local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = (size_t)local_gpu_idx * data_per_gpu_tp_scale;
            if (task_type < NUM_WEIGHT_TASKS) {
              size_t start = (size_t)task_type * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                std::memcpy(w13_weight_dst + start, gate_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = (size_t)chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                std::memcpy(w13_weight_dst + gpu_tp_weight_bytes + start,
                            up_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = (size_t)chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);
              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;
              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);
                std::memcpy(w2_weight_dst + col * weight_per_gpu_col, down_bb_[expert_id]->b + col_offset_weight,
                            weight_per_gpu_col);
                fp32_to_bf16(w2_scale_dst + col * scale_per_gpu_col, down_bb_[expert_id]->d + col_offset_scale,
                             scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fp32_to_bf16(w13_scale_dst, gate_bb_[expert_id]->d + cpu_offset_scale, data_per_gpu_tp_scale);
            } else {
              fp32_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, up_bb_[expert_id]->d + cpu_offset_scale,
                           data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

template <typename K>
class TP_MOE<AVX2_RAW_INT4_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_RAW_INT4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_RAW_INT4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    bool use_per_expert_ptrs = !config.gate_projs.empty();
    if (config.gate_projs.empty() && config.gate_scale == nullptr) {
      throw std::runtime_error("RAWINT4 AVX2 MoE only supports packed INT4 with KGroup scales");
    }

    if (use_per_expert_ptrs) {
      // Direct-pointer mode: inner load_weights() sets BufferB.b directly from mmap'd data.
      // The down projection column-gather required for tp_count > 1 is NOT supported in
      // this mode; enforce single NUMA pool (kt_threadpool_count=1).
      if (this->tp_count > 1) {
        throw std::runtime_error(
            "RAWINT4 per-expert pointer mode requires kt_threadpool_count=1 "
            "(down projection TP column-gather is unsupported with direct pointers)");
      }
      DO_TPS_LOAD_WEIGHTS(pool);
    } else {
      // Flat-buffer mode: build a TP-sliced contiguous buffer per TP partition, then
      // call inner load_weights() which copies into BufferB.b from the flat buffer.
      int group_size = config.quant_config.group_size;
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
              uint8_t* src_gate = (uint8_t*)config.gate_proj +
                                  ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
              uint8_t* src_up =
                  (uint8_t*)config.up_proj + ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
              uint8_t* src_down = (uint8_t*)config.down_proj +
                                  ((expert_id * (size_t)config.intermediate_size * config.hidden_size) >> 1);
              ggml_bf16_t* src_gate_scale =
                  (ggml_bf16_t*)config.gate_scale +
                  expert_id * ((size_t)config.hidden_size / group_size) * config.intermediate_size;
              ggml_bf16_t* src_up_scale =
                  (ggml_bf16_t*)config.up_scale +
                  expert_id * ((size_t)config.hidden_size / group_size) * config.intermediate_size;
              ggml_bf16_t* src_down_scale =
                  (ggml_bf16_t*)config.down_scale +
                  expert_id * ((size_t)config.intermediate_size / group_size) * config.hidden_size;

              std::memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                          src_gate + ((i * weight_elem_count) >> 1), weight_elem_count >> 1);
              std::memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                          src_up + ((i * weight_elem_count) >> 1), weight_elem_count >> 1);
              std::memcpy((ggml_bf16_t*)tpc.gate_scale + expert_id * scales_elem_count,
                          src_gate_scale + i * scales_elem_count, sizeof(ggml_bf16_t) * scales_elem_count);
              std::memcpy((ggml_bf16_t*)tpc.up_scale + expert_id * scales_elem_count,
                          src_up_scale + i * scales_elem_count, sizeof(ggml_bf16_t) * scales_elem_count);

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
        printf("AVX2 RAWINT4 TP %d load weight done.\n", i);
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

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
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

#endif  // CPUINFER_OPERATOR_AVX2_RAW_INT4_MOE_H
