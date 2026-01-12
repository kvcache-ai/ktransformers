/**
 * @Description  : BF16 AMX MoE operator for native BF16 inference
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-01-06
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements BF16 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * BF16 weights are stored without quantization (no scales).
 **/
#ifndef CPUINFER_OPERATOR_AMX_BF16_MOE_H
#define CPUINFER_OPERATOR_AMX_BF16_MOE_H

// #define DEBUG_BF16_MOE

#include "la/amx_kernels.hpp"  // For vec_mul/mat_mul
#include "la/amx_raw_buffers.hpp"
#include "la/amx_raw_kernels.hpp"
#include "la/amx_utils.hpp"  // For transpose_16x16_32bit
#include "moe_base.hpp"

/**
 * @brief BF16 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to GemmKernel224BF16
 *
 * This class provides BF16-specific implementations:
 * - do_gate_up_gemm, do_down_gemm: BF16 weight mat mul (no quantization)
 * - load_weights: Load native BF16 weights (no scales)
 */
template <class T = amx::GemmKernel224BF16>
class AMX_BF16_MOE_TP : public AMX_MOE_BASE<T, AMX_BF16_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_BF16_MOE_TP<T>>;
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

  AMX_BF16_MOE_TP() = default;

  AMX_BF16_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    // Initialization now happens in derived_init() which is called by base constructor
  }

  void derived_init() {
    // BF16 has no quantization, no need to check quant_config
    printf("Created AMX_BF16_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_BF16_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - without group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }

  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k);  // 2 parameters - no group_size
  }

  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }

  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);  // 2 parameters - no group_size
  }

  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Use vec_mul/mat_mul (no group_size)
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    }
  }

#ifdef DEBUG_BF16_MOE
  // Function to dump Buffer B data for debugging
  inline void dump_buffer_b(int expert_idx, const std::string& matrix_type, typename T::BufferB* buffer) {
    printf("[DUMP_BUFFER_B] TP%d BF16 Expert%d %s:\n", tp_part_idx, expert_idx, matrix_type.c_str());

    // Calculate dimensions based on matrix type
    int rows, cols;
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
    }

    // Dump BF16 weights
    size_t weight_size = (size_t)rows * cols;
    ggml_bf16_t* weight_ptr = buffer->b;

    printf("  BF16 Weights[first 16]: ");
    for (int i = 0; i < std::min(16, (int)weight_size); i++) {
      printf("%.6f ", ggml_bf16_to_fp32(weight_ptr[i]));
    }
    printf("\n");

    if (weight_size > 16) {
      printf("  BF16 Weights[last 16]: ");
      int start_idx = std::max(0, (int)weight_size - 16);
      for (int i = start_idx; i < (int)weight_size; i++) {
        printf("%.6f ", ggml_bf16_to_fp32(weight_ptr[i]));
      }
      printf("\n");
    }

    printf("  Matrix dimensions: %dx%d (n x k)\n", rows, cols);
  }
#endif

  /**
   * @brief Load BF16 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj (no scales).
   */
  void load_weights() {
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (config_.gate_proj == nullptr) {
      throw std::runtime_error("BF16 MOE requires native BF16 weight.");
    }

    // Load gate + up weights
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Gate: from BF16 data (no scale)
          gate_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.gate_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);  // 3 parameters: (bf16*, ith, nth)

          // Up: same
          up_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.up_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);
        },
        nullptr);

    // Load down weights
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;

          // Down
          down_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.down_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);
        },
        nullptr);

#ifdef DEBUG_BF16_MOE
    dump_buffer_b(0, "gate", gate_bb_[0].get());
    dump_buffer_b(0, "down", down_bb_[0].get());
#endif
  }

  // Fast 64-byte (512-bit) memcpy using AVX512
  static inline void fast_memcpy_64(void* __restrict dst, const void* __restrict src) {
    __m512i data = _mm512_loadu_si512(src);
    _mm512_storeu_si512(dst, data);
  }

  // Fast memcpy for arbitrary sizes using AVX512
  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      fast_memcpy_64(d, s);
      d += 64;
      s += 64;
    }
    bytes -= chunks * 64;
    if (bytes > 0) {
      std::memcpy(d, s, bytes);
    }
  }

  /**
   * @brief Unpack a single N_STEP x K_STEP block from packed BufferB format to n-major format (BF16 version)
   *
   * This is the inverse of the packing done in BufferBBF16Impl::from_mat.
   * BF16 elements are 2 bytes, and the packed format includes 16x16 32-bit transpose.
   *
   * @param src Pointer to packed data (N_STEP * K_STEP * 2 bytes in packed layout)
   * @param dst Pointer to destination in n-major layout
   * @param dst_row_stride Row stride in destination buffer (number of BF16 elements per row)
   */
  static inline void unpack_nk_block_bf16(const ggml_bf16_t* src, ggml_bf16_t* dst, size_t dst_row_stride) {
    constexpr int N_STEP = T::N_STEP;  // 32
    constexpr int K_STEP = T::K_STEP;  // 32
    constexpr int TILE_N = T::TILE_N;  // 16

    // The packed format has two 16x16 blocks (32-bit view) that were transposed
    // We need to reverse the transpose first, then copy to n-major layout

    // Create aligned temporary buffers for transpose
    alignas(64) __m512i temp_block1[TILE_N];
    alignas(64) __m512i temp_block2[TILE_N];

    // Copy source data to temporary buffers
    const __m512i* src_vec = reinterpret_cast<const __m512i*>(src);
    for (int i = 0; i < TILE_N; i++) {
      temp_block1[i] = src_vec[i];
      temp_block2[i] = src_vec[TILE_N + i];
    }

    // Reverse transpose (transpose is self-inverse)
    amx::transpose_16x16_32bit(temp_block1);
    amx::transpose_16x16_32bit(temp_block2);

    // Copy transposed data to destination in n-major layout
    const ggml_bf16_t* temp1_bf16 = reinterpret_cast<const ggml_bf16_t*>(temp_block1);
    const ggml_bf16_t* temp2_bf16 = reinterpret_cast<const ggml_bf16_t*>(temp_block2);

    // First 16 rows (block 1)
    for (int i = 0; i < TILE_N; i++) {
      std::memcpy(dst + i * dst_row_stride, temp1_bf16 + i * K_STEP, K_STEP * sizeof(ggml_bf16_t));
    }

    // Next 16 rows (block 2)
    for (int i = 0; i < TILE_N; i++) {
      std::memcpy(dst + (TILE_N + i) * dst_row_stride, temp2_bf16 + i * K_STEP, K_STEP * sizeof(ggml_bf16_t));
    }
  }

  /**
   * @brief Reconstruct weights for a single expert to the output buffers
   *
   * Directly unpacks from packed BufferB format to n-major GPU buffers without intermediate storage.
   * BF16 version - no scales needed.
   *
   * @param gpu_tp_count Number of GPU TP parts (1, 2, 4, or 8)
   * @param cpu_tp_count Number of CPU TP parts
   * @param expert_id Expert index to process
   * @param full_config Full configuration (before CPU TP split)
   * @param w13_weight_ptrs Pointers to gate+up weight buffers (one per GPU TP)
   * @param w13_scale_ptrs Pointers to gate+up scale buffers (unused for BF16, kept for interface compatibility)
   * @param w2_weight_ptrs Pointers to down weight buffers (one per GPU TP)
   * @param w2_scale_ptrs Pointers to down scale buffers (unused for BF16, kept for interface compatibility)
   */
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);

    constexpr int N_STEP = T::N_STEP;
    constexpr int K_STEP = T::K_STEP;
    constexpr int N_BLOCK = T::N_BLOCK;
    constexpr int K_BLOCK = T::K_BLOCK;

    // ========= W13 (gate+up): Shape [intermediate, hidden], split by N only =========
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;

    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;

    // ========= W2 (down): Shape [hidden, intermediate], split by K =========
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_n_w2 = full_config.hidden_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;

    // ========= Optimized job layout =========
    constexpr int NUM_W13_TASKS = 32;  // Per matrix (gate or up), total 64 for w13
    constexpr int NUM_W2_TASKS = 32;   // For down matrix

    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    // Calculate N_STEP blocks per task
    const int w13_n_steps = div_up(cpu_n_w13, N_STEP);
    const int w13_steps_per_task = div_up(w13_n_steps, NUM_W13_TASKS);
    const int w2_n_steps = div_up(cpu_n_w2, N_STEP);
    const int w2_steps_per_task = div_up(w2_n_steps, NUM_W2_TASKS);

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w2_weight_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // ========= W13 weight task: process chunk of rows x full K =========
            const bool is_up = task_id >= NUM_W13_TASKS;
            const int chunk_idx = task_id % NUM_W13_TASKS;
            const auto& bb = is_up ? up_bb_[expert_id] : gate_bb_[expert_id];

            const int step_start = chunk_idx * w13_steps_per_task;
            const int step_end = std::min(step_start + w13_steps_per_task, w13_n_steps);
            if (step_start >= w13_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w13);

            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              const int global_n = global_n_offset_w13 + local_n_start;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              ggml_bf16_t* weight_base = (ggml_bf16_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_weight_off = is_up ? gpu_w13_weight_per_mat : 0;

              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w13 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              for (int k_block_begin = 0; k_block_begin < cpu_k_w13; k_block_begin += K_BLOCK) {
                const int k_block_size = std::min(K_BLOCK, cpu_k_w13 - k_block_begin);

                for (int k_begin = 0; k_begin < k_block_size; k_begin += K_STEP) {
                  const ggml_bf16_t* src = bb->b + (size_t)n_block_begin * cpu_k_w13 +
                                           (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                           (size_t)k_begin * N_STEP;
                  ggml_bf16_t* dst =
                      weight_base + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13 + k_block_begin + k_begin;
                  unpack_nk_block_bf16(src, dst, gpu_k_w13);
                }
              }
            }

          } else {
            // ========= W2 weight task: process chunk of rows x all K slices =========
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int step_start = chunk_idx * w2_steps_per_task;
            const int step_end = std::min(step_start + w2_steps_per_task, w2_n_steps);
            if (step_start >= w2_n_steps) return;
            const int chunk_n_start = step_start * N_STEP;
            const int chunk_n_end = std::min(step_end * N_STEP, cpu_n_w2);

            for (int local_n_start = chunk_n_start; local_n_start < chunk_n_end; local_n_start += N_STEP) {
              const int n_block_idx = local_n_start / N_BLOCK;
              const int n_block_begin = n_block_idx * N_BLOCK;
              const int n_block_size = std::min(N_BLOCK, cpu_n_w2 - n_block_begin);
              const int n_in_block = local_n_start - n_block_begin;

              for (int k_slice_start = 0; k_slice_start < cpu_k_w2; k_slice_start += gpu_k_w2) {
                const int k_slice_end = std::min(k_slice_start + gpu_k_w2, cpu_k_w2);

                const int global_k_start = global_k_offset_w2 + k_slice_start;
                const int target_gpu = global_k_start / gpu_k_w2;
                const int k_in_gpu_base = global_k_start % gpu_k_w2;

                ggml_bf16_t* weight_base = (ggml_bf16_t*)w2_weight_ptrs[target_gpu];

                for (int k_abs = k_slice_start; k_abs < k_slice_end; k_abs += K_STEP) {
                  const int k_block_idx = k_abs / K_BLOCK;
                  const int k_block_begin = k_block_idx * K_BLOCK;
                  const int k_block_size = std::min(K_BLOCK, cpu_k_w2 - k_block_begin);
                  const int k_in_block = k_abs - k_block_begin;
                  const int k_in_gpu = k_in_gpu_base + (k_abs - k_slice_start);

                  const ggml_bf16_t* src = bb->b + (size_t)n_block_begin * cpu_k_w2 +
                                           (size_t)k_block_begin * n_block_size + (size_t)n_in_block * k_block_size +
                                           (size_t)k_in_block * N_STEP;
                  ggml_bf16_t* dst = weight_base + (size_t)local_n_start * gpu_k_w2 + k_in_gpu;
                  unpack_nk_block_bf16(src, dst, gpu_k_w2);
                }
              }
            }
          }
        },
        nullptr);
  }
};

template <typename K>
class TP_MOE<AMX_BF16_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_BF16_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_BF16_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    // BF16 has no quantization check needed
    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }

    const bool use_per_expert_ptrs = !config.gate_projs.empty();
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;

      // Allocate BF16 weights (2 bytes/element)
      tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];

      const size_t tp_idx = (size_t)i;
      const size_t gate_up_weight_src_offset = i * tp_weight_elems;
      const size_t down_weight_src_col_offset = i * (size_t)tpc.intermediate_size;

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, &tpc](int expert_id_) {
            const size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            ggml_bf16_t* gate_dst = (ggml_bf16_t*)tpc.gate_proj + expert_id * tp_weight_elems;
            ggml_bf16_t* up_dst = (ggml_bf16_t*)tpc.up_proj + expert_id * tp_weight_elems;
            ggml_bf16_t* down_dst = (ggml_bf16_t*)tpc.down_proj + expert_id * tp_weight_elems;

            const ggml_bf16_t* gate_src;
            const ggml_bf16_t* up_src;
            const ggml_bf16_t* down_src;

            if (use_per_expert_ptrs) {
              gate_src = (const ggml_bf16_t*)config.gate_projs[0][expert_id] + gate_up_weight_src_offset;
              up_src = (const ggml_bf16_t*)config.up_projs[0][expert_id] + gate_up_weight_src_offset;
              down_src = (const ggml_bf16_t*)config.down_projs[0][expert_id];
            } else {
              gate_src =
                  (const ggml_bf16_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src = (const ggml_bf16_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const ggml_bf16_t*)config.down_proj + expert_id * full_weight_elems;
            }

            // Copy gate and up weights
            std::memcpy(gate_dst, gate_src, tp_weight_elems * sizeof(ggml_bf16_t));
            std::memcpy(up_dst, up_src, tp_weight_elems * sizeof(ggml_bf16_t));

            // Copy down weights (row-wise split)
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset,
                          (size_t)tpc.intermediate_size * sizeof(ggml_bf16_t));
            }
          },
          nullptr);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (ggml_bf16_t*)tpc.gate_proj;
      delete[] (ggml_bf16_t*)tpc.up_proj;
      delete[] (ggml_bf16_t*)tpc.down_proj;
    });

    this->weights_loaded = true;
  }

  /**
   * @brief Write weights to GPU buffer for all TP parts
   *
   * BF16 version - no scales needed, scale_ptrs parameters are kept for interface compatibility.
   */
  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Weight pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_BF16_MOE_H
