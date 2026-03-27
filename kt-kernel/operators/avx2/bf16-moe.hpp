/**
 * @Description  : AVX2 BF16 MoE operator (ported from amx/bf16-moe.hpp)
 * @Author       : Claude
 * @Date         : 2026-03-18
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Simplified from AMX version:
 * - BufferB::from_mat is memcpy (no AMX transpose)
 * - No unpack_nk_block_bf16 (no AMX packed format to undo)
 * - write_weights_to_buffer uses direct memcpy with full TP routing logic
 **/
#ifndef CPUINFER_OPERATOR_AVX2_BF16_MOE_H
#define CPUINFER_OPERATOR_AVX2_BF16_MOE_H

#include "avx2_bf16_gemm.hpp"
#include "moe_base.hpp"

template <class T = avx2::GemmKernelAVX2BF16>
class AVX2_BF16_MOE_TP : public AVX2_MOE_BASE<T, AVX2_BF16_MOE_TP<T>> {
  using Base = AVX2_MOE_BASE<T, AVX2_BF16_MOE_TP<T>>;
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

  AVX2_BF16_MOE_TP() = default;

  AVX2_BF16_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    printf("Created AVX2_BF16_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AVX2_BF16_MOE_TP() = default;

  // CRTP buffer creation
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // GEMM dispatch — uses avx2::gemm_bf16
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    avx2::gemm_bf16(m, config_.intermediate_size, config_.hidden_size, *ba, *bb, *bc, ith, nth);
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
    avx2::gemm_bf16(m, config_.hidden_size, config_.intermediate_size,
                    *down_ba_[expert_idx], *down_bb_[expert_idx], *down_bc_[expert_idx], ith, nth);
  }

  /**
   * Load BF16 weights from contiguous memory layout.
   * BufferB::from_mat is a trivial memcpy for AVX2 (no AMX transpose).
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

          gate_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.gate_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);

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

          down_bb_[expert_idx]->from_mat(
              (ggml_bf16_t*)config_.down_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
              ith, nth);
        },
        nullptr);
  }

  /**
   * Write weights to GPU buffer for dynamic expert offload.
   * Preserves full TP routing logic from AMX version but uses direct memcpy
   * instead of unpack_nk_block_bf16 (since BufferB is row-major, not AMX-packed).
   */
  void write_weights_to_buffer(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                               const GeneralMOEConfig& full_config, const std::vector<uintptr_t>& w13_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               [[maybe_unused]] const std::vector<uintptr_t>& w2_scale_ptrs) const {
    auto& config = config_;
    auto pool = config.pool->get_subpool(tp_part_idx);

    // W13 (gate+up): Shape [intermediate, hidden], split by N across GPU TPs
    const int cpu_n_w13 = config.intermediate_size;
    const int cpu_k_w13 = config.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = (size_t)gpu_n_w13 * gpu_k_w13;

    // W2 (down): Shape [hidden, intermediate], split by K across GPU TPs
    const int cpu_n_w2 = config.hidden_size;
    const int cpu_k_w2 = config.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;

    constexpr int NUM_W13_TASKS = 32;
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS * 2 + NUM_W2_TASKS;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w2_weight_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS * 2) {
            // W13 weight task: copy rows from BufferB to GPU buffer
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

              ggml_bf16_t* dst = (ggml_bf16_t*)w13_weight_ptrs[target_gpu];
              const size_t expert_weight_off = is_up ? gpu_w13_weight_per_mat : 0;

              // BufferB is row-major [N, K], direct copy
              std::memcpy(dst + expert_weight_off + (size_t)n_in_gpu * gpu_k_w13,
                          bb->b + (size_t)row * cpu_k_w13,
                          cpu_k_w13 * sizeof(ggml_bf16_t));
            }
          } else {
            // W2 weight task: copy rows, split K across GPU TPs
            const int chunk_idx = task_id - NUM_W13_TASKS * 2;
            const auto& bb = down_bb_[expert_id];

            const int rows_per_task = (cpu_n_w2 + NUM_W2_TASKS - 1) / NUM_W2_TASKS;
            const int row_start = chunk_idx * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w2);
            if (row_start >= cpu_n_w2) return;

            for (int row = row_start; row < row_end; row++) {
              // For W2, K dimension is split across GPU TPs
              // Iterate over all gpu_k_w2-sized slices within this CPU TP's K range
              for (int k_start = 0; k_start < cpu_k_w2; k_start += gpu_k_w2) {
                const int k_slice_end = std::min(k_start + gpu_k_w2, cpu_k_w2);
                const int k_slice_len = k_slice_end - k_start;

                // Map to correct GPU TP
                const int global_k = global_k_offset_w2 + k_start;
                const int target_gpu = global_k / gpu_k_w2;
                const int k_in_gpu = global_k % gpu_k_w2;

                ggml_bf16_t* dst = (ggml_bf16_t*)w2_weight_ptrs[target_gpu];

                std::memcpy(dst + (size_t)row * gpu_k_w2 + k_in_gpu,
                            bb->b + (size_t)row * cpu_k_w2 + k_start,
                            k_slice_len * sizeof(ggml_bf16_t));
              }
            }
          }
        },
        nullptr);
  }
};

// ============================================================================
// TP_MOE specialization for AVX2_BF16_MOE_TP
// Handles per-expert pointer loading and TP weight splitting
// (Ported from amx/bf16-moe.hpp TP_MOE<AMX_BF16_MOE_TP<K>>)
// ============================================================================
template <typename K>
class TP_MOE<AVX2_BF16_MOE_TP<K>> : public TP_MOE<AVX2_MOE_BASE<K, AVX2_BF16_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AVX2_MOE_BASE<K, AVX2_BF16_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    if (config.gate_projs.empty() && config.gate_proj == nullptr) {
      throw std::runtime_error("no weight source");
    }

    const bool use_per_expert_ptrs = !config.gate_projs.empty();
    const size_t full_weight_elems = (size_t)config.intermediate_size * config.hidden_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      const size_t tp_weight_elems = (size_t)tpc.intermediate_size * tpc.hidden_size;

      // Allocate temporary BF16 buffers for this TP part
      tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
      tpc.up_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];
      tpc.down_proj = new ggml_bf16_t[tpc.expert_num * tp_weight_elems];

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
              gate_src = (const ggml_bf16_t*)config.gate_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              up_src = (const ggml_bf16_t*)config.up_proj + expert_id * full_weight_elems + gate_up_weight_src_offset;
              down_src = (const ggml_bf16_t*)config.down_proj + expert_id * full_weight_elems;
            }

            // Copy gate and up weights (column-slice for TP)
            std::memcpy(gate_dst, gate_src, tp_weight_elems * sizeof(ggml_bf16_t));
            std::memcpy(up_dst, up_src, tp_weight_elems * sizeof(ggml_bf16_t));

            // Copy down weights (row-wise split: each row picks a slice of columns)
            for (int row = 0; row < config.hidden_size; row++) {
              const size_t src_row_offset = (size_t)row * (size_t)config.intermediate_size + down_weight_src_col_offset;
              const size_t dst_row_offset = (size_t)row * (size_t)tpc.intermediate_size;
              std::memcpy(down_dst + dst_row_offset, down_src + src_row_offset,
                          (size_t)tpc.intermediate_size * sizeof(ggml_bf16_t));
            }
          },
          nullptr);
    });

    // Call per-TP load_weights (which does BufferB::from_mat = memcpy)
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      tps[i]->load_weights();
    });

    // Free temporary buffers
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (ggml_bf16_t*)tpc.gate_proj;
      delete[] (ggml_bf16_t*)tpc.up_proj;
      delete[] (ggml_bf16_t*)tpc.down_proj;
    });

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if ((int)w13_weight_ptrs.size() != gpu_tp_count || (int)w2_weight_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Weight pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AVX2_BF16_MOE_H
