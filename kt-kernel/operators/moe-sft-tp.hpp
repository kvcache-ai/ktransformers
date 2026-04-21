/**
 * @Description  : TP (Tensor Parallel) wrapper for SFT MoE operations.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_SFT_TP_HPP
#define CPUINFER_OPERATOR_MOE_SFT_TP_HPP

static constexpr int kMoeSftTpVersion = 3;

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "amx/la/amx.hpp"
#include "moe-tp.hpp"

struct TPBf16Stats {
  double abs_mean = 0.0;
  double abs_max = 0.0;
  double norm = 0.0;
};

static inline TPBf16Stats compute_tp_bf16_stats(const ggml_bf16_t* buf, size_t size) {
  TPBf16Stats stats;
  if (buf == nullptr || size == 0) {
    return stats;
  }
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double max_abs = 0.0;
  for (size_t i = 0; i < size; i++) {
    float v = GGML_BF16_TO_FP32(buf[i]);
    double a = std::fabs(static_cast<double>(v));
    sum_abs += a;
    sum_sq += static_cast<double>(v) * static_cast<double>(v);
    if (a > max_abs) {
      max_abs = a;
    }
  }
  stats.abs_mean = sum_abs / static_cast<double>(size);
  stats.abs_max = max_abs;
  stats.norm = std::sqrt(sum_sq);
  return stats;
}

static inline void print_tp_bf16_stats(int layer_idx, const char* name, const ggml_bf16_t* buf, size_t size) {
  return;
  if (buf == nullptr) {
    printf("KT MoE TP update stats (layer %d, %s): null\n", layer_idx, name);
    return;
  }
  TPBf16Stats stats = compute_tp_bf16_stats(buf, size);
  printf("KT MoE TP update stats (layer %d, %s): abs_mean=%.6e abs_max=%.6e norm=%.6e\n", layer_idx, name,
         stats.abs_mean, stats.abs_max, stats.norm);
}

// Forward declaration
template <class T, template <class> class BaseMOE, bool SkipLoRA>
class AMX_SFT_MOE_TP;

/**
 * @brief Shared TP backward temporary pools (one buffer per TP index).
 *
 * Backward for different layers runs sequentially in this training path, so
 * per-TP temporary buffers can be reused across layers instead of being kept
 * per-layer/per-instance.
 */
struct SFTTPSharedBackwardPools {
  struct PerTP {
    void* work = nullptr;
    size_t work_bytes = 0;
  };

  std::mutex lock;
  std::vector<PerTP> pools;

  static SFTTPSharedBackwardPools& instance() {
    static SFTTPSharedBackwardPools inst;
    return inst;
  }

  void ensure_tp_count(int n) {
    if ((int)pools.size() < n) pools.resize(n);
  }

  static void* acquire(void*& ptr, size_t& cur_bytes, size_t required, size_t align) {
    required = (required + align - 1) / align * align;
    if (required == 0) return ptr;
    if (required <= cur_bytes) return ptr;
    if (ptr) {
      free(ptr);
      ptr = nullptr;
      cur_bytes = 0;
    }
    void* new_ptr = nullptr;
    int rc = posix_memalign(&new_ptr, align, required);
    if (rc != 0 || !new_ptr) {
      errno = rc;  // posix_memalign returns error code instead of setting errno
      perror("posix_memalign");
      throw std::runtime_error("posix_memalign failed");
    }
    ptr = new_ptr;
    cur_bytes = required;
    return ptr;
  }

  ~SFTTPSharedBackwardPools() {
    for (auto& p : pools) {
      if (p.work) {
        free(p.work);
        p.work = nullptr;
      }
      p.work_bytes = 0;
    }
  }

 private:
  SFTTPSharedBackwardPools() = default;
};

/**
 * @brief TP_MOE_SFT - Tensor Parallel wrapper for SFT MoE with LoRA support.
 *
 * Inherits from TP_MOE<T> and adds SFT-specific methods:
 * - forward_sft: Forward pass with optional caching for backward
 * - backward: Backward pass computing LoRA gradients
 *
 * @tparam T The underlying MoE implementation (e.g., AMX_SFT_MOE_TP<GemmKernel224BF>)
 */
template <class T>
class TP_MOE_SFT : public TP_MOE<T> {
 public:
  static constexpr bool kSkipLoRA = T::kSkipLoRA;

  using Base = TP_MOE<T>;
  using Base::config;
  using Base::local_output_numa;
  using Base::tp_configs;
  using Base::tp_count;
  using Base::tps;
  using Base::weights_loaded;

  MOESFTConfig sft_config;

  // Bug #19 fix: Partitioned LoRA weight pointers for each NUMA node
  // (Need to be freed on update or destruction)
  std::vector<ggml_bf16_t*> partitioned_gate_lora_b_;
  std::vector<ggml_bf16_t*> partitioned_up_lora_b_;
  std::vector<ggml_bf16_t*> partitioned_down_lora_a_;

  // Bug #20 fix: Partitioned base weight pointers for backward pass
  // (Need to be freed on destruction - backward uses original BF16 weights)
  std::vector<ggml_bf16_t*> partitioned_gate_proj_;
  std::vector<ggml_bf16_t*> partitioned_up_proj_;
  std::vector<ggml_bf16_t*> partitioned_down_proj_;

 private:
  static constexpr size_t kAmxAlignment = 64;
  static inline size_t round_up(size_t x, size_t align) { return (x + align - 1) / align * align; }

  void alloc_or_resize_backward_pool(int tp_idx, size_t required_bytes) {
    required_bytes = round_up(required_bytes, kAmxAlignment);
    if (required_bytes == 0) {
      backward_temp_pools_[tp_idx] = nullptr;
      backward_temp_pool_bytes_[tp_idx] = 0;
      return;
    }
    auto& shared = SFTTPSharedBackwardPools::instance();
    {
      std::lock_guard<std::mutex> guard(shared.lock);
      shared.ensure_tp_count(tp_idx + 1);
      auto& p = shared.pools[tp_idx];
      backward_temp_pools_[tp_idx] =
          SFTTPSharedBackwardPools::acquire(p.work, p.work_bytes, required_bytes, kAmxAlignment);
      backward_temp_pool_bytes_[tp_idx] = p.work_bytes;
    }
  }

  void free_backward_temp_pools() {
    // Shared pools are singleton-owned; per-instance destructor should only
    // clear local references.
    for (size_t i = 0; i < backward_temp_pools_.size(); i++) {
      backward_temp_pools_[i] = nullptr;
      backward_temp_pool_bytes_[i] = 0;
    }
  }

  // Async backward repack state (Phase 2: overlap repack with GPU attention backward)
  std::thread repack_thread_;
  std::atomic<bool> repack_in_flight_{false};

  // Per-instance references to shared per-TP backward temporary pools.
  std::vector<void*> backward_temp_pools_;
  std::vector<size_t> backward_temp_pool_bytes_;

  // Cached per-TP pointers into backward_temp_pools_
  std::vector<ggml_bf16_t*> part_grad_gate_lora_b_;
  std::vector<ggml_bf16_t*> part_grad_up_lora_b_;
  std::vector<ggml_bf16_t*> part_grad_down_lora_a_;
  std::vector<ggml_bf16_t*> part_grad_gate_lora_a_;
  std::vector<ggml_bf16_t*> part_grad_up_lora_a_;
  std::vector<ggml_bf16_t*> part_grad_input_;
  std::vector<float*> part_grad_weights_;

 public:
  TP_MOE_SFT(const MOESFTConfig& config) : Base(static_cast<const GeneralMOEConfig&>(config)), sft_config(config) {
    printf("Creating TP_MOE_SFT layer %d\n", config.layer_idx);

    backward_temp_pools_.assign(tp_count, nullptr);
    backward_temp_pool_bytes_.assign(tp_count, 0);
    part_grad_gate_lora_b_.assign(tp_count, nullptr);
    part_grad_up_lora_b_.assign(tp_count, nullptr);
    part_grad_down_lora_a_.assign(tp_count, nullptr);
    part_grad_gate_lora_a_.assign(tp_count, nullptr);
    part_grad_up_lora_a_.assign(tp_count, nullptr);
    part_grad_input_.assign(tp_count, nullptr);
    part_grad_weights_.assign(tp_count, nullptr);

    if constexpr (!kSkipLoRA) {
      // Bug #16 fix: TP_MOE base class uses GeneralMOEConfig (object slicing) which loses
      // LoRA pointers. We need to propagate LoRA pointers to all NUMA node instances.
      if (config.gate_lora_a != nullptr) {
        update_lora_weights(config.gate_lora_a, config.gate_lora_b, config.up_lora_a, config.up_lora_b,
                            config.down_lora_a, config.down_lora_b);
      }

      // Bug #007 fix: TP_MOE base class uses GeneralMOEConfig which doesn't have
      // lora_rank/lora_alpha. Propagate both to all NUMA node instances.
      for (int i = 0; i < tp_count; i++) {
        tps[i]->set_lora_params(config.lora_rank, config.lora_alpha);
      }
    }
  }

  /**
   * @brief Load weights on all NUMA nodes with TP partitioning.
   *
   * Bug #19 fix: The base weights (gate_proj, up_proj, down_proj) need to be partitioned
   * for TP mode, similar to how TP_MOE<AMX_MOE_BASE>::load_weights() does it in moe.hpp.
   * Without this, each NUMA node loads the full weights and computes the full output,
   * resulting in 2x the expected output after merge.
   */
  void load_weights() override {
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    // Bug #27 fix: K2 pre-quantized mode detection
    // K2 uses gate_scale != nullptr and zero_point = false
    // AWQ also has gate_scale but has zero_point = true
    bool is_k2_prequantized = (config.gate_scale != nullptr && !config.quant_config.zero_point);

    if (!config.gate_projs.empty()) {
      // Pre-quantized per-NUMA weights (INT8/INT4 with separate scales)
      printf("TP_MOE_SFT: Pre-quantized per-NUMA mode (gate_projs path)\n");
      pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

      // Check if pre-quantized backward weights are available
      if (!config.gate_bwd_projs.empty()) {
        if (!config.share_backward_bb) {
          printf("  [MEM] Pre-quantized backward weights available, loading via memcpy...\n");
          pool->dispense_backend()->do_numa_job(
              [this](int numa_id) { tps[numa_id]->load_backward_weights_from_projs(); });
        } else {
          printf("  [MEM] share_backward_bb: skipping pre-quantized backward weight load (dynamic repack)\n");
        }
      }
      // Also partition BF16 weights for backward gradient computation if available.
      // C++ backward needs BF16 base weights to compute gate/up LoRA B gradients
      // through the gated MLP chain (prepare_backward_weights checks config_.gate_proj).
      else if (config.gate_proj != nullptr && !config.share_backward_bb) {
        printf("  [MEM] BF16 backward weights available, partitioning for TP...\n");
        std::vector<ggml_bf16_t*> temp_gate(tp_count);
        std::vector<ggml_bf16_t*> temp_up(tp_count);
        std::vector<ggml_bf16_t*> temp_down(tp_count);

        for (int i = 0; i < tp_count; i++) {
          auto& tpc = tp_configs[i];
          size_t gate_up_elcount = (size_t)tpc.intermediate_size * tpc.hidden_size;

          temp_gate[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
          temp_up[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
          temp_down[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i, gate_up_elcount](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                size_t src_gate_offset =
                    expert_id * config.intermediate_size * config.hidden_size + i * gate_up_elcount;
                size_t dst_offset = expert_id * gate_up_elcount;
                size_t copy_bytes = sizeof(ggml_bf16_t) * gate_up_elcount;

                memcpy(temp_gate[i] + dst_offset, (ggml_bf16_t*)config.gate_proj + src_gate_offset, copy_bytes);

                memcpy(temp_up[i] + dst_offset, (ggml_bf16_t*)config.up_proj + src_gate_offset, copy_bytes);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy(
                      temp_down[i] + expert_id * tpc.hidden_size * tpc.intermediate_size + col * tpc.intermediate_size,
                      (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                          col * config.intermediate_size + i * tpc.intermediate_size,
                      sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }

        // Set BF16 weight pointers on sub-MOEs for backward
        for (int i = 0; i < tp_count; i++) {
          tps[i]->prepare_bwd(temp_gate[i], temp_up[i], temp_down[i]);
        }

        // free the memory
        for (int i = 0; i < tp_count; i++) {
          delete[] (temp_gate[i]);
          delete[] (temp_up[i]);
          delete[] (temp_down[i]);
        }
      }
    } else if (is_k2_prequantized) {
      // For K2, weights are already int4-packed with scales
      // tp_configs[i] already has all pointers from config (copied in TP_MOE constructor)
      if (tp_count == 1) {
        // No-TP: just call load_weights directly
        pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
      } else {
        // TP mode with K2 would need int4-aware partitioning (not implemented yet)
        throw std::runtime_error("K2 pre-quantized mode does not support TP > 1 yet");
      }
    } else if (config.gate_proj != nullptr) {
      printf("TP_MOE_SFT: From BF16 with partitioning\n");

      // Temporary storage for partitioned weights
      std::vector<ggml_bf16_t*> temp_gate(tp_count);
      std::vector<ggml_bf16_t*> temp_up(tp_count);
      std::vector<ggml_bf16_t*> temp_down(tp_count);

      // Step 1: For each NUMA, allocate and copy partitioned weights
      for (int i = 0; i < tp_count; i++) {
        // Use tp_configs[i] instead of tps[i]->config_ (which is protected)
        auto& tpc = tp_configs[i];
        size_t gate_up_elcount = (size_t)tpc.intermediate_size * tpc.hidden_size;

        // Allocate partitioned weight space
        temp_gate[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        temp_up[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        temp_down[i] = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

        // Copy partitioned weights
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i, gate_up_elcount](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              // gate_proj/up_proj: [intermediate_size, hidden_size] - contiguous block slice
              memcpy(temp_gate[i] + expert_id * gate_up_elcount,
                     (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                         i * gate_up_elcount,
                     sizeof(ggml_bf16_t) * gate_up_elcount);

              memcpy(temp_up[i] + expert_id * gate_up_elcount,
                     (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                         i * gate_up_elcount,
                     sizeof(ggml_bf16_t) * gate_up_elcount);

              // down_proj: [hidden_size, intermediate_size] - row-wise slice
              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy(temp_down[i] + expert_id * tpc.hidden_size * tpc.intermediate_size + col * tpc.intermediate_size,
                       (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                           col * config.intermediate_size + i * tpc.intermediate_size,
                       sizeof(ggml_bf16_t) * tpc.intermediate_size);
              }
            },
            nullptr);
      }

      // Step 2: Set weight pointers BEFORE load_weights (Bug #24 fix)
      for (int i = 0; i < tp_count; i++) {
        tps[i]->set_weight_pointers_for_forward(temp_gate[i], temp_up[i], temp_down[i]);
      }

      pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

      // Step 3: Prepare backward weights (this also clears weight pointers)
      for (int i = 0; i < tp_count; i++) {
        if (!config.share_backward_bb) {
          tps[i]->prepare_bwd(temp_gate[i], temp_up[i], temp_down[i]);
        }
        tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
      }

      for (int i = 0; i < tp_count; i++) {
        delete[] (temp_gate[i]);
        delete[] (temp_up[i]);
        delete[] (temp_down[i]);
      }
    } else {
      // Other loading methods (from loader or file)
      pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

      // Try loading backward weights from disk (.kt files) — parallel across NUMA nodes.
      if (!config.share_backward_bb) {
        pool->dispense_backend()->do_numa_job(
            [this](int numa_id) { tps[numa_id]->prepare_bwd(nullptr, nullptr, nullptr); });
      } else {
        printf("  [MEM] share_backward_bb: skipping .kt backward weight load (dynamic repack)\n");
      }
    }

    weights_loaded = true;
  }

  /**
   * @brief Merge results from all NUMA nodes.
   */
  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& tp_count_ref = this->tp_count;
    auto& local_output_numa_ref = this->local_output_numa;
    auto& tp_configs_ref = this->tp_configs;

    auto merge_fn = [this, output, incremental, &tp_count_ref, &local_output_numa_ref, &tp_configs_ref](int token_nth) {
      float* merge_to = local_output_numa_ref[0] + token_nth * tp_configs_ref[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count_ref; i++) {
        float* merge_from = local_output_numa_ref[i] + token_nth * tp_configs_ref[i].hidden_size;
        for (int e = 0; e < tp_configs_ref[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };

    auto pool = config.pool;
    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) merge_fn(i);
    } else {
      pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
    }
  }

  /**
   * @brief SFT forward pass with NUMA distribution.
   *
   * @param qlen Number of tokens
   * @param k Number of experts per token
   * @param expert_ids Expert indices [qlen, k]
   * @param weights Expert weights [qlen, k]
   * @param input Input tensor [qlen, hidden_size]
   * @param output Output tensor [qlen, hidden_size]
   * @param save_for_backward Whether to save intermediate values for backward
   */
  void forward_sft(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
                   bool save_for_backward) {
    int qlen_local = qlen;
    forward_sft(&qlen_local, k, expert_ids, weights, input, output, save_for_backward);
  }

  void forward_sft(int* qlen_ptr, int k, const int64_t* expert_ids, const float* weights, const void* input,
                   void* output, bool save_for_backward) {
    if (weights_loaded == false) [[unlikely]] {
      throw std::runtime_error("Weights not loaded");
    }


    int qlen = *qlen_ptr;
    auto pool = config.pool;

    // Reset forward timing before computation
    // Reset per-thread counters in each subpool (to accumulate all do_work_stealing_job calls)
    for (int i = 0; i < tp_count; i++) {
    }

    // Run forward on each NUMA node
    pool->dispense_backend()->do_numa_job([this, qlen, k, expert_ids, input, weights, save_for_backward](int numa_id) {
      tps[numa_id]->forward_sft(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id],
                                save_for_backward);
    });


    // // Collect per-thread timing from all NUMA subpools
    // for (int i = 0; i < tp_count; i++) {
    // }

    // // Print per-thread forward timing

    // Merge results from all NUMA nodes
    this->merge_results(qlen, output);


    pool->dispense_backend()->do_numa_job([&](int numa_id) {
    });
  }

  /**
   * @brief Python binding for forward_sft.
   */
  void forward_sft_binding(intptr_t qlen_ptr, int k, intptr_t expert_ids, intptr_t weights, intptr_t input,
                           intptr_t output, bool save_for_backward) {
    forward_sft((int*)qlen_ptr, k, (const int64_t*)expert_ids, (const float*)weights, (const void*)input, (void*)output,
                save_for_backward);
  }

  /**
   * @brief Backward pass with NUMA distribution and gradient partitioning.
   *
   * Bug #21 fix: Gradients containing intermediate_size dimension need to be partitioned
   * for TP mode, similar to how update_lora_weights() partitions weights.
   * - Forward: partition full weights → each NUMA gets partitioned weights
   * - Backward: each NUMA computes partitioned gradients → merge to full gradients
   *
   * Gradients requiring partitioning:
   * - grad_gate_lora_b: [expert_num, intermediate_size, lora_rank] - contiguous slice
   * - grad_up_lora_b: [expert_num, intermediate_size, lora_rank] - contiguous slice
   * - grad_down_lora_a: [expert_num, lora_rank, intermediate_size] - row-wise slice
   *
   * Gradients NOT requiring partitioning:
   * - grad_gate_lora_a: [expert_num, lora_rank, hidden_size]
   * - grad_up_lora_a: [expert_num, lora_rank, hidden_size]
   * - grad_down_lora_b: [expert_num, hidden_size, lora_rank]
   */
  void backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b,
                void* grad_weights) {
    auto pool = config.pool;


    // Get full intermediate_size (before TP partitioning)
    int full_intermediate_size = sft_config.intermediate_size;
    int expert_num = config.expert_num;
    int lora_rank = sft_config.lora_rank;
    int hidden_size = config.hidden_size;
    int qlen = tps[0]->get_cache_qlen();  // Get qlen from cache

    int k = sft_config.num_experts_per_tok;
    const bool need_grad_weights = (grad_weights != nullptr);

    // SkipLoRA: zero out lora_rank to skip all LoRA buffer allocations
    if constexpr (kSkipLoRA) lora_rank = 0;

    // Snapshot active expert metadata before dispatch (cache is popped inside backward())
    int active_count = tps[0]->get_cache_activated_expert_count();
    std::vector<int> active_expert_map(active_count);
    if (active_count > 0) {
      std::memcpy(active_expert_map.data(), tps[0]->get_cache_expert_id_map(), active_count * sizeof(int));
    }

    // =====================================================================
    // Allocate per-TP temporary buffers.
    //
    // New contract:
    //   Copy-type grads (gate_lora_b, up_lora_b, down_lora_a):
    //     Kernel writes directly to final output tensor TP slices — no per-TP partial buffer.
    //   Reduce-type grads (gate_lora_a, up_lora_a, down_lora_b):
    //     Per-TP sparse FP32 partial buffers scoped to active_count experts.
    //   grad_input, grad_weights: per-TP partial buffers as before.
    // =====================================================================
    const size_t lora_a_sparse_elems = (size_t)active_count * (size_t)lora_rank * (size_t)hidden_size;
    const size_t down_b_sparse_elems = (size_t)active_count * (size_t)hidden_size * (size_t)lora_rank;

    std::vector<size_t> clear_bytes(tp_count, 0);
    for (int i = 0; i < tp_count; i++) {
      const size_t grad_input_elems = (size_t)qlen * (size_t)hidden_size;
      const size_t grad_weights_elems = need_grad_weights ? ((size_t)qlen * (size_t)k) : 0;

      const size_t lora_a_sparse_bytes = lora_a_sparse_elems * sizeof(float);
      const size_t down_b_sparse_bytes = down_b_sparse_elems * sizeof(float);
      const size_t grad_input_bytes = grad_input_elems * sizeof(ggml_bf16_t);
      const size_t grad_weights_bytes = grad_weights_elems * sizeof(float);

      size_t required = 0;
      required += round_up(lora_a_sparse_bytes, kAmxAlignment) * 2;  // gate_lora_a + up_lora_a (sparse FP32)
      required += round_up(down_b_sparse_bytes, kAmxAlignment);      // down_lora_b (sparse FP32)
      required += round_up(grad_input_bytes, kAmxAlignment);
      if (need_grad_weights) {
        required += round_up(grad_weights_bytes, kAmxAlignment);
      }

      alloc_or_resize_backward_pool(i, required);

      auto* base = static_cast<uint8_t*>(backward_temp_pools_[i]);
      size_t offset = 0;
      auto slice = [&](size_t bytes) -> void* {
        if (bytes == 0) return nullptr;
        void* ptr = base + offset;
        offset += round_up(bytes, kAmxAlignment);
        return ptr;
      };

      // Sparse FP32 partials for reduce-type grads
      part_grad_gate_lora_a_[i] = (ggml_bf16_t*)slice(lora_a_sparse_bytes);  // reuse pointer, actually float*
      part_grad_up_lora_a_[i] = (ggml_bf16_t*)slice(lora_a_sparse_bytes);
      part_grad_down_lora_a_[i] = (ggml_bf16_t*)slice(down_b_sparse_bytes);  // reuse for down_lora_b FP32
      // Copy-type grads: no per-TP buffer needed
      part_grad_gate_lora_b_[i] = nullptr;
      part_grad_up_lora_b_[i] = nullptr;
      // grad_input and grad_weights: per-TP as before
      part_grad_input_[i] = (ggml_bf16_t*)slice(grad_input_bytes);
      part_grad_weights_[i] = need_grad_weights ? (float*)slice(grad_weights_bytes) : nullptr;
      clear_bytes[i] = offset;
    }

    // Parallel memset: zero only per-TP sparse partials and per-TP grad_input/grad_weights partials.
    // The caller is responsible for passing zero-initialized final grad tensors.
    struct ClearSeg {
      uint8_t* ptr;
      size_t len;
    };
    std::vector<ClearSeg> clear_segs;
    clear_segs.reserve((size_t)tp_count * 8);

    constexpr size_t kChunkBytes = 2 * 1024 * 1024;

    // Zero per-TP sparse partial pools
    for (int tp_idx = 0; tp_idx < tp_count; tp_idx++) {
      if (!backward_temp_pools_[tp_idx] || clear_bytes[tp_idx] == 0) continue;
      uint8_t* base = static_cast<uint8_t*>(backward_temp_pools_[tp_idx]);
      size_t total = clear_bytes[tp_idx];
      for (size_t off = 0; off < total; off += kChunkBytes) {
        size_t len = std::min(kChunkBytes, total - off);
        clear_segs.push_back(ClearSeg{base + off, len});
      }
    }

    pool->do_work_stealing_job((int)clear_segs.size(), nullptr,
                               [&](int seg_idx) {
                                 const auto& seg = clear_segs[(size_t)seg_idx];
                                 std::memset(seg.ptr, 0, seg.len);
                               },
                               nullptr);


    // Compute TP-slice pointers for copy-type direct writes
    // Each TP writes to its own I-slice of the final output tensor
    std::vector<ggml_bf16_t*> tp_gate_b_ptr(tp_count);
    std::vector<ggml_bf16_t*> tp_up_b_ptr(tp_count);
    std::vector<ggml_bf16_t*> tp_down_a_ptr(tp_count);
    std::vector<float*> tp_fp32_down_b(tp_count);
    std::vector<float*> tp_fp32_gate_a(tp_count);
    std::vector<float*> tp_fp32_up_a(tp_count);

    if constexpr (!kSkipLoRA) {
      int tp_offset = 0;
      for (int i = 0; i < tp_count; i++) {
        // Copy-type: pointer into final tensor at this TP's I-slice
        tp_gate_b_ptr[i] = (ggml_bf16_t*)grad_gate_lora_b + (size_t)tp_offset * lora_rank;
        tp_up_b_ptr[i] = (ggml_bf16_t*)grad_up_lora_b + (size_t)tp_offset * lora_rank;
        tp_down_a_ptr[i] = (ggml_bf16_t*)grad_down_lora_a + tp_offset;  // row-wise, offset added per-row

        // Reduce-type: sparse FP32 partials (reinterpret from part_grad pointers)
        tp_fp32_down_b[i] = (float*)part_grad_down_lora_a_[i];  // reused slot for down_lora_b FP32
        tp_fp32_gate_a[i] = (float*)part_grad_gate_lora_a_[i];
        tp_fp32_up_a[i] = (float*)part_grad_up_lora_a_[i];

        tp_offset += tp_configs[i].intermediate_size;
      }
    }

    // Run backward on each NUMA node
    pool->dispense_backend()->do_numa_job([&](int numa_id) {
      tps[numa_id]->backward(grad_output, part_grad_input_[numa_id],
                             // reduce-type: BF16 pointer unused (FP32 sparse used instead)
                             nullptr,                /* grad_gate_lora_a — unused, FP32 path below */
                             tp_gate_b_ptr[numa_id], /* copy-type: direct write to final tensor */
                             nullptr,                /* grad_up_lora_a — unused */
                             tp_up_b_ptr[numa_id],   /* copy-type: direct write */
                             tp_down_a_ptr[numa_id], /* copy-type: direct write */
                             nullptr,                /* grad_down_lora_b — unused, FP32 path below */
                             part_grad_weights_[numa_id], full_intermediate_size, tp_fp32_down_b[numa_id],
                             tp_fp32_gate_a[numa_id], tp_fp32_up_a[numa_id]);
    });

    // // Collect per-thread timing from all NUMA subpools
    // for (int i = 0; i < tp_count; i++) {
    // }

    // // Print per-thread backward timing

    // // Print expert token distribution for load balancing analysis
    // {
    //   std::vector<int> all_tokens;
    //   for (int i = 0; i < tp_count; i++) {
    //     const auto& tokens = tps[i]->get_expert_token_distribution();
    //     all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    //   }
    //   if (!all_tokens.empty()) {
    //     int max_t = *std::max_element(all_tokens.begin(), all_tokens.end());
    //     int min_t = *std::min_element(all_tokens.begin(), all_tokens.end());
    //     int sum_t = std::accumulate(all_tokens.begin(), all_tokens.end(), 0);
    //     fprintf(stderr, "  expert tokens (%zu): ", all_tokens.size());
    //     for (int t : all_tokens) fprintf(stderr, "%d ", t);
    //     fprintf(stderr, "(max=%d min=%d avg=%.1f)\n", max_t, min_t, (float)sum_t / all_tokens.size());
    //   }
    // }

    // Bug #22 fix: Merge grad_input from all NUMA nodes (sum them together)
    {
      auto* out = (ggml_bf16_t*)grad_input;
      pool->do_work_stealing_job(
          qlen, nullptr,
          [&](int token_id) {
            const ggml_bf16_t* src0 = part_grad_input_[0] + (size_t)token_id * hidden_size;
            const ggml_bf16_t* src1 = (tp_count > 1) ? (part_grad_input_[1] + (size_t)token_id * hidden_size) : nullptr;
            const ggml_bf16_t* src2 = (tp_count > 2) ? (part_grad_input_[2] + (size_t)token_id * hidden_size) : nullptr;
            const ggml_bf16_t* src3 = (tp_count > 3) ? (part_grad_input_[3] + (size_t)token_id * hidden_size) : nullptr;

            ggml_bf16_t* dst = out + (size_t)token_id * hidden_size;

            int h = 0;
            for (; h + 32 <= hidden_size; h += 32) {
              __m512 sum0, sum1;
              avx512_32xbf16_to_32xfp32((__m512i*)(src0 + h), &sum0, &sum1);
              if (src1) {
                __m512 x0, x1;
                avx512_32xbf16_to_32xfp32((__m512i*)(src1 + h), &x0, &x1);
                sum0 = _mm512_add_ps(sum0, x0);
                sum1 = _mm512_add_ps(sum1, x1);
              }
              if (src2) {
                __m512 x0, x1;
                avx512_32xbf16_to_32xfp32((__m512i*)(src2 + h), &x0, &x1);
                sum0 = _mm512_add_ps(sum0, x0);
                sum1 = _mm512_add_ps(sum1, x1);
              }
              if (src3) {
                __m512 x0, x1;
                avx512_32xbf16_to_32xfp32((__m512i*)(src3 + h), &x0, &x1);
                sum0 = _mm512_add_ps(sum0, x0);
                sum1 = _mm512_add_ps(sum1, x1);
              }
              avx512_32xfp32_to_32xbf16(&sum0, &sum1, (__m512i*)(dst + h));
            }
            for (; h < hidden_size; h++) {
              float sum = GGML_BF16_TO_FP32(src0[h]);
              if (src1) sum += GGML_BF16_TO_FP32(src1[h]);
              if (src2) sum += GGML_BF16_TO_FP32(src2[h]);
              if (src3) sum += GGML_BF16_TO_FP32(src3[h]);
              dst[h] = GGML_FP32_TO_BF16(sum);
            }
          },
          nullptr);
    }

    // Merge reduce-type LoRA gradients: sparse FP32 sum across TPs → BF16 final output
    // Copy-type grads (gate/up_lora_b, down_lora_a) were written directly — no merge needed.
    if constexpr (!kSkipLoRA) {
      // Sparse merge for gate_lora_a, up_lora_a: [active_count, r, H] FP32 → [E, r, H] BF16
      {
        const int sparse_rows = active_count * lora_rank;  // e.g. 10*8=80 vs 4096
        auto* out_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
        auto* out_up_a = (ggml_bf16_t*)grad_up_lora_a;
        pool->do_work_stealing_job(
            sparse_rows, nullptr,
            [&](int sparse_row_id) {
              int task = sparse_row_id / lora_rank;
              int r = sparse_row_id % lora_rank;
              int expert_idx = active_expert_map[task];
              size_t src_base = ((size_t)task * lora_rank + r) * hidden_size;
              size_t dst_base = ((size_t)expert_idx * lora_rank + r) * hidden_size;

              ggml_bf16_t* gd = out_gate_a + dst_base;
              ggml_bf16_t* ud = out_up_a + dst_base;

              int h = 0;
              for (; h + 32 <= hidden_size; h += 32) {
                __m512 gs0 = _mm512_loadu_ps((const float*)tp_fp32_gate_a[0] + src_base + h);
                __m512 gs1 = _mm512_loadu_ps((const float*)tp_fp32_gate_a[0] + src_base + h + 16);
                __m512 us0 = _mm512_loadu_ps((const float*)tp_fp32_up_a[0] + src_base + h);
                __m512 us1 = _mm512_loadu_ps((const float*)tp_fp32_up_a[0] + src_base + h + 16);
                for (int tp = 1; tp < tp_count; tp++) {
                  gs0 = _mm512_add_ps(gs0, _mm512_loadu_ps((const float*)tp_fp32_gate_a[tp] + src_base + h));
                  gs1 = _mm512_add_ps(gs1, _mm512_loadu_ps((const float*)tp_fp32_gate_a[tp] + src_base + h + 16));
                  us0 = _mm512_add_ps(us0, _mm512_loadu_ps((const float*)tp_fp32_up_a[tp] + src_base + h));
                  us1 = _mm512_add_ps(us1, _mm512_loadu_ps((const float*)tp_fp32_up_a[tp] + src_base + h + 16));
                }
                avx512_32xfp32_to_32xbf16(&gs0, &gs1, (__m512i*)(gd + h));
                avx512_32xfp32_to_32xbf16(&us0, &us1, (__m512i*)(ud + h));
              }
              for (; h < hidden_size; h++) {
                float gs = ((const float*)tp_fp32_gate_a[0])[src_base + h];
                float us = ((const float*)tp_fp32_up_a[0])[src_base + h];
                for (int tp = 1; tp < tp_count; tp++) {
                  gs += ((const float*)tp_fp32_gate_a[tp])[src_base + h];
                  us += ((const float*)tp_fp32_up_a[tp])[src_base + h];
                }
                gd[h] = GGML_FP32_TO_BF16(gs);
                ud[h] = GGML_FP32_TO_BF16(us);
              }
            },
            nullptr);
      }

      // Sparse merge for down_lora_b: [active_count, H, r] FP32 → [E, H, r] BF16
      {
        const int sparse_rows = active_count;  // one task per active expert
        auto* out_down_b = (ggml_bf16_t*)grad_down_lora_b;
        pool->do_work_stealing_job(
            sparse_rows, nullptr,
            [&](int task) {
              int expert_idx = active_expert_map[task];
              size_t src_expert_base = (size_t)task * hidden_size * lora_rank;
              size_t dst_expert_base = (size_t)expert_idx * hidden_size * lora_rank;

              for (int hh = 0; hh < hidden_size; hh++) {
                size_t src_row = src_expert_base + (size_t)hh * lora_rank;
                size_t dst_row = dst_expert_base + (size_t)hh * lora_rank;
                for (int r = 0; r < lora_rank; r++) {
                  float sum = ((const float*)tp_fp32_down_b[0])[src_row + r];
                  for (int tp = 1; tp < tp_count; tp++) {
                    sum += ((const float*)tp_fp32_down_b[tp])[src_row + r];
                  }
                  out_down_b[dst_row + r] = GGML_FP32_TO_BF16(sum);
                }
              }
            },
            nullptr);
      }
    }  // if constexpr (!kSkipLoRA)

    // Merge grad_weights from all NUMA nodes (sum them together)
    // Each NUMA computes partial grad_weights based on its down_output partition
    if (grad_weights != nullptr) {
      float* out_grad_weights = (float*)grad_weights;
      const size_t total = (size_t)qlen * (size_t)k;
      constexpr size_t kBlock = 4096;
      const int tasks = (int)((total + kBlock - 1) / kBlock);
      pool->do_work_stealing_job(
          tasks, nullptr,
          [&](int task_id) {
            const size_t begin = (size_t)task_id * kBlock;
            size_t end = begin + kBlock;
            if (end > total) end = total;

            const float* s0 = part_grad_weights_[0];
            const float* s1 = (tp_count > 1) ? part_grad_weights_[1] : nullptr;
            const float* s2 = (tp_count > 2) ? part_grad_weights_[2] : nullptr;
            const float* s3 = (tp_count > 3) ? part_grad_weights_[3] : nullptr;

            size_t i = begin;
            for (; i + 16 <= end; i += 16) {
              __m512 v = _mm512_loadu_ps(s0 + i);
              if (s1) v = _mm512_add_ps(v, _mm512_loadu_ps(s1 + i));
              if (s2) v = _mm512_add_ps(v, _mm512_loadu_ps(s2 + i));
              if (s3) v = _mm512_add_ps(v, _mm512_loadu_ps(s3 + i));
              _mm512_storeu_ps(out_grad_weights + i, v);
            }
            for (; i < end; i++) {
              float sum = s0[i];
              if (s1) sum += s1[i];
              if (s2) sum += s2[i];
              if (s3) sum += s3[i];
              out_grad_weights[i] = sum;
            }
          },
          nullptr);
    }

    pool->dispense_backend()->do_numa_job([&](int numa_id) {
    });
  }

  /**
   * @brief Python binding for backward.
   */
  void backward_binding(intptr_t grad_output, intptr_t grad_input, intptr_t grad_gate_lora_a, intptr_t grad_gate_lora_b,
                        intptr_t grad_up_lora_a, intptr_t grad_up_lora_b, intptr_t grad_down_lora_a,
                        intptr_t grad_down_lora_b, intptr_t grad_weights) {
    backward((const void*)grad_output, (void*)grad_input, (void*)grad_gate_lora_a, (void*)grad_gate_lora_b,
             (void*)grad_up_lora_a, (void*)grad_up_lora_b, (void*)grad_down_lora_a, (void*)grad_down_lora_b,
             (void*)grad_weights);
  }

  /**
   * @brief Update LoRA weight pointers on all NUMA nodes.
   *
   * Bug #19 fix: LoRA weights containing intermediate_size dimension need to be partitioned
   * for TP mode, similar to how Bug #8 fixed base weight partitioning.
   *
   * Weights requiring partitioning (contain intermediate_size dimension):
   * - gate_lora_b: [expert_num, intermediate_size, lora_rank] -> slice by intermediate_size
   * - up_lora_b:   [expert_num, intermediate_size, lora_rank] -> slice by intermediate_size
   * - down_lora_a: [expert_num, lora_rank, intermediate_size] -> slice by intermediate_size (row-wise)
   *
   * Weights NOT requiring partitioning:
   * - gate_lora_a: [expert_num, lora_rank, hidden_size]
   * - up_lora_a:   [expert_num, lora_rank, hidden_size]
   * - down_lora_b: [expert_num, hidden_size, lora_rank]
   */
  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    if constexpr (kSkipLoRA) return;  // No LoRA weights to update in SkipLoRA mode
    int full_intermediate_size = sft_config.intermediate_size;
    int expert_num = config.expert_num;
    int lora_rank = sft_config.lora_rank;

    // Allocate partitioned weight buffers on first call
    if (partitioned_gate_lora_b_.empty()) {
      partitioned_gate_lora_b_.resize(tp_count, nullptr);
      partitioned_up_lora_b_.resize(tp_count, nullptr);
      partitioned_down_lora_a_.resize(tp_count, nullptr);
      for (int i = 0; i < tp_count; i++) {
        int tp_inter = tp_configs[i].intermediate_size;
        size_t lora_b_size = (size_t)expert_num * tp_inter * lora_rank;
        partitioned_gate_lora_b_[i] = new ggml_bf16_t[lora_b_size];
        partitioned_up_lora_b_[i] = new ggml_bf16_t[lora_b_size];
        partitioned_down_lora_a_[i] = new ggml_bf16_t[expert_num * lora_rank * tp_inter];
      }
    }

    // Single do_numa_job: work-stealing memcpy + update_lora_weights
    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a,
                                           down_lora_b, full_intermediate_size, expert_num, lora_rank,
                                           pool](int numa_id) {
      int tp_inter = tp_configs[numa_id].intermediate_size;
      size_t lora_b_slice = (size_t)tp_inter * lora_rank;
      auto subpool = pool->get_subpool(numa_id);

      // Work-stealing: copy all weights for this expert (gate + up + down)
      subpool->do_work_stealing_job(
          expert_num,
          [&](int e) {
            // gate_lora_b: [expert_num, intermediate_size, lora_rank]
            memcpy(partitioned_gate_lora_b_[numa_id] + e * lora_b_slice,
                   (ggml_bf16_t*)gate_lora_b + e * full_intermediate_size * lora_rank + numa_id * lora_b_slice,
                   sizeof(ggml_bf16_t) * lora_b_slice);

            // up_lora_b: [expert_num, intermediate_size, lora_rank]
            memcpy(partitioned_up_lora_b_[numa_id] + e * lora_b_slice,
                   (ggml_bf16_t*)up_lora_b + e * full_intermediate_size * lora_rank + numa_id * lora_b_slice,
                   sizeof(ggml_bf16_t) * lora_b_slice);

            // down_lora_a: [expert_num, lora_rank, intermediate_size] - row-wise slice
            for (int r = 0; r < lora_rank; r++) {
              memcpy(partitioned_down_lora_a_[numa_id] + e * lora_rank * tp_inter + r * tp_inter,
                     (ggml_bf16_t*)down_lora_a + e * lora_rank * full_intermediate_size + r * full_intermediate_size +
                         numa_id * tp_inter,
                     sizeof(ggml_bf16_t) * tp_inter);
            }
          });

      // Update weights after all memcpy complete
      tps[numa_id]->update_lora_weights(gate_lora_a, partitioned_gate_lora_b_[numa_id], up_lora_a,
                                        partitioned_up_lora_b_[numa_id], partitioned_down_lora_a_[numa_id],
                                        down_lora_b);
    });
  }

  /**
   * @brief Free previously allocated partitioned LoRA weights.
   */
  void free_partitioned_lora_weights() {
    for (auto ptr : partitioned_gate_lora_b_) {
      if (ptr) delete[] ptr;
    }
    for (auto ptr : partitioned_up_lora_b_) {
      if (ptr) delete[] ptr;
    }
    for (auto ptr : partitioned_down_lora_a_) {
      if (ptr) delete[] ptr;
    }
    partitioned_gate_lora_b_.clear();
    partitioned_up_lora_b_.clear();
    partitioned_down_lora_a_.clear();
  }

  /**
   * @brief Free previously allocated partitioned base weights.
   * Bug #20 fix: These are needed for backward pass and must not be freed in load_weights().
   */
  void free_partitioned_base_weights() {
    for (auto ptr : partitioned_gate_proj_) {
      if (ptr) delete[] ptr;
    }
    for (auto ptr : partitioned_up_proj_) {
      if (ptr) delete[] ptr;
    }
    for (auto ptr : partitioned_down_proj_) {
      if (ptr) delete[] ptr;
    }
    partitioned_gate_proj_.clear();
    partitioned_up_proj_.clear();
    partitioned_down_proj_.clear();
  }

  /**
   * @brief Prepare backward weights from BF16 tensors and save to disk.
   * @param gate BF16 gate_proj pointer [expert_num, intermediate_size, hidden_size]
   * @param up BF16 up_proj pointer
   * @param down BF16 down_proj pointer
   * @param path Output directory path
   */
  void prepare_and_save_bwd(void* gate, void* up, void* down, const std::string& path) {
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    for (int i = 0; i < tp_count; i++) {
      auto& tpc = tp_configs[i];
      size_t gate_up_elcount = (size_t)tpc.intermediate_size * tpc.hidden_size;

      ggml_bf16_t* temp_gate = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      ggml_bf16_t* temp_up = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      ggml_bf16_t* temp_down = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, i, gate_up_elcount](int expert_id_) {
            size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

            size_t src_gate_offset = expert_id * config.intermediate_size * config.hidden_size + i * gate_up_elcount;
            size_t dst_offset = expert_id * gate_up_elcount;
            size_t copy_bytes = sizeof(ggml_bf16_t) * gate_up_elcount;

            memcpy(temp_gate + dst_offset, (ggml_bf16_t*)gate + src_gate_offset, copy_bytes);
            memcpy(temp_up + dst_offset, (ggml_bf16_t*)up + src_gate_offset, copy_bytes);

            for (size_t col = 0; col < config.hidden_size; col++) {
              memcpy(temp_down + expert_id * tpc.hidden_size * tpc.intermediate_size + col * tpc.intermediate_size,
                     (ggml_bf16_t*)down + expert_id * config.intermediate_size * config.hidden_size +
                         col * config.intermediate_size + i * tpc.intermediate_size,
                     sizeof(ggml_bf16_t) * tpc.intermediate_size);
            }
          },
          nullptr);

      tps[i]->prepare_bwd(temp_gate, temp_up, temp_down);

      std::filesystem::path prefix =
          std::filesystem::path(path) / ("_layer_" + std::to_string(config.layer_idx)) / ("_numa_" + std::to_string(i));
      tps[i]->save_backward_weights(prefix);

      delete[] temp_gate;
      delete[] temp_up;
      delete[] temp_down;
    }
  }

  /**
   * @brief Submit async backward weight repack for this layer (non-blocking).
   * Launches a worker thread that repacks backward BB from forward weights across all NUMA nodes.
   * Called from Python after MoE backward completes, to overlap repack with GPU attention backward.
   */
  void submit_backward_repack() {
    if (!config.share_backward_bb) return;

    // Join any previous repack first
    if (repack_thread_.joinable()) repack_thread_.join();

    repack_in_flight_.store(true, std::memory_order_release);
    repack_thread_ = std::thread([this]() {
      config.pool->dispense_backend()->do_numa_job(
          [this](int numa_id) { tps[numa_id]->prepare_backward_bb_for_async(); });
      repack_in_flight_.store(false, std::memory_order_release);
    });
  }

  /**
   * @brief Wait for async backward weight repack to complete (blocking).
   * Must be called before any operation that uses the CPU thread pool (e.g., checkpoint recompute).
   */
  void wait_backward_repack() {
    if (repack_thread_.joinable()) {
      repack_thread_.join();
    }
  }

  /**
   * @brief Destructor - free partitioned weights.
   */
  ~TP_MOE_SFT() {
    wait_backward_repack();
    free_backward_temp_pools();
    free_partitioned_lora_weights();
    free_partitioned_base_weights();
  }

  void update_lora_weights_binding(intptr_t gate_lora_a, intptr_t gate_lora_b, intptr_t up_lora_a, intptr_t up_lora_b,
                                   intptr_t down_lora_a, intptr_t down_lora_b) {
    update_lora_weights((void*)gate_lora_a, (void*)gate_lora_b, (void*)up_lora_a, (void*)up_lora_b, (void*)down_lora_a,
                        (void*)down_lora_b);
  }
};

#endif  // CPUINFER_OPERATOR_MOE_SFT_TP_HPP
