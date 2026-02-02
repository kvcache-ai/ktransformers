/**
 * @Description  : TP (Tensor Parallel) wrapper for SFT MoE operations.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MOE_SFT_TP_HPP
#define CPUINFER_OPERATOR_MOE_SFT_TP_HPP

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>

#include "amx/la/amx.hpp"
#include "moe-tp.hpp"

// Dump utilities for TP backward debugging
namespace tp_dump {
inline bool is_dump_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char* env = getenv("SFT_MOE_DUMP");
    enabled = (env != nullptr && env[0] == '1') ? 1 : 0;
  }
  return enabled == 1;
}

inline const char* get_dump_dir() {
  static const char* dir = nullptr;
  if (dir == nullptr) {
    dir = getenv("SFT_MOE_DUMP_DIR");
    if (dir == nullptr) dir = "./cpp_dump";
  }
  return dir;
}

inline void dump_bf16_matrix(const ggml_bf16_t* data, int rows, int cols, const char* name, int tp_idx) {
  if (!is_dump_enabled()) return;
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s_tp%d.bin", get_dump_dir(), name, tp_idx);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return;
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  for (int i = 0; i < rows * cols; i++) {
    float val = GGML_BF16_TO_FP32(data[i]);
    file.write(reinterpret_cast<const char*>(&val), sizeof(float));
  }
}

inline void dump_bf16_matrix_final(const ggml_bf16_t* data, int rows, int cols, const char* name) {
  if (!is_dump_enabled()) return;
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s.bin", get_dump_dir(), name);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return;
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  for (int i = 0; i < rows * cols; i++) {
    float val = GGML_BF16_TO_FP32(data[i]);
    file.write(reinterpret_cast<const char*>(&val), sizeof(float));
  }
}

inline void dump_fp32_matrix(const float* data, int rows, int cols, const char* name, int tp_idx) {
  if (!is_dump_enabled()) return;
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s_tp%d.bin", get_dump_dir(), name, tp_idx);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return;
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  file.write(reinterpret_cast<const char*>(data), sizeof(float) * rows * cols);
}

inline void dump_fp32_matrix_final(const float* data, int rows, int cols, const char* name) {
  if (!is_dump_enabled()) return;
  char filename[256];
  snprintf(filename, sizeof(filename), "%s/%s.bin", get_dump_dir(), name);
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) return;
  file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
  file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
  file.write(reinterpret_cast<const char*>(data), sizeof(float) * rows * cols);
}
}  // namespace tp_dump

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

  TP_MOE_SFT(const MOESFTConfig& config) : Base(static_cast<const GeneralMOEConfig&>(config)), sft_config(config) {
    printf("Creating TP_MOE_SFT layer %d\n", config.layer_idx);

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

      // Also partition BF16 weights for backward gradient computation if available.
      // C++ backward needs BF16 base weights to compute gate/up LoRA B gradients
      // through the gated MLP chain (prepare_backward_weights checks config_.gate_proj).
      if (config.gate_proj != nullptr) {
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

                size_t src_gate_offset = expert_id * config.intermediate_size * config.hidden_size + i * gate_up_elcount;
                size_t dst_offset = expert_id * gate_up_elcount;
                size_t copy_bytes = sizeof(ggml_bf16_t) * gate_up_elcount;

                memcpy(temp_gate[i] + dst_offset,
                       (ggml_bf16_t*)config.gate_proj + src_gate_offset,
                       copy_bytes);

                memcpy(temp_up[i] + dst_offset,
                       (ggml_bf16_t*)config.up_proj + src_gate_offset,
                       copy_bytes);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy(temp_down[i] + expert_id * tpc.hidden_size * tpc.intermediate_size + col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr, "memcpy_weights_tmp");
        }

        // Set BF16 weight pointers on sub-MOEs for backward
        for (int i = 0; i < tp_count; i++) {
          tps[i]->set_base_weight_pointers(temp_gate[i], temp_up[i], temp_down[i]);
        }

        // free the memory
        for (int i = 0; i < tp_count; i++) {
          delete [] (temp_gate[i]);
          delete [] (temp_up[i]);
          delete [] (temp_down[i]);
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


      pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

      // Step 2: Set weight pointers via public methods and call load_weights
      for (int i = 0; i < tp_count; i++) {
        tps[i]->prepare_bwd(temp_gate[i], temp_up[i], temp_down[i]);
        tps[i]->set_physical_to_logical_map(config.physical_to_logical_map);
      }

      for (int i = 0; i < tp_count; i++) {
        delete [] (temp_gate[i]);
        delete [] (temp_up[i]);
        delete [] (temp_down[i]);
      }
    } else {
      // Other loading methods (from loader or file)
      pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
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
    // sft_timer::reset_forward();
    // Reset per-thread counters in each subpool (to accumulate all do_work_stealing_job calls)
    for (int i = 0; i < tp_count; i++) {
      pool->get_subpool(i)->reset_counters();
    }

    // Run forward on each NUMA node
    pool->dispense_backend()->do_numa_job([this, qlen, k, expert_ids, input, weights, save_for_backward](int numa_id) {
      tps[numa_id]->forward_sft(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id],
                                save_for_backward);
    });

    // // Collect per-thread timing from all NUMA subpools
    // for (int i = 0; i < tp_count; i++) {
    //   sft_timer::collect_forward(pool->get_subpool(i));
    // }

    // // Print per-thread forward timing
    // sft_timer::print_forward();

    // Merge results from all NUMA nodes
    this->merge_results(qlen, output);
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

    // Allocate partitioned gradient buffers for each NUMA
    std::vector<ggml_bf16_t*> part_grad_gate_lora_b(tp_count);
    std::vector<ggml_bf16_t*> part_grad_up_lora_b(tp_count);
    std::vector<ggml_bf16_t*> part_grad_down_lora_a(tp_count);

    // Bug #22 fix: Allocate separate grad_input buffers for each NUMA
    // Each NUMA's backward_gate_up does memset(grad_input, 0) first, which would
    // overwrite the other NUMA's results if they share the same buffer.
    std::vector<ggml_bf16_t*> part_grad_input(tp_count);

    // Allocate separate grad_weights buffers for each NUMA (need to sum across NUMAs)
    int k = sft_config.num_experts_per_tok;
    std::vector<float*> part_grad_weights(tp_count);

    for (int i = 0; i < tp_count; i++) {
      int tp_intermediate = tp_configs[i].intermediate_size;
      // Zero-initialize with ()
      part_grad_gate_lora_b[i] = new ggml_bf16_t[expert_num * tp_intermediate * lora_rank]();
      part_grad_up_lora_b[i] = new ggml_bf16_t[expert_num * tp_intermediate * lora_rank]();
      part_grad_down_lora_a[i] = new ggml_bf16_t[expert_num * lora_rank * tp_intermediate]();
      part_grad_input[i] = new ggml_bf16_t[qlen * hidden_size]();
      part_grad_weights[i] = grad_weights ? new float[qlen * k]() : nullptr;
    }

    // // Reset backward timing before computation
    // sft_timer::reset_backward();
    // // Reset per-thread counters in each subpool (to accumulate all do_work_stealing_job calls)
    // for (int i = 0; i < tp_count; i++) {
    //   pool->get_subpool(i)->reset_counters();
    // }

    // Run backward on each NUMA node with separate grad_input buffers
    pool->dispense_backend()->do_numa_job([this, grad_output, &part_grad_input, grad_gate_lora_a, grad_up_lora_a,
                                           grad_down_lora_b, &part_grad_gate_lora_b, &part_grad_up_lora_b,
                                           &part_grad_down_lora_a, &part_grad_weights](int numa_id) {
      tps[numa_id]->backward(grad_output, part_grad_input[numa_id], grad_gate_lora_a, part_grad_gate_lora_b[numa_id],
                             grad_up_lora_a, part_grad_up_lora_b[numa_id], part_grad_down_lora_a[numa_id],
                             grad_down_lora_b, part_grad_weights[numa_id]);
    });

    // // Collect per-thread timing from all NUMA subpools
    // for (int i = 0; i < tp_count; i++) {
    //   sft_timer::collect_backward(pool->get_subpool(i));
    // }

    // // Print per-thread backward timing
    // sft_timer::print_backward();

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

    // DUMP: per-TP grad_input before merge
    for (int i = 0; i < tp_count; i++) {
      tp_dump::dump_bf16_matrix(part_grad_input[i], qlen, hidden_size, "backward_grad_input", i);
    }

    // Bug #22 fix: Merge grad_input from all NUMA nodes (sum them together)
    {
      for (int i = 0; i < qlen * hidden_size; i++) {
        float sum = 0.0f;
        for (int numa_id = 0; numa_id < tp_count; numa_id++) {
          sum += GGML_BF16_TO_FP32(part_grad_input[numa_id][i]);
        }
        ((ggml_bf16_t*)grad_input)[i] = GGML_FP32_TO_BF16(sum);
      }
    }

    // DUMP: final merged grad_input
    tp_dump::dump_bf16_matrix_final((ggml_bf16_t*)grad_input, qlen, hidden_size, "backward_grad_input_final");

    // Merge partitioned gradients to full gradients
    for (int i = 0; i < tp_count; i++) {
      int tp_intermediate = tp_configs[i].intermediate_size;

      // grad_gate_lora_b/grad_up_lora_b: [expert_num, intermediate_size, lora_rank]
      // Contiguous block slice
      for (int expert_id = 0; expert_id < expert_num; expert_id++) {
        size_t src_offset = (size_t)expert_id * tp_intermediate * lora_rank;
        size_t dst_offset =
            (size_t)expert_id * full_intermediate_size * lora_rank + (size_t)i * tp_intermediate * lora_rank;
        memcpy((ggml_bf16_t*)grad_gate_lora_b + dst_offset, part_grad_gate_lora_b[i] + src_offset,
               tp_intermediate * lora_rank * sizeof(ggml_bf16_t));
        memcpy((ggml_bf16_t*)grad_up_lora_b + dst_offset, part_grad_up_lora_b[i] + src_offset,
               tp_intermediate * lora_rank * sizeof(ggml_bf16_t));
      }

      // grad_down_lora_a: [expert_num, lora_rank, intermediate_size]
      // Row-wise slice
      for (int expert_id = 0; expert_id < expert_num; expert_id++) {
        for (int r = 0; r < lora_rank; r++) {
          size_t src_offset = (size_t)expert_id * lora_rank * tp_intermediate + (size_t)r * tp_intermediate;
          size_t dst_offset = (size_t)expert_id * lora_rank * full_intermediate_size +
                              (size_t)r * full_intermediate_size + (size_t)i * tp_intermediate;
          memcpy((ggml_bf16_t*)grad_down_lora_a + dst_offset, part_grad_down_lora_a[i] + src_offset,
                 tp_intermediate * sizeof(ggml_bf16_t));
        }
      }
    }

    // Merge grad_weights from all NUMA nodes (sum them together)
    // Each NUMA computes partial grad_weights based on its down_output partition
    if (grad_weights != nullptr) {
      float* out_grad_weights = (float*)grad_weights;
      for (int i = 0; i < qlen * k; i++) {
        float sum = 0.0f;
        for (int numa_id = 0; numa_id < tp_count; numa_id++) {
          sum += part_grad_weights[numa_id][i];
        }
        out_grad_weights[i] = sum;
      }
    }

    // Clean up temporary buffers
    for (int i = 0; i < tp_count; i++) {
      delete[] part_grad_gate_lora_b[i];
      delete[] part_grad_up_lora_b[i];
      delete[] part_grad_down_lora_a[i];
      delete[] part_grad_input[i];
      if (part_grad_weights[i]) delete[] part_grad_weights[i];
    }
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
    // Free previously allocated partitioned weights
    free_partitioned_lora_weights();

    // Get full intermediate_size from original sft_config (before TP partitioning)
    int full_intermediate_size = sft_config.intermediate_size;
    int expert_num = config.expert_num;
    int lora_rank = sft_config.lora_rank;

    // size_t gate_a_elems = static_cast<size_t>(expert_num) * lora_rank * config.hidden_size;
    // size_t gate_b_elems = static_cast<size_t>(expert_num) * full_intermediate_size * lora_rank;
    // size_t up_a_elems = static_cast<size_t>(expert_num) * lora_rank * config.hidden_size;
    // size_t up_b_elems = static_cast<size_t>(expert_num) * full_intermediate_size * lora_rank;
    // size_t down_a_elems = static_cast<size_t>(expert_num) * lora_rank * full_intermediate_size;
    // size_t down_b_elems = static_cast<size_t>(expert_num) * config.hidden_size * lora_rank;

    // print_tp_bf16_stats(sft_config.layer_idx, "gate_lora_a", (ggml_bf16_t*)gate_lora_a, gate_a_elems);
    // print_tp_bf16_stats(sft_config.layer_idx, "gate_lora_b", (ggml_bf16_t*)gate_lora_b, gate_b_elems);
    // print_tp_bf16_stats(sft_config.layer_idx, "up_lora_a", (ggml_bf16_t*)up_lora_a, up_a_elems);
    // print_tp_bf16_stats(sft_config.layer_idx, "up_lora_b", (ggml_bf16_t*)up_lora_b, up_b_elems);
    // print_tp_bf16_stats(sft_config.layer_idx, "down_lora_a", (ggml_bf16_t*)down_lora_a, down_a_elems);
    // print_tp_bf16_stats(sft_config.layer_idx, "down_lora_b", (ggml_bf16_t*)down_lora_b, down_b_elems);

    // Initialize partition vectors
    partitioned_gate_lora_b_.resize(tp_count, nullptr);
    partitioned_up_lora_b_.resize(tp_count, nullptr);
    partitioned_down_lora_a_.resize(tp_count, nullptr);

    // Partition LoRA weights for each NUMA node
    for (int i = 0; i < tp_count; i++) {
      int tp_intermediate = tp_configs[i].intermediate_size;  // Partitioned size

      // gate_lora_b/up_lora_b: [expert_num, intermediate_size, lora_rank]
      // Slice contiguously by intermediate_size dimension
      size_t lora_b_slice_size = (size_t)tp_intermediate * lora_rank;
      partitioned_gate_lora_b_[i] = new ggml_bf16_t[expert_num * lora_b_slice_size];
      partitioned_up_lora_b_[i] = new ggml_bf16_t[expert_num * lora_b_slice_size];

      for (int expert_id = 0; expert_id < expert_num; expert_id++) {
        // Source offset: expert_id * full_intermediate * lora_rank + i * tp_intermediate * lora_rank
        memcpy(partitioned_gate_lora_b_[i] + expert_id * lora_b_slice_size,
               (ggml_bf16_t*)gate_lora_b + expert_id * full_intermediate_size * lora_rank + i * lora_b_slice_size,
               sizeof(ggml_bf16_t) * lora_b_slice_size);

        memcpy(partitioned_up_lora_b_[i] + expert_id * lora_b_slice_size,
               (ggml_bf16_t*)up_lora_b + expert_id * full_intermediate_size * lora_rank + i * lora_b_slice_size,
               sizeof(ggml_bf16_t) * lora_b_slice_size);
      }

      // down_lora_a: [expert_num, lora_rank, intermediate_size]
      // Need to slice row-wise by intermediate_size dimension
      partitioned_down_lora_a_[i] = new ggml_bf16_t[expert_num * lora_rank * tp_intermediate];

      for (int expert_id = 0; expert_id < expert_num; expert_id++) {
        for (int r = 0; r < lora_rank; r++) {
          // Source: expert_id * lora_rank * full_intermediate + r * full_intermediate + i * tp_intermediate
          memcpy(partitioned_down_lora_a_[i] + expert_id * lora_rank * tp_intermediate + r * tp_intermediate,
                 (ggml_bf16_t*)down_lora_a + expert_id * lora_rank * full_intermediate_size +
                     r * full_intermediate_size + i * tp_intermediate,
                 sizeof(ggml_bf16_t) * tp_intermediate);
        }
      }
    }

    // Update each NUMA node with partitioned weights
    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this, gate_lora_a, up_lora_a, down_lora_b](int numa_id) {
      tps[numa_id]->update_lora_weights(gate_lora_a,                        // Not partitioned
                                        partitioned_gate_lora_b_[numa_id],  // Partitioned
                                        up_lora_a,                          // Not partitioned
                                        partitioned_up_lora_b_[numa_id],    // Partitioned
                                        partitioned_down_lora_a_[numa_id],  // Partitioned
                                        down_lora_b);                       // Not partitioned
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
   * @brief Destructor - free partitioned weights.
   */
  ~TP_MOE_SFT() {
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
