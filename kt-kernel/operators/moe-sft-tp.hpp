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

#include "amx/la/amx.hpp"
#include "moe-tp.hpp"

// Forward declaration
template <class T>
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

  TP_MOE_SFT(MOESFTConfig config) : Base(static_cast<GeneralMOEConfig>(config)), sft_config(config) {
    printf("Creating TP_MOE_SFT layer %d\n", config.layer_idx);

    // Bug #16 fix: TP_MOE base class uses GeneralMOEConfig (object slicing) which loses
    // LoRA pointers. We need to propagate LoRA pointers to all NUMA node instances.
    if (config.gate_lora_a != nullptr) {
      update_lora_weights(config.gate_lora_a, config.gate_lora_b, config.up_lora_a, config.up_lora_b,
                          config.down_lora_a, config.down_lora_b);
    }
  }

  /**
   * @brief Load weights on all NUMA nodes.
   */
  void load_weights() override {
    auto pool = config.pool;
    pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
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

    // Run forward on each NUMA node
    pool->dispense_backend()->do_numa_job([this, qlen, k, expert_ids, input, weights, save_for_backward](int numa_id) {
      tps[numa_id]->forward_sft(qlen, k, expert_ids, weights, input, this->local_output_numa[numa_id],
                                save_for_backward);
    });

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
   * @brief Backward pass with NUMA distribution.
   *
   * Computes gradients for LoRA weights across all NUMA nodes.
   *
   * @param grad_output Gradient of loss w.r.t. output [qlen, hidden_size]
   * @param grad_input Gradient of loss w.r.t. input [qlen, hidden_size]
   * @param grad_gate_lora_a Gradient for gate LoRA A
   * @param grad_gate_lora_b Gradient for gate LoRA B
   * @param grad_up_lora_a Gradient for up LoRA A
   * @param grad_up_lora_b Gradient for up LoRA B
   * @param grad_down_lora_a Gradient for down LoRA A
   * @param grad_down_lora_b Gradient for down LoRA B
   */
  void backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b) {
    auto pool = config.pool;

    // Run backward on each NUMA node
    // Note: Each NUMA node computes partial gradients for the experts it handles
    // The gradients need to be accumulated across NUMA nodes by the caller
    pool->dispense_backend()->do_numa_job([this, grad_output, grad_input, grad_gate_lora_a, grad_gate_lora_b,
                                           grad_up_lora_a, grad_up_lora_b, grad_down_lora_a,
                                           grad_down_lora_b](int numa_id) {
      tps[numa_id]->backward(grad_output, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a,
                             grad_up_lora_b, grad_down_lora_a, grad_down_lora_b);
    });
  }

  /**
   * @brief Python binding for backward.
   */
  void backward_binding(intptr_t grad_output, intptr_t grad_input, intptr_t grad_gate_lora_a, intptr_t grad_gate_lora_b,
                        intptr_t grad_up_lora_a, intptr_t grad_up_lora_b, intptr_t grad_down_lora_a,
                        intptr_t grad_down_lora_b) {
    backward((const void*)grad_output, (void*)grad_input, (void*)grad_gate_lora_a, (void*)grad_gate_lora_b,
             (void*)grad_up_lora_a, (void*)grad_up_lora_b, (void*)grad_down_lora_a, (void*)grad_down_lora_b);
  }

  /**
   * @brief Update LoRA weight pointers on all NUMA nodes.
   */
  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    // Debug code for Bug #18 - commented out after fix verified
    // printf("[DEBUG TP update_lora_weights] tp_count=%d, gate_lora_a=%p\n", tp_count, gate_lora_a);
    auto pool = config.pool;
    // printf("[DEBUG TP update_lora_weights] before do_numa_job, pool=%p\n", (void*)pool);
    pool->dispense_backend()->do_numa_job(
        [this, gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b](int numa_id) {
          // printf("[DEBUG TP do_numa_job] numa_id=%d, calling tps[%d]->update_lora_weights\n", numa_id, numa_id);
          tps[numa_id]->update_lora_weights(gate_lora_a, gate_lora_b, up_lora_a, up_lora_b, down_lora_a, down_lora_b);
          // printf("[DEBUG TP do_numa_job] numa_id=%d done\n", numa_id);
        });
    // printf("[DEBUG TP update_lora_weights] after do_numa_job\n");
    // Verify pointers were updated
    // for (size_t i = 0; i < tps.size(); i++) {
    //   printf("[VERIFY] tps[%zu]->get_gate_lora_a() = %p (should equal %p)\n", i, tps[i]->get_gate_lora_a(),
    //   gate_lora_a);
    // }
  }

  void update_lora_weights_binding(intptr_t gate_lora_a, intptr_t gate_lora_b, intptr_t up_lora_a, intptr_t up_lora_b,
                                   intptr_t down_lora_a, intptr_t down_lora_b) {
    update_lora_weights((void*)gate_lora_a, (void*)gate_lora_b, (void*)up_lora_a, (void*)up_lora_b, (void*)down_lora_a,
                        (void*)down_lora_b);
  }
};

#endif  // CPUINFER_OPERATOR_MOE_SFT_TP_HPP
