/**
 * @Description  : AMX MoE SFT (Supervised Fine-Tuning) implementation with LoRA support.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_SFT_MOE_H
#define CPUINFER_OPERATOR_AMX_SFT_MOE_H

#include <cmath>
#include <stdexcept>
#include <vector>

#include "moe.hpp"

/**
 * @brief Forward cache structure for gradient checkpointing.
 *
 * Stores intermediate values from forward pass needed for backward computation.
 * Supports multiple cache slots for gradient checkpointing (multiple forwards before backward).
 */
struct ForwardCache {
  // Intermediate values (need to be copied as next layer's forward will overwrite)
  ggml_bf16_t* input_cache = nullptr;         // [qlen, hidden_size]
  ggml_bf16_t* gate_output_cache = nullptr;   // [tokens_total, intermediate_size]
  ggml_bf16_t* up_output_cache = nullptr;     // [tokens_total, intermediate_size]
  ggml_bf16_t* intermediate_cache = nullptr;  // [tokens_total, intermediate_size] (after activation)

  // Routing information
  std::vector<int64_t> expert_ids_cache;
  std::vector<float> weights_cache;
  std::vector<int> m_local_num_cache;
  std::vector<std::vector<int>> m_local_pos_cache;
  std::vector<int> m_expert_id_map_cache;
  int qlen_cache = 0;
  int k_cache = 0;
  int activated_expert_cache = 0;

  bool valid = false;
};

/**
 * @brief AMX SFT MoE implementation with LoRA support.
 *
 * Inherits from AMX_MOE_TP and adds:
 * - LoRA computation for gate/up/down projections
 * - Forward cache for gradient checkpointing
 * - Backward pass implementation
 *
 * @tparam T The GEMM kernel type (e.g., GemmKernel224BF, GemmKernel224Int8)
 */
template <class T>
class AMX_SFT_MOE_TP : public AMX_MOE_TP<T> {
 private:
  using Base = AMX_MOE_TP<T>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_expert_id_map_;
  using Base::m_local_down_output_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_gate_output_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_input_;
  using Base::m_local_input_ptr_;
  using Base::m_local_num_;
  using Base::m_local_pos_;
  using Base::m_local_up_output_;
  using Base::m_local_up_output_ptr_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

  // SFT configuration
  MOESFTConfig sft_config_;

  // LoRA configuration (from MOESFTConfig)
  int lora_rank_;
  float lora_scaling_;

  // LoRA weight pointers (directly pointing to Python tensors)
  ggml_bf16_t* gate_lora_a_;  // [expert_num, lora_rank, hidden_size]
  ggml_bf16_t* gate_lora_b_;  // [expert_num, intermediate_size, lora_rank]
  ggml_bf16_t* up_lora_a_;
  ggml_bf16_t* up_lora_b_;
  ggml_bf16_t* down_lora_a_;
  ggml_bf16_t* down_lora_b_;

  // LoRA intermediate buffer (using shared_mem_buffer pool allocation)
  // For lora_A @ x results
  ggml_bf16_t* lora_intermediate_;  // [max_len * k, lora_rank]
  void* lora_intermediate_pool_;
  size_t lora_intermediate_pool_bytes_;

  // Forward cache stack (for gradient checkpointing)
  std::vector<ForwardCache> cache_stack_;
  int cache_stack_top_ = 0;  // Stack top pointer
  int max_cache_depth_;

  // Cache buffer pools
  void* cache_input_pool_ = nullptr;
  void* cache_gate_output_pool_ = nullptr;
  void* cache_up_output_pool_ = nullptr;
  void* cache_intermediate_pool_ = nullptr;
  size_t cache_slot_bytes_input_;
  size_t cache_slot_bytes_intermediate_;

  // Gradient intermediate buffers
  ggml_bf16_t* grad_intermediate_;  // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_gate_output_;   // [max_len * k, intermediate_size]
  ggml_bf16_t* grad_up_output_;     // [max_len * k, intermediate_size]
  void* grad_intermediate_pool_;
  void* grad_gate_output_pool_;
  void* grad_up_output_pool_;

 public:
  AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    printf("Creating AMX_SFT_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));

    // Initialize LoRA configuration
    lora_rank_ = config.lora_rank;
    lora_scaling_ = config.lora_scaling();
    max_cache_depth_ = config.max_cache_depth;

    // Get LoRA weight pointers
    gate_lora_a_ = (ggml_bf16_t*)config.gate_lora_a;
    gate_lora_b_ = (ggml_bf16_t*)config.gate_lora_b;
    up_lora_a_ = (ggml_bf16_t*)config.up_lora_a;
    up_lora_b_ = (ggml_bf16_t*)config.up_lora_b;
    down_lora_a_ = (ggml_bf16_t*)config.down_lora_a;
    down_lora_b_ = (ggml_bf16_t*)config.down_lora_b;

    // Initialize buffers
    init_lora_buffers();
    init_cache_buffers();
    init_grad_buffers();
  }

  ~AMX_SFT_MOE_TP() = default;

  /**
   * @brief SFT Forward pass with optional caching for backward.
   *
   * Computes: output = Σ weights[i] * down_proj(silu(gate_proj(x) + gate_lora(x)) * (up_proj(x) + up_lora(x))) +
   * down_lora(...)
   *
   * @param qlen Number of tokens
   * @param k Number of experts per token
   * @param expert_ids Expert indices [qlen, k]
   * @param weights Expert weights [qlen, k]
   * @param input Input tensor [qlen, hidden_size]
   * @param output Output tensor [qlen, hidden_size]
   * @param save_for_backward Whether to save intermediate values for backward pass
   */
  void forward_sft(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
                   bool save_for_backward) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Step 1: Expert routing (reuse base class logic)
    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    // Step 2: Buffer pool allocation (reuse base class logic)
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = Base::gate_up_ba_pool_;
    void* gate_bc_pool_ptr = Base::gate_bc_pool_;
    void* up_bc_pool_ptr = Base::up_bc_pool_;
    void* down_ba_pool_ptr = Base::down_ba_pool_;
    void* down_bc_pool_ptr = Base::down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) {
        continue;
      }

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr =
          (void*)((uintptr_t)gate_up_ba_pool_ptr + align64(Base::buffer_a_required_size(max_m, config_.hidden_size)));

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr +
                                 align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr =
          (void*)((uintptr_t)up_bc_pool_ptr + align64(Base::buffer_c_required_size(max_m, config_.intermediate_size)));

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr +
                                 align64(Base::buffer_a_required_size(max_m, config_.intermediate_size)));

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr =
          (void*)((uintptr_t)down_bc_pool_ptr + align64(Base::buffer_c_required_size(max_m, config_.hidden_size)));
    }

    // Step 3: Copy input to expert buffers
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

    // Step 4: Quantize input
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

    // Step 5: Gate + Up GEMM (base projection)
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
          }
        },
        nullptr);

    // Step 5.5: Gate + Up LoRA (NEW)
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      compute_lora_gate_up(qlen, activated_expert);
    }

    // Save gate/up outputs before activation (for backward)
    if (save_for_backward) {
      ForwardCache& cache = push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert);
    }

    // Step 6: Activation (silu(gate) * up)
    Base::apply_activation(activated_expert, nth, qlen);

    // Step 7: Quantize intermediate for down projection
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // Step 8: Down GEMM
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          this->do_down_gemm(expert_idx, ith, nth, qlen);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // Step 8.5: Down LoRA (NEW)
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      compute_lora_down(qlen, activated_expert);
    }

    // Step 9: Weighted merge
    pool->do_work_stealing_job(
        qlen, nullptr,
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i*)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                   m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
          }
        },
        nullptr);
  }

  /**
   * @brief Backward pass for SFT.
   *
   * Computes gradients for LoRA weights using cached intermediate values.
   *
   * @param grad_output Gradient of loss w.r.t. output [qlen, hidden_size]
   * @param grad_input Gradient of loss w.r.t. input [qlen, hidden_size] (output)
   * @param grad_gate_lora_a Gradient for gate LoRA A [expert_num, lora_rank, hidden_size]
   * @param grad_gate_lora_b Gradient for gate LoRA B [expert_num, intermediate_size, lora_rank]
   * @param grad_up_lora_a Gradient for up LoRA A
   * @param grad_up_lora_b Gradient for up LoRA B
   * @param grad_down_lora_a Gradient for down LoRA A
   * @param grad_down_lora_b Gradient for down LoRA B
   */
  void backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b) {
    // Pop cache from stack
    ForwardCache cache = pop_cache();
    if (!cache.valid) {
      throw std::runtime_error("No valid forward cache for backward");
    }

    int qlen = cache.qlen_cache;
    int k = cache.k_cache;
    int activated_expert = cache.activated_expert_cache;

    // Restore routing information
    m_local_num_ = cache.m_local_num_cache;
    m_local_pos_ = cache.m_local_pos_cache;
    m_expert_id_map_ = cache.m_expert_id_map_cache;

    // Recompute pointer offsets
    size_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

    // Step 1: Down projection backward
    backward_down(cache, grad_output, grad_down_lora_a, grad_down_lora_b);

    // Step 2: Activation backward
    backward_activation(cache);

    // Step 3: Gate + Up projection backward
    backward_gate_up(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);

    // Mark cache as invalid
    cache.valid = false;
  }

  /**
   * @brief Update LoRA weight pointers (call when Python tensors are reallocated).
   */
  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    gate_lora_a_ = (ggml_bf16_t*)gate_lora_a;
    gate_lora_b_ = (ggml_bf16_t*)gate_lora_b;
    up_lora_a_ = (ggml_bf16_t*)up_lora_a;
    up_lora_b_ = (ggml_bf16_t*)up_lora_b;
    down_lora_a_ = (ggml_bf16_t*)down_lora_a;
    down_lora_b_ = (ggml_bf16_t*)down_lora_b;
  }

 private:
  void init_lora_buffers() {
    // Allocate LoRA intermediate buffer for (x @ lora_A^T) result
    // Size: max_len * num_experts_per_tok * lora_rank
    lora_intermediate_pool_bytes_ = sizeof(ggml_bf16_t) * config_.max_len * config_.num_experts_per_tok * lora_rank_;

    MemoryRequest mem_requests;
    mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    lora_intermediate_ = (ggml_bf16_t*)lora_intermediate_pool_;
  }

  void init_cache_buffers() {
    // Size of each cache slot
    cache_slot_bytes_input_ = config_.max_len * config_.hidden_size * sizeof(ggml_bf16_t);
    cache_slot_bytes_intermediate_ =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    MemoryRequest mem_requests;
    mem_requests.append_pointer(&cache_input_pool_, cache_slot_bytes_input_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_gate_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_up_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_intermediate_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    // Initialize cache stack
    cache_stack_.resize(max_cache_depth_);
    for (int i = 0; i < max_cache_depth_; i++) {
      cache_stack_[i].input_cache = (ggml_bf16_t*)cache_input_pool_ + i * config_.max_len * config_.hidden_size;
      cache_stack_[i].gate_output_cache = (ggml_bf16_t*)cache_gate_output_pool_ +
                                          i * config_.max_len * config_.num_experts_per_tok * config_.intermediate_size;
      cache_stack_[i].up_output_cache = (ggml_bf16_t*)cache_up_output_pool_ +
                                        i * config_.max_len * config_.num_experts_per_tok * config_.intermediate_size;
      cache_stack_[i].intermediate_cache = (ggml_bf16_t*)cache_intermediate_pool_ + i * config_.max_len *
                                                                                        config_.num_experts_per_tok *
                                                                                        config_.intermediate_size;
      cache_stack_[i].m_local_num_cache.resize(config_.expert_num);
      cache_stack_[i].m_local_pos_cache.resize(config_.max_len);
      for (int j = 0; j < config_.max_len; j++) {
        cache_stack_[i].m_local_pos_cache[j].resize(config_.num_experts_per_tok);
      }
      cache_stack_[i].m_expert_id_map_cache.resize(config_.expert_num);
    }
  }

  void init_grad_buffers() {
    size_t grad_buffer_bytes =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    MemoryRequest mem_requests;
    mem_requests.append_pointer(&grad_intermediate_pool_, grad_buffer_bytes);
    mem_requests.append_pointer(&grad_gate_output_pool_, grad_buffer_bytes);
    mem_requests.append_pointer(&grad_up_output_pool_, grad_buffer_bytes);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    grad_intermediate_ = (ggml_bf16_t*)grad_intermediate_pool_;
    grad_gate_output_ = (ggml_bf16_t*)grad_gate_output_pool_;
    grad_up_output_ = (ggml_bf16_t*)grad_up_output_pool_;
  }

  /**
   * @brief Compute LoRA for gate and up projections.
   *
   * gate_lora_out = (input @ gate_lora_A^T) @ gate_lora_B^T * scaling
   * gate_output += gate_lora_out
   * (similar for up)
   */
  void compute_lora_gate_up(int qlen, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,
        [this](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Get weight pointers
          ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          ggml_bf16_t* lora_b = do_up ? up_lora_b_ : gate_lora_b_;
          ggml_bf16_t* input = m_local_input_ptr_[expert_idx];
          ggml_bf16_t* output = do_up ? m_local_up_output_ptr_[expert_idx] : m_local_gate_output_ptr_[expert_idx];

          if (lora_a == nullptr || lora_b == nullptr) return;

          // Offset to current expert's weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
          size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          ggml_bf16_t* expert_lora_b = lora_b + lora_b_offset;

          // Step 1: intermediate = input @ lora_A^T
          // [num_tokens, hidden_size] @ [lora_rank, hidden_size]^T → [num_tokens, lora_rank]
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int h = 0; h < config_.hidden_size; h++) {
                float inp = GGML_BF16_TO_FP32(input[t * config_.hidden_size + h]);
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                sum += inp * w;
              }
              lora_intermediate_[t * lora_rank_ + r] = GGML_FP32_TO_BF16(sum);
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [intermediate_size, lora_rank]^T → [num_tokens, intermediate_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int i = 0; i < config_.intermediate_size; i++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = GGML_BF16_TO_FP32(lora_intermediate_[t * lora_rank_ + r]);
                float w = GGML_BF16_TO_FP32(expert_lora_b[i * lora_rank_ + r]);
                sum += inter * w;
              }
              // Add to output with scaling
              float out_val = GGML_BF16_TO_FP32(output[t * config_.intermediate_size + i]);
              out_val += sum * lora_scaling_;
              output[t * config_.intermediate_size + i] = GGML_FP32_TO_BF16(out_val);
            }
          }
        },
        nullptr);
  }

  /**
   * @brief Compute LoRA for down projection.
   */
  void compute_lora_down(int qlen, int activated_expert) {
    auto pool = config_.pool->get_subpool(tp_part_idx);

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          ggml_bf16_t* input = m_local_gate_output_ptr_[expert_idx];  // After activation
          ggml_bf16_t* output = m_local_down_output_ptr_[expert_idx];

          if (down_lora_a_ == nullptr || down_lora_b_ == nullptr) return;

          // Offset to current expert's weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
          size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
          ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

          // Step 1: intermediate = input @ lora_A^T
          // [num_tokens, intermediate_size] @ [lora_rank, intermediate_size]^T → [num_tokens, lora_rank]
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int i = 0; i < config_.intermediate_size; i++) {
                float inp = GGML_BF16_TO_FP32(input[t * config_.intermediate_size + i]);
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
                sum += inp * w;
              }
              lora_intermediate_[t * lora_rank_ + r] = GGML_FP32_TO_BF16(sum);
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [hidden_size, lora_rank]^T → [num_tokens, hidden_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int h = 0; h < config_.hidden_size; h++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = GGML_BF16_TO_FP32(lora_intermediate_[t * lora_rank_ + r]);
                float w = GGML_BF16_TO_FP32(expert_lora_b[h * lora_rank_ + r]);
                sum += inter * w;
              }
              // Add to output with scaling
              float out_val = GGML_BF16_TO_FP32(output[t * config_.hidden_size + h]);
              out_val += sum * lora_scaling_;
              output[t * config_.hidden_size + h] = GGML_FP32_TO_BF16(out_val);
            }
          }
        },
        nullptr);
  }

  ForwardCache& push_cache() {
    if (cache_stack_top_ >= max_cache_depth_) {
      throw std::runtime_error("Forward cache stack overflow");
    }
    return cache_stack_[cache_stack_top_++];
  }

  ForwardCache pop_cache() {
    if (cache_stack_top_ <= 0) {
      throw std::runtime_error("Forward cache stack underflow");
    }
    return cache_stack_[--cache_stack_top_];
  }

  void save_to_cache(ForwardCache& cache, int qlen, int k, const int64_t* expert_ids, const float* weights,
                     int activated_expert) {
    cache.qlen_cache = qlen;
    cache.k_cache = k;
    cache.activated_expert_cache = activated_expert;

    // Copy routing information
    cache.expert_ids_cache.resize(qlen * k);
    cache.weights_cache.resize(qlen * k);
    std::copy(expert_ids, expert_ids + qlen * k, cache.expert_ids_cache.begin());
    std::copy(weights, weights + qlen * k, cache.weights_cache.begin());

    cache.m_local_num_cache = m_local_num_;
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        cache.m_local_pos_cache[i][j] = m_local_pos_[i][j];
      }
    }
    for (int i = 0; i < activated_expert; i++) {
      cache.m_expert_id_map_cache[i] = m_expert_id_map_[i];
    }

    // Copy intermediate values
    size_t total_tokens = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      total_tokens += m_local_num_[expert_idx];
    }

    // Copy input (before any processing)
    memcpy(cache.input_cache, m_local_input_, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

    // Copy gate and up outputs (before activation)
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      int num_tokens = m_local_num_[expert_idx];
      memcpy(cache.gate_output_cache + offset * config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      memcpy(cache.up_output_cache + offset * config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      offset += num_tokens;
    }

    cache.valid = true;
  }

  void backward_down(const ForwardCache& cache, const void* grad_output, void* grad_down_lora_a,
                     void* grad_down_lora_b) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Initialize gradient intermediate buffer
    memset(grad_intermediate_, 0,
           config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));

    // Scatter grad_output to per-expert buffers and compute gradients
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, grad_output, grad_down_a, grad_down_b, qlen, k](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Collect gradients for this expert from grad_output
          // grad_output is [qlen, hidden_size], need to scatter based on routing
          std::vector<float> expert_grad_out(num_tokens * config_.hidden_size, 0.0f);

          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                float w = cache.weights_cache[i * k + j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  expert_grad_out[pos * config_.hidden_size + h] +=
                      ((float*)grad_output)[i * config_.hidden_size + h] * w;
                }
              }
            }
          }

          // Get cached intermediate (after activation)
          ggml_bf16_t* intermediate = cache.intermediate_cache;  // Will use gate_output_cache after activation saved

          // Compute grad w.r.t. intermediate: grad_intermediate = grad_output @ down_proj^T
          // For now, we only compute LoRA gradients

          if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
            // Get expert's LoRA weights
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
            size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
            ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

            // Gradient for LoRA B: grad_B = grad_output^T @ (intermediate @ lora_A^T) * scaling
            // = (grad_output^T @ intermediate @ lora_A^T) * scaling
            // Shape: [hidden_size, num_tokens] @ [num_tokens, lora_rank] → [hidden_size, lora_rank]

            // First compute intermediate @ lora_A^T → [num_tokens, lora_rank]
            std::vector<float> inter_proj(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  // Use cached gate_output after activation as intermediate
                  float inp =
                      GGML_BF16_TO_FP32(m_local_gate_output_ptr_[expert_idx][t * config_.intermediate_size + i]);
                  float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
                  sum += inp * w;
                }
                inter_proj[t * lora_rank_ + r] = sum;
              }
            }

            // grad_B = grad_output^T @ inter_proj * scaling
            // [hidden_size, num_tokens] @ [num_tokens, lora_rank] → [hidden_size, lora_rank]
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += expert_grad_out[t * config_.hidden_size + h] * inter_proj[t * lora_rank_ + r];
                }
                // Accumulate gradient
                float current = GGML_BF16_TO_FP32(grad_down_b[lora_b_offset + h * lora_rank_ + r]);
                grad_down_b[lora_b_offset + h * lora_rank_ + r] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }

            // Gradient for LoRA A: more complex, involves backprop through lora_B
            // grad_A = (lora_B^T @ grad_output^T @ intermediate)^T * scaling
            // = intermediate^T @ grad_output @ lora_B * scaling
            // Shape: [intermediate_size, num_tokens] @ [num_tokens, hidden_size] @ [hidden_size, lora_rank]
            //      = [intermediate_size, lora_rank]
            // First: grad_output @ lora_B → [num_tokens, lora_rank]
            std::vector<float> grad_times_b(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int h = 0; h < config_.hidden_size; h++) {
                  float g = expert_grad_out[t * config_.hidden_size + h];
                  float b = GGML_BF16_TO_FP32(expert_lora_b[h * lora_rank_ + r]);
                  sum += g * b;
                }
                grad_times_b[t * lora_rank_ + r] = sum;
              }
            }

            // grad_A = intermediate^T @ grad_times_b * scaling
            // [intermediate_size, num_tokens] @ [num_tokens, lora_rank] → [intermediate_size, lora_rank]
            // But A is stored as [lora_rank, intermediate_size], so we compute for that layout
            for (int r = 0; r < lora_rank_; r++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  float inter =
                      GGML_BF16_TO_FP32(m_local_gate_output_ptr_[expert_idx][t * config_.intermediate_size + i]);
                  sum += inter * grad_times_b[t * lora_rank_ + r];
                }
                float current = GGML_BF16_TO_FP32(grad_down_a[lora_a_offset + r * config_.intermediate_size + i]);
                grad_down_a[lora_a_offset + r * config_.intermediate_size + i] =
                    GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }
          }
        },
        nullptr);
  }

  void backward_activation(const ForwardCache& cache) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;

    // SiLU backward:
    // y = silu(gate) * up = gate * sigmoid(gate) * up
    // dy/d(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate))) * up
    // dy/d(up) = silu(gate) = gate * sigmoid(gate)

    size_t cache_offset = 0;
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, &cache_offset](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Get cached gate and up outputs (before activation)
          // Need to compute offset into cache
          size_t offset = 0;
          for (int i = 0; i < task_id; i++) {
            offset += m_local_num_[m_expert_id_map_[i]];
          }

          ggml_bf16_t* gate_output = cache.gate_output_cache + offset * config_.intermediate_size;
          ggml_bf16_t* up_output = cache.up_output_cache + offset * config_.intermediate_size;
          ggml_bf16_t* grad_inter = grad_intermediate_ + offset * config_.intermediate_size;
          ggml_bf16_t* grad_gate = grad_gate_output_ + offset * config_.intermediate_size;
          ggml_bf16_t* grad_up = grad_up_output_ + offset * config_.intermediate_size;

          for (int i = 0; i < num_tokens * config_.intermediate_size; i++) {
            float g = GGML_BF16_TO_FP32(gate_output[i]);
            float u = GGML_BF16_TO_FP32(up_output[i]);
            float sigmoid_g = 1.0f / (1.0f + expf(-g));
            float silu_g = g * sigmoid_g;

            float grad_i = GGML_BF16_TO_FP32(grad_inter[i]);

            // Compute gradients
            float grad_gate_val = grad_i * u * sigmoid_g * (1.0f + g * (1.0f - sigmoid_g));
            float grad_up_val = grad_i * silu_g;

            grad_gate[i] = GGML_FP32_TO_BF16(grad_gate_val);
            grad_up[i] = GGML_FP32_TO_BF16(grad_up_val);
          }
        },
        nullptr);
  }

  void backward_gate_up(const ForwardCache& cache, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                        void* grad_up_lora_a, void* grad_up_lora_b) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
    ggml_bf16_t* grad_gate_b = (ggml_bf16_t*)grad_gate_lora_b;
    ggml_bf16_t* grad_up_a = (ggml_bf16_t*)grad_up_lora_a;
    ggml_bf16_t* grad_up_b = (ggml_bf16_t*)grad_up_lora_b;

    // Initialize grad_input to zero
    memset(grad_input, 0, qlen * config_.hidden_size * sizeof(float));

    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,
        [this, &cache, grad_input, grad_gate_a, grad_gate_b, grad_up_a, grad_up_b, qlen, k](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Compute offset into gradient buffers
          size_t offset = 0;
          for (int i = 0; i < task_id / 2; i++) {
            offset += m_local_num_[m_expert_id_map_[i]];
          }

          ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                    : (grad_gate_output_ + offset * config_.intermediate_size);
          ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          ggml_bf16_t* lora_b = do_up ? up_lora_b_ : gate_lora_b_;
          ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
          ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;

          if (lora_a == nullptr || lora_b == nullptr) return;

          // Get cached input
          ggml_bf16_t* input = cache.input_cache;

          // Get expert's LoRA weights
          size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
          size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
          ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          ggml_bf16_t* expert_lora_b = lora_b + lora_b_offset;

          // Collect expert's input from cached input using routing
          std::vector<float> expert_input(num_tokens * config_.hidden_size, 0.0f);
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  expert_input[pos * config_.hidden_size + h] = GGML_BF16_TO_FP32(input[i * config_.hidden_size + h]);
                }
              }
            }
          }

          // Gradient for LoRA B: grad_B = grad^T @ (input @ lora_A^T) * scaling
          // First compute input @ lora_A^T → [num_tokens, lora_rank]
          std::vector<float> input_proj(num_tokens * lora_rank_, 0.0f);
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int h = 0; h < config_.hidden_size; h++) {
                float inp = expert_input[t * config_.hidden_size + h];
                float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                sum += inp * w;
              }
              input_proj[t * lora_rank_ + r] = sum;
            }
          }

          // grad_B = grad^T @ input_proj * scaling
          for (int i = 0; i < config_.intermediate_size; i++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int t = 0; t < num_tokens; t++) {
                float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                sum += g * input_proj[t * lora_rank_ + r];
              }
              float current = GGML_BF16_TO_FP32(grad_lora_b[lora_b_offset + i * lora_rank_ + r]);
              grad_lora_b[lora_b_offset + i * lora_rank_ + r] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
            }
          }

          // Gradient for LoRA A
          // First: grad @ lora_B → [num_tokens, lora_rank]
          std::vector<float> grad_times_b(num_tokens * lora_rank_, 0.0f);
          for (int t = 0; t < num_tokens; t++) {
            for (int r = 0; r < lora_rank_; r++) {
              float sum = 0.0f;
              for (int i = 0; i < config_.intermediate_size; i++) {
                float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                float b = GGML_BF16_TO_FP32(expert_lora_b[i * lora_rank_ + r]);
                sum += g * b;
              }
              grad_times_b[t * lora_rank_ + r] = sum;
            }
          }

          // grad_A = input^T @ grad_times_b * scaling
          for (int r = 0; r < lora_rank_; r++) {
            for (int h = 0; h < config_.hidden_size; h++) {
              float sum = 0.0f;
              for (int t = 0; t < num_tokens; t++) {
                sum += expert_input[t * config_.hidden_size + h] * grad_times_b[t * lora_rank_ + r];
              }
              float current = GGML_BF16_TO_FP32(grad_lora_a[lora_a_offset + r * config_.hidden_size + h]);
              grad_lora_a[lora_a_offset + r * config_.hidden_size + h] =
                  GGML_FP32_TO_BF16(current + sum * lora_scaling_);
            }
          }

          // Compute grad_input contribution from this expert's LoRA
          // grad_input += grad @ lora_B @ lora_A * scaling
          // = grad_times_b @ lora_A * scaling
          // Need to scatter back to original positions
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  float sum = 0.0f;
                  for (int r = 0; r < lora_rank_; r++) {
                    sum += grad_times_b[pos * lora_rank_ + r] *
                           GGML_BF16_TO_FP32(expert_lora_a[r * config_.hidden_size + h]);
                  }
                  ((float*)grad_input)[i * config_.hidden_size + h] += sum * lora_scaling_;
                }
              }
            }
          }
        },
        nullptr);
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_MOE_H
