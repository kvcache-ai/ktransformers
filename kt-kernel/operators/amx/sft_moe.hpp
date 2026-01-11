/**
 * @Description  : AMX MoE SFT (Supervised Fine-Tuning) implementation with LoRA support.
 * @Author       : lpl, Claude
 * @Date         : 2025-12-31
 * @Version      : 0.1.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_SFT_MOE_H
#define CPUINFER_OPERATOR_AMX_SFT_MOE_H

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "moe.hpp"

// =====================================================
// Backward timing macros for profiling
// Uses counter to skip warmup iterations and print at specific call
// Default: print at 8th call (warmup=5 + backward_test step 0,1,2)
// Can be overridden via SFT_MOE_PRINT_AT_CALL environment variable
// =====================================================
inline int get_print_at_call() {
  static int print_at = -1;
  if (print_at < 0) {
    const char* env = getenv("SFT_MOE_PRINT_AT_CALL");
    print_at = env ? atoi(env) : 8;  // Default: 8th call
  }
  return print_at;
}

#define BACKWARD_TIMER_START()                                       \
  static int _bwd_call_count = 0;                                    \
  _bwd_call_count++;                                                 \
  bool _bwd_should_print = (_bwd_call_count == get_print_at_call()); \
  auto _bwd_start = std::chrono::high_resolution_clock::now();       \
  auto _bwd_last = _bwd_start;

#define BACKWARD_TIMER_CHECKPOINT(name)                                                          \
  do {                                                                                           \
    auto _bwd_now = std::chrono::high_resolution_clock::now();                                   \
    if (_bwd_should_print) {                                                                     \
      double _elapsed = std::chrono::duration<double, std::milli>(_bwd_now - _bwd_last).count(); \
      double _total = std::chrono::duration<double, std::milli>(_bwd_now - _bwd_start).count();  \
      printf("[BWD TIMER] %s: %.3f ms (total: %.3f ms)\n", name, _elapsed, _total);              \
    }                                                                                            \
    _bwd_last = _bwd_now;                                                                        \
  } while (0)

#define BACKWARD_TIMER_END() (void)0

// =====================================================
// BUG-010: NaN Diagnostic Helper Functions
// =====================================================
struct NaNCheckResult {
  int nan_count = 0;
  int inf_count = 0;
  int first_nan_idx = -1;
  float first_nan_input_val = 0.0f;
};

// Check BF16 buffer for NaN/Inf
inline NaNCheckResult check_bf16_buffer_for_nan(const ggml_bf16_t* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    float val = GGML_BF16_TO_FP32(buf[i]);
    if (std::isnan(val)) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
    if (std::isinf(val)) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf("[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d\n", label, result.nan_count, result.inf_count,
           result.first_nan_idx);
  }
  return result;
}

// Check FP32 buffer for NaN/Inf
inline NaNCheckResult check_fp32_buffer_for_nan(const float* buf, int size, const char* label = nullptr) {
  NaNCheckResult result;
  for (int i = 0; i < size; i++) {
    if (std::isnan(buf[i])) {
      result.nan_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
    if (std::isinf(buf[i])) {
      result.inf_count++;
      if (result.first_nan_idx < 0) {
        result.first_nan_idx = i;
      }
    }
  }
  if (label && (result.nan_count > 0 || result.inf_count > 0)) {
    printf("[NaN TRACE] %s: nan_count=%d, inf_count=%d, first_idx=%d\n", label, result.nan_count, result.inf_count,
           result.first_nan_idx);
  }
  return result;
}

// =====================================================
// Type trait to detect if kernel supports standard mat_mul API
// Only these kernels have the standard amx::mat_mul(m,n,k,ba,bb,bc,ith,nth) overload
// KGroup kernels use mat_mul_kgroup() with different BufferB interface
// =====================================================
template <typename T>
struct supports_standard_mat_mul : std::false_type {};

template <>
struct supports_standard_mat_mul<amx::GemmKernel224BF> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int8> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4> : std::true_type {};
template <>
struct supports_standard_mat_mul<amx::GemmKernel224Int4_1> : std::true_type {};

template <typename T>
inline constexpr bool supports_standard_mat_mul_v = supports_standard_mat_mul<T>::value;

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
 * @tparam BaseMOE The base MOE class template (default: AMX_MOE_TP, can be AMX_AWQ_MOE_TP or AMX_K2_MOE_TP)
 */
template <class T, template <class> class BaseMOE = AMX_MOE_TP>
class AMX_SFT_MOE_TP : public BaseMOE<T> {
 protected:
  using Base = BaseMOE<T>;
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

 private:
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
  ggml_bf16_t* lora_intermediate_;  // [max_len * k, lora_rank] - kept for compatibility but not used
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

  // =====================================================
  // AMX-optimized LoRA GEMM buffers (performance optimization)
  // =====================================================

  // Padded lora_rank for AMX alignment (must be multiple of K_STEP=32)
  int padded_lora_rank_;

  // LoRA weight BufferB for AMX GEMM
  // Step 1 weights: lora_A matrices [padded_lora_rank, hidden_size or intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_a_bb_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_a_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> down_lora_a_bb_;  // [expert_num]

  // Step 2 weights: lora_B matrices [output_dim, padded_lora_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_b_bb_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_b_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>> down_lora_b_bb_;  // [expert_num]
  // Transposed weights for backward GEMM
  std::vector<std::shared_ptr<typename T::BufferB>> gate_lora_a_t_bb_;  // [expert_num] [hidden_size, padded_lora_rank]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_a_t_bb_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferB>>
      gate_lora_b_t_bb_;  // [expert_num] [padded_lora_rank, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> up_lora_b_t_bb_;  // [expert_num]

  // LoRA intermediate BufferA and BufferC
  // For step 1 output / step 2 input: [num_tokens, padded_lora_rank]
  // Gate and Up need SEPARATE buffers to avoid race condition in parallel execution
  std::vector<std::shared_ptr<typename T::BufferA>> lora_gate_intermediate_ba_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferA>> lora_up_intermediate_ba_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_intermediate_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_intermediate_bc_;    // [expert_num]

  // LoRA step 2 output BufferC (for accumulation before adding to main output)
  std::vector<std::shared_ptr<typename T::BufferC>> lora_gate_out_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_up_out_bc_;    // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> lora_down_out_bc_;  // [expert_num]

  // LoRA intermediate output pointers (for step 1 -> step 2)
  // Gate and Up need SEPARATE pointers to avoid race condition in parallel execution
  std::vector<ggml_bf16_t*> lora_gate_intermediate_ptr_;  // [expert_num]
  std::vector<ggml_bf16_t*> lora_up_intermediate_ptr_;    // [expert_num]

  // LoRA buffer pools
  void* lora_bb_pool_ = nullptr;                 // All LoRA weight BufferB
  void* lora_ba_pool_ = nullptr;                 // LoRA intermediate BufferA
  void* lora_bc_inter_pool_ = nullptr;           // LoRA step 1 output BufferC
  void* lora_bc_out_pool_ = nullptr;             // LoRA step 2 output BufferC
  void* lora_intermediate_bf16_pool_ = nullptr;  // BF16 intermediate for step 1->step 2

  // Buffer pool sizes
  size_t lora_bb_pool_bytes_ = 0;
  size_t lora_ba_pool_bytes_ = 0;
  size_t lora_bc_inter_pool_bytes_ = 0;
  size_t lora_bc_out_pool_bytes_ = 0;
  size_t lora_intermediate_bf16_pool_bytes_ = 0;

  // =====================================================
  // Backward pass AMX buffers
  // =====================================================

  // BufferA for grad_output (scattered to per-expert)
  std::vector<std::shared_ptr<typename T::BufferA>> grad_output_ba_;  // [expert_num]

  // BufferC for backward GEMM outputs
  std::vector<std::shared_ptr<typename T::BufferC>> grad_intermediate_bc_;  // [expert_num]
  std::vector<std::shared_ptr<typename T::BufferC>> grad_gate_up_bc_;       // [expert_num]

  // BF16 buffer for scattered grad_output (before quantization to BufferA)
  std::vector<ggml_bf16_t*> grad_output_bf16_ptr_;  // [expert_num]

  // Backward buffer pools
  void* backward_ba_pool_ = nullptr;
  void* backward_bc_pool_ = nullptr;
  void* grad_output_bf16_pool_ = nullptr;

  // Backward buffer pool sizes
  size_t backward_ba_pool_bytes_ = 0;
  size_t backward_bc_pool_bytes_ = 0;
  size_t grad_output_bf16_pool_bytes_ = 0;

  // =====================================================
  // Backward pass BufferB for transposed base weights
  // =====================================================
  // For backward GEMM, we need transposed versions of the base weights:
  // - Forward gate/up: input @ W^T uses gate_bb_[intermediate_size, hidden_size]
  // - Backward gate/up: grad @ W uses BufferB[hidden_size, intermediate_size]
  // - Forward down: intermediate @ W^T uses down_bb_[hidden_size, intermediate_size]
  // - Backward down: grad_output @ W uses BufferB[intermediate_size, hidden_size]
  std::vector<std::shared_ptr<typename T::BufferB>> gate_backward_bb_;  // [hidden_size, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> up_backward_bb_;    // [hidden_size, intermediate_size]
  std::vector<std::shared_ptr<typename T::BufferB>> down_backward_bb_;  // [intermediate_size, hidden_size]

  // Backward BufferB pool
  void* backward_bb_pool_ = nullptr;
  size_t backward_bb_pool_bytes_ = 0;

  // Flag to track if backward weights have been prepared
  bool backward_weights_prepared_ = false;

  // Flag to track if LoRA weights have been converted to BufferB format
  bool lora_weights_prepared_ = false;

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

    // Initialize all buffers in a single alloc() to avoid memory overlap
    // (Bug #15: SharedMemBuffer assigns all alloc() calls from same base address)
    init_all_buffers();
  }

  // Constructor to satisfy MOE_TP_PART concept (takes GeneralMOEConfig)
  AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}

  ~AMX_SFT_MOE_TP() {
    // Free LoRA buffers allocated with aligned_alloc
    if (lora_bb_pool_) free(lora_bb_pool_);
    if (lora_ba_pool_) free(lora_ba_pool_);
    if (lora_bc_inter_pool_) free(lora_bc_inter_pool_);
    if (lora_bc_out_pool_) free(lora_bc_out_pool_);
    if (lora_intermediate_bf16_pool_) free(lora_intermediate_bf16_pool_);
  }

  /**
   * @brief Set LoRA parameters after construction (Bug #007 fix).
   *
   * This is needed because TP_MOE base class uses GeneralMOEConfig which
   * doesn't have lora_rank/lora_alpha fields, causing object slicing.
   * The TP_MOE_SFT wrapper calls this method to propagate correct values.
   *
   * @param rank LoRA rank (typically 8 or 16)
   * @param alpha LoRA alpha for scaling (lora_scaling = alpha / rank)
   */
  void set_lora_params(int rank, float alpha) {
    lora_rank_ = rank;
    lora_scaling_ = alpha / rank;
  }

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

    // Step 5.5: Gate + Up LoRA
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      if constexpr (supports_standard_mat_mul_v<T>) {
        compute_lora_gate_up_amx(qlen, activated_expert);  // AMX-optimized path
      } else {
        compute_lora_gate_up(qlen, activated_expert);  // For-loop fallback for KGroup kernels
      }
    }

    // Save gate/up outputs before activation (for backward)
    if (save_for_backward) {
      ForwardCache& cache = push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert, input);
    }

    // Step 6: Activation (silu(gate) * up)
    Base::apply_activation(activated_expert, nth, qlen);

    // Save intermediate AFTER activation for backward_down (Bug #17c fix)
    if (save_for_backward) {
      ForwardCache& cache = cache_stack_[cache_stack_top_ - 1];  // Get the cache we just pushed
      save_intermediate_to_cache(cache, activated_expert);
    }

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

    // Step 8.5: Down LoRA
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      if constexpr (supports_standard_mat_mul_v<T>) {
        compute_lora_down_amx(qlen, activated_expert);  // AMX-optimized path
      } else {
        compute_lora_down(qlen, activated_expert);  // For-loop fallback for KGroup kernels
      }
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
    BACKWARD_TIMER_START();

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

    // // Compute total tokens for debug
    // size_t total_tokens = 0;
    // for (int i = 0; i < activated_expert; i++) {
    //   total_tokens += m_local_num_[m_expert_id_map_[i]];
    // }

    // printf("[BACKWARD DEBUG] qlen=%d, k=%d, activated_expert=%d, total_tokens=%zu\n", qlen, k, activated_expert,
    //        total_tokens);
    // printf("[BACKWARD DEBUG] grad_output norm: %f\n",
    //        compute_bf16_norm((const ggml_bf16_t*)grad_output, qlen * config_.hidden_size));

    // Step 1: Down projection backward
    if constexpr (supports_standard_mat_mul_v<T>) {
      backward_down_amx(cache, grad_output, grad_down_lora_a, grad_down_lora_b);
    } else {
      backward_down(cache, grad_output, grad_down_lora_a, grad_down_lora_b);
    }
    BACKWARD_TIMER_CHECKPOINT("backward_down");

    // Step 2: Activation backward
    backward_activation(cache);
    BACKWARD_TIMER_CHECKPOINT("backward_activation");

    // Step 3: Gate + Up projection backward
    if constexpr (supports_standard_mat_mul_v<T>) {
      backward_gate_up_amx(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
    } else {
      backward_gate_up(cache, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
    }
    BACKWARD_TIMER_CHECKPOINT("backward_gate_up");
    BACKWARD_TIMER_END();
    // printf("[BACKWARD DEBUG] After backward_gate_up - grad_input norm: %f\n",
    //        compute_bf16_norm((const ggml_bf16_t*)grad_input, qlen * config_.hidden_size));

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

    // Mark weights as needing re-conversion to BufferB format
    lora_weights_prepared_ = false;
  }

  /**
   * @brief Prepare LoRA weights for AMX GEMM.
   *
   * Converts BF16 LoRA weights from Python tensors to AMX BufferB format.
   * This includes padding to K_STEP multiples for AMX alignment.
   * Must be called before forward_sft() if lora_weights_prepared_ is false.
   */
  void prepare_lora_weights() {
    // Only prepare weights for kernels that support standard mat_mul
    if constexpr (!supports_standard_mat_mul_v<T>) {
      return;  // KGroup kernels use for-loop implementation
    }

    if (lora_weights_prepared_) {
      return;
    }
    if (gate_lora_a_ == nullptr) {
      return;  // No LoRA weights to prepare
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel conversion of all LoRA weights to BufferB format
    // 10 matrices per expert: gate/up (A, B, A^T, B^T) + down (A, B)
    pool->do_work_stealing_job(
        config_.expert_num * 10, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 10;
          int lora_type = task_id % 10;

          switch (lora_type) {
            case 0:  // gate_lora_a [lora_rank, hidden_size] -> [padded_lora_rank, hidden_size]
              convert_lora_a_to_buffer_b(gate_lora_a_, gate_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.hidden_size, padded_lora_rank_, config_.hidden_size);
              break;
            case 1:  // up_lora_a [lora_rank, hidden_size]
              convert_lora_a_to_buffer_b(up_lora_a_, up_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.hidden_size, padded_lora_rank_, config_.hidden_size);
              break;
            case 2:  // gate_lora_b [intermediate_size, lora_rank] -> [intermediate_size, padded_lora_rank]
              convert_lora_b_to_buffer_b(gate_lora_b_, gate_lora_b_bb_[expert_idx], expert_idx,
                                         config_.intermediate_size, lora_rank_, config_.intermediate_size,
                                         padded_lora_rank_);
              break;
            case 3:  // up_lora_b [intermediate_size, lora_rank]
              convert_lora_b_to_buffer_b(up_lora_b_, up_lora_b_bb_[expert_idx], expert_idx, config_.intermediate_size,
                                         lora_rank_, config_.intermediate_size, padded_lora_rank_);
              break;
            case 4:  // down_lora_a [lora_rank, intermediate_size] -> [padded_lora_rank, intermediate_size]
              convert_lora_a_to_buffer_b(down_lora_a_, down_lora_a_bb_[expert_idx], expert_idx, lora_rank_,
                                         config_.intermediate_size, padded_lora_rank_, config_.intermediate_size);
              break;
            case 5:  // down_lora_b [hidden_size, lora_rank] -> [hidden_size, padded_lora_rank]
              convert_lora_b_to_buffer_b(down_lora_b_, down_lora_b_bb_[expert_idx], expert_idx, config_.hidden_size,
                                         lora_rank_, config_.hidden_size, padded_lora_rank_);
              break;
            case 6:  // gate_lora_a^T [hidden_size, lora_rank] -> [hidden_size, padded_lora_rank]
              convert_lora_a_transposed_to_buffer_b(gate_lora_a_, gate_lora_a_t_bb_[expert_idx], expert_idx, lora_rank_,
                                                    config_.hidden_size, config_.hidden_size, padded_lora_rank_);
              break;
            case 7:  // up_lora_a^T
              convert_lora_a_transposed_to_buffer_b(up_lora_a_, up_lora_a_t_bb_[expert_idx], expert_idx, lora_rank_,
                                                    config_.hidden_size, config_.hidden_size, padded_lora_rank_);
              break;
            case 8:  // gate_lora_b^T [lora_rank, intermediate_size] -> [padded_lora_rank, intermediate_size]
              convert_lora_b_transposed_to_buffer_b(gate_lora_b_, gate_lora_b_t_bb_[expert_idx], expert_idx,
                                                    config_.intermediate_size, lora_rank_, padded_lora_rank_,
                                                    config_.intermediate_size);
              break;
            case 9:  // up_lora_b^T
              convert_lora_b_transposed_to_buffer_b(up_lora_b_, up_lora_b_t_bb_[expert_idx], expert_idx,
                                                    config_.intermediate_size, lora_rank_, padded_lora_rank_,
                                                    config_.intermediate_size);
              break;
          }
        },
        nullptr);

    lora_weights_prepared_ = true;
  }

  // Debug getter for LoRA pointer verification
  void* get_gate_lora_a() const { return (void*)gate_lora_a_; }

  /**
   * @brief Prepare backward weights for AMX GEMM.
   *
   * Converts base weights to transposed BufferB format for backward pass.
   * For backward GEMM, we need:
   * - gate_backward_bb_: gate_proj transposed [hidden_size, intermediate_size]
   * - up_backward_bb_: up_proj transposed [hidden_size, intermediate_size]
   * - down_backward_bb_: down_proj transposed [intermediate_size, hidden_size]
   *
   * Must be called before backward_down/backward_gate_up if backward_weights_prepared_ is false.
   */
  void prepare_backward_weights() {
    // Only prepare weights for kernels that support standard mat_mul
    if constexpr (!supports_standard_mat_mul_v<T>) {
      return;  // KGroup kernels use for-loop implementation
    }

    if (backward_weights_prepared_) return;
    if (config_.gate_proj == nullptr) return;  // No base weights to prepare

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Parallel conversion of all base weights to transposed BufferB format
    // 3 matrices per expert: gate, up, down
    pool->do_work_stealing_job(
        config_.expert_num * 3, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 3;
          int weight_type = task_id % 3;

          switch (weight_type) {
            case 0: {  // gate_proj: [intermediate_size, hidden_size] -> transposed [hidden_size, intermediate_size]
              const ggml_bf16_t* gate_proj = (const ggml_bf16_t*)config_.gate_proj;
              size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

              // Create transposed matrix: [hidden_size, intermediate_size]
              // Original: [intermediate_size, hidden_size] stored row-major
              // Transposed: [hidden_size, intermediate_size] stored row-major
              std::vector<ggml_bf16_t> transposed(config_.hidden_size * config_.intermediate_size);
              for (int i = 0; i < config_.intermediate_size; i++) {
                for (int h = 0; h < config_.hidden_size; h++) {
                  transposed[h * config_.intermediate_size + i] =
                      gate_proj[expert_offset + i * config_.hidden_size + h];
                }
              }
              gate_backward_bb_[expert_idx]->from_mat(transposed.data(), 0, 1);
              break;
            }
            case 1: {  // up_proj: same as gate
              const ggml_bf16_t* up_proj = (const ggml_bf16_t*)config_.up_proj;
              size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

              std::vector<ggml_bf16_t> transposed(config_.hidden_size * config_.intermediate_size);
              for (int i = 0; i < config_.intermediate_size; i++) {
                for (int h = 0; h < config_.hidden_size; h++) {
                  transposed[h * config_.intermediate_size + i] = up_proj[expert_offset + i * config_.hidden_size + h];
                }
              }
              up_backward_bb_[expert_idx]->from_mat(transposed.data(), 0, 1);
              break;
            }
            case 2: {  // down_proj: [hidden_size, intermediate_size] -> transposed [intermediate_size, hidden_size]
              const ggml_bf16_t* down_proj = (const ggml_bf16_t*)config_.down_proj;
              size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;

              // Create transposed matrix: [intermediate_size, hidden_size]
              // Original: [hidden_size, intermediate_size] stored row-major
              // Transposed: [intermediate_size, hidden_size] stored row-major
              std::vector<ggml_bf16_t> transposed(config_.intermediate_size * config_.hidden_size);
              for (int h = 0; h < config_.hidden_size; h++) {
                for (int i = 0; i < config_.intermediate_size; i++) {
                  transposed[i * config_.hidden_size + h] =
                      down_proj[expert_offset + h * config_.intermediate_size + i];
                }
              }
              // FIX: Use the same nth as mat_mul will use, to ensure BufferB layout matches.
              // mat_mul uses nth = recommended_nth(intermediate_size) which partitions by N_BLOCK.
              int nth = T::recommended_nth(config_.intermediate_size);
              for (int ith = 0; ith < nth; ith++) {
                down_backward_bb_[expert_idx]->from_mat(transposed.data(), ith, nth);
              }
              break;
            }
          }
        },
        nullptr);

    backward_weights_prepared_ = true;
  }

  /**
   * @brief Set base weight pointers for TP partitioning.
   * Used by TP_MOE_SFT::load_weights() to set partitioned weights before calling load_weights().
   */
  void set_base_weight_pointers(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  /**
   * @brief Set physical to logical expert mapping.
   */
  void set_physical_to_logical_map(const void* map) { config_.physical_to_logical_map = const_cast<void*>(map); }

 private:
  /**
   * @brief Initialize all buffers in a single alloc() call.
   *
   * IMPORTANT: SharedMemBuffer is designed to let multiple callers share the same memory pool.
   * Each alloc() call assigns pointers starting from the SAME base address, which means:
   * - Multiple alloc() calls will OVERLAP in memory!
   * - This is intentional for temporary buffers that are not used simultaneously.
   * - But for SFT, cache and grad buffers ARE used simultaneously (cache written during forward,
   *   grad written during backward, both needed in backward_activation).
   *
   * Solution: Combine all buffer requests into a SINGLE alloc() call, so they get
   * consecutive, non-overlapping addresses.
   *
   * Bug #15 root cause: Three separate alloc() calls caused grad_intermediate_ to overlap
   * with cache_gate_output_pool_, and memset in backward_down() zeroed the cache data.
   */
  void init_all_buffers() {
    // =====================================================
    // Calculate padded_lora_rank for AMX alignment
    // AMX requires K dimension to be multiple of K_STEP (32 for BF16)
    // =====================================================
    constexpr int K_STEP = T::K_STEP;
    constexpr int N_STEP = T::N_STEP;
    constexpr int M_STEP = T::M_STEP;
    padded_lora_rank_ = ((lora_rank_ + K_STEP - 1) / K_STEP) * K_STEP;
    // Also need N dimension aligned for BufferB output dimension
    int padded_lora_rank_n = ((lora_rank_ + N_STEP - 1) / N_STEP) * N_STEP;
    // Use the larger of the two for consistency
    padded_lora_rank_ = std::max(padded_lora_rank_, padded_lora_rank_n);

    // Calculate all buffer sizes
    lora_intermediate_pool_bytes_ = sizeof(ggml_bf16_t) * config_.max_len * config_.num_experts_per_tok * lora_rank_;

    cache_slot_bytes_input_ = config_.max_len * config_.hidden_size * sizeof(ggml_bf16_t);
    cache_slot_bytes_intermediate_ =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    size_t grad_buffer_bytes =
        config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t);

    // =====================================================
    // Calculate LoRA AMX buffer sizes
    // Only for kernels that support standard mat_mul API
    // =====================================================
    // Max tokens per expert (with M_STEP alignment)
    // Bug-C Fix: Each expert processes at most max_len tokens (worst case: all tokens select this expert)
    // Previously used max_len * num_experts_per_tok which is incorrect and wastes 8x memory
    size_t max_m = ((config_.max_len + M_STEP - 1) / M_STEP) * M_STEP;

    // Variables for buffer sizes (used in init_lora_amx_buffers)
    size_t lora_a_gate_up_bb_size = 0;
    size_t lora_b_gate_up_bb_size = 0;
    size_t lora_a_gate_up_t_bb_size = 0;
    size_t lora_b_gate_up_t_bb_size = 0;
    size_t lora_a_down_bb_size = 0;
    size_t lora_b_down_bb_size = 0;
    size_t lora_intermediate_ba_size = 0;
    size_t lora_intermediate_bc_size = 0;
    size_t lora_gate_up_out_bc_size = 0;
    size_t lora_down_out_bc_size = 0;
    size_t grad_output_ba_size = 0;
    size_t grad_intermediate_bc_size = 0;
    size_t grad_gate_up_bc_size = 0;
    size_t gate_up_backward_bb_size = 0;
    size_t down_backward_bb_size = 0;

    if constexpr (supports_standard_mat_mul_v<T>) {
      // BufferB sizes for LoRA weights (need to be aligned)
      // gate/up lora_A: [padded_lora_rank, hidden_size] per expert
      lora_a_gate_up_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.hidden_size);
      // gate/up lora_B: [intermediate_size, padded_lora_rank] per expert
      lora_b_gate_up_bb_size = T::BufferB::required_size(config_.intermediate_size, padded_lora_rank_);
      // Transposed weights for backward LoRA GEMM
      // gate/up lora_A^T: [hidden_size, padded_lora_rank] per expert
      lora_a_gate_up_t_bb_size = T::BufferB::required_size(config_.hidden_size, padded_lora_rank_);
      // gate/up lora_B^T: [padded_lora_rank, intermediate_size] per expert
      lora_b_gate_up_t_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size);
      // down lora_A: [padded_lora_rank, intermediate_size] per expert
      lora_a_down_bb_size = T::BufferB::required_size(padded_lora_rank_, config_.intermediate_size);
      // down lora_B: [hidden_size, padded_lora_rank] per expert
      lora_b_down_bb_size = T::BufferB::required_size(config_.hidden_size, padded_lora_rank_);

      // Total BufferB pool size for all experts (10 matrices per expert)
      lora_bb_pool_bytes_ = config_.expert_num * (lora_a_gate_up_bb_size * 2 +    // gate_a, up_a
                                                  lora_b_gate_up_bb_size * 2 +    // gate_b, up_b
                                                  lora_a_gate_up_t_bb_size * 2 +  // gate_a^T, up_a^T
                                                  lora_b_gate_up_t_bb_size * 2 +  // gate_b^T, up_b^T
                                                  lora_a_down_bb_size +           // down_a
                                                  lora_b_down_bb_size);           // down_b

      // Bug-C Fix Step 2: Use shared buffer pool instead of per-expert allocation
      // Max total tokens across all activated experts per forward pass
      // (each of max_len tokens selects num_experts_per_tok experts)
      size_t max_total_tokens = ((config_.max_len * config_.num_experts_per_tok + M_STEP - 1) / M_STEP) * M_STEP;

      // BufferA for LoRA intermediate: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_ba_size = T::BufferA::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_ba_pool_bytes_ = T::BufferA::required_size(max_total_tokens, padded_lora_rank_) * 2;  // gate + up

      // BufferC for LoRA step 1 output: shared pool for all activated experts
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bc_size = T::BufferC::required_size(max_m, padded_lora_rank_);  // per-expert size for set_data
      lora_bc_inter_pool_bytes_ = T::BufferC::required_size(max_total_tokens, padded_lora_rank_) * 2;  // gate + up

      // BufferC for LoRA step 2 output (gate, up, down): shared pool for all activated experts
      lora_gate_up_out_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      lora_down_out_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);           // per-expert size
      lora_bc_out_pool_bytes_ = T::BufferC::required_size(max_total_tokens, config_.intermediate_size) * 2 +
                                T::BufferC::required_size(max_total_tokens, config_.hidden_size);

      // BF16 intermediate buffer for step 1 -> step 2 conversion
      // Need 2x for gate and up separate buffers (to avoid race condition)
      lora_intermediate_bf16_pool_bytes_ = max_total_tokens * padded_lora_rank_ * sizeof(ggml_bf16_t) * 2;  // gate + up

      // =====================================================
      // Calculate Backward pass AMX buffer sizes
      // =====================================================
      // BufferA for scattered grad_output: shared pool for all activated experts
      grad_output_ba_size = T::BufferA::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_ba_pool_bytes_ = T::BufferA::required_size(max_total_tokens, config_.hidden_size);

      // BufferC for backward GEMM outputs: shared pool for all activated experts
      // grad_intermediate: [max_total_tokens, intermediate_size]
      grad_intermediate_bc_size = T::BufferC::required_size(max_m, config_.intermediate_size);  // per-expert size
      // grad_gate_up: [max_total_tokens, hidden_size]
      grad_gate_up_bc_size = T::BufferC::required_size(max_m, config_.hidden_size);  // per-expert size
      backward_bc_pool_bytes_ = T::BufferC::required_size(max_total_tokens, config_.intermediate_size) +
                                T::BufferC::required_size(max_total_tokens, config_.hidden_size);

      // BF16 buffer for scattered grad_output
      grad_output_bf16_pool_bytes_ = max_total_tokens * config_.hidden_size * sizeof(ggml_bf16_t);

      // =====================================================
      // Calculate Backward pass BufferB sizes (transposed base weights)
      // =====================================================
      // For backward GEMM, we need transposed versions of base weights:
      // - gate/up backward: BufferB[hidden_size, intermediate_size] per expert
      // - down backward: BufferB[intermediate_size, hidden_size] per expert
      gate_up_backward_bb_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      down_backward_bb_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
      backward_bb_pool_bytes_ = config_.expert_num * (gate_up_backward_bb_size * 2 + down_backward_bb_size);
    } else {
      // For unsupported kernels (KGroup kernels), set all AMX buffer sizes to 0
      // These kernels will use the original for-loop implementation
      lora_bb_pool_bytes_ = 0;
      lora_ba_pool_bytes_ = 0;
      lora_bc_inter_pool_bytes_ = 0;
      lora_bc_out_pool_bytes_ = 0;
      lora_intermediate_bf16_pool_bytes_ = 0;
      backward_ba_pool_bytes_ = 0;
      backward_bc_pool_bytes_ = 0;
      grad_output_bf16_pool_bytes_ = 0;
      backward_bb_pool_bytes_ = 0;
    }

    // ★ Single alloc() call - all buffers get consecutive, non-overlapping addresses ★
    MemoryRequest mem_requests;

    // LoRA buffers (legacy, kept for compatibility)
    mem_requests.append_pointer(&lora_intermediate_pool_, lora_intermediate_pool_bytes_);

    // Cache buffers (4 pools × max_cache_depth)
    mem_requests.append_pointer(&cache_input_pool_, cache_slot_bytes_input_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_gate_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_up_output_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);
    mem_requests.append_pointer(&cache_intermediate_pool_, cache_slot_bytes_intermediate_ * max_cache_depth_);

    // Gradient buffers (3 pools)
    mem_requests.append_pointer(&grad_intermediate_pool_, grad_buffer_bytes);
    mem_requests.append_pointer(&grad_gate_output_pool_, grad_buffer_bytes);
    mem_requests.append_pointer(&grad_up_output_pool_, grad_buffer_bytes);

    // LoRA AMX buffers - use aligned_alloc for independent allocation
    // (shared_mem_buffer would cause memory sharing issues)
    if (lora_bb_pool_bytes_ > 0) {
      lora_bb_pool_ = aligned_alloc(64, lora_bb_pool_bytes_);
      memset(lora_bb_pool_, 0, lora_bb_pool_bytes_);
    }
    if (lora_ba_pool_bytes_ > 0) {
      lora_ba_pool_ = aligned_alloc(64, lora_ba_pool_bytes_);
      memset(lora_ba_pool_, 0, lora_ba_pool_bytes_);
    }
    if (lora_bc_inter_pool_bytes_ > 0) {
      lora_bc_inter_pool_ = aligned_alloc(64, lora_bc_inter_pool_bytes_);
      memset(lora_bc_inter_pool_, 0, lora_bc_inter_pool_bytes_);
    }
    if (lora_bc_out_pool_bytes_ > 0) {
      lora_bc_out_pool_ = aligned_alloc(64, lora_bc_out_pool_bytes_);
      memset(lora_bc_out_pool_, 0, lora_bc_out_pool_bytes_);
    }
    if (lora_intermediate_bf16_pool_bytes_ > 0) {
      lora_intermediate_bf16_pool_ = aligned_alloc(64, lora_intermediate_bf16_pool_bytes_);
      memset(lora_intermediate_bf16_pool_, 0, lora_intermediate_bf16_pool_bytes_);
    }

    // Backward pass AMX buffers
    mem_requests.append_pointer(&backward_ba_pool_, backward_ba_pool_bytes_);
    mem_requests.append_pointer(&backward_bc_pool_, backward_bc_pool_bytes_);
    mem_requests.append_pointer(&grad_output_bf16_pool_, grad_output_bf16_pool_bytes_);

    // Backward pass BufferB (transposed base weights)
    mem_requests.append_pointer(&backward_bb_pool_, backward_bb_pool_bytes_);

    // Single allocation for all buffers
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);

    // Initialize LoRA and gradient pointers
    lora_intermediate_ = (ggml_bf16_t*)lora_intermediate_pool_;
    grad_intermediate_ = (ggml_bf16_t*)grad_intermediate_pool_;
    grad_gate_output_ = (ggml_bf16_t*)grad_gate_output_pool_;
    grad_up_output_ = (ggml_bf16_t*)grad_up_output_pool_;

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

    // =====================================================
    // Initialize LoRA AMX buffer objects (only for supported kernels)
    // =====================================================
    if constexpr (supports_standard_mat_mul_v<T>) {
      init_lora_amx_buffers(max_m, lora_a_gate_up_bb_size, lora_b_gate_up_bb_size, lora_a_gate_up_t_bb_size,
                            lora_b_gate_up_t_bb_size, lora_a_down_bb_size, lora_b_down_bb_size,
                            lora_intermediate_ba_size, lora_intermediate_bc_size, lora_gate_up_out_bc_size,
                            lora_down_out_bc_size, grad_output_ba_size, grad_intermediate_bc_size, grad_gate_up_bc_size,
                            gate_up_backward_bb_size, down_backward_bb_size);
    }

    // Bug-C Debug: Memory allocation summary (commented out after verification)
    // Uncomment to see memory allocation details
    /*
    printf("\n========== Memory Allocation Summary ==========\n");
    printf("Config: expert_num=%d, hidden_size=%d, intermediate_size=%d\n",
           config_.expert_num, config_.hidden_size, config_.intermediate_size);
    printf("Config: max_len=%d, num_experts_per_tok=%d, lora_rank=%d, padded_lora_rank=%d\n",
           config_.max_len, config_.num_experts_per_tok, lora_rank_, padded_lora_rank_);
    printf("Calculated max_m=%zu, max_total_tokens=%zu\n",
           max_m, (size_t)config_.max_len * config_.num_experts_per_tok);
    printf("\n--- LoRA Buffers (aligned_alloc) ---\n");
    printf("  lora_bb_pool_bytes_:              %12zu bytes (%6.2f MB)\n", lora_bb_pool_bytes_, lora_bb_pool_bytes_ /
    1024.0 / 1024.0); printf("  lora_ba_pool_bytes_:              %12zu bytes (%6.2f MB)\n", lora_ba_pool_bytes_,
    lora_ba_pool_bytes_ / 1024.0 / 1024.0); printf("  lora_bc_inter_pool_bytes_:        %12zu bytes (%6.2f MB)\n",
    lora_bc_inter_pool_bytes_, lora_bc_inter_pool_bytes_ / 1024.0 / 1024.0); printf("  lora_bc_out_pool_bytes_: %12zu
    bytes (%6.2f GB)\n", lora_bc_out_pool_bytes_, lora_bc_out_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("
    lora_intermediate_bf16_pool_bytes_:%12zu bytes (%6.2f MB)\n", lora_intermediate_bf16_pool_bytes_,
    lora_intermediate_bf16_pool_bytes_ / 1024.0 / 1024.0); printf("\n--- Backward Buffers (shared_mem_buffer) ---\n");
    printf("  backward_ba_pool_bytes_:          %12zu bytes (%6.2f GB)\n", backward_ba_pool_bytes_,
    backward_ba_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("  backward_bc_pool_bytes_:          %12zu bytes (%6.2f
    GB)\n", backward_bc_pool_bytes_, backward_bc_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("
    grad_output_bf16_pool_bytes_:     %12zu bytes (%6.2f GB)\n", grad_output_bf16_pool_bytes_,
    grad_output_bf16_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("  backward_bb_pool_bytes_:          %12zu bytes
    (%6.2f GB)\n", backward_bb_pool_bytes_, backward_bb_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf("\n--- Other
    Buffers (shared_mem_buffer) ---\n"); printf("  lora_intermediate_pool_bytes_:    %12zu bytes (%6.2f GB)\n",
    lora_intermediate_pool_bytes_, lora_intermediate_pool_bytes_ / 1024.0 / 1024.0 / 1024.0); printf(" grad_buffer_bytes
    (×3):           %12zu bytes (%6.2f GB)\n", grad_buffer_bytes * 3, grad_buffer_bytes * 3 / 1024.0 / 1024.0 / 1024.0);
    size_t cache_total = (cache_slot_bytes_input_ + cache_slot_bytes_intermediate_ * 3) * max_cache_depth_;
    printf("  cache_total (depth=%d):           %12zu bytes (%6.2f GB)\n", max_cache_depth_, cache_total, cache_total /
    1024.0 / 1024.0 / 1024.0); size_t total_aligned = lora_bb_pool_bytes_ + lora_ba_pool_bytes_ +
    lora_bc_inter_pool_bytes_ + lora_bc_out_pool_bytes_ + lora_intermediate_bf16_pool_bytes_; size_t total_shared =
    backward_ba_pool_bytes_ + backward_bc_pool_bytes_ + grad_output_bf16_pool_bytes_ + backward_bb_pool_bytes_ +
    lora_intermediate_pool_bytes_ + grad_buffer_bytes * 3 + cache_total; printf("\n--- Summary ---\n"); printf("  Total
    aligned_alloc:              %12zu bytes (%6.2f GB)\n", total_aligned, total_aligned / 1024.0 / 1024.0 / 1024.0);
    printf("  Total shared_mem_buffer:          %12zu bytes (%6.2f GB)\n", total_shared, total_shared / 1024.0 / 1024.0
    / 1024.0); printf("  GRAND TOTAL:                      %12zu bytes (%6.2f GB)\n", total_aligned + total_shared,
    (total_aligned + total_shared) / 1024.0 / 1024.0 / 1024.0);
    printf("===============================================\n\n");
    */
  }

  /**
   * @brief Initialize LoRA AMX buffer objects (including backward pass buffers).
   */
  void init_lora_amx_buffers(size_t max_m, size_t lora_a_gate_up_bb_size, size_t lora_b_gate_up_bb_size,
                             size_t lora_a_gate_up_t_bb_size, size_t lora_b_gate_up_t_bb_size,
                             size_t lora_a_down_bb_size, size_t lora_b_down_bb_size, size_t lora_intermediate_ba_size,
                             size_t lora_intermediate_bc_size, size_t lora_gate_up_out_bc_size,
                             size_t lora_down_out_bc_size, size_t grad_output_ba_size, size_t grad_intermediate_bc_size,
                             size_t grad_gate_up_bc_size, size_t gate_up_backward_bb_size,
                             size_t down_backward_bb_size) {
    // Resize vectors - forward pass
    gate_lora_a_bb_.resize(config_.expert_num);
    up_lora_a_bb_.resize(config_.expert_num);
    down_lora_a_bb_.resize(config_.expert_num);
    gate_lora_b_bb_.resize(config_.expert_num);
    up_lora_b_bb_.resize(config_.expert_num);
    down_lora_b_bb_.resize(config_.expert_num);
    gate_lora_a_t_bb_.resize(config_.expert_num);
    up_lora_a_t_bb_.resize(config_.expert_num);
    gate_lora_b_t_bb_.resize(config_.expert_num);
    up_lora_b_t_bb_.resize(config_.expert_num);
    // Separate buffers for gate and up to avoid race condition
    lora_gate_intermediate_ba_.resize(config_.expert_num);
    lora_up_intermediate_ba_.resize(config_.expert_num);
    lora_gate_intermediate_bc_.resize(config_.expert_num);
    lora_up_intermediate_bc_.resize(config_.expert_num);
    lora_gate_out_bc_.resize(config_.expert_num);
    lora_up_out_bc_.resize(config_.expert_num);
    lora_down_out_bc_.resize(config_.expert_num);
    lora_gate_intermediate_ptr_.resize(config_.expert_num);
    lora_up_intermediate_ptr_.resize(config_.expert_num);

    // Resize vectors - backward pass
    grad_output_ba_.resize(config_.expert_num);
    grad_intermediate_bc_.resize(config_.expert_num);
    grad_gate_up_bc_.resize(config_.expert_num);
    grad_output_bf16_ptr_.resize(config_.expert_num);

    // Resize vectors - backward BufferB (transposed base weights)
    gate_backward_bb_.resize(config_.expert_num);
    up_backward_bb_.resize(config_.expert_num);
    down_backward_bb_.resize(config_.expert_num);

    // Calculate offsets and create buffer objects
    // Bug-C Fix Step 2: BufferA/BufferC use shared pools, data will be assigned in forward/backward
    char* bb_ptr = (char*)lora_bb_pool_;

    for (int i = 0; i < config_.expert_num; i++) {
      // BufferB for LoRA weights (still per-expert, as weights are different for each expert)
      gate_lora_a_bb_[i] = std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_bb_size;

      up_lora_a_bb_[i] = std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.hidden_size, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_bb_size;

      gate_lora_b_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_bb_size;

      up_lora_b_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_bb_size;

      gate_lora_a_t_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_t_bb_size;

      up_lora_a_t_bb_[i] = std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_a_gate_up_t_bb_size;

      gate_lora_b_t_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_t_bb_size;

      up_lora_b_t_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_b_gate_up_t_bb_size;

      down_lora_a_bb_[i] =
          std::make_shared<typename T::BufferB>(padded_lora_rank_, config_.intermediate_size, (void*)bb_ptr);
      bb_ptr += lora_a_down_bb_size;

      down_lora_b_bb_[i] = std::make_shared<typename T::BufferB>(config_.hidden_size, padded_lora_rank_, (void*)bb_ptr);
      bb_ptr += lora_b_down_bb_size;

      // BufferA for LoRA intermediate: create with nullptr, will set_data in forward
      lora_gate_intermediate_ba_[i] = std::make_shared<typename T::BufferA>(max_m, padded_lora_rank_, nullptr);
      lora_up_intermediate_ba_[i] = std::make_shared<typename T::BufferA>(max_m, padded_lora_rank_, nullptr);

      // BufferC for LoRA step 1 output: create with nullptr, will set_data in forward
      lora_gate_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, padded_lora_rank_, nullptr);
      lora_up_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, padded_lora_rank_, nullptr);

      // BufferC for LoRA step 2 output: create with nullptr, will set_data in forward
      lora_gate_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);
      lora_up_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);
      lora_down_out_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.hidden_size, nullptr);

      // BF16 intermediate pointer: will be assigned in forward
      lora_gate_intermediate_ptr_[i] = nullptr;
      lora_up_intermediate_ptr_[i] = nullptr;
    }

    // =====================================================
    // Initialize backward pass buffer objects
    // Bug-C Fix Step 2: Use shared pools, data will be assigned in backward
    // =====================================================
    for (int i = 0; i < config_.expert_num; i++) {
      // BufferA for grad_output: create with nullptr, will set_data in backward
      grad_output_ba_[i] = std::make_shared<typename T::BufferA>(max_m, config_.hidden_size, nullptr);

      // BufferC for grad_intermediate: create with nullptr, will set_data in backward
      grad_intermediate_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.intermediate_size, nullptr);

      // BufferC for grad_gate_up: create with nullptr, will set_data in backward
      grad_gate_up_bc_[i] = std::make_shared<typename T::BufferC>(max_m, config_.hidden_size, nullptr);

      // BF16 pointer: will be assigned in backward
      grad_output_bf16_ptr_[i] = nullptr;
    }

    // =====================================================
    // Initialize backward BufferB objects (transposed base weights)
    // =====================================================
    char* backward_bb_ptr = (char*)backward_bb_pool_;
    for (int i = 0; i < config_.expert_num; i++) {
      // BufferB for gate backward: [hidden_size, intermediate_size]
      gate_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      // BufferB for up backward: [hidden_size, intermediate_size]
      up_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, (void*)backward_bb_ptr);
      backward_bb_ptr += gate_up_backward_bb_size;

      // BufferB for down backward: [intermediate_size, hidden_size]
      down_backward_bb_[i] =
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, (void*)backward_bb_ptr);
      backward_bb_ptr += down_backward_bb_size;
    }

    lora_weights_prepared_ = false;
    backward_weights_prepared_ = false;
  }

  /**
   * @brief Convert LoRA A matrix to BufferB format with padding.
   *
   * LoRA A shape: [expert_num, lora_rank, k_dim]
   * Padded shape: [expert_num, padded_lora_rank, k_dim]
   * BufferB expects: [n_dim, k_dim] where n_dim = padded_lora_rank
   *
   * Padding rows with zeros for lora_rank < padded_lora_rank.
   */
  void convert_lora_a_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb, int expert_idx,
                                  int src_n, int src_k, int dst_n, int dst_k) {
    // Create temporary padded matrix
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));

    // Copy source data (with potential padding)
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;
    for (int r = 0; r < src_n && r < dst_n; r++) {
      for (int c = 0; c < src_k && c < dst_k; c++) {
        padded[r * dst_k + c] = expert_src[r * src_k + c];
      }
    }

    // Convert to BufferB format using from_mat
    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Convert LoRA B matrix to BufferB format with padding.
   *
   * LoRA B shape: [expert_num, output_dim, lora_rank]
   * Padded shape: [expert_num, output_dim, padded_lora_rank]
   * BufferB expects: [n_dim, k_dim] where n_dim = output_dim, k_dim = padded_lora_rank
   *
   * Padding columns with zeros for lora_rank < padded_lora_rank.
   */
  void convert_lora_b_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb, int expert_idx,
                                  int src_n, int src_k, int dst_n, int dst_k) {
    // Create temporary padded matrix
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));

    // Copy source data (with potential padding on K dimension)
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int r = 0; r < src_n && r < dst_n; r++) {
      for (int c = 0; c < src_k && c < dst_k; c++) {
        padded[r * dst_k + c] = expert_src[r * src_k + c];
      }
    }

    // Convert to BufferB format using from_mat
    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Convert LoRA A^T matrix to BufferB format with padding on rank dimension.
   *
   * Input shape: [expert_num, lora_rank, hidden_size]
   * Output shape: [expert_num, hidden_size, padded_lora_rank]
   */
  void convert_lora_a_transposed_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb,
                                             int expert_idx, int src_n, int src_k, int dst_n, int dst_k) {
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int h = 0; h < src_k && h < dst_n; h++) {
      for (int r = 0; r < src_n && r < dst_k; r++) {
        padded[h * dst_k + r] = expert_src[r * src_k + h];
      }
    }

    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Convert LoRA B^T matrix to BufferB format with padding on rank dimension.
   *
   * Input shape: [expert_num, intermediate_size, lora_rank]
   * Output shape: [expert_num, padded_lora_rank, intermediate_size]
   */
  void convert_lora_b_transposed_to_buffer_b(const ggml_bf16_t* src, std::shared_ptr<typename T::BufferB>& dst_bb,
                                             int expert_idx, int src_n, int src_k, int dst_n, int dst_k) {
    std::vector<ggml_bf16_t> padded(dst_n * dst_k, GGML_FP32_TO_BF16(0.0f));
    const ggml_bf16_t* expert_src = src + expert_idx * src_n * src_k;

    for (int r = 0; r < src_k && r < dst_n; r++) {
      for (int i = 0; i < src_n && i < dst_k; i++) {
        padded[r * dst_k + i] = expert_src[i * src_k + r];
      }
    }

    dst_bb->from_mat(padded.data(), 0, 1);
  }

  /**
   * @brief Compute LoRA for gate and up projections using AMX GEMM.
   *
   * gate_lora_out = (input @ gate_lora_A^T) @ gate_lora_B^T * scaling
   * gate_output += gate_lora_out
   * (similar for up)
   *
   * This is the AMX-optimized version replacing the naive for-loop implementation.
   */
  void compute_lora_gate_up_amx(int qlen, int activated_expert) {
    if (gate_lora_a_ == nullptr || gate_lora_b_ == nullptr) {
      return;
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Ensure LoRA weights are prepared
    prepare_lora_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate LoRA buffers from shared pool
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Pool pointers for forward LoRA buffers
    char* lora_ba_ptr = (char*)lora_ba_pool_;
    char* lora_bc_inter_ptr = (char*)lora_bc_inter_pool_;
    char* lora_bc_out_ptr = (char*)lora_bc_out_pool_;
    char* bf16_inter_ptr = (char*)lora_intermediate_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferA for intermediate (gate and up)
      lora_gate_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      lora_up_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // Allocate BufferC for intermediate (gate and up)
      lora_gate_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      lora_up_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // Allocate BufferC for output (gate, up, down - but down is done in compute_lora_down_amx)
      lora_gate_out_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_out_bc_[expert_idx]->set_data(lora_bc_out_ptr);
      lora_bc_out_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      lora_up_out_bc_[expert_idx]->max_m = local_max_m;
      lora_up_out_bc_[expert_idx]->set_data(lora_bc_out_ptr);
      lora_bc_out_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      // Allocate BF16 intermediate buffer (gate and up)
      lora_gate_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));

      lora_up_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));
    }

    // =====================================================
    // Step 1: input @ lora_A^T -> lora_intermediate
    // Uses gate_up_ba_ (already quantized input)
    // Gate and Up use SEPARATE intermediate buffers to avoid race condition
    // =====================================================
    int nth = T::recommended_nth(padded_lora_rank_);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = gate_up_ba_[expert_idx];  // Reuse quantized input
          auto& bb = do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx];
          // Use separate BufferC for gate and up to avoid race condition
          auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];

          // GEMM: [m, hidden_size] @ [padded_lora_rank, hidden_size]^T -> [m, padded_lora_rank]
          amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, ba, bb, bc, ith, nth);

          // Convert BufferC to BF16 for step 2 input (separate for gate and up)
          ggml_bf16_t* inter_ptr =
              do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
          bc->to_mat(m, inter_ptr, ith, nth);
        },
        nullptr);

    // =====================================================
    // Step 2: Quantize lora_intermediate to BufferA
    // Need to quantize BOTH gate and up intermediates separately
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert * 2, nullptr,  // 2x tasks for gate and up
        [this](int task_id) {
          bool do_up = task_id % 2;
          int expert_idx = m_expert_id_map_[task_id / 2];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          // Use separate BufferA and BF16 pointer for gate and up
          auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
          ggml_bf16_t* ptr = do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
          ba->from_mat(m, ptr, 0, 1);
        },
        nullptr);

    // =====================================================
    // Step 3: lora_intermediate @ lora_B^T -> lora_output
    // Then add to main gate/up output with scaling
    // =====================================================
    nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          // Use separate BufferA for gate and up
          auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
          auto& bb = do_up ? up_lora_b_bb_[expert_idx] : gate_lora_b_bb_[expert_idx];
          auto& bc = do_up ? lora_up_out_bc_[expert_idx] : lora_gate_out_bc_[expert_idx];

          // GEMM: [m, padded_lora_rank] @ [intermediate_size, padded_lora_rank]^T -> [m, intermediate_size]
          amx::mat_mul(m, config_.intermediate_size, padded_lora_rank_, ba, bb, bc, ith, nth);

          // Add LoRA output to main output with scaling
          ggml_bf16_t* main_output = do_up ? m_local_up_output_ptr_[expert_idx] : m_local_gate_output_ptr_[expert_idx];
          add_lora_output_to_main(bc.get(), main_output, m, config_.intermediate_size, lora_scaling_, ith, nth);
        },
        nullptr);
  }

  /**
   * @brief Compute LoRA for down projection using AMX GEMM.
   */
  void compute_lora_down_amx(int qlen, int activated_expert) {
    if (down_lora_a_ == nullptr || down_lora_b_ == nullptr) return;

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Ensure LoRA weights are prepared
    prepare_lora_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate lora_down_out_bc_ from shared pool
    // Note: lora_gate_intermediate_bc_ and lora_gate_intermediate_ba_ are reused
    // from compute_lora_gate_up_amx (they are not used simultaneously)
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Use offset after gate and up output buffers in lora_bc_out_pool_
    // Pool layout: [gate_out × N] [up_out × N] [down_out × N]
    // But since we allocate dynamically, we need to track the offset
    // Actually, we can reuse the lora_bc_out_pool_ starting position since
    // gate/up outputs are already consumed by this point

    // For simplicity, allocate from the end of the pool (after gate+up)
    // Calculate gate+up total size first
    size_t gate_up_total = 0;
    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      size_t local_max_m = ((m_local_num_[expert_idx] + M_STEP - 1) / M_STEP) * M_STEP;
      gate_up_total += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size)) * 2;  // gate + up
    }

    char* lora_down_bc_ptr = (char*)lora_bc_out_pool_ + gate_up_total;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      lora_down_out_bc_[expert_idx]->max_m = local_max_m;
      lora_down_out_bc_[expert_idx]->set_data(lora_down_bc_ptr);
      lora_down_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));
    }

    // =====================================================
    // Step 1: intermediate @ down_lora_A^T -> lora_intermediate
    // Uses down_ba_ (already quantized intermediate after activation)
    // =====================================================
    int nth = T::recommended_nth(padded_lora_rank_);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = down_ba_[expert_idx];  // Reuse quantized intermediate
          auto& bb = down_lora_a_bb_[expert_idx];
          // Reuse gate intermediate buffer (no race condition for down projection)
          auto& bc = lora_gate_intermediate_bc_[expert_idx];

          // GEMM: [m, intermediate_size] @ [padded_lora_rank, intermediate_size]^T -> [m, padded_lora_rank]
          amx::mat_mul(m, padded_lora_rank_, config_.intermediate_size, ba, bb, bc, ith, nth);

          // Convert BufferC to BF16 for step 2 input
          bc->to_mat(m, lora_gate_intermediate_ptr_[expert_idx], ith, nth);
        },
        nullptr);

    // =====================================================
    // Step 2: Quantize lora_intermediate to BufferA
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          // Reuse gate intermediate buffer (no race condition for down projection)
          lora_gate_intermediate_ba_[expert_idx]->from_mat(m, lora_gate_intermediate_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    // =====================================================
    // Step 3: lora_intermediate @ down_lora_B^T -> lora_output
    // Then add to main down output with scaling
    // =====================================================
    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          // Reuse gate intermediate buffer (no race condition for down projection)
          auto& ba = lora_gate_intermediate_ba_[expert_idx];
          auto& bb = down_lora_b_bb_[expert_idx];
          auto& bc = lora_down_out_bc_[expert_idx];

          // GEMM: [m, padded_lora_rank] @ [hidden_size, padded_lora_rank]^T -> [m, hidden_size]
          amx::mat_mul(m, config_.hidden_size, padded_lora_rank_, ba, bb, bc, ith, nth);

          // Add LoRA output to main output with scaling
          add_lora_output_to_main(bc.get(), m_local_down_output_ptr_[expert_idx], m, config_.hidden_size, lora_scaling_,
                                  ith, nth);
        },
        nullptr);
  }

  /**
   * @brief Add LoRA BufferC output to main BF16 output with scaling.
   *
   * main_output[i] += lora_bc_output[i] * scaling
   */
  void add_lora_output_to_main(typename T::BufferC* bc, ggml_bf16_t* main_output, int m, int n, float scaling, int ith,
                               int nth) {
    auto [n_start, n_end] = T::split_range_n(n, ith, nth);

    for (int m_i = 0; m_i < m; m_i++) {
      for (int n_i = n_start; n_i < n_end; n_i += 32) {
        // Load from BufferC (FP32)
        float* c_ptr = bc->get_submat(m, n, m_i, n_i);

        // Load from main output (BF16)
        __m512 main0, main1;
        avx512_32xbf16_to_32xfp32((__m512i*)(main_output + m_i * n + n_i), &main0, &main1);

        // Add with scaling
        __m512 scale = _mm512_set1_ps(scaling);
        __m512 lora0 = _mm512_load_ps(c_ptr);
        __m512 lora1 = _mm512_load_ps(c_ptr + 16);
        main0 = _mm512_fmadd_ps(lora0, scale, main0);
        main1 = _mm512_fmadd_ps(lora1, scale, main1);

        // Store back to main output (BF16)
        avx512_32xfp32_to_32xbf16(&main0, &main1, (__m512i*)(main_output + m_i * n + n_i));
      }
    }
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

          // Use thread-local intermediate buffer to avoid race conditions
          // This completely eliminates the shared buffer contention issue
          std::vector<float> local_intermediate(num_tokens * lora_rank_);

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
              local_intermediate[t * lora_rank_ + r] = sum;
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [intermediate_size, lora_rank]^T → [num_tokens, intermediate_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int i = 0; i < config_.intermediate_size; i++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = local_intermediate[t * lora_rank_ + r];
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

          // Use thread-local intermediate buffer to avoid race conditions
          std::vector<float> local_intermediate(num_tokens * lora_rank_);

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
              local_intermediate[t * lora_rank_ + r] = sum;
            }
          }

          // Step 2: lora_out = intermediate @ lora_B^T, add to output
          // [num_tokens, lora_rank] @ [hidden_size, lora_rank]^T → [num_tokens, hidden_size]
          for (int t = 0; t < num_tokens; t++) {
            for (int h = 0; h < config_.hidden_size; h++) {
              float sum = 0.0f;
              for (int r = 0; r < lora_rank_; r++) {
                float inter = local_intermediate[t * lora_rank_ + r];
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
                     int activated_expert, const void* input) {
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

    // Copy input (original input, not m_local_input_ which is expert-sorted)
    // Bug #17b fix: backward_gate_up expects original token order
    memcpy(cache.input_cache, input, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

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

    // Debug code commented out - Bug #15 verified fixed
    // printf("[DEBUG save_to_cache] total_tokens=%zu\n", offset);
    // printf("[DEBUG ADDR] cache.gate_output_cache = %p, cache.up_output_cache = %p\n", (void*)cache.gate_output_cache,
    //        (void*)cache.up_output_cache);
    // printf("[DEBUG save_to_cache] gate_output_cache[0..7] = ");
    // for (int i = 0; i < 8 && i < (int)(offset * config_.intermediate_size); i++) {
    //   printf("%.4f ", GGML_BF16_TO_FP32(cache.gate_output_cache[i]));
    // }
    // printf("\n");

    cache.valid = true;
  }

  /**
   * @brief Save intermediate values AFTER activation for backward_down.
   *
   * Must be called after apply_activation() since m_local_gate_output_ptr_
   * now contains silu(gate) * up (the intermediate value).
   */
  void save_intermediate_to_cache(ForwardCache& cache, int activated_expert) {
    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = m_expert_id_map_[i];
      int num_tokens = m_local_num_[expert_idx];
      // m_local_gate_output_ptr_ now contains intermediate (after activation: silu(gate) * up)
      memcpy(cache.intermediate_cache + offset * config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
             num_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
      offset += num_tokens;
    }
  }

  void backward_down(const ForwardCache& cache, const void* grad_output, void* grad_down_lora_a,
                     void* grad_down_lora_b) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Debug code commented out - Bug #15 verified fixed
    // printf("[DEBUG ADDR backward_down] grad_intermediate_ = %p\n", (void*)grad_intermediate_);
    // printf("[DEBUG ADDR backward_down] cache.gate_output_cache = %p\n", (void*)cache.gate_output_cache);
    // printf("[DEBUG ADDR backward_down] cache.up_output_cache = %p\n", (void*)cache.up_output_cache);
    // printf("[DEBUG BEFORE memset] gate_cache[0..3] = %.4f %.4f %.4f %.4f\n",
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[0]), GGML_BF16_TO_FP32(cache.gate_output_cache[1]),
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[2]), GGML_BF16_TO_FP32(cache.gate_output_cache[3]));

    // Initialize gradient intermediate buffer
    memset(grad_intermediate_, 0,
           config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));

    // printf("[DEBUG AFTER memset] gate_cache[0..3] = %.4f %.4f %.4f %.4f\n",
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[0]), GGML_BF16_TO_FP32(cache.gate_output_cache[1]),
    //        GGML_BF16_TO_FP32(cache.gate_output_cache[2]), GGML_BF16_TO_FP32(cache.gate_output_cache[3]));

    // Scatter grad_output to per-expert buffers and compute gradients
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, grad_output, grad_down_a, grad_down_b, qlen, k](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Collect gradients for this expert from grad_output
          // grad_output is [qlen, hidden_size] in bf16, need to scatter based on routing
          const ggml_bf16_t* grad_out_bf16 = (const ggml_bf16_t*)grad_output;
          std::vector<float> expert_grad_out(num_tokens * config_.hidden_size, 0.0f);

          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                float w = cache.weights_cache[i * k + j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  expert_grad_out[pos * config_.hidden_size + h] +=
                      GGML_BF16_TO_FP32(grad_out_bf16[i * config_.hidden_size + h]) * w;
                }
              }
            }
          }

          // Get cached intermediate (after activation)
          ggml_bf16_t* intermediate = cache.intermediate_cache;  // Will use gate_output_cache after activation saved

          // Compute grad w.r.t. intermediate: grad_intermediate = grad_output @ down_proj
          // down_proj layout: [expert_num, hidden_size, intermediate_size]
          // grad_output: [num_tokens, hidden_size], grad_intermediate: [num_tokens, intermediate_size]
          // grad_intermediate[t, i] = sum_h grad_output[t, h] * down_proj[h, i]
          {
            const ggml_bf16_t* down_proj = (const ggml_bf16_t*)config_.down_proj;
            size_t expert_offset = (size_t)expert_idx * config_.hidden_size * config_.intermediate_size;

            // Compute offset into grad_intermediate_ for this expert
            size_t grad_inter_offset = 0;
            for (int e = 0; e < task_id; e++) {
              grad_inter_offset += m_local_num_[m_expert_id_map_[e]];
            }
            grad_inter_offset *= config_.intermediate_size;

            for (int t = 0; t < num_tokens; t++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int h = 0; h < config_.hidden_size; h++) {
                  float grad_out_val = expert_grad_out[t * config_.hidden_size + h];
                  float down_val = GGML_BF16_TO_FP32(down_proj[expert_offset + h * config_.intermediate_size + i]);
                  sum += grad_out_val * down_val;
                }
                grad_intermediate_[grad_inter_offset + t * config_.intermediate_size + i] = GGML_FP32_TO_BF16(sum);
              }
            }
          }

          if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
            // Get expert's LoRA weights
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
            size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
            ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

            // Bug #17c fix: Use cached intermediate (after activation), not gate_output_cache (before activation)
            // The cache is stored in task order (activated expert order), need to compute offset
            size_t cache_offset = 0;
            for (int e = 0; e < task_id; e++) {
              cache_offset += m_local_num_[m_expert_id_map_[e]];
            }
            const ggml_bf16_t* cached_intermediate =
                cache.intermediate_cache + cache_offset * config_.intermediate_size;

            // Gradient for LoRA B: grad_B = grad_output^T @ (intermediate @ lora_A^T) * scaling
            // = (grad_output^T @ intermediate @ lora_A^T) * scaling
            // Shape: [hidden_size, num_tokens] @ [num_tokens, lora_rank] → [hidden_size, lora_rank]

            // First compute intermediate @ lora_A^T → [num_tokens, lora_rank]
            std::vector<float> inter_proj(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  // Use cached intermediate (gate_output after activation)
                  float inp = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
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
                size_t idx = lora_b_offset + h * lora_rank_ + r;
                float current = GGML_BF16_TO_FP32(grad_down_b[idx]);
                grad_down_b[idx] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
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
                  // Bug #17a fix: Use cached_intermediate instead of m_local_gate_output_ptr_
                  float inter = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
                  sum += inter * grad_times_b[t * lora_rank_ + r];
                }
                size_t idx_a = lora_a_offset + r * config_.intermediate_size + i;
                float current = GGML_BF16_TO_FP32(grad_down_a[idx_a]);
                grad_down_a[idx_a] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }
          }
        },
        nullptr);
  }

  /**
   * @brief AMX-optimized backward pass for down projection.
   *
   * Optimizes the main GEMM: grad_intermediate = grad_output @ down_proj
   * Using AMX mat_mul with down_backward_bb_ (transposed weight).
   *
   * LoRA gradient computation is kept as for-loop for now due to complexity
   * and small matrix sizes involved.
   */
  void backward_down_amx(const ForwardCache& cache, const void* grad_output, void* grad_down_lora_a,
                         void* grad_down_lora_b) {
    // Timing for backward_down_amx substeps
    static int _down_call_count = 0;
    _down_call_count++;
    bool _down_should_print = (_down_call_count == get_print_at_call());
    auto _down_start = std::chrono::high_resolution_clock::now();
    auto _down_last = _down_start;

#define DOWN_CHECKPOINT(name)                                                                                       \
  do {                                                                                                              \
    auto _now = std::chrono::high_resolution_clock::now();                                                          \
    if (_down_should_print) {                                                                                       \
      printf("  [DOWN] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _down_last).count()); \
    }                                                                                                               \
    _down_last = _now;                                                                                              \
  } while (0)

    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_down_a = (ggml_bf16_t*)grad_down_lora_a;
    ggml_bf16_t* grad_down_b = (ggml_bf16_t*)grad_down_lora_b;

    // Ensure backward weights are prepared
    prepare_backward_weights();

    // =====================================================
    // Bug-C Fix Step 2: Allocate backward buffers from shared pool
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    char* backward_ba_ptr = (char*)backward_ba_pool_;
    char* backward_bc_ptr = (char*)backward_bc_pool_;
    char* grad_output_bf16_ptr = (char*)grad_output_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferA for grad_output
      grad_output_ba_[expert_idx]->max_m = local_max_m;
      grad_output_ba_[expert_idx]->set_data(backward_ba_ptr);
      backward_ba_ptr += align64(T::BufferA::required_size(local_max_m, config_.hidden_size));

      // Allocate BufferC for grad_intermediate
      grad_intermediate_bc_[expert_idx]->max_m = local_max_m;
      grad_intermediate_bc_[expert_idx]->set_data(backward_bc_ptr);
      backward_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));

      // Allocate BF16 buffer for scattered grad_output
      grad_output_bf16_ptr_[expert_idx] = (ggml_bf16_t*)grad_output_bf16_ptr;
      grad_output_bf16_ptr += align64(local_max_m * config_.hidden_size * sizeof(ggml_bf16_t));
    }

    // Initialize gradient intermediate buffer
    memset(grad_intermediate_, 0,
           config_.max_len * config_.num_experts_per_tok * config_.intermediate_size * sizeof(ggml_bf16_t));

    DOWN_CHECKPOINT("D0_prepare+memset");

    // =====================================================
    // Step 1: Scatter grad_output to per-expert BF16 buffers
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, &cache, grad_output, qlen, k](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          const ggml_bf16_t* grad_out_bf16 = (const ggml_bf16_t*)grad_output;
          ggml_bf16_t* expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];

          // Zero the buffer first
          memset(expert_grad_bf16, 0, num_tokens * config_.hidden_size * sizeof(ggml_bf16_t));

          // Scatter grad_output to expert's buffer (with routing weights)
          for (int i = 0; i < qlen; i++) {
            for (int j = 0; j < k; j++) {
              if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                int pos = cache.m_local_pos_cache[i][j];
                float w = cache.weights_cache[i * k + j];
                for (int h = 0; h < config_.hidden_size; h++) {
                  float val = GGML_BF16_TO_FP32(expert_grad_bf16[pos * config_.hidden_size + h]);
                  val += GGML_BF16_TO_FP32(grad_out_bf16[i * config_.hidden_size + h]) * w;
                  expert_grad_bf16[pos * config_.hidden_size + h] = GGML_FP32_TO_BF16(val);
                }
              }
            }
          }
        },
        nullptr);

    DOWN_CHECKPOINT("D1_scatter");

    // =====================================================
    // Step 2: Quantize scattered grad_output to BufferA
    // =====================================================
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int num_tokens = m_local_num_[expert_idx];

          if (num_tokens == 0) return;

          // Convert scattered grad_output to BufferA
          grad_output_ba_[expert_idx]->from_mat(num_tokens, grad_output_bf16_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    DOWN_CHECKPOINT("D2_quantize");

    // =====================================================
    // Step 3+4: AMX GEMM + to_mat (merged to use same ith/nth)
    // grad_intermediate = grad_output @ down_proj
    // Using: A @ B^T where A = grad_output, B = down_proj^T (stored in down_backward_bb_)
    // m = num_tokens, n = intermediate_size, k = hidden_size
    //
    // BUG FIX: Previously Step 3 used (ith, nth) for mat_mul but Step 4 used (0, 1) for to_mat,
    // which only output the first N_BLOCK columns. Now merged to use same (ith, nth).
    // =====================================================
    int nth = T::recommended_nth(config_.intermediate_size);

    // Pre-compute offsets for each expert (needed for to_mat output location)
    std::vector<size_t> expert_offsets(activated_expert);
    {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        expert_offsets[i] = offset * config_.intermediate_size;
        offset += m_local_num_[m_expert_id_map_[i]];
      }
    }

    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, &expert_offsets](int task_id) {
          int task_idx = task_id / nth;  // Which expert (0 to activated_expert-1)
          int expert_idx = m_expert_id_map_[task_idx];
          int ith = task_id % nth;
          int m = m_local_num_[expert_idx];

          if (m == 0) return;

          auto& ba = grad_output_ba_[expert_idx];
          auto& bb = down_backward_bb_[expert_idx];
          auto& bc = grad_intermediate_bc_[expert_idx];

          // mat_mul: [m, hidden_size] @ [intermediate_size, hidden_size]^T = [m, intermediate_size]
          amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);

          // to_mat: Convert BufferC to BF16 - use same ith, nth as mat_mul!
          bc->to_mat(m, grad_intermediate_ + expert_offsets[task_idx], ith, nth);
        },
        nullptr);

    DOWN_CHECKPOINT("D3_gemm");

    // =====================================================
    // Step 5: LoRA gradient computation (kept as for-loop due to small matrices)
    // =====================================================
    if (down_lora_a_ != nullptr && down_lora_b_ != nullptr) {
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [this, &cache, grad_down_a, grad_down_b](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];

            if (num_tokens == 0) return;

            // Get expert's grad_output (already scattered)
            ggml_bf16_t* expert_grad_bf16 = grad_output_bf16_ptr_[expert_idx];

            // Get expert's LoRA weights
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.intermediate_size;
            size_t lora_b_offset = expert_idx * config_.hidden_size * lora_rank_;
            ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
            ggml_bf16_t* expert_lora_b = down_lora_b_ + lora_b_offset;

            // Get cached intermediate
            size_t cache_offset = 0;
            for (int e = 0; e < task_id; e++) {
              cache_offset += m_local_num_[m_expert_id_map_[e]];
            }
            const ggml_bf16_t* cached_intermediate =
                cache.intermediate_cache + cache_offset * config_.intermediate_size;

            // Convert expert_grad_bf16 to float for computation
            std::vector<float> expert_grad_out(num_tokens * config_.hidden_size);
            for (int i = 0; i < num_tokens * config_.hidden_size; i++) {
              expert_grad_out[i] = GGML_BF16_TO_FP32(expert_grad_bf16[i]);
            }

            // Gradient for LoRA B: grad_B = grad_output^T @ (intermediate @ lora_A^T) * scaling
            // First compute intermediate @ lora_A^T → [num_tokens, lora_rank]
            std::vector<float> inter_proj(num_tokens * lora_rank_, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  float inp = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
                  float w = GGML_BF16_TO_FP32(expert_lora_a[r * config_.intermediate_size + i]);
                  sum += inp * w;
                }
                inter_proj[t * lora_rank_ + r] = sum;
              }
            }

            // grad_B = grad_output^T @ inter_proj * scaling
            for (int h = 0; h < config_.hidden_size; h++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  sum += expert_grad_out[t * config_.hidden_size + h] * inter_proj[t * lora_rank_ + r];
                }
                size_t idx = lora_b_offset + h * lora_rank_ + r;
                float current = GGML_BF16_TO_FP32(grad_down_b[idx]);
                grad_down_b[idx] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }

            // Gradient for LoRA A: grad_A = intermediate^T @ grad_output @ lora_B * scaling
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
            for (int r = 0; r < lora_rank_; r++) {
              for (int i = 0; i < config_.intermediate_size; i++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  float inter = GGML_BF16_TO_FP32(cached_intermediate[t * config_.intermediate_size + i]);
                  sum += inter * grad_times_b[t * lora_rank_ + r];
                }
                size_t idx_a = lora_a_offset + r * config_.intermediate_size + i;
                float current = GGML_BF16_TO_FP32(grad_down_a[idx_a]);
                grad_down_a[idx_a] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }
          },
          nullptr);
      DOWN_CHECKPOINT("D4_lora_grad");
    }

#undef DOWN_CHECKPOINT
  }

  void backward_activation(const ForwardCache& cache) {
    // Timing for backward_activation
    static int _act_call_count = 0;
    _act_call_count++;
    bool _act_should_print = (_act_call_count == get_print_at_call());
    auto _act_start = std::chrono::high_resolution_clock::now();

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

          // Debug code commented out - Bug #15 verified fixed
          // if (task_id == 0) {
          //   printf("[DEBUG backward_activation] task_id=0, expert_idx=%d, num_tokens=%d, offset=%zu\n", expert_idx,
          //          num_tokens, offset);
          //   printf("[DEBUG] gate_output[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(gate_output[dbg]));
          //   }
          //   printf("\n");
          //   printf("[DEBUG] up_output[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(up_output[dbg]));
          //   }
          //   printf("\n");
          //   printf("[DEBUG] grad_inter[0..7] = ");
          //   for (int dbg = 0; dbg < 8 && dbg < num_tokens * config_.intermediate_size; dbg++) {
          //     printf("%.4f ", GGML_BF16_TO_FP32(grad_inter[dbg]));
          //   }
          //   printf("\n");
          // }

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

    if (_act_should_print) {
      auto _act_end = std::chrono::high_resolution_clock::now();
      printf("  [ACT] silu_backward: %.3f ms\n",
             std::chrono::duration<double, std::milli>(_act_end - _act_start).count());
    }
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

    // Initialize grad_input to zero (bf16 type)
    memset(grad_input, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

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

          // First, compute base weight contribution to grad_input (always, regardless of LoRA)
          // grad_input += grad @ W^T (for gate or up, depending on do_up)
          // W layout: [expert_num, intermediate_size, hidden_size]
          // grad: [num_tokens, intermediate_size]
          // grad_input[t, h] += sum_i grad[t, i] * W[i, h]
          {
            ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
            const ggml_bf16_t* base_proj =
                do_up ? (const ggml_bf16_t*)config_.up_proj : (const ggml_bf16_t*)config_.gate_proj;
            size_t expert_offset = (size_t)expert_idx * config_.intermediate_size * config_.hidden_size;

            // Pre-compute grad_input contribution per token, then scatter
            std::vector<float> token_grad_input(num_tokens * config_.hidden_size, 0.0f);
            for (int t = 0; t < num_tokens; t++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                for (int i = 0; i < config_.intermediate_size; i++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  float w = GGML_BF16_TO_FP32(base_proj[expert_offset + i * config_.hidden_size + h]);
                  sum += g * w;
                }
                token_grad_input[t * config_.hidden_size + h] = sum;
              }
            }

            // Scatter back to grad_input
            for (int i = 0; i < qlen; i++) {
              for (int j = 0; j < k; j++) {
                if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                  int pos = cache.m_local_pos_cache[i][j];
                  for (int h = 0; h < config_.hidden_size; h++) {
                    float current = GGML_BF16_TO_FP32(grad_input_bf16[i * config_.hidden_size + h]);
                    grad_input_bf16[i * config_.hidden_size + h] =
                        GGML_FP32_TO_BF16(current + token_grad_input[pos * config_.hidden_size + h]);
                  }
                }
              }
            }
          }

          // LoRA gradients and contribution - only if LoRA is enabled
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
          ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
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
                  // Read current value, add, and write back as bf16
                  float current = GGML_BF16_TO_FP32(grad_input_bf16[i * config_.hidden_size + h]);
                  grad_input_bf16[i * config_.hidden_size + h] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
                }
              }
            }
          }
        },
        nullptr);
  }

  /**
   * @brief AMX-optimized backward pass for gate and up projections.
   *
   * Uses AMX GEMM for base weight contribution and LoRA grad_input. LoRA weight gradients
   * remain small for-loops.
   */
  void backward_gate_up_amx(const ForwardCache& cache, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                            void* grad_up_lora_a, void* grad_up_lora_b) {
    // Timing for backward_gate_up_amx substeps (counter-based)
    static int _gu_call_count = 0;
    _gu_call_count++;
    bool _gu_should_print = (_gu_call_count == get_print_at_call());
    auto _gu_start = std::chrono::high_resolution_clock::now();
    auto _gu_last = _gu_start;

#define GU_CHECKPOINT(name)                                                                                     \
  do {                                                                                                          \
    auto _now = std::chrono::high_resolution_clock::now();                                                      \
    if (_gu_should_print) {                                                                                     \
      printf("  [GU] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _gu_last).count()); \
    }                                                                                                           \
    _gu_last = _now;                                                                                            \
  } while (0)

    auto pool = config_.pool->get_subpool(tp_part_idx);
    int activated_expert = cache.activated_expert_cache;
    int qlen = cache.qlen_cache;
    int k = cache.k_cache;

    ggml_bf16_t* grad_gate_a = (ggml_bf16_t*)grad_gate_lora_a;
    ggml_bf16_t* grad_gate_b = (ggml_bf16_t*)grad_gate_lora_b;
    ggml_bf16_t* grad_up_a = (ggml_bf16_t*)grad_up_lora_a;
    ggml_bf16_t* grad_up_b = (ggml_bf16_t*)grad_up_lora_b;

    prepare_backward_weights();
    if (gate_lora_a_ != nullptr && gate_lora_b_ != nullptr) {
      prepare_lora_weights();
    }

    // =====================================================
    // Bug-C Fix Step 2: Allocate backward buffers from shared pool
    // Note: backward_down_amx already allocated grad_output_ba_ and grad_intermediate_bc_
    // Here we need grad_gate_up_bc_ which uses the remaining part of backward_bc_pool_
    // =====================================================
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };

    // Calculate offset after grad_intermediate_bc_ allocations
    size_t grad_intermediate_total = 0;
    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      size_t local_max_m = ((m_local_num_[expert_idx] + M_STEP - 1) / M_STEP) * M_STEP;
      grad_intermediate_total += align64(T::BufferC::required_size(local_max_m, config_.intermediate_size));
    }

    char* grad_gate_up_bc_ptr = (char*)backward_bc_pool_ + grad_intermediate_total;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // Allocate BufferC for grad_gate_up
      grad_gate_up_bc_[expert_idx]->max_m = local_max_m;
      grad_gate_up_bc_[expert_idx]->set_data(grad_gate_up_bc_ptr);
      grad_gate_up_bc_ptr += align64(T::BufferC::required_size(local_max_m, config_.hidden_size));
    }

    // Allocate LoRA intermediate buffers from shared pools (for LoRA backward pass)
    char* lora_ba_ptr = (char*)lora_ba_pool_;
    char* lora_bc_inter_ptr = (char*)lora_bc_inter_pool_;
    char* bf16_inter_ptr = (char*)lora_intermediate_bf16_pool_;

    for (int task_id = 0; task_id < activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id];
      int m = m_local_num_[expert_idx];
      if (m == 0) continue;

      size_t local_max_m = ((m + M_STEP - 1) / M_STEP) * M_STEP;

      // BufferA for LoRA intermediate (gate)
      lora_gate_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // BufferA for LoRA intermediate (up)
      lora_up_intermediate_ba_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_ba_[expert_idx]->set_data(lora_ba_ptr);
      lora_ba_ptr += align64(T::BufferA::required_size(local_max_m, padded_lora_rank_));

      // BufferC for LoRA step 1 output (gate)
      lora_gate_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_gate_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // BufferC for LoRA step 1 output (up)
      lora_up_intermediate_bc_[expert_idx]->max_m = local_max_m;
      lora_up_intermediate_bc_[expert_idx]->set_data(lora_bc_inter_ptr);
      lora_bc_inter_ptr += align64(T::BufferC::required_size(local_max_m, padded_lora_rank_));

      // BF16 intermediate pointers (gate)
      lora_gate_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));

      // BF16 intermediate pointers (up)
      lora_up_intermediate_ptr_[expert_idx] = (ggml_bf16_t*)bf16_inter_ptr;
      bf16_inter_ptr += align64(local_max_m * padded_lora_rank_ * sizeof(ggml_bf16_t));
    }

    memset(grad_input, 0, qlen * config_.hidden_size * sizeof(ggml_bf16_t));

    GU_CHECKPOINT("GU0_prepare+memset");

    // Offsets into contiguous grad_gate/up buffers
    std::vector<size_t> expert_offsets(activated_expert);
    {
      size_t offset = 0;
      for (int i = 0; i < activated_expert; i++) {
        expert_offsets[i] = offset;
        offset += m_local_num_[m_expert_id_map_[i]];
      }
    }

    auto scatter_to_grad_input = [&](float scale) {
      ggml_bf16_t* grad_input_bf16 = (ggml_bf16_t*)grad_input;
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, scale](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t* contrib = grad_output_bf16_ptr_[expert_idx];

            for (int i = 0; i < qlen; i++) {
              for (int j = 0; j < k; j++) {
                if (cache.expert_ids_cache[i * k + j] == expert_idx) {
                  int pos = cache.m_local_pos_cache[i][j];
                  for (int h = 0; h < config_.hidden_size; h++) {
                    float current = GGML_BF16_TO_FP32(grad_input_bf16[i * config_.hidden_size + h]);
                    float add = GGML_BF16_TO_FP32(contrib[pos * config_.hidden_size + h]) * scale;
                    grad_input_bf16[i * config_.hidden_size + h] = GGML_FP32_TO_BF16(current + add);
                  }
                }
              }
            }
          },
          nullptr);
    };

    auto base_pass = [&](bool do_up) {
      auto _bp_start = std::chrono::high_resolution_clock::now();

      // Quantize grad to BufferA
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            down_ba_[expert_idx]->from_mat(m, grad, 0, 1);
          },
          nullptr);

      int nth = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth](int task_id) {
            int task_idx = task_id / nth;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = down_ba_[expert_idx];
            auto& bb = do_up ? up_backward_bb_[expert_idx] : gate_backward_bb_[expert_idx];
            auto& bc = grad_gate_up_bc_[expert_idx];

            amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
            bc->to_mat(m, grad_output_bf16_ptr_[expert_idx], ith, nth);
          },
          nullptr);

      scatter_to_grad_input(1.0f);

      if (_gu_should_print) {
        auto _bp_end = std::chrono::high_resolution_clock::now();
        printf("  [GU] base_pass(%s): %.3f ms\n", do_up ? "up" : "gate",
               std::chrono::duration<double, std::milli>(_bp_end - _bp_start).count());
      }
    };

    base_pass(false);  // gate
    base_pass(true);   // up
    GU_CHECKPOINT("GU1_base_passes_total");

    if (gate_lora_a_ == nullptr || gate_lora_b_ == nullptr) {
#undef GU_CHECKPOINT
      return;
    }

// Re-define GU_CHECKPOINT for the rest of the function
#define GU_CHECKPOINT(name)                                                                                     \
  do {                                                                                                          \
    auto _now = std::chrono::high_resolution_clock::now();                                                      \
    if (_gu_should_print) {                                                                                     \
      printf("  [GU] %s: %.3f ms\n", name, std::chrono::duration<double, std::milli>(_now - _gu_last).count()); \
    }                                                                                                           \
    _gu_last = _now;                                                                                            \
  } while (0)

    // Re-quantize inputs for LoRA path
    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          int m = m_local_num_[expert_idx];
          if (m == 0) return;
          gate_up_ba_[expert_idx]->from_mat(m, m_local_input_ptr_[expert_idx], 0, 1);
        },
        nullptr);

    GU_CHECKPOINT("GU2_requantize_for_lora");

    auto lora_pass = [&](bool do_up) {
      auto _lp_start = std::chrono::high_resolution_clock::now();
      // Step 1: input @ lora_A^T -> U
      int nth_inter = T::recommended_nth(padded_lora_rank_);
      pool->do_work_stealing_job(
          nth_inter * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_inter](int task_id) {
            int task_idx = task_id / nth_inter;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_inter;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = gate_up_ba_[expert_idx];
            auto& bb = do_up ? up_lora_a_bb_[expert_idx] : gate_lora_a_bb_[expert_idx];
            auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];
            ggml_bf16_t* inter_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            amx::mat_mul(m, padded_lora_rank_, config_.hidden_size, ba, bb, bc, ith, nth_inter);
            bc->to_mat(m, inter_ptr, ith, nth_inter);
          },
          nullptr);

      // Step 2: grad_B = grad^T @ U
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            ggml_bf16_t* grad_lora_b = do_up ? grad_up_b : grad_gate_b;
            size_t lora_b_offset = expert_idx * config_.intermediate_size * lora_rank_;
            ggml_bf16_t* u_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            for (int i = 0; i < config_.intermediate_size; i++) {
              for (int r = 0; r < lora_rank_; r++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  float g = GGML_BF16_TO_FP32(grad[t * config_.intermediate_size + i]);
                  float u = GGML_BF16_TO_FP32(u_ptr[t * padded_lora_rank_ + r]);
                  sum += g * u;
                }
                float current = GGML_BF16_TO_FP32(grad_lora_b[lora_b_offset + i * lora_rank_ + r]);
                grad_lora_b[lora_b_offset + i * lora_rank_ + r] = GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }
          },
          nullptr);

      // Step 3: grad @ lora_B -> G_B
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            size_t offset = expert_offsets[task_id];
            ggml_bf16_t* grad = do_up ? (grad_up_output_ + offset * config_.intermediate_size)
                                      : (grad_gate_output_ + offset * config_.intermediate_size);
            down_ba_[expert_idx]->from_mat(m, grad, 0, 1);
          },
          nullptr);

      int nth_gb = T::recommended_nth(padded_lora_rank_);
      pool->do_work_stealing_job(
          nth_gb * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_gb](int task_id) {
            int task_idx = task_id / nth_gb;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_gb;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = down_ba_[expert_idx];
            auto& bb = do_up ? up_lora_b_t_bb_[expert_idx] : gate_lora_b_t_bb_[expert_idx];
            auto& bc = do_up ? lora_up_intermediate_bc_[expert_idx] : lora_gate_intermediate_bc_[expert_idx];
            ggml_bf16_t* inter_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];

            amx::mat_mul(m, padded_lora_rank_, config_.intermediate_size, ba, bb, bc, ith, nth_gb);
            bc->to_mat(m, inter_ptr, ith, nth_gb);
          },
          nullptr);

      // Step 4: Quantize G_B
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            ggml_bf16_t* g_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
            ba->from_mat(m, g_ptr, 0, 1);
          },
          nullptr);

      // Step 5: G_B @ lora_A -> grad_input (LoRA part)
      int nth_input = T::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth_input * activated_expert, [](int _) { T::config(); },
          [&, do_up, nth_input](int task_id) {
            int task_idx = task_id / nth_input;
            int expert_idx = m_expert_id_map_[task_idx];
            int ith = task_id % nth_input;
            int m = m_local_num_[expert_idx];
            if (m == 0) return;

            auto& ba = do_up ? lora_up_intermediate_ba_[expert_idx] : lora_gate_intermediate_ba_[expert_idx];
            auto& bb = do_up ? up_lora_a_t_bb_[expert_idx] : gate_lora_a_t_bb_[expert_idx];
            auto& bc = grad_gate_up_bc_[expert_idx];

            amx::mat_mul(m, config_.hidden_size, padded_lora_rank_, ba, bb, bc, ith, nth_input);
            bc->to_mat(m, grad_output_bf16_ptr_[expert_idx], ith, nth_input);
          },
          nullptr);

      scatter_to_grad_input(lora_scaling_);

      // Step 6: grad_A = input^T @ G_B
      pool->do_work_stealing_job(
          activated_expert, nullptr,
          [&, do_up](int task_id) {
            int expert_idx = m_expert_id_map_[task_id];
            int num_tokens = m_local_num_[expert_idx];
            if (num_tokens == 0) return;

            ggml_bf16_t* g_ptr =
                do_up ? lora_up_intermediate_ptr_[expert_idx] : lora_gate_intermediate_ptr_[expert_idx];
            ggml_bf16_t* grad_lora_a = do_up ? grad_up_a : grad_gate_a;
            size_t lora_a_offset = expert_idx * lora_rank_ * config_.hidden_size;
            ggml_bf16_t* expert_input = m_local_input_ptr_[expert_idx];

            for (int r = 0; r < lora_rank_; r++) {
              for (int h = 0; h < config_.hidden_size; h++) {
                float sum = 0.0f;
                for (int t = 0; t < num_tokens; t++) {
                  float gb = GGML_BF16_TO_FP32(g_ptr[t * padded_lora_rank_ + r]);
                  float inp = GGML_BF16_TO_FP32(expert_input[t * config_.hidden_size + h]);
                  sum += inp * gb;
                }
                float current = GGML_BF16_TO_FP32(grad_lora_a[lora_a_offset + r * config_.hidden_size + h]);
                grad_lora_a[lora_a_offset + r * config_.hidden_size + h] =
                    GGML_FP32_TO_BF16(current + sum * lora_scaling_);
              }
            }
          },
          nullptr);

      if (_gu_should_print) {
        auto _lp_end = std::chrono::high_resolution_clock::now();
        printf("  [GU] lora_pass(%s): %.3f ms\n", do_up ? "up" : "gate",
               std::chrono::duration<double, std::milli>(_lp_end - _lp_start).count());
      }
    };

    lora_pass(false);  // gate
    lora_pass(true);   // up
    GU_CHECKPOINT("GU3_lora_passes_total");

#undef GU_CHECKPOINT
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_MOE_H
