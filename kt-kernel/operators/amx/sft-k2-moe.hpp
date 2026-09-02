/**
 * @Description  : K2 AMX MoE SFT operator entry point.
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * K2-specific SFT lives here as a sibling of the generic sft_moe.hpp path.
 * The implementation borrows the SFT LoRA/cache structure from sft_moe.hpp
 * and the KGroup packed int4 load/GEMM contract from k2-moe.hpp.
 **/
#ifndef CPUINFER_OPERATOR_AMX_SFT_K2_MOE_H
#define CPUINFER_OPERATOR_AMX_SFT_K2_MOE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2-moe.hpp"
#include "la/avx_kernels.hpp"
#include "../sft_profile.hpp"

/**
 * @brief K2 RAWINT4 SFT MoE operator.
 *
 * This class is intentionally separate from AMX_SFT_MOE_TP in sft_moe.hpp
 * because K2 uses pre-packed signed int4 weights with BF16 group scales.
 * Forward/cache can reuse the SFT structure, but base-weight backward must be
 * rewritten to read the packed int4 tensors directly instead of relying on
 * BF16 shadow weights or standard transposed BufferB objects.
 */
template <class T = amx::GemmKernel224Int4SmallKGroup, bool SkipLoRA = false>
class AMX_K2_SFT_MOE_TP : public AMX_K2_MOE_TP<T> {
 protected:
  using Base = AMX_K2_MOE_TP<T>;
  using Base::config_;
  using Base::tp_part_idx;

  MOESFTConfig sft_config_;

  int lora_rank_ = 0;
  float lora_scaling_ = 0.0f;
  int max_cache_depth_ = 1;

  ggml_bf16_t* gate_lora_a_ = nullptr;
  ggml_bf16_t* gate_lora_b_ = nullptr;
  ggml_bf16_t* up_lora_a_ = nullptr;
  ggml_bf16_t* up_lora_b_ = nullptr;
  ggml_bf16_t* down_lora_a_ = nullptr;
  ggml_bf16_t* down_lora_b_ = nullptr;

  bool k2_packed_weights_loaded_ = false;
  SFTProfiler profiler_;

  struct K2ForwardCache {
    ggml_bf16_t* input_cache = nullptr;
    ggml_bf16_t* gate_output_cache = nullptr;
    ggml_bf16_t* up_output_cache = nullptr;
    ggml_bf16_t* intermediate_cache = nullptr;
    ggml_bf16_t* down_output_cache = nullptr;
    float* down_lora_u_cache = nullptr;
    float* gate_lora_u_cache = nullptr;
    float* up_lora_u_cache = nullptr;

    std::vector<ggml_bf16_t> input_storage;
    std::vector<ggml_bf16_t> gate_output_storage;
    std::vector<ggml_bf16_t> up_output_storage;
    std::vector<ggml_bf16_t> intermediate_storage;
    std::vector<ggml_bf16_t> down_output_storage;
    std::vector<float> down_lora_u_storage;
    std::vector<float> gate_lora_u_storage;
    std::vector<float> up_lora_u_storage;

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

  std::vector<K2ForwardCache> cache_stack_;
  int cache_stack_top_ = 0;
  std::vector<size_t> cache_offsets_;
  mutable std::vector<ggml_bf16_t> gate_up_route_grad_scratch_;

  std::vector<ggml_bf16_t> gate_lora_b_transposed_;
  std::vector<ggml_bf16_t> up_lora_b_transposed_;
  std::vector<ggml_bf16_t> down_lora_b_transposed_;
  bool lora_b_transposed_ = false;

  static inline size_t align64(size_t v) { return (v + 63) & (~static_cast<size_t>(63)); }

  bool has_gate_up_lora() const {
    return !SkipLoRA && lora_rank_ > 0 && gate_lora_a_ != nullptr && gate_lora_b_ != nullptr && up_lora_a_ != nullptr &&
           up_lora_b_ != nullptr;
  }

  bool has_down_lora() const {
    return !SkipLoRA && lora_rank_ > 0 && down_lora_a_ != nullptr && down_lora_b_ != nullptr;
  }

  bool has_any_lora() const { return has_gate_up_lora() || has_down_lora(); }

  static void validate_k2_kgroup_contract(const GeneralMOEConfig& config) {
    const auto& quant_config = config.quant_config;
    if (quant_config.bits != 4 || quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error(
          "K2 RAWINT4 SFT requires signed int4 KGroup weights with group_size=32 and no zero point");
    }
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.expert_num <= 0) {
      throw std::runtime_error("K2 RAWINT4 SFT received invalid MoE dimensions");
    }
    if (config.hidden_size % quant_config.group_size != 0 ||
        config.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error("K2 RAWINT4 SFT hidden and intermediate dimensions must be group-32 aligned");
    }
  }

  static GeneralMOEConfig validated_base_config(const MOESFTConfig& config) {
    GeneralMOEConfig base_config = static_cast<const GeneralMOEConfig&>(config);
    validate_k2_kgroup_contract(base_config);
    return base_config;
  }

  void validate_k2_kgroup_contract() const { validate_k2_kgroup_contract(config_); }

  void clear_transposed_lora_weights() {
    gate_lora_b_transposed_.clear();
    up_lora_b_transposed_.clear();
    down_lora_b_transposed_.clear();
    lora_b_transposed_ = false;
  }

  void prepare_lora_b_transposed() {
    if constexpr (SkipLoRA) {
      return;
    }
    if (lora_rank_ <= 0 || lora_b_transposed_) return;
    if (!has_any_lora()) return;

    const size_t gate_up_b_size =
        static_cast<size_t>(config_.expert_num) * static_cast<size_t>(lora_rank_) * config_.intermediate_size;
    const size_t down_b_size =
        static_cast<size_t>(config_.expert_num) * static_cast<size_t>(lora_rank_) * config_.hidden_size;

    if (gate_lora_b_ != nullptr) gate_lora_b_transposed_.resize(gate_up_b_size);
    if (up_lora_b_ != nullptr) up_lora_b_transposed_.resize(gate_up_b_size);
    if (down_lora_b_ != nullptr) down_lora_b_transposed_.resize(down_b_size);

    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        config_.expert_num * 3, nullptr,
        [this](int task_id) {
          int expert_idx = task_id / 3;
          int lora_type = task_id % 3;

          if (lora_type == 0 && gate_lora_b_ != nullptr && !gate_lora_b_transposed_.empty()) {
            size_t src_offset = static_cast<size_t>(expert_idx) * config_.intermediate_size * lora_rank_;
            size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.intermediate_size;
            avx::transpose_lora_weight(gate_lora_b_ + src_offset, gate_lora_b_transposed_.data() + dst_offset,
                                       config_.intermediate_size, lora_rank_);
          } else if (lora_type == 1 && up_lora_b_ != nullptr && !up_lora_b_transposed_.empty()) {
            size_t src_offset = static_cast<size_t>(expert_idx) * config_.intermediate_size * lora_rank_;
            size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.intermediate_size;
            avx::transpose_lora_weight(up_lora_b_ + src_offset, up_lora_b_transposed_.data() + dst_offset,
                                       config_.intermediate_size, lora_rank_);
          } else if (lora_type == 2 && down_lora_b_ != nullptr && !down_lora_b_transposed_.empty()) {
            size_t src_offset = static_cast<size_t>(expert_idx) * config_.hidden_size * lora_rank_;
            size_t dst_offset = static_cast<size_t>(expert_idx) * lora_rank_ * config_.hidden_size;
            avx::transpose_lora_weight(down_lora_b_ + src_offset, down_lora_b_transposed_.data() + dst_offset,
                                       config_.hidden_size, lora_rank_);
          }
        },
        nullptr);
    lora_b_transposed_ = true;
  }

  K2ForwardCache& push_cache() {
    if (max_cache_depth_ <= 0) {
      throw std::runtime_error("K2 RAWINT4 SFT forward cache depth must be positive");
    }
    if (cache_stack_.empty()) cache_stack_.resize(max_cache_depth_);
    if (cache_stack_top_ >= max_cache_depth_) {
      throw std::runtime_error("K2 RAWINT4 SFT forward cache stack overflow");
    }
    return cache_stack_[cache_stack_top_++];
  }

  void rollback_cache_reservation(K2ForwardCache* reserved) noexcept {
    if (reserved == nullptr) return;
    if (cache_stack_top_ > 0 && &cache_stack_[cache_stack_top_ - 1] == reserved) {
      reserved->valid = false;
      --cache_stack_top_;
      return;
    }

    // A mismatched reservation means the stack cannot be trusted.
    for (int i = 0; i < cache_stack_top_; ++i) cache_stack_[i].valid = false;
    cache_stack_top_ = 0;
  }

  class ForwardCacheReservationGuard {
   public:
    explicit ForwardCacheReservationGuard(AMX_K2_SFT_MOE_TP& owner) : owner_(owner) {}
    ForwardCacheReservationGuard(const ForwardCacheReservationGuard&) = delete;
    ForwardCacheReservationGuard& operator=(const ForwardCacheReservationGuard&) = delete;
    ~ForwardCacheReservationGuard() noexcept { owner_.rollback_cache_reservation(reserved_); }

    void reserve(K2ForwardCache& cache) noexcept { reserved_ = &cache; }
    void commit() noexcept { reserved_ = nullptr; }

   private:
    AMX_K2_SFT_MOE_TP& owner_;
    K2ForwardCache* reserved_ = nullptr;
  };

  void ensure_cache_buffers(K2ForwardCache& cache, int qlen, int k, int total_tokens) {
    const size_t input_elems = static_cast<size_t>(qlen) * config_.hidden_size;
    const size_t inter_elems = static_cast<size_t>(total_tokens) * config_.intermediate_size;
    const size_t down_elems = static_cast<size_t>(total_tokens) * config_.hidden_size;
    const size_t lora_u_elems = static_cast<size_t>(total_tokens) * std::max(lora_rank_, 0);

    cache.input_storage.resize(input_elems);
    cache.gate_output_storage.resize(inter_elems);
    cache.up_output_storage.resize(inter_elems);
    cache.intermediate_storage.resize(inter_elems);
    cache.down_output_storage.resize(down_elems);
    cache.down_lora_u_storage.resize(lora_u_elems);
    cache.gate_lora_u_storage.resize(lora_u_elems);
    cache.up_lora_u_storage.resize(lora_u_elems);

    cache.input_cache = cache.input_storage.data();
    cache.gate_output_cache = cache.gate_output_storage.data();
    cache.up_output_cache = cache.up_output_storage.data();
    cache.intermediate_cache = cache.intermediate_storage.data();
    cache.down_output_cache = cache.down_output_storage.data();
    cache.down_lora_u_cache = cache.down_lora_u_storage.empty() ? nullptr : cache.down_lora_u_storage.data();
    cache.gate_lora_u_cache = cache.gate_lora_u_storage.empty() ? nullptr : cache.gate_lora_u_storage.data();
    cache.up_lora_u_cache = cache.up_lora_u_storage.empty() ? nullptr : cache.up_lora_u_storage.data();

    cache.expert_ids_cache.resize(static_cast<size_t>(qlen) * k);
    cache.weights_cache.resize(static_cast<size_t>(qlen) * k);
    cache.m_local_num_cache.resize(config_.expert_num);
    cache.m_local_pos_cache.assign(qlen, std::vector<int>(k, 0));
    cache.m_expert_id_map_cache.resize(config_.expert_num);
  }

  void setup_expert_buffers() {
    size_t offset = 0;
    void* gate_up_ba_pool_ptr = this->gate_up_ba_pool_;
    void* gate_bc_pool_ptr = this->gate_bc_pool_;
    void* up_bc_pool_ptr = this->up_bc_pool_;
    void* down_ba_pool_ptr = this->down_ba_pool_;
    void* down_bc_pool_ptr = this->down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;

    for (int i = 0; i < config_.expert_num; i++) {
      this->m_local_input_ptr_[i] = this->m_local_input_ + offset * config_.hidden_size;
      this->m_local_gate_output_ptr_[i] = this->m_local_gate_output_ + offset * config_.intermediate_size;
      this->m_local_up_output_ptr_[i] = this->m_local_up_output_ + offset * config_.intermediate_size;
      this->m_local_down_output_ptr_[i] = this->m_local_down_output_ + offset * config_.hidden_size;
      offset += this->m_local_num_[i];

      if (this->m_local_num_[i] == 0) continue;

      size_t max_m = (this->m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      this->gate_up_ba_[i]->max_m = max_m;
      this->gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      gate_up_ba_pool_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(gate_up_ba_pool_ptr) +
                                                    align64(this->buffer_a_required_size(max_m, config_.hidden_size)));

      this->gate_bc_[i]->max_m = max_m;
      this->gate_bc_[i]->set_data(gate_bc_pool_ptr);
      gate_bc_pool_ptr =
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(gate_bc_pool_ptr) +
                                  align64(this->buffer_c_required_size(max_m, config_.intermediate_size)));

      this->up_bc_[i]->max_m = max_m;
      this->up_bc_[i]->set_data(up_bc_pool_ptr);
      up_bc_pool_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(up_bc_pool_ptr) +
                                               align64(this->buffer_c_required_size(max_m, config_.intermediate_size)));

      this->down_ba_[i]->max_m = max_m;
      this->down_ba_[i]->set_data(down_ba_pool_ptr);
      down_ba_pool_ptr =
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(down_ba_pool_ptr) +
                                  align64(this->buffer_a_required_size(max_m, config_.intermediate_size)));

      this->down_bc_[i]->max_m = max_m;
      this->down_bc_[i]->set_data(down_bc_pool_ptr);
      down_bc_pool_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(down_bc_pool_ptr) +
                                                 align64(this->buffer_c_required_size(max_m, config_.hidden_size)));
    }
  }

  int route_tokens(int qlen, int k, const int64_t* expert_ids) {
    int activated_expert = 0;
    std::fill(this->m_local_num_.begin(), this->m_local_num_.end(), 0);

    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        const int64_t expert_id = expert_ids[i * k + j];
        if (config_.should_skip_expert(expert_id)) continue;
        this->m_local_pos_[i][j] = this->m_local_num_[expert_id]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (this->m_local_num_[i] > 0) {
        this->m_expert_id_map_[activated_expert++] = i;
      }
    }
    return activated_expert;
  }

  void copy_inputs_to_expert_buffers(int qlen, int k, const int64_t* expert_ids, const void* input) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, k, expert_ids, input](int i) {
      for (int j = 0; j < k; j++) {
        const int64_t expert_id = expert_ids[i * k + j];
        if (config_.should_skip_expert(expert_id)) continue;
        std::memcpy(this->m_local_input_ptr_[expert_id] + this->m_local_pos_[i][j] * config_.hidden_size,
                    reinterpret_cast<const ggml_bf16_t*>(input) + static_cast<size_t>(i) * config_.hidden_size,
                    sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    };

    if (qlen < 10) {
      for (int i = 0; i < qlen; i++) fn(i);
    } else {
      pool->do_work_stealing_job(qlen, nullptr, fn, nullptr);
    }
  }

  void compute_lora_gate_up(int activated_expert, K2ForwardCache* cache) {
    if (!has_gate_up_lora()) return;
    prepare_lora_b_transposed();

    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int rank = lora_rank_;
    const float scale = lora_scaling_;
    const int nth = 2;

    pool->do_work_stealing_job(
        activated_expert * 2 * nth, nullptr,
        [this, cache, hidden, inter_size, rank, scale, nth](int task_id) {
          bool do_up = (task_id / nth) % 2;
          int expert_task = task_id / (2 * nth);
          int ith = task_id % nth;
          int expert_idx = this->m_expert_id_map_[expert_task];
          int num_tokens = this->m_local_num_[expert_idx];
          if (num_tokens == 0) return;

          int tokens_per_thread = (num_tokens + nth - 1) / nth;
          int t_start = ith * tokens_per_thread;
          int t_end = std::min(t_start + tokens_per_thread, num_tokens);
          if (t_start >= num_tokens) return;

          const ggml_bf16_t* lora_a = do_up ? up_lora_a_ : gate_lora_a_;
          const ggml_bf16_t* lora_b_t = do_up ? up_lora_b_transposed_.data() : gate_lora_b_transposed_.data();
          ggml_bf16_t* output =
              do_up ? this->m_local_up_output_ptr_[expert_idx] : this->m_local_gate_output_ptr_[expert_idx];
          if (lora_a == nullptr || lora_b_t == nullptr) return;

          size_t lora_a_offset = static_cast<size_t>(expert_idx) * rank * hidden;
          size_t lora_b_t_offset = static_cast<size_t>(expert_idx) * rank * inter_size;
          const ggml_bf16_t* expert_lora_a = lora_a + lora_a_offset;
          const ggml_bf16_t* expert_lora_b_t = lora_b_t + lora_b_t_offset;

          int local_num_tokens = t_end - t_start;
          std::vector<float> local_intermediate(static_cast<size_t>(local_num_tokens) * rank);
          avx::lora_bf16_matmul_t4r4(this->m_local_input_ptr_[expert_idx] + static_cast<size_t>(t_start) * hidden,
                                     expert_lora_a, local_intermediate.data(), local_num_tokens, hidden, rank);
          if (cache != nullptr) {
            float* cache_u = do_up ? cache->up_lora_u_cache : cache->gate_lora_u_cache;
            if (cache_u != nullptr) {
              float* dst_u = cache_u + (cache_offsets_[expert_task] + t_start) * static_cast<size_t>(rank);
              std::memcpy(dst_u, local_intermediate.data(),
                          static_cast<size_t>(local_num_tokens) * rank * sizeof(float));
            }
          }
          avx::lora_fp32_bf16_fused_add_transposed(local_intermediate.data(), expert_lora_b_t,
                                                   output + static_cast<size_t>(t_start) * inter_size, local_num_tokens,
                                                   rank, inter_size, scale);
        },
        nullptr);
  }

  void compute_lora_down(int activated_expert, K2ForwardCache* cache) {
    if (!has_down_lora()) return;
    prepare_lora_b_transposed();

    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int inter_size = config_.intermediate_size;
    const int hidden = config_.hidden_size;
    const int rank = lora_rank_;
    const float scale = lora_scaling_;
    const int nth = 2;

    pool->do_work_stealing_job(
        activated_expert * nth, nullptr,
        [this, cache, inter_size, hidden, rank, scale, nth](int task_id) {
          int expert_task = task_id / nth;
          int expert_idx = this->m_expert_id_map_[expert_task];
          int ith = task_id % nth;
          int num_tokens = this->m_local_num_[expert_idx];
          if (num_tokens == 0) return;

          int tokens_per_thread = (num_tokens + nth - 1) / nth;
          int t_start = ith * tokens_per_thread;
          int t_end = std::min(t_start + tokens_per_thread, num_tokens);
          if (t_start >= num_tokens) return;

          size_t lora_a_offset = static_cast<size_t>(expert_idx) * rank * inter_size;
          size_t lora_b_t_offset = static_cast<size_t>(expert_idx) * rank * hidden;
          const ggml_bf16_t* expert_lora_a = down_lora_a_ + lora_a_offset;
          const ggml_bf16_t* expert_lora_b_t = down_lora_b_transposed_.data() + lora_b_t_offset;

          int local_num_tokens = t_end - t_start;
          std::vector<float> local_intermediate(static_cast<size_t>(local_num_tokens) * rank);
          avx::lora_bf16_matmul_t4r4(
              this->m_local_gate_output_ptr_[expert_idx] + static_cast<size_t>(t_start) * inter_size, expert_lora_a,
              local_intermediate.data(), local_num_tokens, inter_size, rank);

          if (cache != nullptr && cache->down_lora_u_cache != nullptr) {
            float* cache_u = cache->down_lora_u_cache + (cache_offsets_[expert_task] + t_start) * rank;
            std::memcpy(cache_u, local_intermediate.data(),
                        static_cast<size_t>(local_num_tokens) * rank * sizeof(float));
          }

          avx::lora_fp32_bf16_fused_add_transposed(
              local_intermediate.data(), expert_lora_b_t,
              this->m_local_down_output_ptr_[expert_idx] + static_cast<size_t>(t_start) * hidden, local_num_tokens,
              rank, hidden, scale);
        },
        nullptr);
  }

  void prepare_cache_for_backward(K2ForwardCache& cache, int qlen, int k, const int64_t* expert_ids,
                                  const float* weights, int activated_expert, const void* input) {
    cache.qlen_cache = qlen;
    cache.k_cache = k;
    cache.activated_expert_cache = activated_expert;
    cache.valid = false;

    cache_offsets_.assign(static_cast<size_t>(activated_expert) + 1, 0);
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = this->m_expert_id_map_[i];
      cache_offsets_[i + 1] = cache_offsets_[i] + this->m_local_num_[expert_idx];
    }
    const int total_tokens = static_cast<int>(cache_offsets_[activated_expert]);
    ensure_cache_buffers(cache, qlen, k, total_tokens);

    std::copy(expert_ids, expert_ids + static_cast<size_t>(qlen) * k, cache.expert_ids_cache.begin());
    std::copy(weights, weights + static_cast<size_t>(qlen) * k, cache.weights_cache.begin());
    cache.m_local_num_cache = this->m_local_num_;
    for (int i = 0; i < qlen; i++) {
      std::memcpy(cache.m_local_pos_cache[i].data(), this->m_local_pos_[i].data(), k * sizeof(int));
    }
    for (int i = 0; i < activated_expert; i++) {
      cache.m_expert_id_map_cache[i] = this->m_expert_id_map_[i];
    }

    std::memcpy(cache.input_cache, input, static_cast<size_t>(qlen) * config_.hidden_size * sizeof(ggml_bf16_t));
  }

  void save_gate_up_to_cache(K2ForwardCache& cache, int activated_expert) {
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = this->m_expert_id_map_[i];
      int num_tokens = this->m_local_num_[expert_idx];
      if (num_tokens == 0) continue;
      size_t offset = cache_offsets_[i];
      std::memcpy(cache.gate_output_cache + offset * config_.intermediate_size,
                  this->m_local_gate_output_ptr_[expert_idx],
                  static_cast<size_t>(num_tokens) * config_.intermediate_size * sizeof(ggml_bf16_t));
      std::memcpy(cache.up_output_cache + offset * config_.intermediate_size, this->m_local_up_output_ptr_[expert_idx],
                  static_cast<size_t>(num_tokens) * config_.intermediate_size * sizeof(ggml_bf16_t));
    }
  }

  void save_intermediate_to_cache(K2ForwardCache& cache, int activated_expert) {
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = this->m_expert_id_map_[i];
      int num_tokens = this->m_local_num_[expert_idx];
      if (num_tokens == 0) continue;
      std::memcpy(cache.intermediate_cache + cache_offsets_[i] * config_.intermediate_size,
                  this->m_local_gate_output_ptr_[expert_idx],
                  static_cast<size_t>(num_tokens) * config_.intermediate_size * sizeof(ggml_bf16_t));
    }
  }

  void save_down_output_to_cache(K2ForwardCache& cache, int activated_expert) {
    for (int i = 0; i < activated_expert; i++) {
      int expert_idx = this->m_expert_id_map_[i];
      int num_tokens = this->m_local_num_[expert_idx];
      if (num_tokens == 0) continue;
      std::memcpy(cache.down_output_cache + cache_offsets_[i] * config_.hidden_size,
                  this->m_local_down_output_ptr_[expert_idx],
                  static_cast<size_t>(num_tokens) * config_.hidden_size * sizeof(ggml_bf16_t));
    }
  }

  const K2ForwardCache& latest_cache() const {
    if (cache_stack_top_ <= 0 || !cache_stack_[cache_stack_top_ - 1].valid) {
      throw std::runtime_error("K2 RAWINT4 SFT forward cache is empty");
    }
    return cache_stack_[cache_stack_top_ - 1];
  }

  void pop_latest_cache() {
    if (cache_stack_top_ <= 0 || !cache_stack_[cache_stack_top_ - 1].valid) {
      throw std::runtime_error("K2 RAWINT4 SFT forward cache is empty");
    }
    cache_stack_[cache_stack_top_ - 1].valid = false;
    cache_stack_top_--;
  }

  struct TP1BackwardLayout {
    std::vector<size_t> expert_base;
    std::vector<int> expert_task_index;
    std::vector<int> row_to_token;
    std::vector<float> row_to_weight;
    size_t total_tokens = 0;
  };

  static inline int decode_signed_int4(uint8_t byte, bool high_nibble) {
    const int nibble = high_nibble ? ((byte >> 4) & 0x0f) : (byte & 0x0f);
    return nibble - 8;
  }

  static inline int packed_int4_value(const uint8_t* packed, int row, int col, int cols) {
    const uint8_t byte = packed[static_cast<size_t>(row) * (cols / 2) + static_cast<size_t>(col / 2)];
    return decode_signed_int4(byte, (col & 1) != 0);
  }

  float kgroup_scale(const float* scales, int row, int col, int cols) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    return scales[static_cast<size_t>(row) * groups_per_row + static_cast<size_t>(col / group_size)];
  }

  float load_kgroup_weight_f32(const uint8_t* packed, const float* scales, int row, int col, int cols) const {
    return static_cast<float>(packed_int4_value(packed, row, col, cols)) * kgroup_scale(scales, row, col, cols);
  }

  static bool dense_coeff_fastpath_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_DENSE_COEFF_FASTPATH");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool gate_up32_fastpath_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_GATE_UP32_FASTPATH");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool short_base_fastpath_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_SHORT_BASE_FASTPATH");
      return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
  }

  static bool sparse_lora_b_accum_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_SPARSE_LORA_B");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool bf16_dot2_lora_u_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_BF16_DOT2_LORA_U");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool down_bprop_rank2_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_DOWN_BPROP_RANK2_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool down_lora_b_rank2_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_DOWN_LORA_B_RANK2_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool gate_up_lora_b_rank2_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_GATE_UP_LORA_B_RANK2_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool gate_up_a_input_rank2_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_GATE_UP_A_INPUT_RANK2_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool tp_gate_up_a_input_rank2_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_TP_GATE_UP_A_INPUT_RANK2_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool rank8_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_RANK8_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool lora_backward_matmat_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_LORA_BWD_MATMAT");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool backward_workspace_v2_env_enabled() {
#if defined(__AVX512BF16__)
    return true;
#else
    return false;
#endif
  }

  static bool down_base_backward_bf16_matmat_enabled() {
#if defined(__AVX512BF16__)
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_DOWN_BASE_BWD_BF16_MATMAT");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
#else
    return false;
#endif
  }

  static bool gate_up_base_backward_bf16_matmat_enabled() {
#if defined(__AVX512BF16__)
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_GATE_UP_BASE_BWD_BF16_MATMAT");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
#else
    return false;
#endif
  }

  static bool reuse_down_lora_bprop_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_REUSE_DOWN_BPROP");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool bf16_write_vec_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_BF16_WRITE_VEC");
      return value == nullptr || value[0] == '\0' || value[0] != '0';
    }();
    return enabled;
  }

  static bool trace_forward_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_TRACE_FWD_VERBOSE");
      return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
  }

  static bool profile_forward_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_PROFILE_FWD");
      if (value == nullptr || value[0] == '\0') {
        value = std::getenv("KT_K2_SFT_TRACE_FWD");
      }
      return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
  }

  void trace_forward_step(const char* step, int qlen, int k, int activated_expert, bool save_for_backward) const {
    if (!trace_forward_enabled()) return;
    fprintf(stderr,
            "[KT_K2_SFT_FWD_TRACE] layer=%d tp_part=%d qlen=%d k=%d active=%d save=%d lora=%d cache_top=%d step=%s\n",
            config_.layer_idx, tp_part_idx, qlen, k, activated_expert, save_for_backward ? 1 : 0,
            has_any_lora() ? 1 : 0, cache_stack_top_, step);
    fflush(stderr);
  }

  struct ForwardProfile {
    using Clock = std::chrono::high_resolution_clock;
    bool enabled = false;
    Clock::time_point start;
    Clock::time_point last;
    long long base_forward_us = 0;
    long long route_us = 0;
    long long setup_us = 0;
    long long copy_input_us = 0;
    long long q_input_us = 0;
    long long gate_up_base_us = 0;
    long long gate_up_lora_us = 0;
    long long save_gate_up_us = 0;
    long long act_us = 0;
    long long save_intermediate_us = 0;
    long long q_intermediate_us = 0;
    long long down_base_us = 0;
    long long down_lora_us = 0;
    long long save_down_us = 0;
    long long merge_us = 0;

    explicit ForwardProfile(bool enabled_) : enabled(enabled_) {
      if (enabled) {
        start = Clock::now();
        last = start;
      }
    }

    void reset() {
      if (!enabled) return;
      start = Clock::now();
      last = start;
    }

    void mark(long long& slot) {
      if (!enabled) return;
      auto now = Clock::now();
      slot = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
      last = now;
    }

    long long total_us() const {
      if (!enabled) return 0;
      return std::chrono::duration_cast<std::chrono::microseconds>(last - start).count();
    }
  };

  static inline void bf16_dot2_scalar(const ggml_bf16_t* input, const ggml_bf16_t* a0, const ggml_bf16_t* a1,
                                      int hidden, float& out0, float& out1) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int h = 0; h < hidden; h++) {
      const float x = GGML_BF16_TO_FP32(input[h]);
      acc0 += x * GGML_BF16_TO_FP32(a0[h]);
      acc1 += x * GGML_BF16_TO_FP32(a1[h]);
    }
    out0 = acc0;
    out1 = acc1;
  }

  static inline void bf16_dot2_lora_u(const ggml_bf16_t* input, const ggml_bf16_t* a0, const ggml_bf16_t* a1,
                                      int hidden, float& out0, float& out1) {
    if (!bf16_dot2_lora_u_enabled()) {
      bf16_dot2_scalar(input, a0, a1, hidden, out0, out1);
      return;
    }
#if defined(__AVX512BF16__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int h = 0;
    for (; h + 31 < hidden; h += 32) {
      const __m512bh x = (__m512bh)_mm512_loadu_si512((const __m512i*)(input + h));
      acc0 = _mm512_dpbf16_ps(acc0, x, (__m512bh)_mm512_loadu_si512((const __m512i*)(a0 + h)));
      acc1 = _mm512_dpbf16_ps(acc1, x, (__m512bh)_mm512_loadu_si512((const __m512i*)(a1 + h)));
    }
    float sum0 = _mm512_reduce_add_ps(acc0);
    float sum1 = _mm512_reduce_add_ps(acc1);
    for (; h < hidden; h++) {
      const float x = GGML_BF16_TO_FP32(input[h]);
      sum0 += x * GGML_BF16_TO_FP32(a0[h]);
      sum1 += x * GGML_BF16_TO_FP32(a1[h]);
    }
    out0 = sum0;
    out1 = sum1;
#else
    bf16_dot2_scalar(input, a0, a1, hidden, out0, out1);
#endif
  }

  static inline __m512 bf16x16_to_fp32(__m256i v) {
    const __m512i expanded = _mm512_cvtepu16_epi32(v);
    return _mm512_castsi512_ps(_mm512_slli_epi32(expanded, 16));
  }

  static inline void bf16_dot8_scalar(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, int hidden, float* out) {
    for (int r = 0; r < 8; r++) {
      const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
      float acc = 0.0f;
      for (int h = 0; h < hidden; h++) {
        acc += GGML_BF16_TO_FP32(input[h]) * GGML_BF16_TO_FP32(a_row[h]);
      }
      out[r] = acc;
    }
  }

  static inline void bf16_dot8_lora_u(const ggml_bf16_t* input, const ggml_bf16_t* lora_a, int hidden, float* out) {
    if (!rank8_vec_enabled()) {
      bf16_dot8_scalar(input, lora_a, hidden, out);
      return;
    }
#if defined(__AVX512BF16__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    int h = 0;
    for (; h + 31 < hidden; h += 32) {
      const __m512bh x = (__m512bh)_mm512_loadu_si512(reinterpret_cast<const __m512i*>(input + h));
      acc0 = _mm512_dpbf16_ps(acc0, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(0) * hidden + h)));
      acc1 = _mm512_dpbf16_ps(acc1, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(1) * hidden + h)));
      acc2 = _mm512_dpbf16_ps(acc2, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(2) * hidden + h)));
      acc3 = _mm512_dpbf16_ps(acc3, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(3) * hidden + h)));
      acc4 = _mm512_dpbf16_ps(acc4, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(4) * hidden + h)));
      acc5 = _mm512_dpbf16_ps(acc5, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(5) * hidden + h)));
      acc6 = _mm512_dpbf16_ps(acc6, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(6) * hidden + h)));
      acc7 = _mm512_dpbf16_ps(acc7, x, (__m512bh)_mm512_loadu_si512(
                                           reinterpret_cast<const __m512i*>(lora_a + static_cast<size_t>(7) * hidden + h)));
    }
    out[0] = _mm512_reduce_add_ps(acc0);
    out[1] = _mm512_reduce_add_ps(acc1);
    out[2] = _mm512_reduce_add_ps(acc2);
    out[3] = _mm512_reduce_add_ps(acc3);
    out[4] = _mm512_reduce_add_ps(acc4);
    out[5] = _mm512_reduce_add_ps(acc5);
    out[6] = _mm512_reduce_add_ps(acc6);
    out[7] = _mm512_reduce_add_ps(acc7);
    for (; h < hidden; h++) {
      const float x = GGML_BF16_TO_FP32(input[h]);
      out[0] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(0) * hidden + h]);
      out[1] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(1) * hidden + h]);
      out[2] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(2) * hidden + h]);
      out[3] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(3) * hidden + h]);
      out[4] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(4) * hidden + h]);
      out[5] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(5) * hidden + h]);
      out[6] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(6) * hidden + h]);
      out[7] += x * GGML_BF16_TO_FP32(lora_a[static_cast<size_t>(7) * hidden + h]);
    }
#else
    bf16_dot8_scalar(input, lora_a, hidden, out);
#endif
  }

  static inline void down_bprop_rank2_scalar(const float* grad_down_row, const ggml_bf16_t* expert_down_b, int hidden,
                                             float& out0, float& out1) {
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    for (int h = 0; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * 2;
      gb0 += g * GGML_BF16_TO_FP32(down_b_row[0]);
      gb1 += g * GGML_BF16_TO_FP32(down_b_row[1]);
    }
    out0 = gb0;
    out1 = gb1;
  }

  static inline void down_bprop_rank2_vec(const float* grad_down_row, const ggml_bf16_t* expert_down_b, int hidden,
                                          float& out0, float& out1) {
    if (!down_bprop_rank2_vec_enabled()) {
      down_bprop_rank2_scalar(grad_down_row, expert_down_b, hidden, out0, out1);
      return;
    }
#if defined(__AVX512BW__)
    alignas(64) static const uint16_t even_idx_values[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                                             0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    alignas(64) static const uint16_t odd_idx_values[32] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                                                            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    const __m512i even_idx = _mm512_load_si512(reinterpret_cast<const __m512i*>(even_idx_values));
    const __m512i odd_idx = _mm512_load_si512(reinterpret_cast<const __m512i*>(odd_idx_values));

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int h = 0;
    for (; h + 15 < hidden; h += 16) {
      const __m512 g = _mm512_loadu_ps(grad_down_row + h);
      const __m512i b01 =
          _mm512_loadu_si512(reinterpret_cast<const __m512i*>(expert_down_b + static_cast<size_t>(h) * 2));
      const __m512i b0_perm = _mm512_permutexvar_epi16(even_idx, b01);
      const __m512i b1_perm = _mm512_permutexvar_epi16(odd_idx, b01);
      const __m512 b0 = bf16x16_to_fp32(_mm512_castsi512_si256(b0_perm));
      const __m512 b1 = bf16x16_to_fp32(_mm512_castsi512_si256(b1_perm));
      acc0 = _mm512_fmadd_ps(g, b0, acc0);
      acc1 = _mm512_fmadd_ps(g, b1, acc1);
    }
    float gb0 = _mm512_reduce_add_ps(acc0);
    float gb1 = _mm512_reduce_add_ps(acc1);
    for (; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * 2;
      gb0 += g * GGML_BF16_TO_FP32(down_b_row[0]);
      gb1 += g * GGML_BF16_TO_FP32(down_b_row[1]);
    }
    out0 = gb0;
    out1 = gb1;
#else
    down_bprop_rank2_scalar(grad_down_row, expert_down_b, hidden, out0, out1);
#endif
  }

  static inline void down_bprop_rank8_scalar(const float* grad_down_row, const ggml_bf16_t* expert_down_b, int hidden,
                                             float* out) {
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    float gb2 = 0.0f;
    float gb3 = 0.0f;
    float gb4 = 0.0f;
    float gb5 = 0.0f;
    float gb6 = 0.0f;
    float gb7 = 0.0f;
    for (int h = 0; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      const ggml_bf16_t* b = expert_down_b + static_cast<size_t>(h) * 8;
      gb0 += g * GGML_BF16_TO_FP32(b[0]);
      gb1 += g * GGML_BF16_TO_FP32(b[1]);
      gb2 += g * GGML_BF16_TO_FP32(b[2]);
      gb3 += g * GGML_BF16_TO_FP32(b[3]);
      gb4 += g * GGML_BF16_TO_FP32(b[4]);
      gb5 += g * GGML_BF16_TO_FP32(b[5]);
      gb6 += g * GGML_BF16_TO_FP32(b[6]);
      gb7 += g * GGML_BF16_TO_FP32(b[7]);
    }
    out[0] = gb0;
    out[1] = gb1;
    out[2] = gb2;
    out[3] = gb3;
    out[4] = gb4;
    out[5] = gb5;
    out[6] = gb6;
    out[7] = gb7;
  }

  static inline __m512 bf16_pair_lo_to_fp32(__m512i v) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_and_si512(v, _mm512_set1_epi32(0xffff)), 16));
  }

  static inline __m512 bf16_pair_hi_to_fp32(__m512i v) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_srli_epi32(v, 16), 16));
  }

  static inline void down_bprop_rank8_vec(const float* grad_down_row, const ggml_bf16_t* expert_down_b, int hidden,
                                          float* out) {
    if (!rank8_vec_enabled()) {
      down_bprop_rank8_scalar(grad_down_row, expert_down_b, hidden, out);
      return;
    }
#if defined(__AVX512F__)
    alignas(64) static const int32_t byte_offsets_values[16] = {0,   16,  32,  48,  64,  80,  96,  112,
                                                                128, 144, 160, 176, 192, 208, 224, 240};
    const __m512i byte_offsets = _mm512_load_si512(reinterpret_cast<const __m512i*>(byte_offsets_values));
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();
    int h = 0;
    for (; h + 15 < hidden; h += 16) {
      const __m512 g = _mm512_loadu_ps(grad_down_row + h);
      const ggml_bf16_t* base = expert_down_b + static_cast<size_t>(h) * 8;
      const __m512i b01 = _mm512_i32gather_epi32(byte_offsets, base + 0, 1);
      const __m512i b23 = _mm512_i32gather_epi32(byte_offsets, base + 2, 1);
      const __m512i b45 = _mm512_i32gather_epi32(byte_offsets, base + 4, 1);
      const __m512i b67 = _mm512_i32gather_epi32(byte_offsets, base + 6, 1);
      acc0 = _mm512_fmadd_ps(g, bf16_pair_lo_to_fp32(b01), acc0);
      acc1 = _mm512_fmadd_ps(g, bf16_pair_hi_to_fp32(b01), acc1);
      acc2 = _mm512_fmadd_ps(g, bf16_pair_lo_to_fp32(b23), acc2);
      acc3 = _mm512_fmadd_ps(g, bf16_pair_hi_to_fp32(b23), acc3);
      acc4 = _mm512_fmadd_ps(g, bf16_pair_lo_to_fp32(b45), acc4);
      acc5 = _mm512_fmadd_ps(g, bf16_pair_hi_to_fp32(b45), acc5);
      acc6 = _mm512_fmadd_ps(g, bf16_pair_lo_to_fp32(b67), acc6);
      acc7 = _mm512_fmadd_ps(g, bf16_pair_hi_to_fp32(b67), acc7);
    }
    out[0] = _mm512_reduce_add_ps(acc0);
    out[1] = _mm512_reduce_add_ps(acc1);
    out[2] = _mm512_reduce_add_ps(acc2);
    out[3] = _mm512_reduce_add_ps(acc3);
    out[4] = _mm512_reduce_add_ps(acc4);
    out[5] = _mm512_reduce_add_ps(acc5);
    out[6] = _mm512_reduce_add_ps(acc6);
    out[7] = _mm512_reduce_add_ps(acc7);
    for (; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      const ggml_bf16_t* b = expert_down_b + static_cast<size_t>(h) * 8;
      out[0] += g * GGML_BF16_TO_FP32(b[0]);
      out[1] += g * GGML_BF16_TO_FP32(b[1]);
      out[2] += g * GGML_BF16_TO_FP32(b[2]);
      out[3] += g * GGML_BF16_TO_FP32(b[3]);
      out[4] += g * GGML_BF16_TO_FP32(b[4]);
      out[5] += g * GGML_BF16_TO_FP32(b[5]);
      out[6] += g * GGML_BF16_TO_FP32(b[6]);
      out[7] += g * GGML_BF16_TO_FP32(b[7]);
    }
#else
    down_bprop_rank8_scalar(grad_down_row, expert_down_b, hidden, out);
#endif
  }

  static inline void accumulate_down_lora_b_rank2_scalar(const float* grad_down_row, float u0_scaled, float u1_scaled,
                                                         int hidden, float* grad_b) {
    for (int h = 0; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      float* grad_b_row = grad_b + static_cast<size_t>(h) * 2;
      grad_b_row[0] += g * u0_scaled;
      grad_b_row[1] += g * u1_scaled;
    }
  }

  static inline void accumulate_down_lora_b_rank2_vec(const float* grad_down_row, float u0_scaled, float u1_scaled,
                                                      int hidden, float* grad_b) {
    if (!down_lora_b_rank2_vec_enabled()) {
      accumulate_down_lora_b_rank2_scalar(grad_down_row, u0_scaled, u1_scaled, hidden, grad_b);
      return;
    }
#if defined(__AVX512F__)
    alignas(64) static const int32_t idx_lo_values[16] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
    alignas(64) static const int32_t idx_hi_values[16] = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    const __m512i idx_lo = _mm512_load_si512(reinterpret_cast<const __m512i*>(idx_lo_values));
    const __m512i idx_hi = _mm512_load_si512(reinterpret_cast<const __m512i*>(idx_hi_values));
    const __m512 u0 = _mm512_set1_ps(u0_scaled);
    const __m512 u1 = _mm512_set1_ps(u1_scaled);

    int h = 0;
    for (; h + 15 < hidden; h += 16) {
      const __m512 g = _mm512_loadu_ps(grad_down_row + h);
      const __m512 add0 = _mm512_mul_ps(g, u0);
      const __m512 add1 = _mm512_mul_ps(g, u1);
      const __m512 interleaved_lo = _mm512_permutex2var_ps(add0, idx_lo, add1);
      const __m512 interleaved_hi = _mm512_permutex2var_ps(add0, idx_hi, add1);
      float* out = grad_b + static_cast<size_t>(h) * 2;
      _mm512_storeu_ps(out, _mm512_add_ps(_mm512_loadu_ps(out), interleaved_lo));
      _mm512_storeu_ps(out + 16, _mm512_add_ps(_mm512_loadu_ps(out + 16), interleaved_hi));
    }
    for (; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      float* grad_b_row = grad_b + static_cast<size_t>(h) * 2;
      grad_b_row[0] += g * u0_scaled;
      grad_b_row[1] += g * u1_scaled;
    }
#else
    accumulate_down_lora_b_rank2_scalar(grad_down_row, u0_scaled, u1_scaled, hidden, grad_b);
#endif
  }

  static inline void accumulate_down_lora_b_rank8_scalar(const float* grad_down_row, const float* u_scaled, int hidden,
                                                         float* grad_b) {
    for (int h = 0; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      float* row = grad_b + static_cast<size_t>(h) * 8;
      row[0] += g * u_scaled[0];
      row[1] += g * u_scaled[1];
      row[2] += g * u_scaled[2];
      row[3] += g * u_scaled[3];
      row[4] += g * u_scaled[4];
      row[5] += g * u_scaled[5];
      row[6] += g * u_scaled[6];
      row[7] += g * u_scaled[7];
    }
  }

  static inline void accumulate_down_lora_b_rank8_vec(const float* grad_down_row, const float* u_scaled, int hidden,
                                                      float* grad_b) {
    if (!rank8_vec_enabled()) {
      accumulate_down_lora_b_rank8_scalar(grad_down_row, u_scaled, hidden, grad_b);
      return;
    }
#if defined(__AVX2__)
    const __m256 u = _mm256_loadu_ps(u_scaled);
    for (int h = 0; h < hidden; h++) {
      const float g = grad_down_row[h];
      if (g == 0.0f) continue;
      float* row = grad_b + static_cast<size_t>(h) * 8;
      _mm256_storeu_ps(row, _mm256_fmadd_ps(_mm256_set1_ps(g), u, _mm256_loadu_ps(row)));
    }
#else
    accumulate_down_lora_b_rank8_scalar(grad_down_row, u_scaled, hidden, grad_b);
#endif
  }

  static inline void accumulate_gate_up_lora_b_rank2_scalar(const float* grad_row, const ggml_bf16_t* lora_b,
                                                            float u0_scaled, float u1_scaled, int inter_size,
                                                            float* grad_b, float& out0, float& out1) {
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    for (int i = 0; i < inter_size; i++) {
      const float g = grad_row[i];
      if (g == 0.0f) continue;
      const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * 2;
      if (grad_b != nullptr) {
        float* grad_b_row = grad_b + static_cast<size_t>(i) * 2;
        grad_b_row[0] += g * u0_scaled;
        grad_b_row[1] += g * u1_scaled;
      }
      gb0 += g * GGML_BF16_TO_FP32(b_row[0]);
      gb1 += g * GGML_BF16_TO_FP32(b_row[1]);
    }
    out0 = gb0;
    out1 = gb1;
  }

  static inline void accumulate_gate_up_lora_b_rank2_vec(const float* grad_row, const ggml_bf16_t* lora_b,
                                                         float u0_scaled, float u1_scaled, int inter_size,
                                                         float* grad_b, float& out0, float& out1) {
    if (!gate_up_lora_b_rank2_vec_enabled()) {
      accumulate_gate_up_lora_b_rank2_scalar(grad_row, lora_b, u0_scaled, u1_scaled, inter_size, grad_b, out0, out1);
      return;
    }
#if defined(__AVX512BW__) && defined(__AVX512F__)
    alignas(64) static const uint16_t even_idx_values[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                                             0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    alignas(64) static const uint16_t odd_idx_values[32] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                                                            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    alignas(64) static const int32_t idx_lo_values[16] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
    alignas(64) static const int32_t idx_hi_values[16] = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    const __m512i even_idx = _mm512_load_si512(reinterpret_cast<const __m512i*>(even_idx_values));
    const __m512i odd_idx = _mm512_load_si512(reinterpret_cast<const __m512i*>(odd_idx_values));
    const __m512i idx_lo = _mm512_load_si512(reinterpret_cast<const __m512i*>(idx_lo_values));
    const __m512i idx_hi = _mm512_load_si512(reinterpret_cast<const __m512i*>(idx_hi_values));
    const __m512 u0 = _mm512_set1_ps(u0_scaled);
    const __m512 u1 = _mm512_set1_ps(u1_scaled);

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < inter_size; i += 16) {
      const __m512 g = _mm512_loadu_ps(grad_row + i);
      const __m512i b01 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lora_b + static_cast<size_t>(i) * 2));
      const __m512i b0_perm = _mm512_permutexvar_epi16(even_idx, b01);
      const __m512i b1_perm = _mm512_permutexvar_epi16(odd_idx, b01);
      const __m512 b0 = bf16x16_to_fp32(_mm512_castsi512_si256(b0_perm));
      const __m512 b1 = bf16x16_to_fp32(_mm512_castsi512_si256(b1_perm));
      acc0 = _mm512_fmadd_ps(g, b0, acc0);
      acc1 = _mm512_fmadd_ps(g, b1, acc1);

      if (grad_b != nullptr) {
        const __m512 add0 = _mm512_mul_ps(g, u0);
        const __m512 add1 = _mm512_mul_ps(g, u1);
        const __m512 interleaved_lo = _mm512_permutex2var_ps(add0, idx_lo, add1);
        const __m512 interleaved_hi = _mm512_permutex2var_ps(add0, idx_hi, add1);
        float* out = grad_b + static_cast<size_t>(i) * 2;
        _mm512_storeu_ps(out, _mm512_add_ps(_mm512_loadu_ps(out), interleaved_lo));
        _mm512_storeu_ps(out + 16, _mm512_add_ps(_mm512_loadu_ps(out + 16), interleaved_hi));
      }
    }
    float gb0 = _mm512_reduce_add_ps(acc0);
    float gb1 = _mm512_reduce_add_ps(acc1);
    for (; i < inter_size; i++) {
      const float g = grad_row[i];
      if (g == 0.0f) continue;
      const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * 2;
      if (grad_b != nullptr) {
        float* grad_b_row = grad_b + static_cast<size_t>(i) * 2;
        grad_b_row[0] += g * u0_scaled;
        grad_b_row[1] += g * u1_scaled;
      }
      gb0 += g * GGML_BF16_TO_FP32(b_row[0]);
      gb1 += g * GGML_BF16_TO_FP32(b_row[1]);
    }
    out0 = gb0;
    out1 = gb1;
#else
    accumulate_gate_up_lora_b_rank2_scalar(grad_row, lora_b, u0_scaled, u1_scaled, inter_size, grad_b, out0, out1);
#endif
  }

  static inline __m256 bf16x8_to_fp32(__m128i v) {
    const __m256i expanded = _mm256_cvtepu16_epi32(v);
    return _mm256_castsi256_ps(_mm256_slli_epi32(expanded, 16));
  }

  static inline void accumulate_gate_up_lora_b_rank8_scalar(const float* grad_row, const ggml_bf16_t* lora_b,
                                                            const float* u_scaled, int inter_size, float* grad_b,
                                                            float* out) {
    float gb0 = 0.0f;
    float gb1 = 0.0f;
    float gb2 = 0.0f;
    float gb3 = 0.0f;
    float gb4 = 0.0f;
    float gb5 = 0.0f;
    float gb6 = 0.0f;
    float gb7 = 0.0f;
    for (int i = 0; i < inter_size; i++) {
      const float g = grad_row[i];
      if (g == 0.0f) continue;
      const ggml_bf16_t* b = lora_b + static_cast<size_t>(i) * 8;
      if (grad_b != nullptr) {
        float* row = grad_b + static_cast<size_t>(i) * 8;
        row[0] += g * u_scaled[0];
        row[1] += g * u_scaled[1];
        row[2] += g * u_scaled[2];
        row[3] += g * u_scaled[3];
        row[4] += g * u_scaled[4];
        row[5] += g * u_scaled[5];
        row[6] += g * u_scaled[6];
        row[7] += g * u_scaled[7];
      }
      gb0 += g * GGML_BF16_TO_FP32(b[0]);
      gb1 += g * GGML_BF16_TO_FP32(b[1]);
      gb2 += g * GGML_BF16_TO_FP32(b[2]);
      gb3 += g * GGML_BF16_TO_FP32(b[3]);
      gb4 += g * GGML_BF16_TO_FP32(b[4]);
      gb5 += g * GGML_BF16_TO_FP32(b[5]);
      gb6 += g * GGML_BF16_TO_FP32(b[6]);
      gb7 += g * GGML_BF16_TO_FP32(b[7]);
    }
    out[0] = gb0;
    out[1] = gb1;
    out[2] = gb2;
    out[3] = gb3;
    out[4] = gb4;
    out[5] = gb5;
    out[6] = gb6;
    out[7] = gb7;
  }

  static inline void accumulate_gate_up_lora_b_rank8_vec(const float* grad_row, const ggml_bf16_t* lora_b,
                                                         const float* u_scaled, int inter_size, float* grad_b,
                                                         float* out) {
    if (!rank8_vec_enabled()) {
      accumulate_gate_up_lora_b_rank8_scalar(grad_row, lora_b, u_scaled, inter_size, grad_b, out);
      return;
    }
#if defined(__AVX2__)
    const __m256 u = _mm256_loadu_ps(u_scaled);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < inter_size; i++) {
      const float g = grad_row[i];
      if (g == 0.0f) continue;
      const __m256 gv = _mm256_set1_ps(g);
      acc = _mm256_fmadd_ps(gv, bf16x8_to_fp32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(
                                       lora_b + static_cast<size_t>(i) * 8))),
                            acc);
      if (grad_b != nullptr) {
        float* row = grad_b + static_cast<size_t>(i) * 8;
        _mm256_storeu_ps(row, _mm256_fmadd_ps(gv, u, _mm256_loadu_ps(row)));
      }
    }
    _mm256_storeu_ps(out, acc);
#else
    accumulate_gate_up_lora_b_rank8_scalar(grad_row, lora_b, u_scaled, inter_size, grad_b, out);
#endif
  }

  static inline void accumulate_gate_up_a_input_rank2_scalar(const ggml_bf16_t* input_row, const ggml_bf16_t* a0,
                                                             const ggml_bf16_t* a1, float gu0, float gu1, int hidden,
                                                             float* grad_input_row, float* grad_a0, float* grad_a1) {
    if (grad_a0 != nullptr && grad_a1 != nullptr) {
      for (int h = 0; h < hidden; h++) {
        const float x = GGML_BF16_TO_FP32(input_row[h]);
        grad_a0[h] += gu0 * x;
        grad_input_row[h] += gu0 * GGML_BF16_TO_FP32(a0[h]);
      }
      for (int h = 0; h < hidden; h++) {
        const float x = GGML_BF16_TO_FP32(input_row[h]);
        grad_a1[h] += gu1 * x;
        grad_input_row[h] += gu1 * GGML_BF16_TO_FP32(a1[h]);
      }
    } else {
      for (int h = 0; h < hidden; h++) {
        grad_input_row[h] += gu0 * GGML_BF16_TO_FP32(a0[h]);
      }
      for (int h = 0; h < hidden; h++) {
        grad_input_row[h] += gu1 * GGML_BF16_TO_FP32(a1[h]);
      }
    }
  }

  static inline void accumulate_gate_up_a_input_rank2_vec(const ggml_bf16_t* input_row, const ggml_bf16_t* a0,
                                                          const ggml_bf16_t* a1, float gu0, float gu1, int hidden,
                                                          float* grad_input_row, float* grad_a0, float* grad_a1,
                                                          bool check_single_tp_env = true) {
    if (check_single_tp_env && !gate_up_a_input_rank2_vec_enabled()) {
      accumulate_gate_up_a_input_rank2_scalar(input_row, a0, a1, gu0, gu1, hidden, grad_input_row, grad_a0, grad_a1);
      return;
    }
#if defined(__AVX512F__)
    const __m512 gu0_vec = _mm512_set1_ps(gu0);
    const __m512 gu1_vec = _mm512_set1_ps(gu1);
    int h = 0;
    if (grad_a0 != nullptr && grad_a1 != nullptr) {
      for (; h + 15 < hidden; h += 16) {
        const __m512 x = bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + h)));
        const __m512 a0v = bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0 + h)));
        const __m512 a1v = bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1 + h)));

        _mm512_storeu_ps(grad_a0 + h, _mm512_add_ps(_mm512_loadu_ps(grad_a0 + h), _mm512_mul_ps(gu0_vec, x)));
        _mm512_storeu_ps(grad_a1 + h, _mm512_add_ps(_mm512_loadu_ps(grad_a1 + h), _mm512_mul_ps(gu1_vec, x)));

        __m512 grad_input = _mm512_loadu_ps(grad_input_row + h);
        grad_input = _mm512_add_ps(grad_input, _mm512_mul_ps(gu0_vec, a0v));
        grad_input = _mm512_add_ps(grad_input, _mm512_mul_ps(gu1_vec, a1v));
        _mm512_storeu_ps(grad_input_row + h, grad_input);
      }
    } else {
      for (; h + 15 < hidden; h += 16) {
        const __m512 a0v = bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0 + h)));
        const __m512 a1v = bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1 + h)));
        __m512 grad_input = _mm512_loadu_ps(grad_input_row + h);
        grad_input = _mm512_add_ps(grad_input, _mm512_mul_ps(gu0_vec, a0v));
        grad_input = _mm512_add_ps(grad_input, _mm512_mul_ps(gu1_vec, a1v));
        _mm512_storeu_ps(grad_input_row + h, grad_input);
      }
    }
    if (h < hidden) {
      accumulate_gate_up_a_input_rank2_scalar(input_row + h, a0 + h, a1 + h, gu0, gu1, hidden - h, grad_input_row + h,
                                              grad_a0 == nullptr ? nullptr : grad_a0 + h,
                                              grad_a1 == nullptr ? nullptr : grad_a1 + h);
    }
#else
    accumulate_gate_up_a_input_rank2_scalar(input_row, a0, a1, gu0, gu1, hidden, grad_input_row, grad_a0, grad_a1);
#endif
  }

  static inline void accumulate_lora_a_input_rank8_scalar(const ggml_bf16_t* input_row, const ggml_bf16_t* lora_a,
                                                          const float* gu_scaled, int width, float* grad_dst,
                                                          float* grad_a) {
    for (int r = 0; r < 8; r++) {
      const float gu = gu_scaled[r];
      const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * width;
      float* grad_a_row = grad_a == nullptr ? nullptr : grad_a + static_cast<size_t>(r) * width;
      for (int h = 0; h < width; h++) {
        if (grad_a_row != nullptr) {
          grad_a_row[h] += gu * GGML_BF16_TO_FP32(input_row[h]);
        }
        if (grad_dst != nullptr) {
          grad_dst[h] += gu * GGML_BF16_TO_FP32(a_row[h]);
        }
      }
    }
  }

  static inline void accumulate_lora_a_input_rank8_vec(const ggml_bf16_t* input_row, const ggml_bf16_t* lora_a,
                                                       const float* gu_scaled, int width, float* grad_dst,
                                                       float* grad_a) {
    if (!rank8_vec_enabled()) {
      accumulate_lora_a_input_rank8_scalar(input_row, lora_a, gu_scaled, width, grad_dst, grad_a);
      return;
    }
#if defined(__AVX512F__)
    const __m512 gu0 = _mm512_set1_ps(gu_scaled[0]);
    const __m512 gu1 = _mm512_set1_ps(gu_scaled[1]);
    const __m512 gu2 = _mm512_set1_ps(gu_scaled[2]);
    const __m512 gu3 = _mm512_set1_ps(gu_scaled[3]);
    const __m512 gu4 = _mm512_set1_ps(gu_scaled[4]);
    const __m512 gu5 = _mm512_set1_ps(gu_scaled[5]);
    const __m512 gu6 = _mm512_set1_ps(gu_scaled[6]);
    const __m512 gu7 = _mm512_set1_ps(gu_scaled[7]);
    int h = 0;
    for (; h + 15 < width; h += 16) {
      const __m512 x = grad_a == nullptr
                           ? _mm512_setzero_ps()
                           : bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input_row + h)));
      const __m512 a0 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(0) * width + h)));
      const __m512 a1 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(1) * width + h)));
      const __m512 a2 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(2) * width + h)));
      const __m512 a3 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(3) * width + h)));
      const __m512 a4 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(4) * width + h)));
      const __m512 a5 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(5) * width + h)));
      const __m512 a6 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(6) * width + h)));
      const __m512 a7 = bf16x16_to_fp32(
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lora_a + static_cast<size_t>(7) * width + h)));

      if (grad_a != nullptr) {
        _mm512_storeu_ps(grad_a + static_cast<size_t>(0) * width + h,
                         _mm512_fmadd_ps(gu0, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(0) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(1) * width + h,
                         _mm512_fmadd_ps(gu1, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(1) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(2) * width + h,
                         _mm512_fmadd_ps(gu2, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(2) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(3) * width + h,
                         _mm512_fmadd_ps(gu3, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(3) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(4) * width + h,
                         _mm512_fmadd_ps(gu4, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(4) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(5) * width + h,
                         _mm512_fmadd_ps(gu5, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(5) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(6) * width + h,
                         _mm512_fmadd_ps(gu6, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(6) * width + h)));
        _mm512_storeu_ps(grad_a + static_cast<size_t>(7) * width + h,
                         _mm512_fmadd_ps(gu7, x, _mm512_loadu_ps(grad_a + static_cast<size_t>(7) * width + h)));
      }
      if (grad_dst != nullptr) {
        __m512 dst = _mm512_loadu_ps(grad_dst + h);
        dst = _mm512_fmadd_ps(gu0, a0, dst);
        dst = _mm512_fmadd_ps(gu1, a1, dst);
        dst = _mm512_fmadd_ps(gu2, a2, dst);
        dst = _mm512_fmadd_ps(gu3, a3, dst);
        dst = _mm512_fmadd_ps(gu4, a4, dst);
        dst = _mm512_fmadd_ps(gu5, a5, dst);
        dst = _mm512_fmadd_ps(gu6, a6, dst);
        dst = _mm512_fmadd_ps(gu7, a7, dst);
        _mm512_storeu_ps(grad_dst + h, dst);
      }
    }
    if (h < width) {
      for (int r = 0; r < 8; r++) {
        const float gu = gu_scaled[r];
        const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * width;
        float* grad_a_row = grad_a == nullptr ? nullptr : grad_a + static_cast<size_t>(r) * width;
        for (int tail = h; tail < width; tail++) {
          if (grad_a_row != nullptr) {
            grad_a_row[tail] += gu * GGML_BF16_TO_FP32(input_row[tail]);
          }
          if (grad_dst != nullptr) {
            grad_dst[tail] += gu * GGML_BF16_TO_FP32(a_row[tail]);
          }
        }
      }
    }
#else
    accumulate_lora_a_input_rank8_scalar(input_row, lora_a, gu_scaled, width, grad_dst, grad_a);
#endif
  }

  static inline void fmadd_int4_group32(const uint8_t* group_packed, __m512 scale_vec, __m512i zero_point,
                                        __m128i nibble_mask, __m512& f0, __m512& f1) {
    const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
    const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
    const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
    const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
    const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
    const __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point);
    const __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point);
    f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale_vec, f0);
    f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale_vec, f1);
  }

#if defined(__AVX512BF16__)
  static inline __m512bh interleave_bf16_pairs(__m512bh values) {
    alignas(64) static constexpr uint16_t pair_indices[32] = {
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
    };
    const __m512i indices = _mm512_load_si512(reinterpret_cast<const __m512i*>(pair_indices));
    return (__m512bh)_mm512_permutexvar_epi16(indices, (__m512i)values);
  }

  static inline __m512bh fp32_pair_to_bf16(float value0, float value1) {
    return interleave_bf16_pairs(_mm512_cvtne2ps_pbh(_mm512_set1_ps(value1), _mm512_set1_ps(value0)));
  }

  static inline __m512bh bf16_pair_broadcast(const ggml_bf16_t* values) {
    uint32_t pair;
    std::memcpy(&pair, values, sizeof(pair));
    return (__m512bh)_mm512_set1_epi32(static_cast<int>(pair));
  }

  static inline void rawint4_weight_pair_bf16(const uint8_t* packed, const float* scales, int row0, int row1,
                                               int cols, int group, __m512bh& weight_lo,
                                               __m512bh& weight_hi) {
    const int groups_per_row = cols / 32;
    const size_t row_bytes = static_cast<size_t>(cols) / 2;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    auto decode = [&](int row, __m512& lo, __m512& hi) {
      const uint8_t* src = packed + static_cast<size_t>(row) * row_bytes + static_cast<size_t>(group) * 16;
      const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
      const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
      const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
      const __m512 scale = _mm512_set1_ps(scales[static_cast<size_t>(row) * groups_per_row + group]);
      lo = _mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(low4, high4)), zero_point)),
          scale);
      hi = _mm512_mul_ps(
          _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(low4, high4)), zero_point)),
          scale);
    };

    __m512 row0_lo, row0_hi, row1_lo, row1_hi;
    decode(row0, row0_lo, row0_hi);
    decode(row1, row1_lo, row1_hi);
    weight_lo = interleave_bf16_pairs(_mm512_cvtne2ps_pbh(row1_lo, row0_lo));
    weight_hi = interleave_bf16_pairs(_mm512_cvtne2ps_pbh(row1_hi, row0_hi));
  }

  void rawint4_backward_matmat_bf16(const uint8_t* packed, const float* scales, const float* input, int m,
                                    int rows, int cols, int col_begin, int col_end, float* output) const {
    for (int m_begin = 0; m_begin < m; m_begin += 8) {
      const int m_count = std::min(8, m - m_begin);
      for (int col = col_begin; col < col_end; col += 32) {
        const int group = col / 32;
        __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
        __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
        __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
        __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
        __m512 acc4_lo = _mm512_setzero_ps(), acc4_hi = _mm512_setzero_ps();
        __m512 acc5_lo = _mm512_setzero_ps(), acc5_hi = _mm512_setzero_ps();
        __m512 acc6_lo = _mm512_setzero_ps(), acc6_hi = _mm512_setzero_ps();
        __m512 acc7_lo = _mm512_setzero_ps(), acc7_hi = _mm512_setzero_ps();

        for (int row = 0; row < rows; row += 2) {
          __m512bh weight_lo, weight_hi;
          rawint4_weight_pair_bf16(packed, scales, row, row + 1, cols, group, weight_lo, weight_hi);

#define KT_K2_SFT_BF16_BWD_ACC(M)                                                                                  \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      const float* input_row = input + static_cast<size_t>(m_begin + M) * rows;                                    \
      const __m512bh coeff = fp32_pair_to_bf16(input_row[row], input_row[row + 1]);                                \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, coeff, weight_lo);                                               \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, coeff, weight_hi);                                               \
    }                                                                                                               \
  } while (0)
          KT_K2_SFT_BF16_BWD_ACC(0);
          KT_K2_SFT_BF16_BWD_ACC(1);
          KT_K2_SFT_BF16_BWD_ACC(2);
          KT_K2_SFT_BF16_BWD_ACC(3);
          KT_K2_SFT_BF16_BWD_ACC(4);
          KT_K2_SFT_BF16_BWD_ACC(5);
          KT_K2_SFT_BF16_BWD_ACC(6);
          KT_K2_SFT_BF16_BWD_ACC(7);
#undef KT_K2_SFT_BF16_BWD_ACC
        }

#define KT_K2_SFT_BF16_BWD_STORE(M)                                                                                \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      float* dst = output + static_cast<size_t>(m_begin + M) * cols + col;                                         \
      _mm512_storeu_ps(dst, acc##M##_lo);                                                                           \
      _mm512_storeu_ps(dst + 16, acc##M##_hi);                                                                      \
    }                                                                                                               \
  } while (0)
        KT_K2_SFT_BF16_BWD_STORE(0);
        KT_K2_SFT_BF16_BWD_STORE(1);
        KT_K2_SFT_BF16_BWD_STORE(2);
        KT_K2_SFT_BF16_BWD_STORE(3);
        KT_K2_SFT_BF16_BWD_STORE(4);
        KT_K2_SFT_BF16_BWD_STORE(5);
        KT_K2_SFT_BF16_BWD_STORE(6);
        KT_K2_SFT_BF16_BWD_STORE(7);
#undef KT_K2_SFT_BF16_BWD_STORE
      }
    }
  }

  void rawint4_backward_matmat_bf16_input(const uint8_t* packed, const float* scales,
                                          const ggml_bf16_t* input, int m, int rows, int cols, int col_begin,
                                          int col_end, float* output) const {
    for (int m_begin = 0; m_begin < m; m_begin += 8) {
      const int m_count = std::min(8, m - m_begin);
      for (int col = col_begin; col < col_end; col += 32) {
        const int group = col / 32;
        __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
        __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
        __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
        __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
        __m512 acc4_lo = _mm512_setzero_ps(), acc4_hi = _mm512_setzero_ps();
        __m512 acc5_lo = _mm512_setzero_ps(), acc5_hi = _mm512_setzero_ps();
        __m512 acc6_lo = _mm512_setzero_ps(), acc6_hi = _mm512_setzero_ps();
        __m512 acc7_lo = _mm512_setzero_ps(), acc7_hi = _mm512_setzero_ps();

        for (int row = 0; row < rows; row += 2) {
          __m512bh weight_lo, weight_hi;
          rawint4_weight_pair_bf16(packed, scales, row, row + 1, cols, group, weight_lo, weight_hi);

#define KT_K2_SFT_BF16_INPUT_BWD_ACC(M)                                                                            \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      const ggml_bf16_t* input_row = input + static_cast<size_t>(m_begin + M) * rows;                              \
      const __m512bh coeff = bf16_pair_broadcast(input_row + row);                                                  \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, coeff, weight_lo);                                               \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, coeff, weight_hi);                                               \
    }                                                                                                               \
  } while (0)
          KT_K2_SFT_BF16_INPUT_BWD_ACC(0);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(1);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(2);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(3);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(4);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(5);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(6);
          KT_K2_SFT_BF16_INPUT_BWD_ACC(7);
#undef KT_K2_SFT_BF16_INPUT_BWD_ACC
        }

#define KT_K2_SFT_BF16_INPUT_BWD_STORE(M)                                                                          \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      float* dst = output + static_cast<size_t>(m_begin + M) * cols + col;                                         \
      _mm512_storeu_ps(dst, acc##M##_lo);                                                                           \
      _mm512_storeu_ps(dst + 16, acc##M##_hi);                                                                      \
    }                                                                                                               \
  } while (0)
        KT_K2_SFT_BF16_INPUT_BWD_STORE(0);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(1);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(2);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(3);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(4);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(5);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(6);
        KT_K2_SFT_BF16_INPUT_BWD_STORE(7);
#undef KT_K2_SFT_BF16_INPUT_BWD_STORE
      }
    }
  }

  void rawint4_gate_up_backward_matmat_bf16(
      const uint8_t* gate_packed, const float* gate_scales, const uint8_t* up_packed, const float* up_scales,
      const float* grad_gate, const float* grad_up, int m, int rows, int cols, int col_begin, int col_end,
      ggml_bf16_t* output) const {
    for (int m_begin = 0; m_begin < m; m_begin += 8) {
      const int m_count = std::min(8, m - m_begin);
      for (int col = col_begin; col < col_end; col += 32) {
        const int group = col / 32;
        __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
        __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
        __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
        __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
        __m512 acc4_lo = _mm512_setzero_ps(), acc4_hi = _mm512_setzero_ps();
        __m512 acc5_lo = _mm512_setzero_ps(), acc5_hi = _mm512_setzero_ps();
        __m512 acc6_lo = _mm512_setzero_ps(), acc6_hi = _mm512_setzero_ps();
        __m512 acc7_lo = _mm512_setzero_ps(), acc7_hi = _mm512_setzero_ps();

        for (int row = 0; row < rows; row += 2) {
          __m512bh gate_weight_lo, gate_weight_hi, up_weight_lo, up_weight_hi;
          rawint4_weight_pair_bf16(gate_packed, gate_scales, row, row + 1, cols, group, gate_weight_lo,
                                   gate_weight_hi);
          rawint4_weight_pair_bf16(up_packed, up_scales, row, row + 1, cols, group, up_weight_lo, up_weight_hi);

#define KT_K2_SFT_BF16_GATE_UP_ACC(M)                                                                              \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      const float* gate_row = grad_gate + static_cast<size_t>(m_begin + M) * rows;                                 \
      const float* up_row = grad_up + static_cast<size_t>(m_begin + M) * rows;                                     \
      const __m512bh gate_coeff = fp32_pair_to_bf16(gate_row[row], gate_row[row + 1]);                             \
      const __m512bh up_coeff = fp32_pair_to_bf16(up_row[row], up_row[row + 1]);                                   \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, gate_coeff, gate_weight_lo);                                     \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, up_coeff, up_weight_lo);                                         \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, gate_coeff, gate_weight_hi);                                     \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, up_coeff, up_weight_hi);                                         \
    }                                                                                                               \
  } while (0)
          KT_K2_SFT_BF16_GATE_UP_ACC(0);
          KT_K2_SFT_BF16_GATE_UP_ACC(1);
          KT_K2_SFT_BF16_GATE_UP_ACC(2);
          KT_K2_SFT_BF16_GATE_UP_ACC(3);
          KT_K2_SFT_BF16_GATE_UP_ACC(4);
          KT_K2_SFT_BF16_GATE_UP_ACC(5);
          KT_K2_SFT_BF16_GATE_UP_ACC(6);
          KT_K2_SFT_BF16_GATE_UP_ACC(7);
#undef KT_K2_SFT_BF16_GATE_UP_ACC
        }

#define KT_K2_SFT_BF16_GATE_UP_STORE(M)                                                                            \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      ggml_bf16_t* dst = output + static_cast<size_t>(m_begin + M) * cols + col;                                   \
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (__m256i)_mm512_cvtneps_pbh(acc##M##_lo));              \
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 16), (__m256i)_mm512_cvtneps_pbh(acc##M##_hi));         \
    }                                                                                                               \
  } while (0)
        KT_K2_SFT_BF16_GATE_UP_STORE(0);
        KT_K2_SFT_BF16_GATE_UP_STORE(1);
        KT_K2_SFT_BF16_GATE_UP_STORE(2);
        KT_K2_SFT_BF16_GATE_UP_STORE(3);
        KT_K2_SFT_BF16_GATE_UP_STORE(4);
        KT_K2_SFT_BF16_GATE_UP_STORE(5);
        KT_K2_SFT_BF16_GATE_UP_STORE(6);
        KT_K2_SFT_BF16_GATE_UP_STORE(7);
#undef KT_K2_SFT_BF16_GATE_UP_STORE
      }
    }
  }

  void rawint4_gate_up_backward_matmat_bf16_direct(
      const uint8_t* gate_packed, const float* gate_scales, const uint8_t* up_packed, const float* up_scales,
      const float* grad_gate, const float* grad_up, int m, int rows, int cols, int col_begin, int col_end,
      const int* row_to_token, float* token_output) const {
    for (int m_begin = 0; m_begin < m; m_begin += 8) {
      const int m_count = std::min(8, m - m_begin);
      for (int col = col_begin; col < col_end; col += 32) {
        const int group = col / 32;
        __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
        __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
        __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
        __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
        __m512 acc4_lo = _mm512_setzero_ps(), acc4_hi = _mm512_setzero_ps();
        __m512 acc5_lo = _mm512_setzero_ps(), acc5_hi = _mm512_setzero_ps();
        __m512 acc6_lo = _mm512_setzero_ps(), acc6_hi = _mm512_setzero_ps();
        __m512 acc7_lo = _mm512_setzero_ps(), acc7_hi = _mm512_setzero_ps();

        for (int row = 0; row < rows; row += 2) {
          __m512bh gate_weight_lo, gate_weight_hi, up_weight_lo, up_weight_hi;
          rawint4_weight_pair_bf16(gate_packed, gate_scales, row, row + 1, cols, group, gate_weight_lo,
                                   gate_weight_hi);
          rawint4_weight_pair_bf16(up_packed, up_scales, row, row + 1, cols, group, up_weight_lo, up_weight_hi);

#define KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(M)                                                                        \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      const float* gate_row = grad_gate + static_cast<size_t>(m_begin + M) * rows;                                 \
      const float* up_row = grad_up + static_cast<size_t>(m_begin + M) * rows;                                     \
      const __m512bh gate_coeff = fp32_pair_to_bf16(gate_row[row], gate_row[row + 1]);                             \
      const __m512bh up_coeff = fp32_pair_to_bf16(up_row[row], up_row[row + 1]);                                   \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, gate_coeff, gate_weight_lo);                                     \
      acc##M##_lo = _mm512_dpbf16_ps(acc##M##_lo, up_coeff, up_weight_lo);                                         \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, gate_coeff, gate_weight_hi);                                     \
      acc##M##_hi = _mm512_dpbf16_ps(acc##M##_hi, up_coeff, up_weight_hi);                                         \
    }                                                                                                               \
  } while (0)
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(0);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(1);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(2);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(3);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(4);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(5);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(6);
          KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC(7);
#undef KT_K2_SFT_BF16_GATE_UP_DIRECT_ACC
        }

#define KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(M)                                                                      \
  do {                                                                                                              \
    if (m_count > M) {                                                                                              \
      float* dst = token_output + static_cast<size_t>(row_to_token[m_begin + M]) * cols + col;                     \
      _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), acc##M##_lo));                                     \
      _mm512_storeu_ps(dst + 16, _mm512_add_ps(_mm512_loadu_ps(dst + 16), acc##M##_hi));                           \
    }                                                                                                               \
  } while (0)
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(0);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(1);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(2);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(3);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(4);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(5);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(6);
        KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE(7);
#undef KT_K2_SFT_BF16_GATE_UP_DIRECT_STORE
      }
    }
  }
#endif

  // Compute eight expert-packed rows together so each RAWINT4 KGroup=32
  // weight tile is unpacked and dequantized once, then reused by all rows.
  // Accumulation stays in FP32 registers for the full K dimension.  This is
  // the production large-M backward path; smaller expert batches keep the
  // existing packed GEMV helpers below.
  void rawint4_backward_matmat_m8_f32(const uint8_t* packed, const float* scales, const float* input, int rows,
                                      int cols, float* output) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size != 32 || (cols % 32) != 0) {
      for (int m = 0; m < 8; m++) {
        add_scaled_packed_rows_f32(packed, scales, input + static_cast<size_t>(m) * rows, rows, cols,
                                   output + static_cast<size_t>(m) * cols, true);
      }
      return;
    }

    const int groups_per_row = cols / 32;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      const int col = group * 32;
      __m512 acc0_lo = _mm512_loadu_ps(output + col);
      __m512 acc0_hi = _mm512_loadu_ps(output + col + 16);
      __m512 acc1_lo = _mm512_loadu_ps(output + static_cast<size_t>(1) * cols + col);
      __m512 acc1_hi = _mm512_loadu_ps(output + static_cast<size_t>(1) * cols + col + 16);
      __m512 acc2_lo = _mm512_loadu_ps(output + static_cast<size_t>(2) * cols + col);
      __m512 acc2_hi = _mm512_loadu_ps(output + static_cast<size_t>(2) * cols + col + 16);
      __m512 acc3_lo = _mm512_loadu_ps(output + static_cast<size_t>(3) * cols + col);
      __m512 acc3_hi = _mm512_loadu_ps(output + static_cast<size_t>(3) * cols + col + 16);
      __m512 acc4_lo = _mm512_loadu_ps(output + static_cast<size_t>(4) * cols + col);
      __m512 acc4_hi = _mm512_loadu_ps(output + static_cast<size_t>(4) * cols + col + 16);
      __m512 acc5_lo = _mm512_loadu_ps(output + static_cast<size_t>(5) * cols + col);
      __m512 acc5_hi = _mm512_loadu_ps(output + static_cast<size_t>(5) * cols + col + 16);
      __m512 acc6_lo = _mm512_loadu_ps(output + static_cast<size_t>(6) * cols + col);
      __m512 acc6_hi = _mm512_loadu_ps(output + static_cast<size_t>(6) * cols + col + 16);
      __m512 acc7_lo = _mm512_loadu_ps(output + static_cast<size_t>(7) * cols + col);
      __m512 acc7_hi = _mm512_loadu_ps(output + static_cast<size_t>(7) * cols + col + 16);

      for (int row = 0; row < rows; row++) {
        const uint8_t* group_packed =
            packed + static_cast<size_t>(row) * (cols / 2) + static_cast<size_t>(group) * 16;
        const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
        const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
        const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
        const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
        const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
        const __m512 scale = _mm512_set1_ps(scales[static_cast<size_t>(row) * groups_per_row + group]);
        const __m512 weight_lo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point)), scale);
        const __m512 weight_hi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point)), scale);

#define KT_K2_SFT_M8_FMA(M)                                                                                       \
  do {                                                                                                             \
    const __m512 coeff = _mm512_set1_ps(input[static_cast<size_t>(M) * rows + row]);                               \
    acc##M##_lo = _mm512_fmadd_ps(weight_lo, coeff, acc##M##_lo);                                                  \
    acc##M##_hi = _mm512_fmadd_ps(weight_hi, coeff, acc##M##_hi);                                                  \
  } while (0)
        KT_K2_SFT_M8_FMA(0);
        KT_K2_SFT_M8_FMA(1);
        KT_K2_SFT_M8_FMA(2);
        KT_K2_SFT_M8_FMA(3);
        KT_K2_SFT_M8_FMA(4);
        KT_K2_SFT_M8_FMA(5);
        KT_K2_SFT_M8_FMA(6);
        KT_K2_SFT_M8_FMA(7);
#undef KT_K2_SFT_M8_FMA
      }

      _mm512_storeu_ps(output + col, acc0_lo);
      _mm512_storeu_ps(output + col + 16, acc0_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(1) * cols + col, acc1_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(1) * cols + col + 16, acc1_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(2) * cols + col, acc2_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(2) * cols + col + 16, acc2_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(3) * cols + col, acc3_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(3) * cols + col + 16, acc3_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(4) * cols + col, acc4_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(4) * cols + col + 16, acc4_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(5) * cols + col, acc5_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(5) * cols + col + 16, acc5_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(6) * cols + col, acc6_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(6) * cols + col + 16, acc6_hi);
      _mm512_storeu_ps(output + static_cast<size_t>(7) * cols + col, acc7_lo);
      _mm512_storeu_ps(output + static_cast<size_t>(7) * cols + col + 16, acc7_hi);
    }
  }

  // Fused gate/up backward for eight expert-packed route rows.  Gate and up
  // share the K/N traversal and the FP32 accumulators; every packed weight
  // tile is unpacked once for the eight routes.  The route result is stored as
  // BF16 for the subsequent contention-free top-k token reduction.
  void rawint4_gate_up_backward_matmat_m8_bf16(
      const uint8_t* gate_packed, const float* gate_scales, const uint8_t* up_packed, const float* up_scales,
      const float* grad_gate, const float* grad_up, int rows, int cols, ggml_bf16_t* output) const {
    const int groups_per_row = cols / 32;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      const int col = group * 32;
      __m512 acc0_lo = _mm512_setzero_ps();
      __m512 acc0_hi = _mm512_setzero_ps();
      __m512 acc1_lo = _mm512_setzero_ps();
      __m512 acc1_hi = _mm512_setzero_ps();
      __m512 acc2_lo = _mm512_setzero_ps();
      __m512 acc2_hi = _mm512_setzero_ps();
      __m512 acc3_lo = _mm512_setzero_ps();
      __m512 acc3_hi = _mm512_setzero_ps();
      __m512 acc4_lo = _mm512_setzero_ps();
      __m512 acc4_hi = _mm512_setzero_ps();
      __m512 acc5_lo = _mm512_setzero_ps();
      __m512 acc5_hi = _mm512_setzero_ps();
      __m512 acc6_lo = _mm512_setzero_ps();
      __m512 acc6_hi = _mm512_setzero_ps();
      __m512 acc7_lo = _mm512_setzero_ps();
      __m512 acc7_hi = _mm512_setzero_ps();

      for (int row = 0; row < rows; row++) {
        const size_t packed_offset = static_cast<size_t>(row) * (cols / 2) + static_cast<size_t>(group) * 16;
        const size_t scale_offset = static_cast<size_t>(row) * groups_per_row + group;
        const __m128i gate16 =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(gate_packed + packed_offset));
        const __m128i up16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(up_packed + packed_offset));

        const __m128i gate_low4 = _mm_and_si128(gate16, nibble_mask);
        const __m128i gate_high4 = _mm_and_si128(_mm_srli_epi16(gate16, 4), nibble_mask);
        const __m128i up_low4 = _mm_and_si128(up16, nibble_mask);
        const __m128i up_high4 = _mm_and_si128(_mm_srli_epi16(up16, 4), nibble_mask);
        const __m512 gate_scale = _mm512_set1_ps(gate_scales[scale_offset]);
        const __m512 up_scale = _mm512_set1_ps(up_scales[scale_offset]);

        const __m512 gate_weight_lo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(
                _mm512_cvtepu8_epi32(_mm_unpacklo_epi8(gate_low4, gate_high4)), zero_point)),
            gate_scale);
        const __m512 gate_weight_hi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_sub_epi32(
                _mm512_cvtepu8_epi32(_mm_unpackhi_epi8(gate_low4, gate_high4)), zero_point)),
            gate_scale);
        const __m512 up_weight_lo = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_unpacklo_epi8(up_low4, up_high4)), zero_point)),
            up_scale);
        const __m512 up_weight_hi = _mm512_mul_ps(
            _mm512_cvtepi32_ps(
                _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_unpackhi_epi8(up_low4, up_high4)), zero_point)),
            up_scale);

#define KT_K2_SFT_GATE_UP_M8_FMA(M)                                                                               \
  do {                                                                                                             \
    const __m512 gate_coeff = _mm512_set1_ps(grad_gate[static_cast<size_t>(M) * rows + row]);                      \
    const __m512 up_coeff = _mm512_set1_ps(grad_up[static_cast<size_t>(M) * rows + row]);                          \
    acc##M##_lo = _mm512_fmadd_ps(gate_weight_lo, gate_coeff, acc##M##_lo);                                        \
    acc##M##_lo = _mm512_fmadd_ps(up_weight_lo, up_coeff, acc##M##_lo);                                            \
    acc##M##_hi = _mm512_fmadd_ps(gate_weight_hi, gate_coeff, acc##M##_hi);                                        \
    acc##M##_hi = _mm512_fmadd_ps(up_weight_hi, up_coeff, acc##M##_hi);                                            \
  } while (0)
        KT_K2_SFT_GATE_UP_M8_FMA(0);
        KT_K2_SFT_GATE_UP_M8_FMA(1);
        KT_K2_SFT_GATE_UP_M8_FMA(2);
        KT_K2_SFT_GATE_UP_M8_FMA(3);
        KT_K2_SFT_GATE_UP_M8_FMA(4);
        KT_K2_SFT_GATE_UP_M8_FMA(5);
        KT_K2_SFT_GATE_UP_M8_FMA(6);
        KT_K2_SFT_GATE_UP_M8_FMA(7);
#undef KT_K2_SFT_GATE_UP_M8_FMA
      }

#define KT_K2_SFT_GATE_UP_M8_STORE(M)                                                                             \
  do {                                                                                                             \
    ggml_bf16_t* dst = output + static_cast<size_t>(M) * cols + col;                                               \
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (__m256i)_mm512_cvtneps_pbh(acc##M##_lo));                \
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 16), (__m256i)_mm512_cvtneps_pbh(acc##M##_hi));           \
  } while (0)
      KT_K2_SFT_GATE_UP_M8_STORE(0);
      KT_K2_SFT_GATE_UP_M8_STORE(1);
      KT_K2_SFT_GATE_UP_M8_STORE(2);
      KT_K2_SFT_GATE_UP_M8_STORE(3);
      KT_K2_SFT_GATE_UP_M8_STORE(4);
      KT_K2_SFT_GATE_UP_M8_STORE(5);
      KT_K2_SFT_GATE_UP_M8_STORE(6);
      KT_K2_SFT_GATE_UP_M8_STORE(7);
#undef KT_K2_SFT_GATE_UP_M8_STORE
    }
  }

  void add_scaled_packed_row_f32(const uint8_t* packed, const float* scales, int row, int cols, float coeff,
                                 float* dst) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    const uint8_t* row_packed = packed + static_cast<size_t>(row) * (cols / 2);
    const float* row_scales = scales + static_cast<size_t>(row) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      const float scale = coeff * row_scales[group];
      const uint8_t* group_packed = row_packed + static_cast<size_t>(group) * (group_size / 2);
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      if (group_size == 32) {
        const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
        const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
        const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
        const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
        const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
        const __m512 scale_vec = _mm512_set1_ps(scale);

        __m512i vals0 = _mm512_cvtepu8_epi32(interleaved_lo);
        vals0 = _mm512_sub_epi32(vals0, zero_point);
        __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(vals0), scale_vec);
        f0 = _mm512_add_ps(_mm512_loadu_ps(group_dst), f0);
        _mm512_storeu_ps(group_dst, f0);

        __m512i vals1 = _mm512_cvtepu8_epi32(interleaved_hi);
        vals1 = _mm512_sub_epi32(vals1, zero_point);
        __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(vals1), scale_vec);
        f1 = _mm512_add_ps(_mm512_loadu_ps(group_dst + 16), f1);
        _mm512_storeu_ps(group_dst + 16, f1);
        continue;
      }
      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx++) {
        const uint8_t byte = group_packed[byte_idx];
        group_dst[byte_idx * 2] += scale * static_cast<float>((byte & 0x0f) - 8);
        group_dst[byte_idx * 2 + 1] += scale * static_cast<float>(((byte >> 4) & 0x0f) - 8);
      }
    }
  }

  void add_scaled_two_packed_rows_f32(const uint8_t* packed_a, const float* scales_a, int row_a, float coeff_a,
                                      const uint8_t* packed_b, const float* scales_b, int row_b, float coeff_b,
                                      int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    const uint8_t* row_a_packed = packed_a + static_cast<size_t>(row_a) * (cols / 2);
    const uint8_t* row_b_packed = packed_b + static_cast<size_t>(row_b) * (cols / 2);
    const float* row_a_scales = scales_a + static_cast<size_t>(row_a) * groups_per_row;
    const float* row_b_scales = scales_b + static_cast<size_t>(row_b) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      const float scale_a = coeff_a * row_a_scales[group];
      const float scale_b = coeff_b * row_b_scales[group];
      const uint8_t* group_a_packed = row_a_packed + static_cast<size_t>(group) * (group_size / 2);
      const uint8_t* group_b_packed = row_b_packed + static_cast<size_t>(group) * (group_size / 2);
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      if (group_size == 32) {
        const __m512 scale_a_vec = _mm512_set1_ps(scale_a);
        const __m512 scale_b_vec = _mm512_set1_ps(scale_b);
        const __m128i packed_a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_a_packed));
        const __m128i packed_b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_b_packed));

        const __m128i a_low4 = _mm_and_si128(packed_a16, nibble_mask);
        const __m128i a_high4 = _mm_and_si128(_mm_srli_epi16(packed_a16, 4), nibble_mask);
        const __m128i b_low4 = _mm_and_si128(packed_b16, nibble_mask);
        const __m128i b_high4 = _mm_and_si128(_mm_srli_epi16(packed_b16, 4), nibble_mask);

        const __m128i a_interleaved_lo = _mm_unpacklo_epi8(a_low4, a_high4);
        const __m128i a_interleaved_hi = _mm_unpackhi_epi8(a_low4, a_high4);
        const __m128i b_interleaved_lo = _mm_unpacklo_epi8(b_low4, b_high4);
        const __m128i b_interleaved_hi = _mm_unpackhi_epi8(b_low4, b_high4);

        __m512i vals_a0 = _mm512_cvtepu8_epi32(a_interleaved_lo);
        __m512i vals_b0 = _mm512_cvtepu8_epi32(b_interleaved_lo);
        vals_a0 = _mm512_sub_epi32(vals_a0, zero_point);
        vals_b0 = _mm512_sub_epi32(vals_b0, zero_point);
        __m512 f0 = _mm512_loadu_ps(group_dst);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals_a0), scale_a_vec, f0);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals_b0), scale_b_vec, f0);
        _mm512_storeu_ps(group_dst, f0);

        __m512i vals_a1 = _mm512_cvtepu8_epi32(a_interleaved_hi);
        __m512i vals_b1 = _mm512_cvtepu8_epi32(b_interleaved_hi);
        vals_a1 = _mm512_sub_epi32(vals_a1, zero_point);
        vals_b1 = _mm512_sub_epi32(vals_b1, zero_point);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals_a1), scale_a_vec, f1);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals_b1), scale_b_vec, f1);
        _mm512_storeu_ps(group_dst + 16, f1);
        continue;
      }
      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx++) {
        const uint8_t byte_a = group_a_packed[byte_idx];
        const uint8_t byte_b = group_b_packed[byte_idx];
        group_dst[byte_idx * 2] +=
            scale_a * static_cast<float>((byte_a & 0x0f) - 8) + scale_b * static_cast<float>((byte_b & 0x0f) - 8);
        group_dst[byte_idx * 2 + 1] += scale_a * static_cast<float>(((byte_a >> 4) & 0x0f) - 8) +
                                       scale_b * static_cast<float>(((byte_b >> 4) & 0x0f) - 8);
      }
    }
  }

  void add_scaled_four_packed_rows_f32(const uint8_t* packed0, const float* scales0, int row0, float coeff0,
                                       const uint8_t* packed1, const float* scales1, int row1, float coeff1,
                                       const uint8_t* packed2, const float* scales2, int row2, float coeff2,
                                       const uint8_t* packed3, const float* scales3, int row3, float coeff3, int cols,
                                       float* dst) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    const uint8_t* row0_packed = packed0 + static_cast<size_t>(row0) * (cols / 2);
    const uint8_t* row1_packed = packed1 + static_cast<size_t>(row1) * (cols / 2);
    const uint8_t* row2_packed = packed2 + static_cast<size_t>(row2) * (cols / 2);
    const uint8_t* row3_packed = packed3 + static_cast<size_t>(row3) * (cols / 2);
    const float* row0_scales = scales0 + static_cast<size_t>(row0) * groups_per_row;
    const float* row1_scales = scales1 + static_cast<size_t>(row1) * groups_per_row;
    const float* row2_scales = scales2 + static_cast<size_t>(row2) * groups_per_row;
    const float* row3_scales = scales3 + static_cast<size_t>(row3) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      const float scale0 = coeff0 * row0_scales[group];
      const float scale1 = coeff1 * row1_scales[group];
      const float scale2 = coeff2 * row2_scales[group];
      const float scale3 = coeff3 * row3_scales[group];
      const uint8_t* group0_packed = row0_packed + static_cast<size_t>(group) * (group_size / 2);
      const uint8_t* group1_packed = row1_packed + static_cast<size_t>(group) * (group_size / 2);
      const uint8_t* group2_packed = row2_packed + static_cast<size_t>(group) * (group_size / 2);
      const uint8_t* group3_packed = row3_packed + static_cast<size_t>(group) * (group_size / 2);
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      if (group_size == 32) {
        const __m512 scale0_vec = _mm512_set1_ps(scale0);
        const __m512 scale1_vec = _mm512_set1_ps(scale1);
        const __m512 scale2_vec = _mm512_set1_ps(scale2);
        const __m512 scale3_vec = _mm512_set1_ps(scale3);
        const __m128i packed0_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group0_packed));
        const __m128i packed1_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group1_packed));
        const __m128i packed2_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group2_packed));
        const __m128i packed3_16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group3_packed));

        __m128i lo0;
        __m128i hi0;
        __m128i lo1;
        __m128i hi1;
        __m128i lo2;
        __m128i hi2;
        __m128i lo3;
        __m128i hi3;
        auto unpack_nibbles = [&](const __m128i packed16, __m128i& lo, __m128i& hi) {
          const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
          const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
          lo = _mm_unpacklo_epi8(low4, high4);
          hi = _mm_unpackhi_epi8(low4, high4);
        };
        unpack_nibbles(packed0_16, lo0, hi0);
        unpack_nibbles(packed1_16, lo1, hi1);
        unpack_nibbles(packed2_16, lo2, hi2);
        unpack_nibbles(packed3_16, lo3, hi3);

        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(lo0), zero_point);
        __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(lo1), zero_point);
        __m512i vals2 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(lo2), zero_point);
        __m512i vals3 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(lo3), zero_point);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale0_vec, f0);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale1_vec, f0);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals2), scale2_vec, f0);
        f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals3), scale3_vec, f0);
        _mm512_storeu_ps(group_dst, f0);

        __m512 f1 = _mm512_loadu_ps(group_dst + 16);
        vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(hi0), zero_point);
        vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(hi1), zero_point);
        vals2 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(hi2), zero_point);
        vals3 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(hi3), zero_point);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale0_vec, f1);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale1_vec, f1);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals2), scale2_vec, f1);
        f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals3), scale3_vec, f1);
        _mm512_storeu_ps(group_dst + 16, f1);
        continue;
      }
      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx++) {
        const uint8_t byte0 = group0_packed[byte_idx];
        const uint8_t byte1 = group1_packed[byte_idx];
        const uint8_t byte2 = group2_packed[byte_idx];
        const uint8_t byte3 = group3_packed[byte_idx];
        group_dst[byte_idx * 2] +=
            scale0 * static_cast<float>((byte0 & 0x0f) - 8) + scale1 * static_cast<float>((byte1 & 0x0f) - 8) +
            scale2 * static_cast<float>((byte2 & 0x0f) - 8) + scale3 * static_cast<float>((byte3 & 0x0f) - 8);
        group_dst[byte_idx * 2 + 1] += scale0 * static_cast<float>(((byte0 >> 4) & 0x0f) - 8) +
                                       scale1 * static_cast<float>(((byte1 >> 4) & 0x0f) - 8) +
                                       scale2 * static_cast<float>(((byte2 >> 4) & 0x0f) - 8) +
                                       scale3 * static_cast<float>(((byte3 >> 4) & 0x0f) - 8);
      }
    }
  }

  void add_scaled_eight_packed_rows_f32(const uint8_t* const* packed_list, const float* const* scales_list,
                                        const int* rows, const float* coeffs, int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    const uint8_t* row_packed[8];
    const float* row_scales[8];
    for (int src = 0; src < 8; src++) {
      row_packed[src] = packed_list[src] + static_cast<size_t>(rows[src]) * (cols / 2);
      row_scales[src] = scales_list[src] + static_cast<size_t>(rows[src]) * groups_per_row;
    }

    if (group_size == 32 && dense_coeff_fastpath_enabled()) {
      const __m512i zero_point = _mm512_set1_epi32(8);
      const __m128i nibble_mask = _mm_set1_epi8(0x0f);
      for (int group = 0; group < groups_per_row; group++) {
        float* group_dst = dst + static_cast<size_t>(group) * group_size;
        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);

        for (int src = 0; src < 8; src++) {
          const __m512 scale_vec = _mm512_set1_ps(coeffs[src] * row_scales[src][group]);
          const uint8_t* group_packed = row_packed[src] + static_cast<size_t>(group) * (group_size / 2);
          const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
          const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
          const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
          const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
          const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
          const __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point);
          const __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point);
          f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale_vec, f0);
          f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale_vec, f1);
        }

        _mm512_storeu_ps(group_dst, f0);
        _mm512_storeu_ps(group_dst + 16, f1);
      }
      return;
    }

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      if (group_size == 32) {
        const __m512i zero_point = _mm512_set1_epi32(8);
        const __m128i nibble_mask = _mm_set1_epi8(0x0f);
        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);
        for (int src = 0; src < 8; src++) {
          const float coeff = coeffs[src];
          if (coeff == 0.0f) continue;
          const float scale = coeff * row_scales[src][group];
          const __m512 scale_vec = _mm512_set1_ps(scale);
          const uint8_t* group_packed = row_packed[src] + static_cast<size_t>(group) * (group_size / 2);
          const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
          const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
          const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
          const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
          const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
          __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point);
          __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point);
          f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale_vec, f0);
          f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale_vec, f1);
        }
        _mm512_storeu_ps(group_dst, f0);
        _mm512_storeu_ps(group_dst + 16, f1);
        continue;
      }

      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx++) {
        float even_acc = 0.0f;
        float odd_acc = 0.0f;
        for (int src = 0; src < 8; src++) {
          const float coeff = coeffs[src];
          if (coeff == 0.0f) continue;
          const float scale = coeff * row_scales[src][group];
          const uint8_t byte = row_packed[src][static_cast<size_t>(group) * (group_size / 2) + byte_idx];
          even_acc += scale * static_cast<float>((byte & 0x0f) - 8);
          odd_acc += scale * static_cast<float>(((byte >> 4) & 0x0f) - 8);
        }
        group_dst[byte_idx * 2] += even_acc;
        group_dst[byte_idx * 2 + 1] += odd_acc;
      }
    }
  }

  void add_scaled_four_gate_up_rows_f32(const uint8_t* gate_packed, const float* gate_scales, const uint8_t* up_packed,
                                        const float* up_scales, int row, const float* gate_coeffs,
                                        const float* up_coeffs, int cols, float* dst) const {
    const uint8_t* packed_list[8] = {gate_packed, up_packed, gate_packed, up_packed,
                                     gate_packed, up_packed, gate_packed, up_packed};
    const float* scales_list[8] = {gate_scales, up_scales, gate_scales, up_scales,
                                   gate_scales, up_scales, gate_scales, up_scales};
    const int rows[8] = {row, row, row + 1, row + 1, row + 2, row + 2, row + 3, row + 3};
    const float coeffs[8] = {gate_coeffs[0], up_coeffs[0], gate_coeffs[1], up_coeffs[1],
                             gate_coeffs[2], up_coeffs[2], gate_coeffs[3], up_coeffs[3]};
    add_scaled_eight_packed_rows_f32(packed_list, scales_list, rows, coeffs, cols, dst);
  }

  void add_scaled_sixteen_packed_rows_f32(const uint8_t* const* packed_list, const float* const* scales_list,
                                          const int* rows, const float* coeffs, int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    const int groups_per_row = cols / group_size;
    const uint8_t* row_packed[16];
    const float* row_scales[16];
    for (int src = 0; src < 16; src++) {
      row_packed[src] = packed_list[src] + static_cast<size_t>(rows[src]) * (cols / 2);
      row_scales[src] = scales_list[src] + static_cast<size_t>(rows[src]) * groups_per_row;
    }

    if (group_size == 32 && dense_coeff_fastpath_enabled()) {
      const __m512i zero_point = _mm512_set1_epi32(8);
      const __m128i nibble_mask = _mm_set1_epi8(0x0f);
      for (int group = 0; group < groups_per_row; group++) {
        float* group_dst = dst + static_cast<size_t>(group) * group_size;
        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);

        for (int src = 0; src < 16; src++) {
          const __m512 scale_vec = _mm512_set1_ps(coeffs[src] * row_scales[src][group]);
          const uint8_t* group_packed = row_packed[src] + static_cast<size_t>(group) * (group_size / 2);
          const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
          const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
          const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
          const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
          const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
          const __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point);
          const __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point);
          f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale_vec, f0);
          f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale_vec, f1);
        }

        _mm512_storeu_ps(group_dst, f0);
        _mm512_storeu_ps(group_dst + 16, f1);
      }
      return;
    }

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      if (group_size == 32) {
        const __m512i zero_point = _mm512_set1_epi32(8);
        const __m128i nibble_mask = _mm_set1_epi8(0x0f);
        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);
        for (int src = 0; src < 16; src++) {
          const float coeff = coeffs[src];
          if (coeff == 0.0f) continue;
          const __m512 scale_vec = _mm512_set1_ps(coeff * row_scales[src][group]);
          const uint8_t* group_packed = row_packed[src] + static_cast<size_t>(group) * (group_size / 2);
          const __m128i packed16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(group_packed));
          const __m128i low4 = _mm_and_si128(packed16, nibble_mask);
          const __m128i high4 = _mm_and_si128(_mm_srli_epi16(packed16, 4), nibble_mask);
          const __m128i interleaved_lo = _mm_unpacklo_epi8(low4, high4);
          const __m128i interleaved_hi = _mm_unpackhi_epi8(low4, high4);
          const __m512i vals0 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_lo), zero_point);
          const __m512i vals1 = _mm512_sub_epi32(_mm512_cvtepu8_epi32(interleaved_hi), zero_point);
          f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals0), scale_vec, f0);
          f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vals1), scale_vec, f1);
        }
        _mm512_storeu_ps(group_dst, f0);
        _mm512_storeu_ps(group_dst + 16, f1);
        continue;
      }

      for (int byte_idx = 0; byte_idx < group_size / 2; byte_idx++) {
        float even_acc = 0.0f;
        float odd_acc = 0.0f;
        for (int src = 0; src < 16; src++) {
          const float coeff = coeffs[src];
          if (coeff == 0.0f) continue;
          const float scale = coeff * row_scales[src][group];
          const uint8_t byte = row_packed[src][static_cast<size_t>(group) * (group_size / 2) + byte_idx];
          even_acc += scale * static_cast<float>((byte & 0x0f) - 8);
          odd_acc += scale * static_cast<float>(((byte >> 4) & 0x0f) - 8);
        }
        group_dst[byte_idx * 2] += even_acc;
        group_dst[byte_idx * 2 + 1] += odd_acc;
      }
    }
  }

  void add_scaled_sixteen_contiguous_packed_rows_f32(const uint8_t* packed, const float* scales, int row,
                                                     const float* coeffs, int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size != 32 || !dense_coeff_fastpath_enabled()) {
      const uint8_t* packed_list[16] = {packed, packed, packed, packed, packed, packed, packed, packed,
                                        packed, packed, packed, packed, packed, packed, packed, packed};
      const float* scales_list[16] = {scales, scales, scales, scales, scales, scales, scales, scales,
                                      scales, scales, scales, scales, scales, scales, scales, scales};
      const int row_ids[16] = {row,     row + 1, row + 2,  row + 3,  row + 4,  row + 5,  row + 6,  row + 7,
                               row + 8, row + 9, row + 10, row + 11, row + 12, row + 13, row + 14, row + 15};
      add_scaled_sixteen_packed_rows_f32(packed_list, scales_list, row_ids, coeffs, cols, dst);
      return;
    }

    const int groups_per_row = cols / group_size;
    const size_t packed_row_stride = static_cast<size_t>(cols / 2);
    const uint8_t* row_packed = packed + static_cast<size_t>(row) * packed_row_stride;
    const float* row_scales = scales + static_cast<size_t>(row) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      __m512 f0 = _mm512_loadu_ps(group_dst);
      __m512 f1 = _mm512_loadu_ps(group_dst + 16);

#define KT_K2_ACCUM_CONTIG_SRC(SRC)                                                                               \
  fmadd_int4_group32(row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16, \
                     _mm512_set1_ps(coeffs[SRC] * row_scales[static_cast<size_t>(SRC) * groups_per_row + group]), \
                     zero_point, nibble_mask, f0, f1)
      KT_K2_ACCUM_CONTIG_SRC(0);
      KT_K2_ACCUM_CONTIG_SRC(1);
      KT_K2_ACCUM_CONTIG_SRC(2);
      KT_K2_ACCUM_CONTIG_SRC(3);
      KT_K2_ACCUM_CONTIG_SRC(4);
      KT_K2_ACCUM_CONTIG_SRC(5);
      KT_K2_ACCUM_CONTIG_SRC(6);
      KT_K2_ACCUM_CONTIG_SRC(7);
      KT_K2_ACCUM_CONTIG_SRC(8);
      KT_K2_ACCUM_CONTIG_SRC(9);
      KT_K2_ACCUM_CONTIG_SRC(10);
      KT_K2_ACCUM_CONTIG_SRC(11);
      KT_K2_ACCUM_CONTIG_SRC(12);
      KT_K2_ACCUM_CONTIG_SRC(13);
      KT_K2_ACCUM_CONTIG_SRC(14);
      KT_K2_ACCUM_CONTIG_SRC(15);
#undef KT_K2_ACCUM_CONTIG_SRC

      _mm512_storeu_ps(group_dst, f0);
      _mm512_storeu_ps(group_dst + 16, f1);
    }
  }

  void add_scaled_thirtytwo_contiguous_packed_rows_f32(const uint8_t* packed, const float* scales, int row,
                                                       const float* coeffs, int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size != 32 || !dense_coeff_fastpath_enabled()) {
      add_scaled_sixteen_contiguous_packed_rows_f32(packed, scales, row, coeffs, cols, dst);
      add_scaled_sixteen_contiguous_packed_rows_f32(packed, scales, row + 16, coeffs + 16, cols, dst);
      return;
    }

    const int groups_per_row = cols / group_size;
    const size_t packed_row_stride = static_cast<size_t>(cols / 2);
    const uint8_t* row_packed = packed + static_cast<size_t>(row) * packed_row_stride;
    const float* row_scales = scales + static_cast<size_t>(row) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      __m512 f0 = _mm512_loadu_ps(group_dst);
      __m512 f1 = _mm512_loadu_ps(group_dst + 16);

#define KT_K2_ACCUM_CONTIG32_SRC(SRC)                                                                             \
  fmadd_int4_group32(row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16, \
                     _mm512_set1_ps(coeffs[SRC] * row_scales[static_cast<size_t>(SRC) * groups_per_row + group]), \
                     zero_point, nibble_mask, f0, f1)
      KT_K2_ACCUM_CONTIG32_SRC(0);
      KT_K2_ACCUM_CONTIG32_SRC(1);
      KT_K2_ACCUM_CONTIG32_SRC(2);
      KT_K2_ACCUM_CONTIG32_SRC(3);
      KT_K2_ACCUM_CONTIG32_SRC(4);
      KT_K2_ACCUM_CONTIG32_SRC(5);
      KT_K2_ACCUM_CONTIG32_SRC(6);
      KT_K2_ACCUM_CONTIG32_SRC(7);
      KT_K2_ACCUM_CONTIG32_SRC(8);
      KT_K2_ACCUM_CONTIG32_SRC(9);
      KT_K2_ACCUM_CONTIG32_SRC(10);
      KT_K2_ACCUM_CONTIG32_SRC(11);
      KT_K2_ACCUM_CONTIG32_SRC(12);
      KT_K2_ACCUM_CONTIG32_SRC(13);
      KT_K2_ACCUM_CONTIG32_SRC(14);
      KT_K2_ACCUM_CONTIG32_SRC(15);
      KT_K2_ACCUM_CONTIG32_SRC(16);
      KT_K2_ACCUM_CONTIG32_SRC(17);
      KT_K2_ACCUM_CONTIG32_SRC(18);
      KT_K2_ACCUM_CONTIG32_SRC(19);
      KT_K2_ACCUM_CONTIG32_SRC(20);
      KT_K2_ACCUM_CONTIG32_SRC(21);
      KT_K2_ACCUM_CONTIG32_SRC(22);
      KT_K2_ACCUM_CONTIG32_SRC(23);
      KT_K2_ACCUM_CONTIG32_SRC(24);
      KT_K2_ACCUM_CONTIG32_SRC(25);
      KT_K2_ACCUM_CONTIG32_SRC(26);
      KT_K2_ACCUM_CONTIG32_SRC(27);
      KT_K2_ACCUM_CONTIG32_SRC(28);
      KT_K2_ACCUM_CONTIG32_SRC(29);
      KT_K2_ACCUM_CONTIG32_SRC(30);
      KT_K2_ACCUM_CONTIG32_SRC(31);
#undef KT_K2_ACCUM_CONTIG32_SRC

      _mm512_storeu_ps(group_dst, f0);
      _mm512_storeu_ps(group_dst + 16, f1);
    }
  }

  void add_scaled_eight_gate_up_rows_f32(const uint8_t* gate_packed, const float* gate_scales, const uint8_t* up_packed,
                                         const float* up_scales, int row, const float* gate_coeffs,
                                         const float* up_coeffs, int cols, float* dst) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size == 32 && dense_coeff_fastpath_enabled()) {
      const int groups_per_row = cols / group_size;
      const size_t packed_row_stride = static_cast<size_t>(cols / 2);
      const uint8_t* gate_row_packed = gate_packed + static_cast<size_t>(row) * packed_row_stride;
      const uint8_t* up_row_packed = up_packed + static_cast<size_t>(row) * packed_row_stride;
      const float* gate_row_scales = gate_scales + static_cast<size_t>(row) * groups_per_row;
      const float* up_row_scales = up_scales + static_cast<size_t>(row) * groups_per_row;
      const __m512i zero_point = _mm512_set1_epi32(8);
      const __m128i nibble_mask = _mm_set1_epi8(0x0f);

      for (int group = 0; group < groups_per_row; group++) {
        float* group_dst = dst + static_cast<size_t>(group) * group_size;
        __m512 f0 = _mm512_loadu_ps(group_dst);
        __m512 f1 = _mm512_loadu_ps(group_dst + 16);

#define KT_K2_ACCUM_GATE_UP_SRC(SRC)                                                                                 \
  fmadd_int4_group32(                                                                                                \
      gate_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,              \
      _mm512_set1_ps(gate_coeffs[SRC] * gate_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]),         \
      zero_point, nibble_mask, f0, f1);                                                                              \
  fmadd_int4_group32(                                                                                                \
      up_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,                \
      _mm512_set1_ps(up_coeffs[SRC] * up_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]), zero_point, \
      nibble_mask, f0, f1)
        KT_K2_ACCUM_GATE_UP_SRC(0);
        KT_K2_ACCUM_GATE_UP_SRC(1);
        KT_K2_ACCUM_GATE_UP_SRC(2);
        KT_K2_ACCUM_GATE_UP_SRC(3);
        KT_K2_ACCUM_GATE_UP_SRC(4);
        KT_K2_ACCUM_GATE_UP_SRC(5);
        KT_K2_ACCUM_GATE_UP_SRC(6);
        KT_K2_ACCUM_GATE_UP_SRC(7);
#undef KT_K2_ACCUM_GATE_UP_SRC

        _mm512_storeu_ps(group_dst, f0);
        _mm512_storeu_ps(group_dst + 16, f1);
      }
      return;
    }

    const uint8_t* packed_list[16] = {gate_packed, up_packed, gate_packed, up_packed, gate_packed, up_packed,
                                      gate_packed, up_packed, gate_packed, up_packed, gate_packed, up_packed,
                                      gate_packed, up_packed, gate_packed, up_packed};
    const float* scales_list[16] = {gate_scales, up_scales, gate_scales, up_scales, gate_scales, up_scales,
                                    gate_scales, up_scales, gate_scales, up_scales, gate_scales, up_scales,
                                    gate_scales, up_scales, gate_scales, up_scales};
    const int rows[16] = {row,     row,     row + 1, row + 1, row + 2, row + 2, row + 3, row + 3,
                          row + 4, row + 4, row + 5, row + 5, row + 6, row + 6, row + 7, row + 7};
    const float coeffs[16] = {gate_coeffs[0], up_coeffs[0], gate_coeffs[1], up_coeffs[1], gate_coeffs[2], up_coeffs[2],
                              gate_coeffs[3], up_coeffs[3], gate_coeffs[4], up_coeffs[4], gate_coeffs[5], up_coeffs[5],
                              gate_coeffs[6], up_coeffs[6], gate_coeffs[7], up_coeffs[7]};
    add_scaled_sixteen_packed_rows_f32(packed_list, scales_list, rows, coeffs, cols, dst);
  }

  void add_scaled_sixteen_gate_up_rows_f32(const uint8_t* gate_packed, const float* gate_scales,
                                           const uint8_t* up_packed, const float* up_scales, int row,
                                           const float* gate_coeffs, const float* up_coeffs, int cols,
                                           float* dst) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size != 32 || !dense_coeff_fastpath_enabled()) {
      add_scaled_eight_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, row, gate_coeffs, up_coeffs,
                                        cols, dst);
      add_scaled_eight_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, row + 8, gate_coeffs + 8,
                                        up_coeffs + 8, cols, dst);
      return;
    }

    const int groups_per_row = cols / group_size;
    const size_t packed_row_stride = static_cast<size_t>(cols / 2);
    const uint8_t* gate_row_packed = gate_packed + static_cast<size_t>(row) * packed_row_stride;
    const uint8_t* up_row_packed = up_packed + static_cast<size_t>(row) * packed_row_stride;
    const float* gate_row_scales = gate_scales + static_cast<size_t>(row) * groups_per_row;
    const float* up_row_scales = up_scales + static_cast<size_t>(row) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      __m512 f0 = _mm512_loadu_ps(group_dst);
      __m512 f1 = _mm512_loadu_ps(group_dst + 16);

#define KT_K2_ACCUM_GATE_UP16_SRC(SRC)                                                                               \
  fmadd_int4_group32(                                                                                                \
      gate_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,              \
      _mm512_set1_ps(gate_coeffs[SRC] * gate_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]),         \
      zero_point, nibble_mask, f0, f1);                                                                              \
  fmadd_int4_group32(                                                                                                \
      up_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,                \
      _mm512_set1_ps(up_coeffs[SRC] * up_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]), zero_point, \
      nibble_mask, f0, f1)
      KT_K2_ACCUM_GATE_UP16_SRC(0);
      KT_K2_ACCUM_GATE_UP16_SRC(1);
      KT_K2_ACCUM_GATE_UP16_SRC(2);
      KT_K2_ACCUM_GATE_UP16_SRC(3);
      KT_K2_ACCUM_GATE_UP16_SRC(4);
      KT_K2_ACCUM_GATE_UP16_SRC(5);
      KT_K2_ACCUM_GATE_UP16_SRC(6);
      KT_K2_ACCUM_GATE_UP16_SRC(7);
      KT_K2_ACCUM_GATE_UP16_SRC(8);
      KT_K2_ACCUM_GATE_UP16_SRC(9);
      KT_K2_ACCUM_GATE_UP16_SRC(10);
      KT_K2_ACCUM_GATE_UP16_SRC(11);
      KT_K2_ACCUM_GATE_UP16_SRC(12);
      KT_K2_ACCUM_GATE_UP16_SRC(13);
      KT_K2_ACCUM_GATE_UP16_SRC(14);
      KT_K2_ACCUM_GATE_UP16_SRC(15);
#undef KT_K2_ACCUM_GATE_UP16_SRC

      _mm512_storeu_ps(group_dst, f0);
      _mm512_storeu_ps(group_dst + 16, f1);
    }
  }

  void add_scaled_thirtytwo_gate_up_rows_f32(const uint8_t* gate_packed, const float* gate_scales,
                                             const uint8_t* up_packed, const float* up_scales, int row,
                                             const float* gate_coeffs, const float* up_coeffs, int cols,
                                             float* dst) const {
    const int group_size = config_.quant_config.group_size;
    if (group_size != 32 || !dense_coeff_fastpath_enabled() || !gate_up32_fastpath_enabled()) {
      add_scaled_sixteen_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, row, gate_coeffs, up_coeffs,
                                          cols, dst);
      add_scaled_sixteen_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, row + 16, gate_coeffs + 16,
                                          up_coeffs + 16, cols, dst);
      return;
    }

    const int groups_per_row = cols / group_size;
    const size_t packed_row_stride = static_cast<size_t>(cols / 2);
    const uint8_t* gate_row_packed = gate_packed + static_cast<size_t>(row) * packed_row_stride;
    const uint8_t* up_row_packed = up_packed + static_cast<size_t>(row) * packed_row_stride;
    const float* gate_row_scales = gate_scales + static_cast<size_t>(row) * groups_per_row;
    const float* up_row_scales = up_scales + static_cast<size_t>(row) * groups_per_row;
    const __m512i zero_point = _mm512_set1_epi32(8);
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);

    for (int group = 0; group < groups_per_row; group++) {
      float* group_dst = dst + static_cast<size_t>(group) * group_size;
      __m512 f0 = _mm512_loadu_ps(group_dst);
      __m512 f1 = _mm512_loadu_ps(group_dst + 16);

#define KT_K2_ACCUM_GATE_UP32_SRC(SRC)                                                                               \
  fmadd_int4_group32(                                                                                                \
      gate_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,              \
      _mm512_set1_ps(gate_coeffs[SRC] * gate_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]),         \
      zero_point, nibble_mask, f0, f1);                                                                              \
  fmadd_int4_group32(                                                                                                \
      up_row_packed + static_cast<size_t>(SRC) * packed_row_stride + static_cast<size_t>(group) * 16,                \
      _mm512_set1_ps(up_coeffs[SRC] * up_row_scales[static_cast<size_t>(SRC) * groups_per_row + group]), zero_point, \
      nibble_mask, f0, f1)
      KT_K2_ACCUM_GATE_UP32_SRC(0);
      KT_K2_ACCUM_GATE_UP32_SRC(1);
      KT_K2_ACCUM_GATE_UP32_SRC(2);
      KT_K2_ACCUM_GATE_UP32_SRC(3);
      KT_K2_ACCUM_GATE_UP32_SRC(4);
      KT_K2_ACCUM_GATE_UP32_SRC(5);
      KT_K2_ACCUM_GATE_UP32_SRC(6);
      KT_K2_ACCUM_GATE_UP32_SRC(7);
      KT_K2_ACCUM_GATE_UP32_SRC(8);
      KT_K2_ACCUM_GATE_UP32_SRC(9);
      KT_K2_ACCUM_GATE_UP32_SRC(10);
      KT_K2_ACCUM_GATE_UP32_SRC(11);
      KT_K2_ACCUM_GATE_UP32_SRC(12);
      KT_K2_ACCUM_GATE_UP32_SRC(13);
      KT_K2_ACCUM_GATE_UP32_SRC(14);
      KT_K2_ACCUM_GATE_UP32_SRC(15);
      KT_K2_ACCUM_GATE_UP32_SRC(16);
      KT_K2_ACCUM_GATE_UP32_SRC(17);
      KT_K2_ACCUM_GATE_UP32_SRC(18);
      KT_K2_ACCUM_GATE_UP32_SRC(19);
      KT_K2_ACCUM_GATE_UP32_SRC(20);
      KT_K2_ACCUM_GATE_UP32_SRC(21);
      KT_K2_ACCUM_GATE_UP32_SRC(22);
      KT_K2_ACCUM_GATE_UP32_SRC(23);
      KT_K2_ACCUM_GATE_UP32_SRC(24);
      KT_K2_ACCUM_GATE_UP32_SRC(25);
      KT_K2_ACCUM_GATE_UP32_SRC(26);
      KT_K2_ACCUM_GATE_UP32_SRC(27);
      KT_K2_ACCUM_GATE_UP32_SRC(28);
      KT_K2_ACCUM_GATE_UP32_SRC(29);
      KT_K2_ACCUM_GATE_UP32_SRC(30);
      KT_K2_ACCUM_GATE_UP32_SRC(31);
#undef KT_K2_ACCUM_GATE_UP32_SRC

      _mm512_storeu_ps(group_dst, f0);
      _mm512_storeu_ps(group_dst + 16, f1);
    }
  }

  void add_scaled_packed_rows_f32(const uint8_t* packed, const float* scales, const float* coeffs, int rows, int cols,
                                  float* dst, bool use_four_row_fast_path) const {
    auto add_pair_or_single = [&](int row) {
      const float g0 = coeffs[row];
      const float g1 = coeffs[row + 1];
      if (g0 != 0.0f && g1 != 0.0f) {
        add_scaled_two_packed_rows_f32(packed, scales, row, g0, packed, scales, row + 1, g1, cols, dst);
      } else if (g0 != 0.0f) {
        add_scaled_packed_row_f32(packed, scales, row, cols, g0, dst);
      } else if (g1 != 0.0f) {
        add_scaled_packed_row_f32(packed, scales, row + 1, cols, g1, dst);
      }
    };

    int row = 0;
    if (use_four_row_fast_path) {
      for (; row + 31 < rows; row += 32) {
        add_scaled_thirtytwo_contiguous_packed_rows_f32(packed, scales, row, coeffs + row, cols, dst);
      }
      for (; row + 15 < rows; row += 16) {
        add_scaled_sixteen_contiguous_packed_rows_f32(packed, scales, row, coeffs + row, cols, dst);
      }
      const uint8_t* packed_list[8] = {packed, packed, packed, packed, packed, packed, packed, packed};
      const float* scales_list[8] = {scales, scales, scales, scales, scales, scales, scales, scales};
      for (; row + 7 < rows; row += 8) {
        const int row_ids[8] = {row, row + 1, row + 2, row + 3, row + 4, row + 5, row + 6, row + 7};
        add_scaled_eight_packed_rows_f32(packed_list, scales_list, row_ids, coeffs + row, cols, dst);
      }
      for (; row + 3 < rows; row += 4) {
        const float g0 = coeffs[row];
        const float g1 = coeffs[row + 1];
        const float g2 = coeffs[row + 2];
        const float g3 = coeffs[row + 3];
        add_scaled_four_packed_rows_f32(packed, scales, row, g0, packed, scales, row + 1, g1, packed, scales, row + 2,
                                        g2, packed, scales, row + 3, g3, cols, dst);
      }
    }

    for (; row + 1 < rows; row += 2) {
      add_pair_or_single(row);
    }
    if (row < rows) {
      const float g = coeffs[row];
      if (g != 0.0f) add_scaled_packed_row_f32(packed, scales, row, cols, g, dst);
    }
  }

  static bool tp1_backward_profile_enabled() {
    static const bool enabled = []() {
      const char* value = std::getenv("KT_K2_SFT_PROFILE_PACKED_BWD");
      if (value == nullptr || value[0] == '\0') {
        value = std::getenv("KT_K2_SFT_PROFILE_TP1_BWD");
      }
      return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
  }

  struct TP1BackwardProfile {
    using Clock = std::chrono::high_resolution_clock;
    bool enabled = false;
    Clock::time_point start;
    Clock::time_point last;
    long long grad_weights_us = 0;
    long long workspace_setup_us = 0;
    long long down_us = 0;
    long long down_lora_grads_us = 0;
    long long down_route_us = 0;
    long long down_write_us = 0;
    long long down_base_us = 0;
    long long down_lora_bprop_us = 0;
    long long down_lora_a_us = 0;
    long long down_lora_b_us = 0;
    long long down_lora_matmat_du_dx_us = 0;
    long long down_lora_matmat_da_db_us = 0;
    long long activation_us = 0;
    long long gate_up_us = 0;
    long long gate_up_base_us = 0;
    long long gate_up_lora_u_us = 0;
    long long gate_up_lora_b_us = 0;
    long long gate_up_lora_b_write_us = 0;
    long long gate_up_lora_a_input_us = 0;
    long long gate_up_lora_matmat_du_dx_us = 0;
    long long gate_up_lora_matmat_da_db_us = 0;
    long long gate_up_write_us = 0;
    long long gate_up_direct_us = 0;
    size_t workspace_bytes = 0;
    const char* down_base_kernel = "fp32_m8";
    const char* gate_up_base_kernel = "fp32_m8";

    explicit TP1BackwardProfile(bool enabled_) : enabled(enabled_) {
      if (enabled) {
        start = Clock::now();
        last = start;
      }
    }

    void mark(long long& slot) {
      if (!enabled) return;
      auto now = Clock::now();
      slot = std::chrono::duration_cast<std::chrono::microseconds>(now - last).count();
      last = now;
    }

    void add_since(Clock::time_point section_start, long long& slot) {
      if (!enabled) return;
      auto now = Clock::now();
      slot += std::chrono::duration_cast<std::chrono::microseconds>(now - section_start).count();
    }

    static Clock::time_point disabled_time_point() { return Clock::time_point{}; }

    Clock::time_point section_start() const { return enabled ? Clock::now() : disabled_time_point(); }

    long long total_us() const {
      if (!enabled) return 0;
      return std::chrono::duration_cast<std::chrono::microseconds>(last - start).count();
    }
  };

  struct BackwardWorkspaceV2 {
    float* grad_inter = nullptr;
    ggml_bf16_t* grad_down = nullptr;
    float* down_du = nullptr;
    float* grad_gate = nullptr;
    float* grad_up = nullptr;
    float* grad_input = nullptr;
    float* gate_du = nullptr;
    float* up_du = nullptr;
    size_t required_bytes = 0;
  };

  size_t backward_workspace_v2_bytes(int qlen, int k) const {
    if (qlen <= 0 || k <= 0) return 0;
    const size_t route_capacity = static_cast<size_t>(qlen) * k;
    const size_t inter_bytes = align64(route_capacity * static_cast<size_t>(config_.intermediate_size) * sizeof(float));
    const size_t down_bytes = align64(route_capacity * static_cast<size_t>(config_.hidden_size) * sizeof(ggml_bf16_t));
    const size_t input_bytes = align64(static_cast<size_t>(qlen) * config_.hidden_size * sizeof(float));
    const size_t du_bytes = align64(route_capacity * 8 * sizeof(float));
    return std::max(inter_bytes + down_bytes + 2 * du_bytes, 2 * inter_bytes + input_bytes + 2 * du_bytes);
  }

  BackwardWorkspaceV2 make_backward_workspace_v2(void* workspace, size_t workspace_bytes, int qlen, int k) const {
    BackwardWorkspaceV2 view;
    view.required_bytes = backward_workspace_v2_bytes(qlen, k);
    if (workspace == nullptr || workspace_bytes < view.required_bytes) {
      throw std::runtime_error("K2 SFT backward workspace is null or too small");
    }
    auto* base = static_cast<uint8_t*>(workspace);
    const size_t route_capacity = static_cast<size_t>(qlen) * k;
    const size_t inter_bytes = align64(route_capacity * static_cast<size_t>(config_.intermediate_size) * sizeof(float));
    const size_t down_bytes = align64(route_capacity * static_cast<size_t>(config_.hidden_size) * sizeof(ggml_bf16_t));
    const size_t input_bytes = align64(static_cast<size_t>(qlen) * config_.hidden_size * sizeof(float));
    const size_t du_bytes = align64(route_capacity * 8 * sizeof(float));

    // Down phase: [grad_inter][BF16 routed dY][dU scratch].
    view.grad_inter = reinterpret_cast<float*>(base);
    view.grad_down = reinterpret_cast<ggml_bf16_t*>(base + inter_bytes);
    view.down_du = reinterpret_cast<float*>(base + inter_bytes + down_bytes);

    // Gate phase aliases the same storage after activation consumes grad_inter.
    view.grad_gate = reinterpret_cast<float*>(base);
    view.grad_up = reinterpret_cast<float*>(base + inter_bytes);
    view.grad_input = reinterpret_cast<float*>(base + 2 * inter_bytes);
    view.gate_du = reinterpret_cast<float*>(base + 2 * inter_bytes + input_bytes);
    view.up_du = reinterpret_cast<float*>(base + 2 * inter_bytes + input_bytes + du_bytes);
    return view;
  }

  bool backward_workspace_v2_eligible() const {
    return backward_workspace_v2_env_enabled() && lora_rank_ == 8 && config_.pool != nullptr &&
           down_base_backward_bf16_matmat_enabled() && gate_up_base_backward_bf16_matmat_enabled();
  }

  size_t gate_up_packed_bytes_per_expert() const {
    return static_cast<size_t>(config_.intermediate_size) * config_.hidden_size / 2;
  }

  size_t down_packed_bytes_per_expert() const {
    return static_cast<size_t>(config_.hidden_size) * config_.intermediate_size / 2;
  }

  size_t gate_up_scale_elems_per_expert() const {
    return static_cast<size_t>(config_.intermediate_size) * (config_.hidden_size / config_.quant_config.group_size);
  }

  size_t down_scale_elems_per_expert() const {
    return static_cast<size_t>(config_.hidden_size) * (config_.intermediate_size / config_.quant_config.group_size);
  }

  bool packed_weight_buffers_ready() const {
    if (!k2_packed_weights_loaded_) return false;
    if (static_cast<int>(this->gate_bb_.size()) < config_.expert_num ||
        static_cast<int>(this->up_bb_.size()) < config_.expert_num ||
        static_cast<int>(this->down_bb_.size()) < config_.expert_num) {
      return false;
    }
    for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
      if (!this->gate_bb_[expert_idx] || !this->up_bb_[expert_idx] || !this->down_bb_[expert_idx]) return false;
      if (this->gate_bb_[expert_idx]->b == nullptr || this->gate_bb_[expert_idx]->d == nullptr ||
          this->up_bb_[expert_idx]->b == nullptr || this->up_bb_[expert_idx]->d == nullptr ||
          this->down_bb_[expert_idx]->b == nullptr || this->down_bb_[expert_idx]->d == nullptr) {
        return false;
      }
    }
    return true;
  }

  void ensure_packed_weight_buffers_ready() const {
    validate_k2_kgroup_contract();
    if (!packed_weight_buffers_ready()) {
      throw std::runtime_error("K2 RAWINT4 SFT packed weight buffers are not loaded");
    }
  }

  // Build the compact expert-packed token layout used by backward helpers.
  // Rows are ordered by active expert, then local token position within that expert.
  TP1BackwardLayout make_tp1_backward_layout(const K2ForwardCache& cache) const {
    TP1BackwardLayout layout;
    layout.expert_base.assign(static_cast<size_t>(config_.expert_num), 0);
    layout.expert_task_index.assign(static_cast<size_t>(config_.expert_num), -1);

    for (int task = 0; task < cache.activated_expert_cache; task++) {
      const int expert_idx = cache.m_expert_id_map_cache[task];
      layout.expert_base[expert_idx] = layout.total_tokens;
      layout.expert_task_index[expert_idx] = task;
      layout.total_tokens += static_cast<size_t>(cache.m_local_num_cache[expert_idx]);
    }
    layout.row_to_token.assign(layout.total_tokens, -1);
    layout.row_to_weight.assign(layout.total_tokens, 0.0f);
    for (int token_idx = 0; token_idx < cache.qlen_cache; token_idx++) {
      for (int route_idx = 0; route_idx < cache.k_cache; route_idx++) {
        const int64_t expert_id =
            cache.expert_ids_cache[static_cast<size_t>(token_idx) * cache.k_cache + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;
        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = layout.expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
        layout.row_to_token[row] = token_idx;
        layout.row_to_weight[row] = cache.weights_cache[static_cast<size_t>(token_idx) * cache.k_cache + route_idx];
      }
    }
    return layout;
  }

  static void write_bf16_array(void* dst, const float* src, size_t count) {
    if (dst == nullptr) return;
    auto* out = reinterpret_cast<ggml_bf16_t*>(dst);
#if defined(__AVX512BF16__)
    size_t i = 0;
    if (bf16_write_vec_enabled()) {
      for (; i + 15 < count; i += 16) {
        const __m512 v = _mm512_loadu_ps(src + i);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), (__m256i)_mm512_cvtneps_pbh(v));
      }
    }
    for (; i < count; i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
#else
    for (size_t i = 0; i < count; i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
#endif
  }

  static void write_bf16_vector(void* dst, const std::vector<float>& src) {
    write_bf16_array(dst, src.data(), src.size());
  }

  static void write_optimizer_bf16_array(void* dst, const float* src, size_t count, bool accumulate,
                                         float optimizer_grad_scale) {
    if (dst == nullptr) return;
    avx::store_optimizer_gradient_bf16(reinterpret_cast<ggml_bf16_t*>(dst), src, count, accumulate,
                                       optimizer_grad_scale);
  }

  static void scale_bf16_row_to_bf16(const ggml_bf16_t* src, ggml_bf16_t* dst, int count, float scale) {
    int i = 0;
#if defined(__AVX512BF16__)
    const __m512 scale_v = _mm512_set1_ps(scale);
    for (; i + 31 < count; i += 32) {
      __m512 lo, hi;
      avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(const_cast<ggml_bf16_t*>(src + i)), &lo, &hi);
      lo = _mm512_mul_ps(lo, scale_v);
      hi = _mm512_mul_ps(hi, scale_v);
      avx512_32xfp32_to_32xbf16(&lo, &hi, reinterpret_cast<__m512i*>(dst + i));
    }
#endif
    for (; i < count; i++) dst[i] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(src[i]) * scale);
  }

  static std::vector<ggml_bf16_t> bf16_vector_from_fp32(const std::vector<float>& src) {
    std::vector<ggml_bf16_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
    return out;
  }

  // Large-M gate/up base backward.  Phase one writes disjoint
  // expert-packed route rows, so expert work stealing has no output races.
  // Phase two assigns one task per token and reduces its top-k routes into the
  // final dX row.  The two pool jobs are sequential, never nested.
  bool compute_gate_up_base_expert_matmat(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                          const float* grad_gate, const float* grad_up,
                                          std::vector<float>& grad_input_fp32,
                                          TP1BackwardProfile* profile = nullptr) const {
    const int qlen = cache.qlen_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    if (qlen < 10 || config_.pool == nullptr || config_.quant_config.group_size != 32 || (hidden % 32) != 0) {
      return false;
    }

    gate_up_route_grad_scratch_.resize(layout.total_tokens * static_cast<size_t>(hidden));
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const bool use_bf16_matmat = gate_up_base_backward_bf16_matmat_enabled();
    if (profile != nullptr && use_bf16_matmat) profile->gate_up_base_kernel = "bf16_matmat";

#if defined(__AVX512BF16__)
    if (use_bf16_matmat) {
      const int nth = T::recommended_nth(hidden);
      pool->do_work_stealing_job(
          nth * cache.activated_expert_cache, nullptr,
          [&](int task_id) {
            const int expert_task = task_id / nth;
            const int ith = task_id % nth;
            const int expert_idx = cache.m_expert_id_map_cache[expert_task];
            const int num_tokens = cache.m_local_num_cache[expert_idx];
            const size_t row_base = layout.expert_base[expert_idx];
            const auto* gate_packed = reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b);
            const auto* up_packed = reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b);
            const float* gate_scales = this->gate_bb_[expert_idx]->d;
            const float* up_scales = this->up_bb_[expert_idx]->d;
            const auto [col_begin, col_end] = T::split_range_n(hidden, ith, nth);
            rawint4_gate_up_backward_matmat_bf16(
                gate_packed, gate_scales, up_packed, up_scales, grad_gate + row_base * inter_size,
                grad_up + row_base * inter_size, num_tokens, inter_size, hidden, col_begin, col_end,
                gate_up_route_grad_scratch_.data() + row_base * hidden);
          },
          nullptr);
    } else
#endif
    {
      pool->do_work_stealing_job(
          cache.activated_expert_cache, nullptr,
          [&](int expert_task) {
          const int expert_idx = cache.m_expert_id_map_cache[expert_task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const auto* gate_packed = reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b);
          const auto* up_packed = reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b);
          const float* gate_scales = this->gate_bb_[expert_idx]->d;
          const float* up_scales = this->up_bb_[expert_idx]->d;

          int local_t = 0;
          for (; local_t + 7 < num_tokens; local_t += 8) {
            const size_t row = row_base + static_cast<size_t>(local_t);
            rawint4_gate_up_backward_matmat_m8_bf16(
                gate_packed, gate_scales, up_packed, up_scales, grad_gate + row * inter_size,
                grad_up + row * inter_size, inter_size, hidden,
                gate_up_route_grad_scratch_.data() + row * hidden);
          }

          const int tail_tokens = num_tokens - local_t;
          if (tail_tokens <= 0) return;
          std::vector<float> tail_scratch(static_cast<size_t>(tail_tokens) * hidden, 0.0f);
          for (int tail_t = 0; tail_t < tail_tokens; tail_t++) {
            const size_t row = row_base + static_cast<size_t>(local_t + tail_t);
            const float* gate_row = grad_gate + row * inter_size;
            const float* up_row = grad_up + row * inter_size;
            float* dst = tail_scratch.data() + static_cast<size_t>(tail_t) * hidden;
            int i = 0;
            for (; i + 15 < inter_size; i += 16) {
              add_scaled_sixteen_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, gate_row + i,
                                                  up_row + i, hidden, dst);
            }
            for (; i + 7 < inter_size; i += 8) {
              add_scaled_eight_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, gate_row + i,
                                                up_row + i, hidden, dst);
            }
            for (; i + 3 < inter_size; i += 4) {
              add_scaled_four_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, gate_row + i,
                                               up_row + i, hidden, dst);
            }
            for (; i + 1 < inter_size; i += 2) {
              add_scaled_four_packed_rows_f32(
                  gate_packed, gate_scales, i, gate_row[i], up_packed, up_scales, i, up_row[i], gate_packed,
                  gate_scales, i + 1, gate_row[i + 1], up_packed, up_scales, i + 1, up_row[i + 1], hidden, dst);
            }
            if (i < inter_size) {
              add_scaled_two_packed_rows_f32(gate_packed, gate_scales, i, gate_row[i], up_packed, up_scales, i,
                                             up_row[i], hidden, dst);
            }
            write_bf16_array(gate_up_route_grad_scratch_.data() + row * hidden, dst, hidden);
          }
          },
          nullptr);
    }

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int token_idx) {
          float* dst = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
          for (int route_idx = 0; route_idx < cache.k_cache; route_idx++) {
            const int64_t expert_id =
                cache.expert_ids_cache[static_cast<size_t>(token_idx) * cache.k_cache + route_idx];
            if (config_.should_skip_expert(expert_id)) continue;
            const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
            const size_t row =
                layout.expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
            if (layout.row_to_token[row] != token_idx) continue;
            const ggml_bf16_t* src = gate_up_route_grad_scratch_.data() + row * hidden;
            int h = 0;
            for (; h + 15 < hidden; h += 16) {
              const __m512 route =
                  bf16x16_to_fp32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + h)));
              _mm512_storeu_ps(dst + h, _mm512_add_ps(_mm512_loadu_ps(dst + h), route));
            }
            for (; h < hidden; h++) dst[h] += GGML_BF16_TO_FP32(src[h]);
          }
        },
        nullptr);
    return true;
  }

  // Shared Backward Step 2.1: router/topk-weight gradient for forward Step 9 weighted merge:
  //   output[token] += weight[token, route] * down_output[token, route].
  // Therefore dL/dweight is dot(grad_output[token], cached_down_output[token, route]).
  // Python passes a null grad_weights pointer when topk weights do not require gradients.
  void compute_tp1_grad_weights(const K2ForwardCache& cache, const TP1BackwardLayout& layout, const void* grad_output,
                                void* grad_weights) const {
    if (grad_weights == nullptr) return;
    if (grad_output == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 grad_weights requires grad_output");
    }

    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    auto* out_grad_weights = reinterpret_cast<float*>(grad_weights);
    auto* grad_out = reinterpret_cast<const ggml_bf16_t*>(grad_output);
    std::fill(out_grad_weights, out_grad_weights + static_cast<size_t>(qlen) * k, 0.0f);

    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      const ggml_bf16_t* token_grad = grad_out + static_cast<size_t>(token_idx) * hidden;
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = layout.expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
        const ggml_bf16_t* down_row = cache.down_output_cache + row * hidden;

        float acc = 0.0f;
        for (int h = 0; h < hidden; h++) {
          acc += GGML_BF16_TO_FP32(token_grad[h]) * GGML_BF16_TO_FP32(down_row[h]);
        }
        out_grad_weights[static_cast<size_t>(token_idx) * k + route_idx] = acc;
      }
    }
  }

  // Shared Backward Steps 2 -> 3 for expert-packed rows:
  //   Shared Backward Step 2.2: scatter token grad_output to grad_down for forward Step 9.
  //   Shared Backward Step 3.1: down base backward for forward Step 8.
  //   Shared Backward Step 3.2: down LoRA B backward produces rank-space grad_times_b.
  //   Shared Backward Step 3.3: down LoRA A contributes to grad_intermediate.
  void compute_tp1_down_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout, const void* grad_output,
                                 void* grad_down, std::vector<float>* grad_down_fp32_out, void* grad_intermediate,
                                 std::vector<float>* grad_inter_fp32_out, void* grad_down_lora_a,
                                 void* grad_down_lora_b, TP1BackwardProfile* profile = nullptr,
                                 std::vector<float>* down_lora_grad_times_b_out = nullptr) const {
    if (grad_output == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 down backward requires grad_output");
    }

    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int rank = lora_rank_;
    const bool need_grad_intermediate = grad_intermediate != nullptr || grad_inter_fp32_out != nullptr;
    const bool need_lora_grads = grad_down_lora_a != nullptr || grad_down_lora_b != nullptr;
    const bool need_down_lora_path = rank > 0 && has_down_lora() && (need_grad_intermediate || need_lora_grads);

    if (need_grad_intermediate) ensure_packed_weight_buffers_ready();
    if (need_lora_grads && rank > 0 && !need_down_lora_path) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 down backward requires down LoRA weights");
    }
    if (grad_down_lora_b != nullptr && rank > 0 && cache.down_lora_u_cache == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 down backward requires cached down LoRA activations");
    }

    auto* grad_out = reinterpret_cast<const ggml_bf16_t*>(grad_output);
    std::vector<float> grad_down_fp32(layout.total_tokens * hidden, 0.0f);

    // Shared Backward Step 2.2: Backprop through forward Step 9 weighted merge.
    // grad_down[token, route] += grad_output[token] * router_weight[token, route].
    auto section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      const ggml_bf16_t* token_grad = grad_out + static_cast<size_t>(token_idx) * hidden;
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = layout.expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
        const float route_weight = cache.weights_cache[static_cast<size_t>(token_idx) * k + route_idx];
        float* grad_down_row = grad_down_fp32.data() + row * hidden;
        for (int h = 0; h < hidden; h++) {
          grad_down_row[h] += GGML_BF16_TO_FP32(token_grad[h]) * route_weight;
        }
      }
    }
    if (profile != nullptr) profile->add_since(section_start, profile->down_route_us);

    section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
    write_bf16_vector(grad_down, grad_down_fp32);
    if (profile != nullptr) profile->add_since(section_start, profile->down_write_us);

    if (!need_grad_intermediate && !need_lora_grads) {
      if (grad_down_fp32_out != nullptr) *grad_down_fp32_out = std::move(grad_down_fp32);
      return;
    }

    std::vector<float> grad_inter_fp32;
    if (need_grad_intermediate) {
      grad_inter_fp32.assign(layout.total_tokens * inter_size, 0.0f);
    }

    std::vector<float> grad_down_a_fp32;
    std::vector<float> grad_down_b_fp32;
    if (need_lora_grads && rank > 0) {
      grad_down_a_fp32.assign(static_cast<size_t>(config_.expert_num) * rank * inter_size, 0.0f);
      grad_down_b_fp32.assign(static_cast<size_t>(config_.expert_num) * hidden * rank, 0.0f);
    }
    if (down_lora_grad_times_b_out != nullptr) {
      if (need_down_lora_path && rank > 0) {
        down_lora_grad_times_b_out->assign(layout.total_tokens * static_cast<size_t>(rank), 0.0f);
      } else {
        down_lora_grad_times_b_out->clear();
      }
    }

    if (need_grad_intermediate) {
      // Shared Backward Step 3.1: expert-packed down base backward.
      // Large batches split each expert over the same 256-column partitions
      // used by forward.  The BF16 path decodes two RAWINT4 reduction rows at
      // a time and reuses the weight pair across eight routed tokens.
      section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      auto compute_expert_base = [&](int expert_task) {
        const int expert_idx = cache.m_expert_id_map_cache[expert_task];
        const int num_tokens = cache.m_local_num_cache[expert_idx];
        const size_t row_base = layout.expert_base[expert_idx];
        const auto* down_packed = reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b);
        const float* down_scales = this->down_bb_[expert_idx]->d;

        int local_t = 0;
        for (; local_t + 7 < num_tokens; local_t += 8) {
          const size_t row = row_base + static_cast<size_t>(local_t);
          rawint4_backward_matmat_m8_f32(down_packed, down_scales, grad_down_fp32.data() + row * hidden, hidden,
                                         inter_size, grad_inter_fp32.data() + row * inter_size);
        }
        for (; local_t < num_tokens; local_t++) {
          const size_t row = row_base + static_cast<size_t>(local_t);
          const float* grad_down_row = grad_down_fp32.data() + row * hidden;
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          add_scaled_packed_rows_f32(down_packed, down_scales, grad_down_row, hidden, inter_size, grad_inter_row,
                                     layout.total_tokens >= 10 || short_base_fastpath_enabled());
        }
      };

      const bool use_bf16_matmat = layout.total_tokens >= 10 && config_.pool != nullptr &&
                                      down_base_backward_bf16_matmat_enabled();
      if (profile != nullptr && use_bf16_matmat) profile->down_base_kernel = "bf16_matmat";
#if defined(__AVX512BF16__)
      if (use_bf16_matmat) {
        auto pool = config_.pool->get_subpool(tp_part_idx);
        const int nth = T::recommended_nth(inter_size);
        pool->do_work_stealing_job(
            nth * cache.activated_expert_cache, nullptr,
            [&](int task_id) {
              const int expert_task = task_id / nth;
              const int ith = task_id % nth;
              const int expert_idx = cache.m_expert_id_map_cache[expert_task];
              const int num_tokens = cache.m_local_num_cache[expert_idx];
              const size_t row_base = layout.expert_base[expert_idx];
              const auto* down_packed = reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b);
              const float* down_scales = this->down_bb_[expert_idx]->d;
              const auto [col_begin, col_end] = T::split_range_n(inter_size, ith, nth);
              rawint4_backward_matmat_bf16(
                  down_packed, down_scales, grad_down_fp32.data() + row_base * hidden, num_tokens, hidden,
                  inter_size, col_begin, col_end, grad_inter_fp32.data() + row_base * inter_size);
            },
            nullptr);
      } else
#endif
      if (layout.total_tokens >= 10 && config_.pool != nullptr) {
        auto pool = config_.pool->get_subpool(tp_part_idx);
        pool->do_work_stealing_job(cache.activated_expert_cache, nullptr, compute_expert_base, nullptr);
      } else {
        for (int expert_task = 0; expert_task < cache.activated_expert_cache; expert_task++) {
          compute_expert_base(expert_task);
        }
      }
      if (profile != nullptr) profile->add_since(section_start, profile->down_base_us);
    }

    std::vector<uint8_t> down_lora_matmat_expert(static_cast<size_t>(config_.expert_num), 0);
    std::vector<float> down_lora_matmat_du;
    const bool try_down_lora_matmat = need_down_lora_path && rank == 8 && lora_backward_matmat_enabled() &&
                                         !down_lora_b_transposed_.empty();
    if (try_down_lora_matmat) {
      down_lora_matmat_du.assign(layout.total_tokens * 8, 0.0f);
      const auto matmat_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      auto compute_expert_lora = [&](int expert_task) {
        const int expert_idx = cache.m_expert_id_map_cache[expert_task];
        const int num_tokens = cache.m_local_num_cache[expert_idx];
        if (num_tokens < 4) return;
        const size_t row_base = layout.expert_base[expert_idx];
        float* du = down_lora_matmat_du.data() + row_base * 8;
        const ggml_bf16_t* down_b_t =
            down_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * hidden;
        avx::lora_backward_du_rank8_matmat(grad_down_fp32.data() + row_base * hidden, down_b_t, du, num_tokens,
                                          hidden);
        if (need_grad_intermediate) {
          const ggml_bf16_t* down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * 8 * inter_size;
          avx::lora_backward_dx_rank8_matmat(du, down_a, grad_inter_fp32.data() + row_base * inter_size, num_tokens,
                                            inter_size, lora_scaling_);
        }
        if (down_lora_grad_times_b_out != nullptr && !down_lora_grad_times_b_out->empty()) {
          std::copy(du, du + static_cast<size_t>(num_tokens) * 8,
                    down_lora_grad_times_b_out->data() + row_base * 8);
        }
        down_lora_matmat_expert[static_cast<size_t>(expert_idx)] = 1;
      };
      if (layout.total_tokens >= 10 && config_.pool != nullptr) {
        auto pool = config_.pool->get_subpool(tp_part_idx);
        pool->do_work_stealing_job(cache.activated_expert_cache, nullptr, compute_expert_lora, nullptr);
      } else {
        for (int task = 0; task < cache.activated_expert_cache; task++) compute_expert_lora(task);
      }
      if (profile != nullptr) {
        profile->add_since(matmat_start, profile->down_lora_matmat_du_dx_us);
      }
    }

    for (int expert_task = 0; expert_task < cache.activated_expert_cache; expert_task++) {
      const int expert_idx = cache.m_expert_id_map_cache[expert_task];
      const int num_tokens = cache.m_local_num_cache[expert_idx];
      const size_t row_base = layout.expert_base[expert_idx];
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
      const bool used_matmat = down_lora_matmat_expert[static_cast<size_t>(expert_idx)] != 0;

      for (int local_t = 0; local_t < num_tokens; local_t++) {
        const size_t row = row_base + static_cast<size_t>(local_t);
        const float* grad_down_row = grad_down_fp32.data() + row * hidden;

        if (!need_down_lora_path) continue;

        // Shared Backward Step 3.2: Backprop through down LoRA B.
        // grad_times_b is the rank-space gradient before applying down LoRA A.
        std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);
        const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;
        section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
        if (used_matmat) {
          const float* cached_du = down_lora_matmat_du.data() + row * 8;
          std::copy(cached_du, cached_du + 8, grad_times_b.begin());
        } else if (rank == 2) {
          float gb0;
          float gb1;
          down_bprop_rank2_vec(grad_down_row, expert_down_b, hidden, gb0, gb1);
          grad_times_b[0] = gb0;
          grad_times_b[1] = gb1;
        } else if (rank == 8) {
          down_bprop_rank8_vec(grad_down_row, expert_down_b, hidden, grad_times_b.data());
        } else {
          for (int h = 0; h < hidden; h++) {
            const float g = grad_down_row[h];
            if (g == 0.0f) continue;
            const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * rank;
            for (int r = 0; r < rank; r++) {
              grad_times_b[r] += g * GGML_BF16_TO_FP32(down_b_row[r]);
            }
          }
        }
        if (!used_matmat && profile != nullptr) profile->add_since(section_start, profile->down_lora_bprop_us);
        if (!used_matmat && down_lora_grad_times_b_out != nullptr && !down_lora_grad_times_b_out->empty()) {
          float* cached_grad_times_b = down_lora_grad_times_b_out->data() + row * static_cast<size_t>(rank);
          std::copy(grad_times_b.begin(), grad_times_b.end(), cached_grad_times_b);
        }

        const ggml_bf16_t* expert_down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * rank * inter_size;
        if (need_grad_intermediate && !used_matmat) {
          // Shared Backward Step 3.3: Add down LoRA contribution to grad_intermediate.
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          if (rank == 8) {
            alignas(32) float gu_scaled[8];
            for (int r = 0; r < 8; r++) gu_scaled[r] = grad_times_b[r] * lora_scaling_;
            accumulate_lora_a_input_rank8_vec(nullptr, expert_down_a, gu_scaled, inter_size, grad_inter_row,
                                              nullptr);
          } else {
            for (int r = 0; r < rank; r++) {
              const float gu = grad_times_b[r] * lora_scaling_;
              const ggml_bf16_t* down_a_row = expert_down_a + static_cast<size_t>(r) * inter_size;
              for (int i = 0; i < inter_size; i++) {
                grad_inter_row[i] += gu * GGML_BF16_TO_FP32(down_a_row[i]);
              }
            }
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_lora_a_us);
        }

        if (grad_down_lora_a != nullptr) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
          float* grad_a = grad_down_a_fp32.data() + static_cast<size_t>(expert_idx) * rank * inter_size;
          if (rank == 8) {
            alignas(32) float gu_scaled[8];
            for (int r = 0; r < 8; r++) gu_scaled[r] = grad_times_b[r] * lora_scaling_;
            accumulate_lora_a_input_rank8_vec(intermediate_row, expert_down_a, gu_scaled, inter_size, nullptr,
                                              grad_a);
          } else {
            for (int r = 0; r < rank; r++) {
              const float gu = grad_times_b[r] * lora_scaling_;
              float* grad_a_row = grad_a + static_cast<size_t>(r) * inter_size;
              for (int i = 0; i < inter_size; i++) {
                grad_a_row[i] += gu * GGML_BF16_TO_FP32(intermediate_row[i]);
              }
            }
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_lora_a_us);
        }

        if (grad_down_lora_b != nullptr) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          const float* down_u_row = cache.down_lora_u_cache + row * rank;
          float* grad_b = grad_down_b_fp32.data() + static_cast<size_t>(expert_idx) * hidden * rank;
          if (rank == 2) {
            accumulate_down_lora_b_rank2_vec(grad_down_row, down_u_row[0] * lora_scaling_,
                                             down_u_row[1] * lora_scaling_, hidden, grad_b);
          } else if (rank == 8) {
            alignas(32) float u_scaled[8];
            for (int r = 0; r < 8; r++) u_scaled[r] = down_u_row[r] * lora_scaling_;
            accumulate_down_lora_b_rank8_vec(grad_down_row, u_scaled, hidden, grad_b);
          } else {
            for (int h = 0; h < hidden; h++) {
              const float g = grad_down_row[h] * lora_scaling_;
              if (g == 0.0f) continue;
              float* grad_b_row = grad_b + static_cast<size_t>(h) * rank;
              for (int r = 0; r < rank; r++) {
                grad_b_row[r] += g * down_u_row[r];
              }
            }
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_lora_b_us);
        }
      }
    }

    write_bf16_vector(grad_intermediate, grad_inter_fp32);
    if (grad_down_lora_a != nullptr && rank > 0) write_bf16_vector(grad_down_lora_a, grad_down_a_fp32);
    if (grad_down_lora_b != nullptr && rank > 0) write_bf16_vector(grad_down_lora_b, grad_down_b_fp32);

    if (grad_down_fp32_out != nullptr) *grad_down_fp32_out = std::move(grad_down_fp32);
    if (grad_inter_fp32_out != nullptr) *grad_inter_fp32_out = std::move(grad_inter_fp32);
  }

  void compute_tp1_activation_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                       const ggml_bf16_t* grad_intermediate, void* grad_gate, void* grad_up) const {
    if (grad_gate == nullptr && grad_up == nullptr) return;
    if (grad_intermediate == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 activation backward requires grad_intermediate");
    }

    auto* out_grad_gate = reinterpret_cast<ggml_bf16_t*>(grad_gate);
    auto* out_grad_up = reinterpret_cast<ggml_bf16_t*>(grad_up);
    const size_t elems = layout.total_tokens * static_cast<size_t>(config_.intermediate_size);

    for (size_t idx = 0; idx < elems; idx++) {
      const float grad_inter = GGML_BF16_TO_FP32(grad_intermediate[idx]);
      const float gate = GGML_BF16_TO_FP32(cache.gate_output_cache[idx]);
      const float up = GGML_BF16_TO_FP32(cache.up_output_cache[idx]);
      const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
      const float silu = gate * sigmoid;

      if (out_grad_gate != nullptr) {
        const float silu_grad = sigmoid * (1.0f + gate * (1.0f - sigmoid));
        out_grad_gate[idx] = GGML_FP32_TO_BF16(grad_inter * up * silu_grad);
      }
      if (out_grad_up != nullptr) {
        out_grad_up[idx] = GGML_FP32_TO_BF16(grad_inter * silu);
      }
    }
  }

  void compute_tp1_activation_backward_fp32(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                            const float* grad_intermediate, std::vector<float>& grad_gate,
                                            std::vector<float>& grad_up) const {
    if (grad_intermediate == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 activation backward requires grad_intermediate");
    }

    const size_t elems = layout.total_tokens * static_cast<size_t>(config_.intermediate_size);
    grad_gate.assign(elems, 0.0f);
    grad_up.assign(elems, 0.0f);

    // Shared Backward Step 4.1: Backprop through forward Step 6 activation, keeping FP32 grads for packed base math.
    for (size_t idx = 0; idx < elems; idx++) {
      const float grad_inter = grad_intermediate[idx];
      const float gate = GGML_BF16_TO_FP32(cache.gate_output_cache[idx]);
      const float up = GGML_BF16_TO_FP32(cache.up_output_cache[idx]);
      const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
      const float silu = gate * sigmoid;
      const float silu_grad = sigmoid * (1.0f + gate * (1.0f - sigmoid));
      grad_gate[idx] = grad_inter * up * silu_grad;
      grad_up[idx] = grad_inter * silu;
    }
  }

  void compute_tp1_gate_up_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                    const float* grad_gate, const float* grad_up, void* grad_input,
                                    void* grad_gate_lora_a, void* grad_gate_lora_b, void* grad_up_lora_a,
                                    void* grad_up_lora_b, TP1BackwardProfile* profile = nullptr) const {
    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int rank = lora_rank_;
    const bool need_lora_grads = grad_gate_lora_a != nullptr || grad_gate_lora_b != nullptr ||
                                 grad_up_lora_a != nullptr || grad_up_lora_b != nullptr;

    if ((grad_input != nullptr || need_lora_grads) && (grad_gate == nullptr || grad_up == nullptr)) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 gate/up backward requires gate and up gradients");
    }
    if (grad_input != nullptr) ensure_packed_weight_buffers_ready();

    const bool use_gate_up_lora = rank > 0 && has_gate_up_lora();
    if (need_lora_grads && !use_gate_up_lora) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 gate/up backward requires gate/up LoRA weights");
    }

    std::vector<float> grad_input_fp32(static_cast<size_t>(qlen) * hidden, 0.0f);
    std::vector<float> grad_gate_a_fp32;
    std::vector<float> grad_gate_b_fp32;
    std::vector<float> grad_up_a_fp32;
    std::vector<float> grad_up_b_fp32;

    if (need_lora_grads && rank > 0) {
      grad_gate_a_fp32.assign(static_cast<size_t>(config_.expert_num) * rank * hidden, 0.0f);
      grad_up_a_fp32.assign(static_cast<size_t>(config_.expert_num) * rank * hidden, 0.0f);
      grad_gate_b_fp32.assign(static_cast<size_t>(config_.expert_num) * inter_size * rank, 0.0f);
      grad_up_b_fp32.assign(static_cast<size_t>(config_.expert_num) * inter_size * rank, 0.0f);
    }

    std::vector<float> lora_u(static_cast<size_t>(rank), 0.0f);
    std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);

    auto add_gate_up_base = [&](int expert_idx, int token_idx, size_t row) {
      if (grad_input == nullptr) return;
      float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
      const auto* gate_packed = reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b);
      const auto* up_packed = reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b);
      const float* gate_scales = this->gate_bb_[expert_idx]->d;
      const float* up_scales = this->up_bb_[expert_idx]->d;
      const float* grad_gate_row = grad_gate + row * inter_size;
      const float* grad_up_row = grad_up + row * inter_size;
      const bool use_four_row_fast_path = qlen >= 10 || short_base_fastpath_enabled();

      auto add_gate_up_base_row = [&](int i) {
        const float gate_g = grad_gate_row[i];
        const float up_g = grad_up_row[i];
        if (gate_g == 0.0f && up_g == 0.0f) return;
        if (gate_g != 0.0f && up_g != 0.0f) {
          add_scaled_two_packed_rows_f32(gate_packed, gate_scales, i, gate_g, up_packed, up_scales, i, up_g, hidden,
                                         grad_input_row);
        } else if (gate_g != 0.0f) {
          add_scaled_packed_row_f32(gate_packed, gate_scales, i, hidden, gate_g, grad_input_row);
        } else {
          add_scaled_packed_row_f32(up_packed, up_scales, i, hidden, up_g, grad_input_row);
        }
      };

      if (!use_four_row_fast_path) {
        for (int i = 0; i < inter_size; i++) {
          add_gate_up_base_row(i);
        }
        return;
      }

      int i = 0;
      for (; i + 31 < inter_size; i += 32) {
        add_scaled_thirtytwo_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                              grad_up_row + i, hidden, grad_input_row);
      }
      for (; i + 15 < inter_size; i += 16) {
        add_scaled_sixteen_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                            grad_up_row + i, hidden, grad_input_row);
      }
      for (; i + 7 < inter_size; i += 8) {
        add_scaled_eight_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                          grad_up_row + i, hidden, grad_input_row);
      }
      for (; i + 3 < inter_size; i += 4) {
        add_scaled_four_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                         grad_up_row + i, hidden, grad_input_row);
      }
      for (; i + 1 < inter_size; i += 2) {
        const float gate_g0 = grad_gate_row[i];
        const float up_g0 = grad_up_row[i];
        const float gate_g1 = grad_gate_row[i + 1];
        const float up_g1 = grad_up_row[i + 1];
        add_scaled_four_packed_rows_f32(gate_packed, gate_scales, i, gate_g0, up_packed, up_scales, i, up_g0,
                                        gate_packed, gate_scales, i + 1, gate_g1, up_packed, up_scales, i + 1, up_g1,
                                        hidden, grad_input_row);
      }
      if (i < inter_size) {
        add_gate_up_base_row(i);
      }
    };

    auto backward_one_projection = [&](int expert_idx, int token_idx, const float* grad_row,
                                       const uint8_t* packed_weight, const float* scales, const ggml_bf16_t* lora_a,
                                       const ggml_bf16_t* lora_b, float* grad_lora_a, float* grad_lora_b,
                                       const float* cached_lora_u, bool do_base, bool do_lora,
                                       bool profile_base_inside) {
      float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
      const ggml_bf16_t* input_row = cache.input_cache + static_cast<size_t>(token_idx) * hidden;

      if (do_base && grad_input != nullptr) {
        const auto section_start = profile_base_inside && profile != nullptr
                                       ? profile->section_start()
                                       : TP1BackwardProfile::disabled_time_point();
        for (int i = 0; i < inter_size; i++) {
          const float g = grad_row[i];
          if (g == 0.0f) continue;
          add_scaled_packed_row_f32(packed_weight, scales, i, hidden, g, grad_input_row);
        }
        if (profile_base_inside && profile != nullptr) profile->add_since(section_start, profile->gate_up_base_us);
      }

      if (!do_lora || !use_gate_up_lora) return;

      std::fill(lora_u.begin(), lora_u.end(), 0.0f);
      std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);

      auto section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      if (cached_lora_u != nullptr) {
        std::copy(cached_lora_u, cached_lora_u + rank, lora_u.begin());
      } else if (rank == 2) {
        const ggml_bf16_t* a0 = lora_a;
        const ggml_bf16_t* a1 = lora_a + hidden;
        float acc0;
        float acc1;
        bf16_dot2_lora_u(input_row, a0, a1, hidden, acc0, acc1);
        lora_u[0] = acc0;
        lora_u[1] = acc1;
      } else if (rank == 8) {
        bf16_dot8_lora_u(input_row, lora_a, hidden, lora_u.data());
      } else {
        for (int r = 0; r < rank; r++) {
          const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
          float acc = 0.0f;
          for (int h = 0; h < hidden; h++) {
            acc += GGML_BF16_TO_FP32(input_row[h]) * GGML_BF16_TO_FP32(a_row[h]);
          }
          lora_u[r] = acc;
        }
      }
      if (profile != nullptr) profile->add_since(section_start, profile->gate_up_lora_u_us);

      section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      if (rank == 2) {
        const float u0_scaled = lora_u[0] * lora_scaling_;
        const float u1_scaled = lora_u[1] * lora_scaling_;
        float gb0 = 0.0f;
        float gb1 = 0.0f;
        float* grad_b =
            grad_lora_b == nullptr ? nullptr : grad_lora_b + static_cast<size_t>(expert_idx) * inter_size * 2;
        accumulate_gate_up_lora_b_rank2_vec(grad_row, lora_b, u0_scaled, u1_scaled, inter_size, grad_b, gb0, gb1);
        grad_times_b[0] = gb0;
        grad_times_b[1] = gb1;
      } else if (rank == 8) {
        alignas(32) float u_scaled[8];
        for (int r = 0; r < 8; r++) u_scaled[r] = lora_u[r] * lora_scaling_;
        float* grad_b = grad_lora_b == nullptr ? nullptr
                                                : grad_lora_b + static_cast<size_t>(expert_idx) * inter_size * 8;
        accumulate_gate_up_lora_b_rank8_vec(grad_row, lora_b, u_scaled, inter_size, grad_b, grad_times_b.data());
      } else {
        for (int i = 0; i < inter_size; i++) {
          const float g = grad_row[i];
          if (g == 0.0f) continue;
          const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * rank;
          if (grad_lora_b != nullptr) {
            float* grad_b_row =
                grad_lora_b + static_cast<size_t>(expert_idx) * inter_size * rank + static_cast<size_t>(i) * rank;
            for (int r = 0; r < rank; r++) {
              grad_b_row[r] += g * lora_u[r] * lora_scaling_;
            }
          }
          for (int r = 0; r < rank; r++) {
            grad_times_b[r] += g * GGML_BF16_TO_FP32(b_row[r]);
          }
        }
      }
      if (profile != nullptr) profile->add_since(section_start, profile->gate_up_lora_b_us);

      section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      if (rank == 2) {
        float* grad_a0 = nullptr;
        float* grad_a1 = nullptr;
        if (grad_lora_a != nullptr) {
          grad_a0 = grad_lora_a + static_cast<size_t>(expert_idx) * 2 * hidden;
          grad_a1 = grad_a0 + hidden;
        }
        accumulate_gate_up_a_input_rank2_vec(input_row, lora_a, lora_a + hidden, grad_times_b[0] * lora_scaling_,
                                             grad_times_b[1] * lora_scaling_, hidden, grad_input_row, grad_a0, grad_a1);
      } else if (rank == 8) {
        alignas(32) float gu_scaled[8];
        for (int r = 0; r < 8; r++) gu_scaled[r] = grad_times_b[r] * lora_scaling_;
        float* grad_a = grad_lora_a == nullptr ? nullptr
                                               : grad_lora_a + static_cast<size_t>(expert_idx) * 8 * hidden;
        accumulate_lora_a_input_rank8_vec(input_row, lora_a, gu_scaled, hidden, grad_input_row, grad_a);
      } else {
        for (int r = 0; r < rank; r++) {
          const float gu = grad_times_b[r] * lora_scaling_;
          const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
          if (grad_lora_a != nullptr) {
            float* grad_a_row =
                grad_lora_a + static_cast<size_t>(expert_idx) * rank * hidden + static_cast<size_t>(r) * hidden;
            for (int h = 0; h < hidden; h++) {
              const float x = GGML_BF16_TO_FP32(input_row[h]);
              grad_a_row[h] += gu * x;
              grad_input_row[h] += gu * GGML_BF16_TO_FP32(a_row[h]);
            }
          } else {
            for (int h = 0; h < hidden; h++) {
              grad_input_row[h] += gu * GGML_BF16_TO_FP32(a_row[h]);
            }
          }
        }
      }
      if (profile != nullptr) profile->add_since(section_start, profile->gate_up_lora_a_input_us);
    };

    auto run_token_routes = [&](int token_idx, bool do_base, bool do_lora, bool profile_base_inside) {
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int expert_idx = static_cast<int>(expert_id);
        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = layout.expert_base[static_cast<size_t>(expert_idx)] + static_cast<size_t>(local_pos);

        if (do_base && grad_input != nullptr) {
          const auto section_start = profile_base_inside && profile != nullptr
                                         ? profile->section_start()
                                         : TP1BackwardProfile::disabled_time_point();
          add_gate_up_base(expert_idx, token_idx, row);
          if (profile_base_inside && profile != nullptr) profile->add_since(section_start, profile->gate_up_base_us);
        }
        if (!do_lora) continue;

        backward_one_projection(
            expert_idx, token_idx, grad_gate + row * inter_size,
            reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b), this->gate_bb_[expert_idx]->d,
            use_gate_up_lora ? gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? gate_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            grad_gate_lora_a != nullptr ? grad_gate_a_fp32.data() : nullptr,
            grad_gate_lora_b != nullptr ? grad_gate_b_fp32.data() : nullptr,
            cache.gate_lora_u_cache == nullptr ? nullptr : cache.gate_lora_u_cache + row * static_cast<size_t>(rank),
            false, true, false);
        backward_one_projection(
            expert_idx, token_idx, grad_up + row * inter_size,
            reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
            use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            grad_up_lora_a != nullptr ? grad_up_a_fp32.data() : nullptr,
            grad_up_lora_b != nullptr ? grad_up_b_fp32.data() : nullptr,
            cache.up_lora_u_cache == nullptr ? nullptr : cache.up_lora_u_cache + row * static_cast<size_t>(rank), false,
            true, false);
      }
    };

    bool expert_matmat_base = false;
    if (grad_input != nullptr) {
      const auto section_start =
          profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      expert_matmat_base =
          compute_gate_up_base_expert_matmat(cache, layout, grad_gate, grad_up, grad_input_fp32, profile);
      if (expert_matmat_base && profile != nullptr) profile->add_since(section_start, profile->gate_up_base_us);
    }
    if (expert_matmat_base) {
      if (use_gate_up_lora) {
        for (int token_idx = 0; token_idx < qlen; token_idx++) run_token_routes(token_idx, false, true, false);
      }
    } else {
      for (int token_idx = 0; token_idx < qlen; token_idx++) {
        run_token_routes(token_idx, true, true, true);
      }
    }

    const auto section_start =
        profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
    write_bf16_vector(grad_input, grad_input_fp32);
    if (profile != nullptr) profile->add_since(section_start, profile->gate_up_write_us);
    write_bf16_vector(grad_gate_lora_a, grad_gate_a_fp32);
    write_bf16_vector(grad_gate_lora_b, grad_gate_b_fp32);
    write_bf16_vector(grad_up_lora_a, grad_up_a_fp32);
    write_bf16_vector(grad_up_lora_b, grad_up_b_fp32);
  }

  void run_tp1_packed_backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a,
                               void* grad_gate_lora_b, void* grad_up_lora_a, void* grad_up_lora_b,
                               void* grad_down_lora_a, void* grad_down_lora_b, void* grad_weights) {
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 backward requires grad_output and grad_input");
    }

    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    if (lora_rank_ == 8 && lora_backward_matmat_enabled()) prepare_lora_b_transposed();
    bool should_pop_cache = true;
    TP1BackwardProfile profile(tp1_backward_profile_enabled());

    try {
      compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);
      profile.mark(profile.grad_weights_us);

      std::vector<float> grad_inter_fp32;
      compute_tp1_down_backward(cache, layout, grad_output, nullptr, nullptr, nullptr, &grad_inter_fp32,
                                grad_down_lora_a, grad_down_lora_b, &profile);
      profile.mark(profile.down_us);

      std::vector<float> grad_gate_fp32;
      std::vector<float> grad_up_fp32;

      compute_tp1_activation_backward_fp32(cache, layout, grad_inter_fp32.data(), grad_gate_fp32, grad_up_fp32);
      profile.mark(profile.activation_us);
      compute_tp1_gate_up_backward(cache, layout, grad_gate_fp32.data(), grad_up_fp32.data(), grad_input,
                                   grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b, &profile);
      profile.mark(profile.gate_up_us);

      if (profile.enabled) {
        std::fprintf(stderr,
                     "[KT_K2_SFT_PROFILE] layer=%d qlen=%d active=%d tokens=%zu down_base_kernel=%s "
                     "gate_up_base_kernel=%s grad_weights_us=%lld down_us=%lld "
                     "down_route_us=%lld down_write_us=%lld down_base_us=%lld down_lora_bprop_us=%lld "
                     "down_lora_a_us=%lld down_lora_b_us=%lld down_lora_matmat_du_dx_us=%lld "
                     "down_lora_matmat_da_db_us=%lld activation_us=%lld gate_up_us=%lld "
                     "gate_up_base_us=%lld gate_up_lora_u_us=%lld "
                     "gate_up_lora_b_us=%lld gate_up_lora_b_write_us=%lld "
                     "gate_up_lora_a_input_us=%lld gate_up_lora_matmat_du_dx_us=%lld "
                     "gate_up_lora_matmat_da_db_us=%lld gate_up_write_us=%lld total_us=%lld\n",
                     sft_config_.layer_idx, cache.qlen_cache, cache.activated_expert_cache, layout.total_tokens,
                     profile.down_base_kernel, profile.gate_up_base_kernel,
                     profile.grad_weights_us, profile.down_us, profile.down_route_us, profile.down_write_us,
                     profile.down_base_us, profile.down_lora_bprop_us, profile.down_lora_a_us, profile.down_lora_b_us,
                     profile.down_lora_matmat_du_dx_us, profile.down_lora_matmat_da_db_us, profile.activation_us,
                     profile.gate_up_us, profile.gate_up_base_us, profile.gate_up_lora_u_us,
                     profile.gate_up_lora_b_us, profile.gate_up_lora_b_write_us, profile.gate_up_lora_a_input_us,
                     profile.gate_up_lora_matmat_du_dx_us, profile.gate_up_lora_matmat_da_db_us,
                     profile.gate_up_write_us, profile.total_us());
      }

      pop_latest_cache();
      should_pop_cache = false;
    } catch (...) {
      if (should_pop_cache) {
        try {
          pop_latest_cache();
        } catch (...) {
        }
      }
      throw;
    }
  }

#if defined(__AVX512BF16__)
  void compute_tp_down_backward_workspace_v2(
      const K2ForwardCache& cache, const TP1BackwardLayout& layout, const void* grad_output,
      ggml_bf16_t* grad_down, float* grad_inter, float* down_du, void* grad_down_lora_a,
      float* fp32_grad_down_lora_b, int full_inter, bool accumulate_optimizer_grads,
      float optimizer_grad_scale, TP1BackwardProfile& profile) const {
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const auto* grad_out = reinterpret_cast<const ggml_bf16_t*>(grad_output);
    auto pool = config_.pool->get_subpool(tp_part_idx);

    auto section_start = profile.section_start();
    constexpr size_t kRowsPerTask = 8;
    const int route_tasks = static_cast<int>((layout.total_tokens + kRowsPerTask - 1) / kRowsPerTask);
    pool->do_work_stealing_job(
        route_tasks, nullptr,
        [&](int task_id) {
          const size_t row_begin = static_cast<size_t>(task_id) * kRowsPerTask;
          const size_t row_end = std::min(row_begin + kRowsPerTask, layout.total_tokens);
          for (size_t row = row_begin; row < row_end; row++) {
            const int token_idx = layout.row_to_token[row];
            scale_bf16_row_to_bf16(grad_out + static_cast<size_t>(token_idx) * hidden, grad_down + row * hidden,
                                    hidden, layout.row_to_weight[row]);
          }
        },
        nullptr);
    profile.add_since(section_start, profile.down_route_us);

    section_start = profile.section_start();
    profile.down_base_kernel = "bf16_workspace_v2";
    const int nth = T::recommended_nth(inter_size);
    pool->do_work_stealing_job(
        nth * cache.activated_expert_cache, nullptr,
        [&](int task_id) {
          const int expert_task = task_id / nth;
          const int ith = task_id % nth;
          const int expert_idx = cache.m_expert_id_map_cache[expert_task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const auto [col_begin, col_end] = T::split_range_n(inter_size, ith, nth);
          rawint4_backward_matmat_bf16_input(
              reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b), this->down_bb_[expert_idx]->d,
              grad_down + row_base * hidden, num_tokens, hidden, inter_size, col_begin, col_end,
              grad_inter + row_base * inter_size);
        },
        nullptr);
    profile.add_since(section_start, profile.down_base_us);

    if (!has_down_lora()) return;
    if (down_lora_b_transposed_.empty()) {
      throw std::runtime_error("K2 SFT workspace v2 requires transposed down LoRA B");
    }

    section_start = profile.section_start();
    auto* out_down_a = reinterpret_cast<ggml_bf16_t*>(grad_down_lora_a);
    pool->do_work_stealing_job(
        cache.activated_expert_cache, nullptr,
        [&](int task) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          float* du = down_du + row_base * 8;
          avx::lora_bf16_matmul_t4r4(
              grad_down + row_base * hidden,
              down_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * hidden, du, num_tokens,
              hidden, 8);
          avx::lora_backward_dx_rank8_matmat(
              du, down_lora_a_ + static_cast<size_t>(expert_idx) * 8 * inter_size,
              grad_inter + row_base * inter_size, num_tokens, inter_size, lora_scaling_);

          if (out_down_a != nullptr) {
            std::vector<float> grad_a(static_cast<size_t>(8) * inter_size, 0.0f);
            if (accumulate_optimizer_grads) {
              for (int r = 0; r < 8; r++) {
                const ggml_bf16_t* old_row =
                    out_down_a + (static_cast<size_t>(expert_idx) * 8 + r) * full_inter;
                float* dst = grad_a.data() + static_cast<size_t>(r) * inter_size;
                for (int i = 0; i < inter_size; i++) dst[i] = GGML_BF16_TO_FP32(old_row[i]);
              }
            }
            avx::lora_backward_da_rank8_matmat(cache.intermediate_cache + row_base * inter_size, nullptr, du,
                                               grad_a.data(), num_tokens, inter_size,
                                               lora_scaling_ * optimizer_grad_scale);
            for (int r = 0; r < 8; r++) {
              write_bf16_array(out_down_a + (static_cast<size_t>(expert_idx) * 8 + r) * full_inter,
                               grad_a.data() + static_cast<size_t>(r) * inter_size, inter_size);
            }
          }
          if (fp32_grad_down_lora_b != nullptr) {
            if (cache.down_lora_u_cache == nullptr) {
              throw std::runtime_error("K2 SFT workspace v2 requires cached down LoRA activations");
            }
            avx::lora_backward_db_rank8_matmat_bf16(
                cache.down_lora_u_cache + row_base * 8, grad_down + row_base * hidden,
                fp32_grad_down_lora_b + static_cast<size_t>(task) * hidden * 8, num_tokens, hidden,
                lora_scaling_ * optimizer_grad_scale, false);
          }
        },
        nullptr);
    profile.add_since(section_start, profile.down_lora_matmat_du_dx_us);
  }

  void compute_tp_activation_backward_workspace_v2(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                                    float* grad_gate, float* grad_up) const {
    const int inter_size = config_.intermediate_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        cache.activated_expert_cache, nullptr,
        [&](int task) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const size_t elem_base = row_base * inter_size;
          const int total = num_tokens * inter_size;
          const ggml_bf16_t* gate = cache.gate_output_cache + elem_base;
          const ggml_bf16_t* up = cache.up_output_cache + elem_base;
          float* gate_out = grad_gate + elem_base;
          float* up_out = grad_up + elem_base;
          const __m512 one = _mm512_set1_ps(1.0f);
          int i = 0;
          for (; i + 31 < total; i += 32) {
            __m512 g0, g1, u0, u1;
            avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(const_cast<ggml_bf16_t*>(gate + i)), &g0,
                                      &g1);
            avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(const_cast<ggml_bf16_t*>(up + i)), &u0, &u1);
            const __m512 gi0 = _mm512_loadu_ps(gate_out + i);
            const __m512 gi1 = _mm512_loadu_ps(gate_out + i + 16);
            const __m512 sig0 = _mm512_div_ps(one, _mm512_add_ps(one, avx512_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g0))));
            const __m512 sig1 = _mm512_div_ps(one, _mm512_add_ps(one, avx512_exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), g1))));
            const __m512 silu0 = _mm512_mul_ps(g0, sig0);
            const __m512 silu1 = _mm512_mul_ps(g1, sig1);
            const __m512 dsilu0 = _mm512_mul_ps(sig0, _mm512_fmadd_ps(g0, _mm512_sub_ps(one, sig0), one));
            const __m512 dsilu1 = _mm512_mul_ps(sig1, _mm512_fmadd_ps(g1, _mm512_sub_ps(one, sig1), one));
            _mm512_storeu_ps(gate_out + i, _mm512_mul_ps(_mm512_mul_ps(gi0, u0), dsilu0));
            _mm512_storeu_ps(gate_out + i + 16, _mm512_mul_ps(_mm512_mul_ps(gi1, u1), dsilu1));
            _mm512_storeu_ps(up_out + i, _mm512_mul_ps(gi0, silu0));
            _mm512_storeu_ps(up_out + i + 16, _mm512_mul_ps(gi1, silu1));
          }
          for (; i < total; i++) {
            const float gi = gate_out[i];
            const float g = GGML_BF16_TO_FP32(gate[i]);
            const float u = GGML_BF16_TO_FP32(up[i]);
            const float sigmoid = 1.0f / (1.0f + std::exp(-g));
            gate_out[i] = gi * u * sigmoid * (1.0f + g * (1.0f - sigmoid));
            up_out[i] = gi * g * sigmoid;
          }
        },
        nullptr);
  }

  void compute_tp_gate_up_backward_workspace_v2(
      const K2ForwardCache& cache, const TP1BackwardLayout& layout, float* grad_gate, float* grad_up,
      float* grad_input_fp32, float* gate_du, float* up_du, void* grad_input, void* grad_gate_lora_b,
      void* grad_up_lora_b, float* fp32_grad_gate_lora_a, float* fp32_grad_up_lora_a, int full_inter,
      bool accumulate_optimizer_grads, float optimizer_grad_scale, TP1BackwardProfile& profile) const {
    const int qlen = cache.qlen_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const bool use_lora = has_gate_up_lora();
    auto pool = config_.pool->get_subpool(tp_part_idx);

    auto section_start = profile.section_start();
    if (use_lora) {
      if (gate_lora_b_transposed_.empty() || up_lora_b_transposed_.empty()) {
        throw std::runtime_error("K2 SFT workspace v2 requires transposed gate/up LoRA B");
      }
      auto* out_gate_b = reinterpret_cast<ggml_bf16_t*>(grad_gate_lora_b);
      auto* out_up_b = reinterpret_cast<ggml_bf16_t*>(grad_up_lora_b);
      pool->do_work_stealing_job(
          cache.activated_expert_cache, nullptr,
          [&](int task) {
            const int expert_idx = cache.m_expert_id_map_cache[task];
            const int num_tokens = cache.m_local_num_cache[expert_idx];
            const size_t row_base = layout.expert_base[expert_idx];
            float* gate_du_expert = gate_du + row_base * 8;
            float* up_du_expert = up_du + row_base * 8;
            avx::lora_backward_du_rank8_matmat(
                grad_gate + row_base * inter_size,
                gate_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * inter_size, gate_du_expert,
                num_tokens, inter_size);
            avx::lora_backward_du_rank8_matmat(
                grad_up + row_base * inter_size,
                up_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * inter_size, up_du_expert,
                num_tokens, inter_size);

            const int* row_indices = layout.row_to_token.data() + row_base;
            if (fp32_grad_gate_lora_a != nullptr) {
              avx::lora_backward_da_rank8_matmat(
                  cache.input_cache, row_indices, gate_du_expert,
                  fp32_grad_gate_lora_a + static_cast<size_t>(task) * 8 * hidden, num_tokens, hidden,
                  lora_scaling_ * optimizer_grad_scale, false);
            }
            if (fp32_grad_up_lora_a != nullptr) {
              avx::lora_backward_da_rank8_matmat(
                  cache.input_cache, row_indices, up_du_expert,
                  fp32_grad_up_lora_a + static_cast<size_t>(task) * 8 * hidden, num_tokens, hidden,
                  lora_scaling_ * optimizer_grad_scale, false);
            }

            thread_local std::vector<float> gate_db;
            thread_local std::vector<float> up_db;
            if (out_gate_b != nullptr) {
              if (cache.gate_lora_u_cache == nullptr) {
                throw std::runtime_error("K2 SFT workspace v2 requires cached gate LoRA activations");
              }
              gate_db.resize(static_cast<size_t>(inter_size) * 8);
              avx::lora_backward_db_rank8_matmat(cache.gate_lora_u_cache + row_base * 8,
                                                 grad_gate + row_base * inter_size, gate_db.data(), num_tokens,
                                                 inter_size, lora_scaling_, false);
              write_optimizer_bf16_array(out_gate_b + static_cast<size_t>(expert_idx) * full_inter * 8,
                                         gate_db.data(), static_cast<size_t>(inter_size) * 8,
                                         accumulate_optimizer_grads, optimizer_grad_scale);
            }
            if (out_up_b != nullptr) {
              if (cache.up_lora_u_cache == nullptr) {
                throw std::runtime_error("K2 SFT workspace v2 requires cached up LoRA activations");
              }
              up_db.resize(static_cast<size_t>(inter_size) * 8);
              avx::lora_backward_db_rank8_matmat(cache.up_lora_u_cache + row_base * 8,
                                                 grad_up + row_base * inter_size, up_db.data(), num_tokens,
                                                 inter_size, lora_scaling_, false);
              write_optimizer_bf16_array(out_up_b + static_cast<size_t>(expert_idx) * full_inter * 8, up_db.data(),
                                         static_cast<size_t>(inter_size) * 8, accumulate_optimizer_grads,
                                         optimizer_grad_scale);
            }
          },
          nullptr);
    }
    profile.add_since(section_start, profile.gate_up_lora_matmat_da_db_us);

    section_start = profile.section_start();
    profile.gate_up_base_kernel = "bf16_direct_fused";
    const int nth = T::recommended_nth(hidden);
    pool->do_work_stealing_job(
        nth, nullptr,
        [&](int ith) {
          const auto [col_begin, col_end] = T::split_range_n(hidden, ith, nth);
          for (int token_idx = 0; token_idx < qlen; token_idx++) {
            float* dst = grad_input_fp32 + static_cast<size_t>(token_idx) * hidden + col_begin;
            std::fill(dst, dst + (col_end - col_begin), 0.0f);
          }
          for (int task = 0; task < cache.activated_expert_cache; task++) {
            const int expert_idx = cache.m_expert_id_map_cache[task];
            const int num_tokens = cache.m_local_num_cache[expert_idx];
            const size_t row_base = layout.expert_base[expert_idx];
            rawint4_gate_up_backward_matmat_bf16_direct(
                reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b), this->gate_bb_[expert_idx]->d,
                reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
                grad_gate + row_base * inter_size, grad_up + row_base * inter_size, num_tokens, inter_size, hidden,
                col_begin, col_end, layout.row_to_token.data() + row_base, grad_input_fp32);
            if (use_lora) {
              avx::lora_backward_dx_rank8_columns_indexed(
                  gate_du + row_base * 8, gate_lora_a_ + static_cast<size_t>(expert_idx) * 8 * hidden,
                  layout.row_to_token.data() + row_base, grad_input_fp32, num_tokens, hidden, col_begin, col_end,
                  lora_scaling_);
              avx::lora_backward_dx_rank8_columns_indexed(
                  up_du + row_base * 8, up_lora_a_ + static_cast<size_t>(expert_idx) * 8 * hidden,
                  layout.row_to_token.data() + row_base, grad_input_fp32, num_tokens, hidden, col_begin, col_end,
                  lora_scaling_);
            }
          }
        },
        nullptr);
    profile.add_since(section_start, profile.gate_up_direct_us);
    profile.gate_up_base_us += profile.gate_up_direct_us;

    section_start = profile.section_start();
    auto* out = reinterpret_cast<ggml_bf16_t*>(grad_input);
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int token_idx) {
          write_bf16_array(out + static_cast<size_t>(token_idx) * hidden,
                           grad_input_fp32 + static_cast<size_t>(token_idx) * hidden, hidden);
        },
        nullptr);
    profile.add_since(section_start, profile.gate_up_write_us);
  }
#endif

  // Normal SFT autograd backward for K2 packed weights.
  //
  // Python uses a mixed gradient transport contract:
  //   - grad_down_lora_a, grad_gate_lora_b, grad_up_lora_b are dense BF16 outputs.
  //   - fp32_grad_down_lora_b, fp32_grad_gate_lora_a, fp32_grad_up_lora_a are
  //     sparse FP32 outputs indexed by active expert task.
  //
  // The base path still reads packed signed-int4 KGroup rows directly for
  // down/gate/up; only trainable LoRA gradients are materialized.
  void run_tp_packed_backward(const void* grad_output, void* grad_input, void* grad_gate_lora_b, void* grad_up_lora_b,
                              void* grad_down_lora_a, void* grad_weights, int full_intermediate_size,
                              float* fp32_grad_down_lora_b, float* fp32_grad_gate_lora_a, float* fp32_grad_up_lora_a,
                              void* external_workspace = nullptr, size_t external_workspace_bytes = 0,
                              bool accumulate_optimizer_grads = false, float optimizer_grad_scale = 1.0f) {
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP backward requires grad_output and grad_input");
    }

    ensure_packed_weight_buffers_ready();

    // Normal TP Backward Step 1: Restore the latest forward cache and compact it into expert-packed row order.
    //
    // Execution order intentionally differs from forward numbering:
    //   Normal TP Backward Step 2 undoes forward Step 9 weighted merge.
    //   Normal TP Backward Step 3 undoes forward Step 8 down projection.
    //   Normal TP Backward Step 4 undoes forward Step 6 activation.
    //   Normal TP Backward Step 5 undoes forward Step 5 gate/up projection.
    //
    // Forward Steps 1-4 and 7 are routing, buffer setup/copies, and quantization staging.
    // They have no separate differentiable backward kernel here; routing is reused to
    // place gradients, and packed base-weight math writes the token-order gradients directly.
    auto canonical_stage_start = profiler_.start();
    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int full_inter = full_intermediate_size > 0 ? full_intermediate_size : inter_size;
    const int rank = lora_rank_;
    profiler_.record(SFTProfileStage::BwdCacheRestore, canonical_stage_start);
    profiler_.record_workload(static_cast<uint64_t>(qlen), static_cast<uint64_t>(qlen) * k,
                              static_cast<uint64_t>(cache.activated_expert_cache));

    if (rank == 8 && lora_backward_matmat_enabled()) prepare_lora_b_transposed();

    if (full_inter < inter_size) {
      throw std::runtime_error("K2 RAWINT4 SFT TP backward full_intermediate_size is smaller than TP local size");
    }

    bool should_pop_cache = true;
    TP1BackwardProfile profile(tp1_backward_profile_enabled());
    try {
#if defined(__AVX512BF16__)
      if (external_workspace != nullptr && backward_workspace_v2_eligible()) {
        BackwardWorkspaceV2 workspace =
            make_backward_workspace_v2(external_workspace, external_workspace_bytes, qlen, k);
        profile.workspace_bytes = workspace.required_bytes;
        profile.mark(profile.workspace_setup_us);

        canonical_stage_start = profiler_.start();
        compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);
        profiler_.record(SFTProfileStage::BwdRouterGrad, canonical_stage_start);
        profile.mark(profile.grad_weights_us);

        canonical_stage_start = profiler_.start();
        compute_tp_down_backward_workspace_v2(
            cache, layout, grad_output, workspace.grad_down, workspace.grad_inter, workspace.down_du,
            grad_down_lora_a, fp32_grad_down_lora_b, full_inter, accumulate_optimizer_grads, optimizer_grad_scale,
            profile);
        profiler_.record(SFTProfileStage::BwdDownTotal, canonical_stage_start);
        profile.mark(profile.down_us);

        canonical_stage_start = profiler_.start();
        compute_tp_activation_backward_workspace_v2(cache, layout, workspace.grad_gate, workspace.grad_up);
        profiler_.record(SFTProfileStage::BwdActivation, canonical_stage_start);
        profile.mark(profile.activation_us);

        canonical_stage_start = profiler_.start();
        compute_tp_gate_up_backward_workspace_v2(
            cache, layout, workspace.grad_gate, workspace.grad_up, workspace.grad_input, workspace.gate_du,
            workspace.up_du, grad_input, grad_gate_lora_b, grad_up_lora_b, fp32_grad_gate_lora_a,
            fp32_grad_up_lora_a, full_inter, accumulate_optimizer_grads, optimizer_grad_scale, profile);
        profiler_.record(SFTProfileStage::BwdGateUpTotal, canonical_stage_start);
        profile.mark(profile.gate_up_us);

        if (profile.enabled) {
          std::fprintf(
              stderr,
              "[KT_K2_SFT_PROFILE] layer=%d tp_part=%d qlen=%d active=%d tokens=%zu workspace_v2=1 "
              "workspace_bytes=%zu down_base_kernel=%s gate_up_base_kernel=%s workspace_setup_us=%lld "
              "grad_weights_us=%lld down_us=%lld down_lora_grads_us=%lld down_route_us=%lld down_write_us=%lld "
              "down_base_us=%lld down_lora_bprop_us=%lld down_lora_a_us=%lld down_lora_b_us=%lld "
              "down_lora_matmat_du_dx_us=%lld down_lora_matmat_da_db_us=%lld activation_us=%lld gate_up_us=%lld "
              "gate_up_base_us=%lld gate_up_lora_u_us=%lld gate_up_lora_b_us=%lld gate_up_lora_b_write_us=%lld "
              "gate_up_lora_a_input_us=%lld gate_up_lora_matmat_du_dx_us=%lld "
              "gate_up_lora_matmat_da_db_us=%lld gate_up_direct_us=%lld gate_up_write_us=%lld total_us=%lld\n",
              sft_config_.layer_idx, tp_part_idx, cache.qlen_cache, cache.activated_expert_cache,
              layout.total_tokens, profile.workspace_bytes, profile.down_base_kernel, profile.gate_up_base_kernel,
              profile.workspace_setup_us, profile.grad_weights_us, profile.down_us, profile.down_lora_grads_us,
              profile.down_route_us, profile.down_write_us, profile.down_base_us, profile.down_lora_bprop_us,
              profile.down_lora_a_us, profile.down_lora_b_us, profile.down_lora_matmat_du_dx_us,
              profile.down_lora_matmat_da_db_us, profile.activation_us, profile.gate_up_us,
              profile.gate_up_base_us, profile.gate_up_lora_u_us, profile.gate_up_lora_b_us,
              profile.gate_up_lora_b_write_us, profile.gate_up_lora_a_input_us,
              profile.gate_up_lora_matmat_du_dx_us, profile.gate_up_lora_matmat_da_db_us,
              profile.gate_up_direct_us, profile.gate_up_write_us, profile.total_us());
        }

        pop_latest_cache();
        should_pop_cache = false;
        return;
      }
#endif
      // Normal TP Backward Step 3.0: Decide which down LoRA gradients this TP shard must materialize.
      const bool use_down_lora = rank > 0 && has_down_lora();
      const bool need_down_a = grad_down_lora_a != nullptr;
      const bool need_down_b = fp32_grad_down_lora_b != nullptr;
      if ((need_down_a || need_down_b) && !use_down_lora) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires down LoRA weights");
      }
      if (need_down_b && cache.down_lora_u_cache == nullptr) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires cached down LoRA activations");
      }

      // Normal TP Backward Step 2.1: Backprop through forward Step 9 weighted merge into router/topk weights.
      compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);
      profile.mark(profile.grad_weights_us);

      // Normal TP Backward Steps 2.2 -> 3.1: Scatter grad_output to grad_down, then run down base backward.
      std::vector<float> grad_down_fp32;
      std::vector<float> grad_inter_fp32;
      std::vector<float> down_lora_grad_times_b_cache;
      const bool use_down_lora_bprop_cache =
          reuse_down_lora_bprop_enabled() && use_down_lora && (need_down_a || need_down_b);
      compute_tp1_down_backward(cache, layout, grad_output, nullptr,
                                (need_down_a || need_down_b) ? &grad_down_fp32 : nullptr, nullptr, &grad_inter_fp32,
                                nullptr, nullptr, &profile,
                                use_down_lora_bprop_cache ? &down_lora_grad_times_b_cache : nullptr);
      profile.mark(profile.down_us);

      if (use_down_lora && (need_down_a || need_down_b)) {
        // Normal TP Backward Steps 3.4/3.5: write down LoRA A directly to dense BF16
        // and down LoRA B to sparse FP32 side buffers indexed by active expert task.
        std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
        const bool have_down_lora_bprop_cache =
            use_down_lora_bprop_cache &&
            down_lora_grad_times_b_cache.size() == layout.total_tokens * static_cast<size_t>(rank);

        auto* out_down_a = reinterpret_cast<ggml_bf16_t*>(grad_down_lora_a);
        std::vector<float> grad_down_a_fp32;

        std::vector<uint8_t> down_weight_matmat_expert(static_cast<size_t>(config_.expert_num), 0);
        const bool try_down_weight_matmat = rank == 8 && lora_backward_matmat_enabled() &&
                                            have_down_lora_bprop_cache;
        if (try_down_weight_matmat) {
          const auto matmat_start = profile.section_start();
          auto compute_expert_lora_grads = [&](int task) {
            const int expert_idx = cache.m_expert_id_map_cache[task];
            const int num_tokens = cache.m_local_num_cache[expert_idx];
            if (num_tokens < 4) return;
            const size_t row_base = layout.expert_base[expert_idx];
            const float* du = down_lora_grad_times_b_cache.data() + row_base * 8;
            if (need_down_a) {
              std::vector<float> grad_a(static_cast<size_t>(8) * inter_size, 0.0f);
              for (int r = 0; r < 8; r++) {
                const ggml_bf16_t* old_row =
                    out_down_a + (static_cast<size_t>(expert_idx) * 8 + r) * full_inter;
                float* dst = grad_a.data() + static_cast<size_t>(r) * inter_size;
                for (int i = 0; i < inter_size; i++) dst[i] = GGML_BF16_TO_FP32(old_row[i]);
              }
              avx::lora_backward_da_rank8_matmat(cache.intermediate_cache + row_base * inter_size, nullptr, du,
                                                 grad_a.data(), num_tokens, inter_size, lora_scaling_);
              for (int r = 0; r < 8; r++) {
                ggml_bf16_t* out_row =
                    out_down_a + (static_cast<size_t>(expert_idx) * 8 + r) * full_inter;
                write_bf16_array(out_row, grad_a.data() + static_cast<size_t>(r) * inter_size, inter_size);
              }
            }
            if (need_down_b) {
              avx::lora_backward_db_rank8_matmat(
                  cache.down_lora_u_cache + row_base * 8, grad_down_fp32.data() + row_base * hidden,
                  fp32_grad_down_lora_b + static_cast<size_t>(task) * hidden * 8, num_tokens, hidden,
                  lora_scaling_);
            }
            down_weight_matmat_expert[static_cast<size_t>(expert_idx)] = 1;
          };
          if (layout.total_tokens >= 10 && config_.pool != nullptr) {
            auto pool = config_.pool->get_subpool(tp_part_idx);
            pool->do_work_stealing_job(cache.activated_expert_cache, nullptr, compute_expert_lora_grads, nullptr);
          } else {
            for (int task = 0; task < cache.activated_expert_cache; task++) compute_expert_lora_grads(task);
          }
          profile.add_since(matmat_start, profile.down_lora_matmat_da_db_us);
        }

        for (int task = 0; task < cache.activated_expert_cache; task++) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          if (down_weight_matmat_expert[static_cast<size_t>(expert_idx)] != 0) continue;
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;
          const ggml_bf16_t* expert_down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * rank * inter_size;

          if (need_down_a) {
            grad_down_a_fp32.assign(static_cast<size_t>(rank) * inter_size, 0.0f);
            for (int r = 0; r < rank; r++) {
              const ggml_bf16_t* old_row = out_down_a + (static_cast<size_t>(expert_idx) * rank + r) * full_inter;
              float* dst_row = grad_down_a_fp32.data() + static_cast<size_t>(r) * inter_size;
              for (int i = 0; i < inter_size; i++) dst_row[i] = GGML_BF16_TO_FP32(old_row[i]);
            }
          }

          for (int local_t = 0; local_t < num_tokens; local_t++) {
            const size_t row = row_base + static_cast<size_t>(local_t);
            const float* grad_down_row = grad_down_fp32.data() + row * hidden;

            std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);
            auto section_start = profile.section_start();
            if (have_down_lora_bprop_cache) {
              const float* cached_grad_times_b =
                  down_lora_grad_times_b_cache.data() + row * static_cast<size_t>(rank);
              std::copy(cached_grad_times_b, cached_grad_times_b + rank, grad_times_b.begin());
            } else {
              if (rank == 2) {
                float gb0;
                float gb1;
                down_bprop_rank2_vec(grad_down_row, expert_down_b, hidden, gb0, gb1);
                grad_times_b[0] = gb0;
                grad_times_b[1] = gb1;
              } else if (rank == 8) {
                down_bprop_rank8_vec(grad_down_row, expert_down_b, hidden, grad_times_b.data());
              } else {
                for (int h = 0; h < hidden; h++) {
                  const float g = grad_down_row[h];
                  if (g == 0.0f) continue;
                  const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * rank;
                  for (int r = 0; r < rank; r++) {
                    grad_times_b[r] += g * GGML_BF16_TO_FP32(down_b_row[r]);
                  }
                }
              }
              profile.add_since(section_start, profile.down_lora_bprop_us);
            }

            if (need_down_a) {
              section_start = profile.section_start();
              const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
              if (rank == 8) {
                alignas(32) float gu_scaled[8];
                for (int r = 0; r < 8; r++) gu_scaled[r] = grad_times_b[r] * lora_scaling_;
                accumulate_lora_a_input_rank8_vec(intermediate_row, expert_down_a, gu_scaled, inter_size, nullptr,
                                                  grad_down_a_fp32.data());
              } else {
                for (int r = 0; r < rank; r++) {
                  const float gu = grad_times_b[r] * lora_scaling_;
                  float* grad_a_row = grad_down_a_fp32.data() + static_cast<size_t>(r) * inter_size;
                  for (int i = 0; i < inter_size; i++) {
                    grad_a_row[i] += gu * GGML_BF16_TO_FP32(intermediate_row[i]);
                  }
                }
              }
              profile.add_since(section_start, profile.down_lora_a_us);
            }

            if (need_down_b) {
              section_start = profile.section_start();
              const float* down_u_row = cache.down_lora_u_cache + row * rank;
              float* grad_b = fp32_grad_down_lora_b + static_cast<size_t>(task) * hidden * rank;
              if (rank == 2) {
                accumulate_down_lora_b_rank2_vec(grad_down_row, down_u_row[0] * lora_scaling_,
                                                 down_u_row[1] * lora_scaling_, hidden, grad_b);
              } else if (rank == 8) {
                alignas(32) float u_scaled[8];
                for (int r = 0; r < 8; r++) u_scaled[r] = down_u_row[r] * lora_scaling_;
                accumulate_down_lora_b_rank8_vec(grad_down_row, u_scaled, hidden, grad_b);
              } else {
                for (int h = 0; h < hidden; h++) {
                  const float g = grad_down_row[h] * lora_scaling_;
                  if (g == 0.0f) continue;
                  float* grad_b_row = grad_b + static_cast<size_t>(h) * rank;
                  for (int r = 0; r < rank; r++) {
                    grad_b_row[r] += g * down_u_row[r];
                  }
                }
              }
              profile.add_since(section_start, profile.down_lora_b_us);
            }
          }

          if (need_down_a) {
            auto section_start = profile.section_start();
            for (int r = 0; r < rank; r++) {
              ggml_bf16_t* out_row = out_down_a + (static_cast<size_t>(expert_idx) * rank + r) * full_inter;
              write_bf16_array(out_row, grad_down_a_fp32.data() + static_cast<size_t>(r) * inter_size, inter_size);
            }
            profile.add_since(section_start, profile.down_lora_a_us);
          }
        }
      }
      profile.mark(profile.down_lora_grads_us);

      // Normal TP Backward Step 4.1: Backprop through silu(gate) * up.
      std::vector<float> grad_gate_fp32;
      std::vector<float> grad_up_fp32;
      compute_tp1_activation_backward_fp32(cache, layout, grad_inter_fp32.data(), grad_gate_fp32, grad_up_fp32);
      profile.mark(profile.activation_us);

      // Normal TP Backward Step 5.0: Decide which gate/up LoRA gradients are requested by the TP wrapper.
      const bool use_gate_up_lora = rank > 0 && has_gate_up_lora();
      const bool need_gate_up_lora = grad_gate_lora_b != nullptr || grad_up_lora_b != nullptr ||
                                     fp32_grad_gate_lora_a != nullptr || fp32_grad_up_lora_a != nullptr;
      if (need_gate_up_lora && !use_gate_up_lora) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires gate/up LoRA weights");
      }

      // Normal TP Backward Step 5.1 setup: accumulate token-order grad_input in FP32 before the final BF16 write.
      std::vector<float> grad_input_fp32(static_cast<size_t>(qlen) * hidden, 0.0f);
      std::vector<float> lora_u(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_gate_b_fp32;
      std::vector<float> grad_up_b_fp32;
      std::vector<uint8_t> gate_up_lora_matmat_expert(static_cast<size_t>(config_.expert_num), 0);
      const bool use_sparse_lora_b = sparse_lora_b_accum_enabled();
      if (use_sparse_lora_b && use_gate_up_lora && grad_gate_lora_b != nullptr) {
        grad_gate_b_fp32.assign(static_cast<size_t>(cache.activated_expert_cache) * inter_size * rank, 0.0f);
      }
      if (use_sparse_lora_b && use_gate_up_lora && grad_up_lora_b != nullptr) {
        grad_up_b_fp32.assign(static_cast<size_t>(cache.activated_expert_cache) * inter_size * rank, 0.0f);
      }

      // Normal TP Backward Step 5.1 helper: base gate/up dInput from packed KGroup rows.
      auto add_gate_up_base = [&](int expert_idx, int token_idx, size_t row) {
        float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
        const auto* gate_packed = reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b);
        const auto* up_packed = reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b);
        const float* gate_scales = this->gate_bb_[expert_idx]->d;
        const float* up_scales = this->up_bb_[expert_idx]->d;
        const float* grad_gate_row = grad_gate_fp32.data() + row * inter_size;
        const float* grad_up_row = grad_up_fp32.data() + row * inter_size;
        const bool use_four_row_fast_path = qlen >= 10 || short_base_fastpath_enabled();

        auto add_gate_up_base_row = [&](int i) {
          const float gate_g = grad_gate_row[i];
          const float up_g = grad_up_row[i];
          if (gate_g == 0.0f && up_g == 0.0f) return;
          if (gate_g != 0.0f && up_g != 0.0f) {
            add_scaled_two_packed_rows_f32(gate_packed, gate_scales, i, gate_g, up_packed, up_scales, i, up_g, hidden,
                                           grad_input_row);
          } else if (gate_g != 0.0f) {
            add_scaled_packed_row_f32(gate_packed, gate_scales, i, hidden, gate_g, grad_input_row);
          } else {
            add_scaled_packed_row_f32(up_packed, up_scales, i, hidden, up_g, grad_input_row);
          }
        };

        if (!use_four_row_fast_path) {
          for (int i = 0; i < inter_size; i++) {
            add_gate_up_base_row(i);
          }
          return;
        }

        int i = 0;
        for (; i + 15 < inter_size; i += 16) {
          add_scaled_sixteen_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                              grad_up_row + i, hidden, grad_input_row);
        }
        for (; i + 7 < inter_size; i += 8) {
          add_scaled_eight_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                            grad_up_row + i, hidden, grad_input_row);
        }
        for (; i + 3 < inter_size; i += 4) {
          add_scaled_four_gate_up_rows_f32(gate_packed, gate_scales, up_packed, up_scales, i, grad_gate_row + i,
                                           grad_up_row + i, hidden, grad_input_row);
        }
        for (; i + 1 < inter_size; i += 2) {
          const float gate_g0 = grad_gate_row[i];
          const float up_g0 = grad_up_row[i];
          const float gate_g1 = grad_gate_row[i + 1];
          const float up_g1 = grad_up_row[i + 1];
          add_scaled_four_packed_rows_f32(gate_packed, gate_scales, i, gate_g0, up_packed, up_scales, i, up_g0,
                                          gate_packed, gate_scales, i + 1, gate_g1, up_packed, up_scales, i + 1, up_g1,
                                          hidden, grad_input_row);
        }
        if (i < inter_size) {
          add_gate_up_base_row(i);
        }
      };

      auto backward_one_projection = [&](int task, int expert_idx, int token_idx, const float* grad_row,
                                         const uint8_t* packed_weight, const float* scales, const ggml_bf16_t* lora_a,
                                         const ggml_bf16_t* lora_b, float* fp32_grad_lora_a, float* fp32_grad_lora_b,
                                         void* grad_lora_b, const float* cached_lora_u, bool do_base, bool do_lora,
                                         bool profile_base_inside) {
        float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
        const ggml_bf16_t* input_row = cache.input_cache + static_cast<size_t>(token_idx) * hidden;

        if (do_base) {
          // Normal TP Backward Step 5.1: grad_input += grad_projection @ packed_base_weight.
          auto section_start =
              profile_base_inside ? profile.section_start() : TP1BackwardProfile::disabled_time_point();
          for (int i = 0; i < inter_size; i++) {
            const float g = grad_row[i];
            if (g == 0.0f) continue;
            add_scaled_packed_row_f32(packed_weight, scales, i, hidden, g, grad_input_row);
          }
          if (profile_base_inside) profile.add_since(section_start, profile.gate_up_base_us);
        }

        if (!do_lora || !use_gate_up_lora) return;

        std::fill(lora_u.begin(), lora_u.end(), 0.0f);
        std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);

        // Normal TP Backward Step 5.2: Reuse forward LoRA u = input @ A^T when available.
        auto section_start = profile.section_start();
        if (cached_lora_u != nullptr) {
          std::copy(cached_lora_u, cached_lora_u + rank, lora_u.begin());
        } else if (rank == 2) {
          const ggml_bf16_t* a0 = lora_a;
          const ggml_bf16_t* a1 = lora_a + hidden;
          float acc0;
          float acc1;
          bf16_dot2_scalar(input_row, a0, a1, hidden, acc0, acc1);
          lora_u[0] = acc0;
          lora_u[1] = acc1;
        } else if (rank == 8) {
          bf16_dot8_lora_u(input_row, lora_a, hidden, lora_u.data());
        } else {
          for (int r = 0; r < rank; r++) {
            const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
            float acc = 0.0f;
            for (int h = 0; h < hidden; h++) {
              acc += GGML_BF16_TO_FP32(input_row[h]) * GGML_BF16_TO_FP32(a_row[h]);
            }
            lora_u[r] = acc;
          }
        }
        profile.add_since(section_start, profile.gate_up_lora_u_us);

        // Normal TP Backward Step 5.3: Accumulate LoRA B gradient and compute grad_times_b = grad_projection @ B.
        auto* out_lora_b = reinterpret_cast<ggml_bf16_t*>(grad_lora_b);
        section_start = profile.section_start();
        if (rank == 2) {
          const float u0_scaled = lora_u[0] * lora_scaling_;
          const float u1_scaled = lora_u[1] * lora_scaling_;
          float gb0 = 0.0f;
          float gb1 = 0.0f;
          if (fp32_grad_lora_b != nullptr) {
            float* grad_b = fp32_grad_lora_b + static_cast<size_t>(task) * inter_size * 2;
            accumulate_gate_up_lora_b_rank2_vec(grad_row, lora_b, u0_scaled, u1_scaled, inter_size, grad_b, gb0, gb1);
          } else {
            for (int i = 0; i < inter_size; i++) {
              const float g = grad_row[i];
              if (g == 0.0f) continue;
              const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * 2;
              if (out_lora_b != nullptr) {
                ggml_bf16_t* out_b_row =
                    out_lora_b + static_cast<size_t>(expert_idx) * full_inter * 2 + static_cast<size_t>(i) * 2;
                out_b_row[0] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_b_row[0]) + g * u0_scaled);
                out_b_row[1] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(out_b_row[1]) + g * u1_scaled);
              }
              gb0 += g * GGML_BF16_TO_FP32(b_row[0]);
              gb1 += g * GGML_BF16_TO_FP32(b_row[1]);
            }
          }
          grad_times_b[0] = gb0;
          grad_times_b[1] = gb1;
        } else if (rank == 8 && fp32_grad_lora_b != nullptr) {
          alignas(32) float u_scaled[8];
          for (int r = 0; r < 8; r++) u_scaled[r] = lora_u[r] * lora_scaling_;
          float* grad_b = fp32_grad_lora_b + static_cast<size_t>(task) * inter_size * 8;
          accumulate_gate_up_lora_b_rank8_vec(grad_row, lora_b, u_scaled, inter_size, grad_b,
                                              grad_times_b.data());
        } else {
          for (int i = 0; i < inter_size; i++) {
            const float g = grad_row[i];
            if (g == 0.0f) continue;
            const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * rank;
            if (fp32_grad_lora_b != nullptr) {
              float* out_b_row =
                  fp32_grad_lora_b + (static_cast<size_t>(task) * inter_size + static_cast<size_t>(i)) * rank;
              for (int r = 0; r < rank; r++) {
                out_b_row[r] += g * lora_u[r] * lora_scaling_;
              }
            } else if (out_lora_b != nullptr) {
              ggml_bf16_t* out_b_row =
                  out_lora_b + static_cast<size_t>(expert_idx) * full_inter * rank + static_cast<size_t>(i) * rank;
              for (int r = 0; r < rank; r++) {
                const float old_v = GGML_BF16_TO_FP32(out_b_row[r]);
                out_b_row[r] = GGML_FP32_TO_BF16(old_v + g * lora_u[r] * lora_scaling_);
              }
            }
            for (int r = 0; r < rank; r++) {
              grad_times_b[r] += g * GGML_BF16_TO_FP32(b_row[r]);
            }
          }
        }
        profile.add_since(section_start, profile.gate_up_lora_b_us);

        // Normal TP Backward Step 5.4: Accumulate sparse FP32 LoRA A gradient and add LoRA dInput.
        section_start = profile.section_start();
        if (rank == 2 && tp_gate_up_a_input_rank2_vec_enabled()) {
          float* grad_a0 = nullptr;
          float* grad_a1 = nullptr;
          if (fp32_grad_lora_a != nullptr) {
            grad_a0 = fp32_grad_lora_a + static_cast<size_t>(task) * 2 * hidden;
            grad_a1 = grad_a0 + hidden;
          }
          accumulate_gate_up_a_input_rank2_vec(input_row, lora_a, lora_a + hidden, grad_times_b[0] * lora_scaling_,
                                               grad_times_b[1] * lora_scaling_, hidden, grad_input_row, grad_a0,
                                               grad_a1, false);
        } else if (rank == 8) {
          alignas(32) float gu_scaled[8];
          for (int r = 0; r < 8; r++) gu_scaled[r] = grad_times_b[r] * lora_scaling_;
          float* grad_a =
              fp32_grad_lora_a == nullptr ? nullptr : fp32_grad_lora_a + static_cast<size_t>(task) * 8 * hidden;
          accumulate_lora_a_input_rank8_vec(input_row, lora_a, gu_scaled, hidden, grad_input_row, grad_a);
        } else {
          for (int r = 0; r < rank; r++) {
            const float gu = grad_times_b[r] * lora_scaling_;
            const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
            if (fp32_grad_lora_a != nullptr) {
              float* grad_a_row = fp32_grad_lora_a + (static_cast<size_t>(task) * rank + r) * hidden;
              for (int h = 0; h < hidden; h++) {
                const float x = GGML_BF16_TO_FP32(input_row[h]);
                grad_a_row[h] += gu * x;
                grad_input_row[h] += gu * GGML_BF16_TO_FP32(a_row[h]);
              }
            } else {
              for (int h = 0; h < hidden; h++) {
                grad_input_row[h] += gu * GGML_BF16_TO_FP32(a_row[h]);
              }
            }
          }
        }
        profile.add_since(section_start, profile.gate_up_lora_a_input_us);
      };

      // Normal TP Backward Step 5.5: Walk token routes and apply Backward Steps 5.1-5.4 to gate and up projections.
      auto run_token_routes = [&](int token_idx, bool do_base, bool do_lora, bool profile_base_inside) {
        for (int route_idx = 0; route_idx < k; route_idx++) {
          const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
          if (config_.should_skip_expert(expert_id)) continue;

          const int expert_idx = static_cast<int>(expert_id);
          const int task = layout.expert_task_index[expert_idx];
          const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
          const size_t row = layout.expert_base[static_cast<size_t>(expert_idx)] + static_cast<size_t>(local_pos);

          if (do_base) {
            auto section_start =
                profile_base_inside ? profile.section_start() : TP1BackwardProfile::disabled_time_point();
            add_gate_up_base(expert_idx, token_idx, row);
            if (profile_base_inside) profile.add_since(section_start, profile.gate_up_base_us);
          }
          if (!do_lora) continue;
          if (gate_up_lora_matmat_expert[static_cast<size_t>(expert_idx)] != 0) continue;

          backward_one_projection(
              task, expert_idx, token_idx, grad_gate_fp32.data() + row * inter_size,
              reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b), this->gate_bb_[expert_idx]->d,
              use_gate_up_lora ? gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
              use_gate_up_lora ? gate_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
              fp32_grad_gate_lora_a, grad_gate_b_fp32.empty() ? nullptr : grad_gate_b_fp32.data(),
              grad_gate_b_fp32.empty() ? grad_gate_lora_b : nullptr,
              cache.gate_lora_u_cache == nullptr ? nullptr : cache.gate_lora_u_cache + row * static_cast<size_t>(rank),
              false, true, false);
          backward_one_projection(
              task, expert_idx, token_idx, grad_up_fp32.data() + row * inter_size,
              reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
              use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
              use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
              fp32_grad_up_lora_a, grad_up_b_fp32.empty() ? nullptr : grad_up_b_fp32.data(),
              grad_up_b_fp32.empty() ? grad_up_lora_b : nullptr,
              cache.up_lora_u_cache == nullptr ? nullptr : cache.up_lora_u_cache + row * static_cast<size_t>(rank), false,
              true, false);
        }
      };

      auto gate_up_base_start = profile.section_start();
      const bool expert_matmat_base = compute_gate_up_base_expert_matmat(
          cache, layout, grad_gate_fp32.data(), grad_up_fp32.data(), grad_input_fp32, &profile);
      if (expert_matmat_base) profile.add_since(gate_up_base_start, profile.gate_up_base_us);

      const bool gate_b_sparse_ok = grad_gate_lora_b == nullptr || !grad_gate_b_fp32.empty();
      const bool up_b_sparse_ok = grad_up_lora_b == nullptr || !grad_up_b_fp32.empty();
      const bool try_gate_up_lora_matmat =
          use_gate_up_lora && rank == 8 && lora_backward_matmat_enabled() && gate_b_sparse_ok && up_b_sparse_ok &&
          cache.gate_lora_u_cache != nullptr && cache.up_lora_u_cache != nullptr &&
          !gate_lora_b_transposed_.empty() && !up_lora_b_transposed_.empty();
      if (try_gate_up_lora_matmat) {
        std::vector<float> gate_du(layout.total_tokens * 8, 0.0f);
        std::vector<float> up_du(layout.total_tokens * 8, 0.0f);
        gate_up_route_grad_scratch_.resize(layout.total_tokens * static_cast<size_t>(hidden));

        auto du_dx_start = profile.section_start();
        auto compute_expert_du_dx = [&](int task) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          if (num_tokens < 4) return;
          const size_t row_base = layout.expert_base[expert_idx];
          float* gate_du_expert = gate_du.data() + row_base * 8;
          float* up_du_expert = up_du.data() + row_base * 8;
          avx::lora_backward_du_rank8_matmat(
              grad_gate_fp32.data() + row_base * inter_size,
              gate_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * inter_size, gate_du_expert,
              num_tokens, inter_size);
          avx::lora_backward_du_rank8_matmat(
              grad_up_fp32.data() + row_base * inter_size,
              up_lora_b_transposed_.data() + static_cast<size_t>(expert_idx) * 8 * inter_size, up_du_expert,
              num_tokens, inter_size);

          std::vector<float> route_dx(static_cast<size_t>(num_tokens) * hidden, 0.0f);
          avx::lora_backward_dx_rank8_matmat(
              gate_du_expert, gate_lora_a_ + static_cast<size_t>(expert_idx) * 8 * hidden, route_dx.data(),
              num_tokens, hidden, lora_scaling_);
          avx::lora_backward_dx_rank8_matmat(
              up_du_expert, up_lora_a_ + static_cast<size_t>(expert_idx) * 8 * hidden, route_dx.data(), num_tokens,
              hidden, lora_scaling_);
          write_bf16_array(gate_up_route_grad_scratch_.data() + row_base * hidden, route_dx.data(),
                           static_cast<size_t>(num_tokens) * hidden);
          gate_up_lora_matmat_expert[static_cast<size_t>(expert_idx)] = 1;
        };
        if (layout.total_tokens >= 10 && config_.pool != nullptr) {
          auto pool = config_.pool->get_subpool(tp_part_idx);
          pool->do_work_stealing_job(cache.activated_expert_cache, nullptr, compute_expert_du_dx, nullptr);
        } else {
          for (int task = 0; task < cache.activated_expert_cache; task++) compute_expert_du_dx(task);
        }

        for (int token_idx = 0; token_idx < qlen; token_idx++) {
          float* dst = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
          for (int route_idx = 0; route_idx < k; route_idx++) {
            const int expert_idx = static_cast<int>(cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx]);
            if (config_.should_skip_expert(expert_idx) ||
                gate_up_lora_matmat_expert[static_cast<size_t>(expert_idx)] == 0) {
              continue;
            }
            const size_t row = layout.expert_base[expert_idx] +
                               static_cast<size_t>(cache.m_local_pos_cache[token_idx][route_idx]);
            const ggml_bf16_t* src = gate_up_route_grad_scratch_.data() + row * hidden;
            for (int h = 0; h < hidden; h++) dst[h] += GGML_BF16_TO_FP32(src[h]);
          }
        }
        profile.add_since(du_dx_start, profile.gate_up_lora_matmat_du_dx_us);

        auto da_db_start = profile.section_start();
        auto compute_expert_da_db = [&](int task) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          if (gate_up_lora_matmat_expert[static_cast<size_t>(expert_idx)] == 0) return;
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const int* row_indices = layout.row_to_token.data() + row_base;
          if (fp32_grad_gate_lora_a != nullptr) {
            avx::lora_backward_da_rank8_matmat(
                cache.input_cache, row_indices, gate_du.data() + row_base * 8,
                fp32_grad_gate_lora_a + static_cast<size_t>(task) * 8 * hidden, num_tokens, hidden, lora_scaling_);
          }
          if (fp32_grad_up_lora_a != nullptr) {
            avx::lora_backward_da_rank8_matmat(
                cache.input_cache, row_indices, up_du.data() + row_base * 8,
                fp32_grad_up_lora_a + static_cast<size_t>(task) * 8 * hidden, num_tokens, hidden, lora_scaling_);
          }
          if (!grad_gate_b_fp32.empty()) {
            avx::lora_backward_db_rank8_matmat(
                cache.gate_lora_u_cache + row_base * 8, grad_gate_fp32.data() + row_base * inter_size,
                grad_gate_b_fp32.data() + static_cast<size_t>(task) * inter_size * 8, num_tokens, inter_size,
                lora_scaling_);
          }
          if (!grad_up_b_fp32.empty()) {
            avx::lora_backward_db_rank8_matmat(
                cache.up_lora_u_cache + row_base * 8, grad_up_fp32.data() + row_base * inter_size,
                grad_up_b_fp32.data() + static_cast<size_t>(task) * inter_size * 8, num_tokens, inter_size,
                lora_scaling_);
          }
        };
        if (layout.total_tokens >= 10 && config_.pool != nullptr) {
          auto pool = config_.pool->get_subpool(tp_part_idx);
          pool->do_work_stealing_job(cache.activated_expert_cache, nullptr, compute_expert_da_db, nullptr);
        } else {
          for (int task = 0; task < cache.activated_expert_cache; task++) compute_expert_da_db(task);
        }
        profile.add_since(da_db_start, profile.gate_up_lora_matmat_da_db_us);
      }

      if (expert_matmat_base) {
        // Normal TP Backward Step 5.6: expert mat-mat base dInput first; LoRA remains race-free afterward.
        if (use_gate_up_lora) {
          for (int token_idx = 0; token_idx < qlen; token_idx++) run_token_routes(token_idx, false, true, false);
        }
      } else {
        for (int token_idx = 0; token_idx < qlen; token_idx++) {
          run_token_routes(token_idx, true, true, true);
        }
      }

      if ((grad_gate_lora_b != nullptr && !grad_gate_b_fp32.empty()) ||
          (grad_up_lora_b != nullptr && !grad_up_b_fp32.empty())) {
        // Normal TP Backward Step 5.7: Convert sparse gate/up LoRA B partials to final dense BF16 TP slices.
        auto section_start = profile.section_start();
        auto write_sparse_lora_b = [&](void* dst_ptr, const std::vector<float>& src) {
          if (dst_ptr == nullptr || src.empty()) return;
          auto* dst = reinterpret_cast<ggml_bf16_t*>(dst_ptr);
          for (int task = 0; task < cache.activated_expert_cache; task++) {
            const int expert_idx = cache.m_expert_id_map_cache[task];
            const float* src_expert = src.data() + static_cast<size_t>(task) * inter_size * rank;
            ggml_bf16_t* dst_expert = dst + static_cast<size_t>(expert_idx) * full_inter * rank;
            write_bf16_array(dst_expert, src_expert, static_cast<size_t>(inter_size) * rank);
          }
        };
        write_sparse_lora_b(grad_gate_lora_b, grad_gate_b_fp32);
        write_sparse_lora_b(grad_up_lora_b, grad_up_b_fp32);
        profile.add_since(section_start, profile.gate_up_lora_b_write_us);
      }

      // Normal TP Backward Step 5.9: Write final grad_input for this TP shard.
      auto section_start = profile.section_start();
      write_bf16_vector(grad_input, grad_input_fp32);
      profile.add_since(section_start, profile.gate_up_write_us);
      profile.mark(profile.gate_up_us);
      if (profile.enabled) {
        std::fprintf(stderr,
                     "[KT_K2_SFT_PROFILE] layer=%d tp_part=%d qlen=%d active=%d tokens=%zu down_base_kernel=%s "
                     "gate_up_base_kernel=%s grad_weights_us=%lld "
                     "down_us=%lld down_lora_grads_us=%lld down_route_us=%lld down_write_us=%lld "
                     "down_base_us=%lld down_lora_bprop_us=%lld down_lora_a_us=%lld down_lora_b_us=%lld "
                     "down_lora_matmat_du_dx_us=%lld down_lora_matmat_da_db_us=%lld "
                     "activation_us=%lld gate_up_us=%lld "
                     "gate_up_base_us=%lld gate_up_lora_u_us=%lld gate_up_lora_b_us=%lld "
                     "gate_up_lora_b_write_us=%lld gate_up_lora_a_input_us=%lld "
                     "gate_up_lora_matmat_du_dx_us=%lld gate_up_lora_matmat_da_db_us=%lld "
                     "gate_up_write_us=%lld total_us=%lld\n",
                     sft_config_.layer_idx, tp_part_idx, cache.qlen_cache, cache.activated_expert_cache,
                     layout.total_tokens, profile.down_base_kernel, profile.gate_up_base_kernel,
                     profile.grad_weights_us, profile.down_us, profile.down_lora_grads_us,
                     profile.down_route_us, profile.down_write_us, profile.down_base_us, profile.down_lora_bprop_us,
                     profile.down_lora_a_us, profile.down_lora_b_us, profile.down_lora_matmat_du_dx_us,
                     profile.down_lora_matmat_da_db_us, profile.activation_us, profile.gate_up_us,
                     profile.gate_up_base_us, profile.gate_up_lora_u_us, profile.gate_up_lora_b_us,
                     profile.gate_up_lora_b_write_us, profile.gate_up_lora_a_input_us,
                     profile.gate_up_lora_matmat_du_dx_us, profile.gate_up_lora_matmat_da_db_us,
                     profile.gate_up_write_us, profile.total_us());
      }
      // Normal TP Backward Step 6: Release the forward cache consumed by this backward.
      pop_latest_cache();
      should_pop_cache = false;
    } catch (...) {
      if (should_pop_cache) {
        try {
          pop_latest_cache();
        } catch (...) {
        }
      }
      throw;
    }
  }

 public:
  static constexpr bool kSkipLoRA = SkipLoRA;
  static constexpr bool kIsInt8Backend = false;
  static constexpr bool kIsFP8Backend = false;
  static constexpr bool kSupportsDirectBf16Reload = false;
  static constexpr bool kSupportsAuthoritativeBaseGrads = false;
  static constexpr bool kSupportsAuthoritativeLoraGrads = !SkipLoRA;
  static constexpr bool kUsesKGroupPackedBaseWeights = true;
  static constexpr bool kHasInt4PackedBackward = true;
  static constexpr bool kSupportsForwardCache = true;
  static constexpr bool kSupportsBackward = true;
  static constexpr bool kSupportsTPReferenceBackward = false;
  static constexpr bool kSupportsTP1DirectBackward = false;
  static constexpr bool kSupportsExternalBackwardWorkspace = true;

  using typename Base::input_t;
  using typename Base::output_t;

  AMX_K2_SFT_MOE_TP() = default;

  AMX_K2_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(validated_base_config(config), tp_part_idx), sft_config_(config) {
    if (config.full_weight_grad) {
      throw std::runtime_error("K2 RAWINT4 SFT supports frozen-base LoRA only");
    }
    if constexpr (!SkipLoRA) {
      if (config.lora_rank != 8) {
        throw std::runtime_error("K2 RAWINT4 SFT currently requires LoRA rank 8");
      }
    }
    lora_rank_ = config.lora_rank;
    lora_scaling_ = config.lora_scaling();
    max_cache_depth_ = std::max(1, config.max_cache_depth);
    cache_stack_.resize(max_cache_depth_);
    cache_offsets_.assign(config.expert_num + 1, 0);

    update_lora_weights(config.gate_lora_a, config.gate_lora_b, config.up_lora_a, config.up_lora_b, config.down_lora_a,
                        config.down_lora_b);

    printf("Creating AMX_K2_SFT_MOE_TP layer=%d tp_part=%d skiplora %s\n", config.layer_idx, tp_part_idx,
           SkipLoRA ? "true" : "false");
  }

  AMX_K2_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : AMX_K2_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}

  void append_profile_stats(std::map<std::string, double>& out, const std::string& prefix,
                            bool reset_after = false) {
    profiler_.append(out, prefix, reset_after);
  }

  void reset_profile_stats() { profiler_.reset(); }

  void set_lora_params(int rank, float alpha) {
    lora_rank_ = rank;
    lora_scaling_ = rank == 0 ? 0.0f : alpha / rank;
    clear_transposed_lora_weights();
  }

  void update_lora_weights(void* gate_lora_a, void* gate_lora_b, void* up_lora_a, void* up_lora_b, void* down_lora_a,
                           void* down_lora_b) {
    if constexpr (SkipLoRA) {
      return;
    }
    gate_lora_a_ = reinterpret_cast<ggml_bf16_t*>(gate_lora_a);
    gate_lora_b_ = reinterpret_cast<ggml_bf16_t*>(gate_lora_b);
    up_lora_a_ = reinterpret_cast<ggml_bf16_t*>(up_lora_a);
    up_lora_b_ = reinterpret_cast<ggml_bf16_t*>(up_lora_b);
    down_lora_a_ = reinterpret_cast<ggml_bf16_t*>(down_lora_a);
    down_lora_b_ = reinterpret_cast<ggml_bf16_t*>(down_lora_b);
    clear_transposed_lora_weights();
  }

  void forward_sft(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output,
                   bool save_for_backward) {
    SFTProfileScope total_scope(profiler_, SFTProfileStage::FwdTotal);
    trace_forward_step("enter", qlen, k, -1, save_for_backward);
    if (qlen > config_.max_len) {
      throw std::runtime_error("K2 RAWINT4 SFT qlen exceeds max_len");
    }

    ForwardProfile profile(profile_forward_enabled());

    // K2 fast path: with no LoRA and no backward cache, delegate to the native packed KGroup forward.
    if (!save_for_backward && !has_any_lora()) {
      trace_forward_step("fast_path_base_forward", qlen, k, -1, save_for_backward);
      Base::forward(qlen, k, expert_ids, weights, input, output);
      profile.mark(profile.base_forward_us);
      if (profile.enabled) {
        fprintf(stderr,
                "[KT_K2_SFT_FWD_PROFILE] layer=%d tp_part=%d qlen=%d k=%d active=-1 save=%d lora=0 cache_top=%d "
                "fast_path=1 base_forward_us=%lld total_us=%lld\n",
                config_.layer_idx, tp_part_idx, qlen, k, save_for_backward ? 1 : 0, cache_stack_top_,
                profile.base_forward_us, profile.total_us());
        fflush(stderr);
      }
      trace_forward_step("done", qlen, k, -1, save_for_backward);
      return;
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    profile.reset();

    // Step 1: Expert routing (reuse base class logic)
    // K2 factors the equivalent routing loop into route_tokens(); it also uses should_skip_expert()
    // so invalid experts and GPU-resident experts follow the shared GeneralMOEConfig skip rule.
    trace_forward_step("step1_route_tokens", qlen, k, -1, save_for_backward);
    int activated_expert = route_tokens(qlen, k, expert_ids);
    profile.mark(profile.route_us);

    // Step 2: Buffer pool allocation (reuse base class logic)
    // K2 keeps the same per-expert buffer layout, but the allocation math is factored into a helper.
    trace_forward_step("step2_setup_expert_buffers", qlen, k, activated_expert, save_for_backward);
    setup_expert_buffers();
    profile.mark(profile.setup_us);

    // Step 3: Copy input to expert buffers
    trace_forward_step("step3_copy_inputs", qlen, k, activated_expert, save_for_backward);
    copy_inputs_to_expert_buffers(qlen, k, expert_ids, input);
    profile.mark(profile.copy_input_us);

    // Small-q_len runs inline to avoid thread-pool overhead; this is a scheduling detail, not a forward step.
    auto direct_or_pool = [&](int count, auto&& fn) {
      if (count <= 0) return;
      if (qlen < 10) {
        for (int i = 0; i < count; i++) fn(i);
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    // Step 4: Quantize input
    trace_forward_step("step4_quantize_input", qlen, k, activated_expert, save_for_backward);
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = this->m_expert_id_map_[task_id];
      this->gate_up_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx], this->m_local_input_ptr_[expert_idx], 0,
                                              1);
    });
    profile.mark(profile.q_input_us);

    // Step 5: Gate + Up GEMM (base projection)
    // K2's do_gate_up_gemm dispatches packed int4 KGroup GEMM instead of the generic AMX BufferB path.
    trace_forward_step("step5_gate_up_base_gemm", qlen, k, activated_expert, save_for_backward);
    int nth = T::recommended_nth(config_.intermediate_size);
    if (activated_expert > 0) {
      pool->do_work_stealing_job(
          nth * activated_expert * 2, [](int _) { T::config(); },
          [this, nth, qlen](int task_id2) {
            int task_id = task_id2 / 2;
            bool do_up = task_id2 % 2;
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
            if (do_up) {
              this->up_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx], this->m_local_up_output_ptr_[expert_idx],
                                               ith, nth);
            } else {
              this->gate_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_gate_output_ptr_[expert_idx], ith, nth);
            }
          },
          nullptr);
    }
    profile.mark(profile.gate_up_base_us);

    K2ForwardCache* cache_ptr = nullptr;
    ForwardCacheReservationGuard cache_guard(*this);
    if (save_for_backward) {
      trace_forward_step("step5_4_prepare_backward_cache", qlen, k, activated_expert, save_for_backward);
      K2ForwardCache& cache = push_cache();
      cache_ptr = &cache;
      cache_guard.reserve(cache);
      prepare_cache_for_backward(cache, qlen, k, expert_ids, weights, activated_expert, input);
      profile.mark(profile.save_gate_up_us);
    }

    // Step 5.5: Gate + Up LoRA (AVX512 BF16 - no BufferB conversion needed)
    trace_forward_step("step5_5_gate_up_lora", qlen, k, activated_expert, save_for_backward);
    compute_lora_gate_up(activated_expert, cache_ptr);
    profile.mark(profile.gate_up_lora_us);

    if (save_for_backward && cache_ptr != nullptr) {
      // Save gate/up outputs before activation (for backward).
      trace_forward_step("step5_6_save_gate_up_cache", qlen, k, activated_expert, save_for_backward);
      save_gate_up_to_cache(*cache_ptr, activated_expert);
      profile.mark(profile.save_gate_up_us);
    }

    // Step 6: Activation (silu(gate) * up)
    trace_forward_step("step6_activation", qlen, k, activated_expert, save_for_backward);
    this->apply_activation(activated_expert, nth, qlen);
    profile.mark(profile.act_us);

    if (save_for_backward && cache_ptr != nullptr) {
      // Save intermediate AFTER activation for backward_down.
      trace_forward_step("step6_5_save_intermediate_cache", qlen, k, activated_expert, save_for_backward);
      save_intermediate_to_cache(*cache_ptr, activated_expert);
      profile.mark(profile.save_intermediate_us);
    }

    // Step 7: Quantize intermediate for down projection
    trace_forward_step("step7_quantize_intermediate", qlen, k, activated_expert, save_for_backward);
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = this->m_expert_id_map_[task_id];
      this->down_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx], this->m_local_gate_output_ptr_[expert_idx],
                                           0, 1);
    });
    profile.mark(profile.q_intermediate_us);

    // Step 8: Down GEMM
    // K2's do_down_gemm dispatches packed int4 KGroup GEMM.
    trace_forward_step("step8_down_base_gemm", qlen, k, activated_expert, save_for_backward);
    nth = T::recommended_nth(config_.hidden_size);
    if (activated_expert > 0) {
      pool->do_work_stealing_job(
          nth * activated_expert, [](int _) { T::config(); },
          [this, nth, qlen](int task_id) {
            int expert_idx = this->m_expert_id_map_[task_id / nth];
            int ith = task_id % nth;
            this->do_down_gemm(expert_idx, ith, nth, qlen);
            this->down_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                               this->m_local_down_output_ptr_[expert_idx], ith, nth);
          },
          nullptr);
    }
    profile.mark(profile.down_base_us);

    // Step 8.5: Down LoRA (AVX512 BF16 - no BufferB conversion needed)
    trace_forward_step("step8_5_down_lora", qlen, k, activated_expert, save_for_backward);
    compute_lora_down(activated_expert, cache_ptr);
    profile.mark(profile.down_lora_us);

    if (save_for_backward && cache_ptr != nullptr) {
      // Save down_output for grad_weights computation.
      trace_forward_step("step8_6_save_down_cache", qlen, k, activated_expert, save_for_backward);
      save_down_output_to_cache(*cache_ptr, activated_expert);
      profile.mark(profile.save_down_us);
    }

    // Step 9: Weighted merge
    trace_forward_step("step9_weighted_merge", qlen, k, activated_expert, save_for_backward);
    direct_or_pool(qlen, [this, output, k, expert_ids, weights](int i) {
      for (int e = 0; e < config_.hidden_size; e += 32) {
        __m512 x0 = _mm512_setzero_ps();
        __m512 x1 = _mm512_setzero_ps();
        for (int j = 0; j < k; j++) {
          const int64_t expert_id = expert_ids[i * k + j];
          if (config_.should_skip_expert(expert_id)) continue;
          __m512 weight = _mm512_set1_ps(weights[i * k + j]);
          __m512 down_output0, down_output1;
          avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(this->m_local_down_output_ptr_[expert_id] +
                                                               this->m_local_pos_[i][j] * config_.hidden_size + e),
                                    &down_output0, &down_output1);
          x0 = _mm512_fmadd_ps(down_output0, weight, x0);
          x1 = _mm512_fmadd_ps(down_output1, weight, x1);
        }
        auto f32out = reinterpret_cast<__m512*>(reinterpret_cast<float*>(output) +
                                                static_cast<size_t>(i) * config_.hidden_size + e);
        f32out[0] = x0;
        f32out[1] = x1;
      }
    });
    profile.mark(profile.merge_us);
    if (profile.enabled) {
      fprintf(stderr,
              "[KT_K2_SFT_FWD_PROFILE] layer=%d tp_part=%d qlen=%d k=%d active=%d save=%d lora=%d cache_top=%d "
              "fast_path=0 route_us=%lld setup_us=%lld copy_input_us=%lld q_input_us=%lld "
              "gate_up_base_us=%lld gate_up_lora_us=%lld save_gate_up_us=%lld act_us=%lld "
              "save_intermediate_us=%lld q_intermediate_us=%lld down_base_us=%lld down_lora_us=%lld "
              "save_down_us=%lld merge_us=%lld total_us=%lld\n",
              config_.layer_idx, tp_part_idx, qlen, k, activated_expert, save_for_backward ? 1 : 0,
              has_any_lora() ? 1 : 0, cache_stack_top_, profile.route_us, profile.setup_us, profile.copy_input_us,
              profile.q_input_us, profile.gate_up_base_us, profile.gate_up_lora_us, profile.save_gate_up_us,
              profile.act_us, profile.save_intermediate_us, profile.q_intermediate_us, profile.down_base_us,
              profile.down_lora_us, profile.save_down_us, profile.merge_us, profile.total_us());
      fflush(stderr);
    }
    trace_forward_step("done", qlen, k, activated_expert, save_for_backward);
    if (cache_ptr != nullptr) {
      cache_ptr->valid = true;
      cache_guard.commit();
    }
  }

  void set_weight_pointers_for_forward(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  void set_k2_packed_weight_scale_pointers(void* gate_proj, void* up_proj, void* down_proj, void* gate_scale,
                                           void* up_scale, void* down_scale) noexcept {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
    config_.gate_scale = gate_scale;
    config_.up_scale = up_scale;
    config_.down_scale = down_scale;
  }

  void set_physical_to_logical_map(const void* map) { config_.physical_to_logical_map = const_cast<void*>(map); }

  void load_weights() {
    Base::load_weights();
    k2_packed_weights_loaded_ = true;
  }

  void backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b,
                void* grad_weights, int full_intermediate_size = 0, float* fp32_grad_down_lora_b = nullptr,
                float* fp32_grad_gate_lora_a = nullptr, float* fp32_grad_up_lora_a = nullptr,
                void* grad_gate_proj = nullptr, void* grad_up_proj = nullptr, void* grad_down_proj = nullptr,
                bool accumulate_optimizer_grads = false, float optimizer_grad_scale = 1.0f) {
    (void)grad_gate_lora_a;
    (void)grad_up_lora_a;
    (void)grad_down_lora_b;
    (void)accumulate_optimizer_grads;
    (void)optimizer_grad_scale;
    if (grad_gate_proj != nullptr || grad_up_proj != nullptr || grad_down_proj != nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT does not support base-weight gradients");
    }
    throw std::runtime_error("K2 RAWINT4 SFT backward requires the TP workspace-v2 path");
  }

  bool backward_workspace_v2_enabled() const { return backward_workspace_v2_eligible(); }

  bool backward_partials_overwrite_enabled() const { return backward_workspace_v2_eligible(); }

  size_t backward_workspace_bytes(int qlen) const {
    return backward_workspace_v2_bytes(qlen, sft_config_.num_experts_per_tok);
  }

  void backward_with_workspace(
      const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
      void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b,
      void* grad_weights, int full_intermediate_size, float* fp32_grad_down_lora_b,
      float* fp32_grad_gate_lora_a, float* fp32_grad_up_lora_a, void* workspace, size_t workspace_bytes,
      bool accumulate_optimizer_grads, float optimizer_grad_scale) {
    SFTProfileScope total_scope(profiler_, SFTProfileStage::BwdTotal);
    (void)grad_gate_lora_a;
    (void)grad_up_lora_a;
    (void)grad_down_lora_b;
    run_tp_packed_backward(grad_output, grad_input, grad_gate_lora_b, grad_up_lora_b, grad_down_lora_a,
                           grad_weights, full_intermediate_size, fp32_grad_down_lora_b, fp32_grad_gate_lora_a,
                           fp32_grad_up_lora_a, workspace, workspace_bytes, accumulate_optimizer_grads,
                           optimizer_grad_scale);
  }

  int get_cache_qlen() const {
    if (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid) {
      return cache_stack_[cache_stack_top_ - 1].qlen_cache;
    }
    return 0;
  }

  int get_cache_activated_expert_count() const {
    return (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
               ? cache_stack_[cache_stack_top_ - 1].activated_expert_cache
               : 0;
  }

  const int* get_cache_expert_id_map() const {
    return (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
               ? cache_stack_[cache_stack_top_ - 1].m_expert_id_map_cache.data()
               : nullptr;
  }

  void prepare_bwd(void*, void*, void*) {
    throw std::runtime_error("K2 RAWINT4 SFT does not use BF16 backward shadows");
  }

  void load_backward_weights_from_projs() {
    throw std::runtime_error("K2 RAWINT4 SFT does not load BF16 backward shadows");
  }

  void save_backward_weights(const std::filesystem::path& path) {
    throw std::runtime_error("K2 RAWINT4 SFT backward weight save is not implemented yet");
  }

  void prepare_backward_bb_for_async() {
    throw std::runtime_error("K2 RAWINT4 SFT async backward repack is not implemented yet");
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_K2_MOE_H
