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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2-moe.hpp"
#include "la/avx_kernels.hpp"

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

  ggml_bf16_t* gate_bwd_shadow_ = nullptr;
  ggml_bf16_t* up_bwd_shadow_ = nullptr;
  ggml_bf16_t* down_bwd_shadow_ = nullptr;
  std::vector<ggml_bf16_t> gate_bwd_shadow_storage_;
  std::vector<ggml_bf16_t> up_bwd_shadow_storage_;
  std::vector<ggml_bf16_t> down_bwd_shadow_storage_;
  bool bwd_shadow_weights_prepared_ = false;
  bool k2_packed_weights_loaded_ = false;

  struct K2ForwardCache {
    ggml_bf16_t* input_cache = nullptr;
    ggml_bf16_t* gate_output_cache = nullptr;
    ggml_bf16_t* up_output_cache = nullptr;
    ggml_bf16_t* intermediate_cache = nullptr;
    ggml_bf16_t* down_output_cache = nullptr;
    float* down_lora_u_cache = nullptr;

    std::vector<ggml_bf16_t> input_storage;
    std::vector<ggml_bf16_t> gate_output_storage;
    std::vector<ggml_bf16_t> up_output_storage;
    std::vector<ggml_bf16_t> intermediate_storage;
    std::vector<ggml_bf16_t> down_output_storage;
    std::vector<float> down_lora_u_storage;

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

  void validate_k2_kgroup_contract() const {
    const auto& quant_config = config_.quant_config;
    if (quant_config.bits != 4 || quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error(
          "K2 RAWINT4 SFT requires signed int4 KGroup weights with group_size=32 and no zero point");
    }
    if (config_.hidden_size <= 0 || config_.intermediate_size <= 0 || config_.expert_num <= 0) {
      throw std::runtime_error("K2 RAWINT4 SFT received invalid MoE dimensions");
    }
  }

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

  void ensure_cache_buffers(K2ForwardCache& cache, int qlen, int k, int total_tokens) {
    const size_t input_elems = static_cast<size_t>(qlen) * config_.hidden_size;
    const size_t inter_elems = static_cast<size_t>(total_tokens) * config_.intermediate_size;
    const size_t down_elems = static_cast<size_t>(total_tokens) * config_.hidden_size;
    const size_t down_lora_u_elems = static_cast<size_t>(total_tokens) * std::max(lora_rank_, 0);

    cache.input_storage.resize(input_elems);
    cache.gate_output_storage.resize(inter_elems);
    cache.up_output_storage.resize(inter_elems);
    cache.intermediate_storage.resize(inter_elems);
    cache.down_output_storage.resize(down_elems);
    cache.down_lora_u_storage.resize(down_lora_u_elems);

    cache.input_cache = cache.input_storage.data();
    cache.gate_output_cache = cache.gate_output_storage.data();
    cache.up_output_cache = cache.up_output_storage.data();
    cache.intermediate_cache = cache.intermediate_storage.data();
    cache.down_output_cache = cache.down_output_storage.data();
    cache.down_lora_u_cache = cache.down_lora_u_storage.empty() ? nullptr : cache.down_lora_u_storage.data();

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

  void compute_lora_gate_up(int activated_expert) {
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
        [this, hidden, inter_size, rank, scale, nth](int task_id) {
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

  void save_to_cache(K2ForwardCache& cache, int qlen, int k, const int64_t* expert_ids, const float* weights,
                     int activated_expert, const void* input) {
    cache.qlen_cache = qlen;
    cache.k_cache = k;
    cache.activated_expert_cache = activated_expert;

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

    cache.valid = true;
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

  size_t gate_up_bwd_shadow_elems() const {
    return static_cast<size_t>(config_.expert_num) * config_.intermediate_size * config_.hidden_size;
  }

  size_t down_bwd_shadow_elems() const {
    return static_cast<size_t>(config_.expert_num) * config_.hidden_size * config_.intermediate_size;
  }

  void assign_bwd_shadow_storage_pointers() {
    gate_bwd_shadow_ = gate_bwd_shadow_storage_.empty() ? nullptr : gate_bwd_shadow_storage_.data();
    up_bwd_shadow_ = up_bwd_shadow_storage_.empty() ? nullptr : up_bwd_shadow_storage_.data();
    down_bwd_shadow_ = down_bwd_shadow_storage_.empty() ? nullptr : down_bwd_shadow_storage_.data();
  }

  void copy_bwd_shadow_weights(const void* gate, const void* up, const void* down) {
    if (gate == nullptr || up == nullptr || down == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow weights require gate/up/down pointers to be all non-null");
    }

    const size_t gate_up_elems = gate_up_bwd_shadow_elems();
    const size_t down_elems = down_bwd_shadow_elems();
    gate_bwd_shadow_storage_.resize(gate_up_elems);
    up_bwd_shadow_storage_.resize(gate_up_elems);
    down_bwd_shadow_storage_.resize(down_elems);

    std::memcpy(gate_bwd_shadow_storage_.data(), gate, gate_up_elems * sizeof(ggml_bf16_t));
    std::memcpy(up_bwd_shadow_storage_.data(), up, gate_up_elems * sizeof(ggml_bf16_t));
    std::memcpy(down_bwd_shadow_storage_.data(), down, down_elems * sizeof(ggml_bf16_t));
    assign_bwd_shadow_storage_pointers();
    bwd_shadow_weights_prepared_ = true;
  }

  void ensure_bwd_shadow_ready() const {
    if (!bwd_shadow_weights_prepared_ || gate_bwd_shadow_ == nullptr || up_bwd_shadow_ == nullptr ||
        down_bwd_shadow_ == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow weights are not prepared");
    }
  }

  static const char* packed_backward_retired_message() {
    return "K2 RAWINT4 SFT packed backward is not implemented yet; BF16 shadow path is retired";
  }

  struct TP1BackwardLayout {
    std::vector<size_t> expert_base;
    std::vector<int> expert_task_index;
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
    return layout;
  }

  static void write_bf16_vector(void* dst, const std::vector<float>& src) {
    if (dst == nullptr) return;
    auto* out = reinterpret_cast<ggml_bf16_t*>(dst);
    for (size_t i = 0; i < src.size(); i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
  }

  static std::vector<ggml_bf16_t> bf16_vector_from_fp32(const std::vector<float>& src) {
    std::vector<ggml_bf16_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
    return out;
  }

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

  void compute_tp1_down_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout, const void* grad_output,
                                 void* grad_down, std::vector<float>* grad_down_fp32_out, void* grad_intermediate,
                                 std::vector<float>* grad_inter_fp32_out, void* grad_down_lora_a,
                                 void* grad_down_lora_b) const {
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

    write_bf16_vector(grad_down, grad_down_fp32);

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

    for (int expert_task = 0; expert_task < cache.activated_expert_cache; expert_task++) {
      const int expert_idx = cache.m_expert_id_map_cache[expert_task];
      const int num_tokens = cache.m_local_num_cache[expert_idx];
      const size_t row_base = layout.expert_base[expert_idx];

      for (int local_t = 0; local_t < num_tokens; local_t++) {
        const size_t row = row_base + static_cast<size_t>(local_t);
        const float* grad_down_row = grad_down_fp32.data() + row * hidden;

        if (need_grad_intermediate) {
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          const auto* down_packed = reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b);
          const float* down_scales = this->down_bb_[expert_idx]->d;

          for (int h = 0; h < hidden; h++) {
            const float g = grad_down_row[h];
            if (g == 0.0f) continue;
            for (int i = 0; i < inter_size; i++) {
              grad_inter_row[i] += g * load_kgroup_weight_f32(down_packed, down_scales, h, i, inter_size);
            }
          }
        }

        if (!need_down_lora_path) continue;

        std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
        const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;
        for (int h = 0; h < hidden; h++) {
          const float g = grad_down_row[h];
          if (g == 0.0f) continue;
          const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * rank;
          for (int r = 0; r < rank; r++) {
            grad_times_b[r] += g * GGML_BF16_TO_FP32(down_b_row[r]);
          }
        }

        const ggml_bf16_t* expert_down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * rank * inter_size;
        if (need_grad_intermediate) {
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          for (int r = 0; r < rank; r++) {
            const float gu = grad_times_b[r] * lora_scaling_;
            const ggml_bf16_t* down_a_row = expert_down_a + static_cast<size_t>(r) * inter_size;
            for (int i = 0; i < inter_size; i++) {
              grad_inter_row[i] += gu * GGML_BF16_TO_FP32(down_a_row[i]);
            }
          }
        }

        if (grad_down_lora_a != nullptr) {
          const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
          float* grad_a = grad_down_a_fp32.data() + static_cast<size_t>(expert_idx) * rank * inter_size;
          for (int r = 0; r < rank; r++) {
            const float gu = grad_times_b[r] * lora_scaling_;
            float* grad_a_row = grad_a + static_cast<size_t>(r) * inter_size;
            for (int i = 0; i < inter_size; i++) {
              grad_a_row[i] += gu * GGML_BF16_TO_FP32(intermediate_row[i]);
            }
          }
        }

        if (grad_down_lora_b != nullptr) {
          const float* down_u_row = cache.down_lora_u_cache + row * rank;
          float* grad_b = grad_down_b_fp32.data() + static_cast<size_t>(expert_idx) * hidden * rank;
          for (int h = 0; h < hidden; h++) {
            const float g = grad_down_row[h] * lora_scaling_;
            if (g == 0.0f) continue;
            float* grad_b_row = grad_b + static_cast<size_t>(h) * rank;
            for (int r = 0; r < rank; r++) {
              grad_b_row[r] += g * down_u_row[r];
            }
          }
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

  void compute_tp1_gate_up_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout,
                                    const ggml_bf16_t* grad_gate, const ggml_bf16_t* grad_up, void* grad_input,
                                    void* grad_gate_lora_a, void* grad_gate_lora_b, void* grad_up_lora_a,
                                    void* grad_up_lora_b) const {
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

    auto backward_one_projection = [&](int expert_idx, int token_idx, const ggml_bf16_t* grad_row,
                                       const uint8_t* packed_weight, const float* scales, const ggml_bf16_t* lora_a,
                                       const ggml_bf16_t* lora_b, float* grad_lora_a, float* grad_lora_b) {
      float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
      const ggml_bf16_t* input_row = cache.input_cache + static_cast<size_t>(token_idx) * hidden;

      if (grad_input != nullptr) {
        for (int i = 0; i < inter_size; i++) {
          const float g = GGML_BF16_TO_FP32(grad_row[i]);
          if (g == 0.0f) continue;
          for (int h = 0; h < hidden; h++) {
            grad_input_row[h] += g * load_kgroup_weight_f32(packed_weight, scales, i, h, hidden);
          }
        }
      }

      if (!use_gate_up_lora) return;

      std::vector<float> lora_u(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);

      for (int r = 0; r < rank; r++) {
        const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
        float acc = 0.0f;
        for (int h = 0; h < hidden; h++) {
          acc += GGML_BF16_TO_FP32(input_row[h]) * GGML_BF16_TO_FP32(a_row[h]);
        }
        lora_u[r] = acc;
      }

      for (int i = 0; i < inter_size; i++) {
        const float g = GGML_BF16_TO_FP32(grad_row[i]);
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
    };

    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int expert_idx = static_cast<int>(expert_id);
        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = layout.expert_base[static_cast<size_t>(expert_idx)] + static_cast<size_t>(local_pos);

        backward_one_projection(
            expert_idx, token_idx, grad_gate + row * inter_size,
            reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b), this->gate_bb_[expert_idx]->d,
            use_gate_up_lora ? gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? gate_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            grad_gate_lora_a != nullptr ? grad_gate_a_fp32.data() : nullptr,
            grad_gate_lora_b != nullptr ? grad_gate_b_fp32.data() : nullptr);
        backward_one_projection(
            expert_idx, token_idx, grad_up + row * inter_size,
            reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
            use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            grad_up_lora_a != nullptr ? grad_up_a_fp32.data() : nullptr,
            grad_up_lora_b != nullptr ? grad_up_b_fp32.data() : nullptr);
      }
    }

    write_bf16_vector(grad_input, grad_input_fp32);
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
    bool should_pop_cache = true;

    try {
      compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);

      std::vector<float> grad_inter_fp32;
      compute_tp1_down_backward(cache, layout, grad_output, nullptr, nullptr, nullptr, &grad_inter_fp32,
                                grad_down_lora_a, grad_down_lora_b);

      std::vector<ggml_bf16_t> grad_inter_bf16 = bf16_vector_from_fp32(grad_inter_fp32);
      std::vector<ggml_bf16_t> grad_gate_bf16(layout.total_tokens * static_cast<size_t>(config_.intermediate_size));
      std::vector<ggml_bf16_t> grad_up_bf16(layout.total_tokens * static_cast<size_t>(config_.intermediate_size));

      compute_tp1_activation_backward(cache, layout, grad_inter_bf16.data(), grad_gate_bf16.data(),
                                      grad_up_bf16.data());
      compute_tp1_gate_up_backward(cache, layout, grad_gate_bf16.data(), grad_up_bf16.data(), grad_input,
                                   grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);

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
  static constexpr bool kUsesKGroupPackedBaseWeights = true;
  static constexpr bool kHasInt4PackedBackward = false;
  static constexpr bool kSupportsForwardCache = true;
  static constexpr bool kSupportsBackward = kHasInt4PackedBackward;
  static constexpr bool kSupportsTPReferenceBackward = false;
  static constexpr bool kSupportsTP1DirectBackward = true;

  using typename Base::input_t;
  using typename Base::output_t;

  AMX_K2_SFT_MOE_TP() = default;

  AMX_K2_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    validate_k2_kgroup_contract();
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
    if (qlen > config_.max_len) {
      throw std::runtime_error("K2 RAWINT4 SFT qlen exceeds max_len");
    }

    // K2 fast path: with no LoRA and no backward cache, delegate to the native packed KGroup forward.
    if (!save_for_backward && !has_any_lora()) {
      Base::forward(qlen, k, expert_ids, weights, input, output);
      return;
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Step 1: Expert routing (reuse base class logic)
    // K2 factors the equivalent routing loop into route_tokens(); it also uses should_skip_expert()
    // so invalid experts and GPU-resident experts follow the shared GeneralMOEConfig skip rule.
    int activated_expert = route_tokens(qlen, k, expert_ids);

    // Step 2: Buffer pool allocation (reuse base class logic)
    // K2 keeps the same per-expert buffer layout, but the allocation math is factored into a helper.
    setup_expert_buffers();

    // Step 3: Copy input to expert buffers
    copy_inputs_to_expert_buffers(qlen, k, expert_ids, input);

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
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = this->m_expert_id_map_[task_id];
      this->gate_up_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx], this->m_local_input_ptr_[expert_idx], 0,
                                              1);
    });

    // Step 5: Gate + Up GEMM (base projection)
    // K2's do_gate_up_gemm dispatches packed int4 KGroup GEMM instead of the generic AMX BufferB path.
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

    // Step 5.5: Gate + Up LoRA (AVX512 BF16 - no BufferB conversion needed)
    compute_lora_gate_up(activated_expert);

    K2ForwardCache* cache_ptr = nullptr;
    if (save_for_backward) {
      // Save gate/up outputs before activation (for backward).
      // Checkpoint recompute overwrites the latest valid cache instead of pushing a duplicate.
      K2ForwardCache& cache = (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
                                  ? cache_stack_[cache_stack_top_ - 1]
                                  : push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert, input);
      cache_ptr = &cache;
    }

    // Step 6: Activation (silu(gate) * up)
    this->apply_activation(activated_expert, nth, qlen);

    if (save_for_backward && cache_ptr != nullptr) {
      // Save intermediate AFTER activation for backward_down.
      save_intermediate_to_cache(*cache_ptr, activated_expert);
    }

    // Step 7: Quantize intermediate for down projection
    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = this->m_expert_id_map_[task_id];
      this->down_ba_[expert_idx]->from_mat(this->m_local_num_[expert_idx], this->m_local_gate_output_ptr_[expert_idx],
                                           0, 1);
    });

    // Step 8: Down GEMM
    // K2's do_down_gemm dispatches packed int4 KGroup GEMM.
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

    // Step 8.5: Down LoRA (AVX512 BF16 - no BufferB conversion needed)
    compute_lora_down(activated_expert, cache_ptr);

    if (save_for_backward && cache_ptr != nullptr) {
      // Save down_output for grad_weights computation.
      save_down_output_to_cache(*cache_ptr, activated_expert);
    }

    // Step 9: Weighted merge
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
  }

  void set_weight_pointers_for_forward(void* gate_proj, void* up_proj, void* down_proj) {
    config_.gate_proj = gate_proj;
    config_.up_proj = up_proj;
    config_.down_proj = down_proj;
  }

  void set_k2_packed_weight_scale_pointers(void* gate_proj, void* up_proj, void* down_proj, void* gate_scale,
                                           void* up_scale, void* down_scale) {
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
                float* fp32_grad_gate_lora_a = nullptr, float* fp32_grad_up_lora_a = nullptr) {
    if constexpr (!kHasInt4PackedBackward) {
      throw std::runtime_error(packed_backward_retired_message());
    }
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT backward requires grad_output and grad_input");
    }

    const K2ForwardCache& cache = latest_cache();
    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int full_inter = full_intermediate_size > 0 ? full_intermediate_size : inter_size;
    const int rank = lora_rank_;

    if (full_inter < inter_size) {
      throw std::runtime_error("K2 RAWINT4 SFT backward full_intermediate_size is smaller than TP local size");
    }

    ensure_bwd_shadow_ready();

    debug_backward_sample(grad_output, nullptr, grad_weights);

    std::vector<size_t> expert_base(static_cast<size_t>(config_.expert_num), 0);
    std::vector<int> expert_task_index(static_cast<size_t>(config_.expert_num), -1);
    size_t total_tokens = 0;
    for (int task = 0; task < cache.activated_expert_cache; task++) {
      const int expert_idx = cache.m_expert_id_map_cache[task];
      expert_base[expert_idx] = total_tokens;
      expert_task_index[expert_idx] = task;
      total_tokens += static_cast<size_t>(cache.m_local_num_cache[expert_idx]);
    }

    auto zero_lora_b_slice = [&](void* ptr) {
      if (ptr == nullptr || rank <= 0) return;
      auto* out = reinterpret_cast<ggml_bf16_t*>(ptr);
      for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
        for (int i = 0; i < inter_size; i++) {
          std::memset(out + (static_cast<size_t>(expert_idx) * full_inter + i) * rank, 0,
                      static_cast<size_t>(rank) * sizeof(ggml_bf16_t));
        }
      }
    };
    auto zero_down_a_slice = [&]() {
      if (grad_down_lora_a == nullptr || rank <= 0) return;
      auto* out = reinterpret_cast<ggml_bf16_t*>(grad_down_lora_a);
      for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
        for (int r = 0; r < rank; r++) {
          std::memset(out + (static_cast<size_t>(expert_idx) * rank + r) * full_inter, 0,
                      static_cast<size_t>(inter_size) * sizeof(ggml_bf16_t));
        }
      }
    };
    zero_lora_b_slice(grad_gate_lora_b);
    zero_lora_b_slice(grad_up_lora_b);
    zero_down_a_slice();

    auto* grad_out = reinterpret_cast<const ggml_bf16_t*>(grad_output);
    std::vector<float> grad_down_fp32(total_tokens * hidden, 0.0f);
    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      const ggml_bf16_t* token_grad = grad_out + static_cast<size_t>(token_idx) * hidden;
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
        const float route_weight = cache.weights_cache[static_cast<size_t>(token_idx) * k + route_idx];
        float* grad_down_row = grad_down_fp32.data() + row * hidden;
        for (int h = 0; h < hidden; h++) {
          grad_down_row[h] += GGML_BF16_TO_FP32(token_grad[h]) * route_weight;
        }
      }
    }

    std::vector<float> grad_inter_fp32(total_tokens * inter_size, 0.0f);
    const bool use_down_lora = rank > 0 && has_down_lora();
    if ((grad_down_lora_a != nullptr || fp32_grad_down_lora_b != nullptr) && !use_down_lora) {
      throw std::runtime_error("K2 RAWINT4 SFT backward requires down LoRA weights for down LoRA grads");
    }

    for (int task = 0; task < cache.activated_expert_cache; task++) {
      const int expert_idx = cache.m_expert_id_map_cache[task];
      const int num_tokens = cache.m_local_num_cache[expert_idx];
      const size_t row_base = expert_base[expert_idx];

      for (int local_t = 0; local_t < num_tokens; local_t++) {
        const size_t row = row_base + static_cast<size_t>(local_t);
        const float* grad_down_row = grad_down_fp32.data() + row * hidden;
        float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
        const ggml_bf16_t* down_shadow = down_bwd_shadow_ + static_cast<size_t>(expert_idx) * hidden * inter_size;

        for (int h = 0; h < hidden; h++) {
          const float g = grad_down_row[h];
          if (g == 0.0f) continue;
          const ggml_bf16_t* down_row = down_shadow + static_cast<size_t>(h) * inter_size;
          for (int i = 0; i < inter_size; i++) {
            grad_inter_row[i] += g * GGML_BF16_TO_FP32(down_row[i]);
          }
        }

        if (!use_down_lora) continue;

        std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
        const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;
        for (int h = 0; h < hidden; h++) {
          const float g = grad_down_row[h];
          if (g == 0.0f) continue;
          const ggml_bf16_t* down_b_row = expert_down_b + static_cast<size_t>(h) * rank;
          for (int r = 0; r < rank; r++) {
            grad_times_b[r] += g * GGML_BF16_TO_FP32(down_b_row[r]);
          }
        }

        const ggml_bf16_t* expert_down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * rank * inter_size;
        for (int r = 0; r < rank; r++) {
          const float gu = grad_times_b[r] * lora_scaling_;
          const ggml_bf16_t* down_a_row = expert_down_a + static_cast<size_t>(r) * inter_size;
          for (int i = 0; i < inter_size; i++) {
            grad_inter_row[i] += gu * GGML_BF16_TO_FP32(down_a_row[i]);
          }
        }

        if (grad_down_lora_a != nullptr) {
          auto* out_down_a = reinterpret_cast<ggml_bf16_t*>(grad_down_lora_a);
          const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
          for (int r = 0; r < rank; r++) {
            const float gu = grad_times_b[r] * lora_scaling_;
            ggml_bf16_t* out_row = out_down_a + (static_cast<size_t>(expert_idx) * rank + r) * full_inter;
            for (int i = 0; i < inter_size; i++) {
              const float old_v = GGML_BF16_TO_FP32(out_row[i]);
              const float add_v = gu * GGML_BF16_TO_FP32(intermediate_row[i]);
              out_row[i] = GGML_FP32_TO_BF16(old_v + add_v);
            }
          }
        }

        if (fp32_grad_down_lora_b != nullptr) {
          const float* down_u_row = cache.down_lora_u_cache + row * rank;
          float* grad_b = fp32_grad_down_lora_b + static_cast<size_t>(task) * hidden * rank;
          for (int h = 0; h < hidden; h++) {
            const float g = grad_down_row[h] * lora_scaling_;
            if (g == 0.0f) continue;
            float* grad_b_row = grad_b + static_cast<size_t>(h) * rank;
            for (int r = 0; r < rank; r++) {
              grad_b_row[r] += g * down_u_row[r];
            }
          }
        }
      }
    }

    std::vector<float> grad_gate_fp32(total_tokens * inter_size, 0.0f);
    std::vector<float> grad_up_fp32(total_tokens * inter_size, 0.0f);
    for (size_t idx = 0; idx < total_tokens * inter_size; idx++) {
      const float grad_inter = grad_inter_fp32[idx];
      const float gate = GGML_BF16_TO_FP32(cache.gate_output_cache[idx]);
      const float up = GGML_BF16_TO_FP32(cache.up_output_cache[idx]);
      const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
      const float silu = gate * sigmoid;
      const float silu_grad = sigmoid * (1.0f + gate * (1.0f - sigmoid));
      grad_gate_fp32[idx] = grad_inter * up * silu_grad;
      grad_up_fp32[idx] = grad_inter * silu;
    }

    const bool use_gate_up_lora = rank > 0 && has_gate_up_lora();
    if ((grad_gate_lora_b != nullptr || grad_up_lora_b != nullptr || fp32_grad_gate_lora_a != nullptr ||
         fp32_grad_up_lora_a != nullptr) &&
        !use_gate_up_lora) {
      throw std::runtime_error("K2 RAWINT4 SFT backward requires gate/up LoRA weights for gate/up LoRA grads");
    }

    std::vector<float> grad_input_fp32(static_cast<size_t>(qlen) * hidden, 0.0f);
    auto backward_one_projection = [&](int task, int expert_idx, int token_idx, const float* grad_row,
                                       const ggml_bf16_t* shadow_weight, const ggml_bf16_t* lora_a,
                                       const ggml_bf16_t* lora_b, float* fp32_grad_lora_a, void* grad_lora_b) {
      float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
      const ggml_bf16_t* input_row = cache.input_cache + static_cast<size_t>(token_idx) * hidden;

      for (int i = 0; i < inter_size; i++) {
        const float g = grad_row[i];
        if (g == 0.0f) continue;
        const ggml_bf16_t* weight_row = shadow_weight + static_cast<size_t>(i) * hidden;
        for (int h = 0; h < hidden; h++) {
          grad_input_row[h] += g * GGML_BF16_TO_FP32(weight_row[h]);
        }
      }

      if (!use_gate_up_lora) return;

      std::vector<float> lora_u(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
      for (int r = 0; r < rank; r++) {
        const ggml_bf16_t* a_row = lora_a + static_cast<size_t>(r) * hidden;
        float acc = 0.0f;
        for (int h = 0; h < hidden; h++) {
          acc += GGML_BF16_TO_FP32(input_row[h]) * GGML_BF16_TO_FP32(a_row[h]);
        }
        lora_u[r] = acc;
      }

      auto* out_lora_b = reinterpret_cast<ggml_bf16_t*>(grad_lora_b);
      for (int i = 0; i < inter_size; i++) {
        const float g = grad_row[i];
        if (g == 0.0f) continue;
        const ggml_bf16_t* b_row = lora_b + static_cast<size_t>(i) * rank;
        if (out_lora_b != nullptr) {
          ggml_bf16_t* out_b_row = out_lora_b + (static_cast<size_t>(expert_idx) * full_inter + i) * rank;
          for (int r = 0; r < rank; r++) {
            const float old_v = GGML_BF16_TO_FP32(out_b_row[r]);
            out_b_row[r] = GGML_FP32_TO_BF16(old_v + g * lora_u[r] * lora_scaling_);
          }
        }
        for (int r = 0; r < rank; r++) {
          grad_times_b[r] += g * GGML_BF16_TO_FP32(b_row[r]);
        }
      }

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
    };

    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      for (int route_idx = 0; route_idx < k; route_idx++) {
        const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
        if (config_.should_skip_expert(expert_id)) continue;

        const int expert_idx = static_cast<int>(expert_id);
        const int task = expert_task_index[expert_idx];
        const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
        const size_t row = expert_base[static_cast<size_t>(expert_idx)] + static_cast<size_t>(local_pos);

        backward_one_projection(
            task, expert_idx, token_idx, grad_gate_fp32.data() + row * inter_size,
            gate_bwd_shadow_ + static_cast<size_t>(expert_idx) * inter_size * hidden,
            use_gate_up_lora ? gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? gate_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            fp32_grad_gate_lora_a, grad_gate_lora_b);
        backward_one_projection(
            task, expert_idx, token_idx, grad_up_fp32.data() + row * inter_size,
            up_bwd_shadow_ + static_cast<size_t>(expert_idx) * inter_size * hidden,
            use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            fp32_grad_up_lora_a, grad_up_lora_b);
      }
    }

    auto* out_grad_input = reinterpret_cast<ggml_bf16_t*>(grad_input);
    for (size_t i = 0; i < grad_input_fp32.size(); i++) {
      out_grad_input[i] = GGML_FP32_TO_BF16(grad_input_fp32[i]);
    }

    pop_latest_cache();
  }

  void backward_tp1_direct(const void* grad_output, void* grad_input, void* grad_gate_lora_a, void* grad_gate_lora_b,
                           void* grad_up_lora_a, void* grad_up_lora_b, void* grad_down_lora_a, void* grad_down_lora_b,
                           void* grad_weights) {
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 backward requires grad_output and grad_input");
    }

    run_tp1_packed_backward(grad_output, grad_input, grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b,
                            grad_down_lora_a, grad_down_lora_b, grad_weights);
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

  std::tuple<int, int, int, std::vector<int>, std::vector<int>> debug_cache_summary() const {
    const K2ForwardCache& cache = latest_cache();
    std::vector<int> active_experts(cache.m_expert_id_map_cache.begin(),
                                    cache.m_expert_id_map_cache.begin() + cache.activated_expert_cache);
    return {cache.qlen_cache, cache.k_cache, cache.activated_expert_cache, active_experts, cache.m_local_num_cache};
  }

  void debug_copy_forward_cache(void* input, void* gate, void* up, void* intermediate, void* down,
                                void* down_lora_u) const {
    const K2ForwardCache& cache = latest_cache();
    size_t total_tokens = 0;
    for (int i = 0; i < cache.activated_expert_cache; i++) {
      int expert_idx = cache.m_expert_id_map_cache[i];
      total_tokens += static_cast<size_t>(cache.m_local_num_cache[expert_idx]);
    }

    if (input != nullptr) {
      std::memcpy(input, cache.input_cache,
                  static_cast<size_t>(cache.qlen_cache) * config_.hidden_size * sizeof(ggml_bf16_t));
    }
    if (gate != nullptr) {
      std::memcpy(gate, cache.gate_output_cache, total_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
    }
    if (up != nullptr) {
      std::memcpy(up, cache.up_output_cache, total_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
    }
    if (intermediate != nullptr) {
      std::memcpy(intermediate, cache.intermediate_cache,
                  total_tokens * config_.intermediate_size * sizeof(ggml_bf16_t));
    }
    if (down != nullptr) {
      std::memcpy(down, cache.down_output_cache, total_tokens * config_.hidden_size * sizeof(ggml_bf16_t));
    }
    if (down_lora_u != nullptr && cache.down_lora_u_cache != nullptr && lora_rank_ > 0) {
      std::memcpy(down_lora_u, cache.down_lora_u_cache, total_tokens * lora_rank_ * sizeof(float));
    }
  }

  void debug_remerge_forward_cache(void* output) const {
    if (output == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT forward cache remerge requires output");
    }

    const K2ForwardCache& cache = latest_cache();
    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;

    std::vector<size_t> expert_base(static_cast<size_t>(config_.expert_num), 0);
    size_t cursor = 0;
    for (int i = 0; i < cache.activated_expert_cache; i++) {
      int expert_idx = cache.m_expert_id_map_cache[i];
      expert_base[expert_idx] = cursor;
      cursor += static_cast<size_t>(cache.m_local_num_cache[expert_idx]);
    }

    float* out = reinterpret_cast<float*>(output);
    for (int token_idx = 0; token_idx < qlen; token_idx++) {
      for (int h = 0; h < hidden; h += 32) {
        __m512 x0 = _mm512_setzero_ps();
        __m512 x1 = _mm512_setzero_ps();
        for (int route_idx = 0; route_idx < k; route_idx++) {
          const int64_t expert_id = cache.expert_ids_cache[static_cast<size_t>(token_idx) * k + route_idx];
          if (config_.should_skip_expert(expert_id)) continue;
          const int local_pos = cache.m_local_pos_cache[token_idx][route_idx];
          const size_t row = expert_base[static_cast<size_t>(expert_id)] + static_cast<size_t>(local_pos);
          const ggml_bf16_t* down_row = cache.down_output_cache + row * hidden + h;
          const __m512 weight = _mm512_set1_ps(cache.weights_cache[static_cast<size_t>(token_idx) * k + route_idx]);
          __m512 down_output0, down_output1;
          avx512_32xbf16_to_32xfp32(reinterpret_cast<__m512i*>(const_cast<ggml_bf16_t*>(down_row)), &down_output0,
                                    &down_output1);
          x0 = _mm512_fmadd_ps(down_output0, weight, x0);
          x1 = _mm512_fmadd_ps(down_output1, weight, x1);
        }
        auto f32out = reinterpret_cast<__m512*>(out + static_cast<size_t>(token_idx) * hidden + h);
        f32out[0] = x0;
        f32out[1] = x1;
      }
    }
  }

  // Archived/debug-only BF16 shadow helpers. KGroup training backward reads packed weights directly.
  void prepare_bwd(void* gate, void* up, void* down) { copy_bwd_shadow_weights(gate, up, down); }

  void load_backward_weights_from_projs() {
    if (config_.gate_bwd_shadow_projs.empty() || config_.up_bwd_shadow_projs.empty() ||
        config_.down_bwd_shadow_projs.empty()) {
      throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow projs are not configured");
    }
    if (tp_part_idx >= static_cast<int>(config_.gate_bwd_shadow_projs.size()) ||
        tp_part_idx >= static_cast<int>(config_.up_bwd_shadow_projs.size()) ||
        tp_part_idx >= static_cast<int>(config_.down_bwd_shadow_projs.size())) {
      throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow projs missing current TP part");
    }
    if (static_cast<int>(config_.gate_bwd_shadow_projs[tp_part_idx].size()) < config_.expert_num ||
        static_cast<int>(config_.up_bwd_shadow_projs[tp_part_idx].size()) < config_.expert_num ||
        static_cast<int>(config_.down_bwd_shadow_projs[tp_part_idx].size()) < config_.expert_num) {
      throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow projs missing expert entries");
    }

    const size_t gate_up_per_expert = static_cast<size_t>(config_.intermediate_size) * config_.hidden_size;
    const size_t down_per_expert = static_cast<size_t>(config_.hidden_size) * config_.intermediate_size;
    gate_bwd_shadow_storage_.resize(gate_up_bwd_shadow_elems());
    up_bwd_shadow_storage_.resize(gate_up_bwd_shadow_elems());
    down_bwd_shadow_storage_.resize(down_bwd_shadow_elems());

    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map, gate_up_per_expert, down_per_expert](int expert_idx) {
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          const void* gate_src = config_.gate_bwd_shadow_projs[tp_part_idx][logical_expert_id];
          const void* up_src = config_.up_bwd_shadow_projs[tp_part_idx][logical_expert_id];
          const void* down_src = config_.down_bwd_shadow_projs[tp_part_idx][logical_expert_id];
          if (gate_src == nullptr || up_src == nullptr || down_src == nullptr) {
            throw std::runtime_error("K2 RAWINT4 SFT BF16 shadow proj entry is null");
          }
          std::memcpy(gate_bwd_shadow_storage_.data() + static_cast<size_t>(expert_idx) * gate_up_per_expert, gate_src,
                      gate_up_per_expert * sizeof(ggml_bf16_t));
          std::memcpy(up_bwd_shadow_storage_.data() + static_cast<size_t>(expert_idx) * gate_up_per_expert, up_src,
                      gate_up_per_expert * sizeof(ggml_bf16_t));
          std::memcpy(down_bwd_shadow_storage_.data() + static_cast<size_t>(expert_idx) * down_per_expert, down_src,
                      down_per_expert * sizeof(ggml_bf16_t));
        },
        nullptr);

    assign_bwd_shadow_storage_pointers();
    bwd_shadow_weights_prepared_ = true;
  }

  std::tuple<bool, int, int, int> debug_bwd_shadow_summary() const {
    return {bwd_shadow_weights_prepared_, config_.expert_num, config_.hidden_size, config_.intermediate_size};
  }

  void debug_copy_bwd_shadow(void* gate, void* up, void* down) const {
    ensure_bwd_shadow_ready();
    const size_t gate_up_elems = gate_up_bwd_shadow_elems();
    const size_t down_elems = down_bwd_shadow_elems();
    if (gate != nullptr) {
      std::memcpy(gate, gate_bwd_shadow_, gate_up_elems * sizeof(ggml_bf16_t));
    }
    if (up != nullptr) {
      std::memcpy(up, up_bwd_shadow_, gate_up_elems * sizeof(ggml_bf16_t));
    }
    if (down != nullptr) {
      std::memcpy(down, down_bwd_shadow_, down_elems * sizeof(ggml_bf16_t));
    }
  }

  std::tuple<bool, int, int, int, int, size_t, size_t, size_t, size_t> debug_packed_weight_summary() const {
    validate_k2_kgroup_contract();
    return {packed_weight_buffers_ready(),
            config_.expert_num,
            config_.hidden_size,
            config_.intermediate_size,
            config_.quant_config.group_size,
            gate_up_packed_bytes_per_expert(),
            down_packed_bytes_per_expert(),
            gate_up_scale_elems_per_expert(),
            down_scale_elems_per_expert()};
  }

  void debug_copy_packed_weights(void* gate, void* up, void* down, void* gate_scale, void* up_scale,
                                 void* down_scale) const {
    ensure_packed_weight_buffers_ready();

    const size_t gate_up_bytes = gate_up_packed_bytes_per_expert();
    const size_t down_bytes = down_packed_bytes_per_expert();
    const size_t gate_up_scale_elems = gate_up_scale_elems_per_expert();
    const size_t down_scale_elems = down_scale_elems_per_expert();

    auto* gate_out = reinterpret_cast<uint8_t*>(gate);
    auto* up_out = reinterpret_cast<uint8_t*>(up);
    auto* down_out = reinterpret_cast<uint8_t*>(down);
    auto* gate_scale_out = reinterpret_cast<float*>(gate_scale);
    auto* up_scale_out = reinterpret_cast<float*>(up_scale);
    auto* down_scale_out = reinterpret_cast<float*>(down_scale);

    for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
      if (gate_out != nullptr) {
        std::memcpy(gate_out + static_cast<size_t>(expert_idx) * gate_up_bytes, this->gate_bb_[expert_idx]->b,
                    gate_up_bytes);
      }
      if (up_out != nullptr) {
        std::memcpy(up_out + static_cast<size_t>(expert_idx) * gate_up_bytes, this->up_bb_[expert_idx]->b,
                    gate_up_bytes);
      }
      if (down_out != nullptr) {
        std::memcpy(down_out + static_cast<size_t>(expert_idx) * down_bytes, this->down_bb_[expert_idx]->b, down_bytes);
      }
      if (gate_scale_out != nullptr) {
        std::memcpy(gate_scale_out + static_cast<size_t>(expert_idx) * gate_up_scale_elems,
                    this->gate_bb_[expert_idx]->d, gate_up_scale_elems * sizeof(float));
      }
      if (up_scale_out != nullptr) {
        std::memcpy(up_scale_out + static_cast<size_t>(expert_idx) * gate_up_scale_elems, this->up_bb_[expert_idx]->d,
                    gate_up_scale_elems * sizeof(float));
      }
      if (down_scale_out != nullptr) {
        std::memcpy(down_scale_out + static_cast<size_t>(expert_idx) * down_scale_elems, this->down_bb_[expert_idx]->d,
                    down_scale_elems * sizeof(float));
      }
    }
  }

  void debug_backward_sample(const void* grad_output, void* grad_input, void* grad_weights) const {
    const K2ForwardCache& cache = latest_cache();
    if (grad_input != nullptr) {
      std::memset(grad_input, 0, static_cast<size_t>(cache.qlen_cache) * config_.hidden_size * sizeof(ggml_bf16_t));
    }
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);
  }

  void debug_backward_down_sample(const void* grad_output, void* grad_down, void* grad_intermediate,
                                  void* grad_down_lora_a, void* grad_down_lora_b) const {
    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    compute_tp1_down_backward(cache, layout, grad_output, grad_down, nullptr, grad_intermediate, nullptr,
                              grad_down_lora_a, grad_down_lora_b);
  }

  void debug_backward_activation_sample(const void* grad_output, void* grad_intermediate, void* grad_gate,
                                        void* grad_up) const {
    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    std::vector<float> grad_inter_fp32;
    compute_tp1_down_backward(cache, layout, grad_output, nullptr, nullptr, grad_intermediate, &grad_inter_fp32,
                              nullptr, nullptr);
    std::vector<ggml_bf16_t> grad_inter_bf16 = bf16_vector_from_fp32(grad_inter_fp32);
    compute_tp1_activation_backward(cache, layout, grad_inter_bf16.data(), grad_gate, grad_up);
  }

  void debug_backward_gate_up_sample(const void* grad_output, void* grad_input, void* grad_gate_lora_a,
                                     void* grad_gate_lora_b, void* grad_up_lora_a, void* grad_up_lora_b) const {
    const K2ForwardCache& cache = latest_cache();
    const int inter_size = config_.intermediate_size;
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);

    std::vector<float> grad_inter_fp32;
    compute_tp1_down_backward(cache, layout, grad_output, nullptr, nullptr, nullptr, &grad_inter_fp32, nullptr,
                              nullptr);
    std::vector<ggml_bf16_t> grad_inter_bf16 = bf16_vector_from_fp32(grad_inter_fp32);
    std::vector<ggml_bf16_t> grad_gate_storage(layout.total_tokens * static_cast<size_t>(inter_size));
    std::vector<ggml_bf16_t> grad_up_storage(layout.total_tokens * static_cast<size_t>(inter_size));

    compute_tp1_activation_backward(cache, layout, grad_inter_bf16.data(), grad_gate_storage.data(),
                                    grad_up_storage.data());
    compute_tp1_gate_up_backward(cache, layout, grad_gate_storage.data(), grad_up_storage.data(), grad_input,
                                 grad_gate_lora_a, grad_gate_lora_b, grad_up_lora_a, grad_up_lora_b);
  }

  void save_backward_weights(const std::filesystem::path& path) {
    throw std::runtime_error("K2 RAWINT4 SFT backward weight save is not implemented yet");
  }

  void prepare_backward_bb_for_async() {
    throw std::runtime_error("K2 RAWINT4 SFT async backward repack is not implemented yet");
  }
};

#endif  // CPUINFER_OPERATOR_AMX_SFT_K2_MOE_H
