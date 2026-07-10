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
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "k2-moe.hpp"
#include "la/avx_kernels.hpp"

namespace kt_sft_nvtx {
inline bool enabled() {
  static const bool value = []() {
    const char* env = std::getenv("KT_SFT_NVTX_CPP");
    if (env == nullptr || env[0] == '\0') env = std::getenv("KT_SFT_NVTX");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  return value;
}

using PushFn = int (*)(const char*);
using PopFn = int (*)();

inline void* load_symbol(const char* name) {
  void* symbol = dlsym(RTLD_DEFAULT, name);
  if (symbol != nullptr) return symbol;

  void* handle = dlopen("libnvToolsExt.so.1", RTLD_LAZY | RTLD_GLOBAL);
  if (handle == nullptr) handle = dlopen("libnvToolsExt.so", RTLD_LAZY | RTLD_GLOBAL);
  if (handle == nullptr) return nullptr;
  return dlsym(handle, name);
}

inline PushFn push_fn() {
  static PushFn fn = reinterpret_cast<PushFn>(load_symbol("nvtxRangePushA"));
  return fn;
}

inline PopFn pop_fn() {
  static PopFn fn = reinterpret_cast<PopFn>(load_symbol("nvtxRangePop"));
  return fn;
}

inline int process_rank() {
  static const int value = []() {
    const char* env_names[] = {"RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"};
    for (const char* name : env_names) {
      const char* env = std::getenv(name);
      if (env != nullptr && env[0] != '\0') return std::atoi(env);
    }
    return -1;
  }();
  return value;
}

inline std::string label(const char* range_name, int layer, int tp) {
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "%s layer=%d tp=%d rank=%d", range_name, layer, tp, process_rank());
  return std::string(buffer);
}

struct Range {
  bool active = false;
  explicit Range(const char* name) {
    PushFn push = push_fn();
    if (enabled() && push != nullptr) {
      push(name);
      active = true;
    }
  }
  ~Range() {
    PopFn pop = pop_fn();
    if (active && pop != nullptr) pop();
  }
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;
};
}  // namespace kt_sft_nvtx

#define KT_SFT_NVTX_CONCAT_INNER(a, b) a##b
#define KT_SFT_NVTX_CONCAT(a, b) KT_SFT_NVTX_CONCAT_INNER(a, b)
#define KT_SFT_NVTX_RANGE(name) ::kt_sft_nvtx::Range KT_SFT_NVTX_CONCAT(_kt_sft_nvtx_range_, __LINE__)(name)

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

  std::string nvtx_label(const char* range_name) const {
    return kt_sft_nvtx::label(range_name, sft_config_.layer_idx, tp_part_idx);
  }
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
          {
            KT_SFT_NVTX_RANGE(do_up ? "up_lora_matmul" : "gate_lora_matmul");
            avx::lora_bf16_matmul_t4r4(this->m_local_input_ptr_[expert_idx] + static_cast<size_t>(t_start) * hidden,
                                       expert_lora_a, local_intermediate.data(), local_num_tokens, hidden, rank);
            avx::lora_fp32_bf16_fused_add_transposed(local_intermediate.data(), expert_lora_b_t,
                                                     output + static_cast<size_t>(t_start) * inter_size, local_num_tokens,
                                                     rank, inter_size, scale);
          }
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
          {
            KT_SFT_NVTX_RANGE("down_lora_matmul");
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
          }
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
    long long down_us = 0;
    long long down_lora_grads_us = 0;
    long long down_route_us = 0;
    long long down_write_us = 0;
    long long down_base_us = 0;
    long long down_lora_bprop_us = 0;
    long long down_lora_a_us = 0;
    long long down_lora_b_us = 0;
    long long activation_us = 0;
    long long gate_up_us = 0;
    long long gate_up_base_us = 0;
    long long gate_up_lora_u_us = 0;
    long long gate_up_lora_b_us = 0;
    long long gate_up_lora_b_write_us = 0;
    long long gate_up_lora_a_input_us = 0;
    long long gate_up_write_us = 0;

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

  static std::vector<ggml_bf16_t> bf16_vector_from_fp32(const std::vector<float>& src) {
    std::vector<ggml_bf16_t> out(src.size());
    for (size_t i = 0; i < src.size(); i++) {
      out[i] = GGML_FP32_TO_BF16(src[i]);
    }
    return out;
  }

  // Reverse of the forward weighted merge:
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

  // Down projection backward in expert-packed token order.
  //
  // K2 base backward reads packed signed-int4 KGroup rows plus BF16 group
  // scales directly. It does not use BF16 shadow weights or AMX transposed
  // BufferB objects.
  //
  // Data flow:
  //   1. Scatter token-order grad_output to per-expert grad_down, multiplying
  //      by saved router weights.
  //   2. Use packed down weights to compute grad_intermediate.
  //   3. If down LoRA is active, add the LoRA contribution to grad_intermediate.
  //   4. If requested, compute down LoRA A/B gradients.
  //
  // In the normal TP autograd path, down LoRA B is accumulated in a sparse
  // FP32 side buffer outside this helper. Dense BF16 grad_down_lora_b is mainly
  // used by TP=1 direct/debug.
  void compute_tp1_down_backward(const K2ForwardCache& cache, const TP1BackwardLayout& layout, const void* grad_output,
                                 void* grad_down, std::vector<float>* grad_down_fp32_out, void* grad_intermediate,
                                 std::vector<float>* grad_inter_fp32_out, void* grad_down_lora_a,
                                 void* grad_down_lora_b, TP1BackwardProfile* profile = nullptr) const {
    if (grad_output == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 down backward requires grad_output");
    }

    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int rank = lora_rank_;
    const std::string bwd_down_base_label = nvtx_label("bwd_down_base_matmul");
    const std::string bwd_down_lora_bprop_label = nvtx_label("bwd_down_lora_bprop");
    const std::string bwd_down_lora_a_input_label = nvtx_label("bwd_down_lora_a_input");
    const std::string bwd_down_lora_grads_label = nvtx_label("bwd_down_lora_grads");
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

    const bool parallel_base = need_grad_intermediate && layout.total_tokens >= 10 && config_.pool != nullptr;
    if (parallel_base) {
      section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      std::vector<int> row_to_expert(layout.total_tokens, -1);
      for (int expert_task = 0; expert_task < cache.activated_expert_cache; expert_task++) {
        const int expert_idx = cache.m_expert_id_map_cache[expert_task];
        const int num_tokens = cache.m_local_num_cache[expert_idx];
        const size_t row_base = layout.expert_base[expert_idx];
        for (int local_t = 0; local_t < num_tokens; local_t++) {
          row_to_expert[row_base + static_cast<size_t>(local_t)] = expert_idx;
        }
      }

      auto pool = config_.pool->get_subpool(tp_part_idx);
      pool->do_work_stealing_job(
          static_cast<int>(layout.total_tokens), nullptr,
          [&](int row_idx) {
            const int expert_idx = row_to_expert[static_cast<size_t>(row_idx)];
            if (expert_idx < 0) return;
            const float* grad_down_row = grad_down_fp32.data() + static_cast<size_t>(row_idx) * hidden;
            float* grad_inter_row = grad_inter_fp32.data() + static_cast<size_t>(row_idx) * inter_size;
            const auto* down_packed = reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b);
            const float* down_scales = this->down_bb_[expert_idx]->d;
            {
              KT_SFT_NVTX_RANGE(bwd_down_base_label.c_str());
              add_scaled_packed_rows_f32(down_packed, down_scales, grad_down_row, hidden, inter_size, grad_inter_row,
                                         true);
            }
          },
          nullptr);
      if (profile != nullptr) profile->add_since(section_start, profile->down_base_us);
    }

    for (int expert_task = 0; expert_task < cache.activated_expert_cache; expert_task++) {
      const int expert_idx = cache.m_expert_id_map_cache[expert_task];
      const int num_tokens = cache.m_local_num_cache[expert_idx];
      const size_t row_base = layout.expert_base[expert_idx];
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);

      for (int local_t = 0; local_t < num_tokens; local_t++) {
        const size_t row = row_base + static_cast<size_t>(local_t);
        const float* grad_down_row = grad_down_fp32.data() + row * hidden;

        if (need_grad_intermediate && !parallel_base) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          const auto* down_packed = reinterpret_cast<const uint8_t*>(this->down_bb_[expert_idx]->b);
          const float* down_scales = this->down_bb_[expert_idx]->d;
          {
            KT_SFT_NVTX_RANGE(bwd_down_base_label.c_str());
            add_scaled_packed_rows_f32(down_packed, down_scales, grad_down_row, hidden, inter_size, grad_inter_row,
                                       layout.total_tokens >= 10 || short_base_fastpath_enabled());
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_base_us);
        }

        if (!need_down_lora_path) continue;

        std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);
        const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;
        section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
        {
          KT_SFT_NVTX_RANGE(bwd_down_lora_bprop_label.c_str());
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
        }
        if (profile != nullptr) profile->add_since(section_start, profile->down_lora_bprop_us);

        const ggml_bf16_t* expert_down_a = down_lora_a_ + static_cast<size_t>(expert_idx) * rank * inter_size;
        if (need_grad_intermediate) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          float* grad_inter_row = grad_inter_fp32.data() + row * inter_size;
          {
            KT_SFT_NVTX_RANGE(bwd_down_lora_a_input_label.c_str());
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
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_lora_a_us);
        }

        if (grad_down_lora_a != nullptr) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
          float* grad_a = grad_down_a_fp32.data() + static_cast<size_t>(expert_idx) * rank * inter_size;
          {
            KT_SFT_NVTX_RANGE(bwd_down_lora_grads_label.c_str());
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
          }
          if (profile != nullptr) profile->add_since(section_start, profile->down_lora_a_us);
        }

        if (grad_down_lora_b != nullptr) {
          section_start = profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
          const float* down_u_row = cache.down_lora_u_cache + row * rank;
          float* grad_b = grad_down_b_fp32.data() + static_cast<size_t>(expert_idx) * hidden * rank;
          {
            KT_SFT_NVTX_RANGE(bwd_down_lora_grads_label.c_str());
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

  // Gate/up projection backward in expert-packed token order.
  //
  // Base path reads packed gate/up KGroup rows and accumulates token-order
  // grad_input. LoRA path computes:
  //   u = input @ A^T
  //   grad_B += grad_projection^T @ u
  //   grad_times_B = grad_projection @ B
  //   grad_A += grad_times_B^T @ input
  //   grad_input += grad_times_B @ A
  //
  // This helper writes dense BF16 LoRA gradients for TP=1 direct/debug. The
  // normal TP autograd path below has an inlined variant to write sparse FP32
  // LoRA A / down B buffers.
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
                                       const ggml_bf16_t* lora_b, float* grad_lora_a, float* grad_lora_b, bool do_base,
                                       bool do_lora, bool profile_base_inside) {
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
      if (rank == 2) {
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
            grad_gate_lora_b != nullptr ? grad_gate_b_fp32.data() : nullptr, false, true, false);
        backward_one_projection(
            expert_idx, token_idx, grad_up + row * inter_size,
            reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
            use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
            use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
            grad_up_lora_a != nullptr ? grad_up_a_fp32.data() : nullptr,
            grad_up_lora_b != nullptr ? grad_up_b_fp32.data() : nullptr, false, true, false);
      }
    };

    const bool parallel_base = grad_input != nullptr && qlen >= 10 && config_.pool != nullptr;
    if (parallel_base) {
      const auto section_start =
          profile != nullptr ? profile->section_start() : TP1BackwardProfile::disabled_time_point();
      auto pool = config_.pool->get_subpool(tp_part_idx);
      pool->do_work_stealing_job(
          qlen, nullptr, [&](int token_idx) { run_token_routes(token_idx, true, false, false); }, nullptr);
      if (profile != nullptr) profile->add_since(section_start, profile->gate_up_base_us);
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

  // TP=1 direct/debug backward. It exercises the packed base-weight math and
  // returns all LoRA gradients through dense BF16 buffers, which makes tests and
  // debug dumps easier to compare tensor-by-tensor.
  void run_tp1_packed_backward(const void* grad_output, void* grad_input, void* grad_gate_lora_a,
                               void* grad_gate_lora_b, void* grad_up_lora_a, void* grad_up_lora_b,
                               void* grad_down_lora_a, void* grad_down_lora_b, void* grad_weights) {
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP=1 backward requires grad_output and grad_input");
    }

    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
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
                     "[KT_K2_SFT_PROFILE] layer=%d qlen=%d active=%d tokens=%zu grad_weights_us=%lld down_us=%lld "
                     "down_route_us=%lld down_write_us=%lld down_base_us=%lld down_lora_bprop_us=%lld "
                     "down_lora_a_us=%lld down_lora_b_us=%lld activation_us=%lld gate_up_us=%lld "
                     "gate_up_base_us=%lld gate_up_lora_u_us=%lld "
                     "gate_up_lora_b_us=%lld gate_up_lora_b_write_us=%lld "
                     "gate_up_lora_a_input_us=%lld gate_up_write_us=%lld total_us=%lld\n",
                     sft_config_.layer_idx, cache.qlen_cache, cache.activated_expert_cache, layout.total_tokens,
                     profile.grad_weights_us, profile.down_us, profile.down_route_us, profile.down_write_us,
                     profile.down_base_us, profile.down_lora_bprop_us, profile.down_lora_a_us, profile.down_lora_b_us,
                     profile.activation_us, profile.gate_up_us, profile.gate_up_base_us, profile.gate_up_lora_u_us,
                     profile.gate_up_lora_b_us, profile.gate_up_lora_b_write_us, profile.gate_up_lora_a_input_us,
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
                              float* fp32_grad_down_lora_b, float* fp32_grad_gate_lora_a, float* fp32_grad_up_lora_a) {
    if (grad_output == nullptr || grad_input == nullptr) {
      throw std::runtime_error("K2 RAWINT4 SFT TP backward requires grad_output and grad_input");
    }

    ensure_packed_weight_buffers_ready();

    const K2ForwardCache& cache = latest_cache();
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);
    const int qlen = cache.qlen_cache;
    const int k = cache.k_cache;
    const int hidden = config_.hidden_size;
    const int inter_size = config_.intermediate_size;
    const int full_inter = full_intermediate_size > 0 ? full_intermediate_size : inter_size;
    const int rank = lora_rank_;
    const std::string bwd_gate_base_label = nvtx_label("bwd_gate_base_matmul");
    const std::string bwd_up_base_label = nvtx_label("bwd_up_base_matmul");
    const std::string bwd_gate_lora_bprop_label = nvtx_label("bwd_gate_lora_bprop");
    const std::string bwd_up_lora_bprop_label = nvtx_label("bwd_up_lora_bprop");
    const std::string bwd_gate_lora_a_input_label = nvtx_label("bwd_gate_lora_a_input");
    const std::string bwd_up_lora_a_input_label = nvtx_label("bwd_up_lora_a_input");
    const std::string bwd_down_lora_bprop_label = nvtx_label("bwd_down_lora_bprop");
    const std::string bwd_down_lora_a_input_label = nvtx_label("bwd_down_lora_a_input");
    const std::string bwd_down_lora_grads_label = nvtx_label("bwd_down_lora_grads");

    if (full_inter < inter_size) {
      throw std::runtime_error("K2 RAWINT4 SFT TP backward full_intermediate_size is smaller than TP local size");
    }

    bool should_pop_cache = true;
    TP1BackwardProfile profile(tp1_backward_profile_enabled());
    try {
      const bool use_down_lora = rank > 0 && has_down_lora();
      const bool need_down_a = grad_down_lora_a != nullptr;
      const bool need_down_b = fp32_grad_down_lora_b != nullptr;
      if ((need_down_a || need_down_b) && !use_down_lora) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires down LoRA weights");
      }
      if (need_down_b && cache.down_lora_u_cache == nullptr) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires cached down LoRA activations");
      }

      compute_tp1_grad_weights(cache, layout, grad_output, grad_weights);
      profile.mark(profile.grad_weights_us);

      std::vector<float> grad_down_fp32;
      std::vector<float> grad_inter_fp32;
      compute_tp1_down_backward(cache, layout, grad_output, nullptr,
                                (need_down_a || need_down_b) ? &grad_down_fp32 : nullptr, nullptr, &grad_inter_fp32,
                                nullptr, nullptr, &profile);
      profile.mark(profile.down_us);

      if (use_down_lora && (need_down_a || need_down_b)) {
        KT_SFT_NVTX_RANGE(bwd_down_lora_grads_label.c_str());
        std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);

        for (int task = 0; task < cache.activated_expert_cache; task++) {
          const int expert_idx = cache.m_expert_id_map_cache[task];
          const int num_tokens = cache.m_local_num_cache[expert_idx];
          const size_t row_base = layout.expert_base[expert_idx];
          const ggml_bf16_t* expert_down_b = down_lora_b_ + static_cast<size_t>(expert_idx) * hidden * rank;

          for (int local_t = 0; local_t < num_tokens; local_t++) {
            const size_t row = row_base + static_cast<size_t>(local_t);
            const float* grad_down_row = grad_down_fp32.data() + row * hidden;

            std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);
            auto section_start = profile.section_start();
            {
              KT_SFT_NVTX_RANGE(bwd_down_lora_bprop_label.c_str());
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
            }
            profile.add_since(section_start, profile.down_lora_bprop_us);

            if (need_down_a) {
              section_start = profile.section_start();
              auto* out_down_a = reinterpret_cast<ggml_bf16_t*>(grad_down_lora_a);
              const ggml_bf16_t* intermediate_row = cache.intermediate_cache + row * inter_size;
              {
                KT_SFT_NVTX_RANGE(bwd_down_lora_a_input_label.c_str());
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
              profile.add_since(section_start, profile.down_lora_a_us);
            }

            if (need_down_b) {
              section_start = profile.section_start();
              const float* down_u_row = cache.down_lora_u_cache + row * rank;
              float* grad_b = fp32_grad_down_lora_b + static_cast<size_t>(task) * hidden * rank;
              {
                KT_SFT_NVTX_RANGE(bwd_down_lora_grads_label.c_str());
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
              }
              profile.add_since(section_start, profile.down_lora_b_us);
            }
          }
        }
      }
      profile.mark(profile.down_lora_grads_us);

      std::vector<float> grad_gate_fp32;
      std::vector<float> grad_up_fp32;
      compute_tp1_activation_backward_fp32(cache, layout, grad_inter_fp32.data(), grad_gate_fp32, grad_up_fp32);
      profile.mark(profile.activation_us);

      const bool use_gate_up_lora = rank > 0 && has_gate_up_lora();
      const bool need_gate_up_lora = grad_gate_lora_b != nullptr || grad_up_lora_b != nullptr ||
                                     fp32_grad_gate_lora_a != nullptr || fp32_grad_up_lora_a != nullptr;
      if (need_gate_up_lora && !use_gate_up_lora) {
        throw std::runtime_error("K2 RAWINT4 SFT TP backward requires gate/up LoRA weights");
      }

      std::vector<float> grad_input_fp32(static_cast<size_t>(qlen) * hidden, 0.0f);
      std::vector<float> lora_u(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_times_b(static_cast<size_t>(rank), 0.0f);
      std::vector<float> grad_gate_b_fp32;
      std::vector<float> grad_up_b_fp32;
      const bool use_sparse_lora_b = sparse_lora_b_accum_enabled();
      if (use_sparse_lora_b && use_gate_up_lora && grad_gate_lora_b != nullptr) {
        grad_gate_b_fp32.assign(static_cast<size_t>(cache.activated_expert_cache) * inter_size * rank, 0.0f);
      }
      if (use_sparse_lora_b && use_gate_up_lora && grad_up_lora_b != nullptr) {
        grad_up_b_fp32.assign(static_cast<size_t>(cache.activated_expert_cache) * inter_size * rank, 0.0f);
      }

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
                                         void* grad_lora_b, bool do_base, bool do_lora, bool profile_base_inside,
                                         const char* base_label, const char* lora_bprop_label,
                                         const char* lora_a_input_label) {
        float* grad_input_row = grad_input_fp32.data() + static_cast<size_t>(token_idx) * hidden;
        const ggml_bf16_t* input_row = cache.input_cache + static_cast<size_t>(token_idx) * hidden;

        if (do_base) {
          auto section_start =
              profile_base_inside ? profile.section_start() : TP1BackwardProfile::disabled_time_point();
          {
            KT_SFT_NVTX_RANGE(base_label);
            for (int i = 0; i < inter_size; i++) {
              const float g = grad_row[i];
              if (g == 0.0f) continue;
              add_scaled_packed_row_f32(packed_weight, scales, i, hidden, g, grad_input_row);
            }
          }
          if (profile_base_inside) profile.add_since(section_start, profile.gate_up_base_us);
        }

        if (!do_lora || !use_gate_up_lora) return;

        std::fill(lora_u.begin(), lora_u.end(), 0.0f);
        std::fill(grad_times_b.begin(), grad_times_b.end(), 0.0f);

        auto section_start = profile.section_start();
        if (rank == 2) {
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

        auto* out_lora_b = reinterpret_cast<ggml_bf16_t*>(grad_lora_b);
        section_start = profile.section_start();
        {
          KT_SFT_NVTX_RANGE(lora_bprop_label);
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
        }
        profile.add_since(section_start, profile.gate_up_lora_b_us);

        section_start = profile.section_start();
        {
          KT_SFT_NVTX_RANGE(lora_a_input_label);
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
        }
        profile.add_since(section_start, profile.gate_up_lora_a_input_us);
      };

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
            {
              KT_SFT_NVTX_RANGE(bwd_gate_base_label.c_str());
              KT_SFT_NVTX_RANGE(bwd_up_base_label.c_str());
              add_gate_up_base(expert_idx, token_idx, row);
            }
            if (profile_base_inside) profile.add_since(section_start, profile.gate_up_base_us);
          }
          if (!do_lora) continue;

          backward_one_projection(
              task, expert_idx, token_idx, grad_gate_fp32.data() + row * inter_size,
              reinterpret_cast<const uint8_t*>(this->gate_bb_[expert_idx]->b), this->gate_bb_[expert_idx]->d,
              use_gate_up_lora ? gate_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
              use_gate_up_lora ? gate_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
              fp32_grad_gate_lora_a, grad_gate_b_fp32.empty() ? nullptr : grad_gate_b_fp32.data(),
              grad_gate_b_fp32.empty() ? grad_gate_lora_b : nullptr, false, true, false,
              bwd_gate_base_label.c_str(), bwd_gate_lora_bprop_label.c_str(),
              bwd_gate_lora_a_input_label.c_str());
          backward_one_projection(
              task, expert_idx, token_idx, grad_up_fp32.data() + row * inter_size,
              reinterpret_cast<const uint8_t*>(this->up_bb_[expert_idx]->b), this->up_bb_[expert_idx]->d,
              use_gate_up_lora ? up_lora_a_ + static_cast<size_t>(expert_idx) * rank * hidden : nullptr,
              use_gate_up_lora ? up_lora_b_ + static_cast<size_t>(expert_idx) * inter_size * rank : nullptr,
              fp32_grad_up_lora_a, grad_up_b_fp32.empty() ? nullptr : grad_up_b_fp32.data(),
              grad_up_b_fp32.empty() ? grad_up_lora_b : nullptr, false, true, false, bwd_up_base_label.c_str(),
              bwd_up_lora_bprop_label.c_str(), bwd_up_lora_a_input_label.c_str());
        }
      };

      const bool parallel_base = qlen >= 10 && config_.pool != nullptr;
      if (parallel_base) {
        auto section_start = profile.section_start();
        auto pool = config_.pool->get_subpool(tp_part_idx);
        pool->do_work_stealing_job(
            qlen, nullptr, [&](int token_idx) { run_token_routes(token_idx, true, false, false); }, nullptr);
        profile.add_since(section_start, profile.gate_up_base_us);
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

      auto section_start = profile.section_start();
      write_bf16_vector(grad_input, grad_input_fp32);
      profile.add_since(section_start, profile.gate_up_write_us);
      profile.mark(profile.gate_up_us);
      if (profile.enabled) {
        std::fprintf(stderr,
                     "[KT_K2_SFT_PROFILE] layer=%d tp_part=%d qlen=%d active=%d tokens=%zu grad_weights_us=%lld "
                     "down_us=%lld down_lora_grads_us=%lld down_route_us=%lld down_write_us=%lld "
                     "down_base_us=%lld down_lora_bprop_us=%lld down_lora_a_us=%lld down_lora_b_us=%lld "
                     "activation_us=%lld gate_up_us=%lld "
                     "gate_up_base_us=%lld gate_up_lora_u_us=%lld gate_up_lora_b_us=%lld "
                     "gate_up_lora_b_write_us=%lld gate_up_lora_a_input_us=%lld "
                     "gate_up_write_us=%lld total_us=%lld\n",
                     sft_config_.layer_idx, tp_part_idx, cache.qlen_cache, cache.activated_expert_cache,
                     layout.total_tokens, profile.grad_weights_us, profile.down_us, profile.down_lora_grads_us,
                     profile.down_route_us, profile.down_write_us, profile.down_base_us, profile.down_lora_bprop_us,
                     profile.down_lora_a_us, profile.down_lora_b_us, profile.activation_us, profile.gate_up_us,
                     profile.gate_up_base_us, profile.gate_up_lora_u_us, profile.gate_up_lora_b_us,
                     profile.gate_up_lora_b_write_us, profile.gate_up_lora_a_input_us, profile.gate_up_write_us,
                     profile.total_us());
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

 public:
  static constexpr bool kSkipLoRA = SkipLoRA;
  static constexpr bool kUsesKGroupPackedBaseWeights = true;
  static constexpr bool kHasInt4PackedBackward = true;
  static constexpr bool kSupportsForwardCache = true;
  static constexpr bool kSupportsBackward = true;
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
            {
              KT_SFT_NVTX_RANGE(do_up ? "up_base_matmul" : "gate_base_matmul");
              this->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
              if (do_up) {
                this->up_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx], this->m_local_up_output_ptr_[expert_idx],
                                                 ith, nth);
              } else {
                this->gate_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                                   this->m_local_gate_output_ptr_[expert_idx], ith, nth);
              }
            }
          },
          nullptr);
    }
    profile.mark(profile.gate_up_base_us);

    // Step 5.5: Gate + Up LoRA (AVX512 BF16 - no BufferB conversion needed)
    trace_forward_step("step5_5_gate_up_lora", qlen, k, activated_expert, save_for_backward);
    compute_lora_gate_up(activated_expert);
    profile.mark(profile.gate_up_lora_us);

    K2ForwardCache* cache_ptr = nullptr;
    if (save_for_backward) {
      // Save gate/up outputs before activation (for backward).
      // Checkpoint recompute overwrites the latest valid cache instead of pushing a duplicate.
      trace_forward_step("step5_6_save_gate_up_cache", qlen, k, activated_expert, save_for_backward);
      K2ForwardCache& cache = (cache_stack_top_ > 0 && cache_stack_[cache_stack_top_ - 1].valid)
                                  ? cache_stack_[cache_stack_top_ - 1]
                                  : push_cache();
      save_to_cache(cache, qlen, k, expert_ids, weights, activated_expert, input);
      cache_ptr = &cache;
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
            {
              KT_SFT_NVTX_RANGE("down_base_matmul");
              this->do_down_gemm(expert_idx, ith, nth, qlen);
              this->down_bc_[expert_idx]->to_mat(this->m_local_num_[expert_idx],
                                                 this->m_local_down_output_ptr_[expert_idx], ith, nth);
            }
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
    // Normal TP path uses sparse FP32 side buffers for gate/up LoRA A and down
    // LoRA B. These dense BF16 arguments are kept for the shared binding ABI.
    (void)grad_gate_lora_a;
    (void)grad_up_lora_a;
    (void)grad_down_lora_b;
    run_tp_packed_backward(grad_output, grad_input, grad_gate_lora_b, grad_up_lora_b, grad_down_lora_a, grad_weights,
                           full_intermediate_size, fp32_grad_down_lora_b, fp32_grad_gate_lora_a, fp32_grad_up_lora_a);
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
    const TP1BackwardLayout layout = make_tp1_backward_layout(cache);

    std::vector<float> grad_inter_fp32;
    compute_tp1_down_backward(cache, layout, grad_output, nullptr, nullptr, nullptr, &grad_inter_fp32, nullptr,
                              nullptr);
    std::vector<float> grad_gate_storage;
    std::vector<float> grad_up_storage;

    compute_tp1_activation_backward_fp32(cache, layout, grad_inter_fp32.data(), grad_gate_storage, grad_up_storage);
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
