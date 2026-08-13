#ifndef LLAMAFILE_MOE_HPP
#define LLAMAFILE_MOE_HPP
#ifdef FORWARD_TIME_PROFILE
#include <fmt/format.h>
#endif
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../moe-tp.hpp"
#include "conversion.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// ---------------------------------------------------------------------------
// KT_MOE_PHASE_TIMING=1 — Session D Phase-0 测量：decode (forward_one) 三段
// (input量化 / gate+up job / down job) + TP 层 merge 的累计耗时，定位固定开销 F
// 的构成。env 关闭时仅一次 getenv + 分支，零扰动。每 tp 每 4300 次调用
// (=43层×100 token) 打一行均值到 stderr。
// 线程安全性：同一 tp_part_idx 的 forward_one 在 decode 流内串行（逐层），
// 不同 tp 并行但写不同槽位 → 无需原子。
// ---------------------------------------------------------------------------
static inline bool kt_phase_timing_on() {
  static const bool on = std::getenv("KT_MOE_PHASE_TIMING") != nullptr;
  return on;
}
struct KtPhaseAcc {
  uint64_t calls = 0;
  uint64_t quant_ns = 0, gateup_ns = 0, down_ns = 0;
};
inline KtPhaseAcc g_kt_phase_acc[16];

inline void debug_quant(void* input, ggml_type type) {
  std::vector<float> output(ggml_blck_size(type));
  to_float(input, output.data(), ggml_blck_size(type), type);
  for (size_t i = 0; i < 10; i++) {
    printf("%f ", output[i]);
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
// kt_effective_vec_dot_type
//   解决 aarch64 (Kunpeng K920 / Cortex-A76, **no SVE / no i8mm**) 上
//   kt-kernel llamafile sgemm 的 BF16/Q8_0 路径不健全问题：
//
//   * BF16 weight + BF16 input：tinyblas_cpu_sgemm.inc ARM_NEON path 要求
//     `Btype == GGML_TYPE_F32`（line 209: `if (Btype != F32) return NOT_SUPPORTED;`），
//     但 ggml type_traits 对 BF16 给出 `vec_dot_type = BF16`，
//     上层 forward_one/forward_many 默认把 BF16 input 喂给 sgemm
//     → llamafile_sgemm 返回 false → `throw "llamafile not supported"`。
//
//   * 解法：在 aarch64-without-SVE 平台上，把 BF16 weight 的有效 vec_dot_type
//     声明为 F32。input 路径会因此走 to_float(bf16 → fp32) + memcpy(F32→F32 buffer)
//     （`from_float()` 已在 conversion.h 对 F32 short-circuit 成 memcpy），
//     buffer 大小自动按 fp32 (4 bytes/elem) 分配 —— 比原 BF16 (2 bytes/elem)
//     大一倍，足够装 fp32 数据。sgemm 则走 ARM_NEON 已支持的
//     `Atype=BF16, Btype=F32, Ctype=F32` 路径（tinyblas_cpu_sgemm.inc line 125-133）。
//
//   * SVE 机器走原 BF16-BF16 path（vec_dot_type 不改），不打扰原性能优化。
//   * 其他 weight type（Q8_0/Q4_K/…）走原 vec_dot_type，不动；
//     若 Q8_0 NaN 仍存在，单独在 sgemm 内部修复，而非这里。
// ---------------------------------------------------------------------------
static inline ggml_type kt_effective_vec_dot_type(ggml_type weight_type) {
#if defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)
  if (weight_type == GGML_TYPE_BF16) {
    return GGML_TYPE_F32;
  }
#endif
  return ggml_internal_get_type_traits(weight_type).vec_dot_type;
}

class LLAMA_MOE_TP {
 private:
  GeneralMOEConfig config_;
  int tp_part_idx;

  uint8_t* m_local_gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  uint8_t* m_local_up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
  uint8_t* m_local_down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

  float* s_input_fp32_;    // [hidden_size]
  uint8_t* s_gate_input_;  // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* s_up_input_;    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                           // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
  std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
  std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
  std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size *
                                             // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                             // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
  float* s_output_fp32_;                     // [hidden_size]

  std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
  std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size *
                                        // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                        // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  std::vector<uint8_t*>
      m_up_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type)
                    // / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  uint8_t* m_local_gate_input_;        // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
  uint8_t* m_local_up_input_;          // [routed_expert_num * group_max_len * hidden_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
  float* m_local_gate_output_;         // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_up_output_;           // [routed_expert_num * group_max_len * intermediate_size]
  float* m_local_intermediate_fp32_;   // [routed_expert_num * group_max_len * intermediate_size]
  uint8_t* m_local_down_input_;        // [routed_expert_num * group_max_len * intermediate_size *
                                       // ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) /
                                       // ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
  float* m_local_down_output_;         // [routed_expert_num * group_max_len * hidden_size]
  std::vector<float*> m_output_fp32_;  // [group_max_len, hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
  std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
  std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
  std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
  std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
  std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
  std::vector<float*> m_local_down_output_ptr_;        // [expert_num]
 public:
  using input_t = ggml_bf16_t;
  using output_t = float;

  LLAMA_MOE_TP(GeneralMOEConfig config, int tp_part_idx) : config_(config), tp_part_idx(tp_part_idx) {
    MemoryRequest mem_requests;
    mem_requests.append_pointer(&s_input_fp32_, sizeof(float) * config_.hidden_size);
    mem_requests.append_pointer(
        &s_gate_input_, config_.hidden_size *
                            ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
                            ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)));
    mem_requests.append_pointer(
        &s_up_input_, config_.hidden_size *
                          ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
                          ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)));
    s_gate_output_.resize(config_.num_experts_per_tok);
    s_up_output_.resize(config_.num_experts_per_tok);
    s_intermediate_fp32_.resize(config_.num_experts_per_tok);
    s_down_input_.resize(config_.num_experts_per_tok);
    s_down_output_.resize(config_.num_experts_per_tok);
    for (int i = 0; i < config_.num_experts_per_tok; i++) {
      mem_requests.append_pointer(&s_gate_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_up_output_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size);
      mem_requests.append_pointer(
          &s_down_input_[i],
          config_.intermediate_size *
              ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
              ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)));
      mem_requests.append_pointer(&s_down_output_[i], sizeof(float) * config_.hidden_size);
    }
    mem_requests.append_pointer(&s_output_fp32_, sizeof(float) * config_.hidden_size);
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, mem_requests);

    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_input_fp32_[i], sizeof(float) * config_.hidden_size);
      mem_requests.append_pointer(
          &m_gate_input_[i],
          config_.hidden_size *
              ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
              ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)));
      mem_requests.append_pointer(
          &m_up_input_[i], config_.hidden_size *
                               ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
                               ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)));
    }
    mem_requests.append_pointer(
        &m_local_gate_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
            ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
            ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)));
    mem_requests.append_pointer(
        &m_local_up_input_, config_.num_experts_per_tok * config_.group_max_len * config_.hidden_size *
                                ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
                                ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)));
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(float) * config_.num_experts_per_tok *
                                                         config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_intermediate_fp32_, sizeof(float) * config_.num_experts_per_tok *
                                                                 config_.group_max_len * config_.intermediate_size);
    mem_requests.append_pointer(
        &m_local_down_input_,
        config_.num_experts_per_tok * config_.group_max_len * config_.intermediate_size *
            ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
            ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)));
    mem_requests.append_pointer(&m_local_down_output_, sizeof(float) * config_.num_experts_per_tok *
                                                           config_.group_max_len * config_.hidden_size);
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      mem_requests.append_pointer(&m_output_fp32_[i], sizeof(float) * config_.hidden_size);
    }
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
    // shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    auto size = 1ll * config.expert_num * config.intermediate_size * config.hidden_size;
    m_local_up_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type)];

    m_local_gate_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.gate_type) / ggml_blck_size((ggml_type)config.gate_type)];
    m_local_down_proj_ =
        new uint8_t[size * ggml_type_size((ggml_type)config.down_type) / ggml_blck_size((ggml_type)config.down_type)];
  }

  void load_weights(int complete_intermediate_size, int offset) {
    auto& config = config_;
    // printf("gate load weights:");
    // debug_quant(config.gate_proj, (ggml_type)config.gate_type);
    // we need to make sure the blck size is correct for size.
    if (config.intermediate_size % ggml_blck_size((ggml_type)config.down_type) != 0) {
      printf("intermediate_size: %d, down_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.down_type));
      throw std::runtime_error("intermediate_size must be a multiple of gate_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.up_type) != 0) {
      printf("intermediate_size: %d, up_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.up_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of up_type blck size");
    }
    if (config.intermediate_size * config.hidden_size % ggml_blck_size((ggml_type)config.gate_type) != 0) {
      printf("intermediate_size: %d, gate_type blck size: %d\n", config.intermediate_size,
             ggml_blck_size((ggml_type)config.gate_type));
      throw std::runtime_error("intermediate_size * hidden_size must be a multiple of gate_type blck size");
    }
    uint8_t* gate_proj = (uint8_t*)config.gate_proj + offset * config.hidden_size *
                                                          ggml_type_size((ggml_type)config.gate_type) /
                                                          ggml_blck_size((ggml_type)config.gate_type);
    uint8_t* up_proj = (uint8_t*)config.up_proj + offset * config.hidden_size *
                                                      ggml_type_size((ggml_type)config.up_type) /
                                                      ggml_blck_size((ggml_type)config.up_type);
    uint8_t* down_proj = (uint8_t*)config.down_proj + offset * ggml_type_size((ggml_type)config.down_type) /
                                                          ggml_blck_size((ggml_type)config.down_type);

    // Per-expert byte strides. The source tensors are laid out with the FULL
    // intermediate_size (complete_intermediate_size); this TP only owns the
    // [offset, offset+intermediate_size) block — hence the base-pointer offset
    // above (src strides) and the smaller local destination strides below.
    const size_t gate_dst_stride = (size_t)config.intermediate_size * config.hidden_size *
                                   ggml_type_size((ggml_type)config.gate_type) /
                                   ggml_blck_size((ggml_type)config.gate_type);
    const size_t gate_src_stride = (size_t)complete_intermediate_size * config.hidden_size *
                                   ggml_type_size((ggml_type)config.gate_type) /
                                   ggml_blck_size((ggml_type)config.gate_type);
    const size_t up_dst_stride = (size_t)config.intermediate_size * config.hidden_size *
                                 ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type);
    const size_t up_src_stride = (size_t)complete_intermediate_size * config.hidden_size *
                                 ggml_type_size((ggml_type)config.up_type) / ggml_blck_size((ggml_type)config.up_type);
    const size_t down_dst_row = (size_t)config.intermediate_size * ggml_type_size((ggml_type)config.down_type) /
                                ggml_blck_size((ggml_type)config.down_type);
    const size_t down_src_row = (size_t)complete_intermediate_size * ggml_type_size((ggml_type)config.down_type) /
                                ggml_blck_size((ggml_type)config.down_type);
    const size_t down_dst_stride = (size_t)config.hidden_size * down_dst_row;
    const size_t down_src_stride = (size_t)config.hidden_size * down_src_row;

    uint8_t* const local_gate_base = m_local_gate_proj_;
    uint8_t* const local_up_base = m_local_up_proj_;
    uint8_t* const local_down_base = m_local_down_proj_;

    // Copy one expert's gate/up/down into the (disjoint) local buffers. Experts
    // write non-overlapping destination regions and read disjoint source spans,
    // so this is embarrassingly parallel across i.
    auto copy_expert = [&](int i) {
      memcpy(local_gate_base + (size_t)i * gate_dst_stride, gate_proj + (size_t)i * gate_src_stride, gate_dst_stride);
      memcpy(local_up_base + (size_t)i * up_dst_stride, up_proj + (size_t)i * up_src_stride, up_dst_stride);
      uint8_t* ld = local_down_base + (size_t)i * down_dst_stride;
      uint8_t* sd = down_proj + (size_t)i * down_src_stride;
      for (int j = 0; j < config.hidden_size; ++j) {
        memcpy(ld, sd, down_dst_row);
        ld += down_dst_row;
        sd += down_src_row;
      }
    };

    // Parallelize the per-expert reshuffle across this NUMA subpool's worker
    // threads. The legacy serial loop left each TP's load 1-wide (8-wide overall
    // via do_numa_job) on a 192-core box. Mirrors forward()'s
    // get_subpool(tp_part_idx)->do_work_stealing_job nesting inside do_numa_job.
    config_.pool->get_subpool(tp_part_idx)->do_work_stealing_job(config.expert_num,
                                                                 [&](int i) { copy_expert(i); });
  }

  void warm_up() {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                               ggml_blck_size((ggml_type)config_.hidden_type));
    std::vector<float> output(config_.hidden_size);
    for (int i = 0; i < config_.hidden_size; i++) {
      input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, (ggml_type)config_.hidden_type);
    for (int i = 0; i < config_.expert_num; i++) {
      int64_t expert_ids = i;
      float weights = 0;
      forward_one(1, &expert_ids, &weights, input.data(), output.data());
    }
  }

  static float act_fn(float gate, float up, float swiglu_limit) {
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fmaxf(-swiglu_limit, fminf(up, swiglu_limit));
    }
    return gate / (1.0f + expf(-gate)) * up;
  }

  void forward_one(int k, const int64_t* expert_ids, const float* weights, const void* input, float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto t0 = std::chrono::high_resolution_clock::now();
#endif
    const bool kt_pt = kt_phase_timing_on();
    std::chrono::high_resolution_clock::time_point kt_pt0, kt_pt1, kt_pt2;
    if (kt_pt) kt_pt0 = std::chrono::high_resolution_clock::now();
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if ((ggml_type)config_.hidden_type == kt_effective_vec_dot_type((ggml_type)config_.gate_type) &&
        (ggml_type)config_.hidden_type == kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
      gate_input_ptr = up_input_ptr = input;
    } else {
      to_float(input, s_input_fp32_, config_.hidden_size, (ggml_type)config_.hidden_type);
      if (kt_effective_vec_dot_type((ggml_type)config_.gate_type) ==
          kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
        from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                   kt_effective_vec_dot_type((ggml_type)config_.gate_type));
        gate_input_ptr = up_input_ptr = s_gate_input_;
      } else {
        if ((ggml_type)config_.hidden_type !=
            kt_effective_vec_dot_type((ggml_type)config_.gate_type)) {
          from_float(s_input_fp32_, s_gate_input_, config_.hidden_size,
                     kt_effective_vec_dot_type((ggml_type)config_.gate_type));
          gate_input_ptr = s_gate_input_;
        } else {
          gate_input_ptr = input;
        }
        if ((ggml_type)config_.hidden_type != kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
          from_float(s_input_fp32_, s_up_input_, config_.hidden_size,
                     kt_effective_vec_dot_type((ggml_type)config_.up_type));
          up_input_ptr = s_up_input_;
        } else {
          up_input_ptr = input;
        }
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("gate_input: ");
    // debug_quant(const_cast<void *>(gate_input_ptr),
    // kt_effective_vec_dot_type((ggml_type)config_.gate_type));
    // printf("up_input: ");
    // debug_quant(const_cast<void *>(up_input_ptr),
    // kt_effective_vec_dot_type((ggml_type)config_.up_type));
    auto t1 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, convert time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

#endif

    if (kt_pt) kt_pt1 = std::chrono::high_resolution_clock::now();

    int activated_expert = 0;
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      activated_expert++;
    }

    int nth = config_.intermediate_size / config_.m_block;

    // Only process activated (CPU) experts; skip GPU experts entirely to keep buffers aligned.
    if (activated_expert > 0) {
      pool->do_work_stealing_job(
          nth * activated_expert, nullptr,
          [&](int task_id) {
            int act_idx = task_id / nth;
            int64_t expert_id = m_expert_id_map_[act_idx];
            if (expert_id == -1) {
              return;
            }
            int ith = task_id % nth;

            void* gate_proj_ptr =
                (uint8_t*)m_local_gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.m_block) *
                                                   config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                                   ggml_blck_size((ggml_type)config_.gate_type);

            float* gate_output_ptr = s_gate_output_[act_idx] + ith * config_.m_block;
            auto ok = llamafile_sgemm(
                config_.m_block, 1, config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr, config_.m_block, 0,
                1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                kt_effective_vec_dot_type((ggml_type)config_.gate_type), GGML_TYPE_F32,
                GGML_PREC_DEFAULT);
            if (ok == false) [[unlikely]] {
              throw std::runtime_error("llamafile not supported");
            }

            void* up_proj_ptr =
                (uint8_t*)m_local_up_proj_ + (expert_id * config_.intermediate_size + ith * config_.m_block) *
                                                 config_.hidden_size * ggml_type_size((ggml_type)config_.up_type) /
                                                 ggml_blck_size((ggml_type)config_.up_type);

            float* up_output_ptr = s_up_output_[act_idx] + ith * config_.m_block;
            llamafile_sgemm(config_.m_block, 1, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
                            up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
                            config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
                            config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
                            kt_effective_vec_dot_type((ggml_type)config_.up_type), GGML_TYPE_F32,
                            GGML_PREC_DEFAULT);

            for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
              s_intermediate_fp32_[act_idx][i] =
                  act_fn(s_gate_output_[act_idx][i], s_up_output_[act_idx][i], config_.swiglu_limit);
            }
            if (config_.m_block %
                    ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) ==
                0) {
              float* intermediate_fp32_ptr = s_intermediate_fp32_[act_idx] + ith * config_.m_block;
              void* down_input_ptr =
                  s_down_input_[act_idx] +
                  ith * config_.m_block *
                      ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
                      ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type));
              from_float(intermediate_fp32_ptr, down_input_ptr, config_.m_block,
                         kt_effective_vec_dot_type((ggml_type)config_.down_type));
            }
          },
          nullptr);
    }

    if (config_.m_block % ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) !=
        0) {
      for (int i = 0; i < activated_expert; i++) {
        from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size,
                   kt_effective_vec_dot_type((ggml_type)config_.down_type));
      }
    }

#ifdef FORWARD_TIME_PROFILE
    // printf("sinter:");
    // debug_f32(s_intermediate_fp32_[expert_ids[0]]);
    auto t2 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, gate/up time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
#endif
    if (kt_pt) kt_pt2 = std::chrono::high_resolution_clock::now();

    nth = config_.hidden_size / config_.m_block;
    pool->do_work_stealing_job(
        nth, nullptr,
        [&](int task_id) {
          int ith = task_id;
          for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
            output[i] = 0;
          }
          for (int expert_idx = 0; expert_idx < activated_expert; expert_idx++) {
            int64_t expert_id = m_expert_id_map_[expert_idx];
            if (expert_id == -1) {
              continue;
            }

            auto expert_offset = expert_id * config_.hidden_size * config_.intermediate_size;
            auto m_block_offset = ith * config_.m_block * config_.intermediate_size;
            void* down_proj_ptr = (uint8_t*)m_local_down_proj_ + (expert_offset + m_block_offset) *
                                                                     ggml_type_size((ggml_type)config_.down_type) /
                                                                     ggml_blck_size((ggml_type)config_.down_type);

            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.m_block;
            llamafile_sgemm(
                config_.m_block, 1, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                down_proj_ptr, config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type),
                down_output_ptr, config_.m_block, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                kt_effective_vec_dot_type((ggml_type)config_.down_type), GGML_TYPE_F32,
                GGML_PREC_DEFAULT);

            float expert_weight = 0.0f;
            for (int j = 0; j < k; j++) {
              if (expert_ids[j] == expert_id) {
                expert_weight = weights[j];
                break;
              }
            }

            for (int i = ith * config_.m_block; i < (ith + 1) * config_.m_block; i++) {
              output[i] += s_down_output_[expert_idx][i] * expert_weight;
            }
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    auto t3 = std::chrono::high_resolution_clock::now();
    fmt::print("numa_node: {}, down time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    fmt::print("numa_node: {}, total time: {}\n", tp_part_idx,
               std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t0).count());
#endif
    if (kt_pt) {
      auto kt_pt3 = std::chrono::high_resolution_clock::now();
      auto ns = [](auto a, auto b) {
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
      };
      auto& acc = g_kt_phase_acc[tp_part_idx & 15];
      acc.calls++;
      acc.quant_ns += ns(kt_pt0, kt_pt1);
      acc.gateup_ns += ns(kt_pt1, kt_pt2);
      acc.down_ns += ns(kt_pt2, kt_pt3);
      if (acc.calls % 4300 == 0) {
        fprintf(stderr, "[KT_PHASE tp%d] n=%llu avg/layer-call: quant=%.1fus gateup=%.1fus down=%.1fus\n",
                tp_part_idx, (unsigned long long)acc.calls, acc.quant_ns / 1e3 / acc.calls,
                acc.gateup_ns / 1e3 / acc.calls, acc.down_ns / 1e3 / acc.calls);
      }
    }
  }

  void forward_many(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                    float* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_gate_input_ptr_[i] =
          m_local_gate_input_ +
          offset * config_.hidden_size *
              ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
              ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type));
      m_local_up_input_ptr_[i] =
          m_local_up_input_ +
          offset * config_.hidden_size *
              ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
              ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type));
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
      m_local_down_input_ptr_[i] =
          m_local_down_input_ +
          offset * config_.intermediate_size *
              ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
              ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type));
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          const void* gate_input_ptr;
          const void* up_input_ptr;
          if ((ggml_type)config_.hidden_type ==
                  kt_effective_vec_dot_type((ggml_type)config_.gate_type) &&
              (ggml_type)config_.hidden_type ==
                  kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                                  ggml_type_size((ggml_type)config_.hidden_type) /
                                                                  ggml_blck_size((ggml_type)config_.hidden_type);
          } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                           ggml_blck_size((ggml_type)config_.hidden_type),
                     m_input_fp32_[i], config_.hidden_size, (ggml_type)config_.hidden_type);
            if (kt_effective_vec_dot_type((ggml_type)config_.gate_type) ==
                kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
              from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                         kt_effective_vec_dot_type((ggml_type)config_.gate_type));
              gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
              if ((ggml_type)config_.hidden_type !=
                  kt_effective_vec_dot_type((ggml_type)config_.gate_type)) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size,
                           kt_effective_vec_dot_type((ggml_type)config_.gate_type));
                gate_input_ptr = m_gate_input_[i];
              } else {
                gate_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                       ggml_type_size((ggml_type)config_.hidden_type) /
                                                       ggml_blck_size((ggml_type)config_.hidden_type);
              }
              if ((ggml_type)config_.hidden_type !=
                  kt_effective_vec_dot_type((ggml_type)config_.up_type)) {
                from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size,
                           kt_effective_vec_dot_type((ggml_type)config_.up_type));
                up_input_ptr = m_up_input_[i];
              } else {
                up_input_ptr = (uint8_t*)input + i * config_.hidden_size *
                                                     ggml_type_size((ggml_type)config_.hidden_type) /
                                                     ggml_blck_size((ggml_type)config_.hidden_type);
              }
            }
          }
          for (int j = 0; j < k; j++) {
            if (config_.should_skip_expert(expert_ids[i * k + j])) {
              continue;
            }
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
                           ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)),
                   gate_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)) /
                       ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.gate_type)));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] +
                       m_local_pos_[i][j] * config_.hidden_size *
                           ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
                           ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)),
                   up_input_ptr,
                   config_.hidden_size *
                       ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)) /
                       ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.up_type)));
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int m_block = QK_K;
    int nth = config_.intermediate_size / m_block;
    // printf("nth: %d, m_block: %d, activated_expert: %d\n", nth, m_block, activated_expert);
    // printf("config_.hidden_size: %d, config_.intermediate_size: %d\n", config_.hidden_size,
    // config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

          void* gate_proj_ptr =
              (uint8_t*)m_local_gate_proj_ + (expert_idx * config_.intermediate_size + ith * m_block) *
                                                 config_.hidden_size * ggml_type_size((ggml_type)config_.gate_type) /
                                                 ggml_blck_size((ggml_type)config_.gate_type);

          float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * m_block;

          // if (ith == 0) {
          //   printf("matrix size: m:%d, n:%d, k:%d\n", m_block, m_local_num_[expert_idx],
          //          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type));
          // }
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_proj_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_input_ptr,
                          config_.hidden_size / ggml_blck_size((ggml_type)config_.gate_type), gate_output_ptr,
                          config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.gate_type,
                          kt_effective_vec_dot_type((ggml_type)config_.gate_type), GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
          void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

          void* up_proj_ptr = (uint8_t*)m_local_up_proj_ + (expert_idx * config_.intermediate_size + ith * m_block) *
                                                               config_.hidden_size *
                                                               ggml_type_size((ggml_type)config_.up_type) /
                                                               ggml_blck_size((ggml_type)config_.up_type);

          float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(
              m_block, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type),
              up_proj_ptr, config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_input_ptr,
              config_.hidden_size / ggml_blck_size((ggml_type)config_.up_type), up_output_ptr,
              config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.up_type,
              kt_effective_vec_dot_type((ggml_type)config_.up_type), GGML_TYPE_F32, GGML_PREC_DEFAULT);
          for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * m_block; j < (ith + 1) * m_block; j++) {
              m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] =
                  act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j],
                         m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j],
                         config_.swiglu_limit);
            }
            float* intermediate_fp32_ptr =
                m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * m_block;
            void* down_input_ptr =
                m_local_down_input_ptr_[expert_idx] +
                i * config_.intermediate_size *
                    ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
                    ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) +
                ith * m_block *
                    ggml_type_size(kt_effective_vec_dot_type((ggml_type)config_.down_type)) /
                    ggml_blck_size(kt_effective_vec_dot_type((ggml_type)config_.down_type));
            from_float(intermediate_fp32_ptr, down_input_ptr, m_block,
                       kt_effective_vec_dot_type((ggml_type)config_.down_type));
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      up_gate_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    m_block = QK_K;
    nth = config_.hidden_size / m_block;
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [&](int task_id) {
          int64_t expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          void* down_input_ptr = m_local_down_input_ptr_[expert_idx];

          auto expert_offset = expert_idx * config_.hidden_size * config_.intermediate_size;
          auto m_block_offset = ith * m_block * config_.intermediate_size;

          void* down_proj_ptr = (uint8_t*)m_local_down_proj_ + (expert_offset + m_block_offset) *
                                                                   ggml_type_size((ggml_type)config_.down_type) /
                                                                   ggml_blck_size((ggml_type)config_.down_type);

          float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * m_block;
          llamafile_sgemm(m_block, m_local_num_[expert_idx],
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_proj_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_input_ptr,
                          config_.intermediate_size / ggml_blck_size((ggml_type)config_.down_type), down_output_ptr,
                          config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, (ggml_type)config_.down_type,
                          kt_effective_vec_dot_type((ggml_type)config_.down_type), GGML_TYPE_F32,
                          GGML_PREC_DEFAULT);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
          }
          for (int j = 0; j < k; j++) {
            if (config_.should_skip_expert(expert_ids[i * k + j])) {
              continue;
            }
            for (int e = 0; e < config_.hidden_size; e++) {
              m_output_fp32_[i][e] +=
                  m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] *
                  weights[i * k + j];
            }
          }
          for (int e = 0; e < config_.hidden_size; e++) {
            output[i * config_.hidden_size + e] = m_output_fp32_[i][e];
          }
        },
        nullptr);
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
  }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output_in) {
    auto output = (float*)output_in;
    if (qlen < config_.group_min_len) {
      for (int i = 0; i < qlen; i++) {
        forward_one(k, expert_ids + i * k, weights + i * k,
                    (uint8_t*)input + i * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                          ggml_blck_size((ggml_type)config_.hidden_type),
                    output + i * config_.hidden_size);
      }
      return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    forward_many(forward_len, k, expert_ids, weights, input, output);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k,
            (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size((ggml_type)config_.hidden_type) /
                                  ggml_blck_size((ggml_type)config_.hidden_type),
            output + forward_len * config_.hidden_size);
  }
};

template <>
class TP_MOE<LLAMA_MOE_TP> : public TP_MOE_Common<LLAMA_MOE_TP> {
 public:
  using TP_MOE_Common<LLAMA_MOE_TP>::TP_MOE_Common;

  void load_weights() {
    auto pool = this->config.pool;

    std::vector<int> tp_offsets(this->tp_count);
    int accumulated_offset = 0;
    for (int i = 0; i < this->tp_count; i++) {
      tp_offsets[i] = accumulated_offset;
      accumulated_offset += this->tp_configs[i].intermediate_size;
    }

    pool->dispense_backend()->do_numa_job([this, pool, tp_offsets](int tp_id) {
      this->tps[tp_id]->load_weights(this->config.intermediate_size, tp_offsets[tp_id]);
    });
    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }

  void merge_results(int qlen, void* output, bool incremental) {
    auto pool = this->config.pool;
    const bool kt_pt = kt_phase_timing_on();
    std::chrono::high_resolution_clock::time_point kt_m0;
    if (kt_pt) kt_m0 = std::chrono::high_resolution_clock::now();
    // Tile over (token, hidden-chunk) so decode (qlen=1) still spreads across the
    // whole pool instead of running the 8-NUMA reduce + from_float on a single core.
    // F-opt Phase 1 (Session D): merge was ~98us/layer single-core at qlen=1
    // (~11% of cpu_moe_wall). Only chunk when hidden_type is unblocked (BF16/F16/F32,
    // blck==1) so an arbitrary element boundary is always valid; block-quant hidden
    // types fall back to per-token (num_chunks=1), preserving the original behavior.
    const int H = config.hidden_size;
    const ggml_type htype = (ggml_type)config.hidden_type;
    const size_t hsz = ggml_type_size(htype);
    const int hblck = ggml_blck_size(htype);
    int num_chunks = 1;
    if (hblck == 1) {
      num_chunks = H / 256;
      if (num_chunks < 1) num_chunks = 1;
      if (num_chunks > 32) num_chunks = 32;
    }
    const int chunk = (H + num_chunks - 1) / num_chunks;
    pool->do_work_stealing_job(
        qlen * num_chunks, nullptr,
        [this, output, incremental, H, htype, hsz, hblck, num_chunks, chunk](int task_id) {
          const int token_nth = task_id / num_chunks;
          const int c = task_id % num_chunks;
          const int e0 = c * chunk;
          const int e1 = std::min(H, e0 + chunk);
          if (e0 >= e1) return;
          float* base0 = local_output_numa[0] + token_nth * H;
          if (incremental) {
            to_float((uint8_t*)output + (size_t)(token_nth * H + e0) * hsz / hblck,
                     local_output + token_nth * H + e0, e1 - e0, htype);
            for (int e = e0; e < e1; e++) base0[e] += local_output[token_nth * H + e];
          }
          for (int i = 1; i < this->tp_count; i++) {
            const float* basei = local_output_numa[i] + token_nth * H;
            for (int e = e0; e < e1; e++) base0[e] += basei[e];
          }
          from_float(base0 + e0, (uint8_t*)output + (size_t)(token_nth * H + e0) * hsz / hblck, e1 - e0, htype);
        },
        nullptr);
    if (kt_pt) {
      auto kt_m1 = std::chrono::high_resolution_clock::now();
      // 槽位 15 专用于 merge（forward_one 只用 tp 0..7）
      auto& acc = g_kt_phase_acc[15];
      acc.calls++;
      acc.quant_ns += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(kt_m1 - kt_m0).count();
      if (acc.calls % 4300 == 0) {
        fprintf(stderr, "[KT_PHASE merge] n=%llu avg/layer-call: merge=%.1fus (qlen=%d)\n",
                (unsigned long long)acc.calls, acc.quant_ns / 1e3 / acc.calls, qlen);
      }
    }
  }
};
#endif
