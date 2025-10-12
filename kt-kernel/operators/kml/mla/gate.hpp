#ifndef KML_GATE_HPP
#define KML_GATE_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../common.hpp"
#include "kblas.h"
#include "la/arm_kml.hpp"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
// #define DEBUG_THIS_MOEGATE
#ifdef DEBUG_THIS_MOEGATE
#include "test/debug.hpp"
#endif

class MoEGate
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  using A = float;
  GeneralGateConfig config;
  using KMatRef = typename arm_kml::MatRef<A>;

  A* bias;
  A* weight;
  KMatRef weight_ref;  // [expert_num, hidden_size]

  A* logits;           // [tokens, expert_num]
  A* score_to_choice;  // [tokens, expert_num]
  A* group_score;      // [tokens, group count]
  size_t* temp_idx;

  const size_t col_block = 256;
  const size_t row_block = 256;

 public:
  using input_t = float;
  using output_int_t = int64_t;
  using output_t = float;

  explicit MoEGate(const GeneralGateConfig& cfg) : config(cfg) {
    ASSERT_RELEASE(config.weight, "cfg.weight must not be null");
    ASSERT_RELEASE(config.n_routed_experts % config.n_group == 0, "E must be divisible by G");
    ASSERT_RELEASE(config.scoring_func == "sigmoid", "Only sigmoid scoring function is supported");
    ASSERT_RELEASE(config.topk_method == "noaux_tc", "Only noaux_tc topk method is supported");
    ASSERT_RELEASE(config.norm_topk_prob, "must normalize topk prob");

    MemoryRequest mem_requests;
    // PUSH_MEM_REQ(input, sizeof(float) * config.max_seqlen * config.hidden_size);
    PUSH_MEM_REQ(logits, sizeof(float) * config.max_seqlen * config.n_routed_experts);
    PUSH_MEM_REQ(score_to_choice, sizeof(float) * config.max_seqlen * config.n_routed_experts);
    PUSH_MEM_REQ(group_score, sizeof(float) * config.max_seqlen * config.n_group);
    PUSH_MEM_REQ(temp_idx, sizeof(size_t) * config.max_seqlen * config.n_routed_experts);

    shared_mem_buffer.alloc(this, mem_requests);

    weight = new A[config.n_routed_experts * config.hidden_size];
    bias = new A[config.n_routed_experts];

    convert_or_copy(weight, config.weight, config.weight_type, config.n_routed_experts * config.hidden_size);
    convert_or_copy(bias, config.e_score_correction_bias, config.e_score_correction_bias_type, config.n_routed_experts);
    weight_ref = KMatRef(weight, config.n_routed_experts, config.hidden_size, config.hidden_size, CblasRowMajor);
  }

  void forward_binding(size_t seq_len, intptr_t input_hidden_states_raw, intptr_t output_topk_idx_raw,
                       intptr_t output_topk_weight_raw) {
    forward(seq_len, (input_t*)input_hidden_states_raw, (output_int_t*)output_topk_idx_raw,
            (output_t*)output_topk_weight_raw);
  }

  void forward(size_t seq_len, input_t* input_hidden_states, output_int_t* output_topk_idx,
               output_t* output_topk_weight_raw) {
    forward(seq_len, input_hidden_states, output_topk_idx, output_topk_weight_raw, config.num_experts_per_tok);
  }

  // forward: hidden_states [B,L,H] → (topk_idx, topk_weight) each [B·L,K]
  // ‑ hidden_states must be contiguous row‑major with H fastest.
  // ‑ outputs are flattened (token first) and resized inside.
  void forward(size_t seq_len, input_t* input_hidden_states, output_int_t* output_topk_idx,
               output_t* output_topk_weight, size_t output_ld) {
#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif

    KMatRef input_ref =
        KMatRef((input_t*)input_hidden_states, config.hidden_size, seq_len, config.hidden_size, CblasColMajor);

#ifdef DEBUG_THIS_MOEGATE
    dump_bin("gate_input", input_ref.data, seq_len * config.hidden_size);
#endif
    KMatRef logits_ref = KMatRef(logits, config.n_routed_experts, seq_len, config.n_routed_experts, CblasColMajor);
    {
      size_t n_routed_experts_block = (seq_len == 1) ? 2 : 64;
      const size_t seq_len_block = 64;
      const size_t n_routed_experts_block_count = div_up(config.n_routed_experts, n_routed_experts_block);
      const size_t seq_len_block_count = div_up(seq_len, seq_len_block);
      auto task_counter = TaskCounter({n_routed_experts_block_count, seq_len_block_count});
      auto task = [&](int task_id) {
        size_t n_routed_experts_block_idx = task_counter.at(task_id, 0);
        size_t seq_len_block_idx = task_counter.at(task_id, 1);
        size_t n_routed_experts_begin = n_routed_experts_block_idx * n_routed_experts_block;
        size_t n_routed_experts_end =
            std::min(n_routed_experts_begin + n_routed_experts_block, config.n_routed_experts);
        size_t seq_len_begin = seq_len_block_idx * seq_len_block;
        size_t seq_len_end = std::min(seq_len_begin + seq_len_block, seq_len);
        if (seq_len == 1) {
          arm_kml::decode_mul_mat_clearc(
              weight_ref.offset_block(n_routed_experts_begin, 0, n_routed_experts_end - n_routed_experts_begin,
                                      config.hidden_size),
              input_ref.offset_block(0, seq_len_begin, config.hidden_size, seq_len_end - seq_len_begin),
              logits_ref.offset_block(n_routed_experts_begin, seq_len_begin,
                                      n_routed_experts_end - n_routed_experts_begin, seq_len_end - seq_len_begin));
        } else {
          arm_kml::mul_mat_clearc(
              weight_ref.offset_block(n_routed_experts_begin, 0, n_routed_experts_end - n_routed_experts_begin,
                                      config.hidden_size),
              input_ref.offset_block(0, seq_len_begin, config.hidden_size, seq_len_end - seq_len_begin),
              logits_ref.offset_block(n_routed_experts_begin, seq_len_begin,
                                      n_routed_experts_end - n_routed_experts_begin, seq_len_end - seq_len_begin));
        }
      };
      config.pool->do_work_stealing_job(task_counter.count(), task);
    }
    // arm_kml::mul_mat_clearc(weight_ref, input_ref, logits_ref);
    // if (seq_len == 1) {
    //   for (int i = 0; i < config.n_routed_experts; i++) {
    //     printf("%f ", logits[i]);
    //   }
    //   printf("\n");
    //   throw std::runtime_error("end");
    // }
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("moe gate logits");
#endif

#ifdef DEBUG_THIS_MOEGATE
    dump_bin("gate_logits", logits_ref.data, seq_len * config.n_routed_experts);
    dump_bin("bias", bias, config.n_routed_experts);
#endif
    for (size_t i = 0; i < seq_len; i++) {
      float* logits_row = logits + i * config.n_routed_experts;
      for (size_t j = 0; j < config.n_routed_experts; j++) {
        logits_row[j] = 1.f / (1.f + std::exp(-logits_row[j]));
      }
    }

    auto top2 = [](float* data, size_t begin, size_t end) {
      if (end - begin < 2) {
        throw std::invalid_argument("top2 requires at least two elements");
      }

      float first = -std::numeric_limits<float>::infinity();
      float second = -std::numeric_limits<float>::infinity();
      for (size_t i = begin; i < end; ++i) {
        float v = data[i];
        if (v > first) {
          second = first;
          first = v;
        } else if (v > second) {
          second = v;
        }
      }
      return first + second;
    };

    for (size_t i = 0; i < seq_len; i++) {
      float* logits_row = logits + i * config.n_routed_experts;
      float* scores_row = score_to_choice + i * config.n_routed_experts;
      for (size_t j = 0; j < config.n_routed_experts; j++) {
        scores_row[j] = logits_row[j] + bias[j];
      }
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("moe gate score to choice");
#endif
#ifdef DEBUG_THIS_MOEGATE
    dump_bin("scores_to_choice", score_to_choice, seq_len * config.n_routed_experts);
#endif
    for (size_t i = 0; i < seq_len; i++) {
      output_int_t* output_topk_idx_row = output_topk_idx + i * output_ld;
      output_t* output_topk_weight_row = output_topk_weight + i * output_ld;
      float* logits_row = logits + i * config.n_routed_experts;
      float* scores_row = score_to_choice + i * config.n_routed_experts;
      float* group_score_row = group_score + i * config.n_group;
      size_t* temp_idx_row = temp_idx + i * config.n_routed_experts;

      size_t group_size = config.n_routed_experts / config.n_group;
      for (size_t g = 0; g < config.n_group; g++) {
        size_t group_begin = g * group_size;
        size_t group_end = group_begin + group_size;
        group_score_row[g] = top2(scores_row, group_begin, group_end);
        temp_idx_row[g] = g;
      }
      std::sort(temp_idx_row, temp_idx_row + config.n_group,
                [&](auto& a, auto& b) { return group_score_row[a] > group_score_row[b]; });

      for (size_t g = config.topk_group; g < config.n_group; g++) {
        size_t group_begin = temp_idx_row[g] * group_size;
        size_t group_end = group_begin + group_size;
        for (size_t j = group_begin; j < group_end; j++) {
          scores_row[j] = -std::numeric_limits<float>::infinity();
        }
      }

      for (int j = 0; j < config.n_routed_experts; j++) {
        temp_idx_row[j] = j;
      }
      std::sort(temp_idx_row, temp_idx_row + config.n_routed_experts,
                [&](auto& a, auto& b) { return scores_row[a] > scores_row[b]; });

      float sum = 1e-20f;
      for (int j = 0; j < config.num_experts_per_tok; j++) {
        output_topk_idx_row[j] = temp_idx_row[j];
        output_topk_weight_row[j] = logits_row[temp_idx_row[j]];
        sum += output_topk_weight_row[j];
      }
      for (int j = 0; j < config.num_experts_per_tok; j++) {
        output_topk_weight_row[j] /= sum;
        output_topk_weight_row[j] *= config.routed_scaling_factor;
      }
    }
#ifdef DEBUG_THIS_MOEGATE
    dump_bin("group_scores", group_score, seq_len * config.n_group);
    dump_bin("gate_logits_toped", score_to_choice, seq_len * config.n_routed_experts);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("gate_logits_toped");
#endif
// printf("Gate Forward Done\n");
// for(size_t i=0;i<seq_len;i++){
//   printf("seq %ld, topk: ", i);
//   for(size_t j=0;j<config.num_experts_per_tok;j++){
//     printf("%ld ", output_topk_idx[i * output_ld + j]);
//   }
//   for(size_t j=0;j<config.num_experts_per_tok;j++){
//     printf("%f ", output_topk_weight[i * output_ld + j]);
//   }
//   printf("\n");
// }
#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "moe gate inner";
    perf_report();
#endif
  }
};

#endif