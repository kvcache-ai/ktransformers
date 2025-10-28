#ifndef KML_DEEPSEEKV3_HPP
#define KML_DEEPSEEKV3_HPP

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>

#include "gate.hpp"
#include "kblas.h"
#include "la/arm_kml.hpp"
#include "llama.cpp/ggml.h"
#include "mla.hpp"
#include "moe.hpp"

// #define DEBUG_LAYER_CORRECT

class DeepseekV3DecoderLayer
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  using A = float;

  GeneralConfig config;

  A* attn_norm;
  A* ffn_norm;

  A* hidden_states;
  ggml_bf16_t* hidden_states_bf16;
  int64_t* expert_ids;
  A* experts_weights;

  size_t layer_idx;

 public:
  std::shared_ptr<MLA_Interface> self_attn = nullptr;
  std::shared_ptr<MoEGate> gate = nullptr;
  std::shared_ptr<MoE_Interface> ffn = nullptr;

  using input_t = float;
  using output_t = float;

  DeepseekV3DecoderLayer(GeneralConfig config, size_t layer_idx) : config(config), layer_idx(layer_idx) {
    init_ggml();
    MemoryRequest mem_requests;
    PUSH_MEM_REQ(hidden_states, sizeof(A) * config.max_qlen * config.hidden_size);
    PUSH_MEM_REQ(hidden_states_bf16, sizeof(ggml_bf16_t) * config.max_qlen * config.hidden_size);
    PUSH_MEM_REQ(expert_ids,
                 sizeof(int64_t) * config.max_qlen * (config.num_experts_per_tok + config.n_shared_experts));
    PUSH_MEM_REQ(experts_weights, sizeof(A) * config.max_qlen * (config.num_experts_per_tok + config.n_shared_experts));

    shared_mem_buffer_for_decoder_layer.alloc(this, mem_requests);
  }
  void load_norm_binding(intptr_t attn_norm_ptr, ggml_type attn_norm_type, intptr_t ffn_norm_ptr,
                         ggml_type mlp_norm_type) {
    load_norm((void*)attn_norm_ptr, attn_norm_type, (void*)ffn_norm_ptr, mlp_norm_type);
  }

  void load_norm(const void* attn_norm, ggml_type attn_norm_type, const void* ffn_norm, ggml_type ffn_norm_type) {
    this->attn_norm = new A[config.hidden_size];
    this->ffn_norm = new A[config.hidden_size];
    convert_or_copy(this->attn_norm, (void*)attn_norm, attn_norm_type, config.hidden_size);
    convert_or_copy(this->ffn_norm, (void*)ffn_norm, ffn_norm_type, config.hidden_size);
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output) {
#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif

    size_t seq_len = 0;
    for (size_t i = 0; i < qlens.size(); i++) {
      seq_len += qlens[i];
    }
    // for (size_t i = 0; i < 5; i++) {
    // debug_f32((input_t *)input + (seq_len - 5 + i) * config.hidden_size, config.hidden_size);
    // }
    // printf("\n");

#ifdef DEBUG_LAYER_CORRECT
    std::string prefix = "Layer_" + std::to_string(layer_idx);
    dump_bin(prefix + "_input", (input_t*)input, seq_len * config.hidden_size);
#endif
    // Residue
    // printf("convert or copy hidden states, %ld,%ld\n", seq_len, config.hidden_size);
    convert_or_copy(hidden_states, (input_t*)input, seq_len * config.hidden_size);
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("copy residue");
#endif
    // Norm
    config.pool->do_work_stealing_job(seq_len, [&](int task_id) {
      A* input_row = (A*)input + task_id * config.hidden_size;
      RMSNorm<A>::rms_norm_single_with_weights(config.hidden_size, attn_norm, input_row);
    });
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("before attn norm");
#endif
    // self attention
    self_attn->forward(qlens, page_tables, kv_lens, input, output);
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("self attn");
#endif
#ifdef DEBUG_LAYER_CORRECT
    dump_bin(prefix + "_after_attn", (input_t*)output, seq_len * config.hidden_size);
#endif

    // Add Residue
    config.pool->do_work_stealing_job(seq_len, [&](int task_id) {
      A* hidden_state_row = hidden_states + task_id * config.hidden_size;
      A* output_row = (A*)output + task_id * config.hidden_size;
      A* input_row = (A*)input + task_id * config.hidden_size;
      for (size_t i = 0; i < config.hidden_size; i++) {
        hidden_state_row[i] += output_row[i];
        input_row[i] = hidden_state_row[i];
      }
    });
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("after attn add residue");
#endif

    // Norm
    config.pool->do_work_stealing_job(seq_len, [&](int task_id) {
      A* input_row = (A*)input + task_id * config.hidden_size;
      RMSNorm<A>::rms_norm_single_with_weights(config.hidden_size, ffn_norm, input_row);
    });
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("after attn norm");
#endif

    // Moe
    // size_t experts_ld = config.num_experts_per_tok;
    size_t experts_ld = config.num_experts_per_tok + config.n_shared_experts;
    if (gate != nullptr) {
      gate->forward(seq_len, (input_t*)input, expert_ids, experts_weights, experts_ld);
      for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < config.n_shared_experts; j++) {
          expert_ids[i * (experts_ld) + config.num_experts_per_tok + j] = config.n_routed_experts + j;
          experts_weights[i * (experts_ld) + config.num_experts_per_tok + j] = 1.0f;
        }
      }
    } else {
      for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < config.num_experts_per_tok + config.n_shared_experts; j++) {
          expert_ids[i * (experts_ld) + j] = j;
          experts_weights[i * (experts_ld) + j] = 1.0f;
        }
      }
    }
    // Debug 打印选中的 expert
    // printf("chosen experts for layer %ld:\n", layer_idx);
    // for (int i = 0; i < seq_len; i++) {
    //   for (int j = 0; j < experts_ld; j++) {
    //     printf("%ld ", expert_ids[i * experts_ld + j]);
    //   }
    //   printf("\n");
    // }
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("moe gate");
    // // Debug 重置选择专家为固定的连续专家
    // for (int i = 0; i < seq_len; i++) {
    //   for (int j = 0; j < experts_ld; j++) {
    //     expert_ids[i * experts_ld + j] = j;
    //   }
    // }
    // Debug 打印选中的 expert
    // printf("chosen experts for layer %ld:\n", layer_idx);
    // for (int i = 0; i < seq_len; i++) {
    //   for (int j = 0; j < experts_ld; j++) {
    //     printf("%ld ", expert_ids[i * experts_ld + j]);
    //   }
    //   printf("\n");
    // }
#endif
#ifdef DEBUG_LAYER_CORRECT
    dump_bin(prefix + "_expert_ids", expert_ids, seq_len * (config.num_experts_per_tok + config.n_shared_experts));
    dump_bin(prefix + "_expert_weights", experts_weights,
             seq_len * (config.num_experts_per_tok + config.n_shared_experts));
#endif
    convert_or_copy(hidden_states_bf16, (input_t*)input, seq_len * config.hidden_size);
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("convert bf16 time");
#endif
    ffn->forward(seq_len, experts_ld, expert_ids, experts_weights, hidden_states_bf16, output);
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("ffn");
#endif
    // Add Residue
    config.pool->do_work_stealing_job(seq_len, [&](int task_id) {
      A* hidden_state_row = hidden_states + task_id * config.hidden_size;
      A* output_row = (A*)output + task_id * config.hidden_size;
      for (size_t i = 0; i < config.hidden_size; i++) {
        output_row[i] += hidden_state_row[i];
      }
    });
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("add ffn residue");
#endif
#ifdef DEBUG_LAYER_CORRECT
    dump_bin(prefix + "_after_mlp", (input_t*)output, seq_len * config.hidden_size);
#endif
#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "DeepseekV3DecoderLayer" + std::to_string(layer_idx);
    perf_report();
#endif
  }
};

class DeepseekV3Model
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  using A = float;
  GeneralConfig config;
  A* norm_weights;

 public:
  using input_t = float;
  using output_t = float;
  std::vector<std::shared_ptr<DeepseekV3DecoderLayer>> layers;
  DeepseekV3Model(GeneralConfig config) : config(config) {
    init_ggml();
    norm_weights = new A[config.hidden_size];
    convert_or_copy(norm_weights, config.norm_weights_ptr, config.norm_weights_type, config.hidden_size);
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output) {
#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif
    size_t seq_len = 0;
    for (size_t i = 0; i < qlens.size(); i++) {
      seq_len += qlens[i];
    }
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i]->forward(qlens, page_tables, kv_lens, input, output);
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP(std::string("layer ") + std::to_string(i));
#endif
      if (i != layers.size() - 1) {
        convert_or_copy((A*)input, (A*)output, seq_len * config.hidden_size);
      } else {
        config.pool->do_work_stealing_job(seq_len, [&](int task_id) {
          A* output_row = (A*)output + task_id * config.hidden_size;
          RMSNorm<A>::rms_norm_single_with_weights(config.hidden_size, norm_weights, output_row);
        });
      }
    }

#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "DeepseekV3Model";
    perf_report();
#endif
  }
};

class DeepseekV3ForCausalLM
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  using GemmKernel = arm_kml::GemmKernelInt4;
  using KMatRefA = typename arm_kml::MatRef<int8_t>;
  using KMatRefB = typename arm_kml::MatRef<GemmKernel::dt>;
  // using KMatRefAB = typename arm_kml::MatRef<int8_t>;
  using KMatRefC = typename arm_kml::MatRef<int32_t>;

  using A = float;

  GeneralConfig config;
  A* input_hidden_states;
  A* output_hidden_states;

  A* lm_heads_ptr;
  GemmKernel::BufferA* lm_heads_ba;
  GemmKernel::BufferB* lm_heads_bb;
  GemmKernel::BufferC* lm_heads_bc;
  A* token_embd;

 public:
  using KMatRef = typename arm_kml::MatRef<float>;
  using input_t = int64_t;
  using output_t = float;
  std::shared_ptr<DeepseekV3Model> model;
  KMatRefB lm_heads;

  DeepseekV3ForCausalLM(GeneralConfig config) : config(config) {
    init_ggml();
    MemoryRequest mem_requests;
    lm_heads_ba = new GemmKernel::BufferA(config.max_qlen, config.hidden_size);
    lm_heads_bb = new GemmKernel::BufferB(config.vocab_size, config.hidden_size, true);
    lm_heads_bc = new GemmKernel::BufferC(config.max_qlen, config.vocab_size);

    mem_requests.append_function([this](void* new_ptr) { lm_heads_ba->set_data(new_ptr); },
                                 lm_heads_ba->required_size());
    lm_heads_bb->set_data(std::aligned_alloc(64, lm_heads_bb->required_size()));
    mem_requests.append_function([this](void* new_ptr) { lm_heads_bc->set_data(new_ptr); },
                                 lm_heads_bc->required_size());
    shared_mem_buffer.alloc(this, mem_requests);
    input_hidden_states = new A[config.max_qlen * config.hidden_size];
    output_hidden_states = new A[config.max_qlen * config.hidden_size];
    lm_heads_ptr = new A[config.vocab_size * config.hidden_size];
    token_embd = new A[config.vocab_size * config.hidden_size];
    convert_or_copy(lm_heads_ptr, config.lm_heads_ptr, config.lm_heads_type, config.vocab_size * config.hidden_size);
    // 做量化
    auto pool = config.pool;
    {
      size_t nth_lm_b = GemmKernel::recommended_nth(config.vocab_size);

      auto task = [&](int task_id) { lm_heads_bb->from_mat(lm_heads_ptr, task_id, nth_lm_b, -1, true); };
      pool->do_work_stealing_job(nth_lm_b, task);
    }
    lm_heads = KMatRefB(lm_heads_bb->b, config.hidden_size, config.vocab_size, config.hidden_size, CblasColMajor,
                        CblasNoTrans, lm_heads_bb->d);
    // lm_heads = KMatRef(lm_heads_ptr, config.vocab_size, config.hidden_size, config.hidden_size, CblasRowMajor);
    convert_or_copy(token_embd, config.token_embd_ptr, config.token_embd_type, config.vocab_size * config.hidden_size);
  }

  void forward_binding(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
                       intptr_t input, intptr_t output) {
    forward(qlens, page_tables, kv_lens, (const void*)input, (void*)output);
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output) {
#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif

    {
      size_t qlen_sum = 0;
      for (size_t i = 0; i < qlens.size(); i++) {
        qlen_sum += qlens[i];
      }
      // printf("DeepseekV3 forward, seq_len %ld\n", qlen_sum);
      for (size_t i = 0; i < qlen_sum; i++) {
        convert_or_copy(input_hidden_states + i * config.hidden_size,
                        token_embd + ((input_t*)input)[i] * config.hidden_size, config.hidden_size);
      }
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("token embd");
#endif
#ifdef DEBUG_LAYER_CORRECT
      dump_bin("input_ids", (input_t*)input, qlen_sum);
#endif
    }

    model->forward(qlens, page_tables, kv_lens, input_hidden_states, output_hidden_states);
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("model forward");
#endif
    // KMatRef hidden_states_ref =
    //     KMatRef(output_hidden_states, config.hidden_size, config.max_qlen, config.hidden_size, CblasColMajor);

    size_t qlen = 0;
    for (size_t i = 0; i < qlens.size(); i++) {
      qlen += qlens[i];
    }
    // printf("qlen is: %ld\n", qlen);
    // KMatRef logits_ref = KMatRef((A *)output, config.vocab_size, config.max_qlen, config.vocab_size, CblasColMajor);
    KMatRefC logits_ref =
        KMatRefC(lm_heads_bc->c, config.max_qlen, config.vocab_size, config.vocab_size, CblasRowMajor);
    KMatRef logits_out_ref = KMatRef((A*)output, config.max_qlen, config.vocab_size, config.vocab_size, CblasRowMajor);

    // 量化输入
    auto pool = config.pool;
    {
      size_t mth = GemmKernel::recommended_mth(qlen);
      auto task_counter = TaskCounter({mth});
      auto task = [&](int task_id) {
        size_t mth_idx = task_counter.at(task_id, 0);
        lm_heads_ba->from_mat(qlen, output_hidden_states, mth_idx, mth);
      };
      DIRECT_OR_POOL_BY(qlen, 10, task_counter.count(), task);
    }
    KMatRefA hidden_states_ref = KMatRefA(lm_heads_ba->a, qlen, config.hidden_size, config.hidden_size, CblasRowMajor);

    size_t qlen_sum = 0;
    for (size_t i = 0; i < qlens.size(); i++) {
      // auto h = hidden_states_ref.offset_block(0, qlen_sum + qlens[i] - 1, config.hidden_size, 1);
      auto h = hidden_states_ref.offset_block(qlen_sum + qlens[i] - 1, 0, 1, config.hidden_size);

      {
        const size_t vocab_block = 256;
        const size_t vocab_block_count = div_up(config.vocab_size, vocab_block);
        config.pool->do_work_stealing_job(vocab_block_count, [&](int task_id) {
          size_t vocab_idx = task_id * vocab_block;
          size_t vocab_begin = vocab_idx;
          size_t vocab_end = std::min(vocab_begin + vocab_block, (size_t)config.vocab_size);
          KMatRefB lm_head_ref = lm_heads.offset_col(vocab_begin, vocab_end - vocab_begin);
          // KMatRef logits_ref_block = logits_ref.offset_block(vocab_begin, i, vocab_end - vocab_begin, 1);
          KMatRefC logits_ref_block = logits_ref.offset_block(i, vocab_begin, 1, vocab_end - vocab_begin);

          // arm_kml::decode_mul_mat_clearc(lm_head_ref, h, logits_ref_block);
          // printf("h.ld: %ld, lm_head_ref.ld: %ld, logits_ref_block.ld: %ld\n", h.ld, lm_head_ref.ld,
          //        logits_ref_block.ld);
          arm_kml::decode_mul_mat_clearc(h, lm_head_ref, logits_ref_block);
          GemmKernel::apply_scale(logits_out_ref.data, logits_out_ref.ld, lm_heads_ba, lm_heads_bb, lm_heads_bc,
                                  qlen_sum + qlens[i] - 1, qlen_sum + qlens[i], vocab_begin, vocab_end, true,
                                  i - (qlen_sum + qlens[i] - 1));
        });
      }

      qlen_sum += qlens[i];
    }
#ifdef DEBUG_LAYER_CORRECT
    dump_bin("output_logits", (output_t*)output, qlens.size() * config.vocab_size);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("lm heads out");
#endif

#ifdef FORWARD_TIME_PROFILE

    time_perf_name = "DeepseekV3ForCausalLM";
    perf_report();

#endif
  }
};

#endif
