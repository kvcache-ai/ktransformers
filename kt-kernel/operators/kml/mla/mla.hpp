#ifndef KML_MLA_HPP
#define KML_MLA_HPP

#include "../mla-tp.hpp"
#include "../reduce.hpp"
#include "../rms-norm.hpp"
#include "../rope.hpp"
#include "../softmax.hpp"
#include "kblas.h"
#include "la/arm_kml.hpp"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"

// #define DEBUG_THIS_MLA
#ifdef DEBUG_THIS_MLA
#include "test/debug.hpp"
#endif
#include <arm_sve.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// #define DEBUG_THIS_MLA
// #define FORWARD_TIME_PROFILE

#define DIRECT_OR_POOL_BY(what, threshold, var, fn)              \
  do {                                                           \
    if ((what) < (threshold)) {                                  \
      for (int i = 0; i < (var); i++) {                          \
        (fn)(i);                                                 \
      }                                                          \
    } else {                                                     \
      pool->do_work_stealing_job((var), nullptr, (fn), nullptr); \
    }                                                            \
  } while (0)

inline void debug_output(const float16_t* data) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", data[i]);
  }
  printf("\n");
}

inline void debug_output(const float* data) {
  for (int i = 0; i < 10; i++) {
    printf("%f ", data[i]);
  }
  printf("\n");
}

inline void debug_output(const bfloat16_t* data) {
  for (int i = 0; i < 10; i++) {
    float x = 0;
    *(bfloat16_t*)(&x) = data[i];
    printf("%f ", x);
  }
  printf("\n");
}

template <typename T>
inline void dump_to(std::string file_name, T* data, size_t count) {
  std::ofstream f(file_name);
  for (int i = 0; i < count; i++) {
    f << data[i] << " ";
  }
  f.close();
}

template <typename A>
class KML_MLA_TP
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  using T_RMSNorm = RMSNorm<A>;
  using T_RopeAngle = DeepseekV3YarnRotaryEmbedding;
  using T_RopeApplier = Rope<T_RopeAngle, A>;
  using T_SoftmaxApplier = Softmax<A>;
  using KMatRef = typename arm_kml::MatRef<A>;
  GeneralMLAConfig config;
  const size_t col_block = 256;
  const size_t row_block = 256;

  int tp_part_idx;
  std::vector<void*> default_attention_masks;

  // std::vector<void *> kv_lora_pages; // [page_count * page_token_count * nope]
  // std::vector<void *> rope_pages;    // [page_count * page_token_count * nope]
  std::vector<A*> cc_pages;  // [page_count * page_token_count * (kv rank + rope size)]
  size_t cc_size;
  std::vector<KMatRef> cc_page_refs, kv_lora_page_refs, rope_page_refs;
  // weights
  A* local_q_a_proj;  // [hidden_size * q_lora_rank]
  A* local_q_a_norm;
  A* local_q_b_proj;            // [num_heads * (nope_size + rope_size))]
  A* local_kv_a_proj_with_mqa;  // [hidden_size * (kv_lora_rank + rope)]
  A* local_kv_a_norm;
  A* local_k_b_proj;
  A* local_v_b_proj;
  // A *local_kv_b_proj; // [(num_heads * (nope_size + nope_size) * kv_lora_rank)],
  // q_absorb:   [num_heads * nope_size * kv_lora_rank]
  // out_absorb: [num_heads * nope_size * kv_lora_rank]
  A* local_w_o;  // [(num_heads * hidden_size * nope_size)]

  std::unique_ptr<T_RopeAngle> rope_angle;

  KMatRef local_q_a_proj_ref;
  KMatRef local_q_b_proj_ref;
  KMatRef local_kv_a_proj_with_mqa_ref;
  KMatRef local_k_b_proj_ref;
  KMatRef local_v_b_proj_ref;
  KMatRef local_w_o_ref;

  // for each query
  A* q_lora_rank;  // [qlen_sum, q_lora_rank]
  A* q_nope;       // [num_heads * max_qlen, nope_size]
  KMatRef q_nope_tmp_ref;

  A* q;  // [num_heads * max_qlen, max(kv_lora_rank,nope_size) + rope_size]
  size_t q_ld;
  KMatRef q_ref, q_pe_absorb_ref, q_pe_noabsorb_ref, q_nope_ref, q_kv_lora_rank_ref, q_attn_absorb_ref,
      q_attn_noabsorb_ref;
  // std::vector<A *> q_pe;              // [num_heads * max_qlen * rope_size]
  // std::vector<A *> q_nope;            // [num_heads * max_qlen * nope_size]
  A* k;  // [num_heads * max_kvlen * (nope_size + rope_size)]
  KMatRef k_ref, k_nope_ref, k_rope_ref;

  A* attention_weights;
  KMatRef attention_weights_ref;  // [max_kvlen, max_qlen* num_heads]
  // std::vector<A *> q_absorb;          // [num_heads, max_qlen, kv_lora_rank],  or [num_heads, kv_lora_rank, max_qlen]
  A* o_absorb_or_v;
  KMatRef o_absorb_ref;  // [num_heads, max_qlen, kv_lora_rank]
  KMatRef v_ref;         // [num_heads,nope,max_kvlen]

  A* attention_output;  // [num_heads * max_qlen * nope]
  KMatRef attention_output_ref;

  size_t sub_num_heads = 8;
  // A *qlen_output; // [max_qlen * hidden_size]

  A softmax_scale;

#ifdef DEBUG_THIS_MLA
  std::string file_name;
#endif

 public:
  using input_t = A;
  using output_t = A;

  KML_MLA_TP(GeneralMLAConfig config, int tp_part_idx) : config(config), tp_part_idx(tp_part_idx) {
    init_ggml();
    cc_size = config.kv_lora_rank + config.rope_size;
    MemoryRequest mem_requests;
    softmax_scale = 1.0 / sqrt(config.rope_size + config.nope_size);
    PUSH_MEM_REQ(q_lora_rank,
                 sizeof(std::remove_pointer_t<decltype(q_lora_rank)>) * config.q_lora_rank * config.max_qlen);

    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          q_nope = (A*)new_ptr;
          q_nope_tmp_ref =
              KMatRef(q_nope, config.nope_size, config.num_heads * config.max_qlen, config.nope_size, CblasColMajor);
        },
        sizeof(std::remove_pointer_t<decltype(q_nope)>) * config.num_heads * config.nope_size * config.max_qlen);

    q_ld = std::max(config.kv_lora_rank, config.nope_size) + config.rope_size;
    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          q = (A*)new_ptr;
          q_ref = KMatRef(q, q_ld, config.num_heads * config.max_qlen, q_ld, CblasColMajor);
          q_pe_absorb_ref = q_ref.offset_row(config.kv_lora_rank, config.rope_size);
          q_pe_noabsorb_ref = q_ref.offset_row(config.nope_size, config.rope_size);
          q_kv_lora_rank_ref = q_ref.offset_row(0, config.kv_lora_rank);
          q_nope_ref = q_ref.offset_row(0, config.nope_size);
          q_attn_absorb_ref = q_ref.offset_row(0, config.kv_lora_rank + config.rope_size);
          q_attn_noabsorb_ref = q_ref.offset_row(0, config.nope_size + config.rope_size);
        },
        sizeof(std::remove_pointer_t<decltype(q)>) * config.num_heads * config.max_qlen * q_ld);
    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          attention_weights = (A*)new_ptr;
          attention_weights_ref = KMatRef(attention_weights, config.max_kvlen, config.max_qlen * config.num_heads,
                                          config.max_kvlen, CblasColMajor);
        },
        sizeof(std::remove_pointer_t<decltype(attention_weights)>) * config.max_kvlen * config.max_qlen *
            config.num_heads);

    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          attention_output = (A*)new_ptr;
          attention_output_ref = KMatRef(attention_output, config.nope_size * config.num_heads, config.max_qlen,
                                         config.nope_size * config.num_heads, CblasColMajor);
        },
        (sizeof(std ::remove_pointer_t<decltype(attention_output)>) * config.num_heads * config.nope_size *
         config.max_qlen));

    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          k = (A*)new_ptr;
          k_ref = KMatRef(k, config.nope_size + config.rope_size, config.max_kvlen * config.num_heads,
                          config.nope_size + config.rope_size, CblasColMajor);
          k_nope_ref = k_ref.offset_row(0, config.nope_size);
          k_rope_ref = k_ref.offset_row(config.nope_size, config.rope_size);
        },
        sizeof(std::remove_pointer_t<decltype(k)>) * config.num_heads * (config.nope_size + config.rope_size) *
            config.max_kvlen);
    size_t o_absorb_or_v_size = std::max(config.kv_lora_rank * config.max_qlen, config.nope_size * config.max_kvlen);
    mem_requests.append_function(
        [this](void* new_ptr) {
          auto& config = this->config;
          o_absorb_or_v = (A*)new_ptr;
          o_absorb_ref = KMatRef(o_absorb_or_v, config.kv_lora_rank, config.max_qlen * config.num_heads,
                                 config.kv_lora_rank, CblasColMajor);
          v_ref = KMatRef(o_absorb_or_v, config.nope_size, config.max_kvlen * config.num_heads, config.nope_size,
                          CblasColMajor);
        },
        sizeof(std::remove_pointer_t<decltype(o_absorb_or_v)>) * o_absorb_or_v_size * config.num_heads);

    rope_angle = std::make_unique<T_RopeAngle>(
        config.rope_size, config.max_position_embeddings, config.rope_theta, config.rope_scaling_factor,
        config.rope_scaling_original_max_position_embeddings, config.rope_scaling_beta_fast,
        config.rope_scaling_beta_slow, config.rope_scaling_mscale, config.rope_scaling_mscale_all_dim);
    rope_angle->init(config.max_kvlen);
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }
  void load_weights(int complete_num_heads, int offset) {
    if constexpr (std::is_same_v<A, float16_t>) {
      ASSERT_RELEASE(config.q_a_proj_type == GGML_TYPE_F16, "q_a_proj_type must be GGML_TYPE_F16");
      ASSERT_RELEASE(config.q_b_proj_type == GGML_TYPE_F16, "q_b_proj_type must be GGML_TYPE_F16");
      ASSERT_RELEASE(config.kv_a_proj_with_mqa_type == GGML_TYPE_F16, "kv_a_proj_with_mqa_type must be GGML_TYPE_F16");
      ASSERT_RELEASE(config.kv_b_proj_type == GGML_TYPE_F16, "kv_b_proj_type must be GGML_TYPE_F16");
      ASSERT_RELEASE(config.w_o_type == GGML_TYPE_F16, "w_o_type must be GGML_TYPE_F16");
    } else if constexpr (std::is_same_v<A, float>) {
    } else {
      throw std::runtime_error("Unsupported type for KML_MLA_TP");
    }

    default_attention_masks.resize(config.max_kvlen);
    for (int i = 0; i < config.max_kvlen; i++) {
      A* mask = new A[config.max_kvlen];

      memset(mask, 0, config.max_kvlen * sizeof(A));
      for (int j = i + 1; j < config.max_kvlen; j++) {
        mask[j] = -std::numeric_limits<float>::infinity();
      }
      default_attention_masks[i] = mask;
    }

    local_q_a_proj = new A[config.hidden_size * config.q_lora_rank];
    convert_or_copy(local_q_a_proj, config.q_a_proj, (ggml_type)config.q_a_proj_type,
                    config.hidden_size * config.q_lora_rank);

    local_q_a_norm = new A[config.q_lora_rank];
    if (config.q_a_norm == nullptr) {
      for (size_t i = 0; i < config.q_lora_rank; i++) {
        local_q_a_norm[i] = 1;
      }
    } else {
      convert_or_copy(local_q_a_norm, config.q_a_norm, (ggml_type)config.q_a_norm_type, config.q_lora_rank);
    }

    local_kv_a_proj_with_mqa = new A[config.hidden_size * (config.kv_lora_rank + config.rope_size)];

    convert_or_copy(local_kv_a_proj_with_mqa, config.kv_a_proj_with_mqa, (ggml_type)config.kv_a_proj_with_mqa_type,
                    config.hidden_size * (config.kv_lora_rank + config.rope_size));

    local_kv_a_norm = new A[config.kv_lora_rank];
    if (config.kv_a_norm == nullptr) {
      for (size_t i = 0; i < config.kv_lora_rank; i++) {
        local_kv_a_norm[i] = 1;
      }
    } else {
      convert_or_copy(local_kv_a_norm, config.kv_a_norm, (ggml_type)config.kv_a_norm_type, config.kv_lora_rank);
    }

    local_q_b_proj = new A[config.num_heads * (config.nope_size + config.rope_size) * config.q_lora_rank];

    convert_or_copy(
        local_q_b_proj,
        offset_pointer(config.q_b_proj, offset * (config.nope_size + config.rope_size) * config.q_lora_rank *
                                            ggml_type_size((ggml_type)config.q_b_proj_type)),
        (ggml_type)config.q_b_proj_type, config.num_heads * (config.nope_size + config.rope_size) * config.q_lora_rank);

    local_k_b_proj = new A[config.num_heads * config.nope_size * config.kv_lora_rank];
    local_v_b_proj = new A[config.num_heads * config.nope_size * config.kv_lora_rank];
    for (size_t i = 0; i < config.num_heads; i++) {
      convert_or_copy(
          local_k_b_proj + i * config.nope_size * config.kv_lora_rank,
          offset_pointer(config.kv_b_proj, (i + offset) * (config.nope_size + config.nope_size) * config.kv_lora_rank *
                                               ggml_type_size((ggml_type)config.kv_b_proj_type)),
          (ggml_type)config.kv_b_proj_type, config.nope_size * config.kv_lora_rank);

      convert_or_copy(
          local_v_b_proj + i * config.nope_size * config.kv_lora_rank,
          offset_pointer(config.kv_b_proj, ((i + offset) * (config.nope_size + config.nope_size) + config.nope_size) *
                                               config.kv_lora_rank * ggml_type_size((ggml_type)config.kv_b_proj_type)),
          (ggml_type)config.kv_b_proj_type, config.nope_size * config.kv_lora_rank);
    }
    local_k_b_proj_ref = KMatRef((A*)local_k_b_proj, config.num_heads * config.nope_size, config.kv_lora_rank,
                                 config.kv_lora_rank, CblasRowMajor);
    local_v_b_proj_ref = KMatRef((A*)local_v_b_proj, config.num_heads * config.nope_size, config.kv_lora_rank,
                                 config.kv_lora_rank, CblasRowMajor);

    local_w_o = new A[config.num_heads * config.hidden_size * config.nope_size];
    for (size_t i = 0; i < config.hidden_size; i++) {
      convert_or_copy(
          local_w_o + i * config.num_heads * config.nope_size,
          offset_pointer(config.o_proj, (i * complete_num_heads * config.nope_size + (offset)*config.nope_size) *
                                            ggml_type_size((ggml_type)config.w_o_type)),
          (ggml_type)config.w_o_type, config.num_heads * config.nope_size);
    }

    local_q_a_proj_ref =
        KMatRef((A*)local_q_a_proj, config.q_lora_rank, config.hidden_size, config.hidden_size, CblasRowMajor);

    local_q_b_proj_ref = KMatRef((A*)local_q_b_proj, config.num_heads * (config.nope_size + config.rope_size),
                                 config.q_lora_rank, config.q_lora_rank, CblasRowMajor);

    local_kv_a_proj_with_mqa_ref = KMatRef((A*)local_kv_a_proj_with_mqa, config.kv_lora_rank + config.rope_size,
                                           config.hidden_size, config.hidden_size, CblasRowMajor);

    local_w_o_ref = KMatRef((A*)local_w_o, config.hidden_size, config.nope_size * config.num_heads,
                            config.nope_size * config.num_heads, CblasRowMajor);
  }

  void set_pages(std::vector<void*> kv_lora_pages, std::vector<void*> pe_pages) {
    // this->kv_lora_pages = kv_lora_pages;
    // this->rope_pages = pe_pages;
  }

  void set_local_pages(int page_count) {
    cc_pages.resize(page_count);
    cc_page_refs.resize(page_count);
    kv_lora_page_refs.resize(page_count);
    rope_page_refs.resize(page_count);
    for (int i = 0; i < page_count; i++) {
      cc_pages[i] = new A[config.token_count_in_page * (config.kv_lora_rank + config.rope_size)];
      cc_page_refs[i] = KMatRef(cc_pages[i], cc_size, config.token_count_in_page, cc_size, CblasColMajor);
      kv_lora_page_refs[i] = cc_page_refs[i].offset_row(0, config.kv_lora_rank);
      rope_page_refs[i] = cc_page_refs[i].offset_row(config.kv_lora_rank, config.rope_size);
    }

    // kv_lora_pages.resize(page_count);
    // rope_pages.resize(page_count);
    // auto kv_lora_page_ptr = new A[page_count * config.token_count_in_page * config.kv_lora_rank];
    // auto rope_page_ptr = new A[page_count * config.token_count_in_page * config.rope_size];
    // memset(kv_lora_page_ptr, 0, page_count * config.token_count_in_page * config.kv_lora_rank * sizeof(A));
    // memset(rope_page_ptr, 0, page_count * config.token_count_in_page * config.rope_size * sizeof(A));

    // for (int i = 0; i < page_count; i++) {
    //   kv_lora_pages[i] =
    //       offset_pointer(kv_lora_page_ptr, i * config.token_count_in_page * config.kv_lora_rank * sizeof(A));
    //   rope_pages[i] = offset_pointer(rope_page_ptr, i * config.token_count_in_page * config.rope_size * sizeof(A));
    // }
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               std::vector<void*> attention_masks, const void* input, void* output) {
    forward_prefill(qlens, page_tables, kv_lens, attention_masks, (input_t*)input, (output_t*)output);
  }

  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output) {
    forward(qlens, page_tables, kv_lens, default_attention_masks, input, output);
  }

  void nope_attention_q_absorb(int qlen, int kvlen, const std::vector<int>& page_table, bool increamental = true) {
    auto pool = config.pool->get_subpool(tp_part_idx);
    {
      // q absorb
      size_t qlen_block = div_up((size_t)qlen, row_block);
      size_t kv_rank_block = div_up(config.kv_lora_rank, col_block);
      auto task_counter = TaskCounter({config.num_heads, qlen_block, kv_rank_block});

      auto task = [&](int task_id) {
        auto [head_idx, qlen_block_idx, kv_rank_block_idx] = task_counter.get<3>(task_id);

        size_t qlen_begin = qlen_block_idx * row_block;
        size_t qlen_end = std::min(qlen_begin + row_block, (size_t)qlen);

        size_t kv_rank_begin = kv_rank_block_idx * col_block;
        size_t kv_rank_end = std::min(kv_rank_begin + col_block, (size_t)config.kv_lora_rank);

        KMatRef this_local_k_b_proj_ref = local_k_b_proj_ref.offset_block(
            head_idx * config.nope_size, kv_rank_begin, config.nope_size, kv_rank_end - kv_rank_begin);
        this_local_k_b_proj_ref = this_local_k_b_proj_ref.t();
        // printf("q absorb %d [%d,%d),[%d,%d)\n", head_idx, qlen_begin, qlen_end, kv_rank_begin, kv_rank_end);
        arm_kml::mul_mat_clearc(this_local_k_b_proj_ref,
                                q_nope_tmp_ref.offset_col(head_idx * qlen + qlen_begin, qlen_end - qlen_begin),

                                q_kv_lora_rank_ref.offset_block(kv_rank_begin, head_idx * qlen + qlen_begin,
                                                                kv_rank_end - kv_rank_begin, qlen_end - qlen_begin));
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }

#ifdef DEBUG_THIS_MLA
    printf("q absorb [%d] \n", tp_part_idx);
    // dump_bin(file_name + "_k_b_lora", (A *)local_kv_b_proj, config.kv_lora_rank * config.nope_size);
    // dump_bin(file_name + "_q_absorb", (A *)q_absorb[0], config.kv_lora_rank * config.max_qlen);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("q absorb");
#endif

    {
      // nope attention

      size_t qlen_block = div_up((size_t)qlen * config.num_heads, col_block);
      // page size % col_block == 0
      size_t kvlen_block = div_up((size_t)kvlen + qlen, col_block);

      TaskCounter task_counter({qlen_block, kvlen_block});
      auto task = [&](int task_id) {
        auto [qlen_block_idx, kvlen_block_idx] = task_counter.get<2>(task_id);

        size_t qlen_begin = qlen_block_idx * col_block;
        size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlen * config.num_heads);

        size_t kvlen_begin = kvlen_block_idx * col_block;
        size_t kvlen_end = std::min(kvlen_begin + col_block, (size_t)kvlen + qlen);

        size_t kvlen_block_size = kvlen_end - kvlen_begin;
        size_t kv_page = kvlen_begin / config.token_count_in_page;
        size_t token_at_in_page = kvlen_begin % config.token_count_in_page;

        KMatRef this_cc_ref =
            KMatRef((A*)cc_pages[page_table[kv_page]], config.token_count_in_page, cc_size, cc_size, CblasRowMajor);
        this_cc_ref = this_cc_ref.offset_row(token_at_in_page, kvlen_end - kvlen_begin);

        KMatRef this_q_aborb_ref = q_attn_absorb_ref.offset_col(qlen_begin, qlen_end - qlen_begin);

        KMatRef this_attention_weights_ref =
            attention_weights_ref.offset_block(kvlen_begin, qlen_begin, kvlen_end - kvlen_begin, qlen_end - qlen_begin);

        arm_kml::mul_mat_clearc(this_cc_ref, this_q_aborb_ref, this_attention_weights_ref);
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }

#ifdef DEBUG_THIS_MLA
    printf("attention weights[%d] \n", tp_part_idx);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("nope attention");
#endif
  }
  void nope_attention_no_absorb(int qlen, int kvlen, const std::vector<int>& page_table, bool increamental = true) {
    auto pool = config.pool->get_subpool(tp_part_idx);
    {
      // k nope
      size_t nope_block = div_up(config.nope_size, row_block);
      size_t kvlen_block = div_up((size_t)kvlen + qlen, col_block);
      auto task_counter = TaskCounter({config.num_heads, kvlen_block, nope_block});

      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t kvlen_block_idx = task_counter.at(task_id, 1);
        size_t nope_block_idx = task_counter.at(task_id, 2);

        size_t kvlen_begin = kvlen_block_idx * col_block;
        size_t kvlen_end = std::min(kvlen_begin + col_block, (size_t)kvlen + qlen);

        size_t kvlen_block_size = kvlen_end - kvlen_begin;
        size_t kv_page = kvlen_begin / config.token_count_in_page;
        size_t token_at_in_page = kvlen_begin % config.token_count_in_page;

        size_t nope_begin = nope_block_idx * row_block;
        size_t nope_end = std::min(nope_begin + row_block, config.nope_size);

        auto k_b_ref = local_k_b_proj_ref.offset_row(head_idx * config.nope_size + nope_begin, nope_end - nope_begin);

        KMatRef cc_ref = kv_lora_page_refs[page_table[kv_page]];
        cc_ref = cc_ref.offset_col(token_at_in_page, kvlen_end - kvlen_begin);

        KMatRef this_k_nope_ref = k_nope_ref.offset_block(nope_begin, head_idx * config.max_kvlen + kvlen_begin,
                                                          nope_end - nope_begin, kvlen_end - kvlen_begin);

        arm_kml::mul_mat_clearc(k_b_ref, cc_ref, this_k_nope_ref);
        if (nope_block_idx == 0) {
          auto this_k_rope_ref =
              k_rope_ref.offset_col(head_idx * config.max_kvlen + kvlen_begin, kvlen_end - kvlen_begin);
          auto this_rope_page_ref =
              rope_page_refs[page_table[kv_page]].offset_col(token_at_in_page, kvlen_end - kvlen_begin);
          for (size_t i = 0; i < this_k_rope_ref.C; i++) {
            memcpy(this_k_rope_ref.data + this_k_rope_ref.ld * i, this_rope_page_ref.data + this_rope_page_ref.ld * i,
                   config.rope_size * sizeof(A));
          }
        }
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("k nope");
#endif

    {
      // nope attention
      size_t kvlen_block = div_up((size_t)kvlen + qlen, row_block);
      size_t qlen_block = div_up((size_t)qlen, col_block);
      auto task_counter = TaskCounter({config.num_heads, kvlen_block, qlen_block});

      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t kvlen_block_idx = task_counter.at(task_id, 1);
        size_t qlen_block_idx = task_counter.at(task_id, 2);

        size_t kvlen_begin = kvlen_block_idx * row_block;
        size_t kvlen_end = std::min(kvlen_begin + row_block, (size_t)kvlen + qlen);
        size_t qlen_begin = qlen_block_idx * col_block;
        size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlen);

        KMatRef this_k_ref = k_ref.offset_col(head_idx * config.max_kvlen + kvlen_begin, kvlen_end - kvlen_begin);
        this_k_ref = this_k_ref.t();

        KMatRef this_q_ref = q_attn_noabsorb_ref.offset_col(head_idx * qlen + qlen_begin, qlen_end - qlen_begin);

        KMatRef this_attention_weights_ref = attention_weights_ref.offset_block(
            kvlen_begin, head_idx * qlen + qlen_begin, kvlen_end - kvlen_begin, qlen_end - qlen_begin);
        if (increamental) {
          arm_kml::mul_mat(this_k_ref, this_q_ref, this_attention_weights_ref);
        } else {
          arm_kml::mul_mat_clearc(this_k_ref, this_q_ref, this_attention_weights_ref);
        }
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("nope attention no absorb");
#endif
  }

  void output_absorb(int query, const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                     const std::vector<int>& kvlens) {
    auto pool = config.pool->get_subpool(tp_part_idx);
    {
      // by page
      size_t page_count = div_up((size_t)kvlens[query] + qlens[query], config.token_count_in_page);
      for (int kv_page = 0; kv_page < page_count; kv_page++) {
        // o absorb

        size_t kvlen_begin = kv_page * config.token_count_in_page;
        size_t kvlen_end = std::min(kvlen_begin + config.token_count_in_page, (size_t)kvlens[query] + qlens[query]);

        size_t page_kv_len = kvlen_end - kvlen_begin;

        size_t qlen_block = div_up((size_t)qlens[query] * config.num_heads, row_block);
        size_t kv_rank_block = div_up(config.kv_lora_rank, col_block);
        auto task_counter = TaskCounter({qlen_block, kv_rank_block});
        auto task = [&](int task_id) {
          auto [qlen_block_idx, kv_rank_block_idx] = task_counter.get<2>(task_id);

          size_t qlen_begin = qlen_block_idx * row_block;
          size_t qlen_end = std::min(qlen_begin + row_block, (size_t)qlens[query] * config.num_heads);
          size_t kv_rank_begin = kv_rank_block_idx * col_block;
          size_t kv_rank_end = std::min(kv_rank_begin + col_block, (size_t)config.kv_lora_rank);
          KMatRef kv_lora_page_ref = kv_lora_page_refs[page_tables[query][kv_page]];
          kv_lora_page_ref = kv_lora_page_ref.offset_block(kv_rank_begin, 0, kv_rank_end - kv_rank_begin, page_kv_len);

          KMatRef this_attention_weights_ref =
              attention_weights_ref.offset_block(kvlen_begin, qlen_begin, page_kv_len, qlen_end - qlen_begin);

          KMatRef this_o_absorb_ref =
              o_absorb_ref.offset_block(kv_rank_begin, qlen_begin, kv_rank_end - kv_rank_begin, qlen_end - qlen_begin);
          if (kv_page == 0) {
            arm_kml::mul_mat_clearc(kv_lora_page_ref, this_attention_weights_ref, this_o_absorb_ref);
          } else {
            arm_kml::mul_mat(kv_lora_page_ref, this_attention_weights_ref, this_o_absorb_ref);
          }
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }
    }

#ifdef DEBUG_THIS_MLA
    printf("o absorb[%d]\n", tp_part_idx);
    for (size_t i = 0; i < config.num_heads; i++)
      dump_bin(file_name + "_o_absorb_" + std::to_string(i), (A*)o_absorb_or_v, config.kv_lora_rank * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("o absorb");
#endif

    {
      // attention output
      auto qlen_block = div_up((size_t)qlens[query], col_block);
      auto nope_block = div_up((size_t)config.nope_size, row_block);
      auto task_counter = TaskCounter({config.num_heads, qlen_block, nope_block});
      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t qlen_block_idx = task_counter.at(task_id, 1);
        size_t nope_block_idx = task_counter.at(task_id, 2);

        size_t qlen_begin = qlen_block_idx * col_block;
        size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlens[query]);
        size_t nope_begin = nope_block_idx * row_block;
        size_t nope_end = std::min(nope_begin + row_block, (size_t)config.nope_size);

        KMatRef this_local_v_b_proj_ref =
            local_v_b_proj_ref.offset_row(head_idx * config.nope_size + nope_begin, nope_end - nope_begin);

        KMatRef this_o_absorb_ref =
            o_absorb_ref.offset_col(head_idx * qlens[query] + qlen_begin, qlen_end - qlen_begin);

        KMatRef this_attention_output_ref = attention_output_ref.offset_block(
            head_idx * config.nope_size + nope_begin, qlen_begin, nope_end - nope_begin, qlen_end - qlen_begin);

        arm_kml::mul_mat_clearc(this_local_v_b_proj_ref, this_o_absorb_ref, this_attention_output_ref);
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }
#ifdef DEBUG_THIS_MLA
    printf("attention output[%d]\n", tp_part_idx);
    dump_bin(file_name + "_attention_output", (A*)attention_output, config.num_heads * config.nope_size * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("attention output");
#endif
  }

  void output_no_absorb(int query, const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                        const std::vector<int>& kvlens) {
    auto pool = config.pool->get_subpool(tp_part_idx);
    {
      // v
      size_t page_count = div_up((size_t)kvlens[query] + qlens[query], config.token_count_in_page);
      size_t nope_block_count = div_up((size_t)config.nope_size, row_block);
      size_t kvlen_in_page_count = div_up(config.token_count_in_page, col_block);
      ASSERT_RELEASE(config.token_count_in_page % col_block == 0, "token_count_in_page must be divisible by col_block");

      auto task_counter = TaskCounter({config.num_heads, page_count, kvlen_in_page_count, nope_block_count});
      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t page_idx = task_counter.at(task_id, 1);
        size_t kvlen_idx = task_counter.at(task_id, 2);
        size_t nope_block_idx = task_counter.at(task_id, 3);

        size_t kvlen_begin = page_idx * config.token_count_in_page + kvlen_idx * col_block;
        size_t kvlen_end = std::min(kvlen_begin + col_block, (size_t)kvlens[query] + qlens[query]);
        if (kvlen_begin >= kvlen_end) return;  // skip the extra block

#ifdef DEBUG_THIS_MLA
        printf("v nope[%d] %d %d %d %d\n", tp_part_idx, head_idx, page_idx, kvlen_begin, kvlen_end);
#endif

        size_t kvlen_begin_in_page = kvlen_begin % config.token_count_in_page;

        size_t nope_begin = nope_block_idx * row_block;
        size_t nope_end = std::min(nope_begin + row_block, (size_t)config.nope_size);

        KMatRef this_local_v_b_proj_ref =
            local_v_b_proj_ref.offset_row(head_idx * config.nope_size + nope_begin, nope_end - nope_begin);
        KMatRef kv_lora_page_ref = kv_lora_page_refs[page_tables[query][page_idx]];
        kv_lora_page_ref =
            kv_lora_page_ref.offset_block(0, kvlen_begin_in_page, config.kv_lora_rank, kvlen_end - kvlen_begin);
        KMatRef this_v_ref = v_ref.offset_block(nope_begin, head_idx * config.max_kvlen + kvlen_begin,
                                                nope_end - nope_begin, kvlen_end - kvlen_begin);
        arm_kml::mul_mat_clearc(this_local_v_b_proj_ref, kv_lora_page_ref, this_v_ref);
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }

#ifdef DEBUG_THIS_MLA
    printf("v nope[%d] done\n", tp_part_idx);
#endif
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("v nope");
#endif

    {
      // attn output
      size_t nope_block_count = div_up((size_t)config.nope_size, row_block);
      size_t qlen_block_count = div_up((size_t)qlens[query], col_block);
      auto task_counter = TaskCounter({config.num_heads, nope_block_count, qlen_block_count});
      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t nope_block_idx = task_counter.at(task_id, 1);
        size_t qlen_block_idx = task_counter.at(task_id, 2);
        size_t nope_begin = nope_block_idx * row_block;
        size_t nope_end = std::min(nope_begin + row_block, (size_t)config.nope_size);
        size_t qlen_begin = qlen_block_idx * col_block;
        size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlens[query]);
        if (qlen_begin >= qlen_end) return;  // skip the extra block

        KMatRef this_v_ref = v_ref.offset_block(nope_begin, head_idx * config.max_kvlen, nope_end - nope_begin,
                                                kvlens[query] + qlens[query]);

        KMatRef this_attention_weights_ref = attention_weights_ref.offset_col(head_idx * qlens[query], qlens[query]);
        this_attention_weights_ref =
            this_attention_weights_ref.offset_block(0, qlen_begin, kvlens[query] + qlens[query], qlen_end - qlen_begin);

        KMatRef this_attention_output_ref = attention_output_ref.offset_block(
            head_idx * config.nope_size + nope_begin, qlen_begin, nope_end - nope_begin, qlen_end - qlen_begin);
#ifdef DEBUG_THIS_MLA
        printf("attn output no absorb[%d] %d %d %d %d, %lx, %lx, %lx\n", tp_part_idx, head_idx, nope_begin, qlen_begin,
               qlen_end, v_nope_ref.data, attention_weights_ref.data, this_attention_output_ref.data);
#endif

        arm_kml::mul_mat_clearc(this_v_ref, this_attention_weights_ref, this_attention_output_ref);
      };
      pool->do_work_stealing_job(task_counter.count(), task);
    }
#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("attn output no absorb");
#endif
  }

  void forward_decode(const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                      const std::vector<int>& kvlens, const std::vector<void*>& attention_masks,
                      const input_t* input_raw, output_t* output_raw) {}

  static bool decide_absorb(size_t qlen, size_t existing_kvlen) {
    double x = existing_kvlen;
    return qlen < (-x + sqrt(x * (x + 2048.0 / 3.0)) / 2.0);
  }

  void forward_prefill(const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                       const std::vector<int>& kvlens, const std::vector<void*>& attention_masks,
                       const input_t* input_raw, output_t* output_raw) {
#ifdef DEBUG_THIS_MLA
    printf("input raw[%d]\n", tp_part_idx);
#endif

#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif

    auto pool = config.pool->get_subpool(tp_part_idx);

    std::vector<int> qlen_split, total_len_split;
    qlen_split.reserve(qlens.size() + 1);
    qlen_split.push_back(0);
    total_len_split.reserve(qlens.size() + 1);
    int qlen_sum = 0;
    int total_len_sum = 0;
    for (size_t i = 0; i < qlens.size(); i++) {
      qlen_sum += qlens[i];
      qlen_split.push_back(qlen_sum);

      total_len_sum += qlens[i] + kvlens[i];
      total_len_split.push_back(total_len_sum);
    }

    auto input_ref =
        KMatRef(const_cast<A*>(input_raw), config.hidden_size, qlen_sum, config.hidden_size, CblasColMajor);
    auto output_ref = KMatRef(output_raw, config.hidden_size, qlen_sum, config.hidden_size, CblasColMajor);
    auto q_lora_rank_ref = KMatRef((A*)q_lora_rank, config.q_lora_rank, qlen_sum, config.q_lora_rank, CblasColMajor);

    for (int query = 0; query < qlens.size(); query++) {
      bool use_absorb = decide_absorb(qlens[query], kvlens[query]);
      // bool use_absorb = true;
      // bool use_absorb = false;
      auto total_len = qlens[query] + kvlens[query];
      size_t query_page_count = div_up((size_t)total_len, config.token_count_in_page);

#ifdef DEBUG_THIS_MLA
      file_name = std::to_string(query);
      file_name = "query_" + file_name + "_tp_" + std::to_string(tp_part_idx);

      if (tp_part_idx == 0) {
        printf("qlen %d, kvlen %d, page table: ", qlens[query], kvlens[query]);
        for (auto x : page_tables[query]) {
          printf(" %d,", x);
        }
        printf("\n");
      }
#endif

      KMatRef qlen_input_ref = input_ref.offset_block(0, qlen_split[query], config.hidden_size, qlens[query]);
      {
        // q lora, kv lora, rope
        size_t cc_page_begin = (kvlens[query]) / config.token_count_in_page;
        size_t cc_page_end = div_up((size_t)kvlens[query] + qlens[query], config.token_count_in_page);
        size_t block_per_page = div_up(config.token_count_in_page, col_block);

        size_t q_lora_rank_block_count = div_up((size_t)config.q_lora_rank, row_block);
        size_t kv_lora_rank_block_count = div_up((size_t)config.kv_lora_rank, row_block);
        size_t k_rope_block_count = div_up((size_t)config.rope_size, row_block);
        TaskCounter task_counter({cc_page_end - cc_page_begin, block_per_page,
                                  q_lora_rank_block_count + kv_lora_rank_block_count + k_rope_block_count});

        auto task = [&](int task_id) {
          size_t cc_page = task_counter.at(task_id, 0) + cc_page_begin;
          size_t in_page_block_idx = task_counter.at(task_id, 1);
          size_t kvlen_begin = std::clamp(cc_page * config.token_count_in_page + in_page_block_idx * col_block,
                                          (size_t)kvlens[query], (size_t)kvlens[query] + qlens[query]);
          size_t kvlen_end = std::clamp(cc_page * config.token_count_in_page + (in_page_block_idx + 1) * col_block,
                                        (size_t)kvlens[query], (size_t)kvlens[query] + qlens[query]);
          // printf("kvlen[%d,%d)\n", kvlen_begin, kvlen_end);

          size_t kvlen_block_size = kvlen_end - kvlen_begin;
          if (kvlen_block_size == 0) {
            return;
          }

          size_t qlen_begin = kvlen_begin - kvlens[query];
          size_t qlen_end = kvlen_end - kvlens[query];

          auto blocked_input = qlen_input_ref.offset_block(0, qlen_begin, config.hidden_size, qlen_end - qlen_begin);

          int q_or_kv_or_krope = task_counter.at(task_id, 2);

          // #ifdef DEBUG_THIS_MLA
          //           if (tp_part_idx == 0) {
          //             printf("task id: %d, q_or_kv_or_krope: %d, kv_page %d, kvlend [%d,%d)
          //             \n",task_id,q_or_kv_or_krope,kv_page, kvlen_begin, kvlen_end);
          //           }
          // #endif

          if (q_or_kv_or_krope < q_lora_rank_block_count) {
            size_t q_lora_rank_block_idx = q_or_kv_or_krope;
            size_t q_lora_rank_begin = q_lora_rank_block_idx * row_block;
            size_t q_lora_rank_end = std::min(config.q_lora_rank, q_lora_rank_begin + row_block);

            arm_kml::mul_mat_clearc(
                local_q_a_proj_ref.offset_block(q_lora_rank_begin, 0, q_lora_rank_end - q_lora_rank_begin,
                                                config.hidden_size),
                blocked_input,
                q_lora_rank_ref.offset_block(q_lora_rank_begin, qlen_begin, q_lora_rank_end - q_lora_rank_begin,
                                             qlen_end - qlen_begin));

          } else if (q_or_kv_or_krope < q_lora_rank_block_count + kv_lora_rank_block_count) {
            size_t kv_lora_rank_block_idx = q_or_kv_or_krope - q_lora_rank_block_count;
            size_t kv_lora_rank_begin = kv_lora_rank_block_idx * row_block;
            size_t kv_lora_rank_end = std::min(config.kv_lora_rank, kv_lora_rank_begin + row_block);
            KMatRef kv_lora_page_ref = kv_lora_page_refs[page_tables[query][cc_page]].offset_block(
                kv_lora_rank_begin, kvlen_begin % config.token_count_in_page, kv_lora_rank_end - kv_lora_rank_begin,
                kvlen_end - kvlen_begin);

            arm_kml::mul_mat_clearc(
                local_kv_a_proj_with_mqa_ref.offset_block(kv_lora_rank_begin, 0, kv_lora_rank_end - kv_lora_rank_begin,
                                                          config.hidden_size),
                blocked_input, kv_lora_page_ref);

          } else if (q_or_kv_or_krope < q_lora_rank_block_count + kv_lora_rank_block_count + k_rope_block_count) {
            // single block for k rope, no norm
            size_t rope_block_idx = q_or_kv_or_krope - q_lora_rank_block_count - kv_lora_rank_block_count;
            size_t rope_begin = rope_block_idx * row_block;
            size_t rope_end = std::min(config.rope_size, rope_begin + row_block);

            KMatRef rope_page_ref = rope_page_refs[page_tables[query][cc_page]].offset_block(
                rope_begin, kvlen_begin % config.token_count_in_page, rope_end - rope_begin, kvlen_end - kvlen_begin);

            arm_kml::mul_mat_clearc(local_kv_a_proj_with_mqa_ref.offset_block(
                                        config.kv_lora_rank + rope_begin, 0, rope_end - rope_begin, config.hidden_size),
                                    blocked_input, rope_page_ref);
            T_RopeApplier::apply_multiple(*rope_angle, rope_page_ref.data, config.rope_size, rope_page_ref.ld,
                                          kvlen_begin, kvlen_block_size);
          } else {
            throw std::runtime_error("task id wrong");
          }
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }
#ifdef DEBUG_THIS_MLA
      printf("q lora, kv lora, rope[%d]\n", tp_part_idx);
      dump_bin(file_name + "_input.bin", qlen_input_ref.data, qlens[query] * config.hidden_size);
      dump_bin(file_name + "_qlora.bin", q_lora_rank, qlens[query] * config.q_lora_rank);

      for (int i = 0; i < query_page_count; i++) {
        dump_bin(file_name + "_page_" + std::to_string(i) + "_cc_pages", (A*)cc_pages[page_tables[query][i]],
                 config.token_count_in_page * cc_size);
      }
#endif

#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("q lora, kv lora, rope");
#endif

      {
        // norm
        auto task_counter = TaskCounter({2, (size_t)qlens[query]});
        auto task = [&](int task_id) {
          int q_or_k_kpe = task_counter.at(task_id, 0);
          size_t qlen_idx = task_counter.at(task_id, 1);
          if (q_or_k_kpe == 0) {
            T_RMSNorm::rms_norm_single_with_weights(config.q_lora_rank, local_q_a_norm,
                                                    q_lora_rank_ref.offset_col(qlen_idx, 1).data);
          } else if (q_or_k_kpe == 1) {
            auto kv_page = (qlen_idx + kvlens[query]) / config.token_count_in_page;
            auto token_at_in_page = (qlen_idx + kvlens[query]) % config.token_count_in_page;
            KMatRef kv_lora_page_ref = kv_lora_page_refs[page_tables[query][kv_page]];
            T_RMSNorm::rms_norm_single_with_weights(config.kv_lora_rank, local_kv_a_norm,
                                                    kv_lora_page_ref.offset_col(token_at_in_page, 1).data);
          } else {
            throw std::runtime_error("unknown task");
          }
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }
#ifdef DEBUG_THIS_MLA
      printf("q lora norm[%d]\n", tp_part_idx);
      dump_bin(file_name + "_qlora_norm.bin", q_lora_rank, qlens[query] * config.q_lora_rank);
      // for (int i = 0; i < query_page_count; i++) {
      //   dump_bin(file_name + "_page_" + std::to_string(i) + "_kv_lora_rank_norm",
      //            (A *)kv_lora_pages[page_tables[query][i]], config.token_count_in_page * config.kv_lora_rank);
      // }
#endif
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("q/kv lora norm");
#endif

      {
        // q rope & nope
        size_t qlen_block = div_up((size_t)qlens[query], col_block);
        TaskCounter task_counter({config.num_heads, 2, qlen_block});

        auto task = [&](int task_id) {
          auto head_idx = task_counter.at(task_id, 0);
          bool nope_or_rope = (task_counter.at(task_id, 1) == 0);
          auto qlen_block_idx = task_counter.at(task_id, 2);
          size_t qlen_begin = qlen_block_idx * col_block;
          size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlens[query]);

          auto b = q_lora_rank_ref.offset_block(0, qlen_begin, config.q_lora_rank, qlen_end - qlen_begin);
          if (nope_or_rope) {
            auto a = local_q_b_proj_ref.offset_row(head_idx * (config.nope_size + config.rope_size), config.nope_size);
            KMatRef c = use_absorb ? q_nope_tmp_ref : q_nope_ref;

            c = c.offset_col(head_idx * qlens[query] + qlen_begin, qlen_end - qlen_begin);

            arm_kml::mul_mat_clearc(a, b, c);
          } else {
            auto a = local_q_b_proj_ref.offset_row(head_idx * (config.nope_size + config.rope_size) + config.nope_size,
                                                   config.rope_size);
            KMatRef c = use_absorb ? q_pe_absorb_ref : q_pe_noabsorb_ref;
            c = c.offset_col(head_idx * qlens[query] + qlen_begin, qlen_end - qlen_begin);

            arm_kml::mul_mat_clearc(a, b, c);
            T_RopeApplier::apply_multiple(*rope_angle, c.data, config.rope_size, c.ld, qlen_begin + kvlens[query],
                                          qlen_end - qlen_begin);
          }
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }

#ifdef DEBUG_THIS_MLA
      printf("q nope/rope[%d]\n", tp_part_idx);
      dump_bin(file_name + "_q_nope", q_nope, config.nope_size * qlens[query]);
      dump_bin(file_name + "_q", q, q_ld * qlens[query]);
      dump_bin(file_name + "_rope_cos", rope_angle->cos(0), config.rope_size / 2 * qlens[query]);
      dump_bin(file_name + "_rope_sin", rope_angle->sin(0), config.rope_size / 2 * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("q nope/rope");
#endif

      if (use_absorb) {
        nope_attention_q_absorb(qlens[query], kvlens[query], page_tables[query], false);
      } else {
        nope_attention_no_absorb(qlens[query], kvlens[query], page_tables[query], false);
      }
#ifdef DEBUG_THIS_MLA
      printf("attention weights[%d] \n", tp_part_idx);
      for (size_t i = 0; i < config.num_heads; i++)
        dump_bin(file_name + "_raw_attention_weights_" + std::to_string(i),
                 attention_weights_ref.offset_col(i * qlens[query], qlens[query]).data,
                 config.max_kvlen * qlens[query]);
#endif
      {
        // attentino mask & soft max
        auto task_counter = TaskCounter({config.num_heads, (size_t)qlens[query]});
        auto task = [&](int task_id) {
          size_t head_idx = task_counter.at(task_id, 0);
          size_t qlen_idx = task_counter.at(task_id, 1);
          size_t qlen_from_start = qlen_idx + kvlens[query];
          A* aw =
              offset_pointer(attention_weights,
                             (config.max_kvlen * qlens[query] * head_idx + config.max_kvlen * qlen_idx) * sizeof(A));
          for (int i = 0; i < kvlens[query] + qlens[query]; i++) {
            aw[i] *= softmax_scale;
            aw[i] += static_cast<A*>(attention_masks[qlen_from_start])[i];
          }

          T_SoftmaxApplier::apply_single(aw, kvlens[query] + qlens[query]);
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }

#ifdef DEBUG_THIS_MLA
      printf("attention weights after softmax[%d] \n", tp_part_idx);
      for (size_t i = 0; i < config.num_heads; i++)
        dump_bin(file_name + "_attention_weights_" + std::to_string(i),
                 attention_weights_ref.offset_col(i * qlens[query], qlens[query]).data,
                 config.max_kvlen * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("attention mask & softmax");
#endif
      if (use_absorb) {
        output_absorb(query, qlens, page_tables, kvlens);
      } else {
        output_no_absorb(query, qlens, page_tables, kvlens);
      }

#ifdef DEBUG_THIS_MLA
      printf("attn output done[%d]\n", tp_part_idx);

#endif
      {
        // output
        KMatRef reduce_qlen_output_ref = output_ref.offset_col(qlen_split[query], qlens[query]);
        auto qlen_block = div_up((size_t)qlens[query], col_block);
        auto hidden_size_block = div_up((size_t)config.hidden_size, row_block);

        for (size_t mhead_idx = 0; mhead_idx < config.num_heads / sub_num_heads; mhead_idx++) {
          auto task_counter = TaskCounter({qlen_block, hidden_size_block});

          auto task = [&](int task_id) {
            size_t head_begin = config.nope_size * mhead_idx * sub_num_heads;
            size_t head_end = head_begin + config.nope_size * sub_num_heads;

            size_t qlen_block_idx = task_counter.at(task_id, 0);
            size_t hidden_size_block_idx = task_counter.at(task_id, 1);

            size_t qlen_begin = qlen_block_idx * col_block;
            size_t qlen_end = std::min(qlen_begin + col_block, (size_t)qlens[query]);
            size_t hidden_size_begin = hidden_size_block_idx * row_block;
            size_t hidden_size_end = std::min(hidden_size_begin + row_block, (size_t)config.hidden_size);

            KMatRef this_local_w_o_ref = local_w_o_ref.offset_block(
                hidden_size_begin, head_begin, hidden_size_end - hidden_size_begin, head_end - head_begin);

            KMatRef this_attention_output_ref =
                attention_output_ref.offset_block(head_begin, qlen_begin, head_end - head_begin, qlen_end - qlen_begin);
            // KMatRef qlen_output_ref =
            //     KMatRef(qlen_output, config.hidden_size, qlens[query], config.hidden_size, CblasColMajor);
            KMatRef this_qlen_output_ref = reduce_qlen_output_ref.offset_block(
                hidden_size_begin, qlen_begin, hidden_size_end - hidden_size_begin, qlen_end - qlen_begin);
#ifdef DEBUG_THIS_MLA
            printf("output by head[%d] %d (%d,%d) (%d,%d)\n", tp_part_idx, mhead_idx, qlen_begin, qlen_end,
                   hidden_size_begin, hidden_size_end);
            // flush stdout
            fflush(stdout);
#endif
            if (mhead_idx == 0) {
              arm_kml::mul_mat_clearc(this_local_w_o_ref, this_attention_output_ref, this_qlen_output_ref);
            } else {
              arm_kml::mul_mat(this_local_w_o_ref, this_attention_output_ref, this_qlen_output_ref);
            }
          };
          pool->do_work_stealing_job(task_counter.count(), task);
        }
      }

#ifdef DEBUG_THIS_MLA
      printf("output by head done[%d]\n", tp_part_idx);

      dump_bin(file_name + "_local_w_o", local_w_o, config.hidden_size * config.nope_size * config.num_heads);
      // dump_bin(file_name + "_qlen_output", (A *)qlen_output, qlens[query] * config.hidden_size);

#endif
#ifdef FORWARD_TIME_PROFILE
      PROFILE_RECORD_TIME_STAMP("output by head");
#endif

      // {
      // merge output
      // KMatRef reduce_qlen_output_ref = output_ref.offset_block(0, qlen_split[query], config.hidden_size,
      // qlens[query]);

      // const size_t sum_block = 1024;
      // const size_t sum_block_count = div_up(config.hidden_size, sum_block);
      // auto task_counter = TaskCounter({sum_block_count, (size_t)qlens[query]});
      // pool->do_work_stealing_job(task_counter.count(), [&](int task_id) {
      //   size_t hidden_idx = task_counter.at(task_id, 0);
      //   size_t hidden_begin = hidden_idx * sum_block;
      //   size_t hidden_end = std::min(hidden_begin + sum_block, (size_t)config.hidden_size);
      //   size_t qlen_idx = task_counter.at(task_id, 1);
      //   reduce_sum(qlen_output.data(), qlen_output.size(), qlen_idx * config.hidden_size + hidden_begin,
      //              qlen_idx * config.hidden_size + hidden_end);
      // });

      // memcpy(reduce_qlen_output_ref.data, qlen_output[0], qlens[query] * config.hidden_size * sizeof(A));

      //   pool->do_work_stealing_job(qlens[query], [&](int token_nth) {
      //     memcpy(&reduce_qlen_output_ref.data[token_nth * config.hidden_size],
      //            &qlen_output[token_nth * config.hidden_size], config.hidden_size * sizeof(A));
      //   });
      // }
      // #ifdef FORWARD_TIME_PROFILE
      //       PROFILE_RECORD_TIME_STAMP("merge output tp");
      // #endif

#ifdef FORWARD_TIME_PROFILE
      time_perf_name = "[mla] layer " + std::to_string(config.layer_idx) +
                       " tp_part_idx: " + std::to_string(tp_part_idx) + ", query: " + std::to_string(query);
      perf_report();

#endif
    }
  }
};
template <typename A>
class TP_MLA<KML_MLA_TP<A>> : public TP_MLA_Common<KML_MLA_TP<A>> {
 public:
  using TP_MLA_Common<KML_MLA_TP<A>>::TP_MLA_Common;

  void load_weights() {
    auto pool = this->config.pool;
    auto tp_num_heads = this->config.num_heads / this->tp_count;
    pool->dispense_backend()->do_numa_job([this, pool, tp_num_heads](int tp_id) {
      this->tps[tp_id]->load_weights(this->config.num_heads, tp_id * tp_num_heads);
    });
    this->weights_loaded = true;
  }

  void merge_results(int qlen, void* output_raw) {
    auto pool = this->config.pool;
    typename KML_MLA_TP<A>::output_t* output = (typename KML_MLA_TP<A>::output_t*)output_raw;
    pool->do_work_stealing_job(qlen, [this, output](int token_nth) {
      auto& tp_count = this->tp_count;
      auto& config = this->config;
      auto& local_output_numa = this->local_output_numa;
      reduce_sum(local_output_numa.data(), tp_count, token_nth * config.hidden_size,
                 token_nth * config.hidden_size + config.hidden_size);
      memcpy(&output[token_nth * config.hidden_size], &local_output_numa[0][token_nth * config.hidden_size],
             config.hidden_size * sizeof(output[0]));
    });

#ifdef DEBUG_THIS_MLA
    dump_bin("output.bin", output, qlen * this->config.hidden_size);

#endif
  }
};

#endif
