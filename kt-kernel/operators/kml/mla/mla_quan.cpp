#include "mla_quan.h"

#include "../mla-tp.hpp"
#include "../reduce.hpp"
#include "../rms-norm.hpp"
#include "../rope.hpp"
#include "../softmax.hpp"
#include "ggml-quants.h"
#include "ggml.h"
#include "kblas.h"
#include "la/arm_kml.hpp"

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
#include <stdexcept>
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

// inline void debug_output(const float16_t *data) {
//   for (int i = 0; i < 10; i++) {
//     printf("%f ", data[i]);
//   }
//   printf("\n");
// }

// inline void debug_output(const float *data) {
//   for (int i = 0; i < 10; i++) {
//     printf("%f ", data[i]);
//   }
//   printf("\n");
// }
// inline void debug_output(const bfloat16_t *data) {
//   for (int i = 0; i < 10; i++) {
//     float x = 0;
//     *(bfloat16_t *)(&x) = data[i];
//     printf("%f ", x);
//   }
//   printf("\n");
// }

// template <typename T> inline void dump_to(std::string file_name, T *data, size_t count) {
//   std::ofstream f(file_name);
//   for (int i = 0; i < count; i++) {
//     f << data[i] << " ";
//   }
//   f.close();
// }
template <typename A, class KERNEL>
KML_MLA_TP_QUAN_TEST<A, KERNEL>::KML_MLA_TP_QUAN_TEST(GeneralMLAConfig config, int tp_part_idx)
    : config(config), tp_part_idx(tp_part_idx) {
  init_ggml();
  cc_size = config.kv_lora_rank + config.rope_size;
  MemoryRequest mem_requests;
  softmax_scale = 1.0 / sqrt(config.rope_size + config.nope_size);
  PUSH_MEM_REQ(q_lora_rank,
               sizeof(std::remove_pointer_t<decltype(q_lora_rank)>) * config.q_lora_rank * config.max_qlen);
  // qlen_decode_output.resize(config.num_heads, nullptr);
  // for (int i = 0; i < config.num_heads; i++) {
  //   mem_requests.append_pointer(&qlen_decode_output[i], sizeof(std::remove_pointer_t<decltype(qlen_decode_output)>) *
  //                                                           config.hidden_size * config.max_qlen);
  // }
  qlen_quant_output.resize((config.num_heads + sub_num_heads_decode - 1) / sub_num_heads_decode, nullptr);
  for (int i = 0; i < (config.num_heads + sub_num_heads_decode - 1) / sub_num_heads_decode; i++) {
    mem_requests.append_pointer(&qlen_quant_output[i], sizeof(std::remove_pointer_t<decltype(qlen_quant_output)>) *
                                                           config.hidden_size * config.max_qlen * sub_num_heads_decode);
  }

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

  // local_q_a_proj_deprecated_ba = new GemmKernel::BufferA(config.q_lora_rank, config.hidden_size, false);
  // local_q_a_proj_deprecated_ba = new typename GemmKernel::BufferA(config.q_lora_rank, config.hidden_size, true);
  // local_q_a_proj_deprecated_bb = new typename GemmKernel::BufferB(config.max_qlen, config.hidden_size);
  // local_q_a_proj_deprecated_bc = new typename GemmKernel::BufferC(config.q_lora_rank, config.max_qlen);

  local_q_a_proj_quant_ba = new typename GemmKernel::BufferA(config.max_qlen, config.hidden_size);
  // local_q_a_proj_quant_bb = new GemmKernel::BufferB(config.q_lora_rank, config.hidden_size, false);
  local_q_a_proj_quant_bb = new typename GemmKernel::BufferB(config.q_lora_rank, config.hidden_size, true);
  local_q_a_proj_quant_bc = new typename GemmKernel::BufferC(config.max_qlen, config.q_lora_rank, true);  // row major

  mem_requests.append_function([this](void* new_ptr) { local_q_a_proj_quant_ba->set_data(new_ptr); },
                               local_q_a_proj_quant_ba->required_size());
  local_q_a_proj_quant_bb->set_data(std::aligned_alloc(64, local_q_a_proj_quant_bb->required_size()));
  mem_requests.append_function([this](void* new_ptr) { local_q_a_proj_quant_bc->set_data(new_ptr); },
                               local_q_a_proj_quant_bc->required_size());

  // local_kv_a_proj_with_mqa_deprecated_ba = new GemmKernel::BufferA(cc_size, config.hidden_size, false);
  // local_kv_a_proj_with_mqa_deprecated_ba = new typename GemmKernel::BufferA(cc_size, config.hidden_size, true);
  // local_kv_a_proj_with_mqa_deprecated_bb = new typename GemmKernel::BufferB(config.max_qlen, config.hidden_size);
  // for (int i = 0; i < config.page_count; i++) {
  // local_kv_a_proj_with_mqa_deprecated_bc.push_back(new
  //  typename GemmKernel::BufferC(cc_size, config.token_count_in_page));
  // }

  local_kv_a_proj_with_mqa_quant_ba = new typename GemmKernel::BufferA(config.max_qlen, config.hidden_size);
  // local_kv_a_proj_with_mqa_quant_bb = new GemmKernel::BufferB(cc_size, config.hidden_size, false);
  local_kv_a_proj_with_mqa_quant_bb = new typename GemmKernel::BufferB(cc_size, config.hidden_size, true);
  for (int i = 0; i < config.page_count; i++) {
    local_kv_a_proj_with_mqa_quant_bc.push_back(
        new typename GemmKernel::BufferC(config.token_count_in_page, cc_size, true));  // row major
  }

  mem_requests.append_function([this](void* new_ptr) { local_kv_a_proj_with_mqa_quant_ba->set_data(new_ptr); },
                               local_kv_a_proj_with_mqa_quant_ba->required_size());
  local_kv_a_proj_with_mqa_quant_bb->set_data(
      std::aligned_alloc(64, local_kv_a_proj_with_mqa_quant_bb->required_size()));

  cc_page_refs_buffer.resize(config.page_count);
  kv_lora_page_refs_buffer.resize(config.page_count);
  rope_page_refs_buffer.resize(config.page_count);

  cc_page_refs_decode_buffer.resize(config.page_count);
  kv_lora_page_refs_decode_buffer.resize(config.page_count);
  rope_page_refs_decode_buffer.resize(config.page_count);
  for (int i = 0; i < config.page_count; i++) {
    mem_requests.append_function(
        [this, i, config](void* new_ptr) {
          local_kv_a_proj_with_mqa_quant_bc[i]->set_data(new_ptr);
          cc_page_refs_buffer[i] = KMatRefC(local_kv_a_proj_with_mqa_quant_bc[i]->c, cc_size,
                                            config.token_count_in_page, cc_size, CblasColMajor);
          kv_lora_page_refs_buffer[i] = cc_page_refs_buffer[i].offset_row(0, config.kv_lora_rank);
          rope_page_refs_buffer[i] = cc_page_refs_buffer[i].offset_row(config.kv_lora_rank, config.rope_size);

          cc_page_refs_decode_buffer[i] = KMatRefC(local_kv_a_proj_with_mqa_quant_bc[i]->c, config.token_count_in_page,
                                                   cc_size, cc_size, CblasRowMajor);
          kv_lora_page_refs_decode_buffer[i] = cc_page_refs_decode_buffer[i].offset_col(0, config.kv_lora_rank);
          rope_page_refs_decode_buffer[i] =
              cc_page_refs_decode_buffer[i].offset_col(config.kv_lora_rank, config.rope_size);
        },
        local_kv_a_proj_with_mqa_quant_bc[i]->required_size());
  }

  // local_w_o_ba = new GemmKernel::BufferA(config.hidden_size, config.num_heads * config.nope_size, false);
  // local_w_o_ba = new typename GemmKernel::BufferA(config.hidden_size, config.num_heads * config.nope_size, true);
  // local_w_o_bb = new typename GemmKernel::BufferB(config.max_qlen, config.num_heads * config.nope_size);
  // local_w_o_bc = new typename GemmKernel::BufferC(config.hidden_size, config.max_qlen);
  for (int i = 0; i < div_up(config.num_heads, sub_num_heads_decode); i++) {
    local_w_o_decode_bc.push_back(new typename GemmKernel::BufferC(config.hidden_size, config.max_qlen));  // col major
    local_w_o_decode_bb.push_back(
        new typename GemmKernel::BufferB(config.hidden_size, sub_num_heads_decode * config.nope_size, true));
  }

  local_w_o_quant_ba = new typename GemmKernel::BufferA(config.max_qlen, config.num_heads * config.nope_size);
  // local_w_o_quant_bb = new GemmKernel::BufferB(config.hidden_size, config.num_heads * config.nope_size, false);
  local_w_o_quant_bb = new typename GemmKernel::BufferB(config.hidden_size, config.num_heads * config.nope_size, true);
  local_w_o_prefill_bc = new typename GemmKernel::BufferC(config.max_qlen, config.hidden_size, true);  // row major

  mem_requests.append_function([this](void* new_ptr) { local_w_o_quant_ba->set_data(new_ptr); },
                               local_w_o_quant_ba->required_size());
  local_w_o_quant_bb->set_data(std::aligned_alloc(64, local_w_o_quant_bb->required_size()));
  mem_requests.append_function([this](void* new_ptr) { local_w_o_prefill_bc->set_data(new_ptr); },
                               local_w_o_prefill_bc->required_size());
  for (int i = 0; i < div_up(config.num_heads, sub_num_heads_decode); i++) {
    mem_requests.append_function([this, i](void* new_ptr) { local_w_o_decode_bc[i]->set_data(new_ptr); },
                                 local_w_o_decode_bc[i]->required_size());
    local_w_o_decode_bb[i]->set_data(std::aligned_alloc(64, local_w_o_decode_bb[i]->required_size()));
  }

  shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
}

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::load_weights(int complete_num_heads, int offset) {
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

  convert_or_copy(local_q_b_proj,
                  offset_pointer(config.q_b_proj, offset * (config.nope_size + config.rope_size) * config.q_lora_rank *
                                                      ggml_type_size((ggml_type)config.q_b_proj_type)),
                  (ggml_type)config.q_b_proj_type,
                  config.num_heads * (config.nope_size + config.rope_size) * config.q_lora_rank);

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
  size_t sub_num_heads_decode_group = div_up(config.num_heads, sub_num_heads_decode);
  local_w_decode_o.resize(sub_num_heads_decode_group);
  for (size_t h = 0; h < sub_num_heads_decode_group; h++) {
    local_w_decode_o[h] = new A[config.hidden_size * sub_num_heads_decode * config.nope_size];
    for (size_t i = 0; i < config.hidden_size; i++) {
      convert_or_copy(local_w_decode_o[h] + i * config.nope_size * sub_num_heads_decode,
                      offset_pointer(config.o_proj, (i * complete_num_heads * config.nope_size +
                                                     (h * sub_num_heads_decode + offset) * config.nope_size) *
                                                        ggml_type_size((ggml_type)config.w_o_type)),
                      (ggml_type)config.w_o_type, config.nope_size * sub_num_heads_decode);
    }
  }

// 做量化
#ifdef DEBUG_THIS_MLA
  printf("KML_MLA_TP_QUAN_TEST::load_weights: local_q_a_proj_quant_bb quantization\n");
  if (tp_part_idx == 0) {
    dump_bin("local_q_a_proj_quant.bin", local_q_a_proj, config.hidden_size * config.q_lora_rank);
  }
#endif
  auto pool = config.pool->get_subpool(tp_part_idx);
  {
    size_t mth_q_a = GemmKernel::recommended_nth(config.q_lora_rank);
    size_t mth_kv_a = GemmKernel::recommended_nth(config.kv_lora_rank + config.rope_size);
    auto task_counter = TaskCounter({mth_q_a + mth_kv_a});
    auto task = [&](int task_id) {
      // 前 nth 是 local_q_a_proj, 后 nth 是 local_kv_a_proj_with_mqa
      size_t mth_idx = task_counter.at(task_id, 0);
      if (mth_idx < mth_q_a) {
        // local_q_a_proj_deprecated_ba->from_mat(config.q_lora_rank, (A *)local_q_a_proj, mth_idx, mth_q_a);
        local_q_a_proj_quant_bb->from_mat((A*)local_q_a_proj, mth_idx, mth_q_a, config.q_lora_rank);
      } else {
        mth_idx -= mth_q_a;
        // local_kv_a_proj_with_mqa_deprecated_ba->from_mat(config.kv_lora_rank + config.rope_size,
        //                                                  (A *)local_kv_a_proj_with_mqa, mth_idx, mth_kv_a);
        local_kv_a_proj_with_mqa_quant_bb->from_mat((A*)local_kv_a_proj_with_mqa, mth_idx, mth_kv_a,
                                                    config.kv_lora_rank + config.rope_size);
      }
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }
#ifdef DEBUG_THIS_MLA
  printf("KML_MLA_TP_QUAN_TEST::load_weights: local_w_o_quant_bb quantization\n");
#endif
  {
    size_t mth_w_o = GemmKernel::recommended_mth(config.hidden_size);
    auto task_counter = TaskCounter({mth_w_o});
    auto task = [&](int task_id) {
      size_t mth_idx = task_counter.at(task_id, 0);
      // local_w_o_ba->from_mat(config.hidden_size, (A *)local_w_o, mth_idx, mth_w_o);
      local_w_o_quant_bb->from_mat((A*)local_w_o, mth_idx, mth_w_o, config.hidden_size);
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }
#ifdef DEBUG_THIS_MLA
  printf("KML_MLA_TP_QUAN_TEST::load_weights: local_w_decode_o quantization\n");
#endif
  {
    size_t mth_w_o = GemmKernel::recommended_mth(config.hidden_size);
    auto task_counter = TaskCounter({sub_num_heads_decode_group, mth_w_o});
    auto task = [&](int task_id) {
      size_t h = task_counter.at(task_id, 0);
      size_t mth_idx = task_counter.at(task_id, 1);
      // local_w_o_decode_ba[h]->from_mat(config.hidden_size, (A *)local_w_decode_o[h], mth_idx, mth_w_o);
      local_w_o_decode_bb[h]->from_mat((A*)local_w_decode_o[h], mth_idx, mth_w_o, config.hidden_size);
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }

  local_q_a_proj_quant_ref =
      KMatRefB(local_q_a_proj_quant_bb->b, config.hidden_size, config.q_lora_rank, config.hidden_size, CblasColMajor,
               CblasNoTrans, local_q_a_proj_quant_bb->if_pack);

  // local_q_a_proj_ref = KMatRefAB(local_q_a_proj_deprecated_ba->a, config.q_lora_rank, config.hidden_size,
  //                                config.hidden_size, CblasRowMajor);

  local_q_b_proj_ref = KMatRef((A*)local_q_b_proj, config.num_heads * (config.nope_size + config.rope_size),
                               config.q_lora_rank, config.q_lora_rank, CblasRowMajor);

  local_kv_a_proj_with_mqa_decode_ref =
      KMatRefB(local_kv_a_proj_with_mqa_quant_bb->b, config.hidden_size, config.kv_lora_rank + config.rope_size,
               config.hidden_size, CblasColMajor, CblasNoTrans, local_kv_a_proj_with_mqa_quant_bb->if_pack);

  local_w_o_ref =
      KMatRefB(local_w_o_quant_bb->b, config.hidden_size, config.nope_size * config.num_heads,
               config.nope_size * config.num_heads, CblasRowMajor, CblasNoTrans, local_w_o_quant_bb->if_pack);
  delete[] local_w_o;
  delete[] local_q_a_proj;
  delete[] local_kv_a_proj_with_mqa;
  for (int i = 0; i < sub_num_heads_decode_group; i++) {
    delete[] local_w_decode_o[i];
  }
  local_w_decode_o.clear();
}
template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::set_pages(std::vector<void*> kv_lora_pages, std::vector<void*> pe_pages) {
  // this->kv_lora_pages = kv_lora_pages;
  // this->rope_pages = pe_pages;
}
template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::set_local_pages(int page_count) {
  cc_pages.resize(page_count);

  cc_page_refs.resize(page_count);
  kv_lora_page_refs.resize(page_count);
  rope_page_refs.resize(page_count);

  // cc_page_refs_buffer.resize(page_count);
  // kv_lora_page_refs_buffer.resize(page_count);
  // rope_page_refs_buffer.resize(page_count);

  cc_page_refs_decode_buffer.resize(page_count);
  kv_lora_page_refs_decode_buffer.resize(page_count);
  rope_page_refs_decode_buffer.resize(page_count);

  for (int i = 0; i < page_count; i++) {
    cc_pages[i] = new A[config.token_count_in_page * (config.kv_lora_rank + config.rope_size)];
    cc_page_refs[i] = KMatRef(cc_pages[i], cc_size, config.token_count_in_page, cc_size, CblasColMajor);
    // cc_page_refs_buffer[i] = KMatRefC(local_kv_a_proj_with_mqa_deprecated_bc[i]->c, cc_size,
    // config.token_count_in_page,
    //                                   cc_size, CblasColMajor);
    kv_lora_page_refs[i] = cc_page_refs[i].offset_row(0, config.kv_lora_rank);
    // kv_lora_page_refs_buffer[i] = cc_page_refs_buffer[i].offset_row(0, config.kv_lora_rank);
    kv_lora_page_refs_decode_buffer[i] = cc_page_refs_decode_buffer[i].offset_col(0, config.kv_lora_rank);
    rope_page_refs[i] = cc_page_refs[i].offset_row(config.kv_lora_rank, config.rope_size);
    // rope_page_refs_buffer[i] = cc_page_refs_buffer[i].offset_row(config.kv_lora_rank, config.rope_size);
    rope_page_refs_decode_buffer[i] = cc_page_refs_decode_buffer[i].offset_col(config.kv_lora_rank, config.rope_size);
  }
}
template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables,
                                              std::vector<int> kv_lens, std::vector<void*> attention_masks,
                                              const void* input, void* output) {
  if (qlens[0] <= 1) {
    forward_decode(qlens, page_tables, kv_lens, attention_masks, (input_t*)input, (output_t*)output);
  } else {
    forward_prefill(qlens, page_tables, kv_lens, attention_masks, (input_t*)input, (output_t*)output);
  }
}

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables,
                                              std::vector<int> kv_lens, const void* input, void* output) {
  forward(qlens, page_tables, kv_lens, default_attention_masks, input, output);
}