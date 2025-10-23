// #define DEBUG_THIS_MLA
#ifdef DEBUG_THIS_MLA
#include "test/debug.hpp"
#endif
#include <cstdio>

#include "la/arm_kml.hpp"
#include "mla_quan.h"

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::forward_prefill(const std::vector<int>& qlens,
                                                      const std::vector<std::vector<int>>& page_tables,
                                                      const std::vector<int>& kvlens,
                                                      const std::vector<void*>& attention_masks,
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

  // 输入进行量化,输入是 hidden_size * qlen 且是 col major
  {
    size_t nth = GemmKernel::recommended_nth(qlen_sum);
    auto task_counter = TaskCounter({nth});
    auto task = [&](int task_id) {
      size_t nth_idx = task_counter.at(task_id, 0);
      // local_q_a_proj_deprecated_bb->from_mat(const_cast<A *>(input_raw), nth_idx, nth, qlen_sum);
      local_q_a_proj_quant_ba->from_mat(qlen_sum, const_cast<A*>(input_raw), nth_idx, nth);
    };
    DIRECT_OR_POOL_BY(qlen_sum, 10, task_counter.count(), task);
  }
#ifdef FORWARD_TIME_PROFILE
  PROFILE_RECORD_TIME_STAMP("input_quant");
#endif
  auto input_ref =
      KMatRefA(local_q_a_proj_quant_ba->a, qlen_sum, config.hidden_size, config.hidden_size, CblasRowMajor);
  auto output_ref = KMatRef(output_raw, qlen_sum, config.hidden_size, config.hidden_size, CblasRowMajor);

  auto output_ref_buffer =
      KMatRefC(local_w_o_prefill_bc->c, qlen_sum, config.hidden_size, config.hidden_size, CblasRowMajor);
  KMatRefC q_lora_rank_ref =
      KMatRefC(local_q_a_proj_quant_bc->c, qlen_sum, config.q_lora_rank, config.q_lora_rank, CblasRowMajor);
  KMatRef q_lora_rank_out_ref =
      KMatRef((A*)q_lora_rank, qlen_sum, config.q_lora_rank, config.q_lora_rank, CblasRowMajor);

  for (int query = 0; query < qlens.size(); query++) {
    bool use_absorb = decide_absorb(qlens[query], kvlens[query]);
    q_lora_kv_lora_rope_quant(query, qlens, kvlens, page_tables, qlen_split, input_ref, q_lora_rank_ref,
                              q_lora_rank_out_ref);
    {
      // norm
      auto task_counter = TaskCounter({2, (size_t)qlens[query]});
      auto task = [&](int task_id) {
        int q_or_k_kpe = task_counter.at(task_id, 0);
        size_t qlen_idx = task_counter.at(task_id, 1);
        if (q_or_k_kpe == 0) {
          T_RMSNorm::rms_norm_single_with_weights(config.q_lora_rank, local_q_a_norm,
                                                  q_lora_rank_out_ref.offset_row(qlen_idx, 1).data);
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

        KMatRef b = q_lora_rank_out_ref.trans_view().offset_col(qlen_begin, qlen_end - qlen_begin);
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
    if (use_absorb) {
      dump_bin(file_name + "_q_nope", q_nope, config.nope_size * qlens[query]);
    } else {
      dump_bin(file_name + "_q_nope", q, q_ld * qlens[query]);
    }
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
               attention_weights_ref.offset_col(i * qlens[query], qlens[query]).data, config.max_kvlen * qlens[query]);
#endif
    {
      // attentino mask & soft max
      auto task_counter = TaskCounter({config.num_heads, (size_t)qlens[query]});
      auto task = [&](int task_id) {
        size_t head_idx = task_counter.at(task_id, 0);
        size_t qlen_idx = task_counter.at(task_id, 1);
        size_t qlen_from_start = qlen_idx + kvlens[query];
        A* aw = offset_pointer(attention_weights,
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
               attention_weights_ref.offset_col(i * qlens[query], qlens[query]).data, config.max_kvlen * qlens[query]);
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
    // 量化输入attention_output_ref->data (attention_output) [config.nope_size * config.num_heads, qlens[query]] col
    // major
    {
      size_t nth = GemmKernel::recommended_nth(qlens[query]);
      auto task_counter = TaskCounter({nth});
      auto task = [&](int task_id) {
        size_t nth_idx = task_counter.at(task_id, 0);
        // local_w_o_bb->from_mat(attention_output_ref.data, nth_idx, nth, qlens[query]);
        local_w_o_quant_ba->from_mat(qlens[query], attention_output_ref.data, nth_idx, nth);
      };
      DIRECT_OR_POOL_BY(qlens[query], 10, task_counter.count(), task);
    }

    {
      // output

      auto qlen_block = div_up((size_t)qlens[query], col_block_out_by_head);
      auto hidden_size_block = div_up((size_t)config.hidden_size, row_block_out_by_head);
      KMatRefA local_w_o_ba_ref = KMatRefA(local_w_o_quant_ba->a, config.nope_size * config.num_heads, qlens[query],
                                           config.nope_size * config.num_heads, CblasColMajor);
      KMatRef reduce_qlen_output_ref = output_ref.offset_row(qlen_split[query], qlens[query]).trans_view();
      KMatRefC reduce_qlen_output_ref_buffer =
          output_ref_buffer.offset_row(qlen_split[query], qlens[query]).trans_view();

      for (size_t mhead_idx = 0; mhead_idx < config.num_heads / sub_num_heads; mhead_idx++) {
        auto task_counter = TaskCounter({qlen_block, hidden_size_block});

        auto task = [&](int task_id) {
          size_t head_begin = config.nope_size * mhead_idx * sub_num_heads;
          size_t head_end = head_begin + config.nope_size * sub_num_heads;

          size_t qlen_block_idx = task_counter.at(task_id, 0);
          size_t hidden_size_block_idx = task_counter.at(task_id, 1);

          size_t qlen_begin = qlen_block_idx * col_block_out_by_head;
          size_t qlen_end = std::min(qlen_begin + col_block_out_by_head, (size_t)qlens[query]);
          size_t hidden_size_begin = hidden_size_block_idx * row_block_out_by_head;
          size_t hidden_size_end = std::min(hidden_size_begin + row_block_out_by_head, (size_t)config.hidden_size);
          KMatRefC this_qlen_output_ref_buffer = reduce_qlen_output_ref_buffer.offset_block(
              hidden_size_begin, qlen_begin, hidden_size_end - hidden_size_begin, qlen_end - qlen_begin);

          KMatRefB this_local_w_o_ref = local_w_o_ref.offset_block(
              hidden_size_begin, head_begin, hidden_size_end - hidden_size_begin, head_end - head_begin);
          KMatRefA this_attention_output_ref =
              local_w_o_ba_ref.offset_block(head_begin, qlen_begin, head_end - head_begin, qlen_end - qlen_begin);
          if (mhead_idx == 0) {
            // arm_kml::mul_mat_clearc(this_local_w_o_ref, this_attention_output_ref, this_qlen_output_ref_buffer);
            arm_kml::mul_mat_clearc(this_attention_output_ref.trans_view(), this_local_w_o_ref.trans_view(),
                                    this_qlen_output_ref_buffer.trans_view());
#ifdef DEBUG_THIS_MLA
            dump_bin(file_name + "_output_by_head_" + std::to_string(mhead_idx), this_qlen_output_ref_buffer.data,
                     qlens[query] * config.hidden_size);
            // printf("if pack: %d\n", this_local_w_o_ref.trans_view().if_pack);
#endif
          } else {
            // arm_kml::mul_mat(this_local_w_o_ref, this_attention_output_ref, this_qlen_output_ref_buffer);
            arm_kml::mul_mat(this_attention_output_ref.trans_view(), this_local_w_o_ref.trans_view(),
                             this_qlen_output_ref_buffer.trans_view());
#ifdef DEBUG_THIS_MLA
            dump_bin(file_name + "_output_by_head_" + std::to_string(mhead_idx), this_qlen_output_ref_buffer.data,
                     qlens[query] * config.hidden_size);
            // printf("if pack: %d\n", this_local_w_o_ref.trans_view().if_pack);
#endif
          }
          if (mhead_idx == config.num_heads / sub_num_heads - 1) {
            GemmKernel::apply_scale(reduce_qlen_output_ref.data, reduce_qlen_output_ref.ld, local_w_o_quant_ba,
                                    local_w_o_quant_bb, local_w_o_prefill_bc, qlen_begin, qlen_end, hidden_size_begin,
                                    hidden_size_end, true, qlen_split[query], 0);
            // GemmKernel::apply_scale(reduce_qlen_output_ref.data, reduce_qlen_output_ref.ld, local_w_o_quant_bb,
            //                         local_w_o_quant_ba, local_w_o_prefill_bc, hidden_size_begin, hidden_size_end,
            //                         qlen_begin, qlen_end, false, 0, qlen_split[query]);
          }
        };
        pool->do_work_stealing_job(task_counter.count(), task);
      }
    }

#ifdef DEBUG_THIS_MLA
    printf("output by head done[%d]\n", tp_part_idx);

    // dump_bin(file_name + "_local_w_o", local_w_o, config.hidden_size * config.nope_size * config.num_heads);
    dump_bin(file_name + "_qlen_output", (A*)output_raw, qlens[query] * config.hidden_size);

#endif

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("output by head");
#endif

#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "[mla] layer " + std::to_string(config.layer_idx) +
                     " tp_part_idx: " + std::to_string(tp_part_idx) + ", query: " + std::to_string(query);
    perf_report();

#endif
  }
}
