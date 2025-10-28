// #define DEBUG_THIS_MLA
#ifdef DEBUG_THIS_MLA
#include "test/debug.hpp"
#endif
#include "mla_quan.h"

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::nope_attention_q_absorb(int qlen, int kvlen, const std::vector<int>& page_table,
                                                              bool increamental, bool is_decode) {
  // 使用 lambda 包装器
  auto mul_mat_clearc = [is_decode](auto a, auto b, auto c) {
    if (is_decode) {
      arm_kml::decode_mul_mat_clearc(a, b, c);
    } else {
      arm_kml::mul_mat_clearc(a, b, c);
    }
  };

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

      KMatRef this_local_k_b_proj_ref = local_k_b_proj_ref.offset_block(head_idx * config.nope_size, kv_rank_begin,
                                                                        config.nope_size, kv_rank_end - kv_rank_begin);
      this_local_k_b_proj_ref = this_local_k_b_proj_ref.t();
      // printf("q absorb %d [%d,%d),[%d,%d)\n", head_idx, qlen_begin, qlen_end, kv_rank_begin, kv_rank_end);
      mul_mat_clearc(this_local_k_b_proj_ref,
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
  if (is_decode) {
    PROFILE_RECORD_TIME_STAMP("decode q absorb");
  } else {
    PROFILE_RECORD_TIME_STAMP("prefill q absorb");
  }
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
  if (is_decode) {
    PROFILE_RECORD_TIME_STAMP("decode nope attention");
  } else {
    PROFILE_RECORD_TIME_STAMP("prefill nope attention");
  }
#endif
}

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::nope_attention_no_absorb(int qlen, int kvlen, const std::vector<int>& page_table,
                                                               bool increamental) {
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

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::output_absorb(int query, const std::vector<int>& qlens,
                                                    const std::vector<std::vector<int>>& page_tables,
                                                    const std::vector<int>& kvlens, bool is_decode) {
  // 使用 lambda 包装器
  auto mul_mat_clearc = [is_decode](auto a, auto b, auto c) {
    if (is_decode) {
      arm_kml::decode_mul_mat_clearc(a, b, c);
    } else {
      arm_kml::mul_mat_clearc(a, b, c);
    }
  };
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
  if (is_decode) {
    PROFILE_RECORD_TIME_STAMP("decode o absorb");
  } else {
    PROFILE_RECORD_TIME_STAMP("prefill o absorb");
  }
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

      KMatRef this_o_absorb_ref = o_absorb_ref.offset_col(head_idx * qlens[query] + qlen_begin, qlen_end - qlen_begin);

      KMatRef this_attention_output_ref = attention_output_ref.offset_block(
          head_idx * config.nope_size + nope_begin, qlen_begin, nope_end - nope_begin, qlen_end - qlen_begin);

      mul_mat_clearc(this_local_v_b_proj_ref, this_o_absorb_ref, this_attention_output_ref);
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }
#ifdef DEBUG_THIS_MLA
  printf("attention output[%d]\n", tp_part_idx);
  dump_bin(file_name + "_attention_output", (A*)attention_output, config.num_heads * config.nope_size * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
  if (is_decode) {
    PROFILE_RECORD_TIME_STAMP("decode attention output");
  } else {
    PROFILE_RECORD_TIME_STAMP("prefill attention output");
  }
#endif
}

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::output_no_absorb(int query, const std::vector<int>& qlens,
                                                       const std::vector<std::vector<int>>& page_tables,
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

      arm_kml::mul_mat_clearc(this_v_ref, this_attention_weights_ref, this_attention_output_ref);
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }
#ifdef DEBUG_THIS_MLA
  dump_bin(file_name + "_attention_output", (A*)attention_output, config.num_heads * config.nope_size * qlens[query]);
#endif
#ifdef FORWARD_TIME_PROFILE
  PROFILE_RECORD_TIME_STAMP("attn output no absorb");
#endif
}

template <typename A, class KERNEL>
void KML_MLA_TP_QUAN_TEST<A, KERNEL>::q_lora_kv_lora_rope_quant(int query, const std::vector<int>& qlens,
                                                                const std::vector<int>& kvlens,
                                                                const std::vector<std::vector<int>>& page_tables,
                                                                std::vector<int>& qlen_split, KMatRefA& input_ref,
                                                                KMatRefC& q_lora_rank_ref, KMatRef& q_lora_rank_out_ref,
                                                                bool is_decode) {
  // 使用 lambda 包装器
  auto mul_mat_clearc = [is_decode](auto a, auto b, auto c) {
    if (is_decode) {
      arm_kml::decode_mul_mat_clearc(a, b, c);
    } else {
      arm_kml::mul_mat_clearc(a, b, c);
    }
  };
  auto pool = config.pool->get_subpool(tp_part_idx);
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

  // KMatRef qlen_input_ref = input_ref.offset_block(0, qlen_split[query], config.hidden_size, qlens[query]);
  // KMatRefAB qlen_input_ref = input_ref.offset_block(0, qlen_split[query], config.hidden_size, qlens[query]);
  KMatRefA qlen_input_ref = input_ref.offset_row(qlen_split[query], qlens[query]);
  auto qlen_input_ba = local_q_a_proj_quant_ba->offset_row(qlen_split[query], qlens[query]);
  // auto local_q_a_proj_qlen_bb = local_q_a_proj_bb->offset_col(qlen_split[query], qlens[query]);
  {
    // q lora, kv lora, rope
    size_t cc_page_begin = (kvlens[query]) / config.token_count_in_page;
    size_t cc_page_end = div_up((size_t)kvlens[query] + qlens[query], config.token_count_in_page);
    size_t block_per_page = div_up(config.token_count_in_page, row_block_q_lora_kv_lora_rope);

    size_t q_lora_rank_block_count = div_up((size_t)config.q_lora_rank, col_block_q_lora_kv_lora_rope);
    size_t kv_lora_rank_block_count = div_up((size_t)config.kv_lora_rank, col_block_q_lora_kv_lora_rope);
    size_t k_rope_block_count = div_up((size_t)config.rope_size, col_block_q_lora_kv_lora_rope);
    TaskCounter task_counter({cc_page_end - cc_page_begin, block_per_page,
                              q_lora_rank_block_count + kv_lora_rank_block_count + k_rope_block_count});

    auto task = [&](int task_id) {
      size_t cc_page = task_counter.at(task_id, 0) + cc_page_begin;
      size_t in_page_block_idx = task_counter.at(task_id, 1);
      size_t kvlen_begin =
          std::clamp(cc_page * config.token_count_in_page + in_page_block_idx * row_block_q_lora_kv_lora_rope,
                     (size_t)kvlens[query], (size_t)kvlens[query] + qlens[query]);
      size_t kvlen_end =
          std::clamp(cc_page * config.token_count_in_page + (in_page_block_idx + 1) * row_block_q_lora_kv_lora_rope,
                     (size_t)kvlens[query], (size_t)kvlens[query] + qlens[query]);
      // printf("kvlen[%d,%d)\n", kvlen_begin, kvlen_end);

      size_t kvlen_block_size = kvlen_end - kvlen_begin;
      if (kvlen_block_size == 0) {
        return;
      }

      size_t qlen_begin = kvlen_begin - kvlens[query];
      size_t qlen_end = kvlen_end - kvlens[query];

      auto blocked_input = qlen_input_ref.offset_row(qlen_begin, qlen_end - qlen_begin);
      // auto blocked_input = qlen_input_ref.offset_block(0, qlen_begin, config.hidden_size, qlen_end - qlen_begin);
      // auto local_q_a_proj_qlen_blocked_bb = local_q_a_proj_qlen_bb.offset_col(qlen_begin, qlen_end - qlen_begin);

      int q_or_kv_or_krope = task_counter.at(task_id, 2);

      if (q_or_kv_or_krope < q_lora_rank_block_count) {
        size_t q_lora_rank_block_idx = q_or_kv_or_krope;
        size_t q_lora_rank_begin = q_lora_rank_block_idx * col_block_q_lora_kv_lora_rope;
        size_t q_lora_rank_end = std::min(config.q_lora_rank, q_lora_rank_begin + col_block_q_lora_kv_lora_rope);

        mul_mat_clearc(blocked_input,
                       local_q_a_proj_quant_ref.offset_col(q_lora_rank_begin, q_lora_rank_end - q_lora_rank_begin),
                       q_lora_rank_ref.offset_block(qlen_begin, q_lora_rank_begin, qlen_end - qlen_begin,
                                                    q_lora_rank_end - q_lora_rank_begin));

        GemmKernel::apply_scale(q_lora_rank_out_ref.data, q_lora_rank_out_ref.ld, &qlen_input_ba,
                                local_q_a_proj_quant_bb, local_q_a_proj_quant_bc, qlen_begin, qlen_end,
                                q_lora_rank_begin, q_lora_rank_end, true);
#ifdef DEBUG_THIS_MLA
        // 打印前十个q_lora_rank_ref，和前十个q_lora_rank_out_ref
        printf("q_lora_rank_begin:%d, q_lora_rank_end:%d\n", q_lora_rank_begin, q_lora_rank_end);
        printf("qlen_begin:%d, qlen_end:%d\n", qlen_begin, qlen_end);
        if (tp_part_idx == 0) {
          for (int i = 0; i < 10; i++) {
            printf("q_lora_rank_ref:%d, %f\n", q_lora_rank_ref.data[i], q_lora_rank_out_ref.data[i]);
          }
        }
#endif

      } else if (q_or_kv_or_krope < q_lora_rank_block_count + kv_lora_rank_block_count) {
        size_t kv_lora_rank_block_idx = q_or_kv_or_krope - q_lora_rank_block_count;
        size_t kv_lora_rank_begin = kv_lora_rank_block_idx * col_block_q_lora_kv_lora_rope;
        size_t kv_lora_rank_end = std::min(config.kv_lora_rank, kv_lora_rank_begin + col_block_q_lora_kv_lora_rope);
        KMatRefC kv_lora_page_ref = kv_lora_page_refs_decode_buffer[page_tables[query][cc_page]].offset_block(
            kvlen_begin % config.token_count_in_page, kv_lora_rank_begin, kvlen_end - kvlen_begin,
            kv_lora_rank_end - kv_lora_rank_begin);
        mul_mat_clearc(
            blocked_input,
            local_kv_a_proj_with_mqa_decode_ref.offset_col(kv_lora_rank_begin, kv_lora_rank_end - kv_lora_rank_begin),
            kv_lora_page_ref);
        KMatRef kv_lora_page_out_ref = kv_lora_page_refs[page_tables[query][cc_page]];
        GemmKernel::apply_scale(
            kv_lora_page_out_ref.data, kv_lora_page_out_ref.ld, &qlen_input_ba, local_kv_a_proj_with_mqa_quant_bb,
            kv_lora_page_refs_decode_buffer[page_tables[query][cc_page]].data, qlen_begin, qlen_end, kv_lora_rank_begin,
            kv_lora_rank_end, true, (kvlen_begin) % config.token_count_in_page - qlen_begin, 0);
      } else if (q_or_kv_or_krope < q_lora_rank_block_count + kv_lora_rank_block_count + k_rope_block_count) {
        // single block for k rope, no norm
        size_t rope_block_idx = q_or_kv_or_krope - q_lora_rank_block_count - kv_lora_rank_block_count;
        size_t rope_begin = rope_block_idx * col_block_q_lora_kv_lora_rope;
        size_t rope_end = std::min(config.rope_size, rope_begin + col_block_q_lora_kv_lora_rope);
        KMatRefC rope_page_ref = rope_page_refs_decode_buffer[page_tables[query][cc_page]].offset_block(
            kvlen_begin % config.token_count_in_page, rope_begin, kvlen_end - kvlen_begin, rope_end - rope_begin);

        mul_mat_clearc(
            blocked_input,
            local_kv_a_proj_with_mqa_decode_ref.offset_col(config.kv_lora_rank + rope_begin, rope_end - rope_begin),
            rope_page_ref);
        KMatRef rope_page_out_ref = rope_page_refs[page_tables[query][cc_page]];
        GemmKernel::apply_scale(rope_page_out_ref.data, rope_page_out_ref.ld, &qlen_input_ba,
                                local_kv_a_proj_with_mqa_quant_bb,
                                rope_page_refs_decode_buffer[page_tables[query][cc_page]].data, qlen_begin, qlen_end,
                                config.kv_lora_rank + rope_begin, config.kv_lora_rank + rope_end, true,
                                (kvlen_begin) % config.token_count_in_page - qlen_begin, -(config.kv_lora_rank));
        rope_page_out_ref = rope_page_out_ref.offset_block(rope_begin, kvlen_begin % config.token_count_in_page,
                                                           rope_end - rope_begin, kvlen_end - kvlen_begin);
        T_RopeApplier::apply_multiple(*rope_angle, rope_page_out_ref.data, config.rope_size, rope_page_out_ref.ld,
                                      kvlen_begin, kvlen_block_size);
      } else {
        throw std::runtime_error("task id wrong");
      }
    };
    pool->do_work_stealing_job(task_counter.count(), task);
  }
#ifdef DEBUG_THIS_MLA
  printf("q lora, kv lora, rope[%d]\n", tp_part_idx);
  // dump_bin(file_name + "_input.bin", qlen_input_ref.data, qlens[query] * config.hidden_size);
  dump_bin(file_name + "_qlora.bin", q_lora_rank, qlens[query] * config.q_lora_rank);

  for (int i = 0; i < query_page_count; i++) {
    dump_bin(file_name + "_page_" + std::to_string(i) + "_cc_pages", (A*)cc_pages[page_tables[query][i]],
             config.token_count_in_page * cc_size);
  }
#endif

#ifdef FORWARD_TIME_PROFILE
  PROFILE_RECORD_TIME_STAMP("q lora, kv lora, rope");
#endif
}