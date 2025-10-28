// #ifndef LLAMAFILE_MLA_HPP
// #define LLAMAFILE_MLA_HPP

// #include "../common.hpp"
// #include "../mla-tp.hpp"
// #include "../rms-norm.hpp"
// #include "../rope.hpp"
// #include "ggml-quants.h"
// #include "ggml.h"
// #include "llamafile/sgemm.h"

// #include <algorithm>
// #include <cstddef>
// #include <utility>
// #include <vector>

// #define DIRECT_OR_POOL_BY(what, threshold, var, fn)                                                                    \
//   do {                                                                                                                 \
//     if ((what) < (threshold)) {                                                                                        \
//       for (int i = 0; i < (var); i++) {                                                                                \
//         (fn)(i);                                                                                                       \
//       }                                                                                                                \
//     } else {                                                                                                           \
//       pool->do_work_stealing_job((var), nullptr, (fn), nullptr);                                                       \
//     }                                                                                                                  \
//   } while (0)

// #define VEC_DOT_TYPE(type) (ggml_internal_get_type_traits((ggml_type)(type)).vec_dot_type)
// #define QUANT_BLCK_COUNT(size, type) (((size_t)(size)) / (ggml_blck_size((ggml_type)(type))))
// #define QUANT_BLCK_SIZE(size, type) (QUANT_BLCK_COUNT(size, type) * (ggml_type_size((ggml_type)(type))))
// #define QUANT_OFFSET(ptr, type, n, n_elements) \
//   (offset_pointer((ptr), (size_t)(n) * QUANT_BLCK_SIZE((n_elements), (type))))

// #define LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(m, n, k, a, a_type, b, b_col, c, c_col)                                      \
//   do {                                                                                                                 \
//     llamafile_sgemm((m), (n), QUANT_BLCK_COUNT((k), (a_type)), (a), QUANT_BLCK_COUNT((k), (a_type)),                   \
//                     QUANT_OFFSET((b), VEC_DOT_TYPE((a_type)), (b_col), (k)),                                           \
//                     QUANT_BLCK_COUNT((k), VEC_DOT_TYPE((a_type))), offset_pointer((c), (c_col) * (m) * sizeof(float)), \
//                     (k), 0, 1, GGML_TASK_TYPE_COMPUTE, (a_type), VEC_DOT_TYPE((a_type)), GGML_TYPE_F32,                \
//                     GGML_PREC_DEFAULT);                                                                                \
//   } while (0)

// #define LLAMAFILE_SGEMM_MATMUL_F32(m, n, k, a, lda, b, ldb, c, ldc)                                                    \
//   do {                                                                                                                 \
//     llamafile_sgemm((m), (n), (k), (a), (lda), (b), (ldb), (c), (ldc), 0, 1, GGML_TASK_TYPE_COMPUTE, GGML_TYPE_F32,    \
//                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_PREC_DEFAULT);                                                  \
//   } while (0)

// // bool decide_absorb(size_t a,int a_type,size_t b,int b_type,size_t c,int c_type,size_t d,int d_type){
// //   size_t flops1 = ;

// // }

// inline void transpose(void *start, size_t dim0, size_t stride, size_t dim1) {
//   // static_assert(false, "TODO");
// }

// template <RMS_NORM T_RMSNorm = RMSNorm, ROPE_APPLIER T_RopeApplier = Rope, ROPE_ANGLE T_RopeAngle = Yarn>
// class LLAMA_MLA_TP {
// private:
//   GeneralMLAConfig config;
//   int tp_part_idx;
//   std::vector<void *> nope_pages;     // [page_count * page_token_count * nope]
//   std::vector<void *> rope_pages;     // [page_count * page_token_count * nope]

//   // weights
//   void *local_q_a_proj;               // [hidden_size * q_lora_rank]
//   void *local_q_a_norm;               // [q_lora_rank]
//   std::vector<void *> local_q_b_proj; // [num_heads * (nope_size + rope_size))]
//   void *local_kv_a_proj_with_mqa;     // [hidden_size * (kv_lora_rank + rope)]
//   void *local_kv_a_norm_with_mqa;
//   void *local_kv_b_proj;                   // [(num_heads * (nope_size + nope_size) * kv_lora_rank)],
//                                            // q_absorb:   [num_heads * nope_size * kv_lora_rank]
//                                            // out_absorb: [num_heads * nope_size * kv_lora_rank]
//   std::vector<void *> local_k_b_proj_nope; // [(num_heads * kv_lora_rank * nope)],
//   void *local_w_o; // [(num_heads * nope_size) * hidden_size]
//   T_RopeAngle rope_angle;

//   // intermediate

//   void *quant_input;           // [qlen, hidden size(Q)]
//   void *q_a_proj_output;       // [qlen, q_lora_rank]
//   void *quant_q_a_proj_output; // [qlen, q_lora_rank(Q)]

//   // for each query
//   std::vector<void *> q_pe;              // [num_heads * max_qlen * rope_size]
//   std::vector<void *> k_pe;              // [num_threads * rope_size]
//   std::vector<void *> q_nope;            // [num_heads * max_qlen * nope_size]
//   std::vector<void *> attention_weights; // [num_heads * max_qlen * max_klen];
//   std::vector<void *> q_absorb;          // [num_heads, max_qlen, kv_lora_rank],  or [num_heads, kv_lora_rank,
//   max_qlen] std::vector<void *> o_absorb;          // [num_heads, max_qlen, kv_lora_rank],  or [num_heads,
//   kv_lora_rank, max_qlen] std::vector<void *> compressed_kv_tmp; // [num_threads * token_count_in_page *
//   kv_lora_rank] std::vector<void *> quant_o_absorb;    // [num_heads, max_qlen, kv_lora_rank],  or [num_heads,
//   kv_lora_rank, max_qlen] std::vector<void *> attention_output;  // [num_threads * max_qlen * nope] std::vector<void
//   *> quant_attention_output; // [num_threads * max_qlen * nope]

// public:
//   using output_t = float;

//   LLAMA_MLA_TP(GeneralMLAConfig config, int tp_part_idx) : config(config), tp_part_idx(tp_part_idx) {
//     std::vector<std::pair<void **, uint64_t>> s_mem_requests;
//   }

//   void set_pages(std::vector<void *> cache_pages) { this->nope_pages = cache_pages; }
//   void set_pages(std::vector<void *> cache_pages, std::vector<void *> pe_pages) {
//     this->nope_pages = cache_pages;
//     this->rope_pages = pe_pages;
//   }

//   void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
//                const void *input, void *output) {}

//   void forward_prefill(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kvlens,
//                        const void *input_raw, void *output) {
//     auto pool = config.pool->get_subpool(tp_part_idx);

//     float *input = (float *)input_raw;
//     std::vector<int> qlen_split, total_len_split;
//     qlen_split.reserve(qlens.size() + 1);
//     qlen_split.push_back(0);
//     total_len_split.reserve(qlens.size() + 1);
//     int qlen_sum = 0;
//     int total_len_sum = 0;
//     for (size_t i = 0; i < qlens.size(); i++) {
//       qlen_sum += qlens[i];
//       qlen_split.push_back(qlen_sum);

//       total_len_sum += qlens[i] + kvlens[i];
//       total_len_split.push_back(total_len_sum);
//     }

//     auto which_query_by_qlen_sum = [&](int token_nth) -> std::pair<size_t, size_t> {
//       auto query_idx = std::upper_bound(qlen_split.begin(), qlen_split.end(), token_nth) - qlen_split.begin() - 1;
//       auto token_nth_from_start = token_nth - qlen_split.at(query_idx) + kvlens.at(query_idx);
//       return {query_idx, token_nth_from_start};
//     };
//     auto which_query_by_total_sum = [&](int token_nth) -> std::pair<size_t, size_t> {
//       auto query_idx =
//           std::upper_bound(total_len_split.begin(), total_len_split.end(), token_nth) - total_len_split.begin() - 1;
//       auto token_nth_from_start = token_nth - total_len_split.at(query_idx);
//       return {query_idx, token_nth_from_start};
//     };

//     auto which_page = [&](int query, int token_nth_from_start) -> std::pair<size_t, size_t> {
//       size_t page_idx = page_tables.at(query).at(div_up((size_t)token_nth_from_start, config.token_count_in_page));

//       size_t token_at_in_page = token_nth_from_start % config.token_count_in_page;
//       return {page_idx, token_at_in_page};
//     };

//     ggml_type vec_dot_type = ggml_internal_get_type_traits((ggml_type)config.q_a_proj_type).vec_dot_type;
//     size_t hidden_size_float_bytes = config.hidden_size * sizeof(float);
//     size_t hidden_size_quant_blck_count = config.hidden_size / ggml_blck_size(vec_dot_type);
//     size_t hidden_size_quant_bytes = hidden_size_quant_blck_count * ggml_type_size(vec_dot_type);
//     // quant to q8 0

//     DIRECT_OR_POOL_BY(qlen_sum, 10, qlen_sum, [&](int token_at_i) {
//       size_t token_at = token_at_i;
//       quantize_q8_0(offset_pointer(input, token_at * config.hidden_size * sizeof(float)),
//                     offset_pointer(quant_input,
//                                    token_at * QUANT_BLCK_SIZE(config.hidden_size,
//                                    VEC_DOT_TYPE(config.q_a_proj_type))),
//                     1, config.hidden_size, nullptr);
//     });

//     {
//       // q lora rank
//       // maybe this should be up to non tp
//       auto proj_lora_a = [&](int task_id) {
//         size_t token_at = task_id % qlen_sum;
//         bool do_q_or_kv = (task_id / qlen_sum) == 0;
//         if (do_q_or_kv) {
//           auto this_q_a_proj_output =
//               (float *)offset_pointer(q_a_proj_output, token_at * config.hidden_size * sizeof(float));

//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(config.q_lora_rank, 1, config.hidden_size, local_q_a_proj,
//                                             config.q_a_proj_type, quant_input, token_at, this_q_a_proj_output, 0);

//           T_RMSNorm::rms_norm_single(config.q_lora_rank, (float *)local_q_a_norm, this_q_a_proj_output);

//           quantize_q8_0(
//               this_q_a_proj_output,
//               offset_pointer(quant_q_a_proj_output,
//                              token_at * QUANT_BLCK_SIZE(config.q_lora_rank, VEC_DOT_TYPE(config.q_b_proj_type))),
//               1, config.q_lora_rank, nullptr);

//         } else {
//           auto [query, token_from_start] = which_query_by_qlen_sum(token_at);
//           auto [page_idx, token_at_in_page] = which_page(query, token_from_start);
//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(config.kv_lora_rank, 1, config.hidden_size, local_kv_a_proj_with_mqa,
//                                             config.kv_a_proj_with_mqa_type, quant_input, token_at,
//                                             rope_pages.at(page_idx), token_at_in_page);
//           T_RMSNorm::rms_norm_single(
//               config.kv_lora_rank, (float *)local_kv_a_norm_with_mqa,
//               (float *)offset_pointer(rope_pages.at(page_idx), token_at_in_page * config.kv_lora_rank *
//               sizeof(float)));
//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(config.rope_size, 1, config.hidden_size,
//                                             QUANT_OFFSET(local_kv_a_proj_with_mqa, config.kv_a_proj_with_mqa_type,
//                                                          config.kv_lora_rank, config.hidden_size),
//                                             config.kv_a_proj_with_mqa_type, quant_input, token_at,
//                                             nope_pages.at(page_idx), token_at_in_page);
//         }
//       };
//       DIRECT_OR_POOL_BY(qlen_sum, 10, qlen_sum * 2, proj_lora_a);
//     }

//     {
//       int task_count = config.num_heads * 2 * qlen_sum; // head, rope/nope, qlen
//       auto q_proj_lora_b = [&](int task_id) {
//         size_t head_idx = task_id / (2 * qlen_sum);
//         task_id %= (2 * qlen_sum);
//         bool nope_or_rope = (task_id / qlen_sum) == 0;
//         task_id %= qlen_sum;
//         size_t token_at = task_id;
//         auto [query, token_from_start] = which_query_by_qlen_sum(token_at);
//         auto [page_idx, token_at_in_page] = which_page(query, token_from_start);

//         if (nope_or_rope) {
//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(
//               config.nope_size, 1, config.q_lora_rank,
//               QUANT_OFFSET(local_q_b_proj.at(head_idx), config.q_b_proj_type,
//                            head_idx * (config.nope_size + config.rope_size), config.q_lora_rank),
//               config.q_b_proj_type, quant_q_a_proj_output, token_at, q_nope.at(head_idx), token_at);
//         } else {
//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(
//               config.rope_size, 1, config.q_lora_rank,
//               QUANT_OFFSET(local_q_b_proj.at(head_idx), config.q_b_proj_type,
//                            head_idx * (config.nope_size + config.rope_size) + config.nope_size, config.q_lora_rank),
//               config.q_b_proj_type, quant_q_a_proj_output, token_at, q_pe.at(head_idx), token_at);
//           T_RopeApplier::apply_single(config.rope_size,
//                                       offset_pointer(q_pe.at(head_idx), token_at * config.rope_size * sizeof(float)),
//                                       rope_angle.cos(token_at), rope_angle.sin(token_at));
//         }
//       };
//       pool->do_work_stealing_job(task_count, nullptr, q_proj_lora_b, nullptr);
//     }

//     for (int query = 0; query < qlens.size(); query++) {
//       {
//         // pe attention
//         // apply k pe online
//         int task_count = config.num_heads * (qlens[query] + kvlens[query]); // by kvlen
//         auto pe_attn = [&](int task_id) {
//           size_t head_idx = task_id / (qlens[query] + kvlens[query]);
//           size_t token_from_start = task_id % (qlens[query] + kvlens[query]);

//           // auto q_token_at = qlen_split[query] + qlens[query];

//           auto [page_idx, token_at_in_page] = which_page(query, token_from_start);
//           memcpy(k_pe[WorkerPool::thread_local_id],
//                  offset_pointer(rope_pages.at(page_idx), token_at_in_page * config.rope_size * sizeof(float)),
//                  sizeof(float) * config.rope_size);
//           T_RopeApplier::apply_single(config.rope_size, k_pe[WorkerPool::thread_local_id],
//                                       rope_angle.cos(token_from_start), rope_angle.sin(token_from_start));

//           LLAMAFILE_SGEMM_MATMUL_F32(1, qlens[query], config.rope_size, k_pe[WorkerPool::thread_local_id],
//                                      config.rope_size, q_pe.at(head_idx), config.rope_size,
//                                      attention_weights[head_idx], config.max_kvlen);
//         };
//         pool->do_work_stealing_job(task_count, pe_attn);
//       }
//       {
//         // clear q absorb
//         pool->do_work_stealing_job(config.num_heads, [&](int task_id) {
//           memset(q_absorb[task_id], 0, config.kv_lora_rank * config.max_qlen * sizeof(float));
//         });

//         // aborb W_uk
//         int task_count = config.num_heads * qlens[query];
//         auto task = [&](int task_id) {
//           size_t head_idx = task_id / qlens[query];
//           size_t token_at = task_id % qlens[query];

//           // q_absorb now [kvrank, max_qlen]
//           LLAMAFILE_SGEMM_MATMUL_F32(qlens[query], config.kv_lora_rank, config.nope_size, q_nope[head_idx],
//                                      config.nope_size, local_k_b_proj_nope[head_idx], config.nope_size,
//                                      q_absorb[head_idx], config.max_qlen);
//           transpose(q_nope[head_idx], config.kv_lora_rank, config.max_qlen, qlens[query]);
//         };
//         pool->do_work_stealing_job(task_count, task);
//       }

//       {
//         // nope attention weights
//         size_t page_count = div_up((size_t)kvlens[query], config.token_count_in_page);
//         int task_count = config.num_heads * page_count;
//         auto task = [&](int task_id) {
//           size_t head_idx = task_id / page_count;
//           size_t page_idx = task_id % page_count;
//           void *page_ptr = nope_pages[page_tables[query][page_idx]]; // mla no head

//           size_t kvlen =
//               page_idx == (page_count - 1) ? (kvlens[query] % config.token_count_in_page) :
//               config.token_count_in_page;

//           LLAMAFILE_SGEMM_MATMUL_F32(
//               kvlen, qlens[query], config.kv_lora_rank, page_ptr, config.kv_lora_rank, q_absorb[head_idx],
//               config.max_qlen,
//               offset_pointer(attention_weights[head_idx], page_idx * config.token_count_in_page * sizeof(float)),
//               config.max_kvlen);
//           // static_assert(false, "soft max todo");
//         };
//         pool->do_work_stealing_job(task_count, task);
//       }

//       {
//         // clear o absorb
//         pool->do_work_stealing_job(config.num_heads, [&](int task_id) {
//           memset(o_absorb[task_id], 0, config.kv_lora_rank * config.max_qlen * sizeof(float));
//         });

//         // o absorb
//         size_t page_count = div_up((size_t)kvlens[query], config.token_count_in_page);
//         int task_count = config.num_heads * page_count;
//         auto task = [&](int task_id) {
//           size_t head_idx = task_id / page_count;
//           size_t page_idx = task_id % page_count;
//           void *page_ptr = nope_pages[page_tables[query][page_idx]]; // mla no head
//           size_t kvlen =
//               page_idx == (page_count - 1) ? (kvlens[query] % config.token_count_in_page) :
//               config.token_count_in_page;

//           memcpy(compressed_kv_tmp[WorkerPool::thread_local_id], page_ptr,
//                  config.token_count_in_page * config.kv_lora_rank * sizeof(float));
//           transpose(compressed_kv_tmp[WorkerPool::thread_local_id], config.token_count_in_page, config.kv_lora_rank,
//                     kvlen);

//           LLAMAFILE_SGEMM_MATMUL_F32(
//               config.kv_lora_rank, qlens[query], kvlen, compressed_kv_tmp[WorkerPool::thread_local_id],
//               config.token_count_in_page,
//               offset_pointer(attention_weights[head_idx], page_idx * config.token_count_in_page * sizeof(float)),
//               config.max_kvlen, o_absorb[head_idx], config.kv_lora_rank);
//         };
//         pool->do_work_stealing_job(task_count, task);
//       }

//       {

//         // clear
//         pool->do_work_stealing_job(config.num_heads, [&](int task_id) {
//           memset(attention_output[task_id], 0, config.nope_size * config.max_qlen * sizeof(float));
//         });

//         // attention output
//         int task_count = config.num_heads * qlens[query];
//         auto task = [&](int task_id) {
//           size_t head_idx = task_id / qlens[query];
//           size_t token_at = task_id % qlens[query];

//           quantize_q8_0((float *)offset_pointer(o_absorb[head_idx], config.kv_lora_rank * token_at * sizeof(float)),
//                         offset_pointer(quant_o_absorb[head_idx],
//                                        QUANT_BLCK_SIZE(config.kv_lora_rank, VEC_DOT_TYPE(config.kv_b_proj_type))),
//                         1, config.kv_lora_rank, nullptr);

//           auto kv_b_proj_ptr =
//               offset_pointer(local_kv_b_proj, ((head_idx * 2 + 1) * config.nope_size) *
//                                                   QUANT_BLCK_SIZE(config.kv_lora_rank, config.kv_b_proj_type));

//           LLAMAFILE_SGEMM_QUANT_FULL_MATMUL(config.nope_size, 1, config.kv_lora_rank, kv_b_proj_ptr,
//                                             config.kv_b_proj_type, quant_o_absorb[head_idx], token_at,
//                                             attention_output[head_idx], token_at);
//         };
//         pool->do_work_stealing_job(task_count, task);
//       }

//       {
//         // quant attention output
//         // static_assert(false,"TODO" );
//       }

//       {
//         // get final output
//         // static_assert(false,"TODO" );
//       }
//     }
//   }

//   void load_weights(int complete_num_heads, int offset) {}
// };
// template <typename Norm, typename Rope, typename RopeAngle>
// class TP_MLA<LLAMA_MLA_TP<Norm, Rope, RopeAngle>> : public TP_MLA_Common<LLAMA_MLA_TP<Norm, Rope, RopeAngle>> {
// public:
//   using TP_MLA_Common<LLAMA_MLA_TP<Norm, Rope, RopeAngle>>::TP_MLA_Common;

//   void load_weights() {
//     auto pool = this->config.pool;
//     auto tp_num_heads = this->config.num_heads / this->tp_count;
//     pool->dispense_backend()->do_numa_job([this, pool, tp_num_heads](int tp_id) {
//       this->tps[tp_id]->load_weights(this->config.num_heads, tp_id * tp_num_heads);
//     });
//     this->weights_loaded = true;
//   }

//   void merge_results(int qlen, void *output) {}
// };

// #endif
