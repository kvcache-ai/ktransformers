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

template <typename A, class KERNEL>
class KML_MLA_TP_QUAN_TEST
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 public:
  using input_t = A;
  using output_t = A;
  using quant_t = int8_t;
  KML_MLA_TP_QUAN_TEST(GeneralMLAConfig config, int tp_part_idx);
  void load_weights(int complete_num_heads, int offset);
  void set_pages(std::vector<void*> kv_lora_pages, std::vector<void*> pe_pages);
  void set_local_pages(int page_count);
  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               std::vector<void*> attention_masks, const void* input, void* output);
  void forward(std::vector<int> qlens, std::vector<std::vector<int>> page_tables, std::vector<int> kv_lens,
               const void* input, void* output);

 private:
  using T_RMSNorm = RMSNorm<A>;
  using T_RopeAngle = DeepseekV3YarnRotaryEmbedding;
  using T_RopeApplier = Rope<T_RopeAngle, A>;
  using T_SoftmaxApplier = Softmax<A>;
  using KMatRefA = typename arm_kml::MatRef<int8_t>;
  using KMatRefB = typename arm_kml::MatRef<typename KERNEL::dt>;
  using KMatRef = typename arm_kml::MatRef<A>;
  using KMatRefC = typename arm_kml::MatRef<int32_t>;
  using GemmKernel = KERNEL;

  GeneralMLAConfig config;
  const size_t col_block = 256;
  const size_t row_block = 256;

  // for quant
  const size_t col_block_q_absorb = 512;  // 上限kv_lora_rank:512
  const size_t row_block_q_absorb = 512;  // 上限qlen

  const size_t col_block_q_lora_kv_lora_rope = 256;
  const size_t row_block_q_lora_kv_lora_rope = 256;

  const size_t col_block_q_nope_rope = 512;
  const size_t row_block_q_nope_rope = 512;

  const size_t col_block_attention_output = 1024;  // 上限qlen的大小
  const size_t row_block_attention_output = 256;   // 上限128

  const size_t col_block_out_by_head = 256;  // 上限 qlen
  const size_t row_block_out_by_head = 256;  // 上限 hidden_size:7168

  // ==========================================
  // decode

  const size_t decode_col_block_q_absorb = 256;  // 上限kv_lora_rank:512
  const size_t decode_row_block_q_absorb = 64;   // 上限qlen

  const size_t decode_col_block_q_lora_kv_lora_rope = 64;
  const size_t decode_row_block_q_lora_kv_lora_rope = 64;

  const size_t decode_col_block_q_nope_rope = 512;
  const size_t decode_row_block_q_nope_rope = 512;

  const size_t decode_col_block_attention_output = 1024;  // 上限qlen的大小
  const size_t decode_row_block_attention_output = 256;   // 上限128

  const size_t decode_col_block_out_by_head = 1;    // 上限 qlen
  const size_t decode_row_block_out_by_head = 512;  // 上限 hidden_size:7168

  // ==========================================

  const size_t col_block_o_absorb = 256;
  const size_t row_block_o_absorb = 256;

  const size_t col_block_nope_attention = 256;
  const size_t row_block_nope_attention = 256;

  const size_t col_block_pe_attention = 256;
  const size_t row_block_pe_attention = 256;

  int tp_part_idx;
  std::vector<void*> default_attention_masks;

  // std::vector<void *> kv_lora_pages; // [page_count * page_token_count * nope]
  // std::vector<void *> rope_pages;    // [page_count * page_token_count * nope]
  std::vector<A*> cc_pages;  // [page_count * page_token_count * (kv rank + rope size)]
  size_t cc_size;
  // col major:[kv_lora_rank, qlen] or row major:[qlen, kv_lora_rank]
  std::vector<KMatRef> cc_page_refs, kv_lora_page_refs, rope_page_refs;
  // col major:[kv_lora_rank, qlen] or [rope_size, qlen]
  std::vector<KMatRefC> cc_page_refs_buffer, kv_lora_page_refs_buffer, rope_page_refs_buffer;

  // row major:[qlen, kv_lora_rank] or [qlen, rope_size]
  std::vector<KMatRefC> cc_page_refs_decode_buffer, kv_lora_page_refs_decode_buffer, rope_page_refs_decode_buffer;
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
  std::vector<A*>
      local_w_decode_o;  // [num_heads/sub_num_heads_decode]*[(sub_num_heads_decode * hidden_size * nope_size)]

  std::unique_ptr<T_RopeAngle> rope_angle;

  // KMatRefAB local_q_a_proj_ref;
  KMatRefB local_q_a_proj_quant_ref;
  KMatRef local_q_b_proj_ref;
  // KMatRefAB local_kv_a_proj_with_mqa_ref;
  KMatRefB local_kv_a_proj_with_mqa_decode_ref;
  KMatRef local_k_b_proj_ref;
  KMatRef local_v_b_proj_ref;
  // KMatRefAB local_w_o_decode_ref;
  KMatRefB local_w_o_ref;

  typename GemmKernel::BufferA* local_q_a_proj_quant_ba;  // [max_qlen,hidden_size]
  typename GemmKernel::BufferB* local_q_a_proj_quant_bb;  // [hidden_size, q_lora_rank]
  typename GemmKernel::BufferC* local_q_a_proj_quant_bc;  // [max_qlen,q_lora_rank] (row major)

  typename GemmKernel::BufferA* local_kv_a_proj_with_mqa_quant_ba;  // [max_qlen, hidden_size]
  typename GemmKernel::BufferB* local_kv_a_proj_with_mqa_quant_bb;  // [hidden_size, kv_lora_rank + rope_size]
  std::vector<typename GemmKernel::BufferC*>
      local_kv_a_proj_with_mqa_quant_bc;  // page_count * [page_token_count, rope_size + kv_lora_rank] (row major)

  // 对应local_w_o

  // 对应local_w_o
  typename GemmKernel::BufferA* local_w_o_quant_ba;  // [max_qlen, num_heads * nope_size]
  typename GemmKernel::BufferB* local_w_o_quant_bb;  // [num_heads * nope_size, hidden_size]
  // qlen_output
  typename GemmKernel::BufferC* local_w_o_prefill_bc;  // [max_qlen, hidden_size]
  std::vector<typename GemmKernel::BufferC*>
      local_w_o_decode_bc;  // [num_heads/sub_num_heads_decode] *[max_qlen, hidden_size]

  // std::vector<typename GemmKernel::BufferA *>
  //     local_w_o_decode_ba; // [num_heads/sub_num_heads_decode]*[hidden_size,sub_num_heads_decode * nope_size] row
  //     major
  std::vector<typename GemmKernel::BufferB*>
      local_w_o_decode_bb;  // [num_heads/sub_num_heads_decode]*[sub_num_heads_decode * nope_size,hidden_size] col major

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
  size_t sub_num_heads = 16;        // 用于并发的子头设置
  size_t sub_num_heads_decode = 8;  // 用于并发的子头设置

  // std::vector<A *> qlen_decode_output; // [max_qlen * hidden_size]
  std::vector<A*> qlen_quant_output;  // [[num_heads/sub_num_heads] * max_qlen * hidden_size] row major

  A softmax_scale;

#ifdef DEBUG_THIS_MLA
  std::string file_name;
#endif

  static bool decide_absorb(size_t qlen, size_t existing_kvlen) {
    double x = existing_kvlen;
    return qlen < (-x + sqrt(x * (x + 2048.0 / 3.0)) / 2.0);
  }
  // 只保留声明，移除实现
  void nope_attention_q_absorb(int qlen, int kvlen, const std::vector<int>& page_table, bool increamental = true,
                               bool is_decode = false);
  void nope_attention_no_absorb(int qlen, int kvlen, const std::vector<int>& page_table, bool increamental = true);
  void output_absorb(int query, const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                     const std::vector<int>& kvlens, bool is_decode = false);
  void output_no_absorb(int query, const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                        const std::vector<int>& kvlens);
  void forward_prefill(const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                       const std::vector<int>& kvlens, const std::vector<void*>& attention_masks,
                       const input_t* input_raw, output_t* output_raw);
  void forward_decode(const std::vector<int>& qlens, const std::vector<std::vector<int>>& page_tables,
                      const std::vector<int>& kvlens, const std::vector<void*>& attention_masks,
                      const input_t* input_raw, output_t* output_raw);
  void q_lora_kv_lora_rope_quant(int query, const std::vector<int>& qlens, const std::vector<int>& kvlens,
                                 const std::vector<std::vector<int>>& page_tables, std::vector<int>& qlen_split,
                                 KMatRefA& input_ref, KMatRefC& q_lora_rank_ref, KMatRef& q_lora_rank_out_ref,
                                 bool is_decode = false);
};

template <typename A, class KERNEL>
class TP_MLA<KML_MLA_TP_QUAN_TEST<A, KERNEL>> : public TP_MLA_Common<KML_MLA_TP_QUAN_TEST<A, KERNEL>> {
 public:
  using TP_MLA_Common<KML_MLA_TP_QUAN_TEST<A, KERNEL>>::TP_MLA_Common;

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
    typename KML_MLA_TP_QUAN_TEST<A, KERNEL>::output_t* output =
        (typename KML_MLA_TP_QUAN_TEST<A, KERNEL>::output_t*)output_raw;
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
    // dump_bin_int8("output.bin", output, qlen * this->config.hidden_size);

#endif
  }
};

template class KML_MLA_TP_QUAN_TEST<float, arm_kml::GemmKernelInt8>;
template class KML_MLA_TP_QUAN_TEST<float, arm_kml::GemmKernelInt4>;
// template class KML_MLA_TP_QUAN_TEST<float16_t>;