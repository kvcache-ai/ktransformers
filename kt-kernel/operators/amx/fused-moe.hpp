/**
 * @file fused-moe.hpp
 * @brief AMXINT4_SMART fused two-stage MoE host.
 *
 * Hosts the per-attribute buffers of the stage pair (upstream node KA =
 * gate/up, downstream node KB = down) and runs the task-specific fused
 * decode (amx::FusedTwoStage) in one pass per activated expert at qlen=1.
 * The layer's RAM keeps gate/up at the upstream precision and the down at
 * the downstream precision — no whole-layer re-quantization. Prefill
 * (qlen>1) falls back to the per-stage path so the batch path is never
 * wrong, only slower.
 *
 * Instantiations (stage pairs):
 *   F4x8   = <GemmKernel224Int4_1, GemmKernel224Int8>  (0 -> 1)
 *   F8x16  = <GemmKernel224Int8,   GemmKernel224BF16>  (1 -> 2)
 *   F4x16  = <GemmKernel224Int4_1, GemmKernel224BF16>  (0 -> 2)
 */
#ifndef CPUINFER_OPERATOR_AMX_FUSED_MOE_H
#define CPUINFER_OPERATOR_AMX_FUSED_MOE_H

#include <cstdlib>
#include <memory>
#include <vector>

#include "moe_base.hpp"
#include "la/amx_fused.hpp"
#include "../gguf/dequant.hpp"

namespace {

inline float fused_bf16_to_f32(ggml_bf16_t v) {
  uint32_t u = (((uint32_t)v.bits) & 0xFFFFu) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

}  // namespace

template <class KA, class KB>
class AMX_FUSED_MOE_TP : public AMX_MOE_BASE<KA, AMX_FUSED_MOE_TP<KA, KB>> {
 public:
  using Base = AMX_MOE_BASE<KA, AMX_FUSED_MOE_TP<KA, KB>>;
  using typename Base::input_t;
  using typename Base::output_t;
  using Base::config_;
  using Base::tp_part_idx;
  using Base::gate_bb_;
  using Base::up_bb_;
  using Base::down_bb_;
  using Base::gate_up_ba_;
  using Base::gate_bc_;
  using Base::up_bc_;
  using Base::down_ba_;
  using Base::down_bc_;
  using Base::m_local_num_;

  using UA = typename KA::BufferA;
  using UB = typename KA::BufferB;
  using UC = typename KA::BufferC;
  using DA = typename KB::BufferA;
  using DB = typename KB::BufferB;
  using DC = typename KB::BufferC;
  using Fused = amx::FusedTwoStage<KA, KB>;

  AMX_FUSED_MOE_TP() = default;
  AMX_FUSED_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    // NOTE: runs from the base ctor — the derived members do NOT exist yet,
    // so nothing here may touch them (the buffers are allocated in
    // load_weights, after construction).
    printf("Creating AMX_FUSED_MOE_TP %d at numa %d (%s->%s)\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()),
           KA::name().c_str(), KB::name().c_str());
  }

  void alloc_buffers() {
    const int e = (int)config_.expert_num;
    const int n = config_.intermediate_size;
    const int k = config_.hidden_size;
    auto alloc_block = [this](size_t sz) -> void* {
      void* p = nullptr;
      if (posix_memalign(&p, 64, std::max<size_t>(sz, 1)) != 0) throw std::bad_alloc();
      alt_mem_blocks_.push_back(p);
      return p;
    };
    const int cm = (config_.max_len + KA::M_STEP - 1) / KA::M_STEP * KA::M_STEP;
    for (int i = 0; i < e; i++) {
      gate_a_.push_back(std::make_shared<UA>(config_.max_len, k, alloc_block(UA::required_size(config_.max_len, k))));
      up_a_.push_back(std::make_shared<UA>(config_.max_len, k, alloc_block(UA::required_size(config_.max_len, k))));
      gate_b_.push_back(std::make_shared<UB>(n, k, alloc_block(UB::required_size(n, k))));
      up_b_.push_back(std::make_shared<UB>(n, k, alloc_block(UB::required_size(n, k))));
      gate_c_.push_back(std::make_shared<UC>(cm, n, alloc_block(UC::required_size(cm, n))));
      up_c_.push_back(std::make_shared<UC>(cm, n, alloc_block(UC::required_size(cm, n))));
      down_a_.push_back(std::make_shared<DA>(config_.max_len, n, alloc_block(DA::required_size(config_.max_len, n))));
      down_b_.push_back(std::make_shared<DB>(k, n, alloc_block(DB::required_size(k, n))));
      down_c_.push_back(std::make_shared<DC>(cm, k, alloc_block(DC::required_size(cm, k))));
      g_.push_back(std::make_shared<std::vector<ggml_bf16_t>>(config_.max_len * n));
      u_.push_back(std::make_shared<std::vector<ggml_bf16_t>>(config_.max_len * n));
      h_.push_back(std::make_shared<std::vector<ggml_bf16_t>>(config_.max_len * n));
      out_.push_back(std::make_shared<std::vector<ggml_bf16_t>>(config_.max_len * k));
    }
  }

  virtual ~AMX_FUSED_MOE_TP() {
    for (void* p : alt_mem_blocks_) std::free(p);
  }

  size_t buffer_a_required_size_impl(size_t m, size_t kk) const { return UA::required_size(m, kk); }
  size_t buffer_b_required_size_impl(size_t nn, size_t kk) const { return UB::required_size(nn, kk); }
  size_t buffer_c_required_size_impl(size_t m, size_t nn) const { return UC::required_size(m, nn); }
  std::shared_ptr<UA> make_buffer_a_impl(size_t m, size_t kk, void* data) const {
    return std::make_shared<UA>(m, kk, data);
  }
  std::shared_ptr<UB> make_buffer_b_impl(size_t nn, size_t kk, void* data) const {
    return std::make_shared<UB>(nn, kk, data);
  }
  std::shared_ptr<UC> make_buffer_c_impl(size_t m, size_t nn, void* data) const {
    return std::make_shared<UC>(m, nn, data);
  }

  // ---- load: per-attribute strips into the per-node buffers ----
  void load_weights() {
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const int tp_count = (int)config_.pool->config.subpool_count;
    const int64_t full_I = config_.gguf_full_intermediate_size > 0
                               ? (int64_t)config_.gguf_full_intermediate_size
                               : (int64_t)config_.intermediate_size * tp_count;
    const int64_t row_off = (int64_t)config_.intermediate_size * tp_part_idx;
    if (config_.gate_gguf == nullptr) throw std::runtime_error("fused MOE requires GGUF source");
    if (gate_b_.empty()) alloc_buffers();  // post-construction member allocation

    // gate/up on KA (per-row INT4: per-row d; INT8: per-row d)
    {
      int nth = KA::recommended_nth(config_.intermediate_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map, row_off](int task_id) {
            int64_t expert_idx = task_id / nth;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;
            auto [n_start, n_end] = KA::split_range_n(config_.intermediate_size, ith, nth);
            if (n_start >= n_end) return;
            thread_local std::vector<ggml_bf16_t> strip;
            strip.resize((size_t)(n_end - n_start) * config_.hidden_size);
            const char* gate_base = (const char*)config_.gate_gguf + logical_expert_id * config_.gate_gguf_stride;
            kt::gguf::dequant_rows_bf16(gate_base, (ggml_type)config_.gate_gguf_type, config_.hidden_size,
                                        row_off + n_start, row_off + n_end, strip.data());
            gate_b_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            if (this->gate_bb_.size() > logical_expert_id) this->gate_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            const char* up_base = (const char*)config_.up_gguf + logical_expert_id * config_.up_gguf_stride;
            kt::gguf::dequant_rows_bf16(up_base, (ggml_type)config_.up_gguf_type, config_.hidden_size,
                                        row_off + n_start, row_off + n_end, strip.data());
            up_b_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            if (this->up_bb_.size() > logical_expert_id) this->up_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
          },
          nullptr);
    }
    // down on KB
    {
      int nth = KB::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map, row_off, full_I](int task_id) {
            int64_t expert_idx = task_id / nth;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;
            auto [n_start, n_end] = KB::split_range_n(config_.hidden_size, ith, nth);
            if (n_start >= n_end) return;
            const int64_t dcol = config_.intermediate_size;
            thread_local std::vector<ggml_bf16_t> strip;
            strip.resize((size_t)(n_end - n_start) * dcol);
            const char* down_base = (const char*)config_.down_gguf + logical_expert_id * config_.down_gguf_stride;
            kt::gguf::dequant_rows_bf16(down_base, (ggml_type)config_.down_gguf_type, full_I, n_start, n_end,
                                        row_off, row_off + dcol, strip.data());
            down_b_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
          },
          nullptr);
    }
    // the base's own down buffer (the prefill fallback runs the upstream
    // kernel on it) needs the KA split — a separate pass with KA strips.
    {
      int nth = KA::recommended_nth(config_.hidden_size);
      pool->do_work_stealing_job(
          nth * config_.expert_num, nullptr,
          [this, nth, physical_to_logical_map, row_off, full_I](int task_id) {
            int64_t expert_idx = task_id / nth;
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
            int ith = task_id % nth;
            auto [n_start, n_end] = KA::split_range_n(config_.hidden_size, ith, nth);
            if (n_start >= n_end) return;
            const int64_t dcol = config_.intermediate_size;
            thread_local std::vector<ggml_bf16_t> strip;
            strip.resize((size_t)(n_end - n_start) * dcol);
            const char* down_base = (const char*)config_.down_gguf + logical_expert_id * config_.down_gguf_stride;
            kt::gguf::dequant_rows_bf16(down_base, (ggml_type)config_.down_gguf_type, full_I, n_start, n_end,
                                        row_off, row_off + dcol, strip.data());
            this->down_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
          },
          nullptr);
    }
  }

  // ---- qlen=1 decode: fused two-stage per activated expert ----
  void forward(int qlen, int kk, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      Base::forward(qlen, kk, expert_ids, weights, input, output);  // prefill: plain path
      return;
    }
    const int H = config_.hidden_size;
    const int I = config_.intermediate_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const ggml_bf16_t* x = (const ggml_bf16_t*)input;
    float* f32out = (float*)output;

    // activated experts
    int act = 0;
    for (int j = 0; j < kk; j++) {
      if (config_.should_skip_expert(expert_ids[j])) continue;
      const int64_t e = expert_ids[j];
      const int m = 1;
      auto& ga = gate_a_[e];
      auto& gb = gate_b_[e];
      auto& gc = gate_c_[e];
      auto& ua = up_a_[e];
      auto& ub = up_b_[e];
      auto& uc = up_c_[e];
      auto& da = down_a_[e];
      auto& db = down_b_[e];
      auto& dc = down_c_[e];
      Fused::run(m, I, H, const_cast<ggml_bf16_t*>(x), ga.get(), gb.get(), gc.get(), ua.get(), ub.get(), uc.get(),
                 da.get(), db.get(), dc.get(), g_[e]->data(), u_[e]->data(), h_[e]->data(), out_[e]->data());
      // weighted blend into the fp32 output
      const float w = weights[j];
      const ggml_bf16_t* src = out_[e]->data();
      float* dst = f32out;
      for (int c = 0; c < H; c++) dst[c] += w * fused_bf16_to_f32(src[c]);
    }
  }

  // ---- prefill fallback: the plain per-node path on the base's own
  // buffers (the base's forward_prefill/decode call these; the qlen=1 fused
  // path overrides forward before they are reached). The down runs on the
  // upstream kernel — the prefill is correctness-first, not fused.
  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    const int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.intermediate_size, config_.hidden_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    const int m = m_local_num_[expert_idx];
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                   down_bc_[expert_idx], ith, nth);
    }
  }

 protected:
  std::vector<std::shared_ptr<UA>> gate_a_, up_a_;
  std::vector<std::shared_ptr<UB>> gate_b_, up_b_;
  std::vector<std::shared_ptr<UC>> gate_c_, up_c_;
  std::vector<std::shared_ptr<DA>> down_a_;
  std::vector<std::shared_ptr<DB>> down_b_;
  std::vector<std::shared_ptr<DC>> down_c_;
  std::vector<std::shared_ptr<std::vector<ggml_bf16_t>>> g_, u_, h_, out_;
  std::vector<void*> alt_mem_blocks_;
};

// ============================================================================
// TP_MOE specialization: the outer dispatcher for the fused hosts (mirrors
// the K2-outer). The GGUF source config is shared to every single-TP part.
// ============================================================================

template <class KA, class KB>
class TP_MOE<AMX_FUSED_MOE_TP<KA, KB>> : public TP_MOE<AMX_MOE_BASE<KA, AMX_FUSED_MOE_TP<KA, KB>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<KA, AMX_FUSED_MOE_TP<KA, KB>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    (void)physical_to_logical_map;
    if (config.gate_gguf == nullptr) {
      throw std::runtime_error("fused MOE requires GGUF source");
    }
    for (auto i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      tpc.gate_gguf = config.gate_gguf;
      tpc.gate_gguf_stride = config.gate_gguf_stride;
      tpc.gate_gguf_type = config.gate_gguf_type;
      tpc.up_gguf = config.up_gguf;
      tpc.up_gguf_stride = config.up_gguf_stride;
      tpc.up_gguf_type = config.up_gguf_type;
      tpc.down_gguf = config.down_gguf;
      tpc.down_gguf_stride = config.down_gguf_stride;
      tpc.down_gguf_type = config.down_gguf_type;
      tpc.gguf_full_intermediate_size = config.gguf_full_intermediate_size;
    }
    DO_TPS_LOAD_WEIGHTS(pool);
    this->weights_loaded = true;
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FUSED_MOE_H