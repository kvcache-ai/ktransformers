/**
 * @Description  : Common AMX MoE base class extracted from K2 implementation.
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 0.1.0
 * @LastEditors  : oql, Codex and Claude
 * @LastEditTime : 2025-12-09
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_BASE_H
#define CPUINFER_OPERATOR_AMX_MOE_BASE_H

// #define FORWARD_TIME_PROFILE

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../common.hpp"
#include "../moe-tp.hpp"
#include "la/amx.hpp"
#include "llama.cpp/ggml.h"

template <class T, class Derived>
class AMX_MOE_BASE {
 public:
  int tp_part_idx = 0;

  // RAM-state diagnostic: shows exactly what the packed buffers hold after
  // load — per attribute: the per-row scale stats (d), the packed-data
  // zero fraction, and the footprint, as an ASCII map. The gate/up/down
  // are stored per-row quantized; a scale of 0 or an all-zero packed
  // region means the strip load left that attribute empty.
  void debug_ram_stats(const char* tag = "moe") {
    const int BAR = 20;
    struct Info {
      const char* name;
      int n, k;
      double fill;      // nonzero fraction of the packed data
      double dz;        // zero fraction of the per-row scales
      float dmin, dmax;
      double mb;        // footprint
      bool ok;
    };
    auto collect = [&](const char* name, const std::shared_ptr<typename T::BufferB>& bb) -> Info {
      if (!bb) return {name, 0, 0, 0.0, 1.0, 0.0f, 0.0f, 0.0, false};
      const int n = bb->n, k = bb->k;
      int64_t d_zeros = 0;
      float dmin = 1e30f, dmax = -1e30f;
      if constexpr (requires { bb->d[0]; }) {
        for (int i = 0; i < n; i++) {
          const float d = bb->d[i];
          if (d == 0.0f) d_zeros++;
          dmin = std::min(dmin, d);
          dmax = std::max(dmax, d);
        }
      }
      const auto* b = (const uint8_t*)bb->b;
      const int64_t total = (int64_t)n * k;
      const int64_t sample = std::min<int64_t>(total, (int64_t)n * 64);
      int64_t b_nonzero = 0;
      for (int64_t i = 0; i < sample; i++) {
        if (b[i] != 0) b_nonzero++;
      }
      if constexpr (requires { bb->d[0]; }) {
        return {name, n, k, (double)b_nonzero / (double)sample, (double)d_zeros / (double)n, dmin, dmax,
                (double)total / (1024.0 * 1024.0), true};
      } else {
        return {name, n, k, (double)b_nonzero / (double)sample, -1.0, 0.0f, 0.0f,
                (double)total / (1024.0 * 1024.0), true};
      }
    };
    Info infos[3] = {collect("gate", gate_bb_.empty() ? nullptr : gate_bb_[0]),
                     collect("up", up_bb_.empty() ? nullptr : up_bb_[0]),
                     collect("down", down_bb_.empty() ? nullptr : down_bb_[0])};

    printf("[ram] %s (tp=%d):\n", tag, tp_part_idx);
    for (auto& inf : infos) {
      if (!inf.ok) {
        printf("[ram]   %-5s : <empty>\n", inf.name);
        continue;
      }
      int filled = (int)(inf.fill * BAR + 0.5);
      if (filled > BAR) filled = BAR;
      printf("[ram]   %-5s : [", inf.name);
      for (int i = 0; i < BAR; i++) printf("%c", i < filled ? '#' : '.');
      if (inf.dz >= 0.0) {
        printf("] %4.0f%% fill  %7.2f MB  (n=%d k=%d)  scales zero-frac %.3f  dmin %.4g  dmax %.4g\n",
               inf.fill * 100.0, inf.mb, inf.n, inf.k, inf.dz, inf.dmin, inf.dmax);
      } else {
        printf("] %4.0f%% fill  %7.2f MB  (n=%d k=%d)  (unscaled format)\n", inf.fill * 100.0, inf.mb, inf.n,
               inf.k);
      }
    }
    // per-row scale samples (first 5 rows of each attribute)
    auto samples = [&](const char* name, const std::shared_ptr<typename T::BufferB>& bb) {
      if (!bb) return;
      if constexpr (requires { bb->d[0]; }) {
        printf("[ram]        %s d-row[0..4]=%.4g %.4g %.4g %.4g %.4g\n", name, bb->d[0], bb->d[1], bb->d[2],
               bb->d[3], bb->d[4]);
      } else {
        printf("[ram]        %s bf16 b-row[0..4]=0x%04x 0x%04x 0x%04x 0x%04x 0x%04x\n", name,
               ((const uint16_t*)bb->b)[0], ((const uint16_t*)bb->b)[1], ((const uint16_t*)bb->b)[2],
               ((const uint16_t*)bb->b)[3], ((const uint16_t*)bb->b)[4]);
      }
    };
    auto stats_expert = [&](int ex) {
      if (gate_bb_.size() <= (size_t)ex) return;
      auto& gb = gate_bb_[ex];
      auto& ub = up_bb_[ex];
      auto& db = down_bb_[ex];
      const int n = gb->n;
      int64_t gz = 0, uz = 0, dz = 0;
      for (int i = 0; i < n; i++) {
        if constexpr (requires { gb->d[0]; }) {
          if (gb->d[i] == 0.0f) gz++;
          if (ub->d[i] == 0.0f) uz++;
        }
        if constexpr (requires { db->d[0]; }) {
          if (db->d[i] == 0.0f) dz++;
        }
      }
      printf("[ram]   expert %d: gate d-zeros=%lld up d-zeros=%lld down d-zeros=%lld\n", ex, (long long)gz,
             (long long)uz, (long long)dz);
      if constexpr (requires { gb->d[0]; }) {
        printf("[ram]        gate d[0]=%.4g up d[0]=%.4g down d[0]=%.4g\n", (float)gb->d[0], (float)ub->d[0],
               (float)db->d[0]);
      }
    };
    printf("[ram]   experts 0..7:\n");
    for (int ex = 0; ex < 8; ex++) stats_expert(ex);
    // forward-side state: the activated-expert counts and the intermediates
    printf("[ram]   forward state: m_local_num_[1..4]=%d %d %d %d activated=%d\n",
           m_local_num_.size() > 1 ? m_local_num_[1] : -1, m_local_num_.size() > 2 ? m_local_num_[2] : -1,
           m_local_num_.size() > 3 ? m_local_num_[3] : -1, m_local_num_.size() > 4 ? m_local_num_[4] : -1,
           (int)m_expert_id_map_.size() ? (int)m_expert_id_map_.size() : -1);
    if (m_local_gate_output_ && m_local_up_output_) {
      printf("[ram]   g[0..4]=%.4g %.4g %.4g %.4g %.4g  u[0..4]=%.4g %.4g %.4g %.4g %.4g\n",
             (double)ggml_bf16_to_fp32(m_local_gate_output_[0]), (double)ggml_bf16_to_fp32(m_local_gate_output_[1]),
             (double)ggml_bf16_to_fp32(m_local_gate_output_[2]), (double)ggml_bf16_to_fp32(m_local_gate_output_[3]),
             (double)ggml_bf16_to_fp32(m_local_gate_output_[4]), (double)ggml_bf16_to_fp32(m_local_up_output_[0]),
             (double)ggml_bf16_to_fp32(m_local_up_output_[1]), (double)ggml_bf16_to_fp32(m_local_up_output_[2]),
             (double)ggml_bf16_to_fp32(m_local_up_output_[3]), (double)ggml_bf16_to_fp32(m_local_up_output_[4]));
    }
    if (!gate_bb_.empty()) samples("gate", gate_bb_[0]);
    if (!up_bb_.empty()) samples("up", up_bb_[0]);
    if (!down_bb_.empty()) samples("down", down_bb_[0]);
    // down-stage internals for expert 1 (the forward uses experts 1,2)
    auto dump_down_internals = [&](int ex) {
      if (down_ba_.size() <= (size_t)ex || down_bc_.size() <= (size_t)ex) return;
      auto& dba = down_ba_[ex];
      auto& dbc = down_bc_[ex];
      printf("[ram]   down internals expert %d:\n", ex);
      if constexpr (requires { dba->d[0]; }) {
        printf("[ram]     down-A d[0..4]=%.4g %.4g %.4g %.4g %.4g (h-quant scales)\n", (float)dba->d[0],
               (float)dba->d[1], (float)dba->d[2], (float)dba->d[3], (float)dba->d[4]);
      }
      float* c = (float*)dbc->c;
      // the full-window stats of the down-C (the whole [max_m][n] pool slice)
      double csum = 0.0;
      int c_nz = 0;
      const int cn = dbc->n;
      const int c_max_m = dbc->max_m;
      const size_t ctotal = (size_t)c_max_m * cn;
      for (size_t i = 0; i < ctotal; i++) {
        if (c[i] != 0.0f) c_nz++;
        csum += c[i];
      }
      printf("[ram]     down-C c[0..4]=%.4g %.4g %.4g %.4g %.4g  c[512..516]=%.4g %.4g %.4g %.4g %.4g\n", c[0], c[1],
             c[2], c[3], c[4], c[512], c[513], c[514], c[515], c[516]);
      printf("[ram]     down-C full: max_m=%d n=%d lanes=%zu nonzero=%d/%zu sum=%.4g\n", c_max_m, cn, ctotal,
             c_nz, c_nz, ctotal, csum);
      if (m_local_down_output_) {
        printf("[ram]     down-scratch out[0..4]=%.4g %.4g %.4g %.4g %.4g\n",
               (double)ggml_bf16_to_fp32(m_local_down_output_[0]),
               (double)ggml_bf16_to_fp32(m_local_down_output_[1]),
               (double)ggml_bf16_to_fp32(m_local_down_output_[2]),
               (double)ggml_bf16_to_fp32(m_local_down_output_[3]),
               (double)ggml_bf16_to_fp32(m_local_down_output_[4]));
      }
    };
    dump_down_internals(1);
    dump_down_internals(2);
    fflush(stdout);
  }

  ggml_bf16_t* m_local_input_ = nullptr;
  ggml_bf16_t* m_local_gate_output_ = nullptr;
  ggml_bf16_t* m_local_up_output_ = nullptr;
  ggml_bf16_t* m_local_down_output_ = nullptr;

  std::vector<std::vector<int>> m_local_pos_;
  std::vector<int> m_local_num_;
  std::vector<int> m_expert_id_map_;
  std::vector<ggml_bf16_t*> m_local_input_ptr_;
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  size_t pool_count_ = 0;
  size_t gate_up_ba_pool_bytes_ = 0;
  size_t gate_bc_pool_bytes_ = 0;
  size_t up_bc_pool_bytes_ = 0;
  size_t down_ba_pool_bytes_ = 0;
  size_t down_bc_pool_bytes_ = 0;
  void* gate_up_ba_pool_ = nullptr;
  void* gate_bc_pool_ = nullptr;
  void* up_bc_pool_ = nullptr;
  void* down_ba_pool_ = nullptr;
  void* down_bc_pool_ = nullptr;

  // aligned_alloc'd blocks owned by this MoE object (one per expert × gate/up/down).
  // Must be freed in the destructor — BufferB itself only wraps the raw pointer
  // and has no destructor, so a defaulted ~AMX_MOE_BASE() leaks ~3.6GB/layer
  // (256 experts × 3 matrices × ~4.7MB for MiniMax 1536×3072 INT8), which
  // accumulates to OOM after ~60 layers during GGUF→AMXINT8 conversion.
  // Same pattern as AVX2_MOE_BASE::owned_aligned_allocs_.
  std::vector<void*> owned_aligned_allocs_;

  GeneralMOEConfig config_;
  using input_t = ggml_bf16_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_MOE_BASE(GeneralMOEConfig config, int tp_part_idx_) : tp_part_idx(tp_part_idx_), config_(config) {
    init();
    derived()->derived_init();
  }

  void init() {
    if (config_.load && config_.path == "") {
      config_.load = false;
    }

    MemoryRequest mem_requests;
    const size_t ml = config_.max_len;
    const size_t k_tok = config_.num_experts_per_tok;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;
    mem_requests.append_pointer(&m_local_input_, sizeof(ggml_bf16_t) * k_tok * ml * H);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * k_tok * ml * I);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * k_tok * ml * H);

    m_local_pos_.resize(config_.max_len);
    for (int i = 0; i < config_.max_len; i++) {
      m_local_pos_[i].resize(config_.num_experts_per_tok);
    }
    m_expert_id_map_.resize(config_.expert_num);
    m_local_num_.resize(config_.expert_num);
    m_local_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(make_buffer_a(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(make_buffer_c(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(make_buffer_a(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(make_buffer_c(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      if (!gate_bb_ptr) throw std::runtime_error("aligned_alloc failed for gate BufferB");
      owned_aligned_allocs_.push_back(gate_bb_ptr);
      gate_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr = std::aligned_alloc(64, buffer_b_required_size(config_.intermediate_size, config_.hidden_size));
      if (!up_bb_ptr) throw std::runtime_error("aligned_alloc failed for up BufferB");
      owned_aligned_allocs_.push_back(up_bb_ptr);
      up_bb_.push_back(make_buffer_b(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, buffer_b_required_size(config_.hidden_size, config_.intermediate_size));
      if (!down_bb_ptr) throw std::runtime_error("aligned_alloc failed for down BufferB");
      owned_aligned_allocs_.push_back(down_bb_ptr);
      down_bb_.push_back(make_buffer_b(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
    }
    // TODO: need update to all *.hpp
    // (config_.expert_num * T::M_STEP) in pool_count_ is to ensure padding for each experts.
    pool_count_ = (size_t)config_.max_len * config_.num_experts_per_tok + config_.expert_num * T::M_STEP;

    gate_up_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;
    gate_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    up_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_ba_pool_bytes_ = buffer_a_required_size(pool_count_, config_.intermediate_size) + pool_count_ * 64;
    down_bc_pool_bytes_ = buffer_c_required_size(pool_count_, config_.hidden_size) + pool_count_ * 64;

    mem_requests.append_pointer(&gate_up_ba_pool_, gate_up_ba_pool_bytes_);
    mem_requests.append_pointer(&gate_bc_pool_, gate_bc_pool_bytes_);
    mem_requests.append_pointer(&up_bc_pool_, up_bc_pool_bytes_);
    mem_requests.append_pointer(&down_ba_pool_, down_ba_pool_bytes_);
    mem_requests.append_pointer(&down_bc_pool_, down_bc_pool_bytes_);

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  virtual ~AMX_MOE_BASE() {
    // Free the aligned_alloc'd BufferB weight blocks owned by this object.
    // BufferB::b/d are raw pointers with no destructor, so without this the
    // per-layer INT8/INT4 weights are leaked every time a MoE object is
    // destroyed (e.g. per-layer GGUF→AMXINT8 conversion loop). This is what
    // made RAM "eagerly fill up" around layer 60 of a 61-layer model.
    for (void* p : owned_aligned_allocs_) std::free(p);
    owned_aligned_allocs_.clear();
  }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(ggml_bf16_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

  virtual void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                     void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
  }

  // Down activation fill, hookable by derived (AMXINT4_SMART routes the alt A).
  // Virtual so the static-typed base pointer still dispatches. Default:
  // int8-KGroup quantize of the gate output into the int4 kernel's A.
  virtual void fill_down_a(int expert_idx, int m, ggml_bf16_t* src) {
    down_ba_[expert_idx]->from_mat(m, src, 0, 1);
  }

  // Down accumulation release, hookable by derived (AMXINT4_SMART routes the
  // alt C's to_mat). Default: reduce the int4 C into the bf16-paired output.
  virtual void down_output(int expert_idx, int m, ggml_bf16_t* dst, int ith, int nth) {
    down_bc_[expert_idx]->to_mat(m, dst, ith, nth);
  }

  template <typename... Args>
  void load_weights(Args&&... args) {
    derived()->load_weights(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void write_weights_to_buffer(Args&&... args) const {
    derived_const()->write_weights_to_buffer(std::forward<Args>(args)...);
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
#ifdef FORWARD_TIME_PROFILE
        max_local_num = std::max(max_local_num, m_local_num_[i]);
#endif
        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    size_t offset = 0;
    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_a = 0, used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;

    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];

      if (m_local_num_[i] == 0) {
        continue;
      }

      size_t max_m = (m_local_num_[i] + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[i]->max_m = max_m;
      gate_up_ba_[i]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);

      gate_bc_[i]->max_m = max_m;
      gate_bc_[i]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[i]->max_m = max_m;
      up_bc_[i]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[i]->max_m = max_m;
      down_ba_[i]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[i]->max_m = max_m;
      down_bc_[i]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_a += ba_size;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }

    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_a <= gate_up_ba_pool_bytes_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) {
          continue;
        }
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (ggml_bf16_t*)input + i * config_.hidden_size, sizeof(ggml_bf16_t) * config_.hidden_size);
      }
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      cpy_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    direct_or_pool(activated_expert, [this](int task_id) {
      int expert_idx = m_expert_id_map_[task_id];
      gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], 0, 1);
    });

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], ith, nth);
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

    apply_activation(activated_expert, nth, qlen);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          this->fill_down_a(expert_idx, m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx]);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          this->down_output(expert_idx, m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
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
        [this, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (config_.should_skip_expert(expert_ids[i * k + j])) {
                continue;
              }
              __m512 weight = _mm512_set1_ps(weights[i * k + j]);
              __m512 down_output0, down_output1;
              avx512_32xbf16_to_32xfp32((__m512i*)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
                                                   m_local_pos_[i][j] * config_.hidden_size + e),
                                        &down_output0, &down_output1);
              x0 = _mm512_fmadd_ps(down_output0, weight, x0);
              x1 = _mm512_fmadd_ps(down_output1, weight, x1);
            }
            auto f32out = (__m512*)((float*)output + i * config_.hidden_size + e);
            f32out[0] = x0;
            f32out[1] = x1;
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
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, prepare: %ld us, cpy_input: %ld us, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us, max_local_num: "
        "%d, qlen: %d\n",
        tp_part_idx, activated_expert, prepare_time, cpy_input_time, q_input_time, up_gate_time, act_time, q_down_time,
        down_time, weight_time, forward_total_time, max_local_num, qlen);
#endif
  }

  void forward_decode(int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    int qlen = 1;
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    long q_input_time = 0, up_gate_time = 0, act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
#endif

    int activated_expert = 0;
    std::fill(m_local_num_.begin(), m_local_num_.end(), 0);
    for (int i = 0; i < k; i++) {
      if (config_.should_skip_expert(expert_ids[i])) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
      m_local_pos_[0][i] = 0;
      m_local_num_[expert_ids[i]] = qlen;
      activated_expert++;
    }

    size_t offset = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      m_local_gate_output_ptr_[expert_idx] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[expert_idx] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[expert_idx] = m_local_down_output_ + offset * config_.hidden_size;
      offset += qlen;
    }

    void* gate_bc_pool_ptr = gate_bc_pool_;
    void* up_bc_pool_ptr = up_bc_pool_;
    void* down_ba_pool_ptr = down_ba_pool_;
    void* down_bc_pool_ptr = down_bc_pool_;
    constexpr size_t M_STEP = T::M_STEP;
    auto align64 = [](size_t v) { return (v + 63) & (~(size_t)63); };
    size_t used_pool_m = 0;
    size_t used_pool_bytes_bc_gate = 0, used_pool_bytes_bc_up = 0, used_pool_bytes_ba_down = 0,
           used_pool_bytes_bc_down = 0;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;

      gate_bc_[expert_idx]->max_m = max_m;
      gate_bc_[expert_idx]->set_data(gate_bc_pool_ptr);
      size_t bc_gate_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      gate_bc_pool_ptr = (void*)((uintptr_t)gate_bc_pool_ptr + bc_gate_size);

      up_bc_[expert_idx]->max_m = max_m;
      up_bc_[expert_idx]->set_data(up_bc_pool_ptr);
      size_t bc_up_size = align64(buffer_c_required_size(max_m, config_.intermediate_size));
      up_bc_pool_ptr = (void*)((uintptr_t)up_bc_pool_ptr + bc_up_size);

      down_ba_[expert_idx]->max_m = max_m;
      down_ba_[expert_idx]->set_data(down_ba_pool_ptr);
      size_t ba_down_size = align64(buffer_a_required_size(max_m, config_.intermediate_size));
      down_ba_pool_ptr = (void*)((uintptr_t)down_ba_pool_ptr + ba_down_size);

      down_bc_[expert_idx]->max_m = max_m;
      down_bc_[expert_idx]->set_data(down_bc_pool_ptr);
      size_t bc_down_size = align64(buffer_c_required_size(max_m, config_.hidden_size));
      down_bc_pool_ptr = (void*)((uintptr_t)down_bc_pool_ptr + bc_down_size);

      used_pool_m += max_m;
      used_pool_bytes_bc_gate += bc_gate_size;
      used_pool_bytes_bc_up += bc_up_size;
      used_pool_bytes_ba_down += ba_down_size;
      used_pool_bytes_bc_down += bc_down_size;
    }
    assert(used_pool_m <= pool_count_);
    assert(used_pool_bytes_bc_gate <= gate_bc_pool_bytes_);
    assert(used_pool_bytes_bc_up <= up_bc_pool_bytes_);
    assert(used_pool_bytes_ba_down <= down_ba_pool_bytes_);
    assert(used_pool_bytes_bc_down <= down_bc_pool_bytes_);

    void* gate_up_ba_pool_ptr = gate_up_ba_pool_;
    for (int i = 0; i < activated_expert; i++) {
      auto expert_idx = m_expert_id_map_[i];
      size_t max_m = (qlen + M_STEP - 1) / M_STEP * M_STEP;
      gate_up_ba_[expert_idx]->max_m = max_m;
      gate_up_ba_[expert_idx]->set_data(gate_up_ba_pool_ptr);
      size_t ba_size = align64(buffer_a_required_size(max_m, config_.hidden_size));
      gate_up_ba_pool_ptr = (void*)((uintptr_t)gate_up_ba_pool_ptr + ba_size);
      gate_up_ba_[expert_idx]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_input_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * activated_expert * 2, [](int _) { T::config(); },
        [this, nth, qlen](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
          if (do_up) {
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            gate_bc_[expert_idx]->to_mat(qlen, m_local_gate_output_ptr_[expert_idx], ith, nth);
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

    apply_activation(activated_expert, nth, qlen);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      act_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          this->fill_down_a(expert_idx, qlen, m_local_gate_output_ptr_[expert_idx]);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      q_down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, [](int _) { T::config(); },
        [this, nth, qlen](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          derived()->do_down_gemm(expert_idx, ith, nth, qlen);
          this->down_output(expert_idx, qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    for (int e = 0; e < config_.hidden_size; e += 32) {
      __m512 x0 = _mm512_setzero_ps();
      __m512 x1 = _mm512_setzero_ps();
      for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[j])) {
          continue;
        }
        __m512 weight = _mm512_set1_ps(weights[j]);
        __m512 down_output0, down_output1;
        avx512_32xbf16_to_32xfp32(
            (__m512i*)(m_local_down_output_ptr_[expert_ids[j]] + m_local_pos_[0][j] * config_.hidden_size + e),
            &down_output0, &down_output1);
        x0 = _mm512_fmadd_ps(down_output0, weight, x0);
        x1 = _mm512_fmadd_ps(down_output1, weight, x1);
      }
      auto f32out = (__m512*)((float*)output + e);
      f32out[0] = x0;
      f32out[1] = x1;
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    printf(
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }

 protected:
  Derived* derived() { return static_cast<Derived*>(this); }
  const Derived* derived_const() const { return static_cast<const Derived*>(this); }

  // ============================================================================
  // Derived class initialization hook
  // Called after base class init() completes, allows derived classes to perform
  // their own initialization that depends on base class being fully initialized
  // ============================================================================
  void derived_init() {
    // Default implementation does nothing - derived classes can override
  }

  // ============================================================================
  // Virtual points for buffer creation and size calculation
  // Default implementations use group_size (for KGroup quantization like K2)
  // Derived classes (like moe.hpp) can override to not use group_size
  // ============================================================================

  size_t buffer_a_required_size(size_t m, size_t k) const { return derived_const()->buffer_a_required_size_impl(m, k); }
  size_t buffer_b_required_size(size_t n, size_t k) const { return derived_const()->buffer_b_required_size_impl(n, k); }
  size_t buffer_c_required_size(size_t m, size_t n) const { return derived_const()->buffer_c_required_size_impl(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a(size_t m, size_t k, void* data) const {
    return derived_const()->make_buffer_a_impl(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b(size_t n, size_t k, void* data) const {
    return derived_const()->make_buffer_b_impl(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c(size_t m, size_t n, void* data) const {
    return derived_const()->make_buffer_c_impl(m, n, data);
  }

  void apply_activation(int activated_expert, int nth, int qlen) {
    apply_activation_to(activated_expert, nth, qlen, m_local_gate_output_ptr_);
  }

  void apply_activation_to(int activated_expert, int nth, int qlen,
                           const std::vector<ggml_bf16_t*>& destination_ptrs) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto fn = [this, nth, &destination_ptrs](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* destination_ptr = &destination_ptrs[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0, config_.swiglu_limit, config_.swiglu_alpha);
          __m512 result1 = amx::act_fn(gate_val1, up_val1, config_.swiglu_limit, config_.swiglu_alpha);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(destination_ptr + j));
        }
      }
    };

    if (activated_expert == 0) {
      return;
    }

    if (qlen < 10) {
      for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
        fn(task_id);
      }
    } else {
      pool->do_work_stealing_job(nth * activated_expert, nullptr, fn, nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_BASE derived classes
// ============================================================================

template <class T, class Derived>
class TP_MOE<AMX_MOE_BASE<T, Derived>>
    : public TP_MOE_Common<AMX_MOE_BASE<T, Derived>, Derived> {
 public:
  using TP_MOE_Common<AMX_MOE_BASE<T, Derived>, Derived>::TP_MOE_Common;

  // Default load_weights implementation - can be overridden by derived TP_MOE classes
  void load_weights() override { throw std::runtime_error("Not Implemented"); }

  void write_weight_scale_to_buffer(int gpu_tp_count, int gpu_experts_num,
                                    const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    throw std::runtime_error("Not Implemented");
  }

  void merge_results(int qlen, void* output, bool incremental) override {
    auto& config = this->config;
    auto& tp_count = this->tp_count;
    auto& local_output_numa = this->local_output_numa;
    auto& tp_configs = this->tp_configs;

    auto merge_fn = [this, output, incremental, &config, &tp_count, &local_output_numa, &tp_configs](int token_nth) {
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
      if (incremental) {
        for (int e = 0; e < config.hidden_size; e += 32) {
          __m512 x0, x1;
          avx512_32xbf16_to_32xfp32((__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e), &x0, &x1);
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), x0);
          *((__m512*)(merge_to + e + 16)) = _mm512_add_ps(*((__m512*)(merge_to + e + 16)), x1);
        }
      }
      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
          *((__m512*)(merge_to + e)) = _mm512_add_ps(*((__m512*)(merge_to + e)), *((__m512*)(merge_from + e)));
        }
      }
      // The output buffer is float32 (the production convention). The old
      // bf16 store packed 32 bf16 into 64 bytes — only HALF the float32
      // lanes, leaving the other half uninitialized (the garbage lanes).
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        *((__m512*)((float*)output + token_nth * config.hidden_size + e)) = x0;
        *((__m512*)((float*)output + token_nth * config.hidden_size + e + 16)) = x1;
      }
    };

    auto pool = config.pool;

    auto direct_or_pool = [&](int count, auto&& fn) {
      if (qlen < 10) {
        for (int i = 0; i < count; i++) {
          fn(i);
        }
      } else {
        pool->do_work_stealing_job(count, nullptr, fn, nullptr);
      }
    };

    direct_or_pool(qlen, merge_fn);
  }

  void merge_results(int qlen, void* output) override { merge_results(qlen, output, false); }
};

#endif  // CPUINFER_OPERATOR_AMX_MOE_BASE_H
