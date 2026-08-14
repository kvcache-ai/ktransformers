/**
 * @Description  : K2 AMX MoE operator for Kimi-K2 native inference
 * @Author       : oql, Codex and Claude
 * @Date         : 2025-12-09
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements K2 Int4 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * K2 weights are stored with group-wise scales (KGroup Int4).
 **/
#ifndef CPUINFER_OPERATOR_AMX_K2_MOE_H
#define CPUINFER_OPERATOR_AMX_K2_MOE_H

// #define LOAD_TIME_PROFILE

#include "moe_base.hpp"
#include "../gguf/dequant.hpp"
#include "la/amx_raw_kernels.hpp"  // GemmKernel224BF16 (+ fp32 dot fallback)
#include "la/amx_raw_buffers.hpp"  // BufferBBF16Impl etc.

/**
 * @brief K2 Int4 MoE operator using CRTP pattern
 * @tparam T Kernel type, defaults to amx::GemmKernel224Int4SmallKGroup
 *
 * This class provides K2-specific GEMM implementations:
 * - do_gate_up_gemm: Int4 weight with KGroup scale + AMX GEMM
 * - do_down_gemm: Same Int4 KGroup GEMM
 * - load_weights: Load Int4 weights with group-wise scales
 */
template <class T = amx::GemmKernel224Int4SmallKGroup>
class AMX_K2_MOE_TP : public AMX_MOE_BASE<T, AMX_K2_MOE_TP<T>> {
 protected:
  using Base = AMX_MOE_BASE<T, AMX_K2_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;
  using Base::m_local_gate_output_ptr_;
  using Base::m_local_down_output_ptr_;
  using Base::m_local_input_;
  using Base::m_local_gate_output_;
  using Base::m_local_up_output_;
  using Base::m_local_down_output_;
  using Base::m_local_input_ptr_;
  using Base::m_local_up_output_ptr_;
  using Base::m_expert_id_map_;
  using Base::m_local_pos_;
  using Base::pool_count_;
  using Base::gate_up_ba_pool_;
  using Base::gate_bc_pool_;
  using Base::up_bc_pool_;
  using Base::down_ba_pool_;
  using Base::down_bc_pool_;
  using Base::gate_up_ba_pool_bytes_;
  using Base::gate_bc_pool_bytes_;
  using Base::up_bc_pool_bytes_;
  using Base::down_ba_pool_bytes_;
  using Base::down_bc_pool_bytes_;
  using Base::apply_activation;

  public:
    using typename Base::input_t;
    using typename Base::output_t;

    // ============================================================================
    // AMXINT4_SMART: the 3-GEMM layer graph.
    // Each attribute carries a precision tag (its "header"): 0 = Int4 KGroup,
    // 1 = Int8, 2 = BF16. Gate/up stay on the KGroup node (Q4_K/Q6_K tensors);
    // the down slot routes to the Int8 or BF16 node for Q8_0/BF16/F16/F32 down
    // tensors. The edges between nodes are the A-side conversions: quantize to
    // int8-KGroup (nodes 0/1) or fp32->bf16 copy (node 2).
    static constexpr int PREC_INT4 = 0;
    static constexpr int PREC_INT8 = 1;
    static constexpr int PREC_BF16 = 2;
    using Int8A = amx::GemmKernel224Int8::BufferA;
    using Int8B = amx::GemmKernel224Int8::BufferB;
    using Int8C = amx::GemmKernel224Int8::BufferC;
    using BF16A = amx::GemmKernel224BF16::BufferA;
    using BF16B = amx::GemmKernel224BF16::BufferB;
    using BF16C = amx::GemmKernel224BF16::BufferC;

    int gate_prec_ = PREC_INT4;
    int up_prec_ = PREC_INT4;
    int down_prec_ = PREC_INT4;

    // Alternate down trios (activated only when down_prec_ != PREC_INT4);
    // allocated in derived_init, freed in the destructor.
    std::vector<std::shared_ptr<Int8A>> down_ba8_;
    std::vector<std::shared_ptr<Int8B>> down_bb8_;
    std::vector<std::shared_ptr<Int8C>> down_bc8_;
    std::vector<std::shared_ptr<BF16A>> down_ba16_;
    std::vector<std::shared_ptr<BF16B>> down_bb16_;
    std::vector<std::shared_ptr<BF16C>> down_bc16_;
    std::vector<void*> alt_mem_blocks_;

  AMX_K2_MOE_TP() = default;

  AMX_K2_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("Kimi-K2 MoE only support KGroup Int4");
    }
    printf("Creating AMX_K2_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  virtual ~AMX_K2_MOE_TP() {
    for (auto p : alt_mem_blocks_) {
      free(p);
    }
  }

  // ============================================================================
  // CRTP buffer creation - with group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    // Dispatch based on qlen threshold
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (down_prec_ == PREC_BF16) {
      if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
        amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, down_ba16_[expert_idx],
                     down_bb16_[expert_idx], down_bc16_[expert_idx], ith, nth);
      } else {
        amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, down_ba16_[expert_idx],
                     down_bb16_[expert_idx], down_bc16_[expert_idx], ith, nth);
      }
      return;
    }
    if (down_prec_ == PREC_INT8) {
      if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
        amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, down_ba8_[expert_idx],
                     down_bb8_[expert_idx], down_bc8_[expert_idx], ith, nth);
      } else {
        amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, down_ba8_[expert_idx],
                     down_bb8_[expert_idx], down_bc8_[expert_idx], ith, nth);
      }
      return;
    }

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  // ============================================================================
  // AMXINT4_SMART forward hooks. The single-tp objects are statically typed as
  // AMX_MOE_BASE, whose forward is non-virtual — so overriding forward here
  // would never dispatch. Instead the base's forward calls these two CRTP
  // hooks (fill_down_a / down_output), which route the down slot by its
  // precision tag: Int8/BF16 nodes fill their alternate A and release their
  // alternate C; the Int4 KGroup node falls through to the base defaults.
  // ============================================================================

  void fill_down_a(int expert_idx, int m, ggml_bf16_t* src) {
    if (down_prec_ == PREC_BF16) {
      down_ba16_[expert_idx]->from_mat(m, src, 0, 1);
    } else if (down_prec_ == PREC_INT8) {
      down_ba8_[expert_idx]->from_mat(m, src, 0, 1);
    } else {
      Base::fill_down_a(expert_idx, m, src);
    }
  }

  void down_output(int expert_idx, int m, ggml_bf16_t* dst, int ith, int nth) {
    if (down_prec_ == PREC_BF16) {
      down_bc16_[expert_idx]->to_mat(m, dst, ith, nth);
    } else if (down_prec_ == PREC_INT8) {
      down_bc8_[expert_idx]->to_mat(m, dst, ith, nth);
    } else {
      Base::down_output(expert_idx, m, dst, ith, nth);
    }
  }

  /**
   * @brief Load Int4 weights from contiguous memory layout
   *
   * Loads weights from config_.gate_proj, up_proj, down_proj with scales
   * from config_.gate_scale, up_scale, down_scale.
   *
   * Note: K2 MOE only supports offline pre-quantized weights (gate_scale must be set).
   * For online quantization, use AWQ MOE instead.
   */
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size == 0 || quant_config.zero_point) {
      throw std::runtime_error("Kimi AVX MOE only support KGroup Int4.");
    }

    // AMXINT4_SMART: read the per-attribute precision headers (must run at
    // load time, AFTER construction — derived_init runs from the base ctor
    // where the derived members do not exist yet) and allocate the alternate
    // down buffer trio when the down slot is routed to the Int8/BF16 node.
    gate_prec_ = config_.gate_precision;
    up_prec_ = config_.up_precision;
    down_prec_ = config_.down_precision;
    if (gate_prec_ != PREC_INT4 || up_prec_ != PREC_INT4) {
      throw std::runtime_error("AMXINT4_SMART: gate/up precision tags must be Int4 KGroup in this build");
    }
    if (down_prec_ != PREC_INT4 && down_prec_ != PREC_INT8 && down_prec_ != PREC_BF16) {
      throw std::runtime_error("AMXINT4_SMART: invalid down precision tag");
    }
    if (down_prec_ != PREC_INT4 && down_bb16_.empty() && down_bb8_.empty()) {
      const char* prec_names[3] = {"INT4_KGROUP", "INT8", "BF16"};
      if (tp_part_idx == 0) {
        std::cout << "  AMXINT4_SMART down node: " << prec_names[down_prec_] << std::endl;
      }
      const int e = (int)config_.expert_num;
      const int n = config_.hidden_size;
      const int k = config_.intermediate_size;
      auto alloc_block = [this](size_t sz) -> void* {
        void* p = nullptr;
        if (posix_memalign(&p, 64, std::max<size_t>(sz, 1)) != 0) throw std::bad_alloc();
        alt_mem_blocks_.push_back(p);
        return p;
      };
      if (down_prec_ == PREC_INT8) {
        for (int i = 0; i < e; i++) {
          down_ba8_.push_back(std::make_shared<Int8A>(config_.max_len, k, alloc_block(Int8A::required_size(config_.max_len, k))));
          down_bb8_.push_back(std::make_shared<Int8B>(n, k, alloc_block(Int8B::required_size(n, k))));
          down_bc8_.push_back(std::make_shared<Int8C>(config_.max_len, n, alloc_block(Int8C::required_size(config_.max_len, n))));
        }
      } else {  // PREC_BF16
        for (int i = 0; i < e; i++) {
          down_ba16_.push_back(std::make_shared<BF16A>(config_.max_len, k, alloc_block(BF16A::required_size(config_.max_len, k))));
          down_bb16_.push_back(std::make_shared<BF16B>(n, k, alloc_block(BF16B::required_size(n, k))));
          down_bc16_.push_back(std::make_shared<BF16C>(config_.max_len, n, alloc_block(BF16C::required_size(config_.max_len, n))));
        }
      }
    }

    // ================= online quant from gguf (K2 KGroup) =================
    // Same strip-dequant flow as AMX_MOE_TP::load_weights: each worker
    // dequantizes only the rows/columns it packs, straight from the mmap'd
    // GGUF blocks. Per k-group scales are computed inside from_mat_strip.
    if constexpr (requires(typename T::BufferB& bb, ggml_bf16_t* s, int i, int n) { bb.from_mat_strip(s, i, n); }) {
      if (config_.gate_gguf != nullptr) {
      if (tp_part_idx == 0) {
        std::cout << "  online quant from gguf (K2 KGroup, group_size=" << group_size << ")" << std::endl;
      }
      const int tp_count = (int)config_.pool->config.subpool_count;
      const int64_t full_I = config_.gguf_full_intermediate_size > 0
                                 ? (int64_t)config_.gguf_full_intermediate_size
                                 : (int64_t)config_.intermediate_size * tp_count;
      const int64_t row_off = (int64_t)config_.intermediate_size * tp_part_idx;
      const int64_t down_col_begin = row_off;
      const int64_t down_col_end = row_off + config_.intermediate_size;
      // gate/up: per-NUMA matrix is [I/tp, H]; strips along I/tp, full columns
      {
        int nth = T::recommended_nth(config_.intermediate_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map, row_off](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              auto [n_start, n_end] =
                  T::BufferB::split_range_n(config_.intermediate_size, ith, nth);
              if (n_start >= n_end) return;
              thread_local std::vector<ggml_bf16_t> strip;
              strip.resize((size_t)(n_end - n_start) * config_.hidden_size);
              const char* gate_base = (const char*)config_.gate_gguf + logical_expert_id * config_.gate_gguf_stride;
              kt::gguf::dequant_rows_bf16(gate_base, (ggml_type)config_.gate_gguf_type, config_.hidden_size,
                                          row_off + n_start, row_off + n_end, strip.data());
              gate_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
              const char* up_base = (const char*)config_.up_gguf + logical_expert_id * config_.up_gguf_stride;
              kt::gguf::dequant_rows_bf16(up_base, (ggml_type)config_.up_gguf_type, config_.hidden_size,
                                          row_off + n_start, row_off + n_end, strip.data());
              up_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            },
            nullptr);
      }
      // down: per-NUMA matrix is [H, I/tp]; strips along H, columns sliced.
      // Routed by the precision tag: Int4 KGroup (default), Int8 or BF16 node.
      if (down_prec_ == PREC_BF16) {
            int nth = amx::GemmKernel224BF16::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map, down_col_begin, down_col_end, full_I](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              auto [n_start, n_end] = amx::GemmKernel224BF16::split_range_n(config_.hidden_size, ith, nth);
              if (n_start >= n_end) return;
              const int64_t dcol = down_col_end - down_col_begin;
              thread_local std::vector<ggml_bf16_t> strip;
              strip.resize((size_t)(n_end - n_start) * dcol);
              const char* down_base =
                  (const char*)config_.down_gguf + logical_expert_id * config_.down_gguf_stride;
              kt::gguf::dequant_rows_bf16(down_base, (ggml_type)config_.down_gguf_type, full_I, n_start, n_end,
                                          down_col_begin, down_col_end, strip.data());
              down_bb16_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            },
            nullptr);
      } else if (down_prec_ == PREC_INT8) {
        int nth = amx::GemmKernel224Int8::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map, down_col_begin, down_col_end, full_I](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              auto [n_start, n_end] = amx::GemmKernel224Int8::split_range_n(config_.hidden_size, ith, nth);
              if (n_start >= n_end) return;
              const int64_t dcol = down_col_end - down_col_begin;
              thread_local std::vector<ggml_bf16_t> strip;
              strip.resize((size_t)(n_end - n_start) * dcol);
              const char* down_base =
                  (const char*)config_.down_gguf + logical_expert_id * config_.down_gguf_stride;
              kt::gguf::dequant_rows_bf16(down_base, (ggml_type)config_.down_gguf_type, full_I, n_start, n_end,
                                          down_col_begin, down_col_end, strip.data());
              down_bb8_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            },
            nullptr);
      } else {
      int nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map, down_col_begin, down_col_end, full_I](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              int ith = task_id % nth;
              auto [n_start, n_end] =
                  T::BufferB::split_range_n(config_.hidden_size, ith, nth);
              if (n_start >= n_end) return;
              const int64_t dcol = down_col_end - down_col_begin;
              thread_local std::vector<ggml_bf16_t> strip;
              strip.resize((size_t)(n_end - n_start) * dcol);
              const char* down_base =
                  (const char*)config_.down_gguf + logical_expert_id * config_.down_gguf_stride;
              kt::gguf::dequant_rows_bf16(down_base, (ggml_type)config_.down_gguf_type, full_I, n_start, n_end,
                                          down_col_begin, down_col_end, strip.data());
              down_bb_[logical_expert_id]->from_mat_strip(strip.data(), ith, nth);
            },
            nullptr);
      }
      return;
      }
    } else if (config_.gate_gguf != nullptr) {
      throw std::runtime_error(
          "K2 GGUF path requires BufferB::from_mat_strip (use AMXInt4_KGroup_MOE / SmallKGroup kernels)");
    }

    if (config_.gate_scale == nullptr) {
      throw std::runtime_error("Kimi AVX MOE only support load native weight.");
    }

    // load weight
    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // gate part
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          // up part
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // down part
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;

          // convert scales from BF16 to FP32
          convert_or_copy(gate_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(up_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.up_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
          convert_or_copy(down_bb_[expert_idx]->d,
                          (ggml_bf16_t*)config_.down_scale + (logical_expert_id * scale_elem_count), scale_elem_count);
        },
        nullptr);
#ifdef DEBUG_K2_MOE
    dump_buffer_b("native", 0, "down", down_bb_[0].get());
#endif
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;

    // Main loop: 512-bit (64-byte) SIMD copies
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      __m512i data = _mm512_loadu_si512((__m512i*)s);
      _mm512_storeu_si512((__m512i*)d, data);
      d += 64;
      s += 64;
    }
    bytes -= chunks * 64;

    // Handle remaining bytes
    if (bytes > 0) {
      std::memcpy(d, s, bytes);
    }
  }

  // Optimized SIMD float32 to bf16 conversion
  static inline void fast_fp32_to_bf16(ggml_bf16_t* __restrict dst, const float* __restrict src, size_t count) {
    size_t i = 0;

    // Process 32 elements at a time (2x __m512, output 1x __m512i = 32 bf16)
    for (; i + 32 <= count; i += 32) {
      __m512 v0 = _mm512_loadu_ps(src + i);
      __m512 v1 = _mm512_loadu_ps(src + i + 16);

      // Convert to bf16 using truncation (shift right 16 bits)
      __m512i i0 = _mm512_srli_epi32(_mm512_castps_si512(v0), 16);
      __m512i i1 = _mm512_srli_epi32(_mm512_castps_si512(v1), 16);

      // Pack 32-bit values to 16-bit
      __m512i packed = _mm512_packus_epi32(i0, i1);

      // Reorder due to packus lane behavior:
      // packus outputs interleaved: [i0[0-3], i1[0-3], i0[4-7], i1[4-7], i0[8-11], i1[8-11], i0[12-15], i1[12-15]]
      // We need sequential: [i0[0-15], i1[0-15]] = [i0[0-3], i0[4-7], i0[8-11], i0[12-15], i1[0-3], i1[4-7], i1[8-11],
      // i1[12-15]] Permutation: [0, 2, 4, 6, 1, 3, 5, 7] (qword indices)
      __m512i permuted = _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), packed);

      _mm512_storeu_si512((__m512i*)(dst + i), permuted);
    }

    // Handle remaining elements with scalar conversion
    for (; i < count; i++) {
      dst[i] = ggml_fp32_to_bf16(src[i]);
    }
  }


  void write_weights_to_buffer_blocked(int gpu_tp_count, [[maybe_unused]] int cpu_tp_count, int expert_id,
                                       const GeneralMOEConfig& full_config,
                                       const std::vector<uintptr_t>& w13_weight_ptrs,
                                       const std::vector<uintptr_t>& w13_scale_ptrs,
                                       const std::vector<uintptr_t>& w2_weight_ptrs,
                                       const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    constexpr int NUM_W13_TASKS = 32;
    constexpr int NUM_W2_TASKS = 32;
    const int total_tasks = NUM_W13_TASKS + NUM_W2_TASKS;

    const int cpu_n_w13 = config_.intermediate_size;
    const int cpu_k_w13 = config_.hidden_size;
    const int gpu_n_w13 = full_config.intermediate_size / gpu_tp_count;
    const int gpu_k_w13 = full_config.hidden_size;
    const int global_n_offset_w13 = tp_part_idx * cpu_n_w13;
    const size_t gpu_w13_weight_per_mat = static_cast<size_t>(gpu_n_w13) * gpu_k_w13 / 2;
    const size_t gpu_w13_scale_per_mat = static_cast<size_t>(gpu_n_w13) * (gpu_k_w13 / group_size);

    const int cpu_n_w2 = config_.hidden_size;
    const int cpu_k_w2 = config_.intermediate_size;
    const int gpu_k_w2 = full_config.intermediate_size / gpu_tp_count;
    const int global_k_offset_w2 = tp_part_idx * cpu_k_w2;

    pool->do_work_stealing_job(
        total_tasks, nullptr,
        [=, &w13_weight_ptrs, &w13_scale_ptrs, &w2_weight_ptrs, &w2_scale_ptrs, this](int task_id) {
          if (task_id < NUM_W13_TASKS) {
            const int rows_per_task = (cpu_n_w13 + NUM_W13_TASKS - 1) / NUM_W13_TASKS;
            const int row_start = task_id * rows_per_task;
            const int row_end = std::min(row_start + rows_per_task, cpu_n_w13);
            for (int local_n = row_start; local_n < row_end; local_n++) {
              const int global_n = global_n_offset_w13 + local_n;
              const int target_gpu = global_n / gpu_n_w13;
              const int n_in_gpu = global_n % gpu_n_w13;

              uint8_t* w13_weight_base = reinterpret_cast<uint8_t*>(w13_weight_ptrs[target_gpu]);
              ggml_bf16_t* w13_scale_base = reinterpret_cast<ggml_bf16_t*>(w13_scale_ptrs[target_gpu]);
              const size_t weight_row_offset = static_cast<size_t>(n_in_gpu) * gpu_k_w13 / 2;
              const size_t scale_row_offset = static_cast<size_t>(n_in_gpu) * (gpu_k_w13 / group_size);

              gate_bb_[expert_id]->copy_weight_rows_to(w13_weight_base + weight_row_offset, local_n, 1, 0, cpu_k_w13,
                                                        gpu_k_w13 / 2);
              up_bb_[expert_id]->copy_weight_rows_to(w13_weight_base + gpu_w13_weight_per_mat + weight_row_offset,
                                                      local_n, 1, 0, cpu_k_w13, gpu_k_w13 / 2);

              gate_bb_[expert_id]->copy_scale_rows_to(w13_scale_base + scale_row_offset, local_n, 1, 0,
                                                       cpu_k_w13 / group_size, gpu_k_w13 / group_size);
              up_bb_[expert_id]->copy_scale_rows_to(w13_scale_base + gpu_w13_scale_per_mat + scale_row_offset, local_n,
                                                     1, 0, cpu_k_w13 / group_size, gpu_k_w13 / group_size);
            }
            return;
          }

          const int w2_task_id = task_id - NUM_W13_TASKS;
          const int rows_per_task = (cpu_n_w2 + NUM_W2_TASKS - 1) / NUM_W2_TASKS;
          const int row_start = w2_task_id * rows_per_task;
          const int row_end = std::min(row_start + rows_per_task, cpu_n_w2);
          for (int row = row_start; row < row_end; row++) {
            int k_local = 0;
            while (k_local < cpu_k_w2) {
              const int global_k = global_k_offset_w2 + k_local;
              const int target_gpu = global_k / gpu_k_w2;
              const int k_in_gpu = global_k % gpu_k_w2;
              const int k_count = std::min(cpu_k_w2 - k_local, gpu_k_w2 - k_in_gpu);

              uint8_t* w2_weight_base = reinterpret_cast<uint8_t*>(w2_weight_ptrs[target_gpu]);
              ggml_bf16_t* w2_scale_base = reinterpret_cast<ggml_bf16_t*>(w2_scale_ptrs[target_gpu]);
              uint8_t* weight_dst = w2_weight_base + static_cast<size_t>(row) * gpu_k_w2 / 2 + k_in_gpu / 2;
              ggml_bf16_t* scale_dst =
                  w2_scale_base + static_cast<size_t>(row) * (gpu_k_w2 / group_size) + k_in_gpu / group_size;

              down_bb_[expert_id]->copy_weight_rows_to(weight_dst, row, 1, k_local, k_count, gpu_k_w2 / 2);
              down_bb_[expert_id]->copy_scale_rows_to(scale_dst, row, 1, k_local / group_size, k_count / group_size,
                                                       gpu_k_w2 / group_size);
              k_local += k_count;
            }
          }
        },
        nullptr);
  }

  // Write a single expert's weights to the output buffers
  // The caller provides pointers that already point to the target expert's location (no offset needed)
  // expert_id: the index of the expert to write
  // Optimized for maximum memory bandwidth using streaming stores
  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    if constexpr (T::BLOCKED_B_LAYOUT) {
      write_weights_to_buffer_blocked(gpu_tp_count, cpu_tp_count, expert_id, full_config, w13_weight_ptrs,
                                      w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
      return;
    }

    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    // Calculate sizes for CPU TP part (this instance)
    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;  // int4 packing
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    // Calculate sizes for GPU TP part
    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;  // int4 packing
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    // Determine mapping: which GPU TP parts should this CPU TP part write to?
    // Since weights are col-major and we slice directly by memory order:
    // - If cpu_tp_count >= gpu_tp_count: multiple(or one) CPU TPs write to one GPU TP
    // - If cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
    if (cpu_tp_count >= gpu_tp_count) {
      // Multiple CPU TPs map to one GPU TP
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      // Get pointers for this GPU TP part (already pointing to target expert's location)
      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      // Calculate offset within the GPU TP buffer (for CPU TP slice within GPU TP)
      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      // Optimized task layout for maximum bandwidth:
      // - Larger chunks to reduce task overhead
      // - Separate large contiguous copies (gate_w, up_w) from strided copies (down)
      // - Scale conversions are relatively small, merge with weight tasks

      // Use fewer, larger tasks for better efficiency
      constexpr int NUM_WEIGHT_TASKS = 8;  // Fewer tasks, larger chunks
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);

      // Total tasks: gate_weight + up_weight + down_weight_scale + gate_scale + up_scale
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;

      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      // Align chunk size to 64 bytes for optimal streaming stores
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              // Gate weight copy - chunked
              int chunk_idx = task_id;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end) {
                uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b;
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + start, gate_weight_src + start, end - start);
              }
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              // Up weight copy - chunked
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end) {
                uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b;
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start, up_weight_src + start,
                            end - start);
              }
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              // Down columns - split by column chunks
              // Each task handles multiple consecutive columns for better cache locality
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; col++) {
                fast_memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            (uint8_t*)down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);

                fast_fp32_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                                  down_bb_[expert_id]->d + col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              // Gate scale convert
              float* gate_scale_src = gate_bb_[expert_id]->d;
              fast_fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale, gate_scale_src, cpu_tp_scale_elem_count);
            } else {
              // Up scale convert
              float* up_scale_src = up_bb_[expert_id]->d;
              fast_fp32_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, up_scale_src,
                                cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      // cpu_tp_count < gpu_tp_count: one CPU TP writes to multiple GPU TPs
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      // Size of data per GPU TP within this CPU TP
      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      // Optimized task layout
      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);

      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;

      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            // Get pointers for this GPU TP part
            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            // Calculate offsets within CPU TP buffers
            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              // Gate weight copy - chunked
              int chunk_idx = task_type;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end) {
                uint8_t* gate_weight_src = (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight;
                fast_memcpy(w13_weight_dst + start, gate_weight_src + start, end - start);
              }
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              // Up weight copy - chunked
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end) {
                uint8_t* up_weight_src = (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight;
                fast_memcpy(w13_weight_dst + gpu_tp_weight_bytes + start, up_weight_src + start, end - start);
              }
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              // Down columns - split by column chunks
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

                fast_memcpy(w2_weight_dst + col * weight_per_gpu_col,
                            (uint8_t*)down_bb_[expert_id]->b + col_offset_weight, weight_per_gpu_col);

                fast_fp32_to_bf16(w2_scale_dst + col * scale_per_gpu_col, down_bb_[expert_id]->d + col_offset_scale,
                                  scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              // Gate scale convert
              float* gate_scale_src = gate_bb_[expert_id]->d + cpu_offset_scale;
              fast_fp32_to_bf16(w13_scale_dst, gate_scale_src, data_per_gpu_tp_scale);
            } else {
              // Up scale convert
              float* up_scale_src = up_bb_[expert_id]->d + cpu_offset_scale;
              fast_fp32_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, up_scale_src, data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_K2_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation.
// NOTE: the base TP_MOE<AMX_MOE_BASE<T, Derived>> specialization (moe_base.hpp)
// now sizes the single-tp objects as the FULL Derived via TP_MOE_Common's
// Concrete parameter, so derived classes (AMX_K2_MOE_TP carries the SMART
// precision tags and alternate down trios) allocate in-bounds. Do NOT add a
// second specialization for AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>> here — it would
// shadow the merge_results provider and make the outer abstract.
// ============================================================================

template <typename K>
class TP_MOE<AMX_K2_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

#ifdef LOAD_TIME_PROFILE
    auto load_start_time = std::chrono::high_resolution_clock::now();
    auto load_last = load_start_time;
    long alloc_and_tp_slice_time = 0, tps_load_time = 0, cleanup_time = 0;
#endif

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    // GGUF source: no raw int4 buffers to slice — each TP part dequantizes its
    // own strips from the mmap'd GGUF blocks (see "online quant from gguf" in
    // AMX_K2_MOE_TP::load_weights) and computes per-k-group scales itself.
    if (config.gate_gguf != nullptr) {
      if (config.quant_config.group_size == 0) {
        throw std::runtime_error("K2 GGUF path requires quant_config.group_size");
      }
      // The K2 kernels apply scales at 16-lane granularity (make_kblock_abscale
      // quarters); 32 is the validated production width (16 hits an A-side
      // quantization gap, coarser widths are covered by the int16/fp32 bounds).
      if (config.quant_config.group_size != 32) {
        throw std::runtime_error("K2 GGUF path requires k_group_size == 32 (got " +
                                 std::to_string(config.quant_config.group_size) + ")");
      }
      printf("From GGUF (K2 KGroup)\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        tpc.gate_gguf = config.gate_gguf;
        tpc.up_gguf = config.up_gguf;
        tpc.down_gguf = config.down_gguf;
        tpc.gate_gguf_stride = config.gate_gguf_stride;
        tpc.up_gguf_stride = config.up_gguf_stride;
        tpc.down_gguf_stride = config.down_gguf_stride;
        tpc.gate_gguf_type = config.gate_gguf_type;
        tpc.up_gguf_type = config.up_gguf_type;
        tpc.down_gguf_type = config.down_gguf_type;
        tpc.gguf_full_intermediate_size = config.gguf_full_intermediate_size;
        tpc.quant_config.group_size = config.quant_config.group_size;
      }
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
      return;
    }

    if (config.gate_projs.empty() && config.gate_scale == nullptr) {
      throw std::runtime_error("K2 MoE only supports Packed Int4 with KGroup Scale");
    }

    if (use_per_expert_ptrs) {
      printf("From per-expert pointers (gate_projs)\n");
    } else {
      printf("From Packed Int4 with KGroup Scale\n");
    }

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
      tpc.gate_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
      tpc.up_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];
      tpc.down_scale = new ggml_bf16_t[(tpc.expert_num * scales_elem_count)];

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              ggml_bf16_t* src_gate_scale = (ggml_bf16_t*)config.gate_scales[0][expert_id];
              ggml_bf16_t* src_up_scale = (ggml_bf16_t*)config.up_scales[0][expert_id];
              ggml_bf16_t* src_down_scale = (ggml_bf16_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));

              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));

              memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), sizeof(ggml_bf16_t) * scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((ggml_bf16_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((ggml_bf16_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                memcpy((ggml_bf16_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (ggml_bf16_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       sizeof(ggml_bf16_t) * scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (sizeof(uint8_t) * tpc.intermediate_size) >> 1);
                  memcpy((ggml_bf16_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (ggml_bf16_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         sizeof(ggml_bf16_t) * (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done.\n", i);
    });

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      alloc_and_tp_slice_time =
          std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
      load_last = load_now_time;
    }
#endif

    DO_TPS_LOAD_WEIGHTS(pool);

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      tps_load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
      load_last = load_now_time;
    }
#endif

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      delete[] (uint8_t*)(tpc.gate_proj);
      delete[] (uint8_t*)(tpc.up_proj);
      delete[] (uint8_t*)(tpc.down_proj);

      delete[] (ggml_bf16_t*)(tpc.gate_scale);
      delete[] (ggml_bf16_t*)(tpc.up_scale);
      delete[] (ggml_bf16_t*)(tpc.down_scale);
    });

#ifdef LOAD_TIME_PROFILE
    {
      auto load_now_time = std::chrono::high_resolution_clock::now();
      cleanup_time = std::chrono::duration_cast<std::chrono::microseconds>(load_now_time - load_last).count();
    }
    auto load_end_time = std::chrono::high_resolution_clock::now();
    auto load_total_time =
        std::chrono::duration_cast<std::chrono::microseconds>(load_end_time - load_start_time).count();
    printf(
        "[K2 MoE Load Weights] tp_count: %d, alloc_and_tp_slice: %ld us, tps_load_weights: %ld us, cleanup: %ld us, "
        "total: %ld us\n",
        tp_count, alloc_and_tp_slice_time, tps_load_time, cleanup_time, load_total_time);
#endif

    this->weights_loaded = true;
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (this->weights_loaded == false) {
      throw std::runtime_error("Not Loaded");
    }
    if (this->tps.empty()) {
      throw std::runtime_error("No TP parts initialized");
    }

    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count) {
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");
    }

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_K2_MOE_TP<K>>>
};

#endif  // CPUINFER_OPERATOR_AMX_K2_MOE_H
