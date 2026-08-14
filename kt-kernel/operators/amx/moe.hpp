/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_MOE_H
#define CPUINFER_OPERATOR_AMX_MOE_H

// #define CHECK
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#include "moe_base.hpp"
#include "../gguf/dequant.hpp"

template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 protected:
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
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

#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
#endif

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                            size_t scale_size) {
    auto quant_path = prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt");
    std::ofstream of(quant_path, std::ios::binary);
    if (of.is_open() == false) {
      throw std::runtime_error("kt cache write failed (cannot open): " + quant_path.string());
    }
    of.write((char*)bb, size - scale_size);
    if (!of) {
      throw std::runtime_error("kt cache write failed (short write): " + quant_path.string());
    }
    of.close();
    auto scale_path = prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                std::to_string(scale_size) + "Byte" + "_scale_" + ".kt");
    of.open(scale_path, std::ios::binary);
    if (of.is_open() == false) {
      throw std::runtime_error("kt cache write failed (cannot open): " + scale_path.string());
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
    if (!of) {
      throw std::runtime_error("kt cache write failed (short write): " + scale_path.string());
    }
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    auto quant_path = prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt");
    std::ifstream f(quant_path, std::ios::binary);
    if (f.is_open() == false) {
      throw std::runtime_error("kt cache missing: " + quant_path.string());
    }
    const size_t quant_part = (size - scale_size) / mat_split;
    f.seekg(mat_split_idex * quant_part);
    f.read(((char*)bb) + mat_split_idex * quant_part, quant_part);
    if (!f) {
      throw std::runtime_error("kt cache short read: " + quant_path.string());
    }
    f.close();
    auto scale_path = prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                std::to_string(scale_size) + "Byte" + "_scale_" + ".kt");
    f.open(scale_path, std::ios::binary);
    if (f.is_open() == false) {
      throw std::runtime_error("kt cache missing: " + scale_path.string());
    }
    const size_t scale_part = scale_size / mat_split;
    f.seekg(mat_split_idex * scale_part);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_part, scale_part);
    if (!f) {
      throw std::runtime_error("kt cache short read: " + scale_path.string());
    }
  }
#ifdef CHECK
  inline void load_check() {
    memcpy(check_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
  }

  void verify_load_right() {
    // printf("varify down bb_0 %d\n", tp_part_idx);
    memcpy(verify_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
    // check if verify_bb_0 equal to check_bb_0
    if (memcmp(verify_bb, check_bb, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)) != 0) {
      printf("verify error\n");
      for (size_t i = 0; i < T::BufferB::required_size(config_.hidden_size, config_.intermediate_size); ++i) {
        if (verify_bb[i] != check_bb[i]) {
          printf("Difference at byte %zu: verify_bb_%d[%zu] = %02x, check_bb[%zu] = %02x\n", i, compare_expers, i,
                 (unsigned char)verify_bb[i], i, (unsigned char)check_bb[i]);
          break;  // find the first difference and exit
        }
      }
      assert(0);
    } else {
      printf("pass verify\n");
      // pick out the 100th~150th byte of scale to see
      printf("numa %d, verify_bb_%d:\n", tp_part_idx, compare_expers);
      size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
      size_t scale_size = config_.hidden_size * sizeof(float);
      for (size_t i = size - scale_size; i < size - scale_size + 50; ++i) {
        printf("%02x ", (unsigned char)verify_bb[i]);
      }
      printf("\n");
    }
  }
#endif

#ifdef FORWARD_TIME_REPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> last_now;
#endif

 public:
  AMX_MOE_TP() = default;

  AMX_MOE_TP(GeneralMOEConfig config, int tp_part_idx = 0) : Base(config, tp_part_idx) {
    // Initialization now happens in derived_init() which is called by base constructor
  }

  void derived_init() {
    printf("Creating AMX_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
    auto& load = config_.load;
    auto& save = config_.save;

    std::filesystem::path prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
    if (save) {
      // Atomic layer writes: files are written into <prefix>.tmp and renamed
      // over <prefix> only after the whole save job completes, so an
      // interrupted first boot can never leave a half-written layer behind.
      std::filesystem::create_directories(prefix);
      std::filesystem::path tmp_prefix = prefix;
      tmp_prefix += ".tmp";
      std::error_code ec;
      std::filesystem::remove_all(tmp_prefix, ec);
      std::filesystem::create_directories(tmp_prefix);
      std::cout << "Creating " << prefix << std::endl;
    }
    if (load) {
      if (std::filesystem::exists(prefix)) {
        std::cout << "Loading from " << prefix << std::endl;
      } else {
        throw std::runtime_error("Path not found: " + prefix.string());
      }
    }
  }

  ~AMX_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - no group_size
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const { return T::BufferB::required_size(n, k); }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  // ============================================================================
  // CRTP virtual points - GEMM dispatch
  // ============================================================================

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    int m = m_local_num_[expert_idx];
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
    int m = m_local_num_[expert_idx];
    auto& ba = down_ba_[expert_idx];
    auto& bb = down_bb_[expert_idx];
    auto& bc = down_bc_[expert_idx];

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul(m, config_.hidden_size, config_.intermediate_size, ba, bb, bc, ith, nth);
    }
  }
  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (config_.gate_projs.size()) {
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_id) {
            // printf("Load layer %d [%d/%d]\n", config_.layer_idx, expert_id, config_.expert_num);
            uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_id);
            {
              size_t scale_size = config_.intermediate_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;

              memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
              }

              memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
              }
            }

            {
              size_t scale_size = config_.hidden_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;

              memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][logical_expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
              }
            }
          },
          nullptr);

    } else {
      int nth = T::recommended_nth(config_.intermediate_size);
      static uint8_t mat_type_all = 3, mat_split = 1;
      std::filesystem::path prefix = config_.path;
      prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));

      if (config_.load) {
        std::cout << "Loading from \"" << prefix << "\"" << std::endl;
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all * mat_split,
            [this, physical_to_logical_map, prefix, mat_type_all, mat_split](int task_id) {
              int64_t expert_idx = task_id / (mat_type_all * mat_split);
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
              uint8_t mat_split_idex = task_id % mat_split;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                read_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              } else {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                read_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, logical_expert_id, size, scale_size,
                             mat_split, mat_split_idex);
              }
            });
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else
#endif
      {
        if constexpr (requires(typename T::BufferB bb, ggml_bf16_t* s, int i, int n) { bb.from_mat_strip(s, i, n); }) {
          if (config_.gate_gguf != nullptr) {
          // ================= online quant from gguf =================
          // Dequantize the 64-row strip each worker needs straight from the
          // mmap'd GGUF blocks and pack it immediately. Per-thread scratch:
          // N_BLOCK x k BF16 (~0.9 MB for gate/up at H=7168). There is never a
          // full BF16 (or FP32) copy of a layer, and the produced bytes are
          // identical to the "online quant from bf16" path for the same
          // BF16 values, so the disk cache is shared between both sources.
          if (tp_part_idx == 0) {
            std::cout << "  online quant from gguf" << std::endl;
          }
          const int tp_count = (int)config_.pool->config.subpool_count;
          const int64_t full_I = config_.gguf_full_intermediate_size > 0
                                     ? (int64_t)config_.gguf_full_intermediate_size
                                     : (int64_t)config_.intermediate_size * tp_count;
          // per-NUMA slice of each expert matrix: gate/up rows [row_off, row_off+I/tp),
          // down columns [row_off, row_off+I/tp) of each of its H rows (k = full I).
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
                  auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
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
          // down: per-NUMA matrix is [H, I/tp]; strips along H, columns sliced
          {
            int nth = T::recommended_nth(config_.hidden_size);
            pool->do_work_stealing_job(
                nth * config_.expert_num, nullptr,
                [this, nth, physical_to_logical_map, down_col_begin, down_col_end, full_I](int task_id) {
                  int64_t expert_idx = task_id / nth;
                  uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
                  int ith = task_id % nth;
                  auto [n_start, n_end] = T::split_range_n(config_.hidden_size, ith, nth);
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
        } else {
          if (tp_part_idx == 0) {
            std::cout << "  online quant from bf16" << std::endl;
          }
          pool->do_work_stealing_job(
              nth * config_.expert_num, nullptr,
              [this, nth, physical_to_logical_map](int task_id) {
                int64_t expert_idx = task_id / nth;
                uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
                int ith = task_id % nth;
                // gate part
                gate_bb_[logical_expert_id]->from_mat(
                    (ggml_bf16_t*)config_.gate_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                    ith, nth);
                // up part
                up_bb_[logical_expert_id]->from_mat(
                    (ggml_bf16_t*)config_.up_proj + logical_expert_id * config_.intermediate_size * config_.hidden_size,
                    ith, nth);
              },
              nullptr);

          nth = T::recommended_nth(config_.hidden_size);
          pool->do_work_stealing_job(
              nth * config_.expert_num, nullptr,
              [this, nth, physical_to_logical_map](int task_id) {
                int64_t expert_idx = task_id / nth;
                uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
                int ith = task_id % nth;
                // down part
                down_bb_[logical_expert_id]->from_mat(
                    (ggml_bf16_t*)config_.down_proj + logical_expert_id * config_.hidden_size * config_.intermediate_size,
                    ith, nth);
              },
              nullptr);
        }
        } else {
          throw std::runtime_error(
              "online quant from gguf requires BufferB::from_mat_strip support (AMXINT8/AMXINT4)");
        }
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        // Write into <prefix>.tmp (created by derived_init) and rename over
        // <prefix> only after every file is on disk, so an interrupted first
        // boot can never leave a half-written layer behind.
        std::filesystem::path tmp_prefix = prefix;
        tmp_prefix += ".tmp";
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this, physical_to_logical_map, tmp_prefix](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              expert_idx = expert_map(physical_to_logical_map, expert_idx);
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(tmp_prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(tmp_prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(tmp_prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size);
              }
            },
            nullptr);
        std::error_code ec;
        std::filesystem::remove_all(prefix, ec);
        std::filesystem::rename(tmp_prefix, prefix, ec);
        if (ec) {
          throw std::runtime_error("kt cache rename failed: " + tmp_prefix.string() + " -> " + prefix.string());
        }
      }
    }
  }

  // forward, forward_prefill, forward_decode, warm_up are inherited from Base
};

// ============================================================================
// TP_MOE specialization for AMX_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation
// ============================================================================

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16\n");
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);
                memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                       (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                           i * gate_up_elcount,
                       sizeof(ggml_bf16_t) * gate_up_elcount);
                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                             col * tpc.intermediate_size,
                         (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                             col * config.intermediate_size + i * tpc.intermediate_size,
                         sizeof(ggml_bf16_t) * tpc.intermediate_size);
                }
              },
              nullptr);
        }
      }

      DO_TPS_LOAD_WEIGHTS(pool);

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.gate_gguf != nullptr) {
      printf("From GGUF\n");
      // No per-NUMA BF16 copies: each TP part dequantizes its own strips
      // straight from the mmap'd GGUF blocks (see "online quant from gguf" in
      // AMX_MOE_TP::load_weights). The per-NUMA slice of each expert matrix is
      // derived from tp_part_idx * (I/tp_count).
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
      }
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file %s\n", config.path.c_str());
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_MOE_TP<K>>>
};

#endif
