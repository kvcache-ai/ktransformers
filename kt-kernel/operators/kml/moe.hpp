#ifndef KML_MOE_HPP
#define KML_MOE_HPP

#include <kblas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../moe-tp.hpp"
#include "la/arm_kml.hpp"
#include "la/batch_gemm_api.hpp"
#include "la/utils.hpp"
#include "llama.cpp/ggml.h"

template <class T>
class KML_MOE_TP
#ifdef FORWARD_TIME_PROFILE
    : protected TimePerf
#endif
{
 private:
  int tp_part_idx;
  std::filesystem::path prefix;

  void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if
                     // quantized)]

  ggml_bf16_t* m_local_input_;  // [routed_expert_num * max_len * hidden_size]
  float* m_local_gate_output_;  // [routed_expert_num * max_len * intermediate_size]
  float* m_local_up_output_;    // [routed_expert_num * max_len * intermediate_size]
  float* m_local_down_output_;  // [routed_expert_num * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;    // [max_len, routed_expert_num]
  std::vector<int> m_local_num_;                 // [expert_num]
  std::vector<int> m_expert_id_map_;             // [expert_num]
  std::vector<ggml_bf16_t*> m_local_input_ptr_;  // [expert_num]
  std::vector<float*> m_local_gate_output_ptr_;  // [expert_num]
  std::vector<float*> m_local_up_output_ptr_;    // [expert_num]
  std::vector<float*> m_local_down_output_ptr_;  // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;

  inline void write_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                            size_t scale_size) {
    // printf("expert %d, size %ld, scale size %ld\n", expert_idx, size, scale_size);
    // std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file: %s", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                            std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                     .c_str());
      // throw std::runtime_error("No such file");
    }
    of.write((char*)bb, size - scale_size);
    of.close();
    // of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (of.is_open() == false) {
      printf("no such file\n");
      // throw std::runtime_error("No such file");
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    // std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx)  + "_quant_" + ".kt"));
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * (size - scale_size) / mat_split);
    f.read(((char*)bb) + mat_split_idex * (size - scale_size) / mat_split, (size - scale_size) / mat_split);
    f.close();
    // f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_scale_" + ".kt"));
    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                     "_scale_" + ".kt"));
    if (f.is_open() == false) {
      printf("no such file: %s\n", (prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                                              std::to_string(scale_size) + "Byte" + "_scale_" + ".kt"))
                                       .c_str());
      // throw std::runtime_error("No such file");
    }
    f.seekg(mat_split_idex * scale_size / mat_split);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_size / mat_split, scale_size / mat_split);
  }

 public:
  using input_t = ggml_bf16_t;
  using output_t = float;

  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  KML_MOE_TP(GeneralMOEConfig config, int tp_part_idx) {
    printf("  Creating KML_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
    auto& load = config.load;
    auto& save = config.save;
    if (load && config.path == "") {
      load = false;
    }

    prefix = config.path;
    prefix = prefix / ("_layer_" + std::to_string(config.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx));
    if (save) {
      std::cout << "Creating " << prefix << std::endl;
      std::filesystem::create_directories(prefix);
    }
    if (load) {
      if (std::filesystem::exists(prefix)) {
        std::cout << "Loading from " << prefix << std::endl;
      } else {
        throw std::runtime_error("Path not found: " + prefix.string());
      }
    }

    this->tp_part_idx = tp_part_idx;
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    MemoryRequest mem_requests;
    mem_requests.append_pointer(&m_local_input_,
                                sizeof(input_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);

    mem_requests.append_pointer(&m_local_gate_output_, sizeof(float) * config_.num_experts_per_tok * config_.max_len *
                                                           config_.intermediate_size);
    mem_requests.append_pointer(
        &m_local_up_output_, sizeof(float) * config_.num_experts_per_tok * config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_,
                                sizeof(float) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);

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

    // printf("tp part %d alloc layer %d, %f GB, on numa %d\n", tp_part_idx, config_.layer_idx,
    //        1e-9 * config_.expert_num *
    //            (T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) * 2 +
    //             T::BufferB::required_size(config_.hidden_size, config_.intermediate_size)),
    //        numa_node_of_cpu(sched_getcpu()));
    // 统一分配一块巨大的内存用于权重：
    size_t gate_up_exp_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) +
                              T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);

    for (uint64_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, nullptr));
      void* gate_up_down_all_exp_ptr = std::aligned_alloc(64, gate_up_exp_size);

      // void *gate_bb_ptr =
      //     std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      // gate_bb_.push_back(
      //     std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr, true));

      // void *up_bb_ptr =
      //     std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      // up_bb_.push_back(
      //     std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr, true));

      // void *down_bb_ptr =
      //     std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      // down_bb_.push_back(
      //     std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr, true));

      gate_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size,
                                                               gate_up_down_all_exp_ptr, true));
      up_bb_.push_back(std::make_shared<typename T::BufferB>(
          config_.intermediate_size, config_.hidden_size,
          offset_pointer(gate_up_down_all_exp_ptr,
                         T::BufferB::required_size(config_.intermediate_size, config_.hidden_size)),
          true));

      void* down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr, true));
    }

    for (int i = 0; i < config_.expert_num; i++) {
      mem_requests.append_function([this, i](void* new_ptr) { gate_up_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.hidden_size));
      mem_requests.append_function([this, i](void* new_ptr) { gate_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { up_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.hidden_size));
    }

    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  KML_MOE_TP(const KML_MOE_TP&) = delete;
  KML_MOE_TP& operator=(const KML_MOE_TP&) = delete;
  KML_MOE_TP(KML_MOE_TP&&) = delete;
  KML_MOE_TP& operator=(KML_MOE_TP&&) = delete;

  ~KML_MOE_TP() {
    // printf("  Destroying KML_MOE_TP %lx\n", (intptr_t)(this));
  }

  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_projs.size()) {
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this](int expert_id) {
            printf("Load layer %d [%d/%d]\n", config_.layer_idx, expert_id, config_.expert_num);
            {
              size_t scale_size = config_.intermediate_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size) - scale_size;

              memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][expert_id], scale_size);
              }

              memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][expert_id], scale_size);
              }
            }

            {
              size_t scale_size = config_.hidden_size * sizeof(float);
              size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size) - scale_size;

              memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][expert_id], size);

              if constexpr (T::BufferB::SCALE) {
                memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][expert_id], scale_size);
              }
            }
          },
          nullptr);

    } else {
      static uint8_t mat_type_all = 3, mat_split = 1;
      if (config_.load) {
        std::cout << "Loading from " << prefix << std::endl;
        for (int task_id = 0; task_id < config_.expert_num * mat_type_all * mat_split; task_id++) {
          int64_t expert_idx = task_id / (mat_type_all * mat_split);
          uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
          uint8_t mat_split_idex = task_id % mat_split;
          if (mat_class == 0) {  // the up matrix
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size, mat_split,
                         mat_split_idex);
          } else if (mat_class == 1) {
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size, mat_split,
                         mat_split_idex);
          } else {
            size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
            size_t scale_size = config_.hidden_size * sizeof(float);
            read_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size, mat_split,
                         mat_split_idex);
          }
        }
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else
#endif
      {
        if (tp_part_idx == 0) {
          std::cout << "  online quant from bf16" << std::endl;
        }
        int nth = T::recommended_nth(config_.intermediate_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth](int task_id) {
              int64_t expert_idx = task_id / nth;
              int ith = task_id % nth;
              // gate part
              gate_bb_[expert_idx]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith,
                  nth, -1, true);
              // up part
              up_bb_[expert_idx]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith,
                  nth, -1, true);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth](int task_id) {
              int64_t expert_idx = task_id / nth;
              int ith = task_id % nth;
              // down part
              down_bb_[expert_idx]->from_mat(
                  (ggml_bf16_t*)config_.down_proj + expert_idx * config_.hidden_size * config_.intermediate_size, ith,
                  nth, -1, true);
              // printf("load down, expert %ld, ith %d, total nth %d\n", expert_idx, ith, nth);
            },
            nullptr);
      }
#ifdef CHECK
      verify_load_right();
#endif
      // save process
      if (config_.save) {
        pool->do_work_stealing_job(
            config_.expert_num * mat_type_all, nullptr,
            [this](int task_id) {
              int64_t expert_idx = task_id / mat_type_all;
              uint8_t mat_class = task_id % mat_type_all;
              if (mat_class == 0) {  // the up matrix
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 1) {
                size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
                size_t scale_size = config_.intermediate_size * sizeof(float);
                write_weights(prefix, "_gate_", (char*)gate_bb_[expert_idx]->b, expert_idx, size, scale_size);
              } else if (mat_class == 2) {
                size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size);
                size_t scale_size = config_.hidden_size * sizeof(float);
                write_weights(prefix, "_down_", (char*)down_bb_[expert_idx]->b, expert_idx, size, scale_size);
              }
            },
            nullptr);
      }
    }
  }

  void warm_up() {
    int qlen = config_.max_len;
    std::vector<uint8_t> input(sizeof(input_t) * qlen * config_.hidden_size);
    std::vector<uint8_t> output(sizeof(output_t) * qlen * config_.hidden_size);
    std::vector<int64_t> expert_ids(qlen * config_.num_experts_per_tok);
    std::vector<float> weights(qlen * config_.num_experts_per_tok);
    for (int i = 0; i < qlen * config_.num_experts_per_tok; i++) {
      expert_ids[i] = i % config_.expert_num;
      weights[i] = 0.01;
    }
    forward(qlen, config_.num_experts_per_tok, expert_ids.data(), weights.data(), input.data(), output.data());
  }

#define DIRECT_OR_POOL_BY_QLEN(var, fn)                          \
  do {                                                           \
    if (qlen < 10) {                                             \
      for (int i = 0; i < (var); i++) {                          \
        (fn)(i);                                                 \
      }                                                          \
    } else {                                                     \
      pool->do_work_stealing_job((var), nullptr, (fn), nullptr); \
    }                                                            \
  } while (0)
  static float act_fn(float x) { return x / (1.0f + expf(-x)); }

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen <= 1) {
      // printf("  KML_MOE_TP decode qlen %d, k %d, expert_ids %p, weights %p, input %p, output %p\n", qlen, k,
      //  (void *)expert_ids, (void *)weights, (void *)input, (void *)output);
      forward_decode(qlen, k, expert_ids, weights, input, output);
    } else {
      // printf("  KML_MOE_TP prefill qlen %d, k %d, expert_ids %p, weights %p, input %p, output %p\n", qlen, k,
      //  (void *)expert_ids, (void *)weights, (void *)input, (void *)output);
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    }
  }

  void forward_decode(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                      void* output) {
    void (*cblas_gemm_s8s8s32)(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                               const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                               const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                               const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                               const int32_t* oc);
    int devide_elements_size = 1;
    if constexpr (std::is_same_v<typename T::dt, int4_2_t>) {
      // 使用 lambda 包装器
      cblas_gemm_s8s8s32 = [](const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                              const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                              const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                              const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                              const int32_t* oc) {
        decode_int4_cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c,
                                       ldc, oc);
      };
      devide_elements_size = 2;  // int4_2_t 是 2 个 int8_t
    } else {
      // 使用 lambda 包装器
      cblas_gemm_s8s8s32 = [](const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                              const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                              const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                              const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                              const int32_t* oc) {
        decode_cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc,
                                  oc);
      };
    }

#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif
    int max_local_num = 0;  // 记录最大的 local num

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // 这个是指的总的激活专家的数量(共 256 个专家 for V3),k 是指每个
    // token 激活的专家数量，V3 是 8
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        // #ifdef FORWARD_TIME_PROFILE
        // #endif
        max_local_num = std::max(max_local_num, m_local_num_[i]);

        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode prepare");
#endif

    // 这里是将输入数据拷贝到每个专家的输入 buffer 中
    pool->do_work_stealing_job(
        qlen, nullptr,
        [&](int i) {
          for (int j = 0; j < k; j++) {
            memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
                   (input_t*)input + i * config_.hidden_size, sizeof(input_t) * config_.hidden_size);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode copy_input");
#endif

    // 量化输入
    {
      size_t mth = T::recommended_mth(max_local_num);
      // printf("mth is %lu\n", mth);
      pool->do_work_stealing_job(
          activated_expert * mth, nullptr,
          [&](int task_id) {
            int task_id_expert = task_id / mth;
            int ith = task_id % mth;
            int expert_idx = m_expert_id_map_[task_id_expert];
            if (ith * T::M_BLOCK >= m_local_num_[expert_idx]) {
              return;  // 这个专家的输入量不够，直接跳过
            }
            gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], ith, mth);
          },
          nullptr);
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode quant_input");
#endif

    int nth = T::recommended_nth_up_gate(config_.intermediate_size);
    int32_t oc = 0;
    // 这里是 gate 和 up 的初次计算，后续会有 gate 去经过激活函数 再和 up 做点积得到 gate
    // 的输出（m_local_gate_output_ptr_）
    pool->do_work_stealing_job(
        nth * activated_expert * 2, nullptr,
        [this, qlen, nth, oc, &cblas_gemm_s8s8s32, devide_elements_size](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // int ith = task_id % nth;
          if (do_up) {
            int8_t* gate_up_ba_ptr = (int8_t*)gate_up_ba_[expert_idx]->a;
            int8_t* up_bb_ptr =
                (int8_t*)up_bb_[expert_idx]->b + ith * config_.hidden_size * T::N_BLOCK_UP_GATE / devide_elements_size;
            int32_t* up_bc_ptr = (int32_t*)up_bc_[expert_idx]->c + ith * T::N_BLOCK_UP_GATE;
            cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                               T::N_BLOCK_UP_GATE, config_.hidden_size, 1.0, gate_up_ba_ptr, config_.hidden_size, 0,
                               up_bb_ptr, config_.hidden_size, 0, 0.0, up_bc_ptr, config_.intermediate_size, &oc);

            // 这里只需要反量化到 fp32 就行了
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), up_bb_[expert_idx].get(), up_bc_[expert_idx].get(), ith, nth,
                           T::N_BLOCK_UP_GATE);
          } else {
            int8_t* gate_up_ba_ptr = (int8_t*)gate_up_ba_[expert_idx]->a;
            int8_t* gate_bb_ptr = (int8_t*)gate_bb_[expert_idx]->b +
                                  ith * config_.hidden_size * T::N_BLOCK_UP_GATE / devide_elements_size;
            int32_t* gate_bc_ptr = (int32_t*)gate_bc_[expert_idx]->c + ith * T::N_BLOCK_UP_GATE;
            cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                               T::N_BLOCK_UP_GATE, config_.hidden_size, 1.0, gate_up_ba_ptr, config_.hidden_size, 0,
                               gate_bb_ptr, config_.hidden_size, 0, 0.0, gate_bc_ptr, config_.intermediate_size, &oc);
            // 这里只需要反量化到 fp32 就行了
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), gate_bb_[expert_idx].get(), gate_bc_[expert_idx].get(), ith,
                           nth, T::N_BLOCK_UP_GATE);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode up_gate");
#endif

    // 目前每个专家的 up or gate 的输出是[m_local_num_[expert_idx], config_.intermediate_size]
    nth = T::recommended_nth(config_.intermediate_size);
    auto up_gate_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        float* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        float* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        // TODO: 使用 SVE 来等效替换加速
        for (int j = n_start; j < n_end; j++) {
          // 上面取出 expert_idx 的 专家 对应的 gate 和 up 的输出，下面沿着一列一列进行激活函数计算,(geate 激活和 up
          // 做点积)
          gate_output_ptr[j] = act_fn(gate_output_ptr[j]) * up_output_ptr[j];
        }
      }
    };
    DIRECT_OR_POOL_BY_QLEN(nth * activated_expert, up_gate_fn);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode act");
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx]);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode quant_down_input");
#endif

    nth = T::recommended_nth_down(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [this, qlen, nth, oc, &cblas_gemm_s8s8s32, devide_elements_size](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int8_t* down_ba_ptr = ((int8_t*)down_ba_[expert_idx]->a);
          int8_t* down_bb_ptr = ((int8_t*)down_bb_[expert_idx]->b) +
                                ith * config_.intermediate_size * T::N_BLOCK_DOWN / devide_elements_size;
          int32_t* down_bc_ptr = ((int32_t*)down_bc_[expert_idx]->c) + ith * T::N_BLOCK_DOWN;
          // printf("taskid:%d down_ba_ptr %p, down_bb_ptr %p, down_bc_ptr %p, T::N_BLOCK_DOWN %d, nth %d\n", task_id,
          //  down_ba_ptr, down_bb_ptr, down_bc_ptr, T::N_BLOCK_DOWN, nth);
          // int8_t *down_bc_ptr
          cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                             T::N_BLOCK_DOWN, config_.intermediate_size, 1.0, down_ba_ptr, config_.intermediate_size, 0,
                             down_bb_ptr, config_.intermediate_size, 0, 0.0, down_bc_ptr, config_.hidden_size, &oc);

          // 这里只需要反量化到 fp32 就行了
          T::apply_scale(m_local_num_[expert_idx], config_.hidden_size, m_local_down_output_ptr_[expert_idx],
                         down_ba_[expert_idx].get(), down_bb_[expert_idx].get(), down_bc_[expert_idx].get(), ith, nth,
                         T::N_BLOCK_DOWN);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("decode down");
#endif
    // 对于每个 qlen 的 token 来说，取出对应的 k 个专家的 down_output 和权重进行加权求和
    size_t block_dim = 512;
    size_t block_num = (config_.hidden_size + block_dim - 1) / block_dim;
    pool->do_work_stealing_job(
        qlen * block_num, nullptr,
        [this, nth, output, k, expert_ids, weights, block_dim, block_num](int i) {
          // CHECK: 目前用纯 C++ 来实现
          int q_idx = i / block_num;
          int block_idx = i % block_num;
          int e_start = block_idx * block_dim;
          int e_end =
              ((block_idx + 1) * block_dim) < config_.hidden_size ? ((block_idx + 1) * block_dim) : config_.hidden_size;
          for (int e = e_start; e < e_end; e++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
              sum += weights[q_idx * k + j] * ((float*)m_local_down_output_ptr_[expert_ids[q_idx * k + j]])
                                                  [m_local_pos_[q_idx][j] * config_.hidden_size + e];
            }
            ((float*)output)[q_idx * config_.hidden_size + e] = sum;
          }
        },
        nullptr);

    // pool->do_work_stealing_job(
    //     qlen, nullptr,
    //     [this, nth, output, k, expert_ids, weights](int i) {
    //       // TODO: 使用 SVE 来等效替换加速
    //       // for (int e = 0; e < config_.hidden_size; e += 32) {
    //       //   __m512 x0 = _mm512_setzero_ps();
    //       //   __m512 x1 = _mm512_setzero_ps();
    //       //   for (int j = 0; j < k; j++) {
    //       //     __m512 weight = _mm512_set1_ps(weights[i * k + j]);
    //       //     __m512 down_output0, down_output1;
    //       //     avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
    //       //                                           m_local_pos_[i][j] * config_.hidden_size + e),
    //       //                               &down_output0, &down_output1);
    //       //     x0 = _mm512_fmadd_ps(down_output0, weight, x0);
    //       //     x1 = _mm512_fmadd_ps(down_output1, weight, x1);
    //       //   }
    //       //   auto f32out = (__m512 *)((float *)output + i * config_.hidden_size + e);
    //       //   f32out[0] = x0;
    //       //   f32out[1] = x1;
    //       // }
    //       // CHECK: 目前用纯 C++ 来实现
    //       for (int e = 0; e < config_.hidden_size; e++) {
    //         float sum = 0;
    //         for (int j = 0; j < k; j++) {
    //           sum +=
    //               weights[i * k + j] *
    //               ((float *)
    //                    m_local_down_output_ptr_[expert_ids[i * k + j]])[m_local_pos_[i][j] * config_.hidden_size +
    //                    e];
    //         }
    //         ((float *)output)[i * config_.hidden_size + e] = sum;
    //       }
    //     },
    //     nullptr);
#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "[moe] decode layer " + std::to_string(config_.layer_idx) +
                     " tp_part_idx: " + std::to_string(tp_part_idx) +
                     ", activated expert: " + std::to_string(activated_expert);
    perf_report();

#endif
  }

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    void (*cblas_gemm_s8s8s32)(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                               const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                               const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                               const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                               const int32_t* oc);
    int devide_elements_size = 1;
    if constexpr (std::is_same_v<typename T::dt, int4_2_t>) {
      // 使用 lambda 包装器
      cblas_gemm_s8s8s32 = [](const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                              const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                              const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                              const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                              const int32_t* oc) {
        prefill_int4_cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta,
                                        c, ldc, oc);
      };
      devide_elements_size = 2;  // int4_2_t 是 2 个 int8_t
    } else {
      // 使用 lambda 包装器
      cblas_gemm_s8s8s32 = [](const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                              const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k,
                              const float alpha, const void* a, const size_t lda, const BLASINT8 oa, const void* b,
                              const size_t ldb, const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc,
                              const int32_t* oc) {
        prefill_cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c,
                                   ldc, oc);
      };
    }

#ifdef FORWARD_TIME_PROFILE
    forward_perf_start();
#endif
    int max_local_num = 0;  // 记录最大的 local num

    auto pool = config_.pool->get_subpool(tp_part_idx);

    // 这个是指的总的激活专家的数量(共 256 个专家 for V3),k 是指每个
    // token 激活的专家数量，V3 是 8
    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
      }
    }

    for (int i = 0; i < config_.expert_num; i++) {
      if (m_local_num_[i] > 0) {
        // #ifdef FORWARD_TIME_PROFILE
        // #endif
        max_local_num = std::max(max_local_num, m_local_num_[i]);

        m_expert_id_map_[activated_expert] = i;
        activated_expert++;
      }
    }

    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("prepare");
#endif

    // 这里是将输入数据拷贝到每个专家的输入 buffer 中
    DIRECT_OR_POOL_BY_QLEN(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size,
               (input_t*)input + i * config_.hidden_size, sizeof(input_t) * config_.hidden_size);
      }
    });

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("copy_input");
#endif

    // 量化输入
    {
      size_t mth = T::recommended_mth(max_local_num);
      // printf("mth is %lu\n", mth);
      DIRECT_OR_POOL_BY_QLEN(activated_expert * mth, [&](int task_id) {
        int task_id_expert = task_id / mth;
        int ith = task_id % mth;
        int expert_idx = m_expert_id_map_[task_id_expert];
        if (ith * T::M_BLOCK >= m_local_num_[expert_idx]) {
          return;  // 这个专家的输入量不够，直接跳过
        }
        gate_up_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_input_ptr_[expert_idx], ith, mth);
      });
    }

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("quant_input");
#endif

    int nth = T::recommended_nth_up_gate(config_.intermediate_size);
    int32_t oc = 0;
    // 这里是 gate 和 up 的初次计算，后续会有 gate 去经过激活函数 再和 up 做点积得到 gate
    // 的输出（m_local_gate_output_ptr_）
    pool->do_work_stealing_job(
        nth * activated_expert * 2, nullptr,
        [this, qlen, nth, oc, &cblas_gemm_s8s8s32, devide_elements_size](int task_id2) {
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;

          // int ith = task_id % nth;
          if (do_up) {
            int8_t* gate_up_ba_ptr = (int8_t*)gate_up_ba_[expert_idx]->a;
            int8_t* up_bb_ptr =
                (int8_t*)up_bb_[expert_idx]->b + ith * config_.hidden_size * T::N_BLOCK_UP_GATE / devide_elements_size;
            int32_t* up_bc_ptr = (int32_t*)up_bc_[expert_idx]->c + ith * T::N_BLOCK_UP_GATE;
            cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                               T::N_BLOCK_UP_GATE, config_.hidden_size, 1.0, gate_up_ba_ptr, config_.hidden_size, 0,
                               up_bb_ptr, config_.hidden_size, 0, 0.0, up_bc_ptr, config_.intermediate_size, &oc);
            // 这里只需要反量化到 fp32 就行了
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_up_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), up_bb_[expert_idx].get(), up_bc_[expert_idx].get(), ith, nth,
                           T::N_BLOCK_UP_GATE);
          } else {
            int8_t* gate_up_ba_ptr = (int8_t*)gate_up_ba_[expert_idx]->a;
            int8_t* gate_bb_ptr = (int8_t*)gate_bb_[expert_idx]->b +
                                  ith * config_.hidden_size * T::N_BLOCK_UP_GATE / devide_elements_size;
            int32_t* gate_bc_ptr = (int32_t*)gate_bc_[expert_idx]->c + ith * T::N_BLOCK_UP_GATE;
            cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                               T::N_BLOCK_UP_GATE, config_.hidden_size, 1.0, gate_up_ba_ptr, config_.hidden_size, 0,
                               gate_bb_ptr, config_.hidden_size, 0, 0.0, gate_bc_ptr, config_.intermediate_size, &oc);
            // 这里只需要反量化到 fp32 就行了
            T::apply_scale(m_local_num_[expert_idx], config_.intermediate_size, m_local_gate_output_ptr_[expert_idx],
                           gate_up_ba_[expert_idx].get(), gate_bb_[expert_idx].get(), gate_bc_[expert_idx].get(), ith,
                           nth, T::N_BLOCK_UP_GATE);
          }
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("up_gate");
#endif

    // 目前每个专家的 up or gate 的输出是[m_local_num_[expert_idx], config_.intermediate_size]
    nth = T::recommended_nth(config_.intermediate_size);
    auto up_gate_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        float* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        float* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        // TODO: 使用 SVE 来等效替换加速
        for (int j = n_start; j < n_end; j++) {
          // 上面取出 expert_idx 的 专家 对应的 gate 和 up 的输出，下面沿着一列一列进行激活函数计算,(geate 激活和 up
          // 做点积)
          gate_output_ptr[j] = act_fn(gate_output_ptr[j]) * up_output_ptr[j];
        }
      }
    };
    DIRECT_OR_POOL_BY_QLEN(nth * activated_expert, up_gate_fn);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("act");
#endif

    pool->do_work_stealing_job(
        activated_expert, nullptr,
        [this](int task_id) {
          int expert_idx = m_expert_id_map_[task_id];
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx]);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("quant_down_input");
#endif

    nth = T::recommended_nth_down(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * activated_expert, nullptr,
        [this, qlen, nth, oc, &cblas_gemm_s8s8s32, devide_elements_size](int task_id) {
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          int8_t* down_ba_ptr = ((int8_t*)down_ba_[expert_idx]->a);
          int8_t* down_bb_ptr = ((int8_t*)down_bb_[expert_idx]->b) +
                                ith * config_.intermediate_size * T::N_BLOCK_DOWN / devide_elements_size;
          int32_t* down_bc_ptr = ((int32_t*)down_bc_[expert_idx]->c) + ith * T::N_BLOCK_DOWN;
          // printf("taskid:%d down_ba_ptr %p, down_bb_ptr %p, down_bc_ptr %p, T::N_BLOCK_DOWN %d, nth %d\n", task_id,
          //  down_ba_ptr, down_bb_ptr, down_bc_ptr, T::N_BLOCK_DOWN, nth);
          // int8_t *down_bc_ptr
          cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m_local_num_[expert_idx],
                             T::N_BLOCK_DOWN, config_.intermediate_size, 1.0, down_ba_ptr, config_.intermediate_size, 0,
                             down_bb_ptr, config_.intermediate_size, 0, 0.0, down_bc_ptr, config_.hidden_size, &oc);

          // 这里只需要反量化到 fp32 就行了
          T::apply_scale(m_local_num_[expert_idx], config_.hidden_size, m_local_down_output_ptr_[expert_idx],
                         down_ba_[expert_idx].get(), down_bb_[expert_idx].get(), down_bc_[expert_idx].get(), ith, nth,
                         T::N_BLOCK_DOWN);
        },
        nullptr);

#ifdef FORWARD_TIME_PROFILE
    PROFILE_RECORD_TIME_STAMP("down");
#endif
    // 对于每个 qlen 的 token 来说，取出对应的 k 个专家的 down_output 和权重进行加权求和
    size_t block_dim = 512;
    size_t block_num = (config_.hidden_size + block_dim - 1) / block_dim;
    pool->do_work_stealing_job(
        qlen * block_num, nullptr,
        [this, nth, output, k, expert_ids, weights, block_dim, block_num](int i) {
          // TODO: 使用 SVE 来等效替换加速
          // for (int e = 0; e < config_.hidden_size; e += 32) {
          //   __m512 x0 = _mm512_setzero_ps();
          //   __m512 x1 = _mm512_setzero_ps();
          //   for (int j = 0; j < k; j++) {
          //     __m512 weight = _mm512_set1_ps(weights[i * k + j]);
          //     __m512 down_output0, down_output1;
          //     avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
          //                                           m_local_pos_[i][j] * config_.hidden_size + e),
          //                               &down_output0, &down_output1);
          //     x0 = _mm512_fmadd_ps(down_output0, weight, x0);
          //     x1 = _mm512_fmadd_ps(down_output1, weight, x1);
          //   }
          //   auto f32out = (__m512 *)((float *)output + i * config_.hidden_size + e);
          //   f32out[0] = x0;
          //   f32out[1] = x1;
          // }
          // CHECK: 目前用纯 C++ 来实现
          int q_idx = i / block_num;
          int block_idx = i % block_num;
          int e_start = block_idx * block_dim;
          int e_end =
              ((block_idx + 1) * block_dim) < config_.hidden_size ? ((block_idx + 1) * block_dim) : config_.hidden_size;
          for (int e = e_start; e < e_end; e++) {
            float sum = 0;
            for (int j = 0; j < k; j++) {
              sum += weights[q_idx * k + j] * ((float*)m_local_down_output_ptr_[expert_ids[q_idx * k + j]])
                                                  [m_local_pos_[q_idx][j] * config_.hidden_size + e];
            }
            ((float*)output)[q_idx * config_.hidden_size + e] = sum;
          }
        },
        nullptr);

    // pool->do_work_stealing_job(
    //     qlen, nullptr,
    //     [this, nth, output, k, expert_ids, weights](int i) {
    //       // TODO: 使用 SVE 来等效替换加速
    //       // for (int e = 0; e < config_.hidden_size; e += 32) {
    //       //   __m512 x0 = _mm512_setzero_ps();
    //       //   __m512 x1 = _mm512_setzero_ps();
    //       //   for (int j = 0; j < k; j++) {
    //       //     __m512 weight = _mm512_set1_ps(weights[i * k + j]);
    //       //     __m512 down_output0, down_output1;
    //       //     avx512_32xbf16_to_32xfp32((__m512i *)(m_local_down_output_ptr_[expert_ids[i * k + j]] +
    //       //                                           m_local_pos_[i][j] * config_.hidden_size + e),
    //       //                               &down_output0, &down_output1);
    //       //     x0 = _mm512_fmadd_ps(down_output0, weight, x0);
    //       //     x1 = _mm512_fmadd_ps(down_output1, weight, x1);
    //       //   }
    //       //   auto f32out = (__m512 *)((float *)output + i * config_.hidden_size + e);
    //       //   f32out[0] = x0;
    //       //   f32out[1] = x1;
    //       // }
    //       // CHECK: 目前用纯 C++ 来实现
    //       for (int e = 0; e < config_.hidden_size; e++) {
    //         float sum = 0;
    //         for (int j = 0; j < k; j++) {
    //           sum +=
    //               weights[i * k + j] *
    //               ((float *)
    //                    m_local_down_output_ptr_[expert_ids[i * k + j]])[m_local_pos_[i][j] * config_.hidden_size +
    //                    e];
    //         }
    //         ((float *)output)[i * config_.hidden_size + e] = sum;
    //       }
    //     },
    //     nullptr);
#ifdef FORWARD_TIME_PROFILE
    time_perf_name = "[moe] layer " + std::to_string(config_.layer_idx) +
                     " tp_part_idx: " + std::to_string(tp_part_idx) +
                     ", activated expert: " + std::to_string(activated_expert);
    perf_report();

#endif
  }
};

template <typename K>
class TP_MOE<KML_MOE_TP<K>> : public TP_MOE_Common<KML_MOE_TP<K>> {
 public:
  using TP_MOE_Common<KML_MOE_TP<K>>::TP_MOE_Common;

  void load_weights() {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });
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
                size_t expert_id = expert_id_;
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

      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file\n");
      pool->dispense_backend()->do_numa_job([this, pool](int numa_id) { this->tps[numa_id]->load_weights(); });
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

  void merge_results(int qlen, void* output) {
    // #ifdef FORWARD_TIME_PROFILE
    //     forward_perf_start();
    // #endif
    auto pool = this->config.pool;
    auto merge_fn = [this, output](int token_nth) {
      auto& local_output_numa = this->local_output_numa;
      auto& tp_configs = this->tp_configs;
      auto& tp_count = this->tp_count;
      auto& config = this->config;
      float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;

      for (int i = 1; i < tp_count; i++) {
        float* merge_from = local_output_numa[i] + token_nth * tp_configs[i].hidden_size;
        // TODO: 后续用 SVE 来加速
        // for (int e = 0; e < tp_configs[i].hidden_size; e += 16) {
        //   *((__m512 *)(merge_to + e)) = _mm512_add_ps(*((__m512 *)(merge_to + e)), *((__m512 *)(merge_from + e)));
        // }
        // CHECK: 目前用普通的纯 C++ 来实现
        for (int e = 0; e < tp_configs[i].hidden_size; e++) {
          merge_to[e] += merge_from[e];
        }
      }

      convert_or_copy((ggml_bf16_t*)output + token_nth * config.hidden_size, merge_to, config.hidden_size);

      // for (int e = 0; e < config.hidden_size; e += 32) {
      // TODO: 这里需要用 SVE 来加速，实现 fp32 到 bf16 的转换
      // __m512 x0 = *(__m512 *)(merge_to + e);
      // __m512 x1 = *(__m512 *)(merge_to + e + 16);
      // avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i *)((ggml_bf16_t *)output + token_nth * config.hidden_size + e));

      // CHECK: 目前用普通的纯 C++ 来实现 fp32 到 bf16 的转换

      // convert_32fp32_to_32bf16_pure_c(merge_to + e,
      // (uint16_t *)((ggml_bf16_t *)output + token_nth * config.hidden_size + e));

      // }
    };
    DIRECT_OR_POOL_BY_QLEN(qlen, merge_fn);
    // #ifdef FORWARD_TIME_PROFILE
    //     PROFILE_RECORD_TIME_STAMP("moe merge done");
    // #endif
    // #ifdef FORWARD_TIME_PROFILE
    //     time_perf_name = "[moe merge] decode layer " + std::to_string(this->config.layer_idx);
    //     perf_report();
    // #endif
  }
};

#endif