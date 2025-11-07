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

#include <cstddef>
#include <cstdint>
#include <cstring>
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../cpu_backend/shared_mem_buffer.h"
#include "../../cpu_backend/worker_pool.h"
#include "../moe-tp.hpp"
#include "la/amx.hpp"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

template <class T>
class AMX_MOE_TP {
 private:
  int tp_part_idx;
  std::filesystem::path prefix;

  void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if
                     // quantized)]
  void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if
                     // quantized)]

  ggml_bf16_t* m_local_input_;        // [num_experts_per_tok * max_len * hidden_size]
  ggml_bf16_t* m_local_gate_output_;  // [num_experts_per_tok * max_len * intermediate_size]
  ggml_bf16_t* m_local_up_output_;    // [num_experts_per_tok * max_len * intermediate_size]
  ggml_bf16_t* m_local_down_output_;  // [num_experts_per_tok * max_len * hidden_size]

  std::vector<std::vector<int>> m_local_pos_;          // [max_len, num_experts_per_tok]
  std::vector<int> m_local_num_;                       // [expert_num]
  std::vector<int> m_expert_id_map_;                   // [expert_num]
  std::vector<ggml_bf16_t*> m_local_input_ptr_;        // [expert_num]
  std::vector<ggml_bf16_t*> m_local_gate_output_ptr_;  // [expert_num]
  std::vector<ggml_bf16_t*> m_local_up_output_ptr_;    // [expert_num]
  std::vector<ggml_bf16_t*> m_local_down_output_ptr_;  // [expert_num]

  std::vector<std::shared_ptr<typename T::BufferA>> gate_up_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> gate_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> gate_bc_;
  std::vector<std::shared_ptr<typename T::BufferB>> up_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> up_bc_;
  std::vector<std::shared_ptr<typename T::BufferA>> down_ba_;
  std::vector<std::shared_ptr<typename T::BufferB>> down_bb_;
  std::vector<std::shared_ptr<typename T::BufferC>> down_bc_;
#ifdef CHECK
  char verify_bb[100000000];
  char check_bb[100000000];
  uint8_t compare_expers = 3;
#endif

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
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_MOE_TP(GeneralMOEConfig config, int tp_part_idx) {
    printf("Creating AMX_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
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
    mem_requests.append_pointer(
        &m_local_input_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok * config_.max_len * config_.hidden_size);
    mem_requests.append_pointer(&m_local_gate_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_up_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                         config_.max_len * config_.intermediate_size);
    mem_requests.append_pointer(&m_local_down_output_, sizeof(ggml_bf16_t) * config_.num_experts_per_tok *
                                                           config_.max_len * config_.hidden_size);

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

    for (size_t i = 0; i < config_.expert_num; i++) {
      gate_up_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, nullptr));
      gate_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, nullptr));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      gate_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, gate_bb_ptr));

      void* up_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size));
      up_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size));
      down_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size, down_bb_ptr));
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

  ~AMX_MOE_TP() {
    // shared_mem_buffer_numa.dealloc(this);
  }
  // pack and quant the weights
  void pack_weights() {}
  void load_weights() {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (config_.gate_projs.size()) {
      pool->do_work_stealing_job(
          config_.expert_num, nullptr,
          [this, physical_to_logical_map](int expert_id) {
            // printf("Load layer %d [%d/%d]\n", config_.layer_idx, expert_id, config_.expert_num);
            uint64_t logical_expert_id = expert_id;
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
      if (config_.load) {
        std::cout << "Loading from " << prefix << std::endl;
        for (int task_id = 0; task_id < config_.expert_num * mat_type_all * mat_split; task_id++) {
          int64_t expert_idx = task_id / (mat_type_all * mat_split);
          uint64_t logical_expert_id = expert_idx;
          uint8_t mat_class = (task_id % (mat_type_all * mat_split)) / mat_split;
          uint8_t mat_split_idex = task_id % mat_split;
          if (mat_class == 0) {  // the up matrix
            size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size);
            size_t scale_size = config_.intermediate_size * sizeof(float);
            read_weights(prefix, "_up_", (char*)up_bb_[expert_idx]->b, logical_expert_id, size, scale_size, mat_split,
                         mat_split_idex);
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
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth](int task_id) {
              int64_t expert_idx = task_id / nth;
              int ith = task_id % nth;
              // gate part
              gate_bb_[expert_idx]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith,
                  nth);
              // up part
              up_bb_[expert_idx]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + expert_idx * config_.intermediate_size * config_.hidden_size, ith,
                  nth);
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
                  nth);
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

  void forward(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input, void* output) {
    if (qlen > 1) {
      forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
      forward_decode(k, expert_ids, weights, input, output);
    }
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

#define MATMUL_OR_VECMUL_BY_QLEN(...)                                  \
  do {                                                                 \
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) { \
      amx::mat_mul(__VA_ARGS__);                                       \
    } else {                                                           \
      amx::vec_mul(__VA_ARGS__);                                       \
    }                                                                  \
  } while (0)

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
#ifdef FORWARD_TIME_PROFILE
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last = start_time;
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
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

    // activated_expert 已经统计完成

    size_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
      m_local_input_ptr_[i] = m_local_input_ + offset * config_.hidden_size;
      m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
      m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
      m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
      offset += m_local_num_[i];
    }
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif

    DIRECT_OR_POOL_BY_QLEN(qlen, [&](int i) {
      for (int j = 0; j < k; j++) {
        if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
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

    DIRECT_OR_POOL_BY_QLEN(activated_expert, [this](int task_id) {
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
          if (do_up) {
            MATMUL_OR_VECMUL_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                     gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            MATMUL_OR_VECMUL_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                     gate_up_ba_[expert_idx], gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth);
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

    auto up_gate_fn = [this, nth](int task_id) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < m_local_num_[expert_idx]; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0);
          __m512 result1 = amx::act_fn(gate_val1, up_val1);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
        }
      }
    };
    DIRECT_OR_POOL_BY_QLEN(nth * activated_expert, up_gate_fn);

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
          down_ba_[expert_idx]->from_mat(m_local_num_[expert_idx], m_local_gate_output_ptr_[expert_idx], 0, 1);
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
          MATMUL_OR_VECMUL_BY_QLEN(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                                   down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
          down_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_down_output_ptr_[expert_idx], ith, nth);
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
        [this, nth, output, k, expert_ids, weights](int i) {
          for (int e = 0; e < config_.hidden_size; e += 32) {
            __m512 x0 = _mm512_setzero_ps();
            __m512 x1 = _mm512_setzero_ps();
            for (int j = 0; j < k; j++) {
              if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
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
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
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
    // 用于保存各阶段耗时（单位：微秒）
    long prepare_time = 0, cpy_input_time = 0, q_input_time = 0, up_gate_time = 0;
    long act_time = 0, q_down_time = 0, down_time = 0, weight_time = 0;
    int max_local_num = 0;  // 记录最大的 local num
#endif

    int activated_expert = 0;
    for (int i = 0; i < k; i++) {
      if (expert_ids[i] < config_.num_gpu_experts || expert_ids[i] >= config_.expert_num) {
        continue;
      }
      m_expert_id_map_[activated_expert] = expert_ids[i];
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

    gate_up_ba_[0]->from_mat(qlen, (ggml_bf16_t*)input, 0, 1);

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
          if (do_up) {
            amx::vec_mul(qlen, config_.intermediate_size, config_.hidden_size, gate_up_ba_[0], up_bb_[expert_idx],
                         up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            amx::vec_mul(qlen, config_.intermediate_size, config_.hidden_size, gate_up_ba_[0], gate_bb_[expert_idx],
                         gate_bc_[expert_idx], ith, nth);
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

    for (int task_id = 0; task_id < nth * activated_expert; task_id++) {
      int expert_idx = m_expert_id_map_[task_id / nth];
      int ith = task_id % nth;
      auto [n_start, n_end] = T::split_range_n(config_.intermediate_size, ith, nth);
      for (int i = 0; i < qlen; i++) {
        ggml_bf16_t* gate_output_ptr = &m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size];
        ggml_bf16_t* up_output_ptr = &m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size];
        for (int j = n_start; j < n_end; j += 32) {
          __m512 gate_val0, gate_val1, up_val0, up_val1;
          avx512_32xbf16_to_32xfp32((__m512i*)(gate_output_ptr + j), &gate_val0, &gate_val1);
          avx512_32xbf16_to_32xfp32((__m512i*)(up_output_ptr + j), &up_val0, &up_val1);
          __m512 result0 = amx::act_fn(gate_val0, up_val0);
          __m512 result1 = amx::act_fn(gate_val1, up_val1);
          avx512_32xfp32_to_32xbf16(&result0, &result1, (__m512i*)(gate_output_ptr + j));
        }
      }
    }

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
          down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
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
          amx::vec_mul(qlen, config_.hidden_size, config_.intermediate_size, down_ba_[expert_idx], down_bb_[expert_idx],
                       down_bc_[expert_idx], ith, nth);
          down_bc_[expert_idx]->to_mat(qlen, m_local_down_output_ptr_[expert_idx], ith, nth);
        },
        nullptr);
#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      down_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
#endif
    for (int i = 0; i < qlen; i++) {
      for (int e = 0; e < config_.hidden_size; e += 32) {
        __m512 x0 = _mm512_setzero_ps();
        __m512 x1 = _mm512_setzero_ps();
        for (int j = 0; j < k; j++) {
          if (expert_ids[i * k + j] < config_.num_gpu_experts || expert_ids[i * k + j] >= config_.expert_num) {
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
    }

#ifdef FORWARD_TIME_PROFILE
    {
      auto now_time = std::chrono::high_resolution_clock::now();
      weight_time = std::chrono::duration_cast<std::chrono::microseconds>(now_time - last).count();
      last = now_time;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto forward_total_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 在函数末尾一次性打印所有阶段的耗时，并附带 max_local_num 和 qlen
    printf(
        "Profiling Results (numa[%d]) decode: activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }
};

template <typename K>
class TP_MOE<AMX_MOE_TP<K>> : public TP_MOE_Common<AMX_MOE_TP<K>> {
 public:
  using TP_MOE_Common<AMX_MOE_TP<K>>::TP_MOE_Common;
  void load_weights() {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;
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

  void merge_results(int qlen, void* output, bool incremental) {
    auto pool = this->config.pool;
    auto merge_fn = [this, output, incremental](int token_nth) {
      auto& local_output_numa = this->local_output_numa;
      auto& tp_configs = this->tp_configs;
      auto& tp_count = this->tp_count;
      auto& config = this->config;
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
      for (int e = 0; e < config.hidden_size; e += 32) {
        __m512 x0 = *(__m512*)(merge_to + e);
        __m512 x1 = *(__m512*)(merge_to + e + 16);
        avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
      }
    };
    DIRECT_OR_POOL_BY_QLEN(qlen, merge_fn);
  }
  void merge_results(int qlen, void* output) { merge_results(qlen, output, false); }
};

#endif
