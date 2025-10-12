/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_AMX_AWQ_MOE_H
#define CPUINFER_OPERATOR_AMX_AWQ_MOE_H

// #define CHECK

#include <cstddef>
#include <cstdint>
#include <cstring>
// #define FORWARD_TIME_PROFILE
// #define FORWARD_TIME_REPORT

#include <immintrin.h>

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

#define expert_map(m, x) (m != nullptr ? m[(x)] : (x))

template <class T>
class AMX_AWQ_MOE_TP {
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
    std::ofstream of(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                               std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (of.is_open() == false) {
      printf("Failed to open weights file for writing\n");
      return;
    }
    of.write((char*)bb, size - scale_size);
    of.close();

    of.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                      "_scale_" + ".kt"));
    if (of.is_open() == false) {
      printf("Failed to open scales file for writing\n");
      return;
    }
    of.write(((char*)bb) + size - scale_size, scale_size);
    of.close();
  }

  // Enhanced version that writes all data including mins for complete comparison
  inline void write_weights(std::filesystem::path prefix, std::string mat_class, typename T::BufferB* buffer,
                            int expert_idx, const std::string& quantization_type = "") {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;

    // Calculate dimensions based on matrix type
    int rows, cols, num_groups;
    size_t scale_elem_count;
    std::string matrix_type = mat_class.substr(1, mat_class.length() - 2);  // Remove leading/trailing underscore
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    }

    size_t weight_size = (rows * cols) / 2;  // INT4 packed
    size_t scale_size = scale_elem_count * sizeof(float);

    // Create filename prefix
    std::string filename_base = T::name() + mat_class + std::to_string(expert_idx);
    if (!quantization_type.empty()) {
      filename_base += "_" + quantization_type;
    }

    // Write quantized weights
    std::ofstream of(prefix / (filename_base + "_" + std::to_string(weight_size) + "Byte_quant.kt"));
    if (of.is_open()) {
      of.write((char*)buffer->b, weight_size);
      of.close();
    }

    // Write scales
    of.open(prefix / (filename_base + "_" + std::to_string(scale_size) + "Byte_scale.kt"));
    if (of.is_open()) {
      of.write((char*)buffer->d, scale_size);
      of.close();
    }

    // Write mins if available
    if (quant_config.zero_point && buffer->mins) {
      of.open(prefix / (filename_base + "_" + std::to_string(scale_size) + "Byte_mins.kt"));
      if (of.is_open()) {
        of.write((char*)buffer->mins, scale_size);
        of.close();
      }
    }
  }

  inline void read_weights(std::filesystem::path prefix, std::string mat_class, char* bb, int expert_idx, size_t size,
                           size_t scale_size, uint8_t mat_split, uint8_t mat_split_idex) {
    std::ifstream f(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" +
                              std::to_string(size - scale_size) + "Byte" + "_quant_" + ".kt"));
    if (f.is_open() == false) {
      printf("Failed to open quantized weights file for reading\n");
      return;
    }
    f.seekg(mat_split_idex * (size - scale_size) / mat_split);
    f.read(((char*)bb) + mat_split_idex * (size - scale_size) / mat_split, (size - scale_size) / mat_split);
    f.close();

    f.open(prefix / (T::name() + mat_class + std::to_string(expert_idx) + "_" + std::to_string(scale_size) + "Byte" +
                     "_scale_" + ".kt"));
    if (f.is_open() == false) {
      printf("Failed to open scales file for reading\n");
      return;
    }
    f.seekg(mat_split_idex * scale_size / mat_split);
    f.read((((char*)bb) + size - scale_size) + mat_split_idex * scale_size / mat_split, scale_size / mat_split);
    f.close();
  }

  // Enhanced version that reads all data including mins
  inline bool read_weights(std::filesystem::path prefix, std::string mat_class, typename T::BufferB* buffer,
                           int expert_idx, const std::string& quantization_type = "") {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;

    // Calculate dimensions based on matrix type
    int rows, cols, num_groups;
    size_t scale_elem_count;
    std::string matrix_type = mat_class.substr(1, mat_class.length() - 2);  // Remove leading/trailing underscore
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    }

    size_t weight_size = (rows * cols) / 2;  // INT4 packed
    size_t scale_size = scale_elem_count * sizeof(float);

    // Create filename prefix
    std::string filename_base = T::name() + mat_class + std::to_string(expert_idx);
    if (!quantization_type.empty()) {
      filename_base += "_" + quantization_type;
    }

    // Read quantized weights
    std::ifstream f(prefix / (filename_base + "_" + std::to_string(weight_size) + "Byte_quant.kt"));
    if (!f.is_open()) {
      return false;
    }
    f.read((char*)buffer->b, weight_size);
    f.close();

    // Read scales
    f.open(prefix / (filename_base + "_" + std::to_string(scale_size) + "Byte_scale.kt"));
    if (!f.is_open()) {
      return false;
    }
    f.read((char*)buffer->d, scale_size);
    f.close();

    // Read mins if available and buffer supports it
    if (quant_config.zero_point && buffer->mins) {
      f.open(prefix / (filename_base + "_" + std::to_string(scale_size) + "Byte_mins.kt"));
      if (f.is_open()) {
        f.read((char*)buffer->mins, scale_size);
        f.close();
      }
    }

    return true;
  }

  // AWQ-specific function to read quantized weights, scales and zeros from files
  inline void read_awq_weights(std::filesystem::path prefix, std::string proj_name, int expert_idx, char* weights_buf,
                               float* scales_buf, uint8_t* zeros_buf, size_t weights_size, size_t scales_size,
                               size_t zeros_size, uint8_t mat_split, uint8_t mat_split_idx) {
    // Read qweights (quantized weights)
    std::string weights_filename = proj_name + ".qweight." + std::to_string(expert_idx) + ".bin";
    std::ifstream weights_file(prefix / weights_filename, std::ios::binary);
    if (!weights_file.is_open()) {
      printf("Failed to open weights file: %s\n", (prefix / weights_filename).c_str());
      throw std::runtime_error("Failed to open weights file: " + weights_filename);
    }

    weights_file.seekg(mat_split_idx * weights_size / mat_split);
    weights_file.read(weights_buf + mat_split_idx * weights_size / mat_split, weights_size / mat_split);
    weights_file.close();

    // Read scales
    std::string scales_filename = proj_name + ".scales." + std::to_string(expert_idx) + ".bin";
    std::ifstream scales_file(prefix / scales_filename, std::ios::binary);
    if (!scales_file.is_open()) {
      printf("Failed to open scales file: %s\n", (prefix / scales_filename).c_str());
      throw std::runtime_error("Failed to open scales file: " + scales_filename);
    }

    scales_file.seekg(mat_split_idx * scales_size / mat_split);
    scales_file.read(reinterpret_cast<char*>(scales_buf) + mat_split_idx * scales_size / mat_split,
                     scales_size / mat_split);
    scales_file.close();

    // Read qzeros (quantized zeros)
    std::string zeros_filename = proj_name + ".qzeros." + std::to_string(expert_idx) + ".bin";
    std::ifstream zeros_file(prefix / zeros_filename, std::ios::binary);
    if (!zeros_file.is_open()) {
      printf("Failed to open zeros file: %s\n", (prefix / zeros_filename).c_str());
      throw std::runtime_error("Failed to open zeros file: " + zeros_filename);
    }

    zeros_file.seekg(mat_split_idx * zeros_size / mat_split);
    zeros_file.read(reinterpret_cast<char*>(zeros_buf) + mat_split_idx * zeros_size / mat_split,
                    zeros_size / mat_split);
    zeros_file.close();
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

  // Function to dump Buffer B data for debugging quantization results
  inline void dump_buffer_b(const std::string& quantization_type, int expert_idx, const std::string& matrix_type,
                            typename T::BufferB* buffer) {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;

    printf("[DUMP_BUFFER_B] TP%d %s Expert%d %s:\n", tp_part_idx, quantization_type.c_str(), expert_idx,
           matrix_type.c_str());

    // Calculate dimensions based on matrix type
    int rows, cols, num_groups;
    size_t scale_elem_count;
    if (matrix_type == "gate" || matrix_type == "up") {
      rows = config_.intermediate_size;
      cols = config_.hidden_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    } else {  // down
      rows = config_.hidden_size;
      cols = config_.intermediate_size;
      num_groups = cols / group_size;
      scale_elem_count = num_groups * rows;
    }

    // Dump scales (as float)
    printf("  Scales[first 16]: ");
    for (int i = 0; i < std::min(16, (int)scale_elem_count); i++) {
      printf("%.6f ", buffer->d[i]);
    }
    printf("\n");

    if (scale_elem_count > 16) {
      printf("  Scales[last 16]: ");
      int start_idx = std::max(0, (int)scale_elem_count - 16);
      for (int i = start_idx; i < (int)scale_elem_count; i++) {
        printf("%.6f ", buffer->d[i]);
      }
      printf("\n");
    }

    // Dump mins (as float) if available
    if (quant_config.zero_point && buffer->mins) {
      printf("  Mins[first 16]: ");
      for (int i = 0; i < std::min(16, (int)scale_elem_count); i++) {
        printf("%.6f ", buffer->mins[i]);
      }
      printf("\n");

      if (scale_elem_count > 16) {
        printf("  Mins[last 16]: ");
        int start_idx = std::max(0, (int)scale_elem_count - 16);
        for (int i = start_idx; i < (int)scale_elem_count; i++) {
          printf("%.6f ", buffer->mins[i]);
        }
        printf("\n");
      }
    }

    // Dump quantized weights (as hex uint8)
    size_t weight_size = (rows * cols) / 2;  // INT4 packed
    uint8_t* weight_ptr = (uint8_t*)buffer->b;

    printf("  Weights[first 32 bytes]: ");
    for (int i = 0; i < std::min(32, (int)weight_size); i++) {
      printf("%02x ", weight_ptr[i]);
    }
    printf("\n");

    if (weight_size > 32) {
      printf("  Weights[last 32 bytes]: ");
      int start_idx = std::max(32, (int)weight_size - 32);
      for (int i = start_idx; i < (int)weight_size; i++) {
        printf("%02x ", weight_ptr[i]);
      }
      printf("\n");
    }

    printf("  Matrix dimensions: %dx%d, Groups: %d, Group size: %d, Scale elements: %zu\n", rows, cols, num_groups,
           group_size, scale_elem_count);
    printf("\n");
  }

  // AVX-optimized function to convert INT4 zeros to float mins
  // mins = zeros * scales (element-wise), where scales is float format
  inline void convert_zeros_to_mins_avx(const uint32_t* zeros_int4_packed, const float* scales, float* mins,
                                        size_t num_elements) {
    constexpr size_t simd_width = 8;  // 每次解 8 个 int4

    for (size_t i = 0; i < num_elements; i += simd_width) {
      uint32_t packed_vals = zeros_int4_packed[i / 8];

      for (int j = 0; j < 8; j++) {
        int v = packed_vals & 0xF;  // 取出4bit
        mins[i + j] = -(scales[i + j] * v);
        packed_vals = packed_vals >> 4;
      }
    }
  }

#ifdef FORWARD_TIME_REPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> last_now;
#endif

 public:
  using input_t = ggml_bf16_t;
  using output_t = float;
  GeneralMOEConfig config_;
  static constexpr double ELEMENT_SIZE = T::ELEMENT_SIZE;

  AMX_AWQ_MOE_TP(GeneralMOEConfig config, int tp_part_idx) {
    auto& quant_config = config.quant_config;
    int& group_size = quant_config.group_size;
    if (quant_config.group_size == 0 || !quant_config.zero_point) {
      throw std::runtime_error("AWQ-Quantization AMX MoE only support KGroup Int4_1");
    }
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
      gate_up_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.hidden_size, group_size, nullptr));
      gate_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      up_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.intermediate_size, nullptr));
      down_ba_.push_back(
          std::make_shared<typename T::BufferA>(config_.max_len, config_.intermediate_size, group_size, nullptr));
      down_bc_.push_back(std::make_shared<typename T::BufferC>(config_.max_len, config_.hidden_size, nullptr));

      void* gate_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, group_size));
      gate_bb_.push_back(std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size,
                                                               group_size, gate_bb_ptr));

      void* up_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, group_size));
      up_bb_.push_back(
          std::make_shared<typename T::BufferB>(config_.intermediate_size, config_.hidden_size, group_size, up_bb_ptr));

      void* down_bb_ptr =
          std::aligned_alloc(64, T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, group_size));
      down_bb_.push_back(std::make_shared<typename T::BufferB>(config_.hidden_size, config_.intermediate_size,
                                                               group_size, down_bb_ptr));
    }
    for (int i = 0; i < config_.expert_num; i++) {
      mem_requests.append_function([this, i](void* new_ptr) { gate_up_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.hidden_size, group_size));
      mem_requests.append_function([this, i](void* new_ptr) { gate_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { up_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.intermediate_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_ba_[i]->set_data(new_ptr); },
                                   T::BufferA::required_size(config_.max_len, config_.intermediate_size, group_size));
      mem_requests.append_function([this, i](void* new_ptr) { down_bc_[i]->set_data(new_ptr); },
                                   T::BufferC::required_size(config_.max_len, config_.hidden_size));
    }
    shared_mem_buffer_numa.alloc(tp_part_idx, this, mem_requests);
  }

  ~AMX_AWQ_MOE_TP() {
    // shared_mem_buffer_numa.dealloc(this);
  }

  void load_weights(const uint64_t* physical_to_logical_map) {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
    if (quant_config.group_size == 0 || !quant_config.zero_point) {
      throw std::runtime_error("AWQ-Quantization AMX MoE only support KGroup Int4_1");
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_projs.size()) {
      throw std::runtime_error("AMX load weights is not support");
      // pool->do_work_stealing_job(
      //     config_.expert_num, nullptr,
      //     [this, physical_to_logical_map](int expert_id) {
      //       // printf("Load layer %d [%d/%d]\n", config_.layer_idx, expert_id, config_.expert_num);
      //       uint64_t logical_expert_id = physical_to_logical_map[expert_id];
      //       auto& quant_config = config_.quant_config;
      //       int& group_size = quant_config.group_size;
      //       {
      //         int num_group = config_.hidden_size / group_size;
      //         size_t scale_size = num_group * config_.intermediate_size * sizeof(float);
      //         size_t size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size, group_size) -
      //         (scale_size << 1);

      //         memcpy(gate_bb_[expert_id]->b, config_.gate_projs[tp_part_idx][logical_expert_id], size);

      //         if constexpr (T::BufferB::SCALE) {
      //           memcpy(gate_bb_[expert_id]->d, config_.gate_scales[tp_part_idx][logical_expert_id], scale_size);
      //         }

      //         memcpy(up_bb_[expert_id]->b, config_.up_projs[tp_part_idx][logical_expert_id], size);

      //         if constexpr (T::BufferB::SCALE) {
      //           memcpy(up_bb_[expert_id]->d, config_.up_scales[tp_part_idx][logical_expert_id], scale_size);
      //         }

      //         if (quant_config.zero_point) {
      //           // Convert INT4 zeros to float mins using AVX optimization
      //           size_t num_elements = num_group * config_.intermediate_size;
      //           convert_zeros_to_mins_avx(
      //               (const uint8_t*)config_.gate_zeros[tp_part_idx][logical_expert_id],
      //               (const float*)config_.gate_scales[tp_part_idx][logical_expert_id],
      //               gate_bb_[expert_id]->mins,
      //               num_elements
      //           );
      //           convert_zeros_to_mins_avx(
      //               (const uint8_t*)config_.up_zeros[tp_part_idx][logical_expert_id],
      //               (const float*)config_.up_scales[tp_part_idx][logical_expert_id],
      //               up_bb_[expert_id]->mins,
      //               num_elements
      //           );
      //         }
      //       }

      //       {
      //         int num_group = config_.intermediate_size / group_size;
      //         size_t scale_size = num_group * config_.hidden_size * sizeof(float);
      //         size_t size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, group_size) -
      //         (scale_size << 1);

      //         memcpy(down_bb_[expert_id]->b, config_.down_projs[tp_part_idx][logical_expert_id], size);

      //         if constexpr (T::BufferB::SCALE) {
      //           memcpy(down_bb_[expert_id]->d, config_.down_scales[tp_part_idx][logical_expert_id], scale_size);
      //         }

      //         if (quant_config.zero_point) {
      //           // Convert INT4 zeros to float mins using AVX optimization
      //           size_t num_elements = num_group * config_.hidden_size;
      //           convert_zeros_to_mins_avx(
      //               (const uint8_t*)config_.down_zeros[tp_part_idx][logical_expert_id],
      //               (const float*)config_.down_scales[tp_part_idx][logical_expert_id],
      //               down_bb_[expert_id]->mins,
      //               num_elements
      //           );
      //         }
      //       }
      //     },
      //     nullptr);

    } else {
      // AWQ Load from file implementation
      int nth = T::recommended_nth(config_.intermediate_size);
      static uint8_t mat_type_all = 3, mat_split = 1;
      if (config_.load) {
        throw std::runtime_error("AMX load weights from file is not support");
        //   std::cout << "Loading AWQ weights from " << prefix << std::endl;

        //   // Use work stealing job for parallel loading
        //   pool->do_work_stealing_job(
        //       config_.expert_num * mat_type_all, nullptr,
        //       [this, physical_to_logical_map, mat_split](int task_id) {
        //         auto& quant_config = config_.quant_config;
        //         int& group_size = quant_config.group_size;

        //         int64_t expert_idx = task_id / mat_type_all;
        //         uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
        //         uint8_t mat_class = task_id % mat_type_all;

        //         if (mat_class == 0) { // gate projection
        //           int num_group = config_.hidden_size / group_size;
        //           size_t weights_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size,
        //           group_size) - (2 * num_group * config_.intermediate_size * sizeof(float)); size_t scales_size =
        //           num_group * config_.intermediate_size * sizeof(float); size_t zeros_size = num_group *
        //           config_.intermediate_size / 2; // INT4 packed format

        //           // Allocate temporary buffer for zeros
        //           std::vector<uint8_t> zeros_buf(zeros_size);

        //           read_awq_weights(prefix, "gate_proj", logical_expert_id,
        //                          (char*)gate_bb_[expert_idx]->b,
        //                          (float*)gate_bb_[expert_idx]->d,
        //                          zeros_buf.data(),
        //                          weights_size, scales_size, zeros_size,
        //                          mat_split, 0);

        //           // Convert INT4 zeros to float mins
        //           if (quant_config.zero_point) {
        //             convert_zeros_to_mins_avx(zeros_buf.data(),
        //                                     (float*)gate_bb_[expert_idx]->d,
        //                                     gate_bb_[expert_idx]->mins,
        //                                     num_group * config_.intermediate_size);
        //           }

        //         } else if (mat_class == 1) { // up projection
        //           int num_group = config_.hidden_size / group_size;
        //           size_t weights_size = T::BufferB::required_size(config_.intermediate_size, config_.hidden_size,
        //           group_size) - (2 * num_group * config_.intermediate_size * sizeof(float)); size_t scales_size =
        //           num_group * config_.intermediate_size * sizeof(float); size_t zeros_size = num_group *
        //           config_.intermediate_size / 2; // INT4 packed format

        //           // Allocate temporary buffer for zeros
        //           std::vector<uint8_t> zeros_buf(zeros_size);

        //           read_awq_weights(prefix, "up_proj", logical_expert_id,
        //                          (char*)up_bb_[expert_idx]->b,
        //                          (float*)up_bb_[expert_idx]->d,
        //                          zeros_buf.data(),
        //                          weights_size, scales_size, zeros_size,
        //                          mat_split, 0);

        //           // Convert INT4 zeros to float mins
        //           if (quant_config.zero_point) {
        //             convert_zeros_to_mins_avx(zeros_buf.data(),
        //                                     (float*)up_bb_[expert_idx]->d,
        //                                     up_bb_[expert_idx]->mins,
        //                                     num_group * config_.intermediate_size);
        //           }

        //         } else { // down projection
        //           int num_group = config_.intermediate_size / group_size;
        //           size_t weights_size = T::BufferB::required_size(config_.hidden_size, config_.intermediate_size,
        //           group_size) - (2 * num_group * config_.hidden_size * sizeof(float)); size_t scales_size = num_group
        //           * config_.hidden_size * sizeof(float); size_t zeros_size = num_group * config_.hidden_size / 2; //
        //           INT4 packed format

        //           // Allocate temporary buffer for zeros
        //           std::vector<uint8_t> zeros_buf(zeros_size);

        //           read_awq_weights(prefix, "down_proj", logical_expert_id,
        //                          (char*)down_bb_[expert_idx]->b,
        //                          (float*)down_bb_[expert_idx]->d,
        //                          zeros_buf.data(),
        //                          weights_size, scales_size, zeros_size,
        //                          mat_split, 0);

        //           // Convert INT4 zeros to float mins
        //           if (quant_config.zero_point) {
        //             convert_zeros_to_mins_avx(zeros_buf.data(),
        //                                     (float*)down_bb_[expert_idx]->d,
        //                                     down_bb_[expert_idx]->mins,
        //                                     num_group * config_.hidden_size);
        //           }
        //         }
        //       },
        //       nullptr);
      }
// check process, store down matrix to check
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else if (config_.gate_scale != nullptr)
#endif
      {
        // Loading quantized weights
        // Loading quantized weights
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              uint64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
              int ith = task_id % nth;
              // gate part
              gate_bb_[expert_idx]->from_raw_mat(
                  (uint8_t*)config_.gate_proj +
                      ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
                  ith, nth);
              // up part
              up_bb_[expert_idx]->from_raw_mat(
                  (uint8_t*)config_.up_proj +
                      ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
                  ith, nth);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              uint64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
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
              uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
              size_t scale_elem_count =
                  (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;

              // convert scales from FP16 to FP32
              convert_or_copy(gate_bb_[expert_idx]->d,
                              (ggml_fp16_t*)config_.gate_scale + (logical_expert_id * scale_elem_count),
                              scale_elem_count);
              convert_or_copy(up_bb_[expert_idx]->d,
                              (ggml_fp16_t*)config_.up_scale + (logical_expert_id * scale_elem_count),
                              scale_elem_count);
              convert_or_copy(down_bb_[expert_idx]->d,
                              (ggml_fp16_t*)config_.down_scale + (logical_expert_id * scale_elem_count),
                              scale_elem_count);

              // Convert INT4 zeros to FP32 mins
              convert_zeros_to_mins_avx(
                  (const uint32_t*)((uint8_t*)config_.gate_zero + ((logical_expert_id * scale_elem_count) >> 1)),
                  gate_bb_[expert_idx]->d, gate_bb_[expert_idx]->mins, scale_elem_count);
              convert_zeros_to_mins_avx(
                  (const uint32_t*)((uint8_t*)config_.up_zero + ((logical_expert_id * scale_elem_count) >> 1)),
                  up_bb_[expert_idx]->d, up_bb_[expert_idx]->mins, scale_elem_count);
              convert_zeros_to_mins_avx(
                  (const uint32_t*)((uint8_t*)config_.down_zero + ((logical_expert_id * scale_elem_count) >> 1)),
                  down_bb_[expert_idx]->d, down_bb_[expert_idx]->mins, scale_elem_count);
            },
            nullptr);

        // Save offline quantization data if requested
        if (config_.save) {
          for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
            write_weights(prefix, "_gate_", gate_bb_[expert_idx].get(), expert_idx, "OFFLINE");
            write_weights(prefix, "_up_", up_bb_[expert_idx].get(), expert_idx, "OFFLINE");
            write_weights(prefix, "_down_", down_bb_[expert_idx].get(), expert_idx, "OFFLINE");
          }
        }
      }
      else {
        // Online Quantization
        assert(config_.gate_proj != nullptr);

        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
              int ith = task_id % nth;
              // gate part
              gate_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.gate_proj +
                      (logical_expert_id * config_.intermediate_size * config_.hidden_size),
                  ith, nth);
              // up part
              up_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.up_proj + (logical_expert_id * config_.intermediate_size * config_.hidden_size),
                  ith, nth);
            },
            nullptr);

        nth = T::recommended_nth(config_.hidden_size);
        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = physical_to_logical_map[expert_idx];
              int ith = task_id % nth;
              // down part
              down_bb_[logical_expert_id]->from_mat(
                  (ggml_bf16_t*)config_.down_proj +
                      (logical_expert_id * config_.hidden_size * config_.intermediate_size),
                  ith, nth);
            },
            nullptr);

        // Save online quantization data if requested
        if (config_.save) {
          for (int expert_idx = 0; expert_idx < config_.expert_num; expert_idx++) {
            write_weights(prefix, "_gate_", gate_bb_[expert_idx].get(), expert_idx, "ONLINE");
            write_weights(prefix, "_up_", up_bb_[expert_idx].get(), expert_idx, "ONLINE");
            write_weights(prefix, "_down_", down_bb_[expert_idx].get(), expert_idx, "ONLINE");
          }
        }
      }
#ifdef CHECK
      verify_load_right();
#endif
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

#define MATMUL_OR_VECMUL_KGROUP_BY_QLEN(...)                           \
  do {                                                                 \
    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) { \
      amx::mat_mul_kgroup(__VA_ARGS__);                                \
    } else {                                                           \
      amx::vec_mul_kgroup(__VA_ARGS__);                                \
    }                                                                  \
  } while (0)

  void forward_prefill(int qlen, int k, const int64_t* expert_ids, const float* weights, const void* input,
                       void* output) {
    auto pool = config_.pool->get_subpool(tp_part_idx);
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
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
          int& group_size = config_.quant_config.group_size;
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          if (do_up) {
            MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                            group_size, gate_up_ba_[expert_idx], up_bb_[expert_idx], up_bc_[expert_idx],
                                            ith, nth);
            up_bc_[expert_idx]->to_mat(m_local_num_[expert_idx], m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.intermediate_size, config_.hidden_size,
                                            group_size, gate_up_ba_[expert_idx], gate_bb_[expert_idx],
                                            gate_bc_[expert_idx], ith, nth);
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
          int& group_size = config_.quant_config.group_size;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          MATMUL_OR_VECMUL_KGROUP_BY_QLEN(m_local_num_[expert_idx], config_.hidden_size, config_.intermediate_size,
                                          group_size, down_ba_[expert_idx], down_bb_[expert_idx], down_bc_[expert_idx],
                                          ith, nth);
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
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
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
          int& group_size = config_.quant_config.group_size;
          int task_id = task_id2 / 2;
          bool do_up = task_id2 % 2;
          int expert_idx = m_expert_id_map_[task_id / nth];

          int ith = task_id % nth;
          if (do_up) {
            amx::vec_mul_kgroup(qlen, config_.intermediate_size, config_.hidden_size, group_size, gate_up_ba_[0],
                                up_bb_[expert_idx], up_bc_[expert_idx], ith, nth);
            up_bc_[expert_idx]->to_mat(qlen, m_local_up_output_ptr_[expert_idx], ith, nth);
          } else {
            amx::vec_mul_kgroup(qlen, config_.intermediate_size, config_.hidden_size, group_size, gate_up_ba_[0],
                                gate_bb_[expert_idx], gate_bc_[expert_idx], ith, nth);
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
          int& group_size = config_.quant_config.group_size;
          int expert_idx = m_expert_id_map_[task_id / nth];
          int ith = task_id % nth;
          amx::vec_mul_kgroup(qlen, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                              down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
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
        "Profiling Results (numa[%d]): activated_expert: %d, q_input: %ld us, "
        "up_gate: %ld us, act: %ld us, q_down: %ld us, down: %ld us, weight: %ld us, total: %ld us\n",
        tp_part_idx, activated_expert, q_input_time, up_gate_time, act_time, q_down_time, down_time, weight_time,
        forward_total_time);
#endif
  }
};

template <typename K>
class TP_MOE<AMX_AWQ_MOE_TP<K>> : public TP_MOE_Common<AMX_AWQ_MOE_TP<K>> {
 public:
  using TP_MOE_Common<AMX_AWQ_MOE_TP<K>>::TP_MOE_Common;
  void load_weights(const uint64_t* physical_to_logical_map) {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    if (config.gate_projs.empty() == false) {
      printf("TP Load from loader\n");
      pool->dispense_backend()->do_numa_job([this, pool, physical_to_logical_map](int numa_id) {
        this->tps[numa_id]->load_weights(physical_to_logical_map);
      });
      this->weights_loaded = true;
    } else if (config.gate_scale != nullptr) {
      printf("From Packed Int4 with KGroup Scale and Zeros\n");
      int& group_size = config.quant_config.group_size;
      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
        tpc.gate_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.up_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];
        tpc.down_proj = new uint8_t[(tpc.expert_num * weight_elem_count) / 2];

        size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

        tpc.gate_scale = new ggml_fp16_t[(tpc.expert_num * scales_elem_count)];
        tpc.up_scale = new ggml_fp16_t[(tpc.expert_num * scales_elem_count)];
        tpc.down_scale = new ggml_fp16_t[(tpc.expert_num * scales_elem_count)];

        tpc.gate_zero = new uint8_t[(tpc.expert_num * scales_elem_count) / 2];
        tpc.up_zero = new uint8_t[(tpc.expert_num * scales_elem_count) / 2];
        tpc.down_zero = new uint8_t[(tpc.expert_num * scales_elem_count) / 2];
        if (tps[i]->config_.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&](int expert_id_) {
                size_t expert_id = expert_id_;

                // weight TP-slicing
                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                // zeros TP-slicing
                memcpy((ggml_fp16_t*)tpc.down_scale + (expert_id * scales_elem_count),
                       (ggml_fp16_t*)config.down_scale +
                           (expert_id * (config.intermediate_size / group_size) * config.hidden_size +
                            i * scales_elem_count),
                       sizeof(ggml_fp16_t) * scales_elem_count);

                memcpy((uint8_t*)tpc.down_zero + ((expert_id * scales_elem_count) >> 1),
                       (uint8_t*)config.down_zero +
                           ((expert_id * (config.intermediate_size / group_size) * config.hidden_size +
                             i * scales_elem_count) >>
                            1),
                       (sizeof(uint8_t) * scales_elem_count) >> 1);

                for (size_t kg = 0; kg < config.hidden_size / group_size; kg++) {
                  // copy scale
                  memcpy((ggml_fp16_t*)tpc.gate_scale + (expert_id * scales_elem_count) + kg * tpc.intermediate_size,
                         (ggml_fp16_t*)config.gate_scale +
                             (expert_id * ((config.hidden_size / group_size) * config.intermediate_size) +
                              kg * config.intermediate_size + i * tpc.intermediate_size),
                         (sizeof(ggml_fp16_t) * tpc.intermediate_size));

                  memcpy((ggml_fp16_t*)tpc.up_scale + (expert_id * scales_elem_count) + kg * tpc.intermediate_size,
                         (ggml_fp16_t*)config.up_scale +
                             (expert_id * ((config.hidden_size / group_size) * config.intermediate_size) +
                              kg * config.intermediate_size + i * tpc.intermediate_size),
                         (sizeof(ggml_fp16_t) * tpc.intermediate_size));

                  // zeros TP-slicing
                  memcpy(
                      (uint8_t*)tpc.gate_zero + (((expert_id * scales_elem_count) + kg * tpc.intermediate_size) >> 1),
                      (uint8_t*)config.gate_zero +
                          ((expert_id * ((config.hidden_size / group_size) * config.intermediate_size) +
                            kg * config.intermediate_size + i * tpc.intermediate_size) >>
                           1),
                      ((sizeof(uint8_t) * tpc.intermediate_size) >> 1));

                  memcpy((uint8_t*)tpc.up_zero + (((expert_id * scales_elem_count) + kg * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.up_zero +
                             ((expert_id * ((config.hidden_size / group_size) * config.intermediate_size) +
                               kg * config.intermediate_size + i * tpc.intermediate_size) >>
                              1),
                         ((sizeof(uint8_t) * tpc.intermediate_size) >> 1));
                }

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (sizeof(uint8_t) * tpc.intermediate_size) >> 1);
                }
              },
              nullptr);
        }
      }

      pool->dispense_backend()->do_numa_job([this, pool, physical_to_logical_map](int numa_id) {
        this->tps[numa_id]->load_weights(physical_to_logical_map);
      });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (uint8_t*)(tpc.gate_proj);
        delete[] (uint8_t*)(tpc.up_proj);
        delete[] (uint8_t*)(tpc.down_proj);

        delete[] (ggml_fp16_t*)(tpc.gate_scale);
        delete[] (ggml_fp16_t*)(tpc.up_scale);
        delete[] (ggml_fp16_t*)(tpc.down_scale);

        delete[] (uint8_t*)(tpc.gate_zero);
        delete[] (uint8_t*)(tpc.up_zero);
        delete[] (uint8_t*)(tpc.down_zero);
      }

      this->weights_loaded = true;
    } else if (config.gate_proj != nullptr) {
      printf("From BF16 Online Quantization.\n");
      fflush(stdout);
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
                size_t expert_id = physical_to_logical_map[expert_id_];
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

      pool->dispense_backend()->do_numa_job([this, pool, physical_to_logical_map](int numa_id) {
        this->tps[numa_id]->load_weights(physical_to_logical_map);
      });

      for (auto i = 0; i < tp_count; i++) {
        auto& tpc = tps[i]->config_;
        delete[] (ggml_bf16_t*)(tpc.gate_proj);
        delete[] (ggml_bf16_t*)(tpc.up_proj);
        delete[] (ggml_bf16_t*)(tpc.down_proj);
      }

      this->weights_loaded = true;
    } else if (config.path != "") {
      printf("TP Load from file\n");
      pool->dispense_backend()->do_numa_job([this, pool, physical_to_logical_map](int numa_id) {
        this->tps[numa_id]->load_weights(physical_to_logical_map);
      });
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
