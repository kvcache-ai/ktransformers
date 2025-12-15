/**
 * @Description  : AWQ Int4 AMX MoE operator with KGroup quantization and zero-point support
 * @Author       : chenht2022, oql
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 2.0.0
 * @LastEditors  : oql
 * @LastEditTime : 2025-12-10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * This file implements AWQ Int4 MoE using CRTP pattern, inheriting from moe_base.hpp.
 * AWQ weights are stored with group-wise scales and zero-points (KGroup Int4 with zeros).
 **/
#ifndef CPUINFER_OPERATOR_AMX_AWQ_MOE_H
#define CPUINFER_OPERATOR_AMX_AWQ_MOE_H

// #define CHECK

#include "moe_base.hpp"

/**
 * @brief AWQ Int4 MoE operator using CRTP pattern
 * @tparam T Kernel type for AWQ quantization
 *
 * This class provides AWQ-specific implementations:
 * - do_gate_up_gemm: Int4 weight with KGroup scale + zeros + AMX GEMM
 * - do_down_gemm: Same Int4 KGroup GEMM
 * - load_weights: Load Int4 weights with group-wise scales and zero-points
 */
template <class T>
class AMX_AWQ_MOE_TP : public AMX_MOE_BASE<T, AMX_AWQ_MOE_TP<T>> {
 private:
  using Base = AMX_MOE_BASE<T, AMX_AWQ_MOE_TP<T>>;
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

  std::filesystem::path prefix;

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
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, config_.quant_config.group_size));
  }

  void verify_load_right() {
    memcpy(verify_bb, (char*)down_bb_[compare_expers]->b,
           T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, config_.quant_config.group_size));
    if (memcmp(verify_bb, check_bb,
               T::BufferB::required_size(config_.hidden_size, config_.intermediate_size,
                                         config_.quant_config.group_size)) != 0) {
      printf("verify error\n");
      for (size_t i = 0; i < T::BufferB::required_size(config_.hidden_size, config_.intermediate_size,
                                                        config_.quant_config.group_size);
           ++i) {
        if (verify_bb[i] != check_bb[i]) {
          printf("Difference at byte %zu: verify_bb_%d[%zu] = %02x, check_bb[%zu] = %02x\n", i, compare_expers, i,
                 (unsigned char)verify_bb[i], i, (unsigned char)check_bb[i]);
          break;
        }
      }
      assert(0);
    } else {
      printf("pass verify\n");
      printf("numa %d, verify_bb_%d:\n", tp_part_idx, compare_expers);
      size_t size =
          T::BufferB::required_size(config_.hidden_size, config_.intermediate_size, config_.quant_config.group_size);
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
  // mins = -(zeros * scales) (element-wise), where scales is float format
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

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_AWQ_MOE_TP() = default;

  AMX_AWQ_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {
    auto& quant_config = config_.quant_config;
    if (quant_config.group_size == 0 || !quant_config.zero_point) {
      throw std::runtime_error("AWQ-Quantization AMX MoE only support KGroup Int4_1");
    }

    printf("Creating AMX_AWQ_MOE_TP %d at numa %d\n", tp_part_idx_, numa_node_of_cpu(sched_getcpu()));

    auto& load = config_.load;
    auto& save = config_.save;

    prefix = config_.path;
    prefix = prefix / ("_layer_" + std::to_string(config_.layer_idx)) / ("_numa_" + std::to_string(tp_part_idx_));
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
  }

  ~AMX_AWQ_MOE_TP() = default;

  // ============================================================================
  // CRTP buffer creation - with group_size (AWQ uses zero-point)
  // ============================================================================

  size_t buffer_a_required_size_impl(size_t m, size_t k) const {
    return T::BufferA::required_size(m, k, config_.quant_config.group_size);
  }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const {
    return T::BufferC::required_size(m, n);
  }

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
  // CRTP virtual points - GEMM dispatch (uses kgroup with zeros)
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

    if (qlen > 4 * config_.expert_num / config_.num_experts_per_tok) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  /**
   * @brief Load Int4 weights with scales and zero-points
   *
   * AWQ weights include:
   * - Quantized INT4 weights
   * - FP16 scales (converted to FP32)
   * - INT4 zeros (converted to FP32 mins = -scale * zero)
   */
  void load_weights() {
    auto& quant_config = config_.quant_config;
    int& group_size = quant_config.group_size;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    if (quant_config.group_size == 0 || !quant_config.zero_point) {
      throw std::runtime_error("AWQ-Quantization AMX MoE only support KGroup Int4_1");
    }

    auto pool = config_.pool->get_subpool(tp_part_idx);
    if (config_.gate_projs.size()) {
      throw std::runtime_error("AMX load weights from gate_projs is not supported");
    } else {
      int nth = T::recommended_nth(config_.intermediate_size);
      if (config_.load) {
        throw std::runtime_error("AMX load weights from file is not supported");
      }
#ifdef CHECK
      load_check();
#endif
#ifndef CHECK
      else if (config_.gate_scale != nullptr)
#endif
      {
        // Loading quantized weights with scales and zeros
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

              // Convert INT4 zeros to FP32 mins: mins = -(scale * zero)
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
        // Online Quantization from BF16
        assert(config_.gate_proj != nullptr);

        pool->do_work_stealing_job(
            nth * config_.expert_num, nullptr,
            [this, nth, physical_to_logical_map](int task_id) {
              int64_t expert_idx = task_id / nth;
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
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
              uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
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

  // forward, forward_prefill, forward_decode, warm_up are inherited from Base
};

// ============================================================================
// TP_MOE specialization for AMX_AWQ_MOE_TP
// Inherits from TP_MOE<AMX_MOE_BASE<...>> to reuse merge_results implementation
// ============================================================================

template <typename K>
class TP_MOE<AMX_AWQ_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_AWQ_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_AWQ_MOE_TP<K>>>;
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
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                // weight TP-slicing
                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       ((sizeof(uint8_t) * weight_elem_count) >> 1));

                // down scales and zeros TP-slicing
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
                  // copy gate/up scales
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

                  // copy gate/up zeros TP-slicing
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

                // down weights TP-slicing (column-wise)
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

      DO_TPS_LOAD_WEIGHTS(pool);

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
    } else if (config.path != "") {
      printf("TP Load from file\n");
      DO_TPS_LOAD_WEIGHTS(pool);
      this->weights_loaded = true;
    } else {
      throw std::runtime_error("no weight source");
    }
  }

  // merge_results is inherited from TP_MOE<AMX_MOE_BASE<K, AMX_AWQ_MOE_TP<K>>>
};

#endif
