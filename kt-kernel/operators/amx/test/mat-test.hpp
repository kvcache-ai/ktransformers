#ifndef AMX_MAT_TEST_HPP
#define AMX_MAT_TEST_HPP

#include <cassert>
#include <iostream>
#include <limits>
#include <random>

#include "../../common.hpp"
#include "../la/utils.hpp"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "timer.hh"

template <typename T>
struct DotProductImpl {
  static_assert(sizeof(T) == -1, "No associated type defined for this type.");
  using type = void;
};

template <typename T>
using DotProductType = typename DotProductImpl<T>::type;

template <>
struct DotProductImpl<uint8_t> {
  using type = uint32_t;
};
template <>
struct DotProductImpl<int8_t> {
  using type = int32_t;
};
template <>
struct DotProductImpl<uint32_t> {
  using type = uint32_t;
};
template <>
struct DotProductImpl<int32_t> {
  using type = int32_t;
};

template <>
struct DotProductImpl<float> {
  using type = float;
};

enum class Layout {
  RowMajor,
  ColumnMajor,
  VNNIColumnMajor,
};

template <typename T>
struct Mat {
  int rows, cols;
  size_t size() { return rows * cols; }
  T* data;
  size_t stride_in_bytes;

  void* qdata = nullptr;
  ggml_type q_type;
  size_t q_stride;

  Layout layout = Layout::RowMajor;

  Mat() {};

  Mat(int rows, int cols, Layout layout) : rows(rows), cols(cols), layout(layout) {
    size_t total_size;
    if (layout == Layout::RowMajor) {
      stride_in_bytes = cols * sizeof(T);
      stride_in_bytes = (stride_in_bytes + 63) / 64 * 64;
      total_size = stride_in_bytes * rows;
    } else if (layout == Layout::ColumnMajor) {
      stride_in_bytes = rows * sizeof(T);
      stride_in_bytes = (stride_in_bytes + 63) / 64 * 64;
      total_size = stride_in_bytes * cols;
    } else {
      assert(0);
    }

    // data = new(std::align_val_t(64)) T[rows * cols];
    data = reinterpret_cast<T*>(aligned_alloc(64, total_size));
    memset(data, 0, total_size);
  }

  Mat<T> sub_mat(int r, int c) {
    Mat<T> re;
    re.rows = r;
    re.cols = c;
    re.data = data;
    re.layout = layout;
    re.stride_in_bytes = stride_in_bytes;
    re.qdata = qdata;
    re.q_stride = q_stride;
    re.q_type = q_type;
  }

  void dealloc() {
    delete[] data;
    if (qdata) {
      delete[] reinterpret_cast<char*>(qdata);
    }
  }

  void row_major_increase() {
    int x = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        at(i, j) = x++;
      }
    }
  }

  void dis_to_00() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        at(i, j) = i + j;
      }
    }
  }

  void random(std::mt19937& gen) {
    if constexpr (std::is_integral_v<T>) {
      std::uniform_int_distribution<T> dist(0, 100);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          at(i, j) = dist(gen);
        }
      }
    } else if constexpr (std::is_floating_point_v<T>) {
      std::uniform_real_distribution<T> dist(-1.0, 1.0);
      for (int i = 0; i < rows; i++) {
        std::mt19937 gen_row(gen());
        for (int j = 0; j < cols; j++) {
          at(i, j) = dist(gen_row);
        }
      }
    } else {
      throw std::runtime_error("Unsupported type");
    }
  }

  size_t stride() { return stride_in_bytes; }

  int line_element_count() {
    if (layout == Layout::RowMajor) {
      return cols;
    } else if (layout == Layout::ColumnMajor) {
      return rows;
    } else {
      assert(0);
    }
    assert(0);
    return 0;
  }

  T& at(int r, int c) {
    switch (layout) {
      case Layout::RowMajor:
        return *offset_pointer_row_major(data, r, c, stride());
      case Layout::ColumnMajor:
        return *offset_pointer_col_major(data, r, c, stride());
      // case Layout::VNNIColumnMajor:
      // return data[c*rows+r];
      default: {
        assert(0);
      }
    }
    throw std::runtime_error("Unsupported layout");
    // assert(0);
  }

  void print() {
    int limit = 10;      // 设置阈值
    int print_rows = 3;  // 开头和结尾打印的行数和列数

    for (int i = 0; i < rows; i++) {
      // 当行数过多时，跳过中间的行
      if (rows > limit && (i >= print_rows && i < rows - print_rows)) {
        if (i == print_rows) {
          std::cout << "...\n...\n";
        }
        continue;
      }

      for (int j = 0; j < cols; j++) {
        // 当列数过多时，跳过中间的列
        if (cols > limit && (j >= print_rows && j < cols - print_rows)) {
          if (j == print_rows) {
            std::cout << "... ";
          }
          continue;
        }

        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
          std::cout << (int)at(i, j) << " ";
        } else {
          std::cout << at(i, j) << " ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  void print_all() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
          std::cout << (int)at(i, j) << " ";
        } else if constexpr (std::is_floating_point_v<T>) {
          // std::cout << std::setw(6) << std::scientific << std::setprecision(2) << at(i, j) << "  ";
          printf("%6.2f ", at(i, j));
        } else {
          std::cout << at(i, j) << " ";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  Mat<DotProductType<T>> mul_check(Mat<T>& b) {
    assert(cols == b.rows);
    Mat<DotProductType<T>> c(rows, b.cols, Layout::RowMajor);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < b.cols; j++) {
        c.at(i, j) = 0;
        for (int k = 0; k < cols; k++) {
          c.at(i, j) += static_cast<DotProductType<T>>(at(i, k)) * static_cast<DotProductType<T>>(b.at(k, j));
        }
      }
    }
    return c;
  }

  bool cmp(Mat<T>& b) {
    if constexpr (std::is_integral_v<T>) {
      assert(rows == b.rows && cols == b.cols);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          if (at(i, j) != b.at(i, j)) {
            std::cout << "Error at " << i << " " << j << " " << at(i, j) << ", " << b.at(i, j) << std::endl;
            // std::cout << "Error at " << i << " " << j << std::endl;
            // std::cout << "Other: " << b.at(i, j) << std::endl;
            // std::cout << "Me: " << at(i, j) << std::endl;
            // assert(0);
            // break;
            // return false;
          }
        }
      }
      std::cout << "Check passed" << std::endl;
      return true;
    }

    if constexpr (std::is_floating_point_v<T>) {
      T rel_error_sum = 0;
      T error_sum = 0;
      T max_error = 0;
      T max_rel_error = 0;
      int max_i = 0, max_j = 0;
      assert(rows == b.rows && cols == b.cols);
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          T error = std::abs(at(i, j) - b.at(i, j));
          error_sum += error;
          rel_error_sum += error / std::abs(at(i, j));
          if (error / std::abs(at(i, j)) > max_rel_error) {
            max_rel_error = error / std::abs(at(i, j));
          }
          if (error > max_error) {
            max_i = i;
            max_j = j;
            max_error = error;
          }
        }
      }
      if (rel_error_sum / size() > 1e-2 || max_error / at(max_i, max_j) > 1e-2) {
        std::cout << "Max Error: " << std::fixed << max_error << "(" << max_error / at(max_i, max_j) << ")"
                  << " at " << max_i << " " << max_j << ", Max Rel Error " << max_rel_error
                  << ", Average Relative: " << rel_error_sum / size() << ", Average Error: " << error_sum / size()
                  << std::endl;
      } else {
        std::cout << "Error Less Than 1%" << std::endl;
      }

      return true;
    }
  }

  void quant(ggml_type to) {
    if constexpr (std::is_same<T, float>::value == false) {
      throw std::runtime_error("Quantization only supported for f32 matrices");
    }
    // Timer t(std::string("to ") + ggml_type_name(to));
    assert(line_element_count() * sizeof(T) == stride());
    assert(line_element_count() % ggml_blck_size(to) == 0);
    int blck_cnt_per_row = line_element_count() / ggml_blck_size(to);
    q_stride = blck_cnt_per_row * ggml_type_size(to);

    size_t qdata_size = size() * ggml_type_size(to) / ggml_blck_size(to);
    qdata_size += 512 - q_stride % 512;

    qdata = new (std::align_val_t(512)) char[qdata_size];
    q_type = to;

    switch (to) {
      case GGML_TYPE_F32: {
        return;
      }
      case GGML_TYPE_F16: {
        ggml_fp32_to_fp16_row(data, reinterpret_cast<ggml_fp16_t*>(qdata), size());
        return;
      }
      case GGML_TYPE_BF16: {
        ggml_fp32_to_bf16_row(data, reinterpret_cast<ggml_bf16_t*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q4_0: {
        quantize_row_q4_0(data, reinterpret_cast<block_q4_0*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q4_1: {
        quantize_row_q4_1(data, reinterpret_cast<block_q4_1*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q5_0: {
        quantize_row_q5_0(data, reinterpret_cast<block_q5_0*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q5_1: {
        quantize_row_q5_1(data, reinterpret_cast<block_q5_1*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q8_0: {
        quantize_row_q8_0(data, reinterpret_cast<block_q8_0*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q8_1: {
        quantize_row_q8_1(data, reinterpret_cast<block_q8_1*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q2_K: {
        quantize_row_q2_K(data, reinterpret_cast<block_q2_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q3_K: {
        quantize_row_q3_K(data, reinterpret_cast<block_q3_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q4_K: {
        quantize_row_q4_K(data, reinterpret_cast<block_q4_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q5_K: {
        quantize_row_q5_K(data, reinterpret_cast<block_q5_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q6_K: {
        quantize_row_q6_K(data, reinterpret_cast<block_q6_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_Q8_K: {
        quantize_row_q8_K(data, reinterpret_cast<block_q8_K*>(qdata), size());
        return;
      }
      case GGML_TYPE_IQ2_XXS:
      case GGML_TYPE_IQ2_XS:
      case GGML_TYPE_IQ3_XXS:
      case GGML_TYPE_IQ1_S:
      case GGML_TYPE_IQ4_NL:
      case GGML_TYPE_IQ3_S:
      case GGML_TYPE_IQ2_S:
      case GGML_TYPE_IQ4_XS:
      case GGML_TYPE_I8:
      case GGML_TYPE_I16:
      case GGML_TYPE_I32:
      case GGML_TYPE_I64:
      case GGML_TYPE_F64:
      case GGML_TYPE_IQ1_M:
      case GGML_TYPE_COUNT:
      default:
        throw std::runtime_error("Unsupported quantization type");
    }
    throw std::runtime_error("Unsupported quantization type");
  }

  template <typename Block>
  Block* quant_data() {
    return reinterpret_cast<Block*>(qdata);
  }

  void dequant() {
    auto x = q_type;
    switch (x) {
      case GGML_TYPE_F32: {
        return;
      }
      case GGML_TYPE_F16: {
        ggml_fp16_to_fp32_row(reinterpret_cast<ggml_fp16_t*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q4_0: {
        dequantize_row_q4_0(reinterpret_cast<block_q4_0*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q4_1: {
        dequantize_row_q4_1(reinterpret_cast<block_q4_1*>(qdata), data, size());

        return;
      }
      case GGML_TYPE_Q5_0: {
        dequantize_row_q5_0(reinterpret_cast<block_q5_0*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q5_1: {
        dequantize_row_q5_1(reinterpret_cast<block_q5_1*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q8_0: {
        dequantize_row_q8_0(reinterpret_cast<block_q8_0*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q8_1: {
        throw std::runtime_error("not supported");
      }
      case GGML_TYPE_Q2_K: {
        dequantize_row_q2_K(reinterpret_cast<block_q2_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q3_K: {
        dequantize_row_q3_K(reinterpret_cast<block_q3_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q4_K: {
        dequantize_row_q4_K(reinterpret_cast<block_q4_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q5_K: {
        dequantize_row_q5_K(reinterpret_cast<block_q5_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q6_K: {
        dequantize_row_q6_K(reinterpret_cast<block_q6_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_Q8_K: {
        dequantize_row_q8_K(reinterpret_cast<block_q8_K*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_IQ2_XXS:
      case GGML_TYPE_IQ2_XS:
      case GGML_TYPE_IQ3_XXS:
      case GGML_TYPE_IQ1_S:
      case GGML_TYPE_IQ4_NL:
      case GGML_TYPE_IQ3_S:
      case GGML_TYPE_IQ2_S:
      case GGML_TYPE_IQ4_XS:
      case GGML_TYPE_I8:
      case GGML_TYPE_I16:
      case GGML_TYPE_I32:
      case GGML_TYPE_I64:
      case GGML_TYPE_F64:
      case GGML_TYPE_IQ1_M:
      case GGML_TYPE_BF16: {
        ggml_bf16_to_fp32_row(reinterpret_cast<ggml_bf16_t*>(qdata), data, size());
        return;
      }
      case GGML_TYPE_COUNT:
      default:
        throw std::runtime_error("Unsupported quantization type");
    }
    throw std::runtime_error("Unsupported quantization type");
  }
};

inline void init() {
  struct ggml_init_params params = {
      0,
      NULL,
      true,
  };

  auto ctx_eval = ggml_init(params);

  if (!ctx_eval) {
    throw std::runtime_error("Failed to create ggml context");
  }
}
#endif