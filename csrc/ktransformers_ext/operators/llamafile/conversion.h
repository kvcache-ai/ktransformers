/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:34:55
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CONVERSION_H
#define CPUINFER_CONVERSION_H

#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include <memory.h>

inline void to_float(const void *input, float *output, int size, ggml_type type) {
  if (type == ggml_type::GGML_TYPE_F32) {
    memcpy(output, input, size * sizeof(float));
  } else {
    if (type == ggml_type::GGML_TYPE_Q8_K) {
      dequantize_row_q8_K((block_q8_K *)input, output, size);
    } else {
      ggml_internal_get_type_traits(type).to_float(input, output, size);
    }
  }
}

inline void from_float(const float *input, void *output, int size, ggml_type type) {
  if (type == ggml_type::GGML_TYPE_F32) {
    memcpy(output, input, size * sizeof(float));
  } else {
    ggml_internal_get_type_traits(type).from_float(input, output, size);
  }
}

#endif