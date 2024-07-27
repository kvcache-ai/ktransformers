/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:35:00
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_LINEAR_H
#define CPUINFER_OPERATOR_LINEAR_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"

struct LinearConfig {
    int input_size;
    int output_size;
    int stride;
    void* proj;
    ggml_type proj_type;
    ggml_type hidden_type;

    LinearConfig() {}

    LinearConfig(int input_size, int output_size, int stride, void* proj, ggml_type proj_type, ggml_type hidden_type)
        : input_size(input_size), output_size(output_size), stride(stride), proj(proj), proj_type(proj_type), hidden_type(hidden_type) {}
};

class Linear {
   public:
    Linear(LinearConfig);
    void warm_up(Backend* backend);
    void forward(const void* input, void* output, Backend* backend);

   private:
    LinearConfig config_;
    void* proj_;  // [output_size * input_size ( /32 if quantized)]

    std::vector<float> input_fp32_;    // [input_size]
    std::vector<uint8_t> proj_input_;  // [input_size * 4]
    std::vector<float> proj_output_;   // [output_size]
};

#endif