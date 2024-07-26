/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:35:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_MLP_H
#define CPUINFER_OPERATOR_MLP_H

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

struct MLPConfig {
    int hidden_size;
    int intermediate_size;
    int stride;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

    MLPConfig() {}

    MLPConfig(int hidden_size, int intermediate_size, int stride, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MLP {
   public:
    MLP(MLPConfig);
    void warm_up(Backend* backend);
    void forward(const void* input, void* output, Backend* backend);

   private:
    MLPConfig config_;
    void* gate_proj_;  // [intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [hidden_size * intermediate_size ( /32 if quantized)]

    std::vector<float> input_fp32_;         // [hidden_size]
    std::vector<uint8_t> gate_input_;       // [hidden_size * 4]
    std::vector<uint8_t> up_input_;         // [hidden_size * 4]
    std::vector<float> gate_output_;        // [intermediate_size]
    std::vector<float> up_output_;          // [intermediate_size]
    std::vector<float> intermediate_fp32_;  // [intermediate_size]
    std::vector<uint8_t> down_input_;       // [intermediate_size * 4]
    std::vector<float> down_output_;        // [hidden_size]
};

#endif