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
#include "shared_mem_buffer.h"

struct MLPConfig {
    int hidden_size;
    int intermediate_size;
    int stride;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;

    MLPConfig() {}

    MLPConfig(int hidden_size, int intermediate_size, int stride, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class MLP {
   public:
    MLP(MLPConfig);
    ~MLP();
    void warm_up(Backend* backend);
    void forward_many(int qlen, const void* input, void* output, Backend* backend);
    void forward(int qlen, const void* input, void* output, Backend* backend);

   private:
    MLPConfig config_;
    void* gate_proj_;  // [intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [hidden_size * intermediate_size ( /32 if quantized)]

    float* input_fp32_;         // [group_max_len * hidden_size]
    uint8_t* gate_input_;       // [group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* up_input_;         // [group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* gate_output_;        // [group_max_len * intermediate_size]
    float* up_output_;          // [group_max_len * intermediate_size]
    float* intermediate_fp32_;  // [group_max_len * intermediate_size]
    uint8_t* down_input_;       // [group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* down_output_;        // [group_max_len * hidden_size]
};

#endif