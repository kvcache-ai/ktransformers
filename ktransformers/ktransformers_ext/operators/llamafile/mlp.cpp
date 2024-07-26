/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-16 10:43:18
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:35:04
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "mlp.h"

MLP::MLP(MLPConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;

    input_fp32_.resize(config_.hidden_size);
    gate_input_.resize(config_.hidden_size * 4);
    up_input_.resize(config_.hidden_size * 4);
    gate_output_.resize(config_.intermediate_size);
    up_output_.resize(config_.intermediate_size);
    intermediate_fp32_.resize(config_.intermediate_size);
    down_input_.resize(config_.intermediate_size * 4);
    down_output_.resize(config_.hidden_size);
}

void MLP::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
    forward(input.data(), output.data(), backend);
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void MLP::forward(const void* input, void* output, Backend* backend) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, input_fp32_.data(), config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(input_fp32_.data(), gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = gate_input_.data();
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(input_fp32_.data(), gate_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = gate_input_.data();
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(input_fp32_.data(), up_input_.data(), config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = up_input_.data();
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth, [&](int task_id) {
        int ith = task_id;
        void* gate_proj_ptr = gate_proj_ + ith * config_.stride * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        float* gate_output_ptr = gate_output_.data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_proj_ptr = up_proj_ + ith * config_.stride * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        float* up_output_ptr = up_output_.data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            intermediate_fp32_[i] = act_fn(gate_output_[i]) * up_output_[i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = intermediate_fp32_.data() + ith * config_.stride;
            void* down_input_ptr = down_input_.data() + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    });
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        from_float(intermediate_fp32_.data(), down_input_.data(), config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, [&](int task_id) {
        int ith = task_id;
        void* down_proj_ptr = down_proj_ + ith * config_.stride * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        float* down_output_ptr = down_output_.data() + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_.data(), config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            void* output_ptr = output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(down_output_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    });
    if (config_.stride % ggml_blck_size(config_.hidden_type) != 0) {
        from_float(down_output_.data(), output, config_.hidden_size, config_.hidden_type);
    }
}
