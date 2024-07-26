/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-12 10:07:58
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022 
 * @LastEditTime : 2024-07-25 10:34:58
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "linear.h"

Linear::Linear(LinearConfig config) {
    config_ = config;
    proj_ = config_.proj;

    input_fp32_.resize(config_.input_size);
    proj_input_.resize(config_.input_size * 4);
    proj_output_.resize(config_.output_size);
}

void Linear::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.input_size);
    std::vector<uint8_t> input(config_.input_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.output_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.input_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.input_size, config_.hidden_type);
    forward(input.data(), output.data(), backend);
}

void Linear::forward(const void* input, void* output, Backend* backend) {
    const void* proj_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.proj_type).vec_dot_type) {
        proj_input_ptr = input;
    } else {
        to_float(input, input_fp32_.data(), config_.input_size, config_.hidden_type);
        from_float(input_fp32_.data(), proj_input_.data(), config_.input_size, ggml_internal_get_type_traits(config_.proj_type).vec_dot_type);
        proj_input_ptr = proj_input_.data();
    }
    int nth = config_.output_size / config_.stride;
    backend->do_work_stealing_job(nth, [&](int task_id) {
        int ith = task_id % nth;
        llamafile_sgemm(config_.output_size, 1, config_.input_size / ggml_blck_size(config_.proj_type), proj_, config_.input_size / ggml_blck_size(config_.proj_type), proj_input_ptr, config_.input_size / ggml_blck_size(config_.proj_type), proj_output_.data(), config_.output_size, ith, nth, GGML_TASK_TYPE_COMPUTE, config_.proj_type, ggml_internal_get_type_traits(config_.proj_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    });
    from_float(proj_output_.data(), output, config_.output_size, config_.hidden_type);
}