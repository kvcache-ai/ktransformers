/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:35:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_OPERATOR_SFT_MOE_H
#define CPUINFER_OPERATOR_SFT_MOE_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "../../cpu_backend/backend.h"
#include "../../cpu_backend/shared_mem_buffer.h"
#include "conversion.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "llamafile/sgemm.h"
#include "sft_moe_forward_cache.h"

struct SFT_MOEConfig {
    long expert_num;
    int routed_expert_num;
    long hidden_size;
    long intermediate_size;
    int stride;
    int group_min_len;
    int group_max_len;
    void* gate_proj;
    void* up_proj;
    void* down_proj;
    ggml_type gate_type;
    ggml_type up_type;
    ggml_type down_type;
    ggml_type hidden_type;
    ggml_type grad_type = GGML_TYPE_BF16;

    SFT_MOEConfig() {}

    SFT_MOEConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size, int stride, int group_min_len, int group_max_len, void* gate_proj, void* up_proj, void* down_proj, ggml_type gate_type, ggml_type up_type, ggml_type down_type, ggml_type hidden_type)
        : expert_num(expert_num), routed_expert_num(routed_expert_num), hidden_size(hidden_size), intermediate_size(intermediate_size), stride(stride), group_min_len(group_min_len), group_max_len(group_max_len), gate_proj(gate_proj), up_proj(up_proj), down_proj(down_proj), gate_type(gate_type), up_type(up_type), down_type(down_type), hidden_type(hidden_type) {}
};

class SFT_MOE {
   public:
    SFT_MOE(SFT_MOEConfig);
    ~SFT_MOE();
    void warm_up(Backend* backend);
    void forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache);
    void forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache);
    void forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache);
	void backward_one(int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache);
	void backward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache);
	void backward(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights,
              const void* input, const void* grad_output, void* grad_input, Backend* backend, const SFT_MoEForwardCache* fwd_cache); // FIXME: expert backward definition for C++
    
    void transpose_expert_matrix(const void* src, void* dst, int R, int C, ggml_type src_type, ggml_type dst_type, uint64_t expert_idx);
    void ensure_fwd_cache(int qlen, int k);
    void get_transpose(Backend* backend);
    SFT_MoEForwardCache* fwd_cache_ptr();

   private:
    SFT_MOEConfig config_;
    void* gate_proj_;  // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* up_proj_;    // [expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    void* down_proj_;  // [expert_num * hidden_size * intermediate_size ( /32 if quantized)]

    float* transpose_buffer_fp32_;  // [expert_num * intermediate_size * hidden_size]
    uint8_t* transpose_buffer_;     // [expert_num * intermediate_size * hidden_size]

    uint8_t* gate_proj_t_;  // [expert_num * hidden_size * intermediate_size]
    uint8_t* up_proj_t_;    // [expert_num * hidden_size * intermediate_size]
    uint8_t* down_proj_t_;  // [expert_num * intermediate_size * hidden_size]

    #ifdef USE_NUMA
    std::vector<void*> gate_proj_numa_;  // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> up_proj_numa_;    // [numa_num, expert_num * intermediate_size * hidden_size ( /32 if quantized)]
    std::vector<void*> down_proj_numa_;  // [numa_num, expert_num * hidden_size * intermediate_size ( /32 if quantized)]
    #endif

    float* s_input_fp32_;                      // [hidden_size]
    uint8_t* s_gate_input_;                    // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* s_up_input_;                      // [hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    std::vector<float*> s_gate_output_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_;          // [routed_expert_num, intermediate_size]
    std::vector<float*> s_intermediate_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_down_input_;       // [routed_expert_num, intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    std::vector<float*> s_down_output_;        // [routed_expert_num, hidden_size]
    float* s_output_fp32_;                     // [hidden_size]

    std::vector<float*> s_down_input_grad_;        // [routed_expert_num, intermediate_size]
    std::vector<float*> s_gate_output_grad_fp32_;  // [routed_expert_num, intermediate_size]
    std::vector<float*> s_up_output_grad_fp32_;    // [routed_expert_num, intermediate_size]
    std::vector<uint8_t*> s_gate_output_grad_;     // [routed_expert_num, intermediate_size * ggml_type_size(grad_type)]
    std::vector<uint8_t*> s_up_output_grad_;       // [routed_expert_num, intermediate_size * ggml_type_size(grad_type)]
    std::vector<float*> s_gate_input_grad_;        // [routed_expert_num, hidden_size]
    std::vector<float*> s_up_input_grad_;          // [routed_expert_num, hidden_size]
    float* s_input_grad_fp32_;                     // [hidden_size]

    std::vector<float*> m_input_fp32_;    // [group_max_len, hidden_size]
    std::vector<uint8_t*> m_gate_input_;  // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    std::vector<uint8_t*> m_up_input_;    // [group_max_len, hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    uint8_t* m_local_gate_input_;         // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(gate_type).vec_dot_type)]
    uint8_t* m_local_up_input_;           // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(ggml_internal_get_type_traits(up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(up_type).vec_dot_type)]
    float* m_local_gate_output_;          // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_;            // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_intermediate_fp32_;    // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_down_input_;         // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(ggml_internal_get_type_traits(down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(down_type).vec_dot_type)]
    float* m_local_down_output_;          // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_output_fp32_;   // [group_max_len, hidden_size]

    std::vector<std::vector<int>> m_local_pos_;          // [group_max_len, routed_expert_num]
    std::vector<int> m_local_num_;                       // [expert_num]
    std::vector<uint8_t*> m_local_gate_input_ptr_;       // [expert_num]
    std::vector<uint8_t*> m_local_up_input_ptr_;         // [expert_num]
    std::vector<float*> m_local_gate_output_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_output_ptr_;          // [expert_num]
    std::vector<float*> m_local_intermediate_fp32_ptr_;  // [expert_num]
    std::vector<uint8_t*> m_local_down_input_ptr_;       // [expert_num]
    std::vector<float*> m_local_down_output_ptr_;        // [expert_num]

    uint8_t* m_local_down_output_grad_;                  // [routed_expert_num * group_max_len * hidden_size * ggml_type_size(grad_type)]
    float* m_local_down_input_grad_;                     // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_gate_output_grad_fp32_;               // [routed_expert_num * group_max_len * intermediate_size]
    float* m_local_up_output_grad_fp32_;                 // [routed_expert_num * group_max_len * intermediate_size]
    uint8_t* m_local_gate_output_grad_;                  // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(grad_type)]
    uint8_t* m_local_up_output_grad_;                    // [routed_expert_num * group_max_len * intermediate_size * ggml_type_size(grad_type)]
    float* m_local_gate_input_grad_;                     // [routed_expert_num * group_max_len * hidden_size]
    float* m_local_up_input_grad_;                       // [routed_expert_num * group_max_len * hidden_size]
    std::vector<float*> m_grad_input_fp32_;              // [group_max_len, hidden_size]

    std::vector<uint8_t*> m_local_down_output_grad_ptr_;     // [expert_num]
    std::vector<float*> m_local_down_input_grad_ptr_;        // [expert_num]
    std::vector<float*> m_local_gate_output_grad_fp32_ptr_;  // [expert_num]
    std::vector<float*> m_local_up_output_grad_fp32_ptr_;    // [expert_num]
    std::vector<uint8_t*> m_local_gate_output_grad_ptr_;     // [expert_num]
    std::vector<uint8_t*> m_local_up_output_grad_ptr_;       // [expert_num]
    std::vector<float*> m_local_gate_input_grad_ptr_;        // [expert_num]
    std::vector<float*> m_local_up_input_grad_ptr_;          // [expert_num]

    int* m_local_token_indices_;                             // [routed_expert_num * group_max_len]
    int* m_local_expert_positions_;                          // [routed_expert_num * group_max_len]
    std::vector<int*> m_local_token_indices_ptr_;            // [expert_num]
    std::vector<int*> m_local_expert_positions_ptr_;         // [expert_num]

	std::vector<SFT_MoEForwardCache> fw_cache_; // 持久缓存，便于backward读取到forward_cache
};

#endif