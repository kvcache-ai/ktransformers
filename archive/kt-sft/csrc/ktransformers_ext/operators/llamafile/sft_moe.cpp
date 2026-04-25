/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-15 07:43:41
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "sft_moe.h"
#include <iostream>
#include <cstdint>
#include <cstring>
#include <time.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

SFT_MOE::SFT_MOE(SFT_MOEConfig config) {
    config_ = config;
    gate_proj_ = config_.gate_proj;
    up_proj_ = config_.up_proj;
    down_proj_ = config_.down_proj;
    
    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    gate_proj_numa_.resize(numa_nodes);
    up_proj_numa_.resize(numa_nodes);
    down_proj_numa_.resize(numa_nodes);
    size_t exp_inter_hidden_mul_ = (size_t)config.expert_num * config.intermediate_size * config.hidden_size;
    for (int i = 0; i < numa_nodes; i++) {
        gate_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type), i);
        up_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type), i);
        down_proj_numa_[i] = numa_alloc_onnode(exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type), i);
        if (!gate_proj_numa_[i]) {
            std::cout << "Memory allocation failed for gate_proj_numa_ on node " << i << std::endl;
        }
        if (!up_proj_numa_[i]) {
            std::cout << "Memory allocation failed for up_proj_numa_ on node " << i << std::endl;
        }
        if (!down_proj_numa_[i]) {
            std::cout << "Memory allocation failed for down_proj_numa_ on node " << i << std::endl;
        }
        memcpy(gate_proj_numa_[i], gate_proj_, exp_inter_hidden_mul_* ggml_type_size(config.gate_type) / ggml_blck_size(config.gate_type));
        memcpy(up_proj_numa_[i], up_proj_, exp_inter_hidden_mul_* ggml_type_size(config.up_type) / ggml_blck_size(config.up_type));
        memcpy(down_proj_numa_[i], down_proj_, exp_inter_hidden_mul_* ggml_type_size(config.down_type) / ggml_blck_size(config.down_type));
    }
    #endif

    std::vector<std::pair<void**, uint64_t>> s_mem_requests;
    s_mem_requests.push_back({(void**)&gate_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&up_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&down_proj_t_, config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    s_mem_requests.push_back({(void**)&transpose_buffer_fp32_, config_.expert_num * config_.intermediate_size * config_.hidden_size * sizeof(float)});
    s_mem_requests.push_back({(void**)&transpose_buffer_, config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.grad_type)});

    s_mem_requests.push_back({(void**)&s_input_fp32_, sizeof(float) * config_.hidden_size});
    s_mem_requests.push_back({(void**)&s_gate_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    s_mem_requests.push_back({(void**)&s_up_input_, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    s_gate_output_.resize(config_.routed_expert_num);
    s_up_output_.resize(config_.routed_expert_num);
    s_intermediate_fp32_.resize(config_.routed_expert_num);
    s_down_input_.resize(config_.routed_expert_num);
    s_down_output_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_gate_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_up_output_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_intermediate_fp32_[i], sizeof(float) * config_.intermediate_size});
        s_mem_requests.push_back({(void**)&s_down_input_[i], config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
        s_mem_requests.push_back({(void**)&s_down_output_[i], sizeof(float) * config_.hidden_size});
    }
    s_mem_requests.push_back({(void**)&s_output_fp32_, sizeof(float) * config_.hidden_size});
        
    s_down_input_grad_.resize(config_.routed_expert_num);
    s_gate_output_grad_fp32_.resize(config_.routed_expert_num);
    s_up_output_grad_fp32_.resize(config_.routed_expert_num);
    s_gate_output_grad_.resize(config_.routed_expert_num);
    s_up_output_grad_.resize(config_.routed_expert_num);
    s_gate_input_grad_.resize(config_.routed_expert_num);
    s_up_input_grad_.resize(config_.routed_expert_num);
    for (int i = 0; i < config_.routed_expert_num; i++) {
        s_mem_requests.push_back({(void**)&s_down_input_grad_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_fp32_[i], config_.intermediate_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_gate_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_up_output_grad_[i], config_.intermediate_size * ggml_type_size(config_.grad_type)});
        s_mem_requests.push_back({(void**)&s_gate_input_grad_[i], config_.hidden_size * sizeof(float)});
        s_mem_requests.push_back({(void**)&s_up_input_grad_[i], config_.hidden_size * sizeof(float)});
    }
    s_mem_requests.push_back({(void**)&s_input_grad_fp32_, config_.hidden_size * sizeof(float)});

    shared_mem_buffer.alloc(this, s_mem_requests);

    std::vector<std::pair<void**, uint64_t>> m_mem_requests;
    m_input_fp32_.resize(config_.group_max_len);
    m_gate_input_.resize(config_.group_max_len);
    m_up_input_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_input_fp32_[i], sizeof(float) * config_.hidden_size});
        m_mem_requests.push_back({(void**)&m_gate_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
        m_mem_requests.push_back({(void**)&m_up_input_[i], config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    }
    m_mem_requests.push_back({(void**)&m_local_gate_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_up_input_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_intermediate_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_down_input_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type)});
    m_mem_requests.push_back({(void**)&m_local_down_output_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_output_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_output_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    m_mem_requests.push_back({(void**)&m_local_down_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.hidden_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_down_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_fp32_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.intermediate_size});
    m_mem_requests.push_back({(void**)&m_local_gate_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_up_output_grad_, config_.routed_expert_num * config_.group_max_len * config_.intermediate_size * ggml_type_size(config_.grad_type)});
    m_mem_requests.push_back({(void**)&m_local_gate_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_up_input_grad_, sizeof(float) * config_.routed_expert_num * config_.group_max_len * config_.hidden_size});
    m_mem_requests.push_back({(void**)&m_local_token_indices_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_mem_requests.push_back({(void**)&m_local_expert_positions_, sizeof(int) * config_.routed_expert_num * config_.group_max_len});
    m_grad_input_fp32_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_mem_requests.push_back({(void**)&m_grad_input_fp32_[i], sizeof(float) * config_.hidden_size});
    }
    
    shared_mem_buffer.alloc(this, m_mem_requests);

    m_local_pos_.resize(config_.group_max_len);
    for (int i = 0; i < config_.group_max_len; i++) {
        m_local_pos_[i].resize(config_.routed_expert_num);
    }
    m_local_num_.resize(config_.expert_num);
    m_local_gate_input_ptr_.resize(config_.expert_num);
    m_local_up_input_ptr_.resize(config_.expert_num);
    m_local_gate_output_ptr_.resize(config_.expert_num);
    m_local_up_output_ptr_.resize(config_.expert_num);
    m_local_intermediate_fp32_ptr_.resize(config_.expert_num);
    m_local_down_input_ptr_.resize(config_.expert_num);
    m_local_down_output_ptr_.resize(config_.expert_num);
    
    // backward_many 专用指针数组初始化
    m_local_down_output_grad_ptr_.resize(config_.expert_num);
    m_local_down_input_grad_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_fp32_ptr_.resize(config_.expert_num);
    m_local_gate_output_grad_ptr_.resize(config_.expert_num);
    m_local_up_output_grad_ptr_.resize(config_.expert_num);
    m_local_gate_input_grad_ptr_.resize(config_.expert_num);
    m_local_up_input_grad_ptr_.resize(config_.expert_num);
    
    // fwd_cache访问映射指针数组初始化
    m_local_token_indices_ptr_.resize(config_.expert_num);
    m_local_expert_positions_ptr_.resize(config_.expert_num);
}

SFT_MOE::~SFT_MOE() {
    shared_mem_buffer.dealloc(this);

    #ifdef USE_NUMA
    int numa_nodes = numa_num_configured_nodes();
    for (int i = 0; i < numa_nodes; i++) {
        numa_free(gate_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type));
        numa_free(up_proj_numa_[i], config_.expert_num * config_.intermediate_size * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type));
        numa_free(down_proj_numa_[i], config_.expert_num * config_.hidden_size * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type));
    }
    #endif
}

void SFT_MOE::warm_up(Backend* backend) {
    std::vector<float> input_fp32(config_.hidden_size);
    std::vector<uint8_t> input(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type));
    for (int i = 0; i < config_.hidden_size; i++) {
        input_fp32[i] = 0;
    }
    from_float(input_fp32.data(), input.data(), config_.hidden_size, config_.hidden_type);
	/* ---------- 仅用于占位的 ForwardCache ---------- */
    SFT_MoEForwardCache dummy_cache; // 内容无用，只为满足接口
	dummy_cache.init(/*k=*/1, config_.intermediate_size);
    for (int i = 0; i < config_.expert_num; i++) {
        uint64_t expert_ids = i;
        float weights = 0;
        forward_one(1, &expert_ids, &weights, input.data(), output.data(), backend, &dummy_cache);
    }
}

static float act_fn(float x) {
    return x / (1.0f + expf(-x));
}

void SFT_MOE::ensure_fwd_cache(int qlen, int k)
{
	// if ((int)fw_cache_.size() < qlen)
	// 	fw_cache_.resize(qlen);
	// /* 只在扩容的那部分做 init，防止重复开辟 */
	// for (int i = 0; i < qlen; ++i)
	// 	fw_cache_[i].init(k, config_.intermediate_size);

	int old_sz = fw_cache_.size();
    if (old_sz < qlen)
    {
        fw_cache_.resize(qlen);
        for (int i = old_sz; i < qlen; ++i)  // 仅初始化新增元素
            fw_cache_[i].init(k, config_.intermediate_size);
    }

	
    // if ((int)fw_cache_.size() < qlen)
    //     fw_cache_.resize(qlen);

    // for (int i = 0; i < qlen; ++i)                          // 每轮都 init
    //     fw_cache_[i].init(k, config_.intermediate_size);    // 但 无重 alloc

}

SFT_MoEForwardCache* SFT_MOE::fwd_cache_ptr()
{
	return fw_cache_.empty() ? nullptr : fw_cache_.data();
}

void SFT_MOE::forward_one(int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    const void* gate_input_ptr;
    const void* up_input_ptr;
    if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
        gate_input_ptr = up_input_ptr = input;
    } else {
        to_float(input, s_input_fp32_, config_.hidden_size, config_.hidden_type);
        if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
            gate_input_ptr = up_input_ptr = s_gate_input_;
        } else {
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                from_float(s_input_fp32_, s_gate_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = s_gate_input_;
            } else {
                gate_input_ptr = input;
            }
            if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(s_input_fp32_, s_up_input_, config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                up_input_ptr = s_up_input_;
            } else {
                up_input_ptr = input;
            }
        }
    }
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
        
        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = s_gate_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = s_up_output_[expert_idx] + ith * config_.stride;
        llamafile_sgemm(config_.stride, 1, config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_intermediate_fp32_[expert_idx][i] = act_fn(s_gate_output_[expert_idx][i]) * s_up_output_[expert_idx][i];
        }
        if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) == 0) {
            float* intermediate_fp32_ptr = s_intermediate_fp32_[expert_idx] + ith * config_.stride;
            void* down_input_ptr = s_down_input_[expert_idx] + ith * config_.stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, config_.stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    if (config_.stride % ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) != 0) {
        for (int i = 0; i < k; i++) {
            from_float(s_intermediate_fp32_[i], s_down_input_[i], config_.intermediate_size, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_output_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            #ifdef USE_NUMA
            void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #else
            void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
            #endif
            
            float* down_output_ptr = s_down_output_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), s_down_input_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_output_fp32_[i] += s_down_output_[expert_idx][i] * weights[expert_idx];
            }
        }
        if (config_.stride % ggml_blck_size(config_.hidden_type) == 0) {
            float* output_fp32_ptr = s_output_fp32_ + ith * config_.stride;
            void* output_ptr = (uint8_t*)output + ith * config_.stride * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
            from_float(output_fp32_ptr, output_ptr, config_.stride, config_.hidden_type);
        }
    }, nullptr);

	for (int e = 0; e < k; ++e) {
        // gate_output_: float[inter_size] per expert
        std::memcpy(fwd_cache->gate_u[e].data(),
                    s_gate_output_[e],
                    sizeof(float) * config_.intermediate_size);

        std::memcpy(fwd_cache->up_v[e].data(),
                    s_up_output_[e],
                    sizeof(float) * config_.intermediate_size);

        // 可选保存 z
        // std::memcpy(fwd_cache->z[e].data(),
        //             s_intermediate_fp32_[e],
        //             sizeof(float) * config_.intermediate_size);
    }
}

void SFT_MOE::forward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_gate_input_ptr_[i] = m_local_gate_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
        m_local_up_input_ptr_[i] = m_local_up_input_ + offset * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
        m_local_gate_output_ptr_[i] = m_local_gate_output_ + offset * config_.intermediate_size;
        m_local_up_output_ptr_[i] = m_local_up_output_ + offset * config_.intermediate_size;
        m_local_intermediate_fp32_ptr_[i] = m_local_intermediate_fp32_ + offset * config_.intermediate_size;
        m_local_down_input_ptr_[i] = m_local_down_input_ + offset * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        m_local_down_output_ptr_[i] = m_local_down_output_ + offset * config_.hidden_size;
        offset += m_local_num_[i];
    }
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        const void* gate_input_ptr;
        const void* up_input_ptr;
        if (config_.hidden_type == ggml_internal_get_type_traits(config_.gate_type).vec_dot_type && config_.hidden_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
            gate_input_ptr = up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
        } else {
            to_float((uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), m_input_fp32_[i], config_.hidden_size, config_.hidden_type);
            if (ggml_internal_get_type_traits(config_.gate_type).vec_dot_type == ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                gate_input_ptr = up_input_ptr = m_gate_input_[i];
            } else {
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_gate_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type);
                    gate_input_ptr = m_gate_input_[i];
                } else {
                    gate_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
                if (config_.hidden_type != ggml_internal_get_type_traits(config_.up_type).vec_dot_type) {
                    from_float(m_input_fp32_[i], m_up_input_[i], config_.hidden_size, ggml_internal_get_type_traits(config_.up_type).vec_dot_type);
                    up_input_ptr = m_up_input_[i];
                } else {
                    up_input_ptr = (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type);
                }
            }
        }
        for (int j = 0; j < k; j++) {
            memcpy(m_local_gate_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type), gate_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.gate_type).vec_dot_type));
            memcpy(m_local_up_input_ptr_[expert_ids[i * k + j]] + m_local_pos_[i][j] * config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type), up_input_ptr, config_.hidden_size * ggml_type_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.up_type).vec_dot_type));
        }
    }, nullptr);
    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* gate_input_ptr = m_local_gate_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* gate_proj_ptr = (uint8_t*)gate_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #else
        void* gate_proj_ptr = (uint8_t*)gate_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.gate_type) / ggml_blck_size(config_.gate_type);
        #endif

        float* gate_output_ptr = m_local_gate_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.gate_type), gate_proj_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_input_ptr, config_.hidden_size / ggml_blck_size(config_.gate_type), gate_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.gate_type, ggml_internal_get_type_traits(config_.gate_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        void* up_input_ptr = m_local_up_input_ptr_[expert_idx];

        #ifdef USE_NUMA
        void* up_proj_ptr = (uint8_t*)up_proj_numa_[Backend::numa_node] + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #else
        void* up_proj_ptr = (uint8_t*)up_proj_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.up_type) / ggml_blck_size(config_.up_type);
        #endif

        float* up_output_ptr = m_local_up_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size / ggml_blck_size(config_.up_type), up_proj_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_input_ptr, config_.hidden_size / ggml_blck_size(config_.up_type), up_output_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.up_type, ggml_internal_get_type_traits(config_.up_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_intermediate_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = act_fn(m_local_gate_output_ptr_[expert_idx][i * config_.intermediate_size + j]) * m_local_up_output_ptr_[expert_idx][i * config_.intermediate_size + j];
            }
            float* intermediate_fp32_ptr = m_local_intermediate_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* down_input_ptr = m_local_down_input_ptr_[expert_idx] + i * config_.intermediate_size * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) + ith * stride * ggml_type_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type) / ggml_blck_size(ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
            from_float(intermediate_fp32_ptr, down_input_ptr, stride, ggml_internal_get_type_traits(config_.down_type).vec_dot_type);
        }
    }, nullptr);
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        void* down_input_ptr = m_local_down_input_ptr_[expert_idx];
        
        #ifdef USE_NUMA
        void* down_proj_ptr = (uint8_t*)down_proj_numa_[Backend::numa_node] + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #else
        void* down_proj_ptr = (uint8_t*)down_proj_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.down_type) / ggml_blck_size(config_.down_type);
        #endif

        float* down_output_ptr = m_local_down_output_ptr_[expert_idx] + ith * stride;
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size / ggml_blck_size(config_.down_type), down_proj_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_input_ptr, config_.intermediate_size / ggml_blck_size(config_.down_type), down_output_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.down_type, ggml_internal_get_type_traits(config_.down_type).vec_dot_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_output_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_output_fp32_[i][e] += m_local_down_output_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] * weights[i * k + j];
            }
        }
        from_float(m_output_fp32_[i], (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), config_.hidden_size, config_.hidden_type);
    }, nullptr);

	/* 把每个 token-expert 的行复制到各自 cache */
    backend->do_work_stealing_job(qlen, nullptr, [&](int token_idx) {
        auto& cache = fwd_cache[token_idx];
        // cache 已在上层 init(k, inter_size)
        for (int j = 0; j < k; ++j) {
            uint64_t  eid   = expert_ids[token_idx*k + j];
            int       row   = m_local_pos_[token_idx][j];
            size_t    ofs   = row * config_.intermediate_size;
            /* gate u */
            std::memcpy(cache.gate_u[j].data(),
                        m_local_gate_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* up v */
            std::memcpy(cache.up_v[j].data(),
                        m_local_up_output_ptr_[eid] + ofs,
                        sizeof(float) * config_.intermediate_size);
            /* 可选 z */
            // std::memcpy(cache.z[j].data(),
            //             m_local_intermediate_fp32_ptr_[eid] + ofs,
            //             sizeof(float) * config_.intermediate_size);
        }
    }, nullptr);
}

void SFT_MOE::forward(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* input, void* output, Backend* backend, SFT_MoEForwardCache* fwd_cache) {
    if (qlen < config_.group_min_len) {
        for (int i = 0; i < qlen; i++) {
			// fwd_cache[i].init(k, config_.intermediate_size);      // 预分配
            forward_one(k, expert_ids + i * k, weights + i * k, (uint8_t*)input + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + i * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + i);
        }
        return;
    }
    int forward_len = std::min(config_.group_max_len, qlen);
    // for (int i = 0; i < forward_len; ++i)
    //     fwd_cache[i].init(k, config_.intermediate_size);
    forward_many(forward_len, k, expert_ids, weights, input, output, backend, fwd_cache);
    forward(qlen - forward_len, k, expert_ids + forward_len * k, weights + forward_len * k, (uint8_t*)input + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), (uint8_t*)output + forward_len * config_.hidden_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), backend, fwd_cache + forward_len);
}

static float act_fn_grad(float x) {
    float sigmoid_x = 1.0f / (1.0f + expf(-x));
    return sigmoid_x * (1. + x * (1. - sigmoid_x));
}

void SFT_MOE::transpose_expert_matrix(const void* src, void* dst, int R, int C, ggml_type src_type, ggml_type dst_type, uint64_t expert_idx) {
    to_float(src, transpose_buffer_fp32_ + (R * C * expert_idx), R * C, src_type);
    from_float(transpose_buffer_fp32_ + (R * C * expert_idx), transpose_buffer_ + (R * C * expert_idx) * ggml_type_size(dst_type), R * C, dst_type);
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            memcpy(
                (uint8_t*)dst + (c * R + r) * ggml_type_size(dst_type),
                (uint8_t*)transpose_buffer_ + (R * C * expert_idx + r * C + c) * ggml_type_size(dst_type),
                ggml_type_size(dst_type));
        }
    }
}

void SFT_MOE::get_transpose(Backend* backend) {
    // Transpose gate_proj_
    int R_gate = config_.intermediate_size;
    int C_gate = config_.hidden_size;
    size_t gate_expert_src_stride_bytes = (size_t)R_gate * C_gate * ggml_type_size(config_.gate_type);
    size_t gate_expert_dst_t_stride_bytes = (size_t)C_gate * R_gate * ggml_type_size(config_.grad_type);
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)gate_proj_ + expert_idx * gate_expert_src_stride_bytes;
        void* dst_expert_t = (uint8_t*)gate_proj_t_ + expert_idx * gate_expert_dst_t_stride_bytes;
        transpose_expert_matrix(src_expert, dst_expert_t, R_gate, C_gate, config_.gate_type, config_.grad_type, expert_idx);
    }, nullptr);

    // Transpose up_proj_
    int R_up = config_.intermediate_size;
    int C_up = config_.hidden_size;
    size_t up_expert_src_stride_bytes = (size_t)R_up * C_up * ggml_type_size(config_.up_type);
    size_t up_expert_dst_t_stride_bytes = (size_t)C_up * R_up * ggml_type_size(config_.grad_type);
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)up_proj_ + expert_idx * up_expert_src_stride_bytes;
        void* dst_expert_t = (uint8_t*)up_proj_t_ + expert_idx * up_expert_dst_t_stride_bytes;
        transpose_expert_matrix(src_expert, dst_expert_t, R_up, C_up, config_.up_type, config_.grad_type, expert_idx);
    }, nullptr);

    // Transpose down_proj_
    int R_down = config_.hidden_size;
    int C_down = config_.intermediate_size;
    size_t down_expert_src_stride_bytes = (size_t)R_down * C_down * ggml_type_size(config_.down_type);
    size_t down_expert_dst_t_stride_bytes = (size_t)C_down * R_down * ggml_type_size(config_.grad_type);
    backend->do_work_stealing_job(config_.expert_num, nullptr, [&](int expert_idx) {
        void* src_expert = (uint8_t*)down_proj_ + expert_idx * down_expert_src_stride_bytes;
        void* dst_expert_t = (uint8_t*)down_proj_t_ + expert_idx * down_expert_dst_t_stride_bytes;
        transpose_expert_matrix(src_expert, dst_expert_t, R_down, C_down, config_.down_type, config_.grad_type, expert_idx);
    }, nullptr);
}

void SFT_MOE::backward_one(int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {
	// clock_t clk1, clk2, clk3, clk4;
	// clock_t clkz1, clkz2, clkz3, clkz4, clkz5;
	// clk1 = clock();
	// clk2 = clock();
    int nth = config_.intermediate_size / config_.stride;
    backend->do_work_stealing_job(nth * k, nullptr, [&](int task_id) {
        int expert_idx = task_id / nth;
        uint64_t expert_id = expert_ids[expert_idx];
        int ith = task_id % nth;
		// clkz1 = clock();
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_id * config_.intermediate_size + ith * config_.stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        float* down_input_grad_ptr = s_down_input_grad_[expert_idx] + ith * config_.stride;
        // clkz2 = clock();
        llamafile_sgemm(config_.stride, 1, config_.hidden_size, down_proj_t_ptr, config_.hidden_size, output_grad, config_.hidden_size, down_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        // clkz3 = clock();
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_down_input_grad_[expert_idx][i] *= weights[expert_idx];

            s_gate_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * fwd_cache->up_v[expert_idx][i] * act_fn_grad(fwd_cache->gate_u[expert_idx][i]); 
            s_up_output_grad_fp32_[expert_idx][i] = s_down_input_grad_[expert_idx][i] * act_fn(fwd_cache->gate_u[expert_idx][i]);
        }
        // clkz4 = clock();
        from_float(s_gate_output_grad_fp32_[expert_idx] + ith * config_.stride, s_gate_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        from_float(s_up_output_grad_fp32_[expert_idx] + ith * config_.stride, s_up_output_grad_[expert_idx] + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
        // clkz5 = clock();
    }, nullptr);

	// clk3 = clock();
    nth = config_.hidden_size / config_.stride;
    backend->do_work_stealing_job(nth, nullptr, [&](int task_id) {
        int ith = task_id;
        for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
            s_input_grad_fp32_[i] = 0;
        }
        for (int expert_idx = 0; expert_idx < k; expert_idx++) {
            uint64_t expert_id = expert_ids[expert_idx];

            void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* gate_input_grad_ptr = s_gate_input_grad_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, s_gate_output_grad_[expert_idx], config_.intermediate_size, gate_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);

            void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_id * config_.hidden_size + ith * config_.stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
            float* up_input_grad_ptr = s_up_input_grad_[expert_idx] + ith * config_.stride;
            llamafile_sgemm(config_.stride, 1, config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, s_up_output_grad_[expert_idx], config_.intermediate_size, up_input_grad_ptr, config_.stride, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
            
            for (int i = ith * config_.stride; i < (ith + 1) * config_.stride; i++) {
                s_input_grad_fp32_[i] += s_gate_input_grad_[expert_idx][i] + s_up_input_grad_[expert_idx][i];
            }
        }
        from_float(s_input_grad_fp32_ + ith * config_.stride, (uint8_t*)input_grad + ith * config_.stride * ggml_type_size(config_.grad_type), config_.stride, config_.grad_type);
    }, nullptr);
	// clk4 = clock();
	// std::cout << "[Δclk12] " << (clk2 - clk1) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclk23] " << (clk3 - clk2) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclk34] " << (clk4 - clk3) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz12] " << (clkz2 - clkz1) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz23] " << (clkz3 - clkz2) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz34] " << (clkz4 - clkz3) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms  [Δclkz45] " << (clkz5 - clkz4) / static_cast<double>(CLOCKS_PER_SEC) * 1000
    //       << " ms\n";

}

void SFT_MOE::backward_many(int qlen, int k, const uint64_t* expert_ids, const float* weights, const void* output_grad, void* input_grad, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_num_[i] = 0;
    }
    for (int i = 0; i < qlen; i++) {
        for (int j = 0; j < k; j++) {
            m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
        }
    }
    uint64_t offset = 0;
    for (int i = 0; i < config_.expert_num; i++) {
        m_local_down_output_grad_ptr_[i] = m_local_down_output_grad_ + offset * config_.hidden_size * ggml_type_size(config_.grad_type);
        m_local_down_input_grad_ptr_[i] = m_local_down_input_grad_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_fp32_ptr_[i] = m_local_gate_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_up_output_grad_fp32_ptr_[i] = m_local_up_output_grad_fp32_ + offset * config_.intermediate_size;
        m_local_gate_output_grad_ptr_[i] = m_local_gate_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_up_output_grad_ptr_[i] = m_local_up_output_grad_ + offset * config_.intermediate_size * ggml_type_size(config_.grad_type);
        m_local_gate_input_grad_ptr_[i] = m_local_gate_input_grad_ + offset * config_.hidden_size;
        m_local_up_input_grad_ptr_[i] = m_local_up_input_grad_ + offset * config_.hidden_size;
        m_local_token_indices_ptr_[i] = m_local_token_indices_ + offset;
        m_local_expert_positions_ptr_[i] = m_local_expert_positions_ + offset;
        offset += m_local_num_[i];
    }

    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int j = 0; j < k; j++) {
            uint64_t expert_id = expert_ids[i * k + j];
            int local_row = m_local_pos_[i][j];
            memcpy(m_local_down_output_grad_ptr_[expert_id] + local_row * config_.hidden_size * ggml_type_size(config_.grad_type), (uint8_t*)output_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size * ggml_type_size(config_.grad_type));
            m_local_token_indices_ptr_[expert_id][local_row] = i;
            m_local_expert_positions_ptr_[expert_id][local_row] = j;
        }
    }, nullptr);

    // get_transpose(backend);

    int stride = QK_K;
    int nth = config_.intermediate_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        
        void* down_proj_t_ptr = (uint8_t*)down_proj_t_ + (expert_idx * config_.intermediate_size + ith * stride) * config_.hidden_size * ggml_type_size(config_.grad_type);
        void* down_output_grad_ptr = m_local_down_output_grad_ptr_[expert_idx];
        float* down_input_grad_ptr = m_local_down_input_grad_ptr_[expert_idx] + ith * stride;
                    
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.hidden_size, down_proj_t_ptr, config_.hidden_size, down_output_grad_ptr, config_.hidden_size, down_input_grad_ptr, config_.intermediate_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        
        for (int i = 0; i < m_local_num_[expert_idx]; i++) {
            int token_idx = m_local_token_indices_ptr_[expert_idx][i];
            int expert_pos = m_local_expert_positions_ptr_[expert_idx][i];
            float weight = weights[token_idx * k + expert_pos];
            
            for (int j = ith * stride; j < (ith + 1) * stride; j++) {
                m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j] *= weight;
                
                float down_input_grad = m_local_down_input_grad_ptr_[expert_idx][i * config_.intermediate_size + j];
                m_local_gate_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * fwd_cache[token_idx].up_v[expert_pos][j] * act_fn_grad(fwd_cache[token_idx].gate_u[expert_pos][j]);
                m_local_up_output_grad_fp32_ptr_[expert_idx][i * config_.intermediate_size + j] = down_input_grad * act_fn(fwd_cache[token_idx].gate_u[expert_pos][j]);
            }
            
            float* gate_output_grad_fp32_ptr = m_local_gate_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(gate_output_grad_fp32_ptr, gate_output_grad_ptr, stride, config_.grad_type);
            
            float* up_output_grad_fp32_ptr = m_local_up_output_grad_fp32_ptr_[expert_idx] + i * config_.intermediate_size + ith * stride;
            void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx] + (i * config_.intermediate_size + ith * stride) * ggml_type_size(config_.grad_type);
            from_float(up_output_grad_fp32_ptr, up_output_grad_ptr, stride, config_.grad_type);
        }
    }, nullptr);
    stride = QK_K;
    nth = config_.hidden_size / stride;
    backend->do_work_stealing_job(nth * config_.expert_num, nullptr, [&](int task_id) {
        uint64_t expert_idx = task_id / nth;
        int ith = task_id % nth;
        
        void* gate_proj_t_ptr = (uint8_t*)gate_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* up_proj_t_ptr = (uint8_t*)up_proj_t_ + (expert_idx * config_.hidden_size + ith * stride) * config_.intermediate_size * ggml_type_size(config_.grad_type);
        void* gate_output_grad_ptr = m_local_gate_output_grad_ptr_[expert_idx];
        void* up_output_grad_ptr = m_local_up_output_grad_ptr_[expert_idx];
        float* gate_input_grad_ptr = m_local_gate_input_grad_ptr_[expert_idx] + ith * stride;
        float* up_input_grad_ptr = m_local_up_input_grad_ptr_[expert_idx] + ith * stride;
        
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size, gate_proj_t_ptr, config_.intermediate_size, gate_output_grad_ptr, config_.intermediate_size, gate_input_grad_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
        llamafile_sgemm(stride, m_local_num_[expert_idx], config_.intermediate_size, up_proj_t_ptr, config_.intermediate_size, up_output_grad_ptr, config_.intermediate_size, up_input_grad_ptr, config_.hidden_size, 0, 1, GGML_TASK_TYPE_COMPUTE, config_.grad_type, config_.grad_type, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    }, nullptr);
    backend->do_work_stealing_job(qlen, nullptr, [&](int i) {
        for (int e = 0; e < config_.hidden_size; e++) {
            m_grad_input_fp32_[i][e] = 0;
        }
        for (int j = 0; j < k; j++) {
            for (int e = 0; e < config_.hidden_size; e++) {
                m_grad_input_fp32_[i][e] += m_local_gate_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e] + m_local_up_input_grad_ptr_[expert_ids[i * k + j]][m_local_pos_[i][j] * config_.hidden_size + e];
            }
        }
        from_float(m_grad_input_fp32_[i], (uint8_t*)input_grad + i * config_.hidden_size * ggml_type_size(config_.grad_type), config_.hidden_size, config_.grad_type);
    }, nullptr);
}

// TODO: input和layer_idx参数可以删除
void SFT_MOE::backward(int layer_idx, int qlen, int k, const uint64_t* expert_ids, const float* weights,
                   const void* input, const void* grad_output, void* grad_input, Backend* backend, const SFT_MoEForwardCache* fwd_cache) {

    get_transpose(backend);
    int remaining_qlen = qlen;
    int processed_offset = 0;
    
    while (remaining_qlen > 0) {
        // config_.group_min_len = 10000000;
        if (remaining_qlen < config_.group_min_len) {
            for (int i = 0; i < remaining_qlen; i++) {
                backward_one(k,
                             expert_ids + (processed_offset + i) * k,
                             weights + (processed_offset + i) * k,
                             (uint8_t*)grad_output + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             (uint8_t*)grad_input + (processed_offset + i) * config_.hidden_size * ggml_type_size(config_.grad_type),
                             backend,
                             fwd_cache + processed_offset + i);
            }
            break;
        } else {
            int backward_len = std::min(config_.group_max_len, remaining_qlen);
            backward_many(backward_len, 
                         k, 
                         expert_ids + processed_offset * k, 
                         weights + processed_offset * k, 
                         (uint8_t*)grad_output + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         (uint8_t*)grad_input + processed_offset * config_.hidden_size * ggml_type_size(config_.grad_type), 
                         backend, 
                         fwd_cache + processed_offset);
            
            remaining_qlen -= backward_len;
            processed_offset += backward_len;
        }
    }
}