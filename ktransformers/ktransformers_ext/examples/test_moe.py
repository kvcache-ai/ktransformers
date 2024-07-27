#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:34:06
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

with torch.inference_mode(mode=True):
    expert_num = 10
    hidden_size = 5120
    intermediate_size = 1536
    stride = 32
    group_min_len = 10
    group_max_len = 1024
    gate_type = 1 # ggml_type::GGML_TYPE_F16
    up_type = 1 # ggml_type::GGML_TYPE_F16
    down_type = 1 # ggml_type::GGML_TYPE_F16
    hidden_type = 1 # ggml_type::GGML_TYPE_F16
    n_routed_experts = 6
    qlen = 30
    layer_num = 10
    CPUInfer = cpuinfer_ext.CPUInfer(48)
    validation_iter = 100
    warm_up_iter = 1000
    test_iter = 10000

    moes = []
    gate_projs = []
    up_projs = []
    down_projs = []
    for _ in range(layer_num):
        gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, stride, group_min_len, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
        moe = cpuinfer_ext.moe.MOE(config)
        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        down_projs.append(down_proj)
        moes.append(moe)

    # validation
    for i in range(validation_iter):
        moe = moes[i % layer_num]
        expert_ids = torch.randint(0, expert_num, (qlen, n_routed_experts), dtype=torch.int64).contiguous()
        weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
        input = torch.randn((qlen, 1, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((qlen, 1, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100
        
        CPUInfer.submit(moe.forward, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        # print('cpuinfer output', output)

        def act_fn(x):
            return x / (1.0 + torch.exp(-x))
        t_output = torch.zeros((qlen, 1, hidden_size), dtype=torch.float32).contiguous()
        gate_proj = gate_projs[i%layer_num]
        up_proj = up_projs[i%layer_num]
        down_proj = down_projs[i%layer_num]
        for token_idx in range(qlen):
            for i, expert_id in enumerate(expert_ids[token_idx]):
                gate_buf = torch.mm(input[token_idx], gate_proj[expert_id].t())
                up_buf = torch.mm(input[token_idx], up_proj[expert_id].t())
                intermediate = act_fn(gate_buf) * up_buf
                expert_output = torch.mm(intermediate, down_proj[expert_id].t())
                t_output[token_idx] += weights[token_idx][i] * expert_output
        # print('torch output', t_output)

        diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
        print('diff = ', diff)
        assert(diff < 0.001)

    # warm up
    for i in range(warm_up_iter):
        moe = moes[i % layer_num]
        expert_ids = torch.randint(0, expert_num, (qlen, n_routed_experts), dtype=torch.int64).contiguous()
        weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
        input = torch.randn((qlen, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((qlen, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100
        CPUInfer.submit(moe.forward, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input.data_ptr(), output.data_ptr())
        CPUInfer.sync()

    # test
    total_time = 0
    for i in range(test_iter):
        moe = moes[i % layer_num]
        expert_ids = torch.randint(0, expert_num, (qlen, n_routed_experts), dtype=torch.int64).contiguous()
        weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
        input = torch.randn((qlen, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((qlen, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100
        start = time.perf_counter()
        CPUInfer.submit(moe.forward, qlen, n_routed_experts, expert_ids.data_ptr(), weights.data_ptr(), input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        end = time.perf_counter()
        total_time += end - start
    print('Time: ', total_time)
    print('Iteration: ', test_iter) 
    print('Time per iteration: ', total_time / test_iter)
    print('Bandwidth: ', hidden_size * intermediate_size * 3 * n_routed_experts * 2 * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
    print("All tasks completed.")