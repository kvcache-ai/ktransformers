#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:34:03
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

with torch.inference_mode(mode=True):
    hidden_size = 5120
    intermediate_size = 3072
    stride = 32
    gate_type = 1 # ggml_type::GGML_TYPE_F16
    up_type = 1 # ggml_type::GGML_TYPE_F16
    down_type = 1 # ggml_type::GGML_TYPE_F16
    hidden_type = 1 # ggml_type::GGML_TYPE_F16
    layer_num = 10
    CPUInfer = cpuinfer_ext.CPUInfer(48)
    validation_iter = 100
    warm_up_iter = 1000
    test_iter = 10000

    mlps = []
    gate_projs = []
    up_projs = []
    down_projs = []
    for _ in range(layer_num):
        gate_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        up_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        down_proj = torch.randn((hidden_size, intermediate_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        config = cpuinfer_ext.mlp.MLPConfig(hidden_size, intermediate_size, stride, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
        mlp = cpuinfer_ext.mlp.MLP(config)
        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        down_projs.append(down_proj)
        mlps.append(mlp)

    # validation
    for i in range(validation_iter):
        mlp = mlps[i % layer_num]
        input = torch.randn((1, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100

        CPUInfer.submit(mlp.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        # print('cpuinfer output', output)

        def act_fn(x):
            return x / (1.0 + torch.exp(-x))
        gate_proj = gate_projs[i%layer_num]
        up_proj = up_projs[i%layer_num]
        down_proj = down_projs[i%layer_num]
        gate_buf = torch.mm(input, gate_proj.t())
        up_buf = torch.mm(input, up_proj.t())
        intermediate = act_fn(gate_buf) * up_buf
        t_output = torch.mm(intermediate, down_proj.t())
        # print('torch output', t_output)

        diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
        print('diff = ', diff)
        assert(diff < 0.001)

    # warm up
    for i in range(warm_up_iter):
        mlp = mlps[i % layer_num]
        input = torch.randn((1, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100
        CPUInfer.submit(mlp.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()

    # test
    total_time = 0
    for i in range(test_iter):
        mlp = mlps[i % layer_num]
        input = torch.randn((1, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100
        start = time.time()
        CPUInfer.submit(mlp.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        end = time.time()
        total_time += end - start
    print('Time: ', total_time)
    print('Iteration: ', test_iter) 
    print('Time per iteration: ', total_time / test_iter)
    print('Bandwidth: ', hidden_size * intermediate_size * 3 * 2 * test_iter / total_time / 1024 / 1024 / 1024, 'GB/s')
    print("All tasks completed.")