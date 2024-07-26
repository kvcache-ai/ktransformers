#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:34:00
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

with torch.inference_mode(mode=True):
    input_size = 16384
    output_size = 5120
    stride = 32
    proj_type = 1 # ggml_type::GGML_TYPE_F16
    hidden_type = 1 # ggml_type::GGML_TYPE_F16
    layer_num = 10
    CPUInfer = cpuinfer_ext.CPUInfer(48)
    validation_iter = 100
    warm_up_iter = 1000
    test_iter = 10000

    linears = []
    projs = []
    for _ in range(layer_num):
        proj = torch.randn((output_size, input_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        config = cpuinfer_ext.linear.LinearConfig(input_size, output_size, stride, proj.data_ptr(), proj_type, hidden_type)
        linear = cpuinfer_ext.linear.Linear(config)
        projs.append(proj)
        linears.append(linear)

    # validation
    for i in range(validation_iter):
        linear = linears[i % layer_num]
        input = torch.randn((1, input_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, output_size), dtype=torch.float16).contiguous()
        input = input / 100

        CPUInfer.submit(linear.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        # print('cpuinfer output', output)

        proj = projs[i%layer_num]
        t_output = torch.mm(input, proj.t())
        # print('torch output', t_output)

        diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
        print('diff = ', diff)
        assert(diff < 0.001)

    # warm up
    for i in range(warm_up_iter):
        linear = linears[i % layer_num]
        input = torch.randn((1, input_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, output_size), dtype=torch.float16).contiguous()
        input = input / 100
        CPUInfer.submit(linear.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()

    # test
    total_time = 0
    for i in range(test_iter):
        linear = linears[i % layer_num]
        input = torch.randn((1, input_size), dtype=torch.float16).contiguous()
        output = torch.empty((1, output_size), dtype=torch.float16).contiguous()
        input = input / 100
        start = time.perf_counter()
        CPUInfer.submit(linear.forward, input.data_ptr(), output.data_ptr())
        CPUInfer.sync()
        end = time.perf_counter()
        total_time += end - start
    print('Time: ', total_time)
    print('Iteration: ', test_iter) 
    print('Time per iteration: ', total_time / test_iter)
    print('Bandwidth: ', input_size * output_size * 2 * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
    print("All tasks completed.")