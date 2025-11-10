#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:36:59
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

input_size = 16384
output_size = 5120
stride = 32
group_max_len = 1024
proj_type = 1 # ggml_type::GGML_TYPE_F16
hidden_type = 1 # ggml_type::GGML_TYPE_F16
qlen = 30
layer_num = 10
CPUInfer = cpuinfer_ext.CPUInfer(48)
validation_iter = 100

with torch.inference_mode(mode=True):
    linears = []
    projs = []
    for _ in range(layer_num):
        proj = torch.randn((output_size, input_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        config = cpuinfer_ext.linear.LinearConfig(input_size, output_size, stride, group_max_len, proj.data_ptr(), proj_type, hidden_type)
        linear = cpuinfer_ext.linear.Linear(config)
        projs.append(proj)
        linears.append(linear)

    # validation
    for i in range(validation_iter):
        linear = linears[i % layer_num]
        input = torch.randn((qlen, input_size), dtype=torch.float16).contiguous()
        output = torch.empty((qlen, output_size), dtype=torch.float16).contiguous()
        input = input / 100

        CPUInfer.submit(
            linear.forward(
                qlen,
                input.data_ptr(),
                output.data_ptr()
            )
        )
        CPUInfer.sync()
        # print('cpuinfer output', output)

        proj = projs[i%layer_num]
        t_output = torch.mm(input, proj.t())
        # print('torch output', t_output)

        diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
        print('diff = ', diff)
        assert(diff < 0.001)
