#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:37:28
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

hidden_size = 5120
intermediate_size = 3072
stride = 32
group_max_len = 1024
gate_type = 1 # ggml_type::GGML_TYPE_F16
up_type = 1 # ggml_type::GGML_TYPE_F16
down_type = 1 # ggml_type::GGML_TYPE_F16
hidden_type = 1 # ggml_type::GGML_TYPE_F16
qlen = 30
layer_num = 10
CPUInfer = cpuinfer_ext.CPUInfer(48)
validation_iter = 100

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret

with torch.inference_mode(mode=True):
    mlps = []
    gate_projs = []
    up_projs = []
    down_projs = []
    for _ in range(layer_num):
        gate_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        up_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        down_proj = torch.randn((hidden_size, intermediate_size), dtype=torch.float16, device = "cuda").to("cpu").contiguous()
        config = cpuinfer_ext.mlp.MLPConfig(hidden_size, intermediate_size, stride, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
        mlp = cpuinfer_ext.mlp.MLP(config)
        gate_projs.append(gate_proj)
        up_projs.append(up_proj)
        down_projs.append(down_proj)
        mlps.append(mlp)

    # validation
    for i in range(validation_iter):
        mlp = mlps[i % layer_num]
        input = torch.randn((qlen, hidden_size), dtype=torch.float16).contiguous()
        output = torch.empty((qlen, hidden_size), dtype=torch.float16).contiguous()
        input = input / 100

        CPUInfer.submit(
            mlp.forward(
                qlen,
                input.data_ptr(), 
                output.data_ptr()
            )
        )
        CPUInfer.sync()
        # print('cpuinfer output', output)

        gate_proj = gate_projs[i%layer_num]
        up_proj = up_projs[i%layer_num]
        down_proj = down_projs[i%layer_num]
        t_output = mlp_torch(input, gate_proj, up_proj, down_proj)
        # print('torch output', t_output)

        diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
        print('diff = ', diff)
        assert(diff < 0.001)
