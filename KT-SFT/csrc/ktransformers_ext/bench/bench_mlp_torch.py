#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-16 10:43:18
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:32:53
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
import torch
import torch.nn.quantized as nnq

scale, zero_point = 0.1, 0  # Adjust scale and zero_point based on your dataset

hidden_size = 5120
intermediate_size = 3072
layer_num = 10
qlen = 1
warm_up_iter = 1000
test_iter = 10000

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def mlp_torch(input, gate_proj, up_proj, down_proj):
    if isinstance(gate_proj, nnq.Linear):
        input_q = torch.quantize_per_tensor(input.to(torch.float32), scale, zero_point, torch.quint8)
        gate_buf = gate_proj(input_q)
        up_buf = up_proj(input_q)
        gate_buf = gate_buf.dequantize()
        up_buf = up_buf.dequantize()
        intermediate = act_fn(gate_buf) * up_buf
        intermediate_q = torch.quantize_per_tensor(intermediate, scale, zero_point, torch.quint8)
        expert_output = down_proj(intermediate_q)
        ret = expert_output.dequantize()
    else:
        gate_buf = torch.mm(input.to(gate_proj.dtype), gate_proj.t())
        up_buf = torch.mm(input.to(up_proj.dtype), up_proj.t())
        intermediate = act_fn(gate_buf) * up_buf
        ret = torch.mm(intermediate.to(down_proj.dtype), down_proj.t())
    return ret

def bench_mlp(quant_mode: str):
    with torch.inference_mode(mode=True):
        if quant_mode == "fp32":
            proj_type = torch.float32
            bytes_per_elem = 4.000000
        elif quant_mode == "fp16":
            proj_type = torch.float16
            bytes_per_elem = 2.000000
        elif quant_mode == "bf16":
            proj_type = torch.bfloat16
            bytes_per_elem = 2.000000
        elif quant_mode == "qint8":
            proj_type = torch.qint8
            bytes_per_elem = 1.000000
        else:
            assert(False)

        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((hidden_size, intermediate_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            if quant_mode == "qint8":
                gate_proj_q = torch.quantize_per_tensor(gate_proj, scale, zero_point, torch.qint8)
                quantized_gate = nnq.Linear(hidden_size, intermediate_size)
                quantized_gate.set_weight_bias(gate_proj_q, None)
                up_proj_q = torch.quantize_per_tensor(up_proj, scale, zero_point, torch.qint8)
                quantized_up = nnq.Linear(hidden_size, intermediate_size)
                quantized_up.set_weight_bias(up_proj_q, None)
                down_proj_q = torch.quantize_per_tensor(down_proj, scale, zero_point, torch.qint8)
                quantized_down = nnq.Linear(intermediate_size, hidden_size)
                quantized_down.set_weight_bias(down_proj_q, None)
                gate_projs.append(quantized_gate)
                up_projs.append(quantized_up)
                down_projs.append(quantized_down)
            else:
                gate_projs.append(gate_proj.to(proj_type))
                up_projs.append(up_proj.to(proj_type))
                down_projs.append(down_proj.to(proj_type))
        input = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()

        # warm up
        for i in range(warm_up_iter):
            mlp_torch(input[i % layer_num], gate_projs[i % layer_num], up_projs[i % layer_num], down_projs[i % layer_num])

        # test
        start = time.perf_counter()
        for i in range(test_iter):
            mlp_torch(input[i % layer_num], gate_projs[i % layer_num], up_projs[i % layer_num], down_projs[i % layer_num])
        end = time.perf_counter()
        total_time = end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter) 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', hidden_size * intermediate_size * 3 * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

bench_mlp("fp32")
bench_mlp("fp16")
bench_mlp("bf16")
bench_mlp("qint8")
