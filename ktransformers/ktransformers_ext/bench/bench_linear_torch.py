#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:31:59
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-07-25 10:32:48
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
import torch
import torch.nn.quantized as nnq

def bench_linear(quant_mode: str):
    with torch.inference_mode(mode=True):
        input_size = 16384
        output_size = 5120
        layer_num = 10
        warm_up_iter = 1000
        test_iter = 10000

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

        projs = []
        for _ in range(layer_num):
            proj = torch.randn((output_size, input_size), dtype = torch.float32, device = "cuda").to("cpu").contiguous()
            if quant_mode == "qint8":
                scale, zero_point = 0.1, 0  # Adjust scale and zero_point based on your dataset
                proj_q = torch.quantize_per_tensor(proj, scale, zero_point, torch.qint8)
                quantized_layer = nnq.Linear(input_size, output_size)
                quantized_layer.set_weight_bias(proj_q, None)
                projs.append(quantized_layer)
            else:
                projs.append(proj.to(proj_type))

        # warm up
        for i in range(warm_up_iter):
            input = torch.randn((1, input_size), dtype=torch.float32).contiguous()
            if quant_mode == "qint8":
                input_q = torch.quantize_per_tensor(input, scale, zero_point, torch.quint8)
                quantized_layer = projs[i % layer_num]
                t_output = quantized_layer(input_q)
            else:
                t_output = torch.mm(input.to(proj_type), projs[i % layer_num].t())

        # test
        total_time = 0
        for i in range(test_iter):
            input = torch.randn((1, input_size), dtype=torch.float32).contiguous()
            start = time.perf_counter()
            if quant_mode == "qint8":
                input_q = torch.quantize_per_tensor(input, scale, zero_point, torch.quint8)
                quantized_layer = projs[i % layer_num]
                t_output = quantized_layer(input_q)
            else:
                t_output = torch.mm(input.to(proj_type), projs[i % layer_num].t())
            end = time.perf_counter()
            total_time += end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter) 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', input_size * output_size * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

bench_linear("fp32")
bench_linear("fp16")
bench_linear("bf16")
bench_linear("qint8")
