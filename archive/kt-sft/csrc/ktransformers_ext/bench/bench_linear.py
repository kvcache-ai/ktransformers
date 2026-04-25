#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:31:59
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:35:35
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

input_size = 16384
output_size = 5120
stride = 16
group_max_len = 1024
layer_num = 10
qlen = 1
CPUInfer = cpuinfer_ext.CPUInfer(64)
warm_up_iter = 1000
test_iter = 10000

def bench_linear(quant_mode: str):
    with torch.inference_mode(mode=True):

        hidden_type = 30 # ggml_type::GGML_TYPE_BF16
        if quant_mode == "fp32":
            proj_type = 0 # ggml_type::GGML_TYPE_F32
            bytes_per_elem = 4.000000
        elif quant_mode == "fp16":
            proj_type = 1 # ggml_type::GGML_TYPE_F16
            bytes_per_elem = 2.000000
        elif quant_mode == "bf16":
            proj_type = 30 # ggml_type::GGML_TYPE_BF16
            bytes_per_elem = 2.000000
        elif quant_mode == "q8_0":
            proj_type = 8 # ggml_type::GGML_TYPE_Q8_0
            bytes_per_elem = 1.062500
        elif quant_mode == "q6_k":
            proj_type = 14 # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.820312
        elif quant_mode == "q5_k_m":
            proj_type = 13 # ggml_type::GGML_TYPE_Q5_K
            bytes_per_elem = 0.687500
        elif quant_mode == "q4_k_m":
            proj_type = 12 # ggml_type::GGML_TYPE_Q4_K
            bytes_per_elem = 0.562500
        elif quant_mode == "q3_k_m":
            proj_type = 11 # ggml_type::GGML_TYPE_Q3_K
            bytes_per_elem = 0.429688
        elif quant_mode == "q2_k":
            proj_type = 10 # ggml_type::GGML_TYPE_Q2_K
            bytes_per_elem = 0.328125
        elif quant_mode == "iq3_xs":
            proj_type = 21 # ggml_type::GGML_TYPE_IQ3_S
            bytes_per_elem = 0.429688
        elif quant_mode == "iq2_xxs":
            proj_type = 16 # ggml_type::GGML_TYPE_IQ2_XXS
            bytes_per_elem = 0.257812
        else:
            assert(False)

        linears = []
        projs = []
        for _ in range(layer_num):
            proj = torch.randn((output_size, input_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.linear.LinearConfig(input_size, output_size, stride, group_max_len, proj.data_ptr(), proj_type, hidden_type)
            linear = cpuinfer_ext.linear.Linear(config)
            projs.append(proj)
            linears.append(linear)
        input = torch.randn((layer_num, qlen, input_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
        output = torch.empty((layer_num, qlen, output_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()

        # warm up
        for i in range(warm_up_iter):
            CPUInfer.submit(
                linears[i % layer_num].forward(
                    qlen, 
                    input[i % layer_num].data_ptr(), 
                    output[i % layer_num].data_ptr()
                )
            )
            CPUInfer.sync()

        # test
        start = time.perf_counter()
        for i in range(test_iter):
            CPUInfer.submit(
                linears[i % layer_num].forward(
                    qlen, 
                    input[i % layer_num].data_ptr(), 
                    output[i % layer_num].data_ptr()
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter) 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', input_size * output_size * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

bench_linear("fp32")
bench_linear("fp16")
bench_linear("bf16")
bench_linear("q8_0")
bench_linear("q6_k")
bench_linear("q5_k_m")
bench_linear("q4_k_m")
bench_linear("q3_k_m")
bench_linear("q2_k")
# Not supported on __x86_64__
# bench_linear("iq3_xs")
# bench_linear("iq2_xxs")
