#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-16 10:43:18
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:36:04
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

hidden_size = 5120
intermediate_size = 3072
stride = 16
group_max_len = 1024
layer_num = 10
qlen = 1
CPUInfer = cpuinfer_ext.CPUInfer(64)
warm_up_iter = 1000
test_iter = 10000

def bench_mlp(quant_mode: str):
    with torch.inference_mode(mode=True):

        hidden_type = 30 # ggml_type::GGML_TYPE_BF16
        if quant_mode == "fp32":
            gate_type = 0 # ggml_type::GGML_TYPE_F32
            up_type = 0 # ggml_type::GGML_TYPE_F32
            down_type = 0 # ggml_type::GGML_TYPE_F32
            bytes_per_elem = 4.000000
        elif quant_mode == "fp16":
            gate_type = 1 # ggml_type::GGML_TYPE_F16
            up_type = 1 # ggml_type::GGML_TYPE_F16
            down_type = 1 # ggml_type::GGML_TYPE_F16
            bytes_per_elem = 2.000000
        elif quant_mode == "bf16":
            gate_type = 30 # ggml_type::GGML_TYPE_BF16
            up_type = 30 # ggml_type::GGML_TYPE_BF16
            down_type = 30 # ggml_type::GGML_TYPE_BF16
            bytes_per_elem = 2.000000
        elif quant_mode == "q8_0":
            gate_type = 8 # ggml_type::GGML_TYPE_Q8_0
            up_type = 8 # ggml_type::GGML_TYPE_Q8_0
            down_type = 8 # ggml_type::GGML_TYPE_Q8_0
            bytes_per_elem = 1.062500
        elif quant_mode == "q6_k":
            gate_type = 14 # ggml_type::GGML_TYPE_Q6_K
            up_type = 14 # ggml_type::GGML_TYPE_Q6_K
            down_type = 14 # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.820312
        elif quant_mode == "q5_k_m":
            gate_type = 13 # ggml_type::GGML_TYPE_Q5_K
            up_type = 13 # ggml_type::GGML_TYPE_Q5_K
            down_type = 14 # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.731771
        elif quant_mode == "q4_k_m":
            gate_type = 12 # ggml_type::GGML_TYPE_Q4_K
            up_type = 12 # ggml_type::GGML_TYPE_Q4_K
            down_type = 14 # ggml_type::GGML_TYPE_Q6_K
            bytes_per_elem = 0.648437
        elif quant_mode == "q3_k_m":
            gate_type = 11 # ggml_type::GGML_TYPE_Q3_K
            up_type = 11 # ggml_type::GGML_TYPE_Q3_K
            down_type = 13 # ggml_type::GGML_TYPE_Q5_K
            bytes_per_elem = 0.515625
        elif quant_mode == "q2_k":
            gate_type = 10 # ggml_type::GGML_TYPE_Q2_K
            up_type = 10 # ggml_type::GGML_TYPE_Q2_K
            down_type = 11 # ggml_type::GGML_TYPE_Q3_K
            bytes_per_elem = 0.328125
        elif quant_mode == "iq3_xs":
            gate_type = 21 # ggml_type::GGML_TYPE_IQ3_S
            up_type = 21 # ggml_type::GGML_TYPE_IQ3_S
            down_type = 21 # ggml_type::GGML_TYPE_IQ3_S
            bytes_per_elem = 0.429688
        elif quant_mode == "iq2_xxs":
            gate_type = 16 # ggml_type::GGML_TYPE_IQ2_XXS
            up_type = 16 # ggml_type::GGML_TYPE_IQ2_XXS
            down_type = 16 # ggml_type::GGML_TYPE_IQ2_XXS
            bytes_per_elem = 0.257812
        else:
            assert(False)


        mlps = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((hidden_size, intermediate_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.mlp.MLPConfig(hidden_size, intermediate_size, stride, group_max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr(), gate_type, up_type, down_type, hidden_type)
            mlp = cpuinfer_ext.mlp.MLP(config)
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            mlps.append(mlp)
        input = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
        output = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()

        # warm up
        for i in range(warm_up_iter):
            CPUInfer.submit(
                mlps[i % layer_num].forward( 
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
                mlps[i % layer_num].forward( 
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
        print('Bandwidth: ', hidden_size * intermediate_size * 3 * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('')

bench_mlp("fp32")
bench_mlp("fp16")
bench_mlp("bf16")
bench_mlp("q8_0")
bench_mlp("q6_k")
bench_mlp("q5_k_m")
bench_mlp("q4_k_m")
bench_mlp("q3_k_m")
bench_mlp("q2_k")
# Not supported on __x86_64__
# bench_linear("iq3_xs")
# bench_linear("iq2_xxs")
