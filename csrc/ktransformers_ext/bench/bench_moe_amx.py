#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2025-04-25 18:28:12
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2025-04-25 18:28:12
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch

expert_num = 8
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
n_routed_experts = 8
layer_num = 10
qlen = 1024
CPUInfer = cpuinfer_ext.CPUInfer(65)
warm_up_iter = 100
test_iter = 100

def bench_moe(quant_mode: str):
    with torch.inference_mode(mode=True):
        if quant_mode == "bf16":
            bytes_per_elem = 2.000000
        elif quant_mode == "int8":
            bytes_per_elem = 1.000000
        else:
            assert(False)


        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.moe.AMX_MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size, max_len, gate_proj.data_ptr(), up_proj.data_ptr(), down_proj.data_ptr())
            if quant_mode == "bf16":
                moe = cpuinfer_ext.moe.AMXBF16_MOE(config)
                CPUInfer.submit(moe.load_weights())
                CPUInfer.sync()
            elif quant_mode == "int8":
                moe = cpuinfer_ext.moe.AMXInt8_MOE(config)
                CPUInfer.submit(moe.load_weights())
                CPUInfer.sync()
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)
        expert_ids = torch.stack([torch.stack([torch.randperm(expert_num, dtype=torch.int64, device = "cuda")[:n_routed_experts] for _ in range(qlen)]) for _ in range(layer_num)]).to("cpu").contiguous()
        weights = torch.rand((layer_num, qlen, n_routed_experts), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
        input = torch.randn((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
        output = torch.empty((layer_num, qlen, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
        qlen_tensor = torch.tensor([qlen], dtype=torch.int32)

        # warm up
        for i in range(warm_up_iter):
            CPUInfer.submit(
                moes[i % layer_num].forward( 
                    qlen, 
                    n_routed_experts, 
                    expert_ids[i % layer_num].data_ptr(), 
                    weights[i % layer_num].data_ptr(),
                    input[i % layer_num].data_ptr(), 
                    output[i % layer_num].data_ptr(),
                    qlen_tensor.data_ptr()
                )
            )
            CPUInfer.sync()

        # test
        start = time.perf_counter()
        for i in range(test_iter):
            CPUInfer.submit(
                moes[i % layer_num].forward( 
                    qlen, 
                    n_routed_experts, 
                    expert_ids[i % layer_num].data_ptr(), 
                    weights[i % layer_num].data_ptr(),
                    input[i % layer_num].data_ptr(), 
                    output[i % layer_num].data_ptr(),
                    qlen_tensor.data_ptr()
                )
            )
            CPUInfer.sync()
        end = time.perf_counter()
        total_time = end - start
        print('Quant mode: ', quant_mode)
        print('Time(s): ', total_time)
        print('Iteration: ', test_iter) 
        print('Time(us) per iteration: ', total_time / test_iter * 1000000)
        print('Bandwidth: ', hidden_size * intermediate_size * 3 * n_routed_experts * bytes_per_elem * test_iter / total_time / 1000 / 1000 / 1000, 'GB/s')
        print('Flops: ', hidden_size * intermediate_size * qlen * 3 * n_routed_experts * 2 * test_iter / total_time / 1000 / 1000 / 1000, 'GFLOPS')
        print('')

bench_moe("bf16")
bench_moe("int8")
