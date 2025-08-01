#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-06 10:38:05
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.insert(0,os.path.dirname(__file__) + '/../build')

import cpuinfer_ext
import torch

expert_num = 16
hidden_size = 7168
intermediate_size = 2048
max_len = 25600 
n_routed_experts = 8
qlen = 80
# qlen = 640
layer_num = 1
CPUInfer = cpuinfer_ext.CPUInfer(72)
# validation_iter = 10000
validation_iter = 10

def act_fn(x):
    return x / (1.0 + torch.exp(-x))

def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret

def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )
    return t_output

def test_moe(quant_mode: str):
    assert(quant_mode == "bf16" or quant_mode == "int8" or quant_mode=="int4")
    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
            up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
            down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device = "cuda").to("cpu").contiguous()
            config = cpuinfer_ext.moe.MOEConfig(expert_num, n_routed_experts, hidden_size, intermediate_size)
            config.max_len = max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.pool = CPUInfer.backend_
            config.save = False
            config.load = False
            if quant_mode == "bf16":
                moe = cpuinfer_ext.moe.AMXBF16_MOE(config)
                CPUInfer.submit(moe.load_weights())
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up())
                CPUInfer.sync()
            elif quant_mode == "int8":
                moe = cpuinfer_ext.moe.AMXInt8_MOE(config)
                CPUInfer.submit(moe.load_weights())
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up())
                CPUInfer.sync()
            elif quant_mode == "int4":
                moe = cpuinfer_ext.moe.AMXInt4_MOE(config)
                CPUInfer.submit(moe.load_weights())
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up())
                CPUInfer.sync()
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        # validation
        for i in range(validation_iter):
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack([torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
            weights = torch.rand((qlen, n_routed_experts), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            input = input / 100
            moe = moes[i % layer_num]
            # print('expert ids:',expert_ids)
            CPUInfer.submit(
                moe.forward( 
                    bsz_tensor.data_ptr(),
                    n_routed_experts, 
                    expert_ids.data_ptr(), 
                    weights.data_ptr(), 
                    input.data_ptr(), 
                    output.data_ptr(),
                    False
                )
            )
            CPUInfer.sync()
            # print('cpuinfer output', output)

            gate_proj = gate_projs[i%layer_num]
            up_proj = up_projs[i%layer_num]
            down_proj = down_projs[i%layer_num]
            t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj)
            # print('torch output', t_output)

            # print(output - t_output)
            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
            print('diff = ', diff)
            if quant_mode=="int4":
                assert(diff < 0.35)
            else:
                assert(diff < 0.05)


test_moe("bf16")
test_moe("int8")
test_moe("int4")
