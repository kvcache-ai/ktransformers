#!/usr/bin/env python
# coding=utf-8
'''
Description  :  
Author       : chenht2022
Date         : 2024-07-25 10:32:05
Version      : 1.0.0
LastEditors  : SkqLiao 
LastEditTime : 2025-03-13 11:38:05
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import os, sys
import time
sys.path.insert(0, os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
from tqdm import tqdm
from cpuinfer_ext.kvcache import ggml_type

torch.manual_seed(0)

expert_num = 8
hidden_size = 2048 #7168
intermediate_size = 2048
stride = 32
group_min_len = 10
group_max_len = 2560
num_experts_per_tok = 8
layer_num = 1
# expert_num = 8
# hidden_size = 7168
# intermediate_size = 2048
# stride = 32
# group_min_len = 10
# group_max_len = 10240
# num_experts_per_tok = 8
# qlen = 1024
# layer_num = 1
CPUInfer = cpuinfer_ext.CPUInfer(64)
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


def to_cpuinfer_tensor(tensor, type):
    size = torch.prod(torch.tensor(tensor.shape, dtype=torch.int32)).item()
    return cpuinfer_ext.utils.from_float(tensor.data_ptr(), size, type)

def from_cpuinfer_tensor(tensor, size, type):
    return cpuinfer_ext.utils.to_float(tensor.data_ptr(), size, type)

qlens = [1,64] #[64, 512, 2048, 8192, 16384]
# gate_types = [ggml_type.FP32, ggml_type.FP16, ggml_type.Q8_0, ggml_type.Q6_K, ggml_type.Q5_K, ggml_type.Q4_K, ggml_type.Q3_K]
# up_types = [ggml_type.FP32, ggml_type.FP16, ggml_type.Q8_0, ggml_type.Q6_K, ggml_type.Q5_K, ggml_type.Q4_K, ggml_type.Q3_K]
# down_types = [ggml_type.FP32, ggml_type.FP16, ggml_type.Q8_0, ggml_type.Q6_K, ggml_type.Q6_K, ggml_type.Q6_K, ggml_type.Q5_K]
gate_types = [ggml_type.Q4_K]
up_types = [ggml_type.Q4_K]
down_types = [ggml_type.Q6_K]
hidden_type = ggml_type.BF16
print(f'Parameters: expert_num: {expert_num} hidden_size: {hidden_size} intermediate_size: {intermediate_size}')
print(f'group_max_len: ', group_max_len)

for qlen in qlens:
    for gate_type, up_type, down_type in zip(gate_types, up_types, down_types):
        with torch.inference_mode(mode=True):
            moes = []
            gate_projs = []
            up_projs = []
            down_projs = []
            print('Preparing data...')
            converted_tensors = []
            for _ in range(layer_num):
                size = expert_num * intermediate_size * hidden_size
                gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
                up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
                down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.float32, device = "cuda").to("cpu").contiguous()
                
                gate_tensor = to_cpuinfer_tensor(gate_proj, gate_type)
                up_tensor = to_cpuinfer_tensor(up_proj, up_type)
                down_tensor = to_cpuinfer_tensor(down_proj, down_type)
                
                config = cpuinfer_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
                config.pool = CPUInfer.backend_
                config.stride = stride
                config.group_min_len = group_min_len
                config.group_max_len = group_max_len
                config.gate_proj = gate_tensor.data_ptr()
                config.up_proj = up_tensor.data_ptr()
                config.down_proj = down_tensor.data_ptr()
                config.gate_type = gate_type
                config.up_type = up_type
                config.down_type = down_type
                config.hidden_type = hidden_type


                moe = cpuinfer_ext.moe.MOE(config)
                gate_projs.append(gate_proj)
                up_projs.append(up_proj)
                down_projs.append(down_proj)    
                CPUInfer.submit(moe.load_weights_task())
                CPUInfer.sync()
                moes.append(moe)
                converted_tensors.append((gate_tensor, up_tensor, down_tensor))
            print('Finished initialization!')

            CPUInfer.submit(moes[0].warm_up_task())
            CPUInfer.sync()
            print('Warm up finished!')

            # validation
            progress_bar = tqdm(range(validation_iter), desc="Starting")
            total_diff = 0
            
            for i in tqdm(progress_bar):
                progress_bar.set_description('Round: {}/{}'.format(i + 1, validation_iter))
                expert_ids = torch.stack([torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
                weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
                input_proj = torch.randn((qlen, hidden_size), dtype=torch.float32).contiguous() / 100
                output_proj = torch.empty((qlen, hidden_size), dtype=torch.float32).contiguous()
                
                input_tensor = to_cpuinfer_tensor(input_proj, hidden_type)
                output_tensor = to_cpuinfer_tensor(output_proj, hidden_type)
                
                qlen_tensor = torch.tensor([qlen], dtype=torch.int32)
                moe = moes[i % layer_num]
                CPUInfer.submit(
                    moe.forward_task( 
                        qlen_tensor.data_ptr(),
                        num_experts_per_tok, 
                        expert_ids.data_ptr(), 
                        weights.data_ptr(), 
                        input_tensor.data_ptr(), 
                        output_tensor.data_ptr(),
                    )
                )
                CPUInfer.sync()
                cpu_output = from_cpuinfer_tensor(output_tensor, qlen * hidden_size, hidden_type)

                gate_proj = gate_projs[i%layer_num]
                up_proj = up_projs[i%layer_num]
                down_proj = down_projs[i%layer_num]
                t_output = moe_torch(input_proj, expert_ids, weights, gate_proj, up_proj, down_proj)
                print('cpuinfer output', cpu_output)
                print('torch output', t_output)
                diff = torch.mean(torch.abs(cpu_output.flatten() - t_output.flatten())) / torch.mean(torch.abs(t_output.flatten()))
                assert diff < 0.5
                total_diff += diff
                
            print(f'gate_type: {gate_type}, up_type: {up_type}, down_type: {down_type}')
            print(f'Average diff: {total_diff / validation_iter:.4f}')
