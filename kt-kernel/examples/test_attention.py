#!/usr/bin/env python
# coding=utf-8
"""
Description  :  
Author       : Jianwei Dong
Date         : 2024-08-28 10:32:05
Version      : 1.0.0
LastEditors  : chenht2022 
LastEditTime : 2024-08-28 10:32:05
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""
import os, sys
import time

sys.path.append(os.path.dirname(__file__) + "/../build")
import cpuinfer_ext
from flash_attn import flash_attn_with_kvcache
import torch

layer_num = 10
kv_head_num = 8
q_head_num = 32
head_dim = 128
block_len = 128
anchor_num = 1
cache_seqlen = 8192
cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device="cpu")
seqlens_zero = torch.zeros((1,), dtype=torch.int32, device="cpu")
anchor_type = cpuinfer_ext.kvcache.AnchorType.DYNAMIC
kv_type = cpuinfer_ext.kvcache.ggml_type.FP16
retrieval_type = cpuinfer_ext.kvcache.RetrievalType.LAYER
layer_step: int = 1
token_step: int = 1
layer_offset: int = 0
max_thread_num: int = 2
max_batch_size: int = 1
max_block_num: int = 512
CPUInfer = cpuinfer_ext.CPUInfer(max_thread_num)
validation_iter = 100

with torch.inference_mode(mode=True):
    config = cpuinfer_ext.kvcache.KVCacheConfig(
        layer_num,
        kv_head_num,
        q_head_num,
        head_dim,
        block_len,
        anchor_num,
        anchor_type,
        kv_type,
        retrieval_type,
        layer_step,
        token_step,
        layer_offset,
        max_block_num,
        max_batch_size,
        max_thread_num,
    )
    local_kvcache = cpuinfer_ext.kvcache.KVCache(config)

    kvcaches = []
    block_table = (
        torch.arange(max_block_num, dtype=torch.int32, device="cpu")
        .contiguous()
        .view(1, -1)
    )

    for layer_idx in range(layer_num):
        k_cache = torch.randn(
            (1, cache_seqlen, kv_head_num, head_dim), dtype=torch.float16, device="cpu"
        ).contiguous()
        v_cache = torch.randn(
            (1, cache_seqlen, kv_head_num, head_dim), dtype=torch.float16, device="cpu"
        ).contiguous()

        CPUInfer.submit(
            local_kvcache.update_kvcache_fp16(
                k_cache.data_ptr(),
                v_cache.data_ptr(),
                layer_idx,
                block_table.data_ptr(),
                1,
                max_block_num,
                seqlens_zero.data_ptr(),
                cache_seqlen,
            )
        )
        CPUInfer.sync()

        kvcaches.append((k_cache.to("cuda"), v_cache.to("cuda")))

    # validation
    for i in range(validation_iter):

        k_cache = kvcaches[i % layer_num][0]
        v_cache = kvcaches[i % layer_num][1]
        input = torch.randn(
            (1, 1, q_head_num, head_dim), dtype=torch.float16, device="cpu"
        ).contiguous()
        output = torch.empty(
            (1, 1, q_head_num, head_dim), dtype=torch.float16, device="cpu"
        ).contiguous()

        # attn_lse: (bsz, q_len, q_head_num)
        attn_lse = torch.empty(
            (1, 1, q_head_num), dtype=torch.float32, device="cpu"
        ).contiguous()
        input = input / 100

        CPUInfer.submit(
            local_kvcache.attn(
                input.data_ptr(),
                output.data_ptr(),
                attn_lse.data_ptr(),
                i % layer_num,
                0,
                1,
                1,
                max_block_num,
                block_table.data_ptr(),
                cache_seqlens.data_ptr(),
                -1,
                -1,
                -1,
            )
        )
        CPUInfer.sync()
        # print("cpuinfer output", output)

        t_output = flash_attn_with_kvcache(
            q=input.to("cuda"),
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens.to("cuda"),
        )
        # print("torch output", t_output)

        diff = torch.mean(torch.abs(output.to("cuda") - t_output)) / torch.mean(
            torch.abs(t_output)
        )
        print("diff = ", diff)
        assert diff < 0.001
