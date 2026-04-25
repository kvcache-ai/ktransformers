#!/usr/bin/env python
# coding=utf-8
"""
Description  :  
Author       : Jianwei Dong
Date         : 2024-08-28 10:32:05
Version      : 1.0.0
LastEditors  : Jianwei Dong 
LastEditTime : 2024-08-28 10:32:05
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""
import os, sys
import time

sys.path.append(os.path.dirname(__file__) + "/../build")
import cpuinfer_ext
import torch

layer_num = 10
kv_head_num = 8
q_head_num = 32
head_dim = 128
block_len = 128
anchor_num = 1

anchor_type = cpuinfer_ext.kvcache.AnchorType.DYNAMIC
kv_type = cpuinfer_ext.kvcache.ggml_type.FP16
retrieval_type = cpuinfer_ext.kvcache.RetrievalType.LAYER
layer_step: int = 1
token_step: int = 1
layer_offset: int = 0
max_thread_num: int = 64
max_batch_size: int = 1
max_block_num: int = 1024
CPUInfer = cpuinfer_ext.CPUInfer(max_thread_num)

warm_up_iter = 1000
test_iter = 10000


def bench_linear(cache_seqlen: int):
    with torch.inference_mode(mode=True):
        cache_seqlens = torch.tensor([cache_seqlen], dtype=torch.int32, device="cpu")
        seqlens_zero = torch.zeros((1,), dtype=torch.int32, device="cpu")

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
        block_table = (
            torch.arange(max_block_num, dtype=torch.int32, device="cpu")
            .contiguous()
            .view(1, -1)
        )

        for layer_idx in range(layer_num):
            k_cache = torch.randn(
                (1, cache_seqlen, kv_head_num, head_dim),
                dtype=torch.float16,
                device="cpu",
            ).contiguous()
            v_cache = torch.randn(
                (1, cache_seqlen, kv_head_num, head_dim),
                dtype=torch.float16,
                device="cpu",
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

        # warm up
        for i in range(warm_up_iter):
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

        # test
        start = time.perf_counter()
        for i in range(test_iter):
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
        end = time.perf_counter()
        total_time = end - start
        print("cache sequence length: ", cache_seqlen)
        print("Time(s): ", total_time)
        print("Iteration: ", test_iter)
        print("Time(us) per iteration: ", total_time / test_iter * 1000000)
        print(
            "Bandwidth: ",
            cache_seqlen
            * kv_head_num
            * head_dim
            * 2
            * 2
            * test_iter
            / total_time
            / 1000
            / 1000
            / 1000,
            "GB/s",
        )
        print("")


bench_linear(1024)
bench_linear(4096)
bench_linear(16384)
bench_linear(32768)
bench_linear(65536)
