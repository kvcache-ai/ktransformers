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
warm_up_iter = 1000
test_iter = 10000


def bench_linear(cache_seqlen: int, device):
    with torch.inference_mode(mode=True):

        kvcaches = []

        for layer_idx in range(layer_num):
            k_cache = torch.randn(
                (1, 32, cache_seqlen, head_dim),
                dtype=torch.float16,
                device=device,
            ).contiguous()
            v_cache = torch.randn(
                (1, 32, cache_seqlen, head_dim),
                dtype=torch.float16,
                device=device,
            ).contiguous()

            kvcaches.append((k_cache, v_cache))

        input = torch.randn(
            (1, q_head_num, 1, head_dim), dtype=torch.float16, device=device
        ).contiguous()
        input = input / 100

        # warm up
        for i in range(warm_up_iter):
            k_cache = kvcaches[i % layer_num][0]
            v_cache = kvcaches[i % layer_num][1]
            torch.nn.functional.scaled_dot_product_attention(input, k_cache, v_cache)

        # test
        start = time.perf_counter()
        for i in range(test_iter):
            k_cache = kvcaches[i % layer_num][0]
            v_cache = kvcaches[i % layer_num][1]
            torch.nn.functional.scaled_dot_product_attention(input, k_cache, v_cache)
        end = time.perf_counter()
        total_time = end - start
        print("cache sequence length: ", cache_seqlen)
        print("Time(s): ", total_time)
        print("Iteration: ", test_iter)
        print("Time(us) per iteration: ", total_time / test_iter * 1000000)
        print(
            "Bandwidth: ",
            cache_seqlen
            * q_head_num
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


bench_linear(1024, "cpu")
bench_linear(4096, "cpu")
bench_linear(1024, "cuda")
bench_linear(4096, "cuda")
bench_linear(16384, "cuda")
bench_linear(32768, "cuda")
bench_linear(65536, "cuda")
