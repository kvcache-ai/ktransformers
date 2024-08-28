#!/usr/bin/env python
# coding=utf-8
"""
Description  : This script defines the `CPUInferKVCache` and `CPUInfer` classes for performing inference 
               with a Key-Value Cache on the CPU. The `CPUInferKVCache` class is responsible for configuring 
               and managing key-value caches, updating and retrieving cache data, and handling attention 
               operations. It supports different cache types (e.g., Q4_0, FP16) and retrieval strategies 
               (e.g., shared, separate). The `CPUInfer` class handles task submission and synchronization 
               on the CPU, with optional CUDA stream integration for tasks involving GPU acceleration. 
               These classes facilitate efficient caching and memory management for deep learning models 
               that leverage key-value attention mechanisms, particularly on CPU-based systems.
Author       : djw
Date         : 2024-08-26 23:25:24
Version      : 1.0.0
LastEditors  : djw 
LastEditTime : 2024-08-26 23:25:24
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
"""
import sys, os
from typing import Any
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Release"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ktransformers_ext", "build", "Debug"))
import cpuinfer_ext
from ktransformers.server.config.config import Config


class CPUInferKVCache:
    def __init__(
        self,
        layer_num: int = 32,
        kv_head_num: int = 8,
        q_head_num: int = 32,
        head_dim: int = 128,
        block_len: int = 256,
        anchor_num: int = 4,
        anchor_type: str = "FIXED",
        kv_type: str = "Q4_0",
        retrieval_type: str = "SHARED",
        layer_step: int = 1,
        token_step: int = 1,
        layer_offset: int = 0,
        max_thread_num: int = 32,
        max_batch_size: int = 4,
        max_block_num: int = 512,
    ):

        if anchor_type == "FIXED":
            anchor_type = cpuinfer_ext.kvcache.AnchorType.FIXED
        elif anchor_type == "QUEST":
            anchor_type = cpuinfer_ext.kvcache.AnchorType.QUEST
        elif anchor_type == "DYNAMIC":
            anchor_type = cpuinfer_ext.kvcache.AnchorType.DYNAMIC
        elif anchor_type == "BLOCK_MEAN":
            anchor_type = cpuinfer_ext.kvcache.AnchorType.BLOCK_MEAN
        elif anchor_type == "BLOCK_MAX":
            anchor_type = cpuinfer_ext.kvcache.AnchorType.BLOCK_MAX
        else:
            raise ValueError(f"Unknown anchor type: {anchor_type}")

        if kv_type == "FP16":
            kv_type = cpuinfer_ext.kvcache.ggml_type.FP16
        elif kv_type == "FP32":
            assert False, "FP32 is not supported yet."
            kv_type = cpuinfer_ext.kvcache.ggml_type.FP32
        elif kv_type == "Q4_0":
            kv_type = cpuinfer_ext.kvcache.ggml_type.Q4_0
        elif kv_type == "Q8_0":
            kv_type = cpuinfer_ext.kvcache.ggml_type.Q8_0
        else:
            raise ValueError(f"Unknown kv type: {kv_type}")

        if retrieval_type == "SHARED":
            retrieval_type = cpuinfer_ext.kvcache.RetrievalType.LAYER
        elif retrieval_type == "INDIVIDUAL":
            retrieval_type = cpuinfer_ext.kvcache.RetrievalType.QHEAD
        elif retrieval_type == "SEPARATE":
            retrieval_type = cpuinfer_ext.kvcache.RetrievalType.KVHEAD

        self.config = cpuinfer_ext.kvcache.KVCacheConfig(
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
        self.kvcache = cpuinfer_ext.kvcache.KVCache(self.config)

    def load_kvcache(self, tensor_file_path: str):
        if not os.path.exists(tensor_file_path):
            raise FileNotFoundError(f"The file {tensor_file_path} does not exist.")
        return self.kvcache.load_kvcache(tensor_file_path,)

    def dump_kvcache(
        self, block_table: torch.Tensor, cache_total_len: int, tensor_file_path: str
    ):
        assert (
            block_table.dim() == 1
            and block_table.dtype == torch.int
            and block_table.is_contiguous()
            and block_table.device == torch.device("cpu")
        ), "block_table dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            block_table.dim(),
            block_table.size(),
            block_table.dtype,
            block_table.is_contiguous(),
            block_table.device,
        )

        assert (
            cache_total_len > 0
            and cache_total_len <= self.config.block_len * block_table.size(0)
        ), "cache_total_len: {}".format(cache_total_len)

        if not os.path.exists(os.path.dirname(tensor_file_path)):
            os.makedirs(os.path.dirname(tensor_file_path))

        return self.kvcache.dump_kvcache(
            block_table.data_ptr(),
            cache_total_len,
            tensor_file_path,
        )

    def update_cache_total_len(self, cache_total_len: int):
        assert cache_total_len > 0, "cache_total_len: {}".format(cache_total_len)
        self.kvcache.update_cache_total_len(cache_total_len)

    # q_in: (bsz, q_len, q_head_num, head_dim)
    # output: (bsz, q_len, q_head_num, head_dim)
    # attn_lse: (bsz, q_len, q_head_num)
    # block_table: (bsz, max_block_num)
    def attn(
        self,
        q_in: torch.Tensor,
        output: torch.Tensor,
        attn_lse: torch.Tensor,
        layer_idx: int,
        generate_token_idx: int,
        block_table: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
        pick_block_num: int | None = None,
        init_block_num: int | None = None,
        local_block_num: int | None = None,
    ):

        assert (
            q_in.dim() == 4
            and q_in.size(2) == self.config.q_head_num
            and q_in.size(3) == self.config.head_dim
            and q_in.dtype == torch.float16
            and q_in.is_contiguous()
            and q_in.device == torch.device("cpu")
        ), "q_in dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            q_in.dim(), q_in.size(), q_in.dtype, q_in.is_contiguous(), q_in.device
        )

        batch_size = q_in.size(0)
        q_len = q_in.size(1)

        assert (block_table is None) or (
            block_table.dim() == 2
            and block_table.size(0) == batch_size
            and block_table.dtype == torch.int
            and block_table.is_contiguous()
            and block_table.device == torch.device("cpu")
        ), "block_table dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            block_table.dim(),
            block_table.size(),
            block_table.dtype,
            block_table.is_contiguous(),
            block_table.device,
        )

        max_block_num = block_table.size(1) if block_table is not None else 0

        assert (
            output.dim() == 4
            and output.size(0) == batch_size
            and output.size(2) == self.config.q_head_num
            and output.size(1) == q_len
            and output.size(3) == self.config.head_dim
            and output.dtype == torch.float16
            and output.is_contiguous()
            and output.device == torch.device("cpu")
        ), "output dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            output.dim(),
            output.size(),
            output.dtype,
            output.is_contiguous(),
            output.device,
        )

        assert (
            attn_lse.dim() == 3
            and attn_lse.size(0) == batch_size
            and attn_lse.size(1) == q_len
            and attn_lse.size(2) == self.config.q_head_num
            and attn_lse.dtype == torch.float32
            and attn_lse.is_contiguous()
            and attn_lse.device == torch.device("cpu")
        ), "attn_lse dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            attn_lse.dim(),
            attn_lse.size(),
            attn_lse.dtype,
            attn_lse.is_contiguous(),
            attn_lse.device,
        )

        assert (
            layer_idx >= 0 and layer_idx < self.config.layer_num
        ), "layer_idx: {}".format(layer_idx)

        assert (cache_seqlens is None) or (
            cache_seqlens.dim() == 1
            and cache_seqlens.size(0) == batch_size
            and cache_seqlens.dtype == torch.int
            and cache_seqlens.is_contiguous()
            and cache_seqlens.device == torch.device("cpu")
        ), "cache_seqlens dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            cache_seqlens.dim(),
            cache_seqlens.size(),
            cache_seqlens.dtype,
            cache_seqlens.is_contiguous(),
            cache_seqlens.device,
        )

        return self.kvcache.attn(
            q_in.data_ptr(),
            output.data_ptr(),
            attn_lse.data_ptr(),
            layer_idx,
            generate_token_idx,
            q_len,
            batch_size,
            max_block_num,
            block_table.data_ptr() if block_table is not None else 0,
            cache_seqlens.data_ptr() if cache_seqlens is not None else 0,
            pick_block_num,
            init_block_num,
            local_block_num,
        )

    # k_in: (block_len, kv_head_num, head_dim)
    # v_in: (block_len, kv_head_num, head_dim)
    def update_kvcache_one_block_fp16(
        self, k_in: torch.Tensor, v_in: torch.Tensor, layer_id: int, block_idx: int
    ):
        assert (
            k_in.dim() == 3
            and k_in.size(1) == self.config.block_len
            and k_in.size(0) == self.config.kv_head_num
            and k_in.size(2) == self.config.head_dim
            and k_in.dtype == torch.float16
            and k_in.is_contiguous()
            and k_in.device == torch.device("cpu")
        ), "k_in dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            k_in.dim(), k_in.size(), k_in.dtype, k_in.is_contiguous(), k_in.device
        )
        assert (
            v_in.dim() == 3
            and v_in.size(1) == self.config.block_len
            and v_in.size(0) == self.config.kv_head_num
            and v_in.size(2) == self.config.head_dim
            and v_in.dtype == torch.float16
            and v_in.is_contiguous()
            and v_in.device == torch.device("cpu")
        ), "v_in dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            v_in.dim(), v_in.size(), v_in.dtype, v_in.is_contiguous(), v_in.device
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.update_one_block_fp16(
            k_in.data_ptr(),
            v_in.data_ptr(),
            layer_id,
            block_idx,
        )

    def get_kvcache_one_block_fp16(
        self, k_in: torch.Tensor, v_in: torch.Tensor, layer_id: int, block_idx: int
    ):
        assert (
            k_in.dim() == 3
            and k_in.size(1) == self.config.block_len
            and k_in.size(0) == self.config.kv_head_num
            and k_in.size(2) == self.config.head_dim
            and k_in.dtype == torch.float16
            and k_in.is_contiguous()
            and k_in.device == torch.device("cpu")
        ), "k_in dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            k_in.dim(), k_in.size(), k_in.dtype, k_in.is_contiguous(), k_in.device
        )
        assert (
            v_in.dim() == 3
            and v_in.size(1) == self.config.block_len
            and v_in.size(0) == self.config.kv_head_num
            and v_in.size(2) == self.config.head_dim
            and v_in.dtype == torch.float16
            and v_in.is_contiguous()
            and v_in.device == torch.device("cpu")
        ), "v_in dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            v_in.dim(), v_in.size(), v_in.dtype, v_in.is_contiguous(), v_in.device
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.get_one_block_fp16(
            k_in.data_ptr(),
            v_in.data_ptr(),
            layer_id,
            block_idx,
        )

    def update_importance_one_block(
        self, importance: torch.Tensor, layer_id: int, block_idx: int
    ):
        assert (
            importance.dim() == 1
            and importance.size(0) == self.config.block_len
            and importance.dtype == torch.float16
            and importance.is_contiguous()
            and importance.device == torch.device("cpu")
        ), "importance dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            importance.dim(),
            importance.size(),
            importance.dtype,
            importance.is_contiguous(),
            importance.device,
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.update_importance_one_block(
            importance.data_ptr(),
            layer_id,
            block_idx,
        )

    def get_importance_one_block(
        self, importance: torch.Tensor, layer_id: int, block_idx: int
    ):
        assert (
            importance.dim() == 1
            and importance.size(0) == self.config.block_len
            and importance.dtype == torch.float16
            and importance.is_contiguous()
            and importance.device == torch.device("cpu")
        ), "importance dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            importance.dim(),
            importance.size(),
            importance.dtype,
            importance.is_contiguous(),
            importance.device,
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.get_importance_one_block(
            importance.data_ptr(),
            layer_id,
            block_idx,
        )

    def get_anchor_one_block(self, anchor: torch.Tensor, layer_id: int, block_idx: int):
        assert (
            anchor.dim() == 3
            and anchor.size(0) == self.config.kv_head_num
            and anchor.size(1) == self.config.anchor_num
            and anchor.size(2) == self.config.head_dim
            and anchor.dtype == torch.float16
            and anchor.is_contiguous()
            and anchor.device == torch.device("cpu")
        ), "anchor dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            anchor.dim(),
            anchor.size(),
            anchor.dtype,
            anchor.is_contiguous(),
            anchor.device,
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.get_anchor_one_block(
            anchor.data_ptr(),
            layer_id,
            block_idx,
        )

    def update_anchor_one_block(
        self, anchor: torch.Tensor, layer_id: int, block_idx: int
    ):
        assert (
            anchor.dim() == 3
            and anchor.size(0) == self.config.kv_head_num
            and anchor.size(1) == self.config.anchor_num
            and anchor.size(2) == self.config.head_dim
            and anchor.dtype == torch.float16
            and anchor.is_contiguous()
            and anchor.device == torch.device("cpu")
        ), "anchor dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            anchor.dim(),
            anchor.size(),
            anchor.dtype,
            anchor.is_contiguous(),
            anchor.device,
        )
        assert (
            layer_id >= 0 and layer_id < self.config.layer_num
        ), "layer_id: {}".format(layer_id)
        assert block_idx >= 0, "block_idx: {}".format(block_idx)
        return self.kvcache.update_anchor_one_block(
            anchor.data_ptr(),
            layer_id,
            block_idx,
        )

    def calc_anchor_all_layers(
        self,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ):
        assert (
            block_table.dim() == 2
            and block_table.size(0) == cache_seqlens.size(0)
            and block_table.dtype == torch.int
            and block_table.is_contiguous()
            and block_table.device == torch.device("cpu")
        ), "block_table dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            block_table.dim(),
            block_table.size(),
            block_table.dtype,
            block_table.is_contiguous(),
            block_table.device,
        )
        assert (
            cache_seqlens.dim() == 1
            and cache_seqlens.dtype == torch.int
            and cache_seqlens.is_contiguous()
            and cache_seqlens.device == torch.device("cpu")
        ), "cache_seqlens dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            cache_seqlens.dim(),
            cache_seqlens.size(),
            cache_seqlens.dtype,
            cache_seqlens.is_contiguous(),
            cache_seqlens.device,
        )
        batch_size = block_table.size(0)
        max_block_num = block_table.size(1)
        return self.kvcache.calc_anchor_all_layers(
            block_table.data_ptr(),
            cache_seqlens.data_ptr(),
            batch_size,
            max_block_num,
        )

    def clear_importance_all_layers(
        self,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ):
        assert (
            block_table.dim() == 2
            and block_table.size(0) == cache_seqlens.size(0)
            and block_table.dtype == torch.int
            and block_table.is_contiguous()
            and block_table.device == torch.device("cpu")
        ), "block_table dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            block_table.dim(),
            block_table.size(),
            block_table.dtype,
            block_table.is_contiguous(),
            block_table.device,
        )
        assert (
            cache_seqlens.dim() == 1
            and cache_seqlens.dtype == torch.int
            and cache_seqlens.is_contiguous()
            and cache_seqlens.device == torch.device("cpu")
        ), "cache_seqlens dim: {}, size: {}, dtype: {}, contiguous: {}, device: {}".format(
            cache_seqlens.dim(),
            cache_seqlens.size(),
            cache_seqlens.dtype,
            cache_seqlens.is_contiguous(),
            cache_seqlens.device,
        )
        batch_size = block_table.size(0)
        max_block_num = block_table.size(1)
        return self.kvcache.clear_importance_all_layers(
            block_table.data_ptr(),
            cache_seqlens.data_ptr(),
            batch_size,
            max_block_num,
        )

    def get_cache_total_len(self):
        return self.kvcache.get_cache_total_len()

    def update_kvcache_q4(
        self,
        k_in: torch.Tensor,
        k_scales: torch.Tensor,
        v_in: torch.Tensor,
        v_scales: torch.Tensor,
        layer_id: int,
        seq_offset: int | None = None,
        seq_len: int | None = None,
        block_table: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    def update_kvcache_fp16(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        layer_idx,
        block_table: torch.Tensor,
        max_block_num,
        past_len: torch.Tensor,
        q_len,
    ):
        batch_size = block_table.size(0)
        return self.kvcache.get_kvcache_fp16(
            k_in.data_ptr(),
            v_in.data_ptr(),
            layer_idx,
            block_table.data_ptr(),
            batch_size,
            max_block_num,
            past_len.data_ptr(),
            q_len
        )

    def get_kvcache_q4(
        self,
        k_in: torch.Tensor,
        k_scales: torch.Tensor,
        v_in: torch.Tensor,
        v_scales: torch.Tensor,
        layer_id: int,
        seq_offset: int | None = None,
        seq_len: int | None = None,
        block_table: torch.Tensor | None = None,
    ):
        raise NotImplementedError

    def get_kvcache_fp16(
        self,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        layer_id: int,
        layer_idx,
        block_table: torch.Tensor,
        max_block_num,
        past_len: torch.Tensor,
    ):
        batch_size = block_table.size(0)
        return self.kvcache.get_kvcache_fp16(
            k_in.data_ptr(),
            v_in.data_ptr(),
            layer_idx,
            block_table.data_ptr(),
            batch_size,
            max_block_num,
            past_len.data_ptr(),
        )

    def get_and_update_kvcache_fp16(
        self,
        k_cache_cpu: torch.Tensor,
        v_cache_cpu: torch.Tensor,
        layer_idx,
        block_table: torch.Tensor,
        max_block_num,
        past_len: torch.Tensor,
        q_len,
    ):
        batch_size = block_table.size(0)
        return self.kvcache.get_and_update_kvcache_fp16(
            k_cache_cpu.data_ptr(),
            v_cache_cpu.data_ptr(),
            layer_idx,
            block_table.data_ptr(),
            batch_size,
            max_block_num,
            past_len.data_ptr(),
            q_len,
        )

    def update_importance(
        self,
        importance_cache: torch.Tensor,
        layer_idx,
        block_table: torch.Tensor,
        max_block_num,
        offset: torch.Tensor,
        width,
    ):
        batch_size = block_table.size(0)
        return self.kvcache.update_importance(
            importance_cache.data_ptr(),
            layer_idx,
            block_table.data_ptr(),
            batch_size,
            max_block_num,
            offset.data_ptr(),
            width,
        )

    # attn_sparsity: ((bsz, q_len, q_head_num), dtype = torch.float32)
    def get_attn_sparsity(
        self,
        q_in: torch.Tensor,
        attn_sparsity: torch.Tensor,
        layer_idx: int,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table_origin: torch.Tensor,
        cache_seqlens_origin: torch.Tensor,
        generate_token_idx: int = 0,
        topk: int | None = None,
        local: int | None = None,
    ):
        batch_size = block_table.size(0)
        max_block_num = block_table.size(1)
        max_block_num_origin = block_table_origin.size(1)
        q_len = q_in.size(1)

        if topk is None or local is None or topk + local >= max_block_num:
            topk = -1
            local = -1
        return self.kvcache.get_attn_sparsity(
            q_in.data_ptr(),
            attn_sparsity.data_ptr(),
            layer_idx,
            generate_token_idx,
            q_len,
            batch_size,
            max_block_num,
            block_table.data_ptr(),
            cache_seqlens.data_ptr(),
            block_table_origin.data_ptr(),
            cache_seqlens_origin.data_ptr(),
            max_block_num_origin,
            topk,
            local,
        )

    def attn_with_kvcache(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        output: torch.Tensor,
        attn_lse: torch.Tensor,
        layer_idx: int,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        generate_token_idx: int = 0,
        topk: int | None = None,
        local: int | None = None,
    ):

        batch_size = block_table.size(0)
        max_block_num = block_table.size(1)
        q_len = q_in.size(1)

        if topk is None or local is None or topk + local >= max_block_num:
            topk = -1
            local = -1
        return self.kvcache.attn_with_kvcache(
            q_in.data_ptr(),
            k_in.data_ptr(),
            v_in.data_ptr(),
            output.data_ptr(),
            attn_lse.data_ptr(),
            layer_idx,
            generate_token_idx,
            q_len,
            batch_size,
            max_block_num,
            block_table.data_ptr(),
            cache_seqlens.data_ptr(),
            topk,
            local,
        )

    def get_all_kvcache_one_layer(
        self, k_in: torch.Tensor, v_in: torch.Tensor, layer_id: int
    ):
        return self.kvcache.get_all_kvcache_one_layer(
            k_in.data_ptr(),
            v_in.data_ptr(),
            layer_id,
        )

    def get_importance(
        self,
        importance: torch.Tensor,
        block_table: torch.Tensor,
    ):
        raise NotImplementedError

    def get_anchor(
        self,
        anchor: torch.Tensor,
        block_table: torch.Tensor,
    ):
        raise NotImplementedError


class CPUInfer:
    cpuinfer = None
    def __init__(self, thread_num):
        CPUInfer.cpuinfer = cpuinfer_ext.CPUInfer(thread_num)

    def submit(self, task):
        CPUInfer.cpuinfer.submit(task)

    def submit_with_cuda_stream(self, current_cuda_stream, task):
        CPUInfer.cpuinfer.submit_with_cuda_stream(current_cuda_stream, task)

    def sync(self):
        CPUInfer.cpuinfer.sync()

    def sync_with_cuda_stream(self, current_cuda_stream):
        CPUInfer.cpuinfer.sync_with_cuda_stream(current_cuda_stream)


        
