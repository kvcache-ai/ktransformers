#!/usr/bin/env python
# coding=utf-8
"""
Description  :  
Author       : Jianwei Dong
Date         : 2024-08-26 23:25:24
Version      : 1.0.0
LastEditors  : Jianwei Dong
LastEditTime : 2024-08-26 23:25:24
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import torch
from transformers import AutoConfig
import sys, os
import logging
logger = logging.getLogger("dynamic_attention")
sys.path.append(os.path.dirname(__file__) + "/../ktransformers_ext/cpu_backend")
from ktransformers.operators.cpuinfer import CPUInfer, CPUInferKVCache
from flash_attn import flash_attn_func, flash_attn_with_kvcache


import math
import json


class DynamicScaledDotProductAttention:
    remaining_length: int

    def __init__(
        self,
        max_seq_len: int,
        block_size: int,
        config: AutoConfig,
        device: torch.device,
        local_windows_len: int,
        topk: int,
        threads_num: int,
        anchor_type: str = "DYNAMIC",
        kv_type: str = "FP16",
        dense_layer_num: int = 0,
        anchor_num: int = 1,
        block_selection_mode: str = "SHARED",
        layer_step: int = 1,
        token_step: int = 1,
        preselect_block: bool = False,
        preselect_block_count: int = 96,
        prefill_chunk_size: int = 20480,
        use_attn_sparsity: bool = False,
    ):
        # assert anchor_num == 1
        # assert anchor_type == "DYNAMIC"
        self.remaining_length = 0
        valid_anchor_types = ["DYNAMIC", "FIXED", "BLOCK_MEAN", "BLOCK_MAX", "QUEST"]
        assert anchor_type in valid_anchor_types
        if anchor_type == "QUEST":
            assert anchor_num == 2
        elif anchor_type != "FIXED" and anchor_type != "DYNAMIC":
            assert anchor_num == 1

        valid_kv_types = ["FP16", "FP32", "Q4_0", "Q8_0"]
        assert kv_type in valid_kv_types
        if kv_type != "FP16" and kv_type != "FP32":
            assert block_size % 32 == 0

        valid_block_selection_modes = ["SHARED", "SEPARATE"]  # individual
        assert block_selection_mode in valid_block_selection_modes

        self.max_seq_len = max_seq_len
        self.block_num = max_seq_len // block_size
        self.block_size = block_size
        self.anchor_type = anchor_type
        self.kv_type = kv_type
        self.anchor_num = anchor_num
        self.threads_num = threads_num
        self.layer_step = layer_step
        self.token_step = token_step
        self.preselect_block = preselect_block
        self.preselect_block_count = preselect_block_count
        self.block_selection_mode = block_selection_mode
        self.use_attn_sparsity = use_attn_sparsity

        # model config
        self.kv_head_num = config.num_key_value_heads
        self.q_head_num = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.layer_num = config.num_hidden_layers

        self.device = device
        self.local_windows_len = local_windows_len
        self.local_block_num = self.local_windows_len // self.block_size + 1
        self.prefill_chunk_size = prefill_chunk_size

        self.topk = topk
        self.dense_layer_num = dense_layer_num
        # self.dense_layer_num = 32
        self.cache_key_states = torch.zeros(
            (self.block_num, block_size, self.kv_head_num, self.head_dim),
            device=device,
            dtype=torch.float16,
        )
        self.cache_value_states = torch.zeros(
            (self.block_num, block_size, self.kv_head_num, self.head_dim),
            device=device,
            dtype=torch.float16,
        )
        # [max_num_block, block_size, head_num]
        self.cache_importance = torch.zeros(
            (self.block_num, block_size, self.q_head_num),
            device=device,
            dtype=torch.float16,
        )

        # key_states: [bsz, q_len, kv_head_num, head_dim]
        # value_states: [bsz, q_len, kv_head_num, head_dim]
        # query_states: [bsz, q_len, q_head_num, head_dim]
        self.q_in_cpu = torch.zeros(
            (1, 1, self.q_head_num, self.head_dim),
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )
        self.k_in_cpu = torch.zeros(
            (1, 1, self.kv_head_num, self.head_dim),
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )
        self.v_in_cpu = torch.zeros(
            (1, 1, self.kv_head_num, self.head_dim),
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )

        self.cache_seqlens_cpu = torch.empty(
            (1,), device="cpu", dtype=torch.int32, pin_memory=True
        )

        self.cache_seqlens_cuda = torch.empty((1,), device=device, dtype=torch.int32)

        self.prefix_block_table = torch.arange(
            self.block_num, device="cpu", dtype=torch.int32, pin_memory=True
        ).view(1, -1)

        self.block_table_cpu = torch.arange(
            self.block_num, device="cpu", dtype=torch.int32, pin_memory=True
        ).view(1, -1)

        # assert (
        #     self.local_windows_len // self.block_size + 1 + self.preselect_block_count
        #     <= self.block_num
        # )

        self.output_cpu = torch.empty(
            (1, 1, self.q_head_num, self.head_dim),
            device="cpu",
            dtype=torch.float16,
            pin_memory=True,
        )
        self.lse_cpu = torch.empty(
            (1, 1, self.q_head_num), device="cpu", dtype=torch.float32, pin_memory=True
        )

        self.output_cuda = torch.empty(
            (1, 1, self.q_head_num, self.head_dim), device=device, dtype=torch.float16
        )

        self.attn_sparsity = torch.zeros(
            (1, 1, self.q_head_num), device="cpu", dtype=torch.float32, pin_memory=True
        )

        if preselect_block == True:
            self.preselect_block_table = torch.zeros(
                self.layer_num,
                self.preselect_block_count,
                device=device,
                dtype=torch.int32,
            )
            self.preselect_block_num = 0  # block_num before preselect
            self.evict_tokens = 0

        self.cpu_infer = CPUInfer(threads_num)
        self.local_thread = CPUInferKVCache(
            self.layer_num,
            self.kv_head_num,
            self.q_head_num,
            self.head_dim,
            self.block_size,
            anchor_num=self.anchor_num,
            anchor_type=anchor_type,
            kv_type=self.kv_type,
            retrieval_type=self.block_selection_mode,
            layer_step=self.layer_step,
            token_step=self.token_step,
            layer_offset=self.dense_layer_num % self.layer_step,
            max_batch_size=1,
            max_block_num=self.block_num,
            max_thread_num=self.threads_num,
        )

        print(
            f"local_windows_len: {local_windows_len}, topk: {topk}, dense_layer_num: {dense_layer_num}, kv_type: {self.kv_type}, anchor_type: {self.anchor_type}, preselect_block: {self.preselect_block}, preselect_block_count: {self.preselect_block_count}, token_step: {self.token_step}, layer_step: {self.layer_step}"
        )

        self.shape_mask = (
            self.q_head_num,
            self.block_size,
            self.block_size,
        )

        mask = torch.zeros(
            self.shape_mask, dtype=torch.uint8, device=device
        ).contiguous()
        elm_idx = torch.arange(self.block_size, device=device)

        for i in range(mask.size(-2)):
            idx = i + mask.size(-1) - mask.size(-2) - elm_idx
            idx = idx[idx >= 0]
            mask[..., i, idx] = 1

        self.tril_mask = mask
        self.triu_mask = mask ^ 1

        self.generate_token_idx = 0

    def get_attn_score_one_block(
        self,
        batch_idx: int,
        max_block_num: int,
        query: torch.Tensor,
        key: torch.Tensor,
        offset: int,
        width: int,
        mask_mode: str | None = None,
        use_softmax: bool = True,
    ):
        n_rep = self.q_head_num // self.kv_head_num
        importance = self.cache_importance.view(-1, self.q_head_num)
        importance = importance.narrow(0, batch_idx * max_block_num + offset, width)
        n_gqa_ = self.q_head_num // self.kv_head_num 
        for head_idx in range(self.q_head_num):
            key_item = key[..., head_idx // n_gqa_, :].view(key.size(0), -1)
            qk = torch.einsum(
                "qd,kd->qk", query[:,head_idx,:], key_item
            )  # (num_attention_heads, len_q, len_k)

            if mask_mode == "tril":
                mask = self.tril_mask
                mask = mask[0, -qk.size(-2) :, -qk.size(-1) :]
                qk = qk * mask
            elif mask_mode == "triu":
                mask = self.triu_mask
                mask = mask[0, -qk.size(-2) :, -qk.size(-1) :]
                qk = qk * mask

            if use_softmax:
                qk = torch.nn.functional.softmax(
                    qk / math.sqrt(self.head_dim), dim=-1, dtype=torch.float32
                ).to(torch.float16)
              
            qk = torch.sum(qk, dim=-2)
            importance[...,head_idx] += qk

    def get_preselect_block_table_and_attn_score(
        self,
        layer_idx: int,
        batch_size: int,
        offset: torch.Tensor,
        width: int,
        query: torch.Tensor,
        key: torch.Tensor,
        union_with_last_layer: bool = True,
    ):
        max_seqs_len = offset.max().item() + width
        max_block_num = (max_seqs_len + self.block_size - 1) // self.block_size

        for batch_idx in range(batch_size):
            query_cur = query[batch_idx][-128:]
            self.get_attn_score_one_block(
                batch_idx,
                max_block_num,
                query_cur,
                key[batch_idx][: offset[batch_idx].item() + width],
                0,
                offset[batch_idx].item() + width,
                mask_mode=None,
            )

        if self.preselect_block:
            self.prefill_block_num = max(
                0, max_block_num - self.local_windows_len // self.block_size
            )
            self.evict_tokens = (
                max(self.prefill_block_num - self.preselect_block_count, 0)
                * self.block_size
            )

            if self.prefill_block_num != 0:
                importance_cache = self.cache_importance.narrow(
                    0, 0, self.prefill_block_num * batch_size
                ).view(
                    batch_size, self.prefill_block_num, self.block_size, self.q_head_num
                )

                importance_r = importance_cache[:, 1:, : self.block_size // 4]
                pad_r = torch.zeros_like(importance_r[:, :1])
                importance_r = torch.cat((importance_r, pad_r), dim=1)
                importance_l = importance_cache[:, :-1, -self.block_size // 4 :]
                pad_l = torch.zeros_like(importance_l[:, :1])
                importance_l = torch.cat((pad_l, importance_l), dim=1)
                importance = torch.cat(
                    (importance_l, importance_cache, importance_r), dim=2
                )
                importance = importance.mean(dim=-1)
                importance = importance.mean(dim=-1)
                # importance: (batch_size, max_block_num)
                topk = min(self.preselect_block_count, self.prefill_block_num)
                values, indices = torch.topk(
                    importance,
                    k=topk,
                    dim=1,
                )

                self.preselect_block_table[
                    layer_idx : layer_idx + 1,
                    :topk,
                ].copy_(indices)

                if union_with_last_layer and layer_idx == 31:
                    for tmp_layer_idx in range(self.layer_num - 1):
                        for i in range(1, min(topk, 6)):
                            x = self.preselect_block_table[-1, i]
                            if x not in self.preselect_block_table[tmp_layer_idx]:
                                self.preselect_block_table[tmp_layer_idx, topk - i] = x
        if self.anchor_type == "DYNAMIC":
            importance_cache = self.cache_importance.narrow(
                0, 0, max_block_num * batch_size
            ).view(batch_size, max_block_num * self.block_size, self.q_head_num)
            importance_cache_cpu = torch.empty_like(
                importance_cache, device="cpu", pin_memory=True
            )

            importance_cache_cpu.copy_(importance_cache)

            block_table_cpu = self.prefix_block_table[:, :max_block_num].to("cpu")
            offset_cpu = offset.contiguous().to("cpu")

            self.cpu_infer.submit(
                self.local_thread.update_importance(
                    importance_cache_cpu,
                    layer_idx,
                    block_table_cpu,
                    max_block_num,
                    offset_cpu,
                    width,
                )
            )
            self.cpu_infer.sync()

        importance_cache = self.cache_importance.narrow(
            0, 0, max_block_num * batch_size
        ).view(batch_size, max_block_num * self.block_size, self.q_head_num)
        importance_cache.zero_()

    # key: [bsz, past_len, head_num, head_dim] float16
    # query: [bsz, q_len, q_head_num, head_dim] float16
    def get_attn_score(
        self,
        layer_idx: int,
        batch_size: int,
        offset: torch.Tensor,
        width: int,
        query: torch.Tensor,
        key: torch.Tensor,
    ):
        max_seqs_len = offset.max().item() + width
        max_block_num = (max_seqs_len + self.block_size - 1) // self.block_size

        for batch_idx in range(batch_size):
            for idx in range(width // self.block_size):
                offset_cur = idx * self.block_size
                query_cur = query[batch_idx, offset_cur : offset_cur + self.block_size]
                self.get_attn_score_one_block(
                    batch_idx,
                    max_block_num,
                    query_cur,
                    key[
                        batch_idx,
                        offset[batch_idx]
                        + offset_cur : offset[batch_idx]
                        + offset_cur
                        + self.block_size,
                    ],
                    offset[batch_idx].item() + offset_cur,
                    self.block_size,
                    mask_mode="tril",
                    use_softmax=False,
                )

                offset_key = (
                    offset[batch_idx].item()
                    + idx * self.block_size
                    - self.local_windows_len
                )
                if offset_key >= 0:
                    self.get_attn_score_one_block(
                        batch_idx,
                        max_block_num,
                        query_cur,
                        key[batch_idx, offset_key : offset_key + self.block_size],
                        offset_key,
                        self.block_size,
                        mask_mode="triu",
                        use_softmax=False,
                    )

                offset_key = max(0, offset_key + self.block_size)
                width_key = (
                    offset[batch_idx].item() + idx * self.block_size - offset_key
                )
                if width_key > 0:
                    self.get_attn_score_one_block(
                        batch_idx,
                        max_block_num,
                        query_cur,
                        key[batch_idx, offset_key : offset_key + width_key],
                        offset_key,
                        width_key,
                        mask_mode=None,
                        use_softmax=False,
                    )

        importance_cache = self.cache_importance.narrow(
            0, 0, max_block_num * batch_size
        ).view(batch_size, max_block_num * self.block_size, self.q_head_num)
        importance_cache_cpu = torch.empty_like(
            importance_cache, device="cpu", pin_memory=True
        )

        importance_cache_cpu.copy_(importance_cache)

        block_table_cpu = self.prefix_block_table[:, :max_block_num].to("cpu")
        offset_cpu = offset.contiguous().to("cpu")

        self.cpu_infer.submit(
            self.local_thread.update_importance(
                importance_cache_cpu,
                layer_idx,
                block_table_cpu,
                max_block_num,
                offset_cpu,
                width,
            )
        )
        self.cpu_infer.sync()
        importance_cache.zero_()

    # key: [bsz, q_len, head_num, head_dim] float16
    # value: [bsz, q_len, head_num, head_dim] float16
    def swap_in_and_swap_out(self, layer_idx, past_len, q_len, key, value):
        batch_size = 1
        max_seqs_len = past_len.max().item() + q_len
        max_block_num = (max_seqs_len + self.block_size - 1) // self.block_size
        k_cache = self.cache_key_states.narrow(0, 0, max_block_num * batch_size).view(
            batch_size, max_block_num * self.block_size, self.kv_head_num, self.head_dim
        )
        v_cache = self.cache_value_states.narrow(0, 0, max_block_num * batch_size).view(
            batch_size, max_block_num * self.block_size, self.kv_head_num, self.head_dim
        )

        for batch_idx in range(batch_size):
            offset = past_len[batch_idx]
            width = q_len
            k_cache[batch_idx][offset : offset + width].copy_(
                key[batch_idx].view(-1, self.kv_head_num, self.head_dim)
            )
            v_cache[batch_idx][offset : offset + width].copy_(
                value[batch_idx].view(-1, self.kv_head_num, self.head_dim)
            )

        k_cache_cpu = torch.empty_like(k_cache, device="cpu", pin_memory=True)
        v_cache_cpu = torch.empty_like(v_cache, device="cpu", pin_memory=True)

        k_cache_cpu.copy_(k_cache)
        v_cache_cpu.copy_(v_cache)

        cur_block_num = (
            q_len + past_len[0].item() + self.block_size - 1
        ) // self.block_size
        block_table_cpu = self.prefix_block_table[:, :cur_block_num].to("cpu")
        past_len_cpu = past_len.contiguous().to("cpu")

        self.cpu_infer.submit(
            self.local_thread.get_and_update_kvcache_fp16(
                k_cache_cpu,
                v_cache_cpu,
                layer_idx,
                block_table_cpu,
                max_block_num,
                past_len_cpu,
                q_len,
            )
        )

        self.cpu_infer.sync()
        k_cache.copy_(k_cache_cpu)
        v_cache.copy_(v_cache_cpu)

        return k_cache, v_cache

    def calc_anchor(self, cache_seqlens: int):
        cur_block_num = (cache_seqlens + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[:, :cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor(
            [cache_seqlens], device="cpu", dtype=torch.int32
        )

        self.cpu_infer.submit(
            self.local_thread.calc_anchor_all_layers(
                block_table_cpu,
                cache_seqlens_cpu,
            )
        )
        self.cpu_infer.sync()

    def clear_importance(self, cache_seqlens: int):
        print(f"clear importance: {cache_seqlens}")
        cur_block_num = (cache_seqlens + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[:, :cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor(
            [cache_seqlens], device="cpu", dtype=torch.int32
        )

        self.cpu_infer.submit(
            self.local_thread.clear_importance_all_layers(
                block_table_cpu,
                cache_seqlens_cpu,
            )
        )
        self.cpu_infer.sync()

    def clear_kvcache(self, cache_seqlens: int):
        cur_block_num = (cache_seqlens + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[:, :cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor(
            [cache_seqlens], device="cpu", dtype=torch.int32
        )

        self.cpu_infer.submit(
            self.local_thread.clear_kvcache_all_layers(
                block_table_cpu,
                cache_seqlens_cpu,
            )
        )
        self.cpu_infer.sync()

    def get_attn_sparsity(
        self,
        q_in: torch.Tensor,
        layer_idx: int,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table_origin: torch.Tensor,
        cache_seqlens_origin: torch.Tensor,
        generate_token_idx: int = 0,
        topk: int | None = None,
        local: int | None = None,
        output_path: str = "./attn_sparsity.json",
    ):
        self.attn_sparsity.zero_()
        self.pcinfer.submit(
            self.local_thread.get_attn_sparsity(
                q_in,
                self.attn_sparsity,
                layer_idx,
                block_table,
                cache_seqlens,
                block_table_origin,
                cache_seqlens_origin,
                generate_token_idx,
                topk,
                local,
            )
        )
        self.cpu_infer.sync()
        with open(output_path, "a") as file:
            for head_idx in range(self.q_head_num):
                sparsity = self.attn_sparsity[0][0][head_idx].item()
                json_obj = {
                    "token_idx": generate_token_idx,
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "sparsity": sparsity,
                }
                json.dump(json_obj, file)
                file.write("\n")

    def apply(
        self,
        layer_idx: int,
        bsz: int,
        past_len: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "prefill",
        generate_token_idx: int = -1,
    ):

        # key_states: [bsz, q_len, kv_head_num, head_dim]
        # value_states: [bsz, q_len, kv_head_num, head_dim]
        # query_states: [bsz, q_len, q_head_num, head_dim]
        assert query_states.dtype == torch.float16
        assert key_states.dtype == torch.float16
        assert value_states.dtype == torch.float16

        assert key_states.size(2) == self.kv_head_num
        assert value_states.size(2) == self.kv_head_num
        assert query_states.size(2) == self.q_head_num

        q_len = query_states.size(1)
        batch_size = query_states.size(0)
        self.cache_seqlens_cuda.fill_(past_len)
        last_chunk = False
        if self.remaining_length <= self.prefill_chunk_size and q_len != 1:
            last_chunk = True
        device = query_states.device
        if layer_idx == 0:
            if q_len == 1:
                self.generate_token_idx += 1
            elif last_chunk:
                self.generate_token_idx = -1

        if mode == "prefill":
            key, value = self.swap_in_and_swap_out(
                layer_idx,
                self.cache_seqlens_cuda,
                q_len,
                key_states,
                value_states,
            )

            if last_chunk and (self.anchor_type == "DYNAMIC" or self.preselect_block):
                self.get_preselect_block_table_and_attn_score(
                    layer_idx,
                    bsz,
                    self.cache_seqlens_cuda,
                    q_len,
                    query_states,
                    key,
                )
            output = flash_attn_with_kvcache(
                q=query_states,
                k_cache=key,
                v_cache=value,
                cache_seqlens=self.cache_seqlens_cuda + q_len,
                causal=True,
            )
            return output.transpose(1, 2)

        elif mode == "generate":
            assert self.generate_token_idx >= 0
            self.q_in_cpu.copy_(query_states, non_blocking=True)
            self.k_in_cpu.copy_(key_states, non_blocking=True)
            self.v_in_cpu.copy_(value_states, non_blocking=True)
            self.cache_seqlens_cpu.copy_(self.cache_seqlens_cuda, non_blocking=True)
            #            print(layer_idx)
            if layer_idx < self.dense_layer_num:
                self.block_table_cpu.copy_(self.prefix_block_table, non_blocking=True)
                self.cpu_infer.submit_with_cuda_stream(
                    torch.cuda.current_stream("cuda").cuda_stream,
                    self.local_thread.attn_with_kvcache(
                        q_in=self.q_in_cpu,
                        k_in=self.k_in_cpu,
                        v_in=self.v_in_cpu,
                        output=self.output_cpu,
                        attn_lse=self.lse_cpu,
                        layer_idx=layer_idx,
                        block_table=self.block_table_cpu,
                        cache_seqlens=self.cache_seqlens_cpu,
                    ),
                )
            else:
                if self.preselect_block:
                    self.cache_seqlens_cpu.copy_(
                        self.cache_seqlens_cuda - self.evict_tokens, non_blocking=True
                    )
                    if self.preselect_block_count < self.prefill_block_num:
                        self.block_table_cpu[:, : self.preselect_block_count].copy_(
                            self.preselect_block_table[layer_idx : layer_idx + 1],
                            non_blocking=True,
                        )

                        self.block_table_cpu[
                            :,
                            self.preselect_block_count : self.preselect_block_count
                            + self.local_block_num,
                        ].copy_(
                            self.prefix_block_table[
                                :,
                                self.prefill_block_num : self.prefill_block_num
                                + self.local_block_num,
                            ],
                            non_blocking=True,
                        )
                    #                   print("submit_with_cuda_stream")
                    self.cpu_infer.submit_with_cuda_stream(
                        torch.cuda.current_stream("cuda").cuda_stream,
                        self.local_thread.attn_with_kvcache(
                            q_in=self.q_in_cpu,
                            k_in=self.k_in_cpu,
                            v_in=self.v_in_cpu,
                            output=self.output_cpu,
                            attn_lse=self.lse_cpu,
                            layer_idx=layer_idx,
                            generate_token_idx=self.generate_token_idx,
                            block_table=self.block_table_cpu,
                            cache_seqlens=self.cache_seqlens_cpu,
                            topk=(
                                self.topk
                                if self.topk <= self.preselect_block_count
                                else None
                            ),
                            local=self.local_windows_len // self.block_size,
                        ),
                    )
                #                    print("submit_with_cuda_stream enqueue\n")
                else:
                    self.block_table_cpu.copy_(
                        self.prefix_block_table, non_blocking=True
                    )
                    self.cpu_infer.submit_with_cuda_stream(
                        torch.cuda.current_stream("cuda").cuda_stream,
                        self.local_thread.attn_with_kvcache(
                            q_in=self.q_in_cpu,
                            k_in=self.k_in_cpu,
                            v_in=self.v_in_cpu,
                            output=self.output_cpu,
                            attn_lse=self.lse_cpu,
                            layer_idx=layer_idx,
                            generate_token_idx=self.generate_token_idx,
                            block_table=self.block_table_cpu,
                            cache_seqlens=self.cache_seqlens_cpu,
                            topk=self.topk,
                            local=self.local_windows_len // self.block_size,
                        ),
                    )
            self.cpu_infer.sync_with_cuda_stream(
                torch.cuda.current_stream("cuda").cuda_stream
            )
            #            print("submit_with_cuda_stream finished\n")
            self.output_cuda.copy_(self.output_cpu, non_blocking=True)
            return self.output_cuda.transpose(1, 2)

    def save(self, path: str, length: int):
        cur_block_num = (length + self.block_size - 1) // self.block_size
        block_table_cpu = self.prefix_block_table[0, :cur_block_num].to("cpu")
        cache_seqlens_cpu = torch.tensor([length], device="cpu", dtype=torch.int32)
        self.cpu_infer.submit(
            self.local_thread.dump_kvcache(
                block_table_cpu,
                cache_seqlens_cpu,
                path,
            )
        )
        self.cpu_infer.sync()

    def load(self, path: str, length: int):
        self.cpu_infer.submit(
            self.local_thread.load_kvcache(
                path,
            )
        )
        self.cpu_infer.sync()
