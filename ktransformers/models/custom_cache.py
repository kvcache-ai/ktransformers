'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
'''
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/cache_utils.py
# Copyright 2018- The Hugging Face team. All rights reserved.
# Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
import torch
import torch.nn as nn
import transformers
from transformers import Cache, PretrainedConfig
from typing import List, Optional, Dict, Any, Tuple
try:
    from ktransformers.server.balance_serve.settings import sched_ext
except:
    print("no balance_serve")

try:
    import torch_npu
    from ktransformers.util import utils
    from ktransformers.server.balance_serve.inference.forward_batch import ForwardMiniBatchCombine, ForwardMiniBatchSplit
    
    use_torch_npu = torch_npu.npu.is_available()
except:
    use_torch_npu = False


class StaticCache(transformers.StaticCache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `dict`):
            The device on which the cache should be initialized. Should be the same as the layer.
            If a `dict`, it should contain the `device` key with the device name as the value.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device: torch.device| dict, dtype=None) -> None:
        Cache.__init__(self)
        self.max_batch_size = max_batch_size

        if use_torch_npu:
            self.position = [0]

        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        if config.architectures[0] == "DeepseekV3ForCausalLM":
            self.head_dim = config.qk_rope_head_dim
        else:
            self.head_dim = (
                config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
            )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        if config.architectures[0] == "DeepseekV2ForCausalLM" or config.architectures[0] == "DeepseekV3ForCausalLM":
            # TODO: for deepseek, cache_shape is different whether using Absorbed MLA, check it automatically

            if use_torch_npu:
                self.page_size = 128
                self.page_size_tensor = torch.tensor(
                self.page_size,
                dtype=torch.int32,
                    ).npu()
                self.max_pages_per_batch = (self.max_cache_len + self.page_size - 1) // self.page_size
                self.max_pages = (self.max_cache_len + self.page_size - 1) // self.page_size * self.max_batch_size
            else:
                self.page_size = 64
                self.max_pages = (self.max_cache_len + self.page_size - 1) // self.page_size
            latent_shape = (self.max_pages, self.page_size, 1, config.kv_lora_rank + config.qk_rope_head_dim)
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_rope_head_dim = config.qk_rope_head_dim
            # TODO: support real page table
            self.page_table_map = dict()
            self.page_table_list = []
            for idx in range(config.num_hidden_layers):
                if isinstance(device, dict):
                    target_device = device[f"blk.{idx}.self_attn"]["generate_device"]
                else:
                    target_device = device
                
                if target_device not in self.page_table_map:
                    if use_torch_npu:
                        page_table = torch.zeros((max_batch_size, self.max_pages_per_batch), dtype=torch.int32, device=target_device)
                        for seq_id in range(max_batch_size):
                            page_table[seq_id, :] = torch.arange(seq_id * self.max_pages_per_batch, seq_id * self.max_pages_per_batch + self.max_pages_per_batch, dtype=torch.int32, device=target_device)
                    else:
                        page_table = torch.zeros((max_batch_size, self.max_pages), dtype=torch.int32, device=target_device)
                        for seq_id in range(max_batch_size):
                            page_table[seq_id, :] = torch.arange(seq_id * self.max_pages, seq_id * self.max_pages + self.max_pages, dtype=torch.int32, device=target_device)
                    self.page_table_map[target_device] = page_table
                    
                self.page_table_list.append(self.page_table_map[target_device])
                    
            self.is_MLA = True
            self.is_page = True
        else:
            key_shape = cache_shape
            value_shape = cache_shape
            self.is_MLA = False

        self.past_tokens = []
        self.num_hidden_layers = config.num_hidden_layers
        for idx in range(self.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            if isinstance(device, dict):
                target_device = device[f"blk.{idx}.self_attn"]["generate_device"]
            else:
                target_device = device
            
            if self.is_MLA:
                new_layer_key_cache = torch.zeros(latent_shape, dtype=self.dtype, device=target_device)
                new_layer_value_cache = None
                torch._dynamo.mark_static_address(new_layer_key_cache)
            else:
                new_layer_key_cache = torch.zeros(key_shape, dtype=self.dtype, device=target_device)
                new_layer_value_cache = torch.zeros(value_shape, dtype=self.dtype, device=target_device)
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
                
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            self.past_tokens.append(0)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        self.past_tokens[layer_idx] += cache_position.size(0)
        #print(cache_position)
        if self.is_MLA:
            if use_torch_npu:
                page_idx = cache_position // self.page_size_tensor
                page_offset = cache_position % self.page_size_tensor

                page_idx = page_idx.unsqueeze(0).expand(self.max_batch_size, -1)
                page_offset = page_offset.unsqueeze(0).expand(self.max_batch_size, -1)

                page_idx_offset = torch.arange(self.max_batch_size, device=page_idx.device) * self.max_pages_per_batch
                page_idx = page_idx + page_idx_offset.unsqueeze(1)

                combined = torch.cat([key_states, value_states], dim=-1)
                combined = combined.contiguous()
                # key shape (self.max_pages, self.page_size, 1, config.kv_lora_rank + config.qk_rope_head_dim)
                k_out[page_idx, page_offset] = combined
            else:
                page_idx = cache_position // self.page_size
                page_offset = cache_position % self.page_size
                # key shape (self.max_pages, self.page_size, 1, config.kv_lora_rank + config.qk_rope_head_dim)
                k_out[page_idx, page_offset, :, :self.kv_lora_rank] = key_states
                k_out[page_idx, page_offset, :, self.kv_lora_rank:] = value_states
            return k_out, self.page_table_list[layer_idx]
        else:
            k_out[:, :, cache_position] = key_states
            v_out[:, :, cache_position] = value_states
            return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return self.past_tokens[layer_idx]
    
    def change_seq_length(self, bias: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        for layer_idx in range(self.num_hidden_layers):
            self.past_tokens[layer_idx] += bias

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            if self.value_cache[layer_idx] is not None:
                self.value_cache[layer_idx].zero_()
            self.past_tokens[layer_idx] = 0
        
        if use_torch_npu:
            self.position = [0]

    def remove_suffix(self, start_pos):
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            if self.is_MLA:
                k_cache = self.key_cache[layer_idx]
                k_cache.view(-1, k_cache.shape[-1])[start_pos:].zero_()
            else:
                self.key_cache[layer_idx][..., start_pos:, :].zero_()
                self.value_cache[layer_idx][..., start_pos:, :].zero_()
            self.past_tokens[layer_idx] = start_pos
    
    def get_max_cache_shape(self) -> Tuple[int, int, int, int]:
        """Returns the maximum shape of the cache."""
        return self.max_cache_len

class KVC2StaticCache(transformers.Cache):
    """
    Static Cache class connect with KVC2
    remind: page_idx & page_offset info need to refs to forward batching, only contains KV Block Tensor here
    """
    def __init__(self, config: PretrainedConfig, max_batch_size, page_size: int = 256, dtype=torch.bfloat16, device=None) -> None:
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = torch.device("npu:0")
        self.kv_lora_rank = config.kv_lora_rank
        self.max_batch_size = max_batch_size
        self.page_size = page_size
        self.k_caches = []
        self.v_caches = []

        self.num_hidden_layers = config.num_hidden_layers

        self.is_MLA = True if config.architectures[0] in ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"] else False
        # kv cache stored in kvc2
        # self.past_tokens = []

    def load(self, inference_context):
        # assert self.is_MLA and len(inference_context.k_cache) == 1, "currently only support MLA and Cache Pool TP=1"
        from ktransformers.util.utils import get_current_device
        for i in range(self.config.num_hidden_layers):
            new_layer_key_cache = inference_context.k_cache[int(torch.distributed.get_rank())][i].to(get_current_device())
            torch._dynamo.mark_static_address(new_layer_key_cache)

            self.k_caches.append(
                new_layer_key_cache  # [TP_idx, layer_idx, page_idx, page_size, kv_head_num, kv_head_dim]
            )

            self.v_caches.append(None)
        self.max_cache_len = self.k_caches[0].shape[0] * self.k_caches[0].shape[1]  # page_len * page_size

    def update(
        self,
        combined: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                must have page_idx (`torch.Tensor`): & page_offset (`torch.Tensor`) & cache_position (`torch.Tensor`)

        Return:
            A tuple containing the updated key and value states.
        """
        page_idx, page_offset = cache_kwargs.get("page_idx"), cache_kwargs.get("page_offset")
        if page_idx is None or page_offset is None:
            raise ValueError('[ERROR] block info:page_idx & page_offset missing!')

        k_out = self.k_caches[layer_idx]
        # v_out = self.value_cache[layer_idx]
        # self.past_tokens[layer_idx] += cache_position.size(0)
        # print(cache_position)
        assert self.is_MLA, "currently only support DeepSeekV3 on NPU balance server"

        # print("############### page_idx id is ", id(page_idx))
        # print("############### page_offset id is ", id(page_offset))
        # 一次性写入到 k_out 的指定位置+
        # print("k_out size is ", k_out.size())
        if page_idx.dim() == 1:
            page_idx_tmp = page_idx.unsqueeze(0)
            page_offset_tmp = page_offset.unsqueeze(0)
        else:
             page_idx_tmp = page_idx
             page_offset_tmp = page_offset

        # # page_idx = page_idx.flatten()
        # # page_offset = page_offset.flatten()
        # if layer_idx == 0:
        #     # print("page_idx ", page_idx, "page_offset", page_offset)
        #     print("page_idx_tmp ", page_idx_tmp.size(), "page_offset_tmp", page_offset_tmp.size())
        #     print("k_out[page_idx, page_offset] size is ", k_out[page_idx_tmp, page_offset_tmp].size())
        #     print("combined size is ", combined.size())

        k_out[page_idx_tmp, page_offset_tmp] = combined
        return k_out, page_idx

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        raise ValueError('kvc2 cache pool no longer hold seq_length info, refer to forward batching')

    def get_usable_length(self, kv_seq_len, layer_idx: Optional[int] = 0) -> int:
        # print('[WARN] currently npu balance serve do not support chunk prefill or prefix cache')
        return 0

    def change_seq_length(self, bias: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        raise ValueError('kvc2 cache pool no longer hold seq_length info, refer to forward batching')

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self, inference_context):
        # print(" KVC2StaticCache reset activate 1!!!!")
        assert self.is_MLA and len(inference_context.k_cache) == 1, "currently only support MLA and Cache Pool TP=1"
        self.k_caches = []
        self.v_caches = []
        for i in range(self.config.num_hidden_layers):
            self.k_caches.append(
                inference_context.k_cache[0][i]
            )
            self.v_caches.append(None)
        self.max_cache_len = self.k_caches[0].shape[0] * self.k_caches[0].shape[1]  # page_len * page_size

    # todo-luo 这个 get_page_table 和 另外连个类的入参不一样
    def get_page_table(self, mini_batch, bsz_tensors: torch.tensor = None, is_prefill=True):
        if is_prefill:
            # TODO add padding support
            q_lens = [mini_batch.p_q_len[idx] for idx in range(mini_batch.prefill_batch)]
            page_local_idx = -1 * torch.ones(mini_batch.prefill_batch, max(q_lens),
                                             dtype=mini_batch.p_position_ids.dtype, device=mini_batch.p_position_ids.device)
            page_offset = -1 * torch.ones_like(page_local_idx)
            # convert merged into batched
            start_ids = 0
            for i in range(mini_batch.prefill_batch):
                page_offset[i, 0:q_lens[i]] = mini_batch.p_position_ids[start_ids:start_ids+q_lens[i]] % self.page_size
                page_local_idx[i, 0:q_lens[i]] = mini_batch.p_position_ids[start_ids:start_ids+q_lens[i]] // self.page_size
                for j in range(q_lens[i]):
                    # get global page idx index by local page idx from block table, as followed decode
                    page_local_idx[i, j] = mini_batch.p_block_tables[i, page_local_idx[i, j]]
                start_ids += q_lens[i]
            page_idx = page_local_idx
            # only padding will cause page_local_idx/page_offset still have -1 value
            # you can use following code as check
            # indices = torch.where(page_offset == -1)
            # assert not indices[0].numel() > 0, 'there still have un-calculated page_idx value'
        else:
            page_local_idx = mini_batch.d_position_ids // self.page_size
            # print("page_local_idx is ", page_local_idx)
            # print("mini_batch.d_block_tables is ", mini_batch.d_block_tables)

            page_offset = mini_batch.d_position_ids % self.page_size
            
            for i in range(mini_batch.decode_batch):
                page_local_idx[i] = mini_batch.d_block_tables[i, page_local_idx[i]]
            
            page_idx = page_local_idx
            # print("page_idx is ", page_idx)
            # print("page_offset is ", page_offset)

        return page_idx, page_offset

class KDeepSeekV3Cache(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        page_size: int = 256,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.kv_lora_rank = config.kv_lora_rank
        self.page_size = page_size
        self.k_caches = []
        self.v_caches = []
        

    def load(self, inference_context: "sched_ext.InferenceContext"):
        
        for i in range(self.config.num_hidden_layers):
            self.k_caches.append(
                inference_context.k_cache[0][i] 
            )
        self.max_cache_len = self.k_caches[0].shape[0]*self.k_caches[0].shape[1]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,

        page_idx: torch.Tensor,
        page_offset: torch.Tensor,

        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        k_out = self.k_caches[layer_idx]

        k_out[page_idx, page_offset, :, :self.kv_lora_rank] = key_states.reshape(-1, *key_states.shape[2:])
        k_out[page_idx, page_offset, :, self.kv_lora_rank:] = value_states.reshape(-1, *value_states.shape[2:])
        return k_out

        
    def get_page_table(self, cache_position: torch.Tensor, q_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, bsz_tensors: torch.tensor):
        page_offset = cache_position % self.page_size  
        page_idx_local = cache_position // self.page_size  
        query_ids = torch.zeros_like(cache_position)
        for i in range(len(q_indptr) - 1):
            start_idx = q_indptr[i]
            end_idx = q_indptr[i + 1]
            query_ids[start_idx:end_idx] = i
        page_idx = torch.zeros_like(page_idx_local)
        for i in range(bsz_tensors[0]):
            query_id = query_ids[i]
            local_block = page_idx_local[i]
            start_block = kv_indptr[query_id]
            if local_block < kv_indptr[query_id + 1] - kv_indptr[query_id]:
                page_idx[i] = kv_indices[start_block + local_block]
        
        return page_idx, page_offset
    
class KGQACache(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        page_size: int = 256,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.page_size = page_size
        self.k_caches = []
        self.v_caches = []
        

    def load(self, inference_context: "sched_ext.InferenceContext"):
        print(self.config.num_hidden_layers)
        for i in range(self.config.num_hidden_layers):
            self.k_caches.append(
                inference_context.k_cache[0][i] 
            )
            self.v_caches.append(
                inference_context.v_cache[0][i]
            )


        self.max_cache_len = self.k_caches[0].shape[0]*self.k_caches[0].shape[1]


        
    def get_page_table(self, cache_position: torch.Tensor, q_indptr: torch.Tensor, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, bsz_tensors: torch.tensor):
        page_offset = cache_position % self.page_size  
        page_idx_local = cache_position // self.page_size  
        query_ids = torch.zeros_like(cache_position)
        for i in range(len(q_indptr) - 1):
            start_idx = q_indptr[i]
            end_idx = q_indptr[i + 1]
            query_ids[start_idx:end_idx] = i
        page_idx = torch.zeros_like(page_idx_local)
        for i in range(bsz_tensors[0]):
            query_id = query_ids[i]
            local_block = page_idx_local[i]
            start_block = kv_indptr[query_id]
            if local_block < kv_indptr[query_id + 1] - kv_indptr[query_id]:
                page_idx[i] = kv_indices[start_block + local_block]
        
        return page_idx, page_offset

    def get_k_cache(self, layer_idx):
        return self.k_caches[layer_idx]

    def get_v_cache(self, layer_idx):
        return self.v_caches[layer_idx]