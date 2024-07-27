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
import transformers
from transformers import Cache, PretrainedConfig
from typing import List, Optional, Dict, Any, Tuple
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
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        Cache.__init__(self)
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
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
        if config.architectures[0] == "DeepseekV2ForCausalLM":
            # key_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, config.qk_rope_head_dim + config.qk_nope_head_dim)
            # value_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, config.v_head_dim)
            key_shape = (max_batch_size, 1, self.max_cache_len, config.qk_rope_head_dim)
            value_shape = (max_batch_size, 1, self.max_cache_len, config.kv_lora_rank)
        else:
            key_shape = cache_shape
            value_shape = cache_shape

        self.past_tokens = []
        self.num_hidden_layers = config.num_hidden_layers
        for _ in range(self.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(key_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(value_shape, dtype=self.dtype, device=device)
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
        #print(cache_position)
        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states
        self.past_tokens[layer_idx] += cache_position.size(0)
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
            self.value_cache[layer_idx].zero_()
