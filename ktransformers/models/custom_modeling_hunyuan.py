"""
Custom Hunyuan model implementation for KTransformers with optimized inference
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Optional, Tuple, Union
import torch.utils.checkpoint
import numpy as np
import os
from datetime import datetime

from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.models.custom_cache import KHunYuanCache
from ktransformers.models.modeling_hunyuan import HunYuanModel, HunYuanPreTrainedModel
from ktransformers.models.configuration_hunyuan import HunYuanConfig
from ktransformers.operators.flashinfer_batch_prefill_wrapper import flashInferAttn

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.bfloat16)

# Simple debug tensor recording
DEBUG_TENSORS = {}

def save_debug_tensors():
    if not DEBUG_TENSORS:
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"hunyuan_debug_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    
    for key, tensor in DEBUG_TENSORS.items():
        tensor_cpu = tensor.cpu().detach()
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_cpu = tensor_cpu.float()
        array = tensor_cpu.numpy()
        np.save(os.path.join(dir_name, f"{key}.npy"), array)
        print(f"Saved {key}: shape={array.shape}")
    
    print(f"Saved {len(DEBUG_TENSORS)} tensors to: {dir_name}")
    return dir_name

try:
    import flashinfer
except ImportError:
    flashinfer = None

class KHunYuanMoEV1ForCausalLM(HunYuanPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    cache: KHunYuanCache
    use_cuda_graph = False
    
    def __init__(
        self,
        config: HunYuanConfig,
        cache = None,
    ):
        # ALWAYS print this to verify our file is being used
        print("=" * 80)
        print("[INIT] KHunYuanMoEV1ForCausalLM from custom_modeling_hunyuan.py is being initialized!")
        print("=" * 80)
        
        super().__init__(config)
        self.model = HunYuanModel(config)
        self.config = config
        self.cache = cache
        self.vocab_size = config.vocab_size
        # Don't create new lm_head weights - use reference to embed_tokens.weight
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Critical: Tie weights to embed_tokens after creation
        self.lm_head.weight = self.model.embed_tokens.weight
        self.attn = [None] * 100
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def init_wrapper(self, use_cuda_graph, device, max_batch_token, max_batch_size, max_pages, cuda_graph_idx = 0):
        if flashinfer:
            self.attn[cuda_graph_idx] = flashInferAttn(
                use_cuda_graph=use_cuda_graph, 
                max_batch_token=max_batch_token, 
                max_batch_size=max_batch_size, 
                max_pages=max_pages, 
                device=device
            )

    def flash_infer_attn_plan(self, batch: ForwardBatchInput, bsz_tensors, num_tokens_tensors,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        causal: bool,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        cuda_graph_idx: int = 0
        ):
        """Plan flashinfer attention computation for the batch"""
        minibatch = batch.minibatch
        if self.attn[cuda_graph_idx] is not None:
            self.attn[cuda_graph_idx].plan(
                minibatch.q_indptr, 
                minibatch.kv_indptr, 
                minibatch.kv_indices, 
                minibatch.kv_last_page_len, 
                bsz_tensors, 
                num_tokens_tensors, 
                num_q_heads, 
                num_kv_heads, 
                head_dim, 
                page_size, 
                causal=causal, 
                q_data_type=q_data_type, 
                kv_data_type=kv_data_type
            )

    def batch_embeddings(self, batch: ForwardBatchInput, device="cuda:0"):
        features = []
        for i in range(batch.batch_size):
            tokens = batch.minibatch.tokens.contiguous()
            
            # Step-by-step embedding processing for debugging
            tokens_cpu = tokens.to(torch.device('cpu'))
            embed_output = self.model.embed_tokens(tokens_cpu)
            
            
            # Convert dtype and device
            feature = embed_output.to(torch.bfloat16).to(device=device)
            
            
            features.append(feature)
        return features

    def forward(
        self,
        batch: ForwardBatchInput | None = None,
        features: List[torch.Tensor] | None = None,
        bsz_tensors: torch.Tensor | None = None,
        num_tokens_tensors: torch.Tensor | None = None,
        page_idx: torch.Tensor | None = None,
        page_offset: torch.Tensor | None = None,
        cuda_graph_idx: int | None = 0
    ) -> ForwardBatchOutput:
        current_stream = torch.cuda.current_stream()
        forward_batch_output = ForwardBatchOutput()
        
        hidden_states = features[0]

        
        
        if flashinfer and self.attn[cuda_graph_idx] is not None:
            self.attn[cuda_graph_idx].calc_batch_indices(hidden_states.shape[0])
        
        with torch.cuda.stream(current_stream):
            # Initialize residual - will be set properly in each layer
            residual = None
            
            # Register layer 0 hooks for detailed tensor tracking
            if not hasattr(self, '_layer0_hooks_registered'):
                self._layer0_hooks_registered = True
                layer0 = self.model.layers[0]

            for i, decode_layer in enumerate(self.model.layers):
                # Handle device transfer if needed
                if hasattr(self.model, 'transfer_map') and self.model.transfer_map is not None and i in self.model.transfer_map:
                    prev_stream = torch.cuda.current_stream()
                    cur_device = self.model.transfer_map[i]
                    
                    if not hasattr(self.model, 'stream_device_map'):
                        self.model.stream_device_map = {}
                    
                    if cur_device not in self.model.stream_device_map:
                        self.model.stream_device_map[cur_device] = torch.cuda.Stream(cur_device)
                    
                    torch.cuda.set_device(cur_device)
                    self.model.stream_device_map[cur_device].wait_stream(prev_stream)
                    torch.cuda.set_stream(self.model.stream_device_map[cur_device])
                    hidden_states = hidden_states.to(self.model.transfer_map[i], non_blocking=True)
                    
                    if batch and batch.minibatch.position_ids is not None:
                        batch.minibatch.position_ids = batch.minibatch.position_ids.to(
                            self.model.transfer_map[i], non_blocking=True
                        )
                
                # Apply layer normalization
                if hasattr(decode_layer, 'input_layernorm'):
                    if num_tokens_tensors is not None:
                        # Save current hidden_states as residual before normalization
                        # This matches standard HunYuan behavior
                        residual = hidden_states
                        
                        # KHunYuanRMSNorm now only does normalization (no residual handling)
                        hidden_states = decode_layer.input_layernorm(
                            hidden_states, batch_size_tensor=num_tokens_tensors
                        )

                    else:
                        # Standard path without batch_size_tensor
                        residual = hidden_states
                        hidden_states = decode_layer.input_layernorm(hidden_states)
                
                # Self-attention with CLA support
                # Track KV states for Cross-Layer Attention
                kv_states_for_cla = None
                if hasattr(self, '_layer_kv_states'):
                    # Check if this layer should use CLA (cross-attention)
                    if hasattr(decode_layer.self_attn, 'attention_type') and decode_layer.self_attn.attention_type == 'cross':
                        # Find the source layer for KV states (should be a layer where idx % cla_share_factor == 0)
                        cla_share_factor = getattr(self.config, 'cla_share_factor', 1)
                        source_layer_idx = (i // cla_share_factor) * cla_share_factor
                        if source_layer_idx in self._layer_kv_states:
                            kv_states_for_cla = self._layer_kv_states[source_layer_idx]
                
                # Call attention with optional kv_states for CLA
                # Check if this is KHunYuanAttention (which always returns tuple)
                has_kv_states_param = (hasattr(decode_layer.self_attn, 'forward') and 
                                      'kv_states' in decode_layer.self_attn.forward.__code__.co_varnames)
                
                if has_kv_states_param:
                    # KHunYuanAttention - pass kv_states and expect tuple return
                    attn_result = decode_layer.self_attn(
                        hidden_states,
                        self.cache,
                        position_ids=batch.minibatch.position_ids if batch else None,
                        wrapper=self.attn[cuda_graph_idx] if self.attn[cuda_graph_idx] is not None else None,
                        bsz_tensors=num_tokens_tensors,
                        page_idx=page_idx,
                        page_offset=page_offset,
                        kv_states=kv_states_for_cla
                    )
                    # KHunYuanAttention always returns (attn_output, (key_states, value_states))
                    attn_output, layer_kv_states = attn_result
                    
                    # Store KV states for potential CLA use by later layers
                    if not hasattr(self, '_layer_kv_states'):
                        self._layer_kv_states = {}
                    # Only store KV states from layers where idx % cla_share_factor == 0
                    if hasattr(self.config, 'use_cla') and self.config.use_cla:
                        cla_share_factor = getattr(self.config, 'cla_share_factor', 1)
                        if i % cla_share_factor == 0:
                            self._layer_kv_states[i] = layer_kv_states
                else:
                    # Other attention types - standard call without kv_states
                    attn_output = decode_layer.self_attn(
                        hidden_states,
                        self.cache,
                        position_ids=batch.minibatch.position_ids if batch else None,
                        wrapper=self.attn[cuda_graph_idx] if self.attn[cuda_graph_idx] is not None else None,
                        bsz_tensors=num_tokens_tensors,
                        page_idx=page_idx,
                        page_offset=page_offset
                    )
                
                # Add residual connection after attention (matching standard HunYuan)
                if residual is not None:
                    hidden_states = residual + attn_output
                else:
                    hidden_states = attn_output
                
                # Post-attention layer norm and MLP
                if hasattr(decode_layer, 'post_attention_layernorm'):
                    # Update residual to current hidden_states before post-attention norm
                    # This matches standard HunYuan behavior
                    residual = hidden_states
                    
                    # KHunYuanRMSNorm now only does normalization (no residual handling)
                    hidden_states = decode_layer.post_attention_layernorm(
                        hidden_states, num_tokens_tensors
                    )
                    
                # MLP layer
                if hasattr(decode_layer, 'mlp'):
                    # Keep original 3D tensor format [batch_size, seq_len, hidden_size] for native HunYuan compatibility
                    mlp_output = decode_layer.mlp(
                        hidden_states, num_tokens_tensors, cuda_graph_idx
                    )
                    # Add residual connection after MLP
                    if residual is not None:
                        hidden_states = residual + mlp_output
                    else:
                        hidden_states = mlp_output
            
            # Final layer norm
            hidden_states = self.model.norm(hidden_states)
            
            # Handle dimension conversion for lm_head (expects 2D input)
            if hidden_states.dim() == 3:
                # For 3D tensor: [batch_size, seq_len, hidden_size] -> take last token
                logits = self.lm_head(hidden_states[:, -1, :], num_tokens_tensors)
            else:
                # For 2D tensor: [batch_size, hidden_size] -> already the last token
                logits = self.lm_head(hidden_states, num_tokens_tensors)
            
        forward_batch_output = ForwardBatchOutput()
        forward_batch_output.logits.append(logits)

        return forward_batch_output