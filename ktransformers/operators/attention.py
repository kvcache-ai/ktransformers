'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
import warnings
from ktransformers.models.configuration_deepseek import DeepseekV2Config
from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from typing import Optional, Tuple
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import Cache

class DeepseekV2AttentionInjected(BaseInjectedModule, DeepseekV2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)

    def get_absorbed(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].reshape(-1, self.kv_lora_rank)
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].reshape(-1, self.kv_lora_rank)
            self.q_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, 
                                      bias=False, dtype=q_absorb.dtype, device=q_absorb.device)
            self.q_absorb.weight.data = q_absorb
            self.out_absorb = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, 
                                        bias=False, dtype=out_absorb.dtype, device=out_absorb.device)
            self.out_absorb.weight.data = out_absorb
            del self.orig_module.kv_b_proj
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return q_absorb, out_absorb

    def forward_chunck(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv_seq_len = k_pe.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        cos, sin = self.rotary_emb(q_pe, position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            compressed_kv = compressed_kv.unsqueeze(1)
            k_pe, compressed_kv = past_key_value.update(k_pe, compressed_kv, self.layer_idx, cache_kwargs)
            compressed_kv = compressed_kv.squeeze(1)
            #if cache_position is not None:  
            #    compressed_kv = compressed_kv[:,: cache_position[-1] + 1,:]
            #    k_pe = k_pe[:,:,: cache_position[-1] + 1,:]
        q_absorb, out_absorb = self.get_absorbed()

        q_nope = torch.matmul(q_nope, q_absorb)
        attn_weights = (torch.matmul(q_pe, k_pe.mT) + torch.matmul(q_nope, compressed_kv.unsqueeze(-3).mT)) * self.softmax_scale
        """
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        assert attention_mask is not None
        """
        if attention_mask is not None:
            """
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            """
            #causal_mask = attention_mask[:, :, :, : kv_seq_len]
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_pe.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.einsum('bhql,blc->bhqc', attn_weights, compressed_kv)

        attn_output = torch.matmul(attn_output, out_absorb.mT) 

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        chunck_size = 256 # TODO, generate chunck_size automatically.
        
        if q_len <= chunck_size:
            return self.forward_chunck(
                            hidden_states,
                            attention_mask,
                            position_ids,
                            past_key_value,
                            output_attentions,
                            use_cache,
                            cache_position,
                            **kwargs
                        )

        assert output_attentions == False, "output_attentions is not supported when using chunked attention"
        attn_output = None
        cur_idx = 0
        while cur_idx < q_len:
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, cur_idx:min(cur_idx + chunck_size, q_len), ...]
            else:
                chunk_mask = None

            cur_output, _, _ = self.forward_chunck(
                            hidden_states[:, cur_idx:min(cur_idx + chunck_size, q_len), ...],
                            chunk_mask,
                            position_ids[:, cur_idx:min(cur_idx + chunck_size, q_len)],
                            past_key_value,
                            output_attentions,
                            use_cache,
                            cache_position[cur_idx:min(cur_idx + chunck_size, q_len)],
                            **kwargs
                        )
            cur_idx += chunck_size
            if attn_output is None:
                attn_output = cur_output
            else:
                attn_output = torch.cat((attn_output, cur_output), dim=-2)
                
        return attn_output, None, past_key_value
