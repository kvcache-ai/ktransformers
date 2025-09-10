'''
Description  :  
Author       : Boxin Zhang
Version      : 0.2.5
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from torch import nn
from ktransformers.models.modeling_deepseek import DeepseekV2Attention, apply_rotary_pos_emb
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeAttention
from ktransformers.models.modeling_qwen3_moe import Qwen3MoeAttention
from ktransformers.models.modeling_smallthinker import SmallthinkerAttention
from ktransformers.models.modeling_glm4_moe import Glm4MoeAttention
from ktransformers.models.modeling_qwen3_next import Qwen3NextGatedDeltaNet
from typing import Optional, Tuple
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.custom_loader import GGUFLoader
import logging
from transformers.configuration_utils import PretrainedConfig
from flashinfer import BatchMLAPagedAttentionWrapper
from ktransformers.operators.flashinfer_batch_prefill_wrapper import flashInferAttn
from ktransformers.models.custom_cache import KDeepSeekV3Cache, KGQACache
logger = logging.getLogger("attention")

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class flashinfer_attn(BaseInjectedModule, DeepseekV2Attention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.


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
            #del self.orig_module.kv_b_proj
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return q_absorb, out_absorb
    

    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KDeepSeekV3Cache,
                position_ids: torch.Tensor,
                wrapper: BatchMLAPagedAttentionWrapper,
                num_tokens_tensors: torch.Tensor,
                page_idx: torch.Tensor,
                page_offset: torch.Tensor,
                ):
        q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states, num_tokens_tensors)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states, num_tokens_tensors), num_tokens_tensors), num_tokens_tensors)
        q = q.view(q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states, num_tokens_tensors)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = compressed_kv.contiguous()
        compressed_kv = self.kv_a_layernorm(compressed_kv, num_tokens_tensors)
        k_pe = k_pe.view(q_len, 1, self.qk_rope_head_dim)
        compressed_kv = compressed_kv.view(q_len, 1, self.kv_lora_rank)
        
        cos, sin = self.rotary_emb(q_pe, position_ids.unsqueeze(0))
        q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=2)
        q_pe = q_pe.squeeze(0)
        if kv_cache is not None:
            
            # page_idx, page_offset = kv_cache.get_page_table(position_ids, q_indptr, kv_indptr, kv_indices)
            cache_kwargs = {"sin": sin, "cos": cos, "page_idx": page_idx, "page_offset": page_offset}  # Specific to RoPE models
            compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, self.layer_idx, page_idx, page_offset, cache_kwargs)
            compressed_kv = compressed_kv_with_k_pe [:, :, :, :self.kv_lora_rank].view(-1, kv_cache.page_size, self.kv_lora_rank)
            k_pe = compressed_kv_with_k_pe [:, :, :, self.kv_lora_rank:].view(-1, kv_cache.page_size, self.qk_rope_head_dim)
            
        q_absorb, out_absorb = self.get_absorbed()
        q_nope = q_nope.transpose(0, 1) # q_len is 1, no GPU overhead, same below
        q_nope = torch.matmul(q_nope, q_absorb) # batched MM
        q_nope = q_nope.transpose(0, 1)
        # q_nope.squeeze_(1)
        # q_pe.squeeze_(1)

        attn_output = wrapper.run(q_nope, q_pe, compressed_kv, k_pe).view(q_len, self.num_heads, self.kv_lora_rank)
        attn_output = attn_output.transpose(0, 1)
        attn_output = torch.matmul(attn_output, out_absorb.mT) # [self.num_heads, q_len, self.v_head_dim]
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output, num_tokens_tensors)
        return attn_output

class KQwen2MoeAttention(BaseInjectedModule, Qwen2MoeAttention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.


    # Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KGQACache,
                position_ids: torch.Tensor,
                wrapper: flashInferAttn,
                bsz_tensors: torch.Tensor,
                page_idx: torch.Tensor,
                page_offset: torch.Tensor,
                ):
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, bsz_tensors)
        key_states = self.k_proj(hidden_states, bsz_tensors)
        value_states = self.v_proj(hidden_states, bsz_tensors)


        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)
        
        cos, sin = self.rotary_emb(value_states.unsqueeze(0), position_ids.unsqueeze(0))
        query_states, key_states = self.apply_rotary_pos_emb(query_states.unsqueeze(0), key_states.unsqueeze(0), cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            q_len, self.num_key_value_heads, self.head_dim
        )

        k_cache = kv_cache.get_k_cache(self.layer_idx)
        v_cache = kv_cache.get_v_cache(self.layer_idx)


        attn_output = wrapper.forward(query_states, k_cache, v_cache, key_states, value_states)
  

        attn_output = self.o_proj(attn_output.view(q_len, self.num_heads * self.head_dim), bsz_tensors)

        return attn_output

class KQwen3MoeAttention(BaseInjectedModule, Qwen3MoeAttention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.


    # Copied from transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KGQACache,
                position_ids: torch.Tensor,
                wrapper: flashInferAttn,
                bsz_tensors: torch.Tensor,
                page_idx: torch.Tensor,
                page_offset: torch.Tensor,
                ):
        q_len, _ = hidden_states.size()

        bsz_tensors_q = bsz_tensors * self.num_heads
        bsz_tensors_kv = bsz_tensors * self.num_key_value_heads

        query_states = self.q_norm(self.q_proj(hidden_states, bsz_tensors), bsz_tensors_q)
        key_states = self.k_norm(self.k_proj(hidden_states, bsz_tensors), bsz_tensors_kv)
        value_states = self.v_proj(hidden_states, bsz_tensors)


        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)
        
        cos, sin = self.rotary_emb(value_states.unsqueeze(0), position_ids.unsqueeze(0))
        query_states, key_states = self.apply_rotary_pos_emb(query_states.unsqueeze(0), key_states.unsqueeze(0), cos, sin, unsqueeze_dim=2)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            q_len, self.num_key_value_heads, self.head_dim
        )

        k_cache = kv_cache.get_k_cache(self.layer_idx)
        v_cache = kv_cache.get_v_cache(self.layer_idx)


        attn_output = wrapper.forward(query_states, k_cache, v_cache, key_states, value_states)
  

        attn_output = self.o_proj(attn_output.view(q_len, self.num_heads * self.head_dim), bsz_tensors)

        return attn_output


class deepseek_torch_attn(BaseInjectedModule, DeepseekV2Attention):
    def __init__(self,
                    key: str,
                    gguf_loader : GGUFLoader,
                    config: PretrainedConfig,
                    orig_module: nn.Module,
                    prefill_device: str = "cuda",
                    generate_device: str = "cuda",
                    chunck_size: int = 1000,
                    **kwargs):
            BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
            self.orig_module.__init__(orig_module.config,
                orig_module.layer_idx)
            self.chunck_size = chunck_size # TODO, generate chunck_size automatically.


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
            #del self.orig_module.kv_b_proj
        q_absorb = self.q_absorb.weight.view(self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank)
        out_absorb = self.out_absorb.weight.view(self.num_heads, self.v_head_dim, self.kv_lora_rank)
        return q_absorb, out_absorb
    


    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KDeepSeekV3Cache,
                position_ids: torch.Tensor,
                wrapper: None,
                num_tokens_tensors: torch.Tensor,
                page_idx: torch.Tensor,
                page_offset: torch.Tensor,
                attention_masks: Optional[list[torch.Tensor]] = None,
                q_indptr: Optional[torch.Tensor] = None,
                kv_indices: Optional[torch.Tensor] = None,
                kv_indptr: Optional[torch.Tensor] = None,
                bsz_tensors: Optional[torch.Tensor] = None,
                last_page_len: Optional[torch.Tensor] = None,
                ):
        # range bsz_tensors
        final_attention_output = torch.tensor([], device=hidden_states.device)
        for i in range(bsz_tensors[0]):
            batch_num_tokens_tensors = q_indptr[i+1] - q_indptr[i]
            batch_last_page_len = last_page_len[i]
            # kv_total_len is kv_len, batch_compressed_kv is compressed_kv, batch_k_pe is k_pe
            batch_page_idx = page_idx[q_indptr[i]:q_indptr[i+1]]
            batch_page_offset = page_offset[q_indptr[i]:q_indptr[i+1]]
            # kv_page_nums is the number of pages for the current batch
            kv_page_nums = kv_indptr[i+1] - kv_indptr[i]
            # kv_total_len is the total length of the kv cache for the current batch (kv_len for algorithm)
            kv_total_len = kv_page_nums * kv_cache.page_size
            if batch_last_page_len is not None:
                kv_total_len = kv_total_len - (kv_cache.page_size - batch_last_page_len)
            # print(f"kv_total_len's shape {kv_total_len.shape}")
            # kv_index is the index of the kv cache pages for the current batch
            kv_index = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
            # we can index [kv_index, page_offset_indices] to get the kv cache for the current batch
            # from q_indptr[i] to q_indptr[i+1] is the range of the current batch
            batch_hidden_states = hidden_states[q_indptr[i]:q_indptr[i+1]]
            batch_position_ids = position_ids[q_indptr[i]:q_indptr[i+1]]
            q_len, _ = batch_hidden_states.size()
            # print("q_len -> ", q_len)

            if self.q_lora_rank is None:
                q = self.q_proj(batch_hidden_states, batch_num_tokens_tensors)
            else:
                q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(batch_hidden_states, batch_num_tokens_tensors), batch_num_tokens_tensors), batch_num_tokens_tensors)
            # for v3, bsz, q_len, num_heads(128), qk_head_dim(192=128(nope)+64(rope))
            q = q.view(q_len, self.num_heads, self.q_head_dim)
            # q_nope is [q_len, num_heads(128), qk_nope_head_dim(128)]
            # q_pe is [q_len, num_heads(128), qk_rope_head_dim(64)]
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            # compressed_kv is [q_len, kv_lora_rank(512) + rope(64)]
            compressed_kv = self.kv_a_proj_with_mqa(batch_hidden_states, batch_num_tokens_tensors)
            # compressed_kv is [q_len, kv_lora_rank(512)], k_pe is [q_len, rope(64)]
            compressed_kv, k_pe = torch.split(
                compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            compressed_kv = compressed_kv.contiguous()
            compressed_kv = self.kv_a_layernorm(compressed_kv, batch_num_tokens_tensors)
            # k_pe is [q_len, 1, qk_rope_head_dim(64)]
            k_pe = k_pe.view(q_len, 1, self.qk_rope_head_dim)
            # compressed_kv is [q_len, 1, kv_lora_rank(512)]
            compressed_kv = compressed_kv.view(q_len, 1, self.kv_lora_rank)
            
            cos, sin = self.rotary_emb(q_pe, batch_position_ids.unsqueeze(0))
            # print(f"q_pe shape{q_pe.shape}, k_pe shape {k_pe.shape}")
            q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=2)
            q_pe = q_pe.squeeze(0)
            # q_pe is [num_heads(128), q_len, qk_rope_head_dim(64)]
            q_pe.transpose_(0, 1)            
            if kv_cache is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "page_idx": batch_page_idx, "page_offset": batch_page_offset}  # Specific to RoPE models
                compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, self.layer_idx, batch_page_idx, batch_page_offset, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe [:, :, :, :self.kv_lora_rank].view(-1, kv_cache.page_size, self.kv_lora_rank)
                k_pe = compressed_kv_with_k_pe [:, :, :, self.kv_lora_rank:].view(-1, kv_cache.page_size, self.qk_rope_head_dim)
            # q_absorb is [num_heads(128), qk_nope_head_dim(128), kv_lora_rank(512)]
            # out_absorb is [num_heads(128), kv_lora_rank(512), v_head_dim(128)] v_head_dim is also the nope dim
            q_absorb, out_absorb = self.get_absorbed()
            # q_nope is [num_heads(128), q_len, qk_nope_head_dim(128)]
            q_nope = q_nope.transpose(0, 1) # q_len is 1, no GPU overhead, same below
            # q_nope is [num_heads(128), q_len, kv_lora_rank(512)]
            q_nope = torch.matmul(q_nope, q_absorb) # batched MM

            # # q_nope is [q_len, num_heads(128), kv_lora_rank(512)]
            # q_nope = q_nope.transpose(0, 1)

            # we need to index out the compressed_kv and k_pe for the current batch
            batch_compressed_kv = None
            batch_k_pe = None
            for page_index in kv_index:
                if kv_total_len > kv_cache.page_size:
                    tmp_compressed_kv = compressed_kv[page_index, 0:kv_cache.page_size, :]
                    tmp_k_pe = k_pe[page_index, 0:kv_cache.page_size, :]
                    if batch_compressed_kv is None or batch_k_pe is None:
                        batch_compressed_kv = tmp_compressed_kv
                        batch_k_pe = tmp_k_pe
                    else: 
                        batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
                        batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
                    kv_total_len -= kv_cache.page_size
                else:
                    tmp_compressed_kv = compressed_kv[page_index, 0:kv_total_len, :]
                    tmp_k_pe = k_pe[page_index, 0:kv_total_len, :]
                    if batch_compressed_kv is None or batch_k_pe is None:
                        batch_compressed_kv = tmp_compressed_kv
                        batch_k_pe = tmp_k_pe
                    else: 
                        batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
                        batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
                    break
            # batch_compressed_kv is [kv_total_len(k_len), kv_lora_rank(512)]
            # batch_k_pe is [kv_total_len(k_len), qk_rope_head_dim(64)]
            attention_weights = (torch.matmul(q_pe,batch_k_pe.mT) + torch.matmul(q_nope, batch_compressed_kv.mT)) * self.softmax_scale
            # attention_weights is [num_heads(128), q_len, k_len]
            
            # attention_weights = attention_weights.transpose(0,1).unsqueeze(0).squeeze(-1).expand(q_len,-1,-1).transpose(0,1)
            
            # attention_masks[i] is [q_len, k_len]
            
            attention_weights = (attention_weights + attention_masks[i])
            # attention_weights shape is [num_heads(128), q_len, k_len]
            attention_weights = nn.functional.softmax(attention_weights,dim=-1,dtype=torch.float32).to(q_pe.dtype)
            attn_output = torch.matmul(attention_weights, batch_compressed_kv) # [num_heads(128),q_len, lora_rank(512)]
            # out_absorb shape is [num_heads(128), kv_lora_rank(512), v_head_dim(128)]
            out_absorb = out_absorb.transpose(1,2)
            # q for q_len, n for num_heads, h for v_head_dim, v for kv_lora_rank
            attn_output = torch.matmul(attn_output, out_absorb) # [num_heads(128), q_len, v_head_dim(128)]
            attn_output = attn_output.transpose(0, 1) # [q_len, num_heads(128), v_head_dim(128)]
            attn_output = attn_output.reshape(q_len, self.num_heads * self.v_head_dim)
            attn_output = self.o_proj(attn_output, batch_num_tokens_tensors)
            final_attention_output = torch.cat((final_attention_output, attn_output), dim=0)
        return final_attention_output

class KSmallthinkerAttention(BaseInjectedModule, SmallthinkerAttention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KGQACache,
                freqs_cis: torch.Tensor,
                wrapper: flashInferAttn,
                bsz_tensors: torch.Tensor,
                position_ids: torch.Tensor = None,
                ):

        if self.use_qk_norm:
            raise NotImplementedError("use_qk_norm is not implemented yet")

        q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states, bsz_tensors)
        key_states = self.k_proj(hidden_states, bsz_tensors)
        value_states = self.v_proj(hidden_states, bsz_tensors)

        query_states = query_states.view(q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)
        
        # cos, sin = freqs_cis
        """
        print(query_states.shape)
        print(key_states.shape)
        print(cos.shape)
        print(sin.shape)
        """
        if freqs_cis:  
            cos, sin = freqs_cis
            query_states, key_states = self.apply_rotary_pos_emb(query_states.unsqueeze(0), key_states.unsqueeze(0), cos, sin, unsqueeze_dim=2)



        query_states = query_states.view(q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        k_cache = kv_cache.get_k_cache(self.layer_idx)
        v_cache = kv_cache.get_v_cache(self.layer_idx)


        attn_output = wrapper.forward(query_states, k_cache, v_cache, key_states, value_states)
  

        attn_output = self.o_proj(attn_output.view(q_len, self.num_attention_heads * self.head_dim), bsz_tensors)

        return attn_output


    

class KGlm4MoeAttention(BaseInjectedModule, Glm4MoeAttention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
        unsqueeze_dim=2
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Keep half or full tensor for later concatenation
        cos = freqs_cis[0]
        sin = freqs_cis[1]
        rotary_dim = cos.shape[-1]

        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        # Apply rotary embeddings on the first half or full tensor
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

        # Concatenate back to full shape
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KGQACache,
                freqs_cis: torch.Tensor,
                wrapper: flashInferAttn,
                bsz_tensors: torch.Tensor,
                position_ids: torch.Tensor = None,
                ):

        q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states, bsz_tensors)
        key_states = self.k_proj(hidden_states, bsz_tensors)
        value_states = self.v_proj(hidden_states, bsz_tensors)


        if self.use_qk_norm:
            query_states = self.q_norm(query_states, bsz_tensors)
            key_states = self.k_norm(key_states, bsz_tensors)


        query_states = query_states.view(q_len, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.config.num_key_value_heads, self.head_dim)
        
        # cos, sin = freqs_cis
        """
        print(query_states.shape)
        print(key_states.shape)
        print(cos.shape)
        print(sin.shape)
        """
        if freqs_cis is not None:  
            query_states, key_states = self.apply_rotary_pos_emb(query_states.unsqueeze(0), key_states.unsqueeze(0), freqs_cis)



        query_states = query_states.view(q_len, self.config.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.config.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.config.num_key_value_heads, self.head_dim)

        k_cache = kv_cache.get_k_cache(self.layer_idx)
        v_cache = kv_cache.get_v_cache(self.layer_idx)


        attn_output = wrapper.forward(query_states, k_cache, v_cache, key_states, value_states)
  

        attn_output = self.o_proj(attn_output.view(q_len, self.config.num_attention_heads * self.head_dim), bsz_tensors)

        return attn_output
    
from ktransformers.models.modeling_qwen3_next import apply_mask_to_padding_states
import torch.nn.functional as F

from ktransformers.models.modeling_qwen3_next import Qwen3NextAttention

class KQwen3NextAttention(BaseInjectedModule, Qwen3NextAttention):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self,
                hidden_states: torch.Tensor,
                kv_cache: KGQACache,
                freqs_cis: torch.Tensor,
                position_ids: torch.Tensor,
                wrapper: flashInferAttn,
                bsz_tensors: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ):

        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states, bsz_tensors)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(q_len, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(q_len, -1)

        key_states = self.k_proj(hidden_states, bsz_tensors)

        query_states = self.q_norm(query_states, bsz_tensors)
        key_states = self.k_norm(key_states, bsz_tensors)


        value_states = self.v_proj(hidden_states, bsz_tensors)

        if freqs_cis:  
            cos, sin = freqs_cis
            query_states, key_states = self.apply_rotary_pos_emb(query_states.unsqueeze(0), key_states.unsqueeze(0), cos, sin, unsqueeze_dim=2)



        query_states = query_states.view(q_len, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        k_cache = kv_cache.get_k_cache(self.layer_idx)
        v_cache = kv_cache.get_v_cache(self.layer_idx)

        attn_output = wrapper.forward(query_states, k_cache, v_cache, key_states, value_states)
  
        attn_output = attn_output.reshape(q_len, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)


        attn_output = self.o_proj(attn_output.view(q_len, self.num_attention_heads * self.head_dim), bsz_tensors)

        return attn_output


class KQwen3NextGatedDeltaNet(BaseInjectedModule, Qwen3NextGatedDeltaNet):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 prefill_device: str = "cuda",
                 generate_device: str = "cuda",
                 chunck_size: int = 1000,
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, prefill_device, **kwargs)
        self.orig_module.__init__(orig_module.config,
            orig_module.layer_idx)
        self.chunck_size = chunck_size # TODO, generate chunck_size automatically.

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_states: Optional[list[torch.Tensor]] = None,
        recurrent_states: Optional[list[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        conv_state = conv_states[self.layer_idx] if conv_states is not None else None
        recurrent_state = (
            recurrent_states[self.layer_idx] if recurrent_states is not None else None
        )

        use_precomputed_states = (
            conv_state is not None
            and recurrent_state is not None
            and seq_len == 1
        )

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # 2. Convolution sequence transformation
            # NOTE: the conv state is updated in `causal_conv1d_update`
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))

            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=conv_state is not None,
                use_qk_l2norm_in_kernel=True,
            )

        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=conv_state is not None,
                use_qk_l2norm_in_kernel=True,
            )

        # Update cache
        recurrent_state = last_recurrent_state

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)

        if conv_state is not None:
            conv_states[self.layer_idx] = conv_state
        if recurrent_state is not None:
            recurrent_states[self.layer_idx] = recurrent_state

        return output