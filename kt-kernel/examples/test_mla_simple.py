import math
import random
import os, sys
import time
import subprocess
import platform
import json
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch.nn.init as init
from torch_attention import apply_rotary_pos_emb,DeepseekV2RMSNorm,KDeepSeekV3Cache,DeepseekV3YarnRotaryEmbedding

import torch
from tqdm import tqdm
from torch import nn
"""
"rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
""" 

rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
}
seed = 42  # 你可以选择任何整数作为种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# "rope_theta": 10000
rope_theta = 10000


hidden_size = 7168
num_heads = 128
kv_lora_rank = 512
q_lora_rank = 512
nope_size = 128
rope_size = 64

# page 的个数
page_nums = 10
page_size = 512
layer_num = 10
max_position_embeddings =  163840


warm_up_iter = 1000
test_iter = 1000

q_len = 200
his_kv_len = 128

bsz_tensors=torch.tensor([1])

softmax_scale = (nope_size + rope_size) ** -0.5
# 1代表的是压缩的kv的头数
k_caches = torch.randn(layer_num,page_nums, page_size,1, kv_lora_rank + rope_size).to(torch.float16)
kv_cache = KDeepSeekV3Cache(page_size=page_size, kv_lora_rank=kv_lora_rank, k_caches=k_caches)

q_a_layernorm = DeepseekV2RMSNorm(q_lora_rank)

x = torch.randn(q_lora_rank, dtype=torch.float16)*100
print(x)
print(q_a_layernorm(x))

kv_a_layernorm = DeepseekV2RMSNorm(kv_lora_rank)

q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False, dtype=torch.float16)
q_b_proj = nn.Linear(q_lora_rank, num_heads * (nope_size+rope_size) , bias=False, dtype=torch.float16)
kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + rope_size, bias=False, dtype=torch.float16)
kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (nope_size + nope_size), bias=False, dtype=torch.float16)
o_proj = nn.Linear(num_heads * nope_size, hidden_size, bias=False, dtype=torch.float16)

init.normal_(q_a_proj.weight, mean=0.0, std=0.02)
init.normal_(q_b_proj.weight, mean=0.0, std=0.02)
init.normal_(kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
init.normal_(kv_b_proj.weight, mean=0.0, std=0.02)
init.normal_(o_proj.weight, mean=0.0, std=0.02)
# # 这里的权重初始化是为了测试
# # 将权重设置为全 1
# with torch.no_grad():
#     q_a_proj.weight.fill_(1.0)
#     q_b_proj.weight.fill_(1.0)
#     kv_a_proj_with_mqa.weight.fill_(1.0)
#     kv_b_proj.weight.fill_(1.0)
#     o_proj.weight.fill_(1.0)

q_absorb = torch.randn(num_heads, nope_size, kv_lora_rank, dtype=torch.float16)
out_absorb = torch.randn(num_heads, nope_size, kv_lora_rank, dtype=torch.float16)

rotary_emb = DeepseekV3YarnRotaryEmbedding(
    rope_size,
    max_position_embeddings=max_position_embeddings,
    scaling_factor=rope_scaling["factor"],
    base=rope_theta,
    beta_fast=rope_scaling["beta_fast"],
    beta_slow=rope_scaling["beta_slow"],
    mscale=rope_scaling["mscale"],
    mscale_all_dim=rope_scaling["mscale_all_dim"],
    original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
)
# 构造一个q_len 长度的输入 hidden_states, 对应的历史 kv_indptr 是[0:bsz]
# kv_indices 是[0:bsz]，page_idx=[0:bsz], page_offset=[his_kv_len:q_len+his_kv_len]
# last_page_len = [q_len+his_kv_len,...] layer_idx = 1
# position_ids = [his_kv_len:q_len+his_kv_len]
hidden_states = torch.randn(q_len, hidden_size, dtype=torch.float16)
q_indptr = torch.tensor([0,q_len]).to(torch.int32)
kv_indptr = torch.tensor(range(0, bsz_tensors[0] + 1)).to(torch.int32)
kv_indices = torch.tensor(range(0, bsz_tensors[0])).to(torch.int32)
page_idx = torch.tensor(range(0, bsz_tensors[0])).to(torch.int32)
page_offset = torch.tensor(range(his_kv_len, his_kv_len + q_len)).to(torch.int32)
last_page_len = torch.tensor([q_len+his_kv_len]*bsz_tensors[0], device=hidden_states.device)
position_ids = torch.tensor(range(his_kv_len, his_kv_len + q_len)).to(torch.int32)


# 按照行创建 mask [q_len,his_kv_len+q_len]
attention_masks = torch.zeros((q_len, his_kv_len + q_len), dtype=torch.float16)
for i in range(q_len):
    attention_masks[i, i + his_kv_len + 1: i + his_kv_len + q_len] = -65504.0


def torch_attn(hidden_states: torch.Tensor,
                kv_cache: KDeepSeekV3Cache,
                position_ids: torch.Tensor,
                page_idx: torch.Tensor,
                page_offset: torch.Tensor,
                attention_masks: Optional[list[torch.Tensor]] = None,
                q_indptr: Optional[torch.Tensor] = None,
                kv_indices: Optional[torch.Tensor] = None,
                kv_indptr: Optional[torch.Tensor] = None,
                bsz_tensors: Optional[torch.Tensor] = None,
                last_page_len: Optional[torch.Tensor] = None,
                layer_idx: Optional[int] = None,
                ):
    global out_absorb
    global q_absorb
    # range bsz_tensors
    final_attention_output = torch.tensor([], device=hidden_states.device)
    for i in range(bsz_tensors[0]):
        print("page_idx", page_idx)
        print("page_offset", page_offset)
        print("q_indptr", q_indptr)
        print("kv_indices", kv_indices)
        print("kv_indptr", kv_indptr)

        batch_num_tokens_tensors = q_indptr[i+1] - q_indptr[i]
        batch_last_page_len = last_page_len[i]
        # kv_total_len is kv_len, batch_compressed_kv is compressed_kv, batch_k_pe is k_pe
        batch_page_idx = page_idx[q_indptr[i]:q_indptr[i+1]]
        print('batch_page_idx',batch_page_idx)
        batch_page_offset = page_offset[q_indptr[i]:q_indptr[i+1]]
        # kv_page_nums is the number of pages for the current batch
        kv_page_nums = kv_indptr[i+1] - kv_indptr[i]
        # kv_total_len is the total length of the kv cache for the current batch (kv_len for algorithm)
        kv_total_len = kv_page_nums * page_size
        if batch_last_page_len is not None:
            kv_total_len = kv_total_len - (page_size - batch_last_page_len)
        # print(f"kv_total_len's shape {kv_total_len.shape}")
        # kv_index is the index of the kv cache pages for the current batch
        kv_index = kv_indices[kv_indptr[i]:kv_indptr[i+1]]
        # we can index [kv_index, page_offset_indices] to get the kv cache for the current batch
        # from q_indptr[i] to q_indptr[i+1] is the range of the current batch
        batch_hidden_states = hidden_states[q_indptr[i]:q_indptr[i+1]]
        batch_position_ids = position_ids[q_indptr[i]:q_indptr[i+1]]
        q_len, _ = batch_hidden_states.size()
        # print("q_len -> ", q_len)
        q_lora = q_a_proj(batch_hidden_states)
        print('q_a_proj',q_a_proj.weight)
        print('q_lora',q_lora)
        
        q = q_b_proj(q_a_layernorm(q_lora))
        print('q_b_proj',q_b_proj.weight)
        # for v3, bsz, q_len, num_heads(128), qk_head_dim(192=128(nope)+64(rope))
        q = q.view(q_len, num_heads, nope_size+rope_size)
        # q_nope is [q_len, num_heads(128), qk_nope_head_dim(128)]
        # q_pe is [q_len, num_heads(128), qk_rope_head_dim(64)]
        q_nope, q_pe = torch.split(
            q, [nope_size, rope_size], dim=-1
        )
        print('q_nope',q_nope)
        print('q_pe',q_pe)
        # compressed_kv is [q_len, kv_lora_rank(512) + rope(64)]
        compressed_kv = kv_a_proj_with_mqa(batch_hidden_states)
        # compressed_kv is [q_len, kv_lora_rank(512)], k_pe is [q_len, rope(64)]
        compressed_kv, k_pe = torch.split(
            compressed_kv, [kv_lora_rank, rope_size], dim=-1
        )
        compressed_kv = compressed_kv.contiguous()
        compressed_kv = kv_a_layernorm(compressed_kv)
        # k_pe is [q_len, 1, qk_rope_head_dim(64)]
        print('compressed_kv ',compressed_kv)
        print('k_pe ',k_pe)
        k_pe = k_pe.view(q_len, 1, rope_size)
        # compressed_kv is [q_len, 1, kv_lora_rank(512)]
        compressed_kv = compressed_kv.view(q_len, 1, kv_lora_rank)
        
        cos, sin = rotary_emb(q_pe, batch_position_ids)
        # print(f"q_pe shape{q_pe.shape}, k_pe shape {k_pe.shape}")
        q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=1)
        q_pe = q_pe.squeeze(0)
        # q_pe is [num_heads(128), q_len, qk_rope_head_dim(64)]
        q_pe.transpose_(0, 1)            
        if kv_cache is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "page_idx": batch_page_idx, "page_offset": batch_page_offset}  # Specific to RoPE models
            compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, layer_idx, batch_page_idx, batch_page_offset, cache_kwargs)
            compressed_kv = compressed_kv_with_k_pe [:, :, :, :kv_lora_rank].view(-1, page_size, kv_lora_rank)
            k_pe = compressed_kv_with_k_pe [:, :, :, kv_lora_rank:].view(-1, page_size, rope_size)
        # q_absorb is [num_heads(128), qk_nope_head_dim(128), kv_lora_rank(512)]
        # out_absorb is [num_heads(128), kv_lora_rank(512), v_head_dim(128)] v_head_dim is also the nope dim
        # q_absorb, out_absorb = get_absorbed()
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
            if kv_total_len > page_size:
                tmp_compressed_kv = compressed_kv[page_index, 0:page_size, :]
                tmp_k_pe = k_pe[page_index, 0:page_size, :]
                if batch_compressed_kv is None or batch_k_pe is None:
                    batch_compressed_kv = tmp_compressed_kv
                    batch_k_pe = tmp_k_pe
                else: 
                    batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
                    batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
                kv_total_len -= page_size
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
        pe_weights = torch.matmul(q_pe,batch_k_pe.mT)
        print('pe_weights',pe_weights)
        attention_weights = (pe_weights + torch.matmul(q_nope, batch_compressed_kv.mT)) * softmax_scale
        # attention_weights is [num_heads(128), q_len, k_len]
        
        # attention_weights = attention_weights.transpose(0,1).unsqueeze(0).squeeze(-1).expand(q_len,-1,-1).transpose(0,1)
        
        # attention_masks[i] is [q_len, k_len]
        
        attention_weights = (attention_weights + attention_masks[i])
        # attention_weights shape is [num_heads(128), q_len, k_len]
        attention_weights = nn.functional.softmax(attention_weights,dim=-1,dtype=torch.float16).to(q_pe.dtype)
        attn_output = torch.matmul(attention_weights, batch_compressed_kv) # [num_heads(128),q_len, lora_rank(512)]
        # out_absorb shape is [num_heads(128), kv_lora_rank(512), v_head_dim(128)]
        out_absorb = out_absorb.transpose(1,2)
        # q for q_len, n for num_heads, h for v_head_dim, v for kv_lora_rank
        attn_output = torch.matmul(attn_output, out_absorb) # [num_heads(128), q_len, v_head_dim(128)]
        attn_output = attn_output.transpose(0, 1) # [q_len, num_heads(128), v_head_dim(128)]
        attn_output = attn_output.reshape(q_len, num_heads * nope_size)
        attn_output = o_proj(attn_output)
        final_attention_output = torch.cat((final_attention_output, attn_output), dim=0)
    return final_attention_output



def torch_attn_for_test(hidden_states,kv_cache,):
    pass

def test_mla_simple():
    result = torch_attn(
        hidden_states,
        kv_cache,
        position_ids,
        page_idx,
        page_offset,
        attention_masks=attention_masks,
        q_indptr=q_indptr,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        bsz_tensors=bsz_tensors,
        last_page_len=last_page_len,
        layer_idx=1
    )
    print(result.shape)
    print(result)
    
test_mla_simple()