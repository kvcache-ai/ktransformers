import logging
import os,sys 
import time
from typing import Optional
os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
from cpuinfer_ext.kvcache import ggml_type
import torch
from torch import inf, nn
from torch.nn import init
from torch_attention import apply_rotary_pos_emb,DeepseekV2RMSNorm,KDeepSeekV3Cache,DeepseekV3YarnRotaryEmbedding
logger = logging.getLogger("reader")

from gguf.gguf_reader import GGUFReader



def load_fp32_tensor_raw(file_path):
    # return torch.zeros(shape)
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    return tensor

def load_fp16_tensor(file_path, shape=None):
    # return load_fp32_tensor(file_path, shape)
    return load_fp32_tensor_raw(file_path)
    # return torch.zeros(shape)
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=weight_type)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

def load_fp32_tensor(file_path, shape):
    # return torch.zeros(shape)
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

def test_torch():
    torch.set_grad_enabled(False)


    hidden_states_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_input.bin')
    hidden_states_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_input.bin')
    # diff = torch.abs(hidden_states_to_check_prefill - hidden_states_to_check_decode).max()
    # print("hidden_states diff -> ", diff)

    q_lora_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_qlora.bin')
    q_lora_to_check_test_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_qlora_test.bin')
    q_lora_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_qlora.bin')
    q_lora_to_check_test_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_qlora_test.bin')
    # diff = torch.abs(q_lora_to_check_prefill - q_lora_to_check_decode).max()
    # diff_test = torch.abs(q_lora_to_check_prefill - q_lora_to_check_decode).max()
    # print("q_lora max diff -> ", diff)
    # print("q_lora max diff test -> ", diff_test)
    # mae =  torch.mean(torch.abs(q_lora_to_check_prefill - q_lora_to_check_decode))
    # mae_test =  torch.mean(torch.abs(q_lora_to_check_prefill - q_lora_to_check_decode))
    # print("q_lora mae -> ", mae)
    # print("q_lora mae test -> ", mae_test)



    # q_lora_norm = q_a_layernorm(q_lora)
    # q_lora_norm_to_check = load_fp16_tensor('./debug/query_0_tp_0_qlora_norm.bin', q_lora_norm.shape)
    # q_lora_norm_to_check_test = load_fp16_tensor('./debug/query_0_tp_0_qlora_norm_test.bin', q_lora_norm.shape)
    # diff = torch.abs(q_lora_norm - q_lora_norm_to_check).max()
    # mae =  torch.mean(torch.abs(q_lora_norm - q_lora_norm_to_check))
    # diff_test = torch.abs(q_lora_norm - q_lora_norm_to_check_test).max()
    # mae_test =  torch.mean(torch.abs(q_lora_norm - q_lora_norm_to_check_test))
    # print("q_lora_norm diff -> ", diff)
    # print("q_lora_norm mae -> ", mae)
    # print("q_lora_norm diff test -> ", diff_test)
    # print("q_lora_norm mae test -> ", mae_test)
    
    # q = q_b_proj(q_lora_norm)
    # for v3, bsz, qlen, num_heads(128), qk_head_dim(192=128(nope)+64(rope))
    # q = q.view(qlen, num_heads, nope_size+rope_size)
    # q_nope is [qlen, num_heads(128), qk_nope_head_dim(128)]
    # q_pe is [qlen, num_heads(128), qk_rope_head_dim(64)]
    # q_nope, q_pe = torch.split(
    #     q, [nope_size, rope_size], dim=-1
    # )
        
    # compressed_kv is [qlen, kv_lora_rank(512) + rope(64)]
    # compressed_kv = kv_a_proj_with_mqa(batch_hidden_states)
    # compressed_kv is [qlen, kv_lora_rank(512)], k_pe is [qlen, rope(64)]
    # compressed_kv, k_pe = torch.split(
    #     compressed_kv, [kv_lora_rank, rope_size], dim=-1
    # )
    # compressed_kv = compressed_kv.contiguous()


    # compressed_kv_page_0 = compressed_kv[0:page_size, :]
    compressed_kv_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_page_0_kv_lora_rank')
    compressed_kv_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_page_0_kv_lora_rank')
    # diff = torch.abs(compressed_kv_to_check_prefill - compressed_kv_to_check_decode).max()
    # mae =  torch.mean(torch.abs(compressed_kv_to_check_prefill - compressed_kv_to_check_decode)) 
    # print("compressed_kv diff -> ", diff)
    # print("compressed_kv mae -> ", mae)

    # compressed_kv = kv_a_layernorm(compressed_kv)
    # k_pe is [qlen, 1, qk_rope_head_dim(64)]

    # compressed_kv_page_0 = compressed_kv[0:page_size, :]
    compressed_kv_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_page_0_kv_lora_rank_norm')
    compressed_kv_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_page_0_kv_lora_rank_norm')
    # diff = torch.abs(compressed_kv_page_0 - compressed_kv_to_check).max()
    # mae =  torch.mean(torch.abs(compressed_kv_page_0 - compressed_kv_to_check))
    # print("compressed_kv diff norm -> ", diff)
    # print("compressed_kv mae norm -> ", mae)
    
                                                


    # k_pe = k_pe.view(qlen, 1, rope_size)
    # compressed_kv is [qlen, 1, kv_lora_rank(512)]
    # compressed_kv = compressed_kv.view(qlen, 1, kv_lora_rank)
    
    # cos, sin = rotary_emb(q_pe, batch_position_ids)

    # q_nope_check = q_nope.transpose(0, 1) # qlen is 1, no GPU overhead, same below

    # q_nope_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_nope', q_nope_check[0].shape)
    # q_nope_0_to_check_test = load_fp16_tensor('./debug/query_0_tp_0_q_nope_test', q_nope_check[0].shape)
    # diff = torch.abs(q_nope_check[0] - q_nope_0_to_check).max()
    # mae =  torch.mean(torch.abs(q_nope_check[0] - q_nope_0_to_check))
    # diff_test = torch.abs(q_nope_check[0] - q_nope_0_to_check_test).max()
    # mae_test =  torch.mean(torch.abs(q_nope_check[0] - q_nope_0_to_check_test))
    # print("q_nope[0] diff -> ", diff)
    # print("q_nope[0] mae -> ", mae)
    # print("q_nope[0] diff test -> ", diff_test)
    # print("q_nope[0] mae test -> ", mae_test)
    
    # q_pe_nope = q_pe.transpose(0,1)
    q_pe_0_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_q_rope')
    q_pe_0_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_q_rope')
    
    # q_pe_0_to_check_decode_test = load_fp16_tensor('./debug_decode/query_0_tp_0_q_rope_test')
    # q_pe_0_to_check_prefill_test = load_fp16_tensor('./debug_prefill/query_0_tp_0_q_rope_test')

    # q_pe_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_rope_no_rope', q_pe_nope[0].shape)
    # q_pe_0_to_check_test = load_fp16_tensor('./debug/query_0_tp_0_q_rope_no_rope_test', q_pe_nope[0].shape)
    # diff = torch.abs(q_pe_nope[0] - q_pe_0_to_check).max()
    # mae =  torch.mean(torch.abs(q_pe_nope[0] - q_pe_0_to_check))
    # diff_test = torch.abs(q_pe_nope[0] - q_pe_0_to_check_test).max()
    # mae_test =  torch.mean(torch.abs(q_pe_nope[0] - q_pe_0_to_check_test))
    # print("q_pe nope[0] diff -> ", diff)
    # print("q_pe nope[0] mae -> ", mae)
    # print("q_pe nope[0] diff test -> ", diff_test)
    # print("q_pe nope[0] mae test -> ", mae_test)

    # cos_to_check = load_fp32_tensor('./debug/query_0_tp_0_rope_cos', (qlen,32))
    # diff = torch.abs(cos[:,:32]-cos_to_check).max()
    # mae =  torch.mean(torch.abs(cos[:,:32]-cos_to_check))
    # print("cos diff -> ", diff)
    # print("cos mae -> ", mae)
    # sin_to_check = load_fp32_tensor('./debug/query_0_tp_0_rope_sin', (qlen,32))
    # diff = torch.abs(sin[:,:32]-sin_to_check).max()
    # mae =  torch.mean(torch.abs(sin[:,:32]-sin_to_check))
    # print("sin diff -> ", diff)
    # print("sin mae -> ", mae)

    # new_q_pe = q_pe.transpose(0, 1)
    # qa = new_q_pe[:,:,range(0,64,2)]
    # qb = new_q_pe[:,:,range(1,65,2)]
    # q1 = (qa * cos[:,:32] - qb * sin[:,:32])
    # q2 = (qb*cos[:,:32] + qa*sin[:,:32])
    # q1 = (qa * cos_to_check - qb * sin_to_check)
    # q2 = (qb*cos_to_check + qa*sin_to_check)
    # q_new = torch.cat((q1,q2), dim=-1)
    # print(f"q_pe shape{q_pe.shape}, k_pe shape {k_pe.shape}")
    # new_q_pe = torch.zeros_like(q_pe)
    # new_q_pe[:,:,range(0,64,2)] = 1
    # new_q_pe[:,:,range(1,65,2)] = 10
    # q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=1)
    # q_pe = q_pe.squeeze(0)
    # q_pe is [num_heads(128), qlen, qk_rope_head_dim(64)]
    # q_pe.transpose_(0, 1)    

    # diff = torch.abs(q_pe - q_new).max()
    # print("q_pe diff -> ", diff)


    # q_pe_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_rope', q_pe[0].shape)
    # diff = torch.abs(q_pe[0] - q_pe_0_to_check).max()
    # mae =  torch.mean(torch.abs(q_pe[0] - q_pe_0_to_check))
    # print("q_pe[0] diff -> ", diff)
    # print("q_pe[0] mae -> ", mae)

    # diff = torch.abs(q_pe_0_to_check - q_new[0]).max()
    # mae =  torch.mean(torch.abs(q_pe_0_to_check - q_new[0]))
    # print("q_pe[0] 2  diff -> ", diff)
    # print("q_pe[0] 2 mae -> ", mae)

    # if kv_cache is not None:
    #     cache_kwargs = {"sin": sin, "cos": cos, "page_idx": batch_page_idx, "page_offset": batch_page_offset}  # Specific to RoPE models
    #     compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, layer_idx, batch_page_idx, batch_page_offset, cache_kwargs)
    #     compressed_kv = compressed_kv_with_k_pe [:, :, :, :kv_lora_rank].view(-1, page_size, kv_lora_rank)
    #     k_pe = compressed_kv_with_k_pe [:, :, :, kv_lora_rank:].view(-1, page_size, rope_size)
    # # q_absorb is [num_heads(128), qk_nope_head_dim(128), kv_lora_rank(512)]
    # # out_absorb is [num_heads(128), kv_lora_rank(512), v_head_dim(128)] v_head_dim is also the nope dim
    # # q_absorb, out_absorb = get_absorbed()
    # # q_nope is [num_heads(128), qlen, qk_nope_head_dim(128)]
    # q_nope = q_nope.transpose(0, 1) # qlen is 1, no GPU overhead, same below

    # q_nope_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_nope', q_nope[0].shape)
    # diff = torch.abs(q_nope[0] - q_nope_0_to_check).max()
    # mae =  torch.mean(torch.abs(q_nope[0] - q_nope_0_to_check))
    # print("q_nope[0] diff -> ", diff)

    # # q_nope is [num_heads(128), qlen, kv_lora_rank(512)]
    # q_nope = torch.matmul(q_nope, q_absorb) # batched MM

    # k_b_proj_check = load_fp16_tensor('./debug/query_0_tp_0_k_b_lora', (nope_size,kv_lora_rank))
    # diff = torch.abs(q_absorb[0] - k_b_proj_check).max()
    # print("kv b lora weight[0] diff -> ", diff)

    # q_absorb_check = load_fp16_tensor('./debug/query_0_tp_0_q_absorb', (kv_lora_rank,1024))
    # q_absorb_check = q_absorb_check[:,0:qlen].transpose(0,1)
    # diff = torch.abs(q_nope[0] - q_absorb_check).max()
    # mae =  torch.mean(torch.abs(q_nope[0] - q_absorb_check))
    # print("q_nope absorb diff -> ", diff)
    # print("q_nope absorb mae -> ", mae)

    # # q_nope is [qlen, num_heads(128), kv_lora_rank(512)]
    # q_nope = q_nope.transpose(0, 1)

    # we need to index out the compressed_kv and k_pe for the current batch
    # batch_compressed_kv = None
    # batch_k_pe = None
    # for page_index in kv_index:
    #     if kv_total_len > page_size:
    #         tmp_compressed_kv = compressed_kv[page_index, 0:page_size, :]
    #         tmp_k_pe = k_pe[page_index, 0:page_size, :]
    #         if batch_compressed_kv is None or batch_k_pe is None:
    #             batch_compressed_kv = tmp_compressed_kv
    #             batch_k_pe = tmp_k_pe
    #         else: 
    #             batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
    #             batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
    #         kv_total_len -= page_size
    #     else:
    #         tmp_compressed_kv = compressed_kv[page_index, 0:kv_total_len, :]
    #         tmp_k_pe = k_pe[page_index, 0:kv_total_len, :]
    #         if batch_compressed_kv is None or batch_k_pe is None:
    #             batch_compressed_kv = tmp_compressed_kv
    #             batch_k_pe = tmp_k_pe
    #         else: 
    #             batch_compressed_kv = torch.cat((batch_compressed_kv, tmp_compressed_kv), dim=0)
    #             batch_k_pe = torch.cat((batch_k_pe, tmp_k_pe), dim=0)
    #         break
    # batch_compressed_kv is [kv_total_len(k_len), kv_lora_rank(512)]
    # batch_k_pe is [kv_total_len(k_len), qk_rope_head_dim(64)]


    k_pe_to_check_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_page_0_k_rope', (256,64))
    k_pe_to_check_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_page_0_k_rope', (256,64))
    # diff = torch.abs(k_pe_to_check_prefill - k_pe_to_check_decode).max()
    # mae =  torch.mean(k_pe_to_check_prefill - k_pe_to_check_decode)
    # print("k_pe diff -> ", diff)
    # print("k_pe mae -> ", mae)

    # pe_weights = torch.matmul(q_pe,batch_k_pe.mT)
    # kv_total_len = kv_page_nums * page_size
    pe_weights_0_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_pe_attention_weights', (1024,4096))
    pe_weights_0_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_pe_attention_weights', (1024,4096))

    # diff = torch.abs(pe_weights[0] - pe_weights_0).max()
    # print("pe_weights[0] diff -> ", diff)

    # attention_weights = (pe_weights + torch.matmul(q_nope, batch_compressed_kv.mT)) 

    # raw_weights = load_fp16_tensor('./debug/query_0_tp_0_raw_attention_weights', (1024, 4096))
    # raw_weights = raw_weights[0:qlen, 0:kv_total_len]
    # diff = torch.abs(attention_weights[0] - raw_weights).max()
    # print("raw attention_weigh/ts[0] diff -> ", diff)

    # attention_weights = attention_weights * softmax_scale
    # attention_weights is [num_heads(128), qlen, k_len]
    
    # attention_weights = attention_weights.transpose(0,1).unsqueeze(0).squeeze(-1).expand(qlen,-1,-1).transpose(0,1)
    
    # attention_masks[i] is [qlen, k_len]
    
    # attention_weights = (attention_weights + attention_masks)
    # attention_weights shape is [num_heads(128), qlen, k_len]


    # attention_weights = nn.functional.softmax(attention_weights,dim=-1,dtype=weight_type).to(q_pe.dtype)

    attention_weights_0_decode = load_fp16_tensor('./debug_decode/query_0_tp_0_attention_weights', (1024, 4096))
    attention_weights_0_prefill = load_fp16_tensor('./debug_prefill/query_0_tp_0_attention_weights', (1024, 4096))

    # attention_weights_0 = attention_weights_0[0:qlen, 0:kv_total_len]
    # diff = torch.abs(attention_weights[0] - attention_weights_0).max()
    # print("attention_weights[0] diff -> ", diff)


    # attn_output = torch.matmul(attention_weights, batch_compressed_kv) # [num_heads(128),qlen, lora_rank(512)]
    # out_absorb shape is [num_heads(128), kv_lora_rank(512), v_head_dim(128)]

    # o_absorb_check = load_fp16_tensor('./debug/query_0_tp_0_o_absorb', (qlen,kv_lora_rank))
    # diff = torch.abs(attn_output[0] - o_absorb_check).max()
    # print("o absorb[0] diff -> ", diff)

    # out_absorb = out_absorb.transpose(1, 2) # [qlen, num_heads(128), v_head_dim(128)]
    # # q for qlen, n for num_heads, h for v_head_dim, v for kv_lora_rank
    # attn_output = torch.matmul(attn_output, out_absorb) # [num_heads(128), qlen, v_head_dim(128)]

    # attn_output_check_0 = load_fp16_tensor('./debug/query_0_tp_0_attention_output', (qlen, nope_size))
    # diff = torch.abs(attn_output[0] - attn_output_check_0).max()
    # print("attn_output[0] diff -> ", diff)

    # attn_output = attn_output.transpose(0, 1) # [qlen, num_heads(128), v_head_dim(128)]
    # attn_output = attn_output.reshape(qlen, num_heads * nope_size)

    # w_o = o_proj.weight.view([hidden_size,num_heads * nope_size])
    # output = torch.matmul(attn_output,w_o.transpose(0,1))
    # output = output.view(qlen, hidden_size)
    
    # output_0_check = load_fp16_tensor('./debug/query_0_tp_0_qlen_output', (qlen, hidden_size))
    # h1_o = w_o[:,:128]
    # local_o_check = load_fp16_tensor('./debug/query_0_tp_0_local_w_o', (hidden_size, 128))
    # diff = torch.abs(local_o_check - h1_o).max()
    # print("local w_o diff -> ", diff)

    # h1_output = torch.matmul(attn_output[:,:128],h1_o.transpose(0,1))
    # diff = torch.abs(h1_output - output_0_check).max()
    # print("h1_output diff -> ", diff)


    output_check_decode = load_fp16_tensor('./debug_decode/output.bin')
    output_check_prefill = load_fp16_tensor('./debug_prefill/output.bin')
    # diff = torch.abs(output - output_check).max()
    # mae =   torch.mean(torch.abs(output - output_check))
    # print("output diff -> ", diff)



        
    return None

torch.set_printoptions(sci_mode=False, precision=5)
# output_cpu = test_cpu_mla()
# output_cpu_quant = test_cpu_mla_quant()
output_torch = test_torch()
# print("Output CPU: ", output_cpu)
# print("Output CPU: ", output_cpu_quant)
# print("Output Torch: ", output_torch)
# diff = (output_cpu - output_torch).abs()
# # 计算相对误差
# diff_relative = diff / (output_cpu.abs())
# # 把 diff_relative 中的 NaN 替换为 0
# diff_relative = torch.where(torch.isnan(diff_relative), torch.zeros_like(diff_relative), diff_relative)
# diff_relative_mean = torch.mean(torch.abs(output_cpu-output_torch)) / torch.mean(torch.abs(output_torch))

# print(f'Diff: ave:{diff.mean()}, max:{diff.max()}, min:{diff.min()},  relative_mean:{diff_relative_mean}, relative_max:{diff_relative.max()}, relative_min:{diff_relative.min()}')
# assert diff_relative_mean < 2e-1, "CPU and Torch outputs are not close enough!"




