import logging
import os,sys 
import time
from typing import Optional
os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__) + '/../build')
import kt_kernel_ext
from kt_kernel_ext.kvcache import ggml_type
import torch
from torch import inf, nn
from torch.nn import init
from torch_attention import apply_rotary_pos_emb,DeepseekV2RMSNorm,KDeepSeekV3Cache,DeepseekV3YarnRotaryEmbedding
logger = logging.getLogger("reader")

from gguf.gguf_reader import GGUFReader

def read_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    # print("Key-Value Pairs:") # noqa: NP100
    # max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        # print(f"{key:{max_key_length}} : {value}") # noqa: NP100
    # print("----") # noqa: NP100

    # List all tensors
    # print("Tensors:") # noqa: NP100
    # tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    # print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
    # print("-" * 80) # noqa: NP100
    re = []
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        # print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)) # noqa: NP100
        re.append(tensor)
    return re

def get_torch_tensor_from_gguf(gguf_weights, name):
    return torch.from_numpy(gguf_weights[name].data).contiguous()

def get_torch_tensor_and_type_from_gguf(gguf_weights, name):
    return torch.from_numpy(gguf_weights[name].data).contiguous(), gguf_weights[name].tensor_type.name

def type_to_ggml_type(type):
    if type == "F32":
        return ggml_type.FP32
    elif type == "F16":
        return ggml_type.FP16
    elif type == "BF16":
        return ggml_type.BF16
    else:
        raise ValueError(f"Unsupported data type: {type}")


use_real_weights = True
gguf_path = "/home/bd/models/DeepSeek-R1-BF16"

seed = 42  # 你可以选择任何整数作为种子
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

qlen = 3212 
kvlen = 0


page_table = range(20)
bsz_tensors=torch.tensor([1])


page_size = 256
pages_count = 200
tp_count = 4


hidden_size = 7168
q_lora_rank = 1536
kv_lora_rank = 512
num_heads = 128
nope_size = 128
rope_size = 64

rope_theta = 10000
max_qlen = 4096
max_kvlen = 4096

max_position_embeddings =  163840



rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
}



CPUInfer = kt_kernel_ext.CPUInfer(30)
validation_iter = 100


# data_type = torch.float32
weight_type = torch.bfloat16
# weight_type = torch.float16


input_type = {torch.float32:torch.float32,
              torch.float16:torch.float16,
              torch.bfloat16:torch.float32,
              }[weight_type]

q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False, dtype=weight_type)
q_b_proj = nn.Linear(q_lora_rank, num_heads * (nope_size+rope_size) , bias=False, dtype=weight_type)
kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + rope_size, bias=False, dtype=weight_type)
kv_b_proj = nn.Linear( num_heads * (nope_size + nope_size),kv_lora_rank, bias=False, dtype=weight_type)
o_proj = nn.Linear(num_heads * nope_size, hidden_size, bias=False, dtype=weight_type)
q_a_norm = torch.ones(hidden_size, dtype=torch.float32)
kv_a_norm = torch.ones(hidden_size, dtype=torch.float32)


def read_gguf_directory(directory):
    """
    Reads all GGUF files in a directory and prints their contents.

    Parameters:
    - directory: Path to the directory containing GGUF files.
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory {directory} does not exist.")
        return

    # List all GGUF files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".gguf")]
    if not files:
        logger.info(f"No GGUF files found in {directory}.")
        return

    re = []
    for file in files:
        file_path = os.path.join(directory, file)
        # print(f"Reading {file_path}:") # noqa: NP100
        # print("\n") # noqa: NP100
        re.extend(read_gguf_file(file_path))
    re = {r.name: r for r in re}
    return re


if use_real_weights := True:
    gguf_weights = read_gguf_directory(gguf_path)
    layer_idx = 0
    q_a_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_a.weight")
    q_a_proj.weight = nn.Parameter(q_a_proj_weight.view(torch.bfloat16), requires_grad=False)
    q_a_type = type

    q_a_norm_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_a_norm.weight")
    q_a_norm = q_a_norm_weight.view(torch.float32)
    # config.q_a_norm = q_a_norm_weight.data_ptr()
    # config.q_a_norm_type = type_to_ggml_type(type)

    q_b_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_b.weight")
    q_b_proj.weight = nn.Parameter(q_b_proj_weight.view(torch.bfloat16), requires_grad=False)

    kv_a_proj_with_mqa_weight, type = get_torch_tensor_and_type_from_gguf(
        gguf_weights, f"blk.{layer_idx}.attn_kv_a_mqa.weight"
    )
    kv_a_proj_with_mqa.weight = nn.Parameter(kv_a_proj_with_mqa_weight.view(torch.bfloat16), requires_grad=False)

    kv_a_norm_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_kv_a_norm.weight")
    kv_a_norm = kv_a_norm_weight.view(torch.float32)
    # config.kv_a_norm = kv_a_norm_weight.data_ptr()
    # config.kv_a_norm_type = type_to_ggml_type(type)

    kv_b_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_kv_b.weight")
    kv_b_proj.weight = nn.Parameter(kv_b_proj_weight.view(torch.bfloat16), requires_grad=False)

    o_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_output.weight")
    o_proj.weight = nn.Parameter(o_proj_weight.view(torch.bfloat16), requires_grad=False)
    
else:
    init.normal_(q_a_proj.weight, mean=0.0, std=0.02)
    init.normal_(q_b_proj.weight, mean=0.0, std=0.02)
    init.normal_(kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
    init.normal_(kv_b_proj.weight, mean=0.0, std=0.02)
    init.normal_(o_proj.weight, mean=0.0, std=0.02)

x_reshaped = kv_b_proj.weight.view(num_heads, 2, nope_size, kv_lora_rank)
q_absorb = x_reshaped[:, 0]
out_absorb = x_reshaped[:, 1]


hidden_states = torch.randn((qlen, hidden_size), dtype=input_type).to('cpu').contiguous()


def test_cpu_mla():
    os.environ["BLAS_NUM_THREADS"] = "1"
    q_a_proj_weight = q_a_proj.weight.to(weight_type).to('cpu').contiguous()
    q_b_proj_weight = q_b_proj.weight.to(weight_type).to('cpu').contiguous()
    kv_a_proj_with_mqa_weight = kv_a_proj_with_mqa.weight.to('cpu').to(weight_type).contiguous()
    kv_b_proj_weight = kv_b_proj.weight.to(weight_type).to('cpu').contiguous()
    o_proj_weight = o_proj.weight.to(weight_type).to('cpu').contiguous()

    config = kt_kernel_ext.mla.MLAConfig(
        hidden_size,
        q_lora_rank,
        kv_lora_rank,
        num_heads,
        nope_size,
        rope_size,
    )
    config.max_qlen = max_qlen
    config.max_kvlen = max_kvlen
    config.max_position_embeddings = max_position_embeddings 
    config.rope_scaling_factor = rope_scaling["factor"]
    config.rope_theta = rope_theta
    config.rope_scaling_beta_fast = rope_scaling["beta_fast"]
    config.rope_scaling_beta_slow = rope_scaling["beta_slow"]
    config.rope_scaling_mscale = rope_scaling["mscale"]
    config.rope_scaling_mscale_all_dim = rope_scaling["mscale_all_dim"]
    config.rope_scaling_original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]

    config.q_a_proj = q_a_proj_weight.data_ptr()
    config.q_b_proj = q_b_proj_weight.data_ptr()
    config.kv_a_proj_with_mqa = kv_a_proj_with_mqa_weight.data_ptr()
    config.kv_b_proj = kv_b_proj_weight.data_ptr()
    config.o_proj = o_proj_weight.data_ptr()

    config.q_a_norm = q_a_norm.data_ptr()
    config.q_a_norm_type = ggml_type.FP32
    config.kv_a_norm = kv_a_norm.data_ptr()
    config.kv_a_norm_type = ggml_type.FP32
    config.page_count = pages_count


    if weight_type == torch.float32:
        config.q_a_proj_type = ggml_type.FP32
        config.q_b_proj_type = ggml_type.FP32
        config.kv_a_proj_with_mqa_type = ggml_type.FP32
        config.kv_b_proj_type = ggml_type.FP32
        config.w_o_type = ggml_type.FP32
    elif weight_type == torch.float16:
        config.q_a_proj_type = ggml_type.FP16
        config.q_b_proj_type = ggml_type.FP16
        config.kv_a_proj_with_mqa_type = ggml_type.FP16
        config.kv_b_proj_type = ggml_type.FP16
        config.w_o_type = ggml_type.FP16
    elif weight_type == torch.bfloat16:
        config.q_a_proj_type = ggml_type.BF16
        config.q_b_proj_type = ggml_type.BF16
        config.kv_a_proj_with_mqa_type = ggml_type.BF16
        config.kv_b_proj_type = ggml_type.BF16
        config.w_o_type = ggml_type.BF16
    else:
        raise ValueError(f"Unsupported data type: {weight_type}")


    config.pool = CPUInfer.backend_


    if weight_type == torch.float32:
        mla = kt_kernel_ext.mla.MLA_F32(config)
    elif weight_type == torch.float16:
        mla = kt_kernel_ext.mla.MLA_F16(config)
    elif weight_type == torch.bfloat16:
        # mla = kt_kernel_ext.mla.MLA_F32(config)
        mla = kt_kernel_ext.mla.MLA_QUAN_F32(config)
    else:
        raise ValueError(f"Unsupported data type: {weight_type}")
    
    mla.load_weights()
    mla.set_local_pages(pages_count)

    output = torch.zeros((qlen, hidden_size), dtype=input_type).to('cpu').contiguous()
    mla.forward([qlen],[page_table],[kvlen],hidden_states.data_ptr(),output.data_ptr())
    print("CPU MLA Output: ",output)
    return output




def load_fp16_tensor(file_path, shape):
    # return load_fp32_tensor(file_path, shape)
    return torch.zeros(shape)
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=weight_type)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

def load_fp32_tensor(file_path, shape):
    return torch.zeros(shape)
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor

def test_torch():
    torch.set_grad_enabled(False)

    softmax_scale = (nope_size + rope_size) ** -0.5
    # 1代表的是压缩的kv的头数
    k_caches = torch.randn(1,pages_count, page_size, 1, kv_lora_rank + rope_size).to(weight_type)
    kv_cache = KDeepSeekV3Cache(page_size=page_size, kv_lora_rank=kv_lora_rank, k_caches=k_caches)

    q_a_layernorm = DeepseekV2RMSNorm(q_lora_rank)
    q_a_layernorm.weight = nn.Parameter( q_a_norm,requires_grad=False)

    x = torch.randn(q_lora_rank, dtype=weight_type)*100
    print(x)
    print(q_a_layernorm(x))

    kv_a_layernorm = DeepseekV2RMSNorm(kv_lora_rank)
    kv_a_layernorm.weight = nn.Parameter(kv_a_norm, requires_grad=False)

  
    # 第三步：拆分成两个 tensor
    # q_absorb, out_absorb = x_permuted[:, 0], x_permuted[:, 1]  # 都是 (num_heads, nope_size, kv_lora_rank
    # q_absorb = kv_b_proj[:, ] # torch.randn(num_heads, nope_size, kv_lora_rank, dtype=data_type)
    # out_absorb = kv_b_proj # torch.randn(num_heads, nope_size, kv_lora_rank, dtype=data_type)

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
    # 构造一个qlen 长度的输入 hidden_states, 对应的历史 kv_indptr 是[0:bsz]
    # kv_indices 是[0:bsz]，page_idx=[0:bsz], page_offset=[kvlen:qlen+kvlen]
    # last_page_len = [qlen+kvlen,...] layer_idx = 1
    # position_ids = [kvlen:qlen+kvlen]
    q_indptr = torch.tensor([0,qlen]).to(torch.int32)

    kv_indptr = torch.tensor([0,(qlen+kvlen+page_size-1)//page_size]).to(torch.int32)
    kv_indices = torch.tensor(range(pages_count)).to(torch.int32)

    page_idx = torch.tensor([i//page_size for i in range(kvlen,kvlen+qlen)] ).to(torch.int32)
    page_offset = torch.tensor( [i%page_size for i in range(kvlen, kvlen + qlen)]).to(torch.int32)

    last_page_len = torch.tensor([256], device=hidden_states.device)
    position_ids = torch.tensor(range(kvlen, kvlen + qlen)).to(torch.int32)


    # 按照行创建 mask [qlen,kvlen+qlen]
    attention_masks = torch.zeros((max_qlen, max_kvlen), dtype=weight_type)
    for i in range(max_qlen):
        attention_masks[i, i + kvlen + 1:] = -inf


    def torch_attn(hidden_states_i: torch.Tensor,
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
        hidden_states = hidden_states_i.to(weight_type)
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
            qlen, _ = batch_hidden_states.size()
            # print("qlen -> ", qlen)

            hidden_states_to_check = load_fp16_tensor('./debug/query_0_tp_0_input.bin',batch_hidden_states.shape)
            diff = torch.abs(batch_hidden_states - hidden_states_to_check).max()
            print("hidden_states diff -> ", diff)

            q_lora = q_a_proj(batch_hidden_states)
            # q_lora_to_check = load_fp16_tensor('./debug/query_0_tp_0_qlora.bin', q_lora.shape)
            # q_lora_to_check_test = load_fp16_tensor('./debug/query_0_tp_0_qlora_test.bin', q_lora.shape)
            # diff = torch.abs(q_lora - q_lora_to_check).max()
            # diff_test = torch.abs(q_lora - q_lora_to_check_test).max()
            # print("q_lora max diff -> ", diff)
            # print("q_lora max diff test -> ", diff_test)
            # mae =  torch.mean(torch.abs(q_lora - q_lora_to_check))
            # mae_test =  torch.mean(torch.abs(q_lora - q_lora_to_check_test))
            # print("q_lora mae -> ", mae)
            # print("q_lora mae test -> ", mae_test)



            q_lora_norm = q_a_layernorm(q_lora)
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
            
            q = q_b_proj(q_lora_norm)
            # for v3, bsz, qlen, num_heads(128), qk_head_dim(192=128(nope)+64(rope))
            q = q.view(qlen, num_heads, nope_size+rope_size)
            # q_nope is [qlen, num_heads(128), qk_nope_head_dim(128)]
            # q_pe is [qlen, num_heads(128), qk_rope_head_dim(64)]
            q_nope, q_pe = torch.split(
                q, [nope_size, rope_size], dim=-1
            )
              
            # compressed_kv is [qlen, kv_lora_rank(512) + rope(64)]
            compressed_kv = kv_a_proj_with_mqa(batch_hidden_states)
            # compressed_kv is [qlen, kv_lora_rank(512)], k_pe is [qlen, rope(64)]
            compressed_kv, k_pe = torch.split(
                compressed_kv, [kv_lora_rank, rope_size], dim=-1
            )
            compressed_kv = compressed_kv.contiguous()


            # compressed_kv_page_0 = compressed_kv[0:page_size, :]
            # compressed_kv_to_check = load_fp16_tensor('./debug/query_0_tp_0_page_0_kv_lora_rank',
            #                                           compressed_kv_page_0.shape)
            # diff = torch.abs(compressed_kv_page_0 - compressed_kv_to_check).max()
            # mae =  torch.mean(torch.abs(compressed_kv_page_0 - compressed_kv_to_check)) 
            # print("compressed_kv diff -> ", diff)
            # print("compressed_kv mae -> ", mae)

            compressed_kv = kv_a_layernorm(compressed_kv)
            # k_pe is [qlen, 1, qk_rope_head_dim(64)]

            # compressed_kv_page_0 = compressed_kv[0:page_size, :]
            # compressed_kv_to_check = load_fp16_tensor('./debug/query_0_tp_0_page_0_kv_lora_rank_norm',
            #                                           compressed_kv_page_0.shape)
            # diff = torch.abs(compressed_kv_page_0 - compressed_kv_to_check).max()
            # mae =  torch.mean(torch.abs(compressed_kv_page_0 - compressed_kv_to_check))
            # print("compressed_kv diff norm -> ", diff)
            # print("compressed_kv mae norm -> ", mae)
            
                                                      


            k_pe = k_pe.view(qlen, 1, rope_size)
            # compressed_kv is [qlen, 1, kv_lora_rank(512)]
            compressed_kv = compressed_kv.view(qlen, 1, kv_lora_rank)
            
            cos, sin = rotary_emb(q_pe, batch_position_ids)

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
            
            q_pe_nope = q_pe.transpose(0,1)
            # q_pe_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_rope', q_pe_nope[0].shape)
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
            # # q1 = (qa * cos[:,:32] - qb * sin[:,:32])
            # # q2 = (qb*cos[:,:32] + qa*sin[:,:32])
            # q1 = (qa * cos_to_check - qb * sin_to_check)
            # q2 = (qb*cos_to_check + qa*sin_to_check)
            # q_new = torch.cat((q1,q2), dim=-1)
            # print(f"q_pe shape{q_pe.shape}, k_pe shape {k_pe.shape}")
            # new_q_pe = torch.zeros_like(q_pe)
            # new_q_pe[:,:,range(0,64,2)] = 1
            # new_q_pe[:,:,range(1,65,2)] = 10
            q_pe, k_pe = apply_rotary_pos_emb(q_pe.unsqueeze(0), k_pe.unsqueeze(0), cos, sin, unsqueeze_dim=1)
            q_pe = q_pe.squeeze(0)
            # q_pe is [num_heads(128), qlen, qk_rope_head_dim(64)]
            q_pe.transpose_(0, 1)    

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

            if kv_cache is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "page_idx": batch_page_idx, "page_offset": batch_page_offset}  # Specific to RoPE models
                compressed_kv_with_k_pe = kv_cache.update(compressed_kv.unsqueeze(0), k_pe, layer_idx, batch_page_idx, batch_page_offset, cache_kwargs)
                compressed_kv = compressed_kv_with_k_pe [:, :, :, :kv_lora_rank].view(-1, page_size, kv_lora_rank)
                k_pe = compressed_kv_with_k_pe [:, :, :, kv_lora_rank:].view(-1, page_size, rope_size)
            # q_absorb is [num_heads(128), qk_nope_head_dim(128), kv_lora_rank(512)]
            # out_absorb is [num_heads(128), kv_lora_rank(512), v_head_dim(128)] v_head_dim is also the nope dim
            # q_absorb, out_absorb = get_absorbed()
            # q_nope is [num_heads(128), qlen, qk_nope_head_dim(128)]
            q_nope = q_nope.transpose(0, 1) # qlen is 1, no GPU overhead, same below

            # q_nope_0_to_check = load_fp16_tensor('./debug/query_0_tp_0_q_nope', q_nope[0].shape)
            # diff = torch.abs(q_nope[0] - q_nope_0_to_check).max()
            # mae =  torch.mean(torch.abs(q_nope[0] - q_nope_0_to_check))
            # print("q_nope[0] diff -> ", diff)

            # q_nope is [num_heads(128), qlen, kv_lora_rank(512)]
            q_nope = torch.matmul(q_nope, q_absorb) # batched MM

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


            # k_pe_to_check = load_fp16_tensor('./debug/query_0_tp_0_page_0_k_rope', (256,64))
            # diff = torch.abs(batch_k_pe[:256] - k_pe_to_check).max()
            # mae =  torch.mean(torch.abs(batch_k_pe[:256] - k_pe_to_check))
            # print("k_pe diff -> ", diff)
            # print("k_pe mae -> ", mae)

            pe_weights = torch.matmul(q_pe,batch_k_pe.mT)
            kv_total_len = kv_page_nums * page_size
            # pe_weights_0 = load_fp16_tensor('./debug/query_0_tp_0_pe_attention_weights', (1024,4096))
            # pe_weights_0 = pe_weights_0[0:qlen, 0:kv_total_len]
            # diff = torch.abs(pe_weights[0] - pe_weights_0).max()
            # print("pe_weights[0] diff -> ", diff)

            attention_weights = (pe_weights + torch.matmul(q_nope, batch_compressed_kv.mT)) 

            # raw_weights = load_fp16_tensor('./debug/query_0_tp_0_raw_attention_weights', (1024, 4096))
            # raw_weights = raw_weights[0:qlen, 0:kv_total_len]
            # diff = torch.abs(attention_weights[0] - raw_weights).max()
            # print("raw attention_weights[0] diff -> ", diff)

            attention_weights = attention_weights * softmax_scale
            # attention_weights is [num_heads(128), qlen, k_len]
            
            # attention_weights = attention_weights.transpose(0,1).unsqueeze(0).squeeze(-1).expand(qlen,-1,-1).transpose(0,1)
            
            # attention_masks[i] is [qlen, k_len]
            
            print(attention_weights.shape)
            print(attention_masks.shape)
            attention_weights = (attention_weights + attention_masks[ :attention_weights.shape[1],:attention_weights.shape[2]])
            # attention_weights shape is [num_heads(128), qlen, k_len]


            attention_weights = nn.functional.softmax(attention_weights,dim=-1,dtype=weight_type).to(q_pe.dtype)

            # attention_weights_0 = load_fp16_tensor('./debug/query_0_tp_0_attention_weights', (1024, 4096))
            # attention_weights_0 = attention_weights_0[0:qlen, 0:kv_total_len]
            # diff = torch.abs(attention_weights[0] - attention_weights_0).max()
            # print("attention_weights[0] diff -> ", diff)


            attn_output = torch.matmul(attention_weights, batch_compressed_kv) # [num_heads(128),qlen, lora_rank(512)]
            # out_absorb shape is [num_heads(128), kv_lora_rank(512), v_head_dim(128)]

            # o_absorb_check = load_fp16_tensor('./debug/query_0_tp_0_o_absorb', (qlen,kv_lora_rank))
            # diff = torch.abs(attn_output[0] - o_absorb_check).max()
            # print("o absorb[0] diff -> ", diff)

            out_absorb = out_absorb.transpose(1, 2) # [qlen, num_heads(128), v_head_dim(128)]
            # q for qlen, n for num_heads, h for v_head_dim, v for kv_lora_rank
            attn_output = torch.matmul(attn_output, out_absorb) # [num_heads(128), qlen, v_head_dim(128)]

            # attn_output_check_0 = load_fp16_tensor('./debug/query_0_tp_0_attention_output', (qlen, nope_size))
            # diff = torch.abs(attn_output[0] - attn_output_check_0).max()
            # print("attn_output[0] diff -> ", diff)

            attn_output = attn_output.transpose(0, 1) # [qlen, num_heads(128), v_head_dim(128)]
            attn_output = attn_output.reshape(qlen, num_heads * nope_size)

            w_o = o_proj.weight.view([hidden_size,num_heads * nope_size])
            output = torch.matmul(attn_output,w_o.transpose(0,1))
            output = output.view(qlen, hidden_size)
            
            # output_0_check = load_fp16_tensor('./debug/query_0_tp_0_qlen_output', (qlen, hidden_size))
            # h1_o = w_o[:,:128]
            # local_o_check = load_fp16_tensor('./debug/query_0_tp_0_local_w_o', (hidden_size, 128))
            # diff = torch.abs(local_o_check - h1_o).max()
            # print("local w_o diff -> ", diff)

            # h1_output = torch.matmul(attn_output[:,:128],h1_o.transpose(0,1))
            # diff = torch.abs(h1_output - output_0_check).max()
            # print("h1_output diff -> ", diff)


            # output_check = load_fp16_tensor('./debug/output.bin', output.shape)
            # diff = torch.abs(output - output_check).max()
            # mae =   torch.mean(torch.abs(output - output_check))
            # print("output diff -> ", diff)


            final_attention_output = torch.cat((final_attention_output, output), dim=0)
        return final_attention_output



    torch_output = torch_attn(
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
            layer_idx=0
        )
    print("Torch Output: ",torch_output)
    return torch_output

torch.set_printoptions(sci_mode=False, precision=5)
output_cpu = test_cpu_mla()
output_torch = test_torch()
print("Output CPU: ", output_cpu)
print("Output Torch: ", output_torch)
diff = (output_cpu - output_torch).abs()
# 计算相对误差
diff_relative = diff / (output_cpu.abs())
# 把 diff_relative 中的 NaN 替换为 0
diff_relative = torch.where(torch.isnan(diff_relative), torch.zeros_like(diff_relative), diff_relative)
diff_relative_mean = torch.mean(torch.abs(output_cpu-output_torch)) / torch.mean(torch.abs(output_torch))

print(f'Diff: ave:{diff.mean()}, max:{diff.max()}, min:{diff.min()},  relative_mean:{diff_relative_mean}, relative_max:{diff_relative.max()}, relative_min:{diff_relative.min()}')
assert diff_relative_mean < 2e-1, "CPU and Torch outputs are not close enough!"




