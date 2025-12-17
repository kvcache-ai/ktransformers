import logging
import os, sys
import time
from typing import Optional

os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__) + "/../build")
from kt_kernel import kt_kernel_ext
from kt_kernel_ext.kvcache import ggml_type
import torch
from torch import inf, nn
from torch.nn import init
from torch_attention import apply_rotary_pos_emb, DeepseekV2RMSNorm, KDeepSeekV3Cache, DeepseekV3YarnRotaryEmbedding

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

qlen = 1024
kvlen = 0


page_table = range(20)
bsz_tensors = torch.tensor([1])


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
max_qlen = 1024
max_kvlen = 4096

max_position_embeddings = 163840


rope_scaling = {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
    "type": "yarn",
}


CPUInfer = kt_kernel_ext.CPUInfer(64)
validation_iter = 100


# data_type = torch.float32
weight_type = torch.bfloat16
# weight_type = torch.float16


input_type = {
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.float32,
}[weight_type]

q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False, dtype=weight_type)
q_b_proj = nn.Linear(q_lora_rank, num_heads * (nope_size + rope_size), bias=False, dtype=weight_type)
kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + rope_size, bias=False, dtype=weight_type)
kv_b_proj = nn.Linear(num_heads * (nope_size + nope_size), kv_lora_rank, bias=False, dtype=weight_type)
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


hidden_states = torch.randn((qlen, hidden_size), dtype=input_type).to("cpu").contiguous()


def build_mla():
    os.environ["BLAS_NUM_THREADS"] = "1"
    q_a_proj_weight = q_a_proj.weight.to(weight_type).to("cpu").contiguous()
    q_b_proj_weight = q_b_proj.weight.to(weight_type).to("cpu").contiguous()
    kv_a_proj_with_mqa_weight = kv_a_proj_with_mqa.weight.to("cpu").to(weight_type).contiguous()
    kv_b_proj_weight = kv_b_proj.weight.to(weight_type).to("cpu").contiguous()
    o_proj_weight = o_proj.weight.to(weight_type).to("cpu").contiguous()

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
        mla = kt_kernel_ext.mla.MLA_F32(config)
    else:
        raise ValueError(f"Unsupported data type: {weight_type}")

    mla.load_weights()
    mla.set_local_pages(pages_count)
    return mla


def load_fp32_tensor(file_path, shape):
    with open(file_path, "rb") as f:
        raw_data = f.read()
    tensor = torch.frombuffer(raw_data, dtype=torch.float32)
    tensor = tensor.view(shape)  # 根据你的 shape reshape
    return tensor


# page3 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/query_0_tp_0_page_3_kv_lora_rank_norm.f32',(page_size,kv_lora_rank))
# page3_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/query_0_tp_0_page_3_kv_lora_rank_norm.f32',(page_size,kv_lora_rank))

# diff = torch.abs(page3 - page3_2)
# print(f'Diff: ave:{diff.mean()}, max:{diff.max()}')

# q_pe_1 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/query_0_tp_0_q_rope.f32',(1, rope_size))
# q_pe_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/query_0_tp_0_q_rope.f32',(qlen, rope_size))
# diff = torch.abs(q_pe_1 - q_pe_2[-1])
# print(f'Q PE Diff: ave:{diff.mean()}, max:{diff.max()}')

# q_nope_1 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/query_0_tp_0_q_nope.f32',(1, nope_size))
# q_nope_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/query_0_tp_0_q_nope.f32',(qlen, nope_size))
# diff = torch.abs(q_nope_1 - q_nope_2[-1])
# print(f'Q Nope Diff: ave:{diff.mean()}, max:{diff.max()}')


# pe_attn_w_1 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/query_0_tp_0_pe_attention_weights.f32',(1,max_kvlen))
# pe_attn_w_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/query_0_tp_0_pe_attention_weights.f32',(qlen,max_kvlen))
# diff = torch.abs(pe_attn_w_1 - pe_attn_w_2[-1])
# print(f'PE Attention Weights Diff: ave:{diff.mean()}, max:{diff.max()}')


# raw_attn_w_1 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/query_0_tp_0_raw_attention_weights.f32',(1,max_kvlen))
# raw_attn_w_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/query_0_tp_0_raw_attention_weights.f32',(qlen,max_kvlen))
# diff = torch.abs(raw_attn_w_1 - raw_attn_w_2[-1])
# print(f'Raw Attention Weights Diff: ave:{diff.mean()}, max:{diff.max()}')


# output_1 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug1/output.bin.f32',shape=(1, hidden_size))
# output_2 = load_fp32_tensor('/home/yzw/xwy/Projects/ktransformers-dev/csrc/ktransformers_ext/examples/debug2/output.bin.f32',shape=(qlen, hidden_size))

# diff = torch.abs(output_1 - output_2[-1])
# print(f'Output Diff: ave:{diff.mean()}, max:{diff.max()}')


mla = build_mla()
output = torch.zeros((qlen, hidden_size), dtype=input_type).to("cpu").contiguous()
mla.forward([qlen], [page_table], [kvlen], hidden_states.data_ptr(), output.data_ptr())
print("CPU MLA Output: ", output[-1])


output_2 = torch.zeros((1, hidden_size), dtype=input_type).to("cpu").contiguous()
mla.forward([1], [page_table], [qlen - 1], hidden_states[-1].data_ptr(), output_2.data_ptr())
print("CPU MLA Output 2: ", output_2[-1])

diff = torch.abs(output[-1] - output_2[-1])
print(f"Diff: ave:{diff.mean()}, max:{diff.max()}")
assert diff.max() < 1e-1, "CPU and Torch outputs are not close enough!"
