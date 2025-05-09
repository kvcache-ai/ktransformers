import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# add path
import sys
sys.path.append("../..")
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
from ktransformers.operators.linear import KTransformersLinear, KLinearMarlin
from ktransformers.operators.experts import KTransformersExperts, KExpertsTorch
from ktransformers.util.custom_loader import GGUFLoader, dequantize_q4_k_gpu, dequantize_q4_k
import torch
import KTransformersOps
torch.set_default_dtype(torch.bfloat16)
import time
from transformers import (
    AutoConfig,
)

gguf_config = GGUFLoader("/data/Qwen2-57B-A14B-Instruct-GGUF/q4_k_m")
model_name = "/data/Qwen2-57B-A14B-Instruct"
key = "blk.0."
target = "ffn_up_exps.weight"

data = gguf_config.get_mmap_tensor(key + target)

_, factors, offsets, qs1, qs2= dequantize_q4_k(data)
factors_cpu = torch.from_numpy(factors)
offsets_cpu = torch.from_numpy(offsets)
qs1_cpu = torch.from_numpy(qs1)
qs2_cpu = torch.from_numpy(qs2)


_, factors, offsets, qs1, qs2 = dequantize_q4_k_gpu(data)

print(torch.allclose(factors.cpu(), factors_cpu))
print(torch.allclose(offsets.cpu(), offsets_cpu))
print(torch.allclose(qs1.cpu(), qs1_cpu))
print(torch.allclose(qs2.cpu(), qs2_cpu))