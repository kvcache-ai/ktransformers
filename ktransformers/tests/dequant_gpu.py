import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# add path
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path+"/../..")
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
# from ktransformers.operators.linear import KTransformerLinear, QuantizedLinearMarlin
# from ktransformers.operators.experts import KTransformersMLPExpert, MLPExpertsTorch
from ktransformers.util.custom_gguf import GGUFLoader
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
target = "ffn_down_exps.weight"

t1 = time.time()
q_weight_cpu = gguf_config.load_gguf_tensor(key+target, "cpu")
# q_weight_cpu = torch.from_numpy(q_weight_cpu)

t2 = time.time()
q_weight_gpu = gguf_config.load_gguf_tensor(key+target, "cuda")
t3 = time.time()
print()
allclose = torch.allclose(q_weight_cpu, q_weight_gpu.cpu().to(torch.float32), atol=1e-6)
print(f"Q6k {key+target}")
print("load gguf tensor from cpu cost: ", t2-t1)
print("load gguf tensor from gpu cost: ", t3-t2)
print("allclose: ", allclose)


key = "blk.1."
target = "ffn_up_shexp.weight"

t1 = time.time()
q_weight_cpu = gguf_config.load_gguf_tensor(key+target, "cpu")
# q_weight_cpu = torch.from_numpy(q_weight_cpu)

t2 = time.time()
q_weight_gpu = gguf_config.load_gguf_tensor(key+target, "cuda")
t3 = time.time()
print()
allclose = torch.allclose(q_weight_cpu, q_weight_gpu.cpu(), atol=1e-6)
print(f"Q4k {key+target}")
print("load gguf tensor from cpu cost: ", t2-t1)
print("load gguf tensor from gpu cost: ", t3-t2)
print("allclose: ", allclose)
