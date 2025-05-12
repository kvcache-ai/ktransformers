import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
# add path
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path+"/../..")
import numpy as np
# from ktransformers.operators.linear import KTransformersLinear, KLinearMarlin
# from ktransformers.operators.experts import KTransformersExperts, KExpertsTorch
from ktransformers.util.custom_loader import GGUFLoader
import torch
import KTransformersOps
torch.set_default_dtype(torch.bfloat16)
import time
from transformers import (
    AutoConfig,
)
import os
# CUDA_LAUNCH_BLOCKING=1
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

gguf_config = GGUFLoader("/data/Qwen2-57B-A14B-Instruct-GGUF/q4_k_m")
model_name = "/data/Qwen2-57B-A14B-Instruct"

# Q4k
key = "blk.1."
target = "attn_q.weight"

t1 = time.time()
q_weight_cpu = gguf_config.load_gguf_tensor(key+target, "cpu")
# q_weight_cpu = torch.from_numpy(q_weight_cpu)

t2 = time.time()
q_weight_gpu = gguf_config.load_gguf_tensor(key+target, "cuda:0")
t3 = time.time()
print()
allclose = torch.allclose(q_weight_cpu, q_weight_gpu.cpu(), atol=1e-6)
print(f"Q4k {key+target}")
print("load gguf tensor from cpu cost: ", t2-t1)
print("load gguf tensor from gpu cost: ", t3-t2)
print("allclose: ", allclose)


# Q6k
key = "blk.0."
target = "ffn_down_exps.weight"

t1 = time.time()
q_weight_cpu = gguf_config.load_gguf_tensor(key+target, "cpu")
t2 = time.time()
q_weight_gpu = gguf_config.load_gguf_tensor(key+target, "cuda:0")
t3 = time.time()
print()
allclose = torch.allclose(q_weight_cpu, q_weight_gpu.cpu().to(torch.float32), atol=1e-6)
print(f"Q6k {key+target}")
print("load gguf tensor from cpu cost: ", t2-t1)
print("load gguf tensor from gpu cost: ", t3-t2)
print("allclose: ", allclose)
