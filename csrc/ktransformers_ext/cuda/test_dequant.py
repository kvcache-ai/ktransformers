import os
import sys
sys.path.insert(0,"/home/zbx/ktransformers")
from ktransformers.util.custom_gguf import GGUFLoader
import torch

gguf_loader_1 = GGUFLoader("/mnt/data/model/DeepseekV3-q4km-gguf")
gguf_loader_2 = GGUFLoader("/mnt/data/chenht/model/gguf_for_ktransformers/DeepSeek-V3-bf16/")

torch.set_default_dtype(torch.bfloat16)

tensor_1 = gguf_loader_1.load_gguf_tensor("blk.0.attn_kv_a_mqa.weight", "cuda")
tensor_2 = gguf_loader_2.load_gguf_tensor("blk.0.attn_kv_a_mqa.weight", "cuda")

print(tensor_1[0, -64:])
print(tensor_2[0, -64:])