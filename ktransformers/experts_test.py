import sys, os
sys.path.append(os.path.dirname(__file__)+"\\..\\")
import torch
from torch.autograd.gradcheck import _check_analytical_jacobian_attributes
from transformers import AutoConfig
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators.experts import MLPCPUExperts, MLPExpertsTorch
model_path = "deepseek-ai/DeepSeek-V2-Lite-Chat"
gguf_path="D:\\models\\DeepSeek-V2-Lite-Chat\\Q4_K_M"
gguf_loader=GGUFLoader(gguf_path)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

torch.set_default_dtype(torch.bfloat16)
cpuinfer_experts = MLPCPUExperts(key="blk.3", gguf_loader=gguf_loader, config=config, n_routed_experts=64, warmup=True)
torch_experts = MLPExpertsTorch(key="blk.3", gguf_loader=gguf_loader, config=config, n_routed_experts=64)
cpuinfer_experts.load()
torch_experts.load()

hidden_size = 2048
input_tensor = torch.ones((2,hidden_size), dtype=torch.bfloat16)
expertids = torch.arange(0,6, dtype=torch.long).expand(2,6)
weights = torch.ones((6), dtype=torch.float32).expand(2,6)
print(input_tensor)
print(expertids)
print(weights)
cpuinfer_out = cpuinfer_experts.forward(input_tensor, expertids, weights)
torch.cuda.synchronize()
torch_out = torch_experts.forward(input_tensor, expertids, weights)
torch.cuda.synchronize()

print("cpuinfer_out", cpuinfer_out)
print("torch_out", torch_out)