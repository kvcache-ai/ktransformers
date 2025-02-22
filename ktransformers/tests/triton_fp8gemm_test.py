import torch
import torch.nn.functional as F
from typing import Optional
import pytest
from typing import Tuple, Optional, Literal

# use dir path
import os
import sys
sys.path.insert(0, "/home/azure/ktransformers")
print(sys.path)
from ktransformers.ktransformers_ext.triton.fp8gemm import fp8_gemm, act_quant, weight_dequant
from safetensors import safe_open

world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
# Assuming `fp8_gemm`, `act_quant`, `weight_dequant` and other relevant functions are already defined

def test_fp8_gemm_vs_torch_matmul():
    # Test case 1: Create random matrices of size (M, K) and (K, N)
    M, K, N = 64, 128, 256  # Matrix dimensions
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

    # Apply act_quant to both matrices
    x_quantized, scale_x = act_quant(x, block_size)
    weight_quantized, scale_w = act_quant(weight, block_size)
    
    # mk continous
    x_quantized = x_quantized.contiguous()
    weight_quantized = weight_quantized.contiguous()
    scale_x = scale_x.contiguous()
    scale_w = scale_w.contiguous()

    # Perform fp8_gemm using the quantized tensors
    result_fp8_gemm = fp8_gemm(x_quantized, scale_x, weight_quantized, scale_w)

    # Perform torch.matmul using the original floating point tensors
    result_torch_matmul = torch.matmul(x, weight.T)
    print(f'result_torch_matmul: {result_torch_matmul.shape}')
    print(f'result_fp8_gemm: {result_fp8_gemm.shape}')

    print(f"result_fp8_gemm:\n {result_fp8_gemm}")
    print(f"result_torch_matmul:\n {result_torch_matmul}")
    
def test_fp8_gemm_vs_torch_matmul_load():
    file_path = "/mnt/data/model/DeepSeek-V3/model-00001-of-000163.safetensors"
    with safe_open(file_path, framework="pt", device=0) as f:
        weight = f.get_tensor("model.layers.0.mlp.down_proj.weight")
        scale = f.get_tensor("model.layers.0.mlp.down_proj.weight_scale_inv")

    # weight_dequant
    weight_dequantized = weight_dequant(weight, scale)
    print(f"weight_dequantized: {weight_dequantized.shape}")
    N, K = weight_dequantized.shape
    M = 64
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    x_quantized, scale_x = act_quant(x, block_size)
    
    # Test case 1: quantized x matmal with undequantized weight
    result_fp8_gemm = fp8_gemm(x_quantized, scale_x, weight, scale)
    print(f"result_fp8_gemm:\n {result_fp8_gemm}")

    # Perform torch.matmul using the original floating point tensors
    result_torch_matmul = torch.matmul(x, weight_dequantized.to(torch.bfloat16).T)
    print(f"result_torch_matmul:\n {result_torch_matmul}")

if __name__ == "__main__":
    test_fp8_gemm_vs_torch_matmul()
    test_fp8_gemm_vs_torch_matmul_load()
    