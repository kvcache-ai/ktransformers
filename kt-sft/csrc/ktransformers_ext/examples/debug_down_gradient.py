#!/usr/bin/env python
# coding=utf-8
'''
Description  : Debug script to verify down_proj LoRA gradient issue
'''
import torch
import numpy as np

# Configuration
expert_num = 3
hidden_size = 8
intermediate_size = 4
lora_rank = 2
lora_scaling = 2.0
qlen = 4
n_routed_experts = 2

dtype = torch.bfloat16

# LoRA weights for one expert
down_lora_A = torch.randn(lora_rank, intermediate_size, dtype=dtype)
down_lora_B = torch.randn(hidden_size, lora_rank, dtype=dtype)

# Input (intermediate activation)
intermediate = torch.randn(qlen, intermediate_size, dtype=dtype)

# Gradient from output
grad_output = torch.randn(qlen, hidden_size, dtype=dtype)

# Expert routing weights for these tokens
weights = torch.tensor([0.6, 0.7, 0.8, 0.9], dtype=torch.float32)

print("=" * 80)
print("Testing down_proj LoRA gradient with/without routing weights")
print("=" * 80)

# Method 1: WITHOUT weight (current C++ implementation)
print("\nMethod 1: Gradient WITHOUT routing weight")
grad_lora_A_no_weight = torch.zeros_like(down_lora_A)
grad_lora_B_no_weight = torch.zeros_like(down_lora_B)

for t in range(qlen):
    inter_t = intermediate[t]  # [intermediate_size]
    grad_t = grad_output[t]    # [hidden_size]

    # tmp = inter @ lora_A^T  [rank]
    tmp = inter_t @ down_lora_A.t().to(torch.float32)

    # grad_lora_B += grad^T @ tmp * scaling  [hidden, rank]
    grad_lora_B_no_weight += torch.outer(grad_t.to(torch.float32), tmp.to(torch.float32)) * lora_scaling

    # grad_tmp = grad @ lora_B * scaling  [rank]
    grad_tmp = grad_t.to(torch.float32) @ down_lora_B.to(torch.float32) * lora_scaling

    # grad_lora_A += grad_tmp^T @ inter  [rank, intermediate]
    grad_lora_A_no_weight += torch.outer(grad_tmp, inter_t.to(torch.float32))

print(f"  grad_lora_A range: [{grad_lora_A_no_weight.min():.4f}, {grad_lora_A_no_weight.max():.4f}]")
print(f"  grad_lora_A mean: {grad_lora_A_no_weight.mean():.4f}")
print(f"  grad_lora_B range: [{grad_lora_B_no_weight.min():.4f}, {grad_lora_B_no_weight.max():.4f}]")
print(f"  grad_lora_B mean: {grad_lora_B_no_weight.mean():.4f}")

# Method 2: WITH weight (correct implementation)
print("\nMethod 2: Gradient WITH routing weight")
grad_lora_A_with_weight = torch.zeros_like(down_lora_A)
grad_lora_B_with_weight = torch.zeros_like(down_lora_B)

for t in range(qlen):
    inter_t = intermediate[t]  # [intermediate_size]
    grad_t = grad_output[t] * weights[t]  # [hidden_size] ← 乘以routing weight!

    # tmp = inter @ lora_A^T  [rank]
    tmp = inter_t @ down_lora_A.t().to(torch.float32)

    # grad_lora_B += grad^T @ tmp * scaling  [hidden, rank]
    grad_lora_B_with_weight += torch.outer(grad_t.to(torch.float32), tmp.to(torch.float32)) * lora_scaling

    # grad_tmp = grad @ lora_B * scaling  [rank]
    grad_tmp = grad_t.to(torch.float32) @ down_lora_B.to(torch.float32) * lora_scaling

    # grad_lora_A += grad_tmp^T @ inter  [rank, intermediate]
    grad_lora_A_with_weight += torch.outer(grad_tmp, inter_t.to(torch.float32))

print(f"  grad_lora_A range: [{grad_lora_A_with_weight.min():.4f}, {grad_lora_A_with_weight.max():.4f}]")
print(f"  grad_lora_A mean: {grad_lora_A_with_weight.mean():.4f}")
print(f"  grad_lora_B range: [{grad_lora_B_with_weight.min():.4f}, {grad_lora_B_with_weight.max():.4f}]")
print(f"  grad_lora_B mean: {grad_lora_B_with_weight.mean():.4f}")

# Compare
print("\n" + "=" * 80)
print("Comparison")
print("=" * 80)
ratio_A = grad_lora_A_with_weight.abs().mean() / grad_lora_A_no_weight.abs().mean()
ratio_B = grad_lora_B_with_weight.abs().mean() / grad_lora_B_no_weight.abs().mean()

print(f"Ratio (with_weight / no_weight):")
print(f"  grad_lora_A: {ratio_A:.4f}x")
print(f"  grad_lora_B: {ratio_B:.4f}x")
print(f"Average routing weight: {weights.mean():.4f}")
print(f"\nConclusion: Gradient {'SHOULD' if abs(ratio_A - weights.mean()) < 0.1 else 'should NOT'} be multiplied by routing weight")
