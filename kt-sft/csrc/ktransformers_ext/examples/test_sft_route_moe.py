#!/usr/bin/env python
# coding=utf-8
'''
Description  : Test script for routed_experts with LoRA fine-tuning
Author       : KT-SFT Team
Date         : 2025-01-25
Version      : 1.0.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
import os, sys
import time
sys.path.append(os.path.dirname(__file__) + '/../build')
import cpuinfer_ext
import torch
from pathlib import Path
import numpy as np

# ==================== Configuration ====================
expert_num = 10  # Total number of experts
hidden_size = 5120  # Model hidden dimension
intermediate_size = 1536  # Expert intermediate dimension
max_len = 1024  # Maximum sequence length

n_routed_experts = 2  # Number of routed experts per token
qlen = 128  # Sequence length for testing
num_threads = 112  # Number of CPU threads
validation_iter = 1  # Number of validation iterations
LAYER_IDX = 0  # Layer index for debugging

# LoRA configuration
lora_rank = 8  # LoRA rank (r)
lora_alpha = 16  # LoRA alpha
lora_scaling = lora_alpha / lora_rank  # LoRA scaling factor

dtype = torch.bfloat16
gradtype = torch.bfloat16

import shutil
folder_path = "/home/lpl/kt-sft/debug_route_moe"
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)
DUMP_DIR = Path(folder_path)

# ==================== Activation Functions ====================
def silu_fwd(x: torch.Tensor) -> torch.Tensor:
    return x / (1. + torch.exp(-x))

def silu_grad(x: torch.Tensor) -> torch.Tensor:
    """SiLU gradient"""
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1. + x * (1. - sigmoid_x))

class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return silu_fwd(inp)

    @staticmethod
    def backward(ctx, grad_out):
        (inp,) = ctx.saved_tensors
        sig = torch.sigmoid(inp)
        return grad_out * (sig + inp * sig * (1. - sig))

silu = SiLU.apply  # Differentiable version

# ==================== LoRA MLP Implementation ====================
class LoRAMLP(torch.nn.Module):
    """MLP with LoRA adaptation"""
    def __init__(self, gate_base, up_base, down_base,
                 gate_lora_A, gate_lora_B, up_lora_A, up_lora_B, down_lora_A, down_lora_B):
        super().__init__()
        # Base weights (frozen)
        self.register_buffer('gate_base', gate_base)
        self.register_buffer('up_base', up_base)
        self.register_buffer('down_base', down_base)

        # LoRA adapters (trainable)
        self.gate_lora_A = gate_lora_A
        self.gate_lora_B = gate_lora_B
        self.up_lora_A = up_lora_A
        self.up_lora_B = up_lora_B
        self.down_lora_A = down_lora_A
        self.down_lora_B = down_lora_B

    def forward(self, x):
        # gate_proj = base + lora_B @ lora_A * scaling
        gate = torch.mm(x, self.gate_base.t())
        gate_lora = torch.mm(torch.mm(x, self.gate_lora_A.t()), self.gate_lora_B.t()) * lora_scaling
        gate = gate + gate_lora

        # up_proj = base + lora_B @ lora_A * scaling
        up = torch.mm(x, self.up_base.t())
        up_lora = torch.mm(torch.mm(x, self.up_lora_A.t()), self.up_lora_B.t()) * lora_scaling
        up = up + up_lora

        # Activation
        inter = silu(gate) * up

        # down_proj = base + lora_B @ lora_A * scaling
        output = torch.mm(inter, self.down_base.t())
        output_lora = torch.mm(torch.mm(inter, self.down_lora_A.t()), self.down_lora_B.t()) * lora_scaling
        output = output + output_lora

        return output

def moe_lora_torch(x, eid, w, mlp_experts):
    """MoE with LoRA - PyTorch reference implementation"""
    T, k = eid.shape
    tok_cnt = torch.zeros(expert_num, dtype=torch.int64)
    for e in eid.view(-1):
        tok_cnt[e] += 1

    # Pack tokens
    order = eid.view(-1).argsort()
    packed = x[order // k]

    outputs, start = [], 0
    for e in range(expert_num):
        num = tok_cnt[e].item()
        if not num:
            continue
        end = start + num
        o = mlp_experts[e](packed[start:end])
        outputs.append(o)
        start = end

    if outputs:
        out_all = torch.cat(outputs, 0)
    else:
        out_all = packed.new_empty(0, hidden_size)

    # Restore order and apply weights
    out_restore = torch.empty_like(out_all)
    out_restore[order] = out_all
    out_restore = out_restore.view(T, k, hidden_size)
    out = (out_restore * w.unsqueeze(-1)).sum(1)
    return out

# ==================== Main Test ====================
def test_sft_route_moe():
    print("=" * 80)
    print("Testing SFT Routed Experts with LoRA Fine-tuning")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Experts: {expert_num}, Routed per token: {n_routed_experts}")
    print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"  Sequence length: {qlen}, LoRA rank: {lora_rank}")
    print("=" * 80)

    # ==================== Initialize Weights ====================
    # Base weights (frozen during LoRA training)
    gate_proj_base = torch.randn(expert_num, intermediate_size, hidden_size,
                                 dtype=torch.bfloat16, requires_grad=False).contiguous()
    up_proj_base = torch.randn_like(gate_proj_base)
    down_proj_base = torch.randn(expert_num, hidden_size, intermediate_size,
                                 dtype=torch.bfloat16, requires_grad=False).contiguous()

    # LoRA adapters (trainable)
    gate_lora_A = torch.randn(expert_num, lora_rank, hidden_size,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    gate_lora_B = torch.randn(expert_num, intermediate_size, lora_rank,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    up_lora_A = torch.randn_like(gate_lora_A)
    up_lora_B = torch.randn_like(gate_lora_B)
    down_lora_A = torch.randn(expert_num, lora_rank, intermediate_size,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    down_lora_B = torch.randn(expert_num, hidden_size, lora_rank,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()

    # ==================== Setup C++ MoE ====================
    # Use the new SFT_ROUTE_MOE interface with separate base and LoRA weights
    cfg = cpuinfer_ext.sft_route_moe.SFT_ROUTE_MOEConfig(
        expert_num, n_routed_experts,
        hidden_size, intermediate_size,
        max_len,
        gate_proj_base.data_ptr(),
        up_proj_base.data_ptr(),
        down_proj_base.data_ptr(),
        gate_lora_A.data_ptr(),
        gate_lora_B.data_ptr(),
        up_lora_A.data_ptr(),
        up_lora_B.data_ptr(),
        down_lora_A.data_ptr(),
        down_lora_B.data_ptr(),
        lora_rank,
        lora_scaling
    )
    moe_cpp = cpuinfer_ext.sft_route_moe.SFT_ROUTE_AMXInt8_MOE(cfg)

    cpu_infer = cpuinfer_ext.CPUInfer(num_threads)
    cpu_infer.submit(moe_cpp.load_weights())
    cpu_infer.sync()

    # ==================== Setup PyTorch MoE ====================
    mlp_experts = []
    for e in range(expert_num):
        mlp = LoRAMLP(
            gate_proj_base[e], up_proj_base[e], down_proj_base[e],
            gate_lora_A[e], gate_lora_B[e],
            up_lora_A[e], up_lora_B[e],
            down_lora_A[e], down_lora_B[e]
        )
        mlp_experts.append(mlp)

    # ==================== Generate Test Data ====================
    expert_ids = torch.stack(
        [torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand(qlen, n_routed_experts, dtype=torch.float32).contiguous()

    input_pt = (torch.randn((qlen, hidden_size), dtype=dtype) / 100)\
               .detach().requires_grad_(True).contiguous()
    input_cpp = input_pt.detach().clone().requires_grad_(True).contiguous()

    # Print input data statistics
    print("\n[Input Data Statistics]")
    print(f"  Expert IDs shape: {expert_ids.shape}")
    print(f"  Expert IDs (first 3 tokens): {expert_ids[:3].tolist()}")
    expert_counts = torch.bincount(expert_ids.view(-1), minlength=expert_num)
    print(f"  Expert usage distribution: {expert_counts.tolist()}")
    print(f"  Weights shape: {weights.shape}, range: [{weights.min():.4f}, {weights.max():.4f}], mean: {weights.mean():.4f}")
    print(f"  Input shape: {input_pt.shape}, range: [{input_pt.min():.4f}, {input_pt.max():.4f}], mean: {input_pt.mean():.4f}")

    # ==================== Forward Pass ====================
    print("\n[Forward Pass]")

    # PyTorch reference
    t0 = time.time()
    out_ref = moe_lora_torch(input_pt, expert_ids, weights, mlp_experts)
    out_ref.retain_grad()
    t1 = time.time()

    # C++ implementation
    out_cpp = torch.empty_like(out_ref, dtype=dtype).contiguous()
    t2 = time.time()
    cpu_infer.submit(moe_cpp.forward(
        qlen, n_routed_experts,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_cpp.data_ptr(), out_cpp.data_ptr()))
    cpu_infer.sync()
    t3 = time.time()

    # Compare results
    diff_fwd = (out_cpp.to(torch.float32) - out_ref.to(torch.float32)).abs()
    rel_fwd = diff_fwd.mean() / out_ref.abs().mean()

    flop_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    print(f"  PyTorch time: {t1-t0:.4f}s | TFLOPS: {flop_fwd/(t1-t0)/1e12:.2f}")
    print(f"  C++ AMX time: {t3-t2:.4f}s | TFLOPS: {flop_fwd/(t3-t2)/1e12:.2f}")
    print(f"  Output shape: {out_ref.shape}")
    print(f"  PyTorch output range: [{out_ref.min():.4f}, {out_ref.max():.4f}], mean: {out_ref.mean():.4f}")
    print(f"  C++ output range: [{out_cpp.min():.4f}, {out_cpp.max():.4f}], mean: {out_cpp.mean():.4f}")
    print(f"  Abs diff - mean: {diff_fwd.mean():.6f}, max: {diff_fwd.max():.6f}, std: {diff_fwd.std():.6f}")
    print(f"  Relative error: {rel_fwd.item():.3e}")

    if rel_fwd.item() < 5e-2:
        print("  ✅ Forward pass matches!")
        print(f"  Sample outputs (PyTorch) : {out_ref}")
        print(f"  Sample outputs (C++)     : {out_cpp}")
    else:
        print("  ❌ Forward pass mismatch!")
        print(f"  Sample outputs (PyTorch) : {out_ref}")
        print(f"  Sample outputs (C++)     : {out_cpp}")

    # ==================== Backward Pass ====================
    print("\n[Backward Pass]")

    grad_out = torch.randn_like(out_ref, dtype=gradtype).contiguous()
    grad_out_cpp = grad_out.clone().contiguous()
    grad_in_cpp = torch.zeros_like(input_cpp, dtype=gradtype).contiguous()

    print(f"  Grad output shape: {grad_out.shape}, range: [{grad_out.min():.4f}, {grad_out.max():.4f}], mean: {grad_out.mean():.4f}")

    # PyTorch backward
    for p in (gate_lora_A, gate_lora_B, up_lora_A, up_lora_B,
              down_lora_A, down_lora_B, input_pt):
        if p.grad is not None:
            p.grad.zero_()

    t4 = time.time()
    out_ref.backward(grad_out, retain_graph=True)
    t5 = time.time()

    # C++ backward
    t6 = time.time()
    cpu_infer.submit(moe_cpp.backward(
        qlen, n_routed_experts,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_cpp.data_ptr(),
        grad_out_cpp.data_ptr(),
        grad_in_cpp.data_ptr()))
    cpu_infer.sync()
    t7 = time.time()

    # Compare results
    flop_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    gcpp = grad_in_cpp.to(torch.float32)
    gref = input_pt.grad.to(torch.float32) if input_pt.grad is not None else torch.zeros_like(input_pt, dtype=torch.float32)

    grad_diff = (gcpp - gref).abs()
    rel_bwd_cpp = grad_diff.mean() / gref.abs().mean()

    print(f"  PyTorch time: {t5-t4:.4f}s | TFLOPS: {flop_bwd/(t5-t4)/1e12:.2f}")
    print(f"  C++ AMX time: {t7-t6:.4f}s | TFLOPS: {flop_bwd/(t7-t6)/1e12:.2f}")
    print(f"  Grad input shape: {gref.shape}")
    print(f"  PyTorch grad range: [{gref.min():.6f}, {gref.max():.6f}], mean: {gref.mean():.6f}")
    print(f"  C++ grad range: [{gcpp.min():.6f}, {gcpp.max():.6f}], mean: {gcpp.mean():.6f}")
    print(f"  Grad diff - mean: {grad_diff.mean():.6f}, max: {grad_diff.max():.6f}, std: {grad_diff.std():.6f}")
    print(f"  Relative error: {rel_bwd_cpp.item():.3e}")

    if rel_bwd_cpp.item() < 5e-2:
        print("  ✅ Backward pass matches!")
        print(f"  Sample gradients (PyTorch) : {gref}")
        print(f"  Sample gradients (C++)     : {gcpp}")
    else:
        print("  ❌ Backward pass mismatch!")
        print(f"  Sample gradients (PyTorch) : {gref}")
        print(f"  Sample gradients (C++)     : {gcpp}")

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("Test Summary:")
    print(f"  Forward error:  {rel_fwd.item():.3e} {'✅' if rel_fwd.item() < 5e-2 else '❌'}")
    print(f"  Backward error: {rel_bwd_cpp.item():.3e} {'✅' if rel_bwd_cpp.item() < 5e-2 else '❌'}")
    print(f"  Speedup (Forward):  {(t1-t0)/(t3-t2):.2f}x")
    print(f"  Speedup (Backward): {(t5-t4)/(t7-t6):.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    torch.manual_seed(42)
    test_sft_route_moe()
