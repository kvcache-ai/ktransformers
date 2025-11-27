#!/usr/bin/env python
# coding=utf-8
'''
Description  : Complete unit test for LoRA gradient computation in routed_experts
Author       : KT-SFT Team
Date         : 2025-01-27
Version      : 2.0.0
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
LAYER_IDX = 0  # Layer index for debugging

# LoRA configuration
lora_rank = 8  # LoRA rank (r)
lora_alpha = 16  # LoRA alpha
lora_scaling = lora_alpha / lora_rank  # LoRA scaling factor

dtype = torch.bfloat16
gradtype = torch.bfloat16

import shutil
folder_path = "/home/lpl/kt-sft/debug_route_moe_lora_grads"
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

# ==================== Gradient Comparison Utility ====================
def compare_gradients(grad_cpp, grad_ref, name, threshold=5e-2):
    """Compare C++ and PyTorch gradients and return detailed statistics"""
    if grad_ref is None:
        print(f"  ❌ {name}: PyTorch gradient is None!")
        return False

    if grad_cpp is None:
        print(f"  ❌ {name}: C++ gradient is None!")
        return False

    grad_cpp_f32 = grad_cpp.to(torch.float32)
    grad_ref_f32 = grad_ref.to(torch.float32)

    diff = (grad_cpp_f32 - grad_ref_f32).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    # Relative error
    ref_mean = grad_ref_f32.abs().mean().item()
    rel_error = mean_diff / ref_mean if ref_mean > 1e-8 else mean_diff

    # Value ranges
    ref_min, ref_max = grad_ref_f32.min().item(), grad_ref_f32.max().item()
    ref_mean_val = grad_ref_f32.mean().item()
    cpp_min, cpp_max = grad_cpp_f32.min().item(), grad_cpp_f32.max().item()
    cpp_mean_val = grad_cpp_f32.mean().item()

    passed = rel_error < threshold

    print(f"\n  {name}:")
    print(f"    Shape: {grad_ref.shape}")
    print(f"    PyTorch: range=[{ref_min:.6f}, {ref_max:.6f}], mean={ref_mean_val:.6f}")
    print(f"    C++:     range=[{cpp_min:.6f}, {cpp_max:.6f}], mean={cpp_mean_val:.6f}")
    print(f"    Diff:    mean={mean_diff:.6f}, max={max_diff:.6f}, std={std_diff:.6f}")
    print(f"    Relative error: {rel_error:.3e} {'✅' if passed else '❌'}")

    return passed

# ==================== Main Test ====================
def test_sft_route_moe_lora_gradients():
    print("=" * 100)
    print("Complete Unit Test: SFT Routed Experts with LoRA Gradient Computation")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Experts: {expert_num}, Routed per token: {n_routed_experts}")
    print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    print(f"  Sequence length: {qlen}, LoRA rank: {lora_rank}, LoRA scaling: {lora_scaling}")
    print("=" * 100)

    # ==================== Initialize Weights ====================
    print("\n[Initializing Weights]")

    # Base weights (frozen during LoRA training)
    gate_proj_base = torch.randn(expert_num, intermediate_size, hidden_size,
                                 dtype=torch.bfloat16, requires_grad=False).contiguous()
    up_proj_base = torch.randn_like(gate_proj_base)
    down_proj_base = torch.randn(expert_num, hidden_size, intermediate_size,
                                 dtype=torch.bfloat16, requires_grad=False).contiguous()

    # LoRA adapters (trainable) - these will have gradients computed
    gate_lora_A = torch.randn(expert_num, lora_rank, hidden_size,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    gate_lora_B = torch.randn(expert_num, intermediate_size, lora_rank,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    up_lora_A = torch.randn_like(gate_lora_A).requires_grad_(True)
    up_lora_B = torch.randn_like(gate_lora_B).requires_grad_(True)
    down_lora_A = torch.randn(expert_num, lora_rank, intermediate_size,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()
    down_lora_B = torch.randn(expert_num, hidden_size, lora_rank,
                             dtype=torch.bfloat16, requires_grad=True).contiguous()

    print(f"  Base weights initialized (frozen)")
    print(f"  LoRA adapters initialized (trainable, 6 parameter sets)")

    # ==================== Initialize Gradient Buffers for C++ ====================
    print("\n[Initializing C++ Gradient Buffers]")

    grad_gate_lora_A_cpp = torch.zeros_like(gate_lora_A).contiguous()
    grad_gate_lora_B_cpp = torch.zeros_like(gate_lora_B).contiguous()
    grad_up_lora_A_cpp = torch.zeros_like(up_lora_A).contiguous()
    grad_up_lora_B_cpp = torch.zeros_like(up_lora_B).contiguous()
    grad_down_lora_A_cpp = torch.zeros_like(down_lora_A).contiguous()
    grad_down_lora_B_cpp = torch.zeros_like(down_lora_B).contiguous()

    print(f"  Gradient buffers created for 6 LoRA parameter sets")

    # ==================== Setup C++ MoE ====================
    print("\n[Setting up C++ MoE Operator]")

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
        lora_scaling,
        grad_gate_lora_A_cpp.data_ptr(),
        grad_gate_lora_B_cpp.data_ptr(),
        grad_up_lora_A_cpp.data_ptr(),
        grad_up_lora_B_cpp.data_ptr(),
        grad_down_lora_A_cpp.data_ptr(),
        grad_down_lora_B_cpp.data_ptr()
    )
    moe_cpp = cpuinfer_ext.sft_route_moe.SFT_ROUTE_AMXBF16_MOE(cfg)

    cpu_infer = cpuinfer_ext.CPUInfer(num_threads)
    cpu_infer.submit(moe_cpp.load_weights())
    cpu_infer.sync()

    print(f"  SFT_ROUTE_AMXBF16_MOE created and weights loaded")

    # ==================== Setup PyTorch MoE ====================
    print("\n[Setting up PyTorch Reference MoE]")

    mlp_experts = []
    for e in range(expert_num):
        mlp = LoRAMLP(
            gate_proj_base[e], up_proj_base[e], down_proj_base[e],
            gate_lora_A[e], gate_lora_B[e],
            up_lora_A[e], up_lora_B[e],
            down_lora_A[e], down_lora_B[e]
        )
        mlp_experts.append(mlp)

    print(f"  {expert_num} LoRAMLP experts created")

    # ==================== Generate Test Data ====================
    print("\n[Generating Test Data]")

    torch.manual_seed(42)
    expert_ids = torch.stack(
        [torch.randperm(expert_num)[:n_routed_experts] for _ in range(qlen)]).contiguous()
    weights = torch.rand(qlen, n_routed_experts, dtype=torch.float32).contiguous()

    input_pt = (torch.randn((qlen, hidden_size), dtype=dtype) / 100)\
               .detach().requires_grad_(True).contiguous()
    input_cpp = input_pt.detach().clone().requires_grad_(True).contiguous()

    expert_counts = torch.bincount(expert_ids.view(-1), minlength=expert_num)
    print(f"  Expert usage distribution: {expert_counts.tolist()}")
    print(f"  Input shape: {input_pt.shape}, range: [{input_pt.min():.4f}, {input_pt.max():.4f}]")

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

    # Compare forward results
    diff_fwd = (out_cpp.to(torch.float32) - out_ref.to(torch.float32)).abs()
    rel_fwd = diff_fwd.mean() / out_ref.abs().mean()

    flop_fwd = 6 * qlen * n_routed_experts * hidden_size * intermediate_size
    print(f"  PyTorch time: {t1-t0:.4f}s | TFLOPS: {flop_fwd/(t1-t0)/1e12:.2f}")
    print(f"  C++ AMX time: {t3-t2:.4f}s | TFLOPS: {flop_fwd/(t3-t2)/1e12:.2f}")
    print(f"  Speedup: {(t1-t0)/(t3-t2):.2f}x")
    print(f"  Relative error: {rel_fwd.item():.3e} {'✅' if rel_fwd.item() < 5e-2 else '❌'}")

    # ==================== Backward Pass ====================
    print("\n[Backward Pass - Computing Gradients]")

    grad_out = torch.randn_like(out_ref, dtype=gradtype).contiguous()
    grad_out_cpp = grad_out.clone().contiguous()
    grad_in_cpp = torch.zeros_like(input_cpp, dtype=gradtype).contiguous()

    # Zero out gradients for PyTorch
    for p in (gate_lora_A, gate_lora_B, up_lora_A, up_lora_B,
              down_lora_A, down_lora_B, input_pt):
        if p.grad is not None:
            p.grad.zero_()

    # PyTorch backward
    t4 = time.time()
    out_ref.backward(grad_out, retain_graph=True)
    t5 = time.time()

    # C++ backward - also computes LoRA gradients
    t6 = time.time()
    cpu_infer.submit(moe_cpp.backward(
        qlen, n_routed_experts,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_cpp.data_ptr(),
        grad_out_cpp.data_ptr(),
        grad_in_cpp.data_ptr()))
    cpu_infer.sync()
    t7 = time.time()

    flop_bwd = 18 * qlen * n_routed_experts * hidden_size * intermediate_size
    print(f"  PyTorch time: {t5-t4:.4f}s | TFLOPS: {flop_bwd/(t5-t4)/1e12:.2f}")
    print(f"  C++ AMX time: {t7-t6:.4f}s | TFLOPS: {flop_bwd/(t7-t6)/1e12:.2f}")
    print(f"  Speedup: {(t5-t4)/(t7-t6):.2f}x")

    # ==================== Gradient Validation ====================
    print("\n" + "=" * 100)
    print("GRADIENT VALIDATION - Comparing All LoRA Gradients")
    print("=" * 100)

    results = {}

    # Compare input gradients
    results['input'] = compare_gradients(
        grad_in_cpp, input_pt.grad, "Input Gradient (∂L/∂input)")

    # Compare all 6 LoRA parameter gradients
    results['gate_lora_A'] = compare_gradients(
        grad_gate_lora_A_cpp, gate_lora_A.grad, "Gate LoRA A Gradient (∂L/∂gate_lora_A)")

    results['gate_lora_B'] = compare_gradients(
        grad_gate_lora_B_cpp, gate_lora_B.grad, "Gate LoRA B Gradient (∂L/∂gate_lora_B)")

    results['up_lora_A'] = compare_gradients(
        grad_up_lora_A_cpp, up_lora_A.grad, "Up LoRA A Gradient (∂L/∂up_lora_A)")

    results['up_lora_B'] = compare_gradients(
        grad_up_lora_B_cpp, up_lora_B.grad, "Up LoRA B Gradient (∂L/∂up_lora_B)")

    results['down_lora_A'] = compare_gradients(
        grad_down_lora_A_cpp, down_lora_A.grad, "Down LoRA A Gradient (∂L/∂down_lora_A)")

    results['down_lora_B'] = compare_gradients(
        grad_down_lora_B_cpp, down_lora_B.grad, "Down LoRA B Gradient (∂L/∂down_lora_B)")

    # ==================== Final Summary ====================
    print("\n" + "=" * 100)
    print("FINAL TEST SUMMARY")
    print("=" * 100)

    all_passed = all(results.values()) and rel_fwd.item() < 5e-2

    print(f"\nForward Pass:")
    print(f"  Relative error: {rel_fwd.item():.3e} {'✅ PASSED' if rel_fwd.item() < 5e-2 else '❌ FAILED'}")
    print(f"  Speedup: {(t1-t0)/(t3-t2):.2f}x")

    print(f"\nBackward Pass (Gradients):")
    for name, passed in results.items():
        status = '✅ PASSED' if passed else '❌ FAILED'
        print(f"  {name:20s}: {status}")

    print(f"\n{'='*100}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! LoRA gradient computation is working correctly!")
    else:
        print("❌ SOME TESTS FAILED! Please check the gradient computation implementation.")
    print(f"{'='*100}\n")

    return all_passed

if __name__ == "__main__":
    torch.manual_seed(42)
    success = test_sft_route_moe_lora_gradients()
    sys.exit(0 if success else 1)
