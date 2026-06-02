"""End-to-end MXFP8 MoE validation against the native MiniMax-M3-Preview ckpt.

Loads layer-`LAYER_ID` experts via :class:`MXFP8SafeTensorLoader`, runs the AMX
MXFP8 backend with swigluoai activation, and compares against a torch reference
that dequantizes FP8 E4M3fn weights with ue8m0 group scales.

Usage:
    python test_mxfp8_moe_m3.py --weight-path /mnt/data/models/Minimax-M3-preview [--layer 3]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/build")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/python")

from kt_kernel import kt_kernel_ext  # noqa: E402
from kt_kernel.utils.loader import MXFP8SafeTensorLoader  # noqa: E402


# ---- Reference implementation ----

def dequantize_mxfp8(weight_u8: torch.Tensor, scale_u8: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize [N, K] FP8 E4M3fn (as uint8) with [N, K/gs] ue8m0 scales → [N, K] BF16."""
    n, k = weight_u8.shape
    assert k % group_size == 0
    assert scale_u8.shape == (n, k // group_size)

    w_fp32 = weight_u8.view(torch.float8_e4m3fn).float()
    s_fp32 = (2.0 ** (scale_u8.to(torch.int32) - 127)).float()
    s_full = s_fp32.repeat_interleave(group_size, dim=-1)
    return (w_fp32 * s_full).to(torch.bfloat16).contiguous()


def reference_mlp_m3(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                     alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Single-expert MLP with swigluoai activation."""
    g = torch.mm(x.float(), gate.float().t()).clamp(-limit, limit)
    u = torch.mm(x.float(), up.float().t()).clamp(-limit, limit)
    act = g * torch.sigmoid(g * alpha) * (u + 1.0)
    return torch.mm(act, down.float().t())


def reference_moe_m3(hidden: torch.Tensor, expert_ids: torch.Tensor, weights: torch.Tensor,
                     gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor,
                     alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """Full MoE forward: route tokens to experts, compute, weighted sum."""
    out = torch.zeros(hidden.shape[0], down_w.shape[1], dtype=torch.float32)
    for tok in range(hidden.shape[0]):
        for slot in range(expert_ids.shape[1]):
            eid = int(expert_ids[tok, slot])
            if eid < 0:
                continue
            w = float(weights[tok, slot])
            y = reference_mlp_m3(hidden[tok:tok+1], gate_w[eid], up_w[eid], down_w[eid], alpha, limit)
            out[tok] += w * y[0]
    return out.to(hidden.dtype)


# ---- Main ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MXFP8 MoE E2E test for MiniMax M3")
    p.add_argument("--weight-path", required=True, help="Path to Minimax-M3-preview safetensors directory.")
    p.add_argument("--layer", type=int, default=3, help="Layer index to validate (default: 3, first MoE layer).")
    p.add_argument("--qlen", type=int, default=1, help="Number of tokens to test.")
    p.add_argument("--top-k", type=int, default=4, help="num_experts_per_tok (M3 default 4).")
    p.add_argument("--cpu-threads", type=int, default=32)
    p.add_argument("--max-experts", type=int, default=0, help="Cap number of experts loaded (0 = all).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(42)

    print(f"[M3-MXFP8] Loading layer {args.layer} from {args.weight_path}")
    loader = MXFP8SafeTensorLoader(args.weight_path)
    weights = loader.load_experts(f"language_model.model.layers.{args.layer}")

    expert_num = len(weights["gate"])
    if args.max_experts and args.max_experts < expert_num:
        for k in ("gate", "up", "down", "gate_scale", "up_scale", "down_scale"):
            weights[k] = weights[k][:args.max_experts]
        expert_num = args.max_experts
    print(f"[M3-MXFP8] expert_num={expert_num}")

    gate0 = weights["gate"][0]
    down0 = weights["down"][0]
    intermediate_size = gate0.shape[0]  # w1: [I, K]
    hidden_size = gate0.shape[1]        # FP8: 1 byte/element, no /2
    assert down0.shape == (hidden_size, intermediate_size), f"unexpected down shape {down0.shape}"

    group_size = hidden_size // weights["gate_scale"][0].shape[1]
    print(f"[M3-MXFP8] hidden={hidden_size} inter={intermediate_size} gs={group_size}")
    assert group_size == 32, f"Expected group_size=32, got {group_size}"

    physical_to_logical = torch.arange(expert_num, dtype=torch.int64).contiguous()

    # ---- C++ MXFP8 forward ----
    cpu_infer = kt_kernel_ext.CPUInfer(args.cpu_threads)
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, args.top_k, hidden_size, intermediate_size, 0)
    cfg.layer_idx = args.layer
    cfg.max_len = max(args.qlen, 1)
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 8
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False
    cfg.swiglu_alpha = 1.702
    cfg.swiglu_limit = 7.0

    cfg.gate_projs = [[t.data_ptr() for t in weights["gate"]]]
    cfg.up_projs = [[t.data_ptr() for t in weights["up"]]]
    cfg.down_projs = [[t.data_ptr() for t in weights["down"]]]
    cfg.gate_scales = [[t.data_ptr() for t in weights["gate_scale"]]]
    cfg.up_scales = [[t.data_ptr() for t in weights["up_scale"]]]
    cfg.down_scales = [[t.data_ptr() for t in weights["down_scale"]]]

    moe = kt_kernel_ext.moe.AMXMXFP8_KGroup_MOE(cfg)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()
    print("[M3-MXFP8] C++ weights loaded")

    qlen = args.qlen
    top_k = args.top_k
    bsz = torch.tensor([qlen], dtype=torch.int32)
    expert_ids = torch.stack([torch.randperm(expert_num)[:top_k] for _ in range(qlen)]).to(torch.int64).contiguous()
    routing = torch.randn((qlen, top_k), dtype=torch.float32).abs().contiguous()
    routing = routing / routing.sum(dim=-1, keepdim=True)  # normalize weights
    x = (torch.randn((qlen, hidden_size), dtype=torch.bfloat16) * 0.01).contiguous()
    y_cpp = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

    cpu_infer.submit(
        moe.forward_task(
            bsz.data_ptr(), top_k, expert_ids.data_ptr(), routing.data_ptr(),
            x.data_ptr(), y_cpp.data_ptr(), False,
        )
    )
    cpu_infer.sync()
    print("[M3-MXFP8] C++ forward done")

    # ---- Torch reference ----
    print("[M3-MXFP8] Building torch reference (dequantizing all loaded experts)...")
    gate_bf16 = torch.stack([dequantize_mxfp8(weights["gate"][i], weights["gate_scale"][i], group_size)
                             for i in range(expert_num)])
    up_bf16 = torch.stack([dequantize_mxfp8(weights["up"][i], weights["up_scale"][i], group_size)
                           for i in range(expert_num)])
    down_bf16 = torch.stack([dequantize_mxfp8(weights["down"][i], weights["down_scale"][i], group_size)
                             for i in range(expert_num)])

    y_ref = reference_moe_m3(x, expert_ids, routing, gate_bf16, up_bf16, down_bf16,
                             alpha=1.702, limit=7.0)

    # ---- Compare ----
    diff = (y_cpp.float() - y_ref.float()).abs()
    ref_mag = y_ref.float().abs().mean() + 1e-12
    rel = diff.mean() / ref_mag
    print(f"[M3-MXFP8] mean abs diff = {diff.mean().item():.4e}")
    print(f"[M3-MXFP8] max  abs diff = {diff.max().item():.4e}")
    print(f"[M3-MXFP8] ref  mean mag = {ref_mag.item():.4e}")
    print(f"[M3-MXFP8] rel mean diff = {rel.item()*100:.3f}%")
    print(f"[M3-MXFP8] cpp[:8]  = {y_cpp.flatten()[:8].tolist()}")
    print(f"[M3-MXFP8] ref[:8]  = {y_ref.flatten()[:8].tolist()}")

    passed = rel.item() < 0.10
    print(f"[M3-MXFP8] {'PASS' if passed else 'FAIL'} (threshold: 10%)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
