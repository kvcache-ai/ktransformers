"""AVX2 MXFP8 MoE validation for MiniMax-M3-Preview.

Forces the AVX2 backend (`kt_kernel_ext.moe.AVX2MXFP8_MOE`) for layer-`LAYER`
experts, compares output against a torch dequant+matmul reference.

Optional `--compare-amx` flag runs both AMX and AVX2 backends on identical
inputs and asserts numerical equivalence. The two paths share buffer layout
and do the same FMA arithmetic; observed differences should be at BF16-noise
level (typical mean abs ~ 1e-4).

Usage:
    python test_mxfp8_moe_avx2.py --weight-path /mnt/data/models/Minimax-M3-preview
    python test_mxfp8_moe_avx2.py --weight-path ... --compare-amx
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


# ---- Reference implementation (shared with test_mxfp8_moe_m3.py) ----

def dequantize_mxfp8(weight_u8: torch.Tensor, scale_u8: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Dequantize [N, K] FP8 E4M3fn (as uint8) with [N, K/gs] ue8m0 scales -> [N, K] BF16."""
    n, k = weight_u8.shape
    assert k % group_size == 0
    assert scale_u8.shape == (n, k // group_size)
    w_fp32 = weight_u8.view(torch.float8_e4m3fn).float()
    s_fp32 = (2.0 ** (scale_u8.to(torch.int32) - 127)).float()
    s_full = s_fp32.repeat_interleave(group_size, dim=-1)
    return (w_fp32 * s_full).to(torch.bfloat16).contiguous()


def reference_mlp_m3(x, gate, up, down, alpha=1.702, limit=7.0):
    g = torch.mm(x.float(), gate.float().t()).clamp(-limit, limit)
    u = torch.mm(x.float(), up.float().t()).clamp(-limit, limit)
    act = g * torch.sigmoid(g * alpha) * (u + 1.0)
    return torch.mm(act, down.float().t())


def reference_moe_m3(hidden, expert_ids, weights, gate_w, up_w, down_w, alpha=1.702, limit=7.0):
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


# ---- Backend dispatch ----

def _backend_cls(backend_name: str):
    if backend_name == "amx":
        cls = getattr(kt_kernel_ext.moe, "AMXMXFP8_KGroup_MOE", None)
        if cls is None:
            raise RuntimeError("AMXMXFP8_KGroup_MOE not in .so — rebuild with AVX-512 + VBMI + AMX")
        return cls
    if backend_name == "avx2":
        cls = getattr(kt_kernel_ext.moe, "AVX2MXFP8_MOE", None)
        if cls is None:
            raise RuntimeError("AVX2MXFP8_MOE not in .so — rebuild with the AVX2 MXFP8 wiring (PR #2041 + this fix)")
        return cls
    raise ValueError(f"unknown backend {backend_name}")


def run_backend(backend_name, weights, expert_num, top_k, hidden_size, intermediate_size,
                layer_idx, qlen, cpu_threads, x, expert_ids, routing, physical_to_logical):
    cpu_infer = kt_kernel_ext.CPUInfer(cpu_threads)
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, top_k, hidden_size, intermediate_size, 0)
    cfg.layer_idx = layer_idx
    cfg.max_len = max(qlen, 1)
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 8
    cfg.quant_config.group_size = 32
    cfg.quant_config.zero_point = False
    cfg.swiglu_alpha = 1.702
    cfg.swiglu_limit = 7.0

    cfg.gate_projs = [[t.data_ptr() for t in weights["gate"]]]
    cfg.up_projs = [[t.data_ptr() for t in weights["up"]]]
    cfg.down_projs = [[t.data_ptr() for t in weights["down"]]]
    cfg.gate_scales = [[t.data_ptr() for t in weights["gate_scale"]]]
    cfg.up_scales = [[t.data_ptr() for t in weights["up_scale"]]]
    cfg.down_scales = [[t.data_ptr() for t in weights["down_scale"]]]

    moe = _backend_cls(backend_name)(cfg)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()

    bsz = torch.tensor([qlen], dtype=torch.int32)
    y = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    cpu_infer.submit(
        moe.forward_task(bsz.data_ptr(), top_k, expert_ids.data_ptr(), routing.data_ptr(),
                         x.data_ptr(), y.data_ptr(), False)
    )
    cpu_infer.sync()
    return y


def parse_args():
    p = argparse.ArgumentParser(description="MXFP8 MoE AVX2 backend test for MiniMax M3")
    p.add_argument("--weight-path", required=True)
    p.add_argument("--layer", type=int, default=3, help="Layer index (default 3 = first MoE layer)")
    p.add_argument("--qlen", type=int, default=1)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=32)
    p.add_argument("--max-experts", type=int, default=0, help="Cap experts loaded (0=all)")
    p.add_argument("--compare-amx", action="store_true",
                   help="Also run AMX backend and assert numerical equivalence with AVX2.")
    p.add_argument("--ref-threshold", type=float, default=0.10,
                   help="Max relative error vs torch reference (default 10%).")
    p.add_argument("--equiv-threshold", type=float, default=0.01,
                   help="Max relative error AMX vs AVX2 (default 1%).")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)

    print(f"[AVX2-MXFP8] Loading layer {args.layer} from {args.weight_path}")
    loader = MXFP8SafeTensorLoader(args.weight_path)
    weights = loader.load_experts(f"language_model.model.layers.{args.layer}")

    expert_num = len(weights["gate"])
    if args.max_experts and args.max_experts < expert_num:
        for k in ("gate", "up", "down", "gate_scale", "up_scale", "down_scale"):
            weights[k] = weights[k][:args.max_experts]
        expert_num = args.max_experts

    gate0 = weights["gate"][0]
    intermediate_size = gate0.shape[0]
    hidden_size = gate0.shape[1]
    group_size = hidden_size // weights["gate_scale"][0].shape[1]
    print(f"[AVX2-MXFP8] expert_num={expert_num} hidden={hidden_size} inter={intermediate_size} gs={group_size}")
    assert group_size == 32, f"Expected group_size=32, got {group_size}"

    physical_to_logical = torch.arange(expert_num, dtype=torch.int64).contiguous()

    qlen = args.qlen
    top_k = args.top_k
    expert_ids = torch.stack([torch.randperm(expert_num)[:top_k] for _ in range(qlen)]).to(torch.int64).contiguous()
    routing = torch.randn((qlen, top_k), dtype=torch.float32).abs().contiguous()
    routing = routing / routing.sum(dim=-1, keepdim=True)
    x = (torch.randn((qlen, hidden_size), dtype=torch.bfloat16) * 0.01).contiguous()

    # ---- AVX2 forward ----
    print("[AVX2-MXFP8] Running AVX2 backend...")
    y_avx2 = run_backend("avx2", weights, expert_num, top_k, hidden_size, intermediate_size,
                         args.layer, qlen, args.cpu_threads, x, expert_ids, routing,
                         physical_to_logical)

    # ---- Torch reference ----
    print("[AVX2-MXFP8] Building torch reference (dequant+matmul)...")
    gate_bf16 = torch.stack([dequantize_mxfp8(weights["gate"][i], weights["gate_scale"][i], group_size)
                             for i in range(expert_num)])
    up_bf16 = torch.stack([dequantize_mxfp8(weights["up"][i], weights["up_scale"][i], group_size)
                           for i in range(expert_num)])
    down_bf16 = torch.stack([dequantize_mxfp8(weights["down"][i], weights["down_scale"][i], group_size)
                             for i in range(expert_num)])
    y_ref = reference_moe_m3(x, expert_ids, routing, gate_bf16, up_bf16, down_bf16,
                             alpha=1.702, limit=7.0)

    diff_ref = (y_avx2.float() - y_ref.float()).abs()
    ref_mag = y_ref.float().abs().mean() + 1e-12
    rel_ref = diff_ref.mean() / ref_mag
    print(f"[AVX2-MXFP8] vs ref:   mean abs={diff_ref.mean().item():.4e}  max abs={diff_ref.max().item():.4e}  rel={rel_ref.item()*100:.3f}%")
    pass_ref = rel_ref.item() < args.ref_threshold

    # ---- (optional) AMX vs AVX2 equivalence ----
    pass_eq = True
    if args.compare_amx:
        print("[AVX2-MXFP8] Running AMX backend for equivalence check...")
        y_amx = run_backend("amx", weights, expert_num, top_k, hidden_size, intermediate_size,
                            args.layer, qlen, args.cpu_threads, x, expert_ids, routing,
                            physical_to_logical)
        diff_eq = (y_amx.float() - y_avx2.float()).abs()
        amx_mag = y_amx.float().abs().mean() + 1e-12
        rel_eq = diff_eq.mean() / amx_mag
        print(f"[AVX2-MXFP8] vs AMX:  mean abs={diff_eq.mean().item():.4e}  max abs={diff_eq.max().item():.4e}  rel={rel_eq.item()*100:.3f}%")
        pass_eq = rel_eq.item() < args.equiv_threshold

    print(f"[AVX2-MXFP8] vs ref:  {'PASS' if pass_ref else 'FAIL'} (rel < {args.ref_threshold*100:.1f}%)")
    if args.compare_amx:
        print(f"[AVX2-MXFP8] vs AMX:  {'PASS' if pass_eq else 'FAIL'} (rel < {args.equiv_threshold*100:.1f}%)")
    return 0 if (pass_ref and pass_eq) else 1


if __name__ == "__main__":
    sys.exit(main())
