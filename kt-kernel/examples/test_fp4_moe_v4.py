"""End-to-end MXFP4 MoE validation against the native DeepSeek-V4-Flash ckpt.

Loads layer-`LAYER_ID` experts via :class:`MXFP4SafeTensorLoader`, runs the AMX
FP4 backend, and compares against a torch reference that dequantizes the same
nibble-packed weights with the OCP E2M1 LUT.

Usage:
    python test_fp4_moe_v4.py --weight-path /path/to/DeepSeek-V4-Flash [--layer 1]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import torch

# Allow running from kt-kernel/examples without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/build")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/python")

from kt_kernel import kt_kernel_ext  # noqa: E402
from kt_kernel.utils.loader import MXFP4SafeTensorLoader  # noqa: E402

# OCP E2M1 codepoints in our LUT order (matches operators/amx/fp4-moe.hpp).
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def dequantize_mxfp4(weight_u8: torch.Tensor, scale_bf16: torch.Tensor, group_size: int) -> torch.Tensor:
    """Decode a [N, K/2] uint8 tensor of nibble-packed E2M1 with [N, K/gs] bf16
    scales into a [N, K] bf16 weight tensor.

    Layout (matches kernel's mxfp4_to_bf16_32): byte `b` low nibble = element K=2b,
    high nibble = element K=2b+1.
    """
    n, k_packed = weight_u8.shape
    k = k_packed * 2
    assert k % group_size == 0, f"K={k} must be divisible by group_size={group_size}"
    assert scale_bf16.shape == (n, k // group_size)

    lo = (weight_u8 & 0x0F).to(torch.long)
    hi = ((weight_u8 >> 4) & 0x0F).to(torch.long)
    nibbles = torch.stack([lo, hi], dim=-1).view(n, k)  # interleave back to K order
    decoded = E2M1_VALUES.to(weight_u8.device)[nibbles]  # [N, K] fp32

    scale_fp32 = scale_bf16.to(torch.float32)
    scale_full = scale_fp32.repeat_interleave(group_size, dim=-1)  # [N, K]
    return (decoded * scale_full).to(torch.bfloat16).contiguous()


def reference_mlp(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    g = torch.mm(x, gate.t())
    u = torch.mm(x, up.t())
    silu = g / (1.0 + torch.exp(-g.float())).to(g.dtype)
    return torch.mm(silu * u, down.t())


def reference_moe(
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate_w: torch.Tensor,  # [E, N, K]
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for tok in range(hidden.shape[0]):
        for slot in range(expert_ids.shape[1]):
            eid = int(expert_ids[tok, slot])
            w = float(weights[tok, slot])
            x = hidden[tok : tok + 1]
            y = reference_mlp(x, gate_w[eid], up_w[eid], down_w[eid])
            out[tok] += w * y[0].float()
    return out.to(hidden.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weight-path", required=True, help="Path to DeepSeek-V4-Flash safetensors directory.")
    p.add_argument("--layer", type=int, default=1, help="Layer index to validate (default: 1).")
    p.add_argument("--qlen", type=int, default=1, help="Number of tokens to test.")
    p.add_argument("--top-k", type=int, default=6, help="num_experts_per_tok (V4 default 6).")
    p.add_argument("--cpu-threads", type=int, default=32)
    p.add_argument("--max-experts", type=int, default=0, help="Cap number of experts loaded (0 = all).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(0)

    print(f"[V4-MXFP4] Loading layer {args.layer} from {args.weight_path}")
    loader = MXFP4SafeTensorLoader(args.weight_path)
    weights = loader.load_experts(f"model.layers.{args.layer}")

    expert_num = len(weights["gate"])
    if args.max_experts and args.max_experts < expert_num:
        for k in ("gate", "up", "down", "gate_scale", "up_scale", "down_scale"):
            weights[k] = weights[k][: args.max_experts]
        expert_num = args.max_experts
    print(f"[V4-MXFP4] expert_num={expert_num}")

    gate0 = weights["gate"][0]
    down0 = weights["down"][0]
    intermediate_size = gate0.shape[0]
    hidden_size = gate0.shape[1] * 2  # nibble-packed K
    assert down0.shape == (hidden_size, intermediate_size // 2), f"unexpected down shape {down0.shape}"

    group_size = hidden_size // weights["gate_scale"][0].shape[1]
    print(f"[V4-MXFP4] hidden={hidden_size} inter={intermediate_size} gs={group_size}")
    assert group_size == 32, "MXFP4 backend hard-codes group_size=32"

    physical_to_logical = torch.arange(expert_num, dtype=torch.int64).contiguous()

    # ----- AMX FP4 forward -----
    cpu_infer = kt_kernel_ext.CPUInfer(args.cpu_threads)
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, args.top_k, hidden_size, intermediate_size, 0)
    cfg.layer_idx = args.layer
    cfg.max_len = max(args.qlen, 1)
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = group_size
    cfg.quant_config.zero_point = False

    cfg.gate_projs = [[t.data_ptr() for t in weights["gate"]]]
    cfg.up_projs = [[t.data_ptr() for t in weights["up"]]]
    cfg.down_projs = [[t.data_ptr() for t in weights["down"]]]
    cfg.gate_scales = [[t.data_ptr() for t in weights["gate_scale"]]]
    cfg.up_scales = [[t.data_ptr() for t in weights["up_scale"]]]
    cfg.down_scales = [[t.data_ptr() for t in weights["down_scale"]]]

    moe = kt_kernel_ext.moe.AMXFP4_KGroup_MOE(cfg)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpu_infer.sync()

    qlen = args.qlen
    top_k = args.top_k
    bsz = torch.tensor([qlen], dtype=torch.int32)
    expert_ids = torch.stack([torch.randperm(expert_num)[:top_k] for _ in range(qlen)]).to(torch.int32).contiguous()
    routing = torch.randn((qlen, top_k), dtype=torch.float32).contiguous()
    x = (torch.randn((qlen, hidden_size), dtype=torch.bfloat16) / 100).contiguous()
    y_amx = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()

    cpu_infer.submit(
        moe.forward_task(
            bsz.data_ptr(), top_k, expert_ids.data_ptr(), routing.data_ptr(),
            x.data_ptr(), y_amx.data_ptr(), False,
        )
    )
    cpu_infer.sync()

    # ----- Torch reference (dequantize same nibbles + scales) -----
    print("[V4-MXFP4] Building torch reference (dequantizing all loaded experts)…")
    gate_bf16 = torch.stack([dequantize_mxfp4(weights["gate"][i], weights["gate_scale"][i], group_size) for i in range(expert_num)])
    up_bf16 = torch.stack([dequantize_mxfp4(weights["up"][i], weights["up_scale"][i], group_size) for i in range(expert_num)])
    down_bf16 = torch.stack([dequantize_mxfp4(weights["down"][i], weights["down_scale"][i], group_size) for i in range(expert_num)])

    y_ref = reference_moe(x, expert_ids, routing, gate_bf16, up_bf16, down_bf16)

    diff = (y_amx.float() - y_ref.float()).abs()
    rel = diff.mean() / (y_ref.float().abs().mean() + 1e-12)
    print(f"[V4-MXFP4] mean abs diff = {diff.mean().item():.4e}")
    print(f"[V4-MXFP4] max  abs diff = {diff.max().item():.4e}")
    print(f"[V4-MXFP4] rel mean diff = {rel.item()*100:.3f}%")
    print(f"[V4-MXFP4] amx[:8]  = {y_amx.flatten()[:8]}")
    print(f"[V4-MXFP4] ref[:8]  = {y_ref.flatten()[:8]}")

    return 0 if rel.item() < 0.10 else 1


if __name__ == "__main__":
    sys.exit(main())
