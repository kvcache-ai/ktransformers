#!/usr/bin/env python
# coding=utf-8
"""NVFP4 experts with a clamped SwiGLU and with swigluoai, through NativeMoEWrapper.

A small ModelOpt-layout NVFP4 checkpoint (E2M1 nibbles, E4M3 per-16 block scales, F32 weight_scale_2) is written
to a temporary directory, loaded with method="NVFP4", and run on the CPU against a float32 reference computed from
the same bytes. Three activations: plain SiLU, SiLU with swiglu_limit (GLM-5 / DeepSeek-V4 style clamp: gate <= limit,
|up| <= limit), and swigluoai with swiglu_alpha and swiglu_limit (MiniMax M3: gate * sigmoid(alpha * gate) * (up + 1),
both halves clamped to +-limit as the kernel's act_fn does).
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default")

E, K, H, I = 8, 2, 256, 256
GS = 16
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def _make_proj(n, k, g):
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
    codes = torch.randint(0x28, 0x48, (n, k // GS), dtype=torch.uint8, generator=g)
    scale_2 = torch.tensor(0.5 + torch.rand((), generator=g).item(), dtype=torch.float32)
    return packed, codes.view(torch.float8_e4m3fn), scale_2


def _dequant(packed, block_scale, scale_2):
    n, kh = packed.shape
    vals = torch.stack((E2M1[(packed & 15).long()], E2M1[(packed >> 4).long()]), -1).reshape(n, kh * 2)
    # the loader folds both levels into one bf16 scale per group: the reference uses the same rounding
    s = (block_scale.to(torch.float32) * scale_2).to(torch.bfloat16).to(torch.float32)
    return vals * s.repeat_interleave(GS, dim=1)


@pytest.fixture(scope="module")
def checkpoint():
    g = torch.Generator().manual_seed(0)
    tensors, ref = {}, []
    for e in range(E):
        mats = []
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            packed, bs, s2 = _make_proj(n, k, g)
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            tensors[base + ".weight"] = packed
            tensors[base + ".weight_scale"] = bs
            tensors[base + ".weight_scale_2"] = s2
            mats.append(_dequant(packed, bs, s2))
        ref.append(mats)
    d = tempfile.mkdtemp(prefix="kt-nvfp4-swiglu-")
    save_file(tensors, os.path.join(d, "model.safetensors"))
    return d, ref


def _reference(x, ids, w, ref, alpha, limit):
    out = torch.zeros(x.shape[0], H)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            gw, uw, dw = ref[int(ids[t, j])]
            xf = x[t].float()
            gate, up = gw @ xf, uw @ xf
            if limit > 0:
                # the kernel's act_fn: swigluoai clamps both halves to +-limit; the SiLU clamp is DeepSeek-V4's
                # asymmetric one (gate <= limit, |up| <= limit)
                gate = gate.clamp(-limit, limit) if alpha > 0 else gate.clamp(max=limit)
                up = up.clamp(-limit, limit)
            if alpha > 0:
                act = gate * torch.sigmoid(alpha * gate) * (up + 1)
            else:
                act = torch.nn.functional.silu(gate) * up
            out[t] += w[t, j] * (dw @ act.to(torch.bfloat16).float())
    return out


@pytest.mark.parametrize("alpha,limit", [(0.0, 0.0), (0.0, 1.5), (1.702, 0.0), (1.702, 1.5)])
def test_nvfp4_swiglu(checkpoint, alpha, limit):
    from kt_kernel.experts import KTMoEWrapper

    path, ref = checkpoint
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=I,
        gpu_experts_mask=torch.zeros(E, dtype=torch.bool),
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=path,
        chunked_prefill_size=64,
        method="NVFP4",
        swiglu_limit=limit,
        swiglu_alpha=alpha,
    )
    wrapper.load_weights(torch.arange(E, dtype=torch.int64))

    torch.manual_seed(1)
    m = 6
    # gate and up of order 1: the clamp at 1.5 bites, and SiLU and swigluoai part ways
    x = (torch.randn(m, H) * 0.015).to(torch.bfloat16).contiguous()
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(m)]).to(torch.int64).contiguous()
    w = torch.rand(m, K, dtype=torch.float32).contiguous()
    y = torch.zeros(m, H, dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([m], dtype=torch.int32)
    wrapper.cpu_infer.submit(
        wrapper.moe.forward_task(bsz.data_ptr(), K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), False)
    )
    wrapper.cpu_infer.sync()

    expect = _reference(x, ids, w, ref, alpha, limit)
    rel = ((y.float() - expect).norm() / expect.norm()).item()
    assert rel < 0.02, f"alpha={alpha} limit={limit}: relative error {rel:.4f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
