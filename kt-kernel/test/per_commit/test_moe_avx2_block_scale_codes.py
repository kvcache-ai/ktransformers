#!/usr/bin/env python
# coding=utf-8
"""One-byte block scale codes in the AVX2 FP4 kernel (KT_SCALE_CODES=1) leave every output bit as it was.

A small MXFP4 checkpoint (E2M1 nibbles with UE8M0 per-32 scales, the DeepSeek-V4-Flash layout) and a small ModelOpt
NVFP4 one (E4M3 per-16 block scales, F32 weight_scale_2) are each loaded through NativeMoEWrapper without and with
the codes, and run on batches of 1 to 7 tokens: the 4-token blocks, the 2-3 remaining rows and the single row of the
group-32 and group-16 paths. The codes index a per-matrix table of the same FP32 scales, so the two runs must agree
bit for bit.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

pytest.importorskip("safetensors")
from safetensors.torch import save_file

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="default")

E, K, H, I = 8, 2, 256, 256


def _mxfp4_checkpoint(g):
    tensors = {}
    for e in range(E):
        for proj, (n, k) in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
            base = f"layers.0.ffn.experts.{e}.{proj}"
            tensors[base + ".weight"] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g).view(
                torch.int8
            )
            # UE8M0 exponents around 2^0: a spread of scales, at most 256 distinct values per matrix
            tensors[base + ".scale"] = torch.randint(118, 136, (n, k // 32), dtype=torch.uint8, generator=g)
    return tensors


def _nvfp4_checkpoint(g):
    tensors = {}
    for e in range(E):
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            tensors[base + ".weight"] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=g)
            tensors[base + ".weight_scale"] = torch.randint(
                0x28, 0x48, (n, k // 16), dtype=torch.uint8, generator=g
            ).view(torch.float8_e4m3fn)
            tensors[base + ".weight_scale_2"] = torch.tensor(
                0.5 + torch.rand((), generator=g).item(), dtype=torch.float32
            )
    return tensors


@pytest.fixture(scope="module", params=["MXFP4", "NVFP4"])
def checkpoint(request):
    g = torch.Generator().manual_seed(0)
    tensors = _mxfp4_checkpoint(g) if request.param == "MXFP4" else _nvfp4_checkpoint(g)
    d = tempfile.mkdtemp(prefix=f"kt-{request.param.lower()}-codes-")
    save_file(tensors, os.path.join(d, "model.safetensors"))
    return request.param, d


def _run(method, path, codes, batches, capfd):
    from kt_kernel.experts import KTMoEWrapper

    os.environ["KT_SCALE_CODES"] = "1" if codes else "0"
    try:
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
            method=method,
        )
        wrapper.load_weights(torch.arange(E, dtype=torch.int64))
    finally:
        os.environ.pop("KT_SCALE_CODES", None)
    loaded = capfd.readouterr().out
    outs = []
    for x, ids, w in batches:
        m = x.shape[0]
        y = torch.zeros(m, H, dtype=torch.bfloat16).contiguous()
        bsz = torch.tensor([m], dtype=torch.int32)
        wrapper.cpu_infer.submit(
            wrapper.moe.forward_task(bsz.data_ptr(), K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), False)
        )
        wrapper.cpu_infer.sync()
        outs.append(y.clone())
    return outs, loaded


def test_scale_codes_bit_identical(checkpoint, capfd):
    method, path = checkpoint
    torch.manual_seed(1)
    batches = []
    for m in (1, 2, 3, 4, 5, 7):
        x = (torch.randn(m, H) * 0.05).to(torch.bfloat16).contiguous()
        ids = torch.stack([torch.randperm(E)[:K] for _ in range(m)]).to(torch.int64).contiguous()
        w = torch.rand(m, K, dtype=torch.float32).contiguous()
        batches.append((x, ids, w))

    plain, _ = _run(method, path, False, batches, capfd)
    coded, loaded = _run(method, path, True, batches, capfd)
    assert f"{3 * E} of {3 * E} matrices read one-byte scale codes" in loaded, loaded
    for (x, _, _), a, b in zip(batches, plain, coded):
        assert a.abs().sum() > 0, f"{method} m={x.shape[0]}: zero output"
        assert torch.equal(
            a.view(torch.int16), b.view(torch.int16)
        ), f"{method} m={x.shape[0]}: {(a != b).sum().item()} of {a.numel()} outputs differ with the codes"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
