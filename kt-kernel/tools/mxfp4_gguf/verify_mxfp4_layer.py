#!/usr/bin/env python3
"""
Acceptance test for the MXFP4 converter: prove the GGUF repack is LOSSLESS.

Two checks:
  1. unit: random [N,K/2] native bytes -> repack -> dequant(GGUF semantics) must
     equal dequant(native semantics), element-wise. Pure logic, no files.
  2. layer: for a real converted layer GGUF, dequant each proj tensor and compare
     element-wise to the native checkpoint dequant (a subset of experts).

Native semantics  : value = FP4_TABLE[nibble] * 2^(e-127), byte i -> Kpos 2i(lo),2i+1(hi)
GGUF  semantics   : value = kvalues_mxfp4[nibble] * 2^(e-128), qs[j] -> Kpos j(lo),j+16(hi)
These are algebraically identical (kvalues_mxfp4 = 2*FP4_TABLE), so equality is bit-exact.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import json

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GGUF_PY = Path(os.environ.get("KT_GGUF_PY", _REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"))
if not _GGUF_PY.is_dir():
    raise SystemExit(
        f"gguf-py not found at {_GGUF_PY}.\n"
        "Initialize the submodule first:\n"
        f"  git -C {_REPO_ROOT} submodule update --init --progress third_party/llama.cpp\n"
        "or point KT_GGUF_PY at a gguf-py that knows GGML_TYPE_MXFP4 (id 39)."
    )
sys.path.insert(0, str(_GGUF_PY))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FP4_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
KVALUES_MXFP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)


def e8m0_to_fp32(e: np.ndarray) -> np.ndarray:
    """2^(e-127) exactly (matches loader _ue8m0_to_bf16 semantics)."""
    bits = (e.astype(np.uint32)) << 23
    return bits.view(np.float32)


def e8m0_to_fp32_half(e: np.ndarray) -> np.ndarray:
    """2^(e-128) == 2^(e-127)*0.5 (matches ggml_e8m0_to_fp32_half)."""
    e = e.astype(np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - 1) << 23).astype(np.uint32)
    return bits.view(np.float32)


def dequant_native(w_u8: np.ndarray, s_u8: np.ndarray) -> np.ndarray:
    """[N,K/2] bytes + [N,K/32] e8m0 -> [N,K] float32 (native consecutive nibble order)."""
    N, kh = w_u8.shape
    K = kh * 2
    nb = K // 32
    lo = FP4_TABLE[(w_u8 & 0x0F)]   # [N, K/2] -> Kpos 0,2,...
    hi = FP4_TABLE[(w_u8 >> 4)]     # [N, K/2] -> Kpos 1,3,...
    vals = np.empty((N, K), dtype=np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    scale = e8m0_to_fp32(s_u8)               # [N, nb]
    scale = np.repeat(scale, 32, axis=1)     # [N, K]
    return vals * scale


def dequant_gguf_blocks(packed_row: np.ndarray, K: int) -> np.ndarray:
    """[N, nb*17] uint8 -> [N, K] float32 (GGUF half-block order)."""
    N = packed_row.shape[0]
    nb = K // 32
    blk = packed_row.reshape(N, nb, 17)
    e = blk[..., 0]                  # [N, nb]
    qs = blk[..., 1:]               # [N, nb, 16]
    lo = KVALUES_MXFP4[(qs & 0x0F)].astype(np.float32)   # Kpos j (0..15)
    hi = KVALUES_MXFP4[(qs >> 4)].astype(np.float32)     # Kpos j+16
    half = e8m0_to_fp32_half(e)[..., None]               # [N, nb, 1]
    g = np.concatenate([lo * half, hi * half], axis=-1)   # [N, nb, 32]
    return g.reshape(N, K)


def unit_test(seed: int = 0) -> None:
    from convert_mxfp4_layer_to_gguf import _repack_consecutive_to_halfblock
    rng = np.random.default_rng(seed)
    N, K = 7, 128
    w = rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8)
    s = rng.integers(100, 140, size=(N, K // 32), dtype=np.uint8)
    qs = _repack_consecutive_to_halfblock(w)                # [N, K/2]
    nb = K // 32
    packed = np.concatenate([s.reshape(N, nb, 1), qs.reshape(N, nb, 16)], axis=-1).reshape(N, nb * 17)
    a = dequant_native(w, s)
    b = dequant_gguf_blocks(packed, K)
    if not np.array_equal(a, b):
        bad = int((a != b).sum())
        raise SystemExit(f"[unit] FAIL: {bad}/{a.size} elements differ (max abs {np.abs(a-b).max()})")
    print(f"[unit] PASS: repack lossless, {a.size} elements bit-exact (N={N},K={K})")


def layer_test(gguf_path: Path, model_dir: Path, layer_idx: int, n_experts_check: int) -> None:
    import gguf
    from safetensors import safe_open
    import torch
    from convert_mxfp4_layer_to_gguf import _load_weight_map, _detect_experts_prefix, _open_shard, _as_u8

    reader = gguf.GGUFReader(str(gguf_path))
    tmap = {t.name: t for t in reader.tensors}
    weight_map = _load_weight_map(model_dir)
    prefix = _detect_experts_prefix(weight_map, layer_idx)
    cache: dict = {}

    # (gguf tensor name, native proj, K)
    cfg = json.loads((model_dir / "config.json").read_text())
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["moe_intermediate_size"])
    projs = [(f"blk.{layer_idx}.ffn_gate_exps.weight", "w1", hidden),
             (f"blk.{layer_idx}.ffn_up_exps.weight", "w3", hidden),
             (f"blk.{layer_idx}.ffn_down_exps.weight", "w2", inter)]

    for gname, proj, K in projs:
        t = tmap[gname]
        # GGUF reader gives ne order [K, N, E] (fastest dim first); data is C-order [E, N, K].
        Kr, N, E = (int(x) for x in t.shape)
        assert Kr == K, f"{gname} K {Kr} != {K}"
        nb = K // 32
        packed = np.asarray(t.data).reshape(E, N, nb * 17)
        ok = True
        for e in range(min(n_experts_check, E)):
            wk = f"{prefix}.{e}.{proj}.weight"
            sk = f"{prefix}.{e}.{proj}.scale"
            h = _open_shard(model_dir, weight_map, cache, wk)
            w_u8 = _as_u8(h.get_tensor(wk))
            s_u8 = _as_u8(h.get_tensor(sk))
            a = dequant_native(w_u8, s_u8)
            b = dequant_gguf_blocks(packed[e], K)
            if not np.array_equal(a, b):
                bad = int((a != b).sum())
                print(f"  [{gname}] expert {e}: FAIL {bad}/{a.size} differ (max abs {np.abs(a-b).max()})")
                ok = False
        print(f"[layer] {gname}: {'PASS' if ok else 'FAIL'} (checked {min(n_experts_check,E)} experts, shape E={E} N={N} K={K})")
        if not ok:
            raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=None, help="Converted layer GGUF (layer_test)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Official DeepSeek-V4-Flash checkpoint directory")
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--n-experts-check", type=int, default=8)
    args = ap.parse_args()

    unit_test()
    if args.gguf is not None:
        if args.model_dir is None:
            raise SystemExit("--gguf needs --model-dir (the official checkpoint)")
        layer_test(args.gguf.expanduser().resolve(), args.model_dir.expanduser().resolve(),
                   args.layer_idx, args.n_experts_check)


if __name__ == "__main__":
    main()
