#!/usr/bin/env python3
"""
Convert one DeepSeek-V4-Flash native MXFP4 MoE layer (E2M1 nibbles + ue8m0 scale)
to a minimal GGUF with the three stacked expert tensors kt-kernel expects:

  blk.{L}.ffn_gate_exps.weight   (MXFP4)
  blk.{L}.ffn_up_exps.weight     (MXFP4)
  blk.{L}.ffn_down_exps.weight   (MXFP4)

This is a *lossless bit repack*, NOT a re-quantization. The native checkpoint
already stores E2M1 codes; we only:
  - copy the ue8m0 group exponent byte verbatim into block_mxfp4.e
  - rearrange the 4-bit codes from the model's CONSECUTIVE packing to the GGUF
    HALF-BLOCK packing (see below)

Nibble order (settled against the checkpoint's own inference/convert.py):
  Native pack (cast_e2m1fn_to_e4m3fn): byte i -> stack([low, high]).flatten,
    i.e. K-position 2i = low nibble, 2i+1 = high nibble  (CONSECUTIVE)
  Upstream GGUF block_mxfp4: qs[j] low nibble = K-pos j, high nibble = K-pos j+16
    (HALF-BLOCK interleave) — this is what ggml_vec_dot_mxfp4_q8_0 assumes.
  So within each 32-element group (16 bytes) we reorder; the per-group scale is
  unaffected (block boundary == scale group == 32 K-positions == 16 bytes).

Numerics: ggml kvalues_mxfp4[n] * GGML_E8M0_TO_FP32_HALF(e)
        == model FP4_TABLE[n] * 2^(e-127), bit-for-bit. Verified by
        verify_mxfp4_layer.py (element-wise equality vs native dequant).

Layout (matches the pointer arithmetic of kt-kernel's LLAMA_MOE_TP backend):
  gate/up: per-expert (intermediate=N, hidden=K) row-major, hidden(K) inner
  down:    per-expert (hidden=N, intermediate=K) row-major, intermediate(K) inner
  K is the GEMM reduce dim and the MXFP4 block direction (block_size 32 along K).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

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

import gguf  # noqa: E402
from safetensors import safe_open  # noqa: E402

MXFP4 = gguf.GGMLQuantizationType.MXFP4


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")
    return json.loads(index_path.read_text())["weight_map"]


def _detect_experts_prefix(weight_map: dict[str, str], layer_idx: int) -> str:
    """Native V4-Flash keys are stripped of the `model.` prefix: layers.{L}.ffn.experts.{i}.w1..."""
    for layer_prefix in (f"model.layers.{layer_idx}.", f"layers.{layer_idx}."):
        for k in weight_map:
            if k.startswith(layer_prefix) and ".experts." in k and ".shared_experts" not in k \
                    and k.endswith(".w1.weight") and ".experts.0." in k:
                before, _ = k.split(".experts.0.", 1)
                return before + ".experts"
    raise ValueError(f"No native MXFP4 experts found for layer {layer_idx}")


def _open_shard(model_dir: Path, weight_map: dict[str, str], cache: dict[str, object], key: str):
    shard = weight_map[key]
    if shard not in cache:
        cache[shard] = safe_open(model_dir / shard, framework="pt")
    return cache[shard]


def _as_u8(t: torch.Tensor) -> np.ndarray:
    """Native I8 weight or F8_E8M0 scale -> raw bytes as uint8 numpy (no value change)."""
    if t.dtype != torch.uint8:
        t = t.view(torch.uint8)
    return t.contiguous().numpy()


def _repack_consecutive_to_halfblock(w_u8: np.ndarray) -> np.ndarray:
    """[N, K/2] native E2M1 (byte i -> Kpos 2i,2i+1) -> [N, K/2] GGUF (byte j -> Kpos j,j+16).

    Per 32-group (16 bytes): rebuild the 32 nibbles in K order, then split first 16
    into low nibbles and last 16 into high nibbles of the output 16 bytes.
    """
    N, kh = w_u8.shape  # kh = K/2
    assert kh % 16 == 0, f"K/2 ({kh}) must be a multiple of 16"
    nb = kh // 16
    w = w_u8.reshape(N, nb, 16)
    lo = (w & 0x0F).astype(np.uint8)        # Kpos 0,2,...,30 within group
    hi = ((w >> 4) & 0x0F).astype(np.uint8)  # Kpos 1,3,...,31 within group
    nib = np.empty((N, nb, 32), dtype=np.uint8)
    nib[..., 0::2] = lo
    nib[..., 1::2] = hi
    gguf_lo = nib[..., 0:16]    # Kpos 0..15
    gguf_hi = nib[..., 16:32]   # Kpos 16..31
    out = (gguf_lo | (gguf_hi << 4)).astype(np.uint8)  # [N, nb, 16]
    return out.reshape(N, kh)


def _build_proj_tensor(model_dir, weight_map, experts_prefix, proj_name, num_experts):
    """Return packed uint8 ndarray [E, N, nblocks*17] for one projection across all experts."""
    cache: dict[str, object] = {}
    rows = []
    n_dim = None
    nblocks = None
    for e in range(num_experts):
        wk = f"{experts_prefix}.{e}.{proj_name}.weight"
        sk = f"{experts_prefix}.{e}.{proj_name}.scale"
        h = _open_shard(model_dir, weight_map, cache, wk)
        w_u8 = _as_u8(h.get_tensor(wk))      # [N, K/2]
        s_u8 = _as_u8(h.get_tensor(sk))      # [N, K/32]
        N, kh = w_u8.shape
        nb = kh // 16                         # K/32 blocks
        assert s_u8.shape == (N, nb), f"scale {s_u8.shape} != ({N},{nb}) for {wk}"
        qs = _repack_consecutive_to_halfblock(w_u8)        # [N, K/2]
        qs = qs.reshape(N, nb, 16)
        block = np.concatenate([s_u8.reshape(N, nb, 1), qs], axis=-1)  # [N, nb, 17]
        rows.append(block.reshape(N, nb * 17))
        if n_dim is None:
            n_dim, nblocks = N, nb
    out = np.stack(rows, axis=0).astype(np.uint8)  # [E, N, nblocks*17]
    return np.ascontiguousarray(out)


def convert_layer(model_dir: Path, layer_idx: int, output_path: Path, num_experts: int,
                  hidden_size: int, moe_intermediate_size: int) -> None:
    weight_map = _load_weight_map(model_dir)
    experts_prefix = _detect_experts_prefix(weight_map, layer_idx)
    print(f"[convert] layer={layer_idx} experts_prefix={experts_prefix!r} experts={num_experts} -> MXFP4")

    gate = _build_proj_tensor(model_dir, weight_map, experts_prefix, "w1", num_experts)
    print(f"[convert] gate packed {gate.shape} ({gate.nbytes/1e9:.3f} GB)")
    up = _build_proj_tensor(model_dir, weight_map, experts_prefix, "w3", num_experts)
    print(f"[convert] up   packed {up.shape}")
    down = _build_proj_tensor(model_dir, weight_map, experts_prefix, "w2", num_experts)
    print(f"[convert] down packed {down.shape}")

    arch = "deepseek2"
    writer = gguf.GGUFWriter(str(output_path), arch)
    writer.add_quantization_version(2)
    writer.add_name(f"dsv4-flash-layer{layer_idx}-moe-mxfp4")
    writer.add_uint32(gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch), num_experts)
    writer.add_uint32(gguf.Keys.LLM.EXPERT_USED_COUNT.format(arch=arch), 6)
    writer.add_uint32(gguf.Keys.LLM.EMBEDDING_LENGTH.format(arch=arch), hidden_size)
    writer.add_uint32(gguf.Keys.LLM.EXPERT_FEED_FORWARD_LENGTH.format(arch=arch), moe_intermediate_size)

    base = f"blk.{layer_idx}"
    writer.add_tensor(f"{base}.ffn_gate_exps.weight", gate, raw_dtype=MXFP4)
    writer.add_tensor(f"{base}.ffn_up_exps.weight", up, raw_dtype=MXFP4)
    writer.add_tensor(f"{base}.ffn_down_exps.weight", down, raw_dtype=MXFP4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer.write_header_to_file(str(output_path))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"[convert] wrote {output_path} ({output_path.stat().st_size/1e9:.3f} GB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Native MXFP4 model dir (safetensors + index.json)")
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--verify-reader", action="store_true")
    args = ap.parse_args()

    model_dir = args.input.expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"--input must be a directory: {model_dir}")
    convert_layer(model_dir, args.layer_idx, args.output.expanduser().resolve(),
                  args.num_experts, args.hidden_size, args.moe_intermediate_size)

    if args.verify_reader:
        from gguf import GGUFReader
        reader = GGUFReader(str(args.output.expanduser().resolve()))
        for t in reader.tensors:
            print(f"  {t.name} shape={t.shape} type={t.tensor_type}")


if __name__ == "__main__":
    main()
