#!/usr/bin/env python3
"""Check the converted MXFP4 GGUF set.

    verify_mxfp4_gguf.py set   --dir DIR [--deep 3 --model-dir CKPT]
    verify_mxfp4_gguf.py layer [--gguf OUT.gguf --model-dir CKPT --layer-idx L]

`set` checks completeness and sizes, optionally sha256 against a manifest, and with
`--deep N` re-executes this file in `layer` mode on N sampled layers.
`layer` always runs the pure-logic unit check first.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    from convert_mxfp4_gguf import _repack_consecutive_to_halfblock
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
    from convert_mxfp4_gguf import _load_weight_map, _detect_experts_prefix, _open_shard, _as_u8

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


def _sha256_file(path: Path) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path.name, h.hexdigest()

def _set_main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="Directory holding dsv4_layer{L}_mxfp4.gguf")
    ap.add_argument("--name-tpl", type=str, default="dsv4_layer{L}_mxfp4.gguf")
    ap.add_argument("--expect-layers", type=int, default=None,
                    help="Layer count. Default: num_hidden_layers from --model-dir, else the file count found")
    ap.add_argument("--expected-size", type=int, default=None,
                    help="Exact bytes per file. Default: the most common size in the directory")
    ap.add_argument("--sha256-manifest", type=Path, default=None,
                    help="`sha256sum`-format manifest to compare against")
    ap.add_argument("--skip-sha256", action="store_true")
    ap.add_argument("--jobs", type=int, default=8, help="sha256 parallelism")
    ap.add_argument("--deep", type=int, default=0, help="L3: how many layers to check element-wise (0 = off)")
    ap.add_argument("--model-dir", type=Path, default=None, help="L3 needs the source checkpoint")
    args = ap.parse_args(argv)

    d = args.dir.expanduser().resolve()
    if not d.is_dir():
        print(f"FAIL: {d} is not a directory")
        return 1
    fail = False

    # ---- how many layers do we expect? ----
    n_layers = args.expect_layers
    if n_layers is None and args.model_dir is not None:
        cfg = args.model_dir.expanduser().resolve() / "config.json"
        if cfg.is_file():
            n_layers = int(json.loads(cfg.read_text())["num_hidden_layers"])
    present = sorted(d.glob(args.name_tpl.replace("{L}", "*")))
    if n_layers is None:
        n_layers = len(present)
        print(f"[L1] no layer count given; assuming the {n_layers} files present are the full set")
    layers = list(range(n_layers))

    # ---- what size should each file be? ----
    expected_size = args.expected_size
    if expected_size is None:
        sizes = Counter(p.stat().st_size for p in present)
        if not sizes:
            print(f"[L1] FAIL: no files matching {args.name_tpl.replace('{L}', '*')} in {d}")
            return 1
        expected_size = sizes.most_common(1)[0][0]
        print(f"[L1] no --expected-size given; using the modal size {expected_size} "
              f"({sizes.most_common(1)[0][1]}/{len(present)} files agree)")

    # ---- L1: presence + exact size ----
    print(f"[L1] checking {n_layers} files in {d} (expected size {expected_size}) ...")
    missing, badsize = [], []
    for lid in layers:
        p = d / args.name_tpl.format(L=lid)
        if not p.is_file():
            missing.append(lid)
        elif p.stat().st_size != expected_size:
            badsize.append((lid, p.stat().st_size))
    if missing:
        print(f"[L1] FAIL missing layers: {missing}")
        fail = True
    if badsize:
        print(f"[L1] FAIL wrong-size layers (truncated conversion — re-convert these): {badsize}")
        fail = True
    if not missing and not badsize:
        print(f"[L1] PASS - {n_layers}/{n_layers} present, all sizes exact")

    # ---- L2: sha256 vs manifest ----
    if not args.skip_sha256 and not fail:
        if args.sha256_manifest is None:
            print("[L2] SKIP - no manifest given (pass --sha256-manifest or --skip-sha256)")
        else:
            ref = {}
            for line in args.sha256_manifest.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    ref[Path(parts[-1]).name] = parts[0]
            total_gib = n_layers * expected_size / (1 << 30)
            print(f"[L2] hashing {n_layers} files with {args.jobs} workers "
                  f"(~{total_gib:.0f} GiB, takes a few minutes) ...")
            mismatch = []
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                futs = {ex.submit(_sha256_file, d / args.name_tpl.format(L=lid)): lid for lid in layers}
                for fu in as_completed(futs):
                    name, hx = fu.result()
                    want = ref.get(name)
                    if want is None:
                        mismatch.append((name, "NOT-IN-MANIFEST"))
                    elif want != hx:
                        mismatch.append((name, f"got {hx[:16]}.. want {want[:16]}.."))
            if mismatch:
                print(f"[L2] FAIL {len(mismatch)} mismatches: {mismatch[:5]}")
                fail = True
            else:
                print(f"[L2] PASS - all {n_layers} sha256 match the manifest")

    # ---- L3: element-wise vs the source checkpoint ----
    if args.deep > 0 and not fail:
        if args.model_dir is None:
            print("[L3] FAIL - --deep needs --model-dir")
            fail = True
        else:
            here = Path(__file__).resolve().parent
            step = max(1, len(layers) // args.deep)
            sample = layers[::step][: args.deep]
            print(f"[L3] element-wise check on layers {sample} (lossless repack => bit-exact required)")
            for lid in sample:
                r = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "layer",
                     "--gguf", str(d / args.name_tpl.format(L=lid)),
                     "--model-dir", str(args.model_dir), "--layer-idx", str(lid),
                     "--n-experts-check", "4"],
                    capture_output=True, text=True,
                )
                ok = r.returncode == 0 and "FAIL" not in (r.stdout + r.stderr)
                print(f"[L3] layer {lid}: {'PASS' if ok else 'FAIL'}")
                if not ok:
                    print((r.stdout + r.stderr)[-800:])
                    fail = True

    print("\nRESULT:", "FAIL - see the details above, fix and re-run"
          if fail else "PASS - the weight set is ready to serve")
    return 1 if fail else 0


def _layer_main(argv) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, default=None, help="Converted layer GGUF (layer_test)")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="Official DeepSeek-V4-Flash checkpoint directory")
    ap.add_argument("--layer-idx", type=int, default=16)
    ap.add_argument("--n-experts-check", type=int, default=8)
    args = ap.parse_args(argv)

    unit_test()
    if args.gguf is not None:
        if args.model_dir is None:
            raise SystemExit("--gguf needs --model-dir (the official checkpoint)")
        layer_test(args.gguf.expanduser().resolve(), args.model_dir.expanduser().resolve(),
                   args.layer_idx, args.n_experts_check)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("set", "layer"):
        print(__doc__)
        return 0 if len(sys.argv) < 2 else 2
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "layer":
        _layer_main(rest)
        return 0
    return _set_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
