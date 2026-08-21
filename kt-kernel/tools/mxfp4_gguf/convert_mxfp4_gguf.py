#!/usr/bin/env python3
"""Convert the DeepSeek-V4-Flash MXFP4 MoE weights to the per-layer GGUF set.

    convert_mxfp4_gguf.py batch --input CKPT --output-dir DIR --jobs 8 --skip-existing
    convert_mxfp4_gguf.py layer --input CKPT --layer-idx 16 --output OUT.gguf

`batch` fans the layers out across processes, re-executing this file in `layer` mode
for each. About 3.19 GiB per layer.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_one_layer(py, model_dir, layer_idx, output_path, num_experts, hidden_size, moe_intermediate_size):
    cmd = [
        py, str(Path(__file__).resolve()), "layer",
        "--input", model_dir,
        "--layer-idx", str(layer_idx),
        "--output", output_path,
        "--num-experts", str(num_experts),
        "--hidden-size", str(hidden_size),
        "--moe-intermediate-size", str(moe_intermediate_size),
    ]
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (proc.stdout or "")[-3000:]
    if proc.stderr:
        tail += "\n--- stderr ---\n" + proc.stderr[-3000:]
    return layer_idx, proc.returncode, tail

def _verify_sample_paths(paths: list[Path]) -> None:
    sys.path.insert(0, str(_GGUF_PY))
    from gguf import GGUFReader

    for p in paths:
        if not p.is_file():
            print(f"[verify-sample] SKIP missing: {p}")
            continue
        reader = GGUFReader(str(p))
        print(f"[verify-sample] {p.name} ({p.stat().st_size / 1e9:.3f} GB) tensors={len(reader.tensors)}")
        for t in reader.tensors:
            tt = t.tensor_type
            print(f"    {t.name} type={getattr(tt, 'name', tt)} shape={list(t.shape)}")


def _layer_main(argv) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Native MXFP4 model dir (safetensors + index.json)")
    ap.add_argument("--layer-idx", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--hidden-size", type=int, default=4096)
    ap.add_argument("--moe-intermediate-size", type=int, default=2048)
    ap.add_argument("--verify-reader", action="store_true")
    args = ap.parse_args(argv)

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


def _batch_main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Official DeepSeek-V4-Flash checkpoint (safetensors + index.json)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layer-start", type=int, default=0)
    ap.add_argument("--layer-end", type=int, default=None,
                    help="Inclusive. Default: num_hidden_layers - 1 from config.json")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--num-experts", type=int, default=None, help="Default: n_routed_experts from config.json")
    ap.add_argument("--hidden-size", type=int, default=None, help="Default: hidden_size from config.json")
    ap.add_argument("--moe-intermediate-size", type=int, default=None,
                    help="Default: moe_intermediate_size from config.json")
    ap.add_argument("--name-prefix", type=str, default="dsv4_layer")
    ap.add_argument("--name-suffix", type=str, default="_mxfp4")
    ap.add_argument("--skip-existing", action="store_true", help="Skip outputs already larger than 1 GiB")
    ap.add_argument("--verify-sample", type=int, default=3,
                    help="Re-open this many random outputs and print their tensor headers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    model_dir = args.input.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: --input is not a directory: {model_dir}", file=sys.stderr)
        return 2

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        print(f"ERROR: no config.json in {model_dir}", file=sys.stderr)
        return 2
    cfg = json.loads(cfg_path.read_text())
    num_experts = args.num_experts or int(cfg["n_routed_experts"])
    hidden_size = args.hidden_size or int(cfg["hidden_size"])
    moe_inter = args.moe_intermediate_size or int(cfg["moe_intermediate_size"])
    layer_end = args.layer_end if args.layer_end is not None else int(cfg["num_hidden_layers"]) - 1
    print(f"[batch] shapes from config.json: experts={num_experts} hidden={hidden_size} "
          f"moe_intermediate={moe_inter} layers={args.layer_start}..{layer_end}")

    out_dir.mkdir(parents=True, exist_ok=True)
    min_skip = 1 << 30
    py = str(args.python.expanduser())

    layers = list(range(args.layer_start, layer_end + 1))
    tasks = []
    for lid in layers:
        outp = out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf"
        if args.skip_existing and outp.is_file() and outp.stat().st_size > min_skip:
            print(f"[batch] skip existing {outp.name}")
            continue
        tasks.append((py, str(model_dir), lid, str(outp), num_experts, hidden_size, moe_inter))

    if not tasks:
        print("[batch] nothing to convert (all skipped)")
    else:
        print(f"[batch] model={model_dir} pending={len(tasks)} jobs={args.jobs}")
        failed = []
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futures = {ex.submit(_run_one_layer, *t): t[2] for t in tasks}
            for fut in as_completed(futures):
                lid = futures[fut]
                try:
                    layer_idx, rc, tail = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append((lid, repr(exc)))
                    print(f"[batch] layer {lid} worker exception: {exc!r}")
                    continue
                if rc != 0:
                    failed.append((layer_idx, f"exit {rc}"))
                    print(f"[batch] layer {layer_idx} FAILED rc={rc}\n{tail[-1500:]}")
                else:
                    print(f"[batch] layer {layer_idx} OK")
        if failed:
            print(f"[batch] {len(failed)} layers failed: {failed[:10]}", file=sys.stderr)
            return 1

    if args.verify_sample > 0:
        rnd = random.Random(args.seed)
        k = min(args.verify_sample, len(layers))
        sample = sorted(rnd.sample(layers, k)) if k > 0 else []
        paths = [out_dir / f"{args.name_prefix}{lid}{args.name_suffix}.gguf" for lid in sample]
        print(f"[batch] verify-sample k={k} layers={sample}")
        _verify_sample_paths(paths)

    print("[batch] done. Now run verify_mxfp4_gguf.py set before serving.")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("layer", "batch"):
        print(__doc__)
        return 0 if len(sys.argv) < 2 else 2
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "layer":
        _layer_main(rest)
        return 0
    return _batch_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
