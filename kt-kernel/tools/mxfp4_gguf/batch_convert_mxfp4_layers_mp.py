#!/usr/bin/env python3
"""
Convert every MoE layer of a DeepSeek-V4-Flash checkpoint into its own GGUF.

Each layer's experts live in their own safetensors shard, so one layer's
conversion reads a single file and the layers parallelise cleanly across
processes. Output is ``{output_dir}/{prefix}{L}{suffix}.gguf``, by default
``dsv4_layer{L}_mxfp4.gguf`` — the naming the ``--kt-weight-path`` template
expects.

The repack is lossless; see ``convert_mxfp4_layer_to_gguf.py`` for the format
details.

Capacity: about 3.19 GiB per layer, so roughly 138 GiB for a 43-layer model
(Q8_0 would be exactly twice that).

Example (all layers, shapes taken from the checkpoint's config.json)::

  python3 kt-kernel/tools/mxfp4_gguf/batch_convert_mxfp4_layers_mp.py \\
      --input /path/to/DeepSeek-V4-Flash \\
      --output-dir /path/to/gguf-cache \\
      --jobs 8 --skip-existing

Do not point two workers at the same layer: an interrupted write leaves a
truncated file that ``--skip-existing`` will not notice (it only skips files
larger than 1 GiB). Always run ``verify_mxfp4_gguf_set.py`` afterwards.
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

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
_CONVERT_SCRIPT = _HERE / "convert_mxfp4_layer_to_gguf.py"


def _gguf_py() -> Path:
    return Path(os.environ.get("KT_GGUF_PY", _REPO_ROOT / "third_party" / "llama.cpp" / "gguf-py"))


def _run_one_layer(py, model_dir, layer_idx, output_path, num_experts, hidden_size, moe_intermediate_size):
    cmd = [
        py, str(_CONVERT_SCRIPT),
        "--input", model_dir,
        "--layer-idx", str(layer_idx),
        "--output", output_path,
        "--num-experts", str(num_experts),
        "--hidden-size", str(hidden_size),
        "--moe-intermediate-size", str(moe_intermediate_size),
    ]
    env = os.environ.copy()
    # Pure CPU numpy/torch work: skip the torch_npu autoload so the workers do
    # not each try to open an NPU device.
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    tail = (proc.stdout or "")[-3000:]
    if proc.stderr:
        tail += "\n--- stderr ---\n" + proc.stderr[-3000:]
    return layer_idx, proc.returncode, tail


def _verify_sample_paths(paths: list[Path]) -> None:
    sys.path.insert(0, str(_gguf_py()))
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


def main() -> int:
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
    args = ap.parse_args()

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

    print("[batch] done. Now run verify_mxfp4_gguf_set.py before serving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
