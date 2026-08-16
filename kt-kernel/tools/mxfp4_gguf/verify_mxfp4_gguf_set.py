#!/usr/bin/env python3
"""
Acceptance gate for a converted MXFP4 GGUF set. Three levels, each stronger
than the last:

  L1 completeness (seconds, no dependencies): every layer present, and every
     file exactly the expected byte count. An interrupted or concurrent
     conversion leaves truncated files behind; a wrong size means that layer
     must be re-converted.
  L2 fingerprint (minutes, no dependencies): sha256 against a manifest.
     Conversion is byte-deterministic — the same checkpoint and the same
     gguf-py reproduce byte-identical output — so a mismatch means the input
     or the environment differed.
  L3 numerics (needs the source checkpoint): dequantize a sample of layers
     from the GGUF and from the checkpoint and compare element-wise. The
     repack is lossless, so anything other than bit-exact equality is a bug.

Usage::

  # L1 only (fast, or when the disk is slow)
  python3 verify_mxfp4_gguf_set.py --dir /path/to/gguf-cache --skip-sha256

  # L1 + L2 against a manifest you generated on a machine you trust
  python3 verify_mxfp4_gguf_set.py --dir /path/to/gguf-cache \\
      --sha256-manifest /path/to/manifest.txt

  # L1 + L2 + L3 (the pre-deployment standard)
  python3 verify_mxfp4_gguf_set.py --dir /path/to/gguf-cache \\
      --deep 3 --model-dir /path/to/DeepSeek-V4-Flash

Generate your own manifest with::

  cd /path/to/gguf-cache && sha256sum dsv4_layer*_mxfp4.gguf | sort -V -k2 > manifest.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _sha256_file(path: Path) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path.name, h.hexdigest()


def main() -> int:
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
    args = ap.parse_args()

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
                    [sys.executable, str(here / "verify_mxfp4_layer.py"),
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


if __name__ == "__main__":
    raise SystemExit(main())
