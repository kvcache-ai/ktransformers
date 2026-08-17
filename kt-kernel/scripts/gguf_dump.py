#!/usr/bin/env python3
"""
gguf_dump.py — Inspect a GGUF file's tensor types and metadata.

Usage:
    python gguf_dump.py <model.gguf> [--metadata] [--tensors] [--filter ffn]

This script dumps:
  1. GGUF metadata keys (general.architecture, expert counts, etc.)
  2. Tensor names, shapes, GGML types, byte sizes
  3. Type distribution summary

It helps identify which GGML block types are present in a UD-Q4 model
before implementing/converting to AMXINT8.
"""

import argparse
import os
import sys
from collections import Counter

# Add parent for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kt_kernel.utils.loader import GGUFLoader, GGMLQuantizationType


def format_size(n_bytes: int) -> str:
    """Format byte size human-readably."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def main():
    parser = argparse.ArgumentParser(description="Dump GGUF tensor types and metadata")
    parser.add_argument("gguf_path", help="Path to .gguf file or directory")
    parser.add_argument("--metadata", action="store_true", help="Show metadata keys")
    parser.add_argument("--tensors", action="store_true", help="Show all tensor details")
    parser.add_argument("--filter", default=None, help="Filter tensor names by keyword")
    parser.add_argument("--summary", action="store_true", default=True,
                        help="Show type distribution summary (default)")
    args = parser.parse_args()

    if not os.path.exists(args.gguf_path):
        print(f"Error: path not found: {args.gguf_path}")
        sys.exit(1)

    loader = GGUFLoader(args.gguf_path)

    print(f"\n{'='*60}")
    print(f"GGUF File: {args.gguf_path}")
    print(f"{'='*60}")
    print(f"Total tensors: {len(loader.tensor_info)}")
    print(f"Metadata keys: {len(loader.metadata)}")
    print(f"Files: {len(loader.file_data_map)}")

    # Metadata
    if args.metadata:
        print(f"\n{'─'*60}")
        print("Metadata:")
        print(f"{'─'*60}")
        for k in sorted(loader.metadata.keys()):
            v = loader.metadata[k]
            if isinstance(v, (list,)) and len(v) == 1:
                v = v[0]
            print(f"  {k}: {v}")

    # Tensor details
    type_counter = Counter()
    total_bytes = 0
    expert_tensors = []

    for name in sorted(loader.tensor_info.keys()):
        info = loader.tensor_info[name]
        type_name = info["dtype"].name
        shape = info["shape"]
        n_elements = info["n_elements"]

        # Calculate byte size
        from kt_kernel.utils.loader import GGML_QUANT_SIZES
        block_size, type_size = GGML_QUANT_SIZES.get(info["dtype"], (1, 1))
        n_bytes = n_elements * type_size // block_size

        type_counter[type_name] += 1
        total_bytes += n_bytes

        is_expert = "ffn_" in name and "exps" in name
        if is_expert:
            expert_tensors.append((name, type_name, shape, n_bytes))

        if args.tensors:
            if args.filter and args.filter not in name:
                continue
            marker = " [EXPERT]" if is_expert else ""
            print(f"  {name}: type={type_name}, shape={shape}, "
                  f"size={format_size(n_bytes)}{marker}")

    # Summary
    print(f"\n{'─'*60}")
    print("Type Distribution:")
    print(f"{'─'*60}")
    for t, c in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {c:5d} tensors")
    print(f"  {'TOTAL':20s}: {sum(type_counter.values()):5d} tensors, "
          f"{format_size(total_bytes)}")

    # Expert tensor summary
    if expert_tensors:
        print(f"\n{'─'*60}")
        print(f"Expert (CPU-executed) Tensors: {len(expert_tensors)}")
        print(f"{'─'*60}")
        expert_types = Counter(t for _, t, _, _ in expert_tensors)
        for t, c in sorted(expert_types.items()):
            supported = "✓" if t in ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16", "F32") else "✗"
            print(f"  {t:20s}: {c:3d} tensors  [{supported}]")

        print(f"\n  Expert tensor details:")
        for name, type_name, shape, n_bytes in expert_tensors[:20]:
            print(f"    {name}: {type_name}, shape={shape}, {format_size(n_bytes)}")
        if len(expert_tensors) > 20:
            print(f"    ... and {len(expert_tensors) - 20} more")

    # Check for unsupported types
    supported = {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16", "BF16", "F32"}
    unsupported = set(type_counter.keys()) - supported
    if unsupported:
        print(f"\n⚠  Unsupported types (need dequantization impl): {unsupported}")
    else:
        print(f"\n✓  All types are supported for AMXINT8 conversion")


if __name__ == "__main__":
    main()
