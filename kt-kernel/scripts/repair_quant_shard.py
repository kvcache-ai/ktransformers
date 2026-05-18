#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from safetensors import safe_open
from safetensors.torch import save_file

from convert_cpu_weights import OnlineQuantConverter, load_model_config


def ordered_non_expert_items(converter: OnlineQuantConverter) -> List[Tuple[str, object]]:
    items: List[Tuple[str, object]] = []
    for key in converter.tensor_file_map.keys():
        if ".mlp.experts." in key:
            continue
        if key.startswith("model."):
            new_key = key.replace("model.layers.", "blk.").replace("model.", "")
            items.append((new_key, converter._load_tensor(key)))
        else:
            items.append((key, converter._load_tensor(key)))
    return items


def collect_shard_items(
    converter: OnlineQuantConverter,
    shard_index: int,
    max_tensors_per_file: int,
) -> List[Tuple[str, object]]:
    start = (shard_index - 1) * max_tensors_per_file
    end = shard_index * max_tensors_per_file
    cursor = 0
    picked: List[Tuple[str, object]] = []

    expert_layers = converter._find_expert_layers()

    for layer_idx, expert_ids in sorted(expert_layers.items()):
        layer_tensors = converter._convert_layer_experts(layer_idx, expert_ids)
        layer_items = list(layer_tensors.items())
        layer_start = cursor
        layer_end = cursor + len(layer_items)
        if layer_end > start and layer_start < end:
            rel_start = max(start, layer_start) - layer_start
            rel_end = min(end, layer_end) - layer_start
            picked.extend(layer_items[rel_start:rel_end])
        cursor = layer_end
        if cursor >= end:
            return picked

    non_expert_items = ordered_non_expert_items(converter)
    non_start = cursor
    non_end = cursor + len(non_expert_items)
    if non_end > start and non_start < end:
        rel_start = max(start, non_start) - non_start
        rel_end = min(end, non_end) - non_start
        picked.extend(non_expert_items[rel_start:rel_end])
    return picked


def validate_safetensor(file_path: Path) -> int:
    with safe_open(str(file_path), framework="pt") as f:
        return len(list(f.keys()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair a single quantized safetensors shard.")
    parser.add_argument("--input-path", required=True, help="Original model directory (bf16/fp16/fp8).")
    parser.add_argument("--input-type", required=True, choices=["fp8", "fp16", "bf16"])
    parser.add_argument("--output-path", required=True, help="Quantized model directory.")
    parser.add_argument("--quant-method", default="int4", choices=["int4", "int8", "moe_int4", "moe_int8"])
    parser.add_argument("--shard-index", type=int, required=True, help="1-based shard index to rebuild.")
    parser.add_argument("--cpuinfer-threads", type=int, default=60)
    parser.add_argument("--threadpool-count", type=int, default=2)
    parser.add_argument("--fp8-expert-batch", type=int, default=16)
    parser.add_argument("--max-tensors-per-file", type=int, default=3000)
    parser.add_argument(
        "--backup-suffix",
        default=".broken",
        help="Suffix used when backing up the old shard before replacement.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    target = output_dir / f"model-{args.shard_index:05d}.safetensors"
    backup = Path(str(target) + args.backup_suffix)
    temp = Path(str(target) + ".tmp")

    print(f"[repair_quant_shard] rebuilding shard {args.shard_index} -> {target}")
    model_config = load_model_config(args.input_path, args.input_type)
    converter = OnlineQuantConverter(
        args.input_path,
        args.output_path,
        model_config,
        cpuinfer_threads=args.cpuinfer_threads,
        threadpool_count=args.threadpool_count,
        input_type=args.input_type,
        quant_method=args.quant_method,
        merge_to_safetensor=True,
        fp8_expert_batch=args.fp8_expert_batch,
    )
    try:
        shard_items = collect_shard_items(converter, args.shard_index, args.max_tensors_per_file)
        if not shard_items:
            raise RuntimeError(f"No tensors selected for shard {args.shard_index}")
        print(f"[repair_quant_shard] selected {len(shard_items)} tensors")
        save_file(dict(shard_items), str(temp))
        tensor_count = validate_safetensor(temp)
        print(f"[repair_quant_shard] validated temp shard with {tensor_count} tensors")
        if target.exists():
            if backup.exists():
                backup.unlink()
            target.rename(backup)
            print(f"[repair_quant_shard] backed up old shard to {backup}")
        temp.rename(target)
        print(f"[repair_quant_shard] replaced {target}")
    finally:
        converter.close()
        if temp.exists():
            temp.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
