#!/usr/bin/env python3

import argparse
import os
import glob
import numpy as np
import torch
from safetensors.torch import save_file
import gc
import json
import shutil
import sys


def discover_layers(input_path: str):
    """Discover all layer folders in the input directory."""
    layer_folders = []
    for item in os.listdir(input_path):
        if item.startswith("_layer_"):
            try:
                layer_idx = int(item.split("_")[-1])
                layer_folders.append((layer_idx, item))
            except ValueError:
                continue
    layer_folders.sort(key=lambda x: x[0])
    return layer_folders


def discover_numa_folders(layer_path: str):
    """Discover all NUMA folders within a layer folder."""
    numa_folders = []
    for item in os.listdir(layer_path):
        if item.startswith("_numa_"):
            try:
                numa_idx = int(item.split("_")[-1])
                numa_folders.append((numa_idx, item))
            except ValueError:
                continue
    numa_folders.sort(key=lambda x: x[0])
    return numa_folders


def detect_quant_method(layer_path: str):
    """Detect quantization method from file names (INT4 vs INT8)."""
    for root, _, files in os.walk(layer_path):
        for f in files:
            if f.startswith("MOE_INT4_"):
                return "moe_int4", "MOE_INT4"
            elif f.startswith("MOE_INT8_"):
                return "moe_int8", "MOE_INT8"
            elif f.startswith("INT4_"):
                return "int4", "INT4"
            elif f.startswith("INT8_"):
                return "int8", "INT8"
    raise ValueError(f"Could not detect quant method in {layer_path}")


def load_binary_tensor(file_path: str) -> torch.Tensor:
    """Load .kt format binary tensor file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        binary_data = f.read()

    if "scale" in file_path:
        np_array = np.frombuffer(binary_data, dtype=np.float32)
    else:
        np_array = np.frombuffer(binary_data, dtype=np.int8)

    return torch.from_numpy(np_array.copy())


def process_layer(layer_path: str, amx_prefix: str, layer_idx: int) -> dict:
    """Process a single layer folder and return all tensors."""
    tensors = {}
    numa_folders = discover_numa_folders(layer_path)

    if not numa_folders:
        print(f"  Warning: No NUMA folders found in {layer_path}", file=sys.stderr)
        return tensors

    proj_mappings = [
        ("down", "ffn_down_exps"),
        ("gate", "ffn_gate_exps"),
        ("up", "ffn_up_exps"),
    ]

    for numa_idx, numa_folder in numa_folders:
        numa_path = os.path.join(layer_path, numa_folder)

        for proj_name, proj_key in proj_mappings:
            quant_pattern = os.path.join(numa_path, f"{amx_prefix}_{proj_name}_*Byte_quant_.kt")
            scale_pattern = os.path.join(numa_path, f"{amx_prefix}_{proj_name}_*Byte_scale_.kt")

            quant_files = sorted(glob.glob(quant_pattern))
            scale_files = sorted(glob.glob(scale_pattern))

            for quant_file in quant_files:
                filename = os.path.basename(quant_file)
                remainder = filename[len(f"{amx_prefix}_{proj_name}_"):]
                try:
                    expert_idx = int(remainder.split("_")[0])
                except (ValueError, IndexError):
                    print(f"    Warning: Could not parse expert index from {filename}", file=sys.stderr)
                    continue

                weight_key = f"blk.{layer_idx}.{proj_key}.{expert_idx}.numa.{numa_idx}.weight"
                tensors[weight_key] = load_binary_tensor(quant_file)

            for scale_file in scale_files:
                filename = os.path.basename(scale_file)
                remainder = filename[len(f"{amx_prefix}_{proj_name}_"):]
                try:
                    expert_idx = int(remainder.split("_")[0])
                except (ValueError, IndexError):
                    print(f"    Warning: Could not parse expert index from {filename}", file=sys.stderr)
                    continue

                scale_key = f"blk.{layer_idx}.{proj_key}.{expert_idx}.numa.{numa_idx}.scale"
                tensors[scale_key] = load_binary_tensor(scale_file)

    return tensors


def write_shards(accumulated_tensors: dict, output_path: str, shard_counter: dict, keep_remainder: bool = True):
    """Write accumulated tensors to one or more shard files.
    
    Args:
        accumulated_tensors: Dict of tensors to write
        output_path: Output directory
        shard_counter: Dict with 'shard' and 'max_tensors' keys
        keep_remainder: If True, keep leftover tensors in accumulator for next batch
    """
    if not accumulated_tensors:
        return

    max_tensors = shard_counter["max_tensors"]
    current_shard = shard_counter["shard"]
    total_tensors = len(accumulated_tensors)

    if total_tensors <= max_tensors:
        if not keep_remainder:
            output_file = os.path.join(output_path, f"model-{current_shard:05d}.safetensors")
            save_file(accumulated_tensors, output_file)
            print(f"  Saved {total_tensors} tensors to {output_file}")
            shard_counter["shard"] = current_shard + 1
            accumulated_tensors.clear()
        else:
            pass  # Keep accumulating until we hit max_tensors
    else:
        full_shards = total_tensors // max_tensors
        remainder = total_tensors % max_tensors
        
        items = list(accumulated_tensors.items())
        
        # Write full shards
        for i in range(full_shards):
            batch = dict(items[i * max_tensors : (i + 1) * max_tensors])
            output_file = os.path.join(output_path, f"model-{current_shard:05d}.safetensors")
            save_file(batch, output_file)
            print(f"  Saved {len(batch)} tensors to {output_file}")
            current_shard += 1
        
        # Keep remainder for next batch if enabled
        if keep_remainder and remainder > 0:
            remainder_items = dict(items[full_shards * max_tensors:])
            accumulated_tensors.clear()
            accumulated_tensors.update(remainder_items)
            print(f"  Rolled over {remainder} tensors to next batch")
        elif remainder > 0:
            # Write remainder as final shard
            batch = dict(items[full_shards * max_tensors:])
            output_file = os.path.join(output_path, f"model-{current_shard:05d}.safetensors")
            save_file(batch, output_file)
            print(f"  Saved {len(batch)} tensors to {output_file}")
            current_shard += 1
            accumulated_tensors.clear()
        
        shard_counter["shard"] = current_shard


def copy_config_files(original_path: str, output_path: str):
    """Copy config and tokenizer files from original model folder."""
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for config_file in config_files:
        src_path = os.path.join(original_path, config_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_path, config_file)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {config_file}")
        else:
            print(f"Warning: {config_file} not found in {original_path}, skipping", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Merge CPU-optimized weights from nested folder structure to sharded safetensors"
    )
    parser.add_argument(
        "--input-path", "-i", required=True, help="Input directory with nested _layer_* folders"
    )
    parser.add_argument("--output", "-o", required=True, help="Output directory for merged safetensors")
    parser.add_argument(
        "--original-path",
        "-r",
        default=None,
        help="Original model folder with config.json and tokenizer files to copy",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=3000,
        help="Maximum tensors per safetensors shard (default: 3000)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Input path does not exist: {args.input_path}", file=sys.stderr)
        return 1

    os.makedirs(args.output, exist_ok=True)

    print("Discovering layer folders...")
    layer_folders = discover_layers(args.input_path)
    if not layer_folders:
        print(f"Error: No _layer_* folders found in {args.input_path}", file=sys.stderr)
        return 1

    print(f"Found {len(layer_folders)} layer folders")

    print("Detecting quantization method...")
    first_layer_path = os.path.join(args.input_path, layer_folders[0][1])
    quant_method, amx_prefix = detect_quant_method(first_layer_path)
    print(f"Detected quant method: {quant_method} (prefix: {amx_prefix})")

    print(f"\nProcessing layers (max {args.max_tensors} tensors per shard)...")

    accumulated_tensors = {}
    shard_counter = {"shard": 1, "max_tensors": args.max_tensors}

    for layer_idx, layer_folder in layer_folders:
        layer_path = os.path.join(args.input_path, layer_folder)
        print(f"Processing layer {layer_idx} ({layer_folder})...")

        layer_tensors = process_layer(layer_path, amx_prefix, layer_idx)
        print(f"  Loaded {len(layer_tensors)} tensors from this layer")

        accumulated_tensors.update(layer_tensors)

        if len(accumulated_tensors) >= args.max_tensors:
            print(f"  Accumulator has {len(accumulated_tensors)} tensors, flushing to shard(s)...")
            write_shards(accumulated_tensors, args.output, shard_counter, keep_remainder=True)

        gc.collect()

    if accumulated_tensors:
        print(f"Flushing remaining {len(accumulated_tensors)} tensors to final shard(s)...")
        write_shards(accumulated_tensors, args.output, shard_counter, keep_remainder=False)

    if args.original_path:
        print(f"\nCopying config files from {args.original_path}...")
        copy_config_files(args.original_path, args.output)

    total_shards = shard_counter["shard"] - 1
    print(f"\nConversion completed! Created {total_shards} shard(s) in {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())
