import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import save_file, safe_open

from compressed_tensors.compressors import unpack_from_int32


def _load_config(model_dir: str, config_path: Optional[str]) -> Tuple[int, int, int]:
    cfg_path = config_path or os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)
    hidden_size = int(cfg.get("hidden_size"))
    inter_size = int(cfg.get("moe_intermediate_size"))
    group_size = int(
        cfg.get("quantization_config", {})
        .get("config_groups", {})
        .get("group_0", {})
        .get("weights", {})
        .get("group_size", 32)
    )
    return hidden_size, inter_size, group_size


def _dequantize_tensor(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if isinstance(weight_shape, torch.Tensor):
        shape = tuple(int(v) for v in weight_shape.view(-1).tolist())
    else:
        shape = tuple(weight_shape)
    weight = unpack_from_int32(weight_packed, 4, shape)
    if group_size > 0:
        scale = weight_scale.to(torch.float32)
        if scale.dim() == 1:
            scale = scale.unsqueeze(1)
        scales = torch.repeat_interleave(scale, repeats=group_size, dim=1)
    else:
        scales = weight_scale.to(torch.float32)
    if scales.shape != weight.shape:
        if scales.numel() == weight.numel():
            scales = scales.reshape_as(weight)
        else:
            raise ValueError(
                f"Scale shape {scales.shape} incompatible with weight shape {weight.shape}"
            )
    bf16 = (weight.to(torch.float32) * scales).to(torch.bfloat16)
    return bf16.contiguous()


def _is_quantized_weight_key(key: str) -> bool:
    if ".mlp.experts." not in key or ".shared_experts." in key:
        return False
    suffixes = ("weight_packed", "weight_scale", "weight_shape")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for suffix in suffixes:
            if key.endswith(f".{proj}.{suffix}"):
                return True
    return False


def convert_file(
    input_path: str,
    output_path: str,
    group_size: int,
    skip_existing: bool = True,
):
    if skip_existing and os.path.exists(output_path):
        print(f"[skip] {output_path} already exists.")
        return

    tensors: Dict[str, torch.Tensor] = {}
    expert_buffers: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))

    with safe_open(input_path, framework="pt") as reader:
        keys = list(reader.keys())
        for key in keys:
            tensor = reader.get_tensor(key).detach().cpu()

            if not _is_quantized_weight_key(key):
                tensors[key] = tensor
                continue

            parts = key.split(".")
            try:
                expert_idx = parts.index("experts")
            except ValueError:
                tensors[key] = tensor
                continue

            prefix = ".".join(parts[: expert_idx + 2])
            project = parts[-2]
            suffix = parts[-1]
            expert_buffers[prefix][project][suffix] = tensor

    stats = {
        "converted": 0,
        "skipped": 0,
    }

    for prefix, components in expert_buffers.items():
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj_data = components.get(proj_name, {})
            required = {"weight_packed", "weight_scale", "weight_shape"}
            if not required.issubset(proj_data.keys()):
                print(f"[warn] Missing components for {prefix}.{proj_name}, keeping quantized tensors.")
                for suffix, value in proj_data.items():
                    tensors[f"{prefix}.{proj_name}.{suffix}"] = value
                stats["skipped"] += 1
                continue

            bf16_weight = _dequantize_tensor(
                proj_data["weight_packed"].to(torch.int32),
                proj_data["weight_scale"].to(torch.float32),
                proj_data["weight_shape"],
                group_size,
            )
            tensors[f"{prefix}.{proj_name}.weight"] = bf16_weight.to(torch.bfloat16)
            stats["converted"] += 1
            print(f"    converted {prefix}.{proj_name}.weight -> bf16")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(tensors, output_path)
    print(
        f"[done] wrote {output_path} (converted={stats['converted']}, skipped={stats['skipped']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MoE experts to BF16 weights.")
    parser.add_argument("--model-dir", required=True, help="Directory containing safetensors checkpoints.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for converted checkpoints (default: <model-dir>_bf16).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Specific safetensor filenames to convert (relative to model-dir). Convert all if omitted.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to config.json for extracting group_size (default: model-dir/config.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite output files even if they already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir or f"{model_dir}_bf16")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    _, _, group_size = _load_config(model_dir, args.config_path)

    if args.files:
        targets = [os.path.join(model_dir, fname) for fname in args.files]
    else:
        targets = [
            os.path.join(model_dir, name)
            for name in sorted(os.listdir(model_dir))
            if name.endswith(".safetensors")
        ]

    if not targets:
        print("No safetensors checkpoints found.")
        return

    total = len(targets)

    for idx, path in enumerate(targets, start=1):
        if not os.path.isfile(path):
            print(f"[skip] {path} is not a file.")
            continue
        rel = os.path.relpath(path, model_dir)
        output_path = os.path.join(output_dir, rel)
        print(f"[{idx}/{total}] converting {rel}")
        convert_file(path, output_path, group_size, skip_existing=not args.overwrite)


if __name__ == "__main__":
    main()
