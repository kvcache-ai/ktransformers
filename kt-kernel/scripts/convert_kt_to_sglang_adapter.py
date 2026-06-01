#!/usr/bin/env python3
"""Convert KT fused expert LoRA checkpoints into an SGLang adapter directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from safetensors.torch import load_file, save_file


FUSED_EXPERT_LORA_FILE = "fused_expert_lora.safetensors"
ADAPTER_MODEL_FILE = "adapter_model.safetensors"
ADAPTER_CONFIG_FILE = "adapter_config.json"

KT_NAME_MAP = {
    "gate_lora_a": ("gate_proj", "lora_A", 1),
    "gate_lora_b": ("gate_proj", "lora_B", 2),
    "up_lora_a": ("up_proj", "lora_A", 1),
    "up_lora_b": ("up_proj", "lora_B", 2),
    "down_lora_a": ("down_proj", "lora_A", 1),
    "down_lora_b": ("down_proj", "lora_B", 2),
}

TARGET_MODULE_ORDER = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "embed_tokens",
    "lm_head",
]

KT_FUSED_KEY_RE = re.compile(r"^layers\.(\d+)\.experts\.([^.]+)$")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Mapping) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _clean_adapter_key(key: str) -> str:
    """Match the existing SGLang converter's PEFT key cleanup."""
    key = key.replace("base_model.model.", "")
    key = key.replace(".orig_module", "")
    return key


def _ordered_target_modules(modules: Iterable[str]) -> list[str]:
    seen = set(modules)
    ordered = [name for name in TARGET_MODULE_ORDER if name in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def _infer_target_module_from_key(key: str) -> str | None:
    if "lora_embedding_A" in key or "lora_embedding_B" in key:
        if "embed_tokens" in key:
            return "embed_tokens"
        if "lm_head" in key or "unembed_tokens" in key:
            return "lm_head"

    marker = ".lora_"
    if marker not in key:
        return None
    prefix = key.split(marker, 1)[0]
    if "." not in prefix:
        return prefix
    return prefix.rsplit(".", 1)[-1]


def _merge_tensor(tensors: Dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
    if key in tensors:
        raise ValueError(f"Duplicate output tensor key: {key}")
    tensors[key] = value.detach().cpu()


def _load_existing_adapter(input_dir: Path) -> tuple[dict[str, torch.Tensor], set[str]]:
    adapter_path = input_dir / ADAPTER_MODEL_FILE
    if not adapter_path.exists():
        return {}, set()

    tensors: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    for key, value in load_file(str(adapter_path)).items():
        cleaned_key = _clean_adapter_key(key)
        _merge_tensor(tensors, cleaned_key, value)
        target_module = _infer_target_module_from_key(cleaned_key)
        if target_module is not None:
            target_modules.add(target_module)
    return tensors, target_modules


def _convert_fused_expert_lora(
    fused_path: Path,
) -> tuple[dict[str, torch.Tensor], int, set[str]]:
    if not fused_path.exists():
        raise FileNotFoundError(f"Missing {FUSED_EXPERT_LORA_FILE}: {fused_path}")

    output: dict[str, torch.Tensor] = {}
    ranks: set[int] = set()
    expert_counts: set[int] = set()
    target_modules: set[str] = set()

    for key, tensor in sorted(load_file(str(fused_path)).items()):
        match = KT_FUSED_KEY_RE.match(key)
        if match is None:
            raise ValueError(f"Unexpected key in {FUSED_EXPERT_LORA_FILE}: {key}")

        layer_idx, kt_name = match.groups()
        if kt_name not in KT_NAME_MAP:
            raise ValueError(f"Unsupported KT fused expert LoRA tensor: {key}")
        if tensor.dim() != 3:
            raise ValueError(f"{key} must be 3D [E, ...], got shape {tuple(tensor.shape)}")

        proj_name, lora_name, rank_dim = KT_NAME_MAP[kt_name]
        expert_count = int(tensor.shape[0])
        rank = int(tensor.shape[rank_dim])
        expert_counts.add(expert_count)
        ranks.add(rank)
        target_modules.add(proj_name)

        for expert_idx in range(expert_count):
            output_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                f"{proj_name}.{lora_name}.weight"
            )
            _merge_tensor(output, output_key, tensor[expert_idx].contiguous())

    if not output:
        raise ValueError(f"No tensors found in {fused_path}")
    if len(expert_counts) != 1:
        raise ValueError(f"Inconsistent expert counts in {FUSED_EXPERT_LORA_FILE}: {sorted(expert_counts)}")
    if len(ranks) != 1:
        raise ValueError(f"Inconsistent LoRA ranks in {FUSED_EXPERT_LORA_FILE}: {sorted(ranks)}")

    return output, next(iter(ranks)), target_modules


def _build_adapter_config(
    input_dir: Path,
    rank: int,
    target_modules: set[str],
    base_model_name_or_path: str,
    lora_alpha: float | None,
    *,
    include_input_target_modules: bool = True,
) -> dict:
    config_path = input_dir / ADAPTER_CONFIG_FILE
    config = _load_json(config_path) if config_path.exists() else {}

    if "lora_alpha" in config:
        final_alpha = config["lora_alpha"]
    elif lora_alpha is not None:
        final_alpha = lora_alpha
    else:
        raise ValueError(
            f"No {ADAPTER_CONFIG_FILE} with lora_alpha found in {input_dir}; "
            "pass --lora-alpha to preserve runtime scaling."
        )

    existing_targets = config.get("target_modules", [])
    if include_input_target_modules and isinstance(existing_targets, list):
        target_modules.update(str(name).split(".")[-1] for name in existing_targets)

    config["peft_type"] = config.get("peft_type", "LORA")
    config["r"] = rank
    config["lora_alpha"] = final_alpha
    config["target_modules"] = _ordered_target_modules(target_modules)
    config["bias"] = config.get("bias", "none")
    config["task_type"] = config.get("task_type", "CAUSAL_LM")
    config["base_model_name_or_path"] = base_model_name_or_path

    return config


def _prepare_output_dir(output_path: Path, input_path: Path, overwrite: bool) -> None:
    _validate_output_dir(output_path, input_path, overwrite)
    if output_path.exists() and any(output_path.iterdir()):
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def _validate_output_dir(output_path: Path, input_path: Path, overwrite: bool) -> None:
    if output_path == input_path:
        raise ValueError("Output directory must be different from input directory.")
    if output_path.exists() and not output_path.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_path}")
    if output_path.exists() and any(output_path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_path}")


def _write_adapter(
    output_path: Path,
    input_path: Path,
    tensors: dict[str, torch.Tensor],
    config: Mapping,
    *,
    overwrite: bool,
) -> None:
    _prepare_output_dir(output_path, input_path, overwrite)
    save_file(tensors, str(output_path / ADAPTER_MODEL_FILE), metadata={"format": "pt"})
    _write_json(output_path / ADAPTER_CONFIG_FILE, config)


def convert_kt_to_sglang_adapter(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    base_model_name_or_path: str,
    lora_alpha: float | None = None,
    overwrite: bool = False,
    expert_output_dir: str | os.PathLike | None = None,
    nonexpert_output_dir: str | os.PathLike | None = None,
) -> dict:
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    expert_output_path = (
        Path(expert_output_dir).expanduser().resolve()
        if expert_output_dir is not None
        else None
    )
    nonexpert_output_path = (
        Path(nonexpert_output_dir).expanduser().resolve()
        if nonexpert_output_dir is not None
        else None
    )

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_paths = [output_path]
    output_paths.extend(path for path in (expert_output_path, nonexpert_output_path) if path is not None)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Merged, expert, and non-expert output directories must be distinct.")
    for path in output_paths:
        _validate_output_dir(path, input_path, overwrite)

    existing_tensors, existing_targets = _load_existing_adapter(input_path)
    fused_tensors, rank, fused_targets = _convert_fused_expert_lora(input_path / FUSED_EXPERT_LORA_FILE)
    if nonexpert_output_path is not None and not existing_tensors:
        raise ValueError(
            f"Cannot write non-expert adapter: no {ADAPTER_MODEL_FILE} found in {input_path}."
        )

    tensors: dict[str, torch.Tensor] = {}
    for key, value in existing_tensors.items():
        _merge_tensor(tensors, key, value)
    for key, value in fused_tensors.items():
        _merge_tensor(tensors, key, value)

    target_modules = set(existing_targets)
    target_modules.update(fused_targets)
    config = _build_adapter_config(
        input_path,
        rank,
        target_modules,
        base_model_name_or_path,
        lora_alpha,
    )

    _write_adapter(output_path, input_path, tensors, config, overwrite=overwrite)

    split_outputs: dict[str, dict] = {}
    if expert_output_path is not None:
        expert_config = _build_adapter_config(
            input_path,
            rank,
            set(fused_targets),
            base_model_name_or_path,
            lora_alpha,
            include_input_target_modules=False,
        )
        _write_adapter(
            expert_output_path,
            input_path,
            fused_tensors,
            expert_config,
            overwrite=overwrite,
        )
        split_outputs["expert"] = {
            "output_dir": str(expert_output_path),
            "tensor_count": len(fused_tensors),
            "target_modules": expert_config["target_modules"],
        }

    if nonexpert_output_path is not None:
        nonexpert_config = _build_adapter_config(
            input_path,
            rank,
            set(existing_targets),
            base_model_name_or_path,
            lora_alpha,
            include_input_target_modules=False,
        )
        _write_adapter(
            nonexpert_output_path,
            input_path,
            existing_tensors,
            nonexpert_config,
            overwrite=overwrite,
        )
        split_outputs["nonexpert"] = {
            "output_dir": str(nonexpert_output_path),
            "tensor_count": len(existing_tensors),
            "target_modules": nonexpert_config["target_modules"],
        }

    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "tensor_count": len(tensors),
        "rank": rank,
        "target_modules": config["target_modules"],
        "lora_alpha": config["lora_alpha"],
        "split_outputs": split_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KT fused expert LoRA weights to an SGLang adapter directory."
    )
    parser.add_argument("input_dir", help="Directory containing fused_expert_lora.safetensors.")
    parser.add_argument("output_dir", help="Destination adapter directory.")
    parser.add_argument(
        "--base-model-name-or-path",
        required=True,
        help="Base model path/name to write into adapter_config.json.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha to use when input adapter_config.json is absent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate output_dir if it already contains files.",
    )
    parser.add_argument(
        "--expert-output-dir",
        default=None,
        help="Optional destination for a split expert-only adapter directory.",
    )
    parser.add_argument(
        "--nonexpert-output-dir",
        default=None,
        help="Optional destination for a split non-expert-only adapter directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = convert_kt_to_sglang_adapter(
        args.input_dir,
        args.output_dir,
        base_model_name_or_path=args.base_model_name_or_path,
        lora_alpha=args.lora_alpha,
        overwrite=args.overwrite,
        expert_output_dir=args.expert_output_dir,
        nonexpert_output_dir=args.nonexpert_output_dir,
    )
    print(
        "Converted KT fused expert LoRA adapter: "
        f"{summary['tensor_count']} tensors, rank={summary['rank']}, "
        f"target_modules={summary['target_modules']}"
    )
    for name, split_summary in summary["split_outputs"].items():
        print(
            f"Wrote {name} adapter: {split_summary['tensor_count']} tensors, "
            f"target_modules={split_summary['target_modules']}, "
            f"output_dir={split_summary['output_dir']}"
        )


if __name__ == "__main__":
    main()
