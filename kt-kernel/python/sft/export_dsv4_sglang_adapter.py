#!/usr/bin/env python3
"""Export a single V4 adapter: merged BF16 non-experts + native MXFP4 expert LoRA.

This is static deployment, not adapter hot swapping. Original checkpoint files
are never modified. The model copy contains no routed-expert base weights.
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from kt_kernel.sft.artifacts import (
    FUSED_EXPERT_LORA_NAME,
    KT_ADAPTER_MANIFEST_NAME,
    KTArtifactError,
    _read_json,
    _sha256_file,
    _write_json_atomic,
)
from kt_kernel.sft.deepseek_v4 import (
    canonical_non_expert_key,
    inspect_native_checkpoint,
    resolve_native_load_plan,
)

from .convert_kt_to_sglang_adapter import (
    _build_adapter_config,
    _convert_fused_expert_lora,
)


_STANDARD = re.compile(
    r"^(?:base_model\.model\.)?(model\.layers\.\d+\..+)\.lora_([AB])(?:\.default)?\.weight$"
)
_TARGET = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_a_proj|q_b_proj|kv_proj|o_b_proj|compressor\.(?:kv_proj|gate_proj))"
    r"|mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj))$"
)


def _validate_fused_experts(path, config, rank):
    expected = {}
    experts = config["n_routed_experts"]
    hidden, inter = config["hidden_size"], config["moe_intermediate_size"]
    for layer in range(config["num_hidden_layers"]):
        for projection, (n, k) in {
            "gate": (inter, hidden),
            "up": (inter, hidden),
            "down": (hidden, inter),
        }.items():
            prefix = f"layers.{layer}.experts.{projection}_lora_"
            expected[prefix + "a"] = [experts, rank, k]
            expected[prefix + "b"] = [experts, n, rank]
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise KTArtifactError(
                "incomplete or unexpected fused expert LoRA inventory"
            )
        for name, shape in expected.items():
            tensor = handle.get_slice(name)
            if tensor.get_shape() != shape or tensor.get_dtype() not in {
                "BF16",
                "F16",
                "F32",
            }:
                raise KTArtifactError(f"invalid fused expert LoRA shape/dtype: {name}")
            if not torch.isfinite(handle.get_tensor(name)).all():
                raise KTArtifactError(f"non-finite fused expert LoRA: {name}")


def export(
    source_path,
    cache_path,
    adapter_path,
    output_path,
    component="all",
    match_expert_kernel=False,
):
    output, adapter = Path(output_path).absolute(), Path(adapter_path).absolute()
    if component not in {"all", "base", "experts", "nonexperts"}:
        raise KTArtifactError(f"invalid export component: {component}")
    if output.exists():
        raise KTArtifactError(f"export destination already exists: {output}")
    for path in (source_path, cache_path, adapter_path):
        source_root = Path(path).resolve()
        if output.resolve().is_relative_to(source_root) or source_root.is_relative_to(
            output.resolve()
        ):
            raise KTArtifactError(
                "export destination must not overlap an input directory"
            )
    plan = resolve_native_load_plan(
        SimpleNamespace(
            kt_weight_path=str(Path(source_path).resolve()),
            kt_non_expert_weight_path=cache_path,
        ),
        source_path,
    )
    manifest = _read_json(adapter / KT_ADAPTER_MANIFEST_NAME, "KT adapter manifest")
    if (
        manifest.get("status") != "ready"
        or manifest.get("expert_weight_format") != "mxfp4"
    ):
        raise KTArtifactError("expected a ready native-MXFP4 training adapter")
    if (
        manifest.get("base", {}).get("fingerprint")
        != plan.manifest["source"]["fingerprint"]
    ):
        raise KTArtifactError("adapter base does not match the native checkpoint")
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        FUSED_EXPERT_LORA_NAME,
    }
    artifacts = manifest.get("artifacts", {})
    if not required.issubset(artifacts):
        raise KTArtifactError("adapter manifest is missing a required artifact")
    for name, record in artifacts.items():
        path = adapter / name
        if Path(name).name != name or path.is_symlink() or not path.is_file():
            raise KTArtifactError(f"invalid adapter artifact: {name}")
        if (
            path.stat().st_size != record["size"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise KTArtifactError(f"adapter artifact failed integrity check: {name}")
    config = _read_json(adapter / "adapter_config.json", "PEFT config")
    rank, alpha = int(config["r"]), float(config["lora_alpha"])
    if rank != 8 or alpha != 16 or config.get("lora_dropout", 0.0) != 0:
        raise KTArtifactError("April V4 static export requires rank=8, alpha=16 and dropout=0")
    if (
        config.get("use_dora")
        or config.get("use_rslora")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise KTArtifactError(
            "V4 static export currently requires uniform standard LoRA"
        )
    if manifest.get("lora") != {"rank": rank, "alpha": alpha}:
        raise KTArtifactError("adapter rank/alpha provenance mismatch")
    standard = load_file(str(adapter / "adapter_model.safetensors"))
    pairs, consumed = {}, []
    for name, tensor in standard.items():
        match = _STANDARD.fullmatch(name)
        if match is None or not _TARGET.fullmatch(match[1]):
            raise KTArtifactError(f"unsupported non-expert adapter tensor: {name}")
        pair = pairs.setdefault(match[1], {})
        if match[2] in pair or tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise KTArtifactError(f"invalid or duplicate adapter tensor: {name}")
        pair[match[2]] = tensor
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise KTArtifactError("incomplete standard LoRA A/B pairs")
    source = inspect_native_checkpoint(source_path)
    _validate_fused_experts(adapter / FUSED_EXPERT_LORA_NAME, source["config"], rank)
    reverse = {
        canonical_non_expert_key(key): key
        for key in source["inventory"]
        if not key.startswith("mtp.")
        and ".experts." not in key
        and not key.endswith(".scale")
    }
    output.mkdir(parents=True)
    model_dir = output / "model"
    model_dir.mkdir()
    weight_map, files, covered = {}, [], set()
    for index, checkpoint in enumerate(plan.checkpoint_files):
        tensors, exported = load_file(checkpoint), {}
        for name, weight in tensors.items():
            module = name.removesuffix(".weight")
            if module in pairs:
                covered.add(module)
                a, b = pairs[module]["A"], pairs[module]["B"]
                if tuple(a.shape) != (rank, weight.shape[1]) or tuple(b.shape) != (
                    weight.shape[0],
                    rank,
                ):
                    raise KTArtifactError(f"LoRA shape mismatch: {module}")
                if component in {"all", "nonexperts"}:
                    weight = (
                        weight.float() + (alpha / rank) * (b.float() @ a.float())
                    ).to(weight.dtype)
                    if not torch.isfinite(weight).all():
                        raise KTArtifactError(f"non-finite merged weight: {module}")
                    consumed.append(module)
            exported[reverse[name]] = weight.contiguous()
        filename = f"non-experts-{index:03d}.safetensors"
        save_file(exported, model_dir / filename, metadata={"format": "pt"})
        weight_map.update({key: filename for key in exported})
        files.append({"name": filename, "sha256": _sha256_file(model_dir / filename)})
    if covered != set(pairs):
        raise KTArtifactError("standard LoRA pair has no matching base weight")
    model_config = dict(source["config"])
    model_config.pop("quantization_config", None)
    model_config.update(
        torch_dtype="bfloat16", num_nextn_predict_layers=0, use_cache=True
    )
    _write_json_atomic(model_dir / "config.json", model_config)
    _write_json_atomic(
        model_dir / "model.safetensors.index.json",
        {"metadata": {}, "weight_map": weight_map},
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        path = Path(source_path) / name
        if path.is_file():
            shutil.copyfile(path, model_dir / name)
    expert_count = 0
    expert_enabled = component in {"all", "experts"}
    if expert_enabled or match_expert_kernel:
        expert_tensors, expert_rank, targets = _convert_fused_expert_lora(
            adapter / FUSED_EXPERT_LORA_NAME
        )
        if (
            expert_rank != rank
            or len(expert_tensors)
            != source["config"]["num_hidden_layers"]
            * source["config"]["n_routed_experts"]
            * 6
        ):
            raise KTArtifactError("expert adapter rank or tensor inventory mismatch")
        if not expert_enabled:
            expert_tensors = {
                name: torch.zeros_like(tensor) if ".lora_B." in name else tensor
                for name, tensor in expert_tensors.items()
            }
        expert_dir = output / "experts"
        expert_dir.mkdir()
        save_file(expert_tensors, expert_dir / "adapter_model.safetensors")
        _write_json_atomic(
            expert_dir / "adapter_config.json",
            _build_adapter_config(
                adapter,
                rank,
                targets,
                source_path,
                alpha,
                include_input_target_modules=False,
            ),
        )
        expert_count = len(expert_tensors)
    report = {
        "version": 1,
        "status": "ready",
        "component": component,
        "deployment": "static-merged-bf16-non-experts/native-mxfp4-expert-lora",
        "source": source_path,
        "source_fingerprint": source["fingerprint"],
        "adapter": str(adapter),
        "adapter_manifest_sha256": _sha256_file(adapter / KT_ADAPTER_MANIFEST_NAME),
        "standard_pairs_consumed": sorted(consumed),
        "standard_source_tensor_count": len(standard),
        "expert_exported_tensor_count": expert_count,
        "expert_effect_enabled": expert_enabled,
        "zero_expert_adapter_control": bool(match_expert_kernel and not expert_enabled),
        "model_files": files,
        "rounding": "standard LoRA merged in FP32 then rounded to BF16",
    }
    _write_json_atomic(output / "deployment_manifest.json", report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"standard_pairs_consumed", "model_files"}
            },
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("source", "cache", "adapter", "output"):
        parser.add_argument("--" + argument, required=True)
    parser.add_argument(
        "--component", choices=("all", "base", "experts", "nonexperts"), default="all"
    )
    parser.add_argument(
        "--match-expert-kernel",
        action="store_true",
        help="Use zero-B expert LoRA in baseline ablations so every component uses the same native SFT forward kernel.",
    )
    args = parser.parse_args()
    export(
        args.source,
        args.cache,
        args.adapter,
        args.output,
        args.component,
        args.match_expert_kernel,
    )


if __name__ == "__main__":
    main()
