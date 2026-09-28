"""Selective loading of native DeepSeek V4 MXFP4 checkpoints for LoRA SFT."""

from __future__ import annotations

import argparse
import fcntl
import logging
import shutil
import stat
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .artifacts import (
    KTArtifactError,
    KTPretrainedLoadPlan,
    KT_NON_EXPERT_INDEX_NAME,
    KT_NON_EXPERT_MANIFEST_NAME,
    _canonical_json_sha256,
    _config_value,
    _distributed_validation_context,
    _fsync_directory,
    _rawint4_shard_record,
    _read_json,
    _safe_root,
    _sha256_file,
    _synchronize_artifact_validation,
    _write_json_atomic,
)

logger = logging.getLogger(__name__)

_KIND = "deepseek-v4-non-expert-bf16"
_PREPARATION_NAME = ".kt-v4-cache-owner.json"
_FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}


def canonical_non_expert_key(key: str) -> str:
    """Map the native April checkpoint to Transformers' V4 module names."""
    if key.startswith("mtp.") or ".experts." in key:
        raise KTArtifactError(f"not a non-expert base-model tensor: {key}")
    heads = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        "hc_head_fn": "model.hc_head.hc_fn",
        "hc_head_base": "model.hc_head.hc_base",
        "hc_head_scale": "model.hc_head.hc_scale",
    }
    if key in heads:
        return heads[key]
    if not key.startswith("layers."):
        raise KTArtifactError(f"unsupported V4 checkpoint key: {key}")
    key = "model." + key
    replacements = (
        (".attn.", ".self_attn."),
        (".ffn.", ".mlp."),
        (".indexer.compressor.", ".compressor.indexer."),
        (".indexer.weights_proj.", ".compressor.indexer.scorer.weights_proj."),
        (".indexer.wq_b.", ".compressor.indexer.q_b_proj."),
        (".attn_norm.", ".input_layernorm."),
        (".ffn_norm.", ".post_attention_layernorm."),
        (".attn_sink", ".sinks"),
        (".q_norm.", ".q_a_norm."),
        (".norm.", ".kv_norm."),
        (".wq_a.", ".q_a_proj."),
        (".wq_b.", ".q_b_proj."),
        (".wkv.", ".kv_proj."),
        (".wgate.", ".gate_proj."),
        (".wo_a.", ".o_a_proj."),
        (".wo_b.", ".o_b_proj."),
        (".shared_experts.w1.", ".shared_experts.gate_proj."),
        (".shared_experts.w2.", ".shared_experts.down_proj."),
        (".shared_experts.w3.", ".shared_experts.up_proj."),
        (".gate.bias", ".gate.e_score_correction_bias"),
        (".ape", ".position_bias"),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    for native, target in (("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")):
        for suffix in ("fn", "base", "scale"):
            key = key.replace(f".{native}_{suffix}", f".{target}.{suffix}")
    return key


def dequantize_fp8_blocks(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode 128x128 E4M3 blocks; UE8M0 stores the multiplier, not its inverse."""
    if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
        raise KTArtifactError("expected E4M3 weights and UE8M0 scales")
    if weight.ndim != 2 or tuple(scale.shape) != tuple(
        math.ceil(d / 128) for d in weight.shape
    ):
        raise KTArtifactError("invalid V4 non-expert FP8 weight/scale shape")
    scales = scale.float()
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise KTArtifactError("non-finite or non-positive FP8 scales")
    result = torch.empty(weight.shape, dtype=torch.bfloat16)
    for row in range(0, weight.shape[0], 128):
        block = weight[row : row + 128].float()
        multipliers = scales[row // 128].repeat_interleave(128)[: weight.shape[1]]
        block.mul_(multipliers)
        if not torch.isfinite(block).all():
            raise KTArtifactError("non-finite FP8 non-expert weight")
        result[row : row + 128].copy_(block)
        if not torch.isfinite(result[row : row + 128]).all():
            raise KTArtifactError("FP8 non-expert weight overflows BF16")
    return result


def inspect_native_checkpoint(source: str | os.PathLike[str]) -> dict:
    """Validate expert headers without materializing the 148 GiB expert payload."""
    root = _safe_root(source, "DeepSeek V4 checkpoint")
    config = _read_json(root / "config.json", "DeepSeek V4 config")
    if config.get("model_type") != "deepseek_v4":
        raise KTArtifactError("native MXFP4 SFT currently requires DeepSeek V4")
    quant = config.get("quantization_config", {})
    if (
        quant.get("quant_method") != "fp8"
        or quant.get("fmt") != "e4m3"
        or quant.get("weight_block_size") != [128, 128]
        or quant.get("scale_fmt") != "ue8m0"
    ):
        raise KTArtifactError("expected the native V4 FP8/UE8M0 non-expert checkpoint")
    index = _read_json(root / KT_NON_EXPERT_INDEX_NAME, "native safetensors index")
    weight_map = index.get("weight_map", {})
    if not weight_map or any(
        not isinstance(v, str) or Path(v).name != v for v in weight_map.values()
    ):
        raise KTArtifactError("invalid native checkpoint shard inventory")
    dimensions = [
        config.get(k)
        for k in (
            "num_hidden_layers",
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
        )
    ]
    if any(type(value) is not int or value <= 0 for value in dimensions):
        raise KTArtifactError("native V4 dimensions must be positive integers")
    layers, experts, hidden, inter = dimensions
    if min(layers, experts, hidden, inter) <= 0 or hidden % 32 or inter % 32:
        raise KTArtifactError("invalid MXFP4 routed-expert dimensions")
    expected = {}
    for layer in range(layers):
        for expert in range(experts):
            for proj, (n, k) in {
                "w1": (inter, hidden),
                "w3": (inter, hidden),
                "w2": (hidden, inter),
            }.items():
                prefix = f"layers.{layer}.ffn.experts.{expert}.{proj}"
                expected[prefix + ".weight"] = ([n, k // 2], {"I8", "U8"})
                expected[prefix + ".scale"] = ([n, k // 32], {"F8_E8M0"})
    inventory, shard_records = {}, []
    for name in sorted(set(weight_map.values())):
        path = root / name
        # Header + sampled payload identity is also used by the native INT4 contract.
        shard_records.append(_rawint4_shard_record(path))
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if weight_map.get(key) != name or key in inventory:
                    raise KTArtifactError(f"native shard/index mismatch: {key}")
                tensor = handle.get_slice(key)
                shape, dtype = tensor.get_shape(), tensor.get_dtype()
                inventory[key] = {"shape": shape, "dtype": dtype}
                if key in expected:
                    expected_shape, expected_dtypes = expected[key]
                    if shape != expected_shape or dtype not in expected_dtypes:
                        raise KTArtifactError(
                            f"invalid native MXFP4 tensor {key}: {shape}, {dtype}"
                        )
                elif ".experts." in key and not key.startswith("mtp."):
                    raise KTArtifactError(f"unexpected routed-expert tensor: {key}")
    if set(inventory) != set(weight_map) or not set(expected).issubset(inventory):
        raise KTArtifactError("incomplete native MXFP4 checkpoint")
    identity = {
        "config_sha256": _sha256_file(root / "config.json"),
        "index_sha256": _sha256_file(root / KT_NON_EXPERT_INDEX_NAME),
        "shards": shard_records,
    }
    return {
        "model_name_or_path": str(root),
        "fingerprint": _canonical_json_sha256(identity),
        "identity": identity,
        "config": config,
        "inventory": inventory,
        "weight_map": weight_map,
    }


def _expected_cache_inventory(source: dict) -> dict:
    result = {}
    for key, spec in source["inventory"].items():
        if key.startswith("mtp.") or ".experts." in key:
            continue
        if key.endswith(".scale"):
            weight = key.removesuffix(".scale") + ".weight"
            if source["inventory"].get(weight, {}).get("dtype") not in _FP8_DTYPES:
                raise KTArtifactError(f"orphan non-expert FP8 scale: {key}")
            continue
        dtype = spec["dtype"]
        if dtype in _FP8_DTYPES:
            scale = key.removesuffix(".weight") + ".scale"
            if source["inventory"].get(scale, {}).get("dtype") != "F8_E8M0":
                raise KTArtifactError(f"missing native FP8 scale: {key}")
            dtype = "BF16"
        if dtype not in {"BF16", "F32", "I64"}:
            raise KTArtifactError(f"unsupported non-expert dtype {dtype}: {key}")
        target = canonical_non_expert_key(key)
        if target in result:
            raise KTArtifactError(f"duplicate converted key: {target}")
        result[target] = {"shape": spec["shape"], "dtype": dtype}
    return result


def _convert_non_expert_cache(source: dict, output: Path) -> dict:
    """Write only inside a private staging directory owned by the cache lock."""
    source_path = source["model_name_or_path"]
    expected = _expected_cache_inventory(source)
    weight_map, records = {}, []
    root = Path(source["model_name_or_path"])
    for shard in sorted(set(source["weight_map"].values())):
        tensors = {}
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if (
                    key.startswith("mtp.")
                    or ".experts." in key
                    or key.endswith(".scale")
                ):
                    continue
                target = canonical_non_expert_key(key)
                tensor = handle.get_tensor(key)
                if source["inventory"][key]["dtype"] in _FP8_DTYPES:
                    scale_key = key.removesuffix(".weight") + ".scale"
                    scale_shard = source["weight_map"][scale_key]
                    with safe_open(
                        root / scale_shard, framework="pt", device="cpu"
                    ) as scales:
                        tensor = dequantize_fp8_blocks(
                            tensor, scales.get_tensor(scale_key)
                        )
                elif tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise KTArtifactError(f"non-finite source tensor: {key}")
                tensors[target] = tensor.contiguous()
        if not tensors:
            continue
        name = f"non-expert-{len(records):03d}.safetensors"
        save_file(tensors, output / name, metadata={"format": "pt"})
        records.append(
            {
                "name": name,
                "size": (output / name).stat().st_size,
                "sha256": _sha256_file(output / name),
            }
        )
        weight_map.update({key: name for key in tensors})
        print(f"converted {shard}: {len(tensors)} non-expert tensors", flush=True)
    if set(weight_map) != set(expected):
        raise KTArtifactError("converted cache is missing non-expert tensors")
    _write_json_atomic(
        output / KT_NON_EXPERT_INDEX_NAME, {"metadata": {}, "weight_map": weight_map}
    )
    if source["identity"] != inspect_native_checkpoint(source_path)["identity"]:
        raise KTArtifactError("source checkpoint changed during conversion")
    payload = {
        "kind": _KIND,
        "version": 1,
        "status": "ready",
        "source": {
            key: source[key]
            for key in ("model_name_or_path", "fingerprint", "identity")
        },
        "expert_weight_format": "mxfp4",
        "files": records,
        "tensors": expected,
        "index_sha256": _sha256_file(output / KT_NON_EXPERT_INDEX_NAME),
    }
    payload["fingerprint"] = _canonical_json_sha256(payload)
    _write_json_atomic(output / KT_NON_EXPERT_MANIFEST_NAME, payload)
    return payload


def _validate_non_expert_cache(source: dict, root: Path, *, verify_payload: bool = True) -> dict:
    manifest_path = root / KT_NON_EXPERT_MANIFEST_NAME
    manifest = _read_json(manifest_path, "V4 non-expert cache manifest")
    if (manifest.get("kind"), manifest.get("version"), manifest.get("status")) != (
        _KIND,
        1,
        "ready",
    ):
        raise KTArtifactError("invalid V4 cache ready manifest")
    body = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if manifest.get("fingerprint") != _canonical_json_sha256(body):
        raise KTArtifactError("V4 cache manifest fingerprint mismatch")
    if manifest.get("source") != {
        key: source[key]
        for key in ("model_name_or_path", "fingerprint", "identity")
    }:
        raise KTArtifactError("native V4 checkpoint changed after cache conversion")
    expected = _expected_cache_inventory(source)
    if manifest.get("tensors") != expected:
        raise KTArtifactError("V4 cache tensor inventory differs from its source")
    index_path = root / KT_NON_EXPERT_INDEX_NAME
    if manifest.get("index_sha256") != _sha256_file(index_path):
        raise KTArtifactError("V4 cache index hash mismatch")
    index = _read_json(index_path, "V4 cache index")["weight_map"]
    observed, files = {}, []
    for record in manifest["files"]:
        name = record["name"]
        if Path(name).name != name or name in files:
            raise KTArtifactError("invalid V4 cache shard name")
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size"]
        ):
            raise KTArtifactError(f"invalid V4 cache shard: {path}")
        if verify_payload and _sha256_file(path) != record["sha256"]:
            raise KTArtifactError(f"V4 cache payload hash mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                if key in observed or index.get(key) != name:
                    raise KTArtifactError(
                        f"duplicate or unindexed cache tensor: {key}"
                    )
                observed[key] = {
                    "shape": tensor.get_shape(),
                    "dtype": tensor.get_dtype(),
                }
        files.append(name)
    if (
        observed != expected
        or set(index) != set(expected)
        or set(index.values()) != set(files)
    ):
        raise KTArtifactError("V4 cache payload inventory mismatch")
    return manifest


def _ensure_non_expert_cache(source: dict, output_path: str, *, verify_payload: bool = True) -> dict:
    if not output_path:
        raise KTArtifactError("Set kt_non_expert_weight_path to a writable non-expert cache directory")
    output = Path(output_path).expanduser().absolute()
    source_root = Path(source["model_name_or_path"])
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(source_root) or source_root.is_relative_to(resolved_output):
        raise KTArtifactError("V4 non-expert cache must not overlap the source checkpoint")
    if output.is_symlink():
        raise KTArtifactError(f"V4 cache must be a real directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink the lock: all processes must keep locking the same inode.
    lock_path = output.with_name(f".{output.name}.kt-v4-cache.lock")
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise KTArtifactError(f"V4 cache lock must be a regular file: {lock_path}")
        logger.info("Waiting for V4 non-expert cache lock: %s", lock_path)
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise KTArtifactError(f"V4 cache must be a real directory: {output}")
        if output.exists() and any(output.iterdir()):
            # Existing caches are immutable. Never silently replace damaged or unrelated data.
            return _validate_non_expert_cache(source, output, verify_payload=verify_payload)

        staging = output.with_name(f".{output.name}.kt-v4-cache.partial")
        owner = {"kind": _KIND, "target": str(output), "source": source["fingerprint"]}
        if staging.is_symlink():
            raise KTArtifactError(f"V4 cache staging directory must not be a symlink: {staging}")
        if staging.exists():
            if not staging.is_dir() or _read_json(staging / _PREPARATION_NAME, "V4 cache owner") != owner:
                raise KTArtifactError(f"Refusing to replace an unowned V4 staging directory: {staging}")
            logger.info("Removing interrupted V4 cache preparation: %s", staging)
            shutil.rmtree(staging)

        sizes = {"BF16": 2, "F32": 4, "I64": 8}
        payload_bytes = sum(math.prod(v["shape"]) * sizes[v["dtype"]] for v in _expected_cache_inventory(source).values())
        required_bytes = payload_bytes + max(16 * 1024**2, payload_bytes // 100)
        free_bytes = shutil.disk_usage(output.parent).free
        if free_bytes < required_bytes:
            raise KTArtifactError(
                f"Insufficient space for V4 non-expert cache at {output}: "
                f"need at least {required_bytes} bytes, available {free_bytes} bytes"
            )
        staging.mkdir(mode=0o700)
        _write_json_atomic(staging / _PREPARATION_NAME, owner)
        logger.info("Preparing V4 non-expert cache at %s (%d tensor bytes)", output, payload_bytes)
        try:
            _convert_non_expert_cache(source, staging)
            manifest = _validate_non_expert_cache(source, staging, verify_payload=verify_payload)
            # rename replaces a pre-existing empty directory, never a non-empty cache.
            os.replace(staging, output)
            _fsync_directory(output.parent)
        except Exception:
            # We created this directory while holding the lock; no unrelated paths are removed.
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
            raise
        logger.info("V4 non-expert cache is ready: %s", output)
        return manifest


def prepare_non_expert_cache(source_path: str, output_path: str) -> dict:
    """Atomically prepare or fully validate/reuse a native V4 non-expert cache."""
    return _ensure_non_expert_cache(inspect_native_checkpoint(source_path), output_path)


def resolve_native_load_plan(kt_config, source_path, explicit_quantization_config=None):
    if explicit_quantization_config is not None:
        raise KTArtifactError("native MXFP4 cannot use an explicit framework quantizer")
    dist, rank, world_size = _distributed_validation_context()
    error, signature, plan = None, None, None
    try:
        source = inspect_native_checkpoint(source_path)
        native_root = _safe_root(
            _config_value(kt_config, "kt_weight_path"), "native MXFP4 experts"
        )
        if str(native_root) != source["model_name_or_path"]:
            raise KTArtifactError(
                "MXFP4 experts and non-expert cache must use the same native base"
            )
        manifest = _ensure_non_expert_cache(
            source, _config_value(kt_config, "kt_non_expert_weight_path"), verify_payload=rank == 0
        )
        root = _safe_root(
            Path(_config_value(kt_config, "kt_non_expert_weight_path")).expanduser().absolute(),
            "V4 non-expert cache",
        )
        expected = _expected_cache_inventory(source)
        files = [record["name"] for record in manifest["files"]]
        manifest_path = root / KT_NON_EXPERT_MANIFEST_NAME
        signature = (manifest["fingerprint"], str(root), str(native_root))
        plan = KTPretrainedLoadPlan(
            source_model_name_or_path=str(native_root),
            weight_path=str(root),
            checkpoint_files=tuple(str(root / name) for name in files),
            weight_keys=frozenset(expected),
            manifest=manifest,
            manifest_path=str(manifest_path),
            routed_weight_path=str(native_root),
            routed_manifest=manifest,
            routed_manifest_path=str(manifest_path),
            lora_rank=_config_value(kt_config, "kt_lora_rank"),
            lora_alpha=_config_value(kt_config, "kt_lora_alpha"),
        )
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, KTArtifactError)
            else KTArtifactError(f"native V4 artifact validation failed: {exc}")
        )
    _synchronize_artifact_validation(dist, rank, world_size, error, signature)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_non_expert_cache(args.source, args.output)


if __name__ == "__main__":
    main()
