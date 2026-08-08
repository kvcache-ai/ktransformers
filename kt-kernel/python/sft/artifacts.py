# KT SFT artifact contracts
# SPDX-License-Identifier: Apache-2.0

"""Validated, framework-neutral artifact interfaces for KT SFT.

Transformers owns model construction and Trainer sequencing.  This module owns
the on-disk KT formats and returns immutable load plans which Transformers can
apply without knowing the manifest schema.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from safetensors import safe_open

from .weight_manifest import validate_persistent_int8_weights


KT_NON_EXPERT_MANIFEST_NAME = "kt_non_expert_manifest.json"
KT_NON_EXPERT_INDEX_NAME = "model.safetensors.index.json"
KT_ADAPTER_MANIFEST_NAME = "kt_adapter_manifest.json"
FUSED_EXPERT_LORA_NAME = "fused_expert_lora.safetensors"
KT_NON_EXPERT_MANIFEST_VERSION = 2
KT_ADAPTER_MANIFEST_VERSION = 1

_LEGACY_NON_EXPERT_VERSION = 1
_LEGACY_NON_EXPERT_PRODUCER = "llamafactory.prepare-kt-cache"
_NON_EXPERT_PRODUCER = "kt-kernel.prepare-non-expert-cache"
_STANDARD_ADAPTER_NAMES = ("adapter_model.safetensors", "adapter_model.bin")
_FUSED_LORA_NAMES = (
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
    "down_lora_a",
    "down_lora_b",
)
_FP32_ROUTER_BIAS = re.compile(r"^model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$")
_ROUTED_EXPERT = re.compile(r"(?:^|\.)experts(?:\.|$)")
_ROUTED_EXPERT_PARAMETER = re.compile(
    r"\.experts\.(?:\d+\.|gate_up_proj(?:\.|$)|down_proj(?:\.|$)|gate_proj(?:\.|$)|up_proj(?:\.|$))"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARAMETER_MARKER = "_is_kt_int8_routed_expert_base_parameter"
_MODULE_MARKER = "_is_kt_int8_routed_expert_base_module"
_MODULE_PATHS = "_kt_int8_routed_expert_module_paths"
_RUNTIME_PARAMETER_MARKER = "_is_kt_routed_expert_runtime_parameter"
_RUNTIME_MODULE_MARKER = "_is_kt_routed_expert_runtime_module"
_RUNTIME_MODULE_PATHS = "_kt_routed_expert_runtime_module_paths"
_RUNTIME_MODULE_REFS = "_kt_routed_expert_runtime_module_refs"
_RUNTIME_TENSOR_CONTRACTS = "_kt_routed_expert_runtime_tensor_contracts"
_SUPPORTED_MOE_ARCHITECTURES = (
    "DeepseekV2",
    "DeepseekV3",
    "Qwen2Moe",
    "Qwen3Moe",
    "Qwen3_5Moe",
    "Glm4Moe",
    "Mixtral",
)
_EXPERT_WEIGHT_FORMATS = frozenset({"bf16", "int8", "fp8"})


class KTArtifactError(RuntimeError):
    """A KT artifact is incomplete, unsafe, or incompatible with the runtime."""


@dataclass(frozen=True)
class KTNonExpertCache:
    """A completely validated BF16 non-expert cache."""

    path: str
    manifest_path: str
    manifest: Mapping[str, Any]
    checkpoint_files: tuple[str, ...]
    weight_keys: frozenset[str]
    source_config: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])


@dataclass(frozen=True)
class KTPretrainedLoadPlan:
    """Immutable instructions for replacing only a checkpoint's weight source."""

    source_model_name_or_path: str
    weight_path: str
    checkpoint_files: tuple[str, ...]
    weight_keys: frozenset[str]
    manifest: Mapping[str, Any]
    manifest_path: str
    routed_weight_path: str
    routed_manifest: Mapping[str, Any]
    routed_manifest_path: str
    lora_rank: int | None = None
    lora_alpha: float | None = None
    disable_source_quantizer: bool = True


@dataclass(frozen=True)
class KTAdapterManifest:
    """Validated combined standard-PEFT and KT adapter manifest."""

    path: str
    payload: Mapping[str, Any]
    artifact_paths: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise KTArtifactError(f"{description} must be a regular file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise KTArtifactError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise KTArtifactError(f"{description} must contain a JSON object: {path}")
    return payload


def _require_string(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise KTArtifactError(f"{path}: {field} must be a non-empty string")
    return value


def _require_sha256(value: Any, field: str, path: Path) -> str:
    value = _require_string(value, field, path)
    if not _SHA256.fullmatch(value):
        raise KTArtifactError(f"{path}: {field} must be a lowercase SHA256 digest")
    return value


def _positive_int(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise KTArtifactError(f"{path}: {field} must be a positive integer")
    return int(value)


def _same_source(expected: str | os.PathLike[str], actual: str) -> bool:
    expected_value = os.fspath(expected)
    if os.path.exists(expected_value) or os.path.exists(actual):
        return os.path.realpath(expected_value) == os.path.realpath(actual)
    return expected_value == actual


def _cache_fingerprint(
    version: int,
    source_fingerprint: str,
    file_records: list[dict[str, Any]],
    tensor_count: int,
    tensor_bytes: int,
    dtype_counts: Mapping[str, int],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"kt-non-expert-cache-v{version}\0".encode())
    digest.update(source_fingerprint.encode())
    digest.update(b"\0")
    digest.update(str(tensor_count).encode())
    digest.update(b"\0")
    digest.update(str(tensor_bytes).encode())
    digest.update(b"\0")
    digest.update(json.dumps(dict(dtype_counts), sort_keys=True, separators=(",", ":")).encode())
    digest.update(b"\0")
    for record in sorted(file_records, key=lambda item: item["name"]):
        digest.update(record["name"].encode())
        digest.update(b"\0")
        digest.update(str(record["size"]).encode())
        digest.update(b"\0")
        digest.update(record["sha256"].encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _source_fingerprint(config_digest: str, index_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"kt-source-checkpoint-v2\0")
    digest.update(config_digest.encode())
    digest.update(b"\0")
    digest.update(index_digest.encode())
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def _invalidate_ready_manifest(path: Path) -> None:
    """Remove an earlier ready marker before replacing any bundle member."""

    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise KTArtifactError(f"existing KT adapter manifest must be a regular file: {path}")
    path.unlink()
    _fsync_directory(path.parent)


def _safe_root(path: str | os.PathLike[str], description: str) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise KTArtifactError(f"{description} must be an absolute path: {root}")
    if root.is_symlink() or not root.is_dir():
        raise KTArtifactError(f"{description} must be a real directory: {root}")
    return root.resolve()


def _distributed_validation_context() -> tuple[Any | None, int, int]:
    """Return an initialized process group, never launcher-env guesses."""

    try:
        import torch.distributed as dist
    except ImportError:
        return None, 0, 1
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return None, 0, 1
    return dist, int(dist.get_rank()), int(dist.get_world_size())


def _synchronize_artifact_validation(
    dist: Any | None,
    rank: int,
    world_size: int,
    error: Exception | None,
    signature: tuple[Any, ...] | None,
) -> None:
    """Make validation failure and artifact disagreement symmetric across ranks."""

    if dist is None or world_size <= 1:
        if error is not None:
            raise error
        return
    local_error = None if error is None else f"rank {rank}: {type(error).__name__}: {error}"
    gathered: list[tuple[str | None, tuple[Any, ...] | None] | None] = [None] * world_size
    dist.all_gather_object(gathered, (local_error, signature))
    failures = [entry[0] for entry in gathered if entry is not None and entry[0] is not None]
    if failures:
        raise KTArtifactError("distributed KT artifact validation failed: " + "; ".join(failures))
    signatures = {entry[1] for entry in gathered if entry is not None}
    if len(signatures) != 1:
        raise KTArtifactError("distributed KT ranks resolved different artifact provenance")


def _inspect_cache_tensors(
    root: Path,
    index: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any],
    verify_hashes: bool,
    file_records: Mapping[str, Mapping[str, Any]],
) -> tuple[frozenset[str], int, dict[str, int], tuple[str, ...]]:
    index_path = root / KT_NON_EXPERT_INDEX_NAME
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise KTArtifactError(f"{index_path}: weight_map must be a non-empty object")
    shard_names: set[str] = set()
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not key:
            raise KTArtifactError(f"{index_path}: tensor names must be non-empty strings")
        if not isinstance(shard_name, str) or shard_name != os.path.basename(shard_name):
            raise KTArtifactError(f"{index_path}: unsafe shard name {shard_name!r}")
        if not shard_name.endswith(".safetensors"):
            raise KTArtifactError(f"{index_path}: every shard must use safetensors")
        shard_names.add(shard_name)

    expected_files = shard_names | {KT_NON_EXPERT_INDEX_NAME}
    if set(file_records) != expected_files:
        raise KTArtifactError(
            f"cache file inventory differs from its index: expected={sorted(expected_files)}, "
            f"actual={sorted(file_records)}"
        )

    layer_count = source_config.get("num_hidden_layers")
    mtp_prefix = f"model.layers.{layer_count}." if isinstance(layer_count, int) else None
    reject_mtp = source_config.get("model_type") == "deepseek_v3" and mtp_prefix is not None
    observed_keys: set[str] = set()
    observed_bytes = 0
    dtype_counts: dict[str, int] = {}
    checkpoint_files: list[str] = []
    for name in sorted(expected_files):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise KTArtifactError(f"cache artifact must be a regular file: {path}")
        record = file_records[name]
        if path.stat().st_size != record.get("size"):
            raise KTArtifactError(f"cache size mismatch for {name}")
        if verify_hashes and _sha256_file(path) != record.get("sha256"):
            raise KTArtifactError(f"cache SHA256 mismatch for {name}")
        if name == KT_NON_EXPERT_INDEX_NAME:
            continue
        checkpoint_files.append(str(path))
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed_keys:
                    raise KTArtifactError(f"duplicate tensor {key!r} across cache shards")
                tensor = handle.get_slice(key)
                dtype = tensor.get_dtype()
                expected_dtype = "F32" if _FP32_ROUTER_BIAS.fullmatch(key) else "BF16"
                if dtype != expected_dtype:
                    raise KTArtifactError(
                        f"cache tensor {key!r} must be {expected_dtype}, got {dtype}"
                    )
                if _ROUTED_EXPERT.search(key) or key.endswith(".weight_scale_inv"):
                    raise KTArtifactError(f"cache contains routed/quantized tensor {key!r}")
                if reject_mtp and key.startswith(mtp_prefix):
                    raise KTArtifactError(f"cache contains excluded MTP tensor {key!r}")
                elements = 1
                for dimension in tensor.get_shape():
                    elements *= dimension
                observed_bytes += elements * (4 if dtype == "F32" else 2)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                observed_keys.add(key)
    if observed_keys != set(weight_map):
        raise KTArtifactError("cache shard keys do not match the safetensors index")
    return frozenset(observed_keys), observed_bytes, dtype_counts, tuple(checkpoint_files)


def write_kt_non_expert_cache_manifest(
    cache_path: str | os.PathLike[str],
    source_model_name_or_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    """Publish a schema-v2 ready manifest for an already converted cache.

    This is the only producer in the public API.  The legacy LLaMA-Factory v1
    producer remains readable but is never emitted by kt-kernel.
    """

    root = _safe_root(cache_path, "KT non-expert cache")
    source = _safe_root(source_model_name_or_path, "source model")
    if (root / KT_NON_EXPERT_MANIFEST_NAME).exists():
        raise KTArtifactError(f"cache manifest already exists: {root / KT_NON_EXPERT_MANIFEST_NAME}")
    source_config_path = source / "config.json"
    source_index_path = source / KT_NON_EXPERT_INDEX_NAME
    source_config = _read_json(source_config_path, "source config")
    _read_json(source_index_path, "source safetensors index")
    cache_index_path = root / KT_NON_EXPERT_INDEX_NAME
    cache_index = _read_json(cache_index_path, "cache safetensors index")

    source_config_digest = _sha256_file(source_config_path)
    source_index_digest = _sha256_file(source_index_path)
    source_fingerprint = _source_fingerprint(source_config_digest, source_index_digest)
    files = [cache_index_path, *(root / name for name in sorted(set(cache_index.get("weight_map", {}).values())))]
    records: dict[str, dict[str, Any]] = {}
    for path in files:
        if path.is_symlink() or not path.is_file() or path.name in records:
            raise KTArtifactError(f"invalid or duplicate cache artifact: {path}")
        records[path.name] = {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    weight_keys, tensor_bytes, dtype_counts, _ = _inspect_cache_tensors(
        root,
        cache_index,
        source_config=source_config,
        verify_hashes=True,
        file_records=records,
    )
    quantization = source_config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise KTArtifactError("source config must describe an FP8 checkpoint")
    block_size = quantization.get("weight_block_size")
    if (
        not isinstance(block_size, (list, tuple))
        or len(block_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_size)
    ):
        raise KTArtifactError("source FP8 weight_block_size must contain two positive integers")
    record_list = [records[name] for name in sorted(records)]
    payload = {
        "version": KT_NON_EXPERT_MANIFEST_VERSION,
        "status": "ready",
        "fingerprint": _cache_fingerprint(
            KT_NON_EXPERT_MANIFEST_VERSION,
            source_fingerprint,
            record_list,
            len(weight_keys),
            tensor_bytes,
            dtype_counts,
        ),
        "source": {
            "model_name_or_path": str(source),
            "fingerprint": source_fingerprint,
            "config_sha256": source_config_digest,
            "index_sha256": source_index_digest,
        },
        "converter": {
            "name": _NON_EXPERT_PRODUCER,
            "version": KT_NON_EXPERT_MANIFEST_VERSION,
            "default_dtype": "BF16",
            "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
            "weight_block_size": list(block_size),
        },
        "files": record_list,
        "tensors": {"count": len(weight_keys), "bytes": tensor_bytes, "dtypes": dtype_counts},
    }
    _write_json_atomic(root / KT_NON_EXPERT_MANIFEST_NAME, payload)
    return payload


def _validate_non_expert_cache(
    cache_path: str | os.PathLike[str],
    source_model_name_or_path: str | os.PathLike[str],
    *,
    verify_hashes: bool = True,
) -> KTNonExpertCache:
    root = _safe_root(cache_path, "KT non-expert cache")
    source = _safe_root(source_model_name_or_path, "source model")
    manifest_path = root / KT_NON_EXPERT_MANIFEST_NAME
    manifest = _read_json(manifest_path, "KT non-expert cache manifest")
    version = manifest.get("version")
    if version not in {_LEGACY_NON_EXPERT_VERSION, KT_NON_EXPERT_MANIFEST_VERSION}:
        raise KTArtifactError(f"{manifest_path}: unsupported version {version!r}")
    if manifest.get("status") != "ready":
        raise KTArtifactError(f"{manifest_path}: status must be 'ready'")
    fingerprint = _require_sha256(manifest.get("fingerprint"), "fingerprint", manifest_path)
    source_record = manifest.get("source")
    if not isinstance(source_record, dict):
        raise KTArtifactError(f"{manifest_path}: source must be an object")
    recorded_source = _require_string(
        source_record.get("model_name_or_path"), "source.model_name_or_path", manifest_path
    )
    if not _same_source(source, recorded_source):
        raise KTArtifactError(
            f"{manifest_path}: cache source {recorded_source!r} does not match {str(source)!r}"
        )
    source_fingerprint = _require_sha256(
        source_record.get("fingerprint"), "source.fingerprint", manifest_path
    )
    source_config_path = source / "config.json"
    source_index_path = source / KT_NON_EXPERT_INDEX_NAME
    expected_source_files = {
        "config_sha256": source_config_path,
        "index_sha256": source_index_path,
    }
    for field, path in expected_source_files.items():
        expected = _require_sha256(source_record.get(field), f"source.{field}", manifest_path)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected:
            raise KTArtifactError(f"{manifest_path}: source.{field} does not match {path}")
    if version == KT_NON_EXPERT_MANIFEST_VERSION:
        expected_source_fingerprint = _source_fingerprint(
            source_record["config_sha256"], source_record["index_sha256"]
        )
        if source_fingerprint != expected_source_fingerprint:
            raise KTArtifactError(f"{manifest_path}: source fingerprint is invalid")
    source_config = _read_json(source_config_path, "source config")
    quantization = source_config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise KTArtifactError(f"{manifest_path}: source config must describe an FP8 checkpoint")
    block_size = quantization.get("weight_block_size")
    if (
        not isinstance(block_size, (list, tuple))
        or len(block_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in block_size)
    ):
        raise KTArtifactError(f"{manifest_path}: invalid source FP8 weight_block_size")
    producer = _LEGACY_NON_EXPERT_PRODUCER if version == 1 else _NON_EXPERT_PRODUCER
    expected_converter = {
        "name": producer,
        "version": version,
        "default_dtype": "BF16",
        "fp32_exceptions": ["model.layers.*.mlp.gate.e_score_correction_bias"],
        "weight_block_size": list(block_size),
    }
    if manifest.get("converter") != expected_converter:
        raise KTArtifactError(f"{manifest_path}: converter contract is unsupported")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise KTArtifactError(f"{manifest_path}: files must be a non-empty list")
    records: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise KTArtifactError(f"{manifest_path}: files[{index}] must be an object")
        name = _require_string(record.get("name"), f"files[{index}].name", manifest_path)
        if name != os.path.basename(name) or name in records:
            raise KTArtifactError(f"{manifest_path}: invalid or duplicate filename {name!r}")
        _positive_int(record.get("size"), f"files[{index}].size", manifest_path)
        _require_sha256(record.get("sha256"), f"files[{index}].sha256", manifest_path)
        records[name] = record
    cache_index = _read_json(root / KT_NON_EXPERT_INDEX_NAME, "cache safetensors index")
    weight_keys, observed_bytes, observed_dtypes, checkpoint_files = _inspect_cache_tensors(
        root,
        cache_index,
        source_config=source_config,
        verify_hashes=verify_hashes,
        file_records=records,
    )
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise KTArtifactError(f"{manifest_path}: tensors must be an object")
    if tensors.get("count") != len(weight_keys):
        raise KTArtifactError(f"{manifest_path}: tensor count does not match the index")
    if tensors.get("bytes") != observed_bytes or tensors.get("dtypes") != observed_dtypes:
        raise KTArtifactError(f"{manifest_path}: tensor metadata does not match cache shards")
    expected_fingerprint = _cache_fingerprint(
        int(version),
        source_fingerprint,
        [dict(record) for record in records.values()],
        len(weight_keys),
        observed_bytes,
        observed_dtypes,
    )
    if fingerprint != expected_fingerprint:
        raise KTArtifactError(f"{manifest_path}: fingerprint does not match its records")
    return KTNonExpertCache(
        path=str(root),
        manifest_path=str(manifest_path),
        manifest=manifest,
        checkpoint_files=checkpoint_files,
        weight_keys=weight_keys,
        source_config=source_config,
    )


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def is_kt_routed_expert_parameter_name(name: str) -> bool:
    """Whether a checkpoint key is a routed expert base parameter owned by KT."""

    return isinstance(name, str) and _ROUTED_EXPERT_PARAMETER.search(name) is not None


def is_kt_supported_moe_model(model: Any) -> bool:
    """Return whether the model architecture is implemented by KT SFT wrappers."""

    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)):
        return False
    return any(
        isinstance(architecture, str)
        and any(marker in architecture for marker in _SUPPORTED_MOE_ARCHITECTURES)
        for architecture in architectures
    )


def _loading_value(loading_info: Any, name: str, default: Any) -> Any:
    if isinstance(loading_info, Mapping):
        return loading_info.get(name, default)
    return getattr(loading_info, name, default)


def validate_kt_prequantized_loading_info(
    kt_config: Any,
    loading_info: Any,
    model: Any | None = None,
) -> None:
    """Fail closed after native FP8/INT8 routed experts were skipped.

    This covers the pre-quantized path without a BF16 non-expert load plan.
    The plan-based path additionally calls :func:`validate_kt_pretrained_load`
    to compare the instantiated model against the cache's exact key set.
    """

    weight_format = str(_config_value(kt_config, "kt_expert_weight_format", "")).lower()
    skip_loading = _config_value(kt_config, "kt_skip_expert_loading", True)
    if weight_format not in {"int8", "fp8"} or not bool(skip_loading):
        return
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    layer_count = getattr(config, "num_hidden_layers", None)
    mtp_prefix = (
        f"model.layers.{layer_count}."
        if model_type == "deepseek_v3" and isinstance(layer_count, int) and not isinstance(layer_count, bool)
        else None
    )
    missing = sorted(
        key
        for key in (_loading_value(loading_info, "missing_keys", ()) or ())
        if not is_kt_routed_expert_parameter_name(key)
    )
    mismatched = sorted(
        mismatch
        for mismatch in (_loading_value(loading_info, "mismatched_keys", ()) or ())
        if not is_kt_routed_expert_parameter_name(mismatch[0])
    )
    conversion_errors = {
        key: error
        for key, error in (_loading_value(loading_info, "conversion_errors", {}) or {}).items()
        if not is_kt_routed_expert_parameter_name(key)
    }
    unexpected = sorted(
        key
        for key in (_loading_value(loading_info, "unexpected_keys", ()) or ())
        if not (mtp_prefix is not None and key.startswith(mtp_prefix))
    )
    error_messages = list(_loading_value(loading_info, "error_msgs", ()) or ())
    failures = []
    if missing:
        failures.append(f"missing_keys={missing}")
    if mismatched:
        failures.append(f"mismatched_keys={mismatched}")
    if conversion_errors:
        failures.append(f"conversion_errors={conversion_errors}")
    if unexpected:
        failures.append(f"unexpected_keys={unexpected}")
    if error_messages:
        failures.append(f"error_msgs={error_messages}")
    if failures:
        raise KTArtifactError(
            f"KT {weight_format.upper()} checkpoint loading requires an exact non-expert model match; "
            + "; ".join(failures)
        )


def resolve_kt_pretrained_artifacts(
    kt_config: Any,
    pretrained_model_name_or_path: str | os.PathLike[str],
    quantization_config: Any | None = None,
) -> KTPretrainedLoadPlan | None:
    """Resolve and fully validate KT-owned pretrained artifacts.

    ``quantization_config`` means an explicit caller override.  The source
    checkpoint's embedded FP8 quantizer is represented by
    ``disable_source_quantizer`` on the returned plan instead.
    """

    cache_path = _config_value(kt_config, "kt_non_expert_weight_path")
    if not cache_path:
        return None
    if str(_config_value(kt_config, "kt_expert_weight_format", "")).lower() != "int8":
        raise KTArtifactError("kt_non_expert_weight_path requires routed INT8 expert weights")
    if quantization_config is not None:
        raise KTArtifactError(
            "kt_non_expert_weight_path cannot be combined with an explicit quantization_config"
        )
    weight_path = _config_value(kt_config, "kt_weight_path")
    if not isinstance(weight_path, str) or not weight_path:
        raise KTArtifactError("routed INT8 loading requires kt_weight_path")

    dist, distributed_rank, world_size = _distributed_validation_context()
    cache = None
    validated_routed = None
    routed_manifest = None
    local_error = None
    signature = None
    try:
        # All ranks retain structural, size, dtype, and inventory checks. Only
        # rank 0 streams the large payloads for SHA256 verification.
        cache = _validate_non_expert_cache(
            cache_path,
            pretrained_model_name_or_path,
            verify_hashes=distributed_rank == 0,
        )
        source_config = cache.source_config
        layer_count = _positive_int(
            source_config.get("num_hidden_layers"), "num_hidden_layers", Path(cache.manifest_path)
        )
        first_moe_layer = source_config.get("first_k_dense_replace", 0)
        if isinstance(first_moe_layer, bool) or not isinstance(first_moe_layer, int) or first_moe_layer < 0:
            raise KTArtifactError("source first_k_dense_replace must be a non-negative integer")
        layer_indices = tuple(range(first_moe_layer, layer_count))
        validated_routed = validate_persistent_int8_weights(
            weight_path,
            layer_indices=layer_indices,
            numa_count=int(_config_value(kt_config, "kt_threadpool_count", 1) or 1),
            expert_num=int(source_config["n_routed_experts"]),
            hidden_size=int(source_config["hidden_size"]),
            intermediate_size=int(source_config["moe_intermediate_size"]),
            # None hashes schema-v2 but keeps legacy v1 on its strict
            # structure/size contract; non-owner ranks never repeat hashes.
            verify_hashes=None if distributed_rank == 0 else False,
        )
        routed_manifest = _read_json(validated_routed.path, "routed INT8 manifest")
        signature = (
            cache.fingerprint,
            cache.path,
            str(validated_routed.root),
            hashlib.sha256(
                json.dumps(routed_manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
    except Exception as exc:
        local_error = (
            exc
            if isinstance(exc, KTArtifactError)
            else KTArtifactError(f"routed INT8 artifact validation failed: {exc}")
        )
    _synchronize_artifact_validation(dist, distributed_rank, world_size, local_error, signature)
    if cache is None or validated_routed is None or routed_manifest is None:
        raise KTArtifactError("KT artifact validation completed without a load plan")
    rank = _config_value(kt_config, "kt_lora_rank")
    alpha = _config_value(kt_config, "kt_lora_alpha")
    return KTPretrainedLoadPlan(
        source_model_name_or_path=os.fspath(pretrained_model_name_or_path),
        weight_path=cache.path,
        checkpoint_files=cache.checkpoint_files,
        weight_keys=cache.weight_keys,
        manifest=cache.manifest,
        manifest_path=cache.manifest_path,
        routed_weight_path=str(validated_routed.root),
        routed_manifest=routed_manifest,
        routed_manifest_path=str(validated_routed.path),
        lora_rank=int(rank) if isinstance(rank, int) and not isinstance(rank, bool) else None,
        lora_alpha=float(alpha) if isinstance(alpha, (int, float)) and not isinstance(alpha, bool) else None,
    )


def _canonical_loaded_key(key: str) -> str | None:
    if "._original_router.weight" in key:
        return None
    if "._original_router.e_score_correction_bias" in key:
        return key.replace(
            "._original_router.e_score_correction_bias", ".gate.e_score_correction_bias"
        )
    return key


def validate_kt_pretrained_load(
    plan: KTPretrainedLoadPlan | None,
    loading_info: Mapping[str, Any],
    model: Any,
) -> KTPretrainedLoadPlan | None:
    """Validate the instantiated model and attach immutable artifact provenance."""

    if plan is None:
        return None
    if loading_info is None:
        raise KTArtifactError("pretrained loader did not return loading diagnostics")
    missing = set(_loading_value(loading_info, "missing_keys", ()) or ())
    invalid_missing = sorted(key for key in missing if not _ROUTED_EXPERT.search(key))
    failures: dict[str, Any] = {}
    if invalid_missing:
        failures["missing_keys"] = invalid_missing
    unexpected = list(_loading_value(loading_info, "unexpected_keys", ()) or ())
    mismatched = [
        value
        for value in (_loading_value(loading_info, "mismatched_keys", ()) or ())
        if not is_kt_routed_expert_parameter_name(value[0])
    ]
    error_messages = list(_loading_value(loading_info, "error_msgs", ()) or ())
    if unexpected:
        failures["unexpected_keys"] = unexpected
    if mismatched:
        failures["mismatched_keys"] = mismatched
    if error_messages:
        failures["error_msgs"] = error_messages
    conversion_errors = {
        key: value
        for key, value in (_loading_value(loading_info, "conversion_errors", {}) or {}).items()
        if not is_kt_routed_expert_parameter_name(key)
    }
    if conversion_errors:
        failures["conversion_errors"] = dict(conversion_errors)
    if failures:
        raise KTArtifactError(f"non-expert checkpoint did not load exactly: {failures}")
    state_dict = model.state_dict()
    loaded_keys = set()
    for key in state_dict:
        canonical = _canonical_loaded_key(key)
        if canonical is not None and not _ROUTED_EXPERT.search(canonical):
            loaded_keys.add(canonical)
    if loaded_keys != set(plan.weight_keys):
        raise KTArtifactError(
            "loaded non-expert keys differ from the validated cache: "
            f"missing={sorted(set(plan.weight_keys) - loaded_keys)[:16]}, "
            f"unexpected={sorted(loaded_keys - set(plan.weight_keys))[:16]}"
        )
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "name_or_path"):
            config.name_or_path = plan.source_model_name_or_path
        if hasattr(config, "_name_or_path"):
            config._name_or_path = plan.source_model_name_or_path
    model._kt_pretrained_load_plan = plan
    model._kt_base_model_name_or_path = plan.source_model_name_or_path
    model._kt_non_expert_cache_path = plan.weight_path
    model._kt_non_expert_cache_manifest = plan.manifest
    model._kt_routed_int8_manifest_path = plan.routed_manifest_path
    model._kt_routed_int8_manifest = plan.routed_manifest
    return plan


def _deepseek_routed_paths(config: Any) -> tuple[str, ...]:
    layer_count = getattr(config, "num_hidden_layers", None)
    first_moe_layer = getattr(config, "first_k_dense_replace", None)
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
        or isinstance(first_moe_layer, bool)
        or not isinstance(first_moe_layer, int)
        or first_moe_layer < 0
        or first_moe_layer >= layer_count
    ):
        raise KTArtifactError("model config has an invalid routed-expert layer range")
    return tuple(f"model.layers.{index}.mlp.experts" for index in range(first_moe_layer, layer_count))


def _effective_runtime_shape(tensor: Any, name: str) -> tuple[int, ...]:
    shape = getattr(tensor, "_kt_original_shape", None) if getattr(tensor, "_kt_zero_storage", False) else tensor.shape
    try:
        resolved = tuple(int(value) for value in shape)
    except (TypeError, ValueError) as exc:
        raise KTArtifactError(f"{name} has invalid routed-expert shape metadata") from exc
    if not resolved or any(value <= 0 for value in resolved):
        raise KTArtifactError(f"{name} has invalid routed-expert shape {resolved}")
    return resolved


def _runtime_tensor_contract(path: str, module: Any) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    entries = []
    seen = set()
    for kind, tensors in (
        ("parameter", module.named_parameters(recurse=True, remove_duplicate=False)),
        ("buffer", module.named_buffers(recurse=True, remove_duplicate=False)),
    ):
        for name, tensor in tensors:
            key = (kind, name)
            if key in seen:
                raise KTArtifactError(f"{path} exposes duplicate routed-expert {kind} {name!r}")
            seen.add(key)
            entries.append((kind, name, _effective_runtime_shape(tensor, f"{path}.{name}")))
    if not any(kind == "parameter" for kind, _, _ in entries):
        raise KTArtifactError(f"routed-expert subtree {path!r} contains no parameters")
    return tuple(sorted(entries))


def _validate_runtime_expert_structure(path: str, experts: Any, moe_config: Any, hidden_size: Any) -> None:
    import torch

    dimensions = (moe_config.expert_num, moe_config.intermediate_size, hidden_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise KTArtifactError(f"{path} has invalid routed-expert dimensions {dimensions}")
    expert_num, intermediate_size, hidden_size = dimensions

    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if isinstance(gate_up, torch.nn.Parameter) or isinstance(down, torch.nn.Parameter):
        if not isinstance(gate_up, torch.nn.Parameter) or not isinstance(down, torch.nn.Parameter):
            raise KTArtifactError(f"{path} must register both fused gate_up_proj and down_proj parameters")
        expected = {
            "gate_up_proj": (expert_num, 2 * intermediate_size, hidden_size),
            "down_proj": (expert_num, hidden_size, intermediate_size),
        }
        for name, parameter in (("gate_up_proj", gate_up), ("down_proj", down)):
            actual = _effective_runtime_shape(parameter, f"{path}.{name}")
            if actual != expected[name]:
                raise KTArtifactError(f"{path}.{name} shape mismatch: expected={expected[name]}, actual={actual}")
        return

    children = tuple(experts.named_children())
    expected_names = tuple(str(index) for index in range(expert_num))
    if tuple(name for name, _ in children) != expected_names:
        raise KTArtifactError(
            f"{path} expert inventory mismatch: expected={list(expected_names)}, "
            f"actual={[name for name, _ in children]}"
        )
    gate_name, up_name, down_name = moe_config.weight_names
    expected_shapes = {
        gate_name: (intermediate_size, hidden_size),
        up_name: (intermediate_size, hidden_size),
        down_name: (hidden_size, intermediate_size),
    }
    for expert_name, expert in children:
        for projection_name, expected_shape in expected_shapes.items():
            projection = getattr(expert, projection_name, None)
            weight = getattr(projection, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                raise KTArtifactError(f"{path}.{expert_name}.{projection_name} does not expose a weight Parameter")
            actual_shape = _effective_runtime_shape(weight, f"{path}.{expert_name}.{projection_name}.weight")
            if actual_shape != expected_shape:
                raise KTArtifactError(
                    f"{path}.{expert_name}.{projection_name}.weight shape mismatch: "
                    f"expected={expected_shape}, actual={actual_shape}"
                )


def _enumerate_runtime_routed_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    if not is_kt_supported_moe_model(model):
        return ()
    try:
        from .arch import _get_layers_prefix, _get_model_container_and_layers, get_moe_arch_config, get_moe_module

        config = model.config
        moe_config = get_moe_arch_config(config)
        _, layers = _get_model_container_and_layers(model, purpose="routed-expert ownership")
        layers_path = _get_layers_prefix(config)
        registered_layers = model.get_submodule(layers_path)
    except Exception as exc:
        raise KTArtifactError(f"could not enumerate KT routed-expert layers: {exc}") from exc
    if registered_layers is not layers:
        raise KTArtifactError(f"model layer path {layers_path!r} does not resolve to the discovered layer container")

    text_config = getattr(config, "text_config", config)
    hidden_size = getattr(text_config, "hidden_size", None)
    modules = []
    identities = set()
    for layer_index, layer in enumerate(layers):
        moe_module = get_moe_module(layer, moe_config)
        if moe_module is None:
            continue
        experts = getattr(moe_module, moe_config.experts_attr, None)
        if experts is None or not hasattr(experts, "named_parameters"):
            raise KTArtifactError(f"layer {layer_index} does not expose a registered routed-expert module")
        path = f"{layers_path}.{layer_index}.{moe_config.moe_layer_attr}.{moe_config.experts_attr}"
        try:
            registered = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise KTArtifactError(f"missing routed-expert subtree {path!r}") from exc
        if registered is not experts:
            raise KTArtifactError(f"routed-expert subtree {path!r} does not preserve module identity")
        if id(experts) in identities:
            raise KTArtifactError(f"routed-expert subtree {path!r} shares a module with another layer")
        identities.add(id(experts))
        _validate_runtime_expert_structure(path, experts, moe_config, hidden_size)
        modules.append((path, experts))
    if not modules:
        raise KTArtifactError("supported KT MoE model contains no routed-expert layers")
    return tuple(modules)


def _validated_runtime_routed_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    metadata = (
        getattr(model, _RUNTIME_MODULE_PATHS, None),
        getattr(model, _RUNTIME_MODULE_REFS, None),
        getattr(model, _RUNTIME_TENSOR_CONTRACTS, None),
    )
    if metadata == (None, None, None):
        return ()
    if any(value is None for value in metadata):
        raise KTArtifactError("routed-expert runtime ownership metadata is incomplete")
    paths, module_refs, contracts = metadata
    if not isinstance(paths, tuple) or not isinstance(module_refs, tuple) or not isinstance(contracts, tuple):
        raise KTArtifactError("routed-expert runtime ownership metadata has invalid types")

    enumerated = _enumerate_runtime_routed_modules(model)
    expected_paths = tuple(path for path, _ in enumerated)
    if paths != expected_paths:
        raise KTArtifactError("routed-expert runtime ownership paths changed")
    if len(module_refs) != len(paths) or len(contracts) != len(paths):
        raise KTArtifactError("routed-expert runtime ownership metadata has inconsistent lengths")

    validated = []
    for index, ((path, current), claimed, contract) in enumerate(zip(enumerated, module_refs, contracts)):
        if current is not claimed:
            raise KTArtifactError(f"routed-expert subtree {path!r} changed module identity")
        if getattr(current, _RUNTIME_MODULE_MARKER, False) is not True:
            raise KTArtifactError(f"routed-expert subtree {path!r} lost its runtime ownership marker")
        if _runtime_tensor_contract(path, current) != contract:
            raise KTArtifactError(f"routed-expert subtree {path!r} changed its tensor contract")
        validated.append((paths[index], current))
    return tuple(validated)


def claim_kt_routed_expert_subtrees(model: Any) -> tuple[str, ...]:
    """Claim routed-expert subtrees that KT will own outside the framework state dict."""

    existing = (
        getattr(model, _RUNTIME_MODULE_PATHS, None),
        getattr(model, _RUNTIME_MODULE_REFS, None),
        getattr(model, _RUNTIME_TENSOR_CONTRACTS, None),
    )
    if existing != (None, None, None):
        return tuple(path for path, _ in _validated_runtime_routed_modules(model))

    modules = _enumerate_runtime_routed_modules(model)
    if not modules:
        return ()
    for path, module in modules:
        if getattr(module, _RUNTIME_MODULE_MARKER, False):
            raise KTArtifactError(f"routed-expert subtree {path!r} has an unowned runtime marker")

    paths = tuple(path for path, _ in modules)
    refs = tuple(module for _, module in modules)
    contracts = tuple(_runtime_tensor_contract(path, module) for path, module in modules)
    marked_parameters = []
    try:
        for _, module in modules:
            setattr(module, _RUNTIME_MODULE_MARKER, True)
            for parameter in module.parameters(recurse=True):
                setattr(parameter, _RUNTIME_PARAMETER_MARKER, True)
                marked_parameters.append(parameter)
        setattr(model, _RUNTIME_MODULE_PATHS, paths)
        setattr(model, _RUNTIME_MODULE_REFS, refs)
        setattr(model, _RUNTIME_TENSOR_CONTRACTS, contracts)
    except BaseException:
        for parameter in marked_parameters:
            with contextlib.suppress(AttributeError):
                delattr(parameter, _RUNTIME_PARAMETER_MARKER)
        for _, module in modules:
            with contextlib.suppress(AttributeError):
                delattr(module, _RUNTIME_MODULE_MARKER)
        for name in (_RUNTIME_MODULE_PATHS, _RUNTIME_MODULE_REFS, _RUNTIME_TENSOR_CONTRACTS):
            with contextlib.suppress(AttributeError):
                delattr(model, name)
        raise
    return paths


def is_kt_routed_expert_base_parameter(parameter: Any) -> bool:
    """Whether a live base parameter belongs to a claimed KT routed-expert subtree."""

    return getattr(parameter, _RUNTIME_PARAMETER_MARKER, False) is True


def mark_kt_int8_routed_expert_base_parameters(
    model: Any, plan: KTPretrainedLoadPlan | None
) -> tuple[str, ...]:
    """Mark native routed-expert tensors omitted from a validated load plan."""

    if plan is None:
        return ()
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "deepseek_v3":
        return ()
    paths = _deepseek_routed_paths(config)
    expected_shapes = {
        "gate_up_proj": (
            getattr(config, "n_routed_experts", None),
            2 * getattr(config, "moe_intermediate_size", 0),
            getattr(config, "hidden_size", None),
        ),
        "down_proj": (
            getattr(config, "n_routed_experts", None),
            getattr(config, "hidden_size", None),
            getattr(config, "moe_intermediate_size", None),
        ),
    }
    if any(any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape) for shape in expected_shapes.values()):
        raise KTArtifactError("model config has invalid routed-expert dimensions")
    modules = []
    parameters = []
    for path in paths:
        try:
            experts = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise KTArtifactError(f"missing routed-expert subtree {path!r}") from exc
        actual = dict(experts.named_parameters(recurse=True))
        if set(actual) != set(expected_shapes):
            raise KTArtifactError(
                f"{path} parameter contract mismatch: expected={sorted(expected_shapes)}, actual={sorted(actual)}"
            )
        for name, shape in expected_shapes.items():
            if tuple(actual[name].shape) != shape:
                raise KTArtifactError(
                    f"{path}.{name} shape mismatch: expected={shape}, actual={tuple(actual[name].shape)}"
                )
            parameters.append(actual[name])
        modules.append(experts)
    runtime_paths = tuple(path for path, _ in _enumerate_runtime_routed_modules(model))
    if runtime_paths != paths:
        raise KTArtifactError(
            "validated INT8 routed-expert paths differ from the runtime ownership contract: "
            f"artifact={paths}, runtime={runtime_paths}"
        )
    if claim_kt_routed_expert_subtrees(model) != paths:
        raise KTArtifactError("INT8 routed-expert ownership claim returned inconsistent paths")
    for module in modules:
        setattr(module, _MODULE_MARKER, True)
    for parameter in parameters:
        setattr(parameter, _PARAMETER_MARKER, True)
    setattr(model, _MODULE_PATHS, paths)
    return paths


def is_kt_int8_routed_expert_base_parameter(parameter: Any) -> bool:
    return getattr(parameter, _PARAMETER_MARKER, False) is True


def _validated_routed_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    paths = getattr(model, _MODULE_PATHS, ())
    if not paths:
        return ()
    if tuple(paths) != _deepseek_routed_paths(model.config):
        raise KTArtifactError("routed-expert ownership metadata changed")
    modules = []
    for path in paths:
        try:
            module = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise KTArtifactError(f"routed-expert subtree {path!r} is no longer registered") from exc
        if getattr(module, _MODULE_MARKER, False) is not True:
            raise KTArtifactError(f"routed-expert subtree {path!r} lost its ownership marker")
        modules.append((path, module))
    return tuple(modules)


@contextlib.contextmanager
def project_kt_routed_experts_out_of_device_map(model: Any) -> Iterator[None]:
    """Temporarily project KT-owned routed experts to zero-sized meta tensors."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        yield
        return
    import torch

    replacements = []
    try:
        for path, experts in modules:
            for module in experts.modules():
                for name, parameter in tuple(module._parameters.items()):
                    if parameter is None:
                        continue
                    if parameter.device.type != "meta":
                        raise KTArtifactError(f"{path}.{name} must be meta before device-map inference")
                    projected = torch.nn.Parameter(
                        torch.empty(0, dtype=parameter.dtype, device="meta"),
                        requires_grad=parameter.requires_grad,
                    )
                    module._parameters[name] = projected
                    replacements.append((module._parameters, name, parameter))
                for name, buffer in tuple(module._buffers.items()):
                    if buffer is None:
                        continue
                    if buffer.device.type != "meta":
                        raise KTArtifactError(f"{path}.{name} must be meta before device-map inference")
                    module._buffers[name] = torch.empty(0, dtype=buffer.dtype, device="meta")
                    replacements.append((module._buffers, name, buffer))
        yield
    finally:
        for registry, name, original in reversed(replacements):
            registry[name] = original


def prepare_kt_non_expert_device_map(model: Any, device_map: Any) -> Any:
    """Remove virtual expert placements and reject host offload of real tensors."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        return device_map
    if not isinstance(device_map, dict):
        raise KTArtifactError("non-expert placement requires a resolved device-map dictionary")
    import torch

    paths = tuple(path for path, _ in modules)
    resolved = {
        name: device
        for name, device in device_map.items()
        if not any(name == path or name.startswith(f"{path}.") for path in paths)
    }
    if not resolved:
        raise KTArtifactError("device map became empty after removing routed experts")

    def is_host(device: Any) -> bool:
        if device == "disk":
            return True
        if isinstance(device, int):
            return False
        try:
            return torch.device(device).type in {"cpu", "meta"}
        except (RuntimeError, TypeError):
            return False

    host_entries = {name: device for name, device in resolved.items() if is_host(device)}
    if host_entries:
        raise KTArtifactError(
            "non-expert device map offloaded real tensors to CPU/disk: "
            + ", ".join(f"{name or '<root>'}={device}" for name, device in sorted(host_entries.items()))
        )
    return resolved


@contextlib.contextmanager
def hide_kt_routed_experts_from_dispatch(model: Any) -> Iterator[None]:
    """Temporarily unregister KT-owned subtrees from parent dispatch hooks."""

    modules = _validated_runtime_routed_modules(model)
    if not modules:
        yield
        return
    import torch

    replacements = []
    try:
        for path, experts in modules:
            parent_path, child_name = path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            if parent._modules.get(child_name) is not experts:
                raise KTArtifactError(f"routed-expert subtree {path!r} changed before dispatch")
            parent._modules[child_name] = torch.nn.Module()
            replacements.append((parent, child_name, experts))
        yield
    finally:
        for parent, child_name, experts in reversed(replacements):
            parent._modules[child_name] = experts


@contextlib.contextmanager
def project_kt_int8_routed_experts_out_of_device_map(model: Any) -> Iterator[None]:
    """Backward-compatible alias for the generic routed-expert projection contract."""

    _validated_routed_modules(model)
    with project_kt_routed_experts_out_of_device_map(model):
        yield


def prepare_kt_int8_non_expert_device_map(model: Any, device_map: Any) -> Any:
    """Backward-compatible alias for generic KT non-expert placement."""

    _validated_routed_modules(model)
    return prepare_kt_non_expert_device_map(model, device_map)


@contextlib.contextmanager
def hide_kt_int8_routed_experts_from_dispatch(model: Any) -> Iterator[None]:
    """Backward-compatible alias for generic KT routed-expert dispatch hiding."""

    _validated_routed_modules(model)
    with hide_kt_routed_experts_from_dispatch(model):
        yield


def _find_wrappers(model: Any) -> list[Any]:
    queue = [model]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        wrappers = getattr(candidate, "_kt_wrappers", None)
        if wrappers is not None:
            return list(wrappers)
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)
    return []


def _find_plan(model: Any) -> KTPretrainedLoadPlan | None:
    queue = [model]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        plan = getattr(candidate, "_kt_pretrained_load_plan", None)
        if isinstance(plan, KTPretrainedLoadPlan):
            return plan
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)
    return None


def _runtime_expert_weight_format(
    model: Any, plan: KTPretrainedLoadPlan | None
) -> str:
    """Resolve routed-expert precision from explicit KT ownership metadata."""

    provenance: list[tuple[str, Any]] = []
    queue = [model]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        value = getattr(candidate, "_kt_expert_weight_format", None)
        if value is not None:
            provenance.append((type(candidate).__name__, value))
        config = getattr(candidate, "config", None)
        value = getattr(config, "kt_expert_weight_format", None)
        if value is not None:
            provenance.append((f"{type(candidate).__name__}.config", value))
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)

    for wrapper in _find_wrappers(model):
        value = getattr(wrapper, "_kt_expert_weight_format", None)
        if value is not None:
            provenance.append((f"KT wrapper layer {getattr(wrapper, 'layer_idx', '?')}", value))

    if plan is not None:
        provenance.append(
            (
                "routed expert manifest",
                plan.routed_manifest.get("expert_weight_format"),
            )
        )

    normalized: list[tuple[str, str]] = []
    for source, value in provenance:
        if not isinstance(value, str) or value.strip().lower() not in _EXPERT_WEIGHT_FORMATS:
            raise KTArtifactError(f"invalid {source} expert_weight_format {value!r}")
        normalized.append((source, value.strip().lower()))
    formats = {value for _, value in normalized}
    if len(formats) > 1:
        details = ", ".join(f"{source}={value}" for source, value in normalized)
        raise KTArtifactError(f"conflicting KT expert weight provenance: {details}")
    if formats:
        return formats.pop()
    if plan is not None:
        raise KTArtifactError("validated pretrained plan lacks routed expert format provenance")
    # Older BF16 wrappers predate explicit ownership metadata. Quantized
    # runtimes never take this compatibility path.
    return "bf16"


def _dtype_name(dtype: Any) -> str:
    values = {
        "torch.bfloat16": "BF16",
        "torch.float16": "F16",
        "torch.float32": "F32",
        "torch.float64": "F64",
    }
    try:
        return values[str(dtype)]
    except KeyError as exc:
        raise KTArtifactError(f"unsupported fused expert LoRA dtype {dtype}") from exc


def _wrapper_uses_fused_lora(wrapper: Any) -> bool:
    rank = getattr(wrapper, "_lora_rank", 0)
    return bool(
        isinstance(rank, int)
        and not isinstance(rank, bool)
        and rank > 0
        and (
            getattr(wrapper, "_use_fused_expert_lora", False)
            or getattr(wrapper, "_fused_experts", False)
            or getattr(wrapper, "_fused_expert_lora_params", None) is not None
        )
    )


def _fused_lora_contract(model: Any) -> dict[str, dict[str, Any]]:
    contract: dict[str, dict[str, Any]] = {}
    seen_layers: set[int] = set()
    for wrapper in _find_wrappers(model):
        fused = getattr(wrapper, "_fused_expert_lora_params", None)
        if not fused and not _wrapper_uses_fused_lora(wrapper):
            continue
        layer_idx = getattr(wrapper, "layer_idx", None)
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx in seen_layers:
            raise KTArtifactError(f"invalid or duplicate KT wrapper layer {layer_idx!r}")
        seen_layers.add(layer_idx)
        if fused and len(fused) != len(_FUSED_LORA_NAMES):
            raise KTArtifactError(
                f"layer {layer_idx}: expected six fused LoRA parameters, got {len(fused)}"
            )
        if fused:
            for name, parameter in zip(_FUSED_LORA_NAMES, fused):
                contract[f"layers.{layer_idx}.experts.{name}"] = {
                    "shape": list(parameter.shape),
                    "dtype": _dtype_name(parameter.dtype),
                }
            continue
        moe_config = getattr(wrapper, "moe_config", None)
        expert_num = getattr(moe_config, "expert_num", None)
        intermediate_size = getattr(moe_config, "intermediate_size", None)
        hidden_size = getattr(wrapper, "hidden_size", None)
        rank = getattr(wrapper, "_lora_rank", None)
        dimensions = (expert_num, intermediate_size, hidden_size, rank)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in dimensions
        ):
            raise KTArtifactError(f"layer {layer_idx}: cannot derive its fused LoRA contract")
        shapes = (
            (expert_num, rank, hidden_size),
            (expert_num, intermediate_size, rank),
            (expert_num, rank, hidden_size),
            (expert_num, intermediate_size, rank),
            (expert_num, rank, intermediate_size),
            (expert_num, hidden_size, rank),
        )
        for name, shape in zip(_FUSED_LORA_NAMES, shapes):
            contract[f"layers.{layer_idx}.experts.{name}"] = {
                "shape": list(shape),
                "dtype": "BF16",
            }
    return dict(sorted(contract.items()))


def _validate_fused_file(path: Path, contract: Mapping[str, Mapping[str, Any]]) -> None:
    if path.is_symlink() or not path.is_file():
        raise KTArtifactError(f"missing regular fused expert LoRA file: {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        expected = set(contract)
        if actual != expected:
            raise KTArtifactError(
                f"invalid fused LoRA keys: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        for key, expected_tensor in contract.items():
            tensor = handle.get_slice(key)
            if list(tensor.get_shape()) != expected_tensor["shape"]:
                raise KTArtifactError(f"{key} shape does not match the runtime")
            if tensor.get_dtype() != expected_tensor["dtype"]:
                raise KTArtifactError(f"{key} dtype does not match the runtime")


def _artifact_record(path: Path, contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise KTArtifactError(f"adapter artifact must be a regular file: {path}")
    record: dict[str, Any] = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
    if contract is not None:
        record.update({"tensor_count": len(contract), "tensors": dict(contract)})
    return record


def _base_model_name(model: Any, plan: KTPretrainedLoadPlan | None) -> str | None:
    if plan is not None:
        return plan.source_model_name_or_path
    queue = [model]
    visited: set[int] = set()
    while queue:
        candidate = queue.pop(0)
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        config = getattr(candidate, "config", None)
        value = getattr(config, "_name_or_path", None) or getattr(config, "name_or_path", None)
        if value:
            return os.fspath(value)
        for attribute in ("base_model", "model", "module"):
            child = getattr(candidate, attribute, None)
            if child is not None and child is not candidate:
                queue.append(child)
    return None


def _adapter_provenance(model: Any, plan: KTPretrainedLoadPlan | None) -> dict[str, Any]:
    wrappers = [wrapper for wrapper in _find_wrappers(model) if _wrapper_uses_fused_lora(wrapper)]
    ranks = {getattr(wrapper, "_lora_rank", None) for wrapper in wrappers}
    if len(ranks) != 1 or isinstance(next(iter(ranks), None), bool) or not isinstance(next(iter(ranks), None), int):
        raise KTArtifactError(f"fused wrappers have inconsistent LoRA ranks: {sorted(ranks, key=str)}")
    alphas = {
        getattr(wrapper, "_lora_alpha", getattr(getattr(wrapper, "wrapper", None), "lora_alpha", None))
        for wrapper in wrappers
    }
    alpha = next(iter(alphas), None)
    if (
        len(alphas) != 1
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
    ):
        raise KTArtifactError(f"fused wrappers have invalid or inconsistent LoRA alpha: {sorted(alphas, key=str)}")
    rank = next(iter(ranks))
    if plan is not None and plan.lora_rank is not None and plan.lora_rank != rank:
        raise KTArtifactError(
            f"fused wrapper LoRA rank {rank} does not match the pretrained plan rank {plan.lora_rank}"
        )
    if plan is not None and plan.lora_alpha is not None and float(plan.lora_alpha) != float(alpha):
        raise KTArtifactError(
            f"fused wrapper LoRA alpha {float(alpha)} does not match the pretrained plan alpha {plan.lora_alpha}"
        )
    expert_weight_format = _runtime_expert_weight_format(model, plan)
    if plan is not None and expert_weight_format != "int8":
        raise KTArtifactError(
            "a KT non-expert load plan requires INT8 routed expert provenance"
        )
    payload: dict[str, Any] = {
        "expert_weight_format": expert_weight_format,
        "base": {"model_name_or_path": _base_model_name(model, plan)},
        "lora": {"rank": rank, "alpha": float(alpha)},
    }
    if plan is not None:
        source = plan.manifest.get("source")
        if not isinstance(source, Mapping):
            raise KTArtifactError("validated non-expert manifest lost its source provenance")
        payload["base"]["fingerprint"] = _require_sha256(
            source.get("fingerprint"), "source.fingerprint", Path(plan.manifest_path)
        )
        cache_fingerprint = _require_sha256(
            plan.manifest.get("fingerprint"), "fingerprint", Path(plan.manifest_path)
        )
        routed_path = Path(plan.routed_manifest_path)
        payload["non_expert_cache"] = {
            "path": plan.weight_path,
            "fingerprint": cache_fingerprint,
        }
        payload["int8_experts"] = {
            "path": plan.routed_weight_path,
            "manifest": routed_path.name,
            "manifest_sha256": _sha256_file(routed_path),
            "fingerprint": plan.routed_manifest.get("fingerprint") or _sha256_file(routed_path),
        }
    return payload


def save_kt_adapter_artifacts(
    model: Any, output_dir: str | os.PathLike[str]
) -> KTAdapterManifest | None:
    """Atomically publish KT adapter files and a ready manifest."""

    from .lora import save_kt_moe_to_adapter

    contract = _fused_lora_contract(model)
    output = Path(output_dir).absolute()
    if output.is_symlink():
        raise KTArtifactError(f"adapter output must not be a symlink: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / KT_ADAPTER_MANIFEST_NAME
    # Invalidate an earlier ready marker before replacing bundle members.
    _invalidate_ready_manifest(manifest_path)
    if not contract:
        save_kt_moe_to_adapter(model, str(output))
        return None
    adapter_config = output / "adapter_config.json"
    standard = [name for name in _STANDARD_ADAPTER_NAMES if (output / name).is_file()]
    if adapter_config.is_symlink() or not adapter_config.is_file():
        raise KTArtifactError(f"KT adapter save requires {adapter_config}")
    if len(standard) != 1:
        raise KTArtifactError(f"KT adapter save requires exactly one standard adapter file, got {standard}")
    plan = _find_plan(model)
    if plan is not None:
        current_cache = _read_json(Path(plan.manifest_path), "KT non-expert cache manifest")
        current_routed = _read_json(Path(plan.routed_manifest_path), "routed INT8 manifest")
        if current_cache != dict(plan.manifest) or current_routed != dict(plan.routed_manifest):
            raise KTArtifactError("KT pretrained artifact provenance changed after model loading")

    staging = Path(tempfile.mkdtemp(dir=output, prefix=".kt-adapter-stage."))
    try:
        shutil.copy2(output / standard[0], staging / standard[0])
        save_kt_moe_to_adapter(model, str(staging))
        fused = staging / FUSED_EXPERT_LORA_NAME
        _validate_fused_file(fused, contract)
        produced = []
        for entry in staging.iterdir():
            if entry.is_symlink() or not entry.is_file():
                raise KTArtifactError(f"unsupported staged adapter entry: {entry}")
            with entry.open("rb") as handle:
                os.fsync(handle.fileno())
            produced.append(entry.name)
        if FUSED_EXPERT_LORA_NAME not in produced:
            raise KTArtifactError(f"staging did not produce {FUSED_EXPERT_LORA_NAME}")
        for name in sorted(produced):
            os.replace(staging / name, output / name)
        _fsync_directory(output)
        artifacts = {
            "adapter_config.json": _artifact_record(adapter_config),
        }
        for name in sorted(produced):
            artifacts[name] = _artifact_record(
                output / name, contract if name == FUSED_EXPERT_LORA_NAME else None
            )
        payload = {
            "version": KT_ADAPTER_MANIFEST_VERSION,
            "status": "ready",
            **_adapter_provenance(model, plan),
            "artifacts": artifacts,
        }
        _write_json_atomic(manifest_path, payload)
        return KTAdapterManifest(
            path=str(manifest_path),
            payload=payload,
            artifact_paths=tuple(str(output / name) for name in sorted(artifacts)),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_adapter_manifest(model: Any, adapter_path: Path) -> KTAdapterManifest:
    manifest_path = adapter_path / KT_ADAPTER_MANIFEST_NAME
    payload = _read_json(manifest_path, "KT adapter manifest")
    if payload.get("version") != KT_ADAPTER_MANIFEST_VERSION or payload.get("status") != "ready":
        raise KTArtifactError(f"{manifest_path}: expected version=1 and status='ready'")
    plan = _find_plan(model)
    expected_provenance = _adapter_provenance(model, plan)
    saved_format = payload.get("expert_weight_format")
    if (
        saved_format is None
        and "non_expert_cache" in payload
        and "int8_experts" in payload
    ):
        # LLaMA-Factory's legacy schema-v1 writer predated the explicit format
        # field; its INT8 provenance fields are unambiguous.
        saved_format = "int8"
    if saved_format != expected_provenance["expert_weight_format"]:
        raise KTArtifactError(f"{manifest_path}: expert_weight_format does not match the runtime")
    for field in ("base", "lora"):
        if payload.get(field) != expected_provenance[field]:
            raise KTArtifactError(f"{manifest_path}: {field} does not match the runtime")
    for field in ("non_expert_cache", "int8_experts"):
        if payload.get(field) != expected_provenance.get(field):
            raise KTArtifactError(f"{manifest_path}: {field} does not match the runtime")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or FUSED_EXPERT_LORA_NAME not in artifacts:
        raise KTArtifactError(f"{manifest_path}: artifacts must include fused expert LoRA")
    standard = {name for name in artifacts if name in _STANDARD_ADAPTER_NAMES}
    if len(standard) != 1 or "adapter_config.json" not in artifacts:
        raise KTArtifactError(f"{manifest_path}: invalid standard adapter inventory")
    paths = []
    for name, record in artifacts.items():
        if name != os.path.basename(name) or not isinstance(record, dict):
            raise KTArtifactError(f"{manifest_path}: invalid artifact record {name!r}")
        path = adapter_path / name
        if record.get("size") != (path.stat().st_size if path.is_file() and not path.is_symlink() else None):
            raise KTArtifactError(f"{manifest_path}: size mismatch for {name}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or _sha256_file(path) != digest:
            raise KTArtifactError(f"{manifest_path}: SHA256 mismatch for {name}")
        paths.append(str(path))
    contract = _fused_lora_contract(model)
    fused_record = artifacts[FUSED_EXPERT_LORA_NAME]
    if fused_record.get("tensor_count") != len(contract) or fused_record.get("tensors") != contract:
        raise KTArtifactError(f"{manifest_path}: fused tensor contract does not match the runtime")
    _validate_fused_file(adapter_path / FUSED_EXPERT_LORA_NAME, contract)
    return KTAdapterManifest(str(manifest_path), payload, tuple(sorted(paths)))


def load_kt_adapter_artifacts(
    model: Any, adapter_path: str | os.PathLike[str]
) -> KTAdapterManifest | None:
    """Restore KT state after PEFT load without allocating on non-owner ranks."""

    from .lora import kt_adapt_peft_lora, load_kt_moe_from_adapter

    root = _safe_root(adapter_path, "KT adapter")
    if any(getattr(wrapper, "wrapper", None) is not None for wrapper in _find_wrappers(model)):
        kt_adapt_peft_lora(model)
    contract = _fused_lora_contract(model)
    manifest_path = root / KT_ADAPTER_MANIFEST_NAME
    manifest = None
    if manifest_path.exists():
        manifest = _validate_adapter_manifest(model, root)
    elif contract:
        raise KTArtifactError(f"fused adapter is missing {manifest_path}")
    load_kt_moe_from_adapter(model, str(root))
    return manifest


__all__ = [
    "FUSED_EXPERT_LORA_NAME",
    "KT_ADAPTER_MANIFEST_NAME",
    "KT_NON_EXPERT_MANIFEST_NAME",
    "KTAdapterManifest",
    "KTArtifactError",
    "KTNonExpertCache",
    "KTPretrainedLoadPlan",
    "claim_kt_routed_expert_subtrees",
    "hide_kt_int8_routed_experts_from_dispatch",
    "hide_kt_routed_experts_from_dispatch",
    "is_kt_int8_routed_expert_base_parameter",
    "is_kt_routed_expert_base_parameter",
    "is_kt_routed_expert_parameter_name",
    "is_kt_supported_moe_model",
    "load_kt_adapter_artifacts",
    "mark_kt_int8_routed_expert_base_parameters",
    "prepare_kt_int8_non_expert_device_map",
    "prepare_kt_non_expert_device_map",
    "project_kt_int8_routed_experts_out_of_device_map",
    "project_kt_routed_experts_out_of_device_map",
    "resolve_kt_pretrained_artifacts",
    "save_kt_adapter_artifacts",
    "validate_kt_prequantized_loading_info",
    "validate_kt_pretrained_load",
    "write_kt_non_expert_cache_manifest",
]
