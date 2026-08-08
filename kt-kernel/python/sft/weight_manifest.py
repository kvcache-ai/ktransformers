# Persistent pre-quantized INT8 expert-weight manifest validation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable

from .backend import INT8_WEIGHT_LAYOUT

MANIFEST_NAME = "kt-weight-manifest.json"
LEGACY_MANIFEST_NAME = "kt-ephemeral-manifest.json"
SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SCHEMA_VERSION})
_KT_FILE_RE = re.compile(r"^INT8_(gate|up|down)_(\d+)_(\d+)Byte_(quant|scale)_\.kt$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidatedKTWeightManifest:
    root: Path
    path: Path
    schema_version: int
    layout: str
    layer_indices: tuple[int, ...]
    file_count: int
    size_bytes: int

    @property
    def is_legacy(self) -> bool:
        return self.schema_version == 1


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"INT8 manifest {name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"INT8 manifest {name} must be a non-negative integer")
    return int(value)


def _resolve_root(root: str | os.PathLike[str]) -> Path:
    path = Path(root)
    if not path.is_absolute():
        raise ValueError("persistent INT8 weight root must be an absolute path")
    if path.is_symlink():
        raise ValueError(f"persistent INT8 weight root must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"persistent INT8 weight root must be a directory: {resolved}")
    return resolved


def _load_manifest(root: Path) -> tuple[Path, dict[str, Any], int]:
    candidates = [
        path
        for path in (
            root / MANIFEST_NAME,
            root / LEGACY_MANIFEST_NAME,
        )
        if path.exists() or path.is_symlink()
    ]
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates) or "none"
        raise ValueError(
            "persistent INT8 weight root requires exactly one manifest "
            f"({MANIFEST_NAME} or legacy {LEGACY_MANIFEST_NAME}); found {names}"
        )
    path = candidates[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"INT8 manifest must be a regular file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read INT8 manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("INT8 manifest root must be a JSON object")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in _SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "unsupported INT8 manifest schema_version "
            f"{schema_version!r}; supported={sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if schema_version == 1 and path.name != LEGACY_MANIFEST_NAME:
        raise ValueError(f"schema-1 INT8 manifests must use {LEGACY_MANIFEST_NAME}")
    if schema_version == SCHEMA_VERSION and path.name != MANIFEST_NAME:
        raise ValueError(f"schema-{SCHEMA_VERSION} INT8 manifests must use {MANIFEST_NAME}")
    return path, manifest, int(schema_version)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_layer(
    root: Path,
    layer: dict[str, Any],
    *,
    schema_version: int,
    expected_numa_count: int,
    expected_expert_num: int,
    expected_hidden_size: int,
    expected_intermediate_size: int,
    verify_hashes: bool,
) -> tuple[int, int]:
    if not isinstance(layer, dict):
        raise ValueError("INT8 manifest layer entries must be JSON objects")
    layer_idx = _nonnegative_int(layer.get("index"), name="layer.index")
    if layer.get("state") != "ready":
        raise ValueError(f"persistent INT8 layer {layer_idx} must be in state 'ready'")
    numa_count = _positive_int(
        layer.get("numa_count"),
        name=f"layer[{layer_idx}].numa_count",
    )
    if numa_count != expected_numa_count:
        raise ValueError(
            f"persistent INT8 layer {layer_idx} NUMA count mismatch: "
            f"expected {expected_numa_count}, got {numa_count}"
        )

    layer_dir = root / f"_layer_{layer_idx}"
    if layer_dir.is_symlink() or not layer_dir.is_dir():
        raise ValueError(f"missing or unsafe persistent INT8 layer directory: {layer_dir}")
    expected_numa_dirs = {f"_numa_{numa_idx}" for numa_idx in range(expected_numa_count)}
    layer_entries = list(layer_dir.iterdir())
    if any(entry.is_symlink() or not entry.is_dir() for entry in layer_entries):
        raise ValueError(f"persistent INT8 layer {layer_idx} contains entries outside NUMA directories")
    actual_numa_dirs = {entry.name for entry in layer_entries}
    if actual_numa_dirs != expected_numa_dirs:
        raise ValueError(
            f"persistent INT8 layer {layer_idx} NUMA directories mismatch: "
            f"expected {sorted(expected_numa_dirs)}, got {sorted(actual_numa_dirs)}"
        )
    actual_file_sizes: dict[str, int] = {}
    for numa_idx in range(expected_numa_count):
        numa_dir = layer_dir / f"_numa_{numa_idx}"
        for entry in numa_dir.iterdir():
            entry_stat = entry.lstat()
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ValueError(f"unexpected non-regular persistent INT8 entry: {entry}")
            relative = entry.relative_to(root).as_posix()
            actual_file_sizes[relative] = int(entry_stat.st_size)

    files = layer.get("files")
    expected_file_count = expected_numa_count * expected_expert_num * 3 * 2
    if not isinstance(files, list) or len(files) != expected_file_count:
        raise ValueError(f"persistent INT8 layer {layer_idx} must list exactly " f"{expected_file_count} files")

    seen_paths: set[str] = set()
    seen_identities: set[tuple[int, str, int, str]] = set()
    actual_size = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"persistent INT8 layer {layer_idx} has a non-object file entry")
        relative = item.get("path")
        if not isinstance(relative, str) or relative in seen_paths:
            raise ValueError(f"persistent INT8 layer {layer_idx} has an invalid or duplicate path")
        seen_paths.add(relative)
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe INT8 manifest path: {relative!r}")
        if len(relative_path.parts) != 3:
            raise ValueError(f"unexpected INT8 manifest path depth: {relative!r}")
        expected_layer_dir = f"_layer_{layer_idx}"
        if relative_path.parts[0] != expected_layer_dir:
            raise ValueError(f"INT8 manifest path is outside layer {layer_idx}: {relative!r}")
        numa_match = re.fullmatch(r"_numa_(\d+)", relative_path.parts[1])
        if numa_match is None:
            raise ValueError(f"invalid INT8 NUMA path: {relative!r}")
        numa_idx = int(numa_match.group(1))
        if not 0 <= numa_idx < expected_numa_count:
            raise ValueError(f"out-of-range INT8 NUMA index in {relative!r}")
        filename_match = _KT_FILE_RE.fullmatch(relative_path.name)
        if filename_match is None:
            raise ValueError(f"unexpected INT8 .kt filename: {relative_path.name}")
        projection, expert_text, encoded_size_text, kind = filename_match.groups()
        expert_idx = int(expert_text)
        if not 0 <= expert_idx < expected_expert_num:
            raise ValueError(f"layer {layer_idx} has out-of-range expert {expert_idx}")
        identity = (numa_idx, projection, expert_idx, kind)
        if identity in seen_identities:
            raise ValueError(f"layer {layer_idx} has duplicate INT8 file identity {identity}")
        seen_identities.add(identity)

        absolute = root / relative_path
        if relative not in actual_file_sizes:
            raise ValueError(f"missing or unsafe INT8 manifest file: {absolute}")
        if not os.access(absolute, os.R_OK):
            raise PermissionError(f"INT8 weight file is not readable: {absolute}")
        expected_size = _positive_int(
            item.get("size_bytes"),
            name=f"{relative}.size_bytes",
        )
        intermediate_per_numa = expected_intermediate_size // expected_numa_count
        expected_projection_size = {
            ("gate", "quant"): expected_hidden_size * intermediate_per_numa,
            ("up", "quant"): expected_hidden_size * intermediate_per_numa,
            ("down", "quant"): expected_hidden_size * intermediate_per_numa,
            ("gate", "scale"): intermediate_per_numa * 4,
            ("up", "scale"): intermediate_per_numa * 4,
            ("down", "scale"): expected_hidden_size * 4,
        }[(projection, kind)]
        encoded_size = int(encoded_size_text)
        if encoded_size != expected_projection_size or expected_size != expected_projection_size:
            raise ValueError(
                f"INT8 size contract mismatch for {relative}: expected {expected_projection_size}, "
                f"filename encodes {encoded_size}, manifest declares {expected_size}"
            )
        observed_size = actual_file_sizes[relative]
        if observed_size != expected_size:
            raise ValueError(
                f"INT8 manifest size mismatch for {absolute}: " f"expected {expected_size}, got {observed_size}"
            )
        actual_size += observed_size

        expected_hash = item.get("sha256")
        if schema_version >= 2:
            if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
                raise ValueError(f"schema-{schema_version} INT8 file {relative} requires " "a lowercase SHA256 digest")
        if verify_hashes:
            if not isinstance(expected_hash, str):
                raise ValueError("full hash verification is unavailable for legacy schema-1 " f"file {relative}")
            actual_hash = _sha256(absolute)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"INT8 SHA256 mismatch for {absolute}: " f"expected {expected_hash}, got {actual_hash}"
                )

    expected_identities = {
        (numa_idx, projection, expert_idx, kind)
        for numa_idx in range(expected_numa_count)
        for projection in ("gate", "up", "down")
        for expert_idx in range(expected_expert_num)
        for kind in ("quant", "scale")
    }
    if seen_identities != expected_identities:
        missing = sorted(expected_identities - seen_identities)[:3]
        raise ValueError(f"persistent INT8 layer {layer_idx} file inventory is incomplete; " f"sample={missing}")

    actual_paths = set(actual_file_sizes)
    if actual_paths != seen_paths:
        extra = sorted(actual_paths - seen_paths)[:3]
        missing = sorted(seen_paths - actual_paths)[:3]
        raise ValueError(
            f"persistent INT8 layer {layer_idx} file set differs from manifest: " f"extra={extra}, missing={missing}"
        )
    manifest_size = _positive_int(
        layer.get("bytes"),
        name=f"layer[{layer_idx}].bytes",
    )
    if actual_size != manifest_size:
        raise ValueError(
            f"persistent INT8 layer {layer_idx} byte total mismatch: " f"expected {manifest_size}, got {actual_size}"
        )
    return len(files), actual_size


def validate_persistent_int8_weights(
    root: str | os.PathLike[str],
    *,
    layer_indices: Iterable[int],
    numa_count: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
    verify_hashes: bool | None = False,
) -> ValidatedKTWeightManifest:
    """Validate the complete persistent INT8 tree before any C++ load.

    ``verify_hashes=None`` selects the strongest verification supported by the
    manifest: schema-v2 files are hashed, while legacy schema-v1 files retain
    strict size, filename, inventory, and directory checks without requiring
    hashes that their producer never recorded.
    """

    expected_numa_count = _positive_int(
        int(numa_count),
        name="threadpool_count",
    )
    expected_expert_num = _positive_int(int(expert_num), name="expert_num")
    expected_hidden_size = _positive_int(int(hidden_size), name="hidden_size")
    expected_intermediate_size = _positive_int(
        int(intermediate_size),
        name="intermediate_size",
    )
    if expected_intermediate_size % expected_numa_count:
        raise ValueError(
            "persistent INT8 intermediate_size must be divisible by threadpool_count: "
            f"{expected_intermediate_size} % {expected_numa_count} != 0"
        )
    expected_layers = tuple(sorted(set(int(index) for index in layer_indices)))
    if not expected_layers or any(index < 0 for index in expected_layers):
        raise ValueError("persistent INT8 validation requires non-negative expert layer indices")

    resolved = _resolve_root(root)
    manifest_path, manifest, schema_version = _load_manifest(resolved)
    effective_verify_hashes = schema_version >= 2 if verify_hashes is None else bool(verify_hashes)
    expected_metadata = {
        "state": "ready",
        "expert_weight_format": "int8",
        "threadpool_count": expected_numa_count,
        "expert_num": expected_expert_num,
        "hidden_size": expected_hidden_size,
        "intermediate_size": expected_intermediate_size,
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"persistent INT8 manifest {key} mismatch: " f"expected {expected!r}, got {manifest.get(key)!r}"
            )

    if schema_version == 1:
        backend = str(manifest.get("backend", "")).strip().lower()
        if backend not in {"amxint8", "int8"}:
            raise ValueError("legacy schema-1 INT8 manifest backend must be AMXINT8 or INT8")
        layout = INT8_WEIGHT_LAYOUT
    else:
        layout = manifest.get("layout")
        if layout != INT8_WEIGHT_LAYOUT:
            raise ValueError(
                "persistent INT8 manifest layout mismatch: " f"expected {INT8_WEIGHT_LAYOUT!r}, got {layout!r}"
            )
        if "backend" in manifest and str(manifest["backend"]).strip().lower() not in {
            "auto",
            "int8",
        }:
            raise ValueError("schema-2 INT8 manifest backend must be hardware-neutral " "('INT8' or 'auto')")

    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise ValueError("persistent INT8 manifest layers must be a list")
    actual_layers: list[int] = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise ValueError("persistent INT8 manifest layers must contain objects")
        index = layer.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("persistent INT8 layer index must be non-negative")
        actual_layers.append(int(index))
    if len(set(actual_layers)) != len(actual_layers):
        raise ValueError("persistent INT8 manifest has duplicate layer indices")
    if tuple(sorted(actual_layers)) != expected_layers:
        raise ValueError(
            "persistent INT8 layer set mismatch: " f"expected {list(expected_layers)}, got {sorted(actual_layers)}"
        )

    file_count = 0
    size_bytes = 0
    for layer in layers:
        layer_files, layer_bytes = _validate_layer(
            resolved,
            layer,
            schema_version=schema_version,
            expected_numa_count=expected_numa_count,
            expected_expert_num=expected_expert_num,
            expected_hidden_size=expected_hidden_size,
            expected_intermediate_size=expected_intermediate_size,
            verify_hashes=effective_verify_hashes,
        )
        file_count += layer_files
        size_bytes += layer_bytes

    manifest_bytes = _positive_int(manifest.get("bytes"), name="bytes")
    if size_bytes != manifest_bytes:
        raise ValueError(
            "persistent INT8 manifest byte total mismatch: " f"expected {manifest_bytes}, got {size_bytes}"
        )
    expected_root_entries = {
        manifest_path.name,
        *(f"_layer_{index}" for index in expected_layers),
    }
    root_entries = list(resolved.iterdir())
    if any(entry.is_symlink() for entry in root_entries):
        raise ValueError("persistent INT8 root must not contain symlinks")
    actual_root_entries = {entry.name for entry in root_entries}
    if actual_root_entries != expected_root_entries:
        raise ValueError(
            "persistent INT8 root entries differ from its manifest: "
            f"extra={sorted(actual_root_entries - expected_root_entries)}, "
            f"missing={sorted(expected_root_entries - actual_root_entries)}"
        )

    return ValidatedKTWeightManifest(
        root=resolved,
        path=manifest_path,
        schema_version=schema_version,
        layout=layout,
        layer_indices=expected_layers,
        file_count=file_count,
        size_bytes=size_bytes,
    )
