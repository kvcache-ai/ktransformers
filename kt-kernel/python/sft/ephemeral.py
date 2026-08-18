# Ephemeral pre-quantized expert-weight lifecycle
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import re
import signal
import stat
import time
from typing import Any, Iterable


MANIFEST_NAME = "kt-ephemeral-manifest.json"
STAGING_LEASE_NAME = "kt-ephemeral-staging-lease.json"
SCHEMA_VERSION = 1
_EPHEMERAL_BASE = Path("/dev/shm")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$")
_KT_FILE_RE = re.compile(r"^INT8_(gate|up|down)_(\d+)_.*Byte_(quant|scale)_\.kt$")


def _validated_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "ephemeral INT8 run_id must be 6-128 characters containing only "
            "letters, digits, '.', '_' or '-'"
        )
    return run_id


def validate_ephemeral_run_id(run_id: str) -> str:
    """Validate a run id before it is used to construct a /dev/shm path."""
    return _validated_run_id(run_id)


def _validate_owned_root(
    root: str | os.PathLike[str],
    *,
    run_id: str,
    staging: bool,
) -> Path:
    run_id = _validated_run_id(run_id)
    base = _EPHEMERAL_BASE.resolve(strict=True)
    path = Path(root)
    if not path.is_absolute():
        raise ValueError("ephemeral INT8 weight root must be an absolute path")
    expected_name = f"kt-int8-{run_id}" + (".staging" if staging else "")
    if path.name != expected_name:
        raise ValueError(
            f"ephemeral INT8 weight root must be named {expected_name!r}, got {path.name!r}"
        )
    if path.is_symlink():
        raise ValueError(f"ephemeral INT8 weight root must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if resolved.parent != base:
        raise ValueError(
            f"ephemeral INT8 weight root must be a direct child of {base}, got {resolved}"
        )
    root_stat = resolved.stat()
    if root_stat.st_uid != os.getuid():
        raise PermissionError(
            f"ephemeral INT8 weight root is owned by uid {root_stat.st_uid}, "
            f"expected {os.getuid()}"
        )
    if root_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(
            "ephemeral INT8 weight root must not be group- or world-writable"
        )
    return resolved


def _atomic_write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    target = root / MANIFEST_NAME
    temporary = root / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _scan_layer(
    root: Path,
    layer_idx: int,
    *,
    numa_count: int,
    expert_num: int,
) -> dict[str, Any]:
    layer_dir = root / f"_layer_{layer_idx}"
    if not layer_dir.is_dir() or layer_dir.is_symlink():
        raise ValueError(f"missing or unsafe INT8 layer directory: {layer_dir}")

    files: list[dict[str, Any]] = []
    observed: set[tuple[int, str, int, str]] = set()
    expected_dirs = {f"_numa_{numa_idx}" for numa_idx in range(numa_count)}
    layer_entries = list(layer_dir.iterdir())
    if any(not entry.is_dir() for entry in layer_entries):
        raise ValueError(f"layer {layer_idx} contains entries outside NUMA directories")
    actual_dirs = {entry.name for entry in layer_entries}
    if actual_dirs != expected_dirs:
        raise ValueError(
            f"layer {layer_idx} NUMA directories mismatch: "
            f"expected {sorted(expected_dirs)}, got {sorted(actual_dirs)}"
        )

    for numa_idx in range(numa_count):
        numa_dir = layer_dir / f"_numa_{numa_idx}"
        if numa_dir.is_symlink():
            raise ValueError(f"NUMA directory must not be a symlink: {numa_dir}")
        for entry in numa_dir.iterdir():
            if entry.is_symlink() or not entry.is_file():
                raise ValueError(f"unexpected non-regular INT8 weight entry: {entry}")
            match = _KT_FILE_RE.fullmatch(entry.name)
            if match is None:
                raise ValueError(f"unexpected INT8 .kt filename: {entry.name}")
            projection, expert_text, kind = match.groups()
            expert_idx = int(expert_text)
            if not 0 <= expert_idx < expert_num:
                raise ValueError(
                    f"layer {layer_idx} has out-of-range expert {expert_idx}"
                )
            identity = (numa_idx, projection, expert_idx, kind)
            if identity in observed:
                raise ValueError(
                    f"layer {layer_idx} has duplicate INT8 file for {identity}"
                )
            observed.add(identity)
            size_bytes = entry.stat().st_size
            if size_bytes <= 0:
                raise ValueError(f"INT8 .kt file is empty: {entry}")
            files.append(
                {
                    "path": entry.relative_to(root).as_posix(),
                    "size_bytes": size_bytes,
                }
            )

    expected = {
        (numa_idx, projection, expert_idx, kind)
        for numa_idx in range(numa_count)
        for projection in ("gate", "up", "down")
        for expert_idx in range(expert_num)
        for kind in ("quant", "scale")
    }
    missing = expected - observed
    if missing:
        sample = sorted(missing)[:3]
        raise ValueError(
            f"layer {layer_idx} is missing {len(missing)} INT8 files; sample={sample}"
        )
    return {
        "index": layer_idx,
        "numa_count": numa_count,
        "state": "ready",
        "files": sorted(files, key=lambda item: item["path"]),
        "bytes": sum(item["size_bytes"] for item in files),
    }


def publish_ephemeral_int8_weights(
    staging_root: str | os.PathLike[str],
    *,
    run_id: str,
    layer_indices: Iterable[int],
    numa_count: int,
    expert_num: int,
    hidden_size: int,
    intermediate_size: int,
) -> Path:
    """Validate a complete staging tree, publish its manifest, then rename it."""
    if numa_count <= 0 or expert_num <= 0 or hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError(
            "ephemeral INT8 model dimensions and NUMA count must be positive"
        )
    staging = _validate_owned_root(staging_root, run_id=run_id, staging=True)
    lease_path = staging / STAGING_LEASE_NAME
    if lease_path.exists() or lease_path.is_symlink():
        if lease_path.is_symlink() or not lease_path.is_file():
            raise ValueError(f"unsafe ephemeral staging lease: {lease_path}")
        with lease_path.open("r", encoding="utf-8") as handle:
            lease = json.load(handle)
        expected_lease = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "owner_uid": os.getuid(),
        }
        for key, expected in expected_lease.items():
            if lease.get(key) != expected:
                raise ValueError(
                    f"ephemeral staging lease {key} mismatch: "
                    f"expected {expected!r}, got {lease.get(key)!r}"
                )
        lease_path.unlink()
    unique_layers = sorted(set(int(index) for index in layer_indices))
    expected_root_entries = {f"_layer_{index}" for index in unique_layers}
    actual_root_entries = {entry.name for entry in staging.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ValueError(
            "ephemeral INT8 staging root entries mismatch: "
            f"expected {sorted(expected_root_entries)}, got {sorted(actual_root_entries)}"
        )
    layers = [
        _scan_layer(
            staging,
            int(layer_idx),
            numa_count=int(numa_count),
            expert_num=int(expert_num),
        )
        for layer_idx in unique_layers
    ]
    if not layers:
        raise ValueError("ephemeral INT8 manifest requires at least one expert layer")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": "ready",
        "run_id": run_id,
        "owner_uid": os.getuid(),
        "producer_pid": os.getpid(),
        "created_unix_ns": time.time_ns(),
        "expert_weight_format": "int8",
        "backend": "AMXINT8",
        "threadpool_count": int(numa_count),
        "expert_num": int(expert_num),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
        "layers": layers,
        "bytes": sum(layer["bytes"] for layer in layers),
    }
    _atomic_write_manifest(staging, manifest)

    ready = staging.with_name(f"kt-int8-{run_id}")
    if ready.exists() or ready.is_symlink():
        raise FileExistsError(f"ephemeral INT8 ready root already exists: {ready}")
    os.replace(staging, ready)
    return ready


def cleanup_ephemeral_int8_staging(
    staging_root: str | os.PathLike[str],
    *,
    run_id: str,
) -> None:
    """Remove only a validated, caller-owned staging root without following links."""
    root = _validate_owned_root(staging_root, run_id=run_id, staging=True)
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        current_path = Path(current)
        for filename in files:
            entry = current_path / filename
            entry.unlink()
        for dirname in directories:
            entry = current_path / dirname
            if entry.is_symlink():
                entry.unlink()
            else:
                entry.rmdir()
    root.rmdir()


def write_ephemeral_staging_lease(
    staging_root: str | os.PathLike[str],
    *,
    run_id: str,
    producer_pid: int,
) -> None:
    """Record the live producer so an orphaned staging tree can be reclaimed safely."""
    if int(producer_pid) <= 0:
        raise ValueError(f"producer_pid must be positive, got {producer_pid}")
    root = _validate_owned_root(staging_root, run_id=run_id, staging=True)
    lease = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "owner_uid": os.getuid(),
        "producer_pid": int(producer_pid),
    }
    target = root / STAGING_LEASE_NAME
    temporary = root / f".{STAGING_LEASE_NAME}.{os.getpid()}.tmp"
    payload = json.dumps(lease, sort_keys=True, separators=(",", ":"))
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def reclaim_stale_ephemeral_int8_staging(
    staging_root: str | os.PathLike[str],
    *,
    run_id: str,
) -> bool:
    """Remove a strict, owned staging tree only when its recorded producer is gone."""
    root = _validate_owned_root(staging_root, run_id=run_id, staging=True)
    lease_path = root / STAGING_LEASE_NAME
    if lease_path.is_symlink() or not lease_path.is_file():
        raise RuntimeError(
            f"refusing to reclaim staging root without a regular {STAGING_LEASE_NAME}: {root}"
        )
    with lease_path.open("r", encoding="utf-8") as handle:
        lease = json.load(handle)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "owner_uid": os.getuid(),
    }
    for key, value in expected.items():
        if lease.get(key) != value:
            raise RuntimeError(
                f"refusing to reclaim staging root with mismatched {key}: "
                f"expected {value!r}, got {lease.get(key)!r}"
            )
    producer_pid = int(lease.get("producer_pid", 0))
    if producer_pid <= 0:
        raise RuntimeError(
            "refusing to reclaim staging root with an invalid producer_pid"
        )
    if Path(f"/proc/{producer_pid}").exists():
        return False
    cleanup_ephemeral_int8_staging(root, run_id=run_id)
    return True


class EphemeralKTWeightStore:
    """Manifest-guarded, exact-file cleanup for one ephemeral INT8 run."""

    def __init__(self, root: Path, manifest: dict[str, Any]):
        self.root = root
        self.manifest = manifest
        self._closed = False
        self._previous_handlers: dict[signal.Signals, Any] = {}
        atexit.register(self.cleanup)
        self._install_signal_handlers()

    @classmethod
    def open(
        cls,
        root: str | os.PathLike[str],
        *,
        layer_indices: Iterable[int],
        numa_count: int,
        expert_num: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> "EphemeralKTWeightStore":
        manifest_path = Path(root) / MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"ephemeral INT8 weight root requires {MANIFEST_NAME}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        run_id = str(manifest.get("run_id", ""))
        resolved = _validate_owned_root(root, run_id=run_id, staging=False)
        expected_metadata = {
            "schema_version": SCHEMA_VERSION,
            "state": "ready",
            "owner_uid": os.getuid(),
            "expert_weight_format": "int8",
            "backend": "AMXINT8",
            "threadpool_count": int(numa_count),
            "expert_num": int(expert_num),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
        }
        for key, expected in expected_metadata.items():
            if manifest.get(key) != expected:
                raise ValueError(
                    f"ephemeral INT8 manifest {key} mismatch: "
                    f"expected {expected!r}, got {manifest.get(key)!r}"
                )

        expected_layers = sorted(set(int(index) for index in layer_indices))
        layers = manifest.get("layers")
        if not isinstance(layers, list):
            raise ValueError("ephemeral INT8 manifest layers must be a list")
        actual_layers = sorted(layer.get("index") for layer in layers)
        if actual_layers != expected_layers:
            raise ValueError(
                f"ephemeral INT8 layer set mismatch: "
                f"expected {expected_layers}, got {actual_layers}"
            )
        for layer in layers:
            if layer.get("state") != "ready":
                raise ValueError(
                    "ephemeral INT8 runs cannot be resumed after partial consumption"
                )
            cls._validate_manifest_layer(resolved, layer)
        expected_root_entries = {
            MANIFEST_NAME,
            *(f"_layer_{index}" for index in expected_layers),
        }
        actual_root_entries = {entry.name for entry in resolved.iterdir()}
        if actual_root_entries != expected_root_entries:
            raise ValueError(
                "ephemeral INT8 root contains files outside its manifest: "
                f"{sorted(actual_root_entries - expected_root_entries)}"
            )
        return cls(resolved, manifest)

    @staticmethod
    def _validate_manifest_layer(root: Path, layer: dict[str, Any]) -> None:
        layer_idx = int(layer["index"])
        files = layer.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"layer {layer_idx} manifest has no files")
        actual_bytes = 0
        seen: set[str] = set()
        for item in files:
            relative = item.get("path")
            if not isinstance(relative, str) or relative in seen:
                raise ValueError(
                    f"layer {layer_idx} has invalid or duplicate file entry"
                )
            seen.add(relative)
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"unsafe manifest path: {relative!r}")
            if relative_path.parts[:1] != (f"_layer_{layer_idx}",):
                raise ValueError(
                    f"manifest path is outside layer {layer_idx}: {relative!r}"
                )
            absolute = root / relative_path
            if absolute.is_symlink() or not absolute.is_file():
                raise ValueError(f"missing or unsafe manifest file: {absolute}")
            expected_size = int(item.get("size_bytes", -1))
            actual_size = absolute.stat().st_size
            if expected_size <= 0 or actual_size != expected_size:
                raise ValueError(
                    f"manifest size mismatch for {absolute}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            actual_bytes += actual_size
        layer_dir = root / f"_layer_{layer_idx}"
        actual_files = {
            entry.relative_to(root).as_posix()
            for current, directories, filenames in os.walk(layer_dir, followlinks=False)
            for entry in [Path(current) / filename for filename in filenames]
        }
        if actual_files != seen:
            raise ValueError(f"layer {layer_idx} file set differs from its manifest")
        if actual_bytes != int(layer.get("bytes", -1)):
            raise ValueError(
                f"layer {layer_idx} byte total mismatch: "
                f"expected {layer.get('bytes')}, got {actual_bytes}"
            )

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous = signal.getsignal(signum)
            self._previous_handlers[signum] = previous

            def _handler(received, frame, *, _previous=previous):
                self.cleanup()
                if callable(_previous):
                    _previous(received, frame)
                elif _previous == signal.SIG_IGN:
                    return
                else:
                    raise SystemExit(128 + received)

            signal.signal(signum, _handler)

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._previous_handlers.items():
            signal.signal(signum, previous)
        self._previous_handlers.clear()

    def _layer(self, layer_idx: int) -> dict[str, Any]:
        for layer in self.manifest["layers"]:
            if int(layer["index"]) == int(layer_idx):
                return layer
        raise KeyError(f"layer {layer_idx} is not listed in the ephemeral manifest")

    def consume_layer(self, layer_idx: int) -> None:
        """Delete exactly one manifest-listed layer after synchronous C++ load."""
        if self._closed:
            raise RuntimeError("ephemeral INT8 store is already closed")
        layer = self._layer(layer_idx)
        if layer["state"] != "ready":
            raise RuntimeError(
                f"ephemeral INT8 layer {layer_idx} is in state {layer['state']!r}"
            )
        self._validate_manifest_layer(self.root, layer)
        layer["state"] = "consuming"
        _atomic_write_manifest(self.root, self.manifest)
        for item in layer["files"]:
            path = self.root / item["path"]
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"refusing to delete unsafe ephemeral file: {path}")
            path.unlink()
        layer_dir = self.root / f"_layer_{layer_idx}"
        for numa_idx in reversed(range(int(layer["numa_count"]))):
            (layer_dir / f"_numa_{numa_idx}").rmdir()
        layer_dir.rmdir()
        layer["state"] = "consumed"
        _atomic_write_manifest(self.root, self.manifest)

    def finish(self) -> None:
        if any(layer["state"] != "consumed" for layer in self.manifest["layers"]):
            raise RuntimeError(
                "cannot finish ephemeral INT8 store with unconsumed layers"
            )
        (self.root / MANIFEST_NAME).unlink()
        self.root.rmdir()
        self._closed = True
        self._restore_signal_handlers()
        atexit.unregister(self.cleanup)

    def cleanup(self) -> None:
        """Best-effort cleanup restricted to files enumerated by this manifest."""
        if self._closed:
            return
        self._closed = True
        try:
            for layer in self.manifest.get("layers", []):
                if layer.get("state") == "consumed":
                    continue
                for item in layer.get("files", []):
                    relative = Path(str(item.get("path", "")))
                    if relative.is_absolute() or ".." in relative.parts:
                        continue
                    path = self.root / relative
                    if path.is_file() and not path.is_symlink():
                        path.unlink()
                layer_dir = self.root / f"_layer_{int(layer['index'])}"
                for numa_idx in reversed(range(int(layer.get("numa_count", 0)))):
                    numa_dir = layer_dir / f"_numa_{numa_idx}"
                    if numa_dir.is_dir() and not numa_dir.is_symlink():
                        try:
                            numa_dir.rmdir()
                        except OSError:
                            pass
                if layer_dir.is_dir() and not layer_dir.is_symlink():
                    try:
                        layer_dir.rmdir()
                    except OSError:
                        pass
            for manifest_name in (
                MANIFEST_NAME,
                f".{MANIFEST_NAME}.{os.getpid()}.tmp",
            ):
                path = self.root / manifest_name
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            try:
                self.root.rmdir()
            except OSError:
                pass
        finally:
            self._restore_signal_handlers()
            atexit.unregister(self.cleanup)
