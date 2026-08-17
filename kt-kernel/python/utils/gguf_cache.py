"""
GGUF -> AMXINT8 quantized cache manager.

First boot with ``--kt-weight-path <gguf-dir> --kt-method AMXINT8`` dequantizes
the MoE expert tensors straight from the mmap'd GGUF blocks in C++ and writes
the standard INT8 ``.kt`` cache as a side effect. Every later boot loads the
same files (``load=True``, the untouched fast path). The cache is keyed so a
swapped model, a different method, a different NUMA split, or a differently
built binary can never silently load the wrong bytes.

Layout (the ``.kt`` format itself is unchanged):

    <root>/<key>/manifest.json
    <root>/<key>/_layer_0/_numa_0/INT8_gate_0_..._quant_.kt   (C++ save path)
    ...

Root resolution order:
    1. ``KT_GGUF_CACHE_DIR`` (explicit relocation)
    2. ``<kt-weight-path>/.kt_cache`` if the GGUF dir is writable
    3. ``~/.cache/kt-kernel/gguf/``

Env:
    ``KT_GGUF_CACHE=0``       disable the cache entirely (quantize every boot)
    ``KT_GGUF_CACHE=refresh`` ignore any existing cache and rebuild
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Bump whenever GemmKernel224Int8::BufferB's packed tile layout or scale
# format changes — the .kt payload is the packed tile layout, and loading it
# on a differently-built binary would be silent garbage.
PACK_FORMAT_VERSION = 1


def _host_cpu_flags() -> List[str]:
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("flags"):
                    return line.split(":", 1)[1].strip().split()
    except OSError:
        pass
    return []


def _isa_tag() -> str:
    """Compile+runtime ISA tag for the cache key (payload layout depends on it)."""
    flags = set(_host_cpu_flags())
    parts = []
    if "amx_tile" in flags and "amx_int8" in flags:
        parts.append("amx")
    if "avx512_bf16" in flags:
        parts.append("avx512bf16")
    if "avx512_vnni" in flags:
        parts.append("avx512vnni")
    if "avx512bw" in flags:
        parts.append("avx512bw")
    elif "avx512f" in flags:
        parts.append("avx512f")
    return ",".join(parts) if parts else "scalar"


def resolve_cache_root(gguf_dir: str) -> str:
    """Resolve the cache root per the design's precedence order."""
    env = os.getenv("KT_GGUF_CACHE_DIR", "").strip()
    if env:
        return env
    if os.path.isdir(gguf_dir) and os.access(gguf_dir, os.W_OK):
        return os.path.join(gguf_dir, ".kt_cache")
    return os.path.join(os.path.expanduser("~"), ".cache", "kt-kernel", "gguf")


def _gguf_fingerprint(gguf_dir: str, loader) -> str:
    """Stable fingerprint of the GGUF files: general.uuid when present, else
    sorted (filename, size, mtime_ns)."""
    uuid = loader.metadata.get("general.uuid")
    if uuid is not None:
        return f"uuid:{uuid}"
    entries = []
    if os.path.isfile(gguf_dir) and gguf_dir.endswith(".gguf"):
        paths = [gguf_dir]
    else:
        paths = [os.path.join(gguf_dir, f) for f in sorted(os.listdir(gguf_dir)) if f.endswith(".gguf")]
    for p in paths:
        try:
            st = os.stat(p)
            entries.append((os.path.basename(p), st.st_size, st.st_mtime_ns))
        except OSError:
            continue
    if not entries:
        raise FileNotFoundError(f"No .gguf files found under {gguf_dir}")
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]


class GGUFCacheManager:
    """Per-model INT8 cache handle shared by all AMXMoEWrapper layers."""

    def __init__(
        self,
        gguf_dir: str,
        loader,
        method: str,
        threadpool_count: int,
        hidden_size: int,
        moe_intermediate_size: int,
        expert_num: int,
    ):
        self.gguf_dir = gguf_dir
        self.method = method
        self.threadpool_count = threadpool_count

        mode = os.getenv("KT_GGUF_CACHE", "").strip().lower()
        self.enabled = mode != "0"
        self.refresh = mode == "refresh"
        if not self.enabled:
            logger.info("[GGUFCache] KT_GGUF_CACHE=0 — quantizing from GGUF every boot, no disk cache")
            self.root = ""
            self.key = ""
            self.cache_dir = ""
            self.manifest_path = ""
            self.manifest: Dict = {}
            self.valid = False
            return

        fingerprint = _gguf_fingerprint(gguf_dir, loader)
        key_fields = {
            "gguf_fingerprint": fingerprint,
            "method": method,
            "threadpool_count": threadpool_count,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
            "expert_num": expert_num,
            "pack_format_version": PACK_FORMAT_VERSION,
            "isa": _isa_tag(),
        }
        self.key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
        self.root = resolve_cache_root(gguf_dir)
        self.cache_dir = os.path.join(self.root, f"{method.lower()}-{self.key}")
        self.manifest_path = os.path.join(self.cache_dir, "manifest.json")
        self.manifest = self._load_manifest()
        # Key fields are merged into the manifest only at write time (see
        # mark_layer_complete) so a freshly read-back manifest compares
        # faithfully against key_fields — tampering with a key field must
        # invalidate rather than being silently overwritten.
        self.key_fields = key_fields
        self.valid = self._validate(key_fields)
        if self.refresh:
            logger.info(
                "[GGUFCache] KT_GGUF_CACHE=refresh — ignoring existing cache at %s", self.cache_dir
            )
            self.valid = False
        if self.valid:
            logger.info(
                "[GGUFCache] cache hit: %s (layers %d..%d complete)",
                self.cache_dir,
                min(self.manifest.get("layers_complete", [-1])),
                max(self.manifest.get("layers_complete", [-1])),
            )
        else:
            logger.info("[GGUFCache] cache miss, will rebuild: %s", self.cache_dir)

    def _load_manifest(self) -> Dict:
        try:
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _validate(self, key_fields: Dict) -> bool:
        if not self.manifest:
            return False
        for field, value in key_fields.items():
            if self.manifest.get(field) != value:
                logger.warning(
                    "[GGUFCache] manifest field %s mismatch (%r != %r) — rebuilding",
                    field, self.manifest.get(field), value,
                )
                return False
        if not isinstance(self.manifest.get("layers_complete"), list):
            return False
        return True

    def layer_complete(self, layer_idx: int) -> bool:
        """True when the manifest is valid and this layer's files are on disk."""
        if not (self.enabled and self.valid):
            return False
        if layer_idx not in self.manifest.get("layers_complete", []):
            return False
        layer_dir = os.path.join(self.cache_dir, f"_layer_{layer_idx}")
        if not os.path.isdir(layer_dir):
            logger.warning("[GGUFCache] layer %d marked complete but %s missing — rebuilding", layer_idx, layer_dir)
            return False
        return True

    def mark_layer_complete(self, layer_idx: int) -> None:
        """Record a fully-saved layer in the manifest (atomic write)."""
        if not self.enabled:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        layers = list(self.manifest.get("layers_complete", []))
        if layer_idx not in layers:
            layers.append(layer_idx)
            layers.sort()
        self.manifest["layers_complete"] = layers
        self.manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if "created" not in self.manifest:
            self.manifest["created"] = self.manifest["updated"]
        # Persist the key fields with every write so a later boot can validate.
        self.manifest.update(self.key_fields)
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)
        self.valid = True