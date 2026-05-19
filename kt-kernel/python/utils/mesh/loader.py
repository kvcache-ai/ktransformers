from __future__ import annotations

import json
import hashlib
import os
import re
import struct

import numpy as np


def index_safetensor_file(loader, file_path: str):
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    data_base_offset = 8 + header_size
    for key, info in header.items():
        if key == "__metadata__":
            continue
        loader.tensor_info_map[key] = {
            "file_path": file_path,
            "dtype": info["dtype"],
            "shape": tuple(info["shape"]),
            "offset": data_base_offset + int(info["data_offsets"][0]),
            "size": int(info["data_offsets"][1]) - int(info["data_offsets"][0]),
        }


def get_dense_moe_layout(loader, base_key: str) -> tuple[list[int], list[int]]:
    up_base_key = re.escape(f"{base_key}.ffn_up_exps")
    pattern = re.compile(rf"^{up_base_key}\.(\d+)\.numa\.(\d+)\.weight$")
    expert_ids = set()
    numa_ids = set()
    for key in loader.tensor_file_map:
        match = pattern.match(key)
        if match is None:
            continue
        expert_ids.add(int(match.group(1)))
        numa_ids.add(int(match.group(2)))

    if not expert_ids:
        raise ValueError(f"No experts found for key {base_key}")

    sorted_expert_ids = sorted(expert_ids)
    sorted_numa_ids = sorted(numa_ids)
    if sorted_expert_ids != list(range(len(sorted_expert_ids))):
        raise ValueError(f"Expert ids for {base_key} must be dense and start at 0, got {sorted_expert_ids}")
    if sorted_numa_ids != list(range(len(sorted_numa_ids))):
        raise ValueError(f"NUMA ids for {base_key} must be dense and start at 0, got {sorted_numa_ids}")
    return sorted_numa_ids, sorted_expert_ids


def get_file_fd(loader, file_path: str, use_direct_io: bool = True) -> int:
    fd_key = (file_path, bool(use_direct_io))
    fd = loader.file_fd_map.get(fd_key)
    if fd is not None:
        return fd

    flags = os.O_RDONLY
    if use_direct_io and hasattr(os, "O_DIRECT"):
        flags |= os.O_DIRECT

    fd = os.open(file_path, flags)
    loader.file_fd_map[fd_key] = fd
    return fd


def get_file_slot(loader, key: str, use_direct_io: bool = True) -> tuple[int, int, int]:
    if key not in loader.tensor_info_map:
        raise KeyError(f"Key {key} not found in Safetensor files")
    info = loader.tensor_info_map[key]
    fd = get_file_fd(loader, info["file_path"], use_direct_io=use_direct_io)
    return fd, int(info["offset"]), int(info["size"])


def _is_false_env(value: str | None) -> bool:
    return value is not None and value.strip() in ("0", "false", "False", "FALSE", "no", "No", "NO")


def _bf16_cache_root() -> str:
    explicit = os.environ.get("KT_MESH_BF16_EXPERT_CACHE_DIR") or os.environ.get("KT_MESH_EXPERT_CACHE_DIR")
    if explicit:
        return explicit
    for start in (os.path.abspath(os.path.dirname(__file__)), os.getcwd()):
        current = start
        while True:
            if os.path.isdir(os.path.join(current, "kt-kernel")):
                return os.path.join(current, ".cache")
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return (
        os.path.join(os.getcwd(), ".cache")
    )


def _bf16_cache_enabled(loader) -> bool:
    raw = os.environ.get("KT_MESH_BF16_EXPERT_CACHE", os.environ.get("KT_MESH_EXPERT_CACHE", "1"))
    # Non-packed BF16 io_uring needs a normalized cache layout. Packed Qwen can
    # read the original safetensors directly, so only that path may opt out.
    return not _is_false_env(raw) or loader._detected_format != "packed"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _tensor_nbytes(info: dict) -> int:
    size = 1
    for dim in info["shape"]:
        size *= int(dim)
    if info["dtype"] not in ("BF16", "U16"):
        raise ValueError(f"BF16 io_uring expected BF16 tensor data, got dtype={info['dtype']}")
    expected = size * np.dtype(np.uint16).itemsize
    if expected != int(info["size"]):
        raise ValueError(
            f"BF16 io_uring tensor size mismatch for {info['file_path']}: "
            f"shape={info['shape']} dtype={info['dtype']} expected={expected} actual={info['size']}"
        )
    return expected


def _copy_file_slice(src_handles: dict[str, object], src_path: str, offset: int, size: int, dst) -> None:
    src = src_handles.get(src_path)
    if src is None:
        src = open(src_path, "rb")
        src_handles[src_path] = src
    src.seek(offset)
    remaining = size
    chunk_size = 8 * 1024 * 1024
    while remaining > 0:
        data = src.read(min(chunk_size, remaining))
        if not data:
            raise EOFError(f"Unexpected EOF while reading {src_path} at offset={offset} size={size}")
        dst.write(data)
        remaining -= len(data)


def _bf16_find_unpacked_experts(loader, base_key: str) -> tuple[str, str, str, str, int]:
    experts_prefix_candidates = bf16_get_experts_prefix_candidates(loader, base_key)
    gate_name, up_name, down_name = bf16_get_proj_names(loader)
    for prefix in experts_prefix_candidates:
        expert_count = 0
        while loader.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
            expert_count += 1
        if expert_count > 0:
            return prefix, gate_name, up_name, down_name, expert_count
    raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")


def _bf16_build_iouring_layout(loader, base_key: str) -> dict:
    if loader._detected_format == "packed":
        experts_prefix = bf16_resolve_packed_experts_prefix(loader, base_key)
        gate_up_key = f"{experts_prefix}.gate_up_proj"
        down_key = f"{experts_prefix}.down_proj"
        gate_up_info = loader.tensor_info_map[gate_up_key]
        down_info = loader.tensor_info_map[down_key]
        gate_up_shape = tuple(gate_up_info["shape"])
        down_shape = tuple(down_info["shape"])
        if len(gate_up_shape) != 3 or len(down_shape) != 3:
            raise ValueError(f"Unexpected packed BF16 expert shapes: gate_up={gate_up_shape}, down={down_shape}")

        expert_count, gate_up_rows, hidden_size = gate_up_shape
        down_expert_count, down_hidden_size, intermediate_size = down_shape
        if down_expert_count != expert_count or down_hidden_size != hidden_size:
            raise ValueError(f"Mismatched packed BF16 expert shapes: gate_up={gate_up_shape}, down={down_shape}")
        if gate_up_rows != 2 * intermediate_size:
            raise ValueError(f"Expected gate_up second dim to be 2*I, got gate_up={gate_up_shape}, down={down_shape}")
        _tensor_nbytes(gate_up_info)
        _tensor_nbytes(down_info)
        return {
            "format": "packed",
            "base_key": base_key,
            "expert_count": int(expert_count),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(intermediate_size),
            "gate_key": gate_up_key,
            "up_key": gate_up_key,
            "down_key": down_key,
            "gate_up_info": gate_up_info,
            "down_info": down_info,
        }

    prefix, gate_name, up_name, down_name, expert_count = _bf16_find_unpacked_experts(loader, base_key)
    gate0 = loader.tensor_info_map[f"{prefix}.0.{gate_name}.weight"]
    up0 = loader.tensor_info_map[f"{prefix}.0.{up_name}.weight"]
    down0 = loader.tensor_info_map[f"{prefix}.0.{down_name}.weight"]
    gate_shape = tuple(gate0["shape"])
    up_shape = tuple(up0["shape"])
    down_shape = tuple(down0["shape"])
    if len(gate_shape) != 2 or len(up_shape) != 2 or len(down_shape) != 2:
        raise ValueError(f"Unexpected unpacked BF16 expert shapes: gate={gate_shape}, up={up_shape}, down={down_shape}")
    if gate_shape != up_shape:
        raise ValueError(f"Mismatched unpacked BF16 gate/up shapes: gate={gate_shape}, up={up_shape}")
    intermediate_size, hidden_size = gate_shape
    if down_shape != (hidden_size, intermediate_size):
        raise ValueError(
            f"Unexpected unpacked BF16 down shape for {prefix}: "
            f"expected={(hidden_size, intermediate_size)} actual={down_shape}"
        )

    for expert_id in range(expert_count):
        for proj_name, proj in (("gate", gate_name), ("up", up_name), ("down", down_name)):
            key = f"{prefix}.{expert_id}.{proj}.weight"
            if key not in loader.tensor_info_map:
                raise KeyError(f"Missing BF16 expert tensor: {key}")
            info = loader.tensor_info_map[key]
            if proj_name in ("gate", "up") and tuple(info["shape"]) != gate_shape:
                raise ValueError(f"BF16 {proj_name} shape mismatch for {key}: {info['shape']} vs {gate_shape}")
            if proj_name == "down" and tuple(info["shape"]) != down_shape:
                raise ValueError(f"BF16 down shape mismatch for {key}: {info['shape']} vs {down_shape}")
            _tensor_nbytes(info)

    return {
        "format": loader._detected_format,
        "base_key": base_key,
        "prefix": prefix,
        "gate_name": gate_name,
        "up_name": up_name,
        "down_name": down_name,
        "expert_count": int(expert_count),
        "hidden_size": int(hidden_size),
        "intermediate_size": int(intermediate_size),
    }


def _bf16_layout_fingerprint(loader, layout: dict) -> str:
    keys = []
    if layout["format"] == "packed":
        keys.extend([layout["gate_key"], layout["down_key"]])
    else:
        prefix = layout["prefix"]
        for expert_id in range(layout["expert_count"]):
            keys.extend(
                [
                    f"{prefix}.{expert_id}.{layout['gate_name']}.weight",
                    f"{prefix}.{expert_id}.{layout['up_name']}.weight",
                    f"{prefix}.{expert_id}.{layout['down_name']}.weight",
                ]
            )

    payload = {
        "version": 1,
        "format": layout["format"],
        "base_key": layout["base_key"],
        "expert_count": layout["expert_count"],
        "hidden_size": layout["hidden_size"],
        "intermediate_size": layout["intermediate_size"],
        "tensors": [],
    }
    for key in keys:
        info = loader.tensor_info_map[key]
        stat = os.stat(info["file_path"])
        payload["tensors"].append(
            {
                "key": key,
                "file": os.path.abspath(info["file_path"]),
                "file_size": stat.st_size,
                "file_mtime_ns": stat.st_mtime_ns,
                "dtype": info["dtype"],
                "shape": list(info["shape"]),
                "offset": int(info["offset"]),
                "size": int(info["size"]),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bf16_cache_paths(loader, layout: dict) -> tuple[str, str]:
    fingerprint = _bf16_layout_fingerprint(loader, layout)
    cache_dir = os.path.join(_bf16_cache_root(), fingerprint[:2])
    name = f"bf16_{_safe_name(layout['base_key'])}_{fingerprint[:24]}"
    return os.path.join(cache_dir, f"{name}.bin"), os.path.join(cache_dir, f"{name}.json")


def _bf16_materialize_cache_file(loader, layout: dict, cache_path: str, manifest_path: str) -> bool:
    gate_full_bytes = layout["intermediate_size"] * layout["hidden_size"] * np.dtype(np.uint16).itemsize
    up_full_bytes = gate_full_bytes
    down_full_bytes = layout["hidden_size"] * layout["intermediate_size"] * np.dtype(np.uint16).itemsize
    expert_blob_bytes = gate_full_bytes + up_full_bytes + down_full_bytes
    expected_size = expert_blob_bytes * layout["expert_count"]

    if os.path.exists(cache_path) and os.path.exists(manifest_path) and os.path.getsize(cache_path) == expected_size:
        return False

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    src_handles = {}
    try:
        with open(tmp_path, "wb") as dst:
            if layout["format"] == "packed":
                gate_up_info = layout["gate_up_info"]
                down_info = layout["down_info"]
                gate_up_bytes_per_expert = 2 * gate_full_bytes
                for expert_id in range(layout["expert_count"]):
                    gate_up_base = int(gate_up_info["offset"]) + expert_id * gate_up_bytes_per_expert
                    down_base = int(down_info["offset"]) + expert_id * down_full_bytes
                    _copy_file_slice(src_handles, gate_up_info["file_path"], gate_up_base, gate_full_bytes, dst)
                    _copy_file_slice(
                        src_handles, gate_up_info["file_path"], gate_up_base + gate_full_bytes, up_full_bytes, dst
                    )
                    _copy_file_slice(src_handles, down_info["file_path"], down_base, down_full_bytes, dst)
            else:
                prefix = layout["prefix"]
                for expert_id in range(layout["expert_count"]):
                    for proj in (layout["gate_name"], layout["up_name"], layout["down_name"]):
                        key = f"{prefix}.{expert_id}.{proj}.weight"
                        info = loader.tensor_info_map[key]
                        _copy_file_slice(src_handles, info["file_path"], int(info["offset"]), int(info["size"]), dst)
            dst.flush()
            os.fsync(dst.fileno())
        if os.path.getsize(tmp_path) != expected_size:
            raise RuntimeError(
                f"BF16 expert cache size mismatch for {cache_path}: "
                f"expected={expected_size} actual={os.path.getsize(tmp_path)}"
            )
        os.replace(tmp_path, cache_path)
        manifest = {
            "version": 1,
            "format": layout["format"],
            "base_key": layout["base_key"],
            "expert_count": layout["expert_count"],
            "hidden_size": layout["hidden_size"],
            "intermediate_size": layout["intermediate_size"],
            "gate_full_bytes": gate_full_bytes,
            "up_full_bytes": up_full_bytes,
            "down_full_bytes": down_full_bytes,
            "expert_blob_bytes": expert_blob_bytes,
            "cache_path": cache_path,
        }
        tmp_manifest = f"{manifest_path}.tmp.{os.getpid()}"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
        os.replace(tmp_manifest, manifest_path)
        return True
    finally:
        for src in src_handles.values():
            src.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _bf16_slots_from_cache(loader, layout: dict, cache_path: str, use_direct_io: bool) -> dict:
    bytes_per_elem = np.dtype(np.uint16).itemsize
    expert_count = layout["expert_count"]
    hidden_size = layout["hidden_size"]
    intermediate_size = layout["intermediate_size"]
    if intermediate_size % layout["tp_count"] != 0:
        raise ValueError(f"intermediate_size={intermediate_size} must be divisible by tp_count={layout['tp_count']}")

    tp_intermediate = intermediate_size // layout["tp_count"]
    gate_bytes_per_tp = tp_intermediate * hidden_size * bytes_per_elem
    up_bytes_per_tp = gate_bytes_per_tp
    gate_full_bytes = intermediate_size * hidden_size * bytes_per_elem
    up_full_bytes = gate_full_bytes
    down_full_bytes = hidden_size * intermediate_size * bytes_per_elem
    expert_blob_bytes = gate_full_bytes + up_full_bytes + down_full_bytes

    direct_io = bool(use_direct_io)
    if direct_io:
        for name, size in (
            ("gate.weight", gate_bytes_per_tp),
            ("up.weight", up_bytes_per_tp),
            ("down.weight", down_full_bytes),
        ):
            if size % 512 != 0:
                raise RuntimeError(f"BF16 expert cache direct I/O requires 512-byte aligned {name} size, got {size}")

    fd = get_file_fd(loader, cache_path, use_direct_io=direct_io)
    gate_slots = [[] for _ in range(layout["tp_count"])]
    up_slots = [[] for _ in range(layout["tp_count"])]
    down_slots = [[] for _ in range(layout["tp_count"])]
    for tp_idx in range(layout["tp_count"]):
        tp_offset = tp_idx * gate_bytes_per_tp
        for expert_id in range(expert_count):
            expert_base = expert_id * expert_blob_bytes
            gate_slot = (fd, expert_base + tp_offset, gate_bytes_per_tp)
            up_slot = (fd, expert_base + gate_full_bytes + tp_offset, up_bytes_per_tp)
            down_slot = (fd, expert_base + gate_full_bytes + up_full_bytes, down_full_bytes)
            if direct_io:
                for slot in (gate_slot, up_slot, down_slot):
                    if int(slot[1]) % 512 != 0 or int(slot[2]) % 512 != 0:
                        raise RuntimeError(
                            "BF16 expert cache direct I/O requires 512-byte aligned slots; "
                            f"slot={slot} cache_path={cache_path}"
                        )
            gate_slots[tp_idx].append(gate_slot)
            up_slots[tp_idx].append(up_slot)
            down_slots[tp_idx].append(down_slot)

    empty_scales = [[] for _ in range(layout["tp_count"])]
    return {
        "gate": gate_slots,
        "up": up_slots,
        "down": down_slots,
        "gate_scale": empty_scales,
        "up_scale": empty_scales,
        "down_scale": empty_scales,
        "direct_io": direct_io,
        "packed_bf16": layout["format"] == "packed",
        "bf16_expert_cache": True,
        "cache_path": cache_path,
    }


def _bf16_discover_base_keys(loader) -> list[str]:
    base_keys = set()
    if loader._detected_format == "packed":
        pattern = re.compile(r"^(?P<base>.+)\.mlp\.experts\.gate_up_proj$")
    else:
        path_tpl, gate_name, _, _ = BF16_MOE_FORMATS[loader._detected_format]
        marker = re.escape(path_tpl.format(base="__BASE__")).replace("__BASE__", r"(?P<base>.+)")
        pattern = re.compile(rf"^{marker}\.0\.{re.escape(gate_name)}\.weight$")

    for key in loader.tensor_file_map:
        match = pattern.match(key)
        if match is not None:
            base_keys.add(match.group("base"))
    return sorted(base_keys, key=lambda item: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", item)])


def precache_bf16_experts_iouring(loader, tp_count: int, use_direct_io: bool = True, base_keys: list[str] | None = None):
    if tp_count <= 0:
        raise ValueError(f"tp_count must be positive, got {tp_count}")
    selected_base_keys = list(base_keys) if base_keys is not None else _bf16_discover_base_keys(loader)
    if not selected_base_keys:
        raise ValueError(f"No BF16 MoE layers found for detected format {loader._detected_format!r}")

    generated = 0
    reused = 0
    cache_paths = []
    for base_key in selected_base_keys:
        layout = _bf16_build_iouring_layout(loader, base_key)
        layout["tp_count"] = int(tp_count)
        cache_path, manifest_path = _bf16_cache_paths(loader, layout)
        if _bf16_materialize_cache_file(loader, layout, cache_path, manifest_path):
            generated += 1
        else:
            reused += 1
        # Open once with the runtime I/O mode, so direct-I/O alignment problems
        # fail during startup instead of the first expert miss.
        slots = _bf16_slots_from_cache(loader, layout, cache_path, use_direct_io=use_direct_io)
        cache_paths.append(slots["cache_path"])

    return {
        "layers": len(selected_base_keys),
        "generated": generated,
        "reused": reused,
        "cache_root": _bf16_cache_root(),
        "cache_paths": cache_paths,
    }


def load_amx_experts_iouring(loader, base_key: str, use_direct_io: bool = True):
    up_base_key = f"{base_key}.ffn_up_exps"
    gate_base_key = f"{base_key}.ffn_gate_exps"
    down_base_key = f"{base_key}.ffn_down_exps"
    numa_ids, expert_ids = get_dense_moe_layout(loader, base_key)

    tensor_keys = []
    for numa_id in numa_ids:
        for expert_id in expert_ids:
            tensor_keys.extend(
                [
                    f"{up_base_key}.{expert_id}.numa.{numa_id}.weight",
                    f"{gate_base_key}.{expert_id}.numa.{numa_id}.weight",
                    f"{down_base_key}.{expert_id}.numa.{numa_id}.weight",
                    f"{up_base_key}.{expert_id}.numa.{numa_id}.scale",
                    f"{gate_base_key}.{expert_id}.numa.{numa_id}.scale",
                    f"{down_base_key}.{expert_id}.numa.{numa_id}.scale",
                ]
            )

    direct_io = use_direct_io
    if use_direct_io:
        for key in tensor_keys:
            info = loader.tensor_info_map[key]
            if int(info["offset"]) % 512 != 0 or int(info["size"]) % 512 != 0:
                raise RuntimeError(
                    "io_uring direct I/O requires 512-byte aligned safetensors entries; "
                    f"tensor {key} is misaligned (offset={info['offset']}, size={info['size']}). "
                    "Repack or repair the AMX safetensors with 512-byte data_offsets and sizes, "
                    "or set KT_IOURING_DIRECT=0 to use buffered io_uring for debugging."
                )

    up_weights = [[] for _ in numa_ids]
    gate_weights = [[] for _ in numa_ids]
    down_weights = [[] for _ in numa_ids]
    up_scales = [[] for _ in numa_ids]
    gate_scales = [[] for _ in numa_ids]
    down_scales = [[] for _ in numa_ids]

    for numa_id in numa_ids:
        for expert_id in expert_ids:
            up_weights[numa_id].append(
                get_file_slot(loader, f"{up_base_key}.{expert_id}.numa.{numa_id}.weight", direct_io)
            )
            gate_weights[numa_id].append(
                get_file_slot(loader, f"{gate_base_key}.{expert_id}.numa.{numa_id}.weight", direct_io)
            )
            down_weights[numa_id].append(
                get_file_slot(loader, f"{down_base_key}.{expert_id}.numa.{numa_id}.weight", direct_io)
            )
            up_scales[numa_id].append(
                get_file_slot(loader, f"{up_base_key}.{expert_id}.numa.{numa_id}.scale", direct_io)
            )
            gate_scales[numa_id].append(
                get_file_slot(loader, f"{gate_base_key}.{expert_id}.numa.{numa_id}.scale", direct_io)
            )
            down_scales[numa_id].append(
                get_file_slot(loader, f"{down_base_key}.{expert_id}.numa.{numa_id}.scale", direct_io)
            )

    return {
        "up": up_weights,
        "gate": gate_weights,
        "down": down_weights,
        "up_scale": up_scales,
        "gate_scale": gate_scales,
        "down_scale": down_scales,
        "direct_io": direct_io,
    }


def close_mesh_handles(loader):
    for fd in loader.file_fd_map.values():
        os.close(fd)
    loader.file_fd_map.clear()


BF16_MOE_FORMATS = {
    "qwen35_unfused": ("{base}.mlp.experts.experts", "gate_proj", "up_proj", "down_proj"),
    "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
    "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
    "mistral": ("{base}.experts", "w1", "w3", "w2"),
}


def bf16_detect_format(loader):
    sample_keys = list(loader.tensor_file_map.keys())[:1000]

    for key in sample_keys:
        if key.endswith(".mlp.experts.gate_up_proj"):
            loader._detected_format = "packed"
            print("[BF16SafeTensorLoader] Detected format: packed (Qwen3.5 MoE style)")
            return

    for key in sample_keys:
        if ".mlp.experts.experts." in key and ".gate_proj.weight" in key:
            loader._detected_format = "qwen35_unfused"
            print("[BF16SafeTensorLoader] Detected format: qwen35_unfused")
            return

    for fmt_name, (path_tpl, gate, up, down) in BF16_MOE_FORMATS.items():
        for key in sample_keys:
            if ".experts." in key and f".{gate}.weight" in key:
                if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                    loader._detected_format = fmt_name
                    print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                    return
                if "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                    loader._detected_format = fmt_name
                    print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                    return
                if fmt_name == "mistral" and ".mlp.experts" not in key and ".block_sparse_moe.experts" not in key:
                    loader._detected_format = fmt_name
                    print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                    return

    loader._detected_format = "deepseek"
    print("[BF16SafeTensorLoader] No MoE format detected, defaulting to: deepseek")


def bf16_get_experts_prefix_candidates(loader, base_key: str) -> list[str]:
    path_tpl, _, _, _ = BF16_MOE_FORMATS[loader._detected_format]
    candidates = [path_tpl.format(base=base_key)]
    if base_key.startswith("model."):
        candidates.append(path_tpl.format(base=base_key[len("model.") :]))
    return list(dict.fromkeys(candidates))


def bf16_get_proj_names(loader):
    _, gate, up, down = BF16_MOE_FORMATS[loader._detected_format]
    return gate, up, down


def bf16_resolve_packed_experts_prefix(loader, base_key: str) -> str:
    experts_prefix = f"{base_key}.mlp.experts"
    if loader.has_tensor(f"{experts_prefix}.gate_up_proj"):
        return experts_prefix

    parts = base_key.split(".", 1)
    if len(parts) == 2:
        alt_base = f"{parts[0]}.language_model.{parts[1]}"
        experts_prefix = f"{alt_base}.mlp.experts"
        if loader.has_tensor(f"{experts_prefix}.gate_up_proj"):
            return experts_prefix

    raise ValueError(f"No packed experts found for base_key '{base_key}'.")


def bf16_load_experts_packed(loader, base_key: str, device: str = "cpu"):
    experts_prefix = bf16_resolve_packed_experts_prefix(loader, base_key)
    gate_up = loader.load_tensor(f"{experts_prefix}.gate_up_proj", device)
    down = loader.load_tensor(f"{experts_prefix}.down_proj", device)
    mid = gate_up.shape[1] // 2
    return {
        "gate": [gate_up[i, :mid, :].contiguous() for i in range(gate_up.shape[0])],
        "up": [gate_up[i, mid:, :].contiguous() for i in range(gate_up.shape[0])],
        "down": [down[i].contiguous() for i in range(down.shape[0])],
    }


def bf16_load_experts(loader, base_key: str, device: str = "cpu"):
    if loader._detected_format == "packed":
        return bf16_load_experts_packed(loader, base_key, device)

    experts_prefix_candidates = bf16_get_experts_prefix_candidates(loader, base_key)
    gate_name, up_name, down_name = bf16_get_proj_names(loader)
    expert_count = 0
    experts_prefix = None
    for prefix in experts_prefix_candidates:
        expert_count = 0
        while loader.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
            expert_count += 1
        if expert_count > 0:
            experts_prefix = prefix
            break

    if expert_count == 0 or experts_prefix is None:
        raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")

    gate_weights = [None] * expert_count
    up_weights = [None] * expert_count
    down_weights = [None] * expert_count
    for exp_id in range(expert_count):
        gate_weights[exp_id] = loader.load_tensor(f"{experts_prefix}.{exp_id}.{gate_name}.weight", device).contiguous()
        up_weights[exp_id] = loader.load_tensor(f"{experts_prefix}.{exp_id}.{up_name}.weight", device).contiguous()
        down_weights[exp_id] = loader.load_tensor(f"{experts_prefix}.{exp_id}.{down_name}.weight", device).contiguous()

    return {
        "gate": gate_weights,
        "up": up_weights,
        "down": down_weights,
    }


def patch_bf16_loader(loader_cls) -> None:
    loader_cls.MOE_FORMATS = BF16_MOE_FORMATS
    loader_cls._detect_format = bf16_detect_format
    loader_cls._get_experts_prefix_candidates = bf16_get_experts_prefix_candidates
    loader_cls._get_proj_names = bf16_get_proj_names
    loader_cls.load_experts = bf16_load_experts
    loader_cls._resolve_packed_experts_prefix = bf16_resolve_packed_experts_prefix
    loader_cls._load_experts_packed = bf16_load_experts_packed
    loader_cls.load_experts_iouring = lambda self, base_key, tp_count, use_direct_io=True: load_bf16_experts_iouring(
        self, base_key, tp_count=tp_count, use_direct_io=use_direct_io
    )
    loader_cls.precache_experts_iouring = lambda self, tp_count, use_direct_io=True, base_keys=None: precache_bf16_experts_iouring(
        self, tp_count=tp_count, use_direct_io=use_direct_io, base_keys=base_keys
    )


def _bf16_packed_original_slots(loader, layout: dict, use_direct_io: bool) -> dict:
    gate_up_info = layout["gate_up_info"]
    down_info = layout["down_info"]
    expert_count = layout["expert_count"]
    hidden_size = layout["hidden_size"]
    intermediate_size = layout["intermediate_size"]
    tp_count = layout["tp_count"]
    if intermediate_size % tp_count != 0:
        raise ValueError(f"intermediate_size={intermediate_size} must be divisible by tp_count={tp_count}")

    bytes_per_elem = np.dtype(np.uint16).itemsize
    tp_intermediate = intermediate_size // tp_count
    gate_bytes_per_tp = tp_intermediate * hidden_size * bytes_per_elem
    up_bytes_per_tp = gate_bytes_per_tp
    gate_bytes_per_expert = intermediate_size * hidden_size * bytes_per_elem
    gate_up_bytes_per_expert = 2 * gate_bytes_per_expert
    down_bytes_per_expert = hidden_size * intermediate_size * bytes_per_elem

    direct_io = bool(use_direct_io)
    slots_to_validate = []
    gate_fd = get_file_fd(loader, gate_up_info["file_path"], use_direct_io=direct_io)
    down_fd = get_file_fd(loader, down_info["file_path"], use_direct_io=direct_io)

    gate_slots = [[] for _ in range(tp_count)]
    up_slots = [[] for _ in range(tp_count)]
    down_slots = [[] for _ in range(tp_count)]
    for tp_idx in range(tp_count):
        tp_offset = tp_idx * gate_bytes_per_tp
        for expert_id in range(expert_count):
            expert_gate_up_base = int(gate_up_info["offset"]) + expert_id * gate_up_bytes_per_expert
            expert_down_base = int(down_info["offset"]) + expert_id * down_bytes_per_expert
            gate_slot = (gate_fd, expert_gate_up_base + tp_offset, gate_bytes_per_tp)
            up_slot = (gate_fd, expert_gate_up_base + gate_bytes_per_expert + tp_offset, up_bytes_per_tp)
            down_slot = (down_fd, expert_down_base, down_bytes_per_expert)
            gate_slots[tp_idx].append(gate_slot)
            up_slots[tp_idx].append(up_slot)
            down_slots[tp_idx].append(down_slot)
            slots_to_validate.extend((gate_slot, up_slot, down_slot))

    if direct_io:
        for fd, offset, size in slots_to_validate:
            if int(offset) % 512 != 0 or int(size) % 512 != 0:
                raise RuntimeError(
                    "BF16 io_uring direct I/O requires 512-byte aligned original safetensors entries; "
                    f"fd={fd} offset={offset} size={size} is misaligned. "
                    "Enable KT_MESH_BF16_EXPERT_CACHE=1 to read from normalized expert cache, "
                    "or set KT_IOURING_DIRECT=0 for buffered io_uring."
                )

    empty_scales = [[] for _ in range(tp_count)]
    return {
        "gate": gate_slots,
        "up": up_slots,
        "down": down_slots,
        "gate_scale": empty_scales,
        "up_scale": empty_scales,
        "down_scale": empty_scales,
        "direct_io": direct_io,
        "packed_bf16": True,
        "bf16_expert_cache": False,
    }


def load_bf16_experts_iouring(loader, base_key: str, tp_count: int, use_direct_io: bool = True):
    if tp_count <= 0:
        raise ValueError(f"tp_count must be positive, got {tp_count}")
    layout = _bf16_build_iouring_layout(loader, base_key)
    layout["tp_count"] = int(tp_count)

    if not _bf16_cache_enabled(loader):
        if layout["format"] != "packed":
            raise RuntimeError("Non-packed BF16 io_uring requires normalized expert cache")
        return _bf16_packed_original_slots(loader, layout, use_direct_io=use_direct_io)

    cache_path, manifest_path = _bf16_cache_paths(loader, layout)
    generated = _bf16_materialize_cache_file(loader, layout, cache_path, manifest_path)
    slots = _bf16_slots_from_cache(loader, layout, cache_path, use_direct_io=use_direct_io)
    slots["cache_generated"] = generated
    return slots
