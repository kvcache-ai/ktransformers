from __future__ import annotations

import json
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


def load_bf16_experts_iouring(loader, base_key: str, tp_count: int, use_direct_io: bool = True):
    if tp_count <= 0:
        raise ValueError(f"tp_count must be positive, got {tp_count}")
    if loader._detected_format != "packed":
        raise NotImplementedError("BF16 io_uring currently supports packed Qwen-style expert tensors only")

    experts_prefix = loader._resolve_packed_experts_prefix(base_key)
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
                    "Run kt-kernel/scripts/align_safetensors_for_direct_io.py on a copied model directory, "
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
    }
