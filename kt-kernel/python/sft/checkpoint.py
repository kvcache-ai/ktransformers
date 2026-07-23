# Full-weight checkpoint utilities for KT SFT
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import torch

from .dist_utils import _distributed_rank_world_size

logger = logging.getLogger(__name__)

FULL_WEIGHT_CHECKPOINT_FORMAT = "kt_full_experts"
FULL_WEIGHT_CHECKPOINT_VERSION = 1
FULL_WEIGHT_INDEX_NAME = "kt_full_experts.index.json"
_PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")
_DTYPE_NAME = "BF16"


def resolve_full_weight_checkpoint(path: str | os.PathLike[str] | None) -> str | None:
    """Return the checkpoint directory when *path* contains a KT Full index."""
    if path is None:
        return None
    candidate = os.fspath(path)
    if os.path.isfile(candidate):
        if os.path.basename(candidate) != FULL_WEIGHT_INDEX_NAME:
            return None
        return os.path.dirname(os.path.abspath(candidate))
    index_path = os.path.join(candidate, FULL_WEIGHT_INDEX_NAME)
    if os.path.isfile(index_path):
        return os.path.abspath(candidate)
    return None


def _full_weight_wrappers(wrappers: list[Any]) -> list[Any]:
    return [wrapper for wrapper in wrappers if bool(getattr(wrapper, "_full_weight_grad", False))]


def _expected_shapes(wrapper: Any) -> dict[str, tuple[int, ...]]:
    moe_config = wrapper.moe_config
    expert_num = int(moe_config.expert_num)
    intermediate_size = int(moe_config.intermediate_size)
    hidden_size = int(wrapper.hidden_size)
    return {
        "gate_proj": (expert_num, intermediate_size, hidden_size),
        "up_proj": (expert_num, intermediate_size, hidden_size),
        "down_proj": (expert_num, hidden_size, intermediate_size),
    }


def _backend_parameters(wrapper: Any) -> dict[str, torch.nn.Parameter]:
    backend = getattr(wrapper, "wrapper", None)
    if backend is None:
        raise RuntimeError(f"Layer {wrapper.layer_idx}: rank 0 does not own the KT Full backend")

    params: dict[str, torch.nn.Parameter] = {}
    for name in _PROJECTION_NAMES:
        param = getattr(backend, f"{name}_buf", None)
        if not isinstance(param, torch.nn.Parameter):
            raise RuntimeError(f"Layer {wrapper.layer_idx}: missing authoritative Parameter {name}_buf")
        params[name] = param
    return params


def _validate_parameter_set(wrapper: Any, params: dict[str, torch.nn.Parameter]) -> None:
    expected_shapes = _expected_shapes(wrapper)
    for name, param in params.items():
        if tuple(param.shape) != expected_shapes[name]:
            raise RuntimeError(
                f"Layer {wrapper.layer_idx} {name}_buf shape mismatch: "
                f"got {tuple(param.shape)}, expected {expected_shapes[name]}"
            )
        if param.dtype != torch.bfloat16:
            raise RuntimeError(
                f"Layer {wrapper.layer_idx} {name}_buf dtype mismatch: got {param.dtype}, expected torch.bfloat16"
            )
        if param.device.type != "cpu":
            raise RuntimeError(f"Layer {wrapper.layer_idx} {name}_buf must be on CPU, got device={param.device}")
        if not param.is_contiguous():
            raise RuntimeError(f"Layer {wrapper.layer_idx} {name}_buf must be contiguous")
        if not param.requires_grad:
            raise RuntimeError(f"Layer {wrapper.layer_idx} {name}_buf must require gradients")


def _read_index(checkpoint_dir: str) -> dict[str, Any]:
    index_path = os.path.join(checkpoint_dir, FULL_WEIGHT_INDEX_NAME)
    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing KT Full checkpoint index: {index_path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read KT Full checkpoint index {index_path}: {exc}") from exc

    if index.get("format") != FULL_WEIGHT_CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"Unsupported KT Full checkpoint format {index.get('format')!r}; "
            f"expected {FULL_WEIGHT_CHECKPOINT_FORMAT!r}"
        )
    if index.get("version") != FULL_WEIGHT_CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported KT Full checkpoint version {index.get('version')!r}; "
            f"expected {FULL_WEIGHT_CHECKPOINT_VERSION}"
        )
    if index.get("dtype") != _DTYPE_NAME:
        raise RuntimeError(f"Unsupported KT Full checkpoint dtype {index.get('dtype')!r}; expected {_DTYPE_NAME}")
    if not isinstance(index.get("generation"), str) or not index["generation"]:
        raise RuntimeError("KT Full checkpoint index must contain a non-empty generation")
    layers = index.get("layers")
    if not isinstance(layers, dict):
        raise RuntimeError("KT Full checkpoint index must contain a 'layers' object")
    return index


def _safe_shard_path(checkpoint_dir: str, filename: Any) -> str:
    if not isinstance(filename, str) or not filename:
        raise RuntimeError(f"Invalid KT Full checkpoint shard filename: {filename!r}")
    if os.path.basename(filename) != filename:
        raise RuntimeError(f"KT Full checkpoint shard must be a basename: {filename!r}")
    return os.path.join(checkpoint_dir, filename)


def _inspect_layer_shard(
    checkpoint_dir: str,
    layer_idx: int,
    layer_entry: Any,
    expected_shapes: dict[str, tuple[int, ...]],
    expected_generation: str,
) -> str:
    from safetensors import safe_open

    if not isinstance(layer_entry, dict):
        raise RuntimeError(f"Layer {layer_idx}: invalid KT Full checkpoint index entry")
    indexed_shapes = layer_entry.get("shapes")
    expected_index_shapes = {name: list(expected_shapes[name]) for name in _PROJECTION_NAMES}
    if indexed_shapes != expected_index_shapes:
        raise RuntimeError(
            f"Layer {layer_idx}: KT Full checkpoint index shapes mismatch: "
            f"got {indexed_shapes!r}, expected {expected_index_shapes!r}"
        )
    shard_path = _safe_shard_path(checkpoint_dir, layer_entry.get("file"))
    if not os.path.isfile(shard_path):
        raise FileNotFoundError(f"Layer {layer_idx}: missing KT Full checkpoint shard {shard_path}")
    expected_size = layer_entry.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise RuntimeError(f"Layer {layer_idx}: invalid shard size in KT Full checkpoint index")
    actual_size = os.path.getsize(shard_path)
    if actual_size != expected_size:
        raise RuntimeError(
            f"Layer {layer_idx}: KT Full checkpoint shard size mismatch: "
            f"got {actual_size}, expected {expected_size}"
        )

    try:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            expected_keys = set(_PROJECTION_NAMES)
            if keys != expected_keys:
                raise RuntimeError(
                    f"Layer {layer_idx}: KT Full checkpoint keys mismatch: "
                    f"got {sorted(keys)}, expected {sorted(expected_keys)}"
                )
            metadata = handle.metadata() or {}
            if metadata.get("format") != FULL_WEIGHT_CHECKPOINT_FORMAT:
                raise RuntimeError(f"Layer {layer_idx}: invalid KT Full checkpoint shard format metadata")
            if metadata.get("version") != str(FULL_WEIGHT_CHECKPOINT_VERSION):
                raise RuntimeError(f"Layer {layer_idx}: invalid KT Full checkpoint shard version metadata")
            if metadata.get("generation") != expected_generation:
                raise RuntimeError(f"Layer {layer_idx}: shard generation metadata does not match the index")
            if metadata.get("layer_idx") != str(layer_idx):
                raise RuntimeError(f"Layer {layer_idx}: shard layer metadata does not match the index")

            for name in _PROJECTION_NAMES:
                tensor_slice = handle.get_slice(name)
                shape = tuple(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if shape != expected_shapes[name]:
                    raise RuntimeError(
                        f"Layer {layer_idx} {name} shape mismatch: " f"got {shape}, expected {expected_shapes[name]}"
                    )
                if dtype != _DTYPE_NAME:
                    raise RuntimeError(f"Layer {layer_idx} {name} dtype mismatch: got {dtype}, expected {_DTYPE_NAME}")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Layer {layer_idx}: failed to inspect KT Full checkpoint shard: {exc}") from exc
    return shard_path


def save_full_weight_checkpoint(wrappers: list[Any], output_dir: str) -> bool:
    """Atomically publish rank-0 Full/Hybrid expert Parameters as per-layer shards."""
    full_wrappers = _full_weight_wrappers(wrappers)
    if not full_wrappers:
        return False

    rank, _ = _distributed_rank_world_size()
    if rank != 0:
        return False

    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    generation = uuid.uuid4().hex
    layer_entries: dict[str, dict[str, Any]] = {}
    created_paths: list[str] = []
    temporary_paths: list[str] = []
    seen_layers: set[int] = set()
    previous_index = None
    previous_checkpoint = resolve_full_weight_checkpoint(output_dir)
    if previous_checkpoint is not None:
        try:
            previous_index = _read_index(previous_checkpoint)
        except Exception:
            logger.warning("Existing KT Full checkpoint index is invalid; it will not be cleaned up", exc_info=True)

    try:
        for wrapper in sorted(full_wrappers, key=lambda item: int(item.layer_idx)):
            layer_idx = int(wrapper.layer_idx)
            if layer_idx < 0:
                raise RuntimeError(f"Invalid negative KT Full layer index {layer_idx}")
            if layer_idx in seen_layers:
                raise RuntimeError(f"Duplicate KT Full wrapper for layer {layer_idx}")
            seen_layers.add(layer_idx)

            params = _backend_parameters(wrapper)
            _validate_parameter_set(wrapper, params)
            filename = f"kt_full_experts-{generation}-layer-{layer_idx:05d}.safetensors"
            shard_path = os.path.join(output_dir, filename)
            temp_path = f"{shard_path}.tmp"
            temporary_paths.append(temp_path)
            tensors = {name: param.detach() for name, param in params.items()}
            save_file(
                tensors,
                temp_path,
                metadata={
                    "format": FULL_WEIGHT_CHECKPOINT_FORMAT,
                    "version": str(FULL_WEIGHT_CHECKPOINT_VERSION),
                    "generation": generation,
                    "layer_idx": str(layer_idx),
                },
            )
            os.replace(temp_path, shard_path)
            temporary_paths.remove(temp_path)
            created_paths.append(shard_path)
            layer_entries[str(layer_idx)] = {
                "file": filename,
                "size_bytes": os.path.getsize(shard_path),
                "shapes": {name: list(params[name].shape) for name in _PROJECTION_NAMES},
            }

        index = {
            "format": FULL_WEIGHT_CHECKPOINT_FORMAT,
            "version": FULL_WEIGHT_CHECKPOINT_VERSION,
            "dtype": _DTYPE_NAME,
            "generation": generation,
            "layers": layer_entries,
        }
        index_path = os.path.join(output_dir, FULL_WEIGHT_INDEX_NAME)
        temp_index_path = f"{index_path}.{generation}.tmp"
        temporary_paths.append(temp_index_path)
        with open(temp_index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_index_path, index_path)
        temporary_paths.remove(temp_index_path)
    except Exception:
        for path in temporary_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        for path in created_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise

    if previous_index is not None:
        for entry in previous_index.get("layers", {}).values():
            if not isinstance(entry, dict):
                continue
            try:
                old_path = _safe_shard_path(output_dir, entry.get("file"))
            except RuntimeError:
                continue
            if old_path not in created_paths:
                try:
                    os.remove(old_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("Failed to remove superseded KT Full checkpoint shard %s", old_path)

    logger.info(
        "Saved KT Full checkpoint generation %s with %d layer shards to %s",
        generation,
        len(layer_entries),
        output_dir,
    )
    return True


def load_full_weight_layer(
    checkpoint_path: str,
    *,
    layer_idx: int,
    expected_shapes: dict[str, tuple[int, ...]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and validate one logical layer for initial KT backend construction."""
    from safetensors import safe_open

    checkpoint_dir = resolve_full_weight_checkpoint(checkpoint_path)
    if checkpoint_dir is None:
        raise FileNotFoundError(f"No {FULL_WEIGHT_INDEX_NAME} found at {checkpoint_path}")
    index = _read_index(checkpoint_dir)
    entry = index["layers"].get(str(int(layer_idx)))
    if entry is None:
        raise RuntimeError(f"Layer {layer_idx}: missing from KT Full checkpoint index")
    shard_path = _inspect_layer_shard(
        checkpoint_dir,
        int(layer_idx),
        entry,
        expected_shapes,
        index["generation"],
    )
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensors = tuple(handle.get_tensor(name).contiguous() for name in _PROJECTION_NAMES)
    return tensors


def _load_full_weight_checkpoint_rank0(full_wrappers: list[Any], checkpoint_dir: str) -> int:
    from safetensors import safe_open

    wrapper_map: dict[int, Any] = {}
    parameter_map: dict[int, dict[str, torch.nn.Parameter]] = {}
    shard_map: dict[int, str] = {}
    index = _read_index(checkpoint_dir)

    for wrapper in full_wrappers:
        layer_idx = int(wrapper.layer_idx)
        if layer_idx in wrapper_map:
            raise RuntimeError(f"Duplicate KT Full wrapper for layer {layer_idx}")
        wrapper_map[layer_idx] = wrapper
        params = _backend_parameters(wrapper)
        _validate_parameter_set(wrapper, params)
        parameter_map[layer_idx] = params

    saved_layers: set[int] = set()
    for key in index["layers"]:
        try:
            layer_idx = int(key)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid layer key in KT Full checkpoint index: {key!r}") from exc
        if layer_idx < 0 or str(layer_idx) != key or layer_idx in saved_layers:
            raise RuntimeError(f"Non-canonical or duplicate layer key in KT Full checkpoint index: {key!r}")
        saved_layers.add(layer_idx)
    expected_layers = set(wrapper_map)
    if saved_layers != expected_layers:
        missing = sorted(expected_layers - saved_layers)
        unexpected = sorted(saved_layers - expected_layers)
        raise RuntimeError(f"KT Full checkpoint layer set mismatch: missing={missing}, unexpected={unexpected}")

    for layer_idx, wrapper in wrapper_map.items():
        shard_map[layer_idx] = _inspect_layer_shard(
            checkpoint_dir,
            layer_idx,
            index["layers"][str(layer_idx)],
            _expected_shapes(wrapper),
            index["generation"],
        )

    try:
        with torch.no_grad():
            for layer_idx in sorted(wrapper_map):
                with safe_open(shard_map[layer_idx], framework="pt", device="cpu") as handle:
                    for name in _PROJECTION_NAMES:
                        parameter_map[layer_idx][name].copy_(handle.get_tensor(name))
        for wrapper in wrapper_map.values():
            wrapper.wrapper._base_weights_dirty = True
            wrapper.wrapper._kt_full_checkpoint_load_failed = False
    except Exception:
        for wrapper in wrapper_map.values():
            wrapper.wrapper._kt_full_checkpoint_load_failed = True
        raise
    return len(wrapper_map)


def load_full_weight_checkpoint(wrappers: list[Any], checkpoint_path: str) -> bool:
    """Restore Full/Hybrid Parameters in place and synchronize failure across ranks."""
    full_wrappers = _full_weight_wrappers(wrappers)
    if not full_wrappers:
        return False

    checkpoint_dir = resolve_full_weight_checkpoint(checkpoint_path)
    rank, world_size = _distributed_rank_world_size()
    error_text: str | None = None
    loaded_layers = 0

    if rank == 0:
        if checkpoint_dir is None:
            error_text = f"No {FULL_WEIGHT_INDEX_NAME} found at {checkpoint_path}"
        else:
            try:
                loaded_layers = _load_full_weight_checkpoint_rank0(full_wrappers, checkpoint_dir)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"

    if world_size > 1:
        import torch.distributed as dist

        if dist.is_initialized():
            status: list[Any] = [{"error": error_text, "loaded_layers": loaded_layers} if rank == 0 else None]
            dist.broadcast_object_list(status, src=0)
            error_text = status[0]["error"]
            loaded_layers = int(status[0]["loaded_layers"])

    if error_text is not None:
        raise RuntimeError(f"Failed to load KT Full checkpoint: {error_text}")
    logger.info("Loaded KT Full checkpoint for %d layers from %s", loaded_layers, checkpoint_path)
    return True
