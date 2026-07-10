# Lightweight SFT profiling helpers.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import itertools
import json
import os
import socket
import time
from typing import Any, Iterator

try:
    import torch
except Exception:  # pragma: no cover - import guard for non-torch tooling
    torch = None  # type: ignore[assignment]


_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_CALL_COUNTER = itertools.count()
_STEP_CONTEXT: dict[str, Any] = {
    "global_step_before": None,
    "micro_step_index": None,
    "optimizer_step_index": None,
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in _TRUE_VALUES


def _profile_path() -> str | None:
    path = os.environ.get("KT_SFT_PROFILE_JSONL")
    return path if path else None


def profiling_enabled() -> bool:
    return _profile_path() is not None


def nvtx_enabled() -> bool:
    return _env_bool("KT_SFT_NVTX", profiling_enabled())


def sync_enabled() -> bool:
    return _env_bool("KT_SFT_PROFILE_SYNC", profiling_enabled())


def _rank() -> int:
    for name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value != "":
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def _local_rank() -> int:
    value = os.environ.get("LOCAL_RANK")
    if value is None or value == "":
        return _rank()
    try:
        return int(value)
    except ValueError:
        return _rank()


def _rank_allowed(rank: int) -> bool:
    spec = os.environ.get("KT_SFT_PROFILE_RANKS", "all").strip().lower()
    if spec in ("", "all", "*"):
        return True
    allowed: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            allowed.add(int(item))
        except ValueError:
            continue
    return rank in allowed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def set_step_context(
    *,
    global_step_before: int | None = None,
    micro_step_index: int | None = None,
    optimizer_step_index: int | None = None,
) -> None:
    _STEP_CONTEXT["global_step_before"] = global_step_before
    _STEP_CONTEXT["micro_step_index"] = micro_step_index
    _STEP_CONTEXT["optimizer_step_index"] = optimizer_step_index


def clear_step_context() -> None:
    set_step_context()


def _cuda_sync(device: Any = None) -> None:
    if not sync_enabled() or torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        return


def nvtx_push(name: str) -> None:
    if not nvtx_enabled() or torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
    except Exception:
        return


def nvtx_pop() -> None:
    if not nvtx_enabled() or torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
    except Exception:
        return


@contextlib.contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    enabled = nvtx_enabled()
    if enabled:
        nvtx_push(name)
    try:
        yield
    finally:
        if enabled:
            nvtx_pop()


def _write_record(record: dict[str, Any]) -> None:
    path = _profile_path()
    if not path:
        return
    rank = int(record.get("rank", _rank()))
    if not _rank_allowed(rank):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@contextlib.contextmanager
def profile_scope(
    phase: str,
    *,
    direction: str | None = None,
    layer_idx: int | None = None,
    nvtx_name: str | None = None,
    device: Any = None,
    **metadata: Any,
) -> Iterator[None]:
    do_profile = profiling_enabled()
    do_nvtx = nvtx_enabled() and bool(nvtx_name or phase)
    range_name = nvtx_name or phase
    exc: BaseException | None = None

    if do_nvtx:
        nvtx_push(range_name)
    if do_profile:
        _cuda_sync(device)
        start = time.perf_counter()

    try:
        yield
    except BaseException as error:
        exc = error
        raise
    finally:
        if do_profile:
            _cuda_sync(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rank = _rank()
            record = {
                "record_type": "python_phase",
                "phase": phase,
                "direction": direction,
                "layer_idx": layer_idx,
                "elapsed_ms": elapsed_ms,
                "call_index": next(_CALL_COUNTER),
                "rank": rank,
                "local_rank": _local_rank(),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "status": "error" if exc is not None else "ok",
                **_STEP_CONTEXT,
                **metadata,
            }
            if exc is not None:
                record["error_type"] = type(exc).__name__
            _write_record(record)
        if do_nvtx:
            nvtx_pop()


def cuda_profiler_start() -> None:
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStart()
    except Exception:
        return


def cuda_profiler_stop() -> None:
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaProfilerStop()
    except Exception:
        return
