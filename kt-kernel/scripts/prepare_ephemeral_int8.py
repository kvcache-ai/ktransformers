#!/usr/bin/env python3
"""Convert model experts to a one-run INT8 .kt tree in /dev/shm."""

from __future__ import annotations

import atexit
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
KT_KERNEL_DIR = SCRIPT_DIR.parent
_EPHEMERAL_SPEC = importlib.util.spec_from_file_location(
    "kt_sft_ephemeral_converter", KT_KERNEL_DIR / "python" / "sft" / "ephemeral.py"
)
assert _EPHEMERAL_SPEC is not None and _EPHEMERAL_SPEC.loader is not None
_EPHEMERAL = importlib.util.module_from_spec(_EPHEMERAL_SPEC)
_EPHEMERAL_SPEC.loader.exec_module(_EPHEMERAL)
cleanup_ephemeral_int8_staging = _EPHEMERAL.cleanup_ephemeral_int8_staging
publish_ephemeral_int8_weights = _EPHEMERAL.publish_ephemeral_int8_weights
reclaim_stale_ephemeral_int8_staging = _EPHEMERAL.reclaim_stale_ephemeral_int8_staging
validate_ephemeral_run_id = _EPHEMERAL.validate_ephemeral_run_id
write_ephemeral_staging_lease = _EPHEMERAL.write_ephemeral_staging_lease


def _model_dimensions(model_path: Path) -> tuple[int, int, int]:
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config", config)
    expert_num = text_config.get("n_routed_experts", text_config.get("num_experts"))
    hidden_size = text_config.get("hidden_size")
    intermediate_size = text_config.get(
        "moe_intermediate_size", text_config.get("intermediate_size")
    )
    values = (expert_num, hidden_size, intermediate_size)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(
            "config.json must define positive expert, hidden and MoE intermediate sizes"
        )
    return values


def _discover_layers(staging: Path) -> list[int]:
    layers: list[int] = []
    for entry in staging.iterdir():
        if (
            entry.is_dir()
            and not entry.is_symlink()
            and entry.name.startswith("_layer_")
        ):
            try:
                layers.append(int(entry.name.removeprefix("_layer_")))
            except ValueError:
                continue
    if not layers:
        raise RuntimeError("INT8 converter produced no expert layer directories")
    return sorted(layers)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create manifest-guarded, ephemeral INT8 expert weights"
    )
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpuinfer-threads", type=int, default=60)
    parser.add_argument("--threadpool-count", type=int, default=2)
    args = parser.parse_args()

    run_id = validate_ephemeral_run_id(args.run_id)
    input_path = Path(args.input_path).resolve(strict=True)
    staging = Path("/dev/shm") / f"kt-int8-{run_id}.staging"
    ready = staging.with_name(f"kt-int8-{run_id}")
    if staging.exists() and not staging.is_symlink():
        if not reclaim_stale_ephemeral_int8_staging(staging, run_id=run_id):
            raise RuntimeError(
                f"ephemeral INT8 staging root still has a live producer: {staging}"
            )
    if staging.exists() or staging.is_symlink() or ready.exists() or ready.is_symlink():
        raise FileExistsError(
            f"refusing to reuse existing ephemeral path: {staging} or {ready}"
        )
    staging.mkdir(mode=0o700)
    write_ephemeral_staging_lease(
        staging,
        run_id=run_id,
        producer_pid=os.getpid(),
    )

    child: subprocess.Popen | None = None
    received_signal: int | None = None

    def cleanup_staging() -> None:
        if staging.exists() and not staging.is_symlink():
            cleanup_ephemeral_int8_staging(staging, run_id=run_id)

    atexit.register(cleanup_staging)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def handle_signal(signum, frame) -> None:
        nonlocal received_signal
        del frame
        received_signal = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    for signum in previous_handlers:
        signal.signal(signum, handle_signal)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "convert_cpu_weights.py"),
        "--input-path",
        str(input_path),
        "--input-type",
        "fp8",
        "--output",
        str(staging),
        "--quant-method",
        "int8",
        "--cpuinfer-threads",
        str(args.cpuinfer_threads),
        "--threadpool-count",
        str(args.threadpool_count),
        "--no-merge-safetensor",
    ]
    try:
        child = subprocess.Popen(command)
        write_ephemeral_staging_lease(
            staging,
            run_id=run_id,
            producer_pid=child.pid,
        )
        return_code = child.wait()
        if received_signal is not None:
            raise SystemExit(128 + received_signal)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        expert_num, hidden_size, intermediate_size = _model_dimensions(input_path)
        published = publish_ephemeral_int8_weights(
            staging,
            run_id=run_id,
            layer_indices=_discover_layers(staging),
            numa_count=args.threadpool_count,
            expert_num=expert_num,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    except BaseException:
        cleanup_staging()
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    atexit.unregister(cleanup_staging)
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
