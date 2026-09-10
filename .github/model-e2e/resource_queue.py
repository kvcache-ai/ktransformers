"""Wait without allocating GPU memory; hold one host-wide advisory lock."""

from __future__ import annotations

import contextlib
import fcntl
import json
import subprocess
import time
from pathlib import Path


class ResourceUnavailable(RuntimeError):
    pass


def gpu_busy():
    """Fail closed on driver/query failures, and include non-compute VRAM users."""
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=20,
    ).strip()
    stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=20,
    ).strip()
    if not stats:
        raise ResourceUnavailable("No GPUs found")
    try:
        rows = [
            [int(value.strip()) for value in row.split(",")]
            for row in stats.splitlines()
        ]
        busy = bool(apps) or any(
            memory > 256 or utilization > 5 for memory, utilization in rows
        )
    except (ValueError, TypeError) as exc:
        raise ResourceUnavailable("Cannot determine GPU usage") from exc
    return busy, {"compute_pids": apps, "gpu_memory_mib_utilization": rows}


@contextlib.contextmanager
def reservation(
    lock_path, log_path, timeout=21600, interval=15, idle_samples=3, probe=gpu_busy
):
    """No Actions concurrency group: it can replace older pending jobs.

    A persistent lock inode must never be deleted after use. Manual workloads
    should acquire the same lock to close the idle-check/start race.
    """
    lock_path = Path(lock_path)
    if not lock_path.parent.is_dir():
        raise ResourceUnavailable(
            "Administrator must provision the shared lock directory"
        )
    deadline = time.monotonic() + timeout
    stable = 0
    with lock_path.open("a") as lock, Path(log_path).open("a") as log:
        owned = False
        try:
            while time.monotonic() < deadline:
                if not owned:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        owned = True
                    except BlockingIOError:
                        pass
                state = {
                    "status": "waiting_for_qj5090",
                    "reason": "another CI reservation",
                }
                if owned:
                    busy, evidence = probe()
                    stable = 0 if busy else stable + 1
                    state.update(evidence)
                    state["reason"] = (
                        "GPU workload present" if busy else "checking stable idle"
                    )
                    if stable >= idle_samples:
                        log.write(json.dumps({"status": "reserved"}) + "\n")
                        log.flush()
                        yield
                        return
                print(json.dumps(state), flush=True)
                log.write(json.dumps(state) + "\n")
                log.flush()
                time.sleep(interval)
            raise ResourceUnavailable("Queue wait expired; no model test was run")
        finally:
            if owned:
                fcntl.flock(lock, fcntl.LOCK_UN)
