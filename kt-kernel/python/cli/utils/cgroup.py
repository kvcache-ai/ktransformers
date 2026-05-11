from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, List, Tuple


_ALWAYS_FORWARD_ENV_KEYS = {
    "HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "TERM",
}

_FORWARD_ENV_PREFIXES = (
    "KT_",
    "HF_",
    "TRANSFORMERS_",
    "TORCH_",
    "PYTORCH_",
    "OMP_",
    "MKL_",
    "NCCL_",
    "TRITON_",
    "SGLANG_",
)


def _should_forward_env_var(key: str, value: str, parent_env: Dict[str, str]) -> bool:
    if key in _ALWAYS_FORWARD_ENV_KEYS:
        return True
    if any(key.startswith(prefix) for prefix in _FORWARD_ENV_PREFIXES):
        return True
    return parent_env.get(key) != value


def _collect_env_assignments(env: Dict[str, str]) -> List[str]:
    parent_env = os.environ
    assignments: List[str] = []
    for key, value in env.items():
        value_str = str(value)
        if _should_forward_env_var(key, value_str, parent_env):
            assignments.append(f"{key}={value_str}")
    return assignments


def maybe_wrap_command_with_cgroup(
    cmd: List[str],
    env: Dict[str, str],
    memory_max_gb: float | None,
) -> Tuple[List[str], bool]:
    """Wrap a launch command with systemd-run MemoryMax when available.

    The cgroup cap is only applied on Linux and only when a positive memory
    budget is explicitly provided. The wrapped command uses `/usr/bin/env`
    followed by selected environment overrides so the launched service sees the
    same KT/CUDA/conda context as the caller.

    On cgroup v2, MemoryMax constrains total memory usage for the cgroup,
    including anonymous memory and page cache. We also set MemorySwapMax=0 so
    the service stays RAM-bounded instead of silently shifting pressure into
    swap.
    """

    if memory_max_gb is None or memory_max_gb <= 0:
        return cmd, False
    if not sys.platform.startswith("linux"):
        return cmd, False

    systemd_run = shutil.which("systemd-run", path=env.get("PATH") or os.environ.get("PATH"))
    if not systemd_run:
        return cmd, False

    env_cmd = shutil.which("env", path=env.get("PATH") or os.environ.get("PATH")) or "/usr/bin/env"
    env_assignments = _collect_env_assignments(env)
    memory_arg = f"MemoryMax={memory_max_gb:g}G"

    wrapped_cmd = [
        systemd_run,
        "--scope",
        "--quiet",
        "--collect",
        "-p",
        memory_arg,
        "-p",
        "MemorySwapMax=0",
        env_cmd,
        *env_assignments,
        *cmd,
    ]
    return wrapped_cmd, True
