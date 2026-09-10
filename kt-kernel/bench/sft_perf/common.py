# SPDX-License-Identifier: Apache-2.0
"""Pure result/contract helpers; no training framework or kernel imports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, pattern: str = "*.py") -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob(pattern)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def window_metrics(start_tokens: int, end_tokens: int, seconds: float, steps: int) -> dict:
    if not math.isfinite(seconds) or seconds <= 0 or steps <= 0:
        raise ValueError("measurement duration and optimizer-step count must be positive")
    tokens = end_tokens - start_tokens
    if tokens <= 0:
        raise ValueError("global non-padding token counter did not advance")
    return {
        "global_non_padding_tokens": tokens,
        "wall_seconds": seconds,
        "optimizer_steps": steps,
        "tokens_per_second": tokens / seconds,
        "seconds_per_optimizer_step": seconds / steps,
    }


def validate_training_config(config: dict, warmup: int, measured: int) -> None:
    if warmup < 1 or measured < 1:
        raise ValueError("at least one warmup and measured optimizer step are required")
    required = {
        "stage": "sft",
        "do_train": True,
        "do_eval": False,
        "finetuning_type": "lora",
        "use_kt": True,
        "include_num_input_tokens_seen": "non_padding",
        "save_strategy": "no",
        "logging_nan_inf_filter": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, got {config.get(key)!r}")
    if config.get("kt_config", {}).get("kt_expert_weight_format") != "fp8":
        raise ValueError("this harness accepts native FP8 expert storage only")
    if config.get("max_steps") != warmup + measured:
        raise ValueError("max_steps must equal warmup + measured optimizer steps")
    if config.get("resume_from_checkpoint"):
        raise ValueError("each comparison must start from the same initial state")
