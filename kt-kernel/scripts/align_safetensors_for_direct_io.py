#!/usr/bin/env python3
"""Pad safetensors headers so tensor payloads are aligned for O_DIRECT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


def iter_safetensors(path: Path):
    if path.is_file():
        yield path
        return
    yield from sorted(path.rglob("*.safetensors"))


def read_header(path: Path) -> tuple[int, bytes, dict]:
    with path.open("rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise RuntimeError(f"{path}: missing safetensors header size")
        header_size = struct.unpack("<Q", raw_size)[0]
        header = f.read(header_size)
        if len(header) != header_size:
            raise RuntimeError(f"{path}: truncated safetensors header")
    return header_size, header, json.loads(header)


def validate_tensor_sizes(path: Path, metadata: dict, alignment: int) -> None:
    for key, info in metadata.items():
        if key == "__metadata__":
            continue
        start, end = info["data_offsets"]
        size = int(end) - int(start)
        if size <= 0:
            raise RuntimeError(f"{path}: tensor {key} has invalid size {size}")
        if size % alignment != 0:
            raise RuntimeError(
                f"{path}: tensor {key} size={size} is not {alignment}-byte aligned; "
                "header padding alone cannot make this file usable with O_DIRECT"
            )


def align_file(path: Path, alignment: int, backup_suffix: str, dry_run: bool, strict_all_tensors: bool) -> str:
    header_size, header, metadata = read_header(path)
    if strict_all_tensors:
        validate_tensor_sizes(path, metadata, alignment)

    current_base = 8 + header_size
    pad = (-current_base) % alignment
    if pad == 0:
        return "already-aligned"

    new_header = header + (b" " * pad)
    new_header_size = len(new_header)
    temp_path = path.with_name(path.name + ".directio.tmp")
    backup_path = path.with_name(path.name + backup_suffix) if backup_suffix else None

    if dry_run:
        return f"would-pad {pad}"

    with path.open("rb") as src, temp_path.open("wb") as dst:
        src.seek(8 + header_size)
        dst.write(struct.pack("<Q", new_header_size))
        dst.write(new_header)
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

    if backup_path is not None:
        if backup_path.exists():
            backup_path.unlink()
        path.rename(backup_path)
    else:
        path.unlink()
    temp_path.rename(path)
    return f"padded {pad}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Safetensors file or directory to rewrite.")
    parser.add_argument("--alignment", type=int, default=512)
    parser.add_argument("--backup-suffix", default=".unaligned")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-all-tensors",
        action="store_true",
        help="Also require every tensor payload size to be aligned. By default only the data section base is aligned; "
        "expert slot alignment is validated by SafeTensorLoader when io_uring direct I/O is used.",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise FileNotFoundError(root)

    changed = 0
    for file_path in iter_safetensors(root):
        status = align_file(file_path, args.alignment, args.backup_suffix, args.dry_run, args.strict_all_tensors)
        print(f"{file_path}: {status}")
        if status.startswith("padded") or status.startswith("would-pad"):
            changed += 1
    print(f"checked={sum(1 for _ in iter_safetensors(root))} changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
