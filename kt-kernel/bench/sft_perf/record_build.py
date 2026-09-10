# SPDX-License-Identifier: Apache-2.0
"""Save actual native build inputs/flags, separate from a caller's revision label."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from common import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("build records are append-only")
    source = args.source.resolve(strict=True)
    build = args.build.resolve(strict=True)
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if not line or line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        name = key.split(":", 1)[0]
        if name.startswith(("KTRANSFORMERS_", "LLAMA_", "CUDAToolkit_")) or name in (
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER",
            "CMAKE_CUDA_COMPILER",
            "CMAKE_CUDA_ARCHITECTURES",
        ):
            cache[name] = value
    inputs = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(
            part in ("__pycache__", "test", "tests", "bench", "examples") for part in relative.parts
        ):
            continue
        if path.suffix in (".h", ".hpp", ".c", ".cpp", ".cu", ".py", ".cmake") or path.name == "CMakeLists.txt":
            inputs[str(relative)] = sha256(path)
    records = {}
    for name in ("CMakeCache.txt", "CMakeFiles/kt_kernel_ext.dir/flags.make", "CMakeFiles/kt_kernel_ext.dir/link.txt"):
        path = build / name
        records[name] = {"sha256": sha256(path)}
        if name != "CMakeCache.txt":
            records[name]["contents"] = path.read_text()
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "caller_attested_source_revision": args.revision,
        "source_archive_sha256": sha256(args.archive),
        "extension_sha256": sha256(args.extension),
        "production_source_sha256_at_record_time": inputs,
        "selected_cmake_cache": cache,
        "generated_build_files": records,
        "scope": "build context evidence; external/system dependencies are not a hermetic source archive",
    }
    write_json(args.output, result)
    print(json.dumps({"extension_sha256": result["extension_sha256"], "source_files": len(inputs)}))


if __name__ == "__main__":
    main()
