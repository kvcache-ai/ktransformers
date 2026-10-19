#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned SGLang source without modifying its submodule checkout.

Install the returned directory with ``python -m pip install --no-deps DIR/python``
inside an already configured KT/SGLang environment. The exported dependency
metadata follows this checkout's kt-kernel version; runtime source is unchanged.
"""

import argparse
import io
import runpy
import tomllib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

PINNED_SGLANG = "541ddc37cbc92c60dc748db5ff1a2aad0b069a80"
REPO_ROOT = Path(__file__).resolve().parents[2]


def prepare(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"Output already exists; select a new directory: {output}")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PINNED_SGLANG:
        raise ValueError(
            f"Expected pinned SGLang {PINNED_SGLANG}, got {revision}; " "initialize the repository submodules first"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Export tracked files from the exact pin. Untracked/local source edits are
    # neither copied nor overwritten. The temporary destination is isolated from
    # enclosing git repositories.
    archive = subprocess.check_output(["git", "-C", str(source), "archive", "--format=tar", revision])
    with tempfile.TemporaryDirectory(prefix=".kt-sglang-", dir=output.parent) as tmp:
        staging = Path(tmp) / "sglang"
        staging.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(staging, filter="data")
        # The pinned submodule can still depend on an older released kernel.
        # Keep the exact source pin while aligning only that package constraint
        # to the kernel built from this checkout. Do not relax Torch/other pins.
        kernel_version = runpy.run_path(str(REPO_ROOT / "version.py"))["__version__"]
        project = staging / "python/pyproject.toml"
        text = project.read_text()
        requirements = tomllib.loads(text)["project"]["dependencies"]
        kernel_requirements = [r for r in requirements if r.startswith("kt-kernel==")]
        if len(kernel_requirements) != 1 or text.count('"' + kernel_requirements[0] + '"') != 1:
            raise ValueError("Expected exactly one pinned kt-kernel dependency")
        project.write_text(text.replace('"' + kernel_requirements[0] + '"', '"kt-kernel==' + kernel_version + '"'))
        manifest = {
            "source_commit": revision,
            "kernel_version": kernel_version,
        }
        (staging / ".kt-sglang-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = prepare(REPO_ROOT / "third_party/sglang", args.output)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    print(result)


if __name__ == "__main__":
    main()
