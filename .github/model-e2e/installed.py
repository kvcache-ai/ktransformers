"""Record the actual wheel-owned imports from the fresh test interpreter."""

import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

from contracts import digest, require, write_json


def main():
    require(sys.prefix != sys.base_prefix, "Test interpreter is not a venv")
    for name in ("transformers", "accelerate", "sglang", "sgl-kernel", "sgl-kernel-kt"):
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        raise ValueError(f"Conflicting upstream distribution: {name}")
    result = {}
    for package, module in {
        "ktransformers": "ktransformers",
        "kt-kernel": "kt_kernel",
        "sglang-kt": "sglang",
        "transformers-kt": "transformers",
        "accelerate-kt": "accelerate",
    }.items():
        dist = metadata.distribution(package)
        imported = importlib.import_module(module)
        path = Path(imported.__file__).resolve()
        require(
            path.is_relative_to(Path(sys.prefix).resolve()),
            "Imported code is outside the clean venv",
        )
        files = {Path(dist.locate_file(entry)).resolve() for entry in dist.files or []}
        require(path in files, f"Import is not owned by {package}")
        # Hash installed Python/native payload files as well as the entrypoint;
        # tooling installation must not silently overwrite shared namespaces.
        payloads = {
            str(path.relative_to(sys.prefix)): digest(path)
            for path in sorted(files)
            if path.is_file()
            and path.suffix in (".py", ".so")
            and path.is_relative_to(sys.prefix)
        }
        result[package] = {
            "version": dist.version,
            "import_path": str(path),
            "payload_sha256": payloads,
        }
    from accelerate.utils.dataclasses import KTransformersPlugin  # noqa: F401

    write_json(sys.argv[1], result)


if __name__ == "__main__":
    main()
