"""Repackage approved PEFT/TRL dependencies without changing runtime files."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
import json
from pathlib import Path
import tempfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

from carriers import pack, require, unpack
from four_main import inspect_wheel, save_json, sha256


def repack(source, output, *, expected_sha256, local_tag, versions):
    source, output = Path(source), Path(output)
    require(sha256(source) == expected_sha256, "Unexpected upstream wheel SHA256")
    identity = inspect_wheel(source)
    name = identity["name"]
    require(name in {"peft", "trl"}, "Only PEFT and TRL repackaging is approved")
    upstream_version = Version(identity["version"])
    require(upstream_version.local is None, "Input must be an upstream public version")
    version = Version(f"{upstream_version}+{local_tag}")
    require(version.local == local_tag, "Use a normalized, nonempty PEP 440 local tag")
    filename = source.name.replace(f"-{upstream_version}-", f"-{version}-", 1)
    destination = output / filename
    require(not destination.exists(), "Refusing to overwrite an existing wheel")

    with tempfile.TemporaryDirectory(prefix="kt-training-tool-") as temporary:
        root = Path(temporary) / "wheel"
        unpack(source, root)
        dist = next(root.glob("*.dist-info"))
        old_dist_name = dist.name
        protected = {
            path.relative_to(root).as_posix(): sha256(path)
            for path in root.rglob("*")
            if path.is_file() and path not in {dist / "METADATA", dist / "RECORD"}
        }
        metadata = BytesParser(policy=policy.compat32).parsebytes(
            (dist / "METADATA").read_bytes()
        )
        requirements = metadata.get_all("Requires-Dist", [])
        del metadata["Requires-Dist"]
        changes = []
        for original in requirements:
            requirement = Requirement(original)
            dependency = canonicalize_name(requirement.name)
            if dependency in {"transformers", "accelerate"}:
                require(
                    requirement.url is None,
                    "Direct upstream dependency URLs require separate review",
                )
                require(
                    requirement.specifier.contains(
                        versions[dependency], prereleases=True
                    ),
                    f"Candidate {dependency} does not satisfy {original}",
                )
                requirement.name = dependency + "-kt"
                replacement = str(requirement)
                changes.append({"before": original, "after": replacement})
            else:
                replacement = original
            metadata["Requires-Dist"] = replacement
        require(
            {
                canonicalize_name(Requirement(change["before"]).name)
                for change in changes
            }
            == {"transformers", "accelerate"},
            "Expected both upstream dependency declarations",
        )
        metadata.replace_header("Version", str(version))
        (dist / "METADATA").write_bytes(metadata.as_bytes())
        new_dist = root / f"{name.replace('-', '_')}-{version}.dist-info"
        dist.rename(new_dist)
        for original, digest in protected.items():
            relocated = original.replace(old_dist_name + "/", new_dist.name + "/", 1)
            require(sha256(root / relocated) == digest, "Protected wheel file changed")
        provenance = {
            "schema_version": 1,
            "upstream_filename": source.name,
            "upstream_sha256": expected_sha256,
            "name": name,
            "version": str(version),
            "candidate_versions": versions,
            "dependency_changes": changes,
            "runtime_files_modified": False,
            "protected_source_files": protected,
        }
        save_json(new_dist / "KT-TOOLING-PROVENANCE.json", provenance)
        output.mkdir(parents=True, exist_ok=True)
        pack(root, destination)
        unpack(destination, Path(temporary) / "verified")
    return provenance | {"filename": filename, "sha256": sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--local-tag", required=True)
    parser.add_argument("--transformers-version", required=True)
    parser.add_argument("--accelerate-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = repack(
        args.source,
        args.output,
        expected_sha256=args.sha256,
        local_tag=args.local_tag,
        versions={
            "transformers": args.transformers_version,
            "accelerate": args.accelerate_version,
        },
    )
    save_json(args.output / (result["filename"] + ".provenance.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
