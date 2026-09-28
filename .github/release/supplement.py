"""Add a CPython wheel without repartitioning an accepted CUDA payload.

Compile and auditwheel-repair both native inputs from the accepted source lock
first. Only CPython-specific extensions are replaced; all runtime Python files,
ABI3 libraries, payload parts and dependency metadata must stay unchanged.
This command does not upload or authorize a release.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from carriers import LIMIT, MODULES, PLATFORM, pack, require, retag, unpack
from four_main import inspect_wheel, save_json, sha256

EXTENSION = re.compile(r"^(.*)\.cpython-(\d+)-([^.]+)\.so$")
MSCCLPP_SUPPORT = {
    "lib/libmscclpp_static.a",
    "lib/libmscclpp_nccl_static.a",
    *(
        "lib/" + name + suffix
        for name in ("mscclpp.so", "mscclpp_nccl.so")
        for suffix in ("", ".0", ".0.6.0")
    ),
}


def needed_libraries(path):
    dynamic = subprocess.check_output(["readelf", "--dynamic", str(path)], text=True)
    return set(re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic))


def extensions(root):
    result = {}
    for path in sorted(root.rglob("*.so")):
        name = path.relative_to(root).as_posix()
        match = EXTENSION.fullmatch(name)
        if match:
            key = match[1] + ".cpython-*" + "-" + match[3] + ".so"
            require(key not in result, "Duplicate extension: " + key)
            result[key] = (path, "cp" + match[2])
    return result


def manifest_values(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        require(
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name),
            "Payload manifest must contain literal assignments only",
        )
        name = node.targets[0].id
        require(name not in result, "Duplicate manifest field")
        result[name] = ast.literal_eval(node.value)
    return result


def supplement(reference, kt_wheel, sgl_wheel, lock, output, evidence):
    reference = reference.resolve(strict=True)
    evidence.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), "Output must be a new directory")
    accepted = lock["reference_wheels"]
    paths = sorted(reference.glob("*.whl"))
    require(
        {path.name for path in paths} == set(accepted), "Reference inventory mismatch"
    )
    for path in paths:
        require(sha256(path) == accepted[path.name], "Reference checksum mismatch")
    entries = {inspect_wheel(path)["name"]: path for path in paths}
    require(
        set(entries) == set(MODULES) and len(paths) == 5, "Need five accepted carriers"
    )
    for name, path in (("kt-kernel", kt_wheel), ("sgl-kernel-kt", sgl_wheel)):
        record = inspect_wheel(path)
        require(record["name"] == name, "Wrong native input")
        require(
            record["sha256"] == lock["native_wheels"][path.name],
            "Native checksum mismatch",
        )
        require(
            any(PLATFORM in tag for tag in record["tags"]),
            "Native input must be repaired",
        )

    with tempfile.TemporaryDirectory(prefix="supplement-", dir=evidence.parent) as temp:
        temp = Path(temp)
        roots = {name: temp / name for name in entries}
        for name, path in entries.items():
            unpack(path, roots[name])
        fresh_kt, fresh_sgl = temp / "fresh-kt", temp / "fresh-sgl"
        unpack(kt_wheel, fresh_kt)
        unpack(sgl_wheel, fresh_sgl)
        target = roots["kt-kernel"]
        dist = next(target.glob("*.dist-info"))
        native_dist = next(fresh_kt.glob("*.dist-info"))
        require(
            (dist / "METADATA").read_bytes() == (native_dist / "METADATA").read_bytes(),
            "KT dependency/version metadata changed",
        )
        sgl_dist = next(fresh_sgl.glob("*.dist-info"))
        require(
            (dist / "sgl-native-origin/METADATA").read_bytes()
            == (sgl_dist / "METADATA").read_bytes(),
            "SGL dependency/version metadata changed",
        )

        old = extensions(target)
        new = extensions(fresh_kt)
        sgl_extensions = extensions(fresh_sgl)
        require(not set(new) & set(sgl_extensions), "Native input path collision")
        new.update(sgl_extensions)
        require(old and set(old) == set(new), "CPython extension inventory changed")
        old_tags = {tag for _, tag in old.values()}
        tags = {tag for _, tag in new.values()}
        require(
            len(old_tags) == len(tags) == 1 and old_tags != tags,
            "Mixed or unchanged CPython ABI",
        )
        python = tags.pop()
        require(python in ("cp311", "cp312"), "Unsupported target interpreter")
        require(
            any(
                tag.startswith(python + "-" + python + "-")
                for tag in inspect_wheel(kt_wheel)["tags"]
            ),
            "KT wheel tag/extension ABI mismatch",
        )

        # Keep the original standalone C++ libraries. The new binding statically
        # links MSCCL++ and must not introduce a dependency on those old objects.
        for path, _ in new.values():
            require(
                not needed_libraries(path) & {"mscclpp.so.0", "mscclpp_nccl.so.0"},
                "New CPython binding depends on retained MSCCL++ libraries",
            )
        retained_support = {}
        for source in (fresh_kt, fresh_sgl):
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(source)
                if relative.parts[0].endswith(".dist-info"):
                    continue
                if EXTENSION.fullmatch(relative.as_posix()):
                    continue
                if path.name.endswith(".abi3.so"):
                    continue  # Retain the original stable-ABI CUDA objects.
                if source == fresh_sgl and relative.as_posix() in MSCCLPP_SUPPORT:
                    original = target / relative
                    require(original.is_file(), "Missing original MSCCL++ support file")
                    retained_support[relative.as_posix()] = sha256(original)
                    continue
                require(
                    path.suffix != ".so",
                    "Unclassified native extension: " + str(relative),
                )
                candidates = [roots[name] / relative for name in MODULES]
                matches = [candidate for candidate in candidates if candidate.is_file()]
                if source == fresh_sgl and relative.parts[0].endswith(".libs"):
                    # These are stored in the checksummed CUDA archive, not loose.
                    manifest = manifest_values(
                        target / "sgl_kernel/_payload_manifest.py"
                    )
                    require(
                        manifest["FILES"].get(relative.as_posix()) == sha256(path),
                        "Shared payload library changed: " + str(relative),
                    )
                else:
                    require(
                        len(matches) == 1 and sha256(matches[0]) == sha256(path),
                        "Runtime file changed: " + str(relative),
                    )

        payload_manifest = target / "sgl_kernel/_payload_manifest.py"
        manifest = manifest_values(payload_manifest)
        require(
            tuple(manifest["PAYLOAD_MODULES"]) == tuple(MODULES.values()),
            "Payload module mismatch",
        )
        archive_hash = hashlib.sha256()
        parts = {}
        for name, module in MODULES.items():
            part = roots[name] / module / "payload.part"
            parts[name] = sha256(part)
            with part.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    archive_hash.update(block)
        require(
            archive_hash.hexdigest() == manifest["ARCHIVE_SHA256"],
            "Mixed CUDA payload carriers",
        )
        replacements = []
        for key, (source, _) in new.items():
            previous, _ = old[key]
            relative = source.relative_to(
                fresh_kt if source.is_relative_to(fresh_kt) else fresh_sgl
            )
            destination = target / relative
            require(not destination.exists(), "Replacement destination exists")
            replacements.append(
                {
                    "old": previous.relative_to(target).as_posix(),
                    "new": relative.as_posix(),
                    "sha256": sha256(source),
                }
            )
            previous.unlink()
            shutil.copyfile(source, destination)
        require(
            {tag for _, tag in extensions(target).values()} == {python},
            "Stale CPython binary",
        )
        retag(target, python, python)
        provenance = {
            "reference_wheels": accepted,
            "native_wheels": lock["native_wheels"],
            "source_lock": lock["source_lock"],
            "replacements": replacements,
            "retained_payload_parts": parts,
            "retained_native_support": retained_support,
            "archive_sha256": manifest["ARCHIVE_SHA256"],
            "scope": "New CPython extensions; original runtime, ABI3 payload and dependencies retained.",
        }
        save_json(dist / "cpython-supplement.json", provenance)
        version = inspect_wheel(kt_wheel)["version"]
        output.mkdir()
        wheel = output / f"kt_kernel-{version}-{python}-{python}-{PLATFORM}.whl"
        pack(target, wheel)
        require(
            wheel.stat().st_size < LIMIT,
            "Supplement exceeds PyPI size limit; use a new release",
        )
        require(
            inspect_wheel(wheel)["requires_dist"]
            == inspect_wheel(kt_wheel)["requires_dist"],
            "Supplement changed dependencies",
        )
        save_json(
            evidence / "supplement.json",
            provenance
            | {
                "wheel": inspect_wheel(wheel),
                "status": "ASSEMBLED_RUNTIME_ACCEPTANCE_REQUIRED",
            },
        )
        return wheel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "kt-wheel", "sgl-wheel", "lock", "output", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    supplement(
        args.reference,
        args.kt_wheel,
        args.sgl_wheel,
        json.loads(args.lock.read_text()),
        args.output,
        args.evidence,
    )


if __name__ == "__main__":
    main()
