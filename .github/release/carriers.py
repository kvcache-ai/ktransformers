"""Assemble fresh five-project carriers without version changes or source overlays.

Only WHEEL/RECORD and the generated CUDA payload manifest are rewritten. Runtime
files (including SM90 and licenses) are retained. Reject oversized wheels instead
of deleting architectures. Native inputs must first pass auditwheel repair.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import io
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from four_main import inspect_wheel, save_json, sha256

LIMIT = 104_000_000
PLATFORM = "manylinux_2_35_x86_64"
MODULES = {
    "kt-kernel": "sgl_kernel_kt_payload_core",
    "transformers-kt": "transformers_kt_sgl_kernel_payload",
    "sglang-kt": "sglang_kt_sgl_kernel_payload",
    "ktransformers": "ktransformers_sgl_kernel_payload",
    "accelerate-kt": "accelerate_kt_sgl_kernel_payload",
}
LARGE = ("flash_ops.abi3.so", "sm100/common_ops.abi3.so")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unpack(wheel, root):
    root.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP entries")
        records = [
            name
            for name in names
            if name.endswith(".dist-info/RECORD")
            and len(PurePosixPath(name).parts) == 2
        ]
        require(len(records) == 1, "Require one wheel RECORD")
        rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
        record = {row[0]: row[1:] for row in rows}
        require(len(record) == len(rows), "Duplicate RECORD entries")
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and "\\" not in info.filename,
                "Unsafe wheel path",
            )
            require(not stat.S_ISLNK(info.external_attr >> 16), "Symlink in wheel")
            if info.is_dir():
                continue
            require(info.filename in record, "Wheel file not recorded")
            target = root / info.filename
            require(
                path.as_posix() == info.filename and not target.exists(),
                "Noncanonical or colliding wheel path",
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if info.filename != records[0]:
                expected = (
                    "sha256="
                    + base64.urlsafe_b64encode(bytes.fromhex(sha256(target)))
                    .rstrip(b"=")
                    .decode()
                )
                require(
                    record[info.filename] == [expected, str(target.stat().st_size)],
                    "Wheel RECORD mismatch",
                )
        require(
            set(record) == {name for name in names if not name.endswith("/")},
            "Unexpected RECORD entries",
        )


def pack(root, output):
    dist = next(root.glob("*.dist-info"))
    record = dist / "RECORD"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != record:
            encoded = (
                base64.urlsafe_b64encode(bytes.fromhex(sha256(path)))
                .rstrip(b"=")
                .decode()
            )
            rows.append(
                (
                    path.relative_to(root).as_posix(),
                    "sha256=" + encoded,
                    path.stat().st_size,
                )
            )
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(
                    path.relative_to(root).as_posix(), (2020, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with (
                    path.open("rb") as source,
                    archive.open(info, "w", force_zip64=True) as target,
                ):
                    shutil.copyfileobj(source, target)


def retag(root, python, abi):
    metadata = next(root.glob("*.dist-info/WHEEL"))
    lines = [
        line
        for line in metadata.read_text().splitlines()
        if line and not line.startswith(("Tag:", "Root-Is-Purelib:"))
    ]
    metadata.write_text(
        "\n".join(lines + ["Root-Is-Purelib: false", f"Tag: {python}-{abi}-{PLATFORM}"])
        + "\n"
    )


def archive_payload(sgl, output):
    hashes = {}
    paths = [sgl / "sgl_kernel" / name for name in LARGE]
    for directory in sorted(sgl.glob("*.libs")):
        paths.extend(path for path in sorted(directory.rglob("*")) if path.is_file())
    with (
        output.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for source in paths:
                name = source.relative_to(sgl).as_posix()
                hashes[name] = sha256(source)
                info = archive.gettarinfo(str(source), arcname=name)
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
    return hashes


def binary_evidence(roots):
    evidence = {}
    for package, root in roots.items():
        for path in sorted(root.rglob("*.so*")):
            if not path.is_file():
                continue
            sections = subprocess.check_output(
                ["readelf", "--wide", "--sections", str(path)], text=True
            )
            if ".nv_fatbin" not in sections and ".nvFatBinSegment" not in sections:
                # Most KT CPU variants and repaired system libraries have no CUDA code.
                continue
            result = subprocess.run(
                ["cuobjdump", "--list-elf", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
            arches = sorted(set(re.findall(r"sm_([0-9]+[af]?)", result.stdout)))
            evidence[package + "/" + path.relative_to(root).as_posix()] = {
                "sha256": sha256(path),
                "sass": arches,
            }
    common = evidence.get("sgl-kernel-kt/sgl_kernel/sm100/common_ops.abi3.so", {})
    required = {80, 86, 89, 90, 120}
    normalize = lambda values: {int(re.match(r"[0-9]+", value)[0]) for value in values}
    require(
        required <= normalize(common.get("sass", [])),
        "SGL common_ops is missing required CUDA SASS architectures",
    )
    kt = [entry for name, entry in evidence.items() if name.startswith("kt-kernel/")]
    require(
        not kt or any(required <= normalize(entry["sass"]) for entry in kt),
        "KT CUDA extension is missing required SASS architectures",
    )
    return evidence


def assemble(raw, output, evidence_dir):
    output.mkdir(exist_ok=False)
    entries = [
        inspect_wheel(path) | {"path": path} for path in sorted(raw.glob("*.whl"))
    ]
    by_name = {entry["name"]: entry for entry in entries}
    expected = set(MODULES) | {"sgl-kernel-kt"}
    require(
        set(by_name) == expected and len(entries) == len(expected),
        "Need six fresh inputs",
    )
    for package in ("kt-kernel", "sgl-kernel-kt"):
        require(
            any(PLATFORM in tag for tag in by_name[package]["tags"]),
            "Native input must be auditwheel-repaired for " + PLATFORM,
        )
    require(
        not by_name["sgl-kernel-kt"]["requires_dist"],
        "SGL native dependencies must be explicitly carried by main package metadata",
    )
    with tempfile.TemporaryDirectory(
        prefix="carrier-", dir=evidence_dir.parent
    ) as temp:
        temp = Path(temp)
        roots = {name: temp / name for name in by_name}
        for name, entry in by_name.items():
            unpack(entry["path"], roots[name])
        save_json(evidence_dir / "cuda-binaries.json", binary_evidence(roots))
        sgl = roots["sgl-kernel-kt"]
        for name in ("payload_runtime.py", "load_utils.py", "flash_attn.py"):
            require(
                (sgl / "sgl_kernel" / name).is_file(),
                "Locked SGL main lacks the carrier loader: " + name,
            )
        archive = temp / "payload.tar.gz"
        hashes = archive_payload(sgl, archive)
        # Only the two objects supported by main's lazy loader are externalized.
        # No SM90 deletion or copied Python implementation from another checkout.
        for name in LARGE:
            (sgl / "sgl_kernel" / name).unlink()
        manifest = sgl / "sgl_kernel/_payload_manifest.py"
        require(
            not manifest.exists(), "Raw SGL input already contains a payload manifest"
        )
        manifest.write_text(
            f"VERSION = {by_name['sgl-kernel-kt']['version']!r}\n"
            f"ARCHIVE_SHA256 = {sha256(archive)!r}\n"
            f"PAYLOAD_MODULES = {tuple(MODULES.values())!r}\nFILES = {hashes!r}\n"
            f"BINARIES = {dict((name, 'sgl_kernel/' + name) for name in LARGE)!r}\n"
        )
        kt = roots["kt-kernel"]
        kt_dist = next(kt.glob("*.dist-info"))
        # Keep Hopper's directly loaded object, but use the lightweight umbrella
        # carrier's size budget. Pip installs these non-overlapping paths into the
        # same sgl_kernel directory; no loader changes or namespace .pth hacks.
        direct_files = []
        sm90 = sgl / "sgl_kernel/sm90"
        if sm90.exists():
            direct_files = [
                path.relative_to(sgl).as_posix()
                for path in sm90.rglob("*")
                if path.is_file()
            ]
            destination = roots["ktransformers"] / "sgl_kernel/sm90"
            require(not destination.exists(), "SM90 carrier path collision")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sm90), destination)
        for path in sorted(sgl.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(sgl)
            if relative.parts[0].endswith(".dist-info"):
                # Preserve the raw native distribution's licenses/metadata as
                # provenance, without creating a conflicting installed package.
                target = kt_dist / "sgl-native-origin" / Path(*relative.parts[1:])
            else:
                require(
                    not relative.parts[0].endswith(".data"),
                    "Unsupported SGL wheel .data layout",
                )
                target = kt / relative
            require(not target.exists(), "Carrier path collision: " + str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        filenames = {}
        capacities = {}
        for name in MODULES:
            root = roots[name]
            python, abi = ("cp312", "cp312") if name == "kt-kernel" else ("py3", "none")
            if name == "kt-kernel":
                require(
                    any(
                        tag.startswith("cp312-cp312-") for tag in by_name[name]["tags"]
                    ),
                    "Initial release supports CPython 3.12 only",
                )
            retag(root, python, abi)
            filename = f"{name.replace('-', '_')}-{by_name[name]['version']}-{python}-{abi}-{PLATFORM}.whl"
            filenames[name] = filename
            baseline = temp / filename
            pack(root, baseline)
            capacities[name] = LIMIT - baseline.stat().st_size - 1_000_000
            require(
                capacities[name] > 0, "Base wheel exceeds carrier capacity: " + name
            )
        require(
            sum(capacities.values()) >= archive.stat().st_size,
            "CUDA archive exceeds five-project size budget; do not drop architectures",
        )
        parts = {}
        remaining = archive.stat().st_size
        with archive.open("rb") as source:
            for index, (name, module) in enumerate(MODULES.items()):
                # Keep a nonempty piece for every loader module, even for tiny fixtures.
                size = min(capacities[name], remaining - (len(MODULES) - index - 1))
                require(size > 0, "Invalid payload partition")
                directory = roots[name] / module
                directory.mkdir()
                (directory / "__init__.py").write_text(
                    "# Generated checksummed CUDA payload carrier.\n"
                )
                part = directory / "payload.part"
                with part.open("wb") as target:
                    pending = size
                    while pending:
                        block = source.read(min(pending, 8 * 1024 * 1024))
                        require(block, "Truncated payload archive")
                        target.write(block)
                        pending -= len(block)
                remaining -= size
                parts[name] = {"bytes": size, "sha256": sha256(part)}
                final = output / filenames[name]
                pack(roots[name], final)
                require(
                    final.stat().st_size < LIMIT,
                    "Final wheel exceeds PyPI size limit: " + name,
                )
                require(
                    inspect_wheel(final)["requires_dist"]
                    == by_name[name]["requires_dist"],
                    "Carrier changed dependency metadata",
                )
            require(remaining == 0 and not source.read(1), "Incomplete payload split")
        save_json(
            evidence_dir / "assembly.json",
            {
                "raw_wheels": [
                    {k: v for k, v in entry.items() if k != "path"} for entry in entries
                ],
                "final_wheels": [
                    inspect_wheel(path) for path in sorted(output.glob("*.whl"))
                ],
                "archive_sha256": sha256(archive),
                "payload_files": hashes,
                "parts": parts,
                "direct_native_carriers": {"ktransformers": direct_files},
                "runtime_overlays": [],
                "version_overrides": [],
                "dependency_overrides": [],
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    assemble(args.raw, args.output, args.evidence)


if __name__ == "__main__":
    main()
