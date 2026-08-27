#!/usr/bin/env python3
"""Build the GLM config hotfix wheels from the verified post2 carriers.

The CUDA payload and unchanged runtime files are copied byte-for-byte from the
published post2 wheels.  Only the reviewed GLM Python source and distribution
metadata are changed.  This keeps the post2 carrier stack intact.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


POST2_HASHES = {
    "kt_kernel": "b44b68effe42f7cfbfe9bb8e07af57ef29194c3648e20a1fdd06273b42aca7ca",
    "ktransformers": "1bf1e51e115314388e41aeed5d7c380985bf67a448e8133af1e7ff81de5d9f8f",
    "sglang": "b13705d35c577530efaec915b02ba66a6bf67f88a5429f4758bb59885e723971",
    "transformers": "b8e7c1baeb19123b251dfb4f6e6c2be615abd29b6feb23113deeaef5e7175811",
}
MAX_PYPI_WHEEL_SIZE = 104_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def write_record(root: Path, dist_info: Path) -> None:
    record = dist_info / "RECORD"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != record:
            rows.append(
                (path.relative_to(root).as_posix(), record_hash(path), str(path.stat().st_size))
            )
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def pack_wheel(root: Path, output: Path) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as wheel:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                wheel.write(path, path.relative_to(root).as_posix())
    if output.stat().st_size >= MAX_PYPI_WHEEL_SIZE:
        raise RuntimeError(f"PyPI wheel is too large: {output} ({output.stat().st_size} bytes)")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one {label}: {old!r}")
    return text.replace(old, new, 1)


def unpack(input_wheel: Path, root: Path) -> None:
    with zipfile.ZipFile(input_wheel) as wheel:
        wheel.extractall(root)


def build_sglang(input_wheel: Path, source_file: Path, output_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        unpack(input_wheel, root)
        old_dist_info = root / "sglang_kt-0.7.0.post2.dist-info"
        new_dist_info = root / "sglang_kt-0.7.0.post3.dist-info"
        if not old_dist_info.is_dir():
            raise RuntimeError(f"Missing expected dist-info: {old_dist_info}")

        destination = root / "sglang/srt/models/glm5_next_dsa.py"
        old_source_hash = sha256(destination)
        new_source_hash = sha256(source_file)
        if old_source_hash == new_source_hash:
            raise RuntimeError("The reviewed GLM hotfix is not different from post2")

        payload = root / "sglang_kt_sgl_kernel_payload/payload.part"
        payload_hash = sha256(payload)
        shutil.copy2(source_file, destination)
        if sha256(payload) != payload_hash:
            raise RuntimeError("SGLang payload changed while applying the Python hotfix")

        old_dist_info.rename(new_dist_info)
        metadata = new_dist_info / "METADATA"
        metadata.write_text(
            replace_once(
                metadata.read_text(),
                "Version: 0.7.0.post2",
                "Version: 0.7.0.post3",
                label="SGLang version",
            )
        )
        write_record(root, new_dist_info)
        output = output_dir / (
            "sglang_kt-0.7.0.post3-py3-none-manylinux_2_35_x86_64.whl"
        )
        pack_wheel(root, output)
    return output


def build_ktransformers(input_wheel: Path, output_dir: Path) -> Path:
    with tempfile.TemporaryDirectory() as temp_name:
        root = Path(temp_name)
        unpack(input_wheel, root)
        old_dist_info = root / "ktransformers-0.7.0.post2.dist-info"
        new_dist_info = root / "ktransformers-0.7.0.post3.dist-info"
        if not old_dist_info.is_dir():
            raise RuntimeError(f"Missing expected dist-info: {old_dist_info}")

        payload = root / "ktransformers_sgl_kernel_payload/payload.part"
        payload_hash = sha256(payload)
        old_dist_info.rename(new_dist_info)
        metadata = new_dist_info / "METADATA"
        text = metadata.read_text()
        text = replace_once(
            text,
            "Version: 0.7.0.post2",
            "Version: 0.7.0.post3",
            label="KTransformers version",
        )
        text = replace_once(
            text,
            "Requires-Dist: sglang-kt==0.7.0.post2; extra == \"sglang\"",
            "Requires-Dist: sglang-kt==0.7.0.post3; extra == \"sglang\"",
            label="SGLang dependency",
        )
        metadata.write_text(text)
        if sha256(payload) != payload_hash:
            raise RuntimeError("KTransformers payload changed while updating metadata")

        write_record(root, new_dist_info)
        output = output_dir / (
            "ktransformers-0.7.0.post3-py3-none-manylinux_2_35_x86_64.whl"
        )
        pack_wheel(root, output)
    return output


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Unexpected frozen wheel hash for {path}: {actual} != {expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post2-dir", required=True, type=Path)
    parser.add_argument("--sglang-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sglang-sha", required=True)
    parser.add_argument("--ktransformers-sha", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    inputs = {
        "kt_kernel": args.post2_dir
        / "kt_kernel-0.7.0.post2-cp312-cp312-manylinux_2_35_x86_64.whl",
        "ktransformers": args.post2_dir
        / "ktransformers-0.7.0.post2-py3-none-manylinux_2_35_x86_64.whl",
        "sglang": args.post2_dir
        / "sglang_kt-0.7.0.post2-py3-none-manylinux_2_35_x86_64.whl",
        "transformers": args.post2_dir
        / "transformers_kt-5.6.0.post4-py3-none-manylinux_2_35_x86_64.whl",
    }
    for name, path in inputs.items():
        require_hash(path, POST2_HASHES[name])

    outputs = [
        build_sglang(inputs["sglang"], args.sglang_source, args.output_dir),
        build_ktransformers(inputs["ktransformers"], args.output_dir),
    ]
    for name in ("kt_kernel", "transformers"):
        destination = args.output_dir / inputs[name].name
        shutil.copy2(inputs[name], destination)
        require_hash(destination, POST2_HASHES[name])
        outputs.append(destination)

    manifest = {
        "source": {
            "sglang": args.sglang_sha,
            "ktransformers": args.ktransformers_sha,
        },
        "wheels": {path.name: sha256(path) for path in sorted(outputs)},
    }
    (args.output_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(outputs))
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
