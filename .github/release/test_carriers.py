"""Exercise carrier assembly with tiny synthetic wheels, not CUDA execution."""

import sys
import tarfile
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carriers
from four_main import inspect_wheel


def raw_wheels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for package in (*carriers.MODULES, "accelerate-kt", "sgl-kernel-kt"):
        root = tmp_path / package
        root.mkdir()
        dist = root / (package.replace("-", "_") + "-1.0.dist-info")
        dist.mkdir()
        (dist / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package}\nVersion: 1.0\n"
        )
        tag = (
            "cp312-cp312-" + carriers.PLATFORM
            if package in ("kt-kernel", "sgl-kernel-kt")
            else "py3-none-any"
        )
        (dist / "WHEEL").write_text(
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {tag}\n"
        )
        module = root / package.replace("-", "_")
        module.mkdir()
        (module / "__init__.py").write_text("VALUE = 1\n")
        if package == "sgl-kernel-kt":
            module = root / "sgl_kernel"
            module.mkdir()
            for name in (
                *carriers.LARGE,
                "sm90/common_ops.abi3.so",
                "payload_runtime.py",
                "flash_attn.py",
                "load_utils.py",
            ):
                path = module / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture-data" * 20)
            (dist / "LICENSE").write_text("SGL license fixture")
        path = raw / f"{package.replace('-', '_')}-1.0-{tag}.whl"
        carriers.pack(root, path)
    return raw


def test_fresh_carriers_preserve_runtime_versions_and_sm90(tmp_path, monkeypatch):
    raw = raw_wheels(tmp_path)
    monkeypatch.setattr(carriers, "binary_evidence", lambda roots: {"fixture": True})
    monkeypatch.setattr(carriers.subprocess, "check_output", lambda *args, **kwargs: "")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    output = tmp_path / "final"
    carriers.assemble(raw, output, evidence)
    entries = [inspect_wheel(path) for path in output.iterdir()]
    assert len(entries) == 5
    assert {entry["version"] for entry in entries} == {"1.0"}
    kt = next(output.glob("kt_kernel-*.whl"))
    with zipfile.ZipFile(kt) as wheel:
        assert wheel.read("sgl_kernel/payload_runtime.py") == b"fixture-data" * 20
        assert any(
            name.endswith("sgl-native-origin/LICENSE") for name in wheel.namelist()
        )
        assert "sgl_kernel/flash_ops.abi3.so" not in wheel.namelist()
        assert "sgl_kernel/_payload_manifest.py" in wheel.namelist()
    with zipfile.ZipFile(next(output.glob("ktransformers-*.whl"))) as wheel:
        assert "sgl_kernel/sm90/common_ops.abi3.so" in wheel.namelist()
    # Every final RECORD is independently checked, including regenerated payloads.
    for index, path in enumerate(output.iterdir()):
        carriers.unpack(path, tmp_path / f"verify-{index}")


def test_tampered_raw_wheel_is_rejected(tmp_path):
    raw = raw_wheels(tmp_path)
    path = next(raw.glob("accelerate*.whl"))
    with zipfile.ZipFile(path, "a") as wheel:
        wheel.writestr("unrecorded.py", "bad")
    with pytest.raises(ValueError, match="not recorded"):
        carriers.unpack(path, tmp_path / "unpacked")


def test_size_limit_fails_without_dropping_architectures(tmp_path, monkeypatch):
    raw = raw_wheels(tmp_path)
    monkeypatch.setattr(carriers, "binary_evidence", lambda roots: {})
    monkeypatch.setattr(carriers.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(carriers, "LIMIT", 100)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(ValueError, match="capacity"):
        carriers.assemble(raw, tmp_path / "final", evidence)


def test_lazy_libraries_are_relocated_and_hashed_without_python_overlay(tmp_path, monkeypatch):
    sgl = tmp_path / "sgl"
    library = sgl / "sgl_kernel_kt.libs/libnuma-test.so.1"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"small-system-library")
    for name in carriers.LARGE:
        path = sgl / "sgl_kernel" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-elf")
    calls = []
    monkeypatch.setattr(carriers.subprocess, "check_output", lambda *a, **kw: "Shared library: [libnuma-test.so.1]")
    monkeypatch.setattr(carriers.subprocess, "run", lambda command, **kw: calls.append(command))
    libraries, changes = carriers.relocate_lazy_dependencies(sgl)
    assert {call[2] for call in calls} == {"$ORIGIN/.libs", "$ORIGIN/../.libs"}
    assert len(changes) == 2
    archive = tmp_path / "payload.tar.gz"
    hashes = carriers.archive_payload(sgl, archive, libraries)
    assert set(hashes) == {*carriers.LARGE, ".libs/libnuma-test.so.1"}
    with tarfile.open(archive, "r:gz") as bundle:
        assert set(bundle.getnames()) == set(hashes)
        assert bundle.extractfile(".libs/libnuma-test.so.1").read() == library.read_bytes()


def test_cuda_vendor_libraries_must_not_be_bundled_again(tmp_path):
    library = tmp_path / "sgl_kernel_kt.libs/libcublas-bad.so.12"
    library.parent.mkdir()
    library.touch()
    with pytest.raises(ValueError, match="NVIDIA libraries"):
        carriers.relocate_lazy_dependencies(tmp_path)


@pytest.mark.skipif(not shutil.which("gcc") or not shutil.which("patchelf"), reason="Native ELF packaging test needs gcc and patchelf")
def test_real_elf_dependency_loads_after_cache_relocation(tmp_path):
    sgl = tmp_path / "site-packages"
    libs = sgl / "sgl_kernel_kt.libs"
    libs.mkdir(parents=True)
    dependency = tmp_path / "dependency.c"
    dependency.write_text("int kt_dependency_value(void) { return 42; }\n")
    consumer = tmp_path / "consumer.c"
    consumer.write_text("extern int kt_dependency_value(void); int kt_fixture_value(void) { return kt_dependency_value(); }\n")
    soname = "libkt_fixture_dependency-private.so.1"
    subprocess.run(["gcc", "-shared", "-fPIC", str(dependency), "-Wl,-soname," + soname, "-o", str(libs / soname)], check=True)
    for name in carriers.LARGE:
        target = sgl / "sgl_kernel" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        rpath = "$ORIGIN/../../sgl_kernel_kt.libs" if "/" in name else "$ORIGIN/../sgl_kernel_kt.libs"
        subprocess.run(["gcc", "-shared", "-fPIC", str(consumer), "-L" + str(libs), "-l:" + soname, "-Wl,-rpath," + rpath, "-o", str(target)], check=True)
    broken = tmp_path / "broken-cache"
    shutil.copytree(sgl / "sgl_kernel", broken)
    probe = "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).kt_fixture_value() == 42"
    failed = subprocess.run([sys.executable, "-c", probe, str(broken / carriers.LARGE[1])], capture_output=True)
    assert failed.returncode != 0
    libraries, changes = carriers.relocate_lazy_dependencies(sgl)
    assert len(changes) == 2
    archive = tmp_path / "payload.tar.gz"
    hashes = carriers.archive_payload(sgl, archive, libraries)
    cached = tmp_path / "valid-cache"
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(cached, filter="data")
    for name, expected in hashes.items():
        assert carriers.sha256(cached / name) == expected
    # Separate interpreters prevent preloaded libraries from hiding missing files.
    for name in carriers.LARGE:
        subprocess.run([sys.executable, "-c", probe, str(cached / name)], check=True)
