"""Exercise carrier assembly with tiny synthetic wheels, not CUDA execution."""

import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carriers
from four_main import inspect_wheel


def raw_wheels(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for package in (*carriers.MODULES, "sgl-kernel-kt"):
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
            for name in (
                "deep_gemm/__init__.py",
                "triton_kernels/__init__.py",
                "flash_attn_origin/__init__.py",
                "include/fixture.h",
                "lib/libfixture.so",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture-data" * 20)
            (dist / "LICENSE").write_text("SGL license fixture")
        path = raw / f"{package.replace('-', '_')}-1.0-{tag}.whl"
        carriers.pack(root, path)
    return raw


def test_fresh_carriers_assign_sglang_runtime_without_shadowing_standalone_kt(
    tmp_path, monkeypatch
):
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
        names = wheel.namelist()
        for namespace in (
            "sgl_kernel",
            "deep_gemm",
            "triton_kernels",
            "flash_attn_origin",
            "include",
            "lib",
        ):
            assert not any(
                name == namespace or name.startswith(namespace + "/") for name in names
            )
        assert "sgl_kernel_kt_payload_core/payload.part" in names
    with zipfile.ZipFile(next(output.glob("sglang_kt-*.whl"))) as wheel:
        for name in (
            "sgl_kernel/payload_runtime.py",
            "deep_gemm/__init__.py",
            "triton_kernels/__init__.py",
            "flash_attn_origin/__init__.py",
            "include/fixture.h",
            "lib/libfixture.so",
        ):
            assert wheel.read(name) == b"fixture-data" * 20
        assert any(
            name.endswith("sgl-native-origin/LICENSE") for name in wheel.namelist()
        )
        assert "sgl_kernel/flash_ops.abi3.so" not in wheel.namelist()
        assert "sgl_kernel/_payload_manifest.py" in wheel.namelist()
    with zipfile.ZipFile(next(output.glob("ktransformers-*.whl"))) as wheel:
        assert "sgl_kernel/sm90/common_ops.abi3.so" in wheel.namelist()
    with zipfile.ZipFile(next(output.glob("accelerate_kt-*.whl"))) as wheel:
        assert "accelerate_kt_sgl_kernel_payload/payload.part" in wheel.namelist()
    # Every final RECORD is independently checked, including regenerated payloads.
    for index, path in enumerate(output.iterdir()):
        carriers.unpack(path, tmp_path / f"verify-{index}")


@pytest.mark.parametrize("scheme", ["purelib", "platlib"])
@pytest.mark.parametrize("namespace", sorted(carriers.SGLANG_RUNTIME_ROOTS))
def test_assembly_rejects_sglang_files_in_kt_data_scheme(
    tmp_path, monkeypatch, scheme, namespace
):
    raw = raw_wheels(tmp_path)
    root = tmp_path / "kt-kernel"
    misplaced = root / f"kt_kernel-1.0.data/{scheme}/{namespace}/fixture.py"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_text("# pip installs this outside the KT namespace\n")
    carriers.pack(root, next(raw.glob("kt_kernel-*.whl")))
    monkeypatch.setattr(carriers, "binary_evidence", lambda roots: {})
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    with pytest.raises(ValueError, match="standalone kt-kernel"):
        carriers.assemble(raw, tmp_path / "final", evidence)


@pytest.mark.parametrize("scheme", ["purelib", "platlib"])
def test_ownership_accepts_private_kt_data_scheme(tmp_path, scheme):
    private = tmp_path / f"kt_kernel-1.0.data/{scheme}/kt_kernel/private.py"
    private.parent.mkdir(parents=True)
    private.write_text("# KT's own module\n")
    carriers.verify_runtime_ownership("kt-kernel", tmp_path)


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


def test_payload_preserves_wheel_relative_libraries(tmp_path):
    raw_wheels(tmp_path)
    root = tmp_path / "sgl-kernel-kt"
    library = root / "sgl_kernel_kt.libs/libnuma.so.1"
    library.parent.mkdir()
    library.write_bytes(b"dependency")
    target = tmp_path / "payload.tar.gz"
    hashes = carriers.archive_payload(root, target)
    assert set(hashes) == {"sgl_kernel/" + name for name in carriers.LARGE} | {
        "sgl_kernel_kt.libs/libnuma.so.1"
    }
    with tarfile.open(target) as archive:
        assert set(archive.getnames()) == set(hashes)


def test_embedded_metadata_does_not_shadow_wheel_metadata(tmp_path):
    raw_wheels(tmp_path)
    root = tmp_path / "accelerate-kt"
    nested = root / "accelerate_kt/vendor/example.dist-info"
    nested.mkdir(parents=True)
    for name in ("METADATA", "RECORD"):
        (nested / name).write_text("vendored metadata")
    wheel = tmp_path / "accelerate_kt-1.0-py3-none-any.whl"
    carriers.pack(root, wheel)
    assert inspect_wheel(wheel)["name"] == "accelerate-kt"
    carriers.unpack(wheel, tmp_path / "unpacked")


@pytest.mark.parametrize("kt_cuda", [False, True])
def test_cpu_only_kt_is_accepted_without_weakening_cuda_audit(
    tmp_path, monkeypatch, kt_cuda
):
    kt, sgl = tmp_path / "kt", tmp_path / "sgl"
    kt.mkdir()
    (sgl / "sgl_kernel/sm100").mkdir(parents=True)
    (kt / "extension.so").touch()
    (sgl / "sgl_kernel/sm100/common_ops.abi3.so").touch()
    monkeypatch.setattr(
        carriers.subprocess,
        "check_output",
        lambda args, **kw: (
            ".nv_fatbin" if kt_cuda or sgl in Path(args[-1]).parents else ""
        ),
    )
    monkeypatch.setattr(
        carriers.subprocess,
        "run",
        lambda args, **kw: SimpleNamespace(
            stdout=(
                "sm_80 sm_86 sm_89 sm_90 sm_120"
                if sgl in Path(args[-1]).parents
                else "sm_80"
            )
        ),
    )
    if kt_cuda:
        with pytest.raises(ValueError, match="KT CUDA extension"):
            carriers.binary_evidence({"kt-kernel": kt, "sgl-kernel-kt": sgl})
    else:
        carriers.binary_evidence({"kt-kernel": kt, "sgl-kernel-kt": sgl})
