"""Exercise carrier assembly with tiny synthetic wheels, not CUDA execution."""

import sys
from types import SimpleNamespace
import zipfile
from pathlib import Path

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
    with zipfile.ZipFile(next(output.glob("accelerate_kt-*.whl"))) as wheel:
        assert wheel.read("accelerate_kt/__init__.py") == b"VALUE = 1\n"
        assert "accelerate_kt_sgl_kernel_payload/payload.part" in wheel.namelist()
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


def test_embedded_dist_info_is_data_not_installed_metadata(tmp_path):
    raw = raw_wheels(tmp_path)
    root = tmp_path / "accelerate-kt"
    nested = root / "accelerate_kt/vendor/upstream-9.9.dist-info"
    nested.mkdir(parents=True)
    (nested / "METADATA").write_text("Name: upstream\nVersion: 9.9\n")
    (nested / "RECORD").write_text("upstream metadata only\n")
    wheel = next(raw.glob("accelerate*.whl"))
    carriers.pack(root, wheel)
    assert inspect_wheel(wheel)["name"] == "accelerate-kt"
    carriers.unpack(wheel, tmp_path / "checked")


def native_fixture(tmp_path, monkeypatch, *, cuda_in_kt=False):
    roots = {name: tmp_path / name for name in ("kt-kernel", "sgl-kernel-kt")}
    for variant in carriers.CPU_VARIANTS:
        path = (
            roots["kt-kernel"]
            / f"kt_kernel/_kt_kernel_ext_{variant}.cpython-312-x86_64-linux-gnu.so"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"CPU fixture")
    sgl = roots["sgl-kernel-kt"] / "sgl_kernel/sm100/common_ops.abi3.so"
    sgl.parent.mkdir(parents=True)
    sgl.write_bytes(b"CUDA fixture")
    arches = "sm_80 sm_86 sm_89 sm_90 sm_120"

    def sections(command, **kwargs):
        path = Path(command[-1])
        return ".nv_fatbin" if path == sgl or cuda_in_kt else ".text"

    monkeypatch.setattr(carriers.subprocess, "check_output", sections)
    monkeypatch.setattr(
        carriers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=arches),
    )
    return roots


def test_six_cpu_variants_do_not_require_cuda_sass(tmp_path, monkeypatch):
    roots = native_fixture(tmp_path, monkeypatch)
    evidence = carriers.binary_evidence(roots)
    assert len(evidence) == 7
    assert sum(item["kind"] == "host" for item in evidence.values()) == 6


def test_missing_cpu_variant_is_rejected(tmp_path, monkeypatch):
    roots = native_fixture(tmp_path, monkeypatch)
    next(roots["kt-kernel"].rglob("*amx*.so")).unlink()
    with pytest.raises(ValueError, match="Missing KT CPU variants"):
        carriers.binary_evidence(roots)


def test_missing_sgl_cuda_architecture_is_rejected(tmp_path, monkeypatch):
    roots = native_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        carriers.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="sm_80 sm_86 sm_89 sm_90"),
    )
    with pytest.raises(ValueError, match="SGL common_ops"):
        carriers.binary_evidence(roots)


def test_present_kt_cuda_still_requires_architectures(tmp_path, monkeypatch):
    roots = native_fixture(tmp_path, monkeypatch, cuda_in_kt=True)

    def cubins(command, **kwargs):
        return SimpleNamespace(
            stdout=(
                "sm_80"
                if "kt-kernel" in command[-1]
                else "sm_80 sm_86 sm_89 sm_90 sm_120"
            )
        )

    monkeypatch.setattr(carriers.subprocess, "run", cubins)
    with pytest.raises(ValueError, match="KT CUDA extension"):
        carriers.binary_evidence(roots)
