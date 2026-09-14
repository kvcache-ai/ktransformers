"""Small artifact tests; native compilation and model acceptance are separate."""

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carriers
import supplement
from four_main import sha256
from test_carriers import raw_wheels


def make_native(root, python):
    for package in ("kt-kernel", "sgl-kernel-kt"):
        package_root = root / package
        path = package_root / (
            f"kt_kernel/_kt_kernel_ext_avx2.cpython-{python}-x86_64-linux-gnu.so"
            if package == "kt-kernel"
            else f"_mscclpp.cpython-{python}-x86_64-linux-gnu.so"
        )
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture-compiled-for-" + python.encode())
        (old,) = (root / "raw").glob(package.replace("-", "_") + "-*.whl")
        old.unlink()
        carriers.retag(package_root, "cp" + python, "cp" + python)
        output = (
            root
            / "raw"
            / f"{package.replace('-', '_')}-1.0-cp{python}-cp{python}-{carriers.PLATFORM}.whl"
        )
        carriers.pack(package_root, output)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    reference_build = tmp_path / "old"
    fresh = tmp_path / "fresh"
    reference_build.mkdir()
    fresh.mkdir()
    raw_wheels(reference_build)
    raw_wheels(fresh)
    for root, contents in ((reference_build, b"accepted-cpp"), (fresh, b"rebuilt-cpp")):
        for name in supplement.MSCCLPP_SUPPORT:
            path = root / "sgl-kernel-kt" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(contents)
    make_native(reference_build, "312")
    make_native(fresh, "311")
    monkeypatch.setattr(carriers, "binary_evidence", lambda roots: {})
    monkeypatch.setattr(supplement, "needed_libraries", lambda path: {"libc.so.6"})
    reference = tmp_path / "reference"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    carriers.assemble(reference_build / "raw", reference, evidence)
    (kt,) = (fresh / "raw").glob("kt_kernel-*.whl")
    (sgl,) = (fresh / "raw").glob("sgl_kernel_kt-*.whl")
    lock = {
        "reference_wheels": {
            path.name: sha256(path) for path in reference.glob("*.whl")
        },
        "native_wheels": {path.name: sha256(path) for path in (kt, sgl)},
        "source_lock": {"unit_fixture": True},
    }
    return reference, kt, sgl, lock, tmp_path / "output", evidence


def test_adds_abi_correct_wheel_and_preserves_payload(inputs):
    reference, kt, sgl, lock, output, evidence = inputs
    wheel = supplement.supplement(*inputs)
    assert "cp311-cp311" in wheel.name
    assert {path.name: sha256(path) for path in reference.glob("*.whl")} == lock[
        "reference_wheels"
    ]
    (old,) = reference.glob("kt_kernel-*.whl")
    with zipfile.ZipFile(old) as before, zipfile.ZipFile(wheel) as after:
        for name in before.namelist():
            if "cpython-312" in name or name.endswith(("/RECORD", "/WHEEL")):
                continue
            assert before.read(name) == after.read(name), name
        assert not any("cpython-312" in name for name in after.namelist())
        assert (
            after.read("_mscclpp.cpython-311-x86_64-linux-gnu.so")
            == b"fixture-compiled-for-311"
        )
        for name in supplement.MSCCLPP_SUPPORT:
            assert after.read(name) == b"accepted-cpp"
    carriers.unpack(wheel, output / "verified")
    assert (
        json.loads((evidence / "supplement.json").read_text())["status"]
        == "ASSEMBLED_RUNTIME_ACCEPTANCE_REQUIRED"
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("checksum", "checksum"),
        ("abi", "Mixed"),
        ("runtime", "Runtime file changed"),
        ("metadata", "metadata changed"),
        ("inventory", "inventory changed"),
        ("payload", "Mixed CUDA payload"),
        ("tag", "tag/extension ABI mismatch"),
    ],
)
def test_rejects_incompatible_supplement(inputs, tmp_path, mutation, match):
    reference, kt, sgl, lock, output, evidence = inputs
    if mutation == "checksum":
        lock["native_wheels"][kt.name] = "0" * 64
    else:
        path = sgl if mutation == "abi" else kt
        if mutation == "payload":
            (path,) = reference.glob("accelerate_kt-*.whl")
        root = tmp_path / "mutated"
        carriers.unpack(path, root)
        if mutation == "abi":
            (binding,) = root.glob("_mscclpp*.so")
            binding.rename(binding.with_name(binding.name.replace("311", "312")))
        elif mutation == "runtime":
            (root / "kt_kernel/__init__.py").write_text("CHANGED = True\n")
        elif mutation == "metadata":
            (metadata,) = root.glob("*.dist-info/METADATA")
            metadata.write_text(metadata.read_text() + "Requires-Dist: unexpected\n")
        elif mutation == "inventory":
            (binding,) = (root / "kt_kernel").glob("*.so")
            binding.unlink()
        elif mutation == "payload":
            (root / "accelerate_kt_sgl_kernel_payload/payload.part").write_bytes(
                b"wrong"
            )
        elif mutation == "tag":
            del lock["native_wheels"][path.name]
            path = path.with_name(path.name.replace("cp311-cp311", "cp312-cp312"))
            carriers.retag(root, "cp312", "cp312")
            inputs = reference, path, sgl, lock, output, evidence
        carriers.pack(root, path)
        key = "reference_wheels" if mutation == "payload" else "native_wheels"
        lock[key][path.name] = sha256(path)
    with pytest.raises(ValueError, match=match):
        supplement.supplement(*inputs)


def test_rejects_size_overflow_and_existing_output(inputs, monkeypatch):
    monkeypatch.setattr(supplement, "LIMIT", 1)
    with pytest.raises(ValueError, match="size limit"):
        supplement.supplement(*inputs)
    with pytest.raises(ValueError, match="new directory"):
        supplement.supplement(*inputs)


def test_rejects_new_dependency_on_retained_cpp_libraries(inputs, monkeypatch):
    monkeypatch.setattr(supplement, "needed_libraries", lambda path: {"mscclpp.so.0"})
    with pytest.raises(ValueError, match="depends on retained"):
        supplement.supplement(*inputs)


def test_rejects_unclassified_native_library(inputs, tmp_path):
    reference, kt, sgl, lock, output, evidence = inputs
    root = tmp_path / "unclassified"
    carriers.unpack(sgl, root)
    (root / "lib/unknown.so").write_bytes(b"new library")
    carriers.pack(root, sgl)
    lock["native_wheels"][sgl.name] = sha256(sgl)
    with pytest.raises(ValueError, match="Unclassified native extension"):
        supplement.supplement(*inputs)
