"""Repackaging must preserve code, bounds, provenance, and input integrity."""

from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carriers import pack
from four_main import inspect_wheel, sha256
from repack_training_tools import repack

VERSIONS = {"transformers": "5.6.0.post5", "accelerate": "1.14.0.post3"}


def fixture_wheel(tmp_path, name="peft"):
    root = tmp_path / "source"
    dist = root / f"{name}-0.18.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.18.1\n"
        'Requires-Dist: transformers>=4.55.0; python_version >= "3.10"\n'
        "Requires-Dist: accelerate>=0.21.0\nRequires-Dist: torch>=1.13.0\n"
    )
    (dist / "WHEEL").write_text(
        "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    package = root / name
    package.mkdir()
    (package / "__init__.py").write_bytes(b"__version__ = '0.18.1'\n")
    (dist / "LICENSE").write_text("license fixture")
    wheel = tmp_path / f"{name}-0.18.1-py3-none-any.whl"
    pack(root, wheel)
    return wheel


def invoke(source, output, **kwargs):
    return repack(
        source,
        output,
        expected_sha256=kwargs.pop("expected_sha256", sha256(source)),
        local_tag="kt.20260912",
        versions=kwargs.pop("versions", VERSIONS),
        **kwargs,
    )


def test_only_dependency_metadata_changes_and_output_is_reproducible(tmp_path):
    source = fixture_wheel(tmp_path)
    first = invoke(source, tmp_path / "first")
    second = invoke(source, tmp_path / "second")
    assert first["sha256"] == second["sha256"]
    output = tmp_path / "first" / first["filename"]
    info = inspect_wheel(output)
    assert info["version"] == "0.18.1+kt.20260912"
    assert 'transformers-kt>=4.55.0; python_version >= "3.10"' in info["requires_dist"]
    assert "accelerate-kt>=0.21.0" in info["requires_dist"]
    assert "torch>=1.13.0" in info["requires_dist"]
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(output) as after:
        assert before.read("peft/__init__.py") == after.read("peft/__init__.py")
        assert (
            after.read("peft-0.18.1+kt.20260912.dist-info/LICENSE")
            == b"license fixture"
        )
        assert any(
            name.endswith("/KT-TOOLING-PROVENANCE.json") for name in after.namelist()
        )
    with pytest.raises(ValueError, match="overwrite"):
        invoke(source, tmp_path / "first")


def test_lf_repackaging_is_not_implicitly_allowed(tmp_path):
    with pytest.raises(ValueError, match="Only PEFT and TRL"):
        invoke(fixture_wheel(tmp_path, "llamafactory"), tmp_path / "out")


def test_upstream_digest_is_required(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        invoke(fixture_wheel(tmp_path), tmp_path / "out", expected_sha256="0" * 64)


def test_does_not_relax_incompatible_dependency_bounds(tmp_path):
    with pytest.raises(ValueError, match="does not satisfy"):
        invoke(
            fixture_wheel(tmp_path),
            tmp_path / "out",
            versions=VERSIONS | {"transformers": "4.0.0"},
        )


def test_input_record_is_verified(tmp_path):
    source = fixture_wheel(tmp_path)
    with zipfile.ZipFile(source, "a") as wheel:
        wheel.writestr("injected.py", b"bad")
    with pytest.raises(ValueError, match="not recorded"):
        invoke(source, tmp_path / "out")
