# SPDX-License-Identifier: Apache-2.0
"""Pinned SGLang exports preserve runtime source and align the kernel dependency."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_sglang.py"
spec = importlib.util.spec_from_file_location("prepare_sglang_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "source with spaces"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "tracked").write_text("original\n")
    (root / "python").mkdir()
    (root / "python/pyproject.toml").write_text('[project]\ndependencies = ["kt-kernel==0.1", "torch==2.9.1"]\n')
    subprocess.run(["git", "-C", str(root), "add", "tracked", "python/pyproject.toml"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(module, "PINNED_SGLANG", pin)
    return root


def test_exports_pin_preserves_runtime_and_dirty_source(source, tmp_path):
    (source / "tracked").write_text("local edit\n")
    (source / "untracked").write_text("private local file\n")
    output = tmp_path / "build parent" / "prepared"
    module.prepare(source, output)
    assert (output / "tracked").read_text() == "original\n"
    assert not (output / "untracked").exists()
    assert (source / "tracked").read_text() == "local edit\n"
    assert (source / "untracked").read_text() == "private local file\n"
    assert not (output / ".git").exists()
    version = module.runpy.run_path(str(module.REPO_ROOT / "version.py"))["__version__"]
    project = module.tomllib.loads((output / "python/pyproject.toml").read_text())
    assert project["project"]["dependencies"] == [f"kt-kernel=={version}", "torch==2.9.1"]
    assert json.loads((output / ".kt-sglang-source.json").read_text()) == {
        "source_commit": module.PINNED_SGLANG,
        "kernel_version": version,
    }


def test_rejects_unknown_pin(source, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "PINNED_SGLANG", "0" * 40)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Expected pinned SGLang"):
        module.prepare(source, output)
    assert not output.exists()


def test_preserves_existing_destination(source, tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep").write_text("keep\n")
    with pytest.raises(ValueError, match="already exists"):
        module.prepare(source, output)
    assert (output / "keep").read_text() == "keep\n"


@pytest.mark.parametrize("dependencies", [[], ["kt-kernel==0.1", "kt-kernel==0.2"]])
def test_invalid_kernel_dependency_does_not_publish_partial_tree(source, tmp_path, dependencies):
    project = source / "python/pyproject.toml"
    project.write_text("[project]\ndependencies = " + json.dumps(dependencies) + "\n")
    subprocess.run(["git", "-C", str(source), "add", "python/pyproject.toml"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "--amend",
            "--no-edit",
            "-q",
        ],
        check=True,
    )
    module.PINNED_SGLANG = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="exactly one pinned kt-kernel"):
        module.prepare(source, output)
    assert not output.exists()
    assert (source / "tracked").read_text() == "original\n"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
