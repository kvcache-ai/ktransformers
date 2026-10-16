# SPDX-License-Identifier: Apache-2.0
"""Exercise the public installer with isolated dependency/build/pip stand-ins."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def installer(tmp_path):
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    shutil.copy2(REPO / "install.sh", root / "install.sh")
    (root / "version.py").write_text('__version__ = "0.7.2"\n')
    (root / "third_party/sglang/python").mkdir(parents=True)
    (root / "third_party/sglang/python/pyproject.toml").write_text('[project]\nname="sglang-kt"\n')
    (root / "kt-kernel/scripts").mkdir(parents=True)
    (root / "kt-kernel/scripts/prepare_sglang.py").write_text(
        'import pathlib,sys\np=pathlib.Path(sys.argv[sys.argv.index("--output")+1]); (p/"python").mkdir(parents=True)\n'
    )
    (root / "kt-kernel/install.sh").write_text(
        '#!/bin/bash\npython3 -c \'import json,os,sys; print(json.dumps({"kind":"kernel", "args":sys.argv[1:], '
        '"env":{k:v for k,v in os.environ.items() if k.startswith("CPUINFER_")}}))\' "$@" >> "$INSTALL_EVENTS"\n'
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    python = tools / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "args=sys.argv[1:]\n"
        'if args[:2]==["-m","pip"]:\n'
        ' with open(os.environ["INSTALL_EVENTS"],"a") as f: f.write(json.dumps({"kind":"pip","args":args,"cwd":os.getcwd()})+"\\n")\n'
        'elif args[:1]==["-c"] and "SYCLXMXGPTQInt4_MOE" in args[1]:\n'
        ' with open(os.environ["INSTALL_EVENTS"],"a") as f: f.write(json.dumps({"kind":"preflight", "script":args[1], "ld_library_path":os.environ.get("LD_LIBRARY_PATH"), "ld_preload":os.environ.get("LD_PRELOAD")})+"\\n")\n'
        'else: os.execv(os.environ["INSTALL_REAL_PYTHON"],[os.environ["INSTALL_REAL_PYTHON"],*args])\n'
    )
    python.chmod(0o755)
    (tools / "icpx").write_text("#!/bin/sh\nexit 0\n")
    (tools / "icpx").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CPUINFER_", "SGLANG_KT_VERSION"))}
    env.update(
        PATH=str(tools) + os.pathsep + env["PATH"],
        INSTALL_EVENTS=str(tmp_path / "events.jsonl"),
        INSTALL_REAL_PYTHON=sys.executable,
        ONEAPI_SETVARS=str(tmp_path / "absent-setvars"),
    )
    return root, env


def run(installer, *args, extra=None):
    root, env = installer
    result = subprocess.run(
        ["bash", str(root / "install.sh"), *args], env=env | (extra or {}), text=True, capture_output=True
    )
    events = Path(env["INSTALL_EVENTS"])
    return result, [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []


def test_sycl_all_builds_kernel_before_installing_aligned_source(installer):
    result, events = run(installer, "--sycl")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e["kind"] for e in events] == ["kernel", "kernel", "preflight", "preflight", "pip"]
    build = events[1]
    assert build["args"] == ["build", "--manual"]
    for key, value in {
        "CPUINFER_USE_SYCL": "1",
        "CPUINFER_USE_CUDA": "0",
        "CPUINFER_CPU_INSTRUCT": "AVX2",
        "CPUINFER_ENABLE_AMX": "OFF",
        "CPUINFER_LLAMA_BUILD_COPY": "ON",
        "CPUINFER_PARALLEL": "2",
    }.items():
        assert build["env"][key] == value
    assert events[-1]["args"] == ["-m", "pip", "install", "./python"]
    assert "/build/sglang-sycl." in events[-1]["cwd"]


def test_plain_all_preserves_original_install_order(installer):
    result, events = run(installer)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e["kind"] for e in events] == ["kernel", "pip", "kernel"]
    assert events[1]["args"] == ["-m", "pip", "install", "./python[all]"]
    assert events[-1]["args"] == ["build"]
    assert "CPUINFER_USE_SYCL" not in events[-1]["env"]


def test_sycl_kernel_only_forwards_build_arguments_and_overrides(installer):
    result, events = run(installer, "kt-kernel", "--sycl", "--no-clean", extra={"CPUINFER_PARALLEL": "3"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e["kind"] for e in events] == ["kernel", "preflight"]
    assert events[0]["args"] == ["build", "--no-clean", "--manual"]
    assert events[0]["env"]["CPUINFER_PARALLEL"] == "3"


def test_sycl_conflicting_gpu_backend_fails_before_install(installer):
    result, events = run(installer, "--sycl", extra={"CPUINFER_USE_CUDA": "1"})
    assert result.returncode != 0 and "conflicts with CPUINFER_USE_CUDA" in result.stderr
    assert not events


def test_sycl_sglang_only_uses_source_copy_and_editable_target(installer):
    result, events = run(installer, "sglang", "--sycl", "--editable")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e["kind"] for e in events] == ["preflight", "pip"]
    assert events[-1]["args"] == ["-m", "pip", "install", "-e", "./python"]


def test_help_does_not_initialize_toolchain_or_install(installer):
    result, events = run(installer, "--sycl", "--help")
    assert "--sycl" in result.stdout and not events


def test_sycl_post_build_check_does_not_inherit_runtime_library_path(installer):
    result, events = run(installer, "kt-kernel", "--sycl", extra={"LD_LIBRARY_PATH": "/unused/oneapi-runtime"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert [e["kind"] for e in events] == ["kernel", "preflight"]
    assert events[-1]["ld_library_path"] is None
    assert events[-1]["ld_preload"] is None


@pytest.mark.parametrize("installed_version", ["0.7.2", "0.7.0.post4"])
def test_sglang_install_checks_kernel_package_version_before_pip(installer, installed_version):
    result, events = run(installer, "sglang", "--sycl")
    assert result.returncode == 0, result.stdout + result.stderr
    assert events[0]["kind"] == "preflight"
    extension = SimpleNamespace(
        moe=SimpleNamespace(SYCLXMXGPTQInt4_MOE=SimpleNamespace(implementation_revision=lambda: 7))
    )
    package = SimpleNamespace(kt_kernel_ext=extension)
    with (
        patch.dict(sys.modules, {"kt_kernel": package}),
        patch("importlib.metadata.version", return_value=installed_version),
        patch.dict(os.environ, {"SGLANG_KT_VERSION": "0.7.2"}),
    ):
        if installed_version == "0.7.2":
            exec(events[0]["script"], {})
        else:
            with pytest.raises(RuntimeError, match="does not match this checkout"):
                exec(events[0]["script"], {})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
