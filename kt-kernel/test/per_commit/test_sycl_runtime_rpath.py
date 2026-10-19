# SPDX-License-Identifier: Apache-2.0
"""Load build/install artifacts without LD_LIBRARY_PATH after conda RPATH setup."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")
HELPER = Path(__file__).resolve().parents[2] / "cmake/SYCLRuntime.cmake"


@pytest.mark.skipif(sys.platform != "linux", reason="ELF runtime search paths")
@pytest.mark.parametrize("install_rpath_during_build", [False, True])
def test_runtime_directory_survives_conda_rpath_and_install(tmp_path, install_rpath_during_build):
    cc = shutil.which("cc")
    cmake = shutil.which("cmake")
    if not cc or not cmake:
        pytest.skip("requires a C compiler and CMake")
    runtime = tmp_path / "oneapi runtime with spaces"
    runtime.mkdir()
    implementation = tmp_path / "runtime.c"
    implementation.write_text("int kt_test_runtime(void) { return 42; }\n")
    subprocess.run(
        [cc, "-shared", "-fPIC", str(implementation), "-Wl,-soname,libiomp5.so", "-o", str(runtime / "libiomp5.so")],
        check=True,
    )
    shutil.copy2(runtime / "libiomp5.so", runtime / "libsycl.so")
    compiler = tmp_path / "compiler"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        f"root=pathlib.Path({str(runtime)!r})\n"
        "name=sys.argv[1].split('=',1)[1]; candidate=root/name\n"
        "print(candidate if candidate.exists() else name)\n"
    )
    compiler.chmod(0o755)
    source = tmp_path / "source"
    source.mkdir()
    (source / "wrapper.c").write_text(
        "int kt_test_runtime(void);\nint kt_test_result(void) { return kt_test_runtime(); }\n"
    )
    conda = tmp_path / "conda lib"
    conda.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(test_runtime C)\n"
        f"set(CMAKE_CXX_COMPILER {json.dumps(str(compiler))})\n"
        f"set(CMAKE_BUILD_WITH_INSTALL_RPATH {'TRUE' if install_rpath_during_build else 'FALSE'})\n"
        "add_library(wrapper SHARED wrapper.c)\n"
        f"target_link_libraries(wrapper PRIVATE {json.dumps(str(runtime / 'libiomp5.so'))})\n"
        f"set_target_properties(wrapper PROPERTIES BUILD_RPATH {json.dumps(str(conda))} INSTALL_RPATH {json.dumps(str(conda))})\n"
        f"include({json.dumps(str(HELPER))})\nkt_add_sycl_runtime_rpath(wrapper)\n"
        "install(TARGETS wrapper LIBRARY DESTINATION lib)\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LIBRARY_PATH")}
    build = tmp_path / "build"
    subprocess.run([cmake, "-S", str(source), "-B", str(build)], env=env, capture_output=True, text=True, check=True)
    subprocess.run([cmake, "--build", str(build)], env=env, capture_output=True, text=True, check=True)
    installed = tmp_path / "installed"
    subprocess.run(
        [cmake, "--install", str(build), "--prefix", str(installed)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    for artifact in (build / "libwrapper.so", installed / "lib/libwrapper.so"):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes,sys; assert ctypes.CDLL(sys.argv[1]).kt_test_result() == 42",
                str(artifact),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
