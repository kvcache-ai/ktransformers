#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lightweight packaging script for building and distributing kt-kernel,
a high-performance kernel operations library for KTransformers.

    pip install kt-kernel
    >>> from kt_kernel import AMXMoEWrapper

This script drives your existing CMake build (root `CMakeLists.txt`) and
only needs a working C++ toolchain, CMake (>=3.16), and pybind11 (vendored
already in the repo).

Environment knobs (export before running pip install .):
  CPUINFER_FORCE_REBUILD=1        Always rebuild (ignore any cached build)
  CPUINFER_BUILD_TYPE=Release     Debug / RelWithDebInfo / Release
  CPUINFER_PARALLEL=8             Parallel build jobs (auto = detected cores)
  CPUINFER_CPU_INSTRUCT=FANCY     One of: NATIVE|FANCY|AVX512|AVX2 (maps to CMake flags)
  CPUINFER_ENABLE_AMX=ON          ON/OFF -> -DKTRANSFORMERS_CPU_USE_AMX
  CPUINFER_ENABLE_MLA=OFF         ON/OFF -> -DKTRANSFORMERS_CPU_MLA
  CPUINFER_ENABLE_LTO=ON          ON/OFF -> -DCPUINFER_ENABLE_LTO (your added option)
  CPUINFER_LTO_JOBS=8             Forward to -DCPUINFER_LTO_JOBS
  CPUINFER_LTO_MODE=auto          Forward to -DCPUINFER_LTO_MODE
  CPUINFER_NATIVE=ON               (override LLAMA_NATIVE)

GPU backends (if ever added later, keep placeholders):
  CPUINFER_USE_CUDA=0/1           -DKTRANSFORMERS_USE_CUDA
  CPUINFER_USE_ROCM=0/1           -DKTRANSFORMERS_USE_ROCM
  CPUINFER_USE_MUSA=0/1           -DKTRANSFORMERS_USE_MUSA

Usage:
  pip install .
Or build wheel:
  python -m build  (if you have build/installed)

Resulting wheel exposes a top-level package `kt_kernel` with AMXMoEWrapper and other kernel wrappers.
"""
from __future__ import annotations
import os
import re
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import shutil

################################################################################
# Helpers
################################################################################

REPO_ROOT = Path(__file__).parent.resolve()

CPU_FEATURE_MAP = {
    "FANCY": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON -DLLAMA_AVX512_FANCY_SIMD=ON",
    "AVX512": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON",
    "AVX2": "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON",
    "NATIVE": "-DLLAMA_NATIVE=ON",
}


def default_build_type() -> str:
    return os.environ.get("CPUINFER_BUILD_TYPE", "Release")


def detect_parallel_jobs() -> str:
    if "CPUINFER_PARALLEL" in os.environ:
        return os.environ["CPUINFER_PARALLEL"]
    try:
        import multiprocessing

        return str(multiprocessing.cpu_count())
    except Exception:
        return "1"


def cpu_feature_flags() -> list[str]:
    mode = os.environ.get("CPUINFER_CPU_INSTRUCT", "NATIVE").upper()
    return [tok for tok in CPU_FEATURE_MAP.get(mode, CPU_FEATURE_MAP["NATIVE"]).split() if tok]


################################################################################
# CMakeExtension + builder
################################################################################


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake present
        try:
            subprocess.run(["cmake", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:  # pragma: no cover
            raise RuntimeError("CMake is required to build this project") from e
        super().run()

    def build_extension(self, ext: CMakeExtension):
        # Auto-detect CUDA toolkit if user did not explicitly set CPUINFER_USE_CUDA
        def detect_cuda_toolkit() -> bool:
            # Respect CUDA_HOME
            cuda_home = os.environ.get("CUDA_HOME")
            if cuda_home:
                nvcc_path = Path(cuda_home) / "bin" / "nvcc"
                if nvcc_path.exists():
                    return True
            # PATH lookup
            if shutil.which("nvcc") is not None:
                return True
            # Common default install prefix
            if Path("/usr/local/cuda/bin/nvcc").exists():
                return True
            return False

        if os.environ.get("CPUINFER_USE_CUDA") is None:
            auto_cuda = detect_cuda_toolkit()
            os.environ["CPUINFER_USE_CUDA"] = "1" if auto_cuda else "0"
            print(f"-- CPUINFER_USE_CUDA not set; auto-detected CUDA toolkit: {'YES' if auto_cuda else 'NO'}")

        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = default_build_type()
        build_temp = Path(self.build_temp) / f"{ext.name}_{cfg}"
        build_temp.mkdir(parents=True, exist_ok=True)

        # Base CMake args
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}/",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        # CPU feature flags mapping
        cmake_args += cpu_feature_flags()

        # Optional AMX / MLA toggles
        if os.environ.get("CPUINFER_ENABLE_AMX"):
            cmake_args.append(f"-DKTRANSFORMERS_CPU_USE_AMX={os.environ['CPUINFER_ENABLE_AMX']}")
        if os.environ.get("CPUINFER_ENABLE_KML"):
            cmake_args.append(f"-DKTRANSFORMERS_CPU_USE_KML={os.environ['CPUINFER_ENABLE_KML']}")
        if os.environ.get("CPUINFER_ENABLE_MLA"):
            cmake_args.append(f"-DKTRANSFORMERS_CPU_MLA={os.environ['CPUINFER_ENABLE_MLA']}")

        # LTO toggles if user added them in CMakeLists
        if os.environ.get("CPUINFER_ENABLE_LTO"):
            cmake_args.append(f"-DCPUINFER_ENABLE_LTO={os.environ['CPUINFER_ENABLE_LTO']}")
        if os.environ.get("CPUINFER_LTO_JOBS"):
            cmake_args.append(f"-DCPUINFER_LTO_JOBS={os.environ['CPUINFER_LTO_JOBS']}")
        if os.environ.get("CPUINFER_LTO_MODE"):
            cmake_args.append(f"-DCPUINFER_LTO_MODE={os.environ['CPUINFER_LTO_MODE']}")

        # GPU backends (mutually exclusive expected)
        if os.environ.get("CPUINFER_USE_CUDA") == "1":
            cmake_args.append("-DKTRANSFORMERS_USE_CUDA=ON")
            print("-- Enabling CUDA backend (-DKTRANSFORMERS_USE_CUDA=ON)")
        if os.environ.get("CPUINFER_USE_ROCM") == "1":
            cmake_args.append("-DKTRANSFORMERS_USE_ROCM=ON")
        if os.environ.get("CPUINFER_USE_MUSA") == "1":
            cmake_args.append("-DKTRANSFORMERS_USE_MUSA=ON")

        # Respect user extra CMAKE_ARGS (space separated)
        extra = os.environ.get("CMAKE_ARGS")
        if extra:
            cmake_args += [a for a in extra.split() if a]

        # Force rebuild? (delete cache)
        if os.environ.get("CPUINFER_FORCE_REBUILD") == "1":
            cache = build_temp / "CMakeCache.txt"
            if cache.exists():
                cache.unlink()

        print("-- CMake configure args:")
        for a in cmake_args:
            print("   ", a)

        # Configure
        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)

        # Build
        build_args = ["--build", ".", "--config", cfg]
        jobs = detect_parallel_jobs()
        if jobs:
            build_args += ["--parallel", jobs]
        print("-- CMake build args:", " ".join(build_args))
        subprocess.run(["cmake", *build_args], cwd=build_temp, check=True)

        # On some systems LTO + CMake + pybind may place the built .so inside build tree; move if needed
        built_candidates = list(build_temp.rglob(f"{ext.name}*.so"))
        for cand in built_candidates:
            if cand.parent != extdir:
                target = extdir / cand.name
                target.parent.mkdir(parents=True, exist_ok=True)
                # Overwrite stale
                if not target.exists() or target.stat().st_mtime < cand.stat().st_mtime:
                    print(f"-- Copying {cand} -> {target}")
                    target.write_bytes(cand.read_bytes())


################################################################################
# Version (simple). If you later add a python package dir, you can read from it.
################################################################################

VERSION = os.environ.get("CPUINFER_VERSION", "0.1.0")

################################################################################
# Setup
################################################################################

setup(
    name="kt-kernel",
    version=VERSION,
    description="KT-Kernel: High-performance kernel operations for KTransformers (AMX/AVX/KML optimizations)",
    author="kvcache-ai",
    license="Apache-2.0",
    python_requires=">=3.8",
    packages=["kt_kernel"],
    package_dir={"kt_kernel": "python"},
    ext_modules=[CMakeExtension("cpuinfer_ext", str(REPO_ROOT))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
)
