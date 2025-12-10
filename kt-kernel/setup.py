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
  CPUINFER_ENABLE_AMX=OFF         ON/OFF -> -DKTRANSFORMERS_CPU_USE_AMX
  CPUINFER_ENABLE_MLA=OFF         ON/OFF -> -DKTRANSFORMERS_CPU_MLA
  CPUINFER_ENABLE_AMD=OFF         ON/OFF -> -DKTRANSFORMERS_CPU_MOE_AMD
  CPUINFER_ENABLE_KML=OFF         ON/OFF -> -DKTRANSFORMERS_CPU_USE_KML
  CPUINFER_ENABLE_AVX512=OFF      ON/OFF -> -DKTRANSFORMERS_CPU_USE_AMX_AVX512
  CPUINFER_BLIS_ROOT=/path/to/blis  Forward to -DBLIS_ROOT


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

# -------------------------
# Env parsing helpers
# -------------------------
def _env_get_bool(name: str, default: bool | None = None) -> bool | None:
    v = os.environ.get(name)
    if v is None:
        return default
    val = v.strip().lower()
    if val in ("1", "on", "true", "yes", "y", "enable", "enabled"):
        return True
    if val in ("0", "off", "false", "no", "n", "disable", "disabled"):
        return False
    return default


def _cmake_onoff(flag: bool) -> str:
    return "ON" if flag else "OFF"


def _forward_bool_env(cmake_args: list[str], env_name: str, cmake_flag: str) -> bool:
    """If env exists, forward it to CMake as -D<flag>=ON/OFF and return True; else return False."""
    b = _env_get_bool(env_name, None)
    if b is None:
        return False
    cmake_args.append(f"-D{cmake_flag}={_cmake_onoff(b)}")
    print(f"-- Forward {env_name} -> -D{cmake_flag}={_cmake_onoff(b)}")
    return True


def _forward_str_env(cmake_args: list[str], env_name: str, cmake_flag: str) -> bool:
    v = os.environ.get(env_name)
    if not v:
        return False
    cmake_args.append(f"-D{cmake_flag}={v}")
    print(f"-- Forward {env_name} -> -D{cmake_flag}={v}")
    return True

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

    def detect_cpu_info(self) -> dict:
        """Detect CPU vendor/arch and instruction set features.

        Returns a dict like:
            {
                'vendor': 'intel'|'amd'|'arm'|'unknown',
                'arch': platform.machine().lower(),
                'features': set(['AVX2','AVX512','AMX']),
                'raw': { 'flags': set([...]) }
            }
        """
        info = {
            "vendor": "unknown",
            "arch": platform.machine().lower(),
            "features": set(),
            "raw": {"flags": set()},
        }
        try:
            sysname = platform.system()
            if sysname == "Linux":
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                    cpuinfo = f.read()
                low = cpuinfo.lower()

                # vendor
                if "vendor_id" in low:
                    # Typical x86 linux
                    m = re.search(r"vendor_id\s*:\s*(\S+)", cpuinfo)
                    if m:
                        v = m.group(1).lower()
                        if "genuineintel" in v:
                            info["vendor"] = "intel"
                        elif "authenticamd" in v:
                            info["vendor"] = "amd"
                # ARM sometimes has 'model name' or 'Hardware'
                if info["vendor"] == "unknown":
                    if any(tok in low for tok in ["aarch64", "armv8", "arm cortex", "kunpeng", "kirin", "huawei"]):
                        info["vendor"] = "arm"

                # flags collection (x86 uses 'flags', arm uses 'Features')
                flags = set()
                for key in ("flags", "Features", "features"):
                    m = re.search(rf"^{key}\s*:\s*(.+)$", cpuinfo, re.IGNORECASE | re.MULTILINE)
                    if m:
                        flags.update(m.group(1).lower().split())
                info["raw"]["flags"] = flags

                # feature summary
                if any(f in flags or f in low for f in ["avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512vnni"]):
                    info["features"].add("AVX512")
                if "avx2" in flags or "avx2" in low:
                    info["features"].add("AVX2")
                # AMX flags on Linux are with underscores; keep hyphen fallback just in case
                if any(
                    f in flags or f in low
                    for f in ["amx_bf16", "amx_int8", "amx_tile", "amx-bf16", "amx-int8", "amx-tile"]
                ):
                    info["features"].add("AMX")

            elif sysname == "Darwin":
                # macOS: Apple Silicon (arm64) vs Intel
                arch = platform.machine().lower()
                info["arch"] = arch
                if arch in ("arm64", "aarch64"):
                    info["vendor"] = "arm"
                else:
                    info["vendor"] = "intel"
                # No AVX/AMX on Apple Silicon; assume none

            elif sysname == "Windows":
                # Minimal detection via arch; detailed CPUID omitted for brevity
                arch = platform.machine().lower()
                info["arch"] = arch
                if arch in ("arm64", "aarch64"):
                    info["vendor"] = "arm"
                else:
                    # Could be Intel or AMD; leave unknown
                    info["vendor"] = "unknown"
        except Exception as e:
            print(f"Warning: CPU detection failed: {e}")
        return info

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

        # Locate nvcc executable (without forcing user to set -DCMAKE_CUDA_COMPILER)
        def find_nvcc_path() -> str | None:
            cuda_home = os.environ.get("CUDA_HOME")
            if cuda_home:
                cand = Path(cuda_home) / "bin" / "nvcc"
                if cand.exists():
                    return str(cand)
            which_nvcc = shutil.which("nvcc")
            if which_nvcc:
                return which_nvcc
            # Common fallbacks (ordered by preference)
            for cand in [
                "/usr/local/cuda-12.6/bin/nvcc",
                "/usr/local/cuda/bin/nvcc",
                "/usr/bin/nvcc",
                "/usr/lib/nvidia-cuda-toolkit/bin/nvcc",
            ]:
                if Path(cand).exists():
                    return cand
            return None

        # Note: We no longer set CMAKE_CUDA_ARCHITECTURES by default.
        # If users want to specify CUDA archs, they can set env CPUINFER_CUDA_ARCHS
        # (e.g. "89" or "86;89") or pass it via CMAKE_ARGS.
        auto_moe_kernel_ = False
        # Normalize CPUINFER_USE_CUDA: if unset, auto-detect; otherwise respect truthy/falsey values
        cuda_env = _env_get_bool("CPUINFER_USE_CUDA", None)
        if cuda_env is None:
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

        # CPU feature flags mapping: if user specified CPUINFER_CPU_INSTRUCT, honor it;
        # else auto-pick based on detection (x86 only)
        cmake_args += cpu_feature_flags()
        d = self.detect_cpu_info()
        print(f"Detected CPU info: {d}")
        cpu_mode = os.environ.get("CPUINFER_CPU_INSTRUCT", "NATIVE").upper()

        # Vendor / feature specific toggles
        # AMD MoE: explicit env overrides; otherwise default ON on AMD CPU
        if not _forward_bool_env(cmake_args, "CPUINFER_ENABLE_AMD", "KTRANSFORMERS_CPU_MOE_AMD"):
            if d.get("vendor") == "amd":
                auto_moe_kernel_ = True
                cmake_args.append("-DKTRANSFORMERS_CPU_MOE_AMD=ON")
                print("-- Detected AMD CPU; enabling AMD MoE kernel (-DKTRANSFORMERS_CPU_MOE_AMD=ON)")
                _forward_str_env(cmake_args, "CPUINFER_BLIS_ROOT", "BLIS_ROOT")

        # KML: explicit env overrides; otherwise default ON on ARM
        if not _forward_bool_env(cmake_args, "CPUINFER_ENABLE_KML", "KTRANSFORMERS_CPU_USE_KML"):
            if d.get("vendor") == "arm":
                auto_moe_kernel_ = True
                cmake_args.append("-DKTRANSFORMERS_CPU_USE_KML=ON")
                print("-- Detected ARM CPU; enabling KML (-DKTRANSFORMERS_CPU_USE_KML=ON)")

        # AMX: explicit env overrides; else enable if detected
        if not _forward_bool_env(cmake_args, "CPUINFER_ENABLE_AMX", "KTRANSFORMERS_CPU_USE_AMX"):
            if "AMX" in d["features"]:
                cmake_args.append("-DKTRANSFORMERS_CPU_USE_AMX=ON")
                print("-- AMX support detected; enabling (-DKTRANSFORMERS_CPU_USE_AMX=ON)")

        # AVX512 umbrella (AMX/AVX512 kernels):
        # - If user explicitly sets CPUINFER_ENABLE_AVX512 -> honor it
        # - Otherwise, only auto-enable when CPU mode actually wants AVX512
        #   (NATIVE/FANCY/AVX512). In AVX2 mode we do NOT enable this, so
        #   RAWINT4 / K2 kernels are not compiled.
        if not _forward_bool_env(cmake_args, "CPUINFER_ENABLE_AVX512", "KTRANSFORMERS_CPU_USE_AMX_AVX512"):
            if cpu_mode in ("NATIVE", "FANCY", "AVX512") and ("AMX" in d["features"] or "AVX512" in d["features"]):
                cmake_args.append("-DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON")
                print("-- Enabling AMX/AVX512 umbrella (-DKTRANSFORMERS_CPU_USE_AMX_AVX512=ON)")
            else:
                print(f"-- CPUINFER_CPU_INSTRUCT={cpu_mode}; not auto-enabling AMX/AVX512 umbrella")

        # Auto-enable MOE kernel only when env explicitly turns on AMD or KML backend
        # (Do not enable purely on vendor auto-detection to avoid surprise behavior.)
        amd_env = _env_get_bool("CPUINFER_ENABLE_AMD", None)
        kml_env = _env_get_bool("CPUINFER_ENABLE_KML", None)
        if amd_env or kml_env:
            auto_moe_kernel_ = True
        already_set = any("KTRANSFORMERS_CPU_MOE_KERNEL" in a for a in cmake_args)
        if not already_set and auto_moe_kernel_:
            cmake_args.append("-DKTRANSFORMERS_CPU_MOE_KERNEL=ON")
            print("-- Auto-enabling MOE kernel (-DKTRANSFORMERS_CPU_MOE_KERNEL=ON) because CPUINFER_ENABLE_AMD or CPUINFER_ENABLE_KML is ON")

        # Friendly summary
        print(
            f"-- CPU detection: vendor={d.get('vendor')} arch={d.get('arch')} features={sorted(list(d.get('features', [])))}"
        )

        # MLA toggle (string/boolean allowed)
        if not _forward_bool_env(cmake_args, "CPUINFER_ENABLE_MLA", "KTRANSFORMERS_CPU_MLA"):
            _forward_str_env(cmake_args, "CPUINFER_ENABLE_MLA", "KTRANSFORMERS_CPU_MLA")

        # LTO toggles
        _forward_bool_env(cmake_args, "CPUINFER_ENABLE_LTO", "CPUINFER_ENABLE_LTO")
        _forward_str_env(cmake_args, "CPUINFER_LTO_JOBS", "CPUINFER_LTO_JOBS")
        _forward_str_env(cmake_args, "CPUINFER_LTO_MODE", "CPUINFER_LTO_MODE")

        # GPU backends (mutually exclusive expected)
        if _env_get_bool("CPUINFER_USE_CUDA", False):
            cmake_args.append("-DKTRANSFORMERS_USE_CUDA=ON")
            print("-- Enabling CUDA backend (-DKTRANSFORMERS_USE_CUDA=ON)")
            # Inject nvcc compiler path automatically unless user already specified one.
            user_specified_compiler = any("CMAKE_CUDA_COMPILER" in a for a in cmake_args)
            if not user_specified_compiler:
                extra_env = os.environ.get("CMAKE_ARGS", "")
                if "CMAKE_CUDA_COMPILER" in extra_env:
                    user_specified_compiler = True
            if not user_specified_compiler:
                nvcc_path = find_nvcc_path()
                if nvcc_path:
                    cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")
                    print(f"-- Auto-detected nvcc: {nvcc_path} (adding -DCMAKE_CUDA_COMPILER)")
                else:
                    print("-- Warning: nvcc not found via CUDA_HOME/PATH/common prefixes; CUDA configure may fail.")
            # Optional host compiler for nvcc if user set CUDAHOSTCXX
            if os.environ.get("CUDAHOSTCXX"):
                hostcxx = os.environ["CUDAHOSTCXX"]
                cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={hostcxx}")
                print(f"-- Using CUDA host compiler from CUDAHOSTCXX: {hostcxx}")
            # Respect user-provided architectures only (no default auto-detection).
            archs_env = os.environ.get("CPUINFER_CUDA_ARCHS", "").strip()
            if archs_env and not any("CMAKE_CUDA_ARCHITECTURES" in a for a in cmake_args):
                cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={archs_env}")
                print(f"-- Set CUDA architectures from CPUINFER_CUDA_ARCHS: {archs_env}")
        if _env_get_bool("CPUINFER_USE_ROCM", False):
            cmake_args.append("-DKTRANSFORMERS_USE_ROCM=ON")
        if _env_get_bool("CPUINFER_USE_MUSA", False):
            cmake_args.append("-DKTRANSFORMERS_USE_MUSA=ON")

        # Respect user extra CMAKE_ARGS (space separated)
        extra = os.environ.get("CMAKE_ARGS")
        if extra:
            cmake_args += [a for a in extra.split() if a]

        # Force rebuild? (delete cache)
        if _env_get_bool("CPUINFER_FORCE_REBUILD", True):
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
    packages=["kt_kernel", "kt_kernel.utils"],
    package_dir={
        "kt_kernel": "python",
        "kt_kernel.utils": "python/utils",
    },
    ext_modules=[CMakeExtension("kt_kernel_ext", str(REPO_ROOT))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
)
