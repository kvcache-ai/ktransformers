#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : chenxl
Date         : 2024-07-27 16:15:27
Version      : 1.0.0
LastEditors  : chenxl
LastEditTime : 2024-08-14 16:36:19
Adapted from:
https://github.com/Dao-AILab/flash-attention/blob/v2.6.3/setup.py
Copyright (c) 2023, Tri Dao.
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''

import os
import sys
import re
import ast
import subprocess
import platform
import shutil
import http.client
import urllib.request
import urllib.error
from pathlib import Path
from packaging.version import parse
import torch.version
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools import setup, Extension
from cpufeature.extension import CPUFeature
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
try:
    from torch_musa.utils.simple_porting import SimplePorting
    from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension, MUSA_HOME
except ImportError:
    MUSA_HOME=None

class CpuInstructInfo:
    CPU_INSTRUCT = os.getenv("CPU_INSTRUCT", "NATIVE")
    FANCY = "FANCY"
    AVX512 = "AVX512"
    AVX2 = "AVX2"
    CMAKE_NATIVE = "-DLLAMA_NATIVE=ON"
    CMAKE_FANCY = "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON -DLLAMA_AVX512_FANCY_SIMD=ON"
    CMAKE_AVX512 = "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON -DLLAMA_AVX512=ON"
    CMAKE_AVX2 = "-DLLAMA_NATIVE=OFF -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON"

class VersionInfo:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_NAME = "ktransformers"
    BASE_WHEEL_URL:str = (
        "https://github.com/kvcache-ai/ktransformers/releases/download/{tag_name}/{wheel_filename}"
    )
    FORCE_BUILD = os.getenv("KTRANSFORMERS_FORCE_BUILD", "FALSE") == "TRUE"

    def get_musa_bare_metal_version(self, musa_dir):
        raw_output = subprocess.run(
            [musa_dir + "/bin/mcc", "-v"], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.decode("utf-8")
        output = raw_output.split()
        release_idx = output.index("version") + 1
        bare_metal_version = parse(output[release_idx].split(",")[0])
        musa_version = f"{bare_metal_version.major}{bare_metal_version.minor}"
        return musa_version

    def get_cuda_bare_metal_version(self, cuda_dir):
        raw_output = subprocess.check_output(
            [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        bare_metal_version = parse(output[release_idx].split(",")[0])
        cuda_version = f"{bare_metal_version.major}{bare_metal_version.minor}"
        return cuda_version

    def get_cuda_version_of_torch(self):
        torch_cuda_version = parse(torch.version.cuda)
        cuda_version = f"{torch_cuda_version.major}{torch_cuda_version.minor}"
        return cuda_version

    def get_platform(self,):
        """
        Returns the platform name as used in wheel filenames.
        """
        if sys.platform.startswith("linux"):
            return f'linux_{platform.uname().machine}'
        elif sys.platform == "win32":
            return "win_amd64"
        else:
            raise ValueError("Unsupported platform: {}".format(sys.platform))

    def get_cpu_instruct(self,):
        if CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.FANCY:
            return "fancy"
        elif CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.AVX512:
            return "avx512"
        elif CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.AVX2:
            return "avx2"
        else:
            print("Using native cpu instruct")
        if sys.platform.startswith("linux"):
            with open('/proc/cpuinfo', 'r', encoding="utf-8") as cpu_f:
                cpuinfo = cpu_f.read()
            flags_line = [line for line in cpuinfo.split(
                '\n') if line.startswith('flags')][0]
            flags = flags_line.split(':')[1].strip().split(' ')
            # fancy with AVX512-VL, AVX512-BW, AVX512-DQ, AVX512-VNNI
            for flag in flags:
                if 'avx512bw' in flag:
                    return 'fancy'
            for flag in flags:
                if 'avx512' in flag:
                    return 'avx512'
            for flag in flags:
                if 'avx2' in flag:
                    return 'avx2'
            raise ValueError(
                "Unsupported cpu Instructions: {}".format(flags_line))
        elif sys.platform == "win32":
            if CPUFeature.get("AVX512bw", False):
                return 'fancy'
            if CPUFeature.get("AVX512f", False):
                return 'avx512'
            if CPUFeature.get("AVX2", False):
                return 'avx2'
            raise ValueError(
                "Unsupported cpu Instructions: {}".format(str(CPUFeature)))
        else:
            raise ValueError("Unsupported platform: {}".format(sys.platform))

    def get_torch_version(self,):
        torch_version_raw = parse(torch.__version__)
        torch_version = f"{torch_version_raw.major}{torch_version_raw.minor}"
        return torch_version

    def get_flash_version(self,):
        version_file = os.path.join(
            Path(VersionInfo.THIS_DIR), VersionInfo.PACKAGE_NAME, "__init__.py")
        with open(version_file, "r", encoding="utf-8") as f:
            version_match = re.search(
                r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
        flash_version = ast.literal_eval(version_match.group(1))
        return flash_version

    def get_package_version(self, full_version=False):
        flash_version = str(self.get_flash_version())
        torch_version = self.get_torch_version()
        cpu_instruct = self.get_cpu_instruct()
        backend_version = ""
        if CUDA_HOME is not None:
            backend_version = f"cu{self.get_cuda_bare_metal_version(CUDA_HOME)}"
        elif MUSA_HOME is not None:
            backend_version = f"mu{self.get_musa_bare_metal_version(MUSA_HOME)}"
        else:
            raise ValueError("Unsupported backend: CUDA_HOME and MUSA_HOME are not set.")
        package_version = f"{flash_version}+{backend_version}torch{torch_version}{cpu_instruct}"
        if full_version:
            return package_version
        if not VersionInfo.FORCE_BUILD:
            return flash_version
        return package_version


class BuildWheelsCommand(_bdist_wheel):
    def get_wheel_name(self,):
        version_info = VersionInfo()
        package_version = version_info.get_package_version(full_version=True)
        flash_version = version_info.get_flash_version()
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheel_filename = f"{VersionInfo.PACKAGE_NAME}-{package_version}-{python_version}-{python_version}-{version_info.get_platform()}.whl"
        wheel_url = VersionInfo.BASE_WHEEL_URL.format(tag_name=f"v{flash_version}", wheel_filename=wheel_filename)
        return wheel_filename, wheel_url


    def run(self):
        if VersionInfo.FORCE_BUILD:
            super().run()
            return
        wheel_filename, wheel_url = self.get_wheel_name()
        print("Guessing wheel URL: ", wheel_url)
        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)
            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not os.path.exists(self.dist_dir):
                os.makedirs(self.dist_dir)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = os.path.join(self.dist_dir, archive_basename + ".whl")
            print("Raw wheel path", wheel_path)
            shutil.move(wheel_filename, wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError, http.client.RemoteDisconnected):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(
            Path(sourcedir).resolve() / "ktransformers" / "ktransformers_ext")


class CMakeBuild(BuildExtension):

    def build_extension(self, ext) -> None:
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)
                    ) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        if CUDA_HOME is not None:
            cmake_args += ["-DKTRANSFORMERS_USE_CUDA=ON"]
        elif MUSA_HOME is not None:
            cmake_args += ["-DKTRANSFORMERS_USE_MUSA=ON"]
        else:
            raise ValueError("Unsupported backend: CUDA_HOME and MUSA_HOME are not set.")

        build_args = []
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.FANCY:
            cpu_args = CpuInstructInfo.CMAKE_FANCY
        elif CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.AVX512:
            cpu_args = CpuInstructInfo.CMAKE_AVX512
        elif CpuInstructInfo.CPU_INSTRUCT == CpuInstructInfo.AVX2:
            cpu_args = CpuInstructInfo.CMAKE_AVX2
        else:
            cpu_args = CpuInstructInfo.CMAKE_NATIVE

        cmake_args += [
            item for item in cpu_args.split(" ") if item
        ]
        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]
        if self.compiler.compiler_type != "msvc":
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(
                x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})
            if not single_config and not contains_arch and cmake_generator:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            cpu_count = os.cpu_count()
            if cpu_count is None:
                cpu_count = 1
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"--parallel={self.parallel}"]
            else:
                build_args += [f"--parallel={cpu_count}"]
        print("CMake args:", cmake_args)
        build_temp = Path(ext.sourcedir) / "build"
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        result = subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True , capture_output=True
        )
        print("Standard output:", result.stdout)
        print("Standard error:", result.stderr)
        subprocess.run(
            ["cmake", "--build", ".", "--verbose", *build_args], cwd=build_temp, check=True
        )

if CUDA_HOME is not None:
    ops_module = CUDAExtension('KTransformersOps', [
        'ktransformers/ktransformers_ext/cuda/custom_gguf/dequant.cu',
        'ktransformers/ktransformers_ext/cuda/binding.cpp',
        'ktransformers/ktransformers_ext/cuda/gptq_marlin/gptq_marlin.cu'
    ],
    extra_compile_args={
            'cxx': ['-O3', '-DKTRANSFORMERS_USE_CUDA'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-Xcompiler', '-fPIC',
                '-DKTRANSFORMERS_USE_CUDA',
            ]
        }
    )
elif MUSA_HOME is not None:
    SimplePorting(cuda_dir_path="ktransformers/ktransformers_ext/cuda", mapping_rule={
        # Common rules
        "at::cuda": "at::musa",
        "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
        "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
        "nv_bfloat16": "mt_bfloat16",
        }).run()
    ops_module = MUSAExtension('KTransformersOps', [
        'ktransformers/ktransformers_ext/cuda_musa/custom_gguf/dequant.mu',
        'ktransformers/ktransformers_ext/cuda_musa/binding.cpp',
        # TODO: Add Marlin support for MUSA.
        # 'ktransformers/ktransformers_ext/cuda_musa/gptq_marlin/gptq_marlin.mu'
    ],
    extra_compile_args={
            'cxx': ['force_mcc'],
            'mcc': [
                '-O3',
                '-DKTRANSFORMERS_USE_MUSA',
                '-DTHRUST_IGNORE_CUB_VERSION_CHECK',
            ]
        }
    )
else:
    raise ValueError("Unsupported backend: CUDA_HOME and MUSA_HOME are not set.")

setup(
    version=VersionInfo().get_package_version(),
    cmdclass={"bdist_wheel":BuildWheelsCommand ,"build_ext": CMakeBuild},
    ext_modules=[
        CMakeExtension("cpuinfer_ext"),
        ops_module,
    ]
)
