"""
Installation strategies for different installation methods.
"""

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from kt_kernel.cli.utils.environment import detect_cpu_build_features

from .package_installer import create_sglang_installer
from .system_deps import SystemDependencyManager


class InstallMode(str, Enum):
    """Installation mode."""

    INFERENCE = "inference"
    SFT = "sft"
    FULL = "full"


class InstallationStrategy(ABC):
    """Base class for installation strategies."""

    def __init__(self, mode: InstallMode, skip_torch: bool, yes: bool = False):
        self.mode = mode
        self.skip_torch = skip_torch
        self.yes = yes

    @abstractmethod
    def execute(self) -> None:
        """Execute the installation."""
        pass


class PyPIWheelStrategy(InstallationStrategy):
    """Install from PyPI pre-built wheels."""

    def __init__(self, mode: InstallMode, skip_torch: bool, force: bool, yes: bool):
        super().__init__(mode, skip_torch, yes)
        self.force = force

    def execute(self) -> None:
        """Execute PyPI wheel installation."""
        from .helpers import (
            _check_dependencies,
            _display_dependency_table,
            _install_packages_pip,
        )

        # Check existing packages
        console.print()
        print_step(t("install_checking_deps"))

        deps = _check_dependencies(self.mode)
        missing = [d for d in deps if d["status"] == "missing"]
        outdated = [d for d in deps if d["status"] == "outdated"]

        _display_dependency_table(deps)

        if not missing and not outdated and not self.force:
            console.print()
            print_success(t("install_already_installed"))
            return

        # Confirm installation
        if not self.yes:
            console.print()
            to_install = len(missing) + len(outdated)
            if not confirm(t("install_confirm", count=to_install)):
                raise typer.Abort()

        # Install packages
        console.print()
        print_step(t("install_installing_deps"))

        _install_packages_pip(self.mode, self.skip_torch)


class PyPISourceStrategy(InstallationStrategy):
    """Build from PyPI source (sdist) with custom CPU flags."""

    def __init__(
        self,
        mode: InstallMode,
        skip_torch: bool,
        cpu_instruct: Optional[str],
        enable_amx: Optional[bool],
        build_type: str,
        yes: bool,
        verify: bool,
    ):
        super().__init__(mode, skip_torch, yes)
        self.cpu_instruct = cpu_instruct
        self.enable_amx = enable_amx
        self.build_type = build_type
        self.verify = verify

    def execute(self) -> None:
        """Execute PyPI source build installation."""
        from .helpers import (
            _display_verification_result,
            _install_other_deps,
            _install_torch,
            _verify_kt_kernel_installation,
        )

        # Step 1: Check and install system dependencies
        sys_dep_mgr = SystemDependencyManager()

        console.print()
        print_step(t("install_checking_system_deps"))
        deps_status = sys_dep_mgr.check_all()
        sys_dep_mgr.display_status(deps_status)

        missing = [d for d in deps_status if not d["installed"]]
        if missing:
            if not self.yes:
                console.print()
                if not confirm(t("install_deps_install_prompt")):
                    print_warning(t("install_deps_skipped"))
                else:
                    if not sys_dep_mgr.install_missing(yes=True):
                        print_error(t("install_deps_failed"))
                        raise typer.Exit(1)
            else:
                if not sys_dep_mgr.install_missing(yes=True):
                    print_error(t("install_deps_failed"))
                    raise typer.Exit(1)

        # Step 2: Detect CPU and configure environment variables
        console.print()
        print_step(t("install_auto_detect_cpu"))
        cpu_features = detect_cpu_build_features()

        # Display detected features
        features_list = []
        if cpu_features.has_amx:
            features_list.append("AMX")
        if cpu_features.has_avx512:
            features_list.append("AVX512")
        if cpu_features.has_avx512_vnni:
            features_list.append("AVX512_VNNI")
        if cpu_features.has_avx512_bf16:
            features_list.append("AVX512_BF16")
        if cpu_features.has_avx2:
            features_list.append("AVX2")

        if features_list:
            print_info(t("install_cpu_features", features=", ".join(features_list)))
        else:
            print_warning(t("install_cpu_no_features"))

        # Configure environment variables
        env = os.environ.copy()

        # CPU instruction set
        if self.cpu_instruct:
            env["CPUINFER_CPU_INSTRUCT"] = self.cpu_instruct.upper()
        else:
            env["CPUINFER_CPU_INSTRUCT"] = cpu_features.recommended_instruct

        # AMX support
        if self.enable_amx is not None:
            env["CPUINFER_ENABLE_AMX"] = "ON" if self.enable_amx else "OFF"
        elif cpu_features.has_amx:
            env["CPUINFER_ENABLE_AMX"] = "ON"
        else:
            env["CPUINFER_ENABLE_AMX"] = "OFF"

        # Build type
        env["CPUINFER_BUILD_TYPE"] = self.build_type

        # Step 3: Display build configuration
        console.print()
        print_step(t("install_build_config"))
        console.print(f"  CPUINFER_CPU_INSTRUCT = {env['CPUINFER_CPU_INSTRUCT']}")
        console.print(f"  CPUINFER_ENABLE_AMX   = {env['CPUINFER_ENABLE_AMX']}")
        console.print(f"  CPUINFER_BUILD_TYPE   = {env['CPUINFER_BUILD_TYPE']}")

        if env["CPUINFER_CPU_INSTRUCT"] == "NATIVE":
            console.print()
            print_warning(t("install_native_warning"))

        # Step 4: Install PyTorch if needed
        if not self.skip_torch:
            console.print()
            print_step(t("install_installing_pytorch"))
            _install_torch()

        # Step 5: Install kt-kernel from source
        console.print()
        print_step(t("install_building_from_source"))
        console.print()

        pip_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "kt-kernel",
            "--no-binary",
            "kt-kernel",
            "-v",
        ]

        try:
            # Run pip install with custom environment
            process = subprocess.Popen(
                pip_cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Show real-time output (simplified)
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    # Filter to show only important lines
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "building",
                            "compiling",
                            "cmake",
                            "installing",
                            "error",
                            "warning",
                        ]
                    ):
                        console.print(f"  [dim]{line[:80]}{'...' if len(line) > 80 else ''}[/dim]")

            process.wait()

            if process.returncode != 0:
                print_error(t("install_build_failed"))
                raise typer.Exit(1)

            print_success(t("install_build_success"))

        except subprocess.CalledProcessError as e:
            print_error(f"{t('install_build_failed')}: {e}")
            raise typer.Exit(1)
        except FileNotFoundError:
            print_error("pip not found")
            raise typer.Exit(1)

        # Step 6: Install other dependencies based on mode
        if self.mode != InstallMode.INFERENCE or self.mode == InstallMode.FULL:
            console.print()
            print_step(t("install_installing_deps"))
            _install_other_deps(self.mode)

        # Step 7: Verify installation
        if self.verify:
            console.print()
            print_step(t("install_verifying"))
            result = _verify_kt_kernel_installation()
            _display_verification_result(result)


class LocalSourceStrategy(InstallationStrategy):
    """Install from local source directory."""

    def __init__(
        self,
        mode: InstallMode,
        skip_torch: bool,
        source: Path,
        editable: bool,
        yes: bool,
    ):
        super().__init__(mode, skip_torch, yes)
        self.source = source
        self.editable = editable

    def execute(self) -> None:
        """Execute local source installation."""
        from .helpers import _install_other_deps, _install_torch

        source = Path(self.source).resolve()

        # Check if source exists
        if not source.exists():
            print_error(f"Source directory not found: {source}")
            raise typer.Exit(1)

        # Detect what's in the source directory
        console.print()
        print_step("Analyzing source directory...")

        packages_to_install = []

        # Check for kt-kernel
        kt_kernel_path = source / "kt-kernel"
        if kt_kernel_path.exists() and (kt_kernel_path / "pyproject.toml").exists():
            packages_to_install.append(("kt-kernel", kt_kernel_path))
            print_info(f"Found kt-kernel at: {kt_kernel_path}")

        # Check for kt-sft (ktransformers)
        kt_sft_path = source / "kt-sft"
        if kt_sft_path.exists() and (kt_sft_path / "pyproject.toml").exists():
            packages_to_install.append(("ktransformers", kt_sft_path))
            print_info(f"Found ktransformers at: {kt_sft_path}")

        # Check if source itself is a package
        if (source / "pyproject.toml").exists() or (source / "setup.py").exists():
            # Determine package name from directory name
            pkg_name = source.name
            if pkg_name in ("kt-kernel", "ktransformers", "kt-sft"):
                packages_to_install.append((pkg_name, source))
                print_info(f"Found {pkg_name} at: {source}")

        if not packages_to_install:
            print_error("No installable packages found in source directory")
            console.print()
            console.print("Expected structure:")
            console.print("  <source>/kt-kernel/pyproject.toml")
            console.print("  <source>/kt-sft/pyproject.toml")
            console.print("Or:")
            console.print("  <source>/pyproject.toml (for single package)")
            raise typer.Exit(1)

        # Install PyTorch first if needed
        if not self.skip_torch:
            console.print()
            print_step("Installing PyTorch...")
            _install_torch()

        # Install packages from source
        console.print()
        print_step("Building and installing from source...")
        console.print()

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if self.editable:
            pip_cmd.append("-e")

        for pkg_name, pkg_path in packages_to_install:
            console.print(f"  Installing {pkg_name}...")
            try:
                subprocess.run(
                    pip_cmd + [str(pkg_path)],
                    check=True,
                    cwd=pkg_path,
                )
                print_success(f"  {pkg_name} installed successfully")
            except subprocess.CalledProcessError as e:
                print_error(f"  Failed to install {pkg_name}: {e}")
                raise typer.Exit(1)

        # Install other dependencies based on mode
        console.print()
        print_step("Installing additional dependencies...")
        _install_other_deps(self.mode)
