"""
Install command for kt-cli.

Handles installation of KTransformers and its dependencies.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    create_progress,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from kt_kernel.cli.utils.environment import (
    detect_cpu_build_features,
    detect_cuda_version,
    detect_env_managers,
    get_current_env_name,
    get_installed_package_version,
    is_in_virtual_env,
)


class InstallMode(str, Enum):
    """Installation mode."""

    INFERENCE = "inference"
    SFT = "sft"
    FULL = "full"


# Requirements files location
REQUIREMENTS_DIR = Path(__file__).parent.parent / "requirements"

# Package requirements for each mode (fallback if requirements files not found)
REQUIREMENTS = {
    "inference": [
        "torch>=2.4.0",
        "kt-kernel>=0.4.0",
        "sglang>=0.4.0",
        "transformers>=4.45.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.20.0",
    ],
    "sft": [
        "torch>=2.4.0",
        "ktransformers>=0.4.0",
        "llamafactory>=0.9.0",
        "peft>=0.12.0",
        "transformers>=4.45.0",
        "datasets>=2.14.0",
        "accelerate>=0.30.0",
    ],
}

def get_requirements_file(mode: str) -> Optional[Path]:
    """Get the requirements file path for a given mode."""
    req_file = REQUIREMENTS_DIR / f"{mode}.txt"
    if req_file.exists():
        return req_file
    return None

# Source repositories
SOURCE_REPOS = {
    "kt-kernel": {
        "repo": "https://github.com/kvcache-ai/ktransformers.git",
        "subdir": "kt-kernel",
        "branch": "main",
    },
    "ktransformers": {
        "repo": "https://github.com/kvcache-ai/ktransformers.git",
        "subdir": "kt-sft",
        "branch": "main",
    },
}


# System dependencies for source builds
@dataclass
class SystemDep:
    """System dependency information."""

    name: str
    display_name: str
    check_command: list[str]
    install_commands: dict[str, list[str]]  # os_type -> command
    required: bool = True


SYSTEM_DEPS = [
    SystemDep(
        name="cmake",
        display_name="CMake",
        check_command=["cmake", "--version"],
        install_commands={
            "conda": ["conda", "install", "-y", "cmake"],
            "debian": ["sudo", "apt", "install", "-y", "cmake"],
            "fedora": ["sudo", "dnf", "install", "-y", "cmake"],
            "arch": ["sudo", "pacman", "-S", "--noconfirm", "cmake"],
        },
    ),
    SystemDep(
        name="hwloc",
        display_name="libhwloc-dev",
        check_command=["pkg-config", "--exists", "hwloc"],
        install_commands={
            "debian": ["sudo", "apt", "install", "-y", "libhwloc-dev"],
            "fedora": ["sudo", "dnf", "install", "-y", "hwloc-devel"],
            "arch": ["sudo", "pacman", "-S", "--noconfirm", "hwloc"],
        },
    ),
    SystemDep(
        name="pkg-config",
        display_name="pkg-config",
        check_command=["pkg-config", "--version"],
        install_commands={
            "debian": ["sudo", "apt", "install", "-y", "pkg-config"],
            "fedora": ["sudo", "dnf", "install", "-y", "pkgconfig"],
            "arch": ["sudo", "pacman", "-S", "--noconfirm", "pkgconf"],
        },
    ),
]


def _detect_os_type() -> str:
    """Detect OS type for package management."""
    if os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read()
            for line in content.split("\n"):
                if line.startswith("ID="):
                    os_id = line.split("=")[1].strip().strip('"').lower()
                    if os_id in ("debian", "ubuntu", "linuxmint", "pop"):
                        return "debian"
                    elif os_id in ("fedora", "rhel", "centos", "rocky", "almalinux"):
                        return "fedora"
                    elif os_id in ("arch", "manjaro"):
                        return "arch"
        except (OSError, IOError):
            pass

    # Check for Debian-based
    if os.path.exists("/etc/debian_version"):
        return "debian"
    # Check for Red Hat-based
    if os.path.exists("/etc/redhat-release"):
        return "fedora"

    return "unknown"


def _check_system_dep(dep: SystemDep) -> bool:
    """Check if a system dependency is installed."""
    try:
        result = subprocess.run(
            dep.check_command,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _check_all_system_deps() -> list[dict]:
    """Check all system dependencies and return status list."""
    results = []
    for dep in SYSTEM_DEPS:
        installed = _check_system_dep(dep)
        results.append({
            "name": dep.name,
            "display_name": dep.display_name,
            "installed": installed,
            "required": dep.required,
        })
    return results


def _display_deps_table(deps_status: list[dict]) -> None:
    """Display system dependencies status table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column(t("install_dep_name"))
    table.add_column(t("install_dep_status"))

    for dep in deps_status:
        if dep["installed"]:
            status = f"[green]{t('install_dep_ok')}[/green]"
        else:
            status = f"[red]{t('install_dep_missing')}[/red]"
        table.add_row(dep["display_name"], status)

    console.print(table)


def _install_system_deps(yes: bool = False) -> bool:
    """Install missing system dependencies."""
    os_type = _detect_os_type()

    # Check if conda is available (preferred for cmake)
    has_conda = shutil.which("conda") is not None

    missing_deps = [
        dep for dep in SYSTEM_DEPS
        if not _check_system_dep(dep)
    ]

    if not missing_deps:
        print_success(t("install_deps_all_installed"))
        return True

    console.print()
    print_step(t("install_installing_system_deps"))

    for dep in missing_deps:
        # Choose install command
        if dep.name == "cmake" and has_conda and "conda" in dep.install_commands:
            cmd = dep.install_commands["conda"]
        elif os_type in dep.install_commands:
            cmd = dep.install_commands[os_type]
        else:
            print_warning(t("install_dep_no_install_cmd", name=dep.display_name, os=os_type))
            continue

        console.print(f"  {t('install_installing_dep', name=dep.display_name)}...")
        try:
            subprocess.run(cmd, check=True)
            print_success(f"  {dep.display_name} {t('install_dep_ok')}")
        except subprocess.CalledProcessError as e:
            print_error(f"  {t('install_dep_install_failed', name=dep.display_name)}: {e}")
            return False

    return True


def _verify_kt_kernel_installation() -> dict:
    """
    Verify kt-kernel installation.

    Returns dict with success, version, cpu_variant, error.
    """
    try:
        # Need to reimport to get fresh module
        import importlib
        import kt_kernel
        importlib.reload(kt_kernel)

        return {
            "success": True,
            "version": getattr(kt_kernel, "__version__", "unknown"),
            "cpu_variant": getattr(kt_kernel, "__cpu_variant__", "unknown"),
            "error": None,
        }
    except ImportError as e:
        return {
            "success": False,
            "version": None,
            "cpu_variant": None,
            "error": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "version": None,
            "cpu_variant": None,
            "error": str(e),
        }


def _display_verification_result(result: dict) -> None:
    """Display kt-kernel verification result."""
    if result["success"]:
        print_success(
            t("install_verify_success",
              version=result["version"],
              variant=result["cpu_variant"])
        )
    else:
        print_error(t("install_verify_failed", error=result["error"]))


def _show_docker_guide() -> None:
    """Show Docker installation guide."""
    console.print()
    print_info(t("install_docker_guide_title"))
    console.print()
    console.print(t("install_docker_guide_desc"))
    console.print()
    console.print("  [cyan]https://github.com/kvcache-ai/ktransformers/tree/main/docker[/cyan]")
    console.print()


def install(
    mode: InstallMode = typer.Argument(
        InstallMode.INFERENCE,
        help="Installation mode: inference, sft, or full",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Install from source directory (local path or will clone from GitHub)",
    ),
    branch: str = typer.Option(
        "main",
        "--branch",
        "-b",
        help="Git branch to use when cloning from GitHub",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force reinstall, ignore version checks",
    ),
    skip_torch: bool = typer.Option(
        False,
        "--skip-torch",
        help="Skip PyTorch installation (if already installed)",
    ),
    editable: bool = typer.Option(
        False,
        "--editable",
        "-e",
        help="Install in editable/development mode (requires --source)",
    ),
    # New options for source builds
    from_source: bool = typer.Option(
        False,
        "--from-source",
        help="Build from source (PyPI sdist) instead of using pre-built wheel",
    ),
    cpu_instruct: Optional[str] = typer.Option(
        None,
        "--cpu-instruct",
        help="CPU instruction set: NATIVE, AVX512, AVX2, FANCY (--from-source only)",
    ),
    enable_amx: Optional[bool] = typer.Option(
        None,
        "--enable-amx/--disable-amx",
        help="Enable/disable Intel AMX support (--from-source only)",
    ),
    build_type: str = typer.Option(
        "Release",
        "--build-type",
        help="Build type: Release, Debug, RelWithDebInfo (--from-source only)",
    ),
    deps_only: bool = typer.Option(
        False,
        "--deps-only",
        help="Install system dependencies only (for source builds)",
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Show Docker installation guide",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify installation after completion",
    ),
) -> None:
    """Install KTransformers and dependencies.

    By default, installs from PyPI (pre-built wheel). Use --from-source to build
    from source code for optimized performance on your CPU.

    Examples:
        kt install                          # Install from PyPI (wheel)
        kt install --from-source            # Build from source (NATIVE optimization)
        kt install --from-source --cpu-instruct AVX512  # Build for AVX512
        kt install --from-source --disable-amx  # Build without AMX
        kt install --deps-only              # Install system dependencies only
        kt install --docker                 # Show Docker installation guide
        kt install --source /path/to/repo   # Install from local source directory
    """
    console.print()

    # Handle --docker option first
    if docker:
        _show_docker_guide()
        return

    # Handle --deps-only option
    if deps_only:
        print_step(t("install_checking_system_deps"))
        deps_status = _check_all_system_deps()
        _display_deps_table(deps_status)

        missing = [d for d in deps_status if not d["installed"]]
        if missing:
            if not yes:
                console.print()
                if not confirm(t("install_deps_install_prompt")):
                    raise typer.Abort()
            _install_system_deps(yes=True)
        else:
            print_success(t("install_deps_all_installed"))
        return

    # Validate options
    if editable and not source:
        print_error("--editable requires --source to be specified")
        raise typer.Exit(1)

    if cpu_instruct and not from_source and not source:
        print_warning("--cpu-instruct is only effective with --from-source or --source")

    if enable_amx is not None and not from_source and not source:
        print_warning("--enable-amx/--disable-amx is only effective with --from-source or --source")

    # Step 1: Check if in virtual environment
    if not is_in_virtual_env():
        _show_venv_warning()
        if not yes:
            if not confirm(t("install_continue_without_venv"), default=False):
                console.print()
                _show_venv_instructions()
                raise typer.Exit(0)
    else:
        env_name = get_current_env_name()
        print_success(t("install_in_venv", name=env_name or "virtual environment"))

    # Step 2: Determine installation method
    if source:
        # Local source installation
        console.print()
        print_info(f"Installation method: [bold]Local Source[/bold] ({source})")
        _install_from_source(mode, source, branch, editable, skip_torch, yes)
    elif from_source:
        # PyPI source (sdist) installation
        console.print()
        print_info("Installation method: [bold]PyPI Source (sdist)[/bold]")
        _install_from_source_pypi(
            mode, cpu_instruct, enable_amx, build_type, skip_torch, yes, verify
        )
    else:
        # PyPI wheel installation (default)
        console.print()
        print_info("Installation method: [bold]PyPI (wheel)[/bold]")
        _install_from_pypi(mode, skip_torch, force, yes)

    # Step 3: Verify installation if requested
    if verify and not source:  # source has its own verification
        console.print()
        print_step(t("install_verifying"))
        result = _verify_kt_kernel_installation()
        _display_verification_result(result)

    # Step 4: Show completion message
    console.print()
    print_success(t("install_complete"))
    console.print()
    console.print(f"  {t('install_start_hint')}")
    console.print()
    console.print("  [dim]Tip: Run 'kt doctor' to verify your environment[/dim]")
    console.print()


def update(
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Update from source directory (git pull + rebuild)",
    ),
    pypi: bool = typer.Option(
        False,
        "--pypi",
        help="Update from PyPI",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Update KTransformers to the latest version.

    You must specify either --source or --pypi to indicate the update method.

    Examples:
        kt update --pypi                    # Update from PyPI
        kt update --source /path/to/repo    # Update from source (git pull + rebuild)
    """
    console.print()

    # Validate: must specify one method
    if not source and not pypi:
        print_error("Please specify update method: --pypi or --source <path>")
        console.print()
        console.print("Examples:")
        console.print("  kt update --pypi                  # Update from PyPI")
        console.print("  kt update --source /path/to/repo  # Update from source")
        raise typer.Exit(1)

    if source and pypi:
        print_error("Cannot specify both --source and --pypi")
        raise typer.Exit(1)

    # Check if in virtual environment
    if not is_in_virtual_env():
        _show_venv_warning()
        if not yes:
            if not confirm(t("install_continue_without_venv"), default=False):
                raise typer.Exit(0)

    if pypi:
        _update_from_pypi(yes)
    else:
        _update_from_source(source, yes)


def _install_from_pypi(mode: InstallMode, skip_torch: bool, force: bool, yes: bool) -> None:
    """Install packages from PyPI."""
    # Check existing packages
    console.print()
    print_step(t("install_checking_deps"))

    deps = _check_dependencies(mode)
    missing = [d for d in deps if d["status"] == "missing"]
    outdated = [d for d in deps if d["status"] == "outdated"]

    _display_dependency_table(deps)

    if not missing and not outdated and not force:
        console.print()
        print_success(t("install_already_installed"))
        return

    # Confirm installation
    if not yes:
        console.print()
        to_install = len(missing) + len(outdated)
        if not confirm(t("install_confirm", count=to_install)):
            raise typer.Abort()

    # Install packages
    console.print()
    print_step(t("install_installing_deps"))

    _install_packages_pip(mode, skip_torch)


def _install_from_source_pypi(
    mode: InstallMode,
    cpu_instruct: Optional[str],
    enable_amx: Optional[bool],
    build_type: str,
    skip_torch: bool,
    yes: bool,
    verify: bool,
) -> None:
    """Install kt-kernel from PyPI source (sdist) with custom build configuration."""
    # Step 1: Check and install system dependencies
    console.print()
    print_step(t("install_checking_system_deps"))
    deps_status = _check_all_system_deps()
    _display_deps_table(deps_status)

    missing = [d for d in deps_status if not d["installed"]]
    if missing:
        if not yes:
            console.print()
            if not confirm(t("install_deps_install_prompt")):
                print_warning(t("install_deps_skipped"))
            else:
                if not _install_system_deps(yes=True):
                    print_error(t("install_deps_failed"))
                    raise typer.Exit(1)
        else:
            if not _install_system_deps(yes=True):
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
    if cpu_instruct:
        env["CPUINFER_CPU_INSTRUCT"] = cpu_instruct.upper()
    else:
        env["CPUINFER_CPU_INSTRUCT"] = cpu_features.recommended_instruct

    # AMX support
    if enable_amx is not None:
        env["CPUINFER_ENABLE_AMX"] = "ON" if enable_amx else "OFF"
    elif cpu_features.has_amx:
        env["CPUINFER_ENABLE_AMX"] = "ON"
    else:
        env["CPUINFER_ENABLE_AMX"] = "OFF"

    # Build type
    env["CPUINFER_BUILD_TYPE"] = build_type

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
    if not skip_torch:
        console.print()
        print_step(t("install_installing_pytorch"))
        _install_torch()

    # Step 5: Install kt-kernel from source
    console.print()
    print_step(t("install_building_from_source"))
    console.print()

    pip_cmd = [
        sys.executable, "-m", "pip", "install",
        "kt-kernel", "--no-binary", "kt-kernel", "-v"
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
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                # Filter to show only important lines
                if any(keyword in line.lower() for keyword in [
                    "building", "compiling", "cmake", "installing", "error", "warning"
                ]):
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
    if mode != InstallMode.INFERENCE or mode == InstallMode.FULL:
        console.print()
        print_step(t("install_installing_deps"))
        _install_other_deps(mode)

    # Step 7: Verify installation
    if verify:
        console.print()
        print_step(t("install_verifying"))
        result = _verify_kt_kernel_installation()
        _display_verification_result(result)


def _install_from_source(
    mode: InstallMode,
    source: Path,
    branch: str,
    editable: bool,
    skip_torch: bool,
    yes: bool,
) -> None:
    """Install packages from source code."""
    source = Path(source).resolve()

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
    if not skip_torch:
        console.print()
        print_step("Installing PyTorch...")
        _install_torch()

    # Install packages from source
    console.print()
    print_step("Building and installing from source...")
    console.print()

    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
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
    _install_other_deps(mode)


def _update_from_pypi(yes: bool) -> None:
    """Update packages from PyPI."""
    print_step("Checking installed packages...")

    packages = [
        ("kt-kernel", "kt-kernel"),
        ("ktransformers", "ktransformers"),
        ("sglang", "sglang"),
    ]

    installed = []
    for display_name, pkg_name in packages:
        current = get_installed_package_version(pkg_name)
        if current:
            installed.append((display_name, pkg_name, current))
            print_info(f"{display_name}: {current}")

    if not installed:
        print_warning("No KTransformers packages found. Run 'kt install' first.")
        return

    if not yes:
        console.print()
        if not confirm("Update these packages from PyPI?"):
            raise typer.Abort()

    console.print()
    print_step("Updating packages from PyPI...")

    with create_progress() as progress:
        task = progress.add_task("Updating...", total=len(installed))

        for display_name, pkg_name, _ in installed:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", pkg_name],
                    capture_output=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                print_warning(f"Failed to update {display_name}")
            progress.advance(task)

    console.print()
    print_success("Update complete!")


def _update_from_source(source: Path, yes: bool) -> None:
    """Update packages from source (git pull + rebuild)."""
    source = Path(source).resolve()

    if not source.exists():
        print_error(f"Source directory not found: {source}")
        raise typer.Exit(1)

    # Check if it's a git repository
    git_dir = source / ".git"
    if not git_dir.exists():
        print_error(f"Not a git repository: {source}")
        raise typer.Exit(1)

    print_step(f"Updating source from git: {source}")

    # Git pull
    console.print()
    print_info("Running git pull...")
    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=source,
            capture_output=True,
            text=True,
            check=True,
        )
        console.print(f"  [dim]{result.stdout.strip()}[/dim]")
    except subprocess.CalledProcessError as e:
        print_error(f"Git pull failed: {e.stderr}")
        raise typer.Exit(1)

    # Find and rebuild packages
    packages_to_rebuild = []

    kt_kernel_path = source / "kt-kernel"
    if kt_kernel_path.exists() and (kt_kernel_path / "pyproject.toml").exists():
        packages_to_rebuild.append(("kt-kernel", kt_kernel_path))

    kt_sft_path = source / "kt-sft"
    if kt_sft_path.exists() and (kt_sft_path / "pyproject.toml").exists():
        packages_to_rebuild.append(("ktransformers", kt_sft_path))

    if not packages_to_rebuild:
        print_warning("No packages found to rebuild")
        return

    if not yes:
        console.print()
        pkg_names = ", ".join(p[0] for p in packages_to_rebuild)
        if not confirm(f"Rebuild {pkg_names}?"):
            raise typer.Abort()

    # Rebuild packages
    console.print()
    print_step("Rebuilding packages...")

    for pkg_name, pkg_path in packages_to_rebuild:
        console.print(f"  Rebuilding {pkg_name}...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_path)],
                check=True,
                cwd=pkg_path,
            )
            print_success(f"  {pkg_name} rebuilt successfully")
        except subprocess.CalledProcessError as e:
            print_error(f"  Failed to rebuild {pkg_name}: {e}")

    console.print()
    print_success("Update complete!")


def _install_torch() -> None:
    """Install PyTorch with CUDA support if available."""
    cuda_version = detect_cuda_version()
    pip_cmd = [sys.executable, "-m", "pip", "install"]

    try:
        if cuda_version:
            cuda_major = cuda_version.split(".")[0]
            if cuda_major == "12":
                index_url = "https://download.pytorch.org/whl/cu121"
            else:
                index_url = "https://download.pytorch.org/whl/cu118"

            subprocess.run(
                pip_cmd + ["torch", "torchvision", "torchaudio", "--index-url", index_url],
                check=True,
            )
        else:
            subprocess.run(
                pip_cmd + ["torch", "torchvision", "torchaudio"],
                check=True,
            )
        print_success("PyTorch installed")
    except subprocess.CalledProcessError as e:
        print_warning(f"Failed to install PyTorch: {e}")


def _install_other_deps(mode: InstallMode) -> None:
    """Install other dependencies based on mode."""
    packages = REQUIREMENTS.get(mode.value, [])
    if mode == InstallMode.FULL:
        packages = list(set(REQUIREMENTS["inference"] + REQUIREMENTS["sft"]))

    # Filter out torch and kt-kernel/ktransformers (already installed)
    other_packages = [
        p for p in packages
        if not p.startswith("torch") and
        not p.startswith("kt-kernel") and
        not p.startswith("ktransformers")
    ]

    pip_cmd = [sys.executable, "-m", "pip", "install"]

    for pkg in other_packages:
        try:
            subprocess.run(
                pip_cmd + [pkg],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            print_warning(f"Failed to install {pkg}")


def _install_packages_pip(mode: InstallMode, skip_torch: bool) -> None:
    """Install packages using pip from PyPI with real-time progress."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    import re

    pip_cmd = [sys.executable, "-m", "pip", "install"]

    # Install PyTorch first if needed
    if not skip_torch:
        console.print()
        print_step(t("install_installing_pytorch"))
        cuda_version = detect_cuda_version()

        if cuda_version:
            cuda_major = cuda_version.split(".")[0]
            if cuda_major == "12":
                index_url = "https://download.pytorch.org/whl/cu124"
            else:
                index_url = "https://download.pytorch.org/whl/cu118"
            torch_cmd = pip_cmd + ["torch", "torchvision", "torchaudio", "--index-url", index_url]
        else:
            torch_cmd = pip_cmd + ["torch", "torchvision", "torchaudio"]

        _run_pip_realtime(torch_cmd, "PyTorch")

    # Install from requirements file if available
    req_file = get_requirements_file(mode.value)

    if req_file:
        console.print()
        print_step(t("install_installing_from_requirements"))
        console.print(f"  [dim]{req_file}[/dim]")
        console.print()

        # Use pip install -r with real-time output
        _run_pip_realtime(pip_cmd + ["-r", str(req_file)], "dependencies")
    else:
        # Fallback to individual package installation
        console.print()
        print_step(t("install_installing_deps"))

        packages = REQUIREMENTS.get(mode.value, [])
        if mode == InstallMode.FULL:
            packages = list(set(REQUIREMENTS["inference"] + REQUIREMENTS["sft"]))

        # Filter out torch packages (already installed)
        other_packages = [p for p in packages if not p.startswith("torch")]

        for pkg in other_packages:
            pkg_name = pkg.split(">=")[0].split("==")[0]
            _run_pip_realtime(pip_cmd + [pkg], pkg_name)


def _run_pip_realtime(cmd: list[str], name: str) -> bool:
    """Run pip command with real-time output display."""
    from rich.live import Live
    from rich.text import Text
    import re

    status_text = Text()
    status_text.append(f"📦 {name}: ", style="bold")
    status_text.append("Starting...", style="yellow")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with Live(status_text, refresh_per_second=10, console=console, transient=True) as live:
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if not line:
                    continue

                # Update status based on pip output
                status_text = Text()
                status_text.append(f"📦 {name}: ", style="bold")

                if "Downloading" in line:
                    # Extract size if present
                    size_match = re.search(r'(\d+\.?\d*)\s*(kB|MB|GB)', line)
                    if size_match:
                        status_text.append(f"Downloading ({size_match.group()})", style="yellow")
                    else:
                        # Extract filename
                        file_match = re.search(r'Downloading\s+(\S+)', line)
                        if file_match:
                            filename = file_match.group(1).split('/')[-1]
                            if len(filename) > 40:
                                filename = filename[:37] + "..."
                            status_text.append(f"Downloading {filename}", style="yellow")
                        else:
                            status_text.append("Downloading...", style="yellow")

                elif "Collecting" in line:
                    pkg_match = re.search(r'Collecting\s+(\S+)', line)
                    if pkg_match:
                        status_text.append(f"Collecting {pkg_match.group(1)}", style="yellow")
                    else:
                        status_text.append("Collecting...", style="yellow")

                elif "Installing" in line or "Building" in line:
                    status_text.append(line[:50] + "..." if len(line) > 50 else line, style="yellow")

                elif "already satisfied" in line.lower():
                    status_text.append("Already installed ✓", style="dim")

                elif "Successfully installed" in line:
                    status_text.append("Installed ✓", style="green")

                else:
                    status_text.append(line[:50] + "..." if len(line) > 50 else line, style="dim")

                live.update(status_text)

        process.wait()

        # Final status
        if process.returncode == 0:
            console.print(f"[green]✓[/green] {name} installed successfully")
            return True
        else:
            console.print(f"[red]✗[/red] {name} installation failed")
            return False

    except Exception as e:
        console.print(f"[red]✗[/red] {name} error: {e}")
        return False


def _show_venv_warning() -> None:
    """Show warning about not being in a virtual environment."""
    console.print()
    console.print("[bold yellow]⚠ Warning: Not in a virtual environment[/bold yellow]")
    console.print()
    console.print("  Installing packages to your system Python may cause conflicts")
    console.print("  with other projects. It's recommended to use a virtual environment.")
    console.print()


def _show_venv_instructions() -> None:
    """Show instructions for creating a virtual environment."""
    console.print("[bold]How to create a virtual environment:[/bold]")
    console.print()

    env_managers = detect_env_managers()
    manager_names = [m.name for m in env_managers]

    if "conda" in manager_names or "mamba" in manager_names:
        console.print("  [cyan]Using conda/mamba:[/cyan]")
        console.print("    conda create -n kt python=3.10")
        console.print("    conda activate kt")
        console.print("    kt install")
        console.print()

    if "uv" in manager_names:
        console.print("  [cyan]Using uv (fast):[/cyan]")
        console.print("    uv venv kt-env")
        console.print("    source kt-env/bin/activate")
        console.print("    kt install")
        console.print()

    console.print("  [cyan]Using venv (built-in):[/cyan]")
    console.print("    python -m venv kt-env")
    console.print("    source kt-env/bin/activate")
    console.print("    kt install")
    console.print()


def _check_dependencies(mode: InstallMode) -> list[dict]:
    """Check installed package versions against requirements."""
    deps = []
    packages = REQUIREMENTS.get(mode.value, [])
    if mode == InstallMode.FULL:
        packages = list(set(REQUIREMENTS["inference"] + REQUIREMENTS["sft"]))

    for pkg_spec in packages:
        if ">=" in pkg_spec:
            name, required = pkg_spec.split(">=")
        elif "==" in pkg_spec:
            name, required = pkg_spec.split("==")
        else:
            name, required = pkg_spec, "any"

        installed = get_installed_package_version(name)

        if installed is None:
            status = "missing"
        elif required == "any":
            status = "ok"
        else:
            status = "ok" if _compare_versions(installed, required) else "outdated"

        deps.append({
            "name": name,
            "installed": installed or "-",
            "required": f">={required}",
            "status": status,
        })

    return deps


def _compare_versions(installed: str, required: str) -> bool:
    """Compare version strings. Returns True if installed >= required."""
    try:
        from packaging import version
        return version.parse(installed) >= version.parse(required)
    except Exception:
        return installed >= required


def _display_dependency_table(deps: list[dict]) -> None:
    """Display dependency status table."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("Package")
    table.add_column("Installed")
    table.add_column("Required")
    table.add_column("Status")

    for dep in deps:
        status = dep["status"]
        if status == "ok":
            status_str = f"[green]{t('install_dep_ok')}[/green]"
        elif status == "outdated":
            status_str = f"[yellow]{t('install_dep_outdated')}[/yellow]"
        else:
            status_str = f"[red]{t('install_dep_missing')}[/red]"

        table.add_row(
            dep["name"],
            dep["installed"],
            dep["required"],
            status_str,
        )

    console.print(table)
