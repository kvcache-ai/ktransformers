"""
Install command for kt-cli.

Handles installation of KTransformers and its dependencies.
"""

import os
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from ktcli.i18n import t
from ktcli.utils.console import (
    confirm,
    console,
    create_progress,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from ktcli.utils.environment import (
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
) -> None:
    """Install KTransformers and dependencies.

    By default, installs from PyPI. Use --source to install from source code.

    Examples:
        kt install                          # Install from PyPI
        kt install --source /path/to/repo   # Install from local source
        kt install --source .               # Install from current directory
        kt install -s /path/to/repo -e      # Editable install from source
    """
    console.print()

    # Validate options
    if editable and not source:
        print_error("--editable requires --source to be specified")
        raise typer.Exit(1)

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
        console.print()
        print_info(f"Installation method: [bold]Source[/bold] ({source})")
        _install_from_source(mode, source, branch, editable, skip_torch, yes)
    else:
        console.print()
        print_info("Installation method: [bold]PyPI[/bold]")
        _install_from_pypi(mode, skip_torch, force, yes)

    # Step 3: Show completion message
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
