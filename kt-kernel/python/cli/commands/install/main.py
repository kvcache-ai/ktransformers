"""
Main entry points for install and update commands.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

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
    detect_env_managers,
    get_current_env_name,
    get_installed_package_version,
    is_in_virtual_env,
)

from .helpers import _display_verification_result, _verify_kt_kernel_installation
from .strategies import InstallMode, LocalSourceStrategy, PyPISourceStrategy, PyPIWheelStrategy
from .system_deps import SystemDependencyManager
from .validation import validate_install_params


def _show_docker_guide() -> None:
    """Show Docker installation guide."""
    console.print()
    print_info(t("install_docker_guide_title"))
    console.print()
    console.print(t("install_docker_guide_desc"))
    console.print()
    console.print("  [cyan]https://github.com/kvcache-ai/ktransformers/tree/main/docker[/cyan]")
    console.print()


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


def _handle_venv_warning() -> None:
    """Handle virtual environment warning and prompt user."""
    _show_venv_warning()
    if not confirm(t("install_continue_without_venv"), default=False):
        console.print()
        _show_venv_instructions()
        raise typer.Exit(0)


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
        sys_dep_mgr = SystemDependencyManager()
        print_step(t("install_checking_system_deps"))
        deps_status = sys_dep_mgr.check_all()
        sys_dep_mgr.display_status(deps_status)

        missing = [d for d in deps_status if not d["installed"]]
        if missing:
            if not yes:
                console.print()
                if not confirm(t("install_deps_install_prompt")):
                    raise typer.Abort()
            sys_dep_mgr.install_missing(yes=True)
        else:
            print_success(t("install_deps_all_installed"))
        return

    # Validate options
    errors = validate_install_params(
        source=source,
        editable=editable,
        cpu_instruct=cpu_instruct,
        enable_amx=enable_amx,
        from_source=from_source,
    )
    if errors:
        for error in errors:
            print_error(error)
        raise typer.Exit(1)

    # Check if in virtual environment
    if not is_in_virtual_env():
        if not yes:
            _handle_venv_warning()
    else:
        env_name = get_current_env_name()
        print_success(t("install_in_venv", name=env_name or "virtual environment"))

    # Select installation strategy
    if source:
        # Local source installation
        console.print()
        print_info(f"Installation method: [bold]Local Source[/bold] ({source})")
        strategy = LocalSourceStrategy(mode, skip_torch, source, editable, yes)
    elif from_source:
        # PyPI source (sdist) installation
        console.print()
        print_info("KT-Kernel installation: [bold]PyPI Source (sdist)[/bold]")
        strategy = PyPISourceStrategy(mode, skip_torch, cpu_instruct, enable_amx, build_type, yes, verify)
    else:
        # PyPI wheel installation (default)
        console.print()
        print_info("KT-Kernel installation: [bold]PyPI (wheel)[/bold]")
        if mode in (InstallMode.INFERENCE, InstallMode.FULL):
            console.print("  [dim]Note: SGLang will be installed from GitHub source[/dim]")
        strategy = PyPIWheelStrategy(mode, skip_torch, force, yes)

    # Execute installation
    strategy.execute()

    # Verify installation if requested (for non-source installations)
    if verify and not source:
        console.print()
        print_step(t("install_verifying"))
        result = _verify_kt_kernel_installation()
        _display_verification_result(result)

    # Show completion message
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
