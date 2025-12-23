"""
Version command for kt-cli.

Displays version information for kt-cli and related packages.
"""

import platform
from typing import Optional

import typer

from kt_kernel.cli import __version__
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import console, print_version_table
from kt_kernel.cli.utils.environment import detect_cuda_version, get_installed_package_version


def _get_sglang_info() -> str:
    """Get sglang version and installation source information."""
    try:
        import sglang
        version = getattr(sglang, "__version__", None)

        if not version:
            version = get_installed_package_version("sglang")

        if not version:
            return t("version_not_installed")

        # Try to detect installation source
        from pathlib import Path
        import subprocess

        if hasattr(sglang, "__file__") and sglang.__file__:
            location = Path(sglang.__file__).parent.parent
            git_dir = location / ".git"

            if git_dir.exists():
                # Installed from git (editable install)
                try:
                    # Get remote URL
                    result = subprocess.run(
                        ["git", "remote", "get-url", "origin"],
                        cwd=location,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if result.returncode == 0:
                        remote_url = result.stdout.strip()
                        # Simplify GitHub URLs
                        if "github.com" in remote_url:
                            repo_name = remote_url.split("/")[-1].replace(".git", "")
                            owner = remote_url.split("/")[-2]
                            return f"{version} [dim](GitHub: {owner}/{repo_name})[/dim]"
                        return f"{version} [dim](Git: {remote_url})[/dim]"
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    pass

        # Default: installed from PyPI
        return f"{version} [dim](PyPI)[/dim]"

    except ImportError:
        return t("version_not_installed")


def version(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed version info"),
) -> None:
    """Show version information."""
    console.print(f"\n[bold]{t('version_info')}[/bold] v{__version__}\n")

    # Basic info
    versions = {
        t("version_python"): platform.python_version(),
        t("version_platform"): f"{platform.system()} {platform.release()}",
    }

    # CUDA version
    cuda_version = detect_cuda_version()
    versions[t("version_cuda")] = cuda_version or t("version_cuda_not_found")

    print_version_table(versions)

    # Always show key packages with installation source
    console.print("\n[bold]Packages:[/bold]\n")

    key_packages = {
        t("version_kt_kernel"): get_installed_package_version("kt-kernel") or t("version_not_installed"),
        t("version_sglang"): _get_sglang_info(),
    }

    print_version_table(key_packages)

    if verbose:
        console.print("\n[bold]Additional Packages:[/bold]\n")

        package_versions = {
            t("version_ktransformers"): get_installed_package_version("ktransformers") or t("version_not_installed"),
            t("version_llamafactory"): get_installed_package_version("llamafactory") or t("version_not_installed"),
            "typer": get_installed_package_version("typer") or t("version_not_installed"),
            "rich": get_installed_package_version("rich") or t("version_not_installed"),
            "torch": get_installed_package_version("torch") or t("version_not_installed"),
            "transformers": get_installed_package_version("transformers") or t("version_not_installed"),
        }

        print_version_table(package_versions)

    console.print()
