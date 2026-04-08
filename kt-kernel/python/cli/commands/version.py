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
    """Get sglang-kt version and installation source information."""
    from kt_kernel.cli.utils.sglang_checker import check_sglang_installation

    info = check_sglang_installation()

    if not info["installed"]:
        return t("version_not_installed")

    # Get version from package metadata (prefer sglang-kt)
    version = get_installed_package_version("sglang-kt")
    if not version:
        version = get_installed_package_version("sglang")
    if not version:
        version = info.get("version") or "unknown"

    # Determine source label
    if info.get("is_kvcache_fork"):
        if info["from_source"] and info.get("git_info"):
            git_remote = info["git_info"].get("remote", "")
            return f"{version} [dim](Source: {git_remote})[/dim]"
        elif info["editable"]:
            return f"{version} [dim](editable)[/dim]"
        else:
            return f"{version} [dim](sglang-kt)[/dim]"
    elif info["from_source"]:
        if info.get("git_info"):
            git_remote = info["git_info"].get("remote", "")
            return f"{version} [dim](Source: {git_remote})[/dim]"
        return f"{version} [dim](source)[/dim]"
    else:
        return f"{version} [dim](PyPI)[/dim]"


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

    sglang_info = _get_sglang_info()
    key_packages = {
        t("version_kt_kernel"): get_installed_package_version("kt-kernel") or t("version_not_installed"),
        t("version_sglang"): sglang_info,
    }

    print_version_table(key_packages)

    # Show SGLang installation hint if not installed
    if sglang_info == t("version_not_installed"):
        from kt_kernel.cli.utils.sglang_checker import print_sglang_install_instructions

        console.print()
        print_sglang_install_instructions()

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
