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

    if verbose:
        console.print("\n[bold]Packages:[/bold]\n")

        package_versions = {
            t("version_kt_kernel"): get_installed_package_version("kt-kernel"),
            t("version_ktransformers"): get_installed_package_version("ktransformers"),
            t("version_sglang"): get_installed_package_version("sglang"),
            t("version_llamafactory"): get_installed_package_version("llamafactory"),
            "typer": get_installed_package_version("typer"),
            "rich": get_installed_package_version("rich"),
            "torch": get_installed_package_version("torch"),
            "transformers": get_installed_package_version("transformers"),
        }

        print_version_table(package_versions)

    console.print()
