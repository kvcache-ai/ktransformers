"""
System dependency management for source builds.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass

from rich.table import Table

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    console,
    print_error,
    print_step,
    print_success,
    print_warning,
)


@dataclass
class SystemDep:
    """System dependency information."""

    name: str
    display_name: str
    check_command: list[str]
    install_commands: dict[str, list[str]]  # os_type -> command
    required: bool = True


# System dependencies required for source builds
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


class SystemDependencyManager:
    """Manages system dependencies for source builds."""

    def __init__(self):
        self.deps = SYSTEM_DEPS
        self.os_type = self._detect_os_type()

    def _detect_os_type(self) -> str:
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

    def _check_dep(self, dep: SystemDep) -> bool:
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

    def check_all(self) -> list[dict]:
        """Check all system dependencies and return status list."""
        results = []
        for dep in self.deps:
            installed = self._check_dep(dep)
            results.append(
                {
                    "name": dep.name,
                    "display_name": dep.display_name,
                    "installed": installed,
                    "required": dep.required,
                }
            )
        return results

    def display_status(self, deps_status: list[dict]) -> None:
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

    def install_missing(self, yes: bool = False) -> bool:
        """Install missing system dependencies.

        Args:
            yes: Skip confirmation prompts

        Returns:
            True if all dependencies installed successfully, False otherwise
        """
        # Check if conda is available (preferred for cmake)
        has_conda = shutil.which("conda") is not None

        missing_deps = [dep for dep in self.deps if not self._check_dep(dep)]

        if not missing_deps:
            print_success(t("install_deps_all_installed"))
            return True

        console.print()
        print_step(t("install_installing_system_deps"))

        for dep in missing_deps:
            # Choose install command
            if dep.name == "cmake" and has_conda and "conda" in dep.install_commands:
                cmd = dep.install_commands["conda"]
            elif self.os_type in dep.install_commands:
                cmd = dep.install_commands[self.os_type]
            else:
                print_warning(t("install_dep_no_install_cmd", name=dep.display_name, os=self.os_type))
                continue

            console.print(f"  {t('install_installing_dep', name=dep.display_name)}...")
            try:
                subprocess.run(cmd, check=True)
                print_success(f"  {dep.display_name} {t('install_dep_ok')}")
            except subprocess.CalledProcessError as e:
                print_error(f"  {t('install_dep_install_failed', name=dep.display_name)}: {e}")
                return False

        return True
