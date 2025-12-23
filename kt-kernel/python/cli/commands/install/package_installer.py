"""
Generic package installer with support for PyPI and GitHub sources.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.utils.console import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)


@dataclass
class PackageConfig:
    """Configuration for package installation."""

    name: str
    source: str  # "pypi", "github", "local"
    repo_url: Optional[str] = None
    branch: Optional[str] = None
    local_path: Optional[Path] = None
    editable: bool = False
    pip_args: list[str] = field(default_factory=list)


class PackageInstaller:
    """Generic package installer supporting PyPI and GitHub sources.

    Preserves all existing SGLang installation logic including:
    - Version checking
    - Git repository detection
    - Branch/remote verification
    - Editable installation support
    """

    def __init__(self, config: PackageConfig):
        self.config = config

    def check_installation(self) -> dict:
        """Check if package is installed and get its metadata.

        Returns:
            dict with keys:
            - installed: bool
            - version: str or None
            - location: str or None (installation path)
            - editable: bool (whether installed in editable mode)
            - git_info: dict or None (git remote and branch if available)
            - from_source: bool (whether installed from source repository)
        """
        try:
            # Try to import the package
            module = __import__(self.config.name.replace("-", "_"))
            version = getattr(module, "__version__", None)

            # Use pip show to get detailed package information
            location = None
            editable = False
            git_info = None
            from_source = False

            try:
                # Get pip show output
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", self.config.name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    pip_info = {}
                    for line in result.stdout.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            pip_info[key.strip()] = value.strip()

                    location = pip_info.get("Location")
                    editable_location = pip_info.get("Editable project location")

                    if editable_location:
                        editable = True
                        location = editable_location
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                # Fallback to module location
                if hasattr(module, "__file__") and module.__file__:
                    location = str(Path(module.__file__).parent.parent)

            # Check if it's installed from source (has .git directory)
            # Search in location and parent directories (for packages in subdirectories)
            if location:
                git_root = None
                check_path = Path(location)

                # Check current directory and up to 2 parent directories
                for _ in range(3):
                    git_dir = check_path / ".git"
                    if git_dir.exists():
                        git_root = check_path
                        from_source = True
                        break
                    if check_path.parent == check_path:  # Reached root
                        break
                    check_path = check_path.parent

                if from_source and git_root:
                    # Try to get git remote and branch info
                    try:
                        # Get remote URL
                        result = subprocess.run(
                            ["git", "remote", "get-url", "origin"],
                            cwd=git_root,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        remote_url = result.stdout.strip() if result.returncode == 0 else None

                        # Get current branch
                        result = subprocess.run(
                            ["git", "branch", "--show-current"],
                            cwd=git_root,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        branch = result.stdout.strip() if result.returncode == 0 else None

                        if remote_url or branch:
                            git_info = {
                                "remote": remote_url,
                                "branch": branch,
                            }
                    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                        pass

            return {
                "installed": True,
                "version": version,
                "location": location,
                "editable": editable,
                "git_info": git_info,
                "from_source": from_source,
            }
        except ImportError:
            return {
                "installed": False,
                "version": None,
                "location": None,
                "editable": False,
                "git_info": None,
                "from_source": False,
            }

    def needs_reinstall(self, pkg_info: dict) -> tuple[bool, str]:
        """Determine if package needs to be reinstalled.

        Args:
            pkg_info: Result from check_installation()

        Returns:
            Tuple of (needs_reinstall: bool, reason: str)
        """
        if not pkg_info["installed"]:
            return True, "not installed"

        # If we want GitHub source
        if self.config.source == "github":
            # If current installation is not from source (from PyPI), reinstall
            if not pkg_info["from_source"]:
                return True, "installed from PyPI but source installation required"

            # If no git info available, reinstall for safety
            if not pkg_info["git_info"]:
                return True, "source installation found but git info unavailable"

            git_info = pkg_info["git_info"]

            # Check if repo URL matches (normalize URLs for comparison)
            current_remote = git_info.get("remote", "").rstrip("/").replace(".git", "")
            desired_remote = (self.config.repo_url or "").rstrip("/").replace(".git", "")

            if current_remote and desired_remote and current_remote != desired_remote:
                return (
                    True,
                    f"different repository (current: {current_remote}, desired: {desired_remote})",
                )

            # Check if branch matches
            current_branch = git_info.get("branch", "")
            if current_branch and self.config.branch and current_branch != self.config.branch:
                return (
                    True,
                    f"different branch (current: {current_branch}, desired: {self.config.branch})",
                )

        # If we want PyPI source but currently installed from git
        elif self.config.source == "pypi":
            if pkg_info["from_source"]:
                return True, "installed from source but PyPI installation required"

        return False, "already installed with correct source"

    def install(self, force: bool = False) -> bool:
        """Install the package.

        Args:
            force: If True, reinstall even if already installed

        Returns:
            True if installation successful, False otherwise
        """
        # Check current installation
        pkg_info = self.check_installation()

        if not force:
            needs_reinstall, reason = self.needs_reinstall(pkg_info)

            if not needs_reinstall:
                if pkg_info["version"]:
                    print_success(f"{self.config.name} {pkg_info['version']} already installed")
                else:
                    print_success(f"{self.config.name} already installed")

                # Show current installation info
                if pkg_info["git_info"]:
                    git_info = pkg_info["git_info"]
                    console.print(f"  [dim]Source: GitHub ({git_info.get('remote', 'unknown')})[/dim]")
                    console.print(f"  [dim]Branch: {git_info.get('branch', 'unknown')}[/dim]")
                elif pkg_info["from_source"]:
                    console.print(f"  [dim]Source: GitHub (editable)[/dim]")
                else:
                    console.print(f"  [dim]Source: PyPI[/dim]")

                return True
            else:
                print_warning(f"{self.config.name} needs reinstall: {reason}")
                # Uninstall current version first
                console.print(f"  [dim]Uninstalling current version...[/dim]")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "uninstall", "-y", self.config.name],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    pass  # Ignore uninstall errors

        # Perform installation based on source
        if self.config.source == "github":
            return self._install_from_github()
        elif self.config.source == "pypi":
            return self._install_from_pypi()
        elif self.config.source == "local":
            return self._install_from_local()
        else:
            print_error(f"Unknown source: {self.config.source}")
            return False

    def _install_from_github(self) -> bool:
        """Install package from GitHub repository."""
        if not self.config.repo_url or not self.config.branch:
            print_error("GitHub installation requires repo_url and branch")
            return False

        print_info(f"Installing {self.config.name} from GitHub: {self.config.repo_url} (branch: {self.config.branch})")

        # Use persistent directory for editable installation
        deps_dir = Path.home() / ".ktransformers" / "deps"
        deps_dir.mkdir(parents=True, exist_ok=True)
        clone_path = deps_dir / self.config.name

        try:
            # Clone or update the repository
            if clone_path.exists():
                console.print(f"  [dim]Updating existing repository at {clone_path}...[/dim]")
                console.print()
                # Pull latest changes
                subprocess.run(
                    ["git", "fetch", "origin", self.config.branch],
                    cwd=clone_path,
                    check=True,
                )
                subprocess.run(
                    ["git", "checkout", self.config.branch],
                    cwd=clone_path,
                    check=True,
                )
                subprocess.run(
                    ["git", "pull"],
                    cwd=clone_path,
                    check=True,
                )
            else:
                console.print(f"  [dim]Cloning {self.config.repo_url} to {clone_path}...[/dim]")
                console.print()
                # Clone with progress
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--progress",
                        "--branch",
                        self.config.branch,
                        self.config.repo_url,
                        str(clone_path),
                    ],
                    check=True,
                )

            # Install in editable mode to preserve git info
            console.print()
            console.print(f"  [dim]Installing in editable mode...[/dim]")
            console.print()

            pip_cmd = [sys.executable, "-m", "pip", "install", "-e", str(clone_path), "-v"]
            pip_cmd.extend(self.config.pip_args)

            subprocess.run(
                pip_cmd,
                check=True,
                cwd=clone_path,
            )

            print_success(f"{self.config.name} installed from GitHub (editable)")
            console.print(f"  [dim]Location: {clone_path}[/dim]")
            return True

        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {self.config.name} from GitHub: {e}")
            # For SGLang compatibility, try PyPI fallback
            if self.config.name == "sglang":
                print_warning("Falling back to PyPI installation...")
                console.print()
                try:
                    return self._install_from_pypi()
                except Exception:
                    pass
            return False
        except FileNotFoundError:
            print_error("Git not found. Please install git or use PyPI source in config.")
            raise typer.Exit(1)

    def _install_from_pypi(self) -> bool:
        """Install package from PyPI."""
        if self.config.name == "sglang":
            print_warning(f"Installing {self.config.name} from PyPI (may not be compatible with kt-kernel)")
            console.print(
                f"  [dim]Recommend using GitHub source: kt config set dependencies.sglang.source github[/dim]"
            )
        else:
            print_info(f"Installing {self.config.name} from PyPI")

        console.print()

        pip_cmd = [sys.executable, "-m", "pip", "install", self.config.name, "-v"]
        pip_cmd.extend(self.config.pip_args)

        try:
            subprocess.run(pip_cmd, check=True)
            print_success(f"{self.config.name} installed from PyPI")
            if self.config.name == "sglang":
                print_warning("⚠ PyPI version may not be compatible with kt-kernel")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {self.config.name} from PyPI: {e}")
            return False

    def _install_from_local(self) -> bool:
        """Install package from local source directory."""
        if not self.config.local_path:
            print_error("Local installation requires local_path")
            return False

        local_path = Path(self.config.local_path)
        if not local_path.exists():
            print_error(f"Local path not found: {local_path}")
            return False

        print_info(f"Installing {self.config.name} from local source: {local_path}")

        pip_cmd = [sys.executable, "-m", "pip", "install"]
        if self.config.editable:
            pip_cmd.append("-e")
        pip_cmd.append(str(local_path))
        pip_cmd.extend(self.config.pip_args)

        try:
            subprocess.run(pip_cmd, check=True, cwd=local_path)
            print_success(f"{self.config.name} installed from local source")
            return True
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install {self.config.name} from local source: {e}")
            return False


def create_sglang_installer() -> PackageInstaller:
    """Create a PackageInstaller configured for SGLang.

    Reads configuration from settings to preserve existing behavior.
    """
    settings = get_settings()

    return PackageInstaller(
        PackageConfig(
            name="sglang",
            source=settings.get("dependencies.sglang.source", "github"),
            repo_url=settings.get("dependencies.sglang.repo", "https://github.com/kvcache-ai/sglang"),
            branch=settings.get("dependencies.sglang.branch", "main"),
            editable=True,
        )
    )
