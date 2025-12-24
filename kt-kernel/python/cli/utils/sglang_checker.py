"""
SGLang installation checker and installation instructions provider.

This module provides utilities to:
- Check if SGLang is installed and get its metadata
- Provide installation instructions when SGLang is not found
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import console


def check_sglang_installation() -> dict:
    """Check if SGLang is installed and get its metadata.

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
        # Try to import sglang
        import sglang

        version = getattr(sglang, "__version__", None)

        # Use pip show to get detailed package information
        location = None
        editable = False
        git_info = None
        from_source = False

        try:
            # Get pip show output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "sglang"],
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
            if hasattr(sglang, "__file__") and sglang.__file__:
                location = str(Path(sglang.__file__).parent.parent)

        # Check if it's installed from source (has .git directory)
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

                    # Extract org/repo from URL
                    remote_short = None
                    if remote_url:
                        # Handle both https and git@ URLs
                        if "github.com" in remote_url:
                            parts = remote_url.rstrip("/").replace(".git", "").split("github.com")[-1]
                            remote_short = parts.lstrip("/").lstrip(":")

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
                            "remote": remote_short or remote_url,
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


def get_sglang_install_instructions(lang: Optional[str] = None) -> str:
    """Get SGLang installation instructions.

    Args:
        lang: Language code ('en' or 'zh'). If None, uses current language setting.

    Returns:
        Formatted installation instructions string.
    """
    from kt_kernel.cli.i18n import get_lang

    if lang is None:
        lang = get_lang()

    if lang == "zh":
        return """
[bold yellow]SGLang \u672a\u5b89\u88c5[/bold yellow]

\u8bf7\u6309\u7167\u4ee5\u4e0b\u6b65\u9aa4\u5b89\u88c5 SGLang:

[bold]1. \u514b\u9686\u4ed3\u5e93:[/bold]
   git clone https://github.com/kvcache-ai/sglang.git
   cd sglang

[bold]2. \u5b89\u88c5 (\u4e8c\u9009\u4e00):[/bold]

   [cyan]\u65b9\u5f0f A - pip \u5b89\u88c5 (\u63a8\u8350):[/cyan]
   pip install -e "python[all]"

   [cyan]\u65b9\u5f0f B - uv \u5b89\u88c5 (\u66f4\u5feb):[/cyan]
   pip install uv
   uv pip install -e "python[all]"

[dim]\u6ce8\u610f: \u8bf7\u786e\u4fdd\u5728\u6b63\u786e\u7684 Python \u73af\u5883\u4e2d\u6267\u884c\u4ee5\u4e0a\u547d\u4ee4[/dim]
"""
    else:
        return """
[bold yellow]SGLang is not installed[/bold yellow]

Please follow these steps to install SGLang:

[bold]1. Clone the repository:[/bold]
   git clone https://github.com/kvcache-ai/sglang.git
   cd sglang

[bold]2. Install (choose one):[/bold]

   [cyan]Option A - pip install (recommended):[/cyan]
   pip install -e "python[all]"

   [cyan]Option B - uv install (faster):[/cyan]
   pip install uv
   uv pip install -e "python[all]"

[dim]Note: Make sure to run these commands in the correct Python environment[/dim]
"""


def print_sglang_install_instructions() -> None:
    """Print SGLang installation instructions to console."""
    instructions = get_sglang_install_instructions()
    console.print(instructions)


def check_sglang_and_warn() -> bool:
    """Check if SGLang is installed, print warning if not.

    Returns:
        True if SGLang is installed, False otherwise.
    """
    info = check_sglang_installation()

    if not info["installed"]:
        print_sglang_install_instructions()
        return False

    # Check if installed from PyPI (not recommended)
    if info["installed"] and not info["from_source"]:
        from kt_kernel.cli.utils.console import print_warning

        print_warning(t("sglang_pypi_warning"))
        console.print()
        console.print("[dim]" + t("sglang_recommend_source") + "[/dim]")
        console.print()

    return True


def _get_sglang_kt_kernel_cache_path() -> Path:
    """Get the path to the sglang kt-kernel support cache file."""
    cache_dir = Path.home() / ".ktransformers" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "sglang_kt_kernel_supported"


def _is_sglang_kt_kernel_cache_valid() -> bool:
    """Check if the sglang kt-kernel support cache is valid.

    The cache is considered valid if:
    1. The cache file exists
    2. The cache file contains 'true' (indicating previous check passed)

    Returns:
        True if cache is valid and indicates support, False otherwise.
    """
    cache_path = _get_sglang_kt_kernel_cache_path()
    if cache_path.exists():
        try:
            content = cache_path.read_text().strip().lower()
            return content == "true"
        except (OSError, IOError):
            pass
    return False


def _save_sglang_kt_kernel_cache(supported: bool) -> None:
    """Save the sglang kt-kernel support check result to cache."""
    cache_path = _get_sglang_kt_kernel_cache_path()
    try:
        cache_path.write_text("true" if supported else "false")
    except (OSError, IOError):
        pass  # Ignore cache write errors


def clear_sglang_kt_kernel_cache() -> None:
    """Clear the sglang kt-kernel support cache, forcing a re-check on next run."""
    cache_path = _get_sglang_kt_kernel_cache_path()
    try:
        if cache_path.exists():
            cache_path.unlink()
    except (OSError, IOError):
        pass


def check_sglang_kt_kernel_support(use_cache: bool = True, silent: bool = False) -> dict:
    """Check if SGLang supports kt-kernel parameters (--kt-gpu-prefill-token-threshold).

    This function runs `python -m sglang.launch_server --help` and checks if the
    output contains the `--kt-gpu-prefill-token-threshold` parameter. This parameter
    is only available in the kvcache-ai/sglang fork, not in the official sglang.

    The result is cached after the first successful check to avoid repeated checks.

    Args:
        use_cache: If True, use cached result if available. Default is True.
        silent: If True, don't print checking message. Default is False.

    Returns:
        dict with keys:
        - supported: bool - True if kt-kernel parameters are supported
        - help_output: str or None - The help output from sglang.launch_server
        - error: str or None - Error message if check failed
        - from_cache: bool - True if result was from cache
    """
    from kt_kernel.cli.utils.console import print_step

    # Check cache first
    if use_cache and _is_sglang_kt_kernel_cache_valid():
        return {
            "supported": True,
            "help_output": None,
            "error": None,
            "from_cache": True,
        }

    # Print checking message
    if not silent:
        print_step(t("sglang_checking_kt_kernel_support"))

    try:
        result = subprocess.run(
            [sys.executable, "-m", "sglang.launch_server", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        help_output = result.stdout + result.stderr

        # Check if --kt-gpu-prefill-token-threshold is in the help output
        supported = "--kt-gpu-prefill-token-threshold" in help_output

        # Save to cache if supported
        if supported:
            _save_sglang_kt_kernel_cache(True)

        return {
            "supported": supported,
            "help_output": help_output,
            "error": None,
            "from_cache": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "supported": False,
            "help_output": None,
            "error": "Timeout while checking sglang.launch_server --help",
            "from_cache": False,
        }
    except FileNotFoundError:
        return {
            "supported": False,
            "help_output": None,
            "error": "Python interpreter not found",
            "from_cache": False,
        }
    except Exception as e:
        return {
            "supported": False,
            "help_output": None,
            "error": str(e),
            "from_cache": False,
        }


def print_sglang_kt_kernel_instructions() -> None:
    """Print instructions for installing the kvcache-ai fork of SGLang with kt-kernel support."""
    from kt_kernel.cli.i18n import get_lang

    lang = get_lang()

    if lang == "zh":
        instructions = """
[bold red]SGLang 不支持 kt-kernel[/bold red]

您当前安装的 SGLang 不包含 kt-kernel 支持。
kt-kernel 需要使用 kvcache-ai 维护的 SGLang 分支。

[bold]请按以下步骤重新安装 SGLang:[/bold]

[cyan]1. 卸载当前的 SGLang:[/cyan]
   pip uninstall sglang -y

[cyan]2. 克隆 kvcache-ai 的 SGLang 仓库:[/cyan]
   git clone https://github.com/kvcache-ai/sglang.git
   cd sglang

[cyan]3. 安装 SGLang:[/cyan]
   pip install -e "python[all]"

[dim]注意: 请确保在正确的 Python 环境中执行以上命令[/dim]
"""
    else:
        instructions = """
[bold red]SGLang does not support kt-kernel[/bold red]

Your current SGLang installation does not include kt-kernel support.
kt-kernel requires the kvcache-ai maintained fork of SGLang.

[bold]Please reinstall SGLang with the following steps:[/bold]

[cyan]1. Uninstall current SGLang:[/cyan]
   pip uninstall sglang -y

[cyan]2. Clone the kvcache-ai SGLang repository:[/cyan]
   git clone https://github.com/kvcache-ai/sglang.git
   cd sglang

[cyan]3. Install SGLang:[/cyan]
   pip install -e "python[all]"

[dim]Note: Make sure to run these commands in the correct Python environment[/dim]
"""
    console.print(instructions)
