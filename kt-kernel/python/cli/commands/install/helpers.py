"""
Helper functions for installation commands.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from rich.live import Live
from rich.table import Table
from rich.text import Text

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    console,
    print_step,
    print_success,
    print_warning,
)
from kt_kernel.cli.utils.environment import (
    detect_cuda_version,
    get_installed_package_version,
)

from .package_installer import create_sglang_installer
from .strategies import InstallMode


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
    from kt_kernel.cli.utils.console import print_error

    if result["success"]:
        print_success(t("install_verify_success", version=result["version"], variant=result["cpu_variant"]))
    else:
        print_error(t("install_verify_failed", error=result["error"]))


def _install_other_deps(mode: InstallMode) -> None:
    """Install other dependencies based on mode."""
    packages = REQUIREMENTS.get(mode.value, [])
    if mode == InstallMode.FULL:
        packages = list(set(REQUIREMENTS["inference"] + REQUIREMENTS["sft"]))

    # Filter out torch, kt-kernel/ktransformers (already installed), and sglang (handled separately)
    other_packages = [
        p
        for p in packages
        if not p.startswith("torch")
        and not p.startswith("kt-kernel")
        and not p.startswith("ktransformers")
        and not p.startswith("sglang")
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

    # Install sglang separately based on configuration
    if mode in (InstallMode.INFERENCE, InstallMode.FULL):
        console.print()
        print_step("Installing SGLang from source...")
        sglang_installer = create_sglang_installer()
        sglang_installer.install()


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

        deps.append(
            {
                "name": name,
                "installed": installed or "-",
                "required": f">={required}",
                "status": status,
            }
        )

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


def _run_pip_realtime(cmd: list[str], name: str) -> bool:
    """Run pip command with real-time output display."""
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
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue

                # Update status based on pip output
                status_text = Text()
                status_text.append(f"📦 {name}: ", style="bold")

                if "Downloading" in line:
                    # Extract size if present
                    size_match = re.search(r"(\d+\.?\d*)\s*(kB|MB|GB)", line)
                    if size_match:
                        status_text.append(f"Downloading ({size_match.group()})", style="yellow")
                    else:
                        # Extract filename
                        file_match = re.search(r"Downloading\s+(\S+)", line)
                        if file_match:
                            filename = file_match.group(1).split("/")[-1]
                            if len(filename) > 40:
                                filename = filename[:37] + "..."
                            status_text.append(f"Downloading {filename}", style="yellow")
                        else:
                            status_text.append("Downloading...", style="yellow")

                elif "Collecting" in line:
                    pkg_match = re.search(r"Collecting\s+(\S+)", line)
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


def _install_packages_pip(mode: InstallMode, skip_torch: bool) -> None:
    """Install packages using pip from PyPI with real-time progress."""
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

        # Filter out torch and sglang packages (already installed or handled separately)
        other_packages = [p for p in packages if not p.startswith("torch") and not p.startswith("sglang")]

        for pkg in other_packages:
            pkg_name = pkg.split(">=")[0].split("==")[0]
            _run_pip_realtime(pip_cmd + [pkg], pkg_name)

    # Install sglang separately if in inference or full mode
    if mode in (InstallMode.INFERENCE, InstallMode.FULL):
        console.print()
        print_step("Installing SGLang from source...")
        sglang_installer = create_sglang_installer()
        sglang_installer.install()
