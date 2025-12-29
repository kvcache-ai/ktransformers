"""
Console utilities for kt-cli.

Provides Rich-based console output helpers for consistent formatting.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

from kt_kernel.cli.i18n import t

# Custom theme for kt-cli
KT_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "highlight": "bold magenta",
        "muted": "dim",
    }
)

# Global console instance
console = Console(theme=KT_THEME)


def print_info(message: str, **kwargs) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ[/info] {message}", **kwargs)


def print_success(message: str, **kwargs) -> None:
    """Print a success message."""
    console.print(f"[success]✓[/success] {message}", **kwargs)


def print_warning(message: str, **kwargs) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠[/warning] {message}", **kwargs)


def print_error(message: str, **kwargs) -> None:
    """Print an error message."""
    console.print(f"[error]✗[/error] {message}", **kwargs)


def print_step(message: str, **kwargs) -> None:
    """Print a step indicator."""
    console.print(f"[highlight]→[/highlight] {message}", **kwargs)


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print a header panel."""
    content = f"[bold]{title}[/bold]"
    if subtitle:
        content += f"\n[muted]{subtitle}[/muted]"
    console.print(Panel(content, expand=False))


def print_version_table(versions: dict[str, Optional[str]]) -> None:
    """Print a version information table."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Component", style="bold")
    table.add_column("Version")

    for name, version in versions.items():
        if version:
            table.add_row(name, f"[success]{version}[/success]")
        else:
            table.add_row(name, f"[muted]{t('version_not_installed')}[/muted]")

    console.print(table)


def print_dependency_table(deps: list[dict]) -> None:
    """Print a dependency status table."""
    table = Table(title=t("install_checking_deps"))
    table.add_column(t("version_info"), style="bold")
    table.add_column("Current")
    table.add_column("Required")
    table.add_column("Status")

    for dep in deps:
        status = dep.get("status", "ok")
        if status == "ok":
            status_str = f"[success]{t('install_dep_ok')}[/success]"
        elif status == "outdated":
            status_str = f"[warning]{t('install_dep_outdated')}[/warning]"
        else:
            status_str = f"[error]{t('install_dep_missing')}[/error]"

        table.add_row(
            dep["name"],
            dep.get("installed", "-"),
            dep.get("required", "-"),
            status_str,
        )

    console.print(table)


def confirm(message: str, default: bool = True) -> bool:
    """Ask for confirmation."""
    return Confirm.ask(message, default=default, console=console)


def prompt_choice(message: str, choices: list[str], default: Optional[str] = None) -> str:
    """Prompt for a choice from a list."""
    # Display numbered choices
    console.print(f"\n[bold]{message}[/bold]")
    for i, choice in enumerate(choices, 1):
        console.print(f"  [highlight][{i}][/highlight] {choice}")

    while True:
        response = Prompt.ask(
            "\n" + t("prompt_select"),
            console=console,
            default=str(choices.index(default) + 1) if default else None,
        )
        try:
            idx = int(response) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            # Check if response matches a choice directly
            if response in choices:
                return response

        print_error(f"Please enter a number between 1 and {len(choices)}")


def prompt_text(message: str, default: Optional[str] = None) -> str:
    """Prompt for text input."""
    return Prompt.ask(message, console=console, default=default)


def create_progress() -> Progress:
    """Create a progress bar for general tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_download_progress() -> Progress:
    """Create a progress bar for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def print_model_table(models: list[dict]) -> None:
    """Print a table of models."""
    table = Table(title=t("download_list_title"))
    table.add_column("Name", style="bold")
    table.add_column("Repository")
    table.add_column("Type")
    table.add_column("Requirements")

    for model in models:
        reqs = []
        if model.get("gpu_vram_gb"):
            reqs.append(f"GPU: {model['gpu_vram_gb']}GB")
        if model.get("cpu_ram_gb"):
            reqs.append(f"RAM: {model['cpu_ram_gb']}GB")

        table.add_row(
            model.get("name", ""),
            model.get("hf_repo", ""),
            model.get("type", ""),
            ", ".join(reqs) if reqs else "-",
        )

    console.print(table)


def print_hardware_info(gpu_info: str, cpu_info: str, ram_info: str) -> None:
    """Print hardware information."""
    table = Table(show_header=False, box=None)
    table.add_column("Icon", width=3)
    table.add_column("Info")

    table.add_row("🖥️", gpu_info)
    table.add_row("💻", cpu_info)
    table.add_row("🧠", ram_info)

    console.print(Panel(table, title="Hardware", expand=False))


def print_server_info(mode: str, host: str, port: int, gpu_experts: int, cpu_threads: int) -> None:
    """Print server startup information."""
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row(t("run_server_mode").split(":")[0], mode)
    table.add_row("Host", host)
    table.add_row("Port", str(port))
    table.add_row(t("run_gpu_experts").split(":")[0], f"{gpu_experts}/layer")
    table.add_row(t("run_cpu_threads").split(":")[0], str(cpu_threads))

    console.print(Panel(table, title=t("run_server_started"), expand=False, border_style="green"))


def print_api_info(host: str, port: int) -> None:
    """Print API endpoint information."""
    api_url = f"http://{host}:{port}"
    docs_url = f"http://{host}:{port}/docs"

    console.print()
    console.print(f"  {t('run_api_url', host=host, port=port)}")
    console.print(f"  {t('run_docs_url', host=host, port=port)}")
    console.print()
    console.print(f"  [muted]Test command:[/muted]")
    console.print(
        f"  [dim]curl {api_url}/v1/chat/completions -H 'Content-Type: application/json' "
        f'-d \'{{"model": "default", "messages": [{{"role": "user", "content": "Hello"}}]}}\'[/dim]'
    )
    console.print()
    console.print(f"  [muted]{t('run_stop_hint')}[/muted]")
