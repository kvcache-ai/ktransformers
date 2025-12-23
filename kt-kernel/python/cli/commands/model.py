"""
Model command for kt-cli.

Manages models: download, list, and storage paths.
"""

import os
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    prompt_choice,
)

app = typer.Typer(
    help="Manage models and storage paths",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback()
def callback(ctx: typer.Context) -> None:
    """
    Model management commands.

    Run without arguments to see available models.
    """
    # If no subcommand is provided, show the model list
    if ctx.invoked_subcommand is None:
        show_model_list()


def show_model_list() -> None:
    """Display available models with their status and paths."""
    from rich.table import Table
    from kt_kernel.cli.utils.model_registry import get_registry
    from kt_kernel.cli.i18n import get_lang

    registry = get_registry()
    settings = get_settings()

    console.print()
    console.print("[bold cyan]KTransformers Supported Models[/bold cyan]\n")

    # Get local models mapping
    local_models = {m.name: p for m, p in registry.find_local_models()}

    # Create table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")

    all_models = registry.list_all()
    for model in all_models:
        if model.name in local_models:
            status = "[green]✓ Local[/green]"
        else:
            status = "[dim]-[/dim]"

        table.add_row(model.name, status)

    console.print(table)
    console.print()

    # Usage instructions
    console.print("[bold]Usage:[/bold]")
    console.print("  • Download a model:  [cyan]kt model download <model-name>[/cyan]")
    console.print("  • List local models: [cyan]kt model list --local[/cyan]")
    console.print("  • Search models:     [cyan]kt model search <query>[/cyan]")
    console.print()

    # Show model storage paths
    model_paths = settings.get_model_paths()
    console.print("[bold]Model Storage Paths:[/bold]")
    for path in model_paths:
        marker = "[green]✓[/green]" if path.exists() else "[dim]✗[/dim]"
        console.print(f"  {marker} {path}")
    console.print()


@app.command(name="download")
def download(
    model: Optional[str] = typer.Argument(
        None,
        help="Model name or HuggingFace repo (e.g., deepseek-v3, Qwen/Qwen3-30B)",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Custom download path",
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available models",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume incomplete downloads",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Download model weights from HuggingFace."""
    import subprocess
    from kt_kernel.cli.i18n import get_lang
    from kt_kernel.cli.utils.console import print_model_table, print_step
    from kt_kernel.cli.utils.model_registry import get_registry

    settings = get_settings()
    registry = get_registry()

    console.print()

    # List mode
    if list_models or model is None:
        print_step(t("download_list_title"))
        console.print()

        models = registry.list_all()
        model_dicts = []
        for m in models:
            lang = get_lang()
            desc = m.description_zh if lang == "zh" and m.description_zh else m.description
            model_dicts.append(
                {
                    "name": m.name,
                    "hf_repo": m.hf_repo,
                    "type": m.type,
                    "gpu_vram_gb": m.gpu_vram_gb,
                    "cpu_ram_gb": m.cpu_ram_gb,
                }
            )

        print_model_table(model_dicts)
        console.print()

        if model is None:
            console.print("[dim]Usage: kt model download <model-name>[/dim]")
            console.print()
            return

    # Search for model
    print_step(t("download_searching", name=model))

    # Check if it's a direct HuggingFace repo path
    if "/" in model:
        hf_repo = model
        model_info = None
        model_name = model.split("/")[-1]
    else:
        matches = registry.search(model)

        if not matches:
            print_error(t("run_model_not_found", name=model))
            console.print()
            console.print("Use 'kt model download --list' to see available models.")
            console.print("Or specify a HuggingFace repo directly: kt model download org/model-name")
            raise typer.Exit(1)

        if len(matches) == 1:
            model_info = matches[0]
        else:
            console.print()
            print_info(t("download_multiple_found"))
            choices = [f"{m.name} ({m.hf_repo})" for m in matches]
            selected = prompt_choice(t("download_select"), choices)
            idx = choices.index(selected)
            model_info = matches[idx]

        hf_repo = model_info.hf_repo
        model_name = model_info.name

    print_success(t("download_found", name=hf_repo))

    # Determine download path
    if path is None:
        download_path = settings.models_dir / model_name.replace(" ", "-")
    else:
        download_path = path

    console.print()
    print_info(t("download_destination", path=str(download_path)))

    # Check if already exists
    if download_path.exists() and (download_path / "config.json").exists():
        print_warning(t("download_already_exists", path=str(download_path)))
        if not yes:
            if not confirm(t("download_overwrite_prompt"), default=False):
                raise typer.Abort()

    # Confirm download
    if not yes:
        console.print()
        if not confirm(t("prompt_continue")):
            raise typer.Abort()

    # Download using huggingface-cli
    console.print()
    print_step(t("download_starting"))

    cmd = [
        "huggingface-cli",
        "download",
        hf_repo,
        "--local-dir",
        str(download_path),
    ]

    if resume:
        cmd.append("--resume-download")

    # Add mirror if configured
    mirror = settings.get("download.mirror", "")
    if mirror:
        cmd.extend(["--endpoint", mirror])

    try:
        process = subprocess.run(cmd, check=True)

        console.print()
        print_success(t("download_complete"))
        console.print()
        console.print(f"  Model saved to: {download_path}")
        console.print()
        console.print(f"  Start with: kt run {model_name}")
        console.print()

    except subprocess.CalledProcessError as e:
        print_error(f"Download failed: {e}")
        raise typer.Exit(1)
    except FileNotFoundError:
        print_error("huggingface-cli not found. Install with: pip install huggingface-hub")
        raise typer.Exit(1)


@app.command(name="list")
def list_models(
    local_only: bool = typer.Option(False, "--local", help="Show only locally downloaded models"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info including paths"),
) -> None:
    """List available models."""
    from rich.table import Table
    from kt_kernel.cli.utils.model_registry import get_registry

    registry = get_registry()
    console.print()

    if local_only:
        # Show only local models
        local_models = registry.find_local_models()

        if not local_models:
            print_warning("No locally downloaded models found")
            console.print()
            console.print("  Download a model with: [cyan]kt model download <model-name>[/cyan]")
            console.print()
            return

        table = Table(title="Locally Downloaded Models", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", no_wrap=True)
        if verbose:
            table.add_column("Local Path", style="dim")

        for model_info, model_path in local_models:
            if verbose:
                table.add_row(model_info.name, str(model_path))
            else:
                table.add_row(model_info.name)

        console.print(table)
    else:
        # Show all registered models
        all_models = registry.list_all()
        local_models_dict = {m.name: p for m, p in registry.find_local_models()}

        table = Table(title="Available Models", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        if verbose:
            table.add_column("Local Path", style="dim")

        for model in all_models:
            if model.name in local_models_dict:
                status = "[green]✓ Local[/green]"
                local_path = str(local_models_dict[model.name])
            else:
                status = "[dim]-[/dim]"
                local_path = "[dim]Not downloaded[/dim]"

            if verbose:
                table.add_row(model.name, status, local_path)
            else:
                table.add_row(model.name, status)

        console.print(table)

    console.print()


@app.command(name="path-list")
def path_list() -> None:
    """List all configured model storage paths."""
    settings = get_settings()
    model_paths = settings.get_model_paths()

    console.print()
    console.print("[bold]Model Storage Paths:[/bold]\n")

    for i, path in enumerate(model_paths, 1):
        marker = "[green]✓[/green]" if path.exists() else "[red]✗[/red]"
        console.print(f"  {marker} [{i}] {path}")

    console.print()


@app.command(name="path-add")
def path_add(
    path: str = typer.Argument(..., help="Path to add"),
) -> None:
    """Add a new model storage path."""
    # Expand user home directory
    path = os.path.expanduser(path)

    # Check if path exists or can be created
    path_obj = Path(path)
    if not path_obj.exists():
        console.print(f"[yellow]Path does not exist: {path}[/yellow]")
        if confirm(f"Create directory {path}?", default=True):
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]✓[/green] Created directory: {path}")
            except (OSError, PermissionError) as e:
                print_error(f"Failed to create directory: {e}")
                raise typer.Exit(1)
        else:
            raise typer.Abort()

    # Add to configuration
    settings = get_settings()
    settings.add_model_path(path)
    print_success(f"Added model path: {path}")


@app.command(name="path-remove")
def path_remove(
    path: str = typer.Argument(..., help="Path to remove"),
) -> None:
    """Remove a model storage path from configuration."""
    # Expand user home directory
    path = os.path.expanduser(path)

    settings = get_settings()
    if settings.remove_model_path(path):
        print_success(f"Removed model path: {path}")
    else:
        print_error(f"Path not found in configuration or cannot remove last path: {path}")
        raise typer.Exit(1)


@app.command(name="search")
def search(
    query: str = typer.Argument(..., help="Search query (model name or keyword)"),
) -> None:
    """Search for models in the registry."""
    from rich.table import Table
    from kt_kernel.cli.utils.model_registry import get_registry

    registry = get_registry()
    matches = registry.search(query)

    console.print()

    if not matches:
        print_warning(f"No models found matching '{query}'")
        console.print()
        return

    table = Table(title=f"Search Results for '{query}'", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("HuggingFace Repo", style="dim")
    table.add_column("Aliases", style="yellow")

    for model in matches:
        aliases = ", ".join(model.aliases[:3])
        if len(model.aliases) > 3:
            aliases += f" +{len(model.aliases) - 3} more"
        table.add_row(model.name, model.hf_repo, aliases)

    console.print(table)
    console.print()
