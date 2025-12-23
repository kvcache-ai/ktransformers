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

app = typer.Typer(help="Manage models and storage paths")


@app.command(name="download")
def download(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Download path (default: configured models directory)",
    ),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume interrupted download"),
    mirror: Optional[str] = typer.Option(None, "--mirror", help="HuggingFace mirror URL"),
) -> None:
    """Download a model from HuggingFace."""
    from kt_kernel.cli.utils.model_registry import get_registry

    settings = get_settings()
    registry = get_registry()

    console.print()
    print_step = lambda msg: console.print(f"[bold cyan]→[/bold cyan] {msg}")

    # Search for model in registry
    print_step(t("download_searching", name=model))
    matches = registry.search(model)

    if not matches:
        print_error(f"Model '{model}' not found in registry")
        raise typer.Exit(1)

    # Handle multiple matches
    if len(matches) > 1:
        console.print()
        print_info(t("download_multiple_found"))
        console.print()
        for i, m in enumerate(matches[:5], 1):
            console.print(f"  [{i}] {m.name} - {m.hf_repo}")
        console.print()

        choice = prompt_choice(
            t("download_select"),
            [str(i) for i in range(1, min(len(matches), 5) + 1)],
            default="1",
        )
        model_info = matches[int(choice) - 1]
    else:
        model_info = matches[0]
        console.print()
        print_info(t("download_found", name=model_info.name))

    # Determine download path
    if path is None:
        download_path = settings.models_dir / model_info.name.replace(" ", "-")
    else:
        download_path = path

    console.print()
    print_info(t("download_destination", path=str(download_path)))

    # Check if already exists
    if download_path.exists():
        console.print()
        print_warning(t("download_already_exists", path=str(download_path)))
        if not confirm(t("download_overwrite_prompt"), default=False):
            raise typer.Abort()

    # Download using huggingface-hub
    console.print()
    print_step(t("download_starting"))

    try:
        from huggingface_hub import snapshot_download

        # Apply mirror if specified
        if mirror:
            os.environ["HF_ENDPOINT"] = mirror
        elif settings.get("download.mirror"):
            os.environ["HF_ENDPOINT"] = settings.get("download.mirror")

        snapshot_download(
            repo_id=model_info.hf_repo,
            local_dir=str(download_path),
            resume_download=resume,
        )

        console.print()
        print_success(t("download_complete"))
        console.print()
        print_info(f"Model downloaded to: {download_path}")
        console.print()

    except Exception as e:
        console.print()
        print_error(f"Download failed: {e}")
        raise typer.Exit(1)


@app.command(name="list")
def list_models(
    local_only: bool = typer.Option(False, "--local", help="Show only locally downloaded models"),
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

        table = Table(title="Locally Downloaded Models", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Path", style="dim")

        for model_info, model_path in local_models:
            table.add_row(model_info.name, str(model_path))

        console.print(table)
    else:
        # Show all registered models
        all_models = registry.list_all()

        table = Table(title="Available Models", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("HuggingFace Repo", style="dim")
        table.add_column("Status", style="green")

        local_models = {m.name: p for m, p in registry.find_local_models()}

        for model in all_models:
            status = "✓ Local" if model.name in local_models else "Cloud"
            table.add_row(model.name, model.hf_repo, status)

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
