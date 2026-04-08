"""
Debug utility to inspect saved run configurations.

Usage: python -m kt_kernel.cli.utils.debug_configs
"""

from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def main():
    """Show all saved configurations."""
    config_file = Path.home() / ".ktransformers" / "run_configs.yaml"

    console.print()
    console.print(f"[bold]Configuration file:[/bold] {config_file}")
    console.print()

    if not config_file.exists():
        console.print("[red]✗ Configuration file does not exist![/red]")
        console.print()
        console.print("No configurations have been saved yet.")
        return

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        console.print(f"[red]✗ Failed to load configuration file: {e}[/red]")
        return

    console.print(f"[green]✓[/green] Configuration file loaded")
    console.print()

    configs = data.get("configs", {})

    if not configs:
        console.print("[yellow]No saved configurations found.[/yellow]")
        return

    console.print(f"[bold]Found configurations for {len(configs)} model(s):[/bold]")
    console.print()

    for model_id, model_configs in configs.items():
        console.print(f"[cyan]Model ID:[/cyan] {model_id}")
        console.print(f"[dim]  {len(model_configs)} configuration(s)[/dim]")
        console.print()

        if not model_configs:
            continue

        # Display configs in a table
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Method", style="yellow")
        table.add_column("TP", justify="right", style="green")
        table.add_column("GPU Experts", justify="right", style="magenta")
        table.add_column("Created", style="dim")

        for i, cfg in enumerate(model_configs, 1):
            method = cfg.get("inference_method", "?")
            kt_method = cfg.get("kt_method", "?")
            method_display = f"{method.upper()}"
            if method == "raw":
                method_display += f" ({cfg.get('raw_method', '?')})"
            elif method == "amx":
                method_display += f" ({kt_method})"

            table.add_row(
                str(i),
                cfg.get("config_name", f"Config {i}"),
                method_display,
                str(cfg.get("tp_size", "?")),
                str(cfg.get("gpu_experts", "?")),
                cfg.get("created_at", "Unknown")[:19] if cfg.get("created_at") else "Unknown",
            )

        console.print(table)
        console.print()

    # Also check user_models.yaml to show model names
    console.print("[bold]Checking model registry...[/bold]")
    console.print()

    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    try:
        registry = UserModelRegistry()
        all_models = registry.list_models()

        console.print(f"[green]✓[/green] Found {len(all_models)} registered model(s)")
        console.print()

        # Map model IDs to names
        id_to_name = {m.id: m.name for m in all_models}

        console.print("[bold]Model ID → Name mapping:[/bold]")
        console.print()

        for model_id in configs.keys():
            model_name = id_to_name.get(model_id, "[red]Unknown (model not found in registry)[/red]")
            console.print(f"  {model_id[:8]}... → {model_name}")

        console.print()

    except Exception as e:
        console.print(f"[yellow]⚠ Could not load model registry: {e}[/yellow]")
        console.print()


if __name__ == "__main__":
    main()
