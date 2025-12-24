"""
SFT command for kt-cli.

Fine-tuning with LlamaFactory integration.
"""

import typer

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import console

app = typer.Typer(help="Fine-tuning with LlamaFactory (coming soon)")


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """Fine-tuning commands (coming soon)."""
    if ctx.invoked_subcommand is None:
        console.print()
        console.print(f"[yellow]{t('feature_coming_soon')}[/yellow]")
        console.print()
        console.print("[dim]kt sft train   - Train a model[/dim]")
        console.print("[dim]kt sft chat    - Chat with a trained model[/dim]")
        console.print("[dim]kt sft export  - Export a trained model[/dim]")
        console.print()


@app.command(name="train")
def train() -> None:
    """Train a model using LlamaFactory (coming soon)."""
    console.print()
    console.print(f"[yellow]{t('feature_coming_soon')}[/yellow]")
    console.print()
    raise typer.Exit(0)


@app.command(name="chat")
def chat() -> None:
    """Chat with a trained model using LlamaFactory (coming soon)."""
    console.print()
    console.print(f"[yellow]{t('feature_coming_soon')}[/yellow]")
    console.print()
    raise typer.Exit(0)


@app.command(name="export")
def export() -> None:
    """Export a trained model using LlamaFactory (coming soon)."""
    console.print()
    console.print(f"[yellow]{t('feature_coming_soon')}[/yellow]")
    console.print()
    raise typer.Exit(0)
