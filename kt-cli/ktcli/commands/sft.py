"""
SFT command for kt-cli.

Fine-tuning with LlamaFactory integration.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ktcli.config.settings import get_settings
from ktcli.i18n import t
from ktcli.utils.console import (
    console,
    print_error,
    print_info,
    print_step,
    print_success,
)

app = typer.Typer(help="Fine-tuning with LlamaFactory")


@app.command(name="train")
def train(
    config: Path = typer.Argument(
        ...,
        help="Path to training config YAML file",
        exists=True,
    ),
    use_kt: bool = typer.Option(
        True,
        "--use-kt/--no-kt",
        help="Enable KTransformers optimization",
    ),
) -> None:
    """Train a model using LlamaFactory."""
    console.print()
    print_step(t("sft_mode_train"))
    print_info(t("sft_config_path", path=str(config)))
    console.print()

    _run_llamafactory("train", config, use_kt)


@app.command(name="chat")
def chat(
    config: Path = typer.Argument(
        ...,
        help="Path to chat config YAML file",
        exists=True,
    ),
    use_kt: bool = typer.Option(
        True,
        "--use-kt/--no-kt",
        help="Enable KTransformers optimization",
    ),
) -> None:
    """Chat with a trained model using LlamaFactory."""
    console.print()
    print_step(t("sft_mode_chat"))
    print_info(t("sft_config_path", path=str(config)))
    console.print()

    _run_llamafactory("chat", config, use_kt)


@app.command(name="export")
def export(
    config: Path = typer.Argument(
        ...,
        help="Path to export config YAML file",
        exists=True,
    ),
    use_kt: bool = typer.Option(
        True,
        "--use-kt/--no-kt",
        help="Enable KTransformers optimization",
    ),
) -> None:
    """Export a trained model using LlamaFactory."""
    console.print()
    print_step(t("sft_mode_export"))
    print_info(t("sft_config_path", path=str(config)))
    console.print()

    _run_llamafactory("export", config, use_kt)


@app.command(name="eval")
def evaluate(
    config: Path = typer.Argument(
        ...,
        help="Path to evaluation config YAML file",
        exists=True,
    ),
    use_kt: bool = typer.Option(
        True,
        "--use-kt/--no-kt",
        help="Enable KTransformers optimization",
    ),
) -> None:
    """Evaluate a model using LlamaFactory."""
    console.print()
    print_step("Evaluation mode")
    print_info(t("sft_config_path", path=str(config)))
    console.print()

    _run_llamafactory("eval", config, use_kt)


def _run_llamafactory(command: str, config: Path, use_kt: bool) -> None:
    """Run a LlamaFactory command."""
    settings = get_settings()

    # Build environment
    env = os.environ.copy()
    env.update(settings.get_env_vars())

    if use_kt:
        env["USE_KT"] = "1"
        print_info("KTransformers optimization: enabled")
    else:
        print_info("KTransformers optimization: disabled")

    console.print()
    print_step(t("sft_starting", mode=command))

    # Build command
    cmd = ["llamafactory-cli", command, str(config)]

    # Add extra args from settings
    extra_args = settings.get("advanced.llamafactory_args", [])
    if extra_args:
        cmd.extend(extra_args)

    console.print()
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    console.print()

    try:
        process = subprocess.run(cmd, env=env)

        if process.returncode == 0:
            console.print()
            print_success(t("sft_complete", mode=command))
        else:
            print_error(f"LlamaFactory {command} failed with exit code {process.returncode}")
            raise typer.Exit(process.returncode)

    except FileNotFoundError:
        print_error("llamafactory-cli not found. Install with: kt install sft")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print()
        print_info("Interrupted by user.")
        raise typer.Exit(130)
