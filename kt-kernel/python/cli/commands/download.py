"""
Download command for kt-cli.

Downloads model weights from HuggingFace Hub.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import get_lang, t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    print_error,
    print_info,
    print_model_table,
    print_step,
    print_success,
    print_warning,
    prompt_choice,
)
from kt_kernel.cli.utils.model_registry import get_registry


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
    """Download model weights."""
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
            model_dicts.append({
                "name": m.name,
                "hf_repo": m.hf_repo,
                "type": m.type,
                "gpu_vram_gb": m.gpu_vram_gb,
                "cpu_ram_gb": m.cpu_ram_gb,
            })

        print_model_table(model_dicts)
        console.print()

        if model is None:
            console.print("[dim]Usage: kt download <model-name>[/dim]")
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
            console.print("Use 'kt download --list' to see available models.")
            console.print("Or specify a HuggingFace repo directly: kt download org/model-name")
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
        "huggingface-cli", "download",
        hf_repo,
        "--local-dir", str(download_path),
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
