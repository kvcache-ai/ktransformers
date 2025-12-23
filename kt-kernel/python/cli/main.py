"""
Main entry point for kt-cli.

KTransformers CLI - A unified command-line interface for KTransformers.
"""

import sys

import typer

from kt_kernel.cli import __version__
from kt_kernel.cli.commands import bench, chat, config, doctor, install, model, quant, run, sft, version
from kt_kernel.cli.i18n import t, set_lang

# Create main app
app = typer.Typer(
    name="kt",
    help="KTransformers CLI - A unified command-line interface for KTransformers.",
    no_args_is_help=True,
    add_completion=False,  # Use static completion scripts instead of dynamic completion
    rich_markup_mode="rich",
)

# Register commands
app.command(name="version", help="Show version information")(version.version)
app.command(name="install", help="Install KTransformers and dependencies")(install.install)
app.command(name="update", help="Update KTransformers to the latest version")(install.update)
app.command(name="run", help="Start model inference server")(run.run)
app.command(name="chat", help="Interactive chat with running model")(chat.chat)
app.command(name="quant", help="Quantize model weights")(quant.quant)
app.command(name="bench", help="Run full benchmark")(bench.bench)
app.command(name="microbench", help="Run micro-benchmark")(bench.microbench)
app.command(name="doctor", help="Diagnose environment issues")(doctor.doctor)

# Register sub-apps
app.add_typer(model.app, name="model", help="Manage models and storage paths")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(sft.app, name="sft", help="Fine-tuning with LlamaFactory")


def check_first_run() -> None:
    """Check if this is the first run and prompt for language setup."""
    import os

    # Skip if not running in interactive terminal
    if not sys.stdin.isatty():
        return

    from kt_kernel.cli.config.settings import DEFAULT_CONFIG_FILE

    # Only check if config file exists - don't create it yet
    if not DEFAULT_CONFIG_FILE.exists():
        # First run - show welcome and language selection
        from kt_kernel.cli.config.settings import get_settings

        settings = get_settings()
        _show_first_run_setup(settings)
    else:
        # Config exists - check if initialized
        from kt_kernel.cli.config.settings import get_settings

        settings = get_settings()
        if not settings.get("general._initialized"):
            _show_first_run_setup(settings)


def _show_first_run_setup(settings) -> None:
    """Show first-run setup wizard."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.spinner import Spinner
    from rich.live import Live

    from kt_kernel.cli.utils.environment import scan_storage_locations, format_size_gb, scan_models_in_location

    console = Console()

    # Welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Welcome to KTransformers CLI! / 欢迎使用 KTransformers CLI![/bold cyan]\n\n"
            "Let's set up your preferences.\n"
            "让我们设置您的偏好。",
            title="kt-cli",
            border_style="cyan",
        )
    )
    console.print()

    # Language selection
    console.print("[bold]Select your preferred language / 选择您的首选语言:[/bold]")
    console.print()
    console.print("  [cyan][1][/cyan] English")
    console.print("  [cyan][2][/cyan] 中文 (Chinese)")
    console.print()

    while True:
        choice = Prompt.ask("Enter choice / 输入选择", choices=["1", "2"], default="1")

        if choice == "1":
            lang = "en"
            break
        elif choice == "2":
            lang = "zh"
            break

    # Save language setting
    settings.set("general.language", lang)
    set_lang(lang)

    # Confirmation message
    console.print()
    if lang == "zh":
        console.print("[green]✓[/green] 语言已设置为中文")
    else:
        console.print("[green]✓[/green] Language set to English")

    # Model storage path selection
    console.print()
    console.print(f"[bold]{t('setup_model_path_title')}[/bold]")
    console.print()
    console.print(f"[dim]{t('setup_model_path_desc')}[/dim]")
    console.print()

    # Scan for storage locations
    console.print(f"[dim]{t('setup_scanning_disks')}[/dim]")
    locations = scan_storage_locations(min_size_gb=50.0)
    console.print()

    if locations:
        # Scan for models in each location
        console.print(f"[dim]{t('setup_scanning_models')}[/dim]")
        location_models: dict[str, list] = {}
        for loc in locations[:5]:
            models = scan_models_in_location(loc, max_depth=2)
            if models:
                location_models[loc.path] = models
        console.print()

        # Show options
        for i, loc in enumerate(locations[:5], 1):  # Show top 5 options
            available = format_size_gb(loc.available_gb)
            total = format_size_gb(loc.total_gb)

            # Build the option string
            if i == 1:
                option_str = t("setup_disk_option_recommended", path=loc.path, available=available, total=total)
            else:
                option_str = t("setup_disk_option", path=loc.path, available=available, total=total)

            # Add model count if any
            if loc.path in location_models:
                model_count = len(location_models[loc.path])
                option_str += f" [green]✓ {t('setup_location_has_models', count=model_count)}[/green]"

            console.print(f"  [cyan][{i}][/cyan] {option_str}")

            # Show first few models found in this location
            if loc.path in location_models:
                for model in location_models[loc.path][:3]:  # Show up to 3 models
                    size_str = format_size_gb(model.size_gb)
                    console.print(f"      [dim]• {model.name} ({size_str})[/dim]")
                if len(location_models[loc.path]) > 3:
                    remaining = len(location_models[loc.path]) - 3
                    console.print(f"      [dim]  ... +{remaining} more[/dim]")

        # Custom path option
        custom_idx = min(len(locations), 5) + 1
        console.print(f"  [cyan][{custom_idx}][/cyan] {t('setup_custom_path')}")
        console.print()

        valid_choices = [str(i) for i in range(1, custom_idx + 1)]
        path_choice = Prompt.ask(t("prompt_select"), choices=valid_choices, default="1")

        if path_choice == str(custom_idx):
            # Custom path
            selected_path = _prompt_custom_path(console, settings)
        else:
            selected_path = locations[int(path_choice) - 1].path
    else:
        # No large storage found, ask for custom path
        console.print(f"[yellow]{t('setup_no_large_disk')}[/yellow]")
        console.print()
        selected_path = _prompt_custom_path(console, settings)

    # Ensure the path exists
    import os
    from pathlib import Path

    if not os.path.exists(selected_path):
        if Confirm.ask(t("setup_path_not_exist"), default=True):
            try:
                Path(selected_path).mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                console.print(f"[red]{t('error')}: {e}[/red]")
                # Fall back to default
                selected_path = str(Path.home() / ".ktransformers" / "models")
                Path(selected_path).mkdir(parents=True, exist_ok=True)

    # Check available space and warn if low
    from kt_kernel.cli.utils.environment import detect_disk_space_gb

    available_gb, _ = detect_disk_space_gb(
        selected_path if os.path.exists(selected_path) else str(Path(selected_path).parent)
    )
    if available_gb < 100:
        console.print(f"[yellow]{t('setup_path_low_space')}[/yellow]")

    # Save the path
    settings.set("paths.models", selected_path)
    settings.set("general._initialized", True)

    console.print()
    console.print(f"[green]✓[/green] {t('setup_model_path_set', path=selected_path)}")
    console.print()

    # Tips
    if lang == "zh":
        console.print("[dim]提示: 运行 'kt config show' 查看所有配置[/dim]")
    else:
        console.print("[dim]Tip: Run 'kt config show' to view all settings[/dim]")

    console.print()


def _prompt_custom_path(console, settings) -> str:
    """Prompt user to enter a custom path."""
    from rich.prompt import Prompt
    from pathlib import Path
    import os

    default_path = str(Path.home() / ".ktransformers" / "models")

    while True:
        custom_path = Prompt.ask(t("setup_enter_custom_path"), default=default_path)

        # Expand user home
        custom_path = os.path.expanduser(custom_path)

        # Check if path exists or parent is writable
        if os.path.exists(custom_path):
            if os.access(custom_path, os.W_OK):
                return custom_path
            else:
                console.print(f"[red]{t('setup_path_no_write')}[/red]")
        else:
            # Check if we can create it (parent writable)
            parent = str(Path(custom_path).parent)
            while not os.path.exists(parent) and parent != "/":
                parent = str(Path(parent).parent)

            if os.access(parent, os.W_OK):
                return custom_path
            else:
                console.print(f"[red]{t('setup_path_no_write')}[/red]")


def _install_shell_completion_silent() -> tuple[bool, str]:
    """Silently install shell completion for the current shell.

    Returns:
        Tuple of (success, shell_name)
    """
    import os
    import shutil
    from pathlib import Path

    # Detect current shell
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        shell_name = "zsh"
    elif "fish" in shell:
        shell_name = "fish"
    elif "bash" in shell:
        shell_name = "bash"
    else:
        # Default to bash
        shell_name = "bash"

    try:
        # Get the static completion script path
        cli_dir = Path(__file__).parent
        completions_dir = cli_dir / "completions"

        home = Path.home()

        if shell_name == "bash":
            src_file = completions_dir / "kt-completion.bash"
            dest_dir = home / ".bash_completions"
            dest_file = dest_dir / "kt.sh"
        elif shell_name == "zsh":
            src_file = completions_dir / "_kt"
            dest_dir = home / ".zsh_completions"
            dest_file = dest_dir / "_kt"
        elif shell_name == "fish":
            src_file = completions_dir / "kt.fish"
            dest_dir = home / ".config" / "fish" / "completions"
            dest_file = dest_dir / "kt.fish"
        else:
            return False, shell_name

        # Create destination directory if it doesn't exist
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy the static completion script
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            return True, shell_name
        else:
            return False, shell_name

    except (OSError, IOError):
        return False, shell_name


def _auto_install_completion() -> None:
    """Automatically install shell completion on first run."""
    from kt_kernel.cli.config.settings import get_settings
    from kt_kernel.cli.i18n import t
    from rich.console import Console
    from rich.panel import Panel
    import os

    settings = get_settings()

    # Check if already installed
    if settings.get("general._completion_installed", False):
        return

    # Try to install silently
    success, shell_name = _install_shell_completion_silent()

    # Mark as installed regardless of success (to avoid repeated attempts)
    settings.set("general._completion_installed", True)

    # Show activation message if successful and in interactive terminal
    if success and sys.stdin.isatty():
        console = Console(stderr=True)

        # Determine the activation command based on shell
        home = os.path.expanduser("~")
        if shell_name == "bash":
            completion_file = f"{home}/.bash_completions/kt.sh"
            activate_cmd = f"source {completion_file}"
        elif shell_name == "zsh":
            # Try to find the actual completion file location
            possible_paths = [
                f"{home}/.zsh_completions/_kt",
                f"{home}/.zfunc/_kt",
            ]
            completion_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    completion_file = path
                    break
            if completion_file:
                activate_cmd = f"source {completion_file}"
            else:
                activate_cmd = "exec $SHELL"  # Fallback: restart shell
        elif shell_name == "fish":
            completion_file = f"{home}/.config/fish/completions/kt.fish"
            activate_cmd = f"source {completion_file}"
        else:
            activate_cmd = "exec $SHELL"

        console.print()
        console.print(
            Panel.fit(
                f"[green]✓[/green] {t('completion_installed_for', shell=shell_name)}\n\n"
                f"{t('completion_activate_now')}\n"
                f"[yellow]{activate_cmd}[/yellow]\n\n"
                f"[dim]{t('completion_next_session')}[/dim]",
                title=f"[bold cyan]{t('completion_installed_title')}[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print()


def _apply_saved_language() -> None:
    """Apply the saved language setting."""
    from kt_kernel.cli.config.settings import get_settings

    settings = get_settings()
    lang = settings.get("general.language", "auto")

    if lang != "auto":
        set_lang(lang)


def main():
    """Main entry point."""
    # Check for first run (but not for certain commands)
    # Skip first-run check for: --help, config commands, version
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    skip_commands = ["--help", "-h", "config", "version", "--version"]

    should_check_first_run = True
    for arg in args:
        if arg in skip_commands:
            should_check_first_run = False
            break

    # Auto-install shell completion (silent, on first run only)
    if should_check_first_run:
        _auto_install_completion()

    # Check first run before applying saved language (to avoid creating config)
    if should_check_first_run and args:
        check_first_run()

    # Apply saved language setting (skip for completion commands to avoid I/O overhead)
    if should_check_first_run:
        _apply_saved_language()

    app()


if __name__ == "__main__":
    main()
