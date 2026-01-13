"""
Main entry point for kt-cli.

KTransformers CLI - A unified command-line interface for KTransformers.
"""

import sys

import typer

from kt_kernel.cli import __version__
from kt_kernel.cli.commands import bench, chat, config, doctor, model, quant, run, sft, tuna, version
from kt_kernel.cli.i18n import t, set_lang, get_lang


def _get_app_help() -> str:
    """Get app help text based on current language."""
    lang = get_lang()
    if lang == "zh":
        return "KTransformers CLI - KTransformers 统一命令行界面"
    return "KTransformers CLI - A unified command-line interface for KTransformers."


def _get_help(key: str) -> str:
    """Get help text based on current language."""
    help_texts = {
        "version": {"en": "Show version information", "zh": "显示版本信息"},
        "run": {"en": "Start model inference server", "zh": "启动模型推理服务器"},
        "chat": {"en": "Interactive chat with running model", "zh": "与运行中的模型进行交互式聊天"},
        "quant": {"en": "Quantize model weights", "zh": "量化模型权重"},
        "edit": {"en": "Edit model information", "zh": "编辑模型信息"},
        "bench": {"en": "Run full benchmark", "zh": "运行完整基准测试"},
        "microbench": {"en": "Run micro-benchmark", "zh": "运行微基准测试"},
        "doctor": {"en": "Diagnose environment issues", "zh": "诊断环境问题"},
        "model": {"en": "Manage models and storage paths", "zh": "管理模型和存储路径"},
        "config": {"en": "Manage configuration", "zh": "管理配置"},
        "sft": {"en": "Fine-tuning with LlamaFactory", "zh": "使用 LlamaFactory 进行微调"},
    }
    lang = get_lang()
    return help_texts.get(key, {}).get(lang, help_texts.get(key, {}).get("en", key))


# Create main app with dynamic help
app = typer.Typer(
    name="kt",
    help="KTransformers CLI - A unified command-line interface for KTransformers.",
    no_args_is_help=False,  # Handle no-args case manually to support first-run setup
    add_completion=False,  # Use static completion scripts instead of dynamic completion
    rich_markup_mode="rich",
)


def _update_help_texts() -> None:
    """Update all help texts based on current language setting."""
    # Update main app help
    app.info.help = _get_app_help()

    # Update command help texts
    for cmd_info in app.registered_commands:
        # cmd_info is a CommandInfo object
        if hasattr(cmd_info, "name") and cmd_info.name:
            cmd_info.help = _get_help(cmd_info.name)

    # Update sub-app help texts
    for group_info in app.registered_groups:
        if hasattr(group_info, "name") and group_info.name:
            group_info.help = _get_help(group_info.name)


# Commands are registered later after tui_command is defined


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

    from kt_kernel.cli.utils.environment import scan_storage_locations, format_size_gb

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

    choice = Prompt.ask("Enter choice / 输入选择", choices=["1", "2"], default="1")
    lang = "en" if choice == "1" else "zh"

    # Save language setting
    settings.set("general.language", lang)
    set_lang(lang)

    # Confirmation message
    console.print()
    if lang == "zh":
        console.print("[green]✓[/green] 语言已设置为中文")
    else:
        console.print("[green]✓[/green] Language set to English")

    # Model discovery section
    console.print()
    if lang == "zh":
        console.print("[bold]发现模型权重[/bold]")
        console.print()
        console.print("[dim]扫描系统中已有的模型权重文件，以便快速添加到模型列表。[/dim]")
        console.print()
        console.print("  [cyan][1][/cyan] 全局扫描 (自动扫描所有非系统路径)")
        console.print("  [cyan][2][/cyan] 手动指定路径 (可添加多个)")
        console.print("  [cyan][3][/cyan] 跳过 (稍后手动添加)")
        console.print()
        scan_choice = Prompt.ask("选择扫描方式", choices=["1", "2", "3"], default="1")
    else:
        console.print("[bold]Discover Model Weights[/bold]")
        console.print()
        console.print("[dim]Scan existing model weights on your system to quickly add them to the model list.[/dim]")
        console.print()
        console.print("  [cyan][1][/cyan] Global scan (auto-scan all non-system paths)")
        console.print("  [cyan][2][/cyan] Manual paths (add multiple paths)")
        console.print("  [cyan][3][/cyan] Skip (add manually later)")
        console.print()
        scan_choice = Prompt.ask("Select scan method", choices=["1", "2", "3"], default="1")

    if scan_choice == "1":
        # Global scan
        from kt_kernel.cli.utils.model_discovery import discover_and_register_global, format_discovery_summary

        console.print()
        try:
            total_found, new_found, registered = discover_and_register_global(
                min_size_gb=2.0, max_depth=6, show_progress=True, lang=lang
            )

            format_discovery_summary(
                total_found=total_found,
                new_found=new_found,
                registered=registered,
                lang=lang,
                show_models=True,
                max_show=10,
            )

        except Exception as e:
            console.print(f"[yellow]Warning: Scan failed - {e}[/yellow]")

    elif scan_choice == "2":
        # Manual path specification
        from kt_kernel.cli.utils.model_discovery import discover_and_register_path
        import os

        discovered_paths = set()  # Track paths discovered in this session
        total_registered = []

        while True:
            console.print()
            if lang == "zh":
                path = Prompt.ask("输入要扫描的路径 (例如: /mnt/data/models)")
            else:
                path = Prompt.ask("Enter path to scan (e.g., /mnt/data/models)")

            # Expand and validate path
            path = os.path.expanduser(path)

            if not os.path.exists(path):
                if lang == "zh":
                    console.print(f"[yellow]警告: 路径不存在: {path}[/yellow]")
                else:
                    console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
                continue

            if not os.path.isdir(path):
                if lang == "zh":
                    console.print(f"[yellow]警告: 不是一个目录: {path}[/yellow]")
                else:
                    console.print(f"[yellow]Warning: Not a directory: {path}[/yellow]")
                continue

            # Scan this path
            console.print()
            try:
                total_found, new_found, registered = discover_and_register_path(
                    path=path, min_size_gb=2.0, existing_paths=discovered_paths, show_progress=True, lang=lang
                )

                # Update discovered paths
                for model in registered:
                    discovered_paths.add(model.path)
                total_registered.extend(registered)

                console.print()
                if lang == "zh":
                    console.print(f"[green]✓[/green] 在此路径找到 {total_found} 个模型，其中 {new_found} 个为新模型")
                else:
                    console.print(f"[green]✓[/green] Found {total_found} models in this path, {new_found} are new")

                if new_found > 0:
                    for model in registered[:5]:
                        console.print(f"  • {model.name} ({model.format})")

                    if len(registered) > 5:
                        if lang == "zh":
                            console.print(f"  [dim]... 还有 {len(registered) - 5} 个新模型[/dim]")
                        else:
                            console.print(f"  [dim]... and {len(registered) - 5} more new models[/dim]")

            except Exception as e:
                console.print(f"[red]Error scanning path: {e}[/red]")

            # Ask if continue
            console.print()
            if lang == "zh":
                continue_scan = Confirm.ask("是否继续添加其他路径?", default=False)
            else:
                continue_scan = Confirm.ask("Continue adding more paths?", default=False)

            if not continue_scan:
                break

        if total_registered:
            console.print()
            if lang == "zh":
                console.print(f"[green]✓[/green] 总共发现 {len(total_registered)} 个新模型")
            else:
                console.print(f"[green]✓[/green] Total {len(total_registered)} new models discovered")

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
        # Show storage location options
        for i, loc in enumerate(locations[:5], 1):  # Show top 5 options
            available = format_size_gb(loc.available_gb)
            total = format_size_gb(loc.total_gb)

            # Build the option string
            if i == 1:
                option_str = t("setup_disk_option_recommended", path=loc.path, available=available, total=total)
            else:
                option_str = t("setup_disk_option", path=loc.path, available=available, total=total)

            console.print(f"  [cyan][{i}][/cyan] {option_str}")

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


def _install_shell_completion() -> None:
    """Install shell completion scripts to user directories.

    Uses standard locations that are auto-loaded by shell completion systems:
    - Bash: ~/.local/share/bash-completion/completions/kt (auto-loaded by bash-completion 2.0+)
    - Zsh: ~/.zfunc/_kt (requires fpath setup, but commonly used)
    - Fish: ~/.config/fish/completions/kt.fish (auto-loaded)
    """
    import os
    import shutil
    from pathlib import Path

    from kt_kernel.cli.config.settings import get_settings

    settings = get_settings()

    # Check if already installed
    if settings.get("general._completion_installed", False):
        return

    # Detect current shell
    shell = os.environ.get("SHELL", "")
    shell_name = "zsh" if "zsh" in shell else "fish" if "fish" in shell else "bash"

    try:
        cli_dir = Path(__file__).parent
        completions_dir = cli_dir / "completions"
        home = Path.home()

        def install_completion(src_name: str, dest_dir: Path, dest_name: str) -> None:
            """Install completion file from source to destination."""
            src_file = completions_dir / src_name
            if src_file.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dest_dir / dest_name)

        if shell_name == "bash":
            install_completion(
                "kt-completion.bash", home / ".local" / "share" / "bash-completion" / "completions", "kt"
            )
        elif shell_name == "zsh":
            install_completion("_kt", home / ".zfunc", "_kt")
        elif shell_name == "fish":
            install_completion("kt.fish", home / ".config" / "fish" / "completions", "kt.fish")

        # Mark as installed
        settings.set("general._completion_installed", True)

        # For bash/zsh, completion will work in new terminals automatically
        # (bash-completion 2.0+ auto-loads from ~/.local/share/bash-completion/completions/)

    except (OSError, IOError):
        # Silently ignore errors - completion is not critical
        pass


def _apply_saved_language() -> None:
    """Apply the saved language setting.

    Priority:
    1. KT_LANG environment variable (if already set, don't override)
    2. Config file setting
    3. System locale (auto)
    """
    import os

    # Don't override if KT_LANG is already set by user
    if os.environ.get("KT_LANG"):
        return

    from kt_kernel.cli.config.settings import get_settings

    settings = get_settings()
    lang = settings.get("general.language", "auto")

    if lang != "auto":
        set_lang(lang)


def tui_command():
    """Launch the TUI (Text User Interface) for interactive model management."""
    # First check if textual is available
    try:
        import textual
    except ImportError:
        from rich.console import Console

        console = Console()
        console.print()
        console.print("[yellow]Interactive TUI not available - Textual not installed.[/yellow]")
        console.print()
        console.print("  Install with: [cyan]pip install textual>=0.47.0[/cyan]")
        console.print("  Or use CLI commands: [cyan]kt --help[/cyan]")
        console.print()
        return

    # Try to import TUI app
    try:
        from kt_kernel.cli.tui import ModelManagerApp

        if ModelManagerApp is None:
            raise ImportError("TUI module not properly initialized")

        app_instance = ModelManagerApp()
        app_instance.run()

    except ImportError as e:
        # Check if the error is about kt_kernel extension (expected in dev environment)
        error_msg = str(e)
        if "kt_kernel_ext" in error_msg or "_PyThreadState" in error_msg:
            from rich.console import Console

            console = Console()
            console.print()
            console.print("[yellow]TUI cannot run in current environment (kt_kernel extension not loaded)[/yellow]")
            console.print()
            console.print("  This is a development environment issue.")
            console.print("  The TUI will work correctly in a properly installed environment.")
            console.print()
            console.print("  For now, use CLI commands: [cyan]kt model list[/cyan]")
            console.print()
        else:
            # Real Textual import error
            from rich.console import Console

            console = Console()
            console.print()
            console.print(f"[red]Failed to load TUI: {e}[/red]")
            console.print()
            console.print("  Use CLI commands: [cyan]kt --help[/cyan]")
            console.print()

    except Exception as e:
        from rich.console import Console

        console = Console()
        console.print()
        console.print(f"[red]TUI error: {e}[/red]")
        console.print()
        console.print("  Use CLI commands: [cyan]kt --help[/cyan]")
        console.print()


# Register commands after tui_command is defined
app.command(name="version", help="Show version information")(version.version)
app.command(name="tui", help="Launch TUI (Text User Interface)")(tui_command)
app.command(name="tuna", help="Optimize model with TUNA")(tuna.tuna)
# Run command is handled specially in main() to allow extra args
# (not registered here to avoid typer's argument parsing)
app.command(name="chat", help="Interactive chat with running model")(chat.chat)
app.command(name="quant", help="Quantize model weights")(quant.quant)
app.command(name="edit", help="Edit model information")(model.edit_model)
app.command(name="bench", help="Run full benchmark")(bench.bench)
app.command(name="microbench", help="Run micro-benchmark")(bench.microbench)
app.command(name="doctor", help="Diagnose environment issues")(doctor.doctor)

# Register sub-apps
app.add_typer(model.app, name="model", help="Manage models and storage paths")
app.add_typer(config.app, name="config", help="Manage configuration")
app.add_typer(sft.app, name="sft", help="Fine-tuning with LlamaFactory")


def main():
    """Main entry point."""
    # Apply saved language setting first (before anything else for correct help display)
    _apply_saved_language()

    # Update help texts based on language
    _update_help_texts()

    # Check for first run (but not for certain commands)
    # Skip first-run check for: --help, config commands, version
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    skip_commands = ["--help", "-h", "config", "version", "--version", "--no-tui"]

    should_check_first_run = True
    for arg in args:
        if arg in skip_commands:
            should_check_first_run = False
            break

    # Handle no arguments case
    if not args:
        # Check if this is first run
        from kt_kernel.cli.config.settings import DEFAULT_CONFIG_FILE, get_settings

        is_first_run = False
        if not DEFAULT_CONFIG_FILE.exists():
            is_first_run = True
        else:
            settings = get_settings()
            if not settings.get("general._initialized"):
                is_first_run = True

        if is_first_run:
            # First run - start initialization
            _install_shell_completion()
            check_first_run()
            return
        else:
            # Not first run - show help
            app(["--help"])
            return

    # Auto-install shell completion on first run
    if should_check_first_run:
        _install_shell_completion()

    # Check first run before running commands
    if should_check_first_run:
        check_first_run()

    # Handle "run" command specially to pass through unknown options
    if args and args[0] == "run":
        # Get args after "run"
        run_args = args[1:]
        # Use click command directly with ignore_unknown_options
        from kt_kernel.cli.commands import run as run_module

        sys.exit(run_module.run.main(args=run_args, standalone_mode=False))

    app()


if __name__ == "__main__":
    main()
