"""
Config command for kt-cli.

Manages kt-cli configuration.
"""

from typing import Optional

import typer
import yaml
from rich.syntax import Syntax

from ktcli.config.settings import get_settings
from ktcli.i18n import t
from ktcli.utils.console import confirm, console, print_error, print_success

app = typer.Typer(help="Manage kt-cli configuration")


@app.command(name="init")
def init() -> None:
    """Initialize or re-run the first-time setup wizard."""
    from ktcli.main import _show_first_run_setup
    from ktcli.config.settings import get_settings

    settings = get_settings()
    _show_first_run_setup(settings)


@app.command(name="show")
def show(
    key: Optional[str] = typer.Argument(None, help="Configuration key to show (e.g., server.port)"),
) -> None:
    """Show current configuration."""
    settings = get_settings()

    if key:
        value = settings.get(key)
        if value is not None:
            if isinstance(value, (dict, list)):
                console.print(yaml.dump({key: value}, default_flow_style=False, allow_unicode=True))
            else:
                console.print(t("config_get_value", key=key, value=value))
        else:
            print_error(t("config_get_not_found", key=key))
            raise typer.Exit(1)
    else:
        console.print(f"\n[bold]{t('config_show_title')}[/bold]\n")
        console.print(f"[dim]{t('config_file_location', path=str(settings.config_path))}[/dim]\n")

        config_yaml = yaml.dump(settings.get_all(), default_flow_style=False, allow_unicode=True)
        syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=False)
        console.print(syntax)


@app.command(name="set")
def set_config(
    key: str = typer.Argument(..., help="Configuration key (e.g., server.port)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value."""
    settings = get_settings()

    # Try to parse value as JSON/YAML for complex types
    parsed_value = _parse_value(value)

    settings.set(key, parsed_value)
    print_success(t("config_set_success", key=key, value=parsed_value))


@app.command(name="get")
def get_config(
    key: str = typer.Argument(..., help="Configuration key (e.g., server.port)"),
) -> None:
    """Get a configuration value."""
    settings = get_settings()
    value = settings.get(key)

    if value is not None:
        if isinstance(value, (dict, list)):
            console.print(yaml.dump(value, default_flow_style=False, allow_unicode=True))
        else:
            console.print(str(value))
    else:
        print_error(t("config_get_not_found", key=key))
        raise typer.Exit(1)


@app.command(name="reset")
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Reset configuration to defaults."""
    if not yes:
        if not confirm(t("config_reset_confirm"), default=False):
            raise typer.Abort()

    settings = get_settings()
    settings.reset()
    print_success(t("config_reset_success"))


@app.command(name="path")
def path() -> None:
    """Show configuration file path."""
    settings = get_settings()
    console.print(str(settings.config_path))


def _parse_value(value: str):
    """Parse a string value into appropriate Python type."""
    # Try boolean
    if value.lower() in ("true", "yes", "on", "1"):
        return True
    if value.lower() in ("false", "no", "off", "0"):
        return False

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try YAML/JSON parsing for lists/dicts
    try:
        parsed = yaml.safe_load(value)
        if isinstance(parsed, (dict, list)):
            return parsed
    except yaml.YAMLError:
        pass

    # Return as string
    return value
