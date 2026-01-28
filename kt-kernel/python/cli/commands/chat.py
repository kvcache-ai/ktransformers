"""
Chat command for kt-cli.

Provides interactive chat interface with running model server.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
)

# Try to import OpenAI SDK
try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def chat(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        "-H",
        help="Server host address",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (if server hosts multiple models)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0.0 to 2.0)",
    ),
    max_tokens: int = typer.Option(
        2048,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt",
    ),
    save_history: bool = typer.Option(
        True,
        "--save-history/--no-save-history",
        help="Save conversation history",
    ),
    history_file: Optional[Path] = typer.Option(
        None,
        "--history-file",
        help="Path to save conversation history",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Enable streaming output",
    ),
) -> None:
    """Start interactive chat with a running model server.

    Examples:
        kt chat                          # Connect to default server
        kt chat --host 127.0.0.1 -p 8080 # Connect to specific server
        kt chat -t 0.9 --max-tokens 4096 # Adjust generation parameters
    """
    if not HAS_OPENAI:
        print_error(t("chat_openai_required"))
        console.print()
        console.print(t("chat_install_hint"))
        console.print("  pip install openai")
        raise typer.Exit(1)

    settings = get_settings()

    # Resolve server connection
    final_host = host or settings.get("server.host", "127.0.0.1")
    final_port = port or settings.get("server.port", 30000)

    # Construct base URL for OpenAI-compatible API
    base_url = f"http://{final_host}:{final_port}/v1"

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]{t('chat_title')}[/bold cyan]\n\n"
            f"{t('chat_server')}: [yellow]{final_host}:{final_port}[/yellow]\n"
            f"{t('chat_temperature')}: [cyan]{temperature}[/cyan] | {t('chat_max_tokens')}: [cyan]{max_tokens}[/cyan]\n\n"
            f"[dim]{t('chat_help_hint')}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    # Check for proxy environment variables
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]
    detected_proxies = {var: os.environ.get(var) for var in proxy_vars if os.environ.get(var)}

    if detected_proxies:
        proxy_info = ", ".join(f"{k}={v}" for k, v in detected_proxies.items())
        console.print()
        print_warning(t("chat_proxy_detected"))
        console.print(f"  [dim]{proxy_info}[/dim]")
        console.print()

        use_proxy = Confirm.ask(t("chat_proxy_confirm"), default=False)

        if not use_proxy:
            # Temporarily disable proxy for this connection
            for var in proxy_vars:
                if var in os.environ:
                    del os.environ[var]
            print_info(t("chat_proxy_disabled"))
            console.print()

    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",  # SGLang doesn't require API key
        )

        # Test connection
        print_info(t("chat_connecting"))
        models = client.models.list()
        available_models = [m.id for m in models.data]

        if not available_models:
            print_error(t("chat_no_models"))
            raise typer.Exit(1)

        # Select model
        if model:
            if model not in available_models:
                print_warning(t("chat_model_not_found", model=model, available=", ".join(available_models)))
                selected_model = available_models[0]
            else:
                selected_model = model
        else:
            selected_model = available_models[0]

        print_success(t("chat_connected", model=selected_model))
        console.print()

    except Exception as e:
        print_error(t("chat_connect_failed", error=str(e)))
        console.print()
        console.print(t("chat_server_not_running"))
        console.print("  kt run <model>")
        raise typer.Exit(1)

    # Initialize conversation history
    messages = []

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Setup history file
    if save_history:
        if history_file is None:
            history_dir = settings.config_dir / "chat_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = history_dir / f"chat_{timestamp}.json"
        else:
            history_file = Path(history_file)
            history_file.parent.mkdir(parents=True, exist_ok=True)

    # Main chat loop
    try:
        while True:
            # Get user input
            try:
                user_input = Prompt.ask(f"[bold green]{t('chat_user_prompt')}[/bold green]")
            except (EOFError, KeyboardInterrupt):
                console.print()
                print_info(t("chat_goodbye"))
                break

            if not user_input.strip():
                continue

            # Handle special commands
            if user_input.startswith("/"):
                if _handle_command(user_input, messages, temperature, max_tokens):
                    continue
                else:
                    break  # Exit command

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            # Generate response
            console.print()
            console.print(f"[bold cyan]{t('chat_assistant_prompt')}[/bold cyan]")

            try:
                if stream:
                    # Streaming response
                    response_content = _stream_response(client, selected_model, messages, temperature, max_tokens)
                else:
                    # Non-streaming response
                    response_content = _generate_response(client, selected_model, messages, temperature, max_tokens)

                # Add assistant response to history
                messages.append({"role": "assistant", "content": response_content})

                console.print()

            except Exception as e:
                print_error(t("chat_generation_error", error=str(e)))
                # Remove the user message that caused the error
                messages.pop()
                continue

            # Save history if enabled
            if save_history:
                _save_history(history_file, messages, selected_model)

    except KeyboardInterrupt:
        console.print()
        console.print()
        print_info(t("chat_interrupted"))

    # Final history save
    if save_history and messages:
        _save_history(history_file, messages, selected_model)
        console.print(f"[dim]{t('chat_history_saved', path=str(history_file))}[/dim]")
        console.print()


def _stream_response(
    client: "OpenAI",
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate streaming response and display in real-time."""
    response_content = ""
    reasoning_content = ""

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta:
                reasoning_content += reasoning_delta
                console.print(reasoning_delta, end="", style="dim")
            if delta.content:
                content = delta.content
                response_content += content
                console.print(content, end="")

        console.print()  # Newline after streaming

    except Exception as e:
        raise Exception(f"Streaming error: {e}")

    return response_content


def _generate_response(
    client: "OpenAI",
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> str:
    """Generate non-streaming response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        content = response.choices[0].message.content

        # Display as markdown
        md = Markdown(content)
        console.print(md)

        return content

    except Exception as e:
        raise Exception(f"Generation error: {e}")


def _handle_command(command: str, messages: list, temperature: float, max_tokens: int) -> bool:
    """Handle special commands. Returns True to continue chat, False to exit."""
    cmd = command.lower().strip()

    if cmd in ["/quit", "/exit", "/q"]:
        console.print()
        print_info(t("chat_goodbye"))
        return False

    elif cmd in ["/help", "/h"]:
        console.print()
        console.print(
            Panel(
                f"[bold]{t('chat_help_title')}[/bold]\n\n{t('chat_help_content')}",
                title="Help",
                border_style="cyan",
            )
        )
        console.print()
        return True

    elif cmd in ["/clear", "/c"]:
        messages.clear()
        console.print()
        print_success(t("chat_history_cleared"))
        console.print()
        return True

    elif cmd in ["/history", "/hist"]:
        console.print()
        if not messages:
            print_info(t("chat_no_history"))
        else:
            console.print(
                Panel(
                    _format_history(messages),
                    title=t("chat_history_title", count=len(messages)),
                    border_style="cyan",
                )
            )
        console.print()
        return True

    elif cmd in ["/info", "/i"]:
        console.print()
        console.print(
            Panel(
                f"[bold]{t('chat_info_title')}[/bold]\n\n{t('chat_info_content', temperature=temperature, max_tokens=max_tokens, messages=len(messages))}",
                title="Info",
                border_style="cyan",
            )
        )
        console.print()
        return True

    elif cmd in ["/retry", "/r"]:
        if len(messages) >= 2 and messages[-1]["role"] == "assistant":
            # Remove last assistant response
            messages.pop()
            print_info(t("chat_retrying"))
            console.print()
        else:
            print_warning(t("chat_no_retry"))
            console.print()
        return True

    else:
        print_warning(t("chat_unknown_command", command=command))
        console.print(f"[dim]{t('chat_unknown_hint')}[/dim]")
        console.print()
        return True


def _format_history(messages: list) -> str:
    """Format conversation history for display."""
    lines = []
    for i, msg in enumerate(messages, 1):
        role = msg["role"].capitalize()
        content = msg["content"]

        # Truncate long messages
        if len(content) > 200:
            content = content[:200] + "..."

        lines.append(f"[bold]{i}. {role}:[/bold] {content}")

    return "\n\n".join(lines)


def _save_history(file_path: Path, messages: list, model: str) -> None:
    """Save conversation history to file."""
    try:
        history_data = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print_warning(f"Failed to save history: {e}")
