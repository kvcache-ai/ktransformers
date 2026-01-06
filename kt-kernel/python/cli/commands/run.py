"""
Run command for kt-cli.

Starts the model inference server using SGLang + kt-kernel.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    print_api_info,
    print_error,
    print_info,
    print_server_info,
    print_step,
    print_success,
    print_warning,
    prompt_choice,
)
from kt_kernel.cli.utils.environment import detect_cpu_info, detect_gpus, detect_ram_gb
from kt_kernel.cli.utils.user_model_registry import UserModelRegistry


@click.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    add_help_option=False,  # We'll handle help manually to avoid conflicts
)
@click.argument("model", required=False, default=None)
@click.option("--host", "-H", default=None, help="Server host address")
@click.option("--port", "-p", type=int, default=None, help="Server port")
@click.option("--gpu-experts", type=int, default=None, help="Number of GPU experts per layer")
@click.option("--cpu-threads", type=int, default=None, help="Number of CPU inference threads")
@click.option("--numa-nodes", type=int, default=None, help="Number of NUMA nodes")
@click.option(
    "--tensor-parallel-size", "--tp", "tensor_parallel_size", type=int, default=None, help="Tensor parallel size"
)
@click.option("--model-path", type=click.Path(), default=None, help="Custom model path")
@click.option("--weights-path", type=click.Path(), default=None, help="Custom quantized weights path")
@click.option("--kt-method", default=None, help="KT quantization method")
@click.option(
    "--kt-gpu-prefill-threshold", "kt_gpu_prefill_threshold", type=int, default=None, help="GPU prefill token threshold"
)
@click.option("--attention-backend", default=None, help="Attention backend")
@click.option("--max-total-tokens", "max_total_tokens", type=int, default=None, help="Maximum total tokens")
@click.option("--max-running-requests", "max_running_requests", type=int, default=None, help="Maximum running requests")
@click.option("--chunked-prefill-size", "chunked_prefill_size", type=int, default=None, help="Chunked prefill size")
@click.option("--mem-fraction-static", "mem_fraction_static", type=float, default=None, help="Memory fraction static")
@click.option("--watchdog-timeout", "watchdog_timeout", type=int, default=None, help="Watchdog timeout")
@click.option("--served-model-name", "served_model_name", default=None, help="Served model name")
@click.option(
    "--disable-shared-experts-fusion",
    "disable_shared_experts_fusion",
    is_flag=True,
    default=None,
    help="Disable shared experts fusion",
)
@click.option(
    "--enable-shared-experts-fusion",
    "enable_shared_experts_fusion",
    is_flag=True,
    default=False,
    help="Enable shared experts fusion",
)
@click.option("--quantize", "-q", is_flag=True, default=False, help="Quantize model")
@click.option("--advanced", is_flag=True, default=False, help="Show advanced options")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Show command without executing")
@click.pass_context
def run(
    ctx: click.Context,
    model: Optional[str],
    host: Optional[str],
    port: Optional[int],
    gpu_experts: Optional[int],
    cpu_threads: Optional[int],
    numa_nodes: Optional[int],
    tensor_parallel_size: Optional[int],
    model_path: Optional[str],
    weights_path: Optional[str],
    kt_method: Optional[str],
    kt_gpu_prefill_threshold: Optional[int],
    attention_backend: Optional[str],
    max_total_tokens: Optional[int],
    max_running_requests: Optional[int],
    chunked_prefill_size: Optional[int],
    mem_fraction_static: Optional[float],
    watchdog_timeout: Optional[int],
    served_model_name: Optional[str],
    disable_shared_experts_fusion: Optional[bool],
    enable_shared_experts_fusion: bool,
    quantize: bool,
    advanced: bool,
    dry_run: bool,
) -> None:
    """Start model inference server.

    \b
    Examples: kt run deepseek-v3 | kt run m2 --tensor-parallel-size 2 | kt run /path/to/model --gpu-experts 4

    \b
    Custom Options: Pass any SGLang server option directly (e.g., kt run m2 --fp8-gemm-backend triton).
    Common: --fp8-gemm-backend, --tool-call-parser, --reasoning-parser, --dp-size, --enable-ma
    For full list: python -m sglang.launch_server --help
    """
    # Handle --help manually since we disabled it
    # Check sys.argv for --help or -h since ctx.args may not be set yet
    if "--help" in sys.argv or "-h" in sys.argv:
        click.echo(ctx.get_help())
        return

    # Handle disable/enable shared experts fusion flags
    if enable_shared_experts_fusion:
        disable_shared_experts_fusion = False
    elif disable_shared_experts_fusion is None:
        disable_shared_experts_fusion = None

    # Convert Path objects from click
    model_path_obj = Path(model_path) if model_path else None
    weights_path_obj = Path(weights_path) if weights_path else None

    # Get extra args that weren't parsed (unknown options)
    # click stores these in ctx.args when ignore_unknown_options=True
    extra_cli_args = list(ctx.args) if ctx.args else []

    # Remove --help from extra args if present (already handled)
    extra_cli_args = [arg for arg in extra_cli_args if arg not in ["--help", "-h"]]

    # Call the actual run function implementation
    _run_impl(
        model=model,
        host=host,
        port=port,
        gpu_experts=gpu_experts,
        cpu_threads=cpu_threads,
        numa_nodes=numa_nodes,
        tensor_parallel_size=tensor_parallel_size,
        model_path=model_path_obj,
        weights_path=weights_path_obj,
        kt_method=kt_method,
        kt_gpu_prefill_threshold=kt_gpu_prefill_threshold,
        attention_backend=attention_backend,
        max_total_tokens=max_total_tokens,
        max_running_requests=max_running_requests,
        chunked_prefill_size=chunked_prefill_size,
        mem_fraction_static=mem_fraction_static,
        watchdog_timeout=watchdog_timeout,
        served_model_name=served_model_name,
        disable_shared_experts_fusion=disable_shared_experts_fusion,
        quantize=quantize,
        advanced=advanced,
        dry_run=dry_run,
        extra_cli_args=extra_cli_args,
    )


def _run_impl(
    model: Optional[str],
    host: Optional[str],
    port: Optional[int],
    gpu_experts: Optional[int],
    cpu_threads: Optional[int],
    numa_nodes: Optional[int],
    tensor_parallel_size: Optional[int],
    model_path: Optional[Path],
    weights_path: Optional[Path],
    kt_method: Optional[str],
    kt_gpu_prefill_threshold: Optional[int],
    attention_backend: Optional[str],
    max_total_tokens: Optional[int],
    max_running_requests: Optional[int],
    chunked_prefill_size: Optional[int],
    mem_fraction_static: Optional[float],
    watchdog_timeout: Optional[int],
    served_model_name: Optional[str],
    disable_shared_experts_fusion: Optional[bool],
    quantize: bool,
    advanced: bool,
    dry_run: bool,
    extra_cli_args: list[str],
) -> None:
    """Actual implementation of run command."""
    # Check if SGLang is installed before proceeding
    from kt_kernel.cli.utils.sglang_checker import (
        check_sglang_installation,
        check_sglang_kt_kernel_support,
        print_sglang_install_instructions,
        print_sglang_kt_kernel_instructions,
    )

    sglang_info = check_sglang_installation()
    if not sglang_info["installed"]:
        console.print()
        print_error(t("sglang_not_found"))
        console.print()
        print_sglang_install_instructions()
        raise typer.Exit(1)

    # Check if SGLang supports kt-kernel (has --kt-gpu-prefill-token-threshold parameter)
    kt_kernel_support = check_sglang_kt_kernel_support()
    if not kt_kernel_support["supported"]:
        console.print()
        print_error(t("sglang_kt_kernel_not_supported"))
        console.print()
        print_sglang_kt_kernel_instructions()
        raise typer.Exit(1)

    settings = get_settings()
    user_registry = UserModelRegistry()

    console.print()

    # If no model specified, show interactive selection
    if model is None:
        model = _interactive_model_selection(user_registry, settings)
        if model is None:
            raise typer.Exit(0)

    # Step 1: Detect hardware
    print_step(t("run_detecting_hardware"))
    gpus = detect_gpus()
    cpu = detect_cpu_info()
    ram = detect_ram_gb()

    if gpus:
        gpu_info = f"{gpus[0].name} ({gpus[0].vram_gb}GB VRAM)"
        if len(gpus) > 1:
            gpu_info += f" + {len(gpus) - 1} more"
        print_info(t("run_gpu_info", name=gpus[0].name, vram=gpus[0].vram_gb))
    else:
        print_warning(t("doctor_gpu_not_found"))
        gpu_info = "None"

    print_info(t("run_cpu_info", name=cpu.name, cores=cpu.cores, numa=cpu.numa_nodes))
    print_info(t("run_ram_info", total=int(ram)))

    # Step 2: Resolve model
    console.print()
    print_step(t("run_checking_model"))

    user_model = None
    resolved_model_path = model_path

    # Check if model is a path
    if Path(model).exists():
        resolved_model_path = Path(model)
        print_info(t("run_model_path", path=str(resolved_model_path)))

        # Try to find in user registry by path
        user_model = user_registry.find_by_path(str(resolved_model_path))
        if user_model:
            print_info(f"Using registered model: {user_model.name}")
        else:
            print_warning("Using unregistered model path. Consider adding it with 'kt model add'")
    else:
        # Search in user registry by name
        user_model = user_registry.get_model(model)

        if not user_model:
            print_error(t("run_model_not_found", name=model))
            console.print()

            # Show available models
            all_models = user_registry.list_models()
            if all_models:
                console.print("Available registered models:")
                for m in all_models[:5]:
                    console.print(f"  - {m.name}")
                if len(all_models) > 5:
                    console.print(f"  ... and {len(all_models) - 5} more")
            else:
                console.print("No models registered yet.")

            console.print()
            console.print(f"Add your model with: [cyan]kt model add /path/to/model[/cyan]")
            console.print(f"Or scan for models: [cyan]kt model scan[/cyan]")
            raise typer.Exit(1)

        # Use model path from registry
        resolved_model_path = Path(user_model.path)

        # Verify path exists
        if not resolved_model_path.exists():
            print_error(f"Model path does not exist: {resolved_model_path}")
            console.print()
            console.print(f"Run 'kt model refresh' to check all models")
            raise typer.Exit(1)

        print_info(t("run_model_path", path=str(resolved_model_path)))

    # Step 3: Check quantized weights (only if explicitly requested)
    resolved_weights_path = None

    # Only use quantized weights if explicitly specified by user
    if weights_path is not None:
        # User explicitly specified weights path
        resolved_weights_path = weights_path
        if not resolved_weights_path.exists():
            print_error(t("run_weights_not_found"))
            console.print(f"  Path: {resolved_weights_path}")
            raise typer.Exit(1)
        print_info(f"Using quantized weights: {resolved_weights_path}")
    elif quantize:
        # User requested quantization
        console.print()
        print_step(t("run_quantizing"))
        # TODO: Implement quantization
        print_warning("Quantization not yet implemented. Please run 'kt quant' manually.")
        raise typer.Exit(1)
    else:
        # Default: use original precision model without quantization
        console.print()
        print_info("Using original precision model (no quantization)")

    # Step 4: Build command
    # Resolve all parameters (CLI > config > auto-detect)
    final_host = host or settings.get("server.host", "0.0.0.0")
    final_port = port or settings.get("server.port", 30000)

    # Determine tensor parallel size
    # Priority: CLI > config > auto-detect
    if tensor_parallel_size:
        final_tensor_parallel_size = tensor_parallel_size
    elif settings.get("inference.tensor_parallel_size"):
        final_tensor_parallel_size = settings.get("inference.tensor_parallel_size")
    else:
        # Auto-detect from GPUs
        final_tensor_parallel_size = len(gpus) if gpus else 1

    # CPU/GPU configuration with smart defaults
    # kt-cpuinfer: default to 80% of total CPU threads (cores * NUMA nodes)
    total_threads = cpu.cores * cpu.numa_nodes
    final_cpu_threads = cpu_threads or settings.get("inference.cpu_threads") or int(total_threads * 0.8)

    # kt-threadpool-count: default to NUMA node count
    final_numa_nodes = numa_nodes or settings.get("inference.numa_nodes") or cpu.numa_nodes

    # kt-num-gpu-experts: use CLI or config value
    final_gpu_experts = gpu_experts or settings.get("inference.gpu_experts", 1)

    # KT-kernel options
    final_kt_method = kt_method or settings.get("inference.kt_method", "AMXINT4")
    final_kt_gpu_prefill_threshold = kt_gpu_prefill_threshold or settings.get(
        "inference.kt_gpu_prefill_token_threshold", 4096
    )

    # SGLang options
    final_attention_backend = attention_backend or settings.get("inference.attention_backend", "triton")
    final_max_total_tokens = max_total_tokens or settings.get("inference.max_total_tokens", 40000)
    final_max_running_requests = max_running_requests or settings.get("inference.max_running_requests", 32)
    final_chunked_prefill_size = chunked_prefill_size or settings.get("inference.chunked_prefill_size", 4096)
    final_mem_fraction_static = mem_fraction_static or settings.get("inference.mem_fraction_static", 0.98)
    final_watchdog_timeout = watchdog_timeout or settings.get("inference.watchdog_timeout", 3000)
    final_served_model_name = served_model_name or settings.get("inference.served_model_name", "")

    # Performance flags
    if disable_shared_experts_fusion is not None:
        final_disable_shared_experts_fusion = disable_shared_experts_fusion
    else:
        final_disable_shared_experts_fusion = settings.get("inference.disable_shared_experts_fusion", False)

    # Pass extra CLI parameters
    extra_params = {}

    cmd = _build_sglang_command(
        model_path=resolved_model_path,
        weights_path=resolved_weights_path,
        host=final_host,
        port=final_port,
        gpu_experts=final_gpu_experts,
        cpu_threads=final_cpu_threads,
        numa_nodes=final_numa_nodes,
        tensor_parallel_size=final_tensor_parallel_size,
        kt_method=final_kt_method,
        kt_gpu_prefill_threshold=final_kt_gpu_prefill_threshold,
        attention_backend=final_attention_backend,
        max_total_tokens=final_max_total_tokens,
        max_running_requests=final_max_running_requests,
        chunked_prefill_size=final_chunked_prefill_size,
        mem_fraction_static=final_mem_fraction_static,
        watchdog_timeout=final_watchdog_timeout,
        served_model_name=final_served_model_name,
        disable_shared_experts_fusion=final_disable_shared_experts_fusion,
        settings=settings,
        extra_model_params=extra_params,
        extra_cli_args=extra_cli_args,
    )

    # Prepare environment variables
    env = os.environ.copy()
    # Add environment variables from advanced.env
    env.update(settings.get_env_vars())
    # Add environment variables from inference.env
    inference_env = settings.get("inference.env", {})
    if isinstance(inference_env, dict):
        env.update({k: str(v) for k, v in inference_env.items()})

    # Step 5: Show configuration summary
    console.print()
    print_step("Configuration")

    # Model info
    # Display model name
    if user_model:
        console.print(f"  Model: [bold]{user_model.name}[/bold]")
    else:
        console.print(f"  Model: [bold]{resolved_model_path.name}[/bold]")

    console.print(f"  Path: [dim]{resolved_model_path}[/dim]")

    # Key parameters
    console.print()
    console.print(f"  GPU Experts: [cyan]{final_gpu_experts}[/cyan] per layer")
    console.print(f"  CPU Threads (kt-cpuinfer): [cyan]{final_cpu_threads}[/cyan]")
    console.print(f"  NUMA Nodes (kt-threadpool-count): [cyan]{final_numa_nodes}[/cyan]")
    console.print(f"  Tensor Parallel: [cyan]{final_tensor_parallel_size}[/cyan]")
    console.print(f"  Method: [cyan]{final_kt_method}[/cyan]")
    console.print(f"  Attention: [cyan]{final_attention_backend}[/cyan]")

    # Weights info
    if resolved_weights_path:
        console.print()
        console.print(f"  Quantized weights: [yellow]{resolved_weights_path}[/yellow]")

    console.print()
    console.print(f"  Server: [green]http://{final_host}:{final_port}[/green]")
    console.print()

    # Step 6: Show or execute
    if dry_run:
        console.print()
        console.print("[bold]Command:[/bold]")
        console.print()
        console.print(f"  [dim]{' '.join(cmd)}[/dim]")
        console.print()
        return

    # Execute with prepared environment variables
    # Don't print "Server started" or API info here - let sglang's logs speak for themselves
    # The actual startup takes time and these messages are misleading

    # Print the command being executed
    console.print()
    console.print("[bold]Launching server with command:[/bold]")
    console.print()
    console.print(f"  [dim]{' '.join(cmd)}[/dim]")
    console.print()

    try:
        # Execute directly without intercepting output or signals
        # This allows direct output to terminal and Ctrl+C to work naturally
        process = subprocess.run(cmd, env=env)
        sys.exit(process.returncode)

    except FileNotFoundError:
        from kt_kernel.cli.utils.sglang_checker import print_sglang_install_instructions

        print_error(t("sglang_not_found"))
        console.print()
        print_sglang_install_instructions()
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        raise typer.Exit(1)


# Dead code removed: _find_model_path() and _find_weights_path()
# These functions were part of the old builtin model system


def _build_sglang_command(
    model_path: Path,
    weights_path: Optional[Path],
    host: str,
    port: int,
    gpu_experts: int,
    cpu_threads: int,
    numa_nodes: int,
    tensor_parallel_size: int,
    kt_method: str,
    kt_gpu_prefill_threshold: int,
    attention_backend: str,
    max_total_tokens: int,
    max_running_requests: int,
    chunked_prefill_size: int,
    mem_fraction_static: float,
    watchdog_timeout: int,
    served_model_name: str,
    disable_shared_experts_fusion: bool,
    settings,
    extra_model_params: Optional[dict] = None,  # New parameter for additional params
    extra_cli_args: Optional[list[str]] = None,  # Extra args from CLI to pass to sglang
) -> list[str]:
    """Build the SGLang launch command."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        str(model_path),
    ]

    # Add kt-kernel options
    # kt-kernel is needed for:
    # 1. Quantized models (when weights_path is provided)
    # 2. MoE models with CPU offloading (when kt-cpuinfer > 0 or kt-num-gpu-experts is configured)
    use_kt_kernel = False

    # Check if we should use kt-kernel
    if weights_path:
        # Quantized model - always use kt-kernel
        use_kt_kernel = True
    elif cpu_threads > 0 or gpu_experts > 1:
        # CPU offloading configured - use kt-kernel
        use_kt_kernel = True

    if use_kt_kernel:
        # Add kt-weight-path: use quantized weights if available, otherwise use model path
        weight_path_to_use = weights_path if weights_path else model_path

        # Add kt-kernel configuration
        cmd.extend(
            [
                "--kt-weight-path",
                str(weight_path_to_use),
                "--kt-cpuinfer",
                str(cpu_threads),
                "--kt-threadpool-count",
                str(numa_nodes),
                "--kt-num-gpu-experts",
                str(gpu_experts),
                "--kt-method",
                kt_method,
                "--kt-gpu-prefill-token-threshold",
                str(kt_gpu_prefill_threshold),
            ]
        )

    # Add SGLang options
    cmd.extend(
        [
            "--attention-backend",
            attention_backend,
            "--trust-remote-code",
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--chunked-prefill-size",
            str(chunked_prefill_size),
            "--max-running-requests",
            str(max_running_requests),
            "--max-total-tokens",
            str(max_total_tokens),
            "--watchdog-timeout",
            str(watchdog_timeout),
            "--enable-mixed-chunk",
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--enable-p2p-check",
        ]
    )

    # Add served model name if specified
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    # Add performance flags
    if disable_shared_experts_fusion:
        cmd.append("--disable-shared-experts-fusion")

    # Add any extra parameters from model defaults that weren't explicitly handled
    if extra_model_params:
        # List of parameters already handled above
        handled_params = {
            "kt-num-gpu-experts",
            "kt-cpuinfer",
            "kt-threadpool-count",
            "kt-method",
            "kt-gpu-prefill-token-threshold",
            "attention-backend",
            "tensor-parallel-size",
            "max-total-tokens",
            "max-running-requests",
            "chunked-prefill-size",
            "mem-fraction-static",
            "watchdog-timeout",
            "served-model-name",
            "disable-shared-experts-fusion",
        }

        for key, value in extra_model_params.items():
            if key not in handled_params:
                # Add unhandled parameters dynamically
                cmd.append(f"--{key}")
                if isinstance(value, bool):
                    # Boolean flags don't need a value
                    if not value:
                        # For False boolean, skip the flag entirely
                        cmd.pop()  # Remove the flag we just added
                else:
                    cmd.append(str(value))

    # Add extra args from settings
    extra_args = settings.get("advanced.sglang_args", [])
    if extra_args:
        cmd.extend(extra_args)

    # Add extra CLI args (user-provided options not defined in kt CLI)
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    return cmd


def _interactive_model_selection(user_registry, settings) -> Optional[str]:
    """Show interactive model selection interface.

    Returns:
        Selected model name or None if cancelled.
    """
    from rich.panel import Panel
    from rich.prompt import Prompt

    # Get all user models
    all_models = user_registry.list_models()

    if not all_models:
        console.print()
        print_warning("No models registered.")
        console.print()
        console.print(f"  Add models with: [cyan]kt model scan[/cyan]")
        console.print(f"  Or manually: [cyan]kt model add /path/to/model[/cyan]")
        console.print()
        return None

    console.print()
    console.print(
        Panel.fit(
            "Select a model to run",
            border_style="cyan",
        )
    )
    console.print()

    # Build choices list
    choices = []
    choice_map = {}  # index -> model name

    # Show all user models
    console.print(f"[bold green]Available Models:[/bold green]")
    console.print()

    for i, model in enumerate(all_models, 1):
        # Check if path exists
        path_status = "✓" if model.path_exists() else "✗ Missing"
        console.print(f"  [cyan][{i}][/cyan] [bold]{model.name}[/bold] [{path_status}]")
        console.print(f"      [dim]{model.format} - {model.path}[/dim]")
        choices.append(str(i))
        choice_map[str(i)] = model.name

    console.print()

    # Add cancel option
    cancel_idx = str(len(choices) + 1)
    console.print(f"  [cyan][{cancel_idx}][/cyan] [dim]Cancel[/dim]")
    choices.append(cancel_idx)
    console.print()

    # Prompt for selection
    try:
        selection = Prompt.ask(
            "Select model",
            choices=choices,
            default="1" if choices else cancel_idx,
        )
    except KeyboardInterrupt:
        console.print()
        return None

    if selection == cancel_idx:
        return None

    return choice_map.get(selection)
