"""
Tuna command for kt-cli.

Auto-tunes GPU experts configuration for optimal performance.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import console, print_error, print_info, print_step, print_success
from kt_kernel.cli.utils.environment import detect_cpu_info, detect_gpus, detect_ram_gb
from kt_kernel.cli.utils.model_registry import get_registry
from kt_kernel.cli.utils.tuna_engine import run_tuna


def tuna(
    model: str = typer.Argument(..., help="Model name or path to tune"),
    max_total_tokens: Optional[int] = typer.Option(
        None, "--max-total-tokens", help="Maximum total tokens (default: from model config or 40000)"
    ),
    tensor_parallel_size: Optional[int] = typer.Option(
        None, "--tensor-parallel-size", "--tp", help="Tensor parallel size (default: auto-detect)"
    ),
    kt_method: Optional[str] = typer.Option(None, "--kt-method", help="KT quantization method (default: AMXINT4)"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Custom model path"),
    weights_path: Optional[str] = typer.Option(None, "--weights-path", help="Custom quantized weights path"),
    cpu_threads: Optional[int] = typer.Option(None, "--cpu-threads", help="Number of CPU inference threads"),
    numa_nodes: Optional[int] = typer.Option(None, "--numa-nodes", help="Number of NUMA nodes"),
    attention_backend: Optional[str] = typer.Option(None, "--attention-backend", help="Attention backend"),
    kt_gpu_prefill_threshold: Optional[int] = typer.Option(
        None, "--kt-gpu-prefill-threshold", help="GPU prefill token threshold"
    ),
    chunked_prefill_size: Optional[int] = typer.Option(None, "--chunked-prefill-size", help="Chunked prefill size"),
    mem_fraction_static: Optional[float] = typer.Option(None, "--mem-fraction-static", help="Memory fraction static"),
    max_running_requests: Optional[int] = typer.Option(None, "--max-running-requests", help="Maximum running requests"),
    watchdog_timeout: Optional[int] = typer.Option(None, "--watchdog-timeout", help="Watchdog timeout"),
    disable_shared_experts_fusion: bool = typer.Option(
        False, "--disable-shared-experts-fusion", help="Disable shared experts fusion"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed test logs"),
) -> None:
    """
    Auto-tune GPU experts configuration for optimal performance.

    Finds the maximum number of GPU experts that can fit in VRAM
    through binary search by testing actual server launches.

    Example:
        kt tuna deepseek-v3
        kt tuna deepseek-v3 --max-total-tokens 50000
        kt tuna /path/to/model --kt-method AMXINT4
    """
    console.print()

    # Step 1: Detect hardware
    print_step("Detecting hardware")
    gpus = detect_gpus()
    cpu = detect_cpu_info()
    ram = detect_ram_gb()

    if not gpus:
        print_error("No GPUs detected. Tuna requires GPUs for tuning.")
        console.print()
        console.print("Tuna automatically optimizes GPU expert placement.")
        console.print("For CPU-only inference, use --gpu-experts 0")
        raise typer.Exit(1)

    gpu_info = f"{len(gpus)}× {gpus[0].name} ({gpus[0].vram_gb}GB)"
    print_info(f"GPUs: {gpu_info}")
    print_info(f"CPU: {cpu.name} ({cpu.cores} cores, {cpu.numa_nodes} NUMA nodes)")
    print_info(f"RAM: {int(ram)}GB")

    # Step 2: Resolve model
    console.print()
    print_step("Resolving model")

    settings = get_settings()
    registry = get_registry()

    model_info = None
    resolved_model_path = None

    if model_path:
        resolved_model_path = Path(model_path)
    elif Path(model).exists():
        resolved_model_path = Path(model)
    else:
        # Search in registry
        matches = registry.search(model)

        if not matches:
            print_error(f"Model '{model}' not found")
            console.print()
            console.print("Available models:")
            for m in registry.list_all()[:5]:
                console.print(f"  - {m.name} ({', '.join(m.aliases[:2])})")
            raise typer.Exit(1)

        if len(matches) == 1:
            model_info = matches[0]
        else:
            print_error(f"Multiple matches found for '{model}', please be more specific")
            raise typer.Exit(1)

        # Find model path
        from kt_kernel.cli.commands.run import _find_model_path

        resolved_model_path = _find_model_path(model_info, settings)
        if resolved_model_path is None:
            print_error(f"Model '{model_info.name}' not found on disk")
            console.print()
            console.print(f"Download with: kt model download {model_info.aliases[0]}")
            raise typer.Exit(1)

    if not resolved_model_path.exists():
        print_error(f"Model path does not exist: {resolved_model_path}")
        raise typer.Exit(1)

    print_info(f"Model path: {resolved_model_path}")

    # Step 3: Resolve parameters
    console.print()
    print_step("Configuration")

    # Get defaults from model info if available
    model_defaults = model_info.default_params if model_info else {}

    # Determine tensor parallel size
    if tensor_parallel_size is None:
        # Auto-detect from GPUs
        detected_gpu_count = len(gpus)
        if model_info and model_info.max_tensor_parallel_size is not None:
            final_tensor_parallel_size = min(detected_gpu_count, model_info.max_tensor_parallel_size)
        else:
            final_tensor_parallel_size = detected_gpu_count
    else:
        final_tensor_parallel_size = tensor_parallel_size

    # Apply model's max constraint
    if model_info and model_info.max_tensor_parallel_size is not None:
        if final_tensor_parallel_size > model_info.max_tensor_parallel_size:
            print_info(
                f"Reducing TP from {final_tensor_parallel_size} to {model_info.max_tensor_parallel_size} "
                f"(model limit)"
            )
            final_tensor_parallel_size = model_info.max_tensor_parallel_size

    # Other parameters
    final_max_total_tokens = max_total_tokens or model_defaults.get("max-total-tokens") or 40000
    final_kt_method = kt_method or model_defaults.get("kt-method") or "AMXINT4"

    total_threads = cpu.cores * cpu.numa_nodes
    final_cpu_threads = cpu_threads or model_defaults.get("kt-cpuinfer") or int(total_threads * 0.8)
    final_numa_nodes = numa_nodes or model_defaults.get("kt-threadpool-count") or cpu.numa_nodes
    final_attention_backend = attention_backend or model_defaults.get("attention-backend") or "triton"
    final_kt_gpu_prefill_threshold = (
        kt_gpu_prefill_threshold or model_defaults.get("kt-gpu-prefill-token-threshold") or 4096
    )
    final_chunked_prefill_size = chunked_prefill_size or model_defaults.get("chunked-prefill-size") or 4096
    final_mem_fraction_static = mem_fraction_static or model_defaults.get("mem-fraction-static") or 0.98
    final_max_running_requests = max_running_requests or model_defaults.get("max-running-requests") or 1
    final_watchdog_timeout = watchdog_timeout or model_defaults.get("watchdog-timeout") or 3000

    if disable_shared_experts_fusion is False and "disable-shared-experts-fusion" in model_defaults:
        disable_shared_experts_fusion = model_defaults["disable-shared-experts-fusion"]

    # Display configuration
    if model_info:
        console.print(f"  Model: [bold]{model_info.name}[/bold]")
    else:
        console.print(f"  Model: [bold]{resolved_model_path.name}[/bold]")

    console.print(f"  Max Total Tokens: [cyan]{final_max_total_tokens}[/cyan]")
    console.print(f"  Tensor Parallel: [cyan]{final_tensor_parallel_size}[/cyan]")
    console.print(f"  Method: [cyan]{final_kt_method}[/cyan]")
    console.print(f"  Hardware: [cyan]{gpu_info}[/cyan]")

    # Step 4: Run tuning
    console.print()
    print_step("Running tuning (this may take a few minutes)")

    # Prepare environment
    env = os.environ.copy()
    env.update(settings.get_env_vars())
    inference_env = settings.get("inference.env", {})
    if isinstance(inference_env, dict):
        env.update({k: str(v) for k, v in inference_env.items()})

    # Prepare config for tuning
    config = {
        "tensor_parallel_size": final_tensor_parallel_size,
        "max_total_tokens": final_max_total_tokens,
        "kt_method": final_kt_method,
        "cpu_threads": final_cpu_threads,
        "numa_nodes": final_numa_nodes,
        "attention_backend": final_attention_backend,
        "kt_gpu_prefill_threshold": final_kt_gpu_prefill_threshold,
        "chunked_prefill_size": final_chunked_prefill_size,
        "mem_fraction_static": final_mem_fraction_static,
        "max_running_requests": final_max_running_requests,
        "watchdog_timeout": final_watchdog_timeout,
        "disable_shared_experts_fusion": disable_shared_experts_fusion,
        "env": env,
    }

    if weights_path:
        config["weights_path"] = Path(weights_path)

    # Run tuna
    try:
        optimal_gpu_experts = run_tuna(
            model_path=resolved_model_path,
            verbose=verbose,
            **config,
        )
    except ValueError as e:
        print_error(f"Tuning failed: {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        raise typer.Exit(0)

    # Step 5: Show results
    console.print()
    console.print(
        Panel.fit(
            f"[bold green]--gpu-experts {optimal_gpu_experts}[/bold green]",
            title="Optimal Configuration",
            border_style="green",
        )
    )
    console.print()

    # Show command to run
    print_success("Tuning completed!")
    console.print()
    console.print("[bold]Run with optimized settings:[/bold]")
    console.print()

    # Build command
    run_cmd_parts = ["kt run", model if not model_path else str(resolved_model_path)]
    run_cmd_parts.append(f"--gpu-experts {optimal_gpu_experts}")

    if max_total_tokens:
        run_cmd_parts.append(f"--max-total-tokens {final_max_total_tokens}")
    if tensor_parallel_size:
        run_cmd_parts.append(f"--tensor-parallel-size {final_tensor_parallel_size}")
    if kt_method:
        run_cmd_parts.append(f"--kt-method {final_kt_method}")
    if weights_path:
        run_cmd_parts.append(f"--weights-path {weights_path}")

    run_cmd = " ".join(run_cmd_parts)
    console.print(f"  [cyan]{run_cmd}[/cyan]")
    console.print()
