"""
Run command for kt-cli.

Starts the model inference server using SGLang + kt-kernel.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from ktcli.config.settings import get_settings
from ktcli.i18n import t
from ktcli.utils.console import (
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
from ktcli.utils.environment import detect_cpu_info, detect_gpus, detect_ram_gb
from ktcli.utils.model_registry import ModelInfo, get_registry


def run(
    model: str = typer.Argument(
        ...,
        help="Model name or path (e.g., deepseek-v3, qwen3-30b)",
    ),
    host: str = typer.Option(
        None,
        "--host",
        "-H",
        help="Server host address",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Server port",
    ),
    gpu_experts: Optional[int] = typer.Option(
        None,
        "--gpu-experts",
        help="Number of GPU experts per layer",
    ),
    cpu_threads: Optional[int] = typer.Option(
        None,
        "--cpu-threads",
        help="Number of CPU inference threads",
    ),
    numa_nodes: Optional[int] = typer.Option(
        None,
        "--numa-nodes",
        help="Number of NUMA nodes",
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model-path",
        help="Custom model path",
    ),
    weights_path: Optional[Path] = typer.Option(
        None,
        "--weights-path",
        help="Custom quantized weights path",
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize",
        "-q",
        help="Quantize model if weights not found",
    ),
    advanced: bool = typer.Option(
        False,
        "--advanced",
        help="Show advanced options",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show command without executing",
    ),
) -> None:
    """Start model inference server."""
    settings = get_settings()
    registry = get_registry()

    console.print()

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

    model_info = None
    resolved_model_path = model_path

    # Check if model is a path
    if Path(model).exists():
        resolved_model_path = Path(model)
        print_info(t("run_model_path", path=str(resolved_model_path)))
    else:
        # Search in registry
        matches = registry.search(model)

        if not matches:
            print_error(t("run_model_not_found", name=model))
            console.print()
            console.print("Available models:")
            for m in registry.list_all()[:5]:
                console.print(f"  - {m.name} ({', '.join(m.aliases[:2])})")
            raise typer.Exit(1)

        if len(matches) == 1:
            model_info = matches[0]
        else:
            # Multiple matches - prompt user
            console.print()
            print_info(t("run_multiple_matches"))
            choices = [f"{m.name} ({m.hf_repo})" for m in matches]
            selected = prompt_choice(t("run_select_model"), choices)
            idx = choices.index(selected)
            model_info = matches[idx]

        # Find model path
        if model_path is None:
            resolved_model_path = _find_model_path(model_info, settings)
            if resolved_model_path is None:
                print_error(t("run_model_not_found", name=model_info.name))
                console.print()
                console.print(f"  Download with: kt download {model_info.aliases[0] if model_info.aliases else model_info.name}")
                raise typer.Exit(1)

        print_info(t("run_model_path", path=str(resolved_model_path)))

    # Step 3: Check quantized weights
    resolved_weights_path = weights_path
    if resolved_weights_path is None and model_info:
        resolved_weights_path = _find_weights_path(model_info, settings)

    if resolved_weights_path is None or not resolved_weights_path.exists():
        print_warning(t("run_weights_not_found"))

        if quantize:
            console.print()
            print_step(t("run_quantizing"))
            # TODO: Implement quantization
            print_warning("Quantization not yet implemented. Please run 'kt quant' manually.")
            raise typer.Exit(1)
        else:
            console.print()
            if confirm(t("run_quant_prompt")):
                print_warning("Quantization not yet implemented. Please run 'kt quant' manually.")
                raise typer.Exit(1)
            else:
                # Try to continue without quantized weights
                resolved_weights_path = None

    # Step 4: Build command
    final_host = host or settings.get("server.host", "0.0.0.0")
    final_port = port or settings.get("server.port", 30000)
    final_gpu_experts = gpu_experts or settings.get("inference.gpu_experts", 1)
    final_cpu_threads = cpu_threads or settings.get("inference.cpu_threads", 0)
    final_numa_nodes = numa_nodes or settings.get("inference.numa_nodes", 0)

    # Auto-detect if not specified
    if final_cpu_threads == 0:
        final_cpu_threads = cpu.cores
    if final_numa_nodes == 0:
        final_numa_nodes = cpu.numa_nodes

    cmd = _build_sglang_command(
        model_path=resolved_model_path,
        weights_path=resolved_weights_path,
        model_info=model_info,
        host=final_host,
        port=final_port,
        gpu_experts=final_gpu_experts,
        cpu_threads=final_cpu_threads,
        numa_nodes=final_numa_nodes,
        settings=settings,
    )

    # Step 5: Show or execute
    if dry_run:
        console.print()
        console.print("[bold]Command:[/bold]")
        console.print()
        console.print(f"  [dim]{' '.join(cmd)}[/dim]")
        console.print()
        return

    console.print()
    print_step(t("run_starting_server"))
    print_server_info(
        mode="SGLang + kt-kernel",
        host=final_host,
        port=final_port,
        gpu_experts=final_gpu_experts,
        cpu_threads=final_cpu_threads,
    )

    print_api_info(final_host, final_port)

    # Execute
    env = os.environ.copy()
    env.update(settings.get_env_vars())

    try:
        process = subprocess.Popen(cmd, env=env)

        # Handle Ctrl+C gracefully
        def signal_handler(signum, frame):
            process.terminate()
            process.wait()
            console.print()
            print_info("Server stopped.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        process.wait()

    except FileNotFoundError:
        print_error("SGLang not found. Please install with 'kt install inference'.")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        raise typer.Exit(1)


def _find_model_path(model_info: ModelInfo, settings) -> Optional[Path]:
    """Find the model path on disk."""
    models_dir = settings.models_dir

    # Check common path patterns
    possible_paths = [
        models_dir / model_info.name,
        models_dir / model_info.name.lower(),
        models_dir / model_info.name.replace(" ", "-"),
        models_dir / model_info.hf_repo.split("/")[-1],
        models_dir / model_info.hf_repo.replace("/", "--"),
    ]

    # Add alias-based paths
    for alias in model_info.aliases:
        possible_paths.append(models_dir / alias)
        possible_paths.append(models_dir / alias.lower())

    for path in possible_paths:
        if path.exists() and (path / "config.json").exists():
            return path

    return None


def _find_weights_path(model_info: ModelInfo, settings) -> Optional[Path]:
    """Find the quantized weights path on disk."""
    models_dir = settings.models_dir
    weights_dir = settings.weights_dir

    # Check common patterns
    base_names = [
        model_info.name,
        model_info.name.lower(),
        model_info.hf_repo.split("/")[-1],
    ]

    suffixes = ["-INT4", "-int4", "_INT4", "_int4", "-quant", "-quantized"]

    for base in base_names:
        for suffix in suffixes:
            for dir_path in [weights_dir, models_dir] if weights_dir else [models_dir]:
                if dir_path:
                    path = dir_path / f"{base}{suffix}"
                    if path.exists():
                        return path

    return None


def _build_sglang_command(
    model_path: Path,
    weights_path: Optional[Path],
    model_info: Optional[ModelInfo],
    host: str,
    port: int,
    gpu_experts: int,
    cpu_threads: int,
    numa_nodes: int,
    settings,
) -> list[str]:
    """Build the SGLang launch command."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--host", host,
        "--port", str(port),
        "--model", str(model_path),
    ]

    # Add kt-kernel options if weights are available
    if weights_path:
        cmd.extend([
            "--kt-weight-path", str(weights_path),
            "--kt-cpuinfer", str(cpu_threads),
            "--kt-threadpool-count", str(numa_nodes),
            "--kt-num-gpu-experts", str(gpu_experts),
        ])

    # Add model-specific defaults
    if model_info and model_info.default_params:
        for key, value in model_info.default_params.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])

    # Add common options
    cmd.extend([
        "--attention-backend", settings.get("inference.attention_backend", "triton"),
        "--trust-remote-code",
        "--mem-fraction-static", str(settings.get("inference.mem_fraction_static", 0.98)),
        "--chunked-prefill-size", str(settings.get("inference.chunked_prefill_size", 4096)),
        "--max-running-requests", str(settings.get("inference.max_running_requests", 32)),
        "--max-total-tokens", str(settings.get("inference.max_total_tokens", 40000)),
        "--enable-mixed-chunk",
        "--tensor-parallel-size", "1",
        "--enable-p2p-check",
    ])

    # Add extra args from settings
    extra_args = settings.get("advanced.sglang_args", [])
    if extra_args:
        cmd.extend(extra_args)

    return cmd
