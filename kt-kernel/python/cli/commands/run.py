"""
Run command for kt-cli.

Starts the model inference server using SGLang + kt-kernel.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

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
from kt_kernel.cli.utils.model_registry import MODEL_COMPUTE_FUNCTIONS, ModelInfo, get_registry


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
    # CPU/GPU configuration
    gpu_experts: Optional[int] = typer.Option(
        None,
        "--gpu-experts",
        help="Number of GPU experts per layer",
    ),
    cpu_threads: Optional[int] = typer.Option(
        None,
        "--cpu-threads",
        help="Number of CPU inference threads (kt-cpuinfer, defaults to 80% of CPU cores)",
    ),
    numa_nodes: Optional[int] = typer.Option(
        None,
        "--numa-nodes",
        help="Number of NUMA nodes",
    ),
    tensor_parallel_size: Optional[int] = typer.Option(
        None,
        "--tensor-parallel-size",
        "--tp",
        help="Tensor parallel size (number of GPUs)",
    ),
    # Model paths
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
    # KT-kernel options
    kt_method: Optional[str] = typer.Option(
        None,
        "--kt-method",
        help="KT quantization method (AMXINT4, RAWFP8, etc.)",
    ),
    kt_gpu_prefill_token_threshold: Optional[int] = typer.Option(
        None,
        "--kt-gpu-prefill-threshold",
        help="GPU prefill token threshold for kt-kernel",
    ),
    # SGLang options
    attention_backend: Optional[str] = typer.Option(
        None,
        "--attention-backend",
        help="Attention backend (triton, flashinfer)",
    ),
    max_total_tokens: Optional[int] = typer.Option(
        None,
        "--max-total-tokens",
        help="Maximum total tokens",
    ),
    max_running_requests: Optional[int] = typer.Option(
        None,
        "--max-running-requests",
        help="Maximum running requests",
    ),
    chunked_prefill_size: Optional[int] = typer.Option(
        None,
        "--chunked-prefill-size",
        help="Chunked prefill size",
    ),
    mem_fraction_static: Optional[float] = typer.Option(
        None,
        "--mem-fraction-static",
        help="Memory fraction for static allocation",
    ),
    watchdog_timeout: Optional[int] = typer.Option(
        None,
        "--watchdog-timeout",
        help="Watchdog timeout in seconds",
    ),
    served_model_name: Optional[str] = typer.Option(
        None,
        "--served-model-name",
        help="Custom model name for API responses",
    ),
    # Performance flags
    disable_shared_experts_fusion: Optional[bool] = typer.Option(
        None,
        "--disable-shared-experts-fusion/--enable-shared-experts-fusion",
        help="Disable/enable shared experts fusion",
    ),
    # Other options
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

        # Try to infer model type from path to use default configurations
        # Check directory name against known models
        dir_name = resolved_model_path.name.lower()
        for registered_model in registry.list_all():
            # Check if directory name matches model name or aliases
            if dir_name == registered_model.name.lower():
                model_info = registered_model
                print_info(f"Detected model type: {registered_model.name}")
                break
            for alias in registered_model.aliases:
                if dir_name == alias.lower() or alias.lower() in dir_name:
                    model_info = registered_model
                    print_info(f"Detected model type: {registered_model.name}")
                    break
            if model_info:
                break

        # Also check HuggingFace repo format (org--model)
        if not model_info:
            for registered_model in registry.list_all():
                repo_slug = registered_model.hf_repo.replace("/", "--").lower()
                if repo_slug in dir_name or dir_name in repo_slug:
                    model_info = registered_model
                    print_info(f"Detected model type: {registered_model.name}")
                    break

        if not model_info:
            print_warning("Could not detect model type from path. Using default parameters.")
            console.print("  [dim]Tip: Use model name (e.g., 'kt run m2') to apply optimized configurations[/dim]")
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
                console.print(
                    f"  Download with: kt download {model_info.aliases[0] if model_info.aliases else model_info.name}"
                )
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
    # Resolve all parameters (CLI > model defaults > config > auto-detect)
    final_host = host or settings.get("server.host", "0.0.0.0")
    final_port = port or settings.get("server.port", 30000)

    # Get defaults from model info if available
    model_defaults = model_info.default_params if model_info else {}

    # Determine tensor parallel size first (needed for GPU expert calculation)
    # Priority: CLI > model defaults > config > auto-detect (with model constraints)

    # Check if explicitly specified by user or configuration
    explicitly_specified = (
        tensor_parallel_size  # CLI argument (highest priority)
        or model_defaults.get("tensor-parallel-size")  # Model defaults
        or settings.get("inference.tensor_parallel_size")  # Config file
    )

    if explicitly_specified:
        # Use explicitly specified value
        requested_tensor_parallel_size = explicitly_specified
    else:
        # Auto-detect from GPUs, considering model's max constraint
        detected_gpu_count = len(gpus) if gpus else 1
        if model_info and model_info.max_tensor_parallel_size is not None:
            # Automatically limit to model's maximum to use as many GPUs as possible
            requested_tensor_parallel_size = min(detected_gpu_count, model_info.max_tensor_parallel_size)
        else:
            requested_tensor_parallel_size = detected_gpu_count

    # Apply model's max_tensor_parallel_size constraint if explicitly specified value exceeds it
    final_tensor_parallel_size = requested_tensor_parallel_size
    if model_info and model_info.max_tensor_parallel_size is not None:
        if requested_tensor_parallel_size > model_info.max_tensor_parallel_size:
            console.print()
            print_warning(
                f"Model {model_info.name} only supports up to {model_info.max_tensor_parallel_size}-way "
                f"tensor parallelism, but {requested_tensor_parallel_size} was requested. "
                f"Reducing to {model_info.max_tensor_parallel_size}."
            )
            final_tensor_parallel_size = model_info.max_tensor_parallel_size

    # CPU/GPU configuration with smart defaults
    # kt-cpuinfer: default to 80% of total CPU threads (cores * NUMA nodes)
    total_threads = cpu.cores * cpu.numa_nodes
    final_cpu_threads = (
        cpu_threads
        or model_defaults.get("kt-cpuinfer")
        or settings.get("inference.cpu_threads")
        or int(total_threads * 0.8)
    )

    # kt-threadpool-count: default to NUMA node count
    final_numa_nodes = (
        numa_nodes
        or model_defaults.get("kt-threadpool-count")
        or settings.get("inference.numa_nodes")
        or cpu.numa_nodes
    )

    # kt-num-gpu-experts: use model-specific computation if available and not explicitly set
    if gpu_experts is not None:
        # User explicitly set it
        final_gpu_experts = gpu_experts
    elif model_info and model_info.name in MODEL_COMPUTE_FUNCTIONS and gpus:
        # Use model-specific computation function (only if GPUs detected)
        vram_per_gpu = gpus[0].vram_gb
        compute_func = MODEL_COMPUTE_FUNCTIONS[model_info.name]
        final_gpu_experts = compute_func(final_tensor_parallel_size, vram_per_gpu)
        console.print()
        print_info(
            f"Auto-computed kt-num-gpu-experts: {final_gpu_experts} (TP={final_tensor_parallel_size}, VRAM={vram_per_gpu}GB per GPU)"
        )
    else:
        # Fall back to defaults
        final_gpu_experts = model_defaults.get("kt-num-gpu-experts") or settings.get("inference.gpu_experts", 1)

    # KT-kernel options
    final_kt_method = kt_method or model_defaults.get("kt-method") or settings.get("inference.kt_method", "AMXINT4")
    final_kt_gpu_prefill_threshold = (
        kt_gpu_prefill_token_threshold
        or model_defaults.get("kt-gpu-prefill-token-threshold")
        or settings.get("inference.kt_gpu_prefill_token_threshold", 4096)
    )

    # SGLang options
    final_attention_backend = (
        attention_backend
        or model_defaults.get("attention-backend")
        or settings.get("inference.attention_backend", "triton")
    )
    final_max_total_tokens = (
        max_total_tokens or model_defaults.get("max-total-tokens") or settings.get("inference.max_total_tokens", 40000)
    )
    final_max_running_requests = (
        max_running_requests
        or model_defaults.get("max-running-requests")
        or settings.get("inference.max_running_requests", 32)
    )
    final_chunked_prefill_size = (
        chunked_prefill_size
        or model_defaults.get("chunked-prefill-size")
        or settings.get("inference.chunked_prefill_size", 4096)
    )
    final_mem_fraction_static = (
        mem_fraction_static
        or model_defaults.get("mem-fraction-static")
        or settings.get("inference.mem_fraction_static", 0.98)
    )
    final_watchdog_timeout = (
        watchdog_timeout or model_defaults.get("watchdog-timeout") or settings.get("inference.watchdog_timeout", 3000)
    )
    final_served_model_name = (
        served_model_name or model_defaults.get("served-model-name") or settings.get("inference.served_model_name", "")
    )

    # Performance flags
    if disable_shared_experts_fusion is not None:
        final_disable_shared_experts_fusion = disable_shared_experts_fusion
    elif "disable-shared-experts-fusion" in model_defaults:
        final_disable_shared_experts_fusion = model_defaults["disable-shared-experts-fusion"]
    else:
        final_disable_shared_experts_fusion = settings.get("inference.disable_shared_experts_fusion", False)

    # Pass all model default params to handle any extra parameters
    extra_params = model_defaults if model_info else {}

    cmd = _build_sglang_command(
        model_path=resolved_model_path,
        weights_path=resolved_weights_path,
        model_info=model_info,
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
    if model_info:
        console.print(f"  Model: [bold]{model_info.name}[/bold]")
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
        print_error("SGLang not found. Please install with 'kt install inference'.")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        raise typer.Exit(1)


def _find_model_path(model_info: ModelInfo, settings) -> Optional[Path]:
    """Find the model path on disk by searching all configured model paths."""
    model_paths = settings.get_model_paths()

    # Search in all configured model directories
    for models_dir in model_paths:
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
    """Find the quantized weights path on disk by searching all configured paths."""
    model_paths = settings.get_model_paths()
    weights_dir = settings.weights_dir

    # Check common patterns
    base_names = [
        model_info.name,
        model_info.name.lower(),
        model_info.hf_repo.split("/")[-1],
    ]

    suffixes = ["-INT4", "-int4", "_INT4", "_int4", "-quant", "-quantized"]

    # Prepare search directories
    search_dirs = [weights_dir] if weights_dir else []
    search_dirs.extend(model_paths)

    for base in base_names:
        for suffix in suffixes:
            for dir_path in search_dirs:
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
    elif model_info and model_info.type == "moe":
        # MoE model - likely needs kt-kernel for expert offloading
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

    return cmd
