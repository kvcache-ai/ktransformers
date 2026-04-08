"""
Quant command for kt-cli.

Quantizes model weights for CPU inference.
"""

import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    create_progress,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
)
from kt_kernel.cli.utils.environment import detect_cpu_info


class QuantMethod(str, Enum):
    """Quantization method."""

    INT4 = "int4"
    INT8 = "int8"


def quant(
    model: Optional[str] = typer.Argument(
        None,
        help="Model name or path to quantize",
    ),
    method: Optional[QuantMethod] = typer.Option(
        None,
        "--method",
        "-m",
        help="Quantization method",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for quantized weights",
    ),
    input_type: Optional[str] = typer.Option(
        None,
        "--input-type",
        "-i",
        help="Input weight type (fp8, fp16, bf16)",
    ),
    cpu_threads: Optional[int] = typer.Option(
        None,
        "--cpu-threads",
        help="Number of CPU threads for quantization",
    ),
    numa_nodes: Optional[int] = typer.Option(
        None,
        "--numa-nodes",
        help="Number of NUMA nodes",
    ),
    no_merge: bool = typer.Option(
        False,
        "--no-merge",
        help="Don't merge safetensor files",
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="Use GPU for conversion (faster)",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Quantize model weights for CPU inference.

    If no model is specified, interactive mode will be activated.
    """
    settings = get_settings()

    # Check if we should use interactive mode
    # Interactive mode triggers when: no model, or missing critical parameters
    needs_interactive = model is None or method is None or cpu_threads is None or numa_nodes is None
    is_interactive = False

    if needs_interactive and sys.stdin.isatty():
        # Use interactive configuration (includes verification in Step 1.5)
        from kt_kernel.cli.utils.quant_interactive import interactive_quant_config

        console.print()
        console.print(f"[bold cyan]═══ {t('quant_interactive_title')} ═══[/bold cyan]")
        console.print()
        console.print(f"[yellow]{t('quant_new_model_notice')}[/yellow]")
        console.print()

        config = interactive_quant_config()
        if config is None:
            # User cancelled
            raise typer.Exit(0)

        # Extract configuration
        model_obj = config["model"]
        model = model_obj.id
        input_path = Path(model_obj.path)
        method = QuantMethod(config["method"])
        input_type = config["input_type"]
        cpu_threads = config["cpu_threads"]
        numa_nodes = config["numa_nodes"]
        output = config["output_path"]
        gpu = config["use_gpu"]
        is_interactive = True

        console.print()
        print_success(t("quant_config_complete"))
        console.print()
    else:
        # Non-interactive mode - require model parameter
        if model is None:
            print_error("Model argument is required in non-interactive mode")
            console.print()
            console.print("Usage: kt quant <model>")
            console.print("   Or: kt quant  (for interactive mode)")
            raise typer.Exit(1)

        # Set defaults for optional parameters
        method = method or QuantMethod.INT4
        input_type = input_type or "fp8"

        console.print()

        # Resolve input path
        input_path = _resolve_input_path(model, settings)
        if input_path is None:
            print_error(t("quant_input_not_found", path=model))
            raise typer.Exit(1)

        # Pre-quantization verification (only in non-interactive mode)
        # Interactive mode already did verification in interactive_quant_config()
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
        from kt_kernel.cli.utils.model_verifier import pre_operation_verification

        user_registry = UserModelRegistry()
        user_model_obj = user_registry.find_by_path(str(input_path))

        if user_model_obj and user_model_obj.format == "safetensors":
            pre_operation_verification(user_model_obj, user_registry, operation_name="quantizing")

    # Get user model info for both modes (needed later for registering quantized model)
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    user_registry = UserModelRegistry()
    user_model_obj = user_registry.find_by_path(str(input_path))

    # Validate that it's a MoE model (not AMX or GGUF)
    from kt_kernel.cli.commands.model import is_amx_weights

    # Check if it's AMX (already quantized)
    is_amx, _ = is_amx_weights(str(input_path))
    if is_amx:
        print_error("Cannot quantize AMX models (already quantized)")
        console.print()
        console.print(f"  The model at {input_path} is already in AMX format.")
        raise typer.Exit(1)

    # Check if it's a MoE model
    from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

    moe_result = None  # Store for later use when registering quantized model
    try:
        moe_result = analyze_moe_model(str(input_path), use_cache=True)
        if not moe_result or not moe_result.get("is_moe"):
            print_error("Only MoE models can be quantized to AMX format")
            console.print()
            console.print(f"  The model at {input_path} is not a MoE model.")
            console.print("  AMX quantization is designed for MoE models (e.g., DeepSeek-V3).")
            raise typer.Exit(1)
    except Exception as e:
        print_warning(f"Could not detect MoE information: {e}")
        console.print()
        if not yes:
            if not confirm("Continue quantization anyway?", default=False):
                raise typer.Exit(1)

    # Detect CPU configuration and resolve output path (only needed in non-interactive mode)
    if not is_interactive:
        print_info(t("quant_input_path", path=str(input_path)))

        # Detect CPU configuration (needed for output path)
        cpu = detect_cpu_info()
        final_cpu_threads = cpu_threads or cpu.cores
        final_numa_nodes = numa_nodes or cpu.numa_nodes

        # Resolve output path
        if output is None:
            # Priority: paths.weights > paths.models[0] > model's parent directory
            weights_dir = settings.weights_dir

            if weights_dir and weights_dir.exists():
                # Use configured weights directory (highest priority)
                output = weights_dir / f"{input_path.name}-AMX{method.value.upper()}-NUMA{final_numa_nodes}"
            else:
                # Use first model storage path
                model_paths = settings.get_model_paths()
                if model_paths and model_paths[0].exists():
                    output = model_paths[0] / f"{input_path.name}-AMX{method.value.upper()}-NUMA{final_numa_nodes}"
                else:
                    # Fallback to model's parent directory
                    output = input_path.parent / f"{input_path.name}-AMX{method.value.upper()}-NUMA{final_numa_nodes}"

        print_info(t("quant_output_path", path=str(output)))
        print_info(t("quant_method", method=method.value.upper()))
        print_info(t("quant_cpu_threads", threads=final_cpu_threads))
        print_info(t("quant_numa_nodes", nodes=final_numa_nodes))

        # Calculate space requirements
        console.print()
        console.print(f"[bold cyan]{t('quant_disk_analysis')}[/bold cyan]")
        console.print()

        # Calculate source model size
        try:
            total_bytes = sum(f.stat().st_size for f in input_path.glob("*.safetensors") if f.is_file())
            source_size_gb = total_bytes / (1024**3)
        except Exception:
            source_size_gb = 0.0

        # Estimate quantized size
        input_bits = {"fp8": 8, "fp16": 16, "bf16": 16}
        quant_bits = {"int4": 4, "int8": 8}
        input_bit = input_bits.get(input_type, 16)
        quant_bit = quant_bits.get(method.value, 4)
        ratio = quant_bit / input_bit
        estimated_size_gb = source_size_gb * ratio

        # Check available space
        import shutil

        try:
            check_path = output.parent if not output.exists() else output
            while not check_path.exists() and check_path != check_path.parent:
                check_path = check_path.parent
            stat = shutil.disk_usage(check_path)
            available_gb = stat.free / (1024**3)
        except Exception:
            available_gb = 0.0

        is_sufficient = available_gb >= (estimated_size_gb * 1.2)

        console.print(f"  {t('quant_source_size'):<26} {source_size_gb:.2f} GB")
        console.print(f"  {t('quant_estimated_size'):<26} {estimated_size_gb:.2f} GB")
        console.print(f"  {t('quant_available_space'):<26} {available_gb:.2f} GB")
        console.print()

        if not is_sufficient:
            required_with_buffer = estimated_size_gb * 1.2
            print_warning(t("quant_insufficient_space"))
            console.print()
            console.print(f"  {t('quant_required_space'):<26} {required_with_buffer:.2f} GB")
            console.print(f"  {t('quant_available_space'):<26} {available_gb:.2f} GB")
            console.print(f"  {t('quant_shortage'):<26} {required_with_buffer - available_gb:.2f} GB")
            console.print()
            console.print(f"  {t('quant_may_fail')}")
            console.print()

            if not yes:
                if not confirm(t("quant_continue_anyway"), default=False):
                    raise typer.Abort()
            console.print()

        # Check if output exists and generate unique name
        if output.exists():
            print_warning(t("quant_output_exists", path=str(output)))
            console.print()

            # Generate unique name by adding suffix
            original_name = output.name
            parent_dir = output.parent
            counter = 2

            while output.exists():
                new_name = f"{original_name}-{counter}"
                output = parent_dir / new_name
                counter += 1

            print_success(t("quant_using_unique", path=str(output)))
            console.print()

        # Confirm (only show if not using --yes flag)
        if not yes:
            console.print()
            print_warning(t("quant_time_warning"))
            console.print()

            if not confirm(t("prompt_continue")):
                raise typer.Abort()
    else:
        # Interactive mode: cpu_threads and numa_nodes already set
        final_cpu_threads = cpu_threads
        final_numa_nodes = numa_nodes

    # Find conversion script
    kt_kernel_path = _find_kt_kernel_path()
    if kt_kernel_path is None:
        print_error("kt-kernel not found. Install with: kt install inference")
        raise typer.Exit(1)

    script_path = kt_kernel_path / "scripts" / "convert_cpu_weights.py"
    if not script_path.exists():
        print_error(f"Conversion script not found: {script_path}")
        raise typer.Exit(1)

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--input-path",
        str(input_path),
        "--input-type",
        input_type,
        "--output",
        str(output),
        "--quant-method",
        method.value,
        "--cpuinfer-threads",
        str(final_cpu_threads),
        "--threadpool-count",
        str(final_numa_nodes),
    ]

    if no_merge:
        cmd.append("--no-merge-safetensor")

    if gpu:
        cmd.append("--gpu")

    # Run quantization
    console.print()
    print_step(t("quant_starting"))
    console.print()
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    console.print()
    console.print("[dim]" + "=" * 80 + "[/dim]")
    console.print()

    try:
        # Run with real-time stdout/stderr output
        import os
        import time

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Disable Python output buffering

        # Record start time
        start_time = time.time()

        process = subprocess.run(
            cmd,
            stdout=None,  # Inherit parent's stdout (real-time output)
            stderr=None,  # Inherit parent's stderr (real-time output)
            env=env,
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        console.print()
        console.print("[dim]" + "=" * 80 + "[/dim]")
        console.print()

        if process.returncode == 0:
            print_success(t("quant_complete"))
            console.print()

            # Display elapsed time
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            console.print(f"  [cyan]{t('quant_time_elapsed')} {time_str}[/cyan]")
            console.print()
            console.print(f"  Quantized weights saved to: {output}")
            console.print()

            # Auto-register the quantized model
            try:
                from kt_kernel.cli.utils.user_model_registry import UserModel

                # Generate model name from output path
                base_name = output.name
                suggested_name = user_registry.suggest_name(base_name)

                # Determine MoE information and source model name
                if user_model_obj:
                    is_moe_val = user_model_obj.is_moe
                    num_experts = user_model_obj.moe_num_experts
                    num_active = user_model_obj.moe_num_experts_per_tok
                    repo_type_val = user_model_obj.repo_type
                    repo_id_val = user_model_obj.repo_id
                    source_model_name = user_model_obj.name  # Store source model name
                elif moe_result:
                    is_moe_val = moe_result.get("is_moe", True)
                    num_experts = moe_result.get("num_experts")
                    num_active = moe_result.get("num_experts_per_tok")
                    repo_type_val = None
                    repo_id_val = None
                    source_model_name = input_path.name  # Use folder name as fallback
                else:
                    is_moe_val = None
                    num_experts = None
                    num_active = None
                    repo_type_val = None
                    repo_id_val = None
                    source_model_name = input_path.name  # Use folder name as fallback

                # Create new model entry (AMX format uses "safetensors" format, detected by is_amx_weights())
                new_model = UserModel(
                    name=suggested_name,
                    path=str(output),
                    format="safetensors",  # AMX files are safetensors format
                    repo_type=repo_type_val,
                    repo_id=repo_id_val,
                    sha256_status="not_checked",  # AMX weights don't need verification
                    # Inherit MoE information from source model
                    is_moe=is_moe_val,
                    moe_num_experts=num_experts,
                    moe_num_experts_per_tok=num_active,
                    # AMX quantization metadata
                    amx_source_model=source_model_name,
                    amx_quant_method=method.value,  # "int4" or "int8"
                    amx_numa_nodes=final_numa_nodes,
                )

                user_registry.add_model(new_model)
                console.print()
                print_success(t("quant_registered", name=suggested_name))
                console.print()
                console.print(f"  {t('quant_view_with')} [cyan]kt model list[/cyan]")
                console.print(f"  {t('quant_use_with')}  [cyan]kt run {suggested_name}[/cyan]")
                console.print()
            except Exception as e:
                # Non-fatal error - quantization succeeded but registration failed
                console.print()
                print_warning(t("quant_register_failed", error=str(e)))
                console.print()
                console.print(f"  {t('quant_use_with')}")
                console.print(f"    kt run {model} --weights-path {output}")
                console.print()
        else:
            print_error(f"Quantization failed with exit code {process.returncode}")
            raise typer.Exit(process.returncode)

    except FileNotFoundError as e:
        print_error(f"Failed to run quantization: {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print()
        print_warning("Quantization interrupted.")
        raise typer.Exit(130)


def _resolve_input_path(model: str, settings) -> Optional[Path]:
    """Resolve the input model path."""
    # Check if it's already a path
    path = Path(model)
    if path.exists() and (path / "config.json").exists():
        return path

    # Search in models directory
    from kt_kernel.cli.utils.model_registry import get_registry

    registry = get_registry()
    matches = registry.search(model)

    if matches:
        model_info = matches[0]
        # Try to find in all configured model directories
        model_paths = settings.get_model_paths()

        for models_dir in model_paths:
            possible_paths = [
                models_dir / model_info.name,
                models_dir / model_info.name.lower(),
                models_dir / model_info.hf_repo.split("/")[-1],
            ]

            for p in possible_paths:
                if p.exists() and (p / "config.json").exists():
                    return p

    return None


def _find_kt_kernel_path() -> Optional[Path]:
    """Find the kt-kernel installation path."""
    try:
        import kt_kernel

        return Path(kt_kernel.__file__).parent.parent
    except ImportError:
        pass

    # Check common locations
    possible_paths = [
        Path.home() / "Projects" / "ktransformers" / "kt-kernel",
        Path.cwd().parent / "kt-kernel",
        Path.cwd() / "kt-kernel",
    ]

    for path in possible_paths:
        if path.exists() and (path / "scripts").exists():
            return path

    return None
