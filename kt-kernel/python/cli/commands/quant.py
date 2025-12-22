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
    model: str = typer.Argument(
        ...,
        help="Model name or path to quantize",
    ),
    method: QuantMethod = typer.Option(
        QuantMethod.INT4,
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
    input_type: str = typer.Option(
        "fp8",
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
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompts",
    ),
) -> None:
    """Quantize model weights for CPU inference."""
    settings = get_settings()
    console.print()

    # Resolve input path
    input_path = _resolve_input_path(model, settings)
    if input_path is None:
        print_error(t("quant_input_not_found", path=model))
        raise typer.Exit(1)

    print_info(t("quant_input_path", path=str(input_path)))

    # Resolve output path
    if output is None:
        output = input_path.parent / f"{input_path.name}-{method.value.upper()}"

    print_info(t("quant_output_path", path=str(output)))
    print_info(t("quant_method", method=method.value.upper()))

    # Detect CPU configuration
    cpu = detect_cpu_info()
    final_cpu_threads = cpu_threads or cpu.cores
    final_numa_nodes = numa_nodes or cpu.numa_nodes

    print_info(f"CPU threads: {final_cpu_threads}")
    print_info(f"NUMA nodes: {final_numa_nodes}")

    # Check if output exists
    if output.exists():
        print_warning(f"Output path already exists: {output}")
        if not yes:
            if not confirm("Overwrite?", default=False):
                raise typer.Abort()

    # Confirm
    if not yes:
        console.print()
        console.print("[bold]Quantization Settings:[/bold]")
        console.print(f"  Input: {input_path}")
        console.print(f"  Output: {output}")
        console.print(f"  Method: {method.value.upper()}")
        console.print(f"  Input type: {input_type}")
        console.print()
        print_warning("Quantization may take 30-60 minutes depending on model size.")
        console.print()

        if not confirm(t("prompt_continue")):
            raise typer.Abort()

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
        sys.executable, str(script_path),
        "--input-path", str(input_path),
        "--input-type", input_type,
        "--output", str(output),
        "--quant-method", method.value,
        "--cpuinfer-threads", str(final_cpu_threads),
        "--threadpool-count", str(final_numa_nodes),
    ]

    if no_merge:
        cmd.append("--no-merge-safetensor")

    # Run quantization
    console.print()
    print_step(t("quant_starting"))
    console.print()
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    console.print()

    try:
        process = subprocess.run(cmd)

        if process.returncode == 0:
            console.print()
            print_success(t("quant_complete"))
            console.print()
            console.print(f"  Quantized weights saved to: {output}")
            console.print()
            console.print("  Use with:")
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
        # Try to find in models directory
        models_dir = settings.models_dir
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
