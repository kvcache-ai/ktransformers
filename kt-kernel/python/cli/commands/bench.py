"""
Bench commands for kt-cli.

Runs benchmarks for performance testing.
"""

import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import (
    console,
    print_error,
    print_info,
    print_step,
    print_success,
)


class BenchType(str, Enum):
    """Benchmark type."""

    INFERENCE = "inference"
    MLA = "mla"
    MOE = "moe"
    LINEAR = "linear"
    ATTENTION = "attention"
    ALL = "all"


def bench(
    type: BenchType = typer.Option(
        BenchType.ALL,
        "--type",
        "-t",
        help="Benchmark type",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to benchmark",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)",
    ),
    iterations: int = typer.Option(
        10,
        "--iterations",
        "-n",
        help="Number of iterations",
    ),
) -> None:
    """Run full benchmark suite."""
    console.print()
    print_step(t("bench_starting"))
    print_info(t("bench_type", type=type.value))
    console.print()

    if type == BenchType.ALL:
        _run_all_benchmarks(model, output, iterations)
    elif type == BenchType.INFERENCE:
        _run_inference_benchmark(model, output, iterations)
    elif type == BenchType.MLA:
        _run_component_benchmark("mla", output, iterations)
    elif type == BenchType.MOE:
        _run_component_benchmark("moe", output, iterations)
    elif type == BenchType.LINEAR:
        _run_component_benchmark("linear", output, iterations)
    elif type == BenchType.ATTENTION:
        _run_component_benchmark("attention", output, iterations)

    console.print()
    print_success(t("bench_complete"))
    if output:
        console.print(f"  Results saved to: {output}")
    console.print()


def microbench(
    component: str = typer.Argument(
        "moe",
        help="Component to benchmark (moe, mla, linear, attention)",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Batch size",
    ),
    seq_len: int = typer.Option(
        1,
        "--seq-len",
        "-s",
        help="Sequence length",
    ),
    iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of iterations",
    ),
    warmup: int = typer.Option(
        10,
        "--warmup",
        "-w",
        help="Warmup iterations",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for results (JSON)",
    ),
) -> None:
    """Run micro-benchmark for specific components."""
    console.print()
    print_step(t("bench_starting"))
    print_info(f"Component: {component}")
    print_info(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print_info(f"Iterations: {iterations} (warmup: {warmup})")
    console.print()

    # Try to find the benchmark script
    kt_kernel_path = _find_kt_kernel_path()

    if kt_kernel_path is None:
        print_error("kt-kernel not found. Install with: kt install inference")
        raise typer.Exit(1)

    bench_dir = kt_kernel_path / "bench"

    # Map component to script
    component_scripts = {
        "moe": "bench_moe.py",
        "mla": "bench_mla.py",
        "linear": "bench_linear.py",
        "attention": "bench_attention.py",
        "mlp": "bench_mlp.py",
    }

    script_name = component_scripts.get(component.lower())
    if script_name is None:
        print_error(f"Unknown component: {component}")
        console.print(f"Available: {', '.join(component_scripts.keys())}")
        raise typer.Exit(1)

    script_path = bench_dir / script_name
    if not script_path.exists():
        print_error(f"Benchmark script not found: {script_path}")
        raise typer.Exit(1)

    # Run benchmark
    cmd = [
        sys.executable, str(script_path),
        "--batch-size", str(batch_size),
        "--seq-len", str(seq_len),
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]

    if output:
        cmd.extend(["--output", str(output)])

    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    console.print()

    try:
        process = subprocess.run(cmd)

        if process.returncode == 0:
            console.print()
            print_success(t("bench_complete"))
            if output:
                console.print(f"  Results saved to: {output}")
        else:
            print_error(f"Benchmark failed with exit code {process.returncode}")
            raise typer.Exit(process.returncode)

    except FileNotFoundError as e:
        print_error(f"Failed to run benchmark: {e}")
        raise typer.Exit(1)


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
        Path("/opt/ktransformers/kt-kernel"),
        Path.cwd() / "kt-kernel",
    ]

    for path in possible_paths:
        if path.exists() and (path / "bench").exists():
            return path

    return None


def _run_all_benchmarks(model: Optional[str], output: Optional[Path], iterations: int) -> None:
    """Run all benchmarks."""
    components = ["moe", "mla", "linear", "attention"]

    for component in components:
        console.print(f"\n[bold]Running {component} benchmark...[/bold]")
        _run_component_benchmark(component, None, iterations)


def _run_inference_benchmark(
    model: Optional[str], output: Optional[Path], iterations: int
) -> None:
    """Run inference benchmark."""
    if model is None:
        print_error("Model required for inference benchmark. Use --model flag.")
        raise typer.Exit(1)

    print_info(f"Running inference benchmark on {model}...")
    console.print()
    console.print("[dim]This will start the server and run test requests.[/dim]")
    console.print()

    # TODO: Implement actual inference benchmarking
    print_error("Inference benchmarking not yet implemented.")


def _run_component_benchmark(
    component: str, output: Optional[Path], iterations: int
) -> None:
    """Run a component benchmark."""
    kt_kernel_path = _find_kt_kernel_path()

    if kt_kernel_path is None:
        print_error("kt-kernel not found.")
        return

    bench_dir = kt_kernel_path / "bench"
    script_map = {
        "moe": "bench_moe.py",
        "mla": "bench_mla.py",
        "linear": "bench_linear.py",
        "attention": "bench_attention.py",
    }

    script_name = script_map.get(component)
    if script_name is None:
        print_error(f"Unknown component: {component}")
        return

    script_path = bench_dir / script_name
    if not script_path.exists():
        print_error(f"Script not found: {script_path}")
        return

    cmd = [sys.executable, str(script_path), "--iterations", str(iterations)]

    try:
        subprocess.run(cmd)
    except Exception as e:
        print_error(f"Benchmark failed: {e}")
