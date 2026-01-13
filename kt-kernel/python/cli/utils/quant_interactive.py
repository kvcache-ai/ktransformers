"""
Interactive configuration for kt quant command.

Provides rich, multi-step interactive configuration for model quantization.
"""

from typing import Optional, Dict, Any
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from kt_kernel.cli.i18n import t


console = Console()


def select_model_to_quantize() -> Optional[Any]:
    """Select model to quantize interactively."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.commands.model import is_amx_weights, SHA256_STATUS_MAP
    from kt_kernel.cli.utils.model_table_builder import build_moe_gpu_table

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Filter MoE models only (safetensors, not AMX, is_moe=True)
    quant_models = []
    for model in all_models:
        if model.format == "safetensors":
            # Skip AMX models
            is_amx, _ = is_amx_weights(model.path)
            if is_amx:
                continue

            # Only include MoE models
            if model.is_moe:
                quant_models.append(model)

    if not quant_models:
        console.print(f"[yellow]{t('quant_no_moe_models')}[/yellow]")
        console.print()
        console.print(f"  {t('quant_only_moe')}")
        console.print()
        console.print(f"  {t('quant_add_models', command='kt model scan')}")
        console.print(f"  {t('quant_add_models', command='kt model add <path>')}")
        return None

    # Display models
    console.print()
    console.print(f"[bold green]{t('quant_moe_available')}[/bold green]")
    console.print()

    # Use shared table builder
    table, displayed_models = build_moe_gpu_table(
        models=quant_models, status_map=SHA256_STATUS_MAP, show_index=True, start_index=1
    )

    console.print(table)
    console.print()

    choice = IntPrompt.ask(t("quant_select_model"), default=1, show_choices=False)

    if choice < 1 or choice > len(displayed_models):
        console.print(f"[red]{t('quant_invalid_choice')}[/red]")
        return None

    return displayed_models[choice - 1]


def configure_quantization_method() -> Dict[str, str]:
    """Select quantization method and input type."""
    console.print()
    console.print(Panel(f"[bold cyan]{t('quant_step2_method')}[/bold cyan]", expand=False))
    console.print()

    # Method selection
    console.print(f"[bold]{t('quant_method_label')}[/bold]")
    console.print(f"  [cyan][1][/cyan] {t('quant_int4_desc')}")
    console.print(f"  [cyan][2][/cyan] {t('quant_int8_desc')}")
    console.print()

    method_choice = Prompt.ask(t("quant_select_method"), choices=["1", "2"], default="1")
    method = "int4" if method_choice == "1" else "int8"

    console.print()
    console.print(f"[bold]{t('quant_input_type_label')}[/bold]")
    console.print(f"  [cyan][1][/cyan] {t('quant_fp8_desc')}")
    console.print(f"  [cyan][2][/cyan] {t('quant_fp16_desc')}")
    console.print(f"  [cyan][3][/cyan] {t('quant_bf16_desc')}")
    console.print()

    input_choice = Prompt.ask(t("quant_select_input_type"), choices=["1", "2", "3"], default="1")
    input_type_map = {"1": "fp8", "2": "fp16", "3": "bf16"}
    input_type = input_type_map[input_choice]

    return {"method": method, "input_type": input_type}


def configure_cpu_params(max_cores: int, max_numa: int) -> Dict[str, Any]:
    """Configure CPU parameters."""
    console.print()
    console.print(Panel(f"[bold cyan]{t('quant_step3_cpu')}[/bold cyan]", expand=False))
    console.print()

    def clamp(value: int, min_val: int, max_val: int, default: int) -> int:
        """Clamp value to range or return default if out of bounds."""
        if min_val <= value <= max_val:
            return max(min_val, min(value, max_val))
        return default

    default_threads = int(max_cores * 0.8)
    cpu_threads = IntPrompt.ask(t("quant_cpu_threads_prompt", max=max_cores), default=default_threads)
    cpu_threads = clamp(cpu_threads, 1, max_cores, default_threads)

    numa_nodes = IntPrompt.ask(t("quant_numa_nodes_prompt", max=max_numa), default=max_numa)
    numa_nodes = clamp(numa_nodes, 1, max_numa, max_numa)

    # Ask about GPU usage
    console.print()
    console.print(f"[bold]{t('quant_use_gpu_label')}[/bold]")
    console.print(f"  [dim]{t('quant_gpu_speedup')}[/dim]")
    console.print()
    use_gpu = Confirm.ask(t("quant_enable_gpu"), default=True)

    return {"cpu_threads": cpu_threads, "numa_nodes": numa_nodes, "use_gpu": use_gpu}


def configure_output_path(model: Any, method: str, numa_nodes: int) -> Path:
    """Configure output path for quantized weights."""
    from kt_kernel.cli.config.settings import get_settings

    console.print()
    console.print(Panel(f"[bold cyan]{t('quant_step4_output')}[/bold cyan]", expand=False))
    console.print()

    # Generate default output path
    model_path = Path(model.path)
    method_upper = method.upper()
    settings = get_settings()

    # Priority: paths.weights > paths.models[0] > model's parent directory
    weights_dir = settings.weights_dir
    if weights_dir and weights_dir.exists():
        # Use configured weights directory (highest priority)
        default_output = weights_dir / f"{model_path.name}-AMX{method_upper}-NUMA{numa_nodes}"
    else:
        # Use first model storage path
        model_paths = settings.get_model_paths()
        if model_paths and model_paths[0].exists():
            default_output = model_paths[0] / f"{model_path.name}-AMX{method_upper}-NUMA{numa_nodes}"
        else:
            # Fallback to model's parent directory
            default_output = model_path.parent / f"{model_path.name}-AMX{method_upper}-NUMA{numa_nodes}"

    console.print(f"[dim]{t('quant_default_path')}[/dim]", default_output)
    console.print()

    use_default = Confirm.ask(t("quant_use_default"), default=True)

    if use_default:
        return default_output

    custom_path = Prompt.ask(t("quant_custom_path"), default=str(default_output))

    return Path(custom_path)


def calculate_quantized_size(source_path: Path, input_type: str, quant_method: str) -> tuple[float, float]:
    """
    Calculate source model size and estimated quantized size.

    Args:
        source_path: Path to source model
        input_type: Input type (fp8, fp16, bf16)
        quant_method: Quantization method (int4, int8)

    Returns:
        Tuple of (source_size_gb, estimated_quant_size_gb)
    """
    # Calculate source model size
    try:
        total_bytes = sum(f.stat().st_size for f in source_path.glob("*.safetensors") if f.is_file())
        source_size_gb = total_bytes / (1024**3)
    except Exception:
        return 0.0, 0.0

    # Bits mapping
    input_bits = {"fp8": 8, "fp16": 16, "bf16": 16}
    quant_bits = {"int4": 4, "int8": 8}

    input_bit = input_bits.get(input_type, 16)
    quant_bit = quant_bits.get(quant_method, 4)

    # Estimate: source_size * (quant_bits / input_bits)
    ratio = quant_bit / input_bit
    estimated_size_gb = source_size_gb * ratio

    return source_size_gb, estimated_size_gb


def check_disk_space(output_path: Path, required_size_gb: float) -> tuple[float, bool]:
    """
    Check available disk space at output path.

    Args:
        output_path: Target output path
        required_size_gb: Required space in GB

    Returns:
        Tuple of (available_gb, is_sufficient)
        is_sufficient is True if available >= required * 1.2
    """
    import shutil

    try:
        # Get parent directory that exists
        check_path = output_path.parent if not output_path.exists() else output_path
        while not check_path.exists() and check_path != check_path.parent:
            check_path = check_path.parent

        stat = shutil.disk_usage(check_path)
        available_gb = stat.free / (1024**3)

        # Check if available space >= required * 1.2 (20% buffer)
        is_sufficient = available_gb >= (required_size_gb * 1.2)

        return available_gb, is_sufficient
    except Exception:
        return 0.0, False


def interactive_quant_config() -> Optional[Dict[str, Any]]:
    """
    Interactive configuration for kt quant.

    Returns configuration dict or None if cancelled.
    """
    from kt_kernel.cli.utils.environment import detect_cpu_info

    # Get CPU info
    cpu_info = detect_cpu_info()

    # Step 1: Select model
    model = select_model_to_quantize()
    if not model:
        return None

    # Step 1.5: Pre-quantization verification (optional)
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.utils.model_verifier import pre_operation_verification

    user_registry = UserModelRegistry()
    user_model_obj = user_registry.find_by_path(model.path)

    if user_model_obj and user_model_obj.format == "safetensors":
        pre_operation_verification(user_model_obj, user_registry, operation_name="quantizing")

    # Step 2: Configure quantization method
    quant_config = configure_quantization_method()

    # Step 3: Configure CPU parameters
    cpu_config = configure_cpu_params(cpu_info.cores, cpu_info.numa_nodes)

    # Step 4: Configure output path
    output_path = configure_output_path(model, quant_config["method"], cpu_config["numa_nodes"])

    # Step 4.5: Check if output path already exists and generate unique name
    if output_path.exists():
        console.print()
        console.print(t("quant_output_exists_warn", path=str(output_path)))
        console.print()

        # Generate unique name by adding suffix
        original_name = output_path.name
        parent_dir = output_path.parent
        counter = 2

        while output_path.exists():
            new_name = f"{original_name}-{counter}"
            output_path = parent_dir / new_name
            counter += 1

        console.print(t("quant_using_unique_name", path=str(output_path)))
        console.print()

    # Step 5: Calculate space requirements and check availability
    console.print()
    console.print(Panel(f"[bold cyan]{t('quant_disk_analysis')}[/bold cyan]", expand=False))
    console.print()

    source_size_gb, estimated_size_gb = calculate_quantized_size(
        Path(model.path), quant_config["input_type"], quant_config["method"]
    )

    available_gb, is_sufficient = check_disk_space(output_path, estimated_size_gb)

    console.print(f"  {t('quant_source_size'):<26} [cyan]{source_size_gb:.2f} GB[/cyan]")
    console.print(f"  {t('quant_estimated_size'):<26} [yellow]{estimated_size_gb:.2f} GB[/yellow]")
    console.print(
        f"  {t('quant_available_space'):<26} [{'green' if is_sufficient else 'red'}]{available_gb:.2f} GB[/{'green' if is_sufficient else 'red'}]"
    )
    console.print()

    if not is_sufficient:
        required_with_buffer = estimated_size_gb * 1.2
        console.print(f"[bold red]⚠ {t('quant_insufficient_space')}[/bold red]")
        console.print()
        console.print(f"  {t('quant_required_space'):<26} [yellow]{required_with_buffer:.2f} GB[/yellow]")
        console.print(f"  {t('quant_available_space'):<26} [red]{available_gb:.2f} GB[/red]")
        console.print(f"  {t('quant_shortage'):<26} [red]{required_with_buffer - available_gb:.2f} GB[/red]")
        console.print()
        console.print(f"  {t('quant_may_fail')}")
        console.print()

        if not Confirm.ask(f"[yellow]{t('quant_continue_anyway')}[/yellow]", default=False):
            console.print(f"[yellow]{t('quant_cancelled')}[/yellow]")
            return None
        console.print()

    # Summary and confirmation
    console.print()
    console.print(Panel(f"[bold cyan]{t('quant_config_summary')}[/bold cyan]", expand=False))
    console.print()
    console.print(f"  {t('quant_summary_model'):<15} {model.name}")
    console.print(f"  {t('quant_summary_method'):<15} {quant_config['method'].upper()}")
    console.print(f"  {t('quant_summary_input_type'):<15} {quant_config['input_type'].upper()}")
    console.print(f"  {t('quant_summary_cpu_threads'):<15} {cpu_config['cpu_threads']}")
    console.print(f"  {t('quant_summary_numa'):<15} {cpu_config['numa_nodes']}")
    console.print(f"  {t('quant_summary_gpu'):<15} {t('yes') if cpu_config['use_gpu'] else t('no')}")
    console.print(f"  {t('quant_summary_output'):<15} {output_path}")
    console.print()

    if not Confirm.ask(f"[bold green]{t('quant_start_question')}[/bold green]", default=True):
        console.print(f"[yellow]{t('quant_cancelled')}[/yellow]")
        return None

    return {
        "model": model,
        "method": quant_config["method"],
        "input_type": quant_config["input_type"],
        "cpu_threads": cpu_config["cpu_threads"],
        "numa_nodes": cpu_config["numa_nodes"],
        "use_gpu": cpu_config["use_gpu"],
        "output_path": output_path,
    }
