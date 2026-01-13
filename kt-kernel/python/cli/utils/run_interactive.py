"""
Interactive configuration for kt run command.

Provides rich, multi-step interactive configuration for running models.
"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich import box
import torch


console = Console()


def get_gpu_info() -> List[Dict[str, Any]]:
    """Get real-time GPU information with free VRAM."""
    from kt_kernel.cli.utils.environment import detect_gpus

    gpus = detect_gpus()
    gpu_info_list = []

    for i, gpu in enumerate(gpus):
        total_vram_gb = gpu.vram_gb
        free_vram_gb = gpu.vram_gb  # Default fallback

        # Try to get real-time free VRAM
        if torch.cuda.is_available() and i < torch.cuda.device_count():
            try:
                free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info(i)
                free_vram_gb = free_vram_bytes / (1024**3)
                total_vram_gb = total_vram_bytes / (1024**3)
            except Exception:
                pass  # Use fallback values

        gpu_info_list.append(
            {
                "id": i,
                "name": gpu.name,
                "total_vram_gb": total_vram_gb,
                "free_vram_gb": free_vram_gb,
            }
        )

    return gpu_info_list


def select_gpu_model() -> Optional[Any]:
    """Select GPU model interactively with full model info table."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.commands.model import is_amx_weights, SHA256_STATUS_MAP
    from kt_kernel.cli.utils.model_table_builder import build_moe_gpu_table

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Filter GPU models (safetensors, not AMX, MoE only)
    gpu_models = []
    for model in all_models:
        if model.format == "safetensors":
            is_amx, _ = is_amx_weights(model.path)
            if not is_amx and model.is_moe:
                gpu_models.append(model)

    if not gpu_models:
        console.print("[yellow]No MoE GPU models found. Please add models first with:[/yellow]")
        console.print("  kt model scan")
        console.print("  Or download with: kt model download")
        return None

    # Display models in a table (same format as kt model list)
    console.print()
    console.print(Panel("[bold cyan]Step 1: Select MoE GPU Model[/bold cyan]", expand=False))
    console.print()

    # Use shared table builder
    table, displayed_models = build_moe_gpu_table(
        models=gpu_models, status_map=SHA256_STATUS_MAP, show_index=True, start_index=1
    )

    console.print(table)
    console.print()

    # Prompt for selection
    choice = IntPrompt.ask("Select GPU model", default=1, show_choices=False)

    if choice < 1 or choice > len(displayed_models):
        console.print("[red]Invalid choice[/red]")
        return None

    return displayed_models[choice - 1]


def select_cpu_model(gpu_model: Any) -> Tuple[Optional[str], Optional[int]]:
    """Select CPU model interactively with separate tables for AMX and GGUF.

    Returns:
        Tuple of (cpu_model_id, numa_nodes_selected)
        numa_nodes_selected is None if user didn't select an AMX model
    """
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.commands.model import is_amx_weights, SHA256_STATUS_MAP
    from kt_kernel.cli.utils.model_table_builder import build_amx_table, build_gguf_table

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Get linked CPU model IDs from GPU model
    linked_ids = set(gpu_model.gpu_model_ids or [])

    # Filter and categorize CPU models (show all, don't filter by NUMA)
    amx_models = []
    gguf_models = []

    for model in all_models:
        if model.format == "gguf":
            gguf_models.append(model)
        elif model.format == "safetensors":
            is_amx, _ = is_amx_weights(model.path)
            if is_amx:
                amx_models.append(model)

    if not amx_models and not gguf_models:
        console.print("[yellow]No CPU models found. You can quantize a model with 'kt quant'[/yellow]")
        return None, None

    # Build list of all CPU models with index
    all_cpu_models = []

    console.print()
    console.print(Panel("[bold cyan]Step 2: Select CPU Model[/bold cyan]", expand=False))
    console.print()

    # Ask for NUMA nodes if there are AMX models (for filtering)
    numa_nodes_for_filter = None
    if amx_models:
        from kt_kernel.cli.utils.environment import detect_cpu_info

        cpu_info = detect_cpu_info()

        console.print("[dim]AMX models are quantized for specific NUMA configurations.[/dim]")
        console.print()
        numa_nodes_for_filter = IntPrompt.ask(f"NUMA Nodes (1 to {cpu_info.numa_nodes})", default=cpu_info.numa_nodes)
        console.print()

        # Filter AMX models by NUMA
        filtered_amx = []
        for model in amx_models:
            is_amx, numa_count = is_amx_weights(model.path)
            if numa_count == numa_nodes_for_filter:
                filtered_amx.append(model)
        amx_models = filtered_amx

        if not amx_models and not gguf_models:
            console.print(
                f"[yellow]No AMX models found for NUMA={numa_nodes_for_filter}. Showing GGUF models only.[/yellow]"
            )
            console.print()

    # Table 1: AMX Models
    if amx_models:
        console.print("[bold magenta]AMX Models (CPU)[/bold magenta]")
        table, amx_displayed = build_amx_table(
            models=amx_models,
            status_map=SHA256_STATUS_MAP,
            show_index=True,
            start_index=len(all_cpu_models) + 1,
            show_linked_gpus=False,
        )
        all_cpu_models.extend(amx_displayed)
        console.print(table)
        console.print()

    # Table 2: GGUF Models
    if gguf_models:
        console.print("[bold yellow]GGUF Models (Llamafile)[/bold yellow]")
        table, gguf_displayed = build_gguf_table(
            models=gguf_models, status_map=SHA256_STATUS_MAP, show_index=True, start_index=len(all_cpu_models) + 1
        )
        all_cpu_models.extend(gguf_displayed)
        console.print(table)
        console.print()

    # Find paired/recommended models (AMX models quantized from selected GPU model)
    recommended_models = []
    for model in amx_models:
        # Check if this AMX model was quantized from the selected GPU model
        if model.amx_source_model == gpu_model.name:
            recommended_models.append(model)
        # Also check linked IDs
        elif model.id in linked_ids:
            recommended_models.append(model)

    # Display recommendation if any
    if recommended_models:
        console.print("[bold green]Recommended (paired with selected GPU model):[/bold green]")
        for model in recommended_models:
            # Find index in all_cpu_models
            model_idx = all_cpu_models.index(model) + 1
            console.print(f"  [green]→[/green] #{model_idx}: {model.name}")
        console.print()

    # Prompt for selection
    choice = IntPrompt.ask(
        "Select CPU model",
        default=1 if not recommended_models else all_cpu_models.index(recommended_models[0]) + 1,
        show_choices=False,
    )

    if choice < 1 or choice > len(all_cpu_models):
        console.print("[red]Invalid choice[/red]")
        return None, None

    return all_cpu_models[choice - 1].id, numa_nodes_for_filter


def select_gpus() -> Tuple[List[int], int]:
    """Select GPUs interactively. Returns (selected_gpu_ids, tp_size)."""
    gpu_info_list = get_gpu_info()

    if not gpu_info_list:
        console.print("[red]No GPUs detected[/red]")
        return [], 0

    console.print()
    console.print(Panel("[bold cyan]Step 3: Select GPUs[/bold cyan]", expand=False))
    console.print()
    console.print("[dim]TP (Tensor Parallel) size must be a power of 2 (1, 2, 4, 8, ...)[/dim]")
    console.print()

    # Display GPUs
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Free VRAM", justify="right", style="green")
    table.add_column("Total VRAM", justify="right", style="dim")

    for gpu in gpu_info_list:
        table.add_row(str(gpu["id"]), gpu["name"], f"{gpu['free_vram_gb']:.1f} GB", f"{gpu['total_vram_gb']:.1f} GB")

    console.print(table)
    console.print()

    # Prompt for GPU selection
    console.print("Enter GPU IDs separated by commas (e.g., 0,1,2,3)")
    console.print(f"  Or press Enter to use all {len(gpu_info_list)} GPUs")
    console.print()

    gpu_input = Prompt.ask("GPU IDs", default=",".join(str(i) for i in range(len(gpu_info_list))))

    # Parse GPU IDs
    try:
        selected_gpu_ids = [int(x.strip()) for x in gpu_input.split(",")]
    except ValueError:
        console.print("[red]Invalid GPU IDs[/red]")
        return [], 0

    # Validate GPU IDs
    for gpu_id in selected_gpu_ids:
        if gpu_id < 0 or gpu_id >= len(gpu_info_list):
            console.print(f"[red]Invalid GPU ID: {gpu_id}[/red]")
            return [], 0

    tp_size = len(selected_gpu_ids)

    # Validate TP is power of 2
    if tp_size & (tp_size - 1) != 0:
        console.print(f"[red]TP size ({tp_size}) must be a power of 2 (1, 2, 4, 8, ...)[/red]")
        return [], 0

    console.print()
    console.print(f"[green]✓[/green] Selected {tp_size} GPU(s): {selected_gpu_ids}")

    return selected_gpu_ids, tp_size


def configure_parameters(
    gpu_model: Any, max_cpu_cores: int, max_numa_nodes: int, num_experts: int, default_numa_nodes: Optional[int] = None
) -> Dict[str, Any]:
    """Configure run parameters interactively.

    Args:
        gpu_model: Selected GPU model
        max_cpu_cores: Maximum CPU cores available
        max_numa_nodes: Maximum NUMA nodes available
        num_experts: Number of experts in the model
        default_numa_nodes: Default NUMA value from CPU model selection (if AMX was chosen)
    """
    console.print()
    console.print(Panel("[bold cyan]Step 4: Configure Parameters[/bold cyan]", expand=False))
    console.print()

    def clamp(value: int, min_val: int, max_val: int, default: int) -> int:
        """Clamp value to range or return default if out of bounds."""
        if min_val <= value <= max_val:
            return max(min_val, min(value, max_val))
        return default

    # GPU Experts
    default_experts = min(8, num_experts)
    gpu_experts = IntPrompt.ask(f"GPU Experts (0 to {num_experts})", default=default_experts)
    gpu_experts = clamp(gpu_experts, 0, num_experts, default_experts)

    # Total Tokens (KV Cache)
    total_tokens = IntPrompt.ask("Total Tokens for KV Cache (1 to 10000)", default=4096)
    total_tokens = clamp(total_tokens, 1, 10000, 4096)

    # CPU Threads
    default_threads = int(max_cpu_cores * 0.8)
    cpu_threads = IntPrompt.ask(f"CPU Threads (1 to {max_cpu_cores})", default=default_threads)
    cpu_threads = clamp(cpu_threads, 1, max_cpu_cores, default_threads)

    # NUMA Nodes (use default from CPU model selection if available)
    numa_default = default_numa_nodes if default_numa_nodes is not None else max_numa_nodes
    if default_numa_nodes is not None:
        console.print(f"[dim](Using NUMA={default_numa_nodes} from selected AMX model)[/dim]")
    numa_nodes = IntPrompt.ask(f"NUMA Nodes (1 to {max_numa_nodes})", default=numa_default)
    numa_nodes = clamp(numa_nodes, 1, max_numa_nodes, numa_default)

    return {
        "gpu_experts": gpu_experts,
        "total_tokens": total_tokens,
        "cpu_threads": cpu_threads,
        "numa_nodes": numa_nodes,
    }


def calculate_and_display_vram(
    gpu_model: Any, moe_result: Dict[str, Any], params: Dict[str, Any], selected_gpus: List[int], tp_size: int
) -> bool:
    """Calculate and display VRAM requirements. Returns True if sufficient."""
    from kt_kernel.cli.utils.kv_cache_calculator import get_kv_size_gb

    console.print()
    console.print(Panel("[bold cyan]VRAM Requirements Analysis[/bold cyan]", expand=False))
    console.print()

    # Calculate KV cache
    try:
        kv_result = get_kv_size_gb(
            model_path=gpu_model.path, max_total_tokens=params["total_tokens"], tp=tp_size, dtype="auto", verbose=False
        )
        kv_cache_gb = kv_result["total_size_gb"]
    except Exception as e:
        console.print(f"[yellow]Warning: Could not calculate KV cache: {e}[/yellow]")
        kv_cache_gb = 0

    # Calculate per-GPU VRAM
    skeleton_per_gpu = moe_result["rest_size_gb"] / tp_size
    moe_per_gpu = params["gpu_experts"] * moe_result["single_expert_size_gb"] / tp_size
    kv_per_gpu = kv_cache_gb  # Already per-GPU from calculator
    total_per_gpu = skeleton_per_gpu + moe_per_gpu + kv_per_gpu

    # Display breakdown
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Component", style="white")
    table.add_column("Per-GPU Size", justify="right", style="yellow")

    table.add_row("Skeleton (sharded)", f"{skeleton_per_gpu:.2f} GB")
    table.add_row(f"MoE ({params['gpu_experts']} experts, sharded)", f"{moe_per_gpu:.2f} GB")
    table.add_row(f"KV Cache ({params['total_tokens']} tokens)", f"{kv_per_gpu:.2f} GB")
    table.add_row("[bold]Total per GPU[/bold]", f"[bold yellow]{total_per_gpu:.2f} GB[/bold yellow]")

    console.print(table)
    console.print()

    # Check each GPU
    gpu_info_list = get_gpu_info()
    gpu_info_map = {gpu["id"]: gpu for gpu in gpu_info_list}

    all_sufficient = True
    for gpu_id in selected_gpus:
        if gpu_id not in gpu_info_map:
            console.print(f"[red]✗ GPU {gpu_id}: Not found[/red]")
            all_sufficient = False
            continue

        gpu = gpu_info_map[gpu_id]
        free_vram = gpu["free_vram_gb"]
        available_vram = free_vram * 0.95  # 95% safety margin

        if available_vram >= total_per_gpu:
            console.print(f"[green]✓ GPU {gpu_id}: {free_vram:.1f} GB free (need {total_per_gpu:.2f} GB)[/green]")
        else:
            console.print(
                f"[red]✗ GPU {gpu_id}: {free_vram:.1f} GB free (need {total_per_gpu:.2f} GB) - INSUFFICIENT[/red]"
            )
            all_sufficient = False

    console.print()
    return all_sufficient


def interactive_run_config() -> Optional[Dict[str, Any]]:
    """
    Interactive configuration for kt run.

    Returns configuration dict or None if cancelled.
    """
    from kt_kernel.cli.utils.environment import detect_cpu_info

    # Get CPU info
    cpu_info = detect_cpu_info()

    # Step 1: Select GPU model
    gpu_model = select_gpu_model()
    if not gpu_model:
        return None

    # Analyze MOE
    from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

    try:
        moe_result = analyze_moe_model(gpu_model.path)
    except Exception as e:
        console.print(f"[red]Error analyzing model: {e}[/red]")
        return None

    num_experts = moe_result.get("num_experts", 0)

    # Step 2: Select CPU model
    cpu_model_id, numa_nodes_selected = select_cpu_model(gpu_model)
    if not cpu_model_id:
        console.print("[yellow]No CPU model selected - continuing without CPU model[/yellow]")
        cpu_model_id = None
        numa_nodes_selected = None

    # Step 3: Select GPUs
    selected_gpus, tp_size = select_gpus()
    if not selected_gpus:
        return None

    # Step 4: Configure parameters
    params = configure_parameters(
        gpu_model, cpu_info.cores, cpu_info.numa_nodes, num_experts, default_numa_nodes=numa_nodes_selected
    )

    # Step 5: Calculate and display VRAM
    vram_ok = calculate_and_display_vram(gpu_model, moe_result, params, selected_gpus, tp_size)

    if not vram_ok:
        console.print()
        if not Confirm.ask("[yellow]VRAM insufficient on some GPUs. Continue anyway?[/yellow]", default=False):
            console.print("[red]Cancelled[/red]")
            return None

    # Final confirmation
    console.print()
    console.print(Panel("[bold cyan]Configuration Summary[/bold cyan]", expand=False))
    console.print()
    console.print(f"  GPU Model:    {gpu_model.name}")
    console.print(f"  CPU Model:    {cpu_model_id or 'None'}")
    console.print(f"  GPUs:         {selected_gpus} (TP={tp_size})")
    console.print(f"  GPU Experts:  {params['gpu_experts']}")
    console.print(f"  Total Tokens: {params['total_tokens']}")
    console.print(f"  CPU Threads:  {params['cpu_threads']}")
    console.print(f"  NUMA Nodes:   {params['numa_nodes']}")
    console.print()

    if not Confirm.ask("[bold green]Start model server?[/bold green]", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return None

    return {
        "gpu_model": gpu_model,
        "cpu_model_id": cpu_model_id,
        "selected_gpus": selected_gpus,
        "tensor_parallel": tp_size,
        **params,
    }
