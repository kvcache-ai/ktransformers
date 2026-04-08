"""
Shared model table builders for consistent UI across commands.

Provides reusable table construction functions for displaying models
in kt model list, kt quant, kt run, etc.
"""

from typing import List, Optional, Tuple
from pathlib import Path
from rich.table import Table
from rich.console import Console
import json


def format_model_size(model_path: Path, format_type: str) -> str:
    """Calculate and format model size."""
    from kt_kernel.cli.utils.model_scanner import format_size

    try:
        if format_type == "safetensors":
            files = list(model_path.glob("*.safetensors"))
        elif format_type == "gguf":
            files = list(model_path.glob("*.gguf"))
        else:
            return "[dim]-[/dim]"

        total_size = sum(f.stat().st_size for f in files if f.exists())
        return format_size(total_size)
    except Exception:
        return "[dim]-[/dim]"


def format_repo_info(model) -> str:
    """Format repository information."""
    if model.repo_id:
        repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
        return f"{repo_abbr}:{model.repo_id}"
    return "[dim]-[/dim]"


def format_sha256_status(model, status_map: dict) -> str:
    """Format SHA256 verification status."""
    return status_map.get(model.sha256_status or "not_checked", "[dim]?[/dim]")


def build_moe_gpu_table(
    models: List, status_map: dict, show_index: bool = True, start_index: int = 1
) -> Tuple[Table, List]:
    """
    Build MoE GPU models table.

    Args:
        models: List of MoE GPU model objects
        status_map: SHA256_STATUS_MAP for formatting status
        show_index: Whether to show # column for selection (default: True)
        start_index: Starting index number

    Returns:
        Tuple of (Table object, list of models in display order)
    """
    table = Table(show_header=True, header_style="bold", show_lines=False)

    if show_index:
        table.add_column("#", justify="right", style="cyan", no_wrap=True)

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim", overflow="fold")
    table.add_column("Total", justify="right")
    table.add_column("Exps", justify="center", style="yellow")
    table.add_column("Act", justify="center", style="green")
    table.add_column("Repository", style="dim", overflow="fold")
    table.add_column("SHA256", justify="center")

    displayed_models = []

    for i, model in enumerate(models, start_index):
        displayed_models.append(model)

        # Calculate size
        size_str = format_model_size(Path(model.path), "safetensors")

        # MoE info
        num_experts = str(model.moe_num_experts) if model.moe_num_experts else "[dim]-[/dim]"
        num_active = str(model.moe_num_experts_per_tok) if model.moe_num_experts_per_tok else "[dim]-[/dim]"

        # Repository and SHA256
        repo_str = format_repo_info(model)
        sha256_str = format_sha256_status(model, status_map)

        row = []
        if show_index:
            row.append(str(i))

        row.extend([model.name, model.path, size_str, num_experts, num_active, repo_str, sha256_str])

        table.add_row(*row)

    return table, displayed_models


def build_amx_table(
    models: List,
    status_map: dict = None,  # Kept for API compatibility but not used
    show_index: bool = True,
    start_index: int = 1,
    show_linked_gpus: bool = False,
    gpu_models: Optional[List] = None,
) -> Tuple[Table, List]:
    """
    Build AMX models table.

    Note: AMX models are locally quantized, so no SHA256 verification column.

    Args:
        models: List of AMX model objects
        status_map: (Unused - kept for API compatibility)
        show_index: Whether to show # column for selection (default: True)
        start_index: Starting index number
        show_linked_gpus: Whether to show sub-rows for linked GPU models
        gpu_models: List of GPU models (required if show_linked_gpus=True)

    Returns:
        Tuple of (Table object, list of models in display order)
    """
    table = Table(show_header=True, header_style="bold", show_lines=False)

    if show_index:
        table.add_column("#", justify="right", style="cyan", no_wrap=True)

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim", overflow="fold")
    table.add_column("Total", justify="right")
    table.add_column("Method", justify="center", style="yellow")
    table.add_column("NUMA", justify="center", style="green")
    table.add_column("Source", style="dim", overflow="fold")

    # Build reverse map if needed
    amx_used_by_gpu = {}
    if show_linked_gpus and gpu_models:
        for model in models:
            if model.gpu_model_ids:
                gpu_names = []
                for gpu_id in model.gpu_model_ids:
                    for gpu_model in gpu_models:
                        if gpu_model.id == gpu_id:
                            gpu_names.append(gpu_model.name)
                            break
                if gpu_names:
                    amx_used_by_gpu[model.id] = gpu_names

    displayed_models = []

    for i, model in enumerate(models, start_index):
        displayed_models.append(model)

        # Calculate size
        size_str = format_model_size(Path(model.path), "safetensors")

        # Read metadata from config.json or UserModel fields
        method_from_config = None
        numa_from_config = None
        try:
            config_path = Path(model.path) / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    amx_quant = config.get("amx_quantization", {})
                    if amx_quant.get("converted"):
                        method_from_config = amx_quant.get("method")
                        numa_from_config = amx_quant.get("numa_count")
        except Exception:
            pass

        # Priority: UserModel fields > config.json > ?
        method_display = (
            model.amx_quant_method.upper()
            if model.amx_quant_method
            else method_from_config.upper() if method_from_config else "[dim]?[/dim]"
        )
        numa_display = (
            str(model.amx_numa_nodes)
            if model.amx_numa_nodes
            else str(numa_from_config) if numa_from_config else "[dim]?[/dim]"
        )
        source_display = model.amx_source_model or "[dim]-[/dim]"

        row = []
        if show_index:
            row.append(str(i))

        row.extend([model.name, model.path, size_str, method_display, numa_display, source_display])

        table.add_row(*row)

        # Add sub-row showing linked GPUs
        if show_linked_gpus and model.id in amx_used_by_gpu:
            gpu_list = amx_used_by_gpu[model.id]
            gpu_names_str = ", ".join([f"[dim]{name}[/dim]" for name in gpu_list])
            sub_row = []
            if show_index:
                sub_row.append("")
            sub_row.extend([f"  [dim]↳ GPU: {gpu_names_str}[/dim]", "", "", "", "", ""])
            table.add_row(*sub_row, style="dim")

    return table, displayed_models


def build_gguf_table(
    models: List, status_map: dict, show_index: bool = True, start_index: int = 1
) -> Tuple[Table, List]:
    """
    Build GGUF models table.

    Args:
        models: List of GGUF model objects
        status_map: SHA256_STATUS_MAP for formatting status
        show_index: Whether to show # column for selection (default: True)
        start_index: Starting index number

    Returns:
        Tuple of (Table object, list of models in display order)
    """
    table = Table(show_header=True, header_style="bold", show_lines=False)

    if show_index:
        table.add_column("#", justify="right", style="cyan", no_wrap=True)

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim", overflow="fold")
    table.add_column("Total", justify="right")
    table.add_column("Repository", style="dim", overflow="fold")
    table.add_column("SHA256", justify="center")

    displayed_models = []

    for i, model in enumerate(models, start_index):
        displayed_models.append(model)

        # Calculate size
        size_str = format_model_size(Path(model.path), "gguf")

        # Repository and SHA256
        repo_str = format_repo_info(model)
        sha256_str = format_sha256_status(model, status_map)

        row = []
        if show_index:
            row.append(str(i))

        row.extend([model.name, model.path, size_str, repo_str, sha256_str])

        table.add_row(*row)

    return table, displayed_models
