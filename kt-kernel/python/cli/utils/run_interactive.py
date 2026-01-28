"""
Interactive configuration for kt run command - New Implementation.

Provides step-by-step interactive configuration for running models.
"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box
import torch

from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.input_validators import (
    prompt_int_with_retry,
    prompt_float_with_retry,
    prompt_choice_with_retry,
    prompt_int_list_with_retry,
)


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


def select_model() -> Optional[Any]:
    """Step 1: Select a safetensors MoE model.

    Returns:
        Selected UserModel object or None if cancelled.
    """
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.commands.model import is_amx_weights

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Filter: safetensors models only (exclude AMX and GGUF)
    # Then filter to only show MoE models (matching kt model list behavior)
    moe_models = []
    for model in all_models:
        if model.format == "safetensors" and model.path_exists():
            is_amx, _ = is_amx_weights(model.path)
            if not is_amx:
                # Only include MoE models (is_moe == True)
                # Also include models not yet analyzed (is_moe == None) for backwards compatibility
                if model.is_moe is True or model.is_moe is None:
                    moe_models.append(model)

    if not moe_models:
        console.print(f"[yellow]{t('run_int_no_moe_models')}[/yellow]")
        console.print(f"  {t('run_int_add_models')}")
        console.print(f"  {t('run_int_list_all')}")
        return None

    console.print()
    console.print(Panel(f"[bold cyan]{t('run_int_step1_title')}[/bold cyan]", expand=False))
    console.print()

    # Display models using same format as kt model list
    from kt_kernel.cli.utils.model_scanner import format_size
    from kt_kernel.cli.commands.model import SHA256_STATUS_MAP

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Path", style="dim", overflow="fold")
    table.add_column("Total", justify="right")
    table.add_column("Exps", justify="center", style="yellow")
    table.add_column("Act", justify="center", style="green")
    table.add_column("MoE Size", justify="right", style="cyan")
    table.add_column("Repo", style="dim", overflow="fold")
    table.add_column("SHA256", justify="center")

    for i, model in enumerate(moe_models, 1):
        # Calculate size
        if model.path_exists():
            path_obj = Path(model.path)
            try:
                files = list(path_obj.glob("*.safetensors"))
                total_size = sum(f.stat().st_size for f in files if f.exists())
                size_display = format_size(total_size)
            except:
                size_display = "[dim]-[/dim]"
        else:
            size_display = "[dim]-[/dim]"

        # Format MoE info
        experts = f"[yellow]{model.moe_num_experts}[/yellow]" if model.moe_num_experts else "[dim]-[/dim]"
        active = f"[green]{model.moe_num_experts_per_tok}[/green]" if model.moe_num_experts_per_tok else "[dim]-[/dim]"
        moe_size = f"[cyan]{size_display}[/cyan]" if model.moe_num_experts else "[dim]-[/dim]"

        # Format repo info
        if model.repo_id:
            repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
            repo_display = f"{repo_abbr}:{model.repo_id}"
        else:
            repo_display = "[dim]-[/dim]"

        # Format SHA256 status
        sha256_display = SHA256_STATUS_MAP.get(model.sha256_status, model.sha256_status)

        table.add_row(
            str(i),
            model.name,
            str(model.path),
            size_display,
            experts,
            active,
            moe_size,
            repo_display,
            sha256_display,
        )

    console.print(table)
    console.print()

    choice = prompt_int_with_retry(
        t("run_int_select_model"),
        default=1,
        min_val=1,
        max_val=len(moe_models),
    )

    return moe_models[choice - 1]


def select_inference_method(model: Any) -> Optional[Dict[str, Any]]:
    """Step 2: Select inference method.

    Args:
        model: Selected UserModel

    Returns:
        Dict with 'method' (raw/amx/gguf/saved), and method-specific fields, or None if cancelled.
    """
    from kt_kernel.cli.utils.run_configs import RunConfigManager

    config_manager = RunConfigManager()
    saved_configs = config_manager.list_configs(model.id)

    # Debug output (can be removed later)
    if False:  # Set to True for debugging
        console.print()
        console.print(f"[dim]DEBUG: Model ID: {model.id}[/dim]")
        console.print(f"[dim]DEBUG: Saved configs count: {len(saved_configs)}[/dim]")
        if saved_configs:
            console.print(f"[dim]DEBUG: Configs: {[c.get('config_name', '?') for c in saved_configs]}[/dim]")
        console.print()

    console.print()
    console.print(Panel("[bold cyan]Step 2: Select Inference Method[/bold cyan]", expand=False))
    console.print()

    options = []
    option_map = {}

    # Option 1: Use saved configuration (if any)
    if saved_configs:
        option_idx = len(options) + 1
        console.print(f"  [cyan][{option_idx}][/cyan] [bold]Use Saved Configuration[/bold]")
        console.print(f"      [dim]{len(saved_configs)} saved config(s) available[/dim]")
        options.append(str(option_idx))
        option_map[str(option_idx)] = "saved"

    # Option 2: Raw precision inference
    option_idx = len(options) + 1
    console.print(f"  [cyan][{option_idx}][/cyan] [bold]Raw Precision Inference[/bold]")
    console.print("      [dim]FP8 / FP8_PERCHANNEL / BF16 / RAW_INT4[/dim]")
    options.append(str(option_idx))
    option_map[str(option_idx)] = "raw"

    # Option 3: AMX quantized inference
    option_idx = len(options) + 1
    console.print(f"  [cyan][{option_idx}][/cyan] [bold]AMX Quantized Inference[/bold]")
    console.print("      [dim]INT4 / INT8 (CPU optimized)[/dim]")
    options.append(str(option_idx))
    option_map[str(option_idx)] = "amx"

    # Option 4: GGUF inference
    option_idx = len(options) + 1
    console.print(f"  [cyan][{option_idx}][/cyan] [bold]GGUF Inference[/bold]")
    console.print("      [dim]Llamafile format[/dim]")
    options.append(str(option_idx))
    option_map[str(option_idx)] = "gguf"

    console.print()

    choice = prompt_choice_with_retry("Select method", choices=options, default="1")
    method = option_map[choice]

    if method == "saved":
        return _select_saved_config(model, saved_configs)
    elif method == "raw":
        return _configure_raw_inference(model)
    elif method == "amx":
        return _configure_amx_inference(model)
    elif method == "gguf":
        return _configure_gguf_inference(model)

    return None


def _select_saved_config(model: Any, saved_configs: List[Dict]) -> Optional[Dict[str, Any]]:
    """Select from saved configurations with detailed display."""
    console.print()
    console.print("[bold]Saved Configurations:[/bold]")
    console.print()

    for i, cfg in enumerate(saved_configs, 1):
        # Build method display
        method_display = cfg.get("inference_method", "unknown").upper()
        kt_method = cfg.get("kt_method", "unknown")

        if cfg.get("inference_method") == "raw":
            raw_method = cfg.get("raw_method", "unknown")
            method_display = f"{raw_method}"
        elif cfg.get("inference_method") == "amx":
            method_display = kt_method
        elif cfg.get("inference_method") == "gguf":
            method_display = "LLAMAFILE"
        else:
            method_display = kt_method

        # Display config header
        console.print(f"  [cyan][{i}][/cyan] [bold]{cfg.get('config_name', f'Config {i}')}[/bold]")
        console.print()

        # Display detailed parameters
        console.print(f"      [yellow]KT Method:[/yellow]       {method_display}")
        console.print(f"      [yellow]NUMA Nodes:[/yellow]      {cfg.get('numa_nodes', '?')}")
        console.print(f"      [yellow]CPU Threads:[/yellow]     {cfg.get('cpu_threads', '?')}")
        console.print(f"      [yellow]GPU Experts:[/yellow]     {cfg.get('gpu_experts', '?')}")
        console.print(f"      [yellow]TP Size:[/yellow]         {cfg.get('tp_size', '?')}")
        console.print(f"      [yellow]Memory Fraction:[/yellow] {cfg.get('mem_fraction_static', '?')}")
        console.print(f"      [yellow]Server:[/yellow]          {cfg.get('host', '0.0.0.0')}:{cfg.get('port', 30000)}")

        # Display KV cache info if present
        if cfg.get("kv_cache"):
            console.print(f"      [yellow]KV Cache:[/yellow]        {cfg.get('kv_cache', '?')}")
            console.print(f"      [yellow]Chunk Prefill:[/yellow]   {cfg.get('chunk_prefill', '?')}")
            console.print(f"      [yellow]GPU Prefill Thr:[/yellow] {cfg.get('gpu_prefill_threshold', '?')}")

        # Display parser info if present
        if cfg.get("tool_call_parser") or cfg.get("reasoning_parser"):
            if cfg.get("tool_call_parser"):
                console.print(f"      [yellow]Tool Call Parser:[/yellow] {cfg.get('tool_call_parser')}")
            if cfg.get("reasoning_parser"):
                console.print(f"      [yellow]Reasoning Parser:[/yellow] {cfg.get('reasoning_parser')}")

        console.print()

        # Build and display command preview
        cmd_preview = _build_command_preview(model, cfg)
        console.print("      [dim]Command:[/dim]")
        console.print()
        for line in cmd_preview:
            console.print(f"      {line}")
        console.print()

    choice = prompt_int_with_retry(
        "Select configuration",
        default=1,
        min_val=1,
        max_val=len(saved_configs),
    )

    selected_config = saved_configs[choice - 1].copy()
    selected_config["method"] = "saved"
    return selected_config


def _build_command_preview(model: Any, cfg: Dict[str, Any]) -> List[str]:
    """Build command preview for saved configuration.

    Args:
        model: UserModel object
        cfg: Saved configuration dict

    Returns:
        List of command lines for display
    """
    import sys

    host = cfg.get("host", "0.0.0.0")
    port = cfg.get("port", 30000)

    lines = [
        "python -m sglang.launch_server \\",
        f"    --host {host} \\",
        f"    --port {port} \\",
        f"    --model {cfg.get('model_path', '?')} \\",
        f"    --kt-weight-path {cfg.get('weights_path', '?')} \\",
        f"    --kt-cpuinfer {cfg.get('cpu_threads', '?')} \\",
        f"    --kt-threadpool-count {cfg.get('numa_nodes', '?')} \\",
        f"    --kt-num-gpu-experts {cfg.get('gpu_experts', '?')} \\",
        f"    --kt-method {cfg.get('kt_method', '?')} \\",
    ]

    # Add GPU prefill threshold (use saved value or default)
    gpu_prefill = cfg.get("gpu_prefill_threshold", 500)
    lines.append(f"    --kt-gpu-prefill-token-threshold {gpu_prefill} \\")
    lines.append("    --kt-enable-dynamic-expert-update \\")

    # Add attention backend
    lines.append("    --attention-backend flashinfer \\")
    lines.append("    --trust-remote-code \\")

    # Add memory and performance settings
    lines.append(f"    --mem-fraction-static {cfg.get('mem_fraction_static', 0.9)} \\")

    # Add KV cache settings
    chunk_prefill = cfg.get("chunk_prefill", 32768)
    max_tokens = cfg.get("kv_cache", 32768)
    lines.append(f"    --chunked-prefill-size {chunk_prefill} \\")
    lines.append(f"    --max-total-tokens {max_tokens} \\")

    lines.append("    --max-running-requests 4 \\")
    lines.append("    --watchdog-timeout 3000 \\")
    lines.append("    --enable-mixed-chunk \\")

    # Add TP size (will be updated with actual GPU selection)
    lines.append(f"    --tensor-parallel-size {cfg.get('tp_size', '?')} \\")
    lines.append("    --enable-p2p-check \\")

    # Add FP8 backend if using FP8
    kt_method = cfg.get("kt_method", "")
    if "FP8" in kt_method.upper():
        lines.append("    --fp8-gemm-backend triton \\")

    # Add parsers if configured
    if cfg.get("tool_call_parser"):
        lines.append(f"    --tool-call-parser {cfg['tool_call_parser']} \\")
    if cfg.get("reasoning_parser"):
        lines.append(f"    --reasoning-parser {cfg['reasoning_parser']} \\")

    # Remove trailing backslash from last line
    if lines:
        lines[-1] = lines[-1].rstrip(" \\")

    return lines


def _configure_raw_inference(model: Any) -> Dict[str, Any]:
    """Configure raw precision inference."""
    console.print()
    console.print("[bold]Select Raw Precision Type:[/bold]")
    console.print()
    console.print("  [cyan][1][/cyan] FP8")
    console.print("  [cyan][2][/cyan] FP8_PERCHANNEL")
    console.print("  [cyan][3][/cyan] BF16")
    console.print("  [cyan][4][/cyan] RAW_INT4")
    console.print()

    choice = prompt_choice_with_retry("Select precision", choices=["1", "2", "3", "4"], default="1")

    precision_map = {
        "1": "FP8",
        "2": "FP8_PERCHANNEL",
        "3": "BF16",
        "4": "RAW_INT4",
    }

    raw_method = precision_map[choice]

    return {
        "method": "raw",
        "raw_method": raw_method,
        "kt_method": raw_method,
        "model_path": model.path,
        "weights_path": model.path,  # Same as model path for raw
    }


def _configure_amx_inference(model: Any) -> Optional[Dict[str, Any]]:
    """Configure AMX quantized inference."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.commands.model import is_amx_weights

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Filter AMX models
    amx_models = []
    for m in all_models:
        if m.format == "safetensors":
            is_amx, numa = is_amx_weights(m.path)
            if is_amx:
                # Check if it's derived from the selected model
                if m.amx_source_model == model.name:
                    amx_models.insert(0, m)  # Prioritize matched models
                else:
                    amx_models.append(m)

    if not amx_models:
        console.print("[yellow]No AMX quantized models found.[/yellow]")
        console.print("  Quantize your model with: [cyan]kt quant[/cyan]")
        return None

    console.print()
    console.print("[bold]Select AMX Weights:[/bold]")
    console.print()

    for i, m in enumerate(amx_models, 1):
        is_amx, numa = is_amx_weights(m.path)
        method_str = m.amx_quant_method.upper() if m.amx_quant_method else "Unknown"
        match_indicator = "[green]★[/green]" if m.amx_source_model == model.name else " "
        console.print(f"  {match_indicator} [cyan][{i}][/cyan] {m.name}")
        console.print(
            f"      [dim]Method: AMX{method_str}, NUMA: {numa}, Source: {m.amx_source_model or 'Unknown'}[/dim]"
        )

    console.print()
    choice = prompt_int_with_retry(
        "Select AMX weights",
        default=1,
        min_val=1,
        max_val=len(amx_models),
    )

    selected_amx = amx_models[choice - 1]
    is_amx, numa = is_amx_weights(selected_amx.path)
    kt_method = f"AMX{selected_amx.amx_quant_method.upper()}" if selected_amx.amx_quant_method else "AMXINT4"

    return {
        "method": "amx",
        "kt_method": kt_method,
        "model_path": model.path,
        "weights_path": selected_amx.path,
        "amx_numa_nodes": numa,
    }


def _configure_gguf_inference(model: Any) -> Optional[Dict[str, Any]]:
    """Configure GGUF inference."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()
    all_models = registry.list_models()

    # Filter GGUF models
    gguf_models = [m for m in all_models if m.format == "gguf"]

    if not gguf_models:
        console.print("[yellow]No GGUF models found.[/yellow]")
        console.print("  Add GGUF models with: [cyan]kt model add /path/to/model.gguf[/cyan]")
        return None

    console.print()
    console.print("[bold]Select GGUF Weights:[/bold]")
    console.print()

    for i, m in enumerate(gguf_models, 1):
        console.print(f"  [cyan][{i}][/cyan] {m.name}")
        console.print(f"      [dim]Path: {m.path}[/dim]")

    console.print()
    choice = prompt_int_with_retry(
        "Select GGUF weights",
        default=1,
        min_val=1,
        max_val=len(gguf_models),
    )

    selected_gguf = gguf_models[choice - 1]

    return {
        "method": "gguf",
        "kt_method": "LLAMAFILE",
        "model_path": model.path,
        "weights_path": selected_gguf.path,
    }


def configure_numa_and_cpu(method_config: Dict[str, Any]) -> Dict[str, int]:
    """Step 3: Configure NUMA and CPU threads.

    Args:
        method_config: Config from step 2 (may contain amx_numa_nodes hint)

    Returns:
        Dict with 'numa_nodes' and 'cpu_threads'
    """
    from kt_kernel.cli.utils.environment import detect_cpu_info

    cpu_info = detect_cpu_info()
    max_numa = cpu_info.numa_nodes
    max_cores = cpu_info.cores

    console.print()
    console.print(Panel("[bold cyan]Step 3: NUMA and CPU Configuration[/bold cyan]", expand=False))
    console.print()

    # Show AMX hint if applicable
    if method_config.get("method") == "amx" and method_config.get("amx_numa_nodes"):
        amx_numa = method_config["amx_numa_nodes"]
        console.print(f"[yellow]⚠ Note: This AMX model was quantized with NUMA={amx_numa}[/yellow]")
        console.print(f"[yellow]  For optimal performance, use the same NUMA setting.[/yellow]")
        console.print()
        default_numa = amx_numa
    else:
        default_numa = max_numa

    numa_nodes = prompt_int_with_retry(
        f"NUMA Nodes (1 to {max_numa})",
        default=default_numa,
        min_val=1,
        max_val=max_numa,
    )

    default_threads = int(max_cores * 0.8)
    cpu_threads = prompt_int_with_retry(
        f"CPU Threads (1 to {max_cores})",
        default=default_threads,
        min_val=1,
        max_val=max_cores,
    )

    return {
        "numa_nodes": numa_nodes,
        "cpu_threads": cpu_threads,
    }


def configure_gpu_experts(model: Any) -> int:
    """Step 4: Configure GPU expert count.

    Args:
        model: Selected model

    Returns:
        Number of GPU experts
    """
    from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

    console.print()
    console.print(Panel("[bold cyan]Step 4: GPU Experts Configuration[/bold cyan]", expand=False))
    console.print()

    # Try to get num_experts from model
    try:
        moe_result = analyze_moe_model(model.path)
        num_experts = moe_result.get("num_experts", 256)
    except Exception:
        num_experts = 256  # Default fallback

    console.print(f"[dim]Model has {num_experts} experts total[/dim]")
    console.print()
    console.print("[yellow]⚠ Tip: More GPU experts = faster inference, but uses more VRAM[/yellow]")
    console.print()

    default_experts = min(8, num_experts)
    gpu_experts = prompt_int_with_retry(
        f"GPU Experts per layer (0 to {num_experts})",
        default=default_experts,
        min_val=0,
        max_val=num_experts,
    )

    return gpu_experts


def configure_kv_cache(is_raw_inference: bool) -> Optional[Dict[str, int]]:
    """Step 5: Configure KV Cache (only for raw inference).

    Args:
        is_raw_inference: True if using raw precision inference

    Returns:
        Dict with 'kv_cache', 'chunk_prefill', 'gpu_prefill_threshold' or None if not applicable
    """
    if not is_raw_inference:
        return None

    console.print()
    console.print(Panel("[bold cyan]Step 5: KV Cache and Prefill Configuration[/bold cyan]", expand=False))
    console.print()
    console.print("[dim]These settings control memory allocation and prefill batch size[/dim]")
    console.print("[dim]gpu-prefill-token-threshold: maximum length for single layerwise prefill[/dim]")
    console.print()

    kv_cache = prompt_int_with_retry("KV Cache Size (max_total_tokens)", default=32768, min_val=1)
    chunk_prefill = prompt_int_with_retry("Chunk Prefill Size", default=32768, min_val=1)
    gpu_prefill_threshold = prompt_int_with_retry("GPU Prefill Token Threshold", default=500, min_val=1)

    return {
        "kv_cache": kv_cache,
        "chunk_prefill": chunk_prefill,
        "gpu_prefill_threshold": gpu_prefill_threshold,
    }


def select_gpus_and_tp(
    required_tp_size: Optional[int] = None, saved_mem_fraction: Optional[float] = None
) -> Tuple[List[int], int, float]:
    """Step 6: Select GPUs, TP size, and memory fraction.

    Args:
        required_tp_size: If specified, user must select exactly this many GPUs.
                         If None, TP size can be any power of 2.
        saved_mem_fraction: If specified, use this memory fraction instead of prompting.
                           Used when loading saved configurations.

    Returns:
        Tuple of (selected_gpu_ids, tp_size, mem_fraction_static)
    """
    gpu_info_list = get_gpu_info()

    if not gpu_info_list:
        console.print("[red]No GPUs detected[/red]")
        return [], 0, 0.9

    console.print()
    if required_tp_size is not None:
        console.print(Panel(f"[bold cyan]Select {required_tp_size} GPUs (for saved config)[/bold cyan]", expand=False))
        console.print()
        console.print(f"[yellow]Required TP size: {required_tp_size}[/yellow]")
        console.print(f"[yellow]You must select exactly {required_tp_size} GPU(s)[/yellow]")
    else:
        console.print(Panel("[bold cyan]Step 6: GPU Selection and Memory[/bold cyan]", expand=False))
        console.print()
        console.print("[dim]TP (Tensor Parallel) size must be a power of 2: 1, 2, 4, 8, ...[/dim]")
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

    # Validator function
    def validate_tp_requirements(gpu_ids: List[int]) -> tuple[bool, Optional[str]]:
        """Validate TP requirements based on required_tp_size."""
        actual_count = len(gpu_ids)

        if required_tp_size is not None:
            # Exact count required
            if actual_count != required_tp_size:
                return False, f"Must select exactly {required_tp_size} GPU(s), but you selected {actual_count}."
        else:
            # Must be power of 2
            if actual_count & (actual_count - 1) != 0:
                return (
                    False,
                    f"TP size ({actual_count}) must be a power of 2. Valid sizes: 1, 2, 4, 8, 16, 32, ...\nYou selected {actual_count} GPU(s). Please select a different number.",
                )

        return True, None

    # Generate default GPU selection
    if required_tp_size is not None:
        # For saved config: select first N GPUs
        if required_tp_size <= len(gpu_info_list):
            default_gpus = ",".join(str(i) for i in range(required_tp_size))
        else:
            default_gpus = ",".join(str(i) for i in range(len(gpu_info_list)))
        prompt_text = f"Enter {required_tp_size} GPU ID(s) separated by commas (e.g., 0,1,2,3)"
    else:
        # For new config: select all GPUs
        default_gpus = ",".join(str(i) for i in range(len(gpu_info_list)))
        prompt_text = "Enter GPU IDs separated by commas (e.g., 0,1,2,3)"
        console.print(prompt_text)
        console.print(f"  Or press Enter to use all {len(gpu_info_list)} GPUs")

    console.print()

    selected_gpu_ids = prompt_int_list_with_retry(
        "GPU IDs",
        default=default_gpus,
        min_val=0,
        max_val=len(gpu_info_list) - 1,
        validator=validate_tp_requirements,
    )

    tp_size = len(selected_gpu_ids)

    console.print()
    console.print(f"[green]✓[/green] Selected {tp_size} GPU(s): {selected_gpu_ids}")
    console.print()

    # Memory fraction - use saved value if provided, otherwise prompt
    if saved_mem_fraction is not None:
        mem_fraction = saved_mem_fraction
        console.print(f"[dim]Using saved memory fraction: {mem_fraction}[/dim]")
    else:
        mem_fraction = prompt_float_with_retry(
            "Static Memory Fraction (0.0-1.0)",
            default=0.9,
            min_val=0.0,
            max_val=1.0,
        )

    return selected_gpu_ids, tp_size, mem_fraction


def configure_parsers() -> Dict[str, Optional[str]]:
    """Step 7: Configure parsers (optional).

    Returns:
        Dict with 'tool_call_parser' and 'reasoning_parser' (can be None)
    """
    console.print()
    console.print(Panel("[bold cyan]Step 7: Parser Configuration (Optional)[/bold cyan]", expand=False))
    console.print()
    console.print("[dim]Press Enter to skip (no parser will be added)[/dim]")
    console.print()

    tool_call_parser = Prompt.ask("Tool Call Parser (e.g., glm47)", default="")
    tool_call_parser = tool_call_parser.strip() if tool_call_parser else None

    reasoning_parser = Prompt.ask("Reasoning Parser (e.g., glm45)", default="")
    reasoning_parser = reasoning_parser.strip() if reasoning_parser else None

    if tool_call_parser or reasoning_parser:
        console.print()
        if tool_call_parser:
            console.print(f"[green]✓[/green] Tool Call Parser: {tool_call_parser}")
        if reasoning_parser:
            console.print(f"[green]✓[/green] Reasoning Parser: {reasoning_parser}")
    else:
        console.print()
        console.print("[dim]No parsers configured[/dim]")

    return {
        "tool_call_parser": tool_call_parser,
        "reasoning_parser": reasoning_parser,
    }


def configure_host_and_port() -> Dict[str, Any]:
    """Step 8: Configure host and port with availability check.

    Returns:
        Dict with 'host' and 'port'
    """
    from kt_kernel.cli.utils.port_checker import is_port_available

    console.print()
    console.print(Panel("[bold cyan]Step 8: Server Configuration[/bold cyan]", expand=False))
    console.print()

    # Get host
    host = Prompt.ask("Server Host", default="0.0.0.0")

    # Get port with availability check
    while True:
        port = prompt_int_with_retry(
            "Server Port",
            default=30000,
            min_val=1024,
            max_val=65535,
        )

        # Check if port is available
        console.print()
        console.print(f"[dim]Checking port {port} availability...[/dim]")

        if is_port_available(host, port):
            console.print(f"[green]✓[/green] Port {port} is available")
            break
        else:
            console.print(f"[red]✗[/red] Port {port} is already in use")
            console.print()

            # Suggest next available port
            from kt_kernel.cli.utils.port_checker import find_available_port

            found, suggested_port = find_available_port(host, port + 1, max_attempts=100)
            if found:
                console.print(f"[yellow]Suggestion:[/yellow] Port {suggested_port} is available")
            console.print()

    console.print()
    console.print(f"[green]✓[/green] Server will listen on {host}:{port}")

    return {
        "host": host,
        "port": port,
    }


def save_config_prompt(model: Any, full_config: Dict[str, Any]) -> bool:
    """Step 7: Prompt to save configuration.

    Args:
        model: Selected model
        full_config: Complete configuration dict

    Returns:
        True if saved, False otherwise
    """
    console.print()
    console.print(Panel("[bold cyan]Step 7: Save Configuration[/bold cyan]", expand=False))
    console.print()

    if not Confirm.ask("Save this configuration for future use?", default=True):
        return False

    config_name = Prompt.ask("Configuration name", default=f"Config {full_config.get('inference_method', 'default')}")

    from kt_kernel.cli.utils.run_configs import RunConfigManager

    config_manager = RunConfigManager()

    # Prepare config to save (exclude runtime-only fields and non-serializable objects)
    save_config = {
        "config_name": config_name,
        "inference_method": full_config["inference_method"],
        "kt_method": full_config["kt_method"],
        "model_path": str(full_config["model_path"]),
        "weights_path": str(full_config["weights_path"]),
        "numa_nodes": full_config["numa_nodes"],
        "cpu_threads": full_config["cpu_threads"],
        "gpu_experts": full_config["gpu_experts"],
        "tp_size": full_config["tp_size"],
        "mem_fraction_static": full_config["mem_fraction_static"],
        "host": full_config["host"],
        "port": full_config["port"],
        # Note: selected_gpus is NOT saved - user will select GPUs when loading config
    }

    # Add parser config if present
    if full_config.get("tool_call_parser"):
        save_config["tool_call_parser"] = full_config["tool_call_parser"]
    if full_config.get("reasoning_parser"):
        save_config["reasoning_parser"] = full_config["reasoning_parser"]

    # Add raw-specific config if present
    if full_config.get("raw_method"):
        save_config["raw_method"] = full_config["raw_method"]

    if full_config.get("kv_cache"):
        save_config["kv_cache"] = full_config["kv_cache"]
        save_config["chunk_prefill"] = full_config["chunk_prefill"]
        save_config["gpu_prefill_threshold"] = full_config["gpu_prefill_threshold"]

    config_manager.save_config(model.id, save_config)

    console.print()
    console.print(f"[green]✓[/green] Configuration saved: {config_name}")

    return True


def interactive_run_config() -> Optional[Dict[str, Any]]:
    """
    Main interactive configuration flow for kt run.

    Returns:
        Complete configuration dict or None if cancelled.
    """
    # Step 1: Select model
    model = select_model()
    if not model:
        return None

    # Step 2: Select inference method
    method_config = select_inference_method(model)
    if not method_config:
        return None

    # If using saved config, add model object and return directly
    if method_config.get("method") == "saved":
        console.print()
        console.print("[green]✓[/green] Using saved configuration")

        # Let user select GPUs (must match saved TP size)
        saved_tp_size = method_config.get("tp_size", 1)

        console.print()
        console.print(f"[yellow]This configuration requires TP={saved_tp_size}[/yellow]")
        console.print(f"[yellow]Please select {saved_tp_size} GPU(s)[/yellow]")

        # Get saved memory fraction
        saved_mem_fraction = method_config.get("mem_fraction_static", 0.9)

        selected_gpus, actual_tp_size, _ = select_gpus_and_tp(
            required_tp_size=saved_tp_size, saved_mem_fraction=saved_mem_fraction
        )
        if not selected_gpus:
            return None

        # Update config with selected GPUs (keep saved mem_fraction_static)
        method_config["selected_gpus"] = selected_gpus
        # tp_size is already in method_config from saved data

        # Check port availability
        from kt_kernel.cli.utils.port_checker import is_port_available, find_available_port

        saved_host = method_config.get("host", "0.0.0.0")
        saved_port = method_config.get("port", 30000)

        console.print()
        console.print(f"[dim]Checking port {saved_port} availability...[/dim]")

        if is_port_available(saved_host, saved_port):
            console.print(f"[green]✓[/green] Port {saved_port} is available")
            method_config["port"] = saved_port
            method_config["host"] = saved_host
        else:
            console.print(f"[red]✗[/red] Port {saved_port} is already in use")
            console.print()

            # Suggest next available port
            found, suggested_port = find_available_port(saved_host, saved_port + 1, max_attempts=100)
            if found:
                console.print(f"[yellow]Suggestion:[/yellow] Port {suggested_port} is available")
            console.print()

            # Ask user for new port
            while True:
                new_port = prompt_int_with_retry(
                    "Enter new port",
                    default=suggested_port if found else saved_port + 1,
                    min_val=1024,
                    max_val=65535,
                )

                console.print()
                console.print(f"[dim]Checking port {new_port} availability...[/dim]")

                if is_port_available(saved_host, new_port):
                    console.print(f"[green]✓[/green] Port {new_port} is available")
                    method_config["port"] = new_port
                    method_config["host"] = saved_host
                    break
                else:
                    console.print(f"[red]✗[/red] Port {new_port} is already in use")
                    console.print()

        # Add model object for run.py compatibility
        method_config["model"] = model

        # Ensure paths are Path objects
        from pathlib import Path

        if "model_path" in method_config:
            method_config["model_path"] = Path(method_config["model_path"])
        if "weights_path" in method_config:
            method_config["weights_path"] = Path(method_config["weights_path"])

        # Display configuration summary
        console.print()
        console.print(Panel("[bold cyan]Saved Configuration[/bold cyan]", expand=False))
        console.print()
        _display_config_summary(method_config)
        console.print()

        # Start directly without confirmation when using saved config
        return method_config

    # Step 3: Configure NUMA and CPU
    numa_cpu_config = configure_numa_and_cpu(method_config)

    # Step 4: Configure GPU experts
    gpu_experts = configure_gpu_experts(model)

    # Step 5: Configure KV Cache (only for raw)
    is_raw = method_config.get("method") == "raw"
    kv_config = configure_kv_cache(is_raw)

    # Step 6: Select GPUs and TP
    selected_gpus, tp_size, mem_fraction = select_gpus_and_tp()
    if not selected_gpus:
        return None

    # Step 7: Configure parsers (optional)
    parser_config = configure_parsers()

    # Step 8: Configure host and port
    server_config = configure_host_and_port()

    # Build complete configuration
    full_config = {
        "model": model,
        "inference_method": method_config["method"],
        "kt_method": method_config["kt_method"],
        "model_path": method_config["model_path"],
        "weights_path": method_config["weights_path"],
        **numa_cpu_config,
        "gpu_experts": gpu_experts,
        "selected_gpus": selected_gpus,
        "tp_size": tp_size,
        "mem_fraction_static": mem_fraction,
        **parser_config,  # Add parser config
        **server_config,  # Add server config (host, port)
    }

    # Add raw-specific config
    if kv_config:
        full_config["raw_method"] = method_config.get("raw_method")
        full_config.update(kv_config)

    # Step 9: Save configuration
    save_config_prompt(model, full_config)

    # Final confirmation
    console.print()
    console.print(Panel("[bold cyan]Configuration Complete[/bold cyan]", expand=False))
    console.print()
    _display_config_summary(full_config)
    console.print()

    if not Confirm.ask("[bold green]Start model server with this configuration?[/bold green]", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return None

    return full_config


def _display_config_summary(config: Dict[str, Any]):
    """Display configuration summary."""
    model = config["model"]
    console.print(f"  Model:           {model.name}")
    console.print(f"  KT Method:       {config['kt_method']}")
    console.print(f"  NUMA Nodes:      {config['numa_nodes']}")
    console.print(f"  CPU Threads:     {config['cpu_threads']}")
    console.print(f"  GPU Experts:     {config['gpu_experts']}")

    # Handle both new config and saved config format
    tp_size = config.get("tp_size", len(config.get("selected_gpus", [])))
    selected_gpus = config.get("selected_gpus", [])

    console.print(f"  GPUs:            {selected_gpus} (TP={tp_size})")
    console.print(f"  Memory Fraction: {config['mem_fraction_static']}")

    # Server config
    host = config.get("host", "0.0.0.0")
    port = config.get("port", 30000)
    console.print(f"  Server:          {host}:{port}")

    if config.get("kv_cache"):
        console.print(f"  KV Cache:        {config['kv_cache']}")
        console.print(f"  Chunk Prefill:   {config['chunk_prefill']}")
        console.print(f"  GPU Prefill Thr: {config['gpu_prefill_threshold']}")

    # Display parsers if configured
    if config.get("tool_call_parser") or config.get("reasoning_parser"):
        console.print()
        if config.get("tool_call_parser"):
            console.print(f"  Tool Call Parser: {config['tool_call_parser']}")
        if config.get("reasoning_parser"):
            console.print(f"  Reasoning Parser: {config['reasoning_parser']}")
