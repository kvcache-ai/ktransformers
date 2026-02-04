"""
Model Discovery Utilities

Shared functions for discovering and registering new models across different commands.
"""

from typing import List, Optional, Tuple
from pathlib import Path
from rich.console import Console

from kt_kernel.cli.utils.model_scanner import (
    discover_models,
    scan_directory_for_models,
    ScannedModel,
)
from kt_kernel.cli.utils.user_model_registry import UserModelRegistry, UserModel


console = Console()


def discover_and_register_global(
    min_size_gb: float = 2.0, max_depth: int = 6, show_progress: bool = True, lang: str = "en"
) -> Tuple[int, int, List[UserModel]]:
    """
    Perform global model discovery and register new models.

    Args:
        min_size_gb: Minimum model size in GB
        max_depth: Maximum search depth
        show_progress: Whether to show progress messages
        lang: Language for messages ("en" or "zh")

    Returns:
        Tuple of (total_found, new_found, registered_models)
    """
    registry = UserModelRegistry()

    if show_progress:
        if lang == "zh":
            console.print("[dim]正在扫描系统中的模型权重，这可能需要30-60秒...[/dim]")
        else:
            console.print("[dim]Scanning system for model weights, this may take 30-60 seconds...[/dim]")

    # Global scan
    all_models = discover_models(mount_points=None, min_size_gb=min_size_gb, max_depth=max_depth)

    # Filter out existing models
    new_models = []
    for model in all_models:
        if not registry.find_by_path(model.path):
            new_models.append(model)

    # Register new models
    registered = []
    for model in new_models:
        user_model = _create_and_register_model(registry, model)
        if user_model:
            registered.append(user_model)

    return len(all_models), len(new_models), registered


def discover_and_register_path(
    path: str,
    min_size_gb: float = 2.0,
    existing_paths: Optional[set] = None,
    show_progress: bool = True,
    lang: str = "en",
) -> Tuple[int, int, List[UserModel]]:
    """
    Discover models in a specific path and register new ones.

    Args:
        path: Directory path to scan
        min_size_gb: Minimum model file size in GB
        existing_paths: Set of already discovered paths in this session (optional)
        show_progress: Whether to show progress messages
        lang: Language for messages ("en" or "zh")

    Returns:
        Tuple of (total_found, new_found, registered_models)
    """
    registry = UserModelRegistry()

    if show_progress:
        if lang == "zh":
            console.print(f"[dim]正在扫描 {path}...[/dim]")
        else:
            console.print(f"[dim]Scanning {path}...[/dim]")

    # Scan directory
    model_info = scan_directory_for_models(path, min_file_size_gb=min_size_gb)

    if not model_info:
        return 0, 0, []

    # Convert to ScannedModel and filter
    new_models = []
    for dir_path, (format_type, size_bytes, file_count, files) in model_info.items():
        # Check if already in registry
        if registry.find_by_path(dir_path):
            continue

        # Check if already discovered in this session
        if existing_paths and dir_path in existing_paths:
            continue

        model = ScannedModel(
            path=dir_path, format=format_type, size_bytes=size_bytes, file_count=file_count, files=files
        )
        new_models.append(model)

    # Register new models
    registered = []
    for model in new_models:
        user_model = _create_and_register_model(registry, model)
        if user_model:
            registered.append(user_model)

    return len(model_info), len(new_models), registered


def _create_and_register_model(registry: UserModelRegistry, scanned_model: ScannedModel) -> Optional[UserModel]:
    """
    Create a UserModel from ScannedModel and register it.

    Handles name conflicts by suggesting a unique name (e.g., model-2, model-3).
    Automatically detects repo_id from README.md YAML frontmatter.
    Automatically detects and caches MoE information for safetensors models.

    Args:
        registry: UserModelRegistry instance
        scanned_model: ScannedModel to register

    Returns:
        Registered UserModel or None if failed
    """
    # Use suggest_name to get a unique name (adds -2, -3, etc. if needed)
    unique_name = registry.suggest_name(scanned_model.folder_name)

    user_model = UserModel(name=unique_name, path=scanned_model.path, format=scanned_model.format)

    # Auto-detect repo_id from README.md (only YAML frontmatter)
    try:
        from kt_kernel.cli.utils.repo_detector import detect_repo_for_model

        repo_info = detect_repo_for_model(scanned_model.path)
        if repo_info:
            repo_id, repo_type = repo_info
            user_model.repo_id = repo_id
            user_model.repo_type = repo_type
    except Exception:
        # Silently continue if detection fails
        pass

    # Auto-detect MoE information for safetensors models
    if scanned_model.format == "safetensors":
        try:
            from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

            moe_result = analyze_moe_model(scanned_model.path, use_cache=True)
            if moe_result and moe_result.get("is_moe"):
                user_model.is_moe = True
                user_model.moe_num_experts = moe_result.get("num_experts")
                user_model.moe_num_experts_per_tok = moe_result.get("num_experts_per_tok")
            else:
                user_model.is_moe = False
        except Exception:
            # Silently continue if MoE detection fails
            # is_moe will remain None
            pass

    try:
        registry.add_model(user_model)
        return user_model
    except Exception:
        # Should not happen since we used suggest_name, but handle gracefully
        return None


def format_discovery_summary(
    total_found: int,
    new_found: int,
    registered: List[UserModel],
    lang: str = "en",
    show_models: bool = True,
    max_show: int = 10,
) -> None:
    """
    Print formatted discovery summary.

    Args:
        total_found: Total models found
        new_found: New models found
        registered: List of registered UserModel objects
        lang: Language ("en" or "zh")
        show_models: Whether to show model list
        max_show: Maximum models to show
    """
    console.print()

    if new_found == 0:
        if total_found > 0:
            if lang == "zh":
                console.print(f"[green]✓[/green] 扫描完成：找到 {total_found} 个模型，所有模型均已在列表中")
            else:
                console.print(f"[green]✓[/green] Scan complete: found {total_found} models, all already in the list")
        else:
            if lang == "zh":
                console.print("[yellow]未找到模型[/yellow]")
            else:
                console.print("[yellow]No models found[/yellow]")
        return

    # Show summary
    if lang == "zh":
        console.print(f"[green]✓[/green] 扫描完成：找到 {total_found} 个模型，其中 {new_found} 个为新模型")
    else:
        console.print(f"[green]✓[/green] Scan complete: found {total_found} models, {new_found} are new")

    # Show registered count
    if len(registered) > 0:
        if lang == "zh":
            console.print(f"[green]✓[/green] 成功添加 {len(registered)} 个新模型到列表")
        else:
            console.print(f"[green]✓[/green] Successfully added {len(registered)} new models to list")

    # Show model list
    if show_models and registered:
        console.print()
        if lang == "zh":
            console.print(f"[dim]新发现的模型（前{max_show}个）:[/dim]")
        else:
            console.print(f"[dim]Newly discovered models (first {max_show}):[/dim]")

        for i, model in enumerate(registered[:max_show], 1):
            # Get size from registry or estimate
            size_str = "?.? GB"
            # Try to find the ScannedModel to get size
            # For now just show name and path
            console.print(f"  {i}. {model.name} ({model.format})")
            console.print(f"     [dim]{model.path}[/dim]")

        if len(registered) > max_show:
            remaining = len(registered) - max_show
            if lang == "zh":
                console.print(f"  [dim]... 还有 {remaining} 个新模型[/dim]")
            else:
                console.print(f"  [dim]... and {remaining} more new models[/dim]")
