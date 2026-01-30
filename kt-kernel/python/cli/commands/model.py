"""
Model command for kt-cli.

Manages models: download, list, and storage paths.
"""

import os
from pathlib import Path
from typing import Optional, List

import typer

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t, get_lang
from kt_kernel.cli.utils.console import (
    confirm,
    console,
    print_error,
    print_info,
    print_step,
    print_success,
    print_warning,
    prompt_choice,
)


# Common SHA256 status display mapping used across multiple commands
SHA256_STATUS_MAP = {
    "not_checked": "[dim]Not Checked[/dim]",
    "checking": "[yellow]Checking...[/yellow]",
    "passed": "[green]✓ Passed[/green]",
    "failed": "[red]✗ Failed[/red]",
    "no_repo": "[dim]-[/dim]",
}

# Plain text version for panels and verbose output
SHA256_STATUS_MAP_PLAIN = {
    "not_checked": "Not Checked",
    "checking": "Checking...",
    "passed": "✓ Passed",
    "failed": "✗ Failed",
    "no_repo": "-",
}


def is_amx_weights(model_path) -> tuple[bool, int]:
    """
    Determine if a model uses AMX weights and count NUMA nodes.

    Returns:
        (is_amx, numa_count): Tuple where is_amx indicates AMX weights,
        and numa_count is the number of NUMA nodes (0 if not AMX).
    """
    import re
    from pathlib import Path
    from safetensors import safe_open

    model_path = Path(model_path)
    safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        return False, 0

    numa_indices = set()
    numa_pattern = re.compile(r"\.numa\.(\d+)\.")

    # Check first 3 files for NUMA keys
    for file_path in safetensors_files[:3]:
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if ".numa." in key:
                        match = numa_pattern.search(key)
                        if match:
                            numa_indices.add(int(match.group(1)))
        except Exception:
            continue

    if not numa_indices:
        return False, 0

    return True, len(numa_indices)


app = typer.Typer(
    help="Manage models and storage paths",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback()
def callback(ctx: typer.Context) -> None:
    """
    Model management commands.

    Run without arguments to see available models.
    """
    # If no subcommand is provided, show the full model list
    if ctx.invoked_subcommand is None:
        list_models(verbose=False, all_models=False, show_moe=True, no_cache=False)


@app.command(name="download")
def download(
    repo: Optional[str] = typer.Argument(None, help="Repository ID (optional, interactive mode if not provided)"),
    local_dir: Optional[str] = typer.Option(
        None,
        "--local-dir",
        "-d",
        help="Local directory to download to (default: auto-detect from config)",
    ),
    repo_type: Optional[str] = typer.Option(
        None,
        "--repo-type",
        "-t",
        help="Repository type: huggingface or modelscope",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Resume incomplete downloads",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip all prompts and use defaults",
    ),
) -> None:
    """Download model from HuggingFace or ModelScope (interactive mode)."""
    import subprocess
    import os
    from pathlib import Path
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry, UserModel
    from kt_kernel.cli.utils.model_scanner import scan_single_path, format_size
    from kt_kernel.cli.utils.model_verifier import check_huggingface_connectivity
    from kt_kernel.cli.utils.download_helper import (
        list_remote_files_hf,
        list_remote_files_ms,
        filter_files_by_pattern,
        calculate_total_size,
        format_file_list_table,
        verify_repo_exists,
    )

    settings = get_settings()
    user_registry = UserModelRegistry()

    console.print()

    # ========== Step 1: Select repository type ==========
    if not repo_type and not yes:
        console.print("[bold cyan]Step 1: Select Repository Source[/bold cyan]\n")
        console.print("  1. HuggingFace")
        console.print("  2. ModelScope")
        console.print()

        choice = Prompt.ask("Select source", choices=["1", "2"], default="1")
        repo_type = "huggingface" if choice == "1" else "modelscope"
        console.print()
    elif not repo_type:
        repo_type = "huggingface"  # Default for --yes mode

    # Validate repo_type
    if repo_type not in ["huggingface", "modelscope"]:
        print_error(f"Invalid repo type: {repo_type}. Must be 'huggingface' or 'modelscope'")
        raise typer.Exit(1)

    # Check HuggingFace connectivity and auto-switch to mirror if needed
    use_mirror = False
    if repo_type == "huggingface":
        with console.status("[dim]Checking HuggingFace connectivity...[/dim]"):
            is_accessible, message = check_huggingface_connectivity(timeout=5)

        if not is_accessible:
            print_warning("HuggingFace Connection Failed")
            console.print()
            console.print(f"  {message}")
            console.print()
            console.print("  [yellow]Auto-switching to HuggingFace mirror:[/yellow] [cyan]hf-mirror.com[/cyan]")
            console.print()
            use_mirror = True

    # ========== Step 2: Input repository ID ==========
    while True:
        if not repo and not yes:
            console.print("[bold cyan]Step 2: Enter Repository ID[/bold cyan]\n")
            console.print("  Examples:")
            console.print("    • HuggingFace: deepseek-ai/DeepSeek-V3")
            console.print("    • ModelScope: Qwen/Qwen3-Coder-480B-A35B-Instruct")
            console.print()

            repo = Prompt.ask("Repository ID")
            console.print()
        elif not repo:
            print_error("Repository ID is required")
            raise typer.Exit(1)

        # Verify repository exists
        with console.status(f"[dim]Verifying repository: {repo}...[/dim]"):
            exists, msg = verify_repo_exists(repo, repo_type, use_mirror)

        if exists:
            print_success(f"✓ Repository found: {repo}")
            console.print()
            break
        else:
            print_error(msg)
            console.print()
            if yes:
                raise typer.Exit(1)
            repo = None  # Reset to ask again

    # ========== Step 3: Input file pattern and preview files ==========
    files_to_download = []
    file_pattern = "*"

    while True:
        if not yes:
            console.print("[bold cyan]Step 3: Select Files to Download[/bold cyan]\n")
            console.print("  File pattern (glob syntax):")
            console.print("    • *                  - All files (default)")
            console.print("    • *.safetensors      - Only safetensors files")
            console.print("    • *.gguf             - Only GGUF files")
            console.print("    • *Q4_K_M.gguf       - Specific GGUF quant")
            console.print()

            pattern_input = Prompt.ask("File pattern", default="*")
            file_pattern = pattern_input
            console.print()

        # Fetch remote file list
        with console.status(f"[dim]Fetching file list from {repo_type}...[/dim]"):
            try:
                if repo_type == "huggingface":
                    all_files = list_remote_files_hf(repo, use_mirror)
                else:
                    all_files = list_remote_files_ms(repo)

                files_to_download = filter_files_by_pattern(all_files, file_pattern)
            except Exception as e:
                print_error(f"Failed to fetch file list: {e}")
                raise typer.Exit(1)

        if not files_to_download:
            print_warning(f"No files match pattern: {file_pattern}")
            console.print()
            if yes:
                raise typer.Exit(1)
            continue  # Ask for pattern again

        # Display matched files
        total_size = calculate_total_size(files_to_download)
        print_success(f"Found {len(files_to_download)} files (Total: {format_size(total_size)})")
        console.print()

        file_table = format_file_list_table(files_to_download, max_display=10)
        console.print(file_table)
        console.print()

        # Confirm or retry
        if yes:
            break

        action = Prompt.ask("Action", choices=["continue", "retry", "cancel"], default="continue")

        if action == "continue":
            console.print()
            break
        elif action == "cancel":
            console.print()
            print_info("Download cancelled")
            console.print()
            return
        # else retry - loop continues

    # ========== Step 4: Select download path ==========
    download_path = None

    if local_dir:
        download_path = Path(os.path.expanduser(local_dir)).resolve()
    elif not yes:
        console.print("[bold cyan]Step 4: Select Download Location[/bold cyan]\n")

        # Get configured model paths
        model_paths = settings.get_model_paths()
        if not model_paths:
            print_error("No model storage paths configured.")
            console.print()
            console.print(f"  Add a path with: [cyan]kt model path-add <path>[/cyan]")
            console.print()
            raise typer.Exit(1)

        # Display configured paths
        console.print("  Configured storage paths:")
        for i, path in enumerate(model_paths, 1):
            console.print(f"    {i}. {path}")
        console.print(f"    {len(model_paths) + 1}. Custom path (manual input)")
        console.print()

        path_choice = Prompt.ask("Select path", choices=[str(i) for i in range(1, len(model_paths) + 2)], default="1")

        if int(path_choice) <= len(model_paths):
            base_path = model_paths[int(path_choice) - 1]
        else:
            custom = Prompt.ask("Enter custom path")
            base_path = Path(os.path.expanduser(custom)).resolve()

        console.print()

        # Ask for folder name
        default_folder = repo.split("/")[-1]
        folder_name = Prompt.ask("Folder name", default=default_folder)

        download_path = base_path / folder_name
        console.print()
    else:
        # --yes mode: use default
        model_paths = settings.get_model_paths()
        if not model_paths:
            print_error("No model storage paths configured.")
            raise typer.Exit(1)

        default_folder = repo.split("/")[-1]
        download_path = model_paths[0] / default_folder

    # ========== Step 5: Confirm and download ==========
    print_info(f"Download destination: {download_path}")
    console.print()

    # Check if path exists
    if download_path.exists():
        existing = user_registry.find_by_path(str(download_path))
        if existing:
            print_warning(f"Model already registered as: {existing.name}")
            console.print()
            if not yes and not Confirm.ask("Re-download anyway?", default=False):
                return
        else:
            print_warning(f"Directory already exists: {download_path}")
            if not yes and not Confirm.ask("Overwrite?", default=False):
                return
        console.print()

    # Final confirmation
    if not yes:
        console.print("[bold]Download Summary:[/bold]")
        console.print(f"  Source:      {repo_type}:{repo}")
        console.print(
            f"  Files:       {len(files_to_download)} files ({format_size(calculate_total_size(files_to_download))})"
        )
        console.print(f"  Pattern:     {file_pattern}")
        console.print(f"  Destination: {download_path}")
        console.print()

        if not Confirm.ask("Start download?", default=True):
            console.print()
            print_info("Download cancelled")
            console.print()
            return

    # Download
    console.print()
    print_step("Downloading model files...")
    console.print()

    # Set mirror for HuggingFace if needed
    original_hf_endpoint = os.environ.get("HF_ENDPOINT")
    if use_mirror and repo_type == "huggingface" and not original_hf_endpoint:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        if repo_type == "huggingface":
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo,
                local_dir=str(download_path),
                allow_patterns=file_pattern if file_pattern != "*" else None,
                local_dir_use_symlinks=False,
                resume_download=resume,
            )

        else:  # modelscope
            from modelscope.hub.snapshot_download import snapshot_download

            snapshot_download(
                model_id=repo,
                local_dir=str(download_path),
                allow_file_pattern=file_pattern if file_pattern != "*" else None,
            )

    except ImportError as e:
        pkg = "huggingface_hub" if repo_type == "huggingface" else "modelscope"
        print_error(f"{pkg} not installed. Install: pip install {pkg}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Download failed: {e}")
        raise typer.Exit(1)
    finally:
        # Restore HF_ENDPOINT
        if use_mirror and repo_type == "huggingface" and not original_hf_endpoint:
            os.environ.pop("HF_ENDPOINT", None)
        elif original_hf_endpoint:
            os.environ["HF_ENDPOINT"] = original_hf_endpoint

    # ========== Step 6: Scan and register ==========
    console.print()
    print_success("Download complete!")

    console.print()
    print_step("Scanning downloaded model...")

    try:
        scanned = scan_single_path(download_path)
    except Exception as e:
        print_error(f"Failed to scan model: {e}")
        console.print()
        console.print(f"  You can manually add it: [cyan]kt model add {download_path}[/cyan]")
        console.print()
        raise typer.Exit(1)

    if not scanned:
        print_warning("No model files found in downloaded directory.")
        console.print()
        console.print("  Supported formats: .safetensors, .gguf")
        console.print()
        return

    # Auto-generate model name
    model_name = download_path.name
    if user_registry.check_name_conflict(model_name):
        model_name = user_registry.suggest_name(model_name)

    # Create and register model
    user_model = UserModel(
        name=model_name,
        path=str(download_path),
        format=scanned.format,
        repo_type=repo_type,
        repo_id=repo,
        sha256_status="not_checked",
    )

    try:
        user_registry.add_model(user_model)
        console.print()
        print_success(f"Model registered as: {model_name}")
        console.print()
        console.print(f"  View details:     [cyan]kt model info {model_name}[/cyan]")
        console.print(f"  Run model:        [cyan]kt run {model_name}[/cyan]")
        console.print(f"  Verify integrity: [cyan]kt model verify {model_name}[/cyan]")
        console.print()
    except Exception as e:
        print_error(f"Failed to register model: {e}")
        console.print()
        console.print(f"  You can manually add it: [cyan]kt model add {download_path}[/cyan]")
        console.print()
        raise typer.Exit(1)


@app.command(name="list")
def list_models(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed info including paths"),
    all_models: bool = typer.Option(False, "--all", help="Show all models (reserved for future use)"),
    show_moe: bool = typer.Option(True, "--moe/--no-moe", help="Show MoE model information (default: enabled)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Force re-analyze all models, ignore cache"),
) -> None:
    """List user-registered models."""
    from rich.table import Table
    from rich.panel import Panel
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.utils.model_scanner import format_size
    import sys
    from pathlib import Path as PathLib

    # Try to import analyze_moe_model from multiple locations
    analyze_moe_model = None
    try:
        # Try 1: From kt_kernel.cli.utils
        from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model
    except ImportError:
        try:
            # Try 2: From parent directories
            analyze_moe_path = PathLib(__file__).parent.parent.parent.parent.parent.parent / "analyze_moe_model.py"
            if analyze_moe_path.exists():
                sys.path.insert(0, str(analyze_moe_path.parent))
                from analyze_moe_model import analyze_moe_model
        except (ImportError, Exception):
            try:
                # Try 3: Absolute path
                sys.path.insert(0, "/mnt/data2/ljq/ktransformers")
                from analyze_moe_model import analyze_moe_model
            except (ImportError, Exception):
                analyze_moe_model = None

    registry = UserModelRegistry()
    models = registry.list_models()

    console.print()

    if not models:
        print_warning(t("model_no_registered_models"))
        console.print()
        console.print(f"  {t('model_scan_hint')} [cyan]kt model scan[/cyan]")
        console.print(f"  {t('model_add_hint')} [cyan]kt model add <path>[/cyan]")
        console.print()
        return

    # Check for models with non-existent paths and remove them automatically
    models_to_remove = []
    for model in models:
        if not model.path_exists():
            models_to_remove.append(model)

    if models_to_remove:
        console.print(f"[yellow]Found {len(models_to_remove)} model(s) with non-existent paths:[/yellow]")
        for model in models_to_remove:
            console.print(f"  [dim]✗ {model.name}: {model.path}[/dim]")
            registry.remove_model(model.name)
        console.print(f"[green]✓ Automatically removed {len(models_to_remove)} model(s) with missing paths[/green]")
        console.print()

        # Refresh the models list
        models = registry.list_models()

        if not models:
            console.print(f"[dim]No models remaining after cleanup.[/dim]")
            console.print()
            console.print(f"  {t('model_scan_hint')} [cyan]kt model scan[/cyan]")
            console.print(f"  {t('model_add_hint')} [cyan]kt model add <path>[/cyan]")
            console.print()
            return

    if verbose:
        # Verbose mode: detailed cards
        console.print(f"[bold cyan]{t('model_registered_models_title')}[/bold cyan]\n")

        for i, model in enumerate(models, 1):
            # Check if path exists
            path_status = "[green]✓ Exists[/green]" if model.path_exists() else "[red]✗ Missing[/red]"

            # Format repo info
            if model.repo_id:
                repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
                repo_info = f"{repo_abbr}:{model.repo_id}"
            else:
                repo_info = "-"

            # Format SHA256 status
            sha256_display = SHA256_STATUS_MAP_PLAIN.get(model.sha256_status, model.sha256_status)

            # Calculate folder size if exists
            if model.path_exists():
                from pathlib import Path

                path_obj = Path(model.path)
                try:
                    if model.format == "safetensors":
                        files = list(path_obj.glob("*.safetensors"))
                    else:
                        files = list(path_obj.glob("*.gguf"))

                    total_size = sum(f.stat().st_size for f in files if f.exists())
                    size_str = format_size(total_size)
                    file_count = len(files)
                    size_info = f"{size_str} ({file_count} files)"
                except:
                    size_info = "Unknown"
            else:
                size_info = "-"

            # Create panel content
            content = f"""[bold]Path:[/bold]   {model.path}
[bold]Format:[/bold] {model.format}
[bold]Repo:[/bold]   {repo_info}
[bold]SHA256:[/bold] {sha256_display}
[bold]Size:[/bold]   {size_info}
[bold]Status:[/bold] {path_status}"""

            panel = Panel(content, title=f"[cyan]{model.name}[/cyan]", border_style="cyan", padding=(0, 1))
            console.print(panel)

        console.print()
        console.print(f"[dim]Total: {len(models)} model(s)[/dim]\n")
    else:
        # Compact mode: separate tables by model type
        from rich.align import Align
        from pathlib import Path

        # Categorize models
        gguf_models = []
        amx_models = []
        gpu_models = []

        for model in models:
            if model.format == "gguf":
                gguf_models.append(model)
            elif model.format == "safetensors" and model.path_exists():
                is_amx, numa_count = is_amx_weights(model.path)
                if is_amx:
                    amx_models.append((model, numa_count))
                else:
                    gpu_models.append(model)
            else:
                gpu_models.append(model)

        # Pre-analyze GPU MoE models concurrently if enabled
        moe_results = {}
        moe_failed_models = []  # Track models that failed MoE analysis
        if show_moe and analyze_moe_model and gpu_models:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading

            # Collect GPU models that need MoE analysis
            # Priority: use cached MoE info from UserModel, only analyze if is_moe is None
            models_to_analyze = []
            models_need_update = []  # Track models that need registry update

            for model in gpu_models:
                # Check if MoE info is already cached in UserModel (and not using --no-cache)
                if not no_cache and model.is_moe is not None:
                    # Use cached info from UserModel
                    if model.is_moe:
                        moe_results[model.name] = {
                            "is_moe": True,
                            "num_experts": model.moe_num_experts,
                            "num_experts_per_tok": model.moe_num_experts_per_tok,
                            "cached": True,
                        }
                    # If is_moe is False, don't add to moe_results
                else:
                    # Need to analyze (is_moe is None or --no-cache)
                    path_obj = Path(model.path)
                    models_to_analyze.append((model.name, str(path_obj)))
                    models_need_update.append(model)

            if models_to_analyze:
                # Use lock for thread-safe console output
                print_lock = threading.Lock()
                completed_count = [0]  # Use list to allow modification in nested function

                def analyze_with_progress(model_info):
                    model_name, model_path = model_info
                    try:
                        with print_lock:
                            console.print(f"[dim]Analyzing MoE: {model_name}...[/dim]")
                        result = analyze_moe_model(model_path, use_cache=not no_cache)

                        # Check if analysis returned valid results
                        if result is None or result.get("num_experts", 0) == 0:
                            with print_lock:
                                completed_count[0] += 1
                                console.print(
                                    f"[dim]✗ [{completed_count[0]}/{len(models_to_analyze)}] {model_name} - Not a MoE model or analysis failed[/dim]"
                                )
                            return (model_name, None, "Not a MoE model or analysis failed")

                        with print_lock:
                            completed_count[0] += 1
                            cached_tag = "[green](cached)[/green]" if result and result.get("cached") else ""
                            console.print(
                                f"[dim]✓ [{completed_count[0]}/{len(models_to_analyze)}] {model_name} {cached_tag}[/dim]"
                            )
                        return (model_name, result, None)
                    except Exception as e:
                        with print_lock:
                            completed_count[0] += 1
                            error_msg = str(e)[:80]
                            console.print(
                                f"[dim]✗ [{completed_count[0]}/{len(models_to_analyze)}] {model_name} - Error: {error_msg}[/dim]"
                            )
                        return (model_name, None, error_msg)

                if no_cache:
                    console.print(f"\n[yellow]Force re-analyzing (--no-cache): ignoring cached results[/yellow]")
                console.print(
                    f"\n[cyan]Analyzing {len(models_to_analyze)} MoE model(s) with {min(16, len(models_to_analyze))} threads...[/cyan]\n"
                )

                # Analyze concurrently with up to 16 workers
                with ThreadPoolExecutor(max_workers=16) as executor:
                    futures = {
                        executor.submit(analyze_with_progress, model_info): model_info
                        for model_info in models_to_analyze
                    }

                    for future in as_completed(futures):
                        model_name, result, error = future.result()
                        if error:
                            # Find the model object
                            failed_model = next((m for m in gpu_models if m.name == model_name), None)
                            if failed_model:
                                moe_failed_models.append((failed_model, error))
                                # Update model registry: mark as non-MoE
                                registry.update_model(model_name, {"is_moe": False})
                        else:
                            moe_results[model_name] = result
                            # Update model registry with MoE info
                            if result and result.get("is_moe"):
                                registry.update_model(
                                    model_name,
                                    {
                                        "is_moe": True,
                                        "moe_num_experts": result.get("num_experts"),
                                        "moe_num_experts_per_tok": result.get("num_experts_per_tok"),
                                    },
                                )
                            else:
                                registry.update_model(model_name, {"is_moe": False})

                console.print(f"\n[green]✓ MoE analysis complete[/green]\n")

                # Remove failed models from gpu_models list
                if moe_failed_models:
                    failed_names = {m.name for m, _ in moe_failed_models}
                    gpu_models = [m for m in gpu_models if m.name not in failed_names]

        # Separate MoE and non-MoE GPU models
        moe_gpu_models = []
        non_moe_gpu_models = []
        for model in gpu_models:
            if model.name in moe_results:
                moe_gpu_models.append(model)
            else:
                non_moe_gpu_models.append(model)

        # Count failed MoE models (these are also non-MoE)
        total_non_moe_count = len(non_moe_gpu_models) + len(moe_failed_models)

        # Filter display based on --all flag
        if not all_models:
            # Default: only show MoE models
            gpu_models_to_display = moe_gpu_models
            show_failed_table = False
        else:
            # --all: show all GPU models including non-MoE and failed
            gpu_models_to_display = gpu_models
            show_failed_table = True
            total_non_moe_count = 0  # Don't show hint when displaying all

        # Helper function to create table rows
        def format_model_row(model, moe_info=None, numa_count=None):
            from kt_kernel.cli.utils.model_scanner import format_size

            # Calculate size
            if model.path_exists():
                path_obj = Path(model.path)
                try:
                    if model.format == "safetensors":
                        files = list(path_obj.glob("*.safetensors"))
                    else:
                        files = list(path_obj.glob("*.gguf"))

                    total_size = sum(f.stat().st_size for f in files if f.exists())
                    size_display = format_size(total_size)
                except:
                    size_display = "[dim]-[/dim]"
            else:
                size_display = "[dim]-[/dim]"

            # Format repo info
            if model.repo_id:
                repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
                repo_display = f"{repo_abbr}:{model.repo_id}"
            else:
                repo_display = "[dim]-[/dim]"

            # Format SHA256 status
            sha256_display = SHA256_STATUS_MAP.get(model.sha256_status, model.sha256_status)

            row = [model.name, model.path, size_display]

            # Add type-specific columns
            if numa_count is not None:
                # AMX model
                row.append(f"[yellow]{numa_count} NUMA[/yellow]")
            elif moe_info:
                # GPU MoE model
                experts_display = f"[yellow]{moe_info['num_experts']}[/yellow]"
                activated_display = f"[green]{moe_info['num_experts_per_tok']}[/green]"
                moe_total_display = f"[cyan]{size_display}[/cyan]"
                row.extend([experts_display, activated_display, moe_total_display])
            elif show_moe and analyze_moe_model and model.format == "safetensors":
                # GPU non-MoE model
                row.extend(["[dim]-[/dim]", "[dim]-[/dim]", "[dim]-[/dim]"])

            row.extend([repo_display, sha256_display])
            return row

        # Display tables
        title = Align.center(f"[bold cyan]{t('model_registered_models_title')}[/bold cyan]")
        console.print(title)
        console.print()

        # Table 1: GGUF Models (Llamafile)
        if gguf_models:
            console.print("[bold yellow]GGUF Models (Llamafile)[/bold yellow]")
            table = Table(show_header=True, header_style="bold")
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column(t("model_column_name"), style="cyan", no_wrap=True)
            table.add_column("Path", style="dim", overflow="fold")
            table.add_column("Total", justify="right")
            table.add_column(t("model_column_repo"), style="dim", overflow="fold")
            table.add_column(t("model_column_sha256"), justify="center")

            for i, model in enumerate(gguf_models, 1):
                row = [str(i)] + format_model_row(model)
                table.add_row(*row)

            console.print(table)
            console.print()

        # Table 2: AMX Models
        if amx_models:
            from kt_kernel.cli.utils.model_scanner import format_size
            import json

            console.print("[bold magenta]AMX Models (CPU)[/bold magenta]")
            table = Table(show_header=True, header_style="bold", show_lines=False)
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column(t("model_column_name"), style="cyan", no_wrap=True)
            table.add_column("Path", style="dim", overflow="fold")
            table.add_column("Total", justify="right")
            table.add_column("Method", justify="center", style="yellow")
            table.add_column("NUMA", justify="center", style="green")
            table.add_column("Source", style="dim", overflow="fold")

            # Build reverse map: AMX model ID -> GPU models using it
            amx_used_by_gpu = {}  # {amx_model_id: [gpu_model_names]}
            for model, _ in amx_models:
                if model.gpu_model_ids:
                    # This AMX is linked to these GPU models
                    gpu_names = []
                    for gpu_id in model.gpu_model_ids:
                        # Find GPU model by ID
                        for gpu_model in gpu_models:
                            if gpu_model.id == gpu_id:
                                gpu_names.append(gpu_model.name)
                                break
                    if gpu_names:
                        amx_used_by_gpu[model.id] = gpu_names

            for i, (model, numa_count) in enumerate(amx_models, 1):
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

                # Read AMX metadata from config.json (fallback if not in UserModel)
                method_from_config = None
                numa_from_config = None
                if model.path_exists():
                    config_path = Path(model.path) / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, "r", encoding="utf-8") as f:
                                config = json.load(f)
                                amx_quant = config.get("amx_quantization", {})
                                if amx_quant.get("converted"):
                                    method_from_config = amx_quant.get("method")
                                    numa_from_config = amx_quant.get("numa_count")
                        except:
                            pass

                # AMX-specific metadata (priority: UserModel > config.json > detected numa_count)
                method_display = (
                    model.amx_quant_method.upper()
                    if model.amx_quant_method
                    else method_from_config.upper() if method_from_config else "[dim]?[/dim]"
                )
                numa_display = (
                    str(model.amx_numa_nodes)
                    if model.amx_numa_nodes
                    else (
                        str(numa_from_config) if numa_from_config else str(numa_count) if numa_count else "[dim]?[/dim]"
                    )
                )
                source_display = model.amx_source_model if model.amx_source_model else "[dim]-[/dim]"

                table.add_row(
                    str(i), model.name, model.path, size_display, method_display, numa_display, source_display
                )

                # Add linked GPU models info below this AMX model
                if model.id in amx_used_by_gpu:
                    gpu_list = amx_used_by_gpu[model.id]
                    gpu_names_str = ", ".join([f"[dim]{name}[/dim]" for name in gpu_list])
                    # Create a sub-row with empty cells except for the first column (7 columns total with #)
                    sub_row = ["", f"  [dim]↳ GPU: {gpu_names_str}[/dim]", "", "", "", "", ""]
                    table.add_row(*sub_row, style="dim")

            console.print(table)
            console.print()

        # Table 3: GPU Models (Safetensors)
        if gpu_models_to_display:
            console.print("[bold green]GPU Models (Safetensors)[/bold green]")
            table = Table(show_header=True, header_style="bold", show_lines=False)
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column(t("model_column_name"), style="cyan", no_wrap=True)
            table.add_column("Path", style="dim", overflow="fold")
            table.add_column("Total", justify="right")

            if show_moe and analyze_moe_model:
                table.add_column("Exps", justify="center", style="yellow")
                table.add_column("Act", justify="center", style="green")
                table.add_column("MoE Size", justify="right", style="cyan")

            table.add_column(t("model_column_repo"), style="dim", overflow="fold")
            table.add_column(t("model_column_sha256"), justify="center")

            # Build a map of GPU model UUID -> attached CPU models
            attached_cpu_models = {}  # {gpu_model_id: [(cpu_model, type)]}
            for model in gguf_models:
                if model.gpu_model_ids:
                    for gpu_id in model.gpu_model_ids:
                        if gpu_id not in attached_cpu_models:
                            attached_cpu_models[gpu_id] = []
                        attached_cpu_models[gpu_id].append((model, "GGUF"))

            for model, numa_count in amx_models:
                if model.gpu_model_ids:
                    for gpu_id in model.gpu_model_ids:
                        if gpu_id not in attached_cpu_models:
                            attached_cpu_models[gpu_id] = []
                        attached_cpu_models[gpu_id].append((model, "AMX"))

            for i, model in enumerate(gpu_models_to_display, 1):
                moe_info = moe_results.get(model.name) if show_moe and analyze_moe_model else None
                row = [str(i)] + format_model_row(model, moe_info=moe_info)
                table.add_row(*row)

                # Add attached CPU models info below this GPU model (using UUID matching)
                if model.id in attached_cpu_models:
                    cpu_list = attached_cpu_models[model.id]
                    cpu_names = ", ".join([f"[dim]{m.name} ({t})[/dim]" for m, t in cpu_list])
                    # Create a sub-row with empty cells except for the first column
                    num_cols = len(row)
                    sub_row = ["", f"  [dim]↳ CPU: {cpu_names}[/dim]"] + [""] * (num_cols - 2)
                    table.add_row(*sub_row, style="dim")

            console.print(table)
            console.print()

        # Table 4: Failed MoE Analysis (only show with --all)
        if show_failed_table and moe_failed_models:
            console.print("[bold red]Failed MoE Analysis[/bold red]")
            console.print("[yellow]These models may not be MoE models or have analysis errors:[/yellow]\n")
            table = Table(show_header=True, header_style="bold")
            table.add_column("#", justify="right", style="cyan", no_wrap=True)
            table.add_column(t("model_column_name"), style="red", no_wrap=True)
            table.add_column("Path", style="dim", overflow="fold")
            table.add_column("Total", justify="right")
            table.add_column("Error", style="yellow", overflow="fold")

            for i, (model, error) in enumerate(moe_failed_models, 1):
                from kt_kernel.cli.utils.model_scanner import format_size

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

                table.add_row(str(i), model.name, model.path, size_display, error)

            console.print(table)
            console.print()

        # Show hint if non-MoE models are hidden (display before summary)
        if total_non_moe_count > 0:
            hint_text = t("model_non_moe_hidden_hint", count=total_non_moe_count)
            console.print(f"[dim]{hint_text}[/dim]")
            console.print()

        # Summary
        total_count = len(gguf_models) + len(amx_models) + len(gpu_models)
        failed_count = len(moe_failed_models)
        if failed_count > 0:
            console.print(
                f"[dim]Total: {total_count} model(s) | GGUF: {len(gguf_models)} | AMX: {len(amx_models)} | GPU: {len(gpu_models)} | [red]Failed: {failed_count}[/red][/dim]\n"
            )
        else:
            console.print(
                f"[dim]Total: {total_count} model(s) | GGUF: {len(gguf_models)} | AMX: {len(amx_models)} | GPU: {len(gpu_models)}[/dim]\n"
            )

        # Show usage hints (only in non-verbose mode)
        if not verbose and models:
            console.print(f"[bold cyan]{t('model_usage_title')}[/bold cyan]")
            console.print(f"  {t('model_usage_info'):<17} [cyan]kt model info <name>[/cyan]")
            console.print(f"  {t('model_usage_edit'):<17} [cyan]kt model edit <name>[/cyan]")
            console.print(f"  {t('model_usage_verify'):<17} [cyan]kt model verify <name>[/cyan]")
            console.print(f"  {t('model_usage_quant'):<17} [cyan]kt quant <name>[/cyan]")
            console.print(f"  {t('model_usage_run'):<17} [cyan]kt run <name>[/cyan]")
            console.print()
            console.print(f"  {t('model_usage_scan'):<17} [cyan]kt model scan[/cyan]")
            console.print(f"  {t('model_usage_add'):<17} [cyan]kt model add <path>[/cyan]")
            console.print()


@app.command(name="clear-cache")
def clear_cache() -> None:
    """Clear MoE analysis cache."""
    from pathlib import Path
    import json

    cache_file = Path.home() / ".ktransformers" / "cache" / "moe_analysis.json"

    if not cache_file.exists():
        console.print()
        console.print("[dim]No MoE cache found.[/dim]")
        console.print()
        return

    # Read cache to count entries
    try:
        with open(cache_file, "r") as f:
            cache_data = json.load(f)
        cache_count = len(cache_data)
    except Exception:
        cache_count = 0

    if cache_count == 0:
        console.print()
        console.print("[dim]MoE cache is empty.[/dim]")
        console.print()
        return

    console.print()
    console.print(f"[yellow]Found {cache_count} cached model(s) in:[/yellow]")
    console.print(f"  {cache_file}")
    console.print()

    if confirm("Clear all MoE analysis cache?", default=False):
        cache_file.unlink()
        console.print(f"[green]✓ Cleared cache for {cache_count} model(s)[/green]")
    else:
        console.print("[dim]Cache clear cancelled.[/dim]")

    console.print()


@app.command(name="path-list")
def path_list() -> None:
    """List all configured model storage paths."""
    settings = get_settings()
    model_paths = settings.get_model_paths()

    console.print()
    console.print(f"[bold]{t('model_storage_paths_title')}:[/bold]\n")

    for i, path in enumerate(model_paths, 1):
        marker = "[green]✓[/green]" if path.exists() else "[red]✗[/red]"
        console.print(f"  {marker} [{i}] {path}")

    console.print()


@app.command(name="link-cpu")
def link_cpu(
    cpu_model: str = typer.Argument(..., help="Name of the CPU model (GGUF/AMX)"),
    gpu_models: List[str] = typer.Argument(..., help="Name(s) of GPU model(s) to link with"),
) -> None:
    """Link a CPU model (GGUF/AMX) with one or more GPU models for joint startup."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()

    # Check if CPU model exists
    cpu_model_obj = registry.get_model(cpu_model)
    if not cpu_model_obj:
        print_error(f"CPU model '{cpu_model}' not found in registry.")
        console.print()
        console.print(f"  Use [cyan]kt model list[/cyan] to see registered models")
        console.print()
        raise typer.Exit(1)

    # Check if it's actually a CPU model (GGUF or AMX)
    if cpu_model_obj.format == "safetensors":
        # Check if it's AMX by looking for .numa. pattern
        is_amx, _ = is_amx_weights(cpu_model_obj.path)
        if not is_amx:
            print_error(f"Model '{cpu_model}' is a GPU model (safetensors), not a CPU model.")
            console.print()
            console.print(f"  Only GGUF and AMX models can be linked to GPU models")
            console.print()
            raise typer.Exit(1)

    # Verify all GPU models exist and collect their UUIDs
    gpu_model_uuids = []
    missing_models = []
    for gpu_name in gpu_models:
        gpu_model_obj = registry.get_model(gpu_name)
        if not gpu_model_obj:
            missing_models.append(gpu_name)
        else:
            gpu_model_uuids.append(gpu_model_obj.id)

    if missing_models:
        print_error(f"GPU model(s) not found: {', '.join(missing_models)}")
        console.print()
        console.print(f"  Use [cyan]kt model list[/cyan] to see registered models")
        console.print()
        raise typer.Exit(1)

    # Update the CPU model with GPU links (using UUIDs for stability)
    registry.update_model(cpu_model, {"gpu_model_ids": gpu_model_uuids})

    console.print()
    print_success(f"Linked CPU model '{cpu_model}' with GPU model(s):")
    for gpu_name in gpu_models:
        console.print(f"  [green]✓[/green] {gpu_name}")
    console.print()
    console.print(f"  View the relationship with [cyan]kt model list[/cyan]")
    console.print()


@app.command(name="unlink-cpu")
def unlink_cpu(
    cpu_model: str = typer.Argument(..., help="Name of the CPU model to unlink"),
) -> None:
    """Remove GPU model links from a CPU model."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()

    # Check if model exists
    model = registry.get_model(cpu_model)
    if not model:
        print_error(f"Model '{cpu_model}' not found in registry.")
        console.print()
        raise typer.Exit(1)

    if not model.gpu_model_ids:
        console.print()
        console.print(f"[yellow]Model '{cpu_model}' has no GPU links.[/yellow]")
        console.print()
        return

    # Remove links
    registry.update_model(cpu_model, {"gpu_model_ids": None})

    console.print()
    print_success(f"Removed all GPU links from '{cpu_model}'")
    console.print()


@app.command(name="path-add")
def path_add(
    path: str = typer.Argument(..., help="Path to add"),
) -> None:
    """Add a new model storage path."""
    # Expand user home directory
    path = os.path.expanduser(path)

    # Check if path exists or can be created
    path_obj = Path(path)
    if not path_obj.exists():
        console.print(f"[yellow]{t('model_path_not_exist', path=path)}[/yellow]")
        if confirm(t("model_create_directory", path=path), default=True):
            try:
                path_obj.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]✓[/green] {t('model_created_directory', path=path)}")
            except (OSError, PermissionError) as e:
                print_error(t("model_create_dir_failed", error=str(e)))
                raise typer.Exit(1)
        else:
            raise typer.Abort()

    # Add to configuration
    settings = get_settings()
    settings.add_model_path(path)
    print_success(t("model_path_added", path=path))


@app.command(name="path-remove")
def path_remove(
    path: str = typer.Argument(..., help="Path to remove"),
) -> None:
    """Remove a model storage path from configuration."""
    # Expand user home directory
    path = os.path.expanduser(path)

    settings = get_settings()
    if settings.remove_model_path(path):
        print_success(t("model_path_removed", path=path))
    else:
        print_error(t("model_path_not_found", path=path))
        raise typer.Exit(1)


@app.command(name="scan")
def scan(
    min_size: float = typer.Option(2.0, "--min-size", help="Minimum model file size in GB (default: 2.0)"),
    max_depth: int = typer.Option(6, "--max-depth", help="Maximum search depth (default: 6)"),
) -> None:
    """Perform global scan for models and add new ones to registry."""
    from kt_kernel.cli.utils.model_discovery import discover_and_register_global, format_discovery_summary
    from kt_kernel.cli.config.settings import get_settings

    settings = get_settings()
    lang = settings.get("general.language", "en")

    console.print()
    if lang == "zh":
        print_info("全局扫描模型权重")
        console.print()
    else:
        print_info("Global Model Scan")
        console.print()

    try:
        total_found, new_found, registered = discover_and_register_global(
            min_size_gb=min_size, max_depth=max_depth, show_progress=True, lang=lang
        )

        format_discovery_summary(
            total_found=total_found,
            new_found=new_found,
            registered=registered,
            lang=lang,
            show_models=True,
            max_show=20,
        )

        if new_found > 0:
            console.print()
            if lang == "zh":
                console.print("[dim]下一步:[/dim]")
                console.print(f"  • 查看模型列表: [cyan]kt model list[/cyan]")
                console.print(f"  • 编辑模型信息: [cyan]kt model edit <name>[/cyan]")
                console.print(f"  • 验证模型: [cyan]kt model verify <name>[/cyan]")
            else:
                console.print("[dim]Next steps:[/dim]")
                console.print(f"  • View model list: [cyan]kt model list[/cyan]")
                console.print(f"  • Edit model info: [cyan]kt model edit <name>[/cyan]")
                console.print(f"  • Verify models: [cyan]kt model verify <name>[/cyan]")
            console.print()

    except Exception as e:
        print_error(f"Scan failed: {e}")
        raise typer.Exit(1)


@app.command(name="add")
def add_model(
    path: str = typer.Argument(..., help="Path to scan for models"),
) -> None:
    """Scan a directory and add all found models to the registry."""
    from pathlib import Path
    from kt_kernel.cli.utils.model_discovery import discover_and_register_path
    from kt_kernel.cli.config.settings import get_settings

    settings = get_settings()
    lang = settings.get("general.language", "en")

    # Expand and validate path
    path_obj = Path(os.path.expanduser(path)).resolve()

    if not path_obj.exists():
        print_error(f"Path does not exist: {path_obj}")
        raise typer.Exit(1)

    if not path_obj.is_dir():
        print_error(f"Not a directory: {path_obj}")
        raise typer.Exit(1)

    # Scan and register models
    console.print()
    try:
        total_found, new_found, registered = discover_and_register_path(
            path=str(path_obj), min_size_gb=2.0, existing_paths=None, show_progress=True, lang=lang
        )

        console.print()
        if new_found == 0:
            if total_found > 0:
                if lang == "zh":
                    console.print(f"[yellow]在此路径找到 {total_found} 个模型，但所有模型均已在列表中[/yellow]")
                else:
                    console.print(
                        f"[yellow]Found {total_found} models in this path, but all already in the list[/yellow]"
                    )
            else:
                if lang == "zh":
                    console.print("[yellow]未找到模型[/yellow]")
                    console.print()
                    console.print("  支持的格式: *.gguf, *.safetensors (需要 config.json)")
                else:
                    console.print("[yellow]No models found[/yellow]")
                    console.print()
                    console.print("  Supported formats: *.gguf, *.safetensors (with config.json)")
        else:
            if lang == "zh":
                console.print(
                    f"[green]✓[/green] 在此路径找到 {total_found} 个模型，成功添加 {len(registered)} 个新模型"
                )
            else:
                console.print(
                    f"[green]✓[/green] Found {total_found} models in this path, added {len(registered)} new models"
                )

            if registered:
                console.print()
                if lang == "zh":
                    console.print("[dim]新添加的模型:[/dim]")
                else:
                    console.print("[dim]Newly added models:[/dim]")

                for model in registered:
                    console.print(f"  • {model.name} ({model.format})")
                    console.print(f"    [dim]{model.path}[/dim]")

        console.print()

    except Exception as e:
        print_error(f"Failed to scan path: {e}")
        raise typer.Exit(1)


@app.command(name="edit")
def edit_model(
    name: Optional[str] = typer.Argument(
        None, help="Name of model to edit (optional - will show selection if not provided)"
    ),
) -> None:
    """Edit model information interactively."""
    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel
    from rich.table import Table
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()

    # If no name provided, show interactive selection
    if name is None:
        all_models = registry.list_models()

        # Filter to only show MoE GPU models (safetensors that are not AMX)
        moe_models = []
        for m in all_models:
            if m.format == "safetensors":
                is_amx_model, _ = is_amx_weights(m.path)
                if not is_amx_model:
                    moe_models.append(m)

        if not moe_models:
            print_error(t("model_edit_no_models"))
            console.print()
            console.print(f"  {t('model_edit_add_hint_scan')} [cyan]kt model scan[/cyan]")
            console.print(f"  {t('model_edit_add_hint_add')} [cyan]kt model add <path>[/cyan]")
            console.print()
            raise typer.Exit(1)

        # Display models table with # column
        console.print()
        console.print(f"[bold cyan]{t('model_edit_select_title')}[/bold cyan]")
        console.print()

        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Format", style="dim")
        table.add_column("Path", style="dim", overflow="fold")

        for i, model_item in enumerate(moe_models, 1):
            table.add_row(str(i), model_item.name, model_item.format, model_item.path)

        console.print(table)
        console.print()

        from rich.prompt import IntPrompt

        choice = IntPrompt.ask(t("model_edit_select_model"), default=1, show_choices=False)

        if choice < 1 or choice > len(moe_models):
            print_error(t("model_edit_invalid_choice"))
            raise typer.Exit(1)

        model = moe_models[choice - 1]
    else:
        # Load model by name
        model = registry.get_model(name)
        if not model:
            print_error(t("model_edit_not_found", name=name))
            console.print()
            console.print(f"  {t('model_edit_list_hint')} [cyan]kt model list[/cyan]")
            console.print()
            raise typer.Exit(1)

    # Keep track of original values to detect changes
    original_name = model.name
    original_repo_type = model.repo_type
    original_repo_id = model.repo_id
    original_gpu_model_ids = model.gpu_model_ids.copy() if model.gpu_model_ids else None

    # Working copy for edits (not saved until user confirms)
    edited_name = model.name
    edited_repo_type = model.repo_type
    edited_repo_id = model.repo_id
    edited_gpu_model_ids = model.gpu_model_ids.copy() if model.gpu_model_ids else None

    has_changes = False

    while True:
        # Display current configuration (show edited values)
        console.print()
        console.print(f"[bold cyan]{t('model_edit_current_config')}[/bold cyan]\n")

        # Format SHA256 status (from original model)
        sha256_display = SHA256_STATUS_MAP_PLAIN.get(model.sha256_status, model.sha256_status)

        # Check if this is a CPU model (GGUF or AMX)
        is_cpu_model = model.format == "gguf"
        if not is_cpu_model and model.format == "safetensors":
            is_amx, _ = is_amx_weights(model.path)
            is_cpu_model = is_amx

        # Format GPU links info (for CPU models)
        gpu_links_info = ""
        if is_cpu_model and edited_gpu_model_ids:
            gpu_names = []
            for gpu_id in edited_gpu_model_ids:
                gpu_obj = registry.get_model_by_id(gpu_id)
                if gpu_obj:
                    gpu_names.append(gpu_obj.name)
                else:
                    gpu_names.append(f"[dim red]{gpu_id[:8]}... (deleted)[/dim red]")
            gpu_links_info = f"\n[bold]{t('model_edit_gpu_links')}[/bold]  {', '.join(gpu_names)}"

        content = f"""[bold]Name:[/bold]       {edited_name}
[bold]Path:[/bold]       {model.path}
[bold]Format:[/bold]     {model.format}
[bold]Repo Type:[/bold]  {edited_repo_type or '-'}
[bold]Repo ID:[/bold]    {edited_repo_id or '-'}
[bold]SHA256:[/bold]     {sha256_display}{gpu_links_info}"""

        panel = Panel(content, border_style="cyan", padding=(0, 1))
        console.print(panel)
        console.print()

        # Check if there are any changes
        has_changes = (
            edited_name != original_name
            or edited_repo_type != original_repo_type
            or edited_repo_id != original_repo_id
            or edited_gpu_model_ids != original_gpu_model_ids
        )

        # Show menu
        console.print(f"[bold]{t('model_edit_what_to_edit')}[/bold]")
        console.print("  [1] " + t("model_edit_option_name"))
        console.print("  [2] " + t("model_edit_option_repo"))
        console.print("  [3] " + t("model_edit_option_delete"))
        if is_cpu_model:
            console.print("  [4] " + t("model_edit_manage_gpu_links"))
            save_option = "5"
            cancel_option = "6"
            console.print(
                f"  [{save_option}] {t('model_edit_save_changes')}"
                + (
                    f" [cyan]{t('model_edit_has_changes')}[/cyan]"
                    if has_changes
                    else f" [dim]{t('model_edit_no_changes')}[/dim]"
                )
            )
            console.print(f"  [{cancel_option}] " + t("model_edit_option_cancel"))
            console.print()
            choice = Prompt.ask(t("model_edit_choice_prompt"), choices=["1", "2", "3", "4", "5", "6"], default="6")
        else:
            save_option = "4"
            cancel_option = "5"
            console.print(
                f"  [{save_option}] {t('model_edit_save_changes')}"
                + (
                    f" [cyan]{t('model_edit_has_changes')}[/cyan]"
                    if has_changes
                    else f" [dim]{t('model_edit_no_changes')}[/dim]"
                )
            )
            console.print(f"  [{cancel_option}] " + t("model_edit_option_cancel"))
            console.print()
            choice = Prompt.ask(t("model_edit_choice_prompt"), choices=["1", "2", "3", "4", "5"], default="5")

        if choice == "1":
            # Edit name (update working copy only)
            console.print()
            new_name = Prompt.ask(t("model_edit_new_name"), default=edited_name)

            if new_name != edited_name:
                # Check for conflict (excluding both original and edited names)
                if new_name != original_name and registry.check_name_conflict(new_name, exclude_name=original_name):
                    print_error(t("model_edit_name_conflict", name=new_name))
                    continue

                edited_name = new_name
                console.print()
                print_info(f"[dim]{t('model_edit_name_pending')}[/dim]")

        elif choice == "2":
            # Edit repo configuration (update working copy only)
            console.print()
            console.print(t("model_edit_repo_type_prompt"))
            console.print("  [1] HuggingFace")
            console.print("  [2] ModelScope")
            console.print("  [3] " + t("model_edit_repo_remove"))
            console.print()

            repo_choice = Prompt.ask(t("model_edit_choice_prompt"), choices=["1", "2", "3"], default="3")

            if repo_choice == "3":
                # Remove repo
                edited_repo_type = None
                edited_repo_id = None
                console.print()
                print_info(f"[dim]{t('model_edit_repo_remove_pending')}[/dim]")
            else:
                # Set repo
                repo_type = "huggingface" if repo_choice == "1" else "modelscope"
                example = "deepseek-ai/DeepSeek-V3" if repo_choice == "1" else "deepseek/DeepSeek-V3"

                current_default = edited_repo_id if edited_repo_id and edited_repo_type == repo_type else ""
                repo_id = Prompt.ask(
                    t("model_edit_repo_id_prompt", example=example),
                    default=current_default if current_default else None,
                )

                edited_repo_type = repo_type
                edited_repo_id = repo_id
                console.print()
                print_info(f"[dim]{t('model_edit_repo_update_pending')}[/dim]")

        elif choice == "3":
            # Delete model
            console.print()
            console.print(f"[bold yellow]{t('model_edit_delete_warning')}[/bold yellow]")
            console.print(f"  {t('model_edit_delete_note')}")
            console.print()

            if Confirm.ask(t("model_edit_delete_confirm", name=model.name), default=False):
                registry.remove_model(model.name)
                console.print()
                print_success(t("model_edit_deleted", name=model.name))
                console.print()
                return
            else:
                console.print()
                print_info(t("model_edit_delete_cancelled"))

        elif choice == "4" and is_cpu_model:
            # Manage GPU Links (only for CPU models) - update working copy
            console.print()
            console.print(f"[bold cyan]{t('model_edit_gpu_links_title', name=edited_name)}[/bold cyan]")
            console.print()

            # Show current links (from edited values)
            if edited_gpu_model_ids:
                console.print(f"[bold]{t('model_edit_current_gpu_links')}[/bold]")
                for i, gpu_id in enumerate(edited_gpu_model_ids, 1):
                    gpu_obj = registry.get_model_by_id(gpu_id)
                    if gpu_obj:
                        console.print(f"  [{i}] {gpu_obj.name}")
                    else:
                        console.print(f"  [{i}] [red]{gpu_id[:8]}... (deleted)[/red]")
                console.print()
            else:
                console.print(f"[dim]{t('model_edit_no_gpu_links')}[/dim]")
                console.print()

            console.print(f"{t('model_edit_gpu_options')}")
            console.print(f"  [1] {t('model_edit_gpu_add')}")
            console.print(f"  [2] {t('model_edit_gpu_remove')}")
            console.print(f"  [3] {t('model_edit_gpu_clear')}")
            console.print(f"  [4] {t('model_edit_gpu_back')}")
            console.print()

            link_choice = Prompt.ask(t("model_edit_gpu_choose_option"), choices=["1", "2", "3", "4"], default="4")

            if link_choice == "1":
                # Add GPU link
                # Get all GPU models (safetensors that are not AMX)
                all_models = registry.list_models()
                available_gpu_models = []
                for m in all_models:
                    if m.format == "safetensors":
                        is_amx_model, _ = is_amx_weights(m.path)
                        if not is_amx_model:
                            available_gpu_models.append(m)

                if not available_gpu_models:
                    console.print()
                    console.print(f"[yellow]{t('model_edit_gpu_none_available')}[/yellow]")
                    console.print()
                else:
                    console.print()
                    console.print(f"{t('model_edit_gpu_available_models')}")
                    for i, gpu_m in enumerate(available_gpu_models, 1):
                        already_linked = edited_gpu_model_ids and gpu_m.id in edited_gpu_model_ids
                        status = f" [dim]{t('model_edit_gpu_already_linked')}[/dim]" if already_linked else ""
                        console.print(f"  [{i}] {gpu_m.name}{status}")
                    console.print()

                    gpu_choice = Prompt.ask(t("model_edit_gpu_enter_number"), default="0")
                    try:
                        gpu_idx = int(gpu_choice) - 1
                        if 0 <= gpu_idx < len(available_gpu_models):
                            selected_gpu = available_gpu_models[gpu_idx]

                            # Add to edited_gpu_model_ids
                            current_ids = list(edited_gpu_model_ids) if edited_gpu_model_ids else []
                            if selected_gpu.id not in current_ids:
                                current_ids.append(selected_gpu.id)
                                edited_gpu_model_ids = current_ids
                                console.print()
                                print_info(f"[dim]{t('model_edit_gpu_link_pending', name=selected_gpu.name)}[/dim]")
                            else:
                                console.print()
                                console.print(f"[yellow]{t('model_edit_gpu_already_exists')}[/yellow]")
                        else:
                            console.print()
                            console.print(f"[red]{t('model_edit_gpu_invalid_choice')}[/red]")
                    except ValueError:
                        console.print()
                        console.print(f"[red]{t('model_edit_gpu_invalid_input')}[/red]")

            elif link_choice == "2":
                # Remove GPU link
                if not edited_gpu_model_ids:
                    console.print()
                    console.print(f"[yellow]{t('model_edit_gpu_none_to_remove')}[/yellow]")
                    console.print()
                else:
                    console.print()
                    console.print(f"{t('model_edit_gpu_choose_to_remove')}")
                    gpu_list = []
                    for i, gpu_id in enumerate(edited_gpu_model_ids, 1):
                        gpu_obj = registry.get_model_by_id(gpu_id)
                        gpu_name = gpu_obj.name if gpu_obj else f"{gpu_id[:8]}... (deleted)"
                        gpu_list.append((gpu_id, gpu_name))
                        console.print(f"  [{i}] {gpu_name}")
                    console.print()

                    remove_choice = Prompt.ask(t("model_edit_gpu_enter_to_remove"), default="0")
                    try:
                        remove_idx = int(remove_choice) - 1
                        if 0 <= remove_idx < len(gpu_list):
                            removed_id, removed_name = gpu_list[remove_idx]
                            new_ids = [gid for gid in edited_gpu_model_ids if gid != removed_id]
                            edited_gpu_model_ids = new_ids if new_ids else None
                            console.print()
                            print_info(f"[dim]{t('model_edit_gpu_remove_pending', name=removed_name)}[/dim]")
                        else:
                            console.print()
                            console.print(f"[red]{t('model_edit_gpu_invalid_choice')}[/red]")
                    except ValueError:
                        console.print()
                        console.print(f"[red]{t('model_edit_gpu_invalid_input')}[/red]")

            elif link_choice == "3":
                # Clear all GPU links
                if not edited_gpu_model_ids:
                    console.print()
                    console.print(f"[yellow]{t('model_edit_gpu_none_to_clear')}[/yellow]")
                    console.print()
                else:
                    if Confirm.ask(t("model_edit_gpu_clear_confirm"), default=False):
                        edited_gpu_model_ids = None
                        console.print()
                        print_info(f"[dim]{t('model_edit_gpu_clear_pending')}[/dim]")
                    else:
                        console.print()
                        print_info(t("model_edit_cancelled_short"))

        elif choice == save_option:
            # Save changes
            if not has_changes:
                console.print()
                print_info(f"[dim]{t('model_edit_no_changes_to_save')}[/dim]")
                continue

            console.print()
            console.print(f"[bold cyan]{t('model_edit_saving')}[/bold cyan]")
            console.print()

            # Determine if repo info changed (for verification prompt)
            repo_changed = (original_repo_id is None and edited_repo_id is not None) or (
                original_repo_id != edited_repo_id
            )

            # Build updates dict
            updates = {}
            if edited_name != original_name:
                updates["name"] = edited_name
            if edited_repo_type != original_repo_type:
                updates["repo_type"] = edited_repo_type
            if edited_repo_id != original_repo_id:
                updates["repo_id"] = edited_repo_id
                # Update SHA256 status when repo changes
                if edited_repo_id is None:
                    updates["sha256_status"] = "no_repo"
                else:
                    updates["sha256_status"] = "not_checked"
            if edited_gpu_model_ids != original_gpu_model_ids:
                updates["gpu_model_ids"] = edited_gpu_model_ids

            # Save to registry
            registry.update_model(original_name, updates)
            print_success(t("model_edit_saved"))

            # Update local model object
            if "name" in updates:
                model.name = edited_name
            if "repo_type" in updates:
                model.repo_type = edited_repo_type
            if "repo_id" in updates:
                model.repo_id = edited_repo_id
            if "sha256_status" in updates:
                model.sha256_status = updates["sha256_status"]
            if "gpu_model_ids" in updates:
                model.gpu_model_ids = edited_gpu_model_ids

            # Update original values for next iteration
            original_name = edited_name
            original_repo_type = edited_repo_type
            original_repo_id = edited_repo_id
            original_gpu_model_ids = edited_gpu_model_ids.copy() if edited_gpu_model_ids else None

            # Display updated configuration
            console.print()
            console.print(f"[bold cyan]{t('model_edit_updated_config')}[/bold cyan]\n")

            sha256_display = SHA256_STATUS_MAP_PLAIN.get(model.sha256_status, model.sha256_status)
            gpu_links_info = ""
            if is_cpu_model and model.gpu_model_ids:
                gpu_names = []
                for gpu_id in model.gpu_model_ids:
                    gpu_obj = registry.get_model_by_id(gpu_id)
                    if gpu_obj:
                        gpu_names.append(gpu_obj.name)
                    else:
                        gpu_names.append(f"[dim red]{gpu_id[:8]}... (deleted)[/dim red]")
                gpu_links_info = f"\n[bold]{t('model_edit_gpu_links')}[/bold]  {', '.join(gpu_names)}"

            content = f"""[bold]Name:[/bold]       {model.name}
[bold]Path:[/bold]       {model.path}
[bold]Format:[/bold]     {model.format}
[bold]Repo Type:[/bold]  {model.repo_type or '-'}
[bold]Repo ID:[/bold]    {model.repo_id or '-'}
[bold]SHA256:[/bold]     {sha256_display}{gpu_links_info}"""

            panel = Panel(content, border_style="green", padding=(0, 1))
            console.print(panel)
            console.print()

            # If repo changed, suggest verification
            if repo_changed and model.repo_id:
                console.print()
                console.print(f"[bold yellow]{t('model_edit_repo_changed_warning')}[/bold yellow]")
                console.print()
                console.print(f"  {t('model_edit_verify_hint')}")
                console.print()

            return

        elif choice == cancel_option:
            # Cancel
            console.print()
            if has_changes:
                if Confirm.ask(f"[yellow]{t('model_edit_discard_changes')}[/yellow]", default=False):
                    print_info(t("model_edit_cancelled"))
                    console.print()
                    return
                else:
                    # Go back to menu
                    continue
            else:
                print_info(t("model_edit_cancelled"))
                console.print()
                return


@app.command(name="info")
def info_model(
    name: str = typer.Argument(..., help="Name of model to display"),
) -> None:
    """Display detailed information about a model."""
    from rich.panel import Panel
    from pathlib import Path
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.utils.model_scanner import format_size

    registry = UserModelRegistry()

    # Load model
    model = registry.get_model(name)
    if not model:
        print_error(t("model_info_not_found", name=name))
        console.print()
        console.print(f"  {t('model_info_list_hint')} [cyan]kt model list[/cyan]")
        console.print()
        raise typer.Exit(1)

    console.print()

    # Check if path exists
    path_status = "[green]✓ Exists[/green]" if model.path_exists() else "[red]✗ Missing[/red]"

    # Format repo info
    if model.repo_id:
        repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
        repo_info = f"{repo_abbr}:{model.repo_id}"
    else:
        repo_info = "-"

    # Format SHA256 status
    sha256_display = SHA256_STATUS_MAP_PLAIN.get(model.sha256_status, model.sha256_status)

    # Calculate folder size and list files if exists
    moe_info = ""
    amx_info = ""

    if model.path_exists():
        path_obj = Path(model.path)
        try:
            if model.format == "safetensors":
                files = list(path_obj.glob("*.safetensors"))

                # Check for AMX weights
                is_amx, numa_count = is_amx_weights(str(path_obj))
                if is_amx:
                    amx_info = f"\n[bold]AMX Format:[/bold]   Yes (NUMA: {numa_count})"
                else:
                    # Check for MOE model
                    try:
                        from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

                        moe_result = analyze_moe_model(str(path_obj))
                        if moe_result and moe_result.get("num_experts", 0) > 0:
                            moe_info = f"""
[bold]MoE Info:[/bold]
  • Total Experts:     {moe_result['num_experts']}
  • Activated Experts: {moe_result['num_experts_per_tok']} experts/token
  • Hidden Layers:     {moe_result['num_hidden_layers']}
  • Total Model Size:  {moe_result['total_size_gb']:.2f} GB"""
                    except Exception:
                        pass  # Not a MoE model or analysis failed
            else:
                files = list(path_obj.glob("*.gguf"))

            total_size = sum(f.stat().st_size for f in files if f.exists())
            size_str = format_size(total_size)
            file_count = len(files)
            size_info = f"{size_str} ({file_count} files)"

            # List first few files
            file_list = "\n".join([f"  • {f.name}" for f in sorted(files)[:5]])
            if len(files) > 5:
                file_list += f"\n  ... and {len(files) - 5} more files"
        except Exception as e:
            size_info = f"Error calculating size: {e}"
            file_list = "-"
    else:
        size_info = "-"
        file_list = "[red]Path does not exist[/red]"

    # Format created/verified dates
    from datetime import datetime

    try:
        created_date = datetime.fromisoformat(model.created_at).strftime("%Y-%m-%d %H:%M:%S")
    except:
        created_date = model.created_at

    if model.last_verified:
        try:
            verified_date = datetime.fromisoformat(model.last_verified).strftime("%Y-%m-%d %H:%M:%S")
        except:
            verified_date = model.last_verified
    else:
        verified_date = "-"

    # Create detailed panel
    content = f"""[bold]Name:[/bold]         {model.name}
[bold]Path:[/bold]         {model.path}
[bold]Path Status:[/bold]  {path_status}
[bold]Format:[/bold]       {model.format}
[bold]Size:[/bold]         {size_info}{amx_info}{moe_info}
[bold]Repo Type:[/bold]    {model.repo_type or '-'}
[bold]Repo ID:[/bold]      {model.repo_id or '-'}
[bold]SHA256:[/bold]       {sha256_display}
[bold]Created:[/bold]      {created_date}
[bold]Last Verified:[/bold] {verified_date}

[bold]Files:[/bold]
{file_list}"""

    panel = Panel(content, title=f"[cyan]Model Information: {model.name}[/cyan]", border_style="cyan", padding=(1, 2))
    console.print(panel)
    console.print()


@app.command(name="remove")
def remove_model(
    name: str = typer.Argument(..., help="Name of model to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Remove a model from the registry (does not delete files)."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()

    # Check if model exists
    model = registry.get_model(name)
    if not model:
        print_error(t("model_remove_not_found", name=name))
        console.print()
        console.print(f"  {t('model_remove_list_hint')} [cyan]kt model list[/cyan]")
        console.print()
        raise typer.Exit(1)

    console.print()
    console.print(f"[bold yellow]{t('model_remove_warning')}[/bold yellow]")
    console.print(f"  {t('model_remove_note')}")
    console.print(f"  [dim]Path: {model.path}[/dim]")
    console.print()

    # Check if this GPU model is linked by any CPU models
    model_uuid = model.id
    affected_cpu_models = []

    # Only check for GPU models (safetensors that are not AMX)
    if model.format == "safetensors":
        is_amx, _ = is_amx_weights(model.path)
        if not is_amx:
            # This is a GPU model, check for CPU models that link to it
            for m in registry.list_models():
                if m.gpu_model_ids and model_uuid in m.gpu_model_ids:
                    affected_cpu_models.append(m)

    # If there are affected CPU models, inform the user
    if affected_cpu_models:
        console.print(f"[yellow]This GPU model is linked by {len(affected_cpu_models)} CPU model(s):[/yellow]")
        for cpu_model in affected_cpu_models:
            console.print(f"  • {cpu_model.name}")
        console.print()
        console.print(f"[dim]These links will be automatically removed.[/dim]")
        console.print()

    # Confirm deletion
    if not yes:
        if not confirm(t("model_remove_confirm", name=name), default=False):
            print_info(t("model_remove_cancelled"))
            console.print()
            return

    # Clean up references in CPU models before removing
    if affected_cpu_models:
        for cpu_model in affected_cpu_models:
            # Remove this GPU model's UUID from the cpu_model's gpu_model_ids list
            new_gpu_ids = [gid for gid in cpu_model.gpu_model_ids if gid != model_uuid]
            registry.update_model(cpu_model.name, {"gpu_model_ids": new_gpu_ids if new_gpu_ids else None})

    # Remove from registry
    if registry.remove_model(name):
        console.print()
        print_success(t("model_removed", name=name))
        console.print()
    else:
        print_error(t("model_remove_failed", name=name))
        raise typer.Exit(1)


@app.command(name="refresh")
def refresh_models() -> None:
    """Check all registered models and identify missing ones."""
    from rich.table import Table
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()
    models = registry.list_models()

    if not models:
        print_warning(t("model_no_registered_models"))
        console.print()
        return

    console.print()
    print_info(t("model_refresh_checking"))

    # Refresh status
    status = registry.refresh_status()

    # Check relationship integrity
    broken_relationships = []  # [(cpu_model, gpu_uuid, gpu_name_or_none)]
    for model in models:
        if model.gpu_model_ids:
            for gpu_uuid in model.gpu_model_ids:
                gpu_obj = registry.get_model_by_id(gpu_uuid)
                if not gpu_obj:
                    broken_relationships.append((model.name, gpu_uuid, None))
                elif not gpu_obj.path_exists():
                    broken_relationships.append((model.name, gpu_uuid, gpu_obj.name))

    console.print()

    # Show results
    has_issues = status["missing"] or broken_relationships

    if not has_issues:
        print_success(t("model_refresh_all_valid", count=len(models)))
        console.print(f"  {t('model_refresh_total', total=len(models))}")
        console.print()
        return

    # Show broken relationships
    if broken_relationships:
        print_warning(f"Found {len(broken_relationships)} broken GPU link(s)")
        console.print()

        from rich.table import Table

        rel_table = Table(show_header=True, header_style="bold yellow")
        rel_table.add_column("CPU Model", style="cyan")
        rel_table.add_column("GPU Model", style="dim")
        rel_table.add_column("Issue", style="red")

        for cpu_name, gpu_uuid, gpu_name in broken_relationships:
            if gpu_name is None:
                gpu_display = f"{gpu_uuid[:8]}..."
                issue = "Deleted"
            else:
                gpu_display = gpu_name
                issue = "Path Missing"
            rel_table.add_row(cpu_name, gpu_display, issue)

        console.print(rel_table)
        console.print()
        console.print(f"[dim]Use [cyan]kt model edit <cpu-model>[/cyan] to fix GPU links[/dim]")
        console.print()

    if not status["missing"]:
        # Only broken relationships, no missing models
        return

    # Show missing models
    print_warning(t("model_refresh_missing_found", count=len(status["missing"])))
    console.print()

    table = Table(show_header=True, header_style="bold")
    table.add_column(t("model_column_name"), style="cyan")
    table.add_column(t("model_column_path"), style="dim")
    table.add_column(t("model_column_status"), justify="center")

    for model in models:
        if model.name in status["missing"]:
            status_text = "[red]✗ Missing[/red]"
        else:
            status_text = "[green]✓ Valid[/green]"

        table.add_row(model.name, model.path, status_text)

    console.print(table)
    console.print()

    # Suggest actions
    console.print(f"[bold]{t('model_refresh_suggestions')}:[/bold]")
    console.print(f"  • {t('model_refresh_remove_hint')} [cyan]kt model remove <name>[/cyan]")
    console.print(f"  • {t('model_refresh_rescan_hint')} [cyan]kt model scan[/cyan]")
    console.print()


@app.command(name="verify")
def verify_model(
    name: str = typer.Argument(None, help="Name of model to verify (interactive if not provided)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed SHA256 comparison for each file"),
) -> None:
    """Verify model integrity using SHA256 checksums with interactive repair."""
    from pathlib import Path
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.utils.model_verifier import verify_model_integrity_with_progress, check_huggingface_connectivity

    registry = UserModelRegistry()

    # Helper function to display model selection table
    def show_model_table():
        from kt_kernel.cli.utils.model_scanner import format_size
        from pathlib import Path

        # Import MoE analyzer
        analyze_moe_model = None
        try:
            from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model
        except ImportError:
            pass

        all_models = registry.list_models()

        # Filter: only safetensors models with repo_id
        verifiable_models = [m for m in all_models if m.repo_id and m.format == "safetensors"]

        if not verifiable_models:
            print_warning(t("model_verify_all_no_repos"))
            console.print()
            console.print(f"  {t('model_verify_all_config_hint')}")
            console.print()
            return None

        # Analyze MoE models
        moe_results = {}
        if analyze_moe_model:
            for model in verifiable_models:
                try:
                    result = analyze_moe_model(model.path, use_cache=True)
                    if result and result.get("num_experts", 0) > 0:
                        moe_results[model.name] = result
                except Exception:
                    pass

        # Filter to only show MoE models
        moe_verifiable_models = [m for m in verifiable_models if m.name in moe_results]

        if not moe_verifiable_models:
            console.print()
            console.print("[yellow]No MoE models with repo_id found for verification.[/yellow]")
            console.print()
            console.print(
                f"[dim]Only MoE models can be verified. Use [cyan]kt model list[/cyan] to see all models.[/dim]"
            )
            console.print()
            return None

        console.print()
        console.print("[bold]Select a MoE model to verify:[/bold]\n")

        table = Table(show_header=True, header_style="bold", show_lines=False)
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column(t("model_column_name"), style="cyan", no_wrap=True)
        table.add_column("Path", style="dim", overflow="fold")
        table.add_column("Total", justify="right")
        table.add_column("Exps", justify="center", style="yellow")
        table.add_column("Act", justify="center", style="green")
        table.add_column(t("model_column_repo"), style="dim", overflow="fold")
        table.add_column(t("model_column_sha256"), justify="center")

        for i, model in enumerate(moe_verifiable_models, 1):
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

            # Get MoE info
            moe_info = moe_results.get(model.name)
            experts_display = f"[yellow]{moe_info['num_experts']}[/yellow]" if moe_info else "[dim]-[/dim]"
            activated_display = f"[green]{moe_info['num_experts_per_tok']}[/green]" if moe_info else "[dim]-[/dim]"

            # Repo info
            repo_abbr = "hf" if model.repo_type == "huggingface" else "ms"
            repo_display = f"{repo_abbr}:{model.repo_id}"

            # SHA256 status
            status_icon = {
                "not_checked": "[dim]○[/dim]",
                "checking": "[yellow]◐[/yellow]",
                "passed": "[green]✓[/green]",
                "failed": "[red]✗[/red]",
                "no_repo": "[dim]-[/dim]",
            }.get(model.sha256_status, "[dim]?[/dim]")

            table.add_row(
                str(i),
                model.name,
                model.path,
                size_display,
                experts_display,
                activated_display,
                repo_display,
                status_icon,
            )

        console.print(table)
        console.print()
        console.print("[dim]SHA256 Status: ○ Not checked | ✓ Passed | ✗ Failed[/dim]")
        console.print()

        return moe_verifiable_models

    # Main verification loop
    # Track files to verify (None = all files, list = specific files for re-verification)
    files_to_verify = None

    while True:
        selected_model = None

        # If name provided directly, use it once then switch to interactive
        if name:
            selected_model = registry.get_model(name)
            if not selected_model:
                print_error(t("model_verify_not_found", name=name))
                console.print()
                console.print(f"  {t('model_verify_list_hint')} [cyan]kt model list[/cyan]")
                console.print()
                raise typer.Exit(1)
            name = None  # Clear so next loop is interactive
        else:
            # Show interactive selection
            verifiable_models = show_model_table()
            if not verifiable_models:
                return

            choice = Prompt.ask("Enter model number to verify (or 'q' to quit)", default="1")

            if choice.lower() == "q":
                return

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(verifiable_models):
                    selected_model = verifiable_models[idx]
                    # Reset files_to_verify when selecting a new model
                    files_to_verify = None
                else:
                    print_error(f"Invalid selection: {choice}")
                    console.print()
                    continue
            except ValueError:
                print_error(f"Invalid input: {choice}")
                console.print()
                continue

        # Check model prerequisites
        console.print()

        if not selected_model.repo_id:
            print_warning(t("model_verify_no_repo", name=selected_model.name))
            console.print()
            console.print(f"  {t('model_verify_config_hint', name=selected_model.name)}")
            console.print()
            continue

        if not selected_model.path_exists():
            print_error(t("model_verify_path_missing", path=selected_model.path))
            console.print()
            continue

        # Check HuggingFace connectivity and decide whether to use mirror
        use_mirror = False
        if selected_model.repo_type == "huggingface":
            with console.status("[dim]Checking HuggingFace connectivity...[/dim]"):
                is_accessible, message = check_huggingface_connectivity(timeout=5)

            if not is_accessible:
                print_warning("HuggingFace Connection Failed")
                console.print()
                console.print(f"  {message}")
                console.print()
                console.print("  [yellow]Auto-switching to HuggingFace mirror:[/yellow] [cyan]hf-mirror.com[/cyan]")
                console.print()
                use_mirror = True

        # Perform verification with progress bar
        if files_to_verify:
            print_info(f"Re-verifying {len(files_to_verify)} repaired files: {selected_model.name}")
        else:
            print_info(f"Verifying: {selected_model.name}")
        console.print(f"  Repository: [yellow]{selected_model.repo_type}[/yellow]:{selected_model.repo_id}")
        console.print(f"  Local path: {selected_model.path}")
        console.print()

        # Helper function to fetch remote hashes with timeout (using console.status like connectivity check)
        def fetch_remote_hashes_with_timeout(repo_type, repo_id, use_mirror, timeout_seconds):
            """Fetch remote hashes with timeout, returns (hashes_dict, timed_out)."""
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
            from kt_kernel.cli.utils.model_verifier import fetch_model_sha256

            def fetch_hashes():
                platform = "hf" if repo_type == "huggingface" else "ms"
                return fetch_model_sha256(repo_id, platform, use_mirror=use_mirror, timeout=timeout_seconds)

            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(fetch_hashes)
                hashes = future.result(timeout=timeout_seconds)
                executor.shutdown(wait=False)
                return (hashes, False)
            except (FutureTimeoutError, Exception):
                executor.shutdown(wait=False)
                return (None, True)

        # Step 1: Fetch remote hashes with timeout and fallback
        official_hashes = None

        if selected_model.repo_type == "huggingface":
            # HF fallback chain: HF → HF-mirror → MS

            # Try 1: HuggingFace (or HF-mirror if already set)
            status = console.status(
                "[dim]Fetching remote hashes from HuggingFace{}...[/dim]".format(" mirror" if use_mirror else "")
            )
            status.start()
            official_hashes, timed_out = fetch_remote_hashes_with_timeout(
                repo_type="huggingface", repo_id=selected_model.repo_id, use_mirror=use_mirror, timeout_seconds=10
            )
            status.stop()

            # Try 2: If timed out and not already using mirror, try HF-mirror
            if timed_out and not use_mirror:
                print_warning("HuggingFace Fetch Timeout (10s)")
                console.print()
                console.print("  [yellow]Auto-switching to HuggingFace mirror:[/yellow] [cyan]hf-mirror.com[/cyan]")
                console.print()

                status = console.status("[dim]Fetching remote hashes from HuggingFace mirror...[/dim]")
                status.start()
                official_hashes, timed_out = fetch_remote_hashes_with_timeout(
                    repo_type="huggingface",
                    repo_id=selected_model.repo_id,
                    use_mirror=True,  # Use mirror
                    timeout_seconds=10,
                )
                status.stop()

            # Try 3: If still timed out, try ModelScope with same repo_id
            if timed_out:
                print_warning("HuggingFace Mirror Timeout (10s)")
                console.print()
                console.print("  [yellow]Fallback to ModelScope mirror with same repo_id...[/yellow]")
                console.print()

                status = console.status("[dim]Fetching remote hashes from ModelScope...[/dim]")
                status.start()
                official_hashes, timed_out = fetch_remote_hashes_with_timeout(
                    repo_type="modelscope",
                    repo_id=selected_model.repo_id,  # Use same repo_id
                    use_mirror=False,
                    timeout_seconds=10,
                )
                status.stop()

                if official_hashes:
                    # Success with ModelScope
                    console.print("  [green]✓ Successfully fetched from ModelScope[/green]")
                    console.print()
                elif timed_out:
                    # All failed
                    print_error("All sources timed out (HuggingFace and ModelScope)")
                    console.print()
                    console.print("  Please check your network connection or try again later")
                    console.print()
                    continue

        elif selected_model.repo_type == "modelscope":
            # ModelScope: no fallback, just timeout
            status = console.status("[dim]Fetching remote hashes from ModelScope...[/dim]")
            status.start()
            official_hashes, timed_out = fetch_remote_hashes_with_timeout(
                repo_type="modelscope", repo_id=selected_model.repo_id, use_mirror=False, timeout_seconds=10
            )
            status.stop()

            if timed_out:
                print_error("ModelScope Fetch Timeout (10s)")
                console.print()
                console.print("  Please check your network connection or try again later")
                console.print()
                continue

        # Check if we successfully fetched remote hashes
        if not official_hashes:
            # Already printed error message above, skip to next model
            continue

        # Success - print confirmation
        console.print(f"  [green]✓ Fetched {len(official_hashes)} file hashes from remote[/green]")
        console.print()

        # Step 2 & 3: Calculate local SHA256 and compare (with Progress bar)
        from kt_kernel.cli.utils.model_verifier import calculate_local_sha256

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Step 2: Calculate local SHA256 hashes (no timeout)
            local_dir_path = Path(selected_model.path)

            # Determine which files to hash
            if files_to_verify:
                # Only hash files that need re-verification
                clean_filenames = {
                    Path(f.replace(" (missing)", "").replace(" (hash mismatch)", "").strip()).name
                    for f in files_to_verify
                }
                # Collect files matching *.safetensors, *.json, *.py
                files_to_hash = []
                for pattern in ["*.safetensors", "*.json", "*.py"]:
                    files_to_hash.extend(
                        [f for f in local_dir_path.glob(pattern) if f.is_file() and f.name in clean_filenames]
                    )
            else:
                # Collect all important files: *.safetensors, *.json, *.py
                files_to_hash = []
                for pattern in ["*.safetensors", "*.json", "*.py"]:
                    files_to_hash.extend([f for f in local_dir_path.glob(pattern) if f.is_file()])

            total_files = len(files_to_hash)

            # Create progress task for local hashing
            hash_task_id = progress.add_task("[yellow]Calculating local SHA256...", total=total_files)
            completed_count = [0]

            def local_hash_callback(msg: str):
                if "Using" in msg and "workers" in msg:
                    # Show parallel worker info
                    console.print(f"  [dim]{msg}[/dim]")
                elif "[" in msg and "/" in msg and "]" in msg:
                    # Progress update
                    completed_count[0] += 1
                    if "✓" in msg:
                        filename = msg.split("✓")[1].strip().split("(")[0].strip()
                        progress.update(hash_task_id, advance=1, description=f"[yellow]Hashing: {filename[:40]}...")

            local_hashes = calculate_local_sha256(
                local_dir_path,
                "*.safetensors",
                progress_callback=local_hash_callback,
                files_list=files_to_hash if files_to_verify else None,
            )

            progress.remove_task(hash_task_id)
            console.print(f"  [green]✓ Calculated {len(local_hashes)} local file hashes[/green]")

            # Step 3: Compare hashes
            # If re-verifying specific files, only compare those files
            if files_to_verify:
                # Build set of clean filenames to verify
                clean_verify_filenames = {
                    Path(f.replace(" (missing)", "").replace(" (hash mismatch)", "").strip()).name
                    for f in files_to_verify
                }
                # Filter official_hashes to only include files we're re-verifying
                hashes_to_compare = {
                    filename: hash_value
                    for filename, hash_value in official_hashes.items()
                    if Path(filename).name in clean_verify_filenames
                }
            else:
                # First-time verification: compare all files
                hashes_to_compare = official_hashes

            compare_task_id = progress.add_task("[blue]Comparing hashes...", total=len(hashes_to_compare))

            files_failed = []
            files_missing = []
            files_passed = 0

            for filename, official_hash in hashes_to_compare.items():
                file_basename = Path(filename).name

                # Find matching local file
                local_hash = None
                for local_file, local_hash_value in local_hashes.items():
                    if Path(local_file).name == file_basename:
                        local_hash = local_hash_value
                        break

                if local_hash is None:
                    files_missing.append(filename)
                    if verbose:
                        console.print(f"  [red]✗ {file_basename} (missing)[/red]")
                elif local_hash.lower() != official_hash.lower():
                    files_failed.append(f"{filename} (hash mismatch)")
                    if verbose:
                        console.print(f"  [red]✗ {file_basename} (hash mismatch)[/red]")
                else:
                    files_passed += 1
                    if verbose:
                        console.print(f"  [green]✓ {file_basename}[/green]")

                progress.update(compare_task_id, advance=1)

            progress.remove_task(compare_task_id)

            # Build result
            total_checked = len(hashes_to_compare)  # Use actual compared count
            if files_failed or files_missing:
                all_failed = files_failed + [f"{f} (missing)" for f in files_missing]
                result = {
                    "status": "failed",
                    "files_checked": total_checked,
                    "files_passed": files_passed,
                    "files_failed": all_failed,
                }
            else:
                result = {
                    "status": "passed",
                    "files_checked": total_checked,
                    "files_passed": files_passed,
                    "files_failed": [],
                }

        # Update registry status and display results
        if result["status"] == "passed":
            registry.update_model(selected_model.name, {"sha256_status": "passed"})
            console.print()
            print_success(t("model_verify_passed"))
            console.print()
            console.print(f"  ✓ Files checked: [bold green]{result['files_checked']}[/bold green]")
            console.print(f"  ✓ All files passed SHA256 verification")
            console.print()
        elif result["status"] == "failed":
            registry.update_model(selected_model.name, {"sha256_status": "failed"})
            console.print()
            print_error(f"Verification failed! {len(result['files_failed'])} file(s) have issues")
            console.print()
            console.print(f"  Total files: {result['files_checked']}")
            console.print(f"  ✓ Passed: [green]{result['files_passed']}[/green]")
            console.print(f"  ✗ Failed: [red]{len(result['files_failed'])}[/red]")
            console.print()

            # Show failed files (only if not already shown in verbose mode)
            if not verbose:
                console.print("  [bold red]Failed files:[/bold red]")
                for failed_file in result["files_failed"]:
                    console.print(f"    ✗ {failed_file}")
                console.print()

            # Ask if user wants to repair
            if Confirm.ask("[yellow]Do you want to repair (re-download) the failed files?[/yellow]", default=True):
                console.print()
                print_info("Repairing failed files...")

                # Extract clean filenames by removing status suffixes
                files_to_download = [
                    f.replace(" (missing)", "").replace(" (hash mismatch)", "").strip() for f in result["files_failed"]
                ]

                # Download each failed file
                success_count = 0

                # Set mirror for downloads if needed
                import os

                original_hf_endpoint = os.environ.get("HF_ENDPOINT")
                if use_mirror and selected_model.repo_type == "huggingface" and not original_hf_endpoint:
                    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                    console.print(f"  [dim]Using HuggingFace mirror for downloads[/dim]")

                try:
                    for file_to_repair in files_to_download:
                        console.print(f"  Repairing: [cyan]{file_to_repair}[/cyan]")

                        # Step 1: Delete the corrupted/missing file if it exists
                        local_file_path = Path(selected_model.path) / file_to_repair
                        if local_file_path.exists():
                            try:
                                local_file_path.unlink()
                                console.print(f"    [dim]✓ Deleted corrupted file[/dim]")
                            except Exception as e:
                                console.print(f"    [yellow]⚠ Could not delete file: {e}[/yellow]")

                        # Step 2: Download the fresh file
                        if selected_model.repo_type == "huggingface":
                            # Use hf_hub_download for HuggingFace (inherits HF_ENDPOINT env var)
                            try:
                                from huggingface_hub import hf_hub_download

                                hf_hub_download(
                                    repo_id=selected_model.repo_id,
                                    filename=file_to_repair,
                                    local_dir=selected_model.path,
                                    local_dir_use_symlinks=False,
                                )
                                console.print(f"    [green]✓ Downloaded successfully[/green]")
                                success_count += 1
                            except ImportError:
                                print_error("huggingface_hub not installed. Install: pip install huggingface_hub")
                                break
                            except Exception as e:
                                console.print(f"    [red]✗ Download failed: {e}[/red]")
                        else:
                            # Use modelscope download for ModelScope
                            try:
                                from modelscope.hub.snapshot_download import snapshot_download

                                # Download directly to local_dir
                                snapshot_download(
                                    model_id=selected_model.repo_id,
                                    local_dir=selected_model.path,
                                    allow_file_pattern=file_to_repair,
                                )
                                console.print(f"    [green]✓ Downloaded successfully[/green]")
                                success_count += 1
                            except ImportError:
                                print_error("modelscope not installed. Install: pip install modelscope")
                                break
                            except Exception as e:
                                console.print(f"    [red]✗ Download failed: {e}[/red]")
                finally:
                    # Restore original HF_ENDPOINT
                    if use_mirror and selected_model.repo_type == "huggingface" and not original_hf_endpoint:
                        os.environ.pop("HF_ENDPOINT", None)
                    elif original_hf_endpoint:
                        os.environ["HF_ENDPOINT"] = original_hf_endpoint

                console.print()
                if success_count > 0:
                    print_success(f"Repaired {success_count}/{len(files_to_download)} files")
                    console.print()

                    # Ask if user wants to re-verify
                    if Confirm.ask("Re-verify the model now?", default=True):
                        # Re-verify by continuing the loop with the same model
                        # Only verify the files that were repaired
                        name = selected_model.name
                        files_to_verify = files_to_download
                        continue


@app.command(name="verify-all")
def verify_all_models() -> None:
    """Verify all models with repo configuration (not yet implemented)."""
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

    registry = UserModelRegistry()
    models = registry.list_models()

    # Filter models with repo configuration
    models_with_repo = [m for m in models if m.repo_id]

    if not models_with_repo:
        print_warning(t("model_verify_all_no_repos"))
        console.print()
        console.print(f"  {t('model_verify_all_config_hint')} [cyan]kt model edit <name>[/cyan]")
        console.print()
        return

    console.print()
    print_warning(t("model_verify_not_implemented"))
    console.print()
    console.print(f"  {t('model_verify_all_found', count=len(models_with_repo))}")
    console.print()

    for model in models_with_repo:
        console.print(f"  • {model.name} ({model.repo_type}:{model.repo_id})")

    console.print()
    console.print(f"  [dim]{t('model_verify_future_note')}[/dim]")
    console.print()
    console.print(f"  {t('model_verify_all_manual_hint')} [cyan]kt model verify <name>[/cyan]")
    console.print()


@app.command(name="auto-repo")
def auto_detect_repo(
    apply: bool = typer.Option(
        False, "--apply", "-a", help="Automatically apply detected repo information without confirmation"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d", help="Show what would be detected without making any changes"
    ),
) -> None:
    """
    Auto-detect repository information from model README.md files.

    Scans all models without repo_id (safetensors/gguf only) and attempts to
    extract repository information from README.md metadata (license_link field).

    Examples:
        kt model auto-repo              # Scan and ask for confirmation
        kt model auto-repo --apply      # Scan and apply automatically
        kt model auto-repo --dry-run    # Scan only, no changes
    """
    from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
    from kt_kernel.cli.utils.repo_detector import scan_models_for_repo, format_detection_report, apply_detection_results
    from rich.table import Table

    console.print()
    print_info("Scanning models for repository information...")
    console.print()

    # Get all models
    registry = UserModelRegistry()
    models = registry.list_models()

    if not models:
        print_warning("No models found in registry")
        console.print()
        return

    # Scan for repo information
    print_step("Analyzing README.md files...")
    results = scan_models_for_repo(models)

    # Show results
    console.print()

    if not results["detected"] and not results["not_detected"]:
        print_info("All models already have repository information configured")
        console.print()
        return

    # Create results table
    if results["detected"]:
        console.print("[bold green]✓ Detected Repository Information[/bold green]")
        console.print()

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Model Name", style="yellow")
        table.add_column("Repository", style="cyan")
        table.add_column("Type", style="magenta")

        for model, repo_id, repo_type in results["detected"]:
            table.add_row(model.name, repo_id, repo_type)

        console.print(table)
        console.print()

    if results["not_detected"]:
        console.print(
            f"[bold yellow]✗ No Repository Information Found ({len(results['not_detected'])} models)[/bold yellow]"
        )
        console.print()

        for model in results["not_detected"][:5]:  # Show first 5
            console.print(f"  • {model.name}")

        if len(results["not_detected"]) > 5:
            console.print(f"  ... and {len(results['not_detected']) - 5} more")

        console.print()

    if results["skipped"]:
        console.print(
            f"[dim]⊘ Skipped {len(results['skipped'])} models (already configured or not safetensors/gguf)[/dim]"
        )
        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  • [green]{len(results['detected'])}[/green] detected")
    console.print(f"  • [yellow]{len(results['not_detected'])}[/yellow] not detected")
    console.print(f"  • [dim]{len(results['skipped'])}[/dim] skipped")
    console.print()

    # Exit if dry run or no detections
    if dry_run:
        print_info("Dry run mode - no changes made")
        console.print()
        return

    if not results["detected"]:
        console.print()
        return

    # Ask for confirmation (unless --apply flag)
    if not apply:
        console.print()
        if not confirm(f"Apply repository information to {len(results['detected'])} model(s)?", default=False):
            print_warning("Cancelled - no changes made")
            console.print()
            return

    # Apply changes
    console.print()
    print_step("Applying changes...")

    updated_count = apply_detection_results(results, registry)

    console.print()
    if updated_count > 0:
        print_success(f"✓ Updated {updated_count} model(s) with repository information")
        console.print()
        console.print("  You can now:")
        console.print("  • Run [cyan]kt model verify <name>[/cyan] to verify model integrity")
        console.print("  • Check status with [cyan]kt model list[/cyan]")
        console.print()
    else:
        print_error("Failed to update models")
        console.print()
