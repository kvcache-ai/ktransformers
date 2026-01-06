"""Helper functions for interactive model download."""

from pathlib import Path
from typing import Dict, List, Tuple
import fnmatch


def list_remote_files_hf(repo_id: str, use_mirror: bool = False) -> List[Dict[str, any]]:
    """
    List files in a HuggingFace repository.

    Returns:
        List of dicts with keys: 'path', 'size' (in bytes)
    """
    from huggingface_hub import HfApi
    import os

    # Set mirror if needed
    original_endpoint = os.environ.get("HF_ENDPOINT")
    if use_mirror and not original_endpoint:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        api = HfApi()
        files_info = api.list_repo_tree(repo_id=repo_id, recursive=True)

        result = []
        for item in files_info:
            # Skip directories
            if hasattr(item, "type") and item.type == "directory":
                continue

            # Get file info
            file_path = item.path if hasattr(item, "path") else str(item)
            file_size = item.size if hasattr(item, "size") else 0

            result.append({"path": file_path, "size": file_size})

        return result
    finally:
        # Restore original endpoint
        if use_mirror and not original_endpoint:
            os.environ.pop("HF_ENDPOINT", None)
        elif original_endpoint:
            os.environ["HF_ENDPOINT"] = original_endpoint


def list_remote_files_ms(repo_id: str) -> List[Dict[str, any]]:
    """
    List files in a ModelScope repository.

    Returns:
        List of dicts with keys: 'path', 'size' (in bytes)
    """
    from modelscope.hub.api import HubApi

    api = HubApi()
    files_info = api.get_model_files(model_id=repo_id, recursive=True)

    result = []
    for file_info in files_info:
        file_path = file_info.get("Name", file_info.get("Path", ""))
        file_size = file_info.get("Size", 0)

        result.append({"path": file_path, "size": file_size})

    return result


def filter_files_by_pattern(files: List[Dict[str, any]], pattern: str) -> List[Dict[str, any]]:
    """Filter files by glob pattern."""
    if pattern == "*":
        return files

    filtered = []
    for file in files:
        # Check if filename matches pattern
        filename = Path(file["path"]).name
        full_path = file["path"]

        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(full_path, pattern):
            filtered.append(file)

    return filtered


def calculate_total_size(files: List[Dict[str, any]]) -> int:
    """Calculate total size of files in bytes."""
    return sum(f["size"] for f in files)


def format_file_list_table(files: List[Dict[str, any]], max_display: int = 10):
    """Format file list as a table for display."""
    from rich.table import Table
    from kt_kernel.cli.utils.model_scanner import format_size

    table = Table(show_header=True, header_style="bold")
    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Size", justify="right")

    # Show first max_display files
    for file in files[:max_display]:
        table.add_row(file["path"], format_size(file["size"]))

    if len(files) > max_display:
        table.add_row(f"... and {len(files) - max_display} more files", "[dim]...[/dim]")

    return table


def verify_repo_exists(repo_id: str, repo_type: str, use_mirror: bool = False) -> Tuple[bool, str]:
    """
    Verify if a repository exists.

    Returns:
        (exists: bool, message: str)
    """
    try:
        if repo_type == "huggingface":
            import os

            original_endpoint = os.environ.get("HF_ENDPOINT")
            if use_mirror and not original_endpoint:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

            from huggingface_hub import HfApi

            try:
                api = HfApi()
                api.repo_info(repo_id=repo_id, repo_type="model")
                return True, "Repository found"
            finally:
                if use_mirror and not original_endpoint:
                    os.environ.pop("HF_ENDPOINT", None)
                elif original_endpoint:
                    os.environ["HF_ENDPOINT"] = original_endpoint

        else:  # modelscope
            from modelscope.hub.api import HubApi

            api = HubApi()
            api.get_model(model_id=repo_id)
            return True, "Repository found"

    except Exception as e:
        return False, f"Repository not found: {str(e)}"
