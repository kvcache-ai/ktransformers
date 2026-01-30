"""
Model Verifier

SHA256 verification for model integrity
"""

import hashlib
import requests
import os
from pathlib import Path
from typing import Dict, Any, Literal, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


def _compute_file_sha256(file_path: Path) -> Tuple[str, str, float]:
    """
    Compute SHA256 for a single file (worker function for multiprocessing).

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (filename, sha256_hash, file_size_mb)
    """
    sha256_hash = hashlib.sha256()
    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    # Read file in chunks to handle large files
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192 * 1024), b""):  # 8MB chunks
            sha256_hash.update(byte_block)

    return file_path.name, sha256_hash.hexdigest(), file_size_mb


def check_huggingface_connectivity(timeout: int = 5) -> Tuple[bool, str]:
    """
    Check if HuggingFace is accessible.

    Args:
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (is_accessible, message)
    """
    test_url = "https://huggingface.co"

    try:
        response = requests.head(test_url, timeout=timeout, allow_redirects=True)
        if response.status_code < 500:  # 2xx, 3xx, 4xx are all considered "accessible"
            return True, "HuggingFace is accessible"
    except requests.exceptions.Timeout:
        return False, f"Connection to {test_url} timed out"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to {test_url}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

    return False, "Unknown connection error"


def verify_model_integrity(
    repo_type: Literal["huggingface", "modelscope"],
    repo_id: str,
    local_dir: Path,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Verify local model integrity against remote repository SHA256 hashes.

    Verifies all important files:
    - *.safetensors (weights)
    - *.json (config files)
    - *.py (custom model code)

    Args:
        repo_type: Type of repository ("huggingface" or "modelscope")
        repo_id: Repository ID (e.g., "deepseek-ai/DeepSeek-V3")
        local_dir: Local directory containing model files
        progress_callback: Optional callback function(message: str) for progress updates

    Returns:
        Dictionary with verification results:
        {
            "status": "passed" | "failed" | "error",
            "files_checked": int,
            "files_passed": int,
            "files_failed": [list of filenames],
            "error_message": str (optional)
        }
    """

    def report_progress(msg: str):
        """Helper to report progress"""
        if progress_callback:
            progress_callback(msg)

    try:
        # Convert repo_type to platform format
        platform = "hf" if repo_type == "huggingface" else "ms"

        # 1. Fetch official SHA256 hashes from remote
        report_progress("Fetching official SHA256 hashes from remote repository...")
        official_hashes = fetch_model_sha256(repo_id, platform)
        report_progress(f"✓ Fetched {len(official_hashes)} file hashes from remote")

        if not official_hashes:
            return {
                "status": "error",
                "files_checked": 0,
                "files_passed": 0,
                "files_failed": [],
                "error_message": f"No verifiable files found in remote repository: {repo_id}",
            }

        # 2. Calculate local SHA256 hashes with progress
        report_progress(f"Calculating SHA256 for local files...")

        # Get all local files matching the patterns
        local_files = []
        for pattern in ["*.safetensors", "*.json", "*.py"]:
            local_files.extend([f for f in local_dir.glob(pattern) if f.is_file()])

        if not local_files:
            return {
                "status": "error",
                "files_checked": 0,
                "files_passed": 0,
                "files_failed": [],
                "error_message": f"No verifiable files found in local directory: {local_dir}",
            }

        # Calculate hashes for all files
        local_hashes = calculate_local_sha256(
            local_dir,
            file_pattern="*.safetensors",  # Unused when files_list is provided
            progress_callback=report_progress,
            files_list=local_files,
        )
        report_progress(f"✓ Calculated {len(local_hashes)} local file hashes")

        # 3. Compare hashes with progress
        report_progress(f"Comparing {len(official_hashes)} files...")
        files_failed = []
        files_missing = []
        files_passed = 0

        for idx, (filename, official_hash) in enumerate(official_hashes.items(), 1):
            # Handle potential path separators in filename
            file_basename = Path(filename).name

            # Try to find the file in local hashes
            local_hash = None
            for local_file, local_hash_value in local_hashes.items():
                if Path(local_file).name == file_basename:
                    local_hash = local_hash_value
                    break

            if local_hash is None:
                files_missing.append(filename)
                report_progress(f"  [{idx}/{len(official_hashes)}] ✗ {file_basename} - MISSING")
            elif local_hash.lower() != official_hash.lower():
                files_failed.append(f"{filename} (hash mismatch)")
                report_progress(f"  [{idx}/{len(official_hashes)}] ✗ {file_basename} - HASH MISMATCH")
            else:
                files_passed += 1
                report_progress(f"  [{idx}/{len(official_hashes)}] ✓ {file_basename}")

        # 4. Return results
        total_checked = len(official_hashes)

        if files_failed or files_missing:
            all_failed = files_failed + [f"{f} (missing)" for f in files_missing]
            return {
                "status": "failed",
                "files_checked": total_checked,
                "files_passed": files_passed,
                "files_failed": all_failed,
                "error_message": f"{len(all_failed)} file(s) failed verification",
            }
        else:
            return {
                "status": "passed",
                "files_checked": total_checked,
                "files_passed": files_passed,
                "files_failed": [],
            }

    except ImportError as e:
        return {
            "status": "error",
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": [],
            "error_message": f"Missing required package: {str(e)}. Install with: pip install huggingface-hub modelscope",
            "is_network_error": False,
        }
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ) as e:
        # Network-related errors - suggest mirror
        error_msg = f"Network error: {str(e)}"
        if repo_type == "huggingface":
            error_msg += "\n\nTry using HuggingFace mirror:\n  export HF_ENDPOINT=https://hf-mirror.com"
        return {
            "status": "error",
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": [],
            "error_message": error_msg,
            "is_network_error": True,
        }
    except Exception as e:
        return {
            "status": "error",
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": [],
            "error_message": f"Verification failed: {str(e)}",
            "is_network_error": False,
        }


def calculate_local_sha256(
    local_dir: Path, file_pattern: str = "*.safetensors", progress_callback=None, files_list: list[Path] = None
) -> Dict[str, str]:
    """
    Calculate SHA256 hashes for files in a directory using parallel processing.

    Args:
        local_dir: Directory to scan
        file_pattern: Glob pattern for files to hash (ignored if files_list is provided)
        progress_callback: Optional callback function(message: str) for progress updates
        files_list: Optional pre-filtered list of files to hash (overrides file_pattern)

    Returns:
        Dictionary mapping filename to SHA256 hash
    """
    result = {}

    if not local_dir.exists():
        return result

    # Get all files first to report total
    if files_list is not None:
        files_to_hash = files_list
    else:
        files_to_hash = [f for f in local_dir.glob(file_pattern) if f.is_file()]
    total_files = len(files_to_hash)

    if total_files == 0:
        return result

    # Use min(16, total_files) workers to avoid over-spawning processes
    max_workers = min(16, total_files)

    if progress_callback:
        progress_callback(f"  Using {max_workers} parallel workers for SHA256 calculation")

    # Use ProcessPoolExecutor for CPU-intensive SHA256 computation
    completed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(_compute_file_sha256, file_path): file_path for file_path in files_to_hash}

        # Process results as they complete
        for future in as_completed(future_to_file):
            completed_count += 1
            try:
                filename, sha256_hash, file_size_mb = future.result()
                result[filename] = sha256_hash

                if progress_callback:
                    progress_callback(f"  [{completed_count}/{total_files}] ✓ {filename} ({file_size_mb:.1f} MB)")

            except Exception as e:
                file_path = future_to_file[future]
                if progress_callback:
                    progress_callback(f"  [{completed_count}/{total_files}] ✗ {file_path.name} - Error: {str(e)}")

    return result


def fetch_model_sha256(
    repo_id: str,
    platform: Literal["hf", "ms"],
    revision: str | None = None,
    use_mirror: bool = False,
    timeout: int | None = None,
) -> dict[str, str]:
    """
    获取模型仓库中所有重要文件的 sha256 哈希值。

    包括：
    - *.safetensors (权重文件)
    - *.json (配置文件：config.json, tokenizer_config.json 等)
    - *.py (自定义模型代码：modeling.py, configuration.py 等)

    Args:
        repo_id: 仓库 ID，例如 "Qwen/Qwen3-30B-A3B"
        platform: 平台，"hf" (HuggingFace) 或 "ms" (ModelScope)
        revision: 版本/分支，默认 HuggingFace 为 "main"，ModelScope 为 "master"
        use_mirror: 是否使用镜像（仅对 HuggingFace 有效）
        timeout: 网络请求超时时间（秒），None 表示不设置超时

    Returns:
        dict: 文件名到 sha256 的映射，例如 {"model-00001-of-00016.safetensors": "abc123...", "config.json": "def456..."}
    """
    if platform == "hf":
        # 先尝试直连，失败后自动使用镜像
        try:
            if use_mirror:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=True, timeout=timeout)
            else:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=False, timeout=timeout)
        except Exception as e:
            # 如果不是镜像模式且失败了，自动重试使用镜像
            if not use_mirror:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=True, timeout=timeout)
            else:
                raise e
    elif platform == "ms":
        return _fetch_from_modelscope(repo_id, revision or "master", timeout=timeout)
    else:
        raise ValueError(f"不支持的平台: {platform}，请使用 'hf' 或 'ms'")


def _fetch_from_huggingface(
    repo_id: str, revision: str, use_mirror: bool = False, timeout: int | None = None
) -> dict[str, str]:
    """从 HuggingFace 获取所有重要文件的 sha256

    Args:
        repo_id: 仓库 ID
        revision: 版本/分支
        use_mirror: 是否使用镜像（hf-mirror.com）
        timeout: 网络请求超时时间（秒），None 表示不设置超时
    """
    import os
    import socket

    # 如果需要使用镜像，设置环境变量
    original_endpoint = os.environ.get("HF_ENDPOINT")
    if use_mirror and not original_endpoint:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # Set socket timeout if specified
    original_timeout = socket.getdefaulttimeout()
    if timeout is not None:
        socket.setdefaulttimeout(timeout)

    from huggingface_hub import HfApi, list_repo_files

    try:
        api = HfApi()
        all_files = list_repo_files(repo_id=repo_id, revision=revision)

        # 筛选重要文件：*.safetensors, *.json, *.py
        important_files = [f for f in all_files if f.endswith((".safetensors", ".json", ".py"))]

        if not important_files:
            return {}

        paths_info = api.get_paths_info(
            repo_id=repo_id,
            paths=important_files,
            revision=revision,
        )

        result = {}
        for file_info in paths_info:
            if hasattr(file_info, "lfs") and file_info.lfs is not None:
                sha256 = file_info.lfs.sha256
            else:
                sha256 = getattr(file_info, "blob_id", None)
            result[file_info.path] = sha256

        return result
    finally:
        # 恢复原始 socket timeout
        socket.setdefaulttimeout(original_timeout)

        # 恢复原始环境变量
        if use_mirror and not original_endpoint:
            os.environ.pop("HF_ENDPOINT", None)
        elif original_endpoint:
            os.environ["HF_ENDPOINT"] = original_endpoint


def _fetch_from_modelscope(repo_id: str, revision: str, timeout: int | None = None) -> dict[str, str]:
    """从 ModelScope 获取所有重要文件的 sha256

    Args:
        repo_id: 仓库 ID
        revision: 版本/分支
        timeout: 网络请求超时时间（秒），None 表示不设置超时
    """
    import socket
    from modelscope.hub.api import HubApi

    # Set socket timeout if specified
    original_timeout = socket.getdefaulttimeout()
    if timeout is not None:
        socket.setdefaulttimeout(timeout)

    try:
        api = HubApi()
        files_info = api.get_model_files(model_id=repo_id, revision=revision)

        result = {}
        for file_info in files_info:
            filename = file_info.get("Name", file_info.get("Path", ""))
            # 筛选重要文件：*.safetensors, *.json, *.py
            if filename.endswith((".safetensors", ".json", ".py")):
                sha256 = file_info.get("Sha256", file_info.get("sha256", None))
                result[filename] = sha256

        return result
    finally:
        # 恢复原始 socket timeout
        socket.setdefaulttimeout(original_timeout)


def verify_model_integrity_with_progress(
    repo_type: Literal["huggingface", "modelscope"],
    repo_id: str,
    local_dir: Path,
    progress_callback=None,
    verbose: bool = False,
    use_mirror: bool = False,
    files_to_verify: list[str] | None = None,
    timeout: int | None = None,
) -> Dict[str, Any]:
    """
    Verify model integrity with enhanced progress reporting for Rich Progress bars.

    This is a wrapper around verify_model_integrity() that provides more detailed
    progress information suitable for progress bar display.

    The progress_callback receives:
    - (message: str, total: int, current: int) for countable operations
    - (message: str) for status updates

    Args:
        repo_type: Repository type ("huggingface" or "modelscope")
        repo_id: Repository ID
        local_dir: Local directory path
        progress_callback: Optional callback for progress updates
        verbose: If True, output detailed SHA256 comparison for each file
        use_mirror: If True, use HuggingFace mirror (hf-mirror.com)
        files_to_verify: Optional list of specific files to verify (for re-verification)
        timeout: Network request timeout in seconds (None = no timeout)
    """

    def report_progress(msg: str, total=None, current=None):
        """Enhanced progress reporter"""
        if progress_callback:
            progress_callback(msg, total, current)

    try:
        platform = "hf" if repo_type == "huggingface" else "ms"

        # 1. Fetch official SHA256 hashes
        if files_to_verify:
            report_progress(f"Fetching SHA256 hashes for {len(files_to_verify)} files...")
        elif use_mirror and platform == "hf":
            report_progress("Fetching official SHA256 hashes from mirror (hf-mirror.com)...")
        else:
            report_progress("Fetching official SHA256 hashes from remote repository...")

        official_hashes = fetch_model_sha256(repo_id, platform, use_mirror=use_mirror, timeout=timeout)

        # Filter to only requested files if specified
        if files_to_verify:
            # Extract clean filenames from files_to_verify (remove markers like "(missing)")
            clean_filenames = set()
            for f in files_to_verify:
                clean_f = f.replace(" (missing)", "").replace(" (hash mismatch)", "").strip()
                # Ensure we only use the filename, not full path
                clean_filenames.add(Path(clean_f).name)

            # Filter official_hashes to only include requested files
            # Compare using basename since official_hashes keys might have paths
            official_hashes = {k: v for k, v in official_hashes.items() if Path(k).name in clean_filenames}

        report_progress(f"✓ Fetched {len(official_hashes)} file hashes from remote")

        if not official_hashes:
            return {
                "status": "error",
                "files_checked": 0,
                "files_passed": 0,
                "files_failed": [],
                "error_message": f"No safetensors files found in remote repository: {repo_id}",
            }

        # 2. Calculate local SHA256 hashes
        local_dir_path = Path(local_dir)

        # Only hash the files we need to verify
        if files_to_verify:
            # Extract clean filenames (without markers)
            clean_filenames = set()
            for f in files_to_verify:
                clean_f = f.replace(" (missing)", "").replace(" (hash mismatch)", "").strip()
                # Ensure we only use the filename, not full path
                clean_filenames.add(Path(clean_f).name)

            # Only hash files that match the clean filenames
            files_to_hash = [
                f for f in local_dir_path.glob("*.safetensors") if f.is_file() and f.name in clean_filenames
            ]
        else:
            files_to_hash = [f for f in local_dir_path.glob("*.safetensors") if f.is_file()]

        total_files = len(files_to_hash)

        if files_to_verify:
            report_progress(f"Calculating SHA256 for {total_files} repaired files...", total=total_files, current=0)
        else:
            report_progress(f"Calculating SHA256 for local files...", total=total_files, current=0)

        # Progress wrapper for hashing
        completed_count = [0]  # Use list for mutable closure

        def hash_progress_callback(msg: str):
            if "Using" in msg and "workers" in msg:
                report_progress(msg)
            elif "[" in msg and "/" in msg and "]" in msg:
                # Progress update like: [1/10] ✓ filename (123.4 MB)
                completed_count[0] += 1
                report_progress(msg, total=total_files, current=completed_count[0])

        # Pass the pre-filtered files_to_hash list
        local_hashes = calculate_local_sha256(
            local_dir_path,
            "*.safetensors",
            progress_callback=hash_progress_callback,
            files_list=files_to_hash if files_to_verify else None,
        )
        report_progress(f"✓ Calculated {len(local_hashes)} local file hashes")

        # 3. Compare hashes
        report_progress(f"Comparing {len(official_hashes)} files...", total=len(official_hashes), current=0)

        files_failed = []
        files_missing = []
        files_passed = 0

        for idx, (filename, official_hash) in enumerate(official_hashes.items(), 1):
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
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✗ {file_basename} (missing)\n  Remote: {official_hash}\n  Local:  <missing>",
                        total=len(official_hashes),
                        current=idx,
                    )
                else:
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✗ {file_basename} (missing)",
                        total=len(official_hashes),
                        current=idx,
                    )
            elif local_hash.lower() != official_hash.lower():
                files_failed.append(f"{filename} (hash mismatch)")
                if verbose:
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✗ {file_basename} (hash mismatch)\n  Remote: {official_hash}\n  Local:  {local_hash}",
                        total=len(official_hashes),
                        current=idx,
                    )
                else:
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✗ {file_basename} (hash mismatch)",
                        total=len(official_hashes),
                        current=idx,
                    )
            else:
                files_passed += 1
                if verbose:
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✓ {file_basename}\n  Remote: {official_hash}\n  Local:  {local_hash}",
                        total=len(official_hashes),
                        current=idx,
                    )
                else:
                    report_progress(
                        f"[{idx}/{len(official_hashes)}] ✓ {file_basename}", total=len(official_hashes), current=idx
                    )

        # 4. Return results
        total_checked = len(official_hashes)

        if files_failed or files_missing:
            all_failed = files_failed + [f"{f} (missing)" for f in files_missing]
            return {
                "status": "failed",
                "files_checked": total_checked,
                "files_passed": files_passed,
                "files_failed": all_failed,
                "error_message": f"{len(all_failed)} file(s) failed verification",
            }
        else:
            return {
                "status": "passed",
                "files_checked": total_checked,
                "files_passed": files_passed,
                "files_failed": [],
            }

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
        TimeoutError,  # Socket timeout from socket.setdefaulttimeout()
        OSError,  # Network-related OS errors
    ) as e:
        error_msg = f"Network error: {str(e)}"
        if repo_type == "huggingface":
            error_msg += "\n\nTry using HuggingFace mirror:\n  export HF_ENDPOINT=https://hf-mirror.com"
        return {
            "status": "error",
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": [],
            "error_message": error_msg,
            "is_network_error": True,
        }
    except Exception as e:
        return {
            "status": "error",
            "files_checked": 0,
            "files_passed": 0,
            "files_failed": [],
            "error_message": f"Verification failed: {str(e)}",
            "is_network_error": False,
        }


def pre_operation_verification(user_model, user_registry, operation_name: str = "operation") -> None:
    """Pre-operation verification of model integrity.

    Can be used before running or quantizing models to ensure integrity.

    Args:
        user_model: UserModel object to verify
        user_registry: UserModelRegistry instance
        operation_name: Name of the operation (e.g., "running", "quantizing")
    """
    from rich.prompt import Prompt, Confirm
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    from kt_kernel.cli.i18n import get_lang
    from kt_kernel.cli.utils.console import console, print_info, print_warning, print_error, print_success, print_step
    import typer

    lang = get_lang()

    # Check if already verified
    if user_model.sha256_status == "passed":
        console.print()
        print_info("Model integrity already verified ✓")
        console.print()
        return

    # Model not verified yet
    console.print()
    console.print("[bold yellow]═══ Model Integrity Check ═══[/bold yellow]")
    console.print()

    # Check if repo_id exists
    if not user_model.repo_id:
        # No repo_id - ask user to provide one
        console.print("[yellow]No repository ID configured for this model.[/yellow]")
        console.print()
        console.print("To verify model integrity, we need the repository ID (e.g., 'deepseek-ai/DeepSeek-V3')")
        console.print()

        if not Confirm.ask("Would you like to configure repository ID now?", default=True):
            console.print()
            print_warning(f"Skipping verification. Model will be used for {operation_name} without integrity check.")
            console.print()
            return

        # Ask for repo type
        console.print()
        console.print("Repository type:")
        console.print("  [cyan][1][/cyan] HuggingFace")
        console.print("  [cyan][2][/cyan] ModelScope")
        console.print()

        repo_type_choice = Prompt.ask("Select repository type", choices=["1", "2"], default="1")
        repo_type = "huggingface" if repo_type_choice == "1" else "modelscope"

        # Ask for repo_id
        console.print()
        repo_id = Prompt.ask("Enter repository ID (e.g., deepseek-ai/DeepSeek-V3)")

        # Update model
        user_registry.update_model(user_model.name, {"repo_type": repo_type, "repo_id": repo_id})
        user_model.repo_type = repo_type
        user_model.repo_id = repo_id

        console.print()
        print_success(f"Repository configured: {repo_type}:{repo_id}")
        console.print()

    # Now ask if user wants to verify
    console.print("[dim]Model integrity verification is a one-time check that ensures your[/dim]")
    console.print("[dim]model weights are not corrupted. This helps prevent runtime errors.[/dim]")
    console.print()

    if not Confirm.ask(f"Would you like to verify model integrity before {operation_name}?", default=True):
        console.print()
        print_warning(f"Skipping verification. Model will be used for {operation_name} without integrity check.")
        console.print()
        return

    # Perform verification
    console.print()
    print_step("Verifying model integrity...")
    console.print()

    # Check connectivity first
    use_mirror = False
    if user_model.repo_type == "huggingface":
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

    # Fetch remote hashes with timeout
    def fetch_with_timeout(repo_type, repo_id, use_mirror, timeout):
        """Fetch hashes with timeout."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            platform = "hf" if repo_type == "huggingface" else "ms"
            future = executor.submit(fetch_model_sha256, repo_id, platform, use_mirror=use_mirror, timeout=timeout)
            hashes = future.result(timeout=timeout)
            executor.shutdown(wait=False)
            return (hashes, False)
        except (FutureTimeoutError, Exception):
            executor.shutdown(wait=False)
            return (None, True)

    # Try fetching hashes
    status = console.status("[dim]Fetching remote hashes...[/dim]")
    status.start()
    official_hashes, timed_out = fetch_with_timeout(user_model.repo_type, user_model.repo_id, use_mirror, 10)
    status.stop()

    # Handle timeout with fallback
    if timed_out and user_model.repo_type == "huggingface" and not use_mirror:
        print_warning("HuggingFace Fetch Timeout (10s)")
        console.print()
        console.print("  [yellow]Trying HuggingFace mirror...[/yellow]")
        console.print()

        status = console.status("[dim]Fetching remote hashes from mirror...[/dim]")
        status.start()
        official_hashes, timed_out = fetch_with_timeout(user_model.repo_type, user_model.repo_id, True, 10)
        status.stop()

    if timed_out and user_model.repo_type == "huggingface":
        print_warning("HuggingFace Mirror Timeout (10s)")
        console.print()
        console.print("  [yellow]Fallback to ModelScope...[/yellow]")
        console.print()

        status = console.status("[dim]Fetching remote hashes from ModelScope...[/dim]")
        status.start()
        official_hashes, timed_out = fetch_with_timeout("modelscope", user_model.repo_id, False, 10)
        status.stop()

    if not official_hashes or timed_out:
        print_error("Failed to fetch remote hashes (network timeout)")
        console.print()
        console.print("  [yellow]Unable to verify model integrity due to network issues.[/yellow]")
        console.print()

        if not Confirm.ask(f"Continue {operation_name} without verification?", default=False):
            raise typer.Exit(0)

        console.print()
        return

    console.print(f"  [green]✓ Fetched {len(official_hashes)} file hashes[/green]")
    console.print()

    # Calculate local hashes and compare
    local_dir = Path(user_model.path)
    files_to_hash = [f for f in local_dir.glob("*.safetensors") if f.is_file()]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Calculate local hashes
        task = progress.add_task("[yellow]Calculating local SHA256...", total=len(files_to_hash))

        def hash_callback(msg):
            if "[" in msg and "/" in msg and "]" in msg and "✓" in msg:
                progress.advance(task)

        local_hashes = calculate_local_sha256(local_dir, "*.safetensors", progress_callback=hash_callback)
        progress.remove_task(task)

        console.print(f"  [green]✓ Calculated {len(local_hashes)} local hashes[/green]")
        console.print()

        # Compare hashes
        task = progress.add_task("[blue]Comparing hashes...", total=len(official_hashes))

        files_failed = []
        files_missing = []
        files_passed = 0

        for filename, official_hash in official_hashes.items():
            file_basename = Path(filename).name
            local_hash = None

            for local_file, local_hash_value in local_hashes.items():
                if Path(local_file).name == file_basename:
                    local_hash = local_hash_value
                    break

            if local_hash is None:
                files_missing.append(filename)
            elif local_hash.lower() != official_hash.lower():
                files_failed.append(f"{filename} (hash mismatch)")
            else:
                files_passed += 1

            progress.advance(task)

        progress.remove_task(task)

    console.print()

    # Check results
    if not files_failed and not files_missing:
        # Verification passed
        user_registry.update_model(user_model.name, {"sha256_status": "passed"})
        print_success("Model integrity verification PASSED ✓")
        console.print()
        console.print(f"  All {files_passed} files verified successfully")
        console.print()
    else:
        # Verification failed
        user_registry.update_model(user_model.name, {"sha256_status": "failed"})
        print_error(f"Model integrity verification FAILED")
        console.print()
        console.print(f"  ✓ Passed: [green]{files_passed}[/green]")
        console.print(f"  ✗ Failed: [red]{len(files_failed) + len(files_missing)}[/red]")
        console.print()

        if files_missing:
            console.print(f"  [red]Missing files ({len(files_missing)}):[/red]")
            for f in files_missing[:5]:
                console.print(f"    - {Path(f).name}")
            if len(files_missing) > 5:
                console.print(f"    ... and {len(files_missing) - 5} more")
            console.print()

        if files_failed:
            console.print(f"  [red]Hash mismatch ({len(files_failed)}):[/red]")
            for f in files_failed[:5]:
                console.print(f"    - {f}")
            if len(files_failed) > 5:
                console.print(f"    ... and {len(files_failed) - 5} more")
            console.print()

        console.print("[bold red]⚠ WARNING: Model weights may be corrupted![/bold red]")
        console.print()
        console.print("This could cause runtime errors or incorrect inference results.")
        console.print()

        # Ask if user wants to repair
        if Confirm.ask("Would you like to repair (re-download) the corrupted files?", default=True):
            console.print()
            print_info("Please run: [cyan]kt model verify " + user_model.name + "[/cyan]")
            console.print()
            console.print("The verify command will guide you through the repair process.")
            raise typer.Exit(0)

        # Ask if user wants to continue anyway
        console.print()
        if not Confirm.ask(
            f"[yellow]Continue {operation_name} with potentially corrupted weights?[/yellow]", default=False
        ):
            raise typer.Exit(0)

        console.print()
        print_warning(f"Proceeding with {operation_name} using unverified weights at your own risk...")
        console.print()
