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
                "error_message": f"No safetensors files found in remote repository: {repo_id}",
            }

        # 2. Calculate local SHA256 hashes with progress
        report_progress(f"Calculating SHA256 for local files...")
        local_hashes = calculate_local_sha256(local_dir, "*.safetensors", progress_callback=report_progress)
        report_progress(f"✓ Calculated {len(local_hashes)} local file hashes")

        # Also check for gguf files if any exist
        gguf_hashes = calculate_local_sha256(local_dir, "*.gguf")
        if gguf_hashes:
            # For gguf, we don't verify against remote (usually converted locally)
            # Just check they exist and are readable
            pass

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
) -> dict[str, str]:
    """
    获取模型仓库中所有 safetensors 文件的 sha256 哈希值。

    Args:
        repo_id: 仓库 ID，例如 "Qwen/Qwen3-30B-A3B"
        platform: 平台，"hf" (HuggingFace) 或 "ms" (ModelScope)
        revision: 版本/分支，默认 HuggingFace 为 "main"，ModelScope 为 "master"
        use_mirror: 是否使用镜像（仅对 HuggingFace 有效）

    Returns:
        dict: 文件名到 sha256 的映射，例如 {"model-00001-of-00016.safetensors": "abc123..."}
    """
    if platform == "hf":
        # 先尝试直连，失败后自动使用镜像
        try:
            if use_mirror:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=True)
            else:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=False)
        except Exception as e:
            # 如果不是镜像模式且失败了，自动重试使用镜像
            if not use_mirror:
                return _fetch_from_huggingface(repo_id, revision or "main", use_mirror=True)
            else:
                raise e
    elif platform == "ms":
        return _fetch_from_modelscope(repo_id, revision or "master")
    else:
        raise ValueError(f"不支持的平台: {platform}，请使用 'hf' 或 'ms'")


def _fetch_from_huggingface(repo_id: str, revision: str, use_mirror: bool = False) -> dict[str, str]:
    """从 HuggingFace 获取 safetensors 文件的 sha256

    Args:
        repo_id: 仓库 ID
        revision: 版本/分支
        use_mirror: 是否使用镜像（hf-mirror.com）
    """
    import os

    # 如果需要使用镜像，设置环境变量
    original_endpoint = os.environ.get("HF_ENDPOINT")
    if use_mirror and not original_endpoint:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    from huggingface_hub import HfApi, list_repo_files

    try:
        api = HfApi()
        all_files = list_repo_files(repo_id=repo_id, revision=revision)
        safetensors_files = [f for f in all_files if f.endswith(".safetensors")]

        if not safetensors_files:
            return {}

        paths_info = api.get_paths_info(
            repo_id=repo_id,
            paths=safetensors_files,
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
        # 恢复原始环境变量
        if use_mirror and not original_endpoint:
            os.environ.pop("HF_ENDPOINT", None)
        elif original_endpoint:
            os.environ["HF_ENDPOINT"] = original_endpoint


def _fetch_from_modelscope(repo_id: str, revision: str) -> dict[str, str]:
    """从 ModelScope 获取 safetensors 文件的 sha256"""
    from modelscope.hub.api import HubApi

    api = HubApi()
    files_info = api.get_model_files(model_id=repo_id, revision=revision)

    result = {}
    for file_info in files_info:
        filename = file_info.get("Name", file_info.get("Path", ""))
        if filename.endswith(".safetensors"):
            sha256 = file_info.get("Sha256", file_info.get("sha256", None))
            result[filename] = sha256

    return result


def verify_model_integrity_with_progress(
    repo_type: Literal["huggingface", "modelscope"],
    repo_id: str,
    local_dir: Path,
    progress_callback=None,
    verbose: bool = False,
    use_mirror: bool = False,
    files_to_verify: list[str] | None = None,
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

        official_hashes = fetch_model_sha256(repo_id, platform, use_mirror=use_mirror)

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
