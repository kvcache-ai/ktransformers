#!/usr/bin/env python3
"""
Standalone Model Integrity Verifier

Verifies safetensors files against HuggingFace repository SHA256 checksums.

Usage:
    python standalone_verify.py <local_model_path> <hf_repo_id>

Environment Variables:
    HF_TOKEN: HuggingFace API token (optional, for private repos)
    HF_ENDPOINT: Custom HuggingFace endpoint (optional, e.g., https://hf-mirror.com)

Examples:
    python standalone_verify.py /path/to/model deepseek-ai/DeepSeek-V3
    HF_TOKEN=hf_xxx python standalone_verify.py /path/to/model org/private-model
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
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


def calculate_local_sha256(local_dir: Path, file_pattern: str = "*.safetensors") -> Dict[str, str]:
    """
    Calculate SHA256 hashes for files in a directory using parallel processing.

    Args:
        local_dir: Directory to scan
        file_pattern: Glob pattern for files to hash

    Returns:
        Dictionary mapping filename to SHA256 hash
    """
    result = {}

    if not local_dir.exists():
        print(f"Error: Directory does not exist: {local_dir}")
        return result

    # Get all files
    files_to_hash = [f for f in local_dir.glob(file_pattern) if f.is_file()]
    total_files = len(files_to_hash)

    if total_files == 0:
        print(f"Warning: No {file_pattern} files found in {local_dir}")
        return result

    # Use min(16, total_files) workers to avoid over-spawning processes
    max_workers = min(16, total_files)
    print(f"Using {max_workers} parallel workers for SHA256 calculation")

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
                print(f"  [{completed_count}/{total_files}] ✓ {filename} ({file_size_mb:.1f} MB)")
            except Exception as e:
                file_path = future_to_file[future]
                print(f"  [{completed_count}/{total_files}] ✗ {file_path.name} - Error: {str(e)}")

    return result


def fetch_huggingface_sha256(repo_id: str, revision: str = "main") -> Dict[str, str]:
    """
    Fetch SHA256 hashes from HuggingFace repository.

    Args:
        repo_id: Repository ID (e.g., "deepseek-ai/DeepSeek-V3")
        revision: Branch/revision (default: "main")

    Returns:
        Dictionary mapping filename to SHA256 hash
    """
    try:
        from huggingface_hub import HfApi, list_repo_files
    except ImportError:
        print("Error: huggingface_hub is not installed")
        print("Install it with: pip install huggingface-hub")
        sys.exit(1)

    # Read HF_TOKEN from environment if available
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"Using HF_TOKEN from environment")

    try:
        api = HfApi(token=token)

        print(f"Fetching file list from {repo_id} (revision: {revision})...")
        all_files = list_repo_files(repo_id=repo_id, revision=revision, token=token)
        safetensors_files = [f for f in all_files if f.endswith(".safetensors")]

        if not safetensors_files:
            print(f"Warning: No .safetensors files found in repository")
            return {}

        print(f"Found {len(safetensors_files)} safetensors files")
        print(f"Fetching SHA256 hashes from {repo_id}...")

        paths_info = api.get_paths_info(
            repo_id=repo_id,
            paths=safetensors_files,
            revision=revision,
            token=token,
        )

        result = {}
        for file_info in paths_info:
            if hasattr(file_info, "lfs") and file_info.lfs is not None:
                sha256 = file_info.lfs.sha256
            else:
                sha256 = getattr(file_info, "blob_id", None)
            result[file_info.path] = sha256

        print(f"✓ Fetched {len(result)} file hashes from remote")
        return result

    except Exception as e:
        print(f"Error fetching from HuggingFace: {e}")

        # Suggest mirror if it's a connection error
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            print("\nTry using HuggingFace mirror:")
            print("  export HF_ENDPOINT=https://hf-mirror.com")

        sys.exit(1)


def verify_model(local_dir: Path, repo_id: str, revision: str = "main", verbose: bool = False) -> bool:
    """
    Verify local model integrity against HuggingFace repository.

    Args:
        local_dir: Local directory containing model files
        repo_id: HuggingFace repository ID
        revision: Branch/revision to verify against
        verbose: Show detailed comparison for each file

    Returns:
        True if verification passed, False otherwise
    """
    print("\n" + "=" * 60)
    print(f"Model Integrity Verification")
    print("=" * 60)
    print(f"Local path:  {local_dir}")
    print(f"Repository:  {repo_id}")
    print(f"Revision:    {revision}")
    print("=" * 60 + "\n")

    # 1. Fetch official SHA256 hashes from HuggingFace
    official_hashes = fetch_huggingface_sha256(repo_id, revision)

    if not official_hashes:
        print("\nError: No hashes fetched from remote repository")
        return False

    print()

    # 2. Calculate local SHA256 hashes
    print("Calculating SHA256 for local files...")
    local_hashes = calculate_local_sha256(local_dir, "*.safetensors")

    if not local_hashes:
        print("\nError: No local hashes calculated")
        return False

    print(f"✓ Calculated {len(local_hashes)} local file hashes\n")

    # 3. Compare hashes
    print(f"Comparing {len(official_hashes)} files...\n")

    files_passed = 0
    files_failed = []
    files_missing = []

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
            print(f"  [{idx}/{len(official_hashes)}] ✗ {file_basename} - MISSING")
            if verbose:
                print(f"      Remote: {official_hash}")
                print(f"      Local:  <missing>")
        elif local_hash.lower() != official_hash.lower():
            files_failed.append(filename)
            print(f"  [{idx}/{len(official_hashes)}] ✗ {file_basename} - HASH MISMATCH")
            if verbose:
                print(f"      Remote: {official_hash}")
                print(f"      Local:  {local_hash}")
        else:
            files_passed += 1
            print(f"  [{idx}/{len(official_hashes)}] ✓ {file_basename}")
            if verbose:
                print(f"      SHA256: {official_hash}")

    # 4. Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"Total files:   {len(official_hashes)}")
    print(f"✓ Passed:      {files_passed}")
    print(f"✗ Failed:      {len(files_failed)}")
    print(f"✗ Missing:     {len(files_missing)}")
    print("=" * 60)

    if files_failed:
        print("\nFiles with hash mismatch:")
        for f in files_failed:
            print(f"  - {f}")

    if files_missing:
        print("\nMissing files:")
        for f in files_missing:
            print(f"  - {f}")

    print()

    # Return result
    if files_failed or files_missing:
        print("❌ VERIFICATION FAILED")
        print("\nSome files are corrupted or missing. Consider re-downloading the model.")
        return False
    else:
        print("✅ VERIFICATION PASSED")
        print("\nAll files verified successfully!")
        return True


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify model integrity against HuggingFace repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_verify.py /path/to/model deepseek-ai/DeepSeek-V3
  python standalone_verify.py /path/to/model org/model --revision main --verbose

Environment Variables:
  HF_TOKEN      HuggingFace API token (for private repos)
  HF_ENDPOINT   Custom HuggingFace endpoint (e.g., https://hf-mirror.com)
        """,
    )

    parser.add_argument("local_path", type=str, help="Local model directory path")
    parser.add_argument("repo_id", type=str, help="HuggingFace repository ID (e.g., deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--revision", "-r", type=str, default="main", help="Repository revision/branch (default: main)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed SHA256 comparison")

    args = parser.parse_args()

    # Validate local path
    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists():
        print(f"Error: Local path does not exist: {local_path}")
        sys.exit(1)

    if not local_path.is_dir():
        print(f"Error: Local path is not a directory: {local_path}")
        sys.exit(1)

    # Run verification
    success = verify_model(local_path, args.repo_id, args.revision, args.verbose)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
