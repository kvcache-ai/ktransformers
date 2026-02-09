"""
Model Scanner

Scans directories for model files (safetensors, gguf) and identifies models
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict
from collections import defaultdict
import os
import subprocess
import json


@dataclass
class ScannedModel:
    """Temporary structure for scanned model information"""

    path: str  # Absolute path to model directory
    format: str  # "safetensors" | "gguf" | "mixed"
    size_bytes: int  # Total size in bytes
    file_count: int  # Number of model files
    files: List[str]  # List of model file names

    @property
    def size_gb(self) -> float:
        """Get size in GB"""
        return self.size_bytes / (1024**3)

    @property
    def folder_name(self) -> str:
        """Get the folder name (default model name)"""
        return Path(self.path).name


class ModelScanner:
    """Scanner for discovering models in directory trees"""

    def __init__(self, min_size_gb: float = 10.0):
        """
        Initialize scanner

        Args:
            min_size_gb: Minimum folder size in GB to be considered a model
        """
        self.min_size_bytes = int(min_size_gb * 1024**3)

    def scan_directory(
        self, base_path: Path, exclude_paths: Optional[Set[str]] = None
    ) -> Tuple[List[ScannedModel], List[str]]:
        """
        Scan directory tree for models

        Args:
            base_path: Root directory to scan
            exclude_paths: Set of absolute paths to exclude from results

        Returns:
            Tuple of (valid_models, warnings)
            - valid_models: List of ScannedModel instances
            - warnings: List of warning messages
        """
        if not base_path.exists():
            raise ValueError(f"Path does not exist: {base_path}")

        if not base_path.is_dir():
            raise ValueError(f"Path is not a directory: {base_path}")

        exclude_paths = exclude_paths or set()
        results: List[ScannedModel] = []
        warnings: List[str] = []

        # Walk the directory tree
        for root, dirs, files in os.walk(base_path):
            root_path = Path(root).resolve()

            # Skip if already registered
            if str(root_path) in exclude_paths:
                dirs[:] = []  # Don't descend into this directory
                continue

            # Check for model files
            safetensors_files = [f for f in files if f.endswith(".safetensors")]
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not safetensors_files and not gguf_files:
                continue  # No model files in this directory

            # Calculate total size
            model_files = safetensors_files + gguf_files
            total_size = self._calculate_total_size(root_path, model_files)

            # Check if size meets minimum threshold
            if total_size < self.min_size_bytes:
                continue  # Too small, but keep scanning subdirectories

            # Detect format
            if safetensors_files and gguf_files:
                # Mixed format - issue warning
                warnings.append(
                    f"Mixed format detected in {root_path}: "
                    f"{len(safetensors_files)} safetensors + {len(gguf_files)} gguf files. "
                    "Please separate into different folders and re-scan."
                )
                dirs[:] = []  # Don't descend into mixed format directories
                continue

            # Determine format
            format_type = "safetensors" if safetensors_files else "gguf"

            # Create scanned model
            scanned = ScannedModel(
                path=str(root_path),
                format=format_type,
                size_bytes=total_size,
                file_count=len(model_files),
                files=model_files,
            )

            results.append(scanned)

            # Continue scanning subdirectories - they might also contain models
            # Each subdirectory will be independently checked for size >= 10GB

        return results, warnings

    def scan_single_path(self, path: Path) -> Optional[ScannedModel]:
        """
        Scan a single path for model files

        Args:
            path: Path to scan

        Returns:
            ScannedModel instance or None if not a valid model
        """
        if not path.exists() or not path.is_dir():
            return None

        # Find model files
        safetensors_files = list(path.glob("*.safetensors"))
        gguf_files = list(path.glob("*.gguf"))

        if not safetensors_files and not gguf_files:
            return None

        # Check for mixed format
        if safetensors_files and gguf_files:
            raise ValueError(
                f"Mixed format detected: {len(safetensors_files)} safetensors + "
                f"{len(gguf_files)} gguf files. Please use a single format."
            )

        # Calculate size
        model_files = [f.name for f in safetensors_files + gguf_files]
        total_size = self._calculate_total_size(path, model_files)

        # Determine format
        format_type = "safetensors" if safetensors_files else "gguf"

        return ScannedModel(
            path=str(path.resolve()),
            format=format_type,
            size_bytes=total_size,
            file_count=len(model_files),
            files=model_files,
        )

    def _calculate_total_size(self, directory: Path, filenames: List[str]) -> int:
        """
        Calculate total size of specified files in directory

        Args:
            directory: Directory containing the files
            filenames: List of filenames to sum

        Returns:
            Total size in bytes
        """
        total = 0
        for filename in filenames:
            file_path = directory / filename
            if file_path.exists():
                try:
                    total += file_path.stat().st_size
                except OSError:
                    # File might be inaccessible, skip it
                    pass
        return total


# Convenience functions


def scan_directory(
    base_path: Path, min_size_gb: float = 10.0, exclude_paths: Optional[Set[str]] = None
) -> Tuple[List[ScannedModel], List[str]]:
    """
    Convenience function to scan a directory

    Args:
        base_path: Root directory to scan
        min_size_gb: Minimum folder size in GB
        exclude_paths: Set of paths to exclude

    Returns:
        Tuple of (models, warnings)
    """
    scanner = ModelScanner(min_size_gb=min_size_gb)
    return scanner.scan_directory(base_path, exclude_paths)


def scan_single_path(path: Path) -> Optional[ScannedModel]:
    """
    Convenience function to scan a single path

    Args:
        path: Path to scan

    Returns:
        ScannedModel or None
    """
    scanner = ModelScanner()
    return scanner.scan_single_path(path)


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human-readable string

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "42.3 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# ===== Fast Scanning with Find Command and Tree-based Root Detection =====


def find_files_fast(mount_point: str, pattern: str, max_depth: int = 6, timeout: int = 30) -> List[str]:
    """
    Use find command to quickly locate files

    Args:
        mount_point: Starting directory
        pattern: File pattern (e.g., "config.json", "*.gguf")
        max_depth: Maximum directory depth (default: 6)
        timeout: Command timeout in seconds

    Returns:
        List of absolute file paths
    """
    try:
        # Use shell=False for better security and handling of special characters in paths
        cmd = ["find", mount_point, "-maxdepth", str(max_depth), "-name", pattern, "-type", "f"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            stderr=subprocess.DEVNULL,
        )

        # Return results even if returncode is non-zero (due to permission errors)
        # As long as we got some output
        if result.stdout:
            return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def is_valid_model_directory(directory: Path, min_size_gb: float = 10.0) -> Tuple[bool, Optional[str]]:
    """
    Check if a directory is a valid model directory

    Args:
        directory: Path to check
        min_size_gb: Minimum size in GB

    Returns:
        (is_valid, model_type) where model_type is "safetensors", "gguf", or None
    """
    if not directory.exists() or not directory.is_dir():
        return False, None

    has_config = (directory / "config.json").exists()
    safetensors_files = list(directory.glob("*.safetensors"))
    gguf_files = list(directory.glob("*.gguf"))

    # Determine model type
    model_type = None
    if (has_config and safetensors_files) or safetensors_files:
        model_type = "safetensors"
    elif gguf_files:
        model_type = "gguf"
    else:
        return False, None

    # Check size - only count model files (fast!)
    total_size = 0
    if model_type == "safetensors":
        for f in safetensors_files:
            try:
                total_size += f.stat().st_size
            except OSError:
                pass
    else:  # gguf
        for f in gguf_files:
            try:
                total_size += f.stat().st_size
            except OSError:
                pass

    size_gb = total_size / (1024**3)
    if size_gb < min_size_gb:
        return False, None

    return True, model_type


def scan_all_models_fast(mount_points: List[str], min_size_gb: float = 10.0, max_depth: int = 6) -> List[str]:
    """
    Fast scan for all model paths using find command

    Args:
        mount_points: List of mount points to scan
        min_size_gb: Minimum model size in GB
        max_depth: Maximum search depth (default: 6)

    Returns:
        List of valid model directory paths
    """
    model_paths = set()

    for mount in mount_points:
        if not os.path.exists(mount):
            continue

        # Find all config.json files
        config_files = find_files_fast(mount, "config.json", max_depth=max_depth)
        for config_path in config_files:
            model_dir = Path(config_path).parent
            is_valid, model_type = is_valid_model_directory(model_dir, min_size_gb)
            if is_valid:
                model_paths.add(str(model_dir.resolve()))

        # Find all *.gguf files
        gguf_files = find_files_fast(mount, "*.gguf", max_depth=max_depth)
        for gguf_path in gguf_files:
            model_dir = Path(gguf_path).parent
            is_valid, model_type = is_valid_model_directory(model_dir, min_size_gb)
            if is_valid:
                model_paths.add(str(model_dir.resolve()))

    return sorted(model_paths)


def get_root_subdirs() -> List[str]:
    """
    Get subdirectories of / that are worth scanning

    Filters out system paths only

    Returns:
        List of directories to scan
    """
    # System paths to exclude
    excluded = {
        "dev",
        "proc",
        "sys",
        "run",
        "boot",
        "tmp",
        "usr",
        "lib",
        "lib64",
        "bin",
        "sbin",
        "etc",
        "opt",
        "var",
        "snap",
    }

    scan_dirs = []

    try:
        for entry in os.scandir("/"):
            if not entry.is_dir():
                continue

            # Skip excluded paths
            if entry.name in excluded:
                continue

            scan_dirs.append(entry.path)

    except PermissionError:
        pass

    return sorted(scan_dirs)


def scan_directory_for_models(directory: str, min_file_size_gb: float = 2.0) -> Dict[str, tuple]:
    """
    Scan a directory for models using find command with size filter

    Uses find -size +2G to only locate large model files (>=2GB)

    Args:
        directory: Directory to scan
        min_file_size_gb: Minimum individual file size in GB (default: 2.0)

    Returns:
        Dict mapping model_path -> (model_type, size_bytes, file_count, files)
    """
    model_info = {}

    # Convert GB to find's format (e.g., 2GB = +2G)
    if min_file_size_gb >= 1.0:
        size_filter = f"+{int(min_file_size_gb)}G"
    else:
        size_mb = int(min_file_size_gb * 1024)
        size_filter = f"+{size_mb}M"

    # 1. Find *.gguf files >= 2GB
    gguf_cmd = ["find", directory, "-name", "*.gguf", "-type", "f", "-size", size_filter, "-printf", "%p\t%s\n"]
    result = subprocess.run(gguf_cmd, shell=False, capture_output=True, text=True, timeout=120, stderr=subprocess.DEVNULL)

    # Group by directory
    gguf_dirs = defaultdict(list)
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        file_path, size_str = parts
        file_path_obj = Path(file_path)
        dir_path = str(file_path_obj.parent)
        gguf_dirs[dir_path].append((file_path_obj.name, int(size_str)))

    # Add all gguf directories
    for dir_path, files in gguf_dirs.items():
        total_size = sum(size for _, size in files)
        model_info[dir_path] = ("gguf", total_size, len(files), [name for name, _ in files])

    # 2. Find *.safetensors files >= 2GB
    safetensors_cmd = ["find", directory, "-name", "*.safetensors", "-type", "f", "-size", size_filter, "-printf", "%p\t%s\n"]
    result = subprocess.run(safetensors_cmd, shell=False, capture_output=True, text=True, timeout=120, stderr=subprocess.DEVNULL)

    # Group by directory
    safetensors_dirs = defaultdict(list)
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        file_path, size_str = parts
        file_path_obj = Path(file_path)
        dir_path = str(file_path_obj.parent)
        safetensors_dirs[dir_path].append((file_path_obj.name, int(size_str)))

    # 3. Check each safetensors directory for config.json
    for dir_path, files in safetensors_dirs.items():
        if os.path.exists(os.path.join(dir_path, "config.json")):
            total_size = sum(size for _, size in files)
            model_info[dir_path] = ("safetensors", total_size, len(files), [name for name, _ in files])

    return model_info


def scan_all_models_with_info(
    mount_points: Optional[List[str]] = None, min_size_gb: float = 10.0, max_depth: int = 6
) -> Dict[str, tuple]:
    """
    Fast scan with parallel directory scanning

    Strategy:
    1. Use provided directories or auto-detect root subdirectories
    2. Scan each directory in parallel (one thread per directory)
    3. Use find -size +2G to find large model files (>=2GB)

    Args:
        mount_points: Specific directories to scan, or None to auto-detect from / subdirs
        min_size_gb: Not used anymore (kept for API compatibility)
        max_depth: Not used anymore (kept for API compatibility)

    Returns:
        Dict mapping model_path -> (model_type, size_bytes, file_count, files)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Get directories to scan
    if mount_points is None:
        # Get root subdirectories (exclude system paths)
        scan_dirs = get_root_subdirs()
    else:
        scan_dirs = mount_points

    if not scan_dirs:
        return {}

    model_info = {}

    # Scan each directory in parallel (max 8 concurrent)
    # Use 2GB threshold to find model files
    with ThreadPoolExecutor(max_workers=min(len(scan_dirs), 8)) as executor:
        futures = {executor.submit(scan_directory_for_models, d, 2.0): d for d in scan_dirs}

        for future in as_completed(futures):
            try:
                dir_results = future.result()
                model_info.update(dir_results)
            except Exception as e:
                # Skip directories with errors
                pass

    return model_info


def find_model_roots_from_paths(model_paths: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Find optimal root paths from model paths using tree-based algorithm

    Algorithm:
    1. Build path tree with all intermediate paths
    2. DFS to calculate f(x) = subtree sum (number of models in subtree)
    3. Find roots where f(parent) = f(x) > max(f(children))

    Args:
        model_paths: List of model directory paths

    Returns:
        (root_paths, subtree_sizes) where:
        - root_paths: List of inferred root directories
        - subtree_sizes: Dict mapping each root to number of models
    """
    if not model_paths:
        return [], {}

    # 1. Build path set (including all intermediate paths)
    all_paths = set()
    model_set = set(model_paths)

    for model_path in model_paths:
        path = Path(model_path)
        for i in range(1, len(path.parts) + 1):
            all_paths.add(str(Path(*path.parts[:i])))

    # 2. Build parent-child relationships
    children_map = defaultdict(list)
    for path in all_paths:
        path_obj = Path(path)
        if len(path_obj.parts) > 1:
            parent = str(path_obj.parent)
            if parent in all_paths:
                children_map[parent].append(path)

    # 3. DFS to calculate f(x) and max_child_f(x)
    f = {}  # path -> subtree sum
    max_child_f = {}  # path -> max(f(children))
    visited = set()

    def dfs(path: str) -> int:
        if path in visited:
            return f[path]
        visited.add(path)

        # Current node weight (1 if it's a model path, 0 otherwise)
        weight = 1 if path in model_set else 0

        # Recursively calculate children
        children = children_map.get(path, [])
        if not children:
            # Leaf node
            f[path] = weight
            max_child_f[path] = 0
            return weight

        # Calculate f values for all children
        children_f_values = [dfs(child) for child in children]

        # Calculate f(x) and max_child_f(x)
        f[path] = weight + sum(children_f_values)
        max_child_f[path] = max(children_f_values) if children_f_values else 0

        return f[path]

    # Find top-level nodes (no parent in all_paths)
    top_nodes = []
    for path in all_paths:
        parent = str(Path(path).parent)
        if parent not in all_paths or parent == path:
            top_nodes.append(path)

    # Execute DFS from all top nodes
    for top in top_nodes:
        dfs(top)

    # 4. Find root nodes: f(parent) = f(x) >= max(f(children))
    # Note: Use >= instead of > to handle the case where a directory contains only one model
    candidate_roots = []
    for path in all_paths:
        # Skip model paths themselves (leaf nodes in model tree)
        if path in model_set:
            continue

        parent = str(Path(path).parent)

        # Check condition: f(parent) = f(x) and f(x) >= max(f(children))
        if parent in f and f.get(parent, 0) == f.get(path, 0):
            if f.get(path, 0) >= max_child_f.get(path, 0) and f.get(path, 0) > 0:
                candidate_roots.append(path)

    # 5. Remove redundant roots (prefer deeper paths)
    # If a root is an ancestor of another root with the same f value, remove it
    roots = []
    candidate_roots_sorted = sorted(candidate_roots, key=lambda p: -len(Path(p).parts))

    for root in candidate_roots_sorted:
        # Check if this root is a parent of any already selected root
        is_redundant = False
        for selected in roots:
            if selected.startswith(root + "/"):
                # selected is a child of root
                # Only keep root if it has more models (shouldn't happen by algorithm)
                if f.get(root, 0) == f.get(selected, 0):
                    is_redundant = True
                    break

        if not is_redundant:
            # Also filter out very shallow paths (< 3 levels)
            if len(Path(root).parts) >= 3:
                roots.append(root)

    # Build subtree sizes for roots
    subtree_sizes = {root: f.get(root, 0) for root in roots}

    return sorted(roots), subtree_sizes


@dataclass
class ModelRootInfo:
    """Information about a detected model root path"""

    path: str
    model_count: int
    models: List[ScannedModel]


def discover_models(
    mount_points: Optional[List[str]] = None, min_size_gb: float = 10.0, max_depth: int = 6
) -> List[ScannedModel]:
    """
    Discover all model directories on the system

    Fast scan using find command to locate all models that meet the criteria

    Args:
        mount_points: List of mount points to scan (None = auto-detect)
        min_size_gb: Minimum model size in GB (default: 10.0)
        max_depth: Maximum search depth (default: 6)

    Returns:
        List of ScannedModel sorted by path
    """
    # Auto-detect mount points if not provided
    if mount_points is None:
        mount_points = _get_mount_points()

    # Fast scan with cached info (only scan once!)
    model_info = scan_all_models_with_info(mount_points, min_size_gb, max_depth)

    if not model_info:
        return []

    # Convert to ScannedModel objects
    results = []
    for model_path, (model_type, total_size, file_count, files) in model_info.items():
        results.append(
            ScannedModel(path=model_path, format=model_type, size_bytes=total_size, file_count=file_count, files=files)
        )

    # Sort by path
    results.sort(key=lambda m: m.path)
    return results


def _get_mount_points() -> List[str]:
    """
    Get all valid mount points from /proc/mounts, filtering out system paths

    Returns:
        List of mount point paths suitable for model storage
        (excludes root "/" to avoid scanning entire filesystem)
    """
    mount_points = set()

    # System paths to exclude (unlikely to contain model files)
    excluded_paths = [
        "/snap/",
        "/proc/",
        "/sys/",
        "/run/",
        "/boot",
        "/dev/",
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc",
        "/opt",
        "/var",
        "/tmp",
    ]

    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue

                device, mount_point, fs_type = parts[0], parts[1], parts[2]

                # Filter out pseudo filesystems
                pseudo_fs = {
                    "proc",
                    "sysfs",
                    "devpts",
                    "tmpfs",
                    "devtmpfs",
                    "cgroup",
                    "cgroup2",
                    "pstore",
                    "bpf",
                    "tracefs",
                    "debugfs",
                    "hugetlbfs",
                    "mqueue",
                    "configfs",
                    "securityfs",
                    "fuse.gvfsd-fuse",
                    "fusectl",
                    "squashfs",
                    "overlay",  # snap packages
                }

                if fs_type in pseudo_fs:
                    continue

                # Skip root directory (too large to scan)
                if mount_point == "/":
                    continue

                # Filter out system paths
                if any(mount_point.startswith(x) for x in excluded_paths):
                    continue

                # Only include if it exists and is readable
                if os.path.exists(mount_point) and os.access(mount_point, os.R_OK):
                    mount_points.add(mount_point)

        # If no mount points found, add common data directories
        if not mount_points:
            # Add /home if it exists and is not already a separate mount point
            common_paths = ["/home", "/data", "/mnt"]
            for path in common_paths:
                if os.path.exists(path) and os.access(path, os.R_OK):
                    mount_points.add(path)

    except (FileNotFoundError, PermissionError):
        # Fallback to common paths
        mount_points = {"/home", "/mnt", "/data"}

    return sorted(mount_points)
