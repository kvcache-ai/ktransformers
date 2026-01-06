"""
Model Scanner

Scans directories for model files (safetensors, gguf) and identifies models
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple
import os


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
                continue

            # Detect format
            if safetensors_files and gguf_files:
                # Mixed format - issue warning
                warnings.append(
                    f"Mixed format detected in {root_path}: "
                    f"{len(safetensors_files)} safetensors + {len(gguf_files)} gguf files. "
                    "Please separate into different folders and re-scan."
                )
                dirs[:] = []  # Don't descend
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

            # Don't descend into subdirectories (this folder is a model)
            dirs[:] = []

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
