"""
KTransformers CLI - A unified command-line interface for KTransformers.

This CLI provides a user-friendly interface to all KTransformers functionality,
including model inference, fine-tuning, benchmarking, and more.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


try:
    __version__ = version("kt-kernel")
except PackageNotFoundError:
    _version_ns = {}
    _root_version_file = Path(__file__).resolve().parents[3] / "version.py"
    if _root_version_file.exists():
        exec(_root_version_file.read_text(encoding="utf-8"), _version_ns)
        __version__ = _version_ns.get("__version__", "0.6.1")
    else:
        __version__ = "0.6.1"
