"""Top-level Python package for KTransformers.

The runtime kernels live in kt-kernel. Optional SFT support is activated
via pip install "ktransformers[sft]" which adds transformers-kt and
accelerate-kt to the environment.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_repo_version() -> str:
    ns: dict[str, str] = {}
    exec((Path(__file__).resolve().parents[1] / 'version.py').read_text(), ns)
    return ns['__version__']


try:
    __version__ = version('ktransformers')
except PackageNotFoundError:
    __version__ = _read_repo_version()


def has_sft_support() -> bool:
    try:
        import kt_kernel.sft  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ['__version__', 'has_sft_support']
