"""Lightweight top-level package: pip install ktransformers -> installs kt-kernel.

Extras:
  - ktransformers[sft] installs transformers-kt + accelerate-kt
  - ktransformers[vlm-sft] adds the verified ms-swift Conv3D compatibility dependency
  - ktransformers[sglang] installs sglang-kt
"""
from pathlib import Path
from setuptools import setup

_version_file = Path(__file__).resolve().parent / "version.py"
_ns = {}
exec(_version_file.read_text(), _ns)
_v = _ns["__version__"]

setup(
    version=_v,
    install_requires=[
        f"kt-kernel=={_v}",
    ],
    extras_require={
        "sft": [
            "transformers-kt==5.6.0.post1",
            "accelerate-kt==1.14.0.post1",
        ],
        "vlm-sft": [
            "transformers-kt==5.6.0.post1",
            "accelerate-kt==1.14.0.post1",
            f"kt-kernel[vlm-sft]=={_v}",
        ],
        "sglang": [
            f"sglang-kt=={_v}",
        ],
    },
)
