"""KTransformers: CPU-GPU heterogeneous fine-tuning for MoE models.

`ktransformers` is a lightweight meta-package for the KT fine-tuning stack.
Install `transformers-kt` and `accelerate-kt` alongside it for the full
beginner setup.
"""

from pathlib import Path
from setuptools import setup

_version_file = Path(__file__).resolve().parent / "version.py"
_ns = {}
exec(_version_file.read_text(), _ns)
_v = _ns["__version__"]

setup(
    name="ktransformers",
    version=_v,
    description="CPU-GPU heterogeneous fine-tuning for MoE models",
    long_description=open(Path(__file__).resolve().parent / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="kvcache-ai",
    url="https://github.com/kvcache-ai/ktransformers",
    packages=[],
    python_requires=">=3.10",
    install_requires=[
        f"kt-kernel=={_v}",
        "peft>=0.18.0",
        "torch>=2.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
