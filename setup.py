"""KTransformers: CPU-GPU heterogeneous fine-tuning for MoE models.

``pip install ktransformers`` installs:
- ``ktransformers`` — integration glue + auto-patching for HF ecosystem
- ``kt-kernel`` — C++ AMX kernel engine (dependency)
- ``accelerate-kt`` — accelerate fork with KT plugin support
- ``transformers-kt`` — transformers fork with KT training integration
"""

from pathlib import Path
from setuptools import find_packages, setup

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
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        f"kt-kernel=={_v}",
        "transformers-kt>=5.6.0",
        "accelerate-kt>=1.14.0",
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
