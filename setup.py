"""Meta-package: pip install ktransformers → installs kt-kernel + sglang-kt."""
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
        f"sglang-kt=={_v}",
    ],
)
