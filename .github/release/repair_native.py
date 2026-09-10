"""Auditwheel repair with CUDA dependencies provided by the pinned Torch stack."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from four_main import inspect_wheel, save_json

# Do not embed another copy of NVIDIA's large libraries or give them private
# auditwheel SONAMEs. Torch 2.9.1 pins their corresponding nvidia-* wheels.
EXTERNAL = (
    "libcuda.so.1", "libtorch.so", "libtorch_cpu.so", "libtorch_cuda.so",
    "libtorch_python.so", "libc10.so", "libc10_cuda.so", "libcudart.so.12",
    "libcublas.so.12", "libcublasLt.so.12", "libnvrtc.so.12",
    "libnvrtc-builtins.so.12.8", "libnvJitLink.so.12",
)


def repair(raw, destination, evidence):
    destination.mkdir(exist_ok=False)
    for path in sorted(raw.glob("*.whl")):
        entry = inspect_wheel(path)
        if entry["name"] in {"kt-kernel", "sgl-kernel-kt"}:
            command = [sys.executable, "-m", "auditwheel", "repair", str(path), "--plat", "manylinux_2_35_x86_64"]
            for soname in EXTERNAL:
                command += ["--exclude", soname]
            subprocess.run(command + ["-w", str(destination)], check=True)
        else:
            shutil.copyfile(path, destination / path.name)
    save_json(evidence / "native-repair.json", {"external_libraries": list(EXTERNAL), "provider": "pinned torch==2.9.1 CUDA 12.8 dependencies; NVIDIA driver supplies libcuda"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    repair(args.raw, args.output, args.evidence)
