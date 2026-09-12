#!/usr/bin/env python3
"""Source-checkout entry point; installed users can run kt-convert-lora."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "python_tools/convert_lora.py"),
        run_name="__main__",
    )
