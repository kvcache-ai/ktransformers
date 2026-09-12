#!/usr/bin/env python3
"""Source-checkout entry point; installed users can run kt-prepare-kimi-data."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[4] / "kt-kernel/python_tools/prepare_kimi_data.py"),
        run_name="__main__",
    )
