#!/usr/bin/env python3
"""Compatibility entry point; the converter is included in the kt-kernel wheel."""

from kt_kernel.sft.convert_kt_to_sglang_adapter import *  # noqa: F401,F403
from kt_kernel.sft.convert_kt_to_sglang_adapter import main

if __name__ == "__main__":
    main()
