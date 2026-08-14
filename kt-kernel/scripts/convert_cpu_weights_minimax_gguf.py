#!/usr/bin/env python3
"""
DEPRECATED — no pre-conversion step needed anymore.

GGUF → AMXINT8 now happens automatically at first boot:

    python -m sglang.launch_server --model <hf-dir> \
        --kt-weight-path /models/DeepSeek-UD-Q4_K_XL --kt-method AMXINT8 \
        --kt-cpuinfer 64 --kt-threadpool-count 2

The C++ loader dequantizes 64-row strips straight from the mmap'd GGUF blocks
(AVX-512, no BF16/FP32 materialization) and writes the packed INT8 cache
(`<root>/<key>/_layer_<N>/_numa_<M>/`) as a side effect of the first boot.
Every later boot loads that cache directly ("boots at today's AMXINT8 speed").

Cache controls:
    KT_GGUF_CACHE_DIR=<dir>   relocate the cache
    KT_GGUF_CACHE=0           disable the cache (quantize every boot)
    KT_GGUF_CACHE=refresh     force a rebuild

This script previously ran the Python-side GGUFToAMXINT8Adapter
(gguf_amxint8_adapter.py / ggml_dequant.py), which materialized full BF16
tensors per layer and was the source of the RAM explosion. Those modules are
gone; the C++ path replaces them.
"""
import sys

print(
    "convert_cpu_weights_minimax_gguf.py is deprecated: GGUF -> AMXINT8 "
    "conversion now happens automatically at first boot via "
    "--kt-weight-path <gguf-dir> --kt-method AMXINT8 (see the module docstring).",
    file=sys.stderr,
)
sys.exit(1)