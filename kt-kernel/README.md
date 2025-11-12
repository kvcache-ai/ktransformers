# KT-Kernel

High-performance kernel operations for KTransformers, featuring CPU-optimized MoE inference with AMX, AVX, and KML support.

## Note

**Current Support Status:**
- ✅ **Intel CPUs with AMX**: Fully supported
- ⚠️ **LLAMAFILE backend**: In preview, not yet fully complete
- ⚠️ **AMD CPUs with BLIS**: Upcoming, not yet fully integrated

## Features

- **AMX Optimization**: Intel AMX (Advanced Matrix Extensions) support for INT4/INT8 quantized MoE inference
- **Multi-Backend**: Unified `KTMoEWrapper` API supporting multiple backends (AMXINT4, AMXINT8, LLAMAFILE*)
- **Flexible Backends**: AVX512, AVX2 via pluggable backend architecture
- **Efficient MoE**: Optimized Mixture-of-Experts operations with NUMA-aware memory management
- **Async Execution**: Non-blocking `submit_forward` / `sync_forward` API for improved pipelining
- **Easy Integration**: Clean Python API with automatic backend selection

**Note**: *LLAMAFILE backend support is currently in *preview* and not yet fully complete.

## Installation

### Prerequisites

First, initialize git submodules:
```bash
git submodule update --init --recursive
```

### Quick Installation (Recommended)

Step 0: Create and activate a conda environment (recommended):

```bash
conda create -n kt-kernel python=3.11 -y
conda activate kt-kernel
```

You can now install in two clear steps using the same script.

Option A: Two-step (explicit)

```bash
# 1) Install system prerequisites (cmake, hwloc, pkg-config)
./install.sh deps

# 2) Build and install kt-kernel (auto-detects CPU)
#    By default, the script cleans the local ./build directory before compiling.
./install.sh build
```

Option B: One-step (deps + build)

```bash
# Simple one-command installation
./install.sh            # same as: ./install.sh all
# Skip deps step if you already installed them
./install.sh all --skip-deps
```

The install script will:
- Auto-detect CPU capabilities (AMX support)
- Install `cmake` via conda (if available)
- Install system dependencies (`libhwloc-dev`, `pkg-config`) based on your OS

**What gets configured automatically:**
- AMX CPU detected → `NATIVE + AMX=ON`
- No AMX detected → `NATIVE + AMX=OFF`

⚠️ **Important for LLAMAFILE backend users:** If you have an AMX-capable CPU and plan to use the LLAMAFILE backend, do NOT use auto-detection. Use manual mode with `AVX512` or `AVX2` instead of `NATIVE` to avoid compilation issues (see below).

### Manual Configuration (Advanced)

If you need specific build options (e.g., for LLAMAFILE backend, compatibility, or binary distribution):

```bash
# Example for LLAMAFILE backend on AMX CPU with AVX512
export CPUINFER_CPU_INSTRUCT=AVX512  # Options: NATIVE, AVX512, AVX2
export CPUINFER_ENABLE_AMX=OFF       # Options: ON, OFF

# Run with manual mode (build only)
./install.sh build --manual
```

For advanced build options and binary distribution, see the [Build Configuration](#build-configuration) section. If you encounter issues, refer to [Error Troubleshooting](#error-troubleshooting).

## Verification

```bash
python -c "from kt_kernel import KTMoEWrapper; print('✓ kt-kernel installed successfully')"
```

## Usage

```python
from kt_kernel import KTMoEWrapper

# Initialize the MoE wrapper
wrapper = KTMoEWrapper(
    layer_idx=0,
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=4096,
    moe_intermediate_size=14336,
    num_gpu_experts=2,
    cpuinfer_threads=32,
    threadpool_count=2,
    weight_path="/path/to/weights",
    chunked_prefill_size=512,
    method="AMXINT4"  # Options: "AMXINT4", "AMXINT8", "LLAMAFILE" (preview)
)

# Load weights (from disk - pre-quantized)
wrapper.load_weights(physical_to_logical_map)

# Or load weights from tensors (online quantization)
wrapper.load_weights_from_tensors(gate_proj, up_proj, down_proj, physical_to_logical_map)

# Run inference
output = wrapper.forward(hidden_states, topk_ids, topk_weights, cuda_stream)

# Or use async API for better performance
wrapper.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
# ... do other work ...
output = wrapper.sync_forward(hidden_states, cuda_stream)
```

### Advanced Options

```python
# Initialize with additional options
wrapper = KTMoEWrapper(
    layer_idx=0,
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=4096,
    moe_intermediate_size=14336,
    num_gpu_experts=2,
    cpuinfer_threads=32,
    threadpool_count=2,
    weight_path="/path/to/weights",
    chunked_prefill_size=512,
    method="AMXINT4",
    cpu_save=False,  # Keep weights in CPU memory after loading
    max_deferred_experts_per_token=0  # Number of experts to defer (for pipelined execution)
)

# Pre-allocate buffers for specific batch sizes (improves performance)
KTMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])

# Query captured batch sizes
batch_sizes = KTMoEWrapper.get_capture_batch_sizes()

# Clear buffer cache to free memory
KTMoEWrapper.clear_buffer_cache()
```

## Build Configuration

### Manual Installation

If you prefer manual installation without the `install.sh` script, follow these steps:

#### 1. Install System Dependencies

**Prerequisites:**
- `cmake` (recommended: `conda install -y cmake`)
- `libhwloc-dev` and `pkg-config`

#### 2. Set Build Configuration

**Core Options:**

| Variable | Options | Description |
|----------|---------|-------------|
| `CPUINFER_CPU_INSTRUCT` | `NATIVE`, `AVX512`, `AVX2`, `FANCY` | CPU instruction set to use |
| `CPUINFER_ENABLE_AMX` | `ON`, `OFF` | Enable Intel AMX support |
| `CPUINFER_BUILD_TYPE` | `Release`, `Debug`, `RelWithDebInfo` | Build type (default: `Release`) |
| `CPUINFER_PARALLEL` | Number | Parallel build jobs (default: auto-detect) |
| `CPUINFER_VERBOSE` | `0`, `1` | Verbose build output (default: `0`) |

**Instruction Set Details:**

- **`NATIVE`**: Auto-detect and use all available CPU instructions (`-march=native`) - **Recommended for best performance**
- **`AVX512`**: Explicit AVX512 support for Skylake-SP and Cascade Lake
- **`AVX2`**: AVX2 support for maximum compatibility
- **`FANCY`**: AVX512 with full extensions (AVX512F/BW/DQ/VL/VNNI) for Ice Lake+ and Zen 4+. Use this when building pre-compiled binaries to distribute to users with modern CPUs. For local builds, prefer `NATIVE` for better performance.

**Example Configurations:**

```bash
# Maximum performance on AMX CPU
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON

# AVX512 CPU without AMX
export CPUINFER_CPU_INSTRUCT=AVX512
export CPUINFER_ENABLE_AMX=OFF

# Compatibility build
export CPUINFER_CPU_INSTRUCT=AVX2
export CPUINFER_ENABLE_AMX=OFF

# Debug build for development
export CPUINFER_BUILD_TYPE=Debug
export CPUINFER_VERBOSE=1
```

#### 3. Build and Install

```bash
# Editable installation (for development)
pip install -e .

# Standard installation
pip install .
```

## Error Troubleshooting

### CUDA Not Found

```
 -- Looking for a CUDA compiler - NOTFOUND
  CMake Error at CMakeLists.txt:389 (message):
    KTRANSFORMERS_USE_CUDA=ON but CUDA compiler not found
```

Make sure you have the CUDA toolkit installed and `nvcc` is in your system PATH.

Try `export CMAKE_ARGS="-D CMAKE_CUDA_COMPILER=$(which nvcc)"` and reinstall again.

### hwloc Not Found

Run `sudo apt install libhwloc-dev` if on a Debian-based system or build from source: https://www.open-mpi.org/projects/hwloc/.

```
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.2.tar.gz
tar -xzf hwloc-2.12.2.tar.gz
cd hwloc-2.12.2
./configure
make
sudo make install
```

## Weight Quantization

KT-Kernel provides weight quantization tools for CPU-GPU hybrid inference (e.g., integrating with SGLang). Both tools work together to enable heterogeneous expert placement across CPUs and GPUs.

### CPU Weights (for "cold" experts on CPU)

Quantize weights to INT4/INT8 format optimized for AMX inference:

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4
```

**Supported formats:** FP8, FP16, BF16 → INT4/INT8

### GPU Weights (for "hot" experts on GPU)

Apply GPTQ quantization to model weights:

```bash
# Install additional dependencies first
pip install accelerate transformers llmcompressor datasets

# Quantize GPU weights
python scripts/convert_gpu_weights.py \
  --model_id /path/to/model \
  --output_dir /path/to/output \
  --quant_type W4A16
```

**Supported types:** W4A16 (GPTQ4), W8A16 (GPTQ8)

---

For detailed documentation, advanced options, and low-memory mode, see [scripts/README.md](scripts/README.md).

## Before Commit!
your msg should match: Conventional Commits (https://www.conventionalcommits.org/) <br>and format your code before commit:
```shell
cmake -B build
cd build
make format
```
and you may need a new clang-format at least 18, use this command in conda env:
```shell
conda install -c conda-forge clang-format=18
rm -rf build
```
and you may need black for python format:
```shell
conda install black
```
