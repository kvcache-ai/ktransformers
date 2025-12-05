# KT-Kernel

High-performance kernel operations for KTransformers, featuring CPU-optimized MoE inference with AMX, AVX, KML and blis (amd library) support.

- [KT-Kernel](#kt-kernel)
  - [Note](#note)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Quick Installation (Recommended)](#quick-installation-recommended)
    - [Manual Configuration (Advanced)](#manual-configuration-advanced)
  - [Verification](#verification)
  - [Integration with SGLang](#integration-with-sglang)
    - [Installation Steps](#installation-steps)
      - [1. Install SGLang](#1-install-sglang)
      - [2. Prepare Weights](#2-prepare-weights)
      - [3. Launch SGLang Server](#3-launch-sglang-server)
    - [Complete Example: Qwen3-30B-A3B](#complete-example-qwen3-30b-a3b)
      - [Option A: AMX Backend (AMXINT8)](#option-a-amx-backend-amxint8)
      - [Option B: LLAMAFILE Backend (GGUF)](#option-b-llamafile-backend-gguf)
    - [KT-Kernel Parameters](#kt-kernel-parameters)
  - [Direct Python API Usage](#direct-python-api-usage)
    - [Advanced Options](#advanced-options)
  - [Build Configuration](#build-configuration)
    - [Manual Installation](#manual-installation)
      - [1. Install System Dependencies](#1-install-system-dependencies)
      - [2. Set Build Configuration](#2-set-build-configuration)
      - [3. Build and Install](#3-build-and-install)
  - [Error Troubleshooting](#error-troubleshooting)
    - [CUDA Not Found](#cuda-not-found)
    - [hwloc Not Found](#hwloc-not-found)
  - [Weight Quantization](#weight-quantization)
  - [Before Commit!](#before-commit)
## Note

**Current Support Status:**
- ✅ **Intel CPUs with AMX**: Fully supported (using weights converted to INT4/INT8 format)
- ✅ **Universal CPU (llamafile backend)**: Supported (using GGUF-format weights)
- ✅ **AMD CPUs with BLIS**: Supported (for int8 prefill & decode)

## Features

- **CPU-Optimized MoE Kernels**: High-throughput MoE expert kernels optimized for instruction sets.
- **AMX INT4/INT8 Backend**: INT4 / INT8 quantized expert inference backend for AMX-capable servers.
- **Llamafile CPU Backend**: AVX2/AVX512-based MoE backend built on Llamafile for universal CPU deployment.
- **NUMA-Aware Execution**: Thread pool and memory layout designed for multi-socket / multi-NUMA machines.

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

Option A: Two-step (specify dependencies installation and build separately)

```bash
# 1) Install system prerequisites (cmake, hwloc, pkg-config)
./install.sh deps

# 2) Build and install kt-kernel (auto-detects CPU instruction set)
#    By default, the script cleans the local ./build directory before compiling
./install.sh build
```

Option B: One-step

```bash
./install.sh
```

The install script will:
- Auto-detect CPU capabilities (AMX support)
- Install `cmake` via conda (if available)
- Install system dependencies (`libhwloc-dev`, `pkg-config`) based on your OS

**What gets configured automatically:**
- AMX CPU detected → `NATIVE + AMX=ON`
- No AMX detected → `NATIVE + AMX=OFF`

⚠️ **Important for LLAMAFILE backend users:**
If you have an AMX-capable CPU but plan to use the LLAMAFILE backend, do NOT use the default auto-detection build.
Use "manual mode" with `CPUINFER_CPU_INSTRUCT` set to `AVX512` or `AVX2` instead of `NATIVE` to avoid compilation issues (see below).

⚠️ **Important for BLIS AMD backend users:**
for the installation guide, see this [issue](https://github.com/kvcache-ai/ktransformers/issues/1601)


### Manual Configuration (Advanced)

If you need specific build options (e.g., for LLAMAFILE backend, compatibility, or binary distribution):

```bash
# Example for LLAMAFILE backend on AMX CPU with AVX512
export CPUINFER_CPU_INSTRUCT=AVX512  # Options: NATIVE, AVX512, AVX2, FANCY
export CPUINFER_ENABLE_AMX=OFF       # Options: ON, OFF

# Build only (skip auto-detection of instruction set)
./install.sh build --manual
```

For advanced build options and binary distribution, see the [Build Configuration](#build-configuration) section. If you encounter issues, refer to [Error Troubleshooting](#error-troubleshooting).

## Verification

```bash
python -c "from kt_kernel import KTMoEWrapper; print('✓ kt-kernel installed successfully')"
```

## Integration with SGLang

KT-Kernel can be used standalone via [Direct Python API](#direct-python-api-usage) or integrated with SGLang for production deployment. This section describes SGLang integration to enable CPU-GPU heterogeneous inference, where "hot" experts run on GPU and "cold" experts run on CPU for optimal resource utilization.

### Installation Steps

#### 1. Install SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

#### 2. Prepare Weights

You need both GPU weights and CPU-side expert weights for heterogeneous inference. The exact format depends on the backend:

**GPU Weights (for all backends):**  
Use the model weights required by SGLang for GPU inference (for example, the original or already-quantized model directory from Hugging Face).

**CPU Weights (AMX backend: `AMXINT4` / `AMXINT8`):**
Quantize weights to AMX-optimized INT4/INT8 format using the provided script:

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/cpu-weights \
  --quant-method int8  # or int4 or moe_int8 (for amd now) 
```

- `--input-path`: Path to GPU-side original weights
- `--input-type`: Depends on your GPU weights type (`fp8`, `fp16`, or `bf16`)

In SGLang integration, `--kt-weight-path` should point to this converted CPU weights directory.

**Supported input formats:** FP8, FP16, BF16 → INT4/INT8.

**CPU Weights (LLAMAFILE backend: `LLAMAFILE`):**
LLAMAFILE uses pre-quantized **GGUF** weights on the CPU side directly, without running `convert_cpu_weights.py`. You need to:

- Download a GGUF model directly from the web (e.g., GGUF repos on Hugging Face / Modelscope);
- In SGLang integration, use that GGUF directory as `--kt-weight-path`.
  KT-Kernel supports multiple GGUF quantization formats such as `Q4_KM`, `Q4_K`, `Q5_K`, etc. Choose based on your latency and accuracy requirements.

#### 3. Launch SGLang Server

Start the SGLang server with your normal SGLang parameters, and add the following KT-Kernel specific parameters to enable CPU-GPU heterogeneous inference:

**KT-Kernel Parameters to Add:**
- `--kt-method`: Backend method (AMXINT4, AMXINT8, or LLAMAFILE)
- `--kt-weight-path`: Path to the converted CPU weights
- `--kt-cpuinfer`: Number of CPU inference threads (set to physical cores)
- `--kt-threadpool-count`: Number of thread pools (set to NUMA node count)
- `--kt-num-gpu-experts`: Number of experts to keep on GPU
- `--kt-max-deferred-experts-per-token`: Deferred experts for pipelined execution

Example:
```bash
python -m sglang.launch_server \
  [your normal SGLang parameters...] \
  --kt-method AMXINT8 \
  --kt-weight-path /path/to/cpu-weights \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

See [KT-Kernel Parameters](#kt-kernel-parameters) section below for detailed parameter tuning guidelines.

### Complete Example: Qwen3-30B-A3B

This example demonstrates the full workflow from downloading weights to launching the server, showing both **AMX backend** and **LLAMAFILE backend** options.

**Hardware Configuration:**
- **GPU**: NVIDIA RTX 4090 24GB
- **CPU**: 2x Intel Xeon Gold 6454S (64 physical cores total, 128 threads, 2 NUMA nodes)
- **Model**: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

**How to verify your system configuration:**
```bash
# Check CPU configuration
lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core|Socket\(s\)|NUMA node\(s\)"
# Expected output example:
CPU(s):                                  128
Thread(s) per core:                      2
Socket(s):                               2
NUMA node(s):                            2
# → Physical cores = CPU(s) / Thread(s) per core = 128 / 2 = 64
```

**Parameter Rationale:**
- `--kt-cpuinfer 64`: Set to physical cores (64), not hyperthreads (128)
- `--kt-threadpool-count 2`: 2 NUMA nodes detected (dual-socket system)
- `--kt-num-gpu-experts 32`: With 24GB GPU memory, we can fit ~32 experts on GPU for this model (varies by model architecture and actual memory usage)
- `--kt-max-deferred-experts-per-token 2`: Enable pipelined execution; allows CPU to process next batch while GPU completes current batch

---

#### Option A: AMX Backend (AMXINT8)

For Intel CPUs with AMX instruction set support.

**Step 1: Download model weights**

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Download model from Hugging Face
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /mnt/data/models/Qwen3-30B-A3B
```

**Step 2: Convert to CPU weights (AMXINT8)**

```bash
python scripts/convert_cpu_weights.py \
  --input-path /mnt/data/models/Qwen3-30B-A3B \
  --input-type bf16 \
  --output /mnt/data/models/Qwen3-30B-A3B-INT8 \
  --quant-method int8
```

**Step 3: Launch SGLang server**

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /mnt/data/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path /mnt/data/models/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

---

#### Option B: LLAMAFILE Backend (GGUF)

For universal CPUs (no AMX required), using pre-quantized GGUF weights directly.

**Step 1: Download GPU weights (original model)**

```bash
pip install huggingface-hub

huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /mnt/data/models/Qwen3-30B-A3B
```

**Step 2: Download CPU weights (GGUF format)**

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf \
  --local-dir /mnt/data/models/Qwen3-30B-A3B-Q4_K_M
```

**Step 3: Launch SGLang server**

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /mnt/data/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method LLAMAFILE \
  --kt-weight-path /mnt/data/models/Qwen3-30B-A3B-Q4_K_M \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

### KT-Kernel Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--kt-method` | CPU inference backend method | `AMXINT4`, `AMXINT8`, `RAWINT4`, or `LLAMAFILE` |
| `--kt-weight-path` | Path to quantized CPU weights | `/path/to/cpu-weights` |
| `--kt-cpuinfer` | Number of CPU inference threads | `64` (adjust based on CPU cores) |
| `--kt-threadpool-count` | Number of thread pools for parallel execution | `2` (typically 1-4) |
| `--kt-num-gpu-experts` | Number of experts to keep on GPU | `32` (remaining experts go to CPU) |
| `--kt-max-deferred-experts-per-token` | Number of experts per token to defer for pipelined execution | `2` (0 to disable, 1-4 recommended) |
| `--kt-gpu-prefill-token-threshold` | Token count threshold for prefill strategy (RAWINT4 only) | ~`400` |

**Parameter Guidelines:**

- **`kt-method`**: Choose based on your CPU and weight format:
  - `AMXINT4`: Best performance on AMX CPUs with INT4 quantized weights (May cause huge accuracy drop for some models, e.g., Qwen3-30B-A3B)
  - `AMXINT8`: Higher accuracy with INT8 quantized weights on AMX CPUs
  - `RAWINT4`: Native INT4 weights shared by CPU and GPU (AMX backend only, currently supports Kimi-K2-Thinking model). See [Kimi-K2-Thinking Native Tutorial](../doc/en/Kimi-K2-Thinking-Native.md) for details.
  - `LLAMAFILE`: GGUF-based backend

- **`kt-cpuinfer`**: Set to the number of **physical CPU cores** (not hyperthreads).
  - Check physical cores: `lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core"`
  - Physical cores = CPU(s) / Thread(s) per core
  - Example: If CPU(s)=128 and Thread(s) per core=2, then physical cores = 64
  - **Important**: Do NOT set to hyperthread count - this will degrade performance

- **`kt-threadpool-count`**: Set to the number of **NUMA nodes**.
  - Check NUMA count: `lscpu | grep "NUMA node(s)"`
  - Or use: `numactl --hardware | grep "available"`
  - **Note**: NUMA node count is NOT necessarily the number of physical CPUs
    - It represents memory domains, which may be divided within a single CPU or across multiple CPUs
    - Use the NUMA node count from `lscpu`, regardless of physical CPU count
  - Typical values: 1-2 for single-socket, 2-4 for dual-socket systems
  - This enables better memory bandwidth utilization across NUMA domains

- **`kt-num-gpu-experts`**: Determine based on GPU memory and profiling:
  - More GPU experts = lower latency but higher GPU memory usage (May cause OOM)

- **`kt-max-deferred-experts-per-token`**: Enables pipelined execution:
  - `0`: Synchronous execution (simpler, higher latency)
  - `1-4`: Deferred execution (recommended range; good latency/quality balance, requires tuning)
  - `5-7`: Highest latency reduction but may introduce noticeable accuracy loss; use with care

- **`kt-gpu-prefill-token-threshold`** (RAWINT4 only): Controls prefill strategy for native INT4 inference:
  - **≤ threshold**: Uses hybrid CPU+GPU prefill. No extra VRAM needed, but performance degrades slowly as token count increases.
  - **> threshold**: Uses layerwise GPU prefill. Performance scales better with longer sequences, but requires ~9GB+ extra VRAM.
  - Only applicable when `--kt-method RAWINT4` is used. Currently supports Kimi-K2-Thinking model only.

## Direct Python API Usage

For standalone usage without SGLang, you can use KT-Kernel directly via Python API:

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
    method="AMXINT4"  # Options: "AMXINT4", "AMXINT8", "LLAMAFILE"
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

For AMX backends (`AMXINT4` / `AMXINT8`), CPU-side experts must be converted to AMX-friendly INT4/INT8 format using the provided script:

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4
```

**Supported formats:** FP8, FP16, BF16 → INT4/INT8

For LLAMAFILE backend (`LLAMAFILE`), CPU-side experts are loaded directly from **GGUF** weights. You do **not** need to run the AMX conversion script; instead, download a GGUF model from the web (e.g., a GGUF repo on Hugging Face) and point `weight_path` / SGLang `--kt-weight-path` (or `--model` when appropriate) to that GGUF directory. KT-Kernel supports multiple GGUF quantization types such as `Q4_KM`, `Q4_K`, `Q5_K`, etc.

---

For detailed documentation, advanced options, and low-memory mode, see [scripts/README.md](scripts/README.md).

## Before Commit!

Commit messages should follow the Conventional Commits specification: https://www.conventionalcommits.org/

Please format your code before committing:

```shell
cmake -B build
cd build
make format
```

You may need a newer clang-format (at least version 18). In a conda environment:

```shell
conda install -c conda-forge clang-format=18
rm -rf build
```

It's also recommended to install black for Python code formatting:

```shell
conda install black
```
