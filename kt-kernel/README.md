# KT-Kernel

High-performance kernel operations for KTransformers, featuring CPU-optimized MoE inference with AMX, AVX, and KML support.

## Note

**Current Support Status:**
- ✅ **Intel CPUs with AMX**: Fully supported
- ⚠️ **Universal CPU with llamafile**: In preview, not yet fully complete
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
./install.sh
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

You need both GPU weights and CPU weights for heterogeneous inference:

**GPU Weights:** Use the original / quantized model weights.

**CPU Weights:** Quantize to AMX-optimized format using the conversion script:

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \  # Depends on your GPU weights type: fp8, fp16, or bf16
  --output /path/to/cpu-weights \
  --quant-method int8  # or int4
```

**Supported input formats:** FP8, FP16, BF16 → INT4/INT8.

For more details, see:
- [CPU Weights conversion](#cpu-weights-for-cold-experts-on-cpu)
- [GPU Weights quantization](#gpu-weights-for-hot-experts-on-gpu)

**Note:** LLAMAFILE backend supports GGUF format directly, but this feature is still in preview.

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

This example demonstrates the full workflow from downloading weights to launching the server.

**Hardware Configuration:**
- **GPU**: NVIDIA RTX 4090 24GB
- **CPU**: 2x Intel Xeon Gold 6454S (64 physical cores total, 128 threads, 2 NUMA nodes)
- **Model**: [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- **GPU Weights**: BF16 original weights
- **CPU Weights**: AMXINT8 quantized

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
- `--kt-max-deferred-experts-per-token 2`: Enable pipelined execution - allows CPU to process next batch while GPU completes current batch

#### Step 1: Download model weights

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Download model from Hugging Face
hf download Qwen/Qwen3-30B-A3B --local-dir /mnt/data/models/Qwen3-30B-A3B
```

#### Step 2: Convert to CPU weights (AMXINT8)

```bash
python scripts/convert_cpu_weights.py \
  --input-path /mnt/data/models/Qwen3-30B-A3B \
  --input-type bf16 \
  --output /mnt/data/models/Qwen3-30B-A3B-INT8 \
  --quant-method int8
```

#### Step 3: Launch SGLang server

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

### KT-Kernel Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `--kt-method` | CPU inference backend method | `AMXINT4`, `AMXINT8`, or `LLAMAFILE` (preview) |
| `--kt-weight-path` | Path to quantized CPU weights | `/path/to/cpu-weights` |
| `--kt-cpuinfer` | Number of CPU inference threads | `64` (adjust based on CPU cores) |
| `--kt-threadpool-count` | Number of thread pools for parallel execution | `2` (typically 1-4) |
| `--kt-num-gpu-experts` | Number of experts to keep on GPU | `32` (remaining experts go to CPU) |
| `--kt-max-deferred-experts-per-token` | Number of experts per token to defer for pipelined execution | `2` (0 to disable, 1-2 recommended) |

**Parameter Guidelines:**

- **`kt-method`**: Choose based on your CPU and weight format:
  - `AMXINT4`: Best performance on AMX CPUs with INT4 quantized weights (May cause huge accuracy drop for some models, e.g., Qwen3-30B-A3B)
  - `AMXINT8`: Higher accuracy with INT8 quantized weights on AMX CPUs
  - `LLAMAFILE`: Preview support for GGUF format (not fully complete)

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
  - `1-2`: Deferred execution (better latency, requires tuning) - recommended
  - `3-4`: Higher deferred count (possible but rarely beneficial)

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
