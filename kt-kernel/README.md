# KT-Kernel

High-performance kernel operations for KTransformers, featuring CPU-optimized MoE inference with AMX, AVX, and KML support.

## Features

- **AMX Optimization**: Intel AMX (Advanced Matrix Extensions) support for INT4/INT8 quantized MoE inference
- **Multi-Backend**: AVX512, AVX2, and ARM KML support
- **Efficient MoE**: Optimized Mixture-of-Experts operations with NUMA-aware memory management
- **Easy Integration**: Clean Python API with `AMXMoEWrapper` and future wrapper support

## Installation

### Standard Installation
```bash
pip install .
```

### Editable Installation (Development)
```bash
pip install -e .
```

## Usage

```python
from kt_kernel import AMXMoEWrapper

# Initialize the MoE wrapper
wrapper = AMXMoEWrapper(
    layer_idx=0,
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=4096,
    moe_intermediate_size=14336,
    num_gpu_experts=2,
    cpuinfer_threads=32,
    subpool_count=2,
    amx_weight_path="/path/to/weights",
    chunked_prefill_size=512
)

# Load weights
wrapper.load_weights(physical_to_logical_map)

# Run inference
output = wrapper.forward(hidden_states, topk_ids, topk_weights, cuda_stream)
```

## Build Configuration

### CPU Instruction Set Tuning
```bash
export CPUINFER_CPU_INSTRUCT=FANCY   # Options: NATIVE|FANCY|AVX512|AVX2
pip install .
```

### AMX Configuration
```bash
export CPUINFER_ENABLE_AMX=ON        # Enable/disable AMX support
pip install .
```

### Build Type
```bash
export CPUINFER_BUILD_TYPE=Release   # Debug|RelWithDebInfo|Release
pip install .
```

### Parallel Build
```bash
export CPUINFER_PARALLEL=8           # Number of parallel jobs
pip install .
```

### Verbose Build
```bash
export CPUINFER_VERBOSE=1
pip install .
```

## Verification

```bash
python -c "from kt_kernel import AMXMoEWrapper; print('✓ kt-kernel installed successfully')"
```
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
