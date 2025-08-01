# SOSP'25 Artifacts Evaluation

## Overview

This repository contains the open-source code for the paper **[SOSP'25] KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models**.

## Hardware Requirements

To reproduce the main results presented in our paper, the following hardware environment is required:

- **CPU**: Dual-socket Intel Xeon processors with AMX instruction set support (e.g., 4th generation Xeon processors)
- **Memory**: 1.5TB total DRAM capacity
- **GPU**: One CUDA-compatible GPU with at least 40GB VRAM
- **Storage**: 5TB available disk space

**Note:** If artifact reviewers do not have access to the required hardware environment, please contact us for SSH remote access to a pre-configured system that meets these specifications.

## Preparation

### Prerequisites

- **CUDA**: Version 12.1 or higher. If not installed, download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- **Operating System**: Linux x86_64 with GCC/G++ ≥11 and CMake ≥3.25
- **Python Environment**: We recommend using [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) or [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh) to create a virtual environment with Python 3.12

### Environment Setup

1. **Create and activate conda environment:**
   ```bash
   conda create --name ktransformers python=3.12
   conda activate ktransformers
   ```

2. **Install required C++ standard library:**
   ```bash
   conda install -c conda-forge libstdcxx-ng
   ```

3. **Install PyTorch and dependencies:**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip3 install packaging ninja cpufeature numpy
   pip3 install flash-attn --no-build-isolation
   ```

4. **Install NUMA support libraries:**
   ```bash
   sudo apt-get install libnuma-dev libhwloc-dev # example for Ubuntu
   ```

### Installation

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git checkout -b sosp25-ae origin/sosp25-ae
git submodule update --init --recursive
bash install.sh
```

## Baseline (custom-llama.cpp) Installation

```bash
cd sosp25-ae/baseline-custom-llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

## Model Preparation

### Download Safetensors Models

Download the following models to your local disk with sufficient storage capacity:

- [DeepSeek-V3-0324-BF16](https://huggingface.co/unsloth/DeepSeek-V3-0324-BF16) (NOT required for fast run version)
- [DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) (NOT required for fast run version)
- [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) 

### Convert Models for baseline (custom-llama.cpp)

Navigate to the custom-llama.cpp directory and convert models to GGUF format.

**Note:** Even if you already have GGUF format models locally, please re-convert them using the commands below, as custom-llama.cpp has made modifications to the file format.

```bash
cd sosp25-ae/baseline-custom-llama.cpp

# DeepSeek-V3 (NOT required for fast run version)
python convert_hf_to_gguf.py [/path/to/ds3_safetensors_dir] --outtype f16 --outfile [/path/to/ds3_f16.gguf]
./build/bin/llama-quantize [/path/to/ds3_f16.gguf] [/path/to/ds3_q4_0.gguf] q4_0

# DeepSeek-V2.5 (NOT required for fast run version)
python convert_hf_to_gguf.py [/path/to/ds2_safetensors_dir] --outtype f16 --outfile [/path/to/ds2_f16.gguf]
./build/bin/llama-quantize [/path/to/ds2_f16.gguf] [/path/to/ds2_q8_0.gguf] q8_0

# Qwen2
python convert_hf_to_gguf.py [/path/to/qw2_safetensors_dir] --outtype f16 --outfile [/path/to/qw2_f16.gguf]
./build/bin/llama-quantize [/path/to/qw2_f16.gguf] [/path/to/qw2_q8_0.gguf] q8_0
```

### Configure Model Paths

Before running experiments, edit `set_model_paths.sh` to update the default paths with your downloaded and generated model locations.

## Running Experiments

The evaluation script executes the following experiments sequentially:

* **[`Figure11-prefill/`](./Figure11-prefill)**: Evaluates KTransformers' speedup over two baseline systems across different prefill lengths.  
  **Expected results**: KTransformers should significantly outperform both baselines, with the best-case speedup exceeding ~15x.

* **[`Figure12-decode/`](./Figure12-decode)**: Evaluates KTransformers' decode speedup over two baseline systems, including additional acceleration from the proposed expert deferral mechanism.  
  **Expected results**: KTransformers should outperform both baselines, with expert deferral providing additional acceleration. The best-case overall speedup can exceed ~4x.

* **[`Figure13-breakdown/`](./Figure13-breakdown)**: Analyzes the contribution of each optimization component for full precision models, comparing against pure PyTorch based implementation (Fiddler). Optimizations include: fused MoE kernels based on AVX-512/AMX instruction sets (+v/+m), dynamic work scheduling (+d), NUMA-aware tensor parallelism (+n), and CUDA Graph (+c).  
  **Expected results**: In prefill stage, AVX-512 MoE kernel (+v) performs worse than baseline, and CUDA Graph (+c) shows no effect. In decode stage, both AVX-512 MoE kernel (+v) and AMX MoE kernel (+m) outperform baseline, with AVX-512 slightly outperforming AMX. All other optimizations should contribute positively to overall performance.

* **[`Table2-accuracy/`](./Table2-accuracy)**: Validates the impact of expert deferral mechanism on model accuracy across various benchmarks.  
  **Expected results**: Applying expert deferral mechanism cause minimal accuracy degradation. Individual data points may fluctuate due to model randomness, but overall average accuracy drop should not exceed 1%.

Each experiment directory contains a `plot.py` script for visualizing the results. It also includes `reference_xxx` results, which correspond to the evaluation results reported in our paper. These were generated at the time of paper submission and differ slightly from the current open-sourced implementation. However, the overall performance trends remain consistent.

### Fast Run (Quick Validation)

For quick validation, we provide a fast run version that focuses on Qwen2 model experiments and includes only the MBPP benchmark in Table 2's accuracy evaluation. This process takes approximately 3 hours:

```bash
bash run_all.sh fast
```

### Full Run (Complete Evaluation)

This version executes all experiments described in the paper's evaluation section. The complete process requires approximately 2~3 days:

```bash
bash run_all.sh full
```
