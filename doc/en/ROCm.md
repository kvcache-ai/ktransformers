# ROCm Support for ktransformers (Beta)

## Introduction

### Overview
In our effort to expand GPU architecture support beyond NVIDIA, we are excited to introduce **AMD GPU support through ROCm** in ktransformers (Beta release). This implementation has been tested and developed using EPYC 9274F processors and AMD Radeon 7900xtx GPUs.

## Installation Guide

### 1. Install ROCm Driver
Begin by installing the ROCm drivers for your AMD GPU:
- [Official ROCm Installation Guide for Radeon GPUs](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html)

### 2. Set Up Conda Environment
We recommend using Miniconda3/Anaconda3 for environment management:

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Create environment
conda create --name ktransformers python=3.11
conda activate ktransformers

# Install required libraries
conda install -c conda-forge libstdcxx-ng

# Verify GLIBCXX version (should include 3.4.32)
strings ~/anaconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
```

> **Note:** Adjust the Anaconda path if your installation directory differs from `~/anaconda3`

### 3. Install PyTorch for ROCm
Install PyTorch with ROCm 6.2.4 support:

```bash
pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2.4
pip3 install packaging ninja cpufeature numpy
```

> **Tip:** For other ROCm versions, visit [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)

### Hygon DCU / DTK Notes

Hygon DCU uses a ROCm-compatible DTK stack. For DCU systems, use the Hygon DCU
PyTorch environment that matches your DTK release instead of installing the
generic PyPI `torch` package or the official AMD ROCm wheel.

For example, a reported working `gfx936` setup uses a Hygon DCU PyTorch image
from the SourceFind/Hygon developer image portal:

https://sourcefind.cn/#/image/dcu/pytorch

The reported environment provides:

- DTK 26.04, typically under `/opt/dtk`
- PyTorch `2.5.1+das.opt1.dtk2604`
- Python 3.10

Before building, verify that the DCU PyTorch package is active:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.__file__)"
```

Then build `kt-kernel` without letting pip replace the vendor PyTorch package:

```bash
export CPUINFER_USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx936
export ROCM_PATH=/opt/dtk  # change this if DTK is installed elsewhere

cd kt-kernel
pip install . --no-build-isolation --no-deps
```

> **Tip:** Keep `--no-deps` when building in a vendor PyTorch environment. A
> plain `pip install .` may resolve `kt-kernel`'s normal `torch` dependency and
> shadow or replace the installed DCU PyTorch package with a generic torch wheel.

### 4. Build ktransformers

```bash
# Clone repository
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init

# Optional: Compile web interface
# See: api/server/website.md

# Install dependencies
bash install.sh
```

## Running DeepSeek-R1 Models

### Configuration for 24GB VRAM GPUs
Use our optimized configuration for constrained VRAM:

```bash
python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-R1 \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/rocm/DeepSeek-V3-Chat.yaml \
  --cpu_infer <cpu_cores + 1>
```

> **Beta Note:** Current Q8 linear implementation (Marlin alternative) shows suboptimal performance. Expect optimizations in future releases.

### Configuration for 40GB+ VRAM GPUs
For better performance on high-VRAM GPUs:

1. Modify `DeepSeek-V3-Chat.yaml`:
   ```yaml
   # Replace all instances of:
   KLinearMarlin → KLinearTorch
   ```

2. Execute with:
   ```bash
   python ktransformers/local_chat.py \
     --model_path deepseek-ai/DeepSeek-R1 \
     --gguf_path <path_to_gguf_files> \
     --optimize_config_path <modified_yaml_path> \
     --cpu_infer <cpu_cores + 1>
   ```
> **Tip:** If you got 2 * 24GB AMD GPUS, you may also do the same modify and run `ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml` instead.

## Known Limitations
- Marlin operations not supported on ROCm platform
- Current Q8 linear implementation shows reduced performance (Beta limitation)
