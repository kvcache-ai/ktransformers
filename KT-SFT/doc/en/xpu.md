# Intel GPU Support for KTransformers (Beta)

## Introduction

### Overview
We are excited to introduce **Intel GPU support** in KTransformers (Beta release). This implementation has been tested and developed using Intel Xeon Scalable processors and Intel Arc GPUs (such as A770 and B580).

## Installation Guide

### 1. Install Intel GPU Driver
Begin by installing the GPU drivers for your Intel GPU:
- [Official GPU Installation Guide for Intel GPUs](https://dgpu-docs.intel.com/driver/overview.html)

To verify that the kernel and compute drivers are installed and functional:

```bash
clinfo --list | grep Device
 `-- Device #0: 13th Gen Intel(R) Core(TM) i9-13900K
 `-- Device #0: Intel(R) Arc(TM) A770 Graphics
 `-- Device #0: Intel(R) UHD Graphics 770
```

> [!Important]
> Ensure that **Resizable BAR** is enabled in your system's BIOS before proceeding. This is essential for optimal GPU performance and to avoid potential issues such as `Bus error (core dumped)`. For detailed steps, please refer to the official guidance [here](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html).

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

### 3. Install PyTorch and IPEX-LLM
Install PyTorch with XPU backend support and [IPEX-LLM](https://github.com/intel/ipex-llm):

```bash
pip install ipex-llm[xpu_2.6]==2.3.0b20250518 --extra-index-url https://download.pytorch.org/whl/xpu
pip uninstall torch torchvision torchaudio
pip install torch==2.7+xpu torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu # install torch2.7
pip uninstall intel-opencl-rt dpcpp-cpp-rt
```

### 4. Build ktransformers

```bash
# Clone repository
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init

# Install dependencies
bash install.sh --dev xpu
```

## Running DeepSeek-R1 Models

### Configuration for 16B VRAM GPUs
Use our optimized configuration for constrained VRAM:

```bash
export SYCL_CACHE_PERSISTENT=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-R1 \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/xpu/DeepSeek-V3-Chat.yaml \
  --cpu_infer <cpu_cores + 1> \
  --device xpu \
  --max_new_tokens 200
```

## Known Limitations
- Serving function is not supported on Intel GPU platform for now

## Troubleshooting
1. Best Known Config (BKC) to obtain best performance

To obtain best performance on Intel GPU platform, we recommend to lock GPU frequency and set CPU to performance mode by below settings.
```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
echo 0 | sudo tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias
# 2400 is max frequency for Arc A770
sudo xpu-smi config -d 0 -t 0 --frequencyrange 2400,2400
# 2850 is max frequency for Arc B580
# sudo xpu-smi config -d 0 -t 0 --frequencyrange 2850,2850
```

2. Runtime error like `xpu/sycl/TensorCompareKernels.cpp:163: xxx. Aborted (core dumped)`

This error is mostly related to GPU driver. If you meet such error, you could update your `intel-level-zero-gpu` to `1.3.29735.27-914~22.04` (which is a verified version by us) by below command.
```bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list
sudo apt update
# or sudo apt update --allow-insecure-repositories
sudo apt install intel-level-zero-gpu=1.3.29735.27-914~22.04
```

3. `ImportError: cannot import name 'intel' from 'triton._C.libtriton'`

Installing Triton causes pytorch-triton-xpu to stop working. You can resolve the issue with following command:
```bash
pip uninstall triton pytorch-triton-xpu
# Reinstall correct version of pytorch-triton-xpu
pip install pytorch-triton-xpu==3.3.0 --index-url  https://download.pytorch.org/whl/xpu
```

4. `ValueError: Unsupported backend: CUDA_HOME ROCM_HOME MUSA_HOME are not set and XPU is not available.`

Ensure you have permissions to access /dev/dri/renderD*. This typically requires your user to be in the render group:
```bash
sudo gpasswd -a ${USER} render
newgrp render
```

## Additional Information
To run KTransformers on XPU with Docker, please refer to [Docker_xpu.md](./Docker_xpu.md).
