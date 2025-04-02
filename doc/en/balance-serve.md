# balance_serve backend (multi-concurrency) for ktransformers

## Installation Guide

### 1. Set Up Conda Environment
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

### 2. Install dependencies


```bash
sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libfmt-dev libgflags-dev zlib1g-dev patchelf
```

### 3. Build ktransformers

```bash
# Clone repository
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive

# Optional: Compile web interface
# See: api/server/website.md

# Install single NUMA dependencies
sudo env USE_BALANCE_SERVE=1 PYTHONPATH="$(which python)" PATH="$(dirname $(which python)):$PATH" bash ./install.sh
# Install Dual NUMA dependencies
sudo env USE_BALANCE_SERVE=1 USE_NUMA=1 PYTHONPATH="$(which python)" PATH="$(dirname $(which python)):$PATH" bash ./install.sh
```

## Running DeepSeek-R1-Q4KM Models

### Configuration for 24GB VRAM GPUs
Use our optimized configuration for constrained VRAM:

```bash
python ktransformers/server/main.py \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

It features the following arguments:

- `--max_new_tokens`: Maximum number of tokens generated per request.
- `--cache_lens`: Total length of kvcache allocated by the scheduler. All requests share a kvcache space. 
- `--chunk_size`: Maximum number of tokens processed in a single run by the engine.
corresponding to 32768 tokens, and the space occupied will be released after the requests are completed.
- `--max_batch_size`: Maximum number of requests (prefill + decode) processed in a single run by the engine. (Supported only by `balance_serve`)
- `--backend_type`: `balance_serve` is a multi-concurrency backend engine introduced in version v0.2.4. The original single-concurrency engine is `ktransformers`.
