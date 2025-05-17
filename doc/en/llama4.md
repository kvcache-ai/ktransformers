# 🦙 Tutorial: LLaMA 4 Multi-Concurrency Support with KTransformers (Balance Serve Backend)

## 📌 Overview

We are pleased to announce that **KTransformers** now provides **experimental support for LLaMA 4 models** through the powerful `balance_serve` backend introduced in **v0.2.4**. This update is available under the dedicated development branch: [`support-llama4`](https://github.com/kvcache-ai/ktransformers/tree/support-llama4), specifically targeting the newly released **Meta LLaMA 4** model architecture.

⚠️ This support is currently **not available on the main branch** due to dependencies on newer versions of `transformers`, and **compatibility limitations with inference of currently supported models**. Work is underway to integrate this into the mainline once broader stability and compatibility are validated.

💡 **If you already have an environment based on the main branch**, it is **strongly recommended to create a new environment** to avoid potential dependency conflicts.

------

## 🔗 Model & Resource Links

- 🔥 Official LLaMA 4 Release: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
   (Note: LLaMA 4 models are served through the Meta repository. Make sure to **agree to terms** before downloading.)
- 🧠 GGUF Format (quantized models):
  - https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF

------

## 🧪 Demo

https://github.com/user-attachments/assets/449706f1-784b-4931-b2ba-07687c1aca54

------

## Resource Requirements

The Scout model running with 16 Experts requires approximately 65 GB of memory and 10 GB of GPU memory, while the Maverick model with 128 Experts requires approximately 270 GB of memory and 12 GB of GPU memory.

------

## ⚙️ Usage Instructions

### 1. Clone `support-llama4` Branch

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git checkout support-llama4
git submodule update --init --recursive
```

### 2. Set Up Environment

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

sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libfmt-dev libgflags-dev zlib1g-dev patchelf
pip3 install packaging ninja cpufeature numpy openai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Build with Balance Serve Support

```bash
# Install single NUMA dependencies
USE_BALANCE_SERVE=1  bash ./install.sh
# For those who have two cpu and 1T RAM（Dual NUMA）:
USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh
```

### 4. Use our custom config.json

Currently, you need to copy the content of our custom config file into the `config.json` under your `--model_path`.  
- Use [scout_config.json](https://github.com/kvcache-ai/ktransformers/blob/support-llama4/doc/en/scout_config.json) for the Llama-4-Scout-17B-16E model  
- Use [maverick_config.json](https://github.com/kvcache-ai/ktransformers/blob/support-llama4/doc/en/maverick_config.json) for the Llama-4-Maverick-17B-128E model  

Please make sure to replace the content of `config.json` with the appropriate one accordingly.

### 5. Run LLaMA 4 Inference Server

Make sure you have:

- `--model_path` pointing to a local config directory (not a Hugging Face name).
- `--gguf_path` pointing to the folder containing quantized `.gguf` weights.

```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/Llama4-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve \
```

### 5. Access server

```
curl -X POST http://localhost:10002/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "Llama4",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```

------

## 📌 Limitations

- ✅ **Only `balance_serve` backend is supported** for LLaMA 4 models in this version.
- ⚠️ Requires **`transformers==4.51.0`** or newer. Due to potential compatibility issues with older toolchains, we have **not merged this branch to main yet**.
- ❌ Multimodal models are not supported yet in this version. Support will be added in future releases.
