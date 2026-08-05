# Qwen3-Next Support for KTransformers

## Introduction

### Overview
We are very pleased to announce that Ktransformers now supports Qwen3-Next-80B-A3B-Thinking and Qwen3-Next-80B-A3B-Instruct.

### Model & Resource Links

- Official Qwen3-Next-80B-A3B-Thinking Release: 
  - https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking

- Official Qwen3-Next-80B-A3B-Instruct Release
  - https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct


## Installation Guide

### 1. Resource Requirements

The model running with 512 Experts requires approximately 320 GB of memory and 6 GB of GPU memory.

### 2. Prepare Models

```bash
# download gguf
huggingface-cli download --resume-download Qwen/Qwen3-Next-80B-A3B-Instruct

```

### 3. Install ktransformers

Qwen3-Next uses the `balance_serve` backend in the launch command below, so build KTransformers with the balance-serve extension enabled:

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive
USE_BALANCE_SERVE=1 bash ./install.sh
```

If startup fails with `ModuleNotFoundError: No module named 'sched_ext'`, the `balance_serve` scheduler extension was not built in the active environment. Re-run the install command above in a clean environment, then start the server again.

### 4. Run Qwen3-Next Inference Server

```bash
python ktransformers/server/main.py \
  --port 10021 \
  --model_path path-to-Qwen3-Next-80B-A3B-Thinking \
  --gguf_path path-to-Qwen3-Next-80B-A3B-Thinking \
  --model_name Qwen3NextForCausalLM \
  --optimize_config_path <local_path>/ktransformers/optimize/optimize_rules/Qwen3Next-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --no-use_cuda_graph \
  --backend_type balance_serve
```

### 5. Access server

```
curl -X POST http://localhost:10021/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "Qwen3-Next-80B-A3B-Instruct",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```

### 6. Notes

Due to Qwen3-Next’s use of linear attention, CUDA Graph optimization is not yet support — but it’s coming soon! 🚀