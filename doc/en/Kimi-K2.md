# Kimi-K2 Support for KTransformers

## Introduction

### Overview
We are very pleased to announce that Ktransformers now supports Kimi-K2.

On a single-socket CPU with one consumer-grade GPU, running the Q4_K_M model yields roughly 10 TPS and requires about 600 GB of DRAM.  
With a dual-socket CPU and sufficient system memory, enabling NUMA optimizations increases performance to about 14 TPS.

### Model & Resource Links

- Official Kimi-K2 Release: 
  - https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d
- GGUF Format(quantized models):
  - https://huggingface.co/KVCache-ai/Kimi-K2-Instruct-GGUF

## Installation Guide

### 1. Resource Requirements

The model running with 384 Experts requires approximately 2 TB of memory and 14 GB of GPU memory.

### 2. Prepare Models

You can convert the fp8 to bf16.

```bash
# download fp8
huggingface-cli download --resume-download xxx

# convert fp8 to bf16
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
cd inference
python fp8_cast_bf16.py --input-fp8-hf-path <path_to_fp8> --output-bf16-hf-path  <path_to_bf16>

```

### 3. Install ktransformers

To install KTransformers, follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

### 4. Run Kimi-K2 Inference Server

```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path <path_to_safetensor_config> \
  --gguf_path <path_to_bf16_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
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
    "model": "Kimi-K2",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```
