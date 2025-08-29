# HunYuan Support for KTransformers

## Introduction

### Overview
We are excited to announce that **KTransformers now supports HunYuan models with AMX optimization**.

- **HunYuan-Standard (AMX bf16)**: ~12 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~441 GB DRAM. Enhanced performance with Intel AMX acceleration for MoE expert computations.

### Model & Resource Links
- *[Hunyuan-A13B-Instruct](https://huggingface.co/tencent/Hunyuan-A13B-Instruct)*

---

## Installation Guide

### 1. Resource Requirements

| Model                     | Precision  | Experts | DRAM Needed | GPU Memory Needed\* | TPS (approx.)                   |
| ------------------------- | ---------- | ------- | ----------- | ------------------- | --------------------------------------- |
| HunYuan-Standard          | bf16       | 64      | \~441 GB    | 14 GB               | \~12 TPS                    |

\* Exact GPU memory depends on sequence length, batch size, and kernels used.  

### 2. Prepare Models

```bash
# Example: download original safetensors (adjust to your paths/repos)
# (Fill in actual repos/filenames yourself)

# HunYuan-Standard
huggingface-cli download --resume-download https://huggingface.co/tencent/Hunyuan-A13B-Instruct \
  --local-dir ./Hunyuan-A13B-Instruct
```

### 3. Install KTransformers

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

```bash
pip install ktransformers  # or from source if you need bleeding-edge features
```

### 4. Run HunYuan Inference Server

```bash
python ktransformers/server/main.py \
  --port 10002 \
  --model_path /abs/path/to/Hunyuan-A13B-Instruct \
  --model_name Hunyuan-A13B-Instruct \
  --gguf_path /abs/path/to/Hunyuan model files (.gguf or .safetensor) \
  --optimize_config_path ktransformers/optimize/optimize_rules/Hunyuan-serve-amx.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

### 5. Access Server

```bash
curl http://127.0.0.1:10002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Hunyuan-A13B-Instruct",
        "messages": [
          {"role": "user", "content": "介绍一下西伯利亚森林猫"}
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": false
      }'
```


---