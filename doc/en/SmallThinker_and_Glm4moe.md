# GLM-4-MoE Support for KTransformers

## Introduction

### Overview
We are excited to announce that **KTransformers now supports GLM-4-MoE**.

- **GLM-4-MoE 110B (bf16)**: ~11 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~440 GB DRAM.  
- **GLM-4-MoE 110B (AMX INT8)**: prefill ~309 TPS / decode ~16 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~220 GB DRAM.

### Model & Resource Links
- **GLM-4-MoE 110B**
  - *(to be announced)*

## Installation Guide

### 1. Resource Requirements

| Model                     | Precision | Experts | DRAM Needed | GPU Memory Needed\* | TPS (approx.)                  |
| ------------------------- | --------- | ------- | ----------- | ------------------- | ------------------------------ |
| GLM-4-MoE 110B            | bf16      | 128     | \~440 GB    | 14 GB               | \~11 TPS                       |
| GLM-4-MoE 110B (AMX INT8) | int8      | 128     | \~220 GB    | 14 GB               | prefill \~309 TPS / decode \~16 TPS |

\* Exact GPU memory depends on sequence length, batch size, and kernels used.

### 2. Prepare Models

```bash
# Example: download original safetensors (adjust to your paths/repos)
# (Fill in actual repos/filenames yourself)

# GLM-4-MoE 110B
huggingface-cli download --resume-download placeholder-org/Model-TBA \
  --local-dir ./Model-TBA
````

### 3. Install KTransformers

Follow the official Installation Guide.

```bash
pip install ktransformers  # or from source if you need bleeding-edge features
```

### 4. Run GLM-4-MoE 110B Inference Server

```bash
python ktransformers/server/main.py \
  --port 10110 \
  --model_name Glm4MoeForCausalLM \
  --model_path /abs/path/to/GLM-4-MoE-110B-bf16 \
  --optimize_config_path ktransformers/optimize/optimize_rules/Glm4Moe-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

### 5. Access Server

```bash
curl -X POST http://localhost:10110/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "GLM-4-MoE-110B",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```

