# SmallThinker & GLM-4-MoE Support for KTransformers

## Introduction

### Overview
We are excited to announce that **KTransformers now supports both SmallThinker and GLM-4-MoE**.

- **SmallThinker-21BA3B-Instruct (bf16)**: ~26 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~84 GB DRAM.  
- **GLM-4.5-Air (bf16)**: ~11 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~440 GB DRAM.
- **GLM-4.5-Air (AMX INT8)**: prefill ~309 TPS / decode ~16 TPS **on a dual-socket CPU with one consumer-grade GPU**, requiring ~220 GB DRAM.

### Model & Resource Links
- **SmallThinker-21BA3B-Instruct**
  - *[SmallThinker-21BA3B-Instruct](https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct)*
- **GLM-4.5-Air 110B**
  - [*GLM-4.5-Air*](https://huggingface.co/zai-org/GLM-4.5-Air)

---

## Installation Guide

### 1. Resource Requirements

| Model                     | Precision  | Experts | DRAM Needed | GPU Memory Needed\* | TPS (approx.)                   |
| ------------------------- | ---------- | ------- | ----------- | ------------------- | --------------------------------------- |
| SmallThinker-21B-Instruct          | bf16       | 32      | \~42 GB     | 14 GB               | \~26 TPS                    |
| GLM-4.5-Air            | bf16       | 128     | \~220 GB    | 14 GB               | \~11 TPS                    |
| GLM-4.5-Air (AMX INT8) | int8       | 128     | \~220 GB    | 14 GB               |  \~16 TPS


\* Exact GPU memory depends on sequence length, batch size, and kernels used.  

### 2. Prepare Models

```bash
# Example: download original safetensors (adjust to your paths/repos)
# (Fill in actual repos/filenames yourself)

# SmallThinker-21B
huggingface-cli download --resume-download https://huggingface.co/PowerInfer/SmallThinker-21BA3B-Instruct \
  --local-dir ./SmallThinker-21BA3B-Instruct

# GLM-4-MoE 110B
huggingface-cli download --resume-download https://huggingface.co/zai-org/GLM-4.5-Air \
  --local-dir ./GLM-4.5-Air
```


### 3. Install KTransformers

Follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/en/install.html).

```bash
pip install ktransformers  # or from source if you need bleeding-edge features
```

### 4. Run SmallThinker-21B Inference Server

```bash
python ktransformers/server/main.py \
  --port 10021 \
  --model_path /abs/path/to/SmallThinker-21B-bf16 \
  --model_name SmallThinkerForCausalLM \
  --optimize_config_path ktransformers/optimize/optimize_rules/SmallThinker-serve.yaml \
  --max_new_tokens 1024 \
  --cache_lens 32768 \
  --chunk_size 256 \
  --max_batch_size 4 \
  --backend_type balance_serve
```

### 5. Run GLM-4-MoE 110B Inference Server

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

### 6. Access Server

```bash
curl -X POST http://localhost:10021/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "SmallThinker-21BA3B-Instruct",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```

```bash
curl -X POST http://localhost:10110/v1/chat/completions \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "model": "GLM-4.5-Air",
    "temperature": 0.3,
    "top_p": 1.0,
    "stream": true
  }'
```
