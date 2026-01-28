# CPU-GPU Expert Scheduling Tutorial

This tutorial demonstrates how to use the CPU-GPU expert scheduling feature in KTransformers with SGLang. This feature introduces a flexible GPU expert mask system that allows intelligent placement of MoE experts across CPU and GPU, optimizing inference performance based on workload patterns.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch Server with Expert Scheduling](#step-2-launch-server-with-expert-scheduling)
  - [Basic Usage](#basic-usage)
  - [Expert Placement Strategies](#expert-placement-strategies)
  - [Key Parameters](#key-parameters)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)
  - [Option A: Interactive Chat with KT CLI](#option-a-interactive-chat-with-kt-cli)
  - [Option B: OpenAI-Compatible API](#option-b-openai-compatible-api)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA RTX 4090 24 GB (or equivalent with at least 24GB VRAM available)
- **CPU**: x86 CPU with AVX512 support (e.g., Intel Sapphire Rapids, AMD EPYC)
- **RAM**: At least 256GB system memory
- **Storage**: Sufficient space for model weights

**Tested Configuration:**

- **GPU**: 4 x NVIDIA GeForce RTX 4090 (24 GB)
- **CPU**: Intel Xeon Gold 6454S
- **RAM**: 512GB DDR5
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Prerequisites

Before starting, ensure you have:

1. **SGLang installed**

    Note: Currently, please clone our custom SGLang repository:

    ```bash
    git clone https://github.com/kvcache-ai/sglang.git
    cd sglang
    pip install -e "python[all]"
    ```

2. **KTransformers installed**

    ```bash
    git clone https://github.com/kvcache-ai/ktransformers.git
    cd ktransformers/kt-kernel
    bash ./install.sh
    ```

    After installation, verify the CLI is working:

    ```bash
    kt version
    ```

3. **CUDA toolkit** - CUDA 12.0+ recommended
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download your preferred MoE model weights. This feature supports various MoE models including:

* **Qwen3-Next-80B-A3B-Instruct-FP8**

    ```bash
    huggingface-cli download Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --local-dir /path/to/qwen3-next-80b
    ```

## Step 2: Launch Server with Expert Scheduling

### Basic Usage

The simplest way to start the server with expert scheduling:

```bash
python -m sglang.launch_server \
    --model /path/to/model \
    --kt-num-gpu-experts 8 \
    --kt-expert-placement-strategy uniform
```

### Expert Placement Strategies

The system provides four expert placement strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `uniform` | Distributes GPU experts evenly across all MoE layers | Default, no prior statistics needed |
| `frequency` | Places most frequently activated experts on GPU | Best performance when activation statistics are available |
| `front-loading` | Fills GPU experts from the first layer onwards | Testing or specific workload patterns |
| `random` | Randomly selects experts with fixed seed (42) | Baseline comparison |

**Using Frequency Strategy (Recommended for best performance):**

```bash
python -m sglang.launch_server \
    --model /path/to/model \
    --kt-num-gpu-experts 8 \
    --kt-expert-placement-strategy frequency \
    --init-expert-location /path/to/activation_stats.pt
```

**Using Dynamic Expert Update:**

```bash
python -m sglang.launch_server \
    --model /path/to/model \
    --kt-num-gpu-experts 8 \
    --kt-expert-placement-strategy frequency \
    --init-expert-location /path/to/activation_stats.pt \
    --kt-enable-dynamic-expert-update \
    --kt-gpu-prefill-token-threshold 512
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--kt-num-gpu-experts` | Number of GPU experts per MoE layer. Internally multiplied by the number of MoE layers to get the total GPU experts. Ignored if `--kt-gpu-experts-ratio` is set. |
| `--kt-gpu-experts-ratio` | Ratio of total experts to place on GPU (0.0-1.0). If set, overrides `--kt-num-gpu-experts`. Example: 0.1 means 10% of all experts across all layers will be on GPU. |
| `--kt-expert-placement-strategy` | Expert placement strategy: `frequency`, `uniform`, `front-loading`, or `random`. Default: `uniform`. |
| `--init-expert-location` | Path to activation statistics file (`.pt`) for `frequency` strategy. |
| `--kt-enable-dynamic-expert-update` | Enable dynamic expert update during inference. |
| `--kt-gpu-prefill-token-threshold` | Token threshold for triggering dynamic expert redistribution during prefill. |
| `--record-kt-gpu-expert-distribution` | Enable recording of GPU expert distribution for analysis. |
| `--expert-distribution-recorder-mode` | Recording mode: `stat` (default), `stat_approx`, `per_pass`, or `per_token`. |

## Step 3: Send Inference Requests

Once the server is running (default: `http://localhost:30000`), you can interact with the model in several ways:

### Option A: Interactive Chat with KT CLI

The easiest way to chat with the model:

```bash
kt chat
```

This opens an interactive terminal chat session. Type your messages and press Enter to send. Use `Ctrl+C` to exit.

### Option B: OpenAI-Compatible API

The server exposes an OpenAI-compatible API at `http://localhost:30000/v1`.

**curl example (streaming):**

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Performance

### Throughput (tokens/s)

The following benchmarks were measured on Qwen3-Next-80B-A3B-Instruct-FP8 with 4 x RTX 4090, Intel Xeon Gold 6454S, tensor parallel size 4, using ShareGPT dataset:

| GPU Expert Ratio | random | uniform | front-loading | frequency | dynamic-expert-update |
|------------------|--------|---------|---------------|-----------|----------------------|
| 0% | 53.01 | 52.96 | 54.18 | 52.72 | 53.37 |
| 10% | 56.63 | 56.57 | 57.18 | 58.60 | 70.22 |
| 20% | 58.75 | 60.28 | 58.82 | 61.92 | 74.73 |
| 30% | 62.86 | 62.08 | 63.87 | 66.50 | 75.55 |
| 40% | 66.81 | 66.82 | 67.45 | 72.78 | 80.98 |
| 50% | 70.38 | 65.25 | 73.65 | 76.19 | 81.17 |
| 60% | 71.33 | 72.80 | 77.95 | 82.33 | 82.30 |
| 70% | 74.40 | 76.17 | 81.59 | 89.37 | 88.70 |
| 80% | 79.71 | 79.20 | 89.20 | 100.67 | 92.31 |
| 90% | 88.82 | 81.06 | 98.14 | 107.15 | 95.04 |
| 100% | 112.61 | 112.32 | 111.82 | 114.26 | 112.99 |

The `frequency` and `dynamic-expert-update` strategies show significant performance improvements over baseline strategies, especially at lower GPU expert ratios.

## Troubleshooting

### OOM (Out of Memory) Issues

If you encounter OOM, adjust these parameters when launching the server:

| Parameter | VRAM Impact |
|-----------|-------------|
| `--kt-num-gpu-experts` / `--kt-gpu-experts-ratio` | Reduces expert weight VRAM usage |
| `--chunked-prefill-size` | Reduces prefill extra VRAM allocation |
| `--max-total-tokens` | Reduces KV cache VRAM usage |

### Dynamic Expert Update Not Triggering

Ensure all conditions are met:
1. `--kt-enable-dynamic-expert-update` is enabled
2. `--kt-gpu-prefill-token-threshold` is set
3. Prefill length >= threshold value

### Statistics Recording

To save expert distribution statistics to a custom path, set the environment variable:

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/output
```

## Additional Resources

- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KTransformers GitHub](https://github.com/kvcache-ai/ktransformers)
