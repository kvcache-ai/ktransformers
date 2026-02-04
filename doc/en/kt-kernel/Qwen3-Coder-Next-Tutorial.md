# Running Qwen3-Coder-Next with SGLang and KT-Kernel

This tutorial demonstrates how to run Qwen3-Coder-Next (80B-A3B) model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. Qwen3-Coder-Next is a Mixture-of-Experts code generation model. KT-Kernel supports both BF16 and FP8 precision backends, allowing you to choose between maximum quality and reduced memory footprint.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
  - [Key Parameters](#key-parameters)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)
  - [Option A: Interactive Chat with KT CLI](#option-a-interactive-chat-with-kt-cli)
  - [Option B: OpenAI-Compatible API](#option-b-openai-compatible-api)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
  - [OOM (Out of Memory) Issues](#oom-out-of-memory-issues)
- [Additional Resources](#additional-resources)

## Hardware Requirements

**Recommended Configuration:**
- **GPU**: 1 x NVIDIA RTX 4090 24 GB
- **CPU**: x86 CPU with AVX512 support (e.g., Intel Sapphire Rapids, AMD EPYC)
- **RAM**: At least 100GB system memory for FP8 model weights
- **Storage**: >85 GB for FP8 model weights (80.4 GB)

## Prerequisites

Before starting, ensure you have:

1. **SGLang installed**

    Note: Currently, please clone our custom SGLang repository:

    ```bash
    git clone https://github.com/kvcache-ai/sglang.git
    cd sglang
    pip install -e "python[all]"
    ```

    You can follow [SGLang integration steps](https://docs.sglang.io/get_started/install.html)

2. **KT-Kernel installed**

    Please follow [kt-kernel](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

    After installation, verify the CLI is working:

    ```bash
    kt version
    ```

3. **CUDA toolkit** - CUDA 12.0+ recommended (12.8+ for best FP8 support)
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download the Qwen3-Coder-Next weights from Hugging Face.

```bash
# FP8
hf download Qwen/Qwen3-Coder-Next-FP8 \
  --local-dir /path/to/Qwen3-Coder-Next-FP8

# BF16
hf download Qwen/Qwen3-Coder-Next \
  --local-dir /path/to/Qwen3-Coder-Next
```

**Note:** Replace `/path/to/` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.

```bash
# FP8 Precision
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/Qwen3-Coder-Next-FP8 \
  --kt-weight-path /path/to/Qwen3-Coder-Next-FP8 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 100 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 2048 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 16384 \
  --max-running-requests 4 \
  --max-total-tokens 256000 \
  --served-model-name Qwen3-Coder-Next \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --fp8-gemm-backend cutlass \
  --tool-call-parser qwen3_coder \
  --kt-enable-dynamic-expert-update

# BF16 Precision
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/Qwen3-Coder-Next \
  --kt-weight-path /path/to/Qwen3-Coder-Next \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 60 \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 2048 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 16384 \
  --max-running-requests 4 \
  --max-total-tokens 256000 \
  --served-model-name Qwen3-Coder-Next \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --tool-call-parser qwen3_coder \
  --kt-enable-dynamic-expert-update
```

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--kt-method FP8 / BF16` | Inference precision mode. FP8 halves weight memory; BF16 uses full precision. |
| `--kt-cpuinfer` | Number of CPU inference threads. |
| `--kt-threadpool-count` | Number of thread pools. Set to NUMA node count. |
| `--kt-num-gpu-experts` | Number of experts kept on GPU for decoding. |
| `--kt-gpu-prefill-token-threshold` | Token threshold for layerwise prefill strategy. |
| `--kt-enable-dynamic-expert-update` | Enable dynamic expert placement on GPU based on routing statistics. |
| `--kt-expert-placement-strategy` | Expert placement strategy. Default: `uniform`. See [Expert Scheduling Tutorial](experts-sched-Tutorial.md) for other options. |
| `--chunked-prefill-size` | Maximum tokens per prefill batch. |
| `--max-total-tokens` | Maximum total tokens in KV cache. |
| `--tool-call-parser` | Tool call parser for function calling support (use `qwen3_coder`). |
| `--fp8-gemm-backend` | GEMM backend for FP8 computation. |

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
    "model": "Qwen3-Coder-Next",
    "messages": [{"role": "user", "content": "Write a Python function to compute the Fibonacci sequence."}],
    "stream": true
  }'
```

**curl example (non-streaming):**

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-Next",
    "messages": [{"role": "user", "content": "Hello! What can you help me with?"}],
    "stream": false
  }'
```

## Performance

The following benchmarks were measured with single concurrency (Prefill tps / Decode tps):

| GPU | CPU | PCIe | Precision | 64 tokens | 2048 tokens | 8192 tokens | 32768 tokens |
|-----|-----|------|-----------|-------------|-------------|-------------|--------------|
| 1 x RTX 5090 (32 GB) | 2 x AMD EPYC 9355 | PCIe 5.0 | FP8  | 362 / 75.9 | 1746 / 75.6 | 2407 / 69.1 | 6233 / 51.7 | 

## Troubleshooting

### OOM (Out of Memory) Issues

Layerwise prefill requires extra VRAM. If you encounter OOM, adjust these parameters when launching the server:

| Parameter | VRAM Impact |
|-----------|-------------|
| `--kt-num-gpu-experts` | Reduces expert weight VRAM usage |
| `--chunked-prefill-size` | Reduces prefill extra VRAM allocation |
| `--max-total-tokens` | Reduces KV cache VRAM usage |
| `--mem-fraction-static` | Lower values reserve more VRAM headroom (default: 0.80) |

**Tip:** Test with an input of length `chunked-prefill-size` to verify your configuration won't OOM during prefill.

## Additional Resources

- [Qwen3-Coder-Next Model Card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KT-Kernel Parameters Reference](../../../kt-kernel/README.md#kt-kernel-parameters)
