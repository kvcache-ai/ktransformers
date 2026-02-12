# Running Qwen3.5 with SGLang and KT-Kernel

This tutorial demonstrates how to run Qwen3.5 (MoE-400B) model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU.

## Table of Contents

- [Running Qwen3.5 with SGLang and KT-Kernel](#running-qwen35-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
    - [Launch Command (4x RTX 4090 Example)](#launch-command-4x-rtx-4090-example)
  - [Step 3: Send Inference Requests](#step-3-send-inference-requests)
    - [Basic Chat Completion Request](#basic-chat-completion-request)
    - [Example Response](#example-response)

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA 4x RTX 4090 (or equivalent with at least 96GB total VRAM available)
- **CPU**: x86 CPU with AVX512F support (e.g., Intel Sapphire Rapids)
- **RAM**: At least 800GB system memory
- **Storage**: ~800GB for model weights (BF16)

## Prerequisites

Before starting, ensure you have:

1. **KT-Kernel installed**:

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
git checkout qwen3.5
git submodule update --init --recursive
cd kt-kernel && ./install.sh
```

2. **SGLang installed** - Follow [SGLang integration steps](./kt-kernel_intro.md#integration-with-sglang)

Note: Currently, please clone our custom SGLang repository:

```bash
git clone https://github.com/kvcache-ai/sglang.git
git checkout qwen3.5
cd sglang && pip install -e "python[all]"
# Maybe need to reinstall cudnn according to the issue when launching SGLang
pip install nvidia-cudnn-cu12==9.16.0.29
```

3. **CUDA toolkit** - Compatible with your GPU (CUDA 12.8+ recommended)
4. **Hugging Face CLI** - For downloading models:

   ```bash
   pip install huggingface-hub
   ```

## Step 1: Download Model Weights

```bash
# Create a directory for models
mkdir -p /path/to/models
cd /path/to/models

# Download Qwen3.5 (BF16)
huggingface-cli download Qwen/Qwen3.5 \
  --local-dir /path/to/qwen3.5
```

**Note:** Replace `/path/to/models` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.

### Launch Command (4x RTX 4090 Example)

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30005 \
  --model /path/to/qwen3.5 \
  --kt-weight-path /path/to/qwen3.5 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --kt-method BF16 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --max-total-tokens 32000 \
  --served-model-name qwen3.5 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --disable-custom-all-reduce
```

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

## Step 3: Send Inference Requests

Once the server is running, you can send inference requests using the OpenAI-compatible API.

### Basic Chat Completion Request

```bash
curl -s http://localhost:30005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hi, who are you?"}
    ]
  }'
```

### Example Response

```json
{
    "id": "c79f6d63e04f4874acb8853d218e1bf1",
    "object": "chat.completion",
    "created": 1770880035,
    "model": "qwen3.5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm **Qwen**, a large language model developed by **Alibaba Cloud**. I'm designed to provide helpful, accurate, and safe information across a wide range of topics—whether you have questions, need help with writing, coding, analysis, or just want to explore ideas together.\n\nHow can I assist *you* today?",
                "reasoning_content": null,
                "tool_calls": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "matched_stop": 248046
        }
    ],
    "usage": {
        "prompt_tokens": 16,
        "total_tokens": 527,
        "completion_tokens": 511,
        "prompt_tokens_details": null,
        "reasoning_tokens": 0
    },
    "metadata": {
        "weight_version": "default"
    }
}
```
