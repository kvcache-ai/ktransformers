# Running Kimi-K2.5 with SGLang and KT-Kernel

This tutorial demonstrates how to run Kimi-K2.5 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA RTX 2x4090 48GB (or equivalent with at least total 48GB VRAM available)
- **CPU**: x86 CPU with AVX512F support (e.g., Intel Sapphire Rapids)
- **RAM**: At least 600GB system memory
- **Storage**: ~600GB for model weights (native INT4 weight, same weight folder for CPU and GPU)

## Prerequisites

Before starting, ensure you have:

1. **KT-Kernel installed**:

   Note: Latest KTransformers' EPLB feature for Kimi-K2.5 will be supported soon.

```
git clone https://github.com/kvcache-ai/ktransformers.git
git submodule update --init --recursive
cd kt-kernel && ./install.sh
```

2. **SGLang installed** - Follow [SGLang integration steps](./kt-kernel_intro.md#integration-with-sglang)

Note: Currently, please clone our custom SGLang repository:

```
git clone https://github.com/kvcache-ai/sglang.git
cd sglang && pip install -e "python[all]"
// maybe need to reinstall cudnn according to the issue when launching SGLang
// pip install nvidia-cudnn-cu12==9.16.0.29
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

# Download Kimi-K2.5 (RAW-INT4 for both CPU and GPU)
huggingface-cli download moonshotai/Kimi-K2.5 \
  --local-dir /path/to/kimi-k2.5
```

**Note:** Replace `/path/to/models` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.


### Launch Command (4x RTX 4090 Example)

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 31245 \
  --model /path/to/kimi-k2.5 \
  --kt-weight-path /path/to/kimi-k2.5 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method RAWINT4 \
  --kt-gpu-prefill-token-threshold 400 \
  --trust-remote-code \
  --mem-fraction-static 0.94 \
  --served-model-name Kimi-K2.5 \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 32658 \
  --max-total-tokens 50000 \
  --attention-backend flashinfer
```

It takes about 2~3 minutes to start the server.

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

## Step 3: Send Inference Requests

Once the server is running, you can send inference requests using the OpenAI-compatible API.

### Basic Chat Completion Request

```bash
curl -s http://localhost:31245/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K2.5",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hi, who are you?"}
    ]
  }'
```

### Example Response

```json
{
    "id": "2a4e83f8a79b4b57b103b0f298fbaa7d",
    "object": "chat.completion",
    "created": 1769333912,
    "model": "Kimi-K2.5",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": " The user is asking \"hi, who are you?\" which is a simple greeting and identity question. I need to respond appropriately by introducing myself clearly and concisely.\n\nI am Kimi, a large language model trained by Moonshot AI. I should state my name, my nature (AI assistant), and my developer (Moonshot AI). I should keep it friendly and helpful.\n\nKey points to include:\n- Greet them back (\"hi\" or \"hello\")\n- State my name: Kimi\n- State what I am: an AI assistant/language model\n- Mention my developer: Moonshot AI\n- Briefly describe my purpose: to help answer questions, provide information, and assist with various tasks\n- Keep it concise but informative\n- Use a friendly, professional tone\n\nI should avoid overly technical jargon while being accurate. The response should be welcoming and set the stage for further interaction.\n\nPossible response:\n\"Hi! I'm Kimi, an AI assistant created by Moonshot AI. I'm designed to help answer questions, provide information, and assist with a wide range of tasks. How can I help you today?\"\n\nThis covers all the necessary points and invites the user to continue the conversation. </think> Hi! I'm Kimi, an AI assistant created by Moonshot AI. I'm designed to help answer questions, provide information, and assist with a wide range of tasks. How can I help you today?",
                "reasoning_content": null,
                "tool_calls": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "matched_stop": 163586
        }
    ],
    "usage": {
        "prompt_tokens": 32,
        "total_tokens": 317,
        "completion_tokens": 285,
        "prompt_tokens_details": null,
        "reasoning_tokens": 0
    },
    "metadata": {
        "weight_version": "default"
    }
}
```
