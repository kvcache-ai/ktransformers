# Running DeepSeek V3.2 with SGLang and KT-Kernel

This tutorial demonstrates how to run DeepSeek V3.2 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Quantize CPU Weights](#step-2-quantize-cpu-weights)
- [Step 3: Launch SGLang Server](#step-3-launch-sglang-server)
- [Step 4: Send Inference Requests](#step-4-send-inference-requests)

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA L20 48GB (or equivalent with at least 27GB VRAM available)
- **CPU**: Intel Xeon with AMX support (e.g., Sapphire Rapids)
- **RAM**: At least 350GB system memory for INT4 quantization
- **Storage**: ~1TB for model weights (FP8 + INT4 quantized)

**Tested Configuration:**
- **GPU**: NVIDIA L20 48GB
- **CPU**: Intel(R) Xeon(R) Platinum 8488C
- **RAM**: 2TB DDR5
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Prerequisites

Before starting, ensure you have:

1. **KT-Kernel installed** - Follow the [installation guide](./kt-kernel_intro.md#installation)
2. **SGLang installed** - Follow [SGLang integration steps](./kt-kernel_intro.md#integration-with-sglang)
3. **CUDA toolkit** - Compatible with your GPU (CUDA 11.8+ recommended)
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install huggingface-hub
   ```

## Step 1: Download Model Weights

DeepSeek V3.2 requires downloading model repositories:

1. **DeepSeek-V3.2**
2. **DeepSeek-V3.2-Speciale**

```bash
# Create a directory for models
mkdir -p /path/to/models
cd /path/to/models

# Download DeepSeek-V3.2 (FP8 weights for GPU)
huggingface-cli download deepseek-ai/DeepSeek-V3.2 \
  --local-dir /path/to/deepseek-v3.2

# Download DeepSeek-V3.2-Speciale (if needed)
huggingface-cli download deepseek-ai/DeepSeek-V3.2-Speciale \
  --local-dir /path/to/deepseek-v3.2-speciale
```

**Note:** Replace `/path/to/models` with your actual storage path throughout this tutorial.

## Step 2: Quantize CPU Weights

Convert the FP8 GPU weights to INT4 quantized CPU weights using the provided conversion script.

### Conversion Command

For a 2-NUMA system with 60 physical cores:

```bash
cd /path/to/ktransformers/kt-kernel

python scripts/convert_cpu_weights.py \
  --input-path /path/to/deepseek-v3.2 \
  --input-type fp8 \
  --output /path/to/deepseek-v3.2-INT4 \
  --quant-method int4 \
  --cpuinfer-threads 60 \
  --threadpool-count 2 \
  --no-merge-safetensor
```

## Step 3: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.

### Launch Command

For single NVIDIA L20 48GB + 2-NUMA CPU system:

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/deepseek-v3.2 \
  --kt-weight-path /path/to/deepseek-v3.2-INT4 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 1 \
  --attention-backend triton \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 32 \
  --max-total-tokens 40000 \
  --served-model-name DeepSeek-V3.2 \
  --enable-mixed-chunk \
  --tensor-parallel-size 1 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --kt-method AMXINT4
```

### Resource Usage

- **GPU VRAM:** ~27GB (for 1 GPU expert per layer + attention)
- **System RAM:** ~350GB (for INT4 quantized CPU experts)

## Step 4: Send Inference Requests

Once the server is running, you can send inference requests using the OpenAI-compatible API.

### Basic Chat Completion Request

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hi"}
    ]
  }'
```

### Example Response

```json
{
  "id": "adbb44f6aafb4b58b167e42fbbb1eed3",
  "object": "chat.completion",
  "created": 1764675126,
  "model": "DeepSeek-V3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hi there! 👋 \n\nThanks for stopping by! How can I help you today? Feel free to ask me anything - I'm here to assist with questions, explanations, conversations, or whatever you need! 😊\n\nIs there something specific on your mind, or would you like to know more about what I can do?",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 1
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 72,
    "completion_tokens": 67,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}
```

## Additional Resources

- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [DeepSeek V3.2 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
- [SGLang GitHub](https://github.com/sgl-project/sglang)