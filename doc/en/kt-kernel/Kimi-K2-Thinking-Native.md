# Running Kimi-K2-Thinking with SGLang and KT-Kernel

This tutorial demonstrates how to run Kimi-K2 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)

## Hardware Requirements

**Minimum Configuration:**
- **GPU**: NVIDIA RTX 4090 48GB (or equivalent with at least 48GB VRAM available)
- **CPU**: x86 CPU with AVX512 support (e.g., Sapphire Rapids)
- **RAM**: At least 650GB system memory
- **Storage**: ~600GB for model weights (native INT4 weight, same weight dir for CPU and GPU)

**Tested Configuration:**

- **GPU**: 1/2/4/8x NVIDIA RTX 4090/L20 48GB
- **CPU**: 2x Intel(R) Xeon(R) Platinum 8488C
- **RAM**: 2TB DDR5 4800MHz
- **OS**: Linux (Ubuntu 20.04+ recommended)

## Prerequisites

Before starting, ensure you have:

1. **KT-Kernel installed** - Follow the [installation guide](./kt-kernel_intro.md#installation)
2. **SGLang installed** - Follow [SGLang integration steps](./kt-kernel_intro.md#integration-with-sglang)

Note: Currently, please clone our custom SGLang repository:

```
git clone https://github.com/kvcache-ai/sglang.git
cd sglang
git checkout kimi_k2
pip install -e "python[all]"
```

3. **CUDA toolkit** - Compatible with your GPU (CUDA 11.8+ recommended)
4. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install huggingface-hub
   ```

## Step 1: Download Model Weights

```bash
# Create a directory for models
mkdir -p /path/to/models
cd /path/to/models

# Download Kimi-K2-Thinking (INT4 for both CPU and GPU)
huggingface-cli download moonshotai/Kimi-K2-Thinking \
  --local-dir /path/to/kimi-k2-thinking
```

**Note:** Replace `/path/to/models` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.


### Launch Command (2x RTX 4090 Example)

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30001 \
  --model /path/to/kimi-k2-thinking \
  --kt-weight-path /path/to/kimi-k2-thinking \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 8 \
  --kt-method RAWINT4 \
  --kt-gpu-prefill-token-threshold 400 \
  --kt-max-deferred-experts-per-token 1 \
  --trust-remote-code \
  --mem-fraction-static 0.94 \
  --served-model-name Kimi-K2-Thinking \
  --enable-mixed-chunk \
  --tensor-parallel-size 2 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 65536 \
  --max-total-tokens 65536 \
  --attention-backend flashinfer
```

It takes about 2~3 minutes to start the server.

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--kt-method RAWINT4` | CPU and GPU use the same INT4 weight. Set `--model` and `--kt-weight-path` to the same directory. |
| `--kt-num-gpu-experts` | Number of experts kept on GPU for decoding. |
| `--kt-gpu-prefill-token-threshold` | Token count threshold for prefill strategy. Below: hybrid CPU+GPU. Above: layerwise GPU prefill. |
| `--chunked-prefill-size` | Maximum tokens per prefill batch. |
| `--max-total-tokens` | Maximum total tokens in KV cache. |

### About `--kt-gpu-prefill-token-threshold`

This parameter controls the prefill strategy:

- **$\leq$ threshold**: Uses hybrid CPU+GPU prefill. No extra VRAM needed, but performance degrades slowly as token count increases.
- **> threshold**: Uses layerwise GPU prefill. Performance scales near-exponentially until reaching the bottleneck, but requires 9GB+ extra VRAM.

### Troubleshooting OOM

Layerwise prefill requires extra VRAM (~9GB + incremental cost with prefill length). If you encounter OOM, adjust these parameters based on your use case and hardware (refer to the recommended parameters table below):

| Parameter | VRAM Impact |
|-----------|-------------|
| `--kt-num-gpu-experts` | Reduces expert weight VRAM usage |
| `--chunked-prefill-size` | Reduces prefill extra VRAM allocation |
| `--max-total-tokens` | Reduces KV cache VRAM usage |

**Tip:** Test with an input of length `chunked-prefill-size` to verify your configuration won't OOM during prefill.


### Recommended Parameters

| GPU Config | `kt-num-gpu-experts` | `max-total-tokens` | `chunked-prefill-size` |
|------------|----------------------|---------------------|------------------------|
| 1x RTX 4090 (48GB) | 0 | 30000 | 30000 |
| 2x RTX 4090 (48GB) | 8 | 65536 | 65536 |
| 4x RTX 4090 (48GB) | 30 | 80000 | 65536 |
| 8x RTX 4090 (48GB) | 80 | 100000 | 65536 |

**Tip:** If your prefill and total length requirements are low (e.g., processing short texts), you can reduce `max-total-tokens` and `chunked-prefill-size` to free up VRAM for a larger `kt-num-gpu-experts`, which improves decode performance.

### Performance

The following prefill throughput (tokens/s) benchmarks were measured with single concurrency:

| GPU Config | 2048 tokens | 8192 tokens | 32768 tokens |
|------------|-------------|-------------|--------------|
| 1x RTX 4090 (48GB) | 53 | 184 | 290* |
| 2x RTX 4090 (48GB) | 85 | 294 | 529 |
| 4x RTX 4090 (48GB) | 118 | 415 | 818 |
| 8x RTX 4090 (48GB) | 130 | 435 | 1055 |

* Note: 1x RTX 4090 with layerwise prefill OOMs at 32768 tokens, so the 290 tokens/s is measured with qlen=30000.

## Step 3: Send Inference Requests

Once the server is running, you can send inference requests using the OpenAI-compatible API.

### Basic Chat Completion Request

```bash
curl -s http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-K2-Thinking",
    "stream": false,
    "messages": [
      {"role": "user", "content": "hi"}
    ]
  }'
```

### Example Response

```json
{
    "id": "cd0905562bf44513947284f80cc5634b",
    "object": "chat.completion",
    "created": 1764921457,
    "model": "Kimi-K2-Thinking",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": " <think> The user says \"hi\". This is a very simple greeting. I should respond in a friendly and helpful manner. Since I'm an AI assistant, I should be professional but approachable.\n\nPossible responses:\n1. \"Hello! How can I help you today?\"\n2. \"Hi there! What can I do for you?\"\n3. \"Hello! It's nice to hear from you. What would you like to talk about?\"\n4. \"Hi! I'm here to assist you with any questions you might have.\"\n\nI think option 1 is the most standard and professional. It's direct, friendly, and opens the door for the user to ask their question. I should keep it concise.\n\nLet me go with: \"Hello! How can I help you today?\" </think> Hello! How can I help you today?",
                "reasoning_content": null,
                "tool_calls": null
            },
            "logprobs": null,
            "finish_reason": "stop",
            "matched_stop": 163586
        }
    ],
    "usage": {
        "prompt_tokens": 26,
        "total_tokens": 189,
        "completion_tokens": 163,
        "prompt_tokens_details": null,
        "reasoning_tokens": 0
    },
    "metadata": {
        "weight_version": "default"
    }
}
```

## Advance Use Case: Running Claude Code with Native Kimi-K2-Thinking Local Backend

Add the following parameters to the SGLang launch command above to enable tool calling support:

```bash
--tool-call-parser kimi_k2 --reasoning-parser kimi_k2
```

With these parameters enabled, you can use [claude-code-router](https://github.com/musistudio/claude-code-router) to connect Kimi-K2-Thinking as a local backend for [Claude Code](https://github.com/anthropics/claude-code).

## Additional Resources

- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Claude Code Router](https://github.com/musistudio/claude-code-router) - Route Claude Code to custom backends
