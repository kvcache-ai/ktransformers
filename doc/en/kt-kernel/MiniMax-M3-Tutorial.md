# Running MiniMax-M3 with SGLang and KT-Kernel

This tutorial demonstrates how to run MiniMax-M3 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of M3's 128-routed-expert sparse architecture by offloading experts to CPU.

The examples use the public model ID `MiniMaxAI/MiniMax-M3-MXFP8`.

The launch commands below target an 8-GPU TP8 server with 64 CPU inference threads and 2 CPU thread pools. Adjust `--tp-size`, `--kt-cpuinfer`, `--kt-threadpool-count`, and GPU expert counts for your hardware.

## Table of Contents

- [Running MiniMax-M3 with SGLang and KT-Kernel](#running-minimax-m3-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
    - [Hybrid (recommended): 8x H20/H100](#hybrid-recommended-8x-h20h100)
    - [Hybrid: Single GPU](#hybrid-single-gpu)
  - [Step 3: Send Inference Requests](#step-3-send-inference-requests)
    - [Option A: OpenAI-Compatible API](#option-a-openai-compatible-api)
    - [Option B: Tool Calling](#option-b-tool-calling)
  - [Thinking Mode](#thinking-mode)
  - [Recommended Parameters](#recommended-parameters)
  - [Troubleshooting](#troubleshooting)
  - [Additional Resources](#additional-resources)

## Prerequisites

Before starting, ensure you have:


1. **KT-Kernel installed** (required for hybrid CPU offload)

   ```bash
   git clone https://github.com/kvcache-ai/ktransformers.git
   cd ktransformers
   git submodule update --init --recursive
   cd kt-kernel && ./install.sh
   ```

2. **SGLang installed** — install the kvcache-ai fork of SGLang (one of):

   ```bash
   # Option A: One-click install (from ktransformers root)
   ./install.sh

   # Option B: pip install
   pip install sglang-kt
   ```

3. **Supported GPUs:** **SM90 (Hopper: H100 / H200 / H20 / H800)**. Upstream sglang targets **SM100 (Blackwell datacenter: B100 / B200 / GB200)** so far.

4. **CUDA toolkit** — CUDA 12.0+ recommended; CUDA 12.8+ for FP8 / MXFP8 deployments.

5. **Hugging Face CLI** — for downloading models:

   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download the MiniMax-M3 weights from Hugging Face. M3 ships natively in MXFP8 (`fp8 e4m3 + uint8 ue8m0 1x32` scale).

```bash
hf download MiniMaxAI/MiniMax-M3-MXFP8 \
  --local-dir /path/to/MiniMax-M3-MXFP8
```

**Note:** Replace `/path/to/` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server


### Hybrid (recommended): 8x H20

```bash
python -m sglang.launch_server \
  --model-path /path/to/MiniMax-M3-MXFP8 \
  --kt-weight-path /path/to/MiniMax-M3-MXFP8 \
  --kt-method MXFP8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 40 \
  --kt-gpu-prefill-token-threshold 500 \
  --tp-size 8 \
  --quantization mxfp8 \
  --moe-runner-backend triton \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.55 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 1 \
  --tool-call-parser minimax-m3 \
  --reasoning-parser minimax-m3 \
  --served-model-name MiniMax-M3
```

### Hybrid: Single GPU H20

A single 96 GB Hopper card is sufficient if most routed experts stay on CPU:

```bash
python -m sglang.launch_server \
  --model-path /path/to/MiniMax-M3-MXFP8 \
  --kt-weight-path /path/to/MiniMax-M3-MXFP8 \
  --kt-method MXFP8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 4 \
  --kt-gpu-prefill-token-threshold 500 \
  --tp-size 1 \
  --quantization mxfp8 \
  --moe-runner-backend triton \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 4096 \
  --cuda-graph-max-bs 1 \
  --tool-call-parser minimax-m3 \
  --reasoning-parser minimax-m3 \
  --served-model-name MiniMax-M3
```


If you encounter OOM, lower `--kt-num-gpu-experts`, `--mem-fraction-static`, `--chunked-prefill-size`, or `--max-running-requests` (default high).

## Step 3: Send Inference Requests

Once the server is running at `http://localhost:8000`, you can interact with the model in several ways.

### Option A: OpenAI-Compatible API

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

**curl example (non-streaming):**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M3",
    "messages": [{"role": "user", "content": "Solve step by step: 17 * 23"}],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```


### Option B: Tool Calling

M3 emits tool calls in its native `<minimax:tool_call>` XML format. The `--tool-call-parser minimax-m3` flag converts them to the OpenAI `tool_calls` array automatically.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "user", "content": "What's the weather in Shanghai?"}],
    tools=tools,
    tool_choice="auto",
    max_tokens=200,
)
print(response.choices[0].message.tool_calls)
```

## Thinking Mode

M3 supports request-level thinking control via `chat_template_kwargs.thinking_mode`. Reasoning output (if any) is returned under `message.reasoning_content`.

| `thinking_mode` | Behavior |
|---|---|
| `"enabled"` | Force chain-of-thought; the `<mm:think>` start tag is prefilled by the template. |
| `"disabled"` | Suppress thinking; the closing tag is prefilled. |
| `"adaptive"` (default) | Model self-decides; detector handles emitted `<mm:think>` blocks. |

Example:

```python
response = client.chat.completions.create(
    model="MiniMax-M3",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"chat_template_kwargs": {"thinking_mode": "disabled"}},
    max_tokens=50,
)
# content: "4", reasoning_content: None
```

## Recommended Parameters

**Default generation settings:**

- temperature: 0.6 (chat) / 0.0 (greedy benchmark)
- top-p: 0.95
- max-tokens: task-dependent

**KT-Kernel hybrid sizing rule-of-thumb:**

| TP size | Suggested `--kt-num-gpu-experts` | Suggested `--kt-cpuinfer` | Notes |
|---|---|---|---|
| 1 | 2–8 | physical core count | Single 96 GB card, most experts on CPU |
| 4 | 20–40 | physical core count | Mid-range deploy |
| 8 | 40–60 | physical core count | Highest throughput; raise `--mem-fraction-static` |

`--kt-gpu-prefill-token-threshold 500` enables layerwise full-GPU prefill fallback for prompts longer than 500 tokens; set to a larger value to disable for short workloads.

## Troubleshooting

**Model implementation errors**

Make sure your SGLang build is on `feat/minimax-m3` and that `--trust-remote-code` is enabled.

**OOM during startup or serving**

Reduce one or more of:

- `--kt-num-gpu-experts`
- `--mem-fraction-static`
- `--chunked-prefill-size`
- `--max-running-requests`




## Additional Resources

- [MiniMax-M3 Model Card](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8)
- [KT-Kernel Documentation](https://github.com/kvcache-ai/ktransformers/tree/main/doc/en/kt-kernel)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
