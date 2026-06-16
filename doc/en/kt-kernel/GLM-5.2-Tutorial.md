# Running GLM-5.2 with SGLang and KT-Kernel

This tutorial demonstrates how to run GLM-5.2 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU while keeping selected experts on GPU.

The examples use the public model IDs `zai-org/GLM-5.2` and `zai-org/GLM-5.2-FP8`. If you are using an internal mirror, replace only the local paths and keep internal model suffixes out of public documentation.

The launch commands below target an 8-GPU TP8 server with 96 CPU inference threads and 2 CPU thread pools. Adjust `--tp-size`, `--kt-cpuinfer`, `--kt-threadpool-count`, and GPU expert counts for your hardware.

## Table of Contents

- [Running GLM-5.2 with SGLang and KT-Kernel](#running-glm-52-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
    - [Key Parameters](#key-parameters)
  - [Step 3: Send Inference Requests](#step-3-send-inference-requests)
    - [Option A: Interactive Chat with KT CLI](#option-a-interactive-chat-with-kt-cli)
    - [Option B: OpenAI-Compatible API](#option-b-openai-compatible-api)
  - [Reasoning Mode](#reasoning-mode)
  - [Recommended Parameters](#recommended-parameters)
  - [Troubleshooting](#troubleshooting)
  - [Additional Resources](#additional-resources)

## Prerequisites

Before starting, ensure you have:

1. **SGLang with GLM-5.2 and KT integration**

   Install the kvcache-ai SGLang package, or run the one-click installer from the ktransformers root:

   ```bash
   # Option A: One-click install (from ktransformers root)
   ./install.sh

   # Option B: pip install
   pip install sglang-kt
   ```

2. **KT-Kernel installed**

   ```bash
   git clone https://github.com/kvcache-ai/ktransformers.git
   git submodule update --init --recursive
   cd kt-kernel && ./install.sh
   ```

   After installation, verify the CLI:

   ```bash
   kt version
   ```

3. **Transformers 5.3.0**

   GLM-5 family deployments require a recent Transformers version. Use the same version as the GLM-5.1 setup unless your SGLang build documents a different requirement:

   ```bash
   pip install transformers==5.3.0
   ```

   > **Note:** `transformers==5.3.0` may be incompatible with some older models. Use a separate virtual environment for GLM-5/5.1/5.2 if you also serve those models.

4. **CUDA toolkit** - CUDA 12.0+ recommended; CUDA 12.8+ is recommended for FP8 deployments.
5. **Hugging Face CLI** - For downloading models:

   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download the GLM-5.2 weights from Hugging Face.

```bash
# FP8
hf download zai-org/GLM-5.2-FP8 \
  --local-dir /path/to/GLM-5.2-FP8

# BF16
hf download zai-org/GLM-5.2 \
  --local-dir /path/to/GLM-5.2
```

**Note:** Replace `/path/to/` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.

The FP8 command below follows the validated GLM-5.2 KT launch shape. It uses FP8 model weights, FP8 KV cache, NSA attention, TP8, dynamic expert updates, and uniform expert placement.

```bash
# FP8 Precision
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --kt-weight-path /path/to/GLM-5.2-FP8 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --tp-size 8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.97 \
  --kv-cache-dtype fp8_e4m3 \
  --max-total-tokens 4096 \
  --max-running-requests 8 \
  --attention-backend nsa \
  --fp8-gemm-backend cutlass \
  --disable-shared-experts-fusion \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name GLM5.2
```

For BF16 weights, switch the model paths and KT method. The example keeps fewer experts on GPU as a conservative default; increase `--kt-num-gpu-experts` if you have VRAM headroom.

```bash
# BF16 Precision
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
  --model-path /path/to/GLM-5.2 \
  --kt-weight-path /path/to/GLM-5.2 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 10 \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --tp-size 8 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.97 \
  --max-total-tokens 4096 \
  --max-running-requests 8 \
  --attention-backend nsa \
  --disable-shared-experts-fusion \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --served-model-name GLM5.2
```

The example uses `--max-total-tokens 4096` as a conservative serving configuration. For longer context or benchmark runs, increase `--max-total-tokens` together with the KV cache and request concurrency settings your GPU memory can support.

If you encounter OOM, adjust `--kt-num-gpu-experts`, `--mem-fraction-static`, `--max-total-tokens`, and `--max-running-requests` when launching the server.

If you encounter other issues, try `kt doctor` to diagnose your setup.

## Step 3: Send Inference Requests

Once the server is running at `http://localhost:8000`, you can interact with the model in several ways.

### Option A: Interactive Chat with KT CLI

Use the KT CLI and point it to the server port used above:

```bash
kt chat --port 8000 --model GLM5.2
```

This opens an interactive terminal chat session. Type your messages and press Enter to send. Use `Ctrl+C` to exit.

### Option B: OpenAI-Compatible API

The server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

**curl example (streaming):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM5.2",
    "messages": [{"role": "user", "content": "hi, who are you?"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "stream": true
  }'
```

**curl example (non-streaming):**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM5.2",
    "messages": [{"role": "user", "content": "Solve this step by step: what is 17 * 23?"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "stream": false
  }'
```

## Reasoning Mode

Serve GLM-5.2 with `--reasoning-parser glm45`. For benchmark and scoring runs, keep the model-side `reasoning_effort` at the default `max`, or set it explicitly in the request if your client supports it:

```json
{
  "reasoning_effort": "max"
}
```

Do not set `reasoning_effort` to `high` for benchmark runs unless the benchmark or product requirement specifically calls for that setting.

## Recommended Parameters

**Default generation settings:**

- temperature: 1.0
- top-p: 0.95
- reasoning effort: max

**Benchmark context and output lengths from the GLM-5.2 adaptation note:**

| Workload | Context Length | Max New Tokens |
|----------|----------------|----------------|
| Short-text tasks | 262144 | 163840 |
| Reasoning tasks | 262144 | 163840 |
| Long-text tasks | 262144 | 65536 |

The launch command in this tutorial uses a smaller `--max-total-tokens` value for a conservative KT serving example. Increase server-side token limits before running the benchmark-scale settings above.

## Troubleshooting

**Model implementation errors**

Make sure your SGLang build includes GLM-5.2 model support and that `--trust-remote-code` is enabled.

**OOM during startup or serving**

Reduce one or more of:

- `--kt-num-gpu-experts`
- `--max-total-tokens`
- `--max-running-requests`
- `--mem-fraction-static`

For BF16 deployments, you can also try `--kv-cache-dtype fp8_e4m3` if the quality tradeoff is acceptable.

**Requests go to the wrong port**

This tutorial uses `--port 8000`. If you use `kt chat`, pass `--port 8000`; if you use curl or an OpenAI SDK client, use `http://localhost:8000/v1`.

## Additional Resources

- [GLM-5.2 Model Card](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.2-FP8 Model Card](https://huggingface.co/zai-org/GLM-5.2-FP8)
- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KT-Kernel Parameters Reference](../../../kt-kernel/README.md#kt-kernel-parameters)
