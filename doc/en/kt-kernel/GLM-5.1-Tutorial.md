# Running GLM-5.1 with SGLang and KT-Kernel

This tutorial demonstrates how to run GLM-5.1 model inference using SGLang integrated with Ktransformers for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU. KT-Kernel supports both BF16 and FP8 precision backends, allowing you to choose between maximum quality and reduced memory footprint.

GLM-5.1 introduces thinking mode (enabled by default), interleaved and preserved thinking, and MTP (Multi-Token Prediction) weights for both precisions.

## Table of Contents

- [Running GLM-5.1 with SGLang and KT-Kernel](#running-glm-51-with-sglang-and-kt-kernel)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download Model Weights](#step-1-download-model-weights)
  - [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
  - [Step 3: Send Inference Requests](#step-3-send-inference-requests)
    - [Option A: Interactive Chat with KT CLI](#option-a-interactive-chat-with-kt-cli)
    - [Option B: OpenAI-Compatible API](#option-b-openai-compatible-api)
  - [Thinking Mode](#thinking-mode)
  - [Recommended Parameters](#recommended-parameters)
  - [Additional Resources](#additional-resources)

## Prerequisites

Before starting, ensure you have:

1. **SGLang installed**

    Install the kvcache-ai fork of SGLang (one of):

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

3. **Transformers 5.3.0** — GLM-5 and GLM-5.1 require exactly `transformers==5.3.0` (the default pip install gives 4.x, which will not work):
    ```bash
    pip install transformers==5.3.0
    ```
    > **Note:** `transformers==5.3.0` is **not** compatible with some older models (e.g., DeepSeek). If you need to run those models, switch back to a 4.x release. Consider using a separate virtual environment for GLM-5/5.1 to avoid conflicts.

4. **CUDA toolkit** - CUDA 12.0+ recommended (12.8+ for best FP8 support)
5. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download the GLM-5.1 weights from Hugging Face. Both BF16 and FP8 models include MTP weights.

```bash
# FP8
hf download zai-org/GLM-5.1-FP8 \
  --local-dir /path/to/GLM-5.1-FP8

# BF16
hf download zai-org/GLM-5.1 \
  --local-dir /path/to/GLM-5.1
```

**Note:** Replace `/path/to/` with your actual storage path throughout this tutorial.

## Step 2: Launch SGLang Server

Start the SGLang server with KT-Kernel integration for CPU-GPU heterogeneous inference.

```bash
# FP8 Precision
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/GLM-5.1-FP8 \
  --kt-weight-path /path/to/GLM-5.1-FP8 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --trust-remote-code \
  --mem-fraction-static 0.75 \
  --served-model-name GLM5.1 \
  --enable-mixed-chunk \
  --tensor-parallel-size 8 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 16384 \
  --max-running-requests 4 \
  --max-total-tokens 128000 \
  --attention-backend flashinfer \
  --fp8-gemm-backend cutlass \
  --kv-cache-dtype bf16 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --watchdog-timeout 3000

# BF16 Precision
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_JIT_DEEPGEMM=0

python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 30000 \
  --model /path/to/GLM-5.1 \
  --kt-weight-path /path/to/GLM-5.1 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 10 \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --trust-remote-code \
  --mem-fraction-static 0.75 \
  --served-model-name GLM5.1 \
  --enable-mixed-chunk \
  --tensor-parallel-size 8 \
  --enable-p2p-check \
  --disable-shared-experts-fusion \
  --chunked-prefill-size 16384 \
  --max-running-requests 4 \
  --max-total-tokens 128000 \
  --attention-backend flashinfer \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --watchdog-timeout 3000
```

Layerwise prefill requires one extra MoE layer's worth of VRAM.

If you encounter OOM, adjust `--kt-num-gpu-experts`, `--chunked-prefill-size`, `--mem-fraction-static` and `--max-total-tokens` when launching the server.

If you encounter other issues, try `kt doctor` to diagnose your setup.

See [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters) for detailed parameter tuning guidelines.

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
    "model": "GLM5.1",
    "messages": [{"role": "user", "content": "hi, who are you?"}],
    "stream": true
  }'
```

## Thinking Mode

GLM-5.1 has **thinking mode enabled by default**. It supports two reasoning modes:

- **Interleaved Thinking** - Recommended for general conversation scenarios
- **Interleaved + Preserved Thinking** - Recommended for agentic workflows, especially code agents (e.g., Claude Code, Roo Code, Kilo Code)

To enable **interleaved + preserved thinking** with SGLang, pass the following parameters in your API request:

```json
"chat_template_kwargs": {
    "enable_thinking": true,
    "clear_thinking": false
}
```

To **disable** thinking mode:

```json
"chat_template_kwargs": {
    "enable_thinking": false
}
```

## Recommended Parameters

**Default settings (suitable for most tasks):**
- temperature: 1.0
- top-p: 0.95
- max new tokens: 131072

**Terminal Bench:**
- temperature: 0.7
- top-p: 1.0
- max new tokens: 16384
- context length: 202752

**Tau2-Bench:**
- temperature: 0
- max new tokens: 16384

For multi-turn agentic tasks (e.g., Tau2-Bench and Terminal Bench 2), enable **preserved thinking mode**.

## Additional Resources

- [GLM-5.1 Model Card](https://huggingface.co/zai-org/GLM-5.1)
- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KT-Kernel Parameters Reference](../../../kt-kernel/README.md#kt-kernel-parameters)
- [Thinking Mode Guide](https://docs.z.ai/guides/capabilities/thinking-mode)
