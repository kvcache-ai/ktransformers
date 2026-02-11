# Running GLM-5 with SGLang and KT-Kernel

This tutorial demonstrates how to run GLM-5 model inference using SGLang integrated with KT-Kernel for CPU-GPU heterogeneous inference. This setup enables efficient deployment of large MoE models by offloading experts to CPU. KT-Kernel supports both BF16 and FP8 precision backends, allowing you to choose between maximum quality and reduced memory footprint.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Step 1: Download Model Weights](#step-1-download-model-weights)
- [Step 2: Launch SGLang Server](#step-2-launch-sglang-server)
- [Step 3: Send Inference Requests](#step-3-send-inference-requests)
  - [Option A: Interactive Chat with KT CLI](#option-a-interactive-chat-with-kt-cli)
  - [Option B: OpenAI-Compatible API](#option-b-openai-compatible-api)
- [Additional Resources](#additional-resources)

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

3. **transformers reinstalled**

    ```bash
    pip install git+https://github.com/huggingface/transformers.git
    ```

4. **CUDA toolkit** - CUDA 12.0+ recommended (12.8+ for best FP8 support)
5. **Hugging Face CLI** - For downloading models:
   ```bash
   pip install -U huggingface-hub
   ```

## Step 1: Download Model Weights

Download the GLM-5 weights from Hugging Face.

```bash
# FP8
hf download zai-org/GLM-5-FP8 \
  --local-dir /path/to/GLM-5-FP8

# BF16
hf download zai-org/GLM-5 \
  --local-dir /path/to/GLM-5
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
  --model /path/to/GLM-5-FP8 \
  --kt-weight-path /path/to/GLM-5-FP8 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 30 \
  --kt-method FP8 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --trust-remote-code \
  --mem-fraction-static 0.75 \
  --served-model-name GLM5 \
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
  --model /path/to/GLM-5 \
  --kt-weight-path /path/to/GLM-5 \
  --kt-cpuinfer 96 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 10 \
  --kt-method BF16 \
  --kt-gpu-prefill-token-threshold 1024 \
  --kt-enable-dynamic-expert-update \
  --kt-expert-placement-strategy uniform \
  --trust-remote-code \
  --mem-fraction-static 0.75 \
  --served-model-name GLM5 \
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
    "model": "GLM5",
    "messages": [{"role": "user", "content": "hi, who are you?"}],
    "stream": true
  }'
```

## Additional Resources

- [GLM-5 Model Card](https://huggingface.co/zai-org/GLM-5)
- [KT-Kernel Documentation](../../../kt-kernel/README.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [KT-Kernel Parameters Reference](../../../kt-kernel/README.md#kt-kernel-parameters)
