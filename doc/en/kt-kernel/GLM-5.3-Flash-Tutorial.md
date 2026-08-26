# GLM-5.3-flash: Native KTransformers Support for 1M Context

## 1. 1M Context and Native Multimodality

GLM-5.3-flash natively supports **a context window of up to 1M tokens**. A single request can process a large codebase, a long document, or a long-running agent task without repeatedly splitting the context.

The model also supports **images, video, reasoning, and tool calling**, and can be used directly by coding agents through an OpenAI-compatible API. GLM-5.3-flash has approximately 321B parameters. Its 45 layers comprise 34 Linear Attention layers and 11 DSA layers.

## 2. Native-Precision KTransformers Support

KTransformers (KT) reads the official GLM-5.3-flash FP8 weights directly. No model conversion or additional quantization of expert weights is required.

The FP8 model occupies approximately 306 GiB. Reserve at least 350 GB of available system memory. The current implementation supports:

- NVIDIA SM89 and SM120 GPUs (RTX 40 and 50 series)
- The AVX-512 FP8 CPU expert kernel
- Heterogeneous CPU-GPU expert inference and Layerwise Prefill
- A context window of up to 1M tokens
- Multimodality: text, multiple images, video
- Tool calling

## 3. Installation

Use a clean Python 3.11 environment and run:

```bash
pip install "ktransformers[sglang]"
```

This command installs compatible versions of KT Kernel, SGLang-KT, and Transformers-KT.

## 4. Launch

Replace `/path/to/GLM-5.3-flash` with the model directory. The configurations below enable Layerwise Prefill, multimodal input, and decode CUDA Graphs by default. The model supports up to 1M context; the examples use the validated `501025`-token configuration.

### 4.1 Four-GPU Launch

```bash
MODEL_PATH=/path/to/GLM-5.3-flash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --kt-weight-path "$MODEL_PATH" \
  --served-model-name GLM-5.3-flash \
  --host 0.0.0.0 \
  --tp-size 4 \
  --context-length 501025 \
  --mem-fraction-static 0.60 \
  --chunked-prefill-size 4096 \
  --kt-method FP8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 40 \
  --kt-gpu-prefill-token-threshold 2048 \
  --cuda-graph-bs 1 2 4 \
  --limit-mm-data-per-request '{"image":8,"video":1}' \
  --mm-process-config '{"image":{"max_pixels":1254400}}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

### 4.2 Single-GPU Launch

```bash
MODEL_PATH=/path/to/GLM-5.3-flash

CUDA_VISIBLE_DEVICES=0 \
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --kt-weight-path "$MODEL_PATH" \
  --served-model-name GLM-5.3-flash \
  --host 0.0.0.0 \
  --tp-size 1 \
  --context-length 501025 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 2048 \
  --kt-method FP8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 0 \
  --kt-gpu-prefill-token-threshold 2048 \
  --cuda-graph-bs 1 2 4 \
  --limit-mm-data-per-request '{"image":8,"video":1}' \
  --mm-process-config '{"image":{"max_pixels":1254400}}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

When Layerwise Prefill is enabled for GLM-5.3-flash, the current implementation normalizes the resident GPU expert count to zero. The `--kt-num-gpu-experts 40` setting in the four-GPU example therefore does not keep 40 experts resident during Layerwise Prefill.

The server listens on `http://localhost:30000` by default. Check the endpoint after startup:

```bash
curl http://localhost:30000/v1/models
```

The OpenAI-compatible endpoint is:

```text
http://localhost:30000/v1/chat/completions
```

## 5. Multimodal Request Boundaries

- A request may contain text only, or text with up to eight images.
- A request may instead contain text with one video.
- Images and video cannot appear in the same request.
- Video uses the regular mixed-prefill path rather than Layerwise Prefill. Decode CUDA Graphs remain supported.
