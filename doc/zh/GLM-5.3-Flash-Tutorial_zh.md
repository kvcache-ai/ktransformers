# GLM-5.3-flash：1M 长上下文，KTransformers 原生支持

## 1. 1M 长上下文与原生多模态

GLM-5.3-flash 原生支持最高 1M tokens 上下文，可以在一次请求中处理大型代码库、长篇文档和长程 Agent 任务，减少频繁切分上下文带来的信息损失。

模型还原生支持图片、视频、Reasoning 和 Tool Calling，可以直接接入使用 OpenAI 兼容接口的 Coding Agent。GLM-5.3-flash 约 321B 参数；45 层网络中包含 34 层 Linear Attention 和 11 层 DSA。

## 2. KTransformers 原精度支持

KTransformers（KT）直接读取 GLM-5.3-flash 的官方 FP8 权重，不需要转换模型，也不会对 Expert 权重再次量化。

FP8 模型约占 306 GiB，建议至少预留 350 GB 可用系统内存。当前实现支持：

- NVIDIA SM89 和 SM120 GPU（RTX 40、50 系列）
- TP1、TP2、TP4 和 TP8
- AVX-512 FP8 CPU Expert Kernel
- CPU-GPU Expert 异构推理和 Layerwise Prefill
- 文本、多图、单视频、Reasoning 和 Tool Calling

## 3. 安装

建议使用全新的 Python 3.11 环境，然后执行：

```bash
pip install "ktransformers[sglang]"
```

该命令会自动安装匹配的 KT Kernel、SGLang-KT 和 Transformers-KT，无需克隆源码仓库。

## 4. 启动

将 `/path/to/GLM-5.3-flash` 替换为模型目录。下面的配置默认开启 Layerwise Prefill、多模态和 Decode CUDA Graph。模型支持最高 1M 上下文；示例使用已经过验收的 `501025` tokens 配置。

### 4.1 四卡启动

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

### 4.2 单卡启动

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

当 GLM-5.3-flash 开启 Layerwise Prefill 时，当前实现会将 resident GPU experts 归一为 0；因此四卡示例中的 `--kt-num-gpu-experts 40` 不会在 Layerwise 阶段常驻 40 个 GPU experts。

服务默认监听 `http://localhost:30000`。启动完成后可检查接口：

```bash
curl http://localhost:30000/v1/models
```

OpenAI 兼容接口为：

```text
http://localhost:30000/v1/chat/completions
```

## 5. 多模态边界

- 单个请求可以包含纯文本，或纯文本加最多 8 张图片。
- 单个请求也可以包含纯文本加 1 个视频。
- 图片和视频不能出现在同一个请求中。
- 视频使用普通混合预填充，不走 Layerwise Prefill；解码仍支持 CUDA Graph。
