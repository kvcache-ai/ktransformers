# Qwen3.5 MoE KT LoRA 的 SGLang-KT Serving

最后更新：2026-06-23

本文档描述当前 Qwen3.5 MoE 的 KT-FT 闭环：用 KT SFT 完成微调，转换一次输出，再通过 SGLang 用单个 merged adapter path 把微调结果服务化。

```text
KT SFT 原始输出
  -> convert_kt_to_sglang_adapter.py
  -> <MERGED_ADAPTER_DIR>
  -> sglang --lora-paths <name>=<MERGED_ADAPTER_DIR>
  -> server 内部自动拆分 expert / non-expert
  -> 请求 model=<served_model>:<name>
```

训练侧 KT SFT 文档仍然独立维护；本文重点说明从已训练 LoRA artifacts 到在线推理的连接部分。

## 1. 范围

当前已验证路径：

- 基座模型：Qwen3.5 MoE，例如 `Qwen3.5-35B-A3B`
- KTransformers 版本：v0.6.3 或更新版本
- KT expert 权重：AMX/BF16 SFT 兼容的 KT CPU expert 路径
- 用户侧 serving 输入：一个 converted merged adapter 目录
- Runtime 内部仍会 split：expert LoRA 走 KT CPU expert path，non-expert LoRA 走 SGLang LoRA manager，但这一步对用户不可见
- 该工作流面向 KT MoE expert LoRA 产物；普通 dense 模型的标准 PEFT LoRA 通常不需要使用此转换器（converter）。

## 2. 各阶段产物

### 原始 KT SFT 输出

LLaMA-Factory + KT 训练完成后，输出目录里有两个 LoRA 文件：

```text
<KT_SFT_OUTPUT_DIR>/
  adapter_model.safetensors      # non-expert LoRA
  fused_expert_lora.safetensors  # KT fused expert LoRA
  adapter_config.json
```

不要把 raw 训练目录直接传给 SGLang serving。

### Convert 后的 merged adapter

converter 一次性生成 serving 输入：

```text
<MERGED_ADAPTER_DIR>/
  adapter_config.json
  adapter_model.safetensors
```

这个 merged 目录同时包含 expert 和 non-expert LoRA。正常 serving 只需要传这一个目录。

## 3. 转换一次

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  <KT_SFT_OUTPUT_DIR> \
  <MERGED_ADAPTER_DIR> \
  --base-model-name-or-path /path/to/Qwen3.5-35B-A3B \
  --overwrite
```

示例：

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  saves/KT_FT_qwen35B_Moe_custom \
  saves/KT_FT_qwen35B_Moe_custom_sglang \
  --base-model-name-or-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --overwrite
```

converter 会读取 `fused_expert_lora.safetensors` 和已有的 non-expert `adapter_model.safetensors`，写出一个 merged adapter 目录。

如果原始 KT SFT 输出目录没有包含带 `lora_alpha` 的 `adapter_config.json`，需要显式传入 `--lora-alpha <value>`。converter 不会把 LoRA scaling 折进 tensor；运行时 scaling 仍然是 `lora_alpha / r`。

如需调试，也可以额外输出 split 目录：

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  <KT_SFT_OUTPUT_DIR> \
  <MERGED_ADAPTER_DIR> \
  --base-model-name-or-path /path/to/Qwen3.5-35B-A3B \
  --expert-output-dir <EXPERT_ADAPTER_DIR> \
  --nonexpert-output-dir <NONEXPERT_ADAPTER_DIR> \
  --overwrite
```

正常 serving 只需要 `<MERGED_ADAPTER_DIR>`。

## 4. 启动 SGLang

请使用本仓库的 KTransformers SGLang fork，并把 `PYTHONPATH` 指向 `kt-kernel/python` 和 `third_party/sglang/python`。

```bash
cd /path/to/ktransformers

PYTHONPATH=/path/to/ktransformers/kt-kernel/python:/path/to/ktransformers/third_party/sglang/python:$PYTHONPATH \
python -m sglang.launch_server \
  --host 127.0.0.1 \
  --port 30006 \
  --model-path /path/to/Qwen3.5-35B-A3B \
  --tokenizer-path /path/to/Qwen3.5-35B-A3B \
  --kt-weight-path /path/to/Qwen3.5-35B-A3B-AMXINT4 \
  --kt-method AMXINT4 \
  --kt-cpuinfer 60 \
  --kt-threadpool-count 2 \
  --kt-numa-nodes 0 1 \
  --kt-num-gpu-experts 0 \
  --attention-backend flashinfer \
  --trust-remote-code \
  --mem-fraction-static 0.98 \
  --chunked-prefill-size 4096 \
  --max-running-requests 2 \
  --max-total-tokens 32000 \
  --served-model-name qwen3.5-kt-ft \
  --enable-mixed-chunk \
  --tensor-parallel-size 4 \
  --enable-p2p-check \
  --disable-cuda-graph \
  --disable-custom-all-reduce \
  --enable-lora \
  --lora-backend triton \
  --lora-paths qwen35b_lora=/path/to/KT_FT_qwen35B_Moe_custom_sglang \
  --log-level info
```

要点：

- 用户只需要传一个 merged adapter：`--lora-paths <name>=<MERGED_ADAPTER_DIR>`
- 正常 workflow 不要再额外传 `--kt-expert-lora-path`
- server 启动时会自动识别 merged KT MoE adapter，并在 `$TMPDIR/sglang_kt_lora_cache/`（或 `$SGLANG_KT_LORA_CACHE_DIR`）下生成 runtime cache
- Qwen3.5 full LoRA 生成优先使用 `--lora-backend triton`

当前限制：

- 只支持单个 merged KT composite adapter
- `--kt-num-gpu-experts 0`
- 不启用 `--kt-enable-dynamic-expert-update`
- 不使用 `--kt-gpu-prefill-token-threshold`
- `--max-running-requests` 必须至少为 2
- 使用 AMX/BF16 SFT 兼容 KT method，例如 `AMXINT4`、`AMXINT8`、`AMXBF16`、`BF16`

## 5. 请求语义

OpenAI-compatible 请求里的 `model` 字段用 name，不用 path。

```text
--served-model-name qwen3.5-kt-ft
--lora-paths qwen35b_lora=/path/to/merged_adapter
```

当前 single-adapter 实现的请求语义：

```text
model=qwen3.5-kt-ft
=> base + KT expert LoRA

model=qwen3.5-kt-ft:qwen35b_lora
=> base + KT expert LoRA + SGLang non-expert LoRA
```

冒号后的 adapter 名必须和 `--lora-paths` 左侧注册名一致。

如果需要 true base-only 对照，请单独启动一个不带 `--lora-paths` 的服务。

## 6. Smoke Test

```bash
curl -sS http://127.0.0.1:30006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-kt-ft:qwen35b_lora",
    "messages": [{"role": "user", "content": "用一句话解释什么是 LoRA。"}],
    "temperature": 0.7,
    "max_tokens": 160,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

启动日志里应能看到类似输出：

```text
Prepared merged KT LoRA adapter ... for runtime: expert=... nonexpert=...
Loaded KT expert LoRA for layer ...
Using triton as backend of LoRA kernels.
```

## 7. 高级：手动 split serving

旧 split runtime 仍可用于调试：

```bash
--kt-expert-lora-path <EXPERT_ADAPTER_DIR> \
--enable-lora \
--lora-paths <NONEXPERT_LORA_NAME>=<NONEXPERT_ADAPTER_DIR>
```

这不是推荐的用户路径。正常用户只需要通过 `--lora-paths` 传一个 merged adapter 目录。

## 8. Troubleshooting

### `Got LoRA adapter that has never been loaded: lora0`

请求里的 adapter 名必须和 `--lora-paths` 左侧一致。如果启动时写的是 `qwen35b_lora=...`，请求应使用 `model=qwen3.5-kt-ft:qwen35b_lora`，而不是 `:lora0`。

### 看不出 adapter 效果

确认 serving 使用的是 converter 生成的 merged adapter 目录，而不是原始 KT SFT 输出目录或其他测试 adapter。

### `connection refused`

确认 server 监听的端口与 curl 一致；上面的示例绑定的是 `127.0.0.1`，不是 `0.0.0.0`。

### Server 解析到了上游 SGLang，而不是当前 checkout

```bash
python - <<'PY'
import inspect
import sglang.srt.models.qwen3_5 as qwen3_5
print(inspect.getfile(qwen3_5))
PY
```

路径应来自本仓库的 `third_party/sglang`。
