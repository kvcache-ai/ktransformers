# Qwen3.5 MoE KT LoRA Serving with SGLang-KT

Last updated: 2026-06-23

This guide documents the current KT-FT loop for Qwen3.5 MoE: train with KT SFT, convert the output once, and serve the fine-tuned result through SGLang with a single merged adapter path.

```text
KT SFT raw output
  -> convert_kt_to_sglang_adapter.py
  -> <MERGED_ADAPTER_DIR>
  -> sglang --lora-paths <name>=<MERGED_ADAPTER_DIR>
  -> server auto-splits expert / non-expert internally
  -> request model=<served_model>:<name>
```

Training-side KT SFT docs remain separate. This page focuses on the bridge from trained LoRA artifacts to online inference.

## 1. Scope

Current supported and validated workflow:

- Base model: Qwen3.5 MoE, for example `Qwen3.5-35B-A3B`
- KTransformers version: v0.6.3 or newer
- KT expert weights: AMX/BF16 SFT-compatible KT CPU expert path
- User-facing serving input: one converted merged adapter directory
- Runtime split: expert LoRA goes to the KT CPU expert path; non-expert LoRA goes to SGLang's LoRA manager. This split happens automatically at server startup.
- This workflow is for KT MoE expert LoRA artifacts. Standard dense-model PEFT LoRA adapters usually do not need this converter.

## 2. Artifacts At Each Stage

### Raw KT SFT output

After LLaMA-Factory + KT training, the output directory contains two LoRA artifacts:

```text
<KT_SFT_OUTPUT_DIR>/
  adapter_model.safetensors      # non-expert LoRA
  fused_expert_lora.safetensors  # expert LoRA in KT fused format
  adapter_config.json
```

Do not pass this raw directory directly to SGLang serving.

### Converted merged adapter

Run the converter once to produce the serving input:

```text
<MERGED_ADAPTER_DIR>/
  adapter_config.json
  adapter_model.safetensors
```

This merged directory contains both expert and non-expert LoRA tensors in one PEFT-style adapter. Pass only this directory to `--lora-paths`.

## 3. Convert Once

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  <KT_SFT_OUTPUT_DIR> \
  <MERGED_ADAPTER_DIR> \
  --base-model-name-or-path /path/to/Qwen3.5-35B-A3B \
  --overwrite
```

Example:

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  saves/KT_FT_qwen35B_Moe_custom \
  saves/KT_FT_qwen35B_Moe_custom_sglang \
  --base-model-name-or-path /mnt/data3/models/Qwen3.5-35B-A3B \
  --overwrite
```

The converter reads `fused_expert_lora.safetensors` and the existing non-expert `adapter_model.safetensors`, then writes one merged adapter directory.

If the raw KT SFT output does not contain an `adapter_config.json` with `lora_alpha`, pass `--lora-alpha <value>` explicitly. The converter does not fold LoRA scaling into the tensors; runtime scaling remains `lora_alpha / r`.

Optional split outputs for debugging:

```bash
python kt-kernel/scripts/convert_kt_to_sglang_adapter.py \
  <KT_SFT_OUTPUT_DIR> \
  <MERGED_ADAPTER_DIR> \
  --base-model-name-or-path /path/to/Qwen3.5-35B-A3B \
  --expert-output-dir <EXPERT_ADAPTER_DIR> \
  --nonexpert-output-dir <NONEXPERT_ADAPTER_DIR> \
  --overwrite
```

For normal serving, only `<MERGED_ADAPTER_DIR>` is needed.

## 4. Launch SGLang

Use the KTransformers SGLang fork from this repository and point `PYTHONPATH` at both `kt-kernel/python` and `third_party/sglang/python`.

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

Important points:

- Pass only one merged adapter through `--lora-paths`.
- Do not also pass `--kt-expert-lora-path` in the normal user workflow.
- At startup, the server detects the merged KT MoE adapter, splits it internally, and writes runtime cache directories under `$TMPDIR/sglang_kt_lora_cache/` (or `$SGLANG_KT_LORA_CACHE_DIR` if set).
- Prefer `--lora-backend triton` for Qwen3.5 full-LoRA generation.

Current constraints:

- single merged KT composite adapter only
- `--kt-num-gpu-experts 0`
- do not enable `--kt-enable-dynamic-expert-update`
- do not use `--kt-gpu-prefill-token-threshold`
- `--max-running-requests` must be at least 2
- use an AMX/BF16 SFT-compatible KT method such as `AMXINT4`, `AMXINT8`, `AMXBF16`, or `BF16`

## 5. Request Semantics

The OpenAI-compatible request `model` field uses names, not paths.

```text
--served-model-name qwen3.5-kt-ft
--lora-paths qwen35b_lora=/path/to/merged_adapter
```

Request behavior in the current single-adapter implementation:

```text
model=qwen3.5-kt-ft
=> base + KT expert LoRA

model=qwen3.5-kt-ft:qwen35b_lora
=> base + KT expert LoRA + SGLang non-expert LoRA
```

The suffix after `:` must match the left-side name in `--lora-paths`.

If you need a true base-only comparison, launch a separate server without `--lora-paths`.

## 6. Smoke Test

```bash
curl -sS http://127.0.0.1:30006/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-kt-ft:qwen35b_lora",
    "messages": [{"role": "user", "content": "Explain what LoRA is in one sentence."}],
    "temperature": 0.7,
    "max_tokens": 160,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Startup logs should include lines similar to:

```text
Prepared merged KT LoRA adapter ... for runtime: expert=... nonexpert=...
Loaded KT expert LoRA for layer ...
Using triton as backend of LoRA kernels.
```

## 7. Advanced: Manual Split Serving

The older split-runtime contract is still available for debugging:

```bash
--kt-expert-lora-path <EXPERT_ADAPTER_DIR> \
--enable-lora \
--lora-paths <NONEXPERT_LORA_NAME>=<NONEXPERT_ADAPTER_DIR>
```

This is not the recommended user-facing path. Normal users should pass one merged adapter directory through `--lora-paths` only.

## 8. Troubleshooting

### `Got LoRA adapter that has never been loaded: lora0`

The adapter name in the request must match the left side of `--lora-paths`. If you launched with `qwen35b_lora=...`, request `model=qwen3.5-kt-ft:qwen35b_lora`, not `:lora0`.

### No visible adapter effect

Make sure you are serving the converted merged adapter directory produced by the converter, not the raw KT SFT output directory or a different test adapter.

### `connection refused`

Check that the server is listening on the port you curl, and remember the example above binds to `127.0.0.1`, not `0.0.0.0`.

### Server resolves upstream SGLang instead of this checkout

```bash
python - <<'PY'
import inspect
import sglang.srt.models.qwen3_5 as qwen3_5
print(inspect.getfile(qwen3_5))
PY
```

The path should come from this repository's `third_party/sglang`.
