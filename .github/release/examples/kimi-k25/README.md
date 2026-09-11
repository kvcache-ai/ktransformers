# Kimi K2.5: LF main training reference

[中文使用教程](TUTORIAL_zh.md)

Reference for the release hardware gate, not a validated public-PyPI recipe.
Use unmodified LLaMA-Factory official main `100e9a42c6c09f8f7849b70d60f3da445fb2024b`
with its built-in `empty` template. Prepare the data once with the script below;
no private LF template, runtime callback or model-directory patch is required.

## Verification

On sap4, all 9,478 train and 499 evaluation records produced the same input IDs
and answer-only labels as the previous successful template, at both S512 and
S8192. All 230 installed LF Python files match the official main source.
The LF-main 8-GPU, two-step train/save smoke passed. Losses and gradient norms
matched the previous candidate-template run below; both saved adapter files
also had identical SHA-256 hashes. The new artifact's SGLang reload attempt
ended with SIGKILL before serving requests, so that attempt did not pass
inference acceptance. The earlier successful generation result is separate.

The previous candidate-template smoke established:

- 8 GPUs, CPU AMX / TP2 / 96 workers, RAWINT4 routed experts (group size 32).
- 32 Neko examples, B1/GAS1, S512 **upper bound**, two optimizer steps.
- Packing disabled; CPU and GPU activation recomputation enabled.
- Loss: **2.26953125 → 2.0419921875**; gradient norm: **0.88868 → 0.84890**.
- Final save: 610 ordinary LoRA tensors and 360 fused expert LoRA tensors,
  all finite; all 305 ordinary and 180 expert LoRA B tensors nonzero.
- Fresh-process SGLang reload on 4 GPUs: all 60 expert LoRA layers loaded;
  all four TP ranks installed the MLA LoRA correction on 61 layers.
  Three requests selecting `kimi-k25:release` returned nonempty answers with
  `finish_reason=stop`, including `17 + 25` → `42`.

This is a train/save/reload smoke, not a throughput or style-convergence
result. `save_strategy: 'no'` disables intermediate checkpoints; LF still saves
the final adapter. `save_only_model: true` does not support exact optimizer resume.
The attention targets in the YAML are supplemented by fused routed-expert LoRA.
Loading and successful generation do not prove a numerical LoRA effect: an
enabled/disabled logits comparison was not run for this artifact.

## Environment boundary

The successful run used installed **candidate wheels**, without developer
`PYTHONPATH` or edits to the original model directory. Third-party dependencies
were shared with an existing environment. It does not validate the release-CI
wheels, a clean installation, or a portable multi-ISA KT wheel.

| Component | Tested source base + candidate changes |
| --- | --- |
| KT | `63d06ff6f634bb0be5228d9c3051b23f2fbcd5b8`; previously validated AMX binary |
| Transformers-KT | `96fa9fd336dfd088df4404e2529bf9be22e71309` + legacy Kimi checkpoint / MoonViT compatibility |
| Accelerate-KT | Tree identical to main `df853a7da9f28a2b95ad2c8bfcc913e7171b4b10` |
| LLaMA-Factory | Official main `100e9a42c6c09f8f7849b70d60f3da445fb2024b`, unmodified |
| SGLang-KT (reload) | Verified candidate runtime; the four required serving changes were merged in [SGLang #94](https://github.com/kvcache-ai/sglang/pull/94), main `112baf62` |

The Transformers compatibility logic is now in official main `2271a046`.
Other candidate changes are not implied by package version numbers; they must
land and be pinned before advertising a public-package recipe.
In particular, this reference does **not** enable CPU retain / GPU
recompute: the tested RAWINT4 implementation rejects that combination.
The reload stack also packages KT's adapter converter as `kt-convert-lora`.
These checks use Python 3.11 / Torch 2.9.1+cu128, independently of the release
workflow's Python 3.12 build environment.

## Prepare the data once

Download [prepare_kimi_data.py](prepare_kimi_data.py), [train.yaml](train.yaml)
and [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml) as raw files into one working directory on
your training machine. LF does not run the conversion script automatically:
run it before training and point `dataset_dir` to its output, not the raw data.
The commands below use repository-root paths; for a standalone download, follow
the working-directory commands in the [Chinese tutorial](TUTORIAL_zh.md).

Input is a JSON array of text-only, single-turn records:

```json
[{"prompt": "你好", "answer": "你好喵。"}]
```

An optional `system` string is supported. History, tools, media and already
formatted chat/thinking markers are rejected. Keep the original train/eval split;
this script does not split, deduplicate, shuffle, drop or truncate examples.

From this repository root, with the training dependencies installed:

```bash
python .github/release/examples/kimi-k25/prepare_kimi_data.py \
  --model /path/to/Kimi-K2.5 --trust-remote-code \
  --input /path/to/neko_train.json \
  --eval-input /path/to/neko_eval.json \
  --output-dir /path/to/prepared-neko
```

Omit `--eval-input` if no evaluation split is needed. The model's native tokenizer
renders non-thinking prompts and answer end markers. Each record must preserve
the native full-dialogue token sequence when prompt and answer are encoded
separately; any mismatch fails the command. No model weights are loaded.

The new output directory contains `neko_train.json`, optional `neko_eval.json`,
`dataset_info.json` and `preprocessing_manifest.json` with source/output hashes,
template hash, row counts and token/label checksums. Existing directories are
never overwritten. Do not feed prepared data through the script again.

## Train

Set the model, expert-weight, prepared dataset and new output paths in `train.yaml`.
Keep `template: empty`, `train_on_prompt: false`, `packing: false` and
`neat_packing: false`. Thinking is disabled during data preparation, not via an
ineffective `enable_thinking` override on LF's empty template. LF still tokenizes
the prepared strings and applies its normal answer-only loss and length limit.

```bash
USE_KT=1 accelerate launch \
  --config_file .github/release/examples/kimi-k25/fsdp2_8gpu.yaml \
  --main_process_port 29761 --no_python llamafactory-cli train \
  .github/release/examples/kimi-k25/train.yaml
```

The saved directory must include `adapter_model.safetensors`,
`fused_expert_lora.safetensors` and `kt_adapter_manifest.json`.
Inference continues to use Kimi's native chat template, not the training-only
`empty` template. The example performs two steps without evaluation; generating
an eval JSON does not implicitly enable evaluation or turn this into a full run.
