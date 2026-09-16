# Fine-tune Kimi K2.5 with LoRA: installation, training, and chat

[Chinese version](README.md)

This guide uses KTransformers `0.7.0.post4` and the NekoQA dataset to teach Kimi K2.5
a catgirl-inspired conversational style. You will then load the trained LoRA in SGLang
and chat with the model. The original model weights stay unchanged; the LoRA is saved
separately.

Follow the four steps below. You only need to supply paths for the model and training
output. If you want to check your setup first, use the optional four-step smoke
test and resume in the appendix after Step 2.

## Before you begin

Reference setup used for validation:

| Component | Configuration |
| --- | --- |
| GPU | 8 × RTX 5090 for training; 4 × RTX 5090 for inference |
| CPU / RAM | Dual AMD EPYC 9355; approximately 1.5 TiB RAM |
| OS | Linux x86_64, glibc 2.35+; Python 3.11 and 3.12 validated separately |
| CUDA | CUDA 12.8 toolkit and a C++ compiler; tested driver: 580.173.02 |

Check that `nvidia-smi` and `nvcc --version` work and that the required GPUs are free.
In addition to the model, environments, and caches, allow room for training outputs:
**each checkpoint is approximately 29 GiB, and the converted inference LoRA is
approximately 9.6 GiB**. This configuration retains up to two checkpoints, plus the
final LoRA and temporary files. Use persistent storage with ample free space; do not
put the output under `/dev/shm`.

This guide covers **text-only LoRA fine-tuning of Kimi K2.5**.

## 1. Install

**The core packages come from PyPI. Matching LLaMA-Factory, PEFT, and TRL wheels are
included in the release kit.** The commands below install separate training and
inference environments. You do not need to clone repositories or edit source code.

Open a Bash shell on a disk with sufficient space, then download and unpack the kit:

```bash
mkdir kimi-post4-work
cd kimi-post4-work
curl --fail --location --retry 5 --remote-name \
  https://github.com/kvcache-ai/ktransformers/releases/download/v0.7.0.post4/kimi-k25-post4-user-kit-r2.tar.gz
printf '%s  %s\n' \
  88fa5e473e06b67b5ac39605de82da09207ba1b82555f6d4b98d2b6a81de8813 \
  kimi-k25-post4-user-kit-r2.tar.gz | sha256sum --check
tar -xzf kimi-k25-post4-user-kit-r2.tar.gz
cd kimi-k25-post4
```

Run the remaining commands from this `kimi-k25-post4/` directory in the **same shell**.
The kit contains the training YAML files, data-preparation scripts, and locked
installation manifests.

Create two environments: `train-env` for training and `serve-env` for inference. Use
the same Python version for both; neither environment needs to be activated manually.
Set `KIMI_PYTHON` below to either Python 3.12 or 3.11. The matching lockfiles are
selected automatically. If PyPI is slow in your region, you may use the Tsinghua
mirror:

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
export KIMI_PYPI_INDEX=https://pypi.org/simple
# In mainland China, the previous line can be replaced with: https://pypi.tuna.tsinghua.edu.cn/simple
KIMI_PYTHON=python3.12  # Change to python3.11 if using Python 3.11
KIMI_PYTHON_TAG=$("$KIMI_PYTHON" -c '
import sys
assert sys.version_info[:2] in ((3, 11), (3, 12)), "Use Python 3.11 or 3.12"
print("cp%d%d" % sys.version_info[:2])
') || exit 1
"$KIMI_PYTHON" -m venv train-env
train-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
train-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --no-binary=antlr4-python3-runtime \
  --require-hashes --find-links training-tools -r "locks/$KIMI_PYTHON_TAG/train.lock"
train-env/bin/python -m pip check

"$KIMI_PYTHON" -m venv serve-env
serve-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
serve-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --require-hashes -r "locks/$KIMI_PYTHON_TAG/serve.lock"
serve-env/bin/python -m pip check
```

**Success check:** both `pip check` commands print `No broken requirements found.`
The lockfiles pin the compatible versions. Do not install upstream `transformers` or
`accelerate` into these environments, or replace the included LLaMA-Factory wheel
with its latest upstream release.

## 2. Prepare the model and data

Set `KIMI_MODEL` to the model directory and `KIMI_OUTPUT` to a **new** training
output directory. NekoQA is downloaded and processed by the commands below; you do
not need to put a dataset path in the YAML.

If you already have the exact model or dataset revisions, skip their respective
`hf download` commands. Downloads require access to Hugging Face. For an offline
machine, copy the complete model and dataset from a connected machine; changing the
PyPI mirror will not fix a Hugging Face download issue.

```bash
export KIMI_MODEL=/absolute/path/to/Kimi-K2.5
export KIMI_OUTPUT=/absolute/path/to/new-kimi-output
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$KIMI_OUTPUT"

train-env/bin/hf download moonshotai/Kimi-K2.5 \
  --revision 54383e83fa343a1331754112fb9e3410c55efa2f --local-dir "$KIMI_MODEL"
train-env/bin/hf download liumindmind/NekoQA-10K NekoQA-10K.json \
  --repo-type dataset --revision 1b2110c996a8237823b86c1a3d3e8a6762b38430 \
  --local-dir data-source
train-env/bin/python tools/split_nekoqa.py \
  --input data-source/NekoQA-10K.json --output-dir neko-splits
train-env/bin/python tools/prepare_kimi_data.py \
  --model "$KIMI_MODEL" --trust-remote-code \
  --input neko-splits/neko_train.json --eval-input neko-splits/neko_eval.json \
  --output-dir prepared-neko
```

**Success check:** `prepared-neko/` contains processed data and `dataset_info.json`.
The scripts split the training, validation, and held-out questions and format them
for Kimi's chat template. The YAML setting `template: empty` is intentional for this
preprocessed data. Keep it and do not skip preprocessing.

Use the complete original model. Do not convert it to full BF16 or modify its
files. `--trust-remote-code` executes downloaded model code, so verify that you
trust the model source.

## 3. Train the style LoRA

The following command uses 8 GPUs and [train-neko.yaml](train-neko.yaml) for one
epoch over NekoQA. Per-GPU batch size is 1, gradient accumulation is 8, and the
maximum sequence length is 4096. Paths are passed to the YAML on the command line;
you do not need to edit the file:

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-neko.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/neko"
```

**Expected progress:** the log continues to report steps, loss, and grad norm,
without `nan` or `inf`. The epoch has about 149 steps. Training validates and saves
at step 100, producing `$KIMI_OUTPUT/neko/checkpoint-100/`. Let training finish
normally before proceeding to inference.

This configuration trains both attention and expert LoRA with rank 8, alpha 16,
and dropout 0. Packing is disabled, and both CPU and GPU activations are
recomputed. Keep these settings unchanged for your first run. The 4096-token
limit is a maximum; samples are not padded to that length.

The next step uses `checkpoint-100` because that checkpoint has already shown a
clear style change in validation. Keep the one-epoch configuration: **do not set
`max_steps: 100` merely to obtain the step-100 checkpoint**, because doing so
changes the learning-rate schedule. If training is interrupted, keep the complete
checkpoint and append
`resume_from_checkpoint="$KIMI_OUTPUT/neko/checkpoint-100"` to the training command
to resume from that save point. Use an existing checkpoint path and leave the
other training settings unchanged.

## 4. Load the LoRA and chat

**Convert the training artifact first.** The checkpoint contains LoRA weights for
regular modules and experts. This script converts both to SGLang's adapter format
without changing the original checkpoint:

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/neko/checkpoint-100"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-neko"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

**Start the inference server.** Make sure training has exited and GPUs 0–3 are
free, then run the following in the same shell:

```bash
set -o pipefail
PATH="$PWD/serve-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3 \
serve-env/bin/python -m sglang.launch_server \
  --model-path "$KIMI_MODEL" --trust-remote-code \
  --served-model-name kimi --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 4 --dtype bfloat16 --context-length 2048 \
  --max-total-tokens 4096 --chunked-prefill-size 256 --max-running-requests 4 \
  --mem-fraction-static 0.75 --disable-cuda-graph --disable-radix-cache \
  --attention-backend triton --grammar-backend llguidance \
  --kt-method RAWINT4 --kt-weight-path "$KIMI_MODEL" \
  --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --random-seed 42 --watchdog-timeout 900 \
  --enable-lora --lora-backend triton --lora-paths "neko=$KIMI_ADAPTER" \
  2>&1 | tee "$KIMI_ADAPTER.server.log"
```

The first launch may take time to compile kernels and load the model. The server
keeps this terminal occupied. Its log should report `Loaded KT expert LoRA for layer ...`
for expert layers 1–60.

**Send a question.** In a second terminal, wait for `/health` to succeed and then
send a chat request:

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/health
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi:neko","messages":[{"role":"user","content":"我今天学习有点累，请用两句话鼓励我。"}],"temperature":0,"seed":42,"max_tokens":1024,"chat_template_kwargs":{"thinking":false,"enable_thinking":false}}'
```

`kimi:neko` selects the loaded LoRA. Inspect `choices[0].message.content` in the
response: it should be a complete answer in the trained style. The prompt does
not tell the model to play a catgirl; this lets you observe the style it learned.

You have now installed, trained, and run the adapter. The appendices below are
optional.

## Appendix

<details>
<summary>Four-step smoke test and resume</summary>

Run this after Steps 1 and 2, in the same shell and working directory, with 8
free GPUs. It checks training, saving, resuming, and adapter loading. Use Step 3
for the actual style training. The two smoke runs and one adapter conversion
require at least another 140 GiB of persistent disk space, excluding the model,
environments, and caches.

First, run four steps into a separate `smoke-a/` directory:

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-a"
```

A successful run creates `checkpoint-2` and `checkpoint-4`, with finite loss
and grad norm. Restart from the step-2 checkpoint into another directory:

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-b" \
  resume_from_checkpoint="$KIMI_OUTPUT/smoke-a/checkpoint-2"
```

The log should resume at step 2 and finish at step 4. Keep the GPU count, data,
and training configuration unchanged. `max_steps: 4` means four steps total, not
four *additional* steps. `resume_from_checkpoint` restores the training state;
`adapter_name_or_path` only loads the LoRA and is not a substitute.

To check inference, run the conversion below, then use the **server startup and
chat commands** from Step 4. Do not rerun Step 4's `checkpoint-100` path assignment
or conversion command:

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/smoke-b/checkpoint-4"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-smoke"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

When finished, stop the inference server with Ctrl-C and confirm that its GPUs
are free before starting Step 3. The production run starts from the original
model, not the smoke-test LoRA.

</details>

<details>
<summary>Evaluate the trained style on 32 held-out questions</summary>

Step 2 reserves 32 questions in `neko-splits/heldout.json` that are never used
for training. Do not add a catgirl role instruction. Check whether the tone
changes while answers remain relevant and complete.

To compare with the base model, stop the LoRA server and start a **new process**
with the Step 4 command but remove `--enable-lora`, `--lora-backend`, and
`--lora-paths`. Use `model: kimi` in the request. This version loads CPU expert
LoRA at startup; merely changing the model name within the same server does
not disable it.

In a second terminal, change to the same `kimi-k25-post4/` directory and run
the script below. Its output file must not already exist, so previous results
are not overwritten. For a separate base-model comparison, change `model_id`
to `kimi` and use a different output filename.

```bash
serve-env/bin/python - <<'PY'
import json
from pathlib import Path
import requests

model_id = "kimi:neko"
questions = json.loads(Path("neko-splits/heldout.json").read_text())
session = requests.Session()
session.trust_env = False
with Path("neko-heldout-responses.jsonl").open("x", encoding="utf-8") as output:
    for index, row in enumerate(questions, 1):
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": 0, "seed": 42, "max_tokens": 1024,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        response = session.post("http://127.0.0.1:30000/v1/chat/completions", json=body, timeout=900)
        response.raise_for_status()
        output.write(json.dumps({"prompt": row["prompt"], "response": response.json()}, ensure_ascii=False) + "\n")
        output.flush()
        print(f"{index}/{len(questions)}", flush=True)
PY
```

Open `neko-heldout-responses.jsonl` and inspect `choices[0].message.content`
and `finish_reason`. A `finish_reason` of `length` means the answer was cut off
at the token limit and should not count as complete.

Besides style, check factual accuracy and instruction following. Examples
include “17+25, output only the number,” exact text copying, and JSON-format
instructions.

</details>

<details>
<summary>Troubleshooting</summary>

| Symptom | Check |
| --- | --- |
| pip download times out | Retry the same install command or use the Tsinghua mirror. If the mirror lacks a pinned version, switch back to `https://pypi.org/simple`. Keep the version and hash checks. |
| `hf download` times out | Check Hugging Face access. You can download the pinned revision on a connected machine and copy it in full. Changing the PyPI mirror will not help. |
| pip reports dependency conflicts or cannot find LF/PEFT/TRL | Use fresh environments, unpack the matching release kit, and keep `--find-links training-tools`. |
| Unexpected loss or repeated chat-template text | Run data preparation first; keep `template: empty` and `packing: false`. |
| Rank 0 disappears during loading; Gloo connection closes | Check the rank-0 log, host OOM records, and free RAM on **each NUMA node**. One node can run out even if the host has free RAM. Avoid putting large checkpoints in RAM or loading multiple models at once. |
| Resume starts from step 0 | Pass `resume_from_checkpoint` pointing to a complete checkpoint, not just an adapter. |
| Inference looks like the base model or only partly changed | Confirm conversion included both LoRA types, start a new server, check that expert layers 1–60 loaded, and request `kimi:neko`. |

Do not use `--no-deps` or disable version checks to bypass installation
conflicts. Do not set `FORCE_TORCHRUN` or wrap the training command in another
`torchrun`.

</details>

<details>
<summary>Pinned versions, configuration files, and validation details</summary>

The lockfiles contain these versions; you do not need to install them individually:

| Source | Pinned version |
| --- | --- |
| PyPI: `ktransformers`, `kt-kernel`, `sglang-kt` | `0.7.0.post4` |
| PyPI: `transformers-kt` | `5.6.0.post5` |
| PyPI: `accelerate-kt` | `1.14.0.post3` |
| Release `training-tools/`: LLaMA-Factory | `0.9.6.dev0+kt.20260912` |
| Release `training-tools/`: PEFT and TRL | `0.18.1+kt.20260912` and `0.24.0+kt.20260912` |

The LLaMA-Factory wheel comes from the
[pinned KT installation branch](https://github.com/yyj6666667/LlamaFactory/tree/564574eb0214840ea4b9464b2f9178a52a075d73).
The PEFT and TRL wheels only adjust dependency metadata; they do not change
runtime code. These three companion wheels are not on PyPI. Upstream
`transformers` and `accelerate` use the same Python module names as the KT
packages, so do not install both variants together. The installation uses
Torch 2.9.1. ANTLR 4.9.3 is built from a hash-checked official source
distribution; the other dependencies use wheels.

`locks/cp311/` and `locks/cp312/` pin package versions and file hashes for
their respective Python interpreters. Python 3.11 uses NumPy 2.4.6, SciPy
1.17.1, and ContourPy 1.3.3; Python 3.12 retains the separately validated
versions. Both use the same training YAML without parameter changes.

The repository and release-kit YAML values match. File mapping:

| Repository file | Release-kit file | Purpose |
| --- | --- | --- |
| [train-neko.yaml](train-neko.yaml) | `examples/train-neko.yaml` | One-epoch style training |
| [train.yaml](train.yaml) | `examples/train-smoke.yaml` | Four-step smoke test |
| [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml) | `examples/fsdp2_8gpu.yaml` | Eight-GPU launch configuration |

The data are deduplicated and split with seed 42 into 9,477 training examples,
468 validation examples, and 32 held-out questions. Preprocessing uses Kimi's
native non-thinking template; only the answer contributes to loss. Keep
`template: empty` and `train_on_prompt: false`; do not feed raw Neko JSON to
training. The longest processed example is 2,120 tokens.

A complete checkpoint should contain the following. Keep `save_only_model: false`:

- `adapter_model.safetensors`, `fused_expert_lora.safetensors`, and adapter configuration;
- `kt_optimizer.index.json` and eight `optimizer_rank_*.pt` files;
- `scheduler.pt`, `trainer_state.json`, and eight `rng_state_*.pth` files.

After both smoke runs, you can check that continuous training and resumed
training produced identical LoRA tensors. The comparison checks tensor
contents, not file serialization order:

```bash
train-env/bin/python - "$KIMI_OUTPUT" <<'PY'
from pathlib import Path
import sys
import torch
from safetensors import safe_open

root = Path(sys.argv[1])
for name in ("adapter_model.safetensors", "fused_expert_lora.safetensors"):
    a = root / "smoke-a/checkpoint-4" / name
    b = root / "smoke-b/checkpoint-4" / name
    with safe_open(a, framework="pt", device="cpu") as left, safe_open(b, framework="pt", device="cpu") as right:
        assert set(left.keys()) == set(right.keys()), name
        for key in left.keys():
            assert torch.equal(left.get_tensor(key), right.get_tensor(key)), (name, key)
    print(name, "EXACT_RESUME_PASSED")
PY
```

Both files should print `EXACT_RESUME_PASSED`. The full validation also compared
each rank's optimizer and RNG state, scheduler, and post-resume loss. Adapter
conversion exported 610 regular LoRA tensors and 138,240 expert LoRA tensors;
SGLang loaded expert layers 1–60.

For source provenance, package checksums, and validation results, see the
[release and validation record](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.0.post4).
Keep your training logs, complete checkpoints, data splits, and inference
responses to make your own results reproducible.

</details>
