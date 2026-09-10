# Kimi K2.5: verified training reference

Reference for the release hardware gate, not a validated public-PyPI recipe.
The YAML matches the successful sap4 run on 2026-09-10; only four local paths
were replaced. No runtime code, dependency pins or release workflow is changed.

## Verified scope

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
were shared with an existing environment. It does not validate the wheels built
by this PR, a clean installation, or a portable multi-ISA KT wheel.

| Component | Tested source base + candidate changes |
| --- | --- |
| KT | `63d06ff6f634bb0be5228d9c3051b23f2fbcd5b8`; previously validated AMX binary |
| Transformers-KT | `96fa9fd336dfd088df4404e2529bf9be22e71309` + legacy Kimi checkpoint / MoonViT compatibility |
| Accelerate-KT | Tree identical to main `df853a7da9f28a2b95ad2c8bfcc913e7171b4b10` |
| LLaMA-Factory | `5226b026949a6f363f121a08a79927d694b37f0e` + `kimi_k25_nothink` template |
| SGLang-KT (reload) | `3424f35d0e60f03b9e75d9b90941f8f4cb944524` + RAWINT4/composite LoRA, Kimi MLA and static-adapter warmup compatibility |

The candidate changes are prerequisites, not implied by the base SHAs or package
version numbers. They must land and be pinned before advertising a public-package
recipe. In particular, this reference does **not** enable CPU retain / GPU
recompute: the tested RAWINT4 implementation rejects that combination.
The reload stack also packages KT's adapter converter as `kt-convert-lora`.
These checks use Python 3.11 / Torch 2.9.1+cu128, independently of the release
workflow's Python 3.12 build environment.

## Run with the tested stack

Set the model, expert-weight, dataset and new output paths in `train.yaml`.
Use the original Kimi-K2.5 checkpoint with RAWINT4 routed experts. In `dataset_dir`,
provide `neko_smoke.json` containing records with `prompt` and `answer`, and this
`dataset_info.json`:

```json
{
  "neko_smoke": {
    "file_name": "neko_smoke.json",
    "columns": {"prompt": "prompt", "response": "answer"}
  }
}
```

From this repository root:

```bash
USE_KT=1 accelerate launch \
  --config_file .github/release/examples/kimi-k25/fsdp2_8gpu.yaml \
  --main_process_port 29761 --no_python llamafactory-cli train \
  .github/release/examples/kimi-k25/train.yaml
```

The saved directory must include `adapter_model.safetensors`,
`fused_expert_lora.safetensors` and `kt_adapter_manifest.json`.
