# Supported Models and Precision

Checked on: 2026-05-10 Asia/Shanghai

This page is the repository-side support matrix for current KTransformers documentation. It is based on the current `main` code paths in this repository, the local `kt-agent` support matrix, and the current SFT / inference package boundary:

- Inference entry: `kt-kernel + sglang-kt`
- Fine-tuning entry: `ktransformers[sft] + LLaMA-Factory`

Status meanings:

| Status | Meaning |
| --- | --- |
| Current | Current code entry exists and the documented interface matches the current repo. |
| Needs smoke | Code or documentation exists, but the exact model / hardware / runtime tuple should be rerun before production use. |
| Current, narrow | Current support under explicit package, hardware, attention, or model-family constraints. |
| Legacy | Depends on archived or removed paths such as `local_chat.py`, `ktransformers/server/main.py`, `balance_serve`, or old `kt_optimize_rule` workflows. |
| Do not advertise | Enum, draft, or historical code exists, but the current public entry is incomplete. |

## Inference Methods

| `--kt-method` | Current role | Status | Notes |
| --- | --- | --- | --- |
| `BF16` | Native BF16 MoE expert inference | Current | Use only for model families documented with a matching BF16 path. |
| `FP8` | Native FP8 MoE expert inference | Current | Used by DeepSeek V3/V3.2, MiniMax, Qwen, and GLM-style pages; version constraints are model-specific. |
| `FP8_PERCHANNEL` | Per-channel FP8 expert inference | Current, narrow | Mainly GLM-style per-channel FP8; do not generalize to every FP8 model. |
| `RAWINT4` | Native INT4 expert weight path | Current / Needs smoke | Used by Kimi paths. Backend behavior differs across AMX, AVX512, AVX-VNNI, and AVX2. |
| `GPTQ_INT4` | GPTQ INT4 expert inference | Needs smoke | Current inference path, but not a universal INT4 recommendation. |
| `AMXINT4` | AMX INT4 converted expert weights | Current | AMX-capable CPUs only. Older docs must be rewritten away from old server commands. |
| `AMXINT8` | AMX INT8 converted expert weights | Current | AMX-capable CPUs only. Keep inference method wording separate from SFT backend wording. |
| `MXFP4` | DeepSeek V4-Flash native MXFP4 experts | Current, narrow | DeepSeek V4-Flash specific; check GPU, attention backend, TileLang, and FlashInfer requirements. |
| `LLAMAFILE` | GGUF / llamafile CPU backend | Current, secondary | Useful for GGUF compatibility, but not the main path for new SGLang-KT model pages. |
| `MOE_INT4` / `MOE_INT8` | General MoE kernels or specialized backend paths | Advanced / Needs smoke | Do not present as the default user path without a concrete model page and runtime validation. |

## Inference Model Matrix

| Model / family | Current precision | Current entry | Status | Notes |
| --- | --- | --- | --- | --- |
| DeepSeek V4-Flash | `MXFP4` | `kt run deepseek-v4-flash` and [DeepSeek V4-Flash](../DeepSeek-V4-Flash.md) | Needs smoke | Narrow path with special `transformers`, TileLang, FlashInfer, and attention constraints. |
| DeepSeek V3.2 | `FP8` in `kt run`; `AMXINT4` in the current tutorial | `kt run deepseek-v3.2`; [DeepSeek V3.2 SGLang](../kt-kernel/deepseek-v3.2-sglang-tutorial.md) | Needs reconciliation | Keep both paths documented until the registry default and tutorial method are reconciled. |
| DeepSeek V3-0324 / R1-0528 | `AMXINT4` in `kt run` registry | `kt run deepseek-v3` / `kt run deepseek-r1` | Current / Needs docs | Current registry exists; older DeepSeek local-chat/server tutorials are legacy. |
| Kimi K2 Thinking | `RAWINT4` | `kt run kimi-k2-thinking`; [Kimi K2 Thinking Native](../kt-kernel/Kimi-K2-Thinking-Native.md) | Current / Needs smoke | Treat dynamic expert update and backend-specific write-back behavior conservatively. |
| Kimi K2.5 | `RAWINT4` | [Kimi K2.5](../Kimi-K2.5.md) | Needs smoke | Manual page exists, but it is not in the built-in `kt run` registry. |
| MiniMax M2 / M2.1 | `FP8` | `kt run m2` / `kt run m2.1`; [MiniMax M2.1](../kt-kernel/MiniMax-M2.1-Tutorial.md) | Current / Needs smoke | Built-in registry exists; tensor parallel limits are model-specific. |
| MiniMax M2.5 | `FP8` | [MiniMax M2.5](../MiniMax-M2.5.md) | Needs smoke | Manual page exists, but it is not in the built-in `kt run` registry. |
| Qwen3 / Qwen3.5 / Qwen3-Coder-Next | `BF16`, `FP8`, `GPTQ_INT4` examples | [Qwen3.5](../Qwen3.5.md), [Qwen3 Coder Next](../kt-kernel/Qwen3-Coder-Next-Tutorial.md), AVX2 docs | Needs smoke | Some Qwen3.5 docs are branch-sensitive; do not present them as generic mainline recipes without rerun. |
| GLM-5 / GLM-5.1 | `BF16`, `FP8`, `FP8_PERCHANNEL` | [GLM-5](../kt-kernel/GLM-5-Tutorial.md), [GLM-5.1](../kt-kernel/GLM-5.1-Tutorial.md) | Needs smoke | Transformer version constraints should be isolated from other model environments. |
| Ascend NPU DeepSeek/Qwen | Old server path | Chinese Ascend docs | Legacy / Needs revalidation | Do not list as current support until rewritten and revalidated. |
| ROCm / Intel xPU | Old `local_chat` path | Legacy hardware docs | Legacy | Needs a new SGLang-KT or current-entry rewrite before publication as support. |

## Fine-Tuning Backend Matrix

In SFT docs, "precision" refers to the KT MoE expert backend, not the global training mixed precision. Current LLaMA-Factory examples still use BF16 as the global training precision.

| KT backend | Actual KT method | Current entry | Status | Notes |
| --- | --- | --- | --- | --- |
| `AMXBF16` | `AMXBF16_SFT` | LLaMA-Factory Accelerate KT/FSDP2 configs | Current | Uses original BF16 expert checkpoints. |
| `AMXINT8` | `AMXINT8_SFT` | LLaMA-Factory Accelerate KT/FSDP2 configs | Current | Uses pre-converted INT8 expert weights. |
| `AMXINT4` | `AMXINT4_SFT` | LLaMA-Factory Accelerate KT/FSDP2 configs | Current / Needs smoke | Document the exact pre-converted or online weight-preparation path. |
| `AMXBF16_SkipLoRA` | `AMXBF16_SFT_SkipLoRA` | wrapper code | Advanced | Do not use as the default quick-start backend. |
| `AMXINT8_SkipLoRA` | `AMXINT8_SFT_SkipLoRA` | wrapper code | Advanced | Requires explanation of skipped LoRA backward behavior. |
| `AMXINT4_SkipLoRA` | `AMXINT4_SFT_SkipLoRA` | wrapper code | Advanced | Requires explanation of skipped LoRA backward behavior. |
| `AMXINT4_1*` / `KGroup*` | Enum-level names | partial historical code | Do not advertise | The current public SFT backend map does not expose these as maintained user choices. |

## Fine-Tuning Model Matrix

| Model / family | Current example | Recommended KT backend | Status | Notes |
| --- | --- | --- | --- | --- |
| DeepSeek V2 Lite | `deepseek_v2_lora_sft_kt.yaml` | `AMXBF16`, `AMXINT8`, `AMXINT4` | Current / Needs smoke | Supported architecture path exists; rerun exact checkpoint before production claims. |
| DeepSeek V3-0324 BF16 | `deepseek_v3_lora_sft_kt.yaml` | `AMXBF16`, `AMXINT8`, `AMXINT4` | Current / Needs smoke | Treat R1 as family support unless a dedicated R1 example is validated. |
| Qwen3-235B-A22B | `qwen3moe_lora_sft_kt.yaml` | `AMXBF16`, `AMXINT8`, `AMXINT4` | Current / Needs smoke | Strong current KT SFT target; hardware memory planning is still required. |
| Qwen3.5-397B-A17B | `qwen3_5moe_lora_sft_kt.yaml` | `AMXINT8` first, BF16/INT4 as applicable | Needs smoke | Large-memory path; rerun exact environment before broad claims. |
| Kimi K2 / K2.5 SFT | Old Kimi SFT guide | Old branch or old optimize-rule style | Legacy / Experimental | Current public SFT should not use old `kt-sft` or `kt_optimize_rule` as the main path. |
| DPO | Old DPO tutorial | Historical path | Legacy / Experimental | Not part of the current primary SFT support matrix. |

## Current Documentation Rule

When adding or updating model pages, write the support claim as an exact tuple:

```text
model family + checkpoint + kt method/backend + hardware class + serving/training entry + package/version caveat
```

Do not promote a historical tutorial to current support unless the entry path exists in the current source tree and the required runtime smoke has been recorded.
