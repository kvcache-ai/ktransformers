# KTransformers GitHub Docs

This directory contains repository-side documentation for KTransformers. It is intended for users and developers who are already reading the source tree. The public website document is maintained separately in `ktransformers-web` and is available at <https://ktransformers.net/en/docs>.

The current documentation is organized by user task, not by historical implementation path. Older tutorials are still kept in the tree for traceability, but pages that depend on archived entry points such as `local_chat.py`, `ktransformers/server/main.py`, `balance_serve`, or old `kt_optimize_rule` workflows should be treated as legacy unless they are explicitly revalidated.

## Start Here

| Need | Current entry |
| --- | --- |
| Project overview | [Repository README](../README.md) |
| Inference quick start | [KT-Kernel README](../kt-kernel/README.md) |
| Fine-tuning quick start | [SFT Quick Start](en/SFT/KTransformers-Fine-Tuning_Quick-Start.md) |
| Model, precision, and backend support | [Supported models and precision](en/support/supported-models-and-precision.md) |
| Chinese support matrix | [模型与精度支持矩阵](zh/support/supported-models-and-precision.md) |

## Current Inference Docs

- [KT-Kernel README](../kt-kernel/README.md) - main technical entry for `kt-kernel`, `kt run`, SGLang integration, conversion, and backend details.
- [KT CLI reference](en/kt-kernel/kt-cli.md)
- [Native precision tutorial](en/kt-kernel/Native-Precision-Tutorial.md)
- [AVX2 backend tutorial](en/kt-kernel/AVX2-Tutorial.md)
- [AVX2 backend tutorial, Chinese](zh/AVX2-Tutorial_zh.md)
- [DeepSeek V4-Flash](en/DeepSeek-V4-Flash.md)
- [DeepSeek V3.2 SGLang tutorial](en/kt-kernel/deepseek-v3.2-sglang-tutorial.md)
- [Kimi K2 Thinking Native](en/kt-kernel/Kimi-K2-Thinking-Native.md)
- [MiniMax M2.1](en/kt-kernel/MiniMax-M2.1-Tutorial.md)
- [GLM-5](en/kt-kernel/GLM-5-Tutorial.md)
- [GLM-5.1](en/kt-kernel/GLM-5.1-Tutorial.md)

## Current Fine-Tuning Docs

- [SFT docs index](en/SFT/README.md)
- [SFT Quick Start](en/SFT/KTransformers-Fine-Tuning_Quick-Start.md)
- [SFT User Guide](en/SFT/KTransformers-Fine-Tuning_User-Guide.md)
- [SFT Developer Technical Notes](en/SFT/KTransformers-Fine-Tuning_Developer-Technical-Notes.md)

Current public SFT should use `ktransformers[sft]` through LLaMA-Factory. Do not revive old `kt-sft`, automatic patching, or `kt_optimize_rule` flows as the default path.

## Legacy / Needs Revalidation

The following topics are still present in this repository, but should not be linked as current quick starts without a rewrite or runtime validation:

- Old integrated framework docs using `local_chat.py`, `ktransformers/server/main.py`, `balance_serve`, or old optimize-rule paths.
- Legacy Docker, ROCm, Intel xPU, old API server, long-context, and historical benchmark pages.
- Older Kimi SFT and DPO docs that predate the current LLaMA-Factory + `ktransformers[sft]` entry.

Use the [support matrix](en/support/supported-models-and-precision.md) to decide whether a page is current, experimental, or legacy.
