# 模型与精度支持矩阵

检查日期：2026-05-10 Asia/Shanghai

本文是 KTransformers 仓库内的当前支持矩阵，用于区分当前入口、需要复验的入口和历史入口。当前公开口径：

- 推理入口：`kt-kernel + sglang-kt`
- 微调入口：`ktransformers[sft] + LLaMA-Factory`

状态含义：

| 状态 | 含义 |
| --- | --- |
| Current | 当前代码入口存在，文档接口与仓库基本一致。 |
| Needs smoke | 代码或文档入口存在，但具体模型 / 硬件 / runtime 组合需要复跑后再作为生产支持。 |
| Current, narrow | 在明确包版本、硬件、attention backend 或模型族约束下支持。 |
| Legacy | 依赖 `local_chat.py`、`ktransformers/server/main.py`、`balance_serve`、旧 `kt_optimize_rule` 等归档或已移除路径。 |
| Do not advertise | enum、草稿或历史代码存在，但当前公开入口不完整。 |

## 推理 method

| `--kt-method` | 当前定位 | 状态 | 备注 |
| --- | --- | --- | --- |
| `BF16` | 原生 BF16 MoE expert 推理 | Current | 只应在明确支持 BF16 的模型页中使用。 |
| `FP8` | 原生 FP8 MoE expert 推理 | Current | DeepSeek V3/V3.2、MiniMax、Qwen、GLM 等页面使用，版本限制按模型列出。 |
| `FP8_PERCHANNEL` | per-channel FP8 expert 推理 | Current, narrow | 主要面向 GLM 风格 per-channel FP8，不能泛化到所有 FP8 模型。 |
| `RAWINT4` | 原生 INT4 expert 权重路径 | Current / Needs smoke | Kimi 路径使用；AMX、AVX512、AVX-VNNI、AVX2 的后端行为不同。 |
| `GPTQ_INT4` | GPTQ INT4 expert 推理 | Needs smoke | 当前推理路径存在，但不是通用 INT4 推荐。 |
| `AMXINT4` | AMX INT4 转换后 expert 权重 | Current | 仅限 AMX CPU；旧 server 命令文档需要重写。 |
| `AMXINT8` | AMX INT8 转换后 expert 权重 | Current | 仅限 AMX CPU；注意与 SFT backend 术语区分。 |
| `MXFP4` | DeepSeek V4-Flash 原生 MXFP4 experts | Current, narrow | DeepSeek V4-Flash 专用；需要核对 GPU、attention backend、TileLang、FlashInfer 条件。 |
| `LLAMAFILE` | GGUF / llamafile CPU backend | Current, secondary | 可用于 GGUF 兼容，但不是新 SGLang-KT 模型页的主推路径。 |
| `MOE_INT4` / `MOE_INT8` | 通用 MoE kernel 或特定后端 | Advanced / Needs smoke | 没有具体模型页和 runtime 验证前，不作为默认用户路径。 |

## 推理模型矩阵

| 模型/家族 | 当前精度 | 当前入口 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| DeepSeek V4-Flash | `MXFP4` | `kt run deepseek-v4-flash` 和 [DeepSeek V4-Flash](../../en/DeepSeek-V4-Flash.md) | Needs smoke | 窄路径；有特殊 `transformers`、TileLang、FlashInfer、attention 条件。 |
| DeepSeek V3.2 | `kt run` registry 中为 `FP8`；当前 tutorial 中为 `AMXINT4` | `kt run deepseek-v3.2`；[DeepSeek V3.2 SGLang](../../en/kt-kernel/deepseek-v3.2-sglang-tutorial.md) | Needs reconciliation | registry 默认和教程 method 需要统一或解释两条路径。 |
| DeepSeek V3-0324 / R1-0528 | `kt run` registry 中为 `AMXINT4` | `kt run deepseek-v3` / `kt run deepseek-r1` | Current / Needs docs | registry 存在；旧 DeepSeek local-chat/server 教程属于 legacy。 |
| Kimi K2 Thinking | `RAWINT4` | `kt run kimi-k2-thinking`；[Kimi K2 Thinking Native](../../en/kt-kernel/Kimi-K2-Thinking-Native.md) | Current / Needs smoke | 动态 expert update 和后端 write-back 行为需要保守表述。 |
| Kimi K2.5 | `RAWINT4` | [Kimi K2.5](../../en/Kimi-K2.5.md) | Needs smoke | 手动页面存在，但未列入内置 `kt run` registry。 |
| MiniMax M2 / M2.1 | `FP8` | `kt run m2` / `kt run m2.1`；[MiniMax M2.1](../../en/kt-kernel/MiniMax-M2.1-Tutorial.md) | Current / Needs smoke | registry 存在；tensor parallel 上限按模型处理。 |
| MiniMax M2.5 | `FP8` | [MiniMax M2.5](../../en/MiniMax-M2.5.md) | Needs smoke | 手动页面存在，但未列入内置 `kt run` registry。 |
| Qwen3 / Qwen3.5 / Qwen3-Coder-Next | `BF16`、`FP8`、`GPTQ_INT4` examples | [Qwen3.5](../../en/Qwen3.5.md)、[Qwen3 Coder Next](../../en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)、AVX2 docs | Needs smoke | 部分 Qwen3.5 文档仍有分支敏感内容，复跑前不要当成通用 mainline recipe。 |
| GLM-5 / GLM-5.1 | `BF16`、`FP8`、`FP8_PERCHANNEL` | [GLM-5](../../en/kt-kernel/GLM-5-Tutorial.md)、[GLM-5.1](../../en/kt-kernel/GLM-5.1-Tutorial.md) | Needs smoke | transformers 版本约束需要与其他模型环境隔离。 |
| Ascend NPU DeepSeek/Qwen | 旧 server path | 中文 Ascend 文档 | Legacy / Needs revalidation | 重写并复验前不要列为当前支持。 |
| ROCm / Intel xPU | 旧 `local_chat` path | legacy 硬件文档 | Legacy | 需要新的 SGLang-KT 或当前入口重写后才能公开。 |

## 微调 backend 矩阵

SFT 文档里的“精度”指 KT MoE expert backend，不等于训练全局 mixed precision。当前 LLaMA-Factory examples 的全局训练精度仍以 BF16 为主。

| KT backend | 实际 KT method | 当前入口 | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| `AMXBF16` | `AMXBF16_SFT` | LLaMA-Factory Accelerate KT/FSDP2 配置 | Current | 使用原始 BF16 expert checkpoints。 |
| `AMXINT8` | `AMXINT8_SFT` | LLaMA-Factory Accelerate KT/FSDP2 配置 | Current | 使用预转换 INT8 expert 权重。 |
| `AMXINT4` | `AMXINT4_SFT` | LLaMA-Factory Accelerate KT/FSDP2 配置 | Current / Needs smoke | 需要明确预转换或 online 权重准备路径。 |
| `AMXBF16_SkipLoRA` | `AMXBF16_SFT_SkipLoRA` | wrapper code | Advanced | 不作为 quick start 默认 backend。 |
| `AMXINT8_SkipLoRA` | `AMXINT8_SFT_SkipLoRA` | wrapper code | Advanced | 需要解释跳过 LoRA backward 的含义。 |
| `AMXINT4_SkipLoRA` | `AMXINT4_SFT_SkipLoRA` | wrapper code | Advanced | 需要解释跳过 LoRA backward 的含义。 |
| `AMXINT4_1*` / `KGroup*` | enum 级名称 | 部分历史代码 | Do not advertise | 当前公开 SFT backend map 不把这些作为维护中的用户选项。 |

## 微调模型矩阵

| 模型/家族 | 当前示例 | 推荐 KT backend | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| DeepSeek V2 Lite | `deepseek_v2_lora_sft_kt.yaml` | `AMXBF16`、`AMXINT8`、`AMXINT4` | Current / Needs smoke | 架构路径存在；生产口径前需复跑具体 checkpoint。 |
| DeepSeek V3-0324 BF16 | `deepseek_v3_lora_sft_kt.yaml` | `AMXBF16`、`AMXINT8`、`AMXINT4` | Current / Needs smoke | R1 只应写成家族支持，除非有独立 R1 示例验证。 |
| Qwen3-235B-A22B | `qwen3moe_lora_sft_kt.yaml` | `AMXBF16`、`AMXINT8`、`AMXINT4` | Current / Needs smoke | 当前 KT SFT 重点目标；仍需硬件内存规划。 |
| Qwen3.5-397B-A17B | `qwen3_5moe_lora_sft_kt.yaml` | 优先 `AMXINT8`，BF16/INT4 按场景 | Needs smoke | 大内存路径；泛化宣传前需复跑精确环境。 |
| Kimi K2 / K2.5 SFT | 旧 Kimi SFT guide | 旧分支或旧 optimize-rule 路线 | Legacy / Experimental | 当前公开 SFT 不应把旧 `kt-sft` 或 `kt_optimize_rule` 作为主路径。 |
| DPO | 旧 DPO tutorial | 历史路径 | Legacy / Experimental | 不属于当前主 SFT 支持矩阵。 |

## 当前文档规则

新增或更新模型页时，用精确 tuple 写支持声明：

```text
模型族 + checkpoint + kt method/backend + 硬件类别 + serving/training 入口 + 包/版本限制
```

除非当前源码存在入口且 runtime smoke 已记录，否则不要把历史教程升级成当前支持。
