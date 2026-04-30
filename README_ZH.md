<div align="center">
  <p align="center">

<picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>

</picture>

</p>
  <h3>一个用于体验尖端 LLM 推理/微调优化的灵活框架</h3>
  <strong><a href="#-概览">🎯 概览</a> | <a href="#-推理---kt-kernel-高性能推理">🚀 推理</a> | <a href="#-sft---llama-factory-微调">🎓 SFT</a> | <a href="#-引用">🔥 引用</a> </strong>
</div>

## 🎯 概览

KTransformers 是一个专注于通过 CPU-GPU 异构计算实现大语言模型高效推理和微调的研究项目。目前两个面向用户的能力都来自 kt-kernel 源码目录：[推理](./kt-kernel/README.md) 和 [SFT](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md)。

## 🔥 更新

* **2026 年 4 月 30 日**：KTransformers v0.6.1 更新 kt-kernel 推理和 SFT 文档，提供独立的[推理](./kt-kernel/README.md)和 [SFT Quick Start](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md)入口。
* **2025 年 12 月 5 日**：支持原生 Kimi-K2-Thinking 推理（[教程](./doc/en/kt-kernel/Kimi-K2-Thinking-Native.md)）
* **2025 年 11 月 6 日**：支持 Kimi-K2-Thinking 推理（[教程](./doc/en/Kimi-K2-Thinking.md)）和微调（[教程](./doc/en/SFT_Installation_Guide_KimiK2.md)）
* **2025 年 11 月 4 日**：KTransformers 微调 × LLaMA-Factory 集成（[教程](./doc/en/SFT/KTransformers-Fine-Tuning_User-Guide.md)）
* **2025 年 10 月 27 日**：支持昇腾 NPU（[教程](./doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md)）
* **2025 年 10 月 10 日**：集成到 SGLang（[路线图](https://github.com/sgl-project/sglang/issues/11425)，[博客](https://lmsys.org/blog/2025-10-22-KTransformers/)）
* **2025 年 9 月 11 日**：支持 Qwen3-Next（[教程](./doc/en/Qwen3-Next.md)）
* **2025 年 9 月 5 日**：支持 Kimi-K2-0905（[教程](./doc/en/Kimi-K2.md)）
* **2025 年 7 月 26 日**：支持 SmallThinker 和 GLM4-MoE（[教程](./doc/en/SmallThinker_and_Glm4moe.md)）
* **2025 年 7 月 11 日**：支持 Kimi-K2（[教程](./doc/en/Kimi-K2.md)）
* **2025 年 6 月 30 日**：支持 3 层（GPU-CPU-磁盘）[前缀缓存](./doc/en/prefix_cache.md)复用
* **2025 年 5 月 14 日**：支持 Intel Arc GPU（[教程](./doc/en/xpu.md)）
* **2025 年 4 月 29 日**：支持 AMX-Int8、AMX-BF16 和 Qwen3MoE（[教程](./doc/en/AMX.md)）
* **2025 年 4 月 9 日**：实验性支持 LLaMA 4 模型（[教程](./doc/en/llama4.md)）
* **2025 年 4 月 2 日**：支持多并发（[教程](./doc/en/balance-serve.md)）
* **2025 年 3 月 15 日**：支持 AMD GPU 上的 ROCm（[教程](./doc/en/ROCm.md)）
* **2025 年 3 月 5 日**：支持 unsloth 1.58/2.51 位权重和 [IQ1_S/FP8 混合](./doc/en/fp8_kernel.md)权重。在 24GB VRAM 中支持 DeepSeek-V3 和 R1 的 139K [更长上下文](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel)
* **2025 年 2 月 25 日**：为 DeepSeek-V3 和 R1 支持 [FP8 GPU 内核](./doc/en/fp8_kernel.md)；[更长上下文](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context)
* **2025 年 2 月 15 日**：更长上下文（24GB VRAM 从 4K 到 8K）& 速度稍快（+15%，最高 16 Tokens/s），更新[文档](./doc/en/DeepseekR1_V3_tutorial.md)和[在线手册](https://kvcache-ai.github.io/ktransformers/)
* **2025 年 2 月 10 日**：支持 Deepseek-R1 和 V3 在单 GPU（24GB VRAM）/多 GPU 和 382GB DRAM 上运行，速度提升高达 3~28 倍。详细案例展示和复现教程请参见[这里](./doc/en/DeepseekR1_V3_tutorial.md)
* **2024 年 8 月 28 日**：将 DeepseekV2 所需的 VRAM 从 21GB 降低到 11GB
* **2024 年 8 月 15 日**：更新了关于注入和多 GPU 的详细[教程](doc/en/injection_tutorial.md)
* **2024 年 8 月 14 日**：支持 llamfile 作为线性后端
* **2024 年 8 月 12 日**：支持多 GPU；支持新模型：mixtral 8\*7B 和 8\*22B；支持 GPU 上的 q2k、q3k、q5k 去量化
* **2024 年 8 月 9 日**：支持 Windows 原生环境

---

## 📦 功能入口

### 🚀 [推理](./kt-kernel/README.md) - kt-kernel 高性能推理

用于异构 LLM 推理的 CPU 优化内核操作。

![image-20251011010558909](./doc/assets/heterogeneous_computing.png)

**主要特性：**
- **AMX/AVX 加速**：Intel AMX 和 AVX512/AVX2 优化的内核，用于 INT4/INT8 量化推理
- **MoE 优化**：高效的专家混合推理，具有 NUMA 感知内存管理
- **量化支持**：CPU 端 INT4/INT8 量化权重，GPU 端 GPTQ 支持
- **易于集成**：为 SGLang 和其他框架提供简洁的 Python API

**快速开始：**
```bash
cd kt-kernel
pip install .
```

**使用场景：**

- 大型 MoE 模型的 CPU-GPU 混合推理
- 与 SGLang 集成用于生产服务
- 异构专家放置（热专家在 GPU 上，冷专家在 CPU 上）

**性能示例：**
| 模型 | 硬件配置 | 总吞吐量 | 输出吞吐量 |
|-------|------------------------|------------------|-------------------|
| DeepSeek-R1-0528 (FP8) | 8×L20 GPU + Xeon Gold 6454S | 227.85 tokens/s | 87.58 tokens/s（8 路并发）|

👉 **[完整文档 →](./kt-kernel/README.md)**

---

### 🎓 [SFT](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md) - LLaMA-Factory 微调

KTransformers × LLaMA-Factory 集成，面向超大 MoE 模型微调。

![KTransformers SFT](./doc/assets/image-20251011010558909.png)

**主要特性:**
- **多后端支持**: CPU/GPU 混合微调，支持 INT8/INT4 量化
- **超大 MoE 支持**: 在有限 GPU 内存下微调 DeepSeek-V3/R1 等模型
- **相对 ZeRO-Offload 加速**: 在基准 MoE SFT 任务中训练速度提升 6-12 倍
- **降低 CPU 内存**: 相比上一版 KT SFT 路径，基准配置下 CPU 内存约降至 1/2
- **LLaMA-Factory 集成**: 与流行微调框架无缝集成

| 模型 | GPU 内存 | 训练速度 | 硬件 |
|-------|------------|----------------|----------|
| DeepSeek-V3 | ~80GB 总计 | 3.7 it/s | 4x RTX 4090 |
| DeepSeek-R1 | ~80GB 总计 | 3.7 it/s | 4x RTX 4090 |
| Qwen3-30B-A3B | ~24GB 总计 | 8+ it/s | 1x RTX 4090 |

**快速开始:**
```bash
cd /path/to/LLaMA-Factory
pip install -e .
pip install -r requirements/ktransformers.txt
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file examples/ktransformers/accelerate/fsdp2_kt_int8.yaml \
  src/train.py \
  examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

👉 **[Quick Start →](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md)**
👉 **[完整文档 →](./doc/en/SFT/KTransformers-Fine-Tuning_User-Guide.md)**

---

## 🔥 引用

如果您在研究中使用了 KTransformers，请引用我们的论文：

```bibtex
@inproceedings{10.1145/3731569.3764843,
  title = {KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models},
  author = {Chen, Hongtao and Xie, Weiyu and Zhang, Boxin and Tang, Jingqi and Wang, Jiahao and Dong, Jianwei and Chen, Shaoyuan and Yuan, Ziwei and Lin, Chen and Qiu, Chengyu and Zhu, Yuening and Ou, Qingliang and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  year = {2025}
}
```

## 👥 贡献者与团队

由以下团队开发和维护：
- 清华大学 [MADSys 实验室](https://madsys.cs.tsinghua.edu.cn/)
- [Approaching.AI](http://approaching.ai/)
- 社区贡献者

我们欢迎贡献！请随时提交问题和拉取请求。

## 💬 社区与支持

- **GitHub Issues**：[报告问题或请求功能](https://github.com/kvcache-ai/ktransformers/issues)
- **微信群**：请参见 [archive/WeChatGroup.png](./archive/WeChatGroup.png)

## 📦 KT原仓库

原始的集成 KTransformers 框架已归档到 [`archive/`](./archive/) 目录以供参考。该项目现在围绕 kt-kernel 源码树中的上述两个能力入口组织文档和维护。

有关原始文档以及完整的快速入门指南和示例，请参见：
- [archive/README.md](./archive/README.md)（英文）
- [archive/README_ZH.md](./archive/README_ZH.md)（中文）
