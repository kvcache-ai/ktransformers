<div align="center">
  <p align="center">
    <picture>
      <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
    </picture>
  </p>
  <h3>高性能 CPU-GPU 异构大语言模型推理</h3>
</div>

## 🎯 项目概述

KTransformers 是一个专注于大语言模型高效推理和微调的研究项目，通过 CPU-GPU 异构计算实现资源受限环境下的模型部署。项目已演进为**两个核心模块**：[kt-kernel](./kt-kernel/) 和 [KT-SFT](./KT-SFT/)。

## 🔥 更新

* **2025年11月6日**：支持 Kimi-K2-Thinking 推理和微调
* **2025年11月4日**：KTransformers 微调 × LLaMA-Factory 集成
* **2025年10月27日**：支持 Ascend NPU
* **2025年10月10日**：集成到 SGLang ([路线图](https://github.com/sgl-project/sglang/issues/11425), [博客](https://lmsys.org/blog/2025-10-22-KTransformers/))
* **2025年9月11日**：支持 Qwen3-Next
* **2025年9月5日**：支持 Kimi-K2-0905
* **2025年7月26日**：支持 SmallThinker 和 GLM4-MoE
* **2025年6月30日**：支持 3层（GPU-CPU-磁盘）前缀缓存复用
* **2025年5月14日**：支持 Intel Arc GPU
* **2025年4月29日**：支持 AMX-Int8、AMX-BF16 和 Qwen3MoE
* **2025年4月9日**：实验性支持 LLaMA 4 模型
* **2025年4月2日**：支持多并发
* **2025年3月15日**：支持 AMD GPU 的 ROCm
* **2025年3月5日**：支持 unsloth 1.58/2.51 bits 权重和 IQ1_S/FP8 混合权重；DeepSeek-V3/R1 支持 139K 长上下文
* **2025年2月25日**：支持 DeepSeek-V3 和 R1 的 FP8 GPU 内核
* **2025年2月10日**：支持 Deepseek-R1 和 V3，速度提升最高达 3~28 倍

---

## 📦 核心模块

### 🚀 [kt-kernel](./kt-kernel/) - 高性能推理内核

面向异构 LLM 推理的 CPU 优化内核操作库。

**核心特性：**
- **AMX/AVX 加速**：Intel AMX 和 AVX512/AVX2 优化内核，支持 INT4/INT8 量化推理
- **MoE 优化**：高效的专家混合推理，支持 NUMA 感知内存管理
- **量化支持**：CPU 端 INT4/INT8 量化权重，GPU 端 GPTQ 支持
- **易于集成**：简洁的 Python API，可集成到 SGLang 等框架

**快速开始：**
```bash
cd kt-kernel
pip install .
```

**应用场景：**
- 大型 MoE 模型的 CPU-GPU 混合推理
- 与 SGLang 集成用于生产服务
- 异构专家放置（热门专家在 GPU，冷门专家在 CPU）

**性能示例：**
| 模型 | 硬件配置 | 总吞吐量 | 输出吞吐量 |
|------|---------|---------|-----------|
| DeepSeek-R1-0528 (FP8) | 8×L20 GPU + Xeon Gold 6454S | 227.85 tokens/s | 87.58 tokens/s（8路并发）|

👉 **[完整文档 →](./kt-kernel/README.md)**

---

### 🎓 [KT-SFT](./KT-SFT/) - 微调框架

KTransformers × LLaMA-Factory 集成，支持超大 MoE 模型微调。

**核心特性：**
- **资源高效**：仅需 **70GB 显存** + 1.3TB 内存即可微调 671B DeepSeek-V3
- **LoRA 支持**：完整的 LoRA 微调与异构加速
- **LLaMA-Factory 集成**：与流行微调框架无缝集成
- **生产就绪**：支持对话、批量推理和指标评估

**性能示例：**
| 模型 | 配置 | 吞吐量 | GPU 显存 |
|------|------|--------|----------|
| DeepSeek-V3 (671B) | LoRA + AMX | ~40 tokens/s | 70GB (多卡) |
| DeepSeek-V2-Lite (14B) | LoRA + AMX | ~530 tokens/s | 6GB |

**快速开始：**
```bash
cd KT-SFT
# 按照 KT-SFT/README.md 安装环境
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml
```

👉 **[完整文档 →](./KT-SFT/README.md)**

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

我们欢迎贡献！请随时提交 issues 和 pull requests。

## 💬 社区与支持

- **GitHub Issues**：[报告 bug 或请求功能](https://github.com/kvcache-ai/ktransformers/issues)
- **GitHub Discussions**：[提问和分享想法](https://github.com/kvcache-ai/ktransformers/discussions)
- **微信群**：查看 [archive/WeChatGroup.png](./archive/WeChatGroup.png)

## 📦 历史代码

原完整的 KTransformers 框架代码已归档至 [`archive/`](./archive/) 目录供参考。项目现专注于上述两个核心模块，以实现更好的模块化和可维护性。

关于原始完整文档（包含快速入门指南和示例），请查看：
- [archive/README_LEGACY.md](./archive/README_LEGACY.md) (English)
- [archive/README_ZH_LEGACY.md](./archive/README_ZH_LEGACY.md) (中文)
