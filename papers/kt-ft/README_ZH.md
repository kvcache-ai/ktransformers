# KTransformers-FineTune: Beyond Offload with Layout-Aware and Schedule-Optimized Heterogeneous MoE Fine-Tuning

[English](README.md) · [阅读 PDF](ktransformers-finetune.pdf) · [下载 PDF](https://raw.githubusercontent.com/kvcache-ai/ktransformers/paper-ktft-v1.1/papers/kt-ft/ktransformers-finetune.pdf) · [微调 Cookbook](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/KTransformers-Fine-Tuning_Cookbook_zh.md) · [引用](#引用)

Peilin Li, Xingxing Hao, Hongtao Chen, Weiyu Xie, Yaowei Zheng, Bowen Wu, Yujie Yang, Huanming Shen, Qingliang Ou, Boxin Zhang, Jingqi Tang, Ziwei Yuan, Jianwei Dong, Dongdong Kuang, Zhangchi Feng, Jiaheng Dai, Qianrui Yang, Shaoyuan Chen, Jiahao Wang, Yaochen Han, Yuening Zhu, Jiaqi Liao, Xianglin Chen, Zhiyuan Ai, Yongwei Wu, Mingxing Zhang。

作者预印本 · GitHub 发布日期：2026 年 10 月 8 日 · 更新日期：2026 年 10 月 10 日 · 版本：`paper-ktft-v1.1`。

公开 arXiv 链接可用后将补充。通讯作者：Mingxing Zhang。

**KTransformers-FineTune（KT-FT）让超大 MoE 模型的本地微调拥有更低的显存门槛和更高的训练效率。** MoE 的稀疏激活减少了计算量，offload 范式仍需经 PCIe 搬运大量专家权重。KT-FT 让计算靠近数据：由 CPU 执行路由专家、GPU 执行注意力，形成 Co-computation 范式。但传统 PyTorch+OneDNN 实现的 CPU 计算效率不足，难以满足较长上下文的微调吞吐需求。KT-FT 通过贯穿不同设备、不同阶段的布局优化，以及应对专家负载不均的动态调度，将协同计算的潜力转化为实际训练效率。

在论文评测中，KT-FT 的微调吞吐达到所比较 offload 系统的 **2.5–21.2 倍**，并支持最长 **128K token 上下文**。配合主机 CPU 与充足内存，仅需 **16GB GPU 显存**，即可对 **671B 参数的 DeepSeek 模型进行 BF16、2K 上下文的 LoRA 微调**。

LoRA Experts 将剩余显存用于共享的模型适配路径，通过系统-算法 co-design 加快收敛并改善最终质量。**通过与 LlamaFactory 团队合作，这套能力已接入成熟的微调生态**：LlamaFactory 承接数据处理与训练流程，KT-FT 提供底层异构加速，让研究者和企业沿用熟悉的工作流，在本地完成私有数据上的大模型定制。

## 论文图件

![Offload 与 CPU-GPU 协同计算范式](assets/offload-vs-co-compute.png)

*图 1：MoE 微调中的 offload 与 CPU-GPU 协同计算范式。*

![端到端微调吞吐量](assets/end-to-end-throughput.png)

*图 8：论文中的端到端微调吞吐量。*

## 引用

完整的 26 作者 BibTeX 见 [CITATION.bib](CITATION.bib)。

```bibtex
@misc{li2026ktft,
  title = {{KTransformers-FineTune: Beyond Offload with Layout-Aware and Schedule-Optimized Heterogeneous MoE Fine-Tuning}},
  author = {Li, Peilin and Hao, Xingxing and Chen, Hongtao and Xie, Weiyu and Zheng, Yaowei and Wu, Bowen and Yang, Yujie and Shen, Huanming and Ou, Qingliang and Zhang, Boxin and Tang, Jingqi and Yuan, Ziwei and Dong, Jianwei and Kuang, Dongdong and Feng, Zhangchi and Dai, Jiaheng and Yang, Qianrui and Chen, Shaoyuan and Wang, Jiahao and Han, Yaochen and Zhu, Yuening and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  year = {2026},
  note = {Author preprint, version paper-ktft-v1.1},
  howpublished = {Author preprint, GitHub},
  url = {https://github.com/kvcache-ai/ktransformers/tree/paper-ktft-v1.1/papers/kt-ft}
}
```

PDF 的 SHA-256 见 [SHA256SUMS](SHA256SUMS)，论文版本记录见 [CHANGELOG.md](CHANGELOG.md)。
