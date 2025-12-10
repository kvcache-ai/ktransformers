<div align="center">
  <p align="center">

<picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>

</picture>

</p>
  <h3>A Flexible Framework for Experiencing Cutting-edge LLM Inference/Fine-tune Optimizations</h3>
  <strong><a href="#-overview">🎯 Overview</a> | <a href="#-kt-kernel---high-performance-inference-kernels">🚀 kt-kernel</a> | <a href="#-kt-sft---fine-tuning-framework">🎓 kt-sft</a> | <a href="#-citation">🔥 Citation</a> | <a href="https://github.com/kvcache-ai/ktransformers/issues/1582">🚀 Roadmap(2025Q4)</a>  </strong>
</div>

## 🎯 Overview

KTransformers is a research project focused on efficient inference and fine-tuning of large language models through CPU-GPU heterogeneous computing. The project has evolved into **two core modules**: [kt-kernel](./kt-kernel/) and [kt-sft](./kt-sft/).

## 🔥 Updates

* **Dec 5, 2025**: Support Native Kimi-K2-Thinking inference ([Tutorial](./doc/en/Kimi-K2-Thinking-Native.md))
* **Nov 6, 2025**: Support Kimi-K2-Thinking inference ([Tutorial](./doc/en/Kimi-K2-Thinking.md)) and fine-tune ([Tutorial](./doc/en/SFT_Installation_Guide_KimiK2.md))
* **Nov 4, 2025**: KTransformers Fine-Tuning × LLaMA-Factory Integration. ([Tutorial](./doc/en/KTransformers-Fine-Tuning_User-Guide.md))
* **Oct 27, 2025**: Support Ascend NPU. ([Tutorial](./doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md))
* **Oct 10, 2025**: Integrating into SGLang. ([Roadmap](https://github.com/sgl-project/sglang/issues/11425), [Blog](https://lmsys.org/blog/2025-10-22-KTransformers/))
* **Sept 11, 2025**: Support Qwen3-Next. ([Tutorial](./doc/en/Qwen3-Next.md))
* **Sept 05, 2025**: Support Kimi-K2-0905. ([Tutorial](./doc/en/Kimi-K2.md))
* **July 26, 2025**: Support SmallThinker and GLM4-MoE. ([Tutorial](./doc/en/SmallThinker_and_Glm4moe.md))
* **July 11, 2025**: Support Kimi-K2. ([Tutorial](./doc/en/Kimi-K2.md))
* **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) [prefix cache](./doc/en/prefix_cache.md) reuse.
* **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./doc/en/xpu.md)).
* **Apr 29, 2025**: Support AMX-Int8、 AMX-BF16 and Qwen3MoE ([Tutorial](./doc/en/AMX.md))
* **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./doc/en/llama4.md)).
* **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./doc/en/balance-serve.md)).
* **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./doc/en/ROCm.md)).
* **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./doc/en/fp8_kernel.md) weights. Support 139K [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022--v023-longer-context--fp8-kernel) for DeepSeek-V3 and R1 in 24GB VRAM.
* **Feb 25, 2025**: Support [FP8 GPU kernel](./doc/en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./doc/en/DeepseekR1_V3_tutorial.md#v022-longer-context).
* **Feb 15, 2025**: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%, up to 16 Tokens/s), update [docs](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).
* **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. For detailed show case and reproduction tutorial, see [here](./doc/en/DeepseekR1_V3_tutorial.md).
* **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
* **Aug 15, 2024**: Update detailed [tutorial](doc/en/injection_tutorial.md) for injection and multi-GPU.
* **Aug 14, 2024**: Support llamfile as linear backend.
* **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
* **Aug 9, 2024**: Support windows native.

---

## 📦 Core Modules

### 🚀 [kt-kernel](./kt-kernel/) - High-Performance Inference Kernels

CPU-optimized kernel operations for heterogeneous LLM inference.

<img width="1049" height="593" alt="image" src="https://github.com/user-attachments/assets/68f423da-3f55-4025-bdc9-9ceaa554f00b" />


**Key Features:**
- **AMX/AVX Acceleration**: Intel AMX and AVX512/AVX2 optimized kernels for INT4/INT8 quantized inference
- **MoE Optimization**: Efficient Mixture-of-Experts inference with NUMA-aware memory management
- **Quantization Support**: CPU-side INT4/INT8 quantized weights, GPU-side GPTQ support
- **Easy Integration**: Clean Python API for SGLang and other frameworks

**Quick Start:**
```bash
cd kt-kernel
pip install .
```

**Use Cases:**

- CPU-GPU hybrid inference for large MoE models
- Integration with SGLang for production serving
- Heterogeneous expert placement (hot experts on GPU, cold experts on CPU)

**Performance Examples:**
| Model | Hardware Configuration | Total Throughput | Output Throughput |
|-------|------------------------|------------------|-------------------|
| DeepSeek-R1-0528 (FP8) | 8×L20 GPU + Xeon Gold 6454S | 227.85 tokens/s | 87.58 tokens/s (8-way concurrency) |

👉 **[Full Documentation →](./kt-kernel/README.md)**

---

### 🎓 [kt-sft](./kt-sft/) - Fine-Tuning Framework

KTransformers × LLaMA-Factory integration for ultra-large MoE model fine-tuning.

![image-20251011010558909](./doc/assets/image-20251011010558909.png)

**Key Features:**

- **Resource Efficient**: Fine-tune 671B DeepSeek-V3 with just **70GB GPU memory** + 1.3TB RAM
- **LoRA Support**: Full LoRA fine-tuning with heterogeneous acceleration
- **LLaMA-Factory Integration**: Seamless integration with popular fine-tuning framework
- **Production Ready**: Chat, batch inference, and metrics evaluation

**Performance Examples:**

| Model | Configuration | Throughput | GPU Memory |
|-------|--------------|------------|------------|
| DeepSeek-V3 (671B) | LoRA + AMX | ~40 tokens/s | 70GB (multi-GPU) |
| DeepSeek-V2-Lite (14B) | LoRA + AMX | ~530 tokens/s | 6GB |

**Quick Start:**
```bash
cd kt-sft
# Install environment following kt-sft/README.md
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml
```

👉 **[Full Documentation →](./kt-sft/README.md)**

---

## 🔥 Citation

If you use KTransformers in your research, please cite our paper:

```bibtex
@inproceedings{10.1145/3731569.3764843,
  title = {KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE Models},
  author = {Chen, Hongtao and Xie, Weiyu and Zhang, Boxin and Tang, Jingqi and Wang, Jiahao and Dong, Jianwei and Chen, Shaoyuan and Yuan, Ziwei and Lin, Chen and Qiu, Chengyu and Zhu, Yuening and Ou, Qingliang and Liao, Jiaqi and Chen, Xianglin and Ai, Zhiyuan and Wu, Yongwei and Zhang, Mingxing},
  booktitle = {Proceedings of the ACM SIGOPS 31st Symposium on Operating Systems Principles},
  year = {2025}
}
```

## 👥 Contributors & Team

Developed and maintained by:
- [MADSys Lab](https://madsys.cs.tsinghua.edu.cn/) @ Tsinghua University
- [Approaching.AI](http://approaching.ai/)
- [9#AISoft](https://github.com/aisoft9)
- Community contributors

We welcome contributions! Please feel free to submit issues and pull requests.

## 💬 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/kvcache-ai/ktransformers/issues)
- **WeChat Group**: See [archive/WeChatGroup.png](./archive/WeChatGroup.png)

## 📦 KT original Code

The original integrated KTransformers framework has been archived to the [`archive/`](./archive/) directory for reference. The project now focuses on the two core modules above for better modularity and maintainability.

For the original documentation with full quick-start guides and examples, see:
- [archive/README.md](./archive/README.md) (English)
- [archive/README_ZH.md](./archive/README_ZH.md) (中文)
