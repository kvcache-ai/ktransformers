<div align="center">
  <p align="center">
    <picture>
      <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
    </picture>
  </p>
  <h3>High-Performance CPU-GPU Hybrid Inference for Large Language Models</h3>
</div>

## 🎯 Overview

KTransformers is a research project focused on efficient inference and fine-tuning of large language models through CPU-GPU heterogeneous computing. The project has evolved into **two core modules**:

## 🔥 Updates

* **Nov 6, 2025**: Support Kimi-K2-Thinking inference and fine-tune
* **Nov 4, 2025**: KTransformers Fine-Tuning × LLaMA-Factory Integration
* **Oct 27, 2025**: Support Ascend NPU
* **Oct 10, 2025**: Integrating into SGLang ([Roadmap](https://github.com/sgl-project/sglang/issues/11425))
* **Sept 11, 2025**: Support Qwen3-Next
* **Sept 05, 2025**: Support Kimi-K2-0905
* **July 26, 2025**: Support SmallThinker and GLM4-MoE
* **June 30, 2025**: Support 3-layer (GPU-CPU-Disk) prefix cache reuse
* **May 14, 2025**: Support Intel Arc GPU
* **Apr 29, 2025**: Support AMX-Int8、AMX-BF16 and Qwen3MoE
* **Apr 9, 2025**: Experimental support for LLaMA 4 models
* **Apr 2, 2025**: Support Multi-concurrency
* **Mar 15, 2025**: Support ROCm on AMD GPU
* **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and IQ1_S/FP8 hybrid weights; 139K longer context for DeepSeek-V3/R1
* **Feb 25, 2025**: Support FP8 GPU kernel for DeepSeek-V3 and R1
* **Feb 10, 2025**: Support Deepseek-R1 and V3, up to 3~28x speedup
* 
---

## 📦 Core Modules

### 🚀 [kt-kernel](./kt-kernel/) - High-Performance Inference Kernels

CPU-optimized kernel operations for heterogeneous LLM inference.

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

👉 **[Full Documentation →](./kt-kernel/README.md)**

---

### 🎓 [KT-SFT](./KT-SFT/) - Fine-Tuning Framework

KTransformers × LLaMA-Factory integration for ultra-large MoE model fine-tuning.

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
cd KT-SFT
# Install environment following KT-SFT/README.md
USE_KT=1 llamafactory-cli train examples/train_lora/deepseek3_lora_sft_kt.yaml
```

👉 **[Full Documentation →](./KT-SFT/README.md)**

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
- Community contributors

We welcome contributions! Please feel free to submit issues and pull requests.

## 💬 Community & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/kvcache-ai/ktransformers/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/kvcache-ai/ktransformers/discussions)
- **WeChat Group**: See [archive/WeChatGroup.png](./archive/WeChatGroup.png)

## 📦 Legacy Code

The original integrated KTransformers framework has been archived to the [`archive/`](./archive/) directory for reference. The project now focuses on the two core modules above for better modularity and maintainability.

For the original documentation with full quick-start guides and examples, see:
- [archive/README_LEGACY.md](./archive/README_LEGACY.md) (English)
- [archive/README_ZH_LEGACY.md](./archive/README_ZH_LEGACY.md) (中文)

