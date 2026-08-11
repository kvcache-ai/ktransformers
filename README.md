<div align="center">
  <p align="center">

<picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>

</picture>

</p>
  <h3>A Flexible Framework for Experiencing Cutting-edge LLM Inference/Fine-tune Optimizations</h3>
  <strong><a href="#-overview">🎯 Overview</a> | <a href="#-inference---high-performance-kt-kernel-serving">🚀 Inference</a> | <a href="#-sft---fine-tuning-with-llama-factory">🎓 SFT</a> | <a href="#-citation">🔥 Citation</a> | <a href="https://github.com/kvcache-ai/ktransformers/issues/1921">🚀 Roadmap(2026Q2)</a>  </strong>
</div>

## 🎯 Overview

KTransformers is a research project focused on efficient inference and fine-tuning of large language models through CPU-GPU heterogeneous computing. The project now exposes two user-facing capabilities from the kt-kernel source tree: [Inference](./kt-kernel/README.md) and [SFT](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md).

## 🔥 Updates
* **June 21, 2026**: MiniMax-M3 Day0 Support! ([Tutorial](./doc/en/kt-kernel/MiniMax-M3-Tutorial.md))
* **June 17, 2026**: GLM-5.2 Day0 Support! ([Tutorial](./doc/en/kt-kernel/GLM-5.2-Tutorial.md))
* **May 6, 2026**: KTransformers at [GOSIM Paris 2026](https://paris2026.gosim.org/zh/schedule/) — "Agentic AI on Edge" track. We'll present KT's inference performance on consumer hardware.
* **May 02, 2026**: DeepSeek-V4-Flash Support! ([Tutorial](./doc/en/DeepSeek-V4-Flash.md))
* **Apr 30, 2026**: KTransformers v0.6.1 refreshes kt-kernel inference and SFT docs with separate [Inference](./kt-kernel/README.md) and [SFT Quick Start](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md) entry points.
* **Mar 26, 2026**: Support AVX2-only CPU backend for KT-Kernel inference. ([Tutorial](./doc/en/kt-kernel/AVX2-Tutorial.md))
* **Feb 13, 2026**: MiniMax-M2.5 Day0 Support! ([Tutorial](./doc/en/MiniMax-M2.5.md))
* **Feb 12, 2026**: GLM-5 Day0 Support! ([Tutorial](./doc/en/kt-kernel/GLM-5-Tutorial.md))
* **Jan 27, 2026**: Kimi-K2.5 Day0 Support! ([Tutorial](./doc/en/Kimi-K2.5.md)) ([SFT Tutorial](./doc/en/SFT_Installation_Guide_KimiK2.5.md))
* **Jan 22, 2026**: Support [CPU-GPU Expert Scheduling](./doc/en/kt-kernel/experts-sched-Tutorial.md), [Native BF16 and FP8 per channel Precision](./doc/en/kt-kernel/Native-Precision-Tutorial.md) and [AutoDL unified fine-tuning and inference](./doc/zh/【云端低价训推】%20KTransformers%2BAutoDL%2BLlamaFactory：随用随租的低成本超大模型「微调%2B推理」一体化流程.pdf)
* **Dec 24, 2025**: Support Native MiniMax-M2.1 inference. ([Tutorial](./doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md))
* **Dec 22, 2025**: Support RL-DPO fine-tuning with LLaMA-Factory. ([Tutorial](./doc/en/SFT/DPO_tutorial.md))
* **Dec 5, 2025**: Support Native Kimi-K2-Thinking inference ([Tutorial](./doc/en/kt-kernel/Kimi-K2-Thinking-Native.md))
* **Nov 6, 2025**: Support Kimi-K2-Thinking inference ([Tutorial](./doc/en/Kimi-K2-Thinking.md)) and fine-tune ([Tutorial](./doc/en/SFT_Installation_Guide_KimiK2.md))
* **Nov 4, 2025**: KTransformers Fine-Tuning × LLaMA-Factory Integration. ([Tutorial](./doc/en/SFT/KTransformers-Fine-Tuning_User-Guide.md))
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

## 📦 Capabilities

### 🚀 [Inference](./kt-kernel/README.md) - High-Performance kt-kernel Serving

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

### 🎓 [SFT](./doc/en/SFT/KTransformers-Fine-Tuning_Quick-Start.md) - Fine-Tuning with LLaMA-Factory

KTransformers × LLaMA-Factory integration for ultra-large MoE model fine-tuning.

![KTransformers SFT](https://raw.githubusercontent.com/kvcache-ai/ktransformers/main/doc/assets/image-20251011010558909.png)

**Key Features:**
- **Multi-Backend Support**: CPU/GPU hybrid fine-tuning with INT8/INT4 quantization
- **Ultra-Large MoE Support**: Fine-tune models like DeepSeek-V3/R1 on limited GPU memory
- **Faster than ZeRO-Offload**: 6-12x training speedup in benchmarked MoE SFT workloads
- **Lower CPU Memory**: About half the CPU memory of the previous KT SFT path in the benchmarked setup
- **LLaMA-Factory Integration**: Seamless integration with popular fine-tuning framework

| Model | GPU Memory | Training Speed | Hardware |
|-------|------------|----------------|----------|
| DeepSeek-V3 | ~80GB total | 3.7 it/s | 4x RTX 4090 |
| DeepSeek-R1 | ~80GB total | 3.7 it/s | 4x RTX 4090 |
| Qwen3-30B-A3B | ~24GB total | 8+ it/s | 1x RTX 4090 |

**Quick Start:**
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
👉 **[Full Documentation →](./doc/en/SFT/KTransformers-Fine-Tuning_User-Guide.md)**

#### Qwen3.5 VLM LoRA

1. **Clone the two integration repositories.** Keep both checkouts under the same parent directory so the commands below can use relative paths.

   ```bash
   git clone https://github.com/Illumination111/LlamaFactory.git
   git clone https://github.com/Illumination111/ktransformers.git
   ```

2. **Install LlamaFactory first.** This establishes the base training environment before the KT-patched packages are installed.

   ```bash
   cd LlamaFactory
   pip install -e .
   ```

3. **Install the VLM LoRA dependency set from the KTransformers root.** The dedicated requirements file pins `torch==2.9.1`, `torchvision==0.24.1`, and `torchaudio==2.9.1`; installs the local `kt-kernel[vlm-sft]` and `ktransformers[vlm-sft]`; and pulls in the verified `transformers-kt==5.6.0.post1`, `accelerate-kt==1.14.0.post1`, and `ms-swift>=4.4.2,<4.5` compatibility stack. Keep this step after the LlamaFactory installation so the KT forks are the active implementations.

   ```bash
   cd ../ktransformers
   pip install -r requirements-vlm-lora.txt
   ```

4. **Prepare the model and dataset.** Put the Qwen3.5 VLM weights at a local or Hugging Face path, register a custom multimodal dataset in `LlamaFactory/data/dataset_info.json`, and set `model_name_or_path` and `dataset` below. The built-in `mllm_demo` dataset can be used for a smoke test without additional registration.

5. **Create or edit the training YAML.** The repository already provides `examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml`; the following is a complete `all`-scope example:

   ```yaml
   ### model
   model_name_or_path: Qwen/Qwen3.5-122B-A10B
   image_max_pixels: 262144
   video_max_pixels: 16384
   trust_remote_code: true

   ### method
   stage: sft
   do_train: true
   finetuning_type: lora
   lora_rank: 8
   lora_alpha: 16
   lora_target: all
   vlm_lora_scope: all

   ### dataset
   dataset: mllm_demo
   template: qwen3_5
   cutoff_len: 512
   overwrite_cache: true
   preprocessing_num_workers: 4
   dataloader_num_workers: 1

   ### output
   output_dir: saves/KT_FT_qwen35Moe_VLM_all
   logging_steps: 1
   save_steps: 100
   plot_loss: true
   overwrite_output_dir: true
   report_to: none

   ### train
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 1
   learning_rate: 1.0e-4
   num_train_epochs: 3.0
   lr_scheduler_type: cosine
   warmup_ratio: 0.1
   bf16: true
   gradient_checkpointing: true
   gradient_checkpointing_kwargs: {use_reentrant: false}
   ddp_timeout: 360000000

   ### ktransformers
   use_kt: true
   ```

   Set `vlm_lora_scope` to `text`, `vision`, or `all` to train language modules, vision/projector modules, or both. Keep `lora_target: all` for every non-default VLM scope. Matching ready-to-run YAML files for all three scopes are under `LlamaFactory/examples/ktransformers/train_lora/`.

6. **Launch training from the LlamaFactory root and check the configured output directory.**

   ```bash
   cd ../LlamaFactory
   LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 \
     llamafactory-cli train \
     examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml
   ```

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

The original integrated KTransformers framework has been archived to the [`archive/`](./archive/) directory for reference. The project now organizes the two capabilities above from the kt-kernel source tree for clearer documentation and maintenance.

For the original documentation with full quick-start guides and examples, see:
- [archive/README.md](./archive/README.md) (English)
- [archive/README_ZH.md](./archive/README_ZH.md) (中文)
