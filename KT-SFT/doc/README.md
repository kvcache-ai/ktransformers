<div align="center">
  <!-- <h1>KTransformers</h1> -->
  <p align="center">

<picture>
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>

</picture>

</p>

</div>

<h2 id="intro">🎉 Introduction</h2>
KTransformers, pronounced as Quick Transformers, is designed to enhance your 🤗 <a href="https://github.com/huggingface/transformers">Transformers</a> experience with advanced kernel optimizations and placement/parallelism strategies.
<br/><br/>
KTransformers is a flexible, Python-centric framework designed with extensibility at its core. 
By implementing and injecting an optimized module with a single line of code, users gain access to a Transformers-compatible
interface, RESTful APIs compliant with OpenAI and Ollama, and even a simplified ChatGPT-like web UI. 
<br/><br/>
Our vision for KTransformers is to serve as a flexible platform for experimenting with innovative LLM inference optimizations. Please let us know if you need any other features.

<h2 id="Updates">🔥 Updates</h2>

* **May 14, 2025**: Support Intel Arc GPU ([Tutorial](./en/xpu.md)).
* **Apr 9, 2025**: Experimental support for LLaMA 4 models ([Tutorial](./en/llama4.md)).
* **Apr 2, 2025**: Support Multi-concurrency. ([Tutorial](./en/balance-serve.md)).
* **Mar 27, 2025**: Support Multi-concurrency.
* **Mar 15, 2025**: Support ROCm on AMD GPU ([Tutorial](./en/ROCm.md)).
* **Mar 5, 2025**: Support unsloth 1.58/2.51 bits weights and [IQ1_S/FP8 hybrid](./en/fp8_kernel.md) weights. Support 139K [Longer Context](./en/DeepseekR1_V3_tutorial.md#v022-longer-context) for DeepSeek-V3 and R1 in 24GB VRAM.
* **Feb 25, 2025**: Support [FP8 GPU kernel](./en/fp8_kernel.md) for DeepSeek-V3 and R1; [Longer Context](./en/DeepseekR1_V3_tutorial.md#v022-longer-context).
* **Feb 10, 2025**: Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup. The detailed tutorial is [here](./en/DeepseekR1_V3_tutorial.md).
* **Aug 28, 2024**: Support 1M context under the InternLM2.5-7B-Chat-1M model, utilizing 24GB of VRAM and 150GB of DRAM. The detailed tutorial is [here](./en/long_context_tutorial.md).
* **Aug 28, 2024**: Decrease DeepseekV2's required VRAM from 21G to 11G.
* **Aug 15, 2024**: Update detailed [TUTORIAL](./en/injection_tutorial.md) for injection and multi-GPU.
* **Aug 14, 2024**: Support llamfile as linear backend.
* **Aug 12, 2024**: Support multiple GPU; Support new model: mixtral 8\*7B  and 8\*22B; Support q2k, q3k, q5k dequant on gpu.
* **Aug 9, 2024**: Support windows native.
