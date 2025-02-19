<header>
  <div align="center">
    <img alt="KTransformers" src="https://github.com/user-attachments/assets/d5a2492f-a415-4456-af99-4ab102f13f8b" width=50%>
    <h3>A Framework for Bleeding-edge LLM Inference Optimization</h3>
  </div>
</header>

[🌟 Show Cases](#show-cases) | [🚀 Quick Start](#quick-start) | [📃 Tutorial](#tutorial) | [💬 Discussion](https://github.com/kvcache-ai/ktransformers/discussions) | [🙋 FAQ](#FAQ)


## 🎉 Introduction

KTransformers, pronounced as Quick Transformers, enhances [🤗 Transformers](https://github.com/huggingface/transformers) with advanced kernel optimizations, parallelism, and placement strategies.

It includes a Transformers compatible interface, RESTful APIs compatible with OpenAI and Ollama schema, and a simple ChatGPT-inspired web client.

KTransformers aims to provide a versatile platform for experimenting with novel LLM inference optimizations. Please contact us or open an issue if you request any additional features.

## 🔥 Updates

- **Feb 15, 2025**: __KTransformers V0.2.1__ Extended context length (from 4K to 8K for 24GB VRAM) and increased inference speed (15% improvement, up to 16 tokens/sec). Updated documentation is available [here](./doc/en/DeepseekR1_V3_tutorial.md) and in the [KTransformer book](https://kvcache-ai.github.io/ktransformers/).
- **Feb 10, 2025**: Support for Deepseek-R1 and V3 on single (24GB VRAM) and multi-GPU systems, as well as 382GB DRAM, achieving a 3~28x speedup. Detailed showcase and reproduction tutorial [here](./doc/en/DeepseekR1_V3_tutorial.md).
- **Aug 28, 2024**: Reduced VRAM requirement for DeepseekV2 from 21GB to 11GB.
- **Aug 15, 2024**: Updated [tutorial](doc/en/injection_tutorial.md) with injection and multi-GPU usage.
- **Aug 14, 2024**: Introduced LlamaFile as a linear backend.
- **Aug 12, 2024**: Enabled multi-GPU inference and introduced new models: Mixtral 8x7B and 8x22B; support added for q2k, q3k, q5k quants on GPUs.
- **Aug 9, 2024**:  Native Windows support added.

## <h2 id="show-cases">🌟 Show Cases</h2>

### O1-level Local VSCode Copilot with only 24GB VRAM

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

- **[NEW!]** Local 671B DeepSeek-Coder-V3/R1: Runs its Q4_K_M version using just 14GB VRAM and 382GB DRAM ([Tutorial](./doc/en/DeepseekR1_V3_tutorial.md)).
  - **Prefill Speed (tokens/sec):**
    - KTransformers: 54.21 (32 cores) → 74.36 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
    - Compared to 10.31 tokens/sec in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
  - **Decode Speed (tokens/sec):**
    - KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
    - Compared to 4.51 tokens/sec in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.
  - **Upcoming Open Source Release:**
    - AMX optimizations and selective expert activation will be open-sourced in V0.3.
    - Currently available only in a preview binary distribution, which can be downloaded [here](./doc/en/DeepseekR1_V3_tutorial.md).

- **Local 236B DeepSeek-Coder-V2:** Runs at Q4_K_M with 21GB VRAM and 136GB DRAM, suitable for desktop PCs, outperforming GPT4-0613 in [BigCodeBench](https://huggingface.co/blog/leaderboard-bigcodebench).

![DeepSeek-Coder-V2 Score](https://github.com/user-attachments/assets/d052924e-8631-44de-aad2-97c54b965693)

- **Faster Inference:** Achieves 126 tokens/sec for 2K prompt prefill and 13.6 tokens/sec for generation using MoE offloading and use of optimized kernels from [LlamaFile](https://github.com/Mozilla-Ocho/llamafile/) and [Marlin](https://github.com/IST-DASLab/marlin).
- **VSCode Integration:** Features an OpenAI and Ollama-compatible API for seamless integration as a backend for [Tabby](https://github.com/TabbyML/tabby) and various other frontends.

<h2 id="quick-start">🚀 Quick Start</h2>


Getting started with KTransformers is simple! Follow the steps below to set up and start using it.

### 📥 Installation

To install KTransformers, follow the official [Installation Guide](https://kvcache-ai.github.io/ktransformers/).

<h2 id="quick-start">🚀 Quick Start</h2>

At the core of KTransformers is a user-friendly, template-based injection framework. This allows researchers to effortlessly replace original torch modules with optimized variants. It also simplifies the process of combining multiple optimizations to explore their synergistic effects.

![Inject-Structure](https://github.com/user-attachments/assets/6b4c1e54-9f6d-45c5-a3fc-8fa45e7d257e)

Considering vLLM already serves as an excellent framework for large-scale deployment optimizations, KTransformers primarily focuses on local deployments constrained by limited resources. We pay special attention to heterogeneous computing opportunities, like GPU/CPU offloading of quantized models. For example, we support the efficient [LlamaFile](https://github.com/Mozilla-Ocho/llamafile/) and [Marlin](https://github.com/IST-DASLab/marlin) kernels for both the CPU and GPU. More details can be found [here](doc/en/operators/llamafile.md).

### Example Usage

To use the provided kernels, users only need to create a YAML-based injection template and add the call to `optimize_and_load_gguf` before using the Transformers model.

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
...
generated = prefill_and_generate(model, tokenizer, input_tensor.cuda(), max_new_tokens=1000)
```

In this example, the AutoModel is first initialized on the meta device to avoid taking any memory. Then, `optimize_and_load_gguf` iterates through all sub-modules, matcheing the rules specified, and replacing them with the optimized modules.

After injection, the original `generate` interface is available, but we also provide a compatible `prefill_and_generate` method, which enables further optimizations like CUDAGraph to improve inference speed.

### How to Customize Your Model

A detailed tutorial of the injection and multi-GPU using DeepSeek-V2 as an example is given [here](doc/en/injection_tutorial.md).

Below is an example of a YAML template for replacing all original Linear modules with Marlin, an advanced 4-bit quantization kernel.

```yaml
- match:
    name: "^model\\.layers\\..*$"  # regular expression
    class: torch.nn.Linear  # only match modules matching name and class simultaneously
  replace:
    class: ktransformers.operators.linear.KTransformerLinear  # optimized Kernel on quantized data types
    device: "cpu"   # which devices to load this module when initializing
    kwargs:
      generate_device: "cuda"
      generate_linear_type: "QuantizedLinearMarlin"
```

Each rule in the YAML file has two parts: `match` and `replace`. The `match` part specifies which module should be replaced, and the `replace` part specifies the module to be injected into the model along with the initialization keywords.

You can find example rule templates for optimizing DeepSeek-V2 and Qwen2-57B-A14, two SOTA MoE models, in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory. These templates are used to power the `local_chat.py` demo.

If you are interested in our design principles and the implementation of the injection framework, please refer to the [design document](doc/en/deepseek-v2-injection.md).

## 🙋 FAQ

Some common questions are answered in the [FAQ](doc/en/FAQ.md)

## Discussion

If you have any questions, feel free to open an issue. Alternatively, you can join our WeChat group for further discussion. QR Code: [WeChat Group](WeChatGroup.png)

## Acknowledgments and Contributors

KTransformers builds upon the flexible and versatile framework provided by 🤗 Transformers. We have also benefited from advanced kernels such as GGUF/GGML, LlamaFile, Marlin, Sglang, and FlashInfer. We plan to contribute back to the community by upstreaming our modifications.

KTransformers is actively maintained and developed by contributors from the [MADSys group](https://madsys.cs.tsinghua.edu.cn/) at Tsinghua University, along with members from [Approaching.AI](https://approaching.ai/). We welcome new contributors to join us in making KTransformers faster and easier to use.
