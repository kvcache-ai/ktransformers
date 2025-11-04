<!-- omit in toc -->

# GPT-4/o1-level Local VSCode Copilot on a Desktop with only 24GB VRAM

- [SUMMARY](#summary)
  - [Show Case Environment](#show-case-environment)
  - [Bench Result](#bench-result)
    - [V0.2.1](#v021)
      - [Memory consumption:](#memory-consumption)
      - [Change Log](#change-log)
      - [Benchmark Results](#benchmark-results)
    - [V0.2](#v02)
      - [Settings](#settings)
      - [Memory consumption:](#memory-consumption-1)
      - [Benchmark Results](#benchmark-results-1)
    - [V0.3-Preview](#v03-preview)
      - [Settings](#settings-1)
      - [Memory consumptions:](#memory-consumptions)
      - [Benchmark results](#benchmark-results-2)
  - [How to Run](#how-to-run)
    - [v0.2.2 \& v0.2.3 longer context \& FP8 kernel](#v022--v023-longer-context--fp8-kernel)
      - [longer context](#longer-context)
      - [FP8 kernel](#fp8-kernel)
    - [V0.2 \& V0.2.1 Showcase](#v02--v021-showcase)
      - [Single socket version (32 cores)](#single-socket-version-32-cores)
      - [Dual socket version (64 cores)](#dual-socket-version-64-cores)
    - [V0.3 Showcase](#v03-showcase)
      - [Dual socket version (64 cores)](#dual-socket-version-64-cores-1)
  - [Some Explanations](#some-explanations)
  - [Next](#next)
    - [Faster](#faster)
    - [Easier](#easier)
  - [FAQ](#faq)
    - [R1 No Thinking](#r1-no-thinking)
    - [More FAQ](#more-faq)

# SUMMARY

> **Feb 10, 2025**: Support DeepseekR1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, up to 3~28x speedup.<br>

Hi, we're the KTransformers team (formerly known for our local CPU/GPU hybrid inference open source project with DeepSeek-V2).

We've heard your requests for DeepSeek-R1/V3 support—and we're excited to finally deliver!
Apologies for the wait, but we've been cooking up something truly amazing!

Today, we're proud to announce that we not only support DeepSeek-R1/V3, as showcased in the video below:

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

</p>

- **[NEW!!!] Local 671B DeepSeek-Coder-V3/R1:** Running its Q4_K_M version using only 14GB VRAM and 382GB DRAM.
  - Prefill Speed (tokens/s):
    - KTransformers: 54.21 (32 cores) → 74.362 (dual-socket, 2×32 cores) → 255.26 (optimized AMX-based MoE kernel, V0.3 only) → 286.55 (selectively using 6 experts, V0.3 only)
    - Compared to 10.31 tokens/s in llama.cpp with 2×32 cores, achieving up to **27.79× speedup**.
  - Decode Speed (tokens/s):
    - KTransformers: 8.73 (32 cores) → 11.26 (dual-socket, 2×32 cores) → 13.69 (selectively using 6 experts, V0.3 only)
    - Compared to 4.51 tokens/s in llama.cpp with 2×32 cores, achieving up to **3.03× speedup**.

We also give our upcoming optimizations previews, including an Intel AMX-accelerated kernel and a selective expert activation method, which will significantly enhance performance. With V0.3-preview, we achieve up to 286 tokens/s for prefill, making it up to **28× faster than llama.cpp** for local inference.
The binary distribution is available now and the source code will come ASAP! Check out the wheel package [here](https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl)

> **Feb 15, 2025**: KTransformers V0.2.1: Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed （+15%) (Up to 16 Tokens/s), update docs [here](./doc/en/DeepseekR1_V3_tutorial.md) and [online books](https://kvcache-ai.github.io/ktransformers/).

We speed up the decode and prefill speed a littlt bit. The reason for the limited performance improvement mainly lies in the fact that the inference process is still constrained by the CPU's computational speed and memory bandwidth. The MLA part handled by the GPU accounts for a relatively small proportion.

Besides the improvements in speed, we've also significantly updated the documentation to enhance usability, including:<br>

- Added Multi-GPU configuration tutorial.
- Consolidated installation guide.
- Add a detailed tutorial on registering extra GPU memory with ExpertMarlin;

## Show Case Environment

We run our best performance tests (V0.2) on <br>
CPU: Intel (R) Xeon (R) Gold 6454S 1T DRAM (2 NUMA nodes) <br>
GPU: 4090D 24G VRAM <br>
Memory: standard DDR5-4800 server DRAM (1 TB), each socket with 8×DDR5-4800

## Bench Result

### V0.2.1

- Model: DeepseekV3-q4km (int4)<br>
- CPU: cpu_model_name: Intel (R) Xeon (R) Gold 6454S, 32 cores per socket, 2 sockets, 2 numa nodes
- GPU: 4090 24G VRAM
- We test after enough warm up

#### Memory consumption:

- Single socket: 382G DRAM, at least 14GB VRAM
- Dual socket: 1T DRAM, at least 14GB VRAM

#### Change Log

- Longer Context (from 4K to 8K for 24GB VRAM) and Slightly Faster Speed （+15%):<br>
  Integrated the highly efficient Triton MLA Kernel from the fantastic sglang project, enable much longer context length and slightly faster prefill/decode speed
- We suspect that some of the improvements come from the change of hardware platform (4090D->4090)

#### Benchmark Results

"6 experts" case is part of V0.3's preview


| Prompt               | hi (2)   | 1K (969)  | 2K (1930) | 4K (3846)               | 8K (7678) |
| -------------------- | -------- | --------- | --------- | ----------------------- | --------- |
| Output length        | 10tokens | 300tokens | 300tokens | 300tokens               | 300tokens |
| **6 experts V0.2.0** |          |           |           |                         |           |
| Prefill token/s      | 13       | 105       | 102       | 88                      | CUDA OOM  |
| decode token/s       | 16.8     | 15.4      | 14.2      | 13.0                    | CUDA OOM  |
| **6 experts V0.2.1** |          |           |           |                         |           |
| Prefill token/s      | 13       | 111       | 112.5     | 102**(1.16x speedup)**  | 101       |
| decode token/s       | 16.8     | 15.9      | 15.4      | 14.9**(1.15x speedup)** | 13.9      |
| **8 experts V0.2.1** |          |           |           |                         |           |
| Prefill token/s      | 12.2     | 88.2      | 88.5      | 81.9                    | 80        |
| Decode token/s       | 13.4     | 13.5      | 13.4      | 13.2                    | 12.4      |

### V0.2

#### Settings

- Model: DeepseekV3-q4km (int4)<br>
- CPU: cpu_model_name: Intel (R) Xeon (R) Gold 6454S, 32 cores per socket, 2 sockets, 2 numa nodes
- GPU: 4090D 24G VRAM
- We test after enough warm up

#### Memory consumption:

- Single socket: 382G DRAM, at least 14GB VRAM
- Dual socket: 1T DRAM, at least 14GB VRAM

#### Benchmark Results

"6 experts" case is part of V0.3's preview


| Prompt<br>(500 tokens) | Dual socket Ktrans (6 experts) | Dual socket Ktrans (8 experts) | Single socket Ktrans (6 experts) | Single socket Ktrans (8 experts) | llama.cpp (8 experts) |
| ---------------------- | ------------------------------ | ------------------------------ | -------------------------------- | -------------------------------- | --------------------- |
| Prefill token/s        | 97.32                          | 82.94                          | 65.14                            | 54.21                            | 10.31                 |
| Decode token/s         | 13.69                          | 12.208                         | 10.303                           | 8.73                             | 4.51                  |

**The highest speedup reaches up to <u>3.03x</u> in decoding and <u>9.44x</u> in prefill.**

### V0.3-Preview

#### Settings

- Model: DeepseekV3-BF16 (online quant into int8 for CPU and int4 for GPU)
- CPU: cpu_model_name: Intel (R) Xeon (R) Gold 6454S, 32 cores per socket, 2 socket, 2 numa nodes
- GPU: (1~4)x 4090D 24GVRAM (requires more VRAM for longer prompt)

#### Memory consumptions:

- 644GB DRAM, at least 14GB VRAM

#### Benchmark results


| Prompt length                      | 1K     | 2K     | 4K     | 8K     |
| ---------------------------------- | ------ | ------ | ------ | ------ |
| KTrans (8 experts) Prefill token/s | 185.96 | 255.26 | 252.58 | 195.62 |
| KTrans (6 experts) Prefill token/s | 203.70 | 286.55 | 271.08 | 207.20 |

**The prefill of KTrans V0.3 is up to <u>3.45x</u> times faster than KTrans V0.2, and is up to <u>27.79x</u> times faster than llama.cpp.**
**The decoding speed is the same as KTrans V0.2 (6 experts version) so it is omitted**

The main acceleration comes from

- Intel AMX instruction set and our specially designed cache friendly memory layout
- Expert selection strategy that selects fewer experts based on offline profile results of out of domain data

*From our research on DeepSeekV2, DeepSeekV3 and DeepSeekR1,
when we slightly decrease the activation experts num in inference,
the output quality doesn't change. But the speed of decoding and prefill
is speed up which is inspiring. So our showcase makes use of this finding*

## How to Run

### v0.2.4 
We provide a server script, which supports multi-concurrency functionality in version v0.2.4.

```
python ktransformers/server/main.py --model_path /mnt/data/models/DeepSeek-V3 --gguf_path /mnt/data/models/DeepSeek-V3-GGUF/DeepSeek-V3-Q4_K_M/ --cpu_infer 62 --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml --port 10002 --chunk_size 256 --max_new_tokens 1024 --max_batch_size 4 --port 10002 --cache_lens 32768 --backend_type balance_serve
```
It features the following arguments:

- `--chunk_size`: Maximum number of tokens processed in a single run by the engine.
- `--cache_lens`: Total length of kvcache allocated by the scheduler. All requests share a kvcache space corresponding to 32768 tokens, and the space occupied will be released after the requests are completed.
- `--backend_type`: `balance_serve` is a multi-concurrency backend engine introduced in version v0.2.4. The original single-concurrency engine is `ktransformers`.
- `--max_batch_size`: Maximum number of requests (prefill + decode) processed in a single run by the engine. (Supported only by `balance_serve`)

### v0.2.2 & v0.2.3 longer context & FP8 kernel

#### longer context

To use this feature, [install flashinfer](https://github.com/flashinfer-ai/flashinfer) first.

Note: The latest MLA kernel in FlashInfer still has a few minor issues. They are continuously fixing them on the main branch. If you are using FlashInfer, please install it from the main source code.

If you want to use long context(longer than 20K) for prefill, enable the matrix absorption MLA during the prefill phase, which will significantly reduce the size of the kv cache. Modify yaml file like this:

```
- match:
    name: "^model\\.layers\\..*\\.self_attn$"
  replace:
    class: ktransformers.operators.attention.KDeepseekV2Attention # optimized MLA implementation
    kwargs:
      generate_device: "cuda"
      prefill_device: "cuda"
      absorb_for_prefill: True # change this to True to enable long context(prefill may slower).
```

If the VRAM is still insufficient, try reducing the `chunk_size` parameter (default is 8192) to further decrease the intermediate results during chunk prefill.

#### FP8 kernel

The DeepSeek-AI team provides FP8 safetensors for DeepSeek-R1/V3 models. We achieve performance optimization through the following works:

- **FP8 GPU Kernel Integration**: FP8 linear layer acceleration kernels integrated in KTransformers
- **Hybrid Quantization Architecture**:
  - Attention and Shared-Expert modules use FP8 precision (enhances computational accuracy)
  - Experts modules retain GGML quantization (GGUF format, reside in CPU to save GPU memory)

So those who are persuing the best performance can use the FP8 linear kernel for DeepSeek-V3/R1.

The detailed guide is [here](./fp8_kernel.md).

### V0.2 & V0.2.1 Showcase

#### Single socket version (32 cores)

Our local_chat test command is:

```shell
numactl -N 1 -m 1 python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 33 --max_new_tokens 1000
<when you see chat, then press enter to load the text prompt_file>
```

`<your model path>` can be local or set from online huggingface like deepseek-ai/DeepSeek-V3. If online encounters connection problem, try use mirror (hf-mirror.com) <br>
`<your gguf path>` can also be online, but as its large we recommend you download it and quantize the model to what you want (notice it's the dir path) <br>
`--max_new_tokens 1000` is the max output token length. If you find the answer is truncated, you
can increase the number for longer answer (But be aware of OOM, and increase it will slow down the generation rate.).

The command `numactl -N 1 -m 1` aims to advoid data transfer between numa nodes<br>
Attention! If you are testing R1 and it may skip thinking. So you can add arg: `--force_think true`. This is explained in [FAQ](#faq) part

#### Dual socket version (64 cores)

Make sure before you install (use install.sh or `make dev_install`), setting the env var `USE_NUMA=1` by `export USE_NUMA=1` (if already installed, reinstall it with this env var set). You may check the doc [here](./install.md) for install details. <br>

Test Command:

```shell
# ---For those who have not installed ktransformers---
# git clone https://github.com/kvcache-ai/ktransformers.git
# cd ktransformers
# git submodule init
# git submodule update
# export USE_NUMA=1
# make dev_install # or sh ./install.sh
# ----------------------------------------------------
python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 65 --max_new_tokens 1000
<when you see chat, then press enter to load the text prompt_file>
```

The parameters' meaning is the same. But As we use dual socket, we set cpu_infer to 65

### V0.3 Showcase

#### Dual socket version (64 cores)

Our local_chat test command is:

```shell
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
pip install ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
python -m ktransformers.local_chat --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 65 --max_new_tokens 1000
<when you see chat, then press enter to load the text prompt_file>
```

The parameters' meaning is the same with V0.2. But As we  use dual socket, we set cpu_infer to 65

## Some Explanations

1. Also we want to make further use of our two NUMA nodes on Xeon Gold cpu.
   To avoid the cost of data transfer between nodes, we "copy" the critical matrix on
   both nodes which takes more memory consumption but accelerates the prefill and decoding process.
   But this method takes huge memory and slow when loading weights, So be patient when loading
   and monitor the memory usage. We are going to optimize this huge memory overhead. Stay tuned~ <br>
2. The command args `--cpu_infer 65` specifies how many cores to use (it's ok that it exceeds the physical number,
   but it's not the more the better. Adjust it slightly lower to your actual number of cores)<br>
3. Why CPU/GPU Hybrid Inference?
   DeepSeek's MLA operators are highly computationally intensive. While running everything on CPU is possible, offloading the heavy computations to the GPU results in a massive performance boost.
4. Where Does the Speedup Come From?

   - Expert Offload: Unlike traditional layer-based or KVCache offloading (as seen in llama.cpp), we offload the expert computation to the CPU and MLA/KVCache to GPU, aligning perfectly with DeepSeek’s architecture for optimal efficiency.
   - Intel AMX Optimization – Our AMX-accelerated kernel is meticulously tuned, running several times faster than existing llama.cpp implementations. We plan to open-source this kernel after cleansing and are considering upstream contributions to llama.cpp.
5. Why Intel CPUs?
   Intel is currently the only CPU vendor that supports AMX-like instructions, which delivers significantly better performance compared to AVX-only alternatives.

## Next

### Faster

* The FlashInfer (https://github.com/flashinfer-ai/flashinfer) project is releasing an even more efficient fused MLA operator, promising further speedups
* vLLM has explored multi-token prediction in DeepSeek-V3, and support is on our roadmap for even better performance
* We are collaborating with Intel to enhance the AMX kernel (v0.3) and optimize for Xeon6/MRDIMM

### Easier

* Official Docker images to simplify installation
* Fix the server integration for web API access
* Fix the local chat only accepting a single line prompt (currently \n begins generating prompt)
* Support for more quantization types, including the highly requested dynamic quantization from unsloth

Stay tuned for more updates!

## FAQ

### R1 No Thinking

Attention! If you are testing R1 and it may skip thinking. So you can add arg: `--force_think true`. The detail is in [FAQ](./FAQ.md) part <br>

### More FAQ

[See detail](./FAQ.md)
