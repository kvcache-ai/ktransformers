<!-- omit in toc -->

# GPT-4/o1 级别本地 VSCode Copilot 在仅 24GB 显存的台式机上的表现

- [摘要](#摘要)
  - [先决条件](#先决条件)
  - [基准测试结果](#基准测试结果)
    - [V0.2](#v02)
      - [设置](#设置)
      - [内存占用](#内存占用)
      - [基准测试结果](#基准测试结果)
    - [V0.3-Preview](#V0.3-Preview)
      - [设置](#设置-1)
      - [内存占用](#内存占用-1)
      - [基准测试结果](#基准测试结果-1)
  - [如何运行](#如何运行)
    - [V0.2 展示](#v02-展示)
      - [单插槽版本 (32 核心)](#单插槽版本（32 核心）)
      - [双插槽版本 (64 核心)](#双插槽版本（64 核心）)
    - [V0.3 展示](#v03-展示)
      - [双插槽版本 (64 核心)](#双插槽版本（64 核心）-1)
  - [一些解释](#一些解释)
  - [常见问题解答](#常见问题解答)
    - [R1 不思考](#R1 不返回思考过程)
    - [更多常见问题解答](#更多常见问题解答)

# 摘要

> **2025年2月10日**: 支持在单个（24GB 显存）/多个 GPU 和 382GB 内存上运行 DeepseekR1 和 V3，速度提升高达 3~28 倍。<br>

嗨，我们是 KTransformers 团队（以前因本地 CPU/GPU 混合推理开源项目 DeepSeek-V2 而闻名）。

我们听到了您对 DeepSeek-R1/V3 支持的请求——我们很高兴终于可以交付了！很抱歉让您久等了，但我们一直在酝酿一些真正令人惊叹的东西！

今天，我们自豪地宣布，我们不仅支持 DeepSeek-R1/V3，如下视频所示：

https://github.com/user-attachments/assets/ebd70bfa-b2c1-4abb-ae3b-296ed38aa285

</p>

- **[NEW!!!] 本地 671B DeepSeek-Coder-V3/R1:** 仅使用 14GB 显存和 382GB 内存运行其 Q4_K_M 版本。
  - 预填充(Prefill)速度 (tokens/s):
    - KTransformers: 54.21 (32 核心) → 74.362 (双插槽，2×32 核心) → 255.26 (优化的 AMX 基 MoE 内核，仅 V0.3) → 286.55 (选择性使用 6 个专家，仅 V0.3)
    - 与 llama.cpp 在 2×32 核心下 10.31 tokens/s 相比，速度提升高达 **27.79 倍**
  - 解码(Decode)速度 (tokens/s):
    - KTransformers: 8.73 (32 核心) → 11.26 (双插槽， 2×32 核心) → 13.69 (选择性使用 6 个专家，仅 V0.3)
    - 与 llama.cpp 在 2×32 核心下 4.51 tokens/s 相比，速度提升高达 **3.03 倍**

我们还提供了即将推出的优化预览，包括英特尔 AMX 加速内核和选择性专家激活方法，这将显著提升性能。通过 V0.3 预览版，我们在预填充方面实现了高达 286 tokens/s 的速度，比本地推理的 llama.cpp **快 28 倍**。二进制发行版现已可用，源代码即将推出！请查看 wheel 包 [此处](https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl) 。

## 先决条件

我们在以下配置下进行了最佳性能测试（V0.2）： <br>
CPU: Intel (R) Xeon (R) Gold 6454S 1T 内存 (2 NUMA 节点) <br>
GPU: 4090D 24G 显存 <br>
内存: 标准 DDR5-4800 服务器内存 (1 TB)

## 基准测试结果

### V0.2

#### 设置

- Model: DeepseekV3-q4km (int4)<br>
- CPU: cpu_model_name: Intel (R) Xeon (R) Gold 6454S，每个插槽 32 核心，2 个插槽，2 个 NUMA 节点
- GPU: 4090D 24G 显存
- 我们在充分预热后进行测试

#### 内存占用:

- 单插槽: 382G 内存，至少 14GB 显存
- 双插槽: 1T 内存，至少 14GB 显存

#### 基准测试结果

“6 个专家” 情况是 V0.3 预览版中内容


| Prompt<br>(500 tokens)  | 双插槽 Ktrans (6 个专家) | 双插槽 Ktrans (8 个专家) | Single socket Ktrans (6 个专家) | Single socket Ktrans (8 个专家) | llama.cpp (8 个专家) |
| ----------------------- | ------------------------ | ------------------------ | ------------------------------- | ------------------------------- | -------------------- |
| 预填充(Prefill) token/s | 97.32                    | 82.94                    | 65.14                           | 54.21                           | 10.31                |
| 解码(Decode) token/s    | 13.69                    | 12.208                   | 10.303                          | 8.73                            | 4.51                 |

**最高加速比在解码方面达到 <u>3.03x</u> 倍，在预填充方面达到 <u>9.44x</u> 倍。**

### V0.3-Preview

#### 设置

- Model: DeepseekV3-BF16 (在线量化为 CPU 的 int8 和 GPU 的 int4)
- CPU: cpu_model_name: Intel (R) Xeon (R) Gold 6454S，每个插槽 32 核心，2 个插槽，2 个 NUMA 节点
- GPU: (1~4)x 4090D 24G 显存 (更长的 prompt 需要更多显存)

#### 内存占用:

- 644GB 内存，至少 14GB 显存

#### 基准测试结果


| Prompt length                     | 1K     | 2K     | 4K     | 8K     |
| --------------------------------- | ------ | ------ | ------ | ------ |
| KTrans (8 个专家) Prefill token/s | 185.96 | 255.26 | 252.58 | 195.62 |
| KTrans (6 个专家) Prefill token/s | 203.70 | 286.55 | 271.08 | 207.20 |

**KTrans V0.3 的预填充速度比 KTrans V0.2 快 <u>3.45x</u> 倍，比 llama.cpp 快 <u>27.79x</u> 倍。**
**解码速度与 KTrans V0.2（6 个专家版本）相同，因此省略。**

主要加速来自于

- 英特尔 AMX 指令集和我们专门设计的缓存友好内存布局
- 专家选择策略，根据离线配置文件结果选择更少的专家

*从我们对 DeepSeekV2、DeepSeekV3 和 DeepSeekR1 的研究中，当我们略微减少推理中的激活专家数量时，输出质量没有变化。但解码和预填充的速度加快了，这令人鼓舞。因此，我们的展示利用了这一发现。*

## 如何运行

### 多并发展示

多并发需要额外编译调度器 c++ 代码

```shell
sudo apt install libtbb-dev libssl-dev libcurl4-openssl-dev libaio1 libaio-dev libfmt-dev
sudo apt-get install libgflags-dev zlib1g-dev patchelf
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule update --init --recursive
# 如果使用双 numa 版本
USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh
# 如果使用单 numa 版本
USE_BALANCE_SERVE=1 bash ./install.sh
# 启动命令
python ktransformers/server/main.py --model_path <your model path> --gguf_path <your gguf path> --cpu_infer 62 --optimize_config_path <inject rule path> --port 10002 --chunk_size 256 --max_new_tokens 1024 --max_batch_size 4 --port 10002 --cache_lens 32768 --backend_type balance_serve
```

`<your model path>` 可以是本地路径，也可以是在线路径，例如 deepseek-ai/DeepSeek-V3。如果在线连接出现问题，可以尝试使用镜像（hf-mirror.com） <br>
`<your gguf path>` 也可以是在线路径，但由于其体积较大，我们建议您下载并量化模型（注意这是目录路径）

`<inject rule path>` 注入规则 yaml 文件地址，我们在 `ktransformers/optimize/optimize_rules/ ` 目录下提供了 `DeepSeek-V3-Chat-serve.yaml` 和 `DeepSeek-V3-Chat-fp8-linear-ggml-experts-serve.yaml` 分别对应 [`DeepSeek-V3/R1-q4km`](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M) 和 [`DeepSeek-V3/R1-hybrid`](https://huggingface.co/KVCache-ai/DeepSeek-R1-GGML-FP8-Hybrid/tree/main)

`--max_new_tokens 1000` 是最大输出 token 长度。如果发现答案被截断，可以增加此数字以获得更长的答案（但要注意内存不足问题，增加此数字会降低生成速度）.

`--chunk_size 256` 引擎单次运行最大 token 个数

`--cache_lens 32768`  调度器申请 kvcache 的总长度。所有请求共享 32768 个 tokens 对应 kvcache 空间，请求完成后会释放其所占用的 kvcache 空间。

`--backend_type balance_serve` `balance_serve`是 v0.2.4新增的后端引擎，原本的单并发引擎为`ktransformers`

`--max_batch_size 4` 引擎单次运行最多处理 4 个请求(prefill + decode),(仅用于`balance_serve`)

<br>命令 numactl -N 1 -m 1 的目的是避免 NUMA 节点之间的数据传输<br>
注意！如果测试 R1 可能会跳过思考。因此，可以添加参数：`--force_think`，这在 [常见问题解答](#常见问题解答) 部分中解释。

### V0.2 展示

#### 单插槽版本（32 核心）

我们的 local_chat 测试命令是:

```shell
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule init
git submodule update
numactl -N 1 -m 1 python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 33 --max_new_tokens 1000
<当您看到聊天时，按回车键加载文本提示文件>
```

#### 双插槽版本（64 核心）

在安装之前（使用 install.sh 或 `make dev_install`），请确保设置环境变量 `USE_NUMA=1`，方法是 `export USE_NUMA=1`（如果已经安装，请重新安装并设置此环境变量） <br>
我们的 local_chat 测试命令是：

```shell
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git submodule init
git submodule update
export USE_NUMA=1
make dev_install # or sh ./install.sh
python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 65 --max_new_tokens 1000
<当您看到聊天时，按回车键加载文本提示文件>
```

参数的含义相同。但因为我们使用双插槽，所以将 cpu_infer 设置为 65。

### V0.3 展示

#### 双插槽版本（64 核心）

我们的 local_chat 测试命令是：

```shell
wget https://github.com/kvcache-ai/ktransformers/releases/download/v0.1.4/ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
pip install ./ktransformers-0.3.0rc0+cu126torch26fancy-cp311-cp311-linux_x86_64.whl
python -m ktransformers.local_chat --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your prompt txt file>  --cpu_infer 65 --max_new_tokens 1000
<当您看到聊天时，按回车键加载文本提示文件>
```

参数的含义与 V0.2 相同。但因为我们使用双插槽，所以将 cpu_infer 设置为 65。

## 一些解释

1. 我们还想进一步利用 Xeon Gold CPU 上的两个 NUMA 节点。为了避免节点之间的数据传输成本，我们在两个节点上 "copy" 了关键矩阵，这会增加内存占用，但会加速预填充和解码过程。但这种方法占用大量内存，加载权重时速度较慢，因此加载时请耐心等待并监控内存使用情况。我们计划优化这一巨大的内存开销。敬请期待。
2. 命令参数 `--cpu_infer 65` 指定使用多少核心（超过物理核心数量是可以的，但并不是越多越好。根据实际核心数量适当降低此值）。<br>
3. 为什么使用 CPU/GPU 混合推理？
   DeepSeek 的 MLA 操作符计算密集。虽然全部在 CPU 上运行是可行的，但将繁重的计算任务卸载到 GPU 上能带来巨大的性能提升。
4. 加速来自哪里？

   - 专家卸载：与传统的基于层或 KVCache 卸载（如 llama.cpp 中的）不同，我们将专家计算卸载到 CPU，将 MLA/KVCache 卸载到 GPU，与 DeepSeek 的架构完美对齐，实现最佳效率。
   - 英特尔 AMX 优化 – 我们的 AMX 加速内核经过精心调优，运行速度是现有 llama.cpp 实现的数倍。我们计划在清理后开源此内核，并考虑向 llama.cpp 上游贡献代码。
5. 为什么选择英特尔 CPU？
   英特尔目前是唯一支持 AMX 类似指令的 CPU 供应商，与仅支持 AVX 的替代方案相比，性能显著更好。

## 常见问题解答

### R1 不返回思考过程

注意！如果测试 R1 可能会跳过思考。因此，可以添加参数：`--force_think true`。详细信息在 [常见问题解答](./FAQ.md) 部分中。 <br>

## 问题

* 修复服务器集成功能以实现网络API访问支持
* 修复本地聊天功能仅支持单行提示输入的问题（目前输入换行符(\n)即开始生成提示）

### 更多常见问题解答

[详见](./FAQ.md)
