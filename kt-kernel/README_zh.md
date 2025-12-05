# KT-Kernel

高性能 KTransformers 内核库，提供面向 CPU 的高效 MoE 推理内核，支持 AMX 和 AVX 等后端。

- [KT-Kernel](#kt-kernel)
  - [说明](#说明)
  - [特性](#特性)
  - [安装](#安装)
    - [先决条件](#先决条件)
    - [快速安装（推荐）](#快速安装推荐)
    - [手动配置（进阶）](#手动配置进阶)
  - [验证安装](#验证安装)
  - [与 SGLang 集成](#与-sglang-集成)
    - [安装步骤](#安装步骤)
      - [1. 安装 SGLang](#1-安装-sglang)
      - [2. 准备权重](#2-准备权重)
      - [3. 启动 SGLang Server](#3-启动-sglang-server)
    - [完整示例：Qwen3-30B-A3B](#完整示例qwen3-30b-a3b)
      - [方案 A：AMX 后端（AMXINT8）](#方案-aamx-后端amxint8)
      - [方案 B：LLAMAFILE 后端（GGUF）](#方案-bllamafile-后端gguf)
    - [KT-Kernel 参数](#kt-kernel-参数)
  - [直接使用 Python API](#直接使用-python-api)
    - [高级选项](#高级选项)
  - [构建配置](#构建配置)
    - [手动安装](#手动安装)
      - [1. 安装系统依赖](#1-安装系统依赖)
      - [2. 配置构建参数](#2-配置构建参数)
      - [3. 构建并安装](#3-构建并安装)
  - [错误排查](#错误排查)
    - [找不到 CUDA](#找不到-cuda)
    - [找不到 hwloc](#找不到-hwloc)
  - [权重量化](#权重量化)
  - [提交前必读](#提交前必读)

## 说明

**当前支持状态：**
- ✅ **带 AMX 的 Intel CPU**：已支持（基于转换为 INT4/INT8 格式的权重）
- ✅ **通用 CPU（llamafile 后端）**：已支持（基于 GGUF 格式的权重）
- ✅ **带 BLIS 的 AMD CPU**：已支持（int8 的 prefill 和 decode）

## 特性

- **CPU 友好的 MoE 内核**：针对指令集优化的高吞吐 MoE 专家内核。
- **AMX INT4/INT8 后端**：面向支持 AMX 的服务器提供 INT4 / INT8 量化专家推理后端。
- **Llamafile CPU 后端**：基于 Llamafile 的 AVX2/AVX512 MoE 后端，适用于通用 CPU 部署。
- **NUMA 感知执行**：为多路 / 多 NUMA 机器设计的线程池和内存布局。


## 安装

### 先决条件

首先初始化子模块：
```bash
git submodule update --init --recursive
```

### 快速安装（推荐）

第 0 步：创建并激活一个 conda 环境（推荐）：

```bash
conda create -n kt-kernel python=3.11 -y
conda activate kt-kernel
```

随后可以用同一个脚本分两步或一步安装。

方案 A：两步（可以指定依赖安装与编译构建）

```bash
# 1）安装系统依赖（cmake, hwloc, pkg-config）
./install.sh deps

# 2）构建并安装 kt-kernel（自动检测 CPU 指令集）
#    默认会在编译前清理本地 ./build 目录
./install.sh build
```

方案 B：一步

```bash
./install.sh
```

安装脚本会：
- 自动检测 CPU 能力（是否支持 AMX）
- 尝试通过 conda 安装 `cmake`（若可用）
- 根据你的操作系统安装系统依赖（`libhwloc-dev`、`pkg-config`）

**自动配置内容：**
- 检测到 AMX CPU → 使用 `NATIVE + AMX=ON`
- 未检测到 AMX → 使用 `NATIVE + AMX=OFF`

⚠️ **LLAMAFILE 后端用户特别说明：**  
如果你有带 AMX 的 CPU，但是计划使用 LLAMAFILE 后端，请不要使用默认的自动检测构建方式。  
请使用“手动模式”，并将 `CPUINFER_CPU_INSTRUCT` 设为 `AVX512` 或 `AVX2` 而非 `NATIVE`，以避免编译期异常（见下文）。

### 手动配置（进阶）

如果你需要更精细的构建选项（例如为 LLAMAFILE 后端、兼容性或二进制分发配置）：

```bash
# 在带 AMX 的 CPU 上构建 LLAMAFILE 后端的示例（使用 AVX512）
export CPUINFER_CPU_INSTRUCT=AVX512  # 选项: NATIVE, AVX512, AVX2, FANCY
export CPUINFER_ENABLE_AMX=OFF       # 选项: ON, OFF

# 仅构建（不进行指令集的自动检测）
./install.sh build --manual
```

更多构建选项和二进制分发配置，请参见 [构建配置](#构建配置) 一节。  
如果遇到问题，可参考 [错误排查](#错误排查)。

## 验证安装

```bash
python -c "from kt_kernel import KTMoEWrapper; print('✓ kt-kernel installed successfully')"
```

## 与 SGLang 集成

KT-Kernel 可以单独通过 [Python API](#直接使用-python-api) 使用，也可以集成到 SGLang 中用于生产部署。  
本节描述如何与 SGLang 集成，实现 CPU-GPU 混合（异构）推理：将“热” experts 放在 GPU 上，“冷” experts 放在 CPU 上，以达到资源利用和性价比的平衡。

### 安装步骤

#### 1. 安装 SGLang

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"
```

#### 2. 准备权重

要进行异构推理，需要同时准备 GPU 权重和 CPU 侧 experts 对应的权重，具体格式取决于后端类型：

**GPU 权重：**  
使用 SGLang 所需的模型权重（例如 Hugging Face 上的原始模型目录或已量化好的 GPU 权重）。

**CPU 权重（AMX 后端：`AMXINT4` / `AMXINT8`）：**  
通过提供的脚本将权重量化为适配 AMX 的 INT4/INT8 格式：

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/cpu-weights \
  --quant-method int8  # 或 int4 或 moe_int8（用于 amd 的）
```

- `--input-path`：GPU 侧原始权重路径
- `--input-type`：取决于 GPU 侧权重类型（`fp8`、`fp16` 或 `bf16`）

在 SGLang 集成中，`--kt-weight-path` 应指向该转换后的 CPU 权重目录。

**支持的输入格式：** FP8、FP16、BF16 → INT4/INT8。

**CPU 权重（LLAMAFILE 后端：`LLAMAFILE`）：**  
LLAMAFILE 在 CPU 侧直接使用预量化的 **GGUF** 权重，无需运行 `convert_cpu_weights.py`。你需要：

- 直接从互联网上下载 GGUF 模型（例如 Hugging Face / Modelscope 上的 GGUF 仓库）；
- 在 SGLang 集成中，将该 GGUF 目录作为 `--kt-weight-path`。
  KT-Kernel 支持多种 GGUF 量化格式，例如 `Q4_KM`、`Q4_K`、`Q5_K` 等，可根据延迟和效果需求选择。

#### 3. 启动 SGLang Server

在通常的 SGLang 启动参数基础上，增加如下 KT-Kernel 相关参数，以启用 CPU-GPU 异构推理：

**需要增加的 KT-Kernel 参数：**
- `--kt-method`：后端类型（AMXINT4、AMXINT8、或 LLAMAFILE）
- `--kt-weight-path`：转换后的 CPU 权重路径
- `--kt-cpuinfer`：CPU 推理线程数（建议设为物理核数）
- `--kt-threadpool-count`：线程池数量（建议设为 NUMA 节点个数）
- `--kt-num-gpu-experts`：留在 GPU 上的 experts 数量
- `--kt-max-deferred-experts-per-token`：每个 token 延迟到 CPU 的 experts 数量，用于流水线执行

示例：
```bash
python -m sglang.launch_server \
  [your normal SGLang parameters...] \
  --kt-method AMXINT8 \
  --kt-weight-path /path/to/cpu-weights \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

更多调优建议见 [KT-Kernel 参数](#kt-kernel-参数) 一节。

### 完整示例：Qwen3-30B-A3B

该示例展示从下载权重到启动服务的完整流程，分别演示 **AMX 后端** 和 **LLAMAFILE 后端** 两种方案。

**硬件配置：**
- **GPU**：NVIDIA RTX 4090 24GB
- **CPU**：2x Intel Xeon Gold 6454S（共 64 个物理核，128 线程，2 个 NUMA 节点）
- **模型**：[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)

**如何检查系统配置：**
```bash
# 查看 CPU 配置
lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core|Socket\(s\)|NUMA node\(s\)"
# 期望输出示例:
CPU(s):                                  128
Thread(s) per core:                      2
Socket(s):                               2
NUMA node(s):                            2
# → 物理核数 = CPU(s) / Thread(s) per core = 128 / 2 = 64
```

**参数选型说明：**
- `--kt-cpuinfer 64`：设为物理核数（64），而不是 128 线程
- `--kt-threadpool-count 2`：检测到 2 个 NUMA 节点（双路系统）
- `--kt-num-gpu-experts 32`：在 24GB 显存下，对该模型可以大约放 32 个 experts 在 GPU 上（具体取决于模型结构和实际内存占用）
- `--kt-max-deferred-experts-per-token 2`：启用流水线执行；允许 CPU 处理下一批 token 的同时，GPU 完成当前批次

---

#### 方案 A：AMX 后端（AMXINT8）

适用于支持 AMX 指令集的 Intel CPU。

**步骤 1：下载模型权重**

```bash
# 如未安装 huggingface-cli，请先安装
pip install huggingface-hub

# 从 Hugging Face 下载模型
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /mnt/data/models/Qwen3-30B-A3B
```

**步骤 2：转换为 CPU 权重（AMXINT8）**

```bash
python scripts/convert_cpu_weights.py \
  --input-path /mnt/data/models/Qwen3-30B-A3B \
  --input-type bf16 \
  --output /mnt/data/models/Qwen3-30B-A3B-INT8 \
  --quant-method int8
```

**步骤 3：启动 SGLang 服务**

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /mnt/data/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method AMXINT8 \
  --kt-weight-path /mnt/data/models/Qwen3-30B-A3B-INT8 \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

---

#### 方案 B：LLAMAFILE 后端（GGUF）

适用于通用 CPU（无需 AMX 支持），直接使用预量化的 GGUF 权重。

**步骤 1：下载 GPU 权重（原始模型）**

```bash
pip install huggingface-hub

huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /mnt/data/models/Qwen3-30B-A3B
```

**步骤 2：下载 CPU 权重（GGUF 格式）**

```bash
huggingface-cli download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf \
  --local-dir /mnt/data/models/Qwen3-30B-A3B-Q4_K_M
```

**步骤 3：启动 SGLang 服务**

```bash
python -m sglang.launch_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /mnt/data/models/Qwen3-30B-A3B \
  --trust-remote-code \
  --mem-fraction-static 0.92 \
  --chunked-prefill-size 4096 \
  --served-model-name Qwen3-30B-A3B \
  --enable-mixed-chunk \
  --kt-method LLAMAFILE \
  --kt-weight-path /mnt/data/models/Qwen3-30B-A3B-Q4_K_M \
  --kt-cpuinfer 64 \
  --kt-threadpool-count 2 \
  --kt-num-gpu-experts 32 \
  --kt-max-deferred-experts-per-token 2
```

### KT-Kernel 参数

| 参数 | 描述 | 示例值 |
|------|------|--------|
| `--kt-method` | CPU 推理后端类型 | `AMXINT4`、`AMXINT8`、`RAWINT4` 或 `LLAMAFILE` |
| `--kt-weight-path` | 量化后的 CPU 权重路径 | `/path/to/cpu-weights` |
| `--kt-cpuinfer` | CPU 推理线程数 | `64`（根据 CPU 核心数调整） |
| `--kt-threadpool-count` | 并行执行的线程池数量 | `2`（通常为 1–4） |
| `--kt-num-gpu-experts` | 保留在 GPU 上的 experts 数量 | `32`（其余 experts 由 CPU 承担） |
| `--kt-max-deferred-experts-per-token` | 每个 token 延迟到 CPU 的 experts 数量（用于流水线执行） | `2`（0 关闭，1–4 推荐） |
| `--kt-gpu-prefill-token-threshold` | Prefill 策略的 token 数量阈值（仅 RAWINT4） | ~`400` |

**参数建议：**

- **`kt-method`**：根据 CPU 能力和权重格式选择：
  - `AMXINT4`：在 AMX CPU 上 INT4 量化时具有最佳性能（但可能对某些模型有较大精度影响，例如 Qwen3-30B-A3B）
  - `AMXINT8`：在 AMX CPU 上提供更高精度的 INT8 量化方案
  - `RAWINT4`：CPU 和 GPU 共享原生 INT4 权重（仅限 AMX 后端，目前仅支持 Kimi-K2-Thinking 模型）。详见 [Kimi-K2-Thinking 原生推理教程](../doc/en/Kimi-K2-Thinking-Native.md)。
  - `LLAMAFILE`：基于 AVX2/AVX512 的通用 CPU 后端，性能较 AMX 略低，但适用范围更广

- **`kt-cpuinfer`**：设置为 **物理核数**（不是线程数）。
  - 查看物理核数：`lscpu | grep -E "^CPU\(s\)|Thread\(s\) per core"`
  - 计算方式：物理核数 = CPU(s) / Thread(s) per core
  - 例：若 CPU(s)=128 且 Thread(s) per core=2，则物理核数=64
  - **重要**：不要设置为超线程总数，否则会降低性能

- **`kt-threadpool-count`**：设置为 **NUMA 节点数**。
  - 查看 NUMA 数：`lscpu | grep "NUMA node(s)"`
  - 或：`numactl --hardware | grep "available"`
  - **注意**：NUMA 节点数不等同于物理 CPU 数量：
    - 它表示内存域，可能在单颗 CPU 内被拆分，也可能跨多颗 CPU。
    - 请以 `lscpu` 输出的 NUMA 节点数为准。
  - 常见配置：单路 1–2，双路 2–4
  - 正确设置有助于充分利用跨 NUMA 域的内存带宽。

- **`kt-num-gpu-experts`**：根据 GPU 显存和实际性能测试决定：
  - GPU 上的 experts 越多 → 延迟越低，但显存占用越高（可能 OOM）

- **`kt-max-deferred-experts-per-token`**：用于开启 CPU-GPU 流水线：
  - `0`：完全同步执行（简单但延迟较高）
  - `1–4`：推荐范围，一部分 experts 延迟到 CPU，在延迟和质量之间取得较好平衡（需要按模型调参）
  - `5–7`：可以获得更低延迟，但存在明显精度下降风险，请谨慎使用

- **`kt-gpu-prefill-token-threshold`**（仅 RAWINT4）：控制原生 INT4 推理的 prefill 策略：
  - **≤ 阈值**：使用 CPU+GPU 混合 prefill。无需额外显存，但随着 token 数量增加性能会缓慢下降。
  - **> 阈值**：使用分层 GPU prefill。长序列性能更好，但需要约 9GB+ 额外显存。
  - 仅在使用 `--kt-method RAWINT4` 时生效。目前仅支持 Kimi-K2-Thinking 模型。

## 直接使用 Python API

如果不集成 SGLang，也可以直接通过 Python API 单独使用 KT-Kernel：

```python
from kt_kernel import KTMoEWrapper

# 初始化 MoE 包装器
wrapper = KTMoEWrapper(
    layer_idx=0,
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=4096,
    moe_intermediate_size=14336,
    num_gpu_experts=2,
    cpuinfer_threads=32,
    threadpool_count=2,
    weight_path="/path/to/weights",
    chunked_prefill_size=512,
    method="AMXINT4"  # 选项: "AMXINT4", "AMXINT8", "LLAMAFILE"
)

# 从磁盘加载权重（预先量化好）
wrapper.load_weights(physical_to_logical_map)

# 或者从张量加载权重（在线量化）
wrapper.load_weights_from_tensors(gate_proj, up_proj, down_proj, physical_to_logical_map)

# 执行推理
output = wrapper.forward(hidden_states, topk_ids, topk_weights, cuda_stream)

# 或使用异步 API 获取更好的流水线效果
wrapper.submit_forward(hidden_states, topk_ids, topk_weights, cuda_stream)
# ... 做一些其他工作 ...
output = wrapper.sync_forward(hidden_states, cuda_stream)
```

### 高级选项

```python
# 使用更多高级选项初始化
wrapper = KTMoEWrapper(
    layer_idx=0,
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=4096,
    moe_intermediate_size=14336,
    num_gpu_experts=2,
    cpuinfer_threads=32,
    threadpool_count=2,
    weight_path="/path/to/weights",
    chunked_prefill_size=512,
    method="AMXINT4",
    cpu_save=False,  # 加载后是否将权重常驻 CPU 内存
    max_deferred_experts_per_token=0  # 每个 token 延迟的 experts 数量（用于流水线）
)

# 为特定 batch size 预分配缓冲区（提升性能）
KTMoEWrapper.set_capture_batch_sizes([1, 2, 4, 8, 16])

# 查看当前捕获的 batch size
batch_sizes = KTMoEWrapper.get_capture_batch_sizes()

# 清理缓冲区缓存以释放内存
KTMoEWrapper.clear_buffer_cache()
```

## 构建配置

### 手动安装

如果你不想使用 `install.sh`，可以按以下步骤手动构建：

#### 1. 安装系统依赖

**前置依赖：**
- `cmake`（推荐：`conda install -y cmake`）
- `libhwloc-dev` 和 `pkg-config`

#### 2. 配置构建参数

**核心选项：**

| 变量 | 取值 | 描述 |
|------|------|------|
| `CPUINFER_CPU_INSTRUCT` | `NATIVE`, `AVX512`, `AVX2`, `FANCY` | 使用的 CPU 指令集 |
| `CPUINFER_ENABLE_AMX` | `ON`, `OFF` | 是否启用 Intel AMX 支持 |
| `CPUINFER_BUILD_TYPE` | `Release`, `Debug`, `RelWithDebInfo` | 构建类型（默认：`Release`） |
| `CPUINFER_PARALLEL` | 数值 | 并行构建的 Job 数（默认：自动检测） |
| `CPUINFER_VERBOSE` | `0`, `1` | 是否启用详细构建日志（默认：`0`） |

**指令集说明：**

- **`NATIVE`**：自动检测并启用所有可用 CPU 指令（`-march=native`）——**本机运行时首选**
- **`AVX512`**：为 Skylake-SP / Cascade Lake 显式开启 AVX512
- **`AVX2`**：开启 AVX2，兼容性较好
- **`FANCY`**：开启完整 AVX512 扩展（AVX512F/BW/DQ/VL/VNNI），适用于 Ice Lake+ 和 Zen 4+ 等较新平台。  
  用于向用户分发预编译二进制时推荐；本地构建推荐使用 `NATIVE` 以获得更优性能。

**配置示例：**

```bash
# 在 AMX CPU 上获得最高性能
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON

# 仅 AVX512，无 AMX
export CPUINFER_CPU_INSTRUCT=AVX512
export CPUINFER_ENABLE_AMX=OFF

# 兼容性优先构建
export CPUINFER_CPU_INSTRUCT=AVX2
export CPUINFER_ENABLE_AMX=OFF

# 调试构建
export CPUINFER_BUILD_TYPE=Debug
export CPUINFER_VERBOSE=1
```

#### 3. 构建并安装

```bash
# 开发模式（可编辑安装）
pip install -e .

# 普通安装
pip install .
```

## 错误排查

### 找不到 CUDA

```
 -- Looking for a CUDA compiler - NOTFOUND
  CMake Error at CMakeLists.txt:389 (message):
    KTRANSFORMERS_USE_CUDA=ON but CUDA compiler not found
```

请确认已安装 CUDA Toolkit 且 `nvcc` 在系统 PATH 中。

可以尝试：

```bash
export CMAKE_ARGS="-D CMAKE_CUDA_COMPILER=$(which nvcc)"
pip install .
```

然后重新安装。

### 找不到 hwloc

在 Debian 系发行版上可以直接：

```bash
sudo apt install libhwloc-dev
```

或从源码构建：https://www.open-mpi.org/projects/hwloc/

```bash
wget https://download.open-mpi.org/release/hwloc/v2.12/hwloc-2.12.2.tar.gz
tar -xzf hwloc-2.12.2.tar.gz
cd hwloc-2.12.2
./configure
make
sudo make install
```

## 权重量化

对于 AMX 后端（`AMXINT4` / `AMXINT8`），CPU 侧 experts 需要通过提供的脚本转换为适配 AMX 的 INT4/INT8 格式：

```bash
python scripts/convert_cpu_weights.py \
  --input-path /path/to/model \
  --input-type bf16 \
  --output /path/to/output \
  --quant-method int4
```

**支持的格式：** FP8、FP16、BF16 → INT4/INT8

对于 LLAMAFILE 后端（`LLAMAFILE`），CPU 侧 experts 直接从 **GGUF** 权重中加载。  
你**不需要**运行 AMX 转换脚本；只需从互联网上下载 GGUF 模型（例如 Hugging Face 上的 GGUF 仓库），并在 `weight_path` 或 SGLang 的 `--kt-weight-path` / `--model` 中指向该 GGUF 目录即可。KT-Kernel 支持多种 GGUF 量化格式，如 `Q4_KM`、`Q4_K`、`Q5_K` 等。

---

更多详细文档、高级参数和低显存模式，请参见 [scripts/README.md](scripts/README.md)。

## 提交前必读

提交信息应符合 Conventional Commits 规范：https://www.conventionalcommits.org/  
在提交前请先格式化代码：

```shell
cmake -B build
cd build
make format
```

你可能需要一个较新的 clang-format（至少 18），在 conda 环境中可以：

```shell
conda install -c conda-forge clang-format=18
rm -rf build
```

并且建议安装 black 用于 Python 代码格式化：

```shell
conda install black
```
