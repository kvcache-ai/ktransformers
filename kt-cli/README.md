# kt-cli

[English](#english) | [中文](#中文)

---

## English

**kt-cli** is a unified command-line interface for KTransformers. It provides a user-friendly way to access all KTransformers functionality including model inference, fine-tuning, benchmarking, and more.

### Features

- 🚀 **Easy Model Serving**: Start inference servers with a single command
- 📦 **Smart Installation**: Auto-detects environment and installs dependencies
- 🔍 **Fuzzy Model Matching**: Find models by partial names
- 🌍 **Bilingual Support**: Full English and Chinese language support
- ⚙️ **Flexible Configuration**: Persistent settings with environment variable support
- 🏥 **Environment Diagnostics**: Built-in health checks with `kt doctor`

### Installation

```bash
# Install from source
cd kt-cli
pip install -e .

# Or install from PyPI (coming soon)
pip install kt-cli
```

### First Run

On first run, kt-cli will prompt you to select your preferred language:

```
╭────────────────────────── kt-cli ───────────────────────────╮
│ Welcome to KTransformers CLI! / 欢迎使用 KTransformers CLI! │
│                                                             │
│ Let's set up your preferences.                              │
│ 让我们设置您的偏好。                                        │
╰─────────────────────────────────────────────────────────────╯

Select your preferred language / 选择您的首选语言:

  [1] English
  [2] 中文 (Chinese)
```

You can re-run this setup anytime with `kt config init`.

### Quick Start

```bash
# Check your environment
kt doctor

# Download a model
kt download deepseek-v3

# Start inference server
kt run deepseek-v3

# Fine-tune with LlamaFactory
kt sft train config.yaml
```

### Commands

#### `kt version`

Display version information.

```bash
kt version          # Basic info
kt version -v       # Detailed package versions
```

#### `kt install`

Install KTransformers and dependencies.

**Important**: It's recommended to run `kt install` inside a virtual environment.

```bash
# First, create and activate a virtual environment
conda create -n kt python=3.10 && conda activate kt
# or: python -m venv kt-env && source kt-env/bin/activate

# Install from PyPI (default)
kt install                    # Install inference components
kt install inference          # Install inference components
kt install sft                # Install fine-tuning components
kt install full               # Install all components

# Install from source
kt install --source /path/to/ktransformers    # Build from local source
kt install -s /path/to/repo -e                # Editable install (for development)
kt install -s . -e                            # Editable install from current dir

# Options
kt install -y                 # Skip confirmations
kt install -f                 # Force reinstall
kt install --skip-torch       # Skip PyTorch (if already installed)
kt install -b dev             # Use specific git branch (with --source)
```

#### `kt update`

Update KTransformers to the latest version.

**You must specify the update method explicitly.**

```bash
# Update from PyPI
kt update --pypi

# Update from source (git pull + rebuild)
kt update --source /path/to/ktransformers

# Options
kt update --pypi -y           # Skip confirmations
```

#### `kt run`

Start model inference server (SGLang + kt-kernel).

```bash
kt run deepseek-v3            # Start with default settings
kt run qwen3-30b -p 8080      # Custom port
kt run /path/to/model         # Use local model path

# Options
--host, -H          Server host (default: 0.0.0.0)
--port, -p          Server port (default: 30000)
--gpu-experts       GPU experts per layer (default: 1)
--cpu-threads       CPU inference threads (auto-detected)
--numa-nodes        NUMA node count (auto-detected)
--model-path        Custom model path
--weights-path      Custom quantized weights path
--quantize, -q      Quantize if weights not found
--dry-run           Show command without executing
```

#### `kt download`

Download model weights from HuggingFace.

```bash
kt download deepseek-v3       # Download by name
kt download --list            # List available models
kt download Qwen/Qwen3-30B    # Direct HuggingFace repo

# Options
--path, -p          Custom download path
--resume            Resume incomplete downloads (default: on)
```

#### `kt quant`

Quantize model weights for CPU inference.

```bash
kt quant deepseek-v3                  # Quantize to INT4 (default)
kt quant deepseek-v3 --method int8    # Quantize to INT8
kt quant /path/to/model -o /output    # Custom output path

# Options
--method, -m        Quantization method: int4, int8
--output, -o        Output path
--input-type        Input type: fp8, fp16, bf16
--cpu-threads       CPU threads for quantization
--no-merge          Don't merge safetensor files
```

#### `kt bench` / `kt microbench`

Run benchmarks.

```bash
kt bench                      # Run full benchmark suite
kt bench --type moe           # Benchmark specific component
kt microbench moe             # Micro-benchmark MoE layer

# Options
--type, -t          Benchmark type: inference, moe, mla, linear, attention, all
--model, -m         Model to benchmark
--output, -o        Output file (JSON)
--iterations, -n    Number of iterations
```

#### `kt config`

Manage configuration.

```bash
kt config init                # Run first-time setup wizard
kt config show                # Show all settings
kt config show server.port    # Show specific setting
kt config set server.port 8080
kt config get server.port
kt config reset               # Reset to defaults
kt config path                # Show config file path
```

#### `kt doctor`

Diagnose environment issues.

```bash
kt doctor                     # Run diagnostics
kt doctor -v                  # Verbose output
```

#### `kt sft`

Fine-tuning with LlamaFactory.

```bash
kt sft train config.yaml      # Train model
kt sft chat config.yaml       # Chat with model
kt sft export config.yaml     # Export model
kt sft eval config.yaml       # Evaluate model

# Options
--use-kt/--no-kt    Enable/disable KTransformers optimization
```

### Configuration

Configuration is stored in `~/.ktransformers/config.yaml`.

```yaml
general:
  language: auto              # auto, en, zh
  color: true
  verbose: false

paths:
  models: ~/.ktransformers/models
  cache: ~/.ktransformers/cache
  weights: ""                 # Custom weights path

server:
  host: 0.0.0.0
  port: 30000

inference:
  cpu_threads: 0              # 0 = auto-detect
  numa_nodes: 0               # 0 = auto-detect
  gpu_experts: 1
  attention_backend: triton
  max_total_tokens: 40000
  max_running_requests: 32

download:
  mirror: ""                  # HuggingFace mirror URL
  resume: true

advanced:
  env: {}                     # Environment variables
  sglang_args: []             # Extra SGLang arguments
  llamafactory_args: []       # Extra LlamaFactory arguments
```

### Environment Variables

- `KT_LANG`: Override language (en, zh)
- `KT_CONFIG`: Custom config file path

### Supported Models

| Model | Aliases | GPU VRAM | CPU RAM |
|-------|---------|----------|---------|
| DeepSeek-V3.2 | deepseek-v3.2, dsv3.2 | 27GB | 350GB |
| DeepSeek-V3 | deepseek-v3, dsv3 | 27GB | 350GB |
| DeepSeek-V2.5 | deepseek-v2.5, dsv2.5 | 16GB | 128GB |
| Qwen3-30B-A3B | qwen3-30b, qwen3 | 12GB | 64GB |
| Kimi-K2 | kimi-k2, kimi, k2 | 24GB | 256GB |
| Mixtral-8x7B | mixtral, mixtral-moe | 12GB | 48GB |
| Mixtral-8x22B | mixtral-8x22b | 24GB | 176GB |

---

## 中文

**kt-cli** 是 KTransformers 的统一命令行界面。它提供了一种用户友好的方式来访问 KTransformers 的所有功能，包括模型推理、微调、基准测试等。

### 特性

- 🚀 **简单的模型服务**：一条命令启动推理服务器
- 📦 **智能安装**：自动检测环境并安装依赖
- 🔍 **模糊模型匹配**：通过部分名称查找模型
- 🌍 **双语支持**：完整的中英文语言支持
- ⚙️ **灵活配置**：持久化设置，支持环境变量
- 🏥 **环境诊断**：内置健康检查 `kt doctor`

### 安装

```bash
# 从源码安装
cd kt-cli
pip install -e .

# 或从 PyPI 安装（即将推出）
pip install kt-cli
```

### 首次运行

首次运行时，kt-cli 会提示您选择首选语言：

```
╭────────────────────────── kt-cli ───────────────────────────╮
│ Welcome to KTransformers CLI! / 欢迎使用 KTransformers CLI! │
│                                                             │
│ Let's set up your preferences.                              │
│ 让我们设置您的偏好。                                        │
╰─────────────────────────────────────────────────────────────╯

Select your preferred language / 选择您的首选语言:

  [1] English
  [2] 中文 (Chinese)
```

您可以随时使用 `kt config init` 重新运行此设置。

### 快速开始

```bash
# 检查环境
kt doctor

# 下载模型
kt download deepseek-v3

# 启动推理服务器
kt run deepseek-v3

# 使用 LlamaFactory 微调
kt sft train config.yaml
```

### 命令说明

#### `kt version`

显示版本信息。

```bash
kt version          # 基本信息
kt version -v       # 详细的包版本
```

#### `kt install`

安装 KTransformers 及其依赖。

**重要**：建议在虚拟环境中运行 `kt install`。

```bash
# 首先，创建并激活虚拟环境
conda create -n kt python=3.10 && conda activate kt
# 或: python -m venv kt-env && source kt-env/bin/activate

# 从 PyPI 安装（默认）
kt install                    # 安装推理组件
kt install inference          # 安装推理组件
kt install sft                # 安装微调组件
kt install full               # 安装所有组件

# 从源码安装
kt install --source /path/to/ktransformers    # 从本地源码编译
kt install -s /path/to/repo -e                # 可编辑安装（用于开发）
kt install -s . -e                            # 从当前目录可编辑安装

# 选项
kt install -y                 # 跳过确认
kt install -f                 # 强制重新安装
kt install --skip-torch       # 跳过 PyTorch（如果已安装）
kt install -b dev             # 使用指定 git 分支（配合 --source）
```

#### `kt update`

更新 KTransformers 到最新版本。

**必须显式指定更新方式。**

```bash
# 从 PyPI 更新
kt update --pypi

# 从源码更新（git pull + 重新编译）
kt update --source /path/to/ktransformers

# 选项
kt update --pypi -y           # 跳过确认
```

#### `kt run`

启动模型推理服务器（SGLang + kt-kernel）。

```bash
kt run deepseek-v3            # 使用默认设置启动
kt run qwen3-30b -p 8080      # 自定义端口
kt run /path/to/model         # 使用本地模型路径

# 选项
--host, -H          服务器地址（默认：0.0.0.0）
--port, -p          服务器端口（默认：30000）
--gpu-experts       每层 GPU 专家数（默认：1）
--cpu-threads       CPU 推理线程数（自动检测）
--numa-nodes        NUMA 节点数（自动检测）
--model-path        自定义模型路径
--weights-path      自定义量化权重路径
--quantize, -q      如果找不到权重则进行量化
--dry-run           显示命令但不执行
```

#### `kt download`

从 HuggingFace 下载模型权重。

```bash
kt download deepseek-v3       # 按名称下载
kt download --list            # 列出可用模型
kt download Qwen/Qwen3-30B    # 直接使用 HuggingFace 仓库

# 选项
--path, -p          自定义下载路径
--resume            断点续传（默认开启）
```

#### `kt quant`

量化模型权重以用于 CPU 推理。

```bash
kt quant deepseek-v3                  # 量化为 INT4（默认）
kt quant deepseek-v3 --method int8    # 量化为 INT8
kt quant /path/to/model -o /output    # 自定义输出路径

# 选项
--method, -m        量化方法：int4, int8
--output, -o        输出路径
--input-type        输入类型：fp8, fp16, bf16
--cpu-threads       量化使用的 CPU 线程数
--no-merge          不合并 safetensor 文件
```

#### `kt bench` / `kt microbench`

运行基准测试。

```bash
kt bench                      # 运行完整基准测试套件
kt bench --type moe           # 测试特定组件
kt microbench moe             # MoE 层微基准测试

# 选项
--type, -t          测试类型：inference, moe, mla, linear, attention, all
--model, -m         要测试的模型
--output, -o        输出文件（JSON）
--iterations, -n    迭代次数
```

#### `kt config`

管理配置。

```bash
kt config init                # 运行首次设置向导
kt config show                # 显示所有设置
kt config show server.port    # 显示特定设置
kt config set server.port 8080
kt config get server.port
kt config reset               # 重置为默认值
kt config path                # 显示配置文件路径
```

#### `kt doctor`

诊断环境问题。

```bash
kt doctor                     # 运行诊断
kt doctor -v                  # 详细输出
```

#### `kt sft`

使用 LlamaFactory 进行微调。

```bash
kt sft train config.yaml      # 训练模型
kt sft chat config.yaml       # 与模型对话
kt sft export config.yaml     # 导出模型
kt sft eval config.yaml       # 评估模型

# 选项
--use-kt/--no-kt    启用/禁用 KTransformers 优化
```

### 配置

配置存储在 `~/.ktransformers/config.yaml`。

```yaml
general:
  language: auto              # auto, en, zh
  color: true
  verbose: false

paths:
  models: ~/.ktransformers/models
  cache: ~/.ktransformers/cache
  weights: ""                 # 自定义权重路径

server:
  host: 0.0.0.0
  port: 30000

inference:
  cpu_threads: 0              # 0 = 自动检测
  numa_nodes: 0               # 0 = 自动检测
  gpu_experts: 1
  attention_backend: triton
  max_total_tokens: 40000
  max_running_requests: 32

download:
  mirror: ""                  # HuggingFace 镜像地址
  resume: true

advanced:
  env: {}                     # 环境变量
  sglang_args: []             # 额外的 SGLang 参数
  llamafactory_args: []       # 额外的 LlamaFactory 参数
```

### 环境变量

- `KT_LANG`：覆盖语言设置（en, zh）
- `KT_CONFIG`：自定义配置文件路径

### 支持的模型

| 模型 | 别名 | GPU 显存 | CPU 内存 |
|------|------|----------|----------|
| DeepSeek-V3.2 | deepseek-v3.2, dsv3.2 | 27GB | 350GB |
| DeepSeek-V3 | deepseek-v3, dsv3 | 27GB | 350GB |
| DeepSeek-V2.5 | deepseek-v2.5, dsv2.5 | 16GB | 128GB |
| Qwen3-30B-A3B | qwen3-30b, qwen3 | 12GB | 64GB |
| Kimi-K2 | kimi-k2, kimi, k2 | 24GB | 256GB |
| Mixtral-8x7B | mixtral, mixtral-moe | 12GB | 48GB |
| Mixtral-8x22B | mixtral-8x22b | 24GB | 176GB |

---

## License

Apache 2.0

## Contributing

Contributions are welcome! Please see the main KTransformers repository for contribution guidelines.
