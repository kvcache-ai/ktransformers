# 独立模型验证脚本 (Standalone Model Verifier)

这是一个独立的 Python 脚本,用于验证本地模型文件的完整性。它会将本地的 safetensors 文件与 HuggingFace 仓库中的官方 SHA256 校验和进行对比。

## 功能特性

- ✅ 并行计算本地文件的 SHA256 哈希值 (最多使用 16 个进程)
- ✅ 从 HuggingFace 获取官方 SHA256 校验和
- ✅ 支持私有仓库 (通过 HF_TOKEN 环境变量)
- ✅ 支持 HuggingFace 镜像 (通过 HF_ENDPOINT 环境变量)
- ✅ 详细的验证报告,包括通过/失败/缺失文件统计
- ✅ 可选的详细模式,显示每个文件的完整 SHA256 值

## 依赖要求

```bash
pip install huggingface-hub
```

## 使用方法

### 基本用法

```bash
python standalone_verify.py <本地模型路径> <HF仓库ID>
```

### 示例

1. **验证公开模型**

```bash
python standalone_verify.py /path/to/local/model deepseek-ai/DeepSeek-V3
```

2. **验证特定分支/版本**

```bash
python standalone_verify.py /path/to/model org/model-name --revision v1.0
```

3. **详细模式 (显示每个文件的完整 SHA256)**

```bash
python standalone_verify.py /path/to/model org/model-name --verbose
```

4. **使用 HF_TOKEN 验证私有模型**

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
python standalone_verify.py /path/to/model org/private-model
```

或者一行命令:

```bash
HF_TOKEN=hf_xxx python standalone_verify.py /path/to/model org/private-model
```

5. **使用 HuggingFace 镜像**

```bash
export HF_ENDPOINT=https://hf-mirror.com
python standalone_verify.py /path/to/model org/model-name
```

## 参数说明

### 位置参数

- `local_path`: 本地模型目录的路径
- `repo_id`: HuggingFace 仓库 ID (格式: `组织/模型名`, 例如 `deepseek-ai/DeepSeek-V3`)

### 可选参数

- `--revision`, `-r`: 指定仓库的分支或版本 (默认: `main`)
- `--verbose`, `-v`: 显示详细的 SHA256 对比信息
- `--help`, `-h`: 显示帮助信息

### 环境变量

- `HF_TOKEN`: HuggingFace API token (用于访问私有仓库)
- `HF_ENDPOINT`: 自定义 HuggingFace endpoint (例如: `https://hf-mirror.com`)

## 输出示例

### 验证通过

```
============================================================
Model Integrity Verification
============================================================
Local path:  /path/to/model
Repository:  deepseek-ai/DeepSeek-V3
Revision:    main
============================================================

Fetching file list from deepseek-ai/DeepSeek-V3 (revision: main)...
Found 152 safetensors files
Fetching SHA256 hashes from deepseek-ai/DeepSeek-V3...
✓ Fetched 152 file hashes from remote

Calculating SHA256 for local files...
Using 16 parallel workers for SHA256 calculation
  [1/152] ✓ model-00001-of-00152.safetensors (4985.2 MB)
  [2/152] ✓ model-00002-of-00152.safetensors (4998.1 MB)
  ...
✓ Calculated 152 local file hashes

Comparing 152 files...

  [1/152] ✓ model-00001-of-00152.safetensors
  [2/152] ✓ model-00002-of-00152.safetensors
  ...

============================================================
Verification Summary
============================================================
Total files:   152
✓ Passed:      152
✗ Failed:      0
✗ Missing:     0
============================================================

✅ VERIFICATION PASSED

All files verified successfully!
```

### 验证失败

```
============================================================
Verification Summary
============================================================
Total files:   152
✓ Passed:      150
✗ Failed:      1
✗ Missing:     1
============================================================

Files with hash mismatch:
  - model-00050-of-00152.safetensors

Missing files:
  - model-00151-of-00152.safetensors

❌ VERIFICATION FAILED

Some files are corrupted or missing. Consider re-downloading the model.
```

## 退出码

- `0`: 验证通过
- `1`: 验证失败或发生错误

## 核心功能说明

脚本的核心验证逻辑包括:

1. **获取远程 SHA256**: 使用 `huggingface_hub` API 从仓库获取所有 safetensors 文件的官方 SHA256 校验和

2. **计算本地 SHA256**: 使用多进程并行计算本地文件的 SHA256 哈希值,每个文件分块读取(8MB chunks)以支持大文件

3. **对比校验**: 逐个对比本地和远程的 SHA256 值,检测以下问题:
   - 文件缺失
   - SHA256 不匹配 (文件损坏)

4. **生成报告**: 输出详细的验证报告,包括每个文件的状态和总体统计

## 注意事项

- 脚本只验证 `.safetensors` 文件
- 大模型的 SHA256 计算可能需要几分钟时间
- 使用多进程并行计算可以显著加快速度
- 如果遇到网络问题,建议使用 HF_ENDPOINT 设置镜像地址

## 故障排除

### 问题: 连接 HuggingFace 超时

**解决方案**: 使用镜像

```bash
export HF_ENDPOINT=https://hf-mirror.com
python standalone_verify.py /path/to/model org/model-name
```

### 问题: 私有仓库访问被拒绝

**解决方案**: 设置 HF_TOKEN

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
python standalone_verify.py /path/to/model org/private-model
```

### 问题: 缺少 huggingface_hub 包

**解决方案**: 安装依赖

```bash
pip install huggingface-hub
```

## 与 kt-cli 的关系

这个脚本是从 kt-cli 的 `kt model verify` 命令中提取出来的核心验证逻辑,去除了 TUI 界面和交互式修复功能,适合:

- 在 CI/CD 流程中进行自动化验证
- 编写自定义的模型管理脚本
- 快速验证模型完整性,无需完整的 kt-cli 环境

如果需要完整的交互式体验和自动修复功能,请使用 kt-cli 的 `kt model verify` 命令。
