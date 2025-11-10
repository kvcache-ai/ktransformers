# Archive - Legacy KTransformers Code

This directory contains the original integrated KTransformers framework code that has been archived as part of the repository restructuring.

## 📋 What's Here

This archive preserves the complete original KTransformers implementation, including:

- **Core Framework** (`ktransformers/`): Original integrated inference framework
- **C/C++ Extensions** (`csrc/`): Low-level kernel implementations
- **Third-party Dependencies** (`third_party/`): Vendored external libraries
- **Git Submodules** (`.gitmodules`): Complete submodule configuration for legacy dependencies
- **Build System**: Installation scripts, Dockerfiles, and configuration files
- **Legacy Documentation**: Original README files with full quick-start guides

## 📚 Documentation

### Original README Files

- **[English README (Legacy)](./README_LEGACY.md)**: Complete original English documentation with:
  - Quick Start guides
  - Show cases and benchmarks
  - Injection tutorial
  - Full installation instructions

- **[中文 README (Legacy)](./README_ZH_LEGACY.md)**: 完整的原始中文文档，包含：
  - 快速入门指南
  - 案例展示和基准测试
  - 注入教程
  - 完整安装说明

## 🔄 Migration to New Structure

The KTransformers project has evolved into two focused modules:

### For Inference (CPU-optimized kernels):
→ Use **[kt-kernel](../kt-kernel/)** instead

### For Fine-tuning (LLaMA-Factory integration):
→ Use **[KT-SFT](../KT-SFT/)** instead

## ⚠️ Status

This code is **archived for reference only**. For active development and support:

- **Inference**: See [kt-kernel](../kt-kernel/)
- **Fine-tuning**: See [KT-SFT](../KT-SFT/)
- **Documentation**: See [doc](../doc/) directory
- **Issues**: Visit [GitHub Issues](https://github.com/kvcache-ai/ktransformers/issues)

## 🔧 Git Submodules (For Researchers)

The root `.gitmodules` only contains kt-kernel's dependencies to keep the repository lightweight. If you need to build the legacy code, you can use the archived submodule configuration:

```bash
# Copy the complete submodule configuration
cp archive/.gitmodules .gitmodules

# Initialize legacy submodules
git submodule update --init --recursive archive/third_party/
```

**Note**: This will download ~500MB of additional dependencies.

## 📦 Contents Overview

```
archive/
├── README.md              # This file
├── README_LEGACY.md       # Original English documentation
├── README_ZH_LEGACY.md    # Original Chinese documentation
├── .gitmodules            # Complete git submodule configuration (7 legacy submodules)
├── ktransformers/         # Original framework code
├── csrc/                  # C/C++ extensions
├── third_party/           # External dependencies (submodules not initialized by default)
├── setup.py               # Original installation script
├── pyproject.toml         # Python project configuration
├── Dockerfile*            # Container configurations
├── install*.sh            # Installation scripts
└── ...                    # Other legacy files
```

## 💡 Why Archived?

The original monolithic framework has been refactored into modular components for:

1. **Better Maintainability**: Separated concerns between inference and fine-tuning
2. **Easier Integration**: Cleaner APIs for external frameworks (SGLang, LLaMA-Factory)
3. **Focused Development**: Dedicated modules with specific optimization goals
4. **Reduced Complexity**: Smaller, more manageable codebases

## 🔗 Related Resources

- **Main Repository**: [../README.md](../README.md)
- **kt-kernel Documentation**: [../kt-kernel/README.md](../kt-kernel/README.md)
- **KT-SFT Documentation**: [../KT-SFT/README.md](../KT-SFT/README.md)
- **Project Website**: https://kvcache-ai.github.io/ktransformers/

---

<div align="center">
  <sub>Archived on 2025-11 as part of repository restructuring</sub>
</div>
