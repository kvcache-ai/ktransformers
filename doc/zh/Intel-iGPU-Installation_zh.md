# Intel iGPU XMX 源码安装

本页仅介绍源码安装。支持范围、完整启动命令和精度边界见 [使用文档](Intel-iGPU_zh.md)。
本方案需要 NVIDIA GPU 执行 Attention 等算子，同时需要 Intel GPU 的 Level Zero 驱动和 SYCL 运行时。

## 已验证环境

Core Ultra 7 356H / Intel `b0a0`、RTX 5060 Laptop 8GB、约30GiB可用物理内存，
Python 3.11、oneAPI DPC++ 2026.1.1、Intel compute runtime 25.18.33578.77、
Level Zero loader 1.24.0、CUDA Toolkit 13.2、PyTorch 2.9.1、Triton 3.5.1，
Linux 7.0.0-38-generic。这是实测版本记录，不表示任意版本组合都已验证。

构建前需要安装：

- 支持本机 Intel GPU 的内核和 Level Zero 驱动，参照 Intel 的
  [GPU 驱动安装说明](https://dgpu-docs.intel.com/driver/installation.html)。
- 包含 `icpx`、SYCL runtime、`sycl-ls` 的 Intel oneAPI DPC++ 编译器。
- 可正常使用 NVIDIA GPU 的 CUDA Toolkit 和 CUDA 版 PyTorch。
- 源码构建依赖：Git、CMake、Ninja、C/C++工具链、hwloc和NUMA开发库。

Ubuntu可先安装系统开发依赖：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build git libnuma-dev libhwloc-dev pkg-config
```

在独立 Python 3.11 环境中检查两类 GPU：

```bash
source /opt/intel/oneapi/setvars.sh --force
icpx --version
sycl-ls
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
ls -l /dev/dri/renderD*
groups
```

`sycl-ls` 应列出 Intel Level Zero GPU，当前用户应能读写对应 render node。
若需加入 `render` 用户组，请按发行版配置并重新登录。`ONEAPI_DEVICE_SELECTOR` 用于选择 SYCL
设备，NVIDIA设备仍由CUDA管理。这里假定当前Python环境已按项目要求安装CUDA版PyTorch。

## 通过官方安装入口从源码安装

激活 Python 环境后，在源码目录运行根目录的 `install.sh`：

```bash
git clone --recursive https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
# 切换到包含本后端的提交后运行：
./install.sh --sycl
```

`--sycl` 选择 Intel SYCL 构建，安装脚本会完成以下步骤：

1. 初始化仓库子模块并检查系统依赖；已安装依赖时不重复调用包管理器。
2. 加载默认位置的 oneAPI 环境，使用 `icpx` 构建本机的 AVX2+SYCL 内核。
3. 使用已验证的构建设置：关闭 AMX/AVX512、MLA、LTO 和 crash tracing，编译并行度默认为2。
4. 在构建目录副本中应用已有 llama.cpp 补丁，保留原子模块 checkout。
5. 构建并安装本仓库的 kt-kernel，再从固定公开 SGLang 源码安装推理服务。

Linux 源码构建会将所选 oneAPI 编译器的运行库目录写入扩展的运行时搜索路径，包括
Intel OpenMP 和 SYCL 库。因此，完成安装后新终端中的普通 `import kt_kernel` 不应依赖
安装脚本内的临时 `LD_LIBRARY_PATH`。oneAPI 仍需保留在构建时的位置；启动命令中的
`source .../setvars.sh` 用于初始化完整工具链环境，不会将 oneAPI 打包进扩展。
安装器还会清除该环境变量并启动新 Python 进程检查扩展能否加载，避免安装时的临时环境掩盖问题。

不传 `--sycl` 时保留原来的默认安装流程。

如需指定非默认 oneAPI 路径，可设置 `ONEAPI_SETVARS`。如果确实需要调整编译并行度，
可显式设置 `CPUINFER_PARALLEL`，否则保持默认即可。这些是高级覆盖项，不是安装前提。
已经显式启用其它互斥 GPU 构建后端的终端需先取消相应设置，安装器会明确提示冲突。

替换原生扩展前停止模型服务；安装后重启 Python 加载新二进制。`./install.sh kt-kernel --sycl`
可只重建内核；首次安装使用上面的完整入口。

## 验证安装

```bash
source /opt/intel/oneapi/setvars.sh --force
python -c 'from kt_kernel import kt_kernel_ext as e; print(e.__file__); assert e.moe.SYCLXMXGPTQInt4_MOE.implementation_revision() >= 7'
git submodule status
python -m pip check
```

记录实际加载的扩展路径和hash，并运行使用文档中的硬件检查；仅能import不能证明完整功能正常。
安装后按照[完整启动命令](Intel-iGPU_zh.md#完整启动命令)启动服务。

缺少Intel设备时先处理驱动和权限；缺少SYCL共享库时加载构建所用oneAPI；
缺少XMX类或revision过旧时检查当前Python环境和`.so`路径。
模型需要对称GPTQ INT4、`desc_act=false`和满足要求的group及维度，不能直接传入FP8/BF16权重。
记录内存和swap状态，保持已验证的chunk及请求上限；全量logprob请求可能比正常生成占用更多显存。

如果旧版构建报 `libiomp5.so: cannot open shared object file`，可先在运行推理的同一终端执行
`source /opt/intel/oneapi/setvars.sh --force`。安装脚本作为子进程运行，不能修改父终端的环境。
更新到包含运行库路径修复的源码后，执行 `./install.sh kt-kernel --sycl --no-clean` 重建安装；
无需手动复制或软链接运行库。若 oneAPI 已移动或被卸载，则恢复相应工具链或重新构建。
