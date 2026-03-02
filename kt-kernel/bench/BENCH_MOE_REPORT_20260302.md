# MoE Benchmark Report (2026-03-02)

## 1. 目标与结论

本次工作目标：
- 对齐 `bench_moe_torch.py` 与 `bench_moe_amx.py` 的实验口径（参数、线程、计时策略）。
- 在 torch 侧同时保留两类流程：`逐 expert` 和 `batched`（`batched_bmm`/`batched_einsum`）。
- 记录可复现实验命令与结果，并给出同设置下的性能差异。

同设置实测结论（本次记录参数见第 5 节）：
- BF16：KT-AMX 比 PyTorch `expert` 快约 `10.21x`，比 `batched_bmm` 快约 `4.09x`，比 `batched_einsum` 快约 `8.59x`。
- INT8：KT-AMX 比 PyTorch `qint8 expert` 快约 `19.48x`。

## 2. 为什么要这样改

原始对比存在 4 类不等效，导致结果不能直接做 apples-to-apples：
- 模型规模不一致（E/H/I/topk/layer）。
- 执行路径不一致（torch 逐 expert Python 循环 vs AMX 单一 C++ `forward_task`）。
- 量化流程计时不一致（前向内/加载阶段）。
- 线程与 NUMA 设置不一致。

对应改动原则：
- 同一组参数可直接传给两边脚本。
- 同一组线程参数可直接传给两边脚本。
- torch 既给出 `expert`，也给出 `batched`，方便你比较“流程差异对性能”的影响。
- 默认不把 torch `qint8` 输入量化计入测试循环（与 AMX “load 阶段量化”更接近的口径）。

## 3. 脚本改动摘要

### 3.1 `bench_moe_torch.py`
- 新增 CLI 参数：`--expert-num --hidden-size --intermediate-size --num-experts-per-tok --layer-num --qlen --warm-up-iter --test-iter --gen-iter --threads --interop-threads --modes --exec-paths --include-input-quant-time`
- 新增执行路径：
  - `expert`
  - `batched_bmm`
  - `batched_einsum`
- `qint8` 默认 `exclude_input_quant_time=True`（可通过 `--include-input-quant-time` 打开）。
- 统一在 CPU 上生成张量，不再依赖 `cuda -> cpu` 搬运。

### 3.2 `bench_moe_amx.py`
- 新增 CLI 参数：`--expert-num --hidden-size --intermediate-size --max-len --num-experts-per-tok --layer-num --qlen --warm-up-iter --test-iter --gen-iter --threads --subpool-count --interop-threads --quant-modes --no-progress`
- 统一线程配置：
  - 设置 `OMP_NUM_THREADS/MKL_NUM_THREADS`
  - 设置 `torch.set_num_threads` 和 `torch.set_num_interop_threads`
  - 根据 `--threads` 与 `--subpool-count` 自动拆分 `subpool_thread_count`
- 统一在 CPU 上生成张量，不再依赖 `cuda -> cpu` 搬运。
- 与 torch 一致的带宽/FLOPS 计算口径（按 `work_elems=H*I*qlen*3*topk`）。

## 4. 运行环境与解释器

- Python：`/mnt/data/lpl/anaconda3/envs/kt-ref/bin/python`
- 工作目录：`/home/lpl/kt-refactor/ktransformers`
- CPU：Intel Xeon Platinum 8488C（来自 AMX 运行记录）

## 5. 本次实际运行命令（可复现）

说明：大模型尺寸（如 `E=256,H=7168,I=2048`）在 torch 侧准备与执行耗时极高，本次报告采用可快速复现且完整跑通的一组参数。

### 5.1 Torch
```bash
/mnt/data/lpl/anaconda3/envs/kt-ref/bin/python -u /home/lpl/kt-refactor/ktransformers/kt-kernel/bench/bench_moe_torch.py \
  --expert-num 64 --hidden-size 1024 --intermediate-size 512 --num-experts-per-tok 4 \
  --layer-num 3 --qlen 1 --warm-up-iter 20 --test-iter 200 --gen-iter 512 \
  --threads 64 --interop-threads 1 \
  --modes bf16,qint8 --exec-paths expert,batched_bmm,batched_einsum
```

日志文件：
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_torch_run_20260302_093925.log`

### 5.2 AMX
```bash
/mnt/data/lpl/anaconda3/envs/kt-ref/bin/python -u /home/lpl/kt-refactor/ktransformers/kt-kernel/bench/bench_moe_amx.py \
  --expert-num 64 --hidden-size 1024 --intermediate-size 512 --num-experts-per-tok 4 \
  --layer-num 3 --qlen 1 --warm-up-iter 20 --test-iter 200 --gen-iter 512 \
  --threads 64 --subpool-count 2 --interop-threads 1 \
  --quant-modes bf16,int8 --no-progress
```

日志文件：
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_amx_run_20260302_093953.log`

## 6. 结果（同参数、同线程设置）

### 6.1 Torch
| quant | exec_path | time(s) | us/iter | bandwidth (GB/s) | flops (TFLOPS) |
|---|---|---:|---:|---:|---:|
| bf16 | expert | 0.33315430022776127 | 1665.7715 | 7.5538 | 0.0075538 |
| bf16 | batched_bmm | 0.13339894823729992 | 666.9947 | 18.8651 | 0.0188651 |
| bf16 | batched_einsum | 0.2802201397716999 | 1401.1007 | 8.9807 | 0.0089807 |
| qint8 | expert | 0.3401824776083231 | 1700.9124 | 3.6989 | 0.0073977 |

注：
- `qint8` 当前仅支持 `expert` 路径；`batched_bmm/einsum` 被脚本显式跳过。
- 本次 `qint8` 为 `Exclude input quantization time: True`。

### 6.2 AMX
| quant | time(s) | us/iter | bandwidth (GB/s) | flops (TFLOPS) |
|---|---:|---:|---:|---:|
| bf16 | 0.03262175153940916 | 163.1088 | 77.1443 | 0.0771443 |
| int8 | 0.017463482916355133 | 87.3174 | 72.0527 | 0.1441054 |

## 7. 性能差异（同设置）

### 7.1 BF16
- vs torch `expert`：`1665.77 / 163.11 = 10.21x`
- vs torch `batched_bmm`：`666.99 / 163.11 = 4.09x`
- vs torch `batched_einsum`：`1401.10 / 163.11 = 8.59x`

### 7.2 INT8
- vs torch `qint8 expert`：`1700.91 / 87.32 = 19.48x`

## 8. 对结果的解释边界

- 这组数字是“当前两套实现+当前脚本路径”的结果，不是数学意义上“纯内核指令级”单点结论。
- torch `qint8` 路径在本脚本中不保证就是 oneDNN-AMX int8 的最佳路径（脚本未强制 `quantized.engine='onednn'`）。
- AMX 日志中的 `From BF16 / online quant from bf16` 出现在加载阶段，不在测试循环里。
- 日志里的 `Failed to set thread name: Permission denied` 为环境权限噪声，不影响本次计时完成。

## 9. 附：本次保留的历史日志

以下为长任务尝试日志（保留供追溯）：
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_torch_run_20260302_092205.log`
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_torch_run_20260302_092807.log`
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_torch_run_20260302_093246.log`
- `/home/lpl/kt-refactor/ktransformers/kt-kernel/bench/moe_torch_run_20260302_093533.log`
