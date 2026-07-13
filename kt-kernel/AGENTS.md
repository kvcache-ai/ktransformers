# AGENTS.md: kt-kernel Full FT 开发指南

本文件作用于 `kt-kernel/` 及其全部子目录，用于指导 Agent 开发和验证 MoE
Full Fine-Tuning（Full FT）功能。重点是专家基座权重的梯度、optimizer 更新
和 AMX 权重重新量化，不讨论普通 GPU 参数训练。

当前开发快照：

- 仓库：`Illumination111/ktransformers-fullFT_development`
- Full FT 基线提交：`e99b5e1`，基于上游 `8e46e58`
- Full FT 来源：poryfly 的 `239bac5`、`25fa2fd`
- 当前状态：TP 子核配置、梯度分片/stride、临时区容量和逐步清零已修复；GDB 已将
  首步 SIGSEGV 定位为 base-weight backward 错用路由表，现改用 expert-major packed buffer；
  down 梯度为零进一步定位为 route-weighted `grad_output` 被 gate/up backward 临时区覆盖，
  现以独立只读快照保留，并用 NUMA 子线程池并行计算 AMX BF16 基座梯度

## 1. 开发前先确认

1. 在 `ktransformers` 中运行 `git status --short --branch`，不要覆盖已有改动。
2. 使用符号名定位代码，不依赖本文行号。
3. 先确认目标是 `full`、`hybrid` 还是 `lora`：
   - `full`：训练 expert 基座权重，通常 `lora_rank=0`。
   - `hybrid`：同时训练基座权重和 LoRA。
   - `lora`：不得计算或更新基座权重。
4. 修改前记录 TP 数量、NUMA 数量、`E/H/F` 和权重 dtype。

## 2. Full FT 数据契约

完整链路：

```text
KTConfig
  -> wrapper 创建 CPU BF16 nn.Parameter
  -> forward 将 Parameter 挂入 KTMoEFunction
  -> C++ backward 写 grad_*_proj_buf
  -> autograd 返回基座梯度
  -> optimizer 更新 *_proj_buf
  -> update_base_weights 重新量化到 AMX BufferB
```

权威数据必须唯一：

- `gate/up/down_proj_buf`：optimizer 可见的 CPU BF16 基座权重。
- `grad_gate/up/down_proj_buf`：C++ backward 的输出。
- AMX 量化权重：forward 使用的派生副本，optimizer 后必须重建。
- HF expert 原权重只是占位，不得作为 Full FT 的更新依据。

## 3. 文件职责

路径均相对本文件所在的 `kt-kernel/`。

| 路径 | 职责 |
|---|---|
| `python/sft/config.py` | 将 `full/hybrid` 映射为 `kt_full_weight_grad=True` |
| `python/sft/base.py` | 创建 `*_proj_buf`、`grad_*_proj_buf` |
| `python/sft/wrapper.py` | 包装 MoE、确定权威权重 |
| `python/sft/layer.py` | 将三个基座 Parameter 传入 autograd；dirty 时 requant |
| `python/sft/autograd.py` | 返回 C++ 写入的基座梯度 |
| `python/sft/lora.py` | optimizer 参数注入、分布式同步、dirty 标记 |
| `python/sft/amx.py` | 设置 C++ config、传梯度指针、触发重新量化 |
| `operators/common.hpp` | `MOESFTConfig.full_weight_grad` |
| `operators/moe-tp.hpp` | 通用 TP 切分；当前会把 SFT config 切成基类 |
| `operators/moe-sft-tp.hpp` | SFT TP 调度和全局梯度 buffer 分片 |
| `operators/amx/sft_moe.hpp` | NUMA 子核 backward 和基座梯度计算 |
| `ext_bindings.cpp` | Python/C++ 参数和 task 绑定 |

## 4. TP 张量布局

定义：

- `E`：expert 数量。
- `H`：hidden size。
- `F`：完整 `intermediate_size`。
- `I`：当前 TP 子核的本地 `intermediate_size`。
- `tp_offset`：当前 TP 在完整 `F` 维上的起点。

完整梯度布局：

```text
gate/up: [E, F, H]
down:    [E, H, F]
```

TP wrapper 传给子核的起始指针：

```text
gate/up = base + tp_offset * H
down    = base + tp_offset
```

子核循环和 FP32 累加使用本地 `I`，但写回完整张量时必须使用 `F`
作为 expert/row stride：

```text
gate/up: expert * F * H + local_i * H + h
down:    expert * H * F + h * F + local_i
```

所有 TP 必须写入互不重叠的区域；对空指针不得做偏移运算。

## 5. 历史根因与失败基线

`full_weight_grad=True` 已正确到达顶层 `TP_MOE_SFT`，但
`TP_MOE` 创建子核时执行：

```cpp
GeneralMOEConfig tp_config = config;
```

这会丢失 `MOESFTConfig.full_weight_grad`。子核随后通过
`MOESFTConfig(GeneralMOEConfig)` 重建配置，字段回到默认 `false`，
因此 `backward_base_weight_grad` 被门控跳过。

只恢复该布尔值仍不够：当前 TP backward 把相同的完整梯度首地址传给
所有子核，子核又按本地 `I` 写回，会导致覆盖和错误布局。

修复前的 `20260710_103230_1gpu_AMX_BF16_FULLFT_EXPERTONLY` run 已冻结
其他权重并只向 optimizer 注入 144 个 expert 基座参数，但 15 步
`grad_norm=0`，step 0 到 step 15 的 probe 为 `changed=0/12`、
`max_abs_delta=0`。720 次 requant 不能证明权重更新。

修复后的 `20260710_192456_1gpu_AMX_BF16_FULLFT_EXPERTONLY` run 未进入
第一个训练 step。`phase4/train.log` 记录训练子进程 PID 3296907 在
`2026-07-10 19:27:33 +08:00` 以 `exitcode=-11` 退出，并明确报告
`Signal 11 (SIGSEGV)`；外层 `accelerate` 的 `exit_code.txt=1` 只是 launcher
退出码。`phase4/log_analysis.txt` 正确识别到崩溃，但旧 `summary.md` 的
“未检测到崩溃”是汇总脚本误判，不得作为反证。该 run 没有 C++ backtrace、
源码行号或 core，因而“C++ 梯度索引越界”目前只是诊断假设，并非已确认根因。

带符号 GDB 的 `20260713_105328_1gpu_AMX_BF16_FULLFT_EXPERTONLY` run
确认根因位于 `backward_base_weight_grad`：`m_local_pos_cache` 的真实布局是
`[token_idx][route_slot]`，内层长度为 `k=8`，旧代码却按
`m_local_pos_cache[expert_idx][t]` 访问。首个 expert 执行到 `t=8` 时越界，
读出垃圾 `tok_pos=81349952`，最终在读取 `input_row[0]` 时于
`operators/amx/sft_moe.hpp:1968` 触发 SIGSEGV。两个 NUMA 子核均进入同一错误路径。
最小修复不再反查路由表，而是直接使用 backward 已恢复/生成的
`m_local_input_ptr_[expert_idx]` 和 `grad_output_bf16_ptr_[expert_idx]`；后者已包含
router weight，因而同时保证 down projection 基座梯度的权重语义正确。

## 6. 当前两文件实现

生产代码限制在两个文件，未修改 Python 接口。

### 6.1 `operators/amx/sft_moe.hpp`

1. `set_full_weight_grad(bool)` 更新子核的
   `sft_config_.full_weight_grad`。
2. `full_intermediate_size` 已传给 `backward_base_weight_grad`。
3. 计算循环使用本地 `I`，写回 stride 使用完整 `F`。
4. 门控要求开关为 true 且三个梯度指针均非空。
5. `forward_pool_` 保证至少 `3 * I * H * sizeof(float)` 字节。
6. `lora_rank=0` 时 scaling 明确为 0，避免纯 Full FT 产生 Inf。
7. 基座梯度读取 expert-major packed input/grad-output，不把
   `m_local_pos_cache[token][route]` 误当作 expert token 列表。
8. down 基座梯度读取独立的 route-weighted `grad_output` 快照；工作用
   `grad_output_bf16_ptr_` 后续可继续被 gate/up grad-input 路径复用。
9. AMX BF16 基座梯度按 expert、projection 和 `32x32` 输出 tile 提交给当前 NUMA
   subpool。gate/up 共用输入 tile，AMX BF16 乘法以 FP32 累加，每个任务独占输出区域。
10. `lora_rank=0` 在完成 gate/up 基座 grad-input 后直接跳过 LoRA remainder，避免纯 Full FT
    进入零秩 LoRA 临时区路径。

### 6.2 `operators/moe-sft-tp.hpp`

1. 子核创建后逐个调用 `set_full_weight_grad`。
2. 传播逻辑位于 `if constexpr (!kSkipLoRA)` 外；纯 Full FT 的
   `lora_rank=0` 也必须执行。
3. backward dispatch 前按第 4 节公式生成每个 TP 的三个梯度指针。
4. 完整 `F` 传给子核作为全局 stride，并检查所有 TP slice 完整覆盖 `F`。
5. TP dispatch 前统一并行清零三组完整梯度，避免跨 step 残留和 TP 清零竞争。

暂不修改通用 `operators/moe-tp.hpp`。保留派生 config 的泛型重构影响
所有 MoE backend，应作为后续独立改动。

## 7. 验证顺序

当前已通过 `clang-format --dry-run --Werror`、AMX/CUDA Release
`build_ext --inplace`，以及 AMX BF16 `32x32x37` 分块乘法与标量 BF16 参考的独立校验
（`max_abs_error=0`）。端到端短训练仍须确认 gate/up/down 三组梯度均非零且有限。
但当前 Kllama 环境中的 `kt_kernel_ext` 已 stripped，
普通 GDB 只能得到地址或有限符号；定位 SIGSEGV 前应使用相同 Python 环境重新构建
并安装带符号的 `RelWithDebInfo` 版本：

```bash
cd /mnt/data2/wbw/ktransformers
CPUINFER_BUILD_TYPE=RelWithDebInfo \
  /mnt/data2/wbw/conda/envs/Kllama/bin/python3.12 \
  kt-kernel/setup.py build_ext --inplace
CPUINFER_BUILD_TYPE=RelWithDebInfo \
  /mnt/data2/wbw/conda/envs/Kllama/bin/python3.12 -m pip install \
  --no-build-isolation --no-deps --force-reinstall ./kt-kernel
```

确认测试实际 import 的 `.so` 含 `.debug_info`/`.debug_line`。然后使用 expert-only
runner 的 `--gdb`；只有磁盘空间足够时才使用 `--gdb-core`。修复后的 expert-only
短训练仍是必需验收项。

1. **静态检查**：确认 setter 不在 LoRA 条件分支内；确认所有指针先判空再偏移。
2. **构建**：重新编译并安装 `kt_kernel_ext`。
3. **参考梯度测试**：
   - 分别覆盖 TP=1 和 TP=2。
   - 使用 `expert_id > 0`，避免错误 expert stride 被 expert 0 掩盖。
   - 与 PyTorch outer-product 参考值比较 gate/up/down 三组梯度。
   - TP=2 时分别检查两个 `F` 分片，确认无覆盖。
4. **跨 step 测试**：连续两步激活不同 expert，确认未激活 expert 不保留旧梯度。
5. **模式回归**：`full_weight_grad=false` 不写基座梯度；纯 LoRA 结果不变。
6. **短训练**：
   - step 内三个 `grad_*_proj_buf.abs().max() > 0`。
   - optimizer 后至少一个 `*_proj_buf` 发生有限变化。
   - requant 后下一次 forward 使用更新后的权重。

不要用 loss 下降证明 Full FT 生效；attention、router 或其他非 expert 参数也能
让 loss 下降。

## 8. 完成标准

- C++ 子核实际收到 `full_weight_grad=true`。
- 三组基座梯度非零、有限，并与参考实现一致。
- TP 分片覆盖完整 `F`，无重叠、越界或错误 stride。
- optimizer 能找到三个基座 Parameter，并在 step 后更新它们。
- inactive expert 无跨 step 残留梯度。
- `full`、`hybrid`、`lora` 三种模式行为符合各自契约。
- 纯 LoRA 和现有 AMX MoE 测试无回归。

## 9. 调试资料

```text
FFTtest/Qwen3-30B-A3B/test_log/20260710_103230_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  summary.md
  expert_weight_change_check.{txt,json}
  phase4/expert_buf_probe.json
  phase4/train.log

FFTtest/Qwen3-30B-A3B/test_log/20260710_192456_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  phase4/train.log          # exitcode=-11，Signal 11 (SIGSEGV)
  phase4/log_analysis.txt   # 正确检测到 SIGSEGV
  phase4/exit_code.txt      # 外层 launcher 退出码 1
  summary.md                # P5 旧结论错误，不得作为崩溃判据

FFTtest/Qwen3-30B-A3B/test_log/20260713_105328_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  phase4/gdb_sigsegv.log    # t=8 越界、非法 tok_pos 和双 NUMA 原生栈
  phase4/train.log
  summary.md

FFTtest/Qwen3-30B-A3B/expert_buf_probe.py
FFTtest/Qwen3-30B-A3B/gdb_sigsegv.gdb
FFTtest/Qwen3-30B-A3B/run_full_ft_test_1gpu_bf16_frozen.sh --gdb
```

该 run 的直接证据是 `changed=0/12`、`max_abs_delta=0`；训练 loss
仍下降不能否定 expert 基座梯度失效。后续 run 的直接崩溃证据必须以原始
`phase4/train.log` 和 `phase4/gdb_sigsegv.log` 为准，不能只依赖生成的 summary。
