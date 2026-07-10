# AGENTS.md: kt-kernel Full FT 开发指南

本文件作用于 `kt-kernel/` 及其全部子目录，用于指导 Agent 开发和验证 MoE
Full Fine-Tuning（Full FT）功能。重点是专家基座权重的梯度、optimizer 更新
和 AMX 权重重新量化，不讨论普通 GPU 参数训练。

当前开发快照：

- 仓库：`Illumination111/ktransformers-fullFT_development`
- Full FT 基线提交：`e99b5e1`，基于上游 `8e46e58`
- Full FT 来源：poryfly 的 `239bac5`、`25fa2fd`
- 当前状态：TP 子核配置、梯度分片/stride、临时区容量和逐步清零已修复；等待修复后训练验证

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

当前已通过 `clang-format --dry-run --Werror` 和 AMX/CUDA Release
`build_ext --inplace`。修复后的 expert-only 短训练仍是必需验收项。

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

FFTtest/Qwen3-30B-A3B/expert_buf_probe.py
```

该 run 的直接证据是 `changed=0/12`、`max_abs_delta=0`；训练 loss
仍下降不能否定 expert 基座梯度失效。
