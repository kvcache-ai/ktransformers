# AGENTS.md：kt-kernel MoE Full FT 开发、差异与调试记录

本文件作用于 `kt-kernel/` 及其全部子目录，用于说明个人 Full Fine-Tuning
（Full FT）分支相对官方仓库的真实代码差异，并保留从“专家权重不更新”到
“SIGSEGV、down 梯度为零、性能过低”这一系列问题的试错记录。

这里的 Full FT 专指 KTransformers CPU/AMX MoE expert 基座权重
`gate_proj`、`up_proj`、`down_proj` 的训练，不把普通 GPU attention、router、
embedding 等参数的更新当作 expert Full FT 已生效的证据。

## 1. 仓库、基线与比较口径

### 1.1 分支关系

截至 2026-07-13，本地比较对象为：

- 官方仓库：`kvcache-ai/ktransformers`，remote 名为 `upstream`。
- 个人仓库：`Illumination111/ktransformers-fullFT_development`，remote 名为
  `origin`。
- 个人开发分支：`fullft-development`；Full-FT 核心生产代码截止 `5ce0767`，
  后续纯文档提交不改变该代码基线。
- 共同基线：官方提交 `8e46e58`。
- Full FT 初始快照：`e99b5e1`，来源包含 poryfly 的 `239bac5`、`25fa2fd`。
- 后续修复：`2d81e86`、`20f645c`、`5ce0767`。

官方 `origin/main` 当时已前进到 `7c021b4`，比共同基线多 3 个与本轮 Full FT
基本无关的提交：

```text
79b265b  normalize compressed RAWINT4 weights
cb9f47d  detect bound ports before launch
7c021b4  bump sglang submodule
```

因此，统计“个人做了什么”时必须使用共同基线或三点 diff：

```bash
git merge-base origin/main fullft-development
git diff --shortstat origin/main...fullft-development
git diff --numstat origin/main...fullft-development
```

不要直接把 `git diff origin/main fullft-development` 的全部结果都归到个人修改；
那样会把个人分支尚未合入的 3 个官方提交也统计为差异。

### 1.2 修改规模

个人 Full FT 分支相对共同基线 `8e46e58`：

| 口径 | 文件数 | 新增 | 删除 | 说明 |
|---|---:|---:|---:|---|
| 当前工作树（含本次扩写后的文档） | 16 | 1581 | 160 | 全部位于 `kt-kernel/` |
| 仅生产代码 | 15 | 1036 | 160 | 总代码 churn 约 1196 行 |
| Python SFT 接入 | 11 | 658 | 143 | 约占生产代码改动的三分之二 |
| C++ config/binding/AMX/TP | 4 | 378 | 17 | 约占生产代码改动的三分之一 |

应区分两个阶段：

1. `e99b5e1` 是完整 Full FT 功能接入，涉及 15 个代码文件，`+809/-155`，
   修改面较广。
2. 后续 3 个 debug/fix 提交的生产代码只修改两个文件，合计约 `+255/-33`：
   `operators/amx/sft_moe.hpp` 和 `operators/moe-sft-tp.hpp`。

提交 `5ce0767` 时旧版本文档为 251 行；本次仅扩写说明和试错记录，没有改变上述
15 个生产代码文件的统计。

因此，“个人 Full FT 分支总体改动”属于中等偏大；“本轮 SIGSEGV/down 梯度/
AMX 性能 debug”则高度集中在两个 `.hpp` 文件。

## 2. 官方代码与个人 Full FT 代码的行为差异

### 2.1 官方基线的限制

官方基线的 SFT expert 路径以 LoRA 为主要训练对象：

- 没有 `full/hybrid/lora` 的明确 KT train-mode 映射。
- `lora_rank` 必须为正数，纯 Full FT 的 `lora_rank=0` 不成立。
- expert 原始权重只用于加载和 AMX forward，不是 optimizer 可见的
  `nn.Parameter`。
- Python autograd 不把三个 expert 基座权重作为输入，也不返回对应梯度。
- C++ `MOESFTConfig` 没有 `full_weight_grad` 和三个基座梯度指针。
- C++ backward 不会把 gate/up/down 基座梯度写回 Python tensor。
- optimizer 更新后没有把 BF16 expert 基座权重重新量化到 AMX BufferB 的闭环。

所以，仅在 LLaMA-Factory 中设置 `finetuning_type: full`，不能证明 KT CPU experts
正在 Full FT；attention/router 等 GPU 参数仍可能让 loss 下降。

### 2.2 个人分支建立的完整数据链路

```text
LLaMA-Factory finetuning_type
  -> KTConfig.kt_train_mode / kt_full_weight_grad
  -> wrapper 创建 CPU BF16 gate/up/down nn.Parameter
  -> KTMoEFunction.forward 显式接收三个 Parameter
  -> C++ backward 写 grad_gate/up/down_proj_buf
  -> KTMoEFunction.backward 返回三个基座梯度
  -> Trainer optimizer 注入并更新三个 *_proj_buf
  -> optimizer.step 后标记 _base_weights_dirty
  -> 下一次 forward 调用 update_base_weights
  -> BF16 基座权重重新量化到 AMX BufferB
```

权威数据定义：

- `gate_proj_buf`、`up_proj_buf`、`down_proj_buf`：optimizer 可见的 CPU BF16
  expert 基座权重。
- `grad_gate_proj_buf`、`grad_up_proj_buf`、`grad_down_proj_buf`：C++ backward
  写入、autograd 返回的基座梯度。
- AMX 量化权重：forward 使用的派生副本，optimizer 后必须重建。
- HF model tree 中的 expert 权重：清理为 zero-storage placeholder，不再是 Full FT
  的权威副本，也不能用于训练前后权重比较。

### 2.3 模式语义

- `full`：训练 expert 基座权重，通常 `lora_rank=0`。
- `hybrid`：同时训练 expert 基座权重和 LoRA，允许 `lora_rank>0`。
- `lora`：只训练 LoRA，不得计算、同步或更新 expert 基座权重。

修改任何共享路径前必须分别确认三种模式，不能用 Full FT 修复破坏 LoRA。

## 3. 逐文件说明：个人分支具体修改了什么

路径均相对 `kt-kernel/`。

### 3.1 Python 训练接入层

| 文件 | 相对官方的主要修改 |
|---|---|
| `python/sft/config.py` | 增加 `kt_train_mode`、`kt_full_weight_grad`；读取 `ACCELERATE_KT_TRAIN_MODE`，将 `full/hybrid` 映射为基座梯度开启。 |
| `python/experts.py` | `KTMoEWrapper` 和 SFT wrapper factory 接受并传递 `full_weight_grad`。 |
| `python/sft/wrapper.py` | 将 LLaMA-Factory `finetuning_type` 映射到 KT train mode；把 `lora_rank=0` 当作合法纯 Full FT；初始化三个权威 BF16 Parameter/grad buffer；释放重复权重；把 HF expert 权重替换为 zero-storage placeholder。 |
| `python/sft/base.py` | 保存 `_full_weight_grad` 和 `_base_weights_dirty`；创建 `[E,F,H]` gate/up 与 `[E,H,F]` down Parameter/grad buffer；允许 Full FT 在无 LoRA 时运行；定义 `update_base_weights()` 接口；避免 `lora_rank=0` 除零。 |
| `python/sft/layer.py` | Full FT 时强制进入 autograd 路径；将三个基座 Parameter 传入 `KTMoEFunction`；optimizer 后发现 dirty 权重时触发 requant；Full FT 时保留 router 梯度；兼容 transformers v5 TopKRouter/GLM4 路由。 |
| `python/sft/autograd.py` | forward 增加三个 expert 基座 Parameter 输入；backward 返回 C++ 写入的 gate/up/down 梯度，使 PyTorch 给 Parameter 建立 `.grad`。 |
| `python/sft/lora.py` | 拆分 LoRA 参数与 Full-FT 基座参数收集；把 48 层 × 3 投影共 144 个 Parameter 注入 optimizer；纯 Full FT 跳过 LoRA buffer；分布式时同步基座梯度；optimizer 后标记基座权重 dirty。 |
| `python/sft/amx.py` | 将 Full-FT 开关和三个梯度 data pointer 写入 C++ config/backward task；`lora_rank=0` 时传空 LoRA 指针；增加 `update_base_weights()`，优先复用 C++ 对象并重新量化，缺少 binding 时才完整重建。 |
| `python/sft/weights.py` | 明确 `*_proj_buf` 为 Full FT 权威权重；清理 model tree 中的冗余 expert 参数并打 `_kt_zero_storage` 标记，避免重复计数和重复占内存。 |
| `python/sft/arch.py` | 增加 GLM4 MoE 架构识别；这属于同期兼容性修改，不是本次 Qwen3 Full FT bugfix 的核心。 |
| `python/sft/__init__.py` | 导出新增的 Full-FT 参数收集和相关 SFT API。 |

### 3.2 C++ 配置与绑定

| 文件 | 相对官方的主要修改 |
|---|---|
| `operators/common.hpp` | 在 `MOESFTConfig` 中增加 `full_weight_grad` 和 `grad_gate/up/down_proj` 三个零拷贝指针。 |
| `ext_bindings.cpp` | backward binding 接收并转发三个基座梯度指针；向 Python 暴露 Full-FT config 字段；增加 `set_base_weight_pointers()`，支持 optimizer 后复用既有 C++ MoE 对象并 requant。 |

### 3.3 两个核心 `.hpp`

#### `operators/moe-sft-tp.hpp`

该文件负责顶层 TP 调度和完整梯度 tensor 的切片：

1. 子核创建后显式调用 `set_full_weight_grad()`，补回派生 config 被
   `GeneralMOEConfig` slicing 丢失的字段。
2. 传播逻辑移到 `if constexpr (!kSkipLoRA)` 外，确保纯 Full FT 的
   `lora_rank=0` 仍启用基座梯度。
3. backward dispatch 前按 TP offset 计算 gate/up/down 三个 slice 指针。
4. 将完整 `F` 传给子核作为 global stride，而不是误用当前 TP 的本地 `I`。
5. 检查所有 TP slice 对完整 `F` 的覆盖，无重叠、无缺口。
6. dispatch 前统一并行清零三组完整梯度，避免跨 step 残留和多个 TP 子核竞争清零。
7. 空指针必须先判断再做 offset，避免对 `nullptr` 做未定义指针运算。

#### `operators/amx/sft_moe.hpp`

该文件负责每个 NUMA/TP 子核的真实 backward 和基座梯度计算：

1. 保存并更新 `sft_config_.full_weight_grad`。
2. `backward_base_weight_grad` 同时接收本地 `I` 和完整 `F`；本地计算用 `I`，
   写回 expert/row stride 用 `F`。
3. gate/up/down 三组指针全部有效时才计算基座梯度。
4. 扩大/复用工作区，保证基座 FP32 累加需要的容量。
5. `lora_rank=0` 时 scaling 为 0，完成 gate/up grad-input 后跳过 LoRA remainder，
   不进入零秩临时区路径。
6. 不再把 token-major 路由表当作 expert-major token list；直接读取 backward 已
   打包好的 `m_local_input_ptr_[expert]` 和 expert-major grad-output。
7. 为 down 梯度保存独立、只读、已经乘过 router weight 的 `grad_output` 快照；
   gate/up grad-input 可以继续复用工作 buffer，但不能覆盖 down 所需的 dY。
8. 使用 AMX BF16 `32x32` output tile、FP32 累加计算基座梯度。
9. 任务按 expert、projection、tile 投递给当前 NUMA subpool，而不是在每个 NUMA
   节点只用单核执行三重标量循环。
10. gate/up 共用输入 tile，每个任务写独占输出区域，避免锁和写竞争。

暂不修改通用 `operators/moe-tp.hpp`。将其改成保留派生 config 的泛型重构会影响
所有 MoE backend，应另开改动并做完整回归。

## 4. TP 梯度布局契约

定义：

- `E`：expert 数量。
- `H`：hidden size。
- `F`：完整 intermediate size。
- `I`：当前 TP 子核的本地 intermediate size。
- `tp_offset`：当前 TP slice 在完整 `F` 维的起点。

完整梯度布局：

```text
gate/up: [E, F, H]
down:    [E, H, F]
```

顶层 TP wrapper 传给子核的起始指针：

```text
gate/up = base + tp_offset * H
down    = base + tp_offset
```

子核写回公式：

```text
gate/up: expert * F * H + local_i * H + h
down:    expert * H * F + h * F + local_i
```

易错点：

- 计算循环范围是本地 `I`，但 expert/row stride 必须是完整 `F`。
- 所有 TP 必须写入互不重叠的区域。
- `expert_id=0` 可能掩盖错误 expert stride，参考测试必须包含 `expert_id>0`。
- 梯度清零只能由顶层统一完成，不能让多个 TP 子核同时清相同完整 buffer。

## 5. 修改与试错时间线

本节保留失败路径，因为这些失败揭示了仅看 loss、requant 日志或单步权重抽检会得出
错误结论。

### 5.1 阶段 0：官方路径不能形成 expert Full FT 闭环

最初仅设置 LLaMA-Factory `finetuning_type=full`，但 KT expert 权重不是 optimizer
可见 Parameter，C++ 也没有基座梯度输出。loss 变化最多说明其他参数在训练，不能说明
CPU experts 更新。

结论：必须建立 Parameter → C++ grad → autograd → optimizer → requant 的完整闭环。

### 5.2 阶段 1：`e99b5e1` 接入 Full FT，但专家权重仍不更新

`e99b5e1` 加入 Python/C++ Full-FT 数据链路。随后运行：

```text
20260710_103230_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

测试已冻结非 expert 参数，并向 optimizer 注入 144 个 expert 基座 Parameter，但结果为：

- 15 步 `grad_norm=0`。
- step 0 到 step 15 抽检 `changed=0/12`。
- `max_abs_delta=0`。
- 有 720 次 requant 日志。

试错结论：

- optimizer 参数数量正确，不等于 C++ 真正产生了梯度。
- requant 被调用，不等于其输入权重发生了变化。
- 问题继续下沉到 C++ 子核配置和 TP 梯度写回。

根因之一是顶层 `TP_MOE_SFT` 收到 `full_weight_grad=true` 后，创建子核时把派生配置
赋给 `GeneralMOEConfig`，发生 slicing；子核重建 `MOESFTConfig` 后开关恢复为 false，
`backward_base_weight_grad` 被静默跳过。

### 5.3 阶段 2：`2d81e86` 修复开关/TP stride 后出现 SIGSEGV

`2d81e86` 尝试修复：

- 将 `full_weight_grad` 显式传播到子核。
- 给每个 TP 传独立梯度 slice。
- 使用完整 `F` 作为全局 stride。
- 扩大临时区并统一清零。

但测试：

```text
20260710_192456_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

在第一个训练 step 前后发生 SIGSEGV：

- child `exitcode=-11`，原始日志明确记录 `Signal 11 (SIGSEGV)`。
- 外层 `exit_code.txt=1` 只是 accelerate launcher 退出码。
- 当时的自动 `summary.md` 错误写成“未检测到崩溃”，不能作为反证。
- 该 run 没有 C++ backtrace，最初只能假设是 gradient index/stride 越界。

失败原因：修复了“是否计算”和“写到哪里”，但新实现为了计算基座梯度又错误理解了
路由缓存布局。

### 5.4 阶段 3：加入带符号 GDB，定位错误路由表解释

为避免继续猜测，在 FFTtest runner 中增加 batch GDB，并用同一 Kllama Python 环境
构建 `RelWithDebInfo` 扩展。日志：

```text
20260713_105328_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

GDB 证据：

- `m_local_pos_cache` 的真实布局是 `[token_idx][route_slot]`。
- 每个 token 的 route 长度是 `k=8`。
- 旧代码却按 `m_local_pos_cache[expert_idx][t]` 把它当作 expert token list。
- 第一个 expert 执行到 `t=8` 即越界，读出垃圾 `tok_pos=81349952`。
- 最终在读取 `input_row[0]` 时于当时的 `sft_moe.hpp:1968` 触发 SIGSEGV。
- 两个 NUMA 子核都进入了相同错误路径。

这一步推翻了“只是 TP output stride 错误”的单一假设。崩溃发生在读取输入行，而不是
最终写回梯度的位置。

### 5.5 阶段 4：`20f645c` 最小修复消除 SIGSEGV，但单步测试无效

`20f645c` 不再反查 token-major 路由表，改用 backward 已生成的 expert-major packed
buffer：

```text
m_local_input_ptr_[expert_idx]
grad_output_bf16_ptr_[expert_idx]
```

随后单步测试：

```text
20260713_111729_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

结果：

- GDB 显示进程正常退出，无 SIGSEGV。
- backward 约 890.9 秒。
- 权重抽检仍是 `changed=0/12`。

这里不能得出“梯度仍无效”的结论，因为只运行 1 step，而第一个 optimizer step 的
learning rate 是 0。这个 run 证明了崩溃消失，但不能验证权重更新。

试错教训：至少需要 2 个 optimizer step，最好 3～5 步；必须同时记录每步 LR。

### 5.6 阶段 5：三步测试证明 gate/up 更新，但暴露 down 梯度为零

继续运行：

```text
20260713_120636_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

结果：

- 3 个 step 正常退出，无 SIGSEGV。
- 权重抽检 `changed=5/12`，证明 Full-FT optimizer 链路总体已经能更新 expert。
- gate 梯度 `48/48` 非零。
- up 梯度 `48/48` 非零。
- down 梯度 `0/48`，所以总计只有 `96/144` 非零。
- backward 平均约 754.5 秒/step。
- 梯度最大值达到 `4e16～9e16`，虽有限但数值明显异常。

为了区分“C++ buffer 有值但 autograd/optimizer 看不到”和“C++ 本身没有写值”，测试脚本
增加了对 `grad_*_proj_buf` 与独立 `Parameter.grad` 的全量扫描。二者的非零数量和最大值
一致，说明 down=0 发生在 C++ 计算链路，而不是 Python autograd 丢梯度。

进一步检查发现，down 基座梯度需要的 route-weighted dY 原本位于
`grad_output_bf16_ptr_`，但 gate/up grad-input backward 随后复用了同一 buffer，down 计算
时读到的内容已被覆盖。

失败方案：简单地把 `grad_output_bf16_ptr_` 当作长期只读 down dY。该指针只是工作 buffer，
生命周期不满足要求。

### 5.7 阶段 6：`5ce0767` 保存 down 快照并用 AMX/NUMA 并行

`5ce0767` 同时处理正确性和性能：

- 在工作 buffer 被 gate/up 路径复用前，保存 route-weighted dY 的独立只读快照。
- down 基座梯度只读取该快照。
- gate/up/down 基座梯度改用 AMX BF16 `32x32` tile、FP32 累加。
- 任务提交给每个 NUMA 节点已有的 subpool，而不是每节点单核标量循环。
- 纯 Full FT `lora_rank=0` 跳过 LoRA remainder。

验证日志：

```text
20260713_140101_1gpu_AMX_BF16_FULLFT_EXPERTONLY
```

结果：

- 5/5 step 正常退出，无 SIGSEGV。
- C++ grad buffer `144/144` 非零，`Parameter.grad` 也是 `144/144` 非零。
- gate/up/down 分别为 `48/48`、`48/48`、`48/48`。
- 权重抽检 `changed=9/12`；其中 down 抽检 `4/4` 全部变化。
- 最大权重差约 `3.8147e-05`，没有非有限权重。
- backward 从约 754.5 秒降到约 22.8 秒，约 33 倍加速、97% 降幅。

但该 run 只证明“链路连通、无非有限值、权重可更新”，不能证明数值完全正确：

- 梯度最大值仍在 `1e16～1e17`。
- loss 约 1～2 时，这个量级明显可疑。
- AdamW 会归一化梯度，权重变化合理不能反推原始梯度合理。

### 5.8 探针本身造成的性能误判

全量梯度探针每个 step 扫描：

- 144 个 C++ grad buffer。
- 144 个独立 `Parameter.grad`。
- 合计约 580 亿个 BF16 元素。
- `isfinite`、`count_nonzero`、`abs().max()` 会重复遍历。

在 `20260713_140101` 中：

- backward 已降到约 22.8 秒/step。
- `step_other` 仍约 256.7 秒/step，占 61.8%。
- 该 `other` 主要是 pre-optimizer 全量梯度扫描，不是 GDB。

因此诊断 run 与性能 run 必须分开：

- 数值诊断：1～3 step，抽样或分阶段记录 C++ 中间量。
- 性能测试：不使用 GDB、不扫描 expert 梯度/权重，只保留内存、显存和轻量 step timing。
- 正式 TPS：固定 15 step，跳过前 5 个 warmup，用后 10 个 step 总 token/总时间计算。

## 6. 当前状态与仍未解决的问题

### 6.1 已确认解决

- `full_weight_grad` 能从 Python 到达顶层和所有 TP/NUMA 子核。
- 三个 expert 基座 Parameter 能进入 optimizer。
- TP gradient slice、global stride 和逐步清零已修复。
- 错误解释 `m_local_pos_cache` 导致的 SIGSEGV 已修复。
- down dY 被工作 buffer 覆盖导致的 down 梯度全零已修复。
- gate/up/down 三组权重均有实际更新证据。
- AMX BF16/NUMA subpool 已替代极慢的标量单核基座梯度循环。

### 6.2 尚未完成

- `1e16～1e17` 梯度量级仍需分阶段定位；`PASS` 只表示结构检查通过。
- 需要 PyTorch 小尺寸参考梯度逐元素验证 gate/up/down，不能只检查非零。
- 需要 TP=1、TP=2，且包含 `expert_id>0` 的 stride 覆盖测试。
- 需要连续两步激活不同 expert，验证 inactive expert 无残留梯度。
- 需要 `full/hybrid/lora` 三模式回归，确认纯 LoRA 行为不变。
- 个人分支仍需合入共同基线之后的官方 main 提交并解决潜在冲突。

### 6.3 容易误读的日志

- Trainer 的 `grad_norm=0` 可能只统计 model tree 中的 named parameters；KT 注入 optimizer
  的 expert Parameter 可能不在该统计路径中。
- `Number of trainable params = 0` 也可能来自 HF expert zero-storage placeholder；应检查
  optimizer 中是否存在 144 个 KT expert Parameter。
- requant 次数只能证明调用发生，不能证明权重变化。
- loss 下降可能来自 attention/router 等非 expert 参数。
- 自动 summary 可能漏报 child SIGSEGV；崩溃以原始 `train.log`、GDB 和 child exitcode 为准。
- 一步测试若 LR=0，`changed=0` 不是更新链路失败证据。

## 7. 开发与验证要求

### 7.1 修改前

1. 运行 `git status --short --branch`，不要覆盖现有改动。
2. 记录目标模式、TP 数、NUMA 数、`E/H/F/I`、dtype 和 `lora_rank`。
3. 使用符号名定位，不依赖本文记录的历史行号。
4. 判断修改属于 Python Full-FT 接入、顶层 TP 调度还是 AMX 子核，不要跨层打补丁。

### 7.2 构建

普通性能构建使用 Release；需要 GDB 源码行时使用同一 Python 环境构建
`RelWithDebInfo`：

```bash
cd /mnt/data2/wbw/ktransformers
CPUINFER_BUILD_TYPE=RelWithDebInfo \
  /mnt/data2/wbw/conda/envs/Kllama/bin/python3.12 \
  kt-kernel/setup.py build_ext --inplace
CPUINFER_BUILD_TYPE=RelWithDebInfo \
  /mnt/data2/wbw/conda/envs/Kllama/bin/python3.12 -m pip install \
  --no-build-isolation --no-deps --force-reinstall ./kt-kernel
```

安装后必须确认测试实际 import 的 `.so` 与仓库 build 产物一致，并检查 debug build
包含 `.debug_info`/`.debug_line`。

### 7.3 验证顺序

1. 静态检查：setter 不在 LoRA 条件分支内；所有指针先判空再 offset。
2. 格式与构建：`clang-format --dry-run --Werror`，AMX/CUDA build 成功。
3. AMX tile 单测：BF16 `32x32xK` 与标量 BF16 参考比较。
4. 小尺寸参考梯度：gate/up/down 分别与 PyTorch outer-product 比较。
5. TP 测试：TP=1、TP=2，检查两个 `F` slice 无覆盖、无缺口。
6. 跨 step 测试：不同 active expert，检查清零和残留。
7. 模式回归：`full_weight_grad=false` 不写基座梯度；LoRA 结果不变。
8. 短训练：三个 grad buffer 非零且有限，optimizer 后权重发生有限变化，下一次
   forward 使用 requant 后的新权重。
9. 性能测试：关闭 GDB 和重型 probe，至少 15 step，前 5 step warmup，后 10 step TPS。

## 8. 完成标准

结构正确性：

- 所有子核收到正确的 Full-FT 开关和 TP slice。
- gate/up/down C++ grad 与 `Parameter.grad` 一致。
- optimizer 更新权威 BF16 Parameter，requant 使用更新后的指针。
- 无 SIGSEGV、越界、TP 覆盖或跨 step 残留。

数值正确性：

- 三组梯度与 PyTorch 参考实现误差在约定容差内。
- 梯度尺度有合理解释，不仅仅是“finite/nonzero”。
- full/hybrid/lora 均无回归。

性能正确性：

- TPS 结果不包含 GDB 和全量梯度扫描开销。
- 报告明确 batch size、sequence length、GAS、warmup 和稳定 step 数。
- CPU 内存、GPU 显存、backward、optimizer、requant 分项可追溯。

只有同时满足结构、数值和模式回归，才可宣称 Full FT 完全正确。目前已确认结构链路和
权重更新生效，但梯度尺度问题仍未关闭。

## 9. 调试资料索引

```text
FFTtest/Qwen3-30B-A3B/test_log/20260710_103230_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  # 15 步但 changed=0/12；证明初始 Full-FT 链路未生效

FFTtest/Qwen3-30B-A3B/test_log/20260710_192456_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  phase4/train.log          # child exitcode=-11 / Signal 11
  phase4/log_analysis.txt
  summary.md                # 历史汇总漏报崩溃，不得单独采用

FFTtest/Qwen3-30B-A3B/test_log/20260713_105328_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  phase4/gdb_sigsegv.log    # 路由缓存布局误用的直接证据

FFTtest/Qwen3-30B-A3B/test_log/20260713_111729_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  # 1 step、LR=0；只证明无 SIGSEGV，不能判断权重更新

FFTtest/Qwen3-30B-A3B/test_log/20260713_120636_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  expert_gradient_check.txt # gate/up 非零，down 0/48
  expert_weight_change_check.txt
  phase4/step_timing/

FFTtest/Qwen3-30B-A3B/test_log/20260713_140101_1gpu_AMX_BF16_FULLFT_EXPERTONLY/
  expert_gradient_check.txt # gate/up/down 144/144
  expert_weight_change_check.txt # changed=9/12，down=4/4
  phase4/step_timing/        # backward 约 22.8s，probe 进入 other

FFTtest/Qwen3-30B-A3B/gdb_sigsegv.gdb
FFTtest/Qwen3-30B-A3B/run_full_ft_test_1gpu_bf16_frozen.sh --gdb
FFTtest/Qwen3-30B-A3B/expert_buf_probe.py
```

调试时优先读取原始 `phase4/train.log`、GDB backtrace、逐步 timing 和 probe JSON；自动
生成的 `summary.md` 只能作为索引，不能覆盖原始证据。
