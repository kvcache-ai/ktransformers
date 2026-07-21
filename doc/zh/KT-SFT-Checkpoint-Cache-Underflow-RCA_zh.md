# KT SFT checkpoint cache underflow 简要记录

## 现象

Qwen3-30B LoRA 启用 reentrant activation checkpointing 后，首个 backward 在相邻 MoE 层间触发：

```text
[KT-MOE ERROR] Forward cache stack underflow! cache_stack_top_=0
```

关闭 gradient checkpointing 可以绕过问题，但不是根因修复。

## 根因

Layer N backward 完成后会异步为 Layer N-1 repack backward 权重；reentrant checkpoint 随即重算 Layer N-1 forward。两者并发调用同一个 `NumaJobDistributor::do_numa_job()`，而该调度器共用 `compute_func` 和任务状态，不支持并发提交。

结果是 Python 已将 cache depth 记为 1，但 C++ forward cache 未正确 push，随后 backward pop 时 `cache_stack_top_` 仍为 0。

两个 GC 开关只是将执行顺序切换到 reentrant 路径，从而暴露竞态。Non-reentrant 路径会先进入 `KTMoEFunction.backward()` 等待 repack，再触发重算，因此没有命中该窗口。

## 修复

在 Python wrapper 中记录 backward repack pending 状态。所有可能提交 CPU pool 任务的 sync/async forward、backward 入口，在 pending 时先调用 `wait_backward_repack()`；无 pending 的正常路径不增加 C++ wait。

相关代码：

- `kt-kernel/python/sft/base.py`
- `kt-kernel/test/per_commit/test_sft_authoritative_grad.py`

## 验证

- 原配置、不等待 repack：稳定复现 Layer 46 cache underflow。
- 相同配置、forward 前等待 repack：完整完成一个 forward/backward/optimizer step。
- 修复后诊断 step：`loss=3.219`，`grad_norm=3.221`，Layer 46/47 cache 均正常完成 `0 -> 1 -> 0`。
- Python 相关回归测试：`15 passed`。

远端诊断位置：

```text
/mnt/sft_yyj_yyj/nekoqa-qwen3-30b-instruct-2507-golden-20260721/diagnostics/cache-underflow/
/mnt/sft_yyj_yyj/nekoqa-qwen3-30b-instruct-2507-golden-20260721/logs/diag-cache-underflow.log
/mnt/sft_yyj_yyj/nekoqa-qwen3-30b-instruct-2507-golden-20260721/runtime/diag-cache-underflow-wait/
```

## 后续

当前最小修复覆盖训练竞态。长期可让 `NumaJobDistributor` 自身串行化并发提交，或为 repack 使用独立线程池，从调度器层面消除同类风险。
