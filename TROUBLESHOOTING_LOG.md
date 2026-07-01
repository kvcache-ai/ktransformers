# 踩坑日志

## 2026-06-21: 35B TP>1 输出坍缩为 `!`

### 现象
- 35B AMXINT4 模型在 TP>1（TP2/TP4）时，输出全部坍缩为 `!`
- TP1 正常
- layer 3 的 GPU w2_weight 包含 NaN/Inf

### 根因
启动脚本中 `--model` 和 `--kt-weight-path` 都指向 AMXINT4 格式 checkpoint（`Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH`）。

AMXINT4 格式的 checkpoint key 为 `blk.N.ffn_down_exps.E.numa.T.weight` + `.scale`，按 NUMA/TP 预切分存储。sglang 标准 weight_loader（`_weight_loader_physical` → `_weight_loader_impl` → `_load_w2`）只处理标准格式 key（`experts.N.down_proj.`），不识别 AMXINT4 格式。

因此 GPU 的 `w2_weight`（由 `UnquantizedFusedMoEMethod.create_weights` 创建为 `torch.empty`）从未被 checkpoint 填充，保持未初始化状态。TP1 时恰好不含 NaN，TP>1 时 layer 3 的 w2_weight 包含 NaN/Inf，导致输出 NaN 坍缩为 `!`。

### 修复
- `--model` 改为标准 bf16 格式 checkpoint（`Qwen3.5-35B-A3B-Unfused`）
- `--kt-weight-path` 保持 AMXINT4 格式（供 CPU AMX 内核使用）

```
--model /mnt/data2/tmp/qujing_mesh/Qwen3.5-35B-A3B-Unfused/
--kt-weight-path /mnt/data2/models/Qwen3.5-35B-A3B-AMXINT4-NUMA2-MESH/
```

### 验证
- DIAG 输出确认 `_weight_loader_physical` 和 `_load_w2` 被正确调用
- w2_weight 被正确加载（nan=False, inf=False）
- TP 切分正确（shard_dim=1, shard_size=256, tp_rank, narrow 后 shape=[2048, 256]）
- CPU expert 被正确跳过（mask=False → SKIP）
- TP2 服务成功启动，英文/中文/长文本输出全部正常

### 关键代码路径
- GPU 权重创建：`UnquantizedFusedMoEMethod.create_weights()` → `w2_weight = torch.empty(...)`（未初始化）
- GPU 权重加载：sglang 标准 weight_loader 只处理 `experts.N.down_proj.` 格式 key
- CPU 权重加载：kt-kernel 的 `SafeTensorLoader`（`kt-kernel/python/utils/loader.py`）解析 AMXINT4 格式 key
- KTEPWrapperMethod：`create_weights` 调用 `gpu_method.create_weights`，`process_weights_after_loading` 调用 `wrapper.load_weights`（CPU 权重）

### 教训
1. **`--model` 必须指向标准 bf16 checkpoint**，sglang 标准 weight_loader 才能识别并加载 GPU 权重
2. **`--kt-weight-path` 指向 AMXINT4 checkpoint**，供 kt-kernel 的 AMX 内核使用
3. **MESH 模式下同样需要修复 `--model` 路径**，因为 GPU 专家（GE>0）仍需要从标准 checkpoint 加载 w2_weight
4. **`torch.empty` 不会自动清零**，未初始化的权重在 TP>1 时可能包含 NaN/Inf
5. **本地代码必须与远程同步**：远程 .venv 是 pip install 的旧版，本地子模块是最新 commit，需要 rsync 同步

### 排查过程中排除的假设
- ❌ 不是输入 NaN/Inf 传播
- ❌ 不是 w13_weight NaN
- ❌ 不是 topk_weights NaN
- ❌ 不是 CPU 专家问题
- ❌ 不是 `--kt-enable-dynamic-expert-update` 导致
- ❌ 不是 `_prepare_weight_bf16` 导致（`load()` 未被调用）
- ❌ 不是 `process_weights_after_loading` 导致（bf16 no-op）
- ✅ **是 `--model` 指向 AMXINT4 格式 checkpoint，GPU w2_weight 从未被填充**

### 涉及文件
- `third_party/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py` — weight_loader 实现
- `third_party/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py` — KTEPWrapperMethod
- `third_party/sglang/python/sglang/srt/layers/quantization/unquant.py` — UnquantizedFusedMoEMethod
- `third_party/sglang/python/sglang/srt/models/qwen3_5.py` — 模型定义
- `kt-kernel/python/utils/loader.py` — SafeTensorLoader（CPU 权重加载）
- `kt-kernel/python/utils/amx.py` — AMXMoEWrapper
