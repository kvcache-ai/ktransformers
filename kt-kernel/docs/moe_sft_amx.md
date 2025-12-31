# MOE SFT AMX 算子接口文档

## 概述

`moe_sft_amx` 是用于 MoE (Mixture of Experts) 层 LoRA 微调的高性能算子，基于 Intel AMX (Advanced Matrix Extensions) 加速。该算子支持 BF16 和 INT8 量化模式，提供前向传播和反向传播功能。

### 主要特性

- **LoRA 微调**: 支持在 gate/up/down 三个投影矩阵上应用 LoRA 适配器
- **量化模式**: 支持 BF16 和 INT8 两种精度
- **AMX 加速**: 利用 Intel AMX 指令集进行高效矩阵运算
- **异步执行**: 通过 CPUInfer 实现异步任务提交和执行
- **零拷贝设计**: LoRA 权重通过指针直接访问 Python tensor，无需复制
- **梯度检查点**: 支持 forward 保存中间值用于 backward

---

## 数据流

```
Training Step (新 API - 零拷贝设计):
┌─────────────────────────────────────────────────────────────────────┐
│  Python                           │           C++                   │
├───────────────────────────────────┼─────────────────────────────────┤
│  config.gate_lora_a = ptr         │  直接访问 Python tensor 内存    │
│  (零拷贝, 在初始化时设置)          │                                 │
│                                   │                                 │
│  1. forward_sft_task()        ────>  前向传播 (保存中间值)          │
│     output <──────────────────────   返回输出 (float32)             │
│                                   │                                 │
│  2. backward_task()           ────>  反向传播                       │
│     grad_lora_* <─────────────────   写入 LoRA 梯度到指定 buffer    │
│                                   │                                 │
│  3. optimizer.step()              │                                 │
│     原地更新 LoRA 权重             │  下次 forward 自动看到更新      │
│     (零拷贝, 无需同步)             │                                 │
│                                   │                                 │
│  4. 下一个 step, 回到 1           │                                 │
└───────────────────────────────────┴─────────────────────────────────┘
```

---

## 配置参数

### MOESFTConfig

| 参数 | 类型 | 说明 |
|------|------|------|
| `expert_num` | int | 专家总数 |
| `num_experts_per_tok` | int | 每个 token 激活的专家数 (top-k) |
| `hidden_size` | int | 隐藏层维度 |
| `intermediate_size` | int | MLP 中间层维度 |
| `lora_rank` | int | LoRA 秩 (r) |
| `lora_alpha` | float | LoRA 缩放因子 (alpha) |
| `layer_idx` | int | 层索引 |
| `max_len` | int | 最大序列长度 |
| `max_cache_depth` | int | 最大缓存深度 (用于梯度检查点) |
| `gate_proj` | int64 | gate 投影权重指针 |
| `up_proj` | int64 | up 投影权重指针 |
| `down_proj` | int64 | down 投影权重指针 |
| `gate_lora_a` | int64 | gate LoRA A 权重指针 (零拷贝) |
| `gate_lora_b` | int64 | gate LoRA B 权重指针 (零拷贝) |
| `up_lora_a` | int64 | up LoRA A 权重指针 (零拷贝) |
| `up_lora_b` | int64 | up LoRA B 权重指针 (零拷贝) |
| `down_lora_a` | int64 | down LoRA A 权重指针 (零拷贝) |
| `down_lora_b` | int64 | down LoRA B 权重指针 (零拷贝) |
| `pool` | WorkerPool* | CPUInfer 后端线程池 |

---

## 权重格式

### 基础权重 (冻结)

```python
gate_proj: Tensor  # [expert_num, intermediate_size, hidden_size], bf16
up_proj: Tensor    # [expert_num, intermediate_size, hidden_size], bf16
down_proj: Tensor  # [expert_num, hidden_size, intermediate_size], bf16
```

### LoRA 适配器权重 (可训练)

每个投影矩阵有两个 LoRA 矩阵 A 和 B:

```python
# Gate 投影 LoRA
gate_lora_a: Tensor  # [expert_num, lora_rank, hidden_size], bf16
gate_lora_b: Tensor  # [expert_num, intermediate_size, lora_rank], bf16

# Up 投影 LoRA
up_lora_a: Tensor    # [expert_num, lora_rank, hidden_size], bf16
up_lora_b: Tensor    # [expert_num, intermediate_size, lora_rank], bf16

# Down 投影 LoRA
down_lora_a: Tensor  # [expert_num, lora_rank, intermediate_size], bf16
down_lora_b: Tensor  # [expert_num, hidden_size, lora_rank], bf16
```

### LoRA 计算公式

```
output = input @ W^T + (input @ A^T @ B^T) * (alpha / rank)
```

其中:
- `W` 是基础权重 (冻结)
- `A` 和 `B` 是 LoRA 适配器矩阵 (可训练)
- `alpha / rank` 是缩放因子

---

## 接口说明

### 1. 创建实例

```python
import kt_kernel
kt_kernel_ext = kt_kernel.kt_kernel_ext

# 创建 CPUInfer 实例
CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

# 创建配置 (新 API - 使用属性设置)
config = kt_kernel_ext.moe.MOESFTConfig()
config.expert_num = expert_num
config.num_experts_per_tok = num_experts_per_tok
config.hidden_size = hidden_size
config.intermediate_size = intermediate_size
config.lora_rank = lora_rank
config.lora_alpha = lora_alpha
config.max_cache_depth = 1  # 梯度检查点缓存深度
config.max_len = max_len
config.layer_idx = 0

# 设置基础权重指针
config.gate_proj = gate_proj.data_ptr()
config.up_proj = up_proj.data_ptr()
config.down_proj = down_proj.data_ptr()

# 设置 LoRA 权重指针 (零拷贝 - 直接指向 Python tensor)
config.gate_lora_a = gate_lora_a.data_ptr()
config.gate_lora_b = gate_lora_b.data_ptr()
config.up_lora_a = up_lora_a.data_ptr()
config.up_lora_b = up_lora_b.data_ptr()
config.down_lora_a = down_lora_a.data_ptr()
config.down_lora_b = down_lora_b.data_ptr()

config.pool = CPUInfer.backend_

# 创建 MOE SFT 实例
# BF16 模式:
moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)
# 或 INT8 模式:
moe = kt_kernel_ext.moe.AMXInt8_SFT_MOE(config)
```

### 2. 加载基础权重

```python
# 加载并量化基础权重
CPUInfer.submit(moe.load_weights_task())
CPUInfer.sync()
```

### 3. 预热 (可选)

```python
CPUInfer.submit(moe.warm_up_task())
CPUInfer.sync()
```

### 4. 前向传播

```python
# 输入张量
bsz_tensor = torch.tensor([qlen], device="cpu")           # 批大小
expert_ids = torch.tensor(..., dtype=torch.int64)         # [qlen, k]
weights = torch.tensor(..., dtype=torch.float32)          # [qlen, k]
input_data = torch.tensor(..., dtype=torch.bfloat16)      # [qlen, hidden_size]
output = torch.zeros((qlen, hidden_size), dtype=torch.float32)  # 输出为 float32

CPUInfer.submit(moe.forward_sft_task(
    bsz_tensor.data_ptr(),
    num_experts_per_tok,
    expert_ids.data_ptr(),
    weights.data_ptr(),
    input_data.data_ptr(),
    output.data_ptr(),
    save_for_backward=True  # 是否保存中间值用于反向传播
))
CPUInfer.sync()
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `bsz_ptr` | int64 | 批大小张量指针 |
| `num_experts_per_tok` | int | 每 token 专家数 |
| `expert_ids_ptr` | int64 | 专家 ID 张量指针 [qlen, k] |
| `weights_ptr` | int64 | 路由权重张量指针 [qlen, k] |
| `input_ptr` | int64 | 输入张量指针 [qlen, hidden_size] |
| `output_ptr` | int64 | 输出张量指针 [qlen, hidden_size], float32 |
| `save_for_backward` | bool | 是否保存中间值 |

### 5. 反向传播

```python
# 分配梯度缓冲区
grad_output = torch.tensor(..., dtype=torch.bfloat16)     # [qlen, hidden_size]
grad_input = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16)

grad_gate_lora_a = torch.zeros_like(gate_lora_a)
grad_gate_lora_b = torch.zeros_like(gate_lora_b)
grad_up_lora_a = torch.zeros_like(up_lora_a)
grad_up_lora_b = torch.zeros_like(up_lora_b)
grad_down_lora_a = torch.zeros_like(down_lora_a)
grad_down_lora_b = torch.zeros_like(down_lora_b)

CPUInfer.submit(moe.backward_task(
    grad_output.data_ptr(),
    grad_input.data_ptr(),
    grad_gate_lora_a.data_ptr(),
    grad_gate_lora_b.data_ptr(),
    grad_up_lora_a.data_ptr(),
    grad_up_lora_b.data_ptr(),
    grad_down_lora_a.data_ptr(),
    grad_down_lora_b.data_ptr()
))
CPUInfer.sync()
```

**参数说明**:

| 参数 | 类型 | 说明 |
|------|------|------|
| `grad_output_ptr` | int64 | 上游梯度指针 [qlen, hidden_size] |
| `grad_input_ptr` | int64 | 输入梯度输出指针 [qlen, hidden_size] |
| `grad_gate_lora_a/b_ptr` | int64 | gate LoRA 梯度输出指针 |
| `grad_up_lora_a/b_ptr` | int64 | up LoRA 梯度输出指针 |
| `grad_down_lora_a/b_ptr` | int64 | down LoRA 梯度输出指针 |

### 6. 更新 LoRA 权重指针 (可选)

当 LoRA 权重 tensor 被重新分配时 (例如使用非原地操作), 需要更新指针:

```python
# 例如: 当 tensor 被重新分配后
new_gate_lora_a = some_operation_that_creates_new_tensor(gate_lora_a)

# 更新 C++ 端的指针
CPUInfer.submit(moe.update_lora_weights_task(
    new_gate_lora_a.data_ptr(),
    new_gate_lora_b.data_ptr(),
    new_up_lora_a.data_ptr(),
    new_up_lora_b.data_ptr(),
    new_down_lora_a.data_ptr(),
    new_down_lora_b.data_ptr()
))
CPUInfer.sync()
```

**注意**: 如果使用原地操作 (如 `tensor.add_()`, `optimizer.step()`), 则不需要调用此接口, 因为零拷贝设计会自动看到更新。

---

## 完整训练示例

```python
import torch
import kt_kernel
kt_kernel_ext = kt_kernel.kt_kernel_ext

# 配置
expert_num = 256
hidden_size = 7168
intermediate_size = 2048
num_experts_per_tok = 8
lora_rank = 16
lora_alpha = 32.0
qlen = 4
num_threads = 60

# 初始化基础权重 (冻结)
gate_proj = torch.randn(expert_num, intermediate_size, hidden_size, dtype=torch.bfloat16).contiguous() / 100
up_proj = torch.randn(expert_num, intermediate_size, hidden_size, dtype=torch.bfloat16).contiguous() / 100
down_proj = torch.randn(expert_num, hidden_size, intermediate_size, dtype=torch.bfloat16).contiguous() / 100

# 初始化 LoRA 权重 (可训练)
gate_lora_a = torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16).contiguous() / 100
gate_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
up_lora_a = torch.randn(expert_num, lora_rank, hidden_size, dtype=torch.bfloat16).contiguous() / 100
up_lora_b = torch.zeros(expert_num, intermediate_size, lora_rank, dtype=torch.bfloat16).contiguous()
down_lora_a = torch.randn(expert_num, lora_rank, intermediate_size, dtype=torch.bfloat16).contiguous() / 100
down_lora_b = torch.zeros(expert_num, hidden_size, lora_rank, dtype=torch.bfloat16).contiguous()

# 包装为 nn.Parameter 用于 optimizer
gate_lora_a_param = torch.nn.Parameter(gate_lora_a)
gate_lora_b_param = torch.nn.Parameter(gate_lora_b)
up_lora_a_param = torch.nn.Parameter(up_lora_a)
up_lora_b_param = torch.nn.Parameter(up_lora_b)
down_lora_a_param = torch.nn.Parameter(down_lora_a)
down_lora_b_param = torch.nn.Parameter(down_lora_b)

lora_params = [
    gate_lora_a_param, gate_lora_b_param,
    up_lora_a_param, up_lora_b_param,
    down_lora_a_param, down_lora_b_param
]
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

# 初始化 CPUInfer 和 MOE
CPUInfer = kt_kernel_ext.CPUInfer(num_threads)

# 创建配置 (零拷贝设计)
config = kt_kernel_ext.moe.MOESFTConfig()
config.expert_num = expert_num
config.num_experts_per_tok = num_experts_per_tok
config.hidden_size = hidden_size
config.intermediate_size = intermediate_size
config.lora_rank = lora_rank
config.lora_alpha = lora_alpha
config.max_cache_depth = 1
config.max_len = 25600
config.layer_idx = 0
config.gate_proj = gate_proj.data_ptr()
config.up_proj = up_proj.data_ptr()
config.down_proj = down_proj.data_ptr()
# 零拷贝: 直接指向 Python tensor
config.gate_lora_a = gate_lora_a_param.data.data_ptr()
config.gate_lora_b = gate_lora_b_param.data.data_ptr()
config.up_lora_a = up_lora_a_param.data.data_ptr()
config.up_lora_b = up_lora_b_param.data.data_ptr()
config.down_lora_a = down_lora_a_param.data.data_ptr()
config.down_lora_b = down_lora_b_param.data.data_ptr()
config.pool = CPUInfer.backend_

moe = kt_kernel_ext.moe.AMXBF16_SFT_MOE(config)

# 加载基础权重
CPUInfer.submit(moe.load_weights_task())
CPUInfer.sync()

# 预热
CPUInfer.submit(moe.warm_up_task())
CPUInfer.sync()

# 训练循环
for step in range(100):
    # 生成数据
    expert_ids = torch.stack([
        torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)
    ]).to(torch.int64).contiguous()
    weights = torch.rand(qlen, num_experts_per_tok, dtype=torch.float32).contiguous()
    weights = weights / weights.sum(dim=-1, keepdim=True)
    input_data = torch.randn(qlen, hidden_size, dtype=torch.bfloat16).contiguous() / 100
    target = torch.randn(qlen, hidden_size, dtype=torch.bfloat16).contiguous() / 100
    bsz_tensor = torch.tensor([qlen])

    # 1. 前向传播 (无需同步 LoRA 权重 - 零拷贝设计)
    output = torch.zeros(qlen, hidden_size, dtype=torch.float32).contiguous()
    CPUInfer.submit(moe.forward_sft_task(
        bsz_tensor.data_ptr(), num_experts_per_tok,
        expert_ids.data_ptr(), weights.data_ptr(),
        input_data.data_ptr(), output.data_ptr(), True
    ))
    CPUInfer.sync()

    # 2. 计算 loss
    loss = torch.mean((output - target.float()) ** 2)
    grad_output = (2 * (output - target.float()) / output.numel()).to(torch.bfloat16).contiguous()

    # 3. 反向传播
    grad_input = torch.zeros(qlen, hidden_size, dtype=torch.bfloat16).contiguous()
    grad_gate_lora_a = torch.zeros_like(gate_lora_a_param.data)
    grad_gate_lora_b = torch.zeros_like(gate_lora_b_param.data)
    grad_up_lora_a = torch.zeros_like(up_lora_a_param.data)
    grad_up_lora_b = torch.zeros_like(up_lora_b_param.data)
    grad_down_lora_a = torch.zeros_like(down_lora_a_param.data)
    grad_down_lora_b = torch.zeros_like(down_lora_b_param.data)

    CPUInfer.submit(moe.backward_task(
        grad_output.data_ptr(), grad_input.data_ptr(),
        grad_gate_lora_a.data_ptr(), grad_gate_lora_b.data_ptr(),
        grad_up_lora_a.data_ptr(), grad_up_lora_b.data_ptr(),
        grad_down_lora_a.data_ptr(), grad_down_lora_b.data_ptr()
    ))
    CPUInfer.sync()

    # 4. 复制梯度到 param.grad
    gate_lora_a_param.grad = grad_gate_lora_a
    gate_lora_b_param.grad = grad_gate_lora_b
    up_lora_a_param.grad = grad_up_lora_a
    up_lora_b_param.grad = grad_up_lora_b
    down_lora_a_param.grad = grad_down_lora_a
    down_lora_b_param.grad = grad_down_lora_b

    # 5. 优化器更新 (原地更新, 零拷贝自动生效)
    optimizer.step()
    optimizer.zero_grad()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")
```

---

## 精度要求

| 模式 | 前向传播阈值 | 反向传播阈值 |
|------|-------------|-------------|
| BF16 | < 0.05 | < 0.10 |
| INT8 | < 0.15 | < 0.25 |

精度计算方式:
```python
relative_diff = mean(abs(output - reference)) / mean(abs(reference))
```

---

## 注意事项

1. **零拷贝设计**: LoRA 权重通过指针直接访问 Python tensor, 无需每次 forward 前同步
2. **内存对齐**: 所有张量必须是 contiguous 的
3. **异步执行**: 使用 `CPUInfer.submit()` 提交任务后需要调用 `CPUInfer.sync()` 等待完成
4. **梯度缓冲区**: 反向传播会覆盖梯度缓冲区，不会累积
5. **基础权重冻结**: `load_weights_task()` 只需调用一次，基础权重在训练过程中不变
6. **指针更新**: 如果 LoRA tensor 被重新分配 (非原地操作), 需要调用 `update_lora_weights_task()` 更新指针
7. **输出格式**: `forward_sft_task()` 输出为 float32, 便于后续 loss 计算

---

## API 变更记录

### v2.0 (当前版本) - 零拷贝设计

- **新增**: `MOESFTConfig` 支持直接设置 LoRA 权重指针
- **新增**: `forward_sft_task()` - SFT 专用前向传播
- **新增**: `update_lora_weights_task()` - 更新 LoRA 权重指针
- **移除**: `sync_lora_weights_task()` - 不再需要每次同步
- **变更**: `load_weights_task()` - 替代 `load_base_weights_task()`, 无需 mapping 参数
- **变更**: `backward_task()` - 简化参数, 使用缓存的路由信息
- **变更**: 输出格式从 bf16 改为 float32

### v1.0 (旧版本)

- `sync_lora_weights_task()` - 每次 forward 前同步 LoRA 权重
- `forward_task()` - 通用前向传播
- `backward_task()` - 需要传入完整的路由信息

---

## 测试文件

完整测试用例见: `kt-kernel/examples/test_moe_sft_amx.py`

测试内容:
- `test_moe_sft_forward("bf16")` - BF16 前向传播精度
- `test_moe_sft_forward("int8")` - INT8 前向传播精度
- `test_moe_sft_backward("bf16")` - BF16 反向传播精度
- `test_moe_sft_backward("int8")` - INT8 反向传播精度
- `test_moe_sft_lora_weight_sync()` - LoRA 权重同步和指针更新
- `test_moe_sft_training_loop()` - 完整训练循环
