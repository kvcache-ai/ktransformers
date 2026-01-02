# SFT MoE AMX Bug 调试记录

本文档记录 SFT MoE AMX 实现过程中遇到的 bug 及其修复方案。

---

# 第一板块：语法 Bug

本板块记录编译期和运行时的语法、类型、继承相关问题。

---

## Bug #1: C++ 继承链中的私有成员访问问题

### 问题现象

编译时出现大量错误，提示基类成员在派生类中不可访问：

```
sft_moe.hpp:57:15: error: 'GeneralMOEConfig AMX_MOE_BASE<...>::config_' is private within this context
   57 |   using Base::config_;
      |               ^~~~~~~
moe.hpp:23:15: note: declared private here
   23 |   using Base::config_;
```

涉及的成员变量包括：`config_`, `tp_part_idx`, `down_ba_`, `down_bb_`, `down_bc_`, `gate_bb_`, `gate_bc_`, `gate_up_ba_`, `up_bb_`, `up_bc_`, `m_local_num_` 等。

### 问题原因

在 C++ 中，`using Base::member` 声明的访问级别取决于它在派生类中所处的 section (public/protected/private)。

**继承链结构：**
```
AMX_MOE_BASE<T, Derived>  (所有成员为 public)
      ↓ 继承
AMX_MOE_TP<T>             (private section 中使用 using Base::*)
      ↓ 继承
AMX_SFT_MOE_TP<T>         (尝试访问父类的这些成员)
```

问题出在 `moe.hpp` 中的 `AMX_MOE_TP` 类：

```cpp
template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 private:  // <-- 问题所在：这些 using 声明在 private section
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
  using Base::config_;
  using Base::tp_part_idx;
  // ... 其他成员
```

虽然这些成员在 `AMX_MOE_BASE` 中是 public 的，但在 `AMX_MOE_TP` 中通过 `using` 声明后变成了 private。当 `AMX_SFT_MOE_TP` 继承 `AMX_MOE_TP` 时，无法访问这些私有成员。

### 解决方案

将 `moe.hpp` 中的 using 声明从 `private` section 移到 `protected` section：

```cpp
template <class T>
class AMX_MOE_TP : public AMX_MOE_BASE<T, AMX_MOE_TP<T>> {
 protected:  // 改为 protected
  using Base = AMX_MOE_BASE<T, AMX_MOE_TP<T>>;
  using Base::config_;
  using Base::tp_part_idx;
  using Base::gate_bb_;
  using Base::up_bb_;
  using Base::down_bb_;
  using Base::gate_up_ba_;
  using Base::gate_bc_;
  using Base::up_bc_;
  using Base::down_ba_;
  using Base::down_bc_;
  using Base::m_local_num_;

 private:  // 实际的私有成员放在这里
  std::filesystem::path prefix;
  void* gate_proj_;
  void* up_proj_;
  void* down_proj_;
```

同时，在 `sft_moe.hpp` 中也做相同的修改，将 using 声明移到 protected section。

---

## Bug #2: MOE_TP_PART Concept 不满足

### 问题现象

编译时出现 concept 约束失败错误：

```
moe-tp.hpp:20:5: note: the required expression 'new T' is invalid
   20 |   { new T(config, tp_idx) } -> std::same_as<T*>;
      |     ^~~~~~~~~~~~~~~~~~~~~
```

```
moe-sft-tp.hpp:27:7: error: template constraint failure for 'template<class T>  requires  MOE_TP_PART<T> class TP_MOE'
   27 | class TP_MOE_SFT : public TP_MOE<T> {
      |       ^~~~~~~~~~
```

### 问题原因

`moe-tp.hpp` 中定义的 `MOE_TP_PART` concept 要求类型 T 必须有一个接受 `GeneralMOEConfig` 参数的构造函数：

```cpp
template <typename T>
concept MOE_TP_PART = requires(T t, ..., GeneralMOEConfig config, int tp_idx) {
  typename T::output_t;
  { new T(config, tp_idx) } -> std::same_as<T*>;  // 要求 GeneralMOEConfig 构造函数
  { t.forward(...) } -> std::same_as<void>;
};
```

但是 `AMX_SFT_MOE_TP<T>` 只有接受 `MOESFTConfig` 的构造函数：

```cpp
AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
```

由于没有 `GeneralMOEConfig` 构造函数，concept 检查失败，导致 `TP_MOE<AMX_SFT_MOE_TP<T>>` 无法实例化，进而导致 `TP_MOE_SFT<AMX_SFT_MOE_TP<T>>` 编译失败。

### 解决方案

1. **在 `common.hpp` 中为 `MOESFTConfig` 添加转换构造函数：**

```cpp
struct MOESFTConfig : public GeneralMOEConfig {
  // ... 现有字段 ...

  MOESFTConfig() : GeneralMOEConfig() {}

  MOESFTConfig(int expert_num, int routed_expert_num, int hidden_size, int intermediate_size)
      : GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size) {}

  // 新增：从 GeneralMOEConfig 转换的构造函数
  explicit MOESFTConfig(const GeneralMOEConfig& base) : GeneralMOEConfig(base) {
    // LoRA 字段使用默认值（已在结构体定义中初始化）
  }
};
```

2. **在 `sft_moe.hpp` 中为 `AMX_SFT_MOE_TP` 添加接受 `GeneralMOEConfig` 的构造函数：**

```cpp
public:
  // 主构造函数（现有）
  AMX_SFT_MOE_TP(MOESFTConfig config, int tp_part_idx = 0)
      : Base(static_cast<GeneralMOEConfig>(config), tp_part_idx), sft_config_(config) {
    // ... 初始化代码 ...
  }

  // 新增：满足 MOE_TP_PART concept 的构造函数
  AMX_SFT_MOE_TP(GeneralMOEConfig config, int tp_part_idx)
      : AMX_SFT_MOE_TP(MOESFTConfig(config), tp_part_idx) {}
```

这个新构造函数使用委托构造，将 `GeneralMOEConfig` 转换为 `MOESFTConfig`（使用默认的 LoRA 配置），然后调用主构造函数。

---

## Bug #3: 缺失的成员方法

### 问题现象

编译时提示 `TP_MOE_SFT` 类没有 `warm_up` 和 `load_weights` 方法：

```
ext_bindings.cpp:365:23: error: 'warm_up' is not a member of 'MoeClass' {aka 'TP_MOE_SFT<AMX_SFT_MOE_TP<amx::GemmKernel224BF> >'}
ext_bindings.cpp:366:28: error: 'load_weights' is not a member of 'MoeClass'
```

### 问题原因

这是 Bug #2 的连锁反应。由于 `MOE_TP_PART` concept 检查失败：
1. `TP_MOE<AMX_SFT_MOE_TP<T>>` 无法正确实例化
2. `TP_MOE_SFT<T>` 继承自 `TP_MOE<T>` 失败
3. 原本从 `TP_MOE_Common` 继承的 `warm_up` 和 `load_weights` 方法不可用

### 解决方案

修复 Bug #1 和 Bug #2 后，继承链恢复正常，这些方法将自动可用。

---

## Bug #4: TP_MOE_SFT 是抽象类

### 问题现象

修复 Bug #1-3 后，编译时出现新的错误：

```
error: invalid new-expression of abstract class type 'TP_MOE_SFT<AMX_SFT_MOE_TP<amx::GemmKernel224BF> >'

note: because the following virtual functions are pure within 'TP_MOE_SFT<...>':
  'void TP_MOE_Common<T>::load_weights()'
  'void TP_MOE_Common<T>::merge_results(int qlen, void* output)'
```

### 问题原因

`TP_MOE_Common<T>` 定义了两个纯虚函数 (moe-tp.hpp:215-217):

```cpp
virtual void load_weights() = 0;
virtual void merge_results(int qlen, void* output) = 0;
```

存在一个模板特化 `TP_MOE<AMX_MOE_BASE<T, Derived>>` (moe_base.hpp:700-761) 实现了这两个函数。

**继承链分析：**
```
TP_MOE_Common<T>                     (定义纯虚函数 load_weights, merge_results)
      ↓ 继承
TP_MOE<T>                            (通用模板，没有实现纯虚函数)
      ↓ 继承
TP_MOE_SFT<T>                        (也没有实现，仍然是抽象类)
```

模板特化 `TP_MOE<AMX_MOE_BASE<T, Derived>>` 实现了这些函数，但是：
- `TP_MOE_SFT<T>` 继承自 `TP_MOE<T>`，其中 `T = AMX_SFT_MOE_TP<Kernel>`
- `AMX_SFT_MOE_TP<Kernel>` 不是 `AMX_MOE_BASE<T, Derived>` 类型，而是其派生类
- **C++ 模板特化不会匹配派生类**，所以 `TP_MOE<AMX_SFT_MOE_TP<Kernel>>` 使用的是通用模板而非特化版本
- 通用模板没有实现纯虚函数，因此 `TP_MOE_SFT` 仍然是抽象类

### 解决方案

在 `moe-sft-tp.hpp` 的 `TP_MOE_SFT` 类中直接实现这两个纯虚函数：

```cpp
// 实现纯虚函数 load_weights
void load_weights() override {
  auto pool = config.pool;
  pool->dispense_backend()->do_numa_job([this](int numa_id) {
    tps[numa_id]->load_weights();
  });
  weights_loaded = true;
}

// 实现纯虚函数 merge_results
void merge_results(int qlen, void* output) override {
  merge_results(qlen, output, false);
}

void merge_results(int qlen, void* output, bool incremental) override {
  // 复用 moe_base.hpp 中的 AVX-512 优化逻辑
  // 合并各 NUMA 节点的输出结果
  auto merge_fn = [this, output, incremental](int token_nth) {
    float* merge_to = local_output_numa[0] + token_nth * tp_configs[0].hidden_size;
    // ... AVX-512 SIMD 合并逻辑
  };

  if (qlen < 10) {
    for (int i = 0; i < qlen; i++) merge_fn(i);
  } else {
    pool->do_work_stealing_job(qlen, nullptr, merge_fn, nullptr);
  }
}
```

### 关键知识点

**C++ 模板特化不匹配派生类**：当定义 `TP_MOE<AMX_MOE_BASE<T, Derived>>` 特化时，它只能精确匹配 `AMX_MOE_BASE<T, Derived>` 类型，不会匹配其派生类如 `AMX_MOE_TP<T>` 或 `AMX_SFT_MOE_TP<T>`。

这是 C++ 模板特化的标准行为。如果需要让特化也匹配派生类，可以：
1. 为每个派生类创建单独的特化（不推荐，维护困难）
2. 在派生类的包装器中直接实现虚函数（本文采用的方案）
3. 使用 SFINAE 或 concepts 进行更灵活的匹配

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/amx/moe.hpp` | 将 `using Base::*` 声明从 private 移到 protected section |
| `operators/common.hpp` | 为 `MOESFTConfig` 添加 `explicit MOESFTConfig(const GeneralMOEConfig&)` 构造函数 |
| `operators/amx/sft_moe.hpp` | 1. 将 `using Base::*` 声明从 private 移到 protected section<br>2. 添加 `AMX_SFT_MOE_TP(GeneralMOEConfig, int)` 构造函数 |

---

## 总结

这次编译错误的根本原因是 C++ 中访问控制和模板 concept 的交互问题：

1. **继承中的访问控制**：在派生类中使用 `using Base::member` 时，成员的最终访问级别由 using 声明所在的 section 决定，而非原始基类中的访问级别。

2. **C++20 Concept 约束**：Concept 要求必须严格满足，包括构造函数签名。即使 `MOESFTConfig` 是 `GeneralMOEConfig` 的派生类，也需要显式提供接受 `GeneralMOEConfig` 的构造函数来满足 concept 要求。

修复建议：在设计继承层次时，如果期望派生类能够访问基类成员，应将 using 声明放在 protected section 而非 private section。

---

## Bug #5: 错误的 Include 路径

### 问题现象

编译时出现头文件找不到错误：

```
moe-sft-tp.hpp:15:10: fatal error: amx/llama.cpp/ggml.h: No such file or directory
   15 | #include "amx/llama.cpp/ggml.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~
```

### 问题原因

在 `moe-sft-tp.hpp` 中添加 `merge_results` 实现时，错误地添加了 `#include "amx/llama.cpp/ggml.h"`。

实际上：
1. `llama.cpp/` 目录不在 `amx/` 目录下，正确路径应该是 `"llama.cpp/ggml.h"`
2. 更重要的是，这个 include 完全不需要，因为：
   - `moe-sft-tp.hpp` → `moe-tp.hpp` → `common.hpp` → `ggml.h`
   - `ggml_bf16_t` 类型已经通过这个 include 链可用

### 解决方案

删除多余的错误 include 行：

```cpp
// 修改前
#include <immintrin.h>

#include "moe-tp.hpp"
#include "amx/la/amx.hpp"
#include "amx/llama.cpp/ggml.h"  // 删除这行

// 修改后
#include <immintrin.h>

#include "moe-tp.hpp"
#include "amx/la/amx.hpp"
```

### 关键知识点

在添加 include 时，应该：
1. 检查头文件的实际路径是否正确
2. 检查所需的类型/函数是否已经通过现有 include 链可用，避免重复 include

---

## Bug #6: Python 绑定缺失核心配置字段

### 问题现象

运行 `test_moe_sft_amx.py` 时出现 AttributeError：

```
test_moe_sft_amx.py:628: AttributeError: 'kt_kernel_ext.moe.MOESFTConfig' object has no attribute 'expert_num'
```

测试代码尝试设置配置字段：

```python
config = kt_kernel_ext.moe.MOESFTConfig()
config.expert_num = expert_num  # <-- 报错
config.num_experts_per_tok = num_experts_per_tok
config.hidden_size = hidden_size
config.intermediate_size = intermediate_size
```

### 问题原因

`ext_bindings.cpp` 中的 pybind11 绑定没有暴露 `GeneralMOEConfig` 的核心字段。

**问题代码 (ext_bindings.cpp:691-747)：**

```cpp
py::class_<GeneralMOEConfig>(moe_module, "MOEConfig")
    .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size) {
      return GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size);
    }))
    // 构造函数接受这些参数...
    .def_readwrite("layer_idx", &GeneralMOEConfig::layer_idx)  // 直接跳到其他字段
    .def_readwrite("pool", &GeneralMOEConfig::pool)
    // ... 没有 expert_num, num_experts_per_tok, hidden_size, intermediate_size 的 def_readwrite！
```

虽然构造函数可以接受这些参数进行初始化，但由于没有 `.def_readwrite()` 声明，Python 端无法在构造后读取或修改这些属性。

`MOESFTConfig` 继承自 `GeneralMOEConfig`（通过 `py::class_<MOESFTConfig, GeneralMOEConfig>`），因此也缺失这些属性。

### 解决方案

在 `ext_bindings.cpp` 的 `GeneralMOEConfig` 绑定中添加缺失的字段声明：

```cpp
py::class_<GeneralMOEConfig>(moe_module, "MOEConfig")
    .def(py::init([](int expert_num, int routed_expert_num, int hidden_size, int intermediate_size) {
      return GeneralMOEConfig(expert_num, routed_expert_num, hidden_size, intermediate_size);
    }))
    // ... 其他 init ...
    // 新增：核心配置字段
    .def_readwrite("expert_num", &GeneralMOEConfig::expert_num)
    .def_readwrite("num_experts_per_tok", &GeneralMOEConfig::num_experts_per_tok)
    .def_readwrite("hidden_size", &GeneralMOEConfig::hidden_size)
    .def_readwrite("intermediate_size", &GeneralMOEConfig::intermediate_size)
    .def_readwrite("layer_idx", &GeneralMOEConfig::layer_idx)
    // ... 其余绑定 ...
```

### 关键知识点

**pybind11 继承与属性暴露**：
1. 当派生类通过 `py::class_<Derived, Base>` 声明继承关系时，基类中通过 `.def_readwrite()` 暴露的属性会自动被派生类继承
2. 但构造函数参数不会自动变成可访问的属性——必须显式声明 `.def_readwrite()`
3. 如果基类没有暴露某个字段，所有派生类都无法访问该字段

### 修改文件清单

| 文件 | 修改内容 |
|------|---------||
| `ext_bindings.cpp` | 在 `GeneralMOEConfig` 绑定中添加 `expert_num`, `num_experts_per_tok`, `hidden_size`, `intermediate_size` 的 `.def_readwrite()` 声明 |

---

# 第二板块：数值 Bug

本板块记录计算结果不正确的数值问题。

---

## Bug #7: 测试文件输出 Buffer 数据类型错误

### 问题现象

运行 `test_moe_sft_amx.py` 时，forward 测试出现极大的相对误差：

```
Relative difference: 1.359375
[FAILED] Test failed with error: Forward pass accuracy test failed: diff=1.359375 >= 0.05
```

相比之下，推理测试 `test_moe_amx.py` 的误差约为 0.046，在可接受范围内。

### 问题原因

C++ 实现中 `TP_MOE_SFT::merge_results()` 将最终输出转换为 **bf16** 格式：

```cpp
// moe-sft-tp.hpp:81-84
for (int e = 0; e < config.hidden_size; e += 32) {
  __m512 x0 = *(__m512*)(merge_to + e);
  __m512 x1 = *(__m512*)(merge_to + e + 16);
  avx512_32xfp32_to_32xbf16(&x0, &x1, (__m512i*)((ggml_bf16_t*)output + token_nth * config.hidden_size + e));
}
```

但测试文件 `test_moe_sft_amx.py` 分配的输出 buffer 是 **float32**：

```python
# test_moe_sft_amx.py:698
output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()  # 错误！
```

**数据类型不匹配的后果：**
- bf16 每个元素 2 字节，float32 每个元素 4 字节
- C++ 向 float32 buffer 写入 bf16 数据，只填充了 buffer 的一半
- Python 将这些 bf16 字节解释为 float32 → 得到完全错误的数值

### 解决方案

修改 `test_moe_sft_amx.py`，将所有 SFT forward 输出 buffer 的 dtype 从 `float32` 改为 `bfloat16`：

**修改点：**

| 函数 | 行号 | 修改内容 |
|------|------|---------|
| `test_moe_sft_forward()` | 698 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_forward()` | 712-716 | 删除 `.to(torch.bfloat16)` 转换 |
| `test_moe_sft_backward()` | 854 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_lora_weight_sync()` | 998, 1026, 1068 | `dtype=torch.float32` → `dtype=torch.bfloat16` |
| `test_moe_sft_training_loop()` | 1205 | `dtype=torch.float32` → `dtype=torch.bfloat16` |

**修改示例：**

```python
# 修改前
output = torch.zeros((qlen, hidden_size), dtype=torch.float32).contiguous()
# ...
output_bf16 = output.to(torch.bfloat16)
diff = torch.mean(torch.abs(output_bf16 - torch_output)) / ...

# 修改后
output = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
# ...
diff = torch.mean(torch.abs(output - torch_output)) / ...
```

### 关键知识点

1. **数据类型必须匹配**：C++ 和 Python 之间通过指针传递数据时，双方必须使用相同的数据类型解释内存
2. **SFT forward 输出为 bf16**：与推理模式一致，SFT 的 forward 输出也是 bf16 格式

---

## Bug #8: TP 模式下基础权重未正确分区

### 问题现象

修复 Bug #7 后，输出不再是垃圾值，但仍有较大误差（约 1.71）：

```
[AMX SFT DEBUG] AMX output[:8] = tensor([ 4.2021e-06,  1.1086e-05, ...])
[MOE SFT DEBUG] Final output[:8] = tensor([-1.2457e-05, -5.0366e-06, ...])
Relative difference: 1.710938
```

AMX 输出和 PyTorch 参考输出数值范围相近（都是 1e-5 到 1e-6），但具体值明显不同。

### 问题原因

**TP（Tensor Parallel）模式的工作原理：**
- intermediate_size 被分割到多个 NUMA 节点
- 每个 NUMA 节点处理 intermediate_size / tp_count 的权重
- 各 NUMA 节点的输出结果相加得到最终输出

**推理模式 `TP_MOE<AMX_MOE_TP<K>>::load_weights()`** (moe.hpp:370-430) 正确处理了权重分区：

```cpp
for (auto i = 0; i < tp_count; i++) {
  auto& tpc = tps[i]->config_;
  size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;

  // 分配临时分区 buffer
  tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

  // 复制对应分区的权重（注意 i * gate_up_elcount 偏移）
  memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
         (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
             i * gate_up_elcount,  // <-- 关键：按 NUMA 节点偏移
         sizeof(ggml_bf16_t) * gate_up_elcount);
}
```

**但 SFT 模式 `TP_MOE_SFT::load_weights()`** (moe-sft-tp.hpp:49-53) 没有做分区：

```cpp
void load_weights() override {
  auto pool = config.pool;
  // 直接调用各 NUMA 的 load_weights，没有先分区！
  pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
  weights_loaded = true;
}
```

**导致的问题：**
1. 测试设置 `config.gate_proj` 指向完整权重张量
2. TP_MOE_SFT 构造时，各 NUMA 的 `tp_configs[i].intermediate_size` 被除以 `tp_count`
3. `load_weights()` 调用时，各 NUMA 的 `AMX_MOE_TP::load_weights()` 使用缩小后的 `intermediate_size` 计算偏移
4. 但源指针仍指向完整权重，导致各 NUMA 读取了错误的权重分区
5. NUMA 0 和 NUMA 1 读取相同或重叠的数据，而非正确的分区

### 解决方案

修改 `TP_MOE_SFT::load_weights()`，在加载前正确分区基础权重：

```cpp
void load_weights() override {
  auto pool = config.pool;

  // 如果 gate_proj 直接设置（非预量化），需要分区权重
  if (config.gate_proj != nullptr) {
    // 为每个 NUMA 节点分配临时分区 buffer
    for (int i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      size_t gate_up_elcount = tpc.intermediate_size * tpc.hidden_size;

      tpc.gate_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      tpc.up_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];
      tpc.down_proj = new ggml_bf16_t[tpc.expert_num * gate_up_elcount];

      // 复制分区后的权重
      pool->get_subpool(i)->do_work_stealing_job(
          tpc.expert_num, nullptr,
          [&, i](int expert_id) {
            // gate 和 up: [expert_num, intermediate_size, hidden_size]
            // 每个 NUMA 获取 intermediate_size 的一个切片
            memcpy((ggml_bf16_t*)tpc.gate_proj + expert_id * gate_up_elcount,
                   (ggml_bf16_t*)config.gate_proj + expert_id * config.intermediate_size * config.hidden_size +
                       i * gate_up_elcount,
                   sizeof(ggml_bf16_t) * gate_up_elcount);

            memcpy((ggml_bf16_t*)tpc.up_proj + expert_id * gate_up_elcount,
                   (ggml_bf16_t*)config.up_proj + expert_id * config.intermediate_size * config.hidden_size +
                       i * gate_up_elcount,
                   sizeof(ggml_bf16_t) * gate_up_elcount);

            // down: [expert_num, hidden_size, intermediate_size]
            // 每个 NUMA 获取 intermediate_size 的一个切片（列）
            for (size_t row = 0; row < config.hidden_size; row++) {
              memcpy((ggml_bf16_t*)tpc.down_proj + expert_id * tpc.hidden_size * tpc.intermediate_size +
                         row * tpc.intermediate_size,
                     (ggml_bf16_t*)config.down_proj + expert_id * config.intermediate_size * config.hidden_size +
                         row * config.intermediate_size + i * tpc.intermediate_size,
                     sizeof(ggml_bf16_t) * tpc.intermediate_size);
            }
          },
          nullptr);
    }

    // 在各 NUMA 节点加载权重
    pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });

    // 清理临时 buffer
    for (int i = 0; i < tp_count; i++) {
      auto& tpc = tps[i]->config_;
      delete[] (ggml_bf16_t*)tpc.gate_proj;
      delete[] (ggml_bf16_t*)tpc.up_proj;
      delete[] (ggml_bf16_t*)tpc.down_proj;
    }
  } else {
    // 无需分区（预量化或无权重）
    pool->dispense_backend()->do_numa_job([this](int numa_id) { tps[numa_id]->load_weights(); });
  }

  weights_loaded = true;
}
```

### 关键知识点

1. **TP 模式权重分区**：当使用 Tensor Parallel 时，每个 NUMA 节点只处理 intermediate_size 的一部分。必须在加载前将完整权重按正确偏移分区到各节点。

2. **gate/up vs down 的分区方式不同**：
   - gate_proj, up_proj: 形状为 `[expert_num, intermediate_size, hidden_size]`，按 intermediate_size 维度切片（连续块）
   - down_proj: 形状为 `[expert_num, hidden_size, intermediate_size]`，按 intermediate_size 维度切片（需逐行复制）

3. **SFT 继承推理逻辑**：SFT 模式应尽量复用推理模式的基础设施，包括权重分区逻辑。

### 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `operators/moe-sft-tp.hpp` | 重写 `load_weights()` 方法，添加 TP 权重分区逻辑 |
| `examples/test_moe_sft_amx.py` | 将输出 buffer dtype 从 `float32` 改为 `bfloat16` |
