# MESH 论文三层编写框架

本文档用于指导 `MESH` 论文的正式写作。所谓“三层元学习结构”，指的是：

- **第一层：目录层**
  - 决定整篇论文有哪些章节、章节顺序是什么、每章承担什么作用。
- **第二层：组织层**
  - 决定每一章内部要回答哪些问题、用哪些实验/图表/定义来支撑、哪些内容不能混写。
- **第三层：落稿层**
  - 直接落到具体段落、句式、论述顺序与可填空模板，方便后续生成正式论文文本。

这份文档不是最终论文，而是论文的**总控蓝图**。后续所有写作都应以这份结构为准，避免在写作时不断漂移题目、贡献点和实验主线。

---

## 一、论文总定位

### 1.1 当前题目建议

主标题建议优先使用：

- **MESH: Memory-tiered Expert Switching for Heterogeneous MoE Inference**

如果后续实验更强调 `mmap -> NUMA` 的热专家晋升，也可以考虑：

- **MESH: Memory-tiered Hot Expert Switching for Efficient MoE Inference**

### 1.2 论文核心主张

整篇论文必须围绕以下一句话展开：

> 在资源受限单机环境下，MoE 推理的关键问题不只是“专家卸载”，而是**如何在文件基线、页缓存复用与 NUMA 本地热缓存之间动态管理专家驻留状态**。

### 1.3 论文的真正增量

这篇论文不能写成：

- “我们基于 KTransformers 做了一些优化”
- “我们提出了一个新的 mmap + NUMA 算法”
- “我们做了一个新的 expert offloading 系统”

这篇论文要写成：

- **MESH 把专家驻留状态建模为在线的分层内存管理问题**
- **MESH 在 KTransformers 现有执行路径之上增加了显式的 expert residency manager**
- **MESH 关注的是 hot expert switching，而不是单纯的静态 offloading**

### 1.4 论文不能犯的错误

- 不能把 `OS page cache` 写成一个和 `NUMA-local DRAM` 并列、完全可控的物理层级
- 不能把 `Based on KTransformers` 写成标题主干
- 不能把 novelty 写成“我们也做了异构执行”，因为 KTransformers 本身已经有异构执行
- 不能把摘要和引言写成纯工程总结，必须始终围绕“expert residency management”这个问题展开

---

## 二、第一层：目录层

这一层只回答一个问题：**整篇论文应该由哪些章节组成，每章承担什么功能。**

---

### 2.1 标准目录结构

1. **Abstract**
   - 用 180 到 220 词概括问题、方法、系统、结果。

2. **Introduction**
   - 给出问题背景、痛点、现有方法不足、核心想法、贡献点。

3. **Background and Motivation**
   - 解释 MoE 推理、专家切换、端侧/资源受限部署的系统背景。
   - 给出为什么 `expert residency` 是核心问题。

4. **Problem Formulation and Design Goals**
   - 明确系统目标、资源约束、设计边界。
   - 定义冷专家、热点专家、驻留状态、预算约束。

5. **MESH Design**
   - 给出 MESH 的总体架构和关键机制。
   - 这一章是全篇最重要的技术章节。

6. **Implementation**
   - 写清楚如何在 KTransformers 上实现，哪些部分复用，哪些部分新增。

7. **Evaluation**
   - 给出实验设置、主结果、消融、机制分析、成本分析。

8. **Related Work**
   - 对比 KTransformers、Fiddler、MoE-Infinity、HybriMoE、FlexGen、vLLM 等。

9. **Discussion**
   - 讨论边界、局限性、AI-SSD 兼容前景、尚未解决的问题。

10. **Conclusion**
   - 收束全文，用一段话重新定义贡献。

---

### 2.2 每章一句话职责

| 章节 | 职责 |
|---|---|
| Abstract | 让审稿人 30 秒内知道你解决了什么问题、做了什么、效果如何 |
| Introduction | 让审稿人接受“这是个值得研究的问题，且你的切入点是新的” |
| Background and Motivation | 让审稿人相信“这不是拍脑袋优化，而是由系统瓶颈逼出来的问题” |
| Problem Formulation and Design Goals | 收窄问题边界，防止 reviewer 说你范围失控 |
| MESH Design | 真正讲清楚机制和系统核心创新 |
| Implementation | 证明你的系统可落地，不是概念图 |
| Evaluation | 证明你的系统有效，并拆出每个机制的价值 |
| Related Work | 证明你不是重复已有工作 |
| Discussion | 把现在还没完全做实的边界讲清楚，降低 reviewer 反感 |
| Conclusion | 给整篇论文一个强、短、可记忆的收束 |

---

## 三、第二层：组织层

这一层回答：**每一章内部应该如何组织内容。**

---

### 3.1 Abstract 的组织

必须按下面 6 句逻辑组织：

1. 研究背景与痛点  
2. 现有方法不足  
3. 你提出的系统/方法  
4. 核心机制  
5. 实现平台  
6. 关键结果

#### 必须回答的问题

- 你的问题是什么？
- 现有方法为什么不够？
- MESH 到底做了什么？
- MESH 是显式管理哪两个状态，机会性利用哪个状态？
- 你是独立系统还是基于 KTransformers 增强？
- 结果最强的一句是什么？

#### 不要在摘要里做的事

- 不要解释太多背景术语
- 不要写数据表
- 不要长篇谈 AI-SSD，只能一句点到为止
- 不要把 `page cache` 写成一个完全可控的 tier

---

### 3.2 Introduction 的组织

建议 5 段结构：

1. **问题背景段**
   - 大 MoE 难部署
   - 端侧/资源受限场景越来越重要
   - 问题的核心是 memory capacity + data movement

2. **现有方法不足段**
   - static offloading 不够
   - 纯 execution placement 不够
   - 现有系统没有显式管理 expert residency

3. **核心 insight 段**
   - MoE 专家并不是均匀活跃
   - expert residency 应该是在线动态管理问题
   - 文件基线 + 热点本地缓存 是更合适的系统抽象

4. **MESH 方案段**
   - 一句话说清系统架构
   - 点 3 个最核心机制

5. **贡献点段**
   - 3 条 contributions

#### 必须回答的问题

- 为什么这是重要问题？
- 为什么当前方法不够？
- 你到底把哪个问题抽象出来了？
- 你和 KTransformers 的边界是什么？

#### 推荐贡献点格式

- We identify expert residency as ...
- We propose MESH, a runtime system that ...
- We implement MESH on top of KTransformers and show ...

---

### 3.3 Background and Motivation 的组织

建议 3 小节：

1. **MoE inference and expert switching**
2. **Single-node heterogeneous deployment constraints**
3. **Why expert residency needs explicit runtime management**

#### 这一章要解决的核心问题

- 为什么这不是普通的 LLM serving 问题？
- 为什么不是“把权重 mmap 一下”就完事？
- 为什么需要显式的 hot expert switching？

#### 这里建议放的图

- 图 1：端侧或单机 MoE 推理中的权重驻留与数据流示意图
- 图 2：专家活跃度/热点偏斜示意图

#### 这里建议放的数据

- expert access skew
- repeated activation locality
- 单机环境中 CPU/GPU/host memory 的瓶颈现象

---

### 3.4 Problem Formulation and Design Goals 的组织

建议分成两部分：

#### A. Problem Formulation

定义：

- baseline expert storage
- hot expert cache
- access sequence
- hotness signal
- promotion / demotion
- budget constraint

#### B. Design Goals

建议写成 4 条：

1. **Capacity**
2. **Locality**
3. **Low switching overhead**
4. **Compatibility with existing heterogeneous execution**

#### 这里不能犯的错误

- 不要上来写一堆算法符号却没有系统意义
- 不要把 OS page cache 写成绝对可控
- 不要把问题定义成“最优缓存替换”，那会把 reviewer 引到算法论文预期上

---

### 3.5 MESH Design 的组织

建议拆成 5 小节：

1. **System Overview**
2. **Expert Residency States**
3. **Hotness Tracking and Switching Policy**
4. **NUMA-local Hot Expert Placement**
5. **Integration with KTransformers Execution Path**

#### 这里是全篇最关键的图

建议至少要有：

- 图 3：MESH overall architecture
- 图 4：expert residency state transitions
- 图 5：promotion / demotion workflow

#### 每个小节该讲什么

**System Overview**
- 全局数据流
- baseline、hot cache、runtime manager、execution path 的关系

**Expert Residency States**
- 显式管理的状态：`mmap baseline`、`NUMA-local hot cache`
- 机会性状态：`OS page-cache reuse`
- 不要把 page cache 说成完全可控层

**Hotness Tracking and Switching Policy**
- tracking 粒度
- 更新周期
- promotion/demotion 依据
- 预算约束

**NUMA-local Hot Expert Placement**
- 为什么 NUMA-aware
- 为什么 hot expert 要本地 pin
- 如何处理 locality

**Integration with KTransformers Execution Path**
- 哪些机制复用了 KTransformers
- 哪些逻辑是 MESH 新增的

#### 这一章必须让 reviewer 记住的 3 句话

- MESH 管的是 **expert residency**，不是重新实现整个执行引擎。
- MESH 显式管理的是 **file-backed baseline + NUMA-local hot cache**。
- MESH 的核心是 **online hot-expert switching under a bounded budget**。

---

### 3.6 Implementation 的组织

建议写 4 小节：

1. **Integration with KTransformers**
2. **Memory Mapping and Runtime Metadata**
3. **Promotion/Demotion Execution**
4. **Engineering Considerations**

#### 这里要回答的问题

- 哪些模块复用现有 KTransformers？
- 哪些模块是你新增？
- 运行时怎么记录 expert 热度和驻留状态？
- 切换开销怎么控制？

#### 这里建议明确写出

- Python 层与 C++ 层边界
- expert residency manager 在哪一层
- 对现有 scheduler / expert wrapper / memory path 的改动点

---

### 3.7 Evaluation 的组织

建议拆成 6 个子问题：

1. **Overall effectiveness**
2. **Memory footprint**
3. **Data movement reduction**
4. **Ablation**
5. **Sensitivity**
6. **Case study / deployment feasibility**

#### 主实验必须回答的问题

- 能跑多大模型？
- 在什么资源下能跑？
- 比基线快多少 / 稳多少 / 省多少内存？
- 切换机制每一层到底贡献了什么？

#### 推荐主 baseline

- 原始 KTransformers
- mmap-only / no hot-cache
- static hotset
- MESH full system

#### 一定要有的消融

- no switching
- no NUMA-local cache
- no page-cache-aware warm reuse
- static budget vs dynamic hotness update

---

### 3.8 Related Work 的组织

建议按 4 类组织：

1. **MoE offloading and expert caching**
2. **Heterogeneous CPU-GPU inference**
3. **Tiered memory and runtime systems**
4. **LLM serving systems**

#### 强相关论文

- KTransformers
- Fiddler
- MoE-Infinity
- HybriMoE
- FlexGen
- vLLM / PagedAttention

#### 这里一定要写清

- KTransformers 是执行底座
- MESH 是在其上增加 residency runtime
- 和 Fiddler / MoE-Infinity / HybriMoE 的差异分别是什么

---

### 3.9 Discussion 的组织

建议写 3 部分：

1. **Current limitations**
2. **Compatibility with emerging storage devices such as AI-SSD**
3. **Future extensions**

#### 这章的作用

- 主动处理审稿人担心的问题
- 把尚未完全实现的部分变成“已知边界”，而不是“遗漏”

#### AI-SSD 应该怎么写

现在更适合写成：

- MESH 的设计天然兼容高带宽、低延迟的 file-backed expert serving
- AI-SSD 是潜在受益平台
- 但如果没有真实部署和测量，不要把它写成主结果

---

## 四、第三层：落稿层

这一层回答：**真正写文章时，每章应该怎么写成段落。**

---

### 4.1 Abstract 模板

可直接按下面 6 句填：

1. Large MoE models are difficult to deploy on ...
2. Existing systems address ..., but ...
3. We present MESH, a runtime system for ...
4. MESH explicitly manages ... and ...
5. Implemented on top of KTransformers, MESH ...
6. MESH enables ... / improves ... / reduces ...

---

### 4.2 Introduction 第一段模板

> Large mixture-of-experts (MoE) models offer favorable scaling properties, but their deployment on resource-constrained machines remains difficult. In these settings, expert weights often exceed fast local memory capacity, and frequent expert switching introduces substantial data-movement overhead. As a result, practical MoE inference efficiency is often limited by memory capacity and movement costs rather than peak arithmetic throughput alone.

---

### 4.3 Introduction 第二段模板

> Existing systems have explored expert offloading, caching, and heterogeneous CPU-GPU execution. However, they typically focus on execution placement or coarse-grained offloading decisions, without treating expert residency itself as an explicit runtime optimization problem. This leaves a key gap in single-node NUMA systems, where runtime locality, file-backed capacity, and local-memory scarcity must be jointly managed.

---

### 4.4 Introduction 第三段模板

> We observe that expert accesses in MoE inference exhibit strong skew and temporal locality, making it unnecessary to keep all experts resident in low-latency local memory. This motivates a runtime that maintains a large file-backed baseline while dynamically promoting a small hot-expert set into NUMA-local memory under a bounded budget.

---

### 4.5 Introduction 方案段模板

> Based on this observation, we present MESH, a memory-tiered hot-expert switching runtime for single-node CPU-GPU MoE inference. MESH combines a file-backed mmap baseline, opportunistic page-cache reuse, and a bounded NUMA-local hot-expert cache. It periodically tracks recent expert accesses and performs online promotion and demotion to adapt expert placement to workload locality.

---

### 4.6 贡献点模板

> This work makes the following contributions:
>
> - We identify expert residency management as a first-class runtime problem in resource-constrained MoE inference.
> - We propose MESH, a memory-tiered hot-expert switching runtime that manages expert placement between a file-backed baseline and a bounded NUMA-local hot set.
> - We implement MESH on top of KTransformers and demonstrate that it enables practical large-scale MoE inference under limited hardware resources.

---

### 4.7 Design 章节模板句

**System Overview**

> MESH is organized around two explicitly managed expert residency states: a file-backed mmap baseline and a NUMA-local hot-expert cache. The runtime tracks recent accesses, updates the hot set online, and coordinates expert placement with the underlying heterogeneous execution path.

**Switching Policy**

> MESH periodically updates expert hotness using recent access statistics collected at runtime. Under a fixed local-memory budget, experts with sustained high reuse are promoted into the NUMA-local cache, while cold experts are demoted back to the file-backed baseline.

**KTransformers Integration**

> MESH is implemented as a residency-management layer on top of KTransformers. It reuses KTransformers’ heterogeneous MoE execution path and augments it with explicit runtime management of expert placement and switching.

---

### 4.8 Evaluation 章节模板句

**总结果段**

> We evaluate MESH on large-scale MoE inference under constrained hardware budgets. The results show that MESH improves inference efficiency by reducing repeated expert transfers and better utilizing limited local memory capacity.

**消融段**

> We further evaluate the contribution of each component in MESH. Removing online switching, disabling NUMA-local hot caching, or falling back to a static placement policy each leads to a measurable degradation, showing that MESH’s gains come from the combined effect of runtime tracking, switching, and locality-aware placement.

---

## 五、实际写作流程建议

推荐按下面顺序写：

1. 先写 **第一层目录**
2. 再把每章的 **第二层组织问题** 填满
3. 先写 **Introduction**
4. 再写 **Design**
5. 再写 **Evaluation**
6. 最后回写 **Abstract**

原因：

- Abstract 依赖最终结果，不应最先写死
- Introduction 和 Design 一旦清楚，整篇论文就不会跑偏
- Evaluation 要反过来服务你的主张，而不是想到什么测什么

---

## 六、当前阶段建议的下一份文档

基于本文档，下一步应立即产出两份子文档：

1. **MESH_Section_Outline_Draft.md**
   - 逐章填第一层和第二层内容

2. **MESH_Abstract_Intro_Draft.md**
   - 先写摘要、引言第一段、贡献点

这两份文档完成后，整篇论文就会进入真正可写状态。

---

## 七、一句话提醒

这篇论文最容易写偏的地方，不是技术细节，而是**把它写成“若干优化的集合”**。  
必须始终围绕这条主线：

> **MESH studies expert residency management as an online tiered-memory problem for single-node heterogeneous MoE inference.**

