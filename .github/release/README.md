# 四仓 main 一键发布（Draft，尚未启用）

入口：**KTransformers → Actions → Release four-main stack → Run workflow**。
选择 `main`，`target=build` 只产出候选 artifact；`target=candidate` 构建和验收；
`target=pypi` 才允许上传。
合并 PR、修改版本号均不会自动上传；此 PR 不改变已经安装的 PyPI 包。

`build` 用于验收 runner 尚未就绪时先通过 CI 构建；它不会启用 GPU runner、
跳过发布验收或上传 PyPI，不能把构建成功当成发布成功。管理员可设置
`KT_RELEASE_WORK_ROOT` 为可执行的空闲内存盘目录，避免填满 runner 磁盘；
构建和临时文件都进入该目录下本次创建、带 run ID 标记的独立子目录。

### 模型验收 runner 未就绪时的人工接管

`Promote manually validated CI wheels` 只接管上传，不重新编译、不执行 Kimi
训练，也不伪造自动验收 workflow 的成功记录。它只能从官方 main 手动触发，
读取成功的 `Release four-main stack` 构建 artifact，并要求：

- 用完整 commit SHA 和文件 SHA256 固定维护者的人工验收 JSON；该文件只作为
  数据读取，不能执行其中的代码。
- 核对同一份四仓 source lock、构建 run/attempt、五个 wheels 与完整依赖哈希。
- 核对两台机器的全新环境、两种 extras 安装报告、安装训练工具前后的核心
  文件哈希，及 Qwen/DeepSeek 一个 optimizer step 的有限 raw loss 和 GLM 问答。
- 本次 post4 的人工记录还包含 sap4 Kimi 的完整 LoRA 保存、转换、至少 32 步
  风格训练及未出现在训练集中的风格问答；Kimi 不被添加为 CI 自动训练任务。

凭据仅在最终上传 step 注入，先预检整批版本，再按 Accelerate → Transformers
→ KT-Kernel → SGLang → KTransformers 的顺序上传同一批文件。上传后状态明确为
`uploaded-awaiting-public-e2e`，正式 PyPI 新环境复验仍由维护者完成。
人工记录依赖维护者真实执行、审核与授权，不等同于自动 runner 的执行证明。
没有完成验收记录时，不能通过该入口发布。

```text
Run workflow
  → 两次观测并锁定四仓 main SHA + 工作流 SHA
  → 检查源码版本、依赖 pin、PyPI 版本是否已占用
  → 新环境编译一次 → manylinux 修复 → 组装最终五个 wheels
  → 锁定完整依赖 wheelhouse、两种 extras 的解析结果和 SHA256
  → 调用 #2195：干净环境安装候选，完成三模型验收
  → [仅 target=pypi] prod 环境：核验证据、按依赖顺序上传，KT 最后
  → 调用 #2195：再次新建环境，从正式 PyPI 默认安装并完成三模型验收
  → 核对版本、文件名和 SHA256，标记 released-and-verified
```

## 验收边界

复用 #2195，不复制另一套模型测试。qj5090 有任务时排队，不杀其他进程。

- Qwen3-30B-A3B：LoRA 一个 optimizer step、raw loss 有限、正常退出。
- DeepSeek-V3.1：同上，每个 rank 都必须有证据。
- GLM-5.3-Flash：服务 ready、实际 decode、问答语义正确。
- 同时检查 `pip install "ktransformers[sglang]"` 和
  `pip install "ktransformers[sglang,sft]"`，分别使用全新 venv。
- 候选阶段从已核对的本地 wheelhouse 安装，不访问索引。发布后用默认的、
  **不指定版本**的用户命令从正式 PyPI 安装，核对完整依赖解析结果、下载域名、
  版本、wheel 文件名和 SHA256，拒绝回退旧版本。
- 候选阶段不通过、缺一个测试、资源等待超时，均不能进入发布；上传成功但
  正式 PyPI 复验失败，不能标记“发布验收成功”。

Kimi 的[候选教程](examples/kimi-k25/README.md)包含公共数据准备和保存/续训配置。
它尚不是干净 wheel 的验收证据，也未被加入自动三模型训练任务；Kimi 的完整
训练、真续训和新进程推理仍需独立验收。

## 源码、版本与产物

- 四仓为 `kvcache-ai/{ktransformers,sglang,transformers,accelerate}`，每次自动
  读取 main，无需手填 SHA。快照后不再读取 main；验收期间的新合并进入下一次
  候选。KT main 若不同于点击时的工作流 SHA，要求重新点击，避免新源码搭配旧
  发布逻辑。
- 所有版本和依赖来自仓库源码。**不是只更新 SHA 就能发版**：发布前须在各仓
  main 选择未用版本，并对齐交叉依赖。不覆盖运行时代码、依赖或版本，不从旧
  post2 wheels 补文件。
- 首版统一构建五个最终发行包，不实现沿用旧 wheel。SGL payload 分布在 KT、
  KT-Kernel、SGLang、Transformers 和 Accelerate 中，因此即使某仓 Python 逻辑未改，只要
  载入的新 CUDA 片段变化，其发行版本也必须更新。
- `carriers.py` 消费本次六个 raw wheels，保留 runtime 文件、许可证和 SM90
  对象；仅生成 payload 清单/分片及 `WHEEL`、`RECORD`，不改 METADATA 依赖和
  版本。SGL native wheel 是中间输入，不作为第六个公开发行包上传。
- CPython 3.12、Linux x86-64、CUDA 12.8、Torch 2.9.1；全部 KT CPU variants，
  CUDA `80;86;89;90;120`。auditwheel 修复到 manylinux_2_35 后检查六个 KT CPU
  variants；仅含 CUDA fatbin 的扩展检查 SASS，SGL common_ops 必须保留上述架构。
  不要求纯 CPU 库含有 CUDA SASS。编译/静态检查不等于每种显卡实测。
- 每个最终 wheel 必须小于 104 MB。超限、缺架构、ABI 不兼容直接失败；不删除
  架构、不伪造 manylinux 标签。新版本 native 体积仍需真实构建确认。
- 构建、诊断、验收、上传证据分别保留；候选 wheelhouse 保留 90 天。
  GitHub artifact 并非永久存储，成功发布后建议另行归档用于长期追溯。

## 首次启用清单

**这些条件未完成前，不能宣称已经可以一键发版。**

1. 先合入 **#2195（含 release gate 接口）**，再合入 **#2194**。两份 PR 本身
   不会自动合并或发布。#2194 的 CPU CI 在依赖尚未进入 main 时会明确失败，
   不能忽略此依赖。
2. 准备隔离的 `kt-model-e2e` runner、固定 digest 的 CUDA/Python 3.12 镜像、
   公开 SFT 工具 wheelhouse、三个只读模型快照和可用下载通道。见
   [模型验收 README](../model-e2e/README.md)。不能直接给带发布密钥的旧 runner
   添加社区验收标签。
3. 原生构建 runner 使用 `[self-hosted,linux,x64,gpu,kt-cpu]`，需要 CUDA 12.8、
   C++ 工具链、足够空间和 manylinux_2_35 兼容的系统库。环境过新导致 ABI 不
   兼容时应修复环境，不跳过 auditwheel。
4. 配置 `KT_MODEL_E2E_ENABLED=true`、`KT_FOUR_MAIN_RELEASE_ENABLED=true`；
   `KT_MODEL_E2E_HOST_CONFIG` / `KT_MODEL_E2E_PYTHON` 指向受管理配置及解释器。
5. 在 `prod` environment 配置能上传五个项目的 `PYPI_API_TOKEN`，建议设置
   required reviewers。密钥只进入 GitHub-hosted 的上传 step，不进入编译/GPU
   容器。如果 prod 有审批，点击 Run workflow 后仍需批准该环境。
6. 对齐主线版本及依赖后先运行 `target=candidate`，确认真实编译、组装、干净
   安装和三模型均通过。再由维护者决定运行 `target=pypi`。后者构建自己的候选，
   仅上传该次验收过的同一批 wheels。

同时退休旧 `Release to PyPI` 和独立 `Release sglang-kt to PyPI` 入口：取消
自动 push 触发，手动点击旧入口会提示迁移并失败，旧发布 jobs 不会执行，避免
绕过新门禁。旧实现暂留文件中供审阅/追溯。

## 失败与重试

- 构建/候选验收失败：没有上传。修改源码后重新运行，得到新快照。
- 上传部分失败：PyPI 跨项目不是事务，已上传依赖无法自动撤销；KT 最后上传
  以减小影响。使用 **Re-run failed jobs**，保留原 raw-build 输出和 artifacts，
  不重新编译；仅允许跳过文件名和 SHA256 完全一致的文件，禁止 `--skip-existing`。
  内容不同则停止，不覆盖、不自动删包。
- PyPI 复验失败：保留失败证据，不自动删除/yank 包，由维护者决定如何处理。
- **Re-run all jobs** 是新候选，不保证得到相同字节；若旧版本已上传，版本预检
  会拒绝构建，不能拿它代替“重试上传”。
- 构建完成后只清理本次 mktemp 创建且带 run ID 标记的目录，不删除模型或旧
  环境。GPU 容器由 #2195 清理，宿主其他任务不受影响。

## 本地检查

PEFT/TRL 的获准本地构建由 `repack_training_tools.py` 生成。只替换 Transformers /
Accelerate 的依赖分发名，保留原有版本范围、markers、全部运行时文件和许可证；
用显式 local version 和输入/输出 SHA256 区分，不冒充原始上游 wheel。工具拒绝 LF
重打包和放宽依赖范围。正式交付还需公开提供这两个 wheel 及 provenance；它们不属于
五个核心 PyPI 项目。本项不解决 LF main 自身的依赖冲突，不能绕过最终安装验收。

依赖 #2195 的文件已在同一源码树中时：

```bash
PYTHONPATH=.github/model-e2e python -m pytest -q .github/release
python -m unittest discover -s .github/model-e2e -p 'test_*.py'
actionlint -config-file .github/release/actionlint.yaml .github/workflows/release-four-main.yml
```

CPU 测试覆盖清单篡改、错误 run/attempt、缺少模型结果、同版本不同 wheel、
部分上传、重试和目录清理；不会接触 PyPI 凭据，不能替代真实 CUDA 构建、Actions
及模型 E2E 验收。

实现参考：[GitHub reusable workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations)、
[Python wheel/RECORD 规范](https://packaging.python.org/en/latest/specifications/binary-distribution-format/)。
