# 模型 E2E 验收 CI（独立 Draft）

这个 PR **只负责验收，不构建发行包、不上传 PyPI、不自动合并 PR**。
它与 [四仓 main 发版 PR #2194](https://github.com/kvcache-ai/ktransformers/pull/2194) 分开维护，后续通过候选 wheel artifact 对接。

## 1. 两个用途

| 入口 | 安装对象 | 触发方式 |
| --- | --- | --- |
| 社区 PR | 可信构建流程产出的五个最终 wheels；PR 使用批准的完整 head SHA，其他三仓固定当时的 main SHA | 维护者审阅后手动触发 |
| 正式 PyPI | 在全新环境执行 `pip install "ktransformers[sglang,sft]"`；也支持指定 KT 版本 | 手动触发；预留 `workflow_call` 供发版流程调用 |

四仓指 `kvcache-ai/{ktransformers,sglang,transformers,accelerate}`；五个发行包包含 `kt-kernel`。
测试不会用本地源码、editable 安装或 `PYTHONPATH` 覆盖这些包。
正式 PyPI 模式检查安装报告中的下载地址；PR 模式核对构建 run、四仓 SHA、wheel SHA256、包名、版本和实际导入路径。
首版固定 CPython 3.12；PyPI 模式记录触发时预期的 KT 版本，若 pip 静默回退到旧版本则失败，不能把旧版本可运行误报为新版已验收。

“其他三仓 main”在一次验收开始时固定，不会在运行中漂移。如果候选构建之后 main 已更新，本次检查会拒绝旧候选，要求重新构建。

## 2. 验收标准

| 模型 | 内容 | 通过条件 |
| --- | --- | --- |
| Qwen3-30B-A3B-Instruct-2507 | KT LoRA，1 GPU | 完成 **1 个 optimizer step**，原始 loss 有限，正常退出 |
| DeepSeek-V3.1 | KT LoRA，8 GPU | 同上；每个 rank 都有完整证据 |
| GLM-5.3-Flash | TP4 启动 SGLang，完成一次 OpenAI 兼容问答 | 服务正常；确实生成 token；最终回答语义正确且没有明显乱码 |

**不要求 loss 下降、不做 adapter 重载、不设置速度门槛，也不增加长上下文、多模态或完整 coding-agent 评测。**
训练使用仓库内固定的简单数据，`max_steps=1`、`gradient_accumulation_steps=1`、`logging_nan_inf_filter=false`。
`lora_probe.py` 只观测原有训练函数返回的 loss 和训练回调，不修改 loss 或四仓运行时。
GLM 使用固定问题“法国首都是什么”，检查最终回答中的 Paris / 巴黎；不把 reasoning 非空当成成功。

## 3. qj5090 排队与隔离

1. qj5090 有 GPU 计算进程、明显显存占用或 GPU 活动时，记录“等待 qj5090 空闲”，不加载模型。
2. 获取共享 `flock` 后，连续三次确认空闲，再创建容器、全新 venv 和独立 JIT 缓存。
3. Qwen → DeepSeek → GLM 串行执行；锁覆盖安装、测试和容器销毁。每个模型开始前再次检查 GPU，安装期间若有人工任务启动则继续等待。
4. 出错/超时/取消，只清理本次创建的容器和进程组，绝不按模型名或 GPU PID 批量杀进程。
5. 等待超时记为 **资源未获得**，不归因为模型失败；未完成三个测试也不能显示绿色验收。

不使用会替换旧 pending run 的全局 Actions concurrency group。单 runner 自身排队，额外共享锁防止多个控制器重叠。
锁是 advisory lock：人工任务如果不用同一个锁，仍可能在空闲检查之后启动。需要无竞争调度时，人工任务也应获取该锁；CI 不抢占后来启动的人工任务。

容器通过 CDI 分配 GPU，使用只读模型、只读 harness、非特权宿主身份、去除 capabilities、独立网络/IPC，**不挂载宿主 HOME、Docker socket、发布凭据或旧 venv**。
Rootless Docker 内使用映射到普通宿主用户的容器 UID 0；rootful Docker 内使用普通用户 UID，拒绝 rootful 的 root 控制器。不会为了兼容 rootless 而关闭全局 cgroup 检查或启用 privileged。
运行环境与缓存放入上限 96 GiB 的临时内存文件系统，按需占用内存；容器删除即释放。只保留日志，不将权重、adapter 或环境上传为 artifact。
容器退出并删除之后才导出日志，拒绝符号链接和特殊文件，避免 artifact 收集读取宿主机文件。

## 4. 安全与启用条件

社区代码不应直接运行在当前持有 PyPI 密钥的发布 runner 上。
新增工作流默认关闭，没有配置 `KT_MODEL_E2E_ENABLED=true` 就会停止，不会占用 GPU。

管理员启用前需要完成：

- 准备不持有发布凭据的专用 runner 账号，添加 `kt-model-e2e` 标签；不要直接给旧发布 runner 加标签。
- 配置 `qj5090-model-e2e` environment 的人工审批和 main 分支限制；工作流也只接受从 KT main 触发。
- 准备公开、固定 digest 的 CUDA 12.8 / Python 3.12 测试基础镜像。现有生产服务镜像不能直接作为最终验收工具链。
- 确认 Docker 支持 CDI 且 `nvidia-ctk cdi list` 包含 `nvidia.com/gpu=all`；rootless 不使用会触发 legacy BPF/cgroup 权限错误的 `--gpus all` 路径。
- 在镜像 `/opt/kt-e2e-tooling/` 中准备 `requirements.txt`（全部带哈希）、`wheels/` 和 `provenance.json`，固定公开 LlamaFactory 及辅助依赖的来源。
- **训练工具不能重新安装或覆盖 KT 五包、上游同名命名空间或 Torch**。安装前 dry-run 校验，安装后 `pip check` 和导入文件哈希复核；不能靠忽略依赖错误或旧实验的私有源码覆盖来通过。
- 按 `qj5090.example.json` 固定三份只读模型、模型 revision 和 config SHA256；模型权重完整性应在入库时核验，本工作流不在每次执行时重新哈希 TB 级权重。
- 设置 `KT_MODEL_E2E_HOST_CONFIG`、`KT_MODEL_E2E_PYTHON`，创建共享锁目录。宿主 controller 需要 Python ≥3.11。
- 配置 `KT_MODEL_E2E_BUILD_WORKFLOW` 为可信 main 构建工作流的路径。

容器不是执行任意恶意代码的绝对安全边界；共享 GPU 宿主仍有内核/驱动风险。因此第一版只允许运维/维护者审查完整 SHA 后批准执行，不对所有 fork 自动启动 GPU 工作。
参见 [GitHub Actions 安全建议](https://docs.github.com/en/actions/reference/security/secure-use)。

## 5. 社区 PR 候选 artifact 接口

构建端负责提供名为 `model-e2e-candidate` 的 artifact，内含 `candidate.json` 和 **五个最终可安装 wheels**，不接受 `sgl-kernel-kt` 原始中间包。

`candidate.json` 结构如下，`wheels` 必须包含全部五包：

```json
{
  "schema": 1,
  "build_run_id": 123456,
  "build_workflow_sha": "<构建工作流的完整 commit SHA>",
  "sources": {
    "ktransformers": {"repository": "kvcache-ai/ktransformers", "sha": "<完整 SHA>"},
    "sglang": {"repository": "kvcache-ai/sglang", "sha": "<完整 SHA>"},
    "transformers": {"repository": "kvcache-ai/transformers", "sha": "<完整 SHA>"},
    "accelerate": {"repository": "kvcache-ai/accelerate", "sha": "<完整 SHA>"}
  },
  "wheels": {
    "ktransformers": {"file": "ktransformers-X-py3-none-any.whl", "version": "X", "sha256": "<SHA256>"},
    "kt-kernel": {"file": "kt_kernel-X-cp312-cp312-manylinux_2_35_x86_64.whl", "version": "X", "sha256": "<SHA256>"},
    "sglang-kt": {"file": "sglang_kt-X-py3-none-any.whl", "version": "X", "sha256": "<SHA256>"},
    "transformers-kt": {"file": "transformers_kt-Y-py3-none-any.whl", "version": "Y", "sha256": "<SHA256>"},
    "accelerate-kt": {"file": "accelerate_kt-Z-py3-none-any.whl", "version": "Z", "sha256": "<SHA256>"}
  }
}
```

artifact 必须来自本仓已成功完成的、main 上手动触发的批准构建工作流。哈希证明候选未变化；源码 provenance 依赖可信构建端，不代表仅凭 wheel 哈希就能证明源码来源。
**#2194 目前只有 raw build，尚不提供这个最终 artifact。构建端对接完成之前，不能宣称社区 PR 全链路已可用。**

触发示例（合入并配置之后）：

```bash
gh workflow run model-e2e.yml --repo kvcache-ai/ktransformers --ref main \
  -f mode=pypi -f version=latest

gh workflow run model-e2e.yml --repo kvcache-ai/ktransformers --ref main \
  -f mode=pr -f pr_repository=kvcache-ai/sglang \
  -f pr_number=123 -f pr_sha=<批准的完整SHA> -f build_run_id=<候选构建runID>
```

## 6. 结果与当前边界

- GitHub Summary 展示三项通过/失败/未执行；artifact 保存安装报告、版本、包文件哈希、训练 loss、GLM 请求/回答和服务日志。
- KT PR 将状态回写到**当时批准的 head SHA**。其他三仓先提供报告；跨仓回写需要单独的 GitHub App 权限，不把 PAT 放到 GPU 宿主。
- 发布前候选和发布后 PyPI 复验可以复用这套测试；本 PR 不改发版触发条件、不上传包、不启用自动合并。
- 当前是 Draft。正式启用前仍需完成专用 runner/基础镜像/公开 SFT 工具链配置、候选构建接口对接，以及两种入口各一次完整实机验收。不能用单元测试、历史成绩或一次 GLM 成功替代三项验收。

本地测试（无需 GPU / Torch）：

```bash
python3 -m unittest discover -s .github/model-e2e -p 'test_*.py' -v
```

### qj5090 首次现场检查（2026-09-10）

- 已验证真实 CUDA 进程存在时排队，不启动待测任务；占用探针自行退出。
- 已验证 rootless Docker 的 CDI GPU 路径和 CUDA 小张量操作，无需 privileged 或修改宿主配置。
- 已发现并修正控制器的 rootless 用户映射、memlock 参数，以及误用 Python 3.11 时 pip 回退旧栈的问题。
- 模型 E2E 与完整 Actions 链路尚未验收；以上基础设施检查不代表三个模型通过。
