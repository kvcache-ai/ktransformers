# Kimi K2.5 LoRA 微调：安装、训练与对话

[English version](README_EN.md)

本教程使用 KTransformers `0.7.0.post4` 和 NekoQA 数据集，让 Kimi K2.5
学会猫娘风格的语气，再用 SGLang 加载训练结果进行对话。原始模型权重不变，
训练结果单独保存为 LoRA。

按下面四步执行即可；只需填写模型和输出目录两个路径。想先确认环境能否跑通，
可在完成第 2 步后使用文末的「4 步试跑与续训」；它不是正式训练的必经步骤。

## 开始前

实测参考配置：

| 项目 | 配置 |
| --- | --- |
| GPU | 8 张 RTX 5090 训练，4 张 RTX 5090 推理 |
| CPU / 内存 | 双 AMD EPYC 9355，约 1.5 TiB RAM |
| 系统 | Linux x86_64、glibc 2.35+；Python 3.11 / 3.12 分别验收 |
| CUDA | CUDA 12.8 工具链、C++ 编译器；实测驱动 580.173.02 |

确认 `nvidia-smi`、`nvcc --version` 正常，使用的 GPU 空闲。磁盘除了容纳完整
模型、安装环境和缓存，还要存训练结果：**每个 checkpoint 约 29 GiB，转换后的
推理 LoRA 约 9.6 GiB**。本训练配置最多保留两个 checkpoint，另有最终 LoRA
和临时文件；请使用空间充足的持久化磁盘，不要把输出放到 `/dev/shm`。

本文介绍 **Kimi K2.5 纯文本 LoRA 微调**。

## 1. 安装

**核心组件从 PyPI 安装；LF、PEFT、TRL 三个配套安装包随 Release 工具包提供。**
按以下命令安装完整的训练与推理环境，无需克隆仓库或修改源码。

在容量充足的磁盘上打开 Bash，下载并解压工具包：

```bash
mkdir kimi-post4-work
cd kimi-post4-work
curl --fail --location --retry 5 --remote-name \
  https://github.com/kvcache-ai/ktransformers/releases/download/v0.7.0.post4/kimi-k25-post4-user-kit-r2.tar.gz
printf '%s  %s\n' \
  88fa5e473e06b67b5ac39605de82da09207ba1b82555f6d4b98d2b6a81de8813 \
  kimi-k25-post4-user-kit-r2.tar.gz | sha256sum --check
tar -xzf kimi-k25-post4-user-kit-r2.tar.gz
cd kimi-k25-post4
```

后续命令都在这个 `kimi-k25-post4/` 目录、同一个终端中执行。工具包已经包含
训练 YAML、数据处理脚本和安装清单。

接着创建两个独立环境：`train-env` 用于训练，`serve-env` 用于推理，避免两套
依赖相互影响。两者使用同一种 Python，无需手动激活环境。
只需选择下面的 `KIMI_PYTHON`，安装清单会自动匹配；国内网络可换成清华源：

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="$PWD/cache/huggingface"
export XDG_CACHE_HOME="$PWD/cache"
export TRITON_CACHE_DIR="$PWD/cache/triton"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
export KIMI_PYPI_INDEX=https://pypi.org/simple
# 国内网络可将上行替换为：https://pypi.tuna.tsinghua.edu.cn/simple
KIMI_PYTHON=python3.12  # 使用 Python 3.11 时改为 python3.11
KIMI_PYTHON_TAG=$("$KIMI_PYTHON" -c '
import sys
assert sys.version_info[:2] in ((3, 11), (3, 12)), "Use Python 3.11 or 3.12"
print("cp%d%d" % sys.version_info[:2])
') || exit 1
"$KIMI_PYTHON" -m venv train-env
train-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
train-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --no-binary=antlr4-python3-runtime \
  --require-hashes --find-links training-tools -r "locks/$KIMI_PYTHON_TAG/train.lock"
train-env/bin/python -m pip check

"$KIMI_PYTHON" -m venv serve-env
serve-env/bin/python -m pip install --index-url "$KIMI_PYPI_INDEX" pip==25.2
serve-env/bin/python -m pip install \
  --index-url "$KIMI_PYPI_INDEX" --timeout 120 --retries 10 --resume-retries 10 \
  --only-binary=:all: --require-hashes -r "locks/$KIMI_PYTHON_TAG/serve.lock"
serve-env/bin/python -m pip check
```

**安装成功的标志：**两次 `pip check` 都输出 `No broken requirements found.`。
安装清单已固定配套版本，请不要再往这两个环境里安装上游 `transformers`、
`accelerate` 或替换为最新版 LLaMA-Factory。

## 2. 准备模型和数据

修改下面的 `KIMI_MODEL` 和 `KIMI_OUTPUT`，分别指向模型目录和**新的**训练输出目录。
NekoQA 会自动下载并处理，不需要手动填写数据路径。

若已有相同版本的模型或数据，跳过对应的 `hf download` 命令即可。
下载需要访问 Hugging Face；离线机器可从联网机器完整拷贝模型和数据，
换 PyPI 源不能解决 Hugging Face 下载问题。

```bash
export KIMI_MODEL=/absolute/path/to/Kimi-K2.5
export KIMI_OUTPUT=/absolute/path/to/new-kimi-output
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$KIMI_OUTPUT"

train-env/bin/hf download moonshotai/Kimi-K2.5 \
  --revision 54383e83fa343a1331754112fb9e3410c55efa2f --local-dir "$KIMI_MODEL"
train-env/bin/hf download liumindmind/NekoQA-10K NekoQA-10K.json \
  --repo-type dataset --revision 1b2110c996a8237823b86c1a3d3e8a6762b38430 \
  --local-dir data-source
train-env/bin/python tools/split_nekoqa.py \
  --input data-source/NekoQA-10K.json --output-dir neko-splits
train-env/bin/python tools/prepare_kimi_data.py \
  --model "$KIMI_MODEL" --trust-remote-code \
  --input neko-splits/neko_train.json --eval-input neko-splits/neko_eval.json \
  --output-dir prepared-neko
```

**准备成功的标志：**`prepared-neko/` 中出现处理后的数据和 `dataset_info.json`。
脚本会划分训练、验证和独立问答，并按 Kimi 的聊天格式处理数据。
YAML 中的 `template: empty` 与此配套，保持不变，不要跳过预处理。

使用完整原始模型，不转成全量 BF16，也不修改模型文件。
`--trust-remote-code` 会执行下载的模型代码，请确认模型来源可信。

## 3. 开始风格训练

这条命令使用 8 张 GPU，按 [train-neko.yaml](train-neko.yaml) 训练一轮 NekoQA。
每卡 batch size 为 1，累积 8 次梯度再更新，序列长度上限为 4096。
路径由命令传给 YAML，不必再次编辑配置文件：

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-neko.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/neko"
```

**正常现象：**日志持续输出 step、loss 和 grad norm，没有 `nan` / `inf`；
一轮约 149 步，第 100 步保存并验证，产物在
`$KIMI_OUTPUT/neko/checkpoint-100/`。等训练正常结束，再进入推理步骤。

这个配置训练 attention 和 expert 两部分 LoRA，rank 8、alpha 16、dropout 0；
不开 packing，CPU/GPU activation 均重算。第一次运行先保持这些配置不变。
4096 是长度上限，不会把每条样本补成 4096 token。

后面使用 `checkpoint-100`，因为此前已验证它出现明显风格。保持一轮训练的配置，
**不要为了取第 100 步产物改成 `max_steps: 100`**，那会改变学习率计划。
如果训练中断，保留完整 checkpoint，并在上述命令末尾追加
`resume_from_checkpoint="$KIMI_OUTPUT/neko/checkpoint-100"` 即可从这个保存点继续；
路径应指向实际存在的 checkpoint，其余训练配置保持不变。

## 4. 加载 LoRA，开始对话

**先转换训练产物。**训练保存的 LoRA 分为普通模块和 expert 两部分，下面的脚本
会一起转换成 SGLang 使用的格式，原 checkpoint 不变：

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/neko/checkpoint-100"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-neko"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

**再启动推理服务。**确认训练已经退出，GPU 0–3 空闲，在同一个终端执行：

```bash
set -o pipefail
PATH="$PWD/serve-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3 \
serve-env/bin/python -m sglang.launch_server \
  --model-path "$KIMI_MODEL" --trust-remote-code \
  --served-model-name kimi --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 4 --dtype bfloat16 --context-length 2048 \
  --max-total-tokens 4096 --chunked-prefill-size 256 --max-running-requests 4 \
  --mem-fraction-static 0.75 --disable-cuda-graph --disable-radix-cache \
  --attention-backend triton --grammar-backend llguidance \
  --kt-method RAWINT4 --kt-weight-path "$KIMI_MODEL" \
  --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --random-seed 42 --watchdog-timeout 900 \
  --enable-lora --lora-backend triton --lora-paths "neko=$KIMI_ADAPTER" \
  2>&1 | tee "$KIMI_ADAPTER.server.log"
```

第一次启动可能编译 kernel 并加载模型，需要等待；服务会一直占用这个终端。
日志中应有 `Loaded KT expert LoRA for layer ...`，覆盖第 1–60 个 expert 层。

**最后发一个问题。**另开终端，等 `/health` 返回成功，再发送对话请求：

```bash
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/health
curl --noproxy 127.0.0.1 --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi:neko","messages":[{"role":"user","content":"我今天学习有点累，请用两句话鼓励我。"}],"temperature":0,"seed":42,"max_tokens":1024,"chat_template_kwargs":{"thinking":false,"enable_thinking":false}}'
```

请求中的 `kimi:neko` 表示使用刚加载的 LoRA。查看响应的
`choices[0].message.content`：应该有完整回答，并表现出训练后的语气。
请求没有添加“请扮演猫娘”等提示，用来观察模型自己学到的风格。

至此已完成安装、训练和推理。下面的附录按需使用。

## 附录

<details>
<summary>4 步试跑与续训：想先确认环境能跑通时使用</summary>

在完成第 1、2 步后运行，使用相同终端和工作目录，确认 8 张 GPU 空闲。
4 步用于验证训练、保存、续训和加载；风格训练请使用第 3 步的正式配置。
两次试跑和一次 adapter 转换另需至少 140 GiB 持久化磁盘空间，不含模型、环境和缓存。

先运行 4 步，输出单独放在 `smoke-a/`，不影响正式训练目录：

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-a"
```

成功后应有 `checkpoint-2`、`checkpoint-4`，且 loss / grad norm 没有 `nan` 或 `inf`。
接着从第 2 步的保存点重新启动训练，结果写入另一个目录：

```bash
PATH="$PWD/train-env/bin:$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
train-env/bin/accelerate launch --config_file examples/fsdp2_8gpu.yaml \
  --no_python train-env/bin/llamafactory-cli train examples/train-smoke.yaml \
  model_name_or_path="$KIMI_MODEL" kt_weight_path="$KIMI_MODEL" \
  dataset_dir="$PWD/prepared-neko" output_dir="$KIMI_OUTPUT/smoke-b" \
  resume_from_checkpoint="$KIMI_OUTPUT/smoke-a/checkpoint-2"
```

日志应从 step 2 恢复，最终到 step 4。保持卡数、数据和训练配置不变；
`max_steps: 4` 表示总步数，不是额外训练 4 步。
`resume_from_checkpoint` 恢复训练状态，不能用只加载 LoRA 的 `adapter_name_or_path` 替代。

如需验证推理，执行下面的转换命令，再执行第 4 步的**服务启动和对话命令**。
不要重新执行第 4 步中 `checkpoint-100` 的路径赋值和转换命令：

```bash
export KIMI_CHECKPOINT="$KIMI_OUTPUT/smoke-b/checkpoint-4"
export KIMI_ADAPTER="$KIMI_OUTPUT/sglang-smoke"
train-env/bin/python tools/convert_kt_to_sglang_adapter.py \
  "$KIMI_CHECKPOINT" "$KIMI_ADAPTER" --base-model-name-or-path "$KIMI_MODEL"
```

试跑结束后，在服务终端按 Ctrl-C 停止推理，确认 GPU 释放，再执行第 3 步的正式训练。
正式训练仍从原始模型开始，不继承这 4 步试跑的 LoRA。

</details>

<details>
<summary>检查训练效果：保存 32 条独立问答、与原模型对照</summary>

第 2 步已在 `neko-splits/heldout.json` 中留出 32 条不参与训练的问题。
不要添加猫娘角色提示，分别检查语气是否变化、内容是否相关、回答是否完整。

若要与原模型比较，需要先停止 LoRA 服务，再用第 4 步的命令启动一个新进程，
去掉 `--enable-lora`、`--lora-backend`、`--lora-paths`，请求中改用 `model: kimi`。
本版本 CPU expert LoRA 在启动时加载；只在同一服务中切换模型名，并不能关闭它。

在第二个终端进入同一个 `kimi-k25-post4/` 目录后执行。输出文件必须不存在，
避免覆盖已有结果。跑独立 base 对照时，将 `model_id` 改为 `kimi`，并更换输出文件名。

```bash
serve-env/bin/python - <<'PY'
import json
from pathlib import Path
import requests

model_id = "kimi:neko"
questions = json.loads(Path("neko-splits/heldout.json").read_text())
session = requests.Session()
session.trust_env = False
with Path("neko-heldout-responses.jsonl").open("x", encoding="utf-8") as output:
    for index, row in enumerate(questions, 1):
        body = {
            "model": model_id,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": 0, "seed": 42, "max_tokens": 1024,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        response = session.post("http://127.0.0.1:30000/v1/chat/completions", json=body, timeout=900)
        response.raise_for_status()
        output.write(json.dumps({"prompt": row["prompt"], "response": response.json()}, ensure_ascii=False) + "\n")
        output.flush()
        print(f"{index}/{len(questions)}", flush=True)
PY
```

打开 `neko-heldout-responses.jsonl`，检查 `choices[0].message.content` 和
`finish_reason`。若为 `length`，回答被长度上限截断，不能算完整回答。

除语气变化外，建议同时检查事实准确性和指令遵循，例如“17+25 只输出数字”、
原样输出文字和 JSON 格式指令。

</details>

<details>
<summary>遇到问题时先检查什么</summary>

| 现象 | 先检查 |
| --- | --- |
| pip 下载超时 | 重试相同安装命令，可改用清华源。若镜像还没有该版本，切回 `https://pypi.org/simple`；保留版本和校验参数。 |
| `hf download` 超时 | 检查 Hugging Face 连通性；可在联网机器下载固定 revision 后完整拷贝，换 PyPI 源对此无效。 |
| pip 依赖冲突、找不到 LF/PEFT/TRL 版本 | 是否在新环境，是否下载了配套资料包并使用 `--find-links training-tools`。 |
| 训练 loss 异常或模板重复 | 是否先执行数据准备，且保持 `template: empty`、`packing: false`。 |
| 加载时 rank 0 消失、Gloo connection closed | 查看 rank 0 日志、主机 OOM 记录和各 NUMA 节点的可用 RAM；全机仍有空闲内存时也可能单节点不足。避免把大量 checkpoint 放进 RAM 或同时加载多个模型。 |
| 续训变成从零开始 | 是否传入完整 checkpoint 的 `resume_from_checkpoint`，而非只加载 adapter。 |
| 推理像 base 或只有部分变化 | 转换是否包含两类 LoRA，是否以新进程加载，expert 1–60 层是否都加载，以及请求名是否为 `kimi:neko`。 |

不要使用 `--no-deps` 或关闭版本检查来绕过安装冲突。也不要额外设置
`FORCE_TORCHRUN` 或在训练命令外再套一层 `torchrun`。

</details>

<details>
<summary>固定版本、配置文件与验收细节</summary>

安装清单已包含以下版本，用户不需要逐项手动安装：

| 来源 | 固定版本 |
| --- | --- |
| PyPI：`ktransformers`、`kt-kernel`、`sglang-kt` | `0.7.0.post4` |
| PyPI：`transformers-kt` | `5.6.0.post5` |
| PyPI：`accelerate-kt` | `1.14.0.post3` |
| Release `training-tools/`：LLaMA-Factory | `0.9.6.dev0+kt.20260912` |
| Release `training-tools/`：PEFT、TRL | `0.18.1+kt.20260912`、`0.24.0+kt.20260912` |

LF 使用[固定的 KT 安装分支](https://github.com/yyj6666667/LlamaFactory/tree/564574eb0214840ea4b9464b2f9178a52a075d73)。
PEFT/TRL 只调整依赖元数据，不修改运行时代码；这三个配套包没有上传到 PyPI。
上游 `transformers` / `accelerate` 与 KT 包使用同名 Python 模块，不能混装。
安装使用 Torch 2.9.1；ANTLR 4.9.3 从校验过的官方源码包构建，其余依赖使用 wheel。

`locks/cp311/` 和 `locks/cp312/` 分别固定对应解释器的版本及文件哈希。
3.11 使用 NumPy 2.4.6、SciPy 1.17.1、ContourPy 1.3.3；3.12 保留已验收版本。
两种 Python 共用相同训练 YAML，不需要修改训练参数。

仓库与工具包的 YAML 配置值一致，文件对应关系如下：

| 仓库文件 | 工具包内文件 | 用途 |
| --- | --- | --- |
| [train-neko.yaml](train-neko.yaml) | `examples/train-neko.yaml` | 一轮风格训练 |
| [train.yaml](train.yaml) | `examples/train-smoke.yaml` | 4 步试跑 |
| [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml) | `examples/fsdp2_8gpu.yaml` | 8 卡启动配置 |

数据按 seed 42 去重划分为 9,477 条训练、468 条验证、32 条独立问答。
预处理使用 Kimi 原生 non-thinking 模板，只有回答参与 loss；保持
`template: empty`、`train_on_prompt: false`，不要直接传入未经处理的 Neko JSON。
处理后的最长样本为 2,120 token。

完整 checkpoint 应包含以下内容，`save_only_model: false` 必须保留：

- `adapter_model.safetensors`、`fused_expert_lora.safetensors` 和 adapter 配置；
- `kt_optimizer.index.json`、8 个 `optimizer_rank_*.pt`；
- `scheduler.pt`、`trainer_state.json`、8 个 `rng_state_*.pth`。

运行过附录中的两次试跑后，可比较连续训练与续训的 LoRA 是否完全相同。
比较的是张量内容，不依赖文件的序列化顺序：

```bash
train-env/bin/python - "$KIMI_OUTPUT" <<'PY'
from pathlib import Path
import sys
import torch
from safetensors import safe_open

root = Path(sys.argv[1])
for name in ("adapter_model.safetensors", "fused_expert_lora.safetensors"):
    a = root / "smoke-a/checkpoint-4" / name
    b = root / "smoke-b/checkpoint-4" / name
    with safe_open(a, framework="pt", device="cpu") as left, safe_open(b, framework="pt", device="cpu") as right:
        assert set(left.keys()) == set(right.keys()), name
        for key in left.keys():
            assert torch.equal(left.get_tensor(key), right.get_tensor(key)), (name, key)
    print(name, "EXACT_RESUME_PASSED")
PY
```

两项都应输出 `EXACT_RESUME_PASSED`。完整验收还比较了各 rank 的 optimizer、
RNG、scheduler 和恢复后的 loss。adapter 转换导出 610 个普通 LoRA tensor 和
138,240 个 expert LoRA tensor，SGLang 加载了第 1–60 个 expert 层。

源码来源、包校验值和验收结果见[发布与验证记录](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.0.post4)。
保留训练日志、完整 checkpoint、数据划分记录和推理响应，便于复现自己的结果。

</details>
