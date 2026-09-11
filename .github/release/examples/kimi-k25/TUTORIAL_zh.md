# Kimi K2.5：使用 LF main 微调

路线：原始问答 JSON → 一次 CPU 数据准备 → LF 标准 CLI 训练 → 保存并加载 LoRA。

LF 固定为 official main `100e9a42`，不添加自定义模板或修改 Trainer。
其余运行时仍须使用 [README 中列出的配套版本与补丁](README.md#environment-boundary)；
本文不是“任意 PyPI 最新版本组合均可用”的承诺。
LF main 的两步训练和保存已通过，产物与此前成功版本的哈希一致；
本次新产物推理复测进程收到 SIGKILL，尚未完成。详细验收范围见 README。

## 0. 获取示例文件

在自己的训练机器上新建工作目录，将 [prepare_kimi_data.py](prepare_kimi_data.py)、
[train.yaml](train.yaml) 和 [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml) 下载到同一目录。
在 GitHub 查看文件时，使用 Raw / Download raw file 下载原始内容，不要保存网页。
下文命令均在这个工作目录执行，不依赖我们的 sap4 路径。

`prepare_kimi_data.py` 是本示例附带的独立数据转换脚本，**不会由 LF 自动执行**。
先按第 2 节转换原始数据，再把输出目录填入 YAML 的 `dataset_dir`；不能跳过转换直接开训。

## 1. 准备环境和原始数据

在已经安装配套 LF、Transformers-KT、Accelerate-KT、KT-Kernel 的环境中执行。
参考硬件为 sap4：8 张 GPU、CPU AMX、96 worker threads、CPU TP2。
模型目录使用原始 Kimi-K2.5 checkpoint，routed experts 保持 RAWINT4 group-32，
不需要把全部专家转换成 BF16。

训练数据为普通 JSON 数组，支持纯文本单轮问答及可选的 `system`：

```json
[
  {"prompt": "你好", "answer": "你好喵。"},
  {"system": "请简洁回答。", "prompt": "17 加 25？", "answer": "42。"}
]
```

训练集、验证集提前划分好。脚本不重新划分、去重或打乱数据；不支持多轮、工具、图片和视频。
输入不应包含已经拼好的对话标记或 `<think>` 标记。

## 2. 转换一次

进入本目录，执行：

```bash
python prepare_kimi_data.py \
  --model /path/to/Kimi-K2.5 --trust-remote-code \
  --input /path/to/neko_train.json \
  --eval-input /path/to/neko_eval.json \
  --output-dir /path/to/prepared-neko
```

没有验证集时去掉 `--eval-input`。`--trust-remote-code` 允许执行所选模型的 tokenizer 代码，
请使用可信模型来源。这里只加载 tokenizer，不加载模型权重或占用 GPU 计算。

脚本通过 Kimi 原生模板关闭 thinking，把对话前缀放进 `prompt`，把结束标记放进 `answer`。
它逐条检查独立编码 prompt/answer 后的 token 与原生完整对话一致；不一致直接报错。

输出包括：

- `neko_train.json`、可选的 `neko_eval.json`；
- `dataset_info.json`：LF 所需的数据字段映射；
- `preprocessing_manifest.json`：样本数、原始/输出文件哈希、模板哈希与 token/labels 校验摘要。

原始文件不会被覆盖；输出目录必须是新目录。相同数据和 tokenizer 只需转换一次，
不要将已经转换的文件再次输入脚本。更换模型/tokenizer 后应重新转换。

## 3. 修改两份 YAML

`train.yaml` 中只需先修改四个路径：

```yaml
model_name_or_path: /path/to/Kimi-K2.5
kt_weight_path: /path/to/Kimi-K2.5
dataset_dir: /path/to/prepared-neko
output_dir: /path/to/new-kimi-output
```

保留以下设置，不要把原始 JSON 直接交给 `empty` 模板：

```yaml
dataset: neko_train
template: empty
train_on_prompt: false
packing: false
neat_packing: false
```

非 thinking 格式已由预处理生成；这里不使用 `enable_thinking` 开关。
LF 仍执行正常分词、长度限制及 answer-only loss，prompt 不参与 loss。

`fsdp2_8gpu.yaml` 已配置 8 卡 FSDP2 和参数 CPU offload，8 卡参考测试无需再改。
其中 `DeepseekV3DecoderLayer` 是 Kimi 文本 decoder 的真实类名，不要改成顶层模型名。
当前 CPU/GPU activation 均重算；不要开启 RAWINT4 尚不支持的 CPU retain / GPU recompute 组合。

## 4. 启动短训

确认 8 张卡空闲后，从本目录执行：

```bash
USE_KT=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file fsdp2_8gpu.yaml \
  --main_process_port 29761 --no_python llamafactory-cli train train.yaml
```

这是标准 LF CLI，不需要自定义启动脚本或 `PYTHONPATH`。
当前配置为 B1/GAS1、S512 上限、2 optimizer steps；S512 不是强制补齐长度。
`save_strategy: 'no'` 仅关闭中间 checkpoint，训练结束仍会保存最终 LoRA。

正式长训前先确认短训成功。调整 `cutoff_len` 时同步调整 `kt_config.kt_model_max_length`；
需要中间 checkpoint 时改用 `save_strategy: steps` 并设置 `save_steps`。
若启用验证，还需显式设置 `eval_dataset: neko_eval`、`do_eval` 与验证频率；
仅生成验证 JSON 不会自动开启验证。长序列吞吐和效果不由两步 smoke 保证。

## 5. 检查保存并加载

输出目录应同时包含 `adapter_model.safetensors`、`fused_expert_lora.safetensors`
和状态为 `ready` 的 `kt_adapter_manifest.json`，不能只保留普通 adapter。
当前配置省略 optimizer state；再次加载可做 adapter 续训，不等于精确恢复优化器。
本配置的两份 adapter 合计约 10 GiB，转换与运行时拆分还需要额外空间，避免写满系统盘。

先用配套候选 KT-Kernel 中的转换命令生成 SGLang 格式：

```bash
kt-convert-lora /path/to/new-kimi-output /path/to/converted-kimi-lora \
  --base-model-name-or-path /path/to/Kimi-K2.5 --lora-alpha 16
```

推理使用包含 [SGLang #94](https://github.com/kvcache-ai/sglang/pull/94) 补丁的运行时。
以下为已使用过的 4 卡推理配置；请先结束训练、释放对应 GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /path/to/Kimi-K2.5 --trust-remote-code \
  --host 127.0.0.1 --port 30000 --served-model-name kimi-k25 \
  --tensor-parallel-size 4 --dtype bfloat16 --attention-backend triton \
  --kt-method RAWINT4 --kt-weight-path /path/to/Kimi-K2.5 \
  --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --enable-lora --lora-backend triton --lora-paths neko=/path/to/converted-kimi-lora \
  --context-length 2048 --max-total-tokens 4096 --chunked-prefill-size 256 \
  --max-running-requests 4 --mem-fraction-static 0.75 \
  --disable-cuda-graph --disable-radix-cache --watchdog-timeout 900
```

可通过 `SGLANG_KT_LORA_CACHE_DIR` 指定有足够空间的运行时拆分缓存目录。
这里是静态绑定 LoRA 的服务，不要用裸 base-model 请求混测。

请求必须选择 adapter，且推理继续使用 Kimi 原生对话模板：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k25:neko","messages":[{"role":"user","content":"17加25是多少？"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"thinking":false}}'
```

确认日志包含 expert LoRA 加载、MLA LoRA 修正，并且请求正常完成。
这证明加载与生成闭环，不代表两步训练已经获得风格或质量提升。
