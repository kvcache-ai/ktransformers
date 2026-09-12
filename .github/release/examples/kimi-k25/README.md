# Kimi K2.5：RAWINT4 LoRA 训练、续训与推理

候选教程，**尚未完成干净安装验收，不能作为已发布版本的成功承诺**。
本轮 LF 使用获准的 yyj 分支与显式 KT 构建配置；普通 LF 构建仍保留上游依赖。
正式版本锁、安装命令和 wheel SHA256 必须在依赖正常解析并完成全链路验收后补齐。
不使用 `--no-deps`、共享 site-packages、事后改写 LF wheel 或模型目录补丁。
LF、PEFT、TRL 的本地版本 wheel 将作为显式锁定的训练工具提供；并非宣称已发布到 PyPI。

本文中的两个 `kt-*` 命令是候选 kt-kernel 新增的安装入口，不要求下载开发源码。
下面的流程供候选验收使用；当前 PyPI 包不保证包含这些命令。

## 范围

- Kimi K2.5 原始 RAWINT4 group32 routed experts，冻结 base weights；不转全量 BF16。
- LoRA rank 8、alpha 16、dropout 0；五个 attention 投影加 KT fused expert LoRA。
- 训练使用 8 GPU、FSDP2 CPU parameter offload；CPU/GPU activation 均重算。
- `flash_attn: disabled`，`packing: false`，`neat_packing: false`。
- 参考机器为 sap4：8 张 48 GiB RTX 4090、约 2 TiB RAM。这不是最低硬件需求声明。
- 临时产物可写 RAM，但必须给训练内存留余量；重启前将选定 checkpoint 复制到持久化磁盘并校验 SHA256。

## 1. 固定模型和公共数据

使用原始 [Kimi K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)；不修改下载的配置、tokenizer 或模型代码。
本轮验收固定模型 revision `54383e83fa343a1331754112fb9e3410c55efa2f`。

下载 [NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K) 的固定版本：

```bash
hf download liumindmind/NekoQA-10K NekoQA-10K.json \
  --repo-type dataset --revision 1b2110c996a8237823b86c1a3d3e8a6762b38430 \
  --local-dir /path/to/neko-source

kt-prepare-kimi-data --model /path/to/Kimi-K2.5 \
  --nekoqa /path/to/neko-source/NekoQA-10K.json \
  --trust-remote-code --output-dir /path/to/prepared-neko \
  > /path/to/preparation.log 2>&1
```

工具检查原始 JSON SHA256：
`b4d260ad117c29c9fd64abcb513ad24d62e2fac383640e17ea230d12ae03b849`。
固定 seed 42，先按规范化问题去重，再划分：训练 9,477、验证 468、独立生成测试 32 条。
4 条空记录和 85 条重复问题的原始行号记入 manifest，不静默丢弃。

`heldout.json` 不加入训练或验证集，问题不包含显式猫娘/角色扮演提示。
训练 JSON 已使用原生 Kimi non-thinking 模板，LF 必须配 `template: empty`，不可再次套聊天模板。
每条样本检查分开编码的 prompt + answer 与原生完整对话 token 一致，prompt 不参与 loss。

保留 `preprocessing_manifest.json`，其中记录数据、原生模板、输出及预期 token/label 的 SHA256。
本轮原始 tokenizer CPU 预处理测得训练最长 2,120 token；S4096 是上限，不是每条样本都满长。

自有数据可用 `--input train.json --eval-input eval.json`，格式为
`[{"prompt": "问题", "answer": "回答"}]`，可选 `system`；仅支持纯文本单轮，不接受已格式化对话、历史、工具或图片。

## 2. 先完成保存和真续训 smoke

修改 [train.yaml](train.yaml) 中的模型、权重、数据、输出路径。
配套使用 [fsdp2_8gpu.yaml](fsdp2_8gpu.yaml)，不要额外再套一层 torchrun。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file fsdp2_8gpu.yaml --no_python \
  "$(command -v llamafactory-cli)" train train.yaml
```

默认 S512、B1/GAS1、总计 4 步，保存 checkpoint-2 和 checkpoint-4。
**`save_only_model: false` 必须保留**；两份 checkpoint 都要有普通与 fused expert LoRA，以及完整的 optimizer、scheduler、Trainer 和每 rank RNG 状态。

退出后另起进程，从 checkpoint-2 恢复，写入新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file fsdp2_8gpu.yaml --no_python \
  "$(command -v llamafactory-cli)" train train.yaml \
  resume_from_checkpoint=/path/to/run-a/checkpoint-2 \
  output_dir=/path/to/run-b
```

前后 `max_steps` 都是 4，world size、数据顺序和 scheduler 不变。
这才是 step 2 → 4 的续训；只设置 `adapter_name_or_path` 不等于恢复 optimizer。
缺失任何 rank 状态或 world-size 不匹配应明确失败。
验收需比较 run-a / run-b 的 checkpoint-4 参数、更新及状态，不能只检查“进程退出成功”。

## 3. 再做一轮风格训练

smoke 通过后，从 base 重新开始，使用新输出目录；最多一轮完整数据。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file fsdp2_8gpu.yaml --no_python \
  "$(command -v llamafactory-cli)" train train.yaml \
  output_dir=/path/to/neko-run max_steps=-1 num_train_epochs=1 \
  cutoff_len=4096 kt_config.kt_model_max_length=4096 \
  warmup_ratio=0.03 save_steps=100 \
  do_eval=true eval_dataset=neko_eval eval_strategy=steps eval_steps=100
```

只用训练集和验证集做训练决策；不要根据最终 32 条生成测试反复调参。
观察有限 loss / grad norm、普通与 expert LoRA B 参数的非零变化、验证 loss。
不以短 smoke 的 loss 下降或“能输出喵”代替最终质量验收。

## 4. 转换并在新进程加载

使用独立的推理环境，安装同一锁定发布组合。保留原始训练 checkpoint：

```bash
kt-convert-lora /path/to/neko-run /path/to/neko-sglang \
  --base-model-name-or-path /path/to/Kimi-K2.5
```

转换合并普通 LoRA 与 `fused_expert_lora.safetensors`，保留 rank、alpha 和张量值。
不能仅把普通 `adapter_model.safetensors` 传给 SGLang，否则专家部分没有加载。
输出目录不要复用，也不要指向模型或原始训练目录。

候选推理命令（4 GPU；待最终 wheel 重新验收）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
  --model-path /path/to/Kimi-K2.5 --trust-remote-code \
  --served-model-name kimi-k25 --host 127.0.0.1 --port 30000 \
  --tensor-parallel-size 4 --dtype bfloat16 --context-length 2048 \
  --max-total-tokens 4096 --chunked-prefill-size 256 --max-running-requests 4 \
  --mem-fraction-static 0.75 --disable-cuda-graph --disable-radix-cache \
  --attention-backend triton --grammar-backend llguidance \
  --kt-method RAWINT4 --kt-weight-path /path/to/Kimi-K2.5 \
  --kt-cpuinfer 64 --kt-threadpool-count 2 --kt-num-gpu-experts 0 \
  --random-seed 42 --watchdog-timeout 900 \
  --enable-lora --lora-backend triton --lora-paths neko=/path/to/neko-sglang
```

请求必须选择 adapter 名称：

```bash
curl --fail http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k25:neko","messages":[{"role":"user","content":"我今天学习有点累，请用两句话鼓励我。"}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"thinking":false,"enable_thinking":false}}'
```

最终检查：全部普通/专家张量加载、真实 decode、算术和 JSON 约束生成。
用 `heldout.json` 的原始问题对比 base 和 adapter，不加“请扮演猫娘”之类提示。
**base 对照必须是另一个不加载任何 LoRA 的新服务进程**，不能把已挂载静态 expert LoRA 的同一进程当 base。

质量验收：32 条中至少 24 条呈现目标风格，比 base 至少多 8 条；至少 28 条回答相关且完整；固定算术/指令题全部通过。
保存原始请求、回答、打分、版本与 wheel 哈希。未满足这些条件，不宣布“已训出风格”。
