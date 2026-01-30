# KTransformers + Clawdbot：本地部署 AI 助手方案

> **利用 KTransformers 的 CPU-GPU 混合推理能力，结合 Kimi-K2.5 的高质量推理能力，为 Clawdbot 提供高性能本地推理后端**

---

## 什么是 Clawdbot？

[Clawdbot](https://github.com/openclaw/openclaw) 是一款开源的个人 AI 智能体，支持通过 Telegram、Discord、Signal、WhatsApp 等聊天平台交互，可实现日程管理、邮件发送、数据查询等自动化任务，数据完全本地存储，隐私可控。

> **注意**：Clawdbot 默认不内置飞书（Feishu）Channel，需要额外安装社区插件，详见下方飞书接入章节。

---

## 为什么选择 KTransformers 作为推理后端？

**KTransformers** 使用 CPU-GPU 混合推理架构：

- **CPU-GPU 协同**：GPU 处理高价值推理路径，CPU（AMX 量化）处理专家模块，资源利用率最大化
- **原生 MoE 支持**：支持多种原生精度的 MoE 模型
- **SGLang 高性能引擎**：兼容 OpenAI API，支持多 GPU Tensor Parallel 并行
- **全栈 CLI 工具**：`kt run` 一键启动、`kt model` 模型管理、`kt quant` 智能量化、`kt bench` 性能测试、`kt doctor` 环境诊断

---

## 支持的模型

自 Kimi K2 Thinking 等[原精度模型支持](../en/kt-kernel/Native-Precision-Tutorial.md)以来，我们 Day0 适配了 [Kimi K2.5](../en/Kimi-K2.5.md)。目前，我们已经原精度支持 Kimi K2.5、MiniMax、DeepSeek、Qwen3、GLM 等 MoE 模型，仅使用 24-48G 显存即可完美部署。

---

## 部署架构

```
[用户] → [Telegram / Discord / Signal / 飞书] → [Clawdbot Gateway]
                                                        ↓
                                                  [KTransformers]
                                                   (SGLang API)
                                                        ↓
                                                  [多 GPU 推理]
```

Clawdbot 通过 OpenAI 兼容 API 接入 KTransformers，无需额外 API 密钥，本地推理零费用。

---

## 部署步骤

### 第一步：安装并启动 KTransformers

[Kimi K2.5 使用指南](../en/Kimi-K2.5.md)

[kt kernel 部署指南](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel)。

启动后，KTransformers 会在 `http://<host>:30000/v1` 提供 OpenAI 兼容 API。

### 第二步：安装 Clawdbot

```bash
npm install -g openclaw@latest

openclaw onboard --install-daemon
```

> 关于 Clawdbot 的详细安装与配置，请参考 [Clawdbot 官方文档](https://openclaw.ai) 和 [GitHub 仓库](https://github.com/openclaw/openclaw)。

### 第三步：配置 KTransformers 作为推理后端

编辑 Clawdbot 配置文件（通常位于 `~/.openclaw/openclaw.json`，或通过网页版 `http://127.0.0.1:18789/config`），将模型 provider 指向本地 KTransformers 服务：

```json
{
  "models": {
    "providers": {
      "synthetic": {
        "baseUrl": "http://127.0.0.1:30000/v1",
        "apiKey": "EMPTY",
        "api": "openai-completions",
        "models": [
          {
            "id": "kimi-k2.5",
            "name": "kimi-k2.5",
            "contextWindow": 200000,
            "maxTokens": 16384
          }
        ]
      }
    },
    "routing": {
      "default": {
        "provider": "synthetic",
        "modelId": "kimi-k2.5"
      }
    }
  }
}
```

关键配置说明：
- `baseUrl`：KTransformers SGLang 服务地址
- `apiKey`：填写 `"EMPTY"` 即可，本地服务不需要密钥
- `models`：根据实际运行的模型调整 `id` 和 `contextWindow`

### 第四步：启动 Clawdbot Gateway

```bash
openclaw gateway --port 18789
```

### 第五步：配置消息通道

Clawdbot 原生支持 Telegram、Discord、Signal 等通道：

```bash
# Telegram
openclaw channels login --channel telegram

# Signal
openclaw channels login --channel signal
```

---

## 飞书接入

Clawdbot 默认不包含飞书通道，需要通过社区开发的飞书桥接插件接入。

主要步骤：
1. 在[飞书开放平台](https://open.feishu.cn/)创建企业自建应用，添加"机器人"能力
2. 安装飞书桥接插件（社区项目：[clawdbot-feishu](https://github.com/m1heng/clawdbot-feishu)）
3. 配置 `appId`、`appSecret` 等飞书应用凭据
4. 添加"接收消息"事件，发布应用版本

详细教程可参考：
- [Clawdbot 接入飞书保姆级教程](https://mp.weixin.qq.com/s/_i1fgNbeDrBR5wurEmJf0A)
- [腾讯云：Moltbot 接入飞书保姆级教程](https://cloud.tencent.com/developer/article/2625073)

---

## 硬件参考配置

以下是一个 8 卡 GPU 部署的参考配置：

| 组件 | 配置 |
|------|------|
| GPU | 8 × NVIDIA RTX 5090（32GB 显存） |
| CPU | 双路高核心数处理器（至少需支持 AVX 512 指令集） |
| 内存 | 512GB+ |
| 模型 | Kimi K2.5 / DeepSeek-V3 / GLM-4.7 等 |

```bash
# 启动示例
kt run kimi-k2.5
```

---

## KTransformers 与传统部署对比

| 特性 | KTransformers | 传统部署 |
|------|---------------|----------|
| 显存需求 | 小 | 原始大小 |
| MoE 支持 | CPU-GPU 动态调度 | 无 |
| CPU-GPU 混合 | NUMA 优化 | 无 |
| 管理工具 | kt CLI 全栈工具 | 手动 |
| 故障诊断 | `kt doctor` 自动检测 | 手动调试 |

---

## 适用场景

- **企业部署**：客户服务自动化、文档智能问答、工作流自动化
- **研发团队**：模型快速验证、性能基准测试、实验环境搭建
- **个人用户**：低成本本地 AI 助手、隐私数据可控

---

## 相关链接

- [KTransformers GitHub](https://github.com/KTransformers/ktransformers)
- [Clawdbot 官网](https://openclaw.ai/)
- [Clawdbot GitHub](https://github.com/clawdbot/clawdbot)
- [飞书桥接插件](https://github.com/m1heng/clawdbot-feishu)
