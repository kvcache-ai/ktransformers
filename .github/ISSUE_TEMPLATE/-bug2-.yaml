name: 🐞 BUG报告
description: 创建报告以帮助我们复现并修复BUG
title: "[Bug] "
labels: ['Bug']

body:
- type: checkboxes
  attributes:
    label: 检查清单
    options:
    - label: 1. 我已经搜索过相关问题，但未能获得预期的帮助
    - label: 2. 该问题在最新版本中尚未修复
    - label: 3. 请注意，如果您提交的BUG相关 issue 缺少对应环境信息和最小可复现示例，我们将难以复现和定位问题，降低获得反馈的可能性
    - label: 4. 如果您提出的不是bug而是问题，请在讨论区发起讨论 https://github.com/kvcache-ai/ktransformers/discussions。否则该 issue 将被关闭
    - label: 5. 为方便社区交流，我将使用中文/英文或附上中文/英文翻译（如使用其他语言）。未附带翻译的非中文/英语内容可能会被关闭

- type: textarea
  attributes:
    label: 问题描述
    description: 清晰简洁地描述BUG是什么
  validations:
    required: true
- type: textarea
  attributes:
    label: 复现步骤
    description: |
      你运行了什么命令或脚本？使用的是哪个**模型**？
    placeholder: |
      在此处填写命令
  validations:
    required: true
- type: textarea
  attributes:
    label: 环境信息
    description: |
      请提供必要的环境信息（如操作系统/GPU/CPU），否则该 issue 将被关闭
    placeholder: 在此处填写环境信息
  validations:
    required: true