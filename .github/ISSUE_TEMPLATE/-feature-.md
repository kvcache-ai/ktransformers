name: 🚀 Feature request
description: Suggest an idea for this project
title: "[Feature] "

body:
- type: checkboxes
  attributes:
    label: Checklist
    options:
    - label: 1. If the issue you raised is not a feature but a question, please raise a discussion at https://github.com/kvcache-ai/ktransformers/discussions. Otherwise, it will be closed.
    - label: 2. To help the community, I will use Chinese/English or attach an Chinese/English translation if using another language. Non-English/Chinese content without translation may be closed.
- type: textarea
  attributes:
    label: Motivation
    description: |
      A clear and concise description of the motivation of the feature.
  validations:
    required: true
- type: textarea
  attributes:
    label: Related resources
    description: |
      If there is an official code release or third-party implementations, please also provide the information here, which would be very helpful.