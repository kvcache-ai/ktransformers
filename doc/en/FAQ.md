# Frequently Asked Questions

## 1. SGLang "Using default MoE kernel config" warning at startup

When using kt-kernel with SGLang, you may see a warning like:

```
[2026-05-15 20:31:38] Using default MoE kernel config. Performance might be sub-optimal!
Config file not found at .../fused_moe_triton/configs/...
```

This warning is **expected and can be safely ignored**. kt-kernel replaces SGLang's built-in MoE implementation with its own CPU/GPU hybrid dispatch, so SGLang's fused-MoE Triton kernel configuration is never used. The warning is emitted by SGLang before kt-kernel takes over MoE execution and has no impact on performance or correctness.

## 2. Where can I find more help?

Check the [existing issues](https://github.com/kvcache-ai/ktransformers/issues) or open a [new one](https://github.com/kvcache-ai/ktransformers/issues/new).

