# Frequently Asked Questions

## 1. SGLang "Using default MoE kernel config" warning at startup

When using kt-kernel with SGLang, you may see a warning like:

```
[2026-05-15 20:31:38] Using default MoE kernel config. Performance might be sub-optimal!
Config file not found at .../fused_moe_triton/configs/...
```

This warning is **expected and can be safely ignored**. kt-kernel replaces SGLang's built-in MoE implementation with its own CPU/GPU hybrid dispatch, so SGLang's fused-MoE Triton kernel configuration is never used. The warning is emitted by SGLang before kt-kernel takes over MoE execution and has no impact on performance or correctness.

## 2. Do I need both GPU and CPU weights for kt-kernel + SGLang?

It depends on the backend mode:

- **Original-precision / native modes** such as `BF16`, `FP8`, or `RAWINT4` can reuse the
  original model directory as the KT weight path when the model format is supported.
- **Low-precision CPU expert modes** such as `AMXINT4` / `AMXINT8` need CPU-side expert
  weights converted by `kt-kernel/scripts/convert_cpu_weights.py`. In this case, you normally
  keep the original or quantized GPU model weights for SGLang and a separate converted CPU
  weight directory for KT-Kernel experts.
- **LLAMAFILE mode** uses GGUF weights directly as the CPU-side KT weight path.

See the [KT-Kernel weight preparation guide](../../kt-kernel/README.md#2-prepare-weights) for the
current command examples and backend-specific details.

## 3. Where can I find more help?

Check the [existing issues](https://github.com/kvcache-ai/ktransformers/issues) or open a [new one](https://github.com/kvcache-ai/ktransformers/issues/new).

