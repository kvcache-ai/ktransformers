# FP8 Linear Kernel for DeepSeek-V3/R1

## Overview
The DeepSeek-AI team provides FP8 safetensors for DeepSeek-R1/V3 models. We achieve performance optimization through the following works:
- **FP8 GPU Kernel Integration**: FP8 linear layer acceleration kernels integrated in KTransformers
- **Hybrid Quantization Architecture**:
  - Attention and Shared-Expert modules use FP8 precision (enhances computational accuracy)
  - Experts modules retain GGML quantization (GGUF format, reside in CPU to save GPU memory)

So those who are persuing the best performance can use the FP8 linear kernel for DeepSeek-V3/R1.

## Key Features

✅ Hybrid Precision Architecture (FP8 + GGML)<br>
✅ Memory Optimization (~19GB VRAM usage)

## Quick Start
### Using Pre-Merged Weights

Pre-merged weights are available on Hugging Face:<br>
[KVCache-ai/DeepSeek-V3-GGML-FP8-Hybrid](https://huggingface.co/KVCache-ai/DeepSeek-V3)<br>
[KVCache-ai/DeepSeek-R1-GGML-FP8-Hybrid](https://huggingface.co/KVCache-ai/DeepSeek-R1)

> Please confirm the weights are fully uploaded before downloading. The large file size may extend Hugging Face upload time.


Download Pre-Merged Weights
```shell
pip install -U huggingface_hub

# Optional: Use HF Mirror for faster downloads in special area.
# export HF_ENDPOINT=https://hf-mirror.com 

huggingface-cli download --resume-download KVCache-ai/DeepSeek-V3-GGML-FP8-Hybrid --local-dir <local_dir>
```
### Using merge scripts
If you got local DeepSeek-R1/V3 fp8 safetensors and gguf weights(eg.q4km), you can merge them using the following scripts.

```shell
python merge_tensors/merge_safetensor_gguf.py \
  --safetensor_path <fp8_safetensor_path> \
  --gguf_path <gguf_folder_path> \
  --output_path <merged_output_path>
```

* `--safetensor_path`:	input path of safetensor file([Download](https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main)).
* `--gguf_path`: input path of gguf folder ([Download](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)).
* `--output_path`: output path of merged file.


### Execution Notes

Launch local_chat.py with custom quantized experts
```shell
python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-V3 \
  --gguf_path <merged_weights_folder> \
  --optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-fp8-linear-ggml-experts.yaml \
  --cpu_infer <cpu_cores + 1>
```


## Notes

⚠️ Hardware Requirements<br>
* Recommended minimum 19GB available VRAM for FP8 kernel.
* Requires GPU with FP8 support (e.g., 4090)

⏳ First-Run Optimization
JIT compilation causes longer initial execution (subsequent runs retain optimized speed).

🔄 Temporary Interface<br>
Current weight loading implementation is provisional - will be refined in future versions

📁 Path Specification<br>
Despite hybrid quantization, merged weights are stored as .safetensors - pass the containing folder path to `--gguf_path`