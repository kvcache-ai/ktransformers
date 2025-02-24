# FP8 Linear Kernel.
For DeepSeek-R1/V3, the DeepSeek-AI team provides fp8 safetensors. We have integrated the FP8 GPU kernel into the KTransformers. But to keep the experts still in CPU to save GPU memory, we still use ggml(GGUF tensors) quantization for experts. In this way, we can increase the precision in calculating attention, which may improve the model's performance.

Therefore, to use fp8 linear kernel, we need to merge fp8 weights and gguf files. We have provides prepared weights in huggingface so that you can use them directly. 

[KVCache-ai/DeepSeek-V3](https://huggingface.co/KVCache-ai/DeepSeek-V3/upload/main)


If you want to use other formats of ggml quantization, you can use the following script to merge them.

## Example
To use fp8 linear kernal and q4km experts.
```shell
bash
python convert_model.py \
  --safetensor_path <fp8 safetensor path> \
  --gguf_path <q4km gguf folder path> \
  --output_path <output path>
```
* `--safetensor_path`:	input path of safetensor file
* `--gguf_path`: input path of gguf folder
* `--output_path`: output path of merged file


## To Run DeepSeek-V3 with fp8 linear kernel and q4km experts


```shell
python ktransformers/local_chat.py --model_path deepseek-ai/DeepSeek-V3 --gguf_path <new weights folder> --optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-fp8-linear-ggml-experts.yaml --cpu_infer <cpu cores + 1>
```


> NOTES: 
> 1. Using fp8 linear kernel and q4km experts will consume approximatly 19GB GPU memory. 
> 2. I know the the new way to load module is ugly, we are working on it.
> 3. Though the model is a mixture of fp8 and ggml, they are stored in .safetensor files. Please pass the folder path of the new weights to `--gguf_path`.
