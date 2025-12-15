# 基准测试结果(输出token长度均设置1k, 单并发)

| Prompt length                     | 1K     | 2K     | 4K     |
| --------------------------------- | ------ | ------ | ------ |
| KTrans Prefill token/s | 134.11 | 141.60 |  143.42 |
| KTrans Decode token/s | 11.05 | 10.74 | 10.68 |

## 先决条件
我们在以下配置下进行了Qwen3-235B-A22B MoE最佳性能测试：
- 服务器型号：Atlas 2UP
- NPU：Atlas 300I A2
- CPU: HUAWEI Kunpeng 920 7270Z
- 内存: DDR5服务器内存（1TB）

# 部署

***关于部署过程，此README中只额外描述与同级目录下 `DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md` 不同的部分***

## 物理机安装

部署满血版Qwen3-MoE，需要机器物理内存能够存放下全部路由专家的权重，约200GB。

目前支持的NPU型号：**300I A2**。

在技术人员的支持下完成硬件安装。


## 权重准备

目前，为了满足性能和精度的要求，我们需要准备两份权重，并使用提供的权重合并脚本对权重进行合并，最终只会使用合并后的权重。

Q4权重：[Qwen3-235B-A22B-Instruct-2507-GGUF](https://modelscope.cn/models/unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/files)

W8A8权重：[Qwen3-235B-A22B-w8a8](https://modelers.cn/models/Modelers_Park/Qwen3-235B-A22B-w8a8)

使用[merge_safetensor_gguf_for_qwen3.py](../../merge_tensors/merge_safetensor_gguf_for_qwen3.py)来合并Q4和W8A8权重：

```bash
python merge_safetensor_gguf_for_qwen3.py --safetensor_path /mnt/weights/Qwen3-235B-A22B-Q4_K_M --gguf_path /mnt/weights/Qwen3-235B-A22B-W8A8 --output_path /mnt/weights/Qwen3-235B-A22B-q4km-w8a8
```

## kTransformers部署

将项目文件部署到机器上：

- 初始化third_party。由于此过程耗时较多，且容易受网络影响导致仓库克隆失败，建议初始化一次后，将相关文件进行打包，以便后续直接解压使用。
  ```bash
  git clone https://github.com/kvcache-ai/ktransformers.git
  cd ktransformers
  git submodule update --init --recursive
  ```
- 对于arm平台，注释掉`./third_party/llamafile/iqk_mul_mat_arm82.cpp`中的
  ```cpp
  #define iqk_mul_mat iqk_mul_mat_arm82
  #define iqk_mul_mat_moe iqk_mul_mat_moe_arm82
  ```
- 执行`source /usr/local/Ascend/ascend-toolkit/set_env.sh`（以实际CANN-TOOLKIT安装路径为准）。
- 执行`apt install cmake libhwloc-dev pkg-config`安装依赖。
- 修改项目目录下 /ktransformers/config/config.yaml 中attn部分的page_size: 128  chunk_size: 16384
- 执行`USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh`，等待安装完成。
    ***执行安装命令之前，需要将`./ktransformers/configs/config.yaml`中对于page size的设置改为page size=128(因为attn计算算子`torch_npu.npu_fused_infer_attention_score`支持page_size=16/128)***

此处给出示例balance_serve的启动脚本（由于使用了相对路径，需将该脚本放至项目的根路径下）：

```bash
#!/bin/bash
export USE_MERGE=0
export INF_NAN_MODE_FORCE_DISABLE=1
export TASK_QUEUE_ENABLE=0
export RANK=0
export LOCAL_WORLD_SIZE=1
#export PROF_DECODE=1
#export PROF_PREFILL=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

python ktransformers/server/main.py \
--port 10002 \
--model_path <your model path> \
--gguf_path <your model path> \
--cpu_infer 48 \
--optimize_config_path  ./ktransformers/optimize/optimize_rules/npu/Qwen3-Chat-300IA2-npu-serve.yaml \
--max_new_tokens 1024 \
--cache_lens 16384 \
--max_batch_size 4 \
--use_cuda_graph \
--tp 1 \
--backend_type balance_serve
```

相关参数说明：

- `--model_path`：kTransformers原生参数，str，此处用来指定合并后的模型文件路径
- `--gguf_path`：kTransformers原生参数，str，此处用来指定合并后的模型文件路径
- `--cpu_infer`：kTransformers原生参数，int，用来控制CPU侧实际worker线程数，非必选
- `--optimize_config_path`：kTransformers原生参数，str，用来指定所用的模型优化配置文件，需要注意相对路径的使用，此处为**必选**
- `--cache_lens`：调度器申请 kvcache 的总长度。所有请求共享指定数量（例如 `20480`）的 tokens 对应的 kvcache 空间，请求完成后会释放其所占用的 kvcache 空间，非必选
- `--use_cuda_graph`：kTransformers原生参数，bool，为True表示开启图下沉，为False表示关闭图下沉，非必选
- `--max_new_tokens`：kTransformers原生参数，int，当统计到输出的tokens数量达到该值时，会直接中止输出，非必选
- `--tp`：新增参数，int，用于开启tensor model parallel功能，目前local_chat只支持tp大小与ws大小相同（不支持local_chat使用多dp），非必选


# 其他问题

## 可能存在的其他依赖问题

ImportError: libhccl.so: cannot open shared object file: No such file or directory

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # 以实际CANN安装路径为准
```

ImportError: libascend_hal.so: cannot open shared object file: No such file or directory

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH  # 以实际Driver安装路径为准
```
