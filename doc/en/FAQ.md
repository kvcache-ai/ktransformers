# FAQ
## Install
### Q: ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.32' not found
```
in Ubuntu 22.04 installation need to add the:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install --only-upgrade libstdc++6
```
from-https://github.com/kvcache-ai/ktransformers/issues/117#issuecomment-2647542979
### Q: DeepSeek-R1 not outputting initial <think> token

> from deepseek-R1 doc:<br>
> Additionally, we have observed that the DeepSeek-R1 series models tend to bypass thinking pattern (i.e., outputting "\<think>\n\n\</think>") when responding to certain queries, which can adversely affect the model's performance. To ensure that the model engages in thorough reasoning, we recommend enforcing the model to initiate its response with "\<think>\n" at the beginning of every output.

So we fix this by manually adding "\<think>\n" token at prompt end (you can check out at local_chat.py),
and pass the arg `--force_think true ` can let the local_chat initiate the response with "\<think>\n"

from-https://github.com/kvcache-ai/ktransformers/issues/129#issue-2842799552

## Usage
### Q: If I got more VRAM than the model's requirement, how can I fully utilize it?

1. Get larger context.
   1. local_chat.py: You can increase the context window size by setting `--max_new_tokens` to a larger value.
   2. server: Increase the `--cache_lens' to a larger value.
2. Move more weights to the GPU.
    Refer to the ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu-4.yaml
    ```yaml
    - match:
       name: "^model\\.layers\\.([4-10])\\.mlp\\.experts$" # inject experts in layer 4~10 as marlin expert
     replace:
       class: ktransformers.operators.experts.KTransformersExperts  
       kwargs:
         generate_device: "cuda:0" # run in cuda:0; marlin only support GPU
         generate_op:  "KExpertsMarlin" # use marlin expert
     recursive: False
    ```
    You can modify layer as you want, eg. `name: "^model\\.layers\\.([4-10])\\.mlp\\.experts$"` to `name: "^model\\.layers\\.([4-12])\\.mlp\\.experts$"` to move more weights to the GPU.

    > Note: The first matched rule in yaml will be applied. For example, if you have two rules that match the same layer, only the first rule's replacement will be valid.
    > Note：Currently, executing experts on the GPU will conflict with CUDA Graph. Without CUDA Graph, there will be a significant slowdown. Therefore, unless you have a substantial amount of VRAM (placing a single layer of experts for DeepSeek-V3/R1 on the GPU requires at least 5.6GB of VRAM), we do not recommend enabling this feature. We are actively working on optimization.
    > Note KExpertsTorch is untested.


### Q: If I don't have enough VRAM, but I have multiple GPUs, how can I utilize them?

Use the `--optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml` to load the two optimized rule yaml file. You may also use it as an example to write your own 4/8 gpu optimized rule yaml file.

> Note: The ktransformers' multi-gpu stratigy is pipline, which is not able to speed up the model's inference. It's only for the model's weight distribution.

### Q: How to get the best performance?

You have to set `--cpu_infer` to the number of cores you want to use. The more cores you use, the faster the model will run. But it's not the more the better. Adjust it slightly lower to your actual number of cores.

### Q: My DeepSeek-R1 model is not thinking.

According to DeepSeek, you need to enforce the model to initiate its response with "\<think>\n" at the beginning of every output by passing the arg `--force_think true `.

### Q: Loading gguf error

Make sure you:
1. Have the `gguf` file in the `--gguf_path` directory.
2. The directory only contains gguf files from one model. If you have multiple models, you need to separate them into different directories.
3. The folder name it self should not end with `.gguf`, eg. `Deep-gguf` is correct, `Deep.gguf` is wrong.

### Q: Version `GLIBCXX_3.4.30' not found
The detailed error:
>ImportError: /mnt/data/miniconda3/envs/xxx/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xxx/xxx/ktransformers/./cpuinfer_ext.cpython-312-x86_64-linux-gnu.so)

Running `conda install -c conda-forge libstdcxx-ng` can solve the problem.
