<!-- omit in toc -->
# FAQ
- [Install](#install)
  - [Q: ImportError: /lib/x86\_64-linux-gnu/libstdc++.so.6: version GLIBCXX\_3.4.32' not found](#q-importerror-libx86_64-linux-gnulibstdcso6-version-glibcxx_3432-not-found)
  - [Q: DeepSeek-R1 not outputting initial  token](#q-deepseek-r1-not-outputting-initial--token)
- [Usage](#usage)
  - [Q: If I got more VRAM than the model's requirement, how can I fully utilize it?](#q-if-i-got-more-vram-than-the-models-requirement-how-can-i-fully-utilize-it)
  - [Q: If I don't have enough VRAM, but I have multiple GPUs, how can I utilize them?](#q-if-i-dont-have-enough-vram-but-i-have-multiple-gpus-how-can-i-utilize-them)
  - [Q: How to get the best performance?](#q-how-to-get-the-best-performance)
  - [Q: My DeepSeek-R1 model is not thinking.](#q-my-deepseek-r1-model-is-not-thinking)
  - [Q: Loading gguf error](#q-loading-gguf-error)
  - [Q: Version \`GLIBCXX\_3.4.30' not found](#q-version-glibcxx_3430-not-found)
  - [Q: When running the bfloat16 moe model, the data shows NaN](#q-when-running-the-bfloat16-moe-model-the-data-shows-nan)
  - [Q: Using fp8 prefill very slow.](#q-using-fp8-prefill-very-slow)
  - [Q: Possible ways to run graphics cards using volta and turing architectures](#q-possible-ways-to-run-graphics-cards-using-volta-and-turing-architectures)
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

Use the `--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml` to load the two optimized rule yaml file. You may also use it as an example to write your own 4/8 gpu optimized rule yaml file.

> Note: The ktransformers' multi-gpu stratigy is pipline, which is not able to speed up the model's inference. It's only for the model's weight distribution.

### Q: How to get the best performance?

You have to set `--cpu_infer` to the number of cores you want to use. The more cores you use, the faster the model will run. But it's not the more the better. Adjust it slightly lower to your actual number of cores.

### Q: My DeepSeek-R1 model is not thinking.

According to DeepSeek, you need to enforce the model to initiate its response with "\<think>\n" at the beginning of every output by passing the arg `--force_think True `.

### Q: Loading gguf error

Make sure you:
1. Have the `gguf` file in the `--gguf_path` directory.
2. The directory only contains gguf files from one model. If you have multiple models, you need to separate them into different directories.
3. The folder name it self should not end with `.gguf`, eg. `Deep-gguf` is correct, `Deep.gguf` is wrong.
4. The file itself is not corrupted; you can verify this by checking that the sha256sum matches the one from huggingface, modelscope, or hf-mirror.

### Q: Version `GLIBCXX_3.4.30' not found
The detailed error:
>ImportError: /mnt/data/miniconda3/envs/xxx/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xxx/xxx/ktransformers/./cpuinfer_ext.cpython-312-x86_64-linux-gnu.so)

Running `conda install -c conda-forge libstdcxx-ng` can solve the problem.


### Q: When running the bfloat16 moe model, the data shows NaN
The detailed error:
```shell
Traceback (most recent call last):
  File "/root/ktransformers/ktransformers/local_chat.py", line 183, in <module>
    fire.Fire(local_chat)
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/usr/local/lib/python3.10/dist-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/root/ktransformers/ktransformers/local_chat.py", line 177, in local_chat
    generated = prefill_and_generate(
  File "/root/ktransformers/ktransformers/util/utils.py", line 204, in prefill_and_generate
    next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, use_cuda_graph).to(torch_device)
  File "/root/ktransformers/ktransformers/util/utils.py", line 128, in decode_one_tokens
    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```
**SOLUTION**: The issue of running ktransformers on Ubuntu 22.04 is caused by the current system's g++ version being too old, and the pre-defined macros do not include avx_bf16. We have tested and confirmed that it works on g++ 11.4 in Ubuntu 22.04.

### Q: Using fp8 prefill very slow.

The FP8 kernel is build by JIT, so the first run will be slow. The subsequent runs will be faster.

### Q: Possible ways to run graphics cards using volta and turing architectures

From: https://github.com/kvcache-ai/ktransformers/issues/374

1. First, download the latest source code using git.
2. Then, modify the DeepSeek-V3-Chat-multi-gpu-4.yaml in the source code and all related yaml files, replacing all instances of KLinearMarlin with KLinearTorch.
3. Next, you need to compile from the ktransformers source code until it successfully compiles on your local machine.
4. Then, install flash-attn. It won't be used, but not installing it will cause an error.
5. Then, modify local_chat.py, replacing all instances of flash_attention_2 with eager.
6. Then, run local_chat.py. Be sure to follow the official tutorial's commands and adjust according to your local machine's parameters.
7. During the running process, check the memory usage. Observe its invocation through the top command. The memory capacity on a single CPU must be greater than the complete size of the model. (For multiple CPUs, it's just a copy.)
Finally, confirm that the model is fully loaded into memory and specific weight layers are fully loaded into the GPU memory. Then, try to input content in the chat interface and observe if there are any errors.

Attention, for better perfomance, you can check this [method](https://github.com/kvcache-ai/ktransformers/issues/374#issuecomment-2667520838) in the issue
>
>https://github.com/kvcache-ai/ktransformers/blob/89f8218a2ab7ff82fa54dbfe30df741c574317fc/ktransformers/operators/attention.py#L274-L279
>
>```diff
>+ original_dtype = query_states.dtype
>+ target_dtype = torch.half
>+ query_states = query_states.to(target_dtype)
>+ compressed_kv_with_k_pe = compressed_kv_with_k_pe.to(target_dtype)
>+ compressed_kv = compressed_kv.to(target_dtype)
>+ attn_output = attn_output.to(target_dtype)
>
>decode_attention_fwd_grouped(query_states, compressed_kv_with_k_pe, compressed_kv, attn_output,
>                             page_table,
>                             position_ids.squeeze(0).to(torch.int32)+1, attn_logits,
>                             4, #num_kv_splits # follow vLLM, fix it TODO
>                             self.softmax_scale,
>                             past_key_value.page_size)
>
>+ attn_output = attn_output.to(original_dtype)
>```
>
>https://github.com/kvcache-ai/ktransformers/blob/89f8218a2ab7ff82fa54dbfe30df741c574317fc/ktransformers/operators/attention.py#L320-L326
>
>```diff
>- attn_output = flash_attn_func( 
>-     query_states, 
>-     key_states, 
>-     value_states_padded, 
>-     softmax_scale=self.softmax_scale, 
>-     causal=True, 
>- )
>+ attn_output = F.scaled_dot_product_attention(
>+     query_states.transpose(1, 2),
>+     key_states.transpose(1, 2),
>+     value_states_padded.transpose(1, 2),
>+     scale=self.softmax_scale,
>+     is_causal=True
>+ ).transpose(1, 2)
>```