# FAQ
## Install
### 1 ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.32' not found
```
in Ubuntu 22.04 installation need to add the:
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install --only-upgrade libstdc++6
```
from-https://github.com/kvcache-ai/ktransformers/issues/117#issuecomment-2647542979
### 2 DeepSeek-R1 not outputting initial <think> token

> from deepseek-R1 doc:<br>
> Additionally, we have observed that the DeepSeek-R1 series models tend to bypass thinking pattern (i.e., outputting "\<think>\n\n\</think>") when responding to certain queries, which can adversely affect the model's performance. To ensure that the model engages in thorough reasoning, we recommend enforcing the model to initiate its response with "\<think>\n" at the beginning of every output.

So we fix this by manually adding "\<think>\n" token at prompt end (you can check out at local_chat.py),
and pass the arg `--force_think true ` can let the local_chat initiate the response with "\<think>\n"

from-https://github.com/kvcache-ai/ktransformers/issues/129#issue-2842799552

### 3 version `GLIBCXX_3.4.30' not found
The detailed error:
>ImportError: /mnt/data/miniconda3/envs/xxx/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xxx/xxx/ktransformers/./cpuinfer_ext.cpython-312-x86_64-linux-gnu.so)

It may because of your conda env have no this version. Your can first exit your conda env by `conda deactivate` and use `whereis libstdc++.so.6` to find the path. And re enter your conda env and copy the .so by `cp <path of outter libstdc++> <path of your conda env libstdc++>` 
