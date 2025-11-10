## Benchmark

To conduct a quick and convenient check, we have employed a simple Python script available [here](https://github.com/kvcache-ai/ktransformers/tree/main/ktransformers/tests) to assess the precision of our **[ktransformers](https://github.com/kvcache-ai/ktransformers)** project. For this evaluation, we utilized the same dataset, which was shuffled in a consistent manner and limited to the first 1,000 data points, to test our implementation across a variety of CPU kernels, MLA kernels, and quantization formats.

We selected the DeepSeek-V3 model in its bf16, int8, and q4km versions for this test. The MMLU dataset, which can be found [here](https://huggingface.co/datasets/cais/mmlu), was used (we selected all datasets and shuffled them with a fixed random seed).

**!!! However, we skipped the few-shot part and only chose the first 1,000 data points for a quick check.** Please note that this approach may result in results that are not consistent with the technical report of DeepSeek-V3. And the test of R1 and further more tests are on going.

To verify our results, we chose [cloud service platform](https://cloud.siliconflow.cn/models) as baseline. All tests were conducted using the same script and datasets, allowing us to make a preliminary assessment of our project's precision.

We set the argument `temperature=0.6`, and to simplify the test process, we skipped the few-shot part and used the following prompt: `There is a single choice question. Answer the question by replying A, B, C, D. No other answers are accepted. Just the letter. \nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer: '`. For more details, please refer to the [script](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/tests/mmlu_test.py).

Given that we have only tested 1,000 cases, which provides only a preliminary judgment, some fluctuations in the results are reasonable. We selected all datasets and shuffled them with a fixed random seed to ensure consistency.

## Some Details

- The bf16 model of DeepSeek-V3 is available [here](https://huggingface.co/opensourcerelease/DeepSeek-V3-bf16/tree/main) (you may convert it to gguf by llama.cpp). The q4km model can be found [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M).
    
- The optimization YAML file is located [here](https://github.com/kvcache-ai/ktransformers/tree/main/ktransformers/optimize/optimize_rules). For the GEMM Kernel, you can change `KLinearMarlin` to `KLinearTorch`.
    
- To switch the MLA Kernel from Triton to Torch, you can check and modify [this file](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/attention.py), specifically by using the `forward_windows` method.
    
- When attempting to conduct the bf16 test (both CPU Weight and GPU Weight), you may encounter issues stemming from older versions of g++ and as, particularly when using Ubuntu 20 or earlier versions. To facilitate a smoother experience and enable you to reproduce our results, we have provided a development container. This container offers a pre-configured environment tailored for this purpose. However, please note that the container does not have the ktrans package installed. Therefore, you may still need to manually install certain packages to ensure everything runs smoothly.
    
    - You may config the model mount dir in `devcontainer/devcontainer.json`, check the `"mouts":` config.


## The Result Table
Uses DeepSeek-V3 model (Some specific cases are R1)
|                          |                   |            |                   |         |            |                                                        |              |
| ------------------------ | ----------------- | ---------- | ----------------- | ------- | ---------- | ------------------------------------------------------ | ------------ |
| DataSet                  | CPU Weight Format | CPU Kernel | GPU Weight Format | GEMM Kernel   | MLA Kernel | [Siliconflow](https://cloud.siliconflow.cn/models)<br> | Ktrans Point |
| MMLU<br><br>(shuffle 1k) |               |    |               |    |       |                                                    |          |
|          1                | bf16              | cpuinfer   | bf16              | torch   | torch      | 81.6                                                   | 81.9         |
|           2               | q8_0              | cpuinfer   | bf16              | torch   | torch      | 81.6                                                   | 83.1         |
|             3             | q4km              | cpuinfer   | bf16              | torch   | triton     | 81.6                                                   | 81.4         |
|              4            | q4km              | cpuinfer   | q4km->marlin 8    | marlin  | triton     | 81.6                                                   | 81.1         |
|               5           | q4km              | cpuinfer   | q4km->marlin 4    | marlin  | triton     | 81.6                                                   | 81           |
|                6          | q4km              | cpuinfer   | fp8               | fp8gemm  | triton     | 81.6                                                   | 81.5         |
|                7 (DeepSeek-R1)          |  iq1             | cpuinfer   |     fp8           |  fp8gemm | triton     | 78.6                                                   | 83.6         |
| MMLU-pro<br>(shuffle 1k)                 |               |    |                |  |      |                                                    |          |
| 1                 | q4km              | cpuinfer   | fp8               | fp8gemm | triton     | 57.7                                                   | 57.6         |
|  2             | q4km              | cpuinfer   | q4km->marlin 4    | marlin  | triton     | 57.7                                                   | 57.5         |
|  3 (DeepSeek-R1)             | iq1              | cpuinfer   | fp8    | fp8gem  | triton     | 71.9                                                   | tbd         |
| HumanEval                | tbd               | tbd        | tbd               | tbd     | tbd        | tbd                                                    | tbd          |
| GSM8K                    | tbd               | tbd        | tbd               | tbd     | tbd        | tbd                                                    | tbd          |

**The details for each case are listed below**:

By default, The MLA kernel uses triton in linux and torch in windows. But we need to test torch in linux, so we manually modify the [file](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/attention.py#L592). Just get rid of all the if branch and force it to use `self.forward_windows`

- MMLU test
  1. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml) change all the `KLinearMarlin` to `KLinearTorch` (just find all the usage in this file). The source weight comes from [there](https://huggingface.co/opensourcerelease/DeepSeek-V3-bf16) (you need to use llama.cpp to convert it to gguf)
  2. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml). You need to modify the code to separately load cpu's expert weight. We leave this as comment in these places: [1](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L122), [2](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L136), [3](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L137) (note in 3, change the path to your local weight file path). The weight file for q8_0 is [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q8_0)
  3. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml). You need to modify the code to separately load cpu's expert weight. We leave this as comment in these places: [1](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L122), [2](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L136), [3](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/experts.py#L137) (note in 3, change the path to your local weight file path). The weight file for q4km is [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)
  4. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml). You don't need to change the source code as they both use q4km. But note the yaml file [here](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml#L29) and [here](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml#L18), below these lines you need to add `num_bits: 8` (in other words: add this kwargs to all that use `KLinearMarlin`). The weight file for q4km is [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)
  5. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml). No need to change yaml, just use the default. The weight file for q4km is [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)
  6. You should check the [doc](./fp8_kernel.md) to learn how to test this case. This is a mixture tensor case.
  7. You should check the [doc](./fp8_kernel.md) to learn how to test this case. This is a mixture tensor case.
- MMLU-pro test
  1. You should check the [doc](./fp8_kernel.md) to learn how to test this case. This is a mixture tensor case. 
  2. [v3-chat_yaml](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml). No need to change yaml, just use the default. The weight file for q4km is [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M)
  3. You should check the [doc](./fp8_kernel.md) to learn how to test this case. This is a mixture tensor case.