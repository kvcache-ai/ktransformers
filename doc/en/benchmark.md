## Benchmark

To conduct a quick and convenient check, we have employed a simple Python script available [here](https://github.com/kvcache-ai/ktransformers/tree/main/ktransformers/tests) to assess the precision of our **[ktransformers](https://github.com/kvcache-ai/ktransformers)** project. For this evaluation, we utilized the same dataset, which was shuffled in a consistent manner and limited to the first 1,000 data points, to test our implementation across a variety of CPU kernels, MLA kernels, and quantization formats.

We selected the DeepSeek-V3 model in its bf16, int8, and q4km versions for this test. The MMLU dataset, which can be found [here](https://huggingface.co/datasets/cais/mmlu), was used (we selected all datasets and shuffled them with a fixed random seed).

**!!! However, we skipped the few-shot part and only chose the first 1,000 data points for a quick check.** Please note that this approach may result in results that are not consistent with the technical report of DeepSeek-V3. And the test of R1 and further more tests are on going.

To verify our results, we chose [cloud service platform](https://cloud.siliconflow.cn/models) as baseline. All tests were conducted using the same script and datasets, allowing us to make a preliminary assessment of our project's precision.

We set the argument `temperature=0.6`, and to simplify the test process, we skipped the few-shot part and used the following prompt: `There is a single choice question. Answer the question by replying A, B, C, D. No other answers are accepted. Just the letter. \nQuestion: {question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nAnswer: '`. For more details, please refer to the [script](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/tests/mmlu_test.py).

Given that we have only tested 1,000 cases, which provides only a preliminary judgment, some fluctuations in the results are reasonable. We selected all datasets and shuffled them with a fixed random seed to ensure consistency.

## Some Detail

- The bf16 model of DeepSeek-V3 is available [here](https://huggingface.co/opensourcerelease/DeepSeek-V3-bf16/tree/main) (you may convert it to gguf by llama.cpp). The q4km model can be found [here](https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q4_K_M).
    
- The optimization YAML file is located [here](https://github.com/kvcache-ai/ktransformers/tree/main/ktransformers/optimize/optimize_rules). For the Matrix MUL Kernel, you can change `KLinearMarlin` to `KLinearTorch`.
    
- To switch the MLA Kernel from Triton to Torch, you can check and modify [this file](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/operators/attention.py), specifically by using the `forward_windows` method.
    
- When attempting to conduct the bf16 test (both CPU Weight and GPU Weight), you may encounter issues stemming from older versions of g++ and as, particularly when using Ubuntu 20 or earlier versions. To facilitate a smoother experience and enable you to reproduce our results, we have provided a development container. This container offers a pre-configured environment tailored for this purpose. However, please note that the container does not have the ktrans package installed. Therefore, you may still need to manually install certain packages to ensure everything runs smoothly.
    
    - You may config the model mount dir in `devcontainer/devcontainer.json`, check the `"mouts":` config.


## The Result Table

|                          |                   |            |                   |         |            |                                                        |              |
| ------------------------ | ----------------- | ---------- | ----------------- | ------- | ---------- | ------------------------------------------------------ | ------------ |
| DataSet                  | CPU Weight Format | CPU Kernel | GPU Weight Format | GEMM    | MLA Kernel | [Siliconflow](https://cloud.siliconflow.cn/models)<br> | Ktrans Point |
| MMLU<br><br>(shuffle 1k) | bf16              | cpuinfer   | bf16              | torch   | torch      | 81.6                                                   | 81.9         |
|                          | int8              | cpuinfer   | bf16              | torch   | torch      | 81.6                                                   | 83.1         |
|                          | q4km              | cpuinfer   | bf16              | torch   | torch      | 81.6                                                   | 82.8         |
|                          | q4km              | cpuinfer   | bf16              | torch   | triton     | 81.6                                                   | 81.4         |
|                          | q4km              | cpuinfer   | q4km->marlin 8    | marlin  | triton     | 81.6                                                   | 81.1         |
|                          | q4km              | cpuinfer   | q4km->marlin 4    | marlin  | triton     | 81.6                                                   | 81           |
|                          | q4km              | cpuinfer   | fp8               | marlin  | triton     | 81.6                                                   | 81.5         |
| MMLU-pro                 | q4km              | cpuinfer   | fp8               | fp8gemm | triton     | 57.7                                                   | 57.6         |
| MMLU-pro                 | q4km              | cpuinfer   | q4km->marlin 4    | marlin  | triton     | 57.7                                                   | 57.5         |
| HumanEval                | tbd               | tbd        | tbd               | tbd     | tbd        | tbd                                                    | tbd          |
| GSM8K                    | tbd               | tbd        | tbd               | tbd     | tbd        | tbd                                                    | tbd          |
