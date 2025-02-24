# Tutorial: Inject Operator Step by Step

> Author: Azure-Tang

## TL;DR
This tutorial will guide you through the process of injecting custom operators into a model using the KTransformers framework. We will use the DeepSeekV2-Chat model as an example to demonstrate how to inject custom operators into the model step by step. The tutorial will cover the following topics:
* [How to write injection rules](#how-to-write-injection-rules)
    * [Understanding the structure of the model](#understanding-model-structure)
* [Multi-GPU](#muti-gpu)    
* [How to write a new operator and inject it into the model](#how-to-write-a-new-operator-and-inject-into-the-model)

## How to Write Injection Rules
The basic form of the injection rules for the Inject framework is as follows:
```yaml
- match:
    name: "^model\\.layers\\..*\\.*$"  # Target module name
    class: torch.nn.Linear  # Target module
  replace:
    class: "default"
    kwargs:
      generate_device: "cuda:0"
      # your_op_param_1: 1234
      # your_op_param_2: 5678
  recursive: True
```
* match: This field marks the matching rules, which can appear in two forms, name and class. These two matching rules can appear together or separately; they only match when both criteria are met.
* replace:
	* class: Python class that can be imported to replace the target module. If no replacement is desired, set to default.
	* kwargs: List of parameters needed for module initialization.
	    * generate_device: The device for this module, can be set to “cpu”, “cuda”, “cuda:1”, etc.
* recursive: Whether to recursively inject this module’s submodules, default is True.

For the recursive field: Some modules contain multiple submodules, such as the Self-attention module typically includes q/k/v/o four linear modules. If we replace the self-attention module but do not want the internal linear modules to be covered by other rules, set this rule to False.

## Understanding Model Structure
Using [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) as an example, we can follow the above rules step by step to inject our custom module and run it. KTransformers offers a high degree of flexibility, allowing you to replace/experiment with basic operators. However, it also requires users to clearly understand the structure of the model they are running.

Fortunately, knowing the structure of a model is very simple. Open the file list on the [deepseek-ai/DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/main) homepage, and you can see the following files:
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="../assets/model_structure_guild.png" width=60%>
  </picture>
</p>

From the `.saftensors` file, we can see the name of each layer’s weights, corresponding to the match.name attribute in the injection rules.
From the `modeling_deepseek.py` file, we can see the specific implementation of each module class, with the class name corresponding to the match.class attribute in the injection rules.

The structure of the DeepSeekV2 model from the `.saftensors` and `modeling_deepseek.py` files is as follows:
<p align="center">
  <picture>
    <img alt="Inject-Struction" src="../assets/deepseekv2_structure.png" width=60%>
  </picture>
</p>

Supported operators and their corresponding classes are as follows:

| match     | replace                | backends                | descriptions         |
| --------- | ---------------------- | ----------------------- | -------------------- |
| Linear    | KTransformersLinear    | KLinearMarlin           | Marlin as backend    |
|           |                        | KLinearTorch            | pytorch as backend   |
|           |                        | KLinearCPUInfer         | llamafile as backend |
|           |                        | KLinearFP8         | Triton fp8_gemm kernel. Requires GPU be able to caluculate fp8 data |
| experts   | KTransformersExperts   | KExpertsTorch           | pytorch as backend   |
|           |                        | KExpertsMarlin          | Marlin as backend    |
|           |                        | KExpertsCPU             | llamafile as backend |
| Attention | KDeepseekV2Attention   | KDeepseekV2Attention    | MLA implementation   |
| MoE       | KMistralSparseMoEBlock | KQwen2MoeSparseMoeBlock | MoE for Qwen2        |
|           | KDeepseekV2MoE         | KDeepseekV2MoE          | MoE for DeepseekV2   |
| Model     | KQwen2MoeModel         | KQwen2MoeModel          | Model for Qwen2      |
|           | KDeepseekV2Model       | KDeepseekV2Model        | Model for DeepseekV2 |
| RoPE      | RotaryEmbedding        | RotaryEmbedding         | RoPE module          |
|           | YarnRotaryEmbedding    | YarnRotaryEmbedding     | RoPE module          |

Then we start step-by-step injection of custom modules, our targets are:

* Replace the linear module with custom Marlin linear module.
* Replace the self-attention module with a custom Absorption-based MLA module.
* Replace the experts module with a custom Experts module.
* Replace the MoE module with a custom MoE module.
* Replace the RoPE module with a custom RoPE module.
* Set the running device for each module.

The full implementation of the injection rules can be found in the [here](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat.yaml).

## Matrix Absorption-based MLA Injection

For the injection of the Attention module, we only need to use a regular expression to match the module names used in transformers and replace them with our own MLA module implementation. The YAML injection rule is as follows:
```yaml
- match:
    name: "^model\\.layers\\..*\\.self_attn$"  # Regular expression
  replace:
    class: ktransformers.operators.attention.KDeepseekV2Attention # Optimized MLA implementation
```
As you can see, each rule in the YAML file has two parts: match and replace. The match part specifies the module to be replaced, and the replace part specifies the module to be injected into the model along with the initialization keywords.

## Injection of Routed Experts
For Routed Experts (corresponding to the exps in the diagram), the module we inject is CPUInfer, which is wrapped in the wrapper module KTransformersExperts. KTransformersExperts has multiple implementations, and we need to specify keywords to tell the wrapper module which implementation we want to use and how we plan to use it.

In the source code of the transformer, MoE is implemented using nn.ModuleList. We do not want KTransformers to traverse all submodules in the list and inject them one by one, so in this rule, we set recursive: False to prevent recursive injection into the submodules of this module. The YAML rule is as follows:

```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # Custom MoE kernel with expert parallelism
    kwargs:
      generate_device: "cpu"
      generate_op: "MLPCPUExperts"
      out_device: "cuda"
  recursive: False # Don't recursively inject submodules of this module
```

If we inject Routed Experts as a custom module, we cannot use the interfaces in the original `nn.ModuleList`. Therefore, it is necessary to modify the forward function in the FFN module. The simplest method is to implement a new module with a custom forward function and inject it.
```yaml
- match:
    class: ktransformers.models.modeling_deepseek.DeepseekV2MoE
  replace:
    class: ktransformers.operators.experts.KDeepseekV2MoE     # MLP module with custom forward function
```

## Injection of Linear Layers

For the remaining linear layer modules, we aim to use quantized operators to save storage space while improving performance. Since there is no current research on using MLA and quantization together, we do not want to inject linear into the MLA operator. Therefore, we can modify the regular expression and add a type check in the match part of the rule. Only modules that match both the name and class simultaneously will be injected. We also need to pass some keywords similar to the injection of Routed Experts. The YAML rule is as follows:

```yaml
- match:
    name: "^model\\.layers\\.(?!.*self_attn).*$"  # Regular expression
    class: torch.nn.Linear  # Only match modules matching name and class simultaneously
  replace:
    class: ktransformers.operators.linear.KTransformersLinear  # Optimized kernel on quantized data types
    kwargs:
      generate_device: "cuda"
      generate_op: "QuantizedLinearMarlin"
```
## Injection of Modules with Pre-calculated Buffers

To avoid occupying resources when initializing the injected original model, we use torch’s meta device to initialize the original model. The RoPE module pre-calculates some buffers during initialization, but no calculations are performed when using the meta device. Therefore, we need to compensate for the calculation of the buffer when loading the model. Simply, we inject a custom module into the rotary embedding module, which performs pre-calculation during loading. The YAML rule is as follows:
```yaml
- match:
    class: ktransformers.models.modeling_deepseek.DeepseekV2YarnRotaryEmbedding
  replace:
    class: ktransformers.operators.RoPE.YarnRotaryEmbedding
```

## Specifying Running Devices for Modules

Finally, we set a fallback basic attribute generate_device for all modules:
```yaml
- match:
    name: "^model\\.layers\\..*\\.|^lm_head"
  replace:
    class: "default"
    kwargs:
      generate_device: "cuda"
  
- match:
    name: "^model.embed_tokens"
  replace:
    class: "default"
    kwargs:
        generate_device: "cpu"
```
Through these two rules, we place all previously unmatched layers (and their submodules) and lm_head on cuda, and the embedding on cpu. Note that the properties of a module will be determined by the first rule it matches. For example, if you later set a new replace.kwargs.generate_device in an injected module, the device set earlier will take precedence. If your computer has multiple cards, you can also configure the model to multiple cards.


## Muti-GPU

If you have multiple GPUs, you can set the device for each module to different GPUs. 
DeepseekV2-Chat got 60 layers, if we got 2 GPUs, we can allocate 30 layers to each GPU. Complete multi GPU rule examples [here](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat-multi-gpu.yaml).


<p align="center">
  <picture>
    <img alt="Inject-Struction" src="../assets/multi_gpu.png" width=60%>
  </picture>
</p>

First of all, for multi-GPU, we have to inject an new operator `KDeepseekV2Model`. And set division of the layers to different GPUs. For our case, we have to set the `transfer_map` in the `KDeepseekV2Model` operatoras as follows:

```yaml
- match:
    name: "^model$"
  replace:
    class: "ktransformers.operators.models.KDeepseekV2Model"
    kwargs:
      transfer_map: 
        30: "cuda:1"
```

And we have to set the device for each module in the model. 

For example, for `routed experts`, the yaml for one GPU is:
```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # Custom MoE kernel with expert parallelism
    kwargs:
      generate_device: "cuda:0"
      generate_op: "MLPCUDAExperts"
      out_device: "cuda:0"
  recursive: False # Don't recursively inject submodules of this module
```
But for two GPUs, we need to set the device for each module in the model. 

```yaml
# allcate 0-29 layers‘s out_device to cuda:0
- match:
    name: "^model\\.layers\\.(0|[1-9]|[12][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # custom MoE Kernel with expert paralleism
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:0"
  recursive: False # don't recursively inject submodules of this module

# allocate 30-59 layers‘s out_device to cuda:1
- match:
    name: "^model\\.layers\\.([345][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     # custom MoE Kernel with expert paralleism
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:1"
  recursive: False # don't recursively inject submodules of this module
```
For other modules, we can set the device in the same way.

## How to Write a New Operator and Inject into the Model

In this section, we will explain how to write an operator that can be injected, using the implementation of a new linear as an example.

First, all injectable operators need to inherit from the BaseInjectedModule class, which inherits some attributes required by our injection framework. Its initialization function needs to meet the following basic format:

```python
class LinearTorchInject(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
```
If users have other parameters that need to be passed to this class, they can also be included in the init function and re-passed in the kwargs parameter in the yaml file. For example, if our operator wants to pass a parameter `my_param`, the init function can be written as:
```python
class LinearTorchInject(BaseInjectedModule):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        my_param: bool = True,
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
        self.my_param = my_param
```
Then our injection rule can be written as:
```yaml
- match: 
    name: "^model\\.layers\\..*$"  # Regular expression matches the module name.
    class: torch.nn.Linear  # Type restrictions can be added.
  replace:
    class: ktransformers.operators.linear.LinearTorchInject  # Inject module path
    kwargs: # Extra parameters
      generate_device: "cuda"
      my_param: True
```
For the linear module, it is also necessary to read weights from a gguf file. We provide the `KLinearBase` class to help users read weights from gguf files. Users only need to inherit and implement the load, unload, and forward functions. Therefore, a fully injectable linear class would look like this:
```python
class LinearTorchInject(BaseInjectedModule, KLinearBase):
    def __init__(
        self,
        key: str,
        gguf_loader: GGUFLoader,
        config: PretrainedConfig,
        orig_module: nn.Module = None,
        generate_device: str = "cuda",
        **kwargs,
    ):
        super().__init__(key, gguf_loader, config, orig_module, generate_device, **kwargs)
        KLinearBase.__init__(self)
        self.has_bias = False
        self.dtype = torch.get_default_dtype()
        self.w = None
        self.has_bias = False
    
    def load(self, w: dict | nn.Parameter | tuple | None = None, device: str|None = None):
        if device is None: device = self.device
        if w is None: w = self.load_weight(device=device)

        if isinstance(w, nn.Parameter):
            self.w = w.to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.has_bias = False
        elif isinstance(w, tuple):
            self.w = w[0].to(dtype=self.dtype).view(self.out_features, self.in_features).T
            self.bias = w[1].to(dtype=self.dtype)
            self.has_bias = True
        else:
            raise ValueError("Invalid weight type")
        self.w = self.w.to(device)
        if self.has_bias:
            self.bias = self.bias.to(device)

    def unload(self):
        if self.w is not None:
            self.w = None
        if self.has_bias:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out_device = x.device
        x = x.to(device=self.device, dtype=self.dtype)
        x = x @ self.w
        if self.has_bias:
            x = x + self.bias
        x = x.to(dtype=dtype, device=out_device)
        return x
```
Note that the `self.load_weight` function is provided by the KLinearBase class to help users load weights from a gguf file into the module. The implementation details of KLinearBase can be found on [GITHUB](https://github.com/kvcache-ai/ktransformers/blob/44f57270c9514d79fab224186d90ccf61059331a/ktransformers/operators/linear.py#L31).
