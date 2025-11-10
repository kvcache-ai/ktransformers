
# Muti-GPU

Assume you have read the [Injection Tutorial](./injection_tutorial.md) and have a basic understanding of how to inject a model. In this tutorial, we will show you how to use KTransformers to run a model on multiple GPUs.

If you have multiple GPUs, you can set the device for each module to different GPUs. 
DeepseekV2-Chat got 60 layers, if we got 2 GPUs, we can allocate 30 layers to each GPU. Complete multi GPU rule examples [here](https://github.com/kvcache-ai/ktransformers/blob/main/ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-multi-gpu.yaml).


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

# How to fully utilize multi-GPU's VRAM

When you have multiple GPUs, you can fully utilize the VRAM of each GPU by moving more weights to the GPU.

For example, for DeepSeekV2-Chat, we can move the weights of the experts to the GPU. 

For example, the yaml for two GPUs is:
```yaml
- match:
    name: "^model\\.layers\\.(0|[1-9]|[12][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:0"
  recursive: False
```

But we got extra 60GB VRAM on cuda:0, we can move experts in layer 4~8 to cuda:0. 

```yaml
# Add new rule before old rule.
- match:
    name: "^model\\.layers\\.([4-8])\\.mlp\\.experts$" # inject experts in layer 4~8 as marlin expert
  replace:
    class: ktransformers.operators.experts.KTransformersExperts  
    kwargs:
      generate_device: "cuda:0"
      generate_op:  "KExpertsMarlin"
  recursive: False

- match:
    name: "^model\\.layers\\.(0|[1-9]|[12][0-9])\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts     
    kwargs:
      generate_device: "cpu"
      generate_op:  "KExpertsCPU"
      out_device: "cuda:0"
  recursive: False 
```

Adjust the layer range as you want. Note that:
* The loading speed will be significantly slower for each expert moved to the GPU.
* You have to close the cuda graph if you want to move the experts to the GPU.
* For DeepSeek-R1/V3, each expert moved to the GPU will consume approximately 6GB of VRAM.
* The first matched rule in yaml will be applied. For example, if you have two rules that match the same layer, only the first rule's replacement will be valid.


