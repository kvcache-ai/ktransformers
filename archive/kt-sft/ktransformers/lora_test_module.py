import os
import platform
import sys

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from torchviz import make_dot
from torch import nn
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)

from ktransformers.operators.linear import KLinearTorch, KTransformersLinear
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.util.inference_state import InferenceState

import hiddenlayer as hl

gguf_loader = GGUFLoader(gguf_path="/home/yj/ktransformers/GGUF-DeepSeek-V2-Lite-Chat")
config = AutoConfig.from_pretrained("/home/yj/ktransformers/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
torch.set_default_dtype(config.torch_dtype)

class TestModelLora(nn.Module):
    def __init__(self):
        super().__init__()

        random_linear_layer = nn.Linear(in_features=3072, out_features=2048, bias=False)
        
        orig_linear = KTransformersLinear(
            key='blk.0.attn_q',
            gguf_loader=gguf_loader,
            config=config,
            orig_module=random_linear_layer,
            generate_op="KLinearTorch"
        )
        self.layer = KTransformersLinearLora(
            orig_module=orig_linear,
            adapter_name="lora_test",
            r=8,
            lora_alpha=16
        )
        self.layer.generate_linear.weight = torch.randn(3072, 2048).to("cuda")
        
    def forward(self, x):
        return self.layer(x)
    
class TestModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = KTransformersLinear(
            key="linear",
            gguf_loader=gguf_loader, 
            config=config, 
            orig_module=nn.Linear(in_features=3072, out_features=2048, bias=False),
            generate_op="KLinearTorch"
        )
        # self.layer.generate_linear.weight = torch.randn(3072, 2048).to("cuda")
        weight = torch.randn(3072, 2048, device="cuda")
        self.layer.load(w=nn.Parameter(weight), mode = InferenceState.GENERATE)
        # self.layer.generate_linear.weight = nn.Parameter(torch.randn(3072, 2048).to("cuda"))
        self.fc1 = nn.Linear(3072, 2048, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 3072, bias=False)
        # self.layer.load(mode=InferenceState.GENERATE)

    def forward(self, x):
        x = self.layer(x)
        # x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TestModelTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = KLinearTorch(
            key="linear",
            gguf_loader=gguf_loader, 
            config=config, 
            orig_module=nn.Linear(in_features=3072, out_features=2048, bias=False)
        )
        # self.layer.weight = nn.Parameter(torch.randn(3072, 2048).to("cuda"))
        # self.layer.weight = torch.randn(3072, 2048).to("cuda")
        weight = torch.randn(3072, 2048, device="cuda")
        self.layer.load(w=nn.Parameter(weight), device="cuda")
        self.fc1 = nn.Linear(3072, 2048, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, 3072, bias=False)
        # self.layer.load(mode=InferenceState.GENERATE) 

    def forward(self, x):
        x = self.layer(x)
        # x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# # KLinearTorch Well DONE for test!
# model = TestModelTorch()
# x = torch.randn(2048, 3072, requires_grad=True)
# out = model(x)
# make_dot(out, params=dict(model.named_parameters())).render("KTLinear_graph", format="svg")


# model = TestModelBase()
# x = torch.randn(2048, 3072, requires_grad=True)
# out = model(x)
# make_dot(out, params=dict(model.named_parameters())).render("base_graph", format="svg")

# MyConvNet_graph=hl.build_graph(model,torch.zeros(size=[2048, 3072]))
# MyConvNet_graph.theme=hl.graph.THEMES['blue'].copy()
# MyConvNet_graph.save(path='./base_graph.png',format='png')

# model = TestModelLora()
# x = torch.randn(2048, 3072, requires_grad=True)
# out = model(x)
# make_dot(out, params=dict(model.named_parameters())).render("lora_graph", format="svg")


from peft import LoraConfig, get_peft_model

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 2048, bias=False)
    
    def forward(self, x):
        return self.linear(x)

model = BaseModel().to("cuda")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["linear"],
    lora_dropout=0.0,
    bias="none",
)

peft_model = get_peft_model(model, lora_config)
print(peft_model)

x = torch.randn(2048, 3072, requires_grad=True).to("cuda")

out = peft_model(x)

dot = make_dot(out, 
             params=dict(peft_model.named_parameters()))

dot.render("origin_lora_graph", format="svg")