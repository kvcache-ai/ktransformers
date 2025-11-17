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
import unittest
from torch.autograd import gradcheck

from ktransformers.operators.linear import KLinearTorch, KTransformersLinear
from ktransformers.sft.peft_utils.lora_layer import KTransformersLinearLora
from ktransformers.util.custom_loader import GGUFLoader
from ktransformers.operators.experts import KExpertsTorch
from ktransformers.util.utils import load_weights

gguf_loader = GGUFLoader(gguf_path="/home/yj/ktransformers/GGUF-DeepSeek-V2-Lite-Chat")
config = AutoConfig.from_pretrained("/home/yj/ktransformers/DeepSeek-V2-Lite-Chat", trust_remote_code=True)
torch.set_default_dtype(config.torch_dtype)

class TestKExpertsTorch(unittest.TestCase):
    def setUp(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.num_experts = 8
        
        self.fixed_input = None
        self.fixed_expert_ids = None
        self.fixed_weights = None
        
    def _create_fixed_data(self, device, batch_size=2):
        """创建固定输入数据"""
        if self.fixed_input is None:
            with torch.random.fork_rng():
                torch.manual_seed(42)
                hidden_size = config.hidden_size
                
                self.fixed_input = torch.randn(batch_size, hidden_size)
                
                self.fixed_expert_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
                
                self.fixed_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        
        return (
            self.fixed_input.clone().to(device).requires_grad_(True),
            self.fixed_expert_ids.clone().to(device),
            self.fixed_weights.clone().to(device)
        )

    def _run_single_device_test(self, device, seed=42):
        """在指定设备上运行前向反向传播并返回梯度"""
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
        
        model = KExpertsTorch(
            key="blk.1",
            gguf_loader=gguf_loader,
            config=config,
            n_routed_experts=self.num_experts,
            device=device
        )
        model.load(device=device)
        
        input_tensor, expert_ids, weights = self._create_fixed_data(device)
        
        model.to(device)
        
        with torch.autocast(device_type=device, enabled=False):
            output = model(input_tensor, expert_ids, weights)
            
        loss = output.sum()
        loss.backward()
        
        gradients = {
            "input": input_tensor.grad.detach().cpu(),
            "loss": loss.detach().cpu(),
            "model": [p.grad.detach().cpu() for p in model.parameters() if p.grad is not None]
        }
        return gradients

    def test_forward_gradient(self):
        cpu_gradients = self._run_single_device_test("cpu")
        
        if torch.cuda.is_available():
            gpu_gradients = self._run_single_device_test("cuda")

            print(f"cpu_gradients:{cpu_gradients}")
            print(f"gpu_gradients:{gpu_gradients}")
            
            input_diff = torch.max(torch.abs(cpu_gradients["input"] - gpu_gradients["input"]))
            print(f"input_diff:{input_diff}")
            
            for i, (cpu_g, gpu_g) in enumerate(zip(cpu_gradients["model"], gpu_gradients["model"])):
                param_diff = torch.max(torch.abs(cpu_g - gpu_g))
                print(f"param_diff:{param_diff}")

            for i, (cpu_g, gpu_g) in enumerate(zip(cpu_gradients["model"], gpu_gradients["model"])):
                diff = (cpu_g - gpu_g.cpu()).abs().max()
                print(f"参数梯度 {i} 最大差异: {diff.item()}")
                self.assertTrue(torch.allclose(cpu_g, gpu_g, atol=1e-4, rtol=1e-3),
                            f"参数梯度 {i} 差异超出阈值，最大差异: {diff.item()}")
                
        else:
            self.skipTest("CUDA不可用，跳过GPU测试")

if __name__ == '__main__':
    unittest.main()