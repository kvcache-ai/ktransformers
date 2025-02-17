"""
Description  :  
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
"""

import argparse
import uvicorn
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys
import time
from fastapi import Request
from fastapi.responses import StreamingResponse, JSONResponse
import json
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.models.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from ktransformers.models.modeling_llama import LlamaForCausalLM
from ktransformers.models.modeling_mixtral import MixtralForCausalLM
from ktransformers.util.utils import prefill_and_generate
from ktransformers.server.config.config import Config

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "DeepseekV3ForCausalLM": DeepseekV3ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "MixtralForCausalLM": MixtralForCausalLM,
}

ktransformer_rules_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize", "optimize_rules")
default_optimize_rules = {
    "DeepseekV2ForCausalLM": os.path.join(ktransformer_rules_dir, "DeepSeek-V2-Chat.yaml"),
    "DeepseekV3ForCausalLM": os.path.join(ktransformer_rules_dir, "DeepSeek-V3-Chat.yaml"),
    "Qwen2MoeForCausalLM": os.path.join(ktransformer_rules_dir, "Qwen2-57B-A14B-Instruct.yaml"),
    "LlamaForCausalLM": os.path.join(ktransformer_rules_dir, "Internlm2_5-7b-Chat-1m.yaml"),
    "MixtralForCausalLM": os.path.join(ktransformer_rules_dir, "Mixtral.yaml"),
}

# 全局变量，存储初始化后的模型
chat_model = None

class OpenAIChat:
    def __init__(
        self,
        model_path: str,
        optimize_rule_path: str = None,
        gguf_path: str = None,
        cpu_infer: int = Config().cpu_infer,
        use_cuda_graph: bool = True,
        mode: str = "normal",
    ):
        torch.set_grad_enabled(False)
        Config().cpu_infer = cpu_infer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True) if not Config().cpu_infer else None
        if mode == 'long_context':
            assert config.architectures[0] == "LlamaForCausalLM", "Only LlamaForCausalLM supports long_context mode"
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(config.torch_dtype)

        with torch.device("meta"):
            if config.architectures[0] in custom_models:
                if "Qwen2Moe" in config.architectures[0]:
                    config._attn_implementation = "flash_attention_2"
                if "Llama" in config.architectures[0]:
                    config._attn_implementation = "eager"
                if "Mixtral" in config.architectures[0]:
                    config._attn_implementation = "flash_attention_2"
                model = custom_models[config.architectures[0]](config)
            else:
                model = AutoModelForCausalLM.from_config(
                    config, trust_remote_code=True, attn_implementation="flash_attention_2"
                )

        if optimize_rule_path is None:
            if config.architectures[0] in default_optimize_rules:
                optimize_rule_path = default_optimize_rules[config.architectures[0]]

        optimize_and_load_gguf(model, optimize_rule_path, gguf_path, config)
        
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_path)
        except:
            model.generation_config = GenerationConfig(
                max_length=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        model.eval()
        self.model = model
        self.use_cuda_graph = use_cuda_graph
        self.mode = mode
        logger.info("Model loaded successfully!")

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 0.9,
        force_think: bool = False,
    ) -> Dict:
        input_tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        
        if force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\\n", add_special_tokens=False)],
                                        device=input_tensor.device)
            input_tensor = torch.cat([input_tensor, token_thinks], dim=1)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            do_sample=True  # Ensure do_sample is True if using temperature or top_p
        )

        generated = prefill_and_generate(
            self.model,
            self.tokenizer,
            input_tensor.cuda(),
            max_tokens,
            self.use_cuda_graph,
            self.mode,
            force_think
        )

        # Convert token IDs to text
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                }
            }],
            "usage": {
                "prompt_tokens": input_tensor.shape[1],
                "completion_tokens": len(generated),
                "total_tokens": input_tensor.shape[1] + len(generated)
            }
        }

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]  # 确保 messages 是 Pydantic 模型实例的列表
    model: str = "default-model"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 300
    stream: Optional[bool] = False
    force_think: Optional[bool] = True

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-default"
    object: str = "chat.completion"
    created: int = 0
    model: str = "default-model"
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

app = FastAPI(title="KVCache.AI API Server")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    try:
        # 如果 messages 是 Pydantic 模型实例列表，使用 model_dump
        messages = [m.model_dump() for m in request.messages]
        response = chat_model.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            force_think=request.force_think
        )

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response['choices'][0]['message']['content']
                },
                "finish_reason": "stop"
            }],
            "usage": response['usage']
        }
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

def create_app(model_path: str, gguf_path: str, cpu_infer:int, optimize_rule_path: Optional[str] = None):
    global chat_model
    chat_model = OpenAIChat(
        model_path=model_path,
        gguf_path=gguf_path,
        optimize_rule_path=optimize_rule_path,
        cpu_infer=cpu_infer
    )
    return app

def main():
    parser = argparse.ArgumentParser(description="KVCache.AI API Server")
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace模型路径")
    parser.add_argument("--gguf_path", type=str, required=True, help="GGUF模型文件路径")
    parser.add_argument("--optimize_rule_path", type=str, help="优化规则文件路径")
    parser.add_argument("--port", type=int, default=8000, help="服务端口号")
    parser.add_argument("--cpu_infer", type=int, default=70, help="使用cpu数量")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="绑定地址")
    args = parser.parse_args()

    create_app(
        model_path=args.model_path,
        gguf_path=args.gguf_path,
        optimize_rule_path=args.optimize_rule_path,
        cpu_infer=args.cpu_infer
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        loop="uvloop",
        http="httptools",
        timeout_keep_alive=300,
        log_level="info",
        access_log=False
    )

if __name__ == "__main__":
    main()