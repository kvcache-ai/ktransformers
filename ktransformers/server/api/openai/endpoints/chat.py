import json
from time import time
from uuid import uuid4
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field
import re
from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate
from ktransformers.server.schemas.endpoints.chat import RawUsage,Role
from ktransformers.server.backend.base import BackendInterfaceBase
from ktransformers.server.config.config import Config
from ktransformers.server.config.log import logger

from ktransformers.server.schemas.endpoints.chat import ChatCompletionChunk

# 定义我们自己的数据结构替代 OpenAI 的导入
class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[Dict[str, Any]] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None

class Choice(BaseModel):
    index: int
    message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    delta: Optional[Dict[str, Any]] = None
    content_filter_results: Optional[Dict[str, Any]] = None

class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None
    prompt_filter_results: Optional[List[Dict[str, Any]]] = None

# 仅用于非流式响应构建
class ChatCompletionMessageToolCallFunction(BaseModel):
    name: str
    arguments: str

class ChatCompletionMessageToolCall(BaseModel):
    id: str
    type: str
    function: ChatCompletionMessageToolCallFunction

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

router = APIRouter()

@router.get('/models', tags=['openai'])
async def list_models():
    return {"data": [{"id": Config().model_name, "name": Config().model_name}], "object": "list"}


@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request: Request, create: ChatCompletionCreate):
    id = str(uuid4().hex)
    
    # 1. 使用system提示让模型了解如何使用工具
    enhanced_messages = list(create.messages)
    print("-----------------enhanced_messages-----------------------")
    print(enhanced_messages)
    # 如果有工具，且第一条消息是system，在system提示中添加工具使用指导
    if create.tools and len(create.tools) > 0 and enhanced_messages[0].role == Role.system:
        tool_instructions = "你可以使用以下工具：\n\n"
        for tool in create.tools:
            tool_instructions += f"name - {tool.function.name}: {tool.function.description} parameters: {tool.function.parameters}"
        
        tool_instructions += "\n当你需要使用工具时，请以JSON格式输出，格式为：\n"
        tool_instructions += '{"function": {"name": "工具名称", "arguments": {"参数名": "参数值"}}}\n'
        tool_instructions += "不要尝试解释你在做什么，直接输出工具调用即可。"
        print(tool,tool_instructions)
        enhanced_messages[0].content = enhanced_messages[0].content + "\n\n" + tool_instructions
    
    # 处理请求
    interface: BackendInterfaceBase = get_interface()
    input_message = [json.loads(m.model_dump_json()) for m in enhanced_messages]
    
    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key
    
    if create.stream:
        async def inner():
            chunk = ChatCompletionChunk(
                id=id,
                choices=[],
                object='chat.completion.chunk',
                created=int(time()),
                model=Config().model_name,
                system_fingerprint=f"fp_{uuid4().hex[:12]}",
            )
            
            # 收集模型完整输出
            full_content = ""
            async for res in interface.inference(input_message, id, create.temperature, create.top_p):
                if isinstance(res, RawUsage):
                    # 最后返回使用情况
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens=raw_usage.prefill_count,
                        completion_tokens=raw_usage.decode_count,
                        total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                    )
                    yield chunk
                elif isinstance(res, tuple) and len(res) == 2:
                    token, finish_reason = res
                    
                    # 收集内容以检测是否是工具调用
                    if token:
                        full_content += token
                    
                    # 检查内容是否看起来像是工具调用的JSON
                    tool_call_match = re.search(r'{"function":', full_content)
                    
                    if tool_call_match:
                        # 可能是工具调用，尝试提取完整的JSON
                        try:
                            # 查找可能的JSON对象
                            json_start = full_content.find('{')
                            json_end = full_content.rfind('}') + 1
                            
                            if json_start >= 0 and json_end > json_start:
                                json_str = full_content[json_start:json_end]
                                tool_data = json.loads(json_str)
                                
                                if "function" in tool_data and "name" in tool_data["function"]:
                                    # 成功提取到工具调用，处理成流式响应格式
                                    tool_call_id = f"call_{uuid4().hex[:24]}"
                                    
                                    # 首条工具调用消息
                                    first_delta = {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [{
                                            "index": 0,
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_data["function"]["name"],
                                                "arguments": ""
                                            }
                                        }]
                                    }
                                    
                                    # 发送首条工具调用消息
                                    chunk.choices = [{
                                        "index": 0,
                                        "delta": first_delta,
                                        "finish_reason": None
                                    }]
                                    yield chunk
                                    
                                    # 发送参数
                                    if "arguments" in tool_data["function"]:
                                        args = tool_data["function"]["arguments"]
                                        
                                        # 如果参数是字符串，尝试解析为JSON
                                        if isinstance(args, str):
                                            try:
                                                args = json.loads(args)
                                            except:
                                                pass
                                        
                                        args_json = json.dumps(args)
                                        
                                        # 发送参数
                                        chunk.choices = [{
                                            "index": 0,
                                            "delta": {
                                                "tool_calls": [{
                                                    "index": 0,
                                                    "function": {"arguments": args_json}
                                                }]
                                            },
                                            "finish_reason": None
                                        }]
                                        yield chunk
                                    
                                    # 发送完成消息
                                    chunk.choices = [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "tool_calls"
                                    }]
                                    yield chunk
                                    
                                    # 如果请求包含usage信息
                                    if create.stream_options and create.stream_options.get("include_usage"):
                                        chunk.choices = [{
                                            "index": 0,
                                            "delta": {}
                                        }]
                                        chunk.usage = CompletionUsage(
                                            prompt_tokens=176,  # 默认值，实际应基于模型
                                            completion_tokens=len(full_content) // 4,
                                            total_tokens=176 + len(full_content) // 4
                                        )
                                        yield chunk
                                    
                                    # 工具调用已处理，退出
                                    return
                        except Exception as e:
                            logger.error(f"Error parsing tool call: {e}")
                    
                    # 正常文本输出
                    if finish_reason is not None:
                        # 最终消息
                        chunk.choices = [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason
                        }]
                        yield chunk
                    elif token:
                        # 正常文本块
                        chunk.choices = [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                        yield chunk
            
        return chat_stream_response(request, inner())
    else:
        # 非流式响应处理
        full_content = ""
        finish_reason = None
        
        async for res in interface.inference(input_message, id, create.temperature, create.top_p):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens=raw_usage.prefill_count,
                    completion_tokens=raw_usage.decode_count,
                    total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                )
            elif isinstance(res, tuple) and len(res) == 2:
                token, finish_reason = res
                if token:
                    full_content += token
        
        # 检查是否包含工具调用
        tool_calls = []
        try:
            # 查找可能的JSON对象
            json_match = re.search(r'({.*"function".*})', full_content)
            
            if json_match:
                json_str = json_match.group(1)
                tool_data = json.loads(json_str)
                
                if "function" in tool_data and "name" in tool_data["function"]:
                    tool_call_id = f"call_{uuid4().hex[:24]}"
                    
                    # 构建参数
                    args = tool_data["function"].get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            pass
                    
                    args_json = json.dumps(args)
                    
                    # 创建工具调用
                    tool_calls.append({
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_data["function"]["name"],
                            "arguments": args_json
                        }
                    })
                    
                    # 设置完成原因
                    finish_reason = "tool_calls"
        except Exception as e:
            logger.error(f"Error parsing tool call: {e}")
        
        # 构建响应
        response = {
            "id": id,
            "object": "chat.completion",
            "created": int(time()),
            "model": Config().model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None if tool_calls else full_content,
                    "tool_calls": tool_calls if tool_calls else None
                },
                "finish_reason": finish_reason or "stop"
            }],
            "usage": usage.__dict__,
            "system_fingerprint": f"fp_{uuid4().hex[:12]}"
        }
        
        return response