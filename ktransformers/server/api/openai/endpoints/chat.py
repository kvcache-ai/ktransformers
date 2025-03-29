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
from ktransformers.server.schemas.endpoints.chat import RawUsage, Role
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
    
    # 如果有工具，且第一条消息是system，在system提示中添加工具使用指导
    if create.tools and len(create.tools) > 0 and enhanced_messages[0].role == Role.system:
        tool_instructions = "你可以使用以下工具：\n\n"
        for tool in create.tools:
            tool_instructions += f"name - {tool.function.name}: {tool.function.description} parameters: {tool.function.parameters}"
        
        # 修改工具使用指南，鼓励JSON格式输出
        tool_instructions += "\n当你需要使用工具时，请以JSON格式输出，格式为：\n"
        tool_instructions += '{"function": {"name": "工具名称", "arguments": {"参数名": "参数值"}}}\n'
        tool_instructions += "或者多个工具时：\n"
        tool_instructions += '[{"function": {"name": "工具1", "arguments": {"参数名": "参数值"}}}, {"function": {"name": "工具2", "arguments": {"参数名": "参数值"}}}]\n'
        tool_instructions += "不要尝试解释你在做什么，直接输出工具调用即可。"
        
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
            
            # 收集模型完整输出，但专门处理工具调用
            full_content = ""
            buffer = ""  # 用于临时存储当前文本块
            tool_call_mode = False  # 标记是否正在处理工具调用
            tool_calls = []  # 存储所有检测到的工具调用
            brackets_depth = 0  # 跟踪嵌套括号深度
            
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
                    
                    # 只有在非工具调用模式下才添加到full_content
                    if not tool_call_mode:
                        full_content += token
                    
                    # 检测工具调用开始
                    if not tool_call_mode:
                        buffer += token
                        
                        # 检查是否进入工具调用
                        if "{\"function\":" in buffer or "[{\"function\":" in buffer:
                            tool_call_mode = True
                            # 重置buffer，只保留可能的JSON部分
                            start_idx = buffer.find("{\"function\":") if "{\"function\":" in buffer else buffer.find("[{\"function\":")
                            buffer = buffer[start_idx:]
                            
                            # 调整full_content，删除工具调用部分
                            if len(full_content) > len(buffer):
                                full_content = full_content[:-len(buffer)]
                            else:
                                full_content = ""
                                
                            # 发送当前累积的文本内容（如果有的话）
                            if full_content:
                                chunk.choices = [{
                                    "index": 0,
                                    "delta": {"content": full_content},
                                    "finish_reason": None
                                }]
                                yield chunk
                                full_content = ""
                            
                            # 初始化括号计数
                            brackets_depth = buffer.count('{') - buffer.count('}')
                    else:
                        # 在工具调用模式下，继续收集JSON
                        buffer += token
                        # 更新括号深度计数
                        brackets_depth += token.count('{') - token.count('}')
                        
                        # 如果括号平衡，可能完成了一个工具调用
                        if brackets_depth == 0:
                            # 尝试解析收集的JSON
                            try:
                                # 确定是单个工具调用还是多个工具调用
                                cleaned_buffer = buffer.strip()
                                if cleaned_buffer.startswith('[') and cleaned_buffer.endswith(']'):
                                    # 多个工具调用
                                    tool_data_list = json.loads(cleaned_buffer)
                                    for i, tool_data in enumerate(tool_data_list):
                                        if "function" in tool_data and "name" in tool_data["function"]:
                                            tool_call_id = f"call_{uuid4().hex[:24]}"
                                            tool_calls.append({
                                                "id": tool_call_id,
                                                "index": i,
                                                "type": "function",
                                                "function": {
                                                    "name": tool_data["function"]["name"],
                                                    "arguments": json.dumps(tool_data["function"].get("arguments", {}))
                                                }
                                            })
                                else:
                                    # 单个工具调用
                                    tool_data = json.loads(cleaned_buffer)
                                    if "function" in tool_data and "name" in tool_data["function"]:
                                        tool_call_id = f"call_{uuid4().hex[:24]}"
                                        tool_calls.append({
                                            "id": tool_call_id,
                                            "index": 0,
                                            "type": "function",
                                            "function": {
                                                "name": tool_data["function"]["name"],
                                                "arguments": json.dumps(tool_data["function"].get("arguments", {}))
                                            }
                                        })
                                
                                # 处理完工具调用后，重置状态
                                tool_call_mode = False
                                buffer = ""
                                
                                # 发送工具调用事件
                                for idx, tool_call in enumerate(tool_calls):
                                    # 首条工具调用消息
                                    chunk.choices = [{
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [{
                                                "index": idx,
                                                "id": tool_call["id"],
                                                "type": "function",
                                                "function": {
                                                    "name": tool_call["function"]["name"],
                                                    "arguments": ""
                                                }
                                            }]
                                        },
                                        "finish_reason": None
                                    }]
                                    yield chunk
                                    
                                    # 发送参数
                                    chunk.choices = [{
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [{
                                                "index": idx,
                                                "function": {"arguments": tool_call["function"]["arguments"]}
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
                                
                                # 返回后，不继续处理
                                return
                                
                            except json.JSONDecodeError as e:
                                # 如果JSON解析失败，继续收集更多token
                                logger.debug(f"Still collecting JSON: {e}")
                    
                    # 正常文本输出 (仅在非工具调用模式下)
                    if not tool_call_mode and token:
                        if finish_reason is not None:
                            # 最终消息
                            chunk.choices = [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason
                            }]
                            yield chunk
                        else:
                            # 正常文本块
                            chunk.choices = [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None
                            }]
                            yield chunk
            
            # 如果我们已经到了这里而没有返回，说明没有检测到完整的工具调用
            # 发送常规完成消息
            if not tool_call_mode:
                chunk.choices = [{
                    "index": 0, 
                    "delta": {}, 
                    "finish_reason": "stop"
                }]
                yield chunk
        
        return chat_stream_response(request, inner())
    else:
        # 非流式响应处理
        full_content = ""
        finish_reason = None
        tool_calls = []
        buffer = ""
        tool_call_mode = False
        brackets_depth = 0
        
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
                
                if not tool_call_mode:
                    full_content += token
                    buffer += token
                    
                    # 检查是否进入工具调用
                    if "{\"function\":" in buffer or "[{\"function\":" in buffer:
                        tool_call_mode = True
                        # 重置buffer，只保留可能的JSON部分
                        start_idx = buffer.find("{\"function\":") if "{\"function\":" in buffer else buffer.find("[{\"function\":")
                        buffer = buffer[start_idx:]
                        
                        # 调整full_content，删除工具调用部分
                        if len(full_content) > len(buffer):
                            full_content = full_content[:-len(buffer)]
                        else:
                            full_content = ""
                            
                        # 初始化括号计数
                        brackets_depth = buffer.count('{') - buffer.count('}')
                else:
                    # 在工具调用模式下，继续收集JSON
                    buffer += token
                    # 更新括号深度计数
                    brackets_depth += token.count('{') - token.count('}')
                    
                    # 如果括号平衡，可能完成了一个工具调用
                    if brackets_depth == 0:
                        try:
                            # 确定是单个工具调用还是多个工具调用
                            cleaned_buffer = buffer.strip()
                            if cleaned_buffer.startswith('[') and cleaned_buffer.endswith(']'):
                                # 多个工具调用
                                tool_data_list = json.loads(cleaned_buffer)
                                for i, tool_data in enumerate(tool_data_list):
                                    if "function" in tool_data and "name" in tool_data["function"]:
                                        tool_call_id = f"call_{uuid4().hex[:24]}"
                                        tool_calls.append({
                                            "id": tool_call_id,
                                            "index": i,
                                            "type": "function",
                                            "function": {
                                                "name": tool_data["function"]["name"],
                                                "arguments": json.dumps(tool_data["function"].get("arguments", {}))
                                            }
                                        })
                            else:
                                # 单个工具调用
                                tool_data = json.loads(cleaned_buffer)
                                if "function" in tool_data and "name" in tool_data["function"]:
                                    tool_call_id = f"call_{uuid4().hex[:24]}"
                                    tool_calls.append({
                                        "id": tool_call_id,
                                        "index": 0,
                                        "type": "function",
                                        "function": {
                                            "name": tool_data["function"]["name"],
                                            "arguments": json.dumps(tool_data["function"].get("arguments", {}))
                                        }
                                    })
                            
                            # 如果成功解析了工具调用，设置完成原因
                            if tool_calls:
                                finish_reason = "tool_calls"
                            
                            # 重置状态
                            tool_call_mode = False
                            buffer = ""
                            
                        except json.JSONDecodeError:
                            # 如果解析失败，继续收集
                            pass
        
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