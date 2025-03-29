import json
from time import time
from uuid import uuid4
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field

from fastapi import APIRouter
from fastapi.requests import Request
from ktransformers.server.utils.create_interface import get_interface
from ktransformers.server.schemas.assistants.streaming import chat_stream_response
from ktransformers.server.schemas.endpoints.chat import ChatCompletionCreate
from ktransformers.server.schemas.endpoints.chat import RawUsage
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
async def chat_completion(request:Request, create:ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key

    if create.stream:
        # 为流式响应定义辅助类
        class ChoiceDelta(BaseModel):
            content: Optional[str] = None
            role: Optional[str] = None
            tool_calls: Optional[List[Dict[str, Any]]] = None
            
        class ChatCompletionChunkToolCallFunction(BaseModel):
            name: Optional[str] = None
            arguments: Optional[str] = None
            
        class ChatCompletionChunkToolCall(BaseModel):
            id: Optional[str] = None
            type: Optional[str] = None
            index: Optional[int] = None
            function: Optional[ChatCompletionChunkToolCallFunction] = None
        
        async def inner():
            chunk = ChatCompletionChunk(
                id=id,
                choices=[],
                object='chat.completion.chunk',
                created=int(time()),
                model=Config().model_name,
                system_fingerprint=f"fp_{uuid4().hex[:12]}",
            )
            
            # Pass tools to the inference method
            async for res in interface.inference(
                input_message, 
                id, 
                create.temperature, 
                create.top_p, 
                tools=create.tools
            ):
                if isinstance(res, RawUsage):
                    # at the end of inference, interface.inference() will return the usage of inference
                    raw_usage = res
                    chunk.choices = []
                    chunk.usage = CompletionUsage(
                        prompt_tokens = raw_usage.prefill_count,
                        completion_tokens = raw_usage.decode_count,
                        total_tokens = raw_usage.prefill_count + raw_usage.decode_count
                    )
                    yield chunk
                elif isinstance(res, tuple) and len(res) == 2:
                    token, finish_reason = res
                    if finish_reason is not None:
                        # Final chunk with finish_reason
                        choice = Choice(
                            index=0,
                            delta={},
                            finish_reason=finish_reason,
                            logprobs=None,
                        )
                        chunk.choices = [choice]
                        yield chunk
                    else:
                        # Regular content chunk
                        delta = {"content": token, "role": None, "tool_calls": None}
                        choice = Choice(
                            index=0,
                            delta=delta,
                            finish_reason=None,
                            logprobs=None,
                        )
                        chunk.choices = [choice]
                        yield chunk
                elif isinstance(res, dict) and 'tool_call' in res:
                    # Handle tool call in streaming format
                    tool_call_info = res['tool_call']
                    
                    # First chunk contains the initial tool call info
                    if res.get('first_chunk', False):
                        delta = {
                            "content": None, 
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_call_info.get('id', ''),
                                    "type": "function",
                                    "function": {
                                        "name": tool_call_info.get('function', {}).get('name', ''),
                                        "arguments": ""
                                    }
                                }
                            ]
                        }
                        choice = Choice(
                            index=0,
                            delta=delta,
                            finish_reason=None,
                            logprobs=None,
                        )
                        chunk.choices = [choice]
                        yield chunk
                    
                    # Argument chunks
                    if 'argument_chunk' in res:
                        delta = {
                            "tool_calls": [
                                {
                                    "function": {
                                        "arguments": res['argument_chunk']
                                    },
                                    "index": 0
                                }
                            ]
                        }
                        choice = Choice(
                            index=0,
                            delta=delta,
                            finish_reason=None,
                            logprobs=None,
                        )
                        chunk.choices = [choice]
                        yield chunk
                    
                    # Final chunk
                    if res.get('last_chunk', False):
                        choice = Choice(
                            index=0,
                            delta={},
                            finish_reason="tool_calls",
                            logprobs=None,
                        )
                        chunk.choices = [choice]
                        yield chunk

                        # Add usage info in a final chunk
                        if create.stream_options and create.stream_options.get('include_usage', False):
                            prompt_tokens = res.get('prompt_tokens', 176)
                            completion_tokens = res.get('completion_tokens', 20)
                            chunk.usage = CompletionUsage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=prompt_tokens + completion_tokens,
                                prompt_tokens_details={
                                    "cached_tokens": 0,
                                    "audio_tokens": 0
                                },
                                completion_tokens_details={
                                    "audio_tokens": 0,
                                    "reasoning_tokens": 0,
                                    "accepted_prediction_tokens": 0,
                                    "rejected_prediction_tokens": 0
                                }
                            )
                            choice = Choice(index=0, delta={})
                            chunk.choices = [choice]
                            yield chunk

        return chat_stream_response(request, inner())
    else:
        # Non-streaming response
        content = ""
        finish_reason = None
        tool_calls = []
        
        # Pass tools to the inference method
        async for res in interface.inference(
            input_message, 
            id,
            create.temperature, 
            create.top_p, 
            tools=create.tools
        ):
            if isinstance(res, RawUsage):
                raw_usage = res
                usage = CompletionUsage(
                    prompt_tokens=raw_usage.prefill_count,
                    completion_tokens=raw_usage.decode_count,
                    total_tokens=raw_usage.prefill_count + raw_usage.decode_count
                )
            elif isinstance(res, tuple) and len(res) == 2:
                token, finish_reason = res
                content = content + token
            elif isinstance(res, dict) and 'tool_call' in res:
                # Handle tool call in non-streaming format
                tool_call_info = res['tool_call']
                function_info = ChatCompletionMessageToolCallFunction(
                    name=tool_call_info.get('function', {}).get('name', ''),
                    arguments=tool_call_info.get('function', {}).get('arguments', '')
                )
                
                tool_call = ChatCompletionMessageToolCall(
                    id=tool_call_info.get('id', ''),
                    type=tool_call_info.get('type', 'function'),
                    function=function_info
                )
                
                tool_calls.append(tool_call)
                finish_reason = "tool_calls"

        # Create response message
        message = ChatCompletionMessage(
            role="assistant",
            content=None if tool_calls else content,
            tool_calls=tool_calls if tool_calls else None
        )

        # Create choice
        choice = Choice(
            index=0,
            finish_reason=finish_reason,
            message=message.model_dump(),
            logprobs=None
        )

        # Create complete response
        chat_completion = ChatCompletion(
            id=id,
            choices=[choice],
            created=int(time()),
            model=Config().model_name,
            object='chat.completion',
            usage=usage,
            system_fingerprint=f"fp_{uuid4().hex[:12]}"
        )

        return chat_completion