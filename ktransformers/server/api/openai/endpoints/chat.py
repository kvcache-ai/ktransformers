import json
from time import time
from uuid import uuid4
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
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_function import ChatCompletionMessageToolCallFunction
from openai.types.completion_usage import CompletionUsage


router = APIRouter()

@router.get('/models', tags=['openai'])
async def list_models():
    return {"data": [{"id": Config().model_name, "name": Config().model_name}], "object": "list"}


@router.post('/chat/completions', tags=['openai'])
async def chat_completion(request:Request,create:ChatCompletionCreate):
    id = str(uuid4())

    interface: BackendInterfaceBase = get_interface()
    # input_ids = interface.format_and_tokenize_input_ids(id,messages=create.get_tokenizer_messages())

    input_message = [json.loads(m.model_dump_json()) for m in create.messages]

    if Config().api_key != '':
        assert request.headers.get('Authorization', '').split()[-1] == Config().api_key

    if create.stream:
        from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
        from openai.types.chat.chat_completion_chunk_tool_call import ChatCompletionChunkToolCall
        from openai.types.chat.chat_completion_chunk_tool_call_function import ChatCompletionChunkToolCallFunction
        
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
                        prompt_tokens=raw_usage.prefill_count,
                        completion_tokens=raw_usage.decode_count,
                        total_tokens=raw_usage.prefill_count + raw_usage.decode_count
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
                        delta = ChoiceDelta(content=token, role=None, tool_calls=None)
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
                        delta = ChoiceDelta(
                            content=None, 
                            role="assistant",
                            tool_calls=[
                                ChatCompletionChunkToolCall(
                                    index=0,
                                    id=tool_call_info.get('id', ''),
                                    type="function",
                                    function=ChatCompletionChunkToolCallFunction(
                                        name=tool_call_info.get('function', {}).get('name', ''),
                                        arguments=""
                                    )
                                )
                            ]
                        )
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
                        delta = ChoiceDelta(
                            tool_calls=[
                                ChatCompletionChunkToolCall(
                                    function=ChatCompletionChunkToolCallFunction(
                                        arguments=res['argument_chunk']
                                    ),
                                    index=0
                                )
                            ]
                        )
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
            message=message,
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